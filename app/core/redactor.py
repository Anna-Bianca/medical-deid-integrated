# file: app/core/redactor.py
# description: Builds fine-grained masks for redactable text regions and applies biharmonic inpainting to the image.
# author: Maria Victoria Anconetani; Anna Bianca Marzetti Biggi
# date: 15/06/2026

from __future__ import annotations

import cv2
import numpy as np

from .config import AppConfig
from .types import ProcessedDetection


REDACTION_METHOD = "biharmonic"
MIN_COMPONENT_AREA = 2
CHARACTER_MIN_SEGMENTED_COVERAGE = 0.35
TOKEN_MIN_SEGMENTED_COVERAGE = 0.12
MAX_SEGMENTED_COVERAGE = 0.85
CHARACTER_SEGMENT_DILATION = 3
TOKEN_SEGMENT_DILATION = 1
CHARACTER_BOX_FALLBACK_PADDING = 1
TOKEN_BOX_FALLBACK_PADDING = 1
CHARACTER_BOX_MASK_PADDING = 1


def _normalize_box(
    box: tuple[int, int, int, int],
    image_shape: tuple[int, int],
    padding: int,
) -> tuple[int, int, int, int] | None:
    """
    Converts an `(x, y, w, h)` box into clamped corner coordinates with padding.

    Args:
        box (tuple[int, int, int, int]): Bounding box expressed as `(x, y, width, height)`.
        image_shape (tuple[int, int]): Image shape as `(height, width)`.
        padding (int): Extra pixels to expand around the box on each side.

    Returns:
        tuple[int, int, int, int] | None: Clamped `(x1, y1, x2, y2)` corners, or `None` if invalid.
    """
    height, width = image_shape
    x, y, w, h = box
    x1 = max(0, x - padding)
    y1 = max(0, y - padding)
    x2 = min(width, x + w + padding)
    y2 = min(height, y + h + padding)
    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2


def _roi_box(
    box: tuple[int, int, int, int],
    image_shape: tuple[int, int],
    padding: int,
) -> tuple[int, int, int, int] | None:
    """
    Expands a corner-based ROI while keeping it inside image bounds.

    Args:
        box (tuple[int, int, int, int]): ROI expressed as `(x1, y1, x2, y2)`.
        image_shape (tuple[int, int]): Image shape as `(height, width)`.
        padding (int): Extra pixels to expand around the ROI on each side.

    Returns:
        tuple[int, int, int, int] | None: Clamped expanded ROI, or `None` if it collapses.
    """
    height, width = image_shape
    x1, y1, x2, y2 = box
    x1 = max(0, x1 - padding)
    y1 = max(0, y1 - padding)
    x2 = min(width, x2 + padding)
    y2 = min(height, y2 + padding)
    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2


def _apply_biharmonic_inpaint(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """
    Applies scikit-image biharmonic inpainting over the masked pixels.

    Args:
        image (np.ndarray): Source BGR image as an unsigned 8-bit array.
        mask (np.ndarray): Boolean-like mask where non-zero pixels must be inpainted.

    Returns:
        np.ndarray: Inpainted image with the same shape as the input.

    Raises:
        ImportError: If `scikit-image` is not installed in the active environment.
    """
    try:
        from skimage.restoration import inpaint
    except ImportError as exc:
        raise ImportError(
            "scikit-image is required for biharmonic inpainting. "
            "Install dependencies from requirements.txt."
        ) from exc

    image_float = image.astype(np.float32) / 255.0
    result = inpaint.inpaint_biharmonic(image_float, mask.astype(bool), channel_axis=-1)
    return np.clip(result * 255.0, 0, 255).astype(np.uint8)


def _segment_text_pixels(image: np.ndarray, region: tuple[int, int, int, int]) -> np.ndarray:
    """
    Segments likely text pixels inside a cropped ROI.

    Args:
        image (np.ndarray): Source BGR image from which the ROI is extracted.
        region (tuple[int, int, int, int]): ROI expressed as `(x1, y1, x2, y2)`.

    Returns:
        np.ndarray: Binary mask of segmented text pixels in ROI-local coordinates.
    """
    x1, y1, x2, y2 = region
    crop = image[y1:y2, x1:x2]
    if crop.size == 0:
        return np.zeros((0, 0), dtype=np.uint8)

    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
    blur = cv2.GaussianBlur(clahe, (3, 3), 0)
    adaptive_dark = cv2.adaptiveThreshold(
        blur,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        21,
        7,
    )
    adaptive_light = cv2.adaptiveThreshold(
        blur,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        21,
        7,
    )
    adaptive_light = cv2.bitwise_not(adaptive_light)
    _, otsu_dark = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    _, otsu_light = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    otsu_light = cv2.bitwise_not(otsu_light)
    contrast = cv2.absdiff(clahe, cv2.GaussianBlur(clahe, (0, 0), 1.0))
    _, contrast_mask = cv2.threshold(contrast, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    candidates = [adaptive_dark, adaptive_light, otsu_dark, otsu_light]
    combined = np.zeros_like(gray, dtype=np.uint8)
    kernel = np.ones((2, 2), dtype=np.uint8)

    for candidate in candidates:
        merged = cv2.bitwise_and(candidate, contrast_mask)
        cleaned = cv2.morphologyEx(merged, cv2.MORPH_OPEN, kernel)
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel)
        component_mask, component_count = _filter_connected_components(cleaned)
        pixel_count = int(component_mask.sum() // 255)
        if pixel_count == 0:
            continue
        coverage = pixel_count / max(component_mask.shape[0] * component_mask.shape[1], 1)
        if component_count == 0 or coverage >= MAX_SEGMENTED_COVERAGE:
            continue
        combined = np.maximum(combined, component_mask)

    if not np.any(combined):
        return combined

    combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel)
    refined, _ = _filter_connected_components(combined)
    return refined


def _filter_connected_components(mask: np.ndarray) -> tuple[np.ndarray, int]:
    """
    Removes implausibly small or oversized connected components from a binary mask.

    Args:
        mask (np.ndarray): Binary mask to filter.

    Returns:
        tuple[np.ndarray, int]: Filtered mask and number of kept components.
    """
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    kept = np.zeros_like(mask, dtype=np.uint8)
    kept_components = 0
    total_area = mask.shape[0] * mask.shape[1]
    for label_idx in range(1, num_labels):
        area = int(stats[label_idx, cv2.CC_STAT_AREA])
        if area < MIN_COMPONENT_AREA or area > int(total_area * 0.9):
            continue
        kept[labels == label_idx] = 255
        kept_components += 1
    return kept, kept_components


def _fill_region(mask: np.ndarray, region: tuple[int, int, int, int]) -> None:
    """
    Fills a rectangular region of a mask with foreground pixels.

    Args:
        mask (np.ndarray): Target mask to update in-place.
        region (tuple[int, int, int, int]): Region expressed as `(x1, y1, x2, y2)`.

    Returns:
        None: The input mask is modified in-place.
    """
    x1, y1, x2, y2 = region
    mask[y1:y2, x1:x2] = 255


def _box_mask_from_boxes(
    boxes: list[tuple[int, int, int, int]],
    image_shape: tuple[int, int],
    padding: int,
) -> np.ndarray:
    """
    Builds a full-image mask by filling all normalized character boxes.

    Args:
        boxes (list[tuple[int, int, int, int]]): Character boxes expressed as `(x, y, w, h)`.
        image_shape (tuple[int, int]): Image shape as `(height, width)`.
        padding (int): Extra pixels to add around each box before filling.

    Returns:
        np.ndarray: Binary mask with all valid boxes rasterized.
    """
    mask = np.zeros(image_shape, dtype=np.uint8)
    for box in boxes:
        region = _normalize_box(box, image_shape, padding)
        if region is None:
            continue
        _fill_region(mask, region)
    return mask


def _expand_region(
    region: tuple[int, int, int, int],
    image_shape: tuple[int, int],
    extra_padding: int,
) -> tuple[int, int, int, int]:
    """
    Expands a corner-based region by a fixed number of pixels.

    Args:
        region (tuple[int, int, int, int]): Region expressed as `(x1, y1, x2, y2)`.
        image_shape (tuple[int, int]): Image shape as `(height, width)`.
        extra_padding (int): Padding to add on each side.

    Returns:
        tuple[int, int, int, int]: Expanded region clamped to image bounds.
    """
    if extra_padding <= 0:
        return region
    x1, y1, x2, y2 = region
    height, width = image_shape
    return (
        max(0, x1 - extra_padding),
        max(0, y1 - extra_padding),
        min(width, x2 + extra_padding),
        min(height, y2 + extra_padding),
    )


def _dilate_mask(mask: np.ndarray, pixels: int) -> np.ndarray:
    """
    Dilates a binary mask with an elliptical kernel.

    Args:
        mask (np.ndarray): Binary mask to dilate.
        pixels (int): Radius-like dilation size in pixels.

    Returns:
        np.ndarray: Dilated mask, or the original mask if dilation is unnecessary.
    """
    if pixels <= 0 or not np.any(mask):
        return mask
    kernel_size = (pixels * 2) + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    return cv2.dilate(mask, kernel, iterations=1)


def _mask_from_boxes(
    image: np.ndarray,
    boxes: list[tuple[int, int, int, int]],
    image_shape: tuple[int, int],
    padding: int,
    min_coverage: float,
    segment_dilation: int,
    fallback_padding: int,
) -> tuple[np.ndarray, bool]:
    """
    Builds a text mask from token boxes using segmentation with box-based fallbacks.

    Args:
        image (np.ndarray): Source BGR image used for ROI segmentation.
        boxes (list[tuple[int, int, int, int]]): Token boxes expressed as `(x, y, w, h)`.
        image_shape (tuple[int, int]): Image shape as `(height, width)`.
        padding (int): Padding applied before segmenting each token box.
        min_coverage (float): Minimum segmented coverage required to trust the segmented mask.
        segment_dilation (int): Dilation amount applied after successful segmentation.
        fallback_padding (int): Padding used when falling back to full-box masking.

    Returns:
        tuple[np.ndarray, bool]: Full-image mask and whether any token used the box fallback path.
    """
    mask = np.zeros(image_shape, dtype=np.uint8)
    used_box_fallback = False
    for box in boxes:
        region = _normalize_box(box, image_shape, padding)
        if region is None:
            continue
        x1, y1, x2, y2 = region
        region_mask = _segment_text_pixels(image, region)
        if region_mask.size == 0 or not np.any(region_mask):
            _fill_region(mask, _expand_region(region, image_shape, fallback_padding))
            used_box_fallback = True
            continue
        region_mask = _dilate_mask(region_mask, segment_dilation)
        region_area = max((x2 - x1) * (y2 - y1), 1)
        region_coverage = float(np.count_nonzero(region_mask)) / float(region_area)
        if region_coverage < min_coverage:
            _fill_region(mask, _expand_region(region, image_shape, fallback_padding))
            used_box_fallback = True
            continue
        mask_slice = mask[y1:y2, x1:x2]
        mask[y1:y2, x1:x2] = np.maximum(mask_slice, region_mask)
    return mask, used_box_fallback


def _mark_redaction_source(
    processed: ProcessedDetection,
    image: np.ndarray,
    image_shape: tuple[int, int],
    config: AppConfig,
) -> np.ndarray:
    """
    Chooses the best available masking strategy for one redactable detection.

    Args:
        processed (ProcessedDetection): Detection plus OCR data to annotate with mask metadata.
        image (np.ndarray): Source BGR image used for token segmentation when needed.
        image_shape (tuple[int, int]): Image shape as `(height, width)`.
        config (AppConfig): Runtime redaction settings controlling granularity and padding.

    Returns:
        np.ndarray: Full-image binary mask for the selected source, or an empty mask if unavailable.
    """
    char_boxes = [character.box for character in processed.ocr.characters]
    token_boxes = [token.box for token in processed.ocr.tokens]

    if config.redaction_granularity == "character" and char_boxes:
        character_mask = _box_mask_from_boxes(
            char_boxes,
            image_shape,
            config.character_mask_padding + CHARACTER_BOX_MASK_PADDING,
        )
        if np.any(character_mask):
            processed.mask_source = "character_box_mask"
            processed.mask_pixel_count = int(np.count_nonzero(character_mask))
            return character_mask

    if token_boxes:
        token_mask, used_token_box_fallback = _mask_from_boxes(
            image,
            token_boxes,
            image_shape,
            config.token_mask_padding,
            TOKEN_MIN_SEGMENTED_COVERAGE,
            TOKEN_SEGMENT_DILATION,
            TOKEN_BOX_FALLBACK_PADDING,
        )
        if np.any(token_mask):
            processed.mask_source = "token_box_fallback" if used_token_box_fallback else "token_segmented"
            processed.mask_pixel_count = int(np.count_nonzero(token_mask))
            return token_mask

    roi = _roi_box(processed.detection.expanded_box, image_shape, config.roi_mask_padding)
    if roi is not None:
        x1, y1, x2, y2 = roi
        roi_mask = np.zeros(image_shape, dtype=np.uint8)
        roi_mask[y1:y2, x1:x2] = 255
        processed.mask_source = "roi_fallback"
        processed.mask_pixel_count = int(np.count_nonzero(roi_mask))
        return roi_mask

    processed.mask_source = None
    processed.mask_pixel_count = 0
    return np.zeros(image_shape, dtype=np.uint8)


def redact_image(
    image: np.ndarray,
    detections: list[ProcessedDetection],
    config: AppConfig,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Builds a combined redaction mask and applies biharmonic inpainting to the image.

    Args:
        image (np.ndarray): Source BGR image to redact.
        detections (list[ProcessedDetection]): Processed detections whose `decision` may require redaction.
        config (AppConfig): Runtime redaction settings controlling mask construction.

    Returns:
        tuple[np.ndarray, np.ndarray]: Redacted image and combined binary redaction mask.
    """
    image_shape = image.shape[:2]
    redaction_mask = np.zeros(image_shape, dtype=np.uint8)

    for processed in detections:
        if processed.decision.action != "redact":
            continue
        mask = _mark_redaction_source(processed, image, image_shape, config)
        if np.any(mask):
            redaction_mask = np.maximum(redaction_mask, mask)

    if np.any(redaction_mask):
        bool_mask = redaction_mask.astype(bool)
        redacted_image = _apply_biharmonic_inpaint(image.copy(), bool_mask)
    else:
        redacted_image = image.copy()

    for processed in detections:
        if processed.decision.action == "redact":
            processed.redaction_method = f"inpaint_{REDACTION_METHOD}"

    return redacted_image, redaction_mask
