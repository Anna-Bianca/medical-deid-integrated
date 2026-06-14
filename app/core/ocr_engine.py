from __future__ import annotations

from dataclasses import dataclass
import re
from shutil import which

import cv2
import numpy as np
import pytesseract

from .config import AppConfig
from .types import OCRCharacter, OCRResult, OCRToken


SATURATION_THRESHOLD = 30
BAND_GAP_THRESHOLD = 10
MIN_BAND_HEIGHT = 15
BAND_PADDING = 10
DEFAULT_OVERLAP = 0.6
DEFAULT_MIN_REPETITIONS = 2
DEFAULT_SINGLE_PASS_CONF = 85.0


@dataclass(frozen=True)
class OCRPass:
    name: str
    image: np.ndarray
    x_offset: int = 0
    y_offset: int = 0
    scale: float = 1.0
    tesseract_config: str = ""


@dataclass(frozen=True)
class WordDetection:
    text: str
    conf: float
    x: int
    y: int
    w: int
    h: int
    source_pass: str


class OCRExtractor:
    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self._configure_tesseract()

    def _configure_tesseract(self) -> None:
        if self.config.tesseract_cmd:
            pytesseract.pytesseract.tesseract_cmd = self.config.tesseract_cmd
            return

        detected = which("tesseract")
        candidates = [detected] if detected else []
        candidates.extend(
            [
                r"C:\Program Files\Tesseract-OCR\tesseract.exe",
                r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
            ]
        )
        for candidate in candidates:
            if candidate and PathLike.exists(candidate):
                pytesseract.pytesseract.tesseract_cmd = str(candidate)
                return

    def extract_from_box(self, image: np.ndarray, box: tuple[int, int, int, int]) -> OCRResult:
        x1, y1, x2, y2 = box
        crop = image[y1:y2, x1:x2]
        if crop.size == 0:
            return OCRResult(text="", debug_available=False)

        detections, debug = _collect_detections(crop, use_band_passes=True, base_psm="7")
        mapped = [_offset_detection(det, x1, y1) for det in detections]
        final = _finalize_detections(mapped)
        return _build_ocr_result(final, debug)

    def extract_full_image(self, image: np.ndarray) -> OCRResult:
        detections, debug = _collect_detections(image, use_band_passes=True, base_psm="6")
        final = _finalize_detections(detections)
        return _build_ocr_result(final, debug)

    def extract_characters_from_box(self, image: np.ndarray, box: tuple[int, int, int, int]) -> list[OCRCharacter]:
        x1, y1, x2, y2 = box
        crop = image[y1:y2, x1:x2]
        if crop.size == 0:
            return []
        for ocr_pass in _build_character_passes(crop):
            characters = _extract_characters_from_pass(ocr_pass)
            if not characters:
                continue
            return [_offset_character(character, x1, y1) for character in characters]
        return []


class PathLike:
    @staticmethod
    def exists(path: str) -> bool:
        from pathlib import Path

        return Path(path).exists()


def _offset_detection(det: WordDetection, x_offset: int, y_offset: int) -> WordDetection:
    return WordDetection(
        text=det.text,
        conf=det.conf,
        x=det.x + x_offset,
        y=det.y + y_offset,
        w=det.w,
        h=det.h,
        source_pass=det.source_pass,
    )


def _offset_character(character: OCRCharacter, x_offset: int, y_offset: int) -> OCRCharacter:
    x, y, w, h = character.box
    return OCRCharacter(
        text=character.text,
        box=(x + x_offset, y + y_offset, w, h),
        source_pass=character.source_pass,
    )


def _collect_detections(
    image_bgr: np.ndarray,
    *,
    use_band_passes: bool,
    base_psm: str,
) -> tuple[list[WordDetection], dict]:
    passes = _build_global_passes(image_bgr, base_psm=base_psm)
    if use_band_passes:
        passes.extend(_build_band_passes(image_bgr))

    detections: list[WordDetection] = []
    pass_debug: list[dict] = []
    for ocr_pass in passes:
        raw_lines, pass_detections = _extract_from_pass(ocr_pass)
        detections.extend(pass_detections)
        pass_debug.append(
            {
                "pass_name": ocr_pass.name,
                "raw_preview": raw_lines[:5],
                "detections": len(pass_detections),
            }
        )
    return detections, {"pass_debug": pass_debug}


def _build_global_passes(image_bgr: np.ndarray, *, base_psm: str) -> list[OCRPass]:
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    clahe = _apply_clahe(gray)
    unsharp = _apply_unsharp(clahe)
    adaptive = _adaptive_binary(gray, invert=False)
    adaptive_inv = _adaptive_binary(gray, invert=True)
    config = f"--psm {base_psm} --oem 3"

    return [
        OCRPass("standard_gray", gray, tesseract_config=config),
        OCRPass("inverted_gray", cv2.bitwise_not(gray), tesseract_config=config),
        OCRPass("clahe", clahe, tesseract_config=config),
        OCRPass("clahe_inverted", cv2.bitwise_not(clahe), tesseract_config=config),
        OCRPass("adaptive_binary", adaptive, tesseract_config=config),
        OCRPass("adaptive_binary_inv", adaptive_inv, tesseract_config=config),
        OCRPass("unsharp", unsharp, tesseract_config=config),
        OCRPass("unsharp_inverted", cv2.bitwise_not(unsharp), tesseract_config=config),
        OCRPass(
            "gray_upscale_x2",
            cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC),
            scale=2.0,
            tesseract_config=config,
        ),
        OCRPass(
            "gray_upscale_x3",
            cv2.resize(gray, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC),
            scale=3.0,
            tesseract_config=f"--psm {base_psm} --oem 1",
        ),
        OCRPass(
            "unsharp_upscale_x3",
            cv2.resize(unsharp, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC),
            scale=3.0,
            tesseract_config=config,
        ),
    ]


def _build_character_passes(image_bgr: np.ndarray) -> list[OCRPass]:
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    clahe = _apply_clahe(gray)
    unsharp = _apply_unsharp(clahe)
    return [
        OCRPass(
            "char_gray_x3",
            cv2.resize(gray, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC),
            scale=3.0,
            tesseract_config="--psm 7 --oem 3",
        ),
        OCRPass(
            "char_unsharp_x3",
            cv2.resize(unsharp, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC),
            scale=3.0,
            tesseract_config="--psm 7 --oem 3",
        ),
        OCRPass(
            "char_clahe_x3",
            cv2.resize(clahe, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC),
            scale=3.0,
            tesseract_config="--psm 7 --oem 3",
        ),
    ]


def _build_band_passes(image_bgr: np.ndarray) -> list[OCRPass]:
    passes: list[OCRPass] = []
    for idx, (y1, y2) in enumerate(_detect_colored_bands(image_bgr)):
        if y2 - y1 < MIN_BAND_HEIGHT:
            continue
        y1_pad = max(0, y1 - BAND_PADDING)
        y2_pad = min(image_bgr.shape[0], y2 + BAND_PADDING)
        crop = image_bgr[y1_pad:y2_pad, :]
        crop_gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        crop_unsharp = _apply_unsharp(_apply_clahe(crop_gray))
        passes.extend(
            [
                OCRPass(
                    name=f"band_{idx}_gray_x3",
                    image=cv2.resize(crop_gray, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC),
                    y_offset=y1_pad,
                    scale=3.0,
                    tesseract_config="--psm 7 --oem 3",
                ),
                OCRPass(
                    name=f"band_{idx}_gray_inv_x3",
                    image=cv2.resize(cv2.bitwise_not(crop_gray), None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC),
                    y_offset=y1_pad,
                    scale=3.0,
                    tesseract_config="--psm 7 --oem 3",
                ),
                OCRPass(
                    name=f"band_{idx}_unsharp_x3",
                    image=cv2.resize(crop_unsharp, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC),
                    y_offset=y1_pad,
                    scale=3.0,
                    tesseract_config="--psm 7 --oem 3",
                ),
            ]
        )
    return passes


def _detect_colored_bands(image_bgr: np.ndarray) -> list[tuple[int, int]]:
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    row_sat = hsv[:, :, 1].mean(axis=1)
    colored_rows = np.where(row_sat > SATURATION_THRESHOLD)[0]
    if len(colored_rows) == 0:
        return []

    bands: list[tuple[int, int]] = []
    start = int(colored_rows[0])
    prev = start
    for row in colored_rows[1:]:
        row = int(row)
        if row - prev > BAND_GAP_THRESHOLD:
            bands.append((start, prev))
            start = row
        prev = row
    bands.append((start, prev))
    return bands


def _extract_from_pass(ocr_pass: OCRPass) -> tuple[list[str], list[WordDetection]]:
    raw_lines = pytesseract.image_to_string(ocr_pass.image, config=ocr_pass.tesseract_config).strip().splitlines()
    data = pytesseract.image_to_data(
        ocr_pass.image,
        config=ocr_pass.tesseract_config,
        output_type=pytesseract.Output.DICT,
    )
    detections: list[WordDetection] = []
    for idx in range(len(data.get("level", []))):
        raw_text = data["text"][idx].strip()
        if not raw_text or not _is_valid_token(raw_text):
            continue
        try:
            conf = float(data["conf"][idx])
        except (TypeError, ValueError):
            conf = -1.0
        x = int(round(data["left"][idx] / ocr_pass.scale)) + ocr_pass.x_offset
        y = int(round(data["top"][idx] / ocr_pass.scale)) + ocr_pass.y_offset
        w = int(round(data["width"][idx] / ocr_pass.scale))
        h = int(round(data["height"][idx] / ocr_pass.scale))
        if w < 5 or h < 5:
            continue
        detections.append(
            WordDetection(
                text=raw_text,
                conf=conf,
                x=x,
                y=y,
                w=w,
                h=h,
                source_pass=ocr_pass.name,
            )
        )
    return raw_lines, detections


def _extract_characters_from_pass(ocr_pass: OCRPass) -> list[OCRCharacter]:
    raw = pytesseract.image_to_boxes(ocr_pass.image, config=ocr_pass.tesseract_config)
    if not raw.strip():
        return []
    pass_height = ocr_pass.image.shape[0]
    characters: list[OCRCharacter] = []
    for line in raw.splitlines():
        parts = line.split()
        if len(parts) < 5:
            continue
        char_text = parts[0]
        if not _is_valid_character(char_text):
            continue
        try:
            x1, y1, x2, y2 = map(int, parts[1:5])
        except ValueError:
            continue
        x = int(round(x1 / ocr_pass.scale))
        y = int(round((pass_height - y2) / ocr_pass.scale))
        w = int(round((x2 - x1) / ocr_pass.scale))
        h = int(round((y2 - y1) / ocr_pass.scale))
        if w < 2 or h < 2:
            continue
        characters.append(
            OCRCharacter(
                text=char_text,
                box=(x, y, w, h),
                source_pass=ocr_pass.name,
            )
        )
    return _dedupe_characters(characters, overlap_threshold=DEFAULT_OVERLAP)


def _finalize_detections(detections: list[WordDetection]) -> list[WordDetection]:
    confident = [det for det in detections if det.conf >= 50.0]
    _, overlap_candidates = _apply_overlap_repetition_filter(
        confident,
        min_repetitions=DEFAULT_MIN_REPETITIONS,
        overlap_threshold=DEFAULT_OVERLAP,
        single_pass_conf_threshold=DEFAULT_SINGLE_PASS_CONF,
    )
    final = _dedupe_word_detections(overlap_candidates, overlap_threshold=DEFAULT_OVERLAP)
    final.sort(key=lambda det: (det.y, det.x))
    return final


def _build_ocr_result(detections: list[WordDetection], debug: dict) -> OCRResult:
    tokens: list[OCRToken] = []
    support_passes = sorted({det.source_pass for det in detections})
    words: list[str] = []
    for det in detections:
        clean = _strip_punctuation(det.text)
        if not clean:
            continue
        tokens.append(
            OCRToken(
                text=clean,
                conf=round(det.conf, 2),
                box=(det.x, det.y, det.w, det.h),
                source_pass=det.source_pass,
            )
        )
        words.append(clean)
    words = _dedupe_preserve_order(words)
    conf_values = [token.conf for token in tokens]
    summary = {
        "count": len(tokens),
        "mean_conf": round(float(np.mean(conf_values)), 2) if conf_values else None,
        "max_conf": round(float(np.max(conf_values)), 2) if conf_values else None,
    }
    return OCRResult(
        text="\n".join(words).strip(),
        tokens=tokens,
        conf_summary=summary,
        support_passes=support_passes,
        debug=debug,
        debug_available=bool(debug.get("pass_debug")),
    )


def _apply_clahe(gray: np.ndarray) -> np.ndarray:
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(gray)


def _apply_unsharp(gray: np.ndarray) -> np.ndarray:
    blurred = cv2.GaussianBlur(gray, (0, 0), 1.0)
    return cv2.addWeighted(gray, 1.7, blurred, -0.7, 0)


def _adaptive_binary(gray: np.ndarray, *, invert: bool) -> np.ndarray:
    mode = cv2.THRESH_BINARY_INV if invert else cv2.THRESH_BINARY
    return cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, mode, 31, 5)


def _is_valid_token(text: str) -> bool:
    return bool(re.search(r"[A-Za-z0-9]", text))


def _is_valid_character(text: str) -> bool:
    return bool(text.strip())


def _strip_punctuation(text: str) -> str:
    return re.sub(r"^[^A-Za-z0-9]+|[^A-Za-z0-9]+$", "", text)


def _normalize_token(text: str) -> str:
    normalized = re.sub(r"[^A-Za-z0-9]+", "", text).lower()
    return normalized if normalized else text.strip().lower()


def _dedupe_preserve_order(items: list[str]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for item in items:
        if item not in seen:
            seen.add(item)
            ordered.append(item)
    return ordered


def _bbox_iou(a: WordDetection, b: WordDetection) -> float:
    ax1, ay1, ax2, ay2 = a.x, a.y, a.x + a.w, a.y + a.h
    bx1, by1, bx2, by2 = b.x, b.y, b.x + b.w, b.y + b.h
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter_w = max(0, ix2 - ix1)
    inter_h = max(0, iy2 - iy1)
    inter_area = inter_w * inter_h
    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    union = area_a + area_b - inter_area
    return inter_area / union if union > 0 else 0.0


def _apply_overlap_repetition_filter(
    detections: list[WordDetection],
    *,
    min_repetitions: int,
    overlap_threshold: float,
    single_pass_conf_threshold: float,
) -> tuple[list[dict], list[WordDetection]]:
    clusters: list[dict] = []
    for det in detections:
        token = _normalize_token(det.text)
        assigned = False
        for cluster in clusters:
            if token != cluster["token"]:
                continue
            if any(_bbox_iou(det, member) >= overlap_threshold for member in cluster["members"]):
                cluster["members"].append(det)
                assigned = True
                break
        if not assigned:
            clusters.append({"token": token, "members": [det]})

    valid_clusters: list[dict] = []
    candidates: list[WordDetection] = []
    for cluster in clusters:
        members = cluster["members"]
        pass_support = {member.source_pass for member in members}
        multi_pass_ok = len(pass_support) >= min_repetitions
        high_conf_ok = single_pass_conf_threshold > 0 and any(
            member.conf >= single_pass_conf_threshold for member in members
        )
        if multi_pass_ok or high_conf_ok:
            valid_clusters.append(cluster)
            candidates.extend(members)
    return valid_clusters, candidates


def _dedupe_word_detections(
    detections: list[WordDetection],
    *,
    overlap_threshold: float,
) -> list[WordDetection]:
    kept: list[WordDetection] = []
    for det in sorted(detections, key=lambda item: -item.conf):
        if not any(_bbox_iou(det, existing) >= overlap_threshold for existing in kept):
            kept.append(det)
    return kept


def _character_iou(a: OCRCharacter, b: OCRCharacter) -> float:
    ax, ay, aw, ah = a.box
    bx, by, bw, bh = b.box
    ax2, ay2 = ax + aw, ay + ah
    bx2, by2 = bx + bw, by + bh
    ix1, iy1 = max(ax, bx), max(ay, by)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter_w = max(0, ix2 - ix1)
    inter_h = max(0, iy2 - iy1)
    inter_area = inter_w * inter_h
    area_a = max(0, aw) * max(0, ah)
    area_b = max(0, bw) * max(0, bh)
    union = area_a + area_b - inter_area
    return inter_area / union if union > 0 else 0.0


def _dedupe_characters(characters: list[OCRCharacter], *, overlap_threshold: float) -> list[OCRCharacter]:
    kept: list[OCRCharacter] = []
    for character in characters:
        if not any(
            character.text == existing.text and _character_iou(character, existing) >= overlap_threshold
            for existing in kept
        ):
            kept.append(character)
    kept.sort(key=lambda item: (item.box[1], item.box[0]))
    return kept
