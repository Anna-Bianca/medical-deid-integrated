from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from .config import AppConfig
from .decision_policy import decide_detection
from .detector_yolo import EXPECTED_CLASS_NAMES, YoloDetector
from .ocr_engine import OCRExtractor
from .redactor import redact_image
from .types import OCRResult, PipelineResult, ProcessedDetection


def _log(message: str) -> None:
    print(f"[Pipeline] {message}", flush=True)


def _preview_text(text: str, *, limit: int = 100) -> str:
    normalized = " | ".join(part.strip() for part in text.splitlines() if part.strip())
    if not normalized:
        return "<empty>"
    return normalized[:limit] + ("..." if len(normalized) > limit else "")


class DeidentificationPipeline:
    def __init__(self, config: AppConfig, model_path: Path | None = None) -> None:
        self.config = config
        self.detector = YoloDetector(config, model_path=model_path)
        self.ocr = OCRExtractor(config)

    def run_on_image(self, image: np.ndarray, filename: str) -> PipelineResult:
        height, width = image.shape[:2]
        _log(
            f"Starting run for '{filename}' with image size {width}x{height}. "
            f"Detector conf={self.config.detector_conf}, fallback={self.config.enable_full_image_fallback}"
        )
        detections = self.detector.detect(image)
        _log(f"YOLO returned {len(detections)} detection(s).")
        processed: list[ProcessedDetection] = []
        fallback_used = False
        fallback_ocr: OCRResult | None = None

        if not detections:
            _log("No ROI detections found by YOLO.")

        for index, detection in enumerate(detections, start=1):
            _log(
                f"Detection {index}: class_id={detection.class_id}, class_name='{detection.class_name}', "
                f"conf={detection.detector_conf:.4f}, roi_box={detection.roi_box}"
            )
            if detection.class_id not in EXPECTED_CLASS_NAMES:
                _log(
                    f"WARNING: class_id={detection.class_id} is outside the expected project classes "
                    f"{sorted(EXPECTED_CLASS_NAMES)}. This usually means models/best.pt is not the trained "
                    "medical detector for this repository."
                )
            ocr_result = self.ocr.extract_from_box(image, detection.expanded_box)
            _log(
                f"OCR {index}: tokens={len(ocr_result.tokens)}, "
                f"text={_preview_text(ocr_result.text)}"
            )
            decision = decide_detection(detection, ocr_result)
            _log(f"Decision {index}: action={decision.action}, reason={decision.reason}")
            if decision.action == "redact":
                ocr_result.characters = self.ocr.extract_characters_from_box(image, detection.expanded_box)
                _log(f"Character OCR {index}: characters={len(ocr_result.characters)}")
            processed.append(ProcessedDetection(detection=detection, ocr=ocr_result, decision=decision))

        if not processed and self.config.enable_full_image_fallback:
            fallback_used = True
            _log("Running full-image fallback OCR because no ROI detections were accepted.")
            fallback_ocr = self.ocr.extract_full_image(image)
            _log(
                f"Fallback OCR finished: tokens={len(fallback_ocr.tokens)}, "
                f"text={_preview_text(fallback_ocr.text)}"
            )
        elif not processed:
            _log("Skipping fallback OCR because it is disabled.")

        redacted, redacted_variants, redaction_mask, redaction_methods_run = redact_image(image, processed, self.config)
        redacted_count = sum(1 for item in processed if item.decision.action == "redact")
        review_count = sum(1 for item in processed if item.decision.action == "review")
        _log(
            f"Run complete for '{filename}': regions={len(processed)}, "
            f"redacted={redacted_count}, review={review_count}, fallback_used={fallback_used}"
        )
        return PipelineResult(
            filename=filename,
            processed_detections=processed,
            redacted_image=redacted,
            fallback_used=fallback_used,
            fallback_ocr=fallback_ocr,
            redacted_variants=redacted_variants,
            redaction_mask=redaction_mask,
            redaction_methods_run=redaction_methods_run,
        )

    def run_on_path(self, image_path: Path) -> PipelineResult:
        image = cv2.imread(str(image_path))
        if image is None:
            raise FileNotFoundError(f"Could not read image: {image_path}")
        return self.run_on_image(image=image, filename=image_path.name)
