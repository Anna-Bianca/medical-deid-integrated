# file: app/core/reporting.py
# description: Serializes pipeline results into report dictionaries for the API, CLI, and debug artifacts.
# author: Maria Victoria Anconetani; Anna Bianca Marzetti Biggi
# date: 15/06/2026

from __future__ import annotations

from .redactor import REDACTION_METHOD
from .types import OCRResult, PipelineResult


def _serialize_ocr(ocr: OCRResult) -> dict:
    """
    Converts an OCR result into a compact report-friendly structure.

    Args:
        ocr (OCRResult): OCR result object with text, confidence summary, and support metadata.

    Returns:
        dict: Dictionary containing the OCR fields surfaced in reports.
    """
    return {
        "text": ocr.text,
        "character_count": len(ocr.characters),
        "conf_summary": ocr.conf_summary,
        "support_passes": ocr.support_passes,
        "debug_available": ocr.debug_available,
    }


def build_report(result: PipelineResult, processing_time_s: float) -> dict:
    """
    Builds the canonical report payload for a pipeline execution.

    Args:
        result (PipelineResult): Full pipeline output for one processed image.
        processing_time_s (float): End-to-end processing time in seconds.

    Returns:
        dict: JSON-serializable report consumed by the API, CLI, UI, and debug outputs.
    """
    redacted = sum(1 for item in result.processed_detections if item.decision.action == "redact")
    review = sum(1 for item in result.processed_detections if item.decision.action == "review")
    mask_pixel_count = int((result.redaction_mask > 0).sum()) if result.redaction_mask is not None else 0
    return {
        "filename": result.filename,
        "processing_time_s": round(processing_time_s, 3),
        "detector_used": "yolo",
        "fallback_used": result.fallback_used,
        "redaction_methods_run": [REDACTION_METHOD] if mask_pixel_count > 0 else [],
        "redaction_mask_pixel_count": mask_pixel_count,
        "summary": {
            "regions_found": len(result.processed_detections),
            "redacted": redacted,
            "review": review,
        },
        "fallback_ocr": _serialize_ocr(result.fallback_ocr) if result.fallback_ocr else None,
        "detections": [
            {
                "class_id": item.detection.class_id,
                "class_name": item.detection.class_name,
                "detector_conf": round(item.detection.detector_conf, 4),
                "roi_box": list(item.detection.roi_box),
                "box": list(item.detection.expanded_box),
                "ocr_text": item.ocr.text,
                "ocr_character_count": len(item.ocr.characters),
                "ocr_token_count": len(item.ocr.tokens),
                "ocr_conf_summary": item.ocr.conf_summary,
                "ocr_support_passes": item.ocr.support_passes,
                "decision": item.decision.action,
                "reason": item.decision.reason,
                "source": item.source,
                "redaction_method": item.redaction_method,
                "mask_source": item.mask_source,
                "mask_pixel_count": item.mask_pixel_count,
            }
            for item in result.processed_detections
        ],
    }
