# file: app/core/decision_policy.py
# description: Applies class- and OCR-based rules to decide whether each detected region is redacted or reviewed.
# author: Maria Victoria Anconetani; Anna Bianca Marzetti Biggi
# date: 15/06/2026

from __future__ import annotations

import re

from .types import Decision, OCRResult, Detection


DATE_RE = re.compile(
    r"\b(?:"
    r"\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|"
    r"\d{4}[/-]\d{1,2}[/-]\d{1,2}|"
    r"\d{1,2}[/-](?:JAN|FEB|MAR|APR|MAY|JUN|JUL|AUG|SEP|SEPT|OCT|NOV|DEC)[/-]\d{2,4}|"
    r"(?:JAN|FEB|MAR|APR|MAY|JUN|JUL|AUG|SEP|SEPT|OCT|NOV|DEC)[/-]\d{1,2}[/-]\d{2,4}"
    r")\b",
    re.IGNORECASE,
)
TIME_RE = re.compile(r"\b(?:[01]?\d|2[0-3])[:.][0-5]\d(?:[:.][0-5]\d)?\b")

ALWAYS_REDACT = {"name", "id", "age", "date"}


def decide_detection(detection: Detection, ocr: OCRResult) -> Decision:
    """
    Chooses whether a detection should be redacted or sent to review.

    Args:
        detection (Detection): Detected ROI with class metadata produced by the YOLO stage.
        ocr (OCRResult): OCR evidence extracted from the detected ROI.

    Returns:
        Decision: Action/reason pair describing whether the ROI is redacted or reviewed.
    """
    class_name = detection.class_name
    text = ocr.text.strip()

    if class_name in ALWAYS_REDACT:
        return Decision(action="redact", reason="class_policy_always_redact")

    if class_name == "time":
        if not text:
            return Decision(action="review", reason="ocr_missing_for_ambiguous_class")
        if TIME_RE.search(text):
            return Decision(action="redact", reason="time_pattern_detected")
        return Decision(action="review", reason="time_text_ambiguous")

    if not text:
        return Decision(action="review", reason="ocr_missing_for_ambiguous_class")

    return Decision(action="review", reason="unsupported_class")
