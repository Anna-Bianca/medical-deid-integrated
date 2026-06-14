from __future__ import annotations

import re

from .types import Decision, OCRResult, Detection


DATE_RE = re.compile(r"\b(?:\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|\d{4}[/-]\d{1,2}[/-]\d{1,2})\b")
TIME_RE = re.compile(r"\b(?:[01]?\d|2[0-3])[:.][0-5]\d(?:[:.][0-5]\d)?\b")

ALWAYS_REDACT = {"name", "id", "age"}


def decide_detection(detection: Detection, ocr: OCRResult) -> Decision:
    class_name = detection.class_name
    text = ocr.text.strip()

    if class_name in ALWAYS_REDACT:
        return Decision(action="redact", reason="class_policy_always_redact")

    if not text:
        return Decision(action="review", reason="ocr_missing_for_ambiguous_class")

    if class_name == "date":
        if DATE_RE.search(text):
            return Decision(action="redact", reason="date_pattern_detected")
        return Decision(action="review", reason="date_text_ambiguous")

    if class_name == "time":
        if TIME_RE.search(text):
            return Decision(action="redact", reason="time_pattern_detected")
        return Decision(action="review", reason="time_text_ambiguous")

    return Decision(action="review", reason="unsupported_class")

