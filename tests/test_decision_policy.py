# file: tests/test_decision_policy.py
# description: Verifies class- and OCR-based decision policy outcomes for date and time detections.
# author: Maria Victoria Anconetani; Anna Bianca Marzetti Biggi
# date: 15/06/2026

from __future__ import annotations

import unittest

from app.core.decision_policy import DATE_RE, decide_detection
from app.core.types import Detection, OCRResult


def _detection(class_name: str) -> Detection:
    """
    Builds a minimal detection object for policy unit tests.

    Args:
        class_name (str): Detection class name to assign to the synthetic ROI.

    Returns:
        Detection: Detection configured with stable geometry and confidence for tests.
    """
    class_ids = {"name": 0, "id": 1, "age": 2, "date": 3, "time": 4}
    return Detection(
        class_id=class_ids.get(class_name, 99),
        class_name=class_name,
        detector_conf=0.9,
        roi_box=(10, 10, 20, 20),
        expanded_box=(8, 8, 22, 22),
    )


class DecisionPolicyTests(unittest.TestCase):
    """Covers the expected policy behavior for always-redact and OCR-gated classes."""

    def test_date_with_empty_ocr_is_always_redacted(self) -> None:
        """Ensures `date` is redacted even when OCR returns no text."""
        decision = decide_detection(_detection("date"), OCRResult(text=""))

        self.assertEqual(decision.action, "redact")
        self.assertEqual(decision.reason, "class_policy_always_redact")

    def test_date_with_numeric_text_is_always_redacted(self) -> None:
        """Ensures numeric date formats are still redacted under the always-redact policy."""
        decision = decide_detection(_detection("date"), OCRResult(text="2017-02-08"))

        self.assertEqual(decision.action, "redact")
        self.assertEqual(decision.reason, "class_policy_always_redact")

    def test_date_with_textual_month_is_always_redacted(self) -> None:
        """Ensures textual-month date formats are recognized and still redacted."""
        text = "08-FEB-2017"

        self.assertIsNotNone(DATE_RE.search(text))
        decision = decide_detection(_detection("date"), OCRResult(text=text))

        self.assertEqual(decision.action, "redact")
        self.assertEqual(decision.reason, "class_policy_always_redact")

    def test_time_with_valid_pattern_is_redacted(self) -> None:
        """Ensures valid time strings still trigger automatic redaction."""
        decision = decide_detection(_detection("time"), OCRResult(text="14:35"))

        self.assertEqual(decision.action, "redact")
        self.assertEqual(decision.reason, "time_pattern_detected")

    def test_time_with_ambiguous_text_is_reviewed(self) -> None:
        """Ensures ambiguous time-like text remains routed to review."""
        decision = decide_detection(_detection("time"), OCRResult(text="around noon"))

        self.assertEqual(decision.action, "review")
        self.assertEqual(decision.reason, "time_text_ambiguous")


if __name__ == "__main__":
    unittest.main()
