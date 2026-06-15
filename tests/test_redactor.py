# file: tests/test_redactor.py
# description: Verifies mask selection and inpainting orchestration behavior for the redactor module.
# author: Maria Victoria Anconetani; Anna Bianca Marzetti Biggi
# date: 15/06/2026

from __future__ import annotations

import unittest
from unittest.mock import patch

import numpy as np

from app.core.config import AppConfig
from app.core.redactor import redact_image
from app.core.types import Decision, Detection, OCRCharacter, OCRResult, OCRToken, ProcessedDetection


def _config() -> AppConfig:
    """
    Loads the default application configuration for redactor tests.

    Args:
        None

    Returns:
        AppConfig: Configuration populated from the local environment defaults.
    """
    return AppConfig.from_env()


def _processed_detection(ocr: OCRResult) -> ProcessedDetection:
    """
    Builds a redactable processed detection with stable geometry for tests.

    Args:
        ocr (OCRResult): OCR payload to attach to the synthetic detection.

    Returns:
        ProcessedDetection: Detection configured to exercise mask-selection behavior.
    """
    return ProcessedDetection(
        detection=Detection(
            class_id=0,
            class_name="name",
            detector_conf=0.9,
            roi_box=(10, 10, 40, 30),
            expanded_box=(8, 8, 42, 32),
        ),
        ocr=ocr,
        decision=Decision(action="redact", reason="class_policy_always_redact"),
    )


class RedactorTests(unittest.TestCase):
    """Covers the mask source selection and no-op behavior of the redactor."""

    def setUp(self) -> None:
        """Creates a synthetic image and reusable mask fixture for each test case."""
        self.image = np.full((60, 60, 3), 255, dtype=np.uint8)
        self.segmented_mask = np.zeros((60, 60), dtype=np.uint8)
        self.segmented_mask[12:18, 12:24] = 255

    def test_uses_character_mask_when_available(self) -> None:
        """Prefers character boxes when character OCR geometry is available."""
        processed = _processed_detection(
            OCRResult(
                text="AB",
                characters=[
                    OCRCharacter(text="A", box=(12, 12, 4, 6), source_pass="char"),
                    OCRCharacter(text="B", box=(18, 12, 4, 6), source_pass="char"),
                ],
                tokens=[OCRToken(text="AB", conf=95.0, box=(12, 12, 12, 6), source_pass="token")],
            )
        )

        captured: dict[str, np.ndarray] = {}

        def fake_box_mask_from_boxes(*args, **kwargs) -> np.ndarray:
            return self.segmented_mask.copy()

        def fake_inpaint(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
            captured["mask"] = mask.copy()
            return image

        with (
            patch("app.core.redactor._box_mask_from_boxes", side_effect=fake_box_mask_from_boxes),
            patch("app.core.redactor._apply_biharmonic_inpaint", side_effect=fake_inpaint),
        ):
            redact_image(self.image, [processed], _config())

        self.assertEqual(processed.redaction_method, "inpaint_biharmonic")
        self.assertEqual(processed.mask_source, "character_box_mask")
        self.assertIn("mask", captured)
        self.assertGreater(int(captured["mask"].sum()), 0)
        self.assertEqual(processed.mask_pixel_count, int(np.count_nonzero(self.segmented_mask)))

    def test_falls_back_to_token_mask(self) -> None:
        """Uses token segmentation when character-level geometry is not available."""
        processed = _processed_detection(
            OCRResult(
                text="AB",
                tokens=[OCRToken(text="AB", conf=95.0, box=(12, 12, 12, 6), source_pass="token")],
            )
        )

        with (
            patch("app.core.redactor._mask_from_boxes", return_value=(self.segmented_mask.copy(), False)),
            patch("app.core.redactor._apply_biharmonic_inpaint", return_value=self.image),
        ):
            redact_image(self.image, [processed], _config())

        self.assertEqual(processed.mask_source, "token_segmented")

    def test_falls_back_to_roi_when_no_ocr_geometry_exists(self) -> None:
        """Falls back to the expanded ROI when OCR provides no usable geometry."""
        processed = _processed_detection(OCRResult(text=""))

        with patch("app.core.redactor._apply_biharmonic_inpaint", return_value=self.image):
            redact_image(self.image, [processed], _config())

        self.assertEqual(processed.mask_source, "roi_fallback")

    def test_uses_character_box_mask_when_available(self) -> None:
        """Records the expected mask source when a character-box mask is used."""
        processed = _processed_detection(
            OCRResult(
                text="AB",
                characters=[OCRCharacter(text="A", box=(12, 12, 4, 6), source_pass="char")],
            )
        )

        with (
            patch("app.core.redactor._box_mask_from_boxes", return_value=self.segmented_mask.copy()),
            patch("app.core.redactor._apply_biharmonic_inpaint", return_value=self.image),
        ):
            redact_image(self.image, [processed], _config())

        self.assertEqual(processed.mask_source, "character_box_mask")

    def test_returns_original_image_when_nothing_is_redacted(self) -> None:
        """Returns the original image and an empty mask when all detections are review-only."""
        processed = _processed_detection(
            OCRResult(
                text="AB",
                characters=[OCRCharacter(text="A", box=(12, 12, 4, 6), source_pass="char")],
            )
        )
        processed.decision = Decision(action="review", reason="manual_review_needed")

        redacted_image, redaction_mask = redact_image(self.image, [processed], _config())

        self.assertTrue(np.array_equal(redacted_image, self.image))
        self.assertEqual(int(np.count_nonzero(redaction_mask)), 0)
        self.assertIsNone(processed.redaction_method)


if __name__ == "__main__":
    unittest.main()
