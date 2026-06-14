from __future__ import annotations

from dataclasses import replace
import unittest
from unittest.mock import patch

import numpy as np

from app.core.config import AppConfig
from app.core.redactor import redact_image
from app.core.types import Decision, Detection, OCRCharacter, OCRResult, OCRToken, ProcessedDetection


def _config() -> AppConfig:
    return AppConfig.from_env()


def _processed_detection(ocr: OCRResult) -> ProcessedDetection:
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
    def setUp(self) -> None:
        self.image = np.full((60, 60, 3), 255, dtype=np.uint8)
        self.segmented_mask = np.zeros((60, 60), dtype=np.uint8)
        self.segmented_mask[12:18, 12:24] = 255

    def test_uses_character_mask_when_available(self) -> None:
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

        def fake_inpaint(image: np.ndarray, mask: np.ndarray, method: str, radius: int) -> np.ndarray:
            captured["mask"] = mask.copy()
            captured["method"] = method
            return image

        with (
            patch("app.core.redactor._box_mask_from_boxes", side_effect=fake_box_mask_from_boxes),
            patch("app.core.redactor._apply_inpaint_method", side_effect=fake_inpaint),
        ):
            redact_image(self.image, [processed], _config())

        self.assertEqual(processed.redaction_method, "inpaint_biharmonic")
        self.assertEqual(processed.mask_source, "character_box_mask")
        self.assertIn("mask", captured)
        self.assertGreater(int(captured["mask"].sum()), 0)
        self.assertEqual(processed.mask_pixel_count, int(np.count_nonzero(self.segmented_mask)))
        self.assertEqual(captured["method"], "biharmonic")

    def test_falls_back_to_token_mask(self) -> None:
        processed = _processed_detection(
            OCRResult(
                text="AB",
                tokens=[OCRToken(text="AB", conf=95.0, box=(12, 12, 12, 6), source_pass="token")],
            )
        )

        with (
            patch("app.core.redactor._mask_from_boxes", return_value=(self.segmented_mask.copy(), False)),
            patch("app.core.redactor._apply_inpaint_method", return_value=self.image),
        ):
            redact_image(self.image, [processed], _config())

        self.assertEqual(processed.mask_source, "token_segmented")

    def test_falls_back_to_roi_when_no_ocr_geometry_exists(self) -> None:
        processed = _processed_detection(OCRResult(text=""))

        with patch("app.core.redactor._apply_inpaint_method", return_value=self.image):
            redact_image(self.image, [processed], _config())

        self.assertEqual(processed.mask_source, "roi_fallback")

    def test_uses_character_box_mask_when_available(self) -> None:
        processed = _processed_detection(
            OCRResult(
                text="AB",
                characters=[OCRCharacter(text="A", box=(12, 12, 4, 6), source_pass="char")],
            )
        )

        with (
            patch("app.core.redactor._box_mask_from_boxes", return_value=self.segmented_mask.copy()),
            patch("app.core.redactor._apply_inpaint_method", return_value=self.image),
        ):
            redact_image(self.image, [processed], _config())

        self.assertEqual(processed.mask_source, "character_box_mask")

    def test_compare_mode_runs_all_methods_with_same_mask(self) -> None:
        processed = _processed_detection(
            OCRResult(
                text="AB",
                characters=[OCRCharacter(text="A", box=(12, 12, 4, 6), source_pass="char")],
            )
        )
        config = replace(_config(), compare_inpainting_methods=True, redaction_strategy="telea")
        seen: list[tuple[str, np.ndarray]] = []

        def fake_inpaint(image: np.ndarray, mask: np.ndarray, method: str, radius: int) -> np.ndarray:
            seen.append((method, mask.copy()))
            return image

        with (
            patch("app.core.redactor._box_mask_from_boxes", return_value=self.segmented_mask.copy()),
            patch("app.core.redactor._apply_inpaint_method", side_effect=fake_inpaint),
        ):
            _, variants, _, methods = redact_image(self.image, [processed], config)

        self.assertEqual(methods, ["biharmonic", "telea", "ns"])
        self.assertEqual(set(variants.keys()), {"biharmonic", "telea", "ns"})
        self.assertEqual([method for method, _ in seen], methods)
        self.assertTrue(all(np.array_equal(mask, seen[0][1]) for _, mask in seen[1:]))
        self.assertEqual(processed.redaction_method, "inpaint_compare[biharmonic,telea,ns]")


if __name__ == "__main__":
    unittest.main()
