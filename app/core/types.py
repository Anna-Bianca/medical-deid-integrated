# file: app/core/types.py
# description: Declares the structured data types shared across detection, OCR, decision, redaction, and reporting.
# author: Maria Victoria Anconetani; Anna Bianca Marzetti Biggi
# date: 15/06/2026

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class Detection:
    """Represents one YOLO detection with original and expanded bounding boxes."""

    class_id: int
    class_name: str
    detector_conf: float
    roi_box: tuple[int, int, int, int]
    expanded_box: tuple[int, int, int, int]


@dataclass(frozen=True)
class OCRToken:
    """Represents one OCR token with confidence, geometry, and source pass metadata."""

    text: str
    conf: float
    box: tuple[int, int, int, int]
    source_pass: str


@dataclass(frozen=True)
class OCRCharacter:
    """Represents one OCR character-level detection extracted from a redactable ROI."""

    text: str
    box: tuple[int, int, int, int]
    source_pass: str


@dataclass
class OCRResult:
    """Aggregates OCR text, token geometry, character geometry, and debug metadata."""

    text: str
    tokens: list[OCRToken] = field(default_factory=list)
    characters: list[OCRCharacter] = field(default_factory=list)
    conf_summary: dict[str, float | int | None] = field(default_factory=dict)
    support_passes: list[str] = field(default_factory=list)
    debug: dict[str, Any] = field(default_factory=dict)
    debug_available: bool = False


@dataclass(frozen=True)
class Decision:
    """Stores the action and reason chosen by the decision policy for one detection."""

    action: str
    reason: str


@dataclass
class ProcessedDetection:
    """Bundles detection, OCR, and redaction metadata for one processed ROI."""

    detection: Detection
    ocr: OCRResult
    decision: Decision
    source: str = "roi_yolo"
    redaction_method: str | None = None
    mask_source: str | None = None
    mask_pixel_count: int = 0


@dataclass
class PipelineResult:
    """Captures the final outputs and diagnostics generated for one processed image."""

    filename: str
    processed_detections: list[ProcessedDetection]
    redacted_image: Any
    fallback_used: bool
    fallback_ocr: OCRResult | None = None
    redaction_mask: Any | None = None
