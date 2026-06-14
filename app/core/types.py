from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class Detection:
    class_id: int
    class_name: str
    detector_conf: float
    roi_box: tuple[int, int, int, int]
    expanded_box: tuple[int, int, int, int]


@dataclass(frozen=True)
class OCRToken:
    text: str
    conf: float
    box: tuple[int, int, int, int]
    source_pass: str


@dataclass(frozen=True)
class OCRCharacter:
    text: str
    box: tuple[int, int, int, int]
    source_pass: str


@dataclass
class OCRResult:
    text: str
    tokens: list[OCRToken] = field(default_factory=list)
    characters: list[OCRCharacter] = field(default_factory=list)
    conf_summary: dict[str, float | int | None] = field(default_factory=dict)
    support_passes: list[str] = field(default_factory=list)
    debug: dict[str, Any] = field(default_factory=dict)
    debug_available: bool = False


@dataclass(frozen=True)
class Decision:
    action: str
    reason: str


@dataclass
class ProcessedDetection:
    detection: Detection
    ocr: OCRResult
    decision: Decision
    source: str = "roi_yolo"
    redaction_method: str | None = None
    mask_source: str | None = None
    mask_pixel_count: int = 0


@dataclass
class PipelineResult:
    filename: str
    processed_detections: list[ProcessedDetection]
    redacted_image: Any
    fallback_used: bool
    fallback_ocr: OCRResult | None = None
    redacted_variants: dict[str, Any] = field(default_factory=dict)
    redaction_mask: Any | None = None
    redaction_methods_run: list[str] = field(default_factory=list)
