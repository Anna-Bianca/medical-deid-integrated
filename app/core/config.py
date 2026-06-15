# file: app/core/config.py
# description: Defines runtime configuration values and path helpers for the medical de-identification pipeline.
# author: Maria Victoria Anconetani; Anna Bianca Marzetti Biggi
# date: 15/06/2026

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class AppConfig:
    """Stores filesystem paths and runtime switches used across the application."""

    project_root: Path
    full_dataset_dir: Path
    models_dir: Path
    outputs_dir: Path
    static_dir: Path
    base_weights_path: Path
    best_model_path: Path
    detector_conf: float = 0.25
    box_padding: int = 4
    enable_full_image_fallback: bool = False
    save_debug_images: bool = False
    save_debug_report: bool = False
    redaction_granularity: str = "character"
    redaction_fallback: str = "full_roi_inpaint"
    character_mask_padding: int = 0
    token_mask_padding: int = 1
    roi_mask_padding: int = 4
    tesseract_cmd: str = ""

    @classmethod
    def from_env(cls) -> "AppConfig":
        """
        Builds an application configuration instance from environment variables.

        Args:
            cls (type[AppConfig]): Configuration class to instantiate.

        Returns:
            AppConfig: Frozen configuration populated with project paths and environment overrides.
        """
        project_root = Path(__file__).resolve().parents[2]
        full_dataset_dir = project_root / "full_dataset" / "hackathon_TREE_AIBiomed"
        models_dir = project_root / "models"
        outputs_dir = project_root / "outputs"
        static_dir = project_root / "app" / "static"
        base_weights_path = models_dir / "yolov8s.pt"
        best_model_path = models_dir / "best.pt"
        return cls(
            project_root=project_root,
            full_dataset_dir=full_dataset_dir,
            models_dir=models_dir,
            outputs_dir=outputs_dir,
            static_dir=static_dir,
            base_weights_path=base_weights_path,
            best_model_path=best_model_path,
            detector_conf=float(os.getenv("DEID_DETECTOR_CONF", "0.25")),
            box_padding=int(os.getenv("DEID_BOX_PADDING", "4")),
            enable_full_image_fallback=os.getenv("DEID_ENABLE_FULL_IMAGE_FALLBACK", "false").lower() == "true",
            save_debug_images=os.getenv("DEID_SAVE_DEBUG_IMAGES", "false").lower() == "true",
            save_debug_report=os.getenv("DEID_SAVE_DEBUG_REPORT", "false").lower() == "true",
            redaction_granularity=os.getenv("DEID_REDACTION_GRANULARITY", "character").strip() or "character",
            redaction_fallback=os.getenv("DEID_REDACTION_FALLBACK", "full_roi_inpaint").strip()
            or "full_roi_inpaint",
            character_mask_padding=int(os.getenv("DEID_CHARACTER_MASK_PADDING", "0")),
            token_mask_padding=int(os.getenv("DEID_TOKEN_MASK_PADDING", "1")),
            roi_mask_padding=int(os.getenv("DEID_ROI_MASK_PADDING", "4")),
            tesseract_cmd=os.getenv("TESSERACT_CMD", "").strip(),
        )

    def ensure_runtime_dirs(self) -> None:
        """
        Creates the runtime output directories required by the API and CLI.

        Args:
            self (AppConfig): Configuration holding the output and static directory paths.

        Returns:
            None: This method mutates the filesystem by creating missing directories.
        """
        self.outputs_dir.mkdir(parents=True, exist_ok=True)
        self.static_dir.mkdir(parents=True, exist_ok=True)

    @property
    def full_dataset_data_yaml(self) -> Path:
        """
        Returns the canonical dataset YAML path.

        Args:
            self (AppConfig): Configuration holding the dataset root path.

        Returns:
            Path: Path to `full_dataset/hackathon_TREE_AIBiomed/data.yaml`.
        """
        return self.full_dataset_dir / "data.yaml"

    def full_dataset_images_dir(self, split: str) -> Path:
        """
        Resolves the image directory for a dataset split.

        Args:
            self (AppConfig): Configuration holding the dataset root path.
            split (str): Dataset split name such as `train` or `val`.

        Returns:
            Path: Path to the images directory for the requested split.
        """
        return self.full_dataset_dir / "images" / split

    def full_dataset_labels_dir(self, split: str) -> Path:
        """
        Resolves the label directory for a dataset split.

        Args:
            self (AppConfig): Configuration holding the dataset root path.
            split (str): Dataset split name such as `train` or `val`.

        Returns:
            Path: Path to the labels directory for the requested split.
        """
        return self.full_dataset_dir / "labels" / split
