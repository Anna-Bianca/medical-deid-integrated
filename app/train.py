from __future__ import annotations

from pathlib import Path
import shutil

import torch
from ultralytics import YOLO

from app.core.config import AppConfig
from app.dataset_prep import prepare_yolo_dataset


TRAINING_PROFILES = {
    "historical": {
        "epochs": 100,
        "imgsz": 640,
        "batch": 8,
        "augment": True,
        "fliplr": 0.5,
        "degrees": 5.0,
        "scale": 0.3,
        "mosaic": 1.0,
        "copy_paste": 0.3,
        "hsv_v": 0.4,
        "box": 9.0,
        "cls": 0.7,
        "patience": 20,
        "save": True,
        "save_period": 10,
        "val": True,
        "optimizer": "AdamW",
        "lr0": 0.001,
        "lrf": 0.1,
        "verbose": True,
    }
}


def _resolve_path(path: Path | None, fallback: Path) -> Path:
    return Path(path) if path is not None else fallback


def default_resume_checkpoint(config: AppConfig) -> Path:
    return config.outputs_dir / "train_runs" / "deidentification" / "weights" / "last.pt"


def _require_training_inputs(data_yaml: Path, base_weights_path: Path) -> None:
    if not data_yaml.exists():
        raise FileNotFoundError(
            f"Missing training dataset config at {data_yaml}. "
            "Expected full_dataset/hackathon_TREE_AIBiomed/data.yaml."
        )
    if not base_weights_path.exists():
        raise FileNotFoundError(
            f"Missing base weights at {base_weights_path}. "
            "Copy yolov8s.pt into models/ or pass --base-weights."
        )


def _require_dataset_yaml(data_yaml: Path) -> None:
    if not data_yaml.exists():
        raise FileNotFoundError(
            f"Missing training dataset config at {data_yaml}. "
            "Expected full_dataset/hackathon_TREE_AIBiomed/data.yaml."
        )


def train_model(
    *,
    dataset_root: Path | None = None,
    base_weights_path: Path | None = None,
    profile: str = "historical",
    epochs: int = 100,
    img_size: int = 640,
    batch_size: int = 8,
    resume: bool = False,
    resume_from: Path | None = None,
) -> Path:
    config = AppConfig.from_env()
    config.ensure_runtime_dirs()
    if profile not in TRAINING_PROFILES:
        raise ValueError(f"Unsupported training profile '{profile}'. Available: {', '.join(TRAINING_PROFILES)}")

    dataset_root = _resolve_path(dataset_root, config.full_dataset_dir)
    base_weights_path = _resolve_path(base_weights_path, config.base_weights_path)
    data_yaml = dataset_root / "data.yaml"
    _require_training_inputs(data_yaml, base_weights_path)
    prepared_root = config.outputs_dir / "prepared_datasets" / f"{dataset_root.name}_yolo"
    prepared_root, stats = prepare_yolo_dataset(dataset_root, prepared_root)
    prepared_data_yaml = prepared_root / "data.yaml"

    profile_settings = TRAINING_PROFILES[profile].copy()
    profile_settings["epochs"] = epochs
    profile_settings["imgsz"] = img_size
    profile_settings["batch"] = batch_size

    print(f"[Train] Dataset root: {dataset_root}", flush=True)
    print(f"[Train] Data YAML: {data_yaml}", flush=True)
    print(f"[Train] Prepared dataset root: {prepared_root}", flush=True)
    print(
        "[Train] Prepared dataset stats: "
        f"train_images={stats['images_train']}, train_labels={stats['labels_train']}, "
        f"val_images={stats['images_val']}, val_labels={stats['labels_val']}, "
        f"missing_train={stats['missing_train']}, missing_val={stats['missing_val']}",
        flush=True,
    )
    device = "0" if torch.cuda.is_available() else "cpu"
    runs_dir = config.outputs_dir / "train_runs"
    run_weights_dir = runs_dir / "deidentification" / "weights"

    if resume_from is not None:
        resume = True

    if resume:
        checkpoint_path = _resolve_path(resume_from, default_resume_checkpoint(config))
        if not checkpoint_path.exists():
            raise FileNotFoundError(
                f"Resume checkpoint not found at {checkpoint_path}. "
                "Use --resume-from with a valid last.pt or start a fresh training run."
            )
        print(f"[Train] Resuming from checkpoint: {checkpoint_path}", flush=True)
        model = YOLO(str(checkpoint_path))
        model.train(resume=True)
        best_path = run_weights_dir / "best.pt"
        if not best_path.exists():
            raise FileNotFoundError(f"Resumed training finished but best.pt was not found in {run_weights_dir}")
        shutil.copy2(best_path, config.best_model_path)
        return config.best_model_path

    print(f"[Train] Base weights: {base_weights_path}", flush=True)
    print(f"[Train] Profile: {profile}", flush=True)

    model = YOLO(str(base_weights_path))
    profile_settings.update(
        {
            "data": str(prepared_data_yaml),
            "device": device,
            "project": str(runs_dir),
            "name": "deidentification",
            "exist_ok": True,
        }
    )
    results = model.train(
        **profile_settings,
    )
    best_path = Path(results.save_dir) / "weights" / "best.pt"
    if not best_path.exists():
        raise FileNotFoundError(f"Training finished but best.pt was not found in {best_path.parent}")
    shutil.copy2(best_path, config.best_model_path)
    return config.best_model_path


def validate_trained_model(
    *,
    dataset_root: Path | None = None,
    img_size: int = 640,
) -> dict[str, float]:
    config = AppConfig.from_env()
    dataset_root = _resolve_path(dataset_root, config.full_dataset_dir)
    data_yaml = dataset_root / "data.yaml"
    _require_dataset_yaml(data_yaml)
    prepared_root = config.outputs_dir / "prepared_datasets" / f"{dataset_root.name}_yolo"
    prepared_root, _ = prepare_yolo_dataset(dataset_root, prepared_root)
    prepared_data_yaml = prepared_root / "data.yaml"
    if not config.best_model_path.exists():
        raise FileNotFoundError(f"Missing trained model at {config.best_model_path}")
    model = YOLO(str(config.best_model_path))
    metrics = model.val(data=str(prepared_data_yaml), imgsz=img_size)
    return {
        "map50": float(metrics.box.map50),
        "map50_95": float(metrics.box.map),
        "precision": float(metrics.box.mp),
        "recall": float(metrics.box.mr),
    }
