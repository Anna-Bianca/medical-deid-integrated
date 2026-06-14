from __future__ import annotations

from pathlib import Path
import shutil


ANNOTATED_SUFFIX = "_annotated"
IMAGE_PATTERNS = ("*.png", "*.jpg", "*.jpeg", "*.bmp", "*.tif", "*.tiff")


def resolve_label_for_image(image_path: Path, labels_dir: Path) -> Path | None:
    candidates = [labels_dir / f"{image_path.stem}.txt"]
    if image_path.stem.endswith(ANNOTATED_SUFFIX):
        base_stem = image_path.stem[: -len(ANNOTATED_SUFFIX)]
        candidates.append(labels_dir / f"{base_stem}.txt")
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def prepare_yolo_dataset(source_root: Path, prepared_root: Path) -> tuple[Path, dict[str, int]]:
    if not source_root.exists():
        raise FileNotFoundError(f"Source dataset root not found: {source_root}")

    source_data_yaml = source_root / "data.yaml"
    if not source_data_yaml.exists():
        raise FileNotFoundError(f"Missing data.yaml at {source_data_yaml}")

    if prepared_root.exists():
        shutil.rmtree(prepared_root)

    stats = {
        "images_train": 0,
        "images_val": 0,
        "labels_train": 0,
        "labels_val": 0,
        "missing_train": 0,
        "missing_val": 0,
    }

    for split in ("train", "val"):
        source_images_dir = source_root / "images" / split
        source_labels_dir = source_root / "labels" / split
        target_images_dir = prepared_root / "images" / split
        target_labels_dir = prepared_root / "labels" / split
        target_images_dir.mkdir(parents=True, exist_ok=True)
        target_labels_dir.mkdir(parents=True, exist_ok=True)

        image_paths: list[Path] = []
        for pattern in IMAGE_PATTERNS:
            image_paths.extend(sorted(source_images_dir.glob(pattern)))

        for image_path in image_paths:
            shutil.copy2(image_path, target_images_dir / image_path.name)
            stats[f"images_{split}"] += 1
            source_label = resolve_label_for_image(image_path, source_labels_dir)
            if source_label is None:
                stats[f"missing_{split}"] += 1
                continue
            target_label = target_labels_dir / f"{image_path.stem}.txt"
            shutil.copy2(source_label, target_label)
            stats[f"labels_{split}"] += 1

    if stats["labels_train"] == 0 and stats["labels_val"] == 0:
        raise RuntimeError(
            "Prepared dataset contains zero matched labels. "
            "Check image/label naming inside full_dataset."
        )

    shutil.copy2(source_data_yaml, prepared_root / "data.yaml")
    return prepared_root, stats
