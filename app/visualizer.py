from __future__ import annotations

import random
from pathlib import Path

import cv2
import numpy as np

from app.core.detector_yolo import EXPECTED_CLASS_NAMES
from app.core.config import AppConfig
from app.dataset_prep import resolve_label_for_image


CLASS_COLORS = {
    0: (56, 182, 255),
    1: (80, 220, 80),
    2: (255, 100, 100),
    3: (180, 80, 255),
    4: (255, 220, 40),
}


def draw_boxes(image_path: Path, label_path: Path) -> np.ndarray:
    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")
    height, width = image.shape[:2]
    if not label_path.exists():
        return image

    lines = label_path.read_text(encoding="utf-8").splitlines()
    for line in lines:
        parts = line.split()
        if len(parts) != 5:
            continue
        cls_id = int(parts[0])
        x_center, y_center, bw, bh = map(float, parts[1:])
        x1 = int((x_center - bw / 2) * width)
        y1 = int((y_center - bh / 2) * height)
        x2 = int((x_center + bw / 2) * width)
        y2 = int((y_center + bh / 2) * height)
        color = CLASS_COLORS.get(cls_id, (255, 255, 255))
        label = EXPECTED_CLASS_NAMES.get(cls_id, f"class_{cls_id}")
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        cv2.putText(image, label, (x1, max(18, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
    return image


def visualize_split(split: str, n: int, save: bool = False, seed: int = 42) -> list[Path]:
    config = AppConfig.from_env()
    images_dir = config.full_dataset_images_dir(split)
    labels_dir = config.full_dataset_labels_dir(split)
    output_dir = config.outputs_dir / "visualizations"
    output_dir.mkdir(parents=True, exist_ok=True)

    random.seed(seed)
    all_images = sorted(images_dir.glob("*.*"))
    if not all_images:
        raise FileNotFoundError(f"No images found in {images_dir}")
    sample = random.sample(all_images, min(n, len(all_images)))
    saved_paths: list[Path] = []

    for image_path in sample:
        resolved = resolve_label_for_image(image_path, labels_dir)
        label_path = resolved if resolved is not None else labels_dir / f"{image_path.stem}.txt"
        rendered = draw_boxes(image_path, label_path)
        if save:
            out_path = output_dir / f"viz_{image_path.name}"
            cv2.imwrite(str(out_path), rendered)
            saved_paths.append(out_path)
        else:
            cv2.imshow(f"{image_path.name} | press any key, q to quit", rendered)
            key = cv2.waitKey(0) & 0xFF
            cv2.destroyAllWindows()
            if key == ord("q"):
                break

    return saved_paths
