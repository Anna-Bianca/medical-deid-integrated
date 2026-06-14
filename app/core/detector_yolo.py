from __future__ import annotations

from pathlib import Path

import numpy as np
from ultralytics import YOLO

from .config import AppConfig
from .types import Detection


EXPECTED_CLASS_NAMES = {
    0: "name",
    1: "id",
    2: "age",
    3: "date",
    4: "time",
}


class YoloDetector:
    def __init__(self, config: AppConfig, model_path: Path | None = None) -> None:
        self.config = config
        self.model_path = Path(model_path or config.best_model_path)
        self._model: YOLO | None = None

    @property
    def model(self) -> YOLO:
        if self._model is None:
            if not self.model_path.exists():
                raise FileNotFoundError(
                    f"YOLO model not found at {self.model_path}. "
                    "Train a model or place best.pt in the models folder."
                )
            self._model = YOLO(str(self.model_path))
        return self._model

    def get_model_names(self) -> dict[int, str]:
        names = self.model.names
        if isinstance(names, dict):
            return {int(class_id): str(name) for class_id, name in names.items()}
        return {index: str(name) for index, name in enumerate(names)}

    def resolve_class_name(self, class_id: int) -> str:
        if class_id in EXPECTED_CLASS_NAMES:
            return EXPECTED_CLASS_NAMES[class_id]
        return self.get_model_names().get(class_id, f"class_{class_id}")

    def describe_model(self) -> dict:
        model_names = self.get_model_names()
        expected_ids = sorted(EXPECTED_CLASS_NAMES)
        aligned = set(model_names) == set(expected_ids) and all(
            model_names.get(class_id) == EXPECTED_CLASS_NAMES[class_id]
            for class_id in expected_ids
        )
        return {
            "model_path": str(self.model_path),
            "expected_classes": EXPECTED_CLASS_NAMES,
            "model_classes": model_names,
            "aligned_with_project_classes": aligned,
        }

    def detect(self, image: np.ndarray, conf: float | None = None) -> list[Detection]:
        results = self.model.predict(source=image, conf=conf or self.config.detector_conf, verbose=False)
        height, width = image.shape[:2]
        detections: list[Detection] = []

        for result in results:
            for box in result.boxes:
                cls_id = int(box.cls)
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                roi_box = (x1, y1, x2, y2)
                expanded_box = (
                    max(0, x1 - self.config.box_padding),
                    max(0, y1 - self.config.box_padding),
                    min(width, x2 + self.config.box_padding),
                    min(height, y2 + self.config.box_padding),
                )
                detections.append(
                    Detection(
                        class_id=cls_id,
                        class_name=self.resolve_class_name(cls_id),
                        detector_conf=float(box.conf),
                        roi_box=roi_box,
                        expanded_box=expanded_box,
                    )
                )

        detections.sort(key=lambda d: (d.expanded_box[1], d.expanded_box[0]))
        return detections
