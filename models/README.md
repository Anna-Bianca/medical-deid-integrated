# Modelos

- `yolov8s.pt`: peso base oficial para el retraining reproducible documentado en este repo.
- `best.pt`: peso final entrenado del detector, generado por `python -m app.cli train`.
- `yolov8n.pt`: peso base viejo conservado solo como referencia historica; no es el baseline oficial actual.

El pipeline de inferencia consume siempre `models/best.pt`.
