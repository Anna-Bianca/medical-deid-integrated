<!-- file: docs/ARCHITECTURE.md -->
<!-- description: High-level architecture overview of the integrated medical image de-identification solution. -->
<!-- author: Maria Victoria Anconetani; Anna Bianca Marzetti Biggi -->
<!-- date: 15/06/2026 -->

# Arquitectura final

## Resumen

La solucion integrada toma lo mejor de los dos enfoques previos:

- YOLO para localizar regiones candidatas de PII
- OCR robusto multi-pass para validar y leer texto dentro de cada ROI
- politica de decision por clase y evidencia OCR
- redaccion final mediante mascara fina e inpainting biharmonico
- exposicion por CLI, API y UI

## Flujo end-to-end

```text
full_dataset/hackathon_TREE_AIBiomed
  -> app.cli train / app.train
  -> outputs/prepared_datasets/
  -> outputs/train_runs/
  -> models/best.pt
  -> detector YOLO
  -> OCR robusto por ROI
  -> politica de decision
  -> redaccion con mascara fina
  -> reporte JSON
  -> API / UI / batch CLI
```

## Fuente de verdad

El entrenamiento y la validacion usan exclusivamente:

```text
full_dataset/hackathon_TREE_AIBiomed
```

En la version compartible del repositorio solo se conservan:

- `full_dataset/hackathon_TREE_AIBiomed/data.yaml`
- `full_dataset/hackathon_TREE_AIBiomed/Readme.md`

Las carpetas `images/` y `labels/` quedan locales y fuera de git para evitar exposicion de PII.

## Modulos principales

- `app/train.py`
  Entrenamiento reproducible con la receta historica y copia automatica de `best.pt`.

- `app/core/detector_yolo.py`
  Localiza regiones candidatas y valida que el modelo cargado este alineado con las clases medicas esperadas.

- `app/core/ocr_engine.py`
  Ejecuta OCR multi-pass con variantes de preprocesamiento, validacion por confianza, repeticion e IoU.

- `app/core/decision_policy.py`
  Decide `redact` o `review`.

- `app/core/redactor.py`
  Construye mascaras finas y aplica inpainting biharmonico sobre las regiones redactables.

- `app/core/pipeline.py`
  Orquesta detector, OCR, decision y redaccion.

- `app/api.py`
  Expone endpoints HTTP y la UI estatica.

- `app/cli.py`
  Expone entrenamiento, visualizacion, smoke, run y serve.

## Politica funcional v1

- `name`, `id`, `age`, `date` -> `redact`
- `time`
  - `redact` si OCR confirma patron legible
  - `review` si OCR es ambiguo o insuficiente

## Fallback

Si YOLO no detecta regiones y el fallback esta activo, se corre OCR full-image. En v1 esto sirve para diagnostico y reporte, no para redactar sin cajas.
