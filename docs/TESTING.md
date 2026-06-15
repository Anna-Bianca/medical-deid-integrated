<!-- file: docs/TESTING.md -->
<!-- description: Collects the local testing and validation steps for the medical de-identification project. -->
<!-- author: Maria Victoria Anconetani; Anna Bianca Marzetti Biggi -->
<!-- date: 15/06/2026 -->

# Testing local

## Preparacion

1. Activar entorno virtual.
2. Instalar dependencias.
3. Confirmar Tesseract.
4. Confirmar `models/yolov8s.pt`.
5. Confirmar `full_dataset/hackathon_TREE_AIBiomed`.
6. Si existe, confirmar `models/best.pt`.
7. Si se desea correr `smoke`, crear una carpeta local `samples/` y colocar alli imagenes de prueba.

## Chequeo de entorno

```powershell
python check_env.py
```

## Validacion visual del dataset

```powershell
python -m app.cli visualize --split train --n 5
python -m app.cli visualize --split val --n 5 --save
```

## Entrenamiento

```powershell
python -m app.cli train
python -m app.cli train --profile historical
```

## Smoke test

```powershell
python -m app.cli smoke
```

Por default, `smoke` usa `samples/`, no `train` ni `val`.

Importante:

- `samples/` es una carpeta local y esta en `.gitignore`
- las imagenes usadas en `samples/` pueden contener PII y no deben versionarse
- los artifacts generados en `outputs/smoke/` tambien son locales y no se versionan

## Batch manual

```powershell
python -m app.cli run --input samples --output outputs/manual_run --enable-full-image-fallback --save-debug-report
```

## API y UI

```powershell
python -m app.cli serve --host 127.0.0.1 --port 8000
```

Luego abrir:

```text
http://localhost:8000/
```

## Que revisar

- imagen original vs redactada
- detecciones por clase
- OCR resultante
- casos `review`
- reportes JSON
- si el fallback fue usado
- si `models/best.pt` expone solo `name`, `id`, `age`, `date`, `time`

## Nota de privacidad

El flujo de validacion debe asumirse como local. No se deben subir a git:

- imagenes de `samples/`
- dataset bajo `full_dataset/.../images` y `labels`
- reports, overlays o redacciones generadas en `outputs/smoke/`
- artifacts de entrenamiento en `outputs/train_runs/`
