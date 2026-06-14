# Medical DeID Integrated

Solucion integrada para desidentificacion de imagenes medicas que combina:

- entrenamiento reproducible de un detector YOLO
- OCR robusto multi-pass con Tesseract
- politica de decision por clase y evidencia OCR
- redaccion visual con inpainting fino
- exposicion por `CLI`, `API` y `UI`

## Antes de empezar

### Aclaracion importante sobre el dataset

Las imagenes y labels disponibles en `full_dataset/` no siguen del todo buenas practicas de naming para entrenamiento directo con Ultralytics.

El caso concreto es este:

- imagen: `algo_annotated.png`
- label: `algo.txt`

Ultralytics espera por default:

- imagen: `algo.png`
- label: `algo.txt`

o bien:

- imagen: `algo_annotated.png`
- label: `algo_annotated.txt`

Por eso este repo agrega una etapa interna de normalizacion antes de entrenar. El dataset original no se toca: se prepara una copia temporal compatible dentro de `outputs/prepared_datasets/`.

Tambien es importante que:

- `full_dataset/hackathon_TREE_AIBiomed/data.yaml` sea YAML puro
- no contenga fences de Markdown como `````yaml````` o ``````

## Que es esta solucion

La solucion base es un pipeline Python local. La UI no es obligatoria.

El mismo sistema puede usarse de tres maneras:

- `CLI`: para entrenar, validar, probar y procesar lotes
- `API`: para integrarlo con otro sistema via HTTP
- `UI`: para demo manual de punta a punta sobre la API

### En que se basa la solucion

La estrategia es híbrida:

1. YOLO detecta regiones candidatas de PII en la imagen
2. OCR intenta leer el texto dentro de esas regiones
3. una policy decide si cada deteccion debe ir a `redact` o `review`
4. si la decision es `redact`, el texto se elimina con una mascara fina e inpainting

La prioridad del diseño es:

- preservar privacidad
- conservar la mayor cantidad posible de informacion medica alrededor del texto
- mantener trazabilidad de por que algo fue redactado o enviado a revision

## Como esta estructurado el repo

```text
medical-deid-integrated/
  app/
    core/
      config.py
      detector_yolo.py
      ocr_engine.py
      decision_policy.py
      redactor.py
      pipeline.py
      reporting.py
    api.py
    cli.py
    train.py
    visualizer.py
    dataset_prep.py
    static/
  full_dataset/
    hackathon_TREE_AIBiomed/
      images/
      labels/
      data.yaml
  models/
  outputs/
  samples/
  docs/
  check_env.py
  setup.ps1
```

### Carpetas clave

- `full_dataset/hackathon_TREE_AIBiomed/`: fuente de verdad para training y validation
- `samples/`: imagenes reservadas para prueba manual fuera de `train` y `val`
- `outputs/prepared_datasets/`: staging temporal generado por el repo para compatibilizar imagenes y labels
- `outputs/train_runs/`: artefactos completos del entrenamiento YOLO
- `models/`: peso base y modelo final activo

## Flujo recomendado para ponerlo a funcionar

### 1. Preparar entorno

Desde PowerShell, parado en la raiz del repo:

```powershell
powershell -ExecutionPolicy Bypass -File .\setup.ps1
.\.venv\Scripts\Activate.ps1
```

### 2. Confirmar prerequisitos

Este proyecto necesita:

- Python 3.10+
- Tesseract instalado
- `full_dataset/hackathon_TREE_AIBiomed`
- `models/yolov8s.pt` como peso base oficial

Chequeo rapido:

```powershell
python check_env.py
```

### 3. Instalar dependencias faltantes

Si no corriste `setup.ps1`, o si agregaste dependencias nuevas:

```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Entrenar desde cero

El entrypoint oficial es:

```powershell
python -m app.cli train
```

Ese comando:

- toma `full_dataset/hackathon_TREE_AIBiomed`
- prepara un dataset temporal compatible en `outputs/prepared_datasets/`
- entrena con la receta `historical`
- guarda artefactos en `outputs/train_runs/deidentification/`
- copia el mejor modelo a `models/best.pt`
 
### 5. Reanudar entrenamiento si fue interrumpido

Checkpoint esperado:

```text
outputs/train_runs/deidentification/weights/last.pt
```

Comandos:

```powershell
python -m app.cli train --resume
python -m app.cli train --resume-from outputs/train_runs/deidentification/weights/last.pt
```

### 6. Probar el pipeline

Smoke test sobre `samples/`:

```powershell
python -m app.cli smoke
```

Procesamiento manual:

```powershell
python -m app.cli run --input samples --output outputs/manual_run --enable-full-image-fallback --save-debug-report
```

### 7. Levantar API y UI

```powershell
python -m app.cli serve --host 127.0.0.1 --port 8000
```

Abrir luego:

```text
http://localhost:8000/
```

## Entrenamiento reproducible en detalle

### Defaults del baseline oficial

- dataset: `full_dataset/hackathon_TREE_AIBiomed`
- base weights: `models/yolov8s.pt`
- profile: `historical`
- epochs: `100`
- img size: `640`
- batch size: `8`

### Comandos utiles

```powershell
python -m app.cli train
python -m app.cli train --dataset-root full_dataset/hackathon_TREE_AIBiomed
python -m app.cli train --base-weights models/yolov8s.pt --profile historical
python -m app.cli train --skip-validate
python -m app.cli train --resume
```

### Donde quedan los resultados del training

Los archivos importantes quedan en:

```text
outputs/train_runs/deidentification/
```

En particular:

- `results.png`: curvas y metricas por epoch
- `results.csv`: valores tabulares por epoch
- `weights/best.pt`: mejor checkpoint
- `weights/last.pt`: ultimo checkpoint para resume

Ademas, el repo copia el modelo final activo a:

```text
models/best.pt
```

## Como funciona la redaccion

La redaccion ya no usa un rectangulo negro completo.

Ahora el flujo es:

1. se detecta una ROI con YOLO
2. OCR obtiene tokens y, para detecciones redactables, tambien boxes por caracter
3. se construye una mascara fina sobre los caracteres
4. se aplica inpainting biharmonico

Fallbacks:

- si no hay caracteres OCR, usa boxes por token
- si tampoco hay geometria OCR util, inpainta toda la ROI

Eso permite preservar mejor el contenido medico alrededor del texto sensible.

## API y UI

### Endpoints principales

- `GET /health`
- `POST /deidentify`
- `POST /deidentify/report`

### Que devuelve cada uno

- `/health`: estado del servicio y paths/config relevantes
- `/deidentify`: imagen procesada + headers con resumen
- `/deidentify/report`: solo reporte JSON del pipeline

La UI en `/` consume esos endpoints y permite:

- subir una imagen
- ejecutar el pipeline
- ver original vs procesada
- inspeccionar detecciones, OCR y decisiones

## Diagnostico rapido de problemas comunes

### 1. `data.yaml` con error de sintaxis

Revisar que `full_dataset/hackathon_TREE_AIBiomed/data.yaml` no tenga fences Markdown.

Contenido correcto:

```yaml
train: images/train
val: images/val

nc: 5

names:
  0: name
  1: id
  2: age
  3: date
  4: time
```

### 2. YOLO dice que todo es `background`

Eso suele indicar que no pudo matchear imagenes con labels.

En este repo se resuelve automaticamente con la etapa de preparacion en `outputs/prepared_datasets/`.

### 3. El modelo detecta clases como `bed`, `person` o `class_59`

Eso indica que `models/best.pt` no fue generado por este flujo y se esta usando un modelo generalista ajeno al problema medico.

Este proyecto espera exactamente:

- `0`: `name`
- `1`: `id`
- `2`: `age`
- `3`: `date`
- `4`: `time`

### 4. Falla el inpainting

Revisar que `scikit-image` este instalado:

```powershell
pip install -r requirements.txt
```

### 5. Falta Tesseract

En Windows:

```powershell
winget install UB-Mannheim.TesseractOCR
```

Si no queda en `PATH`:

```powershell
$env:TESSERACT_CMD="C:\Program Files\Tesseract-OCR\tesseract.exe"
```

## Recorrido recomendado para un evaluador externo

Si alguien ya tiene el dataset completo cargado en `full_dataset/`, el camino mas simple es:

```powershell
powershell -ExecutionPolicy Bypass -File .\setup.ps1
.\.venv\Scripts\Activate.ps1
python check_env.py
python -m app.cli train
python -m app.cli smoke
python -m app.cli serve --host 127.0.0.1 --port 8000
```

Con eso puede:

- reconstruir `models/best.pt`
- validar un smoke test sobre `samples/`
- abrir la UI y probar una imagen manualmente

## Reproducibilidad

Si alguien tiene acceso a:

- `full_dataset/hackathon_TREE_AIBiomed`
- `models/yolov8s.pt`

puede reproducir el entrenamiento completo y regenerar el modelo activo desde este mismo repo, sin depender de otro proyecto externo.

Contrato operativo del repo:

- dataset canonico: `full_dataset/hackathon_TREE_AIBiomed`
- peso base oficial: `models/yolov8s.pt`
- modelo final activo: `models/best.pt`

## Documentacion complementaria

Ver tambien:

- [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)
- [docs/PROCESS_HISTORY.md](docs/PROCESS_HISTORY.md)
- [docs/TESTING.md](docs/TESTING.md)
