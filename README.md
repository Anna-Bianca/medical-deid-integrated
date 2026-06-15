<!-- file: README.md -->
<!-- description: Hackathon-first project overview, evaluation guide, and technical entrypoint for the medical de-identification solution. -->
<!-- author: Maria Victoria Anconetani; Anna Bianca Marzetti Biggi -->
<!-- date: 15/06/2026 -->

# Medical DeID Integrated

## Reto y equipo

**Reto Treelogic:** Desidentificacion visual de imagenes medicas.

**Autoras:** Maria Victoria Anconetani y Anna Bianca Marzetti Biggi.

Este repositorio contiene una solucion end-to-end para detectar y anonimizar informacion identificable incrustada en radiografias e imagenes medicas. El sistema combina deteccion YOLO, OCR robusto multi-pass, una politica de decision por clase y evidencia OCR, y una redaccion visual final basada en mascara fina e inpainting biharmonico.

## Resumen ejecutivo

La solucion prioriza dos objetivos simultaneos:

- proteger privacidad y reducir exposicion de PII
- preservar la mayor cantidad posible de contexto medico util alrededor del texto sensible

Hoy el pipeline trata automaticamente como redactables las clases:

- `name`
- `id`
- `age`
- `date`

La clase `time` sigue una politica mas conservadora:

- `redact` si OCR confirma un patron horario valido
- `review` si el OCR resulta ambiguo o insuficiente

## Entrega del hackathon

Este repositorio cubre el componente de **codigo fuente** de la entrega.

Entregables obligatorios:

- `Codigo fuente`: incluido en este repositorio
- `Memoria tecnica PDF`: pendiente de adjuntar
- `Video de evidencia`: pendiente de adjuntar

Placeholders sugeridos para la entrega final:

- `Memoria tecnica`: `docs/ENTREGA_MEMORIA_TECNICA.pdf` o URL externa
- `Video de evidencia`: URL externa o referencia en release/drive

## Como evaluarlo en 2 minutos

### 1. Preparar entorno

Desde PowerShell, en la raiz del repo:

```powershell
powershell -ExecutionPolicy Bypass -File .\setup.ps1
.\.venv\Scripts\Activate.ps1
```

El script `setup.ps1`:

- crea `.venv` si no existe
- actualiza `pip`
- instala dependencias desde `requirements.txt`
- ejecuta `python check_env.py`

### 2. Confirmar prerequisitos

Para ejecutar el proyecto correctamente se necesita:

- Python 3.10+
- Tesseract instalado o accesible via `PATH`
- `models/yolov8s.pt` para reentrenar
- `full_dataset/hackathon_TREE_AIBiomed/` si se quiere reentrenar
- una carpeta local `samples/` si se quiere correr `smoke` con imagenes propias

Chequeo rapido:

```powershell
python check_env.py
```

### 3. Probar inferencia local

Agregar una o mas imagenes de prueba en la carpeta local `samples/` y ejecutar:

```powershell
python -m app.cli smoke
```

Resultado esperado:

- se genera una imagen redactada en `outputs/smoke/`
- se genera un `report_*.json`
- opcionalmente, con flags debug, se generan `mask_*.png`, `overlay_*.png` y `debug_*.json`

### 4. Levantar API y UI

```powershell
python -m app.cli serve --host 127.0.0.1 --port 8000
```

Abrir luego:

```text
http://localhost:8000/
```

La UI permite:

- subir una imagen
- ejecutar el pipeline
- ver original vs procesada
- inspeccionar detecciones, OCR y decisiones

## Resultados y evidencia tecnica

### Resultados de entrenamiento disponibles

El entrenamiento reproducible del detector YOLO ya produjo artefactos locales en `outputs/train_runs/deidentification/`.

Metricas finales registradas en `results.csv`:

- `precision(B)`: `0.99851`
- `recall(B)`: `1.0`
- `mAP50(B)`: `0.995`
- `mAP50-95(B)`: `0.9449`

### Evidencia funcional esperada

En una corrida `smoke` correcta, el jurado deberia poder verificar:

- deteccion de regiones candidatas de PII
- OCR por ROI
- politica de decision por clase
- redaccion final con inpainting biharmonico
- generacion de reporte JSON y artefactos de depuracion locales

Los artifacts locales de `samples/`, `outputs/smoke/`, `outputs/train_runs/` y `runs/` no se versionan en la version compartible del repositorio para evitar exposicion de PII.

## Arquitectura de la solucion

La estrategia es hibrida:

1. YOLO detecta regiones candidatas de PII
2. OCR multi-pass intenta leer y validar el texto dentro de cada ROI
3. una policy decide si cada deteccion va a `redact` o `review`
4. si la decision es `redact`, se construye una mascara fina y se aplica inpainting biharmonico

Formas de uso:

- `CLI`: entrenamiento, smoke, batch manual, visualizacion y server
- `API`: integracion HTTP
- `UI`: demo local sobre la API

Documentacion tecnica complementaria:

- [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)
- [docs/PROCESS_HISTORY.md](docs/PROCESS_HISTORY.md)
- [docs/TESTING.md](docs/TESTING.md)

## Dataset, samples y reproducibilidad

### Dataset de reentrenamiento

La estructura esperada del dataset se documenta en:

- [full_dataset/hackathon_TREE_AIBiomed/Readme.md](full_dataset/hackathon_TREE_AIBiomed/Readme.md)

Se mantienen en git unicamente:

- `full_dataset/hackathon_TREE_AIBiomed/data.yaml`
- `full_dataset/hackathon_TREE_AIBiomed/Readme.md`

No se versionan:

- `full_dataset/hackathon_TREE_AIBiomed/images/`
- `full_dataset/hackathon_TREE_AIBiomed/labels/`

Motivo:

- esas rutas pueden contener imagenes medicas y anotaciones con riesgo de PII

### Muestras locales para inferencia

La carpeta `samples/` es **local** y no se versiona.

Si no existe, puede crearse manualmente. Las imagenes que se quieran probar con `smoke` deben colocarse alli. Formatos admitidos por la CLI:

- `.png`
- `.jpg`
- `.jpeg`
- `.bmp`
- `.tif`
- `.tiff`

Importante:

- cualquier archivo anadido a `samples/` se considera potencialmente sensible
- `samples/` esta en `.gitignore`
- no debe usarse para compartir imagenes del reto ni evidencia publica

### Nota importante sobre naming del dataset

El dataset fuente puede contener pares como:

- imagen: `algo_annotated.png`
- label: `algo.txt`

Ultralytics espera por defecto:

- `algo.png` + `algo.txt`

o bien:

- `algo_annotated.png` + `algo_annotated.txt`

Para no modificar el dataset original, este repo crea un staging temporal compatible en `outputs/prepared_datasets/` antes de entrenar.

### Flujo de reentrenamiento

```powershell
python -m app.cli train
```

Este comando:

- toma `full_dataset/hackathon_TREE_AIBiomed`
- prepara un staging temporal compatible
- entrena con la receta `historical`
- deja el mejor checkpoint activo en `models/best.pt`

## API y superficies de uso

Endpoints principales:

- `GET /health`
- `POST /deidentify`
- `POST /deidentify/report`

Resumen:

- `/health`: estado del servicio y presencia de modelos
- `/deidentify`: devuelve imagen redactada y headers con resumen
- `/deidentify/report`: devuelve solo el reporte JSON

## Limitaciones

- La solucion esta ajustada al reto Treelogic y a la estructura del dataset disponible.
- El rendimiento puede degradarse con layouts, resoluciones o estilos de anotacion muy distintos al dataset de entrenamiento.
- La clase `time` aun depende de validacion OCR y puede caer en `review` si el texto es ambiguo.
- La calidad del OCR influye directamente en la granularidad de la mascara fina y en algunos motivos de decision.
- El sistema es una solucion experimental de hackathon, no un producto clinico validado.

## Consideraciones eticas y de privacidad

- Este proyecto aborda un problema real de privacidad en imagen medica, pero no sustituye procesos formales de gobierno del dato.
- El repositorio compartible debe evitar incluir imagenes, labels, samples, reports o artifacts que puedan exponer PII.
- Los artifacts de inferencia y entrenamiento deben tratarse como material local sensible.
- Cualquier uso posterior sobre datos reales debe respetar la normativa aplicable de proteccion de datos y los acuerdos de uso del dataset.

## Licencias y recursos de terceros

Dependencias principales del proyecto:

- Ultralytics YOLO
- Tesseract OCR
- OpenCV
- scikit-image
- FastAPI

Consideraciones:

- el codigo de este repositorio es propio del equipo salvo dependencias de terceros instaladas por licencia
- cualquier uso de modelos, librerias, datasets o recursos externos debe respetar sus licencias correspondientes
- el dataset del reto debe tratarse como recurso sensible y su uso debe limitarse al marco autorizado del hackathon

## Uso de IA como apoyo durante el desarrollo

Se utilizaron herramientas de IA como apoyo tecnico durante el desarrollo y la documentacion del proyecto:

- Codex
- Claude Code

Estas herramientas se usaron como asistencia para exploracion tecnica, refactorizacion, documentacion y validacion operativa, mientras que el diseno, decisiones y cierre final del proyecto fueron responsabilidad del equipo autor.

## Diagnostico rapido

### `data.yaml` con error de sintaxis

Revisar que `full_dataset/hackathon_TREE_AIBiomed/data.yaml` sea YAML puro y no contenga fences Markdown.

### YOLO detecta clases ajenas al reto

Si aparecen clases como `bed`, `person` o `class_59`, probablemente `models/best.pt` no corresponde al detector entrenado para este reto.

Clases esperadas:

- `0`: `name`
- `1`: `id`
- `2`: `age`
- `3`: `date`
- `4`: `time`

### Falta Tesseract

En Windows:

```powershell
winget install UB-Mannheim.TesseractOCR
```

Si no queda en `PATH`:

```powershell
$env:TESSERACT_CMD="C:\Program Files\Tesseract-OCR\tesseract.exe"
```
