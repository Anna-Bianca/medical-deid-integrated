# Historia del proceso tecnico

## Problema

El objetivo del proyecto es eliminar o anonimizar informacion identificable incrustada visualmente en imagenes medicas.

## Camino 1: OCR-first

En `Anonimization_of_medical_images` se exploro primero un enfoque orientado a OCR sobre imagen completa y tambien una variante restringida por bounding boxes.

Lo mas valioso que dejo ese camino fue:

- multiples pasadas OCR
- preprocesamientos alternativos
- filtros por confianza
- repeticion entre pasadas
- overlap IoU
- deduplicacion
- trazabilidad detallada

Ese trabajo mostro que el OCR full-image podia leer mejor que una caja previa imperfecta.

## Camino 2: YOLO-first

En `anonymization-challenge` se construyo una solucion mas cercana a producto:

- entrenamiento YOLO
- inferencia
- redaccion
- API
- frontend web

Ese camino resolvio mejor la operacion end-to-end, pero usaba un OCR bastante mas simple.

## Decision de integracion

La arquitectura final de este folder combina ambos aprendizajes:

- la carcasa operativa viene del enfoque YOLO-first
- el motor OCR fuerte viene del enfoque OCR-first
- la receta historica de entrenamiento fue reintegrada dentro del repo actual

## Como se resolvio la reproducibilidad

En lugar de depender de un repo externo o de paths absolutos, el proceso de entrenamiento ahora asume un dataset local canonico:

```text
full_dataset/hackathon_TREE_AIBiomed
```

Ese folder queda dentro del repo de trabajo, pero excluido del repositorio publico via `.gitignore`.

De esa manera:

- el flujo es reproducible para quien tiene los datos
- no hace falta llamar imagenes desde otro repo
- no se publican imagenes medicas ni labels completos

## Estado honesto de este workspace

En esta integracion:

- el dataset completo local vive en `full_dataset/`
- `samples/` se usa para pruebas manuales fuera de los splits oficiales
- `models/best.pt` debe generarse con `python -m app.cli train` o copiarse manualmente si ya existe una version confiable

Si `models/best.pt` no proviene de este flujo, el pipeline puede cargar un modelo COCO u otro detector incorrecto y producir clases no medicas como `bed`.
