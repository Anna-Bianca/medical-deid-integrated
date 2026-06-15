<!-- file: full_dataset/hackathon_TREE_AIBiomed/Readme.md -->
<!-- description: Documents the structure and annotation format of the medical radiography de-identification dataset. -->
<!-- author: Maria Victoria Anconetani; Anna Bianca Marzetti Biggi -->
<!-- date: 15/06/2026 -->

# Dataset de Desidentificacion de Radiografias

## Descripcion

Este dataset fue preparado para tareas de desidentificacion automatica de imagenes medicas, concretamente radiografias en formato PNG.

El objetivo principal es detectar informacion sensible incrustada en las imagenes mediante deteccion de objetos en formato YOLO.

## Tipo de imagenes

- Radiografias medicas
- Formato de imagen: `.png`

## Formato de anotaciones

Las anotaciones siguen el formato estandar YOLO:

```text
<class_id> <x_center> <y_center> <width> <height>
```

Donde:

- `class_id`: identificador numerico de la clase
- `x_center`: coordenada X del centro de la caja, normalizada entre `0` y `1`
- `y_center`: coordenada Y del centro de la caja, normalizada entre `0` y `1`
- `width`: ancho de la caja, normalizado entre `0` y `1`
- `height`: alto de la caja, normalizado entre `0` y `1`

Ejemplo:

```text
0 0.512 0.183 0.245 0.052
```

## Clases

- `0`: `name`
- `1`: `id`
- `2`: `age`
- `3`: `date`
- `4`: `time`

## Estructura del dataset

```text
full_dataset/hackathon_TREE_AIBiomed/
  images/
    train/
    val/
  labels/
    train/
    val/
  Readme.md
  data.yaml
```

- `images/`: radiografias en formato PNG
- `labels/`: anotaciones YOLO `.txt` correspondientes a cada imagen

## Como poblar el dataset para reentrenar

Para reentrenar el modelo se deben crear localmente estas carpetas:

- `images/train/`
- `images/val/`
- `labels/train/`
- `labels/val/`

Reglas practicas:

- cada imagen en `images/train/` debe tener un label correspondiente en `labels/train/`
- cada imagen en `images/val/` debe tener un label correspondiente en `labels/val/`
- las anotaciones deben seguir el formato YOLO documentado arriba
- `data.yaml` debe permanecer apuntando a `images/train` y `images/val`

## Naming esperado por el repo

Cada imagen debe tener un `.txt` asociado con el mismo nombre base o un nombre compatible con el staging del repo.

El caso mas simple es:

- `algo.png` + `algo.txt`

Tambien se tolera el caso historico del reto:

- imagen: `algo_annotated.png`
- label: `algo.txt`

Como Ultralytics espera matching por stem, el proyecto crea automaticamente un staging temporal en `outputs/prepared_datasets/` antes de entrenar.

Si se arma un dataset nuevo desde cero, la opcion mas limpia es mantener el mismo stem entre imagen y label:

- `algo.png` + `algo.txt`
- o `algo_annotated.png` + `algo_annotated.txt`

## Division del dataset

El dataset del reto contiene aproximadamente:

- 400 imagenes en total
- 80% para entrenamiento
- 20% para validacion

Distribucion esperada:

- Train: ~320 imagenes
- Validation: ~80 imagenes

## Nota importante sobre `data.yaml`

El archivo `data.yaml` debe guardarse como YAML puro, sin bloques Markdown.

Correcto:

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

Incorrecto:

```text
```yaml
train: images/train
...
```
```

Si el archivo arranca con backticks, Ultralytics falla con un error de sintaxis YAML en la linea 1, columna 1.

## Uso de `samples/` para inferencia local

La carpeta `samples/` no forma parte del dataset de entrenamiento. Se usa solo para pruebas manuales de inferencia con `python -m app.cli smoke`.

Si se quieren probar imagenes manualmente:

1. crear una carpeta local `samples/` en la raiz del repo si no existe
2. copiar alli imagenes de prueba en formatos admitidos (`.png`, `.jpg`, `.jpeg`, `.bmp`, `.tif`, `.tiff`)
3. ejecutar `python -m app.cli smoke`

Importante:

- `samples/` es local y esta en `.gitignore`
- no debe subirse a git
- puede contener material con riesgo de PII

## Compatibilidad con el repo integrado

Para no modificar el dataset fuente, el repo integrado prepara un staging temporal dentro de `outputs/prepared_datasets/` antes de entrenar. En ese staging, los labels se copian con el mismo stem que la imagen para que Ultralytics pueda matchearlos correctamente.

## Consideraciones

- Las coordenadas estan normalizadas segun el estandar YOLO.
- Las imagenes contienen informacion sensible delimitada mediante bounding boxes.
- El dataset esta orientado a tareas de deteccion de objetos y anonimizacion automatica.
- El conjunto mezcla imagenes reales y datos sinteticos para conservar privacidad y anonimato de pacientes.
- En la version compartible del repositorio no se incluyen `images/` ni `labels/`; solo se mantienen `data.yaml` y este `Readme.md` para documentar la estructura esperada.
