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

Cada imagen tiene un fichero `.txt` asociado con el mismo nombre base.

Ejemplo:

```text
images/train/ed9c0dfc-ea25b576-0f8cc069-df4cdf14-0cd60eb7_annotated.png
labels/train/ed9c0dfc-ea25b576-0f8cc069-df4cdf14-0cd60eb7.txt
```

## Division del dataset

El dataset contiene aproximadamente:

- 400 imagenes en total
- 80% para entrenamiento
- 20% para validacion

Distribucion esperada:

- Train: ~320 imagenes
- Validation: ~80 imagenes

## Uso previsto

Este dataset esta disenado para:

- desidentificacion automatica de radiografias
- deteccion de informacion sensible en imagenes medicas
- entrenamiento de modelos
- investigacion en privacidad y anonimizacion medica

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

## Compatibilidad con el repo integrado

En este dataset, varias imagenes usan nombres como `*_annotated.png` mientras que los labels originales usan el nombre base sin ese sufijo.

Para no modificar el dataset fuente, el repo integrado prepara un staging temporal dentro de `outputs/prepared_datasets/` antes de entrenar. En ese staging, los labels se copian con el mismo stem que la imagen para que Ultralytics pueda matchearlos correctamente.

## Consideraciones

- Las coordenadas estan normalizadas segun el estandar YOLO.
- Las imagenes contienen informacion sensible delimitada mediante bounding boxes.
- El dataset esta orientado a tareas de deteccion de objetos y anonimizacion automatica.
- El conjunto mezcla imagenes reales y datos sinteticos para conservar privacidad y anonimato de pacientes.
