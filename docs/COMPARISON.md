# Comparacion consolidada de enfoques

## OCR-first

Fortalezas:

- OCR mas robusto
- mejor debug
- menos dependencia de una deteccion previa perfecta

Debilidades:

- menos orientado a producto end-to-end
- mas experimental
- sin API/UI integradas como superficie principal

## YOLO-first

Fortalezas:

- flujo mas cercano a producto
- redaccion final ya pensada como feature
- API y frontend disponibles

Debilidades:

- OCR mas simple
- mayor dependencia del detector
- paths hardcodeados y duplicacion en el repo de origen

## Sintesis

La mejor solucion no era elegir una sola, sino combinar:

- YOLO para localizar
- OCR robusto para validar y leer
- una politica de decision unificada
- una misma superficie de uso para entrenamiento, inferencia y demo

