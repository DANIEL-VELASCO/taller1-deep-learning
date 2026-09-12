# Taller 1 – Redes Neuronales Recurrentes y Convolucionales

Pontificia Universidad Javeriana – Maestría en Inteligencia Artificial – Aprendizaje Profundo (202630)

## Integrantes

- Daniel Sebastian Velasco Munar – Punto 1 (RNN sobre MeteoNet)
- [NOMBRE COMPAÑERA] – Punto 2 (CNN sobre Fashion-MNIST)

## Estructura de la entrega

```
code/
  punto1_rnn_meteonet/
    00_descarga_y_filtrado_estacion.ipynb  Descarga los datos crudos de MeteoNet (zona NW, 2016-2018),
                                           resume la calidad de todas las estaciones, filtra la estación
                                           escogida y genera el CSV horario que usan los demás notebooks.
    01_eda_y_seleccion_estacion.ipynb      Análisis exploratorio de la estación y justificación de la selección.
    02_modelado_rnn.ipynb                  Preparación de secuencias, experimentos con RNN/LSTM/GRU,
                                           búsqueda de hiperparámetros y evaluación en test (último 30%).
    data/estacion_XXXXX_horaria.csv        Serie horaria de la estación escogida (archivo pequeño, versionado).
  punto2_cnn_fashion_mnist/
    01_cnn_desde_cero.ipynb                CNN diseñada desde cero.
    02_cnn_filtros_red_grande.ipynb        CNN con filtros iniciales de una red grande preentrenada (p. ej. VGG16).
    03_mobilenetv2_transfer.ipynb          Transfer learning con MobileNetV2.
    04_bono_cnn_moderna.ipynb              Bono: extensión a una CNN moderna (EfficientNetV2 / ConvNeXt).
results/
  punto1/                                  Tablas de métricas (CSV) y figuras del Punto 1.
  punto2/                                  Tablas de métricas (CSV) y figuras del Punto 2.
report/
  informe.pdf                              Reporte escrito en formato IEEE Transactions on AI.
requirements.txt                           Librerías utilizadas.
```

## Cómo ejecutar (Google Colab con GPU)

1. `Archivo > Abrir notebook > GitHub` y pegar la URL de este repositorio.
2. `Entorno de ejecución > Cambiar tipo de entorno > GPU (T4)`.
3. Los notebooks clonan el repositorio y guardan sus resultados en `results/`. Para persistir los cambios:
   `Archivo > Guardar una copia en GitHub` (notebook) y commit de los archivos de `results/` desde Colab.

Los datos crudos de MeteoNet (~520 MB comprimidos) **no** están en el repositorio: el notebook `00` los
descarga desde <https://meteonet.umr-cnrm.fr/dataset/data/> y deja únicamente el CSV horario de la estación.
Fashion-MNIST se descarga automáticamente con `tf.keras.datasets.fashion_mnist`.

## Versiones utilizadas

- Python: (anotar)
- TensorFlow: (anotar)
- Entorno: Google Colab, GPU T4
