# Taller 1 – Redes Neuronales Recurrentes y Convolucionales

Pontificia Universidad Javeriana – Maestría en Inteligencia Artificial – Aprendizaje Profundo (202630)
Profesor: Andrés Moreno Barbosa

## Integrantes

- Daniel Sebastian Velasco Munar – Punto 1 (RNN sobre MeteoNet)
- Andrea Barraza – Punto 2 (CNN sobre Fashion-MNIST)

## Resultados principales

**Punto 1.** Pronóstico de las próximas 24 h de temperatura en la estación 62548002 de MeteoNet (Calais, costa
norte de Francia) a partir de las 168 h anteriores de 12 variables. El modelo final es una SimpleRNN de 32
unidades (2 232 parámetros), elegida tras una búsqueda por etapas de 18 corridas. En test (febrero–diciembre
de 2018, el 30 % final de la serie) obtiene un MAE de **1.602 °C**, un 24.7 % menos que la mejor línea base
(«ayer», 2.128 °C).

**Punto 2.** Clasificación de Fashion-MNIST con tres aproximaciones entrenadas con la misma partición,
semilla, lotes y callbacks:

| Modelo | Accuracy en test | IC 95 % | Parámetros | Tiempo en T4 |
|---|---|---|---|---|
| VGG16 (filtros de ImageNet + fine-tuning) | 0.9319 | 0.9269 – 0.9367 | 14.7 M | 2 687 s |
| MobileNetV2 (transfer learning + fine-tuning) | 0.9243 | 0.9190 – 0.9295 | 2.3 M | 340 s |
| CNN desde cero | 0.8904 | 0.8846 – 0.8963 | 28.7 k | 79 s |

Los intervalos de VGG16 y MobileNetV2 se solapan y MobileNetV2 entrena ocho veces más rápido, así que es la
opción más equilibrada. El análisis completo está en el informe.

## Estructura de la entrega

```
code/
  punto1_rnn_meteonet/
    00_descarga_y_filtrado_estacion.ipynb  Descarga los datos crudos de MeteoNet (zona NW, 2016-2018),
                                           resume la calidad de las 287 estaciones, filtra la estación
                                           escogida y genera el CSV horario que usan los demás notebooks.
    01_eda_y_modelo_rnn.ipynb              Todo el desarrollo: exploración de la serie, definición de la tarea
                                           (entrada 168 h -> salida 24 h), imputación y partición cronológica,
                                           líneas base, búsqueda de hiperparámetros (SimpleRNN/LSTM/GRU) y
                                           evaluación en test con análisis de errores. Requiere GPU.
    02_figuras_informe.ipynb               Figuras del informe (fig_p1_*.png) a partir de la serie y de
                                           results/punto1. No entrena nada: corre en CPU en segundos.
    data/estacion_62548002_horaria.csv     Serie horaria de la estación escogida (sale del notebook 00).
    data/estacion_62548002_metadatos.json  Ubicación, cobertura y columnas de la estación.
    data/estacion_62548002_preparada.csv   Serie imputada + variables de calendario (sale del notebook 01).
    data/config_tarea.json                 Cortes de la partición, variables y escalado. Guarda la ventana
                                           inicial de 72 h; la final (168 h) está en metricas_test.json.
  punto2_cnn_fashion_mnist/
    01_cnn_fashion_mnist.ipynb             Todo el desarrollo: preparación de datos, CNN desde cero, VGG16 con
                                           filtros de ImageNet y MobileNetV2, las tres con la misma partición,
                                           semilla, lotes, callbacks y métricas. Búsqueda de hiperparámetros,
                                           fine-tuning, evaluación en test y análisis de errores. Requiere GPU.
    02_figuras_informe.ipynb               Figuras y tablas del informe a partir de results/punto2. No entrena
                                           nada: corre en CPU en segundos.
    03_bono_redes_modernas.ipynb           Bono: EfficientNetV2-B0 en las mismas condiciones que las otras tres
                                           redes, comparación pareada (bootstrap y McNemar) contra la mejor, y
                                           sus figuras y tablas. Requiere GPU para entrenar.
    README.md                              Detalle del diseño del Punto 2 y de los archivos que genera.
results/
  punto1/                                  Resumen de estaciones, figuras exploratorias (eda_*.png) y de
                                           modelado (rnn_*.png), figuras del informe (fig_p1_*.png), tabla de
                                           experimentos, curvas, métricas de test y modelos finales (24 h y 1 h).
  punto2/                                  Tabla de experimentos, métricas de test, tiempos, bootstrap,
                                           predicciones de test, curvas crudas (JSON), figuras del informe
                                           (fig_*.png), tablas (csv/tex/md) y el modelo de la CNN desde cero.
  punto2/bono/                             Resultados del bono: búsqueda, curvas, métricas, predicciones de
                                           test, figuras (fig_bono_*.png) y tablas (tabla_bono_*).
report/
  Informe_Taller1_IEEE.docx                Informe escrito en formato IEEE Transactions on AI (Puntos 1 y 2).
requirements.txt                           Librerías utilizadas.
```

## Cómo ejecutar

Los notebooks de entrenamiento se corren en Google Colab con GPU:

1. `Archivo > Abrir notebook > GitHub` y pegar la URL de este repositorio.
2. `Entorno de ejecución > Cambiar tipo de entorno > GPU (T4)`.
3. Los notebooks clonan el repositorio y guardan sus resultados en `results/`. Para conservar el notebook
   ejecutado: `Archivo > Guardar una copia en GitHub`; los archivos de `results/` se descargan desde Colab y
   se suben al repositorio.

Orden: en el Punto 1, `00` → `01` → `02`; en el Punto 2, `01` → `02` → `03`. Los notebooks `02` de cada punto no
necesitan GPU y también corren en local: solo leen lo que dejaron los anteriores, así que una figura se puede
rehacer sin volver a entrenar.

Los datos crudos de MeteoNet (~520 MB comprimidos) **no** están en el repositorio: el notebook `00` los
descarga desde <https://meteonet.umr-cnrm.fr/dataset/data/> y deja únicamente el CSV horario de la estación.
Fashion-MNIST se descarga automáticamente con `tf.keras.datasets.fashion_mnist`. Los modelos de VGG16 y
MobileNetV2 no se versionan por su tamaño.

## Versiones utilizadas

- TensorFlow 2.20.0 en Google Colab con GPU T4
- Punto 1: semilla 42, lotes de 128, Adam (lr 1e-3), EarlyStopping (paciencia 8) y ReduceLROnPlateau
- Punto 2: semilla 42, lotes de 128, entrada de las redes preentrenadas a 96×96, precisión float32
