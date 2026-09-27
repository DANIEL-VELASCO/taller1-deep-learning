TALLER 1 - REDES NEURONALES RECURRENTES Y CONVOLUCIONALES
Aprendizaje Profundo (202630) - Maestría en Inteligencia Artificial
Pontificia Universidad Javeriana
Profesor: Andrés Moreno

INTEGRANTES
  - Daniel Sebastian Velasco Munar
  - Andrea Barraza

Repositorio: https://github.com/DANIEL-VELASCO/taller1-deep-learning


CONTENIDO DE LA ENTREGA
=======================

README.txt
    Este archivo: integrantes y descripción de cada archivo de la entrega.

requirements.txt
    Librerías de Python utilizadas.

report/
    Informe_Taller1_IEEE.docx   Informe en formato IEEE Transactions on Artificial Intelligence
                                (Puntos 1 y 2, con el bono).
    Informe_Taller1_IEEE.pdf    El mismo informe en PDF. Las menciones a figuras y tablas son
                                enlaces que llevan a cada una.


code/  -  CÓDIGO
================

Los notebooks de entrenamiento se ejecutaron en Google Colab con GPU T4 y TensorFlow 2.20. Los
notebooks de figuras (02) no entrenan nada: leen lo que dejaron los anteriores en results/ y corren en
CPU en segundos. Todos los notebooks se entregan ejecutados, con sus salidas.

code/punto1_rnn_meteonet/  (Punto 1: pronóstico de temperatura con redes recurrentes)
    00_descarga_y_filtrado_estacion.ipynb
        Descarga las observaciones de MeteoNet (zona noroeste, 2016-2018), resume la calidad de las
        estaciones, justifica la elección de la estación 62548002 y genera su serie horaria.
    01_eda_y_modelo_rnn.ipynb
        Desarrollo completo del Punto 1: análisis exploratorio, formulación de la tarea (168 h de entrada
        y 24 h de salida), imputación, partición cronológica 59.5 / 10.5 / 30 % sin fuga de información,
        líneas base, búsqueda de hiperparámetros por etapas (SimpleRNN, LSTM y GRU), evaluación en el 30 %
        final y análisis de errores. Requiere GPU.
    02_figuras_informe.ipynb
        Figuras del Punto 1 que van en el informe (fig_p1_*.png), a partir de los datos y de results/.
    data/estacion_62548002_horaria.csv
        Serie horaria de la estación elegida (sale del notebook 00).
    data/estacion_62548002_metadatos.json
        Ubicación, cobertura y columnas de la estación.
    data/estacion_62548002_preparada.csv
        Serie imputada con las variables de calendario (sale del notebook 01).
    data/config_tarea.json
        Cortes de la partición, variables de entrada y parámetros de estandarización.

code/punto2_cnn_fashion_mnist/  (Punto 2: clasificación de Fashion-MNIST con redes convolucionales)
    01_cnn_fashion_mnist.ipynb
        Desarrollo completo del Punto 2: preparación de datos, CNN desde cero, VGG16 con filtros de
        ImageNet y MobileNetV2 con aprendizaje por transferencia, las tres con la misma partición,
        semilla, lotes, callbacks y métricas. Búsqueda de hiperparámetros, ajuste fino, evaluación única
        en test, análisis de errores e intervalos de confianza. Requiere GPU.
    02_figuras_informe.ipynb
        Figuras y tablas del Punto 2 que van en el informe, a partir de results/punto2.
    03_bono_redes_modernas.ipynb
        Bono: EfficientNetV2-B0 entrenada en las mismas condiciones que las otras tres redes, con
        comparación pareada (bootstrap pareado y prueba de McNemar) contra la mejor. Entrena y genera sus
        propias figuras y tablas en results/punto2/bono. Requiere GPU para entrenar.
    README.md
        Detalle del diseño experimental del Punto 2 y de los archivos que genera.


results/  -  RESULTADOS GENERADOS POR LOS NOTEBOOKS
===================================================

results/punto1/
    resumen_estaciones_NW.csv          Calidad y cobertura de las estaciones candidatas (notebook 00).
    mapa_estaciones_NW.png             Mapa de las estaciones de la zona noroeste, coloreadas por el
                                       porcentaje de datos faltantes de temperatura, con las
                                       candidatas marcadas.
    serie_horaria_estacion_62548002.png  Serie horaria completa de la estación elegida.
    eda_*.png                          Figuras del análisis exploratorio: serie completa, zoom de dos
                                       semanas, distribuciones, correlaciones, ciclos anual y diario,
                                       autocorrelación, datos faltantes de temperatura, partición y
                                       dificultad según el horizonte.
    experimentos_rnn.csv               Una fila por corrida de la búsqueda de hiperparámetros.
    rnn_busqueda_hiperparametros.png   Resultado de la búsqueda por etapas.
    rnn_etapa1_curvas.png              Curvas de entrenamiento de la comparación de celdas.
    historias/*.json                   Curvas de la verificación final con GRU y LSTM.
    metricas_test.json                 Métricas del modelo final y de las líneas base en test.
    rnn_test_*.png                     Análisis en test: error por horizonte, por mes y hora,
                                       distribución del error y ejemplos de pronóstico.
    fig_p1_*.png                       Figuras del informe (notebook 02).
    modelo_rnn_final.keras             Modelo final (SimpleRNN, 24 h de horizonte).
    modelo_rnn_horizonte_1h.keras      Misma arquitectura entrenada a 1 h, usada como referencia.

results/punto2/
    experimentos.csv                   Una fila por corrida de la búsqueda de hiperparámetros.
    historias_busqueda.json            Curvas de entrenamiento de cada corrida de la búsqueda.
    historias_final.json               Curvas del entrenamiento final y del ajuste fino.
    metricas_test.csv                  Accuracy, F1, precisión, exhaustividad y log-loss en test.
    reporte_cnn.csv, reporte_vgg16.csv, reporte_mobilenetv2.csv
                                       Métricas por clase de cada modelo.
    tiempos.csv                        Segundos de entrenamiento por modelo y por etapa.
    bootstrap.csv                      Intervalo de confianza del 95 % del accuracy.
    resumen_final.csv                  Tabla consolidada de los tres modelos.
    entorno.json                       Versión de TensorFlow, GPU, semilla, presupuesto e
                                       hiperparámetros elegidos.
    predicciones_test.npz              Etiquetas reales y probabilidades de los tres modelos en test.
    modelo_cnn_desde_cero.keras        Modelo final de la CNN desde cero.
    busqueda.png, curvas.png, confusion.png, errores.png, ejemplos_clases.png
                                       Figuras generadas durante el entrenamiento (notebook 01).
    fig_*.png                          Figuras del informe (notebook 02): clases, búsqueda, curvas,
                                       matrices de confusión, F1 por clase, comparación y errores.
    tabla_*.csv / .tex / .md           Tablas del informe: comparativa, búsqueda y F1 por clase.

results/punto2/bono/  (notebook 03)
    experimentos_bono.csv, historias_busqueda_bono.json
                                       Búsqueda de hiperparámetros de EfficientNetV2-B0.
    historias_final_bono.json          Curvas del entrenamiento final y del ajuste fino.
    metricas_test_bono.csv             Métricas en test.
    reporte_efficientnetv2b0.csv       Métricas por clase.
    tiempos_bono.csv                   Segundos de entrenamiento por etapa.
    entorno_bono.json                  GPU, presupuesto, precisión y configuración elegida.
    predicciones_efficientnetv2b0.npz  Probabilidades en test.
    fig_bono_*.png                     Figuras: búsqueda, curvas, matriz de confusión, F1 por clase y
                                       comparación de las cuatro redes con la prueba pareada.
    tabla_bono_*.csv / .tex / .md      Tablas: comparativa, prueba pareada, búsqueda y F1 por clase.


CÓMO EJECUTAR
=============

1. Abrir el notebook en Google Colab (Archivo > Abrir notebook > GitHub, con la URL del repositorio).
2. Entorno de ejecución > Cambiar tipo de entorno > GPU (T4).
3. Ejecutar todo. Los notebooks clonan el repositorio y escriben sus resultados en results/.

Orden: Punto 1: 00 -> 01 -> 02.  Punto 2: 01 -> 02 -> 03.

Los datos crudos de MeteoNet (unos 520 MB comprimidos) no se incluyen: el notebook 00 los descarga de
https://meteonet.umr-cnrm.fr/dataset/data/. Fashion-MNIST se descarga automáticamente con
tf.keras.datasets.fashion_mnist. Los modelos de VGG16, MobileNetV2 y EfficientNetV2-B0 no se incluyen
por su tamaño; sus métricas y predicciones sí están en results/.
