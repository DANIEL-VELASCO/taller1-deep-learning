# Punto 2 – CNN sobre Fashion-MNIST

Desarrollo a cargo de **Andrea Barraza**.

Las tres aproximaciones están en **un solo notebook**, `01_cnn_fashion_mnist.ipynb`. La razón es la
rúbrica: el 15 % del punto se juega en la *comparación justa bajo condiciones documentadas*, y con un
único notebook las tres familias comparten partición, semilla, lotes, callbacks y métricas dentro de
la misma corrida, sin depender de que tres archivos distintos hayan quedado sincronizados.

| Sección del notebook | Contenido | Rúbrica |
|---|---|---|
| 1. Datos y diseño experimental | Carga, revisión de integridad, partición estratificada 54 000 / 6 000 / 10 000 y pipeline `tf.data` | 5 % |
| 2-4. CNN desde cero | Arquitectura propia justificada, búsqueda de hiperparámetros y entrenamiento final | 5 % |
| 2-4. VGG16 | `include_top=False`, `weights='imagenet'`, congelada y luego fine-tuning de las últimas 4 capas | 10 % |
| 2-4. MobileNetV2 | Transfer learning, congelada y luego fine-tuning de las últimas 20 capas | 10 % |
| 5-7. Evaluación y análisis | Test (una sola vez), matrices de confusión, errores, bootstrap y comparación | 15 % |
| Pendiente | Bono: EfficientNetV2 / ConvNeXt / ResNeXt | Bono 10 % |

## Cómo está resuelta la comparación justa

- **Mismo split**: `fashion_mnist.load_data()` da 60 000 de desarrollo y 10 000 de test. El 10 % del
  desarrollo se reserva como validación, estratificado y con `SEED = 42`. El test se toca una sola
  vez, en la sección 5.
- **Mismos lotes**: las tres familias consumen los mismos objetos `tf.data`, con el mismo barajado
  sembrado, así que ven las imágenes en el mismo orden.
- **Mismas métricas**: accuracy, F1 macro, precisión y recall macro, log-loss, matriz de confusión,
  curvas de loss/accuracy, tiempo de entrenamiento y número de parámetros (totales y entrenables).
  Además, intervalo de confianza del accuracy por bootstrap sobre test.
- **Mismos callbacks**: `EarlyStopping(patience=3, restore_best_weights=True)` y
  `ReduceLROnPlateau(factor=0.3, patience=2)`, con Adam como optimizador base.
- **Entrada única**: todos los modelos reciben la imagen original de 28×28×1. El redimensionamiento a
  96×96, la replicación del canal gris a RGB y el preprocesamiento de ImageNet ocurren *dentro* de
  cada modelo, así que no conviven dos formatos de datos en el pipeline.
- **Diferencia que hay que declarar en el informe**: VGG16 y MobileNetV2 tienen una fase extra de
  fine-tuning que la CNN desde cero no tiene. Es parte de la aproximación, pero desbalancea el
  presupuesto de cómputo y por eso `tiempos.csv` separa la etapa `base` de la etapa `fine_tune_*`.

## Cómo ejecutarlo

1. Abrir `01_cnn_fashion_mnist.ipynb` en Colab desde GitHub.
2. *Entorno de ejecución > Cambiar tipo de entorno > **GPU (T4)***. Sin GPU el notebook avisa y no
   vale la pena seguir.
3. Verificar que `MODO_RAPIDO = False` en la primera celda de código. En `True` la búsqueda usa solo
   12 000 imágenes y 4 épocas: sirve para depurar en minutos, pero esos números **no** van al informe.
4. *Entorno de ejecución > Ejecutar todo*. Con `MODO_RAPIDO = False` toma entre 1,5 y 2,5 horas en
   una T4; VGG16 se lleva la mayor parte.

El notebook clona el repositorio y va escribiendo en `results/punto2/` a medida que avanza, así que
una desconexión a mitad de camino no obliga a empezar de cero.

## Qué queda en `results/punto2/`

| Archivo | Contenido |
|---|---|
| `experimentos.csv` | Una fila por corrida de la búsqueda: familia, configuración, mejor época, val_loss, val_accuracy, brecha train-val, segundos y parámetros |
| `metricas_test.csv` | Accuracy, F1 macro, precisión/recall macro, log-loss y parámetros de los tres modelos |
| `reporte_<familia>.csv` | `classification_report` por clase de cada modelo |
| `tiempos.csv` | Segundos por familia y por etapa (`base` / `fine_tune_*`) |
| `bootstrap.csv` | Accuracy con intervalo de confianza del 95 % |
| `resumen_final.csv` | Tabla consolidada para el informe |
| `entorno.json` | Versión de TensorFlow, GPU, semilla, hiperparámetros elegidos y conteo de duplicados |
| `historias_busqueda.json`, `historias_final.json` | Curvas de entrenamiento crudas, por si hay que rehacer alguna figura sin reentrenar |
| `*.png` | Ejemplos por clase, resultados de la búsqueda, curvas, matrices de confusión y errores |
| `modelo_cnn_desde_cero.keras` | Único modelo que se versiona (~150 KB) |

Los modelos de VGG16 (~56 MB) y MobileNetV2 (~9 MB) y todos los checkpoints se quedan en `/content`,
fuera del repositorio: GitHub avisa a partir de 50 MB por archivo y no aportan nada al informe que no
esté ya en los CSV.
