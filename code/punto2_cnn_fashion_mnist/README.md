# Punto 2 – CNN sobre Fashion-MNIST

Desarrollo a cargo de **Andrea Barraza**.

Tres notebooks: `01_cnn_fashion_mnist.ipynb` entrena y evalúa, `02_figuras_informe.ipynb` arma las
figuras y las tablas a partir de lo que el primero dejó en `results/punto2/`, y
`03_bono_redes_modernas.ipynb` es el bono con EfficientNetV2-B0. Separar entrenamiento y figuras deja
ajustar una gráfica sin volver a pasar por la GPU.

Las tres aproximaciones están en **un solo notebook de entrenamiento**. La razón es la
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
| Notebook `03` | Bono: EfficientNetV2-B0 en las mismas condiciones, comparación pareada contra la mejor red | Bono 10 % |

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
3. Elegir el presupuesto de cómputo en la constante `PRESUPUESTO` de la primera celda de código:

   | Valor | Búsqueda | Entrenamiento final | Duración en T4 |
   |---|---|---|---|
   | `"completo"` | las 54 000 imágenes, 12 épocas | 30 épocas + 15 de fine-tuning | ~2 h |
   | `"acotado"` | 15 000 estratificadas, 8 épocas | 20 épocas + 8 de fine-tuning | ~1 h |
   | `"depuracion"` | 12 000, 4 épocas | 8 épocas + 3 | ~10 min |

   Buscar hiperparámetros sobre una submuestra estratificada y entrenar el modelo final con las
   54 000 es práctica normal; lo que importa es declararlo, y el notebook lo deja escrito en
   `entorno.json`. `"depuracion"` solo sirve para comprobar que el notebook corre de punta a punta:
   esos números **no** van al informe.
4. *Entorno de ejecución > Ejecutar todo*. VGG16 se lleva la mayor parte del tiempo en cualquiera de
   los tres presupuestos.
5. Al terminar, correr `02_figuras_informe.ipynb`. No necesita GPU y tarda segundos; se puede repetir
   las veces que haga falta para ajustar una figura.

Los tiempos de la tabla son el peor caso, con todas las épocas corridas; en la práctica
`EarlyStopping(patience=3)` corta antes. El notebook intenta activar **precisión mixta** cuando detecta
GPU (`USAR_MIXED_PRECISION`), pero Keras 3 vuelve a la política por defecto en cada
`tf.keras.backend.clear_session()`, y el notebook la llama antes de construir cada modelo. Por eso la
corrida quedó en **float32**, como registra `entorno.json`, y los tiempos reportados son en float32.
El bono se corrió también en float32 para que los tiempos sean comparables.

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
| `entorno.json` | Presupuesto usado, versión de TensorFlow, GPU, semilla, hiperparámetros elegidos y conteo de duplicados |
| `historias_busqueda.json`, `historias_final.json` | Curvas de entrenamiento crudas, por si hay que rehacer alguna figura sin reentrenar |
| `predicciones_test.npz` | Etiquetas reales y probabilidades de los tres modelos sobre test (~400 KB) |
| `modelo_cnn_desde_cero.keras` | Único modelo que se versiona (~150 KB) |

Y del notebook `02`, ya listo para el informe:

| Archivo | Contenido |
|---|---|
| `fig_clases.png` | Una muestra por clase |
| `fig_busqueda.png` | Cada configuración probada y su relación entre rendimiento y sobreajuste |
| `fig_curvas.png` | Accuracy y pérdida de las tres familias, entrenamiento contra validación |
| `fig_confusion.png` | Matrices de confusión normalizadas por fila |
| `fig_f1_clase.png` | F1 por clase, de la más difícil a la más fácil |
| `fig_comparacion.png` | Accuracy contra parámetros y contra tiempo, con intervalo de confianza |
| `fig_errores.png` | Los quince errores más confiados del mejor modelo |
| `tabla_comparativa.*`, `tabla_busqueda.*`, `tabla_f1_por_clase.*` | Las tres tablas en `.csv`, `.tex` y `.md` |

Las figuras usan un color y un marcador fijos por modelo, elegidos para que sigan distinguiéndose en
impresión en gris y con daltonismo, y guardadas a 300 ppp.

Los modelos de VGG16 (~56 MB) y MobileNetV2 (~9 MB) y todos los checkpoints se quedan en `/content`,
fuera del repositorio: GitHub avisa a partir de 50 MB por archivo y no aportan nada al informe que no
esté ya en los CSV.

## Bono: `03_bono_redes_modernas.ipynb`

Extiende el análisis a **EfficientNetV2-B0**, la continuación de la línea de MobileNetV2. Repite las
condiciones del notebook `01`: misma partición, semilla, lotes, callbacks, rejilla de búsqueda, fine-tuning
de las últimas 20 capas con learning rate diez veces menor, float32 y una T4. Entrena y además arma sus
figuras; con `ENTRENAR = False` solo rehace las figuras a partir de lo guardado. El código deja lista
ConvNeXt-Tiny: basta agregar `"convnext_tiny"` a `MODELOS_BONO`, a cambio de unas tres o cuatro veces más
tiempo.

Resultado en test: **0.9273** de accuracy (IC 95 % 0.9218–0.9324) en 603 s, frente a 0.9319 de VGG16
en 2 687 s y 0.9243 de MobileNetV2 en 340 s. En la comparación pareada no se distingue de VGG16
(p = 0.058) ni de MobileNetV2 (p = 0.23); VGG16 sí supera a MobileNetV2 (p = 0.0015).

Todo queda en `results/punto2/bono/`:

| Archivo | Contenido |
|---|---|
| `experimentos_bono.csv`, `historias_busqueda_bono.json` | Búsqueda de hiperparámetros |
| `historias_final_bono.json` | Curvas del entrenamiento final, con la época donde empieza el fine-tuning |
| `metricas_test_bono.csv`, `reporte_efficientnetv2b0.csv`, `tiempos_bono.csv` | Métricas de test, reporte por clase y tiempos por etapa |
| `predicciones_efficientnetv2b0.npz` | Probabilidades sobre test, para rehacer figuras sin reentrenar |
| `entorno_bono.json` | GPU, presupuesto, precisión y configuración elegida |
| `fig_bono_comparacion.png` | Las cuatro redes: accuracy contra parámetros y contra tiempo, y diferencia pareada contra la mejor |
| `fig_bono_busqueda.png`, `fig_bono_curvas.png`, `fig_bono_confusion.png`, `fig_bono_f1_clase.png` | Búsqueda, curvas, matriz de confusión y F1 por clase |
| `tabla_bono_comparativa.*`, `tabla_bono_pareada.*`, `tabla_bono_busqueda.*`, `tabla_bono_f1_por_clase.*` | Tablas en `.csv`, `.tex` y `.md` |
