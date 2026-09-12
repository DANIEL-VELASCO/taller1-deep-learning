# Punto 2 – CNN sobre Fashion-MNIST

Notebooks previstos (uno por aproximación, para que la comparación sea limpia):

| Notebook | Contenido | Rúbrica |
|---|---|---|
| `01_cnn_desde_cero.ipynb` | Carga y preparación de datos (normalización, split train/val/test), CNN propia justificada | 5% + 5% |
| `02_cnn_filtros_red_grande.ipynb` | Filtros iniciales de VGG16 / ResNet50 (`include_top=False`, `weights='imagenet'`), congelados y luego fine-tuning | 10% |
| `03_mobilenetv2_transfer.ipynb` | Transfer learning con MobileNetV2 | 10% |
| `04_bono_cnn_moderna.ipynb` | EfficientNetV2 / ConvNeXt / ResNeXt | Bono 10% |

Reglas para que la comparación sea justa (15% de la rúbrica):

- Mismo split: `fashion_mnist.load_data()` da 60k train / 10k test. Reservar p. ej. 10% de train como
  validación con la **misma semilla** en todos los notebooks. El test de 10k solo se usa al final.
- Mismas métricas: accuracy, F1 macro, matriz de confusión, curvas de loss/accuracy, tiempo de
  entrenamiento y número de parámetros (entrenables / totales).
- Mismos callbacks (EarlyStopping + ReduceLROnPlateau) y mismo optimizador base.
- Las redes preentrenadas exigen entrada 3 canales y tamaño >= 32x32: replicar el canal de grises y
  redimensionar (p. ej. 32x32 o 64x64). Documentar el tamaño elegido porque afecta el costo.
- Guardar las métricas de cada corrida en `results/punto2/experimentos.csv` para armar las tablas del informe.
