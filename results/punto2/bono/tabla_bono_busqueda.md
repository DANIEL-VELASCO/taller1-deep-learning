**Búsqueda de hiperparámetros de EfficientNetV2-B0**

| Modelo            | Configuración                 |   Mejor época |   Accuracy val. |   Pérdida val. |   Brecha |   Tiempo (s) |
|:------------------|:------------------------------|--------------:|----------------:|---------------:|---------:|-------------:|
| EfficientNetV2-B0 | drop 0.2 · sin aug · lr 0.001 |            10 |          0.9077 |         0.2533 |  -0.0150 |     192.5851 |
| EfficientNetV2-B0 | drop 0.3 · aug · lr 0.001     |            11 |          0.8748 |         0.3342 |  -0.0261 |     273.8011 |
| EfficientNetV2-B0 | drop 0.3 · aug · lr 0.0001    |            12 |          0.8480 |         0.4187 |  -0.0187 |     270.1397 |