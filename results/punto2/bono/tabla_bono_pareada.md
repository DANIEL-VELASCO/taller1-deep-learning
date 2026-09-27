**Comparación pareada de cada red contra VGG16 sobre test**

| Modelo            |   Diferencia (VGG16 − modelo) | IC 95% diferencia   |   Solo acierta VGG16 |   Solo acierta el modelo |   p (McNemar) |
|:------------------|------------------------------:|:--------------------|---------------------:|-------------------------:|--------------:|
| CNN desde cero    |                        0.0415 | [0.0360, 0.0467]    |                  630 |                      215 |        0.0000 |
| MobileNetV2       |                        0.0076 | [0.0028, 0.0122]    |                  317 |                      241 |        0.0015 |
| EfficientNetV2-B0 |                        0.0046 | [-0.0002, 0.0094]   |                  305 |                      259 |        0.0580 |