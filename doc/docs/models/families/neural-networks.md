# Redes neuronales secuenciales

La familia `SequentialNNFamily` usa una red feed-forward con capas densas. Su objetivo es capturar relaciones no lineales suaves entre las features financieras.

La forma conceptual es:

\[
h_1=\phi(W_1x+b_1), \quad
h_l=\phi(W_lh_{l-1}+b_l), \quad
\hat{\sigma}=W_oh_L+b_o
\]

donde:

- $h_l$ es la activación de la capa oculta $l$.
- $W_l$ y $b_l$ son pesos y sesgos de la capa $l$.
- $\phi$ es la función de activación.
- $\hat{\sigma}$ es la salida de volatilidad predicha.

## Espacio de búsqueda

| Hiperparámetro | Valores | Influencia |
| --- | --- | --- |
| `hidden_layers` | Arquitecturas de 1 a 4 capas | Capacidad del modelo. |
| `dropout_rate` | 0 a 0.3 | Regularización por apagado aleatorio. |
| `l2_reg` | 0 a 1e-3 | Penalización de pesos. |
| `learning_rate` | 5e-5 a 3e-3 | Velocidad de optimización Adam. |
| `batch_size` | 128 a 2048 | Ruido de gradiente y coste. |
| `activation` | `relu`, `gelu` | No linealidad. |
| `kernel_initializer` | `he_normal`, `he_uniform` | Inicialización de pesos. |
| `use_batch_norm` | `True`, `False` | Estabiliza distribuciones internas. |
| `use_lr_scheduler` | `True`, `False` | Reduce learning rate al estancarse. |
| `loss` | `huber`, `mse` | Robustez frente a errores grandes. |

Se exploran 120 configuraciones. Las features numéricas se escalan con `StandardScaler`; el scaler se guarda junto al modelo.

## Early stopping

El entrenamiento usa `EarlyStopping` sobre `val_rmse`, con paciencia 12 y restauración de los mejores pesos. Si `use_lr_scheduler` está activo, `ReduceLROnPlateau` reduce el learning rate cuando el progreso se estanca.

## Progressive training

En modo progresivo, la red entrena sucesivamente sobre datasets acumulados por cercanía a ATM. Esto no cambia la arquitectura, pero sí el orden de exposición a los datos.

## Interpretación

Las redes tienen baja interpretabilidad directa. Por eso el proyecto no se apoya en pesos de la red para explicar resultados. Se usan [SHAP](../../dashboard/global/shap-fundamentals.md), [regresión simbólica](../../dashboard/global/symbolic-regression.md), [Árboles surrogate](../../dashboard/global/surrogate-trees.md), [ICE](../../dashboard/behaviour/ice.md) y [ALE](../../dashboard/behaviour/ale.md).

