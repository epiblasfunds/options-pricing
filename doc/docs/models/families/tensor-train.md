# Red tensor-train

La familia `QuantumInspiredNNFamily` extiende la red secuencial con una capa tensor-train. Aunque el nombre del paquete habla de inspiración cuántica, la implementación del repositorio es una red neuronal clásica con una factorización tensorial de bajo rango. No ejecuta un circuito cuántico ni requiere hardware cuántico.

El flujo es:

```mermaid
flowchart LR
    A[Features] --> B[Batch normalization]
    B --> C[Embedding denso]
    C --> D[Reshape a tensor]
    D --> E[TensorTrainLayer]
    E --> F[Capa densa posterior]
    F --> G[Volatilidad predicha]
    classDef process fill:#f7f1ff,stroke:#7c4dbe,stroke-width:1.5px,color:#2d1948;
    class A,B,C,D,E,F,G process;
```

## Idea del tensor-train

Una factorización tensor-train representa un tensor de alto orden mediante una cadena de cores de bajo rango. En lugar de aprender un tensor completo, se aprende una secuencia de factores conectados por rangos internos. Esto puede reducir parámetros y forzar una estructura de interacción más compacta.

En el código, `TensorTrainLayer` recibe un tensor de dimensiones `tensor_dims` y produce `tt_output_dim` salidas contrayendo cores con rangos `tt_ranks`.

## Espacio de búsqueda

| Hiperparámetro | Valores | Influencia |
| --- | --- | --- |
| `tt_configuration` | combinaciones de `tensor_dims`, `tt_ranks`, `tt_output_dim` | Define estructura tensorial y capacidad. |
| `dense_units` | 16 a 64 | Capacidad posterior al bloque tensorial. |
| `dropout_rate` | 0 a 0.15 | Regularización. |
| `l2_reg` | 0 a 1e-4 | Penalización de pesos. |
| `learning_rate` | 3e-4 a 1e-3 | Optimización Adam. |
| `batch_size` | 128 a 1024 | Coste y estabilidad. |
| `loss` | `huber`, `mse` | Robustez del ajuste. |
| `use_lr_scheduler` | `True`, `False` | Ajuste dinámico del learning rate. |
| `kernel_initializer` | `glorot_uniform`, `he_uniform` | Inicialización. |

Se exploran 50 configuraciones, menos que en otras familias neuronales por mayor coste estructural.

## Interpretación y cautelas

La capa tensor-train puede capturar interacciones compactas, pero no produce una explicación directa. La interpretación debe venir de los artefactos del dashboard: [SHAP](../../dashboard/global/shap-fundamentals.md), [surrogates](../../dashboard/global/surrogate-trees.md), [regresión simbólica](../../dashboard/global/symbolic-regression.md), [ICE](../../dashboard/behaviour/ice.md), [ALE](../../dashboard/behaviour/ale.md) y superficies.

La documentación debe evitar afirmar ventaja cuántica. Lo defendible es que se prueba una arquitectura clásica con factorización tensorial inspirada en representaciones de redes tensoriales.

