# Quantum Inspired

## Objetivo dentro del proyecto

La familia `QuantumInspiredNNFamily` extiende la [red neuronal secuencial](neural-networks.md) con una estructura tensorial inspirada en conceptos de computación cuántica. Es importante el matiz: el repositorio ejecuta una red neuronal clásica, no un circuito cuántico y no requiere hardware cuántico. Se denomina Quantum Inspired porque reutiliza ideas de representación de estados cuánticos y redes tensoriales para comprimir interacciones entre variables.

## Marco teórico quantum inspired

En computación cuántica, un estado de $n$ qubits se describe mediante amplitudes en un espacio de dimensión $2^n$. Esa representación crece exponencialmente, pero muchos estados relevantes pueden aproximarse de forma compacta mediante redes tensoriales. Una de las formas más conocidas es la descomposición tipo matrix product state, equivalente en aprendizaje automático a una factorización tensor-train.

La intuición que se toma prestada es la siguiente:

- Cada dimensión del tensor actúa como un modo o "subespacio" análogo a componentes de un sistema de qubits.
- Los rangos internos del tensor-train controlan cuánta dependencia cruzada se permite entre modos, de forma parecida a cómo la estructura de entrelazamiento limita o habilita correlaciones en un estado cuántico.
- La contracción de cores tensoriales combina información local y dependencias entre modos sin aprender un tensor denso completo.

Por eso la arquitectura es más extraña que una red secuencial densa: antes de aplicar el bloque tensorial, las features se proyectan a un embedding cuyo tamaño puede reordenarse como tensor. Ese tensor se contrae mediante cores de bajo rango. La aproximación es quantum inspired porque adapta una herramienta usada para representar sistemas cuánticos de muchos cuerpos, pero todo el cálculo se hace con tensores clásicos en Keras.

## Estructura del modelo

El flujo es:

```mermaid
flowchart LR
    A[Features financieras] --> B[Batch normalization]
    B --> C[Embedding denso]
    C --> D[Tensorización]
    D --> E[Contracción tensor-train]
    E --> F[Capa densa posterior]
    F --> G[Volatilidad predicha]
    classDef process fill:#f7f1ff,stroke:#7c4dbe,stroke-width:1.5px,color:#2d1948;
    class A,B,C,D,E,F,G process;
```

En el código, `TensorTrainLayer` recibe un tensor de dimensiones `tensor_dims` y produce `tt_output_dim` salidas contrayendo cores con rangos `tt_ranks`. El nombre interno de la capa conserva la técnica tensor-train porque describe la operación matemática; el nombre de familia en documentación y navegación es Quantum Inspired.

## Configuración y búsqueda

Parámetros fijos principales:

| Parámetro | Valor | Efecto |
| --- | --- | --- |
| `patience` | 12 | Paciencia de early stopping. |
| `epochs` | 150 | Número máximo de épocas. |
| `embedding_activation` | `swish` | Activación del embedding previo a tensorización. |
| `post_tt_activation` | `swish` | Activación después de la contracción tensorial. |
| `use_batch_norm` | `True` | Normaliza entrada antes del embedding. |

Espacio de búsqueda:

| Hiperparámetro | Valores | Influencia |
| --- | --- | --- |
| `tt_configuration` | combinaciones de `tensor_dims`, `tt_ranks`, `tt_output_dim` | Define estructura tensorial, rango efectivo y capacidad. |
| `dense_units` | 16 a 64 | Capacidad posterior al bloque tensorial. |
| `dropout_rate` | 0 a 0.15 | Regularización. |
| `l2_reg` | 0 a 1e-4 | Penalización de pesos. |
| `learning_rate` | 3e-4 a 1e-3 | Optimización Adam. |
| `batch_size` | 128 a 1024 | Coste y estabilidad. |
| `loss` | `huber`, `mse` | Robustez del ajuste. |
| `use_lr_scheduler` | `True`, `False` | Ajuste dinámico del learning rate. |
| `kernel_initializer` | `glorot_uniform`, `he_uniform` | Inicialización. |

Se exploran 50 configuraciones, menos que en otras familias neuronales por mayor coste estructural.

## Entrenamiento y progressive training

El entrenamiento reutiliza la lógica de la familia neuronal: escalado numérico, early stopping, posible reducción de learning rate y guardado del scaler. En modo progresivo, se entrena sucesivamente sobre datasets acumulados por cercanía a ATM, manteniendo la misma arquitectura Quantum Inspired en cada fase.

## Interpretabilidad y cautelas

La arquitectura puede capturar interacciones compactas, pero no produce una explicación directa. La interpretación debe venir de los artefactos del dashboard: [SHAP](../../dashboard/global/shap-fundamentals.md), [surrogates](../../dashboard/global/surrogate-trees.md), [regresión simbólica](../../dashboard/global/symbolic-regression.md), [ICE](../../dashboard/behaviour/ice.md), [ALE](../../dashboard/behaviour/ale.md) y superficies.

La documentación debe evitar afirmar ventaja cuántica. Lo defendible es que se prueba una arquitectura clásica que aplica ideas de computación cuántica, especialmente la representación compacta de estados mediante redes tensoriales y la noción de correlaciones controladas por rangos internos.
