# Familias de modelos

Esta página compara las familias implementadas. El detalle de cada familia está separado en páginas propias: [regresión lineal](families/linear-regression.md), [Random Forest](families/random-forest.md), [XGBoost](families/xgboost.md), [redes neuronales](families/neural-networks.md) y [Quantum Inspired](families/quantum-inspired.md).

Todas las familias comparten una abstracción común (`VolatilityModelFamilyABC`). Cada una declara parámetros fijos, espacio de búsqueda, instanciación, entrenamiento, persistencia y número de configuraciones exploradas.

```mermaid
classDiagram
    class VolatilityModelFamilyABC {
      get_family_name()
      get_fixed_params()
      get_hyperparameter_search_space()
      instantiate_model()
      fit_model()
      save_model()
    }
    class LinearRegressionFamily
    class RandomForestFamily
    class XGBoostFamily
    class SequentialNNFamily
    class QuantumInspiredNNFamily
    VolatilityModelFamilyABC <|-- LinearRegressionFamily
    VolatilityModelFamilyABC <|-- RandomForestFamily
    VolatilityModelFamilyABC <|-- XGBoostFamily
    VolatilityModelFamilyABC <|-- SequentialNNFamily
    SequentialNNFamily <|-- QuantumInspiredNNFamily
```

## Comparativa conceptual

| Familia | Interpretabilidad directa | Flexibilidad | Coste | Escalado | Página |
| --- | --- | --- | --- | --- | --- |
| [Regresión lineal](families/linear-regression.md) | Alta | Baja-media | Bajo | No imprescindible | [Detalle](families/linear-regression.md) |
| [Random Forest](families/random-forest.md) | Media | Media-alta | Medio | No | [Detalle](families/random-forest.md) |
| [XGBoost](families/xgboost.md) | Media-baja | Alta | Medio-alto | No | [Detalle](families/xgboost.md) |
| [Red neuronal secuencial](families/neural-networks.md) | Baja | Alta | Alto | Sí | [Detalle](families/neural-networks.md) |
| [Quantum Inspired](families/quantum-inspired.md) | Baja | Alta | Alto | Sí | [Detalle](families/quantum-inspired.md) |

## Referencias desde código

Cuando el código menciona una familia concreta, la correspondencia documental es:

| Identificador en código | Página |
| --- | --- |
| `LinearRegressionFamily`, `linear_regression` | [Regresión lineal](families/linear-regression.md) |
| `RandomForestFamily`, `random_forest` | [Random Forest](families/random-forest.md) |
| `XGBoostFamily`, `xgboost` | [XGBoost](families/xgboost.md) |
| `SequentialNNFamily`, `sequential_nn` | [Red neuronal secuencial](families/neural-networks.md) |
| `QuantumInspiredNNFamily`, `quantum_inspired_nn` | [Quantum Inspired](families/quantum-inspired.md) |

## Criterio de selección

La selección no se basa en una única ejecución. El pipeline genera candidatos por familia, evalúa por [k-folds temporales](splits-kfolds.md), calcula métricas agregadas y selecciona mediante una métrica compuesta que penaliza error, inestabilidad entre folds y sobreajuste.

```mermaid
flowchart TD
    A[Familia] --> B[Espacio de búsqueda]
    B --> C[Candidatos muestreados]
    C --> D[K-folds temporales]
    D --> E[Métricas train/validation]
    E --> F[Custom error]
    F --> G[Mejor configuración]
    G --> H[Reentrenamiento]
    H --> I[Bundle explicable]
    classDef process fill:#f7f1ff,stroke:#7c4dbe,stroke-width:1.5px,color:#2d1948;
    class A,B,C,D,E,F,G,H,I process;
```

## Relación con explicabilidad

Los modelos más flexibles no son descartados por ser menos interpretables directamente. El proyecto genera artefactos de explicabilidad para todos los modelos finales: [SHAP](../dashboard/global/shap-fundamentals.md), [árboles surrogate](../dashboard/global/surrogate-trees.md), [regresión simbólica](../dashboard/global/symbolic-regression.md), [ICE](../dashboard/behaviour/ice.md), [ALE](../dashboard/behaviour/ale.md) y superficies.

La comparación final debe separar tres dimensiones:

| Dimensión | Pregunta |
| --- | --- |
| Rendimiento | Qué modelo predice mejor volatilidad implícita en test. |
| Estabilidad | Qué modelo es menos sensible a folds, early stopping o regiones de moneyness. |
| Explicabilidad | Qué modelo puede justificarse con contribuciones, reglas, fórmulas y superficies coherentes. |

