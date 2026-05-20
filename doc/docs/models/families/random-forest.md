# Random Forest

## Objetivo dentro del proyecto

[Random Forest](random-forest.md) funciona como modelo tabular robusto y como punto intermedio entre interpretabilidad y capacidad predictiva. Captura no linealidades e interacciones sin exigir escalado de features, y sirve para comparar si una familia de árboles ensamble mejora claramente el baseline de [regresión lineal](linear-regression.md).

## Estructura del modelo

Random Forest es un ensamble de árboles de decisión entrenados sobre muestras bootstrap y subconjuntos de variables. La predicción final es el promedio:

<div class="doc-math">
\[
\hat{\sigma}(x)=\frac{1}{B}\sum_{b=1}^{B}T_b(x)
\]
</div>

donde:

- $\hat{\sigma}(x)$ es la volatilidad predicha.
- $B$ es el número de árboles.
- $T_b(x)$ es la predicción del árbol $b$.

## Configuración y búsqueda

Parámetros fijos principales:

| Parámetro | Valor | Efecto |
| --- | --- | --- |
| `criterion` | `squared_error` | Optimiza reducción de error cuadrático. |
| `random_state` | 42 | Reproducibilidad. |
| `n_jobs` | -1 | Paraleliza entrenamiento. |
| `bootstrap` | `True` | Activa muestreo bootstrap por árbol. |
| `min_weight_fraction_leaf` | 0.0 | Sin restricción adicional por peso mínimo. |

Espacio de búsqueda:

| Hiperparámetro | Valores | Influencia |
| --- | --- | --- |
| `n_estimators` | 300, 400, 500 | Más árboles reducen varianza pero aumentan coste. |
| `max_depth` | `None`, 8, 12, 16 | Profundidad alta aumenta flexibilidad y riesgo de sobreajuste. |
| `min_samples_split` | 2, 5, 10 | Controla cuándo se permite dividir un nodo. |
| `min_samples_leaf` | 1, 2, 5, 10 | Regulariza hojas pequeñas. |
| `max_features` | `sqrt`, `log2`, 0.3, 0.5, 0.8 | Controla diversidad entre árboles. |
| `max_samples` | `None`, 0.6, 0.8, 0.9 | Submuestreo de filas por árbol. |
| `min_impurity_decrease` | 0, 1e-6, 1e-5, 1e-4 | Exige mejora mínima para dividir. |
| `ccp_alpha` | 0, 1e-6, 1e-5, 1e-4 | Poda por complejidad. |

La búsqueda muestrea 120 configuraciones con semilla fija. No agota el espacio completo; busca una cobertura razonable manteniendo coste controlado.

## Entrenamiento y progressive training

En entrenamiento estándar, cada árbol se ajusta sobre una muestra bootstrap y el ensamble promedia sus predicciones. En modo progresivo, esta familia no entrena por fases: recibe pesos de muestra mayores para observaciones cercanas a ATM. Esto prioriza la región central de la superficie sin retirar observaciones de alas.

## Interpretabilidad y cautelas

Random Forest no produce una fórmula compacta. Sus importancias internas pueden estar sesgadas por variables con muchos puntos de corte. Por eso en el dashboard se interpreta mediante [SHAP](../../dashboard/global/shap-fundamentals.md), [árboles surrogate](../../dashboard/global/surrogate-trees.md), [regresión simbólica](../../dashboard/global/symbolic-regression.md), [ICE](../../dashboard/behaviour/ice.md) y [ALE](../../dashboard/behaviour/ale.md).
