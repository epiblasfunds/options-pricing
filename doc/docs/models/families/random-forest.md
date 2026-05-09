# Random Forest

Random Forest es un ensamble de Árboles de decisión entrenados sobre muestras bootstrap y subconjuntos de variables. La predicción final es el promedio:

<div class="doc-math">
\[
\hat{\sigma}(x)=\frac{1}{B}\sum_{b=1}^{B}T_b(x)
\]
</div>

donde:

- $\hat{\sigma}(x)$ es la volatilidad predicha.
- $B$ es el número de árboles.
- $T_b(x)$ es la predicción del árbol $b$.

Esta familia captura no linealidades e interacciones sin exigir escalado de features. Es adecuada como modelo tabular robusto y como punto intermedio entre interpretabilidad y capacidad predictiva.

## Configuración fija

| Parámetro | Valor | Efecto |
| --- | --- | --- |
| `criterion` | `squared_error` | Optimiza reducción de error cuadrático. |
| `random_state` | 42 | Reproducibilidad. |
| `n_jobs` | -1 | Paraleliza entrenamiento. |
| `bootstrap` | `True` | Activa muestreo bootstrap por árbol. |
| `min_weight_fraction_leaf` | 0.0 | Sin restricción adicional por peso mínimo. |

## Espacio de búsqueda

| Hiperparámetro | Valores | Influencia |
| --- | --- | --- |
| `n_estimators` | 300, 400, 500 | Más Árboles reducen varianza pero aumentan coste. |
| `max_depth` | `None`, 8, 12, 16 | Profundidad alta aumenta flexibilidad y riesgo de sobreajuste. |
| `min_samples_split` | 2, 5, 10 | Controla cuándo se permite dividir un nodo. |
| `min_samples_leaf` | 1, 2, 5, 10 | Regulariza hojas pequeñas. |
| `max_features` | `sqrt`, `log2`, 0.3, 0.5, 0.8 | Controla diversidad entre Árboles. |
| `max_samples` | `None`, 0.6, 0.8, 0.9 | Submuestreo de filas por árbol. |
| `min_impurity_decrease` | 0, 1e-6, 1e-5, 1e-4 | Exige mejora mínima para dividir. |
| `ccp_alpha` | 0, 1e-6, 1e-5, 1e-4 | Poda por complejidad. |

La búsqueda muestrea 120 configuraciones con semilla fija. No agota el espacio completo; busca una cobertura razonable manteniendo coste controlado.

## Progressive training

En esta familia, el entrenamiento progresivo se implementa como pesos de muestra. Las observaciones más cercanas a ATM reciben mayor peso. Esto prioriza la región central de la superficie sin retirar observaciones de alas.

## Interpretación

Random Forest no produce una fórmula compacta. Sus importancias internas pueden estar sesgadas por variables con muchos puntos de corte. Por eso en el dashboard se interpreta mediante [SHAP](../../dashboard/global/shap-fundamentals.md), [Árboles surrogate](../../dashboard/global/surrogate-trees.md) y [regresión simbólica](../../dashboard/global/symbolic-regression.md).

