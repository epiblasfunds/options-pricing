# Regresión lineal

## Objetivo dentro del proyecto

La regresión lineal se usa como baseline interpretable. Su función principal no es competir en flexibilidad con [Random Forest](random-forest.md), [XGBoost](xgboost.md) o [redes neuronales](neural-networks.md), sino establecer una referencia mínima: cuánto de la superficie de volatilidad puede explicarse mediante una combinación lineal de las features financieras diseñadas.

## Estructura del modelo

La forma del modelo es:

<div class="doc-math">
\[
\hat{\sigma}=\beta_0+\sum_{j=1}^{p}\beta_j x_j
\]
</div>

donde:

- $\hat{\sigma}$ es la volatilidad implícita predicha.
- $x_j$ representa features como `TTEYears`, `sqrtTTEYears`, `logMoneyness`, `logMoneynessSq`, `logForwardMoneyness`, `rate`, `isCall` e `isPut`.
- $\beta_0$ y $\beta_j$ son los parámetros estimados.

## Configuración y búsqueda

La familia `LinearRegressionFamily` no define espacio de búsqueda. Se entrena una única configuración:

| Parámetro | Valor | Efecto |
| --- | --- | --- |
| `fit_intercept` | `True` | Permite término independiente. |
| `copy_X` | `True` | Evita modificar la matriz de entrada. |
| `n_jobs` | `None` | Ejecución estándar de scikit-learn. |
| `positive` | `False` | No restringe coeficientes a ser positivos. |

## Entrenamiento y progressive training

En entrenamiento estándar se ajusta por mínimos cuadrados. En modo progresivo, el modelo recibe `sample_weight` para ponderar más las observaciones cercanas a ATM:

<div class="doc-math">
\[
\min_{\beta}\sum_i w_i(y_i-\beta_0-x_i^\top\beta)^2
\]
</div>

donde:

- $w_i$ es el peso de la observación cuando se activa progressive training.
- $y_i$ es la volatilidad real y $\beta_0+x_i^\top\beta$ la predicción.

El progressive training no cambia la forma lineal. Solo cambia el objetivo ponderado para priorizar la zona central de la superficie.

## Interpretabilidad y cautelas

La ventaja de esta familia es que los coeficientes tienen lectura directa condicionada al resto de features. Esa lectura exige cuidado: si las features están correlacionadas, un coeficiente no representa un efecto causal aislado. En este proyecto, su valor principal es servir como control.

La regresión lineal puede capturar curvatura solo si la feature ya está creada, como `logMoneynessSq`. No aprende automáticamente interacciones no incluidas. Por eso se compara con familias más flexibles y con los artefactos del dashboard: [SHAP](../../dashboard/global/shap-fundamentals.md), [árboles surrogate](../../dashboard/global/surrogate-trees.md), [regresión simbólica](../../dashboard/global/symbolic-regression.md), [ICE](../../dashboard/behaviour/ice.md) y [ALE](../../dashboard/behaviour/ale.md).
