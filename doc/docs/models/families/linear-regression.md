# Regresión lineal

La regresión lineal se usa como baseline interpretable. Su función principal no es competir en flexibilidad con modelos de árboles o redes, sino establecer una referencia mí­nima: cuánto de la superficie de volatilidad puede explicarse mediante una combinación lineal de las features financieras diseñadas.

La forma del modelo es:

$$
\hat{\sigma}=\beta_0+\sum_{j=1}^{p}\beta_j x_j
$$

donde $x_j$ son features como `TTEYears`, `sqrtTTEYears`, `logMoneyness`, `logMoneynessSq`, `logForwardMoneyness`, `rate`, `isCall` e `isPut`.

## Configuración

La familia `LinearRegressionFamily` no define espacio de búsqueda. Se entrena una única configuración:

| Parámetro | Valor | Efecto |
| --- | --- | --- |
| `fit_intercept` | `True` | Permite término independiente. |
| `copy_X` | `True` | Evita modificar la matriz de entrada. |
| `n_jobs` | `None` | Ejecución estándar de scikit-learn. |
| `positive` | `False` | No restringe coeficientes a ser positivos. |

En modo progresivo, el modelo recibe `sample_weight` para ponderar más las observaciones cercanas a ATM. Esto no cambia la forma lineal, pero modifica el objetivo de mí­nimos cuadrados ponderados:

$$
\min_{\beta}\sum_i w_i(y_i-\beta_0-x_i^\top\beta)^2
$$

## Interpretabilidad

La ventaja de esta familia es que los coeficientes tienen lectura directa condicionada al resto de features. Sin embargo, esa lectura exige cuidado: si las features están correlacionadas, un coeficiente no representa un efecto causal aislado. En este proyecto, su valor principal es servir como control. Si la regresión lineal obtiene rendimiento razonable, las transformaciones financieras ya capturan una parte significativa de la estructura.

## Limitaciones

La regresión lineal puede capturar curvatura solo si la feature ya está creada, como `logMoneynessSq`. No aprende automáticamente interacciones no incluidas. Por eso se compara con familias más flexibles y con artefactos de explicabilidad del dashboard.

