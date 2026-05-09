# XGBoost

XGBoost construye un ensamble secuencial de árboles. Cada nuevo árbol corrige errores residuales del conjunto anterior. En forma simplificada:

$$
\hat{\sigma}^{(m)}(x)=\hat{\sigma}^{(m-1)}(x)+\eta T_m(x)
$$

donde $\eta$ es el learning rate. En el repositorio se usa `XGBRegressor` con `booster="gbtree"`, `tree_method="hist"` y objetivo `reg:pseudohubererror`.

## Configuración fija

| Parámetro | Valor | Efecto |
| --- | --- | --- |
| `objective` | `reg:pseudohubererror` | Pérdida robusta frente a errores grandes. |
| `eval_metric` | `rmse` | Métrica monitorizada. |
| `tree_method` | `hist` | Entrenamiento eficiente por histogramas. |
| `base_score` | 0.20 | Inicialización cercana a la media esperada de volatilidad. |
| `grow_policy` | `depthwise` | Crecimiento por niveles de profundidad. |

## Espacio de búsqueda

| Hiperparámetro | Valores | Influencia |
| --- | --- | --- |
| `n_estimators` | 400 a 1400 | Número máximo de árboles. |
| `learning_rate` | 0.01 a 0.1 | Paso de cada árbol; menor valor suele requerir más estimadores. |
| `max_depth` | 3 a 6 | Complejidad de interacciones por árbol. |
| `min_child_weight` | 1 a 20 | Regulariza nodos con bajo soporte. |
| `subsample` | 0.6 a 1.0 | Submuestreo de filas. |
| `colsample_bytree` | 0.6 a 1.0 | Submuestreo de columnas por árbol. |
| `gamma` | 0 a 0.5 | Mejora mí­nima exigida para split. |
| `reg_alpha` | 0 a 0.5 | Regularización L1. |
| `reg_lambda` | 0.5 a 3.0 | Regularización L2. |
| `max_bin` | 128, 256, 512 | Resolución de histogramas. |
| `num_parallel_tree` | 1, 2, 4 | Permite comportamiento más tipo random forest dentro de boosting. |
| `early_stopping_rounds` | 30, 50, 80 | Detención si no mejora en validación interna. |

La familia explora 200 configuraciones. El conjunto externo de validación del fold no se usa como early stopping; se reserva para medir el candidato.

## Progressive training

En modo progresivo, XGBoost reparte `n_estimators` entre fases acumulativas ordenadas por cercaní­a a ATM. Cada fase entrena con más segmentos de moneyness y continúa desde el booster anterior. Así­, el modelo aprende primero la zona central y después incorpora alas.

## Interpretación

XGBoost suele ser competitivo en datos tabulares, pero su explicación directa no es trivial. En el proyecto se audita con [SHAP](../../dashboard/global/shap-fundamentals.md), [ICE](../../dashboard/behaviour/ice.md), [ALE](../../dashboard/behaviour/ale.md) y surrogates.

