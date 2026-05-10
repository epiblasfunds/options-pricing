# Entrenamiento y selección

El entrenamiento se organiza en dos niveles. Primero se exploran hiperparámetros por familia mediante [k-folds temporales](splits-kfolds.md). Después se toma la mejor configuración y se reentrena en fases posteriores.

```mermaid
flowchart TD
    A[Familia de modelo] --> B[Construir candidatos de hiperparámetros]
    B --> C[Para cada candidato]
    C --> D[Entrenar en k-folds temporales]
    D --> E[Calcular métricas train y val]
    E --> F[Agregar media y desviación]
    F --> G[Calcular custom_error_1]
    G --> H[Seleccionar mejor candidato]
    H --> I[Guardar metadata de familia]
```

## Búsqueda de hiperparámetros

Cada familia define su espacio de búsqueda. Cuando el espacio es vacío, se entrena una única configuración. Cuando hay múltiples opciones, se muestrean configuraciones con semilla fija para reproducibilidad.

La búsqueda no agota necesariamente todo el espacio. Para familias grandes, el número de iteraciones está acotado. El sistema registra cuántas configuraciones se han explorado frente a la cardinalidad máxima.

## Métricas base

Las métricas base son:

<div class="doc-math">
\[
MAE = \frac{1}{n}\sum_i |y_i-\hat{y}_i|
\]
</div>

donde:

- $MAE$ es el error absoluto medio.
- $n$ es el número de observaciones.
- $y_i$ es el valor real observado.
- $\hat{y}_i$ es la predicción del modelo para la observación $i$.

<div class="doc-math">
\[
RMSE = \sqrt{\frac{1}{n}\sum_i (y_i-\hat{y}_i)^2}
\]
</div>

donde:

- $RMSE$ es la raíz del error cuadrático medio.
- $n$ es el número de observaciones.
- $y_i$ es el valor real observado.
- $\hat{y}_i$ es la predicción del modelo para la observación $i$.

<div class="doc-math">
\[
R^2 = 1-\frac{\sum_i (y_i-\hat{y}_i)^2}{\sum_i (y_i-\bar{y})^2}
\]
</div>

donde:

- $R^2$ es el coeficiente de determinación.
- $y_i$ es el valor real observado.
- $\hat{y}_i$ es la predicción del modelo.
- $\bar{y}$ es la media de los valores reales.

Se calculan para train y validation en cada fold. Luego se agregan media y desviación estándar entre folds.

## Métrica de selección

La selección no usa solo RMSE medio. Usa una métrica compuesta que penaliza:

- Error de validation.
- Variabilidad entre folds.
- Gap de sobreajuste cuando validation es peor que train.

<div class="doc-math">
\[
CE_1 = RMSE_{val}+\alpha \cdot std(RMSE_{val})+\beta \cdot \max(0, RMSE_{val}-RMSE_{train})
\]
</div>

donde:

- $CE_1$ es la métrica compuesta de selección.
- $RMSE_{val}$ es el RMSE medio de validación.
- $RMSE_{train}$ es el RMSE medio de entrenamiento.
- $std(RMSE_{val})$ es la desviación estándar del RMSE de validación entre folds.
- $\alpha$ y $\beta$ son penalizaciones configuradas.

Esto evita elegir una configuración que gane por poco en un fold pero sea inestable o claramente sobreajustada.

## Early stopping

El early stopping aparece en las familias con entrenamiento iterativo:

- [XGBoost](families/xgboost.md) usa un conjunto interno de validación para detener boosting cuando no mejora.
- [Redes neuronales](families/neural-networks.md) y [Quantum Inspired](families/quantum-inspired.md) monitorizan RMSE de validación interna y restauran los mejores pesos.
- Las redes pueden reducir learning rate si el progreso se estanca.

```mermaid
flowchart LR
    A[Train fold] --> B[partición interna de ajuste]
    A --> C[partición interna de early stopping]
    B --> D[Entrenamiento iterativo]
    C --> D
    D --> E{Mejora val_rmse?}
    E -->|sí| F[Guardar mejor estado]
    E -->|no durante paciencia| G[Parar]
    F --> D
```

El conjunto externo de validation del fold no se usa para ajustar early stopping; se reserva para estimar rendimiento del candidato. Esto mantiene separadas las decisiones internas de optimización y la evaluación del fold.

## Escalado

Las familias neuronales guardan un scaler para features numéricas. El scaler se ajusta solo con datos permitidos en la fase correspondiente. En evaluación final, el scaler se ajusta con el bloque de entrenamiento disponible y se aplica a test sin aprender de test.

Los modelos de árboles, [Random Forest](families/random-forest.md) y [XGBoost](families/xgboost.md), y la [regresión lineal](families/linear-regression.md) se guardan sin scaler adicional en el flujo actual.

## Artefactos de entrenamiento

Al terminar la búsqueda por familia se guarda:

- Nombre de familia.
- Mejor candidato.
- Parámetros del mejor candidato.
- Tabla de métricas de todos los candidatos.
- Métricas por fold.
- Predicciones de validación usadas para gráficos.
- Historial de épocas cuando la familia lo produce.

Estos metadatos permiten reconstruir la justificación de selección sin reentrenar todo.





