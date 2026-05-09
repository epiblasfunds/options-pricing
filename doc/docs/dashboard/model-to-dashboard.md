# De modelo entrenado a bundle visualizable

El paquete de conversión toma modelos guardados tras el reentrenamiento final y genera un bundle autocontenido. Ese bundle es lo que consume el [dashboard](index.md). La conversión no cambia el modelo principal; calcula artefactos auxiliares que permiten explicarlo sin repetir procesos costosos en cada interacción.

```mermaid
flowchart TD
    A[trained_models] --> B[Cargar runtime del modelo]
    C[retrained_metadata] --> B
    D[Train split raw] --> E[Predicciones train]
    F[Test split raw] --> G[Predicciones test]
    B --> E
    B --> G
    G --> H[Dataset dashboard]
    E --> I[Referencia de vecinos]
    H --> J[SHAP]
    H --> K[Surrogates]
    H --> L[Superficies/ICE/ALE]
    H --> M[Diagnóstico]
    I --> N[Vecinos y PCA]
    J --> O[Bundle saved_models/model_id]
    K --> O
    L --> O
    M --> O
    N --> O
```

## Descubrimiento y carga del runtime

La conversión descubre familias entrenadas en la carpeta de modelos finales. Para cada familia:

- Localiza el fichero del estimador.
- Localiza metadatos de final test.
- Localiza metadatos train/val si existen.
- Carga scaler cuando el modelo lo necesita.
- Fuerza predicción en proceso único cuando el estimador permite paralelismo, para evitar conflictos en runtime.

## Dataset de dashboard

El dataset de dashboard se construye sobre el split de test. Contiene columnas raw, features transformadas, predicción, residual y error absoluto:

\[
Residual_i = y_i-\hat{y}_i
\]

donde:

- $Residual_i$ es el residuo de la observación $i$.
- $y_i$ es la volatilidad real observada.
- $\hat{y}_i$ es la volatilidad predicha.

\[
AbsoluteError_i = |Residual_i|
\]

donde:

- $AE_i$ o $AbsoluteError_i$ es el error absoluto de la observación $i$.
- $y_i$ es el valor real observado.
- $\hat{y}_i$ es la predicción del modelo.
- $Residual_i$ es el residuo de la observación $i$.

El split de train se conserva como referencia para vecinos. Esta separación es deliberada:

- Diagnóstico y visualizaciones de rendimiento se hacen sobre test.
- Vecinos se buscan contra train, para dar contexto histórico sin usar el mismo conjunto evaluado como referencia principal.

## SHAP

Se calculan explicaciones [SHAP](global/shap-fundamentals.md) mediante un explainer de permutación:

- Un fondo pequeño muestreado del dataset.
- Una muestra global para beeswarm, barras y heatmap.
- Una muestra local para waterfalls.

La explicación aproxima la descomposición:

\[
\hat{f}(x)=\phi_0+\sum_{j=1}^{p}\phi_j(x)
\]

donde:

- $\hat{\sigma}$ o $f(x)$ es la predicción del modelo.
- $\phi_0$ es el valor base del explicador.
- $\phi_j$ es la contribución SHAP de la feature $j$.
- $p$ es el número de features explicadas.

donde $\phi_0$ es el valor base y $\phi_j$ la contribución de cada feature.

## Árboles surrogate

Para varias profundidades configuradas se entrena un [árbol surrogate](global/surrogate-trees.md) que imita las predicciones del modelo principal sobre una muestra. Su objetivo no es sustituir al modelo, sino mostrar reglas aproximadas. La fidelidad se mide comparando:

\[
\hat{\sigma}_{modelo} \quad \text{vs} \quad \hat{\sigma}_{árbol}
\]

donde:

- Los símbolos de la fórmula se definen en el contexto técnico inmediatamente anterior.

Se guardan métricas, importancias, reglas textuales, número de hojas y profundidad efectiva.

## Regresión simbólica

La [regresión simbólica](global/symbolic-regression.md) busca una expresión cerrada que aproxime el modelo principal. En el proyecto se implementa con PySR, una búsqueda evolutiva de expresiones, no con reinforcement learning ni con un método de enjambre.

El bundle guarda:

- Ecuación seleccionada.
- Expresión en LaTeX.
- Features usadas.
- Complejidad.
- Tabla de ecuaciones candidatas.
- Fidelidad frente al modelo principal.

## Superficies, ICE y ALE

Las [superficies de volatilidad](behaviour/volatility-surfaces.md) se generan alrededor de anclas representativas del test. Para cada ancla se construye una grilla de moneyness y vencimiento, manteniendo el resto de variables fijas.

[ICE](behaviour/ice.md) y [ALE](behaviour/ale.md) se calculan para features de análisis:

- Strike.
- Subyacente.
- Tiempo a vencimiento.
- Tipo de interés.

[ICE](behaviour/ice.md) muestra respuestas individuales. [ALE](behaviour/ale.md) muestra efecto acumulado centrado por bins, más robusto cuando hay correlaciones entre variables.

## Vecinos

Los [vecinos](sample/neighbours.md) se calculan contra el split de train transformado. Se estandarizan features y se recuperan observaciones más cercanas. También se guarda una proyección PCA para representar distancias en 3D.

## Estructura del bundle

Cada carpeta de modelo contiene:

- `metadata.json` raíz.
- Carpeta `dashboard_model/`.
- Dataset de test con predicciones.
- Referencia de train para vecinos.
- SHAP global/local.
- Frames de vecinos, superficies, ICE y ALE.
- Diagnóstico.
- Árboles surrogate.
- Modelo simbólico.
- Stub de respuesta manual.

El dashboard puede arrancar sin recalcular estos artefactos, lo que reduce tiempo de respuesta.




