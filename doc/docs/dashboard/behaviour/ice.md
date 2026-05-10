# Curvas ICE

Las curvas ICE, del inglés *Individual Conditional Expectation*, muestran cómo cambia la predicción de un modelo para observaciones individuales cuando se modifica una feature y se mantienen las demás variables fijas. En este proyecto se usan para analizar respuestas del modelo de volatilidad ante cambios de strike, subyacente, vencimiento o tipo de interés.

Formalmente, para una feature $j$, una observación $x_i$ y un valor contrafactual $z$:

<div class="doc-math">
\[
ICE_i^{(j)}(z)=f(z, x_{i,-j})
\]
</div>

donde:

- $ICE_i^{(j)}(z)$ es la respuesta individual de la observación $i$ para la feature $j$.
- $z$ es el valor contrafactual asignado a la feature analizada.
- $x_{i,-j}$ son las demás features de la observación $i$.
- $f$ es el modelo de predicción de volatilidad.

donde $x_{i,-j}$ representa todas las variables de la observación salvo la feature analizada. Cada línea ICE es una trayectoria individual de predicción.

```mermaid
flowchart LR
    A[Muestras reales] --> B[Seleccionar feature j]
    B --> C[Construir grid por cuantiles]
    C --> D[Sustituir j por cada valor]
    D --> E[Predicción del modelo]
    E --> F[Curvas individuales ICE]
    classDef data fill:#eaf4ff,stroke:#2f6fa8,stroke-width:1.5px,color:#17324d;
    classDef process fill:#f7f1ff,stroke:#7c4dbe,stroke-width:1.5px,color:#2d1948;
    classDef plot fill:#edf8f1,stroke:#2d7d46,stroke-width:1.5px,color:#173a24;
    class A,C data;
    class B,D,E process;
    class F plot;
```

## Implementación en el proyecto

La función `build_ice_frame`:

1. Muestrea hasta `ice_sample_size` filas del dataset de dashboard.
2. Para cada feature en `ANALYSIS_FEATURE_NAMES`, crea un grid por cuantiles con `curve_points`.
3. Modifica el raw frame mediante `apply_feature_override`.
4. Vuelve a predecir con `predict_raw_frame`.
5. Guarda `feature_name`, `sample_id`, `feature_value` y `prediction`.

Los parámetros actuales son:

| Parámetro | Valor | Efecto |
| --- | ---: | --- |
| `ice_sample_size` | 24 | Número máximo de curvas individuales por feature. |
| `curve_points` | 25 | Resolución de cada curva. |
| Features | `StrikePrice`, `UnderlyingPrice`, `TimeToExpiration`, `Rate` | Variables analizadas en respuesta. |

## Interpretación

Una curva ICE no resume el efecto medio. Muestra el efecto condicional para una observación concreta. Esto es crucial en volatilidad implícita, porque el efecto de mover strike puede depender del subyacente, del vencimiento y del tipo de opción.

Patrones habituales:

| Patrón ICE | Lectura |
| --- | --- |
| Curvas casi paralelas | Efecto similar entre observaciones; poca interacción visible. |
| Curvas con pendientes diferentes | Heterogeneidad; interacción con otras variables. |
| Cruces frecuentes | El ranking de predicciones cambia al mover la feature. |
| Saltos o dientes | Posible discontinuidad del modelo o grid en zona poco soportada. |

Si las curvas ICE de `StrikePrice` divergen mucho, el modelo no está usando strike de forma aislada; probablemente la relación con `UnderlyingPrice` y moneyness domina. Si las curvas ICE de `TimeToExpiration` cambian mucho en corto plazo, conviene cruzar la lectura con el diagnóstico por maturity.

## Diferencia con PDP

Un Partial Dependence Plot promedia las curvas ICE:

<div class="doc-math">
\[
PDP_j(z)=\frac{1}{n}\sum_{i=1}^{n}f(z,x_{i,-j})
\]
</div>

donde:

- $PDP_j(z)$ es la dependencia parcial de la feature $j$.
- $n$ es el número de observaciones promediadas.
- $z$ es el valor contrafactual de la feature analizada.
- $x_{i,-j}$ son las demás features de la observación $i$.

El PDP puede ocultar heterogeneidad. Si algunas curvas suben y otras bajan, el promedio puede parecer plano aunque el modelo tenga efectos locales fuertes. Por ese motivo el dashboard prioriza ICE para inspección detallada y [ALE](ale.md) para estimar efectos acumulados más robustos ante correlación.

## Limitación principal: contrafactuales irreales

ICE puede evaluar combinaciones poco plausibles. Por ejemplo, cambiar `StrikePrice` manteniendo constante el resto puede llevar a moneyness muy alejadas de la distribución observada. El pipeline intenta mitigar esto usando grids por cuantiles, pero no elimina el problema. Por eso una curva ICE debe interpretarse con soporte local:

- ¿Existen vecinos históricos cerca del escenario?
- ¿El valor de feature está dentro de rangos observados?
- ¿El diagnóstico muestra error alto en esa zona?
- ¿La superficie local es suave alrededor de ese punto?

## Referencias

- Goldstein, A., Kapelner, A., Bleich, J. y Pitkin, E. (2015). *Peeking Inside the Black Box: Visualizing Statistical Learning with Plots of Individual Conditional Expectation*. Journal of Computational and Graphical Statistics.
- Molnar, C. (2022). *Interpretable Machine Learning*. Capítulo de ICE y PDP.

