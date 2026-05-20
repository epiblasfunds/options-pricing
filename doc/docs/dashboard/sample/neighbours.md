# Vecinos y soporte local

El análisis de vecinos mide si una muestra está cerca de observaciones históricas usadas como referencia. Es una herramienta de soporte empírico: no explica por sí sola qué feature causa una predicción, pero indica si el escenario que se está explicando pertenece a una región conocida del espacio de datos.

En el proyecto, los vecinos se calculan contra el split de train transformado. Esta decisión evita usar como referencia principal el mismo split de test que se está diagnosticando. El objetivo es contestar: dado un contrato o una entrada manual, qué casos históricos del entrenamiento se parecen más en el espacio de features del modelo.

```mermaid
flowchart TD
    A[Training reference frame] --> B[Feature frame train]
    C[Muestra a explicar] --> D[Feature frame muestra]
    B --> E[Transformación runtime]
    D --> E
    E --> F[Estandarización]
    F --> G[NearestNeighbors]
    G --> H[Tabla de vecinos]
    F --> I[PCA 3D]
    I --> J[Mapa de distancias]
    classDef data fill:#eaf4ff,stroke:#2f6fa8,stroke-width:1.5px,color:#17324d;
    classDef process fill:#f7f1ff,stroke:#7c4dbe,stroke-width:1.5px,color:#2d1948;
    classDef plot fill:#edf8f1,stroke:#2d7d46,stroke-width:1.5px,color:#173a24;
    class A,B,C,D data;
    class E,F,G,I process;
    class H,J plot;
```

## Cálculo de vecinos

La función `build_neighbors_frame`:

1. Muestrea hasta `neighbors_sample_size` observaciones del train reference frame.
2. Convierte train y muestra al feature frame transformado del modelo.
3. Aplica la transformación del runtime, incluido scaler si la familia lo requiere.
4. Estandariza columnas con `StandardScaler`.
5. Ajusta `NearestNeighbors`.
6. Recupera hasta `neighbors_k` vecinos por muestra.

Los parámetros actuales son:

| Parámetro | Valor | Efecto |
| --- | ---: | --- |
| `neighbors_sample_size` | 20000 | Tamaño máximo de la base de referencia. |
| `neighbors_k` | 200 | Número de vecinos guardados por muestra. |
| `neighbor_reference_split` | train | Split usado como referencia histórica. |

La distancia se calcula tras estandarización para que una variable con escala grande no domine por unidades. Esto es imprescindible porque las features pueden estar en días, precios, ratios o indicadores.

## Soporte local

Una muestra con vecinos cercanos tiene mayor soporte local. Esto no garantiza que la predicción sea correcta, pero reduce el riesgo de extrapolación. Por el contrario, una muestra aislada debe interpretarse con cautela: el modelo puede estar extrapolando a una región poco observada.

La lectura recomendada es:

| Situación | Interpretación |
| --- | --- |
| Distancias bajas y vecinos homogéneos | Escenario bien representado por el histórico. |
| Distancias bajas pero volatilidades reales dispersas | Zona conocida pero ruidosa. |
| Distancias altas | Posible extrapolación. |
| Vecinos con moneyness/vencimiento muy distintos | La métrica de distancia puede no estar capturando la similitud financiera deseada. |

## Mapa 3D: proyección PCA

Para poder observar las distancias entre muestras, es necesario hacer una reducción de dimensionalidad. Para ello se ha utilizado PCA con 3 componentes principales. Estas componentes principales no se han calculado con todos los datos de entrenamiento, sino, una vez extraídos los vecinos, las componentes principales se calculan únicamente con este vecindario ya fijado. 

Esto implica que las componentes principales que se muestran en el mapa 3D cambian de la explicación de una muestra a otra. La decisión ha sido tomada porque no se está utilizando para otra cosa más que para ganar una intuición espacial de cercanía visual, y de esta manera, las distancias relativas serían mucho más fieles a la realidad porque habría menos pares para representar distancias.


## Relación con SHAP local

[SHAP local](local-shap-waterfall.md) contesta qué variables empujan la predicción. Los vecinos contestan si hay observaciones parecidas. Juntas permiten una explicación más sólida:

```mermaid
flowchart LR
    A[Waterfall SHAP] --> C[Drivers de predicción]
    B[Vecinos] --> D[Soporte empírico]
    C --> E[Conclusión local]
    D --> E
    classDef method fill:#f7f1ff,stroke:#7c4dbe,stroke-width:1.5px,color:#2d1948;
    class A,B,C,D,E method;
```

Una explicación local sin vecinos puede ser matemáticamente correcta pero empíricamente frágil. Una tabla de vecinos sin waterfall muestra similitud, pero no atribuye la predicción. La combinación es lo que permite defender una predicción individual ante revisión.

## Referencias

- Cover, T. y Hart, P. (1967). *Nearest Neighbor Pattern Classification*. IEEE Transactions on Information Theory.
- Jolliffe, I. T. (2002). *Principal Component Analysis*. Springer.
- Molnar, C. (2022). *Interpretable Machine Learning*. Discusión sobre explicabilidad local y datos fuera de distribución.

