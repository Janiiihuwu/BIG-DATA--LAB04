# Laboratorio 04: Análisis Exploratorio de Datos (EDA) en Big Data

Este proyecto implementa un flujo de Análisis Exploratorio de Datos (EDA) sobre el catálogo masivo de videojuegos de la plataforma **Steam**, utilizando Python y librerías científicas. El proceso es 100% reproducible, automatizado y estructurado bajo las directrices del Laboratorio 04.

---

## 1. Entorno y Versiones
- **Sistema Operativo:** Windows 10/11
- **Lenguaje:** Python `3.15.0` (compatible con `>= 3.10`)
- **Semilla de Reproducibilidad:** `SEED = 42` (configurada en NumPy y muestreos gráficos)

---

## 2. Dataset Utilizado
- **Nombre:** Steam Games Dataset (Mayo 2024 / Marzo 2025)
- **Autor / Origen:** Artemiloff (Kaggle)
- **URL de Descarga:** [https://www.kaggle.com/datasets/artermiloff/steam-games-dataset](https://www.kaggle.com/datasets/artermiloff/steam-games-dataset)
- **Archivo de Entrada:** `datos/games_may2024_full.csv`
- **Volumen Inicial:** 87.806 registros $\times$ 46 columnas (Cumple requisito $\ge 50.000$)
- **Volumen Limpio Final:** 87.803 registros $\times$ 52 columnas

---

## 3. Preguntas de Investigación Formuladas

1. **P1 [Distribución Numérica]:** ¿Cómo se distribuyen los precios (`price`) y las tasas de aprobación (`approval_rate`) de los videojuegos en Steam, y cuál es el impacto de los valores extremos?
2. **P2 [Diferencias entre Grupos]:** ¿Existen diferencias significativas en la recepción de la comunidad (tasa de aprobación) y volumen de reseñas entre títulos gratuitos (*Free-to-Play*) y de pago (*Pay-to-Play*)?
3. **P3 [Relación entre Variables]:** ¿Existe una asociación estadística entre el precio fijado para un título y su nivel de popularidad (`peak_ccu` o `total_reviews`)? *(Nota: Correlación no implica causalidad).*
4. **P4 [Evolución Temporal]:** ¿Cómo ha evolucionado el volumen anual de lanzamientos de videojuegos y el precio promedio a lo largo de las décadas (1997–2024) en Steam?

---

## 4. Instalación de Dependencias

Se recomienda utilizar un entorno virtual de Python:

```bash
# Crear y activar entorno virtual (opcional)
python -m venv venv
# En Windows:
venv\Scripts\activate
# En Linux/macOS:
source venv/bin/activate

# Instalar librerías requeridas
pip install -r requirements.txt
```

### Contenido de `requirements.txt`:
- `pandas>=2.0.0`
- `numpy>=1.24.0`
- `matplotlib>=3.7.0`
- `seaborn>=0.12.0`

---

## 5. Instrucciones de Ejecución

Para ejecutar el flujo completo de forma autónoma (diagnóstico, limpieza, cálculo estadístico, matrices de correlación y renderizado de figuras PNG a 300 DPI):

```bash
python analisis_exploratorio.py
```

*Nota: No se requiere intervención manual. El script utiliza rutas relativas, valida la existencia de directorios y exporta todos los artefactos automáticamente.*

---

## 6. Estructura del Proyecto y Resultados

```text
BIG-DATA--LAB04/
│
├── analisis_exploratorio.py        # Script principal con el flujo completo de EDA
├── requirements.txt                # Dependencias con versiones
├── README.md                       # Documentación técnica de ejecución
├── enunciado.txt                   # Rúbrica y pauta del laboratorio
│
├── datos/
│   ├── games_may2024_full.csv      # Dataset original sin modificar (87.806 filas)
│   └── dataset_limpio.csv          # Dataset tratado y enriquecido (87.803 filas)
│
└── resultados/
    ├── tablas/                     # Tablas exportadas en formato CSV
    │   ├── diagnostico_calidad.csv
    │   ├── log_transformaciones.csv
    │   ├── estadisticas_numericas.csv
    │   ├── frecuencias_generos.csv
    │   ├── frecuencias_modelo_negocio.csv
    │   ├── resumen_por_modelo_negocio.csv
    │   ├── resumen_por_genero_top10.csv
    │   ├── resumen_temporal_anual.csv
    │   ├── matriz_correlacion_spearman.csv
    │   └── matriz_correlacion_pearson.csv
    │
    └── figuras/                    # Visualizaciones de alta resolución (300 DPI)
        ├── fig1_distribucion_precios_aprobacion.png   # [Responde P1]
        ├── fig2_boxplot_valores_extremos.png          # [Responde P1 y P2]
        ├── fig3_dispersion_precio_vs_popularidad.png  # [Responde P3]
        ├── fig4_frecuencia_top10_generos.png          # [Responde P2]
        ├── fig5_evolucion_temporal_lanzamientos.png   # [Responde P4]
        └── fig6_mapa_calor_correlaciones.png          # [Responde P3 / Equipo]
```
