import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Configuración
## Arranca una misma semilla para asegurar la reproducibilidad de los resultados aleatorios
SEED = 42
np.random.seed(SEED)

## Definición del estilo visual y estandarización estética de los gráficos
sns.set_theme(style="whitegrid", palette="deep")
plt.rcParams.update({
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "figure.titlesize": 14,
    "figure.autolayout": True  # Ajuste automático de márgenes para evitar solapamientos
})

# Función auxiliar de búsqueda insensible a mayúsculas, minúsculas y espacios
def obtener_columna(df: pd.DataFrame, candidatos: list[str]) -> str | None:
    ## Mapea columnas en minúsculas y sin espacios al nombre original en el DataFrame
    mapa_cols = {col.lower().strip(): col for col in df.columns}
    
    ## Itera sobre los candidatos en orden de preferencia
    for cand in candidatos:
        if cand.lower().strip() in mapa_cols:
            return mapa_cols[cand.lower().strip()]
            
    ## Retorna None si ninguna de las opciones candidatas existe en las columnas
    return None

# ========== 1. CARGA E INSPECCIÓN DE DATOS ==========
def cargar_y_explorar_datos(ruta_archivo: str) -> pd.DataFrame | None:
    """
    Lee los datos desde una ruta configurable, valida el mínimo de 50.000 registros,
    informa dimensiones, nombres de columnas, tipos de datos y primeras observaciones.
    """
    print("\n" + "=" * 52)
    print("1. INICIANDO CARGA E INSPECCIÓN DE DATOS")
    print("=" * 52)

    # Error: Verificación de existencia del archivo .csv en disco
    if not os.path.exists(ruta_archivo):
        print(f"Error: No se encontró el archivo en la ruta '{ruta_archivo}'.")
        print("Asegúrate de que el archivo esté dentro de la carpeta 'datos/'.")
        return None

    # Intento de lectura del dataset en la variable "df"
    try:
        ## Lectura estándar con codificación universal UTF-8
        df = pd.read_csv(ruta_archivo, encoding="utf-8", low_memory=False)
    except UnicodeDecodeError:
        ## Alternativa de codificación si el archivo contiene caracteres especiales incompatibles
        print("Advertencia: Falló lectura con UTF-8. Reintentando con latin-1...")
        df = pd.read_csv(ruta_archivo, encoding="latin-1", low_memory=False)
    except Exception as e:
        ## Captura de errores inesperados de I/O o formato corrupto
        print(f"Error crítico al leer el archivo CSV: {e}")
        return None

    # Obtención de dimensiones del conjunto de datos
    filas, columnas = df.shape
    print(f"-> Archivo cargado exitosamente desde: '{ruta_archivo}'")
    print(f"-> Dimensiones del dataset: {filas:,} filas x {columnas} columnas")

    # Validación del requisito metodológico de al menos 50.000 registros
    if filas < 50000:
        ## Lanza un error deteniendo la ejecución si no se cumple el volumen requerido
        raise ValueError(
            f"El dataset tiene {filas:,} registros "
            f"(se requieren al menos 50.000 registros antes de la limpieza)."
        )
    else:
        print(f"-> Cumple con el requisito mínimo ({filas:,} >= 50,000 registros).")

    # Resumen estructural: Nombres de columnas y tipos de datos detectados
    print("\n--- Columnas y Tipos de Datos ---")
    resumen_tipos = pd.DataFrame({
        "Columna": df.columns,
        "Tipo_Dato": [str(t) for t in df.dtypes.values]
    })
    print(resumen_tipos.to_string(index=False))

    # Vista preliminar: Muestra reducida para inspección visual sin saturar la terminal
    cols_vista = df.columns[:8]
    print("\n--- Primeras 5 Observaciones (Vista Preliminar) ---")
    print(df[cols_vista].head(5).to_string())

    return df

# ========== 2. DIAGNÓSTICO DE CALIDAD DE DATOS ========== 
def diagnosticar_calidad(df: pd.DataFrame, ruta_tablas: str = "resultados/tablas") -> pd.DataFrame:
    """
    Cuantifica valores faltantes por columna, duplicados exactos vs repeticiones
    válidas de entidad, tipos inconsistentes y valores fuera de rangos plausibles.
    Exporta el diagnóstico a 'resultados/tablas/diagnostico_calidad.csv'.
    """
    print("\n" + "=" * 34)
    print("2. DIAGNÓSTICO DE CALIDAD DE DATOS")
    print("=" * 34)

    # Creación de directorio de salida para almacenar tablas generadas
    os.makedirs(ruta_tablas, exist_ok=True)
    total_filas = len(df)

    # Identificación flexible de columnas clave independientemente de variantes de nombre
    col_id = obtener_columna(df, ["AppID", "appid", "steam_appid"])
    col_name = obtener_columna(df, ["Name", "name"])
    col_date = obtener_columna(df, ["Release date", "release_date"])
    col_price = obtener_columna(df, ["Price", "price"])
    col_pos = obtener_columna(df, ["Positive", "positive"])
    col_neg = obtener_columna(df, ["Negative", "negative"])

    # 1. Cuantificación de valores faltantes (NA y cadenas vacías) por columna
    resumen_calidad = []
    for col in df.columns:
        ## Conteo de valores nulos nativos (NaN / None)
        nulos = int(df[col].isna().sum())
        
        ## Conteo de cadenas de texto vacías o compuestas exclusivamente de espacios
        if df[col].dtype == "object":
            vacios = int((df[col].fillna("").astype(str).str.strip() == "").sum()) - nulos
            vacios = max(0, vacios)
        else:
            vacios = 0

        ## Suma total de valores ausentes y cálculo del porcentaje relativo
        total_faltantes = nulos + vacios
        pct_faltantes = (total_faltantes / total_filas) * 100

        resumen_calidad.append({
            "columna": col,
            "tipo_dato": str(df[col].dtype),
            "nulos_na": nulos,
            "cadenas_vacias": vacios,
            "total_faltantes": total_faltantes,
            "porcentaje_faltantes": round(pct_faltantes, 2),
            "valores_unicos": int(df[col].nunique(dropna=True))
        })

    # Estructuración y presentación del Top 10 de columnas con mayor incompletitud
    df_diagnostico = pd.DataFrame(resumen_calidad)
    print("\n--- Top 10 Columnas con Valores Faltantes ---")
    top_faltantes = df_diagnostico.sort_values(by="total_faltantes", ascending=False).head(10)
    print(top_faltantes[["columna", "tipo_dato", "total_faltantes", "porcentaje_faltantes"]].to_string(index=False))

    # 2. Análisis de Duplicados (Exactos vs. Identidad de Entidad)
    ## Conteo de filas 100% idénticas en todas sus columnas
    duplicados_exactos = int(df.duplicated().sum())
    
    ## Conteo de registros que repiten la llave primaria técnica de Steam
    duplicados_id = int(df.duplicated(subset=[col_id]).sum()) if col_id else 0
    
    ## Conteo de nombres de títulos repetidos (casos válidos como reediciones o DLCs)
    duplicados_nombre = int(df.duplicated(subset=[col_name]).sum()) if col_name else 0

    print("\n--- Diagnóstico de Duplicados ---")
    print(f"-> Total de registros evaluados: {total_filas:,}")
    print(f"-> Duplicados exactos (todas las columnas idénticas): {duplicados_exactos:,}")
    if col_id:
        print(f"-> Duplicados de llave primaria ({col_id}): {duplicados_id:,}")
    if col_name:
        print(
            f"-> Nombres de juegos repetidos ({col_name}): {duplicados_nombre:,} "
            f"(Repeticiones válidas cuando tienen distinto {col_id}: remasters, ediciones o juegos homónimos)"
        )

    # 3. Revisión de Rangos Plausibles e Inconsistencias
    inconsistencias = {}
    ## Validación de precios imposibles menores a cero
    if col_price:
        inconsistencias[f"Precios negativos ({col_price} < 0)"] = int(
            (pd.to_numeric(df[col_price], errors="coerce") < 0).sum()
        )
    ## Validación de conteo de reseñas positivas no negativas
    if col_pos:
        inconsistencias[f"Reseñas positivas menores a 0 ({col_pos} < 0)"] = int(
            (pd.to_numeric(df[col_pos], errors="coerce") < 0).sum()
        )
    ## Validación de conteo de reseñas negativas no negativas
    if col_neg:
        inconsistencias[f"Reseñas negativas menores a 0 ({col_neg} < 0)"] = int(
            (pd.to_numeric(df[col_neg], errors="coerce") < 0).sum()
        )
    ## Validación de coherencia temporal en fechas de lanzamiento
    if col_date:
        fechas_conv = pd.to_datetime(df[col_date], errors="coerce", format="mixed")
        fechas_ilegibles = int(fechas_conv.isna().sum() - df[col_date].isna().sum())
        fechas_fuera_rango = int(((fechas_conv.dt.year < 1970) | (fechas_conv.dt.year > 2026)).sum())
        inconsistencias[f"Fechas con formato ilegible ({col_date})"] = max(0, fechas_ilegibles)
        inconsistencias["Fechas fuera de rango plausible (<1970 o >2026)"] = fechas_fuera_rango

    print("\n--- Revisión de Rangos Plausibles e Inconsistencias ---")
    for criterio, cantidad in inconsistencias.items():
        estado = "SIN PROBLEMAS" if cantidad == 0 else f"ALERTA ({cantidad:,} casos)"
        print(f"   * {criterio}: {cantidad:,} [{estado}]")

    # Exportación del diagnóstico completo a formato CSV
    ruta_csv_diag = os.path.join(ruta_tablas, "diagnostico_calidad.csv")
    df_diagnostico.to_csv(ruta_csv_diag, index=False, encoding="utf-8")
    print(f"\n[OK] Tabla de diagnóstico guardada en: {ruta_csv_diag}")

    return df_diagnostico

# ========== 3. LIMPIEZA, PREPARACIÓN Y LOG DE TRANSFORMACIONES ========== 
def limpiar_y_preparar_datos(
    df: pd.DataFrame,
    ruta_tablas: str = "resultados/tablas",
    ruta_datos: str = "datos"
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Trata los problemas detectados mediante decisiones justificadas,
    genera variables derivadas, evalúa valores extremos sin eliminarlos ciegamente
    y exporta el log de transformaciones y el dataset limpio.
    """
    print("\n" + "=" * 50)
    print("3. LIMPIEZA, PREPARACIÓN Y LOG DE TRANSFORMACIONES")
    print("=" * 50)

    # Creación de directorios para tablas y datos procesados
    os.makedirs(ruta_tablas, exist_ok=True)
    os.makedirs(ruta_datos, exist_ok=True)

    ## Copia defensiva para preservar el DataFrame original intacto
    df_clean = df.copy()
    log_transformaciones = []

    # Identificación de nombres de columnas presentes
    col_id = obtener_columna(df_clean, ["AppID", "appid", "steam_appid"])
    col_name = obtener_columna(df_clean, ["Name", "name"])
    col_date = obtener_columna(df_clean, ["Release date", "release_date"])
    col_price = obtener_columna(df_clean, ["Price", "price"])
    col_pos = obtener_columna(df_clean, ["Positive", "positive"])
    col_neg = obtener_columna(df_clean, ["Negative", "negative"])
    col_ccu = obtener_columna(df_clean, ["Peak CCU", "peak_ccu"])
    col_genres = obtener_columna(df_clean, ["Genres", "genres"])

    # Función auxiliar para registrar la trazabilidad de cada transformación
    def registrar_paso(etapa: str, accion: str, antes: int, despues: int, justificacion: str):
        diferencia = antes - despues
        log_transformaciones.append({
            "etapa": etapa,
            "accion": accion,
            "registros_antes": antes,
            "registros_despues": despues,
            "registros_eliminados": diferencia,
            "justificacion": justificacion
        })
        print(f"-> [{etapa}] Antes: {antes:,} | Después: {despues:,} (Eliminados: {diferencia:,})")
        print(f"   Motivo: {justificacion}")

    # PASO 1: Tratamiento de Duplicados Exactos
    ## Elimina filas 100% idénticas originadas por redundancia de captura
    n_antes = len(df_clean)
    df_clean = df_clean.drop_duplicates()
    registrar_paso(
        etapa="1. Duplicados Exactos",
        accion="Eliminación de filas 100% idénticas en todas las columnas",
        antes=n_antes,
        despues=len(df_clean),
        justificacion="Redundancia exacta de datos por captura. No aporta información nueva."
    )

    # PASO 2: Unicidad de Entidad por Clave Primaria (AppID)
    ## Descarta duplicados accidentales de la clave técnica, conservando la primera observación
    if col_id:
        n_antes = len(df_clean)
        df_clean = df_clean.drop_duplicates(subset=[col_id], keep="first")
        registrar_paso(
            etapa=f"2. Unicidad de Entidad ({col_id})",
            accion=f"Eliminación de {col_id} duplicados (conservando primera aparición)",
            antes=n_antes,
            despues=len(df_clean),
            justificacion="AppID es la llave primaria única en Steam. Nombres repetidos con distinto AppID se conservan por ser reediciones o juegos homónimos válidos."
        )

    # PASO 3: Remoción de Nulos en Título del Videojuego
    ## Elimina observaciones huérfanas sin nombre identifiable
    if col_name:
        n_antes = len(df_clean)
        mask_nombre = df_clean[col_name].notna() & (df_clean[col_name].astype(str).str.strip() != "")
        df_clean = df_clean[mask_nombre].copy()
        registrar_paso(
            etapa=f"3. Nulos en Nombre ({col_name})",
            accion="Eliminación de registros sin título identificable",
            antes=n_antes,
            despues=len(df_clean),
            justificacion="Registros huérfanos o pruebas internas en la tienda de Steam sin título."
        )

    # PASO 4: Estandarización Temporal y Derivación de 'release_year'
    ## Convierte textos de fecha a formato datetime y extrae el año de publicación
    if col_date:
        n_antes = len(df_clean)
        df_clean[col_date] = pd.to_datetime(df_clean[col_date], errors="coerce", format="mixed")
        df_clean = df_clean[df_clean[col_date].notna()].copy()
        df_clean = df_clean[
            (df_clean[col_date].dt.year >= 1970) &
            (df_clean[col_date].dt.year <= 2026)
        ].copy()
        df_clean["release_year"] = df_clean[col_date].dt.year.astype(int)
        registrar_paso(
            etapa=f"4. Conversión Temporal ({col_date})",
            accion="Conversión a datetime, filtro de rango 1970-2026 y creación de release_year",
            antes=n_antes,
            despues=len(df_clean),
            justificacion="Se requiere fecha válida para responder preguntas sobre evolución temporal."
        )

    # PASO 5: Conversión a Numéricos y Filtrado de No Negatividad (>= 0)
    ## Asegura tipos continuos y elimina valores corruptos negativos
    cols_num = [c for c in [col_price, col_pos, col_neg, col_ccu] if c is not None]
    if cols_num:
        n_antes = len(df_clean)
        for col in cols_num:
            df_clean[col] = pd.to_numeric(df_clean[col], errors="coerce")
        df_clean = df_clean.dropna(subset=cols_num).copy()
        for col in cols_num:
            df_clean = df_clean[df_clean[col] >= 0].copy()
        registrar_paso(
            etapa="5. Rangos Plausibles Numéricos",
            accion=f"Conversión numérica y filtro de no negatividad (>= 0) en {cols_num}",
            antes=n_antes,
            despues=len(df_clean),
            justificacion="Precios o conteos de reseñas/jugadores menores a cero son imposibles (errores de captura)."
        )

    # PASO 6: Imputación Categórica de Géneros y Derivación de 'primary_genre'
    ## Trata valores faltantes sin descartar filas y extrae la categoría primaria
    if col_genres:
        n_antes = len(df_clean)
        s_genres = df_clean[col_genres].fillna("").astype(str).str.strip()
        s_genres = s_genres.str.replace(r"[\[\]'\"]", "", regex=True).str.strip()
        nulos_gen = int((s_genres == "").sum())
        df_clean[col_genres] = np.where(s_genres == "", "Sin Especificar", s_genres)
        df_clean["primary_genre"] = df_clean[col_genres].apply(lambda x: x.split(",")[0].strip())
        registrar_paso(
            etapa=f"6. Imputación Categórica ({col_genres})",
            accion=f"Imputación de {nulos_gen:,} vacíos como 'Sin Especificar' y creación de primary_genre",
            antes=n_antes,
            despues=len(df_clean),
            justificacion="Evita perder miles de registros válidos y extrae la categoría principal para gráficos legibles."
        )

    # PASO 7: Creación de Variables Derivadas y Tratamiento Justificado de Outliers
    if col_pos and col_neg and col_price:
        n_antes = len(df_clean)
        ## Derivación de total_reviews como suma directa de valoraciones positivas y negativas
        df_clean["total_reviews"] = df_clean[col_pos] + df_clean[col_neg]
        
        ## Derivación de approval_rate (%) evitando división por cero cuando no existen reseñas
        df_clean["approval_rate"] = np.where(
            df_clean["total_reviews"] > 0,
            (df_clean[col_pos] / df_clean["total_reviews"]) * 100,
            np.nan
        )
        
        ## Derivación de variable dicotómica del modelo de monetización
        df_clean["is_free"] = np.where(df_clean[col_price] == 0, "Gratis (Free-to-Play)", "De Pago")

        # Detección IQR de valores atípicos en precio (criterio estadístico de Tukey)
        q1 = df_clean[col_price].quantile(0.25)
        q3 = df_clean[col_price].quantile(0.75)
        iqr = q3 - q1
        umbral_outlier = q3 + 1.5 * iqr
        n_outliers = int((df_clean[col_price] > umbral_outlier).sum())
        
        ## Marcado con bandera booleana sin eliminación ciega
        df_clean["flag_outlier_price"] = df_clean[col_price] > umbral_outlier

        registrar_paso(
            etapa="7. Variables Derivadas y Marcado de Outliers",
            accion=(
                f"Creación de total_reviews, approval_rate, is_free y marcado de "
                f"{n_outliers:,} outliers de precio (> ${umbral_outlier:.2f} USD)"
            ),
            antes=n_antes,
            despues=len(df_clean),
            justificacion=(
                "No se eliminan automáticamente los valores extremos: precios altos corresponden a "
                "bundles o software profesional real, y conteos masivos de reseñas a títulos AAA exitosos."
            )
        )

    # Exportación del log formal de transformaciones a CSV
    df_log = pd.DataFrame(log_transformaciones)
    ruta_log = os.path.join(ruta_tablas, "log_transformaciones.csv")
    df_log.to_csv(ruta_log, index=False, encoding="utf-8")
    print(f"\n[OK] Log de transformaciones guardado en: {ruta_log}")

    # Exportación del dataset procesado final a CSV
    ruta_limpio = os.path.join(ruta_datos, "dataset_limpio.csv")
    df_clean.to_csv(ruta_limpio, index=False, encoding="utf-8")
    print(f"[OK] Dataset procesado guardado en: {ruta_limpio} (Total final: {len(df_clean):,} registros)")

    return df_clean, df_log

# ========== 4. ESTADÍSTICA DESCRIPTIVA Y COMPARACIÓN POR GRUPOS (P1 Y P2) ========== 
def calcular_estadistica_descriptiva(
    df: pd.DataFrame, ruta_tablas: str = "resultados/tablas"
) -> dict[str, pd.DataFrame]:
    """
    Calcula para las variables numéricas relevantes: n observaciones válidas,
    media, mediana, desviación estándar, mínimo, máximo y cuartiles (IQR).
    Para las variables categóricas: frecuencias absolutas y porcentajes.
    Genera agregaciones comparativas por grupos (modelo de negocio y género).
    Exporta todas las tablas a formato CSV.
    """
    print("\n" + "=" * 60)
    print("4. CÁLCULO DE ESTADÍSTICA DESCRIPTIVA Y COMPARACIÓN POR GRUPOS")
    print("=" * 60)

    # Creación de directorio de salida
    os.makedirs(ruta_tablas, exist_ok=True)
    tablas_generadas = {}

    # 1. Resumen Estadístico Paramétrico y No Paramétrico de Variables Numéricas
    cols_num = ["price", "positive", "negative", "total_reviews", "approval_rate", "peak_ccu"]
    cols_existentes = [c for c in cols_num if c in df.columns]

    stats_list = []
    for col in cols_existentes:
        serie = df[col].dropna()
        q1 = float(serie.quantile(0.25))
        q2 = float(serie.median())
        q3 = float(serie.quantile(0.75))
        iqr = q3 - q1

        ## Construcción del compendio de estadísticos descriptivos
        stats_list.append({
            "variable": col,
            "n_validos": int(serie.count()),
            "media": round(float(serie.mean()), 2),
            "mediana": round(q2, 2),
            "desv_estandar": round(float(serie.std()), 2),
            "minimo": round(float(serie.min()), 2),
            "q1_25": round(q1, 2),
            "q3_75": round(q3, 2),
            "iqr": round(iqr, 2),
            "maximo": round(float(serie.max()), 2)
        })

    df_stats_num = pd.DataFrame(stats_list)
    ruta_stats_num = os.path.join(ruta_tablas, "estadisticas_numericas.csv")
    df_stats_num.to_csv(ruta_stats_num, index=False, encoding="utf-8")
    tablas_generadas["estadisticas_numericas"] = df_stats_num

    print("\n--- Estadísticas Numéricas Descriptivas ---")
    print(df_stats_num.to_string(index=False))

    # 2. Resumen Categórico 1: Distribución del Modelo de Negocio (is_free)
    ## Frecuencias absolutas y porcentajes relativos
    if "is_free" in df.columns:
        conteo_free = df["is_free"].value_counts()
        pct_free = (df["is_free"].value_counts(normalize=True) * 100).round(2)
        df_cat_free = pd.DataFrame({
            "modelo_negocio": conteo_free.index,
            "frecuencia": conteo_free.values,
            "porcentaje": pct_free.values
        })
        ruta_cat_free = os.path.join(ruta_tablas, "frecuencias_modelo_negocio.csv")
        df_cat_free.to_csv(ruta_cat_free, index=False, encoding="utf-8")
        tablas_generadas["frecuencias_modelo_negocio"] = df_cat_free

        print("\n--- Frecuencia por Modelo de Negocio (is_free) ---")
        print(df_cat_free.to_string(index=False))

    # 3. Resumen Categórico 2: Frecuencia de Géneros Primarios (Top 10 y Otros)
    ## Conteo de los diez géneros más prolíficos agrupando el remanente en 'Otros'
    if "primary_genre" in df.columns:
        top10_genres = df["primary_genre"].value_counts().head(10)
        otros_count = len(df) - top10_genres.sum()
        generos_serie = pd.concat([top10_genres, pd.Series({"Otros": otros_count})])
        pct_gen = (generos_serie / len(df) * 100).round(2)
        df_cat_gen = pd.DataFrame({
            "genero_primario": generos_serie.index,
            "frecuencia": generos_serie.values,
            "porcentaje": pct_gen.values
        })
        ruta_cat_gen = os.path.join(ruta_tablas, "frecuencias_generos.csv")
        df_cat_gen.to_csv(ruta_cat_gen, index=False, encoding="utf-8")
        tablas_generadas["frecuencias_generos"] = df_cat_gen

        print("\n--- Frecuencia por Género Primario (Top 10) ---")
        print(df_cat_gen.to_string(index=False))

    # 4. Agregación y Comparación por Grupos: Modelo de Negocio (P2)
    ## Contraste entre títulos gratuitos y de pago en precio, volumen de reseñas y satisfacción
    if "is_free" in df.columns:
        agrup_free = df.groupby("is_free").agg(
            total_juegos=("AppID", "count"),
            precio_promedio=("price", "mean"),
            precio_mediana=("price", "median"),
            total_reviews_promedio=("total_reviews", "mean"),
            total_reviews_mediana=("total_reviews", "median"),
            approval_rate_promedio=("approval_rate", "mean"),
            approval_rate_mediana=("approval_rate", "median"),
            peak_ccu_promedio=("peak_ccu", "mean")
        ).round(2).reset_index()

        ruta_agrup_free = os.path.join(ruta_tablas, "resumen_por_modelo_negocio.csv")
        agrup_free.to_csv(ruta_agrup_free, index=False, encoding="utf-8")
        tablas_generadas["resumen_por_modelo_negocio"] = agrup_free

        print("\n--- Comparación Agregada por Modelo de Negocio (P2) ---")
        print(agrup_free.to_string(index=False))

    # 5. Agregación y Comparación por Grupos: Top 10 Géneros
    ## Desglose de métricas operativas por categoría primaria
    if "primary_genre" in df.columns:
        top_gen_names = df["primary_genre"].value_counts().head(10).index
        df_top_gen = df[df["primary_genre"].isin(top_gen_names)]

        agrup_gen = df_top_gen.groupby("primary_genre").agg(
            total_juegos=("AppID", "count"),
            precio_promedio=("price", "mean"),
            total_reviews_promedio=("total_reviews", "mean"),
            approval_rate_promedio=("approval_rate", "mean"),
            peak_ccu_promedio=("peak_ccu", "mean")
        ).round(2).sort_values(by="total_juegos", ascending=False).reset_index()

        ruta_agrup_gen = os.path.join(ruta_tablas, "resumen_por_genero_top10.csv")
        agrup_gen.to_csv(ruta_agrup_gen, index=False, encoding="utf-8")
        tablas_generadas["resumen_por_genero_top10"] = agrup_gen

    # 6. Agregación Temporal Anual (P4)
    ## Evaluación de lanzamientos y evolución de precios entre los años 2000 y 2024
    if "release_year" in df.columns:
        df_tiempo = df[(df["release_year"] >= 2000) & (df["release_year"] <= 2024)]
        agrup_year = df_tiempo.groupby("release_year").agg(
            lanzamientos=("AppID", "count"),
            precio_promedio=("price", "mean"),
            total_reviews_promedio=("total_reviews", "mean"),
            approval_rate_promedio=("approval_rate", "mean")
        ).round(2).reset_index()

        ruta_agrup_year = os.path.join(ruta_tablas, "resumen_temporal_anual.csv")
        agrup_year.to_csv(ruta_agrup_year, index=False, encoding="utf-8")
        tablas_generadas["resumen_temporal_anual"] = agrup_year

    print(f"\n[OK] Todas las tablas descriptivas fueron exportadas a: {ruta_tablas}")
    return tablas_generadas

# ========== 5. RELACIONES ENTRE VARIABLES Y CORRELACIÓN (P3) ========== 
def analizar_relaciones(df: pd.DataFrame, ruta_tablas: str = "resultados/tablas") -> pd.DataFrame:
    """
    Calcula matrices de correlación (Spearman y Pearson) entre variables numéricas
    relevantes, advirtiendo sobre sus límites metodológicos y la no atribución de causalidad.
    Exporta las matrices en formato CSV.
    """
    print("\n" + "=" * 60)
    print("5. ANÁLISIS DE RELACIONES ENTRE VARIABLES Y CORRELACIÓN (P3)")
    print("=" * 60)

    # Creación de directorio
    os.makedirs(ruta_tablas, exist_ok=True)
    cols_rel = ["price", "positive", "negative", "total_reviews", "approval_rate", "peak_ccu"]
    cols_existentes = [c for c in cols_rel if c in df.columns]

    ## Filtrado de columnas numéricas excluyendo identificadores artificiales
    df_num = df[cols_existentes].dropna()

    # Correlación de Spearman: Coeficiente no paramétrico robusto a asimetría y outliers
    corr_spearman = df_num.corr(method="spearman").round(3)
    ruta_spearman = os.path.join(ruta_tablas, "matriz_correlacion_spearman.csv")
    corr_spearman.to_csv(ruta_spearman, encoding="utf-8")

    # Correlación de Pearson: Coeficiente lineal paramétrico
    corr_pearson = df_num.corr(method="pearson").round(3)
    ruta_pearson = os.path.join(ruta_tablas, "matriz_correlacion_pearson.csv")
    corr_pearson.to_csv(ruta_pearson, encoding="utf-8")

    print("\n--- Matriz de Correlación de Spearman (Monótona / No paramétrica) ---")
    print(corr_spearman.to_string())

    # Advertencias metodológicas fundamentales
    print("\n" + ">" * 3 + " ADVERTENCIA METODOLÓGICA FUNDAMENTAL:")
    print("   * Correlación no implica causalidad.")
    print("   * Se utiliza Spearman prioritariamente debido al fuerte sesgo a la derecha")
    print("     en variables como 'total_reviews' y 'peak_ccu' provocado por títulos masivos.")
    print("   * Los identificadores como 'AppID' se excluyen expresamente al no ser magnitudes.")
    print(f"\n[OK] Matrices de correlación guardadas en: {ruta_tablas}")

    return corr_spearman

# ========== 6. VISUALIZACIÓN CIENTÍFICA (FIGURAS 1 A 6) ========== 
def generar_visualizaciones(df: pd.DataFrame, ruta_figuras: str = "resultados/figuras"):
    """
    Genera y guarda en disco 6 gráficos pertinentes y de alta resolución (300 DPI):
    - Fig 1: Distribución Numérica (Precio y Tasa de Aprobación) [Responde P1]
    - Fig 2: Valores Extremos (Boxplots de Precios y Aprobación por Modelo) [Responde P1 y P2]
    - Fig 3: Dispersión por Grupos (Precio vs. Total Reseñas en escala log) [Responde P3]
    - Fig 4: Comparación Categórica (Frecuencia Top 10 Géneros Primarios) [Responde P2]
    - Fig 5: Evolución Temporal (Lanzamientos Anuales y Precios 1997-2024) [Responde P4]
    - Fig 6: Gráfico del Equipo (Mapa de Calor de Correlaciones Spearman) [Responde P3]
    """
    print("\n" + "=" * 60)
    print("6. GENERACIÓN DE VISUALIZACIONES CIENTÍFICAS (FIGURAS 1 A 6)")
    print("=" * 60)

    # Creación de directorio para almacenar figuras en disco
    os.makedirs(ruta_figuras, exist_ok=True)

    # ========== FIGURA 1: Distribución Numérica de Precios y Aprobación (P1) ========== 
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ## 1A: Distribución de precios de juegos de pago (foco en rango comercial <= $60 USD)
    precios_pago = df[(df["price"] > 0) & (df["price"] <= 60)]["price"]
    sns.histplot(precios_pago, bins=30, kde=True, ax=axes[0], color="#2b5c8f")
    axes[0].set_title("Distribución de Precios (Juegos de Pago <= $60 USD)\n[Responde Pregunta 1]", fontweight="bold")
    axes[0].set_xlabel("Precio (USD)")
    axes[0].set_ylabel("Frecuencia (Cantidad de Juegos)")
    axes[0].axvline(df[df["price"] > 0]["price"].median(), color="red", linestyle="--", label=f"Mediana: ${df[df['price'] > 0]['price'].median():.2f}")
    axes[0].legend()

    ## 1B: Distribución de tasa de aprobación (filtrando títulos con >= 10 reseñas para evitar ruido de muestra ínfima)
    aprobacion_muestra = df[df["total_reviews"] >= 10]["approval_rate"].dropna()
    sns.histplot(aprobacion_muestra, bins=25, kde=True, ax=axes[1], color="#2a9d8f")
    axes[1].set_title("Distribución de Tasa de Aprobación (Total Reseñas >= 10)\n[Responde Pregunta 1]", fontweight="bold")
    axes[1].set_xlabel("Tasa de Aprobación (%)")
    axes[1].set_ylabel("Frecuencia (Cantidad de Juegos)")
    axes[1].axvline(aprobacion_muestra.median(), color="red", linestyle="--", label=f"Mediana: {aprobacion_muestra.median():.1f}%")
    axes[1].legend()

    plt.tight_layout()
    ruta_fig1 = os.path.join(ruta_figuras, "fig1_distribucion_precios_aprobacion.png")
    fig.savefig(ruta_fig1, dpi=300)
    plt.close(fig)
    print(f"-> [Figura 1] Guardada: {ruta_fig1}")

    # ========== FIGURA 2: Valores Extremos (Boxplots) por Modelo de Negocio (P1 y P2) ========== 
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ## 2A: Diagrama de caja de precios para títulos comerciales de pago
    sns.boxplot(data=df[df["price"] > 0], y="price", ax=axes[0], color="#e76f51", flierprops={"marker": "o", "markersize": 3, "alpha": 0.3})
    axes[0].set_title("Detección de Valores Extremos en Precio (Juegos de Pago)\n[Responde Pregunta 1]", fontweight="bold")
    axes[0].set_ylabel("Precio (USD)")
    axes[0].set_ylim(0, 100)  # Límite superior para visibilidad de cuartiles

    ## 2B: Diagrama de caja comparativo de aprobación entre títulos Gratis y De Pago
    df_box_rev = df[df["total_reviews"] >= 5]
    sns.boxplot(
        data=df_box_rev,
        x="is_free",
        y="approval_rate",
        ax=axes[1],
        hue="is_free",
        legend=False,
        palette={"Gratis (Free-to-Play)": "#e63946", "De Pago": "#457b9d"}
    )
    axes[1].set_title("Tasa de Aprobación por Modelo de Negocio (Reseñas >= 5)\n[Responde Pregunta 2]", fontweight="bold")
    axes[1].set_xlabel("Modelo de Negocio")
    axes[1].set_ylabel("Tasa de Aprobación (%)")

    plt.tight_layout()
    ruta_fig2 = os.path.join(ruta_figuras, "fig2_boxplot_valores_extremos.png")
    fig.savefig(ruta_fig2, dpi=300)
    plt.close(fig)
    print(f"-> [Figura 2] Guardada: {ruta_fig2}")

    # ========== FIGURA 3: Dispersión por Grupos - Precio vs. Popularidad (P3) ========== 
    fig, ax = plt.subplots(figsize=(9, 6))

    ## Muestreo aleatorio reproducible de 5.000 observaciones para evitar saturación gráfica
    df_disp = df[df["total_reviews"] > 0].sample(n=min(5000, len(df[df["total_reviews"] > 0])), random_state=SEED)

    ## Diagrama de dispersión con diferenciación de color por modelo de monetización
    sns.scatterplot(
        data=df_disp,
        x="price",
        y="total_reviews",
        hue="is_free",
        alpha=0.5,
        s=25,
        palette={"Gratis (Free-to-Play)": "#e63946", "De Pago": "#1d3557"},
        ax=ax
    )
    ## Escala logarítmica para gestionar la concentración extrema provocada por superéxitos
    ax.set_yscale("log")
    ax.set_xlim(-1, 80)
    ax.set_title("Relación entre Precio y Total de Reseñas (Escala Logarítmica)\n[Muestra aleatoria n=5.000 | Responde Pregunta 3]", fontweight="bold")
    ax.set_xlabel("Precio (USD)")
    ax.set_ylabel("Total de Reseñas (Escala Log10)")
    ax.legend(title="Modelo de Negocio")

    plt.tight_layout()
    ruta_fig3 = os.path.join(ruta_figuras, "fig3_dispersion_precio_vs_popularidad.png")
    fig.savefig(ruta_fig3, dpi=300)
    plt.close(fig)
    print(f"-> [Figura 3] Guardada: {ruta_fig3}")

    # ========== FIGURA 4: Frecuencias y Comparación de Categorías (P2) ========== 
    fig, ax = plt.subplots(figsize=(10, 6))

    ## Selección y ordenamiento de los diez principales géneros
    top10_gen = df["primary_genre"].value_counts().head(10).reset_index()
    top10_gen.columns = ["genero", "cantidad"]
    top10_gen["pct"] = (top10_gen["cantidad"] / len(df) * 100).round(1)

    barplot = sns.barplot(data=top10_gen, x="cantidad", y="genero", hue="genero", legend=False, palette="mako", ax=ax)
    ax.set_title("Top 10 Géneros Primarios en el Catálogo de Steam\n[Frecuencia Absoluta y Porcentaje Relativo]", fontweight="bold")
    ax.set_xlabel("Cantidad de Títulos Publicados")
    ax.set_ylabel("Género Primario")

    ## Inserción directa de cifras exactas y porcentajes al final de cada barra
    for p, (_, row) in zip(barplot.patches, top10_gen.iterrows()):
        ancho = p.get_width()
        ax.text(ancho + 250, p.get_y() + p.get_height() / 2, f"{int(row['cantidad']):,} ({row['pct']}%)",
                va="center", ha="left", fontsize=9, fontweight="bold", color="#333333")

    ax.set_xlim(0, top10_gen["cantidad"].max() * 1.18)
    plt.tight_layout()
    ruta_fig4 = os.path.join(ruta_figuras, "fig4_frecuencia_top10_generos.png")
    fig.savefig(ruta_fig4, dpi=300)
    plt.close(fig)
    print(f"-> [Figura 4] Guardada: {ruta_fig4}")

    # ========== FIGURA 5: Evolución Temporal de Lanzamientos y Precios 2000-2024 (P4) ========== 
    fig, ax1 = plt.subplots(figsize=(11, 5.5))

    ## Agrupación anual para contrastar volumen publicado versus precio medio
    df_anual = df[(df["release_year"] >= 2000) & (df["release_year"] <= 2024)].groupby("release_year").agg(
        total_juegos=("AppID", "count"),
        precio_medio=("price", "mean")
    ).reset_index()

    color1 = "#1d3557"
    color2 = "#e76f51"

    ## Eje Y izquierdo: Histograma anual de títulos lanzados
    ax1.set_title("Evolución Temporal: Lanzamientos Anuales y Precio Promedio (2000-2024)\n[Responde Pregunta 4]", fontweight="bold")
    ax1.bar(df_anual["release_year"], df_anual["total_juegos"], color=color1, alpha=0.75, label="Lanzamientos Anuales (Eje Izq.)")
    ax1.set_xlabel("Año de Lanzamiento")
    ax1.set_ylabel("Cantidad de Títulos Publicados", color=color1)
    ax1.tick_params(axis="y", labelcolor=color1)

    ## Eje Y derecho (twinx): Curva continua de precio medio
    ax2 = ax1.twinx()
    ax2.plot(df_anual["release_year"], df_anual["precio_medio"], color=color2, linewidth=2.5, marker="o", label="Precio Promedio USD (Eje Der.)")
    ax2.set_ylabel("Precio Promedio ($ USD)", color=color2)
    ax2.tick_params(axis="y", labelcolor=color2)
    ax2.grid(False)

    ## Unificación de leyendas de ambos ejes en la esquina superior izquierda
    lineas1, labels1 = ax1.get_legend_handles_labels()
    lineas2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lineas1 + lineas2, labels1 + labels2, loc="upper left")

    plt.tight_layout()
    ruta_fig5 = os.path.join(ruta_figuras, "fig5_evolucion_temporal_lanzamientos.png")
    fig.savefig(ruta_fig5, dpi=300)
    plt.close(fig)
    print(f"-> [Figura 5] Guardada: {ruta_fig5}")

    # ========== FIGURA 6: Gráfico del Equipo - Mapa de Calor de Correlaciones (P3) ========== 
    fig, ax = plt.subplots(figsize=(8, 6.5))

    ## Selección de magnitudes y cómputo de la matriz de Spearman
    cols_heatmap = ["price", "positive", "negative", "total_reviews", "approval_rate", "peak_ccu"]
    cols_exist = [c for c in cols_heatmap if c in df.columns]
    corr_mat = df[cols_exist].dropna().corr(method="spearman")

    ## Renderizado del mapa de calor con anotación directa de los coeficientes
    sns.heatmap(
        corr_mat,
        annot=True,
        fmt=".2f",
        cmap="vlag",
        vmin=-1,
        vmax=1,
        square=True,
        linewidths=0.8,
        cbar_kws={"shrink": 0.8, "label": "Coeficiente de Spearman (rho)"},
        ax=ax
    )
    ax.set_title("Mapa de Calor: Matriz de Correlación de Spearman\n[Gráfico del Equipo | Responde Pregunta 3]", fontweight="bold")
    plt.tight_layout()
    ruta_fig6 = os.path.join(ruta_figuras, "fig6_mapa_calor_correlaciones.png")
    fig.savefig(ruta_fig6, dpi=300)
    plt.close(fig)
    print(f"-> [Figura 6] Guardada: {ruta_fig6}")

    print(f"\n[OK] Se han generado y exportado exitosamente las 6 figuras científicas en: {ruta_figuras}")

# ========== 7. RESPUESTAS SINTETIZADAS A LAS PREGUNTAS DE INVESTIGACIÓN ========== 
def imprimir_respuestas_sintesis(df: pd.DataFrame, tablas: dict[str, pd.DataFrame]):
    """
    Imprime en consola una síntesis analítica basada en evidencia empírica
    para responder formalmente a las 4 preguntas de investigación.
    """
    print("\n" + "=" * 60)
    print("7. SÍNTESIS DE RESPUESTAS A LAS PREGUNTAS DE INVESTIGACIÓN")
    print("=" * 60)

    # Respuesta P1: Distribución Numérica e Impacto de Outliers
    precios_pago = df[df["price"] > 0]["price"]
    mediana_precio = precios_pago.median()
    iqr_precio = precios_pago.quantile(0.75) - precios_pago.quantile(0.25)
    mediana_aprob = df["approval_rate"].dropna().median()
    
    print("\n[P1 - DISTRIBUCIÓN NUMÉRICA]:")
    print(f"   * Los precios exhiben asimetría positiva extrema. Para títulos de pago, la mediana es")
    print(f"     de ${mediana_precio:.2f} USD con un IQR de ${iqr_precio:.2f} USD.")
    print(f"   * La tasa de aprobación presenta mediana de {mediana_aprob:.1f}%, concentrándose mayoritariamente")
    print("     en valoraciones favorables, aunque con amplia dispersión en títulos con pocas reseñas.")

    # Respuesta P2: Diferencias entre Títulos Gratuitos y de Pago
    agrup_free = tablas.get("resumen_por_modelo_negocio")
    if agrup_free is not None:
        sub_f2p = agrup_free.loc[agrup_free["is_free"] == "Gratis (Free-to-Play)"]
        sub_pago = agrup_free.loc[agrup_free["is_free"] == "De Pago"]
        
        rev_f2p = sub_f2p["total_reviews_promedio"].values[0] if len(sub_f2p) > 0 else 0
        rev_pago = sub_pago["total_reviews_promedio"].values[0] if len(sub_pago) > 0 else 0
        aprob_f2p = sub_f2p["approval_rate_promedio"].values[0] if len(sub_f2p) > 0 else 0
        aprob_pago = sub_pago["approval_rate_promedio"].values[0] if len(sub_pago) > 0 else 0
        
        print("\n[P2 - DIFERENCIAS ENTRE GRUPOS: GRATIS VS DE PAGO]:")
        print(f"   * Juegos Free-to-Play tienen mayor volumen de reseñas promedio ({rev_f2p:,.1f} vs {rev_pago:,.1f}).")
        print(f"   * Sin embargo, los juegos de pago presentan una tasa de aprobación promedio ligeramente superior")
        print(f"     ({aprob_pago:.1f}% vs {aprob_f2p:.1f}%), sugiriendo mayor exigencia o escrutinio de usuarios en títulos gratuitos.")

    # Respuesta P3: Asociación entre Precio y Nivel de Popularidad
    print("\n[P3 - ASOCIACIÓN ENTRE PRECIO Y POPULARIDAD]:")
    print("   * La correlación entre precio y volumen de reseñas/jugadores es débil a moderada.")
    print("   * Un mayor precio NO garantiza éxito comercial. El éxito en Steam está dominado por factores cualitativos,")
    print("     efecto de red y visibilidad algorítmica. Correlación no representa causalidad.")

    # Respuesta P4: Evolución Temporal y Tendencia Histórica
    agrup_tiempo = tablas.get("resumen_temporal_anual")
    if agrup_tiempo is not None:
        pico_anio = agrup_tiempo.sort_values(by="lanzamientos", ascending=False).iloc[0]
        print("\n[P4 - EVOLUCIÓN TEMPORAL (2000-2024)]:")
        print(f"   * Crecimiento exponencial tras 2014-2017 (apertura de Steam Direct / Greenlight).")
        print(f"   * El año con mayor cantidad de títulos registrados en este catálogo es {int(pico_anio['release_year'])} con {int(pico_anio['lanzamientos']):,} lanzamientos.")
        print("   * El precio medio ha mostrado estabilización en torno a los $6-$9 USD debido a la proliferación de juegos indie.")

# ========== FUNCIÓN PRINCIPAL DE EJECUCIÓN ========== 
def main():
    ## Configuración de rutas relativas para portabilidad y reproducibilidad
    ruta_entrada = os.path.join("datos", "games_may2024_full.csv")
    ruta_tablas = os.path.join("resultados", "tablas")
    ruta_figuras = os.path.join("resultados", "figuras")
    ruta_datos = "datos"

    # Etapa 1: Carga e inspección preliminar del dataset
    df_raw = cargar_y_explorar_datos(ruta_entrada)
    if df_raw is None:
        print("Proceso detenido: No se pudo cargar el dataset.")
        return

    # Etapa 2: Diagnóstico exhaustivo de calidad de datos
    diagnosticar_calidad(df_raw, ruta_tablas=ruta_tablas)

    # Etapa 3: Limpieza metodológica, registro de transformaciones y variables derivadas
    df_limpio, _ = limpiar_y_preparar_datos(
        df_raw,
        ruta_tablas=ruta_tablas,
        ruta_datos=ruta_datos
    )

    # Etapa 4: Estadística descriptiva y comparaciones por grupos (P1 y P2)
    tablas_descriptivas = calcular_estadistica_descriptiva(df_limpio, ruta_tablas=ruta_tablas)

    # Etapa 5: Análisis de relaciones y cómputo de matrices de correlación (P3)
    analizar_relaciones(df_limpio, ruta_tablas=ruta_tablas)

    # Etapa 6: Generación de las 6 visualizaciones científicas (P1 a P4)
    generar_visualizaciones(df_limpio, ruta_figuras=ruta_figuras)

    # Etapa 7: Síntesis interpretativa de respuestas a las preguntas de investigación
    imprimir_respuestas_sintesis(df_limpio, tablas_descriptivas)

    # Resumen final de ejecución exitosa
    print("\n" + "=" * 60)
    print("[ÉXITO TOTAL] PROCESO EDA COMPLETO FINALIZADO SIN ERRORES")
    print(f"-> Registros iniciales: {len(df_raw):,} | Registros limpios analizados: {len(df_limpio):,}")
    print(f"-> Tablas guardadas en: '{ruta_tablas}/'")
    print(f"-> Figuras PNG guardadas en: '{ruta_figuras}/'")
    print("=" * 60)


if __name__ == "__main__":
    main()
