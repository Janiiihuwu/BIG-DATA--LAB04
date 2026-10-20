#Bibliotecas necesarias
import os
import pandas as pd
import numpy as np

#Semilla de reproducibilidad
SEED = 42
np.random.seed(SEED)

def obtener_columna(df: pd.DataFrame, candidatos: list[str]) -> str | None:
    """
    Función auxiliar que busca y retorna el nombre real de una columna en el DataFrame,
    independiente de si viene en mayúsculas, minúsculas o con guiones bajos.
    """
    mapa_cols = {col.lower().strip(): col for col in df.columns}
    for cand in candidatos:
        if cand.lower().strip() in mapa_cols:
            return mapa_cols[cand.lower().strip()]
    return None

#Carga y exploración inicial de datos
def cargar_y_explorar_datos(ruta_archivo: str) -> pd.DataFrame | None:
    """
    Lee los datos desde una ruta configurable, valida el mínimo de 50.000 registros,
    informa dimensiones, nombres de columnas, tipos de datos y primeras observaciones.
    """
    print("\n" + "=" * 60)
    print("1. INICIANDO CARGA E INSPECCIÓN DE DATOS")
    print("=" * 60)

    #Validar si el archivo existe
    if not os.path.exists(ruta_archivo):
        print(f"Error: No se encontró el archivo en la ruta '{ruta_archivo}'.")
        print("Asegúrate de que el archivo esté dentro de la carpeta 'datos/'.")
        return None

    try:
        #Lectura con codificación utf-8 y low_memory=False para evitar warnings de tipos mixtos
        df = pd.read_csv(ruta_archivo, encoding="utf-8", low_memory=False)
    except UnicodeDecodeError:
        print("Advertencia: Falló lectura con UTF-8. Reintentando con latin-1...")
        df = pd.read_csv(ruta_archivo, encoding="latin-1", low_memory=False)
    except Exception as e:
        print(f"Error crítico al leer el archivo CSV: {e}")
        return None

    #Informar dimensiones
    filas, columnas = df.shape
    print(f"-> Archivo cargado exitosamente desde: '{ruta_archivo}'")
    print(f"-> Dimensiones del dataset: {filas:,} filas x {columnas} columnas")

    #Validación del mínimo de 50.000 registros exigido por la pauta
    if filas < 50000:
        raise ValueError(
            f"Incumplimiento de requisito: El dataset tiene {filas:,} registros "
            f"(se requieren al menos 50.000 registros antes de la limpieza)."
        )
    else:
        print(f"-> Validación de tamaño: CUMPLE ({filas:,} >= 50,000 registros).")

    #Informar nombres de columnas y tipos de datos
    print("\n--- Columnas y Tipos de Datos ---")
    resumen_tipos = pd.DataFrame({
        "Columna": df.columns,
        "Tipo_Dato": [str(t) for t in df.dtypes.values]
    })
    print(resumen_tipos.to_string(index=False))

    #Primeras observaciones (muestra de las primeras 5 filas en columnas principales)
    cols_vista = df.columns[:8]  # Primeras 8 columnas para lectura limpia en consola
    print("\n--- Primeras 5 Observaciones (Vista Preliminar) ---")
    print(df[cols_vista].head(5).to_string())

    return df

#Diagnóstico de calidad de datos
def diagnosticar_calidad(df: pd.DataFrame, ruta_tablas: str = "resultados/tablas") -> pd.DataFrame:
    """
    Cuantifica valores faltantes por columna, duplicados exactos vs repeticiones
    válidas de entidad, tipos inconsistentes y valores fuera de rangos plausibles.
    Exporta el diagnóstico a 'resultados/tablas/diagnostico_calidad.csv'.
    """
    print("\n" + "=" * 60)
    print("2. DIAGNÓSTICO DE CALIDAD DE DATOS")
    print("=" * 60)

    os.makedirs(ruta_tablas, exist_ok=True)
    total_filas = len(df)

    #Identificar nombres exactos de columnas clave en el CSV
    col_id = obtener_columna(df, ["AppID", "appid", "steam_appid"])
    col_name = obtener_columna(df, ["Name", "name"])
    col_date = obtener_columna(df, ["Release date", "release_date"])
    col_price = obtener_columna(df, ["Price", "price"])
    col_pos = obtener_columna(df, ["Positive", "positive"])
    col_neg = obtener_columna(df, ["Negative", "negative"])

    #1. Cuantificación de valores faltantes (NA y cadenas vacías) por columna
    resumen_calidad = []
    for col in df.columns:
        nulos = int(df[col].isna().sum())
        if df[col].dtype == "object":
            vacios = int((df[col].fillna("").astype(str).str.strip() == "").sum()) - nulos
            vacios = max(0, vacios)
        else:
            vacios = 0

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

    df_diagnostico = pd.DataFrame(resumen_calidad)

    #Mostrar las 10 columnas con mayor cantidad de valores faltantes
    print("\n--- Top 10 Columnas con Valores Faltantes ---")
    top_faltantes = df_diagnostico.sort_values(by="total_faltantes", ascending=False).head(10)
    print(top_faltantes[["columna", "tipo_dato", "total_faltantes", "porcentaje_faltantes"]].to_string(index=False))

    #2. Análisis de Duplicados (Exactos vs. Repetición de Entidad)
    duplicados_exactos = int(df.duplicated().sum())
    duplicados_id = int(df.duplicated(subset=[col_id]).sum()) if col_id else 0
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

    #3. Revisión de Rangos Plausibles e Inconsistencias
    inconsistencias = {}
    if col_price:
        inconsistencias[f"Precios negativos ({col_price} < 0)"] = int(
            (pd.to_numeric(df[col_price], errors="coerce") < 0).sum()
        )
    if col_pos:
        inconsistencias[f"Reseñas positivas menores a 0 ({col_pos} < 0)"] = int(
            (pd.to_numeric(df[col_pos], errors="coerce") < 0).sum()
        )
    if col_neg:
        inconsistencias[f"Reseñas negativas menores a 0 ({col_neg} < 0)"] = int(
            (pd.to_numeric(df[col_neg], errors="coerce") < 0).sum()
        )
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

    #Exportar diagnóstico a CSV
    ruta_csv_diag = os.path.join(ruta_tablas, "diagnostico_calidad.csv")
    df_diagnostico.to_csv(ruta_csv_diag, index=False, encoding="utf-8")
    print(f"\n[OK] Tabla de diagnóstico guardada en: {ruta_csv_diag}")

    return df_diagnostico

#Limpieza, preparación y log de transformaciones
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
    print("\n" + "=" * 60)
    print("3. LIMPIEZA, PREPARACIÓN Y LOG DE TRANSFORMACIONES")
    print("=" * 60)

    os.makedirs(ruta_tablas, exist_ok=True)
    os.makedirs(ruta_datos, exist_ok=True)

    df_clean = df.copy()
    log_transformaciones = []

    #Detectar nombres reales de columnas
    col_id = obtener_columna(df_clean, ["AppID", "appid", "steam_appid"])
    col_name = obtener_columna(df_clean, ["Name", "name"])
    col_date = obtener_columna(df_clean, ["Release date", "release_date"])
    col_price = obtener_columna(df_clean, ["Price", "price"])
    col_pos = obtener_columna(df_clean, ["Positive", "positive"])
    col_neg = obtener_columna(df_clean, ["Negative", "negative"])
    col_ccu = obtener_columna(df_clean, ["Peak CCU", "peak_ccu"])
    col_genres = obtener_columna(df_clean, ["Genres", "genres"])

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

    #PASO 1: Eliminación de duplicados exactos de fila completa
    n_antes = len(df_clean)
    df_clean = df_clean.drop_duplicates()
    registrar_paso(
        etapa="1. Duplicados Exactos",
        accion="Eliminación de filas 100% idénticas en todas las columnas",
        antes=n_antes,
        despues=len(df_clean),
        justificacion="Redundancia exacta de datos por captura. No aporta información nueva."
    )

    #PASO 2: Unicidad de llave primaria (AppID), conservando nombres repetidos válidos
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

    #PASO 3: Tratamiento de nulos en el nombre del juego (Name)
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

    #PASO 4: Estandarización de fechas y derivación de variable temporal (release_year)
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

    #PASO 5: Validación de tipos numéricos y rangos plausibles (>= 0)
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

    #PASO 6: Imputación justificada en variable categórica (Genres) y género principal
    if col_genres:
        n_antes = len(df_clean)
        s_genres = df_clean[col_genres].fillna("").astype(str).str.strip()
        # Limpiar corchetes o comillas si vienen como lista en texto: "['Action', 'Indie']"
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

    #PASO 7: Variables derivadas y tratamiento documentado de Valores Extremos (Outliers)
    if col_pos and col_neg and col_price:
        n_antes = len(df_clean)
        df_clean["total_reviews"] = df_clean[col_pos] + df_clean[col_neg]
        df_clean["approval_rate"] = np.where(
            df_clean["total_reviews"] > 0,
            (df_clean[col_pos] / df_clean["total_reviews"]) * 100,
            np.nan
        )
        df_clean["is_free"] = np.where(df_clean[col_price] == 0, "Gratis (Free-to-Play)", "De Pago")

        #Cálculo de umbral de valores extremos en Precio mediante IQR
        q1 = df_clean[col_price].quantile(0.25)
        q3 = df_clean[col_price].quantile(0.75)
        iqr = q3 - q1
        umbral_outlier = q3 + 1.5 * iqr
        n_outliers = int((df_clean[col_price] > umbral_outlier).sum())
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

    # Exportar Log de Transformaciones a CSV
    df_log = pd.DataFrame(log_transformaciones)
    ruta_log = os.path.join(ruta_tablas, "log_transformaciones.csv")
    df_log.to_csv(ruta_log, index=False, encoding="utf-8")
    print(f"\n[OK] Log de transformaciones guardado en: {ruta_log}")

    #Exportar Dataset Limpio a CSV
    ruta_limpio = os.path.join(ruta_datos, "dataset_limpio.csv")
    df_clean.to_csv(ruta_limpio, index=False, encoding="utf-8")
    print(f"[OK] Dataset procesado guardado en: {ruta_limpio} (Total final: {len(df_clean):,} registros)")

    return df_clean, df_log

def main():
    """Ejecuta el flujo completo de forma secuencial y reproducible."""
    #Ruta configurable usando rutas relativas (exigido por la pauta)
    ruta_entrada = os.path.join("datos", "games_may2024_full.csv")

    #1. Carga e inspección inicial
    df_raw = cargar_y_explorar_datos(ruta_entrada)
    if df_raw is None:
        return

    #2. Diagnóstico de calidad
    df_diagnostico = diagnosticar_calidad(df_raw, ruta_tablas="resultados/tablas")

    #3. Limpieza, preparación y log de transformaciones
    df_limpio, df_log = limpiar_y_preparar_datos(
        df_raw,
        ruta_tablas="resultados/tablas",
        ruta_datos="datos"
    )

    print("\n" + "=" * 60)
    print("[ÉXITO] PROCESO DE DIAGNÓSTICO Y LIMPIEZA FINALIZADO")
    print(f"Registros iniciales: {len(df_raw):,} | Registros finales limpios: {len(df_limpio):,}")
    print("=" * 60)


if __name__ == "__main__":
    main()