# Paso 1: Bibliotecas necesarias
import pandas as pd
import os

def cargar_y_explorar_datos(ruta_archivo):
    print("* INICIANDO CARGA DE DATOS")
    
    # Validar si el archivo existe
    if not os.path.exists(ruta_archivo):
        print(f"Error: No se encontró el archivo en la ruta '{ruta_archivo}'.")
        print("Asegúrate de que el archivo esté en la carpeta 'datos/'.")
        return None
    try:
        df = pd.read_csv(ruta_archivo)
        print("Datos cargados exitosamente\n")
    except Exception as e:
        print(f"Error al leer el archivo: {e}")
        return None

    print("* EXPLORACIÓN INICIAL")
    filas, columnas = df.shape
    print(f"Dimensiones del dataset: {filas} filas x {columnas} columnas.\n")
    
    if filas >= 50000:
        print("VALIDACIÓN SUPERADA: El dataset tiene 50.000 o más registros.\n")
    else:
        print("ADVERTENCIA: El dataset tiene menos de 50.000 registros. :(\n")

    print("Columnas disponibles:")
    print(df.columns.tolist())
    print("\nInformación general (Tipos de datos y nulos):")
    df.info()
    
    return df

if __name__ == "__main__":
    ruta_dataset = "games_may2024_full.csv"     
    df_juegos = cargar_y_explorar_datos(ruta_dataset)