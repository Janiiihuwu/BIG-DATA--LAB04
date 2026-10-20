import pandas as pd
import os
import sys

#Configuración de rutas relativas
RUTA_DATOS = os.path.join('datos', 'dataset.csv')
RUTA_TABLAS = os.path.join('resultados', 'tablas')
RUTA_FIGURAS = os.path.join('resultados', 'figuras')

def ingesta_y_validacion(ruta, separador=',', codificacion='utf-8'):
    """
    Lee el dataset de forma configurable, captura errores de codificación/separador
    y valida el umbral mínimo de registros.
    """
    print("--- 1. INGESTA Y VALIDACIÓN DE DATOS ---")
    try:
        print(f"Intentando leer: {ruta} | Separador: '{separador}' | Codificación: {codificacion}")
        df = pd.read_csv(ruta, sep=separador, encoding=codificacion, on_bad_lines='skip')
        
        # Validación de filas
        total_filas, total_columnas = df.shape
        print(f"Dimensiones cargadas: {total_filas} filas x {total_columnas} columnas.")
        
        if total_filas < 50000:
            print("ERROR: El dataset tiene menos de 50.000 registros. No cumple los requisitos del laboratorio.")
            sys.exit(1)
        else:
            print("CHECK: Umbral de 50.000 registros superado.")
            
        print("\nTipos de datos detectados:")
        print(df.dtypes)
        return df

    except UnicodeDecodeError:
        print(f"ERROR DE CODIFICACIÓN: El archivo no pudo ser leído con '{codificacion}'. Prueba con 'latin1' o 'ISO-8859-1'.")
        sys.exit(1)
    except pd.errors.ParserError:
        print(f"ERROR DE SEPARADOR: Las columnas no coinciden con el separador '{separador}'. Revisa el formato del CSV.")
        sys.exit(1)
    except FileNotFoundError:
        print(f"ERROR DE RUTA: No se encontró el archivo en {ruta}.")
        sys.exit(1)
    except Exception as e:
        print(f"ERROR INESPERADO: {e}")
        sys.exit(1)

if __name__ == "__main__":
    os.makedirs(RUTA_TABLAS, exist_ok=True)
    os.makedirs(RUTA_FIGURAS, exist_ok=True)
    
    # Ejecución de la ingesta con parámetros configurables
    df_juegos = ingesta_y_validacion(RUTA_DATOS, separador=',', codificacion='utf-8')
