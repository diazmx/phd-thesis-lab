import pandas as pd

def analizar_archivo(ruta_archivo, max_mostrar=50):
    """
    Analiza un archivo CSV o Excel y muestra valores únicos por columna
    
    Parameters:
    -----------
    ruta_archivo : str
        Ruta al archivo (CSV o Excel)
    max_mostrar : int
        Número máximo de valores únicos a mostrar
    """
    
    # Determinar tipo de archivo y cargarlo
    if ruta_archivo.endswith('.csv'):
        df = pd.read_csv(ruta_archivo)
    elif ruta_archivo.endswith(('.xlsx', '.xls')):
        df = pd.read_excel(ruta_archivo)
    else:
        print("Formato no soportado. Use CSV o Excel.")
        return
    
    print(f"\n📁 Archivo: {ruta_archivo}")
    print(f"📊 Dimensiones: {df.shape[0]} filas x {df.shape[1]} columnas")
    
    # Analizar cada columna
    for columna in df.columns:
        
        valores_unicos = df[columna].unique()
        num_unicos = len(valores_unicos)
        porcentaje = (num_unicos / len(df)) * 100
        
        print(f"\n{'─'*50}")
        print(f"Columna: {columna}")
        print(f"Tipo: {df[columna].dtype}")
        print(f"Valores únicos: {num_unicos} ({porcentaje:.1f}% del total)")
        
        if num_unicos <= max_mostrar:
            print(f"Valores: {valores_unicos[:max_mostrar]}")
        else:
            print(f"Primeros {max_mostrar} valores: {sorted(valores_unicos)[:max_mostrar]}")
            print(f"... y {num_unicos - max_mostrar} valores más")

# Ejemplo de uso:
analizar_archivo('/Users/ddiaz/Documents/code/phd-thesis-lab/12-third_year/00-Data/02-HC/00-Original/0-HC-universal.csv')