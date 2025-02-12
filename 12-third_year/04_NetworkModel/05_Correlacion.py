import igraph as ig
import pandas as pd
import glob
import os
import seaborn as sns
import matplotlib.pyplot as plt


##### **** Variables selection **** #####
DATASET = "AMZ"
DATASET_PATH = "01-AMZ"

DIRPATH = "../00-Data/"+DATASET_PATH+"/02-Graphs/01-Top/"
DIRPATH = "../00-Data/"+DATASET_PATH+"/02-Graphs/02-Bot/"

# Ruta de la carpeta que contiene los archivos .graphml
archivos = glob.glob(os.path.join(DIRPATH, "*.graphml"))

# Lista para almacenar las métricas de cada grafo
metricas = []

# Función para calcular las métricas de un grafo
def calcular_metricas(grafo):
    densidad = grafo.density()
    modularidad = grafo.community_multilevel().modularity
    grado_promedio = sum(grafo.degree()) / grafo.vcount() if grafo.vcount() > 0 else 0
    # Si el grafo contiene una o más 
    if 
    distancia_promedio = grafo.average_path_length()
    #if grafo.is_connected():
    #    distancia_promedio = grafo.average_path_length()
    #else:
    #    # Encontrar el componente más grande
    #    componentes = grafo.decompose()
    #    componente_mas_grande = max(componentes, key=lambda c: c.vcount())
    #    distancia_promedio = componente_mas_grande.average_path_length()
    num_componentes = len(grafo.components())
    coeficiente_clustering = grafo.transitivity_undirected()
    
    return {
        "Densidad": densidad,
        "Modularidad": modularidad,
        "Grado Promedio": grado_promedio,
        "Distancia Promedio": distancia_promedio,
        "Número de Componentes": num_componentes,
        "Coeficiente de Clustering": coeficiente_clustering
    }

# Procesar cada archivo .graphml
for archivo in archivos:
    g = ig.Graph.Read_GraphML(archivo)
    nombre_grafo = os.path.basename(archivo)
    metrica = calcular_metricas(g)
    metrica["Grafo"] = nombre_grafo
    metricas.append(metrica)

# Crear un DataFrame con las métricas
df_metricas = pd.DataFrame(metricas)

# Mostrar estadísticas descriptivas
print(df_metricas.describe())

# Excluir la columna no numérica ('Grafo') antes de calcular la correlación
correlacion = df_metricas.drop(columns=["Grafo"]).corr()
print("\nCorrelación entre métricas:")
print(correlacion)

df_metricas_numericas = df_metricas.drop(columns=["Grafo"])


# Visualizar la matriz de correlación
plt.figure(figsize=(10, 6))
sns.heatmap(correlacion, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
plt.title("Correlación entre métricas de grafos")
plt.show()

from sklearn.preprocessing import MinMaxScaler

# Normalizar todas las columnas numéricas al rango [0, 1]
scaler = MinMaxScaler()
df_normalizado = pd.DataFrame(scaler.fit_transform(df_metricas_numericas), 
                               columns=df_metricas_numericas.columns)

# Visualizar estadísticas normalizadas
plt.figure(figsize=(12, 6))
sns.boxplot(data=df_normalizado, palette='Set3', showmeans=True, 
             meanprops={"marker": "o", "markerfacecolor": "red", "markeredgecolor": "black"})
plt.title("Distribución estadística de métricas de grafos (con Media)")
plt.xticks(rotation=45)
plt.show()