import pandas as pd
import igraph as ig
import auxiliar_path
import numpy as np
from kmodes import KModes
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.cluster.hierarchy import fcluster
import matplotlib.pyplot as plt

### Global variables

DATASET = "AMZ" # AMZ, HC, PM, UN, TOY
NODE_TYPE = True # TRUE = User      FALSE = Resources

PATH_DATASET = auxiliar_path.get_path_dataset(DATASET)
PATH_NODETYPE = auxiliar_path.get_path_topbot(NODE_TYPE)

GLOBAL_PATH = "/Users/ddiaz/Documents/code/phd-thesis-lab/"

DF_PATH = (
    GLOBAL_PATH + "12-third_year/00-Data/" + PATH_DATASET + 
    "/01-DistributionsCSV/" + DATASET + "-MOD.csv"
    )
GRAPH_PATH = (
    GLOBAL_PATH +
    "12-third_year/00-Data/01-AMZ/02-Graphs/01-Top/AMZ_top_hyperbolic_DF_alpha1.graphml"
    )

### Read GRAPH
g = ig.read(GRAPH_PATH)
#g = g.bipartite_projection(which=False)
print(g.summary(), "\n")
print("Graph info:")
print("\t|V| =", g.vcount())
print("\t|E| =", g.ecount())
print("\t d  =", g.density())

import numpy as np
from scipy.sparse import lil_matrix

# Número de aristas
m = g.ecount()

# Matriz dispersa en formato LIL (para eficiente construcción)
sim_matrix = lil_matrix((m, m))

# Precomputar vecinos de cada nodo
neighbors_dict = {v: set(g.neighbors(v)) for v in range(g.vcount())}

# Iterar solo sobre pares de aristas que comparten un nodo
for edge_i in range(m):
    ni, nj = g.es[edge_i].tuple  # Nodos de la arista i
    neis_i = neighbors_dict[ni]

    for edge_j in range(edge_i, m):  # Solo considerar j >= i para evitar cálculos redundantes
        nk, nl = g.es[edge_j].tuple  # Nodos de la arista j
        neis_j = neighbors_dict[nk]

        if ni == nk or ni == nl or nj == nk or nj == nl:  # Solo considerar aristas que compartan nodo
            intersection_size = len(neis_i & neis_j)
            union_size = len(neis_i | neis_j)
            if union_size > 0:
                jaccIdx = intersection_size / union_size
                sim_matrix[edge_i, edge_j] = jaccIdx
                sim_matrix[edge_j, edge_i] = jaccIdx  # Matriz simétrica

# Convertir a formato CSR para eficiencia en acceso
sim_matrix = sim_matrix.toarray()

Z = linkage(sim_matrix, 'single')

threshold = 2  
labels = fcluster(Z, threshold, criterion='distance')
g.es["cluster"] = labels
num_clusters = len(set(labels))
print(num_clusters)