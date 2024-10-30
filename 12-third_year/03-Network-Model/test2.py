import numpy as np
from scipy.stats import binom
import igraph as ig
import pandas as pd
import networkx as nx
from netbone.backbone import Backbone
from netbone.filters import threshold_filter

def noise_corrected(data, alpha_values=[0.05, 0.1, 0.15, 0.2]):
    g = data.copy()
    weights = np.array(g.es["weight"])
    n = np.sum(weights)

    edge_array = np.array(g.get_edgelist())
    sources = edge_array[:, 0]
    targets = edge_array[:, 1]

    degrees = np.array(g.strength(weights="weight"))
    ni = degrees[sources]
    nj = degrees[targets]

    mean_prior_probabilities = ((ni * nj) / n) * (1 / n)
    p_values = 1 - binom.cdf(weights, n, mean_prior_probabilities)
    p_values = np.nan_to_num(p_values)

    # Diccionario para almacenar aristas filtradas por cada alpha
    filtered_edges_dict = {}

    for alpha in alpha_values:
        edges_to_keep = np.where(p_values <= alpha)[0]
        filtered_edges = edge_array[edges_to_keep]
        filtered_weights = weights[edges_to_keep]

        g_filtered = ig.Graph(edges=filtered_edges, edge_attrs={"weight": filtered_weights})
        filtered_edges_dict[alpha] = g_filtered

        # Imprimir las aristas filtradas para comparación
        print(f"Alpha {alpha}: {len(filtered_edges)} aristas conservadas.")
    
    return filtered_edges_dict, p_values


import networkx as nx
from netbone.backbone import Backbone
from netbone.filters import threshold_filter, fraction_filter

def compare_filters(igraph_graph, p_values, alpha, threshold_filter_func):
    # Convertir el grafo de igraph a networkx
    nx_graph = nx.Graph()
    edges = igraph_graph.get_edgelist()
    weights = igraph_graph.es["weight"]

    # Añadir aristas y pesos al grafo de NetworkX
    for (source, target), weight in zip(edges, weights):
        nx_graph.add_edge(source, target, weight=weight)

    # Crear un DataFrame con las aristas y sus p-valores
    data = pd.DataFrame(edges, columns=["source", "target"])
    data["p_value"] = p_values

    # Crear un objeto Backbone
    backbone = Backbone(
        nx_graph,
        method_name="Noise Corrected Filter",
        property_name="p_value",
        ascending=True,
        compatible_filters=[threshold_filter, fraction_filter],
        filter_on="Edges"
    )

    # Aplicar el filtro con el valor de alpha dado
    filtered_nx_graph = threshold_filter_func(backbone, value=alpha)

    # Extraer las aristas resultantes
    nx_edges = set(filtered_nx_graph.edges())
    ig_edges = set(igraph_graph.get_edgelist())

    # Comparar los resultados
    if nx_edges == ig_edges:
        print(f"Los resultados coinciden para alpha={alpha}.")
    else:
        print(f"Diferencia detectada para alpha={alpha}.")
        print(f"Aristas en igraph pero no en networkx: {ig_edges - nx_edges}")
        print(f"Aristas en networkx pero no en igraph: {nx_edges - ig_edges}")


# Crear un grafo de ejemplo
g = ig.Graph(edges=[(0, 1), (1, 2), (2, 3)], edge_attrs={"weight": [1, 2, 3]})

# Aplicar el filtro personalizado
filtered_graphs, p_values = noise_corrected(g, alpha_values=[0.05, 0.1])

# Comparar los resultados con threshold_filter
for alpha, ig_graph in filtered_graphs.items():
    compare_filters(ig_graph, p_values, alpha, threshold_filter)
