#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 11 14:19:25 2024

@author: ddiaz
"""

import cProfile
import pstats
import networkx as nx
import numpy as np
from scipy import integrate
import pandas as pd
from netbone.backbone import Backbone
from netbone.filters import threshold_filter, fraction_filter


def disparity(data, weight='weight'):

    g = data.copy()

    strength = g.degree(weight=weight)
    for node in g.nodes():
        k = g.degree[node]
        if k>1:
            for neighbour in g[node]:
                w = float(g[node][neighbour]['weight'])
                p_ij = w/strength[node]
                alpha_ij = (1-p_ij)**(k-1)
                if 'p_value' in g[node][neighbour]:
                    if alpha_ij < g[node][neighbour]['p_value']:
                        g[node][neighbour]['p_value'] = alpha_ij
                else:
                    g[node][neighbour]['p_value'] = alpha_ij
    
    return Backbone(g, method_name="Disparity Filter", property_name="p_value",
                    ascending=True,
                    compatible_filters=[threshold_filter, fraction_filter],
                    filter_on='Edges')

import numpy as np
import igraph as ig

def disparity(data, weight='weight', alphas=[0.05, 0.01, 0.001, 0.0001]):
    """
    Aplica el filtro de disparidad al grafo y genera subgrafos basados en diferentes valores de alpha.

    Parámetros:
    - data: Grafo de igraph con pesos en las aristas.
    - weight: Nombre del atributo de peso en las aristas.
    - alphas: Lista de valores de alpha para filtrar aristas significativas.

    Retorna:
    - Una lista de subgrafos de igraph correspondientes a los diferentes valores de alpha.
    """
    # Obtenemos el número de nodos y las aristas
    num_nodes = data.vcount()
    edges = np.array(data.get_edgelist())
    weights = np.array(data.es[weight])

    # Calculamos el strength de cada nodo (suma de los pesos de las aristas adyacentes)
    strength = np.zeros(num_nodes)
    for edge, w in zip(edges, weights):
        strength[edge[0]] += w
        strength[edge[1]] += w

    # Calculamos los p-values para cada arista
    edge_p_values = []
    for idx, (source, target) in enumerate(edges):
        w = weights[idx]
        k = data.degree(source, mode="all")  # Grado del nodo fuente
        if k > 1:
            p_ij = w / strength[source]
            alpha_ij = (1 - p_ij) ** (k - 1)
            edge_p_values.append((source, target, alpha_ij, w))

    # Convertimos a un arreglo de numpy para facilitar los cálculos
    edge_p_values = np.array(edge_p_values)

    # Iteramos sobre los valores de alpha y filtramos las aristas significativas
    graphs = []
    for alpha in alphas:
        # Filtramos aristas con p_value <= alpha
        significant_edges = edge_p_values[edge_p_values[:, 2] <= alpha]

        # Creamos un nuevo grafo con las aristas filtradas
        if len(significant_edges) > 0:
            new_graph = ig.Graph(edges=significant_edges[:, :2].astype(int).tolist(), directed=False)
            new_graph.es[weight] = significant_edges[:, 3]
            new_graph.es['p_value'] = significant_edges[:, 2]
            graphs.append(new_graph)
        else:
            # Si no hay aristas significativas, añadimos un grafo vacío
            graphs.append(ig.Graph(directed=False))

    return graphs
