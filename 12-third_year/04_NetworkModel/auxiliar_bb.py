#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wen Oct 30 2024

@author: ddiaz
"""

import numpy as np
from scipy.stats import binom
import igraph as ig

def noise_corrected(data, alpha_values=[0.05, 0.1, 0.15, 0.2]):

    weights = np.array(data.es["weight"])  # Extraer los pesos de las aristas

    # Sumar los pesos totales de todas las aristas
    n = np.sum(weights)

    # Extraer las conexiones (índices fuente y destino de las aristas)
    edge_array = np.array(data.get_edgelist())
    sources = edge_array[:, 0]
    targets = edge_array[:, 1]

    # Obtener los grados ponderados de los nodos fuente y destino
    degrees = np.array(data.strength(weights="weight"))
    ni = degrees[sources]
    nj = degrees[targets]

    # Calcular las probabilidades a priori para cada arista
    mean_prior_probabilities = ((ni * nj) / n) * (1 / n)

    # Calcular los p-valores para todas las aristas
    p_values = 1 - binom.cdf(weights, n, mean_prior_probabilities)
    p_values = np.nan_to_num(p_values)  # Reemplazar NaN por 0

    # Crear un diccionario para guardar los grafos filtrados por cada alpha
    filtered_graphs = {}

    # Iterar sobre cada valor de alpha y crear un grafo filtrado
    for alpha in alpha_values:
        # Identificar las aristas que cumplen con el umbral actual
        edges_to_keep = np.where(p_values < alpha)[0]

        # Extraer las aristas y pesos correspondientes
        filtered_edges = edge_array[edges_to_keep]
        filtered_weights = weights[edges_to_keep]

        # Crear el nuevo grafo con las aristas filtradas
        g_filtered = ig.Graph(edges=filtered_edges,
                              edge_attrs={"weight": filtered_weights})

        # Almacenar el grafo filtrado en el diccionario
        filtered_graphs[alpha] = g_filtered

    # Devolver el diccionario de grafos filtrados
    return filtered_graphs


def disparity(data, weight='weight', alpha_values=[0.05, 0.1, 0.15, 0.2]):

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
    filtered_graphs = {}
    for alpha in alpha_values:
        # Filtramos aristas con p_value <= alpha
        significant_edges = edge_p_values[edge_p_values[:, 2] < alpha]

        # Creamos un nuevo grafo con las aristas filtradas
        if len(significant_edges) > 0:
            new_graph = ig.Graph(edges=significant_edges[:, :2].astype(int).tolist(), directed=False)
            new_graph.es[weight] = significant_edges[:, 3]
            new_graph.es['p_value'] = significant_edges[:, 2]
            filtered_graphs[alpha] = new_graph
        else:
            # Si no hay aristas significativas, añadimos un grafo vacío
            filtered_graphs[alpha] = new_graph

    return filtered_graphs

    

