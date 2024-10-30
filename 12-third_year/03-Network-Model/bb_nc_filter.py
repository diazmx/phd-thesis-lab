#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 11 14:23:56 2024

@author: ddiaz
"""

import cProfile
import pstats
import numpy as np
from scipy.stats import binom
import igraph as ig

def noise_corrected(data, alpha_values=[0.05, 0.1, 0.15, 0.2]):
    # Start profiling

    # Copia del grafo de igraph
    g = data.copy()
    weights = np.array(g.es["weight"])  # Extraer los pesos de las aristas

    # Sumar los pesos totales de todas las aristas
    n = np.sum(weights)

    # Extraer las conexiones (índices fuente y destino de las aristas)
    edge_array = np.array(g.get_edgelist())
    sources = edge_array[:, 0]
    targets = edge_array[:, 1]

    # Obtener los grados ponderados de los nodos fuente y destino
    degrees = np.array(g.strength(weights="weight"))
    ni = degrees[sources]
    nj = degrees[targets]

    # Calcular las probabilidades a priori para cada arista
    mean_prior_probabilities = ((ni * nj) / n) * (1 / n)

    # Calcular los p-valores para todas las aristas
    p_values = 1 - binom.cdf(weights, n, mean_prior_probabilities)
    p_values = np.nan_to_num(p_values)  # Reemplazar NaN por 0
    #print("B - p_values[:20] =", p_values[:50])

    # Crear un diccionario para guardar los grafos filtrados por cada alpha
    filtered_graphs = {}

    # Iterar sobre cada valor de alpha y crear un grafo filtrado
    for alpha in alpha_values:
        # Identificar las aristas que cumplen con el umbral actual
        edges_to_keep = np.where(p_values < alpha)[0]
        #edges_to_keep = p_values[p_values[property_name] < value]

        # Extraer las aristas y pesos correspondientes
        filtered_edges = edge_array[edges_to_keep]
        filtered_weights = weights[edges_to_keep]

        # Crear el nuevo grafo con las aristas filtradas
        g_filtered = ig.Graph(edges=filtered_edges, edge_attrs={"weight": filtered_weights})

        # Almacenar el grafo filtrado en el diccionario
        filtered_graphs[alpha] = g_filtered

    # Devolver el diccionario de grafos filtrados
    return filtered_graphs
