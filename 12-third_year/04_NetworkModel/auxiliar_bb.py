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


def disparity(data, alpha_values=[0.05, 0.1, 0.15, 0.2]):
    
    strength = np.array(data.strength(weights="weight")) # Extraer fuerza de cada nodo
    k_degree = np.array(data.degree()) # Extraer el grado de cada nodo
    
    weights = np.array(data.es["weight"])  # Extraer los pesos de las aristas

    # Extraer las conexiones (índices fuente y destino de las aristas)
    edge_array = np.array(data.get_edgelist())
    sources = edge_array[:, 0]
    targets = edge_array[:, 1]

    # Obtener los grados ponderados de los nodos fuente y destino
    strength = np.array(data.strength(weights="weight"))
    ni = strength[sources]
    nj = strength[targets]

    degrees = np.array(data.degree())
    ki = degrees[sources]
    kj = degrees[targets]

    alphai = (1-(weights/ni))**(ki-1)
    alphaj = (1-(weights/nj))**(kj-1)

    p_values = np.minimum(alphai, alphaj)
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


