#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wen Oct 30 2024

@author: ddiaz
"""

import numpy as np
from scipy.stats import binom
import igraph as ig
from time import time
import math

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
        edges_to_keep = np.where(p_values <= alpha)[0]

        # Extraer las aristas y pesos correspondientes
        filtered_edges = edge_array[edges_to_keep]
        filtered_weights = weights[edges_to_keep]

        # Crear el nuevo grafo con las aristas filtradas
        g_filtered = ig.Graph(edges=filtered_edges,
                              edge_attrs={"weight": filtered_weights})
        
        # Copiar los atributos de los nodos del grafo original al grafo filtrado
        for attr in data.vs.attributes():
            g_filtered.vs[attr] = data.vs[attr]
            
        # Almacenar el grafo filtrado en el diccionario
        filtered_graphs[alpha] = g_filtered

    # Devolver el diccionario de grafos filtrados
    return filtered_graphs


def disparity(data, alpha_values=[0.05, 0.1, 0.15, 0.2]):
    
    strength = np.array(data.strength(weights="weight")) # Extraer fuerza de cada nodo
    
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
        
        # Copiar los atributos de los nodos del grafo original al grafo filtrado
        for attr in data.vs.attributes():
            g_filtered.vs[attr] = data.vs[attr]

        # Almacenar el grafo filtrado en el diccionario
        filtered_graphs[alpha] = g_filtered

    # Devolver el diccionario de grafos filtrados
    return filtered_graphs


def escalar_pesos(grafo, remove_zeros=False):
    """Función para escalar los pesos."""
    ### Se eliminan las aristas igual a 0
    if remove_zeros:
        edges_to_remove = grafo.es.select(weight_eq=0.0)
        idx_edges_to_remove = []
        for edge_ in edges_to_remove:
            idx_edges_to_remove.append(edge_.index)
        grafo.delete_edges(idx_edges_to_remove)
        print(f"Aristas con peso 0 = {len(idx_edges_to_remove)} eliminadas")


    ### Factor de Escalado
    edges_temp = grafo.es["weight"]
    print(f"Peso máximo={max(edges_temp)} y mínimo={min(edges_temp)} en aristas: ")
    print()

    ### Determinar si el peso minimo es 0
    if min(edges_temp) == 0.0:
        # Se obtiene el minimu segundo
        min_val = sorted(set(grafo.es["weight"]))[1]
        factor_escala = math.ceil(1 / min_val)
        print("Factor de escala:", factor_escala)

    else:
        factor_escala = math.ceil(1 / min(edges_temp))
        print("Factor de escala:", factor_escala)
    
    grafo.es["weight"] = (np.array(edges_temp) * factor_escala).round().astype(int)

    if not remove_zeros:
        for edge in grafo.es:
            if edge["weight"] == 0:
                edge["weight"] = 1
    
    return grafo

def apply_backboning(graph, dataset, proj_opcion, nodetype, remove_zeros):
    g_toy = graph.copy() # Graph to analyze
    print("\n##### **** BACKBONING USERS **** #####")
    print("Projection Name:", proj_opcion)
    print("Summary\n",g_toy.summary())
    print("##### END #####")
    print()

    g_toy = escalar_pesos(g_toy, remove_zeros)
    print(f"Peso máximo={max(g_toy.es['weight'])} y mínimo={min(g_toy.es['weight'])} en aristas: ")
    print()

    ### Disparity filter ###
    a = time()
    bb_df = disparity(g_toy)
    b = time() - a
    print("TOP DF - time: %.10f seconds." % b)
    for alpha__, g__ in bb_df.items():
        print(f"Grafo filtrado con alpha={alpha__}: {g__.summary()}")        
        if nodetype == 0:
            flname = (
                dataset+"/top/"+dataset+"_top_" + proj_opcion + "_DF_a" + str(alpha__)[2:] + ".graphml"
            )
        else:
            flname = (
                dataset+"/bot/"+dataset+"_bot_" + proj_opcion + "_DF_a" + str(alpha__)[2:] + ".graphml"
            )
        g__.write_graphml(flname)
    print("================================")



    ### Noise Corrected ###
    a = time()
    bb_nc = noise_corrected(g_toy)
    b = time() - a
    print("TOP NC - time: %.10f seconds." % b)
    for alpha__, g__ in bb_nc.items():
        print(f"Grafo filtrado con alpha={alpha__}: {g__.summary()}")
        if nodetype == 0:
            flname = (
                dataset+"/top/"+dataset+"_top_" + proj_opcion + "_NC_a" + str(alpha__)[2:] + ".graphml"
            )
        else:
            flname = (
                dataset+"/bot/"+dataset+"_bot_" + proj_opcion + "_NC_a" + str(alpha__)[2:] + ".graphml"
            )
        g__.write_graphml(flname)
    print("================================")
    print()
    print("##### ***** Done BACKBONIN GUSERS ***** #####")
    ###### ****** END ****** ######
