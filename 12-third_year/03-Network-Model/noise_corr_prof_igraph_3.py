#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 11 14:23:56 2024

@author: ddiaz
"""

import cProfile
import pstats
import pandas as pd
import numpy as np
from scipy.stats import binom
import igraph as ig
from netbone.backbone import Backbone
from netbone.filters import threshold_filter, fraction_filter

def noise_corrected(data):
    # Start profiling
    profiler = cProfile.Profile()
    profiler.enable()  

    g = data.copy()
    weights = np.array(g.es["weight"])  # Extraer los pesos del grafo

    # Sumar los pesos totales de todas las aristas
    n = np.sum(weights)

    # Extraer los índices de los nodos fuewnte y destinos de las aristas
    edge_array = np.array(g.get_edgelist())
    sources = edge_array[:, 0]
    targets = edge_array[:, 1]

    # Obtener los pesos de los nodso fuentes y destinos
    degrees = np.array(g.strength(weights="weight"))
    ni = degrees[sources]
    nj = degrees[targets]

    # Calcular las probabilidades a priori para cada arista
    mean_prior_probabilities = ((ni * nj) / n) * (1 / n)

    # Calcular los p-valores para todas las aristas
    p_values = 1 - binom.cdf(weights, n, mean_prior_probabilities)
    p_values = np.nan_to_num(p_values)  # Reemplazar NaN por 0

    # Asignar los p-valores a las aristas del grafo
    g.es["p_value"] = p_values

    # STOP profiling    
    profiler.disable()
    ps = pstats.Stats(profiler)
    ps.sort_stats('cumtime').print_stats(10)

    # COnvertir a networkx para el proceso de filtros
    g = g.to_networkx()


    # Crear y devolver el backbone del grafo
    return Backbone(
        g,
        method_name="Noise Corrected Filter",
        property_name="p_value",
        ascending=True,
        compatible_filters=[threshold_filter, fraction_filter],
        filter_on="Edges"
    )
