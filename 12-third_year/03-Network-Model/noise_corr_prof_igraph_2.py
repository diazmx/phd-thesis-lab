#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 11 14:23:56 2024

@author: ddiaz
"""

import cProfile
import pstats
from scipy.stats import binom
import pandas as pd
import igraph as ig
from netbone.backbone import Backbone
from netbone.filters import threshold_filter, fraction_filter

# Función: noise_corrected utilizando igraph
def noise_corrected(data, approximation=True):
    profiler = cProfile.Profile()
    profiler.enable()  # Inicia la profilación

    # Crea el grafo en igraph desde un DataFrame o lo copia si ya es un grafo igraph
    if isinstance(data, pd.DataFrame):
        g = ig.Graph.DataFrame(data, directed=False)
    elif isinstance(data, ig.Graph):
        g = data.copy()
    else:
        print("El argumento 'data' debe ser un DataFrame de pandas o un grafo igraph")
        profiler.disable()  # Detiene la profilación
        return

    n = sum(g.es["weight"])  # Suma de todos los pesos de las aristas
    #strength_cache = {v: g.strength(v, weights="weight") for v in g.vs}
    strength_cache = {v["name"]: g.strength(v, weights="weight") for v in g.vs}
    # Verificar si los nodos tienen nombres personalizados
    use_names = "name" in g.vs.attributes()

    # Crear diccionario de grados ponderados según el tipo de nodo
    strength_cache = {v["name"] if use_names else v.index: g.strength(v, weights="weight") for v in g.vs}


    # Itera sobre las aristas del grafo
    # Procesar cada arista
    for edge in g.es:
        i, j = edge.source, edge.target
        w = edge["weight"]

        ni = strength_cache.get(i, 0.001)
        nj = strength_cache.get(j, 0.001)

        mean_prior_probability = ((ni * nj) / n) * (1 / n)
        kappa = n / (ni * nj)

        if approximation:
            edge["p_value"] = 1 - binom.cdf(w, n, mean_prior_probability)
        else:
            score = ((kappa * w) - 1) / ((kappa * w) + 1)
            edge["score"] = score

    profiler.disable()  # Detiene la profilación

    # Muestra las estadísticas de profilación
    ps = pstats.Stats(profiler)
    ps.sort_stats('cumtime').print_stats(10)  # Imprime las 10 funciones más costosas por tiempo acumulado
    
    g = g.to_networkx()

    # Devuelve el objeto Backbone utilizando el filtro noise corrected
    return Backbone(
        g, 
        method_name="Noise Corrected Filter", 
        property_name="p_value", 
        ascending=True, 
        compatible_filters=[threshold_filter, fraction_filter], 
        filter_on='Edges'
    )
