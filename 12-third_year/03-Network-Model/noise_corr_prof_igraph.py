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

    # Itera sobre las aristas del grafo
    for edge in g.es:
        i, j = edge.source, edge.target  # Nodos conectados por la arista
        w = edge["weight"]  # Peso de la arista

        ni = g.strength(i, weights="weight")  # Grado ponderado del nodo i
        nj = g.strength(j, weights="weight")  # Grado ponderado del nodo j
        mean_prior_probability = ((ni * nj) / n) * (1 / n)
        kappa = n / (ni * nj)

        if approximation:
            # Calcula el valor p usando la aproximación binomial
            edge["p_value"] = 1 - binom.cdf(w, n, mean_prior_probability)
        else:
            # Calcula la puntuación (score) y otros valores si no se usa la aproximación
            score = ((kappa * w) - 1) / ((kappa * w) + 1)
            var_prior_probability = (1 / (n ** 2)) * (ni * nj * (n - ni) * (n - nj)) / ((n ** 2) * (n - 1))
            alpha_prior = (((mean_prior_probability ** 2) / var_prior_probability) * 
                          (1 - mean_prior_probability)) - mean_prior_probability
            beta_prior = (mean_prior_probability / var_prior_probability) * (1 - (mean_prior_probability ** 2)) - (1 - mean_prior_probability)

            alpha_post = alpha_prior + w
            beta_post = n - w + beta_prior
            expected_pij = alpha_post / (alpha_post + beta_post)
            variance_nij = expected_pij * (1 - expected_pij) * n
            d = (1.0 / (ni * nj)) - (n * ((ni + nj) / ((ni * nj) ** 2)))
            variance_cij = variance_nij * (((2 * (kappa + (w * d))) / (((kappa * w) + 1) ** 2)) ** 2)
            sdev_cij = variance_cij ** 0.5

            # Almacena los resultados como atributos de la arista
            edge["nc_sdev"] = sdev_cij
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
