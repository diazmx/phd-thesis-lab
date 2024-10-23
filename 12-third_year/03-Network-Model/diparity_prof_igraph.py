#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 11 14:19:25 2024

@author: ddiaz
"""

import cProfile
import pstats
import networkx as nx
import igraph as ig
import numpy as np
from scipy import integrate
import pandas as pd
from netbone.backbone import Backbone
from netbone.filters import threshold_filter, fraction_filter


def disparity(data, weight='weight'):
    
    profiler = cProfile.Profile()
    profiler.enable()  # Start profiling
    
    if isinstance(data, pd.DataFrame):
        g = ig.Graph.DataFrame(data, directed=False, use_vids=False)
    elif isinstance(data, ig.Graph):
        g = data.copy()
    else:
        print("data should be a panads dataframe or nx graph")
        profiler.disable()
        return

    strength = g.strength(weights=weight)
    
    # Asegurarse de que el atributo 'p_value' existe en las aristas
    if 'p_value' not in g.es.attributes():
        g.es['p_value'] = [None] * g.ecount()
    
    for node in range(g.vcount()):
        k = g.degree(node)
        if k>1:
            for neighbor in g.neighbors(node):
                # Obtener el peso de la arista
                eid = g.get_eid(node, neighbor)
                w = g.es[eid]['weight']
                p_ij = w / strength[node]
                alpha_ij = (1 - p_ij) ** (k - 1)

                # Actualizar p_value si ya existe, o establecer uno nuevo
                current_p_value = g.es[eid]['p_value']
                if current_p_value is not None:
                    g.es[eid]['p_value'] = min(alpha_ij, current_p_value)
                else:
                    g.es[eid]['p_value'] = alpha_ij

    profiler.disable()  # Stop profiling
    ps = pstats.Stats(profiler) # Print the profile statistics
    ps.sort_stats('cumtime').print_stats(10)  # Print the top 10 functions by cumulative time
    
    g = g.to_networkx()
    
    return Backbone(g, method_name="Disparity Filter", property_name="p_value", ascending=True, compatible_filters=[threshold_filter, fraction_filter], filter_on='Edges')
