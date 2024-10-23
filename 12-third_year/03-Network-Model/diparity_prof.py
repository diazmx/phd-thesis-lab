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
    
    profiler = cProfile.Profile()
    profiler.enable()  # Start profiling
    
    if isinstance(data, pd.DataFrame):
        g = nx.from_pandas_edgelist(data, edge_attr='weight', create_using=nx.Graph())
    elif isinstance(data, nx.Graph):
        g = data.copy()
    else:
        print("data should be a panads dataframe or nx graph")
        profiler.disable()
        return

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

    profiler.disable()  # Stop profiling
    ps = pstats.Stats(profiler) # Print the profile statistics
    ps.sort_stats('cumtime').print_stats(10)  # Print the top 10 functions by cumulative time
    
    return Backbone(g, method_name="Disparity Filter", property_name="p_value", ascending=True, compatible_filters=[threshold_filter, fraction_filter], filter_on='Edges')
