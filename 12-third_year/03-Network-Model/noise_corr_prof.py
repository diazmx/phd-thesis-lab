#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 10 12:29:12 2024

@author: ddiaz
"""

import cProfile
import pstats
from scipy.stats import binom
import pandas as pd
import networkx as nx
from netbone.backbone import Backbone
from netbone.filters import threshold_filter, fraction_filter
import time

# Define the noise_corrected function with profiling enabled
def noise_corrected_prof(data, identifier, approximation=True):
    """
    data: Pandas DataFrame or networkx graph
    identifier: String to identify output file names
    approximation: Boolean to determine whether to use approximation
    """

    profiler = cProfile.Profile()
    profiler.enable()  # Start profiling

    if isinstance(data, pd.DataFrame):
        g = nx.from_pandas_edgelist(data, edge_attr='weight', create_using=nx.Graph())
    elif isinstance(data, nx.Graph):
        g = data.copy()
    else:
        print("data should be a pandas dataframe or nx graph")
        profiler.disable()
        return

    n = sum(nx.get_edge_attributes(g, name='weight').values())
    bandera = False

    # Generate file names using the identifier
    f1_name = f"NC_time_edges_mil_{identifier}.txt"
    f2_name = f"NC_time_edges_millon_{identifier}.txt"

    # Open the time files with unique names
    f1 = open(f1_name, "w")
    f2 = open(f2_name, "w")
    
    counter = 0
    start_time = time.time()  # Start total time tracking

    for i, j, w in g.edges(data='weight'):
        counter += 1  # Increment edge counter

        if counter == 100000:
            elapsed_time = time.time() - start_time
            f1.write(f"Tiempo para 100000 aristas: {elapsed_time} segundos\n")
            f1.flush()
            f1.close()

        if counter == 1000000:
            elapsed_time = time.time() - start_time
            f2.write(f"Tiempo para 1000000 aristas: {elapsed_time} segundos\n")
            f2.flush()
            f2.close()

        ni = g.degree(i, weight='weight')
        nj = g.degree(j, weight='weight')
        mean_prior_probability = ((ni * nj) / n) * (1 / n)
        kappa = n / (ni * nj)

        if approximation:
            g[i][j]['p_value'] = 1 - binom.cdf(w, n, mean_prior_probability)
        else:
            score = ((kappa * w) - 1) / ((kappa * w) + 1)
            var_prior_probability = (1 / (n ** 2)) * (ni * nj * (n - ni) * (n - nj)) / ((n ** 2) * ((n - 1)))
            alpha_prior = (((mean_prior_probability ** 2) / var_prior_probability) * (1 - mean_prior_probability)) - mean_prior_probability
            beta_prior = (mean_prior_probability / var_prior_probability) * (1 - (mean_prior_probability ** 2)) - (1 - mean_prior_probability)

            alpha_post = alpha_prior + w
            beta_post = n - w + beta_prior
            expected_pij = alpha_post / (alpha_post + beta_post)
            variance_nij = expected_pij * (1 - expected_pij) * n
            d = (1.0 / (ni * nj)) - (n * ((ni + nj) / ((ni * nj) ** 2)))
            variance_cij = variance_nij * (((2 * (kappa + (w * d))) / (((kappa * w) + 1) ** 2)) ** 2)
            sdev_cij = variance_cij ** 0.5
            g[i][j]['nc_sdev'] = sdev_cij
            g[i][j]['score'] = score
            bandera = True

    profiler.disable()  # Stop profiling

    # Print the profile statistics
    ps = pstats.Stats(profiler)
    ps.sort_stats('cumtime').print_stats(10)  # Print the top 10 functions by cumulative time

    if bandera:
        return Backbone(g, method_name="Noise Corrected Filter", property_name="score", ascending=True, compatible_filters=[threshold_filter, fraction_filter], filter_on='Edges')
    else:
        return Backbone(g, method_name="Noise Corrected Filter", property_name="p_value", ascending=True, compatible_filters=[threshold_filter, fraction_filter], filter_on='Edges')


# Sample usage of the function with profiling
# data = pd.DataFrame(...)  # Replace this with your data
# noise_corrected(data, "identifier", approximation=True)
