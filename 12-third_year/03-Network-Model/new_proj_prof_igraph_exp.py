#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 10 12:30:32 2024

@author: ddiaz

"""
import igraph as ig
import networkx as nx
import netbone as nb
from netbone.filters import threshold_filter
from netbone.measures import node_fraction, edge_fraction, average_degree, reachability, weight_fraction, density
from auxiliar_projections import apply_projection, multiply_weigt, remove_zeros
from time import time
from noise_corr_prof_igraph_3 import noise_corrected as nc1
from bb_nc_filter import noise_corrected as nc2
from diparity_prof_igraph import disparity
#from noisecorr_prof import noise_corrected
#from diparity_prof import disparity

##### **** Variables selection **** #####

dataset = "AMZ"
#filename = "../data/"+dataset+"/binet-"+dataset+"-Rw.gml"
filename = "../00-data/amz/binet-AMZ-new.gml"
#filename = "imdb/user-movie.graphml"
projection_name = "simple" 

# simple weights vector master hyperbolic resall
threshold_nodes = 2

#filename = "../data/AMZ/binet-AMZ-UN.gml"
#filename = "graphs/toy_graph.graphml"

##### END #####

##### **** Selection of dataset **** #####


alpha = [0.05, 0.1, 0.15, 0.2]

g = ig.read(filename)
user_nodes = g.vs.select(type=0)
res_nodes = g.vs.select(type=1)

#print(g.summary())
print("Is the graph bipartite?", g.is_bipartite())
print("|U|=",len(user_nodes), " \t|R|=",len(res_nodes), " \t|U|+|R|=",len(user_nodes)+len(res_nodes), "=", g.vcount())
print()

##### END #####

##### **** Projection **** #####

user_graph, rsrs_graph = apply_projection(g, projection_name)

g_toy = user_graph # Graph to analyze
print("\n##### **** Projection USERS **** #####")
print("Projection Name:", projection_name)
print("Summary\n",g_toy.summary())
#print(g_toy.es()["weight"])
print("##### END #####")

##### END #####

##### **** Preprocessing projected graph **** #####

# Scale the weight
g_toy = multiply_weigt(g_toy, 1000)

# Remove zeros to one
g_toy = remove_zeros(g_toy)

# Convert to integers
g_toy.es["weight"] = [int(x) for x in g_toy.es["weight"]]

# Convert to networkx graph
#g_toy = g_toy.to_networkx()

#print(g_toy)
#print("Avg Degree=", average_degree(g_toy, g_toy))
#print("Reachability=", reachability(g_toy, g_toy))
#print("CC=", nx.average_clustering(g_toy))
#print("Density=", density(g_toy, g_toy))
#print("Components=", len(list(nx.connected_components(g_toy))))

##### END #####


##### **** Backboning **** #####

# Disparity filter
a = time()
bb_df = disparity(g_toy)
#bb_df = nb.disparity(g_toy)
b = time() - a
print("TOP DF - time: %.10f seconds." % b)

# Noise Corrected
a = time()
bb_nc = nc1(g_toy)
#bb_nc = nb.noise_corrected(g_toy)
b = time() - a
print("TOP NC OLD - time: %.10f seconds." % b)

# Noise Corrected
a = time()
filtered_graphs = nc2(g_toy)
#bb_nc = nb.noise_corrected(g_toy)
b = time() - a
print("TOP NC NEW - time: %.10f seconds." % b)

# Convert to networkx graph
g_toy = g_toy.to_networkx()

for i_ in alpha:
    print("*** ### --- Alpha DF =", i_, " --- ### ***")
    backbone = threshold_filter(bb_df, i_)
    print(backbone)
    if len(backbone.nodes)>len(user_nodes)/threshold_nodes and len(backbone.edges) > 0:
        print("TOP-DF-"+str(i_))
        print()
print()


print("NUEVO")
for alpha__, g__ in filtered_graphs.items():
    print(f"Grafo filtrado con alpha={alpha__}: {g__.summary()}")
print("================================")
print()

for i_ in alpha:
    print("*** ### --- Alpha NC OLD =", i_, " --- ### ***")
    backbone = threshold_filter(bb_nc, i_)
    print(backbone)
    if len(backbone.nodes)>len(user_nodes)/threshold_nodes and len(backbone.edges) > 0:
        print("TOP-NC-"+str(i_))
        print()

2
print("##### ***** Done USERS ***** #####")

###############################################################################################
###############################################################################################
###############################################################################################

g_toy = rsrs_graph # Graph to analyze
print("\n##### **** Projection RESOURCE **** #####")
print("Summary\n",g_toy.summary())
print("##### END #####")

##### END #####

##### **** Preprocessing projected graph **** #####

# Scale the weight
g_toy = multiply_weigt(g_toy, 1000)
print("Done Multiplicacion")

# Remove zeros to one
g_toy = remove_zeros(g_toy)
print("Done remove zeros")

# Convert to integers
g_toy.es["weight"] = [int(x) for x in g_toy.es["weight"]]
print("Done conversion enteros")

# Convert to networkx graph
#g_toy = g_toy.to_networkx()
#print("Done grafo a networkx")

#print("Avg Degree=", average_degree(g_toy, g_toy))
#print("Reachability=", reachability(g_toy, g_toy))
#print("CC=", nx.average_clustering(g_toy))
#print("Density=", density(g_toy, g_toy))
#print("Components=", len(list(nx.connected_components(g_toy))))

##### END #####


##### **** Backboning **** #####


# Disparity filter
a = time()
bb_df = disparity(g_toy)
#bb_df = nb.disparity(g_toy)
b = time() - a
print("TOP DF - time: %.10f seconds." % b)

# Noise Corrected
a = time()
bb_nc = nc1(g_toy)
#bb_nc = nb.noise_corrected(g_toy)
b = time() - a
print("TOP NC - time: %.10f seconds." % b)

# Convert to networkx graph
g_toy = g_toy.to_networkx()

for i_ in alpha:
    print("*** ### --- Alpha DF =", i_, " --- ### ***")
    backbone = threshold_filter(bb_df, i_)
    print(backbone)
    if len(backbone.nodes)>len(user_nodes)/threshold_nodes and len(backbone.edges) > 0:
        print("TOP-DF-"+str(i_))
        print()
print()
print("================================")
print()



for i_ in alpha:
    print("*** ### --- Alpha NC =", i_, " --- ### ***")
    backbone = threshold_filter(bb_nc, i_)
    print(backbone)
    if len(backbone.nodes)>len(user_nodes)/threshold_nodes and len(backbone.edges) > 0:
        print("TOP-NC-"+str(i_))
        print()
print("##### ***** Done ***** #####")
