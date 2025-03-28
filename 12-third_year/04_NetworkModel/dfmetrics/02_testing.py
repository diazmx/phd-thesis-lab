#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wen Oct 30 2024

@author: ddiaz

"""
import igraph as ig
from auxiliar_bb import noise_corrected, disparity
from auxiliar_projections import apply_projection, multiply_weigt, remove_zeros
from time import time
from netbone import disparity, noise_corrected
from netbone.filters import threshold_filter

##### **** Variables selection **** #####

DATASET = "AMZ"
#filename = "../data/"+dataset+"/binet-"+dataset+"-Rw.gml"
FILENAME = "../00-data/amz/binet-AMZ-new.gml"
#filename = "imdb/user-movie.graphml"
PROJ_NAME = "simple" 

# simple weights vector master hyperbolic resall
THRESHOLD_NODES = 2

##### END #####



##### **** Selection of dataset **** #####
alpha = [0.05, 0.1, 0.15, 0.2]
g = ig.read(FILENAME)
user_nodes = g.vs.select(type=0)
res_nodes = g.vs.select(type=1)

if(g.is_bipartite()): # Check if the the graph is bipartite
    print("The graph IS bipartite")
print("|U|=",len(user_nodes), " \t|R|=",len(res_nodes), " \t|U|+|R|=",len(user_nodes)+len(res_nodes), "=", g.vcount())
print()
##### END #####



##### **** Projection **** #####
user_graph, rsrs_graph = apply_projection(g, PROJ_NAME)

g_toy = user_graph # Graph to analyze
print("\n##### **** Projection USERS **** #####")
print("Projection Name:", PROJ_NAME)
print("Summary\n",g_toy.summary())
print("##### END #####")
print()
##### END #####



##### **** Preprocessing projected graph **** #####
# Scale the weight
g_toy = multiply_weigt(g_toy, 1000)
# Remove zeros to one
g_toy = remove_zeros(g_toy)
# Convert to integers
g_toy.es["weight"] = [int(x) for x in g_toy.es["weight"]]
# Convert to networkx graph
g_toy = g_toy.to_networkx()
##### END #####



##### **** Backboning **** #####

### Disparity filter ###
a = time()
bb_df = disparity(g_toy)
b = time() - a
print("TOP DF - time: %.10f seconds." % b)
for i_ in alpha:
    print("*** ### --- Alpha DF =", i_, " --- ### ***")
    backbone = threshold_filter(bb_df, i_)
    print(backbone)
    if len(backbone.nodes)>len(user_nodes)/THRESHOLD_NODES and len(backbone.edges) > 0:
        print("TOP-DF-"+str(i_))
        print()
print("================================")

### Noise Corrected ###
a = time()
bb_nc = noise_corrected(g_toy)
b = time() - a
for i_ in alpha:
    print("*** ### --- Alpha NC OLD =", i_, " --- ### ***")
    backbone = threshold_filter(bb_nc, i_)
    print(backbone)
    if len(backbone.nodes)>len(user_nodes)/THRESHOLD_NODES and len(backbone.edges) > 0:
        print("TOP-NC-"+str(i_))
        print()
print("================================")
print()



print("##### ***** Done USERS ***** #####")

###############################################################################################
###############################################################################################
###############################################################################################

g_toy = rsrs_graph # Graph to analyze
print("\n##### **** Projection RESOURCE **** #####")
print("Summary\n",g_toy.summary())
print("##### END #####")
print()
##### END #####



##### **** Preprocessing projected graph **** #####
# Scale the weight
g_toy = multiply_weigt(g_toy, 1000)
# Remove zeros to one
g_toy = remove_zeros(g_toy)
# Convert to integers
g_toy.es["weight"] = [int(x) for x in g_toy.es["weight"]]
# Convert to networkx graph
g_toy = g_toy.to_networkx()
#print("Done grafo a networkx")
##### END #####


##### **** Backboning **** #####

### Disparity filter ###
a = time()
bb_df = disparity(g_toy)
b = time() - a
print("BOT DF - time: %.10f seconds." % b)
for i_ in alpha:
    print("*** ### --- Alpha DF =", i_, " --- ### ***")
    backbone = threshold_filter(bb_df, i_)
    print(backbone)
    if len(backbone.nodes)>len(user_nodes)/THRESHOLD_NODES and len(backbone.edges) > 0:
        print("BOT-DF-"+str(i_))
        print()
print("================================")


# Noise Corrected
a = time()
bb_nc = noise_corrected(g_toy)
b = time() - a
print("BOT NC - time: %.10f seconds." % b)
for i_ in alpha:
    print("*** ### --- Alpha NC OLD =", i_, " --- ### ***")
    backbone = threshold_filter(bb_nc, i_)
    print(backbone)
    if len(backbone.nodes)>len(user_nodes)/THRESHOLD_NODES and len(backbone.edges) > 0:
        print("BOT-NC-"+str(i_))
        print()
print("================================")
print()

print("##### ***** Done RSCS ***** #####")
