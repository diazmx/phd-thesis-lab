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

##### **** Variables selection **** #####

DATASET = "imdb"
#filename = "../data/"+dataset+"/binet-"+dataset+"-Rw.gml"
#FILENAME = "../00-data/amz/binet-AMZ-new.gml"
FILENAME = "../bipsGraphs/imdb/user-movie.graphml"
#filename = "imdb/user-movie.graphml"
PROJ_NAME = "hyperbolic" 

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
#user_graph = apply_projection(g, PROJ_NAME, len(user_nodes), True)
user_graph, rsrs_graph = apply_projection(g, PROJ_NAME)
print("Done PROJ1")
#rsrs_graph = apply_projection(g, PROJ_NAME, len(user_nodes), False)
print("Done PROJ2")

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
#g_toy = g_toy.to_networkx()
##### END #####

###############################################################################################
###############################################################################################
###############################################################################################

##### **** Backboning **** #####

### Disparity filter ###
a = time()
bb_df = disparity(g_toy)
b = time() - a
print("TOP DF - time: %.10f seconds." % b)
for alpha__, g__ in bb_df.items():
    print(f"Grafo filtrado con alpha={alpha__}: {g__.summary()}")
    flname = "../bipsGraphs/"+DATASET+"/top/"+DATASET+"_top_"+PROJ_NAME+"_disparity_alpha"+str(alpha__)[2:]+".graphml"
    g__.write_graphml(flname)
print("================================")

### Noise Corrected ###
a = time()
bb_nc = noise_corrected(g_toy)
b = time() - a
print("TOP NC - time: %.10f seconds." % b)
for alpha__, g__ in bb_nc.items():
    print(f"Grafo filtrado con alpha={alpha__}: {g__.summary()}")
    flname = "../bipsGraphs/"+DATASET+"/top/"+DATASET+"_top_"+PROJ_NAME+"_noise_alpha"+str(alpha__)[2:]+".graphml"
    g__.write_graphml(flname)
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
#g_toy = g_toy.to_networkx()
#print("Done grafo a networkx")
##### END #####


##### **** Backboning **** #####

### Disparity filter ###
a = time()
bb_df = disparity(g_toy)
b = time() - a
print("BOT DF - time: %.10f seconds." % b)
for alpha__, g__ in bb_df.items():
    print(f"Grafo filtrado con alpha={alpha__}: {g__.summary()}")
    flname = "../bipsGraphs/"+DATASET+"/bot/"+DATASET+"_bot_"+PROJ_NAME+"_disparity_alpha"+str(alpha__)[2:]+".graphml"
    g__.write_graphml(flname)
print("================================")


# Noise Corrected
a = time()
bb_nc = noise_corrected(g_toy)
b = time() - a
print("BOT NC - time: %.10f seconds." % b)

for alpha__, g__ in bb_nc.items():
    print(f"Grafo filtrado con alpha={alpha__}: {g__.summary()}")
    flname = "../bipsGraphs/"+DATASET+"/bot/"+DATASET+"_bot_"+PROJ_NAME+"_noise_alpha"+str(alpha__)[2:]+".graphml"
    g__.write_graphml(flname)
print("================================")
print()

print("##### ***** Done RSCS ***** #####")