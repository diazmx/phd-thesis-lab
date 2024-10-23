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
from noise_corr_prof_igraph import noise_corrected
from diparity_prof_igraph import disparity

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
name_file_backboning = "top-"+dataset+"-"+projection_name
#bb = nb.disparity(g_toy, name_file_backboning)
bb_df = disparity(g_toy)
b = time() - a
print("TOP DF - time: %.10f seconds." % b)

#df_bb = bb.to_dataframe()
# Convert to networkx graph
g_toy = g_toy.to_networkx()
for i_ in alpha:
    print("*** ### --- Alpha=", i_, " --- ### ***")
    backbone = threshold_filter(bb_df, i_)
    print(backbone)
    if len(backbone.nodes)>len(user_nodes)/threshold_nodes and len(backbone.edges) > 0:
        print("TOP-DF-"+str(i_))
	#print("Node Frac",node_fraction(g_toy, backbone))
        backbone.graph["fnode"] = node_fraction(g_toy, backbone)
        #print("Edge frac",edge_fraction(g_toy, backbone))
        backbone.graph["fedge"] = edge_fraction(g_toy, backbone)
        #print("weight_fraction",weight_fraction(g_toy, backbone))
        backbone.graph["fweight"] = weight_fraction(g_toy, backbone)
        #print("Avg Degree",average_degree(g_toy, backbone))
        #print("Reachability",reachability(g_toy, backbone))
        #print("CC", nx.average_clustering(backbone))    
        #print("Density", density(g_toy, backbone))
        #print("Components", len(list(nx.connected_components(backbone))))
        print()
        #file_name = "graphs/"+dataset+"/user/"+dataset+"_user_"+projection_name+"_disparity_alpha"+str(i_)+".graphml"
        #file_name = "../00-data/"+dataset+"/top/"+dataset+"_top_"+projection_name+"_disparity_alpha"+str(i_)+".graphml"
        #file_name = dataset+"/top/"+dataset+"_top_"+projection_name+"_disparity_alpha"+str(i_)+".graphml"
        #nx.write_graphml(backbone, file_name)

print()
print("================================")
print()

# Noise Corrected
a = time()
name_file_backboning = "top-"+dataset+"-"+projection_name
#bb = noise_corrected_prof(g_toy, name_file_backboning)
bb = noise_corrected(g_toy)
b = time() - a
print("TOP NC - time: %.10f seconds." % b)
df_bb1 = bb.to_dataframe()

for i_ in alpha:
    print("*** ### --- Alpha=", i_, " --- ### ***")
    backbone = threshold_filter(bb, i_)
    print(backbone)
    if len(backbone.nodes)>len(user_nodes)/threshold_nodes and len(backbone.edges) > 0:
        print("TOP-NC-"+str(i_))
	#print("Node Frac",node_fraction(g_toy, backbone))
        backbone.graph["fnode"] = node_fraction(g_toy, backbone)
        #print("Edge frac",edge_fraction(g_toy, backbone))
        backbone.graph["fedge"] = edge_fraction(g_toy, backbone)
        #print("weight_fraction",weight_fraction(g_toy, backbone))
        backbone.graph["fweight"] = weight_fraction(g_toy, backbone)
        #print("Avg Degree",average_degree(g_toy, backbone))
        #print("Reachability",reachability(g_toy, backbone))
        #print("CC", nx.average_clustering(backbone))    
        #print("Density", density(g_toy, backbone))
        #print("Components", len(list(nx.connected_components(backbone))))
        print()
        #file_name = "graphs/"+dataset+"/user/"+dataset+"_user_"+projection_name+"_noise_alp"+str(i_)+".graphml"
        #file_name = "../00-data/"+dataset+"/top/"+dataset+"_top_"+projection_name+"_noise_alp"+str(i_)+".graphml"
        #file_name = dataset+"/top/"+dataset+"_top_"+projection_name+"_noise_alp"+str(i_)+".graphml"
        #nx.write_graphml(backbone, file_name)
    
print("##### ***** Done USERS ***** #####")

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

#print(g_toy)
#print("Avg Degree=", average_degree(g_toy, g_toy))
#print("Reachability=", reachability(g_toy, g_toy))
#print("CC=", nx.average_clustering(g_toy))
#print("Density=", density(g_toy, g_toy))
#print("Components=", len(list(nx.connected_components(g_toy))))

##### END #####


##### **** Backboning **** #####
print("Iniciando Disparity Filter")
# Disparity filter
a = time()
name_file_backboning = "bot-"+dataset+"-"+projection_name
#bb = nb.disparity(g_toy, name_file_backboning)
bb = disparity(g_toy)
b = time() - a
print("BOT DF - time: %.10f seconds." % b)
"""
df_bb = bb.to_dataframe()

for i_ in alpha:
    print("*** ### --- Alpha=", i_, " --- ### ***")
    backbone = threshold_filter(bb, i_)
    print(backbone)
    if len(backbone.nodes)>len(res_nodes)/threshold_nodes and len(backbone.edges) > 0:
        print("BOT-DF-"+str(i_))
	#print("Node Frac",node_fraction(g_toy, backbone))
        backbone.graph["fnode"] = node_fraction(g_toy, backbone)
        #print("Edge frac",edge_fraction(g_toy, backbone))
        backbone.graph["fedge"] = edge_fraction(g_toy, backbone)
        #print("weight_fraction",weight_fraction(g_toy, backbone))
        backbone.graph["fweight"] = weight_fraction(g_toy, backbone)
        #print("Avg Degree",average_degree(g_toy, backbone))
        #print("Reachability",reachability(g_toy, backbone))
        #print("CC", nx.average_clustering(backbone))    
        #print("Density", density(g_toy, backbone))
        #print("Components", len(list(nx.connected_components(backbone))))
        #print()
        #file_name = "graphs/"+dataset+"/rsrs/"+dataset+"_rsrs_"+projection_name+"_disparity_alpha"+str(i_)+".graphml"
        #file_name = "../00-data/"+dataset+"/bot/"+dataset+"_bot_"+projection_name+"_disparity_alpha"+str(i_)+".graphml"
        #file_name = dataset+"/bot/"+dataset+"_bot_"+projection_name+"_disparity_alpha"+str(i_)+".graphml"
        #nx.write_graphml(backbone, file_name)
print()
print("================================")
print()
"""

print("Iniciando Noise Corrected")
# Noise Corrected
a = time()
name_file_backboning = "bot-"+dataset+"-"+projection_name
#bb = noise_corrected_prof(g_toy, name_file_backboning)
bb = noise_corrected(g_toy)
b = time() - a
print("BOT NC - time: %.10f seconds." % b)
"""
df_bb1 = bb.to_dataframe()

for i_ in alpha:
    print("*** ### --- Alpha=", i_, " --- ### ***")
    backbone = threshold_filter(bb, i_)
    print(backbone)
    if len(backbone.nodes)>len(res_nodes)/threshold_nodes and len(backbone.edges) > 0:
        print("BOT-NC-"+str(i_))
	#print("Node Frac",node_fraction(g_toy, backbone))
        backbone.graph["fnode"] = node_fraction(g_toy, backbone)
        #print("Edge frac",edge_fraction(g_toy, backbone))
        backbone.graph["fedge"] = edge_fraction(g_toy, backbone)
        #print("weight_fraction",weight_fraction(g_toy, backbone))
        backbone.graph["fweight"] = weight_fraction(g_toy, backbone)
        #print("Avg Degree",average_degree(g_toy, backbone))
        #print("Reachability",reachability(g_toy, backbone))
        #print("CC", nx.average_clustering(backbone))    
        #print("Density", density(g_toy, backbone))
        #print("Components", len(list(nx.connected_components(backbone))))
        #print()
        #file_name = "graphs/"+dataset+"/rsrs/"+dataset+"_rsrs_"+projection_name+"_noise_alp"+str(i_)+".graphml"
        #file_name = "../00-data/"+dataset+"/bot/"+dataset+"_bot_"+projection_name+"_noise_alp"+str(i_)+".graphml"
        #file_name = dataset+"/bot/"+dataset+"_bot_"+projection_name+"_noise_alp"+str(i_)+".graphml"
        #nx.write_graphml(backbone, file_name)
"""
    
print("##### ***** Done ***** #####")
