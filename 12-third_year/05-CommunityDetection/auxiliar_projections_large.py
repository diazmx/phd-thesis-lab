    #!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 26 16:40:32 2024

@author: ddiaz

All functions to generate the projections in bipartite graphs.
"""

import numpy as np
from scipy.spatial import distance
import os
import igraph as ig

# Optionally return the list of graphs
def get_graphs_from_directory(directory):
    graphs = []
    for filename in os.listdir(directory):
        if filename.endswith('.graphml'):
            graph = ig.Graph.Read_GraphML(os.path.join(directory, filename))
            graph['name'] = os.path.splitext(filename)[0]
            graphs.append(graph)
    return graphs

def simple_projection(bigraph, typen=True):
    if typen:
        return bigraph.bipartite_projection(which=typen)
    else:
        return bigraph.bipartite_projection(which=typen)

def weights_projection(bigraph, usr_size, typen=True):
    if not typen:
        usr_graph = bigraph.bipartite_projection(which=typen)
        weights = []
        for edge in usr_graph.es:
            # Nodos vecinos en común.
            common_neis = set(bigraph.neighbors(edge.source)) & set(bigraph.neighbors(edge.target))
            temp_sum = 0
            for node in common_neis: # Suma de cada arista con los nodos vecinos en común
                temp_sum += bigraph.es.find(_source=edge.source, _target=node)["weight"]
                temp_sum += bigraph.es.find(_source=edge.target, _target=node)["weight"]
            weights.append(temp_sum)
        usr_graph.es["weight"] = weights
        return usr_graph
    
    else:
        res_graph = bigraph.bipartite_projection(which=typen)
        weights = []
        for edge in res_graph.es:
            # El indice en el 'res_graph' se inicializa en 0, por lo tanto, en el
            # grafo bipartita ocupa la posición 'len(usr_graph)'
            source_res = edge.source + usr_size
            target_res = edge.target + usr_size
            # Nodos vecinos en común.
            common_neis = set(bigraph.neighbors(source_res)) & set(bigraph.neighbors(target_res))
            temp_sum = 0
            for node in common_neis:
                node_res = node
                temp_sum += bigraph.es.find(_source=source_res, _target=node_res)["weight"]
                temp_sum += bigraph.es.find(_source=target_res, _target=node_res)["weight"]
            weights.append(temp_sum)
        res_graph.es["weight"] = weights    
        return res_graph
    
def vectorized_projection(bigraph, usr_size, measure="euclidean", typen=True):
    adj_matrix = bigraph.get_adjacency(attribute="weight")
    if typen:
        usr_graph = bigraph.bipartite_projection(which=typen)
        weights = []
        for edge in usr_graph.es:
            # Se extraen primeros los vectores de cada nodo a comparar.
            # La dimensión del vector es igual a la suma de los nodos, por lo tanto,
            # es necesario hacer el recorte el cual tome en cuenta solo los nodos
            # vecinos. Para lograr lo anterior se usa el ' [usr_graph.vcount():] '
            vector_source = adj_matrix[edge.source][usr_size:]
            vector_target = adj_matrix[edge.target][usr_size:]
            
            temp_weight = 0
            if measure == "cosine": # No tocar
                temp_weight = distance.cosine(vector_source, vector_target)
            elif measure == "hamming":
                temp_weight = distance.hamming(vector_source, vector_target)
            elif measure == "euclidean":
                temp_weight = distance.euclidean(vector_source, vector_target)
            weights.append(temp_weight)
        usr_graph.es["weight"] = weights
        return usr_graph
    
    else:
        res_graph = bigraph.bipartite_projection(which=typen)
        weights = []
        for edge in res_graph.es:
            # En la parte de recursos, se le asigna el numero de usuarios para que
            # logre sincronizarse con los indices originales
            vector_source = adj_matrix[edge.source+usr_size][:usr_size]
            vector_target = adj_matrix[edge.target+usr_size][:usr_size]
            
            temp_weight = 0
            if measure == "cosine":
                temp_weight = distance.cosine(vector_source, vector_target)
            elif measure == "hamming":
                temp_weight = distance.hamming(vector_source, vector_target)
            elif measure == "euclidean":
                temp_weight = distance.euclidean(vector_source, vector_target)
            weights.append(temp_weight)
        res_graph.es["weight"] = weights
        return res_graph

def maestria_projection(bigraph, usr_size, typen=True):
    
    if not typen:
        usr_graph = bigraph.bipartite_projection(which=typen)
        weights = []
        for edge in usr_graph.es:
            temp_weight = 0
            node_source_neis = set(bigraph.neighbors(edge.source))
            node_target_neis = set(bigraph.neighbors(edge.target))
            temp_weight = len(node_source_neis & node_target_neis) ** 2
            temp_weight = temp_weight / (len(node_source_neis) * len(node_target_neis))
            weights.append(temp_weight)
        usr_graph.es["weight"] = weights
        return usr_graph
    else:
        res_graph = bigraph.bipartite_projection(which=typen)
        weights = []
        for edge in res_graph.es:
            temp_weight = 0 
            node_source_neis = set(bigraph.neighbors(edge.source+usr_size))
            node_target_neis = set(bigraph.neighbors(edge.target+usr_size))
            temp_weight = len(node_source_neis & node_target_neis) ** 2
            temp_weight = temp_weight / (len(node_source_neis) * len(node_target_neis))
            weights.append(temp_weight)
        res_graph.es["weight"] = weights    
        return res_graph
 
def maestria_projection_weights(bigraph):
    usr_graph, res_graph = bigraph.bipartite_projection(which="both")
    
    adj_matrix = bigraph.get_adjacency(attribute="weight")
    
    weights = []
    for edge in usr_graph.es:
        temp_weight = 0
        
        vector_source = adj_matrix[edge.source][usr_graph.vcount():]
        vector_target = adj_matrix[edge.target][usr_graph.vcount():]
        
        for i in range(len(vector_source)):
            if vector_source[i] > 0 and vector_target[i] > 0:
                temp_weight += vector_source[i] + vector_target[i]
                
        temp_weight = temp_weight / (sum(vector_source) + sum(vector_target))
        weights.append(temp_weight)
    usr_graph.es["weight"] = weights
    
    weights = []
    for edge in res_graph.es:
        temp_weight = 0
        vector_source = adj_matrix[edge.source+usr_graph.vcount()][:usr_graph.vcount()]
        vector_target = adj_matrix[edge.target+usr_graph.vcount()][:usr_graph.vcount()]
        
        for i in range(len(vector_source)):
            if vector_source[i] != 0 and vector_target[i] != 0:
                temp_weight += vector_source[i] + vector_target[i]
        
        temp_weight = temp_weight / (sum(vector_source) + sum(vector_target))
        weights.append(temp_weight)
    res_graph.es["weight"] = weights
    
    return usr_graph, res_graph

def hyperbolic_projection(bigraph, usr_size, typen=True):
    
    if not typen:
        usr_graph = bigraph.bipartite_projection(which=typen)
        weights = []
        for edge in usr_graph.es:
            temp_weight = 0
            common_neis = set(bigraph.neighbors(edge.source)) & set(bigraph.neighbors(edge.target))
            for z in common_neis:
                temp_weight += (1 / bigraph.vs[z].degree())
            weights.append(temp_weight)
        usr_graph.es["weight"] = weights
        return usr_graph
    else:
        res_graph = bigraph.bipartite_projection(which=typen)
        weights = []
        for edge in res_graph.es:
            temp_weight = 0
            common_neis = set(bigraph.neighbors(edge.source+usr_size)) & set(bigraph.neighbors(edge.target+usr_size))
            for z in common_neis:
                temp_weight += (1 / bigraph.vs[z].degree())
            weights.append(temp_weight)
        res_graph.es["weight"] = weights        
        return res_graph

def hyperbolic_projection_weights(bigraph):
    usr_graph, res_graph = bigraph.bipartite_projection(which="both")
    
    weights = []
    for edge in usr_graph.es:
        temp_weight = 0
        common_neis = set(bigraph.neighbors(edge.source)) & set(bigraph.neighbors(edge.target))
        for z in common_neis:
            temp_weight += (1 / bigraph.vs[z].degree())
        weights.append(temp_weight)
    usr_graph.es["weight"] = weights
    
    weights = []
    for edge in res_graph.es:
        temp_weight = 0
        common_neis = set(bigraph.neighbors(edge.source+usr_graph.vcount())) & set(bigraph.neighbors(edge.target+usr_graph.vcount()))
        print(common_neis)
        for z in common_neis:
            temp_weight += (1 / bigraph.vs[z].degree())
        weights.append(temp_weight)
    res_graph.es["weight"] = weights
    
    return usr_graph, res_graph
    
def resource_allocation_projection(bigraph, usr_size, typen=True):
    
    if not typen:
        usr_graph = bigraph.bipartite_projection(which=typen)
        weights = []
        for edge in usr_graph.es:
            temp_weight_u = 0
            temp_weight_v = 0
            common_neis = set(bigraph.neighbors(edge.source)) & set(bigraph.neighbors(edge.target))
            for z in common_neis:
                temp_weight_u += (1 / (bigraph.vs[z].degree()*bigraph.vs[edge.source].degree()))
                temp_weight_v += (1 / (bigraph.vs[z].degree()*bigraph.vs[edge.target].degree()))
            temp_weight = (temp_weight_u + temp_weight_v) / 2
            weights.append(temp_weight)
        usr_graph.es["weight"] = weights
        return usr_graph
    else:
        res_graph = bigraph.bipartite_projection(which=typen)
        weights = []
        for edge in res_graph.es:
            temp_weight_u = 0
            temp_weight_v = 0
            common_neis = set(bigraph.neighbors(edge.source)) & set(bigraph.neighbors(edge.target))
            for z in common_neis:
                temp_weight_u += (1 / (bigraph.vs[z].degree()*bigraph.vs[edge.source].degree()))
                temp_weight_v += (1 / (bigraph.vs[z].degree()*bigraph.vs[edge.target].degree()))
            temp_weight = (temp_weight_u + temp_weight_v) / 2
            weights.append(temp_weight)
        res_graph.es["weight"] = weights
        return res_graph
   
def resource_allocation_projection_weights(bigraph):
    usr_graph, res_graph = bigraph.bipartite_projection(which="both")
    
    weights = []
    for edge in usr_graph.es:
        temp_weight_u = 0
        temp_weight_v = 0
        common_neis = set(bigraph.neighbors(edge.source)) & set(bigraph.neighbors(edge.target))
        for z in common_neis:
            temp_weight_u += (edge["weight"] / (bigraph.vs[z].degree()*bigraph.vs[edge.source].degree()))
            temp_weight_v += (edge["weight"] / (bigraph.vs[z].degree()*bigraph.vs[edge.target].degree()))
        temp_weight = (temp_weight_u + temp_weight_v) / 2
        weights.append(temp_weight)
    usr_graph.es["weight"] = weights
        
    weights = []
    for edge in res_graph.es:
        temp_weight_u = 0
        temp_weight_v = 0
        common_neis = set(bigraph.neighbors(edge.source+usr_graph.vcount())) & set(bigraph.neighbors(edge.target+usr_graph.vcount()))
        for z in common_neis:
            temp_weight_u += (edge["weight"] / (bigraph.vs[z].degree()*bigraph.vs[edge.source+usr_graph.vcount()].degree()))
            temp_weight_v += (edge["weight"] / (bigraph.vs[z].degree()*bigraph.vs[edge.target+usr_graph.vcount()].degree()))
        temp_weight = (temp_weight_u + temp_weight_v) / 2
        weights.append(temp_weight)
    res_graph.es["weight"] = weights
    
    return usr_graph, res_graph

def remove_zeros(g):    
    g_copy = g.copy()
    for edge in g_copy.es:
        if edge["weight"] <= 0 or np.isnan(edge["weight"]):
            edge["weight"] = 1
    return g_copy

def multiply_weigt(graph_, beta):
    g_copy = graph_.copy()
    g_copy.es["weight"] = np.array(g_copy.es["weight"]) * beta
    return g_copy

def apply_projection(bigraph, projection_name, usr_size, typenn=True):
    """
    Apply the function based on the name of projection

    Parameters
    ----------
    bigraph : igraph.Graph
        Bipartite graph to apply projection.
    projection_name : str
        Name of the projection to apply.

    Returns
    -------
    None.

    """
    if projection_name == "simple":
        return simple_projection(bigraph, typen=typenn)
    elif projection_name == "weights":
        return weights_projection(bigraph, usr_size, typen=typenn)
    elif projection_name == "vector":
        return vectorized_projection(bigraph, usr_size, typen=typenn)
    elif projection_name == "master":
        return maestria_projection(bigraph, usr_size, typen=typenn)
    elif projection_name == "hyperbolic":
        return hyperbolic_projection(bigraph, usr_size, typen=typenn)
    elif projection_name == "resall":
        return resource_allocation_projection(bigraph, usr_size, typen=typenn)
    else:
        print("The name of the projection doesn't exist.")
    