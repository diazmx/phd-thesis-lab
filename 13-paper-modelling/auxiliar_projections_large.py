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
    #adj_matrix = bigraph.get_adjacency(attribute="weight")
    adj_matrix = bigraph.get_adjacency()
    if not typen:
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

def jaccard_projection(bigraph, usr_size, typen=True):

    if not typen:
        usr_graph = bigraph.bipartite_projection(which=typen)
        
        weights = []
        for edge in usr_graph.es:
            # Nodos vecinos en común.            
            common_neis = set(bigraph.neighbors(edge.source)) & set(bigraph.neighbors(edge.target))
            temp_weight = len(common_neis) / ( bigraph.degree(edge.source) + bigraph.degree(edge.target) - len(common_neis))
            weights.append(temp_weight)
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
            temp_weight = len(common_neis) / ( bigraph.degree(source_res) + bigraph.degree(target_res) - len(common_neis))
            weights.append(temp_weight)
        res_graph.es["weight"] = weights    
        return res_graph
    
def weighted_jaccard_projection(bigraph, usr_size, typen=True):
    """
    Realiza la proyección de un grafo bipartito calculando los pesos de las aristas
    basados en una versión ponderada del coeficiente de Jaccard, utilizando los
    pesos del grafo bipartito original.

    Args:
        bigraph (igraph.Graph): El grafo bipartito original. Debe tener un atributo
                                'type' en los vértices (True para un tipo, False para el otro)
                                y un atributo 'weight' en las aristas.
        usr_size (int): El número de nodos del primer tipo (normalmente 'usuarios').
                        Necesario para ajustar los índices en la proyección del segundo tipo.
        typen (bool): Si es True, proyecta sobre el segundo tipo de nodos.
                      Si es False, proyecta sobre el primer tipo de nodos.

    Returns:
        igraph.Graph: El grafo proyectado con los pesos de las aristas calculados.
    """

    # Asegurarse de que el grafo bipartito tiene pesos en las aristas
    if "weight" not in bigraph.es.attributes():
        raise ValueError("El grafo bipartito debe tener un atributo 'weight' en sus aristas.")
    
    # Asegurarse de que el grafo bipartito tiene un atributo 'type'
    if "type" not in bigraph.vs.attributes():
        raise ValueError("El grafo bipartito debe tener un atributo 'type' en sus vértices.")

    # Realizar la proyección bipartita estándar primero
    # igraph ya maneja la creación de la proyección de manera eficiente
    proj_graph = bigraph.bipartite_projection(which=typen)
    
    new_weights = []

    # Iterar sobre las aristas del grafo proyectado para calcular los nuevos pesos
    for edge in proj_graph.es:
        # Los índices de los nodos en proj_graph corresponden a los índices en bigraph
        # para el tipo de nodo seleccionado, pero si estamos proyectando sobre el segundo tipo
        # debemos ajustar el índice.
        
        # Obtener los nodos originales en el bigrafo.
        # En la proyección de `igraph`, los nodos en `proj_graph` corresponden
        # directamente a los nodos del `bigraph` del tipo `typen`.
        # Sin embargo, si `typen` es True (proyección del segundo tipo),
        # los índices de `edge.source` y `edge.target` en `proj_graph` son relativos
        # al subconjunto de nodos del segundo tipo. Para mapearlos de vuelta al `bigraph`,
        # necesitamos sumar `usr_size` a sus índices.
        
        u_bigraph_idx = edge.source + (usr_size if typen else 0)
        v_bigraph_idx = edge.target + (usr_size if typen else 0)

        # Encontrar los vecinos de u y v en el grafo bipartito original
        # y los pesos de las aristas que los conectan.
        
        # Conjuntos de vecinos de u y v
        neighbors_u = set(bigraph.neighbors(u_bigraph_idx))
        neighbors_v = set(bigraph.neighbors(v_bigraph_idx))

        # Nodos vecinos en común (intersección)
        common_neighbors = neighbors_u.intersection(neighbors_v)
        
        # Nodos vecinos en unión (unión)
        union_neighbors = neighbors_u.union(neighbors_v)

        # Calcular la suma de pesos para el numerador
        sum_weights_numerator = 0.0
        for common_n in common_neighbors:
            # Obtener el peso de la arista (u, common_n) en bigraph
            # y (v, common_n) en bigraph
            try:
                # Si es un grafo simple, bigraph.get_eid(u, v) funciona.
                # Si puede haber múltiples aristas, necesitamos iterar.
                # Para simplificar, asumimos una arista única entre cada par de nodos para 'get_eid'.
                # Si hay múltiples aristas entre los mismos nodos, necesitarías un enfoque más sofisticado.
                weight_ux = bigraph.es[bigraph.get_eid(u_bigraph_idx, common_n)]["weight"]
                weight_vx = bigraph.es[bigraph.get_eid(v_bigraph_idx, common_n)]["weight"]
                sum_weights_numerator += (weight_ux + weight_vx)
            except ig.InternalError:
                # Esto ocurre si no hay una arista directa (o no se encuentra el ID)
                # que no debería pasar si common_n es un vecino.
                # Es más robusto buscar la arista si hay múltiples entre los mismos nodos.
                # Para este ejemplo, asumimos get_eid funciona.
                pass # O manejar el error apropiadamente


        # Calcular la suma de pesos para el denominador
        sum_weights_denominator = 0.0
        for union_n in union_neighbors:
            # Añadir el peso de (u, union_n) si existe
            if union_n in neighbors_u:
                try:
                    weight_un = bigraph.es[bigraph.get_eid(u_bigraph_idx, union_n)]["weight"]
                    sum_weights_denominator += weight_un
                except ig.InternalError:
                    pass

            # Añadir el peso de (v, union_n) si existe
            if union_n in neighbors_v:
                try:
                    weight_vn = bigraph.es[bigraph.get_eid(v_bigraph_idx, union_n)]["weight"]
                    sum_weights_denominator += weight_vn
                except ig.InternalError:
                    pass

        # Evitar división por cero
        if sum_weights_denominator == 0:
            temp_weight = 0.0
        else:
            temp_weight = sum_weights_numerator / sum_weights_denominator
        
        new_weights.append(temp_weight)
        
    proj_graph.es["weight"] = new_weights
    return proj_graph

def maestria_projection(bigraph, usr_size, typen=True):
    
    if not typen:
        usr_graph = bigraph.bipartite_projection(which=typen)
        weights = []
        for edge in usr_graph.es:
            temp_weight = 0
            node_source_neis = set(bigraph.neighbors(edge.source))
            node_target_neis = set(bigraph.neighbors(edge.target))
            temp_weight = len(node_source_neis & node_target_neis) / 2
            temp_weight = temp_weight * ((1/len(node_source_neis)) + (1/len(node_target_neis)))
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
            temp_weight = len(node_source_neis & node_target_neis) / 2
            temp_weight = temp_weight * ((1/len(node_source_neis)) * (1/len(node_target_neis)))
            weights.append(temp_weight)
        res_graph.es["weight"] = weights    
        return res_graph
    

def maestria_projection_weights2(bigraph, usr_size, typen=True):
    """
    Calcula una proyección ponderada a partir de un grafo bipartito.

    Args:
        bigraph (igraph.Graph): El grafo bipartito de entrada con pesos en las aristas.
        usr_size (int): El número de nodos en la primera partición (ej., usuarios).
                        Se asume que los primeros 'usr_size' nodos son del primer tipo,
                        y el resto son del segundo tipo.
        typen (bool): Si es True, proyecta la segunda partición (recursos).
                      Si es False, proyecta la primera partición (usuarios).

    Returns:
        igraph.Graph: El grafo proyectado con los pesos de arista calculados.
    """
    
    # Obtenemos la matriz de adyacencia completa del grafo bipartito, incluyendo los pesos.
    # Esto se hace una sola vez al principio de la función.
    adj_matrix = bigraph.get_adjacency(attribute="weight")
    
    # Determinamos el tamaño de la segunda partición (recursos).
    # Esto es crucial para la correcta segmentación (slicing) de la matriz de adyacencia.
    res_size = bigraph.vcount() - usr_size

    if not typen: # Proyectando la primera partición (ej. usuarios)
        # bipartite_projection(which=False) proyecta la primera partición (índices 0 a usr_size-1)
        usr_graph = bigraph.bipartite_projection(which=False) 
        weights = []
        for edge in usr_graph.es:
            temp_weight = 0
            
            # Accedemos a los vectores de la matriz de adyacencia del grafo bipartito original.
            # Para los nodos de usuario (edge.source, edge.target), sus conexiones a los nodos de recurso
            # (que van desde el índice usr_size hasta usr_size + res_size - 1 en el bigraph).
            vector_source = adj_matrix[edge.source][usr_size : usr_size + res_size]
            vector_target = adj_matrix[edge.target][usr_size : usr_size + res_size]
            
            # Calculamos la suma de los pesos para los vecinos comunes.
            # La lógica es (w_uz + w_vz) para cada vecino común 'z'.
            for i in range(len(vector_source)): # Iteramos a través de la dimensión de 'recursos' (vecinos comunes)
                # Verificamos si tanto el nodo de origen como el de destino (usuarios)
                # están conectados a este nodo de recurso 'i'.
                # vector_source[i] > 0 implica que existe una arista y tiene un peso.
                if vector_source[i] > 0 and vector_target[i] > 0:
                    temp_weight += vector_source[i] + vector_target[i]
            
            # Calculamos el denominador como la suma de las "fuerzas" (grados ponderados)
            # de los nodos de origen y destino del grafo proyectado.
            # Solo consideramos las conexiones a la 'otra' partición (recursos).
            sum_source_weights = sum(vector_source)
            sum_target_weights = sum(vector_target)
            
            denominator = sum_source_weights + sum_target_weights
            
            if denominator != 0: # Evitamos la división por cero
                temp_weight = temp_weight / denominator
            else:
                temp_weight = 0 # O asignamos un valor predeterminado adecuado si no hay conexiones comunes
                
            weights.append(temp_weight)
        
        usr_graph.es["weight"] = weights
        return usr_graph
    
    else: # Proyectando la segunda partición (ej. recursos)
        # bipartite_projection(which=True) proyecta la segunda partición (índices usr_size a end)
        res_graph = bigraph.bipartite_projection(which=True) 
        weights = []
        for edge in res_graph.es:
            temp_weight = 0
            
            # Accedemos a los vectores de la matriz de adyacencia del grafo bipartito original.
            # Para los nodos de recurso (edge.source, edge.target en el grafo proyectado),
            # necesitamos sumar 'usr_size' para obtener sus índices reales en el bigraph.
            bigraph_source_idx = edge.source + usr_size
            bigraph_target_idx = edge.target + usr_size
            
            # Obtenemos sus conexiones a los nodos de usuario (que van desde el índice 0 hasta usr_size-1 en el bigraph).
            vector_source = adj_matrix[bigraph_source_idx][0 : usr_size]
            vector_target = adj_matrix[bigraph_target_idx][0 : usr_size]
            
            # Calculamos la suma de los pesos para los vecinos comunes (usuarios en este caso).
            for i in range(len(vector_source)): # Iteramos a través de la dimensión de 'usuarios' (vecinos comunes)
                # Verificamos si ambos nodos de recurso están conectados a este nodo de usuario 'i'.
                if vector_source[i] > 0 and vector_target[i] > 0:
                    temp_weight += vector_source[i] + vector_target[i]
            
            # Calculamos el denominador como la suma de las "fuerzas" de los nodos de origen y destino del grafo proyectado.
            sum_source_weights = sum(vector_source)
            sum_target_weights = sum(vector_target)
            
            denominator = sum_source_weights + sum_target_weights
            
            if denominator != 0: # Evitamos la división por cero
                temp_weight = temp_weight / denominator
            else:
                temp_weight = 0 
                
            weights.append(temp_weight)
            
        res_graph.es["weight"] = weights    
        return res_graph # Retornamos solo el res_graph, consistente con el bloque 'if'.

def proportional_weighted_projection(bigraph, usr_size, typen=True):
    """
    Realiza la proyección de un grafo bipartito calculando los pesos de las aristas
    basados en una métrica proporcional ponderada.

    Args:
        bigraph (igraph.Graph): El grafo bipartito original. Debe tener un atributo
                                'type' en los vértices (True para un tipo, False para el otro)
                                y un atributo 'weight' en las aristas.
        usr_size (int): El número de nodos del primer tipo (normalmente 'usuarios').
                        Necesario para ajustar los índices en la proyección del segundo tipo.
        typen (bool): Si es True, proyecta sobre el segundo tipo de nodos.
                      Si es False, proyecta sobre el primer tipo de nodos.

    Returns:
        igraph.Graph: El grafo proyectado con los pesos de las aristas calculados.
    """

    # Asegurarse de que el grafo bipartito tiene pesos en las aristas
    if "weight" not in bigraph.es.attributes():
        raise ValueError("El grafo bipartito debe tener un atributo 'weight' en sus aristas.")
    
    # Asegurarse de que el grafo bipartito tiene un atributo 'type'
    if "type" not in bigraph.vs.attributes():
        raise ValueError("El grafo bipartito debe tener un atributo 'type' en sus vértices.")

    # Realizar la proyección bipartita estándar primero
    proj_graph = bigraph.bipartite_projection(which=typen)
    
    new_weights = []

    # Iterar sobre las aristas del grafo proyectado para calcular los nuevos pesos
    for edge in proj_graph.es:
        # Obtener los índices de los nodos originales en el bigrafo.
        u_bigraph_idx = edge.source + (usr_size if typen else 0)
        v_bigraph_idx = edge.target + (usr_size if typen else 0)

        # 1. Encontrar los vecinos comunes (N_uv)
        neighbors_u = set(bigraph.neighbors(u_bigraph_idx))
        neighbors_v = set(bigraph.neighbors(v_bigraph_idx))
        common_neighbors = neighbors_u.intersection(neighbors_v)

        # 2. Calcular el numerador para el término de 'u': sum(w_uy) para y en N_uv
        sum_w_uy_common = 0.0
        for common_n in common_neighbors:
            try:
                # Obtener el peso de la arista (u_bigraph_idx, common_n) en bigraph
                weight_ux = bigraph.es[bigraph.get_eid(u_bigraph_idx, common_n)]["weight"]
                sum_w_uy_common += weight_ux
            except ig.InternalError:
                # Esto no debería ocurrir si common_n es un vecino
                pass
        
        # 3. Calcular el denominador para el término de 'u': sum(w_uz) para z en N_u (fuerza de u)
        sum_w_uz_total = 0.0
        for neighbor_of_u in neighbors_u:
            try:
                weight_uz = bigraph.es[bigraph.get_eid(u_bigraph_idx, neighbor_of_u)]["weight"]
                sum_w_uz_total += weight_uz
            except ig.InternalError:
                pass

        # 4. Calcular el numerador para el término de 'v': sum(w_vy) para y en N_uv
        sum_w_vy_common = 0.0
        for common_n in common_neighbors:
            try:
                # Obtener el peso de la arista (v_bigraph_idx, common_n) en bigraph
                weight_vx = bigraph.es[bigraph.get_eid(v_bigraph_idx, common_n)]["weight"]
                sum_w_vy_common += weight_vx
            except ig.InternalError:
                pass

        # 5. Calcular el denominador para el término de 'v': sum(w_vz) para z en N_v (fuerza de v)
        sum_w_vz_total = 0.0
        for neighbor_of_v in neighbors_v:
            try:
                weight_vz = bigraph.es[bigraph.get_eid(v_bigraph_idx, neighbor_of_v)]["weight"]
                sum_w_vz_total += weight_vz
            except ig.InternalError:
                pass

        # 6. Calcular el peso final de la arista (w_uv)
        term_u = 0.0
        if sum_w_uz_total > 0:
            term_u = sum_w_uy_common / sum_w_uz_total
        
        term_v = 0.0
        if sum_w_vz_total > 0:
            term_v = sum_w_vy_common / sum_w_vz_total
        
        # El peso final es el producto de los dos términos
        temp_weight = term_u * term_v
        
        new_weights.append(temp_weight)
        
    proj_graph.es["weight"] = new_weights
    return proj_graph

def maestria_projection_weights(bigraph, usr_size, typen=True):

    if not typen:

        usr_graph = bigraph.bipartite_projection(which=typen)
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
        return usr_graph
    
    else:
        res_graph = bigraph.bipartite_projection(which=typen)
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

import igraph

def hyperbolic_projection_weights2(bigraph, usr_size, typen=True):
    """
    Calculates a weighted projection from a bipartite graph using a metric
    that incorporates node 'strength' (weighted degree).

    Args:
        bigraph (igraph.Graph): The input bipartite graph with edge weights.
        usr_size (int): The number of nodes in the first partition (e.g., users).
                        Assumes the first 'usr_size' nodes are of the first type,
                        and the rest are of the second type.
        typen (bool): If True, projects the second partition (resources).
                      If False, projects the first partition (users).

    Returns:
        igraph.Graph: The projected graph (either user or resource) with calculated edge weights.
    """
    
    # Calculate the strength (weighted degree) of each node in the bigraph.
    # Strength is the sum of incident edge weights. If 'weight' attribute is missing,
    # igraph.strength() defaults to simple degrees.
    node_strengths = bigraph.strength(weights='weight') 

    # --- Determine which projection to create and calculate weights ---
    if not typen: # Projecting the first partition (users)
        
        # Create the user-projected graph
        projected_graph = bigraph.bipartite_projection(which=False) 
        
        calculated_weights = []
        for edge in projected_graph.es:
            temp_weight = 0
            
            # For user nodes (edge.source, edge.target) in the projected graph,
            # their indices are the same in the original bigraph.
            source_bigraph_idx = edge.source
            target_bigraph_idx = edge.target
            
            # Get common neighbors in the original bigraph
            common_neis = set(bigraph.neighbors(source_bigraph_idx)) & \
                          set(bigraph.neighbors(target_bigraph_idx))
            
            for z_bigraph_idx in common_neis:
                # Add 1 divided by the strength of the common neighbor 'z'.
                # Check for zero strength to prevent division by zero.
                if node_strengths[z_bigraph_idx] > 0:
                    temp_weight += (1 / node_strengths[z_bigraph_idx])
            
            calculated_weights.append(temp_weight)
            
        projected_graph.es["weight"] = calculated_weights
        
    else: # Projecting the second partition (resources)
        
        # Create the resource-projected graph
        projected_graph = bigraph.bipartite_projection(which=True)
        
        calculated_weights = []
        # We need usr_size to correctly map resource node indices from the projected graph
        # back to the original bigraph's indices.
        
        for edge in projected_graph.es:
            temp_weight = 0
            
            # For resource nodes in the projected graph, their indices (edge.source, edge.target)
            # need to be offset by usr_size to get their original bigraph indices.
            source_bigraph_idx = edge.source + usr_size
            target_bigraph_idx = edge.target + usr_size
            
            # Get common neighbors in the original bigraph
            common_neis = set(bigraph.neighbors(source_bigraph_idx)) & \
                          set(bigraph.neighbors(target_bigraph_idx))
            
            # print(f"Processing edge {edge.source}-{edge.target} (bigraph_indices: {source_bigraph_idx}-{target_bigraph_idx}), common_neis: {common_neis}") # Optional: for debugging
            
            for z_bigraph_idx in common_neis:
                # Add 1 divided by the strength of the common neighbor 'z'.
                # Check for zero strength to prevent division by zero.
                if node_strengths[z_bigraph_idx] > 0:
                    temp_weight += (1 / node_strengths[z_bigraph_idx])
            
            calculated_weights.append(temp_weight)
            
        projected_graph.es["weight"] = calculated_weights    
    
    return projected_graph
    
def resource_allocation_projection(bigraph, usr_size, typen=True):
    
    if typen:
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
   
import igraph

def resource_allocation_projection_weights2(bigraph, usr_size, typen=True):
    """
    Calcula una proyección ponderada de asignación de recursos desde un grafo bipartito,
    utilizando la 'fuerza' (strength) de los nodos en lugar del grado.

    Args:
        bigraph (igraph.Graph): El grafo bipartito de entrada con pesos en las aristas.
        usr_size (int): El número de nodos en la primera partición (ej., usuarios).
                        Se asume que los primeros 'usr_size' nodos son del primer tipo,
                        y el resto son del segundo tipo.
        typen (bool): Si es True, proyecta la primera partición (usuarios).
                      Si es False, proyecta la segunda partición (recursos).
                      (Nota: Invierte la lógica de tu función original 'typen' para mayor claridad)

    Returns:
        igraph.Graph: El grafo proyectado (ya sea de usuarios o recursos) con los pesos calculados.
    """
    
    # Calcular la fuerza (strength) de cada nodo en el bigraph una sola vez.
    # La fuerza es la suma de los pesos de las aristas incidentes.
    # Si 'bigraph' no tiene el atributo 'weight', igraph.strength() se comporta como degree().
    node_strengths = bigraph.strength(weights='weight')

    # --- Lógica de Proyección Basada en 'typen' ---
    # Nota: He ajustado la lógica 'typen' para que `True` se asocie con la proyección del primer tipo
    # y `False` con la proyección del segundo tipo, que es más intuitivo si 'typen' indica el tipo principal.
    # Si quieres mantener tu lógica original (typen=True para res_graph, typen=False para usr_graph),
    # simplemente invierte los bloques 'if' y 'else' o las asignaciones de `which`.

    if typen: # Proyecta la primera partición (usuarios)
        projected_graph = bigraph.bipartite_projection(which=False) # 'which=False' para la primera partición
        
        calculated_weights = []
        for edge in projected_graph.es:
            temp_weight_u = 0
            temp_weight_v = 0
            
            # Los índices de source y target en usr_graph coinciden con los del bigraph.
            source_bigraph_idx = edge.source
            target_bigraph_idx = edge.target
            
            # Encontrar vecinos comunes en el bigraph
            common_neis = set(bigraph.neighbors(source_bigraph_idx)) & \
                          set(bigraph.neighbors(target_bigraph_idx))
            
            # Calcular la parte de asignación de recursos basada en fuerza
            for z_bigraph_idx in common_neis:
                strength_z = node_strengths[z_bigraph_idx]
                strength_source = node_strengths[source_bigraph_idx]
                strength_target = node_strengths[target_bigraph_idx]

                # Evitar división por cero si alguna fuerza es 0
                if strength_z > 0 and strength_source > 0:
                    temp_weight_u += (1 / (strength_z + strength_source))
                if strength_z > 0 and strength_target > 0:
                    temp_weight_v += (1 / (strength_z + strength_target))
            
            temp_weight = (temp_weight_u + temp_weight_v) / 2
            calculated_weights.append(temp_weight)
            
        projected_graph.es["weight"] = calculated_weights
        
    else: # Proyecta la segunda partición (recursos)
        projected_graph = bigraph.bipartite_projection(which=True) # 'which=True' para la segunda partición
        
        calculated_weights = []
        for edge in projected_graph.es:
            temp_weight_u = 0
            temp_weight_v = 0
            
            # Los índices de source y target en res_graph necesitan ser ajustados
            # para obtener sus índices correctos en el bigraph.
            source_bigraph_idx = edge.source + usr_size
            target_bigraph_idx = edge.target + usr_size
            
            # Encontrar vecinos comunes en el bigraph
            common_neis = set(bigraph.neighbors(source_bigraph_idx)) & \
                          set(bigraph.neighbors(target_bigraph_idx))
            
            # Calcular la parte de asignación de recursos basada en fuerza
            for z_bigraph_idx in common_neis:
                strength_z = node_strengths[z_bigraph_idx]
                strength_source = node_strengths[source_bigraph_idx]
                strength_target = node_strengths[target_bigraph_idx]

                # Evitar división por cero si alguna fuerza es 0
                if strength_z > 0 and strength_source > 0:
                    temp_weight_u += (1 / (strength_z + strength_source))
                if strength_z > 0 and strength_target > 0:
                    temp_weight_v += (1 / (strength_z + strength_target))
            
            temp_weight = (temp_weight_u + temp_weight_v) / 2
            calculated_weights.append(temp_weight)
            
        projected_graph.es["weight"] = calculated_weights
        
    return projected_graph

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
    elif projection_name == "jaccard":
        return jaccard_projection(bigraph, usr_size, typen=typenn)
    elif projection_name == "jaccard_w":
        return weighted_jaccard_projection(bigraph, usr_size, typen=typenn)
    elif projection_name == "neighs":
        return maestria_projection(bigraph, usr_size, typen=typenn)
    elif projection_name == "master_w":
        return proportional_weighted_projection(bigraph, usr_size, typen=typenn)
    elif projection_name == "hyper":
        return hyperbolic_projection(bigraph, usr_size, typen=typenn)
    elif projection_name == "hyper_w":
        return hyperbolic_projection_weights2(bigraph, usr_size, typen=typenn)
    elif projection_name == "resall":
        return resource_allocation_projection(bigraph, usr_size, typen=typenn)
    elif projection_name == "resall_w":
        return resource_allocation_projection_weights2(bigraph, usr_size, typen=typenn)
    else:
        print("The name of the projection doesn't exist.")
    