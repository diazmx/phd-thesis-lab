import igraph as ig
import numpy as np
import os
import pandas as pd
import statistics
import powerlaw
import random
import networkx as nx
#import nx_cugraph as nxcg
from time import time

def bipartite_cc_uu_prime(graph, u_id, u_prime_id):
    """
    Calculates the Jaccard index based clustering coefficient for a pair of vertices
    u and u' from the same set of nodes in a bipartite graph.

    Args:
        graph: An igraph Graph object. Must be bipartite with a 'type' vertex attribute.
        u_id: The ID of the first vertex.
        u_prime_id: The ID of the second vertex.

    Returns:
        The Jaccard index (cc_u_u_prime) or 0 if union of neighbors is empty.
    """
    if not graph.is_bipartite():
        raise ValueError("Graph must be bipartite.")

    # Get neighbors of u and u'
    neighbors_u = set(graph.neighbors(u_id))
    neighbors_u_prime = set(graph.neighbors(u_prime_id))

    # Calculate intersection and union
    intersection = len(neighbors_u.intersection(neighbors_u_prime))
    union = len(neighbors_u.union(neighbors_u_prime))

    if union == 0:
        return 0.0
    return intersection / union
 
def local_bipartite_clustering_coefficient(graph, u_id, U_type_value=False):
    """
    Calculates the local clustering coefficient for a vertex u in a bipartite graph.
    The formula uses neighbors of neighbors (N(N(u))) that are of the same type as u.

    Args:
        graph: An igraph Graph object. Must be bipartite with a 'type' vertex attribute
               (e.g., True/False or 0/1 for the two partitions).
        u_id: The ID of the vertex for which to calculate the local clustering coefficient.
        U_type_value: The boolean value (True/False) that indicates the partition
                      to which node u belongs. All nodes in U should have this type.

    Returns:
        The local clustering coefficient for vertex u, or 0 if N(N(u)) is empty.
    """
    if not graph.is_bipartite():
        raise ValueError("Graph must be bipartite.")
    if "type" not in graph.vs.attributes():
        raise ValueError("Bipartite graph must have a 'type' vertex attribute.")

    # Ensure u_id is of the specified U_type_value
    if graph.vs[u_id]["type"] != U_type_value:
        raise ValueError(f"Vertex {u_id} does not belong to the specified partition U.")

    # Get neighbors of u
    neighbors_u = graph.neighbors(u_id)
    
    # Get neighbors of neighbors of u, filtering for nodes of the same type as u
    # These are the u' nodes in N(N(u)) that are in the same partition U
    nn_u = set()
    for v_neighbor_id in neighbors_u:
        for nn_id in graph.neighbors(v_neighbor_id):
            if graph.vs[nn_id]["type"] == U_type_value and nn_id != u_id: # Exclude u itself
                nn_u.add(nn_id)

    if not nn_u:
        return 0.0

    sum_cc_uu_prime = 0.0
    for u_prime_id in nn_u:
        sum_cc_uu_prime += bipartite_cc_uu_prime(graph, u_id, u_prime_id)

    return sum_cc_uu_prime / len(nn_u)

def average_local_bipartite_clustering_coefficient(graph, U_type_value=False):
    """
    Calculates the average local clustering coefficient for a set of nodes U
    in a bipartite graph.

    Args:
        graph: An igraph Graph object. Must be bipartite with a 'type' vertex attribute.
        U_type_value: The boolean value (True/False) that indicates the partition
                      for which to calculate the average clustering coefficient.

    Returns:
        The average local clustering coefficient for the set U.
    """
    if not graph.is_bipartite():
        raise ValueError("Graph must be bipartite.")
    if "type" not in graph.vs.attributes():
        raise ValueError("Bipartite graph must have a 'type' vertex attribute.")

    # Get all vertices in set U
    U_vertices_ids = [v.index for v in graph.vs if v["type"] == U_type_value]

    if not U_vertices_ids:
        return 0.0

    sum_cc_u = 0.0
    for u_id in U_vertices_ids:
        sum_cc_u += local_bipartite_clustering_coefficient(graph, u_id, U_type_value)

    return sum_cc_u / len(U_vertices_ids)

def compute_power_law_bipartite(gb, type_n):
    """Calcula el alpha del bipartita"""
    fit = powerlaw.Fit(gb.degree(gb.vs.select(type=type_n)), discrete=True, verbose=False)
    return fit.alpha

def compute_power_law(g):
    """Calcula el alpha del proyectado"""
    fit = powerlaw.Fit(g.degree(), discrete=True, verbose=False)
    return fit.alpha

def remove_isolated_nodes(graph, k):
        
    #print(f"Número inicial de vértices: {graph.vcount()}")
    #print(f"Componentes iniciales: {len(graph.clusters())}")

    # 1. Obtener los clústeres
    clusters = graph.clusters()

    # 2. Identificar los IDs de los vértices a eliminar
    vertices_to_delete = []
    for i, cluster_members in enumerate(clusters):
        if len(cluster_members) < k:
            vertices_to_delete.extend(cluster_members)

    # Eliminar duplicados si un vértice pudiera aparecer en múltiples listas por algún error (no debería pasar con clusters)
    # Aunque igraph maneja esto bien, es una buena práctica si la lista se construye de forma menos controlada
    vertices_to_delete = sorted(list(set(vertices_to_delete)), reverse=True) # Eliminar en orden descendente para evitar problemas de reindexación

    # 3. Eliminar los vértices
    if vertices_to_delete:
        graph.delete_vertices(vertices_to_delete)
        #print(f"Nodos eliminados: {len(vertices_to_delete)}")

    #print(f"Número final de vértices: {graph.vcount()}")
    #print(f"Componentes finales: {len(graph.clusters())}")
    return graph


def compute_avg_path_length(g, k):
    G = g.to_networkx()
    nxG = nxcg.from_networkx(G)
    all_nodes = list(G.nodes())
    sample_nodes = random.sample(all_nodes, k)
    total_distances = 0
    reachable_pairs = 0
    for i, src_node in enumerate(sample_nodes):
        sssp_df = nxcg.shortest_path_length(nxG, source=src_node)
        total_distances += sum(sssp_df.values())
        reachable_pairs += len(sssp_df.values())
    apl_approx = float(total_distances) / reachable_pairs
    return apl_approx

def compute_weight_distribution(pesos):
    fit = powerlaw.Fit(pesos, discrete=True, verbose=False)
    return fit.alpha


def compute_bip_metrics(gb, typen):
    """Calcula x1,x2,x3,x8,x9,γ_Ub del grafo bipartito."""
    x1 = len(gb.vs.select(type=0))
    x2 = len(gb.vs.select(type=1))
    x3 = gb.ecount()
    x8 = 1
    x9 = compute_avg_path_length(gb, 100)
    x11 = compute_power_law_bipartite(gb, typen)
    return dict(x1=x1, x2=x2, x3=x3, x8=x8, x9=x9, x11=x11)

def compute_proj_metrics(gu, k):
    """Calcula x4,x5,x6,x7,x10,γ_U de la proyección."""
    x4 = gu.vcount()
    x5 = gu.ecount()
    x6 = len(gu.clusters(mode='weak'))
    x7 = gu.transitivity_undirected(mode="zero")
    x10 = compute_avg_path_length(gu, k)
    x12 = compute_power_law(gu)
    #x13 = compute_weight_distribution(gu.es["weight"])
    x13 = statistics.mean(gu.es["weight"])
    return dict(x4=x4, x5=x5, x6=x6, x7=x7, x10=x10, x12=x12, x13=x13)

def evaluate_solution(bip, proj, typen):
    """Dado bip y proj metrics, arma x, f, g."""
    # unimos diccionarios
    x = {
        **bip,
        **proj
    }
    # objetivos
    f = np.array([
        # Mismo número de nodos
        abs(x["x1"] - x["x4"]) if typen==0 else abs(x["x2"] - x["x4"]),
        (2*x["x5"]) / (x["x4"]*(x["x4"]-1)) if x["x4"]>1 else np.inf,
        # Misma densidad
        abs(((2*x["x5"]) / (x["x4"]*(x["x4"]-1))) - x["x15"]), 
        abs(x["x8"] - x["x7"]),  # CC
        abs(x["x11"] - x["x12"]), # Power Law
        abs(x["x9"] - x["x10"]), # APL
        abs(x["x13"]-x["x16"])  # Grado Promedio
        ])
    # restricciones g_i(x)<=0
    g = np.array([
        #f[1] - x["x3"]/(x["x1"]*x["x2"]) if x["x1"]*x["x2"]>0 else np.inf,
        (x["x1"]/2)-(x["x4"])  if typen==0 else (x["x2"]/2)-(x["x4"]), # mitad de los nodos
        (x["x1"]-1) - x["x5"] if typen==0 else (x["x2"]-1) - x["x5"] # Cuerda
    ])
    return dict(metrics=x, f=f, g=g, graph=proj)
    #return dict(metrics=x, f=f, graph=proj)

def is_feasible(sol):
    return np.all(sol["g"] <= 0)

def pareto_front(sols):
    front = []
    for i, si in enumerate(sols):
        if any(np.all(sj["f"] <= si["f"]) and np.any(sj["f"] < si["f"])
               for j, sj in enumerate(sols) if i!=j):
            continue
        front.append(si)
    return front

def crowding_distance(front):
    N, k = len(front), front[0]["f"].size
    F = np.array([s["f"] for s in front])
    dist = np.zeros(N)
    for m in range(k):
        idx = np.argsort(F[:,m])
        f_min, f_max = F[idx[0],m], F[idx[-1],m]
        dist[idx[0]] = dist[idx[-1]] = np.inf
        if f_max == f_min: continue
        for i in range(1, N-1):
            dist[idx[i]] += (F[idx[i+1],m] - F[idx[i-1],m]) / (f_max - f_min)
    return dist

def pareto_rank_all(solutions):
    """
    Clasifica todas las soluciones en frentes de Pareto.
    Devuelve una lista de listas: cada sublista contiene un frente.
    """
    remaining = solutions.copy()
    fronts = []
    
    while remaining:
        current_front = []
        for i, si in enumerate(remaining):
            dominated = False
            for j, sj in enumerate(remaining):
                if i == j:
                    continue
                if np.all(sj["f"] <= si["f"]) and np.any(sj["f"] < si["f"]):
                    dominated = True
                    break
            if not dominated:
                current_front.append(si)
        
        fronts.append(current_front)
        remaining = [s for s in remaining if all(s is not r for r in current_front)]
    
    return fronts

