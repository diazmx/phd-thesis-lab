import igraph as ig
import numpy as np
import os
import pandas as pd
import powerlaw
from time import time
import matplotlib.pyplot as plt
from auxiliar_bb import noise_corrected, disparity
from auxiliar_projections_large import apply_projection

FILENAME = "TOY-GRAPH.graphml"
FILENAME = "binet-AMZ-Rw.graphml"

###### ****** Read BI GRAPH ****** ######
g = ig.read(FILENAME)
print(g.summary())
print()

user_nodes = g.vs.select(type=0)
res_nodes = g.vs.select(type=1)

if(g.is_bipartite()): # Check if the the graph is bipartite
    print("The graph IS bipartite")
else:
    print("The graph IS NOT bipartite")
    exit()
print("|U|=",len(user_nodes), " \t|R|=",len(res_nodes), " \t|U|+|R|=",
      len(user_nodes)+len(res_nodes), "=", g.vcount())
print()
###### ****** END ****** ######

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

def compute_power_law_bipartite(gb, type_n=1):
    """Calcula el alpha del bipartita"""
    fit = powerlaw.Fit(gb.degree(gb.vs.select(type=type_n)), discrete=True)
    return fit.alpha

def compute_bip_metrics(gb, typen):
    """Calcula x1,x2,x3,x8,x9,γ_Ub del grafo bipartito."""
    x1 = len(gb.vs.select(type=0))
    x2 = len(gb.vs.select(type=1))
    x3 = gb.ecount()
    #x8 = average_local_bipartite_clustering_coefficient(gb, U_type_value=typen)    
    x8 = 0.25
    try:
        #x9 = average_modified_bipartite_shortest_path(gb)
        x9 = g.average_path_length(directed=False)
    except:
        x9 = np.inf
    x11 = compute_power_law_bipartite(gb, typen)
    return dict(x1=x1, x2=x2, x3=x3, x8=x8, x9=x9, x11=x11)

def compute_power_law(g):
    """Calcula el alpha del proyectado"""
    fit = powerlaw.Fit(g.degree(), discrete=True, verbose=False)
    return fit.alpha

def compute_proj_metrics(gu):
    """Calcula x4,x5,x6,x7,x10,γ_U de la proyección."""
    x4 = gu.vcount()
    x5 = gu.ecount()
    x6 = len(gu.clusters())
    x7 = gu.transitivity_undirected(mode="zero")
    try:
        x10 = gu.average_path_length(directed=False)
    except:
        x10 = np.inf
    x12 = compute_power_law(gu)
    return dict(x4=x4, x5=x5, x6=x6, x7=x7, x10=x10, x12=x12)

def evaluate_solution(bip, proj):
    """Dado bip y proj metrics, arma x, f, g."""
    # unimos diccionarios
    x = {
        **bip,
        **proj
    }
    # objetivos
    f = np.array([
        abs(x["x1"] - x["x4"]),
        (2*x["x5"]) / (x["x4"]*(x["x4"]-1)) if x["x4"]>1 else np.inf,
        x["x6"],
        1 - x["x7"],
        abs(x["x9"] - x["x10"]),
        abs(x["x11"] - x["x12"]),
        abs(x["x7"] - x["x8"])
    ])
    # restricciones g_i(x)<=0
    g = np.array([
        f[1] - x["x3"]/(x["x1"]*x["x2"]) if x["x1"]*x["x2"]>0 else np.inf,
    ])
    return dict(metrics=x, f=f, g=g, graph=proj)

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


# —————————— Flujo principal ——————————

# 1) Leer el único grafo bipartito
gb = ig.Graph.Read_GraphML(FILENAME)
bip_metrics = compute_bip_metrics(gb, 1)
print(bip_metrics)

# 2) Escanear carpeta de proyecciones
proj_dir = "grafos2"
proj_files = [f for f in os.listdir(proj_dir)
              if f.endswith(".graphml") and f!="bipartito.graphml"]

# 3) Calcular soluciones
solutions = []
for fname in proj_files:
    gu = ig.Graph.Read_GraphML(os.path.join(proj_dir, fname))
    proj_metrics = compute_proj_metrics(gu)
    sol = evaluate_solution(bip_metrics, proj_metrics)
    sol["filename"] = fname  # <- Añadimos esta línea
    if is_feasible(sol):
        solutions.append(sol)

print("Soluciones factibles ", len(solutions))

# 4) Extraer Pareto y ordenar por crowding distance
pareto = pareto_front(solutions)
print("Solucion pareto", len(pareto))

cd = crowding_distance(pareto)
pareto_sorted = [s for _, s in sorted(zip(-cd, pareto), key=lambda x: x[0])]


# 5) Salida: lista de grafos óptimos (proyecciones)
optimal_graphs = [s["graph"] for s in pareto_sorted]
print("Crowding", len(cd))

# Ejemplo: imprimir nombres de archivo en orden óptimo
for sol in pareto_sorted:
    print(sol["metrics"]["x4"], sol["f"])

print("Grafos óptimos ordenados por diversidad (crowding distance):")
for i, sol in enumerate(pareto_sorted, 1):
    print(f"{i:02d}. {sol['filename']} — f = {sol['f']}")


# Número máximo de grafos a graficar
N = len(pareto_sorted)
labels = [f"f{i+1}" for i in range(7)]  # nombres de funciones f1..f6

# Extraer vectores f y nombres de archivo
F = np.array([s["f"] for s in pareto_sorted[:N]])
names = [s["filename"] for s in pareto_sorted[:N]]

# Normalizar funciones objetivo columna por columna
F_norm = (F - F.min(axis=0)) / (F.max(axis=0) - F.min(axis=0) + 1e-8)

# Preparar figura
plt.figure(figsize=(10, 7))

# Colores y estilos variados
colors = plt.cm.tab10.colors
linestyles = ['-', '--', '-.', ':'] * 3

for i in range(N):
    plt.plot(
        labels,
        F_norm[i],
        label=names[i],
        color=colors[i % len(colors)],
        linestyle=linestyles[i % len(linestyles)],
        linewidth=2,
        marker='o'
    )

plt.ylabel("Función objetivo normalizada")
plt.title("Comparativa de funciones objetivo normalizadas")
plt.grid(True, linestyle="--", alpha=0.4)
plt.legend(title="Grafos", bbox_to_anchor=(1.05, 1), loc="upper left")
plt.tight_layout()
plt.savefig("test2.png")
plt.show()
