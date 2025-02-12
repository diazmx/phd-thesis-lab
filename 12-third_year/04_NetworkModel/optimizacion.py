import os
import igraph as ig
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

# Función para cargar grafos desde un directorio
def load_graphs_from_directory(directory):
    graphs = []
    graph_names = []
    for filename in os.listdir(directory):
        if filename.endswith(".graphml"):
            graph = ig.Graph.Read_GraphML(os.path.join(directory, filename))
            graphs.append(graph)
            graph_names.append(filename)
    return graphs, graph_names

# Función para calcular las métricas para cada grafo
def calculate_metrics(graph, bigraph):
    user_nodes = bigraph.vs.select(type=0)
    res_nodes = bigraph.vs.select(type=1)
    bidensity = bigraph.ecount() / (len(user_nodes)*len(res_nodes))

    num_nodes = 1-(len(graph.vs) / len(user_nodes))
    density = abs(graph.density() - bidensity)
    components = len(graph.components())
    modularity = 1- graph.modularity(graph.community_multilevel())
    #clustering_coefficient = graph.transitivity_avglocal_undirected()
    #avg_degree = 2 * num_edges / num_nodes if num_nodes > 0 else 0
    #avg_path_length = graph.average_path_length() if graph.is_connected() else np.inf
    
    metrics = {
        "num_nodes": num_nodes,
        "density": density,
        "components": components,
        "modularity": modularity,
        #"clustering_coefficient": clustering_coefficient,
        #"avg_degree": avg_degree,
        #"avg_path_length": avg_path_length
    }
    return metrics

# Función para determinar si un grafo domina a otro
def dominates(graph_a, graph_b, metrics):
    dominates_flag = False
    for metric in metrics.keys():
        if metrics[metric]["optimize"] == "minimize":
            if graph_a[metric] > graph_b[metric]:
                return False
            if graph_a[metric] < graph_b[metric]:
                dominates_flag = True
        elif metrics[metric]["optimize"] == "maximize":
            if graph_a[metric] < graph_b[metric]:
                return False
            if graph_a[metric] > graph_b[metric]:
                dominates_flag = True
    return dominates_flag

# Función para calcular los frentes de Pareto
def calculate_pareto_fronts(graph_metrics):
    pareto_fronts = []
    remaining_graphs = list(graph_metrics.items())
    
    while remaining_graphs:
        current_front = []
        for i, (graph_i, metrics_i) in enumerate(remaining_graphs):
            dominated = False
            for j, (graph_j, metrics_j) in enumerate(remaining_graphs):
                if i != j and dominates(metrics_j, metrics_i, metrics_definitions):
                    dominated = True
                    break
            if not dominated:
                current_front.append((graph_i, metrics_i))
        pareto_fronts.append(current_front)
        remaining_graphs = [graph for graph in remaining_graphs if graph not in current_front]
    
    return pareto_fronts

# Visualización de los niveles de Pareto
def visualize_pareto_fronts(pareto_fronts, graph_names):
    colors = plt.cm.tab10(np.linspace(0, 1, len(pareto_fronts)))
    plt.figure(figsize=(12, 8))
    
    id_to_name = {}
    for level, front in enumerate(pareto_fronts):
        for graph_id, (graph, metrics) in enumerate(front):
            plt.scatter(metrics["density"], metrics["components"], color=colors[level], label=f'Pareto Level {level+1}' if graph_id == 0 else "")
            id_to_name[graph] = graph_names[graph]
            plt.text(metrics["density"], metrics["components"], str(graph), fontsize=8, color="b")

    plt.xlabel("Density")
    plt.ylabel("components")
    plt.title("Pareto Fronts Visualization")
    plt.legend(title="Pareto Levels", loc="best")
    plt.grid(True)

    # Crear la leyenda con los nombres de los grafos
    plt.figure(figsize=(8, 4))
    plt.title("Graph Identifiers")
    plt.axis("off")
    for graph_id, graph_name in id_to_name.items():
        plt.text(0.1, 1 - 0.05 * graph_id, f"ID {graph_id}: {graph_name}", fontsize=10)
    
    plt.show()


# Definición de las métricas y sus objetivos
metrics_definitions = {
    "num_nodes": {"optimize": "minimize"},
    "density": {"optimize": "minimize"},
    "components": {"optimize": "minimize"},
    "modularity": {"optimize": "minimize"},
    #"clustering_coefficient": {"optimize": "maximize"},
    #"avg_degree": {"optimize": "minimize"},
    #"avg_path_length": {"optimize": "minimize"}
}

# Main
if __name__ == "__main__":
    directory = "grafos"  # Cambia esta ruta
    graphs, graph_names = load_graphs_from_directory(directory)

    bigraph = ig.read("user-movie-lens.graphml")
    
    graph_metrics = {}
    for i, graph in enumerate(graphs):
        metrics = calculate_metrics(graph, bigraph)
        graph_metrics[i] = metrics
    
    pareto_fronts = calculate_pareto_fronts(graph_metrics)
    visualize_pareto_fronts(pareto_fronts, graph_names)
