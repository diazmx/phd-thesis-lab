import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from community import best_partition
from collections import defaultdict

# Función para calcular la matriz de co-ocurrencia
def calculate_co_occurrence_matrix(graph, n_runs=10):
    n_nodes = graph.number_of_nodes()
    co_occurrence_matrix = np.zeros((n_nodes, n_nodes))  # Inicializar la matriz
    
    for _ in range(n_runs):
        partition = best_partition(graph)
        communities = defaultdict(set)
        
        # Agrupar nodos por comunidad
        for node, community_id in partition.items():
            communities[community_id].add(node)
        
        # Actualizar la matriz de co-ocurrencia
        for community in communities.values():
            nodes_in_community = list(community)
            for i in range(len(nodes_in_community)):
                for j in range(i, len(nodes_in_community)):
                    node_i = int(nodes_in_community[i][1:])
                    node_j = int(nodes_in_community[j][1:])
                    co_occurrence_matrix[node_i, node_j] += 1
                    if node_i != node_j:
                        co_occurrence_matrix[node_j, node_i] += 1
    
    # Normalizar la matriz por el número de ejecuciones
    co_occurrence_matrix /= n_runs
    return co_occurrence_matrix

# Función para graficar el grafo con colores basados en la consistencia
def graph_graph_xd(graph):
    pos = nx.spring_layout(graph)  # Posiciones de los nodos

    
    # Dibujar el grafo en el eje
    nx.draw(graph, pos, cmap=plt.cm.RdYlBu, with_labels=True, node_size=500, edge_color='gray', alpha=0.8)

    
    # Mostrar el gráfico
    plt.show()

# Función para visualizar la matriz de co-ocurrencia
def plot_co_occurrence_matrix(co_occurrence_matrix, node_labels=None):
    plt.figure(figsize=(10, 8))
    
    # Crear el heatmap
    plt.imshow(co_occurrence_matrix, cmap='Reds', interpolation='none', vmin=0, vmax=1)
    plt.colorbar(label="Frecuencia de co-ocurrencia")
    
    # Añadir etiquetas de nodos si se proporcionan
    if node_labels:
        plt.xticks(range(len(node_labels)), node_labels, rotation=90)
        plt.yticks(range(len(node_labels)), node_labels)
    
    # Añadir título y etiquetas de ejes
    plt.title("Matriz de co-ocurrencia")
    plt.xlabel("Nodos")
    plt.ylabel("Nodos")
    
    # Mostrar el gráfico
    plt.tight_layout()
    plt.show()

# Ejemplo de uso
if __name__ == "__main__":
    # Crear un grafo de ejemplo
    graph = nx.karate_club_graph()
    graph = nx.read_graphml("/Users/ddiaz/Documents/code/phd-thesis-lab/12-third_year/00-Data/01-AMZ/02-Graphs/01-Top/AMZ_top_hyperbolic_DF_alpha1.graphml")

    # Calcular la matriz de co-ocurrencia
    co_occurrence_matrix = calculate_co_occurrence_matrix(graph, n_runs=10)
    
    # Visualizar la matriz de co-ocurrencia
    plot_co_occurrence_matrix(co_occurrence_matrix, node_labels=list(graph.nodes()))

    #graph_graph_xd(graph)