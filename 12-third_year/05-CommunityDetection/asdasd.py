import numpy as np
import igraph as ig
import matplotlib.pyplot as plt

# Función para calcular la matriz de asignación de nodos a comunidades
def calculate_community_assignment_matrix(graph, n_runs=10):
    n_nodes = len(graph.vs)  # Número de nodos
    community_counts = {}  # Diccionario para contar asignaciones
    
    # Inicializar el diccionario de conteos
    for node in range(n_nodes):
        community_counts[node] = {}
    
    # Ejecutar el algoritmo multinivel n_runs veces
    for _ in range(n_runs):
        # Obtener la partición de comunidades
        partition = graph.community_multilevel()
        
        # Contar las asignaciones de cada nodo a comunidades
        for node in range(n_nodes):
            community_id = partition.membership[node]
            if community_id in community_counts[node]:
                community_counts[node][community_id] += 1
            else:
                community_counts[node][community_id] = 1
    
    # Obtener todas las comunidades únicas
    unique_communities = set()
    for node_counts in community_counts.values():
        unique_communities.update(node_counts.keys())
    unique_communities = sorted(unique_communities)
    
    # Construir la matriz de asignación
    n_communities = len(unique_communities)
    assignment_matrix = np.zeros((n_nodes, n_communities))
    
    for node in range(n_nodes):
        for community_id in unique_communities:
            if community_id in community_counts[node]:
                assignment_matrix[node, community_id] = community_counts[node][community_id] / n_runs
    
    return assignment_matrix, unique_communities

# Función para visualizar la matriz de asignación
def plot_assignment_matrix(assignment_matrix, communities):
    plt.figure(figsize=(10, 8))
    
    # Crear el heatmap
    plt.imshow(assignment_matrix, cmap='viridis', interpolation='none', vmin=0, vmax=1)
    plt.colorbar(label="Frecuencia de asignación")
    
    # Añadir etiquetas de ejes
    plt.xticks(range(len(communities)), [f"Comunidad {c}" for c in communities], rotation=90)
    plt.yticks(range(assignment_matrix.shape[0]), range(assignment_matrix.shape[0]))
    
    # Añadir título y etiquetas de ejes
    plt.title("Matriz de asignación de nodos a comunidades")
    plt.xlabel("Comunidades")
    plt.ylabel("Nodos")
    
    # Mostrar el gráfico
    plt.tight_layout()
    plt.show()

# Ejemplo de uso
if __name__ == "__main__":
    # Crear un grafo de ejemplo
    graph = ig.Graph.Famous("Zachary")  # Grafo de Karate Club
    
    # Calcular la matriz de asignación
    assignment_matrix, communities = calculate_community_assignment_matrix(graph, n_runs=10)
    
    # Visualizar la matriz de asignación
    plot_assignment_matrix(assignment_matrix, communities)