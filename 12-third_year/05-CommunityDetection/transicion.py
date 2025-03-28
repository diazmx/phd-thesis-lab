import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from community import best_partition
from collections import defaultdict

# Función para calcular el índice de Jaccard entre dos conjuntos
def jaccard_similarity(set1, set2):
    intersection = len(set1 & set2)
    union = len(set1 | set2)
    return intersection / union if union != 0 else 0

# Función para calcular la matriz de transición basada en la similitud de comunidades
def calculate_transition_matrix(graph, n_runs=10, similarity_threshold=0.8):
    # Almacenar las comunidades de cada ejecución
    all_communities = []
    for _ in range(n_runs):
        partition = best_partition(graph)
        communities = defaultdict(set)
        for node, community_id in partition.items():
            communities[community_id].add(node)
        all_communities.append(communities)
    
    # Obtener todas las comunidades únicas (basadas en su composición)
    unique_communities = []
    for communities in all_communities:
        for community in communities.values():
            is_new = True
            for unique_community in unique_communities:
                if jaccard_similarity(community, unique_community) >= similarity_threshold:
                    is_new = False
                    break
            if is_new:
                unique_communities.append(community)
    
    # Crear un mapeo de comunidades a IDs únicos
    community_to_id = {tuple(community): i for i, community in enumerate(unique_communities)}
    
    # Construir la matriz de transición
    n_communities = len(unique_communities)
    transition_matrix = np.zeros((n_communities, n_communities))
    
    for run_idx in range(1, n_runs):
        previous_communities = all_communities[run_idx - 1]
        current_communities = all_communities[run_idx]
        
        # Mapear comunidades anteriores a IDs únicos
        previous_ids = {}
        for community in previous_communities.values():
            for unique_community, unique_id in community_to_id.items():
                if jaccard_similarity(community, set(unique_community)) >= similarity_threshold:
                    previous_ids[tuple(community)] = unique_id
                    break
        
        # Mapear comunidades actuales a IDs únicos
        current_ids = {}
        for community in current_communities.values():
            for unique_community, unique_id in community_to_id.items():
                if jaccard_similarity(community, set(unique_community)) >= similarity_threshold:
                    current_ids[tuple(community)] = unique_id
                    break
        
        # Contar transiciones
        for previous_community, previous_id in previous_ids.items():
            for current_community, current_id in current_ids.items():
                transition_matrix[previous_id, current_id] += 1
    
    # Normalizar la matriz de transición
    row_sums = transition_matrix.sum(axis=1, keepdims=True)
    transition_matrix = transition_matrix / row_sums
    
    return transition_matrix, unique_communities

# Función para visualizar la matriz de transición
def plot_transition_matrix(transition_matrix, communities):
    plt.figure(figsize=(10, 8))
    
    # Crear el heatmap
    plt.imshow(transition_matrix, cmap='viridis', interpolation='none', vmin=0, vmax=1)
    plt.colorbar(label="Probabilidad de transición")
    
    # Añadir etiquetas de comunidades
    labels = [f"Comunidad {i}" for i in range(len(communities))]
    plt.xticks(range(len(communities)), labels, rotation=90)
    plt.yticks(range(len(communities)), labels)
    
    # Añadir título y etiquetas de ejes
    plt.title("Matriz de transición entre comunidades")
    plt.xlabel("Comunidad destino")
    plt.ylabel("Comunidad origen")
    
    # Mostrar el gráfico
    plt.tight_layout()
    plt.show()

# Ejemplo de uso
if __name__ == "__main__":
    # Crear un grafo de ejemplo
    graph = nx.karate_club_graph()
    
    # Calcular la matriz de transición
    transition_matrix, communities = calculate_transition_matrix(graph, n_runs=10, similarity_threshold=0.8)
    
    # Visualizar la matriz de transición
    plot_transition_matrix(transition_matrix, communities)