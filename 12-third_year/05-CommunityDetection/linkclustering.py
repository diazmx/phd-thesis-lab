import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from community import best_partition
from collections import defaultdict

# Funciones anteriores (get_community_members, calculate_community_similarity, calculate_consistency)
def get_community_members(graph, partition):
    community_members = defaultdict(set)
    for node, community_id in partition.items():
        community_members[community_id].add(node)
    return community_members

def calculate_community_similarity(community1, community2):
    intersection = len(community1 & community2)
    union = len(community1 | community2)
    return intersection / union if union != 0 else 0

def calculate_consistency(graph, n_runs=10, similarity_threshold=0.5):
    all_communities = []
    for _ in range(n_runs):
        partition = best_partition(graph)
        community_members = get_community_members(graph, partition)
        all_communities.append(community_members)
    
    consistency = {}
    for node in graph.nodes():
        node_communities = []
        for community_members in all_communities:
            for community_id, members in community_members.items():
                if node in members:
                    node_communities.append(members)
                    break
        
        similarity_scores = []
        for i in range(len(node_communities) - 1):
            for j in range(i + 1, len(node_communities)):
                similarity = calculate_community_similarity(node_communities[i], node_communities[j])
                similarity_scores.append(similarity)
        
        if similarity_scores:
            consistency[node] = np.mean([s >= similarity_threshold for s in similarity_scores])
        else:
            consistency[node] = 1.0
    
    return consistency

# Función para graficar el grafo con colores basados en la consistencia
def plot_graph_with_consistency(graph, consistency):
    pos = nx.spring_layout(graph)  # Posiciones de los nodos
    node_colors = [consistency[node] for node in graph.nodes()]  # Colores basados en la consistencia
    
    # Crear una figura y un eje explícito
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Dibujar el grafo en el eje
    nx.draw(graph, pos, node_color=node_colors, cmap=plt.cm.RdYlBu, with_labels=True, node_size=500, edge_color='gray', alpha=0.8, ax=ax)
    
    # Crear un mapeo de colores para la barra de color
    sm = plt.cm.ScalarMappable(cmap=plt.cm.RdYlBu, norm=plt.Normalize(vmin=0, vmax=1))
    sm.set_array([])
    
    # Añadir la barra de color al gráfico
    plt.colorbar(sm, ax=ax, label="Consistencia")
    
    # Título del gráfico
    ax.set_title("Grafo con nodos coloreados por consistencia")
    
    # Mostrar el gráfico
    plt.show()

# Ejemplo de uso
if __name__ == "__main__":
    # Crear un grafo de ejemplo
    #graph =  nx.karate_club_graph()
    graph = nx.read_graphml("/Users/ddiaz/Documents/code/phd-thesis-lab/12-third_year/00-Data/01-AMZ/02-Graphs/01-Top/AMZ_top_hyperbolic_DF_alpha1.graphml")
    
    # Calcular la consistencia de cada nodo
    consistency = calculate_consistency(graph, n_runs=10, similarity_threshold=0.99)
    consistency = dict(sorted(consistency.items(), key=lambda item: item[1]))
    #print(consistency)
    claves = list(consistency.keys())
    valores = list(consistency.values())
    plt.plot(valores)
    plt.title('Distribución de Consistencia')
    plt.xlabel('ID Nodo')
    plt.ylabel('Consistencia')
    plt.ylim(-0.1, 1.1)
    plt.show()
    # Graficar el grafo con colores basados en la consistencia
    #plot_graph_with_consistency(graph, consistency)