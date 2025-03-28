from new_comms import tracking_louvain_communities
import networkx as nx

import matplotlib.pyplot as plt

GLOBAL_PATH = "/Users/ddiaz/Documents/code/phd-thesis-lab/"

GRAPH_PATH = (
    GLOBAL_PATH +
    "12-third_year/00-Data/01-AMZ/02-Graphs/01-Top/AMZ_top_resall_DF_alpha2.graphml"
    )

#G = nx.karate_club_graph()
G = nx.read_graphml(GRAPH_PATH)
G = nx.convert_node_labels_to_integers(G)  # Opcional: normalizar IDs

communities, change_counts = tracking_louvain_communities(G)

print("Final communities:", communities)
print("Number of changes per node:", change_counts)


keys_ = list(change_counts.keys())
values_ = list(change_counts.values())

fig = plt.figure(0, figsize=(8, 5))
plt.plot(values_)
plt.xlabel('ID of entity')
plt.ylabel('Jaccard Index (Number of entities in common)')
plt.title('Jaccard Index')
plt.xticks(rotation=0)
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Show the plot
plt.show()