import igraph as ig
import networkx as nx
import matplotlib.pyplot as plt
from networkx.algorithms import bipartite

edge_list = [(0, 8, 0.11),
             (1, 8, 0.03), (1, 9, 0.05),
             (2, 8, 0.07), (2, 9, 0.20),
             (3, 8, 0.02), (3, 9, 0.10), (3, 10, 0.05), (3, 11, 0.05),
             (4, 9, 0.04), (4, 10, 0.08,),
             (5, 11, 0.08),(6, 11, 0.02),(7, 11, 0.10)]

g = ig.Graph.TupleList(edge_list, weights=True)

print(g.summary())

if(g.is_bipartite()):
    print("Is bipartite")
else:
    print("Is not bipartite")

# Convertimos a networkx
netx = g.to_networkx()


print("Conversión: OK")
print(netx)

print("="*40)

strenght = netx.degree(weight="weight")
print(netx.degree())
print(strenght)

edges = g.es

print()
print()
print(g.get_edgelist())

print(bipartite.is_bipartite(netx))
# explicitly set positions
pos = {0: (0, 0), 1: (0, 1), 2: (1, 0), 3: (1, 1), 4: (2, 0),
       5: (3, 0), 6: (2, 1), 7: (3, 1), 8: (4, 0), 9: (5, 0),
       10: (6, 0), 11: (7, 0)}

nx.draw_networkx(netx, pos)
# Set margins for the axes so that nodes aren't clipped
ax = plt.gca()
ax.margins(0.20)
plt.axis("off")
plt.show()