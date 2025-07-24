import numpy as np
import pandas as pd
import networkx as nx
import nx_cugraph as nxcg
from time import time
import os

print(f"using networkx version {nx.__version__}")

G = nx.read_graphml("binet-AMZ-Rw.graphml")
print("|N|",G.number_of_nodes(), "   |M|",G.number_of_edges())

nxcg_G = nxcg.from_networkx(G) 

a = time()
#bc_results = nx.betweenness_centrality(G, k=100)
nx.betweenness_centrality(nxcg_G, k=1000)
b = time() - a
print("GPU - time: %.10f seconds." % b)

a = time()
bc_results = nx.betweenness_centrality(G, k=1000)
#nx.betweenness_centrality(nxcg_G, k=1000)
b = time() - a
print("CPU - time: %.10f seconds." % b)