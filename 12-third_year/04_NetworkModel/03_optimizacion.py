import os
import igraph as ig
import numpy as np
import pandas as pd
import seaborn as sns
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
