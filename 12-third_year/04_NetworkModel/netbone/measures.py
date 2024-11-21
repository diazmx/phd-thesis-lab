import igraph as ig
import networkx as nx
import numpy as np
from netbone.utils.utils import lcc
from scipy.stats import entropy



def density(original, graph):
    return round(nx.density(graph), 4)

def weight_entropy(original, graph):
    return entropy(weights(graph), base=2)

def average_degree(original, graph):
    return np.mean(degrees(graph))

def lcc_size(original, graph):
    return len(lcc(graph))



def node_fraction(network, b):
    return len(b)/len(network)

def edge_fraction(network, b):
    return len(b.edges())/len(network.edges())

def weight_fraction(network, b):
    return sum(weights(b))/sum(weights(network))

def reachability(original, G):
    r = 0
    for c in [len(component) for component in nx.connected_components(G)]:
        r += c*(c-1)
    return r/(len(G)*(len(G) - 1))

def number_connected_components(original, G):
    return nx.number_connected_components(G)

def diameter(original, G):
    return ig.Graph.from_networkx(lcc(G)).diameter(directed=False, unconn=True)

def lcc_node_fraction(G):
    return node_fraction(G, lcc(G))

def lcc_edge_fraction(original, G):
    return edge_fraction(G, lcc(G))

def lcc_weight_fraction(original, G):
    return weight_fraction(G, lcc(G))

def weights(G):
    return list(nx.get_edge_attributes(G, 'weight').values())

def degrees(G, weight=None):
    return list(dict(G.degree(weight=weight)).values())

def average_clustering_coefficient(original, G):
    node_clustering = ig.Graph.from_networkx(G).transitivity_local_undirected(mode="nan")
    return np.mean([x for x in node_clustering if isinstance(x, float) and not np.isnan(x)])

def transitivity(original, graph):
    return nx.transitivity(graph)








