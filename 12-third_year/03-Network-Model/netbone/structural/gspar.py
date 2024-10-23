import networkx as nx
from netbone.filters import threshold_filter, fraction_filter
from netbone.backbone import Backbone
from pandas import DataFrame
from netbone.utils.utils import edge_properties

def jaccard(a, b):
    # convert to set
    a = set(a)
    b = set(b)
    # calucate jaccard similarity
    return float(len(a.intersection(b))) / len(a.union(b))

def get_neighbours(graph, node):
    return list(dict(graph[node]).keys()) + [node]

def gspar(data):
    if isinstance(data, DataFrame):
        g = nx.from_pandas_edgelist(data, edge_attr=edge_properties(data))
    elif isinstance(data, nx.Graph):
        g = data.copy()
    else:
        print("data should be a panads dataframe or nx graph")
        return

    for u, v in g.edges():
        g[u][v]['jaccard-sim'] = jaccard(get_neighbours(g, u), get_neighbours(g, v))

    return Backbone(g, method_name="Global Sparsification", property_name="jaccard-sim", ascending=False, compatible_filters=[threshold_filter, fraction_filter], filter_on='Edges')