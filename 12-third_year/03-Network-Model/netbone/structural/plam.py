import networkx as nx
from netbone.filters import boolean_filter
from netbone.backbone import Backbone
from pandas import DataFrame
from netbone.utils.utils import edge_properties

def get_max_weight_edge(graph, node):
    neighbors = graph.neighbors(node)
    max_weight = float('-inf')
    max_edge = None
    for neighbor in neighbors:
        weight = graph[node][neighbor]['weight']
        if weight > max_weight:
            max_weight = weight
            max_edge = (node, neighbor)
    return max_edge[0], max_edge[1], max_weight

def plam(data):
    if isinstance(data, DataFrame):
        g = nx.from_pandas_edgelist(data, edge_attr=edge_properties(data))
    elif isinstance(data, nx.Graph):
        g = data.copy()
    else:
        print("data should be a panads dataframe or nx graph")
        return

    nx.set_edge_attributes(g, False, name='in_backbone')
    for node in g.nodes():
        source, target, weight = get_max_weight_edge(g, node)
        g[source][target]['in_backbone'] = True

    return Backbone(g, method_name="Primary Linkage Analysis", property_name="weight", ascending=False, compatible_filters=[boolean_filter], filter_on='Edges')