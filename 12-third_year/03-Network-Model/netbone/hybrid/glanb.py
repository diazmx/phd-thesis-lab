import networkx as nx
import igraph as ig
import pandas as pd
from netbone.backbone import Backbone
from netbone.filters import threshold_filter, fraction_filter


def count_included_subarrays(arrays, target_array):
    count = 0
    target_len = len(target_array)
    for array in arrays:
        array_len = len(array)
        for i in range(array_len - target_len + 1):
            if array[i:i + target_len] == target_array:
                count += 1
    return count


def glanb(data, c=-1):
    if isinstance(data, pd.DataFrame):
        graph = nx.from_pandas_edgelist(data, edge_attr='weight', create_using=nx.Graph())
    elif isinstance(data, nx.Graph):
        graph = data.copy()
    else:
        print("data should be a panads dataframe or nx graph")
        return
   
    if c == -1:
        print("Please send the c value")
        return
    # convert weights to distances
    wes = nx.get_edge_attributes(graph, 'weight')
    values = {pair: 1 / wes[pair] for pair in wes}
    nx.set_edge_attributes(graph, values, name='distance')

    node_labels = dict(zip(graph.nodes(), range(len(graph))))
    igraph = ig.Graph.from_networkx(graph)
    for source in graph.nodes():
        k_i = graph.degree[source]
        if k_i > 1:
            ig_paths = igraph.get_all_shortest_paths(node_labels[source], weights='distance')
            for u, v in graph.edges(source):
                g_ij = count_included_subarrays(ig_paths, [node_labels[u], node_labels[v]])
                g_is = len(ig_paths) - 1
                I_ij = (g_ij / g_is)
                S_ij = (1 - I_ij) ** ((k_i - 1) ** c)
                if 'SI' in graph[u][v]:
                    if S_ij < graph[u][v]['SI']:
                        graph[u][v]['SI'] = S_ij
                else:
                    graph[u][v]['SI'] = S_ij
    return Backbone(graph, method_name="Globally and Locally Adaptive Backbone Filter", property_name="SI",
                    ascending=True, compatible_filters=[threshold_filter, fraction_filter], filter_on='Edges')
