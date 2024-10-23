import numpy as np
import networkx as nx
from netbone.filters import boolean_filter
from netbone.backbone import Backbone
from pandas import DataFrame
from netbone.utils.utils import edge_properties
from math import isnan
def get_neighbor_weights(graph, node):
    # Get the neighbors and weights of the given node from the graph
    neighbors = graph[node].keys()
    weights = [graph[node][neighbor]['weight'] for neighbor in neighbors]

    # Calculate the total weight
    total_weight = sum(weights)

    # Normalize the weights
    normalized_weights = [weight / total_weight * 100 for weight in weights]

    # Sort the neighbors based on the normalized weights in descending order
    sorted_neighbors = sorted(zip(neighbors, normalized_weights), key=lambda x: x[1], reverse=True)

    return dict(sorted_neighbors)

def get_ideal_distribution(i, total):
    array = [0] * total  # initialize the array with zeros
    percentage = 100 / (i + 1)  # calculate the percentage value for the current loop
    for j in range(i + 1):
        array[j] = percentage # format the percentage value with two decimal places
    return array

def compute_cod(f, y):
    corr_matrix  = np.corrcoef(f,y)
    corr = corr_matrix[0,1]
    return round(corr**2, 2)


def mlam(data):
    if isinstance(data, DataFrame):
        g = nx.from_pandas_edgelist(data, edge_attr=edge_properties(data))
    elif isinstance(data, nx.Graph):
        g = data.copy()
    else:
        print("data should be a panads dataframe or nx graph")
        return

    nx.set_edge_attributes(g, False, name='in_backbone')
    for node in g.nodes():
        edge_index = 0
        neighbors_weights =  get_neighbor_weights(g, node)
        real_distribution = list(neighbors_weights.values())
        neighbors_count = len(neighbors_weights)
        old_cod = 0
        if neighbors_count != 1:
            for i in range(neighbors_count):
                new_cod = compute_cod(real_distribution, get_ideal_distribution(i, neighbors_count))
                if isnan(new_cod):
                    break
                if old_cod <= new_cod:
                    old_cod = new_cod
                    edge_index = i
                else:
                    break
                if i == neighbors_count-1:
                    edge_index = i

        for j, neighbour in enumerate(neighbors_weights.keys()):
            if j>edge_index:
                break
            g[node][neighbour]['in_backbone'] = True


    return Backbone(g, method_name="Multiple Linkage Analysis", property_name="weight", ascending=False, compatible_filters=[boolean_filter], filter_on='Edges')