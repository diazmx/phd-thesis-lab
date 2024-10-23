import networkx as nx
from pandas import DataFrame
from networkx import Graph
from netbone.utils.utils import edge_properties
from netbone.backbone import Backbone
from netbone.filters import boolean_filter
import numpy as np

def compute_mad(values):
    values = np.array(list(values))
    values = values - np.median(values)
    return np.median(np.absolute(values))

def mad(data, property='degree', weighted=True):
    if isinstance(data, DataFrame):
        g = nx.from_pandas_edgelist(data, edge_attr=edge_properties(data))
    elif isinstance(data, Graph):
        g = data.copy()
    else:
        print("data should be a panads dataframe or nx graph")
        return

    if property == 'degree':
        nx.set_node_attributes(g, dict(g.degree(weight='weight' if weighted else None)), name='degree')

    property_dict = nx.get_node_attributes(g,property)
    mad = compute_mad(property_dict.values())
    nodes = g.nodes()
    nx.set_edge_attributes(g, False, name='in_backbone')
    nx.set_node_attributes(g, mad, name='MAD')
    for u, v in g.edges():
        if (nodes[u][property] >= mad) & (nodes[v][property] >= mad):
            g[u][v]['in_backbone'] = True

    return Backbone(g, method_name="Median Absolute Deviation Filter", property_name=property, ascending=False, compatible_filters=[boolean_filter], filter_on='Edges')
