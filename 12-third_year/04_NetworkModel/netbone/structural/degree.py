import networkx as nx
from netbone.filters import threshold_filter, fraction_filter
from netbone.backbone import Backbone
from pandas import DataFrame
from netbone.utils.utils import edge_properties

def degree(data, weighted=False):
    if isinstance(data, DataFrame):
        g = nx.from_pandas_edgelist(data, edge_attr=edge_properties(data))
    elif isinstance(data, nx.Graph):
        g = data.copy()
    else:
        print("data should be a panads dataframe or nx graph")
        return

    if weighted:
        nx.set_node_attributes(g,dict(g.degree(weight='weight')), name='weighted-degree')
        return Backbone(g, method_name="Weighted Degree", property_name="weighted-degree", ascending=False, compatible_filters=[threshold_filter, fraction_filter], filter_on='Nodes')
    else:
        nx.set_node_attributes(g,dict(g.degree()), name='degree')
        return Backbone(g, method_name="Degree", property_name="degree", ascending=False, compatible_filters=[threshold_filter, fraction_filter], filter_on='Nodes')