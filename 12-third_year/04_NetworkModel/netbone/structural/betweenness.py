import networkx as nx
from netbone.filters import threshold_filter, fraction_filter
from netbone.backbone import Backbone
from pandas import DataFrame
from netbone.utils.utils import edge_properties

def betweenness(data, weighted=True, normalized=True):
    if isinstance(data, DataFrame):
        g = nx.from_pandas_edgelist(data, edge_attr=edge_properties(data))
    elif isinstance(data, nx.Graph):
        g = data.copy()
    else:
        print("data should be a panads dataframe or nx graph")
        return


    if weighted:
        nx.set_edge_attributes(g, nx.edge_betweenness_centrality(g, normalized=normalized, weight='weight', seed=100), name='weighted-betweenness')
        return Backbone(g, method_name="Weighted Betweenness", property_name="weighted-betweenness", ascending=False, compatible_filters=[threshold_filter, fraction_filter], filter_on='Edges')
    else:
        nx.set_edge_attributes(g, nx.edge_betweenness_centrality(g, normalized=normalized, seed=100), name='betweenness')
        return Backbone(g, method_name="Betweenness", property_name="betweenness", ascending=False, compatible_filters=[threshold_filter, fraction_filter], filter_on='Edges')


