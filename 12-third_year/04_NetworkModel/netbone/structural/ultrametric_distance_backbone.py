import pandas as pd
import networkx as nx
from netbone.structural.distanceclosure import backbone as dc_backbone
from netbone.backbone import Backbone
from netbone.filters import boolean_filter


def ultrametric_distance_backbone(data):
    G = data.copy()
    if isinstance(data, pd.DataFrame):
        G = nx.from_pandas_edgelist(data, edge_attr='weight', create_using=nx.Graph())

    for u, v in G.edges():
        G[u][v]['distance'] = 1/G[u][v]['weight']

    um_backbone = dc_backbone.ultrametric_backbone(G, weight='distance')
    nx.set_edge_attributes(G, True, name='in_backbone')

    missing_edges = {edge: {"in_backbone": False} for edge in set(G.edges()).difference(set(um_backbone.edges()))}
    nx.set_edge_attributes(G, missing_edges)

    return Backbone(G, method_name="Ultrametric Distance Filter", property_name="distance", ascending=False, compatible_filters=[boolean_filter], filter_on='Edges')
