import networkx as nx
import pandas as pd
from netbone.backbone import Backbone
from netbone.filters import threshold_filter, fraction_filter


def lans(data):
    if isinstance(data, pd.DataFrame):
        g = nx.from_pandas_edgelist(data, edge_attr='weight', create_using=nx.Graph())
    elif isinstance(data, nx.Graph):
        g = data.copy()
    else:
        print("data should be a panads dataframe or nx graph")
        return
    for u, v, w in g.edges(data='weight'):
        g[u][v]['p_value'] = min(compute_pvalue(g, v, w), compute_pvalue(g, u, w))
    return Backbone(g, method_name="Locally Adaptive Network Sparsification Filter", property_name="p_value", ascending=True,
                    compatible_filters=[threshold_filter, fraction_filter], filter_on='Edges')


def compute_pvalue(G, node, w):
    u_degree = G.degree(node, weight='weight')
    puv = w / u_degree
    u_n = G[node]
    count = len([n for n in u_n if u_n[n]['weight'] / u_degree <= puv])
    return 1 - count / len(u_n)
