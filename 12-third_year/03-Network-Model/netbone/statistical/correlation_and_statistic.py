import math
import networkx as nx
import pandas as pd
from netbone.backbone import Backbone
from netbone.filters import threshold_filter, fraction_filter, boolean_filter


def correlation_and_statistic(data):
    if isinstance(data, pd.DataFrame):
        g = nx.from_pandas_edgelist(data, edge_attr='weight', create_using=nx.Graph())
    elif isinstance(data, nx.Graph):
        g = data.copy()
    else:
        print("data should be a panads dataframe or nx graph")
        return
    nx.set_edge_attributes(g, False, name='in_backbone')
    strengths = g.degree(weight='weight')
    weight_sum = sum(list(nx.get_edge_attributes(g, 'weight').values()))
    for u, v, w in g.edges(data='weight'):
        phij = (w * weight_sum - strengths[u] * strengths[v]) / math.sqrt(strengths[u] * strengths[v] * (weight_sum - strengths[u])*(weight_sum - strengths[v]))
        D = max(strengths[u], strengths[v])
        tij = (phij * math.sqrt(D - 2)) / math.sqrt(1 - phij**2)
        g[u][v]['t-statistic'] = tij
        g[u][v]['phi'] = phij
        if tij>2.59:
            g[u][v]['in_backbone'] = True


    return Backbone(g, method_name="Correlation and Statistic Method", property_name="t-statistic", ascending=False, compatible_filters=[boolean_filter, threshold_filter, fraction_filter], filter_on='Edges')
