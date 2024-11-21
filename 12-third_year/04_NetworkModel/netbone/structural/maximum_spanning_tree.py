import networkx as nx
import pandas as pd
from netbone.backbone import Backbone
from netbone.filters import boolean_filter
from netbone.utils.utils import edge_properties
# algo: minimum_spanning_tree
# calculating MSP

def maximum_spanning_tree(data):
    if isinstance(data, pd.DataFrame):
        g = nx.from_pandas_edgelist(data, edge_attr=edge_properties(data))
    elif isinstance(data, nx.Graph):
        g = data.copy()
    else:
        print("data should be a panads dataframe or nx graph")
        return

    nx.set_edge_attributes(g, True, name='in_backbone')
    msp = nx.maximum_spanning_tree(g, weight='weight')

    missing_edges = {edge: {"in_backbone": False} for edge in set(g.edges()).difference(set(msp.edges()))}
    nx.set_edge_attributes(g, missing_edges)

    return Backbone(g, method_name="Maximum Spanning Tree", property_name="weight", ascending=False, compatible_filters=[boolean_filter], filter_on='Edges')



