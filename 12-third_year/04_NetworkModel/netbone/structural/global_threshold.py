import networkx as nx
from pandas import DataFrame
from networkx import Graph,to_pandas_edgelist
from netbone.utils.utils import edge_properties
from netbone.backbone import Backbone
from netbone.filters import fraction_filter, threshold_filter
def global_threshold(data):

    if isinstance(data, DataFrame):
        g = nx.from_pandas_edgelist(data, edge_attr=edge_properties(data))
    elif isinstance(data, Graph):
        g = data.copy()
    else:
        print("data should be a panads dataframe or nx graph")
        return

    return Backbone(g, method_name="Global Threshold Filter", property_name="weight", ascending=False, compatible_filters=[fraction_filter, threshold_filter], filter_on='Edges')