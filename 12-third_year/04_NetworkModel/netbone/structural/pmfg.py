import networkx as nx
from netbone.filters import boolean_filter
from netbone.backbone import Backbone
from pandas import DataFrame
from networkx import Graph
from netbone.utils.utils import edge_properties

def pmfg(data):
    if isinstance(data, DataFrame):
        table = data.copy()
    elif isinstance(data, Graph):
        table = nx.to_pandas_edgelist(data)
    else:
        print("data should be a panads dataframe or nx graph")
        return

    g = nx.from_pandas_edgelist(table, edge_attr=edge_properties(table))
    nx.set_edge_attributes(g, False, name='in_backbone')

    backbone = nx.Graph()
    table = table.sort_values(by='weight', ascending=False)

    for row in table.itertuples():
        backbone.add_edge(row.source, row.target)
        if not nx.is_planar(backbone):
            backbone.remove_edge(row.source, row.target)
        else:
            g[row.source][row.target]['in_backbone'] = True
        if len(backbone.edges()) == 3*(len(g)-2):
            break

    return Backbone(g, method_name="Planar Maximally Filtered Graph", property_name="weight", ascending=False, compatible_filters=[boolean_filter], filter_on='Edges')