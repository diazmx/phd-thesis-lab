import warnings
import numpy as np
import pandas as pd
import networkx as nx
from netbone.backbone import Backbone
from netbone.filters import boolean_filter, threshold_filter, fraction_filter

# algo: doubly_stochastic.py
warnings.filterwarnings('ignore')


def doubly_stochastic(data):
    undirected = True
    return_self_loops = False

    if isinstance(data, pd.DataFrame):
        table = data.copy()
    elif isinstance(data, nx.Graph):
        table = nx.to_pandas_edgelist(data)
    else:
        print("data should be a panads dataframe or nx graph")
        return

    table2 = table.copy()
    original_nodes = len(set(table["source"]) | set(table["target"]))
    table = pd.pivot_table(table, values="weight", index="source",
                           columns="target", aggfunc="sum", fill_value=0) + .0001
    row_sums = table.sum(axis=1)
    attempts = 0
    while np.std(row_sums) > 1e-12:
        table = table.div(row_sums, axis=0)
        col_sums = table.sum(axis=0)
        table = table.div(col_sums, axis=1)
        row_sums = table.sum(axis=1)
        attempts += 1
        if attempts > 1000:
            warnings.warn(
                "Matrix could not be reduced to doubly stochastic. See Sec. 3 of Sinkhorn 1964", RuntimeWarning)
            return pd.DataFrame()
    table = pd.melt(table.reset_index(), id_vars="source")
    table = table[table["source"] < table["target"]]
    table = table[table["value"] > 0].sort_values(by="value", ascending=False)
    table = table.merge(table2[["source", "target", "weight"]], on=[
        "source", "target"])
    i = 0
    doubly_nodes = len(set(table["source"]) | set(table["target"]))
    edges = table.shape[0]
    if undirected:
        G = nx.Graph()
        while nx.number_connected_components(G) != 1 or len(G) < doubly_nodes or nx.is_connected(G) == False:
            if i == edges:
                break
            edge = table.iloc[i]
            G.add_edge(edge["source"], edge["target"], weight=edge["value"])
            table.loc[table.loc[(table['source'] == edge["source"]) & (
                    table['target'] == edge["target"])].index[0], 'in_backbone'] = True
            i += 1
    else:
        G = nx.DiGraph()
        while nx.number_weakly_connected_components(G) != 1 or len(G) < doubly_nodes or nx.is_connected(G) == False:
            if i == edges:
                break
            edge = table.iloc[i]
            G.add_edge(edge["source"], edge["target"], weight=edge["value"])
            table.loc[table.loc[(table['source'] == edge["source"]) & (
                    table['target'] == edge["target"])].index[0], 'in_backbone'] = True
            i += 1

    # table = pd.melt(nx.to_pandas_adjacency(G).reset_index(), id_vars = "index")
    table = table[table["value"] >= 0]
    table.rename(columns={"index": "source",
                          "variable": "target", "value": "score"}, inplace=True)
    table = table.fillna(False)
    if not return_self_loops:
        table = table[table["source"] != table["target"]]
    if undirected:
        table = table[table["source"] <= table["target"]]

    return Backbone(nx.from_pandas_edgelist(table, edge_attr=['weight', 'score', 'in_backbone']),
                    method_name="Doubly Stochastic Filter", property_name="score", ascending=False,
                    compatible_filters=[boolean_filter, threshold_filter, fraction_filter], filter_on='Edges')
