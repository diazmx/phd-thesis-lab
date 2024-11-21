import numpy as np
import pandas as pd
import networkx as nx


def lcc(G):
    return G.subgraph(max(nx.connected_components(G), key=len)).copy()
def cumulative_dist(name, method, values, increasing=True):
    if increasing:
        x = -np.sort(-np.array(values))
    else:
        x = np.sort(values)
    y = np.arange(1, len(x) + 1)/len(x)

    df = pd.DataFrame(index=x)
    df.index.method_name = name
    df[method] = y
    return df

def edge_properties(df):
    columns = list(df.columns)
    columns.remove('source')
    columns.remove('target')
    return columns
