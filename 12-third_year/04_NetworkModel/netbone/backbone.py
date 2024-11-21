import networkx as nx
import pandas as pd
from pandas import DataFrame
from netbone.utils.utils import edge_properties


class Backbone:

    def __init__(self, graph, method_name, property_name, ascending, compatible_filters, filter_on):
        if isinstance(graph, DataFrame):
            graph = nx.from_pandas_edgelist(graph, edge_attr=edge_properties(graph))

        self.graph = graph
        self.method_name = method_name
        self.property_name = property_name
        self.ascending = ascending
        self.filters = compatible_filters
        self.filter_on = filter_on

    def to_dataframe(self):
        if self.filter_on == 'Edges':
            return nx.to_pandas_edgelist(self.graph)
        else:
            node_attrs = {}
            for node in self.graph.nodes():
                node_attrs[node] = self.graph.nodes[node]
            # Convert the dictionary to a Pandas DataFrame
            return pd.DataFrame.from_dict(node_attrs, orient='index')

    def narrate(self):
        match self.method_name:
            case "Disparity Filter":
                print(self.method_name)
            case "Enhanced Configuration Model Filter":
                print(self.method_name)
            case "Marginal Likelihood Filter":
                print(self.method_name)
            case "Locally Adaptive Network Sparsification Filter":
                print(self.method_name)
            case "Noise Corrected Filter":
                print(self.method_name)
            case 'High Salience Skeleton Filter':
                print(self.method_name)
            case 'Modularity Filter':
                print(self.method_name)
            case 'Ultrametric Distance Filter':
                print(self.method_name)
            case 'Maximum Spanning Tree':
                print(self.method_name)
            case 'Metric Distance Filter':
                print(self.method_name)
            case 'H-Backbone Filter':
                print(self.method_name)
            case 'Doubly Stochastic Filter':
                print(self.method_name)
            case 'Global Threshold Filter':
                print(self.method_name)
            case _:
                print("Citation here")

    def compatible_filters(self):
        return self.filters
