import math
import cProfile
import networkx as nx
from netbone.utils.utils import edge_properties
import pstats


def boolean_filter(backbone, narrate=True, value=[]):
    if boolean_filter in backbone.compatible_filters():
        data = backbone.graph
        column = 'in_backbone'
        if isinstance(data, nx.Graph):
            data = nx.to_pandas_edgelist(data)
        if narrate:
            backbone.narrate()
        return nx.from_pandas_edgelist(data[data[column]], edge_attr=edge_properties(data))
    print("The accepted filters for " + backbone.method_name + " are: " + ', '.join(
        [fun.__name__ for fun in backbone.compatible_filters()]))


def threshold_filter(backbone, value, narrate=True, secondary_property='weight', secondary_property_ascending=False,
                     **kwargs):

    data = backbone.to_dataframe()
    property_name = backbone.property_name
    filter_by = [property_name]
    ascending = [backbone.ascending]

    if backbone.filter_on == 'Edges':
        filter_by.append(secondary_property)
        ascending.append(secondary_property_ascending)

    if threshold_filter in backbone.compatible_filters():
        data = data.sort_values(by=filter_by, ascending=ascending)

        if narrate:
            backbone.narrate()

        if backbone.ascending:
            data = data[data[property_name] < value]
            if backbone.filter_on == 'Edges':
                return nx.from_pandas_edgelist(data, edge_attr=edge_properties(data))
            return backbone.graph.subgraph(list(data.index)).copy()
        else:
            data = data[data[property_name] > value]
            if backbone.filter_on == 'Edges':
                return nx.from_pandas_edgelist(data, edge_attr=edge_properties(data))
            return backbone.graph.subgraph(list(data.index)).copy()

    print("The accepted filters for " + backbone.method_name + " are: " + ', '.join(
        [fun.__name__ for fun in backbone.compatible_filters()]))


def fraction_filter(backbone, value, narrate=True, secondary_property='weight', secondary_property_ascending=False,
                    **kwargs):
    data = backbone.to_dataframe()
    filter_by = [backbone.property_name]
    ascending = [backbone.ascending]

    if backbone.filter_on == 'Edges':
        filter_by.append(secondary_property)
        ascending.append(secondary_property_ascending)

    if fraction_filter in backbone.compatible_filters():
        data = data.sort_values(by=filter_by, ascending=ascending)

        if narrate:
            backbone.narrate()

        if backbone.filter_on == 'Edges':
            value = math.ceil(value * len(data))
            return nx.from_pandas_edgelist(data[:value], edge_attr=edge_properties(data))
        else:
            value = math.ceil(value * len(backbone.graph))
            return backbone.graph.subgraph(list(data[:value].index)).copy()

    print("The accepted filters for " + backbone.method_name + " are: " + ', '.join(
        [fun.__name__ for fun in backbone.compatible_filters()]))
