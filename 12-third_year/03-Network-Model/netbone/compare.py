import pandas as pd
import networkx as nx
from netbone.utils.utils import cumulative_dist
from netbone.filters import threshold_filter, boolean_filter
from pandas import DataFrame
from scipy.stats import kstest


class Compare:
    def __init__(self):
        self.network = nx.Graph()
        self.backbones = []
        self.props = dict()
        self.value_name = 'p_value'
        self.filter = threshold_filter
        self.filter_values = []

    def set_network(self, network):
        if isinstance(network, DataFrame):
            columns = list(network.columns)
            columns.remove('source')
            columns.remove('target')
            network = nx.from_pandas_edgelist(network, edge_attr=columns)
        self.network = network

    def add_backbone(self, backbone):
        self.backbones.append(backbone)

    def add_property(self, name, property):
        self.props[name] = property

    def set_filter(self, filter, values=[]):
        self.filter = filter
        self.filter_values = values
        if "fraction" in filter.__name__:
            self.value_name = 'Fraction of Edges'
        elif "threshold" in filter.__name__:
            self.value_name = 'P-value'

    def properties(self):
        if self.filter == boolean_filter:
            self.filter_values = [0] * len(self.backbones)
        if self.filter_values == []:
            raise Exception('Please enter the filter values.')

        results = pd.DataFrame(index=['Original'] + [backbone.method_name for backbone in self.backbones])
        props_arrays = dict()

        for property in self.props:
            props_arrays[property] = [self.props[property](self.network, self.network)]

        for i, backbone in enumerate(self.backbones):
            if self.filter == boolean_filter:
                extracted_backbone = self.filter(backbone, narrate=False)
            else:
                extracted_backbone = self.filter(backbone, value=self.filter_values[i], narrate=False)

            for property in self.props:
                props_arrays[property].append(self.props[property](self.network, extracted_backbone))

        for property in self.props:
            results[property] = props_arrays[property]

        return results

    def properties_progression(self):
        if self.filter == boolean_filter:
            raise Exception('Cann\'t apply the boolean filter in this function.')
        if self.filter_values == []:
            raise Exception('Please enter the filter values.')
        props_res = dict()
        for property in self.props:
            props_res[property] = pd.DataFrame(index=[backbone.method_name for backbone in self.backbones])
        for value in self.filter_values:
            temp_props = dict()
            for property in self.props:
                temp_props[property] = []

            for backbone in self.backbones:
                extracted_backbone = self.filter(backbone, value=value, narrate=False)

                for property in self.props:
                    temp_props[property].append(self.props[property](self.network, extracted_backbone))

            for property in self.props:
                props_res[property][value] = temp_props[property]

        for res in props_res:
            props_res[res] = props_res[res].T
            props_res[res].index.name = self.value_name
        return props_res

    def distribution_ks_statistic(self, increasing=True, consent=False):
        if self.filter == boolean_filter:
            self.filter_values = [0] * len(self.backbones)
        if self.filter_values == []:
            raise Exception('Please enter the filter values.')
        cons = []
        if consent == False:
            for backbone in self.backbones:
                cons.append(False)
            consent = cons

        dist = dict()
        if True in consent:
            ks_statistics = pd.DataFrame(index=[backbone.method_name for backbone in self.backbones] + ['Consensual Backbone'])
        else:
            ks_statistics = pd.DataFrame(index=[backbone.method_name for backbone in self.backbones])
        for property in self.props:
            dist_values = dict()
            vals = []
            values0 = self.props[property](self.network)
            dist_values['Original'] = cumulative_dist(property, 'Original', values0, increasing)

            if True in consent:
                consensual_backbone = ''
                nodes_labels = dict(zip(self.network.nodes(), nx.convert_node_labels_to_integers(self.network.copy()).nodes()))
                inverse_nodes_labels = dict(zip(nx.convert_node_labels_to_integers(self.network.copy()).nodes(), self.network.nodes()))


            for i, backbone in enumerate(self.backbones):
                extracted_backbone = self.filter(backbone, value=self.filter_values[i], narrate=False)
                if consent[i]:
                    if consensual_backbone == '':
                        consensual_backbone = nx.relabel_nodes(extracted_backbone, nodes_labels)
                    else:
                        extracted_backbone = nx.relabel_nodes(extracted_backbone, nodes_labels)
                        old_consensual = consensual_backbone.copy()
                        consensual_backbone.remove_nodes_from(n for n in old_consensual if n not in extracted_backbone)
                        consensual_backbone.remove_edges_from(e for e in old_consensual.edges if e not in extracted_backbone.edges)

                values1 = self.props[property](extracted_backbone)
                dist_values[backbone.method_name] = cumulative_dist(property, backbone.method_name, values1, increasing)
                vals.append(kstest(values0, values1)[0])
            if consent[i]:
                consensual_backbone.remove_nodes_from(list(nx.isolates(consensual_backbone)))
                consensual_backbone = nx.relabel_nodes(consensual_backbone, inverse_nodes_labels)
                values1 = self.props[property](consensual_backbone)
                dist_values['Consensual Backbone'] = cumulative_dist(property, 'Consensual Backbone', values1, increasing)
                vals.append(kstest(values0, values1)[0])

            # ks_statistics = pd.DataFrame(index=['Original'] + [backbone.name for backbone in self.backbones])
            dist[property] = dist_values
            ks_statistics[property] = vals

        if True in consent:
            return ks_statistics, dist, consensual_backbone
        else:
            return ks_statistics, dist

    # def distribution_ks_statistic(self, increasing=True, consent=True):
    #     if self.filter == boolean_filter:
    #         self.filter_values = [0] * len(self.backbones)
    #     if self.filter_values == []:
    #         raise Exception('Please enter the filter values.')
    #
    #     dist = dict()
    #     if consent:
    #         ks_statistics = pd.DataFrame(index=[backbone.method_name for backbone in self.backbones] + ['Consensual Backbone'])
    #     else:
    #         ks_statistics = pd.DataFrame(index=[backbone.method_name for backbone in self.backbones])
    #     for property in self.props:
    #         dist_values = dict()
    #         vals = []
    #         values0 = self.props[property](self.network)
    #         dist_values['Original'] = cumulative_dist(property, 'Original', values0, increasing)
    #
    #         if consent:
    #             consensual_backbone = ''
    #             nodes_labels = dict(zip(self.network.nodes(), nx.convert_node_labels_to_integers(self.network.copy()).nodes()))
    #             inverse_nodes_labels = dict(zip(nx.convert_node_labels_to_integers(self.network.copy()).nodes(), self.network.nodes()))
    #
    #
    #         for i, backbone in enumerate(self.backbones):
    #             extracted_backbone = self.filter(backbone, value=self.filter_values[i], narrate=False)
    #             if consent:
    #                 if i==0:
    #                     consensual_backbone = nx.relabel_nodes(extracted_backbone, nodes_labels)
    #                 else:
    #                     extracted_backbone = nx.relabel_nodes(extracted_backbone, nodes_labels)
    #                     old_consensual = consensual_backbone.copy()
    #                     consensual_backbone.remove_nodes_from(n for n in old_consensual if n not in extracted_backbone)
    #                     consensual_backbone.remove_edges_from(e for e in old_consensual.edges if e not in extracted_backbone.edges)
    #
    #             values1 = self.props[property](extracted_backbone)
    #             dist_values[backbone.method_name] = cumulative_dist(property, backbone.method_name, values1, increasing)
    #             vals.append(kstest(values0, values1)[0])
    #         if consent:
    #             consensual_backbone = nx.relabel_nodes(consensual_backbone, inverse_nodes_labels)
    #             values1 = self.props[property](consensual_backbone)
    #             dist_values['Consensual Backbone'] = cumulative_dist(property, 'Consensual Backbone', values1, increasing)
    #             vals.append(kstest(values0, values1)[0])
    #
    #         # ks_statistics = pd.DataFrame(index=['Original'] + [backbone.name for backbone in self.backbones])
    #         dist[property] = dist_values
    #         ks_statistics[property] = vals
    #
    #     if consent:
    #         return ks_statistics, dist, consensual_backbone
    #     else:
    #         return ks_statistics, dist

    def consent(self):
        if self.filter == boolean_filter:
            self.filter_values = [0] * len(self.backbones)
        if self.filter_values == []:
            raise Exception('Please enter the filter values.')

        nodes_labels = dict(zip(self.network.nodes(), nx.convert_node_labels_to_integers(self.network.copy()).nodes()))
        inverse_nodes_labels = dict(zip(nx.convert_node_labels_to_integers(self.network.copy()).nodes(), self.network.nodes()))
        consensual_backbone = ''
        for i, backbone in enumerate(self.backbones):
            extracted_backbone = self.filter(backbone, value=self.filter_values[i], narrate=False)
            if i==0:
                consensual_backbone = nx.relabel_nodes(extracted_backbone, nodes_labels)
            else:
                extracted_backbone = nx.relabel_nodes(extracted_backbone, nodes_labels)
                old_consensual = consensual_backbone.copy()
                consensual_backbone.remove_nodes_from(n for n in old_consensual if n not in extracted_backbone)
                consensual_backbone.remove_edges_from(e for e in old_consensual.edges if e not in extracted_backbone.edges)

        consensual_backbone.remove_nodes_from(list(nx.isolates(consensual_backbone)))
        return nx.relabel_nodes(consensual_backbone, inverse_nodes_labels)
