from netbone.backbone import Backbone
from netbone.filters import threshold_filter, fraction_filter
import community.community_louvain as community
from scipy.sparse import csr_matrix
from scipy.sparse import diags
import networkx as nx
import pandas as pd
import numpy as np


#
# def swap_key_value_dict(old_dict):
#     new_dict = {}
#     for key, value in old_dict.items():
#         if value not in new_dict:
#             new_dict[value] = []
#         new_dict[value].append(key)
#     return new_dict

def modularity_backbone(data):
    if isinstance(data, pd.DataFrame):
        g = nx.from_pandas_edgelist(data, edge_attr='weight', create_using=nx.Graph())
    elif isinstance(data, nx.Graph):
        g = data.copy()
    else:
        print("data should be a panads dataframe or nx graph")
        return

    node_communities = community.best_partition(g, random_state=123)
    modularity_value = community.modularity(node_communities, g)
    # communities = swap_key_value_dict(node_communities)

    membership = list(node_communities.values())

    weight_key = None
    index = list(range(len(g)))
    m = sum([g.degree(node, weight=weight_key) for node in g.nodes()]) / 2

    A = nx.to_scipy_sparse_matrix(g)

    vals = np.ones(len(membership))
    group_indicator_mat = csr_matrix((vals, (index, membership)), shape=(len(g), max(membership) + 1))

    node_deg_by_group = A * group_indicator_mat

    internal_edges = node_deg_by_group[index, membership].sum() / 2

    degrees = node_deg_by_group.sum(1)
    degrees = np.array(degrees).flatten()
    deg_mat = csr_matrix((degrees, (index, membership)),
                         shape=node_deg_by_group.shape)
    degrees = degrees[:, np.newaxis]

    node_deg_by_group += deg_mat

    group_degs = (deg_mat + diags(A.diagonal()) * group_indicator_mat).sum(0)

    internal_deg = node_deg_by_group[index, membership].transpose() - degrees

    q1_links = (internal_edges - internal_deg) / (m - degrees)

    expected_impact = np.power(group_degs, 2).sum() - 2 * (node_deg_by_group * group_degs.transpose()) + \
                      node_deg_by_group.multiply(node_deg_by_group).sum(1)
    q1_degrees = expected_impact / (4 * (m - degrees) ** 2)
    q1s = q1_links - q1_degrees
    q1s = np.array(q1s).flatten()

    vitalities = (modularity_value - q1s).tolist()

    nx.set_node_attributes(g, dict(zip(list(g.nodes()), np.absolute(vitalities))), name='vitality')

    return Backbone(g, method_name="Modularity Filter", property_name='vitality', ascending=False,
                    compatible_filters=[threshold_filter, fraction_filter], filter_on='Nodes')
