import networkx as nx
import numpy as np
from igraph import Graph
from math import log2, ceil
from collections import Counter
import matplotlib.pyplot as plt


def build_network_model(data, usr_id_name, res_id_name, file_path=None):
    """
    Builds the Access Requests Bipartite Network from Access log.

    Args:
        data (pandas dataframe): The Access Log.
        usr_id_name (str): The name of the ID users column in the Access Log
        res_id_name (str): The name of the ID resources column in the Access Log

    Returns:
        Graph (iGraph): The Access Requests Bipartite Network.

    Raises:
        TypeError: If a network is not Bipartite.
    """

    list_of_edges = []
    bi_network = nx.Graph()  # NetworkX Graph object

    for usr_idx, rsr_idx in data[[usr_id_name, res_id_name]].values:
        # list_of_edges.append((int(usr_idx), int(rsr_idx)))  # Tuple of edges
        list_of_edges.append((usr_idx, rsr_idx))  # Tuple of edges
    bi_network.add_edges_from(list_of_edges)  # Build Network with edges

    # Change networkX object to iGraph object
    bi_network = Graph.from_networkx(bi_network)
    bi_network.vs['name'] = bi_network.vs["_nx_name"]  # Clean name column
    del bi_network.vs["_nx_name"]  # Remove uncleaned name column

    if not bi_network.is_bipartite():
        raise TypeError("The ARBN is not bipartite")

    # Add type of node (user or resource)
    list_of_resources_in_data = list(data[res_id_name])
    list_node_type = []
    for node in bi_network.vs():
        if node['name'] in list_of_resources_in_data:
            list_node_type.append(1)  # A resource
        else:
            list_node_type.append(0)  # An user

    bi_network.vs["typen"] = list_node_type

    # End node type

    if not file_path == None:  # Create a file
        bi_network.write(file_path)

    print("ARBN builded!")
    print(bi_network.summary())
    print("|U-Nodes| =", len(bi_network.vs.select(typen=0)))
    print("|R-Nodes| =", len(bi_network.vs.select(typen=1)))

    return bi_network
# END build_network_model


def get_edge_weight(i_node, j_node):
    """
    Compute the weight of an edge between i and j nodes.

    Args:
        i_node (networkX node): i node.
        j_node (networkX node): j node.

    Returns:
        weight (float): The weight between nodes.

    Raises:
        TypeError: if there are not an intersection
    """
    neighs_i = set(i_node.neighbors())  # Set of neighbors of i
    neighs_j = set(j_node.neighbors())  # Set of neighbors of j

    # Calculate intersection between two previous sets
    insersection_neighbors = neighs_i.intersection(neighs_j)

    weight = (len(insersection_neighbors)*len(insersection_neighbors)
              ) / (len(neighs_i)*len(neighs_j))

    return weight
# END get_edge_weight


def bipartite_projection(biparte_network, node_type=0):
    """
    Generate a monopartite network from bipartite network.

    Parameters:
        bipartite_network (igraph Graph): The bipartie network.
        node_type (int): The set of nodes of the monopartite network.

    Returns:
        Graph (iGraph): The monopartite (projected) network.

    Raises:
        Some
    """

    # Check if the bipartite network is a bipartite network:
    if not biparte_network.is_bipartite():
        raise TypeError("The ARBN is not bipartite")

    # networkX object (more easy to buil)
    g = nx.Graph()

    # All opposite node set
    opposite_nodes = biparte_network.vs.select(typen=1)

    # Check for every node the same type
    for X_node in opposite_nodes:
        # Select all neighbors of the X_node
        neighbordhood = X_node.neighbors()

        for Y_node_i in neighbordhood:
            for Y_node_j in neighbordhood:
                # Ceck if both nodes are the same
                if Y_node_i['name'] != Y_node_j['name']:
                    # If there is no an edge generate
                    if not g.has_edge(Y_node_i['name'], Y_node_j['name']):
                        weight_ = get_edge_weight(Y_node_i, Y_node_j)
                        # print("Peso: ", Y_node_i['name'], "-", Y_node_j['name'], " => ", weight_)
                        g.add_edge(Y_node_i["name"], Y_node_j["name"],
                                   weight=weight_)

    # Convert from networkX graph to igraph graph
    g = Graph.from_networkx(g)
    g.vs["name"] = g.vs["_nx_name"]
    del g.vs["_nx_name"]

    for u_nodes in g.vs:
        rsrcs = biparte_network.vs.find(name=u_nodes["name"]).neighbors()
        rsrcs = [r_node["name"] for r_node in rsrcs]
        u_nodes["rsrcs"] = rsrcs

    print("User Network builded!")
    print(g.summary())
    return g
# END bipartite_projection


def calculate_log_binning(degree_distribution, n_bins):
    """Compute the log-binning y-values in the degree distribution.

    Divides the degree distribution in `n_bins` segments.

    Parameters
    ----------
    degree_distribution: list
        Network degree distribution.
    n_bins:
        Number of bins to assign.

    Returns
    -------
    (list, list)
        The (x_values, y_values_log_bin_list) tuple.
    """
    current_sum = 0
    previous_k = 0
    y_values_log_bin_list = []
    x_values = []

    for i in range(1, n_bins):
        x_values.append(previous_k)
        current_k = 2 ** (i)
        current_sum = current_sum + current_k
        temp_y_value = sum(degree_distribution[previous_k:current_k])
        temp_y_value = temp_y_value / (current_k-previous_k)
        y_values_log_bin_list.append(temp_y_value)
        previous_k = current_k

        if current_sum > len(degree_distribution):
            x_values.append(previous_k)
            temp_y_value = sum(
                degree_distribution[previous_k:len(degree_distribution)])
            temp_y_value = temp_y_value / (len(degree_distribution)-previous_k)
            y_values_log_bin_list.append(temp_y_value)
            break

    return x_values, y_values_log_bin_list
# END calculate_log_binning


def plot_distribution_degree(user_network, name_ds):
    max_degree = max(user_network.degree())

    degree_list = np.zeros(max_degree, dtype=int)

    for node in user_network.vs():
        degree_list[node.degree()-1] = (degree_list[node.degree()-1] + 1)

    degree_list = degree_list / user_network.vcount()

    fig, axs = plt.subplots(2, 2, figsize=(20, 20))

    # LINEAR SCALE
    axs[0, 0].set_title('Linear Scale')
    axs[0, 0].plot(degree_list, color='#e60049')
    axs[0, 0].set(ylabel='P(k)', xlabel='k')

    # LINEAR BINNING
    axs[0, 1].set_title('Linear Binning')
    axs[0, 1].plot(range(max_degree), degree_list,
                   color="#e60049", marker='*', ls='None')
    axs[0, 1].loglog()
    axs[0, 1].set(ylabel='P(k)', xlabel='k')

    # LOG-BINNING
    axs[1, 0].set_title('Log-Binning')
    n_log_bin = ceil(log2(max(user_network.degree())))
    x_values, y_values = calculate_log_binning(degree_list, n_log_bin)
    axs[1, 0].plot(x_values, y_values, color='#e60049',
                   marker="D", ls='None')
    axs[1, 0].loglog()
    axs[1, 0].set(ylabel='P(k)', xlabel='k')

    # COMULATIVE
    axs[1, 1].set_title("Cumulative Distribution")
    degree_sequence = sorted(
        [d for d in user_network.degree()], reverse=True)  # degree sequence
    degreeCount = Counter(degree_sequence)
    deg, cnt = zip(*degreeCount.items())
    cs = np.cumsum(np.array(cnt)/user_network.vcount())
    axs[1, 1].loglog(deg, cs, color='#e60049', marker="D", ls='None')
    axs[1, 1].set(ylabel='P(k)', xlabel='k')

    fig.suptitle(
        "Degree distribution - "+name_ds+" - Frequent resources", fontsize=24)
    fig.tight_layout()
    plt.savefig(name_ds+' - Degree Distribution.png')
