"""
This file contains functions to build:
    - Access Control Bipartite Network
    - User Network (Bipartite Projection)
"""


import networkx as nx
from igraph import Graph


def check_all_ok(vertex_seq, attr_name):
    """
    Check if all vertex of this sequency have null values.
    """
    sum_nones = sum(x is not "None" for x in vertex_seq[attr_name])
    return sum_nones == len(vertex_seq)


def add_attrs(bip_network, df_attrs_nodes, df_attrs_res):
    """
    Add the attributes in a nodes seq.

    Args:
        bip_network (igraph Graph) The biparte network with users and resources.
        dict_attrs (pandas dataframe) The dataframe with the attributes and names
            to add.
        typen (int [0-1]) Type of nodes: 0-User nodes, 1-Resource nodes.

    Raise:
        TypeError: if the attributes are in both types of nodes.

    """
    user_nodes = bip_network.vs.select(typen=0)  # Get user nodes
    rscs_nodes = bip_network.vs.select(typen=1)  # Get resource nodes

    # Attributes nodes
    # Get attrs names except the last
    attrs = list(df_attrs_nodes.columns[:-1])
    for id_attr in attrs:  # For each attribute name
        user_nodes[id_attr] = list(df_attrs_nodes[id_attr])
        if not check_all_ok(rscs_nodes, id_attr):
            raise TypeError("Problmes")

    # Resource nodes
    attrs = list(df_attrs_res.columns[:-1])  # Get attrs names except the last
    for id_attr in attrs:  # For each attribute name
        rscs_nodes[id_attr] = list(df_attrs_res[id_attr])
        if not check_all_ok(user_nodes, id_attr):
            raise TypeError("Problmes")


def build_network_model(data, usr_id_name, res_id_name, df_user_attrs,
                        df_recs_attrs, file_path=None):
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
        list_of_edges.append((int(usr_idx), int(rsr_idx)))  # Tuple of edges
    bi_network.add_edges_from(list_of_edges)  # Build Network with edges

    # Change networkX object to iGraph object
    bi_network = Graph.from_networkx(bi_network)
    bi_network.vs['name'] = bi_network.vs["_nx_name"]  # Clean name column
    del bi_network.vs["_nx_name"]  # Remove uncleaned name column

    print(bi_network.summary())
    print(bi_network.vs())

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

    # Add attributes
    add_attrs(bi_network, df_user_attrs, df_recs_attrs)

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

    return g
# END bipartite_projection
