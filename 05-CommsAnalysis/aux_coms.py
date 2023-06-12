"""This file contains all function in the community detection phase."""


def add_id_comm_to_nodes(user_network, commty_graph, id__comm):
    """Add the id community attribute to the set of users.

    Parameters
    ----------
    user_network: Graph (igraph)
        User networ.
    commty_graph: list
        Users list.
    id_comm: int
        ID of community

    """
    for user in commty_graph.vs["name"]:
        user_node = user_network.vs.find(name=user)
        user_node["commty"] = id__comm

def get_community_graph(network, id_commty):
    """Get the igraph.Graph object of a community.

    The ID community attribute of the network must to be named as *commty*

    Parameters
    ----------
    network: Graph (igraph)
        A network.
    id_commty: int
        ID of the community to get.

    Returns
    -------
    seq_nodes.subgraph(): Graph (igraph)
        The igraph.Graph object of community with the `id_commty`.
    """
    seq_nodes = network.vs.select(commty=id_commty) # Get all nodes.
    
    return seq_nodes.subgraph()

def get_all_resources_in_commty(commty_graph):
    """Get all resources that are accessed by all user in the `commty_graph`.

    Parameters
    ----------
    commty_graph: Graph (igraph)
        A community in igraph Graph object.

    Returns
    -------
    all_resources: list
        The list with all resources that are accessed by all user in the commty.
    """
    # Store all resources, then convert to set to remove duplicates
    all_resources = [] 
    
    for user_node in commty_graph.vs: # Loop over all users in the commty
        all_resources += user_node["rsrcs"]
    
    # Remove duplicates
    all_resources = list(set(all_resources))
    return all_resources


def sub_community_detection(user_network, prev_partition, density_t=0.5):
    """Get all communities including sub-community detection.

    Parameters
    ----------
    user_network: Graph (igraph)
        User network.
    prev_partition: VertexClustering (igraph)
        First communities partition. 
    density_t: flaot
        Density threshold to execute sub-communities. The Louvain algorithm
        is executed one more time in communities with a density value  < 
        `density_t`.

    Returns
    -------
    dict_total_coms: dict
        Dictionary with all communities. The key is the ID of the community.
        The value is a list of two elements: (1) Community graph (igraph object)
        and (2) a list with all the resources that the users of the community
        access.

    """    
    # Create a copy of the user_network
    # copy_user_network = user_network
    # Commts = Communities
    # Commty = Community

    n_commts = len(set(user_network.vs["commty"])) # Get all previous commts
    commty_counter = 0 # A counter to assign an ID to each commty detected

    # Dictionary to store all commts detected. An example: 
    # {id_commty: [subgraph, resources list]}
    dict_commts = {}

    for id_commty in range(n_commts): # Loop over id of previous commts 
        # Get the Graph object of the community
        graph_commty = get_community_graph(user_network, id_commty)

        # Compute the density values of the commty
        if graph_commty.density() < density_t: # If the network is sparse

            # Execute Louvain algorithm
            new_partition = graph_commty.community_multilevel(
                weights = graph_commty.es["weight"] )
            
            for sub_commty in new_partition.subgraphs(): # Loop over new partition
                # Get all resources that are accessed by the commty
                all_rescs_commty = get_all_resources_in_commty(sub_commty) 
                id_commty_str = str(commty_counter) # Convert the id to str
                # Add the new community to the dict
                dict_commts[id_commty_str] = [sub_commty, all_rescs_commty]
                # Add new ID commty to the user in user network
                add_id_comm_to_nodes(user_network, sub_commty, id_commty_str)
                commty_counter += 1
        else: # If the network is dense
            # Get all resources that are accessed by the commty
            all_rescs_commty = get_all_resources_in_commty(graph_commty)
            id_commty_str = str(commty_counter) # Convert the id to str
            # Add the new community to the dict
            dict_commts[id_commty_str] = [graph_commty, all_rescs_commty]
            # Add new ID commty to the user in user network
            add_id_comm_to_nodes(user_network, graph_commty, id_commty_str)
            commty_counter += 1

    return dict_commts






