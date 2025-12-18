"""This file contains all function in the community detection phase."""
import numpy as np

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
    seq_nodes = network.vs.select(commty=id_commty)  # Get all nodes.

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

    for user_node in commty_graph.vs:  # Loop over all users in the commty
        all_resources += user_node["rsrcs"]

    # Remove duplicates
    all_resources = list(set(all_resources))
    return all_resources


def sub_community_detection(user_network, density_t=0.5, refinement=None):
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

    n_commts = len(set(user_network.vs["commty"]))  # Get all previous commts
    commty_counter = 0  # A counter to assign an ID to each commty detected

    # Dictionary to store all commts detected. An example:
    # {id_commty: [subgraph, resources list]}
    dict_commts = {}

    for id_commty in range(n_commts):  # Loop over id of previous commts
        # Get the Graph object of the community
        graph_commty = get_community_graph(user_network, id_commty)

        # Compute the density values of the commty
        if graph_commty.density() < density_t:  # If the network is sparse

            # Execute Louvain algorithm
            new_partition = graph_commty.community_multilevel(
                weights=graph_commty.es["weight"])

            for sub_commty in new_partition.subgraphs():  # Loop over new partition
                # Get all resources that are accessed by the commty
                all_rescs_commty = get_all_resources_in_commty(sub_commty)
                id_commty_str = str(commty_counter)  # Convert the id to str
                # Add the new community to the dict
                if refinement != None:
                    dict_commts[id_commty_str +
                                str(refinement)] = [sub_commty, all_rescs_commty]
                else:
                    dict_commts[id_commty_str] = [sub_commty, all_rescs_commty]
                # Add new ID commty to the user in user network
                add_id_comm_to_nodes(user_network, sub_commty, id_commty_str)
                commty_counter += 1
        else:  # If the network is dense
            # Get all resources that are accessed by the commty
            all_rescs_commty = get_all_resources_in_commty(graph_commty)
            id_commty_str = str(commty_counter)  # Convert the id to str
            # Add the new community to the dict
            if refinement != None:
                dict_commts[id_commty_str +
                            str(refinement)] = [graph_commty, all_rescs_commty]
            else:
                dict_commts[id_commty_str] = [graph_commty, all_rescs_commty]
            # Add new ID commty to the user in user network
            add_id_comm_to_nodes(user_network, graph_commty, id_commty_str)
            commty_counter += 1

    return dict_commts


def add_type_commty(network, id_commty, id_type):
    """Add the type of the community (0: Sparse, 1: Medium, 2: Concentrated) to
    each node.

    Parameters
    ----------
    network: Graph (igraph)
        Original network.
    id_commty: int
        The id of the community to add its type.
    id_type: int
        The type value of the community.

    """
    # Select all nodes in the id_commty
    commty_nodes = network.vs.select(commty=id_commty)

    for user in commty_nodes:
        user["tpcommty"] = id_type


def add_type_commts(network, dict_commts, s_threshold, m_threshold):
    """ Add the id of the type of the commty.

    """
    s_commts = []
    m_commts = []
    c_commts = []  # lists to return
    network.vs["tpcommty"] = -1

    for commty in dict_commts.items():
        # print(commty)
        if len(commty[1][1]) > s_threshold:  # Sparse commts
            s_commts.append((commty[0], [commty[1][0], commty[1][1], 0]))
            add_type_commty(network, commty[0], 0)
        elif len(commty[1][1]) > m_threshold:  # Medium commts
            m_commts.append((commty[0], [commty[1][0], commty[1][1], 1]))
            add_type_commty(network, commty[0], 1)
        else:
            c_commts.append((commty[0], [commty[1][0], commty[1][1], 2]))
            add_type_commty(network, commty[0], 2)

    print("|C|:", len(s_commts)+len(m_commts)+len(c_commts), "==",
          len(dict_commts))
    print("Sparse Comms:", len(s_commts))
    print("Medium Comms:", len(m_commts))
    print("Concentrate Comms:", len(c_commts))

    return s_commts, m_commts, c_commts


def calculate_membership_vector(user_network):
    """
    Calcula el vector de pertenencia a la comunidad para cada nodo y 
    identifica aquellos que tienen aristas con otras comunidades (outsiders).

    Parameters
    ----------
    user_network: Graph (igraph)
        Red de usuarios con el atributo "commty" (ID de comunidad).
    """
    # 1. Obtener el número total de comunidades (basado en los IDs de commty)
    # El atributo 'commty' en los nodos de la red final (después de sub_community_detection)
    # es de tipo string, ej: '0', '1', '2'.
    community_ids = [int(c) for c in set(user_network.vs["commty"])]
    n_communities = len(community_ids)
    
    # Mapear los IDs de comunidad (string) a un índice (int) para el vector.
    # Esto es crucial si los IDs no son secuenciales (0, 1, 2, ...) 
    # aunque con tu código, deberían serlo.
    comm_to_index = {str(comm_id): index for index, comm_id in enumerate(sorted(community_ids))}

    # 2. Inicializar atributos
    # 'membership_vector': Vector de pertenencia
    # 'is_outsider': Booleano (True si tiene conexiones externas)
    user_network.vs["membership_vector"] = [None] * user_network.vcount()
    user_network.vs["is_outsider"] = [False] * user_network.vcount()

    # 3. Iterar sobre cada nodo
    for i, v in enumerate(user_network.vs):
        # La comunidad a la que pertenece el nodo 'v'
        v_commty_id = v["commty"] 
        
        # Obtener los vecinos del nodo 'v'
        neighbor_indices = user_network.neighbors(v.index)
        
        # Obtener el total de aristas del nodo
        total_degree = len(neighbor_indices) 
        
        # Si el nodo no tiene aristas, el vector es de ceros.
        if total_degree == 0:
            v["membership_vector"] = [0.0] * n_communities
            v["is_outsider"] = False
            continue

        # Inicializar el vector de conteo para las aristas por comunidad
        edge_counts = np.zeros(n_communities)
        
        # Bandera para identificar outsiders
        has_external_edges = False

        # 4. Contar aristas por comunidad
        for neighbor_index in neighbor_indices:
            neighbor = user_network.vs[neighbor_index]
            neighbor_commty_id = neighbor["commty"]
            
            # Verificar si la arista cruza la comunidad
            if neighbor_commty_id != v_commty_id:
                has_external_edges = True
            
            # Determinar el índice de la comunidad del vecino para el vector
            # Esto usa el mapeo que hicimos antes para asegurarnos de que el orden
            # sea correcto en el vector.
            if neighbor_commty_id in comm_to_index:
                index_in_vector = comm_to_index[neighbor_commty_id]
                edge_counts[index_in_vector] += 1
            # Si un vecino tuviera un ID de comunidad no detectado, se ignoraría, 
            # lo cual no debería pasar con el código proporcionado.

        # 5. Calcular el vector de pertenencia
        # Normalizar: dividir el conteo de aristas por comunidad entre el grado total
        membership_vector = (edge_counts / total_degree).tolist()

        # 6. Asignar los atributos al nodo
        v["membership_vector"] = membership_vector
        v["is_outsider"] = has_external_edges

    print(f"Número total de comunidades: {n_communities}")
    # Contar y reportar los outsiders
    outsider_count = sum(user_network.vs["is_outsider"])
    print(f"Vértices identificados como Outsiders (con aristas a otra comunidad): {outsider_count}")
    print(f"El atributo 'is_outsider' y 'membership_vector' se han añadido a los nodos.")


def get_membership_vector_statistics(user_network):
    """
    Calcula y muestra estadísticas descriptivas de los vectores de pertenencia.

    Parameters
    ----------
    user_network: Graph (igraph)
        Red de usuarios con el atributo "membership_vector" en los nodos.

    Returns
    -------
    statistics: dict
        Diccionario con las estadísticas calculadas.
    """
    print("\n--- Estadísticas del Vector de Pertenencia ---")
    
    # 1. Recopilar todos los vectores de pertenencia
    membership_vectors = user_network.vs["membership_vector"]
    
    # Filtrar los valores nulos si los hay (aunque la función anterior 
    # debería asegurar que todos los nodos tengan un vector)
    valid_vectors = [v for v in membership_vectors if v is not None]
    
    if not valid_vectors:
        print("Advertencia: No se encontraron vectores de pertenencia válidos.")
        return {}

    # El número de comunidades es la longitud del vector.
    n_communities = len(valid_vectors[0])
    print(f"Número de Comunidades (longitud del vector): {n_communities}")
    
    # Convertir a un arreglo numpy para facilitar las operaciones por columna (comunidad)
    vectors_matrix = np.array(valid_vectors)
    
    statistics = {}

    # --- A. Estadísticas Globales por Distribución (Promedio de las 
    #       fracciones de pertenencia a una comunidad) ---
    
    # Esto calcula las estadísticas sobre TODOS los valores dentro de TODOS 
    # los vectores. Nos da una idea general de cuán 'difusa' o 'concentrada' 
    # es la pertenencia en la red.
    
    all_values = vectors_matrix.flatten()

    statistics['global'] = {
        'total_nodes': len(user_network.vs),
        'mean_membership_value': np.mean(all_values),
        'min_membership_value': np.min(all_values),
        'max_membership_value': np.max(all_values),
        'std_dev_membership_value': np.std(all_values)
    }

    print("\n### 1. Estadísticas Globales (Sobre todos los valores en la red) ###")
    print(f"  > Promedio de los valores de pertenencia: {statistics['global']['mean_membership_value']:.4f}")
    print(f"  > Desviación Estándar de los valores: {statistics['global']['std_dev_membership_value']:.4f}")
    print(f"  > Mínimo: {statistics['global']['min_membership_value']:.4f} \t Máximo: {statistics['global']['max_membership_value']:.4f}")
    
    # --- B. Estadísticas Detalladas por Comunidad ---

    statistics['by_community'] = {}
    print("\n### 2. Estadísticas Detalladas por Comunidad (Columna del Vector) ###")
    
    # Recorrer cada columna (cada comunidad)
    for comm_index in range(n_communities):
        # Seleccionar la columna de la matriz que corresponde a la comunidad 'comm_index'
        comm_values = vectors_matrix[:, comm_index]
        
        # El promedio aquí representa la 'atractividad' promedio de esta comunidad 
        # a todos los nodos de la red.
        mean_val = np.mean(comm_values) 
        
        # El máximo siempre será 1.0 (para el nodo que pertenece a esa comunidad y 
        # todas sus aristas son internas) o un poco menos.
        max_val = np.max(comm_values)
        
        statistics['by_community'][f'Comm_{comm_index}'] = {
            'mean': mean_val,
            'max': max_val,
            'std': np.std(comm_values)
        }
        
        print(f"  > Comunidad {comm_index}:")
        print(f"    - Promedio de Pertenencia (Atractividad): {mean_val:.4f}")
        print(f"    - Desviación Estándar: {np.std(comm_values):.4f}")
        print(f"    - Máximo: {max_val:.4f}")
    
    # --- C. Estadísticas de Outsiders ---
    outsider_count = sum(user_network.vs["is_outsider"])
    total_nodes = len(user_network.vs)
    
    statistics['outsiders'] = {
        'count': outsider_count,
        'fraction': outsider_count / total_nodes if total_nodes > 0 else 0
    }
    
    print("\n### 3. Estadísticas de Vértices Exteriores (Outsiders) ###")
    print(f"  > Vértices Exteriores: {outsider_count} de {total_nodes} nodos.")
    print(f"  > Fracción de Outsiders: {statistics['outsiders']['fraction']:.2%}")
    
    return statistics