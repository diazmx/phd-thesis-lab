import networkx as nx

def get_user_res(dataf, attr_list, user_):
    """Generate a unique identifier to a set of attributes."""
    dict_temp = {}
    for i, j in enumerate(dataf[attr_list].drop_duplicates().values):
        if user_:
            idx = str(i) + "101"
        else:
            idx = str(i) + "202"
        dict_temp[int(idx)] = list(j)
    return dict_temp

def add_col(dataf, dict_, attr_list, name_col):
    """Add new ID column to a Dataframe."""
    list_users = []  # List to save users
    key_list = list(dict_.keys())
    val_list = list(dict_.values())

    for item in dataf[attr_list].values:
        if list(item) in val_list:
            idx = val_list.index(list(item))
            list_users.append(key_list[idx])
        else:
            list_users.append(11111111)

    dataf[name_col] = list_users
    return dataf

def edge_weight(node_u, node_v):
    """Return the edge weight between two nodes."""
    neig_u = set(node_u.neighbors())
    neig_v = set(node_v.neighbors())
    inter_neig_uv = neig_u.intersection(neig_v)

    return (len(inter_neig_uv) / len(neig_u)) * (len(inter_neig_uv) / len(neig_v))

def graph_projectionB(bipartite_graph):
    G = nx.Graph()        
    for r_node in bipartite_graph.vs.select(typen=1): # Every resource
        neighborhood = r_node.neighbors()
        for u_node in neighborhood:
            for v_node in neighborhood:
                if u_node["name"] != v_node["name"]:
                    if G.has_edge(u_node["name"], v_node["name"]):
                        pass
                    else:
                        G.add_edge(u_node["name"], v_node["name"], weight=edge_weight(u_node, v_node) )                                 
    return G

def graph_projectionC(bipartite_graph):
    G = nx.Graph()        
    for r_node in bipartite_graph.vs.select(typen=1): # Every resource
        neighborhood = r_node.neighbors()
        wei_local = 1 / len(neighborhood)
        for i in range(len(neighborhood)): # Load every list
            for j in range(i+1, len(neighborhood)):
                if G.has_edge(neighborhood[i]["name"], neighborhood[j]["name"]):
                    G[neighborhood[i]["name"]][neighborhood[j]["name"]]['weight'] += wei_local
                else:
                    G.add_edge(neighborhood[i]["name"], neighborhood[j]["name"], weight=wei_local)                            
    return G

def partition_quality(G, partition):
    """Returns the coverage and performance of a partition of G.

    The *coverage* of a partition is the ratio of the number of
    intra-community edges to the total number of edges in the graph.

    The *performance* of a partition is the number of
    intra-community edges plus inter-community non-edges divided by the total
    number of potential edges.

    This algorithm has complexity $O(C^2 + L)$ where C is the number of communities and L is the number of links.

    Parameters
    ----------
    G : igraph graph

    partition : sequence
        Partition of the nodes of `G`, represented as a sequence of
        sets of nodes (blocks). Each block of the partition represents a
        community.

    Returns
    -------
    (float, float)
        The (coverage, performance) tuple of the partition, as defined above.

    Raises
    ------
    NetworkXError
        If `partition` is not a valid partition of the nodes of `G`.

    Notes
    -----
    If `G` is a multigraph;
        - for coverage, the multiplicity of edges is counted
        - for performance, the result is -1 (total number of possible edges is not defined)

    References
    ----------
    .. [1] Santo Fortunato.
           "Community Detection in Graphs".
           *Physical Reports*, Volume 486, Issue 3--5 pp. 75--174
           <https://arxiv.org/abs/0906.0612>
    """

    node_community = {}
    for i, community in enumerate(partition.membership):
        #for node in community:
        node_community[i] = community

    # Iterate over the communities, quadratic, to calculate 
    # `possible_inter_community_edges`
    possible_inter_community_edges = sum(
        p1 * p2 for p1, p2 in combinations(partition.sizes(), 2)
    )
    # Compute the number of edges in the complete graph -- `n` nodes,
    # directed or undirected, depending on `G`
    n = G.vcount()
    total_pairs = n * (n - 1)
    total_pairs //= 2

    intra_community_edges = 0
    inter_community_non_edges = possible_inter_community_edges

    # Iterate over the links to count `intra_community_edges` and `inter_community_non_edges`
    for e in G.es():
        if node_community[e.tuple[0]] == node_community[e.tuple[1]]:
            intra_community_edges += 1
        else:
            inter_community_non_edges -= 1

    coverage = intra_community_edges / G.ecount()
    performance = (intra_community_edges + inter_community_non_edges) / total_pairs

    return coverage, performance

def num_recursos(user_sets, grafo_bip):
    all_recursos = set()
    user_sets = [int(float(i)) for i in user_sets]
    for user in user_sets:
        #user_node = grafo_bip.vs.find(name=int(float(user)))
        #print(user)
        try:
            user_node = grafo_bip.vs.find(name=user)
        except:
            continue
        vecinos_recurso = user_node.neighbors()
        vecinos_recurso = [nodo["name"] for nodo in vecinos_recurso]
        vecinos_recurso = set(vecinos_recurso)    
        all_recursos = all_recursos.union(vecinos_recurso)
    return list(all_recursos)            

def numero_registros2(sub_com, dataf):
    """Retorna los registros en una comunidad."""
    dict_resc = {}
    for node in sub_com.vs():
        recursos = list(dataf[dataf.USRID == float(node["label"])]["RESID"])
        for i in recursos:
            if i in dict_resc.keys():
                dict_resc[i] += 1
            else:
                dict_resc[i] = 1
    
    # Remove low values
    dict_temp = {}
    for item in dict_resc.items():
        if item[1] > 2:
            dict_temp[item[0]] = item[1]
    dict_temp = dict(sorted(dict_temp.items(), key=lambda item: item[1], reverse=True))

    if len(dict_temp) < 1:
        return  len(list(dict_temp.keys()))
    return len(list(dict_temp.keys()))

def numero_registros3(sub_com, dataf):
    """Retorna los registros en una comunidad."""
    dict_resc = {}
    for node in sub_com.vs():
        recursos = list(dataf[dataf.USRID == float(node["label"])]["RESID"])
        for i in recursos:
            if i in dict_resc.keys():
                dict_resc[i] += 1
            else:
                dict_resc[i] = 1
    
    # Remove low values
    dict_temp = {}
    for item in dict_resc.items():
        if item[1] > 2:
            dict_temp[item[0]] = item[1]
    dict_temp = dict(sorted(dict_temp.items(), key=lambda item: item[1], reverse=True))

    if len(dict_temp) < 1:
        return  dict_temp
    # return len(list(dict_temp.keys()))
    return dict_temp


def numero_registros4(sub_com, dataf):
    """Retorna los registros en una comunidad."""
    dict_resc = {}
    for node in sub_com.vs():
        recursos = list(dataf[dataf.USRID == int(node["label"])]["RESID"])
        for i in recursos:
            if i in dict_resc.keys():
                dict_resc[i] += 1
            else:
                dict_resc[i] = 1
    
    # Remove low values
    dict_temp = {}
    for item in dict_resc.items():
        if item[1] > 2:
            dict_temp[item[0]] = item[1]
    dict_temp = dict(sorted(dict_temp.items(), key=lambda item: item[1], reverse=True))

    if len(dict_temp) < 1:
        return  dict_temp
    # return len(list(dict_temp.keys()))
    return dict_temp

def get_users_from_resource_comms(resource_id, community, data):
    """Retorna los usuarios de la comunidad que acceden al conjunto de recursos"""
    users_to_ret = []
    for res in resource_id:
        all_user_in_community = community.vs()["label"] # Extraer usuarios de comunidad
        ## all_user_in_community = [int(item) for item in all_user_in_community] # Convertimos a int
        all_user_in_community = set(all_user_in_community) # Covnertimos a conjunto
        solicitudes_en_data = data[data["RESID"]==res]["USRID"].to_list() # Todos los usuarios que acceden al recurso
        solicitudes_en_data = [str(item)+".0" for item in solicitudes_en_data]
        solicitudes_en_data = set(solicitudes_en_data)
        # print(all_user_in_community)
        # print(solicitudes_en_data)
        users_to_ret = users_to_ret + list(all_user_in_community.intersection(solicitudes_en_data)) # Intersección
    users_to_ret = list(set(users_to_ret))
    return users_to_ret

def remove_equal_rules(rules):
    """ Remove equal rules X -> Y == Y -> X"""
    to_remove = []
    lrules = len(rules)
    for i in range(lrules):
        for j in range(i+1,lrules):
            if rules[i][0] == rules[j][1] and rules[i][1] == rules[j][0]:        
                to_remove.append(rules[i])

    #print(to_remove)
    return to_remove

def remove_equal_rules2(rules):
    """ Remove equal rules X -> Y == Y -> X"""
    to_remove = []
    #print(rules)
    for rule in rules:
        descri = []
        for ele in rule[0]:
            descri.append(ele)
        for ele in rule[1]:
            descri.append(ele)
        #print(descri)
        descri.sort()
        if descri in to_remove:
            continue
        else:            
            to_remove.append(descri)

    # Quedarse con la de mayor tamaño
    max_id = 0
    to_ret = None
    for i in to_remove:        
        if len(i) > max_id:            
            max_id = len(i)
            to_ret = i

    return [to_ret]

def remove_equal_rules3(rules):
    """ Remove equal rules X -> Y == Y -> X"""
    to_remove = []
    #print(rules)
    for rule in rules:
        descri = []
        for ele in rule[0]:
            descri.append(ele)
        for ele in rule[1]:
            descri.append(ele)
        #print(descri)
        descri.sort()
        if descri in to_remove:
            continue
        else:            
            to_remove.append(descri)

    # Quedarse con la de mayor tamaño
    max_id = 0
    to_ret = None
    for i in to_remove:        
        if len(i) < 3:            
            #max_id = len(i)
            to_ret = i

    return [to_ret]

def remove_equal_rules4(rules, recurso_atr, reglas_ant):
    """ Remove equal rules X -> Y == Y -> X"""
    to_remove = []
    #print(rules)
    for rule in rules:
        descri = []
        for ele in rule[0]:
            descri.append(ele)
        for ele in rule[1]:
            descri.append(ele)
        #print(descri)
        descri.sort()
        if descri in to_remove:
            continue
        else:            
            to_remove.append(descri)

    # Quedarse con la de mayor tamaño
    used_idx = []
    encontrado = False
    while not encontrado:        
        idx_rand = random.randint(0, len(to_remove)-1)
        if idx_rand in used_idx:
            continue
        else:
            used_idx.append(idx_rand)
            temp_to_ret = to_remove[idx_rand]
            temp_rule = recurso_atr + temp_to_ret
            if temp_rule in reglas_ant:
                continue
            else:
                to_ret = temp_to_ret

    return [to_ret]

# Function to check if a value appear in column
def get_attr_name(value, df_):
    """ Return name of the column of the value."""
    cols = df_.columns  
    if value == 10:
            # Buscar el atributo maximo
        max_val = 0
        attr_  = None
        for i in cols:        
            t = len(df_[df_[i]==10])            
            if t > max_val:
                max_val = t
                attr_ = i
        return attr_
    else:
        for i in range(len(cols)):        
            if len(df_[df_[cols[i]]==value]) > 0:
                return df_.columns[i]

def get_attr_val_in_users(users_id, data):
    """Retorna los atributo valor en comun en un conjunto de usuarios."""
    user_convert = [float(n) for n in users_id] # Convertimos a float
    attr_user_ = data[data["USRID"].isin(user_convert)].drop_duplicates() # Seleccionamos usuarios
    attr_user__ = attr_user_[user_attr+["USRID"]].values.tolist() # Seleccionamos attributos de los usuarios
    attr_user_ = attr_user_[user_attr+["USRID"]]
    # print(attr_user_)
    init_Sup = 0.6
    init_Conf = 0.9
    rules = []
    while len(rules) == 0:
        # print("Sup", init_Sup, "  Conf", init_Conf)
        _, rules = apriori(attr_user__, minSup=init_Sup, minConf=init_Conf) # Apply apriori
        init_Sup -= 0.1
        init_Conf -= 0.1
    # print(rules)
    rules = remove_equal_rules2(rules)
    #print("XXX",rules)
    #for r in rules:
    #    print(r)
    
    reglas_karimi = []
    for r in rules:      
        for t in r:                                    
            col = get_attr_name(t, attr_user_)                                                             
            reglas_karimi.append([col, t])

    return reglas_karimi

def get_attr_val_in_users_2(users_id, data):
    """Retorna los atributo valor en comun en un conjunto de usuarios."""
    user_convert = [float(n) for n in users_id] # Convertimos a float
    attr_user_ = data[data["USRID"].isin(user_convert)].drop_duplicates() # Seleccionamos usuarios
    attr_user__ = attr_user_[user_attr].values.tolist() # Seleccionamos attributos de los usuarios
    attr_user_ = attr_user_[user_attr+["USRID"]]
    # print(attr_user_)
    init_Sup = 0.6
    init_Conf = 0.9
    rules = []
    while len(rules) == 0:
        # print("Sup", init_Sup, "  Conf", init_Conf)
        _, rules = apriori(attr_user__, minSup=init_Sup, minConf=init_Conf) # Apply apriori
        init_Sup -= 0.1
        init_Conf -= 0.1
        if init_Sup < 0:
            return attr_user__
    # print(rules)
    rules = remove_equal_rules2(rules)
    #print("XXX",rules)
    #for r in rules:
    #    print(r)
    
    reglas_karimi = []
    for r in rules:      
        for t in r:                                    
            col = get_attr_name(t, attr_user_)                                                             
            reglas_karimi.append([col, t])

    return reglas_karimi

def get_attr_val_in_res(res_ids, data):
    """Retorna los atributo valor en comun en un conjunto de usuarios."""    
    user_convert = [float(n) for n in res_ids] # Convertimos a float
    attr_user_ = data[data["RESID"].isin(user_convert)].drop_duplicates() # Seleccionamos usuarios
    attr_user__ = attr_user_[rsrc_attr].values.tolist() # Seleccionamos attributos de los usuarios
    attr_user_ = attr_user_[rsrc_attr+["RESID"]]
    #print(attr_user_)
    # print(attr_user_)
    init_Sup = 0.6
    init_Conf = 0.9
    rules = []
    while len(rules) == 0:
        # print("Sup", init_Sup, "  Conf", init_Conf)
        _, rules = apriori(attr_user__, minSup=init_Sup, minConf=init_Conf) # Apply apriori
        init_Sup -= 0.1
        init_Conf -= 0.1
    # _, rules = apriori(attr_user__, minSup=0.6, minConf=0.9) # Apply apriori
    rules = remove_equal_rules2(rules)
    #for r in rules:
    #    print(r)
    
    reglas_karimi = []
    for r in rules:      
        for t in r:                                    
            col = get_attr_name(t, attr_user_)                                                             
            reglas_karimi.append([col, t])

    return reglas_karimi

def get_attr_val_in_res_2(res_ids, data):
    """Retorna los atributo valor en comun en un conjunto de usuarios."""    
    user_convert = [float(n) for n in res_ids] # Convertimos a float
    attr_user_ = data[data["RESID"].isin(user_convert)].drop_duplicates() # Seleccionamos usuarios
    attr_user__ = attr_user_[rsrc_attr].values.tolist() # Seleccionamos attributos de los usuarios
    attr_user_ = attr_user_[rsrc_attr+["RESID"]]
    #print(attr_user_)
    # print(attr_user_)
    init_Sup = 0.6
    init_Conf = 0.9
    rules = []
    while len(rules) == 0:
        # print("Sup", init_Sup, "  Conf", init_Conf)
        _, rules = apriori(attr_user__, minSup=init_Sup, minConf=init_Conf) # Apply apriori
        init_Sup -= 0.1
        init_Conf -= 0.1
        if init_Sup < 0:
            return attr_user__
    # _, rules = apriori(attr_user__, minSup=0.6, minConf=0.9) # Apply apriori
    rules = remove_equal_rules2(rules)
    #for r in rules:
    #    print(r)
    
    reglas_karimi = []
    for r in rules:      
        for t in r:                                    
            col = get_attr_name(t, attr_user_)                                                             
            reglas_karimi.append([col, t])

    return reglas_karimi


def podado_recursos(recursos, subgrafo, umbral):
    """Realiza el podado de los recursos que casi no acceden"""
    n_usuarios = subgrafo.vcount()
    to_ret = {}    
    for i in recursos:
        us_temp = recursos[i]
        if (us_temp / n_usuarios) > umbral:
            to_ret[i] = us_temp
    return to_ret

def recurso_significativo(recursos, subgrafo, umbral):
    """Retorna el recurso significativo. 'None' si no hay."""
    n_usuarios = subgrafo.vcount()
    to_ret = {}    
    for i in recursos:
        us_temp = recursos[i]
        if (us_temp / n_usuarios) >= umbral:
            return to_ret[i]
    return None

def extraer_recursos_similares(recursos, data_, umbral):
    """Retorna registros similares como nuevos registros"""

    for atr in rsrc_attr: # Se reccorre los atributos
        for r1 in recursos: # Se recorre los recursos
            for atr2 in rsrc_attr:
                if atr != atr2: # Se cambian los recursos
                    print()


def get_apriori_final(usuarios, recursos, data):
    """Retorna los atributo valor en comun en un conjunto de usuarios."""
    user_convert = [float(n) for n in usuarios] # Convertimos a float
    attr_user_ = data[data["USRID"].isin(user_convert)].drop_duplicates() # Seleccionamos usuarios
    res_convert = [float(n) for n in recursos] # Convertimos a float
    attr_user_ = attr_user_[attr_user_["RESID"].isin(res_convert)].drop_duplicates() # Seleccionamos recursos
    attr_user__ = attr_user_[user_attr+rsrc_attr].values.tolist() # Seleccionamos attributos de los usuarios
    attr_user_ = attr_user_[user_attr+rsrc_attr]
    # print(attr_user_)
    init_Sup = 0.5
    init_Conf = 0.8
    rules = []
    while len(rules) == 0:
        _, rules = apriori(attr_user__, minSup=init_Sup, minConf=init_Conf) # Apply apriori
        init_Sup -= 0.1
        init_Conf -= 0.1
    # print(rules)
    rules = remove_equal_rules2(rules)
    
    reglas_karimi = []
    for r in rules:      
        for t in r:                                    
            col = get_attr_name(t, attr_user_)                                                             
            reglas_karimi.append([col, t])

    return reglas_karimi

def agregar_diccionario(diccionario, recurso, tupla_datos):
    if recurso in diccionario:
        diccionario[recurso].append(tupla_datos)
    else:
        diccionario[recurso] = [tupla_datos]

def agregar_usuario_com(diccio, com_id, usuario):
    if com_id in diccio:
        if not usuario in diccio[com_id]:
            diccio[com_id].append(usuario)
    else:
        
        diccio[com_id] = [usuario]


# Function to add new cluster id to the nodes
def add_new_cluster_id(dict_comms, user_network):
    """
    Function to add sub-cluster id to the user network nodes.
    """
    temp_list = []
    for node in user_network.vs():
        node_name = node["name"] # Name of the node        
        for i, j in dict_comms.items(): # Looping in the dict
            #print(node_name, j[0].vs().find(name_eq=int(node_name)))
            try:
                is_in = j[0].vs().find(name_eq=int(node_name))
                temp_list.append(i) # Add id cluster
            except:
                continue        

    user_network.vs()["n_cluster"] = temp_list
    return user_network

def apriori_in_resources(data_):
    """Retorna regla apriori basada en los recursos.
    data_: dataframe
    """
    init_Sup = 0.6
    init_Conf = 0.9
    rules = []
    while len(rules) == 0: # AND NOT rules in list_rules
        # print("Sup", init_Sup, "  Conf", init_Conf)
        _, rules = apriori(data_.values.tolist(), minSup=init_Sup, minConf=init_Conf) # Apply apriori
        init_Sup -= 0.1
        init_Conf -= 0.1
    # _, rules = apriori(attr_user__, minSup=0.6, minConf=0.9) # Apply apriori
    rules = remove_equal_rules2(rules)
    #for r in rules:
    #    print(r)
    
    reglas_karimi = []
    for r in rules:      
        for t in r:                                    
            col = get_attr_name(t, data_)                                                             
            reglas_karimi.append([col, t])

    return reglas_karimi

def apriori_in_resources_2(data_):
    """Retorna regla apriori basada en los recursos.
    data_: dataframe
    """
    init_Sup = 0.5
    init_Conf = 0.7
    rules = []
    while len(rules) == 0: # AND NOT rules in list_rules
        # print("Sup", init_Sup, "  Conf", init_Conf)
        _, rules = apriori(data_.values.tolist(), minSup=init_Sup, minConf=init_Conf) # Apply apriori
        init_Sup -= 0.1
        init_Conf -= 0.1
    # _, rules = apriori(attr_user__, minSup=0.6, minConf=0.9) # Apply apriori
    print(data_)
    print(rules)
    rules = remove_equal_rules2(rules)
    #for r in rules:
    #    print(r)
    
    reglas_karimi = []
    for r in rules:      
        print(r)
        for t in r:                                                
            print(t)
            col = get_attr_name(t, data_)                                                             
            print([col, t])
            reglas_karimi.append([col, t])
            
#    print(reglas_karimi)
    return reglas_karimi

def apriori_in_resources_3(data_):
    """Retorna regla apriori basada en los recursos.
    data_: dataframe
    """
    init_Sup = 0.6
    init_Conf = 0.9
    rules = []
    while len(rules) == 0: # AND NOT rules in list_rules        
        _, rules = apriori(data_.values.tolist(), minSup=init_Sup, minConf=init_Conf) # Apply apriori
        init_Sup -= 0.1
        init_Conf -= 0.1
    #print(rules)
    rules = remove_equal_rules3(rules)
    #print(rules)
    #print()
    #for r in rules:
    #    print(r)
    
    reglas_karimi = []
    for r in rules:      
        for t in r:                                    
            col = get_attr_name(t, data_)                                                             
            reglas_karimi.append([col, t])

    return reglas_karimi

def apriori_in_resources_4(data_, recurso_atr, reglas_ante):
    """Retorna regla apriori basada en los recursos.
    data_: dataframe
    """
    init_Sup = 0.6
    init_Conf = 0.9
    rules = []
    while len(rules) == 0: # AND NOT rules in list_rules        
        _, rules = apriori(data_.values.tolist(), minSup=init_Sup, minConf=init_Conf) # Apply apriori
        init_Sup -= 0.1
        init_Conf -= 0.1
    #print(rules)
    #rules = remove_equal_rules4(rules, recurso_atr, reglas_ante)
    #print(rules)
    #print()
    #for r in rules:
    #    print(r)
    
    """ Remove equal rules X -> Y == Y -> X"""
    to_remove = []
    #print(rules)
    for rule in rules:
        descri = []
        for ele in rule[0]:
            descri.append(ele)
        for ele in rule[1]:
            descri.append(ele)
        #print(descri)
        descri.sort()
        if descri in to_remove:
            continue
        else:            
            to_remove.append(descri)
    
    used_idx = []

    # Mientras que no se hayan usado todos los indices
    while len(used_idx) < len(to_remove):                
        idx_rand = random.randint(0, len(to_remove)-1) # Número random
        if idx_rand in used_idx:
            continue
        else:            
            used_idx.append(idx_rand)
            selection_ = to_remove[idx_rand] # Regla random
            
            # Regla en formato atr-valor
            reglas_karimi = [] 
            for r in [selection_]:      
                for t in r:                                    
                    col = get_attr_name(t, data_)                                                             
                    reglas_karimi.append([col, t])
            # end

            regla_candidata = recurso_atr + reglas_karimi # Regla candidata
            
            # Si la regla se encuentra en el set de reglas.
            if regla_candidata in reglas_ante: 
                continue
            else:            
                return reglas_karimi
    return [["control", 10901]]


def podar_recursos_new(subcomunidad, grafo_bip, umbral):
    """Retorna los recursos nuevos ya podados."""    
    usuario_comunidad = subcomunidad.vs()["name"]
    all_recursos = {} # DIccionario con los recursos y su frecuencia
    usuario_comunidad = [int(float(i)) for i in usuario_comunidad]
    for user in usuario_comunidad:
        user_node = grafo_bip.vs.find(name=int(float(user)))
        vecinos_recurso = user_node.neighbors()
        vecinos_recurso = [nodo["name"] for nodo in vecinos_recurso]
        vecinos_recurso = list(set(vecinos_recurso))
        for vecino in vecinos_recurso:
            if vecino in all_recursos.keys():
                all_recursos[vecino] += 1
            else:
                all_recursos[vecino] = 1

    umbral_en_n = int(umbral * subcomunidad.vcount())
    
    nuevos_recursos = []
    for item in all_recursos:
        if all_recursos[item] >= umbral_en_n:
            nuevos_recursos.append(item)

    return nuevos_recursos

def get_recursos_significativos(subcomunidad, grafo_bip, umbral):
    """Obtiene una lista de recursos más significativos en la comunidad."""
    usuario_comunidad = subcomunidad.vs()["name"]
    all_recursos = {} # DIccionario con los recursos y su frecuencia
    usuario_comunidad = [int(float(i)) for i in usuario_comunidad]
    for user in usuario_comunidad:
        user_node = grafo_bip.vs.find(name=int(float(user)))
        vecinos_recurso = user_node.neighbors()
        vecinos_recurso = [nodo["name"] for nodo in vecinos_recurso]
        vecinos_recurso = list(set(vecinos_recurso))
        for vecino in vecinos_recurso:
            if vecino in all_recursos.keys():
                all_recursos[vecino] += 1
            else:
                all_recursos[vecino] = 1

    umbral_en_n = int(umbral * subcomunidad.vcount())
    
    nuevos_recursos = []
    for item in all_recursos:
        if all_recursos[item] >= umbral_en_n:
            nuevos_recursos.append(item)
    
    return nuevos_recursos

def get_recursos_significativos2(subcomunidad, grafo_bip, umbral):
    """Obtiene una lista de recursos más significativos en la comunidad."""
    usuario_comunidad = subcomunidad.vs()["name"]
    all_recursos = {} # DIccionario con los recursos y su frecuencia
    usuario_comunidad = [int(float(i)) for i in usuario_comunidad]
    for user in usuario_comunidad:
        user_node = grafo_bip.vs.find(name=int(float(user)))
        vecinos_recurso = user_node.neighbors()
        vecinos_recurso = [nodo["name"] for nodo in vecinos_recurso]
        vecinos_recurso = list(set(vecinos_recurso))
        for vecino in vecinos_recurso:
            if vecino in all_recursos.keys():
                all_recursos[vecino] += 1
            else:
                all_recursos[vecino] = 1

    umbral_en_n = int(umbral * subcomunidad.vcount())
    
    nuevos_recursos = []
    for item in all_recursos:
        nuevos_recursos.append(item)
    
    return nuevos_recursos

def get_comm_for_user(diccionario_com_user, usuario):
    """Retorna el id de la comunidad al que pertenece el usuario."""
    for item in diccionario_com_user:
        if str(usuario)+".0" in diccionario_com_user[item]: # Si el usuario está en la comunidad
            return item
    return 0

def extraer_reglas_comunidad(lista_reglas, id_comunidad):
    """Retornal una lista de reglas de la comunidad."""
    to_ret = []
    for r in lista_reglas:
        if r[0][1] == id_comunidad: # Si es una regla que cumple con la comunidad
            to_ret.append(r)
    return to_ret

def extraer_reglas_comunidad_list(lista_reglas, list_id_comunidad):
    """Retornal una lista de reglas de la comunidad."""
    to_ret = []
    for r in lista_reglas:
        if r[0][1] in list_id_comunidad: # Si es una regla que cumple con la comunidad
            to_ret.append(r)
    return to_ret

def convert_to_list(dictionario_):
    to_return = []
    for i in dictionario_:
        to_return.append(i[0])
    return to_return

def compute_wsc(policy):
    return sum([len(rule) for rule in policy])

def evaluate_weight(rule_a, rule_b, umbral):
    """Retorna la arista entre dos reglas."""
    rule_atr_a = {}
    for item in rule_a:        
        rule_atr_a[item[0]] = item[1]

    rule_atr_b = {}
    for item in rule_b:
        rule_atr_b[item[0]] = item[1]

    conta_temp = 0
    for attr in rule_atr_a:
        if attr != "id_com":
            if attr in rule_atr_b:
                if rule_atr_a[attr] == rule_atr_b[attr]:
                    conta_temp += 1

    if conta_temp >= umbral:
        return conta_temp
    else:
        return -1

def get_rule_id(set_rules, dict_wiith_idx):
    """Retorna el id de las reglas para verlo en el grafo."""
    list_idx_ret =[]
    key_list = list(dict_wiith_idx.keys())
    val_list = list(dict_wiith_idx.values())
    for i in set_rules:
        position = val_list.index(i)
        list_idx_ret.append(key_list[position])
    return list_idx_ret

def get_neighbors_rules(list_rule_idx, rule_graph, dict_wiith_idx):
    """Retorna el idx de las reglas vecinas."""
    list_id_vecinos = []
    for idx in list_rule_idx: # Por cada regla
        # Se busca su vecino en el grafo
        for i in rule_graph.neighbors(idx):
            list_id_vecinos.append(i)

    list_id_vecinos = list_id_vecinos + list_rule_idx
    list_to_ret = []
    #print(list_id_vecinos)
    for idx in list_id_vecinos:
        list_to_ret.append(dict_wiith_idx[idx])
    return list_to_ret

def calculate_k_i(node):
    sum_to_ret = 0
    for i in node.all_edges():
        sum_to_ret += i["weight"]
    return sum_to_ret

def calculate_k_i_in(node, comm, graph_):
    # Sacar los vecinos del nodo
    vecinos_node = node.neighbors()

    # Ver cuales están en la comunidad
    id_com_ = comm.vs()[0]["comid"] # Id de la comunidad (se busca en primer nodo)

    # Se hace la intersección
    inter_vertex = []
    for veci in vecinos_node:
        if veci["comid"] == id_com_:
            inter_vertex.append(veci)

    # Se busca su peso
    kiin = 0
    for veci in inter_vertex:
        id_edge = graph_.get_eid(node, veci, directed=False)
        kiin += graph_.es()[id_edge]["weight"]
    
    return kiin*2

def calculate_sum_tot(comm, graph_):
    sum_tot = 0
    for i in comm.vs():
        id_label = i["name"]
        
        node = graph_.vs.find(name_eq = id_label)
        #print(node.all_edges())
        #print([i["weight"] for i in node.all_edges()])
        sum_tot += sum([i["weight"] for i in node.all_edges()])
    return sum_tot

def modularity_evaluate(node, comm, graph_):
    """Return the modularity value adding the node in the comm."""    
    sum_tot = calculate_sum_tot(comm, graph_)
    #print(sum_tot)
    sum_in = sum(comm.es()["weight"])*2 
    #print(sum_in)
    k_i_in = calculate_k_i_in(node, comm, graph_)
    #print(k_i_in)
    k_i = calculate_k_i(node)
    #print(k_i)
    m = graph_.ecount()
    #print(m)
    part_a = ((sum_in + 2*k_i_in) / (2*m) ) - ((sum_tot+k_i)/(2*m)*(sum_tot+k_i)/(2*m))
    part_b = (sum_in/(2*m)) - ((sum_tot/(2*m))*(sum_tot/(2*m))) - ((k_i/(2*m))*(k_i/(2*m)))
    return part_a - part_b

def add_new_user_node(user, resource, graph_, data):
    """Add new vertex in the graph based on share resource."""
    # Se extraen los usuarios que tienen el mismo recurso de acceso
    users_same_resource = data[data["RESID"]==resource].drop_duplicates()["USRID"].to_list()
    graph_.add_vertex(graph_.vcount()) # Se agrega el vértice en el grafo
    graph_.vs()[graph_.vcount()-1]["name"] = user # Agregar atributo lable
    user_obj = graph_.vs()[graph_.vcount()-1]
    #print(users_same_resource)
    for usr in users_same_resource:
        x = graph_.vs.find(name_eq=str(usr)+".0")    
        graph_.add_edges([(x, user_obj)])
        graph_.es()[-1]["weight"] = 1
    #print("Done!")

def add_new_user_node_2(user, resource, graph_, data):
    """Add new vertex in the graph based on share resource."""
    # Se extraen los usuarios que tienen el mismo recurso de acceso
    users_same_resource = set(data[data["RESID"]==resource].drop_duplicates()["USRID"].to_list())
    #print(users_same_resource)
    usuaris_grafo = set([int(float(i)) for i in graph_.vs()["name"]])
    users_same_resource = list(users_same_resource.intersection(usuaris_grafo))
    #print(users_same_resource)
    graph_.add_vertex(graph_.vcount()) # Se agrega el vértice en el grafo
    graph_.vs()[graph_.vcount()-1]["name"] = user # Agregar atributo lable
    user_obj = graph_.vs()[graph_.vcount()-1]
    #print(users_same_resource)
    for usr in users_same_resource:
        x = graph_.vs.find(name_eq=usr)    
        graph_.add_edges([(x, user_obj)])
        graph_.es()[-1]["weight"] = 1

def get_comunidades_vecinas(nodo, graph_):
    """Return ids communities neighboors."""
    vecinos_nodo = nodo.neighbors()
    lista_to_ret = []
    for vecino in vecinos_nodo:
        if not vecino["comid"] in lista_to_ret:
            lista_to_ret.append(vecino["comid"])
    return lista_to_ret

def obtener_reglas_comundiad(id_com, reglas):
    """Retorna las reglas con el identificador de comunidad"""
    list_to_ret = []
    for i in reglas:
        if i[0][1] == id_com:
            list_to_ret.append(i)
    return list_to_ret
