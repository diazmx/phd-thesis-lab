
def obtener_reglas_comundiad(id_com, reglas):
    """Retorna las reglas con el identificador de comunidad"""
    list_to_ret = []
    for i in reglas:
        if i[0][1] == id_com:
            list_to_ret.append(i)
    return list_to_ret


def get_rule_id(set_rules, dict_wiith_idx):
    """Retorna el id de las reglas para verlo en el grafo."""
    list_idx_ret = []
    key_list = list(dict_wiith_idx.keys())
    val_list = list(dict_wiith_idx.values())
    for i in set_rules:
        position = val_list.index(i)
        list_idx_ret.append(key_list[position])
    return list_idx_ret


def get_neighbors_rules(list_rule_idx, rule_graph, dict_wiith_idx):
    """Retorna el idx de las reglas vecinas."""
    list_id_vecinos = []
    for idx in list_rule_idx:  # Por cada regla
        # Se busca su vecino en el grafo
        for i in rule_graph.neighbors(idx):
            list_id_vecinos.append(i)

    list_id_vecinos = list_id_vecinos + list_rule_idx
    list_to_ret = []
    # print(list_id_vecinos)
    for idx in list_id_vecinos:
        list_to_ret.append(dict_wiith_idx[idx])
    return list_to_ret


def add_new_user_node_2(user, resource, graph_, data):
    """Add new vertex in the graph based on share resource."""
    # Se extraen los usuarios que tienen el mismo recurso de acceso
    users_same_resource = set(
        data[data["RID"] == resource].drop_duplicates()["UID"].to_list())
    # print(users_same_resource)
    usuaris_grafo = set([i for i in graph_.vs()["name"]])
    users_same_resource = list(users_same_resource.intersection(usuaris_grafo))
    # print(users_same_resource)
    graph_.add_vertex(graph_.vcount())  # Se agrega el vértice en el grafo
    graph_.vs()[graph_.vcount()-1]["name"] = user  # Agregar atributo lable
    user_obj = graph_.vs()[graph_.vcount()-1]
    # print(users_same_resource)
    for usr in users_same_resource:
        x = graph_.vs.find(name_eq=usr)
        graph_.add_edges([(x, user_obj)])
        graph_.es()[-1]["weight"] = 1


def get_comunidades_vecinas(nodo, graph_):
    """Return ids communities neighboors."""
    vecinos_nodo = nodo.neighbors()
    lista_to_ret = []
    for vecino in vecinos_nodo:
        if not vecino["commty"] in lista_to_ret:
            lista_to_ret.append(vecino["commty"])
    return lista_to_ret


def get_FN_logs(data_, user_network, list_rules, rule_network, rules_dict):
    """TRAIN DATA."""
    false_neg = []
    for i, row in data_.iterrows():
        user_id = row["UID"]

        user_node = user_network.vs.find(name=user_id)
        user_commty = user_node["commty"]

        list_coms_user = obtener_reglas_comundiad(user_commty, list_rules)
        list_rules_idx = get_rule_id(list_coms_user, rules_dict)
        list_coms_user = get_neighbors_rules(
            list_rules_idx, rule_network, rules_dict)

        # Evaluación
        denies_count = 0
        for rule in list_coms_user:
            res = True
            for idx_r, attr_val in enumerate(rule[1]):
                if row[attr_val[0]] != attr_val[1]:
                    res = False
                    break
            if res == False:
                denies_count += 1

        if denies_count == len(list_coms_user):
            false_neg.append(row)

    return false_neg


def calculate_sum_tot(comm, graph_):
    sum_tot = 0
    for i in comm.vs():
        id_label = i["name"]

        node = graph_.vs.find(name_eq=id_label)
        # print(node.all_edges())
        # print([i["weight"] for i in node.all_edges()])
        sum_tot += sum([i["weight"] for i in node.all_edges()])
    return sum_tot


def calculate_k_i_in(node, comm, graph_):
    # Sacar los vecinos del nodo
    vecinos_node = node.neighbors()

    # Ver cuales están en la comunidad
    # Id de la comunidad (se busca en primer nodo)
    id_com_ = comm.vs()[0]["commty"]

    # Se hace la intersección
    inter_vertex = []
    for veci in vecinos_node:
        if veci["commty"] == id_com_:
            inter_vertex.append(veci)

    # Se busca su peso
    kiin = 0
    for veci in inter_vertex:
        id_edge = graph_.get_eid(node, veci, directed=False)
        kiin += graph_.es()[id_edge]["weight"]

    return kiin*2


def calculate_k_i(node):
    sum_to_ret = 0
    for i in node.all_edges():
        sum_to_ret += i["weight"]
    return sum_to_ret


def modularity_evaluate(node, comm, graph_):
    """Return the modularity value adding the node in the comm."""
    sum_tot = calculate_sum_tot(comm, graph_)
    # print(sum_tot)
    sum_in = sum(comm.es()["weight"])*2
    # print(sum_in)
    k_i_in = calculate_k_i_in(node, comm, graph_)
    # print(k_i_in)
    k_i = calculate_k_i(node)
    # print(k_i)
    m = graph_.ecount()
    # print(m)
    part_a = ((sum_in + 2*k_i_in) / (2*m)) - \
        ((sum_tot+k_i)/(2*m)*(sum_tot+k_i)/(2*m))
    part_b = (sum_in/(2*m)) - ((sum_tot/(2*m))*(sum_tot/(2*m))) - \
        ((k_i/(2*m))*(k_i/(2*m)))
    return part_a - part_b


def get_FP_logs(data_, user_network, list_rules, rule_network, rules_dict):
    """TRAIN DATA."""
    false_positives = []
    rules_to_fix = []
    for i, row in data_.iterrows():
        user_id = row["UID"]
        res_id = row["RID"]

        # User exist in User network
        if not user_id in user_network.vs["name"]:

            # Identificación de la comunidad
            copy_g = user_network.copy()
            # Se agrega el nodo a la red
            add_new_user_node_2(user_id, res_id, copy_g, data_)
            node_user = copy_g.vs[-1]  # Node de usuario

            # Identificar a las comunidades vecinas
            coms_vecinas = get_comunidades_vecinas(node_user, copy_g)

            # Probar la modularidad máxima
            max_mod = 0
            user_commty = None
            for id_com in coms_vecinas:
                comunidad = copy_g.vs.select(commty=id_com)
                comunidad = copy_g.subgraph(comunidad)
                temp_mod = modularity_evaluate(node_user, comunidad, copy_g)
                if temp_mod > max_mod:
                    max_mod = temp_mod
                    user_commty = id_com
        else:
            user_node = user_network.vs.find(name=user_id)
            user_commty = user_node["commty"]

        list_coms_user = obtener_reglas_comundiad(user_commty, list_rules)
        list_rules_idx = get_rule_id(list_coms_user, rules_dict)
        list_coms_user = get_neighbors_rules(
            list_rules_idx, rule_network, rules_dict)

        denies_count = 0
        for rule in list_coms_user:
            # En esta parte se evalua la regla completa
            res = True
            for idx_r, attr_val in enumerate(rule[1]):
                # print(idx_r, attr_val)
                if row[attr_val[0]] != attr_val[1]:
                    res = False
                    break
            if res == False:
                denies_count += 1
            else:
                if not rule in rules_to_fix:
                    rules_to_fix.append(rule)

        # print("XXX-", denies_count, temp_rules_n, res)
        if denies_count < len(list_coms_user):
            false_positives.append(row)

    return false_positives, rules_to_fix


def get_id_from_attrs(users_attrs_values, dict_users):
    users_attrs_values = str(users_attrs_values.values).replace(" ", "")
    for idx, value in dict_users.items():
        if users_attrs_values == value:
            return idx

    return None


def get_FP_logs_ref(data_, user_network, list_rules, rule_network, rules_dict, neg_rules):
    """TRAIN DATA."""
    false_positives = []
    rules_to_fix = []
    for i, row in data_.iterrows():
        user_id = row["UID"]
        res_id = row["RID"]

        # User exist in User network
        if not user_id in user_network.vs["name"]:

            # Identificación de la comunidad
            copy_g = user_network.copy()
            # Se agrega el nodo a la red
            add_new_user_node_2(user_id, res_id, copy_g, data_)
            node_user = copy_g.vs[-1]  # Node de usuario

            # Identificar a las comunidades vecinas
            coms_vecinas = get_comunidades_vecinas(node_user, copy_g)

            # Probar la modularidad máxima
            max_mod = 0
            user_commty = None
            for id_com in coms_vecinas:
                comunidad = copy_g.vs.select(commty=id_com)
                comunidad = copy_g.subgraph(comunidad)
                temp_mod = modularity_evaluate(node_user, comunidad, copy_g)
                if temp_mod > max_mod:
                    max_mod = temp_mod
                    user_commty = id_com
        else:
            user_node = user_network.vs.find(name=user_id)
            user_commty = user_node["commty"]

        list_coms_user = obtener_reglas_comundiad(user_commty, list_rules)
        list_rules_idx = get_rule_id(list_coms_user, rules_dict)
        list_coms_user = get_neighbors_rules(
            list_rules_idx, rule_network, rules_dict)

        denies_count = 0
        for rule in list_coms_user:
            # En esta parte se evalua la regla completa
            res = True
            for idx_r, attr_val in enumerate(rule[1]):
                # print(idx_r, attr_val)
                if row[attr_val[0]] != attr_val[1]:
                    res = False
                    break
            if res == False:
                denies_count += 1

        if denies_count < len(list_coms_user):

            # Evaluación
            denies_count = 0
            for rule in neg_rules:
                res = True
                for attr_val in rule:
                    if row[attr_val[0]] != attr_val[1]:
                        res = False
                        break
                if res == False:
                    denies_count += 1

            if denies_count == len(list_coms_user):
                false_positives.append(row)

    return false_positives


def get_FN_logs_ref(data_, user_network, list_rules, rule_network, rules_dict):
    """TRAIN DATA."""
    false_neg = []
    for i, row in data_.iterrows():
        user_id = row["UID"]

        #user_node = user_network.vs.find(name=user_id)
        #user_commty = user_node["commty"]

        #list_coms_user = obtener_reglas_comundiad(user_commty, list_rules)
        #list_rules_idx = get_rule_id(list_coms_user, rules_dict)
        #list_coms_user = get_neighbors_rules(
        #    list_rules_idx, rule_network, rules_dict)

        # Evaluación
        denies_count = 0
        for rule in list_rules:
            res = True
            for idx_r, attr_val in enumerate(rule[1]):
                if row[attr_val[0]] != attr_val[1]:
                    res = False
                    break
            if res == False:
                denies_count += 1

        if denies_count == len(list_rules):
            false_neg.append(row)

    return false_neg
