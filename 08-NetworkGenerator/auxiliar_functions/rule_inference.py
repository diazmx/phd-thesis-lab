from collections import Counter
from apriori_python import apriori


def remove_equal_rulesX(rules):
    rules = [set(r[:-1]) for r in rules]

    a = []
    for r in rules:
        if not r in a:
            a.append(r)

    max_id = 0
    to_ret = None
    for i in a:
        if len(i) > max_id:
            max_id = len(i)
            to_ret = i

    return [to_ret]


def frequent_resources(subcomunidad, grafo_bip, umbral):
    """Retorna los recursos nuevos ya podados."""
    usuario_comunidad = subcomunidad.vs()["name"]
    all_recursos = []  # DIccionario con los recursos y su frecuencia
    for user in usuario_comunidad:
        user_node = grafo_bip.vs.find(name=user)
        vecinos_recurso = user_node.neighbors()
        vecinos_recurso = [node["name"] for node in vecinos_recurso]
        vecinos_recurso = list(set(vecinos_recurso))
        # print(vecinos_recurso)
        all_recursos = all_recursos + vecinos_recurso

    umbral_en_n = int(umbral * subcomunidad.vcount())

    all_recursos = dict(Counter(all_recursos))

    frequent_resources = [id_res for (
        id_res, value_) in all_recursos.items() if value_ >= umbral_en_n]

    if len(frequent_resources) == 0:
        frequent_resources = [id_res for (id_res, value_) in all_recursos.items(
        ) if value_ >= max(all_recursos.values())]

    return frequent_resources


def get_attrs_from_res(data_, res_attr, resources):
    df_res_commty = data_[data_["RID"].isin(resources)]
    df_res_commty = df_res_commty[res_attr].drop_duplicates()

    return df_res_commty


def get_attrs_from_user(commty_network, data_, user_attr, resources, bip_network):
    users_commty = commty_network.vs["name"]
    df_users_commty = data_[data_["UID"].isin(users_commty)]
    df_users_commty = data_[data_["RID"].isin(resources)]
    df_users_commty = df_users_commty[user_attr+["UID"]].drop_duplicates()

    return df_users_commty


def get_attrs_from_user_sig(commty_network, data_, user_attr, sig_resource, bip_network):
    users_commty = commty_network.vs["name"]
    user_access_sig_resource = bip_network.vs.find(name=sig_resource)
    user_access_sig_resource = user_access_sig_resource.neighbors()
    user_access_sig_resource = [node["name"]
                                for node in user_access_sig_resource]
    users_commty = list(
        set(user_access_sig_resource).intersection(set(users_commty)))
    df_users_commty = data_[data_["UID"].isin(users_commty)]
    df_users_commty = df_users_commty[user_attr+["UID"]].drop_duplicates()
    return df_users_commty


def get_attr_name_by_value(value, data_):
    # value = list(value)[0]
    for attr in data_.columns:
        if value in list(data_[attr].drop_duplicates()):
            return attr
    return None


def attribute_value_common(data_):
    """Retorna regla apriori basada en los recursos.
    data_: dataframe
    """
    init_Sup = 0.6
    init_Conf = 0.9
    rules = []
    while len(rules) == 0:
        # print("Sup", init_Sup, "  Conf", init_Conf)
        _, rules = apriori(data_.values.tolist(), minSup=init_Sup,
                           minConf=init_Conf)  # Apply apriori
        init_Sup -= 0.1
        init_Conf -= 0.1

        if init_Sup < 0:
            rules = [data_.values[0][:-1], [], [1]]
            print(rules)
            break

    # print(rules)
    rules = [list(r[0])+list(r[1])+[r[-1]] for r in rules]
    rules = remove_equal_rulesX(rules)
    rules = [list(r) for r in rules]
    #print(rules)

    reglas_karimi = []
    for r in rules:
        for t in r:
            col = get_attr_name_by_value(t, data_)
            reglas_karimi.append([col, t])

    return reglas_karimi


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
