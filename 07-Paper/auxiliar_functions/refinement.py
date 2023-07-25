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
    # print(rules)

    reglas_karimi = []
    for r in rules:
        for t in r:
            col = get_attr_name_by_value(t, data_)
            reglas_karimi.append([col, t])

    return reglas_karimi


def generate_negative_rules(neg_data, rules_to_fix, n_init_rules):
    focus_rules = []
    counter_ = 0
    for rule in rules_to_fix:
        if int(rule[0][1]) > n_init_rules:
            focus_rules.append([counter_, rule])
            counter_ += 1

    dict_rules_to_fix = {}
    false_neg = []
    for i, row in neg_data.iterrows():
        for rule in focus_rules:
            denies_count = 0
            res = True
            for j, attr_val in enumerate(rule[1][1]):
                if row[attr_val[0]] != attr_val[1]:
                    res = False
                    break
            if res == False:
                denies_count += 1
            else:
                if rule[0] in dict_rules_to_fix.keys():
                    dict_rules_to_fix[rule[0]
                                      ] = dict_rules_to_fix[rule[0]] + 1
                else:
                    dict_rules_to_fix[rule[0]] = 1

            if denies_count == len(focus_rules):
                false_neg.append(row)

    max_frequency = max(dict_rules_to_fix.values())
    rules_to_fix = []
    for idx, val in dict_rules_to_fix.items():
        if val > max_frequency/2:
            for rule in focus_rules:
                if idx == rule[0]:
                    rules_to_fix.append([idx, rule[1]])

    print(rules_to_fix)

    logs_in_rules = {}
    for rule in rules_to_fix:
        for i, row in neg_data.iterrows():
            res = True
            for attr_val in (rule[1][1]):
                if row[attr_val[0]] != attr_val[1]:
                    res = False
                    break
            if res:
                if not rule[0] in logs_in_rules.keys():
                    logs_in_rules[rule[0]] = [row]
                else:
                    logs_in_rules[rule[0]] = logs_in_rules[rule[0]] + [row]

    print(logs_in_rules)
