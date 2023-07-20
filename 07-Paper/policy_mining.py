import networkx as nx
from auxiliar_functions.data_preprocessing import add_new_index
from auxiliar_functions.network_model import build_network_model, bipartite_projection, plot_distribution_degree
from auxiliar_functions.community_detection import sub_community_detection, add_type_commts
from auxiliar_functions.rule_inference import frequent_resources, get_attrs_from_user_sig, get_attrs_from_user, get_attrs_from_res, attribute_value_common, evaluate_weight
from auxiliar_functions.evaluation import get_FN_logs, get_FP_logs
from sklearn.model_selection import StratifiedShuffleSplit

import numpy as np
import pandas as pd
pd.options.mode.chained_assignment = None


class PolicyMining:

    def __init__(self, file_name, name_dataset, user_attrs, resource_attrs) -> None:
        self.df_data = pd.read_csv(file_name)
        self.name_ds = name_dataset
        self.user_attrs = user_attrs
        self.resource_attrs = resource_attrs
        print("File loaded! \n")

        cross_validation_settings = {
            "k": 10,
            "id_k": 0,
            "test_size": 0.2,
            "random_state": 1
        }

        # Cross-Validation
        if cross_validation_settings["random_state"] != None:  # With random state
            kfold = StratifiedShuffleSplit(n_splits=cross_validation_settings["k"],
                                           test_size=cross_validation_settings["test_size"],
                                           random_state=cross_validation_settings["random_state"])
        else:
            kfold = StratifiedShuffleSplit(n_splits=cross_validation_settings["k"],
                                           test_size=cross_validation_settings["test_size"])

        data_partition = kfold.split(self.df_data, self.df_data.ACTION)
        data_corpus = []  # List with all data partitions

        for train_data, test_data in data_partition:
            X_train, X_test = self.df_data.iloc[train_data], self.df_data.iloc[test_data]
            data_corpus.append([X_train, X_test])

        print("Cross-Validation - DONE")
        print("- k =", cross_validation_settings["k"])
        print("- Percentage Train-Test:",
              (1-cross_validation_settings["test_size"])*100, "-",
              cross_validation_settings["test_size"]*100)

        # Selection of a split with the id_k ID.
        self.df_train_k = data_corpus[cross_validation_settings["id_k"]][0]
        self.df_test_k = data_corpus[cross_validation_settings["id_k"]][1]

        print("# Access requests in Train:", len(self.df_train_k),
              " %: {:.2f}".format((len(self.df_train_k)/(len(self.df_train_k)+len(self.df_test_k)))*100))
        print("# Access requests in Test:", len(self.df_test_k),
              " %: {:.2f}".format((len(self.df_test_k)/(len(self.df_train_k)+len(self.df_test_k)))*100))
        print("# Access requests:", len(self.df_train_k)+len(self.df_test_k))
        print()

    def data_preprocessing(self):
        print("\n##############################")
        print(" PHASE 1: Data Preprocessing.")
        print("##############################\n")
        if self.name_ds == 'AMZ':  # AMZ Dataset
            ###### ***** TASK 1 ***** #####
            # Handling missing and null values.
            print("\nTASK 1: Done!\n")  # Not applicable

            ###### ***** TASK 2 ***** #####
            # Converting continuous values to categorical values.
            print("TASK 2: Done!\n")  # Not applicable
            self.df_train_k = add_new_index(
                self.df_train_k, self.user_attrs, type="U")

            ###### ***** TASK 3 ***** #####
            # Removing duplicated access requests.
            self.df_train_k_pos = self.df_train_k[self.df_train_k.ACTION == 1]
            self.df_train_k_neg = self.df_train_k[self.df_train_k.ACTION == 0]
            # print(self.df_train_k_pos.columns[1:])
            self.df_train_k_pos = self.df_train_k_pos[self.df_train_k_pos.columns[1:]].drop_duplicates(
            )
            self.df_train_k_neg = self.df_train_k_neg[self.df_train_k_neg.columns[1:]].drop_duplicates(
            )
            print("# (+) access requests:", len(self.df_train_k_pos),
                  " %: {:.2f}".format((len(self.df_train_k_pos)/len(self.df_train_k))*100))
            print("# (-) access requests:", len(self.df_train_k_neg),
                  " %: {:.2f}".format((len(self.df_train_k_neg)/len(self.df_train_k))*100))
            print("TASK 3: Done!\n")

            ###### ***** TASK 4 ***** #####
            # Selecting the most used resources.
            n1 = 0
            n2 = 149
            top_list = self.df_train_k_pos.RID.value_counts()[:len(
                self.df_train_k_pos.RID.drop_duplicates())].index.tolist()
            # Filter the interval between n1 and n2
            top_list = top_list[n1:n2+1]
            boolean_series = self.df_train_k_pos.RID.isin(top_list)
            self.df_train_k_pos = self.df_train_k_pos[boolean_series]
            bolean_series = self.df_train_k_neg.RID.isin(top_list)
            self.df_train_k_neg = self.df_train_k_neg[bolean_series]
            print("TASK 4: Done!\n")

        elif self.name_ds == 'HC':
            ###### ***** TASK 1 ***** #####
            # Handling missing and null values.
            mapping = {"addnote": 0, "none": 0,
                       "additem": 0, "read": 0, "1": 1}
            self.df_train_k["ACTION"] = self.df_train_k["ACTION"].replace(
                mapping)
            self.df_test_k["ACTION"] = self.df_test_k["ACTION"].replace(
                mapping)

            mapping = {"none": 10, "doctor": 11, "nurse": 12}  # role
            self.df_train_k["role"] = self.df_train_k["role"].replace(mapping)
            self.df_test_k["role"] = self.df_test_k["role"].replace(mapping)

            mapping = {"note": 110, "cardiology": 111, "nursing": 112,
                       "oncology": 113, "none": 114}  # speacialty
            self.df_train_k["specialty"] = self.df_train_k["specialty"].replace(
                mapping)
            self.df_test_k["specialty"] = self.df_test_k["specialty"].replace(
                mapping)

            mapping = {"oncteam1": 1101, "carteam1": 1111,
                       "carteam2": 1121, "oncteam2": 1131, "none": 1141}  # tem
            self.df_train_k["team"] = self.df_train_k["team"].replace(mapping)
            self.df_test_k["team"] = self.df_test_k["team"].replace(mapping)

            mapping = {"carward": 11011,
                       "oncward": 11111, "none": 11211}  # uward
            self.df_train_k["uward"] = self.df_train_k["uward"].replace(
                mapping)
            self.df_test_k["uward"] = self.df_test_k["uward"].replace(mapping)

            mapping = {"oncpat1": 111011, "carpat1": 111111,  # agentfor
                       "oncpat2": 111211, "carpat2": 111311, "none": 111411}
            self.df_train_k["agentfor"] = self.df_train_k["agentfor"].replace(
                mapping)
            self.df_test_k["agentfor"] = self.df_test_k["agentfor"].replace(
                mapping)

            mapping = {"hr": 1110111, "hritem": 1111111,
                       "none": 1112111}  # type
            self.df_train_k["type"] = self.df_train_k["type"].replace(mapping)
            self.df_test_k["type"] = self.df_test_k["type"].replace(mapping)

            mapping = {"oncpat1": 211012, "carpat1": 211112,  # patient
                       "oncpat2": 211212, "carpat2": 211312, "none": 211412}
            self.df_train_k["patient"] = self.df_train_k["patient"].replace(
                mapping)
            self.df_test_k["patient"] = self.df_test_k["patient"].replace(
                mapping)

            mapping = {"oncteam1": 2102, "carteam1": 2112,
                       "carteam2": 2122, "oncteam2": 2132, "none": 2142}  # treatingteam
            self.df_train_k["treatingteam"] = self.df_train_k["treatingteam"].replace(
                mapping)
            self.df_test_k["treatingteam"] = self.df_test_k["treatingteam"].replace(
                mapping)

            mapping = {"carward": 21012,
                       "oncward": 21112, "none": 21212}  # oward
            self.df_train_k["oward"] = self.df_train_k["oward"].replace(
                mapping)
            self.df_test_k["oward"] = self.df_test_k["oward"].replace(mapping)

            mapping = {"note": 210, "cardiology": 211,
                       "nursing": 212, "oncology": 213, "none": 214}  # topic
            self.df_train_k["topic"] = self.df_train_k["topic"].replace(
                mapping)
            self.df_test_k["topic"] = self.df_test_k["topic"].replace(mapping)

            mapping = {"oncdoc2": 11110111, "carnurse1": 11111111, "oncnurse2": 11112111,  # author
                       "carnurse2": 11113111, "oncdoc1": 11114111, "oncnurse1": 11115111, "none": 11116111}
            self.df_train_k["author"] = self.df_train_k["author"].replace(
                mapping)
            self.df_test_k["author"] = self.df_test_k["author"].replace(
                mapping)

            self.df_train_k = self.df_train_k.drop(['user'], axis=1)
            self.df_train_k = add_new_index(
                self.df_train_k, self.user_attrs, type="U")
            self.df_train_k = add_new_index(
                self.df_train_k, self.resource_attrs, type="R")
            print("\nTASK 1: Done!\n")  # Not applicable

            ###### ***** TASK 2 ***** #####
            # Converting continuous values to categorical values.
            print("TASK 2: Done!\n")  # Not applicable

            ###### ***** TASK 3 ***** #####
            # Removing duplicated access requests.
            self.df_train_k_pos = self.df_train_k[self.df_train_k.ACTION == 1]
            self.df_train_k_neg = self.df_train_k[self.df_train_k.ACTION == 0]

            self.df_train_k_pos = self.df_train_k_pos[self.df_train_k_pos.columns[1:]].drop_duplicates(
            )
            self.df_train_k_neg = self.df_train_k_neg[self.df_train_k_neg.columns[1:]].drop_duplicates(
            )
            print("# (+) access requests:", len(self.df_train_k_pos),
                  " %: {:.2f}".format((len(self.df_train_k_pos)/len(self.df_train_k))*100))
            print("# (-) access requests:", len(self.df_train_k_neg),
                  " %: {:.2f}".format((len(self.df_train_k_neg)/len(self.df_train_k))*100))
            print("TASK 3: Done!\n")

            ###### ***** TASK 4 ***** #####
            # Selecting the most used resources.
            n1 = 0
            n2 = 210
            top_list = self.df_train_k_pos.RID.value_counts()[:len(
                self.df_train_k_pos.RID.drop_duplicates())].index.tolist()
            # Filter the interval between n1 and n2
            top_list = top_list[n1:n2+1]
            boolean_series = self.df_train_k_pos.RID.isin(top_list)
            self.df_train_k_pos = self.df_train_k_pos[boolean_series]
            bolean_series = self.df_train_k_neg.RID.isin(top_list)
            self.df_train_k_neg = self.df_train_k_neg[bolean_series]

            print("TASK 4: Done!\n")

        elif self.name_ds == 'CAV':
            print(self.name_ds)
        else:
            print("Invalid dataset:", self.name_ds)

        self.n_users = len(self.df_train_k.UID.drop_duplicates())
        self.n_rsrcs = len(self.df_train_k.RID.drop_duplicates())
        print("|U|: ", self.n_users, " -\t|R|: ",
              self.n_rsrcs)  # Unique resources

    def network_model(self):
        print("\n#########################")
        print(" PHASE 2: Network Model.")
        print("#########################\n")

        ###### ***** TASK 1 ***** #####
        # Access request bipartite network
        self.bip_network = build_network_model(
            self.df_train_k_pos, 'UID', 'RID')
        print("TASK 1: Done!\n")

        ###### ***** TASK 2 ***** #####
        # User network 3
        self.user_network = bipartite_projection(self.bip_network, 0)

        # Complex Network Analysis
        # avg_degree = sum(self.user_network.degree()) / \
        #     self.user_network.vcount()
        # print("\nNetwork Analysis")
        # print("- Avg. degree", "{:.4f}".format(avg_degree))

        # print("- Density:", "{:.4f}".format(self.user_network.density()))

        # cc = self.user_network.transitivity_avglocal_undirected()
        # print("- Clustering Coefficient:", "{:.4f}".format(cc))

        # L = self.user_network.average_path_length()
        # print("- Average Path Length :", "{:.4f}".format(L))

        # plot_distribution_degree(self.user_network, self.name_ds)
        # print("TASK 2: Done!\n")

    def community_detection(self):
        print("\n###############################")
        print(" PHASE 3: Community Detection.")
        print("###############################\n")

        ###### ***** TASK 1 ***** #####
        # Community detection
        partition = self.user_network.community_multilevel(
            weights=self.user_network.es()["weight"])

        # Modularity score
        print("Modularity: %.4f" % partition.modularity)

        # Add cluster attribute
        self.user_network.vs["commty"] = partition.membership

        print(self.user_network.summary())
        print(partition.summary())
        print("TASK 1: Done!\n")

        dict_commts = sub_community_detection(
            self.user_network, 0.5, None)

        ###### ***** TASK 2 ***** #####
        # Community calssification
        # Obtener el máximo valor de recursos en el total de comunidades
        n_res_in_comms = [len(i[1]) for i in dict_commts.values()]
        max_n_res = max(n_res_in_comms)
        # print("Comunidad con # mayor recursos", max_n_res)

        # Umbrales para la clasificación de comunidades
        big_threshold = int(0.50 * max_n_res)
        med_threshold = int(0.25 * max_n_res)
        print("Big Threshold: ", big_threshold,
              " \t\t Med Threshold", med_threshold)

        s_commts, m_commts, c_commts = add_type_commts(self.user_network, dict_commts,
                                                       big_threshold, med_threshold)
        self.all_commts = s_commts + m_commts + c_commts
        print("TASK 2: Done!\n")

    def rule_inference(self, th_rule_sim):
        print("\n##########################")
        print(" PHASE 4: Rule Inference.")
        print("##########################\n")

        ###### ***** TASK 1 ***** #####
        # Rule extraction
        self.list_rules = []  # Lista de reglas
        th_lfr = 0.2
        th_sr = 0.5

        for commty_ in self.all_commts:
            commty_resources = commty_[1][1]  # Get resources
            # print()

            if commty_[1][2] != 2:

                commty_resources = frequent_resources(
                    commty_[1][0], self.bip_network, th_lfr)

                if commty_[1][2] == 0:
                    commty_significant_res = frequent_resources(
                        commty_[1][0], self.bip_network, th_sr)
                    for sig_resource in commty_significant_res:

                        # Create a rule
                        # Se comienza generando la regla.
                        rule_i = [["id_com", str(commty_[0])], []]
                        for attr in self.resource_attrs:
                            logs_with_resource = self.df_train_k_pos[self.df_train_k_pos["RID"]
                                                                     == sig_resource].iloc[0]
                            rule_i[1].append([attr, logs_with_resource[attr]])

                        # Atributos frecuentes en usuarios
                        df_users_commty = get_attrs_from_user_sig(
                            commty_[1][0], self.df_train_k_pos, self.user_attrs, sig_resource, self.bip_network)

                        rule_user_attrs = attribute_value_common(
                            df_users_commty)

                        rule_i[1] = rule_i[1] + rule_user_attrs
                        self.list_rules.append(rule_i)

                    # print("TIPO 3 SIG:", commty_significant_res)
                    commty_resources = [
                        i for i in commty_resources if i not in commty_significant_res]

                    if len(commty_resources) < 1:
                        continue
                    # print("TIPO 3 RESTA:", commty_resources)

            # Create the other rule
            # Se comienza generando la regla.
            rule_i = [["id_com", str(commty_[0])], []]
            if self.name_ds != "AMZ":
                df_res_commty = get_attrs_from_res(
                    self.df_train_k_pos, self.resource_attrs, commty_resources)

                rule_res_attrs = attribute_value_common(df_res_commty)
                rule_i[1] = rule_i[1] + rule_res_attrs

            # Atributos frecuentes en usuarios
            df_users_commty = get_attrs_from_user(
                commty_[1][0], self.df_train_k_pos, self.user_attrs,
                commty_resources, self.bip_network)
            # print(df_users_commty.values[0])
            rule_user_attrs = attribute_value_common(df_users_commty)
            rule_i[1] = rule_i[1] + rule_user_attrs
            self.list_rules.append(rule_i)
        print("|R|:", len(self.list_rules))
        print("TASK 1: Done!\n")

        ###### ***** TASK 2 ***** #####
        # Rule Network
        self.rules_with_idx = {}
        for idx, rule in enumerate(self.list_rules):
            self.rules_with_idx[idx] = rule

        # Create the graph
        edge_list = []
        for idxA in range(len(self.list_rules)):
            for idxB in range(idxA, len(self.list_rules)):
                edge_temp = evaluate_weight(
                    self.rules_with_idx[idxA][1], self.rules_with_idx[idxB][1],
                    th_rule_sim)
                if edge_temp != -1:
                    edge_list.append((idxA, idxB, edge_temp))

        self.rule_network = nx.Graph()
        self.rule_network.add_weighted_edges_from(edge_list)

        # Add rule attribute
        self.rule_network.remove_edges_from(
            nx.selfloop_edges(self.rule_network))

        # print(self.rule_network.nodes, len(self.rule_network.nodes))
        # print(self.rules_with_idx.keys(), len(self.rules_with_idx.keys()))
        isolated_nodes = [
            i for i in self.rules_with_idx.keys() if i not in self.rule_network.nodes]
        # print(isolated_nodes, len(isolated_nodes))
        self.rule_network.add_nodes_from(isolated_nodes)

        print("Rule Network\n", nx.info(self.rule_network))
        print("TASK 2: Done!\n")

    def evaluation(self):
        print("\n#############")
        print(" Evaluation.")
        print("#############\n")

        ###### ***** TASK 1 ***** #####
        # FN Refinemente
        self.fn_logs = get_FN_logs(
            self.df_train_k_pos, self.user_network, self.list_rules,
            self.rule_network, self.rules_with_idx)

        self.fp_logs = get_FP_logs(
            self.df_train_k_neg, self.user_network, self.list_rules,
            self.rule_network, self.rules_with_idx)

        TP = len(self.df_train_k_pos) - len(self.fn_logs)
        # TN = len(self.df_train_k_neg) - len(self.fp_logs)

        precision = TP / (TP + len(self.fp_logs))

        recall = TP / (TP + len(self.fn_logs))

        fscore = 2*(precision*recall)/(precision+recall)

        print("FN:", len(self.fn_logs),
              " - {:.2f}%".format((len(self.fn_logs)/len(self.df_train_k_pos))*100))
        print("FP:", len(self.fp_logs),
              " - {:.2f}%".format((len(self.fp_logs)/len(self.df_train_k_neg))*100))
        print("Precision:", precision)
        print("Recall:", recall)
        print("F-score", fscore)

        wsc = sum([len(rule[1]) for rule in self.list_rules])

        print("# Rules:", len(self.list_rules))
        print("WSC:", wsc)

    def policy_refinement(self, th_rule_sim):

        print("\n#############################")
        print(" PHASE 5: Policy Refinement.")
        print("#############################\n")

        df_fn = pd.DataFrame(self.fn_logs)

        ###### ***** TASK 1 ***** #####
        # Access request bipartite network
        self.bip_network_ref = build_network_model(df_fn, 'UID', 'RID')
        print("TASK 1: Done!\n")

        ###### ***** TASK 2 ***** #####
        # User network 3
        self.user_network_ref = bipartite_projection(self.bip_network_ref, 0)

        ###### ***** TASK 1 ***** #####
        # Community detection
        partition_ref = self.user_network_ref.community_multilevel(
            weights=self.user_network_ref.es()["weight"])

        # Modularity score
        print("Modularity: %.4f" % partition_ref.modularity)

        # Add cluster attribute
        self.user_network_ref.vs["commty"] = partition_ref.membership

        print(self.user_network_ref.summary())
        print(partition_ref.summary())
        print("TASK 1: Done!\n")

        dict_commts_ref = sub_community_detection(
            self.user_network_ref, 0.5, len(self.rule_network))

        ###### ***** TASK 2 ***** #####
        # Community calssification
        # Obtener el máximo valor de recursos en el total de comunidades
        n_res_in_comms = [len(i[1]) for i in dict_commts_ref.values()]
        max_n_res = max(n_res_in_comms)
        # print("Comunidad con # mayor recursos", max_n_res)

        # Umbrales para la clasificación de comunidades
        big_threshold = int(0.50 * max_n_res)
        med_threshold = int(0.25 * max_n_res)
        print("Big Threshold: ", big_threshold,
              " \t\t Med Threshold", med_threshold)

        s_commts, m_commts, c_commts = add_type_commts(self.user_network_ref, dict_commts_ref,
                                                       big_threshold, med_threshold)
        self.all_commts_ref = s_commts + m_commts + c_commts
        print("TASK 2: Done!\n")

        ###### ***** TASK 1 ***** #####
        # Rule extraction
        self.list_rules_ref = []  # Lista de reglas
        th_lfr = 0.2
        th_sr = 0.5

        for commty_ in self.all_commts_ref:
            commty_resources = commty_[1][1]  # Get resources
            # print(commty_[0])
            # print(commty_[1][2])

            if commty_[1][2] != 2:

                commty_resources = frequent_resources(
                    commty_[1][0], self.bip_network_ref, th_lfr)

                if commty_[1][2] == 0:
                    commty_significant_res = frequent_resources(
                        commty_[1][0], self.bip_network_ref, th_sr)
                    for sig_resource in commty_significant_res:

                        # Create a rule
                        # Se comienza generando la regla.
                        rule_i = [["id_com", str(commty_[0])], []]
                        for attr in self.resource_attrs:
                            logs_with_resource = df_fn[df_fn["RID"]
                                                       == sig_resource].iloc[0]
                            rule_i[1].append([attr, logs_with_resource[attr]])

                        # Atributos frecuentes en usuarios
                        df_users_commty = get_attrs_from_user_sig(
                            commty_[1][0], df_fn, self.user_attrs, sig_resource, self.bip_network_ref)

                        rule_user_attrs = attribute_value_common(
                            df_users_commty)

                        rule_i[1] = rule_i[1] + rule_user_attrs
                        self.list_rules_ref.append(rule_i)

                    # print("TIPO 3 SIG:", commty_significant_res)
                    commty_resources = [
                        i for i in commty_resources if i not in commty_significant_res]

                    if len(commty_resources) < 1:
                        continue
                    # print("TIPO 3 RESTA:", commty_resources)

            # Create the other rule
            # Se comienza generando la regla.
            rule_i = [["id_com", str(commty_[0])], []]
            if self.name_ds != "AMZ":
                df_res_commty = get_attrs_from_res(
                    df_fn, self.resource_attrs, commty_resources)

                rule_res_attrs = attribute_value_common(df_res_commty)
                rule_i[1] = rule_i[1] + rule_res_attrs

            # Atributos frecuentes en usuarios
            df_users_commty = get_attrs_from_user(
                commty_[1][0], df_fn, self.user_attrs,
                commty_resources, self.bip_network_ref)
            # print(df_users_commty)
            rule_user_attrs = attribute_value_common(df_users_commty)
            # print(rule_user_attrs)
            rule_i[1] = rule_i[1] + rule_user_attrs
            self.list_rules_ref.append(rule_i)
        print("|R|:", len(self.list_rules_ref))
        print("TASK 1: Done!\n")

        ###### ***** TASK 2 ***** #####
        # Rule Network
        self.rules_with_idx = {}
        self.total_rules = self.list_rules + self.list_rules_ref
        for idx, rule in enumerate(self.total_rules):
            self.rules_with_idx[idx] = rule

        # Create the graph
        edge_list = []
        for idxA in range(len(self.total_rules)):
            for idxB in range(idxA, len(self.total_rules)):
                edge_temp = evaluate_weight(
                    self.rules_with_idx[idxA][1], self.rules_with_idx[idxB][1],
                    th_rule_sim)
                if edge_temp != -1:
                    edge_list.append((idxA, idxB, edge_temp))

        self.rule_network_ref = nx.Graph()
        self.rule_network_ref.add_weighted_edges_from(edge_list)

        # Add rule attribute
        self.rule_network_ref.remove_edges_from(
            nx.selfloop_edges(self.rule_network_ref))

        isolated_nodes = [
            i for i in self.rules_with_idx.keys() if i not in self.rule_network_ref.nodes]
        # print(isolated_nodes, len(isolated_nodes))
        self.rule_network_ref.add_nodes_from(isolated_nodes)

        print("Rule Network\n", nx.info(self.rule_network_ref))
        print("TASK 2: Done!\n")

        copy_g_proj2 = self.user_network.copy()  # Copia del grafo
        modificaciones = 0

        lista_comid = []
        # Se agrega su id de comunidad
        for node in copy_g_proj2.vs():
            bandera = True
            for i in dict_commts_ref:
                if str(node["name"]) in dict_commts_ref[i][0].vs()["name"]:
                    # print(i)
                    modificaciones += 1
                    lista_comid.append(i)
                    bandera = False
            if bandera:
                lista_comid.append(node["commty"])

        print("Usuarios modificar: ", modificaciones)
        print("Tamano lista Comid: ", len(lista_comid))
        copy_g_proj2.vs["commty"] = lista_comid
        # print(copy_g_proj2.summary())
        # print(copy_g_proj2.vs()[0:10])
        print()

        ###### ***** EVALUATION ***** ##########################
        df_test_k_pos = self.df_test_k[self.df_test_k.ACTION == 1]

        df_users = pd.DataFrame(
            self.df_train_k_pos[self.user_attrs+["UID"]].drop_duplicates())
        users_ids_list = list(df_users.UID)
        users_attrs_list = list(df_users[self.user_attrs].values)
        users_attrs_list = [str(i).replace(" ", "") for i in users_attrs_list]
        users_attrs_dict = dict(zip(users_ids_list, users_attrs_list))

        df_test_k_neg = self.df_test_k[self.df_test_k.ACTION == 0]
        # print(df_test_k_neg.columns)
        # FN Refinemente
        # self.fn_logs = get_FN_logs(
        #    df_test_k_pos, self.user_network, self.total_rules,
        #    self.rule_network_ref, self.rules_with_idx)

        # self.fp_logs = get_FP_logs(
        #    df_test_k_neg, self.user_network, self.total_rules,
        #    self.rule_network_ref, self.rules_with_idx)

        TP = len(df_test_k_pos) - len(self.fn_logs)
        TN = len(df_test_k_neg) - len(self.fp_logs)

        precision = TP / (TP + len(self.fp_logs))

        recall = TP / (TP + len(self.fn_logs))

        fscore = 2*(precision*recall)/(precision+recall)

        print("FN:", len(self.fn_logs),
              " - {:.2f}%".format((len(self.fn_logs)/len(df_test_k_pos))*100))
        print("FP:", len(self.fp_logs),
              " - {:.2f}%".format((len(self.fp_logs)/len(df_test_k_neg))*100))
        print("Precision:", precision)
        print("Recall:", recall)
        print("F-score", fscore)

        wsc = sum([len(rule[1]) for rule in self.total_rules])

        print("# Rules:", len(self.total_rules))
        print("WSC:", wsc)
