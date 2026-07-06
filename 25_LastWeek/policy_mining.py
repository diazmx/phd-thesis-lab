import numpy as np
import pandas as pd
import networkx as nx
import igraph as ig
import matplotlib.pyplot as plt
import random
import copy
from itertools import combinations
from collections import Counter, defaultdict
import sys
import json
from auxiliar_functions.data_preprocessing import add_new_index
from auxiliar_functions.network_model import build_network_model, bipartite_projection, plot_distribution_degree
from auxiliar_functions.community_detection import sub_community_detection, add_type_commts, sub_community_detection_nx, add_type_commts_nx
from auxiliar_functions.rule_inference import frequent_resources, get_attrs_from_user_sig, get_attrs_from_user, get_attrs_from_res, attribute_value_common, evaluate_weight
from auxiliar_functions.evaluation import get_FN_logs, get_FP_logs, get_FP_logs_ref, get_FN_logs_refi, get_FN_logs_dos, eliminar_reglas_duplicadas
from auxiliar_functions.refinement import generate_negative_rules
from sklearn.model_selection import StratifiedShuffleSplit

pd.options.mode.chained_assignment = None

# Función para convertir tipos numpy a tipos nativos de Python
def convertir_a_nativo(obj):
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convertir_a_nativo(val) for key, val in obj.items()}
    elif isinstance(obj, list):
        return [convertir_a_nativo(elemento) for elemento in obj]
    elif isinstance(obj, tuple):
        return tuple(convertir_a_nativo(elemento) for elemento in obj)
    else:
        return obj


def jaccard_index(set1, set2):
    intersection = len(set1 & set2)
    union = len(set1 | set2)
    return intersection / union if union != 0 else 0.0

def run_louvain_iterations(g, iterations=10):
    same_community_neighbors = defaultdict(list)

    for _ in range(iterations):
        partition = g.community_multilevel()
        membership = partition.membership

        node_to_comm = {v.index: membership[v.index] for v in g.vs}

        for v in g.vs:
            v_id = v["id"]
            community_neighbors = set()

            for neighbor in g.neighbors(v):
                if node_to_comm[neighbor] == node_to_comm[v.index]:
                    community_neighbors.add(g.vs[neighbor]["id"])

            same_community_neighbors[v_id].append(community_neighbors)

    return same_community_neighbors

def compute_jaccard_stability(same_community_neighbors):
    stability_scores = {}

    for node_id, neighbor_sets in same_community_neighbors.items():
        if len(neighbor_sets) < 2:
            stability_scores[node_id] = 1.0
            continue

        jaccard_scores = []
        for s1, s2 in combinations(neighbor_sets, 2):
            jaccard_scores.append(jaccard_index(s1, s2))

        stability_scores[node_id] = sum(jaccard_scores) / len(jaccard_scores)
    return stability_scores


class PolicyMining:

    def __init__(self, file_name, name_dataset, user_attrs, resource_attrs) -> None:
        self.df_data = pd.read_csv(file_name)
        self.name_ds = name_dataset
        self.user_attrs = user_attrs
        self.resource_attrs = resource_attrs
        print("File loaded! \n")
        print("ESTE es")

    def data_preprocessing(self, is_noisy=False, is_sparse=False):
        print("\n##############################")
        print(" PHASE 1: Data Preprocessing.")
        print("##############################\n")
        if self.name_ds == 'AMZ':  # AMZ Dataset
            ###### ***** TASK 1 ***** #####
            # Handling missing and null values.

            print("\nTASK 1: Done!\n")  # Not applicable
            self.df_data = self.df_data.drop_duplicates()

            if is_noisy:
                self.df_data = introducir_ruido(self.df_data)

            if is_sparse:
                self.df_data = crear_version_sparse(self.df_data)
            ###### ***** TASK 2 ***** #####
            # Converting continuous values to categorical values.
            print("TASK 2: Done!\n")  # Not applicable
            

            ###### ***** TASK 3 ***** #####
            # Removing duplicated access requests.
            self.df_data_pos = self.df_data[self.df_data.ACTION == 1]
            self.df_data_neg = self.df_data[self.df_data.ACTION == 0]

            ###### ***** TASK 4 ***** #####
            # Selecting the most used resources.
            n1 = 0
            n2 = 149
            top_list = self.df_data.RID.value_counts()[:len(
                self.df_data.RID.drop_duplicates())].index.tolist()
            # Filter the interval between n1 and n2
            top_list = top_list[n1:n2+1]
            boolean_series = self.df_data_pos.RID.isin(top_list)
            self.df_data_pos = self.df_data_pos[boolean_series]
            bolean_series = self.df_data_neg.RID.isin(top_list)
            self.df_data_neg = self.df_data_neg[bolean_series]
            print("TASK 4: Done!\n")

        elif self.name_ds == 'HC':
            ###### ***** TASK 1 ***** #####
            # Handling missing and null values.
            #mapping = {"addnote": 0, "none": 0,
            #           "additem": 0, "read": 0, "1": 1}
            #self.df_data["ACTION"] = self.df_data["ACTION"].replace(
            #    mapping)

            mapping = {"Newone": 10, "doctor": 11, "nurse": 12}  # role
            self.df_data["position"] = self.df_data["position"].replace(mapping)

            mapping = {"anesthesiology": 110, "cardiology": 111, "pediatrics": 112,
                       "oncology": 113, "Newone": 114}  # speacialty
            self.df_data["specialties"] = self.df_data["specialties"].replace(
                mapping)

            mapping = {"oncTeam1": 1101, "carTeam1": 1111,
                       "carTeam2": 1121, "oncTeam2": 1131, "Newone": 1141}  # tem
            self.df_data["teams"] = self.df_data["teams"].replace(mapping)


            mapping = {"carward": 11011,
                       "oncward": 11111, "Newone": 11211}  # uward
            self.df_data["uward"] = self.df_data["uward"].replace(
                mapping)

            mapping = {"oncPat1": 111011, "carPat1": 111111,  # agentfor
                       "oncpat2": 111211, "carpat2": 111311, "Newone": 111411}
            self.df_data["agentfor"] = self.df_data["agentfor"].replace(
                mapping)


            mapping = {"HR": 1110111, "item": 1111111,
                       "Newone": 1112111}  # type
            self.df_data["type"] = self.df_data["type"].replace(mapping)

            mapping = {"oncPat1": 211012, "carPat1": 211112,  # patient
                       "oncpat2": 211212, "carpat2": 211312, "Newone": 211412}
            self.df_data["patient"] = self.df_data["patient"].replace(
                mapping)

            mapping = {"oncTeam1": 2102, "oncTeam2": 2112,
                       "carTeam1": 2122, "carTeam2": 2132, "Newone": 2142}  # treatingteam
            self.df_data["tratingTeam"] = self.df_data["tratingTeam"].replace(
                mapping)

            mapping = {"oncward": 21012,
                       "carward": 21112, "Newone": 21212}  # oward
            self.df_data["rward"] = self.df_data["rward"].replace(
                mapping)

            mapping = {"oncology": 210, "nursing": 211,
                       "note": 212, "Newone": 213, "cardiology": 214}  # topic
            self.df_data["topics"] = self.df_data["topics"].replace(
                mapping)

            #mapping = {"oncNurse1": 11110111, "carNurse1": 11111111, "doc1": 11112111,  # author
            #           "carnurse2": 11113111, "oncdoc1": 11114111, "oncnurse1": 11115111, "Newone": 11116111}
            #self.df_data["author"] = self.df_data["author"].replace(
            #    mapping)

            self.df_data = self.df_data.drop(['uname'], axis=1)
            self.df_data = add_new_index(
                self.df_data, self.user_attrs, type="U")
            self.df_data = add_new_index(
                self.df_data, self.resource_attrs, type="R")
            print("\nTASK 1: Done!\n")  # Not applicable

            ###### ***** TASK 2 ***** #####
            # Converting continuous values to categorical values.
            print("TASK 2: Done!\n")  # Not applicable

            if self.name_ds == "HC":
                self.df_data.rename(columns={'UID': 'uname'}, inplace=True)

            ###### ***** TASK 3 ***** #####
            # Removing duplicated access requests.
            self.df_data_pos = self.df_data[self.df_data.ACTION == 1]

            self.df_data_neg = self.df_data[self.df_data.ACTION == 0]

            self.df_data_pos = self.df_data_pos[self.df_data_pos.columns[1:]].drop_duplicates()
            #self.df_data_pos = self.df_data_pos.drop_duplicates()
            #self.df_data_neg = self.df_data_neg.drop_duplicates()
            self.df_data_neg = self.df_data_neg.drop_duplicates()
            print("# (+) access requests:", len(self.df_data_pos),
                  " %: {:.2f}".format((len(self.df_data_pos)/len(self.df_data))*100))
            print("# (-) access requests:", len(self.df_data_neg),
                  " %: {:.2f}".format((len(self.df_data_neg)/len(self.df_data))*100))
            print("TASK 3: Done!\n")

            ###### ***** TASK 4 ***** #####
            # Selecting the most used resources.
            n1 = 0
            n2 = 210
            top_list = self.df_data_pos.RID.value_counts()[:len(
                self.df_data_pos.RID.drop_duplicates())].index.tolist()
            # Filter the interval between n1 and n2
            top_list = top_list[n1:n2+1]
            boolean_series = self.df_data_pos.RID.isin(top_list)
            self.df_data_pos = self.df_data_pos[boolean_series]
            bolean_series = self.df_data_neg.RID.isin(top_list)
            self.df_data_neg = self.df_data_neg[bolean_series]

            print("TASK 4: Done!\n")

        elif self.name_ds == 'CAV':

            self.df_data = self.df_data[['ACTION'] + self.user_attrs +
                                              self.resource_attrs]
            # Change string values to numerical
            mapping = {'system': 10101, 'human': 10201,
                       'human and system': 10301}  # Control
            self.df_data["control"] = self.df_data["control"].replace(
                mapping)
            mapping = {'system': 20102, 'human': 20202}  # monitoring
            self.df_data["monitoring"] = self.df_data["monitoring"].replace(
                mapping)
            mapping = {'system': 30103, 'human': 30203}  # fallbacj
            self.df_data["fallback"] = self.df_data["fallback"].replace(
                mapping)
            mapping = {0: 40004, 1: 40104, 2: 40204,
                       3: 40304, 4: 40404, 5: 40504}
            self.df_data["driving_task_loa"] = self.df_data["driving_task_loa"].replace(
                mapping)
            mapping = {0: 50005, 1: 50105, 2: 50205,
                       3: 50305, 4: 50405, 5: 50505}
            self.df_data["vehicle_loa"] = self.df_data["vehicle_loa"].replace(
                mapping)
            mapping = {0: 60006, 1: 60106, 2: 60206,
                       3: 60306, 4: 60406, 5: 60506}
            self.df_data["region_loa"] = self.df_data["region_loa"].replace(
                mapping)

            self.df_data = add_new_index(
                self.df_data, self.user_attrs, type="U")
            self.df_data = add_new_index(
                self.df_data, self.resource_attrs, type="R")
            print("\nTASK 1: Done!\n")  #

            ###### ***** TASK 2 ***** #####
            # Converting continuous values to categorical values.
            print("TASK 2: Done!\n")  # Not applicable

            ###### ***** TASK 3 ***** #####
            # Removing duplicated access requests.
            self.df_data_pos = self.df_data[self.df_data.ACTION == 1]
            self.df_data_neg = self.df_data[self.df_data.ACTION == 0]

            self.df_data_pos = self.df_data_pos[self.df_data_pos.columns[1:]].drop_duplicates(
            )
            self.df_data_neg = self.df_data_neg.drop_duplicates(
            )
            print("# (+) access requests:", len(self.df_data_pos),
                  " %: {:.2f}".format((len(self.df_data_pos)/len(self.df_data))*100))
            print("# (-) access requests:", len(self.df_data_neg),
                  " %: {:.2f}".format((len(self.df_data_neg)/len(self.df_data))*100))
            print("TASK 3: Done!\n")

        else:
            print("Invalid dataset:", self.name_ds)
            sys.exit()

        
        

        print(self.df_data_pos)
        self.n_users = len(self.df_data.uname.drop_duplicates())
        self.n_rsrcs = len(self.df_data.RID.drop_duplicates())
        print("|U|: ", self.n_users, " -\t|R|: ",
              self.n_rsrcs, len(self.df_data))  # Unique resources

    def network_model(self):
        print("\n#########################")
        print(" PHASE 2: Network Model.")
        print("#########################\n")

        ###### ***** TASK 1 ***** #####
        # Access request bipartite network
        self.bip_network = build_network_model(
            self.df_data_pos, 'uname', 'RID')
        print("TASK 1: Done!\n")

        ###### ***** TASK 2 ***** #####
        # User network 3
        self.user_network = bipartite_projection(self.bip_network, 0)

        # Complex Network Analysis
        #avg_degree = sum(self.user_network.degree()) / \
        #    self.user_network.vcount()
        #print("\nNetwork Analysis")
        #print("- Avg. degree", "{:.4f}".format(avg_degree))

        #print("- Density:", "{:.4f}".format(self.user_network.density()))

        #cc = self.user_network.transitivity_avglocal_undirected()
        #print("- Clustering Coefficient:", "{:.4f}".format(cc))

        #L = self.user_network.average_path_length()
        #print("- Average Path Length :", "{:.4f}".format(L))

        #plot_distribution_degree(self.user_network, self.name_ds)
        ##### P6: Scale-free distribution p(k) = ∼k−α. #####
        # print("P6: Scale-free distribution p(k) = ∼k^{−α}.")
        # distri_grados = [i/self.user_network.vcount() for i in self.user_network.degree()]
        # plt.plot(sorted(distri_grados,reverse=True),linestyle='', marker='o', color="0.4")
        # plt.yscale("log")
        # plt.xscale("log")
        # plt.xlabel('Degree k.', fontsize=16)
        # plt.ylabel('Fraction of nodes with degree k.', fontsize=16)
        # plt.xticks(fontsize=15)
        # plt.yticks(fontsize=15)
        # plt.tight_layout()
        # plt.savefig('degree-distri-'+self.name_ds+'.png')
        
        #self.generate_random_network()
        

        print("TASK 2: Done!\n")
 
    def community_detection(self, big_threshold_ratio=0.5, med_threshold_ratio=0.25):
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

        dict_commts = sub_community_detection(self.user_network, 0.5, None)

        ###### ***** TASK 2 ***** #####
        # Community classification
        n_res_in_comms = [len(i[1]) for i in dict_commts.values()]
        max_n_res = max(n_res_in_comms)

        # Thresholds based on user-defined ratios
        big_threshold = int(big_threshold_ratio * max_n_res)
        med_threshold = int(med_threshold_ratio * max_n_res)
        print("Big Threshold:", big_threshold, "\t\tMed Threshold:", med_threshold)

        s_commts, m_commts, c_commts = add_type_commts(
            self.user_network, dict_commts, big_threshold, med_threshold)

        self.all_commts = s_commts + m_commts + c_commts
        print("TASK 2: Done!\n")

    def community_detection_nx(self, k_clique_value=3, big_threshold_ratio=0.5, med_threshold_ratio=0.25):
        print("\n###############################")
        print(" PHASE 3: Community Detection (NetworkX k-clique, Ejecución Única).")
        print("###############################\n")

        # --- Pre-Procesamiento: Conversión a NetworkX ---
        
        # 1. Crear una COPIA del grafo en formato NetworkX para la detección
        if isinstance(self.user_network, nx.Graph):
            # Si ya es NetworkX, simplemente lo copiamos para aislar los cambios
            nx_graph_copy = self.user_network.copy()
            print("Grafo: NetworkX (usando copia).")
        elif hasattr(self.user_network, 'to_networkx'):
            # Si tiene el método de conversión (ej. igraph), lo convertimos
            nx_graph_copy = self.user_network.to_networkx()
            print("Grafo: Convertido de igraph a NetworkX (usando copia).")
        else:
            raise TypeError("El objeto self.user_network no es NetworkX ni tiene el método to_networkx() para la conversión.")
            
        G = nx_graph_copy
        
        # ----------------------------------------------------
        # ***** TASK 1 *****: Detección de Comunidades
        # ----------------------------------------------------
        
        print(f"Usando k-clique con k = {k_clique_value}")
        
        # Ejecutar k_clique_communities
        communities_generator = nx.algorithms.community.k_clique_communities(G, k=k_clique_value)
        
        dict_commts = {}
        commty_counter = 0

        # Mapear el generador a la estructura de diccionario y asignar 'commty'
        for community_set in communities_generator:
            id_commty_str = str(commty_counter)
            dict_commts[id_commty_str] = [None, list(community_set)] 
            commty_counter += 1
            
        # Asignar 'commty' (lista de IDs) al atributo de los nodos del grafo temporal G
        for node in G.nodes():
            member_of = [comm_id for comm_id, value in dict_commts.items() if node in value[1]]
            G.nodes[node]["commty"] = member_of
        
        print("Modularidad: N/A (k-clique genera comunidades superpuestas).")
        print(f"Comunidades k={k_clique_value} encontradas: {len(dict_commts)}")
        print("TASK 1: Done!\n")

        # ----------------------------------------------------
        # ***** TASK 2 *****: Clasificación de Comunidades
        # ----------------------------------------------------
        
        n_res_in_comms = [len(i[1]) for i in dict_commts.values()] 
        max_n_res = max(n_res_in_comms) if n_res_in_comms else 0

        big_threshold = int(big_threshold_ratio * max_n_res)
        med_threshold = int(med_threshold_ratio * max_n_res)
        print("Big Threshold:", big_threshold, "\t\tMed Threshold:", med_threshold)

        # Clasificación (asigna 'tpcommty' al grafo temporal G)
        s_commts, m_commts, c_commts = add_type_commts_nx(
            G, dict_commts, big_threshold, med_threshold)
        
        # --- Post-Procesamiento: Transferencia de Atributos ---
        
        # Transferir los atributos 'commty' y 'tpcommty' del grafo temporal (G)
        # de vuelta al grafo original (self.user_network)
        transfer_attributes_to_original_graph(self.user_network, G)

        self.all_commts = s_commts + m_commts + c_commts
        print("TASK 2: Done!\n")
        
        return self.all_commts

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
                            logs_with_resource = self.df_data_pos[self.df_data_pos["RID"]
                                                                     == sig_resource].iloc[0]
                            rule_i[1].append([attr, logs_with_resource[attr]])

                        # Atributos frecuentes en usuarios
                        df_users_commty = get_attrs_from_user_sig(
                            commty_[1][0], self.df_data_pos, self.user_attrs, sig_resource, self.bip_network)

                        rule_user_attrs = attribute_value_common(
                            df_users_commty)
                        if df_users_commty.empty:
                            continue
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
                    self.df_data_pos, self.resource_attrs, commty_resources)

                rule_res_attrs = attribute_value_common(df_res_commty)
                rule_i[1] = rule_i[1] + rule_res_attrs

            # Atributos frecuentes en usuarios
            df_users_commty = get_attrs_from_user(
                commty_[1][0], self.df_data_pos, self.user_attrs,
                commty_resources, self.bip_network)
            # print(df_users_commty.values[0])
            rule_user_attrs = attribute_value_common(df_users_commty)
            rule_i[1] = rule_i[1] + rule_user_attrs
            self.list_rules.append(rule_i)
        print("Antes: |R|:", len(self.list_rules))
        #print(self.list_rules)

        self.list_rules = eliminar_reglas_duplicadas(self.list_rules)
        print("Despues: |R|:", len(self.list_rules))
        
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

        print("Rule Network\n", self.rule_network)
        print("TASK 2: Done!\n")

    def evaluation(self):
        print("\n#############")
        print(" Evaluation.")
        print("#############\n")

        self.rule_usage_counter = {i: 0 for i in range(len(self.list_rules))}

        ###### ***** TASK 1 ***** #####
        # Get False Negative Set
        self.fn_logs = get_FN_logs(
            self.df_data_pos, self.user_network, self.list_rules,
            self.rule_network, self.rules_with_idx)

        # Get False Positive Set
        print(self.df_data_neg.columns)
        self.fp_logs = get_FP_logs(
            self.df_data_neg, self.user_network, self.list_rules,
            self.rule_network, self.rules_with_idx)

        TP = len(self.df_data_pos) - len(self.fn_logs)
        TN = len(self.df_data_neg) - len(self.fp_logs)

        precision = TP / (TP + len(self.fp_logs))

        recall = TP / (TP + len(self.fn_logs))

        fscore = 2*(precision*recall)/(precision+recall)

        print("FN:", len(self.fn_logs),
              " - {:.2f}%".format((len(self.fn_logs)/len(self.df_data_pos))*100))
        print("FP:", len(self.fp_logs),
              " - {:.2f}%".format((len(self.fp_logs)/len(self.df_data_neg))*100))
        print("Precision:", precision)
        print("Recall:", recall)
        print("F-score", fscore)

        # computing Weighted Complexity Score (WSC)
        wsc = sum([len(rule[1]) for rule in self.list_rules])

        print("# Rules:", len(self.list_rules))
        print("WSC:", wsc)

    def policy_refinement(self, th_rule_sim):

        print("\n#############################")
        print(" PHASE 5: Policy Refinement.")
        print("#############################\n")

        # Nombre del archivo de salida
        archivo_salida = "reglas_abac.txt"
        df_fn = pd.DataFrame(self.fn_logs)

        ###### ***** TASK 1 ***** #####
        # Access request bipartite network
        self.bip_network_ref = build_network_model(df_fn, 'uname', 'RID')
        print("TASK 1: Done!\n")

        ###### ***** TASK 2 ***** #####
        # User network 3
        self.user_network_ref = bipartite_projection(self.bip_network_ref, 0)

        ###### ***** TASK 3 ***** #####
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

        ###### ***** TASK 4 ***** #####
        # Community calssification
        n_res_in_comms = [len(i[1]) for i in dict_commts_ref.values()]
        max_n_res = max(n_res_in_comms)

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

        politica_final = combinar_politicas_unicas(self.list_rules, self.list_rules_ref)
        reglas_convertidas = convertir_a_nativo(politica_final)

        with open(archivo_salida, 'w', encoding='utf-8') as f:
            for regla in reglas_convertidas:
                f.write(json.dumps(regla) + '\n')

        print(f"Se han guardado {len(reglas_convertidas)} reglas en {archivo_salida}")

        ###### ***** TASK 2 ***** #####
        # Rule Network
        self.rules_with_idx = {}
        self.total_rules = self.list_rules + self.list_rules_ref
        self.rules_coverage = self.compute_rules_coverage(self.total_rules)

        df_cov = self.save_rules_coverage(self.rules_coverage)

        self.plot_coverage_distribution(df_cov)

        #print("\nRule coverage:")
        #for r in self.rules_coverage:
        #    print("Rule", r["rule_id"], "covers", r["coverage"], "records")
        
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

        nx.write_gml(self.rule_network_ref, "rule_networks.gml")

        print("Rule Network\n", self.rule_network_ref)
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


        self.rule_usage_counter = {i: 0 for i in range(len(self.total_rules))}

        ###### ***** EVALUATION ***** ##########################
        df_users = pd.DataFrame(
            self.df_data_pos[self.user_attrs+["uname"]].drop_duplicates())

        users_ids_list = list(df_users.uname)
        users_attrs_list = list(df_users[self.user_attrs].values)
        users_attrs_list = [str(i).replace(" ", "") for i in users_attrs_list]
        users_attrs_dict = dict(zip(users_ids_list, users_attrs_list))

        df_resources = pd.DataFrame(
            self.df_data_pos[self.resource_attrs+["RID"]].drop_duplicates())

        res_ids_list = list(df_resources.RID)
        res_attrs_list = list(df_resources[self.resource_attrs].values)
        res_attrs_list = [str(i).replace(" ", "") for i in res_attrs_list]
        res_attrs_dict = dict(zip(res_ids_list, res_attrs_list))

        df_test_k_pos = self.df_data_pos[self.df_data_pos.ACTION == 1]
        df_test_k_neg = self.df_data_neg[self.df_data_neg.ACTION == 0]

        self.rule_index = {
            tuple(map(tuple, rule[1])): i
            for i, rule in enumerate(self.total_rules)
        }

        self.fn_logs, rule_usage_counter = get_FN_logs_dos(
            self.df_data_pos,
            copy_g_proj2,
            self.total_rules,
            self.rule_network_ref,
            self.rules_with_idx,
            self.rule_usage_counter,
            self.rule_index
        )
        
        rule_usage_data = []
        for rule_id, count in self.rule_usage_counter.items():
            rule_usage_data.append({
                "rule_id": rule_id,
                "granted_accesses": count,
                "num_conditions": len(self.total_rules[rule_id][1])
            })

        df_usage = pd.DataFrame(rule_usage_data)
        df_usage = df_usage.sort_values("granted_accesses", ascending=False).reset_index(drop=True)

        attr_counter = Counter()

        for rule in self.total_rules:
            for attr_val in rule[1]:
                attr_counter[tuple(attr_val)] += 1
        total_rules = len(self.total_rules)

        rules_output = []

        for i, row in df_usage.iterrows():

            rule_id = int(row["rule_id"])
            rule = self.total_rules[rule_id]

            conditions = rule[1]
            num_conditions = len(conditions)
            coverage = row["granted_accesses"]

            conds_with_importance = []
            total_importance = 0

            for attr_val in conditions:
                freq = attr_counter[tuple(attr_val)]
                importance = freq / total_rules

                total_importance += importance

                conds_with_importance.append([
                    attr_val,
                    round(importance, 4)
                ])

            rules_output.append({
                "rule_id": rule_id,
                "coverage": coverage,
                "num_conditions": num_conditions,
                "total_importance": round(total_importance, 4),
                "conditions_importance": str(conds_with_importance)
            })

        df_rules_analysis = pd.DataFrame(rules_output)
        df_rules_analysis.to_csv("rules_analysis.csv", index=False)
        print("Saved rules_analysis.csv")

        df_usage.to_csv("rule_usage.csv", index=False)

        plt.figure(figsize=(12,6))

        plt.bar(df_usage["rule_id"], df_usage["granted_accesses"])

        plt.xlabel("Rule ID")
        plt.ylabel("Number of Granted Accesses")
        plt.title("Number of Accesses Granted per Rule")

        plt.grid(True, axis="y")

        plt.savefig("rule_usage_histogram.png")

        plt.close()

        print("Histogram saved to rule_usage_histogram.png")

        self.fp_logs, rules_to_fix = get_FP_logs(
            self.df_data_neg, self.user_network, self.total_rules,
            self.rule_network_ref, self.rules_with_idx)

        neg_rules = generate_negative_rules(
            pd.DataFrame(self.fp_logs), rules_to_fix, len(self.list_rules))
        
        print("Reglas NEGATIVAS")
        print(neg_rules)
        print()

        self.fp_logs = get_FP_logs_ref(
            self.df_data_neg, self.user_network, self.total_rules,
            self.rule_network_ref, self.rules_with_idx, neg_rules)

        TP = len(self.df_data_pos) - len(self.fn_logs)
        TN = len(self.df_data_neg) - len(self.fp_logs)

        precision = TP / (TP + len(self.fp_logs))

        recall = TP / (TP + len(self.fn_logs))

        fscore = 2*(precision*recall)/(precision+recall)

        print("FN:", len(self.fn_logs),
              " - {:.2f}%".format((len(self.fn_logs)/len(self.df_data_pos))*100))
        print("FP:", len(self.fp_logs),
              " - {:.2f}%".format((len(self.fp_logs)/len(self.df_data_neg))*100))
        print("Precision:", precision)
        print("Recall:", recall)
        print("F-score", fscore)

        wsc = sum([len(rule[1]) for rule in self.total_rules])

        print("# Rules:", len(self.total_rules))
        print("WSC:", wsc)
        return [precision, recall, fscore, wsc, len(self.total_rules)]