import networkx as nx
import igraph as ig
import matplotlib.pyplot as plt
import random
import copy
from itertools import combinations
from collections import Counter, defaultdict
import sys
from auxiliar_functions.data_preprocessing import add_new_index
from auxiliar_functions.network_model import build_network_model, bipartite_projection, plot_distribution_degree
from auxiliar_functions.community_detection import sub_community_detection, add_type_commts, sub_community_detection_nx, add_type_commts_nx
from auxiliar_functions.rule_inference import frequent_resources, get_attrs_from_user_sig, get_attrs_from_user, get_attrs_from_res, attribute_value_common, evaluate_weight
from auxiliar_functions.evaluation import get_FN_logs, get_FP_logs, get_FP_logs_ref, get_FN_logs_ref
from auxiliar_functions.refinement import generate_negative_rules
from sklearn.model_selection import StratifiedShuffleSplit

import numpy as np
import pandas as pd
pd.options.mode.chained_assignment = None

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
            mapping = {"addnote": 0, "none": 0,
                       "additem": 0, "read": 0, "1": 1}
            self.df_data["ACTION"] = self.df_data["ACTION"].replace(
                mapping)

            mapping = {"none": 10, "doctor": 11, "nurse": 12}  # role
            self.df_data["role"] = self.df_data["role"].replace(mapping)

            mapping = {"note": 110, "cardiology": 111, "nursing": 112,
                       "oncology": 113, "none": 114}  # speacialty
            self.df_data["specialty"] = self.df_data["specialty"].replace(
                mapping)

            mapping = {"oncteam1": 1101, "carteam1": 1111,
                       "carteam2": 1121, "oncteam2": 1131, "none": 1141}  # tem
            self.df_data["team"] = self.df_data["team"].replace(mapping)


            mapping = {"carward": 11011,
                       "oncward": 11111, "none": 11211}  # uward
            self.df_data["uward"] = self.df_data["uward"].replace(
                mapping)

            mapping = {"oncpat1": 111011, "carpat1": 111111,  # agentfor
                       "oncpat2": 111211, "carpat2": 111311, "none": 111411}
            self.df_data["agentfor"] = self.df_data["agentfor"].replace(
                mapping)


            mapping = {"hr": 1110111, "hritem": 1111111,
                       "none": 1112111}  # type
            self.df_data["type"] = self.df_data["type"].replace(mapping)

            mapping = {"oncpat1": 211012, "carpat1": 211112,  # patient
                       "oncpat2": 211212, "carpat2": 211312, "none": 211412}
            self.df_data["patient"] = self.df_data["patient"].replace(
                mapping)

            mapping = {"oncteam1": 2102, "carteam1": 2112,
                       "carteam2": 2122, "oncteam2": 2132, "none": 2142}  # treatingteam
            self.df_data["treatingteam"] = self.df_data["treatingteam"].replace(
                mapping)

            mapping = {"carward": 21012,
                       "oncward": 21112, "none": 21212}  # oward
            self.df_data["oward"] = self.df_data["oward"].replace(
                mapping)

            mapping = {"note": 210, "cardiology": 211,
                       "nursing": 212, "oncology": 213, "none": 214}  # topic
            self.df_data["topic"] = self.df_data["topic"].replace(
                mapping)

            mapping = {"oncdoc2": 11110111, "carnurse1": 11111111, "oncnurse2": 11112111,  # author
                       "carnurse2": 11113111, "oncdoc1": 11114111, "oncnurse1": 11115111, "none": 11116111}
            self.df_data["author"] = self.df_data["author"].replace(
                mapping)

            self.df_data = self.df_data.drop(['user'], axis=1)
            self.df_data = add_new_index(
                self.df_data, self.user_attrs, type="U")
            self.df_data = add_new_index(
                self.df_data, self.resource_attrs, type="R")
            print("\nTASK 1: Done!\n")  # Not applicable

            ###### ***** TASK 2 ***** #####
            # Converting continuous values to categorical values.
            print("TASK 2: Done!\n")  # Not applicable

            ###### ***** TASK 3 ***** #####
            # Removing duplicated access requests.
            self.df_data_pos = self.df_data[self.df_data.ACTION == 1]
            self.df_data_neg = self.df_data[self.df_data.ACTION == 0]

            self.df_data_pos = self.df_data_pos[self.df_data_pos.columns[1:]].drop_duplicates(
            )
            self.df_data_neg = self.df_data_neg[self.df_data_neg.columns[1:]].drop_duplicates(
            )
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
            self.df_data_neg = self.df_data_neg[self.df_data_neg.columns[1:]].drop_duplicates(
            )
            print("# (+) access requests:", len(self.df_data_pos),
                  " %: {:.2f}".format((len(self.df_data_pos)/len(self.df_data))*100))
            print("# (-) access requests:", len(self.df_data_neg),
                  " %: {:.2f}".format((len(self.df_data_neg)/len(self.df_data))*100))
            print("TASK 3: Done!\n")

        else:
            print("Invalid dataset:", self.name_ds)
            sys.exit()

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
 
    def generate_random_network(self):
        print("\n########################################")
        print(" PHASE 2B: Random Network for Comparison")
        print("########################################\n")

        # Obtener el grado de cada nodo
        degree_sequence = self.user_network.degree()

        # Generar red aleatoria con el mismo grado por nodo
        random_graph = ig.Graph.Degree_Sequence(degree_sequence, method="vl")

        

        # Análisis de red aleatoria
        #avg_degree = sum(random_graph.degree()) / random_graph.vcount()
        #print("- Avg. degree", "{:.4f}".format(avg_degree))
        #print("- Density:", "{:.4f}".format(random_graph.density()))
        #cc = random_graph.transitivity_avglocal_undirected()
        #print("- Clustering Coefficient:", "{:.4f}".format(cc))
        #L = random_graph.average_path_length()
        #print("- Average Path Length :", "{:.4f}".format(L))

        # Guardar red aleatoria para comparaciones posteriores

        random_weights = np.random.rand(random_graph.ecount())
        random_graph.es['weight'] = random_weights

        # Copiar atributos de nodos
        for attr in self.user_network.vs.attributes():
            random_graph.vs[attr] = self.user_network.vs[attr]
        self.random_user_network = random_graph

        print("Random Network Generation: Done!\n")


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

    def community_detection_random(self, big_threshold_ratio=0.5, med_threshold_ratio=0.25):
        print("\n###############################")
        print(" PHASE 3B: Community Detection RANDOM.")
        print("###############################\n")

        ###### ***** TASK 1 ***** #####
        # Community detection
        partition = self.random_user_network.community_multilevel(
            weights=self.random_user_network.es()["weight"])

        # Modularity score
        print("Modularity: %.4f" % partition.modularity)

        # Add cluster attribute
        self.random_user_network.vs["commty"] = partition.membership

        print(self.random_user_network.summary())
        print(partition.summary())
        print("TASK 1: Done!\n")

        dict_commts = sub_community_detection(self.random_user_network, 0.5, None)

        ###### ***** TASK 2 ***** #####
        # Community classification
        n_res_in_comms = [len(i[1]) for i in dict_commts.values()]
        max_n_res = max(n_res_in_comms)

        # Thresholds based on user-defined ratios
        big_threshold = int(big_threshold_ratio * max_n_res)
        med_threshold = int(med_threshold_ratio * max_n_res)
        print("Big Threshold:", big_threshold, "\t\tMed Threshold:", med_threshold)

        s_commts, m_commts, c_commts = add_type_commts(
            self.random_user_network, dict_commts, big_threshold, med_threshold)

        self.all_commts_random = s_commts + m_commts + c_commts
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
                            logs_with_resource = self.df_data_pos[self.df_data_pos["RID"]
                                                                     == sig_resource].iloc[0]
                            rule_i[1].append([attr, logs_with_resource[attr]])

                        # Atributos frecuentes en usuarios
                        df_users_commty = get_attrs_from_user_sig(
                            commty_[1][0], self.df_data_pos, self.user_attrs, sig_resource, self.bip_network)

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

        print("Rule Network\n", self.rule_network)
        print("TASK 2: Done!\n")

    def rule_inference_random(self, th_rule_sim):
        print("\n##########################")
        print(" PHASE 4B: Rule Inference RANDOM.")
        print("##########################\n")

        ###### ***** TASK 1 ***** #####
        # Rule extraction
        self.list_rules_random = []  # Lista de reglas
        th_lfr = 0.2
        th_sr = 0.5

        for commty_ in self.all_commts_random:
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

                        rule_i[1] = rule_i[1] + rule_user_attrs
                        self.list_rules_random.append(rule_i)

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
            self.list_rules_random.append(rule_i)
        print("|R|:", len(self.list_rules_random))
        print("TASK 1: Done!\n")

        ###### ***** TASK 2 ***** #####
        # Rule Network
        self.rules_with_idx_random = {}
        for idx, rule in enumerate(self.list_rules_random):
            self.rules_with_idx_random[idx] = rule

        # Create the graph
        edge_list = []
        for idxA in range(len(self.list_rules_random)):
            for idxB in range(idxA, len(self.list_rules_random)):
                edge_temp = evaluate_weight(
                    self.rules_with_idx_random[idxA][1], self.rules_with_idx_random[idxB][1],
                    th_rule_sim)
                if edge_temp != -1:
                    edge_list.append((idxA, idxB, edge_temp))

        self.rule_network_random = nx.Graph()
        self.rule_network_random.add_weighted_edges_from(edge_list)

        # Add rule attribute
        self.rule_network_random.remove_edges_from(
            nx.selfloop_edges(self.rule_network_random))

        # print(self.rule_network.nodes, len(self.rule_network.nodes))
        # print(self.rules_with_idx.keys(), len(self.rules_with_idx.keys()))
        isolated_nodes = [
            i for i in self.rules_with_idx.keys() if i not in self.rule_network_random.nodes]
        # print(isolated_nodes, len(isolated_nodes))
        self.rule_network_random.add_nodes_from(isolated_nodes)

        print("Rule Network\n", self.rule_network_random)
        print("TASK 2: Done!\n")


    def experiment_bridge_node_reassignment(self,
                                        ratio_bridge_nodes=0.20,
                                        iterations=30,
                                        stability_threshold=0.40):
        """
        Experimento:
        1. Detecta nodos puente con estabilidad Jaccard baja.
        2. Elige un porcentaje aleatorio de esos nodos.
        3. Los reasigna a otra comunidad posible.
        4. Corre todo el pipeline de nuevo y compara reglas.
        """

        print("\n============================")
        print(" EXPERIMENTO: Nodos Puente")
        print("============================\n")

        g = self.user_network

        print("→ Ejecutando múltiples Louvain para identificar nodos inestables...")
        neigh = run_louvain_iterations(g, iterations=10)
        stab = compute_jaccard_stability(neigh)

        # Ordenar por estabilidad
        sorted_stab = sorted(stab.items(), key=lambda x: x[1])

        # Consideramos nodos puente = 20% inferiores o con estabilidad < threshold
        bridge_nodes = [node for node, s in sorted_stab if s <= stability_threshold]

        if len(bridge_nodes) == 0:
            print("⚠️ No se detectaron nodos puente según el threshold.")
            return

        print(f"→ Nodos puente detectados (estabilidad ≤ {stability_threshold}): {len(bridge_nodes)}")

        # ITERACIONES EXPERIMENTALES
        for it in range(iterations):
            print(f"\n------ Iteración #{it+1}/{iterations} ------")

            # Seleccionar subset aleatorio según ratio
            k = max(1, int(len(bridge_nodes) * ratio_bridge_nodes))
            selected = random.sample(bridge_nodes, k)

            print(f"→ Seleccionando {k}/{len(bridge_nodes)} nodos puente para mover:")
            print(selected)

            # Detección preliminar de comunidades base
            base_partition = g.community_multilevel()
            base_membership = base_partition.membership
            n_comms = max(base_membership) + 1

            print(f"→ Comunidades detectadas originalmente: {n_comms}")

            # Copia del membership
            new_membership = base_membership.copy()

            # Reasignar cada nodo aleatoriamente
            for node_id in selected:
                idx = g.vs.find(id=node_id).index
                current_comm = new_membership[idx]

                # Opciones: cualquier comunidad excepto su comunidad actual
                possible = list(set(range(n_comms)) - {current_comm})
                new_comm = random.choice(possible)

                new_membership[idx] = new_comm

                print(f"   - Nodo {node_id}: {current_comm} → {new_comm}")

            # Asignar comunidades nuevas al grafo
            g.vs["commty"] = new_membership

            # Re-ejecutar reglas
            print("→ Re-ejecutando rule inference con comunidades modificadas...")
            rules_mod = self.rule_inference(return_rules=True)

            # Volver a baseline para comparar
            g.vs["commty"] = base_membership
            rules_base = self.rule_inference(return_rules=True)

            # COMPARACIÓN
            diff_added = len(rules_mod - rules_base)
            diff_removed = len(rules_base - rules_mod)

            print("\n   >>> RESULTADOS DE LA ITERACIÓN <<<")
            print(f"   - Reglas nuevas que NO estaban antes: {diff_added}")
            print(f"   - Reglas eliminadas respecto a baseline: {diff_removed}")

        print("\n=====================================")
        print(" Experimento terminado.")
        print("=====================================")



    def evaluation(self):
        print("\n#############")
        print(" Evaluation.")
        print("#############\n")

        ###### ***** TASK 1 ***** #####
        # Get False Negative Set
        self.fn_logs = get_FN_logs(
            self.df_data_pos, self.user_network, self.list_rules,
            self.rule_network, self.rules_with_idx)

        # Get False Positive Set
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

    def evaluation_random(self):
        print("\n#############")
        print(" Evaluation RANDOM.")
        print("#############\n")

        ###### ***** TASK 1 ***** #####
        # Get False Negative Set
        self.fn_logs_random = get_FN_logs(
            self.df_data_pos, self.random_user_network, self.list_rules_random,
            self.rule_network_random, self.rules_with_idx_random)

        # Get False Positive Set
        self.fp_logs_random = get_FP_logs(
            self.df_data_neg, self.random_user_network, self.list_rules_random,
            self.rule_network_random, self.rules_with_idx_random)

        TP = len(self.df_data_pos) - len(self.fn_logs_random)
        TN = len(self.df_data_neg) - len(self.fp_logs_random)

        precision = TP / (TP + len(self.fp_logs_random))

        recall = TP / (TP + len(self.fn_logs_random))

        fscore = 2*(precision*recall)/(precision+recall)

        print("FN:", len(self.fn_logs_random),
              " - {:.2f}%".format((len(self.fn_logs_random)/len(self.df_data_pos))*100))
        print("FP:", len(self.fp_logs_random),
              " - {:.2f}%".format((len(self.fp_logs_random)/len(self.df_data_neg))*100))
        print("Precision:", precision)
        print("Recall:", recall)
        print("F-score", fscore)

        # computing Weighted Complexity Score (WSC)
        wsc = sum([len(rule[1]) for rule in self.list_rules_random])

        print("# Rules:", len(self.list_rules_random))
        print("WSC:", wsc)

    def policy_refinement(self, th_rule_sim):

        print("\n#############################")
        print(" PHASE 5: Policy Refinement.")
        print("#############################\n")

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

        self.fn_logs = get_FN_logs(self.df_data_pos, copy_g_proj2,
                                   self.total_rules, self.rule_network_ref, self.rules_with_idx)

        self.fp_logs, rules_to_fix = get_FP_logs(
            self.df_data_neg, self.user_network, self.total_rules,
            self.rule_network_ref, self.rules_with_idx)

        neg_rules = generate_negative_rules(
            pd.DataFrame(self.fp_logs), rules_to_fix, len(self.list_rules))

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
    

    def policy_refinement_random(self, th_rule_sim):

        print("\n#############################")
        print(" PHASE 5B: Policy Refinement Random.")
        print("#############################\n")

        df_fn = pd.DataFrame(self.fn_logs_random)

        ###### ***** TASK 1 ***** #####
        # Access request bipartite network
        self.bip_network_ref = build_network_model(df_fn, 'uname', 'RID')
        print("TASK 1: Done!\n")

        ###### ***** TASK 2 ***** #####
        # User network 3
        self.user_network_ref_random = bipartite_projection(self.bip_network_ref, 0)

        ###### ***** TASK 3 ***** #####
        # Community detection
        partition_ref = self.user_network_ref_random.community_multilevel(
            weights=self.user_network_ref_random.es()["weight"])

        # Modularity score
        print("Modularity: %.4f" % partition_ref.modularity)

        # Add cluster attribute
        self.user_network_ref_random.vs["commty"] = partition_ref.membership

        print(self.user_network_ref_random.summary())
        print(partition_ref.summary())
        print("TASK 1: Done!\n")

        dict_commts_ref = sub_community_detection(
            self.user_network_ref_random, 0.5, len(self.rule_network))

        ###### ***** TASK 4 ***** #####
        # Community calssification
        n_res_in_comms = [len(i[1]) for i in dict_commts_ref.values()]
        max_n_res = max(n_res_in_comms)

        big_threshold = int(0.50 * max_n_res)
        med_threshold = int(0.25 * max_n_res)
        print("Big Threshold: ", big_threshold,
              " \t\t Med Threshold", med_threshold)

        s_commts, m_commts, c_commts = add_type_commts(self.user_network_ref_random, dict_commts_ref,
                                                       big_threshold, med_threshold)
        self.all_commts_ref_random = s_commts + m_commts + c_commts
        print("TASK 2: Done!\n")

        ###### ***** TASK 1 ***** #####
        # Rule extraction
        self.list_rules_ref_random = []  # Lista de reglas
        th_lfr = 0.2
        th_sr = 0.5

        for commty_ in self.all_commts_ref_random:
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
                        self.list_rules_ref_random.append(rule_i)

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
            self.list_rules_ref_random.append(rule_i)
        print("|R|:", len(self.list_rules_ref_random))
        print("TASK 1: Done!\n")

        ###### ***** TASK 2 ***** #####
        # Rule Network
        self.rules_with_idx_random = {}
        self.total_rules_random = self.list_rules_random + self.list_rules_ref_random
        for idx, rule in enumerate(self.total_rules_random):
            self.rules_with_idx_random[idx] = rule

        # Create the graph
        edge_list = []
        for idxA in range(len(self.total_rules_random)):
            for idxB in range(idxA, len(self.total_rules_random)):
                edge_temp = evaluate_weight(
                    self.rules_with_idx_random[idxA][1], self.rules_with_idx_random[idxB][1],
                    th_rule_sim)
                if edge_temp != -1:
                    edge_list.append((idxA, idxB, edge_temp))

        self.rule_network_ref_random = nx.Graph()
        self.rule_network_ref_random.add_weighted_edges_from(edge_list)

        # Add rule attribute
        self.rule_network_ref_random.remove_edges_from(
            nx.selfloop_edges(self.rule_network_ref_random))

        isolated_nodes = [
            i for i in self.rules_with_idx.keys() if i not in self.rule_network_ref_random.nodes]
        # print(isolated_nodes, len(isolated_nodes))
        self.rule_network_ref_random.add_nodes_from(isolated_nodes)

        print("Rule Network\n", self.rule_network_ref_random)
        print("TASK 2: Done!\n")

        copy_g_proj2 = self.random_user_network.copy()  # Copia del grafo
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

        self.fn_logs_random = get_FN_logs(self.df_data_pos, copy_g_proj2,
                                   self.total_rules_random, self.rule_network_ref_random, self.rules_with_idx_random)

        self.fp_logs_random, rules_to_fix = get_FP_logs(
            self.df_data_neg, self.random_user_network, self.total_rules_random,
            self.rule_network_ref_random, self.rules_with_idx_random)

        neg_rules = generate_negative_rules(
            pd.DataFrame(self.fp_logs_random), rules_to_fix, len(self.list_rules_random))

        self.fp_logs_random = get_FP_logs_ref(
            self.df_data_neg, self.random_user_network, self.total_rules_random,
            self.rule_network_ref_random, self.rules_with_idx_random, neg_rules)

        TP = len(self.df_data_pos) - len(self.fn_logs_random)
        TN = len(self.df_data_neg) - len(self.fp_logs_random)

        precision = TP / (TP + len(self.fp_logs_random))

        recall = TP / (TP + len(self.fn_logs_random))

        fscore = 2*(precision*recall)/(precision+recall)

        print("FN:", len(self.fn_logs_random),
              " - {:.2f}%".format((len(self.fn_logs_random)/len(self.df_data_pos))*100))
        print("FP:", len(self.fp_logs_random),
              " - {:.2f}%".format((len(self.fp_logs_random)/len(self.df_data_neg))*100))
        print("Precision:", precision)
        print("Recall:", recall)
        print("F-score", fscore)

        wsc = sum([len(rule[1]) for rule in self.total_rules_random])

        print("# Rules:", len(self.total_rules_random))
        print("WSC:", wsc)
        return [precision, recall, fscore, wsc, len(self.total_rules_random)]

# Los datasets noisy son aquellos en donde de forma aleatoria se modifican el 10% de las solicitudes.
def introducir_ruido(df, columna='ACTION', porcentaje=0.1, random_state=None):
    """
    Cambia un porcentaje de valores en la columna 'ACTION' de 0 a 1 y de 1 a 0.
    Mantiene la proporción original de aparición de 0s y 1s.

    Parámetros:
    - df: DataFrame original
    - columna: nombre de la columna binaria
    - porcentaje: proporción de valores a modificar (entre 0 y 1)
    - random_state: semilla para reproducibilidad

    Retorna:
    - df_modificado: copia del DataFrame con ruido introducido
    """
    np.random.seed(random_state)
    df_modificado = df.copy()

    # Índices de valores 0 y 1
    indices_0 = df_modificado[df_modificado[columna] == 0].index
    indices_1 = df_modificado[df_modificado[columna] == 1].index

    # Cantidad a modificar
    n_0 = int(len(indices_0) * porcentaje)
    n_1 = int(len(indices_1) * porcentaje)

    # Selección aleatoria de índices para cambiar
    indices_0_a_cambiar = np.random.choice(indices_0, size=n_0, replace=False)
    indices_1_a_cambiar = np.random.choice(indices_1, size=n_1, replace=False)

    # Aplicar cambios
    df_modificado.loc[indices_0_a_cambiar, columna] = 1
    df_modificado.loc[indices_1_a_cambiar, columna] = 0
    print("Se ingresó ruido")

    return df_modificado

def crear_version_sparse(df, columna='ACTION', porcentaje=0.9, random_state=None):
    """
    Crea una versión 'sparse' del DataFrame seleccionando un porcentaje de registros
    manteniendo la proporción original de la columna binaria 'ACTION'.

    Parámetros:
    - df: DataFrame original
    - columna: nombre de la columna binaria
    - porcentaje: proporción de registros a conservar (entre 0 y 1)
    - random_state: semilla para reproducibilidad

    Retorna:
    - df_sparse: DataFrame reducido con proporción original de la columna
    """
    np.random.seed(random_state)
    
    # Separar por clase
    df_0 = df[df[columna] == 0]
    df_1 = df[df[columna] == 1]

    # Calcular cuántos registros conservar de cada clase
    n_0 = int(len(df_0) * porcentaje)
    n_1 = int(len(df_1) * porcentaje)

    # Seleccionar aleatoriamente
    df_0_sparse = df_0.sample(n=n_0, random_state=random_state)
    df_1_sparse = df_1.sample(n=n_1, random_state=random_state)

    # Combinar y reordenar
    df_sparse = pd.concat([df_0_sparse, df_1_sparse]).sample(frac=1, random_state=random_state).reset_index(drop=True)
    print("SE redujo dataset")
    return df_sparse
