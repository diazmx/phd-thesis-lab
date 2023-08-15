import networkx as nx
import sys
from auxiliar_functions.data_preprocessing import add_new_index
from auxiliar_functions.network_model import build_network_model, bipartite_projection, plot_distribution_degree
from auxiliar_functions.community_detection import sub_community_detection, add_type_commts

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

            self.df_train_k = self.df_train_k[['ACTION'] + self.user_attrs +
                                              self.resource_attrs]
            # Change string values to numerical
            mapping = {'system': 10101, 'human': 10201,
                       'human and system': 10301}  # Control
            self.df_train_k["control"] = self.df_train_k["control"].replace(
                mapping)
            mapping = {'system': 20102, 'human': 20202}  # monitoring
            self.df_train_k["monitoring"] = self.df_train_k["monitoring"].replace(
                mapping)
            mapping = {'system': 30103, 'human': 30203}  # fallbacj
            self.df_train_k["fallback"] = self.df_train_k["fallback"].replace(
                mapping)
            mapping = {0: 40004, 1: 40104, 2: 40204,
                       3: 40304, 4: 40404, 5: 40504}
            self.df_train_k["driving_task_loa"] = self.df_train_k["driving_task_loa"].replace(
                mapping)
            mapping = {0: 50005, 1: 50105, 2: 50205,
                       3: 50305, 4: 50405, 5: 50505}
            self.df_train_k["vehicle_loa"] = self.df_train_k["vehicle_loa"].replace(
                mapping)
            mapping = {0: 60006, 1: 60106, 2: 60206,
                       3: 60306, 4: 60406, 5: 60506}
            self.df_train_k["region_loa"] = self.df_train_k["region_loa"].replace(
                mapping)

            self.df_train_k = add_new_index(
                self.df_train_k, self.user_attrs, type="U")
            self.df_train_k = add_new_index(
                self.df_train_k, self.resource_attrs, type="R")
            print("\nTASK 1: Done!\n")  #

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

        else:
            print("Invalid dataset:", self.name_ds)
            sys.exit()

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
        avg_degree = sum(self.user_network.degree()) / \
            self.user_network.vcount()
        print("\nNetwork Analysis")
        print("- Avg. degree", "{:.4f}".format(avg_degree))

        print("- Density:", "{:.4f}".format(self.user_network.density()))

        cc = self.user_network.transitivity_avglocal_undirected()
        print("- Clustering Coefficient:", "{:.4f}".format(cc))

        L = self.user_network.average_path_length()
        print("- Average Path Length :", "{:.4f}".format(L))

        plot_distribution_degree(self.user_network, self.name_ds)
        print("TASK 2: Done!\n")

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
        n_res_in_comms = [len(i[1]) for i in dict_commts.values()]
        max_n_res = max(n_res_in_comms)
        # print("Comunidad con # mayor recursos", max_n_res)

        # Sparse and Med Thresholds
        big_threshold = int(0.50 * max_n_res)
        med_threshold = int(0.25 * max_n_res)
        print("Big Threshold: ", big_threshold,
              " \t\t Med Threshold", med_threshold)

        s_commts, m_commts, c_commts = add_type_commts(self.user_network, dict_commts,
                                                       big_threshold, med_threshold)
        self.all_commts = s_commts + m_commts + c_commts
        print("TASK 2: Done!\n")
