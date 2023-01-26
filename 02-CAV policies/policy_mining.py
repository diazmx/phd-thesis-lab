# import libraries
import numpy as np
import pandas as pd
import igraph as ig
import random
import networkx as nx
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold, StratifiedShuffleSplit
from apriori_python import apriori
from auxiliar_funcs import *

# Load data
url_file = '../00-Data/cav_policies.csv'
cav_data = pd.read_csv(url_file)

# Get a smaller sample: 15K positive and 15k negative.
# cav_data = cav_data.groupby('result').sample(n=15000)
print("Columns: ", cav_data.columns)
print("Lenght: ", len(cav_data)); print()

user_attr = ['control', 'monitoring', 'fallback', 'weather', 'visibility', 
        'traffic_congestion']
#user_attr = ['control', 'monitoring', 'fallback']
rsrc_attr = ['driving_task_loa', 'vehicle_loa', 'region_loa']
cav_data = cav_data[user_attr + rsrc_attr + ['result']]

# Change string values to numerical
mapping = {'system': 10101, 'human': 10201, 'human and system': 10301} # Control
cav_data.control = cav_data.control.replace(mapping)

mapping = {'system': 20102, 'human': 20202} # monitoring
cav_data.monitoring = cav_data.monitoring.replace(mapping)

mapping = {'system': 30103, 'human': 30203} # fallbacj
cav_data.fallback = cav_data.fallback.replace(mapping)

mapping = {0: 40004, 1: 40104, 2: 40204, 3: 40304, 4: 40404, 5: 40504}
cav_data.driving_task_loa = cav_data.driving_task_loa.replace(mapping)

mapping = {0: 50005, 1: 50105, 2: 50205, 3: 50305, 4: 50405, 5: 50505}
cav_data.vehicle_loa = cav_data.vehicle_loa.replace(mapping)

mapping = {0: 60006, 1: 60106, 2: 60206, 3: 60306, 4: 60406, 5: 60506}
cav_data.region_loa = cav_data.region_loa.replace(mapping)


print("# User attr:", len(user_attr))
print("# Rsrc attr:", len(rsrc_attr)); print()

# Data statictics:
n_users = len(cav_data[user_attr].drop_duplicates())
n_rscrc = len(cav_data[rsrc_attr].drop_duplicates())
print("|U| =", n_users)
print("|R| =", n_rscrc); print()

# Add user and resource id columns
user_dict = get_user_res(cav_data, user_attr, True)
rsrc_dict = get_user_res(cav_data, rsrc_attr, False)
cav_data = add_col(cav_data, user_dict, user_attr, "USRID")
cav_data = add_col(cav_data, rsrc_dict, rsrc_attr, "RESID")

# Accepted and rejected requests
cav_pos = cav_data[cav_data.result == 'approved']
cav_neg = cav_data[cav_data.result == 'rejected']
print("|L+| =", len(cav_pos), "{:.2f}%".format((len(cav_pos) 
        / len(cav_data) ) * 100))
print("|L-| =", len(cav_neg), "{:.2f}%".format((len(cav_neg) 
        / len(cav_data) ) * 100))


###### ****** Cross validation ***** #####
k = 10
test_size = 0.2
kfold = StratifiedShuffleSplit(n_splits=k, test_size=test_size, random_state=1)

data_partition = kfold.split(cav_data, cav_data.result)
data_curpus = [] # A list to storage the k folds

for train_data, test_data in data_partition:
    X_train, X_test = cav_data.iloc[train_data], cav_data.iloc[test_data]
    data_curpus.append([X_train, X_test])

print("Done!")
print(" - k =", k)
print(" - Train-Test size: ", len(data_curpus[0][0]), "(", 
    (1-test_size)*100, ") \t", len(data_curpus[0][1]), "(", test_size*100, ")")


###### ****** DATA PREPROCESSING ***** #####
id_kfold = 2

cav_train, cav_test = data_curpus[id_kfold][0], data_curpus[id_kfold][1]
print("# Train access request =", len(cav_train), "{:.2f}%".format(
    len(cav_train)/(len(cav_train)+len(cav_test))*100))
print("# Train access request =", len(cav_test), "{:.2f}%".format(
    len(cav_train)/(len(cav_train)+len(cav_test))*100))
print("Total =", len(cav_train)+len(cav_test)); print()

#### **** SELECT FUNCTIONAL ATTRIBUTES **** ####
cav_train = cav_train[user_attr + rsrc_attr + ['USRID', 'RESID', 'result']]
cav_test = cav_test[user_attr + rsrc_attr + ['USRID', 'RESID', 'result']]

##### Task 1: Null and uknwokn values #####
print("TASK 1: Done!"); print() # NA


##### TASK 2: convert continuous values to categorical values #####
print("TASK 2: Done!"); print() # NA 

##### TASK 3: Drop duplicates access requests #####
print("TASK 3: Drop duplicates access requests")

positive_cav_train = cav_train[cav_train.result=='approved']
positive_cav_test = cav_test[cav_test.result=='approved']
negative_cav_train = cav_train[cav_train.result=='rejected']
negative_cav_test = cav_test[cav_test.result=='rejected']

print(" -TRAIN DATA: Removing", 
    len(positive_cav_train.drop_duplicates()) - 
    len(positive_cav_train), "positive access requests")
print(" -TRAIN DATA: Removing", 
    len(negative_cav_train.drop_duplicates()) - 
    len(negative_cav_train), "negative access requests")
print(" -TEST DATA: Removing", 
    len(positive_cav_test.drop_duplicates()) - 
    len(positive_cav_test), "positive access requests")
print(" -TEST DATA: Removing", 
    len(negative_cav_test.drop_duplicates()) - 
    len(negative_cav_test), "negative access requests")

# Filter resources
n1 = 0
n2 = 84
top_list = positive_cav_train.RESID.value_counts()[:len(positive_cav_train.RESID.drop_duplicates())].index.tolist()
# Filter the interval between n1 and n2
top_list = top_list[n1:n2+1]
print('#Filtered resources:', len(top_list))

boolean_series = positive_cav_train.RESID.isin(top_list)
positive_cav_train = positive_cav_train[boolean_series]
#bolean_series = negative_cav_train.RESID.isin(top_list)
#negative_cav_train = negative_cav_train[bolean_series]
print("Filtered resource Done!")

###### ****** Network model ***** #####

### TASK 1: Bipartite Access Requests Network ###
edges = []
for usr_idx, rsr_idx in positive_cav_train[['USRID', 'RESID']].values:
    edges.append((int(usr_idx), int(rsr_idx)))
gb = nx.Graph()
gb.add_edges_from(edges)

gb = ig.Graph.from_networkx(gb)
gb.vs["name"] = gb.vs["_nx_name"]
del gb.vs["_nx_name"]

# Identify user and resources NODES
list_temp = list(positive_cav_train.USRID)
list_type = []
for node in gb.vs():
    if node['name'] in list_temp:
        list_type.append(0)
    else:
        list_type.append(1)

gb.vs['typen'] = list_type
print(gb.summary())
print("User nodes =", len(gb.vs.select(typen=0)))
print("Rsrc nodes =", len(gb.vs.select(typen=1)))
print("Is bipartite? ", gb.is_bipartite())
ig.write(gb, "output-files/bi-cav.gml")


g_proj_B = graph_projectionB(gb) # Bipartite projection
print(nx.info(g_proj_B)) # Networkx network

# Convert to igraph object graph
g_proj_B = ig.Graph.from_networkx(g_proj_B)
g_proj_B.vs["name"] = g_proj_B.vs["_nx_name"]
del g_proj_B.vs["_nx_name"]
print(ig.summary(g_proj_B)) # igraph network

###### ****** Communtity Detection ***** #####

### TASK 1: Community detection ###
partition = g_proj_B.community_multilevel(weights=g_proj_B.es()["weight"])
print("Modularity: %.4f" % partition.modularity)

# Agregar atributos al grafo bipartito
for attr in user_attr:    
    list_temp = []
    for node in g_proj_B.vs():                
        id_name = float(node["name"])
        temp = pd.DataFrame(positive_cav_train[positive_cav_train.USRID==id_name])            
        temp = temp[attr].drop_duplicates()            
        temp = temp.values[0]
        list_temp.append(temp)            
    g_proj_B.vs[attr] = list_temp

# Add cluster attribute to nodes
g_proj_B.vs["cluster"] = partition.membership
ig.write(g_proj_B, "output-files/proj-cav.gml")
print(g_proj_B.summary())

# Se obtienen las comunidades y sub-comunidades
g = g_proj_B

density_threshold = 0.5

n_coms = len(set(g.vs["cluster"]))
count_n_coms = 0 # Contador de número de comunidades detectadas
# Diccionario con las comunidades. Id de la comunidad como key y una lista 
# como value {01: [subgrafo, recursos_list]}
dict_total_coms = {} 

for idx_comm in range(n_coms): # Recorrer cada comunidad en la red
    subgraph_nodes = g.vs.select(cluster=idx_comm) # Sacamos los nodos con el cluster
    comm_subgraph = subgraph_nodes.subgraph() # Objeto de subgrafo

    # Se realiza nuevamente una partición
    new_partition = comm_subgraph.community_multilevel(
        weights =comm_subgraph.es["weight"] )        

    for sub_com in new_partition.subgraphs(): # Se recorre cada nueva sub comunidad
        # Usuarios en la comunidad
        user_set_comm = sub_com.vs()["name"]   
        # Recursos en la comunidad
        n_res = num_recursos(user_set_comm, gb)            
        dict_total_coms[str(count_n_coms)] = [sub_com, n_res]
        count_n_coms += 1

print("# de Comunidades: ", len(dict_total_coms))

### TASK 2: Community classification

# Obtener el máximo valor de recursos en el total de comunidades
n_res_in_comms = [len(i[1]) for i in dict_total_coms.values()]
max_n_res = max(n_res_in_comms)
print("Comunidad con # mayor recursos", max_n_res)

# Umbrales para la clasificación de comunidades
big_threshold = int(0.75 * max_n_res)
med_threshold = int(0.30 * max_n_res)
print("Big Threshold: ", big_threshold, " \t\t Med Threshold", med_threshold)

big_comms = [] # Almacenar las comunidades grandes
med_comms = [] # Almacenar las comunidades medianas
sma_comms = [] # Almacenar las comunidades pequeñas

for idx_com, com in enumerate(dict_total_coms.values()):
    if len(com[1]) > big_threshold: # Es comunidad grande
        big_comms.append([idx_com]+com)
    elif len(com[1]) > med_threshold: # Es comunidad mediana
        med_comms.append([idx_com]+com)
    else:
        sma_comms.append([idx_com]+com)

print("# Comunidades:",len(big_comms)+len(med_comms)+len(sma_comms), "==", len(dict_total_coms))
print("Big Comms:", len(big_comms))
print("Med Comms:", len(med_comms))
print("Sma Comms:", len(sma_comms))

###### ****** Rule inference ***** #####

dict_ress_coms = {} # Diccionario de recursos
list_rules = [] # Lista de reglas
dict_res_in_coms = {} # Diccionarios de recursos en comunidad
nuevas_reglas = [] # Lista de nuevas reglas.

### SMALL Comms. ###
counter_rules = 0
for comm in sma_comms:
    resorces_coms = comm[2] # Se extrae los recursos al que accede.
    if len(resorces_coms) == 1: # Si sólo hay un recurso
        # Se extraen los atributos del único recurso
        regla_i = [ ["id_com", str(comm[0])], [] ] # Se comienza generando la regla.        
        logs_with_resource = positive_cav_train[positive_cav_train["RESID"]==comm[2][0]].iloc[0]

        for attr in rsrc_attr: # Se agrega la regla con atr de recurso
            regla_i[1].append([attr, logs_with_resource[attr]])
        
        # Usuarios en la comunidad
        users_in_sub_com = set([int(float(i)) for i in comm[1].vs()["name"]])
        df_users_in_comm = positive_cav_train[positive_cav_train["USRID"].isin(users_in_sub_com)]
        df_users_in_comm = df_users_in_comm[user_attr+["USRID"]].drop_duplicates()                    
        
        # Obtener regla apriori
        regla_atr_usuario = apriori_in_resources_4(
            data_=df_users_in_comm[["control", "monitoring", "fallback"]],
            recurso_atr=regla_i[1], reglas_ante=nuevas_reglas)
        regla_i[1] = regla_i[1] + regla_atr_usuario

        list_rules.append(regla_i)
        nuevas_reglas.append(regla_i[1])

        counter_rules += 1
    else:
        regla_i = [ ["id_com", str(comm[0])], [] ] # Se comienza generando la regla.        
        
        # Solicitudes que incluyen el recurso
        logs_with_resource = positive_cav_train[positive_cav_train['RESID'].isin(comm[2])]
        logs_with_resource = logs_with_resource[rsrc_attr+["USRID","RESID"]].drop_duplicates()
        df_resources = logs_with_resource[rsrc_attr+["RESID"]].drop_duplicates()
        
        # Usuarios en la comunidad
        users_in_sub_com = set([int(float(i)) for i in comm[1].vs()["name"]])
        df_users_in_comm = positive_cav_train[positive_cav_train["USRID"].isin(users_in_sub_com)]
        df_users_in_comm = df_users_in_comm[user_attr+["USRID"]].drop_duplicates()                    
        
        # Generar regla con los atributos de los recursos
        regla_atr_recurso = apriori_in_resources_3(data_=df_resources[rsrc_attr]) 
        regla_atr_usuario = apriori_in_resources_4(
            data_=df_users_in_comm[["control", "monitoring", "fallback"]], 
            recurso_atr=regla_atr_recurso, reglas_ante=nuevas_reglas)        

        # Se agrega a la regla
        regla_i[1] = regla_i[1] + regla_atr_recurso + regla_atr_usuario
        # print(regla_i)
        
        list_rules.append(regla_i)
        nuevas_reglas.append(regla_i[1])
        counter_rules += 1

print("Small generated rules:", counter_rules)         

### MEDIUMN Comms. ###

counter_rules = 0
umbral_podado = 0.2
for comm in med_comms:
    resorces_coms = podar_recursos_new(comm[1], gb, umbral_podado)    
    if len(resorces_coms) == 1: # Si sólo hay un recurso
        # Se extraen los atributos del único recurso
        regla_i = [ ["id_com", str(comm[0])], [] ] # Se comienza generando la regla.        
        logs_with_resource = positive_cav_train[positive_cav_train["RESID"]==comm[2][0]].iloc[0]

        for attr in rsrc_attr: # Se agrega la regla con atr de recurso
            regla_i[1].append([attr, logs_with_resource[attr]])
        
        # Usuarios en la comunidad
        users_in_sub_com = set([int(float(i)) for i in comm[1].vs()["name"]])
        df_users_in_comm = positive_cav_train[positive_cav_train["USRID"].isin(users_in_sub_com)]
        df_users_in_comm = df_users_in_comm[user_attr+["USRID"]].drop_duplicates()                    
        
        regla_atr_usuario = apriori_in_resources_4(
            data_=df_users_in_comm[["control", "monitoring", "fallback"]],
            recurso_atr=regla_i[1], reglas_ante=nuevas_reglas)
        regla_i[1] = regla_i[1] + regla_atr_usuario

        list_rules.append(regla_i)
        nuevas_reglas.append(regla_i[1])
        counter_rules += 1
    else:
        regla_i = [ ["id_com", str(comm[0])], [] ] # Se comienza generando la regla.        
        
        # Solicitudes que incluyen el recurso
        logs_with_resource = positive_cav_train[positive_cav_train['RESID'].isin(comm[2])]
        logs_with_resource = logs_with_resource[rsrc_attr+["USRID","RESID"]].drop_duplicates()
        df_resources = logs_with_resource[rsrc_attr+["RESID"]].drop_duplicates()
        
        # Usuarios en la comunidad
        users_in_sub_com = set([int(float(i)) for i in comm[1].vs()["name"]])
        df_users_in_comm = positive_cav_train[positive_cav_train["USRID"].isin(users_in_sub_com)]
        df_users_in_comm = df_users_in_comm[user_attr+["USRID"]].drop_duplicates()                    
        
        # Generar regla con los atributos de los recursos
        regla_atr_recurso = apriori_in_resources_3(data_=df_resources[rsrc_attr]) 
        regla_atr_usuario = apriori_in_resources_4(
            data_=df_users_in_comm[["control", "monitoring", "fallback"]], 
            recurso_atr=regla_atr_recurso, reglas_ante=nuevas_reglas)  

        # Se agrega a la regla
        regla_i[1] = regla_i[1] + regla_atr_recurso + regla_atr_usuario
        # print(regla_i)
        
        list_rules.append(regla_i)
        nuevas_reglas.append(regla_i[1])
        counter_rules += 1

print("Medium generated rules:", counter_rules)            

### BIG Cooms. ###

counter_rules = 0
umbral_podado = 0.2
umbral_rec_sig = 0.5

for comm in big_comms:
    resorces_coms = podar_recursos_new(comm[1], gb, umbral_podado)    
    if len(resorces_coms) == 1: # Si sólo hay un recurso
        # Se extraen los atributos del único recurso
        regla_i = [ ["id_com", str(comm[0])], [] ] # Se comienza generando la regla.        
        logs_with_resource = positive_cav_train[positive_cav_train["RESID"]==comm[2][0]].iloc[0]

        for attr in rsrc_attr: # Se agrega la regla con atr de recurso
            regla_i[1].append([attr, logs_with_resource[attr]])
        
        # Usuarios en la comunidad
        users_in_sub_com = set([int(float(i)) for i in comm[1].vs()["name"]])
        df_users_in_comm = positive_cav_train[positive_cav_train["USRID"].isin(users_in_sub_com)]
        df_users_in_comm = df_users_in_comm[user_attr+["USRID"]].drop_duplicates()                    
        
        # Obtener regla apriori
        regla_atr_usuario = apriori_in_resources_4(
            data_=df_users_in_comm[["control", "monitoring", "fallback"]], 
            recurso_atr=regla_i[1], reglas_ante=nuevas_reglas)  
        regla_i[1] = regla_i[1] + regla_atr_usuario

        list_rules.append(regla_i)
        nuevas_reglas.append(regla_i[1])
        counter_rules += 1
    else:

        # Obtención de recursos significativos
        recursos_significativos = get_recursos_significativos2(comm[1], gb, umbral_rec_sig)

    
        for rec in recursos_significativos:
            regla_i = [ ["id_com", str(comm[0])], [] ] # Se comienza generando la regla.
            for attr in rsrc_attr: # Se agrega la regla con atr de recurso
                logs_with_resource = positive_cav_train[positive_cav_train["RESID"]==rec].iloc[0]
                regla_i[1].append([attr, logs_with_resource[attr]])

            # Usuarios en la comunidad
            users_in_sub_com = set([int(float(i)) for i in comm[1].vs()["name"]])
            df_users_in_comm = positive_cav_train[positive_cav_train["USRID"].isin(users_in_sub_com)]
            df_users_in_comm = df_users_in_comm[user_attr+["USRID"]].drop_duplicates()                    
            
            # Obtener regla apriori
            regla_atr_usuario = apriori_in_resources_4(
                data_=df_users_in_comm[["control", "monitoring", "fallback"]], 
                recurso_atr=regla_i[1], reglas_ante=nuevas_reglas)  
            
            regla_i[1] = regla_i[1] + regla_atr_usuario
            list_rules.append(regla_i)
            nuevas_reglas.append(regla_i[1])
            counter_rules += 1


print("Big generate rules:", counter_rules)            

new_rules_test = []
for i in list_rules:
    if not i[1] in new_rules_test:
        new_rules_test.append(i[1])
print("De ", len(list_rules), " - ", len(new_rules_test))
new_rules_test

false_neg  = []
for i,row in positive_cav_train.iterrows():
#for i, row in pos_test.iterrows():
    #print(row)
    user_id = row["USRID"]
    res_id = row["RESID"]    

    # Evaluación
    denies_count = 0
    temp_rules_n = 0
    for rule in new_rules_test:                                      
        # En esta parte se evalua la regla completa
        res = True                        
        for idx_r, attr_val in enumerate(rule):
            # print(idx_r, attr_val)                                    
            if row[attr_val[0]] != attr_val[1]:                                
                #print("Fallo en -- Row:",row[attr_val[0]], " --- Reg:", attr_val[1], " --- DIFE:", attr_val)
                res = False
                break                                            
        if res == False:
            denies_count += 1                                
    #print("XXX-", denies_count, temp_rules_n, res)
    if denies_count == len(new_rules_test):
        false_neg.append(row)
        #print("FP-2")
    #else:
        #print("ENtra PAPA")

FN = len(false_neg)
print("Tasa FN: {:.2f}".format((FN/ len(positive_cav_train))*100))
print("FN: ", FN, " de ", len(positive_cav_train))

false_pos  = []
for i,row in negative_cav_train.iterrows():
#for i, row in pos_test.iterrows():
    user_id = row["USRID"]
    res_id = row["RESID"]    

    # Evaluación
    denies_count = 0
    temp_rules_n = 0
    for rule in new_rules_test:                                      
        # En esta parte se evalua la regla completa
        res = True                        
        for idx_r, attr_val in enumerate(rule):
            # print(idx_r, attr_val)                                
            if row[attr_val[0]] != attr_val[1]:                                
                #print("Fallo en -- Row:",row[attr_val[0]], " --- Reg:", attr_val[1], " --- DIFE:", attr_val)
                res = False
                break                                            
        if res == False:
            denies_count += 1                                
    #print("XXX-", denies_count, temp_rules_n, res)
    if denies_count < len(new_rules_test):
        false_pos.append(row)
        #print("FP-2")    
    #else:
    #    print("ENtra PAPA")

FP = len(false_pos)
print("Tasa FP: {:.2f}".format((FP/ len(negative_cav_train))*100))
print("FN: ", FP, " de ", len(negative_cav_train))


TP = len(positive_cav_train) - FN
#TP = 50 - FN
TN = len(negative_cav_train) - FP
#TN = 50 - FP

precision = TP / (TP + FP)

recall = TP / (TP + FN)

fscore = 2*(precision*recall)/(precision+recall)

print("FN:", FN, " - {:.2f}".format((FN/len(positive_cav_train))*100))
#print("FN:", FN, " - {:.2f}".format((FN/50)*100))
print("FP:", FP, " - {:.2f}".format((FP/len(negative_cav_train))*100))
#print("FP:", FP, " - {:.2f}".format((FP/50)*100))
print("Precision:", precision)
print("Recall:", recall)
print("F-score", fscore)

def compute_wsc(policy):
    return sum([len(rule[1]) for rule in policy])

print("# Rules:", len(new_rules_test))
print("WSC:", compute_wsc(new_rules_test))