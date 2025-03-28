import os
import pandas as pd
import igraph as ig
import auxiliar_path
import matplotlib.pyplot as plt
from kmodes import KModes

def load_graphs_from_folder(folder_path):
    graphs = []
    for file in os.listdir(folder_path):
        if file.endswith(".graphml"):
            graph = ig.read(os.path.join(folder_path, file))
            graphs.append((file, graph))
    return graphs

def get_neis_comms(graph, not_in):
    dict_user = {}
    for user in graph.vs():
        cluster_ = user["cls"]
        if cluster_ not in not_in:
            neis_user = list(graph.vs.select(cls_eq=cluster_)["id"])
            dict_user[int(user["id"])] = set(map(int, neis_user))
    return dict_user

def get_neis(data, cluster_avoid):
    dict_user, dict_res = {}, {}
    for user in data.uname.drop_duplicates():
        all_clusters = set(data[data.uname == user]["cls"])
        all_clusters.difference_update(cluster_avoid)
        dict_user[user] = set(data[data.cls.isin(all_clusters)]["uname"])
    
    for res in data.rname.drop_duplicates():
        all_clusters = set(data[data.rname == res]["cls"])
        all_clusters.difference_update(cluster_avoid)
        dict_res[res] = set(data[data.cls.isin(all_clusters)]["rname"])
    
    return dict_user, dict_res

def jaccard_sim(dict_kmodes, dict_louvain):
    total_mean = []
    dict_resul = {
        key_: len(item_.intersection(dict_louvain.get(key_, set()))) / len(item_.union(dict_louvain.get(key_, set())))
        for key_, item_ in dict_kmodes.items() if key_ in dict_louvain
    }
    total_mean = list(dict_resul.values())
    return dict_resul, total_mean

def plot_jaccard_comparison(results):
    plt.figure(figsize=(10, 6))
    for graph_name, jacc_list in results:
        plt.plot(sorted(jacc_list), label=f'{graph_name}')

    plt.xlabel('Usuario')
    plt.ylabel('Jaccard Index')
    plt.title('Comparación del Índice de Jaccard')
    #plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.show()

# === Main Execution ===
DATA_FOLDER = "/Users/ddiaz/Documents/code/phd-thesis-lab/12-third_year/00-Data/01-AMZ/02-Graphs/01-Top/"
GRAPHS = load_graphs_from_folder(DATA_FOLDER)

### Global variables
### Global variables

DATASET = "AMZ" # AMZ, HC, PM, UN, TOY
NODE_TYPE = True

PATH_DATASET = auxiliar_path.get_path_dataset(DATASET)
PATH_NODETYPE = auxiliar_path.get_path_topbot(NODE_TYPE)

GLOBAL_PATH = "/Users/ddiaz/Documents/code/phd-thesis-lab/"

# File CSV
FILENAME = GLOBAL_PATH + "12-third_year/00-Data/"+PATH_DATASET+"/01-DistributionsCSV/"+DATASET+"-MOD.csv"
### Read CSV

df = pd.read_csv(FILENAME)

# Remove noisy column
#df = df.drop(columns=["Unnamed: 0"])
print(df.info()) # Info
print()


# Some information about access requests
n_user = len(df.uname.drop_duplicates())
n_rscs = len(df.rname.drop_duplicates())
print(f"|U| = {n_user}")
print(f"|R| = {n_rscs}")
print(f"|U+R| = {n_user+n_rscs}")
print()

# Possible edges
n_acc_res = len(df.drop_duplicates(["uname", "rname"]))
df_pos = df[df.ACTION == 1]
n_ar_pos = len(df_pos.drop_duplicates())
n_ar_neg = len(df[df.ACTION == 0].drop_duplicates())

print(f"|L| = {n_acc_res}")
print(f"|L+| = {n_ar_pos}")
print(f"|L-| = {n_ar_neg}")
print()

if n_acc_res == n_ar_pos+n_ar_neg:
    print("*"*43)
    print("** CORRECT FLAG: Same number L = L+ + L- **")
    print("*"*43)


all_results = []
for graph_name, g in GRAPHS:
    comms = g.community_multilevel(weights=g.es["weight"])
    g.vs["cls"] = comms.membership

    comms_with_one = [i for i, subgraph in enumerate(comms.subgraphs()) if len(subgraph.vs) > 5]
    dict_res_comms = get_neis_comms(g, comms_with_one)


    # Select the number of clusters###
    num_clusters = len(comms_with_one)
    kmodes_huang = KModes(n_clusters=num_clusters, init='Huang', verbose=0)
    cluster_labels = kmodes_huang.fit_predict(df_pos.drop(columns=["rname", "uname"]))
    centroids = kmodes_huang.cluster_centroids_
    df_pos["cls"] = cluster_labels
    print('Ready!')   

    comms_with_one = [i for i, subgraph in enumerate(comms.subgraphs()) if len(subgraph.vs) <= 5]
    dict_res_comms = get_neis_comms(g, comms_with_one)
    dict_res_comms

    list_cluster_check = []
    for i in range(max(df_pos["cls"])):
        temp = df_pos[df_pos["cls"]==i]
        if len(temp["uname"].drop_duplicates()) <= 5:
            list_cluster_check.append(i)

    list_cluster_check = set(list_cluster_check)
    list_cluster_check, len(list_cluster_check)

    dict_user, dict_res = get_neis(df_pos, set())
    _, jacc_list = jaccard_sim(dict_user, dict_res_comms)

    all_results.append((graph_name, jacc_list))
    

plot_jaccard_comparison(all_results)
