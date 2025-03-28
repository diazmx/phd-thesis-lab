import pandas as pd
import igraph as ig
import auxiliar_path
import numpy as np
from kmodes import KModes
import matplotlib.pyplot as plt

### Global variables

DATASET = "AMZ" # AMZ, HC, PM, UN, TOY
NODE_TYPE = True # TRUE = User      FALSE = Resources

PATH_DATASET = auxiliar_path.get_path_dataset(DATASET)
PATH_NODETYPE = auxiliar_path.get_path_topbot(NODE_TYPE)

GLOBAL_PATH = "/Users/ddiaz/Documents/code/phd-thesis-lab/"

DF_PATH = (
    GLOBAL_PATH + "12-third_year/00-Data/" + PATH_DATASET + 
    "/01-DistributionsCSV/" + DATASET + "-MOD.csv"
    )
GRAPH_PATH = (
    GLOBAL_PATH +
    "12-third_year/00-Data/01-AMZ/02-Graphs/01-Top/AMZ_top_resall_DF_alpha2.graphml"
    )


def get_filtered_community_sizes_distribution(filtered_comms):
    """
    Get the size distribution of a list of communities.
    filtered_comms: list of list:
        (i, subgraph):
            i = index
            subgraph = community
    """
    size_distri = [c.vcount() for i, c in filtered_comms]
    return size_distri


def members(cluster_id, node_type, df_):
    """
    Get all members of a cluster
    """
    temp = df_[df_.cls==cluster_id]
    if node_type:
        return list(temp.uname.drop_duplicates())
    else:
        return list(temp.rname.drop_duplicates())


def belongs(id_entity, node_type, df_):
    """
    Get all clusters of an entity
    """
    temp = None
    if node_type:
        temp = df_[df_.uname==id_entity]
    else:
        temp = df_[df_.rname==id_entity]
    return list(temp["cls"].drop_duplicates()) # Lista de todos los clusters

def get_Neis_Kmodes(id_ent, node_type, df_):
    """
    Get all entity neighborhood of an entity.
    """
    neis_to_ret = []
    
    # Get all 
    clusters_i_belongs = belongs(id_ent, node_type, df_)

    # For all cluster
    for i in clusters_i_belongs:
        neis_to_ret = neis_to_ret + members(id_ent, node_type, df_)
    
    return list(set(neis_to_ret))

def jaccard_index_nodes(df_, graph_, node_type):
    dict_to_ret = {}
    for node in graph_.vs():
        id_node = node["id"]

        # Seleccionar todos los nodos que tienen el mismo cluster
        neis_graph_i = list(graph_.vs.select(cls_eq=node["cls"]))

        if len(neis_graph_i) == 0:
            neis_graph_i = set()
        else:
            new_neis = []
            for i in neis_graph_i:
                new_neis.append(int(i["id"]))
            neis_graph_i = set(new_neis)
        
        neis_df_i = get_Neis_Kmodes(id_node, node_type, df_)

        idx_jacc = len(neis_graph_i.intersection(neis_df_i)) / len(neis_graph_i.union(neis_df_i))

        dict_to_ret[id_node] = idx_jacc

    #dict_to_ret = dict(sorted(dict_to_ret.items(), key=lambda item: item[1]))

    return dict_to_ret


        


        

### Read CSV
df = pd.read_csv(DF_PATH)
print(df.info()) # Info
print()

n_user = len(df.uname.drop_duplicates())
n_rscs = len(df.rname.drop_duplicates())
print(f"|U| = {n_user}")
print(f"|R| = {n_rscs}")
print(f"|U+R| = {n_user+n_rscs}")
print()

n_acc_res = len(df.drop_duplicates(["uname", "rname"]))
df_pos = df[df.ACTION == 1]
n_ar_pos = len(df_pos.drop_duplicates())
n_ar_neg = len(df[df.ACTION == 0].drop_duplicates())

print(f"|L| = {n_acc_res}")
print(f"|L+| = {n_ar_pos}")
print(f"|L-| = {n_ar_neg}")
print()


### Read GRAPH
g = ig.read(GRAPH_PATH)
print(g.summary(), "\n")
print("Graph info:")
print("\t|V| =", g.vcount())
print("\t|E| =", g.ecount())
print("\t d  =", g.density())

### Compute Louvain Algorithm
comms = g.community_multilevel(weights=g.es["weight"])
g.vs["cls"] = comms.membership
print(comms.summary(), "\n")
print(f"Number of Detected Communities: {len(comms)}")

### How many graphs with less than 5 nodes.
accepted_comms = [
    [i, subgraph] for i, subgraph in enumerate(comms.subgraphs())
    if len(subgraph.vs) > 5
    ]
deleted_comms = [
    [i, subgraph] for i, subgraph in enumerate(comms.subgraphs())
    if len(subgraph.vs) <= 5
    ]
print(f"\nAccepted Comms: {len(accepted_comms)} of {len(comms)}")
accepted_size_distri = get_filtered_community_sizes_distribution(accepted_comms)
print(f"Deleted Comms: {len(deleted_comms)} of {len(comms)}")


### Compute K-Modes Algorithm with k = len(accepted_comms)
df_pos_unique = df_pos.drop_duplicates()
k_clusters = int((len(df_pos_unique) * len(accepted_comms)) / g.vcount())
centroids = []
kmodes_huang = KModes(n_clusters=k_clusters, init='Huang', verbose=0)
cluster_labels = kmodes_huang.fit_predict(
    df_pos_unique.drop(columns=["rname", "uname"])
    )
centroids = kmodes_huang.cluster_centroids_
df_pos_unique["cls"] = cluster_labels
print("\nK-modes Completed!")
print(f"Number of Generated Clusters: {len(df_pos_unique.cls.value_counts())}")


### Show Community Distribution
fig0 = plt.figure(0, figsize=(8, 5))
plt.plot(sorted(accepted_size_distri, reverse=True))
plt.xlabel('ID of Community')
plt.ylabel('Size (Number of nodes)')
plt.title('Community Size Distribution')
plt.xticks(rotation=0)
plt.grid(axis='y', linestyle='--', alpha=0.7)

### Show Clusters Distribution
fig1 = plt.figure(1, figsize=(8, 5))
value_counts = df_pos_unique["cls"].value_counts()
plt.plot(sorted(value_counts, reverse=True))
plt.xlabel('ID of Cluster')
plt.ylabel('Size (Number of access requests)')
plt.title('Clusters Size Distribution')
plt.xticks(rotation=0)
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Show entity per community
ent_per_cluster = []
ent_per_cluster_data = []
for i in range(max(df_pos_unique["cls"])):
    if NODE_TYPE:
        entity_cluster = df_pos_unique[df_pos_unique["cls"]==i]["uname"].drop_duplicates()
    else:
        entity_cluster = df_pos_unique[df_pos_unique["cls"]==i]["rname"].drop_duplicates()
    ent_per_cluster.append(len(entity_cluster))
    ent_per_cluster_data.append(entity_cluster)

fig2 = plt.figure(2, figsize=(8, 5))
plt.plot(sorted(ent_per_cluster, reverse=True))
plt.xlabel('ID of Cluster')
plt.ylabel('Entity Size (Number of access users/resources)')
plt.title('Clusters User/Resource Size Distribution')
plt.xticks(rotation=0)
plt.grid(axis='y', linestyle='--', alpha=0.7)


# Obtener los vecinos del nodo
res = jaccard_index_nodes(df_pos_unique, g, NODE_TYPE)

keys_ = list(res.keys())
values_ = list(res.values())

fig3 = plt.figure(3, figsize=(8, 5))
plt.plot(keys_, values_,)
plt.xlabel('ID of entity')
plt.ylabel('Jaccard Index (Number of entities in common)')
plt.title('Jaccard Index')
plt.xticks(rotation=0)
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Show the plot
plt.show()