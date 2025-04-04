#True=MAC        False=Ubuntu
ENVIRONMENT = False
DATASET = "AMZ" # AMZ, HC, PM, UN, TOY
NODE_TYPE = False
GRAPH_FILE = "AMZ_bot_weights_NC_alph2.graphml"
FILELOG = open("OUT-04-BridgeNodes.txt", "w")

def internal_external_degree_node(node):
    """Compute the internal degree k_i^int of node i in a community C."""
    node_neighs = node.neighbors()
    node_community = node["cls"]

    internal_degree = [1 for i in node_neighs if i["cls"]==node_community]
    external_degree = [1 for i in node_neighs if i["cls"]!=node_community]
    
    return sum(internal_degree), sum(external_degree)


def check_strong_community(graph, communities):
    """Returns if it is a strong community."""

    dict_to_ret = {}

    for id_c in set(g.vs["cls"]):
        flag_weak_comm = False
        porcentaje = 0
        comms = g.vs.select(cls_eq=id_c)
        for node in comms:
            vertex = g.vs.find(id_eq=node["id"])
            int_degree, ext_degree = internal_external_degree_node(vertex)
            if int_degree <= ext_degree:
                flag_weak_comm = True
                porcentaje += 1
        
        if flag_weak_comm:
            dict_to_ret[id_c] = [comms, False, porcentaje/len(comms)]
        else:
            dict_to_ret[id_c] = [comms, True, 0]

    return dict_to_ret

# Check the path environment
import sys
if ENVIRONMENT:
    sys.path.append('/home/daniel/Documents/phd/phd-thesis-lab/12-third_year/05-CommunityDetection/')
else:
    sys.path.append('/home/daniel/Documents/phd/phd-thesis-lab/12-third_year/05-CommunityDetection/')

### Import libraries
import pandas as pd
import igraph as ig
import numpy as np
import matplotlib.pyplot as plt
import auxiliar_path
from datetime import datetime


### Global variables
PATH_DATASET = auxiliar_path.get_path_dataset(DATASET)
PATH_NODETYPE = auxiliar_path.get_path_topbot(NODE_TYPE)

if ENVIRONMENT:
    GLOBAL_PATH = "/Users/ddiaz/Documents/code/phd-thesis-lab/12-third_year/"    
else:
    GLOBAL_PATH = "/home/daniel/Documents/phd/phd-thesis-lab/12-third_year/"

GRAPH_PATH = ( GLOBAL_PATH + "00-Data/" + PATH_DATASET + "/02-Graphs/" +
              PATH_NODETYPE + "/" + GRAPH_FILE)

FILELOG.write("04-BridgeNodes\n")
FILELOG.write(datetime.today().strftime('%Y-%m-%d %H:%M:%S')+"\n\n")

### Load Graph
g = ig.Graph.Read_Edgelist("/home/daniel/Documents/phd/phd-thesis-lab/12-third_year/05-CommunityDetection/youtube.edgelist")
g = g.as_undirected()
g.vs["id"] = [v.index for v in g.vs()]
FILELOG.write(f"{g.summary()}\n")

FILELOG.write("Graph info:\n")
FILELOG.write(f"\t|V| = {g.vcount()} \n")
FILELOG.write(f"\t|E| = {g.ecount()} \n")
FILELOG.write(f"\t d  = {g.density()} \n\n")


### Community Detection
FILELOG.write(" === Starting Community Detection === \n")
#comms = g.community_multilevel(weights=g.es["weight"])
comms = g.community_multilevel()
g.vs["cls"] = comms.membership
FILELOG.write("Community Detection Resumen:\n")
FILELOG.write(comms.summary() + "\n\n")


### Removing communities with less than 5 elements
comms_with_one = {subgraph.vs["cls"][0]: subgraph for i, subgraph in enumerate(comms.subgraphs()) if len(subgraph.vs) > 5}
comms_to_remove = [sg for sg in comms.subgraphs() if len(sg.vs) <= 5]
FILELOG.write(f"Communities > 5: {len(comms_with_one)}\n\n")


### Nodes to remove
nodes_to_remove = []
for i in comms_to_remove:
    for node in i.vs:
        nodes_to_remove.append(node["id"])
asd = []
for node in nodes_to_remove:
    asd.append(g.vs.find(id_eq=node).index)
FILELOG.write(f"Nodes to remove: {len(asd)}\n\n")


### Remove nodes
g.delete_vertices(asd)
FILELOG.write("Deleting nodes: Done\n\n")


### SUbcommunity detection
new_ids_comms = 1500
for com in comms_with_one.values():
    if com.density() > 0.5:
        #new_partition = com.community_multilevel(weights=com.es["weight"])
        new_partition = com.community_multilevel()
        for new_comms in new_partition.subgraphs():
            for nodes in new_comms.vs():
                node_to_add = g.vs.find(id_eq=nodes["id"])
                node_to_add["cls"] = new_ids_comms
            new_ids_comms += 1


### Number of strong communities.
strongcomms = check_strong_community(g, comms_with_one)


contador = 0
contador_striong = 0
for porcentaje in strongcomms.values():
    contador += porcentaje[2]
    if porcentaje[1]:
        contador_striong+=1

FILELOG.write(f"Nodos puente {contador / len(strongcomms)}\n")
FILELOG.write(f"Comunidades strong {contador_striong}\n")
FILELOG.write(f"Comunidades weak {len(strongcomms)-contador_striong}\n")

print("Done\n")