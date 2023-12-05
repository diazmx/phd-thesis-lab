### Import libraries
import igraph as ig
import pandas as pd
import numpy as np


# Load data
file_name = "../data/IoT-expo-universal.csv"
acc_log = pd.read_csv(file_name)
acc_log = acc_log[acc_log.columns[1:]]
acc_log["rname"] = acc_log["rname"] + max(acc_log.uname.unique()) + 1

print("Done!")

user_attributes = ["role", "age", "health", "uname"]
users = acc_log[user_attributes].drop_duplicates()
users = users.reset_index(drop=True)
print("|U| =", len(users))
# users.head()

res_attributes = ["area", "mode", "temperature", "lockstatus", "rname"]
resrs = acc_log[res_attributes].drop_duplicates()
resrs = resrs.reset_index(drop=True)
print("|R| =", len(resrs))
#resrs.head()

edges_attributes = ["uname", "rname", "ACTION"]
edges = acc_log[edges_attributes].drop_duplicates()
edges = edges.reset_index(drop=True)
edges["weight"] = np.round(np.random.random(len(edges)), 2)
print("|E| =", len(edges))
# edges.head()

### Create a graph

# Iteration over tuples in the dataframe
tuple_list = edges[["uname", "rname", "ACTION"]].itertuples(index=False)

# Using the TupleList method to build the network
bip_network = ig.Graph.TupleList(tuple_list, directed=False, edge_attrs=["ACTION"])


print(bip_network.summary())

# Check if it is a bipartite network
if bip_network.is_bipartite():
    print("It is bipartite!")
else:
    print("The network is not bipartite.")

### Add user attributes

# Add type of node:     0=User      1=Resource
user_nodes = bip_network.vs.select(name_le=max(users.uname.unique()))
resource_nodes = bip_network.vs.select(name_gt=max(users.uname.unique()))
user_nodes["type"] = 0
resource_nodes["type"] = 1

# Add attributes
for attr in user_attributes[:-1]:   # User attributes
    user_nodes[attr] = users[attr]
for attr in res_attributes[:-1]:    # Resource attributes
    resource_nodes[attr] = resrs[attr]

# Remove objecto to free memory
del user_nodes, resource_nodes

bip_network.write("iot_bip_graph.graphml")