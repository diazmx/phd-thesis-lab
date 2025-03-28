import igraph as ig
from auxiliar_bb import noise_corrected, disparity
from auxiliar_projections_large import apply_projection, multiply_weigt, remove_zeros
from time import time
from auxiliar_path import get_path_dataset
import sys

##### ***** Global variables ***** *****
#DATASET = "TOY" # AMZ, HC, PM, UN, TOY
DATASET = sys.argv[1]
if not DATASET in ["AMZ", "HC", "PM", "UN", "TOY"]:
    print("\n ***** ERROR: Incorrect Dataset *****\n")
    sys.exit(1)
#PROJ_NAME = simple weights vector master hyperbolic resall
PROJ_NAME = sys.argv[2]
if not PROJ_NAME in ["simple", "weights", "vector",
                     "master", "hyperbolic", "resall"]:
    print("\n ***** ERROR: Incorrect PROJECTION NAME *****\n")
    sys.exit(1)

PATH_DATASET = get_path_dataset(DATASET)
GLOBAL_PATH = "/Users/ddiaz/Documents/code/phd-thesis-lab/"

# GRAPHML BIPARTITE FILE
FILENAME = (GLOBAL_PATH + "12-third_year/00-Data/" + PATH_DATASET +
            "/02-Graphs/binet-"+DATASET+"-Rw.graphml")
###### ****** END ****** ######


###### ****** Read BI GRAPH ****** ######
g = ig.read(FILENAME)
print(g.summary())
print()

user_nodes = g.vs.select(type=0)
res_nodes = g.vs.select(type=1)

if(g.is_bipartite()): # Check if the the graph is bipartite
    print("The graph IS bipartite")
else:
    print("The graph IS NOT bipartite")
    exit()
print("|U|=",len(user_nodes), " \t|R|=",len(res_nodes), " \t|U|+|R|=",
      len(user_nodes)+len(res_nodes), "=", g.vcount())
print()
###### ****** END ****** ######


###### ****** Projections ****** ######
user_graph = apply_projection(g, PROJ_NAME,
                              len(user_nodes), False) # False = Users = 0
print("Done PROJ1 - Users Projection")
edges_temp = user_graph.es()["weight"]
print(f"Peso máximo={max(edges_temp)} y mínimo={min(edges_temp)} en aristas: ")

rsrs_graph = apply_projection(g, PROJ_NAME,
                              len(user_nodes), True) # True = Resources = 1
print("\nDone PROJ2 - Resources Projection")
edges_temp = rsrs_graph.es()["weight"]
print(f"Peso máximo={max(edges_temp)} y mínimo={min(edges_temp)} en aristas: ")
print()
###### ****** END ****** ######


###### ****** BACKBONING USERS ****** ######
g_toy = user_graph # Graph to analyze
print("\n##### **** BACKBONING USERS **** #####")
print("Projection Name:", PROJ_NAME)
print("Summary\n",g_toy.summary())
print("##### END #####")
print()
### Disparity filter ###
a = time()
bb_df = disparity(g_toy)
b = time() - a
print("TOP DF - time: %.10f seconds." % b)
for alpha__, g__ in bb_df.items():
    print(f"Grafo filtrado con alpha={alpha__}: {g__.summary()}")
    flname = (
        "../00-Data/" + PATH_DATASET + "/02-Graphs/01-Top/"  + DATASET +
        "_top_" + PROJ_NAME + "_DF_alpha" + str(alpha__)[2:] + ".graphml"
    )
    g__.write_graphml(flname)
print("================================")
### Noise Corrected ###
a = time()
bb_nc = noise_corrected(g_toy)
b = time() - a
print("TOP NC - time: %.10f seconds." % b)
for alpha__, g__ in bb_nc.items():
    print(f"Grafo filtrado con alpha={alpha__}: {g__.summary()}")
    flname = (
        "../00-Data/" + PATH_DATASET + "/02-Graphs/01-Top/"  + DATASET +
        "_top_" + PROJ_NAME + "_NC_alpha" + str(alpha__)[2:] + ".graphml"
    )
    g__.write_graphml(flname)
print("================================")
print()
print("##### ***** Done BACKBONIN GUSERS ***** #####")
###### ****** END ****** ######


###### ****** BACKBONING RESOURCES ****** ######
g_toy = rsrs_graph # Graph to analyze
print("\n##### **** Projection RESOURCE **** #####")
print("Summary\n",g_toy.summary())
print("##### END #####")
print()
### Disparity filter ###
a = time()
bb_df = disparity(g_toy)
b = time() - a
print("BOT DF - time: %.10f seconds." % b)
for alpha__, g__ in bb_df.items():
    print(f"Grafo filtrado con alpha={alpha__}: {g__.summary()}")
    flname = (
        "../00-Data/" + PATH_DATASET + "/02-Graphs/02-Bot/" + DATASET +
        "_bot_" + PROJ_NAME + "_DF_alpha" + str(alpha__)[2:] + ".graphml"
    )
    g__.write_graphml(flname)
print("================================")
# Noise Corrected
a = time()
bb_nc = noise_corrected(g_toy)
b = time() - a
print("BOT NC - time: %.10f seconds." % b)
for alpha__, g__ in bb_nc.items():
    print(f"Grafo filtrado con alpha={alpha__}: {g__.summary()}")
    flname = (
        "../00-Data/" + PATH_DATASET + "/02-Graphs/02-Bot/"  +DATASET +
        "_bot_" + PROJ_NAME + "_NC_alph" + str(alpha__)[2:] + ".graphml"
    )
    g__.write_graphml(flname)
print("================================")
print()
print("##### ***** Done RSCS ***** #####")
###### ****** END ****** ######