
### Import libraries
import pandas as pd
import igraph as ig
import auxiliar_path
import sys


##### ***** Global variables ***** *****
#DATASET = "TOY" # AMZ, HC, PM, UN, TOY
DATASET = sys.argv[1]
if not DATASET in ["AMZ", "HC", "PM", "UN", "TOY"]:
    print("\n ***** ERROR: Incorrect Dataset *****\n")
    sys.exit(1)

PATH_DATASET = auxiliar_path.get_path_dataset(DATASET)
GLOBAL_PATH = "/Users/ddiaz/Documents/code/phd-thesis-lab/"
# File CSV INPUT
FILENAME = (GLOBAL_PATH + "12-third_year/00-Data/" + PATH_DATASET +
            "/01-DistributionsCSV/" + DATASET + "-Rw.csv")

##### ***** Clean dataset ***** #####
### Read CSV
df = pd.read_csv(FILENAME)
print("\n CSV loaded!", "\n")

# Remove noisy column
df = df.drop(columns=["Unnamed: 0"])
print(df.info()) # Info
print()

# Obtener identificadores únicos por tipo
unique_source = sorted(df['uname'].unique())  # Nodos tipo 1
unique_target = sorted(df['rname'].unique())  # Nodos tipo 2

# Crear nuevo mapeo de IDs
source_mapping = {old_id: new_id for new_id, old_id in enumerate(unique_source)}
# El primer ID del target será el último del source + 1
start_target_id = len(unique_source)
target_mapping = {old_id: new_id for new_id, old_id in enumerate(unique_target,
                                                                 start=start_target_id)}

# Aplicar el mapeo al dataframe
df_mapped = df.replace({'uname': source_mapping, 'rname': target_mapping})

# Contar la frecuencia correcta después del mapeo
source_counts = df_mapped['uname'].value_counts().to_dict()
target_counts = df_mapped['rname'].value_counts().to_dict()

# Fusionar ambas frecuencias.
node_frequencies = {node: source_counts.get(node, 0) + 
                    target_counts.get(node, 0) for node in range(len(source_mapping) + 
                                                                 len(target_mapping))}

# Contar la frecuencia de cada arista después del mapeo
edge_counts = df_mapped.groupby(['uname', 'rname']).size().to_dict()

# Some information about access requests
n_user = len(df_mapped.uname.drop_duplicates())
n_rscs = len(df_mapped.rname.drop_duplicates())
print(f"|U| = {n_user}")
print(f"|R| = {n_rscs}")
print(f"|U+R| = {n_user+n_rscs}")
print()

# Possible edges
n_acc_res = len(df_mapped.drop_duplicates(["uname", "rname"]))
df_pos = df_mapped[df_mapped.ACTION == 1]
n_ar_pos = len(df_pos.drop_duplicates())
n_ar_neg = len(df_mapped[df_mapped.ACTION == 0].drop_duplicates())

print(f"|L| = {n_acc_res}")
print(f"|L+| = {n_ar_pos}")
print(f"|L-| = {n_ar_neg}")
print()

if n_acc_res == n_ar_pos+n_ar_neg:
    print("*"*43)
    print("** CORRECT FLAG: Same number L = L+ + L- **")
    print("*"*43)

# To generate a new .CSV file with the clean data
filename_csv = (GLOBAL_PATH + "12-third_year/00-Data/" + PATH_DATASET +
                "/01-DistributionsCSV/" + DATASET + "-MOD.csv")
df_mapped.to_csv(filename_csv, index=False)
print("\n New CSV Generated!", "\n")

##### ***** Generate Bipartite GRaph ***** #####
### Generate bipartite graph

# Crear el grafo bipartito en igraph
edges = list(edge_counts.keys())  # Lista de aristas sin duplicados
g = ig.Graph(edges=edges, directed=False)

# Agregar identificador
g.vs['id'] = list(range(g.vcount()))

# Etiquetar los nodos con su tipo: 0 para tipo 1, 1 para tipo 2
g.vs['type'] = [0] * len(source_mapping) + [1] * len(target_mapping)  

# Agregar el atributo de frecuencia de los nodos
g.vs['freq'] = [node_frequencies[node] for node in range(len(g.vs))]

# Agregar el atributo de peso a las aristas
g.es['weight'] = [edge_counts[edge] for edge in edges]
print("\n Bipartite Graph Generated!", "\n")
# Number of nodes
print(g.summary())
print(f"|V| = {g.vcount()}")
print(f"|U| = {len(g.vs.select(type_eq=0))}")
print(f"|R| = {len(g.vs.select(type_eq=1))}")
print(f"|E| = {g.ecount()}")
print(f"Is bipartite = {g.is_bipartite()}")

### Save the graph
FILE_GRAPH = (GLOBAL_PATH + "12-third_year/00-Data/" + PATH_DATASET +
              "/02-Graphs/binet-" + DATASET + "-Rw.graphml")
g.write_graphml(FILE_GRAPH)

print("\n\n ***** DONE! Everything is OK! ***** \n\n")