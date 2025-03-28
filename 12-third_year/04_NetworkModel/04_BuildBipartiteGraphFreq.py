import pandas as pd
import igraph as ig

# Cargar el DataFrame
GLOBAL_PATH = "/Users/ddiaz/Documents/code/phd-thesis-lab/"
FILENAME = GLOBAL_PATH + "12-third_year/00-Data/01-AMZ/01-DistributionsCSV/AMZ-Rw.csv"
df = pd.read_csv(FILENAME)

# Filtrar las filas donde ACTION es 1
df = df[df.ACTION == 1]

# Crear conjuntos de nombres únicos
set_rname = set(df["rname"])
set_uname = set(df["uname"])

# Ajustar los nombres de rname para evitar colisiones con uname
df["rname"] = df["rname"] + len(set_uname) + 1
set_rname = set(df["rname"])

# Crear una lista de aristas y contar la frecuencia de cada arista
edges = []
edge_frequencies = {}
for _, row in df.iterrows():
    edge = (row['rname'], row['uname'])
    edges.append(edge)
    if edge in edge_frequencies:
        edge_frequencies[edge] += 1
    else:
        edge_frequencies[edge] = 1

# Contar la frecuencia de aparición de cada nodo en el dataset
node_frequencies = {}
for node in set_rname:
    node_frequencies[node] = df[df["rname"] == node].shape[0]
for node in set_uname:
    node_frequencies[node] = df[df["uname"] == node].shape[0]

# Crear el grafo bipartito
graph = ig.Graph(directed=False)

# Añadir vértices al grafo
graph.add_vertices(len(set_rname) + len(set_uname))

# Añadir el atributo 'type' a cada vértice
graph.vs["type"] = 0  # Inicializar todos los vértices con type 0
graph.vs[len(set_rname):]["type"] = 1  # Establecer type 1 para los vértices de set_uname

# Crear un mapeo entre nombres de nodos e índices
rname_to_index = {rname: i for i, rname in enumerate(set_rname)}
uname_to_index = {uname: i for i, uname in enumerate(set_uname)}

# Añadir aristas al grafo usando el mapeo creado y asignar la frecuencia como atributo
for edge in edges:
    rname, uname = edge
    if rname in rname_to_index and uname in uname_to_index:
        graph.add_edge(rname_to_index[rname], len(set_rname) + uname_to_index[uname])
        # Asignar la frecuencia como atributo de la arista
        graph.es[graph.ecount() - 1]["weight"] = edge_frequencies[edge]

# Asignar la frecuencia de aparición como atributo a los nodos
for node, index in rname_to_index.items():
    graph.vs[index]["frequency"] = node_frequencies[node]
for node, index in uname_to_index.items():
    graph.vs[len(set_rname) + index]["frequency"] = node_frequencies[node]

# Imprimir información sobre el grafo
print(f"Number of vertices: {graph.vcount()}")
print(f"Number of edges: {graph.ecount()}")
print(f"Is bipartite: {graph.is_bipartite()}")

# Guardar el grafo en formato GraphML
graph.write_graphml("TESTAMZ.graphml")