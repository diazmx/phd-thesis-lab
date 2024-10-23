import networkx as nx
import numpy as np
from scipy import integrate
import pandas as pd
from netbone.backbone import Backbone
from netbone.filters import threshold_filter, fraction_filter
import time

def disparity(data, identifier, weight='weight'):
    if isinstance(data, pd.DataFrame):
        g = nx.from_pandas_edgelist(data, edge_attr='weight', create_using=nx.Graph())
    elif isinstance(data, nx.Graph):
        g = data.copy()
    else:
        print("data should be a panads dataframe or nx graph")
        return
    # Archivos únicos usando el identificador
    f1_name = f"time_nodes_1000_{identifier}.txt"
    f2_name = f"time_nodes_10000_{identifier}.txt"
    
    f1 = open(f1_name, "w")
    f2 = open(f2_name, "w")

    start_time = time.time()
    node_counter = 0
    nodes_logged_1000 = False
    nodes_logged_10000 = False
    
    strength = g.degree(weight=weight)
    #print(strength)
    for node in g.nodes(): 
        node_counter += 1  # Contador de nodos procesados

        if node_counter == 1000 and not nodes_logged_1000:
            elapsed_time = time.time() - start_time
            f1.write(f"Tiempo para 1000 nodos: {elapsed_time:.4f} segundos\n")
            f1.flush()  # Forzamos la escritura al archivo
            f1.close()  # Cerramos el archivo para permitir su lectura mientras se sigue ejecutando el código
            nodes_logged_1000 = True  # Marcamos como completado

        if node_counter == 10000 and not nodes_logged_10000:
            elapsed_time = time.time() - start_time
            f2.write(f"Tiempo para 10000 nodos: {elapsed_time:.4f} segundos\n")
            f2.flush()  # Forzamos la escritura al archivo
            f2.close()  # Cerramos el archivo para permitir su lectura mientras se sigue ejecutando el código
            nodes_logged_10000 = True  # Marcamos como completado
            
        k = g.degree[node]
        if k>1:
            for neighbour in g[node]:
                node_counter += 1  # Contador de nodos procesados
                w = float(g[node][neighbour]['weight'])
                # if strength[node] == 0:
                #     strength[node] = 0.0001
                p_ij = w/strength[node]
                alpha_ij = (1-p_ij)**(k-1)
                if 'p_value' in g[node][neighbour]:
                    if alpha_ij < g[node][neighbour]['p_value']:
                        g[node][neighbour]['p_value'] = alpha_ij
                else:
                    g[node][neighbour]['p_value'] = alpha_ij
    return Backbone(g, method_name="Disparity Filter", property_name="p_value", ascending=True, compatible_filters=[threshold_filter, fraction_filter], filter_on='Edges')
