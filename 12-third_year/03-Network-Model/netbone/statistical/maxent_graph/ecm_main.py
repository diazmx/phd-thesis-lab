import scipy.sparse
import networkx as nx
from netbone.statistical.maxent_graph import ECM
from pandas import DataFrame
from netbone.backbone import Backbone
from netbone.filters import threshold_filter, fraction_filter
def ecm(data):
    #pip install jax tabulate
    #https://github.com/google/jax#installat

    weight = "weight"

    if isinstance(data, DataFrame):
        data = nx.from_pandas_edgelist(data, edge_attr='weight')
    else:
        data = data.copy()

    
    g = nx.convert_node_labels_to_integers(data)
    W = nx.adjacency_matrix(g, weight=weight)

    model = ECM(W)
    initial_guess = model.get_initial_guess()
    solution = model.solve(initial_guess)

    pval_M = model.get_pval_matrix(solution.x, W)
    lower_pval_M = scipy.sparse.tril(pval_M).toarray()

    for (i,j) in zip(*lower_pval_M.nonzero()):
        p = lower_pval_M[i,j]
        g[i][j]['p_value'] = p #-np.log(p)

    # def filter_edge(n1, n2):
    #     return g[n1][n2]['p_value'] > -np.log(0.10)
    
    g = nx.relabel_nodes(g, dict(zip(g.nodes(), data.nodes())))

    nx.set_edge_attributes(data, {(u,v):w for u,v,w in list(g.edges(data='p_value'))}, name='p_value')
    #subgraph = nx.subgraph_view(g)#, filter_edge=filter_edge)
    return Backbone(data, method_name="Enhanced Configuration Model Filter", property_name="p_value", ascending=True, compatible_filters=[threshold_filter, fraction_filter], filter_on='Edges')