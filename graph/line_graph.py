import numpy as np
import networkx as nx
from scipy.sparse import csr_matrix


def build_weighted_line_graph(G):
    edges = list(G.edges(data=True))
    E = len(edges)
    if E == 0:
        return nx.Graph()
    nodes = list(G.nodes())
    n2i = {n: i for i, n in enumerate(nodes)}
    weights = np.zeros(E)
    rows, cols = [], []
    for i, (u, v, d) in enumerate(edges):
        weights[i] = d['weight']
        rows.extend([n2i[u], n2i[v]])
        cols.extend([i, i])
    B = csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(len(nodes), E))
    A = B.T @ B
    A.setdiag(0)
    A.eliminate_zeros()
    LG = nx.Graph()
    LG.add_nodes_from(range(E))
    coo = A.tocoo()
    if len(coo.row) > 0:
        ew = (weights[coo.row] + weights[coo.col]) / 2.0
        LG.add_weighted_edges_from(zip(coo.row.tolist(), coo.col.tolist(), ew.tolist()))
    for i, w in enumerate(weights):
        LG.nodes[i]['weight'] = w
    return LG
