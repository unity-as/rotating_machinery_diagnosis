import numpy as np
from scipy.sparse import csr_matrix
from utils.gpu_apsp import gpu_apsp

def compute_ndd_words(G, cfg):
    """
    使用节点的度数作为标签。
    """
    delta = cfg['ndd_delta']
    thresholds = cfg['ndd_thresholds']
    n_nodes = G.number_of_nodes()
    if n_nodes <= 1:
        return []

    nodes = list(G.nodes())
    # 计算每个节点的度数
    degree_map = {node: G.degree(node) for node in nodes}
    # 建立节点到索引的映射（用于距离矩阵索引）
    n2i = {n: i for i, n in enumerate(nodes)}
    rows, cols = [], []
    for u, v in G.edges():
        i, j = n2i[u], n2i[v]
        rows.extend([i, j])
        cols.extend([j, i])
    if not rows:
        return []
    adj_cpu = csr_matrix((np.ones(len(rows), dtype=np.float32), (rows, cols)), shape=(n_nodes, n_nodes))
    dist = gpu_apsp(adj_cpu)
    if hasattr(dist, 'get'):
        dist = dist.get()
    np.fill_diagonal(dist, np.inf)
    finite = dist[np.isfinite(dist)]
    if len(finite) == 0:
        return []
    max_d = max(int(np.max(finite)), 1)
    bins = np.arange(0, max_d + delta, delta)
    nbins = len(bins) - 1

    probs = np.zeros((n_nodes, nbins))
    for idx in range(n_nodes):
        valid = dist[idx][np.isfinite(dist[idx])]
        if len(valid) > 0:
            probs[idx] = np.histogram(valid, bins=bins)[0] / (n_nodes - 1)

    words = []
    for idx in range(n_nodes):
        node = nodes[idx]
        deg = degree_map[node]                 # 度数
        sorted_bins = np.argsort(-probs[idx])
        for th in thresholds:
            selected = []
            for bin_idx in sorted_bins:
                if probs[idx][bin_idx] >= th:
                    selected.append(str(int(bins[bin_idx])))
                else:
                    break
            if selected:
                word = "NDD_" + str(deg) + "_" + "_".join(selected)
                words.append(word)
    return words


def compute_tm_words(G, cfg):
    """
    使用节点的度数作为标签。
    """
    s_steps = cfg['tm_s_steps']
    thresholds = cfg['tm_thresholds']
    top_k = cfg['tm_top_k']
    nodes = list(G.nodes())
    n = len(nodes)
    if n == 0:
        return []
    degree_map = {node: G.degree(node) for node in nodes}
    n2i = {n: i for i, n in enumerate(nodes)}
    row, col, data = [], [], []
    for u, v, w in G.edges(data='weight'):
        i, j = n2i[u], n2i[v]
        row.extend([i, j])
        col.extend([j, i])
        data.extend([w, w])
    W = csr_matrix((data, (row, col)), shape=(n, n))
    rs = np.array(W.sum(axis=1)).flatten()
    rs[rs == 0] = 1.0
    P = csr_matrix((1.0 / rs, (range(n), range(n))), shape=(n, n)) @ W

    words = []
    for s in s_steps:
        Ps = P
        for _ in range(s - 1):
            Ps = Ps @ P
        Pd = Ps.toarray()
        for i in range(n):
            src_node = nodes[i]
            src_deg = degree_map[src_node]
            sorted_targets = np.argsort(-Pd[i])[:top_k]
            for th in thresholds:
                selected = []
                for j in sorted_targets:
                    if Pd[i][j] >= th:
                        tgt_node = nodes[j]
                        tgt_deg = degree_map[tgt_node]
                        selected.append(str(tgt_deg))
                    else:
                        break
                if selected:
                    word = f"TM{s}_{src_deg}_" + "_".join(selected)
                    words.append(word)
    return words


def graph_to_words(G, cfg):
    return compute_ndd_words(G, cfg) + compute_tm_words(G, cfg)
