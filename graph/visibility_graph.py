import numpy as np
import networkx as nx
from numba import njit


@njit(cache=True)
def _compute_edges_numba(series, tau):
    n = len(series)
    y = series.astype(np.float64)
    A, B, W = [], [], []
    for a in range(n - 2):
        ya = y[a].item()
        for b in range(a + 2, n):
            yb = y[b].item()
            slope = (yb - ya) / (b - a)
            min_d = 1e308
            ok = True
            vs, ve = a + tau + 1, b - tau
            for c in range(a + tau + 1, b - tau):
                yc = y[c].item()
                yl = ya + slope * (c - a)
                d = yl - yc
                if vs <= c < ve and yc >= yl:
                    ok = False
                    break
                if d < min_d:
                    min_d = d
            if ok:
                A.append(a)
                B.append(b)
                W.append(0.0 if min_d > 1e300 else min_d)
    if not A:
        return (np.zeros(0, dtype=np.int64),
                np.zeros(0, dtype=np.int64),
                np.zeros(0, dtype=np.float64))
    return (np.array(A, dtype=np.int64),
            np.array(B, dtype=np.int64),
            np.array(W, dtype=np.float64))


def build_delayed_weighted_visibility_graph(series, tau=2):
    n = len(series)
    G = nx.Graph()
    G.add_nodes_from(range(n))
    if n < 3:
        return G
    a, b, w = _compute_edges_numba(series, tau)
    if len(a) > 0:
        G.add_weighted_edges_from(zip(a.tolist(), b.tolist(), w.tolist()))
    return G
