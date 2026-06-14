import cupy as cp
import cupyx.scipy.sparse as cpx_sparse
import numpy as np
from scipy.sparse import csr_matrix


def gpu_apsp(adj_cpu, batch_size=256):
    n = adj_cpu.shape[0]
    coo = adj_cpu.tocoo()
    A = cpx_sparse.csr_matrix(
        (cp.ones(coo.nnz, dtype=cp.float32),
         (cp.asarray(coo.row, dtype=cp.int32),
          cp.asarray(coo.col, dtype=cp.int32))),
        shape=(n, n)
    )
    dist = cp.full((n, n), cp.inf, dtype=cp.float32)
    dist[cp.arange(n), cp.arange(n)] = 0.0

    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        num = end - start
        sources = cp.arange(start, end)
        cur = cp.zeros((num, n), dtype=cp.float32)
        cur[cp.arange(num), sources] = 1.0
        d_batch = cp.full((num, n), cp.inf, dtype=cp.float32)
        d_batch[cp.arange(num), sources] = 0.0
        step = 1
        while True:
            nxt = A.dot(cur.T).T
            nxt = nxt * (d_batch == cp.inf)
            mask = (nxt > 0)
            if not cp.any(mask):
                break
            d_batch[mask] = step
            cur = mask.astype(cp.float32)
            step += 1
            if step > n:
                break
        dist[start:end] = d_batch

    return dist


def gpu_apsp_batch(adj_list, batch_size=256):
    results = []
    for adj_cpu in adj_list:
        dist = gpu_apsp(adj_cpu, batch_size)
        results.append(cp.asnumpy(dist))
    cp.get_default_memory_pool().free_all_blocks()
    return results


def gpu_apsp_safe(adj_cpu, max_batch_gpu_mb=512):
    n = adj_cpu.shape[0]
    bytes_per_element = 4
    max_bytes = max_batch_gpu_mb * 1024 * 1024
    max_batch = max(1, int(max_bytes / (n * bytes_per_element)))
    batch_size = min(256, max_batch)
    result = gpu_apsp(adj_cpu, batch_size=batch_size)
    return cp.asnumpy(result)
