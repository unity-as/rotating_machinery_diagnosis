import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import networkx as nx
from omegaconf import OmegaConf
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import gc
from tqdm import tqdm
from scipy.sparse import csr_matrix
import pickle

from data.dataset import collect_windows_per_class
from graph.visibility_graph import build_delayed_weighted_visibility_graph
from graph.line_graph import build_weighted_line_graph
from utils.gpu_apsp import gpu_apsp_safe


FEATURE_NAMES = [
    'avg_deg', 'max_deg', 'avg_path', 'diameter', 'clustering',
    'deg_median', 'deg_std', 'deg_skew', 'deg_kurt',
    'p10', 'p25', 'p75', 'p90'
]


def compute_graph_features(G):
    n = G.number_of_nodes()
    if n == 0:
        return None

    degrees = np.array([d for _, d in G.degree()], dtype=np.float64)
    avg_deg = np.mean(degrees)
    max_deg = np.max(degrees)

    try:
        clustering = nx.average_clustering(G)
    except:
        clustering = 0.0

    avg_path, diameter = 0.0, 0.0
    if n > 1:
        target = G if nx.is_connected(G) else G.subgraph(max(nx.connected_components(G), key=len))
        tn = target.number_of_nodes()
        if tn > 1:
            rows, cols = [], []
            for u, v in target.edges():
                rows.extend([u, v])
                cols.extend([v, u])
            if rows:
                nodes = list(target.nodes())
                n2i = {node: i for i, node in enumerate(nodes)}
                mapped_rows = [n2i[r] for r in rows]
                mapped_cols = [n2i[c] for c in cols]
                adj = csr_matrix((np.ones(len(mapped_rows), dtype=np.float32),
                                  (mapped_rows, mapped_cols)), shape=(tn, tn))
                try:
                    dist = gpu_apsp_safe(adj)
                    finite = np.isfinite(dist) & (dist > 0)
                    if np.any(finite):
                        avg_path = np.mean(dist[finite])
                        diameter = np.max(dist[finite])
                except Exception as e:
                    print(f"    GPU 最短路径失败: {e}")

    deg_median = np.median(degrees)
    deg_std = np.std(degrees)
    if deg_std > 0:
        deg_skew = np.mean((degrees - avg_deg)**3) / (deg_std**3)
        deg_kurt = np.mean((degrees - avg_deg)**4) / (deg_std**4)
    else:
        deg_skew = deg_kurt = 0.0
    p10, p25, p75, p90 = np.percentile(degrees, [10, 25, 75, 90])

    return np.array([
        avg_deg, max_deg, avg_path, diameter, clustering,
        deg_median, deg_std, deg_skew, deg_kurt,
        p10, p25, p75, p90
    ], dtype=np.float64)


def process_window(args):
    import cupy as cp
    cp.cuda.Device().use()
    class_name, idx, seg, tau = args
    fea_G = fea_LG = None
    try:
        G = build_delayed_weighted_visibility_graph(seg, tau)
        LG = build_weighted_line_graph(G)
        fea_G = compute_graph_features(G)
        fea_LG = compute_graph_features(LG)
    except Exception as e:
        print(f"警告：窗口 {class_name}_{idx} 失败: {e}")
    finally:
        try:
            del G, LG
        except:
            pass
        gc.collect()
        cp.get_default_memory_pool().free_all_blocks()
        cp.cuda.Device().synchronize()
    return class_name, idx, fea_G, fea_LG


def main():
    cfg = OmegaConf.load("config/default.yaml")
    config = OmegaConf.to_container(cfg, resolve=True)
    raw_dir = config['raw_data_dir']
    tau = config.get('tau', 2)

    SAMPLES_PER_CLASS = 50
    NUM_WORKERS = config.get('num_workers', 4)
    class_map = {'normal': 0, 'inner': 1, 'outer': 2, 'combine': 3, 'roll': 4}
    class_names = list(class_map.keys())

    print("=" * 60)
    print("图特征提取调试脚本")
    print("=" * 60)
    print(f"特征维度: {len(FEATURE_NAMES)}")
    print(f"特征名称: {FEATURE_NAMES}")
    print(f"每类采样: {SAMPLES_PER_CLASS} 窗口")
    print()

    print("采样窗口...")
    windows_per_class = collect_windows_per_class(
        class_map, raw_dir, config['window_size'],
        SAMPLES_PER_CLASS, config['random_state']
    )

    tasks = []
    for cls in class_names:
        for i, seg in enumerate(windows_per_class[cls]):
            tasks.append((cls, i, seg, tau))

    print(f"总窗口数: {len(tasks)}，并行进程数: {NUM_WORKERS}")
    results = {cls: {'G': [], 'LG': []} for cls in class_names}

    import multiprocessing as mp
    with mp.Pool(processes=NUM_WORKERS) as pool:
        for class_name, idx, fg, flg in tqdm(
            pool.imap_unordered(process_window, tasks), total=len(tasks),
            desc="特征提取"):
            if fg is not None:
                results[class_name]['G'].append((idx, fg))
            if flg is not None:
                results[class_name]['LG'].append((idx, flg))

    X_G, y_G = [], []
    X_LG, y_LG = [], []
    for cls in class_names:
        items_G = sorted(results[cls]['G'], key=lambda x: x[0])
        items_LG = sorted(results[cls]['LG'], key=lambda x: x[0])
        for (_, fg), (_, flg) in zip(items_G, items_LG):
            X_G.append(fg)
            y_G.append(class_map[cls])
            X_LG.append(flg)
            y_LG.append(class_map[cls])

    X_G = np.array(X_G)
    y_G = np.array(y_G)
    X_LG = np.array(X_LG)
    y_LG = np.array(y_LG)

    print()
    print("=" * 60)
    print("特征统计")
    print("=" * 60)
    print(f"原图特征 shape: {X_G.shape}")
    print(f"线图特征 shape: {X_LG.shape}")
    print()

    print("原图特征均值:")
    for i, name in enumerate(FEATURE_NAMES):
        print(f"  {name:15s}: {np.mean(X_G[:, i]):.4f} ± {np.std(X_G[:, i]):.4f}")
    print()

    print("线图特征均值:")
    for i, name in enumerate(FEATURE_NAMES):
        print(f"  {name:15s}: {np.mean(X_LG[:, i]):.4f} ± {np.std(X_LG[:, i]):.4f}")
    print()

    has_nan_G = np.any(np.isnan(X_G))
    has_nan_LG = np.any(np.isnan(X_LG))
    has_inf_G = np.any(np.isinf(X_G))
    has_inf_LG = np.any(np.isinf(X_LG))
    print(f"原图 NaN: {has_nan_G}, Inf: {has_inf_G}")
    print(f"线图 NaN: {has_nan_LG}, Inf: {has_inf_LG}")
    print()

    output_dir = config['output_dir']
    os.makedirs(output_dir, exist_ok=True)

    np.save(os.path.join(output_dir, 'graph_features_G.npy'), X_G)
    np.save(os.path.join(output_dir, 'graph_features_LG.npy'), X_LG)
    print(f"特征已保存至 {output_dir}")

    scaler_G = StandardScaler()
    scaler_LG = StandardScaler()
    X_G_scaled = scaler_G.fit_transform(X_G)
    X_LG_scaled = scaler_LG.fit_transform(X_LG)
    X_Both = np.concatenate([X_G_scaled, X_LG_scaled], axis=1)

    perplexity = min(30, len(y_G) - 1)
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    datasets = [
        (X_G_scaled, y_G, 'Original Graph Features (standardized)'),
        (X_LG_scaled, y_LG, 'Line Graph Features (standardized)'),
        (X_Both, y_G, 'Combined Features (standardized + concatenated)')
    ]
    for ax, (X, y, title) in zip(axes, datasets):
        print(f"t-SNE: {title} ...")
        tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity, max_iter=600)
        X_2d = tsne.fit_transform(X)
        for i, name in enumerate(class_names):
            ax.scatter(X_2d[y == i, 0], X_2d[y == i, 1],
                       c=colors[i], label=name, s=10, alpha=0.6)
        ax.set_title(title)
        ax.legend(markerscale=4)
        ax.grid(alpha=0.2)

    plt.tight_layout()
    save_path = os.path.join(output_dir, 'graph_features_tsne.png')
    plt.savefig(save_path, dpi=200)
    plt.close()
    print(f"完成！图片已保存为 {save_path}")


if __name__ == "__main__":
    import multiprocessing as mp
    mp.freeze_support()
    main()
