import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import gc
import warnings
import pickle
from tqdm import tqdm
from omegaconf import OmegaConf
import multiprocessing as mp

from data.dataset import collect_windows_per_class
from graph.visibility_graph import build_delayed_weighted_visibility_graph, _compute_edges_numba
from graph.line_graph import build_weighted_line_graph
from graph.word_generator import graph_to_words
from utils.io_utils import validate_and_clean_cached

warnings.filterwarnings('ignore')

try:
    import cupy
    print("CuPy 已安装，GPU 加速可用")
except ImportError:
    raise ImportError("未安装 cupy，请 pip install cupy-cuda12x")


def process_one_window(args):
    import cupy as cp
    cp.cuda.Device().use()

    class_name, idx, seg, cfg, words_dir_G, words_dir_LG = args
    G = build_delayed_weighted_visibility_graph(seg, cfg['tau'])
    LG = build_weighted_line_graph(G)

    wG = graph_to_words(G, cfg)
    _save_words(wG, class_name, idx, words_dir_G)

    wLG = graph_to_words(LG, cfg)
    _save_words(wLG, class_name, idx, words_dir_LG)

    del G, LG
    gc.collect()
    cp.get_default_memory_pool().free_all_blocks()
    cp.cuda.Device().synchronize()

    return (class_name, idx)


def _save_words(words, class_name, idx, words_dir):
    class_dir = os.path.join(words_dir, class_name)
    os.makedirs(class_dir, exist_ok=True)
    with open(os.path.join(class_dir, f'{idx}.pkl'), 'wb') as f:
        pickle.dump(words, f)


def main():
    cfg = OmegaConf.load("config/default.yaml")
    config_dict = OmegaConf.to_container(cfg, resolve=True)

    print("预热 Numba ...")
    _compute_edges_numba(np.random.rand(50).astype(np.float64), config_dict['tau'])
    print("预热完成\n")

    class_map = {'normal': 0, 'inner': 1, 'outer': 2, 'combine': 3, 'roll': 4}

    windows_per_class = collect_windows_per_class(
        class_map, config_dict['raw_data_dir'], config_dict['window_size'],
        config_dict['max_segments_per_file'], config_dict['random_state']
    )

    for d in (config_dict['words_dir_G'], config_dict['words_dir_LG']):
        os.makedirs(d, exist_ok=True)

    samples = []
    for class_name, segs in windows_per_class.items():
        for i, seg in enumerate(segs):
            samples.append((class_name, i, seg))

    total = len(samples)
    print(f"总窗口数: {total} (每个类别 {config_dict['max_segments_per_file']} 个)")

    print("正在验证已有缓存文件的完整性...")
    cached_G_set = validate_and_clean_cached(config_dict['words_dir_G'], class_map)
    cached_LG_set = validate_and_clean_cached(config_dict['words_dir_LG'], class_map)
    print(f"清理后，完好的原图缓存数: {len(cached_G_set)}，线图缓存数: {len(cached_LG_set)}")

    need_process = []
    for class_name, idx, seg in samples:
        need_G = (class_name, idx) not in cached_G_set
        need_LG = (class_name, idx) not in cached_LG_set
        if need_G or need_LG:
            need_process.append((class_name, idx, seg, config_dict,
                                 config_dict['words_dir_G'], config_dict['words_dir_LG']))

    print(f"需提取原图/线图的窗口数: {len(need_process)} / {total}")

    if need_process:
        print("\n========== 阶段1: 并行提取单词 ==========")
        num_workers = config_dict.get('num_workers', min(mp.cpu_count(), 4))
        print(f"使用 {num_workers} 个进程并行处理")

        with mp.Pool(processes=num_workers) as pool:
            for _ in tqdm(pool.imap_unordered(process_one_window, need_process),
                          total=len(need_process), desc="提取单词"):
                pass
        print("单词提取完成")
    else:
        print("所有单词已缓存，跳过提取")

    print("\n========== 构建语料库 ==========")
    label_order = []
    with open(config_dict['corpus_file_G'], 'w', encoding='utf-8') as fG, \
         open(config_dict['corpus_file_LG'], 'w', encoding='utf-8') as fLG:

        for class_name, idx, _ in samples:
            label_order.append(class_name)

            pkl_path_G = os.path.join(config_dict['words_dir_G'], class_name, f'{idx}.pkl')
            words_G = []
            if os.path.exists(pkl_path_G):
                try:
                    with open(pkl_path_G, 'rb') as pf:
                        words_G = pickle.load(pf)
                except (EOFError, pickle.UnpicklingError) as e:
                    print(f"严重警告：刚生成的文件损坏 {pkl_path_G}，错误: {e}")
            fG.write(' '.join(words_G) + '\n')

            pkl_path_LG = os.path.join(config_dict['words_dir_LG'], class_name, f'{idx}.pkl')
            words_LG = []
            if os.path.exists(pkl_path_LG):
                try:
                    with open(pkl_path_LG, 'rb') as pf:
                        words_LG = pickle.load(pf)
                except (EOFError, pickle.UnpicklingError) as e:
                    print(f"严重警告：刚生成的文件损坏 {pkl_path_LG}，错误: {e}")
            fLG.write(' '.join(words_LG) + '\n')

    label_order_path = os.path.join(config_dict['output_dir'], 'label_order.txt')
    os.makedirs(config_dict['output_dir'], exist_ok=True)
    with open(label_order_path, 'w') as f:
        for lbl in label_order:
            f.write(lbl + '\n')
    print(f"标签顺序已保存至 {label_order_path}")

    print("单词提取和语料构建完成。请运行 step2_train_pvdbow.py 继续训练。")


if __name__ == "__main__":
    mp.freeze_support()
    main()
