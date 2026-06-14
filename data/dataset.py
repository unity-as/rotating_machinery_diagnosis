import numpy as np
import random
import os


def load_and_segment(fp, win=800, max_seg=None):
    sig = np.load(fp)
    if max_seg is None:
        return []
    n = len(sig)
    if n < win:
        return []
    segs = []
    for _ in range(max_seg):
        start = random.randint(0, n - win)
        segs.append(sig[start:start+win])
    return segs


def collect_windows_per_class(class_map, raw_data_dir, window_size, num_windows_per_class, random_state):
    random.seed(random_state)
    np.random.seed(random_state)

    class_files = {cls: [] for cls in class_map}
    for fname in os.listdir(raw_data_dir):
        if not fname.endswith('.npy'):
            continue
        for cls in class_map:
            if cls in fname.lower():
                class_files[cls].append(os.path.join(raw_data_dir, fname))
                break

    windows_per_class = {}
    for cls, files in class_files.items():
        if not files:
            print(f"警告：类别 '{cls}' 没有找到任何文件，将采样 0 个窗口")
            windows_per_class[cls] = []
            continue

        valid_files = []
        for fp in files:
            try:
                mmap = np.load(fp, mmap_mode='r')
                if mmap.shape[0] >= window_size:
                    valid_files.append(fp)
                else:
                    print(f"警告：文件 {fp} 长度 {mmap.shape[0]} < 窗口大小 {window_size}，跳过")
            except Exception as e:
                print(f"警告：无法读取文件 {fp}，跳过。错误: {e}")

        if not valid_files:
            print(f"类别 '{cls}' 没有长度足够的文件，采样 0 个窗口")
            windows_per_class[cls] = []
            continue

        segs = []
        for _ in range(num_windows_per_class):
            idx = random.randint(0, len(valid_files) - 1)
            fp = valid_files[idx]
            sig = np.load(fp, mmap_mode='r')
            max_start = len(sig) - window_size
            if max_start < 0:
                continue
            start = random.randint(0, max_start)
            seg = sig[start:start+window_size].copy()
            segs.append(seg)
        windows_per_class[cls] = segs

    return windows_per_class
