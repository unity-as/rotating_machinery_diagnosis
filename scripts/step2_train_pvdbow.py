import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import time
from omegaconf import OmegaConf
from models.pvdbow import train_pvdbow


def main():
    cfg = OmegaConf.load("config/default.yaml")
    config_dict = OmegaConf.to_container(cfg, resolve=True)

    label_order_path = os.path.join(config_dict['output_dir'], 'label_order.txt')
    if not os.path.exists(label_order_path):
        raise RuntimeError("未找到 label_order.txt，请先运行 step1_extract_words.py")
    with open(label_order_path, 'r') as f:
        total = sum(1 for _ in f)
    print(f"总文档数: {total}")
    print(f"配置: min_count={config_dict['pvdbow_min_count']}, vector_size={config_dict['vector_size']}, epochs={config_dict['pvdbow_epochs']}")
    print()

    start_all = time.time()

    print("=" * 50)
    print("阶段1/2: 训练原图 PV-DBOW")
    print("=" * 50)
    vec_G = train_pvdbow(
        config_dict['corpus_file_G'],
        os.path.join(config_dict['output_dir'], 'pvdbow_G.model'),
        config_dict
    )
    np.save(os.path.join(config_dict['output_dir'], 'vec_G.npy'), vec_G)
    print(f"原图向量已保存, shape: {vec_G.shape}")
    print()

    print("=" * 50)
    print("阶段2/2: 训练线图 PV-DBOW")
    print("=" * 50)
    vec_LG = train_pvdbow(
        config_dict['corpus_file_LG'],
        os.path.join(config_dict['output_dir'], 'pvdbow_LG.model'),
        config_dict
    )
    np.save(os.path.join(config_dict['output_dir'], 'vec_LG.npy'), vec_LG)
    print(f"线图向量已保存, shape: {vec_LG.shape}")

    elapsed = time.time() - start_all
    print(f"\nPV-DBOW 全部完成, 总耗时 {elapsed:.0f}s ({elapsed/60:.1f}min)")
    print("请运行 step3_fuse_and_save.py 继续。")


if __name__ == "__main__":
    main()
