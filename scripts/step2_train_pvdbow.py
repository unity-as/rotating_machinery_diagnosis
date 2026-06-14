import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
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

    print("训练原图 PV-DBOW ...")
    vec_G = train_pvdbow(
        config_dict['corpus_file_G'],
        os.path.join(config_dict['output_dir'], 'pvdbow_G.model'),
        config_dict
    )
    np.save(os.path.join(config_dict['output_dir'], 'vec_G.npy'), vec_G)
    print(f"原图向量已保存, shape: {vec_G.shape}")

    print("训练线图 PV-DBOW ...")
    vec_LG = train_pvdbow(
        config_dict['corpus_file_LG'],
        os.path.join(config_dict['output_dir'], 'pvdbow_LG.model'),
        config_dict
    )
    np.save(os.path.join(config_dict['output_dir'], 'vec_LG.npy'), vec_LG)
    print(f"线图向量已保存, shape: {vec_LG.shape}")

    print("PV-DBOW 训练完成。请运行 step3_fuse_and_save.py 继续。")


if __name__ == "__main__":
    main()
