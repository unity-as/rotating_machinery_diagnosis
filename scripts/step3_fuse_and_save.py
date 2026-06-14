import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from omegaconf import OmegaConf
from utils.visualizer import plot_tsne, plot_tsne_comparison


def main():
    cfg = OmegaConf.load("config/default.yaml")
    config_dict = OmegaConf.to_container(cfg, resolve=True)

    vec_G_path = os.path.join(config_dict['output_dir'], 'vec_G.npy')
    vec_LG_path = os.path.join(config_dict['output_dir'], 'vec_LG.npy')

    if not os.path.exists(vec_G_path) or not os.path.exists(vec_LG_path):
        raise RuntimeError("未找到 vec_G.npy 或 vec_LG.npy，请先运行 step2_train_pvdbow.py")

    vec_G = np.load(vec_G_path)
    vec_LG = np.load(vec_LG_path)

    print(f"原图向量 shape: {vec_G.shape}, 线图向量 shape: {vec_LG.shape}")

    fused = np.concatenate([vec_G, vec_LG], axis=1)
    print(f"融合向量 shape: {fused.shape}")

    fused_path = os.path.join(config_dict['output_dir'], 'fused_features.npy')
    np.save(fused_path, fused.astype(np.float32))
    print(f"融合特征已保存: {fused_path}")

    if config_dict.get('run_tsne', False):
        label_order_path = os.path.join(config_dict['output_dir'], 'label_order.txt')
        if os.path.exists(label_order_path):
            with open(label_order_path, 'r') as f:
                labels = [line.strip() for line in f]
            class_names = list(dict.fromkeys(labels))
            class_map = {name: i for i, name in enumerate(class_names)}
            y = np.array([class_map[l] for l in labels])

            datasets = [
                (vec_G, y, 'Original Graph Vectors'),
                (vec_LG, y, 'Line Graph Vectors'),
                (fused, y, 'Fused Vectors')
            ]
            plot_tsne_comparison(
                datasets, class_names,
                save_path=os.path.join(config_dict['output_dir'], 'tsne_pvdbow.png')
            )

    print("特征拼接完成。请运行 step4_train_classifier.py 继续。")


if __name__ == "__main__":
    main()
