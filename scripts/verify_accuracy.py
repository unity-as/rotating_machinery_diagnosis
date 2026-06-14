import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import random
from collections import defaultdict
from omegaconf import OmegaConf
from models.classifier import Classifier
from models.pvdbow import load_pvdbow_model
from graph.visibility_graph import build_delayed_weighted_visibility_graph
from graph.line_graph import build_weighted_line_graph
from graph.word_generator import graph_to_words


def main():
    cfg = OmegaConf.load("config/default.yaml")
    config_dict = OmegaConf.to_container(cfg, resolve=True)
    device = torch.device(config_dict['device'] if torch.cuda.is_available() else 'cpu')

    CLASS_NAMES = ['normal', 'inner', 'outer', 'combine', 'roll']
    CLASS_LABELS = {
        'normal': '正常', 'inner': '内圈故障', 'outer': '外圈故障',
        'combine': '复合故障', 'roll': '滚动体故障'
    }

    raw_dir = config_dict['raw_data_dir']
    win = config_dict['window_size']
    num_samples = 50

    print(f"设备: {device}")
    print(f"每类采样: {num_samples} 个窗口")
    print(f"窗口大小: {win}")
    print()

    print("加载模型...")
    fused_path = os.path.join(config_dict['output_dir'], 'fused_features.npy')
    X_ref = np.load(fused_path)
    input_dim = X_ref.shape[1]

    model = Classifier(input_dim, len(CLASS_NAMES), config_dict).to(device)
    model_path = os.path.join(config_dict['output_dir'], config_dict['save_best_model'])
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    dv_G = load_pvdbow_model(os.path.join(config_dict['output_dir'], 'pvdbow_G.model'))
    dv_LG = load_pvdbow_model(os.path.join(config_dict['output_dir'], 'pvdbow_LG.model'))
    print("模型加载完成\n")

    all_preds = []
    all_labels = []
    all_probs = []
    class_correct = defaultdict(int)
    class_total = defaultdict(int)

    for cls in CLASS_NAMES:
        print(f"测试类别: {cls} ({CLASS_LABELS[cls]})")
        cls_files = [f for f in os.listdir(raw_dir) if cls in f.lower() and f.endswith('.npy')]
        if not cls_files:
            print(f"  未找到 {cls} 的数据文件，跳过")
            continue

        correct = 0
        total = 0
        for _ in range(num_samples):
            fname = random.choice(cls_files)
            sig = np.load(os.path.join(raw_dir, fname), mmap_mode='r')
            if len(sig) < win:
                continue
            start = random.randint(0, len(sig) - win)
            window = sig[start:start+win]

            G = build_delayed_weighted_visibility_graph(window, config_dict['tau'])
            LG = build_weighted_line_graph(G)
            wG = graph_to_words(G, config_dict)
            wLG = graph_to_words(LG, config_dict)

            vec_G = dv_G.infer_vector(wG, epochs=30).reshape(1, -1)
            vec_LG = dv_LG.infer_vector(wLG, epochs=30).reshape(1, -1)
            fused = np.concatenate([vec_G, vec_LG], axis=1).astype(np.float32)

            with torch.no_grad():
                x = torch.tensor(fused, dtype=torch.float32).to(device)
                output = model(x)
                probs = torch.softmax(output, dim=1).cpu().numpy()[0]

            pred_idx = int(np.argmax(probs))
            pred_cls = CLASS_NAMES[pred_idx]

            all_preds.append(pred_cls)
            all_labels.append(cls)
            all_probs.append(probs)
            total += 1
            if pred_cls == cls:
                correct += 1
                class_correct[cls] += 1
            class_total[cls] += 1

        acc = correct / total * 100 if total > 0 else 0
        print(f"  正确: {correct}/{total}, 准确率: {acc:.1f}%")

    print("\n" + "=" * 60)
    print("汇总结果")
    print("=" * 60)

    total_correct = sum(class_correct.values())
    total_samples = sum(class_total.values())
    overall_acc = total_correct / total_samples * 100 if total_samples > 0 else 0

    print(f"\n{'类别':<10} {'正确':<8} {'总数':<8} {'准确率':<10}")
    print("-" * 40)
    for cls in CLASS_NAMES:
        c = class_correct.get(cls, 0)
        t = class_total.get(cls, 0)
        acc = c / t * 100 if t > 0 else 0
        print(f"{CLASS_LABELS[cls]:<10} {c:<8} {t:<8} {acc:.1f}%")
    print("-" * 40)
    print(f"{'总计':<10} {total_correct:<8} {total_samples:<8} {overall_acc:.1f}%")

    print("\n混淆矩阵:")
    print(f"{'':>12}", end="")
    for cls in CLASS_NAMES:
        print(f"{CLASS_LABELS[cls]:>8}", end="")
    print()
    for true_cls in CLASS_NAMES:
        print(f"{CLASS_LABELS[true_cls]:>12}", end="")
        for pred_cls in CLASS_NAMES:
            count = sum(1 for p, l in zip(all_preds, all_labels) if p == pred_cls and l == true_cls)
            print(f"{count:>8}", end="")
        print()


if __name__ == "__main__":
    main()
