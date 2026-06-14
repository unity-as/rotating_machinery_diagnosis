import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from omegaconf import OmegaConf
from models.classifier import Classifier


def evaluate(model, X, y, device):
    model.eval()
    dataset = TensorDataset(torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.long))
    loader = DataLoader(dataset, batch_size=64, shuffle=False)
    all_preds, all_labels, all_probs = [], [], []
    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch = X_batch.to(device)
            outputs = model(X_batch)
            probs = torch.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y_batch.numpy())
            all_probs.extend(probs.cpu().numpy())
    return np.array(all_preds), np.array(all_labels), np.array(all_probs)


def print_confusion(preds, labels, class_names):
    cm = confusion_matrix(labels, preds)
    print(f"\n{'':>12}", end="")
    for name in class_names:
        print(f"{name:>10}", end="")
    print()
    for i, name in enumerate(class_names):
        print(f"{name:>12}", end="")
        for j in range(len(class_names)):
            print(f"{cm[i][j]:>10}", end="")
        print()


def main():
    cfg = OmegaConf.load("config/default.yaml")
    config_dict = OmegaConf.to_container(cfg, resolve=True)
    device = torch.device(config_dict['device'] if torch.cuda.is_available() else 'cpu')

    CLASS_NAMES = ['normal', 'inner', 'outer', 'combine', 'roll']

    X = np.load(os.path.join(config_dict['output_dir'], 'fused_features.npy'))
    with open(os.path.join(config_dict['output_dir'], 'label_order.txt')) as f:
        labels = [l.strip() for l in f]
    class_map = {n: i for i, n in enumerate(dict.fromkeys(labels))}
    y = np.array([class_map[l] for l in labels])

    train_idx, test_idx = train_test_split(
        np.arange(len(X)), test_size=config_dict['test_ratio'],
        random_state=config_dict['random_state'], stratify=y
    )
    X_train, y_train = X[train_idx], y[train_idx]
    X_test, y_test = X[test_idx], y[test_idx]

    model = Classifier(X_train.shape[1], len(CLASS_NAMES), config_dict).to(device)
    model.load_state_dict(torch.load(
        os.path.join(config_dict['output_dir'], config_dict['save_best_model']),
        map_location=device
    ))

    pred_train, label_train, probs_train = evaluate(model, X_train, y_train, device)
    pred_test, label_test, probs_test = evaluate(model, X_test, y_test, device)

    acc_train = accuracy_score(label_train, pred_train)
    acc_test = accuracy_score(label_test, pred_test)

    print("=" * 60)
    print(f"训练集准确率: {acc_train*100:.2f}% ({len(X_train)} 样本)")
    print("=" * 60)
    print_confusion(pred_train, label_train, CLASS_NAMES)

    print(f"\n{'=' * 60}")
    print(f"测试集准确率: {acc_test*100:.2f}% ({len(X_test)} 样本)")
    print("=" * 60)
    print_confusion(pred_test, label_test, CLASS_NAMES)

    print(f"\n{'=' * 60}")
    print("每类别详细对比")
    print("=" * 60)
    print(f"\n{'类别':<10} {'训练集':>10} {'测试集':>10}")
    print("-" * 35)
    for i, name in enumerate(CLASS_NAMES):
        train_correct = np.sum((pred_train == i) & (label_train == i))
        train_total = np.sum(label_train == i)
        test_correct = np.sum((pred_test == i) & (label_test == i))
        test_total = np.sum(label_test == i)
        train_acc = train_correct / train_total * 100
        test_acc = test_correct / test_total * 100
        print(f"{name:<10} {train_acc:>9.1f}% {test_acc:>9.1f}%")

    print(f"\n{'=' * 60}")
    print("训练集错误样本分析")
    print("=" * 60)
    wrong_train = np.where(pred_train != label_train)[0]
    if len(wrong_train) == 0:
        print("训练集无错误样本")
    else:
        print(f"错误数: {len(wrong_train)}")
        for idx in wrong_train[:10]:
            true_cls = CLASS_NAMES[label_train[idx]]
            pred_cls = CLASS_NAMES[pred_train[idx]]
            conf = probs_train[idx][pred_train[idx]]
            print(f"  样本{idx}: 真实={true_cls}, 预测={pred_cls}, 置信度={conf:.4f}")

    print(f"\n{'=' * 60}")
    print("测试集错误样本分析")
    print("=" * 60)
    wrong_test = np.where(pred_test != label_test)[0]
    if len(wrong_test) == 0:
        print("测试集无错误样本")
    else:
        print(f"错误数: {len(wrong_test)}")
        for idx in wrong_test[:10]:
            true_cls = CLASS_NAMES[label_test[idx]]
            pred_cls = CLASS_NAMES[pred_test[idx]]
            conf = probs_test[idx][pred_test[idx]]
            print(f"  样本{idx}: 真实={true_cls}, 预测={pred_cls}, 置信度={conf:.4f}")


if __name__ == "__main__":
    main()
