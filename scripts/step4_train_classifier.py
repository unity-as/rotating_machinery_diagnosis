import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from omegaconf import OmegaConf

from models.classifier import Classifier
from data.augment import AugmentedDataset
from utils.visualizer import plot_tsne, plot_training_curve


def main():
    cfg = OmegaConf.load("config/default.yaml")
    config_dict = OmegaConf.to_container(cfg, resolve=True)
    device = torch.device(config_dict['device'] if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    fused_path = os.path.join(config_dict['output_dir'], 'fused_features.npy')
    label_order_path = os.path.join(config_dict['output_dir'], 'label_order.txt')

    if not os.path.exists(fused_path):
        raise RuntimeError("未找到 fused_features.npy，请先运行 step3_fuse_and_save.py")

    X = np.load(fused_path)
    with open(label_order_path, 'r') as f:
        labels = [line.strip() for line in f]

    class_names = list(dict.fromkeys(labels))
    class_map = {name: i for i, name in enumerate(class_names)}
    y = np.array([class_map[l] for l in labels])
    num_classes = len(class_names)

    indices = np.arange(len(X))
    train_idx, test_idx = train_test_split(
        indices, test_size=config_dict['test_ratio'],
        random_state=config_dict['random_state'], stratify=y
    )
    X_train, y_train = X[train_idx], y[train_idx]
    X_test, y_test = X[test_idx], y[test_idx]

    rng = np.random.RandomState(config_dict['random_state'])
    train_shuffle = rng.permutation(len(X_train))
    X_train, y_train = X_train[train_shuffle], y_train[train_shuffle]
    test_shuffle = rng.permutation(len(X_test))
    X_test, y_test = X_test[test_shuffle], y_test[test_shuffle]

    input_dim = X_train.shape[1]
    print(f"特征维度: {input_dim}")
    print(f"训练样本: {len(X_train)}, 测试样本: {len(X_test)}")

    if config_dict.get('run_tsne', False):
        all_X = np.concatenate([X_train, X_test], axis=0)
        all_y = np.concatenate([y_train, y_test], axis=0)
        plot_tsne(all_X, all_y, class_names,
                  save_path=os.path.join(config_dict['output_dir'], 'tsne_features.png'),
                  title='Fused Features t-SNE')

    train_dataset = AugmentedDataset(
        X_train, y_train,
        augment=config_dict['use_augmentation'],
        noise_scale=config_dict['noise_scale'],
        drop_prob=config_dict['drop_prob']
    )
    test_dataset = TensorDataset(
        torch.tensor(X_test, dtype=torch.float32),
        torch.tensor(y_test, dtype=torch.long)
    )
    train_loader = DataLoader(train_dataset, batch_size=config_dict['classifier_batch_size'], shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=config_dict['classifier_batch_size'], shuffle=False)

    model = Classifier(input_dim, num_classes, config_dict).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config_dict['classifier_lr'], weight_decay=config_dict['weight_decay'])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=10)

    model.eval()
    init_preds, init_labels = [], []
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch = X_batch.to(device)
            outputs = model(X_batch)
            _, preds = torch.max(outputs, 1)
            init_preds.extend(preds.cpu().numpy())
            init_labels.extend(y_batch.numpy())
    init_acc = accuracy_score(init_labels, init_preds)
    print(f"\n初始随机权重测试准确率: {init_acc*100:.2f}% (理论值≈20%)")
    print(f"初始预测分布: {dict(zip(*np.unique(init_preds, return_counts=True)))}")
    print(f"真实标签分布: {dict(zip(*np.unique(init_labels, return_counts=True)))}\n")

    best_acc = 0.0
    best_epoch = 0
    patience_counter = 0
    train_losses, test_accs = [], []

    for epoch in range(config_dict['classifier_epochs']):
        model.train()
        total_loss = 0.0
        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += loss.item() * X_batch.size(0)
        avg_loss = total_loss / len(train_dataset)

        model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for X_batch, y_batch in test_loader:
                X_batch = X_batch.to(device)
                y_batch = y_batch.to(device)
                outputs = model(X_batch)
                _, preds = torch.max(outputs, 1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(y_batch.cpu().numpy())
        test_acc = accuracy_score(all_labels, all_preds)
        train_losses.append(avg_loss)
        test_accs.append(test_acc)
        scheduler.step(test_acc)

        print(f"Epoch {epoch+1:3d}/{config_dict['classifier_epochs']} | Loss: {avg_loss:.4f} | Test Acc: {test_acc*100:.2f}% | Best: {best_acc*100:.2f}%")

        if test_acc > best_acc:
            best_acc = test_acc
            best_epoch = epoch + 1
            patience_counter = 0
            torch.save(model.state_dict(), os.path.join(config_dict['output_dir'], config_dict['save_best_model']))
        else:
            patience_counter += 1
            if patience_counter >= config_dict['early_stop_patience']:
                print(f"Early stopping after {epoch+1} epochs.")
                break

    model.load_state_dict(torch.load(os.path.join(config_dict['output_dir'], config_dict['save_best_model']), map_location=device))
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            outputs = model(X_batch)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y_batch.cpu().numpy())
    final_acc = accuracy_score(all_labels, all_preds)
    print(f"\nBest test accuracy: {best_acc*100:.2f}% at epoch {best_epoch}")
    print(f"Final test accuracy: {final_acc*100:.2f}%")
    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds, target_names=class_names))
    print("Confusion Matrix:")
    print(confusion_matrix(all_labels, all_preds))

    if config_dict.get('plot_curve', False):
        plot_training_curve(train_losses, test_accs,
                           save_path=os.path.join(config_dict['output_dir'], 'training_curve.png'))


if __name__ == "__main__":
    main()
