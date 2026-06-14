import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
import numpy as np


def plot_tsne(X, y, class_names, save_path='tsne_features.png', title='t-SNE'):
    perplexity = min(30, len(y) - 1)
    if perplexity < 1:
        return
    print(f"Running t-SNE: {title} ...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity, max_iter=600)
    X_2d = tsne.fit_transform(X_scaled)
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    plt.figure(figsize=(8, 6))
    for i, name in enumerate(class_names):
        mask = y == i
        if np.any(mask):
            plt.scatter(X_2d[mask, 0], X_2d[mask, 1], c=colors[i], label=name, s=10, alpha=0.6)
    plt.legend(markerscale=4)
    plt.title(title)
    plt.grid(alpha=0.2)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()
    print(f"t-SNE 图已保存: {save_path}")


def plot_tsne_comparison(datasets, class_names, save_path='tsne_comparison.png'):
    n = len(datasets)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 5))
    if n == 1:
        axes = [axes]
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    perplexity = min(30, min(d[1].shape[0] for d in datasets) - 1)
    for ax, (X, y, title) in zip(axes, datasets):
        print(f"t-SNE: {title} ...")
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity, max_iter=600)
        X_2d = tsne.fit_transform(X_scaled)
        for i, name in enumerate(class_names):
            mask = y == i
            if np.any(mask):
                ax.scatter(X_2d[mask, 0], X_2d[mask, 1], c=colors[i], label=name, s=10, alpha=0.6)
        ax.set_title(title)
        ax.legend(markerscale=4)
        ax.grid(alpha=0.2)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()
    print(f"对比 t-SNE 图已保存: {save_path}")


def plot_training_curve(train_losses, test_accs, save_path='training_curve.png'):
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss Curve')
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(test_accs, label='Test Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Accuracy Curve')
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"训练曲线已保存: {save_path}")
