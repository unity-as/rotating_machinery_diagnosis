import torch
from torch.utils.data import TensorDataset


class AugmentedDataset(TensorDataset):
    def __init__(self, X, y, augment=True, noise_scale=0.01, drop_prob=0.1):
        super().__init__(torch.tensor(X, dtype=torch.float32),
                         torch.tensor(y, dtype=torch.long))
        self.augment = augment
        self.noise_scale = noise_scale
        self.drop_prob = drop_prob

    def __getitem__(self, index):
        x, y = super().__getitem__(index)
        if self.augment:
            noise = torch.randn_like(x) * self.noise_scale
            x = x + noise
            mask = torch.bernoulli(torch.ones_like(x) * (1 - self.drop_prob))
            x = x * mask
        return x, y
