import torch.nn as nn


class Classifier(nn.Module):
    def __init__(self, input_dim, num_classes, cfg):
        super().__init__()
        hidden_dims = cfg['classifier_hidden_dims']
        activation = cfg['classifier_activation']
        use_bn = cfg['classifier_use_bn']
        dropout_rate = cfg['dropout_rate']

        layers = []
        prev_dim = input_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, h_dim))
            if use_bn:
                layers.append(nn.BatchNorm1d(h_dim))
            if activation == 'relu':
                layers.append(nn.ReLU())
            elif activation == 'tanh':
                layers.append(nn.Tanh())
            elif activation == 'leaky_relu':
                layers.append(nn.LeakyReLU())
            else:
                layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
            prev_dim = h_dim
        layers.append(nn.Linear(prev_dim, num_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)
