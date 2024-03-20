import torch.nn as nn
import torch
import math
from collections import OrderedDict

class MLP(nn.Module):
    def __init__(self, in_size: int, out_size: int, num_layers: int = 3, dropout: list[int]  = []) -> None:
        # feature: anche dropout parte con indice 0
        super(MLP, self).__init__()
        layers = OrderedDict()

        # Linearly spaced linear layers
        step = math.ceil((in_size-out_size)/num_layers)
        in_features = in_size
        for i in range(num_layers):
            # In order to respect the out_size requirement
            out_features = max(in_features-step, out_size)

            layers['lin_{}'.format(i)] = nn.Linear(in_features, out_features)
            if i != num_layers-1:
                layers['relu_{}'.format(i)] = nn.ReLU()
            if i in dropout:
                layers['drop_{}'.format(i)] = nn.Dropout(p=0.5)
            in_features -= step

        self.net = nn.Sequential(layers)

    def forward(self, x):
        x = torch.squeeze(x)
        return self.net(x)