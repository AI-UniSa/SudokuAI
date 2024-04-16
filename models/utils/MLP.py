import torch.nn as nn
import torch
import numpy as np
from collections import OrderedDict

class MLP(nn.Module):
    def __init__(self, in_size: int, out_size: int, num_layers: int = 3, dropout: list  = []) -> None:
        # feature: anche dropout parte con indice 0
        super(MLP, self).__init__()
        layers = OrderedDict()

        # Linearly spaced linear layers
        features=np.linspace(in_size,out_size,num_layers+1,dtype=int)
        for i in range(num_layers):

            layers['lin_{}'.format(i)] = nn.Linear(features[i], features[i+1])
            if i != num_layers-1:
                layers['relu_{}'.format(i)] = nn.ReLU()
            if i in dropout:
                layers['drop_{}'.format(i)] = nn.Dropout(p=0.5)

        self.net = nn.Sequential(layers)

    def forward(self, x):
        x = torch.squeeze(x)
        x=self.net(x)
        return x