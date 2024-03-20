import torch.nn as nn
import torch as t
from abc import ABC, abstractmethod


class Helper(nn.Module,ABC):
    "Abstract class to help managing the network's output"
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
    
    @abstractmethod
    def activate(self,input:t.Tensor)->t.Tensor:
        raise NotImplementedError(f'{type(self)}\'s activate method has not been implemented yet')
    
    @abstractmethod
    def evaluate(self,input:t.Tensor,target:t.Tensor)->t.Tensor:
        raise NotImplementedError(f'{type(self)}\'s evaluate method has not been implemented yet')

