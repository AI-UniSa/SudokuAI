import torch.nn as nn
import torch as t
from abc import ABC, abstractmethod


class Helper(nn.Module, ABC):
    """
        Abstract class to help managing the network's output
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    @abstractmethod
    def activate(self, input: t.Tensor) -> t.Tensor:
        """
            This method manages the activation of a network's output
        """
        raise NotImplementedError(
            f'{type(self)}\'s activate method has not been implemented yet')

    @abstractmethod
    def evaluate(self, input: t.Tensor, target: t.Tensor) -> t.Tensor:
        """
            This method manages the evaluation of a network's loss
        """
        raise NotImplementedError(
            f'{type(self)}\'s evaluate method has not been implemented yet')
    
    @abstractmethod
    def extract(self, input: t.Tensor) -> t.Tensor:
        """
            This method extracts the output of the network from the ACTIVATED output
        """
        raise NotImplementedError(
            f'{type(self)}\'s extract method has not been implemented yet')
