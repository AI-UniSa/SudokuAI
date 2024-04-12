import torch.nn as nn
import torch as t


from models.utils.MLP import MLP
from models.utils.helper import Helper
from models.utils import constants as c


class Sudoku_MLP(nn.Module):
    def __init__(self, inner_size: int = 5, num_layers=6, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        input_layers_num = int(num_layers/2)
        output_layers_num = num_layers-input_layers_num

        self._input_layer = MLP(c.INPUT_SIZE, inner_size, input_layers_num)
        self._output_layer = MLP(inner_size, c.OUTPUT_SIZE, output_layers_num)

    def forward(self, x: t.Tensor):
        x = self._input_layer(x)
        x = self._output_layer(x)
        return x
    


class Sudoku_MLP_helper(Helper):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self._loss = nn.MSELoss(reduction='sum')

    def activate(self, input: t.Tensor) -> t.Tensor:
        return input

    def evaluate(self, input: t.Tensor, target: t.Tensor) -> t.Tensor:
        return self._loss(input, target)


if __name__ == '__main__':
    model = Sudoku_MLP()
    model.to('cuda')
    for param in model.parameters():
        print(param.is_cuda)
        break

    crit = Sudoku_MLP_helper()
    print(type(crit))
    crit.cuda()
