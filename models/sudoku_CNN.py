import torch 
import torch.nn as nn
import math

from models.utils.helper import Helper


class Sudoku_CNN(nn.Module):
    def __init__(self, input_channels = 1, num_blocks=1, filters=None, kernel_size=1, activation_fn="relu", decay=0.99):
        super(Sudoku_CNN, self).__init__()
        if isinstance(filters, list) and len(filters) != num_blocks:
            raise ValueError("If filters is a list, then a number of filters must be provided for each block")
        self.num_blocks = num_blocks
        self.filters = filters
        self.kernel_size = kernel_size
        self.decay = decay
        self.activation_fn = activation_fn

        self._model = nn.ModuleList()

        for i in range(self.num_blocks):
            if i == 0:                
                filters = filters[i] if isinstance(filters, list) else filters
                self._model.append(self.create_conv_block(input_channels, filters))
            else:
                prev_filters = filters[i-1] if isinstance(filters, list) else filters
                filters = filters[i] if isinstance(filters, list) else filters
                self._model.append(self.create_conv_block(prev_filters, filters))

        # Dense layers
        self._model.append(nn.Flatten())
        # TODO: Consider to give in input also the original sudoku
        self._model.append(nn.Linear(filters * 81, 9*9*9)) # Output of last cnn block -> 9x9 grid with 9 possible values

    def create_conv_block(self, in_channels=None, out_channels=None, preserve = True):
        conv_block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=self.kernel_size, padding= math.ceil((self.kernel_size - 1)/2) if preserve else 0, groups = 1),
            nn.BatchNorm2d(out_channels, momentum=self.decay),
            nn.ReLU(inplace=True) if self.activation_fn.lower() == "relu" else nn.ReLU(inplace=True)
        )
        return conv_block

    def forward(self, x):
        # Since input is a Bx81 tensor, we need to reshape it to B x 1 x 9 x 9 to emulate an image
        x = x.view(-1, 1, 9, 9)
        # Apply CNN blocks
        for block in self._model[:-2]:
            x = block(x)

        # Flatten the output
        x = self._model[-2](x)
        # Dense layers
        x = self._model[-1](x)

        return x.reshape(x.shape[0], 9, -1) # B x C x 81
    
class Sudoku_CNN_helper(Helper):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self._loss = nn.CrossEntropyLoss()
        self._flatten = nn.Flatten()

    def activate(self, input: torch.Tensor) -> torch.Tensor:
        return torch.softmax(input, dim=1)  # N x C x 81 -> C is the number of classes

    def evaluate(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Require NOT ACTIVATED input"""
        return self._loss(input, target.long())

    def extract(self, input: torch.Tensor) -> torch.Tensor:
        """Require ACTIVATED input"""
        selected = torch.argmax(input, dim=1)
        # add + 1 on class because the number of sudoku goes from 1 to 9
        return selected + 1
    

if __name__ == '__main__':
    model = Sudoku_CNN(input_channels = 1, num_blocks=2, filters=[32, 64], kernel_size=3)
    helper = Sudoku_CNN_helper()

    print(model)    

    x = torch.randn(1, 81)
    probs = helper.activate(model(x))
    print(probs.shape)

    # Estrazione del massimo valore lungo l'asse delle classi
    result = helper.extract(probs)

    print(result.shape)  # Stampa la forma del tensore risultante



