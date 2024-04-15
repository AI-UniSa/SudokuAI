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

        # Add the final block
        input_channels = self.filters[-1] if isinstance(self.filters, list) else self.filters
        self._model.append(self.create_conv_block(input_channels, 9))

    def create_conv_block(self, in_channels=None, out_channels=None, preserve = True):
        conv_block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=self.kernel_size, padding= math.ceil((self.kernel_size - 1)/2) if preserve else 0, groups = 1),
            nn.BatchNorm2d(out_channels, momentum=self.decay),
            nn.ReLU(inplace=True) if self.activation_fn.lower() == "relu" else nn.ReLU(inplace=True)
        )
        return conv_block

    def forward(self, inputs):
        # Since input is a 1x81 tensor, we need to reshape it to 1x1x9x9
        inputs = inputs.view(-1, 1, 9, 9)
        outputs = inputs
        for block in self._model:
            outputs = block(outputs)

        return outputs
        
        # # now outputs are the logits
        # probs = torch.softmax(outputs, dim=1)
        
        # return probs
    
class Sudoku_CNN_helper(Helper):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self._loss = nn.CrossEntropyLoss()
        self._flatten = nn.Flatten()

    def activate(self, input: torch.Tensor) -> torch.Tensor:
        return torch.softmax(input, dim=1)  # N x C x H x W -> C is the number of classes

    def evaluate(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Require NOT ACTIVATED input"""

        # from input BxCxHxW to BxCx(H*W)
        input_loss = input.reshape(input.shape[0], input.shape[1], -1)

        # print("input_loss\n", input_loss) # B x C x 81
        # print("target\n", target) # B x 81

        return self._loss(input_loss, target.long())

    def extract(self, input: torch.Tensor) -> torch.Tensor:
        # Require ACTIVATED input
        return self._flatten(torch.argmax(input, dim=1))
    

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



