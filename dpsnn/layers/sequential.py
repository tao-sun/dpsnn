import torch.nn as nn

class Sequential(nn.Sequential):
    """
        Sequential that accepts multiple inputs.
    """

    def forward(self, *input):
        output = input[0]
        for module in self._modules.values():
            # print(f"sequential input[0] shape: {input[0].shape}, {input[1]}")
            output = module(output, *input[1:])
            # print(f"sequential output[0] shape: {input[0].shape}, {input[1]}")
        return output