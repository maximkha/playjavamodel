import torch
from torch import nn
import torch.nn.functional as F

class rac1d(nn.Conv1d):
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return self._conv_forward(input, F.pad(self.weight, (0, self.weight.shape[-1] -1), "constant", 0.), self.bias)