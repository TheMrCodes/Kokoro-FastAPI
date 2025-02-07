from typing import Optional
import torch
from torch import nn
    
class Model(nn.Module):
    
    def __init__(self, op=None):
        super().__init__()
        self.op = op
    
    def forward(self, input):
        if self.op is not None:
            input = self.op(input)
        return input

with_op = Model(nn.ReLU())
with_op_js = torch.jit.script(with_op)

with_none = Model(None)
with_none_js = torch.jit.script(with_none)
print(with_none_js.graph)