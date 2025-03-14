from .block import Block
from .math import *
import torch
import torch.nn as nn
import torch.nn.functional as F


class BlockRotationLayer(nn.Module):
    def __init__(self, block1: Block, block2: Block, steps: int):
        super(BlockRotationLayer, self).__init__()

        self.p1 =  nn.Parameter(block1.points)
        self.p2 =  nn.Parameter(block2.points)
        self.block1 = block1
        self.block2 = block2
        self.steps = steps

    def forward(self):
        angles = torch.linspace(-180, 180, self.steps)
        out = []
        for angle in angles:
            out.append(self.block1.calculate_attraction(self.block2))
        return torch.tensor(out)
        
