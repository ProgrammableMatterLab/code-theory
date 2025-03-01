import torch
import math
from .math import calculate_distances, circle_intersection_area, rotate
from .utils import tensor_to_points
from typing import Tuple

class Block:
    def __init__(self, tensor: torch.Tensor):
        """
        Initialize a Block object.

        Args:
            tensor (Tensor): A 2D tensor of shape (n, m) representing the block's structure.

        Returns:
            None
        """
        self.tensor = tensor
        self.polarities = torch.flatten(tensor)
        self.points, self.radius = tensor_to_points(tensor)
        self.numel = len(self.points)

    @classmethod
    def from_block(cls, other):
        new_block = cls(other.tensor.clone())
        return new_block

    @staticmethod
    def calculate_attraction(block1, block2):
        # Implementation of calculate_attraction function
        dist = calculate_distances(block1.points, block2.points)
        intersects = circle_intersection_area(block1.radius, block2.radius, dist)
        pols = block1.polarities.reshape(-1, 1) @ block2.polarities.reshape(1, -1)
        F = pols * intersects
        return F, torch.sum(F) / block1.numel

    def rotated(self, theta, mode='d'):
        assert theta >= 0
        points = rotate(self.points, theta, mode)
        b = Block(self.tensor)
        b.points = points
        return b

    def rotate(self, theta, mode='d'):
        assert theta >= 0
        self.points = rotate(self.points, theta, mode)
