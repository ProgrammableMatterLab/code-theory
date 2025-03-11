import torch
import math
from .math import calculate_distances, circle_intersection_area, rotate
from .utils import tensor_to_points
from typing import Tuple, Optional, List

class Block:
    def __init__(
        self, 
        points: torch.Tensor, polarities: torch.Tensor, radii: torch.Tensor
    ):
        """
        Initialize the Face class with either a face representation tensor or a list of points and polarities tensors.

        Args:
            face_repr (Optional[torch.Tensor]): A tensor representing the face.
            blob (Optional[Tuple[torch.Tensor, torch.Tensor, float]]): A tuple with the (points, polarities, radius)
        """
        self.points = points
        self.polarities = polarities
        self.radii = radii
        self.numel = len(self.points)

    @classmethod
    def from_block(cls, other):
        new_block = cls(other.points.clone(), other.polarities.clone(), other.radii.clone())
        return new_block

    @staticmethod
    def calculate_attraction(block1, block2):
        # Implementation of calculate_attraction function
        dist = calculate_distances(block1.points, block2.points)
        intersects = circle_intersection_area(block1.radii, block2.radii, dist)
        pols = block1.polarities.reshape(-1, 1) @ block2.polarities.reshape(1, -1)
        F = pols * intersects
        return F, torch.sum(F) / block1.numel

    def rotated(self, theta, mode='d'):
        points = rotate(self.points, theta, mode)
        b = Block(blob=(points, self.polarities, self.radius))
        return b

    def rotate(self, theta, mode='d'):
        self.points = rotate(self.points, theta, mode)