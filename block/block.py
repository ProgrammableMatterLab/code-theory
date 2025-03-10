import torch
import math
from .math import calculate_distances, circle_intersection_area, rotate
from .utils import tensor_to_points
from typing import Tuple, Optional, List

class Block:
    def __init__(
        self, 
        face_repr: Optional[torch.Tensor] = None, 
        blob: Optional[Tuple[torch.Tensor, torch.Tensor, float]] = None
    ):
        """
        Initialize the Face class with either a face representation tensor or a list of points and polarities tensors.

        Args:
            face_repr (Optional[torch.Tensor]): A tensor representing the face.
            blob (Optional[Tuple[torch.Tensor, torch.Tensor, float]]): A tuple with the (points, polarities, radius)
        """
        if face_repr and blob:
            raise ValueError("choose only one of 'face_repr' or 'blob' to init")
        elif face_repr:
            self.polarities = torch.flatten(face_repr)
            self.points, self.radius = tensor_to_points(face_repr)
            self.numel = len(self.points)
        elif blob:
            if len(blob) != 3:
                raise ValueError("points_polarities must contain exactly two tensors.")
            self.points = blob[0]
            self.polarities = blob[1]
            self.radius = blob[2]
            self.numel = len(self.points)
        else:
            raise ValueError("Either 'face_repr' or 'blob' must be provided.")

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
        points = rotate(self.points, theta, mode)
        b = Block(blob=(points, self.polarities, self.radius))
        return b

    def rotate(self, theta, mode='d'):
        self.points = rotate(self.points, theta, mode)