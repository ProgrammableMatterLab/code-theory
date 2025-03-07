import torch
import math
from .math import calculate_distances, circle_intersection_area, rotate
from .utils import tensor_to_points
from typing import Tuple, Optional, List

class Block:
    def __init__(
        self, 
        face_repr: Optional[torch.Tensor] = None, 
         points_polarities: Optional[List[torch.Tensor]] = None
    ):
        """
        Initialize the Face class with either a face representation tensor or a list of points and polarities tensors.

        Args:
            face_repr (Optional[torch.Tensor]): A tensor representing the face.
            points_polarities (Optional[List[torch.Tensor]]): A list containing two tensors: points and polarities.
        """
        if face_repr is not None:
            self.tensor = face_repr
            self.polarities = torch.flatten(face_repr)
            self.points, self.radius = tensor_to_points(face_repr)
            self.numel = len(self.points)
        elif points_polarities is not None:
            if len(points_polarities) != 2:
                raise ValueError("points_polarities must contain exactly two tensors.")
            self.points = points_polarities[0]
            self.polarities = points_polarities[1]
            self.numel = len(self.points)
        else:
            raise ValueError("Either 'face_repr' or 'points_polarities' must be provided.")

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
