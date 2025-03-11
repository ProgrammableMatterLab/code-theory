import torch
import math
from .math import calculate_distances, circle_intersection_area, rotate, calculate_attraction
from .utils import tensor_to_points
from typing import Tuple, Optional, List
from block import Block

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

    def clone(self) -> Block:
        return Block(self.points.clone(), self.polarities.clone(), self.radii.clone())

    def calculate_attraction(self, other: Block) -> Tuple[torch.Tensor, float]:
        return calculate_attraction(self, other)

    def as_tuple(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return (self.points, self.polarities, self.radii)

    def rotated(self, theta, mode='d') -> Block:
        points = rotate(self.points, theta, mode)
        return Block(blob=(points, self.polarities, self.radius))

    def rotate(self, theta, mode='d') -> None:
        self.points = rotate(self.points, theta, mode)