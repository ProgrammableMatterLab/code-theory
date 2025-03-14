import torch
from .math import intersection_area, rotate_points, transform_points, translate_points
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

    def mate(self) -> 'Block':
        return Block(self.points.clone(), -1 * self.polarities.clone(), self.radii.clone())

    def clone(self) -> 'Block':
        return Block(self.points.clone(), self.polarities.clone(), self.radii.clone())

    def calculate_attraction(self, other: 'Block') -> Tuple[torch.Tensor, float]:
        return calculate_attraction(self, other)

    def as_tuple(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return (self.points, self.polarities, self.radii)

    def rotate(self, theta, mode='d') -> None:
        self.points = rotate_points(self.points, theta, mode)
    
    def transform(self, A: torch.Tensor) -> None:
        self.points = transform_points(self.points, A)
    
    def translate(self, A: torch.Tensor) -> None:
        self.points = translate_points(self.points, A)
    
    def __str__(self):
        return f'points:\n {self.points},\n polarities:\n {self.polarities},\n radii:\n {self.radii},\n numel:\n {self.numel}'
    

def rotate(block: 'Block', theta, mode='d') -> 'Block':
  points = rotate_points(block.points, theta, mode)
  new_block = block.clone()
  new_block.points = points
  return new_block

def translate(block: 'Block', A: torch.Tensor) -> 'Block':
  points = translate_points(block.points, A)
  new_block = block.clone()
  new_block.points = points
  return new_block

def calculate_attraction(block1: 'Block', block2: 'Block') -> Tuple[torch.Tensor, float]:
  '''
  calculates the attraction between two sets of points p1 and p2
  Args:
    block1 (Block)
    block2 (Block)
  Returns:
    Tuple: a tuple of the attraction for each point and the sum of the attractive forces
  '''
  points1, polarities1, radii1 = block1.as_tuple()
  points2, polarities2, radii2 = block2.as_tuple()
  intersects = intersection_area(points1, points2, radii1, radii2)
  pols = polarities1.reshape(-1, 1) @ polarities2.reshape(1, -1)
  F = pols * intersects
  return F, torch.sum(F)