import torch
import math
from block import Block
from typing import Tuple

def _intersection_area(points1: torch.Tensor, points2: torch.Tensor, radii1: torch.Tensor, radii2: torch.Tensor) -> torch.Tensor:
    """
    Calculate the intersection between two sets of circles with radii r1 and r2

    Args:
    points1 (torch.Tensor): Centers of the first set of circles (N, 2)
    points2 (torch.Tensor): Centers of the second set of circles (M, 2)
    radii1 (torch.Tensor): Radii of the first set of circles (N,)
    radii2 (torch.Tensor): Radii of the second set of circles (M,)

    Returns:
    torch.Tensor: Tensor of intersection areas (N, M)
    """
    dists = _ppower_distances(points1, points2, power=2) ** 0.5
    r1_expanded = radii1.unsqueeze(1)
    r2_expanded = radii2.unsqueeze(1)

    # Check if circles intersect
    no_intersection = dists >= r1_expanded + r2_expanded
    full_containment = dists <= torch.abs(r1_expanded - r2_expanded)
    
    # Calculate the intersection area
    a = (r1_expanded**2 - r2_expanded**2 + dists**2) / (2 * dists)
    h = torch.sqrt(torch.clamp(r1_expanded**2 - a**2, min=0))
    
    angle1 = torch.acos(torch.clamp(a / r1_expanded, min=-1, max=1))
    angle2 = torch.acos(torch.clamp((dists - a) / r2_expanded, min=-1, max=1))

    area1 = r1_expanded**2 * angle1
    area2 = r2_expanded**2 * angle2
    area_triangle = r1_expanded * h

    intersection_area = torch.where(no_intersection, torch.zeros_like(dists),
                                    torch.where(full_containment, math.pi * torch.min(r1_expanded, r2_expanded)**2,
                                                area1 + area2 - area_triangle))
    return intersection_area

# Calculate distances
def _ppower_distances(p1: torch.Tensor, p2: torch.Tensor, power: float = 2):
  '''
  calculates ||p - p'||_power ^ power for every p in p1 for every p in p2

  Args:
  p1 (torch.Tensor): tensor of points (N, 3)
  p2 (torch.Tensor): tensor of points (M, 3)
  power (float): power
  '''
  # Reshape p1 to (n1, 1, 2) and p2 to (1, n2, 2)
  p1_expanded = p1.unsqueeze(1)
  p2_expanded = p2.unsqueeze(0)

  # Calculate squared differences
  return ((p1_expanded - p2_expanded) ** power).sum(dim=-1)


def calculate_attraction(block1: Block, block2: Block) -> Tuple[torch.Tensor, float]:
  '''
  calculates the attraction between two sets of points p1 and p2
  Args:
  block1 (Block)
  block2 (Block)

  Returns:
  a tuple of the attraction for each point and the sum of the attractive forces
  '''
  points1, polarities1, radii1 = block1.as_tuple()
  points2, polarities2, radii2 = block2.as_tuple()
  intersects = _intersection_area(points1, points2, radii1, radii2)
  pols = polarities1.reshape(-1, 1) @ polarities2.reshape(1, -1)
  F = pols * intersects
  return F, torch.sum(F)

def rotate(points: torch.Tensor, theta, mode='d'):
  '''
  returns the points rotated by theta
  Args:
  points (torch.Tensor): points to be rotated
  theta (float): angle
  mode (None or 'd'): if 'd' then mode is degrees otherwise radians
  '''
  if mode == 'd':
    theta = math.pi * theta / 180
  rot_mat = torch.tensor([[math.cos(theta), -1 * math.sin(theta)], [math.sin(theta), math.cos(theta)]])
  return points @ rot_mat