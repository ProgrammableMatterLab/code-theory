import torch
from typing import Tuple
from .block import Block, rotate, calculate_attraction, translate

def had_to_points(tensor: torch.Tensor, dim=2) -> torch.tensor:
  points = torch.nonzero(tensor, as_tuple=False).float()
  points -= torch.mean(points, dim=0)
  points /= torch.max(points) + 0.5
  return points, 1 / len(tensor)



def attraction_per_angle(block1: Block, block2: Block, num_angles: int = 160) -> Tuple[list, torch.tensor]:
  """
  returns array of (angle, attrative force)

  Args:
  block1 (Block): block
  block1 (Block): block
  num_angles (int): Number of angles to evaluate (default 160)

  Returns:
  Tuple[list, torch.Tensor]: list of angles and tensor of values
  """
  angles = torch.linspace(-180, 180, num_angles)
  results = []
  for angle in angles:
    rotated = rotate(block2, angle)
    _, res = calculate_attraction(block1, rotated)
    results.append(res.item())

  result_tensor = torch.tensor(results)
  result_tensor /= torch.max(torch.abs(result_tensor))
  return angles, result_tensor


def attraction_per_translation(block1: Block, block2: Block, num_x: int, num_y: int) -> Tuple[list, torch.tensor]:
  """
  returns array of (angle, attrative force)

  Args:
  block1 (Block): block
  block1 (Block): block
  num_angles (int): Number of angles to evaluate (default 160)

  Returns:
  Tuple[list, torch.Tensor]: list of angles and tensor of values
  """
  X = torch.linspace(-2, 2, num_x)
  Y = torch.linspace(-2, 2, num_y)
  results = []
  pointsX = []
  pointsY = []
  for x in X:
    for y in Y:
      translated = translate(block2, torch.tensor([x, y]))
      _, res = calculate_attraction(block1, translated)
      results.append(res.item())
      pointsX.append(x)
      pointsY.append(y)

  result_tensor = torch.tensor(results)
  result_tensor /= torch.max(torch.abs(result_tensor))
  return pointsX, pointsY, result_tensor

  