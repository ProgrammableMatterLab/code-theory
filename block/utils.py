import torch

def had_to_points(tensor: torch.Tensor, dim=2) -> torch.tensor:
  points = torch.nonzero(tensor, as_tuple=False).float()
  points -= torch.mean(points, dim=0)
  points /= torch.max(points) + 0.5
  return points, 1 / len(tensor)