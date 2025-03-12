import torch
import math

def intersection_area(points1: torch.Tensor, points2: torch.Tensor, radii1: torch.Tensor, radii2: torch.Tensor, power=2) -> torch.Tensor:
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
    dists = _pnorm_distances(points1, points2, power=power)
    r1_expanded = radii1.unsqueeze(1)
    r2_expanded = radii2.unsqueeze(0)

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

def _pnorm_distances(p1: torch.Tensor, p2: torch.Tensor, power: float = 2) -> torch.Tensor:
  '''
  calculates ||p - p'||_power for every p in p1 for every p in p2
  Args:
    p1 (torch.Tensor): tensor of points (N, 3)
    p2 (torch.Tensor): tensor of points (M, 3)
    power (float): power
  Returns:
    torch.Tensor: distances
  '''
  assert power > 0
  # Reshape p1 to (n1, 1, 2) and p2 to (1, n2, 2)
  p1_expanded = p1.unsqueeze(1)
  p2_expanded = p2.unsqueeze(0)
  # Calculate p norm of difference
  return ((p1_expanded - p2_expanded) ** power).sum(dim=-1) ** (1 / power)

def rotate_points(points: torch.Tensor, theta, mode='d') -> torch.Tensor:
  '''
  returns the points rotated by theta
  Args:
    points (torch.Tensor): points to be rotated
    theta (float): angle
    mode (None or 'd'): if 'd' then mode is degrees otherwise radians
  Returns:
    torch.Tensor: a new tensor of the rotated points
  '''
  if mode == 'd':
    theta = math.pi * theta / 180
  rot_mat = torch.tensor([[math.cos(theta), -1 * math.sin(theta)], [math.sin(theta), math.cos(theta)]])
  return transform(points, rot_mat)

def transform(points: torch.Tensor, A: torch.Tensor) -> torch.Tensor:
  '''
  applies a linear transformation to points
  Args:
  points (torch.Tensor): points (N,2)
  A (torch.Tensor): transformation matrix (2, 2)
  Returns:
  torch.Tensor: transformed points
  '''
  return points @ A

def translate(points: torch.Tensor, A: torch.Tensor) -> torch.Tensor:
  '''
  translates points
  Args:
  points (torch.Tensor): points (N,2)
  A (torch.Tensor): translation matrix (2,)
  Returns:
  torch.Tensor: translated points
  '''
  return points + A

def is_overlapping(points: torch.Tensor, radii: torch.Tensor) -> bool:
  '''
  checks if a set of points is overlapping or note
  Args:
    points (torch.Tensor): set of points
    radii (torch.Tensor): radius for each circle centered at p in points
  Returns:
    bool: true if no circles overlap, otherwise false
  '''
  area = intersection_area(points, points, radii, radii)
  return True if torch.sum(area) < math.pi * torch.sum(radii**2) else False