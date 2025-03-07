import torch
import math

def circle_intersection_area(r1: float, r2: float, d: torch.Tensor) -> torch.Tensor:
    """
    Calculate the intersection area of two circles for multiple distances.

    Args:
    r1 (float): Radius of the first circle
    r2 (float): Radius of the second circle
    d (torch.Tensor): Tensor of distances between the centers of the circles

    Returns:
    torch.Tensor: Tensor of intersection areas corresponding to each distance in d


    proof: https://calculator.academy/area-between-two-intersecting-circles-calculator/
    """
    # Check if circles intersect
    no_intersection = d >= r1 + r2
    full_containment = d <= abs(r1 - r2)
    
    # Calculate the intersection area
    a = (r1**2 - r2**2 + d**2) / (2 * d)
    h = torch.sqrt(torch.clamp(r1**2 - a**2, min=0))
    
    angle1 = torch.acos(torch.clamp(a / r1, min=-1, max=1))
    angle2 = torch.acos(torch.clamp((d - a) / r2, min=-1, max=1))
    
    area1 = r1**2 * angle1
    area2 = r2**2 * angle2
    area_triangle = r1 * h
    
    intersection_area = torch.where(no_intersection, torch.zeros_like(d),
                                    torch.where(full_containment, torch.full_like(d, math.pi * min(r1, r2)**2),
                                                area1 + area2 - area_triangle))
    
    return intersection_area

# # Example usage:
# d = torch.tensor([4])
# result = circle_intersection_area(2, 2.0, d)
# print(result)

# Calculate distances
def calculate_distances(p1, p2):
    # Reshape p1 to (n1, 1, 2) and p2 to (1, n2, 2)
    p1_expanded = p1.unsqueeze(1)
    p2_expanded = p2.unsqueeze(0)

    # Calculate squared differences
    squared_diff = (p1_expanded - p2_expanded) ** 2

    # Sum along the last dimension and take the square root
    distances = squared_diff.sum(dim=-1) ** 0.5

    return distances


def calculate_attraction(p1, p2, pols1, pols2, rad1, rad2):
  dist = calculate_distances(p1, p2)

  intersects = circle_intersection_area(rad1, rad2, dist)
  # dist = torch.where(dist <= rad1, dist, float('inf'))
  # dist += 1
  #dist = torch.where(torch.isinf(dist), dist, 1.0) # tmp, use distance to factor in attriactive force based on common surface area
  pols = pols1.reshape(-1, 1) @ pols2.reshape(1, -1)
  F = pols * intersects
  F = F / (math.pi * rad1**2)
  
  return F, torch.sum(F) / (len(p1))

def rotate(tensor, theta, mode='d'):
  if mode == 'd':
    theta = math.pi * theta / 180
  rot_mat = torch.tensor([[math.cos(theta), -1 * math.sin(theta)], [math.sin(theta), math.cos(theta)]])
  return tensor @ rot_mat