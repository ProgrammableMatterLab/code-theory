from .block import Block
from scipy.linalg import hadamard
from .utils import had_to_points
import torch
from .math import is_overlapping, intersection_area
import numpy as np
from typing import Callable, Tuple, Optional
import random

def gen_had_block(n: int) -> Block:
    '''
    generate a block based on a nxn hadamard matrix to describe the pols
    Args:
        n: side length of the hadamard matrix
    Returns:
        a Block object
    '''
    if n <= 0:
        raise Exception(f'n must be a positive number and a power of 2: {n} is invalid!')
    had = torch.from_numpy(hadamard(n))
    points, r = had_to_points(had)
    return Block(points, torch.flatten(had), torch.tensor([r] * len(points)))


def gen_rand_block(n: int, func: Callable[[], float], bounds: Tuple[float, float, float, float]  = (-1, 1, -1, 1), polarities: Optional[torch.Tensor] = None) -> Block:
    '''
    generate a block based on a nxn hadamard matrix to describe the pols
    Args:
        n (int): number of points to generate
        func (Callable[[], float]): function to generate a radius for a point
        bounds (Tuple[float, float, float, float]): bounds describing where the points can be placed
        polarities (Optional[torch.Tensor]): polarities for the points
    Returns:
        a Block object
    '''
    if n <= 0:
        raise Exception(f'n must be a positive number: {n} is invalid!')
    points, radii = _generate_non_overlapping_points(n, func, bounds)
    if not polarities:
        polarities = torch.randint(0, 2, (n, ))  
    return Block(points, polarities, radii)

    
def _generate_non_overlapping_points(num_points: int, func: Callable[[], float], bounds: Tuple[float, float, float, float]) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Generate non-intersecting points.

    Args:
    num_points (int): Number of points to generate.
    func (callable): A function that takes one float argument and returns a float.
    bounds (tuple): Bounds for generating points.

    Returns:
    Tuple: points, radii
    """
    points = []
    radii = []
    while len(points) < num_points:
        new_point = [random.uniform(bounds[0], bounds[1]), random.uniform(bounds[2], bounds[3])]
        radius =  [func()]
        # Check if the new point intersects with any existing points
        if torch.sum(intersection_area(torch.tensor(points), torch.tensor(new_point), torch.tensor(radii), torch.tensor(radius))) == 0:
            points.append(new_point)
            radii.append(radius)
    return torch.tensor(points), torch.tensor(radii)
    