from block import Block
from scipy.linalg import hadamard
from .utils import tensor_to_points
import math
import torch
from .math import is_overlapping
import numpy as np
from typing import Callable, Tuple

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
    points, r = tensor_to_points(had)
    return Block(points, torch.flatten(had), r)


def gen_rand_block(n: int, func: Callable[[], float], bounds = (-1, 1, -1, 1), polarities = None) -> Block:
    '''
    generate a block based on a nxn hadamard matrix to describe the pols
    Args:
        n: side length of the hadamard matrix
    Returns:
        a Block object
    '''
    if n <= 0:
        raise Exception(f'n must be a positive number: {n} is invalid!')
    points, radii = _generate_non_overlapping_points(n, func, bounds)
    if not polarities:
        polarities = torch.randint(0, 2, (n, ))  
    return Block(points, polarities, radii)

    
def _generate_non_overlapping_points(num_points: int, func: Callable[[], float], bl: Tuple[float, float], tr: Tuple[float, float]) -> Tuple[torch.Tensor, torch.Tensor]:
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
        new_point = torch.tensor([np.random.uniform(bl[0], tr[0]), np.random.uniform(bl[1], tr[1])])
        radius =  func()
        tmp_points = torch.stack(points + [new_point])
        tmp_radii = torch.stack(radii + [radius])
        # Check if the new point intersects with any existing points
        if not is_overlapping(tmp_points, tmp_radii):
            points.append(new_point)
            radii.append(radius)
    return torch.stack(points), torch.stack(radii)

    