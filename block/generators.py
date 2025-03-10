from block import Block
from scipy.linalg import hadamard
import math
import torch
import numpy as np

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
    return Block(torch.from_numpy(hadamard(n)))

def gen_rand_block(n: int, radius: float, bounds = (-1, 1, -1, 1), polarities = None) -> Block:
    '''
    generate a block based on a nxn hadamard matrix to describe the pols
    Args:
        n: side length of the hadamard matrix
    Returns:
        a Block object
    '''
    if n <= 0:
        raise Exception(f'n must be a positive number: {n} is invalid!')
    points = generate_non_intersecting_points(n, radius, bounds)
    if not polarities:
        polarities = torch.randint(0, 2, (n, ))  
    return Block(blob=(points, polarities, radius))

    
def generate_non_intersecting_points(num_points, radius, bounds):
    """
    Generate random points within specified bounds that do not intersect with each other.

    Parameters:
    - num_points: Number of points to generate.
    - radius: Radius of each point.
    - bounds: Tuple of (min_x, max_x, min_y, max_y) defining the area where points can be generated.

    Returns:
    - points: A tensor of shape (num_points, 2) containing the generated points.
    """
    points = []
    while len(points) < num_points:
        new_point = torch.tensor([np.random.uniform(bounds[0], bounds[1]), np.random.uniform(bounds[2], bounds[3])])

        # Check if the new point intersects with any existing points
        intersects = False
        for point in points:
            distance = torch.norm(new_point - point)
            if distance < 2 * radius:  # Check if distance is less than twice the radius
                intersects = True
                break

        if not intersects:
            points.append(new_point)

    return torch.stack(points)

    