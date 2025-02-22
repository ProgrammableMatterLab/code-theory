import torch
import math
from .math import calculate_distances, circle_intersection_area, rotate
from .utils import tensor_to_points

class Block:
    def __init__(self, tensor, polarities):
        self.tensor = tensor
        self.polarities = polarities
        self.points, self.radius = tensor_to_points(tensor)
        self.numel = len(self.points)

    @staticmethod
    def calculate_attraction(block1, block2):
        # Implementation of calculate_attraction function
        dist = calculate_distances(block1.points, block2.points)
        intersects = circle_intersection_area(block1.radius, block2.radius, dist)
        pols = block1.polarities.reshape(-1, 1) @ block2.polarities.reshape(1, -1)
        F = pols * intersects
        return F, torch.sum(F) / block1.numel

    def rotated(points, theta, mode='d'):
        assert theta >= 0
        return rotate(points, theta, mode)
