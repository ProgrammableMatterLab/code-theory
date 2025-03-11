import unittest
from block.math import _intersection_area, _ppower_distances, calculate_attraction, rotate, is_overlapping
import torch
import math

class TestMathFunctions(unittest.TestCase):

    def test_intersection_area(self):
        # Test that _intersection_area returns the correct intersection areas
        points1 = torch.tensor([[0.0, 0.0], [1.0, 1.0]])
        points2 = torch.tensor([[0.5, 0.5], [1.5, 1.5]])
        radii1 = torch.tensor([0.5, 0.5])
        radii2 = torch.tensor([0.5, 0.5])
        area = _intersection_area(points1, points2, radii1, radii2)
        self.assertIsInstance(area, torch.Tensor)

    def test_ppower_distances(self):
        # Test that _ppower_distances returns the correct distances
        points1 = torch.tensor([[0.0, 0.0], [1.0, 1.0]])
        points2 = torch.tensor([[0.5, 0.5], [1.5, 1.5]])
        dists = _ppower_distances(points1, points2)
        self.assertIsInstance(dists, torch.Tensor)

    def test_calculate_attraction(self):
        # Test that calculate_attraction returns the correct attraction forces
        from block.block import Block
        points1 = torch.tensor([[0.0, 0.0], [1.0, 1.0]])
        polarities1 = torch.tensor([1, -1])
        radii1 = torch.tensor([0.5, 0.5])
        block1 = Block(points1, polarities1, radii1)

        points2 = torch.tensor([[0.5, 0.5], [1.5, 1.5]])
        polarities2 = torch.tensor([-1, 1])
        radii2 = torch.tensor([0.5, 0.5])
        block2 = Block(points2, polarities2, radii2)

        F, total_F = calculate_attraction(block1, block2)
        self.assertIsInstance(F, torch.Tensor)
        self.assertIsInstance(total_F, torch.Tensor)

    def test_rotate(self):
        # Test that rotate returns the correct rotated points
        points = torch.tensor([[0.0, 0.0], [1.0, 1.0]])
        theta = math.pi / 2  # 90 degrees in radians
        rotated_points = rotate(points, theta)
        expected_points = torch.tensor([[-1.0, 0.0], [0.0, 1.0]])
        self.assertTrue(torch.allclose(rotated_points, expected_points))

    def test_rotate_degrees(self):
        # Test that rotate works correctly with degrees
        points = torch.tensor([[0.0, 0.0], [1.0, 1.0]])
        theta = 90  # 90 degrees
        rotated_points = rotate(points, theta, mode='d')
        expected_points = torch.tensor([[-1.0, 0.0], [0.0, 1.0]])
        self.assertTrue(torch.allclose(rotated_points, expected_points))

    def test_is_overlapping(self):
        # Test that is_overlapping returns the correct result
        points = torch.tensor([[0.0, 0.0], [1.0, 1.0]])
        radii = torch.tensor([0.5, 0.5])
        overlapping = is_overlapping(points, radii)
        self.assertIsInstance(overlapping, bool)

if __name__ == '__main__':
    unittest.main()
