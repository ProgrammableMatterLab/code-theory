import unittest
import torch
import math
from block.math import intersection_area, _pnorm_distances, rotate_points, transform_points, translate_points, is_overlapping

class TestMathFunctions(unittest.TestCase):

    def test_intersection_area(self):
        # Test with two circles that do not intersect
        points1 = torch.tensor([[0.0, 0.0]])
        points2 = torch.tensor([[10.0, 0.0]])
        radii1 = torch.tensor([1.0])
        radii2 = torch.tensor([1.0])
        area = intersection_area(points1, points2, radii1, radii2)
        self.assertAlmostEqual(area.item(), 0.0, delta=1e-4)

        # Test with two circles that fully contain each other
        points1 = torch.tensor([[0.0, 0.0]])
        points2 = torch.tensor([[0.0, 0.0]])
        radii1 = torch.tensor([2.0])
        radii2 = torch.tensor([1.0])
        area = intersection_area(points1, points2, radii1, radii2)
        self.assertAlmostEqual(area.item(), math.pi * (1.0 ** 2), delta=1e-4)

        # Test with two circles that partially intersect
        points1 = torch.tensor([[0.0, 0.0]])
        points2 = torch.tensor([[1.0, 0.0]])
        radii1 = torch.tensor([1.5])
        radii2 = torch.tensor([1.5])
        area = intersection_area(points1, points2, radii1, radii2)
        self.assertGreater(area.item(), 0.0)

    def test_pnorm_distances(self):
        points1 = torch.tensor([[0.0, 0.0], [1.0, 1.0]])
        points2 = torch.tensor([[0.0, 0.0], [2.0, 2.0]])
        distances = _pnorm_distances(points1, points2)
        self.assertAlmostEqual(distances[0, 0].item(), 0.0)
        self.assertAlmostEqual(distances[0, 1].item(), math.sqrt(8))
        self.assertAlmostEqual(distances[1, 0].item(), math.sqrt(2))
        self.assertAlmostEqual(distances[1, 1].item(), math.sqrt(2))


    def test_is_overlapping(self):
        # Test with non-overlapping circles
        points = torch.tensor([[0.0, 0.0], [0.0, 0.0]])
        radii = torch.tensor([1.0, 1.0])
        self.assertTrue(is_overlapping(points, radii))

        # Test with overlapping circles
        points = torch.tensor([[0.0, 0.0], [5.0, 5.0]])
        radii = torch.tensor([1.5, 1.5])
        self.assertFalse(is_overlapping(points, radii))

if __name__ == "__main__":
    unittest.main()
