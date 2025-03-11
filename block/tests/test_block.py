import unittest
from block.block import Block
import torch
from block.math import calculate_distances, circle_intersection_area
from block.utils import tensor_to_points

class TestBlock(unittest.TestCase):

    def test_init(self):
        # Test that Block initializes correctly
        points = torch.tensor([[0.0, 0.0], [1.0, 1.0]])
        polarities = torch.tensor([1, -1])
        radii = torch.tensor([0.5, 0.5])
        block = Block(points, polarities, radii)
        self.assertIsInstance(block, Block)
        self.assertTrue(torch.allclose(block.points, points))
        self.assertTrue(torch.allclose(block.polarities, polarities))
        self.assertTrue(torch.allclose(block.radii, radii))

    def test_from_block(self):
        # Test that from_block creates a new Block instance
        points = torch.tensor([[0.0, 0.0], [1.0, 1.0]])
        polarities = torch.tensor([1, -1])
        radii = torch.tensor([0.5, 0.5])
        block = Block(points, polarities, radii)
        new_block = Block.from_block(block)
        self.assertIsInstance(new_block, Block)
        self.assertTrue(torch.allclose(new_block.points, points))
        self.assertTrue(torch.allclose(new_block.polarities, polarities))
        self.assertTrue(torch.allclose(new_block.radii, radii))

    def test_calculate_attraction(self):
        # Test that calculate_attraction returns the correct attraction force
        points1 = torch.tensor([[0.0, 0.0], [1.0, 1.0]])
        polarities1 = torch.tensor([1, -1])
        radii1 = torch.tensor([0.5, 0.5])
        block1 = Block(points1, polarities1, radii1)

        points2 = torch.tensor([[0.5, 0.5], [1.5, 1.5]])
        polarities2 = torch.tensor([-1, 1])
        radii2 = torch.tensor([0.5, 0.5])
        block2 = Block(points2, polarities2, radii2)

        F, avg_F = Block.calculate_attraction(block1, block2)
        self.assertIsInstance(F, torch.Tensor)
        self.assertIsInstance(avg_F, torch.Tensor)

    def test_as_tuple(self):
        # Test that as_tuple returns the correct tuple
        points = torch.tensor([[0.0, 0.0], [1.0, 1.0]])
        polarities = torch.tensor([1, -1])
        radii = torch.tensor([0.5, 0.5])
        block = Block(points, polarities, radii)
        points_out, polarities_out, radii_out = block.as_tuple()
        self.assertTrue(torch.allclose(points_out, points))
        self.assertTrue(torch.allclose(polarities_out, polarities))
        self.assertTrue(torch.allclose(radii_out, radii))

    def test_rotated(self):
        # Test that rotated returns a new Block instance with rotated points
        points = torch.tensor([[0.0, 0.0], [1.0, 1.0]])
        polarities = torch.tensor([1, -1])
        radii = torch.tensor([0.5, 0.5])
        block = Block(points, polarities, radii)
        new_block = block.rotated(theta=math.pi / 2, mode='d')
        self.assertIsInstance(new_block, Block)

    def test_rotate(self):
        # Test that rotate modifies the Block instance correctly
        points = torch.tensor([[0.0, 0.0], [1.0, 1.0]])
        polarities = torch.tensor([1, -1])
        radii = torch.tensor([0.5, 0.5])
        block = Block(points, polarities, radii)
        block.rotate(theta=math.pi / 2, mode='d')
        self.assertTrue(torch.allclose(block.points, torch.tensor([[-1.0, 0.0], [0.0, 1.0]])))

if __name__ == '__main__':
    unittest.main()
