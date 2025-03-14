import unittest
from block.block import Block
import torch
from block.generators import gen_had_block

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

    def test_clone(self):
        # Test that from_block creates a new Block instance
        points = torch.tensor([[0.0, 0.0], [1.0, 1.0]])
        polarities = torch.tensor([1, -1])
        radii = torch.tensor([0.5, 0.5])
        block = Block(points, polarities, radii)
        new_block = block.clone()
        self.assertIsInstance(new_block, Block)
        self.assertTrue(torch.allclose(new_block.points, points))
        self.assertTrue(torch.allclose(new_block.polarities, polarities))
        self.assertTrue(torch.allclose(new_block.radii, radii))

    def test_mate(self):
        # Test that calculate_attraction returns the correct attraction force
        block1 = gen_had_block(2)
        block2 = block1.mate()
        self.assertTrue(torch.allclose(block2.points, block1.points))
        self.assertTrue(torch.allclose(block2.radii, block1.radii))
        self.assertTrue(torch.allclose(block2.polarities, -1 * block1.polarities))

    def test_calculate_attraction(self):
        # Test that calculate_attraction returns the correct attraction force
        block1 = gen_had_block(2)
        block2 = block1.mate()

        A, F = Block.calculate_attraction(block1, block2)
        self.assertIsInstance(A, torch.Tensor)
        self.assertIsInstance(F, torch.Tensor)
        self.assertTrue(torch.allclose(A, torch.tensor([[-0.7854, -0.0000, -0.0000,  0.0000],
        [-0.0000, -0.7854, -0.0000,  0.0000],
        [-0.0000, -0.0000, -0.7854,  0.0000],
        [ 0.0000,  0.0000,  0.0000, -0.7854]])))
        self.assertTrue(torch.allclose(F, torch.tensor(-3.1416)))

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

if __name__ == '__main__':
    unittest.main()
