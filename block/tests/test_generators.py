import unittest
from block.generators import gen_had_block, gen_rand_block, _generate_non_overlapping_points
from block import Block
import torch
import numpy as np

class TestGenerators(unittest.TestCase):

    def test_gen_had_block(self):
        # Test that gen_had_block returns a Block object
        block = gen_had_block(2)
        self.assertIsInstance(block, Block)
        self.assertTrue(torch.allclose(block.points, torch.tensor([[-0.5000, -0.5000],
        [-0.5000,  0.5000],
        [ 0.5000, -0.5000],
        [ 0.5000,  0.5000]])))
        self.assertTrue(torch.allclose(block.polarities, torch.tensor([ 1,  1,  1, -1])))
    
    def test_gen_had_block_invalid_n(self):
        # Test that gen_had_block raises an exception for invalid n
        with self.assertRaises(Exception):
            gen_had_block(0)

    def test_gen_rand_block(self):
        # Test that gen_rand_block returns a Block object
        n = 4
        def func():
            return 0.1  # Simple function to generate radii
        block = gen_rand_block(n, func)
        self.assertIsInstance(block, Block)

        # Check that the block has the correct number of points and polarities
        self.assertEqual(len(block.points), n)
        self.assertEqual(len(block.polarities), n)

    def test_generate_non_overlapping_points(self):
        # Test that _generate_non_overlapping_points generates points correctly
        n = 4
        def func():
            return 0.1  # Simple function to generate radii
        points, radii = _generate_non_overlapping_points(n, func, (-1, 1, -1, 1))
        self.assertEqual(len(points), n)
        self.assertEqual(len(radii), n)

if __name__ == '__main__':
    unittest.main()
