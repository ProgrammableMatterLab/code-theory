import unittest
from block.generators import gen_had_block, gen_rand_block, _generate_non_overlapping_points
from block import Block
import torch
import numpy as np
from scipy.linalg import hadamard
from block.utils import tensor_to_points

class TestGenerators(unittest.TestCase):

    def test_gen_had_block(self):
        # Test that gen_had_block returns a Block object
        n = 4  # Must be a power of 2
        block = gen_had_block(n)
        self.assertIsInstance(block, Block)
        
        # Check that the block's points and polarities are correctly generated
        had = torch.from_numpy(hadamard(n))
        points, r = tensor_to_points(had)
        self.assertTrue(torch.allclose(block.points, points))
        self.assertTrue(torch.allclose(block.polarities, torch.flatten(had)))

    def test_gen_had_block_invalid_n(self):
        # Test that gen_had_block raises an exception for invalid n
        with self.assertRaises(Exception):
            gen_had_block(0)

    def test_gen_rand_block(self):
        # Test that gen_rand_block returns a Block object
        n = 4
        def func():
            return 1.0  # Simple function to generate radii
        block = gen_rand_block(n, func)
        self.assertIsInstance(block, Block)

        # Check that the block has the correct number of points and polarities
        self.assertEqual(len(block.points), n)
        self.assertEqual(len(block.polarities), n)

    def test_gen_rand_block_invalid_n(self):
        # Test that gen_rand_block raises an exception for invalid n
        with self.assertRaises(Exception):
            gen_rand_block(0, lambda: 1.0)

    def test_generate_non_overlapping_points(self):
        # Test that _generate_non_overlapping_points generates points correctly
        n = 4
        def func():
            return 0.1  # Simple function to generate radii
        points, radii = _generate_non_overlapping_points(n, func, (-1, 1), (-1, 1))
        self.assertEqual(len(points), n)
        self.assertEqual(len(radii), n)

if __name__ == '__main__':
    unittest.main()
