import torch
import numpy as np
from typing import List
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import Optimizer
import matplotlib.pyplot as plt
import numpy as np
import math
import torch
import numpy as np
import matplotlib.pyplot as plt
from .utils import tensor_to_points
import matplotlib.pyplot as plt
from .math import rotate, calculate_attraction, calculate_distances
from matplotlib.patches import Circle
from .block import Block
from typing import List
from scipy.linalg import hadamard

def plot_faces(blocks: Block, colors: List[str], zoom_factor: float = 2, alpha: float = 0.5):
  
  if not isinstance(blocks, list):
     blocks = [blocks]
  tensors = []
  pols = []
  radii = []
  for block in blocks:
    tensors.append(block.points)
    pols.append(block.polarities)
    radii.append(block.radius)  
  fig, ax = plt.subplots(figsize=(8, 8))

  for tensor, pol, radius, color in zip(tensors, pols, radii, colors):
    points = tensor.clone().detach().numpy()
    values = pol
    for point, value in zip(points, values):
        circle = Circle((point[0], point[1]), radius, facecolor=color, edgecolor='none', alpha=alpha)
        ax.add_artist(circle)
        marker = 'N' if value > 0 else 'S'
        ax.text(point[0], point[1], marker, ha='center', va='center', color='black', fontweight='bold')

  ax.set_xlim(-zoom_factor, zoom_factor)
  ax.set_ylim(-zoom_factor, zoom_factor)
  ax.set_aspect('equal', adjustable='box')
  ax.set_xlabel('X')
  ax.set_ylabel('Y')
  ax.set_title('Circle Plot of Multiple Tensors')
  ax.grid(True)
  plt.show()


def plot_attraction_vs_rotation_for_multiple_N(N_values, num_angles=260):
    """
    Plot attraction vs rotation for multiple N x N Hadamard matrices using bar graphs.

    Args:
    N_values (list): List of N values (must be powers of 2)
    num_angles (int): Number of angles to evaluate (default 360)

    Returns:
    None (displays plots)
    """
    angles = torch.linspace(-180, 180, num_angles)

    for N in N_values:
        if not (N & (N-1) == 0) or N == 0:
            print(f"Skipping N={N} as it's not a power of 2")
            continue

        # # Generate Hadamard matrix
        had = hadamard(N)
        # pols1 = torch.tensor(had.flatten(), dtype=torch.float32)
        # pols2 = -pols1

        # # Transform into sets of points
        # t1 = torch.tensor(had, dtype=torch.float32)
        # t2 = torch.tensor(had, dtype=torch.float32)
        # p1, r1 = tensor_to_points(t1)
        # p2, r2 = tensor_to_points(t2)



     

        plot_faces(
          tensors=[p1, p2],
          pols=[pols1, pols2],
          radii=[r1, r2],
          colors=['blue', 'red'],
          zoom_factor=2
      )

        results = []
        for angle in angles:
            rotated = rotate(p2, angle)
            _, res = calculate_attraction(p1, rotated, pols1, pols2, r1, r2)
            results.append(res.item())

        # Convert angles and results to numpy arrays
        angles_np = angles.numpy()
        results_np = np.array(results)
        results_np = results_np / np.max(np.absolute(results_np))


        # Normalize results to use for color mapping
        # norm_results = (results_np - results_np.min()) / (results_np.max() - results_np.min())

        # # Create the bar plot
        fig, ax = plt.subplots(figsize=(12, 6))
        bars = ax.bar(angles_np, results_np, width=360/num_angles)

        # Color the bars based on the normalized results
        sm = plt.cm.ScalarMappable(cmap='viridis', norm=plt.Normalize(vmin=-1, vmax=1))
        sm.set_array([])

        # Add colorbar and specify the axis
        cbar = fig.colorbar(sm, ax=ax, label='Normalized Attraction')

        for bar, norm_res in zip(bars, results):
            bar.set_color(plt.cm.viridis(norm_res))

        # Set labels and title
        ax.set_xlabel('Angle (degrees)')
        ax.set_ylabel('Attractive Force')
        ax.set_title(f'Attractive Force vs Rotation Angle (N={N})')
        ax.set_ylim(-1, 1)

        plt.show()

    


# def plot_attraction_vs_rotation_for_multiple_N(N_values, num_angles=260):
#     """
#     Plot attraction vs rotation for multiple N x N Hadamard matrices using bar graphs.
    
#     Args:
#     N_values (list): List of N values (must be powers of 2)
#     num_angles (int): Number of angles to evaluate (default 360)
    
#     Returns:
#     None (displays plots)
#     """
#     angles = torch.linspace(-180, 180, num_angles)

#     for N in N_values:
#         if not (N & (N-1) == 0) or N == 0:
#             print(f"Skipping N={N} as it's not a power of 2")
#             continue
        
#         # Generate Hadamard matrix
#         had = hadamard(N)
#         pols1 = torch.tensor(had.flatten(), dtype=torch.float32)
#         pols2 = -pols1

#         had = torch.from_numpy(hadamard(N))
#         block = Block(had)
#         invb = Block.from_block(block)
#         invb.polarities = invb.polarities * -1

#         # pols1, p1, r1 = block.polarities, block.points, block.radius
#         # pols2, p2, r2 = invb.polarities, invb.points, invb.radius
        
#         # # Transform into sets of points
#         # t1 = torch.tensor(had, dtype=torch.float32)
#         # t2 = torch.tensor(had, dtype=torch.float32)
#         # p1, r1 = tensor_to_points(t1)
#         # p2, r2 = tensor_to_points(t2)
        
#         results = []
#         for angle in angles:
#             invb.rotate(angle)
#             _, res = Block.calculate_attraction(block, invb)
#             results.append(res.item())
        
#         # Convert angles and results to numpy arrays
#         angles_np = angles.numpy()
#         results_np = np.array(results)
#         # Normalize results to use for color mapping
#         norm_results = (results_np - results_np.min()) / (results_np.max() - results_np.min())
#         # Create the bar plot
#         fig, ax = plt.subplots(figsize=(12, 6))
#         bars = ax.bar(angles_np, results_np, width=360/num_angles)
#         # Color the bars based on the normalized results
#         sm = plt.cm.ScalarMappable(cmap='viridis', norm=plt.Normalize(vmin=-1, vmax=1))
#         sm.set_array([])
#         # Add colorbar and specify the axis
#         cbar = fig.colorbar(sm, ax=ax, label='Normalized Attraction')

#         for bar, norm_res in zip(bars, norm_results):
#             bar.set_color(plt.cm.viridis(norm_res))
#         # Set labels and title
#         ax.set_xlabel('Angle (degrees)')
#         ax.set_ylabel('Attractive Force')
#         ax.set_title(f'Attractive Force vs Rotation Angle (N={N})')
#         ax.set_ylim(-1, 1)

#         plt.show()
