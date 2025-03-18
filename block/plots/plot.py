import torch
import numpy as np
from typing import List
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Polygon, Rectangle
from ..utils import attraction_per_angle, attraction_per_translation
from ..block import Block

def plot_faces(blocks: list, zoom_factor: float = 2, alpha: float = 0.3):
  
  b_points = []
  b_polarities = []
  b_radii = []
  for block in blocks:
    b_points.append(block.points)
    b_polarities.append(block.polarities)
    b_radii.append(block.radii) 

  fig, ax = plt.subplots(figsize=(8, 8))
  for points, polarities, radii in zip(b_points, b_polarities, b_radii):
    for point, polarity, radius in zip(points, polarities, radii):
        color = 'r' if polarity > 0 else 'b'
        circle = Circle((point[0], point[1]), radius, facecolor=color, edgecolor='black', alpha=alpha)
        ax.add_artist(circle)
        marker = 'N' if polarity > 0 else 'S'
        ax.text(point[0], point[1], marker, ha='center', va='center', color='black', fontweight='bold')
       
  ax.set_xlim(-zoom_factor, zoom_factor)
  ax.set_ylim(-zoom_factor, zoom_factor)
  ax.set_aspect('equal', adjustable='box')
  ax.set_xlabel('X')
  ax.set_ylabel('Y')
  ax.set_title('Plot of Blocks')
  ax.grid(True)
  plt.show()

def plot_rotation_attraction(block1: Block, block2: Block, num_angles: int = 160) -> None:
  """
  Plot attraction vs rotation Nlock

  Args:
  block1 (Block): block
  block1 (Block): block
  num_angles (int): Number of angles to evaluate (default 160)
  """

  angles, results = attraction_per_angle(block1, block2, num_angles)

  fig, ax = plt.subplots(figsize=(12, 6))
  bars = ax.bar(angles, results, width=360/num_angles)

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
  plt.show()

def plot_translation_attraction(block1: Block, block2: Block, num_x: int = 20, num_y: int = 20, lo: int = -2, hi: int = 2) -> None:
    """
    Plot attraction vs translation of blocks.

    Args:
    block1 (Block): First block.
    block2 (Block): Second block.
    num_x (int): Number of x positions to evaluate (default 20).
    num_y (int): Number of y positions to evaluate (default 20).
    """
    X, Y, result = attraction_per_translation(block1, block2, num_x, num_y, lo, hi)
    # Plot
    plt.figure(figsize=(8, 8))
    plt.scatter(X, Y, c=result.ravel(), cmap='RdBu')
    plt.colorbar(label='Attraction')
    plt.xlabel('X Translation')
    plt.ylabel('Y Translation')
    plt.title('Attraction vs Translation')
    plt.show()
