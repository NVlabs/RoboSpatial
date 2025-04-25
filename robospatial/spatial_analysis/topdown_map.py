# Copyright (c) 2024-2025 NVIDIA CORPORATION. All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.


import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull
import matplotlib.path as mpath

from spatial_analysis.relationship_utils import project_to_floor

DEBUG_EMPTY_SPACE=False

# Function to create the grid representing the floor
def create_floor_grid(floor_bound, grid_resolution=0.1):

    min_bound = floor_bound[0]
    max_bound = floor_bound[1]

    x_range = np.arange(min_bound[0], max_bound[0], grid_resolution)
    y_range = np.arange(min_bound[1], max_bound[1], grid_resolution)
    
    return np.meshgrid(x_range, y_range)

# Function to mark occupied areas on the grid
def mark_occupied_areas(grid, boxes, occupied, floor=False):
    x_flat = grid[0].ravel()
    y_flat = grid[1].ravel()
    points_array = np.column_stack((x_flat, y_flat))

    for box in boxes:
        projected_points = project_to_floor(box)
        hull = ConvexHull(projected_points)
        hull_vertices = projected_points[hull.vertices]
        path = mpath.Path(hull_vertices)

        # Vectorized point-in-polygon test
        if floor:
            inside = ~path.contains_points(points_array)
        else:
            inside = path.contains_points(points_array)

        # Update the occupied grid
        occupied.ravel()[inside] = True

    return occupied

# Function to find empty areas on the grid
def find_empty_areas(occupied):
    empty_areas = np.logical_not(occupied)
    return empty_areas

def get_empty_space(floor_bound, boxes, grid_resolution=0.01):
    grid = create_floor_grid(floor_bound, grid_resolution)
    empty_occupied = np.zeros(grid[0].shape, dtype=bool)
    occupied = mark_occupied_areas(grid, boxes, empty_occupied)
    empty_areas = find_empty_areas(occupied)
    
    if DEBUG_EMPTY_SPACE:
        plt.figure(figsize=(10, 10))
        
        # Create color-coded grid
        color_grid = np.zeros((*empty_areas.shape, 3), dtype=np.uint8)
        color_grid[empty_areas] = [0, 255, 0]  # Green for empty
        color_grid[~empty_areas] = [255, 0, 0]  # Red for occupied
        
        # Plot the grid
        plt.imshow(color_grid, 
                  extent=[grid[0].min(), grid[0].max(), grid[1].min(), grid[1].max()],
                  origin='lower')
        
        # Plot boxes as outlines
        for box in boxes:
            corners = project_to_floor(box)
            plt.plot(corners[:, 0], corners[:, 1], 'blue', linewidth=2)
            plt.fill(corners[:, 0], corners[:, 1], 'blue', alpha=0.3)
        
        # If floor_box is not a list, plot it as an outline
        min_bound = floor_bound[0]
        max_bound = floor_bound[1]
        plt.plot(min_bound[0], min_bound[1], 'black', linewidth=2)
        plt.fill(min_bound[0], min_bound[1], 'black', alpha=0.1)
        plt.plot(max_bound[0], max_bound[1], 'black', linewidth=2)
        plt.fill(max_bound[0], max_bound[1], 'black', alpha=0.1)
        plt.plot(min_bound[0], max_bound[1], 'black', linewidth=2)
        plt.fill(min_bound[0], max_bound[1], 'black', alpha=0.1)
        plt.plot(max_bound[0], min_bound[1], 'black', linewidth=2)
        plt.fill(max_bound[0], min_bound[1], 'black', alpha=0.1)

        plt.title("Empty Space Grid")
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.grid(True)
        # Create a legend instead of a colorbar
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='red', edgecolor='black', label='Occupied'),
            Patch(facecolor='green', edgecolor='black', label='Empty')
        ]
        plt.legend(handles=legend_elements, loc='lower right')
        plt.show()
    
    return empty_areas, grid, occupied

