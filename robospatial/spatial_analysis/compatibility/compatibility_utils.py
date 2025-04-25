# Copyright (c) 2024-2025 NVIDIA CORPORATION. All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.


import numpy as np
from scipy.spatial import ConvexHull
import matplotlib.path as mpath
import open3d as o3d

from spatial_analysis.relationship_utils import project_to_floor



def is_hull_within_bounds(hull_vertices, bounds):
    """Check if a convex hull is completely within given bounds.
    
    Args:
        hull_vertices (np.ndarray): Vertices of the convex hull
        bounds (tuple): (x_min, x_max, y_min, y_max)
        
    Returns:
        bool: True if hull is within bounds
    """
    x_min, x_max, y_min, y_max = bounds
    hull_min = np.min(hull_vertices, axis=0)
    hull_max = np.max(hull_vertices, axis=0)
    return (hull_min[0] >= x_min and hull_max[0] <= x_max and 
            hull_min[1] >= y_min and hull_max[1] <= y_max)





def can_fit_at_point(grid, box, occupied, point, environment_boxes, box_b, min_distance=0.02, buffer_ratio=0.3, box_name="Object A", box_b_name="Object B", frame=None, direction=None, DEBUG_FIT=False):
    """Check if an object can fit at a given point, trying different rotations.
    
    Args:
        grid: The floor grid (meshgrid output: grid[0]=X, grid[1]=Y)
        box: The object's bounding box
        occupied: Occupancy grid (boolean, same shape as grid[0])
        point: The point (x, y) to try placing at
        environment_boxes: List of all environment boxes
        box_b: The reference object box
        min_distance: Minimum distance required between objects (in meters)
        buffer_ratio: Ratio of buffer zone to object size
        box_name: Name of the object being placed
        box_b_name: Name of the reference object
        frame (str, optional): The reference frame for visualization context.
        direction (str, optional): The direction for visualization context.
    """
    x_coords, y_coords = grid
    h, w = x_coords.shape
    x_min, y_min = x_coords.min(), y_coords.min()
    x_max, y_max = x_coords.max(), y_coords.max()
    bounds = (x_min, x_max, y_min, y_max)
    
    grid_res_x = 0
    grid_res_y = 0
    if w > 1:
         grid_res_x = x_coords[0, 1] - x_coords[0, 0]
    if h > 1:
         grid_res_y = y_coords[1, 0] - y_coords[0, 0]

    if grid_res_x > 1e-6 and grid_res_y > 1e-6:
        grid_resolution = min(grid_res_x, grid_res_y)
    elif grid_res_x > 1e-6:
        grid_resolution = grid_res_x
    elif grid_res_y > 1e-6:
        grid_resolution = grid_res_y
    else:
        grid_resolution = 0 # Indicate failure to determine resolution
        print(f"Warning: Could not determine valid grid resolution for buffer calculation.")

    buffer_size_in_cells = 0 # Default buffer size in grid cells
    if grid_resolution > 1e-6: # Check if grid resolution is valid
        buffer_size_in_cells = int(np.ceil(min_distance / grid_resolution))
    else:
         print(f"Warning: Invalid grid resolution ({grid_resolution}). Setting buffer_size_in_cells to 0.")

    try:
        original_projected_points = project_to_floor(box)
        # Calculate the 2D center of the projected points
        box_center_2d = np.mean(original_projected_points, axis=0)
        # Calculate the initial hull
        initial_hull = ConvexHull(original_projected_points)
        # Store hull vertices relative to the projected center
        relative_hull_vertices = original_projected_points[initial_hull.vertices] - box_center_2d
    except Exception as e:
        return False # Cannot proceed if base hull fails

    rotations = [0, np.pi/4, np.pi/2]
    
    for rotation_idx, rotation in enumerate(rotations):
        # --- Transform the precomputed hull ---
        cos_theta = np.cos(rotation)
        sin_theta = np.sin(rotation)
        rotation_matrix_2d = np.array([
            [cos_theta, -sin_theta],
            [sin_theta,  cos_theta]
        ])
        # Apply rotation to relative vertices
        rotated_vertices = relative_hull_vertices @ rotation_matrix_2d.T # Note the transpose for point rotation
        # Apply translation (move center to the target 'point')
        hull_vertices = rotated_vertices + point[:2] # Use only x,y from point

        # --- Check 1: Is hull within grid bounds? ---
        if not is_hull_within_bounds(hull_vertices, bounds):
            if DEBUG_FIT:
                # Need a 3D rotation matrix for visualization
                rotation_matrix_3d = np.array([
                    [cos_theta, -sin_theta, 0],
                    [sin_theta, cos_theta, 0],
                    [0, 0, 1]
                ])
                visualize_placement(grid, box, occupied, point, environment_boxes, box_b, 
                                   rotation_matrix_3d, f"Rotation {np.degrees(rotation):.0f}° - Out of Bounds", 
                                   bounds, box_name, box_b_name, frame, direction)
            continue
            
        # --- Check 2: Hull Occupancy (Vectorized) ---
        path = mpath.Path(hull_vertices)
        
        # Find the bounding box of the hull to minimize grid points checked
        hull_min_x, hull_min_y = np.min(hull_vertices, axis=0)
        hull_max_x, hull_max_y = np.max(hull_vertices, axis=0)

        # Convert hull bounds to grid indices (clamp to grid dimensions)
        min_ix = np.clip(int(np.floor((hull_min_x - x_min) / grid_res_x)) if grid_res_x > 1e-6 else 0, 0, w - 1)
        max_ix = np.clip(int(np.ceil((hull_max_x - x_min) / grid_res_x)) if grid_res_x > 1e-6 else w - 1, 0, w - 1)
        min_iy = np.clip(int(np.floor((hull_min_y - y_min) / grid_res_y)) if grid_res_y > 1e-6 else 0, 0, h - 1)
        max_iy = np.clip(int(np.ceil((hull_max_y - y_min) / grid_res_y)) if grid_res_y > 1e-6 else h - 1, 0, h - 1)
        
        # Create subset of grid points and indices within the hull's bounding box
        sub_x, sub_y = np.meshgrid(np.arange(min_ix, max_ix + 1), np.arange(min_iy, max_iy + 1))
        sub_points_x = x_coords[sub_y.ravel(), sub_x.ravel()]
        sub_points_y = y_coords[sub_y.ravel(), sub_x.ravel()]
        sub_grid_points = np.vstack((sub_points_x, sub_points_y)).T
        sub_grid_indices = (sub_y.ravel(), sub_x.ravel()) # Indices into the original 'occupied' grid

        if sub_grid_points.size == 0:
            if DEBUG_FIT:
                # Need a 3D rotation matrix for visualization
                rotation_matrix_3d = np.array([
                    [cos_theta, -sin_theta, 0],
                    [sin_theta, cos_theta, 0],
                    [0, 0, 1]
                ])
                visualize_placement(grid, box, occupied, point, environment_boxes, box_b, 
                                   rotation_matrix_3d, f"Rotation {np.degrees(rotation):.0f}° - No Grid Points", 
                                   bounds, box_name, box_b_name, frame, direction)
            continue # Hull is likely too small or outside grid center area
        
        # Check which subset points are inside the actual hull polygon
        inside_hull_mask_flat = path.contains_points(sub_grid_points)
        
        # Get the indices of the grid cells that are inside the hull
        hull_indices_flat = tuple(idx[inside_hull_mask_flat] for idx in sub_grid_indices)
        
        # Check occupancy for points *inside* the hull
        occupied_inside_hull = occupied[hull_indices_flat]
        num_occupied_inside = np.sum(occupied_inside_hull)
        total_cells_inside = len(occupied_inside_hull)

        # --- Check 3: Buffer Zone Occupancy ---
        buffer_occupied_cells = 0
        checked_buffer_indices = set() # Keep track of checked indices to avoid double counting

        # Only check buffer if hull itself is not significantly occupied
        if total_cells_inside == 0 or num_occupied_inside / total_cells_inside <= 0.0: # Do not allow overlap
            # Iterate through the grid cells *inside* the hull
            for iy_hull, ix_hull in zip(*hull_indices_flat):
                # Check the neighborhood (buffer) around this hull cell
                for dy in range(-buffer_size_in_cells, buffer_size_in_cells + 1):
                    for dx in range(-buffer_size_in_cells, buffer_size_in_cells + 1):
                        if dx == 0 and dy == 0:
                            continue # Skip the cell itself

                        ix_buffer = ix_hull + dx
                        iy_buffer = iy_hull + dy

                        # Check if the buffer cell index is valid and hasn't been checked
                        if 0 <= ix_buffer < w and 0 <= iy_buffer < h:
                            buffer_idx_tuple = (iy_buffer, ix_buffer)
                            if buffer_idx_tuple not in checked_buffer_indices:
                                # Check if this buffer cell is outside the hull but occupied
                                buffer_point = (x_coords[iy_buffer, ix_buffer], y_coords[iy_buffer, ix_buffer])
                                if not path.contains_point(buffer_point) and occupied[iy_buffer, ix_buffer]:
                                    buffer_occupied_cells += 1
                                checked_buffer_indices.add(buffer_idx_tuple)
        
        # --- Final Decision for this rotation --- 
        # Conditions: 
        # 1. Hull must have some cells under it. 
        # 2. Significant overlap inside the hull is not allowed. 
        # 3. Significant occupation in the buffer zone is not allowed.
        fit_this_rotation = False
        overlap_ratio = 0
        current_buffer_ratio = 0
        
        if total_cells_inside > 0:
            overlap_ratio = num_occupied_inside / total_cells_inside
            current_buffer_ratio = buffer_occupied_cells / total_cells_inside # Compare buffer count to hull size
            
            # Do not allow overlap and limited buffer occupation (e.g. < 50%)
            # These thresholds might need tuning based on grid resolution and object sizes
            if overlap_ratio <= 0.0 and current_buffer_ratio < buffer_ratio: 
                fit_this_rotation = True
        
        # Debug visualization
        if DEBUG_FIT:
            result_text = "SUCCESS" if fit_this_rotation else "FAILED"
            # Need a 3D rotation matrix for visualization
            rotation_matrix_3d = np.array([
                [cos_theta, -sin_theta, 0],
                [sin_theta, cos_theta, 0],
                [0, 0, 1]
            ])
            visualize_placement(grid, box, occupied, point, environment_boxes, box_b, 
                               rotation_matrix_3d, f"Rotation {np.degrees(rotation):.0f}° - {result_text} (Overlap: {overlap_ratio:.2f}, Buffer: {current_buffer_ratio:.2f})", 
                               bounds, box_name, box_b_name, frame, direction)

        if fit_this_rotation:
            return True # Found a valid rotation

    return False



def visualize_placement(grid, box, occupied, point, environment_boxes, box_b, rotation=None, step_name="", bounds=None, box_name="Object A", box_b_name="Object B", frame=None, direction=None):
    """Visualize the placement attempt in a top-down view.
    
    Args:
        grid: The floor grid
        box: The object's bounding box
        occupied: Occupancy grid
        point: The point to try placing at
        environment_boxes: List of all environment boxes
        box_b: The reference object box
        rotation: Current rotation being tried
        step_name: Name of the current step
        bounds: Grid bounds (x_min, x_max, y_min, y_max)
        box_name: Name of the object being placed
        box_b_name: Name of the reference object
        frame (str, optional): The reference frame (e.g., 'objectcentric').
        direction (str, optional): The direction within the frame (e.g., 'infront').
    """
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle, Polygon
    
    # Create figure
    plt.figure(figsize=(10, 10))
    
    # Create color-coded grid
    color_grid = np.zeros((*occupied.shape, 3), dtype=np.uint8)
    color_grid[occupied] = [255, 0, 0]  # Red for occupied
    color_grid[~occupied] = [0, 255, 0]  # Green for empty
    
    # Plot the grid
    plt.imshow(color_grid, 
              extent=[grid[0].min(), grid[0].max(), grid[1].min(), grid[1].max()],
              origin='lower')
    
    # Plot environment boxes
    for env_box in environment_boxes:
        if env_box == box_b:
            color = 'magenta'  # Reference box
            label = box_b_name
        else:
            color = 'red'
            label = 'Environment'
        corners = project_to_floor(env_box)
        plt.fill(corners[:, 0], corners[:, 1], color, alpha=0.3, label=label)
        plt.plot(corners[:, 0], corners[:, 1], color)
    
    # Plot the current placement attempt
    if rotation is not None:
        translated_box = o3d.geometry.OrientedBoundingBox()
        translated_box.center = np.array([point[0], point[1], box.extent[2] / 2])
        translated_box.R = rotation
        translated_box.extent = box.extent
        
        corners = project_to_floor(translated_box)
        plt.fill(corners[:, 0], corners[:, 1], 'blue', alpha=0.3, label=box_name)
        plt.plot(corners[:, 0], corners[:, 1], 'blue')
    
    # Plot the point
    plt.scatter(point[0], point[1], c='yellow', s=100, marker='*', label='Target Point')
    
    # Add title and labels
    title = f"Placement Attempt: {step_name}"
    if frame and direction:
        title += f" ({frame} - {direction} relative to {box_b_name})"
    else:
        title += f" (relative to {box_b_name})"
    plt.title(title)
    plt.xlabel('X')
    plt.ylabel('Y')
    
    # Show grid bounds if provided
    if bounds:
        x_min, x_max, y_min, y_max = bounds
        plt.axvline(x=x_min, color='k', linestyle='--')
        plt.axvline(x=x_max, color='k', linestyle='--')
        plt.axhline(y=y_min, color='k', linestyle='--')
        plt.axhline(y=y_max, color='k', linestyle='--')
    
    plt.grid(True)
    plt.legend()
    plt.show()