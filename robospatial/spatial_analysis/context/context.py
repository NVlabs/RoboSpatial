# Copyright (c) 2024-2025 NVIDIA CORPORATION. All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

import numpy as np
import cv2
import matplotlib.pyplot as plt
import random


from spatial_analysis.topdown_map import get_empty_space
from spatial_analysis.relationship_utils import project_to_floor, get_min_max_visible_depth_for_box
from spatial_analysis.context.context_utils import compute_distances_to_bbox, project_points_to_image


DEBUG_POINT=False


def get_point_in_space_relative_to_object(
    floor_bound,
    environment_boxes,
    ref_obj,
    extrinsic,
    intrinsic,
    image_size,
    have_face,
    num_samples,
    individual_occupancy_maps, # Added
    env_occupancy_map,         # Combined map for environment check
    threshold=0.5,
    grid_resolution=0.001,
    image_path=None,
    empty_areas=None,
    grid=None,
    occupied=None,
):
    """
    Samples potentially reachable points in the environment relative to a reference object.

    This function identifies empty space near a specified reference object, projects these
    candidate points into the camera view, categorizes them based on their spatial
    relationship to the object (infront, behind, left, right) in different reference
    frames (object-centric, camera-centric, world-centric), filters out points
    occluded by other objects in the image, and finally samples a specified number
    of points for each category.

    Args:
        floor_bound (tuple): Min/max coordinates defining the floor area.
        environment_boxes (list): List of OBBs representing objects in the environment (excluding the reference object).
        ref_obj (dict): Dictionary containing 'obb' (OrientedBoundingBox) and 'name' (str) of the reference object.
        extrinsic (np.ndarray): 4x4 camera extrinsic matrix (world to camera).
        intrinsic (np.ndarray): 3x3 camera intrinsic matrix.
        image_size (tuple): (width, height) of the image.
        have_face (bool): Flag indicating if the reference object has a defined front face (for object-centric directions).
        num_samples (int): The maximum number of points to sample for each category.
        individual_occupancy_maps (dict): Dictionary mapping object names to their 2D image occupancy maps (boolean arrays).
        env_occupancy_map (np.ndarray): Combined 2D image occupancy map for the entire environment.
        threshold (float, optional): Maximum distance (in world units) from the reference object's 2D footprint to consider points. Defaults to 0.5.
        grid_resolution (float, optional): Resolution for the top-down grid generation. Defaults to 0.001.
        image_path (str, optional): Path to the image file for debugging visualization. Defaults to None.
        empty_areas (np.ndarray, optional): Pre-computed boolean grid indicating empty space. If None, it's computed.
        grid (tuple, optional): Pre-computed grid coordinates (X, Y). If None, it's computed.
        occupied (np.ndarray, optional): Pre-computed boolean grid indicating occupied space. If None, it's computed.

    Returns:
        tuple:
            - sampled_points (dict): Dictionary containing sampled 2D pixel coordinates categorized by frame and direction.
            - sampled_3d_points (dict): Dictionary containing sampled 3D world coordinates categorized by frame and direction.
            - visible_points_3d_all (dict): Dictionary containing *all* potentially visible 3D points (before sampling and occlusion check) categorized by frame and direction.
            - generated_something (bool): Flag indicating if any valid, non-occluded points were found and added to the potential samples.
    """

    # --- Initialization and Grid Calculation ---
    # If pre-computed grid information is not provided, calculate it based on floor bounds and environment objects.
    if empty_areas is None or grid is None or occupied is None:
        empty_areas, grid, occupied = get_empty_space(floor_bound, environment_boxes, grid_resolution)
    
    ref_obb = ref_obj['obb'] # Get OBB from the dictionary
    ref_obj_name = ref_obj['name'] # Get name from the dictionary

    # Project reference object's OBB to the floor (2D footprint).
    box_a_corners = project_to_floor(ref_obb) # Use ref_obb

    # Flatten the grid and identify points that are *not* occupied by environment objects initially.
    X_flat = grid[0].flatten()
    Y_flat = grid[1].flatten()
    points = np.vstack((X_flat, Y_flat)).T  # Shape: (N, 2)
    occupied_flat = occupied.flatten()
    not_occupied_mask = ~occupied_flat
    points_to_check = points[not_occupied_mask]

    # --- Identify Points Near Reference Object ---
    # Calculate distances from empty points to the reference object's 2D footprint.
    distances = compute_distances_to_bbox(points_to_check, box_a_corners)

    # Filter points to keep only those within the specified distance threshold.
    within_threshold_mask = distances <= threshold
    points_within_distance = points_to_check[within_threshold_mask]

    # Handle the case where no points are found close enough to the object.
    if len(points_within_distance) == 0:
        # Return empty dictionaries if no suitable points are found nearby.
        sampled_points = {
            'objectcentric': {key: [] for key in ['infront', 'behind', 'left', 'right']},
            'cameracentric': {key: [] for key in ['infront', 'behind', 'left', 'right']},
            'worldcentric': {key: [] for key in ['infront', 'behind', 'left', 'right']}
        }
        sampled_3d_points = {
            'objectcentric': {key: [] for key in ['infront', 'behind', 'left', 'right']},
            'cameracentric': {key: [] for key in ['infront', 'behind', 'left', 'right']},
            'worldcentric': {key: [] for key in ['infront', 'behind', 'left', 'right']}
        }
        visible_points_3d_all = {
            'objectcentric': {key: [] for key in ['infront', 'behind', 'left', 'right']},
            'cameracentric': {key: [] for key in ['infront', 'behind', 'left', 'right']},
            'worldcentric': {key: [] for key in ['infront', 'behind', 'left', 'right']}
        }
        return sampled_points, sampled_3d_points, visible_points_3d_all, False

    # --- Project Points to Image and Filter Visibility ---
    # Elevate the 2D floor points to 3D using the minimum Z-coordinate of the reference object's OBB.
    z_coordinate = np.min(np.asarray(ref_obb.get_box_points())[:, 2]) # Use ref_obb
    points_within_distance_3d = np.hstack((
        points_within_distance,
        np.full((points_within_distance.shape[0], 1), z_coordinate)
    ))

    # Project the 3D points onto the 2D image plane.
    points_pixel, points_depth = project_points_to_image(points_within_distance_3d, extrinsic, intrinsic)
    points_pixel = np.array(points_pixel)
    points_depth = np.array(points_depth)

    # Filter out points that project behind the camera (negative depth).
    valid_indices = points_depth > 0
    points_pixel = points_pixel[valid_indices]
    points_depth = points_depth[valid_indices] # Filter depth as well
    points_within_distance = points_within_distance[valid_indices]
    points_within_distance_3d = points_within_distance_3d[valid_indices] # Filter 3D points

    # Filter out points that project outside the image boundaries.
    def is_point_in_image_array(points, image_size):
        # Check if projected pixel coordinates fall within the image dimensions.
        x = points[:, 0]
        y = points[:, 1]
        result = (x >= 0) & (x < image_size[0]) & (y >= 0) & (y < image_size[1])
        return result
    in_image_mask = is_point_in_image_array(points_pixel, image_size)
    points_pixel_visible = points_pixel[in_image_mask]
    points_depth_visible = points_depth[in_image_mask] # Filter depth based on image visibility
    points_within_distance = points_within_distance[in_image_mask]
    points_within_distance_3d = points_within_distance_3d[in_image_mask] # These are all potentially visible 3D points

    # --- Initialize Result Dictionaries and Directional Helpers ---
    # Dictionaries to store categorized points (all visible, temporarily sampled 2D, temporarily sampled 3D).
    visible_points_3d_all = {
        frame: {direction: [] for direction in ['infront', 'behind', 'left', 'right']}
        for frame in ['objectcentric', 'cameracentric', 'worldcentric']
    }
    # Temporary lists to hold points before the final sampling step.
    temp_sampled_points = {
        frame: {direction: [] for direction in ['infront', 'behind', 'left', 'right']}
        for frame in ['objectcentric', 'cameracentric', 'worldcentric']
    }
    temp_sampled_3d_points = {
        frame: {direction: [] for direction in ['infront', 'behind', 'left', 'right']}
        for frame in ['objectcentric', 'cameracentric', 'worldcentric']
    }
    generated_something = False # Flag to track if any valid points are found

    # --- Calculate Helpers for Direction Checking ---
    # Extract properties from the reference object's OBB needed for direction calculations.
    R = ref_obb.R # Use ref_obb (Rotation matrix defines local axes)
    extents = ref_obb.extent # Use ref_obb
    half_extents = extents / 2
    x_axis = R[:, 0] # Object's local X axis (often considered 'forward')
    y_axis = R[:, 1] # Object's local Y axis (often considered 'left')

    # --- Camera/World-centric: Calculate Reference Object Bounds in Image ---
    # Calculate min/max horizontal pixel coordinates occupied by the *reference object*.
    # Retrieve the specific occupancy map for the reference object.
    ref_obj_occupancy_map = individual_occupancy_maps.get(ref_obj_name)
    if ref_obj_occupancy_map is None:
        # Fallback if the map is missing (should ideally not happen).
        print(f"Error: Occupancy map not found for '{ref_obj_name}' in get_point_in_space_relative_to_object. Using empty map.")
        ref_obj_occupancy_map = np.zeros((image_size[1], image_size[0]), dtype=bool) # h, w

    occupied_pixels_coords = np.argwhere(ref_obj_occupancy_map) # Get (row, col) indices where True
    if occupied_pixels_coords.size > 0:
        min_x_bound = np.min(occupied_pixels_coords[:, 1]) # Min column index (x) for the ref obj
        max_x_bound = np.max(occupied_pixels_coords[:, 1]) # Max column index (x) for the ref obj
    else:
        # Handle case where the reference object doesn't project onto the image.
        min_x_bound = None
        max_x_bound = None

    # Calculate the range of depths occupied by the reference object in the camera view.
    min_visible_depth, max_visible_depth = get_min_max_visible_depth_for_box(ref_obb, extrinsic, intrinsic, image_size) # Use ref_obb

    # --- Calculate Image Occupancy for Candidate Points ---
    # Use the pre-computed occupancy map of the *environment* (excluding the reference object).
    image_occupancy_map = env_occupancy_map # Combined map for checking occlusion of candidate points

    def is_point_occupied_array(occupancy_map, points):
        # Check if candidate points (projected pixels) fall on occupied pixels in the environment map.
        # Use clipping and rounding consistent with projection to safely index the map.
        x = np.clip(np.round(points[:, 0]), 0, image_size[0] - 1).astype(int)
        y = np.clip(np.round(points[:, 1]), 0, image_size[1] - 1).astype(int)
        # Check occupancy using the calculated indices.
        occupied = occupancy_map[y, x]
        return occupied
    # Initialize occupancy mask for all potentially visible points.
    occupied_mask = np.zeros(points_pixel_visible.shape[0], dtype=bool)
    if points_pixel_visible.shape[0] > 0:
        # Determine which potentially visible points are occluded by the environment.
         occupied_mask = is_point_occupied_array(image_occupancy_map, points_pixel_visible)

    # --- Iterate Through Visible Points, Categorize Directions, and Check Occlusion ---
    if points_within_distance_3d.shape[0] > 0:
        for i in range(points_within_distance_3d.shape[0]):
            point_3d = points_within_distance_3d[i]      # 3D coords of the candidate point
            point_pixel = points_pixel_visible[i]     # 2D pixel coords of the candidate point
            point_depth = points_depth_visible[i]     # Depth of the candidate point
            is_occupied = occupied_mask[i]            # Is this point occluded by the environment?

            # --- Object-centric Direction Check ---
            # This check only runs if the object has a defined 'face'.
            # It uses the object's local coordinate system defined by its OBB.
            if have_face:
                # Calculate the 3D vector from the object's center to the candidate point.
                relative_vector_3d = point_3d - ref_obb.center
                
                # Project this vector onto the object's local X (forward) and Y (left) axes.
                dot_forward = np.dot(relative_vector_3d, x_axis)
                dot_left = np.dot(relative_vector_3d, y_axis) # Assuming Y is 'left'

                # Determine direction based on projection relative to object's extents.
                obj_directions = []
                extent_epsilon = 1e-4 # Small epsilon to handle points exactly on the boundary.
                # Check if point is beyond the front/back faces.
                if dot_forward > half_extents[0] + extent_epsilon:
                    obj_directions.append('infront')
                elif dot_forward < -half_extents[0] - extent_epsilon:
                    obj_directions.append('behind')
                # Check if point is beyond the left/right faces.
                if dot_left > half_extents[1] + extent_epsilon: # Positive Y projection -> left
                    obj_directions.append('left')
                elif dot_left < -half_extents[1] - extent_epsilon: # Negative Y projection -> right
                    obj_directions.append('right')

                # If the point falls into a category, store it.
                if obj_directions:
                    for direction in obj_directions:
                         # Store all potentially visible points first.
                         visible_points_3d_all['objectcentric'][direction].append(point_3d.tolist())
                         # If the point is *not* occluded by the environment, add it to the temporary sampling lists.
                         if not is_occupied:
                             temp_sampled_points['objectcentric'][direction].append(point_pixel.tolist())
                             temp_sampled_3d_points['objectcentric'][direction].append(point_3d.tolist())
                             generated_something = True # Mark that we found at least one valid point.

            # --- Camera-centric / World-centric Direction Check ---
            # These frames use image-based and depth-based comparisons relative to the reference object's projection.
            px, py = point_pixel # Pixel coordinates of the candidate point

            # Check Left/Right: Compare candidate point's x-pixel to the min/max x-pixels of the reference object.
            is_left = min_x_bound is not None and px < min_x_bound
            is_right = max_x_bound is not None and px > max_x_bound

            # Check Infront/Behind: Compare candidate point's depth to the min/max depth of the reference object.
            is_infront = (min_visible_depth is not None) and (point_depth < min_visible_depth) # Closer to camera
            is_behind = (max_visible_depth is not None) and (point_depth > max_visible_depth) # Farther from camera

            # Assign directions (a point can be both left/right AND infront/behind).
            cam_directions = []
            if is_left: cam_directions.append('left')
            if is_right: cam_directions.append('right')
            if is_behind: cam_directions.append('behind')
            if is_infront: cam_directions.append('infront')

            # If the point falls into any camera/world category...
            if cam_directions:
                for direction in cam_directions:
                    # Store all potentially visible points (same logic applies to both camera and world).
                    visible_points_3d_all['cameracentric'][direction].append(point_3d.tolist())
                    visible_points_3d_all['worldcentric'][direction].append(point_3d.tolist())
                    # If the point is *not* occluded by the environment, add it to the temporary sampling lists.
                    if not is_occupied:
                        temp_sampled_points['cameracentric'][direction].append(point_pixel.tolist())
                        temp_sampled_3d_points['cameracentric'][direction].append(point_3d.tolist())
                        temp_sampled_points['worldcentric'][direction].append(point_pixel.tolist())
                        temp_sampled_3d_points['worldcentric'][direction].append(point_3d.tolist())
                        generated_something = True # Mark that we found at least one valid point.

    # --- Final Sampling Step ---
    # Initialize final dictionaries to store the randomly sampled points.
    sampled_points = {
        frame: {direction: [] for direction in ['infront', 'behind', 'left', 'right']}
        for frame in ['objectcentric', 'cameracentric', 'worldcentric']
    }
    sampled_3d_points = {
        frame: {direction: [] for direction in ['infront', 'behind', 'left', 'right']}
        for frame in ['objectcentric', 'cameracentric', 'worldcentric']
    }

    def uniform_sample(points, num_samples):
        """Helper function to randomly sample points from a list."""
        if not points: # Handle empty list.
            return []
        # If more points available than requested, sample randomly.
        if len(points) > num_samples:
            indices = random.sample(range(len(points)), num_samples)
            return [points[i] for i in indices]
        # Otherwise, return all available points.
        else:
            return points

    # Perform uniform sampling for each category from the non-occluded points found.
    for frame in temp_sampled_points.keys():
        for direction in temp_sampled_points[frame].keys():
            # Sample both the 2D pixel coordinates and the corresponding 3D world coordinates.
            sampled_points[frame][direction] = uniform_sample(temp_sampled_points[frame][direction], num_samples)
            sampled_3d_points[frame][direction] = uniform_sample(temp_sampled_3d_points[frame][direction], num_samples)
    
    # --- Debug Visualization ---
    # If DEBUG_POINT is True, an image path is provided, and valid points were generated,
    # create plots showing the occupancy map, sampled 2D points on the image, and
    # all potentially visible 3D points projected onto the top-down grid.
    if DEBUG_POINT and generated_something and image_path:
        # Import necessary libraries locally for plotting.
        import math
        import os
        image = cv2.imread(image_path)
        if image is not None:
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            # Define colors and markers for different frames and directions.
            colors = {'objectcentric': 'red', 'cameracentric': 'blue', 'worldcentric': 'green'}
            markers = {'infront': 'o', 'behind': 's', 'left': '^', 'right': 'v'}

            # --- Figure 0: Visualize Environment Occupancy Map ---
            plt.figure(figsize=(10, 8))
            plt.imshow(env_occupancy_map, cmap='gray', interpolation='none')
            plt.title(f"Image Occupancy Map ({os.path.basename(image_path)}) - Ref: {ref_obj_name}")
            plt.xlabel('X Pixel')
            plt.ylabel('Y Pixel')

            # --- Figure 1: Grid of Image Visualizations with Sampled 2D points ---
            plots_to_make_2d = []
            # Collect the *sampled* 2D points for plotting.
            for frame, directions_data in sampled_points.items():
                for direction, points_list in directions_data.items():
                    if points_list:
                        plots_to_make_2d.append((frame, direction, np.array(points_list)))

            if plots_to_make_2d:
                num_plots_2d = len(plots_to_make_2d)
                ncols_2d = min(4, num_plots_2d) # Arrange plots in up to 4 columns.
                nrows_2d = math.ceil(num_plots_2d / ncols_2d)
                fig_imgs, ax_imgs = plt.subplots(nrows_2d, ncols_2d, figsize=(ncols_2d * 6, nrows_2d * 5), squeeze=False)
                ax_flat_imgs = ax_imgs.flatten()

                for i, (frame, direction, points_array_2d) in enumerate(plots_to_make_2d):
                    ax = ax_flat_imgs[i]
                    color = colors.get(frame, 'black')
                    marker = markers.get(direction, 'x')

                    ax.imshow(image_rgb) # Display the original image.
                    # Overlay the sampled 2D points.
                    ax.scatter(points_array_2d[:, 0], points_array_2d[:, 1], c=color, marker=marker, s=20, alpha=0.7)
                    ax.set_title(f"Image: {frame} - {direction} (Ref: {ref_obj_name})")
                    ax.axis('off') # Hide plot axes.

                # Hide axes for any unused subplots in the grid.
                for i in range(num_plots_2d, len(ax_flat_imgs)):
                    ax_flat_imgs[i].axis('off')

                fig_imgs.suptitle(f"Sampled 2D Points on Image ({os.path.basename(image_path)}) - Ref: {ref_obj_name}", fontsize=16)
                fig_imgs.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust layout.

            # --- Figure 2: Grid of Top-Down Visualizations with *All* Visible 3D points ---
            # Create a colored grid representing the initial top-down empty/occupied space.
            color_grid = np.zeros((*empty_areas.shape, 3), dtype=np.uint8)
            color_grid[empty_areas] = [0, 255, 0]  # Green = empty
            color_grid[~empty_areas] = [255, 0, 0]  # Red = occupied by environment

            # Get the 2D footprint of the reference object for plotting.
            ref_box_corners_2d = project_to_floor(ref_obb)

            # Collect *all* potentially visible 3D points (before sampling/occlusion check) for plotting.
            plots_to_make = []
            for frame, directions_data in visible_points_3d_all.items():
                for direction, points_list_3d in directions_data.items():
                    if points_list_3d:
                        plots_to_make.append((frame, direction, np.array(points_list_3d)))

            if plots_to_make:
                num_plots = len(plots_to_make)
                ncols = min(4, num_plots) # Arrange plots in up to 4 columns.
                nrows = math.ceil(num_plots / ncols)
                fig_grids, ax_grids = plt.subplots(nrows, ncols, figsize=(ncols * 5, nrows * 5), squeeze=False)
                ax_flat = ax_grids.flatten()

                for i, (frame, direction, points_array_3d) in enumerate(plots_to_make):
                    ax = ax_flat[i]
                    color = colors.get(frame, 'black')
                    marker = markers.get(direction, 'x')

                    # Plot the background empty/occupied grid.
                    ax.imshow(color_grid, extent=[grid[0].min(), grid[0].max(), grid[1].min(), grid[1].max()], origin='lower')

                    # Plot the reference object's 2D footprint.
                    ax.plot(ref_box_corners_2d[:, 0], ref_box_corners_2d[:, 1], 'k-', linewidth=1)
                    ax.fill(ref_box_corners_2d[:, 0], ref_box_corners_2d[:, 1], 'magenta', alpha=0.5, label=f'{ref_obj_name} (Ref)')

                    # Plot the 2D projection (X, Y) of the visible 3D points for this category.
                    ax.scatter(points_array_3d[:, 0], points_array_3d[:, 1], c=color, marker=marker, s=20, alpha=0.7)

                    ax.set_title(f'Top-Down: {frame} - {direction} (Ref: {ref_obj_name})')
                    ax.set_xlabel('X')
                    ax.set_ylabel('Y')
                    ax.legend()

                # Hide axes for any unused subplots.
                for i in range(num_plots, len(ax_flat)):
                    ax_flat[i].axis('off')

                fig_grids.tight_layout()

            # --- Display Plots ---
            plt.show()
        elif image_path is None and generated_something:
            # Warning if debugging is on and points were found, but no image was provided.
            print("Warning: Points generated in get_point_in_space_relative_to_object, but no image_path provided for visualization.")
    # --- Visualization End ---

    # Return the final sampled points (2D and 3D), all visible points, and the flag.
    return sampled_points, sampled_3d_points, visible_points_3d_all, generated_something