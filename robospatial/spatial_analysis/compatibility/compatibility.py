# Copyright (c) 2024-2025 NVIDIA CORPORATION. All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

import numpy as np

from spatial_analysis.context.context import get_point_in_space_relative_to_object
from spatial_analysis.topdown_map import get_empty_space
from spatial_analysis.compatibility.compatibility_utils import can_fit_at_point

DEBUG_FIT=False

def can_fit_object_a_in_relation_to_b(
    floor_bound,
    environment_boxes,
    obj_a,        # Dictionary of the object being placed (contains name, obb, etc.)
    obj_b,        # Dictionary of the reference object (contains name, obb, etc.)
    have_face,
    extrinsic,
    intrinsic,
    image_size,
    image_path,
    grid_resolution,
    num_samples,
    individual_occupancy_maps, 
    env_occupancy_map,         
    threshold=0.5,
    min_distance=0.2,
    buffer_ratio=0.3
):
    """Checks if object A (the target object) can be placed in empty space relative to object B (the reference object).

    The function operates in several steps:
    1.  Calculates the available empty space in the environment using `get_empty_space`, generating a 2D grid representation.
    2.  Determines a dynamic threshold for empty space sampling based on the sizes of object A and B.
    3.  Samples a set of potential placement points (`num_samples`) around object B in various directions
        (infront, behind, left, right) across different reference frames (object-centric, camera-centric, world-centric)
        using `get_point_in_space_relative_to_object`. This sampling considers precomputed occupancy maps.
    4.  For each sampled 3D point corresponding to a specific frame and direction, it checks if object A's
        oriented bounding box (OBB) can be placed at that point without colliding with the environment or object B using `can_fit_at_point`. This check utilizes the 2D occupancy grid.
    5.  Aggregates the results, indicating whether *at least one* valid placement position was found for 
        each frame/direction combination.

    Args:
        floor_bound (list): A list defining the bounding box of the walkable floor area, used for empty space calculation.
        environment_boxes (list): A list of open3d.geometry.OrientedBoundingBox objects representing static obstacles
                                  in the environment.
        obj_a (dict): Dictionary representing object A (the object being placed), must contain 'obb' (open3d OBB)
                      and 'name' (str).
        obj_b (dict): Dictionary representing the reference object B, must contain 'obb' (open3d OBB) and 'name' (str).
        have_face (bool): Indicates if object B has a defined 'face' or primary orientation, affecting object-centric sampling.
        extrinsic (np.ndarray): 4x4 camera extrinsic matrix (camera-to-world transformation).
        intrinsic (np.ndarray): 3x3 or 4x4 camera intrinsic matrix (only the top-left 3x3 portion is used if 4x4).
        image_size (tuple): Size of the image (width, height), used for camera-centric calculations.
        image_path (str): Path to the associated scene image, primarily used for debugging visualizations within called functions.
        grid_resolution (float): The resolution (e.g., meters per grid cell) of the 2D occupancy grid used for collision checks.
        num_samples (int): The number of candidate points to sample around object B for potential placement checks.
        individual_occupancy_maps (dict): Precomputed 2D occupancy numpy arrays for each individual dynamic object (including A and B).
                                          Keys are object names, values are the occupancy map arrays.
        env_occupancy_map (np.ndarray): Precomputed combined 2D occupancy numpy array representing the static environment.
        threshold (float, optional): Base distance threshold used in empty space calculation. This is dynamically adjusted based on
                                     object sizes. Defaults to 0.5.

    Returns:
        dict: A nested dictionary indicating whether a valid placement was found for object A relative to object B
              for each combination of reference frame and direction.
              Example: `{'objectcentric': {'infront': True, 'behind': False, ...}, 'cameracentric': {...}, ...}`
              `True` means at least one valid point was found for that relative position.
    """
    empty_areas, grid, occupied = get_empty_space(floor_bound, environment_boxes, grid_resolution)

    box_a = obj_a['obb'] # Extract OBB from obj_a dictionary
    obj_a_name = obj_a['name'] # Extract name from obj_a dictionary
    box_b = obj_b['obb'] # Extract OBB from obj_b dictionary
    obj_b_name = obj_b['name'] # Extract name from obj_b dictionary

    # Adjust the sampling distance threshold based on the average horizontal size of the two objects.
    max_extent_a = np.max(box_a.extent[:2])  # Max extent in world x-y plane
    max_extent_b = np.max(box_b.extent[:2])  # Max extent in world x-y plane
    dynamic_threshold = threshold + (max_extent_a + max_extent_b) / 2

    # Sample potential placement points around obj_b using the precomputed occupancy information.
    _, _, visible_points_3d_all, _ = get_point_in_space_relative_to_object(
        floor_bound, environment_boxes,
        ref_obj=obj_b,
        extrinsic=extrinsic, intrinsic=intrinsic, image_size=image_size,
        have_face=have_face, num_samples=num_samples,
        individual_occupancy_maps=individual_occupancy_maps,
        env_occupancy_map=env_occupancy_map,
        threshold=dynamic_threshold, grid_resolution=grid_resolution,
        image_path=image_path,
        empty_areas=empty_areas, grid=grid, occupied=occupied,
    )

    results = {}

    # Check placement possibility for each defined reference frame and direction
    frames_to_check = ['objectcentric', 'cameracentric', 'worldcentric']
    directions_to_check = ['infront', 'behind', 'left', 'right']

    for frame in frames_to_check:
        results[frame] = {}
        for direction in directions_to_check:
            # Retrieve the list of 3D candidate points sampled for this specific frame/direction
            points_in_direction_3d = visible_points_3d_all.get(frame, {}).get(direction, [])

            if not points_in_direction_3d:
                results[frame][direction] = False # No candidate points found for this relative position
                continue

            # Check if obj_a fits at any of the sampled points without collision
            can_fit = False
            for point_3d in points_in_direction_3d:
                # Check collision using the 2D grid, environment OBBs, and the reference object B's OBB
                if can_fit_at_point(grid, box_a, occupied, point_3d[:2], environment_boxes, box_b, min_distance=min_distance, buffer_ratio=buffer_ratio, box_name=obj_a_name, box_b_name=obj_b_name, frame=frame, direction=direction, DEBUG_FIT=DEBUG_FIT):
                    can_fit = True
                    break # Found a valid spot for this direction

            results[frame][direction] = can_fit

    if DEBUG_FIT:
        # Print final fitting results if debug flag is enabled
        print(f"Can fit results for {obj_a_name} relative to {obj_b_name}:")
        print(results)

    return results


def can_fit_on_top(top_box, base_box):
    """Determines if the top OrientedBoundingBox can fit horizontally on top of the base OrientedBoundingBox.
        
    Args:
        top_box (o3d.geometry.OrientedBoundingBox): The bounding box of the object to be placed on top.
        base_box (o3d.geometry.OrientedBoundingBox): The bounding box of the base object.
        
    Returns:
        bool: True if the top box's x and y extents are less than or equal to the base box's,
              False otherwise.
    """
    base_extent = base_box.extent
    top_extent = top_box.extent
    
    # Simple check: Top object's horizontal dimensions must be <= base object's dimensions.
    # Assumes alignment of the boxes' principal axes with the world axes for this check.
    if (top_extent[0] <= base_extent[0] and
        top_extent[1] <= base_extent[1]):
        result = True
    else:
        result = False
    return result