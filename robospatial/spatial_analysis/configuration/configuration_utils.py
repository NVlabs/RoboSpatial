# Copyright (c) 2024-2025 NVIDIA CORPORATION. All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

import numpy as np

from spatial_analysis.relationship_utils import get_min_max_visible_depth_for_box


# --- Calculate Metrics for Both Objects ---
def get_object_metrics(obj, extrinsic, intrinsic, image_size, individual_occupancy_maps):
    metrics = {}
    obb = obj.get('obb') # Use .get for safety
    obj_name = obj.get('name') # Get object name

    if obb is None or obj_name is None: # Check if retrieval failed
            print(f"Warning: Skipping metrics calculation due to missing 'obb' or 'name' in object: {obj}")
            return {'is_visible': False} # Return basic structure indicating not visible

    world_coords = np.asarray(obb.get_box_points())

    # Pixel Occupancy & 2D Metrics
    # --- Retrieve pre-calculated occupancy map ---
    occupancy_map = individual_occupancy_maps.get(obj_name)
    if occupancy_map is None:
            # Removed fallback calculation, error is printed instead.
            print(f"Error: Occupancy map not found for '{obj_name}' in get_object_metrics. Aborting.")
            return {'is_visible': False}

    occupied_coords = np.argwhere(occupancy_map) # Shape (N, 2), cols=[row, col]

    # Visible Depth Range
    min_depth, max_depth = get_min_max_visible_depth_for_box(obb, extrinsic, intrinsic, image_size)

    # Determine overall visibility
    metrics['is_visible'] = occupied_coords.size > 0 and min_depth is not None

    if metrics['is_visible']:
        metrics['min_px'] = np.min(occupied_coords[:, 1]) # Min X pixel
        metrics['max_px'] = np.max(occupied_coords[:, 1]) # Max X pixel
        metrics['min_py'] = np.min(occupied_coords[:, 0]) # Min Y pixel
        metrics['max_py'] = np.max(occupied_coords[:, 0]) # Max Y pixel
        metrics['center_px'] = np.mean(occupied_coords[:, ::-1], axis=0) # Centroid (x, y)
        metrics['min_depth'] = min_depth
        metrics['max_depth'] = max_depth
        metrics['visible_depth_avg'] = (min_depth + max_depth) / 2.0
    else:
        # Set pixel/depth metrics to None if not visible
        metrics['min_px'] = metrics['max_px'] = metrics['min_py'] = metrics['max_py'] = None
        metrics['center_px'] = None
        metrics['min_depth'] = metrics['max_depth'] = None
        metrics['visible_depth_avg'] = None

    # World Z Metrics (calculated regardless of visibility)
    metrics['world_z_min'] = np.min(world_coords[:, 2])
    metrics['world_z_max'] = np.max(world_coords[:, 2])
    # Handle potential empty world_coords if needed, though unlikely for OBB
    if world_coords.size > 0:
            metrics['world_z_avg'] = np.mean(world_coords[:, 2])
    else:
            metrics['world_z_avg'] = None # Should not happen with OBB

    return metrics