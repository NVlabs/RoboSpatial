# Copyright (c) 2024-2025 NVIDIA CORPORATION. All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.


import numpy as np

def get_object_grounding(obj, occupancy_map):
    """
    Generates 2D bounding boxes based on pixel occupancy for a given 3D object.
    Uses a pre-calculated occupancy map.

    Args:
        obj (dict): A single object dict with at least 'obb' and 'name'.
        extrinsic (np.ndarray): Extrinsic matrix.
        intrinsic (np.ndarray): Intrinsic matrix.
        image_size (tuple): (width, height).
        occupancy_map (np.ndarray): Pre-calculated 2D boolean occupancy map for this object.

    Returns:
        dict or None: A dictionary containing:
              - 'name': Object name (str)
              - 'clipped_bbox': Axis-aligned clipped [xmin, ymin, xmax, ymax] (list)
              - 'square_bbox': Axis-aligned square clipped [xmin, ymin, xmax, ymax] (list)
              - 'bbox_3d': Original 3D bounding box coordinates (list) - if available in input
              - 'obb': Original OrientedBoundingBox - needed later for occupancy map
              Returns None if the object has no OBB or is not visible.
    """
    if 'obb' not in obj:
        print(f"Warning: Skipping object {obj.get('name', 'Unknown')} because 'obb' is missing.")
        return None
    if occupancy_map is None:
         print(f"Warning: Occupancy map is None for object {obj.get('name', 'Unknown')}. Cannot calculate grounding.")
         return None

    # Use the provided occupancy_map
    occupied_coords = np.argwhere(occupancy_map) # Shape (N, 2), cols=[row, col] -> (y, x)

    if occupied_coords.size == 0:
        # Object does not project onto any pixels
        # Object does not project onto any pixels according to the map
        return None

    # --- Calculate Clipped Axis-Aligned BBox from Occupied Pixels ---
    # Remember: occupied_coords are (row, col) -> (y, x)
    clipped_min_y = np.min(occupied_coords[:, 0])
    clipped_max_y = np.max(occupied_coords[:, 0])
    clipped_min_x = np.min(occupied_coords[:, 1])
    clipped_max_x = np.max(occupied_coords[:, 1])

    clipped_coords_bbox = [clipped_min_x, clipped_min_y, clipped_max_x, clipped_max_y]

    # --- Store Info ---
    info = {
        'name': obj.get("name"),
        'clipped_bbox': [float(c) for c in clipped_coords_bbox], # Ensure floats
        'bbox_3d': obj.get("bbox_3d"),
        'obb': obj['obb']
    }
    # Return the dictionary directly, not a list containing the dictionary
    return info