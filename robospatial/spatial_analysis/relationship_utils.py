# Copyright (c) 2024-2025 NVIDIA CORPORATION. All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

# Utils for spatial relationships
import numpy as np
from matplotlib.path import Path
import open3d as o3d
from scipy.spatial import ConvexHull




def get_min_max_visible_depth_for_box(box, extrinsic, intrinsic, image_size):
    """
    Calculates the minimum and maximum depth of the visible parts of a box.

    Args:
        box (o3d.geometry.OrientedBoundingBox): The bounding box.
        extrinsic (np.ndarray): 4x4 extrinsic matrix (camera to world).
        intrinsic (np.ndarray): 4x4 camera intrinsic matrix.
        image_size (tuple): (width, height) of the image.

    Returns:
        tuple: (min_visible_depth, max_visible_depth) or (None, None) if the box
               is not visible or behind the camera.
    """
    width, height = image_size
    extrinsic_w2c = np.linalg.inv(extrinsic)
    EPS = 1e-6 # Small epsilon for depth checks

    # 1. Get world corners
    corners_world = np.asarray(box.get_box_points())
    corners_world_hom = np.hstack((corners_world, np.ones((8, 1)))) # Homogeneous coordinates

    # 2. Transform to camera coordinates
    corners_cam_hom = corners_world_hom @ extrinsic_w2c.T
    
    # Validate transformation results
    if not np.all(np.isfinite(corners_cam_hom)):
        # print("Warning: Non-finite values encountered during world to camera transformation.")
        return None, None

    # corners_cam = corners_cam_hom[:, :3] / corners_cam_hom[:, 3][:, np.newaxis] # Normalize if W is not 1
    corners_cam = corners_cam_hom[:, :3] # Assume w=1 from standard transformation
    depths = corners_cam[:, 2] # Z-coordinate is depth

    # 3. Filter points behind the camera
    valid_depth_mask = depths > EPS
    if not np.any(valid_depth_mask):
        return None, None # Box entirely behind camera

    valid_corners_cam = corners_cam[valid_depth_mask]
    valid_depths = depths[valid_depth_mask]

    # If no valid points in front of the camera, return None
    if valid_corners_cam.shape[0] == 0:
        return None, None

    # 4. Project *all* valid camera points (not just visible ones) to pixel coordinates to check overlap
    # This helps catch cases where vertices are off-screen but faces/edges are visible
    valid_corners_cam_hom_for_proj = np.hstack((valid_corners_cam, np.ones((valid_corners_cam.shape[0], 1))))
    corners_proj = valid_corners_cam_hom_for_proj @ intrinsic.T

    # Validate projection results
    if not np.all(np.isfinite(corners_proj)):
        # print("Warning: Non-finite values encountered during projection.")
        return None, None

    proj_depths = corners_proj[:, 2]
    # Filter points where projection depth is too small (avoids division by zero)
    valid_proj_mask = np.abs(proj_depths) >= EPS
    if not np.any(valid_proj_mask):
        return None, None # All points projected onto image plane or behind
    
    corners_proj = corners_proj[valid_proj_mask]
    proj_depths = proj_depths[valid_proj_mask]
    corners_pixel = corners_proj[:, :2] / proj_depths[:, np.newaxis]
    # We also need to filter the original depths to match the filtered projected points
    valid_depths = valid_depths[valid_proj_mask]

    corners_pixel_rounded = np.round(corners_pixel).astype(int)

    # 5. Check visibility: At least one vertex inside image bounds?
    in_image_mask = (corners_pixel_rounded[:, 0] >= 0) & (corners_pixel_rounded[:, 0] < width) & \
                    (corners_pixel_rounded[:, 1] >= 0) & (corners_pixel_rounded[:, 1] < height)
    any_vertex_visible = np.any(in_image_mask)

    # 6. Check visibility: Projected bounding box overlaps image?
    min_px, min_py = np.min(corners_pixel_rounded, axis=0)
    max_px, max_py = np.max(corners_pixel_rounded, axis=0)
    bbox_overlaps_image = not (max_px < 0 or min_px >= width or max_py < 0 or min_py >= height)

    # 7. Determine if any part is visible
    is_visible = any_vertex_visible or bbox_overlaps_image

    # 8. Return min/max depth if visible
    if is_visible and valid_depths.size > 0: # Ensure there are depths to calculate min/max from
        min_visible_depth = np.min(valid_depths)
        max_visible_depth = np.max(valid_depths)
        return min_visible_depth, max_visible_depth
    else:
        return None, None



def calculate_occupied_pixels(objects, extrinsic, intrinsic, img_shape):
    """Compute occupancy map for the given objects (Optimized).

    Also returns individual occupancy maps for each object.

    Args:
        objects (list of dict): List of object dictionaries, each must contain
                                'obb' (o3d.geometry.OrientedBoundingBox) and
                                'name' (str).
        extrinsic (np.ndarray): 4x4 extrinsic matrix (camera to world transformation).
        intrinsic (np.ndarray): 4x4 camera intrinsic matrix.
        img_shape (tuple): Shape of the image (width, height).

    Returns:
        tuple: (combined_occupancy_map, individual_occupancy_maps)
            - combined_occupancy_map (np.ndarray): 2D boolean array for all objects.
            - individual_occupancy_maps (dict): Dictionary where keys are object
                                                names and values are individual
                                                2D boolean occupancy maps.
    """
    EPS = 1e-6
    extrinsic_w2c = np.linalg.inv(extrinsic)
    w, h = img_shape # Correctly unpack width and height
    combined_occupancy_map = np.zeros((h, w), dtype=bool)
    individual_occupancy_maps = {} # Store individual maps here

    faces = [
        [0, 1, 2, 3], [4, 5, 6, 7],
        [0, 1, 5, 4], [3, 2, 6, 7],
        [0, 3, 7, 4], [1, 2, 6, 5]
    ]

    for obj_info in objects:
        box = obj_info.get('obb')
        obj_name = obj_info.get('name')

        if box is None or obj_name is None:
             print(f"Warning: Skipping object due to missing 'obb' or 'name'. Info: {obj_info}")
             continue

        # Initialize individual map for this object
        current_obj_map = np.zeros((h, w), dtype=bool)

        # Get the corners of the box
        corners = np.asarray(box.get_box_points())
        # Reorder corners to match the original code
        corners = corners[[0, 1, 7, 2, 3, 6, 4, 5]]
        # Add homogeneous coordinate
        corners_hom = np.concatenate([corners, np.ones((corners.shape[0], 1))], axis=1)
        # Project corners to image plane
        corners_img = (intrinsic @ extrinsic_w2c @ corners_hom.T).T

        # Check for invalid depths early
        if np.any(np.abs(corners_img[:, 2]) < EPS):
            pass # Face-by-face check will handle points behind camera

        # Normalize projected points using actual z-coordinate
        corners_pixel = np.zeros((corners_img.shape[0], 2))
        valid_proj_mask = np.abs(corners_img[:, 2]) >= EPS
        if np.any(valid_proj_mask):
             corners_pixel[valid_proj_mask] = corners_img[valid_proj_mask, :2] / corners_img[valid_proj_mask, 2][:, np.newaxis]

        for face in faces:
             # Check if all vertices of the face are behind the camera
            if np.any(corners_img[face, 2] < EPS):
                 continue # Skip faces that are entirely or partially behind the camera plane

            pts = corners_pixel[face]

            # Calculate the bounding box of the projected face
            min_coords = np.min(pts, axis=0)
            max_coords = np.max(pts, axis=0)

            # Determine the subgrid boundaries, clamping to image dimensions
            min_x = max(0, int(np.floor(min_coords[0])))
            min_y = max(0, int(np.floor(min_coords[1])))
            max_x = min(w - 1, int(np.ceil(max_coords[0])))
            max_y = min(h - 1, int(np.ceil(max_coords[1])))

            # If the bounding box is outside the image or has no area, skip
            if max_x < min_x or max_y < min_y:
                continue

            # Create coordinate grid only for the bounding box region
            sub_x, sub_y = np.meshgrid(np.arange(min_x, max_x + 1), np.arange(min_y, max_y + 1))
            pixel_points_sub = np.vstack((sub_x.flatten(), sub_y.flatten())).T

            if pixel_points_sub.size == 0:
                continue # Skip if subgrid is empty

            # Check containment using Path for the subgrid
            p = Path(pts)
            mask_sub = p.contains_points(pixel_points_sub).reshape((max_y - min_y + 1, max_x - min_x + 1))

            # Update the *individual* occupancy map
            current_obj_map[min_y:max_y+1, min_x:max_x+1] |= mask_sub

        # Store the individual map
        individual_occupancy_maps[obj_name] = current_obj_map
        # Combine into the main map
        combined_occupancy_map |= current_obj_map

    return combined_occupancy_map, individual_occupancy_maps



# Function to project the bounding box to the floor (2D)
def project_to_floor(box):
    corners = np.asarray(box.get_box_points())
    # corners[:, 2] = 0  # Set the z-coordinate to 0 to project onto the floor
    return corners[:, :2]  # Return only the x,y coordinates



# Adapted from: https://github.com/OpenRobotLab/EmbodiedScan/blob/main/embodiedscan/visualization/utils.py
# License: Apache 2.0
def _9dof_to_box(box, label=None, color_selector=None, color=None):
    """Convert 9-DoF box from array/tensor to open3d.OrientedBoundingBox.

    Args:
        box (numpy.ndarray|torch.Tensor|List[float]):
            9-DoF box with shape (9,).
        label (int, optional): Label of the box. Defaults to None.
        color_selector (:obj:`ColorSelector`, optional):
            Color selector for boxes. Defaults to None.
        color (tuple[int], optional): Color of the box.
            You can directly specify the color.
            If you do, the color_selector and label will be ignored.
            Defaults to None.
    """
    if isinstance(box, list):
        box = np.array(box)
    else:
        print("box is not a list!")
        print(type(box))
    # if isinstance(box, Tensor):  #NOTE omitted to not load in torch for just this!
    #     box = box.cpu().numpy()
    center = box[:3].reshape(3, 1)
    scale = box[3:6].reshape(3, 1)
    rot = box[6:].reshape(3, 1)
    rot_mat = o3d.geometry.OrientedBoundingBox.get_rotation_matrix_from_zxy(rot)
    geo = o3d.geometry.OrientedBoundingBox(center, rot_mat, scale)

    if color is not None:
        geo.color = [x / 255.0 for x in color]
        return geo

    if label is not None and color_selector is not None:
        color = color_selector.get_color(label)
        color = [x / 255.0 for x in color]
        geo.color = color
    return geo