# Copyright (c) 2024-2025 NVIDIA CORPORATION. All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.



import numpy as np
from matplotlib.path import Path






def compute_distances_to_bbox(points, bbox_corners):
    """
    Compute the shortest distance from multiple points to a bounding box (convex polygon).

    Parameters:
    - points: A NumPy array of shape (N, 2) representing N points.
    - bbox_corners: A NumPy array of shape (M, 2) representing the corners of the bounding box in order.

    Returns:
    - distances: A NumPy array of shape (N,) containing the shortest distances from each point to the bounding box.
    """
    # Create a Path object for the bounding box
    path = Path(bbox_corners)

    # Determine which points are inside the bounding box
    inside = path.contains_points(points)

    # Initialize distances array
    distances = np.zeros(points.shape[0])

    # Points outside the polygon
    outside_points = points[~inside]

    if outside_points.size > 0:
        num_edges = bbox_corners.shape[0]
        distances_outside = np.full(outside_points.shape[0], np.inf)

        # Compute distances from points to each edge
        for i in range(num_edges):
            A = bbox_corners[i]
            B = bbox_corners[(i + 1) % num_edges]
            AB = B - A
            AB_squared = np.dot(AB, AB)

            if AB_squared == 0:
                # A and B are the same point
                distances_edge = np.linalg.norm(outside_points - A, axis=1)
            else:
                AP = outside_points - A
                t = np.dot(AP, AB) / AB_squared
                t = np.clip(t, 0, 1)
                closest = A + t[:, np.newaxis] * AB
                distances_edge = np.linalg.norm(outside_points - closest, axis=1)

            distances_outside = np.minimum(distances_outside, distances_edge)

        # Assign distances to the corresponding points
        distances[~inside] = distances_outside

    # Points inside have zero distance
    distances[inside] = 0.0

    return distances


def project_points_to_image(points, extrinsic, intrinsic):
    extrinsic_w2c = np.linalg.inv(extrinsic)
    points = np.concatenate([points, np.ones((points.shape[0], 1))], axis=1)
    points_img = intrinsic @ extrinsic_w2c @ points.transpose()
    points_img = points_img.transpose()
    
    # Normalize homogeneous coordinates
    w = points_img[:, 3]
    points_img = points_img[:, :3] / w[:, np.newaxis]
    
    # Initialize output arrays
    points_pixel = points_img[:, :2] / points_img[:, 2][:, np.newaxis]
    points_depth = points_img[:, 2]
    
    return np.round(points_pixel).astype(int).tolist(), points_depth.tolist()