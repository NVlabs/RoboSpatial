# Copyright (c) 2024-2025 NVIDIA CORPORATION. All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.


import numpy as np

from spatial_analysis.configuration.configuration_utils import get_object_metrics


def check_spatial_configuration_relationships(obj1, obj2, extrinsic, intrinsic, image_size, individual_occupancy_maps, strictness='lenient'):
    """Calculates the spatial relationship between two 3D objects (obj1 relative to obj2) from multiple perspectives.

    This function determines relationships like 'left', 'right', 'in front of', 'behind', 'above',
    'below', and 'overlapping' based on the objects' oriented bounding boxes (OBBs). It leverages
    pre-calculated object metrics (pixel projections, depth ranges, world coordinates) obtained
    via `get_object_metrics`.

    Relationships are assessed in three reference frames:
    1.  **camera_centric**: Based on the objects' 2D projections onto the image plane (for left/right)
        and their depth relative to the camera (for in front/behind). Vertical relationships (above/below)
        in this frame are determined using world Z coordinates.
    2.  **world_centric**: Uses the same logic as camera_centric for horizontal and depth relationships
        in this implementation, but explicitly defines overlap based on world Z-axis separation.
    3.  **object_centric**: Determines relationships based on the relative position of obj1's center
        with respect to obj2's orientation (forward and right vectors). It uses the Separating Axis 
        Theorem (SAT) to check for OBB overlap in 3D, influencing directional judgments. Vertical
        relationships (above/below) use the world Z coordinates.

    The `strictness` parameter controls the calculation logic for camera_centric and world_centric:
    -   **'strict'**: Requires clear separation based on the minimum and maximum bounds of the objects'
        metrics (pixel coordinates, depth values, world Z). Objects are considered overlapping if their
        bounds intersect, even slightly. This mode is sensitive to full visibility.
    -   **'lenient'**: Uses object centers (for pixel projection), average visible depth, and a combination 
        of average/min/max world Z coordinates. It's more robust to partial occlusions or near overlaps. It may still use strict bounds checks in ambiguous cases (e.g., very close average depths).

    Args:
        obj1 (dict): First object, containing at least 'name' (str) and 'obb' (open3d.geometry.OrientedBoundingBox).
        obj2 (dict): Second object, with the same structure as obj1.
        extrinsic (np.ndarray): 4x4 extrinsic matrix representing the camera-to-world transformation.
        intrinsic (np.ndarray): 3x3 or 4x4 camera intrinsic matrix. Only the top-left 3x3 portion is used.
        image_size (tuple): A tuple representing the image size as (width, height).
        individual_occupancy_maps (dict): A dictionary containing precomputed individual occupancy maps for objects,
                                          used by `get_object_metrics`. Keys should match object names.
        strictness (str, optional): The mode for relationship checks ('strict' or 'lenient'). Defaults to 'lenient'.

    Returns:
        dict: A dictionary containing boolean spatial relationships for each reference frame ('camera_centric',
              'world_centric', 'object_centric'). Each frame contains keys: 'left', 'right', 'infront',
              'behind', 'above', 'below', 'overlapping'.
    """
    EPSILON = 1e-6 # Tolerance for floating point comparisons


    box1_metrics = get_object_metrics(obj1, extrinsic, intrinsic, image_size, individual_occupancy_maps)
    box2_metrics = get_object_metrics(obj2, extrinsic, intrinsic, image_size, individual_occupancy_maps)

    # Clamp negative world Z values to 0 based on the assumption that nothing is below ground
    for metrics in [box1_metrics, box2_metrics]:
        if metrics.get('world_z_min') is not None and metrics['world_z_min'] < 0:
            metrics['world_z_min'] = 0.0
        if metrics.get('world_z_max') is not None and metrics['world_z_max'] < 0:
             # If max is negative, min must also be negative (or None), so both are clamped to 0
             metrics['world_z_max'] = 0.0
        if metrics.get('world_z_avg') is not None and metrics['world_z_avg'] < 0:
            metrics['world_z_avg'] = 0.0

    # Nested Function to Determine Relationships Based on Metrics
    def determine_camera_and_world_relationship(m1, m2):
        """Determines strict and lenient relationships based on pre-calculated metrics."""

        # Strict Relationship Calculation
        def get_relationship_strict():
            horizontal_relation = "overlapping"
            # Check visibility before comparing strict horizontal bounds
            if m1.get('max_px') is not None and m2.get('min_px') is not None and m1['max_px'] < m2['min_px'] - EPSILON:
                 horizontal_relation = "left"
            elif m1.get('min_px') is not None and m2.get('max_px') is not None and m1['min_px'] > m2['max_px'] + EPSILON:
                 horizontal_relation = "right"

            depth_relation = "overlapping"
            # Check visibility before comparing strict depth bounds
            if m1.get('max_depth') is not None and m2.get('min_depth') is not None and m1['max_depth'] < m2['min_depth'] - EPSILON:
                depth_relation = "in front of"
            elif m1.get('min_depth') is not None and m2.get('max_depth') is not None and m1['min_depth'] > m2['max_depth'] + EPSILON:
                depth_relation = "behind"

            vertical_world_relation = "overlapping"
            # World Z is always calculated
            if m1.get('world_z_max') is not None and m2.get('world_z_min') is not None and m1['world_z_max'] < m2['world_z_min'] - EPSILON:
                vertical_world_relation = "below"
            elif m1.get('world_z_min') is not None and m2.get('world_z_max') is not None and m1['world_z_min'] > m2['world_z_max'] + EPSILON:
                vertical_world_relation = "above"

            return {
                "left": horizontal_relation == "left",
                "right": horizontal_relation == "right",
                "infront": depth_relation == "in front of",
                "behind": depth_relation == "behind",
                "cam_overlapping": horizontal_relation == "overlapping", # Overlap based on pixel projection
                "above": vertical_world_relation == "above",
                "below": vertical_world_relation == "below",
                "world_overlapping": vertical_world_relation == "overlapping" # Overlap based on world Z
            }

        # Lenient Relationship Calculation
        def get_relationship_lenient():
            # Check if centers are comparable
            centers_comparable = m1.get('center_px') is not None and m2.get('center_px') is not None
            
            horizontal_relation = "overlapping"
            if centers_comparable:
                # Check containment using pixel bounds
                box1_center_in_box2_px = (m1.get('min_px') is not None and # Check all required bounds exist
                                           m1.get('max_px') is not None and
                                           m1.get('min_py') is not None and
                                           m1.get('max_py') is not None and
                                           m2.get('min_px') is not None and 
                                           m2.get('max_px') is not None and 
                                           m2.get('min_py') is not None and 
                                           m2.get('max_py') is not None and 
                                           m2['min_px'] <= m1['center_px'][0] <= m2['max_px'] and
                                           m2['min_py'] <= m1['center_px'][1] <= m2['max_py'])

                box2_center_in_box1_px = (m1.get('min_px') is not None and # Check all required bounds exist
                                           m1.get('max_px') is not None and
                                           m1.get('min_py') is not None and
                                           m1.get('max_py') is not None and
                                           m2.get('min_px') is not None and 
                                           m2.get('max_px') is not None and 
                                           m2.get('min_py') is not None and 
                                           m2.get('max_py') is not None and 
                                           m1['min_px'] <= m2['center_px'][0] <= m1['max_px'] and
                                           m1['min_py'] <= m2['center_px'][1] <= m1['max_py'])

                if m1['center_px'][0] < m2['center_px'][0] - EPSILON and not box2_center_in_box1_px:
                    horizontal_relation = "left"
                elif m1['center_px'][0] > m2['center_px'][0] + EPSILON and not box1_center_in_box2_px:
                    horizontal_relation = "right"

            # Lenient depth check based on average visible depth
            depth_relation = "overlapping"
            avg_depths_comparable = m1.get('visible_depth_avg') is not None and m2.get('visible_depth_avg') is not None
            if avg_depths_comparable:
                 # Optional: Add hybrid check using strict bounds if averages are close
                 if abs(m1['visible_depth_avg'] - m2['visible_depth_avg']) < EPSILON:
                      # Averages are close, fall back to strict check only if strictly separated
                      if m1.get('max_depth') is not None and m2.get('min_depth') is not None and m1['max_depth'] < m2['min_depth'] - EPSILON:
                           depth_relation = "in front of"
                      elif m1.get('min_depth') is not None and m2.get('max_depth') is not None and m1['min_depth'] > m2['max_depth'] + EPSILON:
                           depth_relation = "behind"
                      # else: stays overlapping
                 elif m1['visible_depth_avg'] < m2['visible_depth_avg']:
                     depth_relation = "in front of"
                 elif m1['visible_depth_avg'] > m2['visible_depth_avg']:
                     depth_relation = "behind"

            # Lenient vertical check based on world Z, prioritizing separation
            vertical_world_relation = "overlapping" # Default to overlapping

            # Check if metrics are available for comparison
            m1_z_max = m1.get('world_z_max')
            m1_z_min = m1.get('world_z_min')
            m1_z_avg = m1.get('world_z_avg')
            m2_z_max = m2.get('world_z_max')
            m2_z_min = m2.get('world_z_min')
            m2_z_avg = m2.get('world_z_avg')

            all_metrics_exist = all(v is not None for v in [m1_z_max, m1_z_min, m1_z_avg, m2_z_max, m2_z_min, m2_z_avg])

            if all_metrics_exist:
                 # 1. Check strict separation first
                 if m1_z_max < m2_z_min - EPSILON:
                      vertical_world_relation = "below"
                 elif m1_z_min > m2_z_max + EPSILON:
                      vertical_world_relation = "above"
                 # 2. If strictly overlapping, check lenient conditions (avg vs max/min)
                 else:
                      # Check if average of 1 is above max of 2
                      if m1_z_avg > m2_z_max + EPSILON:
                           vertical_world_relation = "above"
                      # Check if average of 2 is above max of 1 (meaning 1 is below 2)
                      elif m2_z_avg > m1_z_max + EPSILON:
                           vertical_world_relation = "below"
                      # Otherwise, they remain overlapping

            return {
                "left": horizontal_relation == "left",
                "right": horizontal_relation == "right",
                "infront": depth_relation == "in front of",
                "behind": depth_relation == "behind",
                "cam_overlapping": horizontal_relation == "overlapping", # Overlap based on pixel projection centroid logic
                "above": vertical_world_relation == "above",
                "below": vertical_world_relation == "below",
                "world_overlapping": vertical_world_relation == "overlapping" # Overlap based on world Z average logic
            }

        # Return both results
        return {
            "strict": get_relationship_strict(),
            "lenient": get_relationship_lenient()
        }

    # Calculate Camera/World Relationships
    # Object visibility/comparability is handled by None checks within determine_camera_and_world_relationship.
    cam_world_relations = determine_camera_and_world_relationship(box1_metrics, box2_metrics)

    # Calculate Object Centric Relationships
    def get_object_centric_relationship(obj1, obj2):
        def get_facing_direction(box):
            rotation_matrix = np.asarray(box.R)
            forward_direction = rotation_matrix[:, 0]
            return forward_direction

        def check_overlap(box1, box2):
            box1_points = np.asarray(box1.get_box_points())
            box2_points = np.asarray(box2.get_box_points())

            def project_points(points, axis):
                return np.dot(points, axis)

            def overlap_on_axis(box1_proj, box2_proj):
                box1_min, box1_max = np.min(box1_proj), np.max(box1_proj)
                box2_min, box2_max = np.min(box2_proj), np.max(box2_proj)
                return not (box1_max < box2_min or box2_max < box1_min)

            # Use OBB axes for Separating Axis Theorem (more robust than just world axes diffs)
            axes = []
            axes.extend(box1.R.T) # Box 1 axes
            axes.extend(box2.R.T) # Box 2 axes
            # Add cross products of axes (simplified common implementation)
            # Calculate the 9 potential separating axes derived from cross products
            # of each edge direction of box1 with each edge direction of box2.
            # Since OBB axes are parallel to edge directions, we cross the axes vectors.
            for i in range(3):
                for j in range(3):
                    # Cross product of box1 axis i and box2 axis j
                    cross_product = np.cross(box1.R[:, i], box2.R[:, j])
                    if np.linalg.norm(cross_product) > EPSILON: # Avoid zero vectors
                         axes.append(cross_product / np.linalg.norm(cross_product))

            for axis in axes:
                if not overlap_on_axis(project_points(box1_points, axis),
                                       project_points(box2_points, axis)):
                    # Separating axis found, no overlap
                    return False

            # If no separating axis is found by SAT, the OBBs are considered overlapping.
            return True

        # Simplified overlap check based on SAT result for object-centric logic
        overlap_obj_centric = check_overlap(obj1["obb"], obj2["obb"])

        obj2_forward = get_facing_direction(obj2["obb"])
        obj1_center = np.mean(np.asarray(obj1["obb"].get_box_points()), axis=0)
        obj2_center = np.asarray(obj2["obb"].get_center())

        relative_position = obj1_center - obj2_center
        dot_product = np.dot(relative_position, obj2_forward)
        # Use Up vector (assuming Z is up) for cross product for left/right relative to forward
        world_up = np.array([0, 0, 1]) 
        obj2_right = np.cross(obj2_forward, world_up) 
        # Ensure obj2_right is normalized if needed, though only sign matters for dot product
        if np.linalg.norm(obj2_right) > EPSILON:
             obj2_right /= np.linalg.norm(obj2_right)
        else:
             # Handle cases where forward is aligned with up (e.g. object pointing straight up/down)
             # Use world X or Y as a fallback 'right' ? This needs careful thought.
             # For now, if right vector is invalid, horizontal relation is ambiguous/overlapping
             obj2_right = None

        horizontal_dot = np.dot(relative_position, obj2_right) if obj2_right is not None else 0


        # Object-centric depth uses dot product with forward vector
        depth_relation = "overlapping"
        if not overlap_obj_centric: # Only assign directional if not overlapping
             if dot_product > EPSILON:
                 depth_relation = "in front of"
             elif dot_product < -EPSILON:
                 depth_relation = "behind"
             # else: stays overlapping (or on the plane)

        # Object-centric horizontal uses dot product with right vector
        horizontal_relation = "overlapping"
        if not overlap_obj_centric and obj2_right is not None: # Only assign if not overlapping and right vector is valid
             if horizontal_dot > EPSILON: # Project onto right vector: positive is "right"
                 horizontal_relation = "right"
             elif horizontal_dot < -EPSILON: # Negative is "left"
                 horizontal_relation = "left"
             # else: stays overlapping (or directly in front/behind)

        return horizontal_relation, depth_relation

    obj_centric_horizontal, obj_centric_depth = get_object_centric_relationship(obj1, obj2)

    # Select strict or lenient results based on parameter
    chosen_relation = cam_world_relations.get(strictness, cam_world_relations['lenient']) # Default to lenient

    # Assemble Final Result
    relationships = {
        "camera_centric": {
            "left": chosen_relation["left"],
            "right": chosen_relation["right"],
            "infront": chosen_relation["infront"],
            "behind": chosen_relation["behind"],
            # Use world vertical for camera frame above/below
            "above": chosen_relation["above"],
            "below": chosen_relation["below"],
            "overlapping": chosen_relation["cam_overlapping"],
        },
        "world_centric": {
            # World uses same planar relationships as camera in this implementation
            "left": chosen_relation["left"],
            "right": chosen_relation["right"],
            "infront": chosen_relation["infront"],
            "behind": chosen_relation["behind"],
            "above": chosen_relation["above"],
            "below": chosen_relation["below"],
            "overlapping": chosen_relation["world_overlapping"] # Use Z-based overlap here
        },
        "object_centric": {
            "left": obj_centric_horizontal == "left",
            "right": obj_centric_horizontal == "right",
            "infront": obj_centric_depth == "in front of",
            "behind": obj_centric_depth == "behind",
            # Use world vertical for object frame above/below
            "above": chosen_relation["above"],
            "below": chosen_relation["below"],
            # Object centric overlap combines horizontal and depth states
            "overlapping": obj_centric_horizontal == "overlapping" or obj_centric_depth == "overlapping"
        }
    }

    return relationships