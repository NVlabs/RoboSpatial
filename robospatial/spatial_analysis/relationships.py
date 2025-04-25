# Copyright (c) 2024-2025 NVIDIA CORPORATION. All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""High-level interface for spatial relationship analysis between 3D objects.

This module provides wrapper functions that simplify the calculation of various
spatial relationships by utilizing lower-level functions from the `context`,
`configuration`, and `compatibility` submodules.

Key functionalities include:
- `get_spatial_context`: Determines points in space relative to a reference object
  (e.g., 'in front of', 'behind').
- `get_spatial_configuration`: Calculates 3D directional relationships between two
  objects (e.g., 'left of', 'above').
- `get_spatial_compatibility`: Assesses whether one object can physically fit
  relative to another (e.g., 'on top of', 'next to').

These functions are typically used by higher-level annotation generation scripts.
"""

from spatial_analysis.obj_properties import items_with_face, movable_and_placeable_items, flat_surface_items
from spatial_analysis.context.context import get_point_in_space_relative_to_object
from spatial_analysis.configuration.configuration import check_spatial_configuration_relationships
from spatial_analysis.compatibility.compatibility import can_fit_object_a_in_relation_to_b, can_fit_on_top


def get_spatial_context(obj, extrinsic, intrinsic, floor_bound, obbs, image_size, image_path,
                        individual_occupancy_maps, env_occupancy_map,
                        threshold, grid_resolution, num_samples):
    """Generates points relative to an object (e.g., in front, behind) for contextual understanding."""

    have_face = obj["name"] in items_with_face

    # Generate potential points relative to the object's spatial context.
    sampled_points, sampled_3d_points, visible_points_3d_all, generated_something = get_point_in_space_relative_to_object(
        floor_bound, obbs,
        ref_obj=obj,
        extrinsic=extrinsic, intrinsic=intrinsic, image_size=image_size, have_face=have_face,
        num_samples=num_samples, threshold=threshold, grid_resolution=grid_resolution,
        individual_occupancy_maps=individual_occupancy_maps,
        env_occupancy_map=env_occupancy_map,
        image_path=image_path,
    )

    if generated_something:
        return sampled_points, sampled_3d_points, True
    return None, None, False


def get_spatial_configuration(obj1, obj2, extrinsic, intrinsic, image_size, individual_occupancy_maps, strictness='lenient'):
    """Calculates spatial configuration relationships (left/right, above/below, etc.) between two objects."""

    obj_configuration_relationships = check_spatial_configuration_relationships(
        obj1, obj2, extrinsic, intrinsic, image_size, individual_occupancy_maps, strictness
    )

    return obj_configuration_relationships


def get_spatial_compatibility(obj1, obj2, extrinsic, intrinsic, floor_bound, obbs, image_size, image_path,
                              individual_occupancy_maps, env_occupancy_map,
                              grid_resolution, num_samples, min_distance, buffer_ratio):
    """Checks if obj1 can fit in relation to obj2 (e.g., on top, next to)."""

    # Check if the anchor object (obj2) has a face, as this influences spatial context calculations.
    have_face = obj2["name"] in items_with_face

    # Check fitting in various spatial relations using sampling-based methods.
    results = can_fit_object_a_in_relation_to_b(
        floor_bound, obbs,
        obj_a=obj1,
        obj_b=obj2,
        have_face=have_face,
        extrinsic=extrinsic, intrinsic=intrinsic, image_size=image_size, image_path=image_path,
        grid_resolution=grid_resolution,
        min_distance=min_distance,
        num_samples=num_samples,
        individual_occupancy_maps=individual_occupancy_maps,
        env_occupancy_map=env_occupancy_map,
        buffer_ratio=buffer_ratio
    )

    # Specifically check 'on_top' relationship using direct OBB comparison
    # for movable items on flat surfaces, as this is a common and simpler case.
    fits_on_top = False
    if obj1["name"] in movable_and_placeable_items and obj2["name"] in flat_surface_items:
        fits_on_top = can_fit_on_top(obj1["obb"], obj2["obb"])

    if "worldcentric" not in results:
        results["worldcentric"] = {}
    results["worldcentric"]["on_top"] = fits_on_top
    
    return results