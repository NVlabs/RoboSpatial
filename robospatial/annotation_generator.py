# Copyright (c) 2024-2025 NVIDIA CORPORATION. All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Core annotation generation logic for a single scene.

This module defines the `generate_and_save_annotations` function, which is responsible
for processing all images within a given scene. It calculates various spatial
relationships and grounding information based on object data (OBBs, categories)
and camera parameters.

It utilizes functions from the `spatial_analysis` package to compute:
- Object Grounding: Bounding boxes in 2D.
- Spatial Context: Points relative to an object (e.g., in front, behind).
- Spatial Compatibility: Fit assessment (e.g., can A fit on B).
- Spatial Configuration: Relative positioning (e.g., left/right, above/below).

The generated annotations are saved as JSON files, one per processed image.
This module is typically called by a higher-level script (e.g. `run_generation.py`)
that handles dataset iteration and overall workflow management.
"""


import itertools
import os
import json
import cv2
from collections import defaultdict
import numpy as np
from tqdm import tqdm

from spatial_analysis.grounding import get_object_grounding
from spatial_analysis.relationships import get_spatial_configuration, get_spatial_compatibility, get_spatial_context
from spatial_analysis.relationship_utils import calculate_occupied_pixels



# --- Main Annotation Generation Function ---

def generate_and_save_annotations(loader, dataset_name, scene_name, images_ann_dict, config, num_workers):
    """
    Generates and saves annotations for a scene based on the configuration.
    Handles multiple annotation types: localization, compatibility, point_grounding, bbox_grounding.
    """

    # --- Statistics Initialization ---
    stats = defaultdict(int)
    stats['num_total_images'] = 0

    # --- Read Compatibility Check Configs --- 
    comp_grid_res = config["data_generation"]["generation_options"]["compatibility_grid_resolution"]
    comp_min_distance = config["data_generation"]["generation_options"]["compatibility_min_distance"]
    comp_buffer_ratio = config["data_generation"]["generation_options"]["compatibility_buffer_ratio"]
    comp_num_samples = config["data_generation"]["generation_options"]["compatibility_num_samples"]

    # --- Read Spatial Context Configs --- 
    context_threshold = config["data_generation"]["generation_options"]["context_threshold"] 
    context_grid_res = config["data_generation"]["generation_options"]["context_grid_resolution"] 
    context_num_samples = config["data_generation"]["generation_options"]["context_num_samples"] 

    # --- Read Spatial Configuration Strictness ---
    spatial_config_strictness = config["data_generation"]["generation_options"]["spatial_configuration_strictness"]

    # --- Read Pairwise Relationship Mode ---
    pairwise_mode = config["data_generation"]["generation_options"]["pairwise_relationship_mode"]


    # --- Generate Annotations ---
    # Determine the iterator based on whether tqdm should be used
    image_iterator = images_ann_dict.items()
    if num_workers <= 1:
        # Wrap with tqdm only if single-threaded
        image_iterator = tqdm(image_iterator, desc=f"Processing images in {scene_name}", leave=False)

    for image_name, image_ann in image_iterator:

        # Initial setup
        relationships_to_generate = config["data_generation"]["generation_options"]["spatial_relationship_types"]
        extrinsic = image_ann['extrinsic']
        intrinsic = image_ann['intrinsic']

        # --- Image Setup ---
        image_path = os.path.join(config["data_loading"]["image_root"], image_ann["img_path"])  # Use path as identifier
        image_file = cv2.imread(image_path)
        
        if image_file is None:
            print(f"Warning: Could not read image {image_path}. Skipping.")
            continue
        h, w, _ = image_file.shape
        image_size = (w, h)

        # Process visible objects in the image
        vis_objs, unique_vis_categories, multi_vis_categories, floor_bound, all_objs = loader.list_objects(dataset_name, scene_name, image_ann)

        if len(all_objs) == 0:
            print(f"Warning: No objects detected in image {image_name}. Skipping image.")
            continue

        # Get all OBBs
        obbs = {obj["name"]: obj["obb"] for obj in all_objs.values() if "name" in obj and "obb" in obj}

        # --- Precompute Environment Occupancy Maps (Combined and Individual) ---
        # Pass list of object dictionaries to calculate_occupied_pixels
        objects_for_occupancy = [obj for obj in all_objs.values() if 'obb' in obj and 'name' in obj]
        if not objects_for_occupancy:
            print(f"Warning: No objects with OBB and name found for occupancy calculation in {image_name}. Skipping image.")
            continue # Or handle appropriately

        env_occupancy_map, individual_occupancy_maps = calculate_occupied_pixels(
            objects_for_occupancy, extrinsic, intrinsic, image_size
        )

        # --- Annotation Generation ---
        spatial_relationships = {
            "unary_relations": [],
            "pairwise_relations": []
        }
        # Initialize defaultdict to create a dictionary with empty lists for grounding keys
        object_grounding = []
        generated_something_for_image = False

        # 1. Generate Grounding Annotations (per object type)
        if 'spatial_context' in relationships_to_generate or 'object_grounding' in relationships_to_generate:

            for obj_name, obj in vis_objs.items():

                category = obj["category"]
                category = obj.get("category")
                obj_map = individual_occupancy_maps.get(obj_name) # Get precomputed map

                if category is None or obj_map is None:
                    print(f"Warning: Skipping object {obj_name} due to missing category or precomputed occupancy map.")
                    continue

                # No need to differentiate unique/multi here as we iterate through vis_objs directly

                #NOTE object grounding handles both single and multi instance objects
                if 'object_grounding' in relationships_to_generate:
                    grounding_info = get_object_grounding(obj, obj_map)

                    if grounding_info:
                        results = {
                            "name": obj_name,
                            "category": category,
                            "bbox": grounding_info["clipped_bbox"],
                            "bbox_3d": obj["bbox_3d"],
                        }
                        object_grounding.append(results)
                        generated_something_for_image = True
                        stats['num_object_grounding_generated'] += 1

                #NOTE spatial context handles single instance objects
                if 'spatial_context' in relationships_to_generate and category in unique_vis_categories:
                    # Filter obbs to exclude the current object
                    context_obbs = [obb for name, obb in obbs.items() if name != obj_name]
                    # Pass precomputed maps to get_spatial_context
                    points_2d, points_3d, generated = get_spatial_context(
                        obj, extrinsic, intrinsic, floor_bound, context_obbs, image_size, image_path,
                        individual_occupancy_maps=individual_occupancy_maps, # Pass individual maps
                        env_occupancy_map=env_occupancy_map,                 # Pass combined env map
                        threshold=context_threshold, # Pass configured threshold
                        grid_resolution=context_grid_res, # Pass configured grid resolution
                        num_samples=context_num_samples, # Pass configured num samples
                    )
                    if generated:
                        results = {
                            "name": obj_name,
                            "category": category,
                            "point_space_2d": points_2d,
                            "point_space_3d": points_3d,
                        }
                        spatial_relationships["unary_relations"].append(results)
                        generated_something_for_image = True
                        stats['num_spatial_context_generated'] += 1
        
        # 2. Generate Relationship Annotations (per object pair)
        generate_pairwise = 'spatial_configuration' in relationships_to_generate or 'spatial_compatibility' in relationships_to_generate
        objects_available_for_pairwise = (pairwise_mode == 'unique_categories_only' and len(unique_vis_categories) >= 2) or \
                                         (pairwise_mode == 'all_visible_objects' and len(vis_objs) >= 2)


        if generate_pairwise and objects_available_for_pairwise:
            # Determine the iterator based on the mode
            if pairwise_mode == 'unique_categories_only':
                iterator = itertools.permutations(unique_vis_categories, 2)
                get_obj = lambda cat: vis_objs[cat] # Function to get object by category
                # Use unique_vis_categories which are typically names/keys in vis_objs for unique items
                iterator = itertools.permutations(unique_vis_categories, 2)
                # Need to handle potential KeyError if a category name isn't directly a key in vis_objs
                # Assuming unique_vis_categories contains keys that *are* in vis_objs
                get_obj = lambda cat_key: vis_objs.get(cat_key)
            else: # pairwise_mode == 'all_visible_objects'
                iterator = itertools.permutations(vis_objs.keys(), 2)
                get_obj = lambda key: vis_objs[key] # Function to get object by key
                get_obj = lambda key: vis_objs.get(key)


            for item1_key, item2_key in iterator:
                obj1 = get_obj(item1_key)
                obj2 = get_obj(item2_key)

                # Skip if objects couldn't be retrieved (e.g., bad key from unique_vis_categories)
                if obj1 is None or obj2 is None:
                     print(f"Warning: Could not retrieve objects for pair ({item1_key}, {item2_key}). Skipping.")
                     continue

                # Get object names
                obj1_name = obj1["name"]
                obj2_name = obj2["name"]
                # Get object names and categories safely
                obj1_name = obj1.get("name")
                obj2_name = obj2.get("name")
                obj1_cat = obj1.get("category")
                obj2_cat = obj2.get("category")

                if not all([obj1_name, obj2_name, obj1_cat, obj2_cat]):
                     print(f"Warning: Missing name or category for objects in pair ({item1_key}, {item2_key}). Skipping.")
                     continue

                pair_result = {
                    "pair": (obj1_name, obj2_name),
                    "pair_category": (obj1["category"], obj2["category"]),
                    "pair_category": (obj1_cat, obj2_cat),
                }


                if 'spatial_configuration' in relationships_to_generate:
                    # Pass individual maps to get_spatial_configuration
                    config_rels = get_spatial_configuration(
                        obj1, obj2, extrinsic, intrinsic, image_size, individual_occupancy_maps,spatial_config_strictness)
                    pair_result["spatial_configuration"] = config_rels
                    generated_something_for_image = True
                    stats['num_spatial_configuration_pairs'] += 1

                if 'spatial_compatibility' in relationships_to_generate:
                    # Filter obbs to exclude obj1
                    compatibility_obbs = [obb for name, obb in obbs.items() if name != obj1_name]
                    # Pass individual and combined maps to get_spatial_compatibility
                    comp_rels = get_spatial_compatibility(
                        obj1, obj2, extrinsic, intrinsic, floor_bound, compatibility_obbs, image_size, image_path,
                        individual_occupancy_maps=individual_occupancy_maps, # Pass individual maps
                        env_occupancy_map=env_occupancy_map,                 # Pass combined env map
                        grid_resolution=comp_grid_res, # Pass configured grid resolution
                        num_samples=comp_num_samples, # Pass configured num samples
                        min_distance=comp_min_distance, # Pass configured min distance
                        buffer_ratio=comp_buffer_ratio # Pass configured buffer ratio
                    )
                    pair_result["spatial_compatibility"] = comp_rels
                    generated_something_for_image = True
                    stats['num_spatial_compatibility_pairs'] += 1

                if len(pair_result) > 2: # Check if more than just pair info was added
                    spatial_relationships["pairwise_relations"].append(pair_result)

        # --- Save Results ---
        if generated_something_for_image:
            stats['num_total_images'] += 1

            image_results = {
                "dataset": dataset_name,
                "scene_name": scene_name,
                "image_identifier": image_name,
                "image_path": image_path,
                "image_size": image_size,
                "depth_path": image_ann.get("depth_path", ""),
                "visible_instance_ids": image_ann.get('visible_instance_ids', []),
            }
            
            cam_ann = {}
            for key in ['extrinsic', 'intrinsic']:
                if key in image_ann and image_ann[key] is not None:
                     cam_ann[key] = image_ann[key].tolist()
            image_results["camera_annotations"] = cam_ann

            if object_grounding:
                image_results["object_grounding"] = object_grounding
            if spatial_relationships:
                image_results["spatial_relationships"] = spatial_relationships

            folder_path = os.path.join(config["data_generation"]["output_dir"], scene_name)
            os.makedirs(folder_path, exist_ok=True)

            output_suffix = config.get("output_suffix", ".annotations.json")
            file_name = f"{image_ann['image_basename']}{output_suffix}"
            file_path = os.path.join(folder_path, file_name)

            with open(file_path, 'w') as json_file:
                json.dump(image_results, json_file, indent=4)
        
    # --- Return Scene Statistics ---
    scene_ann_stats = {
        'dataset_name': dataset_name,
        'scene_name': scene_name,
        'num_processed_images': stats['num_total_images'],
        'num_spatial_configuration_pairs': stats['num_spatial_configuration_pairs'],
        'num_spatial_compatibility_pairs': stats['num_spatial_compatibility_pairs'],
        'num_object_grounding_generated': stats['num_object_grounding_generated'],
        'num_spatial_context_generated': stats['num_spatial_context_generated'],
    }

    return scene_ann_stats 