# Copyright (c) 2024-2025 NVIDIA CORPORATION. All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.


import os
import json
from collections import defaultdict
import glob

import numpy as np
from data_loader.base_loader import BaseLoader

from spatial_analysis.relationship_utils import _9dof_to_box

# Top-level function for defaultdict factory (picklable)
def nested_dict_factory():
    return defaultdict(dict)

class ExampleLoader(BaseLoader):
    """
    Loader for example data from JSON annotation files.

    Inherits from BaseLoader and implements its interface methods.
    """

    def __init__(self, config):
        """
        Initialize the Example loader.

        Args:
            config (dict): Data loader configuration dictionary.
                           Expected keys:
                           - annotation_dir (str): Path to the directory containing JSON annotation files.
                           - verbose (bool): Optional verbosity flag.
        """
        self.verbose = config.get("verbose", False)
        self.config = config

        if self.verbose:
            print('Loading Example Dataset JSON annotations...')
        annotation_dir = config.get("annotation_dir")
        if not annotation_dir or not os.path.isdir(annotation_dir):
            raise ValueError("Config must contain a valid 'annotation_dir' pointing to the JSON annotations.")

        self.data = defaultdict(nested_dict_factory) # Use the named function
        # Recursively find all json files in the annotation_dir
        json_files = glob.glob(os.path.join(annotation_dir, '**', '*.json'), recursive=True)


        if not json_files:
             print(f"Warning: No JSON files found in {annotation_dir}")
             return

        for file_path in json_files:
            with open(file_path, 'r') as f:
                image_data = json.load(f)

            # Validate basic structure (can be expanded)
            required_keys = ['dataset', 'scene_name', 'image_name', 'objects', 'camera_annotations']
            if not all(k in image_data for k in required_keys):
                if self.verbose:
                    print(f"Warning: Skipping file {file_path} due to missing one or more required keys: {required_keys}.")
                continue
            if not all(k in image_data['camera_annotations'] for k in ['extrinsic', 'intrinsic']):
                if self.verbose:
                    print(f"Warning: Skipping file {file_path} due to missing extrinsic/intrinsic in camera_annotations.")
                    continue

            dataset_name = image_data['dataset']
            # Use the scene name provided in the JSON
            scene_name = image_data['scene_name']
            image_name = image_data['image_name']

            # Store the loaded data, grouping by dataset and scene
            # self.data[dataset_name][scene_name] will hold a dict of {image_identifier: image_data}
            self.data[dataset_name][scene_name][image_name] = image_data
            # except Exception as e:
            #     if self.verbose:
            #         print(f"Warning: Skipping file {file_path} due to error: {e}")

        if self.verbose:
            total_scenes = 0
            total_images = 0
            for dataset_name, scenes in self.data.items():
                print(f"Loaded {len(scenes)} scenes from {dataset_name}")
                total_scenes += len(scenes)
                for scene_name, images in scenes.items():
                    total_images += len(images)
            print(f'Loading complete. Total scenes: {total_scenes}, Total images: {total_images}')


    def list_scenes(self, dataset_list):
        """
        Implementation of BaseLoader.list_scenes for Example Dataset dataset loaded from JSON.

        Args:
            dataset_list (list): List of dataset names to query.

        Returns:
            generator: Yields tuples of (dataset_name, scene_idx, scene_name).
        """

        for dataset_name in dataset_list:
            if dataset_name in self.data:
                # Ensure consistent scene indexing if needed, otherwise enumerate keys
                # Using enumerate(self.data[dataset_name]) provides a simple index
                # The scene_name now directly comes from the JSON data structure keys
                for scene_idx, scene_name in enumerate(self.data[dataset_name]):
                    yield dataset_name, scene_idx, scene_name

    def list_images(self, dataset_name, scene_name):
        """
        Implementation of BaseLoader.list_images for Example Dataset loaded from JSON.

        Args:
            dataset_name (str): Name of the dataset.
            scene_name (str): Name of the scene. Example: gr00t/deduped_data_normal

        Returns:
            dict: Dictionary of image annotations, keyed by image_identifier.
        """
        if dataset_name not in self.data or scene_name not in self.data[dataset_name]:
            if self.verbose:
                print(f"Warning: Scene {scene_name} not found in dataset {dataset_name}")
            return {}

        scene_images_data = self.data[dataset_name][scene_name]
        image_annotations = {}

        for image_identifier, image_data in scene_images_data.items():
            image_ann = {}
            # Basic info from the loaded JSON data
            image_ann['dataset'] = dataset_name
            image_ann['scene'] = scene_name
            image_ann['image_identifier'] = image_identifier
            image_ann['img_path'] = dataset_name + "/" + scene_name + "/" + image_identifier #NOTE image_path is combined with image_root from config to create the absolute image path
            image_ann['image_basename'] = os.path.basename(image_data['image_name'])
            image_ann['image_size'] = image_data.get('image_size') # Optional

            # Camera parameters from the loaded JSON data
            cam_ann = image_data.get('camera_annotations', {}) # Presence checked in __init__
            image_ann['extrinsic'] = np.array(cam_ann.get('extrinsic'))
            image_ann['intrinsic'] = np.array(cam_ann.get('intrinsic'))
            image_ann['objects'] = image_data.get('objects')

            # visible_instance_ids is no longer the primary way to get objects for list_objects
            # image_ann['visible_instance_ids'] = image_data.get('visible_instance_ids', [])

            # Use image_identifier as the key
            image_annotations[image_identifier] = image_ann



        return image_annotations

    def list_objects(self, dataset_name, scene_name, image_ann):
        """
        Implementation of BaseLoader.list_objects for Example Dataset from JSON.

        Processes objects listed in the 'object_grounding' field of the image annotation.

        Args:
            dataset_name (str): Name of the dataset (provides context).
            scene_name (str): Name of the scene (provides context).
            image_ann (dict): Image annotation dictionary (from list_images).

        Returns:
            tuple: A 5-element tuple containing:
                - vis_objs (dict): Dictionary of visible, non-environmental objects. Keys are categories (indexed if duplicates exist, e.g., 'chair_0', 'chair_1'). Values are instance dictionaries with 'obb' and 'name'.
                - unique_vis_categories (set): Set of original category names for objects appearing only once (excluding environmental/object categories).
                - multi_vis_categories (set): Set of original category names for objects appearing multiple times (excluding environmental/object categories).
                - floor_bound (list): Min and max floor boundaries [min_bound, max_bound] calculated from *this image's* non-environmental objects and floor (if present). Can be None.
                - all_objs (dict): Dictionary of all non-environmental objects in this image, keyed by their potentially indexed name (same as vis_objs in this implementation).
        """
        image_identifier = image_ann.get('image_identifier', 'unknown_image') # For logging
        objects = image_ann.get('objects', [])

        if not objects:
            # Return empty structures matching the expected return type
            return {}, set(), set(), None, {}

        # First pass to count occurrences of each non-environmental category in this image
        category_total_counts = defaultdict(int)
        parsed_objects = [] # Store parsed objects temporarily
        for obj_data in objects:
            category = obj_data.get("Name") #NOTE Name is the category name for gr00t
            bbox_3d_list = obj_data.get("bbox_3d")

            # Basic validation
            if not category:
                if self.verbose: print(f"Warning: Skipping object with missing Name in image {image_identifier}")
                continue
            if not bbox_3d_list or not isinstance(bbox_3d_list, list) or not bbox_3d_list:
                 if self.verbose: print(f"Warning: Skipping object '{category}' with missing or invalid bbox_3d in image {image_identifier}")
                 continue


            # Assuming bbox_3d is 9 DOF
            bbox_3d_params = bbox_3d_list[0]
            # Validate that we indeed have 9 parameters
            if not isinstance(bbox_3d_params, list) or len(bbox_3d_params) != 9:
                 if self.verbose: print(f"Warning: Skipping object '{category}' due to invalid bbox_3d params (expected 9DOF, got {len(bbox_3d_params)}) in image {image_identifier}. Params: {bbox_3d_params}")
                 continue

            # Removed padding logic as bbox_3d is guaranteed 9DOF.

            instance = {
                "name": category, #NOTE name == category for gr00t
                "category": category, # Original name from JSON
                "bbox_3d": bbox_3d_params, # Store the 9DoF params
            }

            try:
                 # Calculate OBB immediately
                 instance["obb"] = _9dof_to_box(bbox_3d_params)
                 parsed_objects.append(instance) # Add to list for further processing
                 # Count non-environmental categories (case-insensitive check)
                 if category.lower() not in ["wall", "ceiling", "floor", "object"]:
                     category_total_counts[category] += 1 # Count using original name
            except ValueError as e:
                 # Catch potential errors from _9dof_to_box if params are still invalid
                 if self.verbose:
                    print(f"Error converting bbox for object '{category}' in image {image_identifier}: {e}. Params: {bbox_3d_params}")
            except Exception as e: # Catch other potential exceptions
                 if self.verbose:
                    print(f"Unexpected error processing object '{category}' in image {image_identifier}: {e}")


        # Process parsed instances to create the final dictionaries
        vis_objs = {}
        unique_vis_categories = set()
        multi_vis_categories = set()
        category_indices = defaultdict(int) # To track current index for duplicate categories
        env_objs = {}
        all_objs_for_bounds = [] # Collect OBBs for floor calculation

        for instance in parsed_objects:
            category = instance["category"] # Original name

            # Handle environmental objects (for floor calculation)
            # Use lower case for comparison to identify type
            cat_lower = category.lower()
            if cat_lower in ["floor", "wall", "ceiling"]:
                # Assuming only one of each environmental object per image annotation
                # Store with lowercase key for easy lookup
                env_objs[cat_lower] = instance
                if cat_lower == "floor":
                    # Add floor OBB for bound calculation if it exists
                    if "obb" in instance:
                        all_objs_for_bounds.append(instance["obb"])
                continue # Skip adding env objects to vis_objs/multi/unique sets

            # Process non-environmental objects (already counted)
            if category in category_total_counts:
                total_count = category_total_counts[category]
                if total_count == 1:
                    obj_key = category # Use original name as key
                    instance["name"] = obj_key # Store potentially indexed name
                    unique_vis_categories.add(category) # Store original category name
                else:
                    # Use original category name for indexing
                    current_index = category_indices[category]
                    obj_key = f"{category}_{current_index}"
                    instance["name"] = obj_key # Store potentially indexed name
                    multi_vis_categories.add(category) # Store original category name
                    category_indices[category] += 1

                vis_objs[obj_key] = instance
                 # Add OBB for floor calculation if it exists
                if "obb" in instance:
                    all_objs_for_bounds.append(instance["obb"])


        # Calculate floor bounds based on OBBs from this image (non-env + floor if present)
        floor_bound = None
        if all_objs_for_bounds:
            try:
                # Combine points from all relevant OBBs
                all_points = np.vstack([box.get_box_points() for box in all_objs_for_bounds])
                min_bound = np.min(all_points, axis=0)
                max_bound = np.max(all_points, axis=0)

                # The bounds derived this way represent the extent of the objects considered.
                floor_bound = [min_bound.tolist(), max_bound.tolist()]
            except Exception as e:
                if self.verbose:
                    print(f"Error calculating floor bounds for image {image_identifier}: {e}")
                floor_bound = None # Indicate failure


        # `all_objs` in the original return signature was intended for occupancy map calculation,
        # usually containing all non-environmental objects in the scene.
        # In this JSON-based, image-level loading, it effectively becomes the same as `vis_objs`
        # as we only process objects visible/annotated in the current image JSON.
        all_objs = vis_objs.copy()

        return vis_objs, unique_vis_categories, multi_vis_categories, floor_bound, all_objs
