# Copyright (c) 2024-2025 NVIDIA CORPORATION. All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.
#
# This code is partially adapted from
# https://github.com/OpenRobotLab/EmbodiedScan/blob/main/embodiedscan/explorer.py
# under the Apache 2.0 license.


import os
import pickle
from collections import defaultdict

import numpy as np
from data_loader.base_loader import BaseLoader

from spatial_analysis.relationship_utils import _9dof_to_box


class EmbodiedScanLoader(BaseLoader):
    """
    Loader for EmbodiedScan datasets (3RScan, ScanNet, Matterport3D).
    
    Inherits from BaseLoader and implements its interface methods.
    """
    
    def __init__(self, config):
        """
        Initialize the EmbodiedScan loader.
        
        Args:
            config (dict): Data loader configuration dictionary.
        """
        self.verbose = config["verbose"]

        if self.verbose:
            print('Loading EmbodiedScan...')
        
        # Get annotation key name from config
        annotation_key = config.get("annotation_key", "embodiedscan_ann")
        
        # Get splits from config
        splits = config.get("split")
        
        # Load annotation files based on splits
        ann_files = []
        if annotation_key in config:
            for split in splits:
                if split in config[annotation_key]:
                    ann_files.append(config[annotation_key][split])
        
        self.ann_files = ann_files

        
        self.metainfo = None
        ## Load embodiedscan annotated scan datasets (scannet, matterport3d, 3rscan, arkitscenes)
        data_list = []
        for file in self.ann_files:
            with open(file, 'rb') as f:
                data = pickle.load(f)
            
            if self.metainfo is None:
                self.metainfo = data['metainfo']
            else:
                assert self.metainfo == data['metainfo']
            
            data_list += data['data_list']


        if isinstance(self.metainfo['categories'], list):
            self.classes = self.metainfo['categories']
            self.id_to_index = {i: i for i in range(len(self.classes))}
        elif isinstance(self.metainfo['categories'], dict):
            self.classes = list(self.metainfo['categories'].keys())
            self.id_to_index = {
                i: self.classes.index(classes)
                for classes, i in self.metainfo['categories'].items()
            }

        # Check if certain scan exists
        self.data = defaultdict(dict)
        for data in data_list:

            splits = data['sample_idx'].split('/') # sample_idx is scene name
            dataset_name = splits[0]

            data['dataset'] = dataset_name
            if dataset_name == 'scannet':
                region = splits[1]
                dirpath = os.path.join(config['image_root'], dataset_name, 'posed_images',
                                        region)
            elif dataset_name == '3rscan':
                region = splits[1]
                dirpath = os.path.join(config['image_root'], dataset_name, region)
            elif dataset_name == 'matterport3d':
                building, region = splits[1], splits[2]
                dirpath = os.path.join(config['image_root'], dataset_name,
                                        building)
            else:
                region = splits[1]
                dirpath = os.path.join(self.data_root[dataset_name], region)
            if os.path.exists(dirpath):
                # scene_name is the scene name in the dataset with dataset name prepended if it is not already present
                scene_name = data['sample_idx']
                if not data['sample_idx'].startswith(dataset_name):
                    scene_name = f"{dataset_name}/{data['sample_idx']}"
                self.data[dataset_name][scene_name] = data
        # self.dataset_stats = {}
        # for dataset, data in self.data.items():
        #     self.dataset_stats[dataset] = len(data)

        if self.verbose:
            for dataset_name, data in self.data.items():
                print(f"Loaded {len(data)} scenes from {dataset_name}")
            print('Loading complete')
    
    def list_scenes(self, dataset_list):
        """
        Implementation of BaseLoader.list_scenes for EmbodiedScan datasets.
        
        Args:
            dataset_list (list): List of dataset names to query.
            
        Returns:
            generator: Yields tuples of (dataset_name, scene_idx, scene_name). 
            #NOTE scene_name is <dataset_name>/<scene_name>
        """
        for dataset_name in dataset_list:
            if dataset_name in self.data:
                for scene_idx, scene_name in enumerate(self.data[dataset_name]):
                    yield dataset_name, scene_idx, scene_name
    
    def list_images(self, dataset_name, scene_name):
        """
        Implementation of BaseLoader.list_images for EmbodiedScan datasets.
        
        Args:
            dataset_name (str): Name of the dataset.
            scene_name (str): Name of the scene. Example: scannet/scene00191_00
            
        Returns:
            list: List of image annotations.
        """
        if scene_name not in self.data[dataset_name]:
            if self.verbose:
                print(f"Warning: Scene {scene_name} not found in annotations")
            return []
        
        # Extract scene-wide annotations
        axis_align_matrix = np.array(self.data[dataset_name][scene_name]['axis_align_matrix']) # scene wide
        if "cam2img" in self.data[dataset_name][scene_name]:
            cam2img = np.array(self.data[dataset_name][scene_name]['cam2img']) # scene wide
        else:
            cam2img = np.array(self.data[dataset_name][scene_name]['images'][0]['cam2img']) # Some scenes have cam2img in images
        if "depth_cam2img" in self.data[dataset_name][scene_name]:
            depth_cam2img = np.array(self.data[dataset_name][scene_name]['depth_cam2img']) # scene wide
        else:
            depth_cam2img = []

        # Add scene-wide annotations to each image annotation
        image_annotations = {}
        for image_ann in self.data[dataset_name][scene_name]['images']:                 
            # Add dataset and scene information to image annotation
            image_ann['dataset'] = dataset_name
            image_ann['scene'] = scene_name
            image_ann['image_basename'] = os.path.basename(image_ann["img_path"]) #NOTE Actual image filename
            image_ann['extrinsic'] = axis_align_matrix @ image_ann['cam2global'] # Camera to world
            image_ann['intrinsic'] = cam2img # Camera to image
            image_ann['cam2img'] = cam2img
            image_ann['axis_align_matrix'] = axis_align_matrix
            image_ann['cam2global'] = image_ann['cam2global']
            image_ann['depth_cam2img'] = depth_cam2img
            image_ann['depth_path'] = image_ann['depth_path']
            image_ann['visible_instance_ids'] = image_ann['visible_instance_ids']
            image_name = scene_name + "/" + image_ann['image_basename']
            image_annotations[image_name] = image_ann #NOTE Image name is <scene_name>/<image_filename>
        
        return image_annotations
    
    def list_objects(self, dataset_name, scene_name, image_ann):
        """
        Implementation of BaseLoader.list_objects for EmbodiedScan datasets.
        
        Processes visible objects in an image and organizes them into multiple categories:
        - unique_vis_categories: Objects that appear exactly once (dictionary keyed by category)
        - multi_vis_categories: Objects that appear multiple times (dictionary keyed by category_0, category_1, etc.)
        - vis_objs: All visible objects
        - all_objs: All non-environmental objects
        
        Also calculates the floor boundaries for the scene.
        
        Args:
            dataset_name (str): Name of the dataset.
            scene_name (str): Name of the scene.
            image_ann (dict): Image annotation dictionary.
            
        Returns:
            tuple: A 5-element tuple containing:
                - vis_objs (dict): Dictionary of visible, non-environmental objects. Keys are categories (indexed if duplicates exist, e.g., 'chair_0', 'chair_1'). Values are instance dictionaries.
                - unique_vis_categories (set): Set of categories for objects appearing only once (excluding environmental/object categories).
                - multi_vis_categories (set): Set of categories for objects appearing multiple times (excluding environmental/object categories).
                - floor_bound (list): Min and max floor boundaries as [min_bound, max_bound].
                - all_objs (dict): Dictionary of all non-environmental objects (floor, wall, ceiling excluded), keyed by their potentially indexed name (e.g., 'chair_0'). Used for occupancy map calculation.
        """
        # Get visible instance ids from image annotation
        #NOTE you can use different ways to get this.
        visible_instance_ids = image_ann['visible_instance_ids']    
        
        if scene_name not in self.data[dataset_name]:
            if self.verbose:
                print(f"Warning: Scene {scene_name} not found in annotations")
            return {}, set(), set(), [], {} # Return empty structures matching the new return type

        # First pass to count occurrences of each non-environmental category
        category_total_counts = defaultdict(int)
        for i in visible_instance_ids:
            instance = self.data[dataset_name][scene_name]['instances'][i]
            category = self.classes[self.id_to_index[instance['bbox_label_3d']]]
            # Exclude environmental or generic object categories from indexed naming
            if category not in ["wall", "ceiling", "floor", "object"]:
                category_total_counts[category] += 1
                
        # Process instances to create the unified vis_objs dictionary
        vis_objs = {}
        unique_vis_categories = set()
        multi_vis_categories = set()
        category_indices = defaultdict(int) # To track current index for duplicate categories
        env_objs = {}
        all_objs = {} # Still needed for floor bounding box calculation

        for i in visible_instance_ids:
            instance = self.data[dataset_name][scene_name]['instances'][i]
            category = self.classes[self.id_to_index[instance['bbox_label_3d']]]
            instance["category"] = category # Keep original label name in instance dict
            instance["obb"] = _9dof_to_box(instance["bbox_3d"]) # We use Open3D obb for all spatial relationships

            # Handle environmental objects (for floor calculation)
            if category in ["floor", "wall", "ceiling"]:
                env_objs[category] = instance

            # Parse categories to handle duplicates, assume there is only one floor, wall, and ceiling
            total_count = category_total_counts[category]
            if total_count == 1:
                obj_key = category
                instance["name"] = obj_key
                unique_vis_categories.add(category)
            else:
                current_index = category_indices[category]
                obj_key = f"{category}_{current_index}"
                instance["name"] = obj_key
                multi_vis_categories.add(category)
                category_indices[category] += 1
            
            # Add to vis_objs if it is not an environmental object
            if category not in ["wall", "ceiling", "floor", "object"]:
                vis_objs[obj_key] = instance
            
            # Get all objects for occupancy map calculation
            if category not in ["floor", "wall", "ceiling"]:
                all_objs[obj_key] = instance 

            # Track all non-floor/wall/ceiling objects for OBB calculation - This part is now handled above
            # if category not in ["floor", "wall", "ceiling"]:
            #    all_objs[i] = instance # Use original instance id as key
        
        all_obbs = [obj["obb"] for obj in all_objs.values()]

        # Create floor box representation automatically
        # Ensure floor object exists before accessing it
        if "floor" in env_objs:
            floor_obj = env_objs["floor"]
            floor_obb = _9dof_to_box(floor_obj["bbox_3d"])
            min_bound = np.min([box.get_min_bound() for box in all_obbs + [floor_obb]], axis=0)
            max_bound = np.max([box.get_max_bound() for box in all_obbs + [floor_obb]], axis=0)
            floor_bound = [min_bound, max_bound]
        else:
            # Handle cases where there might not be a floor object detected/annotated
            if len(all_obbs) > 0:
                min_bound = np.min([box.get_min_bound() for box in all_obbs], axis=0)
                max_bound = np.max([box.get_max_bound() for box in all_obbs], axis=0)
                floor_bound = [min_bound, max_bound] # Use bounds from other objects if floor is missing
            else:
                floor_bound = None

        return vis_objs, unique_vis_categories, multi_vis_categories, floor_bound, all_objs
