# Copyright (c) 2024-2025 NVIDIA CORPORATION. All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.


from abc import ABC, abstractmethod

class BaseLoader(ABC):
    """
    Abstract base class for dataset loaders.
    
    This class defines the interface that dataset loaders must implement
    to be compatible with the annotation generation pipeline.
    """
    
    @abstractmethod
    def __init__(self, config):
        """
        Initialize the dataset loader.
        
        Args:
            config (dict): Configuration dictionary containing dataset parameters.
        """
        pass
    
    @abstractmethod
    def list_scenes(self, dataset_list):
        """
        List all scenes available in the specified datasets.
        
        Args:
            dataset_list (list): List of dataset names to query.
            
        Returns:
            generator: Yields tuples of (dataset_name, scene_idx, scene_name).
        """
        pass
    
    @abstractmethod
    def list_images(self, dataset_name, scene_name):
        """
        List all images available in the specified scene.
        
        Args:
            dataset_name (str): Name of the dataset.
            scene_name (str): Name of the scene (e.g., 'scannet/scene00191_00').
            
        Returns:
            dict: Dictionary of image annotations keyed by image name (e.g., '<scene_name>/<image_filename>'),
                  each containing at minimum:
                - extrinsic: Camera to global transformation matrix.
                - intrinsic: Camera to image transformation matrix.
                - img_path: Path to the image file.
                - (If needed) axis_align_matrix: Matrix to align to world coordinates.
        """
        pass
    
    @abstractmethod
    def list_objects(self, dataset_name, scene_name, image_ann):
        """
        List all object instances visible in an image in the specified scene.
        
        Processes visible objects in an image and organizes them.
        
        Args:
            dataset_name (str): Name of the dataset.
            scene_name (str): Name of the scene.
            image_ann (dict): Image annotation dictionary from list_images.
            
        Returns:
            tuple: A 5-element tuple containing:
                - vis_objs (dict): Dictionary of visible, non-environmental objects.
                                   Keys are categories (indexed if duplicates exist, e.g., 'chair_0').
                                   Values are instance dictionaries.
                - unique_vis_categories (set): Set of categories for objects appearing only once
                                               (excluding environmental/generic object categories).
                - multi_vis_categories (set): Set of categories for objects appearing multiple times
                                              (excluding environmental/generic object categories).
                - floor_bound (list): Min and max floor boundaries derived from object OBBs,
                                      as [min_bound, max_bound].
                - all_objs (dict): Dictionary of all non-environmental objects (floor, wall, ceiling excluded),
                                   keyed by their potentially indexed name (e.g., 'chair_0').
                                   Used for occupancy map calculation or other downstream tasks.
        """
        pass 