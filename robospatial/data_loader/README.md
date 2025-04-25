# Implementing a Custom Data Loader

This document outlines the steps and requirements for implementing a custom data loader compatible with the RoboSpatial annotation generation pipeline.

## Overview

The data loader is responsible for interfacing with your specific dataset format and providing the necessary information (scenes, images, object instances, metadata) to the generation pipeline. To ensure compatibility, your custom loader must inherit from the `BaseLoader` abstract base class (`robospatial.data_loader.base_loader.BaseLoader`) and implement its required methods.

## BaseLoader Interface

Your custom loader class must implement the following methods:

### `__init__(self, config)`

*   **Purpose:** Initializes the data loader. This typically involves loading annotations, setting up paths, and potentially pre-processing metadata.
*   **Args:**
    *   `config (dict)`: A dictionary containing the `data_loading` section from the configuration file (e.g., `configs/embodiedscan.yaml`). This allows access to dataset paths, annotation file locations, selected datasets, splits, and other relevant parameters.
*   **Implementation Notes:**
    *   Use the `config` dictionary to locate and load your dataset's annotation files.
    *   Store necessary metadata, such as class labels and mappings, as instance variables.
    *   Organize the loaded data in a way that facilitates efficient retrieval by the other methods (e.g., nested dictionaries keyed by dataset and scene name, as seen in `EmbodiedScanLoader`).

### `list_scenes(self, dataset_list)`

*   **Purpose:** Provides a generator that yields information about each scene within the specified datasets.
*   **Args:**
    *   `dataset_list (list)`: A list of dataset names (strings) requested by the pipeline (e.g., `['scannet', '3rscan']`).
*   **Returns:**
    *   `generator`: Yields tuples of `(dataset_name, scene_idx, scene_name)`.
        *   `dataset_name (str)`: The name of the dataset the scene belongs to.
        *   `scene_idx (int)`: A unique index for the scene within its dataset (can be a simple counter).
        *   `scene_name (str)`: A unique identifier for the scene, often including the dataset prefix (e.g., `'scannet/scene0000_00'`). This name is used in subsequent calls.

### `list_images(self, dataset_name, scene_name)`

*   **Purpose:** Lists all images (or viewpoints) associated with a specific scene.
*   **Args:**
    *   `dataset_name (str)`: The name of the dataset.
    *   `scene_name (str)`: The unique identifier of the scene (obtained from `list_scenes`).
*   **Returns:**
    *   `dict`: A dictionary where keys are unique image identifiers (e.g., `'<scene_name>/<image_filename>'` or `'<scene_name>/<frame_id>'`) and values are dictionaries containing image-specific annotations. Each image annotation dictionary **must** include:
        *   `extrinsic` (or equivalent): 4x4 Transformation matrix (e.g., NumPy array or list of lists) from camera coordinates to the global/world coordinate system of the scene.
          ```python
          # Example:
          [[ -0.9897,  0.1085,  0.0927,  1.2120],
           [ -0.0330,  0.4577, -0.8884,  0.3075],
           [ -0.1388, -0.8824, -0.4494,  1.4804],
           [  0.    ,  0.    ,  0.    ,  1.    ]]
          ```
        *   `intrinsic`: 4x4 Camera intrinsics matrix (e.g., NumPy array or list of lists).
          ```python
          # Example:
          [[ 1170.18,    0.  ,  647.75,    0.  ],
           [    0.  , 1170.18,  483.75,    0.  ],
           [    0.  ,    0.  ,    1.  ,    0.  ],
           [    0.  ,    0.  ,    0.  ,    1.  ]]
          ```
        *   `img_path`: Absolute or relative path to the image file. `img_path` gets joined with `image_root` path in the config file.
        *   Any other metadata required by `list_objects` (e.g., `visible_instance_ids` in `EmbodiedScanLoader`).

### `list_objects(self, dataset_name, scene_name, image_ann)`

*   **Purpose:** Identifies and processes object instances visible from a specific viewpoint (image). It organizes objects based on visibility and category, handles duplicate categories, and calculates scene bounds.
*   **Args:**
    *   `dataset_name (str)`: The name of the dataset.
    *   `scene_name (str)`: The unique identifier of the scene.
    *   `image_ann (dict)`: The annotation dictionary for a single image, obtained from the output of `list_images`.
*   **Returns:**
    *   `tuple`: A 5-element tuple containing:
        1.  `vis_objs (dict)`: Dictionary of *visible*, *non-environmental* objects.
            *   Keys: Object category name. If multiple instances of the same category are visible, append an index (e.g., `'chair_0'`, `'chair_1'`). Environmental objects like 'wall', 'floor', 'ceiling', and generic 'object' categories should be excluded.
            *   Values: Instance annotation dictionaries. Each dictionary should contain at least:
                *   `category (str)`: The original object category label.
                *   `name (str)`: The potentially indexed name used as the key in `vis_objs`.
                *   `bbox_3d` (or equivalent, optional but recommended): The original 3D bounding box representation from your dataset (e.g., 9 DoF parameters: center, size, orientation). While the pipeline primarily uses the `obb` for calculations, this original `bbox_3d` is saved in the final annotations if provided.
                *   `obb`: The Open3D `OrientedBoundingBox` representation (`open3d.geometry.OrientedBoundingBox`). **This is crucial for spatial relationship calculations.** Your `list_objects` implementation is responsible for creating this, often by converting from `bbox_3d` (see `EmbodiedScanLoader` line ~241 for an example using `_9dof_to_box`) or by generating it directly if your dataset provides OBBs.
        2.  `unique_vis_categories (set)`: A set of category names (strings) for objects that appear *exactly once* in the `vis_objs` dictionary (excluding environmental/generic categories).
        3.  `multi_vis_categories (set)`: A set of category names (strings) for objects that appear *multiple times* in the `vis_objs` dictionary (excluding environmental/generic categories).
        4.  `floor_bound (list)`: A list containing two `numpy.ndarray`s representing the minimum and maximum coordinates `[min_bound, max_bound]` that encompass the floor and all non-environmental objects. This is often derived from the combined OBBs of relevant objects.
        5.  `all_objs (dict)`: Dictionary of *all* non-environmental objects associated with the *scene* (not just the current view), keyed by their potentially indexed name (e.g., 'chair_0').
                                   Used for occupancy map generation or other downstream tasks. The structure mirrors `vis_objs` but includes objects not necessarily visible in the current `image_ann`.
                                   Each object dictionary must contain at least `category`, `name`, and `obb`. Including `bbox_3d` is recommended if available.
                                   *Note: Depending on your dataset structure, you might populate this similarly to `vis_objs` based on `visible_instance_ids` or load all scene objects separately.*

## Configuration

To use your custom data loader, update the `data_loading` section in your configuration file (e.g., `configs/example_config.yaml`):

```yaml
data_loading:
  # ... other settings ...
  loader_class: path.to.your.module.YourCustomLoaderClassName # Update this line
  # Provide any custom keys your loader's __init__ needs
  your_custom_annotation_path:
    train: /path/to/your/train_annotations.pkl
    val:   /path/to/your/val_annotations.pkl
  # ... other dataset-specific paths or parameters ...
```

*   Set `loader_class` to the fully qualified Python path of your custom loader class.
*   Ensure any necessary configuration parameters (like annotation file paths) needed by your loader's `__init__` method are present in the `data_loading` section.

## Example

Refer to `data_loader.embodiedscan_loader.EmbodiedScanLoader` for a concrete implementation example using datasets like ScanNet, Matterport3D, and 3RScan.

Additionally, refer to `data_loader.example_loader.py` for a simpler implementation tailored specifically to the JSON annotation format found in the `example_data/` directory. This loader demonstrates how to handle the example annotations provided for testing the pipeline.

## Visualizing Your Loader Output

To verify that your custom data loader is producing the correct outputs (specifically the object instances with their 3D bounding boxes and camera parameters), you can use the provided visualization script: `scripts/visualize_input.py`.

**Purpose:**

This script takes an image file and a corresponding intermediate annotation JSON file (similar to those in `example_data/annotations/`, representing the data your loader would prepare for a single image) as input. It reads the camera parameters (`extrinsic`, `intrinsic`) and the object information (specifically `bbox_3d`) from the JSON. It then projects the 3D bounding boxes onto the 2D image and displays the result.

This helps you visually confirm:

*   Camera parameters (`extrinsic`, `intrinsic`) are correct.
*   Oriented object bounding boxes (derived from `bbox_3d`) align with the objects in the image.
*   The data format your loader prepares is being interpreted correctly before passing it to the main pipeline.

**Important Note:**

The provided visualization script, `scripts/visualize_input.py`, is designed to help debug your custom loader's output *before* running the full generation pipeline. It reads an intermediate JSON file (like those in `example_data/annotations/`) which represents the data your loader passes for a single image.

Currently, this script expects the JSON to contain an `objects` array. For each object in this array, it specifically looks for a `bbox_3d` field containing a list with 9 DoF parameters (center, size, rotation) as its first element. It uses these parameters to generate an Open3D `OrientedBoundingBox` (`obb`) via the `_9dof_to_box` function for visualization.

*   **If your custom loader generates an intermediate JSON where the 3D bounding box information is stored differently (e.g., different format within `bbox_3d`, different field name, or only providing a pre-computed `obb`),** you will need to modify the `visualize_single_image` function in `scripts/visualize_input.py` (around line 195) to correctly parse your data and create the `o3d_box` for drawing.
