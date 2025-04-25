# RoboSpatial Annotation Generation Logic

For those who wants to hack the codebase!

## Annotation Generation Details

This section provides a more detailed overview of the logic used to generate each type of spatial annotation and highlights the key configuration parameters found in `configs/example_config.yaml` (and other configuration files) that control this process.

### 1. Object Grounding (`spatial_analysis/grounding.py`)

*   **Purpose:** Generates a tight 2D axis-aligned bounding box (`clipped_bbox`) encompassing all visible pixels of an object in the image.
*   **Logic:**
    *   Relies on a pre-calculated 2D boolean `occupancy_map` for each object, which indicates the precise pixels covered by the object's 3D model when projected onto the image.
    *   It finds the minimum and maximum `x` (column) and `y` (row) coordinates within this occupancy map.
    *   These bounds directly define the `[xmin, ymin, xmax, ymax]` coordinates of the 2D bounding box.
*   **Key Parameters:** None directly in the configuration for this step; it depends on the accuracy of the input 3D models and the camera parameters used to generate the `occupancy_map`.

### 2. Spatial Context (`spatial_analysis/context/context.py`)

*   **Purpose:** Samples points in the empty space surrounding a *reference object* and categorizes them based on their spatial relationship (infront, behind, left, right) in three different frames: `objectcentric`, `cameracentric`, and `worldcentric`.
*   **Logic:**
    1.  Calculates empty space on the floor using a top-down 2D grid based on environment geometry.
    2.  Identifies empty grid points within a specific distance (`threshold`) from the reference object's 2D footprint.
    3.  Projects these candidate 3D points (at the object's base height) onto the image, filtering those outside the view or behind the camera.
    4.  Checks if the projected 2D points are occluded by *other* objects using a pre-computed environment occupancy map.
    5.  Categorizes the non-occluded points based on their position relative to the reference object in the three frames (using object orientation for object-centric, and pixel/depth coordinates for camera/world-centric).
    6.  Randomly samples up to `num_samples` non-occluded points for each valid category (frame + direction).
*   **Key Parameters (`configs/example_config.yaml` -> `data_generation.generation_options`):**
    *   `context_threshold`: Maximum distance (in world units, e.g., meters) from the reference object's footprint to consider sampling points.
    *   `context_grid_resolution`: The size of each cell in the temporary top-down 2D grid used for finding nearby empty space. Smaller values are more precise but computationally more expensive.
    *   `context_num_samples`: The maximum number of points to sample and output for each valid category (e.g., max 10 points for 'camera_centric' 'left').

### 3. Spatial Configuration (`spatial_analysis/configuration/configuration.py`)

*   **Purpose:** Determines the pairwise spatial relationship between two objects (`obj1` relative to `obj2`) across the three reference frames (camera, world, object). Relationships include left/right, infront/behind, above/below, and overlapping.
*   **Logic:**
    1.  Calculates various geometric metrics for both objects (projected 2D bounds, average visible depth, world Z bounds, etc.) using their individual pre-computed `occupancy_map`s.
    2.  **Camera/World-centric:** Compares these metrics. The `strictness` parameter controls the comparison method:
        *   `'strict'`: Uses the absolute min/max bounds. Requires clear separation; considers objects overlapping if their projected bounds intersect at all. Sensitive to partial occlusions.
        *   `'lenient'`: Uses projected centers, average visible depths, and average Z coordinates. More robust to partial occlusion but might misclassify tightly packed objects.
    3.  **Object-centric:** Uses the Separating Axis Theorem (SAT) on the 3D OBBs to check for overlap. If not overlapping, it determines the direction based on the relative position of `obj1`'s center projected onto `obj2`'s local forward and right axes. Above/below still uses world Z coordinates.
*   **Key Parameters (`configs/example_config.yaml` -> `data_generation.generation_options`):**
    *   `spatial_configuration_strictness`: (`'strict'` or `'lenient'`) Selects the comparison logic for camera-centric and world-centric frames. Default is `'lenient'`.
    *   `pairwise_relationship_mode`: (`'unique_categories_only'` or `'all_visible_objects'`) Determines which pairs of objects are considered for configuration analysis. `'unique_categories_only'` only considers pairs where each object is the only instance of its category visible, while `'all_visible_objects'` considers all permutations of visible objects.

### 4. Spatial Compatibility (`spatial_analysis/compatibility/compatibility.py`)

*   **Purpose:** Assesses whether one object (`obj_a`) *could* be placed in the empty space relative to another (`obj_b`) without collision. It checks directions like left, right, in front, behind, and specifically `on_top`.
*   **Logic:**
    1.  Samples potential placement points around `obj_b` using the Spatial Context logic (`get_point_in_space_relative_to_object`), using a dynamic threshold based on the sizes of `obj_a` and `obj_b`.
    2.  For each sampled point, it simulates placing `obj_a` horizontally centered at that point's 2D location.
    3.  It checks for collisions between the placed `obj_a` (potentially with a `buffer_ratio`) and:
        *   The static environment (using a 2D occupancy grid).
        *   The reference object `obj_b` (maintaining a `min_distance`).
    4.  A relationship (e.g., 'left') is considered compatible (`True`) if *any* sampled point corresponding to that relationship allows `obj_a` to fit.
    5.  A separate, simpler check (`can_fit_on_top`) determines the 'on_top' relationship by comparing the horizontal dimensions of `obj_a` and `obj_b`, but only if `obj_a` is placeable and `obj_b` has a flat surface.
*   **Key Parameters (`configs/example_config.yaml` -> `data_generation.generation_options`):**
    *   `compatibility_grid_resolution`: Resolution of the 2D grid used for collision checking against the environment.
    *   `compatibility_num_samples`: How many potential placement points to sample around `obj_b`.
    *   `compatibility_min_distance`: The minimum required distance (in world units) between the placed `obj_a` and the reference `obj_b`.
    *   `compatibility_buffer_ratio`: A ratio applied to `obj_a`'s dimensions during collision checks, effectively adding a safety margin. 0 means no buffer, 0.1 means 10% buffer.
    *   `context_threshold`: The *base* threshold used for sampling points (dynamically increased based on object sizes).

---

## Project Structure

*   `configs/`: Contains YAML configuration files (e.g., `example_config.yaml`).
*   `data_loader/`: Modules for loading and interfacing with different 3D datasets. Includes `embodiedscan_loader.py` and a [README](data_loader/README.md) explaining how to add custom loaders.
*   `spatial_analysis/`: Modules performing the core spatial reasoning and annotation generation logic.
    *   `context/`: Logic for spatial context (points relative to an object).
    *   `configuration/`: Logic for spatial configuration (relative position between objects).
    *   `compatibility/`: Logic for spatial compatibility (fitting assessment).
    *   `grounding.py`: Logic for 2D object grounding.
    *   `relationships.py`: High-level wrappers for spatial analysis functions.
    *   `relationship_utils.py`: Utility functions for geometry and projections.
    *   `topdown_map.py`: Functions for creating 2D top-down occupancy grids.
    *   `obj_properties.py`: Lists defining object properties (e.g., `items_with_face`).
*   `annotation_generator.py`: Orchestrates the generation process for a single scene.
*   `run_generation.py`: Main script to run annotation generation across datasets/scenes.

## Output Files

*   **`<output_dir>/<dataset>/<scene_id>/<image_name>.annotations.json`**: The primary output. Contains the generated spatial annotations for a single image, structured by type (grounding, unary relations, pairwise relations).
*   **`generation_progress.json`**: Stores a map of datasets to lists of scene names that have been successfully processed. Allows the script to resume if interrupted. Located in the directory where `run_generation.py` is executed.
*   **`generation_stats.json`**: Contains aggregated statistics about the generated annotations (e.g., counts of each annotation type) overall and per-dataset. Located in the directory where `run_generation.py` is executed.
