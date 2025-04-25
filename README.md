# RoboSpatial: Teaching Spatial Understanding to 2D and 3D Vision-Language Models for Robotics

[**🌐 Homepage**](https://chanh.ee/RoboSpatial/) | [**📖 arXiv**](https://arxiv.org/abs/2411.16537) | [**📂 Benchmark**](https://huggingface.co/datasets/chanhee-luke/RoboSpatial-Home) | [**📊 Evaluation**](https://github.com/chanhee-luke/RoboSpatial-Eval)

**✨ CVPR 2025 (Oral) ✨**

Authors: [Chan Hee Song](https://chanh.ee)<sup>1</sup>, [Valts Blukis](https://research.nvidia.com/person/valts-blukis)<sup>2</sup>, [Jonathan Tremblay](https://research.nvidia.com/person/jonathan-tremblay)<sup>2</sup>, [Stephen Tyree](https://research.nvidia.com/person/stephen-tyree)<sup>2</sup>, [Yu Su](https://ysu1989.github.io/)<sup>1</sup>, [Stan Birchfield](https://sbirchfield.github.io/)<sup>2</sup>

 <sup>1</sup> The Ohio State University  <sup>2</sup> NVIDIA

---

## 🔔News

- **🔥[2025-04-24]: Released the RoboSpatial data generation pipeline, RoboSpatial-Home dataset, and evaluation script!**

---

**Project Components:**

This repository contains the code for **generating** the spatial annotations used in the RoboSpatial dataset.

*   **Benchmark Dataset:** [**📂 RoboSpatial-Home**](https://huggingface.co/datasets/chanhee-luke/RoboSpatial-Home)
*   **Evaluation Script:** [**📊 RoboSpatial-Eval**](https://github.com/chanhee-luke/RoboSpatial-Eval)

**Coming up!**

-    [ ] Unified data loader supporting [BOP datasets](https://bop.felk.cvut.cz/datasets/) and [GraspNet dataset](https://graspnet.net/). (Turn object pose estimation datasets into spatial QA!)
-    [ ] Support for additional scan datasets like [SCRREAM](https://sites.google.com/view/scrream/about).

---

# RoboSpatial Annotation Generation

This codebase generates rich spatial annotations for 3D scan datasets. While initially built using the [EmbodiedScan](https://github.com/OpenRobotLab/EmbodiedScan) conventions, it is designed to be extensible to other data formats through custom data loaders (see [Data Loader Documentation](#data-loader-documentation)). It extracts various spatial relationships from image data and associated 3D information, including:

*   **Object Grounding:** Locating objects mentioned in text within the image.
*   **Spatial Context:** Identifying points in empty space relative to objects (e.g., "in front of the chair").
*   **Spatial Configuration:** Describing the relative arrangement of multiple objects (e.g., "the chair is next to the table").
*   **Spatial Compatibility:** Determining if an object *could* fit in a specific location.

The generated annotations are saved in JSON format, one file per image.

## Prerequisites

1.  **Python Environment:** Ensure you have a Python environment set up (e.g., using `conda` or `venv`). Required packages can be installed via `pip install -r requirements.txt`.
2.  **Datasets:** You need access to the 3D scan datasets you intend to process.
   * **Note:** For specific instructions on downloading and setting up the **EmbodiedScan** dataset, please refer to the guide in [**`data/README.md`**](data/README.md).
3.  **Configuration:** The main configuration file (e.g., `robospatial/configs/embodiedscan.yaml`) needs to be updated with paths relevant to your chosen data loader and dataset:
    *   `data_loading.loader_class`: Specifies the Python class for your data loader (e.g., `data_loader.embodiedscan_loader.EmbodiedScanLoader`).
    *   Dataset-specific paths (e.g., `image_root`, format-specific annotation files like `embodiedscan_ann`). Consult the configuration file and your data loader's requirements. See [Data Loader Documentation](#data-loader-documentation) for more details on adding custom formats.
    *   `data_generation.output_dir`: The directory where the generated `.annotations.json` files will be saved.

## Running Annotation Generation

The core script for generating annotations is `robospatial/run_generation.py`.

**Running with Provided Example Data (Recommended First Step):**

We provide a small example scene with input annotations and images in the `example_data/` directory. This allows you to test the generation pipeline without downloading large datasets.

1.  **Navigate to the `robospatial` directory:**
    ```bash
    cd robospatial
    ```
2.  **Run the generation script:**
    ```bash
    python run_generation.py --config configs/example_dataset.yaml
    ```
    This will process only the example scene defined in `example_dataset.yaml` and generate the annotation in the `example_data/example_qa` folder.

**Running on Full Datasets:**

Once you have confirmed the example works and have downloaded your target datasets:

1.  **Configure your data loader:** Ensure the `data_loading` section in your chosen configuration file (e.g., `configs/example_dataset.yaml`) correctly points to your dataset paths and uses the appropriate `loader_class`.
2.  **Run the script:**
    ```bash
    cd robospatial
    python run_generation.py --config configs/your_chosen_config.yaml
    ```

This command will process all scenes found by the data loader using the settings defined in `your_chosen_config.yaml`.

**Command-Line Options:**

*   `--config <path>`: **(Required)** Specifies the path to the YAML configuration file.
*   `--scene <dataset/scene_id>`: Process only a single specific scene.
    ```bash
    python run_generation.py --config configs/embodiedscan.yaml --scene "scannet/scene0191_00"
    ```
*   `--image <image_basename>`: Process only a single specific image within the specified scene (requires `--scene`). Useful for debugging.
    ```bash
    python run_generation.py --config configs/embodiedscan.yaml --scene "scannet/scene0191_00" --image "00090.jpg"
    ```
*   `--range <start_idx> <end_idx>`: Process a specific range of scenes based on their index in the loaded list (inclusive start, inclusive end).
    ```bash
    python run_generation.py --config configs/embodiedscan.yaml --range 0 10 # Process first 11 scenes
    ```
*   `--num_workers <int>`: Specify the number of parallel worker threads to use for processing scenes. Overrides the `num_workers` setting in the config file. Defaults to `min(os.cpu_count(), 4)` if neither is provided.
    ```bash
    python run_generation.py --config configs/embodiedscan.yaml --num_workers 8
    ```
*   `--dry-run`: Process only the first 5 images of each scene. Useful for quickly testing the pipeline.
    ```bash
    python run_generation.py --config configs/embodiedscan.yaml --dry-run
    ```

## Visualizing Input/Outputs

Two scripts are provided in the `scripts/` directory for visualizing inputs/outputs:

### 1. Visualizing Input Data (`scripts/visualize_input.py`)

Use this script to check if your input annotations (e.g., 3D bounding boxes from your dataset's original format, after conversion by your data loader) are being loaded and interpreted correctly. It reads the intermediate JSON format produced by the data loader for a single image and overlays the 3D bounding boxes onto the image.

**Usage:**

```bash
python scripts/visualize_input.py \
    --image_path <path_to_specific_image.jpg> \
    --annotation_file <path_to_intermediate_json_for_image>
```

*   Replace `<path_to_specific_image.jpg>` with the direct path to the image file.
*   Replace `<path_to_intermediate_json_for_image>` with the path to the JSON file representing the *input* annotations for that image (this file's location and naming depend on your data loader implementation).

**Example using the provided example data:**
```bash
python scripts/visualize_input.py \
    --image_path example_data/images/example_dataset/example_scene/example_image.jpg \
    --annotation_file example_data/annotations/example_input.json
```

### 2. Visualizing Generated Output (`scripts/visualize_output.py`)

Use this script to debug and inspect the spatial relationships generated by `run_generation.py`. It reads the final `.annotations.json` file for a specific image and allows you to visualize different types of generated annotations, including object grounding and spatial relationships (context, configuration, compatibility).

**Usage:**

```bash
python scripts/visualize_output.py \
    --image_path <path_to_specific_image.jpg> \
    --annotation_file <path_to_output_dir>/<dataset>/<scene_id>/<image_name>.annotations.json \
    --object_3d_grounding \
    --context
```

*   Replace `<path_to_specific_image.jpg>` with the direct path to the image file.
*   Replace `<path_to_output_dir>` with the path used in your configuration's `data_generation.output_dir`.
*   Adjust `<dataset>`, `<scene_id>`, and `<image_name>` to match the specific output file you want to visualize.
*   Include flags like `--object_2d_grounding`, `--object_3d_grounding`, `--context`, `--configuration`, or `--compatibility` to select what to visualize. Use the `--verbose` or `-v` flag for more detailed output. Refer to the script's internal documentation (`--help`) for detailed controls and options.

**Example using the provided example data (run the generation first):**
```bash
python scripts/visualize_output.py \
    --image_path example_data/images/example_dataset/example_scene/example_image.jpg \
    --annotation_file example_data/example_qa/example_scene/example_image.jpg.annotations.json \
    --object_3d_grounding \
    --context
```

## Data Loader Documentation

This project supports adding custom data loaders to handle different 3D dataset formats. The configuration file (`data_loading.loader_class`) specifies which loader to use.

For detailed instructions on the expected interface for a data loader and how to implement your own, please refer to the README within the data loader directory: [**`robospatial/data_loader/README.md`**](robospatial/data_loader/README.md)

## Project Structure

For a detailed explanation of the annotation generation logic and hyperparameters within the `spatial_analysis` modules, please refer to the [**`robospatial/README.md`**](robospatial/README.md).

*   `robospatial/`: Main source code directory.
    *   `configs/`: Contains YAML configuration files (e.g., `example_config.yaml`).
    *   `data_loader/`: Contains modules for loading and interfacing with different 3D datasets. Includes examples like `embodiedscan_loader.py` and can be extended with custom loaders. See the [README](robospatial/data_loader/README.md) in this directory for details.
    *   `spatial_analysis/`: Modules performing the core spatial reasoning and annotation generation logic.
    *   `annotation_generator.py`: Orchestrates the generation process for a single scene by calling functions from `spatial_analysis`.
    *   `run_generation.py`: Main script to run the annotation generation across datasets/scenes based on configuration.
    *   `visualize_annotations.py`: (Presumed) Script for visualizing generated annotations.

## Output Files

*   **`<output_dir>/<dataset>/<scene_id>/<image_name>.annotations.json`**: The primary output. Contains the generated spatial annotations for a single image.
*   **`generation_progress.json`**: Stores a list of scenes that have been successfully processed. This allows the script to resume if interrupted. Located in the directory where `run_generation.py` is executed.
*   **`generation_stats.json`**: Contains aggregated statistics about the generated annotations (e.g., counts of each annotation type) overall and per-dataset. Located in the directory where `run_generation.py` is executed.

## Acknowledgements

We thank the authors of [EmbodiedScan](https://github.com/OpenRobotLab/EmbodiedScan/tree/main) for providing their unified annotations for various 3D scan datasets, which served as the foundation for this project's data loading capabilities.

## Contact
- Luke Song: song.1855@osu.edu
- Or Github Issues!

## Citation

**BibTex:**
```bibtex
@inproceedings{song2025robospatial,
  author    = {Song, Chan Hee and Blukis, Valts and Tremblay, Jonathan and Tyree, Stephen and Su, Yu and Birchfield, Stan},
  title     = {{RoboSpatial}: Teaching Spatial Understanding to {2D} and {3D} Vision-Language Models for Robotics},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year      = {2025},
  note      = {Oral Presentation},
}
```
