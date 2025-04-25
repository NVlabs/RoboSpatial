# Copyright (c) 2024-2025 NVIDIA CORPORATION. All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Main entry script for generating spatial annotations from 3D scan datasets.

This script orchestrates the annotation generation process based on a YAML
configuration file. It handles:
- Parsing command-line arguments for configuration path, scene filtering (range, specific scene, specific image), and dry runs.
- Loading the dataset configuration and initializing the appropriate data loader.
- Iterating through specified datasets and scenes.
- Calling the `generate_and_save_annotations` function from `annotation_generator.py`
  to perform the core annotation generation for each scene.
- Tracking generation progress across scenes and saving it periodically.
- Aggregating and saving final statistics (overall and per-dataset).

Supported annotation types (configured via YAML) include:
- Object Grounding
- Spatial Context
- Spatial Configuration
- Spatial Compatibility

Usage:
    python robospatial/run_generation.py --config path/to/your/config.yaml [options]
"""

# run_generation.py
# Main entry script to generate annotations from 3D scan datasets
# Supports flexible configuration of annotation types (object grounding, spatial context, spatial configuration, spatial compatibility)

import argparse
import yaml
import os
import json
import importlib
import concurrent.futures
from tqdm import tqdm
from collections import defaultdict

# Import the new generator function
from annotation_generator import generate_and_save_annotations

def parse_args():
    parser = argparse.ArgumentParser(description="Parse configuration file for annotation generation.")
    parser.add_argument('--config', type=str, default="configs/base_local.yaml", help='Path to the configuration YAML file.')
    parser.add_argument('--range', type=int, nargs=2, help='Range of scene indices to process (inclusive start, exclusive end).')
    parser.add_argument('--scene', type=str, help='Specific scene name to process (e.g., "scannet/scene0190_00").')  # New argument for specific scene
    parser.add_argument('--image', type=str, help='Specific image basename (e.g., "rgb_00010") to process within the specified scene for debugging.') # New argument for specific image
    parser.add_argument('--num_workers', type=int, help='Number of worker threads to use for processing scenes.')
    parser.add_argument('--dry_run', action='store_true', help='Enable dry run mode (processes only the first 5 images per scene).')
    args = parser.parse_args()

    if args.range:
        start, end = args.range
        # Make range inclusive by adding 1 to end for Python range behavior
        args.range = range(start, end + 1)  # Store as a range object
        if start > end:
            parser.error("Start of range must not be greater than end.")

    return args

def load_config(config_file):
    with open(config_file, 'r') as file:
        config = yaml.safe_load(file)
    # Add default output suffix if not present
    if "output_suffix" not in config.get("data_generation", {}):
        if "data_generation" not in config:
            config["data_generation"] = {}
        config["data_generation"]["output_suffix"] = ".annotations.json"
    return config

def create_loader(config):
    """Create a loader instance based on configuration."""
    # Default to EmbodiedScanLoader if not specified
    loader_class_path = config.get("loader_class")
    if loader_class_path is None:
        raise ValueError("loader_class not specified in config[data_loading]")

    module_name, class_name = loader_class_path.rsplit('.', 1)
    module = importlib.import_module(module_name)
    loader_class = getattr(module, class_name)
    
    # Create the loader instance with config data
    loader = loader_class(config)
    
    return loader

# Define the scene processing function outside run or nested inside run
def process_scene(args_tuple):
    loader, dataset_name, scene_idx, scene_name, config, specific_image, dry_run, num_workers = args_tuple # Unpack num_workers

    # Only print if not multi-threaded
    if num_workers <= 1:
        tqdm.write(f"\nProcessing {dataset_name} scene: {scene_name} ({scene_idx+1})") # Note: total count isn't readily available here

    try: # Add try/except block for robustness in threads
        images_ann_dict_full = loader.list_images(dataset_name, scene_name)

        # Filter images if a specific image name is provided
        if specific_image:
            if specific_image in images_ann_dict_full:
                images_ann_dict = {specific_image: images_ann_dict_full[specific_image]}
                if num_workers <= 1:
                    tqdm.write(f"  - Specific image requested: Processing only '{specific_image}'.")
            else:
                if num_workers <= 1:
                    tqdm.write(f"  - Warning: Specific image '{specific_image}' not found in scene '{scene_name}'. Skipping scene.")
                return dataset_name, scene_name, None # Return None for stats if skipped

        # Limit images if dry_run is enabled AND no specific image was requested
        elif dry_run and len(images_ann_dict_full) > 5:
            images_ann_dict = dict(list(images_ann_dict_full.items())[:5])
            if num_workers <= 1:
                tqdm.write(f"  - Dry run enabled: Processing only the first 5 images out of {len(images_ann_dict_full)}.")
        else:
            images_ann_dict = images_ann_dict_full

        # Print total only if not dry run or specific image, and not multi-threaded
        if num_workers <= 1:
            if not dry_run and not specific_image:
                tqdm.write(f"  - Listed {len(images_ann_dict_full)} total images")
            elif dry_run and not specific_image: # Also print total if dry run
                tqdm.write(f"  - Listed {len(images_ann_dict_full)} total images")

        if not images_ann_dict:
            if num_workers <= 1:
                tqdm.write(f"Warning: No images found for scene {scene_name}. Skipping.")
            return dataset_name, scene_name, None # Return None for stats if skipped

        scene_stats = generate_and_save_annotations(
            loader,
            dataset_name,
            scene_name,
            images_ann_dict,
            config,
            num_workers
        )
        if num_workers <= 1:
            tqdm.write(f"Finished scene {scene_name}. Stats: {dict(scene_stats)}")
        return dataset_name, scene_name, scene_stats
    except Exception as e:
        # Always write errors, regardless of num_workers
        tqdm.write(f"Error processing scene {scene_name}: {e}")
        # Optionally re-raise or log the full traceback
        import traceback
        # Use tqdm.write for traceback as well
        tqdm.write(f"Traceback for error in scene {scene_name}:\n{traceback.format_exc()}")
        return dataset_name, scene_name, None # Indicate failure


def run(config, specific_scene=None, dry_run=False, specific_image=None, num_workers_arg=None): # Added num_workers_arg
    # Normal execution path
    print("Starting annotation generation with configuration:")
    print(yaml.dump(config, indent=2))

    # --- Determine Number of Workers ---
    num_workers = num_workers_arg # CLI argument takes precedence
    if num_workers is None:
        num_workers = config.get("data_generation", {}).get("num_workers")
    if num_workers is None:
        num_workers = 1 # Default to 1
        print(f"Number of workers not specified, defaulting to {num_workers}")
    print(f"Using {num_workers} worker threads.")


    # --- Dataset Loading ---
    dataset_list = config["data_loading"]["datasets"]
    
    if not dataset_list:
        print("Error: No valid datasets specified. Please include valid datasets in the config.")
        return
    
    # Create the loader instance
    loader = create_loader(config["data_loading"])
    print(f"Loader initialized.")


    # --- Statistics Initialization ---
    total_stats = defaultdict(lambda: defaultdict(int))
    overall_stats = defaultdict(int)
    generated_something = False
    progress_file_path = config["data_generation"].get("progress_file", "generation_progress.json")
    completed_scenes_map = defaultdict(list)

    # Load progress if file exists
    if os.path.exists(progress_file_path):
        with open(progress_file_path, 'r') as f:
            loaded_progress = json.load(f)
            if isinstance(loaded_progress, dict):
                completed_scenes_map.update(loaded_progress)
            else:
                print(f"Warning: Progress file {progress_file_path} has unexpected format. Starting fresh.")

    # --- Prepare Scene List ---
    print("\n--- Preparing Scene List ---")
    scene_list_all = list(loader.list_scenes(dataset_list))
    print(f"Found {len(scene_list_all)} total scenes across specified datasets.")

    scenes_to_process_info = []
    skipped_count = 0
    for idx, (dataset_name, scene_idx, scene_name) in enumerate(scene_list_all):
        # Apply filters
        if specific_scene and scene_name != specific_scene:
            skipped_count += 1
            continue
        if config.get("range") and idx not in config["range"]:
             skipped_count += 1
             continue
        if scene_name in completed_scenes_map.get(dataset_name, []):
             skipped_count += 1
             continue

        # If not skipped, add to list
        scenes_to_process_info.append((loader, dataset_name, idx, scene_name, config, specific_image, dry_run, num_workers))

    if skipped_count > 0:
        print(f"Skipped {skipped_count} scenes (due to filters: specific_scene, range, or already completed).")
    print(f"Processing {len(scenes_to_process_info)} scenes.")

    if not scenes_to_process_info:
        print("No scenes left to process based on filters.")
        # Skip the rest if nothing to process
        print("\n--- Generation Complete ---")
        print("No new annotations were generated.")
        return


    # --- Generation Loop (Parallelized) ---
    print("\n--- Processing Scenes ---")
    generated_something = False # Reset here, check results later

    with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
        # Submit tasks and store Future objects
        futures = [executor.submit(process_scene, args) for args in scenes_to_process_info]

        # Process results as they complete
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(scenes_to_process_info), desc="Processing Scenes"):
            try:
                result = future.result() # Get the result from the completed future
                # --- Aggregation logic using 'result' ---
                if result: # Check if result is not None (i.e., processing didn't fail or skip)
                    dataset_name, scene_name, scene_stats = result
                    if scene_stats is not None: # Check if stats were successfully generated
                        # Aggregate stats
                        for key, value in scene_stats.items():
                            if isinstance(value, (int, float)):
                                total_stats[dataset_name][key] += value
                                overall_stats[key] += value
                        generated_something = True
                        completed_scenes_map[dataset_name].append(scene_name)
                    # else: scene processing might have skipped internally or failed
                # --- End of aggregation logic ---
            except Exception as exc:
                # Handle exceptions raised within the process_scene function
                # Find the arguments that caused the exception for better logging (optional, requires mapping futures back to args)
                tqdm.write(f'\nError: A scene generation task generated an exception: {exc}')
                # Optionally log the full traceback
                # import traceback
                # tqdm.write(f"Traceback:\n{traceback.format_exc()}")

    # --- Final Statistics and Cleanup ---
    print("\n--- Generation Complete ---")
    if generated_something:
        print("\n--- Overall Statistics ---")
        print(json.dumps(overall_stats, indent=4))

        print("\n--- Per-Dataset Statistics ---")
        print(json.dumps(total_stats, indent=4))

        with open(progress_file_path, 'w') as f:
            json.dump(completed_scenes_map, f, indent=4)
        print(f"Final progress saved to {progress_file_path}")

        stats_file_path = config["data_generation"].get("stats_file", "generation_stats.json")
        final_stats_data = {
            "overall_stats": overall_stats,
            "per_dataset_stats": total_stats
        }
        with open(stats_file_path, 'w') as f:
            json.dump(final_stats_data, f, indent=4)
        print(f"Final statistics saved to {stats_file_path}")
    else:
        print("No new annotations were generated.")

    return

if __name__ == "__main__":
    args = parse_args()
    config = load_config(args.config)
    if args.range:
        config["range"] = args.range
    # Pass num_workers from args
    run(config, specific_scene=args.scene, dry_run=args.dry_run, specific_image=args.image, num_workers_arg=args.num_workers) 