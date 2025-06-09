"""
RoboSpatial QA Generator

This module generates question-answer pairs for spatial relationships in images, supporting both
pairwise and unary spatial contexts. It processes annotation files containing spatial relationship
data and generates corresponding QA pairs that can be used for training or validation datasets.

The module supports two output modes:
1. JSON: Saves QA pairs locally in JSON format
2. Hugging Face: Uploads the dataset to Hugging Face Hub with train/validation splits

Key Features:
- Generates QA pairs for pairwise spatial relationships (e.g., "Is object A to the left of object B?")
- Generates QA pairs for unary spatial contexts (e.g., "Where is the empty space in front of object A?")
- Supports multiple reference frames (camera-centric, world-centric, object-centric)
- Implements stratified sampling for balanced dataset creation
- Handles image loading and validation
- Supports both local JSON output and Hugging Face Hub upload

Usage:
    python qa_generator.py <input_dir> --output-mode [json|huggingface] --output-path <path>
    
    Example for JSON output:
        python qa_generator.py /path/to/data --output-mode json --output-path output.json
        
    Example for Hugging Face upload:
        python qa_generator.py /path/to/data --output-mode huggingface --output-path username/datasetname
        --num-samples 10000 --num-val-samples 1000
"""

import json
import os
import argparse
import random
import glob
from PIL import Image
from PIL import UnidentifiedImageError
import traceback
import collections

try:
    import datasets
    from datasets import Image as HFImage
    from datasets import Dataset, DatasetDict, Features, Value
    DATASETS_AVAILABLE = True
except ImportError:
    DATASETS_AVAILABLE = False
    HFImage = None
    datasets = None
    Dataset = None
    DatasetDict = None
    Features = None
    Value = None
    print("Warning: 'datasets' library not found. Hugging Face upload mode will not be available.")
    print("Install it using: pip install datasets huggingface_hub")

positional_opposites = {
    "left": "right", "right": "left",
    "infront": "behind", "behind": "infront",
    "above": "below", "below": "above"
}

frame_descriptors = {
    "camera_centric": "from the camera's perspective",
    "world_centric": "from the world's perspective",
    "object_centric": "from the second object's perspective"
}

compat_descriptors = {
    "objectcentric": "from the second object's perspective",
    "cameracentric": "from the camera's perspective",
    "worldcentric": "from the world's perspective"
}

unary_frame_descriptors = {
    "cameracentric": "from the camera's perspective",
    "worldcentric": "from the world's perspective",
    "objectcentric": "from the object's perspective"
}

def _generate_prompts_for_relation(relation):
    """
    Generator yielding (prompt_text, answer_string, qa_type) tuples for a single pairwise relation dict.
    Assumes relation dict structure is valid.
    """
    obj_a_name = relation['pair'][0]
    obj_b_name = relation['pair'][1]

    if 'spatial_configuration' in relation:
        config = relation['spatial_configuration']
        for frame, frame_dict in config.items():
            descriptor = frame_descriptors[frame]
            processed = set()
            for rel1, rel2 in positional_opposites.items():
                if rel1 in processed or rel2 in processed:
                    continue
                val1 = frame_dict.get(rel1)
                val2 = frame_dict.get(rel2)

                rel1_display = "in front" if rel1 == "infront" else rel1
                rel2_display = "in front" if rel2 == "infront" else rel2

                if val1 is True:
                    prompt1 = f"<image>Is the {obj_a_name} {rel1_display} of the {obj_b_name} {descriptor}? Final answer should be yes or no."
                    yield (prompt1, 'Yes', 'pairwise_positional')
                    prompt2 = f"<image>Is the {obj_a_name} {rel2_display} of the {obj_b_name} {descriptor}? Final answer should be yes or no."
                    yield (prompt2, 'No', 'pairwise_positional')
                    processed.add(rel1)
                    processed.add(rel2)
                elif val1 is False and val2 is True:
                    prompt1 = f"<image>Is the {obj_a_name} {rel2_display} of the {obj_b_name} {descriptor}? Final answer should be yes or no."
                    yield (prompt1, 'Yes', 'pairwise_positional')
                    prompt2 = f"<image>Is the {obj_a_name} {rel1_display} of the {obj_b_name} {descriptor}? Final answer should be yes or no."
                    yield (prompt2, 'No', 'pairwise_positional')
                    processed.add(rel1)
                    processed.add(rel2)

    if 'spatial_compatibility' in relation:
        compat = relation['spatial_compatibility']
        for frame, frame_dict in compat.items():
            descriptor = compat_descriptors[frame]
            for key, value in frame_dict.items():
                relation_text = key.replace('_', ' ')
                relation_text_display = relation_text.replace("infront", "in front")

                if value is True:
                    prompt = f"<image>Can the {obj_a_name} fit {relation_text_display} the {obj_b_name} {descriptor}? Final answer should be yes or no."
                    yield (prompt, 'Yes', 'pairwise_compatibility')
                elif value is False:
                    prompt = f"<image>Can the {obj_a_name} fit {relation_text_display} the {obj_b_name} {descriptor}? Final answer should be yes or no."
                    yield (prompt, 'No', 'pairwise_compatibility')

def _generate_prompts_for_spatial_context(relation, image_width, image_height):
    """
    Generator yielding (prompt_text, answer_string) tuples for a single unary relation dict,
    focusing on spatial context (empty space). The answer is a string formatted as a list of tuples
    containing normalized pixel coordinates (relative to image width and height, range [0, 1]).
    """
    if image_width <= 0 or image_height <= 0:
        return

    obj_name = relation.get('name', 'the object')
    obj_name_cleaned = obj_name.rstrip('.')

    if 'point_space_2d' in relation:
        point_space = relation['point_space_2d']
        for frame, frame_dict in point_space.items():
            descriptor = unary_frame_descriptors[frame]
            for direction in ["infront", "behind", "left", "right"]:
                points = frame_dict.get(direction, [])
                if not points:
                    continue

                answer_points_list = points
                normalized_points = []
                for p in answer_points_list:
                    if len(p) == 2:
                        x_norm = round(p[0] / image_width, 3)
                        y_norm = round(p[1] / image_height, 3)
                        x_norm = max(0.0, min(1.0, x_norm))
                        y_norm = max(0.0, min(1.0, y_norm))
                        normalized_points.append((x_norm, y_norm))

                if not normalized_points:
                    continue

                if len(normalized_points) < 3:
                    continue

                direction_display = "in front" if direction == "infront" else direction

                prompt = (
                    f"<image>In the image, there is the {obj_name_cleaned}. "
                    f"Pinpoint several points within the vacant space situated {direction_display} of the {obj_name_cleaned} {descriptor}. "
                    f"Your final answer should be formatted as a list of tuples, i.e. [(x1, y1), ...], where each tuple contains the x and y coordinates of a point satisfying the conditions above. The coordinates should be between 0 and 1, indicating the normalized pixel locations of the points."
                )

                answer_string = "[" + ", ".join(f"({p[0]:.3f}, {p[1]:.3f})" for p in normalized_points) + "]"
                yield (prompt, answer_string, 'unary_spatial_context')

def _collect_all_qa_pairs(data, image_path):
    """
    Collects all possible QA pairs from the annotation data for a single image.
    Requires the image_path to be included in the output dict.
    Returns a list of dictionaries.
    """
    all_qa_pairs = []
    item_base = {'image_path': image_path}

    try:
        with Image.open(image_path) as img:
            image_width, image_height = img.size
    except FileNotFoundError:
        print(f"\nWarning: Image file not found at {image_path} during QA pair collection. Skipping unary prompts for this image.")
        image_width, image_height = 0, 0
    except UnidentifiedImageError:
        print(f"\nWarning: Cannot identify image file (possibly corrupt) at {image_path}. Skipping unary prompts for this image.")
        image_width, image_height = 0, 0
    except Exception as e:
        print(f"\nWarning: An unexpected error occurred loading image {image_path}: {e}. Skipping unary prompts for this image.")
        image_width, image_height = 0, 0

    if 'spatial_relationships' in data and 'pairwise_relations' in data['spatial_relationships']:
        for relation in data['spatial_relationships']['pairwise_relations']:
            for prompt, answer, qa_type in _generate_prompts_for_relation(relation):
                item = item_base.copy()
                item.update({'question': prompt, 'answer': answer, 'qa_type': qa_type})
                all_qa_pairs.append(item)

    if image_width > 0 and image_height > 0 and 'spatial_relationships' in data and 'unary_relations' in data['spatial_relationships']:
        for relation in data['spatial_relationships']['unary_relations']:
            for prompt, answer, qa_type in _generate_prompts_for_spatial_context(relation, image_width, image_height):
                item = item_base.copy()
                item.update({'question': prompt, 'answer': answer, 'qa_type': qa_type})
                all_qa_pairs.append(item)

    return all_qa_pairs

def _calculate_and_print_stats(qa_list, description):
    """
    Calculates and prints statistics for the QA types in the given list.
    """
    if not qa_list:
        print(f"No QA pairs found for {description}. Cannot calculate statistics.")
        return

    qa_type_counts = collections.Counter(item.get('qa_type', 'Unknown') for item in qa_list)

    print(f"\n--- Statistics for {description} ({len(qa_list)} pairs total) ---")
    total_counted = 0
    for qa_type, count in qa_type_counts.items():
        print(f"  - {qa_type}: {count}")
        total_counted += count
    print(f"  - Total Counted: {total_counted}")
    print("---")

def generate_qa_for_json(input_dir):
    """
    Generates QA pairs suitable for JSON output by processing all annotation
    files found in the specified directory structure.
    Includes pairwise relations and unary spatial context.
    Returns a master list of dictionaries.
    """
    master_qa_list = []
    search_pattern = os.path.join(input_dir, '**', '*.json')
    annotation_files = glob.glob(search_pattern, recursive=True)

    if not annotation_files:
        print(f"Warning: No JSON files found in directory: {input_dir}")
        return []

    print(f"Found {len(annotation_files)} JSON files. Processing...")

    for i, annotation_path in enumerate(annotation_files):
        print(f"Processing file {i+1}/{len(annotation_files)}: {annotation_path}", end='\r')
        try:
            with open(annotation_path, 'r') as f:
                data = json.load(f)

            json_dir = os.path.dirname(annotation_path)
            relative_image_path = data['image_path']
            image_full_path = os.path.abspath(os.path.join(json_dir, relative_image_path))

            if not os.path.exists(image_full_path):
                print(f"\nWarning: Image file specified in {annotation_path} not found at {image_full_path}. Skipping this file.")
                continue

            qa_pairs_for_file = _collect_all_qa_pairs(data, image_path=image_full_path)
            master_qa_list.extend(qa_pairs_for_file)

        except FileNotFoundError as e:
            print(f"\nError processing {annotation_path}: {e}")
        except json.JSONDecodeError as e:
            print(f"\nError decoding JSON in {annotation_path}: {e}")
        except KeyError as e:
            print(f"\nMissing key {e} in {annotation_path}")
        except Exception as e:
            print(f"\nAn unexpected error occurred processing {annotation_path}: {e}")
            traceback.print_exc()

    print("\nFinished processing all files.")
    return master_qa_list

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Generate QA pairs from annotations and save/upload.")
    parser.add_argument("input_dir", help="Path to the root directory containing the annotation files.")
    parser.add_argument("--output-mode", choices=['json', 'huggingface'], required=True,
                        help="Output mode: 'json' to save locally, 'huggingface' to upload.")
    parser.add_argument("--output-path", required=True,
                        help="Output file path for 'json' mode, or Hugging Face Hub repo ID (e.g., 'username/datasetname') for 'huggingface' mode.")
    parser.add_argument("--num-samples", type=int, default=10000,
                        help="Number of QA pairs to sample for the training split (default: 10000).")
    parser.add_argument("--validation-split-ratio", type=float, default=0.0,
                        help="Ratio of total generated data to reserve for validation before sampling (e.g., 0.1 for 10%%, default: 0.0).")
    parser.add_argument("--num-val-samples", type=int, default=1000,
                        help="Number of QA pairs to sample for the validation split (default: 1000, requires --validation-split-ratio > 0).")
    parser.add_argument("--seed", type=int, default=55,
                        help="Optional random seed for reproducible sampling.")

    args = parser.parse_args()

    if args.seed is not None:
        print(f"Setting random seed to: {args.seed}")
        random.seed(args.seed)

    print(f"Starting QA generation from input directory: {args.input_dir}")

    master_qa_list = generate_qa_for_json(args.input_dir)
    total_generated = len(master_qa_list)
    print(f"\nGenerated a total of {total_generated} potential QA pairs from all files.")

    _calculate_and_print_stats(master_qa_list, "Total Generated")

    if total_generated == 0:
        print("No QA pairs generated overall. Exiting.")
        exit(0)

    if args.output_mode == 'json':
        print(f"Saving aggregated QA pairs to {args.output_path}...")
        with open(args.output_path, 'w') as f:
            json.dump(master_qa_list, f, indent=4)
        print(f"Successfully saved aggregated QA pairs to {args.output_path}")

    elif args.output_mode == 'huggingface':
        if not DATASETS_AVAILABLE:
            print("Error: The 'datasets' library is required for Hugging Face upload.")
            print("Please install it: pip install datasets huggingface_hub")
            exit(1)

        if args.num_samples <= 0:
            print("Error: --num-samples must be a positive integer.")
            exit(1)

        if args.validation_split_ratio < 0 or args.validation_split_ratio >= 1:
            print("Warning: --validation-split-ratio is ignored for stratified sampling. Use --num-val-samples directly.")

        if args.num_val_samples < 0:
            print("Error: --num-val-samples cannot be negative.")
            exit(1)
        create_validation_split = args.num_val_samples > 0

        print("\nPreparing data for Hugging Face upload using STRATIFIED SAMPLING...")

        grouped_qa = collections.defaultdict(list)
        for item in master_qa_list:
            grouped_qa[item.get('qa_type', 'Unknown')].append(item)

        qa_types = list(grouped_qa.keys())
        num_types = len(qa_types)
        print(f"Found {num_types} QA types: {', '.join(qa_types)}")
        if num_types == 0:
            print("Error: No QA types found. Cannot perform stratified sampling.")
            exit(1)

        train_sampled = []
        remaining_qa = collections.defaultdict(list)
        target_train_per_type = args.num_samples // num_types
        print(f"Targeting ~{target_train_per_type} samples per type for training (total requested: {args.num_samples}).")

        if target_train_per_type == 0 and args.num_samples > 0:
            print("Warning: Requested number of training samples is less than the number of QA types. Sampling will be uneven.")

        actual_total_train = 0
        for qa_type in qa_types:
            type_list = grouped_qa[qa_type]
            random.shuffle(type_list)
            num_available = len(type_list)
            num_to_sample = min(target_train_per_type, num_available)

            if num_to_sample < target_train_per_type:
                print(f"  Warning: Only {num_available} samples available for type '{qa_type}' (target: {target_train_per_type}). Using {num_available}.")

            sampled_for_type = type_list[:num_to_sample]
            train_sampled.extend(sampled_for_type)
            remaining_qa[qa_type].extend(type_list[num_to_sample:])
            actual_total_train += num_to_sample

        remainder_train = args.num_samples - actual_total_train
        if remainder_train > 0 and remainder_train < num_types:
            print(f"Distributing {remainder_train} remainder training samples...")
            eligible_types_for_remainder = [t for t in qa_types if len(remaining_qa[t]) > 0]
            random.shuffle(eligible_types_for_remainder)
            for i in range(min(remainder_train, len(eligible_types_for_remainder))):
                selected_type = eligible_types_for_remainder[i]
                item_to_add = remaining_qa[selected_type].pop(0)
                train_sampled.append(item_to_add)
                actual_total_train += 1

        random.shuffle(train_sampled)
        print(f"Sampled {len(train_sampled)} training pairs.")
        _calculate_and_print_stats(train_sampled, "Sampled Train (Stratified)")

        val_sampled = []
        if create_validation_split:
            target_val_per_type = args.num_val_samples // num_types
            print(f"\nTargeting ~{target_val_per_type} samples per type for validation (total requested: {args.num_val_samples}).")
            if target_val_per_type == 0 and args.num_val_samples > 0:
                print("Warning: Requested number of validation samples is less than the number of QA types. Sampling will be uneven.")

            actual_total_val = 0
            for qa_type in qa_types:
                type_list = remaining_qa[qa_type]
                num_available = len(type_list)
                num_to_sample = min(target_val_per_type, num_available)

                if num_to_sample < target_val_per_type:
                    print(f"  Warning: Only {num_available} remaining samples available for validation for type '{qa_type}' (target: {target_val_per_type}). Using {num_available}.")

                sampled_for_type = type_list[:num_to_sample]
                val_sampled.extend(sampled_for_type)
                remaining_qa[qa_type] = type_list[num_to_sample:]
                actual_total_val += num_to_sample

            remainder_val = args.num_val_samples - actual_total_val
            if remainder_val > 0 and remainder_val < num_types:
                print(f"Distributing {remainder_val} remainder validation samples...")
                eligible_types_for_remainder_val = [t for t in qa_types if len(remaining_qa[t]) > 0]
                random.shuffle(eligible_types_for_remainder_val)
                for i in range(min(remainder_val, len(eligible_types_for_remainder_val))):
                    selected_type = eligible_types_for_remainder_val[i]
                    item_to_add = remaining_qa[selected_type].pop(0)
                    val_sampled.append(item_to_add)
                    actual_total_val += 1

            random.shuffle(val_sampled)
            print(f"Sampled {len(val_sampled)} validation pairs.")
            _calculate_and_print_stats(val_sampled, "Sampled Validation (Stratified)")
        else:
            print("\nNo validation samples requested (--num-val-samples=0).")

        local_json_base = args.output_path.replace("/", "_")
        train_json_path = f"{local_json_base}_train.json"
        val_json_path = f"{local_json_base}_validation.json"

        print(f"\nSaving sampled training data locally to: {train_json_path}")
        with open(train_json_path, 'w') as f:
            json.dump(train_sampled, f, indent=4)

        if val_sampled:
            print(f"Saving sampled validation data locally to: {val_json_path}")
            with open(val_json_path, 'w') as f:
                json.dump(val_sampled, f, indent=4)
        print("Finished saving local JSON files.\n")

        final_features = Features({
            'images': [HFImage()],
            'question': Value('string'),
            'answer': Value('string'),
            'qa_type': Value('string')
        })

        dataset_dict_data = {}
        if train_sampled:
            ds_train = Dataset.from_list(train_sampled)
            ds_train = ds_train.map(
                lambda example: {'images': [example['image_path']]},
                remove_columns=['image_path']
            )
            ds_train = ds_train.cast(final_features)
            dataset_dict_data['train'] = ds_train
        else:
            print("Warning: No training samples selected. Training split will be empty.")

        if val_sampled:
            ds_val = Dataset.from_list(val_sampled)
            ds_val = ds_val.map(
                lambda example: {'images': [example['image_path']]},
                remove_columns=['image_path']
            )
            ds_val = ds_val.cast(final_features)
            dataset_dict_data['validation'] = ds_val

        if not dataset_dict_data:
            print("Error: No data selected for either train or validation split. Cannot create dataset.")
            exit(1)

        hf_dataset_dict = DatasetDict(dataset_dict_data)
        print(f"Created DatasetDict with splits: {list(hf_dataset_dict.keys())}")
        print(f"Train samples: {len(hf_dataset_dict.get('train', []))}, Validation samples: {len(hf_dataset_dict.get('validation', []))}")

        print(f"Attempting to push dataset splits to Hugging Face Hub: {args.output_path}")
        print("Ensure you are logged in: run `huggingface-cli login` if needed.")
        hf_dataset_dict.push_to_hub(args.output_path)
        print(f"Successfully uploaded dataset splits to {args.output_path}")
