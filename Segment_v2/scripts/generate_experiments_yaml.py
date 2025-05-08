import yaml
import os
import argparse

# --- Generator Function ---

def generate_experiment_yaml(
    dataset_configurations: dict, # Expecting {dataset_name: {... 'mask_subdirs': {mask_type: {split: subdir, 'type_overrides': {...}}}}}
    model_arch_map: dict,
    experiments_to_generate: dict,
    base_config_path: str,
    results_csv_name: str,
    output_yaml_path: str
):
    """
    Generates an experiments YAML file based on specified configurations.

    Args:
        dataset_configurations: Dictionary defining dataset properties,
                                expecting mask_subdirs to be {mask_type: {split: subdir, 'type_overrides': {...}}}.
        model_arch_map: Dictionary mapping architecture names to encoder names.
        experiments_to_generate: Dictionary specifying the desired experiment combinations
                                 as {dataset_name: {mask_type: {arch: [encoders]}}}.
        base_config_path: Path to the base configuration YAML file.
        results_csv_name: Name for the output results CSV file.
        output_yaml_path: Path where the generated experiments YAML file will be saved.
    """
    print(f"Generating experiments YAML to: {output_yaml_path}")

    generated_experiments_list = []

    # Iterate through the desired combinations specified in experiments_to_generate
    for dataset_name, mask_types_filter in experiments_to_generate.items():
        # Look up parameters for this dataset
        if dataset_name not in dataset_configurations:
            print(f"Warning: Dataset '{dataset_name}' specified in experiments_to_generate but not found in dataset_configurations. Skipping.")
            continue

        dataset_params = dataset_configurations[dataset_name]
        base_dir = dataset_params['base_dir']
        in_channels = dataset_params['in_channels']
        image_subdirs = dataset_params.get('image_subdirs', {}) # Use .get for safety
        # Get the overall mask_subdirs definition for this dataset (new structure with type_overrides)
        dataset_mask_subdirs_definition = dataset_params.get('mask_subdirs', {})
        other_dataset_overrides = dataset_params.get('other_overrides', {})

        # Validate that required image subdirs exist for this dataset
        required_image_splits = ['train', 'val', 'test']
        if not all(split in image_subdirs for split in required_image_splits):
             print(f"Warning: Dataset '{dataset_name}' is missing image subdirectory definition for one or more required splits ({required_image_splits}) in its image_subdirs definition. Skipping combinations for this dataset.")
             continue


        # Iterate through the mask types allowed by the filter for this dataset
        for mask_type, arch_filter in mask_types_filter.items():
             # Validate mask_type exists in the dataset's mask_subdirs definition
             if mask_type not in dataset_mask_subdirs_definition:
                 print(f"Warning: Mask type '{mask_type}' specified in experiments_to_generate for dataset '{dataset_name}' but not found in its mask_subdirs definition. Skipping combinations for this mask type.")
                 continue

             # --- Get the split subdirectories and type overrides for this specific mask type ---
             mask_type_definition = dataset_mask_subdirs_definition[mask_type] # This is the dict like {'train': ..., 'val': ..., ..., 'type_overrides': {...}}

             # Get split subdirs (ensure they exist)
             mask_split_subdirs = {}
             required_mask_splits = ['train', 'val', 'test']
             for split in required_mask_splits:
                 if split not in mask_type_definition:
                      print(f"Warning: Mask type '{mask_type}' in dataset '{dataset_name}' is missing subdirectory definition for split '{split}' in its mask_subdirs definition. Skipping combinations for this mask type.")
                      # Skip the whole mask type if any split is missing
                      mask_split_subdirs = None
                      break
                 mask_split_subdirs[split] = mask_type_definition[split]

             if mask_split_subdirs is None: # Skip if any required split subdir was missing
                  continue

             # Get type-specific overrides (num_classes, IoU_index, class_names etc.)
             type_overrides = mask_type_definition.get('type_overrides', {})


             # Iterate through architectures allowed by the filter for this mask type
             for arch, encoders_filter in arch_filter.items():
                 # Validate arch exists in model map
                 if arch not in model_arch_map:
                     print(f"Warning: Architecture '{arch}' specified in experiments_to_generate for mask_type '{mask_type}' but not found in model_arch_map. Skipping combinations for this architecture.")
                     continue

                 # Determine encoders to include for this arch (from filter or all from map)
                 encoders_to_include = encoders_filter if isinstance(encoders_filter, list) and encoders_filter else model_arch_map.get(arch, [])

                 for encoder in encoders_to_include:
                     # Validate encoder exists for the architecture in the base map
                     if encoder not in model_arch_map.get(arch, []):
                         print(f"Warning: Encoder '{encoder}' specified for arch '{arch}' in experiments_to_generate, but not found in model_arch_map. Skipping this specific combination.")
                         continue

                     # --- Construct Experiment Dictionary ---
                     experiment_name = f"{dataset_name}/{mask_type}/{arch}/{encoder}"

                     # Start with general dataset overrides, then add type-specific, then arch/encoder
                     overrides = {
                         'arch': arch,
                         'encoder': encoder,
                         'dataset_dir': base_dir,
                         'in_channels': in_channels,
                         # Add optional dataset-level overrides (e.g., mean/std if constant per dataset type)
                         **other_dataset_overrides,
                         # Add optional mask-type-specific overrides (e.g., num_classes, IoU_index, class_names)
                         **type_overrides, # <-- Add type-specific overrides here (now includes class_names)
                         # Add split-specific relative paths (these should likely override anything before)
                         'train_images_dir': image_subdirs['train'],
                         'train_masks_dir': mask_split_subdirs['train'],
                         'val_images_dir': image_subdirs['val'],
                         'val_masks_dir': mask_split_subdirs['val'],
                         'test_images_dir': image_subdirs['test'],
                         'test_masks_dir': mask_split_subdirs['test'],
                     }

                     # Add the experiment name to overrides as well (used by logger)
                     overrides['experiment_name'] = experiment_name


                     generated_experiments_list.append({
                         'name': experiment_name,
                         'dataset_name': f"{dataset_name}_{mask_type}", # Add mask type to dataset_name field
                         'overrides': overrides
                     })

    # --- Final YAML Structure ---
    yaml_structure = {
        'base_config': base_config_path,
        'results_csv': results_csv_name,
        'experiments': generated_experiments_list
    }

    # --- Write to YAML file ---
    try:
        with open(output_yaml_path, 'w') as outfile:
            yaml.dump(yaml_structure, outfile, default_flow_style=False, sort_keys=False)

        print(f"Successfully generated {len(generated_experiments_list)} experiments to {output_yaml_path}")

    except Exception as e:
        print(f"Error writing YAML file: {e}")
        # You might want to raise the exception after printing
        # raise


# --- Example Usage (when run as a script) ---

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate experiments YAML file.')
    parser.add_argument('--output', type=str, default='./config/experiments.yaml',
                        help='Path to save the generated experiments YAML file.')
    args = parser.parse_args()


    # --- Define Example Configurations (Hardcoded for standalone script) ---
    # Replace with your actual datasets, models, and desired combinations

    # 1. Define dataset configurations (***UPDATED STRUCTURE with class_names in type_overrides***)
    dataset_configurations_example = {
        'RGB': {
            'base_dir': '/root/data_temp/RGB',
            'in_channels': 3,
            'image_subdirs': {'train': 'train/data', 'val': 'val/data', 'test': 'test/data'},
            'mask_subdirs': { # <--- UPDATED STRUCTURE: {mask_type: {split: subdir, 'type_overrides': {...}}}
                'severity': {
                    'train': 'train/label_severity', 'val': 'val/label_severity', 'test': 'test/label_severity',
                    'type_overrides': {
                        'num_classes': 7,
                        'IoU_index': [3, 4, 6], # Example indices for severity
                        'class_names': ['road', 'sugarcane', 'rice_normal', 'rice_severe', 'rice_mild', 'weed', 'abnormal'] # <-- 添加 class_names
                    },
                },
                'none_severity': {
                    'train': 'train/label_none_severity', 'val': 'val/label_none_severity', 'test': 'test/label_none_severity',
                     'type_overrides': {
                        'num_classes': 6,
                        'IoU_index': [3, 5], # Example indices for none_severity
                        'class_names': ['road', 'sugarcane', 'rice_normal', 'rice_lodging', 'weed', 'abnormal'] # <-- 添加 class_names
                    },
                },
            },
             # Example: If mean/std are constant per dataset type (regardless of mask type), add here
            # 'other_overrides': {'mean': [0.485, 0.456, 0.406], 'std': [0.229, 0.224, 0.225]},
        },
        'ALL': {
            'base_dir': 'E:/ALL_Datasets/ALL',
            'in_channels': 55, # Note: Previous example had 55 channels for ALL, double check this
            'image_subdirs': {'train': 'train/data', 'val': 'val/data', 'test': 'test/data'},
            'mask_subdirs': { # <--- UPDATED STRUCTURE
                'severity': {
                    'train': 'train/label_severity', 'val': 'val/label_severity', 'test': 'test/label_severity',
                    'type_overrides': {
                        'num_classes': 7,
                        'IoU_index': [3, 4, 6],
                        'class_names': ['road', 'sugarcane', 'rice_normal', 'rice_severe', 'rice_mild', 'weed', 'abnormal']
                    },
                },
                'none_severity': {
                    'train': 'train/label_none_severity', 'val': 'val/label_none_severity', 'test': 'test/label_none_severity',
                     'type_overrides': {
                         'num_classes': 6,
                         'IoU_index': [3, 5],
                         'class_names': ['road', 'sugarcane', 'rice_normal', 'rice_lodging', 'weed', 'abnormal']
                     },
                },
            },
             # 'other_overrides': {'mean': [...], 'std': [...]},
        },
        'CHM': {
            'base_dir': '/root/data_temp/CHM',
            'in_channels': 1,
            'image_subdirs': {'train': 'train/data', 'val': 'val/data', 'test': 'test/data'},
            'mask_subdirs': { # <--- UPDATED STRUCTURE
                'severity': {
                    'train': 'train/label_severity', 'val': 'val/label_severity', 'test': 'test/label_severity',
                     'type_overrides': {
                         'num_classes': 7,
                         'IoU_index': [3, 4, 6],
                         'class_names': ['road', 'sugarcane', 'rice_normal', 'rice_severe', 'rice_mild', 'weed', 'abnormal']
                     },
                },
                'none_severity': {
                    'train': 'train/label_none_severity', 'val': 'val/label_none_severity', 'test': 'test/label_none_severity',
                     'type_overrides': {
                         'num_classes': 6,
                         'IoU_index': [3, 5],
                         'class_names': ['road', 'sugarcane', 'rice_normal', 'rice_lodging', 'weed', 'abnormal']
                     },
                },
            },
            # 'other_overrides': {'mean': [...], 'std': [...]},
        },
        # Add other datasets...
    }

    # 2. Define model architectures and their corresponding encoders
    model_arch_map_example = {
        'unet': ['resnet50'],
        'segformer': ['mit_b3'],
        "deeplabv3plus": ["resnet50"],
    }

    # 3. Define which specific combinations of (Dataset, Mask Type, Architecture, Encoder) you want to generate.
    # This structure acts as a filter and driver for the loops.
    # Structure: {dataset_name: {mask_type: {architecture_name: [encoder_names_list]}}}
    # Use `[]` or `None` for the encoder_names_list to include all encoders defined in `model_arch_map_example` for that architecture.
    # Omit a key (dataset, mask_type, arch) to exclude that specific combination subtree.
    experiments_to_generate_example = {
        # 'RGB': {
        #     'severity': { # For RGB dataset using 'severity' masks
        #         'unet': ['resnet50'],        # Generate RGB/severity/unet/resnet18
        #         'segformer': ['mit_b3'],     # Generate RGB/severity/segformer/mit_b0
        #         "deeplabv3plus": ["resnet50"],
        #     },
        #     'none_severity': { # For RGB dataset using 'none_severity' masks
        #         'unet': ['resnet50'],        # Generate RGB/severity/unet/resnet18
        #         'segformer': ['mit_b3'],     # Generate RGB/severity/segformer/mit_b0
        #         "deeplabv3plus": ["resnet50"],
        #     },
        # },
        # 'CHM': { # For CHM dataset
        #      'severity': {
        #         'unet': ['resnet50'],        # Generate RGB/severity/unet/resnet18
        #         'segformer': ['mit_b3'],     # Generate RGB/severity/segformer/mit_b0
        #         "deeplabv3plus": ["resnet50"],
        #     },
        #     'none_severity': {
        #         'unet': ['resnet50'],        # Generate RGB/severity/unet/resnet18
        #         'segformer': ['mit_b3'],     # Generate RGB/severity/segformer/mit_b0
        #         "deeplabv3plus": ["resnet50"],
        #     },
        # },
        'ALL': { # For ALL dataset
             'severity': {
                'unet': ['resnet50'],        # Generate RGB/severity/unet/resnet18
                'segformer': ['mit_b3'],     # Generate RGB/severity/segformer/mit_b0
                "deeplabv3plus": ["resnet50"],
            },
            'none_severity': {
                'unet': ['resnet50'],        # Generate RGB/severity/unet/resnet18
                'segformer': ['mit_b3'],     # Generate RGB/severity/segformer/mit_b0
                "deeplabv3plus": ["resnet50"],
            },
        },
        # Add definitions for other datasets/mask types/models
    }

    # Other base parameters
    base_config_path_example = './config/fusedRGB.yaml'
    results_csv_name_example = 'experiment_results.csv'


    # --- Call the function with example configurations ---
    generate_experiment_yaml(
        dataset_configurations=dataset_configurations_example,
        model_arch_map=model_arch_map_example,
        experiments_to_generate=experiments_to_generate_example,
        base_config_path=base_config_path_example,
        results_csv_name=results_csv_name_example,
        output_yaml_path=args.output # Use the output path from arguments
    )