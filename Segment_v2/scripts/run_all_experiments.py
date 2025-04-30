import os
import yaml
import csv
import datetime
import time
import torch # For GPU memory
import pandas as pd # To read overall.csv
import sys
from pytorch_lightning.callbacks import ModelCheckpoint

# Add the parent directory of 'src' to sys.path if running from project root
# sys.path.append('.') # Uncomment if needed based on your project structure

# Assume these functions are created in train.py and predict.py
# by moving the core logic out of if __name__ == '__main__':
try:
    # Modify train.py: Wrap core logic in 'def run_training(config): ... return trainer_instance'
    from auto_train import run_training
    # Modify predict.py: Wrap core logic in 'def run_prediction(config, checkpoint_path): ... return predictor_log_dir'
    from auto_predict import run_prediction
    # Import SegmentationModel to potentially get input size for GPU memory calculation
    # from src.models.segmentation import SegmentationModel # Needed if you want to calculate FLOPs/Params too
except ImportError as e:
    print(f"Import Error: Could not import training or prediction functions. {e}")
    print("Please ensure 'train.py' and 'predict.py' are in a directory included in sys.path and")
    print("that they contain 'run_training(config)' and 'run_prediction(config, checkpoint_path)' functions respectively.")
    sys.exit(1)


# --- Helper Functions ---

def update_config(base_config, overrides):
    """Updates a dictionary (config) with values from another dictionary (overrides)."""
    config = base_config.copy()
    for key, value in overrides.items():
        # Simple key override is assumed here.
        # For nested structures, a more sophisticated update might be needed.
        config[key] = value
    return config

def get_peak_gpu_memory_after_run():
    """
    Gets the peak allocated GPU memory in MB after a process has run.
    Note: This might not capture the absolute peak during the run if memory was released.
    A more accurate method would involve integrating into the training loop or using GPUStatsMonitor.
    """
    if torch.cuda.is_available():
        # Returns peak memory in bytes for the current device
        # Use max_memory_allocated(device=None) for the current device
        peak_bytes = torch.cuda.max_memory_allocated()
        return peak_bytes / (1024 * 1024) # Convert to MB
    return 0.0 # Return 0 if no GPU available


def read_overall_metrics(log_dir):
    """Reads mIoU and OA from the overall.csv file in the log directory."""
    overall_csv_path = os.path.join(log_dir, "overall.csv")
    test_miou = 'N/A'
    test_oa = 'N/A'
    if os.path.exists(overall_csv_path):
        try:
            df_overall = pd.read_csv(overall_csv_path)
            if not df_overall.empty:
                test_miou = df_overall['mIoU'].iloc[0]
                test_oa = df_overall['OA'].iloc[0]
            else:
                print(f"Warning: {overall_csv_path} is empty.")
        except Exception as csv_e:
            print(f"Error reading {overall_csv_path}: {csv_e}")
    else:
         print(f"Warning: {overall_csv_path} not found.")
    return test_miou, test_oa


# --- Main Automation Script ---

def run_all_experiments(experiment_config_path="./config/experiments.yaml"):
    """Runs all experiments defined in the config file and logs results."""

    # Load the experiment configurations
    if not os.path.exists(experiment_config_path):
        print(f"Error: Experiment configuration file not found at {experiment_config_path}")
        sys.exit(1)

    with open(experiment_config_path, encoding='utf-8') as f:
        experiment_configs = yaml.safe_load(f)

    base_config_path = experiment_configs.get('base_config')
    results_csv_path = experiment_configs.get('results_csv', 'experiment_results.csv') # Default name
    experiments = experiment_configs.get('experiments', [])

    if not base_config_path or not os.path.exists(base_config_path):
         print(f"Error: Base configuration file not specified or not found at {base_config_path}")
         sys.exit(1)

    if not experiments:
        print("No experiments defined in the config file.")
        sys.exit(0)

    # Load the base configuration once
    with open(base_config_path, encoding='utf-8') as f:
        base_config = yaml.safe_load(f)

    # --- CSV Header ---
    CSV_HEADER = [
        'Date',
        'Model',
        'Encoder',
        'Dataset',
        'Best Val IoU', # Assuming val/IoU was monitored
        'Test mIoU',
        'Test OA',
        'Peak GPU Memory (MB)', # Note: Approximation after run
        'Avg Epoch Time (s)',
        'Total Train Time (s)',
        'Completed Epochs',
        'Status', # e.g., 'Completed', 'Failed', 'Prediction Failed'
        'Log Dir', # Path to the specific log directory for this run
        'Checkpoint Path', # Path to the best checkpoint
    ]

    # Check if the results CSV exists, write header if not
    file_exists = os.path.exists(results_csv_path)
    with open(results_csv_path, 'a', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=CSV_HEADER)
        if not file_exists:
            writer.writeheader()

    print(f"--- Starting Automation Run ({len(experiments)} experiments) ---")
    print(f"Results will be saved to: {results_csv_path}")

    for i, experiment in enumerate(experiments):
        exp_name = experiment.get('name', f'experiment_{i}')
        dataset_name = experiment.get('dataset_name', 'UnknownDataset')
        overrides = experiment.get('overrides', {})

        print(f"\n--- Running Experiment {i+1}/{len(experiments)}: {exp_name} ---")

        # Create a config for the current run by updating the base config
        current_config = update_config(base_config, overrides)

        # Update log and checkpoint directories in the config for the specific run
        # TensorBoardLogger creates save_dir/name/version_X
        # We set save_dir and name in the config, Logger determines version_X
        base_log_dir = current_config.get('log_dir', './logs')
        base_checkpoint_dir = current_config.get('checkpoint_dir', './checkpoints')

        # Use the experiment name from the experiment config
        current_config['experiment_name'] = exp_name.replace('/', '_').replace('\\', '_') # Sanitize name for directory
        current_config['log_dir'] = base_log_dir # Base log directory
        current_config['checkpoint_dir'] = base_checkpoint_dir # Base checkpoint directory

        print(f"Experiment Name for Logging: {current_config['experiment_name']}")
        print(f"Base Log Dir: {current_config['log_dir']}")
        print(f"Base Checkpoint Dir: {current_config['checkpoint_dir']}")


        # --- Run Training ---
        start_time = time.time()
        trainer = None
        best_checkpoint_path = 'N/A'
        best_val_iou = 'N/A'
        completed_epochs = 0
        total_train_time = 0
        avg_epoch_time = 0
        peak_gpu_memory = 0.0
        status = 'Failed: Training Initialization'
        run_log_dir = 'N/A' # Will store the actual log directory with version_X

        try:
            # Call the training function, passing the specific config
            # Assumes run_training returns the trainer instance after fit()
            print("Starting training...")
            # IMPORTANT: The run_training function needs to handle data loading and model init based on config
            trainer = run_training(current_config)
            end_time = time.time()
            total_train_time = end_time - start_time

            # --- Collect Training Metrics from Trainer ---
            # The actual log directory with version_X
            if hasattr(trainer, 'logger') and hasattr(trainer.logger, 'log_dir'):
                 run_log_dir = trainer.logger.log_dir
                 print(f"Actual run log directory: {run_log_dir}")
            else:
                 print("Warning: Could not get actual run log directory from trainer.")
                 # Try to infer it (less reliable)
                 run_log_dir = os.path.join(current_config['log_dir'], current_config['experiment_name'], 'version_0') # Assume version_0 if not found

            # Get best checkpoint path and score from ModelCheckpoint callback
            # Assuming ModelCheckpoint is the first callback of its type
            checkpoint_callback = None
            for callback in trainer.callbacks:
                if isinstance(callback, ModelCheckpoint):
                    checkpoint_callback = callback
                    break

            if checkpoint_callback:
                 best_checkpoint_path = checkpoint_callback.best_model_path
                 best_val_iou = checkpoint_callback.best_model_score
                 print(f"Best checkpoint found: {best_checkpoint_path} with Val IoU: {best_val_iou}")
            else:
                 print("Warning: ModelCheckpoint callback not found on trainer. Cannot determine best checkpoint.")
                 best_checkpoint_path = 'N/A'
                 best_val_iou = 'N/A'

            # Get completed epochs
            # trainer.current_epoch is the index of the last epoch started (0-indexed)
            # If early stopping occurs at epoch N, current_epoch is N. N+1 epochs completed.
            # If max_epochs is reached (say 100), current_epoch is 99. 100 epochs completed.
            completed_epochs = trainer.current_epoch + 1
            print(f"Completed Epochs: {completed_epochs}")


            # Calculate average epoch time
            avg_epoch_time = total_train_time / completed_epochs if completed_epochs > 0 else 0

            # Get peak GPU memory *after* training finishes
            # This is an approximation, not the peak during the entire training process
            peak_gpu_memory = get_peak_gpu_memory_after_run()
            print(f"Peak GPU Memory Recorded (after run): {peak_gpu_memory:.2f} MB")


            print("Training finished successfully.")
            status = 'Training Completed'

        except Exception as e:
            print(f"Training Failed: {e}")
            end_time = time.time()
            total_train_time = end_time - start_time # Still calculate total time up to failure
            avg_epoch_time = 0 # Cannot calculate meaningful avg epoch time
            completed_epochs = trainer.current_epoch + 1 if trainer is not None else 0 # Get epochs completed before failure
            status = f'Failed: Training ({type(e).__name__})'
            best_checkpoint_path = 'N/A'
            best_val_iou = 'N/A'
            run_log_dir = trainer.logger.log_dir if trainer is not None and hasattr(trainer, 'logger') and hasattr(trainer.logger, 'log_dir') else 'N/A'
            print("Skipping prediction due to training failure.")


        # --- Run Prediction (only if training was successful and a best checkpoint was found) ---
        test_miou = 'N/A'
        test_oa = 'N/A'
        prediction_status = 'Skipped' # Initial status for prediction

        # Proceed with prediction if training was successful AND we have a valid checkpoint path
        if status == 'Training Completed' and best_checkpoint_path != 'N/A' and os.path.exists(best_checkpoint_path):
            try:
                print("Starting prediction...")
                # Set the checkpoint_path in the config for the predictor
                current_config['checkpoint_path'] = best_checkpoint_path
                # Pass the config and checkpoint path to the prediction function
                # Assumes run_prediction saves results in the log_dir specified in config
                # We already updated config['log_dir'] to the actual run_log_dir
                predictor_log_dir_returned = run_prediction(current_config, best_checkpoint_path)
                # Note: predictor_log_dir_returned should ideally be the same as run_log_dir if configured consistently
                print(f"Prediction finished. Results should be in: {predictor_log_dir_returned}")

                # --- Read Test Metrics from CSV ---
                # Read from the log directory where prediction saved results
                test_miou, test_oa = read_overall_metrics(predictor_log_dir_returned)

                prediction_status = 'Completed' # Mark prediction as completed

            except Exception as e:
                print(f"Prediction Failed: {e}")
                prediction_status = f'Failed: Prediction ({type(e).__name__})'
                # If prediction fails, update the overall status
                status = prediction_status # The final status is prediction failure

        elif status == 'Training Completed' and (best_checkpoint_path == 'N/A' or not os.path.exists(best_checkpoint_path)):
             # Training completed, but no checkpoint found for some reason (e.g., only 1 epoch run, no validation metric improvement)
             prediction_status = 'Skipped: No Checkpoint Found'
             status = 'Training Completed, No Checkpoint' # Update overall status


        # --- Determine Final Status ---
        if status.startswith('Training Completed') and prediction_status == 'Completed':
            final_status = 'Completed'
        elif status.startswith('Training Completed') and prediction_status.startswith('Skipped'):
            final_status = status # Keep the training completion status with skip reason
        elif status.startswith('Failed: Training'):
             final_status = status # Keep training failure status
        elif prediction_status.startswith('Failed: Prediction'):
             final_status = prediction_status # Keep prediction failure status
        else:
             final_status = status # Catch any other cases


        # --- Write Results to CSV ---
        current_date = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        row_data = {
            'Date': current_date,
            'Model': overrides.get('arch', 'N/A'),
            'Encoder': overrides.get('encoder', 'N/A'),
            'Dataset': dataset_name,
            'Best Val IoU': round(float(best_val_iou), 4) if isinstance(best_val_iou, (int, float)) else best_val_iou,
            'Test mIoU': round(float(test_miou), 4) if isinstance(test_miou, (int, float)) else test_miou,
            'Test OA': round(float(test_oa), 4) if isinstance(test_oa, (int, float)) else test_oa,
            'Peak GPU Memory (MB)': round(peak_gpu_memory, 2),
            'Avg Epoch Time (s)': round(avg_epoch_time, 2),
            'Total Train Time (s)': round(total_train_time, 2),
            'Completed Epochs': completed_epochs,
            'Status': final_status,
            'Log Dir': run_log_dir, # Record the actual log directory for this run
            'Checkpoint Path': best_checkpoint_path,
        }

        with open(results_csv_path, 'a', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=CSV_HEADER)
            writer.writerow(row_data)

        print(f"--- Experiment {i+1}/{len(experiments)} Finished. Status: {final_status}. ---")

    print("\n=== Automation Run Completed ===")

if __name__ == '__main__':
    # You can optionally add argument parsing here to specify the experiments.yaml path
    # Example:
    # import argparse
    # parser = argparse.ArgumentParser(description='Run multiple segmentation experiments.')
    # parser.add_argument('--config', type=str, default='experiments.yaml', help='Path to the experiment configuration YAML file.')
    # args = parser.parse_args()
    # run_all_experiments(args.config)

    # For now, just run with the default path
    run_all_experiments()