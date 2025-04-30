import sys
sys.path.append('./') # 将src的上级目录加入sys.path
import os
os.environ['ALBUMENTATIONS_DISABLE_CHECK'] = '1' # 禁用版本检查
import argparse
import yaml

# Import necessary components
from src.data.datasets import SegmentationDataset, RiceRGBVisDataset # Make sure imported
from src.data.transforms import get_fusedRGB_transforms # Make sure imported
from src.models.segmentation import SegmentationModel # Make sure imported (needed by Predictor)
from src.core.predictor import SegmentationPredictor # Assuming SegmentationPredictor is a class you defined
from torch.utils.data import DataLoader # Make sure imported
import os
from src.utils.email_util import send_email # Make sure imported
import pytorch_lightning as pl # Make sure imported

# --- Modifications Start Here ---

# Assuming SegmentationPredictor is a class similar to pl.Trainer or uses pl.Trainer internally
# And assuming it has a logger attribute after initialization or prediction

def run_prediction(config, checkpoint_path):
    """
    Runs the prediction process using a specific checkpoint and configuration.

    Args:
        config (dict): Dictionary containing the overall configuration (including dataset paths).
        checkpoint_path (str): Path to the model checkpoint file (.ckpt).

    Returns:
        str: The path to the directory where prediction results (overall.csv, class.csv) are saved.
             Returns None if an error occurs before predictor initialization or logging dir is inaccessible.
    """
    # Set random seed (important for reproducible data loading/transforms if they are random)
    seed = config['random_seed']
    pl.seed_everything(seed, workers=True)

    print(f"Running prediction using checkpoint: {checkpoint_path}")
    print(f"Using config (primarily for dataset paths): {config}") # Print config for debugging


    predictor = None # Initialize to None
    predictor_log_dir = None # Initialize log dir to None

    try:
        # Initialize the Predictor. It likely loads the model from the checkpoint.
        # Assumes SegmentationPredictor takes the checkpoint path as the primary argument.
        # It might also load config from the checkpoint metadata.
        predictor = SegmentationPredictor(checkpoint_path)
        print("Predictor initialized successfully.")

        # Initialize data for prediction (test set)
        # Use the dataset paths from the *current* config, not necessarily the one saved in the checkpoint
        # This is important if the automation script is testing the same model on a different test set.
        test_transform = get_fusedRGB_transforms(config, 'test')

        dataset_dir = config['dataset_dir']
        test_images_dir = os.path.join(dataset_dir, config['test_images_dir'])
        test_masks_dir = os.path.join(dataset_dir, config['test_masks_dir']) # Masks might not be needed for prediction, but dataset might load them

        test_dataset = RiceRGBVisDataset( # Use the correct Dataset class
            test_images_dir,
            test_masks_dir, # Pass masks dir even if masks aren't used by the predictor itself, Dataset class might require it
            transform=test_transform,
        )

        # Create data loader
        test_loader = DataLoader(
            test_dataset,
            batch_size=config['batch_size'], # Use batch size from the current config
            shuffle=False,
            num_workers=config['num_workers'] # Use num_workers from the current config
            # pin_memory=True, # Add if desired and supported
        )
        print("Test data loaded successfully.")

        # Run prediction
        print("Starting predictor.predict()...")
        predictor.predict(test_loader) # Assumes this method runs prediction and saves results
        print("predictor.predict() finished.")

        # Get the log directory where results were saved
        # Assumes SegmentationPredictor has a logger attribute after predict()
        if hasattr(predictor, 'logger') and hasattr(predictor.logger, 'log_dir'):
             predictor_log_dir = predictor.logger.log_dir
             print(f"Prediction results log dir: {predictor_log_dir}")
        else:
             print("Warning: Could not get prediction log directory from predictor.")
             # Fallback: try to infer based on checkpoint path? Less reliable.
             # Or based on config['log_dir'] and experiment_name if Predictor uses the same logger setup
             pass # Need a reliable way to get the output dir if predictor.logger is not available


        # Note: Email sending is done in the original main, moved here
        if config.get('send_to_email', False):
            try:
                subject = f"Prediction Completed: {config.get('experiment_name', 'N/A')}"
                body = f"Prediction for experiment '{config.get('experiment_name', 'N/A')}' on dataset '{config.get('dataset_name', 'N/A')}' using checkpoint '{os.path.basename(checkpoint_path)}' finished."
                if predictor_log_dir is not None:
                    body += f"\nResults Directory: {predictor_log_dir}"
                send_email(subject, body) # Modify send_email to accept subject and body
                print("Email sent.")
            except Exception as email_e:
                 print(f"Error sending email: {email_e}")


        return predictor_log_dir # Return the log directory path

    except Exception as e:
        print(f"Error during prediction or predictor initialization: {e}")
        # Note: Email sending on failure might be better handled in the automation script
        # if config.get('send_to_email', False):
        #      try:
        #          subject = f"Prediction Failed: {config.get('experiment_name', 'N/A')}"
        #          body = f"Prediction for experiment '{config.get('experiment_name', 'N/A')}' on dataset '{config.get('dataset_name', 'N/A')}' using checkpoint '{os.path.basename(checkpoint_path)}' failed with error: {e}"
        #          if predictor is not None and hasattr(predictor, 'logger') and hasattr(predictor.logger, 'log_dir'):
        #               body += f"\nLog Directory (before failure): {predictor.logger.log_dir}"
        #          send_email(subject, body)
        #      except Exception as email_e:
        #          print(f"Error sending email about failure: {email_e}")
        raise # Re-raise the exception to be caught by the automation script


# --- Modifications End Here ---


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default="./config/fusedRGB.yaml", help="Path to the base YAML configuration file (used for dataset paths, batch size, etc.).")
    parser.add_argument('--checkpoint', type=str, required=True, help="Path to the model checkpoint file (.ckpt) for prediction.")
    args = parser.parse_args()

    with open(args.config, encoding='utf-8') as f:
        config = yaml.safe_load(f)

    print(f"Running predict.py as a standalone script with config: {args.config} and checkpoint: {args.checkpoint}")

    # Call the new prediction function
    try:
        prediction_log_dir = run_prediction(config, args.checkpoint)
        if prediction_log_dir:
             print(f"\nStandalone prediction completed. Results in: {prediction_log_dir}")
             # You can optionally read and print results here if needed
             # test_miou, test_oa = read_overall_metrics(prediction_log_dir)
             # print(f"Test mIoU: {test_miou}, Test OA: {test_oa}")

    except Exception as e:
        print(f"\nStandalone prediction failed: {e}")
        sys.exit(1) # Exit with a non-zero code to indicate failure