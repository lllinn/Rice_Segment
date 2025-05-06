import sys
sys.path.append('./') # 将src的上级目录加入sys.path
import os
os.environ['ALBUMENTATIONS_DISABLE_CHECK'] = '1' # 禁用版本检查
import argparse
import yaml

# Import necessary components
from src.data.datasets import SegmentationDataset, RiceRGBVisDataset # Make sure these are imported outside the function
from src.data.transforms import get_transform_from_config, get_fusedRGB_transforms # Make sure these are imported outside the function
from src.models.segmentation import SegmentationModel # Make sure this is imported outside the function
from src.core.trainer import SegmentationTrainer # Make sure this is imported outside the function
from torch.utils.data import DataLoader # Make sure this is imported outside the function
import os
from src.utils.email_util import send_email # Make sure this is imported outside the function
import pytorch_lightning as pl # Make sure this is imported outside the function
import time # Used for timing




def run_training(config):
    """
    Runs the training process based on the provided configuration dictionary.

    Args:
        config (dict): Dictionary containing the training configuration.

    Returns:
        pytorch_lightning.Trainer: The trainer instance after fitting.
                                   Returns None if an error occurs before trainer initialization.
    """
    # Set random seed
    seed = config['random_seed']
    pl.seed_everything(seed, workers=True) # 固定随机种子，workers=True 确保 DataLoader 的子进程也使用固定的种子

    print(f"Running training for experiment: {config.get('experiment_name', 'N/A')}")
    print(f"Config: {config}") # Print config for debugging

    # Initialize data
    try:
        train_transform = get_fusedRGB_transforms(config, 'train')
        val_transform = get_fusedRGB_transforms(config, 'val')

        dataset_dir = config['dataset_dir']
        train_images_dir = os.path.join(dataset_dir, config['train_images_dir'])
        train_masks_dir = os.path.join(dataset_dir, config['train_masks_dir'])
        val_images_dir = os.path.join(dataset_dir, config['val_images_dir'])
        val_masks_dir = os.path.join(dataset_dir, config['val_masks_dir'])

        train_dataset = RiceRGBVisDataset(
            train_images_dir,
            train_masks_dir,
            transform=train_transform,
        )

        val_dataset = RiceRGBVisDataset(
            val_images_dir,
            val_masks_dir,
            transform=val_transform
        )

        # 创建数据加载器
        train_loader = DataLoader(
            train_dataset,
            batch_size=config['batch_size'],
            shuffle=True,
            num_workers=config['num_workers'],
            pin_memory=True,
            persistent_workers=True,
            drop_last=True,   # 丢弃最后一个批次，防止数据集大小不是batch_size的整数倍
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=config['batch_size'],
            num_workers=config['num_workers'],
            shuffle=False,
            pin_memory=True,
            persistent_workers=True,
        )
        print("Data loaded successfully.")
    except Exception as e:
        print(f"Error loading data: {e}")
        raise # Re-raise the exception to be caught by the automation script


    # Initialize model
    try:
        # Note: 'resume' logic might need adjustment if handling resuming multiple runs
        # For automation, we typically train from scratch for each experiment config.
        # If resuming is needed per experiment, the automation script needs to manage checkpoints.
        # Assuming for automation, resume is False or handled externally per experiment.
        if config.get('resume', False): # Check if resume key exists and is True
            # Ensure checkpoint_path is provided in config for resuming
            checkpoint_path = config.get('checkpoint_path')
            if not checkpoint_path or not os.path.exists(checkpoint_path):
                 raise FileNotFoundError(f"Resume=True but checkpoint_path missing or not found: {checkpoint_path}")
            print(f"Resuming training from checkpoint: {checkpoint_path}")
            model = SegmentationModel.load_from_checkpoint(checkpoint_path)
        else:
            print("Initializing model from scratch.")
            model = SegmentationModel(config) # Pass the entire config

        print("Model initialized successfully.")
    except Exception as e:
        print(f"Error initializing model: {e}")
        raise # Re-raise the exception

    # Initialize SegmentationTrainer and run fit
    segmentation_trainer_instance = None # Initialize to None
    try:
        print("Initializing SegmentationTrainer...")
        segmentation_trainer_instance = SegmentationTrainer(config)

        print("Starting trainer.fit()...")
        # Assumes SegmentationTrainer.fit returns the pl.Trainer instance
        trainer = segmentation_trainer_instance.fit(model, [train_loader, val_loader])
        print("trainer.fit() finished.")

        # Note: Email sending is done in the original main, moved here
        if config.get('send_to_email', False):
            try:
                # Provide more context in the email
                subject = f"Training Completed: {config.get('experiment_name', 'N/A')}"
                body = f"Training for experiment '{config.get('experiment_name', 'N/A')}' on dataset '{config.get('dataset_name', 'N/A')}' finished successfully."
                # You might want to add summary stats here if available
                if hasattr(trainer, 'logger') and hasattr(trainer.logger, 'log_dir'):
                     body += f"\nLog Directory: {trainer.logger.log_dir}"
                send_email(subject, body) # Modify send_email to accept subject and body
                print("Email sent.")
            except Exception as email_e:
                print(f"Error sending email: {email_e}")


        return trainer # Return the pytorch_lightning.Trainer instance

    except Exception as e:
        print(f"Error during training or trainer initialization: {e}")
        # Note: Email sending on failure might be better handled in the automation script
        # if config.get('send_to_email', False):
        #     try:
        #          subject = f"Training Failed: {config.get('experiment_name', 'N/A')}"
        #          body = f"Training for experiment '{config.get('experiment_name', 'N/A')}' on dataset '{config.get('dataset_name', 'N/A')}' failed with error: {e}"
        #          if segmentation_trainer_instance is not None and hasattr(segmentation_trainer_instance, 'logger') and hasattr(segmentation_trainer_instance.logger, 'log_dir'):
        #              body += f"\nLog Directory: {segmentation_trainer_instance.logger.log_dir}"
        #          send_email(subject, body)
        #     except Exception as email_e:
        #          print(f"Error sending email about failure: {email_e}")
        raise # Re-raise the exception to be caught by the automation script


# --- Modifications End Here ---


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default="./config/fusedRGB.yaml", help="Path to the YAML configuration file.")
    args = parser.parse_args()

    with open(args.config, encoding='utf-8') as f:
        config = yaml.safe_load(f)

    print(f"Running train.py as a standalone script with config: {args.config}")

    # Call the new training function
    try:
        trainer_instance = run_training(config)
        # You can access trainer_instance properties here if needed after standalone run
        if trainer_instance:
            print("\nStandalone training completed.")
            if hasattr(trainer_instance, 'checkpoint_callback') and trainer_instance.checkpoint_callback:
                print(f"Best checkpoint path: {trainer_instance.checkpoint_callback.best_model_path}")
                print(f"Best validation score: {trainer_instance.checkpoint_callback.best_model_score}")
            if hasattr(trainer_instance, 'logger') and hasattr(trainer_instance.logger, 'log_dir'):
                 print(f"Log Directory: {trainer_instance.logger.log_dir}")

    except Exception as e:
        print(f"\nStandalone training failed: {e}")
        sys.exit(1) # Exit with a non-zero code to indicate failure