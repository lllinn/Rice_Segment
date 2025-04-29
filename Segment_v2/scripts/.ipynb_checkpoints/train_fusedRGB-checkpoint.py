import sys
sys.path.append('./')  # 将src的上级目录加入sys.path
import os
os.environ['ALBUMENTATIONS_DISABLE_CHECK'] = '1'  # 禁用版本检查
import argparse
import yaml
from src.data.datasets import SegmentationDataset, RiceRGBVisDataset
from src.data.transforms import get_transform_from_config, get_fusedRGB_transforms
from src.models.segmentation import SegmentationModel
from src.core.trainer import SegmentationTrainer
from torch.utils.data import DataLoader
import os
from src.utils.email_util import send_email
import pytorch_lightning as pl

def main(config):
    # 设置随机种子
    seed = config['random_seed']
    pl.seed_everything(seed, workers=True) # 固定随机种子，workers=True 确保 DataLoader 的子进程也使用固定的种子
    
    assert len(config['mean']) == config['in_channels'] == len(config['std']) , "mean, std, and in_channels must be equal"
    
    # 初始化数据
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
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        num_workers=config['num_workers'],
        shuffle=False,
        pin_memory=True, 
        persistent_workers=True,
    )
    
    # 初始化模型
    # 初始化模型
    if config['resume']:
        model = SegmentationModel.load_from_checkpoint(config['checkpoint_path'])
    else:
        model = SegmentationModel(config)
    
    # 训练
    trainer = SegmentationTrainer(config)
    trainer.fit(model, [train_loader, val_loader])

    # 发送邮箱
    if config['send_to_email']:
        send_email("训练完成")
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default="./config/fusedRGB.yaml")
    args = parser.parse_args()
    
    with open(args.config, encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    main(config)
