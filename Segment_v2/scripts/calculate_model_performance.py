import sys
sys.path.append(r'./')  # 将src的上级目录加入sys.path
import os
os.environ['ALBUMENTATIONS_DISABLE_CHECK'] = '1'  # 禁用版本检查
import argparse
import yaml
from src.data.datasets import SegmentationDataset
from src.data.transforms import get_transform_from_config, get_transforms
from src.models.segmentation import SegmentationModel
from src.core.trainer import SegmentationTrainer
from torch.utils.data import DataLoader
from src.utils.email_util import send_email
import pytorch_lightning as pl
from datetime import datetime
import torch

def main(config):
    # 设置随机种子
    seed = config['random_seed']
    pl.seed_everything(seed, workers=True) # 固定随机种子，workers=True 确保 DataLoader 的子进程也使用固定的种子
    
    assert len(config['mean']) == len(config['std']) == config['in_channels'], "mean, std and in_channels must have the same length"
    assert config['num_classes'] == len(config['class_names']), "num_classes must be equal to the length of class_names"

    # 检查是否有可用的GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    model = SegmentationModel(config)
    # model = model.to(device)  # 将模型移动到正确的设备上
    
    model.calcute_model_performance()

    send_email("模型性能计算完毕...")
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default="./config/default.yaml")
    args = parser.parse_args()
    
    with open(args.config, encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    start_time = datetime.now()
    main(config)

