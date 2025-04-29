import sys
sys.path.append('./')  # 将src的上级目录加入sys.path
import os
os.environ['ALBUMENTATIONS_DISABLE_CHECK'] = '1'  # 禁用版本检查
import argparse
import yaml
from src.data.datasets import SegmentationDataset, RiceRGBVisDataset
from src.data.transforms import get_fusedRGB_transforms
from src.models.segmentation import SegmentationModel
from src.core.predictor import SegmentationPredictor
from torch.utils.data import DataLoader
import os
from src.utils.email_util import send_email
import pytorch_lightning as pl

def main(config):
    # 设置随机种子
    seed = config['random_seed']
    pl.seed_everything(seed, workers=True) # 固定随机种子，workers=True 确保 DataLoader 的子进程也使用固定的种子
    
    trainer = SegmentationPredictor(config["checkpoint_path"])
    config = trainer.config

    # 初始化数据
    test_transform = get_fusedRGB_transforms(config, 'test')
    
    dataset_dir = config['dataset_dir']
    test_images_dir = os.path.join(dataset_dir, config['test_images_dir'])
    test_masks_dir = os.path.join(dataset_dir, config['test_masks_dir'])
    
    test_dataset = RiceRGBVisDataset(
        test_images_dir,
        test_masks_dir,
        transform=test_transform,
    )

    # 创建数据加载器
    test_loader = DataLoader(
        test_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=config['num_workers']
    )
    
    
    trainer.predict(test_loader)
    print("log dir is", trainer.logger.log_dir)
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
