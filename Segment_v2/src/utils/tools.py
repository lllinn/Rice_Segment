import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import yaml
import os
from PIL import Image
from torch.utils.data import Dataset
from tqdm import tqdm
import albumentations as A
from albumentations.pytorch import ToTensorV2


class CustomImageDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        """
        自定义数据集类，用于加载图像文件。
        
        Args:
            root_dir (str): 数据集根目录路径。
            transform (callable, optional): 可选的变换操作。
        """
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = [os.path.join(root_dir, fname) for fname in os.listdir(root_dir)
                            if fname.lower().endswith(('png', 'jpg', 'jpeg', 'bmp', 'tiff'))]

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image

class FusedImageDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        """
        自定义数据集类，用于加载图像文件。
        
        Args:
            root_dir (str): 数据集根目录路径。
            transform (callable, optional): 可选的变换操作。
        """
        self.root_dir = root_dir
        self.transform = transform
        self.npy_paths = [os.path.join(root_dir, fname) for fname in os.listdir(root_dir)
                            if fname.lower().endswith(('npy'))]

    def __len__(self):
        return len(self.npy_paths)

    def __getitem__(self, idx):
        npy_path = self.npy_paths[idx]
        npy = np.load(npy_path, allow_pickle=True)
        
        if self.transform:
            npy = self.transform(image=npy)['image']
        return npy

def calculate_mean_std_rgb(dataset_dir):
    # 定义数据加载器，不进行任何变换
    transform = A.Compose([ToTensorV2()])
    # dataset = CustomImageDataset(dataset_dir, transform=transform)
    dataset = FusedImageDataset(dataset_dir, transform=transform)
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=8)

    # 初始化变量
    pixel_sum = np.zeros(13)       # 累积像素值的总和（用于计算均值）
    pixel_squared_sum = np.zeros(13)  # 累积像素值平方的总和（用于计算标准差）
    nb_pixels = 0                 # 总像素数

    for data in tqdm(loader, desc='Calculating Mean and Standard Deviation...'):
        batch_samples = data.size(0)  # 获取当前批次的样本数
        data = data.view(batch_samples, data.size(1), -1)  # 将图像展平为 (C, H*W)

        # 累积像素值的总和
        pixel_sum += data.sum(dim=2).sum(dim=0).cpu().numpy()

        # 累积像素值平方的总和
        pixel_squared_sum += (data ** 2).sum(dim=2).sum(dim=0).cpu().numpy()

        # 更新总像素数
        nb_pixels += data.numel()  # 总像素数（batch_size * channels * H * W）

    # 计算全局均值
    mean = pixel_sum / nb_pixels

    # 计算全局标准差
    std = np.sqrt(pixel_squared_sum / nb_pixels - mean ** 2)

    return mean, std


def calculate_mean_std_rgb_v2(dataset_dir):
    # 定义数据加载器，不进行任何变换
    transform = transforms.Compose([transforms.ToTensor()])
    dataset = CustomImageDataset(dataset_dir, transform=transform)
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=8)

    # 初始化变量    
    pixel_sum = np.zeros(3)       # 累积像素值的总和（用于计算均值）    
    pixel_squared_sum = np.zeros(3)  # 累积像素值平方的总和（用于计算标准差）    
    nb_pixels = 0                 # 总像素数

    for data in tqdm(loader, desc='Calculating Mean and Standard Deviation...'):        
        batch_samples = data.size(0)  # 获取当前批次的样本数        
        data = data.view(batch_samples, data.size(1), -1)  # 将图像展平为 (C, H*W)
                # 累积像素值的总和        
        pixel_sum += data.sum(dim=2).sum(dim=0).cpu().numpy()
                # 累积像素值平方的总和        
        pixel_squared_sum += (data ** 2).sum(dim=2).sum(dim=0).cpu().numpy()
                # 更新总像素数        
        nb_pixels += data.numel()  # 总像素数（batch_size * channels * H * W）
        # 计算全局均值    
    mean = pixel_sum / nb_pixels
        # 计算全局标准差    
    std = np.sqrt(pixel_squared_sum / nb_pixels - mean ** 2)

    return mean, std



if __name__ == "__main__":
    with open("config/fusedRGB.yaml", 'r') as f:
        config = yaml.safe_load(f)
    
    # 调用函数
    dataset_dir = config['dataset_dir']
    train_dir = os.path.join(dataset_dir, config['train_images_dir'])
    mean, std = calculate_mean_std_rgb(train_dir)
    print(f"Mean: {mean}, Std: {std}")