import os
import cv2
import numpy as np
import albumentations as A
from torch.utils.data import Dataset
from PIL import Image
from pathlib import Path





class SegmentationDataset(Dataset):
    """支持多种数据增强的语义分割数据集"""
    
    def __init__(self, images_dir, masks_dir, transform=None, load_on_ram=False):
        self.ids = sorted(os.listdir(images_dir))
        self.images_fps = [os.path.join(images_dir, img_id) for img_id in self.ids]
        self.masks_fps = [os.path.join(masks_dir, img_id) for img_id in self.ids]
        self.transform = transform
        self.load_on_ram = load_on_ram
        
        if self.load_on_ram:
            self._preload_data()

    def _preload_data(self):
        self.images, self.masks = [], []
        for img_path, mask_path in zip(self.images_fps, self.masks_fps):
            image = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
            mask = cv2.imread(mask_path, 0)
            self.images.append(image)
            self.masks.append(mask)
        # 转为numpy数组, 避免内存泄漏
        self.images = np.array(self.images)
        self.masks = np.array(self.masks)
        # self.masks = np.append(self.masks, mask)

    def __getitem__(self, idx):
        if self.load_on_ram:
            image, mask = self.images[idx], self.masks[idx]
        else:            
            image = cv2.cvtColor(cv2.imread(self.images_fps[idx]), cv2.COLOR_BGR2RGB)
            mask = cv2.imread(self.masks_fps[idx], 0)

        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image, mask = augmented['image'], augmented['mask']

        return image, mask.long() # .long() 转为索引张量, one hot 编码要求用long整数表示类别

    def __len__(self):
        return len(self.ids)
    

def preconvert(src_dir, dst_dir):
    dst_dir = Path(dst_dir)
    dst_dir.mkdir(exist_ok=True)
    
    for npy_path in Path(src_dir).glob("*.npy"):
        arr = np.load(npy_path)
        mmap_path = dst_dir / (npy_path.stem + ".dat")
        mmap = np.memmap(mmap_path, dtype=arr.dtype, mode='w+', shape=arr.shape)
        mmap[:] = arr[:]
        mmap.flush()


class RiceRGBVisDataset(Dataset):
    """读取带有植被指数的rgb数据"""
    
    def __init__(self, images_dir, masks_dir, transform=None):
        self.ids = sorted(os.listdir(images_dir))
        self.npys_fps = [os.path.join(images_dir, img_id) for img_id in self.ids]
        self.masks_fps = [os.path.join(masks_dir, img_id.replace(".npy", ".png")) for img_id in self.ids]
        self.transform = transform


    def __getitem__(self, idx):
        # npy = np.load(self.npys_fps[idx], allow_pickle=True)
        npy = np.load(self.npys_fps[idx], mmap_mode='c') # 共享内存映射
        mask = cv2.imread(self.masks_fps[idx], 0)

        if self.transform:
            augmented = self.transform(image=npy, mask=mask)
            npy, mask = augmented['image'], augmented['mask']

        return npy, mask.long() # .long() 转为索引张量, one hot 编码要求用long整数表示类别

    def __len__(self):
        return len(self.ids)