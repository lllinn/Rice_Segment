import albumentations as A
from albumentations.pytorch import ToTensorV2
from torchvision import transforms
transforms.ToTensor()
def get_transforms(config, phase='train'):
    """获取数据增强管道"""
    # size = config['input_size']
    # 默认使用ImageNet的均值和标准差
    mean = config['mean'] if 'mean' in config else [0.485, 0.456, 0.406]
    std = config['std'] if 'std' in config else [0.229, 0.224, 0.225]
    
    if phase == 'train':
        return A.Compose([
        # 水平翻转
            A.HorizontalFlip(p=0.5),
            # 放大，旋转, 平移 
            # A.ShiftScaleRotate(scale_limit=0.5, rotate_limit=0, shift_limit=0.1, p=1, border_mode=0),
            A.ShiftScaleRotate(scale_limit=0.1, rotate_limit=0, shift_limit=0.1, p=1, border_mode=0),
            # A.PadIfNeeded(min_height=320, min_width=320, always_apply=True),
            # A.RandomCrop(height=320, width=320, always_apply=True), # 随机裁剪
            A.RandomCrop(height=512, width=512, always_apply=True), # 随机裁剪
            A.GaussNoise(p=0.2),   # 高斯噪声
            A.Perspective(p=0.5),  # 透视变换
            A.OneOf(
                [
                    A.CLAHE(p=1),
                    A.RandomBrightnessContrast(p=1),
                    A.RandomGamma(p=1),
                ],
                p=0.9,
            ),
            A.OneOf(
                [
                    A.Sharpen(p=1),
                    A.Blur(blur_limit=3, p=1),
                    A.MotionBlur(blur_limit=3, p=1),
                ],
                p=0.9,
            ),
            A.OneOf(
                [
                    A.RandomBrightnessContrast(p=1),
                    A.HueSaturationValue(p=1),
                ],
                p=0.9,
            ),
            A.Normalize(mean=mean, std=std, max_pixel_value=255), # 这里经历了先除以255, 再减去均值除以标准差, 先将255归一化到[0, 1]范围, 再减去均值除以标准差, 与传统的torch不一样, torch的ToTensor会除以255归一化到[0, 1]范围
            ToTensorV2() # 将图像转换为张量, HWC -> CHW, 原封不动的转为tensor, 与torch不一样, torch的ToTensor会除以255归一化到[0, 1]范围
        ])
    else:
        return A.Compose([
            # A.PadIfNeeded(320, 320),
            A.Normalize(mean=mean, std=std, max_pixel_value=255),
            ToTensorV2()
        ])

# 
def get_fusedRGB_transforms(config, phase='train'):
    """获取融合rgb数据增强管道"""
    # mean = config['mean']
    # std = config['std']

    if phase == 'train':
        return A.Compose([
        # 水平翻转
            # A.HorizontalFlip(p=0.5),
            # # 放大，旋转, 平移 
            # A.ShiftScaleRotate(scale_limit=0.1, rotate_limit=0, shift_limit=0.1, p=1, border_mode=0),
            # # A.PadIfNeeded(min_height=320, min_width=320, always_apply=True),
            # A.RandomCrop(height=512, width=512, always_apply=True), # 随机裁剪
            # A.GaussNoise(p=0.2),   # 高斯噪声
            # A.Perspective(p=0.5),  # 透视变换
            # A.OneOf(
            #     [
            #         A.Sharpen(p=1),
            #         A.Blur(blur_limit=3, p=1),
            #         A.MotionBlur(blur_limit=3, p=1),
            #     ],
            #     p=0.9,
            # ),
            # A.Normalize(mean=mean, std=std, max_pixel_value=1.0), # 这里经历了先除以255, 再减去均值除以标准差, 先将255归一化到[0, 1]范围, 再减去均值除以标准差, 与传统的torch不一样, torch的ToTensor会除以255归一化到[0, 1]范围
            ToTensorV2() # 将图像转换为张量, HWC -> CHW, 原封不动的转为tensor, 与torch不一样, torch的ToTensor会除以255归一化到[0, 1]范围
        ])
    else:
        return A.Compose([
            # A.PadIfNeeded(320, 320),
            # A.Normalize(mean=mean, std=std, max_pixel_value=1.0), # 这里经历了先除以255, 再减去均值除以标准差, 先将255归一化到[0, 1]范围, 再减去均值除以标准差, 与传统的torch不一样, torch的ToTensor会除以255归一化到[0, 1]范围
            ToTensorV2()
        ])
        


# 根据配置选择 transform
def get_transform_from_config(config, phase='train'):
    transform_type = config.get('transform_type', 'default')  # 默认使用 'default'
    if transform_type == 'default':
        return get_transforms(config, phase)
    elif transform_type == 'fusedRGB':
        return get_fusedRGB_transforms(config, phase)
    else:
        raise ValueError(f"Unsupported transform_type: {transform_type}")