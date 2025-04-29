# coding: utf-8
import sys
sys.path.append('./')  # 将src的上级目录加入sys.path
import os
os.environ['ALBUMENTATIONS_DISABLE_CHECK'] = '1'  # 禁用版本检查
import argparse
import yaml
from src.data.datasets import SegmentationDataset
from src.data.transforms import get_transform_from_config, get_transforms
from src.models.segmentation import SegmentationModel
from src.core.trainer import SegmentationTrainer
from torch.utils.data import DataLoader
import os
from src.utils.email_util import send_email
import pytorch_lightning as pl
from PIL import Image
from torch.utils.data import Dataset as BaseDataset
import cv2
import torch
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

class OutputDataset(BaseDataset):
    def __init__(self, images_dir, output_dir, masks_dir = None, classes=None, augmentation=None, load_on_ram=False):
        # classes: 用户自行提供的class类别
        self.ids = os.listdir(images_dir)
        # 对应图像和masks的标签路径
        self.images_fps = [os.path.join(images_dir, image_id) for image_id in self.ids]
        self.masks_fps = [os.path.join(masks_dir, image_id.replace(".tif", ".png")) for image_id in self.ids]
        self.output_fps = [os.path.join(output_dir, image_id) for image_id in self.ids]
        self.augmentation = augmentation

    def __getitem__(self, i):

        # Read the image
        image = cv2.imread(self.images_fps[i])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
        mask = cv2.imread(self.masks_fps[i], 0)
        # print(self.masks_fps[i])
        if self.augmentation:
            sample = self.augmentation(image=image, mask=mask)
            image, mask = sample["image"], sample["mask"]
            # sample = self.augmentation(image=image)
            # image = sample["image"]

        return image, mask.long(), self.output_fps[i]
        # return image, self.output_fps[i]


    def __len__(self):
        return len(self.ids)

def postprocess(model_outputs, threshold=0.85):
    """
    后处理流程:
    1. 生成五分类结果
    2. 在倒伏区域应用严重度阈值
    """
    # 获取原始分类结果
    model_probs = model_outputs.softmax(dim=1)
    classes = model_probs.argmax(dim=1)
    lodged_probs = model_probs[:, 4]
    # 创建最终标签图
    final_labels = classes.clone()
    # 获取倒伏区域
    lodged_mask = classes == 4
    # 在倒伏区域应用严重度阈值
    final_labels[lodged_mask] = torch.where(
        lodged_probs[lodged_mask] > threshold,
        4, # 重度倒伏
        5  # 轻度倒伏
    )
    
    return final_labels

# 统计所有倒伏区域的概率值
g_lodged_probs = [] # 全局变量，用于存储所有倒伏区域的概率值
def compute_severity_probabilities(model_outputs, class_id = 4):
    model_probs = model_outputs.softmax(dim=1)
    classes = model_probs.argmax(dim=1)
    lodged_mask = classes == class_id
    lodged_probs = model_probs[:, class_id]
    g_lodged_probs.extend(lodged_probs[lodged_mask].detach().cpu().numpy())

def visualize_prob_hist(save_path = "./severity_probability_hist.png", auto_range=True):
    # --- 计算直方图 <button class="citation-flag" data-index="7"><button class="citation-flag" data-index="10">
    # 设置bins为0-1之间的256个等间距区间
    bins = np.linspace(0, 1, 256)  # 生成256个分界点
    hist, bin_edges = np.histogram(g_lodged_probs, bins=bins)

    if auto_range:
        # 自动计算数据的95%核心范围（排除极端值）
        data_min = np.percentile(g_lodged_probs, 2.5)  # 排除最低2.5%
        data_max = np.percentile(g_lodged_probs, 97.5)  # 排除最高2.5%
        
    # --- 可视化直方图 <button class="citation-flag" data-index="1"><button class="citation-flag" data-index="2">
    plt.figure(figsize=(10, 5))
    plt.bar(bin_edges[:-1], hist, width=1/256, align='edge', color='gray', alpha=0.7)
    plt.title(f"Grayscale Histogram ({data_min:.1f}-{data_max:.1f} Range)")
    plt.xlabel(f"Pixel Value ({data_min:.1f}-{data_max:.1f})")
    plt.ylabel("Frequency")
    plt.xlim(data_min, data_max)
    plt.savefig(save_path)
    # plt.show()


def visualize_prob_distribution(bins=50, auto_threshold=True):
    """
    可视化概率分布：
    1. 概率直方图
    2. 累积分布曲线
    3. 自动阈值建议
    """
    probs = np.array(g_lodged_probs).flatten()
    print(f"[调试信息] 概率数据形状: {probs.shape}")  # 应为 (N,)
    print(f"[调试信息] 数据类型: {probs.dtype}")     # 应为 float32

    if len(probs) == 0:
        print("未检测到倒伏区域！")
        return

    plt.figure(figsize=(15, 6))
    
    # ------------------
    # 1. 直方图
    # ------------------
    plt.subplot(1, 2, 1)
    # 修改为
    n, bins, patches = plt.hist(probs.flatten(), bins=bins, density=True, 
                            alpha=0.7, color='skyblue',
                            edgecolor='black')
    
    # 自动寻找阈值（双峰谷底法）
    if auto_threshold:
        threshold = find_valley_threshold(n, bins)
        # plt.axvline(threshold, color='red', linestyle='--', 
        #             linewidth=2, label=f'建议阈值: {threshold:.2f}')
        plt.axvline(threshold, color='red', linestyle='--', 
                    linewidth=2, label=f'suggest thres: {threshold:.2f}')
    
    # plt.xlabel('倒伏严重度概率', fontsize=12)
    # plt.ylabel('概率密度', fontsize=12)
    # plt.title('倒伏区域概率分布直方图', fontsize=14)
    plt.xlabel('lodged prob', fontsize=12)
    plt.ylabel('prob midu', fontsize=12)
    plt.title('hist', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()

    # ------------------
    # 2. 累积分布曲线
    # ------------------
    plt.subplot(1, 2, 2)
    sorted_probs = np.sort(probs)
    cdf = np.arange(1, len(sorted_probs)+1) / len(sorted_probs)
    plt.plot(sorted_probs, cdf, linewidth=3, color='darkorange')
    
    if auto_threshold:
        idx = np.searchsorted(sorted_probs, threshold)
        # plt.scatter(threshold, cdf[idx], color='red', zorder=5,
        #             label=f'阈值点 ({threshold:.2f}, {cdf[idx]:.2f})')
        plt.scatter(threshold, cdf[idx], color='red', zorder=5,
                    label=f'thres ({threshold:.2f}, {cdf[idx]:.2f})')
    
    # plt.xlabel('倒伏严重度概率', fontsize=12)
    # plt.ylabel('累积概率', fontsize=12)
    # plt.title('累积分布曲线 (CDF)', fontsize=14)
    plt.xlabel('lodged prob', fontsize=12)
    plt.ylabel('gailv', fontsize=12)
    plt.title(' (CDF)', fontsize=14)

    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()

    plt.tight_layout()
    plt.show()

    return threshold if auto_threshold else None

def find_valley_threshold(hist_counts, bins):
    """
    基于直方图的双峰谷底法寻找最佳阈值
    """
    # 平滑直方图
    from scipy.ndimage import gaussian_filter1d
    smoothed = gaussian_filter1d(hist_counts, sigma=2)
    
    # 寻找局部最小值
    valleys = []
    for i in range(1, len(smoothed)-1):
        if smoothed[i-1] > smoothed[i] < smoothed[i+1]:
            valleys.append(i)
    
    if len(valleys) == 0:
        return 0.5  # 默认阈值
    
    # 取第一个显著谷底
    valley_idx = valleys[0]
    return bins[valley_idx]

def visualize_errors(pred_masks, true_masks, index):
    # error_masks = torch.zeros_like(true_masks).to(true_masks.device)
    # severe_masks = torch.zeros_like(true_masks).to(true_masks.device)
    # mild_masks = torch.zeros_like(true_masks).to(true_masks.device)
    # weed_masks = torch.zeros_like(true_masks).to(true_masks.device)
    # abnormal_masks = torch.zeros_like(true_masks).to(true_masks.device)
    error_masks = torch.zeros_like(true_masks).to(true_masks.device)
    # def make_masks(in_masks, index):
    error_masks[(pred_masks==index)] = 1  # 预测为异常区域
    error_masks[(true_masks==index)] = 2  # 实际标签为异常区域
    error_masks[(true_masks==index) & (pred_masks!=index)] = 3 # 实际标签为异常区域，但是预测不正确的
    # 生成错误掩码：
    # - 1表示预测为异常区域，但实际标签不是异常区域
    # - 2表示正确的预测到的异常区域
    # - 3表示实际标为异常区域，但预测为正常水稻的区域
    # error_masks[(pred_masks==7)] = 1
    # error_masks[(true_masks == 7)] = 2
    # error_masks[(true_masks == 7) & (pred_masks == 3)] = 3
    # error_masks[(true_masks == 7) & (pred_masks == 4)] = 3
    # make_masks(severe_masks, 4)
    # make_masks(mild_masks, 5)
    # make_masks(weed_masks, 6)
    # make_masks(abnormal_masks, 7)
    return error_masks

def main(config):
    # 设置随机种子
    seed = config['random_seed']
    pl.seed_everything(seed, workers=True) # 固定随机种子，workers=True 确保 DataLoader 的子进程也使用固定的种子
    
    assert len(config['mean']) == len(config['std']) == config['in_channels'], "mean, std and in_channels must have the same length"
    assert config['num_classes'] == len(config['class_names']), "num_classes must be equal to the length of class_names"
    
    DATA_DIR = config['output_dir']
    input_dir = r"Png"
    output_dir   = r"Label_Png"
    label_dir = "Label_True_Png"
    input_dir = os.path.join(DATA_DIR, input_dir)
    output_dir = os.path.join(DATA_DIR, output_dir)
    label_dir = os.path.join(DATA_DIR, label_dir)
    os.makedirs(output_dir, exist_ok=True)
    
    test_transform = get_transforms(config, 'test')
    # 创建数据集
    output_dataset = OutputDataset(
        input_dir,
        output_dir,
        masks_dir=label_dir,
        augmentation=test_transform,
    )
    
    # 构造数据加载器
    output_dataloader = DataLoader(output_dataset, batch_size=config['batch_size'],
        shuffle=False,
        num_workers=config['num_workers'])

    # 设备选择
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # device = torch.device("cpu" if torch.cuda.is_available() else "cpu")

    # 加载训练好的模型
    # TODO: 记得换成最新的路径
    model = SegmentationModel.load_from_checkpoint(config['checkpoint_path'])
    model = model.to(device)


    model.model.eval() # !!!!!! TODO: 很重要很重要很重也好很重要很重要

    with torch.no_grad():
        for data in tqdm(output_dataloader, desc="Create all label..."):
            inputs, true_masks, outputs_path = data
            # inputs, outputs_path = data
            inputs = inputs.to(device)
            true_masks = true_masks.to(device)
            outputs = model(inputs)
            # TODO: 轻重倒伏区分
            # outputs = postprocess(outputs, threshold=0.99)
            
            # TODO: 统计预测为倒伏的概率汇总到一个数组中
            # compute_severity_probabilities(outputs, class_id=5)

            # TODO: 置信度大于90%作为真正的类别，否则作为背景
            # probs = outputs.softmax(dim=1)  # [batch, num_classes, H, W]
            # max_probs, max_indices = torch.max(probs, dim=1)  # 取通道维度最大值
            # threshold = 0.8
            # outputs = torch.where(max_probs >= threshold, max_indices, 0)

            # TODO: 正常推理
            outputs = outputs.softmax(dim=1)   # 对通道间进行softmax操作，得到每个像素属于每个类别的概率
            #  获取倒伏的概率, 最后一个通道
            outputs = outputs.argmax(dim=1)
            
            # TODO: 可视化错误的样本
            # 正确的rice_mild, 预测成rice_normal, 预测成rice_severe
            outputs = visualize_errors(outputs, true_masks, index=5) # 异常区域

            # 保存图像
            for image, path in zip(outputs, outputs_path):
                image = image.cpu().numpy()
                image = Image.fromarray(image.astype('uint8'))
                image.save(path)

    # visualize_prob_hist(save_path="./mild_probability_histogram_v2.png")
    send_email("模型预测", "模型预测完成..")

    # 发送邮箱
    if config['send_to_email']:
        send_email("训练完成")
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default="./config/default.yaml")
    args = parser.parse_args()
    
    with open(args.config, encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    main(config)

    # model_outputs = torch.randn(12, 5, 256, 256)
    # compute_severity_probabilities(model_outputs)
    # optimal_threshold = visualize_prob_distribution(bins=100)
    
    # print(optimal_threshold)