import os
from PIL import Image
import numpy as np
from tqdm import tqdm
from osgeo import gdal
from ..utils.file_io import read_tif

def convert_one_band_tif_to_png(input_folder, output_folder):
    """
    将单波段TIFF图像转换为PNG图像。
    
    Args:
        input_folder (str): 输入TIFF图像所在的文件夹路径。
        output_folder (str): 输出PNG图像所在的文件夹路径。
    
    Returns:
        None
    
    """

    print("Converting one band TIFF images to PNG.....................")
    # 确保输出文件夹存在，如果不存在则创建
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # 遍历输入文件夹中的所有文件
    for filename in tqdm(os.listdir(input_folder)):
        if filename.endswith(".tif") or filename.endswith(".tiff"):
            # 构建输入文件和输出文件的完整路径
            input_file = os.path.join(input_folder, filename)
            output_file = os.path.join(output_folder, os.path.splitext(filename)[0] + ".png")
            
            _, _, im_data, _, _, _ = read_tif(input_file)

            # 将数据转换为灰度图像
            image = Image.fromarray(im_data.astype(np.uint8))
            image = image.convert('L')

            # 保存为png图片
            image.save(output_file)


def convert_three_bands_tif_to_png(input_folder, output_folder):
    """
    将三波段TIFF图像转换为PNG图像。
    
    Args:
        input_folder (str): 包含TIFF图像的输入文件夹路径。
        output_folder (str): 转换后的PNG图像将被保存到的输出文件夹路径。
    
    Returns:
        None
    
    """

    print("Converting three bands TIFF images to PNG.....................")
    # 确保输出文件夹存在，如果不存在则创建
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # 遍历输入文件夹中的所有文件
    for filename in tqdm(os.listdir(input_folder), desc="Convert to png"):
        if filename.endswith(".tif") or filename.endswith(".tiff"):
            # 构建输入文件和输出文件的完整路径
            input_file = os.path.join(input_folder, filename)
            output_file = os.path.join(output_folder, os.path.splitext(filename)[0] + ".png")
            
            dataset = gdal.Open(input_file, gdal.GA_ReadOnly)  
            # 读取图像的所有波段数据为NumPy数组  
            data = []
            for i in range(1, 4):
                band = dataset.GetRasterBand(i)
                data.append(band.ReadAsArray())
            
            # 将波段数据堆叠为RGB图像
            if len(data) == 3:
                image = Image.fromarray(np.stack(data, axis=-1).astype(np.uint8))
            else:
                raise ValueError("Image does not have exactly 3 bands, cannot convert to RGB")

            # 保存为png图片
            image.save(output_file)
