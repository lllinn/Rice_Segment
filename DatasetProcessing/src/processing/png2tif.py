import pathlib

import numpy as np
from osgeo import gdal
import os
import shutil
from tqdm import tqdm
from PIL import Image
from ..utils.file_io import write_tif, read_tif
from ..utils.email_utils import send_email

# 根据已有的tif将png转为tif
# png_folder: 推理得到的png图片
# tif_folder: 标签的tif文件
# output_folder: 保存结果的文件夹
# 实现思路: 拿之前的已有的tif的"地理信息", 应用到新生成的png图像信息, 生成新的tif文件(地理信息+png)
# 缩放因子: scale
def png2tif(png_folder, tif_folder, output_folder, scale=1):
    """
    将指定目录下的所有png文件转换为tif文件。
    
    Args:
        png_folder (str): png文件所在的目录路径。
        tif_folder (str): 对应的tif文件所在的目录路径。
        output_folder (str): 转换后的tif文件存放的目录路径。
        scale (float, optional): 缩放比例，默认为1。
    
    Returns:
        None
    
    """
    for png_file in tqdm(os.listdir(png_folder), desc='Png to Tif ...') :
        tif_file = png_file.replace(".png", ".tif") # 将png文件名转为tif文件名
        output_file = os.path.join(output_folder, tif_file) # 获取输出路径 .tif
        tif_file = os.path.join(tif_folder, tif_file) # 原来的tif文件路径
        im_proj, im_geotrans, _, _, _, _ = read_tif(tif_file) # 获取原来tif文件的地理信息
        png_file = os.path.join(png_folder, png_file) # 使用网络推理得到的图片信息
        im_data = Image.open(png_file)
        im_data = np.array(im_data)   # 获取新图像的数据
        if len(im_data.shape) == 3:
            # hwc -> chw
            im_data = im_data.transpose(2, 0, 1)

        im_geotrans = list(im_geotrans)
        im_geotrans[1] = im_geotrans[1] / scale
        im_geotrans[5] = im_geotrans[5] / scale
        im_geotrans = tuple(im_geotrans)

        write_tif(im_data, im_geotrans, im_proj, output_file) # 将png图片的信息写入到正确的tif文件中
        

def remove_shuffix(folder, shuffix="_out"):
    for file in tqdm(os.listdir(folder), desc='Remove Shuffix Out ...'):
        new_file = file.replace(shuffix, '')
        old_file = os.path.join(folder, file)
        new_file = os.path.join(folder, new_file)
        shutil.move(old_file, new_file)



if __name__ == "__main__":

    # TODO: Label数据拼接
    all_folder = r"/home/music/wzl/segment-task/Meiju1_v7_merge"
    label_png_folder = "Label_Png"
    label_folder = "Label"
    label_True_folder = "Label_True"
    label_png_folder = os.path.join(all_folder, label_png_folder)
    label_folder = os.path.join(all_folder, label_folder)
    label_True_folder = os.path.join(all_folder, label_True_folder)
    os.makedirs(label_folder, exist_ok=True)


    png2tif(label_png_folder, label_True_folder, label_folder)
    send_email("图像拼接")    



