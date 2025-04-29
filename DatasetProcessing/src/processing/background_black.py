import os
from osgeo import gdal
import numpy as np
from tqdm import tqdm
from ..utils.file_io import write_tif, read_tif, write_png, read_png, write_png_from_tif
from datetime import datetime
import json
from .convert import tif_to_npy

def set_background_black_file(original_image_path, original_label_path, output_folder):
    """
    将输入图像文件中标签值为0的像素位置替换为黑色，并保存到指定输出目录。
    
    Args:
        original_image_path (str): 输入图像文件路径。
        original_label_path (str): 输入标签文件路径。
        output_folder (str): 输出目录路径。
    
    Raises:
        Exception: 如果输入图像文件或标签文件不存在，或者标签数据为空，则抛出异常。
    
    Returns:
        None
    
    """


    # 创建输出目录，如果目录已存在则不报错
    os.makedirs(output_folder, exist_ok=True)
    
    # 如果标签文件不存在，则抛出异常
    if not (os.path.exists(original_image_path) and os.path.exists(original_label_path)):
        raise Exception(f"{original_image_path} or {original_label_path} not exists")
    
    # 读取label数据
    _, _, label_data, _, _, _ = read_tif(original_label_path)
       
    # 生成标签掩码，标记为0的像素
    label_mask = label_data == 0

    # 释放label内存
    del label_data

    im_proj, im_geotrans, im_data, im_width, im_height, im_bands = read_tif(original_image_path)

    # 遍历图像的每个通道，将label文件对应像素值为黑色的位置用到images的位置上，赋予黑色
    # 将img前三个通道赋值为黑色
    for i in range(3):
        im_data[i, label_mask] = 0
    
    # 生成输出图像路径
    output_image_path = os.path.join(output_folder, os.path.basename(original_image_path))
    write_tif(im_data, im_geotrans, im_proj, output_image_path)


def set_background_black_folder(images_folder, labels_folder, output_folder):
    """
    将输入图像文件夹中的所有图像文件中标签值为0的像素位置替换为黑色，并保存到指定输出目录。
    
    Args:
        images_path (str): 输入图像文件夹路径。
        labels_path (str): 输入标签文件夹路径。
        output_images_path (str): 输出图像文件夹路径。
    
    Returns:
        None
    
    """

    print("Processing images and labels from {} to {}..................".format(images_folder, output_folder))
    # 创建输出目录，如果目录已存在则不报错
    os.makedirs(output_folder, exist_ok=True)
    
    for file_name in tqdm(os.listdir(images_folder)):
        # 只处理以 .tif 结尾的文件
        if file_name.endswith(".tif"):
            image_path = os.path.join(images_folder, file_name)
            label_path = os.path.join(labels_folder, file_name)
            set_background_black_file(image_path, label_path, output_folder)


def is_almost_all_black(im_data, threshold=0.98, black_val=0):
    """
    判断图像中是否几乎全是黑色像素。
    
    Args:
        im_data (numpy.ndarray): 图像数据，可以是灰度图像或多波段图像。
        threshold (float, optional): 黑色像素占比的阈值，默认为0.98。如果黑色像素占比超过该阈值，则认为图像几乎全是黑色。
    
    Returns:
        bool: 如果图像中黑色像素占比超过阈值，则返回True；否则返回False。
    
    """

    # 检查图像中黑色像素的比例是否超过threshold
    if len(im_data.shape) == 2:
        black_pixels = np.sum(im_data == black_val)
        total_pixels = im_data.size
    else:
        # 对于多波段图像，检查所有波段是否都为0
        black_pixels = np.sum(np.all(im_data == black_val, axis=0))
        total_pixels = im_data.shape[1] * im_data.shape[2]
    
    black_ratio = black_pixels / total_pixels
    return black_ratio > threshold



def delete_almost_all_black_tiffs(folder_path, label_path=None,threshold=0.98):
    """
    删除文件夹中几乎所有像素为黑色的TIFF文件。
    
    Args:
        folder_path (str): 要处理的文件夹路径。
        label_path (str, optional): 标签文件所在的文件夹路径。如果提供，将同时删除对应的标签文件。默认为None。
        threshold (float, optional): 判断图像是否几乎全黑的阈值。图像中黑色像素的比例超过此阈值时，该文件将被删除。默认为0.98。
    
    Returns:
        None
    
    """
    print("Deleting almost all black TIFs.................................")
    # 获取文件夹中所有Tiff文件的列表
    tif_files = [f for f in os.listdir(folder_path) if f.lower().endswith('.tif')]
    tif_files.sort(key=lambda x: int(x.split('.')[0].split('_')[0]))
    delete_num = 0
    # 使用tqdm创建进度条
    for filename in tqdm(tif_files, desc="Processing files", unit="file"):
        file_path = os.path.join(folder_path, filename)
        _, _, im_data, _, _, _ = read_tif(file_path)
        if is_almost_all_black(im_data, threshold):
            os.remove(file_path)  # 删除文件
            if label_path:
                label_file_path = os.path.join(label_path, filename)
                os.remove(label_file_path)
            delete_num += 1
    print("all_num:{0}, delete_num:{1}, out_num:{2}".format(len(tif_files), delete_num, len(tif_files)-delete_num))


def delete_or_set_background_black(images_folder, labels_folder, output_folder, labels_output_folder,threshold=0.98):
    """
    从输入图像文件夹中删除几乎全部像素为黑色的TIFF文件，并将其余文件复制到输出文件夹中。
    
    Args:
        images_folder (str): 输入图像文件夹路径。
        labels_folder (str): 输入标签文件夹路径。
        output_folder (str): 输出文件夹路径。
        threshold (float, optional): 判断图像是否几乎全黑的阈值。图像中黑色像素的比例超过此阈值时，该文件将被删除。默认为0.98。
    
    Returns:
        None
    
    """
    print("Deleting or set almost all black pngs.................................")
    # 获取文件夹中所有Tiff文件的列表
    tif_files = [f for f in os.listdir(images_folder) if f.lower().endswith('.tif')]
    delete_num = 0
    for filename in tqdm(tif_files, desc="Processing files", unit="file"):
        file_path = os.path.join(images_folder, filename)
        _, _, im_data, _, _, _ = read_tif(file_path)
        if is_almost_all_black(im_data, threshold):
            delete_num += 1
        else: # 设置黑色背景
            label_file_path = os.path.join(labels_folder, filename)
            _, _, label_data, _, _, _ = read_tif(label_file_path)
            label_mask = label_data == 0
            output_file = os.path.splitext(filename)[0] + ".png"
            write_png_from_tif(label_data, os.path.join(labels_output_folder, output_file))
            del label_data
            # print(im_data.shape)
            im_data[:, label_mask] = 0
            # write_png(im_data, os.path.join(output_folder, filename))
            write_png_from_tif(im_data[0:3], os.path.join(output_folder, output_file))
    print("all_num:{0}, delete_num:{1}, out_num:{2}".format(len(tif_files), delete_num, len(tif_files)-delete_num))


def record_image_skip(images_folder, threshold=0.9, skip_log: str = "skip_log.json", black_val=0):
    print("record image skip.................................")
    # 初始化跳过记录系统
    skip_records = {}
    # if os.path.exists(skip_log):
    #     with open(skip_log, 'r') as f:
    #         skip_records = json.load(f)
    #     print(f"已加载历史跳过记录：{len(skip_records)} 条")
    # else:
    #     # 创建文件夹
    os.makedirs(os.path.dirname(skip_log), exist_ok=True)
    # 获取文件夹中所有Tiff文件的列表

    tif_files = [f for f in os.listdir(images_folder) if f.lower().endswith('.tif')]
    delete_num = 0
    for filename in tqdm(tif_files, desc="Processing files", unit="file"):
        file_path = os.path.join(images_folder, filename)
        _, _, im_data, _, _, _ = read_tif(file_path)
        if is_almost_all_black(im_data, threshold, black_val=black_val):
            delete_num += 1
            # 更新记录
            skip_records[filename] = {
                "timestamp": np.datetime64('now').astype(str),
            }
    # 写入文件
    with open(skip_log, 'w') as f:
        json.dump(skip_records, f, indent=2)
    print("all_num:{0}, delete_num:{1}, out_num:{2}".format(len(tif_files), delete_num, len(tif_files)-delete_num))




def smart_image_converter(
    image_dir: str,
    output_img_dir: str,
    shuffix: str = ".npy", # 默认是png
    skip_log: str = "conversion_skip.json",
):
    """
    智能图像转换函数，支持跳过记录复用和选择性标签处理
    
    参数说明：
    - image_dir: 原始图像目录
    - output_img_dir: 处理后的图像输出目录
    - skip_log: 跳过记录文件路径
    """
    
    # 初始化跳过记录系统
    skip_records = {}
    if os.path.exists(skip_log):
        with open(skip_log, 'r') as f:
            skip_records = json.load(f)
        print(f"已加载历史跳过记录：{len(skip_records)} 条")
    else:
        raise FileNotFoundError("跳过记录文件不存在")

    # 准备待处理文件列表
    all_images = [f for f in os.listdir(image_dir) if f.lower().endswith('.tif')]
    
    # 创建输出目录
    os.makedirs(output_img_dir, exist_ok=True)

    processed_count = 0
    skip_count = 0

    with tqdm(total=len(all_images), desc="转换进度") as pbar:
        for filename in all_images:
            # 跳过已记录文件
            if filename in skip_records:
                skip_count += 1
                pbar.update(1)
                continue
            
            img_path = os.path.join(image_dir, filename)
            if shuffix == ".npy": # 转换Image为npy
                npy_path = os.path.join(output_img_dir, filename.replace('.tif', '.npy'))          
                tif_to_npy(img_path, npy_path)
            elif shuffix == ".png": # 转换Mask为png
                png_path = os.path.join(output_img_dir, filename.replace('.tif', '.png'))
                _, _, label_data, _, _, _ = read_tif(img_path)
                write_png_from_tif(label_data, png_path)

            processed_count += 1
            pbar.update(1)


    # 输出统计结果
    print("\n转换结果摘要：")
    print(f"总文件数: {len(all_images)}")
    print(f"成功转换: {processed_count}")
    print(f"跳过文件: {skip_count}")


def print_result(total: int, skipped: int, processed: int):
    """输出处理结果"""
    print("\n处理结果摘要:")
    print(f"总文件数: {total}")
    print(f"本次跳过文件: {skipped}")
    print(f"成功处理文件: {processed}")
    print(f"保留未处理文件: {total - skipped - processed}")


def check_skip_log(skip_log1, skip_log2):
    # 判断内容是否一致
    skip_records_1 = {}
    if os.path.exists(skip_log1):
        with open(skip_log1, 'r') as f:
            skip_records_1 = json.load(f)
        print(f"已加载历史跳过记录：{len(skip_records_1)} 条")
    else:
        print("skip log 1 加载失败")

    skip_records_2 = {}
    if os.path.exists(skip_log2):
        with open(skip_log2, 'r') as f:
            skip_records_2 = json.load(f)
        print(f"已加载历史跳过记录：{len(skip_records_2)} 条")
    else:
        print("skip log 2 加载失败")

    # 判断1和2的键是不是一样
    print(set(skip_records_1.keys()) == set(skip_records_2.keys()))




if __name__ == "__main__":
    # python -m src.processing.background_black

    # print("start...")
    # start_time = datetime.now()
    # original_image_path = r"D:\2024\pytorch_project\tif\dataset_proc\Loding-12.05-v3.0\images_origin"
    # original_label_path = r"D:\2024\pytorch_project\tif\dataset_proc\Loding-12.05-v3.0\labels_tif"
    # file_head = r"D:\2024\pytorch_project\tif\dataset_proc\Loding-12.05-v3.0"
    # images_name = r"images"
    # images_output_path = os.path.join(file_head, images_name)
    # set_background_black_folder(original_image_path, original_label_path, images_output_path)

    # image_folder = r"E:\Code\RiceLodging\datasets\DJ\Test_Skip\Meiju1u8-04.10-7-640-0.1-0.6-0.2-0.2-v4\images_origin"
    # # output_folder = r"E:\Code\RiceLodging\datasets\DJ\Test_Skip\1-04.02-7-640-0.1-0.6-0.2-0.2-v3\images"
    # record_image_skip(images_folder=image_folder, skip_log="logs/Meiju2_u8.json")
    check_skip_log("logs/Meiju2.json", r"logs\Meiju2_u8.json")    

    # float32_json = r"logs\Meiju1_Float32.json"
    # u8_json = r"logs\Meiju1_u8.json"
    # float_skip_records = {}
    # if os.path.exists(float32_json):
    #     with open(float32_json, 'r') as f:
    #         float_skip_records = json.load(f)
    #     print(f"已加载历史跳过记录：{len(float_skip_records)} 条")

    # u8_skip_records = {}
    # if os.path.exists(u8_json):
    #     with open(u8_json, 'r') as f:
    #         u8_skip_records = json.load(f)
    #     print(f"已加载历史跳过记录：{len(u8_skip_records)} 条")
    # # 对比两个字典是否一致
    # print(set(float_skip_records.keys()) == set(u8_skip_records.keys()))
    # smart_image_converter(image_dir=image_folder, output_img_dir=output_folder, skip_log="logs\skip_log.json")


    pass