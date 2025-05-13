import sys
sys.path.append('./')
from src.processing.convert import tif_to_npy_folder
from src.processing.background_black import smart_image_converter
from src.processing.stack_data import stack_npy_files, split_npy_files_threaded
from src.processing.crop import crop_with_repetition, crop_with_repetition_and_convert_delete, crop_with_repetition_and_save_skip_log
from src.utils.email_utils import send_email
from src.utils.file_io import remove_folder, rename_files
from src.processing.split import split_dataset
import os
import argparse
from tqdm import tqdm
from datetime import datetime
import numpy as np
from collections import defaultdict
from typing import Dict, List, Tuple
import json
from calc_mean_std import calc_mean_and_std, visualize_feature
import shutil
from osgeo import gdal
from PIL import Image
import concurrent.futures # 导入并发处理模块

def create_dataset_image(image_folder, output_folder, resolution, crop_size=640, repetition_rate=0.1, tif_shuffix = ".tif", shuffix=".npy", skip_log="conversion_skip.json"):
    # 先裁剪Tif(Image_Tif)和标签文件夹(Label_Tif)
    # 删除多余的图片(Image_Tif)并转换为npy文件(Image_Npy)和png文件(Label_Png)
    # 最后删除标签和tif文件夹(Image_Tif, Label_Tif)
    image_tif_folder = "Image_Tif"
    # label_tif_folder = "Label_Tif"
    image_npy_folder = "Image_Npy"
    # label_png_folder = "Label_Png"
    image_tif_folder = os.path.join(output_folder, image_tif_folder)
    # label_tif_folder = os.path.join(output_folder, label_tif_folder)
    image_npy_folder = os.path.join(output_folder, image_npy_folder)
    # label_png_folder = os.path.join(output_folder, label_png_folder)

    os.makedirs(image_tif_folder, exist_ok=True)
    os.makedirs(image_npy_folder, exist_ok=True)

    # 判断文件夹是不是空的
    if len(os.listdir(image_npy_folder)) != 0:
        # send_email(f"输入{output_folder}是已经生成过了，无需重复输入！！！")
        print(f"输入文件夹{output_folder}是已经生成过了，无需重复输入！！！")
        # 删除文件夹
        os.removedirs(image_tif_folder)
        return
        raise ValueError("输入文件夹是已经生成过了，无需重复输入！！！")

    # 裁剪图片和标签
    for file in os.listdir(image_folder):
        if file.endswith(tif_shuffix):
            print("正在处理{}.........".format(file))
            crop_with_repetition_and_convert_delete(os.path.join(image_folder, file), image_tif_folder, 
                                                    crop_size, repetition_rate, image_npy_folder, shuffix=shuffix, skip_log=skip_log, resolution=resolution)


def normalize_dataset(stack_folder, max_value = [], min_value = []):
    max_value = np.asarray(max_value, dtype=np.float32)
    min_value = np.asarray(min_value, dtype=np.float32)
    range_val = max_value - min_value
    for file in tqdm(os.listdir(stack_folder)):
        if file.endswith(".npy"):
            # 对 [h, w, c]的channel进行归一化处理
            npy_data = np.load(os.path.join(stack_folder, file)) # 加载矩阵 (h,w,c)
            # 对 [h, w, c]的channel进行归一化处理
            # TODO: 这里的最后一个维度不进行归一化处理, CSM的问题
            npy_data = (npy_data - min_value[np.newaxis, np.newaxis, :]) / range_val[np.newaxis, np.newaxis, :]
            # for i in range(npy_data.shape[-1]):
            #     npy_data[:, :, i] = (npy_data[:, :, i]-min_value[i])/(max_value[i]-min_value[i])
            np.save(os.path.join(stack_folder, file), npy_data) # 对其重新写入

def get_band_statistics(
    base_folder: str,
    input_folders_name: List[str],
    json_path: Dict[str, Dict] = 'None',
    ignore_case: bool = True
) -> Tuple[List[float], List[float], List[float], List[float]]:
    """
    根据文件夹结构和波段顺序提取统计信息
    
    Args:
        json_data (Dict): 包含统计信息的JSON数据结构
        base_folder (str): 原始数据根目录路径
        input_folders_name (List[str]): 按优先级排序的文件夹结构列表
        ignore_case (bool): 是否忽略路径大小写，默认为True
    
    Returns:
        Tuple[List[float], List[float], List[float], List[float]]: 
        (max_values, min_values, means, stds)
    
    Example:
        >>> maxs, mins, means, stds = get_band_statistics(
        ...     json_data,
        ...     r"D:\GLCM\Meiju1\Stack\split",
        ...     ["RGB", "Vegetation-Index/band8to11"],
        ...     ignore_case=True
        ... )
    """
    if os.path.exists(json_path):
        with open(json_path, 'r') as f:
            json_data = json.load(f)
        print(f"已加载历史记录：{len(json_data)} 条")

    def normalize_path(path: str) -> str:
        """路径规范化处理"""
        path = path.replace('\\', '/')
        return path.lower() if ignore_case else path

    # 构建路径匹配字典
    search_patterns = {}
    for folder in input_folders_name:
        full_path = normalize_path(os.path.join(base_folder, folder))
        search_patterns[full_path] = folder

    # 按文件夹结构收集波段信息
    folder_bands = defaultdict(list)
    for key in json_data.keys():
        try:
            # 分解路径和波段信息
            dir_part = normalize_path(os.path.dirname(key))
            band_str = key.split("-band")[-1]
            band_num = int(''.join(filter(str.isdigit, band_str)))
            
            # 匹配文件夹结构
            matched_folder = None
            for pattern in search_patterns:
                if pattern in dir_part:
                    matched_folder = search_patterns[pattern]
                    break
            
            if matched_folder:
                folder_bands[matched_folder].append( (band_num, key) )
                
        except (IndexError, ValueError) as e:
            print(f"[Warning] 跳过无效键值: {key} ({str(e)})")
            continue

    # 按输入顺序和波段号排序
    ordered_results = []
    for folder in input_folders_name:
        if folder not in folder_bands:
            print(f"[Warning] 文件夹未找到数据: {folder}")
            continue
            
        # 按自然数排序波段
        sorted_bands = sorted(folder_bands[folder], key=lambda x: x[0])
        
        # 验证排序结果
        band_nums = [b[0] for b in sorted_bands]
        if band_nums != sorted(band_nums):
            print(f"[Warning] 非连续波段号: {folder} - {band_nums}")
        
        ordered_results.extend( [json_data[b[1]] for b in sorted_bands] )

    # 提取统计信息
    max_values = [item["max"] for item in ordered_results]
    min_values = [item["min"] for item in ordered_results]
    # means = [item["mean"] for item in ordered_results]
    # stds = [item["std"] for item in ordered_results]

    # return max_values, min_values, means, stds
    return max_values, min_values

def change_future_data(folder, band_index):
    band_index -= 1  # 实际指定是从1开始,在代码中是从0开始
    for file in os.listdir(folder):
        file_path = os.path.join(folder, file)
        print(file_path)
        data = np.load(file_path, mmap_mode='c')
        print(data.shape)
        band_data = data[:,:,band_index-1]
        print(band_data*(85-1)+1)
        exit()

def process_label_images(input_folder, output_folder):
    """
    处理文件夹中的灰度 PNG 标签图像。

    Args:
        input_folder (str): 包含原始 PNG 标签图的文件夹路径。
        output_folder (str): 保存处理后的 PNG 标签图的文件夹路径。
    """
    # 创建输出文件夹（如果不存在）
    os.makedirs(output_folder, exist_ok=True)

    # 遍历输入文件夹中的所有文件
    for filename in tqdm(os.listdir(input_folder), unit="files", ncols=100):
        if filename.lower().endswith('.png'):
            input_path = os.path.join(input_folder, filename)
            output_path = os.path.join(output_folder, filename)

            try:
                # 打开图像并确保是灰度图 (L模式)
                img = Image.open(input_path).convert('L')
                img_array = np.array(img)

                # 检查像素值范围是否在 0-7 内 (可选，用于验证)
                # if np.max(img_array) > 7 or np.min(img_array) < 0:
                #     print(f"Warning: {filename} contains pixel values outside the expected 0-7 range.")

                # 创建一个用于存放处理结果的新数组，初始化为原数组的副本
                processed_array = img_array.copy()

                # 规则 1: 将值为 0 的地方改为 255
                processed_array[img_array == 0] = 255

                # 规则 2: 将值为 1 到 7 的地方值减去 1
                # 使用布尔索引选择需要修改的像素
                mask_1_to_7 = (img_array >= 1) & (img_array <= 7)
                processed_array[mask_1_to_7] = img_array[mask_1_to_7] - 1

                # 将 NumPy 数组转换回 PIL 图像，并确保数据类型正确 (uint8)
                processed_img = Image.fromarray(processed_array.astype(np.uint8))

                # 保存处理后的图像
                processed_img.save(output_path)
                # print(f"Processed and saved {filename} to {output_folder}")

            except Exception as e:
                print(f"Error processing {filename}: {e}")

def create_none_severity_label(input_folder, output_folder):
    """
    处理文件夹中的灰度 PNG 标签图像，应用第二个转换规则。

    规则:
    - 3 和 4 都变为 3
    - 5 变为 4
    - 6 变为 5

    Args:
        input_folder (str): 包含原始 PNG 标签图的文件夹路径。
        output_folder (str): 保存处理后的 PNG 标签图的文件夹路径。
    """
    # 创建输出文件夹（如果不存在）
    os.makedirs(output_folder, exist_ok=True)

    print(f"Creating none severity label from '{input_folder}' to '{output_folder}'...")

    # 遍历输入文件夹中的所有文件
    for filename in tqdm(os.listdir(input_folder), unit="files", ncols=100):
        if filename.lower().endswith('.png'):
            input_path = os.path.join(input_folder, filename)
            output_path = os.path.join(output_folder, filename)

            try:
                # 打开图像并确保是灰度图 (L模式)
                img = Image.open(input_path).convert('L')
                img_array = np.array(img)

                # 创建一个用于存放处理结果的新数组，初始化为原数组的副本
                processed_array = img_array.copy()

                # 规则 1: 将值为 3 或 4 的地方都变为 3
                processed_array[(img_array == 3) | (img_array == 4)] = 3

                # 规则 2: 将值为 5 或 6 的地方值减去 1 (5->4, 6->5)
                # 使用布尔索引选择需要修改的像素
                mask_5_to_6 = (img_array >= 5) & (img_array <= 6)
                processed_array[mask_5_to_6] = img_array[mask_5_to_6] - 1

                # 将 NumPy 数组转换回 PIL 图像，并确保数据类型正确 (uint8)
                # 注意：这里的像素值范围会是 0-6 (来自旧的 1-7 和新的 5-6) 以及 3 (来自旧的 3/4)
                # 最终范围会是 0-6 加上可能存在的其他未修改的像素值
                processed_img = Image.fromarray(processed_array.astype(np.uint8))

                # 保存处理后的图像
                processed_img.save(output_path)
                # print(f"  Processed and saved {filename}")

            except Exception as e:
                print(f"  Error processing {filename}: {e}")
        else:
             print(f"  Skipping non-PNG file: {filename}")
    print("Create none severity label complete.")


if __name__ == '__main__':

    base_folder = r"F:/Rice2024/Meiju1/Split_Stretch"  # 数据源
    output_base_folder = r"D:/Rice2024/Meiju1/Datasets/Samples"  # 归一化后的数据集位置
    output_stack_folder = r"G:/Rice2024/Meiju1/Datasets/Stack_Norm_All"  # 最终归一化后的堆叠数据集位置
    output_label_folder = r"F:/Rice2024/Meiju1/Labels/Temp" # Labels 位置
    output_label_severity = r"F:/Rice2024/Meiju1/Labels/Rice_Lodging_Severity"
    ouptut_label_none_severity = r"F:/Rice2024/Meiju1/Labels/Rice_Lodging_None_Severity"

    black_val = 1e-34
    resolution = (84765, 70876)

    # 分文件夹, 对应输入和输出的文件夹
    input_folders_name = [
        r'1_RGB-3',
        r'2_CIs-10\B4-B7',
        r'2_CIs-10\B8-B11',
        r'2_CIs-10\B12-B13',
        r'3_MSI-4',
        r'4_VIs-13\B18-B21',
        r'4_VIs-13\B22-B25',
        r'4_VIs-13\B26-B28',
        r'4_VIs-13\B29-B30',
        r'5_R-GLCM-8\B31-B34',
        r'5_R-GLCM-8\B35-B38',
        r'6_G-GLCM-8\B39-B42',
        r'6_G-GLCM-8\B43-B46',
        r'7_B-GLCM-8\B47-B50',
        r'7_B-GLCM-8\B51-B54',
        r"8_CHM-1"
    ]
    # 不同文件对应的忽略值, TODO: 需要进一步确定, 先不管
    # ignore_values = {
    #     r'RGB' : 0,
    #     r'Multi-spectral' : 1e-34,
    #     r"band8to11": 1e-34,
    #     r'band12to15': 1e-34,
    #     r"band16to18": 1e-34,
    #     r'band19to20': 1e-34,
    #     r'band21to24': 0,
    #     r'band25to28': 0,
    #     r'band29to32': 0,
    #     r'band33to36': 0,
    #     r'band37to40': 0,
    #     r'band41to44': 0,
    #     r'DSM' : -9999,
    # }
    ignore_value = 1e-34
    
    channel_mapping = {
        r'1_RGB-3': [0, 1, 2],
        r'2_CIs-10\B4-B7': [3, 4, 5, 6],
        r'2_CIs-10\B8-B11': [7, 8, 9, 10],
        r'2_CIs-10\B12-B13': [11, 12],
        r'3_MSI-4': [13, 14, 15, 16],
        r'4_VIs-13\B18-B21': [17, 18, 19, 20],
        r'4_VIs-13\B22-B25': [21, 22, 23, 24],
        r'4_VIs-13\B26-B28': [25, 26, 27],
        r'4_VIs-13\B29-B30': [28, 29],
        r'5_R-GLCM-8\B31-B34': [30, 31, 32, 33],
        r'5_R-GLCM-8\B35-B38': [34, 35, 36, 37],
        r'6_G-GLCM-8\B39-B42': [38, 39, 40, 41],
        r'6_G-GLCM-8\B43-B46': [42, 43, 44, 45],
        r'7_B-GLCM-8\B47-B50': [46, 47, 48, 49],
        r'7_B-GLCM-8\B51-B54': [50, 51, 52, 53],
        r"8_CHM-1": [54],
    }




    # 特征通道提取
    features_to_process = [r'1_RGB-3', 
                           r'2_CIs-10\B4-B7', 
                           r"2_CIs-10\B8-B11",
                           r'2_CIs-10\B12-B13', 
                           r'3_MSI-4', 
                           r"4_VIs-13\B18-B21",
                           r"4_VIs-13\B22-B25",
                           r'4_VIs-13\B26-B28', 
                           r'4_VIs-13\B29-B30',
                           r"8_CHM-1",
                           ]
    features_output_dir = r'D:/Rice2024/Meiju1/Datasets/Stack_Norm_RGB-CIs -VIs-CHM' # 特征提取输出文件夹通道

    input_folders = [] # 输入文件夹，每个特征对应一个
    # for i, folder in enumerate(input_folders_name):
    #     input_folders.append(os.path.join(base_folder, folder))
    output_folders = [] # 输出文件夹，每个特征对应一个
    # for i, folder in enumerate(input_folders_name):
    #     output_folders.append(os.path.join(output_base_folder, folder))
    #     os.makedirs(output_folders[i], exist_ok=True)

    # 标签存放位置
    label_folder = r"F:/Rice2024/Meiju1/Labels-shp" # TODO: 还没确定好
    label_name = r"Meiju1_2_Lingtangkou_v5.tif" # 标签名字
    label_path = os.path.join(label_folder, label_name)
    train_val_test_ratio = (0.6, 0.2, 0.2)
    crop_size = 640
    repetition_rate = 0.1
    threshold = 0.9
    skip_log = "logs/Meiju1_All_55.json" # 跳过文件路径, 用于裁剪后 适当跳过黑色背景的图片
    normalize_log = "logs/Meiju1_All_55_Normalized.json" # 归一化文件路径，用于图像归一化，记录参数文件
    normalize_excel_output_path = "logs/Meiju1_All_55_Normalized.xlsx" # 归一化文件路径，用于图像归一化，记录参数文件

    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=str, default="create")
    args = parser.parse_args()

    image_npy_folder = "Image_Npy"  # 转换成.npy文件的下级文件夹名字
    image_tif_folder = "Image_Tif"  # 转换成.tif文件的下级文件夹名字

    start_time = datetime.now()

    if args.run == "get_input_folders":
        folders = []
        for dir, _, files in os.walk(base_folder):
            # print(files)
            if len(files):
                folders.append(dir)

        # print(folders[1:])
        for folder in folders:
            print('r'+'\''+folder+'\''+',')
        print("注意下通道的顺序位置😊")

    elif args.run == "create_preprocess": # 生成跳过记录
        for input_image_folder, output_image_folder  in zip(input_folders, output_folders):
        # 先裁剪后生成
            if 'RGB' in input_image_folder:
                print(f'{skip_log}跳过记录生成....')
                crop_with_repetition_and_save_skip_log(os.path.join(input_image_folder, "Split_Stretch_RGB.tif"),
                                                       os.path.join(output_image_folder, "skip_record_tif_temp"), crop_size=crop_size,
                                                       repetition_rate=repetition_rate, skip_log=skip_log, threshold=threshold, resolution=resolution, black_val=black_val)
            else:
                send_email('生成跳过记录失败...')
                raise "Not RGB"

    elif args.run == "create":  # 切分数据集
        print(input_folders)
        for input_image_folder, output_image_folder  in zip(input_folders, output_folders):
            create_dataset_image(input_image_folder, output_image_folder, resolution=resolution, crop_size=640, repetition_rate=repetition_rate,
                                 skip_log=skip_log)
        send_email(f'numpy数据集制作完成, 用时:{datetime.now() - start_time}', "数据集制作完成..")  

    elif args.run == "move":  # 把npy文件夹移动到父文件夹中
        npy_folders = []
        tif_folders = []
        # 加上Npy路径
        for i, folder in enumerate(output_folders):
            npy_folders.append(os.path.join(folder, image_npy_folder))
            tif_folders.append(os.path.join(folder, image_tif_folder))
        print(output_folders)
        print(npy_folders)

        for target_folder, source_folder in zip(output_folders, npy_folders):
            for file in tqdm(os.listdir(source_folder), desc=f"Move {os.path.basename(source_folder)} Files"):
                shutil.move(os.path.join(source_folder, file), os.path.join(target_folder, file))

        # 删除对应的Folder
        for npy_folder, tif_folder in zip(npy_folders, tif_folders):
            os.removedirs(npy_folder)
            if os.path.exists(tif_folder):
                os.removedirs(tif_folders)
            print(f"Successfully removed {npy_folder} and {tif_folder}")
            # remove_folder(folder)

    elif args.run == "stack":  # 堆叠数据集
        # 加上Npy路径
        # for i, folder in enumerate(output_folders):
        #     output_folders[i] = os.path.join(folder, image_npy_folder)
        print(output_folders)
        stack_npy_files(output_folders, output_stack_folder)
        send_email(f'stack数据集制作完成, 用时:{datetime.now() - start_time}', "数据集制作完成..")  

    elif args.run == "label":
        create_dataset_image(label_folder, output_label_folder, resolution=resolution,crop_size=640, repetition_rate=repetition_rate,
                             tif_shuffix=label_name, shuffix=".png", skip_log=skip_log)
        send_email(f"Label数据集合成, 用时: {datetime.now() - start_time}")

    elif args.run == "label_severity":
        # 生成区分和不区分倒伏程度的标签图片
        process_label_images(output_label_folder, output_label_severity)
        create_none_severity_label(output_label_severity, ouptut_label_none_severity)

    elif args.run == "calc_normlize": # 计算获取归一化信息
        for folder in input_folders:
            for file in os.listdir(folder):
                # if files.endswith(".tif"):
                if file.startswith("Split_Stretch_") and file.endswith(".tif") and "CHM" not in file:
                    base_name = os.path.basename(folder)
                    print(f'{base_name}: {file}')
                    calc_mean_and_std(os.path.join(folder, file), label_path=label_path, calc_logs=normalize_log, ignore_value=ignore_value, excel_output_path=normalize_excel_output_path)
        send_email("计算归一化指数完成")

    elif args.run == 'normlize_print':  # 打印归一化信息
        max_vals, min_vals = get_band_statistics(base_folder, input_folders_name, json_path=normalize_log)
        print("Max values: ", max_vals)
        print("Min values: ", min_vals)
        # print("Means: ", means)
        # print("Stds: ", stds)
        print("bands num", len(max_vals))

    elif args.run == 'normlize': # 进行归一化
        max_vals, min_vals = get_band_statistics(base_folder, input_folders_name, json_path=normalize_log)
        normalize_dataset(output_stack_folder, max_value=max_vals, min_value=min_vals)
        send_email("归一化转换完成")

    elif args.run == "split":  # 切分数据集成 train, val, test
        train_ratio, valid_ratio, test_ratio = train_val_test_ratio
        output_label_folder = os.path.join(output_label_folder, image_npy_folder)
        # 重命名文件
        rename_files(image_folder=output_stack_folder, label_folder=output_label_folder, label_end_with='.png')
        # 切分数据集
        split_dataset(output_stack_folder, output_label_folder, train_ratio=train_ratio, val_ratio=valid_ratio, test_ratio=test_ratio, labels_suffix=".png")
        send_email(f"切分数据集, 用时: {datetime.now() - start_time}")
    
    elif args.run == "feature":   # 提取指定的特征到新文件夹上
        # Define the base input directory to walk through
        base_input_folder = r"/data/Rice2024/ALL"
        # Define the base output directory for the extracted features
        base_output_folder = r'/data/Rice2024/RGB_Color_Spectra_Texture'

        # Define the features to process (same for all data subdirectories)
        features_to_process = [
            r'1_RGB-3',
            r'2_CIs-10\B4-B7',
            r'2_CIs-10\B8-B11',
            r'2_CIs-10\B12-B13',
            r'3_MSI-4',
            r'4_VIs-13\B18-B21',
            r'4_VIs-13\B22-B25',
            r'4_VIs-13\B26-B28',
            r'4_VIs-13\B29-B30',
            r'5_R-GLCM-8\B31-B34',
            r'5_R-GLCM-8\B35-B38',
            r'6_G-GLCM-8\B39-B42',
            r'6_G-GLCM-8\B43-B46',
            r'7_B-GLCM-8\B47-B50',
            r'7_B-GLCM-8\B51-B54',
            # r"8_CHM-1"
            ]

        print(f"--- Starting feature extraction across {base_input_folder} ---")
        print(f"Saving extracted features to base directory: {base_output_folder}")


        MAX_WORKERS = 24
        # Walk through all subdirectories starting from the base input folder
        # os.walk yields tuples (dirpath, dirnames, filenames)
        # dirpath is the current directory being walked (e.g., /root/datasets/ALL/train/scene1)
        # dirnames is a list of names of the subdirectories in dirpath
        # filenames is a list of names of the files in dirpath
        for dirpath, dirnames, filenames in os.walk(base_input_folder):
            # We are only interested in directories that actually contain files (data subdirectories)
            # This check prevents processing parent directories like /root/datasets/ALL/train or /root/datasets/ALL
            # unless they also happen to directly contain data files, which is less typical
            if filenames:
                print(f"\nProcessing data directory: {dirpath}")

                # Calculate the path of the current data directory relative to the base input folder
                # e.g., if dirpath is /root/datasets/ALL/train/scene1, relative_path will be train/scene1
                relative_path = os.path.relpath(dirpath, base_input_folder)

                # Construct the corresponding output subdirectory path by joining the base output folder
                # with the relative path.
                # e.g., os.path.join(/root/data_temp/CHM, train/scene1) -> /root/data_temp/CHM/train/scene1
                output_subdir = os.path.join(base_output_folder, relative_path)
                print(f"Saving features to: {output_subdir}")
                
                # Ensure the output directory structure exists before trying to save files into it
                # exist_ok=True prevents errors if the directory already exists
                os.makedirs(output_subdir, exist_ok=True)

                # # Call the function to split/extract features
                # # It takes the input directory (dirpath), channel mapping, output directory, and features list
                split_npy_files_threaded(dirpath, channel_mapping,
                                output_subdir, features_to_process, max_threads=MAX_WORKERS)

    elif args.run == "change_future":
        change_future_data(output_stack_folder, 10)

    elif args.run == "visualize":
        # print(input_folders)
        for folder in input_folders:
            # if "B4-B7" in folder:
            for file in os.listdir(folder):
                file_path = os.path.join(folder, file)
                print(file)
                # print(file)
                output_folder = folder.replace("F", "G")
                dataset = gdal.Open(file_path)
                bands = dataset.RasterCount
                for band_index in range(1, bands+1):
                    print(f"band index is {band_index}")
                    visualize_feature(file_path, label_path, ignore_value, band_index, output_folder)
                print(f"run one feature time is {datetime.now() - start_time}")


    elif args.run == "check":
        # Define the base input directory to walk through
        base_input_folder = r"/data/Rice2024/ALL"
        # Define the base output directory for the extracted features
        base_output_folder = r'/data/Rice2024/RGB_Color_Spectra_Texture'

        check_folder = "train/data"

        input_folder = os.path.join(base_input_folder, check_folder)
        output_folder = os.path.join(base_output_folder, check_folder)

        check_channels = list(range(0, 54))
        print("channels number is", len(set(check_channels)))
        print(check_channels)
        for i, file in enumerate(os.listdir(input_folder)):
            all_file_path = os.path.join(input_folder, file)
            check_file_path = os.path.join(output_folder, file)
            all_file_data = np.load(all_file_path, mmap_mode='c')
            check_file_data = np.load(check_file_path, mmap_mode='c')
            print(i, all_file_data[:,:,check_channels].shape, check_file_data.shape)
            assert np.array_equal(all_file_data[:,:,check_channels], check_file_data)
            if i == 100:
                break
            # for i in check_channels:
                # assert np.array_equal(all_file_data[i], check_file_data[i])
        print("check successfully..")

    print('run time is {}'.format(datetime.now()-start_time))
    






