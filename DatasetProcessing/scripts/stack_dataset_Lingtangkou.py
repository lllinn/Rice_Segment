import sys
sys.path.append('./')
from src.processing.convert import tif_to_npy_folder
from src.processing.background_black import smart_image_converter
from src.processing.stack_data import stack_npy_files, split_npy_files
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
from calc_mean_std import calc_mean_and_std
import shutil
from stack_dataset_Meiju1 import process_label_images, create_none_severity_label

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
            npy_data[:,:,:-1] = (npy_data[:,:,:-1] - min_value[np.newaxis, np.newaxis, :]) / range_val[np.newaxis, np.newaxis, :]
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


if __name__ == '__main__':

    base_folder = r"E:/Rice2024/Lingtangkou/Split_Stretch"  # 数据源
    output_base_folder = r"E:/Rice2024/Lingtangkou/Datasets/Samples"  # 归一化后的数据集位置
    output_stack_folder = r"D:/Rice2024/Lingtangkou/Datasets/Stack_Norm_All"  # 最终归一化后的堆叠数据集位置
    output_label_folder = r"E:/Rice2024/Lingtangkou/Labels/Temp" # Labels 位置
    output_label_severity = r"E:/Rice2024/Lingtangkou/Labels/Rice_Lodging_Severity"
    ouptut_label_none_severity = r"E:/Rice2024/Lingtangkou/Labels/Rice_Lodging_None_Severity"

    black_val = 1e-34   # 用于裁剪时跳过的值, 默认值是0，现在改成1e-34
    resolution = (61858, 35951)

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
    features_output_dir = r'C:/Rice2024/Lingtangkou/Datasets/Stack_Norm_RGB-CIs-VIs-CHM' # 特征提取输出文件夹通道

    input_folders = [] # 输入文件夹，每个特征对应一个
    for i, folder in enumerate(input_folders_name):
        input_folders.append(os.path.join(base_folder, folder))
    output_folders = [] # 输出文件夹，每个特征对应一个
    for i, folder in enumerate(input_folders_name):
        output_folders.append(os.path.join(output_base_folder, folder))
        os.makedirs(output_folders[i], exist_ok=True)

    # 标签存放位置
    label_folder = r"E:/Rice2024/Lingtangkou/Labels-shp" # TODO: 还没确定好
    label_name = r"Meiju1_2_Lingtangkou_v5.tif" # 标签名字
    label_path = os.path.join(label_folder, label_name)
    train_val_test_ratio = (0.6, 0.2, 0.2)
    crop_size = 640
    repetition_rate = 0.1
    threshold = 0.9
    skip_log = "logs/Lingtangkou_All_55.json" # 跳过文件路径, 用于裁剪后 适当跳过黑色背景的图片
    normalize_log = "logs/Lingtangkou_All_55_Normalized.json" # 归一化文件路径，用于图像归一化，记录参数文件
    normalize_excel_output_path = "logs/Lingtangkou_All_55_Normalized.xlsx" # 归一化文件路径，用于图像归一化，记录参数文件

    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=str, default="create")
    args = parser.parse_args()

    image_npy_folder = "Image_Npy"  # 转换成.npy文件的下级文件夹名字


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

    elif args.run == "create":
        print(input_folders)
        for input_image_folder, output_image_folder  in zip(input_folders, output_folders):
            create_dataset_image(input_image_folder, output_image_folder, resolution=resolution, crop_size=640, repetition_rate=repetition_rate,
                                 skip_log=skip_log)
        send_email(f'numpy数据集制作完成, 用时:{datetime.now() - start_time}', "数据集制作完成..")  

    elif args.run == "move":  # 把npy文件夹移动到父文件夹中
        npy_folders = []
        # 加上Npy路径
        for i, folder in enumerate(output_folders):
            npy_folders.append(os.path.join(folder, image_npy_folder))
        print(output_folders)
        print(npy_folders)
        for target_folder, source_folder in zip(output_folders, npy_folders):
            if not os.path.exists(source_folder):
                continue
            for file in tqdm(os.listdir(source_folder), desc=f"Move {os.path.basename(source_folder)} Files"):
                shutil.move(os.path.join(source_folder, file), os.path.join(target_folder, file))

        # 删除对应的Folder
        for folder in npy_folders:
            if not os.path.exists(folder):
                continue
            os.removedirs(folder)
            print(f"Successfully removed {folder}")
            # remove_folder(folder)

    elif args.run == "stack":
        # 加上Npy路径
        # for i, folder in enumerate(output_folders):
        #     output_folders[i] = os.path.join(folder, image_npy_folder)
        print(output_folders)
        stack_npy_files(output_folders, output_stack_folder)
        send_email(f'stack数据集制作完成, 用时:{datetime.now() - start_time}', "数据集制作完成..")  

    elif args.run == "label":
        create_dataset_image(label_folder, output_label_folder, resolution=resolution, crop_size=640, repetition_rate=repetition_rate,
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
                if file.startswith("Split_Stretch_") and file.endswith(".tif") and "DSM" not in file:
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

    elif args.run == "split":
        train_ratio, valid_ratio, test_ratio = train_val_test_ratio
        output_label_folder = os.path.join(output_label_folder, image_npy_folder)
        # 重命名文件
        rename_files(image_folder=output_stack_folder, label_folder=output_label_folder, label_end_with='.png')
        # 切分数据集
        split_dataset(output_stack_folder, output_label_folder, train_ratio=train_ratio, val_ratio=valid_ratio, test_ratio=test_ratio, labels_suffix=".png")
        send_email(f"切分数据集, 用时: {datetime.now() - start_time}")
    
    elif args.run == "feature":
        for dir, name, files in os.walk(output_stack_folder):
            # print(dir, name)
            if len(files):
                print(dir)
                print(os.path.join(features_output_dir, os.path.basename(dir)))
                split_npy_files(dir, channel_mapping, 
                                os.path.join(features_output_dir, os.path.basename(dir)), features_to_process)
            # if not files:
            #     continue
        # split_npy_files(output_stack_folder, channel_mapping, features_output_dir, features_to_process)

    print('run time is {}'.format(datetime.now()-start_time))



