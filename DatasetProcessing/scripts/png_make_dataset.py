import sys
sys.path.append('./')
from src.processing.crop import crop_with_repetition, crop_with_repetition_png
from src.processing.background_black import set_background_black_folder, delete_almost_all_black_tiffs, delete_or_set_background_black
from src.processing.convert import convert_three_bands_tif_to_png, convert_one_band_tif_to_png, convert_one_band_tif_to_png_file
from src.processing.split import split_dataset
from src.utils.email_utils import send_email
from src.utils.file_io import remove_folder, rename_files
import os
import argparse
from tqdm import tqdm
from datetime import datetime
import cv2
from pathlib import Path
import numpy as np
import time # 引入 time 库用于计时


def process_and_remap_masks(input_mask_dir, output_mask_dir,
                              original_ignore_value=0, final_ignore_value=255,
                              class_mapping=None):
    """
    读取分割标签图片，将原始忽略值映射到新的忽略值，
    并根据类别映射重新映射类别 ID，保留最终忽略值不变，
    并将修改后的图片保存到新的目录下。

    Args:
        input_mask_dir (str): 包含原始标签图片的目录路径。
        output_mask_dir (str): 保存修改后标签图片的目录路径。
        original_ignore_value (int): 原始用作忽略区域的像素值 (默认为 0)。
        final_ignore_value (int): 目标用作忽略区域的像素值 (默认为 255)。
        class_mapping (dict, optional): 类别 ID 映射字典 (例如 {1: 0, 2: 1, ..., 7: 6})。
                                        如果不提供，只处理忽略值。
    """
    input_dir = Path(input_mask_dir)
    output_dir = Path(output_mask_dir)

    # 创建输出目录，如果不存在，则创建父目录
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"扫描原始标签目录: {input_dir}")
    print(f"修改后标签将保存到: {output_dir}")
    print(f"原始忽略值 '{original_ignore_value}' 将映射到最终忽略值 '{final_ignore_value}'")
    if class_mapping:
        print(f"应用类别映射: {class_mapping}")
    else:
        print("未提供类别映射，只处理忽略值。")

    processed_count = 0
    skipped_count = 0

    # --- 构建高效的查找表 ---
    # 我们需要一个查找表来同时处理忽略值的改变和类别的映射
    # 查找表的大小需要至少覆盖到可能出现的原始最大像素值
    # 假设原始标签像素值在 0-255 之间 (uint8)
    lookup_table_size = 256
    # 如果原始忽略值或类别映射中的原始 ID 或最终忽略值超出了 255，需要更大的表
    if original_ignore_value >= lookup_table_size or final_ignore_value >= lookup_table_size:
        lookup_table_size = max(original_ignore_value, final_ignore_value) + 1
    if class_mapping:
        max_original_id = max(class_mapping.keys()) if class_mapping else 0
        max_new_id = max(class_mapping.values()) if class_mapping else 0
        lookup_table_size = max(lookup_table_size, max_original_id + 1, max_new_id + 1)


    # 初始化查找表：默认情况下，像素值映射到自身
    # 使用 int64 以确保在处理较大值时安全，尽管最终会转回 uint8
    lookup_table = np.arange(lookup_table_size, dtype=np.int64)

    # --- 应用转换规则到查找表 ---

    # 1. 应用类别映射 (如果提供)
    if class_mapping:
        for old_id, new_id in class_mapping.items():
            if old_id >= lookup_table_size:
                 print(f"警告: 原始类别 ID {old_id} 超出查找表范围 ({lookup_table_size})。它将不会被映射。")
                 continue
            # 确保新的类别 ID 不会覆盖原始的忽略值或最终的忽略值 (这是一种健壮性检查)
            # if new_id == original_ignore_value:
            #      print(f"错误: 新类别 ID {new_id} 与原始忽略值冲突！请检查映射规则。")
            #      return
            if new_id == final_ignore_value:
                 print(f"错误: 新类别 ID {new_id} 与最终忽略值冲突！请检查映射规则。")
                 return

            lookup_table[old_id] = new_id

    # 2. 将原始忽略值映射到最终忽略值
    # 这步在类别映射之后执行，确保原始忽略值不会被任何类别映射覆盖
    # 并且它将所有原始的 original_ignore_value 都变成 final_ignore_value
    if original_ignore_value >= lookup_table_size:
         print(f"错误: 原始忽略值 {original_ignore_value} 超出查找表范围 ({lookup_table_size})！请检查标签数据类型。")
         return # 无法处理，退出
    lookup_table[original_ignore_value] = final_ignore_value


    # 3. 确保最终忽略值在查找表中映射到自身 (如果它在查找表范围内)
    # 这确保在应用映射后，任何已经是 final_ignore_value 的像素值不会改变
    if final_ignore_value < lookup_table_size:
        lookup_table[final_ignore_value] = final_ignore_value
    # 注意：如果 final_ignore_value >= lookup_table_size，直接索引 lookup_table[mask] 会出错，
    # 但我们已经通过调整 lookup_table_size 尽量避免这种情况了。


    # 遍历输入目录，包括子目录
    for root, _, files in os.walk(input_dir):
        # 构建当前子目录在输出路径下的对应目录
        relative_path = Path(root).relative_to(input_dir)
        current_output_subdir = output_dir / relative_path
        current_output_subdir.mkdir(parents=True, exist_ok=True) # 创建对应的输出子目录

        for file in files:
            # 假设标签文件是图片格式
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')):
                input_path = Path(root) / file
                output_path = current_output_subdir / file # 在对应的输出子目录下保存，文件名不变

                try:
                    # 读取标签图片为灰度图 (保证是单通道)
                    mask = cv2.imread(str(input_path), cv2.IMREAD_GRAYSCALE)

                    if mask is None:
                        print(f"警告: 无法读取标签文件: {input_path}. 跳过.")
                        skipped_count += 1
                        continue

                    # 确保 mask 是 numpy 数组
                    mask = np.array(mask)

                    # --- 应用查找表映射 ---
                    # 检查 mask 中的最大值是否在查找表范围内 (除了原始忽略值，因为已经被特殊处理)
                    # mask_values_to_check = mask[mask != original_ignore_value]
                    # if mask_values_to_check.size > 0 and mask_values_to_check.max() >= lookup_table_size:
                    #      print(f"错误: 标签文件 {input_path} 中的像素值 ({mask_values_to_check.max()}) 超出查找表范围 ({lookup_table_size})。无法正确映射。跳过.")
                    #      skipped_count += 1
                    #      continue

                    # 更简单的检查，确保所有 mask 中的值都在查找表索引范围内
                    if mask.max() >= lookup_table_size or mask.min() < 0:
                         print(f"错误: 标签文件 {input_path} 中的像素值 ({mask.min()}-{mask.max()}) 超出查找表范围 ({lookup_table_size})。无法正确映射。跳过.")
                         skipped_count += 1
                         continue


                    # 使用 numpy 查找表进行高效映射
                    modified_mask = lookup_table[mask]

                    # --- 确保数据类型适合保存为图片 ---
                    # 假设你的新类别 ID 和最终 ignore_value 都在 [0, 255] 范围内
                    if not np.issubdtype(modified_mask.dtype, np.integer) or modified_mask.max() > 255 or modified_mask.min() < 0:
                         print(f"警告: 修改后标签像素值超出 0-255 范围 ({modified_mask.min()}-{modified_mask.max()}), 数据类型 {modified_mask.dtype}. 保存为标准图片格式可能失真，请检查输出.")
                         # 尝试转换为 uint8，会进行截断
                         modified_mask = np.clip(modified_mask, 0, 255).astype(np.uint8)
                    else:
                         # 转换为 uint8
                         modified_mask = modified_mask.astype(np.uint8)


                    # 保存修改后的标签图片
                    # 对于标签图，推荐保存为 PNG 以避免 JPEG 压缩引起的失真
                    cv2.imwrite(str(output_path), modified_mask)

                    processed_count += 1
                    if processed_count % 100 == 0:
                        print(f"已处理 {processed_count} 张图片...")

                except Exception as e:
                    print(f"处理文件 {input_path} 时发生错误: {e}. 跳过.")
                    skipped_count += 1


    print("-" * 30)
    print(f"处理完成.")
    print(f"成功处理数量: {processed_count}")
    print(f"因错误跳过数量: {skipped_count}")

def main(all_dataname: list, input_data_dir, input_data_end_with ,input_label_folder_name, label_end_with, output_dir, output_name,
         crop_size=640, repetition_rate=0.4, train_val_test_ratio=(0.6, 0.2, 0.2), threshold=0.9):
    # 获取运行的时间
    start_time = datetime.now()
    output_dir = os.path.join(output_dir, output_name)
    images_origin_name = r"images_origin" # 裁剪后的原始数据集
    images_name = r"images"   # 标黑数据集
    labels_origin_name = r"labels_origin"
    labels_name = r"labels"       # 裁剪后的标签数据集

    images_origin_path = os.path.join(output_dir, images_origin_name)
    images_path = os.path.join(output_dir, images_name)
    labels_path = os.path.join(output_dir, labels_name)
    labels_origin_path = os.path.join(output_dir, labels_origin_name)
    os.makedirs(images_origin_path, exist_ok=True)
    os.makedirs(images_path, exist_ok=True)
    os.makedirs(labels_path, exist_ok=True)
    os.makedirs(labels_origin_path, exist_ok=True)

    print('cropping dataset...')
    for dataname in all_dataname:
        original_image_path = os.path.join(input_data_dir, dataname)
        original_label_path = os.path.join(input_data_dir, dataname, input_label_folder_name)
        # for file_name in tqdm(os.listdir(original_image_path)):
        #     if file_name.endswith(input_data_end_with):
        #         file_path = os.path.join(original_image_path, file_name)
        #         print(f'now cropping image {file_path}.')
        #         # crop_with_repetition_png(file_path, images_origin_path, crop_size, repetition_rate)
        #         crop_with_repetition(file_path, images_origin_path, crop_size, repetition_rate)
                
        #         exit()
        # 裁剪标签数据集
        for file_name in tqdm(os.listdir(original_label_path)):
            if file_name.endswith(label_end_with):
                # 转格式
                file_path = os.path.join(original_label_path, file_name)
                # png_path = os.path.join(original_label_path, file_name.replace(".tif", ".png"))
                # convert_one_band_tif_to_png_file(file_path, png_path)
                print(f'now cropping image {file_path}.')
                # crop_with_repetition_png(file_path, labels_path, crop_size, repetition_rate)
                crop_with_repetition(file_path, labels_origin_path, crop_size, repetition_rate)

    # # 将tif背景设置为黑色
    delete_or_set_background_black(images_origin_path, labels_origin_path, images_path, labels_path, threshold=threshold)
    # 重命名文件下的文件按顺序
    rename_files(images_path, labels_path, label_end_with='.png')
    # 删除image_origin
    remove_folder(images_origin_path)
    remove_folder(labels_origin_path)

    # 分割数据集
    train_ratio, valid_ratio, test_ratio = train_val_test_ratio
    split_dataset(images_path, labels_path, train_ratio=train_ratio, val_ratio=valid_ratio, test_ratio=test_ratio, labels_suffix=".png")

    print(f"finished at {datetime.now() - start_time}")
    # 发送邮箱
    send_email(f"数据集制作, 用时：{datetime.now() - start_time}")




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # all_dataname = ['20240912-Rice-M3M-50m-Lingtangkou\map', '20240911-Rice-M3M-50m-Meiju-1', '20240911-Rice-M3M-50m-Meiju-2/map', '20240913-Rice-M3M-50m-Xipo-1', '20240912-Rice-M3M-50m-Xipo-2']
    # parser.add_argument("--all_dataname", type=list, default=['20240911-Rice-M3M-50m-Meiju-1', '20240911-Rice-M3M-50m-Meiju-2/map', '20240912-Rice-M3M-50m-Lingtangkou\map'], help="all data name")
    parser.add_argument("--all_dataname", type=list, default=['20240911-Rice-M3M-50m-Meiju-2/map'], help="all data name")
    # parser.add_argument("--all_dataname", type=list, default=['20240912-Rice-M3M-50m-Lingtangkou\map'], help="all data name")
    # parser.add_argument("--all_dataname", type=list, default=['20240911-Rice-M3M-50m-Meiju-2/map'], help="all data name")
    parser.add_argument("--input_data_dir", type=str, default=r"G:/Data/DJITerra_Export_Rice2024_UAV-RGB-MSI", help="input data dir")
    parser.add_argument("--input_data_end_with", type=str, default="rgb_u8_v2.tif", help="input data end with")
    parser.add_argument("--input_label_folder_name", type=str, default="Labels", help="input label folder name")
    parser.add_argument("--label_end_with", type=str, default="Meiju1_2_Lingtangkou_v5.tif", help="label end with version")
    # parser.add_argument("--output_dir", type=str, default=r"E:\Code\RiceLodging\datasets\DJ\Meiju1_2_Lingtangkou", help="output dir")
    # parser.add_argument("--output_name", type=str, default="abnormal", help="output name")
    parser.add_argument("--output_dir", type=str, default=r"E:/Code/RiceLodging/datasets/DJ/Meiju2_YOLO_Test", help="output dir")
    parser.add_argument("--output_name", type=str, default="YOLO", help="output name")
    # 一般不用改
    parser.add_argument("--crop_size", type=int, default=640, help="crop size")
    parser.add_argument("--repetition_rate", type=float, default=0.1, help="repetition rate")
    parser.add_argument("--train_val_test_ratio", type=tuple, default=(0.6, 0.2, 0.2), help="train val test ratio")
    parser.add_argument("--threshold", type=float, default=0.9, help="threshold")
    now_version = "v5"

    # TODO: 起提醒作用
    class_num = 7
    class_names = {0: 'road', 1: 'sugarcane', 2: 'rice_normal', 3:'rice_severe', 4:'rice_mild', 5:'weed', 6:'abnormal'}
    # class_names = {0: 'road', 1: 'sugarcane', 2: 'rice_normal', 3:'rice_lodging'}
    assert class_num == len(class_names), "class num not equal to class names"

    args = parser.parse_args()

    # 获取今天日期
    today = datetime.now().date().strftime('%m.%d')
    train_ratio, valid_ratio, test_ratio = args.train_val_test_ratio
    args.output_name = f"{args.output_name}-{today}-{class_num}-{args.crop_size}-{args.repetition_rate}-{train_ratio}-{valid_ratio}-{test_ratio}-{now_version}"
    
    main(**vars(args))





