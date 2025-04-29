import sys
sys.path.append('./')
from src.processing.crop import crop_with_repetition
from src.processing.background_black import set_background_black_folder, delete_almost_all_black_tiffs
from src.processing.convert import convert_three_bands_tif_to_png, convert_one_band_tif_to_png, convert_15_bands_tif_to_npy
from src.processing.split import split_dataset
from src.utils.email_utils import send_email
from src.utils.file_io import remove_folder, rename_files
import os
import argparse
from tqdm import tqdm
from datetime import datetime


def main(all_dataname: list, input_data_dir, input_data_end_with ,input_label_folder_name, label_end_with, output_dir, output_name,
         crop_size=640, repetition_rate=0.4, train_val_test_ratio=(0.6, 0.2, 0.2), threshold=0.9):
    
    start_time = datetime.now()

    output_dir = os.path.join(output_dir, output_name)
    images_origin_name = r"images_origin"
    images_black_name = r"images_black"
    images_png_name = r"images"
    labels_tif_name = r"labels_tif"
    labels_png_name = r"labels"

    images_origin_path = os.path.join(output_dir, images_origin_name)
    images_black_path = os.path.join(output_dir, images_black_name)
    images_png_path = os.path.join(output_dir, images_png_name)
    labels_tif_path = os.path.join(output_dir, labels_tif_name)
    labels_png_path = os.path.join(output_dir, labels_png_name)
    os.makedirs(images_origin_path, exist_ok=True)
    os.makedirs(labels_tif_path, exist_ok=True)
    os.makedirs(labels_png_path, exist_ok=True)
    os.makedirs(images_black_path, exist_ok=True)
    os.makedirs(images_png_path, exist_ok=True)


    print('cropping dataset...')
    for dataname in all_dataname:
        original_image_path = os.path.join(input_data_dir, dataname)
        original_label_path = os.path.join(input_data_dir, dataname, input_label_folder_name)
        for file_name in tqdm(os.listdir(original_image_path)):
            if file_name.endswith(input_data_end_with):
                file_path = os.path.join(original_image_path, file_name)
                print(f'now cropping image {file_path}.')
                crop_with_repetition(file_path, images_origin_path, crop_size, repetition_rate)
        # 裁剪标签数据集
        for file_name in tqdm(os.listdir(original_label_path)):
            if file_name.endswith(label_end_with):
                file_path = os.path.join(original_label_path, file_name)
                print(f'now cropping image {file_path}.')
                crop_with_repetition(file_path, labels_tif_path, crop_size, repetition_rate)

    # 删除几乎全黑的图片, 删除超过50%的像素点为黑色的图片
    delete_almost_all_black_tiffs(images_origin_path, labels_tif_path, threshold=threshold)

    # 重命名文件下的文件按顺序
    rename_files(images_origin_path, labels_tif_path)

    # 将tif背景设置为黑色
    set_background_black_folder(images_origin_path, labels_tif_path, images_black_path)
    # 删除image_origin
    remove_folder(images_origin_path)

    # mask tif转png
    convert_one_band_tif_to_png(labels_tif_path, labels_png_path)
    # 删除label tif文件
    remove_folder(labels_tif_path)
    # image 三通道tif转png
    convert_15_bands_tif_to_npy(images_black_path, images_png_path)
    remove_folder(images_black_path)

    # 分割数据集
    train_ratio, valid_ratio, test_ratio = train_val_test_ratio
    split_dataset(images_png_path, labels_png_path, train_ratio=train_ratio, val_ratio=valid_ratio, test_ratio=test_ratio, labels_suffix=".png")
    

    print("run time is {}".format(datetime.now()-start_time))
    # 发送邮箱
    send_email(f"数据集制作, 用时:{datetime.now()-start_time}")



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # all_dataname = ['20240912-Rice-M3M-50m-Lingtangkou-RGB', '20240913-Rice-M3M-50m-Meiju-1-RGB', '20240911-Rice-M3M-50m-Meiju-2-RGB', '20240913-Rice-M3M-50m-Xipo-1-RGB', '20240912-Rice-M3M-50m-Xipo-2-RGB']
    parser.add_argument("--all_dataname", type=list, default=['L', 'M'], help="all data name")
    parser.add_argument("--input_data_dir", type=str, default=r"H:\1", help="input data dir")
    parser.add_argument("--input_data_end_with", type=str, default="Feature.tif", help="input data end with")
    parser.add_argument("--input_label_folder_name", type=str, default="Labels", help="input label folder name")
    parser.add_argument("--label_end_with", type=str, default="Label_no_abnormal_rice_mild_Out.tif", help="label end with version")
    parser.add_argument("--output_dir", type=str, default=r"H:\1\datasets", help="output dir")
    parser.add_argument("--output_name", type=str, default="no_abnormal_rice_mild", help="output name")

    # 一般不用改
    parser.add_argument("--crop_size", type=int, default=128, help="crop size")
    parser.add_argument("--repetition_rate", type=float, default=0.0, help="repetition rate")
    parser.add_argument("--train_val_test_ratio", type=tuple, default=(0.6, 0.2, 0.2), help="train val test ratio")
    parser.add_argument("--threshold", type=float, default=0.9, help="threshold")
    now_version = "v1"

    # TODO: 起提醒作用
    class_num = 6
    class_names = {0: 'road', 1: 'sugarcane', 2: 'rice_normal', 3:'rice_severe', 4:'rice_mild', 5:"weed"}
    # class_names = {0: 'road', 1: 'sugarcane', 2: 'rice_normal', 3:'rice_lodging'}
    assert class_num == len(class_names), "class num not equal to class names"

    args = parser.parse_args()

    # 获取今天日期
    today = datetime.now().date().strftime('%m.%d')
    train_ratio, valid_ratio, test_ratio = args.train_val_test_ratio
    args.output_name = f"{args.output_name}-{today}-{class_num}-{args.crop_size}-{train_ratio}-{valid_ratio}-{test_ratio}-{now_version}"
    main(**vars(args))




