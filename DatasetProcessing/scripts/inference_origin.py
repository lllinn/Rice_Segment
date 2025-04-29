## 实现模型推理后，拼接回原图
# 1. 运行create_dataset.py文件, 需要将__name__改为__test__, 然后分别改变量：数据集位置，输出的位置：用途制作数据集用于推理拼接的原始数据集
# 得出：Tif文件夹（原始的tif文件），Png文件夹(用于推理的原图)，Label_True文件夹(真实对应的标签), Black文件夹(涂黑后的tif文件)
# 2. 进行模型推理中的train.py, 需要将__name__改为__visualize_all_label__，然后分别改变量：DATA_DIR: 用途用于推理拼接
# 得出：Label_Png文件夹(推理后的标签图片)
# 3. 对推理后的图片进行拼接, 运行png2tif.py文件, 需要将__name__改为__main__, 然后改变量: all_folder
# 得出: Label文件夹(得到的tif标签文件夹)
# 4. 最后运行merge_tif.py文件, 需要将__name__改为__main__, 然后改变量: folder
# 得出: Label_Merge文件夹(最终拼接好的tif标签文件)
# 改完3和4后，可以使用命令`python png2tif.py && python merge_tif.py`一起运行3和4
import sys
sys.path.append('./')
from src.processing.crop import crop_without_repetition
from src.processing.background_black import set_background_black_folder
from src.processing.convert import convert_three_bands_tif_to_png, png2tif, convert_one_band_tif_to_png
from src.processing.merge import merge_tif
from src.utils.email_utils import send_email
from src.utils.file_io import remove_folder
import os
import argparse
from datetime import datetime

def create_dataset(image_path, label_path, output_folder):

    tif_folder = f"Tif"
    png_folder = f"Png"
    label_folder = f"Label_True"
    black_folder = f"Black"
    label_png_folder = f"Label_True_Png"
    tif_folder = os.path.join(output_folder, tif_folder)
    png_folder = os.path.join(output_folder, png_folder)
    label_folder = os.path.join(output_folder, label_folder)
    black_folder = os.path.join(output_folder, black_folder)
    label_png_folder = os.path.join(output_folder, label_png_folder)

    # 裁剪为x, y
    crop_without_repetition(image_path, tif_folder, 640)
    
    # 裁剪为x, y
    crop_without_repetition(label_path, label_folder, 640)

    # 将tif背景设置为黑色
    set_background_black_folder(tif_folder, label_folder, black_folder)

    # 转为png
    convert_one_band_tif_to_png(label_folder, label_png_folder)
    # 转为png
    convert_three_bands_tif_to_png(black_folder, png_folder)


    # # 裁剪为x, y
    # crop_without_repetition(image_path, tif_folder, 640)
    
    # 裁剪为x, y
    # crop_without_repetition(label_path, label_folder, 640)

    # # 将tif背景设置为黑色
    # # set_background_black_folder(tif_folder, label_folder, black_folder)

    # # 转为png
    # convert_one_band_tif_to_png(label_folder, label_png_folder)
    # # 转为png
    # convert_three_bands_tif_to_png(tif_folder, png_folder)
    

# 模型推理
def predict():
    pass


def png2tif_and_merge(output_folder):
    """
    将PNG格式的图片转换为TIFF格式并合并。
    
    Args:
        output_folder (str): 输出文件夹路径。
    
    Returns:
        None
    
    """
    label_png_folder = "Label_Png"
    label_folder = "Label"
    label_True_folder = "Label_True"
    label_png_folder = os.path.join(output_folder, label_png_folder)
    label_folder = os.path.join(output_folder, label_folder)
    label_True_folder = os.path.join(output_folder, label_True_folder)
    os.makedirs(label_folder, exist_ok=True)

    png2tif(label_png_folder, label_True_folder, label_folder)

    merge_folder = "Label_Merge"
    merge_folder = os.path.join(output_folder, merge_folder)
    os.makedirs(merge_folder, exist_ok=True)

    merge_path = os.path.join(merge_folder, "merge.tif")
    merge_tif(label_folder, merge_path)
    

def delete_unused_folders(output_folder):
    """
    删除没用的folder, 用于重复优化。
    
    Args:
        output_folder (str): 要检查的输出文件夹路径。
    
    Returns:
        None
    
    """
    folders = ["Label_Png", "Label", "Label_Merge"]
    for folder in folders:
        folder_path = os.path.join(output_folder, folder)
        remove_folder(folder_path)



if __name__ == '__main__':


    original_folder = r"G:/Data/DJITerra_Export_Rice2024_UAV-RGB-MSI"
    # all_dataname = ['20240912-Rice-M3M-50m-Lingtangkou\map', '20240911-Rice-M3M-50m-Meiju-1', '20240911-Rice-M3M-50m-Meiju-2/map', '20240913-Rice-M3M-50m-Xipo-1', '20240912-Rice-M3M-50m-Xipo-2']
    # TODO: 这里需要改成自己的数据集名称
    dataname = '20240911-Rice-M3M-50m-Meiju-2/map'
    original_image_path = os.path.join(original_folder, dataname)
    image_end_with = "rgb_u8_v2.tif"
    label_path = "Labels"
    # TODO: 这里需要改label的名称
    label_end_with = "Meiju1_2_Lingtangkou_v5.tif"
    # TODO: 这里需要改成自己的输出文件夹
    output_folder = r"G:/Self-training/Meiju1/iter_4"

    label_path = os.path.join(original_image_path, label_path, label_end_with)

    for path in os.listdir(original_image_path):
        if path.endswith(image_end_with):
            image_path = os.path.join(original_image_path, path)     

    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=str, default="create")
    args = parser.parse_args()
    

    start_time = datetime.now()
    if args.run == "create":
        create_dataset(image_path, label_path, output_folder)
        send_email(f'数据集制作, 用时:{datetime.now() - start_time}', "数据集制作完成..")  

    elif args.run == "predict":
        predict()

    elif args.run == "merge":
        png2tif_and_merge(output_folder)
        send_email(f"数据集合成, 用时: {datetime.now() - start_time}")

    elif args.run == "delete":
        delete_unused_folders(output_folder)
        send_email(f"删除没用的文件夹, 用时: {datetime.now() - start_time}")
    print('run time is {}'.format(datetime.now()-start_time))
