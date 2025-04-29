# 先裁剪
# 后分别每个图计算推理
# 最后合并
import sys
sys.path.append('./')
from src.processing.crop import crop_without_repetition, crop_without_repetition_all
from src.processing.merge import merge_tif
from src.utils.file_io import remove_folder
import os
from datetime import datetime
import argparse
from src.utils.email_utils import send_email
from src.utils.file_io import read_tif, write_tif
import numpy as np
from tqdm import tqdm
from src.processing.merge import merge_tif
import pandas as pd # 导入 pandas 库
import matplotlib.pyplot as plt # 导入 matplotlib
import seaborn as sns # 导入 seaborn
import matplotlib.font_manager as fm # 导入字体管理器
from scipy import stats # 导入 scipy.stats 库


chinese_fonts = ['SimHei', 'Microsoft YaHei', 'STSong', 'STKaiti', 'LiSu', 'FangSong', 'KaiTi', 'YouYuan', 'PingFang SC']
for font_name in chinese_fonts:
    if any(font.name == font_name for font in fm.fontManager.ttflist):
        plt.rcParams['font.family'] = font_name
        break
else:
     print(f"警告: 未找到常见中文字体，尝试使用默认字体，中文可能无法正常显示。请手动配置字体路径。")
     # 如果都没有找到，可以回退到指定文件路径的方法，或者跳过设置让 Matplotlib 使用默认字体

# 设置正常显示负号
plt.rcParams['axes.unicode_minus'] = False

def crop_data(dsm_filepath, output_folder, crop_size=640):
    dsm_folder = "dsm"
    dsm_folder = os.path.join(output_folder, dsm_folder)
    os.makedirs(dsm_folder, exist_ok=True)

    # 裁剪数据集
    crop_without_repetition_all(dsm_filepath, dsm_folder, crop_size)




def convert_dsm_to_csm(output_folder, ignore_val = -9999):
    dsm_folder = "dsm"
    dsm_folder = os.path.join(output_folder, dsm_folder)
    csm_folder = "csm"
    csm_folder = os.path.join(output_folder, csm_folder)
    os.makedirs(csm_folder, exist_ok=True)
    
    # 读取一个文件, 取第1%个值作为最小值, 其他用这个值减去, 得到新的值, 保存到新的文件中
    for file in tqdm(os.listdir(dsm_folder), desc="convert DSM to CSM", unit="files"):
        file_path = os.path.join(dsm_folder, file)
        # print(file_path)
        im_proj, im_geotrans, im_data, im_width, im_height, im_bands = read_tif(file_path)
        # 有效值
        valid_mask = im_data != ignore_val
        valid_data = im_data[valid_mask]
        # 检查有效数据是否为空
        if valid_data.size == 0:
            # 直接写入
            write_tif(im_data, im_geotrans, im_proj, os.path.join(csm_folder, file))
            continue
        min_val = np.percentile(valid_data, 1)
        # print(min_val)
        
        csm_data = np.where(~valid_mask, 
                            ignore_val,  # 保留忽略值不变
                            im_data - min_val) # 减去最小值

        # 将小于1%的部分统一赋值为0    
        csm_data = np.where(valid_mask & (im_data < min_val),
                            0,
                            csm_data)
        
        # 写入数据
        write_tif(csm_data, im_geotrans, im_proj, os.path.join(csm_folder, file))

def merge_csm(output_folder, file_name):
    csm_folder = "csm"
    csm_folder = os.path.join(output_folder, csm_folder)
    csm_merge_folder = "csm_merge"
    csm_merge_folder = os.path.join(output_folder, csm_merge_folder)
    os.makedirs(csm_merge_folder, exist_ok=True)
    # file_name = "csm_64.tif"
    file_path = os.path.join(csm_merge_folder, file_name)
    merge_tif(csm_folder, file_path)



def convert_dsm_to_csm_file(input_path, output_folder, output_name, ignore_val=-9999):
    csm_folder = "csm_merge"
    csm_folder = os.path.join(output_folder, csm_folder)
    os.makedirs(csm_folder, exist_ok=True)
    
    # 读取一个文件, 取第1%个值作为最小值, 其他用这个值减去, 得到新的值, 保存到新的文件中
    # for file in tqdm(os.listdir(dsm_folder), desc="convert DSM to CSM", unit="files"):
        # file_path = os.path.join(dsm_folder, file)
        # print(file_path)
    im_proj, im_geotrans, im_data, im_width, im_height, im_bands = read_tif(input_path)
    # 有效值
    valid_mask = im_data != ignore_val
    valid_data = im_data[valid_mask]
    # 检查有效数据是否为空
    if valid_data.size == 0:
        # 直接写入
        write_tif(im_data, im_geotrans, im_proj, os.path.join(csm_folder, output_name))
        return
    min_val = np.percentile(valid_data, 1)
    # print(min_val)
    
    csm_data = np.where(~valid_mask, 
                        ignore_val,  # 保留忽略值不变
                        im_data - min_val) # 减去最小值

    # 将小于1%的部分统一赋值为0    
    csm_data = np.where(valid_mask & (im_data < min_val),
                        0,
                        csm_data)
    
    # 写入数据
    write_tif(csm_data, im_geotrans, im_proj, os.path.join(csm_folder, output_name))



def delete_unused_folders(output_folder):
    dsm_folder = "dsm"
    dsm_folder = os.path.join(output_folder, dsm_folder)
    csm_folder = "csm"
    csm_folder = os.path.join(output_folder, csm_folder)
    
    remove_folder(dsm_folder)
    remove_folder(csm_folder)

# 取一个像素周围640*640的数据, 然后求1%最小值，然后用这个中心点减去它
def convert_dsm_to_csm_v2(intput_path, output_folder, output_name, window_size=640, ignore_val=-9999, percentile_threshold=1):
    
    csm_folder = "csm_merge"
    csm_folder = os.path.join(output_folder, csm_folder)
    os.makedirs(csm_folder, exist_ok=True)
    

    im_proj, im_geotrans, im_data, im_width, im_height, im_bands = read_tif(intput_path)

    print(f"DSM dimensions: {im_height}x{im_width}")

    # 初始化输出的 CSM 数据数组，与输入 DSM 相同尺寸，初始值为 ignore_val
    csm_data = np.full_like(im_data, ignore_val, dtype=im_data.dtype)

    half_window = window_size // 2

    print("Processing DSM pixels with sliding window...")
    # 遍历每一个像素 (行 r, 列 c)
    # 使用 tqdm 显示进度
    for r in tqdm(range(im_height), desc="Converting DSM to CSM", unit="rows"):
        for c in range(im_width):

            # 获取当前中心像素的 DSM 值
            central_dsm_value = im_data[r, c]

            # 如果中心像素是忽略值，则 CSM 保持忽略值，跳过计算
            if central_dsm_value == ignore_val:
                continue # 跳到下一个像素

            # 定义滑动窗口的边界 (确保不超出影像范围)
            window_row_start = max(0, r - half_window)
            window_row_end = min(im_height, r + half_window + 1)
            window_col_start = max(0, c - half_window)
            window_col_end = min(im_width, c + half_window + 1)

            # 提取局部窗口的数据
            local_window_data = im_data[window_row_start:window_row_end,
                                        window_col_start:window_col_end]

            # 找到局部窗口内所有非忽略值的像素
            valid_mask = local_window_data != ignore_val
            valid_data = local_window_data[valid_mask]

            # 如果局部窗口内存在有效数据
            if valid_data.size > 0:
                # 计算有效数据的第 1% 分位数 (局部最小值)
                min_val = np.percentile(valid_data, percentile_threshold)

                # 计算 CSM 值 = 中心 DSM 值 - 局部最小值
                csm_value = central_dsm_value - min_val

                # 如果计算出的 CSM 值小于 0，或者中心 DSM 值小于局部最小值，则设置为 0
                # 这里的 im_data[r, c] < min_val 包含了 csm_value < 0 的情况，但显式检查更清晰
                if central_dsm_value < min_val:
                    csm_value = 0.0 # 使用浮点数以保持一致性

                # 将计算出的 CSM 值存储到输出数组中
                csm_data[r, c] = csm_value

            # else:
            #     # 如果中心像素有效，但其周围窗口内没有任何有效像素
            #     # 根据之前的逻辑，设置 CSM 为 0.0
            #     # 这表示无法找到局部地面参考，假定冠层高度为 0。
            #     # csm_data[r, c] = 0.0
            #     continue


    csm_output_path = os.path.join(csm_folder, output_name)
    print(f"Writing CSM file: {csm_output_path}")
    # 写数据
    write_tif(csm_data, im_geotrans, im_proj, csm_output_path)
    print("Convertion completed successfully.")


# 取一个像素周围640*640的数据, 然后求1%最小值，然后用这个中心点减去它
def convert_dsm_to_csm_v3(intput_path, output_folder, output_name, calc_window_size=640, processing_block_size=64, ignore_val=-9999, percentile_threshold=1):
    
    csm_folder = "csm_merge"
    csm_folder = os.path.join(output_folder, csm_folder)
    os.makedirs(csm_folder, exist_ok=True)
    

    im_proj, im_geotrans, im_data, im_width, im_height, im_bands = read_tif(intput_path)

    print(f"DSM dimensions: {im_height}x{im_width}")

    # 初始化输出的 CSM 数据数组，与输入 DSM 相同尺寸，初始值为 ignore_val
    csm_data = np.full_like(im_data, ignore_val, dtype=im_data.dtype)

    # 计算计算窗口大小和偏移量
    calc_offset = (calc_window_size - processing_block_size) // 2 # 这就是 288



    print("Processing DSM pixels with sliding window...")
    # 遍历每一个像素 (行 r, 列 c)
    # 使用 tqdm 显示进度
    for r_start in tqdm(range(0, im_height, processing_block_size), desc="Processing Blocks (Rows)", unit="rows"):
        for c_start in range(0, im_width, processing_block_size):


            # 确定当前处理块的实际边界 (处理到影像边缘时可能不足 processing_block_size)
            r_end = min(r_start + processing_block_size, im_height)
            c_end = min(c_start + processing_block_size, im_width)


            # 窗口左上角基于当前处理块的左上角加上/减去计算偏移
            calc_r_start = max(0, r_start - calc_offset)
            calc_r_end = min(im_height, r_end + calc_offset)
            calc_c_start = max(0, c_start - calc_offset)
            calc_c_end = min(im_width, c_end + calc_offset)

            # 提取用于计算局部最小值的 640x640 窗口数据
            # 注意：如果计算窗口范围无效（例如 start >= end），NumPy 切片会返回空数组
            calc_window_data = im_data[calc_r_start:calc_r_end, calc_c_start:calc_c_end]


            # 找到计算窗口内所有非忽略值的像素
            valid_calc_window_mask = calc_window_data != ignore_val
            valid_calc_data = calc_window_data[valid_calc_window_mask]

            # --- 计算局部最小值 ---
            min_val = float(ignore_val) # 默认局部最小值为 ignore_val
            if valid_calc_data.size > 0:
                try:
                    # 计算有效数据的第 1% 分位数
                    min_val = float(np.percentile(valid_calc_data, percentile_threshold))
                except Exception as e:
                    # 如果 percentile 计算失败 (极少数情况，如数据分布异常)，保留 ignore_val
                    print(f"\nWarning: Percentile calculation failed for block starting at ({r_start}, {c_start}). Using ignore_val as local min. Error: {e}")
                    min_val = float(ignore_val)
            

            # --- 处理当前 64x64 块的数据 ---
            # 提取当前处理块的原始 DSM 数据
            dsm_block_data = im_data[r_start:r_end, c_start:c_end]

            # 初始化当前块的 CSM 数据数组，使用浮点类型
            csm_block_data = np.full_like(dsm_block_data, float(ignore_val), dtype=np.float32)

            # 找到块内非忽略值的像素
            valid_block_mask = dsm_block_data != ignore_val

            # 在块内有效像素位置进行计算
            # 将有效像素数据转换为浮点类型进行计算
            dsm_block_valid_float = dsm_block_data[valid_block_mask].astype(np.float32)

            # 计算初步的 CSM 值 = 块内有效 DSM 值 - 局部最小值
            # 这里的 min_val 是对整个 640x640 窗口计算得出的单一值
            preliminary_block_csm = dsm_block_valid_float - min_val

            # 应用 CSM 值不小于 0 的规则
            final_block_csm_values = np.maximum(0.0, preliminary_block_csm)

            # 将计算结果填充回当前块的 CSM 数组中对应的有效位置
            csm_block_data[valid_block_mask] = final_block_csm_values

            # --- 将处理好的块数据复制到主 CSM 数组中 ---
            # CSM 数据已经初始化为 ignore_val，所以只需要复制计算好的块
            csm_data[r_start:r_end, c_start:c_end] = csm_block_data



    csm_output_path = os.path.join(csm_folder, output_name)
    print(f"Writing CSM file: {csm_output_path}")
    # 写数据
    write_tif(csm_data, im_geotrans, im_proj, csm_output_path)
    print("Convertion completed successfully.")


def convert_dsm_to_csm_v4(intput_path, output_folder, output_name, calc_window_size=640, processing_block_size=64, ignore_val=-9999, percentile_threshold=1, threshold=0.8):
    
    csm_folder = "csm_merge"
    csm_folder = os.path.join(output_folder, csm_folder)
    os.makedirs(csm_folder, exist_ok=True)
    

    im_proj, im_geotrans, im_data, im_width, im_height, im_bands = read_tif(intput_path)

    print(f"DSM dimensions: {im_height}x{im_width}")

    # 初始化输出的 CSM 数据数组，与输入 DSM 相同尺寸，初始值为 ignore_val
    csm_data = np.full_like(im_data, ignore_val, dtype=im_data.dtype)

    # 计算计算窗口大小和偏移量
    calc_offset = (calc_window_size - processing_block_size) // 2 # 这就是 288



    print("Processing DSM pixels with sliding window...")
    # 遍历每一个像素 (行 r, 列 c)
    # 使用 tqdm 显示进度
    for r_start in tqdm(range(0, im_height, processing_block_size), desc="Processing Blocks (Rows)", unit="rows"):
        for c_start in range(0, im_width, processing_block_size):


            # 确定当前处理块的实际边界 (处理到影像边缘时可能不足 processing_block_size)
            r_end = min(r_start + processing_block_size, im_height)
            c_end = min(c_start + processing_block_size, im_width)

            # 提取当前处理块的原始 DSM 数据
            dsm_block_data = im_data[r_start:r_end, c_start:c_end]
            # 创建一个布尔掩膜，标记出处理块内非忽略值的像素位置
            valid_block_mask = dsm_block_data != ignore_val

            # 窗口左上角基于当前处理块的左上角加上/减去计算偏移
            calc_r_start = max(0, r_start - calc_offset)
            calc_r_end = min(im_height, r_end + calc_offset)
            calc_c_start = max(0, c_start - calc_offset)
            calc_c_end = min(im_width, c_end + calc_offset)

            # 提取用于计算局部最小值的 640x640 窗口数据
            # 注意：如果计算窗口范围无效（例如 start >= end），NumPy 切片会返回空数组
            calc_window_data = im_data[calc_r_start:calc_r_end, calc_c_start:calc_c_end]


            # 找到计算窗口内所有非忽略值的像素
            valid_calc_window_mask = calc_window_data != ignore_val
            valid_calc_data = calc_window_data[valid_calc_window_mask]

            # --- 计算局部最小值 ---
            min_val = float(ignore_val) # 默认局部最小值为 ignore_val
            min_in_block_val = float(ignore_val) # 初始化处理块内最小值为 ignore_val
            # 1. **改进点：** 找到当前处理块内的最小值 (排除忽略值)
            if valid_block_mask.any(): # 检查处理块内是否有任何有效数据
                 min_in_block_val = np.min(dsm_block_data[valid_block_mask]) # 计算块内有效数据的最小值
            
            if valid_calc_data.size > 0:
                # 初始化用于计算百分位数的数据集，默认为计算窗口内的所有有效数据
                data_for_percentile = valid_calc_data
                # 3. **改进点：** 如果处理块内找到了有效的最小值，则根据块内最小值过滤计算窗口数据
                if min_in_block_val != ignore_val:
                    # 计算排除阈值: 处理块内的最小值 减去 新增的 threshold 参数
                    exclusion_threshold_val = min_in_block_val - threshold

                    # **关键过滤步骤：** 过滤计算窗口内的有效数据
                    # 只保留那些值 大于等于 排除阈值 的数据点，作为用于计算百分位数的新数据集
                    data_for_percentile = valid_calc_data[valid_calc_data >= exclusion_threshold_val]

                # 4. **改进点：** 检查过滤后的数据集是否有数据，有数据才计算百分位数
                if data_for_percentile.size > 0:
                    try:
                        # 在过滤后的数据集上计算指定的百分位数 (默认为第1个百分位数)
                        min_val = float(np.percentile(data_for_percentile, percentile_threshold))
                    except Exception as e:
                        # 如果百分位数计算失败 (极少数情况)，保持 min_val 为 ignore_val
                        print(f"\n警告: 在 ({r_start}, {c_start}) 开始的块，过滤后计算百分位数失败。使用 ignore_val 作为局部最小值。错误: {e}")
                        min_val = float(ignore_val)

                # try:
                #     # 计算有效数据的第 1% 分位数
                #     min_val = float(np.percentile(valid_calc_data, percentile_threshold))
                # except Exception as e:
                #     # 如果 percentile 计算失败 (极少数情况，如数据分布异常)，保留 ignore_val
                #     print(f"\nWarning: Percentile calculation failed for block starting at ({r_start}, {c_start}). Using ignore_val as local min. Error: {e}")
                #     min_val = float(ignore_val)
            

            # --- 处理当前 64x64 块的数据 ---
            # 初始化当前块的 CSM 数据数组，使用浮点类型
            csm_block_data = np.full_like(dsm_block_data, float(ignore_val), dtype=np.float32)


            # 如果当前处理块有任何有效数据点 (非 ignore_val)
            if valid_block_mask.any():
                # 提取块内有效 DSM 数据，转换为浮点类型用于计算
                dsm_block_valid_float = dsm_block_data[valid_block_mask].astype(np.float32)

                # 只有当找到有效的局部最小值 (min_val) 时才进行 CSM 计算
                if min_val != ignore_val:
                     # 计算初步的 CSM 值 = 块内有效 DSM 值 - 局部最小值 (min_val 来自计算窗口，可能已过滤)
                     preliminary_block_csm = dsm_block_valid_float - min_val

                     # 应用 CSM 值不小于 0 的规则 (冠层高度不能是负的)
                     final_block_csm_values = np.maximum(0.0, preliminary_block_csm)

                     # 将计算结果填充回当前块的 CSM 数组中对应的有效位置
                     csm_block_data[valid_block_mask] = final_block_csm_values
                     # 块内原本就是 ignore_val 的像素会继续保持 ignore_val
            # 找到块内非忽略值的像素
            # valid_block_mask = dsm_block_data != ignore_val

            # # 在块内有效像素位置进行计算
            # # 将有效像素数据转换为浮点类型进行计算
            # dsm_block_valid_float = dsm_block_data[valid_block_mask].astype(np.float32)

            # # 计算初步的 CSM 值 = 块内有效 DSM 值 - 局部最小值
            # # 这里的 min_val 是对整个 640x640 窗口计算得出的单一值
            # preliminary_block_csm = dsm_block_valid_float - min_val

            # # 应用 CSM 值不小于 0 的规则
            # final_block_csm_values = np.maximum(0.0, preliminary_block_csm)

            # # 将计算结果填充回当前块的 CSM 数组中对应的有效位置
            # csm_block_data[valid_block_mask] = final_block_csm_values

            # --- 将处理好的块数据复制到主 CSM 数组中 ---
            # CSM 数据已经初始化为 ignore_val，所以只需要复制计算好的块
            csm_data[r_start:r_end, c_start:c_end] = csm_block_data



    csm_output_path = os.path.join(csm_folder, output_name)
    print(f"Writing CSM file: {csm_output_path}")
    # 写数据
    write_tif(csm_data, im_geotrans, im_proj, csm_output_path)
    print("Convertion completed successfully.")




def convert_dsm_to_csm_v5(intput_path, output_path, calc_window_size=640, processing_block_size=64, ignore_val=-9999, percentile_threshold=1):
    

    im_proj, im_geotrans, im_data, im_width, im_height, im_bands = read_tif(intput_path)

    print(f"DSM dimensions: {im_height}x{im_width}")

    # 初始化输出的 CSM 数据数组，与输入 DSM 相同尺寸，初始值为 ignore_val
    csm_data = np.full_like(im_data, ignore_val, dtype=im_data.dtype)

    # 计算计算窗口大小和偏移量
    calc_offset = (calc_window_size - processing_block_size) // 2 # 这就是 288



    print("Processing DSM pixels with sliding window...")
    # 遍历每一个像素 (行 r, 列 c)
    # 使用 tqdm 显示进度
    for r_start in tqdm(range(0, im_height, processing_block_size), desc="Processing Blocks (Rows)", unit="rows"):
        for c_start in range(0, im_width, processing_block_size):


            # 确定当前处理块的实际边界 (处理到影像边缘时可能不足 processing_block_size)
            r_end = min(r_start + processing_block_size, im_height)
            c_end = min(c_start + processing_block_size, im_width)


            # 窗口左上角基于当前处理块的左上角加上/减去计算偏移
            calc_r_start = max(0, r_start - calc_offset)
            calc_r_end = min(im_height, r_end + calc_offset)
            calc_c_start = max(0, c_start - calc_offset)
            calc_c_end = min(im_width, c_end + calc_offset)

            # 提取用于计算局部最小值的 640x640 窗口数据
            # 注意：如果计算窗口范围无效（例如 start >= end），NumPy 切片会返回空数组
            calc_window_data = im_data[calc_r_start:calc_r_end, calc_c_start:calc_c_end]


            # 找到计算窗口内所有非忽略值的像素
            valid_calc_window_mask = calc_window_data != ignore_val
            valid_calc_data = calc_window_data[valid_calc_window_mask]
            
            if valid_calc_data.size == 0:   # 不符合就跳过
                continue

            # --- 计算局部最小值 ---
            min_val = float(ignore_val) # 默认局部最小值为 ignore_val
            if valid_calc_data.size > 0:
                try:
                    # 计算有效数据的第 1% 分位数
                    min_val = float(np.percentile(valid_calc_data, percentile_threshold))
                except Exception as e:
                    # 如果 percentile 计算失败 (极少数情况，如数据分布异常)，保留 ignore_val
                    print(f"\nWarning: Percentile calculation failed for block starting at ({r_start}, {c_start}). Using ignore_val as local min. Error: {e}")
                    min_val = float(ignore_val)
            

            # --- 处理当前 64x64 块的数据 ---
            # 提取当前处理块的原始 DSM 数据
            dsm_block_data = im_data[r_start:r_end, c_start:c_end]

            # 初始化当前块的 CSM 数据数组，使用浮点类型
            csm_block_data = np.full_like(dsm_block_data, float(ignore_val), dtype=np.float32)

            # 找到块内非忽略值的像素
            valid_block_mask = dsm_block_data != ignore_val

            # 在块内有效像素位置进行计算
            # 将有效像素数据转换为浮点类型进行计算
            dsm_block_valid_float = dsm_block_data[valid_block_mask].astype(np.float32)

            # 计算初步的 CSM 值 = 块内有效 DSM 值 - 局部最小值
            # 这里的 min_val 是对整个 640x640 窗口计算得出的单一值
            preliminary_block_csm = dsm_block_valid_float - min_val

            # 应用 CSM 值不小于 0 的规则
            final_block_csm_values = np.maximum(0.0, preliminary_block_csm)

            # 将计算结果填充回当前块的 CSM 数组中对应的有效位置
            csm_block_data[valid_block_mask] = final_block_csm_values

            # --- 将处理好的块数据复制到主 CSM 数组中 ---
            # CSM 数据已经初始化为 ignore_val，所以只需要复制计算好的块
            csm_data[r_start:r_end, c_start:c_end] = csm_block_data



    print(f"Writing CSM file: {output_path}")
    # 写数据
    write_tif(csm_data, im_geotrans, im_proj, output_path)
    print("Convertion completed successfully.")


import numpy as np
from tqdm import tqdm
# 假设 read_tif 和 write_tif 函数可用，并且处理 GDAL/rasterio 相关操作
# 示例占位符：
# def read_tif(filepath):
#     # 使用 rasterio 或 GDAL 读取影像
#     # rasterio 示例：
#     # with rasterio.open(filepath) as src:
#     #     im_data = src.read(1) # 读取第1波段
#     #     im_geotrans = src.transform # 仿射变换
#     #     im_proj = src.crs # 坐标参考系统
#     #     im_width = src.width
#     #     im_height = src.height
#     #     im_bands = src.count # 波段数
#     # return im_proj, im_geotrans, im_data, im_width, im_height, im_bands
#     pass

# def write_tif(data, geotrans, proj, filepath, nodata_val=None):
#     # 使用 rasterio 或 GDAL 写入影像
#     # rasterio 示例：
#     # from rasterio.transform import from_bounds
#     # from rasterio.crs import CRS
#     # with rasterio.open(
#     #     filepath,
#     #     'w',
#     #     driver='GTiff',
#     #     height=data.shape[0],
#     #     width=data.shape[1],
#     #     count=1, # 单波段输出
#     #     dtype=data.dtype,
#     #     crs=proj,
#     #     transform=geotrans,
#     #     nodata=nodata_val # 设置 NoData 值
#     # ) as dst:
#     #     dst.write(data, 1)
#     pass


def convert_dsm_to_csm_v5_optimized(intput_path, output_path, calc_window_size=640, processing_block_size=64, ignore_val=-9999, percentile_threshold=1):
    """
    使用滑动窗口计算局部最小值（百分位数），将数字表面模型（DSM）转换为冠层表面模型（CSM）。

    参数：
        intput_path (str): 输入 DSM GeoTIFF 文件的路径。
        output_path (str): 输出 CSM GeoTIFF 文件的保存路径。
        calc_window_size (int): 用于计算局部地面高度（百分位数）的方形窗口大小。
        processing_block_size (int): 每次迭代处理的方形块大小。CSM 值将为此块内的像素计算。
        ignore_val (int 或 float): 输入 DSM 中表示无数据或忽略像素的值。
        percentile_threshold (float): 在计算窗口内用于确定局部地面高度的百分位数 (0-100)。
    """

    # --- 读取输入 DSM ---
    # 读取 DSM。假设 read_tif 处理文件的打开和元数据的获取。
    # im_data_orig 将是一个 NumPy 数组。
    im_proj, im_geotrans, im_data_orig, im_width, im_height, im_bands = read_tif(intput_path)

    print(f"DSM 尺寸: {im_height}x{im_width}")
    
    # 为了高效计算，一次性将输入数据转换为 float32 类型
    # 存储原始的 ignore value 并将其转换为目标浮点类型
    original_ignore_val = ignore_val
    ignore_val_float = float(ignore_val)
    
    # 将 DSM 数据转换为 float32，保留 ignore values
    # 处理原始 ignore_val 可能超出浮点范围的潜在转换问题
    # 对于 -9999，float32 是没问题的。对于非常大/小的整数，可能需要谨慎转换。
    # 对于 -9999，简单的转换即可。
    im_data = im_data_orig.astype(np.float32)
    
    # 在浮点数组中更新 ignore value
    im_data[im_data_orig == original_ignore_val] = ignore_val_float


    # --- 初始化输出 CSM ---
    # 初始化输出 CSM 数据数组为 float32 类型，尺寸与输入相同，
    # 并用浮点的 ignore_val 填充。
    csm_data = np.full_like(im_data, ignore_val_float, dtype=np.float32)


    # --- 设置窗口和块参数 ---
    # 计算将计算窗口中心对齐到处理块所需的偏移量
    if calc_window_size < processing_block_size:
         raise ValueError("calc_window_size 必须大于或等于 processing_block_size")
    calc_offset = (calc_window_size - processing_block_size) // 2 # 对于 640 和 64，这是 288

    print("正在使用滑动窗口处理 DSM 像素...")

    # --- 遍历处理块 ---
    # 使用 tqdm 显示行处理进度
    for r_start in tqdm(range(0, im_height, processing_block_size), desc="处理块 (行)", unit="行"):
        for c_start in range(0, im_width, processing_block_size):

            # --- 确定当前处理块的边界 ---
            # 这些是我们将计算 CSM 值的区域边界
            r_end = min(r_start + processing_block_size, im_height)
            c_end = min(c_start + processing_block_size, im_width)

            # --- 确定当前计算窗口的边界 ---
            # 这些是用于提取数据以查找局部最小值的较大窗口边界
            calc_r_start = max(0, r_start - calc_offset)
            calc_r_end = min(im_height, r_end + calc_offset)
            calc_c_start = max(0, c_start - calc_offset)
            calc_c_end = min(im_width, c_end + calc_offset)

            # --- 提取用于局部最小值计算的数据 ---
            # 从较大的计算窗口中提取数据。
            # 注意：如果窗口边界无效 (start >= end)，NumPy 切片将返回空数组
            calc_window_data = im_data[calc_r_start:calc_r_end, calc_c_start:calc_c_end]

            # 查找计算窗口内所有非忽略值像素
            valid_calc_window_mask = calc_window_data != ignore_val_float
            valid_calc_data = calc_window_data[valid_calc_window_mask]

            # --- 计算局部最小值 ---
            # 默认局部最小值为忽略值 (或者通过跳过处理来处理)
            min_val = ignore_val_float # 初始化 min_val 为浮点 ignore value

            # 只有当计算窗口内有有效数据时才计算百分位数
            if valid_calc_data.size > 0:
                try:
                    # 计算计算窗口内有效数据的百分位数
                    min_val = float(np.percentile(valid_calc_data, percentile_threshold))
                except Exception as e:
                    # 备用方案：如果百分位数计算意外失败（在 valid_calc_data.size > 0 时应该很少见），
                    # 将 min_val 视为 ignore_val。
                    print(f"\n警告：覆盖块 ({r_start}, {c_start}) 的窗口的百分位数计算失败。使用 ignore_val 作为局部最小值。错误：{e}")
                    min_val = ignore_val_float
            else:
                 # 如果计算窗口内没有有效数据，我们无法确定地面高度。
                 # 该块中原本有效的像素在输出中将保持 ignore_val，
                 # 因为我们不会在下面为它们执行 CSM 计算。
                 # 因此，我们可以跳到下一个块。
                 continue # 如果计算窗口内没有有效数据，跳过处理此块


            # --- 处理当前的 64x64 处理块 ---
            # 我们将为范围 [r_start:r_end, c_start:c_end] 内的像素计算 CSM，
            # 使用从较大窗口计算出的 min_val。

            # 获取主 DSM 和 CSM 数组中当前处理块的视图
            dsm_block_view = im_data[r_start:r_end, c_start:c_end]
            csm_block_view = csm_data[r_start:r_end, c_start:c_end] # 这是我们将写入结果的地方

            # 在当前处理块内查找有效像素
            valid_block_mask = dsm_block_view != ignore_val_float

            # 如果此块中有有效像素 并且 我们计算出了一个有效的 min_val
            # （即 min_val 不是 ignore_val_float，意味着 valid_calc_data.size > 0 为真）
            # 注意：上面的 `continue` 处理了 valid_calc_data.size == 0 的情况。
            # 我们仍然需要检查 *处理块* 本身中是否有有效像素。
            if np.any(valid_block_mask):
                # 仅对块内有效像素执行计算
                # 使用视图上的布尔索引直接访问/修改相关像素
                
                # 获取处理块中有效的 DSM 值
                dsm_block_valid_values = dsm_block_view[valid_block_mask]

                # 计算初步的 CSM 值（对于有效像素）： DSM - local_min
                # min_val 是从大窗口得出的单个标量值
                preliminary_block_csm = dsm_block_valid_values - min_val

                # 应用规则：CSM 值必须非负
                final_block_csm_values = np.maximum(0.0, preliminary_block_csm)

                # 将计算出的 CSM 值分配回主 csm_data 数组中，
                # 使用 csm_block_view 上的掩膜
                csm_block_view[valid_block_mask] = final_block_csm_values

            # csm_data 中该块内不属于 valid_block_mask 的像素
            # (即，它们在输入 DSM 中是 ignore_val) 将保持为 ignore_val_float，
            # 因为我们将 csm_data 初始化为 ignore_val_float，并且只
            # 将值分配给了 valid_block_mask 的位置。


    print(f"正在写入 CSM 文件: {output_path}")
    # --- 写入输出 CSM ---
    # 如果需要，将最终的 CSM 数据转换回原始输入数据类型，
    # 或者写入为 float32。写入为 float32 通常更安全，能保留精度。
    # 确保写入时正确处理 ignore_val。
    # csm_data 数组已经是 float32 类型，包含 ignore_val_float。
    # write_tif 需要处理写入 float 数组并设置 nodata 值。
    # 如果输出格式必须是整数，你需要决定如何进行四舍五入/类型转换，
    # 以及如何表示 ignore_val (例如，将浮点 ignore_val 转换回整数 ignore_val)。
    # 建议坚持输出 float32 以保留精度。
    write_tif(csm_data, im_geotrans, im_proj, output_path) # 传递 nodata_val 参数

    print("转换成功完成。")



def convert_dsm_to_csm_v6_optimized(intput_path, output_path, ignore_val=-9999, percentile_threshold=1):
    """
    使用滑动窗口计算局部最小值（百分位数），将数字表面模型（DSM）转换为冠层表面模型（CSM）。

    参数：
        intput_path (str): 输入 DSM GeoTIFF 文件的路径。
        output_path (str): 输出 CSM GeoTIFF 文件的保存路径。
        calc_window_size (int): 用于计算局部地面高度（百分位数）的方形窗口大小。
        processing_block_size (int): 每次迭代处理的方形块大小。CSM 值将为此块内的像素计算。
        ignore_val (int 或 float): 输入 DSM 中表示无数据或忽略像素的值。
        percentile_threshold (float): 在计算窗口内用于确定局部地面高度的百分位数 (0-100)。
    """

    # --- 读取输入 DSM ---
    # 读取 DSM。假设 read_tif 处理文件的打开和元数据的获取。
    # im_data_orig 将是一个 NumPy 数组。
    im_proj, im_geotrans, im_data, im_width, im_height, im_bands = read_tif(intput_path)

    print(f"DSM 尺寸: {im_height}x{im_width}")
      

    # --- 初始化输出 CSM ---
    # 初始化输出 CSM 数据数组为 float32 类型，尺寸与输入相同，
    # 并用浮点的 ignore_val 填充。
    csm_data = np.full_like(im_data, ignore_val, dtype=np.float32)

    # 直接全图操作
    valid_mask = im_data != ignore_val
    valid_data = im_data[valid_mask]
    
    # 求取1%的大小
    min_val = np.percentile(valid_data, 1)
    print("min val is", min_val)
    # 减去这个最小值
    preliminary_block_csm = im_data[valid_mask] - min_val
    final_block_csm_values = np.maximum(0.0, preliminary_block_csm)
    csm_data[valid_mask] = final_block_csm_values
    

    print(f"正在写入 CSM 文件: {output_path}")

    write_tif(csm_data, im_geotrans, im_proj, output_path) # 传递 nodata_val 参数

    print("转换成功完成。")


def merge_csm_all(input_folder, output_name,suffix="_v2.tif", ignore_val=-9999):
    input_paths = []
    for file in os.listdir(input_folder):
        if file.endswith(suffix):
            print(file)
            input_paths.append(os.path.join(input_folder, file))


    input_datas = []
    im_proj, im_geotrans = None, None

    for file in input_paths:
        im_proj, im_geotrans, data, width, height, _ = read_tif(file)
        print(data.shape)
        input_datas.append(data)

    csm_data = np.full_like(input_datas[0], ignore_val, dtype=np.float32)

    # 依次赋值
    for data in input_datas:
        valid_mask = data != ignore_val
        valid_data = data[valid_mask]
        print(valid_data.size)
        csm_data[valid_mask] = valid_data

    print(im_proj)
    print(im_geotrans)
    # 写入数据
    write_tif(csm_data, im_geotrans, im_proj, output_name) # 传递 nodata_val 参数


def dsm_get_max_min(input_folder, output_csv_path="dsm_csm_data_v6.csv", output_plot_folder="data_fig",ignore_val=-9999, suffix=".tif", normality_alpha=0.05):
    """
    遍历指定文件夹下符合条件（后缀为 suffix 且文件名长度为 9）的 GeoTIFF 文件，
    计算每个文件的忽略忽略值后的统计数据（最小值、1%、最大值、99%），
    进行正态分布检验，计算均值和标准差，并将所有结果保存到指定的 CSV 文件中。
    如果指定了 output_plot_folder，则同时为每个文件生成数据分布图。

    参数：
        input_folder (str): 包含 GeoTIFF 文件的文件夹路径。
        output_csv_path (str): 输出 CSV 文件的完整路径，用于保存统计和检验结果。
        output_plot_folder (str, optional): 保存分布图的文件夹路径。如果为 None，则不生成图。
                                            默认值：None。
        ignore_val (int 或 float): GeoTIFF 文件中的忽略值 (NoData)。
        suffix (str): 筛选文件的后缀。
        normality_alpha (float): 正态性检验的显著性水平 (alpha)。如果 p-value > alpha，
                                 则认为数据“符合正态分布”。默认值：0.05。
    """
    input_paths = []
    # 查找符合条件的文件
    for file in os.listdir(input_folder):
        file_path = os.path.join(input_folder, file)
        # 检查是否是文件，是否以指定后缀结尾，并且文件名（不含路径）长度是否为 9
        # if os.path.isfile(file_path) and file.endswith(suffix) and len(file) == 9:
        if os.path.isfile(file_path) and file.endswith(suffix) and len(file) == 9:
            input_paths.append(file_path)

    if not input_paths:
        print(f"在文件夹 '{input_folder}' 中未找到符合条件 (后缀: '{suffix}', 文件名长度: 9) 的文件。")
        # 创建一个空的 CSV 文件，只包含表头
        df_empty = pd.DataFrame(columns=['文件', '最小值', '1%分位数', '最大值', '99%分位数',
                                          '正态性检验P值', '符合正态分布 (P>Alpha)', '均值', '标准差'])
        df_empty.to_csv(output_csv_path, index=False, encoding='utf-8-sig')
        print(f"已创建一个空的 CSV 文件：'{output_csv_path}'")
        return

    print(f"找到 {len(input_paths)} 个符合条件的文件。")

    # 用于存储每个文件统计结果的列表
    results_list = []

    # 如果指定了输出图文件夹，则创建该文件夹（如果不存在）
    if output_plot_folder:
        os.makedirs(output_plot_folder, exist_ok=True)
        print(f"图将保存到文件夹：'{output_plot_folder}'")

    # 遍历每个找到的文件，使用 tqdm 显示进度
    for file_path in tqdm(input_paths, desc="处理文件", unit="文件"):
        file_name = os.path.basename(file_path)
        
        # 初始化当前文件的结果字典，使用 None 或 np.nan 作为默认值
        current_file_results = {
            '文件': file_name,
            '最小值': np.nan,
            '1%分位数': np.nan,
            '最大值': np.nan,
            '99%分位数': np.nan,
            '正态性检验P值': np.nan,
            '符合正态分布 (P>Alpha)': 'N/A', # 默认标记为不适用
            '均值': np.nan,
            '标准差': np.nan
        }

        try:
            # 读取数据
            im_proj, im_geotrans, data, width, height, _ = read_tif(file_path)

            # 忽略 ignore_val 的数据
            valid_data = data[data != ignore_val]

            # 检查是否存在有效数据
            if valid_data.size > 0:
                # --- 计算基本统计值 ---
                current_file_results['最小值'] = np.min(valid_data)
                current_file_results['1%分位数'] = np.percentile(valid_data, 1)
                current_file_results['最大值'] = np.max(valid_data)
                current_file_results['99%分位数'] = np.percentile(valid_data, 99)
                
                # --- 计算均值和标准差（无论是否符合正态分布，都是有效描述性统计量） ---
                current_file_results['均值'] = np.mean(valid_data)
                current_file_results['标准差'] = np.std(valid_data) # 默认计算样本标准差

                print(current_file_results['均值'])
                # --- 正态分布检验 ---
                # D'Agostino-Pearson test 需要样本大小 > 8
                # if valid_data.size > 8:
                #     try:
                #         statistic, p_value = stats.normaltest(valid_data)
                #         current_file_results['正态性检验P值'] = p_value
                #         # 判断是否符合正态分布
                #         if p_value > normality_alpha:
                #             current_file_results['符合正态分布 (P>Alpha)'] = f'是 (P>{normality_alpha:.3f})'
                #         else:
                #             current_file_results['符合正态分布 (P>Alpha)'] = f'否 (P<={normality_alpha:.3f})'
                #     except Exception as test_e:
                #          print(f"\n警告: 文件 '{file_name}' 进行正态性检验时出错: {test_e}")
                #          current_file_results['正态性检验P值'] = '错误'
                #          current_file_results['符合正态分布 (P>Alpha)'] = '检验失败'

                # else:
                #     # 样本太小不足以进行 normaltest
                #     current_file_results['符合正态分布 (P>Alpha)'] = f'样本太小 ({valid_data.size})'
                #     # P值和检验结果保持默认的 NaN/N/A

                # --- 可视化数据分布（如果指定了输出文件夹） ---
                # if output_plot_folder:
                #     try:
                #         plt.figure(figsize=(10, 6))
                #         # 绘制直方图和核密度估计图
                #         sns.histplot(valid_data, kde=True, color='skyblue', bins=50)
                        
                #         # 可选：如果数据量足够且通过正态性检验，可以叠加拟合的正态分布曲线
                #         if current_file_results['符合正态分布 (P>Alpha)'].startswith('是'):
                #              # 绘制拟合的正态分布曲线
                #              xmin, xmax = plt.xlim()
                #              x = np.linspace(xmin, xmax, 100)
                #              p = stats.norm.pdf(x, current_file_results['均值'], current_file_results['标准差'])
                #              # seaborn histplot 默认会归一化，频数/总样本数/bin宽度，所以PDF需要乘以一些系数来匹配
                #              # 或者更简单地，如果kde=True，seaborn已经画了基于KDE的密度，可以作为参考
                #              # 直接绘制PDF曲线需要小心比例匹配，暂不强制添加，以免混淆
                #              # plt.plot(x, p, 'k', linewidth=2, label='Fitted Normal PDF')
                #              pass # 暂时不叠加PDF，以免复杂化图例和比例

                #         plt.title(f'文件数据分布: {file_name}\n(均值: {current_file_results["均值"]:.2f}, 标准差: {current_file_results["标准差"]:.2f})', fontsize=12)
                #         plt.xlabel('像素值', fontsize=12)
                #         plt.ylabel('频数', fontsize=12)
                #         plt.grid(True, linestyle='--', alpha=0.6)

                #         # 构建输出图的文件路径
                #         plot_filename = f"{os.path.splitext(file_name)[0]}_distribution.png"
                #         plot_output_path = os.path.join(output_plot_folder, plot_filename)

                #         # 保存图
                #         plt.savefig(plot_output_path)
                #         plt.close()

                #     except Exception as plot_e:
                #         print(f"\n警告: 为文件 '{file_name}' 生成或保存图时发生错误: {plot_e}")


            else:
                # 文件中没有有效数据，之前的初始化结果已经包含了 NaN/N/A
                print(f"\n警告: 文件 '{file_name}' 中不含有效数据 (忽略值外的数据)，跳过统计计算、正态性检验和图生成。")

        except Exception as e:
            print(f"\n错误: 处理文件 '{file_name}' 时发生错误: {e}")
            # 发生错误时，也记录一行，标记为错误
            current_file_results = {
                '文件': file_name,
                '最小值': '错误', '1%分位数': '错误', '最大值': '错误', '99%分位数': '错误',
                '正态性检验P值': '错误', '符合正态分布 (P>Alpha)': '错误', '均值': '错误', '标准差': '错误'
            }

        # 将处理结果（无论成功与否）添加到列表中
        results_list.append(current_file_results)


    # 将结果列表转换为 pandas DataFrame
    results_df = pd.DataFrame(results_list)

    # 将 DataFrame 保存为 CSV 文件
    try:
        # 重新排列表的列顺序，让文件名在前
        ordered_columns = ['文件', '最小值', '1%分位数', '最大值', '99%分位数',
                           '均值', '标准差', '正态性检验P值', '符合正态分布 (P>Alpha)']
        results_df = results_df[ordered_columns] # 确保列的顺序

        results_df.to_csv(output_csv_path, index=False, encoding='utf-8-sig')
        print(f"\n统计和检验结果已成功保存到 '{output_csv_path}'")
    except Exception as e:
        print(f"\n错误: 保存 CSV 文件时发生错误: {e}")

def norm_csm_all(input_folder, suffix=".tif", ignore_val=-9999):
    input_paths = []
    # 查找符合条件的文件
    for file in os.listdir(input_folder):
        file_path = os.path.join(input_folder, file)
        # 检查是否是文件，是否以指定后缀结尾，并且文件名（不含路径）长度是否为 9
        if os.path.isfile(file_path) and file.endswith(suffix) and len(file) == 9:
            input_paths.append(file_path)
    print(input_paths)
    sub_values =  [1.191961, 1.281902, 0.787999, 0, -1.377523, 0.426319]
    for file_path, sub_value in zip(input_paths, sub_values):
        csm_filepath = file_path.replace(".tif", "_norm.tif")
        im_proj, im_geotrans, data, width, height, _ = read_tif(file_path)
        valid_mask = data != ignore_val
        valid_data = data[valid_mask]
        new_data = valid_data + sub_value
        data[valid_mask] = new_data
        print(data)
        write_tif(data, im_geotrans, im_proj, csm_filepath)
        print("save file:", csm_filepath)
        # 获取新一轮的均值和方差
        print(np.mean(new_data), np.max(new_data), np.percentile(new_data, 1))



def norm_csm_all_v2(input_folder, suffix=".tif", ignore_val=-9999, sub_values = None):
    input_paths = []
    # 查找符合条件的文件
    for file in os.listdir(input_folder):
        file_path = os.path.join(input_folder, file)
        # 检查是否是文件，是否以指定后缀结尾，并且文件名（不含路径）长度是否为 9
        if os.path.isfile(file_path) and file.endswith(suffix) and len(file) == 9:
            input_paths.append(file_path)
    print(input_paths)
    # sub_values =  [-0.805754, -0.197908, -0.618795, 0]
    for file_path, sub_value in zip(input_paths, sub_values):
        csm_filepath = file_path.replace(".tif", "_norm.tif")
        im_proj, im_geotrans, data, width, height, _ = read_tif(file_path)
        valid_mask = data != ignore_val
        valid_data = data[valid_mask]
        new_data = valid_data + sub_value
        # 计算归一化
        # 获取99%最大值, 和1%最小值
        max_value = np.percentile(new_data, 99)
        min_value = np.percentile(new_data, 1)
        new_data = (new_data - min_value) / (max_value - min_value)
        data[valid_mask] = new_data
        # print(data)
        write_tif(data, im_geotrans, im_proj, csm_filepath)
        print("save file:", csm_filepath)
        # 获取新一轮的均值和方差
        print('均值:', np.mean(new_data), ', 最大值:', np.max(new_data), ', 1%分位数:', np.percentile(new_data, 1), ', 99%分位数:', np.percentile(new_data, 99), "最小值:", np.min(new_data))
        print(np.mean(new_data), np.max(new_data), np.percentile(new_data, 1))

def merge_csm_selected(input_files, output_name, ignore_val):
    input_datas = []
    im_proj, im_geotrans = None, None

    for file in input_files:
        im_proj, im_geotrans, data, width, height, _ = read_tif(file)
        print(data.shape)
        input_datas.append(data)

    csm_data = np.full_like(input_datas[0], ignore_val, dtype=np.float32)

    # 依次赋值
    for data in input_datas:
        valid_mask = data != ignore_val
        valid_data = data[valid_mask]
        print(valid_data.size)
        csm_data[valid_mask] = valid_data

    print(im_proj)
    print(im_geotrans)
    # 写入数据
    write_tif(csm_data, im_geotrans, im_proj, output_name) # 传递 nodata_val 参数


if __name__ == '__main__':


    dsm_filepath = r"F:/Rice2024/Meiju1/Raw/Structure/DSM.tif"

    output_folder = r"F:/Data/UAV/DJITerra_Export_Rice2024_UAV-RGB-MSI/20240911-Rice-M3M-50m-Meiju-1/CSM"

    area = "area5"
    area_dsm_path = f"F:/Data/UAV/Labels-shp/20240913-Rice-M3M-50m-Meiju-1-RGB/CSM/{area}.tif"

    area_output_path = f"F:/Data/UAV/Labels-shp/20240913-Rice-M3M-50m-Meiju-1-RGB/CSM/{area}_csm_v2.tif"

    crop_size = "v3"
    processing_block_size=16
    calc_window_size=640
    threshold = 0.8
    ignore_val = -3.4028235e+38
    file_name = f"csm_{processing_block_size}_{calc_window_size}_{threshold}_{crop_size}.tif"
    file_name = f"area1_csm.tif"

    os.makedirs(output_folder, exist_ok=True)


    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=str, default="None")
    args = parser.parse_args()
    

    start_time = datetime.now()

    # args.runs = ["create", "csm", "merge", "delete"]

    # for args.run in args.runs:
    if args.run == "create":
        crop_data(dsm_filepath, output_folder, crop_size=crop_size)
        send_email(f'数据集制作, 用时:{datetime.now() - start_time}', "数据集制作完成..")  

    elif args.run == "csm":
        convert_dsm_to_csm(output_folder)
        send_email(f'数据集制作, 用时:{datetime.now() - start_time}', "数据集制作完成..")  

    elif args.run == "all":
        convert_dsm_to_csm_file(dsm_filepath, output_folder, file_name)
        send_email(f"转换完整的数据集, 用时: {datetime.now() - start_time}")

    elif args.run == "csm_v2":
        convert_dsm_to_csm_v2(dsm_filepath, output_folder, file_name)
        send_email(f"转换成csm数据, 用时: {datetime.now() - start_time}")
        
    elif args.run == "csm_v3":
        convert_dsm_to_csm_v3(dsm_filepath, output_folder, file_name, calc_window_size=calc_window_size, processing_block_size=processing_block_size)
        send_email(f"转换成csm_v3数据, 用时: {datetime.now() - start_time}")

    elif args.run == "csm_v4":
        convert_dsm_to_csm_v4(dsm_filepath, output_folder, file_name, calc_window_size=calc_window_size, processing_block_size=processing_block_size, threshold=threshold)
        send_email(f"转换成csm_v4数据, 用时: {datetime.now() - start_time}")

    elif args.run == "csm_area":
        convert_dsm_to_csm_v5(area_dsm_path, area_output_path, calc_window_size=calc_window_size, processing_block_size=processing_block_size, ignore_val=ignore_val)
        send_email(f"转换成csm_v4数据, 用时: {datetime.now() - start_time}")


    elif args.run == "csm_area_optim": # 5. 其他区域使用之前的方法
        area_dsm_path = r"E:/Rice2024/Meiju2/Raw/Structure/Temp/ELSE_AREA/Else_Area_Erase.tif"
        area_output_path = r"E:/Rice2024/Meiju2/Raw/Structure/Temp/ELSE_AREA/Else_Area_Erase_CSM.tif"
        calc_window_size = 1280
        processing_block_size = 16
        ignore_val = -3.4028235e+38
        convert_dsm_to_csm_v5_optimized(area_dsm_path, area_output_path, calc_window_size=calc_window_size, processing_block_size=processing_block_size, ignore_val=ignore_val)
        send_email(f"转换成csm_v5_optim数据, 用时: {datetime.now() - start_time}")

    elif args.run == "csm_area_all": # 3 整块区域求取csm, 全局取1%最小值，然后分别减去
        # 全局操作
        input_folder = "E:/Rice2024/Meiju2/Raw/Structure/Temp"
        for file in os.listdir(input_folder):
            if file.endswith("_norm.tif"):
                file_path = os.path.join(input_folder, file)
                area_output_path = os.path.join(input_folder, file.replace("_norm.tif", "_norm_csm.tif"))
                print(file)
                print(area_output_path)
                convert_dsm_to_csm_v6_optimized(file_path, area_output_path, ignore_val=ignore_val)
        send_email(f"转换成csm_v6_optim数据, 用时: {datetime.now() - start_time}")
    
    elif args.run == "csm_area_all_merge":    # 4 合并
        input_folder = "E:/Rice2024/Lingtangkou/Raw/Structure/Temp"
        output_name = "area_norm_merge_v2.tif"
        merge_csm_all(input_folder, os.path.join(input_folder, output_name), suffix="_norm_csm.tif", ignore_val=ignore_val)

    elif args.run == "area_get_max_min":    # 1 先统计最大最小值和均值, 求取差值
        input_folder = "E:/Rice2024/Meiju2/Raw/Structure/Temp"
        output_csv_path = "csv/Meiju2_dsm_data.csv"
        dsm_get_max_min(input_folder, ignore_val=ignore_val, suffix=".tif",  output_csv_path=output_csv_path)
        send_email(f"转换成csm_v6_optim数据, 用时: {datetime.now() - start_time}")

    elif args.run == "norm":
        input_folder = "F:/Data/UAV/Labels-shp/20240913-Rice-M3M-50m-Meiju-1-RGB/CSM"
        # output_name = "area_csm_merge.tif"
        norm_csm_all(input_folder, suffix=".tif", ignore_val=ignore_val)

    elif args.run == "norm_v2":  # 2 进行均值“归一化”，使得不同区域均值相等
        input_folder = "E:/Rice2024/Meiju2/Raw/Structure/Temp"
        sub_values =  [0] # 注意要对应上
        # output_name = "area_csm_merge.tif"
        norm_csm_all_v2(input_folder, suffix=".tif", ignore_val=ignore_val, sub_values=sub_values)

    elif args.run == "merge_selected":  # 5. 合并指定的区域
        input_folders = ["E:/Rice2024/Lingtangkou/Raw/Structure/Temp/area_norm_merge_v2_abnormal.tif", 
                         "E:/Rice2024/Lingtangkou/Raw/Structure/Temp/ELSE_Area/Else_Area_Erase_CSM.tif"]
        output_name = "E:/Rice2024/Lingtangkou/Raw/Structure/CHM_v2.tif"
        merge_csm_selected(input_folders, output_name, ignore_val=ignore_val)

    elif args.run == "merge":
        merge_csm(output_folder, file_name)
        send_email(f"数据集合成, 用时: {datetime.now() - start_time}")

    elif args.run == "delete":
        delete_unused_folders(output_folder)
        send_email(f"删除没用的文件夹, 用时: {datetime.now() - start_time}")

    elif args.run == "abnormal":
        input_path = "E:/Rice2024/Lingtangkou/Raw/Structure/Temp/area_norm_merge_v2.tif"
        output_path = "E:/Rice2024/Lingtangkou/Raw/Structure/Temp/area_norm_merge_v2_abnormal.tif"
        im_proj, im_geotrans, im_data, im_width, im_height, im_bands = read_tif(input_path)
        
        # 获取大于0.8~1的值，如果有异常值的话
        print(im_data.max(), im_data.min())
        print(np.percentile(im_data, 99))        
        # 取0.8 ~ 1之间的值
        max_mask_start = im_data >= 0.8
        max_mask_end = im_data <= 1
        max_mask = max_mask_start & max_mask_end
        max_data = im_data[max_mask]
        # 0.99999285 0.80000573 0.8684752 0.049237646
        # 0.6 ~ 0.8 
        print(max_data.max(), max_data.min(), max_data.mean(), max_data.std())
        max_data -= 0.2
        im_data[max_mask] = max_data
        write_tif(im_data, im_geotrans, im_proj, output_path)


    
    print('run time is {}'.format(datetime.now()-start_time))



# 1. 先获取最大最小值，均值，方差等数值
# 2. 利用均值，来进行归一化
# 3. 进行csm生成
# 4. 合并
