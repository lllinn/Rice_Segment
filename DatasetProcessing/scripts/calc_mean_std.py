import sys
sys.path.append('./')
from osgeo import gdal
import os
from tqdm import tqdm
import json
import argparse
from src.utils.email_utils import send_email
from src.utils.file_io import read_tif, write_tif
import pandas as pd # 导入 pandas 库
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import matplotlib.font_manager as fm # 导入字体管理器
from scipy import stats # 导入 scipy.stats 库
import matplotlib as mpl # For rcParams
import re

# --- 解决中文显示问题 (推荐使用指定字体文件的方法) ---
# 请将 'C:/Windows/Fonts/simhei.ttf' 替换为你找到的中文字体文件的实际路径
# 如果你不确定路径，可以先运行前面提供的字体列表代码来查找
# 注意 r'' 前缀表示这是一个原始字符串，可以避免反斜杠的转义问题
# 在 Windows 系统上，常见的字体路径示例如下：
# r'C:\Windows\Fonts\simhei.ttf'  # 黑体
# r'C:\Windows\Fonts\msyh.ttc'    # 微软雅黑 (注意是 .ttc)
# r'C:\Windows\Fonts\simsun.ttc'  # 宋体 (注意是 .ttc)
chinese_font_path = r'C:\Windows\Fonts\simhei.ttf' # <-- **请修改为你的字体文件路径**

# 创建 FontProperties 对象，直接指向字体文件
font_properties = None
if os.path.exists(chinese_font_path):
    try:
        font_properties = fm.FontProperties(fname=chinese_font_path)
        print(f"成功加载指定字体文件: {chinese_font_path}")
    except Exception as e:
        print(f"加载字体文件时发生错误: {e}")
        font_properties = None
else:
    print(f"警告: 指定的字体文件不存在！请检查路径: {chinese_font_path}")
    print("将尝试使用 Matplotlib 默认字体或 RCParams 设置。")
    # 如果字体文件不存在，可以尝试回退到 RCParams 设置，前提是前面已经设置过
    # 注意：这里不再次设置 RCParams，假设你已经在脚本开头设置过
    pass

# 确保负号正确显示
mpl.rcParams['axes.unicode_minus'] = False


def calc_mean_and_std(filepath, label_path, calc_logs=None, ignore_value=1e-34, excel_output_path=None):
    """
    Calculates band statistics (ignore count, min, max) for a GeoTIFF file,
    optionally saves logs to JSON, and writes stats to an Excel file.

    Args:
        filepath (str): Path to the input GeoTIFF file.
        label_path (str): Path to the label GeoTIFF file. Non-zero values in the label file
                          indicate valid regions for calculation.
        calc_logs (str, optional): Path to save/load calculation history in JSON format.
                                   Defaults to None (no JSON logging).
        ignore_value (float, optional): Value to ignore in the input data.
                                        Defaults to 1e-34.
        excel_output_path (str, optional): Path to the output Excel file.
                                           If the file exists, new data will be appended
                                           to the 'Band Stats' sheet. Defaults to None
                                           (no Excel output).
    """
    records = {}
    if os.path.exists(calc_logs):
        with open(calc_logs, 'r') as f:
            records = json.load(f)
        print(f"已加载历史记录：{len(records)} 条")
    else:
        os.makedirs(os.path.dirname(calc_logs), exist_ok=True)
    
    
    dataset = gdal.Open(filepath)
    bands = dataset.RasterCount
    
    # 读取Label数据
    _, _, label_data, _, _, _ = read_tif(label_path)
    label_mask = label_data != 0

    band_stats_list = [] # List to collect stats for the current file (for Excel)

    # 遍历所有波段，计算每个波段的平均值和标准差
    for i in range(1, bands + 1):
        band = dataset.GetRasterBand(i)
        data = band.ReadAsArray()
        # 忽略值
        valid_data = data[(data != ignore_value) & label_mask]
        valid_max = valid_data.max()
        valid_min = valid_data.min()
        valid_max_99_9999 = np.percentile(valid_data, 99.9999)
        valid_max_99 = np.percentile(valid_data, 99)
        valie_min_1 = np.percentile(valid_data, 1)
        valid_max_98 = np.percentile(valid_data, 98)
        valid_min_2 = np.percentile(valid_data, 2)
        # # 归一化处理
        # valid_data = (valid_data - valid_min) / (valid_max - valid_min)
        # # 计算均值方差
        # mean = valid_data.mean()
        # std = valid_data.std()
        # 计算一共忽略的值
        ignore_num = data.size - valid_data.size
        # 存储到 records (用于 JSON 日志)
        records[filepath + f"-band{i}"] = {
            # 'mean': float(mean), 
            # 'std': float(std), 
            'ignore_num' : int(ignore_num),
            'max': float(valid_max),
            'min': float(valid_min),
            'max_99': float(valid_max_99),
            'min_1': float(valie_min_1),
            'max_98': float(valid_max_98),
            'min_2': float(valid_min_2),
        }
        
        # 收集数据到 list (用于 Excel 输出)
        band_stats_list.append({
            'Filepath': filepath,
            'Band': i,
            'Ignore Count': int(ignore_num), # Convert to int
            'Max Value': valid_max,
            'Min Value': valid_min,
            'Max Value 99%': valid_max_99,
            'Min Value 1%': valie_min_1,
            'Max Value 98%': valid_max_98,
            'Min Value 2%': valid_min_2,
            # Add 'Mean', 'Std' here if calculated and needed in Excel
        })


        print(f'波段 {i}: \
              忽略值数量: {ignore_num}, \
              最大值: {valid_max:.5f},最小值: {valid_min:.5f},\
                最大值99%: {valid_max_99:.5f},最小值1%: {valie_min_1:.5f},\
                最大值98%: {valid_max_98:.5f},最小值2%: {valid_min_2}\n')
        
        # print(f"波段 {i}: 平均值: {mean:.2f}, 标准差: {std:.2f},忽略值数量: {ignore_num}")
    with open(calc_logs, 'w') as f:
        json.dump(records, f)


    # 将统计数据写入 Excel 文件
    if excel_output_path and band_stats_list: # Only write if path is provided and there are stats
        new_df = pd.DataFrame(band_stats_list)
        combined_df = pd.DataFrame() # Initialize empty DataFrame

        excel_sheet_name = 'Band Stats' # Define a sheet name for the output

        # 检查 Excel 文件是否存在，如果存在则读取现有数据并追加
        if os.path.exists(excel_output_path):
            try:
                # 读取现有数据，指定工作表名称
                # 使用 engine='openpyxl' 是为了兼容性
                existing_df = pd.read_excel(excel_output_path, sheet_name=excel_sheet_name, engine='openpyxl')
                # 合并新旧数据
                combined_df = pd.concat([existing_df, new_df], ignore_index=True)
                print(f"数据已追加到Excel文件: {excel_output_path}")
            except FileNotFoundError:
                 # This should not happen if os.path.exists is True, but good practice
                 print(f"警告：尝试读取Excel文件 {excel_output_path} 失败，将创建新文件。")
                 combined_df = new_df
            except ValueError:
                # Happens if the sheet_name does not exist in the file
                print(f"警告：Excel文件 {excel_output_path} 中未找到工作表 '{excel_sheet_name}'，将创建此工作表。")
                combined_df = new_df
            except Exception as e:
                print(f"读取或处理现有Excel文件时发生错误: {e}，将仅写入新数据。")
                combined_df = new_df # Fallback to writing just the new data
        else:
            # 如果文件不存在，则直接使用新数据
            combined_df = new_df
            print(f"创建新的Excel文件: {excel_output_path}")

        # 写入（或覆盖）Excel 文件
        try:
             # 确保输出目录存在
             os.makedirs(os.path.dirname(excel_output_path), exist_ok=True)
             # 将合并后的 DataFrame 写入 Excel
             # index=False 表示不写入 DataFrame 的索引列
             combined_df.to_excel(excel_output_path, sheet_name=excel_sheet_name, index=False, engine='openpyxl')
             print(f"统计数据已写入或追加到Excel文件: {excel_output_path}")
        except Exception as e:
             print(f"写入数据到Excel文件 {excel_output_path} 时发生错误: {e}")


def plot_single_histogram(data, save_path, bins=30, title="Histogram", xlabel="Value", ylabel="Frequency"):
    """
    Generates a single histogram using seaborn and matplotlib and saves it to a file.

    Args:
        data: The data for the histogram (expected to be a 1D array or Series).
        save_path (str): The full path including filename and extension to save the plot.
        bins (int or sequence): Number of bins (int) or bin edges (sequence). Defaults to 30.
        title (str): The title of the plot. Defaults to "Histogram".
        xlabel (str): The label for the x-axis. Defaults to "Value".
        ylabel (str): The label for the y-axis. Defaults to "Frequency".
    """
    print(f"\n--- Generating single histogram and saving to {save_path} ---")

    # 检查输入数据类型，并尝试转换为 NumPy 数组
    if not isinstance(data, np.ndarray) and not isinstance(data, pd.Series):
         print("警告: 输入数据不是 NumPy 数组或 Pandas Series，可能无法正确绘制。")
         try:
             data = np.asarray(data)
             print("尝试将数据转换为 NumPy 数组。")
             if data.ndim > 1:
                 print("警告: 输入数据有多维，直方图将使用展平后的数据。")
                 data = data.flatten() # 如果是多维，展平
         except Exception as e:
             print(f"错误: 无法将输入数据转换为 NumPy 数组。{e}")
             print("--- Histogram generation skipped ---")
             return
    elif isinstance(data, np.ndarray) and data.ndim > 1:
         print("警告: 输入数据是多维 NumPy 数组，直方图将使用展平后的数据。")
         data = data.flatten() # 如果是多维，展平


    # Create figure and axes
    fig, ax = plt.subplots(figsize=(8, 6)) # Adjust figure size as needed

    # Generate the histogram plot
    try:
        # sns.histplot() takes data, bins, and ax
        # For 1D data, just passing data and bins works
        sns.histplot(data=data, bins=bins, ax=ax, kde=True) # **主要修改：使用 sns.histplot**
    except Exception as e:
        print(f"错误: 使用 sns.histplot 绘制时发生错误: {e}")
        plt.close(fig) # Close the figure if plotting fails
        print("--- Histogram generation skipped ---")
        return

    # Set title and labels, applying font properties if available
    # ylabel 默认值从 Density 改为 Frequency (频数)
    if 'font_properties' in globals() and font_properties is not None:
        ax.set_title(title, fontproperties=font_properties, fontsize=14)
        ax.set_xlabel(xlabel, fontproperties=font_properties, fontsize=12)
        ax.set_ylabel(ylabel, fontproperties=font_properties, fontsize=12)
    else:
        ax.set_title(title, fontsize=14)
        ax.set_xlabel(xlabel, fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)

    # Add grid (optional - less common for histograms showing counts, but possible)
    # ax.grid(True, linestyle='--', alpha=0.6)

    # Adjust layout
    plt.tight_layout()

    # Save the plot
    try:
        # Ensure the directory exists
        save_dir = os.path.dirname(save_path)
        if save_dir and not os.path.exists(save_dir):
            os.makedirs(save_dir)
            print(f"Created directory: {save_dir}")

        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved successfully to {save_path}")
    except Exception as e:
        print(f"Error saving plot to {save_path}: {e}")

    # Close the figure to free memory and prevent display
    plt.close(fig)
    print("Figure closed.")
    print("--- Histogram generation complete ---")


def visualize_feature(filepath, label_path, ignore_value, band_index, output_folder, max_value_per=99, min_value_per=1):
    '''
    band_index: index对应的值，从1开始
    '''
    dataset = gdal.Open(filepath)
    bands = dataset.RasterCount
    band = dataset.GetRasterBand(band_index)

    data = band.ReadAsArray()
    im_width = dataset.RasterXSize
    im_height = dataset.RasterYSize
    im_bands = dataset.RasterCount
    im_geotrans = dataset.GetGeoTransform()
    im_proj = dataset.GetProjection()

    # Label 数据
    _, _, label_data, _, _, _ = read_tif(label_path)
    label_mask = label_data != 0   # label有效mask
    valid_mask = (data != ignore_value) & label_mask

    valid_data = data[valid_mask] # 有效数据

    # 创建一个空值
    # label_data = np.zeros_like(data, dtype=np.uint8)
    # label_data = np.zeros((1, im_height, im_width), dtype=np.uint8)

    # 获取99%最大值
    max_value = np.percentile(valid_data, max_value_per) # 99% 最大值
    min_value = np.percentile(valid_data, min_value_per)  # 1%  最小值

    # max_mask = data >= max_value
    # min_mask = data <= min_value
    # 将大于等于99%最大值的像素标记为1
    # label_data[max_mask & valid_mask] = 1
    # label_data[min_mask & valid_mask] = 2
    
    print(f"Max value {max_value_per}% is {max_value}, Min value {min_value_per}% is {min_value}")
    
    # 保存label为tif文件
    # write_tif(label_data, im_geotrans, im_proj, os.path.join(output_folder, f"visualization_B{band_index}_99.9999%_Max_Value.tif"))
    # print(f"Write label image completed...")
    
    plot_single_histogram(valid_data[valid_data >= max_value], os.path.join(output_folder, f"visualization_B{band_index}_{max_value_per}%_max_value.png"), title=f"Hist Plot", 
                          xlabel=f"{max_value_per}% Max Value")
    print(f"Save {max_value_per}% max value kde plot completed...")
    
    plot_single_histogram(valid_data[valid_data <= min_value], os.path.join(output_folder, f"visualization_B{band_index}_{min_value_per}%_min_value.png"), title=f"Hist Plot", 
                          xlabel=f"{min_value_per}% Min Value")
    print(f"Save 1% min value kde plot completed...")

    # 把值打印出来
    max_values_print = valid_data[valid_data >= max_value]
    max_values_print = sorted(max_values_print, reverse=True)
    
    with open(os.path.join(output_folder, f"visualization_B{band_index}_{max_value_per}%_max_value.txt"), mode='w') as f:
        for i, value in enumerate(max_values_print):
            f.write(f"{i+1}. {value}\n")

    # 把值打印出来
    min_values_print = valid_data[valid_data >= min_value]
    min_values_print = sorted(max_values_print, reverse=True)
    
    with open(os.path.join(output_folder, f"visualization_B{band_index}_{min_value_per}%_max_value.txt"), mode='w') as f:
        for i, value in enumerate(min_values_print):
            f.write(f"{i+1}. {value}\n")



import os
from PIL import Image, ImageDraw, ImageFont
import math
import sys # 导入 sys 模块用于检查 Pillow 版本

def combine_images_from_folder(input_folder, output_path="combined_image.png", cols=2, padding=20, title_height=40, font_path=None, font_size=25, background_color=(255, 255, 255), title_color=(0, 0, 0)):
    """
    查找指定文件夹下的所有图片文件，并将它们拼接成一张图，并在每张图上方添加文件名作为标题。

    Args:
        input_folder (str): 包含图片文件的文件夹路径。
        output_path (str): 输出文件的保存路径和名称。
        cols (int): 每行的图片数量。
        padding (int): 图片和标题之间的以及图片边缘的留白（像素）。
        title_height (int): 每张图片上方标题区域的高度（像素）。
        font_path (str, optional): 字体文件的路径（如 .ttf 文件）。如果为 None，将使用默认字体。
        font_size (int): 标题的字体大小。
        background_color (tuple): 输出图片的背景颜色 (R, G, B)。默认为白色。
        title_color (tuple): 标题文字的颜色 (R, G, B)。默认为黑色。
    """

    if not os.path.isdir(input_folder):
        print(f"错误：输入的路径不是一个有效的文件夹：{input_folder}")
        return

    image_paths = []
    # 支持的图片扩展名 (不区分大小写)
    image_extensions = ('.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tiff')

    # 1. 查找文件夹下的所有图片文件
    print(f"正在查找文件夹 {input_folder} 下的图片文件...")
    output_name = os.path.basename(input_folder)
    try:
        # os.listdir 列出文件夹内容，不包括子文件夹
        for item in os.listdir(input_folder):
            full_path = os.path.join(input_folder, item)
            if os.path.isfile(full_path) and item.lower().endswith(image_extensions):
                if full_path == output_path:
                    continue
                image_paths.append(full_path)
                print(f"找到图片文件: {full_path}")
    except Exception as e:
        print(f"查找文件时发生错误：{e}")
        return


    if not image_paths:
        print(f"在文件夹 {input_folder} 中没有找到支持的图片文件。")
        return

    # 按文件名排序，使输出顺序可预测
    image_paths.sort()

    images = []
    titles = []
    max_width = 0
    max_height = 0
    valid_images_count = 0

    # 2. 打开所有图片并获取信息
    print("正在读取图片和提取标题...")
    for path in image_paths:
        try:
            img = Image.open(path).convert("RGB") # 转换为RGB模式以避免潜在的通道问题
            images.append(img)
            # 提取文件名作为标题 (去除路径和后缀)
            title = os.path.splitext(os.path.basename(path))[0]
            titles.append(title)

            max_width = max(max_width, img.width)
            max_height = max(max_height, img.height)
            valid_images_count += 1
            # print(f"成功读取: {path}") # 可以注释掉，避免输出过多
        except Exception as e:
            print(f"警告：无法打开或处理文件，跳过 {path}: {e}")

    if valid_images_count == 0:
        print("没有找到有效的图片文件。")
        return

    # 使用找到的最大宽度和高度作为拼接时每个“瓦片”的标准大小
    tile_width = max_width
    tile_height = max_height

    # 3. 计算布局参数
    rows = math.ceil(valid_images_count / cols)
    canvas_width = cols * tile_width + (cols + 1) * padding
    canvas_height = rows * tile_height + rows * title_height + (rows + 1) * padding

    # 4. 创建一个空白画布
    print(f"正在创建画布 ({canvas_width}x{canvas_height})...")
    canvas = Image.new('RGB', (canvas_width, canvas_height), color=background_color)
    draw = ImageDraw.Draw(canvas)

    # 5. 加载字体
    font = None
    try:
        if font_path and os.path.exists(font_path):
            font = ImageFont.truetype(font_path, font_size)
            print(f"使用指定字体: {font_path}")
        else:
             # 尝试加载一个常见的系统字体或使用默认（可能不太美观）
            try:
                # 尝试加载arial，如果失败再尝试加载默认
                font = ImageFont.truetype("arial.ttf", font_size)
                print("使用系统默认字体: arial.ttf")
            except IOError:
                 print("警告：未找到arial.ttf字体，使用Pillow默认字体。考虑指定font_path参数获取更好的效果。")
                 font = ImageFont.load_default()
                 # Adjust title_height and canvas if default font is used and no font_path was given
                 if font_path is None:
                     # Default font has fixed size, need to recalculate title_height and canvas
                     # Get text size for default font (e.g., "Test")
                     try:
                         # Pillow 8.0+
                         text_bbox = draw.textbbox((0, 0), "Test", font=font)
                         default_text_height = text_bbox[3] - text_bbox[1]
                     except AttributeError:
                          # Older Pillow versions
                         default_text_height = draw.textsize("Test", font=font)[1]

                     # Make title_height just enough for the default font text plus a little buffer
                     title_height = default_text_height + padding // 2
                     print(f"调整标题高度为 {title_height} 适应默认字体。")
                     # Recalculate canvas height based on new title_height
                     canvas_height = rows * tile_height + rows * title_height + (rows + 1) * padding
                     canvas = Image.new('RGB', (canvas_width, canvas_height), color=background_color)
                     draw = ImageDraw.Draw(canvas) # Need to recreate draw object for the new canvas

    except Exception as e:
        print(f"警告：加载字体失败，将不绘制标题文本：{e}")
        font = None # Ensure font is None if loading failed


    # 6. 将图片粘贴到画布上并绘制标题
    print("正在粘贴图片和绘制标题...")
    img_index = 0
    for r in range(rows):
        for c in range(cols):
            if img_index < valid_images_count:
                img = images[img_index]
                title = titles[img_index]

                # 计算当前图片和标题的位置
                x_offset = c * (tile_width + padding) + padding
                y_offset = r * (tile_height + title_height + padding) + padding

                # 调整图片大小到统一尺寸
                # 注意：简单的resize可能会改变图片的纵横比，如果需要保持纵横比，需要更复杂的逻辑
                img_resized = img.resize((tile_width, tile_height))

                # 将图片粘贴到画布上 (图片位置在标题下方)
                canvas.paste(img_resized, (x_offset, y_offset + title_height))

                # 绘制标题
                if draw and font:
                    # 计算文本位置（居中）
                    try:
                         # Pillow 8.0+ uses textbbox for more accurate size
                        text_bbox = draw.textbbox((0, 0), title, font=font)
                        text_width = text_bbox[2] - text_bbox[0]
                        text_height = text_bbox[3] - text_bbox[1] # Use height from bbox
                    except AttributeError:
                         # For older Pillow versions
                         text_width, text_height = draw.textsize(title, font=font)


                    text_x = x_offset + (tile_width - text_width) // 2
                    text_y = y_offset + (title_height - text_height) // 2 # 垂直居中在标题区域

                    # Ensure text_y is not negative due to small title_height or large text
                    text_y = max(y_offset, text_y)


                    draw.text((text_x, text_y), title, fill=title_color, font=font)

                img_index += 1

    # 7. 保存结果
    try:
        # 确保输出文件夹存在
        output_dir = os.path.dirname(output_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
            print(f"创建输出文件夹: {output_dir}")

        canvas.save(output_path)
        print(f"图片已成功保存到: {output_path}")
    except Exception as e:
        print(f"保存图片失败: {e}")



def normlize_tif(input_path, output_path, band_index=1, max_value = 1.11904764175415, min_value = 0.5):
    dataset = gdal.Open(input_path)
    bands = dataset.RasterCount
    band = dataset.GetRasterBand(band_index)

    data = band.ReadAsArray()
    im_width = dataset.RasterXSize
    im_height = dataset.RasterYSize
    im_bands = dataset.RasterCount
    im_geotrans = dataset.GetGeoTransform()
    im_proj = dataset.GetProjection()

    data = (data - min_value) / (max_value - min_value)
    write_tif(data, im_geotrans, im_proj, output_path)

    # 进行归一化
    # pass


def standard_tif(intput_path, output_path, label_path, band_index=1, ignore_value=1e-34):
    dataset = gdal.Open(input_path)
    bands = dataset.RasterCount
    band = dataset.GetRasterBand(band_index)

    data = band.ReadAsArray()
    im_width = dataset.RasterXSize
    im_height = dataset.RasterYSize
    im_bands = dataset.RasterCount
    im_geotrans = dataset.GetGeoTransform()
    im_proj = dataset.GetProjection()

    # Label 数据
    _, _, label_data, _, _, _ = read_tif(label_path)
    label_mask = label_data != 0   # label有效mask
    valid_mask = (data != ignore_value) & label_mask
    
    valid_data = data[valid_mask]
    print(f'有效像素占比: {len(valid_data)/data.size}%')
    mean = np.mean(valid_data)
    std = np.std(valid_data)
    print(f'均值: {mean}, 标准差: {std}')
    data = (data - mean) / std
    write_tif(data, im_geotrans, im_proj, output_path)

def get_feature_band(input_path, output_folder, start_x, start_y, end_x, end_y, band_num=54):
    dataset = gdal.Open(input_path)
    bands = dataset.RasterCount
    for band_idx in tqdm(range(1, band_num+1), unit="bands", ncols=100):
        print(f"\n now band is {band_idx:02d}")
        band = dataset.GetRasterBand(band_idx)

        data = band.ReadAsArray()
        im_width = dataset.RasterXSize
        im_height = dataset.RasterYSize
        im_bands = dataset.RasterCount
        im_geotrans = dataset.GetGeoTransform()
        im_proj = dataset.GetProjection()
        print(data.shape)
        new_data = data[start_y:end_y, start_x:end_x]
        output_name = f"B{band_idx:02d}_.tif"
        output_path = os.path.join(output_folder, output_name)
        print(output_path)
        write_tif(new_data, im_geotrans, im_proj, output_path)



def rename_files_by_group(folder_path):
    """
    根据预定义的组别和规则重命名文件夹中的 TIFF 文件。

    Args:
        folder_path (str): 包含要重命名文件的文件夹路径。
    """
    if not os.path.isdir(folder_path):
        print(f"错误：找不到文件夹 '{folder_path}'")
        return

    # 定义组别、波段范围以及特定波段后缀
    # start_index 和 end_index 是波段号（从 1 开始）
    # suffixes 是对应波段的特定后缀列表，如果为 None 则只使用组名
    groups = [
        {'name': 'RGB', 'start_index': 1, 'end_index': 3, 'suffixes': ['R', 'G', 'B']},
        {'name': 'Color', 'start_index': 4, 'end_index': 13, 'suffixes': ['RGRI','RGBVI','NGRDI', 'NGBDI', 'NDI', 'MGRVI', 'INT', 'IKAW',
                                                                          'ExR', 'ExG']},
        {'name': 'Spectral', 'start_index': 14, 'end_index': 30, 'suffixes': ['Green', "Red",  "RE", "NIR", "WDRI", "SAVI","RVI","PVI","NGRDI","NDWI","NDVI","NDRE","GI","DVI","CVI","CIredge","CIgre"]},
        {'name': 'Texture_R', 'start_index': 31, 'end_index': 38, 'suffixes': ["Mean", "Variance", "Homogeneity", "Contrast", "Dissimilarity", "Entropy", "Second_Moment", "Correlation"]},
        {'name': 'Texture_G', 'start_index': 39, 'end_index': 46, 'suffixes': ["Mean", "Variance", "Homogeneity", "Contrast", "Dissimilarity", "Entropy", "Second_Moment", "Correlation"]},
        {'name': 'Texture_B', 'start_index': 47, 'end_index': 54, 'suffixes': ["Mean", "Variance", "Homogeneity", "Contrast", "Dissimilarity", "Entropy", "Second_Moment", "Correlation"]},

        # 如果有其他组，可以在这里添加
        # {'name': 'AnotherGroup', 'start_index': 55, 'end_index': 60, 'suffixes': [...]},
    ]

    # 正则表达式匹配 Bxx_.tif 文件
    file_pattern = re.compile(r'^B(\d{2})_\.tif$', re.IGNORECASE)

    print(f"正在扫描文件夹: {folder_path}")

    renamed_count = 0
    skipped_count = 0
    not_matched_count = 0

    # 遍历文件夹中的所有文件
    for filename in os.listdir(folder_path):
        match = file_pattern.match(filename)
        if match:
            # 提取波段编号
            band_number_str = match.group(1)
            band_number = int(band_number_str)

            original_filepath = os.path.join(folder_path, filename)

            # 查找文件所属的组别
            target_group = None
            for group in groups:
                if group['start_index'] <= band_number <= group['end_index']:
                    target_group = group
                    break # 找到组后停止查找

            if target_group:
                new_filename_base = f"B{band_number_str}_{target_group['name']}"
                new_filename_suffix = ""

                # 检查是否有特定波段后缀
                if target_group['suffixes']:
                    # 计算当前波段在组内的索引
                    index_in_group = band_number - target_group['start_index']
                    if 0 <= index_in_group < len(target_group['suffixes']):
                         new_filename_suffix = target_group['suffixes'][index_in_group]
                    else:
                         print(f"警告: 文件 {filename} 属于组 {target_group['name']} (波段 {band_number})，但在后缀列表中找不到对应索引 {index_in_group}。将仅使用组名重命名。")


                new_filename = f"{new_filename_base}-{new_filename_suffix}.tif"
                new_filepath = os.path.join(folder_path, new_filename)

                # 检查新文件名是否与原文件名相同
                if original_filepath == new_filepath:
                    print(f"跳过: 文件 {filename} 无需重命名。")
                    skipped_count += 1
                # 检查目标文件是否已经存在
                elif os.path.exists(new_filepath):
                     print(f"跳过: 目标文件 {new_filename} 已存在，未重命名 {filename}。")
                     skipped_count += 1
                else:
                    try:
                        os.rename(original_filepath, new_filepath)
                        print(f"成功重命名: {filename} -> {new_filename}")
                        renamed_count += 1
                    except OSError as e:
                        print(f"错误: 无法重命名文件 {filename} 到 {new_filename} - {e}")
                        skipped_count += 1
            else:
                print(f"警告: 文件 {filename} (波段 {band_number}) 不属于任何已定义的组，跳过重命名。")
                not_matched_count += 1
        else:
            # 文件名不符合 Bxx_.tif 格式
            # print(f"跳过: 文件 {filename} 不符合预期的命名格式。") # 可以选择打印非匹配文件
            pass # 通常我们只关心符合格式的文件，所以可以忽略不符合的

    print("\n--- 重命名摘要 ---")
    print(f"成功重命名文件数: {renamed_count}")
    print(f"跳过文件数 (已是目标名称或目标已存在/其他错误): {skipped_count}")
    print(f"未匹配组的文件数: {not_matched_count}")
    print(f"总共处理符合格式的文件数: {renamed_count + skipped_count + not_matched_count}")


# --- 主程序 ---
# if __name__ == "__main__":
#     # 获取用户输入的文件夹路径
#     target_folder = input("请输入包含要重命名文件的文件夹路径: ")

#     # 调用重命名函数
#     rename_files_by_group(target_folder)
    


# --- 使用示例 ---
if __name__ == "__main__":
    # input_folder = "G:/Rice2024/Meiju1/Stack/ROI"
    # rename_files_by_group(input_folder)
    # for i in range(1, 55):
    #     with open(os.path.join(input_folder, f"B{i:02d}_.tif"), 'w') as f:
    #         pass
    
    input_path = "F:/Rice2024/Meiju1/Stack/RGB/Stack_R-CSM.dat"
    output_folder = "G:/Rice2024/Meiju1/Stack/ROI"
    start_x = 11570
    end_x = 38037
    start_y = 28497
    end_y = 53153
    get_feature_band(input_path, output_folder, start_x, start_y, end_x, end_y, band_num=2)
    # input_path = "F:/Rice2024/Meiju1/Split_Stretch/2_CIs-10/B4-B7/Split_Stretch_B4-B7.tif"
    # output_path = "G:/Rice2024/Meiju1/Split_Stretch/2_CIs-10/B4-B7/Split_Stretch_Standard_B4-B7.tif"
    # label_path = "F:/Rice2024/Meiju1/Labels-shp/Meiju1_2_Lingtangkou_v5.tif"
    # # normlize_tif(input_path, output_path, max_value=26.8000011444092, min_value=0.00689655169844627)

    # standard_tif(input_path, output_path, label_path)
    # # 将你的输入文件夹路径替换下面的字符串
    # input_folder_path = "G:/Rice2024/Meiju1/Split_Stretch/2_CIs-10/B4-B7" # 包含你的图片的文件夹


    # output_name = "combined_visualization_from_folder.png"

    # # 设置输出文件路径
    # output_file = os.path.join(input_folder_path, output_name) # 你希望保存合并图片的路径

    # # 调用函数进行拼接
    # # 可以调整 cols 参数来改变每行的图片数量，padding, title_height, font_size 等参数调整样式
    # # 如果需要指定字体文件，修改 font_path 参数，例如: font_path="C:/Windows/Fonts/simhei.ttf" (宋体或其他字体文件路径)
    # combine_images_from_folder(
    #     input_folder=input_folder_path,
    #     output_path=output_file,
    #     cols=2, # 例如，每行放 2 张图
    #     padding=30, # 增加留白
    #     title_height=50, # 增加标题高度
    #     font_size=50, # 调整字体大小
    #     # font_path="C:/Windows/Fonts/arial.ttf" # 可选：指定字体文件路径
    # )


    # parser = argparse.ArgumentParser()
    # parser.add_argument("--path", type=str, default="rgb_u8_v2.tif", help="input data path")
    # parser.add_argument("--ignore", type=float, default=1e-34, help="repetition rate")

    # args = parser.parse_args()
    # calc_mean_and_std(args.path, ignore_value=args.ignore)

    # path_list = [
    #     "D:/GLCM/Meiju1/Stack/split/Band1-Texture/band21to24/split_band21to24_tosame.tif",
    #     "D:/GLCM/Meiju1/Stack/split/Band1-Texture/band25to28/split_band25to28_tosame.tif",
    #     "D:/GLCM/Meiju1/Stack/split/Band2-Texture/band29to32/split_band29to32_tosame.tif",
    #     "D:/GLCM/Meiju1/Stack/split/Band2-Texture/band33to36/split_band33to36_tosame.tif",
    #     "D:/GLCM/Meiju1/Stack/split/Band3-Texture/band37to40/split_band37to40_tosame.tif",
    #     "D:/GLCM/Meiju1/Stack/split/Band3-Texture/band41to44/split_band41to44_tosame.tif",
    #     "D:/GLCM/Meiju1/Stack/split/DSM/split_dsm_tosame.tif",
    # ]
    # ignores = [
    #     0, 0, 0, 0, 0, 0, -9999
    # ]
    # for path, ignore in tqdm(zip(path_list, ignores)):
    #     calc_mean_and_std(path, ignore_value=ignore)
    # send_email("标准化计算完成，结果见附件。")