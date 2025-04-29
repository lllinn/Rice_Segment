import numpy as np
from osgeo import gdal
import os
from tqdm import tqdm
from PIL import Image
from ..utils.file_io import write_tif, read_tif

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


def convert_15_bands_tif_to_npy(input_folder, output_folder):
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
            output_file = os.path.join(output_folder, os.path.splitext(filename)[0] + ".npy")
            
            dataset = gdal.Open(input_file, gdal.GA_ReadOnly)  
            # 读取图像的所有波段数据为NumPy数组  
            data = []
            for i in range(1, 16):
                band = dataset.GetRasterBand(i)
                data.append(band.ReadAsArray())

            npy_data = np.stack(data, axis=-1).astype(np.float32)
            # 保存为.npy文件（自动保留原始数据类型）
            np.save(output_file, npy_data)
            # # 将波段数据堆叠为RGB图像
            # if len(data) == 3:
            #     image = Image.fromarray(np.stack(data, axis=-1).astype(np.uint8))
            # else:
            #     raise ValueError("Image does not have exactly 3 bands, cannot convert to RGB")

            # 保存为png图片
            # image.save(output_file)

def convert_one_band_tif_to_png_file(input_file, output_file):
    """
    将单波段TIFF图像转换为PNG图像。
    
    Args:
        input_file (str): 输入TIFF图像所在的文件路径。
        output_file (str): 输出PNG图像所在的文件路径。
    
    Returns:
        None
    
    """

    print("Converting one band TIFF images to PNG.....................")    
    _, _, im_data, _, _, _ = read_tif(input_file)

    # 将数据转换为灰度图像
    image = Image.fromarray(im_data.astype(np.uint8))
    image = image.convert('L')

    # 保存为png图片
    image.save(output_file)


def convert_three_bands_tif_to_png_file(input_file, output_file):
    """
    将三波段TIFF图像转换为PNG图像。
    
    Args:
        input_file (str): 输入TIFF图像所在的文件路径。
        output_file (str): 输出PNG图像所在的文件路径。
    
    Returns:
        None
    
    """

    print("Converting three bands TIFF images to PNG.....................")
    
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



def gdal_type_to_numpy(gdal_dtype: int) -> np.dtype:
    """GDAL数据类型转numpy数据类型"""
    type_map = {
        gdal.GDT_Byte: np.uint8,
        gdal.GDT_UInt16: np.uint16,
        gdal.GDT_Int16: np.int16,
        gdal.GDT_UInt32: np.uint32,
        gdal.GDT_Int32: np.int32,
        gdal.GDT_Float32: np.float32,
        gdal.GDT_Float64: np.float64,
        gdal.GDT_CInt16: np.complex64,
        gdal.GDT_CInt32: np.complex64,
        gdal.GDT_CFloat32: np.complex64,
        gdal.GDT_CFloat64: np.complex128,
    }
    return type_map.get(gdal_dtype, np.float32)  # 默认float32


def tif_to_npy(tif_path: str, npy_path: str):
    """
    将TIF文件转换为NPY格式，自动处理单通道/多通道
    
    Args:
        tif_path (str): 输入TIF文件路径
        npy_path (str): 输出NPY文件路径
        
    Returns:
        tuple: (影像数据, 地理变换, 投影信息)
    """
    # 打开TIF文件
    dataset = gdal.Open(tif_path)
    if dataset is None:
        raise IOError(f"无法打开TIF文件: {tif_path}")
    
    # 获取基本信息
    width = dataset.RasterXSize
    height = dataset.RasterYSize
    bands = dataset.RasterCount
    dtype = dataset.GetRasterBand(1).DataType
    
    # 创建目标数组 (H, W, C)
    np_data = np.zeros((height, width, bands), 
                      dtype=gdal_type_to_numpy(dtype))
    
    # 逐波段读取数据
    for band_idx in range(bands):
        band = dataset.GetRasterBand(band_idx + 1)  # GDAL波段从1开始
        np_data[:, :, band_idx] = band.ReadAsArray()
    
    # # 处理单通道数据的维度
    # if bands == 1:
    #     np_data = np.squeeze(np_data, axis=2)  # 移除多余维度
    #     np_data = np.expand_dims(np_data, axis=-1)  # 强制添加通道维度
    
    # 确保输出目录存在
    # os.makedirs(os.path.dirname(npy_path), exist_ok=True)
    
    # 保存numpy文件
    np.save(npy_path, np_data)

def tif_to_npy_folder(input_folder, output_folder):
    """
    批量转换TIF文件夹到NPY格式，自动处理单通道/多通道
    
    Args:
        input_folder (str): 输入TIF文件夹路径
        output_folder (str): 输出NPY文件夹路径
        
    Returns:
        None
    """
    os.makedirs(output_folder, exist_ok=True)
    # 遍历输入文件夹下的所有TIF文件
    for file in tqdm(os.listdir(input_folder), desc="Tif to npy...", unit="files"):
        if file.lower().endswith('.tif'):
            input_path = os.path.join(input_folder, file)
            output_path = os.path.join(output_folder, file.replace(".tif", ".npy"))
            
            tif_to_npy(input_path, output_path)



if __name__ == '__main__':
    # convert_one_band_tif_to_png_file(r"F:\Data\UAV\Pix4DfIelds_Export\20240913-Rice-M3M-50m-Meiju-1-RGB\Labels\merge_Out_v22.tif", 
    #                             r"F:\Data\UAV\Pix4DfIelds_Export\20240913-Rice-M3M-50m-Meiju-1-RGB\Labels\merge_Out_v22.png")
    # python -m src.processing.convert
    # convert_three_bands_tif_to_png_file(r"F:\Data\UAV\Pix4DfIelds_Export\20240913-Rice-M3M-50m-Meiju-1-RGB\Labels\merge_Out_v22.tif",
    #                                     r"F:\Data\UAV\Pix4DfIelds_Export\20240913-Rice-M3M-50m-Meiju-1-RGB\Labels\merge_Out_v22.png")
    convert_one_band_tif_to_png(r"E:\Code\RiceLodging\Segment\datasets\Merge\Lingtangkou\v17\Label_True",
                                r"E:\Code\RiceLodging\Segment\datasets\Merge\Lingtangkou\v17\Label_True_Png")

