import os
from osgeo import gdal
import numpy as np
from tqdm import tqdm
from PIL import Image
import cv2
from datetime import datetime
import rasterio

# 提升像素限制至更高值（例如 100 亿像素）
Image.MAX_IMAGE_PIXELS = 100_000_000_000_000_000  # 根据实际需求调整

def read_tif(filename: str):
    """
    读取TIF文件并返回投影、地理变换参数和图像数据。
    
    Args:
        filename (str): TIF文件路径
    
    Returns:
        tuple: 包含投影、地理变换参数、图像数据、宽度、高度和波段数的元组
    """
    dataset = gdal.Open(filename)
    if dataset is None:
        raise FileNotFoundError(f"File not found or cannot be opened: {filename}")
    
    im_width = dataset.RasterXSize
    im_height = dataset.RasterYSize
    im_bands = dataset.RasterCount
    im_geotrans = dataset.GetGeoTransform()
    im_proj = dataset.GetProjection()
    
    im_data = dataset.ReadAsArray(0, 0, im_width, im_height)
    # del dataset  # 释放资源, 会导致有的文件卡住
    return im_proj, im_geotrans, im_data, im_width, im_height, im_bands


# def read_tif(filename: str):
#     """
#     读取TIF文件并返回投影、地理变换参数和图像数据。
    
#     Args:
#         filename (str): TIF文件路径
    
#     Returns:
#         tuple: 包含投影、地理变换参数、图像数据、宽度、高度和波段数的元组
#     """
#     # with rasterio.open(filename) as src:
#     src = rasterio.open(filename)
#     data = src.read()
#     # print(data.shape)
#     return data

# GDAL数据类型与numpy数据类型之间的映射关系
DType2GDAL = {"uint8": gdal.GDT_Byte,
              "uint16": gdal.GDT_UInt16,
              "int16": gdal.GDT_Int16,
              "uint32": gdal.GDT_UInt32,
              "int32": gdal.GDT_Int32,
              "float32": gdal.GDT_Float32,
              "float64": gdal.GDT_Float64,
              "cint16": gdal.GDT_CInt16,
              "cint32": gdal.GDT_CInt32,
              "cfloat32": gdal.GDT_CFloat32,
              "cfloat64": gdal.GDT_CFloat64
              }


def write_tif(im_data: np.ndarray, im_geotrans: tuple, im_proj: str, path: str):
    """
    将图像数据保存为TIF文件。
    
    Args:
        im_data (np.ndarray): 图像数据
        im_geotrans (tuple): 地理变换参数
        im_proj (str): 投影信息
        path (str): 输出文件路径
    """
    if im_data.dtype.name in DType2GDAL:
        datatype = DType2GDAL[im_data.dtype.name]
    else:
        datatype = gdal.GDT_Float32

        
    if len(im_data.shape) == 3: # 彩色图
        im_bands, im_height, im_width = im_data.shape
    else: # 灰度图
        im_bands, im_height, im_width = 1, *im_data.shape
    
    driver = gdal.GetDriverByName("GTiff")
    dataset = driver.Create(path, im_width, im_height, im_bands, datatype)
    
    if dataset is not None:
        dataset.SetGeoTransform(im_geotrans)
        dataset.SetProjection(im_proj)
        for i in range(im_bands):
            dataset.GetRasterBand(i + 1).WriteArray(im_data[i] if im_bands > 1 else im_data)
    
    del dataset  # 释放资源

# def write_tif(im_data: np.ndarray, path: str, nodata_value=None, crs='EPSG:4326', transform=None):
#     """
#     将图像数据保存为TIF文件。
    
#     Args:
#         im_data (np.ndarray): 图像数据
#         im_geotrans (tuple): 地理变换参数
#         im_proj (str): 投影信息
#         path (str): 输出文件路径
#     """
#     # if im_data.dtype.name in DType2GDAL:
#     #     datatype = DType2GDAL[im_data.dtype.name]
#     # else:
#     #     datatype = gdal.GDT_Float32

        
#     if len(im_data.shape) == 3: # 彩色图
#         im_bands, im_height, im_width = im_data.shape
#     else: # 灰度图
#         im_bands, im_height, im_width = 1, *im_data.shape
    
#     # 使用 rasterio 打开文件并写入数据
#     with rasterio.open(
#         path, 'w', driver='GTiff',
#         height=height, width=width,
#         count=im_bands, dtype=im_data.dtype, crs=crs,
#         transform=transform, nodata=nodata_value
#     ) as dst:
#         # 写入每个波段数据
#         for band in range(im_bands):
#             dst.write(im_bands[band], band + 1)  # 写入每个波段，注意波段从1开始

def remove_folder(folder_path):
    """
    删除指定的文件夹及其内容
    
    :param folder_path: 要删除的文件夹路径
    """
    print('removing {}....................................'.format(folder_path))
    for root, dirs, files in os.walk(folder_path, topdown=False):
        print('deleting {}...'.format(root))
        for name in tqdm(files):
            os.remove(os.path.join(root, name))
        os.rmdir(root)


# def rename_files(image_folder, label_folder=None, end_with='.tif'):
#     print('Renaming files.................')
#     new_name = 0
#     tif_files = os.listdir(image_folder)
#     # tif_files.sort(key=lambda x: int(x.split('.')[0].split('_')[0]))

#     for filename in tqdm(tif_files):
#         if filename.endswith(end_with):
#             # print(filename)
#             new_file_basename = f'{new_name}{end_with}'
#             new_file_name = os.path.join(image_folder, new_file_basename)
#             os.rename(os.path.join(image_folder, filename), new_file_name)

#             if label_folder is not None:
#                 old_label_name = os.path.join(label_folder, filename)
#                 new_label_name = os.path.join(label_folder, new_file_basename)
#                 os.rename(old_label_name, new_label_name)

#             new_name += 1

def rename_files(image_folder, label_folder=None, label_end_with='.tif'):
    print('Renaming files.................')
    new_name = 0
    tif_files = os.listdir(image_folder)
    # tif_files.sort(key=lambda x: int(x.split('.')[0].split('_')[0]))

    for filename in tqdm(tif_files):
            image_shuffix = filename.split('.')[1]
            new_file_basename = f'{new_name}.{image_shuffix}'
            new_file_name = os.path.join(image_folder, new_file_basename)

            os.rename(os.path.join(image_folder, filename), new_file_name)

            if label_folder is not None:
                prefix = filename.split('.')[0]
                old_label_basename = f'{prefix}{label_end_with}'
                new_label_basename = f'{new_name}{label_end_with}'
                old_label_path = os.path.join(label_folder, old_label_basename)
                new_label_path = os.path.join(label_folder, new_label_basename)
                os.rename(old_label_path, new_label_path)

            new_name += 1

def read_png(filename: str) -> np.ndarray:
    """
    读取 PNG 图像文件，返回图像数据（NumPy 数组格式）

    Args:
        filename (str): PNG 文件路径

    Returns:
        np.ndarray: 图像数据，形状为 (H, W) 或 (H, W, C)，数据类型为 uint8
        width (int): 图像宽度
        height (int): 图像高度
    """
    with Image.open(filename) as img:
        img_array = np.array(img)
    return img_array, *img.size
    # img_array = cv2.imread(filename, cv2.IMREAD_UNCHANGED)
    # # 处理不同通道数
    # if img_array.ndim == 3:
    #     # 转换 BGR -> RGB（OpenCV 默认读取为 BGR）
    #     img_array = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)
    
    # return img_array, *img_array.size

def write_png(img_array: np.ndarray, filename: str):
    """
    将 NumPy 数组格式的图像数据写入 PNG 图像文件

    Args:
        img_array (np.ndarray): 图像数据，形状为 (H, W) 或 (H, W, C)，数据类型为 uint8
        filename (str): PNG 文件路径
    """
    # with Image.fromarray(img_array) as img:
    # Image.fromarray(img_array).save(filename)
    # 处理通道顺序（如果需要保存为 RGB）
    if img_array.ndim == 3:
        img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
    cv2.imwrite(filename, img_array)

def read_png_with_gdal(filename: str):
    """
    使用GDAL库读取PNG图像文件，返回图像数据（NumPy数组格式）

    Args:
        filename (str): PNG文件路径

    Returns:
        np.ndarray: 图像数据，形状为(H,W)或(H,W,C)，数据类型为uint8
        width (int): 图像宽度
        height (int): 图像高度
    """
    dataset = gdal.Open(filename)
    if dataset is None:
        raise FileNotFoundError(f"File not found or cannot be opened: {filename}")
    
    im_width = dataset.RasterXSize
    im_height = dataset.RasterYSize
    
    im_data = dataset.ReadAsArray(0, 0, im_width, im_height)
    del dataset  # 释放
    return im_data, im_width, im_height

# TODO: 目前还存在问题，需要进一步解决
def write_png_with_gdal(img_array: np.ndarray, filename: str):
    """
    使用GDAL库将NumPy数组格式的图像数据写入PNG图像文件

    Args:
        img_array (np.ndarray): 图像数据，形状为(H,W)或(H,W,C)，数据类型为uint8
        filename (str): PNG文件路径
    """
    print(img_array.shape)
    if len(img_array.shape) == 3:
        img_array = img_array.transpose(2, 0, 1)  # 转换为 (H, W, C)
        img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        print(img_array.shape)
    cv2.imwrite(filename, img_array)


def write_png_from_tif(img_array: np.ndarray, filename: str):
    """
    使用GDAL库将NumPy数组格式的图像数据写入PNG图像文件

    Args:
        img_array (np.ndarray): 图像数据，形状为(H,W)或(H,W,C)，数据类型为uint8
        filename (str): PNG文件路径
    """
    # print(img_array.shape)
    if len(img_array.shape) == 3:
        # img_array = img_array.transpose(2, 0, 1)  # 转换为 (H, W, C)
        # (C, H, W) -> (H, W, C)
        img_array = img_array.transpose(1, 2, 0)
        img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
    # img_array = Image.fromarray(img_array)
    

        # print(img_array.shape)
    cv2.imwrite(filename, img_array)
    # img_array.save(filename)

def print_info(filename):
    filename = r"D:/GLCM/Meiju2/Stack/Split/Vegetation-Index/band8to11/split_band8to11_tosame.tif"
    dataset = gdal.Open(filename)
    if dataset is None:
        raise FileNotFoundError(f"File not found or cannot be opened: {filename}")
    
    im_width = dataset.RasterXSize
    im_height = dataset.RasterYSize
    im_bands = dataset.RasterCount
    im_geotrans = dataset.GetGeoTransform()
    im_proj = dataset.GetProjection()
    
    im_data = dataset.ReadAsArray(0, 0, im_width, im_height)
    # del dataset  # 释放资源, 会导致有的文件卡住
    print(im_data)

    exit()

# 合并数据
def change_tif_data(origin_path, change_path, output_path):
    im_proj, im_geotrans, im_data, im_width, im_height, im_bands = read_tif(origin_path)
    _, _, change_data, _, _, _ = read_tif(change_path)
    im_data[2, :,:] = change_data
    write_tif(im_data, im_geotrans, im_proj, output_path)

def change_npy_data(origin_folder, change_folder, output_folder):
    file_names = os.listdir(origin_folder)
    for file_name in tqdm(file_names):
        origin_path = os.path.join(origin_folder, file_name)
        change_path = os.path.join(change_folder, file_name)
        output_path = os.path.join(output_folder, file_name)
        origin_data = np.load(origin_path, mmap_mode='c')
        change_data = np.load(change_path, mmap_mode='c')
        origin_data[:, :, 9] = change_data[:,:,0]
        np.save(output_path, origin_data)


# 提取堆叠的数据到新的文件上
def extract_stacked_data(input_path: str, output_path: str, index: int) -> None:
    im_proj, im_geotrans, im_data, im_width, im_height, im_bands = read_tif(input_path)
    output_data = im_data[index, :, :]
    write_tif(output_data, im_geotrans, im_proj, output_path)


def add_channel(input_folder: str, add_folder: str, output_folder:str) -> None:
    os.makedirs(output_folder, exist_ok=True)
    for file in tqdm(os.listdir(input_folder)):
        stack_data = np.load(os.path.join(input_folder, file), mmap_mode='c')
        add_data = np.load(os.path.join(add_folder, file), mmap_mode='c')
        new_data = np.concatenate((stack_data, add_data), axis=2)
        np.save(os.path.join(output_folder, file), new_data)




if __name__ == '__main__':
    input_folder = 'G:/Rice2024/Meiju1/Datasets/Stack_Norm_All'
    add_folder = 'D:/Rice2024/Meiju1/Datasets/Samples/8_CHM-1'
    output_folder = 'G:/Rice2024/Meiju1/Datasets/Stack_Norm_All_v2'
    add_channel(input_folder, add_folder, output_folder)


# if __name__ == '__main__':


#     origin_path = "F:/Rice2024/Meiju1/Split_Stretch/2_CIs-10/B8-B11/Split_Stretch_B8-B11.tif"
#     change_path = "F:/Rice2024/Meiju1/Split_Stretch/2_CIs-10/B8-B11/Split_Stretch_B10.tif"
#     output_path = "F:/Rice2024/Meiju1/Split_Stretch/2_CIs-10/B8-B11/Split_Stretch_B8-B11_v2.tif"
#     change_tif_data(origin_path, change_path, output_path)

    # filename = r"D:/GLCM/Lingtangkou/Stack/Split/Multi-spectral/split_mulit_spectral_tosame.tif"
    # dataset = gdal.Open(filename)
    # if dataset is None:
    #     raise FileNotFoundError(f"File not found or cannot be opened: {filename}")
    
    # im_width = dataset.RasterXSize
    # im_height = dataset.RasterYSize
    # im_bands = dataset.RasterCount
    # im_geotrans = dataset.GetGeoTransform()
    # im_proj = dataset.GetProjection()
    
    # im_data = dataset.ReadAsArray(0, 0, im_width, im_height)
    # # del dataset  # 释放资源, 会导致有的文件卡住
    # print(im_data)
    # print(im_data[-1,-1])
    # print(im_data.max())
    # print(im_data.min())
    # print('ok')
    # 获取众数
    # print(im_data.argmax())
    # start_time = datetime.now()
    # img, width, height = read_png_with_gdal(r"F:\Data\UAV\Pix4DfIelds_Export\20240913-Rice-M3M-50m-Meiju-1-RGB\Orthomosaic.data.png")
    # print(img.shape)
    # print(width, height)
    # write_png_with_gdal(img, r"F:\Data\UAV\Pix4DfIelds_Export\20240913-Rice-M3M-50m-Meiju-1-RGB\Orthomosaic_test.png")
    # img2, width, height = read_png_with_gdal(r"F:\Data\UAV\Pix4DfIelds_Export\20240913-Rice-M3M-50m-Meiju-1-RGB\Orthomosaic_test.png")
    # # 判断两者是否一样
    # print(np.allclose(img, img2))
    # print('run time is {}'.format(datetime.now()-start_time))
    # print(img.shape)
    # write_png(img, r'F:\Data\UAV\Pix4DfIelds_Export\20240913-Rice-M3M-50m-Meiju-1-RGB\Labels\merge_Out_v22_test.png')
    # img_2 = read_png("F:/Data/UAV/Pix4DfIelds_Export/20240913-Rice-M3M-50m-Meiju-1-RGB/Labels/merge_Out_v22_test.png")
    # 对比img和img2
    # print(np.allclose(img, img_2))

