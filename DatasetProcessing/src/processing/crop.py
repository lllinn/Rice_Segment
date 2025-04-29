import os
from tqdm import tqdm
from ..utils.file_io import read_tif, write_tif, read_png, write_png
from osgeo import gdal
from .background_black import smart_image_converter, record_image_skip
from ..utils.file_io import remove_folder
from ..utils.email_utils import send_email

def crop_with_repetition(tif_path: str, save_path: str, crop_size: int, repetition_rate: float):
    """
    带重复率的滑动窗口裁剪。
    
    Args:
        tif_path (str): 输入TIF文件路径
        save_path (str): 输出目录
        crop_size (int): 裁剪尺寸
        repetition_rate (float): 重复率
    """
    gdal.UseExceptions()  # 启用 GDAL 异常处理

    os.makedirs(save_path, exist_ok=True)
    # proj, geotrans, img, width, height, _ = read_tif(tif_path)
    dataset = gdal.Open(tif_path)
    width = dataset.RasterXSize
    height = dataset.RasterYSize
    bands = dataset.RasterCount
    geotrans = dataset.GetGeoTransform()
    proj = dataset.GetProjection()
    img = dataset.ReadAsArray(0, 0, width, height)

    
    new_name = len(os.listdir(save_path)) + 1
    for i in tqdm(range(int((height - crop_size * repetition_rate) / (crop_size * (1 - repetition_rate)))), desc="Rows"):
        for j in tqdm(range(int((width - crop_size * repetition_rate) / (crop_size * (1 - repetition_rate)))), desc="Cols", leave=False):
            if len(img.shape) == 2: # 图像是单波段
                cropped = img[
                    int(i * crop_size * (1 - repetition_rate)): int(i * crop_size * (1 - repetition_rate)) + crop_size,
                    int(j * crop_size * (1 - repetition_rate)): int(j * crop_size * (1 - repetition_rate)) + crop_size
                ]
            else: # 图像时多波段
                cropped = img[:,
                    int(i * crop_size * (1 - repetition_rate)): int(i * crop_size * (1 - repetition_rate)) + crop_size,
                    int(j * crop_size * (1 - repetition_rate)): int(j * crop_size * (1 - repetition_rate)) + crop_size
                ]
            
            local_geotrans = list(geotrans)
            local_geotrans[0] += int(j * crop_size * (1 - repetition_rate)) * geotrans[1]
            local_geotrans[3] += int(i * crop_size * (1 - repetition_rate)) * geotrans[5]
            
            write_tif(cropped, tuple(local_geotrans), proj, os.path.join(save_path, f"{new_name}_rice.tif"))
            new_name += 1

    # 向前裁剪最后一列
    for i in range(int((height - crop_size * repetition_rate) / (crop_size * (1 - repetition_rate)))):
        if len(img.shape) == 2:
            cropped = img[
                int(i * crop_size * (1 - repetition_rate)): int(i * crop_size * (1 - repetition_rate)) + crop_size,
                width - crop_size : width
            ]
        else:
            cropped = img[:,
                int(i * crop_size * (1 - repetition_rate)): int(i * crop_size * (1 - repetition_rate)) + crop_size,
                width - crop_size : width
            ]
        
        local_geotrans = list(geotrans)
        local_geotrans[0] += (width - crop_size) * geotrans[1]
        local_geotrans[3] += int(i * crop_size * (1 - repetition_rate)) * geotrans[5]
        
        write_tif(cropped, tuple(local_geotrans), proj, os.path.join(save_path, f"{new_name}_rice.tif"))
        new_name += 1

    # 向前裁剪最后一行
    for j in range(int((width - crop_size * repetition_rate) / (crop_size * (1 - repetition_rate)))):
        if len(img.shape) == 2:
            cropped = img[
                height - crop_size : height,
                int(j * crop_size * (1 - repetition_rate)): int(j * crop_size * (1 - repetition_rate)) + crop_size
            ]
        else:
           
            cropped = img[:,
                height - crop_size : height,
                int(j * crop_size * (1 - repetition_rate)): int(j * crop_size * (1 - repetition_rate)) + crop_size
            ]
        
        local_geotrans = list(geotrans)
        local_geotrans[0] += int(j * crop_size * (1 - repetition_rate)) * geotrans[1]
        local_geotrans[3] += (height - crop_size) * geotrans[5]
        
        write_tif(cropped, tuple(local_geotrans), proj, os.path.join(save_path, f"{new_name}_rice.tif"))
        new_name += 1
    
    # 裁剪右下角
    if len(img.shape) == 2:
        cropped = img[
            height - crop_size : height,
            width - crop_size : width
        ]
    else:
        cropped = img[:,
            height - crop_size : height,
            width - crop_size : width
        ]
    
    local_geotrans = list(geotrans)
    local_geotrans[0] += (width - crop_size) * geotrans[1]
    local_geotrans[3] += (height - crop_size) * geotrans[5]
    
    write_tif(cropped, tuple(local_geotrans), proj, os.path.join(save_path, f"{new_name}_rice.tif"))
    new_name += 1

    # img = None
    # dataset = None

# TODO: 用于现在函数退不出去的bug
def crop_with_repetition_and_convert_delete(tif_path: str, save_path: str, crop_size: int, repetition_rate: float, image_npy_folder, shuffix, skip_log, resolution):
    """
    带重复率的滑动窗口裁剪。
    
    Args:
        tif_path (str): 输入TIF文件路径
        save_path (str): 输出目录
        crop_size (int): 裁剪尺寸
        repetition_rate (float): 重复率
    """
    gdal.UseExceptions()  # 启用 GDAL 异常处理

    os.makedirs(save_path, exist_ok=True)
    # proj, geotrans, img, width, height, _ = read_tif(tif_path)
    dataset = gdal.Open(tif_path)
    width = dataset.RasterXSize
    height = dataset.RasterYSize
    bands = dataset.RasterCount
    geotrans = dataset.GetGeoTransform()
    proj = dataset.GetProjection()
    img = dataset.ReadAsArray(0, 0, width, height)
    if resolution:
        if resolution[0] != width or resolution[1] != height:
            # print("Resolution not match")
            print(f'Resolution must be equal to width and height"+f"this is {resolution}, but the real is {width}*{height}')
            return
        # assert width == resolution[0] and height == resolution[1], "Resolution must be equal to width and height"+f"this is {resolution}, but the real is {width}*{height}"
        
    
    new_name = len(os.listdir(save_path)) + 1
    for i in tqdm(range(int((height - crop_size * repetition_rate) / (crop_size * (1 - repetition_rate)))), desc="Rows"):
        for j in tqdm(range(int((width - crop_size * repetition_rate) / (crop_size * (1 - repetition_rate)))), desc="Cols", leave=False):
            if len(img.shape) == 2: # 图像是单波段
                cropped = img[
                    int(i * crop_size * (1 - repetition_rate)): int(i * crop_size * (1 - repetition_rate)) + crop_size,
                    int(j * crop_size * (1 - repetition_rate)): int(j * crop_size * (1 - repetition_rate)) + crop_size
                ]
            else: # 图像时多波段
                cropped = img[:,
                    int(i * crop_size * (1 - repetition_rate)): int(i * crop_size * (1 - repetition_rate)) + crop_size,
                    int(j * crop_size * (1 - repetition_rate)): int(j * crop_size * (1 - repetition_rate)) + crop_size
                ]
            
            local_geotrans = list(geotrans)
            local_geotrans[0] += int(j * crop_size * (1 - repetition_rate)) * geotrans[1]
            local_geotrans[3] += int(i * crop_size * (1 - repetition_rate)) * geotrans[5]
            
            write_tif(cropped, tuple(local_geotrans), proj, os.path.join(save_path, f"{new_name}_rice.tif"))
            new_name += 1

    # 向前裁剪最后一列
    for i in range(int((height - crop_size * repetition_rate) / (crop_size * (1 - repetition_rate)))):
        if len(img.shape) == 2:
            cropped = img[
                int(i * crop_size * (1 - repetition_rate)): int(i * crop_size * (1 - repetition_rate)) + crop_size,
                width - crop_size : width
            ]
        else:
            cropped = img[:,
                int(i * crop_size * (1 - repetition_rate)): int(i * crop_size * (1 - repetition_rate)) + crop_size,
                width - crop_size : width
            ]
        
        local_geotrans = list(geotrans)
        local_geotrans[0] += (width - crop_size) * geotrans[1]
        local_geotrans[3] += int(i * crop_size * (1 - repetition_rate)) * geotrans[5]
        
        write_tif(cropped, tuple(local_geotrans), proj, os.path.join(save_path, f"{new_name}_rice.tif"))
        new_name += 1

    # 向前裁剪最后一行
    for j in range(int((width - crop_size * repetition_rate) / (crop_size * (1 - repetition_rate)))):
        if len(img.shape) == 2:
            cropped = img[
                height - crop_size : height,
                int(j * crop_size * (1 - repetition_rate)): int(j * crop_size * (1 - repetition_rate)) + crop_size
            ]
        else:
           
            cropped = img[:,
                height - crop_size : height,
                int(j * crop_size * (1 - repetition_rate)): int(j * crop_size * (1 - repetition_rate)) + crop_size
            ]
        
        local_geotrans = list(geotrans)
        local_geotrans[0] += int(j * crop_size * (1 - repetition_rate)) * geotrans[1]
        local_geotrans[3] += (height - crop_size) * geotrans[5]
        
        write_tif(cropped, tuple(local_geotrans), proj, os.path.join(save_path, f"{new_name}_rice.tif"))
        new_name += 1
    
    # 裁剪右下角
    if len(img.shape) == 2:
        cropped = img[
            height - crop_size : height,
            width - crop_size : width
        ]
    else:
        cropped = img[:,
            height - crop_size : height,
            width - crop_size : width
        ]
    
    local_geotrans = list(geotrans)
    local_geotrans[0] += (width - crop_size) * geotrans[1]
    local_geotrans[3] += (height - crop_size) * geotrans[5]
    
    write_tif(cropped, tuple(local_geotrans), proj, os.path.join(save_path, f"{new_name}_rice.tif"))
    new_name += 1

    # 转换为npy文件
    smart_image_converter(save_path, image_npy_folder, shuffix=shuffix, skip_log=skip_log)

    # 删除文件夹
    remove_folder(save_path)

    send_email(f'{save_path} npy数据集制作完成', "数据集制作完成..")  
    # return 
    # exit()

# TODO: 用于现在函数退不出去的bug
def crop_with_repetition_and_save_skip_log(tif_path: str, save_path: str, crop_size: int, repetition_rate: float, skip_log, resolution, threshold=0.9, black_val=0):
    """
    带重复率的滑动窗口裁剪。
    
    Args:
        tif_path (str): 输入TIF文件路径
        save_path (str): 输出目录
        crop_size (int): 裁剪尺寸
        repetition_rate (float): 重复率
    """
    gdal.UseExceptions()  # 启用 GDAL 异常处理

    os.makedirs(save_path, exist_ok=True)
    # proj, geotrans, img, width, height, _ = read_tif(tif_path)
    dataset = gdal.Open(tif_path)
    width = dataset.RasterXSize
    height = dataset.RasterYSize
    bands = dataset.RasterCount
    geotrans = dataset.GetGeoTransform()
    proj = dataset.GetProjection()
    img = dataset.ReadAsArray(0, 0, width, height)

    if resolution:
        assert width == resolution[0] and height == resolution[1], "Resolution must be equal to width and height"+f"this is {resolution}, but the real is {width}*{height}"


    new_name = len(os.listdir(save_path)) + 1
    for i in tqdm(range(int((height - crop_size * repetition_rate) / (crop_size * (1 - repetition_rate)))), desc="Rows"):
        for j in tqdm(range(int((width - crop_size * repetition_rate) / (crop_size * (1 - repetition_rate)))), desc="Cols", leave=False):
            if len(img.shape) == 2: # 图像是单波段
                cropped = img[
                    int(i * crop_size * (1 - repetition_rate)): int(i * crop_size * (1 - repetition_rate)) + crop_size,
                    int(j * crop_size * (1 - repetition_rate)): int(j * crop_size * (1 - repetition_rate)) + crop_size
                ]
            else: # 图像时多波段
                cropped = img[:,
                    int(i * crop_size * (1 - repetition_rate)): int(i * crop_size * (1 - repetition_rate)) + crop_size,
                    int(j * crop_size * (1 - repetition_rate)): int(j * crop_size * (1 - repetition_rate)) + crop_size
                ]
            
            local_geotrans = list(geotrans)
            local_geotrans[0] += int(j * crop_size * (1 - repetition_rate)) * geotrans[1]
            local_geotrans[3] += int(i * crop_size * (1 - repetition_rate)) * geotrans[5]
            
            write_tif(cropped, tuple(local_geotrans), proj, os.path.join(save_path, f"{new_name}_rice.tif"))
            new_name += 1

    # 向前裁剪最后一列
    for i in range(int((height - crop_size * repetition_rate) / (crop_size * (1 - repetition_rate)))):
        if len(img.shape) == 2:
            cropped = img[
                int(i * crop_size * (1 - repetition_rate)): int(i * crop_size * (1 - repetition_rate)) + crop_size,
                width - crop_size : width
            ]
        else:
            cropped = img[:,
                int(i * crop_size * (1 - repetition_rate)): int(i * crop_size * (1 - repetition_rate)) + crop_size,
                width - crop_size : width
            ]
        
        local_geotrans = list(geotrans)
        local_geotrans[0] += (width - crop_size) * geotrans[1]
        local_geotrans[3] += int(i * crop_size * (1 - repetition_rate)) * geotrans[5]
        
        write_tif(cropped, tuple(local_geotrans), proj, os.path.join(save_path, f"{new_name}_rice.tif"))
        new_name += 1

    # 向前裁剪最后一行
    for j in range(int((width - crop_size * repetition_rate) / (crop_size * (1 - repetition_rate)))):
        if len(img.shape) == 2:
            cropped = img[
                height - crop_size : height,
                int(j * crop_size * (1 - repetition_rate)): int(j * crop_size * (1 - repetition_rate)) + crop_size
            ]
        else:
           
            cropped = img[:,
                height - crop_size : height,
                int(j * crop_size * (1 - repetition_rate)): int(j * crop_size * (1 - repetition_rate)) + crop_size
            ]
        
        local_geotrans = list(geotrans)
        local_geotrans[0] += int(j * crop_size * (1 - repetition_rate)) * geotrans[1]
        local_geotrans[3] += (height - crop_size) * geotrans[5]
        
        write_tif(cropped, tuple(local_geotrans), proj, os.path.join(save_path, f"{new_name}_rice.tif"))
        new_name += 1
    
    # 裁剪右下角
    if len(img.shape) == 2:
        cropped = img[
            height - crop_size : height,
            width - crop_size : width
        ]
    else:
        cropped = img[:,
            height - crop_size : height,
            width - crop_size : width
        ]
    
    local_geotrans = list(geotrans)
    local_geotrans[0] += (width - crop_size) * geotrans[1]
    local_geotrans[3] += (height - crop_size) * geotrans[5]
    
    write_tif(cropped, tuple(local_geotrans), proj, os.path.join(save_path, f"{new_name}_rice.tif"))
    new_name += 1

    # 转换为npy文件
    # smart_image_converter(save_path, image_npy_folder, shuffix=shuffix, skip_log=skip_log)
    record_image_skip(save_path, threshold=threshold, skip_log=skip_log, black_val=black_val)

    # 删除文件夹
    remove_folder(save_path)

    send_email(f'{save_path} 智能跳过记录生成完成')  
    # return 
    # exit()





# def crop_with_repetition(tif_path: str, save_path: str, crop_size: int, repetition_rate: float):
#     """
#     带重复率的滑动窗口裁剪。
    
#     Args:
#         tif_path (str): 输入TIF文件路径
#         save_path (str): 输出目录
#         crop_size (int): 裁剪尺寸
#         repetition_rate (float): 重复率
#     """
#     os.makedirs(save_path, exist_ok=True)
#     img = read_tif(tif_path)
#     print('read image successfully')
#     if len(img.shape) == 3: # 彩色图
#         im_bands, height, width = img.shape
#     else: # 灰度图
#         im_bands, height, width = 1, *img.shape

#     new_name = len(os.listdir(save_path)) + 1
#     for i in tqdm(range(int((height - crop_size * repetition_rate) / (crop_size * (1 - repetition_rate)))), desc="Rows"):
#         for j in tqdm(range(int((width - crop_size * repetition_rate) / (crop_size * (1 - repetition_rate)))), desc="Cols", leave=False):
#             if len(img.shape) == 2: # 图像是单波段
#                 cropped = img[
#                     int(i * crop_size * (1 - repetition_rate)): int(i * crop_size * (1 - repetition_rate)) + crop_size,
#                     int(j * crop_size * (1 - repetition_rate)): int(j * crop_size * (1 - repetition_rate)) + crop_size
#                 ]
#             else: # 图像时多波段
#                 cropped = img[:,
#                     int(i * crop_size * (1 - repetition_rate)): int(i * crop_size * (1 - repetition_rate)) + crop_size,
#                     int(j * crop_size * (1 - repetition_rate)): int(j * crop_size * (1 - repetition_rate)) + crop_size
#                 ]
            

            
#             write_tif(cropped, os.path.join(save_path, f"{new_name}_rice.tif"))
#             new_name += 1

#     # 向前裁剪最后一列
#     for i in range(int((height - crop_size * repetition_rate) / (crop_size * (1 - repetition_rate)))):
#         if len(img.shape) == 2:
#             cropped = img[
#                 int(i * crop_size * (1 - repetition_rate)): int(i * crop_size * (1 - repetition_rate)) + crop_size,
#                 width - crop_size : width
#             ]
#         else:
#             cropped = img[:,
#                 int(i * crop_size * (1 - repetition_rate)): int(i * crop_size * (1 - repetition_rate)) + crop_size,
#                 width - crop_size : width
#             ]
        
        
#         write_tif(cropped, os.path.join(save_path, f"{new_name}_rice.tif"))
#         new_name += 1

#     # 向前裁剪最后一行
#     for j in range(int((width - crop_size * repetition_rate) / (crop_size * (1 - repetition_rate)))):
#         if len(img.shape) == 2:
#             cropped = img[
#                 height - crop_size : height,
#                 int(j * crop_size * (1 - repetition_rate)): int(j * crop_size * (1 - repetition_rate)) + crop_size
#             ]
#         else:
           
#             cropped = img[:,
#                 height - crop_size : height,
#                 int(j * crop_size * (1 - repetition_rate)): int(j * crop_size * (1 - repetition_rate)) + crop_size
#             ]
    
        
#         write_tif(cropped, os.path.join(save_path, f"{new_name}_rice.tif"))
#         new_name += 1
    
#     # 裁剪右下角
#     if len(img.shape) == 2:
#         cropped = img[
#             height - crop_size : height,
#             width - crop_size : width
#         ]
#     else:
#         cropped = img[:,
#             height - crop_size : height,
#             width - crop_size : width
#         ]
    
    
#     write_tif(cropped, os.path.join(save_path, f"{new_name}_rice.tif"))
#     new_name += 1


def crop_without_repetition(tif_path: str, save_path: str, crop_size: int):
    """
    无重复率的滑动窗口裁剪。
    
    Args:
        tif_path (str): 输入TIF文件路径
        save_path (str): 输出目录
        crop_size (int): 裁剪尺寸
    """
    # 确保保存路径存在
    os.makedirs(save_path, exist_ok=True)
    
    dataset = gdal.Open(tif_path)
    width = dataset.RasterXSize
    height = dataset.RasterYSize
    bands = dataset.RasterCount
    geotrans = dataset.GetGeoTransform()
    proj = dataset.GetProjection()
    img = dataset.ReadAsArray(0, 0, width, height)

    for x in tqdm(range(0, width - crop_size, crop_size), desc="Rows_Crop"):
        for y in tqdm(range(0, height - crop_size, crop_size), desc="Cols_Crop", leave=False):
            #  如果图像是单波段
            if len(img.shape) == 2:
                cropped = img[
                    y : y + crop_size,
                    x : x + crop_size]
            # 如果图像是多波段
            else:
                cropped = img[:,
                    y : y + crop_size,
                    x : x + crop_size]
            # 更新裁剪后的位置信息
            local_geotrans = list(geotrans) #每一张裁剪图的本地放射变化参数，0，3代表左上角坐标
            local_geotrans[0] += x * geotrans[1]#分别更新为裁剪后的每一张局部图的左上角坐标，为滑动过的像素数量乘以分辨率
            local_geotrans[3] += y * geotrans[5]
            local_geotrans = tuple(local_geotrans)
            # 写图像
            write_tif(cropped, local_geotrans, proj, os.path.join(save_path, f"{x:05d}_{y:05d}_{crop_size}_{crop_size}_rice.tif")) #数组、仿射变化参数、投影、保存路径



def crop_without_repetition_all(tif_path: str, save_path: str, crop_size: int):
    """
    无重复率的滑动窗口裁剪。同时裁剪剩下没有办法整除的区域
    
    Args:
        tif_path (str): 输入TIF文件路径
        save_path (str): 输出目录
        crop_size (int): 裁剪尺寸
    """
    # 确保保存路径存在
    os.makedirs(save_path, exist_ok=True)
    
    dataset = gdal.Open(tif_path)
    width = dataset.RasterXSize
    height = dataset.RasterYSize
    bands = dataset.RasterCount
    geotrans = dataset.GetGeoTransform()
    proj = dataset.GetProjection()
    img = dataset.ReadAsArray(0, 0, width, height)

    for x in tqdm(range(0, width - crop_size, crop_size), desc="Rows_Crop"):
        for y in tqdm(range(0, height - crop_size, crop_size), desc="Cols_Crop", leave=False):
            #  如果图像是单波段
            if len(img.shape) == 2:
                cropped = img[
                    y : y + crop_size,
                    x : x + crop_size]
            # 如果图像是多波段
            else:
                cropped = img[:,
                    y : y + crop_size,
                    x : x + crop_size]
            # 更新裁剪后的位置信息
            local_geotrans = list(geotrans) #每一张裁剪图的本地放射变化参数，0，3代表左上角坐标
            local_geotrans[0] += x * geotrans[1]#分别更新为裁剪后的每一张局部图的左上角坐标，为滑动过的像素数量乘以分辨率
            local_geotrans[3] += y * geotrans[5]
            local_geotrans = tuple(local_geotrans)
            # 写图像
            write_tif(cropped, local_geotrans, proj, os.path.join(save_path, f"{x:05d}_{y:05d}_{crop_size}_{crop_size}_rice.tif")) #数组、仿射变化参数、投影、保存路径

    start_x = width - width % crop_size
    # 裁剪最后一列
    for y in tqdm(range(0, height - crop_size, crop_size), desc="Last Col Crop"):
        if len(img.shape) == 2:
            cropped = img[
                y : y + crop_size,
                start_x : width]
        else:
            cropped = img[:,
                y : y + crop_size,
                start_x : width]
                    # 更新裁剪后的位置信息
        local_geotrans = list(geotrans) #每一张裁剪图的本地放射变化参数，0，3代表左上角坐标
        local_geotrans[0] += start_x * geotrans[1]#分别更新为裁剪后的每一张局部图的左上角坐标，为滑动过的像素数量乘以分辨率
        local_geotrans[3] += y * geotrans[5]
        local_geotrans = tuple(local_geotrans)
        write_tif(cropped, local_geotrans, proj, os.path.join(save_path, f"{start_x:05d}_{y:05d}_{width-start_x}_{crop_size}_rice.tif"))

    start_y = height - height % crop_size
    # 裁剪最后一行
    for x in tqdm(range(0, width - crop_size, crop_size), desc="Last Row Crop"):
        if len(img.shape) == 2:
            cropped = img[
                start_y : height,
                x : x + crop_size]
        else:
            cropped = img[:,
                start_y : height,
                x : x + crop_size]
        local_geotrans = list(geotrans) #每一张裁剪图的本地放射变化参数，0，3代表左上角坐标
        local_geotrans[0] += x * geotrans[1]#分别更新为裁剪后的每一张局部图的左上角坐标，为滑动过的像素数量乘以分辨率
        local_geotrans[3] += start_y * geotrans[5]
        local_geotrans = tuple(local_geotrans)
        write_tif(cropped, local_geotrans, proj, os.path.join(save_path, f"{x:05d}_{start_y:05d}_{crop_size}_{height-start_y}_rice.tif"))

    # 裁剪最后一行一列的块
    if len(img.shape) == 2:
        cropped = img[
            start_y : height,
            start_x : width]
    else:
        cropped = img[:,
            start_y : height,
            start_x : width]
    local_geotrans = list(geotrans) #每一张裁剪图的本地放射变化参数，0，3代表左上角坐标
    local_geotrans[0] += start_x * geotrans[1]#分别更新为裁剪后的每一张局部图的左上角坐标，为滑动过的像素数量乘以分辨率
    local_geotrans[3] += start_y * geotrans[5]
    local_geotrans = tuple(local_geotrans)
    write_tif(cropped, local_geotrans, proj, os.path.join(save_path, f"{start_x:05d}_{start_y:05d}_{width-start_x}_{height-start_y}_rice.tif"))



def crop_with_repetition_png(tif_path: str, save_path: str, crop_size: int, repetition_rate: float):
    """
    带重复率的滑动窗口裁剪。
    
    Args:
        png_path (str): 输入png文件路径
        save_path (str): 输出目录
        crop_size (int): 裁剪尺寸
        repetition_rate (float): 重复率
    """
    os.makedirs(save_path, exist_ok=True)
    # img, width, height = read_png(png_path)
    proj, geotrans, img, width, height, _ = read_tif(tif_path)

    new_name = len(os.listdir(save_path)) + 1
    for i in tqdm(range(int((height - crop_size * repetition_rate) / (crop_size * (1 - repetition_rate)))), desc="Rows"):
        for j in tqdm(range(int((width - crop_size * repetition_rate) / (crop_size * (1 - repetition_rate)))), desc="Cols", leave=False):
            if len(img.shape) == 2: # 图像是单波段
                cropped = img[
                    int(i * crop_size * (1 - repetition_rate)): int(i * crop_size * (1 - repetition_rate)) + crop_size,
                    int(j * crop_size * (1 - repetition_rate)): int(j * crop_size * (1 - repetition_rate)) + crop_size
                ]
            else: # 图像时多波段
                cropped = img[:,
                    int(i * crop_size * (1 - repetition_rate)): int(i * crop_size * (1 - repetition_rate)) + crop_size,
                    int(j * crop_size * (1 - repetition_rate)): int(j * crop_size * (1 - repetition_rate)) + crop_size
                ]
            
            write_png(cropped, os.path.join(save_path, f"{new_name}_rice.png"))
            new_name += 1

    # 向前裁剪最后一列
    for i in range(int((height - crop_size * repetition_rate) / (crop_size * (1 - repetition_rate)))):
        if len(img.shape) == 2:
            cropped = img[
                int(i * crop_size * (1 - repetition_rate)): int(i * crop_size * (1 - repetition_rate)) + crop_size,
                width - crop_size : width
            ]
        else:
            cropped = img[:,
                int(i * crop_size * (1 - repetition_rate)): int(i * crop_size * (1 - repetition_rate)) + crop_size,
                width - crop_size : width
            ]
        
        write_png(cropped, os.path.join(save_path, f"{new_name}_rice.png"))
        new_name += 1

    # 向前裁剪最后一行
    for j in range(int((width - crop_size * repetition_rate) / (crop_size * (1 - repetition_rate)))):
        if len(img.shape) == 2:
            cropped = img[
                height - crop_size : height,
                int(j * crop_size * (1 - repetition_rate)): int(j * crop_size * (1 - repetition_rate)) + crop_size
            ]
        else:
           
            cropped = img[:,
                height - crop_size : height,
                int(j * crop_size * (1 - repetition_rate)): int(j * crop_size * (1 - repetition_rate)) + crop_size
            ]
        
        write_png(cropped, os.path.join(save_path, f"{new_name}_rice.png"))
        new_name += 1
    
    # 裁剪右下角
    if len(img.shape) == 2:
        cropped = img[
            height - crop_size : height,
            width - crop_size : width
        ]
    else:
        cropped = img[:,
            height - crop_size : height,
            width - crop_size : width
        ]
    
    write_png(cropped, os.path.join(save_path, f"{new_name}_rice.png"))
    new_name += 1


# TODO: 有个新想法，就是不保存为tif文件, 直接保存为npy文件，这样可以节省空间，而且可以直接用numpy读取数据，这样就不会有tif文件的读写操作了
# 这是ai生成的想法, 我自己的想法就是保存为png图片, 这样就省了后面的转换步骤

