import numpy as np
from .file_io import read_tif, write_tif
from .visualization import plot_classes_areas

def calculate_class_areas(input_path: str, class_names: list) -> np.ndarray:
    """
    计算每个类别的面积。不对background进行计算
    
    Args:
        input_path (str): 输入图像数据文件位置
        class_names (list): 类别名称列表  ["road", "sugarcane", "rice_normal", "rice_lodging"]
    
    Returns:
        np.ndarray: 每个类别的面积数组
    """

    im_proj, im_geotrans, im_data, im_width, im_height, im_Band = read_tif(input_path)

    # 一个pixel的面积
    area_per_pixel = abs(im_geotrans[1] * im_geotrans[5])
    bin_max = np.max(im_data)
    assert len(class_names) == bin_max, f"The number of classes({len(class_names)}) does not match the number of bins({bin_max}) ❌"
    
    bin_areas = np.zeros(bin_max)
    for label in range(bin_max):
        bin_areas[label] = np.sum((im_data == (label + 1)).astype("uint8")) * area_per_pixel
    
    return bin_areas



