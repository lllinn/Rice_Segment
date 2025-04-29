from ..utils.file_io import write_tif, read_tif
from tqdm import tqdm
import os
import numpy as np
from ..utils.email_utils import send_email

def imagexy2geo(im_geotrans, row, col):
    """
    相片坐标计算坐标位置
    :param im_geotrans:图像放射变换参数
    :param row: 行数
    :param col: 列数
    :return: 坐标位置
    """
    px = im_geotrans[0] + col * im_geotrans[1] + row * im_geotrans[2]
    py = im_geotrans[3] + col * im_geotrans[4] + row * im_geotrans[5]
    return [px, py]


def setGeotrans(im_geotrans, row, col):
    """
    根据影像大小重算仿射变换参数
    :param im_geotrans:
    :param row:
    :param col:
    :return: 仿射变换参数
    """
    coords00 = imagexy2geo(im_geotrans, 0, 0)
    coords01 = imagexy2geo(im_geotrans, row, 0)
    coords10 = imagexy2geo(im_geotrans, 0, col)

    trans = [0 for i in range(6)]
    trans[0] = coords00[0]
    trans[3] = coords00[1]
    trans[2] = (coords01[0] - trans[0]) / row
    trans[5] = (coords01[1] - trans[3]) / row
    trans[1] = (coords10[0] - trans[0]) / col
    trans[4] = (coords10[1] - trans[3]) / col
    return trans


def merge_tif(input_folder: str, output_path: str):
    """
    将多个TIF格式的影像合并为一个大TIF影像

    Args:
        input_folder (str): 输入文件夹路径，其中包含要合并的TIF影像
        output_path (str): 输出文件路径，即合并后的大TIF影像保存的位置

    Returns:
        None
    """
    # 生成文件路径列表
    filepaths = [os.path.join(input_folder, file) for file in tqdm(os.listdir(input_folder), desc='Reading files...')]

    # 读取第一个影像，获取其地理信息和图像数据
    im_proj, im_geotrans, im_data, im_width, im_height, im_bands = read_tif(filepaths[0])
    
    if len(im_data.shape) == 2:
        # 为Label文件
        axis_bias = 1
        print("Label File")
    else: # 为三维数组
        axis_bias = 0
        print('Image File')

    # 创建img字典列表, 每个字典包含信息: filepath, x, y, sort_id
    img_dict = []
    # 生成字典列表
    for path in tqdm(filepaths, desc='Sorting files...'):
        item = {"filepath": path}
        for output in map(read_tif, [path]):
            item["x"], item["y"] = output[1][0], output[1][3]
        img_dict.append(item)
    # 先根据y, 再根据x进行排序
    img_dict = sorted(img_dict, key=lambda k: (k['y'], k['x']))

    # 获取正确的起始点
    start_path = max(img_dict, key=lambda k: k['y'])['filepath']
    print('start_path:', start_path)
    im_proj, im_geotrans, im_data, _,  _, _ = read_tif(start_path)

    # 对字典中含有的一样的'y'赋予一样的id
    first_y = img_dict[0]['y']
    sort_id = 0
    for i, item in tqdm(enumerate(img_dict), desc="Assign id..."):
        if item['y'] != first_y:
            sort_id += 1
            first_y = item['y']
        item['sort_id'] = sort_id

    # 合成大数据
    img_data_id = []
    # 创建一个空的numpy数组，用于存储合并后的影像数据
    now_concat_data = read_tif(img_dict[0]['filepath'])[2]
    sort_id_now = 100

    # 横向拼接 x
    for item in tqdm(img_dict, desc="Concat images X..."):
        if item['sort_id'] != sort_id_now:
            sort_id_now = item['sort_id']
            now_concat_data = read_tif(item['filepath'])[2]
            img_data_id.append(now_concat_data)
        else:
            now_concat_data = np.concatenate((now_concat_data, read_tif(item['filepath'])[2]), axis=2-axis_bias)
            img_data_id[-1] = now_concat_data

    # 纵向拼接 y, 将y倒过来, 因为y是-增长
    img_Data = np.concatenate(img_data_id[::-1], axis=1-axis_bias)
    print(img_Data.shape, img_Data.dtype)
    # 获取仿射变换参数
    geotrans = setGeotrans(im_geotrans, img_Data.shape[0], img_Data.shape[1])
    # 写入图像中
    write_tif(img_Data, geotrans, im_proj, output_path)

    print('done....')
    print('output:', output_path)

    send_email('数据集合成', "数据集合成完成..")
