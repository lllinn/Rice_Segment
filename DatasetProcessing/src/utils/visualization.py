import matplotlib.pyplot as plt
import numpy as np
import os

def plot_classes_areas(bin_areas: np.ndarray, class_names: list, output_path: str = None):
    """
    绘制每个类别的面积条形图和饼图。
    
    Args:
        bin_areas (np.ndarray): 每个类别的面积数组
        class_names (list): 类别名称列表
        output_path (str, optional): 输出图表路径
    """
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(18, 9))
    axes[0].bar(range(len(bin_areas)), bin_areas)
    axes[0].set_xticks(range(len(bin_areas)))
    axes[0].set_xticklabels([f'{class_names[i]}' for i in range(len(bin_areas))], rotation=45)
    axes[0].set_xlabel('Class')
    axes[0].set_ylabel('Area (sq m.)')
    axes[0].set_title('Bar Chart')

    for i, area in enumerate(bin_areas):
        axes[0].text(i, area, str(round(area, 1)), ha='center', va='bottom')

    axes[1].pie(bin_areas, labels=class_names)
    axes[1].set_title('Pie Chart')

    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True) # 确保父文件夹存在
        fig.savefig(output_path)
    else:
        plt.show()