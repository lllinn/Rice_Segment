import sys
sys.path.append(r".")
from src.utils.image_utils import calculate_class_areas
from src.utils.visualization import plot_classes_areas


def main():
    input_path = r"/home/music/wzl/Data/20240912-Rice-M3M-50m-Lingtangkou-RGB/Labels_All/merge_Out_v5.tif"
    name = "Lingtangkou"
    class_names = ["road", "sugarcane", "rice_normal", "rice_lodging"]
    areas = calculate_class_areas(input_path=input_path, class_names=class_names)
    print(f'{name}: {areas}')
    plot_classes_areas(bin_areas=areas, class_names=class_names, output_path=None)

if __name__ == '__main__':
    main()
