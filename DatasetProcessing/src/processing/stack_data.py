import numpy as np
import os
from tqdm import tqdm


def stack_npy_files(input_folders, output_folder):
    # 创建输出目录
    os.makedirs(output_folder, exist_ok=True)

    print("验证文件是否能用于堆叠..........")
    # 获取并验证所有文件夹的文件列表, 每个元素代表一个文件夹中的所有文件名
    file_lists = []
    for folder in input_folders:
        # 按数字顺序排序文件名
        files = sorted(os.listdir(folder), 
                    key=lambda x: int(x.split('.')[0].split('_')[0]))
        file_lists.append(files)


    # 验证所有文件夹包含相同数量的文件
    file_counts = [len(fl) for fl in file_lists]
    if len(set(file_counts)) != 1:
        raise ValueError("所有文件夹必须包含相同数量的文件")

    # 验证文件名一致性并获取总文件数
    total_files = file_counts[0]
    for i in range(total_files):
        filenames = [fl[i] for fl in file_lists]
        if len(set(filenames)) != 1:
            raise ValueError(f"文件序号 {i+1} 在不同文件夹中名称不一致: {filenames}")

    print('检查合格✔...')

    print("开始堆叠.......")
    # 处理每个文件
    for file_idx in tqdm(range(total_files), desc="Processing Files", unit="files", ncols=100):
        # 获取当前文件的统一文件名
        filename = file_lists[0][file_idx]
        
        # 加载所有对应的npy文件
        arrays = []
        for folder_idx, folder in enumerate(input_folders):
            file_path = os.path.join(folder, filename)
            arr = np.load(file_path, mmap_mode='c')
            arrays.append(arr)
        
        # 沿通道维度堆叠
        stacked_arr = np.concatenate(arrays, axis=2)
        
        # 保存结果
        output_path = os.path.join(output_folder, filename)
        np.save(output_path, stacked_arr)
        # print(f"已处理并保存: {filename}")

    print("所有文件处理完成！")


def split_npy_files(input_dir, channel_mapping, output_dir, features_to_process):
    """
    从指定输入目录下的 .npy 文件中提取 features_to_process 中所有特征涉及的通道，
    并将这些通道合并保存到输出目录的同名文件中。

    Args:
        input_dir (str): 包含所有 .npy 文件的输入目录路径。
        channel_mapping (dict): 字典，键为特征名称，值是包含要提取的通道索引的列表。
        output_dir (str): 保存处理后 .npy 文件的输出目录路径。
        features_to_process (list): 包含要处理的特征名称的列表（这些名称必须是 channel_mapping 中的键）。
    """
    os.makedirs(output_dir, exist_ok=True)

    # 获取所有需要提取的唯一通道索引
    all_channels_to_extract = set()
    for feature_name in features_to_process:
        if feature_name in channel_mapping:
            all_channels_to_extract.update(channel_mapping[feature_name])

    # 将集合转换为排序后的列表，以保持通道顺序
    sorted_channels_to_extract = sorted(list(all_channels_to_extract))

    for filename in tqdm(os.listdir(input_dir), desc="Spliting Files", unit="files"):
        if filename.endswith('.npy'):
            input_file_path = os.path.join(input_dir, filename)
            output_file_path = os.path.join(output_dir, filename)

            try:
                data = np.load(input_file_path)
                if sorted_channels_to_extract:
                    selected_channels = data[:, :, sorted_channels_to_extract]
                    np.save(output_file_path, selected_channels)
                    # print(f"Processed: {input_file_path} -> {output_file_path} (extracted channels: {sorted_channels_to_extract})")
                else:
                    print(f"Warning: No channels to extract for file: {input_file_path} based on provided features.")

            except Exception as e:
                print(f"Error processing {input_file_path}: {e}")

    print("Finished processing.")


def generate_sample_npy(file_path, height=640, width=640, channels=45):
    """生成指定shape的随机.npy文件。"""
    data = np.random.rand(height, width, channels).astype(np.float32)
    np.save(file_path, data)
    print(f"Generated sample data: {file_path}")

if __name__ == '__main__':
    # python -m src.processing.stack_data
    test_input_dir = 'sample_input_npy'
    # os.makedirs(test_input_dir, exist_ok=True)

    # for file in os.listdir(sample_input_dir):
    #     print(np.load(os.path.join(sample_input_dir, file)).shape)

    # print(f"Sample data generated in: {sample_input_dir}") 


    channel_mapping = {
        r'RGB': [0, 1, 2],
        r'Multi-spectral': [3, 4, 5, 6],
        r"Vegetation-Index/band8to11": [7, 8, 9, 10],
        r'Vegetation-Index/band12to15': [11, 12, 13, 14],
        r"Vegetation-Index/band16to18": [15, 16, 17],
        r'Vegetation-Index/band19to20': [18, 19],
        r'R-Texture\band21to24': [20, 21, 22, 23],
        r'R-Texture\band25to28': [24, 25, 26, 27],
        r'G-Texture\band29to32': [28, 29, 30, 31],
        r'G-Texture\band33to36': [32, 33, 34, 35],
        r'B-Texture\band37to40': [36, 37, 38, 39],
        r'B-Texture\band41to44': [40, 41, 42, 43],
        r'DSM': [44],
    }

    features_to_process = [r'RGB', 'Multi-spectral', r"Vegetation-Index/band8to11",r'DSM', r'Vegetation-Index/band12to15', r"Vegetation-Index/band16to18",
                           r'Vegetation-Index/band19to20', ]
    test_output_dir = 'test_output_merged_npy'
     # 运行处理函数
    split_npy_files(test_input_dir, channel_mapping, test_output_dir, features_to_process)

    # split_npy_files(input_root_dir, channel_mapping, output_root_dir, features_to_process)
    
     # --- 测试环节 ---
    print("\n--- Running Tests ---")

    input_files = [f for f in os.listdir(test_input_dir) if f.endswith('.npy')]
    output_files = [f for f in os.listdir(test_output_dir) if f.endswith('.npy')]
    assert len(input_files) == len(output_files), f"Test Failed: Expected {len(input_files)} output files, but got {len(output_files)}"
    print(f"Test Passed: Number of output files matches input files ({len(output_files)}).")

    # 获取期望的通道索引
    expected_channels_indices = set()
    for feature in features_to_process:
        if feature in channel_mapping:
            channels = channel_mapping[feature]
            expected_channels_indices.update(channels)
    expected_channels_indices = sorted(list(expected_channels_indices))

    for filename in input_files:
        input_file_path = os.path.join(test_input_dir, filename)
        output_file_path = os.path.join(test_output_dir, filename)

        original_data = np.load(input_file_path)
        processed_data = np.load(output_file_path)

        expected_channels_data = original_data[:, :, expected_channels_indices]

        assert np.array_equal(processed_data, expected_channels_data), \
            f"Test Failed: Data mismatch for file {filename}"
        print(f"Test Passed: Data in {filename} matches the expected extracted channels.")

    print("\nAll tests passed!")

    # 可以选择保留输出目录以供检查
    # shutil.rmtree(test_output_dir)