import numpy as np
import os
from PIL import Image # 需要安装 Pillow 库：pip install Pillow
import argparse
from tqdm import tqdm
import shutil
import random
import sys
import glob # Import glob to find files by pattern


def create_synthetic_data(output_dir, num_files, height, width, features_num, label_configs, filename_prefix="synthetic"):
    """
    创建带有空间模式的虚拟 .npy 数据文件和对应的多种 .png 标签文件。
    标签文件是根据数据文件通过阈值派生出来的。

    Args:
        output_dir (str): 数据文件保存的目录。
        num_files (int): 要生成的文件组数量 (.npy 数据 + 对应的多个 .png 标签)。
        height (int): 生成数据的图像高度。
        width (int): 生成数据的图像宽度。
        features_num (int): .npy 数据文件的特征数量 (例如 RGB 为 3)。
        label_configs (list): 包含标签类型配置的列表。
                               每个配置是一个字典，包含 'name' (标注类型名, e.g., 'severity')
                               和 'num_classes' (该类型标注的类别数量)。
        filename_prefix (str): 文件名的前缀。
    """
    if not label_configs:
        print("错误: 必须提供至少一种标签类型的配置 (label_configs)。", file=sys.stderr)
        sys.exit(1)

    # 确保输出目录存在，如果不存在则创建
    os.makedirs(output_dir, exist_ok=True)
    print(f"输出目录: {os.path.abspath(output_dir)}")
    print(f"标签类型配置: {label_configs}")
    print(f"开始生成 {num_files} 组带有模式的虚拟数据...")

    # 使用 tqdm 显示进度条
    for i in tqdm(range(num_files), desc="Generating files", unit="files"):
        # 使用零填充的索引，方便排序和命名
        file_idx = f"{i:05d}" # 例如：00000, 00001, ...
        base_filename = f"{filename_prefix}_{file_idx}" # 文件基础名

        # --- 生成带有模式的 .npy 数据文件 ---
        # 创建一个基于像素坐标的简单模式，并随文件索引变化
        npy_data = np.zeros((height, width, features_num), dtype=np.float32)
        r_coords, c_coords = np.ogrid[0:height, 0:width] # 获取行和列的坐标网格

        # 创建一个基础空间模式 (例如，对角线梯度)
        # 归一化到 0-1 范围，并结合文件索引和特征索引增加变化
        spatial_pattern = (r_coords / height + c_coords / width) # 基础空间模式 0 到 ~2

        for feature in range(features_num):
             # 每个特征通道的模式：基础空间模式 + 文件索引影响 + 特征索引影响
             # 使用 sin 函数增加一些非线性变化，并缩放到一定范围
             pattern_value = spatial_pattern + (i * 0.05) + (feature * 0.1)
             npy_data[:, :, feature] = np.sin(pattern_value * np.pi * 2) * 50 + 100 + (i % 20) # 添加一些变化和偏移
             # 可以根据需要调整缩放和偏移，确保数据分布适合生成标签


        npy_filename = f"{base_filename}.npy"
        npy_filepath = os.path.join(output_dir, npy_filename)
        np.save(npy_filepath, npy_data)

        # --- 根据数据生成多种 .png 标签文件 ---
        # 派生标签的基础数据 (例如，使用数据的第一个特征或平均值)
        data_for_label_derivation = np.mean(npy_data, axis=-1) # 使用所有特征的平均值作为派生标签的基础


        for label_config in label_configs:
            label_name = label_config['name']
            num_classes = label_config['num_classes']

            # 生成标签数据，范围从 0 到 num_classes - 1
            # 通过对派生数据应用阈值来创建标签
            if num_classes <= 0:
                 print(f"警告: 标签类型 '{label_name}' 的类别数量为 {num_classes}，跳过生成标签。", file=sys.stderr)
                 continue
            elif num_classes == 1:
                 # 如果只有一类，所有像素都是 0
                 label_data = np.zeros((height, width), dtype=np.uint8)
                 mode = 'L'
            else:
                # 创建 num_classes - 1 个阈值，将数据范围划分为 num_classes 个区间
                # 这些阈值将根据当前派生数据的实际值范围来确定，以确保每个类别至少有可能出现
                min_val = np.min(data_for_label_derivation)
                max_val = np.max(data_for_label_derivation)

                # 避免 min == max 导致 linspace 错误
                if min_val == max_val:
                    print(f"警告: 文件 {base_filename} 的派生数据 '{label_name}' 的 min/max 值相同 ({min_val})，无法生成多类别标签。将全部设为类别 0。", file=sys.stderr)
                    label_data = np.zeros((height, width), dtype=np.uint8)
                    mode = 'L'
                else:
                    # 创建 num_classes - 1 个均匀分布的阈值，用于 np.digitize 分类
                    # thresholds 数组的长度是 num_classes - 1
                    # np.digitize(x, bins) 会将 x 中的元素归入 bins 定义的区间。
                    # 结果是 bins 的索引，范围从 0 到 len(bins)。
                    # 例如：bins=[t1, t2, t3], t1<t2<t3
                    # 值 < t1 -> 索引 0
                    # t1 <= 值 < t2 -> 索引 1
                    # t2 <= 值 < t3 -> 索引 2
                    # 值 >= t3 -> 索引 3
                    # 我们需要 num_classes-1 个阈值来得到 0 到 num_classes-1 的索引结果。
                    # np.linspace(start, stop, num) 生成 num 个样本，包含 start 和 stop。
                    # 我们需要 num_classes 个边界来定义 num_classes 个区间，所以 linspace 需要 num_classes+1 个点来定义边界，
                    # 但 np.digitize 使用的是 区间右边界，所以我们只需要 num_classes 个点来定义 num_classes-1 个边界。
                    # 最简单的是生成 num_classes+1 个点，然后取中间 num_classes-1 个作为阈值
                    thresholds = np.linspace(min_val, max_val, num_classes + 1)[1:-1] # 移除 min 和 max

                    # 使用 digitize 将数据值转换为类别索引 (0 到 num_classes - 1)
                    label_data = np.digitize(data_for_label_derivation, thresholds)

                    # 确保数据类型适合保存为 PNG 灰度图
                    # PNG 灰度图 (L) 是 8-bit (0-255)
                    # PNG 灰度图 (I;16) 是 16-bit (0-65535)
                    if num_classes > 256:
                        # 类别数超过 256，需要 16-bit 深度
                        label_data = label_data.astype(np.uint16)
                        mode = 'I;16' # Pillow mode for 16-bit grayscale
                        # 警告用户，因为某些软件对 16-bit PNG 支持可能有限
                        if i == 0: # 只在第一次生成时警告
                             print(f"警告: 标签类型 '{label_name}' 的类别数量 ({num_classes}) > 256。标签将保存为 16-bit 灰度 PNG ('I;16' 模式)。", file=sys.stderr)
                    else:
                        # 类别数在 0-255 范围内，可以使用 8-bit
                        label_data = label_data.astype(np.uint8)
                        mode = 'L' # Pillow mode for 8-bit grayscale

            # 定义标签文件名，包含标签类型名
            # 文件名格式：{prefix}_{index}_{label_name}.png
            png_filename = f"{base_filename}_{label_name}.png"
            png_filepath = os.path.join(output_dir, png_filename)

            # 保存 .png 标签文件
            try:
                # 如果 num_classes > 256 且 mode='I;16'，Pillow 可以处理
                label_img = Image.fromarray(label_data, mode=mode)
                label_img.save(png_filepath)
            except Exception as e:
                 print(f"错误: 保存标签文件 {png_filepath} 时发生错误: {e}", file=sys.stderr)


    print("\n带有模式的虚拟数据生成完成！")


def split_dataset(input_dir, output_dir, train_ratio=0.6, val_ratio=0.2, test_ratio=0.2, seed=None, mode='copy', label_dest_subdir_map=None):
    """
    将 .npy 数据文件和对应的多种 .png 标签文件拆分成训练集、验证集和测试集，
    并在每个集合下创建 data/ 和 label_*/ 子目录。

    Args:
        input_dir (str): 包含原始文件组 (.npy 和带标签类型后缀的 .png) 的目录。
        output_dir (str): 拆分后数据集保存的根目录。
        train_ratio (float): 训练集占总数的比例。
        val_ratio (float): 验证集占总数的比例。
        test_ratio (float): 测试集占总数的比例。
        seed (int, optional): 随机种子。
        mode (str): 文件操作模式 ('copy' 或 'move')。
        label_dest_subdir_map (dict): 字典映射标签类型名 (e.g., 'severity') 到
                                       其在输出目录中的目标子目录名 (e.g., 'label_severity')。
    """
    if label_dest_subdir_map is None or not label_dest_subdir_map:
         print("错误: 必须提供标签类型到目标子目录的映射 (label_dest_subdir_map)。", file=sys.stderr)
         sys.exit(1)

    # 验证比例
    if not (0 <= train_ratio <= 1 and 0 <= val_ratio <= 1 and 0 <= test_ratio <= 1):
        print("错误：训练集、验证集和测试集的比例必须在 0 到 1 之间。", file=sys.stderr)
        sys.exit(1)
    # check sum is approx 1.0
    if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-6:
         print("警告：训练集、验证集和测试集的比例之和不等于 1.0 (当前和: {:.2f})。将根据比例计算数量，余下的分给测试集。".format(train_ratio + val_ratio + test_ratio), file=sys.stderr)


    # 设置随机种子
    if seed is not None:
        random.seed(seed)
        print(f"设置随机种子为: {seed}")

    # --- 扫描输入目录并分组文件 ---
    # 目标是找到所有的 .npy 文件，并为每个 .npy 文件找到所有对应的带标签类型后缀的 .png 文件。
    print(f"正在扫描输入目录: {os.path.abspath(input_dir)}")

    # 找到所有 .npy 文件，它们是文件组的基础
    npy_files = glob.glob(os.path.join(input_dir, "*.npy"))
    if not npy_files:
        print("错误：在指定目录中未找到任何 .npy 文件。", file=sys.stderr)
        sys.exit(1)

    # 构建文件组: base_name -> { 'data': npy_filepath, 'labels': { label_name: png_filepath } }
    file_groups = {}
    expected_label_suffixes = [f"_{name}.png" for name in label_dest_subdir_map.keys()]

    for npy_filepath in npy_files:
        base_name = os.path.splitext(os.path.basename(npy_filepath))[0]
        file_groups[base_name] = {'data': npy_filepath, 'labels': {}}

    # 找到所有 .png 文件并将其归类到对应的 base_name 和 label_name 下
    png_files = glob.glob(os.path.join(input_dir, "*.png"))
    for png_filepath in png_files:
         filename = os.path.basename(png_filepath)
         base_name_match = False
         for base_name in file_groups.keys():
             # Check if filename starts with base_name and has an expected label suffix
             if filename.startswith(base_name) and filename[len(base_name):].startswith('_'): # Ensure underscore separator exists
                 suffix = filename[len(base_name):] # e.g., "_severity.png"
                 # Find which label suffix it matches
                 for label_name, dest_subdir in label_dest_subdir_map.items():
                     if suffix == f"_{label_name}.png":
                         file_groups[base_name]['labels'][label_name] = png_filepath
                         base_name_match = True
                         break # Found the label match
             if base_name_match:
                 break # Found the base_name match for this png

         if not base_name_match:
              print(f"警告: PNG 文件 '{filename}' 不符合预期的命名模式或没有对应的 .npy 文件，已跳过。", file=sys.stderr)


    # 过滤出完整的文件组 (包含 .npy 和所有预期的标签类型)
    complete_file_groups_bases = []
    expected_label_count = len(label_dest_subdir_map)
    for base_name, group_data in file_groups.items():
        if 'data' in group_data and len(group_data['labels']) == expected_label_count:
            complete_file_groups_bases.append(base_name)
        else:
             missing_items = []
             if 'data' not in group_data: missing_items.append('.npy file')
             missing_labels = [name for name in label_dest_subdir_map.keys() if name not in group_data['labels']]
             if missing_labels: missing_items.append(f"missing labels: {missing_labels}")

             if missing_items:
                print(f"警告: 文件组 '{base_name}' 不完整 ({', '.join(missing_items)})，已跳过。", file=sys.stderr)


    if not complete_file_groups_bases:
        print("错误：在指定目录中未找到完整的文件组 (.npy + 所有预期的标签类型)。", file=sys.stderr)
        sys.exit(1)

    print(f"找到 {len(complete_file_groups_bases)} 个完整的文件组。")

    # 随机打乱基础文件名列表
    random.shuffle(complete_file_groups_bases)

    # 计算每个集合的文件组数量
    total_count = len(complete_file_groups_bases)
    train_count = int(total_count * train_ratio)
    val_count = int(total_count * val_ratio)
    # 测试集数量取剩余所有，避免四舍五入问题
    test_count = total_count - train_count - val_count

    print(f"计划拆分数量 (文件组): 训练集={train_count}, 验证集={val_count}, 测试集={test_count}")

    # 拆分基础文件名列表
    train_bases = complete_file_groups_bases[:train_count]
    val_bases = complete_file_groups_bases[train_count : train_count + val_count] # Fix: Use complete_file_groups_bases
    test_bases = complete_file_groups_bases[train_count + val_count:]         # Fix: Use complete_file_groups_bases

    # Define output subdirectories and their corresponding base names list
    split_sets = {
        'train': train_bases,
        'val': val_bases,
        'test': test_bases
    }

    # Create output root directory
    os.makedirs(output_dir, exist_ok=True)
    print(f"输出根目录: {os.path.abspath(output_dir)}")

    # --- 创建每个拆分集合及其内部的 data/ 和 label_*/ 目录 ---
    for split_name in split_sets.keys():
        split_dir = os.path.join(output_dir, split_name)
        split_data_dir = os.path.join(split_dir, 'data')

        os.makedirs(split_data_dir, exist_ok=True) # Create data dir for the split
        print(f"创建目录结构: {split_name}/data/")

        # Create label subdirectories for each label type within the split
        for dest_subdir_name in label_dest_subdir_map.values():
             split_label_dir = os.path.join(split_dir, dest_subdir_name)
             os.makedirs(split_label_dir, exist_ok=True)
             print(f"创建目录结构: {split_name}/{dest_subdir_name}/")


    # --- 复制或移动文件 ---
    file_operation = shutil.copy2 if mode == 'copy' else shutil.move
    print(f"\n文件操作模式: {'复制' if mode == 'copy' else '移动'}")

    for split_name, base_names_list in split_sets.items():
        split_data_dir = os.path.join(output_dir, split_name, 'data')

        print(f"\n正在处理 {split_name} 集 ({len(base_names_list)} 个文件组)...")
        for i, base_name in enumerate(tqdm(base_names_list, desc=f"Processing {split_name}", unit="group", ncols=100)): # Use tqdm for inner loop
            group_data = file_groups[base_name] # Get the file paths for this group

            # Process .npy data file
            src_npy_path = group_data['data']
            # NPY 文件名保持不变
            dest_npy_filename = os.path.basename(src_npy_path)
            dest_npy_path = os.path.join(split_data_dir, dest_npy_filename)
            try:
                 file_operation(src_npy_path, dest_npy_path)
            except Exception as e:
                 print(f"\n警告: 处理数据文件 {src_npy_path} 时发生错误: {e}", file=sys.stderr)

            # Process .png label files
            for label_name, src_png_path in group_data['labels'].items():
                 # Get the destination subdirectory name for this label type
                 dest_subdir_name = label_dest_subdir_map[label_name] # This should always exist due to earlier validation
                 split_label_dir = os.path.join(output_dir, split_name, dest_subdir_name)

                 # --- 修改这里：构建目标标签文件名，移除 _标注类型 后缀 ---
                 original_png_filename = os.path.basename(src_png_path) # e.g., synthetic_00001_severity.png

                 # 假设原始文件名格式是 base_name_label_name.png
                 # 我们需要找到最后一个 "_" 之前的部分作为新的 base_name
                 # 找到 _label_name.png 这个后缀在原始文件名中的起始位置
                 suffix_to_find = f"_{label_name}.png"
                 if original_png_filename.endswith(suffix_to_find):
                     # 截取掉后缀，得到原始的基础文件名
                     dest_png_base_name_without_ext = original_png_filename[:-len(suffix_to_find)] # e.g., "synthetic_00001"
                     dest_png_filename = f"{dest_png_base_name_without_ext}.png" # 新文件名 e.g., "synthetic_00001.png"
                 else:
                     # 如果文件名不符合预期的后缀格式（理论上不会发生），打印警告并使用原始文件名
                     print(f"\nWarning: PNG filename '{original_png_filename}' does not end with expected suffix '{suffix_to_find}'. Saving with original name.", file=sys.stderr)
                     dest_png_filename = original_png_filename

                 # --- 目标路径使用新的文件名 ---
                 dest_png_path = os.path.join(split_label_dir, dest_png_filename)

                 try:
                     file_operation(src_png_path, dest_png_path)
                 except Exception as e:
                     print(f"\n警告: 处理标签文件 {src_png_path} 时发生错误: {e}", file=sys.stderr)


            # Progress within tqdm is automatic now


    print("\n数据集拆分完成！")
    print(f"结果保存在: {os.path.abspath(output_dir)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="创建并拆分虚拟的 .npy 数据文件和多种 .png 标签文件。")

    # --- Arguments for create_synthetic_data ---
    parser.add_argument("--generate_dir", type=str, default="./synthetic_temp",
                        help="临时目录：用于保存生成但尚未拆分的原始文件 (默认为 ./synthetic_temp)")
    parser.add_argument("--num_files", type=int, required=True,
                        help="要生成的文件组数量 (.npy + 对应的多种 .png)")
    parser.add_argument("--height", type=int, default=640,
                        help="生成数据的图像高度 (默认为 640)")
    parser.add_argument("--width", type=int, default=640,
                        help="生成数据的图像宽度 (默认为 640)")
    parser.add_argument("--num_features", type=int, default=1,
                        help=".npy 数据文件的特征数量 (默认为 1)")
    parser.add_argument("--prefix", type=str, default="synthetic",
                        help="文件名的前缀 (默认为 synthetic)")

    # --- Arguments for split_dataset ---
    parser.add_argument("--output_dir", type=str, default="./synthetic_dataset_chm",
                        help="最终拆分后数据集保存的根目录 (默认为 ./synthetic_dataset_split)")
    parser.add_argument("--train_ratio", type=float, default=0.6,
                        help="训练集占总数的比例 (默认为 0.6)")
    parser.add_argument("--val_ratio", type=float, default=0.2,
                        help="验证集占总数的比例 (默认为 0.2)")
    parser.add_argument("--test_ratio", type=float, default=0.2,
                        help="测试集占总数的比例 (默认为 0.2)")
    parser.add_argument("--seed", type=int, default=42,
                        help="随机种子 (默认为 42)")
    parser.add_argument("--mode", type=str, default='move', choices=['copy', 'move'],
                        help="文件操作模式 ('copy' 或 'move') (默认为 copy)")

    # --- Configuration for Label Types (Hardcoded for simplicity in this example) ---
    # In a real-world scenario, you might load this from a config file.
    label_configurations = [
        {'name': 'severity', 'num_classes': 7}, # Severity label: e.g., 7 classes
        {'name': 'none_severity', 'num_classes': 6}, # None-severity label: e.g., 3 classes
        # Add other label types here if needed
    ]

    # --- Define mapping from label_name to destination subdirectory name ---
    # This maps the 'name' from label_configurations to the directory name in the split output.
    label_destination_subdirectories = {
        'severity': 'label_severity',
        'none_severity': 'label_none_severity',
        # Add mappings for other label types
    }


    args = parser.parse_args()

    # --- Run Data Generation ---
    # Create a temporary directory for initial generation if it's different from final output
    # Using generate_dir for temporary storage
    temp_generate_dir = args.generate_dir # Use generate_dir as the temporary space

    create_synthetic_data(
        output_dir=temp_generate_dir,
        num_files=args.num_files,
        height=args.height,
        width=args.width,
        features_num=args.num_features,
        label_configs=label_configurations, # Pass the list of label configs
        filename_prefix=args.prefix,
    )

    # --- Run Dataset Splitting ---
    # Split from the temporary generation directory to the final output directory
    split_dataset(
        input_dir=temp_generate_dir, # Input is the temporary generation directory
        output_dir=args.output_dir,  # Output is the final split directory
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
        mode=args.mode,
        label_dest_subdir_map=label_destination_subdirectories, # Pass the subdir map
    )

    # Optional: Clean up the temporary generation directory if mode was 'move'
    if args.mode == 'move' and os.path.exists(temp_generate_dir) and args.output_dir != temp_generate_dir:
        print(f"\nCleaning up temporary generation directory: {os.path.abspath(temp_generate_dir)}")
        try:
            shutil.rmtree(temp_generate_dir)
            print("Cleanup complete.")
        except Exception as e:
            print(f"Warning: Failed to remove temporary directory {temp_generate_dir}: {e}", file=sys.stderr)