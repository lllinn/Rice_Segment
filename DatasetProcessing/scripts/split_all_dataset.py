import os
import shutil
import glob
import random
from tqdm import tqdm
import json
import sys # Import sys to exit if record file not found


def split_and_rename_data_and_record(
    input_root_dir,
    output_root_dir,
    split_record_path, # New parameter for record file path
    input_base_name="Stack_ALL",
    label_base_name="Label",
    train_ratio=0.6,
    val_ratio=0.2,
    test_ratio=0.2
):
    """
    遍历指定目录下符合 *<input_base_name>* 模式的文件夹，查找 *.npy 数据文件，
    并找到对应的 *<label_base_name>* 文件夹中的 *.png 标签文件，
    将文件按对收集，切分数据集，记录切分信息，并将文件移动/复制到目标目录。

    Args:
        input_root_dir (str): 包含数据和标签文件夹的根目录。
        output_root_dir (str): 数据集切分后存放的目标根目录。
        split_record_path (str): 用于保存切分记录的 JSON 文件路径。
        input_base_name (str): 数据文件夹名称中用于识别的基准部分 (例如 "Stack_ALL")。
        label_base_name (str): 标签文件夹名称中用于识别的基准部分 (例如 "Label")。
        train_ratio (float): 训练集比例 (0 到 1)。
        val_ratio (float): 验证集比例 (0 到 1)。
        test_ratio (float): 测试集比例 (0 到 1)。
    """
    # Check ratios
    if not (0.99 <= train_ratio + val_ratio + test_ratio <= 1.01):
        print("警告: 训练、验证、测试集的比例之和不等于 1。请检查输入比例。")
        # Pass allows execution to continue with potentially slightly off ratios for test set
        pass

    # Create output directory structure
    print(f"Creating output directory: {output_root_dir}")
    # Using 'val' and 'test' as per user's modified code
    for set_name in ['train', 'val', 'test']:
        os.makedirs(os.path.join(output_root_dir, set_name, 'data'), exist_ok=True)
        os.makedirs(os.path.join(output_root_dir, set_name, 'labels'), exist_ok=True)

    # Collect all stack directories matching the pattern
    stack_dirs = glob.glob(os.path.join(input_root_dir, f'*{input_base_name}*'))

    if not stack_dirs:
        print(f"No directories matching '*{input_base_name}*' found in {input_root_dir}. Please check input path and directory naming.")
        return

    all_file_pairs = [] # Store (data_filepath, label_filepath, original_stack_dir_name, base_filename) tuples

    # Iterate through Stack directories, find corresponding Label directories, and collect file pairs
    print("Collecting data (.npy) and label (.png) file pairs...")
    for stack_dir in stack_dirs:
        dir_name = os.path.basename(stack_dir)

        # Infer corresponding Label directory name
        stack_indicator = input_base_name
        label_indicator = label_base_name
        if stack_indicator in dir_name:
            label_dir_name = dir_name.replace(stack_indicator, label_indicator)
            label_dir = os.path.join(input_root_dir, label_dir_name)

            if os.path.isdir(label_dir):
                 print(f"Processing directory pair: {dir_name} and {label_dir_name}")

                 # Find .npy files in stack directory
                 data_files = glob.glob(os.path.join(stack_dir, '*.npy'))

                 # Find .png files in label directory and map by base filename
                 label_files_map = {}
                 for label_path in glob.glob(os.path.join(label_dir, '*.png')):
                     label_filename = os.path.basename(label_path)
                     base_name, _ = os.path.splitext(label_filename)
                     label_files_map[base_name] = label_path

                 # Match data files (.npy) and label files (.png)
                 for data_file_path in data_files:
                     data_filename = os.path.basename(data_file_path)
                     base_name, data_ext = os.path.splitext(data_filename) # data_ext should be '.npy'

                     # Look for matching base filename in label files map
                     if base_name in label_files_map:
                         matched_label_file_path = label_files_map[base_name]
                         # Store the full path AND the identifying info
                         all_file_pairs.append((data_file_path, matched_label_file_path, dir_name, base_name))
                     else:
                         print(f"Warning: No matching label file (.png) found for data file {data_filename} in {label_dir_name}. Skipping this file.")

            else:
                 print(f"Warning: Corresponding label directory {label_dir_name} not found for {dir_name}. Skipping this directory pair.")
        else:
            # This check is mostly redundant if glob pattern is correct, but good as a fallback
            print(f"Warning: Directory name {dir_name} does not contain '{stack_indicator}'. Skipping this directory.")

    if not all_file_pairs:
        print("No valid data (.npy) / label (.png) file pairs found. Script terminated.")
        return

    print(f"Found a total of {len(all_file_pairs)} file pairs.")

    # Shuffle the list of file pairs
    random.shuffle(all_file_pairs)

    # Calculate split counts
    total_files = len(all_file_pairs)
    train_count = int(total_files * train_ratio)
    val_count = int(total_files * val_ratio)
    test_count = total_files - train_count - val_count # Assign remaining to test to ensure all are used

    print(f"Split counts: Train {train_count}, val {val_count}, Test {test_count}")

    # Split the list
    train_pairs = all_file_pairs[:train_count]
    val_pairs = all_file_pairs[train_count:train_count + val_count]
    test_pairs = all_file_pairs[train_count + val_count:]

    # --- Record the split BEFORE moving files ---
    print(f"Recording split information to {split_record_path}...")
    split_record = {
        'train': [{'stack_dir': pair[2], 'base_name': pair[3]} for pair in train_pairs],
        'val': [{'stack_dir': pair[2], 'base_name': pair[3]} for pair in val_pairs],
        'test': [{'stack_dir': pair[2], 'base_name': pair[3]} for pair in test_pairs]
    }
    try:
        with open(split_record_path, 'w') as f:
            json.dump(split_record, f, indent=4) # Use indent for pretty printing
        print("Split information recorded successfully.")
    except IOError as e:
        print(f"Error writing split record file {split_record_path}: {e}")
        # Decide if you want to continue moving files if recording fails

    # Move and rename files
    print("Moving and renaming files...")
    for set_name, file_pairs in [('train', train_pairs), ('val', val_pairs), ('test', test_pairs)]: # Use 'val' here to match output folder
        print(f"Processing {set_name} set ({len(file_pairs)} pairs)...")
        dest_data_dir = os.path.join(output_root_dir, set_name, 'data')
        dest_labels_dir = os.path.join(output_root_dir, set_name, 'labels')

        for data_path, label_path, original_stack_dir_name, base_name in file_pairs:
            original_data_filename = os.path.basename(data_path)
            original_label_filename = os.path.basename(label_path)

            # Get original extensions
            _, data_ext = os.path.splitext(original_data_filename)
            _, label_ext = os.path.splitext(original_label_filename)

            # Construct new filenames using the original stack directory name as prefix
            new_base_filename = f"{original_stack_dir_name}_{base_name}"
            new_data_filename = f"{new_base_filename}{data_ext}"
            new_label_filename = f"{new_base_filename}{label_ext}"

            new_data_path = os.path.join(dest_data_dir, new_data_filename)
            new_label_path = os.path.join(dest_labels_dir, new_label_filename)

            try:
                # Changed from move to copy2 - move is destructive to the source
                # If you intended to move, use shutil.move, but recording is safer with copy
                shutil.move(data_path, new_data_path) # Using move as in user's last example
                shutil.copy2(label_path, new_label_path) # Using move as in user's last example
            except FileNotFoundError:
                 print(f"Error: Source file not found during move (already moved or missing?) - {data_path} or {label_path}")
            except IOError as e:
                print(f"Error moving file {data_path} or {label_path}: {e}")
            except Exception as e:
                print(f"An unknown error occurred while moving files {data_path} or {label_path}: {e}")

    # Delete original folders - Be CAREFUL with this part!
    print("\nAttempting to remove original folders (only if empty or using rmtree)...")
    for stack_dir in stack_dirs:
        dir_name = os.path.basename(stack_dir)
        stack_indicator = input_base_name
        label_indicator = label_base_name

        if stack_indicator in dir_name:
             label_dir_name = dir_name.replace(stack_indicator, label_indicator)
             label_dir = os.path.join(input_root_dir, label_dir_name)

             # Use rmtree cautiously, it deletes contents! removedirs only deletes empty.
             # If you want to remove non-empty, use shutil.rmtree(path, ignore_errors=True)
             # Be very sure before enabling rmtree.
             try:
                 # os.removedirs will only remove empty directories
                 os.removedirs(stack_dir)
                 print(f"Removed empty directory: {stack_dir}")
             except OSError as e:
                 print(f"Warning: Could not remove directory {stack_dir} (might not be empty or other error): {e}")

            #  if os.path.exists(label_dir): # Check exists before trying to remove
            #      try:
            #         # os.removedirs will only remove empty directories
            #         os.removedirs(label_dir)
            #         print(f"Removed empty directory: {label_dir}")
            #      except OSError as e:
            #         print(f"Warning: Could not remove directory {label_dir} (might not be empty or other error): {e}")


    print("\nFile splitting, renaming/moving, and recording completed!")
    print(f"Split record saved to: {split_record_path}")


def split_and_rename_data_and_record_dual_labels(
    input_root_dir,
    output_root_dir,
    split_record_path,
    input_base_name="Stack_ALL",
    label_none_severity_base_name="label_none_severity", # 第一个标签文件夹的基准名
    label_severity_base_name="label_severity",       # 第二个标签文件夹的基准名
    train_ratio=0.6,
    val_ratio=0.2,
    test_ratio=0.2
):
    """
    遍历指定目录下符合 *<input_base_name>* 模式的数据文件夹，
    查找对应的 *<label_none_severity_base_name>* 和 *<label_severity_base_name>* 标签文件夹，
    收集数据 (.npy) 和标签 (.png) 文件对，切分数据集，记录切分信息，
    并将文件移动/复制到目标目录中对应的子文件夹 (data, label_none_severity, label_severity)。

    Args:
        input_root_dir (str): 包含数据和标签文件夹的根目录。
        output_root_dir (str): 数据集切分后存放的目标根目录。
        split_record_path (str): 用于保存切分记录的 JSON 文件路径。
        input_base_name (str): 数据文件夹名称中用于识别的基准部分 (例如 "Stack_ALL")。
        label_none_severity_base_name (str): 不区分倒伏严重程度的标签文件夹基准名 (例如 "label_none_severity")。
        label_severity_base_name (str): 区分倒伏严重程度的标签文件夹基准名 (例如 "label_severity")。
        train_ratio (float): 训练集比例 (0 到 1)。
        val_ratio (float): 验证集比例 (0 到 1)。
        test_ratio (float): 测试集比例 (0 到 1)。
    """
    # Check ratios
    total_ratio = train_ratio + val_ratio + test_ratio
    if not (0.99 <= total_ratio <= 1.01):
        print(f"警告: 训练、验证、测试集的比例之和 ({total_ratio:.2f}) 不等于 1。请检查输入比例。")
        # 可以选择在这里退出或者继续，这里选择继续
        pass

    # Create output directory structure
    print(f"Creating output directory structure under: {output_root_dir}")
    for set_name in ['train', 'val', 'test']:
        os.makedirs(os.path.join(output_root_dir, set_name, 'data'), exist_ok=True)
        os.makedirs(os.path.join(output_root_dir, set_name, 'label_none_severity'), exist_ok=True)
        os.makedirs(os.path.join(output_root_dir, set_name, 'label_severity'), exist_ok=True)

    # Collect all stack directories matching the pattern
    stack_dirs = glob.glob(os.path.join(input_root_dir, f'*{input_base_name}*'))

    if not stack_dirs:
        print(f"No directories matching '*{input_base_name}*' found in {input_root_dir}. Please check input path and directory naming.")
        return

    # Store file info: { 'data_path': ..., 'label_none_severity_path': ..., 'label_severity_path': ..., 'stack_dir_name': ..., 'base_name': ... }
    all_file_info = []

    # Iterate through Stack directories, find corresponding Label directories, and collect file info
    print("Collecting data (.npy) and dual label (.png) file information...")
    for stack_dir in sorted(stack_dirs): # Use sorted for consistent processing order
        dir_name = os.path.basename(stack_dir)

        # Infer corresponding Label directory names
        label_none_severity_dir_name = dir_name.replace(input_base_name, label_none_severity_base_name)
        label_none_severity_dir = os.path.join(input_root_dir, label_none_severity_dir_name)

        label_severity_dir_name = dir_name.replace(input_base_name, label_severity_base_name)
        label_severity_dir = os.path.join(input_root_dir, label_severity_dir_name)

        print(f"Processing directory pair inference: {dir_name} -> {label_none_severity_dir_name} and {label_severity_dir_name}")

        # Check if at least one label directory exists for this stack dir
        has_none_severity_labels = os.path.isdir(label_none_severity_dir)
        has_severity_labels = os.path.isdir(label_severity_dir)

        if not has_none_severity_labels and not has_severity_labels:
             print(f"Warning: No corresponding label directories ({label_none_severity_dir_name} or {label_severity_dir_name}) found for {dir_name}. Skipping data files in this directory.")
             continue # Skip to the next stack directory

        # Find .npy files in stack directory
        data_files = glob.glob(os.path.join(stack_dir, '*.npy'))

        if not data_files:
            print(f"Warning: No .npy files found in data directory {dir_name}. Skipping this directory.")
            continue # Skip to the next stack directory

        # Collect label files from found label directories and map by base filename
        label_none_severity_files_map = {}
        if has_none_severity_labels:
            for label_path in glob.glob(os.path.join(label_none_severity_dir, '*.png')):
                 label_filename = os.path.basename(label_path)
                 base_name, _ = os.path.splitext(label_filename)
                 label_none_severity_files_map[base_name] = label_path

        label_severity_files_map = {}
        if has_severity_labels:
            for label_path in glob.glob(os.path.join(label_severity_dir, '*.png')):
                 label_filename = os.path.basename(label_path)
                 base_name, _ = os.path.splitext(label_filename)
                 label_severity_files_map[base_name] = label_path


        # Match data files (.npy) and collect information for all found labels
        for data_file_path in sorted(data_files): # Sort data files for consistent processing
            data_filename = os.path.basename(data_file_path)
            base_name, data_ext = os.path.splitext(data_filename) # data_ext should be '.npy'

            file_info = {
                'data_path': data_file_path,
                'label_none_severity_path': label_none_severity_files_map.get(base_name), # Use .get() to handle missing keys (no corresponding label)
                'label_severity_path': label_severity_files_map.get(base_name),         # Use .get() to handle missing keys
                'stack_dir_name': dir_name,        # Original Stack dir name
                'base_name': base_name             # Base filename (without extension)
            }

            # Optionally, you might want to skip data files that have NO corresponding labels at all
            # if file_info['label_none_severity_path'] is None and file_info['label_severity_path'] is None:
            #     print(f"Warning: Data file {data_filename} has no corresponding labels in either directory type. Skipping.")
            #     continue # Skip this data file

            all_file_info.append(file_info)


    if not all_file_info:
        print("No valid data (.npy) files found with corresponding directories. Script terminated.")
        return

    print(f"Found a total of {len(all_file_info)} data files (potential label pairs).")

    # Shuffle the list of file info dictionaries
    random.seed(42) # Optional: use a fixed seed for reproducible splits
    random.shuffle(all_file_info)

    # Calculate split counts
    total_files = len(all_file_info)
    train_count = int(total_files * train_ratio)
    val_count = int(total_files * val_ratio)
    test_count = total_files - train_count - val_count # Assign remaining to test to ensure all are used

    print(f"Split counts: Train {train_count}, Val {val_count}, Test {test_count}")

    # Split the list of file info dictionaries
    train_items = all_file_info[:train_count]
    val_items = all_file_info[train_count:train_count + val_count]
    test_items = all_file_info[train_count + val_count:]

    # --- Record the split BEFORE moving files ---
    print(f"Recording split information to {split_record_path}...")
    split_record = {
        'train': [{'stack_dir': item['stack_dir_name'], 'base_name': item['base_name']} for item in train_items],
        'val': [{'stack_dir': item['stack_dir_name'], 'base_name': item['base_name']} for item in val_items],
        'test': [{'stack_dir': item['stack_dir_name'], 'base_name': item['base_name']} for item in test_items]
    }
    try:
        # Ensure the directory for the record file exists
        os.makedirs(os.path.dirname(split_record_path), exist_ok=True)
        with open(split_record_path, 'w', encoding='utf-8') as f: # Use utf-8 encoding
            json.dump(split_record, f, indent=4, ensure_ascii=False) # Use indent for pretty printing, ensure_ascii=False for potential non-ASCII chars in paths/names
        print("Split information recorded successfully.")
    except IOError as e:
        print(f"Error writing split record file {split_record_path}: {e}")
        # Decide if you want to continue moving files if recording fails
    except Exception as e:
        print(f"An unexpected error occurred while recording split information: {e}")


    # Move and rename files
    print("\nMoving and renaming files...")
    for set_name, file_items in [('train', train_items), ('val', val_items), ('test', test_items)]:
        print(f"Processing {set_name} set ({len(file_items)} items)...")
        dest_data_dir = os.path.join(output_root_dir, set_name, 'data')
        dest_labels_none_sev_dir = os.path.join(output_root_dir, set_name, 'label_none_severity')
        dest_labels_sev_dir = os.path.join(output_root_dir, set_name, 'label_severity')

        for item in file_items:
            data_path = item['data_path']
            label_none_severity_path = item['label_none_severity_path']
            label_severity_path = item['label_severity_path']
            stack_dir_name = item['stack_dir_name']
            base_name = item['base_name']

            # Construct new base filename using the original stack directory name as prefix
            new_base_filename = f"{stack_dir_name}_{base_name}"
            new_data_filename = f"{new_base_filename}.npy" # Assuming .npy extension for data
            new_label_filename = f"{new_base_filename}.png" # Assuming .png extension for labels

            new_data_path = os.path.join(dest_data_dir, new_data_filename)

            try:
                # Move the data file
                shutil.move(data_path, new_data_path)
                # print(f"    Moved data: {os.path.basename(data_path)} -> {new_data_filename}") # Verbose logging

                # Move label_none_severity if it exists
                if label_none_severity_path and os.path.exists(label_none_severity_path):
                     new_label_none_sev_path = os.path.join(dest_labels_none_sev_dir, new_label_filename)
                     shutil.move(label_none_severity_path, new_label_none_sev_path) # Using move as requested implicitly by the task
                     # print(f"    Moved label (none_sev): {os.path.basename(label_none_severity_path)} -> {new_label_filename}") # Verbose logging
                elif label_none_severity_path: # It was recorded but doesn't exist now? (Shouldn't happen if source check is good)
                     print(f"    Warning: Expected label_none_severity file not found during move: {label_none_severity_path}")

                # Move label_severity if it exists
                if label_severity_path and os.path.exists(label_severity_path):
                    new_label_sev_path = os.path.join(dest_labels_sev_dir, new_label_filename)
                    shutil.move(label_severity_path, new_label_sev_path) # Using move
                    # print(f"    Moved label (sev): {os.path.basename(label_severity_path)} -> {new_label_filename}") # Verbose logging
                elif label_severity_path: # It was recorded but doesn't exist now?
                    print(f"    Warning: Expected label_severity file not found during move: {label_severity_path}")


            except FileNotFoundError:
                 print(f"    Error: Source file not found during move (already moved or missing?) for base {base_name} in {stack_dir_name}")
            except shutil.Error as e: # Catch specific shutil errors like same file
                 print(f"    Shutil error moving file(s) for {new_base_filename}: {e}")
            except IOError as e: # Catch other IO errors
                 print(f"    IOError moving file(s) for {new_base_filename}: {e}")
            except Exception as e:
                 print(f"    An unknown error occurred while moving files for {new_base_filename}: {e}")

    # Delete original folders - Be CAREFUL with this part!
    # This part attempts to remove the original directories IF they become empty after moving.
    # If you used shutil.copy2 instead of shutil.move, these directories will NOT be empty.
    # If you INTEND to remove source directories even if not empty, use shutil.rmtree (with caution!)
    print("\nAttempting to remove original folders (only if empty)...")
    # Collect all potential source directories
    all_source_dirs = set(stack_dirs)
    for stack_dir in stack_dirs:
         dir_name = os.path.basename(stack_dir)
         if input_base_name in dir_name:
            none_sev_label_dir_name = dir_name.replace(input_base_name, label_none_severity_base_name)
            none_sev_label_dir = os.path.join(input_root_dir, none_sev_label_dir_name)
            if os.path.exists(none_sev_label_dir):
                all_source_dirs.add(none_sev_label_dir)

            sev_label_dir_name = dir_name.replace(input_base_name, label_severity_base_name)
            sev_label_dir = os.path.join(input_root_dir, sev_label_dir_name)
            if os.path.exists(sev_label_dir):
                 all_source_dirs.add(sev_label_dir)

    for source_dir in sorted(list(all_source_dirs)): # Sort for consistent output
        try:
            # os.removedirs will only remove empty directories and parent chain if they become empty
            # Using os.rmdir here might be slightly safer if you only want to remove the lowest level
            # Let's stick with os.removedirs as in the original, but be aware
            os.removedirs(source_dir)
            print(f"Removed empty directory: {source_dir}")
        except OSError as e:
            # This is expected if the directory is not empty
            # print(f"Info: Could not remove directory {source_dir} (might not be empty): {e}") # Uncomment for less warning output
            pass # Ignore error if directory is not empty


    print("\nFile splitting, renaming/moving, and recording completed!")
    print(f"Split record saved to: {split_record_path}")
    print(f"Processed files are in: {output_root_dir}")

def create_test_dataset(input_folder, input_base_name = "Stack_ALL", label_base_name ="Label",num_images=30):
    
    dirs = [f"Meiju2_{input_base_name}_1", 
            f"Meiju2_{input_base_name}_2", f"Meiju2_{input_base_name}_3", 
            f"Meiju1_{input_base_name}_1", f"Meiju1_{input_base_name}_2", f"Meiju1_{input_base_name}_3", 
            f"Meiju1_{input_base_name}_4", 
            f"Meiju1_{input_base_name}_5", f"Meiju1_{input_base_name}_6",
            f"Lingtangkou_{input_base_name}_1", f"Lingtangkou_{input_base_name}_2", f"Lingtangkou_{input_base_name}_3", ]
    now_area = "Meiju2"
    idx = 0
    for dir_name in dirs:
        area = dir_name.split("_")[0]
        if area != now_area:
            now_area = area
            idx = 0
        
        data_dir = os.path.join(input_folder, dir_name)
        label_dir = os.path.join(input_folder, dir_name.replace(input_base_name, label_base_name))

        os.makedirs(data_dir, exist_ok=True)
        os.makedirs(label_dir, exist_ok=True)
        
        for i in range(num_images):
            data_file = os.path.join(data_dir, f"{idx}_rice.npy")
            label_file = os.path.join(label_dir, f"{idx}_rice.png")

            with open(data_file, 'wb') as f:
                f.write(b'\xFF\xD8\xFF\xE0')

            with open(label_file, 'wb') as f:
                f.write(b'\xFF\xD8\xFF\xE0')
            idx += 1



def create_test_dataset_dual_labels(
    input_folder,
    input_base_name="Stack_ALL",
    label_none_severity_base_name="label_none_severity", # 新增参数
    label_severity_base_name="label_severity",       # 新增参数
    num_images=30
):
    """
    在指定目录下创建符合测试切分脚本结构的 dummy 数据集。
    为每个数据文件夹创建两个对应的标签文件夹 (label_none_severity, label_severity)，
    并在其中生成 dummy .npy 数据文件和 dummy .png 标签文件。

    Args:
        input_folder (str): 将创建 dummy 数据集的根目录。
        input_base_name (str): 数据文件夹名称的基准部分 (例如 "Stack_ALL")。
        label_none_severity_base_name (str): 不区分倒伏严重程度的标签文件夹基准名。
        label_severity_base_name (str): 区分倒伏严重程度的标签文件夹基准名。
        num_images (int): 在每个数据/标签文件夹对中生成的 dummy 文件数量。
    """
    # 硬编码的数据文件夹名称列表，模拟真实数据结构
    # 保持原样，因为它们代表我们要生成的数据文件夹
    dirs = [
        f"Meiju2_{input_base_name}_1",
        f"Meiju2_{input_base_name}_2",
        f"Meiju2_{input_base_name}_3",
        f"Meiju1_{input_base_name}_1",
        f"Meiju1_{input_base_name}_2",
        f"Meiju1_{input_base_name}_3",
        f"Meiju1_{input_base_name}_4",
        f"Meiju1_{input_base_name}_5",
        f"Meiju1_{input_base_name}_6",
        f"Lingtangkou_{input_base_name}_1",
        f"Lingtangkou_{input_base_name}_2",
        f"Lingtangkou_{input_base_name}_3",
    ]

    print(f"Creating dummy dataset structure in: {input_folder}")

    now_area = None # 用于跟踪当前区域以便重置索引
    idx = 0 # 用于生成文件名的索引

    for dir_name in dirs:
        # 提取区域信息，判断何时重置索引
        area = dir_name.split("_")[0]
        if area != now_area:
            now_area = area
            idx = 0 # 在每个新区域开始时重置文件索引

        # 构建数据文件夹路径
        data_dir = os.path.join(input_folder, dir_name)

        # 构建两个标签文件夹路径
        # 使用 replace 来根据数据文件夹名推断标签文件夹名
        label_none_severity_dir_name = dir_name.replace(input_base_name, label_none_severity_base_name)
        label_none_severity_dir = os.path.join(input_folder, label_none_severity_dir_name)

        label_severity_dir_name = dir_name.replace(input_base_name, label_severity_base_name)
        label_severity_dir = os.path.join(input_folder, label_severity_dir_name)


        print(f"  Creating directories: {dir_name}, {label_none_severity_dir_name}, {label_severity_dir_name}")

        # 创建数据文件夹和两个标签文件夹
        os.makedirs(data_dir, exist_ok=True)
        os.makedirs(label_none_severity_dir, exist_ok=True)
        os.makedirs(label_severity_dir, exist_ok=True)

        # 在这些文件夹中生成指定数量的 dummy 文件
        for i in range(num_images):
            # 文件名格式：{idx}_rice.npy / {idx}_rice.png
            data_file_name = f"{idx}_rice.npy"
            label_file_name = f"{idx}_rice.png" # 标签文件的基本文件名相同

            data_file_path = os.path.join(data_dir, data_file_name)
            label_none_severity_file_path = os.path.join(label_none_severity_dir, label_file_name)
            label_severity_file_path = os.path.join(label_severity_dir, label_file_name)


            # 创建 dummy .npy 数据文件
            # 这里的 dummy 数据不重要，只是为了让文件存在
            try:
                with open(data_file_path, 'wb') as f:
                    f.write(b'dummy_npy_data') # 写入一些dummy内容
            except IOError as e:
                print(f"    Error creating dummy data file {data_file_path}: {e}")


            # 创建 dummy .png 标签文件 (在第一个标签文件夹)
            try:
                # PIL 可以用来创建一个真正的 dummy 灰度 PNG，但为了简单，这里只创建空文件
                # 或者写入少量dummy数据
                # Example using PIL (requires Pillow installed):
                # from PIL import Image
                # dummy_img = Image.new('L', (10, 10), color = random.randint(0, 7)) # Create small gray image with random pixel value 0-7
                # dummy_img.save(label_none_severity_file_path)
                with open(label_none_severity_file_path, 'wb') as f:
                     f.write(b'dummy_png_data_none_sev') # 写入一些dummy内容
            except IOError as e:
                 print(f"    Error creating dummy label_none_severity file {label_none_severity_file_path}: {e}")
            # except Exception as e: # Catch potential PIL errors if using that
            #     print(f"    Error creating dummy label_none_severity file with PIL {label_none_severity_file_path}: {e}")


            # 创建 dummy .png 标签文件 (在第二个标签文件夹)
            try:
                 # Example using PIL:
                 # dummy_img = Image.new('L', (10, 10), color = random.randint(0, 7)) # Create small gray image with random pixel value 0-7
                 # dummy_img.save(label_severity_file_path)
                 with open(label_severity_file_path, 'wb') as f:
                      f.write(b'dummy_png_data_sev') # 写入一些dummy内容
            except IOError as e:
                 print(f"    Error creating dummy label_severity file {label_severity_file_path}: {e}")
            # except Exception as e: # Catch potential PIL errors if using that
            #     print(f"    Error creating dummy label_severity file with PIL {label_severity_file_path}: {e}")


            idx += 1 # 增加文件索引

    print("Dummy dataset creation complete.")

def apply_split_record_with_transform(
    input_root_dir,
    output_root_dir,
    split_record_path,
    record_stack_pattern_part="Stack_ALL", # New: part of pattern used in recorded folder names
    current_stack_pattern_part="Stack_ALL", # New: part of pattern used in current folder names
    input_base_name="Stack_ALL", # Existing: used to FIND current stack folders
    label_base_name="Label",     # Existing: used to FIND current label folders
    copy_files=True # Set to False to move files instead of copying
):
    """
    读取切分记录文件，根据记录中的文件夹名应用模式转换，
    在 input_root_dir 中查找对应文件对，并将它们复制或移动到 output_root_dir 中，
    使用记录时的命名规则。

    Args:
        input_root_dir (str): 包含新数据集的根目录（结构应与记录生成时相同，但文件夹模式可能不同）。
        output_root_dir (str): 数据集切分后存放的目标根目录。
        split_record_path (str): 之前保存的切分记录 JSON 文件路径。
        record_stack_pattern_part (str): 记录文件中 Stack 文件夹名中需要被替换的部分 (例如 "Stack_None_GLCM")。
        current_stack_pattern_part (str): 当前数据集 Stack 文件夹名中用于替换的部分 (例如 "Stack_ALL")。
                                         注意: input_base_name 是用来匹配当前文件夹的，
                                         这里的 current_stack_pattern_part 是用于从记录名转换到当前名。
                                         它们通常是一致的，除非你的当前查找模式和转换目标模式不同。
                                         在多数情况下， current_stack_pattern_part 可以等于 input_base_name。
        input_base_name (str): 用于在 input_root_dir 中 glob 匹配 Stack 文件夹的基准部分 (例如 "Stack_ALL")。
        label_base_name (str): 用于在 input_root_dir 中推断 Label 文件夹的基准部分 (例如 "Label")。
        copy_files (bool): 如果为 True 则复制文件，如果为 False 则移动文件 (删除源文件)。
    """

    # Check if transformation is defined
    if record_stack_pattern_part == current_stack_pattern_part:
        print("Info: record_stack_pattern_part is the same as current_stack_pattern_part. No pattern transformation will be applied.")
    else:
         print(f"Applying folder name transformation: replacing '{record_stack_pattern_part}' with '{current_stack_pattern_part}' for lookup.")


    # Load the split record
    print(f"Loading split record from {split_record_path}...")
    if not os.path.exists(split_record_path):
        print(f"Error: Split record file not found at {split_record_path}")
        sys.exit(1)

    try:
        with open(split_record_path, 'r') as f:
            split_record = json.load(f)
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON from {split_record_path}: {e}")
        sys.exit(1)
    except IOError as e:
        print(f"Error reading split record file {split_record_path}: {e}")
        sys.exit(1)

    # Create output directory structure
    print(f"Creating output directory: {output_root_dir}")
    set_names = ['train', 'val', 'test'] # Expected keys in record
    output_set_dirs = {}
    for set_name_key in set_names:
         # Use lowercase for output directory names like 'train', 'val', 'test'
        #  output_folder_name = set_name_key.lower().replace('validation', 'val')
         output_set_dirs[set_name_key] = {
             'data': os.path.join(output_root_dir, set_name_key, 'data'),
             'labels': os.path.join(output_root_dir, set_name_key, 'labels')
         }
         os.makedirs(output_set_dirs[set_name_key]['data'], exist_ok=True)
         os.makedirs(output_set_dirs[set_name_key]['labels'], exist_ok=True)


    # --- Build a lookup map from (current_stack_dir_name, base_filename) to current file paths ---
    # We scan the NEW input_root_dir to find the current location of files
    print(f"Scanning new input directory {input_root_dir} to build file map...")
    # Map: (current_stack_dir_name, base_filename) -> (current_data_path, current_label_path)
    # This map uses the actual folder names found in the CURRENT input directory.
    file_map = {}

    # Use input_base_name to find the current stack directories
    stack_dirs = glob.glob(os.path.join(input_root_dir, f'*{input_base_name}*'))

    if not stack_dirs:
         print(f"No directories matching '*{input_base_name}*' found in {input_root_dir}. Cannot apply split record.")
         sys.exit(1)

    for stack_dir in stack_dirs:
        current_dir_name = os.path.basename(stack_dir) # Get the actual current folder name

        # Infer corresponding Label directory name using the current base names
        stack_indicator_current = input_base_name
        label_indicator_current = label_base_name

        if stack_indicator_current in current_dir_name:
            label_dir_name = current_dir_name.replace(stack_indicator_current, label_indicator_current)
            label_dir = os.path.join(input_root_dir, label_dir_name)

            if os.path.isdir(label_dir):
                 # Find .npy files
                 data_files = glob.glob(os.path.join(stack_dir, '*.npy'))

                 # Find .png files and map by base filename
                 label_files_map = {}
                 for label_path in glob.glob(os.path.join(label_dir, '*.png')):
                     label_filename = os.path.basename(label_path)
                     base_name, _ = os.path.splitext(label_filename)
                     label_files_map[base_name] = label_path

                 # Match data files (.npy) and label files (.png) and add to map
                 for data_file_path in data_files:
                     data_filename = os.path.basename(data_file_path)
                     base_name, _ = os.path.splitext(data_filename) # base_name

                     if base_name in label_files_map:
                         matched_label_file_path = label_files_map[base_name]
                         # Store in map using the CURRENT stack directory name found on disk
                         file_map[(current_dir_name, base_name)] = (data_file_path, matched_label_file_path)
                     # Note: We don't warn here about missing labels during map building; we'll warn later

    print(f"Built map for {len(file_map)} file pairs found in the new input directory.")

    # --- Apply the split record with transformation ---
    print("\nApplying split record...")
    operation = "Copying" if copy_files else "Moving"
    print(f"{operation} files to output directory based on record...")

    total_processed = 0
    total_in_record = 0
    for set_name_key in set_names:
        if set_name_key not in split_record:
            print(f"Warning: Set '{set_name_key}' not found in split record. Skipping.")
            continue

        record_entries = split_record[set_name_key]
        total_in_record += len(record_entries)
        print(f"Processing {len(record_entries)} entries for set: {set_name_key}")

        dest_data_dir = output_set_dirs[set_name_key]['data']
        dest_labels_dir = output_set_dirs[set_name_key]['labels']

        for entry in record_entries:
            # Get the original names from the record
            recorded_stack_dir_name = entry.get('stack_dir')
            base_name = entry.get('base_name')

            if not recorded_stack_dir_name or not base_name:
                 print(f"Warning: Invalid entry in record: {entry}. Skipping.")
                 continue

            # --- Apply the transformation to the recorded stack directory name ---
            expected_current_dir_name = recorded_stack_dir_name
            if record_stack_pattern_part != current_stack_pattern_part:
                 if record_stack_pattern_part in recorded_stack_dir_name:
                     expected_current_dir_name = recorded_stack_dir_name.replace(record_stack_pattern_part, current_stack_pattern_part)
                 else:
                     # This warning means the recorded name doesn't match the expected pattern to transform
                     # It might be an entry from a different type of folder, or an error in parameters
                     print(f"Warning: Recorded stack directory '{recorded_stack_dir_name}' does not contain the expected pattern part '{record_stack_pattern_part}' for transformation. Assuming name is unchanged for lookup.")


            # Look up the current file paths using the transformed identifier
            identifier_for_lookup = (expected_current_dir_name, base_name)

            if identifier_for_lookup in file_map:
                current_data_path, current_label_path = file_map[identifier_for_lookup]

                # Get original extensions from current files (safer than assuming)
                _, data_ext = os.path.splitext(os.path.basename(current_data_path))
                _, label_ext = os.path.splitext(os.path.basename(current_label_path))

                # Construct the new filenames using the *original* stack directory name from the record as prefix
                # This ensures renaming is consistent with the original split
                new_base_filename = f"{recorded_stack_dir_name}_{base_name}"
                new_data_filename = f"{new_base_filename}{data_ext}"
                new_label_filename = f"{new_base_filename}{label_ext}"

                new_data_path = os.path.join(dest_data_dir, new_data_filename)
                new_label_path = os.path.join(dest_labels_dir, new_label_filename)

                try:
                    if copy_files:
                        shutil.move(current_data_path, new_data_path) # Use copy2
                        shutil.copy2(current_label_path, new_label_path)
                    else:
                        shutil.move(current_data_path, new_data_path) # Use move
                        shutil.move(current_label_path, new_label_path)
                    total_processed += 1
                except FileNotFoundError:
                     # Should not happen if file_map was correct, but included for robustness
                     print(f"Error: Source file not found during {operation.lower()} - {current_data_path} or {current_label_path}")
                except IOError as e:
                    print(f"Error {operation.lower()} file {current_data_path} or {current_label_path}: {e}")
                except Exception as e:
                    print(f"An unknown error occurred while {operation.lower()} files {current_data_path} or {current_label_path}: {e}")

            else:
                # This warning means the file pair identifier (after transformation) was not found in the current data scan
                print(f"Warning: File pair identifier from record ('{recorded_stack_dir_name}', '{base_name}') -> (transformed lookup key: '{expected_current_dir_name}', '{base_name}') not found in the current input directory scan ({input_root_dir}). Skipping.")

    print(f"\nFinished applying split record. Successfully processed {total_processed} out of {total_in_record} entries from the record.")
    if total_processed < total_in_record:
        print("Note: Some file pairs listed in the record were not found in the input directory after applying the transformation.")

def apply_split_record_with_transform_dual_labels(
    input_root_dir,
    output_root_dir,
    split_record_path,
    record_stack_pattern_part="Stack_ALL", # 记录文件中 Stack 文件夹名的模式部分
    current_stack_pattern_part="Stack_ALL", # 当前输入目录下 Stack 文件夹名的模式部分，用于查找和新文件命名
                                           # 通常 current_stack_pattern_part == input_base_name
    input_base_name="Stack_ALL",           # 用于在 input_root_dir 中 glob 匹配 Stack 文件夹的基准部分
    label_none_severity_base_name="label_none_severity", # 用于在当前输入目录下查找 不区分严重程度标签文件夹
    label_severity_base_name="label_severity",       # 用于在当前输入目录下查找 区分严重程度标签文件夹
    copy_files=True # Set to False to move files instead of copying
):
    """
    读取切分记录文件（应支持记录双标签文件夹），根据记录中的原始文件夹名和文件名，
    应用模式转换，在 input_root_dir 中查找对应的数据和标签文件对（区分两种标签），
    并将它们复制或移动到 output_root_dir 中对应的子文件夹 (data, labels_none_severity, labels_severity)，
    使用当前文件夹模式作为新文件名前缀。并尝试删除源文件夹（如果为空）。

    Args:
        input_root_dir (str): 包含当前数据集的根目录（结构应与记录生成时相似，但 Stack/Label 文件夹模式可能不同）。
        output_root_dir (str): 数据集切分后存放的目标根目录。
        split_record_path (str): 之前保存的切分记录 JSON 文件路径 (应由支持双标签的 split 函数生成)。
        record_stack_pattern_part (str): 记录文件中 Stack 文件夹名中需要被替换的部分 (例如 "Stack_OldPattern")。
        current_stack_pattern_part (str): 当前数据集 Stack 文件夹名中用于替换 record_stack_pattern_part 的部分 (例如 "Stack_NewPattern")。
                                           用于从记录中的文件夹名推断当前文件夹名进行查找，以及用于构建新文件名前缀。通常等于 input_base_name。
        input_base_name (str): 用于在 input_root_dir 中 glob 匹配当前 Stack 文件夹的基准部分 (例如 "Stack_ALL")。
                               用于扫描当前目录。
        label_none_severity_base_name (str): 用于在 input_root_dir 中推断当前 不区分严重程度标签文件夹 的基准部分。
        label_severity_base_name (str): 用于在 input_root_dir 中推断当前 区分严重程度标签文件夹 的基准部分。
        copy_files (bool): 如果为 True 则复制文件，如果为 False 则移动文件 (删除源文件)。
    """

    # Check if transformation is defined for lookup and new naming
    if record_stack_pattern_part == current_stack_pattern_part:
        print("Info: record_stack_pattern_part is the same as current_stack_pattern_part. No pattern transformation will be applied for folder lookup or new naming.")
    else:
        print(f"Applying folder name transformation for lookup and new naming: replacing '{record_stack_pattern_part}' with '{current_stack_pattern_part}'.")

    # Load the split record
    print(f"\nLoading split record from {split_record_path}...")
    if not os.path.exists(split_record_path):
        print(f"Error: Split record file not found at {split_record_path}")
        sys.exit(1)

    try:
        with open(split_record_path, 'r', encoding='utf-8') as f: # Use utf-8 encoding
            split_record = json.load(f)
        print("Split record loaded successfully.")
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON from {split_record_path}: {e}")
        sys.exit(1)
    except IOError as e:
        print(f"Error reading split record file {split_record_path}: {e}")
        sys.exit(1)
    except Exception as e:
         print(f"An unexpected error occurred while loading split record: {e}")
         sys.exit(1)


    # Create output directory structure (including dual label subfolders)
    print(f"\nCreating output directory structure under: {output_root_dir}")
    set_names = ['train', 'val', 'test'] # Expected keys in record
    output_set_dirs = {}
    for set_name_key in set_names:
        output_set_dirs[set_name_key] = {
            'data': os.path.join(output_root_dir, set_name_key, 'data'),
            'labels_none_severity': os.path.join(output_root_dir, set_name_key, 'labels_none_severity'), # New label subfolder
            'labels_severity': os.path.join(output_root_dir, set_name_key, 'labels_severity')       # New label subfolder
        }
        os.makedirs(output_set_dirs[set_name_key]['data'], exist_ok=True)
        os.makedirs(output_set_dirs[set_name_key]['labels_none_severity'], exist_ok=True)
        os.makedirs(output_set_dirs[set_name_key]['labels_severity'], exist_ok=True)

    # --- Build a lookup map from (current_stack_dir_name, base_filename) to current file paths ---
    # We scan the NEW input_root_dir to find the current location of files.
    print(f"\nScanning new input directory {input_root_dir} to build file map for data and dual labels...")
    # Map: (current_stack_dir_name, base_filename) -> {'data_path': path, 'label_none_severity_path': path/None, 'label_severity_path': path/None}
    file_map = {}
    source_dirs_to_consider_for_deletion = set() # Collect all relevant source directories encountered

    # Use input_base_name parameter to find the current stack directories
    stack_dirs = glob.glob(os.path.join(input_root_dir, f'*{input_base_name}*'))

    if not stack_dirs:
        print(f"Error: No directories matching '*{input_base_name}*' found in {input_root_dir}. Cannot build file map.")
        sys.exit(1)

    for stack_dir in sorted(stack_dirs): # Sort for consistent scanning
        current_dir_name = os.path.basename(stack_dir) # Get the actual current folder name
        source_dirs_to_consider_for_deletion.add(stack_dir) # Add data dir

        # Infer corresponding Label directory names using the CURRENT base names from parameters
        none_sev_label_dir_name_current = current_dir_name.replace(input_base_name, label_none_severity_base_name)
        none_sev_label_dir_current = os.path.join(input_root_dir, none_sev_label_dir_name_current)

        sev_label_dir_name_current = current_dir_name.replace(input_base_name, label_severity_base_name)
        sev_label_dir_current = os.path.join(input_root_dir, sev_label_dir_name_current)

        has_none_severity_labels_current = os.path.isdir(none_sev_label_dir_current)
        has_severity_labels_current = os.path.isdir(sev_label_dir_current)

        if has_none_severity_labels_current:
             source_dirs_to_consider_for_deletion.add(none_sev_label_dir_current) # Add label dir if it exists
        if has_severity_labels_current:
             source_dirs_to_consider_for_deletion.add(sev_label_dir_current) # Add label dir if it exists

        if not has_none_severity_labels_current and not has_severity_labels_current:
             # This warning means this current stack folder has no corresponding label folders of the types we expect.
             # Files from here, if referenced in the record, will be processed only for data or might be skipped if labels are mandatory.
             print(f"  Warning: No corresponding label directories ('{none_sev_label_dir_name_current}' or '{sev_label_dir_name_current}') found in current input for '{current_dir_name}'. Data files from here might lack matching labels.")


        # Find .npy files in current stack directory
        data_files = glob.glob(os.path.join(stack_dir, '*.npy'))

        if not data_files:
             print(f"  Warning: No .npy files found in current data directory '{current_dir_name}'. Skipping file mapping for this directory.")
             continue

        # Find .png files in current label directories and map by base filename
        label_none_severity_files_map_current = {}
        if has_none_severity_labels_current:
            for label_path in glob.glob(os.path.join(none_sev_label_dir_current, '*.png')):
                 label_filename = os.path.basename(label_path)
                 base_name, _ = os.path.splitext(label_filename)
                 label_none_severity_files_map_current[base_name] = label_path

        label_severity_files_map_current = {}
        if has_severity_labels_current:
            for label_path in glob.glob(os.path.join(sev_label_dir_current, '*.png')):
                 label_filename = os.path.basename(label_path)
                 base_name, _ = os.path.splitext(label_filename)
                 label_severity_files_map_current[base_name] = label_path


        # Match data files (.npy) and collect info for both label types
        for data_file_path in sorted(data_files): # Sort for consistent map building
             data_filename = os.path.basename(data_file_path)
             base_name, _ = os.path.splitext(data_filename)

             # Build the info dictionary for this base_name under the current stack directory
             file_info = {
                 'data_path': data_file_path,
                 'label_none_severity_path': label_none_severity_files_map_current.get(base_name), # .get() returns None if key not found
                 'label_severity_path': label_severity_files_map_current.get(base_name),       # .get() returns None if key not found
             }

             # Add to the main file_map using the current stack directory name and base name as the key
             file_map[(current_dir_name, base_name)] = file_info

    print(f"Built map for {len(file_map)} data files found in the current input directory (includes info on associated labels).")

    # --- Apply the split record with transformation and move/copy files ---
    print("\nApplying split record and processing files...")
    operation = "Copying" if copy_files else "Moving"
    print(f"{operation} files to output directory based on record...")

    total_processed_entries = 0
    total_in_record = 0
    for set_name_key in set_names:
        if set_name_key not in split_record:
            print(f"Warning: Set '{set_name_key}' not found in split record. Skipping.")
            continue

        record_entries = split_record[set_name_key]
        total_in_record += len(record_entries)
        print(f"Processing {len(record_entries)} entries for set: {set_name_key}")

        # Determine destination directories for this set
        dest_data_dir = output_set_dirs[set_name_key]['data']
        dest_labels_none_sev_dir = output_set_dirs[set_name_key]['labels_none_severity']
        dest_labels_sev_dir = output_set_dirs[set_name_key]['labels_severity']

        move_or_copy = shutil.copy2 if copy_files else shutil.move

        for entry in record_entries:
            # Get the original names from the record
            recorded_stack_dir_name = entry.get('stack_dir')
            base_name = entry.get('base_name')

            if not recorded_stack_dir_name or not base_name:
                 print(f"  Warning: Invalid entry in record: {entry}. Skipping.")
                 continue

            # --- Apply the transformation to the recorded stack directory name for lookup AND new naming ---
            # This calculates what the original recorded folder name *should be* called
            # in the *current* input directory structure based on the pattern transformation parameters.
            expected_current_dir_name = recorded_stack_dir_name
            if record_stack_pattern_part != current_stack_pattern_part:
                 if record_stack_pattern_part in recorded_stack_dir_name:
                      expected_current_dir_name = recorded_stack_dir_name.replace(record_stack_pattern_part, current_stack_pattern_part)
                 else:
                      # This warning means the recorded name doesn't contain the part expected for transformation.
                      # The record might be from a different source or parameters are wrong.
                      print(f"  Warning: Recorded stack directory '{recorded_stack_dir_name}' does not contain the expected pattern part '{record_stack_pattern_part}' for transformation. Assuming name is unchanged for lookup and new naming.")
                      # If transformation fails, we still use the name as is for lookup and renaming.


            # Look up the current file paths using the transformed identifier in the file_map
            identifier_for_lookup = (expected_current_dir_name, base_name)

            if identifier_for_lookup in file_map:
                 # Get the file info dictionary for this item from the map
                 file_info_current = file_map[identifier_for_lookup]

                 current_data_path = file_info_current['data_path']
                 current_label_none_severity_path = file_info_current['label_none_severity_path']
                 current_label_severity_path = file_info_current['label_severity_path']

                 # Determine extensions (assuming data is .npy, labels are .png)
                 # It's safer to get extensions from the actual files found if they could vary
                 # _, data_ext = os.path.splitext(os.path.basename(current_data_path))
                 # _, label_ext = os.path.splitext(os.path.basename(current_label_none_severity_path or current_label_severity_path or ".png")) # Handle case where both labels might be None
                 data_ext = ".npy" # Assuming .npy for data based on typical use case
                 label_ext = ".png" # Assuming .png for labels based on typical use case


                 # --- Construct the NEW filenames using the EXPECTED CURRENT directory name as prefix ---
                 # This fulfills the user's request to use the current pattern part in naming.
                 new_base_filename = f"{expected_current_dir_name}_{base_name}" # <-- Changed here
                 new_data_filename = f"{new_base_filename}{data_ext}"
                 new_label_filename = f"{new_base_filename}{label_ext}" # The label filenames will have the same base name

                 new_data_path_dest = os.path.join(dest_data_dir, new_data_filename)
                 new_label_none_sev_path_dest = os.path.join(dest_labels_none_sev_dir, new_label_filename)
                 new_label_sev_path_dest = os.path.join(dest_labels_sev_dir, new_label_filename)

                 try:
                     # --- Perform Copy or Move ---

                     # Process Data File
                     move_or_copy(current_data_path, new_data_path_dest)
                     # print(f"    {operation} data: {os.path.basename(current_data_path)}") # Verbose

                     # Process label_none_severity file if its path exists in the map and the source file exists
                     if current_label_none_severity_path and os.path.exists(current_label_none_severity_path):
                          move_or_copy(current_label_none_severity_path, new_label_none_sev_path_dest)
                          # print(f"    {operation} label (none_sev): {os.path.basename(current_label_none_severity_path)}") # Verbose
                     elif current_label_none_severity_path is not None:
                         # Path was in map but doesn't exist on disk now? Unusual case.
                          print(f"  Warning: Expected label_none_severity file not found during {operation.lower()} based on map: {current_label_none_severity_path}. Skipping.")
                     # If current_label_none_severity_path is None, it means it wasn't found during the initial scan, which is fine.

                     # Process label_severity file if its path exists in the map and the source file exists
                     if current_label_severity_path and os.path.exists(current_label_severity_path):
                         move_or_copy(current_label_severity_path, new_label_sev_path_dest)
                         # print(f"    {operation} label (sev): {os.path.basename(current_label_severity_path)}") # Verbose
                     elif current_label_severity_path is not None:
                         # Path was in map but doesn't exist on disk now? Unusual case.
                          print(f"  Warning: Expected label_severity file not found during {operation.lower()} based on map: {current_label_severity_path}. Skipping.")
                     # If current_label_severity_path is None, it means it wasn't found during the initial scan, which is fine.

                     total_processed_entries += 1

                 except FileNotFoundError:
                      # This should ideally not happen if the file_map is correctly built and checked with os.path.exists
                      print(f"  Error: Source file not found during {operation.lower()} for record entry ('{recorded_stack_dir_name}', '{base_name}'). Source paths: Data='{current_data_path}', NoneSevLabel='{current_label_none_severity_path}', SevLabel='{current_label_severity_path}'")
                 except shutil.Error as e:
                      print(f"  Shutil error {operation.lower()} file(s) for new file base name '{new_base_filename}': {e}")
                 except IOError as e:
                      print(f"  IOError {operation.lower()} file(s) for new file base name '{new_base_filename}': {e}")
                 except Exception as e:
                      print(f"  An unknown error occurred while {operation.lower()} files for new file base name '{new_base_filename}': {e}")

            else:
                 # This warning means the file pair identifier from the record (after transformation) was NOT found
                 # in the scan of the *current* input directory.
                 print(f"  Warning: File pair identifier from record ('{recorded_stack_dir_name}', '{base_name}') -> (transformed lookup key: '{expected_current_dir_name}', '{base_name}') not found in the current input directory scan ({input_root_dir}). Skipping this entry from the record.")


    print(f"\nFinished applying split record. Successfully processed {total_processed_entries} out of {total_in_record} entries from the record.")
    if total_processed_entries < total_in_record:
        print("Note: Some file pairs listed in the record were not found in the current input directory after applying the pattern transformation.")

    # --- Attempt to remove original source folders ---
    # This attempts to remove the original directories IF they become empty after moving (copy=False).
    # If you used copy_files=True, these directories will NOT be empty and os.removedirs will fail.
    # os.removedirs is safer than shutil.rmtree as it only removes empty directories.
    print("\nAttempting to remove original source folders (only if empty)...")
    # The set 'source_dirs_to_consider_for_deletion' was collected during the initial scan
    sorted_source_dirs = sorted(list(source_dirs_to_consider_for_deletion), reverse=True) # Sort reverse so child is removed before parent

    for source_dir in sorted_source_dirs:
        try:
            # os.removedirs will only remove empty directories and parent chain if they become empty
            os.removedirs(source_dir)
            # print(f"Removed empty directory: {source_dir}") # Verbose success
        except OSError as e:
            # This is expected if the directory is not empty (e.g., due to copy or processing errors)
            # print(f"Info: Could not remove directory {source_dir} (might not be empty): {e}") # Uncomment for less warning output
            pass # Ignore error if directory is not empty


    print("\nDataset application process complete.")


# --- 如何使用 ---
if __name__ == "__main__":

    # 请修改以下路径和比例为你自己的设置
    input_directory = './Test' # 替换为你的包含 MeijuX 文件夹的根目录
    output_directory = './Test/Dataset_None_GLCM' # 替换为你希望存放切分后数据集的目标目录
    split_json = "./Test/record.json"
    record_stack_pattern_part = "Stack_ALL"
    current_stack_pattern_part = "Stack_None_GLCM"
    input_base_name = "Stack_None_GLCM"
    label_none_sev_base = "label_none_severity"
    label_sev_base = "label_severity"

    # create_test_dataset("./Test", input_base_name=input_base_name, label_base_name=label_base_name)
    # create_test_dataset_dual_labels("./Test", input_base_name=input_base_name, label_none_severity_base_name=label_none_sev_base, 
    #                                 label_severity_base_name=label_sev_base,num_images=100)

    # 切分比例设置
    train_ratio = 0.6
    val_ratio = 0.2
    test_ratio = 0.2

    # Run the function
    # apply_split_record_with_transform(
    #     input_directory,
    #     output_directory,
    #     split_json,
    #     input_base_name=input_base_name,
    #     label_base_name=label_base_name,
    #     record_stack_pattern_part=record_stack_pattern_part,
    #     current_stack_pattern_part=current_stack_pattern_part,
    #     copy_files=True
    # )

    # 运行函数, 首次运行
    # split_and_rename_data_and_record_dual_labels(input_directory, output_directory, split_json, input_base_name=input_base_name,
    #                                              label_none_severity_base_name=label_none_sev_base, label_severity_base_name=label_sev_base, 
    #                                              train_ratio=train_ratio, val_ratio=val_ratio, test_ratio=test_ratio)
    apply_split_record_with_transform_dual_labels(input_directory, output_directory, split_json, record_stack_pattern_part=record_stack_pattern_part,
                                                  current_stack_pattern_part=current_stack_pattern_part, copy_files=False, input_base_name=input_base_name,
                                                  label_none_severity_base_name=label_none_sev_base, label_severity_base_name=label_sev_base)


    # split_and_rename_data_and_record(input_directory, output_directory, split_json, input_base_name=input_base_name,  
    #                                 label_base_name=label_base_name, train_ratio=train_ratio, val_ratio=val_ratio, test_ratio=test_ratio)


