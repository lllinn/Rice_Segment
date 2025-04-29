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




# --- 如何使用 ---
if __name__ == "__main__":

    # 请修改以下路径和比例为你自己的设置
    input_directory = './Test' # 替换为你的包含 MeijuX 文件夹的根目录
    output_directory = './Test/Dataset_ALL' # 替换为你希望存放切分后数据集的目标目录
    split_json = "./Test/record.json"
    record_stack_pattern_part = "Stack_None_GLCM"
    current_stack_pattern_part = "Stack_ALL"
    input_base_name = "Stack_ALL"
    label_base_name = "Label"
    create_test_dataset("./Test", input_base_name=input_base_name, label_base_name=label_base_name)


    # 切分比例设置
    train_ratio = 0.6
    val_ratio = 0.2
    test_ratio = 0.2

    # Run the function
    apply_split_record_with_transform(
        input_directory,
        output_directory,
        split_json,
        input_base_name=input_base_name,
        label_base_name=label_base_name,
        record_stack_pattern_part=record_stack_pattern_part,
        current_stack_pattern_part=current_stack_pattern_part,
        copy_files=True
    )

    # 运行函数
    # split_and_rename_data_and_record(input_directory, output_directory, split_json, input_base_name=input_base_name,  
    #                                 label_base_name=label_base_name, train_ratio=train_ratio, val_ratio=val_ratio, test_ratio=test_ratio)


