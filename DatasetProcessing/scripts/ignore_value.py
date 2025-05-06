import numpy as np
import os
from tqdm import tqdm

base_folers = ["/root/datasets/ALL"]

datasets_folders = ["train/data", "val/data", "test/data"]

ignore_value = -3.4028235e+38
new_value = 1e-34


for folder in base_folers:
    for dataset_folder in datasets_folders:
        folder_path = os.path.join(folder, dataset_folder)
        files = os.listdir(folder_path)
        print(folder_path)
        for file in tqdm(files, desc="Processing", unit="files", ncols=100):
            file_path = os.path.join(folder_path, file)
            data = np.load(file_path)
            ignore_mask = data == ignore_value
            data[ignore_mask] = new_value
            np.save(file_path, data)

base_folers = ["/root/data_temp/CHM"]

datasets_folders = ["train/data", "val/data", "test/data"]

for folder in base_folers:
    for dataset_folder in datasets_folders:
        folder_path = os.path.join(folder, dataset_folder)
        files = os.listdir(folder_path)
        print(folder_path)
        for file in tqdm(files, desc="Processing", unit="files", ncols=100):
            file_path = os.path.join(folder_path, file)
            data = np.load(file_path)
            ignore_mask = data == ignore_value
            data[ignore_mask] = new_value
            np.save(file_path, data)


