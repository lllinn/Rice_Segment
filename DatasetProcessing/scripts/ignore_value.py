import numpy as np
import os
from tqdm import tqdm
import concurrent.futures
import time # 导入 time 模块用于计时

# 定义基础文件夹和数据集子文件夹
base_folers = ["E:/ALL_Datasets/ALL"]
datasets_folders = ["train/data", "val/data", "test/data"]

# 定义要查找和替换的值
ignore_value = -3.4028235e+38 # 要忽略的特定浮点值
new_value = 1e-34 # 替换成的新值

# 定义一个函数，用于处理单个文件
# 这个函数将在单独的线程中运行
def process_single_file(file_path, ignore_value, new_value):
    """Loads an npy file, replaces ignore_value with new_value, and saves."""
    try:
        # 使用 np.load 直接加载数据到内存
        # 注意：如果文件非常大，可能需要大量内存。
        # 原代码使用了 mmap_mode='c'，在多线程下进行修改+保存可能行为复杂。
        # 直接加载到内存进行修改再保存通常更可靠，前提是内存足够。
        data = np.load(file_path)

        # 创建一个布尔掩码，找到需要替换的值
        # 注意：浮点数比较 == 可能存在精度问题，但如果 ignore_value 是精确的 Sentinel 值，== 可以工作。
        # 如果是接近某个值，应使用 np.isclose(data, ignore_value, atol=...)
        ignore_mask = (data == ignore_value)

        # 检查是否存在需要替换的值，避免不必要的保存操作
        if np.any(ignore_mask):
            # 替换值
            data[ignore_mask] = new_value

            # 保存修改后的数据到原文件路径（会覆盖原文件）
            np.save(file_path, data)

            # 返回成功信息或文件名
            return f"Processed: {os.path.basename(file_path)}"
        else:
            # 返回无需处理的信息
            return f"No ignore value in: {os.path.basename(file_path)}"

    except Exception as e:
        # 捕获处理文件时发生的错误，并返回错误信息
        return f"Error processing {os.path.basename(file_path)}: {e}"

# --- 主处理逻辑 ---
# 配置线程池执行器
# max_workers 参数决定同时运行的最大线程数。
# 对于文件I/O密集型任务，线程数可以适当多于 CPU 核心数 (例如，CPU 核心数的 2-4 倍)，
# 因为线程在等待I/O时会释放GIL。可以根据你的系统和硬盘性能调整这个值。
# 你之前测试的机器有 24-48 核 CPU，可以尝试 20-50 个线程。
MAX_WORKERS = 30 # 示例值，你可以根据实际情况调整

total_start_time = time.time() # 开始总计时

for folder in base_folers:
    for dataset_folder in datasets_folders:
        folder_path = os.path.join(folder, dataset_folder)
        print(f"--- 开始处理文件夹: {folder_path} ---")

        # 检查文件夹是否存在
        if not os.path.exists(folder_path):
            print(f"警告: 文件夹不存在，跳过: {folder_path}")
            continue

        try:
            # 获取文件夹内所有以 .npy 结尾的文件列表
            files = [f for f in os.listdir(folder_path) if f.endswith('.npy')]
        except Exception as e:
             print(f"错误: 列出文件失败在 {folder_path}: {e}")
             continue

        if not files:
            print(f"信息: 在 {folder_path} 中没有找到 .npy 文件，跳过。")
            continue

        # 使用 ThreadPoolExecutor 创建线程池
        with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            # 提交任务给线程池，每个文件一个任务
            # future_to_file 字典用于关联 future 对象和对应的文件名，方便在结果返回时知道是哪个文件
            future_to_file = {
                executor.submit(process_single_file, os.path.join(folder_path, file), ignore_value, new_value): file
                for file in files
            }

            # 使用 tqdm 包装 concurrent.futures.as_completed 来显示处理进度
            # as_completed 会在一个迭代器中返回已完成的 future，无论它们完成的顺序如何
            for future in tqdm(concurrent.futures.as_completed(future_to_file),
                               total=len(files), # 总任务数
                               desc=f"Processing {os.path.basename(folder_path)}", # 进度条描述
                               unit="file", # 进度条单位
                               ncols=100): # 进度条宽度

                # 获取任务的结果（或异常）
                # 如果 process_single_file 函数抛出异常，future.result() 会重新抛出它
                try:
                    result = future.result()
                    # 打印每个文件的处理结果（可选，可能输出很多）
                    # print(result)
                except Exception as exc:
                    # 如果任务执行中出现异常
                    file_name = future_to_file[future] # 获取是哪个文件出错了
                    print(f'\n错误处理文件 {file_name}: {exc}') # 在进度条下方打印错误信息

        print(f"--- 完成处理文件夹: {folder_path} ---")

total_end_time = time.time() # 结束总计时
print(f"\n=== 所有文件夹处理完成。总耗时: {total_end_time - total_start_time:.2f} 秒 ===")

# import numpy as np
# import os
# from tqdm import tqdm

# base_folers = ["E:/ALL_Datasets/ALL"]

# datasets_folders = ["train/data", "val/data", "test/data"]

# ignore_value = -3.4028235e+38
# new_value = 1e-34


# for folder in base_folers:
#     for dataset_folder in datasets_folders:
#         folder_path = os.path.join(folder, dataset_folder)
#         files = os.listdir(folder_path)
#         print(folder_path)
#         for file in tqdm(files, desc="Processing", unit="files", ncols=100):
#             file_path = os.path.join(folder_path, file)
#             data = np.load(file_path, mmap_mode='c')
#             ignore_mask = data == ignore_value
#             data[ignore_mask] = new_value
#             np.save(file_path, data)

# base_folers = ["/root/data_temp/CHM"]

# datasets_folders = ["train/data", "val/data", "test/data"]

# for folder in base_folers:
#     for dataset_folder in datasets_folders:
#         folder_path = os.path.join(folder, dataset_folder)
#         files = os.listdir(folder_path)
#         print(folder_path)
#         for file in tqdm(files, desc="Processing", unit="files", ncols=100):
#             file_path = os.path.join(folder_path, file)
#             data = np.load(file_path)
#             ignore_mask = data == ignore_value
#             data[ignore_mask] = new_value
#             np.save(file_path, data)


