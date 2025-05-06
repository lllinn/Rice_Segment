import sys
sys.path.append('./')  # 将src的上级目录加入sys.path
import os
os.environ['ALBUMENTATIONS_DISABLE_CHECK'] = '1'  # 禁用版本检查
import argparse
import yaml
from src.data.datasets import SegmentationDataset, RiceRGBVisDataset
from src.data.transforms import get_transform_from_config, get_fusedRGB_transforms
from src.models.segmentation import SegmentationModel
from src.core.trainer import SegmentationTrainer
from torch.utils.data import DataLoader
import os
from src.utils.email_util import send_email
import pytorch_lightning as pl
from tqdm import tqdm
import time
import pytorch_lightning as pl # 即使不训练，seed_everything 可能会用到
from tqdm import tqdm


# --- 测试配置 ---
# 要测试的 num_workers 值列表
# 根据你的 CPU 核心数 (24) 和初步测试结果调整这个列表
# 可以从 0 开始，然后逐步增加，例如 1, 2, 4, 8, 12, 16, 20, 24
num_workers_to_test = [0, 1, 2, 4, 8, 0]

# 每个配置要加载的 Batch 数量 (用于计时)
# 选择一个足够大，能体现稳定加载速度，但又不用遍历整个数据集的值
# 例如，数据集总大小的 1/10 或一个固定较大的值 (例如 100, 200, 500)
# 如果 Batch size 是 2，加载 100 个 Batch 意味着处理 200 张图片
NUM_BATCHES_TO_TEST = 100 # 加载前 100 个 Batch 进行计时

# --- 数据加载测试函数 ---
def test_dataloader_loading_speed(base_config, current_num_workers, num_batches_to_test):
    print("\n" + "=" * 60)
    print(f"Starting test for num_workers = {current_num_workers}")
    print("=" * 60)

    # 复制 config，避免修改原始 base_config
    config = base_config.copy()
    config['num_workers'] = current_num_workers

    # 设置随机种子 (保持和训练一致，虽然对加载速度影响小)
    #pl.seed_everything(config['random_seed'], workers=True) # 如果需要固定种子，取消注释

    # 初始化数据 (这部分使用你原有的代码)
    try:
        train_transform = get_fusedRGB_transforms(config, 'train')
        val_transform = get_fusedRGB_transforms(config, 'val') # 虽然不测 val_loader，但创建它保持结构完整

        dataset_dir = config['dataset_dir']
        train_images_dir = os.path.join(dataset_dir, config['train_images_dir'])
        train_masks_dir = os.path.join(dataset_dir, config['train_masks_dir'])
        val_images_dir = os.path.join(dataset_dir, config['val_images_dir'])
        val_masks_dir = os.path.join(dataset_dir, config['val_masks_dir'])

        print("Initializing datasets...")
        train_dataset = RiceRGBVisDataset(
            train_images_dir,
            train_masks_dir,
            transform=train_transform,
        )

        val_dataset = RiceRGBVisDataset( # 创建 val_dataset 以保持原结构，尽管不测试
            val_images_dir,
            val_masks_dir,
            transform=val_transform
        )
        print(f"Train dataset size: {len(train_dataset)}")
        print(f"Val dataset size: {len(val_dataset)}")

    except Exception as e:
        print(f"Error initializing datasets: {e}")
        print("Skipping this configuration.")
        import traceback
        traceback.print_exc()
        return # 跳过当前配置的测试

    # 创建数据加载器 (使用你原有的代码结构，应用当前的 num_workers)
    try:
        # 检查 num_workers 是否大于 CPU 核心数，给出警告（可选）
        # if current_num_workers > os.cpu_count():
        #     print(f"Warning: num_workers ({current_num_workers}) is greater than system CPU count ({os.cpu_count()}).")

        # DataLoader 参数，如果想测试 persistent_workers 或 prefetch_factor
        # 请取消注释并添加到 config 中，然后添加到下面的 DataLoader 调用中
        # config['persistent_workers'] = True # Example
        # config['prefetch_factor'] = 4 # Example

        loader_params = {
            'batch_size': config['batch_size'],
            'shuffle': True, # 你的代码是 True
            'num_workers': config['num_workers'],
            'pin_memory': config['pin_memory'], # 你的代码是 True
            
        }
        # 只有 num_workers > 0 时，persistent_workers 和 prefetch_factor 才有效
        # if config['num_workers'] > 0:
        #     if 'persistent_workers' in config:
        #         loader_params['persistent_workers'] = config['persistent_workers']
        #     if 'prefetch_factor' in config:
        #          loader_params['prefetch_factor'] = config['prefetch_factor']


        print("Creating DataLoaders...")
        train_loader = DataLoader(train_dataset, **loader_params)

        # val_loader (不参与计时，但创建保持结构完整)
        # loader_params_val = loader_params.copy()
        # loader_params_val['shuffle'] = False # Val loader通常不 shuffle
        # val_loader = DataLoader(val_dataset, **loader_params_val)
        print(f"Train DataLoader created with batch_size={config['batch_size']}, num_workers={config['num_workers']}, pin_memory={config['pin_memory']}")
        # if config['num_workers'] > 0:
        #     print(f"  persistent_workers={loader_params.get('persistent_workers', False)}, prefetch_factor={loader_params.get('prefetch_factor', 2)}") # 默认prefetch_factor=2


    except Exception as e:
        print(f"Error creating DataLoaders with num_workers={current_num_workers}: {e}")
        print("Skipping this configuration.")
        import traceback
        traceback.print_exc()
        return # 跳过当前配置的测试

    # 初始化模型 (使用你原有的代码结构，不参与计时)
    # 这步是为了确保你的代码结构能正常运行到 DataLoader 后面，
    # 但模型的加载/初始化时间不计入数据加载测试时间
    try:
        print("Initializing model (not part of timing)...")
        if config['resume']:
            # 模拟加载模型，如果实际加载模型很慢，这会影响总时间，但不是我们要测的加载时间
             model = SegmentationModel.load_from_checkpoint(config['checkpoint_path'])
        else:
            model = SegmentationModel(config)
        print("Model initialized.")
    except Exception as e:
        print(f"Error initializing model: {e}")
        # 模型初始化失败不影响 DataLoader 测试，但还是打印错误
        import traceback
        traceback.print_exc()


    # --- 数据加载测试循环 ---
    print(f"\nStarting data loading test: Fetching {num_batches_to_test} batches from train_loader...")
    start_time = time.time()

    try:
        # 迭代加载指定数量的 Batch
        # 使用 tqdm 显示进度，并限制迭代次数
        # 注意：如果 num_batches_to_test 大于总 batch 数，tqdm 的 total 参数需要调整
        total_batches_in_epoch = len(train_loader)
        batches_to_fetch = min(num_batches_to_test, total_batches_in_epoch)

        for i, batch in enumerate(tqdm(train_loader, desc=f"Loading batches (workers={current_num_workers})")):
            # 这里是实际从 DataLoader 获取数据的地方
            # print(f"  Fetched batch {i+1}") # 可选：调试用
            # 不需要将数据移动到 GPU，因为我们只测试 CPU 端的加载和预处理速度
            batch = [item.to('cuda') for item in batch] # 真实训练时会做这步

            # if i >= batches_to_fetch - 1:
            #     break # 加载到指定的 batch 数量就停止

        end_time = time.time()
        duration = end_time - start_time

        batches_per_second = batches_to_fetch / duration
        samples_per_second = batches_per_second * config['batch_size']

        print(f"\nResults for num_workers = {current_num_workers}:")
        print(f"  Loaded {batches_to_fetch} batches in {duration:.4f} seconds.")
        print(f"  Loading speed: {batches_per_second:.2f} batches/sec, {samples_per_second:.2f} samples/sec.")

    except Exception as e:
        print(f"\nError during DataLoader iteration with num_workers={current_num_workers}: {e}")
        import traceback
        traceback.print_exc()

    print("=" * 60)


# --- 主执行逻辑 ---
if __name__ == '__main__':
    # 加载配置 (使用你原有的 argparser 和 yaml 加载)
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default="./config/fusedRGB.yaml",
                        help="Path to the configuration file.")
    args = parser.parse_args()

    # 检查配置文件是否存在
    if not os.path.exists(args.config):
        print(f"Error: Configuration file not found at {args.config}")
        sys.exit(1)

    with open(args.config, encoding='utf-8') as f:
        base_config = yaml.safe_load(f)

    print("Configuration loaded.")
    print(f"Base Batch Size: {base_config.get('batch_size', 'N/A')}") # 使用get避免key错误
    print(f"Dataset Directory: {base_config.get('dataset_dir', 'N/A')}")
    print(f"Testing by loading the first {NUM_BATCHES_TO_TEST} batches for each configuration.")
    print("---")


    # 运行数据加载速度测试
    for num_workers in num_workers_to_test:
        # 在测试每个 num_workers 配置之前，稍微等待一下，确保资源被释放
        # 对于 num_workers > 0，这有助于清理之前的 worker 进程
        time.sleep(3)
        test_dataloader_loading_speed(base_config, num_workers, NUM_BATCHES_TO_TEST)

    print("\nDataLoader partial loading speed test finished.")
    print("Compare the loading speeds (batches/sec or samples/sec) for different num_workers values.")
    print("The configuration with the highest speed is likely the most efficient for data loading in your setup.")
    print("Remember this test only measures CPU-side loading/preprocessing and does not include GPU computation.")