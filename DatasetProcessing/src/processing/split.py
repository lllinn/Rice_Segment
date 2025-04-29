import os
import shutil
import random
from tqdm import tqdm


def split_dataset(dataset_path, labels_path=None, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, labels_suffix='.png'):
    """
    将数据集分割为训练集、验证集和测试集。
    
    Args:
        dataset_path (str): 数据集所在的目录路径。
        labels_path (str, optional): 标签文件所在的目录路径，默认为None。如果提供，则会对标签文件也进行相同的分割操作。
        train_ratio (float, optional): 训练集所占的比例，默认为0.8。
        val_ratio (float, optional): 验证集所占的比例，默认为0.1。
        test_ratio (float, optional): 测试集所占的比例，默认为0.1。
        labels_suffix (str, optional): 标签文件的扩展名，默认为'.txt'。
    
    Raises:
        AssertionError: 如果train_ratio、val_ratio和test_ratio的和不为1.0，则抛出断言错误。
    
    """
    assert train_ratio + val_ratio + test_ratio == 1.0, "Ratios must sum to 1.0"
    print(f"Splitting dataset into {train_ratio*100}% training data, {val_ratio*100}% validation data and {test_ratio*100}% testing data.............................")

    # 创建目标文件夹
    train_path = os.path.join(dataset_path, "train")
    val_path = os.path.join(dataset_path, "val")
    test_path = os.path.join(dataset_path, "test")
    for path in [train_path, val_path, test_path]:
        if not os.path.exists(path):
            os.makedirs(path)
    
    # 如果labels_path不为空，则对labels进行同样的操作
    if labels_path:
        labels_train_path = os.path.join(labels_path, "train")
        labels_val_path = os.path.join(labels_path, "val")
        labels_test_path = os.path.join(labels_path, "test")
        for path in [labels_train_path, labels_val_path, labels_test_path]:
            if not os.path.exists(path):
                os.makedirs(path)
    
    # 获取所有文件路径
    file_paths = []
    for root, dirs, files in os.walk(dataset_path):
        for file in files:
            file_path = os.path.join(root, file)
            file_paths.append(file_path)

    random.shuffle(file_paths)

    num_files = len(file_paths)
    train_end = int(num_files * train_ratio)
    val_end = int(num_files * (train_ratio + val_ratio))

    for i, file_path in tqdm(enumerate(file_paths)):
        if i < train_end:
            target_dir = train_path
        elif i < val_end:
            target_dir = val_path
        else:
            target_dir = test_path
        
        filename = os.path.basename(file_path)
        new_file_path = os.path.join(target_dir, filename)
        # 剪切图片到目标文件夹
        shutil.move(file_path, new_file_path)

        # 如果labels_path不为空，则对labels进行同样的剪切操作
        if labels_path:
            # labels_filename = filename.replace('.tif', labels_suffix)  # 假设labels文件名与图片文件名相对应，只是扩展名不同
            labels_filename = filename.split('.')[0] + labels_suffix
            labels_file_path = os.path.join(labels_path, labels_filename)
            # print(labels_file_path)
            labels_new_file_path = os.path.join(target_dir.replace(dataset_path, labels_path), labels_filename)
            # print(labels_new_file_path, new_file_path)
            shutil.move(labels_file_path, labels_new_file_path)



if __name__ == "__main__":
    # 指定总的文件夹路径
    file_head = "/home/music/wzl/segment-task/datasets/(150m)-01.06-(4-640-0.20-0.90-(0.0-0.0-1.0))-v27"
    
    images_name = r"images"
    labels_name = r"labels"  # 新增labels文件夹路径变量
    
    images_path = os.path.join(file_head, images_name)
    labels_path = os.path.join(file_head, labels_name)  # 获取labels文件夹路径
    
    split_dataset(images_path, labels_path=labels_path)
