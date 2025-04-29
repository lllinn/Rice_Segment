import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Optional # 添加这一行



def visualize_data_distribution(data_array: np.ndarray,
                                title: str = "Data Distribution Overview",
                                plot_overall: bool = True,
                                plot_features: bool = True,
                                # 修改这里：list | None 改为 Optional[list]
                                feature_indices_to_plot: Optional[list] = None,
                                n_features_to_plot: int = 5):
    """
    统一可视化输入 NumPy 数组的数据分布。

    Args:
        data_array (np.ndarray): 输入的 NumPy 数组，例如形状为 (H, W, C)。
                                 函数会将其视为 N 个样本，每个样本 C 个特征。
        title (str): 整个可视化图集的总标题。
        plot_overall (bool): 是否绘制整个数据集所有数值的整体分布图。
        plot_features (bool): 是否绘制每个特征（通道）的分布图。
        feature_indices_to_plot (list | None): 如果 plot_features 为 True，
                                               指定要绘制哪些特征（通道）的分布。
                                               例如 [0, 5, 10]。如果为 None，
                                               则根据 n_features_to_plot 绘制前几个或随机几个。
        n_features_to_plot (int): 如果 plot_features 为 True 且 feature_indices_to_plot 为 None，
                                  则绘制前 n_features_to_plot 个特征的分布图。
    """
    if not isinstance(data_array, np.ndarray):
        print("错误: 输入必须是 NumPy 数组。")
        return

    if data_array.ndim < 2:
        print("错误: 输入数组维度太低，至少需要二维数据。")
        return

    # 获取数据形状信息
    original_shape = data_array.shape
    n_features = original_shape[-1] if data_array.ndim > 2 else 1 # 如果是二维，认为只有一个特征

    print(f"分析数据形状: {original_shape}")
    print(f"识别特征数量: {n_features}")

    # 设置 Seaborn 风格，让图表更美观
    sns.set_style("whitegrid")

    # 创建图集
    fig = None
    if plot_overall and plot_features:
         # 如果既绘制整体又绘制特征，需要考虑多个子图
         # 计算特征子图的布局
         if plot_features:
             if feature_indices_to_plot is None:
                 # 绘制前 n_features_to_plot 个特征
                 features_to_viz = list(range(min(n_features, n_features_to_plot)))
             else:
                 # 绘制指定索引的特征
                 features_to_viz = [idx for idx in feature_indices_to_plot if 0 <= idx < n_features]
                 if len(features_to_viz) != len(feature_indices_to_plot):
                     print("警告: 部分指定的特征索引超出范围。")

             n_plots = len(features_to_viz) + 1 # 加1是给整体分布图留位置
             # 简单的子图布局计算
             n_cols = min(4, n_plots) # 每行最多4列
             n_rows = (n_plots + n_cols - 1) // n_cols
             fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 5, n_rows * 4))
             axes = axes.flatten() # 将axes展平，方便索引
             current_plot_idx = 0

    elif plot_overall:
        fig, ax = plt.subplots(figsize=(8, 5))
        axes = [ax]
        current_plot_idx = 0
    elif plot_features:
        if feature_indices_to_plot is None:
            features_to_viz = list(range(min(n_features, n_features_to_plot)))
        else:
            features_to_viz = [idx for idx in feature_indices_to_plot if 0 <= idx < n_features]
            if len(features_to_viz) != len(feature_indices_to_plot):
                print("警告: 部分指定的特征索引超出范围。")

        n_plots = len(features_to_viz)
        n_cols = min(4, n_plots)
        n_rows = (n_plots + n_cols - 1) // n_cols
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 5, n_rows * 4))
        axes = axes.flatten()
        current_plot_idx = 0
    else:
        print("没有选择任何要绘制的图 ('plot_overall' 和 'plot_features' 都为 False)。")
        return

    # 设置总标题
    if fig:
         fig.suptitle(title, y=1.02, fontsize=16) # y=1.02 将标题放在图集上方

    # --- 绘制整体数据分布 ---
    if plot_overall and fig:
        ax = axes[current_plot_idx]
        # 将所有数值拉平到一维数组
        data_flat = data_array.flatten()
        sns.histplot(data_flat, bins=50, kde=True, ax=ax) # 使用直方图+KDE
        ax.set_title("Overall Data Value Distribution")
        ax.set_xlabel("Value")
        ax.set_ylabel("Frequency / Density")
        current_plot_idx += 1

    # --- 绘制各特征的分布 ---
    if plot_features and fig:
         data_reshaped = data_array.reshape(-1, n_features) # 重塑为 (样本数, 特征数)

         if feature_indices_to_plot is None:
             features_to_viz = list(range(min(n_features, n_features_to_plot)))
         else:
             features_to_viz = [idx for idx in feature_indices_to_plot if 0 <= idx < n_features]

         for i, feature_idx in enumerate(features_to_viz):
             if current_plot_idx < len(axes): # 确保有足够的子图位置
                 ax = axes[current_plot_idx]
                 # 绘制当前特征的分布 (KDE图通常更平滑)
                 sns.kdeplot(data_reshaped[:, feature_idx], fill=True, ax=ax) # fill=True 填充曲线下方区域
                 ax.set_title(f"Feature {feature_idx} Distribution")
                 ax.set_xlabel(f"Feature {feature_idx} Value")
                 ax.set_ylabel("Density")
                 current_plot_idx += 1
             else:
                 print(f"警告: 子图空间不足，未能绘制所有指定的特征分布。")
                 break # 没有更多子图位置了，退出循环

    # 隐藏多余的子图（如果子图数量多于实际绘制的图）
    if fig:
        for i in range(current_plot_idx, len(axes)):
            fig.delaxes(axes[i])

        plt.tight_layout() # 自动调整子图，防止重叠
        plt.show()

# --- 示例用法 ---

# 1. 创建一个模拟的 (640, 640, 54) 数据数组
# 实际使用时，你会加载你的 .npy 文件
# 模拟数据包含一些随机值，并可能在不同特征上有不同的均值或范围
dummy_data = np.random.rand(640, 640, 54) * 100 # 随机数在 0-100 之间
# 模拟一些特征有不同的分布
dummy_data[:, :, 0] = np.random.randn(640, 640) * 5 + 50 # 特征0: 正态分布，均值50
dummy_data[:, :, 1] = np.random.poisson(lam=10, size=(640, 640)) # 特征1: 泊松分布
dummy_data[:, :, 5] = np.random.uniform(low=-20, high=20, size=(640, 640)) # 特征5: 均匀分布在-20到20

print("--- 运行整体数据分布可视化 ---")
# 绘制整体数据分布和前5个特征的分布
visualize_data_distribution(dummy_data,
                            title="Simulated Data Distribution Analysis",
                            plot_overall=True,
                            plot_features=True,
                            n_features_to_plot=5)

print("\n--- 运行只绘制特定特征分布可视化 ---")
# 只绘制特定几个特征的分布
visualize_data_distribution(dummy_data,
                            title="Distribution of Selected Features",
                            plot_overall=False,
                            plot_features=True,
                            feature_indices_to_plot=[0, 1, 5, 15, 30, 50]) # 指定要绘制的特征索引

print("\n--- 运行只绘制整体数据分布可视化 ---")
# 只绘制所有数值的整体分布
visualize_data_distribution(dummy_data,
                            title="Overall Distribution of All Values",
                            plot_overall=True,
                            plot_features=False)