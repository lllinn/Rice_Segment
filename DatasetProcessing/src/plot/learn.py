import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import matplotlib.font_manager as fm # 导入字体管理器
from scipy import stats # 导入 scipy.stats 库


chinese_fonts = ['SimHei', 'Microsoft YaHei', 'STSong', 'STKaiti', 'LiSu', 'FangSong', 'KaiTi', 'YouYuan', 'PingFang SC']
for font_name in chinese_fonts:
    if any(font.name == font_name for font in fm.fontManager.ttflist):
        plt.rcParams['font.family'] = font_name
        break
else:
     print(f"警告: 未找到常见中文字体，尝试使用默认字体，中文可能无法正常显示。请手动配置字体路径。")
     # 如果都没有找到，可以回退到指定文件路径的方法，或者跳过设置让 Matplotlib 使用默认字体

# 设置正常显示负号
plt.rcParams['axes.unicode_minus'] = False

def learn_plotting_functions():
    """
    演示和学习常用的数据分布可视化函数（histplot, kdeplot）的用法和含义。
    """
    print("--- 开始学习绘图函数 ---")

    # 设置 Seaborn 风格，让图表更美观
    sns.set_style("whitegrid")

    # --- 1. 创建一些简单的示例数据 ---
    # 数据1：接近正态分布的数据
    data_normal = np.random.randn(200) * 15 + 50 # 200个样本，均值50，标准差15

    # 数据2：均匀分布的数据
    data_uniform = np.random.uniform(low=10, high=90, size=200) # 200个样本，分布在10到90之间

    # 数据3：双峰分布的数据 (两个正态分布叠加)
    data_bimodal = np.concatenate([np.random.randn(100) * 5 + 30, # 第一部分均值30
                                   np.random.randn(100) * 5 + 70]) # 第二部分均值70


    # --- 2. 创建一个图集和多个子图 ---
    # 我们创建2行2列共4个子图，来展示不同的绘图效果
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    # 将axes数组展平，方便我们通过索引访问每个子图
    axes = axes.flatten()

    # --- 3. 在不同的子图上绘制图表进行学习 ---

    # 子图1：绘制直方图 (Histogram)
    print("\n绘制子图1: 直方图 (Histogram)")
    ax1 = axes[0]
    # sns.histplot() 用于绘制直方图
    # data=data_normal: 指定要绘制的数据
    # bins=20: 指定直方图的柱子数量，将数据范围分成20个区间
    # ax=ax1: 指定将图绘制到第一个子图Axes对象上
    sns.histplot(data=data_normal, bins=20, ax=ax1)
    ax1.set_title("子图1: 数据1的直方图 (Histogram)")
    ax1.set_xlabel("数值范围")
    ax1.set_ylabel("频数 (Count)")
    # 解释：直方图显示数据值落在不同区间内的频数，直观展示数据分布的形状。

    # 子图2：绘制核密度估计图 (KDE Plot)
    print("绘制子图2: 核密度估计图 (KDE Plot)")
    ax2 = axes[1]
    # sns.kdeplot() 用于绘制核密度估计图
    # data=data_normal: 指定要绘制的数据
    # fill=True: 填充曲线下方的区域，使其更易于观察
    # ax=ax2: 指定将图绘制到第二个子图Axes对象上
    sns.kdeplot(data=data_normal, fill=True, ax=ax2)
    ax2.set_title("子图2: 数据1的核密度估计图 (KDE Plot)")
    ax2.set_xlabel("数值范围")
    ax2.set_ylabel("密度 (Density)")
    # 解释：KDE图提供数据分布的平滑曲线估计，可以看作是直方图的平滑版本，更侧重于展示分布的形状。

    # 子图3：在直方图上叠加 KDE
    print("绘制子图3: 直方图上叠加 KDE")
    ax3 = axes[2]
    # 在 histplot 中设置 kde=True，就可以在直方图上方叠加绘制 KDE 曲线
    sns.histplot(data=data_normal, bins=20, kde=True, ax=ax3)
    ax3.set_title("子图3: 数据1的直方图 + KDE")
    ax3.set_xlabel("数值范围")
    ax3.set_ylabel("频数 / 密度")
    # 解释：结合直方图和KDE可以同时看到数据的分箱计数和整体平滑趋势。

    # 子图4：比较不同数据的分布 (多个 KDE 曲线)
    print("绘制子图4: 比较不同数据的 KDE 分布")
    ax4 = axes[3]
    # 在同一个Axes对象上多次调用 kdeplot，可以比较不同数据集的分布
    sns.kdeplot(data=data_normal, fill=True, ax=ax4, label='数据1 (正态)')
    sns.kdeplot(data=data_uniform, fill=True, ax=ax4, label='数据2 (均匀)')
    sns.kdeplot(data=data_bimodal, fill=True, ax=ax4, label='数据3 (双峰)')
    ax4.set_title("子图4: 多个数据的 KDE 分布比较")
    ax4.set_xlabel("数值范围")
    ax4.set_ylabel("密度 (Density)")
    ax4.legend() # 添加图例来区分不同的曲线
    # 解释：将多个数据集的KDE曲线绘制在一起，是比较它们分布形状、中心位置和范围的有效方法。

    # --- 4. 调整布局并显示图表 ---
    plt.tight_layout() # 自动调整子图布局，避免重叠
    plt.show() # 显示图表窗口

    print("\n--- 绘图函数学习结束 ---")



# --- 调用函数开始学习 ---
if __name__ == "__main__":
    learn_plotting_functions()