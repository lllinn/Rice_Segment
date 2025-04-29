import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd # 使用pandas方便处理和绘图
import os # 用于处理文件路径和创建目录
import matplotlib.font_manager as fm # 用于字体管理和 FontProperties
import math # 用于计算子图布局
from osgeo import gdal

# --- 解决中文显示问题 (推荐使用指定字体文件的方法) ---
# 请将 'C:/Windows/Fonts/simhei.ttf' 替换为你找到的中文字体文件的实际路径
# 如果你在前面已经成功通过 RCParams 设置了字体，可以注释掉 FontProperties 部分
# 注意 r'' 前缀表示这是一个原始字符串，可以避免反斜杠的转义问题
# 在 Windows 系统上，常见的字体路径示例如下：
# r'C:\Windows\Fonts\simhei.ttf'  # 黑体
# r'C:\Windows\Fonts\msyh.ttc'    # 微软雅黑 (注意是 .ttc)
# r'C:\Windows\Fonts\simsun.ttc'  # 宋体 (注意是 .ttc)
chinese_font_path = r'C:\Windows\Fonts\simhei.ttf' # <-- **请修改为你的字体文件路径**

# 创建 FontProperties 对象，直接指向字体文件
font_properties = None
if os.path.exists(chinese_font_path):
    try:
        font_properties = fm.FontProperties(fname=chinese_font_path)
        print(f"成功加载指定字体文件: {chinese_font_path}")
    except Exception as e:
        print(f"加载字体文件时发生错误: {e}")
        font_properties = None
else:
    print(f"警告: 指定的字体文件不存在！请检查路径: {chinese_font_path}")
    print("将尝试使用 Matplotlib 默认字体或 RCParams 设置。")
    # 如果字体文件不存在，可以尝试回退到 RCParams 设置，前提是前面已经设置过
    # 注意：这里不再次设置 RCParams，假设你已经在脚本开头设置过
    pass

# 确保负号正确显示
mpl.rcParams['axes.unicode_minus'] = False

# -------------------------------------------------------

# 函数名保留你提供的，但功能是绘制各特征子图
def plot_feature(input_data: np.ndarray,
                 label_data: np.ndarray,
                 output_path: str = None,
                 ignore_index: float = 1e-34,
                 n_cols: int = 4, # 控制每行显示多少个子图
                 use_subplots_adjust: bool = False, # 是否使用 subplots_adjust 手动调整间距
                 wspace: float = 0.4, # subplots_adjust 的宽度间距
                 hspace: float = 0.6 # subplots_adjust 的高度间距
                ):
    """
    为每个特征绘制其在指定类别 (标签 1-7) 上的数据分布 (KDE图) 子图。
    图例将集中显示在一个空白子图中。

    Args:
        input_data (np.ndarray): 特征数据，形状为 (..., n_features)。
                                 函数内部会尝试重塑为 (n_samples, n_features)。
        label_data (np.ndarray): 标签数据，形状应与 input_data 的前几维一致，
                                 函数内部会尝试重塑为 (n_samples,)。
        output_path (str, optional): 保存图表的完整文件路径 (包括文件名和扩展名)。
                                     如果为 None，则显示图表。默认为 None。
        ignore_index (float): 在 input_data 中要忽略的数值。这些数值不会用于绘图。
                              默认为 1e-34。
        n_cols (int): 子图网格中每行的列数。
        use_subplots_adjust (bool): 如果为 True, 使用 fig.subplots_adjust 进行手动间距调整;
                                    如果为 False (默认), 使用 plt.tight_layout。
        wspace (float): 使用 subplots_adjust 时子图之间的宽度间距。
        hspace (float): 使用 subplots_adjust 时子图之间的高度间距。
    """
    print("\n--- 开始绘制各特征的分布子图 ---")
    print(f"输入特征数据形状: {input_data.shape}")
    print(f"输入标签数据形状: {label_data.shape}")
    print(f"忽略索引值: {ignore_index}")

    # 定义类别名称映射 (标签值 -> 名称)
    category_names = {
        1: '道路 (Road)',
        2: '甘蔗 (Sugarcane)',
        3: '水稻正常 (Rice Normal)',
        4: '水稻严重倒伏 (Rice Severe)',
        5: '水稻轻微倒伏 (Rice Mild)',
        6: '杂草 (Weed)',
        7: '异常区域 (Abnormal)'
    }
    target_labels = list(category_names.keys()) # [1, 2, 3, 4, 5, 6, 7]

    # --- 1. 重塑数据以进行处理 ---
    # n_samples_input = np.prod(input_data.shape[:-1]) if input_data.ndim > 1 else input_data.shape[0]
    n_features = input_data.shape[0] if input_data.ndim > 1 else 1
    
    try:
        input_data_reshaped = input_data.reshape(n_features, -1)
        label_data_reshaped = label_data.reshape(-1)
    except Exception as e:
        print(f"错误: 数据重塑失败，请检查输入数据形状是否匹配。{e}")
        return

    if input_data_reshaped.shape[1] != label_data_reshaped.shape[0]:
         print("错误: 重塑后特征数据的样本数量与标签数据的样本数量不一致。")
         return

    print(f"重塑后样本数量: {input_data_reshaped.shape[1]}")
    print(f"识别特征数量: {n_features}")

    # --- 2. 筛选有效样本索引 (只保留标签在 1 到 7 之间的) ---
    valid_label_mask = (label_data_reshaped >= min(target_labels)) & (label_data_reshaped <= max(target_labels))
    indices_to_plot_all_features = np.where(valid_label_mask)[0] # 筛选出所有属于目标类别的样本索引

    if len(indices_to_plot_all_features) == 0:
        print("警告: 没有找到标签在 1 到 7 之间的样本数据。无法绘图。")
        return

    # --- 3. 创建子图网格 ---
    n_plots = n_features # 需要绘制特征图的子图数量
    # 我们需要为图例留出一个额外的子图位置
    total_axes_needed = n_features + 1 # 总共需要的子图数量 (特征图数量 + 1 个图例位置)

    # 确保 n_cols 至少为 1
    n_cols = max(1, n_cols)
    # 计算需要的总行数，向上取整
    n_rows_total = math.ceil(total_axes_needed / n_cols)

    # Adjust figsize based on the number of features and columns
    # 增加 figsize 的乘数，以减少重叠
    fig_height_per_row = 5 # 调整每行子图的高度乘数 (英寸)
    fig_width_per_col = 6 # 调整每列子图的宽度乘数 (英寸)

    fig_height_total = n_rows_total * fig_height_per_row
    fig_width_total = n_cols * fig_width_per_col


    # squeeze=False 确保即使只有一个子图，axes也是一个二维数组
    fig, axes = plt.subplots(n_rows_total, n_cols, figsize=(fig_width_total, fig_height_total), squeeze=False)
    axes = axes.flatten() # 将axes数组展平，方便索引

    # --- 4. 遍历每个特征，绘制其在各类别上的分布 ---
    print("\n正在为每个特征提取数据并绘制分布图...")

    # 变量用于存储图例信息
    legend_handles = None
    legend_labels = None
    legend_captured = False # 标记是否已获取图例信息

    # 遍历每一个特征 (0 到 n_features - 1)
    for feature_index in range(n_features):
        # 获取当前特征对应的子图Axes对象
        # 特征图索引是 0 到 n_features-1
        ax = axes[feature_index]

        # 临时的列表，用于存放当前特征在各类别上的数据 DataFrame
        feature_data_list = []

        # 遍历目标类别 1 到 7，为当前特征收集数据
        for label_value in target_labels: # 使用 target_labels 列表
            # 筛选出当前类别、且标签在目标范围内的样本索引
            # 我们已经有了所有目标类别的样本索引 indices_to_plot_all_features
            # 现在从中找出属于当前 label_value 的索引
            current_category_indices_in_target = indices_to_plot_all_features[label_data_reshaped[indices_to_plot_all_features] == label_value]

            if len(current_category_indices_in_target) == 0:
                continue # 跳过当前类别，处理下一个

            # 提取当前类别、当前特征的数据 (形状为 (n_category_samples,))
            # category_feature_values = input_data_reshaped[current_category_indices_in_target, feature_index]
            category_feature_values = input_data_reshaped[feature_index, current_category_indices_in_target]

            # 忽略掉等于 ignore_index 的数值
            filtered_values = category_feature_values[category_feature_values != ignore_index]

            if len(filtered_values) == 0:
                continue # 跳过当前类别，处理下一个

            # 创建当前类别、当前特征的数据 DataFrame (长格式)
            temp_df = pd.DataFrame({
                '数值 (Value)': filtered_values,
                '类别 (Category)': category_names[label_value] # 使用类别名称
            })

            # 将当前类别的数据添加到当前特征的数据列表
            feature_data_list.append(temp_df)

        # 将当前特征的所有类别数据 DataFrame 合并成一个大的 DataFrame
        if not feature_data_list:
            print(f"警告: 特征 {feature_index} 在所有目标类别中都没有有效数据。跳过绘制。")
            # 如果某个子图没有任何数据绘制，直接隐藏它
            ax.set_visible(False)
            continue # 跳过绘制当前特征

        feature_plotting_df = pd.concat(feature_data_list, ignore_index=True)

        # --- 绘制 KDE 图 ---
        # 使用 Seaborn 绘制 KDE 图
        # data=feature_plotting_df: 指定输入数据是这个 DataFrame
        # x='数值 (Value)': 指定X轴是 DataFrame 中的 '数值 (Value)' 列
        # hue='类别 (Category)': 根据 '类别 (Category)' 列的值分组绘制不同的曲线
        # fill=False: 不填充
        # common_norm=False: 每个分布独立标准化
        # ax=ax: 指定绘制到当前子图Axes对象上
        # legend=False: **关键**，关闭当前子图的自动图例
        current_plot = sns.kdeplot(data=feature_plotting_df, x='数值 (Value)', hue='类别 (Category)',
                                   fill=True, common_norm=False, ax=ax, legend=False) # 关闭自动图例

        # --- 获取图例信息 (只需要从任意一个成功绘制的图中获取一次) ---
        if not legend_captured:
             try:
                 # current_plot 是 Axes 对象
                 legend_handles, legend_labels = ax.get_legend_handles_labels()
                 if legend_handles and legend_labels:
                     legend_captured = True
                     print(f"特征 {feature_index}: 成功获取图例信息。Handles count: {len(legend_handles)}, Labels: {legend_labels}")
                 else:
                     print(f"特征 {feature_index}: get_legend_handles_labels() 返回空。")
             except Exception as e:
                 print(f"特征 {feature_index}: 获取图例信息时发生错误: {e}")


        # --- 5. 设置当前子图的标题和标签 ---
        title_text = f"特征 {feature_index} 分布"
        xlabel_text = "数值范围"
        ylabel_text = "密度"

        # 应用字体属性到子图标题和标签
        if font_properties:
            ax.set_title(title_text, fontproperties=font_properties, fontsize=12) # 子图标题字号可以小一点
            ax.set_xlabel(xlabel_text, fontproperties=font_properties, fontsize=10)
            ax.set_ylabel(ylabel_text, fontproperties=font_properties, fontsize=10)
        else:
            ax.set_title(title_text, fontsize=12)
            ax.set_xlabel(xlabel_text, fontsize=10)
            ax.set_ylabel(ylabel_text, fontsize=10)

        ax.grid(True, linestyle='--', alpha=0.6) # 添加网格线

    # --- 处理未使用的子图和集中显示图例 ---
    # 找到所有 Axes 的索引
    all_axes_indices = list(range(len(axes)))
    # 未被用于绘制特征图的 Axes 的索引集合 (最初，所有在 n_features 及之后的索引都是未使用的)
    initially_unused_indices = set(range(n_features, len(axes)))


    legend_placed = False
    # 尝试在最初未使用的子图中的第一个位置放置图例 (即索引为 n_features 的 Axes)
    legend_ax_index = n_features
    # 只有当存在这个 Axes 且图例信息已获取时才尝试放置
    if legend_ax_index < len(axes) and legend_captured:
        legend_ax = axes[legend_ax_index]

        # 在这个子图上创建图例
        # loc='center' 将图例放在子图的中心位置
        legend = legend_ax.legend(legend_handles, legend_labels, loc='center', fontsize=10)

        # 设置图例的字体，如果 font_properties 可用的话
        if font_properties:
            for text in legend.get_texts():
                text.set_fontproperties(font_properties)

        # 隐藏放置图例的子图的坐标轴
        legend_ax.axis('off') # 隐藏坐标轴线、刻度、标签等

        legend_placed = True
        print(f"图例已放置在子图 {legend_ax_index}。")

    elif legend_captured:
         print("警告: 没有找到足够的空白子图来放置图例 (索引 >= n_features 且 < total_axes)。图例未集中显示。")
         # 这个分支理论上不常发生，因为 total_axes_needed = n_features + 1 应该至少留一个位置
         pass # 没有空白子图，图例不会显示在空白区域


    # 隐藏所有未被用于绘制特征图且未用于放置图例的子图
    # 遍历所有最初未使用的子图索引
    for i in initially_unused_indices:
        # 如果当前子图索引不是用于放置图例的那个 (只有图例成功放置时才跳过)
        if not (legend_placed and i == legend_ax_index):
             axes[i].set_visible(False)
        # else: # 如果这个是图例所在的 Axes 且图例成功放置，它应该是可见的（因为只关闭了轴）
        #     axes[i].set_visible(True) # 确保它可见


    # --- 6. 调整布局并显示或保存图表 ---
    # rect参数 [left, bottom, right, top] 调整子图区域相对于Figure的比例
    # 留出底部和顶部的空间给总标题和可能的底部文字
    # 调整 bottom 以防止 X 轴标签与下一行子图重叠
    # 调整 top 以防止总标题与最上面一行子图重叠
    if use_subplots_adjust:
         print(f"使用 subplots_adjust 调整布局: wspace={wspace}, hspace={hspace}")
         # 这些参数值需要根据你的图表密度和文本长度进行调整
         # left, right, bottom, top 控制整个子图区域的边缘
         # wspace, hspace 控制子图之间的间距
         fig.subplots_adjust(left=0.05, right=0.95, bottom=0.08, top=0.92, wspace=wspace, hspace=hspace)
         # 如果使用了 subplots_adjust, suptitle 的位置需要额外调整，或者放到外面
         # 我们还是在外面调用suptitle，并调整y使其不与最上面一行的子图重叠
         #fig.suptitle(suptitle_text, y=0.98, fontsize=16, fontproperties=font_properties if font_properties else None) # 如果suptitle在里面，需要调整y

    else:
         print("使用 tight_layout 调整布局...")
         # rect参数用于tight_layout，为suptitle预留空间
         # 调整 bottom 以防止 X 轴标签与下一行子图重叠
         # 调整 top 以防止总标题与最上面一行子图重叠
         # 这些参数值需要根据你的图表密度和文本长度进行调整
         # [left, bottom, right, top]
         plt.tight_layout(rect=[0, 0.05, 1, 0.95]) # 示例值，可以根据需要调整


    # 添加总标题 (无论使用哪种布局调整)
    suptitle_text = "各特征在不同地物类别上的分布 (KDE)"
    # suptitle的位置在tight_layout之后或subplots_adjust之外设置
    if font_properties:
         fig.suptitle(suptitle_text, y=0.98, fontsize=16, fontproperties=font_properties) # y=0.98 稍微向下调整位置，避免重叠
    else:
         fig.suptitle(suptitle_text, y=0.98, fontsize=16)


    if output_path:
        try:
            # 确保输出目录存在
            output_dir = os.path.dirname(output_path)
            if output_dir and not os.path.exists(output_dir):
                os.makedirs(output_dir)
                print(f"创建输出目录: {output_dir}")

            # 保存图表前再次调整布局，有时有帮助 (如果之前没用subplots_adjust)
            # if not use_subplots_adjust:
            #     fig.tight_layout(rect=[0, 0.05, 1, 0.95]) # 再次尝试调整布局

            plt.savefig(output_path, bbox_inches='tight', dpi=300) # 保存图表
            print(f"图表已保存到: {output_path}")
        except Exception as e:
            print(f"保存图表失败: {e}")
            plt.show() # 保存失败则显示图表
    else:
        plt.show() # 如果没有指定保存路径，则显示图表

    print("\n--- 绘制特征分布子图完成 ---")

# --- 示例用法 ---

# 假设你的原始数据加载如下：
# features_original = np.load('your_features.npy') # 形状 (640, 640, 54)
# labels_original = np.load('your_labels.npy')     # 形状 (640, 640)

# 创建模拟数据 (包含多种标签和 ignore_index)
# 模拟形状 (H, W, C) 和 (H, W)
# H, W, C = 50, 50, 15 # 使用较小的尺寸以便运行示例，特征数量设置为15
# # 模拟一些随机特征数据
# sim_features_3d = np.random.rand(H, W, C) * 100

# # 模拟标签数据 (包含 0 和 1-7 之间的值，以及可能更高的值)
# sim_labels_2d = np.random.randint(0, 9, size=(H, W)) # 标签范围 0-8
# sim_labels_2d[sim_labels_2d == 8] = 7 # 将标签8改为7，确保有标签7
# sim_labels_2d[sim_labels_2d == 0][::5] = 0 # 确保有一些0标签
# sim_labels_2d[sim_labels_2d > 7][::5] = 0 # 确保大于7的标签也被忽略掉 (或者简单地让它们存在，但是不被绘制，当前代码逻辑会忽略它们)

# # 模拟 ignore_index 在特征数据中
# sim_ignore_value = -999.0 # 模拟一个忽略值
# sim_features_3d[::7, ::7, :] = sim_ignore_value # 在一些位置插入忽略值
# sim_features_3d[sim_labels_2d == 3][::3, :] = sim_ignore_value # 在标签为3的一些样本中也插入忽略值

# # 将模拟数据重塑为 (n_samples, n_features) 和 (n_samples,)
# sim_features_reshaped = sim_features_3d.reshape(-1, C)
# sim_labels_reshaped = sim_labels_2d.reshape(-1)
def read_tif(filename: str):
    """
    读取TIF文件并返回投影、地理变换参数和图像数据。
    
    Args:
        filename (str): TIF文件路径
    
    Returns:
        tuple: 包含投影、地理变换参数、图像数据、宽度、高度和波段数的元组
    """
    dataset = gdal.Open(filename)
    if dataset is None:
        raise FileNotFoundError(f"File not found or cannot be opened: {filename}")
    
    im_width = dataset.RasterXSize
    im_height = dataset.RasterYSize
    im_bands = dataset.RasterCount
    im_geotrans = dataset.GetGeoTransform()
    im_proj = dataset.GetProjection()
    
    im_data = dataset.ReadAsArray(0, 0, im_width, im_height)
    # del dataset  # 释放资源, 会导致有的文件卡住
    return im_proj, im_geotrans, im_data, im_width, im_height, im_bands



# 示例调用函数
if __name__ == "__main__":
    sim_ignore_value = 1e-34 # 使用默认忽略值或自定义值
    im_proj, im_geotrans, im_data, im_width, im_height, im_bands = read_tif("F:/Rice2024/Meiju1/Split_Stretch/1_RGB-3/Split_Stretch_RGB.tif")
    im_proj, im_geotrans, label_data, im_width, im_height, im_bands = read_tif("F:/Rice2024/Meiju1/Labels-shp/Meiju1_2_Lingtangkou_v5.tif")
    im_data = im_data[0:1,:,:]
    sim_features_reshaped = im_data# <-- 这里需要你的实际特征数据
    sim_labels_reshaped = label_data# <-- 这里需要你的实际标签数据

    # 假设使用上面的模拟数据
    ignore_value_for_sim = sim_ignore_value # 使用模拟数据中定义的忽略值

    print("使用模拟数据进行绘制...")

    # --- 示例 1: 使用 tight_layout 自动调整布局 (默认 n_cols=4) ---
    # print("\n--- 示例 1: 使用 tight_layout ---")
    # plot_feature(
    #     sim_features_reshaped,
    #     sim_labels_reshaped,
    #     ignore_index=ignore_value_for_sim,
    #     n_cols=4 # 每行显示4个子图
    #     # use_subplots_adjust=False, # 默认就是 False
    # )

    # --- 示例 2: 使用 subplots_adjust 手动调整布局 ---
    # 如果 tight_layout 效果不好，尝试这个
    print("\n--- 示例 2: 使用 subplots_adjust ---")
    plot_feature(
        sim_features_reshaped,
        sim_labels_reshaped,
        ignore_index=ignore_value_for_sim,
        n_cols=5, # 可以尝试不同的列数，例如5列
        use_subplots_adjust=True, # 启用手动调整
        wspace=0.3, # 调整列间距，根据需要增大
        hspace=0.5  # 调整行间距，根据需要增大
    )


    # --- 示例 3: 绘制分布图并保存到文件 ---
    # (注意：这里的路径需要是你的电脑上的有效路径，例如创建一个 output_plots 文件夹)
    # output_file = './output_plots/feature_distributions.png' # 示例保存路径
    # print(f"\n--- 示例 3: 保存图表到文件 {output_file} ---")
    # plot_feature(
    #      sim_features_reshaped,
    #      sim_labels_reshaped,
    #      output_path=output_file,
    #      ignore_index=ignore_value_for_sim,
    #      n_cols=5, # 可以尝试不同的列数
    #      use_subplots_adjust=True, # 保存时也建议使用 subplots_adjust
    #      wspace=0.3,
    #      hspace=0.5
    # )