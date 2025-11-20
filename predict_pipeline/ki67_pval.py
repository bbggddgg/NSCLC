import pandas as pd
from scipy.stats import pearsonr
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

plt.rcParams['font.size'] = 20
plt.rcParams['font.sans-serif'] = ['SimHei']

if __name__ == '__main__':
    # 文件路径
    example_root = r"/path/to/dataset"
    pred_path = os.path.join(example_root, "test_case", "tissue", "Lite-MMT-Net", "predictions.csv")
    stain_path = os.path.join(example_root, "test_case", "tissue", "Lite-MMT-Net", "ki67_2.csv")

    # 读取 CSV 文件
    pred_df = pd.read_csv(pred_path)
    stain_df = pd.read_csv(stain_path)

    # 生成 pred_df 标签列：1表示viable_tumor或necrosis_and_keratinize最大，0表示其他情况
    def generate_label_pred(row):
        max_score = max(row['viable_tumor'], row['stroma'], row['necrosis or keratinization'])
        if row['viable_tumor'] == max_score:
            return 1
        else:
            return 0

    pred_df['label_pred'] = pred_df.apply(generate_label_pred, axis=1)

    # 生成 stain_df 标签列：ratio > 0.1 时为 1，否则为 0
    stain_df['label_stain'] = stain_df['ratio'].apply(lambda x: 1 if x > 0.5 else 0)

    # 根据 x 和 y 坐标合并 pred_df 和 stain_df 数据
    merged_df = pd.merge(pred_df, stain_df, on=['x', 'y'], how='inner', suffixes=('_pred', '_stain'))

    # 为每个数据块添加区域索引，每 256 行作为一个区域
    merged_df['region'] = (merged_df.index // 4096).astype(int)
    print(len(merged_df)//4096)

    # 统计每个区域的统计信息
    region_stats = merged_df.groupby('region').agg(
        pred_count=('label_pred', lambda x: np.sum(x)),           # label_pred为1的个数
        pred_ratio=('label_pred', lambda x: np.mean(x)),          # label_pred为1的比例
        stain_ratio_avg=('ratio', lambda x: np.mean(x))              # stain_label的ratio平均值
    ).reset_index()

    # 计算 pred_ratio 和 stain_ratio_avg 的皮尔逊相关系数
    pearson_corr, p_value = pearsonr(region_stats['pred_ratio'], region_stats['stain_ratio_avg'])

    # 打印结果
    # print(region_stats.head())  # 查看前几个区域的统计结果
    print(region_stats)
    print(f"皮尔逊相关系数: {pearson_corr:.4f}")
    print(f"p-value: {p_value:.4e}")



    # 计算 pred_ratio 和 stain_ratio_avg 的皮尔逊相关系数
    pearson_corr, p_value = pearsonr(region_stats['pred_ratio'], region_stats['stain_ratio_avg'])

    # 绘制相关性散点图
    plt.figure(figsize=(8, 6))

    # 使用 Seaborn 绘制散点图，并添加回归线
    sns.regplot(
        x='pred_ratio',
        y='stain_ratio_avg',
        data=region_stats,
        scatter_kws={'s': 50, 'color': 'royalblue'},  # 散点样式
        line_kws={'color': 'red', 'linewidth': 2},  # 回归线样式
        # ci=None  # 不显示置信区间
    )

    # 添加标题和标签
    plt.title("与Ki-67免疫组化相关性", fontsize=24)
    plt.xlabel("预测的存活肿瘤区域比例", fontsize=20)
    plt.ylabel("Ki-67染色阳性区域比例", fontsize=20)

    # 添加相关性系数和p值的注释
    # plt.text(0.3, 0.8, f"Pearson: {pearson_corr:.2f}\nP-value: {p_value:.1e}", fontsize=12,
    #          bbox=dict(facecolor='white', alpha=0.5, edgecolor='black'), fontname='Arial')
    # 获取数据范围
    x_min, x_max = region_stats['pred_ratio'].min(), region_stats['pred_ratio'].max()
    y_min, y_max = region_stats['stain_ratio_avg'].min(), region_stats['stain_ratio_avg'].max()

    # 动态设置注释位置
    text_x = x_min + (x_max - x_min) * 0.55  # x轴起点右侧 80%
    text_y = y_max - (y_max - y_min) * 0.95  # y轴顶部下方 95%

    # 添加相关性系数和p值的注释
    plt.text(
        text_x, text_y,
        f"皮尔逊相关系数: {pearson_corr:.4f}\n\np值: {p_value:.1e}",
        fontsize=20, color='black',
    )

    # 调整图表布局并保存为高质量图片
    plt.grid(visible=False)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)

    # 去除顶部和右侧边框
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)

    plt.tight_layout()
    save_path = os.path.join(example_root, "test_case", "tissue", "Lite-MMT-Net", "correlation_scatter_plot_ki67.png")
    plt.savefig(save_path, dpi=400, transparent=True)
    plt.show()

