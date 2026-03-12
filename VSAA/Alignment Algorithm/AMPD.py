# 开发时间：2025/4/13 21:07
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
from scipy.signal import savgol_filter

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


def AMPD_optimized(data, max_k=None):

    n = len(data)
    max_k = max_k or min(100, n // 2)
    p_data = np.zeros(n, dtype=int)

    # 第一阶段：确定最优尺度
    arr_rowsum = []
    for k in range(1, max_k + 1):
        mask = (data[k:-k] > data[:-2 * k]) & (data[k:-k] > data[2 * k:])
        arr_rowsum.append(-np.sum(mask))
    max_window_length = np.argmin(arr_rowsum) + 1  # +1因为k从1开始

    # 第二阶段：检测峰值
    for k in range(1, max_window_length + 1):
        mask = (data[k:-k] > data[:-2 * k]) & (data[k:-k] > data[2 * k:])
        p_data[k:-k][mask] += 1

    # 非极大值抑制
    peaks = np.where(p_data == max_window_length)[0]
    final_peaks = []
    for i in peaks:
        if data[i] == np.max(data[max(0, i - max_window_length):min(n, i + max_window_length)]):
            final_peaks.append(i)
    return np.array(final_peaks)


# 读取Excel文件中的数据
excel_filename = "CitData_240617092850.xlsx"
sheet_name = "Sheet1"
data = pd.read_excel(excel_filename, sheet_name=sheet_name)

# 填补空值为0
data['左冲击指数'].fillna(0, inplace=True)

# 创建DataFrame用于存储所有峰值点数据
all_peaks_data = pd.DataFrame()

# 从数据中获取列
accurate_position = data['精确位置']
column1 = data['左轴垂(g)']
column2 = data['左冲击指数']

# 定义分段长度
segment_length = 100

# 循环处理每个分段
start_pos = accurate_position.min()
end_pos = accurate_position.max()
current_start = start_pos
segment_number = 1

while current_start <= end_pos:
    current_end = current_start + segment_length

    # 筛选当前段数据
    segment_mask = (accurate_position >= current_start) & (accurate_position <= current_end)
    segment_data = data[segment_mask]

    if not segment_data.empty:
        segment_position = segment_data['精确位置']
        segment_column1 = segment_data['左轴垂(g)']
        segment_column2 = segment_data['左冲击指数']

        # 对冲击指数进行平滑处理
        segment_column2_smoothed = savgol_filter(segment_column2, window_length=5, polyorder=2)

        # 使用AMPD算法检测峰值
        peaks = AMPD_optimized(segment_column2_smoothed)

        # 如果有峰值点，记录数据
        if len(peaks) > 0:
            current_peaks_data = segment_data.iloc[peaks].copy()
            current_peaks_data['分段编号'] = segment_number
            all_peaks_data = pd.concat([all_peaks_data, current_peaks_data])

        # 创建图形
        fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(10, 8))

        # 绘制左轴垂(g)数据
        ax1.plot(segment_position, segment_column1, color='blue')
        ax1.set_ylabel('轴箱振动加速度/g', size=12)
        ax1.set_ylim(-30, 30)
        ax1.tick_params(axis='y', labelsize=12)

        # 绘制左冲击指数数据和峰值点
        ax2.plot(segment_position, segment_column2, color='black', label='冲击指数')
        ax2.plot(segment_position.iloc[peaks], segment_column2.iloc[peaks], 'ro', markersize=5, label='AMPD峰值')
        ax2.set_ylabel('冲击指数', size=12)
        ax2.set_ylim(0, 4)
        ax2.set_xlabel('里程/m', size=12)
        ax2.tick_params(axis='both', labelsize=12)
        ax2.legend(fontsize=10)

        # 设置X轴范围和标题
        ax1.set_xlim(current_start, current_end)
        ax2.set_xlim(current_start, current_end)
        fig.suptitle(f'段 {segment_number}: {current_start:.2f} 米 - {current_end:.2f} 米', fontsize=12)

        plt.tight_layout()
        plt.show()

        segment_number += 1

    current_start += segment_length

# 保存所有峰值点数据到新Excel文件
if not all_peaks_data.empty:
    output_filename = "冲击指数峰值点数据.xlsx"
    # 重置索引并删除原索引列
    all_peaks_data.reset_index(drop=True, inplace=True)
    all_peaks_data.to_excel(output_filename, index=False)
    print(f"已保存{len(all_peaks_data)}个峰值点数据到: {output_filename}")
else:
    print("未检测到任何峰值点")

print("绘制完所有分段折线图。")