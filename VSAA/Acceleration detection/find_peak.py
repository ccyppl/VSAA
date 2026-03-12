# 开发时间：2025/4/13 20:44
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

# 设置字体样式
matplotlib.rcParams['font.family'] = ['Times New Roman', 'SimHei']  # 同时支持英文字体和中文
matplotlib.rcParams['axes.unicode_minus'] = False


# 读取Excel文件
excel_filename = "Acceleration\CitData_240617092850.xlsx"
sheet_name = "Sheet1"

data = pd.read_excel(excel_filename, sheet_name=sheet_name)

# 填补空值为0
data['左冲击指数'].fillna(0, inplace=True)

peaks_data = pd.DataFrame()

# 定义分段长度
segment_length = 100

# 获取位置和列数据
accurate_position = data['精确位置']
column1 = data['右轴垂(g)']
column2 = data['右冲击指数']

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
        segment_column1 = segment_data['右轴垂(g)']
        segment_column2 = segment_data['右冲击指数']

        # 检测冲击指数峰值
        peaks, properties = find_peaks(segment_column2, height=0.3, distance=1, prominence=0.1)

        # 如果有峰值点，记录数据
        if len(peaks) > 0:
            current_peaks_data = segment_data.iloc[peaks].copy()
            current_peaks_data['Segment number'] = segment_number
            peaks_data = pd.concat([peaks_data, current_peaks_data])

        # 创建图形
        fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(6, 8))

        # 绘制左轴垂(g)数据
        ax1.plot(segment_position, segment_column1, color='black',linewidth=1.5)
        ax1.set_ylabel('Acceleration/g', size=12)
        ax1.set_ylim(-30, 70)
        ax1.tick_params(axis='y', labelsize=12)
        ax1.set_xlim(current_start, current_end)
        # 绘制左冲击指数数据和峰值点
        ax2.plot(segment_position, segment_column2, color='black')
        ax2.plot(segment_position.iloc[peaks], segment_column2.iloc[peaks], 'ro')
        ax2.set_ylabel('Track impact index', size=12)
        ax2.set_ylim(0, 4)
        ax2.set_xlabel('Mileage/km', size=12)
        ax2.tick_params(axis='x', labelsize=12)
        ax2.tick_params(axis='y', labelsize=12)
        #ax2.legend()

        # 设置X轴范围和标题

        ax2.set_xlim(current_start, current_end)
        fig.suptitle(f'{segment_number}: {current_start:.2f}km - {current_end:.2f}km')

        plt.tight_layout()
        plt.show()

        segment_number += 1

    current_start += segment_length

# 保存峰值点数据到新Excel文件
if not peaks_data.empty:
    output_filename = "轨冲击指数峰值点\冲击指数峰值点数据.xlsx"
    peaks_data.to_excel(output_filename, index=False)
    print(f"已保存峰值点数据到: {output_filename}")
else:
    print("未检测到任何峰值点")

print("绘制完所有分段折线图。")