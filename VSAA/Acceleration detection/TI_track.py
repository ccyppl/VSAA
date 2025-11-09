# 开发时间：2024/11/5 11:04
import os
import pandas as pd
import numpy as np
from scipy.signal import butter, lfilter


def bandpass_filter(data, lowcut=10, highcut=500, fs=2000, order=5):
    """带通滤波(10-500Hz)"""
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    return lfilter(b, a, data)


def calculate_moving_rms(signal, window_size=60):
    """快速计算移动有效值"""
    N = len(signal)
    rms_values = np.zeros(N - window_size + 1)

    # 初始窗口平方和
    sum_squares = np.sum(signal[:window_size] ** 2)
    rms_values[0] = np.sqrt(sum_squares / window_size)

    # 递推计算后续窗口
    for i in range(1, N - window_size + 1):
        sum_squares = sum_squares - signal[i - 1] ** 2 + signal[i + window_size - 1] ** 2
        rms_values[i] = np.sqrt(sum_squares / window_size)

    return rms_values


def calculate_tii(acceleration, distance, fs=2000, window_size=60):
    """
    计算轨道冲击指数(TII)
    参数:
        acceleration: 轴箱垂向加速度数据(g)
        distance: 里程数据(m)
        fs: 采样频率(Hz)
        window_size: 移动窗口大小(采样点数)
    """
    # 1. 带通滤波
    filtered_acc = bandpass_filter(acceleration, fs=fs)

    # 2. 计算移动有效值
    rms_values = calculate_moving_rms(filtered_acc, window_size)

    # 3. 按50米划分单元并计算S平均
    unit_length = 50  # 50米一个单元
    unit_rms_max = []
    current_unit_end = distance[0] + unit_length
    current_unit_rms = []

    # 计算各单元最大RMS值
    for i, dist in enumerate(distance[:len(rms_values)]):
        if dist < current_unit_end:
            current_unit_rms.append(rms_values[i])
        else:
            if current_unit_rms:  # 保存当前单元最大值
                unit_rms_max.append(np.max(current_unit_rms))
            # 开始新单元
            current_unit_end += unit_length
            current_unit_rms = [rms_values[i]] if dist < current_unit_end else []

    # 添加最后一个单元
    if current_unit_rms:
        unit_rms_max.append(np.max(current_unit_rms))

    # 计算标定参数S平均(全部单元有效值的平均值)
    S_avg = np.mean(unit_rms_max) if unit_rms_max else 1.0  # 避免除以0

    # 4. 计算轨道冲击指数TII
    tii_values = rms_values / S_avg

    # 处理长度不一致问题(前面补NaN)
    full_length_tii = np.concatenate([np.full(window_size - 1, np.nan), tii_values])

    return full_length_tii


def process_excel_file(file_path):
    """处理单个Excel文件，只添加冲击指数列"""
    try:
        # 读取Excel文件
        df = pd.read_excel(file_path)

        # 检查必要列
        required_cols = ['右轴垂(g)', '位置']
        if not all(col in df.columns for col in required_cols):
            missing = [col for col in required_cols if col not in df.columns]
            print(f"文件 {file_path} 缺少必要列: {missing}")
            return False

        # 计算轨道冲击指数
        acceleration = pd.to_numeric(df['右轴垂(g)'], errors='coerce').values
        distance = pd.to_numeric(df['位置'], errors='coerce').values

        tii_values = calculate_tii(acceleration, distance)

        # 添加冲击指数列(覆盖已有列)
        df['右冲击指数'] = tii_values[:len(df)]  # 确保长度一致

        # 保存回原文件
        df.to_excel(file_path, index=False)

        print(f"成功处理文件: {file_path}")
        return True

    except Exception as e:
        print(f"处理文件 {file_path} 时出错: {str(e)}")
        return False


def process_folder(folder_path):
    """处理文件夹中的所有Excel文件"""
    if not os.path.exists(folder_path):
        print(f"文件夹不存在: {folder_path}")
        return

    processed_count = 0
    for file_name in os.listdir(folder_path):
        if file_name.endswith(('.xlsx', '.xls')):
            file_path = os.path.join(folder_path, file_name)
            if process_excel_file(file_path):
                processed_count += 1

    print(f"处理完成，共处理 {processed_count} 个文件")


# 示例用法
if __name__ == "__main__":
    input_folder = "Acceleration"  # 替换为你的文件夹路径
    process_folder(input_folder)