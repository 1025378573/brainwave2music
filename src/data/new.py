import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import chirp

# 1. 从 CSV 文件加载 EEG 数据并合并前两列
def load_eeg_from_csv(file_path, duration_seconds=30):
    """
    从 CSV 文件中加载 EEG 数据并取前两列的平均值
    :param file_path: CSV 文件路径
    :param duration_seconds: 需要读取的数据时长（秒）
    :return: 合并后的 EEG 数据 (numpy 数组)
    """
    # 读取 CSV 文件
    data = pd.read_csv(file_path)
    # 计算需要的数据点数
    samples_needed = duration_seconds * 256  # 假设采样率为256Hz
    # 取前两列数据并计算平均值，只取需要的时长
    eeg_data = data.iloc[:samples_needed, :2].mean(axis=1).to_numpy()
    return eeg_data

# 2. 定义 Adaptive Chirplet Transform (ACT)
def adaptive_chirplet_transform(signal, sampling_rate, segment_length=256, chirplet_count=5):
    """
    对 EEG 信号应用 Adaptive Chirplet Transform (ACT)
    :param signal: 单通道 EEG 信号
    :param sampling_rate: 采样率
    :param segment_length: 每段信号的长度
    :param chirplet_count: 每段使用的 Chirplet 数量
    :return: (params, projections) 每段的参数和投影系数
    """
    num_segments = len(signal) // segment_length
    params = []  # 保存每段的 chirplet 参数
    projections = []  # 保存每段的投影系数

    # 对每段信号应用 Chirplet 分解
    for i in range(num_segments):
        segment = signal[i * segment_length:(i + 1) * segment_length]
        t = np.linspace(0, segment_length / sampling_rate, segment_length)
        
        # 定义频率范围
        f_min, f_max = 1, sampling_rate // 2
        frequencies = np.linspace(f_min, f_max, chirplet_count + 1)

        segment_params = []
        segment_projections = []

        for j in range(chirplet_count):
            f0, f1 = frequencies[j], frequencies[j + 1]
            chirplet = chirp(t, f0=f0, f1=f1, t1=t[-1], method='linear')
            projection = np.dot(segment, chirplet)  # 计算投影
            segment_params.append((f0, f1))  # 记录频率范围
            segment_projections.append(projection)  # 记录投影值
        
        params.append(segment_params)
        projections.append(segment_projections)
    
    return np.array(params), np.array(projections)

# 3. 转换为 Spectrogram
def act_to_spectrogram(params, projections, sampling_rate):
    """
    将 ACT 结果转换为 Spectrogram
    :param params: Chirplet 参数
    :param projections: 投影系数
    :param sampling_rate: 采样率
    :return: Spectrogram 数据
    """
    num_segments = len(projections)
    num_chirplets = projections.shape[1]

    # 定义频率和时间轴
    freq_bins = np.linspace(0, sampling_rate / 2, num_chirplets)
    time_bins = np.arange(num_segments)

    # 将投影系数转换为绝对值能量
    spectrogram = np.abs(projections)

    return time_bins, freq_bins, spectrogram

def plot_spectrogram_pure(time_bins, freq_bins, spectrogram, save_path=None):
    """
    绘制并保存纯图 Spectrogram（无标题、坐标轴、颜色条等）
    :param time_bins: 时间轴
    :param freq_bins: 频率轴
    :param spectrogram: 频谱数据
    :param save_path: 保存图片的路径
    """
    plt.figure(figsize=(12, 6))
    time_in_seconds = time_bins * (256 / 256)
    
    # 绘制 spectrogram
    plt.pcolormesh(time_in_seconds, freq_bins, spectrogram.T, shading='gouraud', cmap='viridis')
    
    # 去除坐标轴和边框
    plt.axis('off')
    
    if save_path:
        # 保存图片时去除所有边距
        plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
        plt.close()  # 关闭图形以释放内存
    else:
        plt.show()


# 5. 主函数示例
if __name__ == "__main__":
    # CSV 文件路径
    file_path = "/Users/chenyanting/Desktop/test/ECE1724-Brainwave2music/pure/samples/eeg_samples/sub-1/3/segment_0.csv"
    # 添加保存路径
    save_path = "spectrogram.png"
    sampling_rate = 256
    duration_seconds = 30

    # 1. 加载 EEG 数据
    eeg_data = load_eeg_from_csv(file_path, duration_seconds)

    # 2. 对合并信号进行 ACT
    segment_length = 256
    chirplet_count = 5
    params, projections = adaptive_chirplet_transform(eeg_data, sampling_rate, segment_length, chirplet_count)

    # 3. 转换为 Spectrogram
    time_bins, freq_bins, spectrogram = act_to_spectrogram(params, projections, sampling_rate)

    # 4. 保存 Spectrogram 到文件
    plot_spectrogram_pure(time_bins, freq_bins, spectrogram, save_path)
