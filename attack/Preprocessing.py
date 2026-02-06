import warnings
warnings.filterwarnings("ignore")

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from scipy.interpolate import CubicSpline

sys.path.append('attack\\csitool')
import csitools
from csitool.passband import lowpass
from csitool.read_pcap import NEXBeamformReader

USE_BUTTERWORTH = True 
USE_PCA = True 
N_COMPONENTS = 2  
DATA_PATH_RAW = 'data/1-111000/'
DATA_PATH_OUTPUT = 'intermediate variable/1-111000/'
PCAP_FILENAME = 'capture.pcap'


def validate_csi_shape(csi):
    if csi is None or csi.size == 0:
        raise ValueError("读取的CSI数据为空")
    if csi.ndim < 3:
        raise ValueError(f"CSI矩阵维度不足，当前为: {csi.ndim}维")


def preprocess_csi(csi_matrix):

    csi_first = csi_matrix[:, :, 0, 0]
    csi_first[csi_first == -np.inf] = np.nan
    if np.all(np.isnan(csi_first)):
        raise ValueError("CSI矩阵全为NaN，无法处理")

    imputer = SimpleImputer(missing_values=np.nan, strategy='mean')  # 用列的平均填充
    csi_imputed = imputer.fit_transform(csi_first)

    csi_squeezed = np.squeeze(csi_imputed).T  # 转置为 (子载波, 时间戳)
    return csi_squeezed


def trim_to_10_seconds(timestamps, csi_data):

    timestamps = timestamps - timestamps[0]
    ten_index = np.argmax(timestamps > 10)
    if ten_index == 0:
        raise ValueError("数据不足10秒，无法截取")

    return timestamps[:ten_index], csi_data[:, :ten_index], ten_index


def interpolate_csi(csi_data, timestamps):

    new_timestamps = np.linspace(0, timestamps[-1], len(timestamps))
    for i in range(csi_data.shape[0]):
        cs = CubicSpline(timestamps, csi_data[i])
        csi_data[i] = cs(new_timestamps)
    return csi_data, new_timestamps


def apply_butterworth_filter(csi_data, timestamps):

    fs = int(len(timestamps) / 10)
    for i in range(csi_data.shape[0]):
        csi_data[i] = lowpass(csi_data[i], 3, fs, 5)
    return csi_data


def apply_pca(csi_data, n_components):

    pca = PCA(n_components=8)
    transformed = pca.fit_transform(csi_data.T).T
    print("各主成分的方差贡献率:", pca.explained_variance_ratio_)
    reduced = transformed[:n_components]
    avg_rows = np.mean(reduced, axis=1)
    top_indices = np.argsort(avg_rows)[:n_components]
    top_rows = reduced[top_indices]
    summed = np.sum(top_rows, axis=0)
    return summed - np.mean(summed)


def apply_mrc(csi_data):

    csi_conj = np.conj(csi_data)
    epsilon = 1e-3
    weights = csi_conj / (np.sum(csi_conj, axis=0, keepdims=True) + epsilon)
    combined = np.sum(weights * csi_data, axis=0)
    return combined - np.mean(combined)


def plot_csi(time_axis, values, title='TIME-CSI'):
    plt.figure(figsize=(16, 5))
    plt.plot(time_axis, values, marker='o', linestyle='-', color='b', label='CSI Value')
    plt.title(title)
    plt.xlabel('Timestamp (s)')
    plt.ylabel('CSI Amplitude')
    plt.legend()
    plt.grid(True)
    plt.show(block=False)
    plt.waitforbuttonpress() 
    plt.close()


def save_outputs(csi_values, timestamps, path):
    os.makedirs(path, exist_ok=True)
    np.save(os.path.join(path, 'csi.npy'), csi_values)
    np.save(os.path.join(path, 'time_stamps.npy'), timestamps)


def ensure_dir_exists(dir_path: str):

    os.makedirs(dir_path, exist_ok=True)


def main():

    pcap_path = os.path.join(DATA_PATH_RAW, PCAP_FILENAME)
    if not os.path.exists(pcap_path):
        raise FileNotFoundError(f"未找到PCAP文件：{pcap_path}")
    
    reader = NEXBeamformReader()
    csi_data = reader.read_file(pcap_path, scaled=True)
    csi_matrix, n_frames, n_subcarriers = csitools.get_CSI(csi_data)

    print(f"总帧数: {n_frames}, 子载波数: {n_subcarriers}")
    print(f"CSI矩阵形状: {csi_matrix.shape}")

    validate_csi_shape(csi_matrix)
    csi_processed = preprocess_csi(csi_matrix)

    timestamps = csi_data.timestamps
    timestamps_trimmed, csi_trimmed, ten_index = trim_to_10_seconds(timestamps, csi_processed)

    print("前10sCSI矩阵形状:", csi_trimmed.shape)
    print("前10s时间戳矩阵形状:", timestamps_trimmed.shape)
    print("10s索引位置:", ten_index)

    csi_interpolated, uniform_timestamps = interpolate_csi(csi_trimmed, timestamps_trimmed)

    if USE_BUTTERWORTH:
        csi_filtered = apply_butterworth_filter(csi_interpolated, uniform_timestamps)
    else:
        csi_filtered = csi_interpolated

    if USE_PCA:
        csi_final = apply_pca(csi_filtered, N_COMPONENTS)
    else:
        csi_final = apply_mrc(csi_filtered)

    plot_csi(uniform_timestamps, csi_final)
    ensure_dir_exists(DATA_PATH_OUTPUT)
    save_outputs(csi_final, uniform_timestamps, DATA_PATH_OUTPUT)


if __name__ == "__main__":
    main()


