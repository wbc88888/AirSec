import os
import librosa
import numpy as np
from sklearn.cluster import KMeans
import matplotlib
import matplotlib.pyplot as plt

matplotlib.rcParams['font.sans-serif'] = ['Microsoft YaHei']
matplotlib.rcParams['axes.unicode_minus'] = False


DATA_DIR_RAW = 'data/1-111000/'
DATA_DIR_OUT = 'intermediate variable/1-111000/'
AUDIO_FILENAME = 'temp_denoised.wav'

TRIM_DURATION = 0        
TRIM_DURATION_2 = 10     
FRAME_SIZE = 512        
HOP_LENGTH = 256       
PERCENTILE_THRESHOLD = 96 
N_CLUSTERS = 6          
RANDOM_STATE = 42      

# —————————————— 辅助函数 ——————————————
def load_audio(path: str):
    if not os.path.isfile(path):
        raise FileNotFoundError(f"未找到音频文件：{path}")
    y, sr = librosa.load(path, sr=None)
    if y.size == 0 or sr <= 0:
        raise ValueError("加载音频失败，可能文件为空或采样率无效")
    return y, sr

def trim_audio(y: np.ndarray, sr: int, trim_sec: float, trim_sec_2: float):
    trim_samples = int(sr * trim_sec)
    trim_samples_2 = int(sr * trim_sec_2)
    # if y.shape[0] <= trim_samples or y.shape[0] <= trim_samples_2:
    #     print("警告：音频长度小于裁剪时长，跳过裁剪。")
    #     return y
    return y[trim_samples:trim_samples_2]

def compute_short_time_energy(y: np.ndarray, frame_size: int, hop_length: int):
    num_frames = int(np.ceil((len(y) - frame_size) / hop_length)) + 1
    energies = []
    for i in range(num_frames):
        start = i * hop_length
        frame = y[start : start + frame_size]
        if frame.size == 0:
            break
        energies.append(np.sum(frame ** 2))
    return np.array(energies)

def ensure_dir_exists(dir_path: str):
    os.makedirs(dir_path, exist_ok=True)

if __name__ == "__main__":
    audio_path = os.path.join(DATA_DIR_RAW, AUDIO_FILENAME)
    out_dir = DATA_DIR_OUT
    ensure_dir_exists(out_dir)

    y, sr = load_audio(audio_path)
    duration = len(y) / sr
    print(f"音频时长: {duration:.2f} 秒，采样率: {sr} Hz，样本点数: {y.shape[0]}")
    
    y = trim_audio(y, sr, TRIM_DURATION, TRIM_DURATION_2)

    plt.figure(figsize=(16, 4))
    plt.plot(np.linspace(0, len(y)/sr, len(y)), y)
    plt.title("Sound")
    plt.xlabel("Time(second)")
    plt.ylabel("Amplitude")
    plt.tight_layout()
    plt.show(block=False)
    plt.waitforbuttonpress() 
    plt.close()

    frame_energy = compute_short_time_energy(y, FRAME_SIZE, HOP_LENGTH)
    threshold_value = np.percentile(frame_energy, PERCENTILE_THRESHOLD)
    print(f"能量阈值（第 {PERCENTILE_THRESHOLD} 百分位）: {threshold_value:.4f}")

    sound_frames = np.where(frame_energy > threshold_value)[0]
    print("检测到有声音的帧索引：", sound_frames)

    if sound_frames.size == 0:
        raise RuntimeError("未检测到任何高能量帧，无法进行聚类。")

    data = sound_frames.reshape(-1, 1)
    kmeans = KMeans(n_clusters=N_CLUSTERS, random_state=RANDOM_STATE)
    kmeans.fit(data)
    centers = np.sort(kmeans.cluster_centers_.flatten())

    for i in range(5):
        print(centers[i+1]-centers[i])

    times = centers * HOP_LENGTH / sr + TRIM_DURATION
    print("聚类中心时间点 (秒)：", times, centers)

    plt.figure(figsize=(16, 4))
    plt.plot(frame_energy, label="短时能量")
    plt.scatter(centers, np.zeros_like(centers), marker='*', s=100,
                label="击键聚类中心", zorder=3)
    plt.axhline(y=threshold_value, linestyle='--', label="阈值")
    plt.title("短时能量检测与聚类结果")
    plt.xlabel("帧索引")
    plt.ylabel("能量")
    plt.legend()
    plt.tight_layout()
    plt.show(block=False)
    plt.waitforbuttonpress()
    plt.close()

    output_path = os.path.join(out_dir, 'times(sound).npy')
    np.save(output_path, times[:6])
    print(f"已保存检测结果到：{output_path}")


