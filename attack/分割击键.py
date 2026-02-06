import os
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt


# ———— 常量配置 ————
DATA_DIR = Path('intermediate variable/1-111000')
CSI_FILE = 'csi.npy'
TS_FILE = 'time_stamps.npy'
SOUND_TS_FILE = 'times(sound).npy'

CSI_FIRST_TS = 0.30035
SOUND_FIRST_TS = 0.00014
WINDOW_SIZE = 17          # 窗口大小
USE_ENERGY = True         # False 时计算 MAD  True 时计算能量
THRESHOLD_FACTOR = 0.015  # 用于背景阈值计算
BG_MARGIN = 1.0           # 背景阈值窗口大小
MASK = 0                  # 背景阈值是否限定窗口


# ———— 工具函数 ————
def load_numpy_array(file_path: Path) -> np.ndarray:
    """
    从指定文件加载 numpy 数组，确保文件存在且非空。
    """
    if not file_path.is_file():
        raise FileNotFoundError(f"未找到文件: {file_path}")
    arr = np.load(str(file_path))
    if arr.size == 0:
        raise ValueError(f"加载的数组为空: {file_path}")
    return arr


def compute_mad(data: np.ndarray, window: int) -> np.ndarray:
    """
    计算滑动窗口平均绝对离差（MAD）。
    data: 1D 数组；window: 正奇数。
    返回同长度的 MAD 数组。
    """
    if window < 1 or window % 2 == 0:
        raise ValueError("window_size 必须是大于 0 的奇数")
    half = window // 2
    padded = np.pad(data, (half, half), mode='edge')
    mad = np.zeros_like(data, dtype=float)

    for i in range(len(data)):
        window_slice = padded[i : i + window]
        mu = window_slice.mean()
        mad[i] = np.abs(window_slice - mu).mean()
    return mad


def find_nearest_indices(reference: np.ndarray, targets: np.ndarray) -> np.ndarray:
    """
    对于 targets 中的每个值，找到在 reference 数组中最接近的索引。
    """
    if reference.ndim != 1 or targets.ndim != 1:
        raise ValueError("reference 和 targets 必须为一维数组")
    return np.array([int(np.argmin(np.abs(reference - t))) for t in targets])


def segment_waveforms(
    data: np.ndarray, centers: np.ndarray, metric: np.ndarray, bg_threshold: float
) -> tuple[list[int], list[int]]:
    """
    根据中心点、检测度量及背景阈值，搜索每个击键的起点和终点。
    """
    starts, ends = [], []
    N = len(data)

    for c in centers:
        if not (0 <= c < N):
            raise IndexError(f"中心点索引越界: {c}")

        if metric[c] > bg_threshold:
            s = c
            while s > 0 and metric[s] > bg_threshold:
                s -= 1
            e = c
            while e < N - 1 and metric[e] > bg_threshold:
                e += 1
        else:
            left = c
            while left > 0 and metric[left] <= bg_threshold:
                left -= 1
            right = c
            while right < N - 1 and metric[right] <= bg_threshold:
                right += 1

            if (c - left) > (right - c):
                s, e = right - 1, right
                while e < N - 1 and metric[e] > bg_threshold:
                    e += 1
            else:
                s, e = left, left + 1
                while s > 0 and metric[s] > bg_threshold:
                    s -= 1

        starts.append(s)
        ends.append(e)

    return starts, ends


# ———— 主流程 ————
def main():
    # 1. 加载数据
    csi = load_numpy_array(DATA_DIR / CSI_FILE)
    time_stamps = load_numpy_array(DATA_DIR / TS_FILE)
    sound_times = load_numpy_array(DATA_DIR / SOUND_TS_FILE)

    # 2. 时间对齐：将声音时间轴平移至 CSI 时间基准
    aligned_sound = sound_times + (SOUND_FIRST_TS - CSI_FIRST_TS)

    # 3. 找到每次击键在 CSI 数据中的最接近索引
    centers = find_nearest_indices(time_stamps, aligned_sound)
    # print(centers)
    # centers = np.array([240,335,530,700,950,1100])

    # 4. 可视化：原始 CSI 波形与中心点
    plt.figure(figsize=(16, 5))
    plt.plot(csi, linewidth=1.5, label="CSI Amplitude")
    plt.scatter(centers, csi[centers], c='red', marker='*', label="Centers")
    plt.xlabel('Packet Index')
    plt.ylabel('Amplitude')
    plt.title('Raw CSI and Keystroke Centers')
    plt.grid(True)
    plt.tight_layout()
    plt.show(block=False)
    plt.waitforbuttonpress()  # 回车 / 鼠标点击 都会继续
    plt.close()

    # 5. 计算度量：能量或 MAD
    if USE_ENERGY:
        metric = np.convolve(csi**2, np.ones(WINDOW_SIZE)/WINDOW_SIZE, mode='same')
    else:
        metric = compute_mad(csi, WINDOW_SIZE)

    # 6. 动态背景阈值：取第一个与最后一个击键时刻前后各 BG_MARGIN 秒，且不超出数据边界
    raw_start = aligned_sound[0] - BG_MARGIN
    raw_end   = aligned_sound[-1] + BG_MARGIN
    # 边界约束
    start_time = max(raw_start, time_stamps[0])
    end_time   = min(raw_end, time_stamps[-1])

    mask = (time_stamps >= start_time) & (time_stamps <= end_time)

    if not np.any(mask):
        raise ValueError(f"背景阈值计算范围内没有数据: [{start_time}, {end_time}]")
    if MASK: window_metric = metric[mask] 
    else: window_metric = metric

    bg_median = np.median(window_metric)
    bg_max    = np.max(window_metric)
    background_threshold = bg_median + (bg_max - bg_median) * THRESHOLD_FACTOR

    # 7. 分割：找到每个击键的起止
    start_pts, end_pts = segment_waveforms(csi, centers, metric, background_threshold)

    print("起点：", start_pts)
    print("终点：", end_pts)

    # 8. 提取波形段与计算击键间隔
    waveforms = [csi[s:e] for s, e in zip(start_pts, end_pts)]
    intervals = list(np.diff(sound_times))  # 基于原声音时间戳的中心间隔

    print(f"击键波形数量: {len(waveforms)}")
    print("间隔 (s):", intervals)

    # 9. 保存结果
    np.save(DATA_DIR / 'waveforms.npy', np.array(waveforms, dtype=object))
    np.save(DATA_DIR / 'intervals.npy', intervals)

    # 10. 可视化分割效果
    plt.figure(figsize=(16, 5))
    plt.axhline(y=background_threshold, color='r', linestyle='--', label='Background Threshold')
    plt.plot(metric, alpha=0.6, label='Metric')
    for c in centers:
        plt.axvline(x=c, color='blue', linestyle='--')
    plt.xlabel('Index')
    plt.ylabel('Metric Value')
    plt.title('Segmentation on Metric')
    plt.legend()
    plt.show(block=False)
    plt.waitforbuttonpress()  # 回车 / 鼠标点击 都会继续
    plt.close()

    plt.figure(figsize=(16, 5))
    plt.plot(csi, linewidth=1.5, label="CSI Amplitude")
    plt.scatter(start_pts, csi[start_pts], c='green', marker='*', label="Starts")
    plt.scatter(end_pts,   csi[end_pts],   c='magenta', marker='*', label="Ends")
    plt.xlabel('Packet Index')
    plt.ylabel('Amplitude')
    plt.title('Final Keystroke Segments')
    plt.grid(True)
    plt.tight_layout()
    plt.show(block=False)
    plt.waitforbuttonpress()  # 回车 / 鼠标点击 都会继续
    plt.close()

if __name__ == '__main__':
    main()

