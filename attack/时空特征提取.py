import os
import numpy as np
from fastdtw import fastdtw
from sklearn.preprocessing import MinMaxScaler
import random

# --------------------- 全局配置 ---------------------
DATA_PATH = 'intermediate variable/1-111000/'
WAVE_THRESHOLD = 1.4    # DTW 距离阈值
FLIGHT_THRESHOLD = 0.1  # 飞行间隔阈值

# --------------------- 辅助函数 ---------------------

def load_numpy_array(path: str, name: str) -> np.ndarray:
    """
    从指定路径加载 .npy 文件，确保文件存在和数据正确
    :param path: 文件夹路径
    :param name: 文件名
    :return: numpy 数组
    """
    full_path = os.path.join(path, name)
    if not os.path.isfile(full_path):
        raise FileNotFoundError(f"未找到文件: {full_path}")
    arr = np.load(full_path, allow_pickle=True)
    if not isinstance(arr, np.ndarray):
        raise ValueError(f"文件 {full_path} 未返回 numpy 数组")
    return arr


def normalize_waveform(wave: np.ndarray) -> np.ndarray:
    """
    对单条波形进行 Min-Max 归一化
    :param wave: 一维数组
    :return: 归一化后的一维数组
    """
    if wave.ndim != 1:
        raise ValueError("输入波形必须为一维数组")
    scaler = MinMaxScaler(feature_range=(0, 1))
    return scaler.fit_transform(wave.reshape(-1, 1)).flatten()


def compute_dtw_matrix(waves: np.ndarray) -> np.ndarray:
    """
    计算波形之间的 DTW 距离矩阵
    :param waves: 波形列表，每项一维数组
    :return: 对称距离矩阵
    """
    n = len(waves)
    D = np.zeros((n, n), dtype=float)
    for i in range(n):
        w_i = normalize_waveform(waves[i])
        for j in range(i, n):
            w_j = normalize_waveform(waves[j])
            dist, _ = fastdtw(w_i, w_j, dist=lambda x, y: abs(x - y))
            D[i, j] = D[j, i] = dist
    return D


def compute_interval_matrix(intervals: np.ndarray) -> np.ndarray:
    """
    计算飞行间隔相似度矩阵
    :param intervals: 一维飞行间隔数组
    :return: 对称相似度矩阵
    """
    n = len(intervals)
    T = np.zeros((n, n), dtype=float)
    for i in range(n):
        for j in range(i, n):
            a, b = intervals[i], intervals[j]
            if a == 0 and b == 0:
                dist = 0.0
            else:
                dist = abs(a - b) / max(a, b)
            T[i, j] = T[j, i] = dist
    return T


def save_array(path: str, name: str, arr: np.ndarray):
    """
    保存 numpy 数组到 .npy 文件
    """
    full_path = os.path.join(path, name)
    np.save(full_path, arr)

# --------------------- 主流程 ---------------------

# 1. 载入数据并校验
waveforms = load_numpy_array(DATA_PATH, 'waveforms.npy')
intervals = load_numpy_array(DATA_PATH, 'intervals.npy')
if waveforms.ndim != 1 or intervals.ndim != 1:
    raise ValueError("加载的数据必须为一维数组列表")

# 2. 计算特征矩阵
D = compute_dtw_matrix(waveforms)
print('空间特征矩阵：\n', D)
T = compute_interval_matrix(intervals)
print('时间特征矩阵：\n', T)

# 3. 提取空间特征
n_wave = len(waveforms)
s_vector = np.full(n_wave, -1, dtype=int)

# 容错标记示例
# s_vector[2] = -2

cur_class = 0
cur_class2 = 10
for i in range(n_wave):
    # print("空间特征：", s_vector)
    # 丢弃情况
    if s_vector[i] == -2:
        continue
    # 粘合情况判断
    if s_vector[i] > 10:
        if i + 1 < n_wave and D[i, i+1] == 0:
            s_vector[i+1] = cur_class2
        continue
    if i + 1 < n_wave and D[i, i+1] == 0:
        cur_class2 += 1
        s_vector[i] = s_vector[i+1] = cur_class2
        continue
    # 正常情况染色
    if s_vector[i] == -1:
        cur_class += 1
        s_vector[i] = cur_class
    for j in range(i+1, n_wave):
        if s_vector[j] > 10 or s_vector[j] == -2:
            continue
        if D[i, j] < WAVE_THRESHOLD:
            same_group = (s_vector[j] == -1)
            # 验证是否可加入当前分类
            valid = True
            members = np.where(s_vector == s_vector[i])[0]
            for k in members:
                if D[j, k] >= WAVE_THRESHOLD:
                    valid = False
                    break
            if not valid:
                continue
            # 已染色则选最近的，否则直接染色
            if same_group or D[j, i] == np.min(D[j][D[j] != 0]):
                s_vector[j] = s_vector[i]

# 4. 提取时间特征
n_int = len(intervals)
t_vector = np.full(n_int, -1, dtype=int)
cur_tclass = 0
for i in range(n_int):
    # print("时间特征：", t_vector)
    if t_vector[i] == -1:
        cur_tclass += 1
        t_vector[i] = cur_tclass
    for j in range(i+1, n_int):
        if T[i, j] < FLIGHT_THRESHOLD:
            valid = True
            members = np.where(t_vector == t_vector[i])[0]
            for k in members:
                if T[j, k] >= FLIGHT_THRESHOLD:
                    valid = False
                    break
            if valid and (t_vector[j] == -1 or T[j, i] == np.min(T[j][T[j] != 0])):
                t_vector[j] = t_vector[i]

# 5. 按平均间隔排序并分组
unique_groups = np.unique(t_vector)
g_means = [intervals[t_vector == g].mean() for g in unique_groups]
# 排序映射: 原组 -> 新序号
sorted_idx = np.argsort(np.argsort(np.array(g_means))) + 1
ft_vector = sorted_idx[np.array(t_vector) - 1]

print("原时间特征：", ft_vector)

# 6. 根据粘合情况调整时间特征
# for i in range(cur_class2 - 10):
#     idx_min = np.abs(s_vector - 11 - i).argmin()
#     offset = ft_vector[idx_min] - 1
#     ft_vector = np.where(ft_vector <= ft_vector[idx_min], 1, ft_vector - offset)

# 7. 输出与保存
print("空间特征：", s_vector)
print("时间特征：", ft_vector)

save_array(DATA_PATH, 'sprop.npy', s_vector)
save_array(DATA_PATH, 'tprop.npy', ft_vector)  # 保存时间属性
