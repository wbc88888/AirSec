import os
import sys
import logging
from typing import Tuple
import numpy as np

import librosa
import noisereduce as nr
import soundfile as sf

# 配置日志输出格式
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def load_audio(file_path: str) -> Tuple[np.ndarray, int]:  # type: ignore
    """
    加载音频文件。

    参数:
        file_path (str): 音频文件路径

    返回:
        y (np.ndarray): 音频时域信号
        sr (int): 采样率

    异常:
        FileNotFoundError: 文件不存在
        RuntimeError: 加载失败
    """
    if not os.path.isfile(file_path):
        logging.error(f"未找到音频文件: {file_path}")
        raise FileNotFoundError(f"文件不存在: {file_path}")
    try:
        y, sr = librosa.load(file_path, sr=None)
        logging.info(f"已加载音频: {file_path} (采样率={sr}, 样本数={len(y)})")
        return y, sr
    except Exception as e:
        logging.exception("加载音频失败。")
        raise RuntimeError("音频加载错误") from e


def extract_noise_segment(y: np.ndarray, sr: int, duration: float) -> np.ndarray:
    """
    提取音频末尾一段静音/噪声作为降噪样本。

    参数:
        y (np.ndarray): 原始音频信号
        sr (int): 采样率
        duration (float): 噪声时长（秒）

    返回:
        np.ndarray: 噪声片段

    异常:
        ValueError: duration 为负或超过音频长度
    """
    if duration < 0:
        raise ValueError("噪声时长必须为非负数。")
    n_samples = int(sr * duration)
    if n_samples <= 0:
        logging.warning("噪声时长过短，返回空数组。")
        return y[0:0]
    if n_samples > len(y):
        raise ValueError("噪声时长超过音频总时长。")
    noise = y[-n_samples:]
    logging.info(f"提取噪声片段: 后 {duration} 秒 ({n_samples} 样本)")
    return noise


def reduce_noise_audio(
    y: np.ndarray,
    sr: int,
    noise_sample: np.ndarray
) -> np.ndarray:
    """
    使用噪声样本对音频进行降噪处理。

    参数:
        y (np.ndarray): 原始音频信号
        sr (int): 采样率
        noise_sample (np.ndarray): 噪声样本信号

    返回:
        np.ndarray: 降噪后音频信号
    """
    try:
        denoised = nr.reduce_noise(y=y, sr=sr, y_noise=noise_sample)
        logging.info("降噪处理完成。")
        return denoised
    except Exception:
        logging.exception("降噪处理失败。")
        raise


def save_audio(file_path: str, data: np.ndarray, sr: int) -> None:
    """
    将音频信号保存到文件。

    参数:
        file_path (str): 输出文件路径
        data (np.ndarray): 音频时域信号
        sr (int): 采样率

    异常:
        RuntimeError: 保存失败
    """
    try:
        sf.write(file_path, data, sr)
        logging.info(f"已保存降噪后音频: {file_path}")
    except Exception:
        logging.exception("保存音频文件失败。")
        raise RuntimeError("音频保存错误")


def main(
    data_dir: str,
    input_filename: str = 'temp_loud.wav',
    output_filename: str = 'temp_denoised.wav',
    noise_duration: float = 0.5
) -> None:
    """
    主函数：加载、降噪并保存音频。

    参数:
        data_dir (str): 数据目录
        input_filename (str): 输入文件名
        output_filename (str): 输出文件名
        noise_duration (float): 噪声时长（秒）
    """
    input_path = os.path.join(data_dir, input_filename)
    output_path = os.path.join(data_dir, output_filename)

    y, sr = load_audio(input_path)
    noise = extract_noise_segment(y, sr, noise_duration)
    denoised = reduce_noise_audio(y, sr, noise)
    save_audio(output_path, denoised, sr)

    logging.info("整体降噪流程完成。")
    print(f"降噪完成，结果保存在 {output_filename}")


if __name__ == '__main__':
    # 支持命令行指定数据目录，否则使用默认路径
    if len(sys.argv) > 1:
        DATA_DIR = sys.argv[1]
    else:
        DATA_DIR = 'data/1-111000/'
    main(DATA_DIR)
