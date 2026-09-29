# preprocessing/pipeline.py
import numpy as np
from scipy import signal
from preprocessing.base_processor import BaseProcessor

class ResampleProcessor(BaseProcessor):
    """重采样处理器。可进行上采样或下采样。"""
    def __init__(self, target_sfreq: float, original_sfreq: float = None):
        """
        Args:
            target_sfreq: 目标采样率 (Hz)。
            original_sfreq: 原始采样率。如果为None，则期望在process时通过**kwargs传入。
        """
        self.target_sfreq = target_sfreq
        self.original_sfreq = original_sfreq

    def process(self, data: np.ndarray, sfreq: float = None) -> np.ndarray:
        """
        Args:
            data: EEG数据，形状 (..., n_channels, n_times)。最后一维是时间轴。
            sfreq: 数据的原始采样率。如果初始化时未提供，则必须传入。
        Returns:
            resampled_data: 重采样后的数据。
        """
        original_sfreq = self.original_sfreq or sfreq
        if original_sfreq is None:
            raise ValueError("Original sampling rate (sfreq) must be provided.")

        if original_sfreq == self.target_sfreq:
            return data

        # 计算重采样比例
        ratio = self.target_sfreq / original_sfreq
        n_samples_original = data.shape[-1]
        n_samples_new = int(np.round(n_samples_original * ratio))

        # 使用scipy.signal.resample进行重采样（基于FFT，质量较高）
        # 注意：可以沿时间轴（最后一个轴）应用。对于多通道数据，需要循环或向量化操作。
        # 这里使用列表推导式保持清晰，对于大数据可优化为向量化操作。
        resampled = np.apply_along_axis(
            lambda x: signal.resample(x, n_samples_new),
            axis=-1,
            arr=data
        )
        return resampled.astype(data.dtype)  # 保持原始数据类型