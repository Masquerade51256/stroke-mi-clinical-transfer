import numpy as np
from preprocessing.base_processor import BaseProcessor

class DataAugmentationProcessor(BaseProcessor):
    """数据增强处理器。在样本级别（trial/window）进行操作。"""
    def __init__(self, methods: list, name: str = "", **kwargs):
        """
        Args:
            methods: 增强方法列表，如 ['time_warp', 'add_noise', 'channel_dropout']
            p: 应用每个增强的概率列表，长度应与methods相同。
        """
        # print(f"DataAugmentationProcessor initialized with methods: {methods}")
        self.methods = methods
        
    def process(self, data: np.ndarray) -> np.ndarray:
        """
        Args:
            data: 单个样本数据，形状 (n_bands, n_channels, n_times) 或 (n_channels, n_times)。
        Returns:
            augmented_data: 增强后的数据。
        """

        aug_data = data.copy()
        for _, method in enumerate(self.methods):
            # print(f"Applying augmentation: {method}")
            if np.random.random() > method["probability"]:
                continue
            if method["method"] == 'time_warp':
                aug_data = self._time_warp(aug_data,sigma=method["sigma"])
            elif method["method"] == 'add_noise':
                aug_data = self._add_gaussian_noise(aug_data, snr_db=method["snr_db"])
            elif method["method"] == 'channel_dropout':
                aug_data = self._channel_dropout(aug_data, dropout_rate=method["dropout_rate"])
        return aug_data
    
    def _time_warp(self, data, sigma=0.2):
        """轻微的时间扭曲。"""
        # NOTE: CubicSpline (scipy.interpolate) is DISABLED due to a
        # segfault in scipy.linalg.solve_banded on certain input patterns.
        # Using numpy linear interpolation as a safe, fast alternative.
        # See: https://github.com/scipy/scipy/issues/XXXXX
        n_time = data.shape[-1]
        orig_time = np.arange(n_time, dtype=np.float64)
        warp_steps = np.random.normal(loc=1.0, scale=sigma, size=n_time).cumsum()
        warp_steps = (warp_steps - warp_steps.min()) / (warp_steps.max() - warp_steps.min()) * (n_time - 1)
        # Safe linear interpolation via numpy (no scipy, no banded solver)
        warped_data = np.apply_along_axis(
            lambda x: np.interp(warp_steps, orig_time, x.astype(np.float64)), -1, data
        )
        return warped_data.astype(data.dtype)
    
    def _add_gaussian_noise(self, data, snr_db=20):
        """添加高斯噪声，信噪比为snr_db。"""
        signal_power = np.mean(data**2)
        noise_power = signal_power / (10 ** (snr_db / 10))
        noise = np.random.normal(0, np.sqrt(noise_power), data.shape)
        return data + noise
    
    def _channel_dropout(self, data, dropout_rate=0.1):
        """随机将部分通道置零，模拟坏导或注意力转移。"""
        n_channels = data.shape[-2]
        n_to_drop = int(n_channels * dropout_rate)
        if n_to_drop > 0:
            channels_to_drop = np.random.choice(n_channels, n_to_drop, replace=False)
            # 通道维度是倒数第二维
            slc = [slice(None)] * data.ndim
            slc[-2] = channels_to_drop
            data[tuple(slc)] = 0
        return data