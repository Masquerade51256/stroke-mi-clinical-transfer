# preprocessing/pipeline.py
from abc import ABC, abstractmethod
import numpy as np
from typing import Any, Dict, Optional

class BaseProcessor(ABC):
    """所有预处理步骤的抽象基类。"""
    @abstractmethod
    def process(self, data: np.ndarray, **kwargs) -> np.ndarray:
        """处理数据并返回处理后的数据。"""
        pass
    def __call__(self, data: np.ndarray, **kwargs) -> np.ndarray:
        return self.process(data, **kwargs)

class ProcessingPipeline:
    """
    A sequential pipeline of processors that can be applied to data.
    """
    
    def __init__(self, processors: Optional[list] = None):
        """
        Args:
            processors: List of BaseProcessor instances, in execution order.
        """
        self.processors = processors or []
    
    def add_processor(self, processor: BaseProcessor) -> None:
        """Add a processor to the end of the pipeline."""
        self.processors.append(processor)
    
    def insert_processor(self, index: int, processor: BaseProcessor) -> None:
        """Insert a processor at the specified position."""
        self.processors.insert(index, processor)
    
    def process(self, data: np.ndarray, **kwargs) -> np.ndarray:
        """
        Sequentially apply all processors to the input data.
        
        Args:
            data: Input data array.
            **kwargs: Additional parameters passed to each processor.
            
        Returns:
            Processed data after applying all processors.
        """
        result = data.copy() if data.flags.writeable else data
        
        for processor in self.processors:
            # Pass both processor config and runtime kwargs
            result = processor(result, **kwargs)
            
        return result
    
    def __call__(self, data: np.ndarray, **kwargs) -> np.ndarray:
        return self.process(data, **kwargs)
    
    def __len__(self) -> int:
        return len(self.processors)
    
    def __repr__(self) -> str:
        proc_names = [p.name for p in self.processors]
        return f"ProcessingPipeline({proc_names})"



# from preprocessing.bandpass import bandpass_filter




# class ExponentialMovingStandardizeProcessor(BaseProcessor):
#     def __init__(self, init_block_size: int, num_channels: int, factor_new: float = 0.001):
#         self.init_block_size = init_block_size
#         self.num_channels = num_channels
#         self.factor_new = factor_new
#         # 可以在这里导入_pandas_和_numpy_
    
#     def process(self, data: np.ndarray) -> np.ndarray:
#         import pandas as pd
#         import numpy as np
        
#         # 确保数据形状是 (n_channels, n_times)
#         if data.shape[0] != self.num_channels:
#             # 如果输入是 (n_times, n_channels)，则转置
#             if data.shape[1] == self.num_channels:
#                 data = data.T
#             else:
#                 # 尝试自动推断
#                 if data.shape[0] > data.shape[1]:
#                     data = data.T
        
#         df = pd.DataFrame(data.T)
#         meaned = df.ewm(alpha=self.factor_new).mean()
#         demeaned = df - meaned
#         squared = demeaned * demeaned
#         square_ewmed = squared.ewm(alpha=self.factor_new).mean()
#         standardized = demeaned / np.maximum(1e-4, np.sqrt(np.array(square_ewmed)))
#         standardized = np.array(standardized)  # 形状: (n_times, n_channels)
        
#         if self.init_block_size is not None:
#             # 计算初始块的标准化
#             i_time_axis = 1  # 原始数据中时间轴是第1维
#             # 从原始数据中取初始块
#             init_data = data[..., :self.init_block_size]
#             # 沿着时间轴计算均值和标准差
#             init_mean = np.mean(init_data, axis=i_time_axis, keepdims=True)
#             init_std = np.std(init_data, axis=i_time_axis, keepdims=True)
#             init_block_standardized = (init_data - init_mean) / np.maximum(1e-4, init_std)
#             # 注意: init_block_standardized 形状是 (n_channels, init_block_size)
#             # 而 standardized[:init_block_size, :] 需要形状 (init_block_size, n_channels)
#             # 所以需要转置
#             standardized[:self.init_block_size, :] = init_block_standardized.T
        
#         # 返回转置后的结果，使形状变回 (n_channels, n_times)
#         return standardized.T


# class ChannelSelector(BaseProcessor):
#     """通道选择处理器。"""
#     def __init__(self, channels_to_select):
#         self.channels_to_select = channels_to_select

#     def process(self, data: np.ndarray) -> np.ndarray:
#         return data[..., self.channels_to_select, :]
    

# class ResampleProcessor(BaseProcessor):
#     """重采样处理器。可进行上采样或下采样。"""
#     def __init__(self, target_sfreq: float, original_sfreq: float = None):
#         """
#         Args:
#             target_sfreq: 目标采样率 (Hz)。
#             original_sfreq: 原始采样率。如果为None，则期望在process时通过**kwargs传入。
#         """
#         self.target_sfreq = target_sfreq
#         self.original_sfreq = original_sfreq

#     def process(self, data: np.ndarray, sfreq: float = None) -> np.ndarray:
#         """
#         Args:
#             data: EEG数据，形状 (..., n_channels, n_times)。最后一维是时间轴。
#             sfreq: 数据的原始采样率。如果初始化时未提供，则必须传入。
#         Returns:
#             resampled_data: 重采样后的数据。
#         """
#         original_sfreq = self.original_sfreq or sfreq
#         if original_sfreq is None:
#             raise ValueError("Original sampling rate (sfreq) must be provided.")

#         if original_sfreq == self.target_sfreq:
#             return data

#         # 计算重采样比例
#         ratio = self.target_sfreq / original_sfreq
#         n_samples_original = data.shape[-1]
#         n_samples_new = int(np.round(n_samples_original * ratio))

#         # 使用scipy.signal.resample进行重采样（基于FFT，质量较高）
#         # 注意：可以沿时间轴（最后一个轴）应用。对于多通道数据，需要循环或向量化操作。
#         # 这里使用列表推导式保持清晰，对于大数据可优化为向量化操作。
#         resampled = np.apply_along_axis(
#             lambda x: signal.resample(x, n_samples_new),
#             axis=-1,
#             arr=data
#         )
#         return resampled.astype(data.dtype)  # 保持原始数据类型


# class DataAugmentationProcessor(BaseProcessor):
#     """数据增强处理器。在样本级别（trial/window）进行操作。"""
#     def __init__(self, methods: list, p: float = 0.5):
#         """
#         Args:
#             methods: 增强方法列表，如 [‘time_warp’, ‘add_noise’, ‘channel_dropout’]。
#             p: 每个样本应用增强的概率。
#         """
#         self.methods = methods
#         self.p = p
        
#     def process(self, data: np.ndarray) -> np.ndarray:
#         """
#         Args:
#             data: 单个样本数据，形状 (n_bands, n_channels, n_times) 或 (n_channels, n_times)。
#         Returns:
#             augmented_data: 增强后的数据。
#         """
#         if np.random.random() > self.p:
#             return data
            
#         aug_data = data.copy()
#         for method in self.methods:
#             if method == 'time_warp':
#                 aug_data = self._time_warp(aug_data)
#             elif method == 'add_noise':
#                 aug_data = self._add_gaussian_noise(aug_data)
#             elif method == 'channel_dropout':
#                 aug_data = self._channel_dropout(aug_data)
#             # ... 其他增强方法 ...
#         return aug_data
    
#     def _time_warp(self, data, sigma=0.2):
#         """轻微的时间扭曲。"""
#         from scipy.interpolate import CubicSpline
#         n_time = data.shape[-1]
#         orig_time = np.arange(n_time)
#         warp_steps = np.random.normal(loc=1.0, scale=sigma, size=n_time).cumsum()
#         warp_steps = (warp_steps - warp_steps.min()) / (warp_steps.max() - warp_steps.min()) * (n_time - 1)
#         warped_data = np.apply_along_axis(lambda x: CubicSpline(orig_time, x)(warp_steps), -1, data)
#         return warped_data
    
#     def _add_gaussian_noise(self, data, snr_db=20):
#         """添加高斯噪声，信噪比为snr_db。"""
#         signal_power = np.mean(data**2)
#         noise_power = signal_power / (10 ** (snr_db / 10))
#         noise = np.random.normal(0, np.sqrt(noise_power), data.shape)
#         return data + noise
    
#     def _channel_dropout(self, data, dropout_rate=0.1):
#         """随机将部分通道置零，模拟坏导或注意力转移。"""
#         n_channels = data.shape[-2]
#         n_to_drop = int(n_channels * dropout_rate)
#         if n_to_drop > 0:
#             channels_to_drop = np.random.choice(n_channels, n_to_drop, replace=False)
#             # 假设通道维度是倒数第二维
#             slc = [slice(None)] * data.ndim
#             slc[-2] = channels_to_drop
#             data[tuple(slc)] = 0
#         return data


# class ArtifactRemovalProcessor(BaseProcessor):
#     """伪影去除处理器。提供多种方法可选。"""
#     def __init__(self, method: str = 'ica', **kwargs):
#         """
#         Args:
#             method: 伪影去除方法，可选 ‘ica’（独立成分分析）, ‘regression’, ‘ads’（自适应滤波）等。
#             kwargs: 对应方法的参数，如 n_components for ICA。
#         """
#         self.method = method
#         self.params = kwargs
        
#     def process(self, data: np.ndarray, info=None) -> np.ndarray:
#         """
#         Args:
#             data: EEG数据，形状 (n_channels, n_times) 或 (n_trials, n_channels, n_times)。
#             info: 可选的mne.Info对象，包含通道位置等信息，某些方法（如ICA）需要。
#         Returns:
#             cleaned_data: 去除伪影后的数据。
#         """
#         if self.method == 'ica':
#             return self._remove_by_ica(data, info)
#         elif self.method == 'regression':
#             return self._remove_by_regression(data)
#         # ... 其他方法 ...
#         else:
#             raise ValueError(f“Unsupported artifact removal method: {self.method}”)

#     def _remove_by_ica(self, data, info):
#         """使用MNE库进行ICA去伪影。"""
#         # 注意：ICA通常需要在连续数据或长片段上进行，对计算资源要求较高。
#         # 这里仅为示例框架，实际实现需考虑数据格式和性能。
#         if info is None:
#             # 创建一个简易的info对象，如果未提供
#             info = mne.create_info(ch_names=[f‘CH{i}’ for i in range(data.shape[-2])], sfreq=250, ch_types=‘eeg’)
#         raw = mne.io.RawArray(data, info)
#         ica = mne.preprocessing.ICA(n_components=self.params.get(‘n_components’, 0.95), random_state=97)
#         ica.fit(raw)
#         # 这里需要定义或自动检测伪影成分。实际操作更复杂，可能需EOG/ECG通道或模板匹配。
#         # 例如: ica.exclude = [0, 1] # 假设前两个成分为伪影
#         ica.apply(raw)
#         return raw.get_data()
    
#     def _remove_by_regression(self, data):
#         """使用回归方法去除伪影（例如，利用EOG参考通道）。"""
#         # 简化示例：假设最后两个通道是EOG参考通道
#         eeg_data = data[..., :-2, :]
#         eog_data = data[..., -2:, :]
#         # 为每个EEG通道对EOG通道进行回归并减去
#         # ... 实现回归计算 ...
#         return cleaned_eeg
    
