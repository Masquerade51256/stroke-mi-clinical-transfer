
from preprocessing.base_processor import BaseProcessor
import numpy as np
import scipy.signal


def bandpass_filter(data, low_freq, high_freq, sfreq, order=4):
        """
        Bandpass filter using a Butterworth filter.
        Args:
            data: Input data, shape (..., n_timepoints). This function will filter along the last axis (time axis).
            low_freq: Low cutoff frequency of the passband.
            high_freq: High cutoff frequency of the passband.
            sfreq: Sampling frequency.
            order: Order of the filter.
        Returns:
            filtered_data: Filtered data.
        """
        # Design Butterworth bandpass filter
        nyquist = sfreq / 2.0
        low = low_freq / nyquist
        high = high_freq / nyquist
        b, a = scipy.signal.butter(order, [low, high], btype='band')
        
        # Apply filter along the last axis (axis=-1)
        filtered_data = scipy.signal.filtfilt(b, a, data, axis=-1)
        return filtered_data

class FilterBankProcessor(BaseProcessor):
    def __init__(self, filter_banks, sample_rate):
        self.filter_banks = filter_banks
        self.sample_rate = sample_rate

    def process(self, data: np.ndarray) -> np.ndarray:
        """
        Args:
            data: Shape (n_trials, n_channels, n_times) OR (n_channels, n_times)
        Returns:
            multi_band_data: Shape (..., n_bands, n_channels, n_times)
        """
        # 处理单一样本（无trials维度）的情况
        if data.ndim == 2:
            data = np.expand_dims(data, axis=0)
            has_trial_dim = False
        else:
            has_trial_dim = True

        n_trials, n_channels, n_times = data.shape
        band_data_list = []
        
        for l_freq, h_freq in self.filter_banks:
            filtered_data = np.zeros_like(data)
            for t in range(n_trials):
                # 调用统一的滤波函数
                filtered_data[t] = bandpass_filter(data[t], low_freq=l_freq, high_freq=h_freq, sfreq=self.sample_rate)
            # 逐通道标准化
            # mean = filtered_data.mean(axis=-1, keepdims=True)
            # std = filtered_data.std(axis=-1, keepdims=True)
            # std[std < 1e-8] = 1.0
            # normalized_band_data = (filtered_data - mean) / std
            # 增加频带维度
            # normalized_band_data = normalized_band_data[:, np.newaxis, :, :]
            filtered_data = filtered_data[:, np.newaxis, :, :]
            band_data_list.append(filtered_data)
        
        multi_band_data = np.concatenate(band_data_list, axis=1)  # (n_trials, n_bands, n_channels, n_times)
        
        if not has_trial_dim:
            multi_band_data = np.squeeze(multi_band_data, axis=0) # (n_bands, n_channels, n_times)
        return multi_band_data