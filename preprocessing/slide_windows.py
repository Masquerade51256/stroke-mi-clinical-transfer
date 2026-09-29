import numpy as np
from preprocessing.base_processor import BaseProcessor

class SlidingWindowSegmenter(BaseProcessor):
    """滑动窗口分割处理器。"""
    def __init__(self, window_len_samples: int, window_stride_samples: int):
        self.window_len_samples = window_len_samples
        self.window_stride_samples = window_stride_samples

    def process(self, data: np.ndarray, labels: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Applies a sliding window to each session's data and generates corresponding window labels.
        """
        windowed_data_list = []
        windowed_labels_list = []
        all_group_ids = []

        n_sessions, n_bands, n_channels, n_timepoints = data.shape

        for sess_idx in range(n_sessions):
            session_data = data[sess_idx]  # Shape: (n_bands, n_channels, n_timepoints)
            session_label = labels[sess_idx]

            # Calculate the number of windows for this session
            n_windows = 1 + (n_timepoints - self.window_len_samples) // self.window_stride_samples

            for win_idx in range(n_windows):
                start = win_idx * self.window_stride_samples
                end = start + self.window_len_samples
                window = session_data[:, :, start:end]  # Shape: (n_bands, n_channels, window_len_samples)

                windowed_data_list.append(window)
                windowed_labels_list.append(session_label)

            all_group_ids.extend([sess_idx] * n_windows)

        # Stack all windows
        all_windowed_data = np.stack(windowed_data_list, axis=0).astype(np.float32)  # (n_total_windows, n_bands, n_channels, window_len)
        all_windowed_labels = np.array(windowed_labels_list, dtype=np.int64)
        all_group_ids = np.array(all_group_ids, dtype=np.int64)
        return all_windowed_data, all_windowed_labels, all_group_ids
    
