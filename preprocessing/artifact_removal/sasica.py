"""
SASICA-like artifact removal processor.

Based on Thwe et al. (2026):
"Integrating EEG Artifact Removal with Deep Learning for Accurate Motor Imagery
Classification in Acute Stroke"

SASICA (Semi-Automated Selection of Independent Components for Artifact Correction)
is a semi-automatic method that selects artifact ICs using multiple criteria:
- Correlation with EOG channels
- Power spectrum patterns
- Signal variance
- Kurtosis
"""

from typing import List, Optional, Tuple
import numpy as np
from scipy import signal as sp_signal
from scipy.stats import kurtosis, pearsonr
import mne

from preprocessing.artifact_removal.base import BaseArtifactRemovalProcessor


class SASICAProcessor(BaseArtifactRemovalProcessor):
    """
    SASICA-like artifact removal using multiple heuristics on ICA components.
    
    This processor decomposes the signal with ICA and flags components as artifacts
    if they exceed user-defined thresholds on any of the following metrics:
    - EOG correlation
    - Temporal variance
    - Temporal kurtosis
    - Power spectrum ratio (high-frequency muscle or low-frequency eye dominance)
    """
    
    def __init__(self, name: str,
                 eog_channels: Optional[List[str]] = None,
                 eog_corr_threshold: float = 0.6,
                 variance_threshold: float = 3.0,
                 kurtosis_threshold: float = 3.0,
                 muscle_freq_range: Tuple[float, float] = (30.0, 100.0),
                 muscle_psd_ratio_threshold: float = 0.3,
                 eye_freq_range: Tuple[float, float] = (1.0, 7.0),
                 eye_psd_ratio_threshold: float = 0.4,
                 highpass_freq: Optional[float] = 1.0,
                 apply_car: bool = True,
                 **kwargs):
        """
        Args:
            name: Processor instance name.
            eog_channels: List of EOG channel names. If None, auto-detects EOG types.
            eog_corr_threshold: Pearson correlation threshold with EOG channels.
            variance_threshold: Z-score threshold for component temporal variance.
            kurtosis_threshold: Z-score threshold for component temporal kurtosis.
            muscle_freq_range: Frequency band for muscle artifact detection (Hz).
            muscle_psd_ratio_threshold: Threshold for muscle PSD ratio.
            eye_freq_range: Frequency band for eye artifact detection (Hz).
            eye_psd_ratio_threshold: Threshold for eye PSD ratio.
            highpass_freq: High-pass filter cutoff for ICA fitting (Hz). Set to None to disable.
            apply_car: Whether to apply CAR before ICA. Default True.
            **kwargs: Passed to BaseArtifactRemovalProcessor.
        """
        super().__init__(name=name, **kwargs)
        self.highpass_freq = highpass_freq
        self.eog_channels = eog_channels
        self.eog_corr_threshold = eog_corr_threshold
        self.variance_threshold = variance_threshold
        self.kurtosis_threshold = kurtosis_threshold
        self.muscle_freq_range = muscle_freq_range
        self.muscle_psd_ratio_threshold = muscle_psd_ratio_threshold
        self.eye_freq_range = eye_freq_range
        self.eye_psd_ratio_threshold = eye_psd_ratio_threshold
        self.apply_car = apply_car
    
    def _get_eog_channels(self, inst) -> List[str]:
        """Get EOG channel names from the MNE instance."""
        if self.eog_channels is not None:
            return [ch for ch in self.eog_channels if ch in inst.ch_names]
        return [ch for ch in inst.ch_names if inst.get_channel_types(picks=[ch])[0] == 'eog']
    
    @staticmethod
    def _compute_psd_ratio(psd: np.ndarray, freqs: np.ndarray,
                           band: Tuple[float, float]) -> float:
        """Compute the ratio of power in a specific band to total power."""
        band_mask = (freqs >= band[0]) & (freqs <= band[1])
        total_power = np.sum(psd)
        if total_power == 0:
            return 0.0
        band_power = np.sum(psd[band_mask])
        return float(band_power / total_power)
    
    def process(self, data: np.ndarray, **kwargs) -> np.ndarray:
        """
        Remove artifacts using SASICA-like multi-criteria thresholds.
        
        Args:
            data: EEG data, shape (n_channels, n_times) or (n_trials, n_channels, n_times).
        
        Returns:
            Cleaned EEG data with the same shape as input.
        """
        original_shape = data.shape
        inst = self._array_to_mne(data)
        inst_for_ica = inst.copy()
        
        # Step 0: Optional high-pass filter for ICA fitting
        if self.highpass_freq is not None:
            inst_for_ica.filter(l_freq=self.highpass_freq, h_freq=None, verbose='WARNING')
        
        # Step 1: Apply CAR
        if self.apply_car:
            if isinstance(inst_for_ica, mne.Epochs):
                inst_for_ica.set_eeg_reference('average', projection=True, verbose='WARNING')
                inst_for_ica.apply_proj()
            else:
                inst_for_ica.set_eeg_reference('average', verbose='WARNING')
            if isinstance(inst, mne.Epochs):
                inst.set_eeg_reference('average', projection=True, verbose='WARNING')
                inst.apply_proj()
            else:
                inst.set_eeg_reference('average', verbose='WARNING')
        
        # Step 2: Fit ICA on filtered copy
        ica = self._fit_ica(inst_for_ica)
        
        # Step 3: Get IC time courses
        # For epochs, get_sources returns (n_epochs, n_components, n_times)
        # For raw, get_sources returns (n_components, n_times)
        sources = ica.get_sources(inst_for_ica).get_data()
        
        # Step 4: Detect artifacts
        exclude = set()
        n_components = sources.shape[-2] if sources.ndim == 3 else sources.shape[0]
        
        # EOG correlation
        eog_ch_names = self._get_eog_channels(inst)
        if eog_ch_names:
            eog_data = inst.copy().pick(eog_ch_names).get_data()
            if eog_data.ndim == 3:
                # epochs: (n_epochs, n_eog, n_times)
                eog_data = eog_data.mean(axis=0)  # average across epochs -> (n_eog, n_times)
            for comp_idx in range(n_components):
                comp_data = sources[:, comp_idx, :].mean(axis=0) if sources.ndim == 3 else sources[comp_idx, :]
                for eog_idx in range(eog_data.shape[0]):
                    r, _ = pearsonr(comp_data, eog_data[eog_idx])
                    if abs(r) >= self.eog_corr_threshold:
                        exclude.add(comp_idx)
                        break
        
        # Variance and Kurtosis (computed across concatenated time points)
        all_variances = []
        all_kurtoses = []
        for comp_idx in range(n_components):
            if sources.ndim == 3:
                comp_data = sources[:, comp_idx, :].ravel()
            else:
                comp_data = sources[comp_idx, :]
            all_variances.append(np.var(comp_data))
            all_kurtoses.append(kurtosis(comp_data, fisher=False))
        
        var_z = (np.array(all_variances) - np.mean(all_variances)) / (np.std(all_variances) + 1e-12)
        kur_z = (np.array(all_kurtoses) - np.mean(all_kurtoses)) / (np.std(all_kurtoses) + 1e-12)
        
        for comp_idx in range(n_components):
            if var_z[comp_idx] >= self.variance_threshold:
                exclude.add(comp_idx)
            if kur_z[comp_idx] >= self.kurtosis_threshold:
                exclude.add(comp_idx)
        
        # Power spectrum analysis
        sfreq = inst.info['sfreq']
        for comp_idx in range(n_components):
            if sources.ndim == 3:
                comp_data = sources[:, comp_idx, :].ravel()
            else:
                comp_data = sources[comp_idx, :]
            freqs, psd = sp_signal.welch(comp_data, fs=sfreq, nperseg=min(256, len(comp_data)//4*4))
            muscle_ratio = self._compute_psd_ratio(psd, freqs, self.muscle_freq_range)
            eye_ratio = self._compute_psd_ratio(psd, freqs, self.eye_freq_range)
            if muscle_ratio >= self.muscle_psd_ratio_threshold:
                exclude.add(comp_idx)
            if eye_ratio >= self.eye_psd_ratio_threshold:
                exclude.add(comp_idx)
        
        # Step 5: Apply ICA reconstruction on original data
        if len(exclude) > 0:
            ica.exclude = sorted(exclude)
            ica.apply(inst, verbose='WARNING')
            print(f"[{self.name}] SASICA excluded {len(exclude)} component(s): {sorted(exclude)}")
        
        cleaned_data = self._mne_to_array(inst)
        if cleaned_data.shape != original_shape:
            raise RuntimeError(
                f"Shape mismatch after SASICA: {cleaned_data.shape} vs {original_shape}"
            )
        
        return cleaned_data.astype(data.dtype, copy=False)
