"""Base class for artifact removal processors."""

from abc import abstractmethod
from typing import Any, Dict, List, Optional
import numpy as np
import mne
from preprocessing.base_processor import BaseProcessor


class BaseArtifactRemovalProcessor(BaseProcessor):
    """
    Base class for ICA-based artifact removal processors.
    
    Provides utilities for converting between numpy arrays and MNE objects,
    as well as common ICA fitting and application workflows.
    """
    
    # Default channel names for Xuanwu Stroke dataset (Thwe et al. 2026)
    DEFAULT_XWSTROKE_CH_NAMES = [
        'Fp1', 'Fp2', 'Fz', 'F3', 'F4', 'F7', 'F8', 'FCz', 'FC3', 'FC4',
        'FT7', 'FT8', 'Cz', 'C3', 'C4', 'T3', 'T4', 'CPz', 'CP3', 'CP4',
        'TP7', 'TP8', 'Pz', 'P3', 'P4', 'T5', 'T6', 'Oz', 'O1', 'O2',
        'HEOL', 'HEOR'
    ]
    
    def __init__(self, name: str, sfreq: float = 500.0,
                 ch_names: Optional[List[str]] = None,
                 montage: Optional[Any] = None,
                 n_components: Optional[float] = None,
                 random_state: int = 42,
                 ica_method: str = 'infomax',
                 fit_params: Optional[Dict[str, Any]] = None,
                 **kwargs):
        """
        Args:
            name: Unique name for this processor instance.
            sfreq: Sampling frequency in Hz.
            ch_names: List of channel names. If None and n_channels=32,
                      uses default XWStroke channel names.
            montage: MNE montage object, string name, or None.
            n_components: Number of ICA components. If None, uses 0.95 (variance).
            random_state: Random seed for ICA reproducibility.
            ica_method: ICA algorithm method for MNE.
            fit_params: Additional parameters passed to ICA.fit().
            **kwargs: Additional processor-specific configuration.
        """
        self.name = name
        self.sfreq = sfreq
        self.ch_names = ch_names
        self.montage = montage
        self.n_components = n_components if n_components is not None else 0.95
        self.random_state = random_state
        self.ica_method = ica_method
        self.fit_params = fit_params or {}
        self.kwargs = kwargs
        
    def _get_ch_names(self, n_channels: int) -> List[str]:
        """Get channel names, using defaults if appropriate."""
        if self.ch_names is not None:
            return list(self.ch_names)
        if n_channels == 32:
            return list(self.DEFAULT_XWSTROKE_CH_NAMES)
        return [f'EEG{i}' for i in range(n_channels)]
    
    def _build_mne_info(self, n_channels: int) -> mne.Info:
        """Build MNE Info object with channel types and montage."""
        ch_names = self._get_ch_names(n_channels)
        if len(ch_names) != n_channels:
            raise ValueError(
                f"Number of ch_names ({len(ch_names)}) does not match "
                f"n_channels ({n_channels})"
            )
        
        ch_types = []
        for ch in ch_names:
            ch_upper = ch.upper()
            if any(x in ch_upper for x in ['HEOL', 'HEOR', 'EOG', 'VEOG', 'HEOG']):
                ch_types.append('eog')
            else:
                ch_types.append('eeg')
        
        info = mne.create_info(
            ch_names=ch_names, sfreq=self.sfreq, ch_types=ch_types, verbose='WARNING'
        )
        
        # ICLabel topoplot requires all channels to have positions.
        # EOG channels in our custom montage lack EOG-specific positions,
        # so we temporarily treat them as EEG for montage assignment.
        eog_ch_names = [ch for ch in ch_names if ch.upper() in
                        ['HEOL', 'HEOR', 'EOG', 'VEOG', 'HEOG']]
        
        def _apply_montage(montage_obj):
            if eog_ch_names:
                info.set_channel_types({ch: 'eeg' for ch in eog_ch_names}, verbose='WARNING')
            try:
                info.set_montage(montage_obj, match_case=False, on_missing='warn')
            except Exception as e:
                mne.utils.warn(f"Could not set montage: {e}")
            finally:
                if eog_ch_names:
                    info.set_channel_types({ch: 'eog' for ch in eog_ch_names}, verbose='WARNING')

        if self.montage is not None:
            if isinstance(self.montage, str):
                _apply_montage(self.montage)
            else:
                _apply_montage(self.montage)
        else:
            # Auto-montage: if 32 channels with default names, build custom montage
            if n_channels == 32 and ch_names == self.DEFAULT_XWSTROKE_CH_NAMES:
                montage = self._build_xwstroke_montage()
                if montage is not None:
                    _apply_montage(montage)
            else:
                try:
                    info.set_montage('standard_1020', match_case=False, on_missing='warn')
                except Exception:
                    pass
                    
        return info
    
    @staticmethod
    def _build_xwstroke_montage() -> Optional[mne.channels.DigMontage]:
        """
        Build a custom DigMontage for the Xuanwu Stroke 32-channel cap.
        
        Positions are approximated from Thwe et al. (2026) Table II,
        which lists electrodes in polar coordinates (angle, radial distance).
        We convert these to spherical Cartesian coordinates on a unit sphere.
        
        Coordinate system:
        - 0 degrees = nasion (front)
        - Positive angles = clockwise
        - 90 degrees = right ear
        - 180 degrees = inion (back)
        - -90 degrees = left ear
        """
        try:
            # (label, angle_deg, radial_distance)
            polar_data = [
                ('Fp1', -18, 0.51111), ('Fp2', 18, 0.51111),
                ('Fz', 0, 0.25556), ('F3', -39, 0.33333), ('F4', 39, 0.33333),
                ('F7', -54, 0.51111), ('F8', 54, 0.51111),
                ('FCz', 0, 0.12778), ('FC3', -62, 0.27778), ('FC4', 62, 0.27778),
                ('FT7', -72, 0.51111), ('FT8', 72, 0.51111),
                ('Cz', 90, 0), ('C3', -90, 0.25556), ('C4', 90, 0.25556),
                ('T3', -90, 0.51111), ('T4', 90, 0.51111),
                ('CPz', 180, 0.12778), ('CP3', -118, 0.27778), ('CP4', 118, 0.27778),
                ('TP7', -108, 0.51111), ('TP8', 108, 0.51111),
                ('Pz', 180, 0.25556), ('P3', -141, 0.33333), ('P4', 141, 0.33333),
                ('T5', -126, 0.51111), ('T6', 126, 0.51111),
                ('Oz', 180, 0.51111), ('O1', -162, 0.51111), ('O2', 162, 0.51111),
                ('HEOL', -43, 0.65), ('HEOR', 23, 0.71),
            ]
            
            ch_pos = {}
            for label, angle_deg, r in polar_data:
                theta = np.deg2rad(angle_deg)
                # Convert polar (theta from front, r from center) to Cartesian (x, y, z)
                # x = r * sin(theta), y = r * cos(theta)
                # z = sqrt(1 - r^2)  (positive = upward/outward from sphere center)
                x = r * np.sin(theta)
                y = r * np.cos(theta)
                z = np.sqrt(max(0.0, 1.0 - r * r))
                ch_pos[label] = np.array([x, y, z])
            
            # For ICLabel topoplot we only need channel positions
            montage = mne.channels.make_dig_montage(
                ch_pos=ch_pos, coord_frame='head'
            )
            return montage
        except Exception:
            return None
    
    def _array_to_mne(self, data: np.ndarray) -> Any:
        """Convert numpy array to MNE Raw or Epochs object."""
        if data.ndim == 2:
            n_channels = data.shape[0]
            info = self._build_mne_info(n_channels)
            raw = mne.io.RawArray(data, info, verbose='WARNING')
            return raw
        elif data.ndim == 3:
            n_trials, n_channels, n_times = data.shape
            info = self._build_mne_info(n_channels)
            events = np.zeros((n_trials, 3), dtype=int)
            events[:, 0] = np.arange(n_trials) * n_times
            events[:, 2] = 1
            epochs = mne.EpochsArray(
                data, info, events=events, tmin=0, verbose='WARNING'
            )
            return epochs
        else:
            raise ValueError(f"Unsupported data shape: {data.shape}")
    
    def _mne_to_array(self, inst) -> np.ndarray:
        """Convert MNE Raw or Epochs back to numpy array."""
        if isinstance(inst, mne.Epochs):
            return inst.get_data()
        else:
            return inst.get_data()
    
    def _fit_ica(self, inst) -> mne.preprocessing.ICA:
        """Fit ICA on the provided MNE instance."""
        fit_params = dict(self.fit_params)
        if self.ica_method == 'infomax' and 'extended' not in fit_params:
            fit_params['extended'] = True
        ica = mne.preprocessing.ICA(
            n_components=self.n_components,
            method=self.ica_method,
            random_state=self.random_state,
            fit_params=fit_params,
            verbose='WARNING'
        )
        ica.fit(inst, verbose='WARNING')
        return ica
    
    def _apply_ica(self, ica: mne.preprocessing.ICA, inst) -> Any:
        """Apply ICA (with excluded components) to the instance."""
        ica.apply(inst, verbose='WARNING')
        return inst
    
    @abstractmethod
    def process(self, data: np.ndarray, **kwargs) -> np.ndarray:
        """Process data and return cleaned data."""
        pass
