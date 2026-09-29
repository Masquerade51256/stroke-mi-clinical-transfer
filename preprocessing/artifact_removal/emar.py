"""
EMAR (Eyes and Muscle Artifact Removal) processor.

Based on Thwe et al. (2026):
"Integrating EEG Artifact Removal with Deep Learning for Accurate Motor Imagery
Classification in Acute Stroke"

EMAR uses ICLabel to classify independent components and removes eye and muscle
artifacts based on probability thresholds. The MATLAB reference from the paper is:

    EEG = pop_icflag(EEG, [NaN NaN; 0.9 1; 0.9 1; NaN NaN; NaN NaN; NaN NaN; NaN NaN]);

The 7 rows correspond to ICLabel classes:
1. Brain
2. Muscle
3. Eye
4. Heart
5. Line noise
6. Channel noise
7. Other

Rows 2 and 3 (muscle and eye) use threshold [0.9, 1.0], meaning components with
probability >= 0.9 are flagged as artifacts.
"""

from typing import List, Optional
import numpy as np
import mne

try:
    from mne_icalabel import label_components
    HAS_ICLABEL = True
except ImportError:
    HAS_ICLABEL = False

from preprocessing.artifact_removal.base import BaseArtifactRemovalProcessor


class EMARProcessor(BaseArtifactRemovalProcessor):
    """
    Eyes and Muscle Artifact Removal (EMAR) using ICLabel.
    
    This processor decomposes the signal with ICA, classifies components using
    the ICLabel neural network, and automatically excludes components flagged as
    muscle or eye artifacts.
    """
    
    # ICLabel class order used by mne-icalabel
    ICLABEL_CLASSES = [
        'brain', 'muscle artifact', 'eye blink', 'heart beat',
        'line noise', 'channel noise', 'other'
    ]
    
    def __init__(self, name: str,
                 exclude_labels: Optional[List[str]] = None,
                 probability_threshold: float = 0.9,
                 apply_car: bool = True,
                 filter_band: Optional[List[float]] = None,
                 **kwargs):
        """
        Args:
            name: Processor instance name.
            exclude_labels: List of ICLabel class names to exclude.
                            Defaults to ['muscle artifact', 'eye blink'].
            probability_threshold: Minimum probability to flag a component as artifact.
                                   Default 0.9 matches the paper's EMAR setting.
            apply_car: Whether to apply Common Average Reference before ICA.
                       ICLabel was trained on CAR-referenced data. Default True.
            filter_band: Band-pass filter limits [low, high] before ICA.
                         ICLabel expects [1, 100] Hz. Default [1, 100].
            **kwargs: Passed to BaseArtifactRemovalProcessor.
        """
        super().__init__(name=name, **kwargs)
        self.exclude_labels = exclude_labels or ['muscle artifact', 'eye blink']
        self.probability_threshold = probability_threshold
        self.apply_car = apply_car
        self.filter_band = filter_band or [1.0, 100.0]
        
        if not HAS_ICLABEL:
            raise ImportError(
                "mne-icalabel is required for EMARProcessor. "
                "Install it with: pip install mne-icalabel"
            )
    
    def process(self, data: np.ndarray, **kwargs) -> np.ndarray:
        """
        Remove eye and muscle artifacts using ICLabel-based EMAR.
        
        Args:
            data: EEG data, shape (n_channels, n_times) or (n_trials, n_channels, n_times).
        
        Returns:
            Cleaned EEG data with the same shape as input.
        """
        original_shape = data.shape
        inst = self._array_to_mne(data)
        
        # Prepare a filtered copy for ICA fitting only (preserve original low-freqs in output)
        inst_for_ica = inst.copy()
        
        # Step 1: Optional band-pass filter for ICA fitting (ICLabel expects 1-100 Hz)
        if self.filter_band is not None:
            inst_for_ica.filter(l_freq=self.filter_band[0], h_freq=self.filter_band[1], verbose='WARNING')
        
        # Step 2: Apply Common Average Reference (CAR)
        if self.apply_car:
            if isinstance(inst_for_ica, mne.Epochs):
                inst_for_ica.set_eeg_reference('average', projection=True, verbose='WARNING')
                inst_for_ica.apply_proj()
            else:
                inst_for_ica.set_eeg_reference('average', verbose='WARNING')
            # Also apply CAR to the original data that will be reconstructed
            if isinstance(inst, mne.Epochs):
                inst.set_eeg_reference('average', projection=True, verbose='WARNING')
                inst.apply_proj()
            else:
                inst.set_eeg_reference('average', verbose='WARNING')
        
        # Step 3: Fit ICA on the filtered copy
        ica = self._fit_ica(inst_for_ica)
        
        # Step 4: Label components with ICLabel
        labels_result = label_components(inst_for_ica, ica, method='iclabel')
        labels = labels_result['labels']
        y_pred_proba = labels_result['y_pred_proba']
        
        # Step 5: Determine which components to exclude
        exclude = []
        for idx, (label, proba) in enumerate(zip(labels, y_pred_proba)):
            if label in self.exclude_labels and proba >= self.probability_threshold:
                exclude.append(idx)
        
        # Step 6: Apply ICA reconstruction without artifact components on the original data
        if len(exclude) > 0:
            ica.exclude = exclude
            ica.apply(inst, verbose='WARNING')
        print(f"[{self.name}] EMAR excluded {len(exclude)} component(s): {exclude}")
        
        cleaned_data = self._mne_to_array(inst)
        
        # Ensure shape is preserved (MNE may change precision)
        if cleaned_data.shape != original_shape:
            raise RuntimeError(
                f"Shape mismatch after EMAR: {cleaned_data.shape} vs {original_shape}"
            )
        
        return cleaned_data.astype(data.dtype, copy=False)
