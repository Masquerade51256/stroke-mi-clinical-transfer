"""
MARA-like artifact removal processor.

Based on Thwe et al. (2026):
"Integrating EEG Artifact Removal with Deep Learning for Accurate Motor Imagery
Classification in Acute Stroke"

MARA (Multiple Artifact Rejection Algorithm) is a semi-automatic method that
extracts features from Independent Components (ICs) and classifies them using
a machine learning approach (originally a pre-trained LDA classifier).

Features described in the paper:
- Kurtosis (spatial/topographic)
- Spatial average deviation
- Spatial variance
- Temporal kurtosis
- Temporal variance

Because the original MARA pre-trained classifier is not publicly available in
Python, this implementation provides two fallback strategies:
1. 'unsupervised': Uses a One-Class SVM trained on the feature distribution
   to identify outlier components (artifacts).
2. 'iclabel_proxy': Uses mne-icalabel probabilities as a proxy classifier,
    which aligns well with the paper's ML-based philosophy.

The default is 'unsupervised' to remain independent of external pre-trained
models while staying faithful to the feature-extraction methodology.
"""

from typing import Optional, List
import numpy as np
from scipy.stats import kurtosis
import mne

try:
    from sklearn.svm import OneClassSVM
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

try:
    from mne_icalabel import label_components
    HAS_ICLABEL = True
except ImportError:
    HAS_ICLABEL = False

from preprocessing.artifact_removal.base import BaseArtifactRemovalProcessor


class MARAProcessor(BaseArtifactRemovalProcessor):
    """
    MARA-like artifact removal using IC feature extraction and classification.
    
    This processor extracts spatial and temporal features from each ICA component
    and flags artifact components using an unsupervised outlier detector or an
    ICLabel-based proxy.
    """
    
    def __init__(self, name: str,
                 strategy: str = 'unsupervised',
                 nu: float = 0.3,
                 gamma: float = 'scale',
                 exclude_labels: Optional[List[str]] = None,
                 probability_threshold: float = 0.7,
                 highpass_freq: Optional[float] = 1.0,
                 apply_car: bool = True,
                 **kwargs):
        """
        Args:
            name: Processor instance name.
            strategy: Classification strategy. One of:
                      - 'unsupervised': One-Class SVM on MARA features.
                      - 'iclabel_proxy': Use ICLabel probabilities directly.
            nu: One-Class SVM hyperparameter (outlier fraction estimate).
            gamma: One-Class SVM kernel coefficient.
            exclude_labels: Labels to exclude when strategy='iclabel_proxy'.
            probability_threshold: Probability threshold for iclabel_proxy.
            highpass_freq: High-pass filter cutoff for ICA fitting (Hz). Set to None to disable.
            apply_car: Whether to apply CAR before ICA. Default True.
            **kwargs: Passed to BaseArtifactRemovalProcessor.
        """
        super().__init__(name=name, **kwargs)
        self.highpass_freq = highpass_freq
        self.strategy = strategy
        self.nu = nu
        self.gamma = gamma
        self.exclude_labels = exclude_labels or ['muscle artifact', 'eye blink', 'heart beat', 'line noise', 'channel noise', 'other']
        self.probability_threshold = probability_threshold
        self.apply_car = apply_car
        
        if self.strategy == 'unsupervised' and not HAS_SKLEARN:
            raise ImportError(
                "scikit-learn is required for MARAProcessor with strategy='unsupervised'. "
                "Install it with: pip install scikit-learn"
            )
        if self.strategy == 'iclabel_proxy' and not HAS_ICLABEL:
            raise ImportError(
                "mne-icalabel is required for MARAProcessor with strategy='iclabel_proxy'. "
                "Install it with: pip install mne-icalabel"
            )
    
    @staticmethod
    def _extract_features(ica: mne.preprocessing.ICA, sources: np.ndarray, sfreq: float) -> np.ndarray:
        """
        Extract MARA features for each IC.
        
        Returns an array of shape (n_components, n_features).
        Features (per component):
            1. Spatial kurtosis of the topographic map (ICA mixing matrix column)
            2. Spatial average deviation: mean absolute deviation from uniform
            3. Spatial variance: variance of the topographic weights
            4. Temporal kurtosis (Fisher=False, i.e. normal=3.0)
            5. Temporal variance of the IC time course
            6. (optional) HFO index proxy: ratio of 20-40 Hz power
        """
        n_components = ica.n_components_
        W = ica.mixing_matrix_  # shape (n_channels, n_components)
        
        features = []
        for comp_idx in range(n_components):
            topo = W[:, comp_idx]
            
            # Spatial features
            spatial_kurt = kurtosis(topo, fisher=False)
            spatial_mean_dev = np.mean(np.abs(topo - np.mean(topo)))
            spatial_var = np.var(topo)
            
            # Temporal features
            if sources.ndim == 3:
                comp_data = sources[:, comp_idx, :].ravel()
            else:
                comp_data = sources[comp_idx, :]
            
            temp_kurt = kurtosis(comp_data, fisher=False)
            temp_var = np.var(comp_data)
            
            features.append([
                spatial_kurt,
                spatial_mean_dev,
                spatial_var,
                temp_kurt,
                temp_var,
            ])
        
        return np.array(features)
    
    def _classify_unsupervised(self, features: np.ndarray) -> List[int]:
        """Use One-Class SVM to detect outlier (artifact) components."""
        # Standardize features
        mean = np.mean(features, axis=0)
        std = np.std(features, axis=0) + 1e-12
        X = (features - mean) / std
        
        clf = OneClassSVM(nu=self.nu, gamma=self.gamma, kernel='rbf')
        labels = clf.fit_predict(X)
        # OneClassSVM returns -1 for outliers, +1 for inliers
        exclude = [i for i, label in enumerate(labels) if label == -1]
        return exclude
    
    def _classify_iclabel_proxy(self, inst, ica: mne.preprocessing.ICA) -> List[int]:
        """Use ICLabel as a proxy for the pre-trained MARA classifier."""
        labels_result = label_components(inst, ica, method='iclabel')
        labels = labels_result['labels']
        y_pred_proba = labels_result['y_pred_proba']
        
        exclude = []
        for idx, (label, proba) in enumerate(zip(labels, y_pred_proba)):
            if label in self.exclude_labels and proba >= self.probability_threshold:
                exclude.append(idx)
        return exclude
    
    def process(self, data: np.ndarray, **kwargs) -> np.ndarray:
        """
        Remove artifacts using MARA-like feature extraction and classification.
        
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
        sources = ica.get_sources(inst_for_ica).get_data()
        sfreq = inst.info['sfreq']
        
        # Step 4: Classify and exclude artifacts
        if self.strategy == 'unsupervised':
            features = self._extract_features(ica, sources, sfreq)
            exclude = self._classify_unsupervised(features)
        elif self.strategy == 'iclabel_proxy':
            exclude = self._classify_iclabel_proxy(inst, ica)
        else:
            raise ValueError(f"Unknown MARA strategy: {self.strategy}")
        
        # Step 5: Apply ICA reconstruction on original data
        if len(exclude) > 0:
            ica.exclude = sorted(exclude)
            ica.apply(inst, verbose='WARNING')
            print(f"[{self.name}] MARA excluded {len(exclude)} component(s): {sorted(exclude)}")
        
        cleaned_data = self._mne_to_array(inst)
        if cleaned_data.shape != original_shape:
            raise RuntimeError(
                f"Shape mismatch after MARA: {cleaned_data.shape} vs {original_shape}"
            )
        
        return cleaned_data.astype(data.dtype, copy=False)
