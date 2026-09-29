"""
Dataset implementations for EEG-BCI experiments.

These classes wrap the existing dataset implementations to conform
to the new BaseDataset interface with registry support.
"""

import os
from typing import Tuple
import numpy as np
import scipy.io as sio
import torch

from core.base.base_dataset import BaseDataset


# XWStroke standard 32-channel left-right flip order.
# Maps each left channel to its right counterpart and vice versa.
# Midline channels (Fz, FCz, Cz, CPz, Pz, Oz) remain in place.
# Order follows DEFAULT_XWSTROKE_CH_NAMES:
#   [Fp1, Fp2, Fz, F3, F4, F7, F8, FCz, FC3, FC4,
#    FT7, FT8, Cz, C3, C4, T3, T4, CPz, CP3, CP4,
#    TP7, TP8, Pz, P3, P4, T5, T6, Oz, O1, O2,
#    HEOL, HEOR]
XWSTROKE_FLIP_ORDER = [
    1, 0, 2, 4, 3, 6, 5, 7, 9, 8,
    11, 10, 12, 14, 13, 16, 15, 17, 19, 18,
    21, 20, 22, 24, 23, 26, 25, 27, 29, 28,
    31, 30
]


def _flip_xwstroke_channels(data: np.ndarray) -> np.ndarray:
    """
    Spatially flip XWStroke EEG channels along the left-right axis.

    Args:
        data: Array of shape (n_trials, n_channels, n_times) or
              (n_trials, n_bands, n_channels, n_times).
              Expected n_channels >= 30 (EEG) or 32 (EEG+EOG).

    Returns:
        Flipped array with the same shape.
    """
    n_ch = data.shape[-2]
    if n_ch < len(XWSTROKE_FLIP_ORDER):
        # If fewer than 32 channels (e.g., after some upstream selection),
        # skip flipping to avoid misalignment.
        return data

    # Use only the first n_ch entries of the flip order
    flip_order = XWSTROKE_FLIP_ORDER[:n_ch]

    if data.ndim == 3:
        # (trials, channels, times)
        return data[:, flip_order, :]
    elif data.ndim == 4:
        # (trials, bands, channels, times)
        return data[:, :, flip_order, :]
    else:
        raise ValueError(f"Unexpected data ndim for channel flip: {data.ndim}")


def _load_paralysis_side(subject_id: int, dataset_info: dict) -> str:
    """
    Load paralysis side for a given subject from participants.tsv.
    
    Args:
        subject_id: Subject ID (integer)
        dataset_info: Dataset configuration dictionary
        
    Returns:
        'left' or 'right'
    """
    import pandas as pd
    
    participants_tsv = dataset_info.get('dataset', {}).get('participants_tsv')
    if participants_tsv is None:
        data_dir = dataset_info.get('dataset', {}).get('data_dir', '')
        candidates = [
            os.path.join(os.path.dirname(data_dir), 'participants.tsv'),
            os.path.join(data_dir, 'participants.tsv'),
            'src/datasets/21679035/participants.tsv',
        ]
        for path in candidates:
            if os.path.exists(path):
                participants_tsv = path
                break
        if participants_tsv is None:
            raise FileNotFoundError(
                f"participants.tsv not found. Searched: {candidates}. "
                f"Please specify 'participants_tsv' in dataset config."
            )
    
    participants = pd.read_csv(participants_tsv, sep='\t')
    sub_str = f'sub-{subject_id:02d}'
    sub_info = participants[participants['Participant_ID'] == sub_str]
    
    if len(sub_info) == 0:
        raise ValueError(f"Subject {sub_str} not found in {participants_tsv}")
    
    return str(sub_info['ParalysisSide'].values[0]).lower()


def load_subject_groups(subject_ids: list, dataset_info: dict, field: str = 'ParalysisSide') -> dict:
    """
    Load subject groups from participants.tsv based on a stratification field.

    Args:
        subject_ids: List of subject IDs (integers)
        dataset_info: Dataset configuration dictionary
        field: Column name in participants.tsv to group by (default: 'ParalysisSide')

    Returns:
        Dictionary mapping field values to lists of subject IDs, e.g.:
            {'left': [2, 3, 5, ...], 'right': [1, 4, 6, ...]}
    """
    import pandas as pd

    participants_tsv = dataset_info.get('dataset', {}).get('participants_tsv')
    if participants_tsv is None:
        data_dir = dataset_info.get('dataset', {}).get('data_dir', '')
        candidates = [
            os.path.join(os.path.dirname(data_dir), 'participants.tsv'),
            os.path.join(data_dir, 'participants.tsv'),
            'src/datasets/21679035/participants.tsv',
        ]
        for path in candidates:
            if os.path.exists(path):
                participants_tsv = path
                break
        if participants_tsv is None:
            raise FileNotFoundError(
                f"participants.tsv not found. Searched: {candidates}. "
                f"Please specify 'participants_tsv' in dataset config."
            )

    participants = pd.read_csv(participants_tsv, sep='\t')

    if field not in participants.columns:
        available = list(participants.columns)
        raise ValueError(f"Field '{field}' not found in {participants_tsv}. Available: {available}")

    groups = {}
    for sid in subject_ids:
        sub_str = f'sub-{sid:02d}'
        sub_info = participants[participants['Participant_ID'] == sub_str]
        if len(sub_info) == 0:
            raise ValueError(f"Subject {sub_str} not found in {participants_tsv}")
        value = str(sub_info[field].values[0]).lower().strip()
        groups.setdefault(value, []).append(sid)

    return groups


def _classify_stroke_location(loc_str):
    """Classify free-text stroke location into standardized categories."""
    import pandas as pd
    if pd.isna(loc_str):
        return 'subcortical'
    s = str(loc_str).lower()
    cortical_kw = ['cortex', 'frontal', 'parietal', 'temporal', 'occipital', 'insula', 'watershed']
    subcortical_kw = ['basal ganglia', 'thalamus', 'internal capsule', 'corona radiata',
                      'centrum semiovale', 'paraventricular', 'subcortical']
    brainstem_kw = ['pons', 'medulla oblongata']
    cerebellar_kw = ['cerebellum', 'cerebellar']

    has_cortical = any(kw in s for kw in cortical_kw)
    has_subcortical = any(kw in s for kw in subcortical_kw)
    has_brainstem = any(kw in s for kw in brainstem_kw)
    has_cerebellar = any(kw in s for kw in cerebellar_kw)

    flags = [has_cortical, has_subcortical, has_brainstem, has_cerebellar]
    n_categories = sum(flags)

    if n_categories >= 2:
        return 'mixed'
    elif has_cortical:
        return 'cortical'
    elif has_subcortical:
        return 'subcortical'
    elif has_brainstem:
        return 'brainstem'
    elif has_cerebellar:
        return 'brainstem'  # cerebellar mapped to brainstem for simplicity
    else:
        return 'subcortical'


def get_clinical_feature_vector(subject_id: int, dataset_info: dict) -> torch.Tensor:
    """
    Load and encode clinical features for a subject from participants.tsv.

    Returns a 7-dimensional vector:
        [loc_subcortical, loc_brainstem, loc_cortical, NIHSS_norm, duration_norm, side_left, side_right]

    Stroke location uses continuous encoding to avoid one-hot sparsity (Cortical=1 problem):
        - subcortical: [1.0, 0.0, 0.0]
        - brainstem:   [0.0, 1.0, 0.0]
        - mixed:       [0.5, 0.5, 0.0]  # intermediate between subcortical and brainstem
        - cortical:    [0.0, 0.0, 1.0]

    NIHSS and Duration are min-max normalized to [0, 1].
    """
    import pandas as pd

    participants_tsv = dataset_info.get('dataset', {}).get('participants_tsv')
    if participants_tsv is None:
        data_dir = dataset_info.get('dataset', {}).get('data_dir', '')
        candidates = [
            os.path.join(os.path.dirname(data_dir), 'participants.tsv'),
            os.path.join(data_dir, 'participants.tsv'),
            'src/datasets/21679035/participants.tsv',
        ]
        for path in candidates:
            if os.path.exists(path):
                participants_tsv = path
                break
        if participants_tsv is None:
            return torch.tensor([1.0, 0.0, 0.0, 0.2, 0.2, 1.0, 0.0], dtype=torch.float32)

    try:
        participants = pd.read_csv(participants_tsv, sep='\t')
    except Exception:
        return torch.tensor([1.0, 0.0, 0.0, 0.2, 0.2, 1.0, 0.0], dtype=torch.float32)

    sub_str = f'sub-{subject_id:02d}'
    sub_info = participants[participants['Participant_ID'] == sub_str]

    if len(sub_info) == 0:
        return torch.tensor([1.0, 0.0, 0.0, 0.2, 0.2, 1.0, 0.0], dtype=torch.float32)

    sub_info = sub_info.iloc[0]

    # Stroke location: classify free-text, then encode continuously
    location_raw = sub_info.get('StrokeLocation', 'subcortical')
    location = _classify_stroke_location(location_raw)
    loc_encoding = {
        'subcortical': [1.0, 0.0, 0.0],
        'brainstem': [0.0, 1.0, 0.0],
        'mixed': [0.5, 0.5, 0.0],
        'cortical': [0.0, 0.0, 1.0],
    }.get(location, [1.0, 0.0, 0.0])

    # NIHSS (normalize, typical range 0-25)
    try:
        nihss = float(sub_info.get('NIHSS', 5.0))
    except (ValueError, TypeError):
        nihss = 5.0
    nihss_norm = min(max(nihss / 25.0, 0.0), 1.0)

    # Duration (normalize, typical range 0-30 days)
    try:
        duration = float(sub_info.get('Duration', 7.0))
    except (ValueError, TypeError):
        duration = 7.0
    duration_norm = min(max(duration / 30.0, 0.0), 1.0)

    # Paralysis side
    side = str(sub_info.get('ParalysisSide', 'left')).lower().strip()
    side_encoding = [1.0, 0.0] if side == 'left' else [0.0, 1.0]

    features = loc_encoding + [nihss_norm, duration_norm] + side_encoding
    return torch.tensor(features, dtype=torch.float32)


class BCICIV2aDataset(BaseDataset):
    """
    BCI Competition IV 2a dataset.
    
    Motor imagery dataset with 4 classes (left hand, right hand, foot, tongue)
    from 9 subjects.
    
    Note: Data is loaded from GDF files using MNE, consistent with torch_fold_train.py.
    """
    
    def _load_raw_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load raw data from GDF files using MNE.
        
        Returns:
            Tuple of (data, labels) where data shape is (n_trials, n_channels, n_timepoints)
        """
        import mne
        
        data_dir = self.data_dir or 'src/datasets/BCICIV2a'
        
        # Load training data (T file)
        train_gdf_file = os.path.join(data_dir, f'A{self.subject_id:02d}T.gdf')
        train_mat_file = os.path.join(data_dir, f'A{self.subject_id:02d}T.mat')
        
        # Load test data (E file)
        test_gdf_file = os.path.join(data_dir, f'A{self.subject_id:02d}E.gdf')
        test_mat_file = os.path.join(data_dir, f'A{self.subject_id:02d}E.mat')
        
        # Process training data
        train_data, train_labels = self._load_gdf_session(train_gdf_file, train_mat_file, is_train=True)
        
        # Process test data
        test_data, test_labels = self._load_gdf_session(test_gdf_file, test_mat_file, is_train=False)
        
        # Concatenate train and test data
        combined_data = np.concatenate([train_data, test_data], axis=0)
        combined_labels = np.concatenate([train_labels, test_labels])
        
        return combined_data, combined_labels
    
    def _load_gdf_session(self, gdf_file: str, mat_file: str, is_train: bool) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load a single session (train or test) from GDF file.
        
        Args:
            gdf_file: Path to GDF file
            mat_file: Path to MAT file (for labels)
            is_train: Whether this is training data
            
        Returns:
            Tuple of (data, labels)
        """
        import mne
        
        # Load GDF file
        raw = mne.io.read_raw_gdf(gdf_file, preload=True, verbose=False)
        events, annot = mne.events_from_annotations(raw)
        
        raw.load_data()
        raw.filter(0., 40., fir_design='firwin', verbose=False)
        raw.info['bads'] += ['EOG-left', 'EOG-central', 'EOG-right']
        
        picks = mne.pick_types(raw.info,
                            meg=False,
                            eeg=True,
                            eog=False,
                            stim=False,
                            exclude='bads')
        
        # Define event IDs and time window
        tmin, tmax = 0, 3  # 3 second epochs
        
        if is_train:
            if self.subject_id != 4:
                event_id = {'769': 7, '770': 8, '771': 9, '772': 10}
            else:
                event_id = {'769': 5, '770': 6, '771': 7, '772': 8}
        else:
            event_id = {'783': 7}
        
        epochs = mne.Epochs(raw,
                        events,
                        event_id,
                        tmin,
                        tmax,
                        proj=True,
                        picks=picks,
                        baseline=None,
                        preload=True,
                        verbose=False)
        
        # Get data (convert to microvolts)
        epochs_data = epochs.get_data() * 1e6  # Shape: (n_trials, n_channels, n_timepoints)
        
        # Load labels from mat file
        labels = sio.loadmat(mat_file)['classlabel'].reshape(-1) - 1  # Convert to 0-based
        
        return epochs_data, labels


class XWStrokeDataset(BaseDataset):
    """
    XuanWu Stroke dataset.
    
    Stroke patient motor imagery dataset.
    Supports both original .mat files and preprocessed _4s.mat files.
    """
    
    def _load_raw_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load raw data from .mat files.
        
        Supports two modes:
        1. Original data: sub-XX_task-motor-imagery_eeg.mat
        2. 4s preprocessed data: sub-XX_task-motor-imagery_eeg_4s.mat
        
        Returns:
            Tuple of (data, labels)
        """
        # Get configuration
        data_dir = self.dataset_info.get('dataset', {}).get('data_dir', 'src/datasets/21679035/sourcedata')
        use_4s_data = self.dataset_info.get('dataset', {}).get('use_4s_data', True)
        data_var_name = self.dataset_info.get('dataset', {}).get('data_var_name', 'eeg_data_4s')
        labels_var_name = self.dataset_info.get('dataset', {}).get('labels_var_name', 'labels')
        
        # Build file path
        subject_dir = os.path.join(data_dir, f'sub-{self.subject_id:02d}')
        
        if use_4s_data:
            # Try to load 4s preprocessed data
            subject_file = os.path.join(subject_dir, f'sub-{self.subject_id:02d}_task-motor-imagery_eeg_4s.mat')
            
            if os.path.exists(subject_file):
                mat_data = sio.loadmat(subject_file)
                
                # Extract data from 4s file
                if data_var_name in mat_data:
                    data = mat_data[data_var_name]
                    labels = mat_data[labels_var_name].flatten()
                    
                    # Hemisphere alignment for 4s data path
                    label_mapping = self.dataset_info.get('dataset', {}).get('label_mapping', 'lr')
                    hemisphere_alignment = self.dataset_info.get('dataset', {}).get('hemisphere_alignment', False)
                    if label_mapping != 'lr' and hemisphere_alignment:
                        paralysis_side = _load_paralysis_side(self.subject_id, self.dataset_info)
                        if paralysis_side == 'right':
                            data = _flip_xwstroke_channels(data)
                    
                    return data, labels
            # Fall through to original file if 4s data not available
        
        # Fallback to original file
        subject_file = os.path.join(subject_dir, f'sub-{self.subject_id:02d}_task-motor-imagery_eeg.mat')
        
        if not os.path.exists(subject_file):
            # Try alternative path structure (without sub-XX subdirectory)
            subject_file = os.path.join(data_dir, f'sub-{self.subject_id:02d}_task-motor-imagery_eeg.mat')
        
        mat_data = sio.loadmat(subject_file)
        
        # Extract data from original file
        if 'data' in mat_data:
            data = mat_data['data']
            labels = mat_data['labels'].flatten()
        elif 'eeg_data' in mat_data:
            data = mat_data['eeg_data']
            labels = mat_data['eeg_labels'].flatten()
        else:
            # Try to infer from structure
            data = None
            labels = None
            for key in mat_data.keys():
                if not key.startswith('__'):
                    if isinstance(mat_data[key], np.ndarray):
                        if len(mat_data[key].shape) == 3 and data is None:
                            data = mat_data[key]
                        elif len(mat_data[key].shape) == 1 and labels is None and 'label' in key.lower():
                            labels = mat_data[key]
            
            if data is None:
                raise ValueError(f"[Subject {self.subject_id}] Could not find data array in {subject_file}")
            if labels is None:
                raise ValueError(f"[Subject {self.subject_id}] Could not find labels array in {subject_file}")
        
        # Ensure correct shape (n_trials, n_channels, n_timepoints)
        if data.shape[1] < data.shape[2]:
            # Likely (n_trials, n_timepoints, n_channels), need to transpose
            data = np.transpose(data, (0, 2, 1))
        
        # Hemisphere alignment: spatially flip channels for right-paralysis patients
        # so that affected-side activation maps to the same hemisphere across subjects.
        label_mapping = self.dataset_info.get('dataset', {}).get('label_mapping', 'lr')
        hemisphere_alignment = self.dataset_info.get('dataset', {}).get('hemisphere_alignment', False)
        if label_mapping != 'lr' and hemisphere_alignment:
            paralysis_side = _load_paralysis_side(self.subject_id, self.dataset_info)
            if paralysis_side == 'right':
                data = _flip_xwstroke_channels(data)
        
        return data, labels
    
    def _remap_labels(self, labels: np.ndarray) -> np.ndarray:
        """
        Remap left/right labels to affected/unaffected based on paralysis side.
        
        Configured via dataset_info['dataset']['label_mapping']:
            - 'lr': no remapping (default), 0=left, 1=right
            - 'affected': 0=affected side, 1=unaffected side
            - 'unaffected_first': 0=unaffected side, 1=affected side
        """
        label_mapping = self.dataset_info.get('dataset', {}).get('label_mapping', 'lr')
        if label_mapping == 'lr':
            return labels
        
        paralysis_side = _load_paralysis_side(self.subject_id, self.dataset_info)
        
        # Original after 0-based shift: 0=left, 1=right
        if label_mapping == 'affected':
            # Target: 0=affected, 1=unaffected
            if paralysis_side == 'left':
                # Left is affected -> no change
                return labels
            else:  # 'right'
                # Right is affected -> flip
                return 1 - labels
                
        elif label_mapping == 'unaffected_first':
            # Target: 0=unaffected, 1=affected
            if paralysis_side == 'left':
                # Left is affected -> flip
                return 1 - labels
            else:  # 'right'
                # Right is affected -> no change
                return labels
                
        else:
            raise ValueError(f"Unknown label_mapping: {label_mapping}")


class XWStrokeEDFDataset(BaseDataset):
    """
    XuanWu Stroke dataset loaded from original EDF files.
    
    This dataset loader reads the raw EDF recordings and parses BIDS event
    information to extract the 4-second motor imagery (MI) epochs. It is
    designed to closely match the preprocessing pipeline described in
    Thwe et al. (2026), which starts from .edf files rather than
    pre-processed .mat files.
    
    Expected data layout:
        src/datasets/21679035/edffile/sub-XX/eeg/sub-XX_task-motor-imagery_eeg.edf
    Events are read from the shared BIDS TSV:
        src/datasets/21679035/task-motor-imagery_events.tsv
    """
    
    def _load_raw_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load raw EDF data and extract MI epochs.
        
        Returns:
            Tuple of (data, labels) where data shape is (n_trials, n_channels, n_timepoints)
        """
        import mne
        import pandas as pd
        import os
        
        # Get configuration
        data_dir = self.dataset_info.get('dataset', {}).get('data_dir', 'src/datasets/21679035')
        events_tsv = self.dataset_info.get('dataset', {}).get('events_tsv', 'task-motor-imagery_events.tsv')
        mi_value = self.dataset_info.get('dataset', {}).get('mi_event_value', 2)
        
        # Build EDF file path
        edf_file = os.path.join(
            data_dir, 'edffile', f'sub-{self.subject_id:02d}', 'eeg',
            f'sub-{self.subject_id:02d}_task-motor-imagery_eeg.edf'
        )
        
        if not os.path.exists(edf_file):
            raise FileNotFoundError(f"EDF file not found: {edf_file}")
        
        # Load EDF
        raw = mne.io.read_raw_edf(edf_file, preload=True, verbose='WARNING')
        
        # Remove empty-named channel if present (stim placeholder in this dataset)
        picks = [ch for ch in raw.ch_names if ch.strip()]
        raw.pick(picks)
        
        sfreq = raw.info['sfreq']
        
        # Read events TSV
        events_path = os.path.join(data_dir, events_tsv)
        if not os.path.exists(events_path):
            raise FileNotFoundError(f"Events TSV not found: {events_path}")
        
        events_df = pd.read_csv(events_path, sep='\t')
        
        # Filter MI events
        mi_events = events_df[events_df['value'] == mi_value].copy()
        if len(mi_events) == 0:
            raise ValueError(f"No MI events (value={mi_value}) found in {events_path}")
        
        # Extract epochs
        # The TSV uses milliseconds for onset and duration
        epochs_data = []
        labels = []
        
        for _, row in mi_events.iterrows():
            onset_sec = row['onset'] / 1000.0
            duration_sec = row['duration'] / 1000.0
            
            start_sample = int(round(onset_sec * sfreq))
            end_sample = start_sample + int(round(duration_sec * sfreq))
            
            epoch = raw.get_data(start=start_sample, stop=end_sample)  # (n_channels, n_times)
            epochs_data.append(epoch)
            labels.append(int(row['trial_type']))
        
        data = np.array(epochs_data)  # (n_trials, n_channels, n_times)
        labels = np.array(labels)
        
        # Convert from Volts to uV for consistency with typical EEG processing
        data = data * 1e6
        
        # Reorder channels to match standard XWStroke cap order if needed
        ch_names = [ch.upper() for ch in raw.ch_names]
        standard_order = [
            'FP1', 'FP2', 'FZ', 'F3', 'F4', 'F7', 'F8', 'FCZ', 'FC3', 'FC4',
            'FT7', 'FT8', 'CZ', 'C3', 'C4', 'T3', 'T4', 'CPZ', 'CP3', 'CP4',
            'TP7', 'TP8', 'PZ', 'P3', 'P4', 'T5', 'T6', 'OZ', 'O1', 'O2',
            'HEOL', 'HEOR'
        ]
        
        if set(ch_names) >= set(standard_order):
            reorder_idx = [ch_names.index(ch) for ch in standard_order if ch in ch_names]
            if len(reorder_idx) == data.shape[1]:
                data = data[:, reorder_idx, :]
        
        # Hemisphere alignment: spatially flip channels for right-paralysis patients
        label_mapping = self.dataset_info.get('dataset', {}).get('label_mapping', 'lr')
        hemisphere_alignment = self.dataset_info.get('dataset', {}).get('hemisphere_alignment', False)
        if label_mapping != 'lr' and hemisphere_alignment:
            paralysis_side = _load_paralysis_side(self.subject_id, self.dataset_info)
            if paralysis_side == 'right':
                data = _flip_xwstroke_channels(data)
        
        print(f"[Subject {self.subject_id}] Loaded EDF: {data.shape}, sfreq={sfreq}Hz, labels={labels[:5]}...")
        return data, labels
    
    def _remap_labels(self, labels: np.ndarray) -> np.ndarray:
        """
        Remap left/right labels to affected/unaffected based on paralysis side.
        
        Configured via dataset_info['dataset']['label_mapping']:
            - 'lr': no remapping (default), 0=left, 1=right
            - 'affected': 0=affected side, 1=unaffected side
            - 'unaffected_first': 0=unaffected side, 1=affected side
        """
        label_mapping = self.dataset_info.get('dataset', {}).get('label_mapping', 'lr')
        if label_mapping == 'lr':
            return labels
        
        paralysis_side = _load_paralysis_side(self.subject_id, self.dataset_info)
        
        # Original after 0-based shift: 0=left, 1=right
        if label_mapping == 'affected':
            # Target: 0=affected, 1=unaffected
            if paralysis_side == 'left':
                # Left is affected -> no change
                return labels
            else:  # 'right'
                # Right is affected -> flip
                return 1 - labels
                
        elif label_mapping == 'unaffected_first':
            # Target: 0=unaffected, 1=affected
            if paralysis_side == 'left':
                # Left is affected -> flip
                return 1 - labels
            else:  # 'right'
                # Right is affected -> no change
                return labels
                
        else:
            raise ValueError(f"Unknown label_mapping: {label_mapping}")


class LowerStrokeDataset(BaseDataset):
    """
    Lower Limb Stroke dataset.
    
    Lower limb motor imagery dataset for stroke patients.
    """
    
    def _load_raw_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load raw data from preprocessed files.
        
        Returns:
            Tuple of (data, labels)
        """
        data_dir = self.data_dir or 'src/datasets/LowerStroke'
        
        # Load preprocessed data
        subject_file = os.path.join(data_dir, f'subject_{self.subject_id}.mat')
        
        mat_data = sio.loadmat(subject_file)
        
        # Extract data
        if 'data' in mat_data:
            data = mat_data['data']
            labels = mat_data['labels'].flatten()
        else:
            # Handle alternative formats
            data = mat_data[list(mat_data.keys())[-1]]
            labels = np.zeros(data.shape[0])  # Placeholder
        
        # Ensure correct shape
        if len(data.shape) == 2:
            # Single trial, add batch dimension
            data = np.expand_dims(data, axis=0)
        elif data.shape[1] < data.shape[2]:
            data = np.transpose(data, (0, 2, 1))
        
        return data, labels


class FiveFingerDataset(BaseDataset):
    """
    5F (Five Finger) dataset from Kaya et al. (2018).
    
    Single-hand five-finger motor imagery dataset.
    5 classes: thumb, index finger, middle finger, ring finger, pinkie finger.
    
    Dataset details:
        - 8 subjects (A, B, C, E, F, G, H, I)
        - 20 recording sessions (mix of 200 Hz and 1000 Hz)
        - 22 input channels (19 EEG + A1, A2 + X5 sync)
        - ~900 trials per session, ~300 trials per class per session
        - Marker codes: 1=thumb, 2=index, 3=middle, 4=ring, 5=pinkie
        - Stimulus duration: 1 s, inter-trial interval: 1.5-2.5 s
    
    Reference:
        Kaya, M., Binli, M. K., Ozbay, E., Yanar, H., & Mishchenko, Y. (2018).
        A large electroencephalographic motor imagery dataset for
        electroencephalographic brain computer interfaces.
        Scientific Data, 5, 180211.
    """
    
    # Subject ID mapping: 1->A, 2->B, ..., 8->I (note: no Subject D)
    SUBJECT_MAP = {
        1: 'A', 2: 'B', 3: 'C', 4: 'E',
        5: 'F', 6: 'G', 7: 'H', 8: 'I'
    }
    
    # Inverse mapping for convenience
    SUBJECT_INV_MAP = {v: k for k, v in SUBJECT_MAP.items()}
    
    def _load_raw_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load raw data from .mat files for a specific subject.
        
        Handles mixed sampling rates (200 Hz and 1000 Hz) by resampling
        all trials to a uniform number of time points based on target_sr
        and epoch_duration.
        
        Returns:
            Tuple of (data, labels) where:
            - data: shape (n_trials, n_channels, n_timepoints)
            - labels: shape (n_trials,)
        """
        from scipy import signal
        
        data_dir = self.dataset_info.get('dataset', {}).get('data_dir', 'src/datasets/5F')
        
        # Epoch extraction settings
        epoch_duration = self.dataset_info.get('dataset', {}).get('epoch_duration_sec', 1.0)
        target_sr = self.dataset_info.get('preprocessing', {}).get('resample', {}).get('target_sr', 250)
        
        # Number of time points after resampling
        n_timepoints_target = int(round(epoch_duration * target_sr))
        
        # Get subject letter from ID
        if self.subject_id not in self.SUBJECT_MAP:
            raise ValueError(
                f"Invalid subject_id {self.subject_id} for 5F dataset. "
                f"Valid IDs: {list(self.SUBJECT_MAP.keys())}"
            )
        subject_letter = self.SUBJECT_MAP[self.subject_id]
        
        # Find all .mat files for this subject
        # Pattern: 5F-Subject{LETTER}-*.mat
        all_files = sorted([
            f for f in os.listdir(data_dir)
            if f.endswith('.mat') and f.startswith(f'5F-Subject{subject_letter}-')
        ])
        
        if not all_files:
            raise FileNotFoundError(
                f"No 5F data files found for Subject {subject_letter} (ID={self.subject_id}) "
                f"in {data_dir}"
            )
        
        # Remove duplicate files (e.g., "... (1).mat" identical copies)
        unique_files = []
        seen_base = set()
        for f in all_files:
            # Normalize: remove " (1)" suffix before extension
            base = f.replace(' (1)', '')
            if base not in seen_base:
                seen_base.add(base)
                unique_files.append(f)
        
        all_trials = []
        all_labels = []
        
        for filename in unique_files:
            filepath = os.path.join(data_dir, filename)
            mat_data = sio.loadmat(filepath)
            
            if 'o' not in mat_data:
                raise ValueError(f"Expected structured array 'o' not found in {filepath}")
            
            o = mat_data['o']
            
            # Extract fields from structured array
            marker = o['marker'][0, 0].flatten()
            eeg_data = o['data'][0, 0]  # shape: (n_samples, 22)
            sfreq_raw = float(o['sampFreq'][0, 0].flatten()[0])
            
            # Exclude X5 sync channel (index 21), keep first 21 channels
            eeg_data = eeg_data[:, :21]
            
            # Find trial onsets: marker transitions from 0 to 1-5
            trial_onsets = []
            trial_labels = []
            for i in range(len(marker) - 1):
                if marker[i] == 0 and marker[i + 1] in [1, 2, 3, 4, 5]:
                    trial_onsets.append(i + 1)
                    trial_labels.append(int(marker[i + 1]))
            
            if not trial_onsets:
                print(f"[Warning] No trials found in {filename}")
                continue
            
            # Extract epochs and resample if needed
            for onset, label in zip(trial_onsets, trial_labels):
                # Extract fixed-duration epoch
                n_samples_epoch = int(round(epoch_duration * sfreq_raw))
                end = min(onset + n_samples_epoch, eeg_data.shape[0])
                epoch = eeg_data[onset:end, :].T  # -> (n_channels, n_timepoints)
                
                # If epoch is shorter than expected (last trial), pad with zeros
                if epoch.shape[-1] < n_samples_epoch:
                    pad_width = n_samples_epoch - epoch.shape[-1]
                    epoch = np.pad(epoch, ((0, 0), (0, pad_width)), mode='constant')
                
                # Resample to target number of time points if needed
                if epoch.shape[-1] != n_timepoints_target:
                    epoch = np.apply_along_axis(
                        lambda x: signal.resample(x, n_timepoints_target),
                        axis=-1,
                        arr=epoch
                    )
                
                all_trials.append(epoch.astype(np.float32))
                all_labels.append(label)
        
        if not all_trials:
            raise ValueError(
                f"No valid trials extracted for Subject {subject_letter} (ID={self.subject_id})"
            )
        
        data = np.stack(all_trials, axis=0)  # (n_trials, n_channels, n_timepoints)
        labels = np.array(all_labels, dtype=np.int64)
        
        print(
            f"[Subject {subject_letter} (ID={self.subject_id})] "
            f"Loaded {len(labels)} trials from {len(unique_files)} file(s), "
            f"shape={data.shape}, labels={np.unique(labels, return_counts=True)}"
        )

        return data, labels


class HSStrokeDataset(BaseDataset):
    """
    HS Stroke dataset: upper-limb motor imagery EEG from chronic stroke patients.

    Dataset details (v6, deduplicated release — 23,382 trials total):
        - 14 chronic stroke patients (duration 6-22 months), 11 EEG channels
          (FC3, FC4, C5, C3, C1, Cz, C2, C4, C6, CP3, CP4) @ 256 Hz
        - ~20 sessions per patient over ~20 days (80-100 trials per session)
        - Trials are pre-epoched 4-s MI segments: EEGdata (n_trials, 11, 1024)
        - Labels: 1 = left-hand MI, 2 = right-hand MI (50/50 per session)
        - Clinical metadata in participants.tsv: FMA pre/post, paralysis_side,
          stroke_type, sEMG, MEPs

    Files layout (EEG-BIDS derivatives):
        <data_dir>/sub-XX/ses-YY/eeg/sub-XX_ses-YY_task-MI_eeg.mat

    Optional config keys:
        dataset.sessions: 'all' (default) or list of session ints, e.g. [1, 2, 3]

    Side effect:
        self.trial_sessions (np.ndarray) stores the session index of each trial
        for later cross-session analyses (not used by the base pipeline).

    Reference:
        Lu, R., Luo, J., Zhong, S., & Gao, T. (2026).
        An Upper-Limb Motor Imagery EEG Dataset of Chronic Stroke Patients.
        Scientific Data. https://doi.org/10.1038/s41597-026-07742-x
    """

    def _load_raw_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load pre-epoched trials from per-session .mat files.

        Returns:
            Tuple of (data, labels):
            - data: (n_trials, 11, 1024) float32, microvolts
            - labels: (n_trials,) with values {1, 2} (base class 0-indexes them)
        """
        import glob

        data_dir = self.data_dir or 'src/datasets/HSStroke'
        sub = f'sub-{self.subject_id:02d}'
        sessions = self.dataset_info.get('dataset', {}).get('sessions', 'all')

        pattern = os.path.join(data_dir, sub, 'ses-*', 'eeg', '*_eeg.mat')
        mat_files = sorted(glob.glob(pattern))

        if sessions != 'all':
            wanted = {f'ses-{int(s):02d}' for s in sessions}
            mat_files = [f for f in mat_files
                         if os.path.basename(os.path.dirname(os.path.dirname(f))) in wanted]

        if not mat_files:
            raise FileNotFoundError(
                f"No HSStroke data found for {sub} (sessions={sessions}) "
                f"with pattern {pattern}"
            )

        data_list, label_list, session_list = [], [], []
        for f in mat_files:
            m = sio.loadmat(f, squeeze_me=True)
            trials = np.asarray(m['EEGdata'], dtype=np.float32)
            labels = np.asarray(m['EEGdatalabel']).reshape(-1)
            ses_name = os.path.basename(os.path.dirname(os.path.dirname(f)))
            ses_idx = int(ses_name.replace('ses-', ''))

            # Values are already in microvolts (clipped at ±500 uV);
            # the BIDS channels.tsv "V" unit is inaccurate — do NOT rescale.
            data_list.append(trials)
            label_list.append(labels)
            session_list.append(np.full(len(labels), ses_idx, dtype=np.int64))

        data = np.concatenate(data_list, axis=0)
        labels = np.concatenate(label_list)
        self.trial_sessions = np.concatenate(session_list)

        print(
            f"[HSStroke sub-{self.subject_id:02d}] "
            f"Loaded {len(labels)} trials from {len(mat_files)} session(s), "
            f"shape={data.shape}, labels={np.unique(labels, return_counts=True)}"
        )

        return data, labels


class SHUDataset(BaseDataset):
    """
    SHU Multi-session dataset: motor imagery EEG from healthy subjects,
    purpose-built for cross-session variability research.

    Dataset details (Ma et al., Scientific Data 2022):
        - 25 healthy subjects x 5 sessions (2-3 day intervals)
        - 32 EEG channels @ 250 Hz
          [Fp1,Fp2,Fz,F3,F4,F7,F8,FC1,FC2,FC5,FC6,Cz,C3,C4,T3,T4,
           A1,A2,CP1,CP2,CP5,CP6,Pz,P3,P4,T5,T6,PO3,PO4,Oz,O1,O2]
        - ~100 trials per session (94-100), left/right hand MI, ~50/50
        - EDF files hold concatenated 4-s MI segments (1000 samples each);
          per-trial onsets/labels come from the BIDS events.tsv

    Files layout:
        <data_dir>/edf_raw/edf/sub-XXX_ses-0Y_task_motorimagery_eeg.edf
        <data_dir>/events_raw/events/sub-XXX_ses-0Y_task_motorimagery_events.tsv

    Optional config keys:
        dataset.sessions: 'all' (default) or list of session ints, e.g. [1, 2]

    Side effect:
        self.trial_sessions (np.ndarray) stores the session index of each trial
        for cross-session analyses (N-shot curves, session adaptation).

    Reference:
        Ma, J., Yang, B., Qiu, W., Li, Y., Gao, S., & Xia, X. (2022).
        A large EEG dataset for studying cross-session variability in
        motor imagery brain-computer interface. Scientific Data, 9, 531.
        https://doi.org/10.1038/s41597-022-01647-1
    """

    TRIAL_SAMPLES = 1000  # 4 s @ 250 Hz per MI segment

    def _load_raw_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load per-trial 4-s MI segments from EDF + events.tsv.

        Returns:
            Tuple of (data, labels):
            - data: (n_trials, 32, 1000) float32, microvolts
            - labels: (n_trials,) 1 = left, 2 = right (base class 0-indexes)
        """
        import mne
        import pandas as pd

        data_dir = self.data_dir or 'src/datasets/SHU'
        sub = f'sub-{self.subject_id:03d}'
        sessions = self.dataset_info.get('dataset', {}).get('sessions', 'all')
        if sessions == 'all':
            sessions = [1, 2, 3, 4, 5]
        sessions = [int(s) for s in sessions]

        mne.set_log_level('ERROR')
        data_list, label_list, session_list = [], [], []
        for ses in sessions:
            edf_path = os.path.join(
                data_dir, 'edf_raw', 'edf',
                f'{sub}_ses-{ses:02d}_task_motorimagery_eeg.edf')
            ev_path = os.path.join(
                data_dir, 'events_raw', 'events',
                f'{sub}_ses-{ses:02d}_task_motorimagery_events.tsv')
            if not os.path.exists(edf_path):
                raise FileNotFoundError(f'Missing EDF: {edf_path}')
            if not os.path.exists(ev_path):
                raise FileNotFoundError(f'Missing events: {ev_path}')

            raw = mne.io.read_raw_edf(edf_path, preload=True, verbose=False)
            sig = raw.get_data()  # (32, n_times), volts
            # 部分 events.tsv 缺表头（如 sub-001_ses-04），做兼容读取
            ev_names = ['onset', 'duration', 'trial_type',
                        'response_time', 'sample', 'value']
            events = pd.read_csv(ev_path, sep='\t')
            if 'sample' not in events.columns:
                events = pd.read_csv(ev_path, sep='\t',
                                     header=None, names=ev_names)

            trials, labels = [], []
            for _, row in events.iterrows():
                start = int(row['sample'])
                seg = sig[:, start:start + self.TRIAL_SAMPLES]
                if seg.shape[1] < self.TRIAL_SAMPLES:
                    continue  # 尾部不完整 trial，跳过
                trials.append(seg * 1e6)  # V -> uV
                labels.append(1 if str(row['trial_type']).strip() == 'left' else 2)

            data_list.append(np.asarray(trials, dtype=np.float32))
            label_list.append(np.asarray(labels, dtype=np.int64))
            session_list.append(np.full(len(labels), ses, dtype=np.int64))

        if not data_list:
            raise ValueError(f'No SHU trials loaded for {sub} (sessions={sessions})')

        data = np.concatenate(data_list, axis=0)
        labels = np.concatenate(label_list)
        self.trial_sessions = np.concatenate(session_list)

        print(
            f'[SHU sub-{self.subject_id:03d}] '
            f'Loaded {len(labels)} trials from {len(data_list)} session(s), '
            f'shape={data.shape}, labels={np.unique(labels, return_counts=True)}'
        )

        return data, labels
