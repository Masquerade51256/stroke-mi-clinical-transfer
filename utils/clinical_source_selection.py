"""
Clinical source-subject selection for LOSO transfer learning.

Given a target subject's clinical profile and a pool of source subjects,
select the K most clinically similar source subjects to use as training data.
"""

from typing import List, Optional, Dict, Any
import numpy as np
import pandas as pd


def _encode_location(loc: str) -> np.ndarray:
    """One-hot encode stroke location."""
    categories = ['Brainstem', 'Subcortical', 'Cortical', 'Mixed']
    vec = np.zeros(len(categories))
    if loc in categories:
        vec[categories.index(loc)] = 1.0
    return vec


def _clinical_distance(target: Dict[str, Any], source: Dict[str, Any],
                       location_weight: float = 1.0,
                       duration_weight: float = 1.0,
                       nihss_weight: float = 1.0) -> float:
    """
    Compute a weighted L1 distance between two clinical profiles.

    Features:
      - StrokeLocation_Category: one-hot, exact match distance
      - Duration_Category: binary (acute vs chronic)
      - NIHSS: normalized difference
    """
    loc_dist = np.sum(np.abs(_encode_location(target['location']) -
                             _encode_location(source['location'])))

    dur_dist = float(target['duration'] != source['duration'])

    nihss_diff = abs(target['nihss'] - source['nihss'])
    # Avoid division by zero; use dataset std if available, otherwise 1
    nihss_scale = target.get('nihss_std', 1.0)
    if nihss_scale == 0:
        nihss_scale = 1.0
    nihss_dist = nihss_diff / nihss_scale

    return (location_weight * loc_dist +
            duration_weight * dur_dist +
            nihss_weight * nihss_dist)


def load_clinical_metadata(metadata_path: str,
                           location_col: str = 'StrokeLocation_Category',
                           duration_col: str = 'Duration_Category',
                           nihss_col: str = 'NIHSS',
                           subject_id_col: str = 'subject_id') -> pd.DataFrame:
    """
    Load and normalize clinical metadata.

    Expects a CSV with columns:
      subject_id, StrokeLocation_Category, Duration_Category, NIHSS
    (e.g. results/stratified/stratified_analysis_detailed.csv)
    """
    if metadata_path.endswith('.tsv'):
        raise ValueError("Please provide a stratified CSV with pre-computed categories. "
                         "Use results/stratified/stratified_analysis_detailed.csv")

    df = pd.read_csv(metadata_path)
    # Handle BOM if present
    df.columns = [c.lstrip('\ufeff') for c in df.columns]

    # Ensure column names match expectations
    if subject_id_col not in df.columns and 'Subject' in df.columns:
        df[subject_id_col] = df['Subject']

    # Normalize NIHSS
    nihss_mean = df[nihss_col].mean()
    nihss_std = df[nihss_col].std()
    df['nihss_norm'] = (df[nihss_col] - nihss_mean) / (nihss_std if nihss_std > 0 else 1.0)
    df['nihss_std'] = nihss_std

    # Normalize duration to binary category if needed
    def _norm_duration(d):
        d = str(d)
        if 'hronic' in d or '>' in d:
            return 'Chronic(>3mo)'
        else:
            return 'Acute(≤3mo)'

    df['duration_norm'] = df[duration_col].apply(_norm_duration)

    return df


def select_source_subjects(target_subject_id: int,
                           candidate_source_ids: List[int],
                           metadata_df: pd.DataFrame,
                           k: int = 10,
                           location_weight: float = 1.0,
                           duration_weight: float = 1.0,
                           nihss_weight: float = 1.0,
                           require_same_location: bool = False,
                           subject_id_col: str = 'subject_id') -> List[int]:
    """
    Select top-K clinically similar source subjects for a target subject.

    Args:
        target_subject_id: ID of the LOSO target subject.
        candidate_source_ids: List of available source subject IDs.
        metadata_df: DataFrame from load_clinical_metadata.
        k: Number of source subjects to select.
        location_weight: Weight for stroke location matching.
        duration_weight: Weight for stroke duration matching.
        nihss_weight: Weight for NIHSS similarity.
        require_same_location: If True, only consider sources with the same location.
        subject_id_col: Column name for subject IDs.

    Returns:
        List of selected source subject IDs (sorted by similarity, best first).
    """
    df = metadata_df.set_index(subject_id_col)
    target = df.loc[target_subject_id]

    target_profile = {
        'location': target['StrokeLocation_Category'],
        'duration': target['duration_norm'],
        'nihss': target['NIHSS'],
        'nihss_std': target['nihss_std']
    }

    distances = []
    for sid in candidate_source_ids:
        if sid == target_subject_id:
            continue
        if sid not in df.index:
            continue
        src = df.loc[sid]
        if require_same_location and src['StrokeLocation_Category'] != target_profile['location']:
            continue

        src_profile = {
            'location': src['StrokeLocation_Category'],
            'duration': src['duration_norm'],
            'nihss': src['NIHSS'],
            'nihss_std': target['nihss_std']
        }
        dist = _clinical_distance(target_profile, src_profile,
                                  location_weight, duration_weight, nihss_weight)
        distances.append((sid, dist))

    if not distances:
        # Fallback: return first k candidates
        return candidate_source_ids[:k]

    distances.sort(key=lambda x: x[1])
    k = min(k, len(distances))
    return [sid for sid, _ in distances[:k]]
