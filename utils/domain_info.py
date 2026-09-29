"""
Domain label management for domain adaptation experiments.

Provides utilities to map subjects to domain labels based on clinical metadata
(e.g., stroke location, paralysis side, NIHSS severity).
"""

import os
from typing import Dict, Optional
import pandas as pd


def classify_stroke_location(loc_str: str) -> str:
    """
    Classify stroke location into broad categories.
    Categories: Cortical, Subcortical, Brainstem, Cerebellar, Mixed, Other
    """
    if pd.isna(loc_str):
        return 'Unknown'

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
        return 'Mixed'
    elif has_cortical:
        return 'Cortical'
    elif has_subcortical:
        return 'Subcortical'
    elif has_brainstem:
        return 'Brainstem'
    elif has_cerebellar:
        return 'Cerebellar'
    else:
        return 'Other'


class DomainInfoManager:
    """
    Manager for subject-to-domain mappings.

    Supports multiple domain grouping strategies:
    - stroke_location_4class: Subcortical, Brainstem, Cortical, Mixed
    - stroke_location_2class: Subcortical vs Non-subcortical
    - paralysis_side: Left vs Right
    - nihss_3class: Mild(1-3), Moderate(4-7), Severe(>=8)
    """

    SUPPORTED_STRATEGIES = [
        'stroke_location_4class',
        'stroke_location_2class',
        'paralysis_side',
        'nihss_3class'
    ]

    def __init__(self, participants_tsv: str, strategy: str = 'stroke_location_4class'):
        """
        Initialize domain info manager.

        Args:
            participants_tsv: Path to participants.tsv file
            strategy: Domain grouping strategy
        """
        if strategy not in self.SUPPORTED_STRATEGIES:
            raise ValueError(
                f"Unknown strategy: {strategy}. Supported: {self.SUPPORTED_STRATEGIES}"
            )

        self.strategy = strategy
        self.participants_df = pd.read_csv(participants_tsv, sep='\t')
        self.domain_map = self._build_domain_map()
        self.num_domains = len(set(self.domain_map.values()))

    def _build_domain_map(self) -> Dict[int, int]:
        """Build subject_id -> domain_id mapping."""
        domain_map = {}

        for _, row in self.participants_df.iterrows():
            participant_id = row['Participant_ID']  # e.g., 'sub-01'
            subject_id = int(participant_id.split('-')[1])

            if self.strategy == 'stroke_location_4class':
                loc = classify_stroke_location(row['StrokeLocation'])
                # Map to domain IDs: Subcortical=0, Brainstem=1, Cortical=2, Mixed=3
                loc_to_id = {'Subcortical': 0, 'Brainstem': 1, 'Cortical': 2, 'Mixed': 3}
                domain_id = loc_to_id.get(loc, 0)

            elif self.strategy == 'stroke_location_2class':
                loc = classify_stroke_location(row['StrokeLocation'])
                domain_id = 0 if loc == 'Subcortical' else 1

            elif self.strategy == 'paralysis_side':
                side = str(row['ParalysisSide']).lower()
                domain_id = 0 if side == 'left' else 1

            elif self.strategy == 'nihss_3class':
                nihss = int(row['NIHSS'])
                if nihss <= 3:
                    domain_id = 0  # Mild
                elif nihss <= 7:
                    domain_id = 1  # Moderate
                else:
                    domain_id = 2  # Severe

            domain_map[subject_id] = domain_id

        return domain_map

    def get_domain_id(self, subject_id: int) -> int:
        """Get domain ID for a subject."""
        return self.domain_map.get(subject_id, 0)

    def get_domain_distribution(self) -> Dict[int, int]:
        """Get count of subjects per domain."""
        from collections import Counter
        return dict(Counter(self.domain_map.values()))

    def __repr__(self):
        return (
            f"DomainInfoManager(strategy={self.strategy}, "
            f"num_domains={self.num_domains}, "
            f"dist={self.get_domain_distribution()})"
        )


def get_domain_manager(dataset_info: dict, strategy: Optional[str] = None) -> DomainInfoManager:
    """
    Convenience factory to create DomainInfoManager from dataset config.

    Args:
        dataset_info: Dataset configuration dict
        strategy: Domain grouping strategy (overrides config if provided)

    Returns:
        DomainInfoManager instance
    """
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

    if participants_tsv is None or not os.path.exists(participants_tsv):
        raise FileNotFoundError(f"participants.tsv not found. Tried: {candidates}")

    if strategy is None:
        strategy = dataset_info.get('dataset', {}).get(
            'domain_grouping', 'stroke_location_4class'
        )

    return DomainInfoManager(participants_tsv, strategy)
