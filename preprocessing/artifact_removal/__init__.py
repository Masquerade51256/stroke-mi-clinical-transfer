"""Artifact removal processors for EEG preprocessing."""

from preprocessing.artifact_removal.emar import EMARProcessor
from preprocessing.artifact_removal.sasica import SASICAProcessor
from preprocessing.artifact_removal.mara import MARAProcessor
from preprocessing.artifact_removal_legacy import ArtifactRemovalProcessor

__all__ = ['EMARProcessor', 'SASICAProcessor', 'MARAProcessor', 'ArtifactRemovalProcessor']
