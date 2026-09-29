import numpy as np
from preprocessing.base_processor import BaseProcessor


class EuclideanAlignmentProcessor(BaseProcessor):
    """
    Euclidean Alignment (EA) for EEG data.

    As proposed in He and Wu, IEEE TBME 2020, and revisited in
    Wu, "Revisiting Euclidean Alignment for Transfer Learning in
    EEG-Based Brain-Computer Interfaces", arXiv:2502.09203 (2025).

    For each domain (subject/band), EA computes the mean covariance
    matrix R_bar = (1/N) sum_n X_n @ X_n^T, and whitens each trial
    by tilde{X}_n = R_bar^{-1/2} X_n.

    This processor supports:
      - 3D input: (n_trials, n_channels, n_times)
      - 4D input: (n_trials, n_bands, n_channels, n_times)
        In the 4D case, EA is applied independently per band.
    """

    def __init__(self, name: str = "ea", eps: float = 1e-10):
        super().__init__()
        self.name = name
        self.eps = eps

    def process(self, data: np.ndarray, **kwargs) -> np.ndarray:
        """
        Apply Euclidean Alignment.

        Args:
            data: Input array of shape (n_trials, n_channels, n_times)
                  or (n_trials, n_bands, n_channels, n_times).

        Returns:
            Aligned data with the same shape as input.
        """
        if data.ndim == 3:
            return self._apply_ea(data)
        elif data.ndim == 4:
            aligned = np.zeros_like(data)
            n_bands = data.shape[1]
            for b in range(n_bands):
                aligned[:, b, :, :] = self._apply_ea(data[:, b, :, :])
            return aligned
        else:
            raise ValueError(
                f"EuclideanAlignmentProcessor expects 3D or 4D input, "
                f"got {data.ndim}D with shape {data.shape}"
            )

    def _apply_ea(self, data: np.ndarray) -> np.ndarray:
        """
        Apply EA to 3D data (n_trials, n_channels, n_times).

        Args:
            data: Array of shape (n_trials, n_channels, n_times).

        Returns:
            Aligned array of shape (n_trials, n_channels, n_times).
        """
        # Compute mean covariance matrix: R_bar = (1/N) * sum(X_n @ X_n^T)
        covs = np.einsum('tij,tkj->tik', data, data)  # (n_trials, ch, ch)
        r_bar = covs.mean(axis=0)  # (ch, ch)

        # Compute R_bar^{-1/2} via eigendecomposition
        r_inv_sqrt = self._matrix_inv_sqrt(r_bar)

        # Apply whitening: tilde{X}_n = R_bar^{-1/2} @ X_n
        aligned = np.einsum('ij,tjk->tik', r_inv_sqrt, data)
        return aligned

    def _matrix_inv_sqrt(self, mat: np.ndarray) -> np.ndarray:
        """
        Compute matrix^{-1/2} using symmetric eigendecomposition.

        Args:
            mat: Symmetric positive semi-definite matrix.

        Returns:
            mat^{-1/2}.
        """
        eigvals, eigvecs = np.linalg.eigh(mat)
        eigvals = np.clip(eigvals, a_min=self.eps, a_max=None)
        return eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T
