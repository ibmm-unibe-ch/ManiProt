"""
SO3/manifold.py – Scikit-learn-compatible wrapper for SO(3) geometry
=====================================================================
Provides a high-level ``SO3Manifold`` class that follows the scikit-learn
``fit`` / ``transform`` API for sets of rotation matrices.

Typical workflow
----------------
1. **Fit**: Given a batch of rotation matrices (e.g. per-residue LCS frames
   across trajectory frames), compute the Fréchet mean on SO(3).

2. **Transform**: Map each rotation matrix to a tangent vector at the Fréchet
   mean via the SO(3) logarithmic map.  The resulting 3-D (or n_dim × 3)
   vectors lie in a Euclidean space and can be used directly by downstream
   machine-learning models.

Example
-------
>>> import numpy as np
>>> from maniprot.SO3 import SO3Manifold

>>> # 200 frames, 50 residues, each with a 3×3 LCS frame
>>> R = np.random.randn(200, 50, 3, 3).astype(np.float32)
>>> # (in practice R should be proper rotation matrices from maniprot.helpers.lcs)

>>> manifold = SO3Manifold()
>>> manifold.fit(R)                   # computes the Fréchet mean per residue
>>> R_tangent = manifold.transform(R) # shape (200, 50, 3) – Euclidean vectors
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from .log import log
from .intrinsic_mean import intrinsic_mean


class SO3Manifold:
    """Riemannian manifold on SO(3) with a fit/transform interface.

    Fits a Fréchet mean rotation (or set of per-point rotations) from a
    dataset and subsequently projects new samples into the corresponding
    tangent space via the SO(3) logarithmic map.

    Parameters
    ----------
    mean : np.ndarray of shape (3, 3) or (n_dim, 3, 3), optional
        Pre-computed Fréchet mean rotation matrix (or matrices).  If provided
        the manifold is treated as already fitted.
    learning_rate : float, optional
        Gradient-descent step size used during fitting (default: 1e-2).
    threshold : float, optional
        Convergence criterion on the mean tangent-vector norm (default: 1e-4).
    max_steps : int, optional
        Maximum gradient-descent iterations during fitting (default: 128).

    Attributes
    ----------
    mean : np.ndarray
        The Fréchet mean rotation matrix / matrices (available after fitting).
    """

    def __init__(
        self,
        *,
        mean: Optional[np.ndarray] = None,
    ):
        self.mean = mean  # triggers the setter for validation

    # ------------------------------------------------------------------
    # mean property – validated setter / lazy getter
    # ------------------------------------------------------------------

    @property
    def mean(self) -> np.ndarray:
        """Fréchet mean rotation matrix (or matrices).

        Raises
        ------
        RuntimeError
            If accessed before the manifold has been fitted.
        """
        if self._mean is None:
            raise RuntimeError("Trying to access mean before the manifold has been fitted.")
        return self._mean

    @mean.setter
    def mean(self, value: Optional[np.ndarray]):
        if value is None:
            self._mean = None
        elif isinstance(value, np.ndarray):
            if value.shape[-2:] != (3, 3):
                raise ValueError(
                    f"Invalid shape: {value.shape}.  Expected (..., 3, 3)."
                )
            self._mean = value
        else:
            raise ValueError(f"Invalid type: {type(value)} (expected np.ndarray or None)")

    # ------------------------------------------------------------------
    # fit / transform
    # ------------------------------------------------------------------

    def fit(
        self,
        R: np.ndarray,
        est: Optional[np.ndarray] = None,
        *,
        learning_rate: float = 1e-2,
        threshold: float = 1e-4,
        max_steps: int = 128,
    ) -> "SO3Manifold":
        """Compute the Fréchet mean of a dataset of rotation matrices.

        Parameters
        ----------
        R : np.ndarray, shape (n_samples, n_dim, 3, 3)
            Dataset of rotation matrices.  ``n_dim`` is the number of
            independent rotation slots (e.g. one per residue); each slot
            converges to its own Fréchet mean.
        est : np.ndarray of shape (n_dim, 3, 3), optional
            Initial estimate for the mean.  Defaults to ``R[0]`` if not given.

        Returns
        -------
        self
            The fitted manifold (allows method chaining).

        Raises
        ------
        ValueError
            If ``R`` does not have shape ``(n_samples, n_dim, 3, 3)``.
        """
        if R.ndim != 4 or R.shape[-2:] != (3, 3):
            raise ValueError(f"R must have shape (n_samples, n_dim, 3, 3), got {R.shape}.")

        if est is None:
            est = R[0].copy()

        self.mean = intrinsic_mean(
            R, est, # type: ignore
            learning_rate,
            threshold,
            max_steps,
        )
        return self

    def transform(self, R: np.ndarray) -> np.ndarray:
        """Map rotation matrices to tangent vectors at the Fréchet mean.

        Applies ``log_mean(R[i])`` to each sample, returning Euclidean
        axis-angle vectors in T_mean ≅ ℝ³ (or ℝ^{n_dim × 3}).

        Parameters
        ----------
        R : np.ndarray, shape (n_samples, 3, 3) or (n_samples, n_dim, 3, 3)
            Batch of rotation matrices to embed.

        Returns
        -------
        np.ndarray, shape (n_samples, 3) or (n_samples, n_dim, 3)
            Tangent-space representation.  Each vector encodes the axis-angle
            displacement from ``mean`` to the corresponding rotation.

        Raises
        ------
        RuntimeError
            If called before the manifold has been fitted.
        """
        return log(R, self.mean)