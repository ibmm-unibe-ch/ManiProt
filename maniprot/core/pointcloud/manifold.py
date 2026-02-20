"""
pointcloud/manifold.py – Scikit-learn-compatible wrapper for the point-cloud manifold
======================================================================================
Provides a high-level ``PointcloudManifold`` class that follows the
scikit-learn ``fit`` / ``transform`` API.  Internally it wraps
``pointcloud.intrinsic_mean`` and ``pointcloud.log``.

Typical workflow
----------------
1. **Fit**: Given a set of protein backbone point clouds (one per trajectory
   frame), compute the Fréchet mean — the manifold point that minimises the
   average squared geodesic distance to all samples.

2. **Transform**: Map each point cloud to a tangent vector at the Fréchet mean
   via the logarithmic map.  The resulting vectors lie in a Euclidean space and
   can be consumed by downstream models (PCA, clustering, neural networks, etc.).

Example
-------
>>> import numpy as np
>>> from maniprot.pointcloud import PointcloudManifold

>>> # 200 frames of a 50-point Cα backbone
>>> X = np.random.randn(200, 50, 3).astype(np.float64)

>>> manifold = PointcloudManifold(delta=1.0)
>>> manifold.fit(X)                   # computes the Fréchet mean
>>> X_tangent = manifold.transform(X) # shape (200, 50, 3) Euclidean vectors
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from .log import log
from .intrinsic_mean import intrinsic_mean


class PointcloudManifold:
    """Riemannian point-cloud manifold with a fit/transform interface.

    Represents a protein backbone (or any labelled set of 3-D points) as an
    element of a Riemannian manifold.  After fitting, the manifold stores a
    reference point (Fréchet mean) and can project new samples into the
    corresponding tangent space via the logarithmic map.

    Parameters
    ----------
    delta : float
        Non-negative scalar weighting the gyration-matrix (size/volume) term
        versus the pairwise-distance (shape) term in the manifold metric.
        Setting ``delta=0`` yields a purely shape-based metric; larger values
        emphasise overall size.
    mean : np.ndarray of shape (n_dim, 3), optional
        Pre-computed Fréchet mean.  If provided the manifold is treated as
        already fitted and ``transform`` can be called immediately.

    Attributes
    ----------
    mean : np.ndarray, shape (n_dim, 3)
        The Fréchet mean point cloud (available after calling ``fit`` or
        setting ``mean`` at construction time).
    ndim : int
        Number of points per point cloud (read-only; derived from ``mean``).
    """

    def __init__(
        self,
        delta: float,
        *,
        mean: Optional[np.ndarray] = None,
    ):
        self.delta = delta
        self.mean  = mean   # triggers the setter for validation

    # ------------------------------------------------------------------
    # mean property – validated setter / lazy getter
    # ------------------------------------------------------------------

    @property
    def mean(self) -> np.ndarray:
        """Fréchet mean of the fitted dataset.

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
            if value.ndim != 2 or value.shape[-1] != 3:
                raise ValueError(
                    f"Invalid shape: {value.shape}.  Expected (n_dim, 3)."
                )
            self._mean = value
        else:
            raise ValueError(f"Invalid type: {type(value)} (expected np.ndarray or None)")

    @property
    def ndim(self) -> int:
        """Number of points per point cloud (inferred from the mean)."""
        return self.mean.shape[0]

    # ------------------------------------------------------------------
    # fit / transform
    # ------------------------------------------------------------------

    def fit(
        self,
        X: np.ndarray,
        est: Optional[np.ndarray] = None,
        *,
        learning_rate: float = 1e-3,
        threshold: float = 1e-3,
        max_steps: int = 128,
    ) -> "PointcloudManifold":
        """Compute the Fréchet mean of a dataset of point clouds.

        Parameters
        ----------
        X : np.ndarray, shape (n_samples, n_dim, 3)
            Dataset of point clouds.  Each slice ``X[i]`` is one sample.
        est : np.ndarray of shape (n_dim, 3), optional
            Initial estimate for the mean.  Defaults to ``X[0]`` if not given.
        learning_rate : float, optional
            Gradient-descent step size (default: 1e-3).
        threshold : float, optional
            Convergence criterion on the Riemannian gradient norm (default: 1e-3).
        max_steps : int, optional
            Maximum gradient-descent iterations (default: 128).

        Returns
        -------
        self
            The fitted manifold instance (allows method chaining).

        Raises
        ------
        ValueError
            If ``X`` does not have shape (n_samples, n_dim, 3).
        """
        if X.ndim != 3 or X.shape[-1] != 3:
            raise ValueError("X must have shape (n_samples, n_dim, 3).")

        if est is None:
            est = X[0].copy()

        self.mean = intrinsic_mean(
            X, est, # type: ignore
            self.delta,
            learning_rate,
            threshold,
            max_steps
        )
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Map point clouds to tangent vectors at the Fréchet mean.

        Applies ``log_mean(X[i])`` to each sample, returning Euclidean
        displacement vectors in T_mean that can be used for downstream analysis.

        Parameters
        ----------
        X : np.ndarray, shape (n_samples, n_dim, 3)
            Batch of point clouds to embed.

        Returns
        -------
        np.ndarray, shape (n_samples, n_dim, 3)
            Tangent-space representation of each sample.  The 6 rigid-body
            components (3 translations + 3 rotations) are removed, leaving
            3*n_dim − 6 effective degrees of freedom.

        Raises
        ------
        RuntimeError
            If called before the manifold has been fitted.
        """
        return log(X, self.mean, self.delta)