"""
pointcloud/intrinsic_mean.py – Fréchet mean on the point-cloud manifold
========================================================================
Computes the Fréchet mean of a set of point clouds on the Riemannian
point-cloud manifold using gradient descent in the tangent space.

Mathematical background
-----------------------
The Fréchet mean μ minimises the sum of squared geodesic distances to all
samples {P₁, …, Pₙ}:

    μ* = argmin_μ  Σᵢ d(Pᵢ, μ)²

On the point-cloud manifold this is solved iteratively:

    gradient = −mean_i [ log_μ(Pᵢ) ]    ← average tangent vector at μ
    μ ← μ − lr · gradient               ← Euclidean coordinate update

Convergence is measured by the Riemannian norm of the gradient at μ:
the loop terminates when ‖gradient‖_μ < threshold.

Note: the update step is a direct Euclidean move in ℝ^{n×3} (not a retraction
onto the manifold).  This is valid for small learning rates where the manifold
curvature is locally negligible.

Public API
----------
intrinsic_mean(P, P_ref, delta, ...) -> np.ndarray
    Gradient-descent Fréchet mean on the point-cloud manifold.
"""

import numpy as np

from .norm import norm
from .log import log


def intrinsic_mean(
    P: np.ndarray,
    P_ref: np.ndarray,
    delta: float,
    learning_rate: float = 1e-3,
    threshold: float = 1e-3,
    max_steps: int = 128,
) -> np.ndarray:
    """Compute the Fréchet mean of a batch of point clouds on the manifold.

    Uses Riemannian gradient descent in the tangent space to find the point
    cloud that minimises the average squared geodesic distance to all samples.

    Parameters
    ----------
    P : np.ndarray, shape (n_samples, n_dim, 3)
        Batch of point clouds whose intrinsic mean is sought.
    P_ref : np.ndarray, shape (n_dim, 3)
        Initial estimate of the mean (e.g. the first sample ``P[0]``).
    delta : float
        Weight of the gyration-matrix (size) term in the manifold metric.
    learning_rate : float, optional
        Step size for gradient descent (default: 1e-3).
    threshold : float, optional
        Convergence criterion: stop when the Riemannian gradient norm falls
        below this value (default: 1e-3).
    max_steps : int, optional
        Maximum number of gradient-descent iterations (default: 128).

    Returns
    -------
    np.ndarray, shape (n_dim, 3)
        The estimated Fréchet mean point cloud.

    Notes
    -----
    Convergence is guaranteed in a geodesic ball of sufficient radius around
    the true mean.  For highly dispersed datasets, try a smaller
    ``learning_rate`` or more ``max_steps``.
    """
    mean = P_ref.copy()

    for _ in range(max_steps):
        # Gradient: negative mean of log-maps (tangent vectors toward samples)
        grad = -np.mean(log(P, mean, delta), axis=0)

        # Convergence check via the Riemannian norm at the current mean
        error = norm(grad[None], mean, delta)[0]
        if error < threshold:
            break

        # Euclidean coordinate update (approximate retraction for small steps)
        mean -= learning_rate * grad

    return mean