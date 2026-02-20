"""
pointcloud/norm.py – Riemannian norm of tangent vectors on the point-cloud manifold
====================================================================================
Computes the norm of a tangent vector v ∈ T_P with respect to the Riemannian
metric G_P at a base point P:

    ‖v‖_P = √(vᵀ G_P v) = √( Σ_{i,j} vᵢᵀ G_P[i,j] vⱼ )

where G_P[i,j] is the 3×3 metric block coupling point i and point j.

This norm is used by ``intrinsic_mean`` to measure convergence of the
gradient descent: optimisation stops when ‖gradient‖_P < threshold.

Public API
----------
norm(P, P_ref, delta) -> np.ndarray
    Batched Riemannian norm computation.
"""

import numba as nb
import numpy as np

from .metric_tensor import metric_tensor


# ---------------------------------------------------------------------------
# Private helper (Numba-compiled, not part of the public API)
# ---------------------------------------------------------------------------

@nb.njit(nogil=True)
def _norm(P: np.ndarray, metric_tensor: np.ndarray):
    """Compute the Riemannian norm ‖P‖_G for a single tangent vector.

    Parameters
    ----------
    P : np.ndarray, shape (n_dim, 3)
        Tangent vector (displacement field) to measure.
    metric_tensor : np.ndarray, shape (n_dim, n_dim, 3, 3)
        Metric tensor evaluated at the base point P_ref.

    Returns
    -------
    float
        Riemannian norm √(Pᵀ G P).
    """
    n_dim = P.shape[0]
    acc   = 0.0

    for i in range(n_dim):
        pi = P[i]
        # Diagonal block contribution: vᵢᵀ G[i,i] vᵢ
        acc += (metric_tensor[i, i] * np.outer(pi, pi)).sum()
        for j in range(i + 1, n_dim):
            pj = P[j]
            # Off-diagonal contributions (both sides due to symmetry of G)
            acc += (metric_tensor[i, j] * np.outer(pi, pj)).sum()
            acc += (metric_tensor[j, i] * np.outer(pj, pi)).sum()

    return np.sqrt(acc)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

@nb.njit(nogil=True, parallel=True)
def norm(P: np.ndarray, P_ref: np.ndarray, delta: float):
    """Compute the batched Riemannian norm of tangent vectors at P_ref.

    Parameters
    ----------
    P : np.ndarray, shape (..., n_dim, 3)
        Batch of tangent vectors (e.g. the gradient returned by ``log``).
    P_ref : np.ndarray, shape (n_dim, 3)
        Base point at which the metric tensor is evaluated.
    delta : float
        Weight of the gyration-matrix (size) term in the metric.

    Returns
    -------
    np.ndarray, shape (...,)
        Riemannian norms, one per sample in the batch.

    Examples
    --------
    >>> grad = -np.mean(log(P_batch, mean, delta), axis=0)
    >>> error = norm(grad[None], mean, delta)[0]
    >>> print(f"Gradient norm: {error:.6f}")
    """
    n_dim = P.shape[-2]

    if P.ndim == 2:
        batch_shape = np.array((1,), dtype=np.int64)
    else:
        batch_shape = P.shape[:-2]

    batch_size = 1
    for s in batch_shape:
        batch_size *= s

    flat_P   = P.reshape(batch_size, n_dim, 3)
    flat_res = np.empty((batch_size,), dtype=P.dtype)

    # Metric tensor is shared across all batch elements (single P_ref)
    P_ref_metric_tensor = metric_tensor(P_ref, delta)

    for i in nb.prange(batch_size):
        flat_res[i] = _norm(flat_P[i], P_ref_metric_tensor)

    return flat_res.reshape(P.shape[:-2])