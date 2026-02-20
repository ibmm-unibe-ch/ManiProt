"""
pointcloud/distance.py – Geodesic distance on the point-cloud manifold
=======================================================================
Implements the geodesic distance between two point clouds P and Q as elements
of the Riemannian point-cloud manifold parameterised by ``delta``.

Mathematical background
-----------------------
Given two point clouds  P = {p₁, …, pₙ}  and  Q = {q₁, …, qₙ}  in ℝ³,
the (squared) manifold distance is:

    d²(P, Q) = Σ_{j<k} [ log(‖pⱼ − pₖ‖ / ‖qⱼ − qₖ‖) ]²
             + δ · [ log(det G_P / det G_Q) ]²

where G_P and G_Q are the gyration matrices of P and Q respectively, and δ is
a non-negative scalar weighting the "size" (gyration) component relative to
the "shape" (pairwise distances) component.

This distance is invariant to rigid-body transformations (translations and
rotations) of the point clouds.

Public API
----------
distance(P, Q, delta) -> np.ndarray
    Batched pairwise geodesic distance computation.
"""

import numba as nb
import numpy as np

from .utils import _det3x3, _offsets, _gyration_matrix


# ---------------------------------------------------------------------------
# Private helpers (Numba-compiled, not part of the public API)
# ---------------------------------------------------------------------------

@nb.njit(nogil=True)
def _distance(
    P: np.ndarray,
    Q: np.ndarray,
    delta: float,
    P_offsets: np.ndarray,
    P_gyr: np.ndarray,
    Q_offsets: np.ndarray,
    Q_gyr: np.ndarray,
):
    """Compute the geodesic distance between a single pair (P, Q).

    Parameters
    ----------
    P, Q : np.ndarray, shape (n_dim, 3)
        Point clouds to compare (must share the same number of points n_dim).
    delta : float
        Weight of the gyration-matrix (size) term.  Setting delta=0 yields a
        purely shape-based metric.
    P_offsets, Q_offsets : np.ndarray, shape (n_dim, 3)
        Pre-allocated buffers for the centred point clouds.
    P_gyr, Q_gyr : np.ndarray, shape (3, 3)
        Pre-allocated buffers for the gyration matrices.

    Returns
    -------
    float
        Geodesic distance d(P, Q) ≥ 0.
    """
    n_dim = P.shape[0]

    # Gyration matrices for the size term
    _offsets(P, P_offsets);  _gyration_matrix(P_offsets, P_gyr)
    _offsets(Q, Q_offsets);  _gyration_matrix(Q_offsets, Q_gyr)

    # Accumulate squared log-ratio of all pairwise distances (shape term)
    acc = 0.0
    for i in range(n_dim):
        for j in range(i + 1, n_dim):
            p0, p1, p2 = P[i] - P[j]
            q0, q1, q2 = Q[i] - Q[j]
            acc += np.log(
                np.sqrt(p0*p0 + p1*p1 + p2*p2) /
                np.sqrt(q0*q0 + q1*q1 + q2*q2)
            ) ** 2

    # Add gyration-determinant (size) term
    acc += delta * np.log(_det3x3(P_gyr) / _det3x3(Q_gyr)) ** 2

    # Numerical guard against tiny negative values due to floating-point errors
    if acc < 0:
        return 0.0
    return np.sqrt(acc)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

@nb.njit(nogil=True, parallel=True)
def distance(P: np.ndarray, Q: np.ndarray, delta: float):
    """Compute batched pairwise geodesic distances on the point-cloud manifold.

    Supports broadcasting: P and Q can each be a single point cloud or a batch.
    The result has shape ``P.shape[:-2] + Q.shape[:-2]``, i.e. every element
    of the P batch is compared against every element of the Q batch.

    Parameters
    ----------
    P : np.ndarray, shape (..., n_dim, 3)
        First batch of point clouds.  A single cloud has shape (n_dim, 3).
    Q : np.ndarray, shape (..., n_dim, 3)
        Second batch of point clouds.
    delta : float
        Weight of the gyration-matrix (size) term.

    Returns
    -------
    np.ndarray, shape (P_batch_shape + Q_batch_shape)
        Pairwise distances. ``result[i, j] = d(P[i], Q[j])``.

    Examples
    --------
    >>> # Distance between two single clouds
    >>> d = distance(P, Q, delta=1.0)            # scalar

    >>> # All pairwise distances between two batches
    >>> D = distance(P_batch, Q_batch, delta=1.0) # shape (n_P, n_Q)
    """
    n_dim = P.shape[-2]

    if P.ndim == 2:
        P_batch_shape = np.array((1,), dtype=np.int64)
    else:
        P_batch_shape = P.shape[:-2]

    if Q.ndim == 2:
        Q_batch_shape = np.array((1,), dtype=np.int64)
    else:
        Q_batch_shape = Q.shape[:-2]

    P_batch_size = 1
    for s in P_batch_shape:
        P_batch_size *= s

    Q_batch_size = 1
    for s in Q_batch_shape:
        Q_batch_size *= s

    flat_P = P.reshape(P_batch_size, n_dim, 3)
    flat_Q = Q.reshape(Q_batch_size, n_dim, 3)

    # Pre-allocate working buffers for the parallel loop
    flat_P_offsets = np.empty((P_batch_size * Q_batch_size, n_dim, 3), dtype=P.dtype)
    flat_P_gyr     = np.empty((P_batch_size * Q_batch_size, 3, 3),     dtype=P.dtype)
    flat_Q_offsets = np.empty((P_batch_size * Q_batch_size, n_dim, 3), dtype=P.dtype)
    flat_Q_gyr     = np.empty((P_batch_size * Q_batch_size, 3, 3),     dtype=P.dtype)
    res            = np.empty((P_batch_size * Q_batch_size,),           dtype=P.dtype)

    for i in nb.prange(P_batch_size * Q_batch_size):
        j = i // Q_batch_size   # index into P
        k = i  % Q_batch_size   # index into Q
        res[i] = _distance(
            flat_P[j], flat_Q[k], delta,
            flat_P_offsets[i], flat_P_gyr[i],
            flat_Q_offsets[i], flat_Q_gyr[i],
        )

    return res.reshape(P.shape[:-2] + Q.shape[:-2])