"""
pointcloud/metric_tensor.py – Riemannian metric tensor for the point-cloud manifold
====================================================================================
The metric tensor G at a base point P encodes how infinitesimal displacements
of the point cloud change the geodesic distance.

Mathematical background
-----------------------
For a point cloud P = {p₁, …, pₙ} ⊂ ℝ³ the metric tensor is a block matrix
G ∈ ℝ^{n×n×3×3}, where block (i, j) captures the coupling between
perturbations of point i and point j.

It has two additive contributions:

  Shape term (pairwise distances)
  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  For each pair (i, j) with i < j:

      G[i,i] += (pᵢ − pⱼ)(pᵢ − pⱼ)ᵀ / ‖pᵢ − pⱼ‖⁴
      G[j,j] += (pᵢ − pⱼ)(pᵢ − pⱼ)ᵀ / ‖pᵢ − pⱼ‖⁴
      G[i,j] −= (pᵢ − pⱼ)(pᵢ − pⱼ)ᵀ / ‖pᵢ − pⱼ‖⁴
      G[j,i] −= (pᵢ − pⱼ)(pᵢ − pⱼ)ᵀ / ‖pᵢ − pⱼ‖⁴

  Size term (gyration determinant), weighted by delta
  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  For all pairs (i, j):

      G[i,j] += 4δ · (G_P⁻¹ p̃ᵢ) ⊗ (G_P⁻¹ p̃ⱼ)

  where p̃ᵢ = pᵢ − centroid(P) are centred coordinates and G_P is the
  gyration matrix.

The metric tensor is used by ``norm`` to compute tangent-vector norms, and by
``log`` to project the pre-log map onto the tangent space.

Public API
----------
metric_tensor(P, delta) -> np.ndarray
    Batched metric tensor computation.
"""

import numba as nb
import numpy as np

from .utils import _inv3x3, _offsets, _gyration_matrix


# ---------------------------------------------------------------------------
# Private helpers (Numba-compiled, not part of the public API)
# ---------------------------------------------------------------------------

@nb.njit(nogil=True)
def _metric_tensor(
    P: np.ndarray,
    P_offsets: np.ndarray,
    P_gyr: np.ndarray,
    P_gyr_inv: np.ndarray,
    delta: float,
    res: np.ndarray,
):
    """Compute the metric tensor for a single point cloud in-place.

    Parameters
    ----------
    P : np.ndarray, shape (n_dim, 3)
        Input point cloud (raw, not centred).
    P_offsets : np.ndarray, shape (n_dim, 3)
        Working buffer for the centred point cloud.
    P_gyr : np.ndarray, shape (3, 3)
        Working buffer for the gyration matrix.
    P_gyr_inv : np.ndarray, shape (3, 3)
        Working buffer for the inverse gyration matrix.
    delta : float
        Weight of the size (gyration) contribution.
    res : np.ndarray, shape (n_dim, n_dim, 3, 3)
        Output buffer receiving the metric tensor blocks.
    """
    n_dim = P.shape[0]

    # Pre-compute centred coordinates, gyration matrix, and its inverse
    _offsets(P, P_offsets)
    _gyration_matrix(P_offsets, P_gyr)
    _inv3x3(P_gyr, P_gyr_inv)

    # --- Size term: 4δ (G⁻¹ p̃ᵢ) ⊗ (G⁻¹ p̃ⱼ) ---
    res[:] = 0
    for i in range(n_dim):
        zi = np.dot(P_gyr_inv, P_offsets[i])   # G_P⁻¹ p̃ᵢ
        res[i, i] += 4 * delta * np.outer(zi, zi)

        for j in range(i + 1, n_dim):
            zj    = np.dot(P_gyr_inv, P_offsets[j])
            outer = np.outer(zi, zj)
            res[i, j] += 4 * delta * outer     # upper off-diagonal
            res[j, i] += 4 * delta * outer     # lower off-diagonal (symmetric)

    # --- Shape term: (pᵢ−pⱼ)(pᵢ−pⱼ)ᵀ / ‖pᵢ−pⱼ‖⁴ ---
    for i in range(n_dim):
        for j in range(i + 1, n_dim):
            d        = P[i] - P[j]
            d_norm_sq = d[0]*d[0] + d[1]*d[1] + d[2]*d[2]
            d        /= d_norm_sq              # → d / ‖d‖²
            outer    = np.outer(d, d)          # → d dᵀ / ‖d‖⁴
            res[i, i] += outer
            res[j, j] += outer
            res[i, j] -= outer
            res[j, i] -= outer


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

@nb.njit(nogil=True, parallel=True)
def metric_tensor(P: np.ndarray, delta: float):
    """Compute the Riemannian metric tensor at each point cloud in a batch.

    Parameters
    ----------
    P : np.ndarray, shape (..., n_dim, 3)
        Batch of point clouds.  A single cloud has shape (n_dim, 3).
    delta : float
        Weight of the gyration-matrix (size) contribution to the metric.

    Returns
    -------
    np.ndarray, shape (..., n_dim, n_dim, 3, 3)
        Metric tensor blocks.  ``result[..., i, j]`` is the 3×3 coupling
        matrix between perturbations of point i and point j.

    Notes
    -----
    The returned tensor is symmetric in (i, j) and positive semi-definite.
    It has exactly 6 zero eigenvalues corresponding to the 6 rigid-body
    degrees of freedom (3 translations + 3 rotations).
    """
    n_dim = P.shape[-2]

    if P.ndim == 2:
        batch_shape = np.array((1,), dtype=np.int64)
    else:
        batch_shape = P.shape[:-2]

    batch_size = 1
    for s in batch_shape:
        batch_size *= s

    flat_P         = P.reshape(batch_size, n_dim, 3)
    flat_P_offsets = np.empty((batch_size, n_dim, 3),         dtype=P.dtype)
    flat_P_gyr     = np.empty((batch_size, 3, 3),             dtype=P.dtype)
    flat_P_gyr_inv = np.empty((batch_size, 3, 3),             dtype=P.dtype)
    flat_res       = np.empty((batch_size, n_dim, n_dim, 3, 3))

    for i in nb.prange(batch_size):
        _metric_tensor(
            flat_P[i], flat_P_offsets[i], flat_P_gyr[i], flat_P_gyr_inv[i],
            delta, flat_res[i],
        )

    return flat_res.reshape(P.shape[:-2] + (n_dim, n_dim, 3, 3))