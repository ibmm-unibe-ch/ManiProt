"""
pointcloud/log.py – Logarithmic map on the point-cloud manifold
================================================================
The logarithmic map at a base point P_ref maps a nearby point cloud P into the
tangent space T_{P_ref} of the manifold.  The result is a displacement field
(a 3-D vector per point) that can be treated as a Euclidean vector.

Mathematical background
-----------------------
The computation proceeds in two stages:

1. **Pre-log map** (``prelog``)
   Computes a raw gradient-like vector field  v ∈ ℝ^{n×3}  as the derivative
   of the log-distance from P to P_ref:

       v[i] = 2δ · log(det G_{P_ref} / det G_P) · (G_{P_ref}⁻¹ p̃_ref_i)
            + Σ_{k≠i} log(‖p_ref_i − p_ref_k‖ / ‖pᵢ − pₖ‖) ·
                       (p_ref_i − p_ref_k) / ‖p_ref_i − p_ref_k‖²

   This vector lies in ℝ^{n×3} but retains translational and rotational
   components (it is *not* yet in the tangent space).

2. **Projection onto the tangent space** (``log``)
   The metric tensor G at P_ref has a 6-dimensional null space corresponding
   to rigid-body motions.  ``log`` removes these components via
   eigendecomposition and computes the pseudo-inverse projection:

       log_map(P) = −Q_perp · diag(1/λ_perp) · Q_perp^T · v

   where Q_perp and λ_perp are the non-null eigenvectors/values of G
   (i.e. the 6 smallest eigenvalues are discarded).

Public API
----------
prelog(P, P_ref, delta) -> np.ndarray
    Batched pre-log displacement field.
log(P, P_ref, delta) -> np.ndarray
    Batched logarithmic map (tangent-space embedding).
"""

import numba as nb
import numpy as np

from .utils import _det3x3, _inv3x3, _offsets, _gyration_matrix
from .metric_tensor import metric_tensor


# ---------------------------------------------------------------------------
# Private helpers (Numba-compiled, not part of the public API)
# ---------------------------------------------------------------------------

@nb.njit(nogil=True)
def _A_components(P: np.ndarray, precoef: np.ndarray, components: np.ndarray):
    """Pre-compute pairwise distance coefficients for the shape term of prelog.

    Fills *precoef* with ‖pᵢ − pⱼ‖ and *components* with the normalised
    direction (pᵢ − pⱼ) / ‖pᵢ − pⱼ‖² for all upper-triangular pairs (i < j).

    Parameters
    ----------
    P : np.ndarray, shape (n_dim, 3)
        Reference point cloud.
    precoef : np.ndarray, shape (n_pairs,)
        Output: pairwise distances ‖pᵢ − pⱼ‖.
    components : np.ndarray, shape (n_pairs, 3)
        Output: (pᵢ − pⱼ) / ‖pᵢ − pⱼ‖² for each pair.
    """
    n_dim = P.shape[0]
    i = 0
    for j in range(n_dim):
        for k in range(j + 1, n_dim):
            d0, d1, d2  = P[j] - P[k]
            d_norm_sq   = d0*d0 + d1*d1 + d2*d2
            precoef[i]       = np.sqrt(d_norm_sq)
            components[i, 0] = d0 / d_norm_sq
            components[i, 1] = d1 / d_norm_sq
            components[i, 2] = d2 / d_norm_sq
            i += 1


@nb.njit(nogil=True)
def _B_components(
    P: np.ndarray,
    P_offsets: np.ndarray,
    P_gyr: np.ndarray,
    P_gyr_inv: np.ndarray,
    components: np.ndarray,
):
    """Pre-compute per-point size-term components for the prelog of P_ref.

    Fills *components* with G_P⁻¹ p̃ᵢ for each centred point p̃ᵢ.

    Parameters
    ----------
    P : np.ndarray, shape (n_dim, 3)
        Reference point cloud.
    P_offsets, P_gyr, P_gyr_inv : np.ndarray
        Working buffers (written in-place).
    components : np.ndarray, shape (n_dim, 3)
        Output: G_P⁻¹ · (pᵢ − centroid) for each point i.
    """
    _offsets(P, P_offsets)
    _gyration_matrix(P_offsets, P_gyr)
    _inv3x3(P_gyr, P_gyr_inv)

    n_dim = P.shape[0]
    for i in range(n_dim):
        components[i] = np.dot(P_gyr_inv, P_offsets[i])


@nb.njit(nogil=True)
def _prelog(
    P: np.ndarray,
    P_offsets: np.ndarray,
    P_gyr: np.ndarray,
    A_precoef: np.ndarray,
    A_components: np.ndarray,
    B_precoef: float,
    B_components: np.ndarray,
    delta: float,
    res: np.ndarray,
):
    """Compute the pre-log displacement field for a single point cloud.

    Parameters
    ----------
    P : np.ndarray, shape (n_dim, 3)
        Target point cloud.
    P_offsets : np.ndarray, shape (n_dim, 3)
        Working buffer for the centred P.
    P_gyr : np.ndarray, shape (3, 3)
        Working buffer for the gyration matrix of P.
    A_precoef : np.ndarray, shape (n_pairs,)
        Reference pairwise distances ‖p_ref_i − p_ref_k‖ (pre-computed).
    A_components : np.ndarray, shape (n_pairs, 3)
        Reference directional components (p_ref_i−p_ref_k)/‖·‖² (pre-computed).
    B_precoef : float
        det(G_{P_ref}) (pre-computed).
    B_components : np.ndarray, shape (n_dim, 3)
        G_{P_ref}⁻¹ p̃_ref_i vectors (pre-computed).
    delta : float
        Weight of the size (gyration) term.
    res : np.ndarray, shape (n_dim, 3)
        Output pre-log displacement field.
    """
    n_dim = P.shape[0]

    _offsets(P, P_offsets)
    _gyration_matrix(P_offsets, P_gyr)

    # Size term: 2δ · log(det G_ref / det G_P) · G_ref⁻¹ p̃_ref_i
    res[:] = 2 * delta * np.log(B_precoef / _det3x3(P_gyr)) * B_components

    # Shape term: accumulate log(‖p_ref_ij‖ / ‖p_ij‖) contributions
    i = 0
    for j in range(n_dim):
        for k in range(j + 1, n_dim):
            d0, d1, d2 = P[j] - P[k]
            d_norm     = np.sqrt(d0*d0 + d1*d1 + d2*d2)
            coeff      = np.log(A_precoef[i] / d_norm)
            # Opposite sign for j and k from the antisymmetry of the gradient
            res[j] += coeff * A_components[i]
            res[k] -= coeff * A_components[i]
            i += 1


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

@nb.njit(nogil=True, parallel=True)
def prelog(P: np.ndarray, P_ref: np.ndarray, delta: float):
    """Compute the batched pre-log displacement field relative to P_ref.

    The pre-log field is the raw gradient of the squared distance from P_ref.
    It lives in ℝ^{n×3} but retains rigid-body components.  For the full
    tangent-space projection use ``log``.

    Parameters
    ----------
    P : np.ndarray, shape (batch_size, n_dim, 3)
        Batch of target point clouds.
    P_ref : np.ndarray, shape (n_dim, 3)
        Reference (base) point cloud.
    delta : float
        Weight of the gyration-matrix (size) term.

    Returns
    -------
    np.ndarray, shape (batch_size, n_dim, 3)
        Pre-log displacement field for each sample.
    """
    batch_size = P.shape[0]
    n_dim      = P.shape[1]
    n_pairs    = n_dim * (n_dim - 1) // 2

    P_offsets = np.empty((batch_size, n_dim, 3), dtype=P.dtype)
    P_gyr     = np.empty((batch_size, 3, 3),     dtype=P.dtype)

    # Pre-compute shape-term coefficients from P_ref (shared across batch)
    A_precoef    = np.empty((n_pairs,),    dtype=P.dtype)
    A_components = np.empty((n_pairs, 3), dtype=P.dtype)
    _A_components(P_ref, A_precoef, A_components)

    # Pre-compute size-term coefficients from P_ref (shared across batch)
    P_ref_offsets = np.empty((n_dim, 3), dtype=P.dtype)
    P_ref_gyr     = np.empty((3, 3),     dtype=P.dtype)
    P_ref_gyr_inv = np.empty((3, 3),     dtype=P.dtype)
    B_components  = np.empty((n_dim, 3), dtype=P.dtype)
    _B_components(P_ref, P_ref_offsets, P_ref_gyr, P_ref_gyr_inv, B_components)
    B_precoef = _det3x3(P_ref_gyr)

    prelog_out = np.empty((batch_size, n_dim, 3), dtype=P.dtype)

    for i in nb.prange(batch_size):
        _prelog(
            P[i], P_offsets[i], P_gyr[i],
            A_precoef, A_components,
            B_precoef, B_components,
            delta, prelog_out[i],
        )

    return prelog_out


def log(P: np.ndarray, P_ref: np.ndarray, delta: float):
    """Compute the batched logarithmic map at P_ref.

    Projects each point cloud P onto the tangent space T_{P_ref} by combining
    the pre-log map with the pseudo-inverse of the metric tensor.  The 6
    null directions (rigid-body motions) are removed via eigendecomposition.

    Parameters
    ----------
    P : np.ndarray, shape (batch_size, n_dim, 3)
        Batch of point clouds to embed in the tangent space.
    P_ref : np.ndarray, shape (n_dim, 3)
        Reference (base) point cloud – foot of the tangent space.
    delta : float
        Weight of the gyration-matrix (size) term.

    Returns
    -------
    np.ndarray, shape (batch_size, n_dim, 3)
        Tangent vectors at P_ref.  The 6 rigid-body components are removed,
        leaving 3*n_dim − 6 effective degrees of freedom.

    Notes
    -----
    ``v_dim = 6`` is the dimension of the null space of the metric tensor,
    equal to the number of rigid-body DOF for a 3-D point cloud.  Eigenvalues
    below 1e-6 are clamped to prevent division by near-zero values.
    """
    batch_size = P.shape[0]
    n_dim      = P.shape[1]
    v_dim      = 3 * (3 + 1) // 2  # = 6 rigid-body DOF to remove

    # Step 1: pre-log displacement field, flattened to (batch_size, n_dim*3)
    P_prelog = prelog(P, P_ref, delta).reshape(batch_size, n_dim * 3)

    # Step 2: metric tensor at P_ref, reshaped to (n_dim*3, n_dim*3)
    G = metric_tensor(P_ref, delta)
    G = np.einsum("ijab->iajb", G).reshape(n_dim * 3, n_dim * 3)

    # Step 3: eigendecomposition; smallest 6 eigenvalues span the null space
    L, Q = np.linalg.eigh(G)
    L[L < 1e-6] = 1e-6   # clamp null-space eigenvalues for numerical stability

    # Step 4: pseudo-inverse projection onto the non-null subspace
    #   log(P) = −Q_perp · diag(1/λ_perp) · Q_perp^T · v
    P_log = -np.einsum(
        "ab,b,cb,...c->...a",
        Q[:, v_dim:], 1.0 / L[v_dim:], Q[:, v_dim:], P_prelog,
        optimize=True,
    )

    return P_log.reshape(batch_size, n_dim, 3)