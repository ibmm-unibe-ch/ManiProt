"""
SO3/log.py – Logarithmic map on SO(3)
======================================
Maps a rotation matrix R ∈ SO(3) back to the tangent space at a reference
rotation R_ref, giving the axis-angle vector v such that exp_{R_ref}(v) ≈ R.

Mathematical background
-----------------------
The logarithmic map on SO(3) is the inverse of the matrix exponential.  Given
two rotations R_ref and R, the relative rotation is:

    ΔR = R_refᵀ · R   (rotation that maps R_ref to R)

The axis-angle vector of ΔR is then extracted via:

    θ  = arccos( (tr(ΔR) − 1) / 2 )                     [rotation angle]
    v  = θ / sin(θ) · skew_vec(ΔR)                       [scaled axis]

where  skew_vec(M) = [(M₂₁−M₁₂), (M₀₂−M₂₀), (M₁₀−M₀₁)] / 2.

For small angles (θ < 1e-3) the formula is approximated as:

    v ≈ skew_vec(ΔR)

The returned vector v lives in the tangent space T_{R_ref} ≅ ℝ³ and can be
treated as a Euclidean vector for downstream tasks (PCA, regression, etc.).

Private API (used internally by intrinsic_mean)
-----------------------------------------------
_log_acc(R, R_ref, R_aligned, skew_buf, acc)
    Accumulating variant: computes log_{R_ref}(R) and **adds** the result
    into *acc* (thread-local buffer).  Used for lock-free parallel reduction
    inside ``intrinsic_mean``.

Public API
----------
log(R, R_ref) -> np.ndarray
    Batched logarithmic map: rotation matrix → tangent vector.
"""

from typing import Tuple

import numba as nb
import numpy as np

from .utils import _matmul3x3, _skew_vec


# ---------------------------------------------------------------------------
# Private helpers (Numba-compiled, not part of the public API)
# ---------------------------------------------------------------------------

@nb.njit(nogil=True)
def _log(
    R: np.ndarray,       # rotation matrix to map  (3, 3)
    R_ref: np.ndarray,   # reference rotation       (3, 3)
    dR: np.ndarray,      # working buffer for ΔR    (3, 3)
    R_ref_T: np.ndarray, # working buffer for R_refᵀ (3, 3)
    res: np.ndarray,     # output axis-angle vector  (3,)
) -> None:
    """Compute log_{R_ref}(R) in-place.

    Parameters
    ----------
    R : np.ndarray, shape (3, 3)
        Target rotation matrix.
    R_ref : np.ndarray, shape (3, 3)
        Reference rotation matrix at the foot of the tangent space.
    dR : np.ndarray, shape (3, 3)
        Working buffer receiving the relative rotation ΔR = R_refᵀ · R.
    R_ref_T : np.ndarray, shape (3, 3)
        Working buffer receiving the transpose of R_ref.
    res : np.ndarray, shape (3,)
        Output buffer receiving the axis-angle tangent vector.
    """
    # Transpose of R_ref (= its inverse since R_ref ∈ SO(3))
    for i in range(3):
        for j in range(3):
            R_ref_T[i, j] = R_ref[j, i]

    # Relative rotation: ΔR = R_refᵀ · R
    _matmul3x3(R_ref_T, R, dR)

    # Rotation angle from the trace: cos θ = (tr(ΔR) − 1) / 2
    cos_theta = (dR[0, 0] + dR[1, 1] + dR[2, 2] - 1.0) * 0.5
    # Clamp to [-1, 1] to guard against floating-point rounding
    cos_theta = max(-1.0, min(1.0, cos_theta))
    theta = np.arccos(cos_theta)

    # Extract the skew-symmetric vector (proportional to axis × sin θ)
    _skew_vec(dR, res)

    if theta < 1e-3:
        # Small-angle approximation: v ≈ skew_vec(ΔR) (already computed)
        pass  # res already holds the correct approximation
    else:
        # Full formula: v = θ / sin(θ) · skew_vec(ΔR)
        scale = theta / np.sin(theta)
        res[0] *= scale
        res[1] *= scale
        res[2] *= scale


@nb.njit(nogil=True)
def _log_acc(
    R: np.ndarray,        # target rotation matrix   (3, 3)
    R_ref: np.ndarray,    # reference rotation        (3, 3)
    R_aligned: np.ndarray,# working buffer for ΔR     (3, 3)
    skew_buf: np.ndarray, # working buffer for axis×sinθ (3,)
    acc: np.ndarray,      # accumulation target       (3,)  — written with +=
) -> None:
    """Compute log_{R_ref}(R) and **accumulate** the result into *acc*.

    This is the lock-free accumulation variant of ``_log``, designed for use
    inside the parallel reduction loop of ``intrinsic_mean``.  Instead of
    writing the result to a fresh output buffer it adds (+=) it to a
    thread-local accumulator, avoiding any need for atomic operations or locks.

    Parameters
    ----------
    R : np.ndarray, shape (3, 3)
        Target rotation matrix.
    R_ref : np.ndarray, shape (3, 3)
        Current mean rotation (foot of the tangent space).
    R_aligned : np.ndarray, shape (3, 3)
        Working buffer receiving ΔR = R_refᵀ · R.
    skew_buf : np.ndarray, shape (3,)
        Working buffer receiving the scaled skew-symmetric vector before
        accumulation.
    acc : np.ndarray, shape (3,)
        Thread-local accumulator.  The log-map result is **added** to this
        buffer (+=), not overwritten.
    """
    # ΔR = R_refᵀ · R  (R_ref ∈ SO(3) so its inverse is its transpose)
    for i in range(3):
        for j in range(3):
            R_aligned[i, j] = (
                R_ref[0, i] * R[0, j] +
                R_ref[1, i] * R[1, j] +
                R_ref[2, i] * R[2, j]
            )

    # Rotation angle θ from the trace of ΔR: cos θ = (tr(ΔR) − 1) / 2
    cos_theta = (R_aligned[0, 0] + R_aligned[1, 1] + R_aligned[2, 2] - 1.0) * 0.5
    cos_theta = max(-1.0, min(1.0, cos_theta))  # clamp for numerical safety
    theta = np.arccos(cos_theta)

    # skew_vec(ΔR): the vector whose entries are the off-diagonal differences
    _skew_vec(R_aligned, skew_buf)

    if theta >= 1e-3:
        # Full formula: scale by θ / sin(θ)
        scale = theta / np.sin(theta)
        skew_buf[0] *= scale
        skew_buf[1] *= scale
        skew_buf[2] *= scale
    # else: small-angle approximation — skew_buf already holds the correct value

    # Accumulate into the thread-local sum (no atomic needed: each thread
    # owns its own slice of thread_log_sum)
    acc[0] += skew_buf[0]
    acc[1] += skew_buf[1]
    acc[2] += skew_buf[2]


@nb.njit(nogil=True)
def _log_core(
    R: np.ndarray,
    R_ref: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, int, int]:
    """Validate inputs and allocate working buffers for the parallel log kernel.

    Parameters
    ----------
    R : np.ndarray
        Target rotations.  Shape ``(..., 3, 3)`` for a shared R_ref, or
        ``(..., n_dim, 3, 3)`` for per-element references.
    R_ref : np.ndarray
        Reference rotation(s).  Shape ``(3, 3)`` or ``(n_dim, 3, 3)``.

    Returns
    -------
    flat_R, flat_R_ref, flat_dR, flat_R_ref_T, flat_res : np.ndarray
        Flattened arrays for the parallel loop.
    n_dim, batch_size : int
        Loop dimensions.
    """
    if R_ref.ndim == 2:
        n_dim = 1
        batch_shape = np.array(R.shape[:-2], dtype=np.int64)
        flat_R = R.reshape(-1, 3, 3)
    elif R_ref.ndim == 3:
        n_dim = R_ref.shape[0]
        batch_shape = np.array(R.shape[:-3], dtype=np.int64)
        flat_R = R.reshape(-1, n_dim, 3, 3).reshape(-1, 3, 3)
    else:
        raise NotImplementedError()

    batch_size = 1
    for s in batch_shape:
        batch_size *= s

    flat_R_ref   = R_ref.reshape(n_dim, 3, 3)
    flat_dR      = np.empty((batch_size * n_dim, 3, 3), dtype=R.dtype)
    flat_R_ref_T = np.empty((batch_size * n_dim, 3, 3), dtype=R.dtype)
    flat_res     = np.empty((batch_size * n_dim, 3),    dtype=R.dtype)

    return flat_R, flat_R_ref, flat_dR, flat_R_ref_T, flat_res, n_dim, batch_size


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

@nb.njit(nogil=True, parallel=True)
def log(R: np.ndarray, R_ref: np.ndarray) -> np.ndarray:
    """Compute the batched logarithmic map on SO(3) at R_ref.

    Maps rotation matrices to axis-angle tangent vectors in the tangent space
    T_{R_ref} ≅ ℝ³.  This is the inverse of ``exp``.

    Parameters
    ----------
    R : np.ndarray
        Target rotation(s) to map into the tangent space.

        - Shape ``(..., 3, 3)``          when ``R_ref.ndim == 2`` (shared ref).
        - Shape ``(..., n_dim, 3, 3)``   when ``R_ref.ndim == 3`` (per-element).
    R_ref : np.ndarray
        Reference rotation(s) at the foot of the tangent space.

        - Shape ``(3, 3)``           for a single shared reference.
        - Shape ``(n_dim, 3, 3)``    for one reference per rotation.

    Returns
    -------
    np.ndarray, shape ``R.shape[:-2] + (3,)``  or  ``R.shape[:-3] + (n_dim, 3)``
        Axis-angle tangent vectors.  ``result[..., i]`` is the tangent vector
        at ``R_ref[i]`` pointing toward ``R[..., i]``.

    Examples
    --------
    >>> # Map a batch of rotations to tangent vectors at the identity
    >>> R_batch = some_rotation_matrices   # shape (100, 3, 3)
    >>> R_id = np.eye(3, dtype=np.float32)
    >>> v_batch = log(R_batch, R_id)       # shape (100, 3)

    >>> # Verify round-trip consistency with exp
    >>> from maniprot.SO3.exp import exp
    >>> R_reconstructed = exp(v_batch, R_id)   # should ≈ R_batch
    """
    flat_R, flat_R_ref, flat_dR, flat_R_ref_T, flat_res, n_dim, batch_size = _log_core(R, R_ref)

    for i in nb.prange(batch_size * n_dim):
        c = i % n_dim  # which reference rotation to use
        _log(flat_R[i], flat_R_ref[c], flat_dR[i], flat_R_ref_T[i], flat_res[i])

    if R_ref.ndim == 2:
        return flat_res.reshape(R.shape[:-2] + (3,))
    else:
        return flat_res.reshape(R.shape[:-3] + (n_dim, 3))