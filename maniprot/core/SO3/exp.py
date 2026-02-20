"""
SO3/exp.py – Exponential map on SO(3) via the Rodrigues formula
===============================================================
Maps a tangent vector (axis-angle representation in the Lie algebra 𝔰𝔬(3))
back onto the rotation group SO(3).

Mathematical background
-----------------------
Given a reference rotation R_ref ∈ SO(3) and a tangent vector v ∈ ℝ³ at that
point (an element of the Lie algebra 𝔰𝔬(3)), the exponential map is:

    exp_{R_ref}(v) = R_ref · Exp(v)

where Exp(v) is the matrix exponential of the skew-symmetric matrix [v]×:

           [  0  −v₂   v₁ ]
    [v]× = [  v₂   0  −v₀ ]
           [ −v₁  v₀   0  ]

Using the Rodrigues formula:

    Exp(v) = I  +  sin(‖v‖)/‖v‖ · [v]×  +  (1 − cos(‖v‖))/‖v‖² · [v]×²

For small rotations (‖v‖² < 1e-6) a second-order Taylor approximation is used:

    Exp(v) ≈ I  +  [v]×  +  [v]×²/2

Public API
----------
exp(v, R_ref) -> np.ndarray
    Batched exponential map: tangent vector → rotation matrix.
"""

from typing import Tuple

import numba as nb
import numpy as np

from .utils import _matmul3x3


# ---------------------------------------------------------------------------
# Private helpers (Numba-compiled, not part of the public API)
# ---------------------------------------------------------------------------

@nb.njit(nogil=True)
def _exp(
    v: np.ndarray,           # tangent vector (3,) in ℝ³
    R_ref: np.ndarray,       # reference rotation matrix (3, 3)
    v_skew: np.ndarray,      # pre-allocated buffer for [v]×   (3, 3)
    v_skew_sq: np.ndarray,   # pre-allocated buffer for [v]×²  (3, 3)
    R_aligned: np.ndarray,   # pre-allocated buffer for Exp(v) (3, 3)
    res: np.ndarray,         # output rotation matrix          (3, 3)
) -> None:
    """Compute exp_{R_ref}(v) in-place using the Rodrigues formula.

    Parameters
    ----------
    v : np.ndarray, shape (3,)
        Tangent / axis-angle vector in the Lie algebra 𝔰𝔬(3).
    R_ref : np.ndarray, shape (3, 3)
        Reference rotation matrix (foot of the tangent space).
    v_skew : np.ndarray, shape (3, 3)
        Working buffer receiving the skew-symmetric matrix [v]×.
    v_skew_sq : np.ndarray, shape (3, 3)
        Working buffer receiving [v]×².
    R_aligned : np.ndarray, shape (3, 3)
        Working buffer receiving Exp(v).
    res : np.ndarray, shape (3, 3)
        Output buffer receiving R_ref · Exp(v).
    """
    # Build skew-symmetric matrix [v]×
    v_skew[0, 0] =  0.0;  v_skew[0, 1] = -v[2]; v_skew[0, 2] =  v[1]
    v_skew[1, 0] =  v[2]; v_skew[1, 1] =  0.0;  v_skew[1, 2] = -v[0]
    v_skew[2, 0] = -v[1]; v_skew[2, 1] =  v[0]; v_skew[2, 2] =  0.0

    _matmul3x3(v_skew, v_skew, v_skew_sq)   # [v]×²

    norm_sq = v[0]*v[0] + v[1]*v[1] + v[2]*v[2]
    norm    = np.sqrt(norm_sq)

    if norm_sq < 1e-6:
        # Small-angle approximation: Exp(v) ≈ I + [v]× + [v]×²/2
        R_aligned = (
            np.eye(3, dtype=v.dtype) +
            v_skew +
            v_skew_sq * 0.5
        )
    else:
        # Full Rodrigues formula
        R_aligned = (
            np.eye(3, dtype=v.dtype) +
            np.sin(norm) / norm * v_skew +
            (1.0 - np.cos(norm)) / norm_sq * v_skew_sq
        )

    # Apply the reference rotation: result = R_ref · Exp(v)
    _matmul3x3(R_ref, R_aligned, res)


@nb.njit(nogil=True)
def _exp_core(
    v: np.ndarray,
    R_ref: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, int, int]:
    """Validate inputs and allocate working buffers for the parallel exp kernel.

    Parameters
    ----------
    v : np.ndarray
        Tangent vectors.  Shape (..., 3) for a single-frame R_ref, or
        (..., n_dim, 3) for a per-point R_ref.
    R_ref : np.ndarray
        Reference rotation(s).  Shape (3, 3) for a shared reference, or
        (n_dim, 3, 3) for one reference per rotation.

    Returns
    -------
    flat_v, flat_R_ref, flat_v_skew, flat_v_skew_sq, flat_R_aligned,
    flat_res : np.ndarray
        Flattened arrays for the parallel loop.
    n_dim, batch_size : int
        Loop dimensions.
    """
    if v.ndim < 1 or v.shape[-1:] != (3,):
        raise ValueError()
    if R_ref.ndim < 2 or R_ref.shape[-2:] != (3, 3):
        raise ValueError()

    if R_ref.ndim == 2:
        # Single shared reference rotation
        n_dim = 1
        batch_shape = np.array(v.shape[:-1], dtype=np.int64)
    elif R_ref.ndim == 3:
        # Per-element reference rotations
        n_dim = R_ref.shape[0]
        if v.ndim < 2 or v.shape[-2:] != (n_dim, 3):
            raise ValueError()
        batch_shape = np.array(v.shape[:-2], dtype=np.int64)
    else:
        raise NotImplementedError()

    batch_size = 1
    for s in batch_shape:
        batch_size *= s

    flat_v         = v.reshape(batch_size * n_dim, 3)
    flat_R_ref     = R_ref.reshape(n_dim, 3, 3)
    flat_v_skew    = np.empty((batch_size * n_dim, 3, 3), dtype=v.dtype)
    flat_v_skew_sq = np.empty((batch_size * n_dim, 3, 3), dtype=v.dtype)
    flat_R_aligned = np.empty((batch_size * n_dim, 3, 3), dtype=v.dtype)
    flat_res       = np.empty((batch_size * n_dim, 3, 3), dtype=v.dtype)

    return flat_v, flat_R_ref, flat_v_skew, flat_v_skew_sq, flat_R_aligned, flat_res, n_dim, batch_size


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

@nb.njit(nogil=True, parallel=True)
def exp(v: np.ndarray, R_ref: np.ndarray) -> np.ndarray:
    """Compute the batched exponential map on SO(3) at R_ref.

    Maps tangent vectors from the Lie algebra 𝔰𝔬(3) (axis-angle
    representation) to rotation matrices via the Rodrigues formula.

    Parameters
    ----------
    v : np.ndarray
        Tangent vectors (axis-angle representation).

        - Shape ``(..., 3)``         when ``R_ref.ndim == 2`` (shared reference).
        - Shape ``(..., n_dim, 3)``  when ``R_ref.ndim == 3`` (per-element refs).
    R_ref : np.ndarray
        Reference rotation(s) at the foot of the tangent space.

        - Shape ``(3, 3)``           for a single shared reference.
        - Shape ``(n_dim, 3, 3)``    for one reference per rotation.

    Returns
    -------
    np.ndarray, shape ``v.shape[:-1] + (3, 3)``
        Rotation matrices.  ``result[..., i] = exp_{R_ref[i]}(v[..., i])``.

    Examples
    --------
    >>> # Small rotation around x-axis from identity
    >>> v = np.array([0.1, 0.0, 0.0], dtype=np.float32)
    >>> R_new = exp(v, np.eye(3, dtype=np.float32))
    >>> R_new.shape
    (3, 3)

    >>> # Batch of 100 tangent vectors with a shared reference
    >>> v_batch = np.random.randn(100, 3).astype(np.float32)
    >>> R_batch = exp(v_batch, np.eye(3, dtype=np.float32))
    >>> R_batch.shape
    (100, 3, 3)
    """
    flat_v, flat_R_ref, flat_v_skew, flat_v_skew_sq, flat_R_aligned, flat_res, n_dim, batch_size = (
        _exp_core(v, R_ref)
    )

    for i in nb.prange(batch_size * n_dim):
        c = i % n_dim  # which reference rotation to use
        _exp(
            flat_v[i], flat_R_ref[c],
            flat_v_skew[i], flat_v_skew_sq[i], flat_R_aligned[i],
            flat_res[i],
        )

    return flat_res.reshape(v.shape[:-1] + (3, 3))