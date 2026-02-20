"""
SO3/utils.py – Low-level matrix utilities for the SO(3) sub-package
====================================================================
Contains small, performance-critical helpers used internally by the SO(3)
exponential and logarithmic map implementations.  All functions are compiled
with Numba (``@nb.njit``) and inlined at call sites where possible.

Functions
---------
_matmul3x3(A, B, res)
    In-place product of two 3×3 matrices.
_skew_vec(R, res)
    Extract the axis-angle vector from the skew-symmetric part of a 3×3 matrix.
"""

import numba as nb
import numpy as np


@nb.njit(nogil=True, inline="always")
def _matmul3x3(A: np.ndarray, B: np.ndarray, res: np.ndarray):
    """Compute the product of two 3×3 matrices in-place: res = A @ B.

    Parameters
    ----------
    A, B : np.ndarray, shape (3, 3)
        Input matrices.
    res : np.ndarray, shape (3, 3)
        Output buffer receiving A @ B.  May not alias A or B.
    """
    for i in range(3):
        for j in range(3):
            acc = 0.0
            for k in range(3):
                acc += A[i, k] * B[k, j]
            res[i, j] = acc


@nb.njit(nogil=True, inline="always")
def _skew_vec(R: np.ndarray, res: np.ndarray):
    """Extract the axis-angle vector from the skew-symmetric part of R.

    For a rotation matrix R, the skew-symmetric part  (R − Rᵀ)/2  has the
    form  [v]×  where v is the rotation axis scaled by sin(θ).  This function
    extracts v from those off-diagonal entries.

    Parameters
    ----------
    R : np.ndarray, shape (3, 3)
        Input matrix (typically a rotation matrix or its difference from I).
    res : np.ndarray, shape (3,)
        Output vector: [R₂₁−R₁₂, R₀₂−R₂₀, R₁₀−R₀₁] / 2.
    """
    res[0] = (R[2, 1] - R[1, 2]) * 0.5
    res[1] = (R[0, 2] - R[2, 0]) * 0.5
    res[2] = (R[1, 0] - R[0, 1]) * 0.5