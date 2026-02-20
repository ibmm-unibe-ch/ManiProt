"""
pointcloud/utils.py – Low-level Numba utilities for the point-cloud sub-package
================================================================================
Contains small, performance-critical helpers shared across all point-cloud
manifold modules.  All functions are compiled with Numba (``@nb.njit``) and
inlined at call sites where possible.

Functions
---------
pair_from_index(i, n)
    Recover pair (j, k) from a flat upper-triangular index.
_det3x3(M)
    Determinant of a 3×3 matrix.
_inv3x3(M, res)
    In-place inverse of a 3×3 matrix.
_offsets(P, res)
    Subtract the centroid from every point in a cloud.
_gyration_matrix(P_offsets, res)
    Gyration matrix G = Σᵢ pᵢ pᵢᵀ from centred coordinates.
"""

import numba as nb
import numpy as np


@nb.njit(nogil=True, inline="always")
def pair_from_index(i: int, n: int):
    """Convert a flat upper-triangular index to a pair (j, k) with j < k.

    The upper-triangular pairs of n items are enumerated in row-major order:
    (0,1), (0,2), …, (0,n-1), (1,2), …, (n-2, n-1).  This function inverts
    that enumeration in O(1) using the quadratic formula.

    Parameters
    ----------
    i : int
        Flat pair index in [0, n*(n-1)/2).
    n : int
        Total number of items.

    Returns
    -------
    j, k : int
        Row and column of the pair with j < k.
    """
    discriminant = (2.0 * n - 1.0) * (2.0 * n - 1.0) - 8.0 * i
    root = np.sqrt(discriminant)
    j = int(np.floor((2.0 * n - 1.0 - root) * 0.5))

    Sj = (j * (2 * n - j - 1)) // 2  # cumulative pairs before row j
    r  = i - Sj
    k  = j + 1 + r
    return j, k


@nb.njit(nogil=True, inline="always")
def _det3x3(M: np.ndarray) -> float:
    """Compute the determinant of a 3×3 matrix via cofactor expansion.

    Parameters
    ----------
    M : np.ndarray, shape (3, 3)
        Input matrix.

    Returns
    -------
    float
        det(M).
    """
    return (
        M[0, 0] * (M[1, 1] * M[2, 2] - M[1, 2] * M[2, 1])
      - M[0, 1] * (M[1, 0] * M[2, 2] - M[1, 2] * M[2, 0])
      + M[0, 2] * (M[1, 0] * M[2, 1] - M[1, 1] * M[2, 0])
    )


@nb.njit(nogil=True, inline="always")
def _inv3x3(M: np.ndarray, res: np.ndarray):
    """Compute the inverse of a 3×3 matrix in-place via the adjugate formula.

    Parameters
    ----------
    M : np.ndarray, shape (3, 3)
        Input matrix (must be non-singular).
    res : np.ndarray, shape (3, 3)
        Output buffer that receives M⁻¹.

    Raises
    ------
    ValueError
        If *M* is singular (det = 0).
    """
    det = _det3x3(M)
    if det == 0:
        raise ValueError("Singular matrix cannot be inverted")

    res[0, 0] = (M[1, 1] * M[2, 2] - M[1, 2] * M[2, 1]) / det
    res[0, 1] = (M[0, 2] * M[2, 1] - M[0, 1] * M[2, 2]) / det
    res[0, 2] = (M[0, 1] * M[1, 2] - M[0, 2] * M[1, 1]) / det
    res[1, 0] = (M[1, 2] * M[2, 0] - M[1, 0] * M[2, 2]) / det
    res[1, 1] = (M[0, 0] * M[2, 2] - M[0, 2] * M[2, 0]) / det
    res[1, 2] = (M[0, 2] * M[1, 0] - M[0, 0] * M[1, 2]) / det
    res[2, 0] = (M[1, 0] * M[2, 1] - M[1, 1] * M[2, 0]) / det
    res[2, 1] = (M[0, 1] * M[2, 0] - M[0, 0] * M[2, 1]) / det
    res[2, 2] = (M[0, 0] * M[1, 1] - M[0, 1] * M[1, 0]) / det


@nb.jit(nogil=True)
def _offsets(P: np.ndarray, res: np.ndarray):
    """Compute mean-centred offsets of a point cloud.

    Subtracts the centroid (mean of all points) from every point, storing the
    result in *res*.

    Parameters
    ----------
    P : np.ndarray, shape (n_dim, 3)
        Input point cloud.
    res : np.ndarray, shape (n_dim, 3)
        Output buffer receiving  P[i] − mean(P)  for each i.
    """
    n_dim  = P.shape[0]
    P_mean = P.sum(0) / n_dim
    for i in range(n_dim):
        res[i] = P[i] - P_mean


@nb.njit(nogil=True)
def _gyration_matrix(P_offsets: np.ndarray, res: np.ndarray):
    """Compute the gyration (second-moment) matrix of a centred point cloud.

    Defined as  G = Σᵢ pᵢ pᵢᵀ  where pᵢ are the centred coordinates
    (pᵢ = P[i] − centroid(P)).  The gyration matrix characterises the spatial
    extent and orientation of the point cloud; its determinant enters the
    manifold distance as the "size" term.

    Parameters
    ----------
    P_offsets : np.ndarray, shape (n_dim, 3)
        Mean-centred point cloud (as returned by ``_offsets``).
    res : np.ndarray, shape (3, 3)
        Output buffer receiving G.
    """
    n_dim  = P_offsets.shape[0]
    res[:] = 0.0
    for i in range(n_dim):
        t0, t1, t2 = P_offsets[i]
        res[0, 0] += t0 * t0;  res[0, 1] += t0 * t1;  res[0, 2] += t0 * t2
        res[1, 0] += t1 * t0;  res[1, 1] += t1 * t1;  res[1, 2] += t1 * t2
        res[2, 0] += t2 * t0;  res[2, 1] += t2 * t1;  res[2, 2] += t2 * t2