"""
helpers/lcs.py – Local Coordinate System (LCS) for protein residues
====================================================================
Each amino-acid residue is assigned a right-handed orthonormal frame built from
the three backbone heavy atoms: N, Cα (CA), and C.  The frame is defined as:

    u  = (N − Cα) / ‖N − Cα‖          # primary axis: Cα → N
    n  = (u × t) / ‖u × t‖             # plane normal  (cross product)
    v  = (n × u) / ‖n × u‖             # completes the right-handed frame

where t = (C − Cα) / ‖C − Cα‖ is an auxiliary unit vector from Cα toward C.

The resulting 3×3 rotation matrix  R = [u | n | v]  (columns = frame axes,
stored row-major in the output) is an element of SO(3) and encodes the
orientation of the residue's peptide plane in 3-D space.

These LCS frames are consumed by ``maniprot.SO3`` for orientation-based
analysis and by ``helpers.mdtraj`` for MDTraj trajectory integration.

All heavy computation is JIT-compiled with Numba and runs in parallel over the
combined (frames × residues) batch dimension.

Public API
----------
lcs(xyz, n_indices, ca_indices, c_indices) -> np.ndarray
    Compute LCS rotation matrices for a batch of frames.
"""

from typing import Tuple

import numba as nb
import numpy as np


# ---------------------------------------------------------------------------
# Private helpers (Numba-compiled, not part of the public API)
# ---------------------------------------------------------------------------

@nb.njit(nogil=True)
def _lcs(
    x: np.ndarray,   # N  atom position  (3,)
    y: np.ndarray,   # Cα atom position  (3,) – origin of the local frame
    z: np.ndarray,   # C  atom position  (3,)
    u: np.ndarray,   # working buffer (3,) for the u-axis
    t: np.ndarray,   # working buffer (3,) for the t auxiliary vector
    n: np.ndarray,   # working buffer (3,) for the n-axis
    v: np.ndarray,   # working buffer (3,) for the v-axis
    res: np.ndarray, # output rotation matrix (3, 3)
):
    """Compute a single-residue LCS rotation matrix in-place.

    Builds the orthonormal frame {u, n, v} centred at Cα (y) using the N (x)
    and C (z) atom positions, then stores it column-wise in *res*.

    Parameters
    ----------
    x, y, z : np.ndarray, shape (3,)
        Cartesian coordinates of the N, Cα, and C backbone atoms.
    u, t, n, v : np.ndarray, shape (3,)
        Pre-allocated working buffers (written in-place by this function).
    res : np.ndarray, shape (3, 3)
        Output matrix whose columns are [u | n | v] stored row-major.
    """
    # Primary axis: Cα → N direction, normalised
    u = x - y
    u /= np.sqrt(u[0]*u[0] + u[1]*u[1] + u[2]*u[2])

    # Auxiliary vector: Cα → C direction, normalised (not orthogonal to u)
    t = z - y
    t /= np.sqrt(t[0]*t[0] + t[1]*t[1] + t[2]*t[2])

    # Plane normal via cross product u × t, then normalised
    n[0] = u[1]*t[2] - u[2]*t[1]
    n[1] = u[2]*t[0] - u[0]*t[2]
    n[2] = u[0]*t[1] - u[1]*t[0]
    n /= np.sqrt(n[0]*n[0] + n[1]*n[1] + n[2]*n[2])

    # Third axis completes the right-handed frame: v = n × u
    v[0] = n[1]*u[2] - n[2]*u[1]
    v[1] = n[2]*u[0] - n[0]*u[2]
    v[2] = n[0]*u[1] - n[1]*u[0]
    v /= np.sqrt(v[0]*v[0] + v[1]*v[1] + v[2]*v[2])

    # Pack frame vectors as columns of the output rotation matrix
    res[0, 0] = u[0];  res[0, 1] = n[0];  res[0, 2] = v[0]
    res[1, 0] = u[1];  res[1, 1] = n[1];  res[1, 2] = v[1]
    res[2, 0] = u[2];  res[2, 1] = n[2];  res[2, 2] = v[2]


@nb.njit(nogil=True)
def _lcs_core(
    xyz: np.ndarray,
    n_indices: list[int],
    ca_indices: list[int],
    c_indices: list[int],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, int, int]:
    """Validate inputs and allocate working buffers for the parallel lcs kernel.

    Parameters
    ----------
    xyz : np.ndarray, shape (n_frames, n_atoms, 3)
        Trajectory atom positions.
    n_indices, ca_indices, c_indices : list[int]
        Atom index lists (one entry per selected residue) for N, Cα, C.

    Returns
    -------
    u, t, n, v : np.ndarray, shape (n_frames * n_residues, 3)
        Intermediate vector buffers.
    res : np.ndarray, shape (n_frames * n_residues, 3, 3)
        Flattened output buffer for rotation matrices.
    n_frames, n_residues : int
        Dimensions extracted from the inputs.
    """
    if not len(n_indices) == len(ca_indices) == len(c_indices):
        raise ValueError()

    n_frames   = xyz.shape[0]
    n_residues = len(n_indices)

    u   = np.empty((n_frames * n_residues, 3), dtype=np.float32)
    t   = np.empty((n_frames * n_residues, 3), dtype=np.float32)
    n   = np.empty((n_frames * n_residues, 3), dtype=np.float32)
    v   = np.empty((n_frames * n_residues, 3), dtype=np.float32)
    res = np.empty((n_frames * n_residues, 3, 3), dtype=np.float32)

    return u, t, n, v, res, n_frames, n_residues


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

@nb.njit(nogil=True, parallel=True)
def lcs(
    xyz: np.ndarray,
    n_indices: list[int],
    ca_indices: list[int],
    c_indices: list[int],
) -> np.ndarray:
    """Compute per-residue Local Coordinate System (LCS) rotation matrices.

    For every (frame, residue) pair builds a right-handed orthonormal frame
    from the backbone N, Cα, C atoms and returns it as a 3×3 rotation matrix
    (element of SO(3)).  The computation is parallelised over all
    (frame × residue) pairs using Numba's ``prange``.

    Parameters
    ----------
    xyz : np.ndarray, shape (n_frames, n_atoms, 3), dtype float32
        Cartesian coordinates of all atoms for each trajectory frame.
    n_indices : list[int]
        Global atom indices of the backbone N atoms (one per residue).
    ca_indices : list[int]
        Global atom indices of the backbone Cα atoms (one per residue).
    c_indices : list[int]
        Global atom indices of the backbone C atoms (one per residue).

    Returns
    -------
    np.ndarray, shape (n_frames, n_residues, 3, 3), dtype float32
        Rotation matrices.  ``result[f, r]`` is the 3×3 LCS frame for
        residue *r* in frame *f*, with columns [u | n | v].

    Examples
    --------
    >>> # Use together with maniprot.SO3 for orientation-based analysis
    >>> from maniprot.helpers import lcs
    >>> from maniprot.SO3 import SO3Manifold
    >>> R = lcs(xyz, n_indices, ca_indices, c_indices)
    >>> manifold = SO3Manifold().fit(R)
    >>> T = manifold.transform(R)   # shape (n_frames, n_residues, 3)
    """
    u, t, n, v, res, n_frames, n_residues = _lcs_core(xyz, n_indices, ca_indices, c_indices)

    for i in nb.prange(n_frames * n_residues):
        f = i // n_residues  # frame index
        r = i  % n_residues  # residue index

        n_idx  = n_indices[r]
        ca_idx = ca_indices[r]
        c_idx  = c_indices[r]

        _lcs(
            xyz[f, n_idx], xyz[f, ca_idx], xyz[f, c_idx],
            u[i], t[i], n[i], v[i], res[i],
        )

    return res.reshape((n_frames, n_residues, 3, 3))