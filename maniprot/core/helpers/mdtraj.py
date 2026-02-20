"""
helpers/mdtraj.py – MDTraj integration for LCS computation
============================================================
Bridges the MDTraj trajectory API with ``helpers.lcs``.  MDTraj
(https://mdtraj.org) is a widely used Python library for loading and analysing
molecular dynamics (MD) trajectories.

This module provides a convenience function to extract all inputs required by
``lcs()`` directly from an ``mdtraj.Trajectory`` object, automatically
skipping residues that lack one or more backbone heavy atoms.

Public API
----------
lcs_args(trajectory) -> (xyz, n_indices, ca_indices, c_indices)
    Extract LCS inputs from an MDTraj trajectory.
"""

from typing import List, Tuple

import numpy as np

import mdtraj as md

from .lcs import lcs


def lcs_args(
    trajectory: "md.Trajectory",  # type: ignore[name-defined]
) -> Tuple[np.ndarray, List[int], List[int], List[int]]:
    """Extract backbone atom indices and coordinates from an MDTraj trajectory.

    Iterates over all residues in the trajectory topology and collects the
    global atom indices of the three backbone heavy atoms N, Cα, and C.
    Residues missing any of these atoms (e.g. terminal residues or non-standard
    residues) are silently skipped.

    The returned values can be passed directly to ``lcs()`` (or the bundled
    convenience call shown below) to obtain per-residue LCS rotation matrices.

    Parameters
    ----------
    trajectory : mdtraj.Trajectory
        An MDTraj trajectory object (loaded via ``md.load`` or similar).

    Returns
    -------
    xyz : np.ndarray, shape (n_frames, n_atoms, 3), dtype float32
        Atom coordinates in nanometres (MDTraj convention), wrapped in a
        C-contiguous array as required by the Numba-compiled ``lcs`` kernel.
    n_indices : list[int]
        Atom indices of the backbone N atoms for the selected residues.
    ca_indices : list[int]
        Atom indices of the backbone Cα atoms for the selected residues.
    c_indices : list[int]
        Atom indices of the backbone C atoms for the selected residues.

    Examples
    --------
    >>> import mdtraj as md
    >>> from maniprot.helpers import lcs, lcs_args

    >>> traj = md.load("simulation.xtc", top="topology.pdb")
    >>> xyz, n_idx, ca_idx, c_idx = lcs_args(traj)
    >>> R = lcs(xyz, n_idx, ca_idx, c_idx)
    >>> # R.shape == (n_frames, n_residues, 3, 3)

    Notes
    -----
    MDTraj stores coordinates in **nanometres**.  If your downstream code
    expects Ångströms, multiply ``xyz`` by 10 after calling this function.
    """
    n_indices  = []
    ca_indices = []
    c_indices  = []

    for r in trajectory.topology.residues:
        # Try to obtain the three backbone heavy-atom indices; skip on failure
        try:
            n_idx  = r.atom('N').index
            ca_idx = r.atom('CA').index
            c_idx  = r.atom('C').index
        except KeyError:
            continue

        n_indices.append(n_idx)
        ca_indices.append(ca_idx)
        c_indices.append(c_idx)

    return np.ascontiguousarray(trajectory.xyz), n_indices, ca_indices, c_indices