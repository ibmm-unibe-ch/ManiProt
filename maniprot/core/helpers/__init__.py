"""
maniprot.core.helpers
================
Utility functions for building feature inputs from MD trajectories.

This sub-package provides two complementary tools:

``lcs(xyz, n_indices, ca_indices, c_indices)``
    Compute per-residue Local Coordinate System (LCS) rotation matrices from
    raw atom coordinates.  Returns a ``(n_frames, n_residues, 3, 3)`` array
    of SO(3) rotation matrices suitable for use with ``maniprot.SO3``.

``lcs_args(trajectory)``
    Convenience wrapper for MDTraj trajectories.  Iterates over the topology
    to collect backbone N/Cα/C atom indices and returns everything needed to
    call ``lcs`` in a single step.

Typical usage with MDTraj
-------------------------
>>> import mdtraj as md
>>> from maniprot.core.helpers import lcs, lcs_args

>>> traj = md.load("run.xtc", top="protein.pdb")
>>> xyz, n_idx, ca_idx, c_idx = lcs_args(traj)
>>> R = lcs(xyz, n_idx, ca_idx, c_idx)
>>> # R.shape == (n_frames, n_residues, 3, 3)
"""

from .lcs import lcs
from .mdtraj import lcs_args

__all__ = ["lcs", "lcs_args"]