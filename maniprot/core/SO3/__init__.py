"""
maniprot.core.SO3
============
Riemannian geometry on the rotation group SO(3).

This sub-package provides tools for working with rotation matrices (elements
of SO(3)) as points on a Riemannian manifold.  The key operations are:

- ``exp`` – exponential map (axis-angle → rotation matrix, Rodrigues formula)
- ``log`` – logarithmic map (rotation matrix → axis-angle tangent vector)
- ``intrinsic_mean`` – Fréchet mean via Riemannian gradient descent
- ``SO3Manifold`` – scikit-learn-style fit/transform wrapper

Typical use case: given per-residue Local Coordinate System (LCS) frames
produced by ``maniprot.helpers.lcs``, embed them into a Euclidean space
for downstream analysis.

    >>> from maniprot.core.SO3 import SO3Manifold
    >>> manifold = SO3Manifold()
    >>> manifold.fit(R_frames)           # R_frames: (n_frames, n_residues, 3, 3)
    >>> T = manifold.transform(R_frames) # (n_frames, n_residues, 3) – Euclidean
"""

from .exp import exp
from .log import log
from .intrinsic_mean import intrinsic_mean
from .manifold import SO3Manifold

__all__ = ["exp", "log", "intrinsic_mean", "SO3Manifold"]