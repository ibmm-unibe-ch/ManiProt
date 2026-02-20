"""
maniprot.core.pointcloud
===================
Riemannian geometry on labelled 3-D point clouds.

This sub-package models a protein backbone (e.g. a set of Cα positions) as a
point on a Riemannian manifold whose geodesic distance is invariant to
rigid-body transformations.  The metric is parameterised by a scalar ``delta``
that balances shape information (pairwise distances) against size information
(gyration-matrix determinant).

Key components
--------------
``PointcloudManifold``
    Scikit-learn-style wrapper with ``fit`` (Fréchet mean) and ``transform``
    (logarithmic map) methods.

``distance(P, Q, delta)``
    Batched geodesic distance between point clouds.

``log(P, P_ref, delta)``
    Logarithmic map — embeds P in the tangent space at P_ref.

``metric_tensor(P, delta)``
    Riemannian metric tensor at a point cloud.

``norm(v, P_ref, delta)``
    Riemannian norm of a tangent vector.

``intrinsic_mean(P, P_ref, delta, ...)``
    Fréchet mean via Riemannian gradient descent.

Example
-------
>>> from maniprot.core.pointcloud import PointcloudManifold
>>> manifold = PointcloudManifold(delta=1.0)
>>> manifold.fit(X)           # X: (n_samples, n_dim, 3)
>>> T = manifold.transform(X) # (n_samples, n_dim, 3) Euclidean tangent vectors
"""

from .manifold import PointcloudManifold
from .log import log, prelog
from .distance import distance
from .metric_tensor import metric_tensor
from .norm import norm
from .intrinsic_mean import intrinsic_mean

__all__ = [
    "PointcloudManifold",
    "log", "prelog",
    "distance",
    "metric_tensor",
    "norm",
    "intrinsic_mean",
]