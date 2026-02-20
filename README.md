# maniprot.core

**maniprot.core** is a Python library for Riemannian geometry on protein conformations.
It provides two complementary, GPU-free feature representations for molecular
dynamics (MD) trajectories, both accelerated with [Numba](https://numba.pydata.org)
JIT compilation and multi-core parallelism.

---

## Installation
```
git clone https://github.com/ibmm-unibe-ch/ManiProt.git
cd ManiProt

conda env create -f env.yml
conda activate maniprot
```

## Overview

maniprot is organised around two complementary approaches to protein conformational
analysis.  Both respect the underlying geometry of molecular structures and
produce representations that can be used with standard machine-learning tools.

| Sub-package | What it models | Key invariance |
|---|---|---|
| **SO3** | Per-residue orientation (rotation matrices) | Frame-of-reference choice |
| **pointcloud** | Whole-backbone shape & size (point clouds) | Rigid-body transformations |
| **helpers** | Feature extraction from MD trajectories | — |

---


**Dependencies:** `numpy`, `numba`.  The `helpers.mdtraj` module additionally
requires [MDTraj](https://mdtraj.org).

---

## Sub-package: `maniprot.core.helpers`

The `helpers` sub-package bridges raw MD trajectory data with the two geometry
sub-packages.

### Local Coordinate Systems (LCS)

Each residue is assigned a right-handed orthonormal frame built from its
backbone N, Cα, C atoms:

```
    u = (N − Cα) / ‖N − Cα‖       # primary axis
    n = (u × t)  / ‖u × t‖        # plane normal
    v = (n × u)  / ‖n × u‖        # completes the frame
```

The result is a 3×3 rotation matrix (element of SO(3)) per (frame, residue).

```python
import mdtraj as md
from maniprot.core.helpers import lcs, lcs_args

traj = md.load("simulation.xtc", top="topology.pdb")

# Automatically extract backbone atom indices from the MDTraj topology
xyz, n_indices, ca_indices, c_indices = lcs_args(traj)

# Compute LCS rotation matrices: (n_frames, n_residues, 3, 3)
R = lcs(xyz, n_indices, ca_indices, c_indices)
```

Without MDTraj, supply atom index lists directly:

```python
from maniprot.core.helpers import lcs
R = lcs(xyz, n_indices, ca_indices, c_indices)
```

---

## Sub-package: `maniprot.core.SO3`

Riemannian geometry on the rotation group SO(3).  Treats each per-residue LCS
frame as a point on the manifold and provides tools to compute with them.

### Exponential map (axis-angle → rotation matrix)

```python
from maniprot.core.SO3 import exp
import numpy as np

# Reconstruct rotations from predicted tangent offsets (e.g. in a diffusion model)
v = np.random.randn(100, 3).astype(np.float32)   # tangent vectors
R_ref = np.eye(3, dtype=np.float32)              # reference rotation
R_new = exp(v, R_ref)                            # shape (100, 3, 3)
```

Uses the Rodrigues formula with a second-order Taylor fallback for small angles.

### Logarithmic map (rotation matrix → tangent vector)

```python
from maniprot.core.SO3 import log

# Map rotations to tangent vectors at R_ref (inverse of exp)
v = log(R_batch, R_ref)   # shape (n_frames, n_residues, 3)
```

### Fréchet mean and manifold wrapper

```python
from maniprot.core.SO3 import SO3Manifold

# R: (n_frames, n_residues, 3, 3) LCS rotation matrices
manifold = SO3Manifold()
manifold.fit(R)                    # computes the Fréchet mean per residue
T = manifold.transform(R)          # shape (n_frames, n_residues, 3) – Euclidean
```

---

## Sub-package: `maniprot.core.pointcloud`

Riemannian geometry on labelled 3-D point clouds (e.g. Cα backbone coordinates).

### Mathematical background

A protein backbone is modelled as a labelled point cloud
**P = {p₁, …, pₙ} ⊂ ℝ³**.  The geodesic distance between two such clouds is:

```
d²(P, Q) = Σ_{j<k}  [ log( ‖pⱼ − pₖ‖ / ‖qⱼ − qₖ‖ ) ]²
          + δ · [ log( det G_P / det G_Q ) ]²
```

where **G_P** is the *gyration matrix* (Σᵢ p̃ᵢ p̃ᵢᵀ of centred coordinates) and
**δ ≥ 0** weights the size (gyration) term relative to the shape (pairwise
distances) term.  The distance is invariant to rigid-body transformations.

### High-level API

```python
import numpy as np
from maniprot.core.pointcloud import PointcloudManifold

X = np.load("ca_coords.npy")   # shape (n_frames, n_dim, 3)

manifold = PointcloudManifold(delta=1.0)
manifold.fit(X)                 # computes the Fréchet mean
X_tangent = manifold.transform(X)  # shape (n_frames, n_dim, 3) Euclidean vectors
```

### Geodesic distance

```python
from maniprot.core.pointcloud import distance

d   = distance(P, Q, delta=1.0)            # scalar, single pair
D   = distance(P_batch, Q_batch, delta=1.0) # shape (n_P, n_Q), all pairwise
```

### Logarithmic map

```python
from maniprot.core.pointcloud import log

T = log(X, P_ref, delta=1.0)   # shape (n_samples, n_dim, 3)
```

The log map removes the 6-dimensional rigid-body null space via eigendecomposition
of the metric tensor, leaving **3n − 6** effective degrees of freedom.

### Metric tensor and norm

```python
from maniprot.core.pointcloud import metric_tensor, norm

G     = metric_tensor(P, delta=1.0)   # shape (n_dim, n_dim, 3, 3)
norms = norm(V, P_ref, delta=1.0)     # Riemannian norm of tangent vectors V
```

---

## The δ (delta) parameter

| Value | Effect |
|---|---|
| `delta = 0` | Purely shape-based: only pairwise distances matter; scale-invariant |
| `delta > 0` | Size also contributes; conformations differing in extent are distinguished |

A good default is **δ = 0.1**.

---

## End-to-end example

```python
import mdtraj as md
import numpy as np
from maniprot.core.helpers import lcs, lcs_args
from maniprot.core.SO3 import SO3Manifold
from maniprot.core.pointcloud import PointcloudManifold

# Load trajectory
traj = md.load("run.xtc", top="protein.pdb")

# --- Orientation features (SO3) ---
xyz, n_idx, ca_idx, c_idx = lcs_args(traj)
R = lcs(xyz, n_idx, ca_idx, c_idx)            # (n_frames, n_residues, 3, 3)

so3 = SO3Manifold().fit(R)
R_feat = so3.transform(R)                      # (n_frames, n_residues, 3)

# --- Point-cloud manifold features ---
ca_idx_all = [ca for ca in ca_idx]
X = traj.xyz[:, ca_idx_all, :]                 # (n_frames, n_residues, 3)

pc = PointcloudManifold(delta=1.0).fit(X)
X_feat = pc.transform(X)                       # (n_frames, n_residues, 3)

# Concatenate and feed to a downstream model
features = np.concatenate([
    R_feat.reshape(len(traj), -1),
    X_feat.reshape(len(traj), -1),
], axis=-1)
```

---

## Module reference

| Module | Public symbols | Description |
|---|---|---|
| `maniprot.core.helpers` | `lcs`, `lcs_args` | LCS extraction and MDTraj bridge |
| `maniprot.core.SO3` | `SO3Manifold`, `exp`, `log`, `intrinsic_mean` | SO(3) geometry |
| `maniprot.core.SO3.utils` | *(internal)* | `_matmul3x3`, `_skew_vec` |
| `maniprot.core.pointcloud` | `PointcloudManifold`, `log`, `prelog`, `distance`, `metric_tensor`, `norm`, `intrinsic_mean` | Point-cloud manifold geometry |
| `maniprot.core.pointcloud.utils` | *(internal)* | `_det3x3`, `_inv3x3`, `_offsets`, `_gyration_matrix`, `pair_from_index` |

---
