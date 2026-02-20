"""
SO3/intrinsic_mean.py – Fréchet (intrinsic) mean on SO(3)
==========================================================
Computes the Fréchet mean of a set of rotation matrices on SO(3) using a
fully Numba-parallel implementation of Riemannian gradient descent.

Mathematical background
-----------------------
The Fréchet mean μ ∈ SO(3) minimises the sum of squared geodesic distances
to a set of rotations {R₁, …, Rₙ}:

    μ* = argmin_μ  Σᵢ  ‖log_μ(Rᵢ)‖²

The iterative algorithm is:

    gradient = mean_i [ log_μ(Rᵢ) ]         ← average tangent vector
    μ ← exp_μ( lr · gradient )              ← retract back onto SO(3)

Convergence is checked per-dimension: each of the ``n_dim`` rotation slots
tracks its own gradient norm and stops updating independently once it falls
below ``threshold``.  This avoids redundant computation when some dimensions
converge faster than others.

Parallel reduction strategy
---------------------------
The inner loop (accumulating log-maps across the batch) runs in parallel via
``nb.prange``.  To avoid race conditions on the accumulator, a
**thread-local sum** array of shape ``(n_threads, n_dim, 3)`` is used: each
thread writes only to its own row, and the per-thread sums are collapsed after
the parallel loop.  No atomic operations or locks are required.

Public API
----------
intrinsic_mean(Rs, R_ref, learning_rate, threshold, max_steps) -> np.ndarray
    Fully Numba-parallel Fréchet mean on SO(3).
"""

import numba as nb
import numpy as np

from .exp import _exp
from .log import _log_acc


# ---------------------------------------------------------------------------
# Private helper (Numba-compiled, not part of the public API)
# ---------------------------------------------------------------------------

@nb.njit(nogil=True)
def _intrinsic_mean_core(Rs: np.ndarray, R_ref: np.ndarray):
    """Validate inputs and pre-allocate all working buffers.

    Separating allocation from the main loop keeps the hot path clean and
    allows Numba to compile the parallel kernel without hidden allocations.

    Parameters
    ----------
    Rs : np.ndarray, shape (n_samples, n_dim, 3, 3)
        Batch of rotation matrices.
    R_ref : np.ndarray, shape (n_dim, 3, 3)
        Initial estimate of the Fréchet mean (one rotation per dimension slot).

    Returns
    -------
    A tuple of pre-allocated arrays and scalar dimensions used by the main
    kernel.  See ``intrinsic_mean`` for their roles.

    Raises
    ------
    ValueError
        If ``R_ref`` is not shaped ``(n_dim, 3, 3)`` or ``Rs`` is not shaped
        ``(n_samples, n_dim, 3, 3)``.
    """
    n_threads = nb.get_num_threads()

    if R_ref.ndim != 3 or R_ref.shape[-2:] != (3, 3):
        raise ValueError(f"R_ref must have shape (n_dim, 3, 3), got {R_ref.shape}")

    n_dim = R_ref.shape[0]

    if Rs.ndim != 4 or Rs.shape[-3:] != (n_dim, 3, 3):
        raise ValueError(f"Rs must have shape (n_samples, n_dim, 3, 3), got {Rs.shape}")

    batch_size = Rs.shape[0]

    # Flatten (n_samples, n_dim, 3, 3) -> (n_samples * n_dim, 3, 3) for prange
    flat_Rs    = Rs.reshape(batch_size * n_dim, 3, 3)
    flat_R_ref = R_ref.reshape(n_dim, 3, 3)

    # Per-sample working buffers for the log computation
    flat_R_aligned_log       = np.empty((batch_size * n_dim, 3, 3), dtype=Rs.dtype)  # DeltaR = R_ref^T . R
    flat_R_minus_R_t_vee_log = np.empty((batch_size * n_dim, 3),    dtype=Rs.dtype)  # skew_vec(DeltaR)

    # Thread-local accumulator: shape (n_threads, n_dim, 3)
    # Each thread accumulates its partial sum of log-map vectors independently,
    # eliminating the need for atomic additions or locks.
    thread_log_sum = np.empty((n_threads, n_dim, 3), dtype=Rs.dtype)

    # Gradient and per-dimension convergence tracking
    grad      = np.empty((n_dim, 3), dtype=Rs.dtype)
    grad_norm = np.empty((n_dim,),   dtype=Rs.dtype)

    # Working buffers for the exp retraction step (one set per dimension slot)
    v_skew_exp    = np.empty((n_dim, 3, 3), dtype=Rs.dtype)  # [v]x
    v_skew_sq_exp = np.empty((n_dim, 3, 3), dtype=Rs.dtype)  # [v]x^2
    R_aligned_exp = np.empty((n_dim, 3, 3), dtype=Rs.dtype)  # Exp(v)

    # Output: current mean estimate, to be initialised to R_ref by the caller
    res = np.empty((n_dim, 3, 3), dtype=Rs.dtype)

    return (
        flat_Rs, flat_R_ref,
        flat_R_aligned_log, flat_R_minus_R_t_vee_log,
        thread_log_sum,
        grad, grad_norm,
        v_skew_exp, v_skew_sq_exp, R_aligned_exp,
        res,
        batch_size, n_dim, n_threads,
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

@nb.njit(nogil=True, parallel=True)
def intrinsic_mean(
    Rs: np.ndarray,
    R_ref: np.ndarray,
    learning_rate: float,
    threshold: float,
    max_steps: int,
) -> np.ndarray:
    """Compute the Fréchet mean of a batch of SO(3) rotation matrices.

    Uses a fully Numba-parallel Riemannian gradient descent with thread-local
    accumulation to avoid synchronisation overhead.  Convergence is tracked
    independently per dimension slot, so slots that have already converged are
    skipped in subsequent iterations.

    Parameters
    ----------
    Rs : np.ndarray, shape (n_samples, n_dim, 3, 3)
        Batch of rotation matrices whose Fréchet mean is sought.  ``n_dim``
        is the number of independent rotation slots (e.g. residues), each
        converging to its own mean independently.
    R_ref : np.ndarray, shape (n_dim, 3, 3)
        Initial estimate of the mean for each dimension slot.  A good default
        is ``Rs[0]`` or the identity matrix broadcast to ``(n_dim, 3, 3)``.
    learning_rate : float
        Step size for the gradient descent update applied via ``exp``.  Larger
        values converge faster but may overshoot on dispersed datasets.
    threshold : float
        Per-dimension convergence criterion.  A slot is considered converged
        when its gradient norm (||mean log-map vector||_2) drops below this
        value and is excluded from further updates.
    max_steps : int
        Maximum number of gradient-descent iterations regardless of convergence.

    Returns
    -------
    np.ndarray, shape (n_dim, 3, 3)
        Estimated Fréchet mean rotation matrix for each dimension slot.

    Notes
    -----
    Convergence is guaranteed when all data lie within a geodesic ball of
    radius pi/2 around the true mean (the injectivity radius of SO(3)).

    All working memory is allocated once in ``_intrinsic_mean_core`` and
    reused across iterations, keeping the hot path allocation-free.

    Examples
    --------
    >>> R_mean = intrinsic_mean(Rs, Rs[0], learning_rate=1e-2,
    ...                         threshold=1e-4, max_steps=128)
    >>> R_mean.shape
    (n_dim, 3, 3)
    """
    (flat_Rs, flat_R_ref,
     flat_R_aligned_log, flat_R_minus_R_t_vee_log,
     thread_log_sum,
     grad, grad_norm,
     v_skew_exp, v_skew_sq_exp, R_aligned_exp,
     res,
     batch_size, n_dim, n_threads) = _intrinsic_mean_core(Rs, R_ref)

    # Initialise: all dimension slots are active; mean starts at R_ref
    grad_norm[:] = float("inf")
    res[:]       = flat_R_ref

    for _ in range(max_steps):

        # ------------------------------------------------------------------
        # Step 1 – parallel log-map accumulation (thread-local reduction)
        # ------------------------------------------------------------------
        # Each thread accumulates log_{res[c]}(Rs[i]) into thread_log_sum[t, c].
        # Because each thread exclusively owns its row t, no atomics are needed.
        thread_log_sum[:] = 0.0

        for i in nb.prange(batch_size * n_dim):
            t = nb.get_thread_id()   # owning thread index
            c = i % n_dim            # dimension slot for this (sample, dim) pair

            if grad_norm[c] > threshold:
                _log_acc(
                    flat_Rs[i], res[c],
                    flat_R_aligned_log[i], flat_R_minus_R_t_vee_log[i],
                    thread_log_sum[t, c],  # thread-private accumulator slice
                )

        # Collapse per-thread partial sums into the batch-mean gradient
        # (sequential over n_threads; negligible cost vs. the parallel body)
        grad = thread_log_sum.sum(0) / batch_size

        # ------------------------------------------------------------------
        # Step 2 – per-dimension convergence check (parallel)
        # ------------------------------------------------------------------
        for c in nb.prange(n_dim):
            if grad_norm[c] > threshold:
                g = grad[c]
                grad_norm[c] = np.sqrt(g[0]*g[0] + g[1]*g[1] + g[2]*g[2])

        # ------------------------------------------------------------------
        # Step 3 – gradient retraction onto SO(3) via exp (parallel)
        # ------------------------------------------------------------------
        # Only unconverged slots are updated; converged slots are left intact.
        for c in nb.prange(n_dim):
            if grad_norm[c] > threshold:
                _exp(
                    learning_rate * grad[c], res[c],
                    v_skew_exp[c], v_skew_sq_exp[c], R_aligned_exp[c],
                    res[c],  # res[c] updated in-place
                )

    return res