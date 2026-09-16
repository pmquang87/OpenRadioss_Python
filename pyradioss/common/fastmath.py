"""
Small-array NumPy primitives for the cycle path (M7 performance work).

Why this module exists
----------------------
The pyradioss examples run models of tens to hundreds of elements — the
regime this educational port is used in. Profiling (see PORTING_GUIDE §5,
M7) showed the explicit cycle is dominated not by floating-point work but
by *per-call NumPy overhead* on such small arrays:

* ``np.cross`` spends ~20 µs/call in ``moveaxis``/axis-normalization
  bookkeeping before doing 9 multiplies — 20% of the rigid_impactor
  runtime was np.cross overhead;
* ``np.linalg.det``/``inv`` on stacked (n, 3, 3) Jacobians go through
  LAPACK dispatch (~90 µs/call for n≈100 where the arithmetic is ~1 µs);
* ``np.add.at`` (the force scatter-assembly, Fortran ``asspar``) is the
  documented-slow ufunc path, ~4× slower than a ``bincount`` per
  component.

The replacements below are *formula-identical*: they perform the same
IEEE operations in the same order as the NumPy calls they replace, so
they are **bitwise identical** where noted (verified by tests and by the
M7 micro-benchmarks recorded in the porting guide). Where a replacement
does reassociate a reduction it says so explicitly in its docstring —
that is the "documented per kernel" part of the M7 parity contract.

These helpers are the *NumPy backend's* primitives. The optional numba
backend (``pyradioss.accel``) mirrors whole kernel blocks instead — see
the accel package docstring for the backend architecture.
"""

from __future__ import annotations

import numpy as np

from ..accel import get as _accel_get


def cross3(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Cross product of (..., 3) arrays (broadcasting like ``np.cross``).

    Bitwise identical to ``np.cross`` (same multiply/subtract per
    component), ~2× faster on small arrays: it skips np.cross's
    moveaxis/axis-normalization overhead."""
    a0, a1, a2 = a[..., 0], a[..., 1], a[..., 2]
    b0, b1, b2 = b[..., 0], b[..., 1], b[..., 2]
    out = np.empty(np.broadcast(a, b).shape[:-1] + (3,))
    out[..., 0] = a1 * b2 - a2 * b1
    out[..., 1] = a2 * b0 - a0 * b2
    out[..., 2] = a0 * b1 - a1 * b0
    return out


def norm3(a: np.ndarray) -> np.ndarray:
    """Euclidean norm over the LAST axis of an (..., 3) array.

    Bitwise identical to ``np.linalg.norm(a, axis=-1)`` for real input
    (|x|² == x·x exactly in IEEE, and the 3-term sum associates left to
    right in both)."""
    a0, a1, a2 = a[..., 0], a[..., 1], a[..., 2]
    return np.sqrt(a0 * a0 + a1 * a1 + a2 * a2)


def det_inv33(J: np.ndarray):
    """Determinant and inverse of stacked (n, 3, 3) matrices by explicit
    cofactor expansion.

    Replaces ``np.linalg.det`` + ``np.linalg.inv`` (LAPACK LU) on the
    element Jacobians: ~5× faster at group sizes of O(100) and — unlike
    LU with pivoting — an *arithmetically explicit* formula that the
    numba backend reproduces exactly. NOT bitwise identical to LAPACK
    (different algorithm); the difference is at machine precision
    (measured ≲1e-16 relative on well-conditioned element Jacobians,
    which starter checks guarantee — vol > 0)."""
    is_2d = (J.ndim == 2)
    if is_2d:
        J = J[None, :, :]
    a, b, c = J[:, 0, 0], J[:, 0, 1], J[:, 0, 2]
    d, e, f = J[:, 1, 0], J[:, 1, 1], J[:, 1, 2]
    g, h, i = J[:, 2, 0], J[:, 2, 1], J[:, 2, 2]
    A = e * i - f * h          # cofactors of the first column expansion
    B = f * g - d * i
    C = d * h - e * g
    det = a * A + b * B + c * C
    safe_det = np.where(np.abs(det) < 1.0e-20, np.where(det >= 0.0, 1.0e-20, -1.0e-20), det)
    idet = 1.0 / safe_det
    inv = np.empty_like(J, dtype=np.float64)
    inv[:, 0, 0] = A * idet
    inv[:, 0, 1] = (c * h - b * i) * idet
    inv[:, 0, 2] = (b * f - c * e) * idet
    inv[:, 1, 0] = B * idet
    inv[:, 1, 1] = (a * i - c * g) * idet
    inv[:, 1, 2] = (c * d - a * f) * idet
    inv[:, 2, 0] = C * idet
    inv[:, 2, 1] = (b * g - a * h) * idet
    inv[:, 2, 2] = (a * e - b * d) * idet
    if is_2d:
        return float(det[0]), inv[0]
    return det, inv


def scatter_add3(target: np.ndarray, idx: np.ndarray,
                 values: np.ndarray,
                 color_indices: np.ndarray = None,
                 color_offsets: np.ndarray = None) -> None:
    """``np.add.at(target, idx, values)`` for (m,) int indices into an
    (N, 3) target — the force scatter-assembly (Fortran ``asspar``).

    Implemented as one ``np.bincount`` per component: bincount
    accumulates in input order exactly like ``add.at``, and the combined
    per-node sums are added to ``target`` in one shot. For a node whose
    target row is zero this is bitwise identical to ``add.at``; for a
    node that already carries force from ANOTHER element group the final
    addition is reassociated — ((f+c1)+c2) becomes f+(c1+c2) — an
    ulp-level, deterministic difference (documented M7 reordering).
    Measured ~4× faster than ``np.add.at`` at cycle-path sizes.

    M39: when the numba backend is active, dispatch to ``accel.scatter3``,
    which fuses the three component passes into one pass over the index
    list (measured ~3× faster than the bincount reference on the 65 k-brick
    cliff) while reproducing THIS reference bit-for-bit — bincount
    accumulates each component in input order, and the numba mirror
    accumulates the same values in the same order into a zeroed scratch it
    then adds to ``target``, so cross-group additions associate identically
    (the accel-package parity contract; verified by tests/test_m7_backends
    on zero and non-zero targets). On the NumPy backend ``_accel_get``
    returns None and the bincount reference below runs unchanged (one dict
    lookup, the same negligible dispatch every kernel block already pays)."""
    
    if len(target) == 0 or len(idx) == 0:
        return

    n = len(target)
    valid = (idx >= 0) & (idx < n)
    if not np.all(valid):
        idx = idx[valid]
        values = values[valid]
        if len(idx) == 0:
            return
        color_indices = None
        color_offsets = None

    if color_indices is not None and color_offsets is not None and len(color_indices) > 0:
        jit = _accel_get("scatter3_colored")
        if jit is not None and len(idx) % len(color_indices) == 0:
            npe = len(idx) // len(color_indices)
            jit(target, idx, values, color_indices, color_offsets, npe)
            return

    jit = _accel_get("scatter3")
    if jit is not None:
        jit(target, idx, values)
        return

    # If colored fallback to numpy (for correctness we could group by color here, but 
    # bincount is so fast in serial that we just run it linearly).
    target[:, 0] += np.bincount(idx, weights=values[:, 0], minlength=n)
    target[:, 1] += np.bincount(idx, weights=values[:, 1], minlength=n)
    target[:, 2] += np.bincount(idx, weights=values[:, 2], minlength=n)
