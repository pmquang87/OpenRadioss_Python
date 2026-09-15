"""
Follower-load (pressure) stiffness for /PLOAD under /IMPL/NONLIN (M13).

Fortran origin
--------------
``engine/source/implicit/imp_glob_k.F`` — ``IMP_KPRES`` (called from the
implicit assembly around the concentrated/pressure load table NCONLD) with
its element kernels ``KPQUAD`` (bilinear quad, 2x2 Gauss) and ``KPTRIA``.
Checked against the source, not docs: the original DOES assemble a
pressure load stiffness, but a deliberately reduced one —

* only the OFF-DIAGONAL 3x3 node blocks are assembled (`PUT_KIJ`; every
  `PUT_KII` call is commented out in the source),
* each block is forced antisymmetric (K12(2,1) = -K12(1,2) ...): what is
  kept is the SKEW part of the consistent load stiffness of the
  shape-function-consistent pressure force, scaled by ``SCALN = HALF``
  (the symmetrized-operator trick: for a pressure field enclosing a
  volume the symmetric part of the load stiffness assembles to a
  boundary term, so half the skew operator is the classical
  "conservative-load symmetrization" — see Hibbitt 1979 / Bonet & Wood
  §9.5 on pressure load stiffness symmetry).

The port deliberately DIFFERS (documented deviation): it assembles the
EXACT linearization of ITS OWN /PLOAD discretization instead. Newton's
convergence rate is governed by consistency between the assembled tangent
and the residual actually iterated — and the port's /PLOAD residual
(``engine/kinematics.py``) is not the original's Gauss-integrated force
but the lumped area-vector form

    f_a = p(t) * w_a * A,      A = 1/2 (x3 - x1) x (x4 - x2)

(w_a the corner lumping weights, 1/4 per quad corner / 1/3 per triangle
corner — the diagonal cross product is exact for the bilinear quad AND
the degenerate n4 = n3 triangle). Its exact derivative is

    d f_a / d x_b = p w_a 1/2 [ (d_b3 - d_b1) (-[d24]x)
                              + (d_b4 - d_b2) ( [d13]x) ]

with d13 = x3 - x1, d24 = x4 - x2 the diagonals, [.]x the cross-product
matrix and d_bk the Kronecker column selector — a NONSYMMETRIC 12x12
block per segment (follower loads are non-conservative on an open
surface; the direct solver is LU and does not care). The load stiffness
enters the global tangent as K += -d f_ext/d x (the driver solves
K du = R with R = f_ext + f_int), HHT-weighted like the rest of the
tangent under /IMPL/DYNA.

WHEN it applies: only under /IMPL/NONLIN (both statics and dynamics),
where M13 also moves the /PLOAD residual evaluation from the committed
frame to the TRIAL configuration (the M12 deferral this module removes:
the committed-frame force lagged the geometry by one increment and its
configuration dependence was invisible to K). The small-displacement
path (M8 default) keeps the dead x0 pressure — configuration dependence
is a geometric nonlinearity by definition. /PLOAD combined with
/IMPL/ARCL is REFUSED (statics driver): a follower pressure violates the
proportional-loading assumption f_ext = lambda*q the arc-length
continuation is built on.

Element deletion and /SENSOR follow the residual: deleted segments carry
neither force nor stiffness; sensors are not evaluated under the
implicit clock (the load is active for the whole run, as the interface
warning documents).

The ``impl_load_stiff`` control (default True, no input card) exists so
the validation can run the SAME residual without the tangent term and
assert the Newton iteration count drops when it is on — the converged
answer must agree to the Newton tolerance (the residual, which defines
the equilibrium, is identical).
"""

from __future__ import annotations

import numpy as np

from ..contact import tracking
from .dofmap import DOFS_PER_NODE


def has_follower(loads) -> bool:
    """True when the resolved load set carries at least one /PLOAD with
    segments — the gate for the trial-configuration residual evaluation
    and the load-stiffness assembly under /IMPL/NONLIN."""
    # getattr: the arc-length driver's final step passes a minimal load
    # shim (_ArcLoads) with no /PLOAD table — /PLOAD + /IMPL/ARCL is
    # refused upstream, so the shim is by construction pressure-free
    return any(len(pl[0]) for pl in getattr(loads, "ploads", ()))


def _skew(v):
    """Cross-product matrices [v]x of a vector batch (n, 3) -> (n, 3, 3),
    such that [v]x w = v x w."""
    n = len(v)
    S = np.zeros((n, 3, 3))
    S[:, 0, 1] = -v[:, 2]
    S[:, 0, 2] = v[:, 1]
    S[:, 1, 0] = v[:, 2]
    S[:, 1, 2] = -v[:, 0]
    S[:, 2, 0] = -v[:, 1]
    S[:, 2, 1] = v[:, 0]
    return S


def pload_tangent(loads, model, t, x, dof):
    """Assembled follower-load stiffness -d f_pload/d x (CSR, ndof x ndof)
    of every /PLOAD at time/load-factor ``t`` and trial geometry ``x`` —
    the exact linearization of the residual's own lumped area-vector
    pressure force (module docstring). Added to the element tangent by
    the drivers when /IMPL/NONLIN is active."""
    from . import require_scipy
    sp, _ = require_scipy()
    rows_all, cols_all, vals_all = [], [], []
    # column selectors of the diagonal derivatives (module docstring):
    # d13 = x[2] - x[0] responds to corners 2/0, d24 = x[3] - x[1] to 3/1
    e13 = np.array([-1.0, 0.0, 1.0, 0.0])
    e24 = np.array([0.0, -1.0, 0.0, 1.0])
    for segs, wgt, fct, scale, gtype, elem, deletable, sens in loads.ploads:
        if len(segs) == 0:
            continue
        p = scale * fct.eval(t)
        if p == 0.0:
            continue
        xs = x[segs].copy()                              # (nseg, 4, 3)
        degen = (segs[:, 3] == segs[:, 2]) | (segs[:, 3] <= 0)
        if np.any(degen):
            xs[degen, 3] = xs[degen, 2]
        C13 = _skew(xs[:, 2] - xs[:, 0])                 # [d13]x
        C24 = _skew(xs[:, 3] - xs[:, 1])                 # [d24]x
        w = wgt
        if deletable:
            alive = tracking.alive_segment_mask(model, gtype, elem)
            w = np.where(alive[:, None], wgt, 0.0)       # torn face: no load
        # d f_a/d x_b = p w_a 1/2 [ e13_b (-[d24]x) + e24_b [d13]x ]
        dAdx = 0.5 * (e24[None, :, None, None] * C13[:, None, :, :]
                      - e13[None, :, None, None] * C24[:, None, :, :])
        # (nseg, 4(b), 3, 3) -> the full (nseg, 4(a), 3, 4(b), 3) block,
        # NEGATED: the load stiffness is K += -d f_ext/d x
        ke = -p * np.einsum("na,nbij->naibj", w, dAdx).reshape(
            len(segs), 12, 12)
        edofs = np.zeros((len(segs), 12), dtype=np.int64)
        for k in range(4):
            for c in range(3):
                edofs[:, 3 * k + c] = segs[:, k] * DOFS_PER_NODE + c
        if np.any(degen & (segs[:, 3] <= 0)):
            edofs[degen & (segs[:, 3] <= 0), 9:] = -1
        eq = np.full(edofs.shape, -1, dtype=np.int64)
        valid = (edofs >= 0) & (edofs < len(dof.eq))
        eq[valid] = dof.eq[edofs[valid]]
        row_eq = np.repeat(eq[:, :, None], 12, axis=2)
        col_eq = np.repeat(eq[:, None, :], 12, axis=1)
        keep = (row_eq >= 0) & (col_eq >= 0)
        rows_all.append(row_eq[keep].ravel())
        cols_all.append(col_eq[keep].ravel())
        vals_all.append(ke[keep].ravel())
    rows = (np.concatenate(rows_all) if rows_all
            else np.zeros(0, dtype=np.int64))
    cols = (np.concatenate(cols_all) if cols_all
            else np.zeros(0, dtype=np.int64))
    vals = np.concatenate(vals_all) if vals_all else np.zeros(0)
    return sp.coo_matrix((vals, (rows, cols)),
                         shape=(dof.ndof, dof.ndof)).tocsr()
