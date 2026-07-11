"""
Global sparse tangent-stiffness assembly (M8).

Fortran origin: ``engine/source/implicit/imp_glob_k.F`` / ``imp_fsa_inv.F``
— where OpenRadioss walks the element groups, asks each element for its local
tangent, and scatters the entries into the global sparse matrix addressed by
the equation numbering of ``ind_glob_k.F``. This module is the port of that
scatter.

The assembly is the classic finite-element operation

    K = A_e  P_e^T k_e P_e

("A" the assembly operator, P_e the element's DOF-gather). In practice each
element kernel returns, for a whole group at once:

* ``ke``    — the dense element tangents, shape (n_elem, ndof_e, ndof_e);
* ``edofs`` — the GLOBAL scalar DOF slot id of each local element DOF, shape
              (n_elem, ndof_e), in the ``dofmap.global_dof`` numbering
              (node*6 + component).

Assembly then, for every element and every (i, j) local pair, maps the two
global slots to equation indices through the ``DofMap``; pairs where either
slot is condensed (fixed → equation -1) are simply DROPPED — this is how the
/BCS conditions are enforced by *condensation* (the fixed rows/cols never
enter K) rather than by penalty. The surviving (row, col, value) triplets are
concatenated into COO arrays and handed to ``scipy.sparse`` to build the CSR
matrix (which sums duplicate (row, col) entries — exactly the assembly sum).

scipy is obtained through ``implicit.require_scipy`` (never a top-level
import) so the base explicit install stays NumPy-only.
"""

from __future__ import annotations

import numpy as np

from ..elements import KERNELS
from . import require_scipy
from .dofmap import DofMap

#: element kernels that expose an implicit ``tangent()`` (M8): the 8-node
#: solid (hexa8) and the 4-node shell (BT4). Groups outside this set raise a
#: clear error in ``assemble`` — the implicit path does not silently ignore
#: un-ported element types (tetra4 / sh3n / beam / truss / spring tangents are
#: deferred, see PORTING_GUIDE).
_TANGENT_KERNELS = ("bricks", "shells")


def element_triplets(name, group, x_geom, dof: DofMap, epsp_incr=None):
    """Return (rows, cols, vals) COO triplets in EQUATION space for one
    element group, by calling the kernel's ``tangent()`` and mapping its
    global-DOF addressing through the equation numbering.

    ``x_geom`` is the geometry at which to linearize (the small-strain
    reference frame — the same geometry the residual is evaluated at, so the
    tangent is consistent with f_int). ``epsp_incr`` (per element) is the
    plastic-strain increment of the current load step, forwarded to the
    kernel for the LAW2 consistent tangent; ignored by elastic groups."""
    kernel = KERNELS[name]
    if not hasattr(kernel, "tangent"):
        raise NotImplementedError(
            f"element group '{name}' has no implicit tangent() — the M8 "
            f"implicit solver supports {_TANGENT_KERNELS}. Remove the "
            f"element type or run the explicit solver.")
    ke, edofs = kernel.tangent(group, x_geom, epsp_incr)
    # ke: (n, d, d)   edofs: (n, d) global scalar slot ids
    n, d, _ = ke.shape
    # map each local DOF to its equation index (-1 = condensed/fixed)
    eq = dof.eq[edofs]                        # (n, d)
    # broadcast to all (i, j) local pairs
    row_eq = np.repeat(eq[:, :, None], d, axis=2)     # (n, d, d) row eqns
    col_eq = np.repeat(eq[:, None, :], d, axis=1)     # (n, d, d) col eqns
    keep = (row_eq >= 0) & (col_eq >= 0)              # drop fixed rows/cols
    return (row_eq[keep].ravel(),
            col_eq[keep].ravel(),
            ke[keep].ravel())


def assemble(model, dof: DofMap, x_geom, epsp_incr=None, log=None):
    """Assemble the global tangent K (CSR, ndof x ndof) from every element
    group's element tangent, linearized at geometry ``x_geom`` (the
    small-strain reference frame). See the module docstring for the scatter.

    ``epsp_incr`` maps group name -> per-element plastic-strain increment
    (for the LAW2 consistent tangent); ``None`` means all-elastic."""
    sp, _ = require_scipy()
    rows, cols, vals = [], [], []
    for name, group in model.element_groups():
        if name not in _TANGENT_KERNELS:
            raise NotImplementedError(
                f"element group '{name}' is not supported by the M8 implicit "
                f"solver (supported: {_TANGENT_KERNELS}).")
        ei = None if epsp_incr is None else epsp_incr.get(name)
        r, c, v = element_triplets(name, group, x_geom, dof, ei)
        rows.append(r)
        cols.append(c)
        vals.append(v)
    rows = np.concatenate(rows) if rows else np.zeros(0, dtype=np.int64)
    cols = np.concatenate(cols) if cols else np.zeros(0, dtype=np.int64)
    vals = np.concatenate(vals) if vals else np.zeros(0)
    # coo_matrix -> csr SUMS duplicate (row, col) entries: that summation IS
    # the finite-element assembly of overlapping element contributions.
    K = sp.coo_matrix((vals, (rows, cols)),
                      shape=(dof.ndof, dof.ndof)).tocsr()
    if log is not None:
        log.info(f" TANGENT NNZ (ASSEMBLED)  . . . . . . : {K.nnz}")
    return K
