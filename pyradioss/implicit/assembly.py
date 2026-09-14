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

#: element kernels that expose an implicit ``tangent()``: hexa8 + BT4 since
#: M8, the truss since M9, and — M11 (element tangent COMPLETENESS) — the
#: 4-node tetra, the 3-node C0 triangle shell, the corotational Timoshenko
#: beam and the TYPE4 spring: every element family of the port. The gate
#: stays (a future un-ported family must still fail loudly here, never be
#: silently ignored).
_TANGENT_KERNELS = tuple(KERNELS.keys())


def element_triplets(name, group, x_geom, dof: DofMap, epsp_incr=None,
                     kgeo=False):
    """Return (rows, cols, vals) COO triplets in EQUATION space for one
    element group, by calling the kernel's ``tangent()`` and mapping its
    global-DOF addressing through the equation numbering.

    ``x_geom`` is the geometry at which to linearize (the same geometry the
    residual is evaluated at, so the tangent is consistent with f_int: the
    committed reference frame for the M8 small-strain path, the CURRENT
    trial geometry for the M9 nonlinear-geometry path). ``epsp_incr`` (per
    element) is the plastic-strain increment of the current load step,
    forwarded to the kernel for the LAW2 consistent tangent; ignored by
    elastic groups. ``kgeo=True`` (M9, /IMPL/NONLIN) ADDS the geometric
    (initial-stress) stiffness ``kernel.kgeo`` built from the current stress
    state — the imp_kgeo branch of the original assembly."""
    kernel = KERNELS[name]
    if not hasattr(kernel, "tangent"):
        raise NotImplementedError(
            f"element group '{name}' has no implicit tangent() — the "
            f"implicit solver supports {_TANGENT_KERNELS}. Remove the "
            f"element type or run the explicit solver.")
    ke, edofs = kernel.tangent(group, x_geom, epsp_incr)
    if kgeo:
        kg, _ = kernel.kgeo(group, x_geom)
        ke = ke + kg
    # ke: (n, d, d)   edofs: (n, d) global scalar slot ids
    n, d, _ = ke.shape
    # map each local DOF to its equation index (-1 = condensed/fixed/virtual)
    valid_dof = edofs >= 0
    eq = np.full_like(edofs, -1)
    eq[valid_dof] = dof.eq[edofs[valid_dof]]           # (n, d)
    # broadcast to all (i, j) local pairs
    row_eq = np.repeat(eq[:, :, None], d, axis=2)     # (n, d, d) row eqns
    col_eq = np.repeat(eq[:, None, :], d, axis=1)     # (n, d, d) col eqns
    keep = (row_eq >= 0) & (col_eq >= 0)              # drop fixed rows/cols
    return (row_eq[keep].ravel(),
            col_eq[keep].ravel(),
            ke[keep].ravel())


def assemble(model, dof: DofMap, x_geom, epsp_incr=None, log=None,
             kgeo=False):
    """Assemble the global tangent K (CSR, ndof x ndof) from every element
    group's element tangent, linearized at geometry ``x_geom``. See the
    module docstring for the scatter.

    ``epsp_incr`` maps group name -> per-element plastic-strain increment
    (for the LAW2 consistent tangent); ``None`` means all-elastic.
    ``kgeo=True`` adds the geometric (initial-stress) stiffness of each
    element to its material+hourglass tangent (M9, /IMPL/NONLIN)."""
    sp, _ = require_scipy()
    rows, cols, vals = [], [], []
    for name, group in model.element_groups():
        if name not in _TANGENT_KERNELS:
            raise NotImplementedError(
                f"element group '{name}' is not supported by the implicit "
                f"solver (supported: {_TANGENT_KERNELS}).")
        ei = None if epsp_incr is None else epsp_incr.get(name)
        r, c, v = element_triplets(name, group, x_geom, dof, ei, kgeo)
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


#: element kernels that expose ``consistent_mass()`` — M16 adds the parallel
#: mass operator to every family that carries a tangent (all of them).
_MASS_KERNELS = _TANGENT_KERNELS


def assemble_mass(model, dof: DofMap, x_geom, log=None):
    """Assemble the global CONSISTENT mass matrix M (CSR, ndof x ndof) from
    every element group's ``consistent_mass()`` — the M16 modal analogue of
    ``assemble`` (the tangent scatter), same COO->CSR path through the DofMap.

    Fortran origin: there is no single ``imp_mass.F`` in the open-source engine
    (the implicit dynamics of imp_dyna.F uses the LUMPED MS/IN diagonal); this
    is the consistent-mass assembly the modal eigensolver of ``implicit.modal``
    needs, built exactly like the stiffness assembly so the (K, M) pencil is
    stated on the same equation numbering.

    ``x_geom`` is the geometry at which the (frame-dependent) element masses
    are evaluated — model.x0 for the standard modal problem (mass conserved on
    the reference geometry). The isotropic element masses (solids, shells,
    truss, spring) ignore it; only the corotational beam reads it, to orient
    its local mass with the same frame as its stiffness. This is a NEW
    operator built ALONGSIDE the lumped mass (``dynamics._lumped_mass_eq``),
    which stays bit-identical — the consistent mass never enters the explicit
    or implicit-dynamics time-marching paths, only the opt-in modal solve."""
    sp, _ = require_scipy()
    rows, cols, vals = [], [], []
    for name, group in model.element_groups():
        if name not in _MASS_KERNELS:
            raise NotImplementedError(
                f"element group '{name}' has no consistent_mass() — the modal "
                f"eigensolver supports {_MASS_KERNELS}.")
        kernel = KERNELS[name]
        me, edofs = kernel.consistent_mass(group, x_geom)
        n, d, _ = me.shape
        valid_dof = edofs >= 0
        eq = np.full_like(edofs, -1)
        eq[valid_dof] = dof.eq[edofs[valid_dof]]
        row_eq = np.repeat(eq[:, :, None], d, axis=2)
        col_eq = np.repeat(eq[:, None, :], d, axis=1)
        keep = (row_eq >= 0) & (col_eq >= 0)          # drop condensed DOFs
        rows.append(row_eq[keep].ravel())
        cols.append(col_eq[keep].ravel())
        vals.append(me[keep].ravel())

    # Point masses from /ADMAS (M5) and /ADMAS/NON_UNIFORM (M114) (AUD-026)
    for am in getattr(model, "admas", []):
        g = model.node_groups.get(am.grnod_id)
        if g is None or g.node_idx is None:
            continue
        m_per_node = am.mass if am.mass_type == 0 else (am.mass / max(1, len(g.node_idx)))
        for n_idx in g.node_idx:
            for d in range(3):
                eq_num = dof.eq[n_idx * 6 + d]
                if eq_num >= 0:
                    rows.append(np.array([eq_num], dtype=np.int64))
                    cols.append(np.array([eq_num], dtype=np.int64))
                    vals.append(np.array([m_per_node], dtype=np.float64))

    for an in getattr(model, "admas_non_uniforms", {}).values():
        if an.kind == "NODE":
            for item in getattr(an, "items", []):
                try:
                    n_idx = model.node_index(item.entity_id)
                except (KeyError, ValueError):
                    continue
                for d in range(3):
                    eq_num = dof.eq[n_idx * 6 + d]
                    if eq_num >= 0:
                        rows.append(np.array([eq_num], dtype=np.int64))
                        cols.append(np.array([eq_num], dtype=np.int64))
                        vals.append(np.array([item.mass], dtype=np.float64))
        elif an.kind == "PART":
            for item in getattr(an, "items", []):
                part_id = item.entity_id
                part_nodes = set()
                for _, grp in model.element_groups():
                    mask = grp.state.get("part_ids") == part_id
                    if np.any(mask):
                        conn = grp.state.get("mass_conn", grp.conn)[mask]
                        valid = conn[conn >= 0]
                        part_nodes.update(valid.tolist())
                if not part_nodes:
                    continue
                m_per_node = item.mass / len(part_nodes)
                for n_idx in part_nodes:
                    for d in range(3):
                        eq_num = dof.eq[n_idx * 6 + d]
                        if eq_num >= 0:
                            rows.append(np.array([eq_num], dtype=np.int64))
                            cols.append(np.array([eq_num], dtype=np.int64))
                            vals.append(np.array([m_per_node], dtype=np.float64))

    rows = np.concatenate(rows) if rows else np.zeros(0, dtype=np.int64)
    cols = np.concatenate(cols) if cols else np.zeros(0, dtype=np.int64)
    vals = np.concatenate(vals) if vals else np.zeros(0)
    M = sp.coo_matrix((vals, (rows, cols)),
                      shape=(dof.ndof, dof.ndof)).tocsr()
    if log is not None:
        log.info(f" CONSISTENT MASS NNZ (ASSEMBLED) . . : {M.nnz}")
    return M


def assemble_kgeo(model, dof: DofMap, x_geom):
    """Assemble the geometric (initial-stress) stiffness K_geo ALONE (CSR),
    from the current element stress states at geometry ``x_geom`` — the
    matrix pair (K_material, K_geo) is what the linearized-buckling
    eigenproblem of ``implicit.buckling`` needs (imp_buck.F / /IMPL/BUCKL
    analogue). The regular Newton path never calls this: it gets K_geo added
    into ``assemble(..., kgeo=True)`` instead."""
    sp, _ = require_scipy()
    rows, cols, vals = [], [], []
    for name, group in model.element_groups():
        if name not in _TANGENT_KERNELS:
            raise NotImplementedError(
                f"element group '{name}' is not supported by the implicit "
                f"solver (supported: {_TANGENT_KERNELS}).")
        kernel = KERNELS[name]
        kg, edofs = kernel.kgeo(group, x_geom)
        n, d, _ = kg.shape
        valid_dof = edofs >= 0
        eq = np.full_like(edofs, -1)
        eq[valid_dof] = dof.eq[edofs[valid_dof]]
        row_eq = np.repeat(eq[:, :, None], d, axis=2)
        col_eq = np.repeat(eq[:, None, :], d, axis=1)
        keep = (row_eq >= 0) & (col_eq >= 0)
        rows.append(row_eq[keep].ravel())
        cols.append(col_eq[keep].ravel())
        vals.append(kg[keep].ravel())
    rows = np.concatenate(rows) if rows else np.zeros(0, dtype=np.int64)
    cols = np.concatenate(cols) if cols else np.zeros(0, dtype=np.int64)
    vals = np.concatenate(vals) if vals else np.zeros(0)
    return sp.coo_matrix((vals, (rows, cols)),
                         shape=(dof.ndof, dof.ndof)).tocsr()
