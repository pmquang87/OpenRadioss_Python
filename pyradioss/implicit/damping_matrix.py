"""
Assembled global DAMPING matrix C — M18 (the source of NON-CLASSICAL damping).

Fortran origin
--------------
The open-source OpenRadioss engine assembles NO viscous damping matrix beyond
Rayleigh: ``engine/source/implicit/imp_dyna.F`` (IMP_DYKS/IMP_DYKV, the M11
port ``implicit/dynamics.py``) forms the damping force C v = a M v + b K v ON
THE FLY from the lumped mass and the step-start tangent — it never builds C as
a stored operator, and there is no assembled dashpot term anywhere in the
implicit path (freimpl.F reads only DAMPA_IMP / DAMPB_IMP). The discrete
viscous devices the deck can carry — the /PROP/SPRING dashpot ``c`` term
(rforc3.F, TYPE4) and the /DAMP mass-damper (damping*.F) — are handled in the
time domain (the spring dashpot is a rate device disabled under implicit, the
/DAMP mass damper is applied by the explicit integrating factor of
``engine/damping.py``); neither is ever assembled into a matrix.

M18 needs C AS AN OPERATOR: the damped eigenproblem (lambda^2 M + lambda C +
K) phi = 0 and the complex-mode superposition (``implicit/complex_modal.py``)
are stated on an assembled (K, C, M) pencil, exactly as M16's modal solve is
stated on (K, M). So this module ports the C analogue of M16's
``assemble_mass``: a global C built COO->CSR through the DofMap and condensed
T^T C T under the M12/M14 constraints, from three physical sources:

1. RAYLEIGH  C_R = alpha M + beta K  — the SAME alpha, beta the M11 direct
   integrator (/IMPL/DYNA/DAMP) feeds, with the CONSISTENT mass M
   (``assemble_mass``) and the tangent K (``assemble``). This is PROPORTIONAL
   (classical) damping: it diagonalizes in the M16 real modal basis with the
   ratio zeta_i = 1/2 (alpha/omega_i + beta omega_i) — the M17 Rayleigh map.
   On its own it produces NO complex modes (the whole point of the M17 real-
   mode superposition); it is here so the complex path REDUCES to the M17
   answer when the damping happens to be classical (the M18 consistency check).

2. DISCRETE SPRING DASHPOTS  — the /PROP/SPRING ``c`` term, assembled as the
   element damping matrix c a a^T (``spring.damping_matrix``) exactly like the
   spring's elastic tangent k a a^T. A dashpot on ONE spring among many is the
   canonical NON-CLASSICAL source: C is then not proportional to M or K, the
   real modal basis no longer diagonalizes it, and the damped modes go complex
   (a DOF-dependent phase lag — the signature this milestone extracts).

3. PER-NODE /DAMP MASS DAMPERS  — the /DAMP force f = -alpha_dp m v (M6,
   engine/damping.py) as its contribution to C: the diagonal alpha_dp * m_i on
   each translational equation of the group's nodes (alpha_dp * I_i on the
   rotational ones). Applied to a SUBSET of nodes this is again non-classical
   (mass-proportional only on part of the structure). The time WINDOW
   [tstart, tstop] of the card does not apply to an eigen/linearized analysis
   (a constant-coefficient system has no clock) — the constant alpha_dp is
   used and this is documented; a windowed /DAMP under the complex-mode solve
   is flagged in the listing.

Theory
------
Classical vs non-classical damping (Clough & Penzien, "Dynamics of
Structures", ch. 12-13; Caughey & O'Kelly 1965). The undamped modes phi_i
(M16) diagonalize both M and K. They ALSO diagonalize C iff C is a Caughey
series in M^-1 K — of which Rayleigh alpha M + beta K is the two-term case.
When they do, phi_i^T C phi_j = 2 zeta_i omega_i delta_ij and the equations
decouple into the real SDOFs the M17 mode superposition marches. When they do
NOT (a local dashpot, damping on part of the structure), phi_i^T C phi_j has
off-diagonal terms: the real modes are COUPLED through damping and the correct
free-vibration modes are COMPLEX (``implicit/complex_modal.py``). This module
builds the C that decides which regime a deck is in.

Deliberate deviations / deferrals (documented, not hidden)
----------------------------------------------------------
* Rayleigh C uses the CONSISTENT mass (``assemble_mass``), matching the M16/
  M17 modal pencil — NOT the lumped mass the M11 DIRECT integrator uses in
  imp_dyna.F. The two describe the same physical Rayleigh damping (same
  alpha, beta); they differ only in the mass model, exactly as the consistent
  and lumped modal spectra differ (the M16 bracket). The complex-mode solve is
  a modal capability, so it uses the modal (consistent) mass throughout —
  documented, and the pure-Rayleigh reproduction of the M17 zeta_i is
  validated against it (test_m18).
* the /DAMP time window is ignored (a linearized/eigen analysis has no time),
  and the beta (stiffness-proportional) branch of /DAMP is still absent
  upstream — /DAMP is mass-proportional only (entities.Damping).
"""

from __future__ import annotations

import numpy as np

from ..elements import KERNELS
from . import require_scipy
from .assembly import assemble, assemble_mass
from .dofmap import DofMap, DOFS_PER_NODE


#: element kernels that expose a viscous ``damping_matrix()``. Only the TYPE4
#: spring carries a discrete dashpot (the /PROP/SPRING ``c`` term); every other
#: family's viscosity lives in its material rate terms, not a nodal dashpot, so
#: they contribute nothing to the assembled C (their Rayleigh share comes
#: through alpha M + beta K). The tuple stays so a future dashpot-bearing
#: family (a TYPE13 spring, a /DASHPOT card) is added here explicitly.
_DASHPOT_KERNELS = ("springs",)


def assemble_discrete_damping(model, dof: DofMap, x_geom, log=None):
    """Assemble the DISCRETE viscous damping C_d (CSR, ndof x ndof) — the
    spring dashpots plus the /DAMP per-node mass dampers — WITHOUT the Rayleigh
    alpha M + beta K part (that is added by :func:`assemble_damping`). Built
    COO->CSR through the DofMap exactly like ``assemble`` / ``assemble_mass``.

    ``x_geom`` orients the spring dashpot axes (the current geometry — the
    dashpot resists the rate of axial stretch along the CURRENT axis, the same
    geometry the spring tangent linearizes at)."""
    sp, _ = require_scipy()
    rows, cols, vals = [], [], []

    # ---- (2) discrete spring dashpots: c a a^T element matrices -------------
    for name, group in model.element_groups():
        if name not in _DASHPOT_KERNELS:
            continue
        kernel = KERNELS[name]
        if not hasattr(kernel, "damping_matrix"):
            continue
        ce, edofs = kernel.damping_matrix(group, x_geom)
        n, d, _ = ce.shape
        eq = dof.eq[edofs]
        row_eq = np.repeat(eq[:, :, None], d, axis=2)
        col_eq = np.repeat(eq[:, None, :], d, axis=1)
        keep = (row_eq >= 0) & (col_eq >= 0)          # drop condensed DOFs
        rows.append(row_eq[keep].ravel())
        cols.append(col_eq[keep].ravel())
        vals.append(ce[keep].ravel())

    # ---- (3) /DAMP per-node mass dampers: diagonal alpha_dp * m_i -----------
    # /DAMP is mass-proportional (f = -alpha m v), so its contribution to C is
    # the diagonal alpha_dp * m on each node's translational equations and
    # alpha_dp * inertia on its rotational ones — the "mass" part of a
    # per-group Rayleigh damping restricted to the group's nodes. Windowed
    # /DAMP (tstart/tstop) still contributes its constant alpha here (no clock
    # in a linearized analysis — see the module docstring; flagged below).
    windowed = False
    for dp in getattr(model, "damps", []):
        g = model.node_groups.get(dp.grnod_id)
        if g is None or g.node_idx is None or g.node_idx.size == 0:
            continue
        # real (non-placeholder) nodes only, as engine/damping.py does
        idx = g.node_idx[model.mass[g.node_idx] < 1e29]
        if idx.size == 0:
            continue
        if dp.tstart > 0.0 or dp.tstop < 1e30:
            windowed = True
        alpha_dp = float(dp.alpha)
        for c in range(3):
            e = dof.eq[idx * DOFS_PER_NODE + c]
            act = e >= 0
            if np.any(act):
                rows.append(e[act])
                cols.append(e[act])
                vals.append(alpha_dp * model.mass[idx[act]])
        for c in range(3):
            e = dof.eq[idx * DOFS_PER_NODE + 3 + c]
            act = e >= 0
            if np.any(act):
                rows.append(e[act])
                cols.append(e[act])
                vals.append(alpha_dp * model.inertia[idx[act]])

    if windowed and log is not None:
        log.warning("/DAMP has a time window (tstart/tstop) — a "
                    "linearized/eigenvalue analysis has no clock, so the "
                    "constant alpha is used for the assembled C", "IMPL CEIGV")

    rows = np.concatenate(rows) if rows else np.zeros(0, dtype=np.int64)
    cols = np.concatenate(cols) if cols else np.zeros(0, dtype=np.int64)
    vals = np.concatenate(vals) if vals else np.zeros(0)
    C = sp.coo_matrix((vals, (rows, cols)),
                      shape=(dof.ndof, dof.ndof)).tocsr()
    if log is not None:
        log.info(f" DISCRETE DAMPING NNZ (ASSEMBLED) . . : {C.nnz}")
    return C


def assemble_damping(model, dof: DofMap, x_geom, alpha=0.0, beta=0.0,
                     K=None, M=None, log=None):
    """Assemble the FULL global damping C = alpha M + beta K + C_discrete
    (CSR, ndof x ndof) — the C analogue of ``assemble_mass`` (see the module
    docstring). ``alpha, beta`` are the Rayleigh coefficients (the M11
    /IMPL/DYNA/DAMP a, b); ``C_discrete`` is the spring-dashpot + /DAMP part
    (:func:`assemble_discrete_damping`).

    ``K`` and ``M`` (CSR, eq space) may be passed to avoid re-assembling them
    when the caller already has them (the complex eigensolver does); if omitted
    they are assembled here (K the tangent at ``x_geom``, M the consistent mass
    on ``model.x0``). Returns the CSR C in the FULL (unreduced) equation
    space — the caller reduces it T^T C T alongside K and M."""
    sp, _ = require_scipy()
    C = assemble_discrete_damping(model, dof, x_geom, log=log)
    if alpha != 0.0:
        if M is None:
            M = assemble_mass(model, dof,
                              model.x0 if model.x0.size else x_geom)
        C = C + alpha * M
    if beta != 0.0:
        if K is None:
            K = assemble(model, dof, x_geom)
        C = C + beta * K
    return C.tocsr()
