"""
Modal (free-vibration eigenvalue) analysis — M16.

Fortran origin
--------------
There is no self-contained modal CARD in the open-source OpenRadioss engine:
``engine/source/input/freimpl.F`` reads only /IMPL/DYNA (imp_dyna.F, lumped
mass) and /IMPL/BUCKL (imp_buck.F, the geometric-stiffness eigenproblem), with
no /IMPL/EIGV branch. The natural-frequency extraction of the commercial
solver runs the SAME generalized-eigenproblem machinery imp_buck.F uses for
buckling (the EIGBUCKP / Lanczos family), only with the CONSISTENT MASS matrix
in place of the geometric stiffness. So — exactly as M9 ported the buckling
eigensolver as a library function before M11 added its thin card — this module
ports the consistent-mass modal eigensolver as a clean library capability
(reusing ``buckling.py``'s reduced-pencil + dense ``eigh`` path) and a minimal
/IMPL/EIGV engine card wires it (``statics._run_modal``).

Theory
------
Free vibration of an undamped structure, M ü + K u = 0, has solutions
u = phi e^{i omega t}; substituting gives the symmetric generalized
eigenproblem

    (K - omega^2 M) phi = 0

for the natural angular frequencies omega and mode shapes phi. K is the
tangent stiffness (``assembly.assemble`` — the same material+hourglass tangent
the statics Newton uses, so a plastified/geometrically-nonlinear state is
linearized about its committed configuration) and M is the CONSISTENT element
mass (``assembly.assemble_mass``, M16 — NOT the lumped diagonal the explicit
and implicit-dynamics paths use). The frequency in Hz is f = omega / (2 pi).

PRESTRESSED modes (``prestress=True``): the stiffness is the FULL tangent of a
loaded/spinning state, K = K_mat + K_geo (``assemble(..., kgeo=True)`` — the
imp_kgeo initial-stress term of /IMPL/NONLIN). A tensile prestress stiffens the
structure and raises its frequencies (a tightened guitar string, a spinning
blade); a compressive prestress softens it, and the fundamental frequency
drops to zero exactly at the buckling load (the modal and buckling
eigenproblems meet there — K_mat + mu_cr K_geo is singular). Run an implicit
static analysis first to leave the committed stress in the element buffers.

Constraints and contact (the M12/M14 reduced pencil)
----------------------------------------------------
Stated on the REDUCED equations, identically to buckling:

    ( T^T K T ) phi_red = omega^2 ( T^T M T ) phi_red

with T the constraint transform (``constraints.py`` — /RBODY, /RBE2, tied,
/RBE3, /MPC and chains). Because the inertia term is reduced the SAME way as
the stiffness, a rigid body automatically carries its EXACT condensed 6-DOF
mass (total mass, parallel-axis inertia, m*skew(r) COG coupling) at the
master — the T^T M T identity the M12 dynamics already relies on, now with the
CONSISTENT element mass feeding it. Mode shapes are recovered through T
(``constraints.expand`` — the RECUKIN analogue), so a constrained mode never
violates a kinematic tie. Contact (a body resting on a closed stop) adds the
converged active set's gap tangent to K, exactly as ``buckling.py`` documents.

Reporting
---------
``modal_frequencies`` returns ``(freqs_hz, modes, effective_mass)``:

* ``freqs_hz`` — the lowest ``nev`` natural frequencies (Hz), ascending;
* ``modes``    — MASS-NORMALIZED mode shapes (phi^T M phi = 1), each a
  ``(du, dur)`` pair of per-node (numnod, 3) translation/rotation fields;
* ``effective_mass`` — the (nev, 6) modal effective mass / participation for
  the six rigid-body directions (3 translations + 3 rotations about the
  origin): with mass-normalized modes the effective mass of mode i in
  direction d is (phi_i^T M r_d)^2, and the sum over ALL modes equals the
  structure's total mass / inertia (the classic completeness check — a few
  low modes usually capture most of it).

Deliberate deviations / deferrals (documented, not hidden)
----------------------------------------------------------
* DENSE eigh, not Lanczos/subspace. The original extracts a few eigenpairs of
  a large sparse pencil with a shift-invert Lanczos (EIGBUCKP); the port forms
  the reduced pencil densely and calls ``scipy.linalg.eigh`` — O(nred^3),
  perfectly fine at this port's model sizes (10^2-10^4 DOFs), the same choice
  ``buckling.py`` made and documented. A sparse shift-invert eigensolver
  (``scipy.sparse.linalg.eigsh``) is the upgrade path for larger models.
* Modal SUPERPOSITION transient, frequency response (/FREQ), complex/damped
  eigenvalues, random/spectral response and AMLS substructuring are DEFERRED
  (PORTING_GUIDE M16): this milestone delivers the real eigenpairs, on which
  those build.
"""

from __future__ import annotations

import numpy as np

from . import require_scipy
from .assembly import assemble, assemble_mass
from .dofmap import DofMap, DOFS_PER_NODE


def modal_frequencies(model, nev=6, log=None, constraints=None, contacts=None,
                      prestress=False):
    """Lowest ``nev`` natural frequencies (Hz) and mode shapes of the model's
    current state, from the consistent-mass generalized eigenproblem
    (K - omega^2 M) phi = 0 (see the module docstring).

    Returns ``(freqs_hz, modes, effective_mass)``. ``prestress=True`` adds the
    geometric (initial-stress) stiffness of the committed state to K (the
    stress-stiffened / spinning-structure spectrum). ``constraints`` /
    ``contacts`` follow ``buckling.buckling_factors``: pass the driver's live
    objects, ``None`` to build them from the model here, or the sentinel
    ``()`` to disable."""
    _, _ = require_scipy()             # fail early with the clear message
    import scipy.linalg as sla

    x_geom = model.x if model.x.size else model.x0
    if constraints is None or contacts is None:
        from ..common.messages import MessageLog
        silent = MessageLog()
        if constraints is None:
            from .constraints import build_constraints
            constraints = build_constraints(model, silent)
        if contacts is None:
            from .contact import build_implicit_contacts
            contacts = build_implicit_contacts(model, silent)

    dof = DofMap(model, log, constraints=constraints)
    # STIFFNESS: material (+hourglass) tangent, plus the geometric stiffness of
    # the committed stress state when a prestressed spectrum is requested
    K = assemble(model, dof, x_geom, kgeo=prestress)
    if contacts:
        # a closed stop is a physical support whose stiffness does NOT scale
        # with a load multiplier — add the converged gap tangent to K (the
        # same treatment, and the same documented imp_buck.F deviation, as
        # buckling.py); symmetrize the friction slip blocks for eigh
        from .contact import contact_tangent
        Kc = contact_tangent(contacts, x_geom, dof)
        K = K + 0.5 * (Kc + Kc.T)
    # MASS: the consistent element mass on the reference geometry (mass is
    # conserved; the corotational beam orients its local mass with x0's frame)
    M = assemble_mass(model, dof, model.x0 if model.x0.size else x_geom, log)

    # keep the FULL (unreduced) eq-space mass for the participation factors
    M_full = M
    if constraints is not None:
        constraints.build(dof, x_geom)
        K = constraints.reduce_matrix(K)              # T^T K T
        M = constraints.reduce_matrix(M)              # T^T M T (the exact
        #                                               rigid-body condensation)
    Kd = K.toarray()
    Md = M.toarray()

    # symmetric generalized eigensolve: (K - lambda M) phi = 0, lambda = omega^2
    # eigh returns eigenvalues ascending and M-orthonormal eigenvectors
    # (phi^T M phi = I) — the mass-normalized modes we report.
    lam, vecs = sla.eigh(Kd, Md)

    # RIGID-BODY / MECHANISM filter: a rigid or mechanism mode has zero strain
    # energy, so its lambda = phi^T K phi is numerically ZERO (round-off of the
    # backward-stable eigh). We must skip these — a free-free structure has up
    # to six — WITHOUT discarding genuine low structural modes. The scale is
    # set by the MEDIAN positive eigenvalue (robust both to the near-zero rigid
    # cluster AND to the huge penalty eigenvalues that the drilling / in-plane
    # penalties push to the top of the spectrum — a max-relative threshold, the
    # naive choice, would be inflated by those penalties and wrongly cull the
    # fundamental). lambda below 1e-8 of that median is treated as rigid.
    lam_pos = lam[lam > 0.0]
    lam_scale = float(np.median(lam_pos)) if lam_pos.size else 1.0
    rigid_tol = 1e-8 * lam_scale

    # rigid-body influence vectors r_d (d = 0..5) in FULL eq space, for the
    # participation / effective mass: translation d has 1 on every
    # translational eq of component d; rotation d has 1 on every rotational eq
    # of component d PLUS the moment arm of the translations about the origin
    # (r x e_d) — the standard 6-direction rigid basis.
    rigid = _rigid_body_vectors(model, dof)

    freqs, modes, eff = [], [], []
    for k in range(len(lam)):
        if len(freqs) >= nev:
            break
        lk = lam[k]
        if lk <= rigid_tol:
            # a (near) zero eigenvalue is a rigid-body / mechanism mode — not a
            # structural frequency; skip it (a free-free model has up to 6)
            continue
        omega = np.sqrt(lk)
        freqs.append(omega / (2.0 * np.pi))           # Hz
        phi_red = vecs[:, k]
        phi_full = (constraints.expand(phi_red)
                    if constraints is not None else phi_red)
        du, dur = dof.scatter_solution(phi_full)
        modes.append((du, dur))
        # effective mass in each rigid direction: (phi^T M_full r_d)^2 with
        # phi mass-normalized (phi^T M phi = 1)
        Mphi = M_full @ phi_full
        eff.append([(rigid[:, d] @ Mphi) ** 2 for d in range(6)])

    if log is not None and freqs:
        log.info(" NATURAL FREQUENCIES (HZ) . . . . . . : "
                 + ", ".join(f"{f:.6E}" for f in freqs))
    return np.array(freqs), modes, np.array(eff) if eff else np.zeros((0, 6))


def _rigid_body_vectors(model, dof: DofMap):
    """The 6 rigid-body influence vectors (ndof, 6) about the origin — the
    directions the modal effective mass is projected on. Columns 0-2 = unit
    translations, 3-5 = unit rotations (translation arm r x e_d + the unit
    rotation itself). Only DOFs with an equation are filled."""
    n = model.numnod
    x = model.x0 if model.x0.size else model.x
    R = np.zeros((dof.ndof, 6))
    node = np.arange(n)
    eq = dof.eq
    # translational rows
    tr_eq = np.stack([eq[node * DOFS_PER_NODE + c] for c in range(3)], axis=1)
    rot_eq = np.stack([eq[node * DOFS_PER_NODE + 3 + c] for c in range(3)],
                      axis=1)
    for c in range(3):
        act = tr_eq[:, c] >= 0
        R[tr_eq[act, c], c] = 1.0                      # unit translation
    # rotations about the origin: rotation d gives u = e_d x r on translations
    # and a unit on the rotational DOF d
    for d in range(3):
        e = np.zeros(3); e[d] = 1.0
        arm = np.cross(np.broadcast_to(e, (n, 3)), x)  # (n,3) = e_d x r
        for c in range(3):
            act = tr_eq[:, c] >= 0
            R[tr_eq[act, c], 3 + d] += arm[act, c]
        act = rot_eq[:, d] >= 0
        R[rot_eq[act, d], 3 + d] += 1.0
    return R
