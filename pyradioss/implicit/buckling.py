"""
Linearized (eigenvalue) buckling analysis — M9; constraints & contact M14.

Fortran origin: ``engine/source/implicit/imp_buck.F`` — the /IMPL/BUCKL
branch of the OpenRadioss implicit solver, which extracts the lowest
eigenvalues of the tangent pencil built from the material stiffness and the
geometric (initial-stress) stiffness of a pre-stressed state. The port
exposes the same computation as a library function on an already-solved
implicit model (wired to the /IMPL/BUCKL card by ``statics._run_buckling``).

Theory
------
Take an equilibrium state with stress field sigma_0 produced by a reference
load P_0 (solved by the implicit static driver). For a proportional load
mu * P_0, the linearized-buckling assumption is that the pre-stress scales
with the load while the geometry change stays negligible, so the tangent
stiffness along the fundamental path is

    K_T(mu) ~= K_mat + mu * K_geo(sigma_0)

with K_mat the material + hourglass tangent (``assembly.assemble``) and
K_geo the initial-stress matrix assembled from sigma_0
(``assembly.assemble_kgeo``). Buckling = the fundamental path losing
uniqueness = K_T singular:

    (K_mat + mu * K_geo) phi = 0
        <=>  (-K_geo) phi = theta * K_mat phi,   mu = 1 / theta

a symmetric generalized eigenproblem with K_mat positive definite (a
well-posed constrained static model), solved here with the dense
``scipy.linalg.eigh``. The LARGEST positive theta gives the SMALLEST
positive load multiplier — the classical critical load P_cr = mu_1 * P_0
(Euler's column formula pi^2 EI/(kL)^2 is the closed-form special case the
M9 validation checks against). Negative theta correspond to buckling under
the REVERSED load and are reported separately by sign.

Constraints and contact (M14 — the M12 refusal removed)
--------------------------------------------------------
CONSTRAINTS: the pencil is stated on the REDUCED equations,

    ( T^T K_mat T  +  mu * T^T K_geo T ) phi_red = 0,

exactly what the ORIGINAL does — verified in imp_buck.F: BOTH assemblies
(the IKG=0 material pass and the IKG=1 pass whose difference forms KG)
run through ``UPD_GLOB_K``, the routine that applies the RBY/RBE2/RBE3/
tied-interface condensations, BEFORE the eigensolver (EIGBUCKP) sees them,
and the mode shapes are recovered by ``RECUKIN`` (the RBY_IMPR2-style
dependent-motion rebuild) — the port's ``T @ phi_red`` expansion. A
buckling mode therefore never violates a kinematic constraint: a rigid cap
buckles as a rigid cap.

CONTACT: the ORIGINAL assembles NO contact stiffness in imp_buck.F at all
(no IMP_INT_K call — NDDLI7 is forced 0): a body resting on a contact
support buckles, in the original, as if the support were absent. The port
DEVIATES, documented: the gap tangent of the CONVERGED prestressed active
set (``contact.contact_tangent`` at the prestressed configuration, with
the run's committed friction anchors) is added to K_MAT — the
mu-independent side of the pencil — because a closed stop is a physical
support whose stiffness does NOT scale with the load multiplier; putting
it in K_geo would stiffen the support proportionally to mu, which is
wrong for any unilateral stop that is already closed at the prestress
state, and omitting it (the original) makes a column resting on a stop
report the free-column factor. (For comparison, the one configuration-
dependent stiffness imp_buck.F DOES include — the IMP_KPRES pressure-load
stiffness — goes into its KG, correctly: a follower pressure DOES scale
with the load.) Two linearization caveats are inherent and documented
rather than hidden: the active set is FROZEN at the prestressed state
(the eigenproblem is blind to pairs that would open or close during the
buckle — the frozen-set tangent is bilateral, so a mode may lean ON a
stop it would physically separate from; that is the standard linearized-
buckling idealization of unilateral contact), and the friction state is
frozen with it (stick pairs keep their stick spring).

The dense solve is O(ndof^3): perfectly fine at this port's model sizes
(10^2-10^4 DOFs); a shift-invert sparse eigensolver is the documented
upgrade path for bigger models.
"""

from __future__ import annotations

import numpy as np

from . import require_scipy
from .assembly import assemble, assemble_kgeo
from .dofmap import DofMap


def buckling_factors(model, nev=4, log=None, constraints=None, contacts=None):
    """Lowest positive linearized-buckling load multipliers of the model's
    CURRENT (pre-stressed) state.

    Run an implicit static analysis first (it leaves the committed stress in
    the element buffers and the deformed geometry in ``model.x``); then this
    returns ``(factors, modes)``:

    * ``factors`` — up to ``nev`` positive multipliers mu, ascending: the
      structure buckles at mu times the CURRENT load;
    * ``modes``   — matching list of ``(du, dur)`` nodal mode shapes
      ((numnod, 3) each, fixed DOFs zero), normalized to unit equation-space
      length. With constraints the dependent nodes' motion is recovered
      through the transform (the RECUKIN analogue).

    ``constraints``/``contacts`` (M14): pass the DRIVER's live objects (the
    /IMPL/BUCKL card path does — the contact treatments then carry the
    run's committed friction anchors). ``None`` means build them from the
    model here, so the standalone library call handles constrained/contact
    models too; the explicit sentinel ``()`` disables them.

    The eigenproblem is stated at the current committed geometry with the
    current stress: for the classical "small preload" use (Euler column) that
    is indistinguishable from x0; after a large-displacement /IMPL/NONLIN run
    it is the tangent buckling estimate about the deformed state."""
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
    Km = assemble(model, dof, x_geom)
    if contacts:
        # M14: the converged active set's gap (+ friction) tangent joins
        # K_MAT — the mu-independent side (module docstring: the
        # documented deviation from imp_buck.F, which omits contact)
        from .contact import contact_tangent
        Kc = contact_tangent(contacts, x_geom, dof)
        # the friction slip blocks are nonsymmetric; eigh needs symmetry —
        # symmetrize (the skew part is the nonassociativity of Coulomb
        # slip; the symmetric part is the standard buckling treatment)
        Km = Km + 0.5 * (Kc + Kc.T)
    Kg = assemble_kgeo(model, dof, x_geom)
    if constraints is not None:
        # the reduced pencil T^T (.) T — what UPD_GLOB_K does to BOTH
        # matrices in imp_buck.F before EIGBUCKP
        constraints.build(dof, x_geom)
        Km = constraints.reduce_matrix(Km)
        Kg = constraints.reduce_matrix(Kg)
    Km = Km.toarray()
    Kg = Kg.toarray()

    # (-K_geo) phi = theta K_mat phi ; K_mat PD (scipy checks via Cholesky)
    theta, vecs = sla.eigh(-Kg, Km)
    order = np.argsort(-theta)                    # largest theta first
    factors = []
    modes = []
    for idx in order:
        th = theta[idx]
        if th <= 1e-12 or len(factors) >= nev:
            break
        factors.append(1.0 / th)
        vec = vecs[:, idx]
        if constraints is not None:
            vec = constraints.expand(vec)         # RECUKIN-style recovery
        du, dur = dof.scatter_solution(vec)
        modes.append((du, dur))
    if log is not None and factors:
        log.info(f" BUCKLING FACTORS (LOWEST) . . . . . . : "
                 f"{', '.join(f'{f:.6E}' for f in factors)}")
    return np.array(factors), modes
