"""
Implicit DYNAMIC analysis: Newmark-beta time integration with HHT-alpha
numerical dissipation (M10).

Fortran origin
--------------
``engine/source/implicit/imp_dyna.F`` — the implicit-dynamics companion of
the ``imp_solv.F`` driver, activated by /IMPL/DYNA (IDYNA in the reader,
``engine/source/input/freimpl.F``). Its routines map here as:

* ``DYNA_INI`` / ``DYNA_IN0``  — scheme setup: gamma/beta from the HHT alpha
  (``DY_G = HALF-D_AL``, ``DY_B = FOURTH*(ONE-D_AL)**2`` — /IMPL/DYNA/1) or
  read directly (``DY_G = NM_A``, ``DY_B = NM_B`` — /IMPL/DYNA/2), and the
  dynamic state arrays (DY_D/DY_V/DY_A) seeded from the model velocities
  → :func:`run_implicit_dynamic` startup.
* ``DYNA_INA``  — initial acceleration a_0 = M^-1 (f_ext(0) + f_int(0))
  → the ``a`` seed below.
* ``IMP_DYNAM`` — the effective-stiffness diagonal: the lumped mass /
  inertia scaled by 1/((1+alpha) * beta * dt^2) added to the tangent
  → ``K_eff`` below (the port scales K by (1+alpha) instead of dividing
  the mass term — the same linear system).
* ``IMP_DYNAR`` / ``IMP_FHHT``('1') — the dynamic residual: inertia force
  -M*a added to the static out-of-balance, with the previous time level's
  residual blended in by ``-HHT_A/(1+HHT_A)`` (the HHT weighting)
  → ``_dynamic_residual`` below.
* ``INTE_DYNA`` — the Newmark acceleration/velocity recovery from the
  converged displacement increment → ``_newmark_update``.
* ``DYNA_WEX`` — external-work bookkeeping → the energy history ledger.

M11 added (removing two M10 deferrals):

* ``/IMPL/DYNA/DAMP`` Rayleigh damping — ``IMP_DYKV`` (the damping force
  DY_DAM = DAMPA_IMP*M*v + DAMPB_IMP*K*v, built with the STEP-START
  tangent saved by ``IMP_DYKS``), the ``IDY_DAMP`` branch of ``IMP_DYNAM``
  (its exact contribution to the effective-stiffness diagonal and the
  S0-scaling of the off-diagonal tangent), the ``-DY_DAM`` add into FINT
  of ``IMP_DYNAR`` (so the damping force is HHT-weighted with the internal
  force), and the ``DY_EDAMP`` trapezoidal dissipation ledger of
  ``DYNA_WEX`` — see the "Rayleigh damping" section below;
* the automatic implicit time-step control of ``imp_dt.F`` (``IMP_DTN``,
  IDTC = 1): cut and RETRY on non-convergence, grow back toward
  /IMPL/DTINI on easy steps — shared with the statics driver
  (``statics.StepControl``, where the semantics are documented).

M12 added constraints and contact to the dynamic system:

* KINEMATIC CONSTRAINTS BY CONDENSATION (``constraints.py`` — the
  rby_imp0.F/rbe2_imp0.F/rbe3_imp0.F/i2_imp1.F transformations, called
  by IMP_DYKV/upd_rhs in the original's dynamic path): the whole
  eq-space system of a step — dynamic residual, K_eff, damping — is
  reduced through T before the solve. Because the inertia term -M a is
  built in NODE space and the Newmark kinematics make a_full = T a_red
  (v/a of a dependent node follow its masters exactly), the reduction
  T^T(-M a) IS the condensed-mass term -(T^T M T) a_red: a rigid body
  automatically carries its EXACT 6-DOF mass matrix at the master
  (total mass, parallel-axis inertia tensor, m*skew(r) COG coupling) —
  nothing else to assemble. The initial nodal velocities are projected
  onto the constraint manifold mass-weighted (for a rigid body that is
  precisely the explicit port's momentum projection v_g = p/M,
  w = J^-1 L), and the initial acceleration solves the CONDENSED
  M_red a_red = R_red (zero-mass reduced rows quasi-static, as before).
* PENALTY CONTACT (``contact.py`` — i7ke3.F): the contact force at the
  trial configuration joins f_int in the HHT weighting (it is a
  configuration force like the internal force; the previous level's
  value rides g_n), the active-set gap tangent joins K_eff, and the
  stored spring energy 1/2 K p^2 gets its own ``econt`` ledger channel
  (a state function — the balance closes through impacts up to the
  O(dt^2) trapezoid-vs-quadratic booking noise of the step the set
  changes in).

NOT mirrored (deferred explicitly, PORTING_GUIDE M10/M11): the ``QSTAT_*``
quasi-static-initialization branch, the IDTC = 2/3 displacement-norm /
Riks step controls of ``imp_dt.F``, and /IMPL/DT/FIXP fix points.

Rayleigh damping in the implicit system (M11, /IMPL/DYNA/DAMP)
--------------------------------------------------------------
The semi-discrete balance gains the classical Rayleigh damping matrix

    C = a M + b K,        M a + C v = f_ext + f_int         (a = DAMPA_IMP,
                                                             b = DAMPB_IMP)

with M the lumped mass and K the TANGENT AT THE STEP START (the original
saves the assembled K into DY_DIAK0/DY_LTK0 once per step — IMP_DYKS — and
IMP_DYKV multiplies THAT saved matrix by the current velocity; the port
reassembles the committed-state tangent per step under /IMPL/NONLIN and
reuses the constant K of the linear-geometry path). The damping force is
evaluated at the CURRENT velocity iterate v_{n+1} (the IMP_DYKV comment:
"using v(t+dt) is more stable — especially at beginning") and is folded
into the internal-force side of the HHT balance (IMP_DYNAR adds -DY_DAM to
FINT), so it is alpha-weighted between the time levels exactly like f_int:

    R = (1+a)[f_ext + f_int - C v]_{n+1} - a[f_ext + f_int - C v]_n
        - M a_{n+1}

Because Newmark makes v_{n+1} a kinematic function of the displacement
increment (dv/dDu = gamma/(beta dt)), the EXACT linearization gains

    K_eff = (1+alpha) K_T + M/(beta dt^2)
            + (1+alpha) * gamma/(beta dt) * C

— algebraically identical to IMP_DYNAM's IDY_DAMP branch, which divides
the whole system by (1+alpha) instead: its mass diagonal
BDT = 1/((1+a) b dt^2) + DAMPA*gamma/(b dt) and its off-diagonal scaling
S0 = DAMPB*gamma/(b dt) are this K_eff over (1+alpha), with C's K part
taken equal to the assembled K_T (exact under linear geometry; under
NONLIN K_T moves within the step while C keeps the step-start K — the
same approximation as the original, quadratic convergence degrades only
at strongly-rotating damped steps).

The DISSIPATION is booked exactly as DYNA_WEX books DY_EDAMP: per
converged step, trapezoidally over the step displacement increment,

    E_damp += 1/2 (C v_{n+1} + C v_n) . Du       (free DOFs only — the
                                                  original sums IKC == 0)

and enters the energy BALANCE on the energy side (the original adds it to
Eint rather than W_ext "which makes high error in case of high Rayleigh
damping" — the port keeps it as its own ``edamp`` history channel):
balance = IE + KE + E_damp - W_ext - E0. The M11 validation closes this
on the closed-form damped SDOF (mass-only, stiffness-only and mixed
Rayleigh cases against the exp(-zeta omega t) envelope and the damped
period).

A note on /IMPDISP + damping: the constraint-reaction work on a DRIVEN
DOF is booked from M a - f_ext - f_int, which under damping omits the
C v share of the reaction — the damping matrix lives in condensed
equation space, where driven rows do not exist (the original condenses
DY_DAM the same way). The ledger of a damped, displacement-driven run is
correspondingly approximate; free-vibration and force-driven ledgers are
exact and asserted.

The scheme (Newmark 1959; Hilber–Hughes–Taylor 1977)
----------------------------------------------------
Semi-discrete equation of motion with the port's sign convention (the
element kernels accumulate the internal force NEGATED into ``fint``, so
``fint`` is the elastic force ON the nodes and equilibrium is
``fext + fint = M a``):

    M a(t) = f_ext(t) + f_int(u(t))                  (no damping matrix)

Newmark approximates the new-step kinematics from the displacement
increment Du = u_{n+1} - u_n:

    a_{n+1} = Du/(beta dt^2) - v_n/(beta dt) - (1/(2 beta) - 1) a_n
    v_{n+1} = v_n + dt [ (1-gamma) a_n + gamma a_{n+1} ]

so ``a`` and ``v`` are pure KINEMATIC functions of Du — the only unknown
Newton iterates on is Du, exactly like a static load increment. The HHT
alpha-method states the balance at a weighted point between the time
levels (alpha in [-1/3, 0]; alpha = 0 is plain Newmark):

    M a_{n+1} = (1+alpha) [f_ext + f_int]_{n+1} - alpha [f_ext + f_int]_n

giving the dynamic residual and its exact linearization in Du:

    R(Du)  = (1+alpha)(f_ext,n+1 + f_int(u_n + Du)) - alpha g_n - M a_{n+1}
    K_eff  = (1+alpha) K_T + M/(beta dt^2),        K_eff dDu = R

with K_T the very same tangent statics assembles (material + hourglass
+ K_geo under /IMPL/NONLIN) and g_n = (f_ext + f_int)_n the CONVERGED
force of the previous step (stored, not recomputed). M is the LUMPED
(diagonal) starter mass — ``model.mass`` on translations, the nodal
``model.inertia`` on shell rotations — condensed through the same DofMap
as K; the original is lumped here too (MS/IN in IMP_DYNAM), so no
consistent-mass option is ported.

Stability and accuracy (Hughes, "The Finite Element Method", ch. 9):

* gamma = 1/2, beta = 1/4 (the defaults — the trapezoidal rule / average
  acceleration) is UNCONDITIONALLY stable and second-order accurate, with
  NO amplitude decay (spectral radius = 1 at every frequency: a linear
  free vibration conserves its discrete energy exactly) and a period
  ELONGATION of (omega dt)^2 / 12 + O(dt^4) — the O(dt^2) dispersion the
  M10 validation measures against the closed form.
* Newmark with gamma > 1/2 damps numerically but drops to FIRST order;
  the HHT form keeps second order while damping the unresolvable high
  modes: gamma = 1/2 - alpha, beta = (1-alpha)^2/4, high-frequency
  spectral radius rho_inf = (1+alpha)/(1-alpha) < 1 for alpha < 0.
  (The card takes alpha itself, like the original's HHT_A read — NOT a
  spectral-radius input; see engine_keywords.)
* unconditional stability holds for 2 beta >= gamma >= 1/2 — the implicit
  step is chosen by ACCURACY (resolving the modes that matter), not by
  the mesh's Courant limit: the validation runs stably at dt hundreds of
  times the explicit critical step, where the leapfrog would explode.

Reusing the statics machinery
-----------------------------
Each time step IS a statics increment plus inertia: the residual reuses
``statics._internal_forces`` unchanged (both geometry modes — the M8
frozen-frame small-strain path and the M9 /IMPL/NONLIN updated-Lagrangian
midpoint/end evaluation), the tangent reuses ``assembly.assemble`` (with
K_geo at the trial geometry under NONLIN), the /IMPDISP displacement drive
reuses the same condensation, and on convergence the element state commits
and (under NONLIN) the reference frame advances — the updated-Lagrangian
outer step, now marching real time.

Rate effects: seen truly or deferred — never du/1
-------------------------------------------------
The statics trick drives the rate-form kernels with the step displacement
increment as a pseudo-velocity at dt = 1. Under DYNAMICS there is a real
velocity, and any strain-RATE-dependent term fed that pseudo-velocity
would silently see the rate ``du/1`` instead of ``du/dt`` — wrong by the
time-step magnitude. M10 keeps the statics drive (it is what the element
tangents linearize, including the solid hourglass consistency) and
therefore DISABLES the rate devices explicitly rather than feeding them
garbage:

* the solid BULK VISCOSITY (qa/qb) is zeroed exactly like statics. It is
  the explicit shock-capture device; an implicit dynamic step is orders
  of magnitude above the shock-resolving scale, and the original's
  implicit branch runs without it.
* the LAW2 strain-rate hardening term (c > 0) is zeroed with a WARNING:
  rate-dependent plasticity under implicit dynamics is DEFERRED
  (PORTING_GUIDE M10), not half-implemented — the material runs
  rate-independent. (LAW1 has no rate term; the other laws have no
  implicit tangent to begin with.)

Prescribed motion: /IMPDISP is supported at the real time t (the drive
steps from d(t_n) to d(t_{n+1}); the driven DOFs' velocity/acceleration
follow from the same Newmark kinematics). /IMPVEL is REFUSED with a clear
error — integrating an imposed velocity into the displacement drive is a
one-liner the /IMPDISP card already provides, and silently ignoring it
(as statics may: a static /IMPVEL is meaningless) would drop a REAL
boundary condition of a dynamic run.

Energy ledger (DYNA_WEX analogue): per converged step the history records
kinetic energy (1/2 v M v + 1/2 w I w), internal + hourglass energy from
the element bookings, external work accumulated trapezoidally
(0.5 (f_ext,n + f_ext,n+1) . Du, plus the /IMPDISP constraint-reaction
work M a - f_ext - f_int on the driven DOFs), and the balance
IE + KE - W - E0. The trapezoidal rule conserves the discrete energy of a
linear system EXACTLY, so the balance closes to round-off there (asserted
by the M10 validations); HHT alpha < 0 drains it monotonically — that is
the numerical dissipation doing its documented job, visible in the ledger
rather than hidden.
"""

from __future__ import annotations

import numpy as np

from ..engine.kinematics import LoadsAndConstraints
from .assembly import assemble, _TANGENT_KERNELS
from .dofmap import DofMap, DOFS_PER_NODE
from .linsolve import LinearSolver
from .statics import (ImplicitResult, IncrementResult, _epsp_increments,
                      _internal_forces, _resolve_imposed, _snapshot,
                      _solver_banner)


class ImplicitDynResult(ImplicitResult):
    """Outcome of an implicit dynamic run (attached as
    ``model.implicit_result``): the per-step convergence records of
    :class:`ImplicitResult` (``increments[k].load_factor`` holds the step's
    END TIME) plus the scheme actually run and the energy/motion history."""

    def __init__(self, alpha, gamma, beta, alpha_m=0.0, alpha_f=0.0):
        super().__init__()
        self.alpha = alpha
        self.alpha_m = alpha_m
        self.alpha_f = alpha_f
        self.gamma = gamma
        self.beta = beta
        #: per-converged-step history: "t", kinetic "ke", internal+hourglass
        #: "ie", external work "wext", Rayleigh dissipation "edamp" (M11 —
        #: the DY_EDAMP ledger, zero without /IMPL/DYNA/DAMP), numerical
        #: dissipation "enum" (Chung & Hulbert 1993 Generalized-alpha / HHT /
        #: Wood-Bossak), balance "bal" = ie + ke + edamp + econt + efric - wext - e0,
        #: and "u" — displacement snapshots (numnod, 3) for the validations (kept
        #: while numnod stays example-sized).
        #: "econt" (M12): stored contact penalty-spring energy 1/2 K p^2
        #: (+ M13 the stick spring's 1/2 |f_t|^2/K_t), zero without /INTER,
        #: part of the balance like edamp.
        #: "efric" (M13): frictional SLIP dissipation mu f_n dgamma from the
        #: TYPE7 return mapping, accumulated per committed step — the
        #: contact analogue of plastic work (zero without friction).
        self.history = {"t": [], "ke": [], "ie": [], "wext": [], "edamp": [],
                        "enum": [], "econt": [], "efric": [], "bal": [], "u": []}


#: stop keeping displacement snapshots beyond this many stored floats —
#: the history is a validation/postprocessing device, not a core output
#: (energy scalars are always kept).
_U_HISTORY_CAP = 5_000_000


def _lumped_mass_eq(model, dof):
    """Diagonal mass in EQUATION space: ``model.mass`` on the translational
    equations, ``model.inertia`` on the (shell) rotational equations —
    exactly the lumped starter arrays the explicit leapfrog divides by
    (IMP_DYNAM uses the same MS/IN), condensed through the DofMap.

    A rotational equation with zero nodal inertia keeps M = 0: its balance
    is quasi-static (no Ma term) and K_eff still carries the full rotational
    stiffness, so the system stays well posed.

    Frozen-placeholder masses (1e30 — the Starter's marker for nodes no
    element references) read as ZERO here: before M12 such nodes never had
    equations; now an unfrozen constraint master does, and its phantom
    placeholder must not enter the physics — the body's real mass reaches
    the master through the condensation T^T M T (see constraints.py)."""
    n = model.numnod
    node = np.arange(n)
    mass = np.where(model.mass >= 1e29, 0.0, model.mass)
    M = np.zeros(dof.ndof)
    for c in range(3):
        e = dof.eq[node * DOFS_PER_NODE + c]
        act = e >= 0
        M[e[act]] = mass[act]
        e = dof.eq[node * DOFS_PER_NODE + 3 + c]
        act = e >= 0
        M[e[act]] = model.inertia[act]
    return M


def _disable_rate_devices(model, log, nlg=False):
    """Statics/dynamics share the pseudo-velocity kernel drive, so the
    rate devices must not run (module docstring): zero the solid bulk
    viscosity like statics, and zero the LAW2 rate coefficient with an
    explicit DEFERRED warning (never silently du/1)."""
    nvisc = 0
    for name, group in model.element_groups():
        # shell chvis3 quadratic viscous hourglass damper is a rate device
        # too — disable it in the pseudo-velocity residual exactly like the
        # bulk viscosity (see shell_bt4.forces() _impl_static_hg gate).
        group.state["_impl_static_hg"] = True
        for sl, mat, prop in group.state["slices"]:
            if prop.params.get("qa", 0.0) or prop.params.get("qb", 0.0):
                prop.params["qa"] = 0.0
                prop.params["qb"] = 0.0
                nvisc += 1
    if nvisc:
        log.info(" BULK VISCOSITY (qa/qb) . . . . . . . : DISABLED "
                 "(IMPLICIT)")
    warned = set()
    for name, group in model.element_groups():
        for sl, mat, prop in group.state["slices"]:
            if mat.law == 2 and mat.params.get("c", 0.0) and \
                    id(mat) not in warned:
                warned.add(id(mat))
                mat.params["c"] = 0.0
                log.warning(
                    f"/MAT/LAW2/{mat.id}: strain-rate hardening (c > 0) is "
                    f"DEFERRED under implicit dynamics — the term is "
                    f"disabled and the material runs rate-independent "
                    f"(see PORTING_GUIDE M10)", "IMPL/DYNA")
    from .statics import (_check_total_form_geometry, _law36_static_curve,
                          _warn_spring_dashpot)
    _warn_spring_dashpot(model, log)
    # M13: LAW36 multi-rate curve families run on the static curve only
    # (the pseudo-velocity drive would feed the family a step-size rate)
    _law36_static_curve(model, log)
    # M14: LAW42 requires /IMPL/NONLIN (the total-form stress never sees
    # the trial displacement on the frozen frame — see statics)
    _check_total_form_geometry(model, nlg)


# ----------------------------------------------------------------------------
# The driver
# ----------------------------------------------------------------------------

def run_implicit_dynamic(model, controls, log, out_dir=None, run_name="RUN",
                         run_num=1, alpha_m=None, alpha_f=None, ikt=None,
                         t_fix=None):
    """Run an implicit dynamic (Newmark/HHT/Generalized-alpha) analysis on an initialized
    model. ``controls.impl_dyna`` selects the scheme (1 = HHT from alpha,
    2 = Newmark gamma/beta, 3 = Generalized-alpha); /RUN's final time and /IMPL/DTINI are PHYSICAL.
    Returns the model with ``model.implicit_result`` an
    :class:`ImplicitDynResult`."""
    ip = controls
    n = model.numnod
    nlg = bool(getattr(ip, "impl_nlgeom", False))
    if getattr(ip, "impl_arc", False):
        raise ValueError(
            "/IMPL/ARCL cannot be combined with /IMPL/DYNA: arc length "
            "traces a STATIC equilibrium path (the load factor is the "
            "unknown), while dynamics marches real time. Drop one of the "
            "two cards.")

    # ---- tangent update policy (IKT: 1=KTANG, 2=KTFUL, 4=KTCON) -----------
    if ikt is None:
        ikt = int(getattr(ip, "impl_ikt", getattr(ip, "impl_nonl_ikt", 1)))
    if ikt not in (1, 2, 3, 4):
        ikt = 1

    # ---- time step fixpoints (/IMPL/DT/FIXP, imp_dt.F IMP_DTF) -------------
    if t_fix is None:
        t_fix = getattr(ip, "impl_dt_fixp", None)
    t_fix_list = sorted([float(tf) for tf in t_fix if float(tf) > 0.0]) if t_fix else []

    # ---- scheme constants (DYNA_INI / Chung & Hulbert 1993) ---------------
    if alpha_m is not None or alpha_f is not None:
        am = float(alpha_m) if alpha_m is not None else 0.0
        af = float(alpha_f) if alpha_f is not None else 0.0
        gamma = 0.5 - am + af
        beta = 0.25 * (1.0 - am + af) ** 2
        alpha = -af
    elif getattr(ip, "impl_dyna", 0) == 3:
        am = float(getattr(ip, "impl_dyna_alpha_m", 0.0))
        af = float(getattr(ip, "impl_dyna_alpha_f", 0.0))
        gamma = 0.5 - am + af
        beta = 0.25 * (1.0 - am + af) ** 2
        alpha = -af
    elif ip.impl_dyna == 1:
        alpha = float(ip.impl_dyna_alpha)
        af = -alpha if alpha <= 0.0 else alpha
        am = float(getattr(ip, "impl_dyna_alpha_m", 0.0))
        gamma = 0.5 - am + af
        beta = 0.25 * (1.0 - am + af) ** 2
    else:
        alpha = 0.0                             # plain Newmark (D_AL = ZERO)
        am = 0.0
        af = 0.0
        gamma = float(ip.impl_dyna_gamma)       # DY_G = NM_A
        beta = float(ip.impl_dyna_beta)         # DY_B = NM_B
    if beta <= 0.0:
        raise ValueError(
            f"/IMPL/DYNA: beta = {beta:g} — the implicit Newmark family "
            f"needs beta > 0 (beta = 0 is the explicit central-difference "
            f"member: run the explicit solver instead).")

    log.info("\n     IMPLICIT DYNAMIC ANALYSIS (M10)")
    log.info("     -------------------------------")

    unsupported = [name for name, _ in model.element_groups()
                   if name not in _TANGENT_KERNELS]
    if unsupported:
        raise NotImplementedError(
            f"the implicit solver has no tangent for element group(s) "
            f"{unsupported} (supported: {list(_TANGENT_KERNELS)}). Remove "
            f"them or run the explicit solver.")
    if model.impvel:
        raise NotImplementedError(
            "/IMPVEL under /IMPL/DYNA is not ported — impose the motion "
            "with /IMPDISP (the displacement drive; integrate the velocity "
            "curve into a displacement curve). Silently ignoring a REAL "
            "dynamic boundary condition would be worse than refusing.")
    if getattr(ip, "impl_buckl", 0):
        raise ValueError(
            "/IMPL/BUCKL cannot be combined with /IMPL/DYNA: linearized "
            "buckling is an eigensolve about a STATIC prestressed state "
            "(run the prestress increments with /IMPL and the BUCKL card, "
            "without DYNA).")

    # ---- Rayleigh damping (M11, /IMPL/DYNA/DAMP — IDY_DAMP) ---------------
    damp_on = bool(getattr(ip, "impl_dyna_damp", False))
    da = float(getattr(ip, "impl_dyna_dampa", 0.0)) if damp_on else 0.0
    db = float(getattr(ip, "impl_dyna_dampb", 0.0)) if damp_on else 0.0

    if am > 0.0 and af > 0.0:
        scheme = f"GENERALIZED-ALPHA (alpha_m = {am:g}, alpha_f = {af:g})"
    elif am > 0.0 and af == 0.0:
        scheme = f"WOOD-BOSSAK (alpha_m = {am:g})"
    elif af > 0.0 and am == 0.0:
        scheme = f"HHT-ALPHA (alpha = {-af:g}, alpha_f = {af:g})"
    elif ip.impl_dyna == 1:
        scheme = f"HHT-ALPHA (alpha = {alpha:g})"
    else:
        scheme = "NEWMARK"
    log.info(f" TIME INTEGRATION . . . . . . . . . . : {scheme}")
    ikt_labels = {1: "KTANG (EVERY NEWTON ITERATION)", 2: "KTFUL (START OF EACH TIME STEP)", 4: "KTCON (CONSTANT STIFFNESS)"}
    log.info(f" TANGENT POLICY (IKT) . . . . . . . . : {ikt_labels.get(ikt, f'IKT={ikt}')}")
    if t_fix_list:
        log.info(f" FIXPOINTS (/IMPL/DT/FIXP) . . . . . : {len(t_fix_list)} fixpoints")
    if damp_on:
        log.info(f" RAYLEIGH DAMPING C = a M + b K . . . : a = {da:g}, "
                 f"b = {db:g}  (/IMPL/DYNA/DAMP)")
    log.info(f" NEWMARK GAMMA / BETA . . . . . . . . : {gamma:g} / {beta:g}"
             + ("  (TRAPEZOIDAL RULE)"
                if gamma == 0.5 and beta == 0.25 else ""))
    log.info(f" LINEAR SOLVER  . . . . . . . . . . . : "
             f"{_solver_banner(ip, log)}")
    log.info(f" GEOMETRY . . . . . . . . . . . . . . : "
             f"{'NONLINEAR (UPDATED-LAGRANGIAN + KGEO)' if nlg else 'LINEAR (SMALL STRAIN)'}")

    _disable_rate_devices(model, log, nlg)

    if model.rwalls:
        raise NotImplementedError(
            "/RWALL is not supported by the implicit solver — replace the "
            "wall with /INTER/TYPE7 contact against a meshed (fixed) "
            "surface, or run the explicit solver.")

    imposed, presc = _resolve_imposed(model, log)
    # M12: kinematic constraints (condensation) + penalty contact
    from .constraints import build_constraints
    from .contact import build_implicit_contacts, contact_forces
    constr = build_constraints(model, log)
    contacts = build_implicit_contacts(model, log)
    if constr is not None:
        constr.veto_imposed(imposed)
    dof = DofMap(model, log, prescribed=presc, constraints=constr)
    loads = LoadsAndConstraints(model, log)
    solver = LinearSolver(ip.impl_linsolve, log)
    M_eq = _lumped_mass_eq(model, dof)

    committed = {name: _snapshot(g) for name, g in model.element_groups()}
    x_ref = model.x0.copy()
    model.x = model.x0.copy()
    if constr is not None:
        constr.build(dof, x_ref)
        log.info(f" CONDENSED EQUATIONS (M12)  . . . . . : {constr.nred} "
                 f"(FROM {dof.ndof})")

    # ---- time controls: /RUN t_end is PHYSICAL time, /IMPL/DTINI the step
    t_end = ip.t_end
    if t_end <= 0.0:
        raise ValueError(
            f"implicit dynamic run has a non-positive final time "
            f"({t_end:g}); set /RUN <RunName> <T_stop> with a positive "
            f"value (physical time under /IMPL/DYNA).")
    dt = ip.impl_dt if ip.impl_dt > 0 else t_end
    if ip.impl_dt <= 0.0:
        log.warning("/IMPL/DYNA without /IMPL/DTINI: the whole run is ONE "
                    "time step — set /IMPL/DTINI to resolve the motion",
                    "IMPL/DYNA")
    log.info(f" FINAL TIME (/RUN)  . . . . . . . . . : {t_end:12.5E}")
    log.info(f" TIME STEP (/IMPL/DTINI)  . . . . . . : {dt:12.5E}")
    log.info(f" NEWTON TOLERANCE (RESIDUAL). . . . . : {ip.impl_tol:12.5E}")
    log.info(f" MAX NEWTON ITERATIONS  . . . . . . . : {ip.impl_max_iter}")

    result = ImplicitDynResult(alpha, gamma, beta, alpha_m=am, alpha_f=af)
    # listing cadence: every step for short runs, /PRINT's cycle count for
    # long ones (the explicit-listing convention)
    nsteps_est = int(np.ceil(t_end / dt))
    log_every = 1 if nsteps_est <= 200 else max(1, ip.print_cycles)

    # ---- dynamic state (DYNA_INI / DYNA_INA) -------------------------------
    # velocities seeded from the starter (/INIVEL); the initial acceleration
    # balances the INITIAL out-of-force: M a_0 = f_ext(0) + f_int(0)
    # (f_int(0) is whatever the committed state carries — zero on a virgin
    # model). Zero-mass equations (massless shell rotations) get a = 0.
    v = model.v.copy()
    vr = model.vr.copy()
    # a /BCS-fixed DOF carries no motion: drop any stray /INIVEL component
    # on it (the explicit path zeroes it in the kinematic enforcement)
    v[dof.fix_tra] = 0.0
    vr[dof.fix_rot] = 0.0
    if constr is not None:
        # project the initial velocities onto the constraint manifold,
        # mass weighted — for a rigid body this IS the explicit port's
        # momentum projection (v_g = p/M, w = J^-1 L); an /INIVEL field
        # can only be carried through its constraint-compatible part
        v, vr = constr.project_velocity(dof, M_eq, v, vr, solver)

    # ---- damping matrix C = a M + b K (IMP_DYKS/IMP_DYKV) ------------------
    # b couples through the STEP-START tangent: constant under linear
    # geometry (assembled once here), reassembled at each committed frame
    # under /IMPL/NONLIN (the per-step save of the original). K_damp is the
    # material(+hourglass) tangent of the committed state — kgeo is a
    # stress-stiffening term, not a structural stiffness for damping
    # purposes, and at t = 0 the state is unstressed anyway.
    K_damp = None
    if damp_on and db != 0.0:
        K_damp = assemble(model, dof, x_ref, None, kgeo=False)

    def _damp_force(v_nod, vr_nod):
        """Equation-space Rayleigh damping force C v (IMP_DYKV): zero
        without the card."""
        if not damp_on:
            return None
        v_eq = dof.gather_residual(v_nod, vr_nod)
        fd = da * M_eq * v_eq
        if K_damp is not None:
            fd = fd + db * (K_damp @ v_eq)
        return fd

    fext = np.zeros((n, 3))
    loads.external_forces(0.0, fext, x_ref)
    fint, mint = _internal_forces(model, x_ref, np.zeros((n, 3)),
                                  np.zeros((n, 3)), committed, nlg)
    if contacts:
        # contact force of the INITIAL configuration joins f_int (a
        # configuration force — it rides the HHT previous level too)
        fc0, _ = contact_forces(contacts, model.x, n)
        fint = fint + fc0
    R0 = dof.gather_residual(fext + fint, mint)
    # DYNA_INA: the initial acceleration balances the initial out-of-force,
    # damping included: M a_0 = f_ext(0) + f_int(0) - C v_0
    fd_prev = _damp_force(v, vr)
    if fd_prev is not None:
        R0 = R0 - fd_prev
    if constr is not None:
        # condensed initial acceleration: M_red a_red = R_red (the rigid
        # 6-DOF mass at the master — see the module docstring), zero-mass
        # reduced rows quasi-static like the diagonal path below
        from . import require_scipy
        sp, _spla = require_scipy()
        from .constraints import _solve_semidefinite
        Mred = (constr.Tt @ sp.diags(M_eq) @ constr.T).tocsr()
        a0_eq = constr.expand(
            _solve_semidefinite(Mred, constr.reduce_vector(R0), solver))
    else:
        a0_eq = np.where(M_eq > 0.0, R0 / np.where(M_eq > 0.0, M_eq, 1.0),
                         0.0)
    a, ar = dof.scatter_solution(a0_eq)
    # previous-level force g_n = (f_ext + f_int)_n for the HHT weighting
    g_prev_f = fext + fint
    g_prev_m = mint.copy()
    fext_prev = fext.copy()

    # ---- energy ledger seed -------------------------------------------------
    real = model.mass < 1e29
    econt0 = sum(c.energy(model.x) for c in contacts) if contacts else 0.0
    e0 = _kinetic(model, v, vr, real) + _elem_energy(model) + econt0
    wext = 0.0
    edamp = 0.0
    efric = 0.0
    enum = 0.0
    ke_prev = _kinetic(model, v, vr, real)
    ie_prev = _elem_energy(model)
    econt_prev = econt0
    keep_u = True

    log.info("\n        STEP        TIME    ITER   RESIDUAL-NORM   STATUS")

    # automatic time-step control (M11, imp_dt.F): cut-and-retry on a failed
    # step, grow back toward /IMPL/DTINI on easy ones
    from .statics import StepControl
    ctrl = StepControl(ip, dt, log, what="TIME STEP")

    t = 0.0
    step_no = 0
    cached_K = None
    cached_K_eff = None
    cached_dt = None
    saved_dt_ctrl = None

    while t < t_end * (1.0 - 1e-12):
        dt_target = ctrl.dt
        hit_fixpoint = False
        dt_s = min(dt_target, t_end - t)

        # /IMPL/DT/FIXP: time step fixpoints array t_fix. When advancing time t,
        # adjust dt so that the solver hits every fixpoint exactly without overshooting.
        if t_fix_list:
            for tf in t_fix_list:
                if tf > t + 1e-12 * dt_target:
                    if t + dt_s >= tf - 1e-12 * dt_target:
                        dt_s = tf - t
                        hit_fixpoint = True
                        if saved_dt_ctrl is None:
                            saved_dt_ctrl = ctrl.dt
                    break

        dt_s = min(dt_s, t_end - t)
        if dt_s <= 1e-15:
            break
        t_new = t + dt_s
        step_no += 1

        (inc, u, ur_s, fint, mint, fext_new, fd_new,
         cached_K, cached_K_eff, cached_dt) = _solve_step(
            model, ip, dof, loads, solver, committed, x_ref, imposed,
            t, t_new, dt_s, alpha, gamma, beta, M_eq, v, vr, a, ar,
            g_prev_f, g_prev_m, nlg,
            (da, db, K_damp, fd_prev) if damp_on else None,
            constr, contacts,
            alpha_m=am, alpha_f=af, ikt=ikt,
            cached_K=cached_K, cached_K_eff=cached_K_eff, cached_dt=cached_dt)
        result.increments.append(inc)
        if not inc.converged:
            log.info(f" {step_no:11d} {t_new:11.4E} {inc.iterations:6d} "
                     f"{inc.residuals[-1]:14.5E}   *** NO CONVERGENCE")
            # roll back (the failed step touched neither model.x nor the
            # committed buffers' base — IMP_DTN's TT/NCYCLE rollback) and
            # retry at the cut step
            if hit_fixpoint and saved_dt_ctrl is not None:
                ctrl.dt = saved_dt_ctrl
                saved_dt_ctrl = None
            if ctrl.cut():
                step_no -= 1
                cached_K_eff = None
                continue
            result.converged = False
            result.stop_reason = (
                f"NEWTON DID NOT CONVERGE AT TIME {t_new:.4E} IN "
                f"{ip.impl_max_iter} ITERATIONS "
                f"(||R|| = {inc.residuals[-1]:.4E}) EVEN AT THE MINIMUM "
                f"TIME STEP {ctrl.dt_min:.3E} "
                f"({ctrl.total_cuts} automatic cuts — imp_dt.F control)")
            for gname, group in model.element_groups():
                if gname in committed:
                    _restore(group, committed[gname])
            break

        # ---- commit (the statics commit + the Newmark kinematics) ---------
        # INTE_DYNA: recover a_{n+1}, v_{n+1} from the converged increment
        a_new = (u / (beta * dt_s * dt_s) - v / (beta * dt_s)
                 - (0.5 / beta - 1.0) * a)
        v_new = v + dt_s * ((1.0 - gamma) * a + gamma * a_new)
        ar_new = (ur_s / (beta * dt_s * dt_s) - vr / (beta * dt_s)
                  - (0.5 / beta - 1.0) * ar)
        vr_new = vr + dt_s * ((1.0 - gamma) * ar + gamma * ar_new)

        # ---- external-work booking (DYNA_WEX): trapezoidal f.du over the
        # applied loads, plus the /IMPDISP constraint-reaction work on the
        # driven DOFs (reaction = M a - f_ext - f_int, trapezoid too)
        wext_step = 0.5 * float(((fext_prev + fext_new)[real] * u[real]).sum())
        if imposed:
            # constraint reaction on a driven DOF: what the drive must
            # supply beyond the loads, r = M a - (f_ext + f_int)
            g_new_r = model.mass[:, None] * a_new - (fext_new + fint)
            g_old_r = model.mass[:, None] * a - g_prev_f
            iner = getattr(model, "inertia", None)
            iner_val = iner[:, None] if iner is not None else np.zeros_like(ar_new)
            g_new_rot = iner_val * ar_new - mint
            g_old_rot = iner_val * ar - (g_prev_m if g_prev_m is not None else np.zeros_like(mint))
            for idx, d, fct, scale in imposed:
                if d < 3:
                    wext_step += 0.5 * float(((g_new_r + g_old_r)[idx, d]
                                         * u[idx, d]).sum())
                else:
                    wext_step += 0.5 * float(((g_new_rot + g_old_rot)[idx, d - 3]
                                         * ur_s[idx, d - 3]).sum())
        wext += wext_step

        # ---- Rayleigh dissipation booking (DYNA_WEX's DY_EDAMP): the
        # trapezoid of the damping force over the step displacement
        # increment, free DOFs only (both live in equation space)
        edamp_step = 0.0
        if damp_on:
            u_eq = dof.gather_residual(u, ur_s)
            fd0 = fd_prev if fd_prev is not None else 0.0
            edamp_step = 0.5 * float(u_eq @ (fd_new + fd0))
            edamp += edamp_step
            fd_prev = fd_new

        v, a, vr, ar = v_new, a_new, vr_new, ar_new
        g_prev_f = fext_new + fint
        g_prev_m = mint.copy()
        fext_prev = fext_new
        committed = {name: _snapshot(g) for name, g in model.element_groups()}
        efric_step = 0.0
        if contacts:
            # M13: re-base the friction anchors on the converged step and
            # book the return map's slip work into its own ledger channel
            from .contact import commit_contacts
            efric_step = commit_contacts(contacts, model.x)
            efric += efric_step
        if nlg:
            if constr is not None:
                # exact placement of the dependent nodes before the frame
                # advances (rigid Rodrigues re-placement, tied co-rotated
                # offset — see constraints.commit_placement)
                constr.commit_placement(model, u, ur_s)
            x_ref = model.x.copy()
            if constr is not None:
                constr.build(dof, x_ref)   # re-linearize arms/fits
            if damp_on and db != 0.0:
                # re-save the damping stiffness at the new committed frame
                # (the original's per-step IMP_DYKS save)
                K_damp = assemble(model, dof, x_ref, None, kgeo=False)

        ke = _kinetic(model, v, vr, real)
        ie = _elem_energy(model)
        # stored contact-spring energy (M12): a state function of the
        # configuration — its own ledger channel, like edamp
        econt = (sum(c.energy(model.x) for c in contacts)
                 if contacts else 0.0)

        # Accurate energy dissipation tracking:
        # In exact conservation, wext_step = d_ke + d_ie + edamp_step + d_econt + efric_step.
        # Numerical dissipation d_enum is the deficit absorbed by the integrator.
        d_enum = wext_step - ((ke - ke_prev) + (ie - ie_prev) + edamp_step + (econt - econt_prev) + efric_step)
        enum += d_enum

        ke_prev = ke
        ie_prev = ie
        econt_prev = econt

        t = t_new
        if hit_fixpoint and saved_dt_ctrl is not None:
            ctrl.dt = saved_dt_ctrl
            saved_dt_ctrl = None
        else:
            ctrl.converged(inc.iterations)

        # ---- history --------------------------------------------------------
        h = result.history
        h["t"].append(t)
        h["ke"].append(ke)
        h["ie"].append(ie)
        h["wext"].append(wext)
        h["edamp"].append(edamp)
        h["enum"].append(enum)
        h["econt"].append(econt)
        h["efric"].append(efric)
        h["bal"].append(ie + ke + edamp + econt + efric - wext - e0)
        if keep_u:
            h["u"].append(model.x - model.x0)
            if (len(h["u"]) + 1) * n * 3 > _U_HISTORY_CAP:
                keep_u = False
                h["u_truncated"] = True

        if step_no % log_every == 0 or t >= t_end * (1.0 - 1e-12):
            log.info(f" {step_no:11d} {t_new:11.4E} {inc.iterations:6d} "
                     f"{inc.residuals[-1]:14.5E}   converged")

    # expose the final velocities like the explicit engine (output/restart
    # conventions read model.v / model.vr)
    model.v = v
    model.vr = vr
    model.implicit_result = result
    _dyn_summary(model, result, dof, log, e0)
    return model


def _kinetic(model, v, vr, real):
    """1/2 v M v + 1/2 w I w with the lumped starter mass/inertia — the
    explicit engine's kinetic-energy expression."""
    ke = 0.5 * float((model.mass[real, None] * v[real] ** 2).sum())
    ke += 0.5 * float((model.inertia[:, None] * vr ** 2).sum())
    return ke


def _elem_energy(model):
    """Internal + hourglass energy from the element bookings."""
    e = 0.0
    for _, group in model.element_groups():
        e += float(group.state["eint"].sum())
        e += float(group.state["ehour"].sum())
    return e


# ----------------------------------------------------------------------------
# One time step: Newton on the HHT/Newmark dynamic residual
# ----------------------------------------------------------------------------

def _solve_step(model, ip, dof, loads, solver, committed, x_ref, imposed,
                t_old, t_new, dt, alpha, gamma, beta, M_eq, v, vr, a, ar,
                g_prev_f, g_prev_m, nlgeom, damp=None, constr=None,
                contacts=(), alpha_m=0.0, alpha_f=0.0, ikt=1,
                cached_K=None, cached_K_eff=None, cached_dt=None):
    """Newton-iterate one time step to the Generalized-alpha / HHT-weighted dynamic balance
    (see module docstring). Returns ``(inc, u, ur, fint, mint, fext_new,
    fd_new, cached_K, cached_K_eff, cached_dt)`` with ``u``/``ur`` the converged step displacement increment,
    ``fint``/``mint`` the internal force at the converged state (kept by
    the caller as the next step's intermediate previous-level force) and ``fd_new``
    the equation-space damping force C v_{n+1} at the converged state
    (None without damping)."""
    n = model.numnod
    c0 = 1.0 / (beta * dt * dt)             # kinematic acceleration factor
    c_K = 1.0 - alpha_f
    c_M = (1.0 - alpha_m) / (beta * dt * dt)
    c_C = (1.0 - alpha_f) * gamma / (beta * dt)

    from . import require_scipy
    sp, _ = require_scipy()
    M_diag = sp.diags(c_M * M_eq, format="csr")   # the lumped-mass add to K

    # ---- Rayleigh damping pieces (IMP_DYKV / IMP_DYNAM's IDY_DAMP) --------
    da = db = 0.0
    K_damp = fd_prev = None
    C_eff = None
    if damp is not None:
        da, db, K_damp, fd_prev = damp
        C_eff = sp.diags(c_C * da * M_eq, format="csr")
        if K_damp is not None:
            C_eff = C_eff + (c_C * db) * K_damp

    def _fd(v_nod, vr_nod):
        """Equation-space damping force C v at a velocity state."""
        v_eq = dof.gather_residual(v_nod, vr_nod)
        fd = da * M_eq * v_eq
        if K_damp is not None:
            fd = fd + db * (K_damp @ v_eq)
        return fd

    # external force at the END time level
    fext = np.zeros((n, 3))
    loads.external_forces(t_new, fext, x_ref)
    fext_eq = dof.gather_residual(fext, np.zeros((n, 3)))
    ref = max(float(np.linalg.norm(fext_eq)), 1e-30)

    from .followerload import has_follower, pload_tangent
    follower = nlgeom and has_follower(loads)
    lstiff = follower and bool(getattr(ip, "impl_load_stiff", True))

    def _fext_trial(u_trial):
        if not follower:
            return fext
        fx = np.zeros((n, 3))
        loads.external_forces(t_new, fx, model.x + u_trial)
        return fx

    epsp0 = {name: committed[name].get("epsp")
             for name, _ in model.element_groups()}

    # ---- predictor: constant-acceleration extrapolation (a_{n+1} = a_n) ----
    u = dt * v + 0.5 * dt * dt * a
    ur_s = dt * vr + 0.5 * dt * dt * ar
    u[dof.fix_tra] = 0.0
    ur_s[dof.fix_rot] = 0.0
    if constr is not None:
        u, ur_s = constr.make_consistent(dof, u, ur_s)
        u[dof.fix_tra] = 0.0
        ur_s[dof.fix_rot] = 0.0
    for idx, d, fct, scale in imposed:
        val = scale * (fct.eval(t_new) - fct.eval(t_old))
        if d < 3:
            u[idx, d] = val
        else:
            ur_s[idx, d - 3] = val

    inc = IncrementResult(load_factor=t_new, converged=False, iterations=0)
    fint = mint = None
    fd_new = None
    fcont = None
    fext_new = fext
    massz = np.where(model.mass >= 1e29, 0.0, model.mass)

    def _reduce(vec):
        return constr.reduce_vector(vec) if constr is not None else vec

    for it in range(ip.impl_max_iter):
        # Newmark kinematics of the current trial increment
        a_new = c0 * u - v / (beta * dt) - (0.5 / beta - 1.0) * a
        ar_new = c0 * ur_s - vr / (beta * dt) - (0.5 / beta - 1.0) * ar

        # Generalized-alpha intermediate acceleration: a_{n+1-alpha_m}
        a_int = (1.0 - alpha_m) * a_new + alpha_m * a
        ar_int = (1.0 - alpha_m) * ar_new + alpha_m * ar

        fint, mint = _internal_forces(model, x_ref, u, ur_s, committed,
                                      nlgeom)
        if contacts:
            from .contact import contact_forces
            fcont, _ = contact_forces(contacts, model.x + u, n)
            fint = fint + fcont
        fext_new = _fext_trial(u)

        # Equilibrium at intermediate configuration t_{n+1-alpha_f}:
        # (1 - alpha_f) (f_ext + f_int - C v)_{n+1} + alpha_f (f_ext + f_int - C v)_n - M a_{n+1-alpha_m} = 0
        Rf = (c_K * (fext_new + fint) + alpha_f * g_prev_f
              - massz[:, None] * a_int)
        Rm = (c_K * mint + alpha_f * g_prev_m
              - model.inertia[:, None] * ar_int)
        R = _reduce(dof.gather_residual(Rf, Rm))
        if damp is not None:
            v_new = v + dt * ((1.0 - gamma) * a + gamma * a_new)
            vr_new = vr + dt * ((1.0 - gamma) * ar + gamma * ar_new)
            fd_new = _fd(v_new, vr_new)
            R = R - _reduce(c_K * fd_new + alpha_f * fd_prev)
        rnorm = float(np.linalg.norm(R))
        inc.residuals.append(rnorm)
        inc.iterations = it + 1
        if not np.isfinite(rnorm):
            break
        if it > 0 and rnorm > 1e4 * ref:
            break
        if it == 0:
            ma = _reduce(dof.gather_residual(massz[:, None] * a_int,
                                             model.inertia[:, None]
                                             * ar_int))
            ma_norm = float(np.linalg.norm(ma))
            if np.isfinite(ma_norm):
                ref = max(ref, ma_norm)
            ref = max(ref, rnorm)

        if rnorm <= ip.impl_tol * ref:
            inc.converged = True
            break

        # Tangent update policy (IKT: 1=KTANG, 2=KTFUL, 4=KTCON)
        recompute_K = False
        if ikt == 1:
            recompute_K = True
        elif ikt == 2:
            recompute_K = (it == 0)
        elif ikt == 4:
            recompute_K = (cached_K is None)
        else:
            recompute_K = True

        if recompute_K:
            epsp_incr = _epsp_increments(model, epsp0)
            x_tan = x_ref + u if nlgeom else x_ref
            K = assemble(model, dof, x_tan, epsp_incr, kgeo=nlgeom)
            if contacts:
                from .contact import contact_tangent
                K = K + contact_tangent(contacts, model.x + u, dof)
            if lstiff:
                K = K + pload_tangent(loads, model, t_new, model.x + u, dof)
            cached_K = K
            K_eff = c_K * K + M_diag
            if C_eff is not None:
                K_eff = K_eff + C_eff
            cached_K_eff = K_eff
            cached_dt = dt
        else:
            if cached_K_eff is None or cached_dt != dt:
                K_eff = c_K * cached_K + M_diag
                if C_eff is not None:
                    K_eff = K_eff + C_eff
                cached_K_eff = K_eff
                cached_dt = dt
            else:
                K_eff = cached_K_eff

        try:
            if constr is not None:
                du_eq = constr.expand(
                    solver.solve(constr.reduce_matrix(K_eff), R))
            else:
                du_eq = solver.solve(K_eff, R)
        except (RuntimeError, ValueError):
            inc.converged = False
            break
        du, dur = dof.scatter_solution(du_eq)
        u = u + du
        ur_s = ur_s + dur

        unorm = np.linalg.norm(dof.gather_residual(u, ur_s))
        if np.isfinite(unorm) and np.linalg.norm(du_eq) <= ip.impl_tol * max(unorm, 1e-30) \
                and it > 0:
            a_new = c0 * u - v / (beta * dt) - (0.5 / beta - 1.0) * a
            ar_new = c0 * ur_s - vr / (beta * dt) - (0.5 / beta - 1.0) * ar
            a_int = (1.0 - alpha_m) * a_new + alpha_m * a
            ar_int = (1.0 - alpha_m) * ar_new + alpha_m * ar
            fint, mint = _internal_forces(model, x_ref, u, ur_s, committed,
                                          nlgeom)
            if contacts:
                from .contact import contact_forces
                fcont, _ = contact_forces(contacts, model.x + u, n)
                fint = fint + fcont
            fext_new = _fext_trial(u)
            Rf = (c_K * (fext_new + fint) + alpha_f * g_prev_f
                  - massz[:, None] * a_int)
            Rm = (c_K * mint + alpha_f * g_prev_m
                  - model.inertia[:, None] * ar_int)
            R = _reduce(dof.gather_residual(Rf, Rm))
            if damp is not None:
                v_new = v + dt * ((1.0 - gamma) * a + gamma * a_new)
                vr_new = vr + dt * ((1.0 - gamma) * ar + gamma * ar_new)
                fd_new = _fd(v_new, vr_new)
                R = R - _reduce(c_K * fd_new + alpha_f * fd_prev)
            inc.residuals.append(float(np.linalg.norm(R)))
            inc.iterations = it + 2
            inc.converged = True
            break

    if inc.converged:
        model.x = model.x + u
    return inc, u, ur_s, fint, mint, fext_new, fd_new, cached_K, cached_K_eff, cached_dt


def _dyn_summary(model, result, dof, log, e0):
    """Termination page (the statics summary + the energy balance)."""
    real = model.mass < 1e29
    disp = model.x - model.x0
    umax = float(np.abs(disp[real]).max()) if real.any() else 0.0
    total_it = sum(i.iterations for i in result.increments)
    h = result.history
    log.info("\n     ------------------------------------------------")
    if result.converged:
        log.info("     IMPLICIT DYNAMIC TERMINATION : NORMAL")
    else:
        log.info(f"     IMPLICIT DYNAMIC TERMINATION : ERROR — "
                 f"{result.stop_reason}")
    log.info(f"     TIME STEPS  . . . . . . . : {len(result.increments)}")
    log.info(f"     TOTAL NEWTON ITERATIONS . : {total_it}")
    log.info(f"     FREE DOFS . . . . . . . . : {dof.ndof}")
    log.info(f"     MAX |DISPLACEMENT|  . . . : {umax:14.7E}")
    if h["t"]:
        ref = max(abs(e0), abs(h['ke'][-1] + h['ie'][-1]),
                  abs(h['wext'][-1]), 1e-12)
        log.info(f"     KINETIC ENERGY  . . . . . : {h['ke'][-1]:14.7E}")
        log.info(f"     INTERNAL ENERGY . . . . . : {h['ie'][-1]:14.7E}")
        log.info(f"     EXTERNAL WORK . . . . . . : {h['wext'][-1]:14.7E}")
        if h.get("edamp") and h["edamp"][-1] != 0.0:
            log.info(f"     RAYLEIGH DISSIPATION  . . : "
                     f"{h['edamp'][-1]:14.7E}  (/IMPL/DYNA/DAMP)")
        if h.get("enum") and abs(h["enum"][-1]) > 1e-12:
            log.info(f"     NUMERICAL DISSIPATION . . : "
                     f"{h['enum'][-1]:14.7E}  (GEN-ALPHA/HHT)")
        if h.get("econt") and h["econt"][-1] != 0.0:
            log.info(f"     CONTACT SPRING ENERGY . . : "
                     f"{h['econt'][-1]:14.7E}  (/INTER)")
        if h.get("efric") and h["efric"][-1] != 0.0:
            log.info(f"     FRICTION DISSIPATION  . . : "
                     f"{h['efric'][-1]:14.7E}  (/INTER/TYPE7)")
        log.info(f"     ENERGY BALANCE  . . . . . : "
                 f"{h['bal'][-1] / ref * 100.0:8.2f} %")
    log.info("     ------------------------------------------------")
