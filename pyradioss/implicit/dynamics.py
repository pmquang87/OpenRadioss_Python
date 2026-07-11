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

NOT mirrored (deferred explicitly, PORTING_GUIDE M10): the ``QSTAT_*``
quasi-static-initialization branch, ``/IMPL/DYNA/DAMP`` Rayleigh damping
(``IDY_DAMP``/``DAMPA_IMP``/``DAMPB_IMP``), and the automatic implicit
time-step control of ``imp_dt.F`` (the port stops on non-convergence
instead of cutting dt).

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
from .statics import (ImplicitResult, IncrementResult, _internal_forces,
                      _resolve_imposed, _snapshot, _solver_banner)


class ImplicitDynResult(ImplicitResult):
    """Outcome of an implicit dynamic run (attached as
    ``model.implicit_result``): the per-step convergence records of
    :class:`ImplicitResult` (``increments[k].load_factor`` holds the step's
    END TIME) plus the scheme actually run and the energy/motion history."""

    def __init__(self, alpha, gamma, beta):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.beta = beta
        #: per-converged-step history: "t", kinetic "ke", internal+hourglass
        #: "ie", external work "wext", balance "bal" = ie + ke - wext - e0
        #: (lists of floats), and "u" — displacement snapshots (numnod, 3)
        #: for the validations (kept while numnod stays example-sized).
        self.history = {"t": [], "ke": [], "ie": [], "wext": [], "bal": [],
                        "u": []}


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
    stiffness, so the system stays well posed."""
    n = model.numnod
    node = np.arange(n)
    M = np.zeros(dof.ndof)
    for c in range(3):
        e = dof.eq[node * DOFS_PER_NODE + c]
        act = e >= 0
        M[e[act]] = model.mass[act]
        e = dof.eq[node * DOFS_PER_NODE + 3 + c]
        act = e >= 0
        M[e[act]] = model.inertia[act]
    return M


def _disable_rate_devices(model, log):
    """Statics/dynamics share the pseudo-velocity kernel drive, so the
    rate devices must not run (module docstring): zero the solid bulk
    viscosity like statics, and zero the LAW2 rate coefficient with an
    explicit DEFERRED warning (never silently du/1)."""
    nvisc = 0
    for name, group in model.element_groups():
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


# ----------------------------------------------------------------------------
# The driver
# ----------------------------------------------------------------------------

def run_implicit_dynamic(model, controls, log, out_dir=None, run_name="RUN",
                         run_num=1):
    """Run an implicit dynamic (Newmark/HHT) analysis on an initialized
    model. ``controls.impl_dyna`` selects the scheme (1 = HHT from alpha,
    2 = Newmark gamma/beta); /RUN's final time and /IMPL/DTINI are PHYSICAL.
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

    # ---- scheme constants (DYNA_INI) --------------------------------------
    if ip.impl_dyna == 1:
        alpha = float(ip.impl_dyna_alpha)
        gamma = 0.5 - alpha                     # DY_G = HALF - D_AL
        beta = 0.25 * (1.0 - alpha) ** 2        # DY_B = FOURTH*(1-D_AL)^2
    else:
        alpha = 0.0                             # plain Newmark (D_AL = ZERO)
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

    scheme = (f"HHT-ALPHA (ALPHA = {alpha:g})" if ip.impl_dyna == 1
              else "NEWMARK")
    log.info(f" TIME INTEGRATION . . . . . . . . . . : {scheme}")
    log.info(f" NEWMARK GAMMA / BETA . . . . . . . . : {gamma:g} / {beta:g}"
             + ("  (TRAPEZOIDAL RULE)"
                if gamma == 0.5 and beta == 0.25 else ""))
    log.info(f" LINEAR SOLVER  . . . . . . . . . . . : "
             f"{_solver_banner(ip, log)}")
    log.info(f" GEOMETRY . . . . . . . . . . . . . . : "
             f"{'NONLINEAR (UPDATED-LAGRANGIAN + KGEO)' if nlg else 'LINEAR (SMALL STRAIN)'}")

    _disable_rate_devices(model, log)

    imposed, presc = _resolve_imposed(model, log)
    dof = DofMap(model, log, prescribed=presc)
    loads = LoadsAndConstraints(model, log)
    solver = LinearSolver(ip.impl_linsolve, log)
    M_eq = _lumped_mass_eq(model, dof)

    committed = {name: _snapshot(g) for name, g in model.element_groups()}
    x_ref = model.x0.copy()
    model.x = model.x0.copy()

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

    result = ImplicitDynResult(alpha, gamma, beta)
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
    fext = np.zeros((n, 3))
    loads.external_forces(0.0, fext, x_ref)
    fint, mint = _internal_forces(model, x_ref, np.zeros((n, 3)),
                                  np.zeros((n, 3)), committed, nlg)
    R0 = dof.gather_residual(fext + fint, mint)
    a0_eq = np.where(M_eq > 0.0, R0 / np.where(M_eq > 0.0, M_eq, 1.0), 0.0)
    a, ar = dof.scatter_solution(a0_eq)
    # previous-level force g_n = (f_ext + f_int)_n for the HHT weighting
    g_prev_f = fext + fint
    g_prev_m = mint.copy()
    fext_prev = fext.copy()

    # ---- energy ledger seed -------------------------------------------------
    real = model.mass < 1e29
    e0 = _kinetic(model, v, vr, real) + _elem_energy(model)
    wext = 0.0
    keep_u = True

    log.info("\n        STEP        TIME    ITER   RESIDUAL-NORM   STATUS")

    t = 0.0
    step_no = 0
    while t < t_end * (1.0 - 1e-12):
        dt_s = min(dt, t_end - t)   # clip the final step to land on t_end
        t_new = t + dt_s
        step_no += 1

        inc, u, ur_s, fint, mint, fext_new = _solve_step(
            model, ip, dof, loads, solver, committed, x_ref, imposed,
            t, t_new, dt_s, alpha, gamma, beta, M_eq, v, vr, a, ar,
            g_prev_f, g_prev_m, nlg)
        result.increments.append(inc)
        if not inc.converged:
            result.converged = False
            result.stop_reason = (
                f"NEWTON DID NOT CONVERGE AT TIME {t_new:.4E} IN "
                f"{ip.impl_max_iter} ITERATIONS "
                f"(||R|| = {inc.residuals[-1]:.4E}) — reduce /IMPL/DTINI "
                f"(automatic implicit time-step control is deferred, see "
                f"PORTING_GUIDE M10)")
            log.info(f" {step_no:11d} {t_new:11.4E} {inc.iterations:6d} "
                     f"{inc.residuals[-1]:14.5E}   *** NO CONVERGENCE")
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
        wext += 0.5 * float(((fext_prev + fext_new)[real] * u[real]).sum())
        if imposed:
            # constraint reaction on a driven DOF: what the drive must
            # supply beyond the loads, r = M a - (f_ext + f_int)
            g_new_r = model.mass[:, None] * a_new - (fext_new + fint)
            g_old_r = model.mass[:, None] * a - g_prev_f
            for idx, d, fct, scale in imposed:
                wext += 0.5 * float(((g_new_r + g_old_r)[idx, d]
                                     * u[idx, d]).sum())

        v, a, vr, ar = v_new, a_new, vr_new, ar_new
        g_prev_f = fext_new + fint
        g_prev_m = mint.copy()
        fext_prev = fext_new
        committed = {name: _snapshot(g) for name, g in model.element_groups()}
        if nlg:
            x_ref = model.x.copy()
        t = t_new

        # ---- history --------------------------------------------------------
        ke = _kinetic(model, v, vr, real)
        ie = _elem_energy(model)
        h = result.history
        h["t"].append(t)
        h["ke"].append(ke)
        h["ie"].append(ie)
        h["wext"].append(wext)
        h["bal"].append(ie + ke - wext - e0)
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
                g_prev_f, g_prev_m, nlgeom):
    """Newton-iterate one time step to the HHT-weighted dynamic balance
    (see module docstring). Returns ``(inc, u, ur, fint, mint, fext_new)``
    with ``u``/``ur`` the converged step displacement increment and
    ``fint``/``mint`` the internal force at the converged state (kept by
    the caller as the next step's HHT previous-level force).

    On entry the element buffers hold the committed state; on a converged
    return model.x and the element state hold the new configuration —
    exactly the statics increment contract, plus inertia."""
    n = model.numnod
    c0 = 1.0 / (beta * dt * dt)             # the IMP_DYNAM diagonal factor
    ap1 = 1.0 + alpha
    from . import require_scipy
    sp, _ = require_scipy()
    M_diag = sp.diags(c0 * M_eq, format="csr")   # the lumped-mass add to K

    # external force at the END time level (the HHT combination weights it
    # against the stored previous level below)
    fext = np.zeros((n, 3))
    loads.external_forces(t_new, fext, x_ref)
    # load reference from the end-level force alone (the alpha blend only
    # shifts it by O(alpha dt f_dot), irrelevant to a norm)
    fext_eq = dof.gather_residual(fext, np.zeros((n, 3)))
    ref = max(float(np.linalg.norm(fext_eq)), 1e-30)

    epsp0 = {name: committed[name].get("epsp")
             for name, _ in model.element_groups()}

    # ---- predictor: constant-acceleration extrapolation (a_{n+1} = a_n) ----
    # Newton converges from any start; this one lands the first residual on
    # the true force scale of the step (free vibration has f_ext = 0, so a
    # zero predictor would leave the reference norm empty-handed).
    u = dt * v + 0.5 * dt * dt * a
    ur_s = dt * vr + 0.5 * dt * dt * ar
    # condensed DOFs must not inherit a predictor drift (their equations
    # never enter the solve, so Newton could not correct one)...
    u[dof.fix_tra] = 0.0
    ur_s[dof.fix_rot] = 0.0
    # ...and the /IMPDISP-driven DOFs (part of that mask) are then seeded
    # EXACTLY: the drive steps from d(t_n) to d(t_{n+1})
    for idx, d, fct, scale in imposed:
        u[idx, d] = scale * (fct.eval(t_new) - fct.eval(t_old))

    inc = IncrementResult(load_factor=t_new, converged=False, iterations=0)
    fint = mint = None

    for it in range(ip.impl_max_iter):
        # Newmark kinematics of the current trial increment
        a_new = c0 * u - v / (beta * dt) - (0.5 / beta - 1.0) * a
        ar_new = c0 * ur_s - vr / (beta * dt) - (0.5 / beta - 1.0) * ar

        fint, mint = _internal_forces(model, x_ref, u, ur_s, committed,
                                      nlgeom)
        # R = (1+a)(f_ext + f_int)_{n+1} - a g_n - M a_{n+1}   (node space)
        Rf = (ap1 * (fext + fint) - alpha * g_prev_f
              - model.mass[:, None] * a_new)
        Rm = (ap1 * mint - alpha * g_prev_m
              - model.inertia[:, None] * ar_new)
        R = dof.gather_residual(Rf, Rm)
        rnorm = float(np.linalg.norm(R))
        inc.residuals.append(rnorm)
        inc.iterations = it + 1
        if it == 0:
            # reference: the largest of the applied load, the inertial
            # force of the predicted motion (THE force scale of a free
            # vibration) and the initial out-of-balance
            ma = dof.gather_residual(model.mass[:, None] * a_new,
                                     model.inertia[:, None] * ar_new)
            ref = max(ref, float(np.linalg.norm(ma)), rnorm)

        if rnorm <= ip.impl_tol * ref:
            inc.converged = True
            break

        epsp_incr = {}
        for name, group in model.element_groups():
            if name == "bricks" and epsp0[name] is not None:
                epsp_incr[name] = group.state["epsp"] - epsp0[name]
        x_tan = x_ref + u if nlgeom else x_ref
        K = assemble(model, dof, x_tan, epsp_incr, kgeo=nlgeom)
        # K_eff = (1+alpha) K_T + M/(beta dt^2)  (IMP_DYNAM's diagonal add)
        K_eff = ap1 * K + M_diag
        du_eq = solver.solve(K_eff, R)
        du, dur = dof.scatter_solution(du_eq)
        u = u + du
        ur_s = ur_s + dur

        # displacement-correction convergence (the statics second norm):
        # accept when the correction is negligible against the accumulated
        # increment, re-evaluating the residual at the corrected state
        unorm = np.linalg.norm(dof.gather_residual(u, ur_s))
        if np.linalg.norm(du_eq) <= ip.impl_tol * max(unorm, 1e-30) \
                and it > 0:
            a_new = c0 * u - v / (beta * dt) - (0.5 / beta - 1.0) * a
            ar_new = c0 * ur_s - vr / (beta * dt) - (0.5 / beta - 1.0) * ar
            fint, mint = _internal_forces(model, x_ref, u, ur_s, committed,
                                          nlgeom)
            Rf = (ap1 * (fext + fint) - alpha * g_prev_f
                  - model.mass[:, None] * a_new)
            Rm = (ap1 * mint - alpha * g_prev_m
                  - model.inertia[:, None] * ar_new)
            inc.residuals.append(
                float(np.linalg.norm(dof.gather_residual(Rf, Rm))))
            inc.iterations = it + 2
            inc.converged = True
            break

    if inc.converged:
        model.x = model.x + u
    return inc, u, ur_s, fint, mint, fext


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
        log.info(f"     ENERGY BALANCE  . . . . . : "
                 f"{h['bal'][-1] / ref * 100.0:8.2f} %")
    log.info("     ------------------------------------------------")
