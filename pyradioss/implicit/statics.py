"""
Implicit static analysis: the Newton–Raphson equilibrium driver (M8).

Fortran origin: ``engine/source/implicit/imp_solv.F`` (the implicit driver:
load increments and the outer Newton loop) with ``imp_buck.F`` /
``imp_conv.F`` for the convergence bookkeeping. This is the implicit analogue
of the explicit ``resol.F`` time loop — but instead of marching a stable time
step it walks a sequence of LOAD increments and solves a nonlinear
equilibrium problem at each one.

The problem
-----------
Statics drops the inertia term: equilibrium is simply

    f_ext(lambda) + f_int(u) = 0            (net nodal force zero)

where ``lambda`` is the load factor (the implicit "time"), f_ext the applied
load scaled by lambda and f_int the internal force the ELEMENT KERNELS already
compute. (Sign convention of the port: the kernels accumulate the internal
force NEGATED into ``fint``, so ``fint`` is exactly the nodal force term above
— equilibrium is ``fext + fint = 0``.)

Newton–Raphson linearizes the residual R(u) = f_ext + f_int(u):

    R(u_k) + dR/du . du = 0,   dR/du = -K   ⇒   K du = R(u_k),   u_{k+1}=u_k+du

with K = -d f_int/du the tangent stiffness assembled from the element
tangents (``assembly.assemble``). K is solved by the direct linear solver
(``linsolve``). For a linear-elastic problem R is linear in u, so Newton
converges in ONE step (residual to round-off) — the sharpest form of the
quadratic convergence the M8 validations assert.

Reusing the explicit force kernels for f_int
--------------------------------------------
The kernels are hypoelastic / rate form: they compute a strain INCREMENT from
a velocity field and a step dt, integrate the stress and return the internal
force. The implicit driver reuses them unchanged by feeding the displacement
increment of the current load step as a *pseudo-velocity* at ``dt = 1``:

    forces(group, x = x_ref + u, v = u, vr = ur, dt = 1)

Then the kernel's strain increment ``L*dt`` is exactly grad(u), the stress
integrates C:grad(u) on top of the committed stress, and the returned force is
the internal force at the trial displacement. Because the kernel MUTATES the
element state (stress, plastic strain, hourglass, energies) on every call, the
driver snapshots the committed state before each residual evaluation and
restores it — so each evaluation is a pure function of ``u`` from the same
committed base. On convergence the last (converged) state is kept and becomes
the base for the next increment.

The viscous solid hourglass, driven this way at dt = 1, acts as a linear
stiffness a_h per mode; the element tangent (``solid_hexa8.tangent``)
reproduces exactly that a_h, so residual and tangent stay consistent and the
hourglass modes are stabilized without polluting uniform-strain states. The
shell BLT84 hourglass is already stiffness-type and behaves the same way.

One rate device is switched OFF for statics: the solid BULK VISCOSITY
(qa/qb shock damping). It is a function of the strain RATE, and the
pseudo-velocity drive would otherwise leak a spurious viscous pressure into
every compressive increment (trD < 0) — a statics run must depend on the
displacement state alone, exactly as the original's implicit branch runs
without it. The driver zeroes qa/qb once at startup (the run is
implicit-only, so nothing else reads them).

Nonlinear geometry (M9): the updated-Lagrangian step and K_geo
--------------------------------------------------------------
Fortran origin: the /IMPL/NONLIN branch of ``imp_solv.F`` (the
updated-Lagrangian outer step) with the geometric-stiffness assembly
(``imp_kgeo`` inside ``imp_glob_k.F``). Activated by ``/IMPL/NONLIN``;
the default stays the exact M8 small-strain path (byte-identical).

M8 froze the reference frame at x0 and linearized every increment there —
exact for small strain, blind to stress stiffening and buckling. With
nonlinear geometry ON, three things change (each increment, nothing else):

1. **The committed frame ADVANCES**: after an increment converges, the
   reference geometry becomes the deformed configuration (x_ref += u) — the
   classic updated-Lagrangian outer step. Rotations accumulate increment by
   increment through the corotational kernels.
2. **The residual is evaluated on the trial configuration**: the stress
   increment integrates at the MIDPOINT geometry x_ref + u/2 (the
   Hughes–Winget midpoint rule: for the increment map x_new = R x_old the
   midpoint gradient 2(R-I)(R+I)^-1 is the Cayley transform of R — EXACTLY
   skew for any finite rigid rotation, so a rigid increment produces
   identically zero strain, where an end-point evaluation would leak a
   1-cos(theta) spurious strain per increment); the nodal force is then
   re-assembled on the END geometry x_ref + u, where equilibrium is stated
   (each element's ``static_internal_forces``). Both reuse the explicit
   kernels' machinery; remaining incremental-objectivity error is
   O(dtheta^2) per increment (the linearized Jaumann stress rotation).
3. **The tangent gains the geometric term**: K = K_material + K_hourglass +
   K_geo, all linearized at the trial (end) geometry, with
   K_geo = int G^T [sigma] G dV built from the current stress (each
   element's ``kgeo``; the delta_ij initial-stress operator — see the
   element modules for the derivation). K_geo -> 0 at zero stress, so the
   small-strain limit reproduces M8. Compressive stress makes K lose
   positive definiteness at the buckling load — which is exactly what lets
   the driver *find* limit points instead of marching through them.

Arc-length continuation (M9): past the limit point
--------------------------------------------------
Load control prescribes lambda and solves for u — at a limit point (a peak
of the load-displacement curve, e.g. snap-through) no equilibrium exists at
lambda > lambda_max and Newton diverges: the load increment can only STOP
there. The arc-length method (Riks 1979 / Crisfield 1981) treats lambda as
an UNKNOWN and constrains the step length in (u, lambda) space instead:

    R(u, lambda) = lambda*q + f_int(u) = 0                    (n equations)
    c(u, lambda) = ||Delta_u||^2 + w*Delta_lambda^2 - dl^2 = 0   (constraint)

with the SPHERICAL (Riks) metric weight w = ||K0^-1 q||^2 sampled once at
startup (psi = 1 in Crisfield's psi-scaled family): it makes the load term
commensurate with the displacement term, so on a very stiff branch (where
||du_t|| collapses, e.g. after a snap-through re-stiffens) the constraint
still bounds the LOAD step — the pure cylindrical form (w = 0) lets lambda
jump arbitrarily far there. Each corrector iteration solves the two
auxiliary systems K du_bar = R and K du_t = q, writes
du = du_bar + dlambda*du_t and picks dlambda from the constraint — a scalar
quadratic a*dlambda^2 + b*dlambda + c = 0 (Crisfield's formulation; of the
two roots, keep the one pointing along the incoming path, i.e. maximizing
the metric dot product with the current (Delta_u, Delta_lambda), so the
continuation never doubles back). The predictor direction takes the sign
that continues the previous increment. The arc length dl adapts to the
iteration count and halves on failure (complex roots / no convergence).
This is what "turns the corner": lambda DECREASES on the descending branch
while the displacement keeps growing, and recovers past the snap.
OpenRadioss reaches the same continuation through the arc-length option of
its /IMPL/NONLIN controls; the port exposes it as the minimal sub-card
``/IMPL/ARCL`` (see engine_keywords). Requires proportional loading
(f_ext = lambda*q — asserted at startup) and no /IMPDISP.

Implicit DYNAMICS landed as M10 (``implicit/dynamics.py`` — /IMPL/DYNA,
Newmark/HHT on top of this statics core: each time step reuses this
module's residual evaluation and commit machinery plus inertia). M11 made
the element-tangent set COMPLETE (tetra4 / sh3n / beam / spring joined
hexa8 / BT4 / truss — see the element modules), added the LAW2 shell/truss
consistent tangents, the automatic step control (``StepControl`` below —
imp_dt.F) and the /IMPL/BUCKL engine card (``_run_buckling`` below).
Total-form elements (the spring) and the LAW2 truss carry their own
implicit residual, ``implicit_internal_forces``, dispatched by
``_internal_forces`` instead of forces() — see their module notes.

M12 added the two structural gaps this module had left: KINEMATIC
CONSTRAINTS BY CONDENSATION (/RBODY, /RBE2, /INTER/TYPE2, /RBE3, /MPC —
``constraints.py``: the rby_imp0.F / rbe2_imp0.F / rbe3_imp0.F /
i2_imp1.F transformations K_red = T^T K T, R_red = T^T R around every
Newton solve, rebuilt per committed frame under NLGEOM) and PENALTY
CONTACT in the loop (/INTER/TYPE7 — ``contact.py``: the i7ke3.F force in
the residual at the trial configuration + the exact gap tangent in K,
with the active set re-evaluated every iteration; a chattering set that
fails its Newton budget lands in the M11 StepControl cut, the imp_dt.F
coupling). Rigid walls are REFUSED under implicit (a kinematic device of
the explicit update; silently ignoring a wall would drop a real boundary
condition).

M13 closed the M12 deferral list's head: /INTER/TYPE7 COULOMB FRICTION
(the i7kfor3.F incremental return mapping — ``contact.py``, anchors
committed per increment through ``commit_contacts``), /INTER/TYPE11
edge-to-edge, the /PLOAD FOLLOWER-LOAD stiffness under /IMPL/NONLIN
(trial-configuration pressure residual + ``followerload.pload_tangent``;
/PLOAD + /IMPL/ARCL refused) and the LAW36 consistent tangents (rate
families truncated to the static curve — ``_law36_static_curve``). It
also added the imp_solv.F-style backtracking LINE SEARCH in the Newton
loop below (the ILINE branch: engages ONLY when the residual norm grows,
so every monotone run is bit-identical — it breaks the period-2
assignment cycles a friction stick/slip boundary can fall into) and the
persistent implicit hourglass state (see solid_hexa8.static_stabilization
— the incremental form ratcheted across commits).

M14 closed the remaining structural refusals: constraint CHAINS resolve
by transform substitution (constraints.py), /INTER/TYPE11 gained the
Coulomb return mapping (contact.py), /IMPL/ARCL runs WITH constraints
and contact (the corrector's two auxiliary solves and the spherical
metric in the REDUCED space, the active set re-evaluated inside the
corrector — see _run_arclength for the documented deviation from the
original's full-space Riks norm), /IMPL/BUCKL runs WITH constraints and
contact (the condensed pencil with the converged active set's tangent in
K_mat — buckling.py), and LAW42 gained its consistent spectral tangent
(materials/law42_ogden.py; /IMPL/NONLIN required — the total-form law
has no meaning on the frozen small-strain frame, checked below).

M15 closed the last material/friction refusals: Ifric > 0 friction
models (the mu(p) cone + consistent mu'(p) tangent, static-limit
velocity terms — contact.py), the LAW27 damaged-crack shell tangent and
the LAW2 beam resultant-plasticity tangent (with its own ITERATED
implicit return — beam_type3.implicit_internal_forces, the truss's M11
lesson applied to the resultant space).

Still DEFERRED (documented, not half-done — see the package docstring
and PORTING_GUIDE): rate devices under implicit (disabled loudly),
thermal contact, Inacti/Igap 2/3, IDTC 2/3.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

import numpy as np

from ..elements import KERNELS
from ..engine.kinematics import LoadsAndConstraints
from .assembly import assemble
from .dofmap import DofMap
from .linsolve import LinearSolver


class StepControl:
    """Automatic implicit step-size control (M11).

    Fortran origin: ``engine/source/implicit/imp_dt.F`` — ``IMP_DTN``, the
    routine ``imp_solv.F`` calls after every increment attempt:

    * on NON-convergence (IMCONV < 0) the driver rolls the state back
      (``TT = TT - DT2``, ``NCYCLE = NCYCLE - 1``), multiplies the step by
      SCAL_DTN (bounded below by DT_MIN) and RETRIES — it stops only when
      DT_IMP hits DT_MIN (``IMP_STOP``);
    * on convergence, the IDTC = 1 branch GROWS the step by SCAL_DTP when
      the increment needed at most NL_DTP Newton iterations, capped at
      DT_MAX — the "grow back after easy steps" recovery.

    This is exactly the arc-length radius-adaptation idea applied to plain
    load/time stepping, and the port drives BOTH the statics load
    increments and the dynamics time steps through this one object. The
    IDTC = 2/3 branches (displacement-norm / Riks step control) are
    deferred (PORTING_GUIDE M11).

    Card mapping (/IMPL/DT/1: NL_DTP SCAL_DTP NL_DTN SCAL_DTN, /IMPL/DT/STOP:
    DT_MIN DT_MAX — freimpl.F): the original's DEFAULTS for these live in an
    engine-init routine, not the reader, so the port sets its own DOCUMENTED
    defaults: target iterations 6, growth 1.1, cut 0.5, DT_MAX = the
    /IMPL/DTINI step (the step grows back toward what the user asked for,
    never beyond), DT_MIN = 1e-4 of it (≈13 halvings — a bounded retry
    budget by construction). The control is ALWAYS ON, mirroring the source
    (IMP_DTN cuts regardless of /IMPL/DT); the cards only tune it.
    """

    def __init__(self, controls, dt_ini, log, what="TIME STEP"):
        self.itw = max(1, int(getattr(controls, "impl_dt_itw", 6)))
        self.up = max(1.0, float(getattr(controls, "impl_dt_scaleup", 1.1)))
        self.dn = min(0.99, max(0.01, float(
            getattr(controls, "impl_dt_scaledn", 0.5))))
        dmax = float(getattr(controls, "impl_dt_max", 0.0))
        dmin = float(getattr(controls, "impl_dt_min", 0.0))
        self.dt = float(dt_ini)
        self.dt_max = dmax if dmax > 0.0 else float(dt_ini)
        self.dt_min = dmin if dmin > 0.0 else 1e-4 * float(dt_ini)
        self.total_cuts = 0
        self.log = log
        self.what = what

    def cut(self):
        """Halt-or-retry decision after a failed increment: shrink the step
        and return True (retry), or return False when the step already sits
        at DT_MIN — the IMP_STOP condition."""
        if self.dt <= self.dt_min * (1.0 + 1e-12):
            return False
        old = self.dt
        self.dt = max(self.dt * self.dn, self.dt_min)
        self.total_cuts += 1
        # the IMP_DTN listing line ("--NEXT TIMESTEP IS DECREASED BY--")
        self.log.info(f"     --{self.what} DECREASED {old:.5E} -> "
                      f"{self.dt:.5E}, RETRYING (imp_dt.F)")
        return True

    def converged(self, iterations):
        """Post-convergence growth (IDTC = 1): an easy step (<= target
        iterations) grows the step back toward DT_MAX."""
        if iterations <= self.itw and self.dt < self.dt_max:
            old = self.dt
            self.dt = min(self.dt * self.up, self.dt_max)
            if self.dt != old:
                self.log.info(f"     --{self.what} INCREASED {old:.5E} -> "
                              f"{self.dt:.5E} (imp_dt.F)")


@dataclass
class IncrementResult:
    """Convergence record of one load increment (for the listing + tests)."""

    load_factor: float
    converged: bool
    iterations: int
    residuals: List[float] = field(default_factory=list)  # per-iteration ||R||


@dataclass
class ImplicitResult:
    """Outcome of an implicit static run, attached to the model as
    ``model.implicit_result`` (the analogue of ``model.engine_state``)."""

    increments: List[IncrementResult] = field(default_factory=list)
    converged: bool = True
    stop_reason: str = ""
    #: /IMPL/BUCKL (M11): critical-load multipliers of the current load and
    #: the matching (du, dur) mode shapes — None when the card is absent.
    buckling_factors: object = None
    buckling_modes: object = None
    #: /IMPL/EIGV (M16): natural frequencies (Hz), mass-normalized (du, dur)
    #: mode shapes and the (nev, 6) modal effective mass — None without the
    #: card. Mirrors the buckling fields above.
    modal_frequencies: object = None
    modal_modes: object = None
    modal_effective_mass: object = None
    #: /IMPL/MODAL/DYNA (M17): the per-mode damping ratios and the
    #: mode-superposition transient response history (a dict mirroring the
    #: M10 dynamics ledger — t/u/ke/ie/wext/edamp/bal/q). None without the
    #: card. See implicit/modal_response.py.
    modal_damping: object = None
    modal_transient_history: object = None
    #: /IMPL/FREQ (M17): the harmonic frequency-response sweep (a dict —
    #: omega/freqs/q/U/amp/phase/resonances). None without the card.
    freq_response: object = None
    #: /IMPL/CEIGV (M18): COMPLEX / DAMPED eigenvalues. The complex
    #: eigenvalues lambda_i = -zeta_i omega_i +/- i omega_{d,i}, the derived
    #: natural / damped frequencies (Hz), decay rates, damping ratios, and the
    #: complex (du, dur) mode shapes (phase lag). None without the card. See
    #: implicit/complex_modal.py.
    complex_eigenvalues: object = None
    complex_natural_freqs: object = None
    complex_damped_freqs: object = None
    complex_decay_rates: object = None
    complex_damping_ratios: object = None
    complex_modes: object = None
    #: /IMPL/CEIGV/TRAN and /FRF: the complex-mode superposition transient
    #: history and the damped complex FRF sweep (dicts). None without them.
    complex_transient_history: object = None
    complex_frf: object = None


# ----------------------------------------------------------------------------
# Element-state snapshot / restore (see module docstring)
# ----------------------------------------------------------------------------

def _snapshot(group):
    """Copy every mutable per-element array of a group (the element buffer)
    so a residual evaluation can be rolled back to the committed state."""
    snap = {}
    for k, v in group.state.items():
        if isinstance(v, np.ndarray):
            snap[k] = v.copy()
        elif k == "mat_extra":
            snap[k] = {kk: vv.copy() for kk, vv in v.items()}
    return snap


def _restore(group, snap):
    """Restore a group's element buffer from a snapshot, in place (keeps the
    array identities so nothing else holding a reference is invalidated)."""
    st = group.state
    for k, v in snap.items():
        if k == "mat_extra":
            for kk, vv in v.items():
                st[k][kk][...] = vv
        else:
            st[k][...] = v


def _epsp_increments(model, epsp0):
    """Plastic-strain increment of the current step, per element group, for
    the LAW2 CONSISTENT tangents (M8 solids; M11 shells layer-wise and the
    truss): group name -> (element state ``epsp`` now) - (``epsp`` at the
    committed start of the increment). Shapes follow each group's own state
    — (n,) for solids/trusses, (n, nip) for shell layers — and each kernel
    slices its own. Groups without a plasticity state contribute nothing
    (their kernels ignore the argument)."""
    out = {}
    for name, group in model.element_groups():
        e0 = epsp0.get(name)
        if e0 is not None and "epsp" in group.state:
            out[name] = group.state["epsp"] - e0
    return out


# ----------------------------------------------------------------------------
# Residual (internal force) evaluation — reuses the explicit kernels
# ----------------------------------------------------------------------------

def _internal_forces(model, x_ref, u, ur, committed, nlgeom=False):
    """Internal force/moment at trial increment (u, ur) from the committed
    base state. Restores the element buffers and calls the EXISTING force
    kernels with the displacement increment as a pseudo-velocity at dt = 1
    (see module docstring).

    ``nlgeom=False`` (the M8 small-strain path, byte-identical): everything
    is evaluated at the COMMITTED geometry ``x_ref``. The strain increment
    is grad(u) on the reference frame, so a linear-elastic step is exactly
    K u and Newton converges in one iteration (Hooke's law reproduced
    exactly).

    ``nlgeom=True`` (M9, /IMPL/NONLIN — the updated-Lagrangian increment):
    the kernels integrate the stress at the MIDPOINT geometry x_ref + u/2
    (Hughes–Winget: a finite rigid-rotation increment produces exactly zero
    strain there — see the module docstring) and the nodal force is then
    re-assembled at the END geometry x_ref + u by each element's
    ``static_internal_forces``, so equilibrium is stated on the trial
    configuration. The midpoint call's force output is discarded.

    Restores + calls in place; the element state is left at the trial values
    (the caller commits or re-evaluates from ``committed``)."""
    n = model.numnod
    for name, group in model.element_groups():
        _restore(group, committed[name])
    fint = np.zeros((n, 3))
    mint = np.zeros((n, 3))
    if not nlgeom:
        for name, group in model.element_groups():
            # TOTAL-form kernels (the TYPE4 spring, M11) supply their own
            # implicit residual — the pseudo-velocity trick relies on
            # RATE-form state accumulation they do not have (see
            # spring.implicit_internal_forces).
            own = getattr(KERNELS[name], "implicit_internal_forces", None)
            if own is not None:
                own(group, x_ref, u, ur, fint, mint, nlgeom=False)
                continue
            KERNELS[name].forces(group, x_ref, u, ur, 1.0, fint, mint)
            # some kernels add a STATIC stabilization the (dynamics-tuned)
            # force path cannot supply — the solid stiffness-hourglass, whose
            # explicit form is viscous and far too weak to control hourglass
            # in statics (see solid_hexa8.static_stabilization). Consistent
            # with tangent().
            stab = getattr(KERNELS[name], "static_stabilization", None)
            if stab is not None:
                stab(group, x_ref, u, ur, fint, mint)
        return fint, mint

    # ---- M9 nonlinear geometry: midpoint stress update, end assembly ------
    x_mid = x_ref + 0.5 * u
    x_end = x_ref + u
    junk_f = np.zeros((n, 3))
    junk_m = np.zeros((n, 3))
    for name, group in model.element_groups():
        # total-form kernels evaluate directly at the end configuration
        # (exact — no midpoint objectivity step needed; see the note above)
        own = getattr(KERNELS[name], "implicit_internal_forces", None)
        if own is not None:
            own(group, x_ref, u, ur, fint, mint, nlgeom=True)
            continue
        # stress/hourglass-state update at the midpoint configuration;
        # the returned force (midpoint-configuration) is discarded
        KERNELS[name].forces(group, x_mid, u, ur, 1.0, junk_f, junk_m)
        # nodal force from the updated state, on the END configuration.
        # For solids this includes the FULL hourglass stabilization
        # (a_h + k_stiff) in one term — static_stabilization must NOT be
        # called on top (it would double-count k_stiff).
        KERNELS[name].static_internal_forces(group, x_end, u, ur, fint, mint)
    return fint, mint


# ----------------------------------------------------------------------------
# The driver
# ----------------------------------------------------------------------------

def run_implicit_static(model, controls, log, out_dir=None, run_name="RUN",
                        run_num=1):
    """Run an implicit static analysis on an already-initialized model.

    ``controls`` is an ``EngineControls`` whose ``impl_*`` fields (parsed from
    the /IMPL card) set the final load factor, increment size and Newton
    tolerances. Returns the model with the converged deformed state; a
    ``model.implicit_result`` records the per-increment convergence."""
    ip = controls
    n = model.numnod
    nlg = bool(getattr(ip, "impl_nlgeom", False))
    arc = bool(getattr(ip, "impl_arc", False))
    if arc and not nlg:
        # arc length exists to trace geometrically nonlinear limit points;
        # running it on the frozen-frame small-strain path would trace a
        # LINEAR curve (no limit point ever). Force the consistent pairing.
        nlg = True
        log.info(" /IMPL/ARCL IMPLIES /IMPL/NONLIN  . . : NONLINEAR GEOMETRY ON")

    log.info("\n     IMPLICIT STATIC ANALYSIS (M8/M9)")
    log.info("     --------------------------------")

    # fail fast on un-ported element types (rather than mid-Newton): since
    # M11 every element family carries a tangent, but the gate stays for
    # any future family
    from .assembly import _TANGENT_KERNELS
    unsupported = [name for name, _ in model.element_groups()
                   if name not in _TANGENT_KERNELS]
    if unsupported:
        raise NotImplementedError(
            f"the implicit solver has no tangent for element group(s) "
            f"{unsupported} (supported: {list(_TANGENT_KERNELS)}). Remove "
            f"them or run the explicit solver.")

    log.info(f" LINEAR SOLVER  . . . . . . . . . . . : "
             f"{_solver_banner(ip, log)}")
    log.info(f" GEOMETRY . . . . . . . . . . . . . . : "
             f"{'NONLINEAR (UPDATED-LAGRANGIAN + KGEO)' if nlg else 'LINEAR (SMALL STRAIN)'}")

    # statics carries no rate effects: disable the solid bulk viscosity
    # (see the module docstring — it would leak a spurious rate pressure
    # into compressive increments through the pseudo-velocity drive)
    nvisc = 0
    for name, group in model.element_groups():
        for sl, mat, prop in group.state["slices"]:
            if prop.params.get("qa", 0.0) or prop.params.get("qb", 0.0):
                prop.params["qa"] = 0.0
                prop.params["qb"] = 0.0
                nvisc += 1
    if nvisc:
        log.info(" BULK VISCOSITY (qa/qb) . . . . . . . : DISABLED (STATICS)")
    _warn_spring_dashpot(model, log)
    _law36_static_curve(model, log)
    _check_total_form_geometry(model, nlg)

    # rigid walls are a kinematic device of the explicit velocity update —
    # under implicit they would be silently dropped boundary conditions:
    # refuse loudly (model contact against a fixed body with /INTER/TYPE7)
    if model.rwalls:
        raise NotImplementedError(
            "/RWALL is not supported by the implicit solver — replace the "
            "wall with /INTER/TYPE7 contact against a meshed (fixed) "
            "surface, or run the explicit solver.")

    # /IMPDISP prescribed displacements: known DOFs whose value ramps with the
    # load factor. They are condensed out of the equations (like /BCS) but
    # carried in the displacement vector so f_int feels them — the standard
    # implicit displacement-control treatment (see _solve_increment). /IMPVEL
    # is meaningless for statics and is ignored here.
    imposed, presc = _resolve_imposed(model, log)
    # M12: kinematic constraints (condensation transform) + penalty contact
    from .constraints import build_constraints
    from .contact import build_implicit_contacts
    constr = build_constraints(model, log)
    contacts = build_implicit_contacts(model, log)
    if constr is not None:
        constr.veto_imposed(imposed)
    dof = DofMap(model, log, prescribed=presc, constraints=constr)
    loads = LoadsAndConstraints(model, log)
    solver = LinearSolver(ip.impl_linsolve, log)

    # committed element state (base for each increment's residual evals) and
    # committed plastic strain (for the consistent-tangent increment)
    committed = {name: _snapshot(g) for name, g in model.element_groups()}
    # The reference frame: with LINEAR geometry (M8, the default) it stays at
    # the initial configuration x0 for the whole run — every residual and
    # tangent is linearized there and model.x accumulates the displacement
    # for output only. With NONLINEAR geometry (M9, /IMPL/NONLIN) x_ref
    # ADVANCES to the deformed configuration each time an increment commits
    # — the updated-Lagrangian outer step (see the module docstring).
    x_ref = model.x0.copy()
    model.x = model.x0.copy()
    if constr is not None:
        # the condensation transform, linearized at the committed frame
        # (rebuilt on every frame advance under /IMPL/NONLIN — see
        # constraints.py)
        constr.build(dof, x_ref)
        log.info(f" CONDENSED EQUATIONS (M12)  . . . . . : {constr.nred} "
                 f"(FROM {dof.ndof})")

    result = ImplicitResult()
    lam = 0.0
    dlam = ip.impl_dt if ip.impl_dt > 0 else ip.t_end
    lam_end = ip.t_end
    if lam_end <= 0.0:
        # the final load factor is the /RUN "time"; a non-positive value
        # (usually a forgotten /RUN card) would silently do nothing
        raise ValueError(
            "implicit run has a non-positive final load factor "
            f"({lam_end:g}); set /RUN <RunName> <load_factor> with a "
            "positive value (the load factor is the implicit 'time').")
    log.info(f" FINAL LOAD FACTOR  . . . . . . . . . : {lam_end:12.5E}")
    log.info(f" INCREMENT SIZE . . . . . . . . . . . : {dlam:12.5E}")
    log.info(f" NEWTON TOLERANCE (RESIDUAL). . . . . : {ip.impl_tol:12.5E}")
    log.info(f" MAX NEWTON ITERATIONS  . . . . . . . : {ip.impl_max_iter}")
    log.info("\n   INCREMENT   LOAD-FACTOR   ITER   RESIDUAL-NORM   STATUS")

    # M14: /IMPL/ARCL and /IMPL/BUCKL now run WITH constraints and contact
    # (the M12/M13 refusal removed): the arc corrector solves and its
    # constraint metric live in the REDUCED space (_run_arclength), the
    # buckling pencil is condensed T^T (.) T with the converged contact
    # tangent in K_mat (buckling.py — the documented imp_buck.F deviation).
    if arc and model.ploads:
        # a follower pressure is configuration-dependent: f_ext != lambda*q
        # once the geometry moves, which breaks the proportional-loading
        # assumption the arc-length constraint is built on (M13 — the
        # pattern was previously sampled at the INITIAL frame silently)
        raise NotImplementedError(
            "/PLOAD combined with /IMPL/ARCL is not supported: a follower "
            "pressure violates the proportional-loading assumption "
            "f_ext = lambda*q of the arc-length method (PORTING_GUIDE "
            "M13). Use load control (/IMPL/NONLIN) or replace the "
            "pressure by /CLOAD forces.")

    if arc:
        _run_arclength(model, controls, log, dof, loads, solver, committed,
                       x_ref, imposed, dlam, lam_end, result,
                       constr=constr, contacts=contacts)
        model.implicit_result = result
        if getattr(ip, "impl_buckl", 0) and result.converged:
            # /IMPL/BUCKL/2 flavour: extraction about the traced final state
            _run_buckling(model, ip, log, result, constr, contacts)
        if getattr(ip, "impl_eigv", False) and result.converged:
            _run_modal(model, ip, log, result, constr, contacts)
        if getattr(ip, "impl_modal_dyna", False) and result.converged:
            _run_modal_transient(model, ip, log, result, constr, contacts,
                                 loads)
        if getattr(ip, "impl_freq", False) and result.converged:
            _run_freqresponse(model, ip, log, result, constr, contacts, loads)
        if getattr(ip, "impl_ceigv", False) and result.converged:
            _run_complex_modal(model, ip, log, result, constr, contacts, loads)
        _final_summary(model, result, dof, log)
        return model

    # automatic increment control (M11, imp_dt.F): cut-and-retry on a failed
    # increment, grow back toward the /IMPL/DTINI size on easy ones
    ctrl = StepControl(ip, dlam, log, what="LOAD INCREMENT")
    inc_no = 0
    while lam < lam_end * (1.0 - 1e-12):
        lam_new = min(lam + ctrl.dt, lam_end)
        inc_no += 1
        inc, u_c, ur_c = _solve_increment(
            model, controls, log, dof, loads, solver, committed, x_ref,
            lam, lam_new, imposed, nlg, constr, contacts)
        result.increments.append(inc)
        if not inc.converged:
            log.info(f" {inc_no:9d} {lam_new:13.5E} {inc.iterations:6d} "
                     f"{inc.residuals[-1]:14.5E}   *** NO CONVERGENCE")
            # a failed increment left model.x untouched and the element
            # buffers are re-based from ``committed`` on the next residual
            # evaluation — cutting the increment and retrying is safe
            if ctrl.cut():
                continue
            result.converged = False
            result.stop_reason = (
                f"NEWTON DID NOT CONVERGE AT LOAD FACTOR {lam_new:.4E} "
                f"IN {ip.impl_max_iter} ITERATIONS "
                f"(||R|| = {inc.residuals[-1]:.4E}) EVEN AT THE MINIMUM "
                f"INCREMENT {ctrl.dt_min:.3E} "
                f"({ctrl.total_cuts} automatic cuts — imp_dt.F control)")
            break
        log.info(f" {inc_no:9d} {lam_new:13.5E} {inc.iterations:6d} "
                 f"{inc.residuals[-1]:14.5E}   converged")
        # commit: the element state already holds the converged step; rebase
        # the committed buffers for the next increment. With nonlinear
        # geometry the reference frame advances to the deformed
        # configuration (updated Lagrangian); otherwise it stays at x0.
        committed = {name: _snapshot(g) for name, g in model.element_groups()}
        if contacts:
            # M13: re-base the friction anchors on the converged
            # configuration (the I7KFOR3 CAND_F save — a failed increment
            # never reaches here, so the anchors always match ``committed``)
            from .contact import commit_contacts
            commit_contacts(contacts, model.x)
        if nlg:
            if constr is not None:
                # exact placement of the dependent nodes (rigid bodies via
                # the Rodrigues map of the increment rotation, tied nodes
                # on their co-rotated segment) BEFORE the frame advances —
                # the linearized map would stretch the body O(theta^2) per
                # increment (see constraints.commit_placement)
                constr.commit_placement(model, u_c, ur_c)
            x_ref = model.x.copy()
            if constr is not None:
                constr.build(dof, x_ref)      # re-linearize the arms/fits
        lam = lam_new
        ctrl.converged(inc.iterations)

    model.implicit_result = result
    if getattr(ip, "impl_buckl", 0) and result.converged:
        _run_buckling(model, ip, log, result, constr, contacts)
    if getattr(ip, "impl_eigv", False) and result.converged:
        _run_modal(model, ip, log, result, constr, contacts)
    if getattr(ip, "impl_modal_dyna", False) and result.converged:
        _run_modal_transient(model, ip, log, result, constr, contacts, loads)
    if getattr(ip, "impl_freq", False) and result.converged:
        _run_freqresponse(model, ip, log, result, constr, contacts, loads)
    if getattr(ip, "impl_ceigv", False) and result.converged:
        _run_complex_modal(model, ip, log, result, constr, contacts, loads)
    _final_summary(model, result, dof, log)
    return model


def _run_complex_modal(model, ip, log, result, constr=None, contacts=(),
                       loads=None):
    """/IMPL/CEIGV (M18): complex / damped eigenvalue extraction (+ optional
    complex-mode transient / FRF) — the engine-card wiring of
    ``complex_modal.run_complex_modal`` (a PORT card, mirroring ``_run_modal``
    for M16 and ``_run_modal_transient`` for M17: the open-source engine has
    no complex/damped eigensolver). Runs after the (usually zero-load or
    prestress) static solve, assembling (K, C, M) and extracting the complex
    modes on the committed state."""
    from .complex_modal import run_complex_modal
    run_complex_modal(model, ip, log, result, constr=constr,
                      contacts=contacts, loads=loads)


def _run_modal_transient(model, ip, log, result, constr=None, contacts=(),
                         loads=None):
    """/IMPL/MODAL/DYNA (M17): mode-superposition transient — the engine-card
    wiring of ``modal_response.run_modal_transient`` (a PORT card, the same
    way ``_run_modal`` wires the M16 eigensolver). Runs after the (usually
    zero-load or prestress) static solve, extracting the modes on the
    committed state and integrating the decoupled SDOFs over physical time."""
    from .modal_response import run_modal_transient
    run_modal_transient(model, ip, log, result, constr=constr,
                        contacts=contacts, loads=loads)


def _run_freqresponse(model, ip, log, result, constr=None, contacts=(),
                      loads=None):
    """/IMPL/FREQ (M17): harmonic frequency response — the engine-card wiring
    of ``modal_response.run_freq_response`` (a PORT card, mirroring
    ``_run_modal``). Sweeps the requested band and reports the complex FRF."""
    from .modal_response import run_freq_response
    run_freq_response(model, ip, log, result, constr=constr,
                     contacts=contacts, loads=loads)


def _run_modal(model, ip, log, result, constr=None, contacts=()):
    """/IMPL/EIGV (M16): modal (free-vibration) eigenvalue extraction on the
    CONVERGED state — the engine-card wiring of ``modal.py`` (see that module
    for why this is a PORT card: the open-source freimpl.F has no modal
    branch). Reports the lowest natural frequencies and stores
    (frequencies, modes, effective mass) on the result object, mirroring
    ``_run_buckling``.

    With /IMPL/EIGV/STRS the stiffness is the prestressed tangent
    K = K_mat + K_geo (the static increments above supplied the prestress),
    so a loaded/spinning structure reports its stress-stiffened spectrum.
    The driver's live constraint transform and contact treatments are passed
    through, so the (K, M) pencil is condensed T^T (.) T exactly like the
    buckling path."""
    from .modal import modal_frequencies
    nev = max(1, int(getattr(ip, "impl_eigv_nmode", 6)))
    prestress = bool(getattr(ip, "impl_eigv_prestress", False))
    log.info("\n     ** NATURAL FREQUENCIES COMPUTATION **   (/IMPL/EIGV)")
    if prestress:
        log.info("        (prestressed: K = K_mat + K_geo of the "
                 "committed state)")
    freqs, modes, eff = modal_frequencies(
        model, nev=nev, log=None, constraints=constr, contacts=contacts,
        prestress=prestress)
    result.modal_frequencies = freqs
    result.modal_modes = modes
    result.modal_effective_mass = eff
    log.info(f"      NUMBER OF NATURAL FREQUENCIES     {len(freqs):10d}")
    log.info("      NATURAL FREQUENCIES:")
    log.info("              MODE  FREQUENCY (HZ)")
    for i, f in enumerate(freqs):
        log.info(f"          {i + 1:10d}  {f:14.6E}")
    if len(freqs) == 0:
        log.info("          (no positive frequency extracted — check that "
                 "the model is well constrained and carries mass)")


def _run_buckling(model, ip, log, result, constr=None, contacts=()):
    """/IMPL/BUCKL (M11): linearized buckling extraction on the CONVERGED
    prestressed state — the engine-card wiring of ``buckling.py``
    (imp_buck.F). The static increments above played the role of the
    original's prestress solution; this reports the critical-load
    multipliers of the CURRENT load in the imp_buck.F listing flavour and
    stores (factors, modes) on the result object.

    M14: the driver's live constraint transform and contact treatments
    are passed through, so the eigenproblem is condensed T^T (.) T and
    K_mat carries the CONVERGED contact active set's tangent — with the
    run's committed friction anchors, not fresh ones (see buckling.py)."""
    from .buckling import buckling_factors
    nev = max(1, int(getattr(ip, "impl_buckl_nmode", 4)))
    log.info("\n     ** BUCKLING MODES COMPUTATION **        (/IMPL/BUCKL)")
    factors, modes = buckling_factors(model, nev=nev, log=None,
                                      constraints=constr, contacts=contacts)
    result.buckling_factors = factors
    result.buckling_modes = modes
    log.info(f"      NUMBER OF BUCKLING CRITICAL LOADS  {len(factors):10d}")
    log.info("      CRITICAL LOADS:")
    log.info("              NUMBER  CRITICAL LOAD")
    for i, f in enumerate(factors):
        log.info(f"          {i + 1:10d}  {f:12.5E}")
    if len(factors) == 0:
        log.info("          (no positive multiplier — the current stress "
                 "state does not destabilize under this load direction)")


def _warn_spring_dashpot(model, log):
    """The TYPE4 spring dashpot (c > 0) is a RATE device: the implicit
    residual never evaluates it (spring.implicit_internal_forces — the
    total-form elastic force only), consistent with the bulk-viscosity /
    LAW2-rate-term convention (statics carries no rate effects; under
    dynamics feeding it the pseudo-velocity du/1 would be wrong by the step
    magnitude). Deferred explicitly (PORTING_GUIDE M11) — warn, never
    silently."""
    g = getattr(model, "springs", None)
    if g is not None and g.n and np.any(g.state["cdamp"] > 0.0):
        log.warning(
            "/PROP/SPRING dashpot (c > 0) is DEFERRED under the implicit "
            "solver — the damping force is disabled and the spring runs "
            "elastic (see PORTING_GUIDE M11)", "IMPL")


def _law36_static_curve(model, log):
    """LAW36 rate handling under implicit (M13): a multi-curve family is
    rate-interpolated by the EXPLICIT kernel from the increment's
    pseudo-rate |dev(du)|/1 — a step-size artifact, not a physical rate
    (the same reason the LAW2 strain-rate term is disabled). The implicit
    run therefore TRUNCATES the family to its first (lowest-rate, static)
    curve with a warning, which keeps the residual (the explicit kernel)
    and the M13 consistent tangent (law36_tabulated.consistent_*_tangent,
    which reads the static curve) exactly consistent. Rate-dependent
    tabulated plasticity under implicit stays deferred (PORTING_GUIDE
    M13) — never fed du/1 silently. Single-curve materials are untouched
    (the rate argument then never selects anything)."""
    warned = set()
    for name, group in model.element_groups():
        for sl, mat, prop in group.state["slices"]:
            if mat.law == 36 and len(mat.params.get("curve_x", ())) > 1 \
                    and id(mat) not in warned:
                warned.add(id(mat))
                nfun = len(mat.params["curve_x"])
                for key in ("curve_x", "curve_y", "curve_s"):
                    mat.params[key] = mat.params[key][:1]
                mat.params["rates"] = mat.params["rates"][:1]
                log.warning(
                    f"/MAT/LAW36/{mat.id}: the strain-rate curve family "
                    f"({nfun} curves) is DEFERRED under the implicit "
                    f"solver — only the first (static) curve is used "
                    f"(see PORTING_GUIDE M13)", "IMPL")


def _check_total_form_geometry(model, nlg):
    """LAW42 under implicit REQUIRES /IMPL/NONLIN (M14): the total-form
    stress is a pure function of F, and on the frozen small-strain frame F
    is built from the REFERENCE coordinates — the trial displacement never
    enters it, so a 'small-displacement hyperelastic' run would iterate a
    residual that cannot move. Refused loudly, never run wrong. (Shared by
    statics and dynamics — both geometry modes exist in each.)"""
    from .. import materials
    if nlg:
        return
    for name, group in model.element_groups():
        for sl, mat, prop in group.state["slices"]:
            if materials.needs_defgrad(mat):
                raise NotImplementedError(
                    f"/MAT/LAW{mat.law}/{mat.id} (total-form hyperelastic) "
                    f"under the implicit solver requires /IMPL/NONLIN — "
                    f"the frozen small-displacement frame never feeds the "
                    f"trial displacement into F (PORTING_GUIDE M14).")


def _resolve_imposed(model, log):
    """Resolve /IMPDISP into (index, dof, funct, scale, x0-value) tuples and a
    (numnod, 6) prescribed-DOF mask for the equation numbering."""
    n = model.numnod
    imposed = []
    presc = np.zeros((n, 6), dtype=bool)
    for i in model.impdisp:
        grp = model.node_groups.get(i.grnod_id)
        if grp is None or grp.node_idx is None:
            continue
        idx = grp.node_idx
        imposed.append((idx, i.dof, model.functions[i.funct_id], i.scale))
        presc[idx, i.dof] = True
    if imposed:
        log.info(f" IMPLICIT /IMPDISP PRESCRIBED DOFS . . : "
                 f"{int(presc.sum())}")
    return imposed, presc


def _solve_increment(model, controls, log, dof, loads, solver,
                     committed, x_ref, lam_prev, lam, imposed, nlgeom=False,
                     constr=None, contacts=()):
    """One load increment: Newton-iterate to equilibrium at load factor
    ``lam``. Returns ``(IncrementResult, u, ur)`` — the converged increment
    displacements are what the NLGEOM commit placement needs. On entry the
    element buffers hold the committed state and ``x_ref`` the committed
    geometry; on a converged return model.x and the element state hold the
    new equilibrium.

    ``nlgeom`` selects the M9 nonlinear-geometry increment: residual on the
    trial configuration (see ``_internal_forces``) and tangent with K_geo,
    both linearized at the trial geometry x_ref + u.

    M12: ``constr`` reduces the linear system through the condensation
    transform (K_red = T^T K T, R_red = T^T R, du = T du_red — the
    *_IMP1/*_IMPR1 calls of the original around every solve; convergence
    is measured on the REDUCED residual, the only one that must vanish:
    a dependent row's out-of-balance is by construction carried by its
    masters). ``contacts`` add the penalty force at the TRIAL
    configuration model.x + u to the residual and the active-set gap
    tangent to K — the active set is re-evaluated every iteration."""
    ip = controls
    n = model.numnod

    def _reduce(vec):
        return constr.reduce_vector(vec) if constr is not None else vec

    # M13: under NONLINEAR geometry a /PLOAD is a FOLLOWER load — the
    # residual must evaluate it at the TRIAL configuration (model.x + u,
    # like contact) and the tangent gains the load-stiffness term
    # -d f_ext/d x (implicit/followerload.py). The small-displacement path
    # keeps the dead committed-frame pressure (byte-identical to M8).
    from .followerload import has_follower, pload_tangent
    follower = nlgeom and has_follower(loads)
    lstiff = follower and bool(getattr(controls, "impl_load_stiff", True))

    # external force at this load factor (the loads machinery evaluates the
    # curves at t = lam — the load factor plays the role of the pseudo-time)
    fext = np.zeros((n, 3))
    loads.external_forces(lam, fext, x_ref)
    fext_eq = _reduce(dof.gather_residual(fext, np.zeros((n, 3))))
    ref = max(np.linalg.norm(fext_eq), 1e-30)

    # plastic strain at the start of the increment (per solid group) for the
    # consistent-tangent plastic increment Δεp
    epsp0 = {name: committed[name].get("epsp")
             for name, _ in model.element_groups()}

    u = np.zeros((n, 3))       # translation increment of this load step
    ur = np.zeros((n, 3))      # rotation increment of this load step
    # seed the prescribed-DOF increments: the imposed displacement steps from
    # its committed value D(lam_prev) to D(lam); Newton never touches these
    # DOFs (they are condensed), so f_int feels the imposed motion while the
    # residual balances only the free DOFs (displacement control).
    for idx, d, fct, scale in imposed:
        u[idx, d] = scale * (fct.eval(lam) - fct.eval(lam_prev))
        # a nonzero applied displacement makes the load reference the
        # reaction it induces, so the residual norm is measured relative to
        # the internal force as well (see ``ref`` update after iter 0)
    inc = IncrementResult(load_factor=lam, converged=False, iterations=0)

    def _residual(ut, urt):
        """Residual at trial increment (ut, urt): internal force from the
        committed base + the M12 contact force at the trial CONFIGURATION
        (model.x + u — contact is geometric in both element modes). With a
        follower /PLOAD under NLGEOM (M13) the whole external force is
        re-evaluated at the trial configuration too (only the pressure
        actually depends on it — gravity//CLOAD are dead loads)."""
        fint, mint = _internal_forces(model, x_ref, ut, urt, committed,
                                      nlgeom)
        if contacts:
            from .contact import contact_forces
            fcont, _ = contact_forces(contacts, model.x + ut, n)
            fint = fint + fcont
        fx = fext
        if follower:
            fx = np.zeros((n, 3))
            loads.external_forces(lam, fx, model.x + ut)
        return fint, mint, _reduce(dof.gather_residual(fx + fint, mint))

    # first residual R = f_ext + f_int(u) (+ contact), reduced eqn space
    fint, mint, R = _residual(u, ur)
    rnorm = float(np.linalg.norm(R))
    for it in range(ip.impl_max_iter):
        inc.residuals.append(rnorm)
        inc.iterations = it + 1
        # a NON-FINITE residual can never converge — fail the increment NOW
        # so the StepControl cuts it (found by the M15 perfectly-plastic
        # beam: a Newton walk along an H = 0 plateau overflowed u, unorm
        # became inf and the RELATIVE displacement test below compared
        # against tol*inf — accepting a NaN state as "converged")
        if not np.isfinite(rnorm):
            break
        if it == 0:
            # the residual reference: the larger of the applied load and the
            # INITIAL out-of-balance (the latter carries the reaction scale of
            # a displacement-controlled increment, where the load is zero)
            ref = max(ref, rnorm)

        # convergence: residual small relative to the load / reaction scale
        if rnorm <= ip.impl_tol * ref:
            inc.converged = True
            break

        # tangent: plastic-strain increment of this step (solids, shell
        # layers, trusses), used by the LAW2 consistent tangents; elastic
        # groups ignore it
        epsp_incr = _epsp_increments(model, epsp0)
        # linearize where the residual lives: the committed frame for the
        # small-strain path, the TRIAL configuration (with the geometric
        # stiffness added) for the nonlinear-geometry path
        x_tan = x_ref + u if nlgeom else x_ref
        K = assemble(model, dof, x_tan, epsp_incr, kgeo=nlgeom)
        if contacts:
            # active-set gap tangent at the trial configuration (M12 —
            # the IMP_INT_K assembly step)
            from .contact import contact_tangent
            K = K + contact_tangent(contacts, model.x + u, dof)
        if lstiff:
            # M13: follower-pressure load stiffness -d f_ext/d x at the
            # trial configuration (imp_glob_k.F IMP_KPRES analogue — see
            # followerload.py for the documented deviation)
            K = K + pload_tangent(loads, model, lam, model.x + u, dof)
        try:
            if constr is not None:
                du_eq = constr.expand(
                    solver.solve(constr.reduce_matrix(K), R))
            else:
                du_eq = solver.solve(K, R)
        except RuntimeError:
            # an EXACTLY singular trial tangent (e.g. a perfectly-plastic
            # H = 0 state where a too-large increment spuriously yields
            # enough elements to form a mechanism — the M15 beam-hinge
            # lesson): fail the increment and let the StepControl cut it;
            # smaller increments keep the intermediate states regular.
            break
        du, dur = dof.scatter_solution(du_eq)

        # ---- backtracking line search (M13 — the ILINE branch of
        # imp_solv.F, IMCONV = -1): the FULL Newton step is accepted
        # whenever it does not grow the residual norm, so every monotone
        # (smooth) run is bit-identical to the plain Newton path. A step
        # that GROWS the residual — the signature of the non-smooth
        # assignment cycles a contact active set or a friction stick/slip
        # boundary can fall into (a mixed stick-slip state with pairs
        # parked exactly on the Coulomb cone cycled with period 2 before
        # this) — is backtracked by halving, keeping the best trial.
        alpha, best, alpha_last = 1.0, None, 1.0
        for ls in range(4):
            alpha_last = alpha
            fint_t, mint_t, R_t = _residual(u + alpha * du,
                                            ur + alpha * dur)
            rn_t = float(np.linalg.norm(R_t))
            if best is None or rn_t < best[0]:
                best = (rn_t, alpha, fint_t, mint_t, R_t)
            if rn_t <= rnorm or ls == 3:
                break
            alpha *= 0.5
        rnorm, alpha, fint, mint, R = best
        u = u + alpha * du
        ur = ur + alpha * dur
        if alpha != alpha_last:
            # the element buffers must hold the ACCEPTED trial state: the
            # kept trial was not the last one evaluated — re-evaluate there
            # (a rare path: only when every backtrack failed to improve)
            fint, mint, R = _residual(u, ur)

        # displacement convergence: negligible correction relative to the
        # accumulated increment (catches a converged step whose residual
        # reference is tiny, e.g. a pure displacement-controlled increment)
        unorm = np.linalg.norm(dof.gather_residual(u, ur))
        if np.isfinite(unorm) \
                and alpha * np.linalg.norm(du_eq) \
                <= ip.impl_tol * max(unorm, 1e-30) and it > 0:
            inc.residuals.append(rnorm)
            inc.iterations = it + 2
            inc.converged = True
            break

    if inc.converged:
        # accumulate the increment displacement into the total (deformed)
        # geometry. Small-strain path: model.x is OUTPUT only (x_ref stays
        # at x0). Nonlinear geometry: the caller re-bases x_ref on model.x
        # (updated Lagrangian). The element state already holds the converged
        # trial values from the last _internal_forces call.
        model.x = model.x + u
    return inc, u, ur


# ----------------------------------------------------------------------------
# Arc-length (Riks / Crisfield) continuation — M9 (see the module docstring)
# ----------------------------------------------------------------------------

def _run_arclength(model, controls, log, dof, loads, solver, committed,
                   x_ref, imposed, dlam0, lam_end, result, constr=None,
                   contacts=()):
    """Trace the equilibrium path with the cylindrical (Crisfield) arc-length
    method until the load factor reaches ``lam_end`` — THROUGH limit points,
    where the load factor is free to decrease. Appends to ``result`` and
    leaves model.x / the element state at the final equilibrium.

    The constraint radius dl starts from the first predictor at the
    /IMPL/DTINI load increment (or the /IMPL/ARCL card value), adapts to the
    iteration count of each increment (targeting ``impl_arc_itdes``
    iterations) and halves whenever an increment fails (no convergence, or
    complex roots of the constraint quadratic).

    M14 — constraints and contact in the arc corrector (the M12 refusal
    removed). CONSTRAINTS: both auxiliary solves (K du_bar = R, K du_t = q)
    run on the REDUCED system K_red = T^T K T, and the whole continuation —
    the increment Delta_u, the spherical metric ||Delta_u||^2 + w Dlam^2,
    the Crisfield root-selection dot products and the predictor-sign rule —
    lives in the REDUCED coordinates. The ORIGINAL measures its arc metric
    differently (verified in the source): its Riks control (imp_dt.F IDTC=3
    + the BFAC load rescaling of imp_solv.F) takes UL2 = |Delta u|^2 over
    the FULL nodal field via PRODUT_UHP0 — i.e. AFTER the RBY_IMPR2-style
    recovery, so a rigid body's slaves each re-count their master's motion.
    The port deliberately uses the reduced metric instead: the constraint
    quadratic must live in the same space as the two auxiliary solves it
    combines, and the two metrics differ only by the fixed SPD reweighting
    T^T T — both are members of Crisfield's psi-scaled family of arc
    parametrizations (documented deviation, same spirit as M13's
    IMP_KPRES). Under NLGEOM the transform is REBUILT at every committed
    frame and the exact commit placement runs, exactly like load control;
    the previous-increment direction used by the predictor sign rule is
    kept across the rebuild (the reduced coordinates keep their meaning —
    same independent equations, re-linearized arms).
    CONTACT: the penalty force joins the residual at the trial
    configuration and the active-set tangent joins BOTH auxiliary solves'
    matrix, with the active set re-evaluated every corrector iteration
    (the residual is continuous across activation, so the corrector needs
    no special bookkeeping); friction anchors commit per converged arc
    increment exactly like load control. The corrector keeps PLAIN Newton
    steps (no M13 line search): its non-smooth failure mode — an active
    set that will not settle at the current radius — already lands in the
    radius-halving cut, which is the arc method's own backstop (checked by
    the snap-catch validation: the truss lands on the stop mid-trace and
    the active set changes INSIDE increments without a single cut)."""
    ip = controls
    n = model.numnod
    if imposed:
        raise ValueError(
            "/IMPL/ARCL cannot be combined with /IMPDISP: the arc-length "
            "constraint already controls the step size, and the method "
            "assumes a pure proportional force loading f_ext = lambda*q. "
            "Use load control (/IMPL/NONLIN) for prescribed displacements.")

    # ---- proportional load pattern q (f_ext(lam) = lam * q, asserted) -----
    q_full = np.zeros((n, 3))
    loads.external_forces(lam_end, q_full, x_ref)
    q_full /= lam_end
    probe = np.zeros((n, 3))
    loads.external_forces(0.5 * lam_end, probe, x_ref)
    scale = max(float(np.abs(q_full).max()), 1e-30)
    if not np.allclose(probe, 0.5 * lam_end * q_full, atol=1e-9 * scale):
        raise ValueError(
            "/IMPL/ARCL requires a PROPORTIONAL load history "
            "(f_ext(lambda) = lambda * q): make every load /FUNCT a linear "
            "ramp through the origin over the load-factor range.")
    q_eq = dof.gather_residual(q_full, np.zeros((n, 3)))
    if constr is not None:
        # M14: the load pattern, like every residual, lives in the
        # REDUCED space (T^T q — the *_IMPR1 condensation)
        q_eq = constr.reduce_vector(q_eq)
    qnorm = float(np.linalg.norm(q_eq))
    if qnorm <= 0.0:
        raise ValueError("/IMPL/ARCL: the load pattern is empty (no applied "
                         "force on any free DOF).")

    # ---- spherical metric weight + initial radius --------------------------
    # w_lam = ||K0^-1 q||^2 (the initial tangential displacement per unit
    # load factor, squared) makes the Delta_lambda^2 term of the constraint
    # commensurate with ||Delta_u||^2 — see the module docstring. The
    # radius dl is then set so the FIRST increment is exactly the
    # /IMPL/DTINI load increment on the (still linear) path. M14: reduced
    # system + the committed-configuration contact tangent.
    K0 = _arc_tangent(model, dof, x_ref, None, constr, contacts, model.x)
    duT0 = solver.solve(K0, q_eq)
    wlam = float(duT0 @ duT0)
    dl = ip.impl_arc_dl if getattr(ip, "impl_arc_dl", 0.0) > 0.0 \
        else abs(dlam0) * np.sqrt(2.0 * wlam)
    itdes = max(1, int(getattr(ip, "impl_arc_itdes", 5)))
    maxinc = max(1, int(getattr(ip, "impl_arc_maxinc", 200)))
    log.info(f" ARC-LENGTH (RIKS/CRISFIELD) RADIUS . : {dl:12.5E}")

    lam = 0.0
    dir_prev = None      # previous converged Delta_u (equation space)
    cuts = 0
    while lam < lam_end * (1.0 - 1e-12):
        if len(result.increments) >= maxinc:
            result.converged = False
            result.stop_reason = (
                f"ARC-LENGTH REACHED THE INCREMENT CAP ({maxinc}) AT LOAD "
                f"FACTOR {lam:.4E} (< {lam_end:.4E}) — raise the cap on "
                f"/IMPL/ARCL or check for a runaway path")
            return
        inc, Du_eq, u_c, ur_c = _solve_increment_arc(
            model, ip, dof, solver, committed, x_ref, lam, dl, q_full,
            q_eq, wlam, dir_prev, constr, contacts)
        result.increments.append(inc)
        inc_no = len(result.increments)
        if not inc.converged:
            cuts += 1
            log.info(f" {inc_no:9d} {inc.load_factor:13.5E} "
                     f"{inc.iterations:6d} "
                     f"{(inc.residuals[-1] if inc.residuals else 0.0):14.5E}"
                     f"   *** CUT (dl/2)")
            if cuts > 8:
                result.converged = False
                result.stop_reason = (
                    f"ARC-LENGTH FAILED AFTER 8 RADIUS CUTS AT LOAD FACTOR "
                    f"{lam:.4E} (||R|| = "
                    f"{inc.residuals[-1] if inc.residuals else 0.0:.4E})")
                return
            dl *= 0.5
            continue
        log.info(f" {inc_no:9d} {inc.load_factor:13.5E} {inc.iterations:6d} "
                 f"{inc.residuals[-1]:14.5E}   converged (arc)")
        # commit exactly like load control, PLUS the updated-Lagrangian
        # frame advance and the path direction for the next predictor
        committed = {name: _snapshot(g) for name, g in model.element_groups()}
        if contacts:
            # M14: friction anchors re-based per converged arc increment
            from .contact import commit_contacts
            commit_contacts(contacts, model.x)
        if constr is not None:
            # exact dependent-node placement, then re-linearize T at the
            # advanced frame (same order as load control)
            constr.commit_placement(model, u_c, ur_c)
        x_ref = model.x.copy()
        if constr is not None:
            constr.build(dof, x_ref)
        lam = inc.load_factor
        dir_prev = Du_eq
        cuts = 0
        # adapt the radius toward the target iteration count
        dl *= min(2.0, max(0.5, np.sqrt(itdes / max(inc.iterations, 1))))

    # ---- land exactly on lam_end -------------------------------------------
    # the arc trace rarely stops exactly at the final load factor; finish
    # with one plain load-controlled Newton step (tiny, from the committed
    # near-final state) so the reported state is at exactly lam_end.
    if abs(lam - lam_end) > 1e-12 * max(abs(lam_end), 1.0):
        loads_zero_imposed = []
        inc, u_c, ur_c = _solve_increment(
            model, ip, log, dof, _ArcLoads(q_full), solver, committed,
            x_ref, lam, lam_end, loads_zero_imposed, nlgeom=True,
            constr=constr, contacts=contacts)
        if inc.converged:
            # the landing step commits like any load-control increment
            if contacts:
                from .contact import commit_contacts
                commit_contacts(contacts, model.x)
            if constr is not None:
                constr.commit_placement(model, u_c, ur_c)
        result.increments.append(inc)
        if inc.converged:
            log.info(f" {len(result.increments):9d} {lam_end:13.5E} "
                     f"{inc.iterations:6d} {inc.residuals[-1]:14.5E}"
                     f"   converged (final)")
        else:
            result.converged = False
            result.stop_reason = (
                f"FINAL LOAD-CONTROLLED STEP TO {lam_end:.4E} DID NOT "
                f"CONVERGE AFTER THE ARC TRACE")


class _ArcLoads:
    """Minimal stand-in for LoadsAndConstraints inside the arc driver's final
    load-controlled step: the load is exactly lambda * q by the proportional
    assumption already asserted, so re-walking the /FUNCT curves is not
    needed (and the pattern was sampled at the committed reference frame)."""

    def __init__(self, q_full):
        self.q_full = q_full

    def external_forces(self, lam, fext, x_ref):
        fext += lam * self.q_full


def _arc_tangent(model, dof, x_lin, epsp_incr, constr, contacts, x_cont):
    """The arc corrector's system matrix: element tangent (+ K_geo) at
    ``x_lin``, the M14 contact active-set tangent at the trial
    configuration ``x_cont``, condensed to the reduced space when a
    constraint transform is live — shared by the predictor, the metric
    weight and both corrector auxiliary solves."""
    K = assemble(model, dof, x_lin, epsp_incr, kgeo=True)
    if contacts:
        from .contact import contact_tangent
        K = K + contact_tangent(contacts, x_cont, dof)
    if constr is not None:
        K = constr.reduce_matrix(K)
    return K


def _solve_increment_arc(model, ip, dof, solver, committed, x_ref, lam, dl,
                         q_full, q_eq, wlam, dir_prev, constr=None,
                         contacts=()):
    """One arc-length increment (predictor + Crisfield correctors) from the
    committed state at load factor ``lam``, with constraint radius ``dl``
    and spherical load-metric weight ``wlam`` (see _run_arclength).

    Returns ``(inc, Du_eq, u, ur)``: the increment record
    (``inc.load_factor`` is the CONVERGED load factor — it may be smaller
    than ``lam``: that is the method working, not an error), the converged
    displacement increment in (reduced) equation space (the next
    predictor's direction) and the nodal increment fields (what the M14
    commit placement needs). On failure (no convergence, or complex roots
    of the constraint quadratic) the caller restores nothing — the element
    buffers are re-based from ``committed`` at every residual/tangent
    evaluation.

    M14: with a constraint transform, ``q_eq`` and every solve/metric here
    are REDUCED (see _run_arclength); the nodal fields come back through
    the T-expansion, so the dependent nodes ride their masters exactly as
    in load control. Contact forces join the residual at the trial
    configuration model.x + u; the active set re-evaluates per iteration."""
    n = model.numnod

    def _scatter(Du_red):
        du_eq = constr.expand(Du_red) if constr is not None else Du_red
        return dof.scatter_solution(du_eq)

    def _reduce(vec):
        return constr.reduce_vector(vec) if constr is not None else vec

    # ---- predictor: tangential step of length dl in the (u, lam) metric ----
    for name, group in model.element_groups():
        _restore(group, committed[name])
    K0 = _arc_tangent(model, dof, x_ref, None, constr, contacts, model.x)
    duT = solver.solve(K0, q_eq)
    # continue along the previous increment's direction (the standard
    # predictor-sign rule: it flips exactly where the path folds back).
    # dir_prev carries the previous (Delta_u, Delta_lambda) so the metric
    # dot product stays meaningful on near-vertical path segments.
    sign = 1.0
    if dir_prev is not None:
        Du_prev, dlam_prev = dir_prev
        if float(duT @ Du_prev) + wlam * dlam_prev < 0.0:
            sign = -1.0
    dlam = sign * dl / np.sqrt(float(duT @ duT) + wlam)
    Du = dlam * duT                      # (reduced) equation-space increment
    dlam_tot = dlam                      # accumulated lambda increment
    lam_t = lam + dlam
    dlam_pred = abs(dlam)
    u, ur = _scatter(Du)

    qnorm = float(np.linalg.norm(q_eq))
    epsp0 = {name: committed[name].get("epsp")
             for name, _ in model.element_groups()}
    inc = IncrementResult(load_factor=lam_t, converged=False, iterations=0)

    for it in range(ip.impl_max_iter):
        fint, mint = _internal_forces(model, x_ref, u, ur, committed,
                                      nlgeom=True)
        if contacts:
            from .contact import contact_forces
            fcont, _ = contact_forces(contacts, model.x + u, n)
            fint = fint + fcont
        R = _reduce(dof.gather_residual(lam_t * q_full + fint, mint))
        rnorm = float(np.linalg.norm(R))
        inc.residuals.append(rnorm)
        inc.iterations = it + 1
        inc.load_factor = lam_t
        # reference: the load level actually applied (never below the
        # predictor's own step, so a near-zero crossing of lambda cannot
        # make the tolerance impossible)
        ref = max(qnorm * max(abs(lam_t), dlam_pred), 1e-30)
        if rnorm <= ip.impl_tol * ref:
            inc.converged = True
            break

        epsp_incr = _epsp_increments(model, epsp0)
        K = _arc_tangent(model, dof, x_ref + u, epsp_incr, constr,
                         contacts, model.x + u)
        du_bar = solver.solve(K, R)
        du_t = solver.solve(K, q_eq)

        # constraint quadratic (spherical metric, see module docstring):
        # ||Du + du_bar + r*du_t||^2 + wlam*(dlam_tot + r)^2 = dl^2
        #   ->  a*r^2 + b*r + c = 0
        t = Du + du_bar
        a = float(du_t @ du_t) + wlam
        b = 2.0 * (float(du_t @ t) + wlam * dlam_tot)
        c = float(t @ t) + wlam * dlam_tot * dlam_tot - dl * dl
        disc = b * b - 4.0 * a * c
        if disc < 0.0 or a <= 0.0:
            # the corrector left the constraint sphere with no real
            # intersection: the radius is too large for this part of the
            # path — report failure, the caller halves dl and retries
            inc.converged = False
            return inc, (Du, dlam_tot), u, ur
        sq = np.sqrt(disc)
        r1 = (-b + sq) / (2.0 * a)
        r2 = (-b - sq) / (2.0 * a)
        # keep the root that continues forward along the current increment
        # (maximizes the METRIC dot with (Du, dlam_tot) — Crisfield's
        # criterion; the other root would double back along the traced path)
        def _fwd(r):
            return float(Du @ (t + r * du_t)) \
                + wlam * dlam_tot * (dlam_tot + r)
        pick = r1 if _fwd(r1) >= _fwd(r2) else r2
        Du = t + pick * du_t
        dlam_tot += pick
        lam_t += pick
        u, ur = _scatter(Du)

    if inc.converged:
        model.x = model.x + u
    return inc, (Du, dlam_tot), u, ur


def _solver_banner(ip, log):
    from .linsolve import linsolve_name
    if ip.impl_linsolve:
        from .linsolve import select_linsolve
        return select_linsolve(ip.impl_linsolve, log)
    return linsolve_name(log)


def _final_summary(model, result, dof, log):
    """Termination page, mirroring the explicit engine's final summary."""
    real = model.mass < 1e29
    disp = model.x - model.x0
    umax = float(np.abs(disp[real]).max()) if real.any() else 0.0
    total_it = sum(i.iterations for i in result.increments)
    log.info("\n     ------------------------------------------------")
    if result.converged:
        log.info("     IMPLICIT TERMINATION : NORMAL")
    else:
        log.info(f"     IMPLICIT TERMINATION : ERROR — {result.stop_reason}")
    log.info(f"     LOAD INCREMENTS . . . . . : {len(result.increments)}")
    log.info(f"     TOTAL NEWTON ITERATIONS . : {total_it}")
    log.info(f"     FREE DOFS . . . . . . . . : {dof.ndof}")
    log.info(f"     MAX |DISPLACEMENT|  . . . : {umax:14.7E}")
    log.info("     ------------------------------------------------")
