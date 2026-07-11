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

What is DEFERRED (documented, not half-done — see the package docstring and
PORTING_GUIDE): geometric/initial-stress stiffness (large displacement),
implicit dynamics (Newmark/HHT), contact & constraints in the tangent, and
arc-length continuation. Load control only, small-strain linear geometry.
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


# ----------------------------------------------------------------------------
# Residual (internal force) evaluation — reuses the explicit kernels
# ----------------------------------------------------------------------------

def _internal_forces(model, x_ref, u, ur, committed):
    """Internal force/moment at trial increment (u, ur) from the committed
    base state. Restores the element buffers and calls the EXISTING force
    kernels with the displacement increment as a pseudo-velocity at dt = 1
    (see module docstring), evaluated at the COMMITTED geometry ``x_ref``.

    Evaluating at the committed (reference) geometry — not ``x_ref + u`` — is
    what makes M8 a *small-strain linear-geometry* analysis: the strain
    increment is grad(u) on the reference frame, so a linear-elastic step is
    exactly K u and Newton converges in one iteration (Hooke's law reproduced
    exactly). The reference frame is advanced to the deformed geometry only
    when an increment COMMITS (an updated-Lagrangian outer step), which lets a
    moderate-rotation problem accumulate over several increments while each
    increment stays linear. Large-displacement geometric stiffness WITHIN an
    increment is the deferred sub-step (see the package docstring).

    Restores + calls in place; the element state is left at the trial values
    (the caller commits or re-evaluates from ``committed``)."""
    n = model.numnod
    for name, group in model.element_groups():
        _restore(group, committed[name])
    fint = np.zeros((n, 3))
    mint = np.zeros((n, 3))
    for name, group in model.element_groups():
        KERNELS[name].forces(group, x_ref, u, ur, 1.0, fint, mint)
        # some kernels add a STATIC stabilization the (dynamics-tuned) force
        # path cannot supply — the solid stiffness-hourglass, whose explicit
        # form is viscous and far too weak to control hourglass in statics
        # (see solid_hexa8.static_stabilization). Consistent with tangent().
        stab = getattr(KERNELS[name], "static_stabilization", None)
        if stab is not None:
            stab(group, x_ref, u, ur, fint, mint)
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

    log.info("\n     IMPLICIT STATIC ANALYSIS (M8)")
    log.info("     -----------------------------")

    # fail fast on un-ported element types (rather than mid-Newton): the M8
    # implicit tangent covers only the 8-node solid and the 4-node shell
    from .assembly import _TANGENT_KERNELS
    unsupported = [name for name, _ in model.element_groups()
                   if name not in _TANGENT_KERNELS]
    if unsupported:
        raise NotImplementedError(
            f"the M8 implicit solver has no tangent for element group(s) "
            f"{unsupported} (supported: {list(_TANGENT_KERNELS)}). Remove "
            f"them or run the explicit solver.")

    log.info(f" LINEAR SOLVER  . . . . . . . . . . . : "
             f"{_solver_banner(ip, log)}")

    # /IMPDISP prescribed displacements: known DOFs whose value ramps with the
    # load factor. They are condensed out of the equations (like /BCS) but
    # carried in the displacement vector so f_int feels them — the standard
    # implicit displacement-control treatment (see _solve_increment). /IMPVEL
    # is meaningless for statics and is ignored here.
    imposed, presc = _resolve_imposed(model, log)
    dof = DofMap(model, log, prescribed=presc)
    loads = LoadsAndConstraints(model, log)
    solver = LinearSolver(ip.impl_linsolve, log)

    # committed element state (base for each increment's residual evals) and
    # committed plastic strain (for the consistent-tangent increment)
    committed = {name: _snapshot(g) for name, g in model.element_groups()}
    # SMALL-STRAIN LINEAR GEOMETRY (M8): the reference frame stays at the
    # initial configuration x0 for the whole run — every residual and tangent
    # is linearized there. model.x accumulates the total displacement for
    # output only; it is never fed back as geometry (that would be an
    # updated-Lagrangian / geometric-nonlinear scheme — the deferred sub-step).
    x_ref = model.x0.copy()
    model.x = model.x0.copy()

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

    inc_no = 0
    while lam < lam_end * (1.0 - 1e-12):
        lam_new = min(lam + dlam, lam_end)
        inc_no += 1
        inc = _solve_increment(model, controls, log, dof, loads, solver,
                               committed, x_ref, lam, lam_new, imposed)
        result.increments.append(inc)
        if not inc.converged:
            result.converged = False
            result.stop_reason = (
                f"NEWTON DID NOT CONVERGE AT LOAD FACTOR {lam_new:.4E} "
                f"IN {ip.impl_max_iter} ITERATIONS "
                f"(||R|| = {inc.residuals[-1]:.4E})")
            log.info(f" {inc_no:9d} {lam_new:13.5E} {inc.iterations:6d} "
                     f"{inc.residuals[-1]:14.5E}   *** NO CONVERGENCE")
            break
        log.info(f" {inc_no:9d} {lam_new:13.5E} {inc.iterations:6d} "
                 f"{inc.residuals[-1]:14.5E}   converged")
        # commit: the element state already holds the converged step; rebase
        # the committed buffers for the next increment. x_ref stays at x0.
        committed = {name: _snapshot(g) for name, g in model.element_groups()}
        lam = lam_new

    model.implicit_result = result
    _final_summary(model, result, dof, log)
    return model


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
                     committed, x_ref, lam_prev, lam, imposed):
    """One load increment: Newton-iterate to equilibrium at load factor
    ``lam``. Returns an IncrementResult. On entry the element buffers hold the
    committed state and ``x_ref`` the committed geometry; on a converged
    return model.x and the element state hold the new equilibrium."""
    ip = controls
    n = model.numnod

    # external force at this load factor (the loads machinery evaluates the
    # curves at t = lam — the load factor plays the role of the pseudo-time)
    fext = np.zeros((n, 3))
    loads.external_forces(lam, fext, x_ref)
    fext_eq = dof.gather_residual(fext, np.zeros((n, 3)))
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

    for it in range(ip.impl_max_iter):
        # residual R = f_ext + f_int(u)  (equation space)
        fint, mint = _internal_forces(model, x_ref, u, ur, committed)
        R = dof.gather_residual(fext + fint, mint)
        rnorm = float(np.linalg.norm(R))
        inc.residuals.append(rnorm)
        inc.iterations = it + 1
        if it == 0:
            # the residual reference: the larger of the applied load and the
            # INITIAL out-of-balance (the latter carries the reaction scale of
            # a displacement-controlled increment, where the load is zero)
            ref = max(ref, rnorm)

        # convergence: residual small relative to the load / reaction scale
        if rnorm <= ip.impl_tol * ref:
            inc.converged = True
            break

        # tangent: plastic-strain increment of this step (solids), used by
        # the LAW2 consistent tangent; elastic groups ignore it
        epsp_incr = {}
        for name, group in model.element_groups():
            if name == "bricks" and epsp0[name] is not None:
                epsp_incr[name] = group.state["epsp"] - epsp0[name]
        K = assemble(model, dof, x_ref, epsp_incr)
        du_eq = solver.solve(K, R)
        du, dur = dof.scatter_solution(du_eq)
        u = u + du
        ur = ur + dur

        # displacement convergence: negligible correction relative to the
        # accumulated increment (catches a converged step whose residual
        # reference is tiny, e.g. a pure displacement-controlled increment)
        unorm = np.linalg.norm(dof.gather_residual(u, ur))
        if np.linalg.norm(du_eq) <= ip.impl_tol * max(unorm, 1e-30) \
                and it > 0:
            # re-evaluate the residual at the corrected u for the record
            fint, mint = _internal_forces(model, x_ref, u, ur, committed)
            R = dof.gather_residual(fext + fint, mint)
            inc.residuals.append(float(np.linalg.norm(R)))
            inc.iterations = it + 2
            inc.converged = True
            break

    if inc.converged:
        # accumulate the increment displacement into the total (deformed)
        # geometry for OUTPUT. The reference frame x_ref stays at x0 — this
        # is a small-strain analysis, so model.x is never used as geometry.
        # The element state already holds the converged trial values from the
        # last _internal_forces call.
        model.x = model.x + u
    return inc


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
