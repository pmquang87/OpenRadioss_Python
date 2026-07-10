"""
The Engine main program: the explicit time-integration loop.

Fortran origin: ``engine/source/engine/resol.F`` — the ~5000-line heart of
OpenRadioss. This module is deliberately the best-commented file of the
port: read it top to bottom to understand one complete explicit cycle.

The central-difference scheme
-----------------------------
Explicit dynamics staggers velocities and positions in time (leap-frog):

    known:  x^n  (positions at t_n),  v^{n-1/2}  (velocities at midstep)

    1.  f^n     = f_ext(t_n) - f_int(x^n, state)      forces at t_n
    2.  a^n     = f^n / m                              (lumped mass: trivial
                                                        'solve', the reason
                                                        explicit codes never
                                                        assemble a matrix)
    3.  v^{n+1/2} = v^{n-1/2} + a^n * dt               velocity update
    4.  kinematic conditions overwrite v^{n+1/2}       (BCS, /IMPVEL, walls)
    5.  x^{n+1}   = x^n + v^{n+1/2} * dt               position update
    6.  dt        = k * min(element/interface dt)      next stable step

The scheme is 2nd-order accurate and conditionally stable: dt must resolve
the fastest wave crossing the smallest element — the Courant condition.
Every cycle each element kernel returns its own critical dt and the global
step is the scaled minimum (dt_sca, default 0.9).

Energy balance
--------------
The classic quality metric of an explicit run (printed every /PRINT
cycles, exactly like the original's listing):

    error% = (IE + KE + HE + CE - EW - E0) / max(E0, EW, IE+KE) * 100

with IE internal, KE kinetic, HE hourglass, CE contact energy and EW the
external work (loads + kinematic conditions). A growing |error| signals
instability; the Engine stops beyond ``/STOP``'s threshold (default 15%),
like the original's energy-error check in resol.F.
"""

from __future__ import annotations

import os
import time
from typing import Optional

import numpy as np

from .. import banner
from ..common.constants import EP30
from ..common.messages import MessageLog
from ..contact import build_contacts
from ..elements import KERNELS
from ..input.deck_reader import read_deck
from ..input.engine_keywords import parse_engine_deck
from ..model.model import EngineControls, Model
from ..output import TimeHistory, write_anim_state
from ..starter.restart import read_restart
from .kinematics import LoadsAndConstraints
from .rbe3 import build_rbe3
from .rigid_body import build_rigid_bodies
from .rigid_wall import RigidWalls
from .sections import SectionForces


def run_name_from_input(path: str) -> str:
    """'MYRUN_0001.rad' -> 'MYRUN'."""
    base = os.path.basename(path)
    for suffix in ("_0001.rad", "_0001.RAD"):
        if base.endswith(suffix):
            return base[: -len(suffix)]
    return os.path.splitext(base)[0]


class EngineState:
    """Everything the loop tracks besides the model itself."""

    def __init__(self):
        self.t = 0.0
        self.cycle = 0
        self.wext = 0.0        # accumulated external work
        self.econt = 0.0       # accumulated contact + rigid-wall energy
        self.ndel = 0          # deleted elements reported so far (/FAIL)
        self.epeak = 0.0       # running peak of IE+KE (error reference)
        self.stop_reason = ""


def _deleted_count(model: Model) -> int:
    """Total deleted elements (GBUF%OFF == 0) across all groups."""
    ndel = 0
    for _, group in model.element_groups():
        off = group.state.get("off")
        if off is not None:
            ndel += int((off == 0.0).sum())
    return ndel


def _energies(model: Model, state: EngineState) -> dict:
    ie = he = 0.0
    for _, group in model.element_groups():
        ie += float(group.state["eint"].sum())
        he += float(group.state["ehour"].sum())
    real = model.mass < 1e29
    ke = float(0.5 * (model.mass[real, None] * model.v[real] ** 2).sum())
    total = ie + ke + he + state.econt
    # the error reference is the ENERGY SCALE OF THE RUN: the largest of
    # the initial energy, external work, current energies and the running
    # peak of IE+KE. The initial/peak terms matter for oscillating or
    # fully-arrested systems (a pendulum at its turning point, a block
    # brought to rest on a wall): all instantaneous energies pass through
    # zero there, and normalizing a fixed round-off residual by them
    # would scream divergence where there is none. (The original guards
    # its error the same way, with the initial/reference energy.)
    state.epeak = max(state.epeak, ie + ke)
    ref = max(abs(state.wext), ke, ie, state.epeak, abs(_energies.e0),
              1e-12)
    err = (total - state.wext - _energies.e0) / ref * 100.0
    return {"IE": ie, "KE": ke, "HE": he, "CE": state.econt,
            "EW": state.wext, "ERR": err}


def run_engine(input_file: str, log: Optional[MessageLog] = None) -> Model:
    """Run the Engine on ``RunName_0001.rad`` (+ RunName_0000.rst)."""
    log = log or MessageLog()
    run_name = run_name_from_input(input_file)
    out_dir = os.path.dirname(os.path.abspath(input_file))

    with open(os.path.join(out_dir, f"{run_name}_0001.out"), "w") as listing:
        log.attach_listing(listing)
        log.info(banner())
        log.info(f" ENGINE INPUT FILE  . . . . . . . . . : {input_file}")

        # ---- read controls + restart (engine lectur.F + rdresb.F) --------
        controls = parse_engine_deck(read_deck(input_file), log)
        rst = os.path.join(out_dir, f"{run_name}_0000.rst")
        log.info(f" RESTART FILE . . . . . . . . . . . . : {rst}")
        model = read_restart(rst)
        log.info(f" MODEL TITLE  . . . . . . . . . . . . : {model.title}")
        log.info(f" FINAL TIME (/RUN)  . . . . . . . . . : "
                 f"{controls.t_end:12.5E}")

        model = _integrate(model, controls, log, out_dir, run_name)
    return model


def _integrate(model: Model, controls: EngineControls, log: MessageLog,
               out_dir: str, run_name: str) -> Model:
    state = EngineState()
    n = model.numnod

    # ---- engine-side setup (resol_init) ----------------------------------
    loads = LoadsAndConstraints(model, log)
    walls = RigidWalls(model, log)
    # contact: penalty interfaces (TYPE7/TYPE11, force-based) and tied
    # interfaces (TYPE2, kinematic) hook into the cycle differently
    contacts, tied = build_contacts(model, log)
    # rigid bodies (/RBODY + /RBE2) and interpolation constraints (/RBE3):
    # both kinematic — see engine/rigid_body.py and engine/rbe3.py. The
    # rigid bodies also project the initial nodal velocities onto rigid
    # motion, so they run BEFORE the E0 reference below.
    rbodies = build_rigid_bodies(model, loads, log)
    rbe3s = build_rbe3(model, log)
    sections = SectionForces(model, log)
    th = TimeHistory(os.path.join(out_dir, f"{run_name}T01.csv"), model, log)

    fint = np.zeros((n, 3))    # -internal forces (see elements pkg doc)
    mint = np.zeros((n, 3))    # -internal moments (shell rotations)
    fext = np.zeros((n, 3))
    fcont = np.zeros((n, 3))   # contact forces, kept separate: the contact
    # energy must be booked with the leapfrog-consistent MIDSTEP velocity
    # (see step 5b below), which needs the isolated contact force vector
    # /INTER/TYPE2 mass transfer (i2 init): the EFFECTIVE mass — used for
    # accelerations only — of a tied node's main segment corners includes
    # the secondary mass, M_k += w_k m_s. Same for the /RBE3 dependent
    # node's mass on its masters. Physical masses (energies, momentum,
    # listing) stay in model.mass. See contact/inter_type2.py.
    mass_eff = model.mass.copy()
    for t2 in tied:
        t2.augment_mass(mass_eff)
    for r3 in rbe3s:
        r3.augment_mass(mass_eff)
    # a rigid body whose nodes carry tied secondaries must also carry
    # their inertia in its 6-DOF EOM (see rigid_body.finalize_mass)
    for rb in rbodies:
        rb.finalize_mass(mass_eff)
    inv_mass = 1.0 / mass_eff
    has_inertia = model.inertia > 0.0
    inv_inertia = np.where(has_inertia, 1.0 / np.maximum(model.inertia, 1e-30),
                           0.0)

    # initial kinetic energy = reference E0 of the balance
    real = model.mass < 1e29
    _energies.e0 = float(0.5 * (model.mass[real, None]
                                * model.v[real] ** 2).sum())

    # ---- priming pass: dt=0 'cycle' just to collect the initial critical
    # time step from every kernel (no state advances with dt = 0).
    dt = 0.0
    dt_next = EP30
    for name, group in model.element_groups():
        dt_e = KERNELS[name].forces(group, model.x, model.v, model.vr,
                                    0.0, fint, mint)
        dt_next = min(dt_next, float(dt_e.min()))
    # penalty interfaces bound the step from cycle 0 (their stiffness is
    # static in this port), so an impact on the very first cycles is safe
    for ct in contacts:
        dt_next = min(dt_next, ct.dt_bound)
    fint[:] = 0.0
    mint[:] = 0.0
    dt = controls.dt_scale * dt_next
    log.info(f" INITIAL TIME STEP  . . . . . . . . . : {dt:12.5E}\n")
    log.info("  CYCLE       TIME        TIME-STEP    ENERGY-IE    "
             "ENERGY-KE    ENERGY-HE    ENERGY-CE     EXT-WORK   ERROR%")

    next_th = 0.0
    next_anim = 0.0
    anim_no = 0
    t_wall0 = time.time()

    # ======================================================================
    #                       THE EXPLICIT TIME LOOP
    # ======================================================================
    while state.t < controls.t_end:
        dt = min(dt, controls.t_end - state.t)  # land exactly on t_end

        # ---- 1. internal forces, element by element group ----------------
        fint[:] = 0.0
        mint[:] = 0.0
        dt_next = EP30
        for name, group in model.element_groups():
            dt_e = KERNELS[name].forces(group, model.x, model.v, model.vr,
                                        dt, fint, mint)
            dt_next = min(dt_next, float(dt_e.min()))

        # ---- 2. contact forces -------------------------------------------
        # (into their own array — see the fcont declaration and step 5b;
        # the work increment the interface returns is its own estimate at
        # the pre-update velocities, superseded by the exact booking below)
        fcont[:] = 0.0
        for ct in contacts:
            _, dt_i = ct.forces(model.x, model.v, model.mass, dt,
                                fcont, state.cycle)
            dt_next = min(dt_next, dt_i)

        # ---- 3. external loads (gravity, /CLOAD, /PLOAD) -------------------
        fext[:] = 0.0
        loads.external_forces(state.t, fext, model.x)

        # ---- 3b. tied interfaces (/INTER/TYPE2, i2for3): move the tied
        # nodes' internal + external forces onto their main segments (the
        # constraint carries them; the secondary rows are zeroed). Also
        # polls the deletion release. Does NO work by construction —
        # nothing is booked into econt. /RBE3 distributes its dependent
        # node's forces to the masters the same way (rbe3f).
        for t2 in tied:
            t2.transfer_forces(fint, fext, fcont, mass_eff, inv_mass,
                               state.cycle)
        for r3 in rbe3s:
            r3.transfer_forces(fint, fext, fcont, mint, model.x)

        # ---- 4. acceleration + velocity update (leap-frog) ----------------
        v_old = model.v.copy()     # for wall energy + contact work booking
        acc = (fint + fcont + fext) * inv_mass[:, None]
        model.v += acc * dt
        model.vr += mint * inv_inertia[:, None] * dt

        # ---- 5. kinematic conditions overwrite velocities -----------------
        # 5a. rigid bodies (/RBODY, /RBE2 — rbyfor/rbycor): gather the
        # body forces, advance the 6-DOF EOM, scatter the rigid velocity
        # field. Runs FIRST so everything downstream (walls, the contact
        # work booking, the fext work) sees the velocities the body nodes
        # actually move with. Books only the master /IMPVEL drive work.
        for rb in rbodies:
            state.wext += rb.advance(fint, fext, fcont, mint, model.v,
                                     model.vr, model.x, dt, state.t + dt)
        # (mass_eff: an /IMPVEL driving a tied main node reacts against
        # the secondary inertia it carries too)
        state.wext += loads.apply_kinematic(state.t + dt, model.v, model.vr,
                                            mass_eff, model.x, dt)
        de_wall, dw_wall = walls.apply(model.x, model.v, v_old,
                                       model.mass, dt)
        state.econt += de_wall
        state.wext += dw_wall

        # ---- 5b. contact energy booking (M4 lesson) ------------------------
        # In leap-frog, a force f^n changes the kinetic energy by exactly
        # f . (v^{n-1/2} + v^{n+1/2})/2 * dt — the MIDSTEP average, not
        # either endpoint. Booking contact work at v^{n-1/2} (as a contact
        # kernel alone could) leaves a positive-definite residual
        # f^2 dt^2 / 2m per cycle that reads as spurious energy CREATION
        # whenever penalty springs dominate the energy scale (light nodes,
        # short impacts). Booking from the assembled contact force with
        # the midstep velocity closes the balance to round-off; the
        # remaining CE drift is the real damper/friction dissipation.
        # (The original accumulates interface energies from the same
        # assembled forces in its FSAV blocks.)
        if contacts:
            state.econt -= float(np.einsum(
                "nb,nb->", fcont, 0.5 * (v_old + model.v))) * dt

        # external work of the loads: force x actual displacement, i.e. the
        # POST-enforcement midstep velocity (f^n does its work over
        # x^{n+1}-x^n = v^{n+1/2} dt — consistent leap-frog bookkeeping)
        state.wext += float(np.einsum("nb,nb->", fext, model.v)) * dt

        # ---- 6. position update -------------------------------------------
        model.x += model.v * dt

        # ---- 6b. kinematic placements, dependency order: rigid bodies
        # first (their nodes may carry tied mains / RBE3 masters), then
        # tied interfaces (i2vit3), then RBE3 dependents.
        for rb in rbodies:
            rb.enforce(model.x, model.v, dt)
        for t2 in tied:
            t2.enforce(model.x, model.v, dt)
        for r3 in rbe3s:
            r3.enforce(model.x, model.v, model.vr, dt)

        state.t += dt
        state.cycle += 1

        # ---- 7. outputs ----------------------------------------------------
        if state.t >= next_th and controls.th_dt > 0:
            e = _energies(model, state)
            mom = (model.mass[real, None] * model.v[real]).sum(axis=0)
            # /SECT resultants from this cycle's assembled internal forces
            svals = sections.compute(model.x, fint, mint) if len(sections) \
                else None
            th.write(state.t, e, float(model.mass[real].sum()), mom, svals)
            next_th += controls.th_dt
        if controls.anim_dt > 0 and state.t >= next_anim:
            path = os.path.join(out_dir, f"{run_name}A{anim_no:03d}.vtk")
            write_anim_state(path, model, state.t,
                             controls.anim_vect, controls.anim_elem)
            anim_no += 1
            next_anim += controls.anim_dt
        if state.cycle % controls.print_cycles == 0 or \
                state.t >= controls.t_end:
            # element-deletion report (the original prints a "RUPTURE /
            # DELETE ELEMENT" message per element; the port reports the
            # running total at the listing frequency)
            ndel = _deleted_count(model)
            if ndel > state.ndel:
                log.info(f" -- ELEMENT DELETION: {ndel - state.ndel} "
                         f"ELEMENT(S) DELETED (TOTAL {ndel})")
                state.ndel = ndel
            e = _energies(model, state)
            log.info(f" {state.cycle:6d} {state.t:12.5E} {dt:12.5E} "
                     f"{e['IE']:12.5E} {e['KE']:12.5E} {e['HE']:12.5E} "
                     f"{e['CE']:12.5E} {e['EW']:12.5E} {e['ERR']:8.2f}")
            # ---- divergence guards (the original's energy-error and
            # minimum-time-step stops) --------------------------------
            if abs(e["ERR"]) > controls.energy_error_stop:
                state.stop_reason = (f"ENERGY ERROR {e['ERR']:.1f}% EXCEEDS "
                                     f"LIMIT {controls.energy_error_stop}%")
                break
            if not np.isfinite(e["KE"]):
                state.stop_reason = "NAN/INF DETECTED — RUN DIVERGED"
                break

        # ---- 8. next time step ---------------------------------------------
        dt = controls.dt_scale * dt_next
        if controls.dt_min > 0 and dt < controls.dt_min:
            state.stop_reason = (f"TIME STEP {dt:.3E} BELOW MINIMUM "
                                 f"{controls.dt_min:.3E}")
            break

    # ======================================================================
    # final state + termination summary (like the original's final page)
    if controls.anim_dt > 0:
        path = os.path.join(out_dir, f"{run_name}A{anim_no:03d}.vtk")
        write_anim_state(path, model, state.t,
                         controls.anim_vect, controls.anim_elem)
    th.close()

    e = _energies(model, state)
    log.info("\n     ------------------------------------------------")
    if state.stop_reason:
        log.info(f"     ENGINE TERMINATION : ERROR — {state.stop_reason}")
    else:
        log.info("     ENGINE TERMINATION : NORMAL")
    log.info(f"     FINAL TIME    . . . . . . : {state.t:14.7E}")
    log.info(f"     CYCLES        . . . . . . : {state.cycle}")
    log.info(f"     ENERGY ERROR  . . . . . . : {e['ERR']:8.2f} %")
    log.info(f"     INTERNAL ENERGY . . . . . : {e['IE']:14.7E}")
    log.info(f"     KINETIC ENERGY  . . . . . : {e['KE']:14.7E}")
    log.info(f"     HOURGLASS ENERGY  . . . . : {e['HE']:14.7E}")
    log.info(f"     CONTACT ENERGY  . . . . . : {e['CE']:14.7E}")
    log.info(f"     EXTERNAL WORK . . . . . . : {e['EW']:14.7E}")
    log.info(f"     ELAPSED TIME  . . . . . . : "
             f"{time.time() - t_wall0:10.3f} s")
    log.info("     ------------------------------------------------")
    return model
