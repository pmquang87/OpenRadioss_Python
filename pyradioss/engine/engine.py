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
from typing import List, Optional

import numpy as np

from .. import banner
from ..common.constants import EP30
from ..common.messages import MessageLog
from ..contact import ContactType7
from ..elements import KERNELS
from ..input.deck_reader import read_deck
from ..input.engine_keywords import parse_engine_deck
from ..model.model import EngineControls, Model
from ..output import TimeHistory, write_anim_state
from ..starter.restart import read_restart
from .kinematics import LoadsAndConstraints
from .rigid_wall import RigidWalls


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
    ref = max(abs(state.wext), ke, ie, 1e-12)
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
    contacts: List[ContactType7] = [
        ContactType7(itf, model, log) for itf in model.interfaces]
    th = TimeHistory(os.path.join(out_dir, f"{run_name}T01.csv"), model, log)

    fint = np.zeros((n, 3))    # -internal forces (see elements pkg doc)
    mint = np.zeros((n, 3))    # -internal moments (shell rotations)
    fext = np.zeros((n, 3))
    inv_mass = 1.0 / model.mass
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
        for ct in contacts:
            dwork, dt_i = ct.forces(model.x, model.v, model.mass, dt,
                                    fint, state.cycle)
            state.econt += dwork
            dt_next = min(dt_next, dt_i)

        # ---- 3. external loads (gravity, /CLOAD) --------------------------
        fext[:] = 0.0
        loads.external_forces(state.t, fext)

        # ---- 4. acceleration + velocity update (leap-frog) ----------------
        v_old = model.v.copy() if walls.walls else model.v  # for wall energy
        acc = (fint + fext) * inv_mass[:, None]
        model.v += acc * dt
        model.vr += mint * inv_inertia[:, None] * dt

        # ---- 5. kinematic conditions overwrite velocities -----------------
        state.wext += loads.apply_kinematic(state.t + dt, model.v, model.vr,
                                            model.mass)
        state.econt += walls.apply(model.x, model.v, v_old, model.mass, dt)

        # external work of the loads: force x actual displacement, i.e. the
        # POST-enforcement midstep velocity (f^n does its work over
        # x^{n+1}-x^n = v^{n+1/2} dt — consistent leap-frog bookkeeping)
        state.wext += float(np.einsum("nb,nb->", fext, model.v)) * dt

        # ---- 6. position update -------------------------------------------
        model.x += model.v * dt

        state.t += dt
        state.cycle += 1

        # ---- 7. outputs ----------------------------------------------------
        if state.t >= next_th and controls.th_dt > 0:
            e = _energies(model, state)
            mom = (model.mass[real, None] * model.v[real]).sum(axis=0)
            th.write(state.t, e, float(model.mass[real].sum()), mom)
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
