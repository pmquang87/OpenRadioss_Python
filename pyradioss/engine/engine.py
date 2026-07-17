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
import re
import time
from typing import Optional

import numpy as np

from .. import banner
from ..accel import backend_name
from ..common.constants import EP30
from ..common.messages import MessageLog
from ..contact import build_contacts
from ..elements import KERNELS
from ..input.deck_reader import read_deck
from ..input.engine_keywords import parse_engine_deck
from ..input.mat_reader import refuse_inactive_materials
from ..input.prop_reader import refuse_inactive_properties
from ..model.model import EngineControls, Model
from ..output import TimeHistory, write_anim_state
from ..starter.restart import read_restart, write_restart
from .damping import Dampers
from .kinematics import LoadsAndConstraints
from .mass_scaling import NodalTimeStep
from .mpc import build_mpc
from .rbe3 import build_rbe3
from .rigid_body import build_rigid_bodies
from .rigid_wall import RigidWalls
from .sections import SectionForces
from .sensors import Sensors


def run_name_from_input(path: str):
    """'MYRUN_0002.rad' -> ('MYRUN', 2) — the run number selects the
    restart to resume from (RunName_{nn-1}.rst) and names the outputs
    (RunName_{nn}.out, RunName T{nn} history, RunName_{nn}.rst)."""
    base = os.path.basename(path)
    m = re.match(r"(.+)_(\d{4})\.rad$", base, re.IGNORECASE)
    if m:
        return m.group(1), int(m.group(2))
    return os.path.splitext(base)[0], 1


class EngineState:
    """Everything the loop tracks besides the model itself."""

    def __init__(self):
        self.t = 0.0
        self.cycle = 0
        self.wext = 0.0        # accumulated external work
        self.econt = 0.0       # accumulated contact + rigid-wall energy
        self.ndel = 0          # deleted elements reported so far (/FAIL)
        self.epeak = 0.0       # running peak of IE+KE (error reference)
        self.e_num = 0.0       # numerical dissipation of the elements (M6)
        self.e_booked_prev = 0.0   # helper for the e_num ledger
        self.e_madd = 0.0      # kinetic energy from /DT/NODA/CST mass (M6)
        self.e_damp = 0.0      # /DAMP dissipation (M6)
        self.stop_reason = ""


def _deleted_count(model: Model) -> int:
    """Total deleted elements (GBUF%OFF == 0) across all groups."""
    ndel = 0
    for _, group in model.element_groups():
        off = group.state.get("off")
        if off is not None:
            ndel += int((off == 0.0).sum())
    return ndel


def _element_energy_sum(model: Model) -> float:
    """Sum of the kernels' state-function bookings (eint + ehour) — the
    reference the numerical-dissipation ledger (see the cycle's step 6c)
    differences against."""
    e = 0.0
    for _, group in model.element_groups():
        e += float(group.state["eint"].sum())
        e += float(group.state["ehour"].sum())
    return e


def _energies(model: Model, state: EngineState) -> dict:
    ie = he = 0.0
    for _, group in model.element_groups():
        ie += float(group.state["eint"].sum())
        he += float(group.state["ehour"].sum())
    real = model.mass < 1e29
    ke = float(0.5 * (model.mass[real, None] * model.v[real] ** 2).sum())
    total = ie + ke + he + state.econt + state.e_num + state.e_damp
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
    # e_madd: kinetic energy CREATED by /DT/NODA/CST mass additions at
    # moving nodes — an energy input like external work, reported by the
    # mass-scaling summary (the honesty contract of engine/mass_scaling)
    err = (total - state.wext - state.e_madd - _energies.e0) / ref * 100.0
    # ERRN: the numerical-dissipation ledger on the same % scale. A
    # diverging run drives EN hard NEGATIVE (the discrete elastic force
    # injects energy the state bookings cannot see) — and because EN is
    # part of the balance, |ERR| alone would stay blind to exactly that
    # divergence; the Engine stops on ERRN < -limit (step 7).
    return {"IE": ie, "KE": ke, "HE": he, "CE": state.econt,
            "EN": state.e_num, "DE": state.e_damp, "EW": state.wext,
            "ERR": err, "ERRN": state.e_num / ref * 100.0}


def run_engine(input_file: str, log: Optional[MessageLog] = None) -> Model:
    """Run the Engine on ``RunName_NNNN.rad`` (+ RunName_{NNNN-1}.rst).

    Run number 1 starts fresh from the Starter restart; run numbers 2+
    RESUME from the previous engine run's restart (M6 chaining — see
    starter/restart.py for the contract)."""
    log = log or MessageLog()
    run_name, run_num = run_name_from_input(input_file)
    out_dir = os.path.dirname(os.path.abspath(input_file))

    listing_path = os.path.join(out_dir, f"{run_name}_{run_num:04d}.out")
    with open(listing_path, "w") as listing:
        log.attach_listing(listing)
        log.info(banner())
        log.info(f" ENGINE INPUT FILE  . . . . . . . . . : {input_file}")
        # M7: which compute backend runs the kernels (accel package);
        # resolves PYRADIOSS_BACKEND / -backend here so a requested-but-
        # missing numba surfaces its fallback warning in the listing
        log.info(f" COMPUTE BACKEND  . . . . . . . . . . : "
                 f"{backend_name(log)}")

        # ---- read controls + restart (engine lectur.F + rdresb.F) --------
        controls = parse_engine_deck(read_deck(input_file), log)
        rst = os.path.join(out_dir, f"{run_name}_{run_num - 1:04d}.rst")
        log.info(f" RESTART FILE . . . . . . . . . . . . : {rst}")
        model, saved = read_restart(rst)
        # M37: a model whose element groups reference a parsed-but-not-
        # implemented material (InactiveMaterial) is honestly not
        # simulatable — refuse loudly BEFORE any engine branch runs,
        # naming every offending law (raises InactiveMaterialError).
        refuse_inactive_materials(model, log)
        # M38: the same refusal for element groups that reference a
        # parsed-but-not-implemented PROPERTY (InactiveProperty — INJECT1,
        # TSHELL/TYPE20, composite stacks ...); raises
        # InactivePropertyError.  A property defined but referenced by no
        # elements (an /PROP/INJECT1 used only by a /MONVOL) does not block.
        refuse_inactive_properties(model, log)
        log.info(f" MODEL TITLE  . . . . . . . . . . . . : {model.title}")
        if saved is not None:
            log.info(f" RESUMING FROM TIME . . . . . . . . . : "
                     f"{saved['t']:12.5E} (CYCLE {saved['cycle']})")
        log.info(f" FINAL TIME (/RUN)  . . . . . . . . . : "
                 f"{controls.t_end:12.5E}")

        if controls.implicit:
            # M8: /IMPL switches to the implicit-static Newton driver — a
            # PARALLEL entry point that reuses the element force kernels for
            # the residual and adds the assembled tangent stiffness. The
            # explicit leap-frog loop below is left completely untouched.
            # M10: /IMPL/DYNA selects the implicit DYNAMIC (Newmark/HHT)
            # branch of that driver instead; a bare /IMPL stays static.
            if controls.impl_dyna:
                from ..implicit.dynamics import run_implicit_dynamic
                model = run_implicit_dynamic(model, controls, log, out_dir,
                                             run_name, run_num)
            else:
                from ..implicit.statics import run_implicit_static
                model = run_implicit_static(model, controls, log, out_dir,
                                            run_name, run_num)
        else:
            model = _integrate(model, controls, log, out_dir, run_name,
                               run_num, saved)
    return model


def _integrate(model: Model, controls: EngineControls, log: MessageLog,
               out_dir: str, run_name: str, run_num: int = 1,
               saved: Optional[dict] = None) -> Model:
    state = EngineState()
    n = model.numnod
    resumed = saved is not None
    # reference physical masses (pre-/DT/NODA/CST) — older restarts and
    # hand-built models may not carry them
    if len(getattr(model, "mass0", ())) != n:
        model.mass0 = model.mass.copy()

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
    rbodies = build_rigid_bodies(
        model, loads, log,
        saved_map=saved.get("rbodies") if resumed else None)
    rbe3s = build_rbe3(model, log)
    mpc = build_mpc(model, loads, log)     # /MPC (M6)
    sections = SectionForces(model, log)
    # /DT/NODA[/CST] (M6): nodal time step + mass scaling. Nodes whose
    # motion a constraint prescribes carry no stability constraint of
    # their own (see engine/mass_scaling.py).
    dampers = Dampers(model, log)          # /DAMP   (M6)
    sensors = Sensors(model, log)          # /SENSOR (M6)
    if resumed:                            # latched sensors stay latched
        sensors.fire_time.update(saved.get("sensors", {}))
    noda = NodalTimeStep(model, controls, log) if controls.dt_noda else None
    if noda is not None:
        for rb in rbodies:
            noda.set_prescribed(rb.nodes)
        for t2 in tied:
            noda.set_prescribed(t2.snode[t2.active])
        for r3 in rbe3s:
            noda.set_prescribed(np.array([r3.ref]))
        if resumed and saved.get("noda"):
            sv = saved["noda"]
            noda.mass_added = sv["mass_added"]
            noda.e_madd = sv["e_madd"]
            noda.mom_added = sv["mom_added"].copy()
            noda.mass0 = sv["mass0"]
            noda._reported = sv["reported"]
    # each run of a chain writes its own T-file (T01, T02, ... — the
    # Radioss numbering convention)
    th = TimeHistory(os.path.join(out_dir, f"{run_name}T{run_num:02d}.csv"),
                     model, log)

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
    # /MPC: project the initial velocities onto G v = 0 once — the
    # constraint force then does exactly zero work forever (mpc.py)
    if mpc is not None:
        mpc.enforce(model.v, model.vr, inv_mass, inv_inertia)

    real = model.mass < 1e29
    if resumed:
        # ---- restore the accumulated engine state (M6 chaining) ----------
        # the counters the ledgers depend on, the balance reference, and
        # the NEXT time step (already computed by the previous run —
        # taking anything else would fork from the unchained trajectory)
        for key in ("t", "cycle", "wext", "econt", "epeak", "ndel",
                    "e_num", "e_madd", "e_damp"):
            setattr(state, key, saved[key])
        _energies.e0 = saved["e0"]
        dt = saved["dt"]
        next_th = saved["next_th"]
        next_anim = saved["next_anim"]
        anim_no = saved["anim_no"]
        log.info(f" RESUMED TIME STEP  . . . . . . . . . : {dt:12.5E}\n")
    else:
        # initial energy = reference E0 of the balance: kinetic + any
        # initial internal energy (an /EOS with E0/P0 starts charged, M6)
        _energies.e0 = float(0.5 * (model.mass[real, None]
                                    * model.v[real] ** 2).sum()) \
            + _element_energy_sum(model)

        # ---- priming pass: dt=0 'cycle' just to collect the initial
        # critical time step from every kernel (no state advances at 0).
        dt = 0.0
        dt_next = EP30
        claims = []                # per-group dt_e arrays (for /DT/NODA)
        for name, group in model.element_groups():
            dt_e = KERNELS[name].forces(group, model.x, model.v, model.vr,
                                        0.0, fint, mint)
            claims.append(dt_e)
            dt_next = min(dt_next, float(dt_e.min()))
        if noda is not None:
            # nodal time step (and, with CST, the initial mass addition —
            # a deck whose smallest elements already violate dT_min gets
            # scaled before the first cycle, like the original)
            noda.assemble(claims)
            dt_next = noda.apply(mass_eff, inv_mass, model.v, 0.0)
            state.e_madd = noda.e_madd
        # penalty interfaces bound the step from cycle 0 (their stiffness
        # is static in this port), so an impact on the very first cycles
        # is safe (a /SENSOR-gated interface claims nothing until it
        # fires — its NEAR accumulation pulls dt down when it does, M4)
        for ct in contacts:
            if sensors.active(ct.itf.sens_id):
                dt_next = min(dt_next, ct.dt_bound)
        fint[:] = 0.0
        mint[:] = 0.0
        dt = controls.dt_scale * dt_next
        next_th = 0.0
        next_anim = 0.0
        anim_no = 0
        log.info(f" INITIAL TIME STEP  . . . . . . . . . : {dt:12.5E}\n")
    log.info("  CYCLE       TIME        TIME-STEP    ENERGY-IE    "
             "ENERGY-KE    ENERGY-HE    ENERGY-CE    ENERGY-EN    "
             "EXT-WORK   ERROR%")
    # reference for the numerical-dissipation ledger (step 6c): the state
    # bookings so far (the priming pass books nothing at dt = 0; on a
    # resume this equals the previous run's closing value by definition)
    state.e_booked_prev = _element_energy_sum(model)

    # /STATE/DT (M6): periodic restart snapshots (crash recovery / early
    # chaining points) — each write refreshes RunName_{nn}.rst
    next_state = controls.state_tstart if controls.state_dt > 0 else EP30

    def _engine_snapshot():
        """The accumulated-state dict of the restart contract (see
        starter/restart.py)."""
        return {
            "t": state.t, "cycle": state.cycle, "wext": state.wext,
            "econt": state.econt, "epeak": state.epeak, "ndel": state.ndel,
            "e_num": state.e_num, "e_madd": state.e_madd,
            "e_damp": state.e_damp, "e0": _energies.e0, "dt": dt,
            "next_th": next_th, "next_anim": next_anim, "anim_no": anim_no,
            "sensors": dict(sensors.fire_time),
            "rbodies": {rb.rb.id: {
                "R": rb.R.copy(), "L": rb.L.copy(),
                "v_ref": rb.v_ref.copy(), "w": rb.w.copy(),
                "x_ref": rb.x_ref.copy(), "xg": rb.xg.copy()}
                for rb in rbodies},
            "noda": None if noda is None else {
                "mass_added": noda.mass_added, "e_madd": noda.e_madd,
                "mom_added": noda.mom_added.copy(), "mass0": noda.mass0,
                "reported": noda._reported},
        }

    rst_path = os.path.join(out_dir, f"{run_name}_{run_num:04d}.rst")
    # dt-collapse guard reference: a run whose step implodes by many
    # orders of magnitude is dead whatever the ledgers say — without
    # this, a deck with dT_min = 0 can spin forever at dt ~ 1e-18
    dt_ref = dt
    t_wall0 = time.time()

    # ======================================================================
    #                       THE EXPLICIT TIME LOOP
    # ======================================================================
    # (the relative epsilon on the termination test avoids a degenerate
    # round-off-sized final cycle when the accumulated time lands within
    # one ulp of T_stop — which is exactly what happens at the junction
    # of a restart chain, M6)
    while state.t < controls.t_end * (1.0 - 1e-14):
        dt = min(dt, controls.t_end - state.t)  # land exactly on t_end

        # ---- 0. sensors (M6): poll and latch before anything acts --------
        if len(sensors):
            sensors.update(state.t, log)

        # ---- 1. internal forces, element by element group ----------------
        fint[:] = 0.0
        mint[:] = 0.0
        dt_next = EP30
        claims = []
        for name, group in model.element_groups():
            dt_e = KERNELS[name].forces(group, model.x, model.v, model.vr,
                                        dt, fint, mint)
            claims.append(dt_e)
            dt_next = min(dt_next, float(dt_e.min()))

        # ---- 2. contact forces -------------------------------------------
        # (into their own array — see the fcont declaration and step 5b;
        # the work increment the interface returns is its own estimate at
        # the pre-update velocities, superseded by the exact booking below)
        # With /DT/NODA the interfaces accumulate their spring stiffness
        # into the nodal array instead of bounding dt with their own
        # scalar (mass scaling must be able to HOLD the step against
        # contact springs — the springs are in stifn, so the nodal
        # formula covers them; see engine/mass_scaling.py).
        fcont[:] = 0.0
        for ct in contacts:
            if not sensors.active(ct.itf.sens_id):
                continue                    # /SENSOR-gated, not fired yet
            _, dt_i = ct.forces(model.x, model.v, model.mass, dt,
                                fcont, state.cycle,
                                stifn=None if noda is None else noda.stifn)
            if noda is None:
                dt_next = min(dt_next, dt_i)

        # ---- 3. external loads (gravity, /CLOAD, /PLOAD) -------------------
        fext[:] = 0.0
        loads.external_forces(state.t, fext, model.x, sensors)

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

        # ---- 3c. /DT/NODA[/CST]: nodal time step + mass scaling (M6) ------
        # Runs BEFORE the acceleration so the mass added for the target
        # step already stabilizes THIS cycle's update. The nodal dt
        # replaces the worst-element dt as the next-step bound.
        if noda is not None:
            noda.assemble(claims)
            dt_next = noda.apply(mass_eff, inv_mass, model.v, state.t)
            state.e_madd = noda.e_madd

        # ---- 3d. /MPC Lagrange forces (M6): the tiny coupled solve that
        # makes the ordinary update below satisfy G a = 0 (see mpc.py);
        # runs after mass scaling so it sees the final cycle masses
        if mpc is not None:
            mpc.transfer_forces(fint, fcont, fext, mint,
                                inv_mass, inv_inertia)

        # ---- 4. acceleration + velocity update (leap-frog) ----------------
        v_old = model.v.copy()     # for wall energy + contact work booking
        vr_old = model.vr.copy()   # for the internal-work ledger (6c)
        acc = (fint + fcont + fext) * inv_mass[:, None]
        model.v += acc * dt
        model.vr += mint * inv_inertia[:, None] * dt

        # ---- 4b. /DAMP mass damping (M6): exact integrating factor on the
        # freshly updated velocities; dissipation booked exactly from the
        # kinetic-energy identity (see engine/damping.py). Runs BEFORE the
        # kinematic conditions, which override it where they act.
        if len(dampers):
            v_star = model.v.copy()          # pre-damping velocities
            state.e_damp += dampers.apply(state.t, dt, model.v, model.vr,
                                          model.mass, model.inertia)
            # attribution correction on the damped nodes: the force-stage
            # work bookings (EW at the end-of-cycle velocity, the 6c
            # internal-work ledger at (v_old+v_new)/2) assume the update
            # velocities the damper then rescales — the mismatch is an
            # O(alpha dt) energy residual PER TRANSIT that the damper
            # rectifies into a net drift (measured +3.2% frozen on the
            # /MPC settle test before this correction). The exact
            # attribution of the force stage on a damped node is
            # (f_int+f_ext).(v_old+v_star)/2, so the residual booked by
            # the other ledgers is measured here and moved into EN:
            di = dampers.all_idx
            resid = float(
                np.einsum("nb,nb->", fext[di],
                          model.v[di] - 0.5 * (v_old[di] + v_star[di]))
                + np.einsum("nb,nb->", fint[di],
                            0.5 * (model.v[di] - v_star[di]))) * dt
            state.e_num += resid

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
        # the secondary inertia it carries too. v_old is v^{n-1/2}, the
        # start-of-cycle velocity, so the constraint work is booked at the
        # leapfrog midstep — see LoadsAndConstraints.apply_kinematic; this
        # is what makes an IMPULSIVE imposed-velocity start (RD-V-0700)
        # balance instead of booking twice the work at cycle 1.)
        state.wext += loads.apply_kinematic(state.t + dt, model.v, model.vr,
                                            mass_eff, model.x, dt, v_old)
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
        # /MPC velocity cleanup: remove what walls/BCS/placements may have
        # re-injected into G v (zero booked work — see mpc.py)
        if mpc is not None:
            mpc.enforce(model.v, model.vr, inv_mass, inv_inertia)

        # ---- 6c. numerical-dissipation ledger (M6) --------------------------
        # The kernels book internal energy as a STATE FUNCTION
        # (sigma_mid : deps — exact for the constitutive path), but the
        # internal forces act on the nodes and take from the kinetic
        # energy exactly  -fint . (v_old + v_new)/2 * dt  per cycle (the
        # same leapfrog midstep identity as the M4 contact-work lesson).
        # For smooth motion the two agree to a BOUNDED O(dt^2) wiggle
        # (the elastic work telescopes onto the state function). But when
        # the bulk-viscosity / hourglass dampers act on BARELY-RESOLVED
        # content (omega*dt near 2 — the guide's 2x2x2 ringing
        # reproducer), the discrete elastic force becomes a genuine
        # energy sink the state function cannot see: the M1-era balance
        # drifted secularly (-35% over 2 ms of free flight) with the qb
        # linear damper as the visible culprit. The consistent treatment
        # is to MEASURE the exact internal work each cycle and book the
        # difference against the state bookings into this separate,
        # reported ledger (EN — 'numerical dissipation'):
        #
        #   EN += [- fint.v_avg dt - mint.vr_avg dt] - d(eint + ehour)
        #
        # * healthy runs: EN wiggles around zero and stays negligible
        #   (asserted by the M1-M6 validations);
        # * damped barely-resolved ringing: EN accumulates the real
        #   numerical dissipation and the balance closes honestly;
        # * a DIVERGING run drives EN hard NEGATIVE (the elastic force
        #   injects energy the state function does not book), so the
        #   |ERR| stop still fires — the metric stays alive. A material
        #   law that misbooks its own work now surfaces in EN instead of
        #   ERR; EN is printed in the listing and T01 so it cannot hide.
        e_booked = _element_energy_sum(model)
        w_leave = -0.5 * dt * (
            float(np.einsum("nb,nb->", fint, v_old + model.v))
            + float(np.einsum("nb,nb->", mint, vr_old + model.vr)))
        state.e_num += w_leave - (e_booked - state.e_booked_prev)
        state.e_booked_prev = e_booked

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
        if state.t >= next_state:
            # /STATE/DT snapshot: a full restart, resumable by the next
            # run of the chain (and the crash-recovery point)
            write_restart(model, rst_path, engine=_engine_snapshot())
            log.info(f" -- /STATE: RESTART SNAPSHOT WRITTEN AT TIME "
                     f"{state.t:12.5E}")
            next_state += controls.state_dt
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
                     f"{e['CE']:12.5E} {e['EN']:12.5E} {e['EW']:12.5E} "
                     f"{e['ERR']:8.2f}")
            # ---- divergence guards (the original's energy-error and
            # minimum-time-step stops) --------------------------------
            if abs(e["ERR"]) > controls.energy_error_stop:
                state.stop_reason = (f"ENERGY ERROR {e['ERR']:.1f}% EXCEEDS "
                                     f"LIMIT {controls.energy_error_stop}%")
                break
            # a strongly NEGATIVE numerical-dissipation ledger is energy
            # INJECTION — the signature of instability that the closed
            # balance (which includes EN) cannot show in ERR (M6). The
            # threshold is 2x the ERR limit: EN legitimately WOBBLES by
            # a large fraction of the energy scale when most of a small
            # model's energy sits in a barely-resolved mode (a single
            # spring at omega*dt ~ 1.3 reads -35% with no instability at
            # all), while true injection runs away to -100% and beyond.
            if e["ERRN"] < -2.0 * controls.energy_error_stop:
                state.stop_reason = (
                    f"NUMERICAL ENERGY INJECTION {e['ERRN']:.1f}% EXCEEDS "
                    f"LIMIT {2.0 * controls.energy_error_stop}% — "
                    f"RUN UNSTABLE")
                break
            if not np.isfinite(e["KE"]):
                state.stop_reason = "NAN/INF DETECTED — RUN DIVERGED"
                break

        # ---- 8. next time step ---------------------------------------------
        dt = controls.dt_scale * dt_next
        dt_ref = max(dt_ref, dt)
        # with /DT/NODA/CST the mass scaling guarantees dt >= dT_min by
        # construction (up to round-off) — the minimum-step stop is the
        # very thing the option replaces
        if controls.dt_min > 0 and dt < controls.dt_min and \
                controls.dt_noda != "CST":
            state.stop_reason = (f"TIME STEP {dt:.3E} BELOW MINIMUM "
                                 f"{controls.dt_min:.3E}")
            break
        if dt < 1e-9 * dt_ref:
            state.stop_reason = (f"TIME STEP COLLAPSED ({dt:.3E}, WAS "
                                 f"{dt_ref:.3E}) — RUN DEAD")
            break

    # ======================================================================
    # final state + termination summary (like the original's final page)
    if controls.anim_dt > 0:
        path = os.path.join(out_dir, f"{run_name}A{anim_no:03d}.vtk")
        write_anim_state(path, model, state.t,
                         controls.anim_vect, controls.anim_elem)
    th.close()
    # the ENGINE restart (M6 chaining contract): RunName_{nn+1}.rad
    # resumes from this file — written on ERROR stops too, so a diverged
    # run can be re-tried from its last state with different controls
    write_restart(model, rst_path, engine=_engine_snapshot())
    log.info(f" RESTART FILE WRITTEN . . . . . . . . : {rst_path}")
    # expose the exact accumulated state to callers/tests (not pickled —
    # attached after the restart write)
    model.engine_state = state

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
    log.info(f"     NUMERICAL DISSIPATION . . : {e['EN']:14.7E}")
    log.info(f"     DAMPING DISSIPATION . . . : {e['DE']:14.7E}")
    log.info(f"     EXTERNAL WORK . . . . . . : {e['EW']:14.7E}")
    if noda is not None:
        noda.summary(log)
    log.info(f"     ELAPSED TIME  . . . . . . : "
             f"{time.time() - t_wall0:10.3f} s")
    log.info("     ------------------------------------------------")
    return model
