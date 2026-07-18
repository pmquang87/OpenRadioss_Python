#!/usr/bin/env python
"""tools/profile_cycle.py — explicit-engine cycle profiler (M39 speed work).

Purpose
-------
The M37/M38 perf sweeps recorded a throughput COLLAPSE as the port scales:
~1400 cyc/s at 99 shells (c04) -> ~90 cyc/s at 1000 bricks (c46) -> ~0.9
cyc/s at 65 k elements (c37 gasket).  This tool measures WHERE each cycle's
time goes so the M39 optimizer builders have a ranked, mechanism-level
target list instead of a guess.

It combines the two profiling views the milestone asks for:

1. **Targeted timers** — the engine loop (``pyradioss.engine.engine._integrate``,
   the leap-frog documented stage-by-stage in that file) is instrumented by
   MONKEYPATCHING the stage-boundary callables with accumulating
   ``perf_counter`` wrappers.  NO engine edit: the loop body is untouched;
   we only wrap the functions/methods it already calls.  The wrapped stages
   map 1:1 onto resol.F's cycle:

       skew_update    engine.py:419   NEWSKW (moving skews)          [0b]
       elem_forces    engine.py:426-430   FORINT per element group  [1]
         +-assembly_scatter  fastmath.scatter_add3 (asspar)         (nested)
       contact_forces engine.py:442-449   TYPE7/11 interface forces [2]
         +-contact_broad     i7buce voxel sort (nested)
         +-contact_narrow    i7dst3 closest point (nested)
       ext_loads      engine.py:453   external loads (CLOAD/PLOAD)  [3]
       tied_rbe3      engine.py:461-465   TYPE2 / RBE3 transfer     [3b]
       noda_dt        engine.py:471-474   /DT/NODA nodal step+mass  [3c]
       mpc            engine.py:479   /MPC Lagrange forces          [3d]
       kinematics     engine.py:522-534   rbody/impvel/wall         [5]
       output         engine.py:616-651   T-file, energies, listing [7]
       (residual)     inline arithmetic: accel/vel/pos update, the
                      contact/ext-work/numerical-dissipation einsums,
                      v_old/vr_old copies, the dt min-reduction        [4,4b,5b,6,6c,8]

   The top-level stages are mutually exclusive; their sum + the residual
   equals the measured loop wall clock, so the per-stage SHARES partition
   100% of the cycle.

2. **cProfile** — a second capped run under cProfile gives per-FUNCTION
   self-time (tottime), which decomposes the residual and names the exact
   NumPy primitive inside each stage (np.bincount for the scatter, c_einsum
   for the energy bookings, the material eval inside a kernel, ...).

Both runs are CYCLE-CAPPED to a configurable N via the same mechanism: the
first unconditional per-cycle call, ``SkewSet.update`` (engine.py:419), is
wrapped to raise once N cycles have completed.  The priming pass (dt=0,
before the loop) is excluded because the in_loop flag flips only on the
first in-loop skew call.

Load robustness
---------------
The user's own 12-process MPI job (engine_win64_impi) may share this box.
Absolute ms/cycle are therefore UPPER BOUNDS; the per-stage SHARES are the
load-robust product.  Every record samples ``tasklist`` for the impi
process count at start and end and carries a ``contended`` flag.

Usage
-----
    python tools/profile_cycle.py CASE [--cycles N] [--outdir DIR]
    python tools/profile_cycle.py all           # the five M39 regimes
    python tools/profile_cycle.py --deck ENGINE.rad --starter STARTER.rad

CASE is one of the registered regimes (tensile_bar, c04_E1000, c46_LAW70,
c37_T1040, rigid_impactor) or ``all``.
"""

from __future__ import annotations

import argparse
import contextlib
import cProfile
import json
import os
import pstats
import subprocess
import sys
import time
from collections import defaultdict
from time import perf_counter

# --- repo import path (tool lives in tools/) ---------------------------------
_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

# Single-process fairness: pin every BLAS/OMP pool to one thread so the
# profile measures the port as M38 ran it (one NumPy process, -nt unset) and
# does not spawn worker threads that fight the user's MPI job.  Must be set
# BEFORE numpy imports.
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np  # noqa: E402

# =============================================================================
# Registered regimes (the four milestone regimes + one contact-heavy deck).
# Paths are the profile39 working copies staged next to this run.
# =============================================================================
_SCRATCH = os.path.join(
    os.environ.get("TEMP", ""), "claude",
    "C--Users-pmqua-PycharmProjects-OpenRadioss-Python",
    "3c2dcf1e-08fd-425b-ab67-ec0215fa4d3d", "scratchpad", "profile39")


def _dk(sub, stem, run):
    return os.path.join(_SCRATCH, "decks", sub, f"{stem}_{run:04d}.rad")


CASES = {
    # name: (starter_0000.rad, engine_0001.rad, default cycle cap, blurb)
    "tensile_bar": (
        _dk("tensile_bar", "TENSILE", 0), _dk("tensile_bar", "TENSILE", 1),
        1200, "small solid (40 hexa, 99 nodes) — fixed-overhead regime"),
    "c04_E1000": (
        _dk("c04_E1000", "ROLLING", 0), _dk("c04_E1000", "ROLLING", 1),
        2000, "99 BATOZ shells, 137 nodes — many-cycle shell regime"),
    "c46_LAW70": (
        _dk("c46_LAW70", "BLOCK_H8", 0), _dk("c46_LAW70", "BLOCK_H8", 1),
        2000, "1000 LAW70-foam bricks, 1333 nodes — mid solid regime"),
    "c37_T1040": (
        _dk("c37_T1040", "gasket_completed", 0),
        _dk("c37_T1040", "gasket_completed", 1),
        400, "65439 Ogden bricks, 72047 nodes — THE 0.9 cyc/s cliff"),
    "rigid_impactor": (
        _dk("rigid_impactor", "IMPACTOR", 0),
        _dk("rigid_impactor", "IMPACTOR", 1),
        2000, "shells+bricks + /INTER/TYPE7 — contact-heavy regime"),
}

# =============================================================================
# Instrumentation state
# =============================================================================
_ACC: dict = defaultdict(float)   # stage -> accumulated seconds
_CNT: dict = defaultdict(int)     # stage -> wrapped-call count
_ST = {"in_loop": False, "cycle": 0, "limit": None,
       "t0": None, "t_end": None}
_PATCHES: list = []               # (obj, attr, original) for restore


class _StopProfiling(Exception):
    """Raised by the cycle limiter to end the capped run cleanly."""


def _reset(limit):
    _ACC.clear()
    _CNT.clear()
    _ST.update(in_loop=False, cycle=0, limit=limit, t0=None, t_end=None)


def _timed(orig, stage):
    """Accumulating wrapper; only charges while inside the time loop."""
    def w(*a, **k):
        if not _ST["in_loop"]:
            return orig(*a, **k)
        t = perf_counter()
        try:
            return orig(*a, **k)
        finally:
            _ACC[stage] += perf_counter() - t
            _CNT[stage] += 1
    w.__name__ = getattr(orig, "__name__", stage)
    w.__wrapped__ = orig
    return w


def _cycle_hook(orig, stage, timed):
    """Wrap SkewSet.update (first unconditional per-cycle call, engine.py:419):
    flip in_loop on the first in-loop entry, count cycles, and raise once the
    cap is reached (before the capped cycle runs, so exactly `limit` complete).
    ``timed`` toggles whether we also charge time (cProfile pass sets False)."""
    def w(*a, **k):
        now = perf_counter()
        if _ST["limit"] is not None and _ST["cycle"] >= _ST["limit"]:
            _ST["t_end"] = now
            raise _StopProfiling
        if _ST["t0"] is None:
            _ST["t0"] = now
            _ST["in_loop"] = True
        try:
            return orig(*a, **k)
        finally:
            if timed and _ST["in_loop"]:
                _ACC[stage] += perf_counter() - now
                _CNT[stage] += 1
            _ST["cycle"] += 1
    w.__name__ = "SkewSet.update"
    w.__wrapped__ = orig
    return w


def _patch(obj, attr, new):
    _PATCHES.append((obj, attr, getattr(obj, attr)))
    setattr(obj, attr, new)


def _unpatch_all():
    for obj, attr, orig in reversed(_PATCHES):
        setattr(obj, attr, orig)
    _PATCHES.clear()


def _install(timed: bool):
    """Monkeypatch every stage boundary.  ``timed`` False installs ONLY the
    cycle limiter (for the cProfile pass, whose own timer does the accounting)."""
    from pyradioss.elements import KERNELS
    from pyradioss.model import skew as skew_mod

    # -- cycle limiter (always) : SkewSet.update == engine.py:419 ------------
    _patch(skew_mod.SkewSet, "update",
           _cycle_hook(skew_mod.SkewSet.update, "skew_update", timed))
    if not timed:
        return

    # -- stage 1: element forces, per element type + nested scatter ---------
    # KERNELS[name] is the MODULE; engine calls getattr(module,'forces') each
    # cycle, so patching module.forces is picked up.  scatter_add3 is imported
    # INTO each element module's namespace -> patch it there too (nested sub).
    for name, mod in KERNELS.items():
        if hasattr(mod, "forces"):
            _patch(mod, "forces", _timed(mod.forces, f"elem:{name}"))
        if hasattr(mod, "scatter_add3"):
            _patch(mod, "scatter_add3",
                   _timed(mod.scatter_add3, "sub:assembly_scatter"))

    # -- stage 2: contact interface forces + nested broad/narrow ------------
    from pyradioss.contact import inter_type7, inter_type11
    _patch(inter_type7.ContactType7, "forces",
           _timed(inter_type7.ContactType7.forces, "contact_forces"))
    _patch(inter_type7.ContactType7, "_broad_phase",
           _timed(inter_type7.ContactType7._broad_phase, "sub:contact_broad"))
    _patch(inter_type7, "_narrow",
           _timed(inter_type7._narrow, "sub:contact_narrow"))
    _patch(inter_type11.ContactType11, "forces",
           _timed(inter_type11.ContactType11.forces, "contact_forces"))

    # -- stage 3 / 3b / 3c / 3d ---------------------------------------------
    from pyradioss.engine import kinematics as kin_mod
    _patch(kin_mod.LoadsAndConstraints, "external_forces",
           _timed(kin_mod.LoadsAndConstraints.external_forces, "ext_loads"))
    _patch(kin_mod.LoadsAndConstraints, "apply_kinematic",
           _timed(kin_mod.LoadsAndConstraints.apply_kinematic, "kinematics"))

    from pyradioss.contact.inter_type2 import ContactType2
    _patch(ContactType2, "transfer_forces",
           _timed(ContactType2.transfer_forces, "tied_rbe3"))
    _patch(ContactType2, "enforce",
           _timed(ContactType2.enforce, "kinematics"))
    from pyradioss.engine.rbe3 import Rbe3Constraint
    _patch(Rbe3Constraint, "transfer_forces",
           _timed(Rbe3Constraint.transfer_forces, "tied_rbe3"))
    _patch(Rbe3Constraint, "enforce",
           _timed(Rbe3Constraint.enforce, "kinematics"))

    from pyradioss.engine.mass_scaling import NodalTimeStep
    _patch(NodalTimeStep, "assemble",
           _timed(NodalTimeStep.assemble, "noda_dt"))
    _patch(NodalTimeStep, "apply",
           _timed(NodalTimeStep.apply, "noda_dt"))

    from pyradioss.engine.mpc import MpcConstraints
    if hasattr(MpcConstraints, "transfer_forces"):
        _patch(MpcConstraints, "transfer_forces",
               _timed(MpcConstraints.transfer_forces, "mpc"))

    # -- stage 5: rigid bodies + walls --------------------------------------
    from pyradioss.engine.rigid_body import RigidBodyEngine
    _patch(RigidBodyEngine, "advance",
           _timed(RigidBodyEngine.advance, "kinematics"))
    _patch(RigidBodyEngine, "enforce",
           _timed(RigidBodyEngine.enforce, "kinematics"))
    from pyradioss.engine.rigid_wall import RigidWalls
    _patch(RigidWalls, "apply", _timed(RigidWalls.apply, "kinematics"))

    # -- stage 7: outputs (energies + T-file + sections) --------------------
    from pyradioss.engine import engine as eng_mod
    _patch(eng_mod, "_energies", _EnergiesProxy(eng_mod._energies))
    from pyradioss.output.time_history import TimeHistory
    _patch(TimeHistory, "write", _timed(TimeHistory.write, "output"))
    from pyradioss.engine.sections import SectionForces
    _patch(SectionForces, "compute",
           _timed(SectionForces.compute, "output"))


class _EnergiesProxy:
    """engine._energies carries a mutable attribute ``.e0`` (the balance
    reference), read inside its own body and written by _integrate.  A plain
    function wrapper would break that; this proxy times the call and forwards
    e0 get/set to the wrapped original so the identity ``_energies.e0`` still
    resolves to the same storage."""

    def __init__(self, orig):
        object.__setattr__(self, "_orig", orig)

    def __call__(self, *a, **k):
        if not _ST["in_loop"]:
            return self._orig(*a, **k)
        t = perf_counter()
        try:
            return self._orig(*a, **k)
        finally:
            _ACC["output"] += perf_counter() - t
            _CNT["output"] += 1

    def __getattr__(self, n):
        return getattr(object.__getattribute__(self, "_orig"), n)

    def __setattr__(self, n, v):
        setattr(object.__getattribute__(self, "_orig"), n, v)


# =============================================================================
# Contention sampling
# =============================================================================
def _impi_procs() -> int:
    """Count live engine_win64_impi.exe (the user's MPI job) for the flag."""
    try:
        out = subprocess.run(
            ["tasklist", "/FI", "IMAGENAME eq engine_win64_impi.exe", "/NH"],
            capture_output=True, text=True, timeout=15).stdout
        return sum(1 for ln in out.splitlines()
                   if "engine_win64_impi" in ln.lower())
    except Exception:
        return -1


# =============================================================================
# Model size (per-element-cycle normalisation)
# =============================================================================
def _model_sizes(engine_input):
    """Read the model the way the engine does (starter restart) to count
    nodes and elements per group WITHOUT running a cycle."""
    from pyradioss.engine.engine import run_name_from_input
    from pyradioss.starter.restart import read_restart
    run_name, run_num = run_name_from_input(engine_input)
    out_dir = os.path.dirname(os.path.abspath(engine_input))
    rst = os.path.join(out_dir, f"{run_name}_{run_num - 1:04d}.rst")
    model, _ = read_restart(rst)
    groups = {name: int(g.n) for name, g in model.element_groups()}
    return int(model.numnod), sum(groups.values()), groups


# =============================================================================
# The two runs
# =============================================================================
@contextlib.contextmanager
def _quiet():
    """Silence the engine's per-100-cycle listing prints (still written to the
    .out file via attach_listing); keeps stdout cost out of the measurement."""
    with open(os.devnull, "w") as dn, contextlib.redirect_stdout(dn):
        yield


def _run_starter(starter_input):
    from pyradioss.starter.starter import run_starter
    t = perf_counter()
    with _quiet():
        run_starter(starter_input)
    return perf_counter() - t


def _timed_pass(engine_input, limit):
    """Instrumented run -> per-stage accumulated seconds + loop wall + cycles."""
    from pyradioss.engine.engine import run_engine
    _reset(limit)
    _install(timed=True)
    try:
        with _quiet():
            try:
                run_engine(engine_input)
            except _StopProfiling:
                pass
            else:
                _ST["t_end"] = perf_counter()
    finally:
        _unpatch_all()
    wall = (_ST["t_end"] - _ST["t0"]) if _ST["t0"] else 0.0
    cycles = _ST["cycle"]
    return dict(_ACC), dict(_CNT), wall, cycles


def _cprofile_pass(engine_input, limit):
    """cProfile run (only the cycle limiter installed) -> pstats.Stats."""
    from pyradioss.engine.engine import run_engine
    _reset(limit)
    _install(timed=False)
    pr = cProfile.Profile()
    try:
        with _quiet():
            pr.enable()
            try:
                run_engine(engine_input)
            except _StopProfiling:
                pass
            pr.disable()
    finally:
        _unpatch_all()
    return pstats.Stats(pr), _ST["cycle"]


# =============================================================================
# cProfile -> stage bucketing (self-time / tottime partitions 100%)
# =============================================================================
def _stage_of(filename: str, func: str) -> str:
    """Map a profiled function to a stage bucket by its defining file/name.
    tottime (self time) is additive, so summing it per bucket partitions the
    whole capped run."""
    f = filename.replace("\\", "/")
    base = f.rsplit("/", 1)[-1]
    ELEM = {"solid_hexa8.py": "elem:bricks", "solid_tetra4.py": "elem:tetras",
            "shell_bt4.py": "elem:shells", "shell_tri3.py": "elem:sh3n",
            "truss.py": "elem:trusses", "spring.py": "elem:springs",
            "spring_general.py": "elem:springs", "beam_type3.py": "elem:beams"}
    if base in ELEM:
        return ELEM[base]
    # material constitutive laws + property tables are called BY the element
    # kernels (the per-element stress update) -> element-compute helpers
    if base.startswith("law") or base in ("tables.py", "eos.py",
                                          "materials.py", "hardening.py"):
        return "elem:material"
    if base == "fastmath.py":
        return "assembly_scatter" if func == "scatter_add3" else "elem:fastmath"
    if base == "inter_type7.py":
        if func in ("_broad_phase", "_expand_matches", "key"):
            return "contact_broad"
        if func in ("_narrow", "_closest_point_on_triangle"):
            return "contact_narrow"
        return "contact_other"
    if base in ("inter_type11.py",):
        return "contact_other"
    if base == "inter_type2.py":
        return "tied_rbe3"
    if base == "rbe3.py":
        return "tied_rbe3"
    if base == "mass_scaling.py":
        return "noda_dt"
    if base == "kinematics.py":
        return "kinematics"
    if base in ("rigid_body.py", "rigid_wall.py", "mpc.py"):
        return "kinematics"
    if base == "engine.py":
        return "loop_body(engine.py)"
    if base in ("time_history.py", "anim_vtk.py", "sections.py"):
        return "output"
    if base == "skew.py":
        return "skew_update"
    # NumPy / builtins: the primitives the inline loop body and kernels share
    if "numpy" in f or base.startswith("<") or func.startswith("<"):
        fn = func.lower()
        if "einsum" in fn:
            return "np:einsum"
        if "bincount" in fn:
            return "assembly_scatter"
        if "reduce" in fn or "'min'" in fn or "'sum'" in fn or "'max'" in fn:
            return "np:reduce"
        if "'copy'" in fn or "copyto" in fn or "empty" in fn or "zeros" in fn:
            return "np:alloc/copy"
        if "'at'" in fn:
            return "assembly_scatter"
        return "np:other"
    return f"other:{base}"


def _bucket_cprofile(stats: pstats.Stats):
    """Return (stage->tottime, [top function rows]) from a Stats object."""
    per_stage = defaultdict(float)
    rows = []
    total_tt = 0.0
    for (fn, ln, func), (cc, nc, tt, ct, callers) in stats.stats.items():
        per_stage[_stage_of(fn, func)] += tt
        total_tt += tt
        rows.append((tt, ct, nc, _stage_of(fn, func),
                     f"{fn.replace(chr(92), '/').rsplit('/', 1)[-1]}:{ln}:{func}"))
    rows.sort(reverse=True)
    return dict(per_stage), rows, total_tt


# =============================================================================
# Driver
# =============================================================================
def profile_case(name, starter_input, engine_input, limit, outdir):
    print(f"\n{'=' * 78}\n  PROFILING {name}  (cap {limit} cycles)\n{'=' * 78}")
    impi0 = _impi_procs()
    t_starter = _run_starter(starter_input)   # writes the _0000.rst
    numnod, nelem, groups = _model_sizes(engine_input)
    print(f"  model: {numnod} nodes, {nelem} elements, groups={groups}")

    # pass 1 — targeted timers
    acc, cnt, wall, cyc1 = _timed_pass(engine_input, limit)
    # pass 2 — cProfile
    stats, cyc2 = _cprofile_pass(engine_input, limit)
    impi1 = _impi_procs()
    contended = max(impi0, impi1) > 0

    stage_tt, rows, total_tt = _bucket_cprofile(stats)

    # --- assemble the targeted-timer per-stage table -----------------------
    top_stages = ["skew_update", "elem:bricks", "elem:tetras", "elem:shells",
                  "elem:sh3n", "elem:trusses", "elem:springs", "elem:beams",
                  "contact_forces", "ext_loads", "tied_rbe3", "noda_dt",
                  "mpc", "kinematics", "output"]
    top_sum = sum(acc.get(s, 0.0) for s in top_stages)
    residual = max(wall - top_sum, 0.0)
    per_cycle_ms = (wall / cyc1 * 1e3) if cyc1 else 0.0
    per_ec_us = (wall / cyc1 / nelem * 1e6) if (cyc1 and nelem) else 0.0

    timer_table = []
    for s in top_stages:
        if acc.get(s, 0.0) <= 0 and s not in ("elem:bricks",):
            continue
        sec = acc.get(s, 0.0)
        timer_table.append({
            "stage": s, "ms_per_cycle": sec / cyc1 * 1e3 if cyc1 else 0.0,
            "share_pct": sec / wall * 100 if wall else 0.0,
            "calls_per_cycle": cnt.get(s, 0) / cyc1 if cyc1 else 0.0})
    timer_table.append({
        "stage": "integration_inline(residual)",
        "ms_per_cycle": residual / cyc1 * 1e3 if cyc1 else 0.0,
        "share_pct": residual / wall * 100 if wall else 0.0,
        "calls_per_cycle": 0.0})
    # nested sub-timers (subsets of the stages above; not in the 100% sum)
    subs = {k: v for k, v in acc.items() if k.startswith("sub:")}

    rec = {
        "case": name, "blurb": CASES.get(name, (None, None, None, ""))[3],
        "nodes": numnod, "elements": nelem, "groups": groups,
        "cycles_profiled": cyc1, "loop_wall_s": wall,
        "ms_per_cycle": per_cycle_ms, "us_per_elem_cycle": per_ec_us,
        "starter_s": t_starter,
        "contended": contended, "impi_procs_start": impi0,
        "impi_procs_end": impi1,
        "timer_stages": timer_table,
        "timer_subtimers": {k[4:]: {
            "ms_per_cycle": v / cyc1 * 1e3 if cyc1 else 0.0,
            "share_pct": v / wall * 100 if wall else 0.0} for k, v in subs.items()},
        "cprofile_stage_tottime_share": {
            k: v / total_tt * 100 for k, v in sorted(
                stage_tt.items(), key=lambda kv: -kv[1])},
        "cprofile_top_functions": [
            {"tottime_s": r[0], "cumtime_s": r[1], "ncalls": r[2],
             "stage": r[3], "func": r[4]} for r in rows[:30]],
    }

    # --- console summary ---------------------------------------------------
    flag = "  [CONTENDED: impi live -> abs times UPPER BOUND]" if contended else ""
    print(f"  cycles={cyc1}  loop_wall={wall:.3f}s  "
          f"{per_cycle_ms:.3f} ms/cyc  {per_ec_us:.4f} us/elem-cyc{flag}")
    print(f"  {'stage':<32}{'ms/cyc':>10}{'share%':>9}{'calls/cyc':>11}")
    for r in timer_table:
        if r["ms_per_cycle"] < 1e-4 and r["share_pct"] < 0.05:
            continue
        print(f"  {r['stage']:<32}{r['ms_per_cycle']:>10.4f}"
              f"{r['share_pct']:>9.2f}{r['calls_per_cycle']:>11.2f}")
    if subs:
        print("  -- nested sub-timers (subset of the stage above) --")
        for k, v in subs.items():
            print(f"  {k:<32}{v / cyc1 * 1e3:>10.4f}"
                  f"{v / wall * 100:>9.2f}")
    print("  -- cProfile self-time (tottime) share by stage --")
    for k, v in sorted(stage_tt.items(), key=lambda kv: -kv[1])[:12]:
        if v / total_tt * 100 < 0.3:
            continue
        print(f"  {k:<32}{v:>10.3f}s{v / total_tt * 100:>8.2f}%")

    # --- write per-case artefacts ------------------------------------------
    os.makedirs(outdir, exist_ok=True)
    with open(os.path.join(outdir, f"{name}.json"), "w") as fh:
        json.dump(rec, fh, indent=1)
    # full pstats text dump (top 40 by tottime and by cumtime)
    with open(os.path.join(outdir, f"{name}_pstats.txt"), "w") as fh:
        stats.stream = fh
        fh.write(f"# {name}: {cyc2} cycles under cProfile\n\n== by tottime ==\n")
        stats.sort_stats("tottime").print_stats(40)
        fh.write("\n== by cumtime ==\n")
        stats.sort_stats("cumulative").print_stats(40)
    return rec


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("case", nargs="?", default="all",
                    help="a registered case, or 'all' (default)")
    ap.add_argument("--cycles", type=int, default=0,
                    help="cycle cap override (0 = per-case default)")
    ap.add_argument("--deck", default=None, help="custom engine _0001.rad")
    ap.add_argument("--starter", default=None, help="custom starter _0000.rad")
    ap.add_argument("--outdir", default=os.path.join(_SCRATCH, "data"))
    args = ap.parse_args(argv)

    if args.deck:
        limit = args.cycles or 500
        profile_case(os.path.basename(args.deck).split("_0001")[0],
                     args.starter, args.deck, limit, args.outdir)
        return 0

    names = list(CASES) if args.case == "all" else [args.case]
    summary = []
    for nm in names:
        starter_i, engine_i, cap, _ = CASES[nm]
        if not os.path.exists(engine_i):
            print(f"  !! skip {nm}: missing {engine_i}")
            continue
        summary.append(profile_case(nm, starter_i, engine_i,
                                    args.cycles or cap, args.outdir))
    if summary:
        with open(os.path.join(args.outdir, "summary.json"), "w") as fh:
            json.dump(summary, fh, indent=1)
        print(f"\n  wrote {len(summary)} record(s) -> {args.outdir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
