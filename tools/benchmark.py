#!/usr/bin/env python3
"""
M7 benchmark: run the bundled examples through starter+engine and time
the engine wall clock under each compute backend (see pyradioss/accel).

Usage:
    python tools/benchmark.py                    # all examples, both backends
    python tools/benchmark.py -e rigid_impactor notched_plate
    python tools/benchmark.py -b numpy           # a single backend

Method (kept deliberately honest):

* each (example, backend) run happens in a FRESH temporary copy of the
  example directory — no output files or restart snapshots leak between
  runs, and the repository tree stays clean;
* each run is a FRESH subprocess with PYRADIOSS_BACKEND set — backend
  selection is per-process state, and a subprocess also charges the numba
  path its real import cost. The numba JIT compilation itself is cached
  on disk (accel/jit_kernels is compiled with cache=True), so a warm-up
  compile pass is done once, outside the timing, before the numba runs;
* the reported time is the engine's own "ELAPSED TIME" from the listing
  (pure solve wall clock, excluding interpreter/import startup), plus the
  subprocess wall clock for reference;
* the starter is run once per copy, untimed — M7 is about the cycle path.

The gas_piston example is a two-run restart chain; both engine runs are
executed and their elapsed times summed (chaining is part of its point).
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

#: example directory -> (run name, [engine decks in run order])
EXAMPLES = {
    "tensile_bar": ("TENSILE", ["TENSILE_0001.rad"]),
    "box_beam_impact": ("BOXIMP", ["BOXIMP_0001.rad"]),
    "antenna_mast": ("MAST", ["MAST_0001.rad"]),
    "rubber_block": ("RUBBER", ["RUBBER_0001.rad"]),
    "notched_plate": ("NOTCH", ["NOTCH_0001.rad"]),
    "spot_weld": ("SPOTWELD", ["SPOTWELD_0001.rad"]),
    "edge_impact": ("EDGE", ["EDGE_0001.rad"]),
    "rigid_impactor": ("IMPACTOR", ["IMPACTOR_0001.rad"]),
    "gas_piston": ("GASPISTON", ["GASPISTON_0001.rad",
                                 "GASPISTON_0002.rad"]),
}

_ELAPSED = re.compile(r"ELAPSED TIME\s*\.[\s.]*:\s*([0-9.Ee+-]+)")
_ERROR = re.compile(r"ENERGY ERROR\s*\.[\s.]*:\s*([0-9.Ee+-]+)")


def _run(cmd, cwd, backend=None):
    env = dict(os.environ)
    if backend is not None:
        env["PYRADIOSS_BACKEND"] = backend
    return subprocess.run(cmd, cwd=cwd, env=env, text=True,
                          capture_output=True, check=False)


def bench_example(name, backend, workdir):
    """One (example, backend) measurement in a fresh copy under workdir.
    Returns (engine_elapsed_s, subprocess_wall_s, energy_error_pct)."""
    run_name, engine_decks = EXAMPLES[name]
    src = os.path.join(REPO, "examples", name)
    dst = os.path.join(workdir, f"{name}-{backend}")
    shutil.copytree(src, dst)

    r = _run([sys.executable, "-m", "pyradioss.starter",
              "-i", f"{run_name}_0000.rad"], dst)
    if r.returncode != 0:
        raise RuntimeError(f"starter failed for {name}:\n{r.stdout[-2000:]}")

    elapsed = 0.0
    err = None
    t0 = time.perf_counter()
    for deck in engine_decks:
        r = _run([sys.executable, "-m", "pyradioss.engine", "-i", deck],
                 dst, backend)
        if r.returncode != 0:
            raise RuntimeError(f"engine failed for {name} ({backend}):\n"
                               f"{r.stdout[-2000:]}\n{r.stderr[-2000:]}")
        out_file = os.path.join(
            dst, deck.replace(".rad", ".out"))
        with open(out_file) as fh:
            text = fh.read()
        m = _ELAPSED.search(text)
        if not m:
            raise RuntimeError(f"no ELAPSED TIME in {out_file}")
        elapsed += float(m.group(1))
        me = _ERROR.search(text)
        err = float(me.group(1)) if me else None
    wall = time.perf_counter() - t0
    shutil.rmtree(dst)
    return elapsed, wall, err


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("-e", "--examples", nargs="+",
                    default=list(EXAMPLES),
                    choices=list(EXAMPLES), metavar="NAME",
                    help="examples to run (default: all nine)")
    ap.add_argument("-b", "--backends", nargs="+",
                    default=["numpy", "numba"],
                    choices=["numpy", "numba"],
                    help="backends to time (default: numpy numba)")
    args = ap.parse_args(argv)

    if "numba" in args.backends:
        # warm the on-disk JIT cache OUTSIDE the timings (one-time cost;
        # a cold cache would otherwise be charged to the first example)
        print("warming numba JIT cache ...", flush=True)
        r = _run([sys.executable, "-c",
                  "from pyradioss.accel import select_backend; "
                  "select_backend('numba'); "
                  "import pyradioss.accel.jit_kernels"], REPO, "numba")
        if r.returncode != 0:
            print(r.stderr[-2000:])
            print("numba unavailable — dropping the numba column")
            args.backends = [b for b in args.backends if b != "numba"]

    rows = []
    with tempfile.TemporaryDirectory(prefix="pyradioss-bench-") as workdir:
        for name in args.examples:
            row = {"example": name}
            for backend in args.backends:
                el, wall, err = bench_example(name, backend, workdir)
                row[backend] = el
                row[f"{backend}_err"] = err
                print(f"{name:16s} {backend:6s} engine {el:8.2f} s   "
                      f"(process {wall:7.2f} s, energy error "
                      f"{err if err is not None else float('nan'):7.2f} %)",
                      flush=True)
            rows.append(row)

    # markdown summary (paste-ready for the porting guide / PR)
    print("\n| example | " + " | ".join(args.backends)
          + (" | speedup |" if len(args.backends) == 2 else " |"))
    print("|---" * (len(args.backends) + 1
                    + (1 if len(args.backends) == 2 else 0)) + "|")
    for row in rows:
        cells = [f"{row[b]:.2f} s" for b in args.backends]
        if len(args.backends) == 2:
            a, b = args.backends
            cells.append(f"{row[a] / row[b]:.2f}x")
        print(f"| {row['example']} | " + " | ".join(cells) + " |")
    return 0


if __name__ == "__main__":
    sys.exit(main())
