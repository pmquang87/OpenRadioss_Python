"""
Command-line entry point:  pyradioss-engine -i RunName_0001.rad

Mirrors the original Engine command line (engine_linux64_gf -i ...).
"""

from __future__ import annotations

import argparse
import os
import sys


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        prog="pyradioss-engine",
        description="pyradioss Engine — runs the explicit simulation from "
                    "RunName_0001.rad + the Starter restart file.")
    ap.add_argument("-i", "-input", dest="input", required=True,
                    help="engine input file (RunName_0001.rad)")
    ap.add_argument("-nt", "-nthread", dest="nthread", type=int, default=0,
                    help="number of threads (sets numpy thread env vars)")
    ap.add_argument("-np", dest="nspmd", type=int, default=1,
                    help="SPMD domains (the Starter's -np): N > 1 runs the "
                         "decomposed model — one domain per MPI process "
                         "under mpirun (mpi4py), else N threads of this "
                         "process")
    ap.add_argument("-backend", "--backend", dest="backend", default=None,
                    choices=["numpy", "numba", "cupy", "auto"],
                    help="compute backend: 'auto' (M40 default — cupy GPU or "
                         "numba JIT backend on models large enough to amortise "
                         "warm-up, else numpy) or PIN 'numpy'/'numba'/'cupy' — "
                         "overrides the PYRADIOSS_BACKEND environment "
                         "variable; missing optional backend falls back to "
                         "numpy with a warning")
    ap.add_argument("-linsolve", "--linsolve", dest="linsolve", default=None,
                    choices=["superlu", "cholmod", "mumps"],
                    help="direct linear solver for the implicit run (M8): "
                         "'superlu' (default, ships with SciPy) or the "
                         "optional 'cholmod'/'mumps' backends — overrides "
                         "PYRADIOSS_LINSOLVE; a missing optional library "
                         "falls back to SuperLU with a warning")
    args = ap.parse_args(argv)

    if args.nthread > 0:
        for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                    "MKL_NUM_THREADS"):
            os.environ[var] = str(args.nthread)
    if args.backend is not None:
        from ..accel import select_backend
        select_backend(args.backend)
    if args.linsolve is not None:
        # implicit runs (/IMPL) read this; explicit runs ignore it. Setting
        # the env var keeps the selection lazy (the solver resolves on first
        # use, like the compute backend), and lets it survive into the driver.
        os.environ["PYRADIOSS_LINSOLVE"] = args.linsolve

    from .engine import run_engine
    if not os.path.isfile(args.input):
        print(f"\n     ENGINE TERMINATION : ERROR\n     Engine input file not found: {args.input}")
        return 2
    try:
        from ..spmd.comm import mpi_world_size
        if args.nspmd > 1 or mpi_world_size() > 1:
            # SPMD run (radioss2.F / inipar.F): mpirun -> mpi4py domains,
            # otherwise the in-process thread domains
            from ..spmd.driver import run_engine_spmd
            model = run_engine_spmd(args.input, args.nspmd)
        else:
            model = run_engine(args.input)
    except Exception as exc:
        print(f"\n     ENGINE TERMINATION : ERROR\n     {exc}")
        return 2

    if hasattr(model, "engine_state") and model.engine_state.stop_reason:
        reason = model.engine_state.stop_reason
        if not reason.startswith("/STOP/"):
            return 2
    if hasattr(model, "implicit_result") and not getattr(model.implicit_result, "converged", True):
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
