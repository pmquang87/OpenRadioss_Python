"""
Command-line entry point:  pyradioss-starter -i RunName_0000.rad

Mirrors the original Starter command line (starter_linux64_gf -i ... -np N
-nt N); -np N decomposes the model into N SPMD domains and writes one
restart per domain (RunName_0000_0001.rst ..., pyradioss/spmd/domdec.py),
-nt sets NumPy's thread count environment variables as a best effort.
"""

from __future__ import annotations

import argparse
import os
import sys


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        prog="pyradioss-starter",
        description="pyradioss Starter — reads RunName_0000.rad, checks and "
                    "initializes the model, writes the restart file.")
    ap.add_argument("-i", "-input", dest="input", required=True,
                    help="starter input deck (RunName_0000.rad)")
    ap.add_argument("-nt", "-nthread", dest="nthread", type=int, default=0,
                    help="number of threads (sets numpy thread env vars)")
    ap.add_argument("-np", dest="nspmd", type=int, default=1,
                    help="number of SPMD domains (one restart per domain)")
    args = ap.parse_args(argv)

    if args.nthread > 0:
        for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                    "MKL_NUM_THREADS"):
            os.environ[var] = str(args.nthread)

    from ..common.messages import StarterError
    from .starter import run_starter
    if not os.path.isfile(args.input):
        print(f"\n     STARTER TERMINATION : ERROR\n     Starter input file not found: {args.input}")
        return 2
    try:
        res = run_starter(args.input, nspmd=max(1, args.nspmd))
        if isinstance(res, int) and res != 0:
            return 2
    except (StarterError, FileNotFoundError) as exc:
        print(f"\n     STARTER TERMINATION : ERROR\n     {exc}")
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
