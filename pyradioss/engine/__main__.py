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
                    help="MPI domains (ignored: the port has no MPI)")
    args = ap.parse_args(argv)

    if args.nthread > 0:
        for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                    "MKL_NUM_THREADS"):
            os.environ[var] = str(args.nthread)
    if args.nspmd > 1:
        print(" ** WARNING: -np ignored (no MPI in pyradioss)")

    from .engine import run_engine
    run_engine(args.input)
    return 0


if __name__ == "__main__":
    sys.exit(main())
