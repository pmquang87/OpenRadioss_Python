# HOWTO — build, run and post-process with `pyradioss`

This document is the Python-port equivalent of the original
[OpenRadioss HOWTO.md](https://github.com/OpenRadioss/OpenRadioss/blob/main/HOWTO.md).
Where the original explains how to *compile* the Fortran Starter and Engine and
run them, this one explains the same workflow for the Python port. The section
structure deliberately follows the original so you can read them side by side.

---

## 1. How to "build" OpenRadioss_Python

There is nothing to compile. The original HOWTO's chapters on gfortran,
Intel OneAPI, cmake and MPI collapse into:

```bash
pip install -e .
```

This installs the `pyradioss` package plus two console entry points that play
the role of the compiled binaries in `OpenRadioss/exec`:

| Original binary (in `exec/`) | Python equivalent |
|---|---|
| `starter_linux64_gf`         | `pyradioss-starter` (= `python -m pyradioss.starter`) |
| `engine_linux64_gf`          | `pyradioss-engine`  (= `python -m pyradioss.engine`)  |

Requirements: Python ≥ 3.9, NumPy. For the test suite: pytest.

```bash
pip install -e ".[test]"
pytest                     # runs unit + analytic validation tests
```

## 2. How to run OpenRadioss_Python

A Radioss run is always two stages. Given a run name `MYRUN`:

1. **Starter** — model building and checking:

   ```bash
   pyradioss-starter -i MYRUN_0000.rad
   ```

   * reads the *starter deck* `MYRUN_0000.rad` (mesh, materials, properties,
     parts, loads, boundary conditions, contacts…),
   * prints/writes the listing `MYRUN_0000.out` (model summary, warnings,
     errors, initial mass),
   * writes the **restart file** `MYRUN_0000.rst` consumed by the Engine.
     (In the original this is the binary `MYRUN_0000.rst`/`.r00` restart; here
     it is a Python pickle of the initialized model — same purpose, documented
     in `pyradioss/starter/restart.py`.)

2. **Engine** — time integration:

   ```bash
   pyradioss-engine -i MYRUN_0001.rad
   ```

   * reads the *engine deck* `MYRUN_0001.rad` (final time `/RUN`, time-step
     control `/DT`, output frequencies `/TFILE`, `/ANIM/DT`, …),
   * loads `MYRUN_0000.rst`,
   * runs the explicit loop and writes `MYRUN_0001.out`, `MYRUNT01.csv`
     and `MYRUNA*.vtk` animation states.

Useful flags (mirroring the original command lines):

| Flag | Original meaning | Here |
|---|---|---|
| `-i / -input FILE` | input deck | same |
| `-nt / -nthread N` | SMP threads | sets NumPy thread env vars (best effort) |
| `-np N` | MPI domains | **accepted but ignored** — no MPI in the port (a warning is printed) |
| `-v / --version` | banner | same |

## 3. Input decks

The port reads the open Radioss block format (see the
[OpenRadioss input reference](https://help.altair.com/hwsolvers/rad/index.htm)):
lines starting with `/` open a keyword block (`/NODE`, `/MAT/LAW2/…`), `#` or
`$` start comments, `#include FILE` pulls in another file, and data cards are
whitespace-separated fields (the parser also accepts the classic fixed
10-column layout). The exact subset of keywords currently understood is listed
in [PORTING_GUIDE.md](PORTING_GUIDE.md); unknown keywords produce a clear
warning and are skipped, so real decks degrade gracefully.

Ready-to-run models live in `examples/`:

```bash
cd examples/tensile_bar     # solid bricks + Johnson-Cook plasticity
pyradioss-starter -i TENSILE_0000.rad
pyradioss-engine  -i TENSILE_0001.rad

cd ../box_beam_impact       # shell box crushed against a rigid wall
pyradioss-starter -i BOXIMP_0000.rad
pyradioss-engine  -i BOXIMP_0001.rad

cd ../antenna_mast          # cantilever /BEAM mast under a wind gust
pyradioss-starter -i MAST_0000.rad
pyradioss-engine  -i MAST_0001.rad
```

## 4. How to visualize results

The original HOWTO points users to ParaView / OpenRadioss tools for the binary
ANIM files. The port writes **legacy VTK** state files directly, so:

1. open ParaView,
2. `File → Open`, select the `MYRUNA…vtk` group,
3. press *Apply* and use the animation controls; nodal vectors
   (displacement, velocity) and element fields (von Mises stress, plastic
   strain) are available as point/cell data.

The time history `MYRUNT01.csv` is plain CSV: plot it with anything
(matplotlib, Excel, gnuplot). Column 1 is time; global energies, momentum and
mass follow; then one column group per requested `/TH` object.

## 5. Where to go next

* [PORTING_GUIDE.md](PORTING_GUIDE.md) — feature matrix, Fortran→Python module
  map, roadmap, and the conventions used throughout the code.
* `pyradioss/engine/engine.py` — the ported main loop (`resol.F` equivalent),
  the best single file to start reading: it narrates one complete explicit
  time step in comments.
