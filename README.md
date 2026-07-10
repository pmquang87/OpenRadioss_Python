# OpenRadioss_Python (`pyradioss`)

A **Python port of [OpenRadioss](https://github.com/OpenRadioss/OpenRadioss)**, the
open-source explicit finite-element solver for crash, impact and highly non-linear
transient dynamics.

> **Goal of this repository.** Reproduce the *functional architecture* of
> OpenRadioss — the Starter / Engine split, the Radioss input-deck format, the
> explicit central-difference solver, the element/material/contact libraries and
> the output files — in pure, heavily documented Python. The point is **not**
> speed (Python is orders of magnitude slower than the original Fortran); the
> point is a code base you can *read, understand and modify*. Every module
> carries long explanatory comments and a reference to the original Fortran
> source it was ported from.

---

## How OpenRadioss works (and how this port mirrors it)

OpenRadioss is split into two executables, and so is this port:

| OpenRadioss (Fortran) | This port (Python) | Role |
|---|---|---|
| `starter/` → `starter_linux64_gf` | `pyradioss/starter/` → `pyradioss-starter` | Reads the input deck `<RunName>_0000.rad`, checks the model, builds/initializes all data structures (masses, element volumes, contact surfaces…), writes the `<RunName>_0000.out` listing and a *restart* file for the Engine. |
| `engine/` → `engine_linux64_gf` | `pyradioss/engine/` → `pyradioss-engine` | Reads the engine control file `<RunName>_0001.rad` **and** the Starter restart file, runs the explicit time-integration loop, writes the `<RunName>_0001.out` listing, the time-history file (T01) and animation states (A-files). |

The original build chain (`build_script.sh`, gfortran, MPI…) described in the
upstream [HOWTO.md](https://github.com/OpenRadioss/OpenRadioss/blob/main/HOWTO.md)
is replaced by a plain Python package — see [HOWTO.md](HOWTO.md) in this repo
for the equivalent instructions.

## Installation

```bash
git clone https://github.com/pmquang87/OpenRadioss_Python.git
cd OpenRadioss_Python
pip install -e .          # installs `pyradioss` + the two console scripts
```

The only runtime dependency is **NumPy** (all element/material kernels are
vectorized over elements — this is the single concession to performance, and it
also matches how the Fortran loops over element *groups*).

## Running a simulation

Exactly like the original solver — two steps, same file naming convention:

```bash
cd examples/tensile_bar

# 1. Starter: reads TENSILE_0000.rad, writes TENSILE_0000.out + TENSILE_0000.rst
pyradioss-starter -i TENSILE_0000.rad

# 2. Engine: reads TENSILE_0001.rad + TENSILE_0000.rst, runs the simulation
pyradioss-engine  -i TENSILE_0001.rad
```

(You can also run them without installing: `python -m pyradioss.starter -i …`
and `python -m pyradioss.engine -i …`.)

Outputs:

| File | Content |
|---|---|
| `<Run>_0000.out` | Starter listing: model summary, checks, initial mass/inertia. |
| `<Run>_0001.out` | Engine listing: cycle table (time, time step, energies, momentum) and final energy balance. |
| `<Run>T01.csv` | Time history (global energies + requested `/TH` groups). The original binary T01 format is replaced by documented ASCII/CSV. |
| `<Run>A000.vtk`, `A001.vtk`, … | Animation states. The proprietary ANIM format is replaced by **legacy VTK**, directly readable in [ParaView](https://www.paraview.org/) — the tool the OpenRadioss HOWTO itself recommends for post-processing. |

## What is implemented so far

See [PORTING_GUIDE.md](PORTING_GUIDE.md) for the detailed feature matrix, the
map from every Python module to the original Fortran directory, and the porting
roadmap. Milestones 1–2 (this state of the repository) cover:

- **Input**: the Radioss block-keyword deck format (`/NODE`, `/BRICK`,
  `/TETRA4`, `/SHELL`, `/SH3N`, `/TRUSS`, `/SPRING`, `/BEAM`, `/PART`,
  `/MAT/LAW1`, `/MAT/LAW2`, `/PROP/TYPE1/2/3/4/14`,
  `/BCS`, `/INIVEL`, `/GRAV`, `/CLOAD`, `/IMPVEL`, `/FUNCT`, `/GRNOD`,
  `/RWALL`, `/INTER/TYPE7`, `/TH`, …) with `#include` support.
- **Elements**: 8-node solid (one-point integration + Flanagan–Belytschko
  hourglass control) with degenerated-brick→tetra conversion, 4-node
  constant-strain tetra, 4-node Belytschko–Tsay shell (membrane + bending +
  transverse shear, through-thickness integration, BLT84 stiffness hourglass
  control), 3-node C0 triangle shell, 2-node corotational Timoshenko beam,
  2-node truss, 2-node spring.
- **Materials**: LAW1 (linear elastic, hypoelastic Jaumann update) and LAW2
  (Johnson–Cook elasto-plasticity with strain-rate hardening, 3D and
  plane-stress variants).
- **Engine**: explicit central-difference integration, element/nodal stable
  time step, boundary conditions, initial/imposed velocities, gravity,
  concentrated loads, kinematic rigid walls, TYPE7-style penalty contact with
  bucket search, full energy-balance bookkeeping.
- **Output**: listings, CSV time history, VTK animation states.

## Repository layout

```
pyradioss/
├── common/       # shared low-level services (≈ OpenRadioss common_source/)
├── input/        # deck reader + keyword parsers (≈ starter/source/reader)
├── model/        # in-memory model (≈ Fortran modules/derived types)
├── starter/      # the Starter program        (≈ starter/source)
├── engine/       # the Engine program         (≈ engine/source)
├── elements/     # element kernels            (≈ engine/source/elements)
├── materials/    # material laws              (≈ engine/source/materials)
├── contact/      # contact interfaces         (≈ engine/source/interfaces)
└── output/       # listings, TH, ANIM         (≈ engine/source/output)
examples/         # ready-to-run input decks
tests/            # pytest suite incl. analytic validations
```

## License

GPL-3.0 (see [LICENSE](LICENSE)) — inherited from files derived from
OpenRadioss, © Altair Engineering Inc. This repository is an independent
educational port and is not affiliated with Altair.
