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
roadmap. Milestones 1–5 (this state of the repository) cover:

- **Input**: the Radioss block-keyword deck format (`/NODE`, `/BRICK`,
  `/TETRA4`, `/SHELL`, `/SH3N`, `/TRUSS`, `/SPRING`, `/BEAM`, `/PART`,
  `/MAT/LAW1/2/27/36/42`, `/FAIL/JOHNSON`, `/FAIL/BIQUAD`,
  `/PROP/TYPE1/2/3/4/14`,
  `/BCS`, `/INIVEL`, `/GRAV`, `/CLOAD`, `/PLOAD`, `/IMPVEL`, `/IMPDISP`,
  `/ADMAS`, `/FUNCT`, `/GRNOD`, `/RWALL`, `/RBODY`, `/RBE2`, `/RBE3`,
  `/SECT`, `/INTER/TYPE2/7/11`, `/SURF`, `/LINE`, `/TH`, …) with
  `#include` support.
- **Elements**: 8-node solid (one-point integration + Flanagan–Belytschko
  hourglass control) with degenerated-brick→tetra conversion, 4-node
  constant-strain tetra, 4-node Belytschko–Tsay shell (membrane + bending +
  transverse shear, through-thickness integration, BLT84 stiffness hourglass
  control), 3-node C0 triangle shell, 2-node corotational Timoshenko beam
  (elastic or global-plasticity), 2-node truss, 2-node spring.
- **Materials**: LAW1 (linear elastic, hypoelastic Jaumann update), LAW2
  (Johnson–Cook elasto-plasticity with strain-rate hardening, 3D and
  plane-stress variants), LAW36 (tabulated plasticity from /FUNCT hardening
  curves with a strain-rate curve family), LAW27 (brittle tensile cracking
  for shells, fixed crack direction + unilateral damage), LAW42
  (Ogden/Mooney-Rivlin hyperelasticity for solids, with the nonlinear
  tangent sound speed feeding the stable time step).
- **Failure**: /FAIL/JOHNSON and /FAIL/BIQUAD damage criteria with full
  element deletion (per-layer bookkeeping for shells, deleted elements
  flagged in the VTK output as `OFF`), plus the eps_p_max thresholds of
  LAW2/LAW36.
- **Contact** (M4): /INTER/TYPE7 penalty node-to-surface contact with the
  Istf stiffness variants, constant/variable gap (Igap), self-impact and a
  voxel (bucket) broad phase; /INTER/TYPE2 tied contact (kinematic
  secondary-to-main gluing with co-rotating offsets — does no work by
  construction); /INTER/TYPE11 edge-to-edge penalty contact on /LINE edge
  sets. Contact fully respects /FAIL element deletion: segments and edges
  of deleted elements drop out and orphaned nodes stop being tracked, so
  crack faces behave physically.
- **Constraints & loads** (M5): /RBODY rigid bodies (starter-assembled
  mass/COG/inertia tensor; stable 6-DOF Newton–Euler update that
  conserves angular momentum by construction, exponential-map finite
  rotation; coexists with contact) and /RBE2 rigid links (pivot mode
  from the master's /BCS — physical pendulums), /RBE3 interpolation
  constraints (force distribution without stiffening), rigid walls with
  plane/sphere/cylinder geometry that can move (free with a mass —
  momentum-exact impulse exchange — or velocity-driven), /SECT
  section-force output, /PLOAD follower pressure, /IMPDISP imposed
  displacement, /ADMAS added mass, /INIVEL/AXIS initial spin.
- **Engine**: explicit central-difference integration, element/nodal stable
  time step (including the accumulated contact-spring stiffness), boundary
  conditions, initial/imposed velocities, gravity, concentrated loads,
  kinematic rigid walls, full energy-balance bookkeeping (contact work
  booked at the leapfrog-consistent midstep velocity).
- **Output**: listings, CSV time history (incl. /TH/SECT section
  resultants), VTK animation states.

## Repository layout

```
pyradioss/
├── common/       # shared low-level services (≈ OpenRadioss common_source/)
├── input/        # deck reader + keyword parsers (≈ starter/source/reader)
├── model/        # in-memory model (≈ Fortran modules/derived types)
├── starter/      # the Starter program        (≈ starter/source)
├── engine/       # the Engine program         (≈ engine/source)
├── elements/     # element kernels            (≈ engine/source/elements)
├── materials/    # material laws              (≈ engine/source/materials/mat)
├── failure/      # /FAIL damage criteria      (≈ engine/source/materials/fail)
├── contact/      # contact interfaces         (≈ engine/source/interfaces)
└── output/       # listings, TH, ANIM         (≈ engine/source/output)
examples/         # ready-to-run input decks
tests/            # pytest suite incl. analytic validations
```

## License

GPL-3.0 (see [LICENSE](LICENSE)) — inherited from files derived from
OpenRadioss, © Altair Engineering Inc. This repository is an independent
educational port and is not affiliated with Altair.
