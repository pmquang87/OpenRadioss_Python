# Porting guide — OpenRadioss (Fortran) → `pyradioss` (Python)

This document is the working map of the port: how the original source tree
translates to this repository, what is already ported, the conventions used,
and the roadmap for the remaining work. **Read this first if you plan to
extend the code.**

---

## 1. Architecture recap

OpenRadioss is two programs sharing data through a *restart file*:

```
            MYRUN_0000.rad                MYRUN_0001.rad
                 │                              │
                 ▼                              ▼
   ┌──────────────────────┐   restart   ┌──────────────────────┐
   │        STARTER        │ ─────────► │        ENGINE         │
   │  read + check + init  │  .rst      │  explicit time loop   │
   └──────────────────────┘             └──────────────────────┘
        MYRUN_0000.out                MYRUN_0001.out, T01, ANIM
```

The Engine's heart (`engine/source/engine/resol.F`, ~5000 lines) performs, per
cycle: element internal forces → contact forces → external loads → kinematic
conditions → acceleration/velocity/position update → time-step computation →
output. `pyradioss/engine/engine.py` reproduces exactly this sequence with the
same names in comments.

## 2. Source-tree map

| OpenRadioss (Fortran) | pyradioss (Python) | Notes |
|---|---|---|
| `starter/source/starter/lectur.F` (deck reading driver) | `pyradioss/starter/starter.py` | orchestration |
| `starter/source/reader/*` + `hm_reader` (Altair reader lib) | `pyradioss/input/deck_reader.py` | block/keyword lexer, `#include` |
| `starter/source/elements/reader` per-keyword `hm_read_*.F` | `pyradioss/input/starter_keywords.py` | one function per keyword |
| Fortran derived types / common blocks (`common_source/modules`) | `pyradioss/model/*.py` | dataclasses + NumPy arrays |
| `starter/source/initial_conditions`, `inimass` etc. | `pyradioss/starter/initialization.py` | lumped mass, volumes |
| restart write `starter/source/restart/ddsplit/wrrest.F` | `pyradioss/starter/restart.py` | pickle instead of binary |
| `engine/source/engine/resol.F` | `pyradioss/engine/engine.py` | main loop |
| `engine/source/engine/lectur.F` + `hm_read_*` (engine cards) | `pyradioss/input/engine_keywords.py` | `/RUN /DT /TFILE /ANIM …` |
| `engine/source/assembly/asspar*.F` | `pyradioss/engine/engine.py` (`np.add.at` scatter) | force assembly |
| `engine/source/constraints/general/bcs` | `pyradioss/engine/kinematics.py` | `/BCS`, `/IMPVEL` |
| `engine/source/constraints/general/rwall` | `pyradioss/engine/rigid_wall.py` | kinematic wall |
| `engine/source/elements/solid/solide/` (`sforc3.F`, `srota3.F`, `shour3.F`…) | `pyradioss/elements/solid_hexa8.py` | 1-pt + FB hourglass |
| `engine/source/elements/shell/coque/` (`cforc3.F`, `czforc3.F`…) | `pyradioss/elements/shell_bt4.py` | Belytschko–Tsay |
| `engine/source/elements/truss/` (`tforc3.F`) | `pyradioss/elements/truss.py` | |
| `engine/source/elements/spring/` (`rforc3.F`) | `pyradioss/elements/spring.py` | TYPE4 |
| `engine/source/materials/mat/mat001/sigeps01.F` | `pyradioss/materials/law01_elastic.py` | |
| `engine/source/materials/mat/mat002/sigeps02.F` | `pyradioss/materials/law02_johnson_cook.py` | |
| `engine/source/interfaces/inter3d/` (TYPE7: `i7main*.F`) | `pyradioss/contact/inter_type7.py` | penalty node↔segment |
| `engine/source/output/` (`ecrit.F`, `sortie_main.F`, TH, ANIM) | `pyradioss/output/*.py` | CSV + VTK |
| `common_source/` (constants, tables) | `pyradioss/common/*.py` | |

## 3. Conventions used in this port

1. **Every module docstring names its Fortran origin** and explains the theory
   (with textbook references: Belytschko–Liu–Moran for the corotational shell
   and hourglass control, Wilkins for the radial return, etc.).
2. **Structure of arrays, not array of structures.** Elements of one type are
   processed as NumPy arrays over the whole group — the Python analogue of the
   Fortran element-group loops (`NGROUP`/`MVSIZ` blocks). Scalar per-element
   Python loops are avoided except in setup code.
3. **IDs vs indices.** Decks use arbitrary user IDs; the Starter converts all
   of them to dense 0-based indices (the original does the same via `USR2SYS`
   tables). Arrays in the Engine are indexed, never keyed by user ID; the
   mapping is kept for output.
4. **Explicit is explicit.** No hidden state: each cycle recomputes forces
   from (x, v, internal state). All mutable element state (stress, plastic
   strain, hourglass forces…) lives in per-group `state` dictionaries created
   by the Starter.
5. **Units**: consistent-unit philosophy identical to Radioss — the solver
   never converts units; the deck must be consistent (e.g. mm, ms, kg → kN).
6. **Failure behaviour**: readable exceptions from the Starter for model
   errors (`StarterError`, with the same "** ERROR" flavour as the listing);
   the Engine protects against divergence with an energy-error stop criterion
   like the original (`/STOP` defaults).

## 4. Feature matrix (Milestone 1)

Legend: ✅ ported (functional), 🟡 simplified (functional but reduced options), ❌ not yet.

### Input keywords

| Keyword | Status | Notes |
|---|---|---|
| `/BEGIN`, `/END`, `/TITLE` | ✅ | |
| `#include` | ✅ | recursive |
| `/NODE` | ✅ | |
| `/BRICK` | ✅ | 8-node hexa (degenerate 6/4-node not yet) |
| `/SHELL` | ✅ | 4-node (`/SH3N` ❌) |
| `/TRUSS`, `/SPRING` | ✅ | |
| `/PART`, `/SUBSET` | ✅ / ❌ | |
| `/MAT/LAW1` (`/MAT/ELAST`) | ✅ | |
| `/MAT/LAW2` (`/MAT/PLAS_JOHNS`) | ✅ | εp-rate & hardening; temperature term ❌ |
| `/PROP/TYPE1` (`SHELL`) | 🟡 | thickness, N integration points, hourglass coeff (Ishell fixed = BT) |
| `/PROP/TYPE2` (`TRUSS`) | ✅ | area |
| `/PROP/TYPE4` (`SPRING`) | 🟡 | linear k, c, mass |
| `/PROP/TYPE14` (`SOLID`) | 🟡 | qa/qb bulk viscosity, hourglass coeff (Isolid fixed = 1-pt+FB) |
| `/BCS` | ✅ | translation + rotation fixities |
| `/INIVEL/TRA`, `/INIVEL/AXIS` | ✅ / ❌ | |
| `/IMPVEL` | ✅ | via `/FUNCT`, fixed direction |
| `/GRAV` | ✅ | |
| `/CLOAD` | ✅ | |
| `/FUNCT` | ✅ | piecewise-linear tables |
| `/GRNOD/NODE`, `/GRNOD/PART`, `/GRNOD/BOX` | ✅ | |
| `/BOX/RECTA` | ✅ | |
| `/RWALL/PLANE` | 🟡 | infinite plane, sliding or tied; moving wall ❌ |
| `/INTER/TYPE7` | 🟡 | penalty, constant stiffness option, Coulomb friction; full Istf/Igap variants ❌ |
| `/SURF/PART`, `/SURF/SEG` | ✅ | for contact |
| `/TH/NODE`, `/TH/PART` | ✅ | |
| Engine: `/RUN`, `/VERS`, `/TFILE`, `/ANIM/DT`, `/ANIM/VECT|ELEM`, `/DT`, `/PRINT`, `/STOP` | ✅/🟡 | `/DT/NODA/CST` (mass scaling) ❌ |

### Solver features

| Feature | Status |
|---|---|
| Explicit central-difference integration | ✅ |
| Element time step with scale factor, bulk viscosity | ✅ |
| Exact per-element eigenvalue bound on dt (port improvement: the classic lc/c estimate is up to ~40% above the true one-point-element stability limit; the Starter computes the exact 6×6/3×3 eigenvalue correction once per element — see `solid_hexa8._exact_dt_factor`) | ✅ |
| Solid hexa8, 1-point, Flanagan–Belytschko hourglass control | ✅ |
| Belytschko–Tsay 4-node shell (memb/bend/shear + hourglass) | ✅ |
| Truss, linear spring | ✅ |
| Jaumann objective stress update | ✅ |
| LAW1, LAW2 (3D + plane stress) | ✅ |
| Rigid wall (kinematic, slide/tied) | ✅ |
| TYPE7-style penalty contact + friction | 🟡 |
| Energy balance (int/kin/hourglass/contact/external work), error % | ✅ |
| T01 time history, ANIM (as VTK), listings | ✅ |
| MPI/domain decomposition, SMP | ❌ (out of scope) |

## 5. Roadmap (next milestones)

1. **M2 — element completeness**: `/SH3N` (C0 triangle), 4-node tetra,
   fully-integrated solid (Isolid=17 equivalent), QEPH shell, beams
   (`/PROP/TYPE3`), degenerate bricks.
2. **M3 — materials**: LAW36 (tabulated plasticity), LAW27 (brittle),
   LAW42 (Ogden), /FAIL cards (Johnson–Cook failure, biquad), EOS.
3. **M4 — contact**: TYPE7 full options (Igap, Istf variants, self-impact),
   TYPE2 tied, TYPE11 edge-to-edge.
4. **M5 — constraints & loads**: /RBODY, /RBE2/RBE3, /MPC, /SECT, moving and
   spherical/cylindrical rigid walls, /PLOAD, /IMPDISP.
5. **M6 — engine niceties**: /DT/NODA/CST mass scaling, restarts
   (`_0002.rad` chaining), /STATE, sensors, /DAMP, ALE/CFD (long term).
6. **M7 — performance**: optional numba/JAX backends behind the same API.

## 6. Validation strategy

`tests/` contains two layers:

* **Unit tests** — parser round-trips, single-element material/element checks
  (e.g. one brick under uniaxial strain must return the analytic stress).
* **Analytic validations** — full starter+engine runs compared to closed-form
  results: longitudinal wave speed in a bar, cantilever/plate vibration
  frequency, Johnson–Cook uniaxial yield curve, energy conservation of a
  block bouncing on a rigid wall.

When porting new features, always add at least one analytic validation — this
is how the port stays trustworthy without bit-for-bit comparison against the
Fortran solver (which the different output formats make impractical).
