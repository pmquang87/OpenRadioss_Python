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
| `engine/source/elements/solid/solide4/` (`s4forc3.F`…) | `pyradioss/elements/solid_tetra4.py` | constant-strain tetra |
| `engine/source/elements/shell/coque/` (`cforc3.F`, `czforc3.F`…) | `pyradioss/elements/shell_bt4.py` | Belytschko–Tsay, BLT84 stiffness hourglass |
| `engine/source/elements/sh3n/coque3n/` (`c3forc3.F`…) | `pyradioss/elements/shell_tri3.py` | C0 triangle |
| `engine/source/elements/beam/` (`pforc3.F`, `pdefo3.F`…) | `pyradioss/elements/beam_type3.py` | corotational Timoshenko |
| `engine/source/elements/truss/` (`tforc3.F`) | `pyradioss/elements/truss.py` | |
| `engine/source/elements/spring/` (`rforc3.F`) | `pyradioss/elements/spring.py` | TYPE4 |
| `engine/source/materials/mat/mat001/sigeps01.F` | `pyradioss/materials/law01_elastic.py` | |
| `engine/source/materials/mat/mat002/sigeps02.F` | `pyradioss/materials/law02_johnson_cook.py` | |
| `engine/source/materials/mat/mat027/sigeps27c.F` | `pyradioss/materials/law27_brittle.py` | shells only, like the original |
| `engine/source/materials/mat/mat036/sigeps36.F` (+ `36c`) | `pyradioss/materials/law36_tabulated.py` | |
| `engine/source/materials/mat/mat042/sigeps42.F` | `pyradioss/materials/law42_ogden.py` | solids; returns its own SOUNDSP |
| `engine/source/materials/fail/johnson_cook/`, `fail/biquad/` | `pyradioss/failure/` | /FAIL cards + GBUF%OFF element deletion |
| `engine/source/elements/beam/pmat3.F` (global plasticity) | `pyradioss/elements/beam_type3.py` | LAW2 resultant-space return |
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

## 4. Feature matrix (Milestones 1–3)

Legend: ✅ ported (functional), 🟡 simplified (functional but reduced options), ❌ not yet.

### Input keywords

| Keyword | Status | Notes |
|---|---|---|
| `/BEGIN`, `/END`, `/TITLE` | ✅ | |
| `#include` | ✅ | recursive |
| `/NODE` | ✅ | |
| `/BRICK` | ✅ | 8-node hexa; 4-distinct-node degenerates auto-convert to `/TETRA4`, penta/pyramid rejected with a clear error |
| `/TETRA4` | ✅ | 4-node constant-strain tetra |
| `/SHELL` | ✅ | 4-node Belytschko–Tsay |
| `/SH3N` | ✅ | 3-node C0 triangle (plain C0; DKT-flavoured Ish3n variants ❌) |
| `/TRUSS`, `/SPRING` | ✅ | |
| `/BEAM` | ✅ | N1 N2 + orientation node N3 |
| `/PART`, `/SUBSET` | ✅ / ❌ | |
| `/MAT/LAW1` (`/MAT/ELAST`) | ✅ | |
| `/MAT/LAW2` (`/MAT/PLAS_JOHNS`) | ✅ | εp-rate & hardening, eps_p_max element deletion (M3), beams via the global-plasticity model (M3); temperature term ❌ |
| `/MAT/LAW27` (`/MAT/PLAS_BRIT`) | 🟡 | brittle tensile cracking with fixed crack direction, unilateral damage, layer rupture + element deletion; the plastic block of the original ❌ (shells only, like the original) |
| `/MAT/LAW36` (`/MAT/PLAS_TAB`) | ✅ | tabulated hardening from /FUNCT curves, strain-rate curve family (linear rate interpolation), eps_p_max deletion; Fsmooth/Chard/Fcut and Fscale ❌ |
| `/MAT/LAW42` (`/MAT/OGDEN`) | 🟡 | Ogden/Mooney-Rivlin, incompressible + K(J-1) bulk penalty, exact F from initial gradients, **nonlinear sound speed feeds the time step** (the law stiffens with stretch — verified by a long /DT 0.9 hold at λ≈2); solids only, no shell variant, no Prony viscosity |
| `/FAIL/JOHNSON` | ✅ | D1–D4 + rate term; thermal D5 ❌; Ifail_sh 1/2; element deletion (stress zeroing, dt release, OFF in ANIM) |
| `/FAIL/BIQUAD` | 🟡 | explicit c1–c5 input (two-parabola εf(σ*) fit); M-flag material presets and S-flag ❌ |
| `/PROP/TYPE1` (`SHELL`) | 🟡 | thickness, N integration points, hourglass coeffs (Ishell fixed = BT for quads, C0 for `/SH3N`) |
| `/PROP/TYPE2` (`TRUSS`) | ✅ | area |
| `/PROP/TYPE3` (`BEAM`) | 🟡 | A, Iyy, Izz, Ixx; Timoshenko with full-section shear (no shear factor / Ishear variants), LAW1 only |
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
| Exact per-element eigenvalue bound on dt (port improvement: the classic lc/c estimate is up to ~40% above the true one-point-element stability limit; the Starter computes the exact eigenvalue correction once per element — 6×6 for solids, membrane **and** bending/shear branches for shells since M2, the full 12×12 for beams — see `solid_hexa8._exact_dt_factor`, `shell_bt4._bend_shear_omega2`, `beam_type3._exact_dt`) | ✅ |
| Solid hexa8, 1-point, Flanagan–Belytschko hourglass control | ✅ |
| Solid tetra4, constant strain (no hourglass modes; plain formulation — nodal-pressure Itetra variants ❌) | ✅ |
| Belytschko–Tsay 4-node shell (memb/bend/shear, BLT84 **stiffness** hourglass control since M2 — M1's viscous form artificially damped coarse dynamic bending) | ✅ |
| C0 3-node triangle shell (CST membrane + Mindlin plate, no hourglass modes) | ✅ |
| Corotational Timoshenko beam (axial/2×shear/torsion/2×bending resultants) | ✅ (LAW1 elastic; LAW2 via the global resultant-plasticity model since M3 — yields at exactly W·σy, no elastic-core spread to the 1.5·W·σy hinge; fiber-integrated TYPE18 beam is a roadmap item) |
| Degenerated /BRICK → tetra conversion, penta/pyramid clear check | 🟡 |
| Truss, linear spring | ✅ |
| Jaumann objective stress update | ✅ |
| LAW1, LAW2 (3D + plane stress) | ✅ |
| LAW36 tabulated plasticity (3D + plane stress, rate curve family) | ✅ |
| LAW27 brittle cracking (fixed smeared crack, unilateral damage) | 🟡 (elastic-brittle; original's plastic block ❌) |
| LAW42 Ogden hyperelasticity (total-strain from exact F, nonlinear SOUNDSP → dt) | 🟡 (solids only) |
| /FAIL element deletion plumbing (per-layer for shells, GBUF%OFF, deleted elements keep mass, drop stress/hourglass/dt claim, OFF field in ANIM, deletion count in the listing) | ✅ |
| Equations of state (/EOS) for solids | ❌ (deliberately deferred — see roadmap M6: needs energy-dependent pressure integration per element, out of M3's scope; pressure is currently always the law's own, i.e. linear K·tr(ε) for LAW1/2/36) |
| Rigid wall (kinematic, slide/tied) | ✅ |
| TYPE7-style penalty contact + friction | 🟡 |
| Energy balance (int/kin/hourglass/contact/external work), error % | ✅ |
| T01 time history, ANIM (as VTK), listings | ✅ |
| MPI/domain decomposition, SMP | ❌ (out of scope) |

## 5. Roadmap (next milestones)

1. **M2 — element completeness** ✅ (done): `/SH3N` (C0 triangle), 4-node
   tetra, beams (`/BEAM` + `/PROP/TYPE3`, elastic), degenerate-brick
   handling, shell hourglass upgraded from viscous to BLT84 stiffness
   type, exact-dt coverage extended to the shell bending/shear branch and
   the beam 12×12 eigenproblem. Deferred to later milestones:
   fully-integrated solid (Isolid=17 equivalent), QEPH shell, DKT18
   triangle, plastic beams (global plasticity model), nodal-pressure
   tetra variants.
2. **M3 — materials** ✅ (done): LAW36 tabulated plasticity (/FUNCT
   hardening curves + strain-rate curve family), LAW27 brittle cracking
   (shells), LAW42 Ogden hyperelasticity (solids, with the law feeding
   its **nonlinear sound speed** into the element time step — the M2
   time-step lesson applied to a stiffening material, proven by a long
   /DT 0.9 hold at λ≈2), /FAIL/JOHNSON and /FAIL/BIQUAD with full
   element-deletion plumbing (per-layer shell failure, OFF in ANIM,
   deletion messages), eps_p_max deletion for LAW2/LAW36, and the
   deferred-from-M2 plastic beams (LAW2 global resultant plasticity).
   Deferred out of M3, explicitly:
   * **/EOS (equations of state)** — energy-dependent pressure
     integration (E-p coupling per element, relative-volume state) is a
     solver-loop change, not just a material: it moves to M6 together
     with the thermal Johnson–Cook terms it usually accompanies;
   * thermal terms of LAW2//FAIL/JOHNSON (no thermal solution yet);
   * the plastic block of LAW27, shell LAW42, LAW42 Prony viscosity;
   * /FAIL/BIQUAD M-flag presets and S-flag; LAW36 Fsmooth/Fscale;
   * fiber-integrated beams (/PROP/TYPE18) for true plastic-hinge
     spread.
3. **M4 — contact**: TYPE7 full options (Igap, Istf variants, self-impact),
   TYPE2 tied, TYPE11 edge-to-edge.
4. **M5 — constraints & loads**: /RBODY, /RBE2/RBE3, /MPC, /SECT, moving and
   spherical/cylindrical rigid walls, /PLOAD, /IMPDISP.
5. **M6 — engine niceties**: /DT/NODA/CST mass scaling, restarts
   (`_0002.rad` chaining), /STATE, sensors, /DAMP, /EOS + thermal
   material terms (deferred from M3), ALE/CFD (long term).
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
