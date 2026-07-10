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
| `engine/source/constraints/general/bcs` + `impvel/fixvel.F` | `pyradioss/engine/kinematics.py` | `/BCS`, `/IMPVEL`, `/IMPDISP` |
| `engine/source/constraints/general/rwall` (`rgwal0/s/c/t.F`) | `pyradioss/engine/rigid_wall.py` | kinematic wall: plane/sphere/cylinder, moving (M5) |
| `starter/.../rbody/hm_read_rbody.F`, `rbyini.F` + `engine/.../rbody/rbyfor.F`, `rbycor.F` (and `rbe2/`) | `pyradioss/engine/rigid_body.py` + `starter/initialization.py` (`initialize_rigid_bodies`) | `/RBODY`, `/RBE2` (M5) |
| `starter/.../rbe3/hm_read_rbe3.F` + `engine/.../rbe3/rbe3f.F`, `rbe3v.F` | `pyradioss/engine/rbe3.py` | `/RBE3` interpolation constraint (M5) |
| `engine/source/loads/general/pload/pload.F` | `pyradioss/engine/kinematics.py` (`external_forces`) | `/PLOAD` follower pressure (M5) |
| `engine/source/tools/sect/` (`section.F`, `forint.F`) | `pyradioss/engine/sections.py` | `/SECT` via the side-sum identity (M5) |
| `starter/source/tools/admas/` | `pyradioss/starter/initialization.py` | `/ADMAS` (M5) |
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
| `engine/source/interfaces/int07/` (`i7dst3.F`, `i7for3.F`) + `intsort/i7buce.F` | `pyradioss/contact/inter_type7.py` | penalty node↔segment, voxel broad phase |
| `engine/source/interfaces/int02/` (`i2for3.F`, `i2vit3.F`) | `pyradioss/contact/inter_type2.py` | tied contact (kinematic) |
| `engine/source/interfaces/int11/` (`i11dst3.F`, `i11for3.F`) | `pyradioss/contact/inter_type11.py` | penalty edge↔edge |
| `starter/source/interfaces/inter3d1/` (`i7sti3.F`, `i11sti3.F`, gap setup) | `pyradioss/contact/stiffness.py` | element-based penalty stiffness + gaps |
| interface `IDEL` bookkeeping vs `GBUF%OFF` | `pyradioss/contact/tracking.py` | deleted elements drop out of contact |
| `starter/source/model/sets/hm_read_lines.F` (IGRSLIN) | `pyradioss/starter/initialization.py` (`resolve_lines`) | /LINE edge sets |
| `engine/source/output/` (`ecrit.F`, `sortie_main.F`, TH, ANIM) | `pyradioss/output/*.py` | CSV + VTK |
| `engine/source/time_step/` (`dtnoda.F`, STIFN accumulation) | `pyradioss/engine/mass_scaling.py` | `/DT/NODA[/CST]` nodal dt + mass scaling (M6) |
| `engine/source/output/restart/` (`wrrestp.F`, `rdresb.F`) | `pyradioss/starter/restart.py` + engine resume | engine restarts, `_0002.rad` chaining, /STATE (M6) |
| `engine/source/assembly/damping*.F` | `pyradioss/engine/damping.py` | `/DAMP` mass damping (M6) |
| `starter+engine/source/tools/sensor/` | `pyradioss/engine/sensors.py` | `/SENSOR/TIME`, `/SENSOR/DISP` (M6) |
| `starter+engine/source/constraints/general/mpc/` | `pyradioss/engine/mpc.py` | `/MPC` Lagrange treatment (M6) |
| `starter/source/materials/eos/` + `engine/source/materials/eos/eosmain.F` | `pyradioss/materials/eos.py` + solid kernels | `/EOS` polynomial & ideal gas, implicit E-p (M6) |
| `common_source/` (constants, tables) | `pyradioss/common/*.py` | |
| — (OpenRadioss speed = compiled Fortran + OpenMP/MPI, out of scope) | `pyradioss/accel/` | M7: optional numba backend behind the same kernel API (see the package docstring for architecture + parity contract) |
| — | `pyradioss/common/fastmath.py` | M7: small-array NumPy primitives (bitwise-documented replacements for np.cross / norm / det / inv / add.at) |
| — | `tools/benchmark.py` | M7: NumPy vs numba wall-clock benchmark over the examples |

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

## 4. Feature matrix (Milestones 1–6)

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
| `/MAT/LAW2` (`/MAT/PLAS_JOHNS`) | ✅ | εp-rate & hardening, eps_p_max element deletion (M3), beams via the global-plasticity model (M3); since M6 the full thermal terms in the ADIABATIC approximation (optional card 6 `m T_melt rho_Cp T_i`): plastic work heats the point, dT = σy·dεp/ρCp, and T*^m softens the yield — per-point temperature state on solids and shell layers |
| `/MAT/LAW27` (`/MAT/PLAS_BRIT`) | 🟡 | brittle tensile cracking with fixed crack direction, unilateral damage, layer rupture + element deletion; the plastic block of the original ❌ (shells only, like the original) |
| `/MAT/LAW36` (`/MAT/PLAS_TAB`) | ✅ | tabulated hardening from /FUNCT curves, strain-rate curve family (linear rate interpolation), eps_p_max deletion; Fsmooth/Chard/Fcut and Fscale ❌ |
| `/MAT/LAW42` (`/MAT/OGDEN`) | 🟡 | Ogden/Mooney-Rivlin, incompressible + K(J-1) bulk penalty, exact F from initial gradients, **nonlinear sound speed feeds the time step** (the law stiffens with stretch — verified by a long /DT 0.9 hold at λ≈2); solids only, no shell variant, no Prony viscosity |
| `/FAIL/JOHNSON` | ✅ | D1–D4 + rate term; thermal D5 since M6 (needs the LAW2 thermal card, warned otherwise); Ifail_sh 1/2; element deletion (stress zeroing, dt release, OFF in ANIM) |
| `/FAIL/BIQUAD` | 🟡 | explicit c1–c5 input (two-parabola εf(σ*) fit); M-flag material presets and S-flag ❌ |
| `/PROP/TYPE1` (`SHELL`) | 🟡 | thickness, N integration points, hourglass coeffs (Ishell fixed = BT for quads, C0 for `/SH3N`) |
| `/PROP/TYPE2` (`TRUSS`) | ✅ | area |
| `/PROP/TYPE3` (`BEAM`) | 🟡 | A, Iyy, Izz, Ixx; Timoshenko with full-section shear (no shear factor / Ishear variants), LAW1 only |
| `/PROP/TYPE4` (`SPRING`) | 🟡 | linear k, c, mass |
| `/PROP/TYPE14` (`SOLID`) | 🟡 | qa/qb bulk viscosity, hourglass coeff (Isolid fixed = 1-pt+FB) |
| `/BCS` | ✅ | translation + rotation fixities; on a rigid-body master it becomes a body-level condition (full 111 translations = pivot) |
| `/INIVEL/TRA`, `/INIVEL/AXIS` | ✅ / ✅ | AXIS since M5: rigid-rotation field ω·d×(x−P) (the way a spinning /RBODY is set up); the translational Vt fields of the full AXIS card via an extra /INIVEL/TRA |
| `/IMPVEL` | ✅ | via `/FUNCT`, fixed direction; on a rigid-body master it drives the body (moving rigid die); on a moving-wall node it drives the wall |
| `/IMPDISP` | ✅ | M5 — kinematic like /IMPVEL but enforced at the *position* level (the node lands exactly at x0 + d(t), no velocity-integration drift); work booked from the constraint impulse like /IMPVEL |
| `/GRAV` | ✅ | |
| `/CLOAD` | ✅ | optional /SENSOR gating since M6 (waits for the sensor, then follows f(t − t_fire)) |
| `/EOS/POLYNOMIAL`, `/EOS/IDEAL-GAS` | ✅ | M6 — attaches to LAW1/2/36 like /FAIL; the EOS pressure replaces the law's (deviator stays with the law); implicit E-p coupling per element with relative-volume state, viscous shock heating into E, EOS sound speed feeds the time step (see materials/eos.py) |
| `/DAMP` | 🟡 | M6 — Rayleigh MASS damping (α), Tstart/Tstop window; applied as the exact per-cycle integrating factor (unconditionally stable, claims no dt) with the dissipation booked exactly into the DE ledger; the stiffness (β) branch ❌ (needs K·v products) |
| `/SENSOR/TIME`, `/SENSOR/DISP` | ✅ | M6 — latching sensors gating /CLOAD, /PLOAD and /INTER/TYPE7/11 (fire time survives restarts); other sensor types ❌ |
| `/MPC` | ✅ | M6 (deferred from M5) — general linear rows on translations (+ rotations where the node carries inertia), solved together via the nc×nc Lagrange system on accelerations + a velocity cleanup; zero work by construction, /BCS-fixed DOFs act as ground; redundant row sets fall back to least-squares multipliers |
| `/PLOAD` | ✅ | M5 — follower pressure on a /SURF (current segment normal, p·A lumped to corners, triangles 1/3); segments of /FAIL-deleted elements stop carrying pressure |
| `/ADMAS` | 🟡 | M5 — per-node added mass (Radioss type-0 semantics only); also the way a moving /RWALL gets its inertia |
| `/FUNCT` | ✅ | piecewise-linear tables |
| `/GRNOD/NODE`, `/GRNOD/PART`, `/GRNOD/BOX` | ✅ | |
| `/BOX/RECTA` | ✅ | |
| `/RWALL/PLANE`, `/RWALL/SPHER`, `/RWALL/CYL` | ✅ | M5: three geometries, sliding/tied/friction, and MOVING walls tied to a carrier node (free with /ADMAS+/INIVEL — impulses react on the node, momentum-exact; or /IMPVEL-driven — the drive absorbs the reaction and books external work). Walls do not rotate; containment (nodes inside a sphere/cyl) ❌ |
| `/RBODY` | ✅ | M5 — master + slave node set as one rigid body: starter assembles mass/COG/inertia tensor (point masses + nodal inertias + added Mass/Jxx-Jzz), ICoG=1 master relocation; engine integrates the 6-DOF Newton-Euler EOM (angular-momentum update + exponential-map rotation — L conserved by construction). Sensors, skew/spherical inertia, IKREM, surface envelope ❌ |
| `/RBE2` | 🟡 | M5 — rigid link: same mechanics with a structural master kept at its own position; full 6-DOF tie only (per-DOF flags ❌) |
| `/RBE3` | 🟡 | M5 — interpolation constraint (least-squares rigid fit + its virtual-work dual force distribution — no stiffening, no spurious work); one master group with uniform weights (per-set weights/DOF flags ❌) |
| `/SECT` | 🟡 | M5 — section force/moment time history through a cut, computed by the side-sum identity over one side's node set (see engine/sections.py); output via /TH/SECT. The frame/element-set input of the full card ❌ |
| `/INTER/TYPE7` | ✅ | penalty node↔surface (M4): Istf 0–5 stiffness variants, Igap 0/1 (constant / variable from shell thicknesses) with Gap_min/Gap_max, self-impact (`grnod_ID = 0`), Coulomb friction, voxel broad phase; /SENSOR gating since M6 (the Tstart/Tstop role); Inacti, Igap 2/3, Ifric>0 friction models ❌ |
| `/INTER/TYPE2` | 🟡 | tied contact (M4): kinematic secondary→main gluing, constant-weight projection with co-rotating offset, lumped mass/force transfer, deletion release; rotational-DOF tying (Spotflag) and offset moment redistribution ❌ |
| `/INTER/TYPE11` | ✅ | edge↔edge penalty (M4): /LINE edge sets, Istf/Igap as TYPE7, exact segment-segment closest points; parallel-overlap force distribution simplified to the closest-point pair |
| `/LINE/SURF`, `/LINE/SEG` | ✅ | edge sets for TYPE11 (M4), with element provenance for deletion |
| `/SURF/PART`, `/SURF/SEG` | ✅ | for contact; since M4 every segment carries its parent-element provenance (deletion, stiffness, gap) |
| `/TH/NODE`, `/TH/PART`, `/TH/SECT` | ✅ | SECT since M5: FX FY FZ MX MY MZ |
| Engine: `/RUN`, `/VERS`, `/TFILE`, `/ANIM/DT`, `/ANIM/VECT|ELEM`, `/DT`, `/PRINT`, `/STOP` | ✅/🟡 | |
| Engine: `/DT/NODA`, `/DT/NODA/CST` | ✅ | M6 — nodal time step dt_i = √(2Mᵢ/Kᵢ) with the element stiffness derived from the kernels' own dt claims (kᵢᵉ = 2mᵢᵉ/dt_e², = the element dt on uniform meshes) and the contact NEAR-spring stiffness accumulated in; CST adds mass to hold dT_min — added mass, its momentum and its kinetic energy are tracked, reported (1%-step announcements + termination summary) and the energy enters the balance; prescribed nodes (rigid-body members, tied secondaries, RBE3 dependents) are excluded (see engine/mass_scaling.py) |
| Engine: `/STATE/DT` | 🟡 | M6 — periodic full restart snapshots refreshing `RunName_{nn}.rst` (crash recovery / early chaining); the original's .sta ASCII format ❌ (the pickle restart plays that role) |
| Engine restart chaining (`RunName_0002.rad`) | ✅ | M6 — the Engine ALWAYS writes `RunName_{nn}.rst` at termination; run nn+1 resumes it: clock/ledgers/next-dt/output numbering restored, rigid-body R & L and sensor fire-times carried, everything else deliberately reconstructed from the model arrays (tied projections, contact candidates, fix masks). Acceptance: a chained run reproduces the unchained one exactly — same cycle count, state to round-off (asserted for a spring oscillator and a tumbling /RBODY) |

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
| Equations of state (/EOS) for solids (M6): polynomial + ideal gas, implicit E-p update per element (closed form — p linear in E), relative-volume state, q-work shock heating into E, EOS sound speed → dt; validated against the exact ideal-gas isentrope pV^γ, the Rankine–Hugoniot identity and a quasi-static piston compression | ✅ |
| LAW2 adiabatic thermal terms + /FAIL/JOHNSON D5 (M6): per-point temperature rise from plastic work, (1−T*^m) softening (validated against the closed-form heating ODE and the softened flow stress) | ✅ |
| /DT/NODA/CST mass scaling with honest added-mass accounting (M6) | ✅ |
| /DAMP mass damping via the exact integrating factor, dissipation booked from the KE identity (M6) | ✅ |
| /SENSOR gating with latching + time-shifted load curves (M6) | ✅ |
| /MPC general linear constraints (Lagrange on accelerations + velocity cleanup, zero work by construction) (M6) | ✅ |
| Engine restart chaining, bit-reproducing the unchained run (M6) | ✅ |
| Numerical-dissipation ledger EN (M6): the exact internal-force midstep work measured per cycle against the state-function bookings — closes the balance under damped barely-resolved ringing (the M1-era qb misbooking), stays negligible on healthy runs (asserted), and its strongly-negative excursions are the new energy-INJECTION divergence stop (|ERR| alone is blind to instability once EN is in the balance) | ✅ |
| Rigid wall (kinematic, slide/tied/friction; plane, sphere, cylinder; fixed, free-with-mass, velocity-driven — M5) | ✅ |
| Rigid bodies /RBODY + /RBE2 (6-DOF Newton-Euler: gather → L += T dt → w = (R J0 Rᵀ)⁻¹L → rigid scatter → exponential-map placement; exact L conservation for torque-free bodies, pivot mode from master /BCS, /IMPVEL body drive; slaves coexist with contact and the TYPE2 effective-mass machinery; deletion never changes the inertia — masses stay; elements interior to a body carry exactly zero strain because the enforcement re-scatters the rigid velocity field at the placed positions) | ✅ |
| /RBE3 interpolation constraint (weighted least-squares rigid fit + dual force distribution — transmits force and moment exactly, adds no stiffness, does no work; lumped mass transfer like TYPE2) | 🟡 (uniform weights) |
| /SECT section resultants (side-sum identity: element self-equilibrium cancels everything interior to the side, leaving the through-cut force/moment; no per-side reassembly needed) | ✅ |
| Free moving wall: IMPLICIT joint momentum solve of the carrier node with all hit nodes (an M5 lesson: correcting the nodes against the pre-recoil wall velocity and recoiling afterwards feeds every riding node a one-cycle-stale, faster wall — an energy injection that does **not** vanish with dt, measured at +33% before the fix; the joint 3×3 solve reproduces the exact perfectly-inelastic collision in one cycle, asserted by a closed-form test) | ✅ |
| Kinematic-wall energy booking on the per-node injection identity U = ΔKE − f·v_old dt (books the arrest, the within-cycle acquired velocity, tied drag and friction in one expression; free-wall carrier KE change booked exactly from the joint solve; driven-wall injection booked as external work) | ✅ |
| TYPE7 penalty contact + friction (Istf variants, Igap, self-impact, voxel search) | ✅ |
| TYPE2 tied contact (kinematic; zero-work by construction — asserted in tests) | ✅ (translations; no rotation tying) |
| TYPE11 edge-to-edge penalty contact | ✅ |
| Contact ⇄ element deletion (M3⇄M4): segments/edges of `off == 0` elements drop out per cycle, secondary nodes with no surviving element stop being tracked, tied pairs release | ✅ |
| Interface time step: static node-on-spring bound + per-cycle accumulation of NEAR candidate spring stiffness per node (dt ≤ √(2m/ΣK) — springs stack at corners and in self-impact; proven by /DT 0.9 long-run impacts for Istf 0/2/5) | ✅ |
| Contact energy booked at the leapfrog midstep velocity (an M4 lesson: booking f·v at the pre-update velocity leaves a positive-definite f²dt²/2m residual per cycle that reads as energy creation when penalty springs dominate the scale — the midstep booking closes the balance to round-off) | ✅ |
| Energy balance (int/kin/hourglass/contact/external work), error % | ✅ |
| T01 time history, ANIM (as VTK), listings | ✅ |
| M7 NumPy fast paths (`common/fastmath.py`: formula-identical cross/norm, bincount scatter assembly, cofactor 3×3 det/inv; fused shell rate/hourglass matmuls; LAW36 exact-fixed-point Newton exit) | ✅ |
| Optional numba backend (`pyradioss/accel`): jit mirrors of the measured hotspots — hexa8 pre/post, BT4 shell pre/post, TYPE7 narrow phase — behind the same kernel API, `PYRADIOSS_BACKEND=numba` or `pyradioss-engine -backend numba`, NumPy fallback with a warning when numba is absent; parity asserted at kernel level and on full runs, restart chain bit-match asserted under numba (M7) | ✅ |
| JAX backend | ❌ (deferred — see the M7 roadmap note) |
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
3. **M4 — contact** ✅ (done): TYPE7 full options — Istf 0–5 stiffness
   variants from the element formulas of i7sti3 (shell 0.5·E·t, solid
   B·A²/V), Igap constant/variable gaps with Gap_min/Gap_max, self-impact
   (grnod = 0), and a voxel broad phase replacing M1's all-pairs box
   test; TYPE2 tied contact (kinematic constraint: constant-weight
   projection with a co-rotating offset frame, lumped mass/force
   transfer — momentum-exact and zero-work by construction, both
   asserted); TYPE11 edge-to-edge penalty on /LINE edge sets (exact
   segment-segment closest points). Contact ⇄ /FAIL deletion is fully
   plumbed through per-segment element provenance: crack faces stop
   carrying contact, orphaned nodes stop being tracked, tied pairs
   release (the notched-plate example now runs a self-impact interface
   through full ligament tearing at ~0% energy error). Two hard-won
   solver lessons are recorded in the code: the interface dt must see
   the SUM of the candidate spring stiffnesses per node (springs stack
   at corners/self-impact — i7's STIFN accumulation), and contact work
   must be booked at the leapfrog midstep velocity or the balance reads
   spurious energy creation. Deferred out of M4, explicitly:
   * Inacti initial-penetration treatments and the stiffening
     K·p/(gap−p) near-crossing guard of the original force law;
   * Igap 2/3 (mesh-size-scaled gaps), Tstart/Tstop, sensors,
     Ifric > 0 friction models, thermal contact;
   * TYPE2 rotational-DOF tying (Spotflag) and the moment
     redistribution of offset ties; penalty-formulation TYPE2;
   * TYPE19/24/25 style combined interfaces;
   * parallel-edge overlap force distribution for TYPE11 (resultant is
     right, distribution acts at the closest-point pair).
4. **M5 — constraints & loads** ✅ (done): /RBODY and /RBE2 rigid bodies
   (starter-assembled mass/COG/inertia tensor; a 6-DOF Newton-Euler
   engine update that integrates the ANGULAR MOMENTUM — not the spin —
   and rotates the body frame with the exponential map, so a torque-free
   body conserves L by construction and the finite rotation is stable at
   any step; pivot mode when the master's /BCS clamps all translations —
   the physical-pendulum configuration, inertia transported by
   parallel-axis; /IMPVEL on the master = moving rigid die), /RBE3
   (least-squares rigid fit + its virtual-work dual force distribution,
   following the ContactType2 lumped pattern hook for hook), /SECT
   section resultants (via the side-sum identity — element
   self-equilibrium does the bookkeeping), /PLOAD follower pressure,
   /IMPDISP position-level imposed displacement, /ADMAS,
   /INIVEL/AXIS, and rigid walls completed: SPHER/CYL geometries and
   MOVING walls (free with a carrier-node mass — momentum-exact
   impulse exchange — or /IMPVEL-driven). Two M5 solver lessons are
   recorded in the code:
   * elements fully interior to a rigid body must see the rigid
     velocity field evaluated at their PLACED positions (the enforce()
     re-scatter) or their hypoelastic stress ratchets at O(w^2 dt) per
     cycle — the original deactivates such elements, the port keeps
     them alive at exactly zero strain so their faces stay available to
     contact;
   * a free moving wall must solve its recoil IMPLICITLY with the node
     corrections (one 3×3 system per cycle): the explicit-lag variant
     feeds riding nodes a one-cycle-stale wall velocity and injects
     energy at a rate that does not vanish with dt; the wall energy is
     then booked from the per-node injection identity
     U = dKE − f·v_old dt (rigid_wall.py).
   A pre-existing (M1-era) issue was *isolated* during M5 and left
   for M6 (where it was FIXED — see the M6 entry), documented here
   honestly: the solid **bulk-viscosity work
   under barely-resolved ringing** misbooks — a coarse block left
   ringing violently (strain rates ~1/ms on a single element through
   the thickness) drifts the energy balance over long free flights,
   with the half-step-lagged linear (qb) damper as the isolated
   culprit; it is invisible in normal meshes/loadings (all M1–M5
   validations and examples), and none of the M5 modules touch it —
   reproduce with a 2×2×2 cube given ±0.3 opposite face velocities.
   Deferred out of M5, explicitly:
   * **/MPC** — a general multi-point constraint row couples arbitrary
     DOFs and needs a small implicit solve per constraint (or mass
     redistribution à la Lagrange/penalty) that shares nothing with the
     lumped one-way patterns used here; it moves to M6;
   * per-DOF flags of /RBE2 and /RBE3, RBE3 per-set weights;
   * /RBODY sensors, IKREM, skew/spherical inertia input, the surface
     envelope, merged bodies (/RBODY of /RBODY);
   * wall rotation (a moving wall translates only), containment
     (nodes inside a sphere/cylinder), the Dist search band;
   * /SECT frame output and element-set input; distributed /PLOAD on
     solids' internal faces (only surface segments), /PLOAD Ipinch etc.
5. **M6 — engine niceties** ✅ (done): /DT/NODA + /DT/NODA/CST mass
   scaling (nodal dt from the kernels' own claims + contact NEAR
   springs; added mass/momentum/energy tracked, reported and kept in
   the balance), engine restart chaining (`_0002.rad` resumes
   `_0001.rst`; a chain reproduces the unchained run exactly — the
   acceptance test — with rigid-body R/L and sensor latch state carried
   and everything else deliberately reconstructed), /STATE/DT restart
   snapshots, /DAMP mass damping (exact integrating factor, exact
   KE-identity booking, DE ledger), /SENSOR/TIME + /SENSOR/DISP gating
   loads (time-shifted curves) and interfaces, /MPC (deferred from M5:
   the coupled nc×nc Lagrange solve on accelerations — zero work by
   construction, fixed DOFs as ground), /EOS polynomial + ideal gas with
   the implicit E-p coupling and EOS sound speed (deferred from M3),
   the LAW2 adiabatic thermal terms and /FAIL/JOHNSON D5, and the FIX
   of the M1-era bulk-viscosity misbooking documented in the M5 note.
   The M6 solver lessons, recorded in the code:
   * the qb misbooking was only half the story: booking the damper's
     work trapezoidally (the SDOF-exact midstep booking) still left the
     reproducer at −66%, because a LAGGED damper acting on
     barely-resolved content (ω·dt → 2) drains energy through the
     DISCRETE ELASTIC FORCE — real numerical dissipation that no state
     function can book. The consistent treatment measures the exact
     internal-force midstep work each cycle (one einsum — the same
     identity as the M4 contact lesson) and books the residual into the
     reported EN ledger: the reproducer closes to −0.00% at /DT 0.9 AND
     0.2, healthy runs keep EN ≲ few % (asserted);
   * a balance that CONTAINS its own residual ledger can no longer flag
     instability through |ERR| — divergence now drives EN hard negative
     instead, so the Engine stops on ERRN < −2×limit (energy injection)
     and on time-step collapse (dt < 1e-9 of its running peak — a dead
     run must never spin forever at dt ≈ 1e-18);
   * /DAMP's integrating factor rescales the very velocities the other
     ledgers book work with — the O(α·dt) attribution residual is
     measured on the damped nodes and moved into EN (the /MPC settle
     test read a spurious frozen +3.2% before that correction).
   Deferred out of M6, explicitly:
   * the β (stiffness) branch of /DAMP (needs K·v products);
   * /SENSOR types beyond TIME/DISP, sensor-driven /RBODY activation;
   * /STATE .sta ASCII output (the pickle restart is the state file);
   * Gruneisen/tabulated EOS, Psh/tension cutoffs, EOS on shells;
   * per-DOF /MPC skew frames; heat conduction (thermal stays adiabatic);
   * ALE/CFD (long term).
6. **M7 — performance** ✅ (done): profile first, optimize second, and
   keep the solver architecture (and every result) unchanged.
   * **The profile drove everything** (numbers in PR #7): at this
     port's typical model sizes (10²–10³ elements — the examples) the
     cycle cost is NumPy *per-call overhead*, not flops. On
     rigid_impactor, `np.cross` alone was ~20% of the runtime (its
     moveaxis/axis bookkeeping), stacked `np.linalg.det/inv` on the
     (n,3,3) Jacobians ~90 µs/call through LAPACK dispatch, `np.add.at`
     4× slower than a bincount, ~60 einsum dispatches per cycle; on
     notched_plate the TYPE7 narrow phase (~30 vectorized where/maximum
     passes over the candidate set) was the hottest single block.
   * **Pure-NumPy cheap wins**, benefiting every install:
     `common/fastmath.py` (cross3/norm3 — bitwise identical to the
     NumPy calls they replace; scatter_add3 — bincount assembly,
     bitwise identical into a zero target, reassociated at ulp level
     into an accumulated one; det_inv33 — explicit cofactor 3×3, ~5×
     faster than LAPACK here and machine-precision close); the shell
     kernel's ten per-rate einsums fused into ONE stacked matmul (and
     the five hourglass-mode einsums into one); all-faces-at-once
     characteristic lengths; LAW36's Newton loop exits on its exact
     fixed point (bitwise-identical results, ~2.5× fewer table walks);
     an integer `np.clip` (a hidden np.finfo per call in NumPy 2)
     removed from the LAW36 curve walk. **Preallocated scratch buffers
     were profiled and rejected**: allocations measured < 1% of the
     cycle — call-count reduction is where the time was.
   * **The optional numba backend** (`pyradioss/accel`, the package
     docstring is the specification): `forces()` of solid_hexa8 and
     shell_bt4 split into pre/post array blocks around the pure-Python
     material/failure loop, each block plus the TYPE7 narrow phase
     mirrored as an `@njit(cache=True)` kernel — exactly the three
     measured hotspots, nothing else. Selected explicitly
     (`PYRADIOSS_BACKEND=numba`, `pyradioss-engine -backend numba`, or
     `accel.select_backend`); numba stays an optional dependency
     (`pip install -e ".[accel]"`) and a missing/unknown backend falls
     back to NumPy with a warning. No `fastmath`, no `parallel`: the
     mirrors reproduce the reference math element for element, so the
     backends agree bitwise except for reassociated short reductions
     (documented, ≲1e-15/call). tests/test_m7_backends.py asserts the
     contract at three levels: single-call kernel parity (rtol 1e-12),
     a full starter+engine impact run compared state array by state
     array, and the M6 restart-chain bit-match rerun UNDER numba — the
     canary that would instantly catch nondeterministic reductions.
   * **Measured speedups** (tools/benchmark.py; engine wall clock, same
     machine, numba JIT cache warm — the compile itself is a one-time
     ~10 s paid on the very first run):

     | example (engine s) | M6 baseline | M7 NumPy | M7 numba | numba vs M6 |
     |---|---|---|---|---|
     | rigid_impactor | 78.0 | 55.3 | 26.6 | 2.9× |
     | notched_plate  | 67.7 | 60.1 | 24.3 | 2.8× |
     | edge_impact    | 46.0 | 33.8 | 21.5 | 2.1× |

     (full nine-example table in PR #7; the small quick examples sit at
     1–2× — fixed per-cycle Python overhead dominates them.)
   * Deferred out of M7, explicitly:
     - **the JAX backend** (stretch scope, not started — reasons on
       record): the engine cycle is built on in-place scatter into
       shared force arrays, per-cycle Python branching (sensors,
       deletion, EOS, /DT/NODA) and stateful dict-of-arrays element
       buffers; none of that maps to `jax.jit` without rewriting the
       engine in functional style — a fork, not a backend — while
       un-jitted `jax.numpy` would only add dispatch overhead at these
       model sizes. Revisit only after (if ever) a functional-core
       engine refactor;
     - numba mirrors beyond the measured hotspots: the TYPE11 narrow
       phase, the material laws (LAW36's table walk is the visible
       next candidate on notched_plate), tri3/tetra4/beam kernels;
     - threading: numba `parallel=True` breaks the determinism/parity
       contract (scatter order); MPI/domain decomposition stays out of
       scope for the port.

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
