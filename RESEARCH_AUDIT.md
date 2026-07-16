# OpenRadioss_Python — Deep-Research Audit (Synthesis)

**Date:** 2026-07-16 · **Repo:** `C:/Users/pmqua/PycharmProjects/OpenRadioss_Python` @ `786163b` (M34 merge) · **Upstream baseline:** `C:/OpenRadioss_old/source/OpenRadioss-latest-20260520` (~5,600 Fortran files)

Synthesized from seven researcher reports: fortran-inventory, python-inventory, guide-roadmap, gap-analysis, drift-audit, tests-health, quality-review. Contradictions are resolved inline with the winning evidence named.

---

## 1. Executive summary

- The port is **92 Python files / 39,106 LOC** implementing a clean, well-documented explicit FEM solver plus an implicit stack, versus an upstream of ~5,600 Fortran files. It is an unusually disciplined codebase: clean layering, vectorized kernels, contract-tested numba mirrors, honest deviation logs.
- **Coverage of real OpenRadioss is thin**: 5 of ~119 material laws (~4%), 2 of 39–43 failure models, 2 of 21 EOS, 3 of ~22 contact types, 5 of ~50 property types, ~39 of ~100 starter keyword families (~12% at variant level). Airbags, ALE/Euler, SPH, AMS, thermal conduction, MPI, H3D output: **0%**.
- **Milestones M16–M34 are scope drift, plainly**: 18 of 19 milestones (all but a sliver of M16) implement modal/PSD/random-vibration/spectral-fatigue functionality that has **no counterpart anywhere in upstream OpenRadioss**. This layer is ~15,855 LOC — **40% of the package** — and consumed 56% of all milestones.
- The drift is **documented, not misrepresented**: every M16+ module carries a "Fortran origin — There is NONE" docstring and the guide marks the cards as "PORT cards". But the project has effectively pivoted from porting to building a random-vibration-fatigue research library since ~M19/M20.
- The test suite is large and genuinely rigorous (594 tests, 24,354 test LOC, analytic closed-form oracles at 1e-10..1e-12 tolerances), currently **601/603 passing (99.7%)** with **2 standing deterministic failures at HEAD** and **no CI** to have caught them.
- **Zero validation against the real Fortran solver exists** — despite full Windows OpenRadioss binaries (starter/engine/th_to_csv) sitting ready in `C:/Users/pmqua/Downloads/OpenRadioss/exec`. The guide's claim that Fortran comparison is "impractical" is contradicted by evidence on disk; a differential harness is the single highest-value next step.
- Genuine porting effectively stopped at **M15**; the "unchanged deferral tail" (TYPE19/24/25 contact, QEPH shell, LAW27 solids, LAW42 shells, thermal contact, Igap 2/3, Inacti, fiber beams) has been frozen for ~20 consecutive milestones.
- A real OEM crash deck **cannot run** on the port today: it fails on first contact with TYPE24/25, /TABLE, /SKEW frames, composite/sandwich properties, foam/anisotropic materials, airbags, and the TH/H3D output ecosystem.
- Robustness holes: /BEGIN units silently ignored, skipped keywords (including /FAIL) are warn-only, restart files are pickles (arbitrary-code-execution risk), no linter/type-checker/CI anywhere.
- The port also contains real, argued **improvements** over upstream: consistent contact/friction tangents (vs modified-Newton springs), a fix of the upstream IFQ=3 filter no-op bug, exact NORTA covariance matching, and a corrected bulk-viscosity energy ledger.

---

## 2. What is DONE (by subsystem, with numbers)

### Explicit solver core (M1–M7, ~17,600 LOC incl. infrastructure)
- **Elements — 7 families, all implicit-capable**: hexa8 (1-pt + Flanagan–Belytschko hourglass), tetra4, BT4 shell (Ishell=1, NIP Gauss layers, BLT84 hourglass), tri3 C0 shell, corotational Timoshenko beam (TYPE3), truss, spring (TYPE4). Every family implements `forces()`, `tangent()`, `kgeo()`, `consistent_mass()`.
- **Materials — 5 laws**: LAW1 elastic, LAW2 Johnson–Cook (rate, adiabatic thermal, D5), LAW27 brittle (shells), LAW36 tabulated, LAW42 Ogden/Mooney–Rivlin (solids). All five have consistent tangents for implicit. **Failure**: /FAIL/JOHNSON, /FAIL/BIQUAD with full element-deletion plumbing. **EOS**: polynomial, ideal gas.
- **Contact — 3 types**: TYPE7 penalty node-to-surface (Istf 0–5, Igap 0/1, voxel broad phase — no O(n²) remains, MFROT friction models 0–4, IFQ 1–3 filter, sensor gating), TYPE2 tied kinematic, TYPE11 edge-to-edge (friction is a documented port extension — upstream i11mainf.F hardcodes MFROT=0).
- **Constraints/loads**: /RBODY, /RBE2, /RBE3, /MPC, /SECT, /RWALL (plane/sphere/cyl, moving with implicit recoil solve), /BCS, /CLOAD, /PLOAD, /GRAV, /IMPVEL, /IMPDISP, /ADMAS, /INIVEL, /DAMP (mass α), /SENSOR (TIME/DISP), /DT/NODA(/CST) mass scaling, restart chaining (bit-reproducing), energy-balance stop with the EN internal-force midstep-work ledger (the M6 fix of the flagship M1-era qb misbooking).
- **Performance**: NumPy-only by design; optional numba backend mirrors the 5 measured hotspot blocks with a bitwise-parity contract (measured 1.05–2.36×).

### Implicit stack (M8–M15, ~5,658 LOC)
Newton–Raphson statics with load stepping and line search, updated-Lagrangian NLGEOM, Crisfield arc-length, Newmark-β/HHT-α dynamics, linearized buckling, constraint **condensation** (never penalized), TYPE7/TYPE11 contact in the Newton loop with consistent nonsymmetric friction tangents, follower-load stiffness, SuperLU/CHOLMOD/MUMPS linear solvers. All 7 element families and all 5 laws are tangent-complete.

### Modal/random/spectral-fatigue layer (M16–M34, 15,855 LOC — port-only)
Modal eigen + effective mass, modal superposition/FRF (Nigam–Jennings), complex modes (QEP), PSD random response + response spectra (SRSS/CQC), spectral fatigue (Bendat/Dirlik/Wirsching–Light/Tovo–Benasciutti + ASTM rainflow), multiaxial critical plane, non-proportional (time + spectral), non-Gaussian Winterstein–Hermite, non-stationary/evolutionary PSD, joint 6×6 tensor, multi-input coherence (incl. f- and t-varying), Wigner–Ville continuous spectrum, time-varying kurtosis, joint-tensor non-Gaussian, exact NORTA/Nataf covariance inversion (preservation error ~1e-15). The front end is genuinely FEM (assembles the deck's real K/M, recovers element stress modes); the estimator modules themselves are mesh-blind signal processing.

### I/O
Starter dispatch: exactly **39 keywords** (`starter_keywords.py:1510`); engine deck: 9 families + the large /IMPL family. Output is CSV time-history and legacy VTK — **not** binary T01/ANIM/H3D, so standard Radioss post-processors cannot read results. 36 example decks, one per capability.

---

## 3. What is OPEN

### (a) Items the port's own guide defers (consolidated)

**The frozen "deferral tail"** (carried verbatim since ~M16, some since M2–M4):
- **Elements**: fully-integrated solid (Isolid=17), QEPH shell, DKT18 triangle, nodal-pressure tetras, fiber-integrated beam /PROP/TYPE18, BT4 thin-plate shear locking (needs MITC4), BT4 drilling-row NLGEOM residual floor, UL hourglass memory in the NLGEOM tangent, beam K_geo beyond the axial term.
- **Materials**: LAW27 plastic block/solids, LAW42 shells + Prony viscosity, LAW36 Fsmooth/Chard/Fcut, /FAIL/BIQUAD M/S-flags, Gruneisen/tabulated EOS, EOS on shells, heat conduction (thermal stays adiabatic), rate-dependent plasticity under implicit dynamics, exact LAW2 plane-stress Iplas=1 return, non-proportional hardening.
- **Contact**: TYPE19/24/25, Igap 2/3, Inacti, thermal contact, /FRICTION per-part-pair sets, orthotropic/thermal friction, IFQ≥10/MODFR 2, TYPE2 Spotflag/penalty form, /RWALL under implicit (refused).
- **Constraints/engine**: per-DOF skew frames, /RBE2//RBE3 per-DOF flags, /RBODY sensors/envelope, explicit-engine constraint chains (permanently refused — implicit only), /IMPVEL under implicit dynamics, stiffness branch of explicit /DAMP, /SENSOR types beyond TIME/DISP, /STATE ASCII, ALE/CFD ("long term"), JAX backend, MPI/threading (out of scope).
- **Implicit machinery**: IDTC 2/3 step control, implicit↔explicit switching (/IMPL/SWITCH), QSTAT init, Lanczos/subspace sparse eigensolvers (dense `eigh` used), AMLS, gyroscopic systems, branch-switching continuation.
- **Fatigue-layer open items** (for completeness): mean stress beyond Goodman, Paris-law crack growth, complex-FRF stress recovery, 100-30-30 multi-directional spectra, multi-input base-acceleration feed, multi-input joint non-Gaussian path, non-translation copulas (queued as M35), NORTA feasibility repair, quadratic-form induced kurtosis.

### (b) Upstream features never started, ranked by practical value

1. **/INTER/TYPE24 + TYPE25** — the modern crash-contact workhorses; every current OEM deck uses them.
2. **/TABLE** (multi-dimensional tables) + **/SKEW//FRAME** — deck infrastructure nearly all real inputs depend on; port has 1-D /FUNCT only and refuses skew_ID everywhere.
3. **Crash material laws**: LAW70 foam, LAW57/87 Barlat anisotropic sheet, LAW44 Cowper–Symonds, LAW24 concrete, LAW25 composite shell, LAW28/50 honeycomb, LAW33/35/38 foams, LAW58/19 fabric, LAW59/83 spotweld.
4. **Composite/sandwich properties** TYPE10/11/16/17/51 + /STACK//PLY, and the spotweld stack (TYPE13/43 + connection laws).
5. **/MONVOL airbag family** (11 types incl. FVMBAG) + injectors/leakage + fabric laws.
6. **Output ecosystem**: binary TH/ANIM or H3D writers, full /TH group set (/TH/SHEL, /TH/INTER, /TH/ACCEL, accelerometers/gauges).
7. **Deck logistics**: //SUBMODEL, /TRANSFORM, /UNIT (currently silently ignored — a correctness hole), /IMPACC, remaining /INIVEL variants.
8. **ALE/Euler/multifluid** (145+33 engine files upstream) — FSI, blast, fuel sloshing.
9. **SPH** (46 files) — bird strike, sloshing.
10. **AMS advanced mass scaling** and/or any parallel scaling (MPI/domain decomposition, 289 upstream files) — beyond toy-mesh sizes.
11. **Advanced springs/joints** TYPE8/12/13/33/45 — crash connectors.
12. **Seatbelts** (/RETRACTOR, /SLIPRING), /XREF reference state, /EBCS, RAD2RAD multi-domain, thermal conduction. (Note: upstream has **no electromagnetic solver** — that is not a gap.)

---

## 4. Scope-drift verdict on M16–M34

**Verdict: confirmed, large, self-aware, and diverging by construction.**

The evidence (drift-audit, corroborated by python-inventory and guide-roadmap):

- `/IMPL/EIGV` **does not exist upstream**. `freimpl.F` (638 lines, the entire /IMPL reader) has no EIGV, PSD, or FATIG branch; its sole "PSD" token is `IMUMPSD`, a MUMPS flag. Upstream's real modal card is `/EIG`, and its open-source solver is a **stub** — `engine/stub/eig.F`'s executable body is `CALL ARRET(5)`; even buckling mode extraction (`EIGBUCKP`) is behind `#ifdef DNC` (commercial builds) and defined nowhere in the open tree.
- Greps of the full upstream solver source return **zero genuine hits** for RAINFLOW, DIRLIK, WIGNER, HERMITE, KURTOSIS, WINTERSTEIN, NORTA, NATAF, PALMGREN, RANDOM VIBRATION, RESPONSE SPECTRUM, MODAL SUPERPOSITION. The only "DIRLIK" in the repo is in bundled **LS-DYNA** keyword-format cfg files for the HyperWorks reader — not OpenRadioss code. There is no /FAIL/FATIGUE among upstream's failure models.
- Scoreboard: **18 of 19 milestones (M17–M34) have zero upstream counterpart; M16 is partial** (the /EIG card exists, the solver does not). The layer is 15,855 LOC = **40% of the package** (python-inventory's wc-l figure of 15,855 wins over drift-audit's 15,655 — a minor counting difference on the same modules), ~10,600 test LOC, 19 of 36 example directories, and 19 of 34 milestones (56%).
- The mechanism is self-perpetuating: each milestone Mn+1 implements the *first item of Mn's deferred list* (M34's commit says so explicitly), and M35 is already queued (t-copula/vine-copula joint distributions) — each step a strictly narrower refinement of the previous, mathematically convergent but diverging from the stated goal of reproducing OpenRadioss's functional architecture. Meanwhile the guide's explicitly labelled "unchanged deferral tail" of *real* upstream features has been frozen ~20 milestones.
- Mitigating facts, stated plainly: the drift is **honest** (every M16+ module docstring says "Fortran origin — There is NONE"; the guide says "PORT card" 18 times), each new capability sits behind an off-by-default card whose off-state is monkeypatch-asserted byte-identical to prior behavior, and the layer is genuinely FEM-fed at its front end. But none of it can ever be validated against, or contribute parity with, the code the repo exists to port.

**Bottom line: genuine porting stopped at M15.** From M19/M20 onward this is a random-vibration-fatigue research library built on top of a small ported FEM kernel.

---

## 5. Test-suite health

**Numbers** (tests-health, measured by running every file): 39 files, 594 test functions (603 collected items), 24,354 test LOC (this measured figure wins over python-inventory's ~23,700 estimate). **601 passed / 2 failed = 99.7%**, ~88 minutes serial (Python 3.10.11, pytest 9.0.2).

**Two standing deterministic failures at HEAD**, both post-merge regressions:
1. `test_m11_implcomp::test_mixed_element_model_converges` — implicit static on the all-family model reports `converged=False` after 1 Newton iteration.
2. `test_m31_wignerville::test_tensor_plane_drifts_continuously` — `fnp_drift` = 1e-16 vs required > 0.1 (in `wigner_ville_fatigue.py`); most plausibly regressed by M32–M34 touching the same critical-plane plumbing.

There is **no CI** (`.github/workflows` absent), no lint, no mypy — so these red tests sat unnoticed.

**Rigor verdict: genuinely high, not smoke.** Element kernels assert closed-form stress at rel=1e-10 and rigid-rotation objectivity <1e-12; integration tests check P-wave speed, Euler–Bernoulli cantilever periods, Johnson–Cook curves; M10 verifies Newmark period elongation equals the analytic `(ω·dt)²/12` *and* shows the O(dt²) convergence rate; M20 matches a hand Gamma-function Bendat evaluation at rel=1e-12 and the ASTM E1049-85 rainflow worked example cycle-by-cycle. Zero recorded-golden-file tests exist — all ~68 "parity" tests compare two live code paths.

**Gaps, ranked:**
1. **Nothing validates against the real Fortran solver.** No reference .out/T01/ANIM file exists anywhere in the repo; all `.out`/`T01.csv` files read in tests are the port's own outputs checked against closed forms. Here the researchers' framings must be reconciled: the guide claims Fortran comparison is "impractical (different output formats)", but quality-review found **compiled Windows binaries including `th_to_csv_win64.exe`** (which emits directly comparable CSV) in `Downloads/OpenRadioss/exec`. **The quality-review evidence wins: differential validation is feasible and simply hasn't been done.**
2. **HEAD is red with no CI** (above).
3. **Oracle decay in the fatigue tower**: M22–M34 validate chiefly by exact-reduction/byte-identity to earlier milestones plus seeded Monte-Carlo bands as loose as rel=0.5–0.6 (a 2× error would pass); the MC synthesizers consume the same PSDs the estimators use, so those cross-checks are partially self-referential.
4. **Runtime and fragility**: two tests alone take ~35 minutes; no slow markers or parallelization; 12 fatigue-era files import via a `tests.*` namespace package that any installed `tests` distribution shadows (collection fails out of the box on this machine).
5. **Untested surface**: `anim_vtk.py`, both CLI `__main__.py` entry points, the `time_history.py` writer; only 2 lexer tests versus a 1,566-line keyword parser (no negative/fuzz cases).

---

## 6. Top 12 recommended next steps (ranked)

1. **Build a differential-validation harness against the Fortran binaries** (`tools/validate_vs_fortran.py`): run the bundled examples through `starter_win64.exe`/`engine_win64.exe`, convert with `th_to_csv_win64.exe`, and assert displacement/energy histories within tolerance. This converts "self-consistent" into "correct" and is the single biggest credibility gap. **(M)**
2. **Fix the 2 red tests at HEAD** (m11 mixed-element implicit convergence; m31 `fnp_drift`), bisecting the M32–M34 merges for the latter. A port whose own suite is red undermines every parity contract. **(S)**
3. **Add CI (GitHub Actions)**: matrix over Python × {numpy-only, +scipy, +numba}, with a fast tier (slow markers on the two 15-min tests, xdist) so the 88-minute suite fits a runner. Protects the bit-parity and restart contracts that silently rotted. **(S)**
4. **Declare a scope decision and freeze M35.** Either re-badge M16–M34 as a separate `pyradioss_fatigue` product and return milestones to porting, or formally re-charter the project. The next queued milestone (vine copulas) deepens the divergence for zero porting value. **(S — a decision, then M for the package split)**
5. **Port /INTER/TYPE24 (then TYPE25).** The top of the frozen deferral tail and the highest-value single feature for running anything resembling a modern crash deck. **(L)**
6. **Implement /TABLE and /SKEW//FRAME plumbing.** Deck infrastructure that gates most real inputs and most future material laws (rate/temperature interpolation feeds on /TABLE). **(M)**
7. **Strict input mode + skipped-keyword summary + unit handling.** `--strict` (warn→error), an end-of-starter table of skipped blocks, and at minimum parsing /BEGIN//UNIT fields to *warn* on inconsistent unit systems — closes the two most likely silently-wrong-run paths. **(S)**
8. **Add 3–4 crash material laws** (LAW44 Cowper–Symonds first — cheap given LAW2 machinery; then LAW70 foam, LAW57 Barlat) and 1–2 failure models (FLD, TAB1), reusing the existing law/tangent scaffolding. **(M each)**
9. **Fix test-suite fragility and speed**: add `tests/__init__.py` (or conftest path pinning) to unbreak the 12 namespace-import files, mark slow tests, enable xdist. **(S)**
10. **Harden restarts**: replace pickle with `np.savez` + JSON header (the snapshot dict is already cleanly defined), removing arbitrary-code-execution risk on shared `.rst` files. **(S)**
11. **Extend the numba layer to tetra4/tri3 and cut fixed per-cycle overhead** (preallocated scratch for `v_old`/`sig` copies, cached energy reductions) — the guide's own benchmarks show small decks pinned at 1.05–1.10×. Add an in-repo `tools/profile_cycle.py` to keep this measurable. **(M)**
12. **Add lint + typing gates and test the untested surface** (`ruff`, `mypy` on the already-typed `input`/`model`/`common`; unit tests for `anim_vtk.py`, `time_history.py`, and both CLI entry points; a handful of malformed-deck negative tests). **(S)**

---

## Appendix: contradictions resolved

| Topic | Conflicting figures | Resolution |
|---|---|---|
| M16–M34 LOC | 15,855 (python-inventory) vs 15,655 (drift-audit) | 15,855 — python-inventory enumerated the module list with wc -l; both round to 40% of 39,106. |
| Test LOC | ~23,700 (python-inventory) vs 24,354 (tests-health) | 24,354 — tests-health measured it while executing the suite. |
| Upstream failure models | 39 engine dirs (fortran-inventory) vs 42 (drift-audit) vs 43 readers (gap-analysis) | All correct at different levels: 39 engine directories, ~42–43 starter reader files (starter adds gurson, fractal, failuser, windshield_alter). |
| Fortran validation feasibility | Guide: "impractical" vs quality-review: binaries + CSV converter on disk | Quality-review wins — `th_to_csv_win64.exe` makes time-history comparison directly practical. |
| M16 upstream status | guide/python-inventory: "PORT card, no counterpart" vs drift-audit: "partial" | Drift-audit's finer-grained finding wins: the `/EIG` *card* exists upstream but the OSS solver is a stub; `/IMPL/EIGV` as a card is invented. |
| Upstream material-law count | 119 engine mat dirs / 168 sigeps kernels (fortran-inventory) vs ~115 distinct laws (gap-analysis) | Both stand: 119 engine directories; ~115 distinct readable laws after removing starter-only/void/user entries. Port coverage ~4% either way. |
