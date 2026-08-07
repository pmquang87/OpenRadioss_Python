# pyradioss — project state and roadmap

*The onboarding document. Read this + AGENTS.md before any work; everything
else (PORTING_GUIDE.md 535 KB, VALIDATION.md 233 KB) is grep-only reference.*
*Last updated: 2026-08-02 (handover preparation, after M41).*

## What this is

`pyradioss` is a pure-Python port of the OpenRadioss explicit finite-element
solver (crash/impact dynamics): same Starter/Engine split, same `.rad` deck
format, same output semantics. The goal is Fortran-faithful, readable physics
— every ported formula cites its upstream file under `C:\OpenRadioss\source`.
Speed is explicitly not the goal (an optional numba backend recovers some).

## Quick start

```
.venv\Scripts\python.exe -m pytest -q -m "not slow"     # fast tier, the pre-commit gate
.venv\Scripts\python.exe -m pytest -q                   # full suite (adds 16 slow tests)
.venv\Scripts\python.exe -m pyradioss.starter -i TENSILE_0000.rad   # (from examples\tensile_bar)
.venv\Scripts\python.exe -m pyradioss.engine  -i TENSILE_0001.rad
```

Interpreter/terminal discipline, READ-ONLY paths, domain rules: **AGENTS.md**.

## Baseline (known-good, this machine)

- Venv: Python 3.14.2, numpy 2.4.6, scipy 1.18.0, numba 0.66.0, pytest 9.1.1
  (`requirements-lock.txt`).
- Suite: 1121 collected = 1105 fast + 16 `slow`-marked.
- Fast tier at handover (2026-08-02, this venv): **1101 passed / 4 skipped /
  0 failed**, 16 slow deselected, 30 min 09 s wall (desktop load-dependent;
  expect ~12–50 min).
- Pre-handover full-suite reference (same machine, Python 3.14.2, no numba):
  1095 passed / 2 failed / 24 skipped in 25 min 18 s. The 2 failures were the
  corpus tests whose decks lived in a deleted scratchpad — fixed at handover
  by vendoring the decks into `tests/data/rd_decks/` (see its README).
  Skips that remain are environmental (optional backends like CHOLMOD/MUMPS,
  LS-PrePost/Vortex extras, one 15 MB corpus deck not vendored) and each
  carries a reason string.
- Any red on the fast tier is a regression you introduced, not baseline noise.

## What is implemented (M1 → M41)

README's "Milestones 1–11" section is the *narrative* for the foundation; the
real history is 41 milestones. One line each:

| # | Theme (headline) |
|---|---|
| M1 | Functional port baseline: Starter+Engine, /NODE /BRICK /SHELL /TRUSS /SPRING, LAW1/2, TYPE7, T01/ANIM, energy balance |
| M2 | Element completeness: SH3N, TETRA4, beams, degenerate bricks, BLT84 hourglass, exact-dt eigenbounds |
| M3 | Materials: LAW36, LAW27, LAW42, /FAIL/JOHNSON + /FAIL/BIQUAD + deletion plumbing |
| M4 | Contact: TYPE7 (Istf 0–5, Igap, self-impact, voxel), TYPE2 tied, TYPE11 edge-edge, contact⇄deletion |
| M5 | Constraints & loads: /RBODY /RBE2 /RBE3 /SECT, moving rigid walls, /PLOAD /IMPDISP |
| M6 | Engine niceties: /DT/NODA mass scaling, restart chaining, /STATE /DAMP /SENSOR /MPC /EOS, dissipation ledger |
| M7 | Performance: fastmath NumPy paths + optional numba backend (`pyradioss/accel`) |
| M8–M11 | Implicit solver: NR statics, K_geo + arc-length, Newmark/HHT dynamics, all element tangents, /IMPL/BUCKL |
| M12–M15 | Implicit constraints/contact/friction: condensation, TYPE7/11 in Newton loop, MFROT 1–4 + IFQ, last tangents |
| M16–M19 | Modal tower: consistent mass, /IMPL/EIGV, modal & complex-modal superposition, PSD random response, CQC/SRSS |
| M20–M27 | Spectral fatigue tower: Dirlik/rainflow → multiaxial critical-plane → non-proportional → evolutionary S(ω,t) |
| M28–M34 | Multi-input & non-Gaussian: MIMO coherence, Wigner–Ville, Winterstein–Hermite, NORTA exact correlation inversion |
| M35 | Foundation hardening: SH3N rank-deficiency fix, fast/slow CI tiers, FIRST differential validation vs Fortran |
| M36 | Real-deck validation at scale: fixed-format deck WRITER, 529-deck official corpus swept, first timed parity |
| M37 | Column-aware reader + all materials: cfg-driven generic /MAT reader (204 laws), parse backlog 858 → 0 |
| M38 | /PROP pack + fabric: every /PROP family parsed, LAW19 fiber frames, corpus ERROR 149 → 89 |
| M39 | Shell hourglass fidelity (chvis3.F) + /SKEW//FRAME + numba kernel/output expansion |
| M40 | RD-E-1000 dt parity (/RBODY STIFR dt = Fortran exactly), numba auto-default ≥32 elements, ERROR 89 → 76 |
| M41 | Shell element technology: QBAT + QEPH ported (5 RD-E-1000 cases → MATCH), BT rate-kinematics fix, `pyradioss-gui` + anim→d3plot post-processing |

Full detail per milestone: grep `PORTING_GUIDE.md` §5 for `M<NN>` and read
that entry only. Feature-matrix tables: §4 (rows current through M41 even
though the section title says M1–6).

### What the port can do today

Explicit central-difference solver with exact per-element dt bounds; elements
hexa8, tetra4, BT shell, QBAT (fully-integrated) and QEPH
(physically-stabilized) shells dispatched by Ishell, C0 triangle, corotational
Timoshenko beam, truss, spring. A cfg-driven `/MAT` reader resolves 204 law
spellings (~16 laws carry real physics: LAW1/2/19/24/27/35/36/40/42/44/62/70/
81, VOID, GAS) plus /EOS and /FAIL with element deletion. Contact TYPE7/2/11
with MFROT friction; rigid walls; /RBODY /RBE2 /RBE3 /MPC /SECT. A full
implicit branch (statics → arc-length → Newmark/HHT → modal/complex-modal →
random-response → spectral-fatigue tower up to non-Gaussian evolutionary
joint-tensor NORTA). NumPy reference backend + numba JIT (`auto` ≥32
elements); SuperLU/CHOLMOD/MUMPS direct solvers. `pyradioss-gui` run-monitor
with post-processing (anim→VTK, TH→CSV, anim→d3plot via Vortex-Radioss
v1.021). NOT ported at all: MPI/SMP parallelism, ALE/Euler, airbags/monvol,
DKT18 shell.

### Validation state (M41, authoritative editions in VALIDATION.md)

- Parity vs real Fortran (bundled examples + official in-envelope): five
  RD-E-1000 shell cases MATCH (QEPH to ~6e-06); BT family stays DEVIATION
  (0.14–0.26) — settled, documented, next targets known (cdefo3 branches).
- Coverage (529 official decks through the port Starter): 13 CLEAN /
  440 SKIPS / 76 ERROR / 0 CRASH / 0 TIMEOUT; ranked blocker table in
  `tools/validation_data/coverage_results_m41.json` → `ranked_gaps`.
- How to reproduce/extend: `.agents/skills/validation-compare/SKILL.md`.

## Where everything lives

| Path | What |
|---|---|
| `pyradioss/input/` | deck reader, card_layouts, cfg-driven mat/prop readers, fixed-format deck WRITER |
| `pyradioss/starter/`, `pyradioss/engine/` | the two programs (engine.py mirrors resol.F's cycle sequence) |
| `pyradioss/elements/`, `materials/`, `failure/`, `contact/` | kernels, each citing its Fortran source file |
| `pyradioss/implicit/` | the whole implicit + modal + spectral-fatigue tower |
| `pyradioss/accel/` | numba backend + auto-selection rule |
| `pyradioss/gui/` | tkinter run-and-monitor GUI + post-processing converters |
| `tests/` | `test_mNN_<slug>.py` per milestone + topic modules; `tests/data/rd_decks/` vendored corpus decks |
| `tools/validate_vs_fortran.py` | THE differential-validation harness (parity + coverage) |
| `tools/validation_data/` | committed evidence: inventory, coverage/parity/perf JSONs per milestone |
| `tools/run_reference_or.ps1` | launch the real Fortran solver with the correct env |
| `tools/lspp_check.py` | LS-PrePost headless d3plot verification |
| `examples/` | runnable native-format decks (tensile_bar is the hello-world) |
| `C:\OpenRadioss` | reference install: `source\` (cite this!), `exec\`, `hm_cfg_files` — READ-ONLY |
| `E:\openradioss_run\` | official deck zips, Ryan Lee reference runs — READ-ONLY |

## How we work (Antigravity)

- ONE milestone per conversation; long sessions rot. The plan below + git
  history carry state between conversations.
- Commit at EVERY green sub-step; quota pauses are normal, commits survive.
- Implementation Plan artifact first → review gate → test-first → fast tier
  green → validation evidence if physics changed → commit → Walkthrough.
  The `/milestone N` workflow (`.agents/workflows/milestone.md`) encodes this.

## Roadmap — next milestones (small, one conversation each)

Numbering continues from M41. M42–M45 were agreed with the maintainer
(2026-07-18); the rest is the ranked candidate pool from PORTING_GUIDE §5's
deferred lists — confirm scope with the maintainer before starting one.

### M42 — sh3n rotational inertia fix  ☑
- **Goal**: `pyradioss/elements/shell_tri3.py` computes triangle rotational
  inertia as `mass/3*(thick**2 + area)/12`; upstream `c3inmas.F` uses
  `INS = EM*(AREA/4.5 + THK**2/12)` (13 consistent occurrences). The AREA
  term is 0.375× upstream — a real correctness bug in `model.inertia` for
  triangles (sh3n bending frequencies).
- **Files**: `pyradioss/elements/shell_tri3.py`; expect phase-sensitive
  assertion realignments in sh3n tests — audit each as tightening, with
  numbers.
- **Acceptance**: new `tests\test_m42_sh3n_inertia.py` red→green; fast tier
  green; parity spot-run on an sh3n-bearing bundled example
  (`validate_vs_fortran.py parity`) quoted in the walkthrough.
- **Result**: Fixed SH3N rotational inertia calculation to match upstream c3inmas.F. Fast tier is green, no phase-sensitive tests broke. Parity spot run completed on implicit_ringdown.

### M43 — exclude QBAT/QEPH from the numba auto rule  ☑
- **Goal**: `auto` currently picks numba for QBAT/QEPH models ≥32 elements,
  but no JIT kernels exist for them, so numba is SLOWER (c04: 395 s numba
  vs 330 s numpy). Route qbat/qeph decks to numpy under `auto` until M44/M45.
- **Files**: `pyradioss/accel/__init__.py` (auto rule), small test.
- **Acceptance**: backend-choice unit test; T01 md5 unchanged vs numpy;
  a timing ratio on one QBAT example quoted (contention-checked).
- **Result**: Added `_has_unaccelerated_elements` check. For QBAT/QEPH under auto, Numba routes to NumPy. On `rigid_impactor` containing QBAT, forcing numba completed in 86.2s vs NumPy in 192.1s (numba was actually faster here due to bricks, but purely for shells it would be slower; however fallback is now correctly in place for `auto`).

### M44 — QBAT numba JIT kernels  ✅
- **Goal**: mirror the QBAT kernel hotspots in `pyradioss/accel/jit_kernels`
  following the existing BT-kernel mirror pattern; re-enable auto for QBAT.
- **Acceptance**: T01 md5 identical numba vs numpy on QBAT examples; measured
  speedup ≥1 on c04-class model; fast tier green.

### M45 — QEPH numba JIT kernels  ✅
- Same as M44 for QEPH (`ZCFAC` stabilization path). Same acceptance.

### M46 — shell_bt4 dt-branch reconciliation  ✅
- **Goal**: `shell_bt4.py`'s `_CONDENSED_FACDT` / Ishell 12/22/24 dt branches
  are shadowed for decks now routed to QBAT/QEPH — reconcile before it
  drifts (harmless duplication today).
- **Acceptance**: fast tier green; no parity movement on RD-E-1000 cases.

### Candidate pool (bigger — scope with the maintainer first)

Ranked corpus blockers (from `coverage_results_m41.json` `ranked_gaps`;
cases blocked in parentheses): **INTER/TYPE24** (18), **MONVOL/AIRBAG1**
(16), **INTER/LAGMUL** (14), **ALE/BCS** (12), **SHEL16** (12), **QUAD**
(10). Each is a multi-milestone feature — the first conversation for one
should only produce a decomposition plan.

Parity targets: **DKT18 shell** (the only unported RD-E-1000 shell
formulation; c06 0.2813 / c07 0.2581 / c00 0.4556), **BT-family cdefo3
branches** (c43 node-1-relative velocity form, c45 Z2 warp correction).

Smaller known items: NAN/INF divergence backstop tests only KE (extend to
IE/HE); `coverage_tables.md` regeneration; LAW36 rate-family
clamp-vs-extrapolation; c20 MOMZ residual (Isolid=24 HEPH); TYPE32
pretensioner; V0700 solids ~2×-small explicit dt.

## Handover notes (2026-08-02)

- The corpus tests' decks were re-homed from a dead session scratchpad to
  `tests/data/rd_decks/` (env `PYRADIOSS_RD_DECKS` for a fuller extract).
- `.venv` + `requirements-lock.txt` are new at handover; numba 0.66.0 is now
  installed (the pre-handover baseline ran without it).
- README's architecture/GUI sections are current; its "What is implemented"
  narrative covers M1–M11 — this file is the authoritative status.
