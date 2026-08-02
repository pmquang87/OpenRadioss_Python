---
name: validation-compare
description: Compare pyradioss output against the reference Fortran OpenRadioss - parity mode (T01 channel diff on the bundled examples) and coverage mode (starter keyword census on official decks). Use after ANY change to element/material/contact/dt/energy code, and to produce the validation evidence a milestone requires.
---

# Differential validation against the Fortran solver

One harness does both modes: `tools\validate_vs_fortran.py`. It launches the
reference solvers itself with the full runtime env (see the
run-reference-openradioss skill for what that env is and why).

## Parity mode — "same physics?"

```
.venv\Scripts\python.exe tools\validate_vs_fortran.py parity --workdir <scratch>
.venv\Scripts\python.exe tools\validate_vs_fortran.py parity --only tensile_bar --tol 0.05 --timeout 900 --workdir <scratch>
```

Pin the port to one backend for trend-comparability (set for the session):
`$env:PYRADIOSS_BACKEND = "numpy"`. numba agreement is proven separately by
T01 md5 equality, not by separate parity rows.

What it does per case: runs Fortran starter+engine in a scratch dir, converts
T01 via `th_to_csv_win64.exe`; runs the port on the same deck pair; interpolates
Fortran channels onto the port's time grid over the overlap window; scores
every channel (IE, KE, HE, CE, EW, MASS, MOMX/Y/Z, IE+KE, unambiguous
/TH/PART IE/KE) with `rel_rms = RMS(diff)/max scale`.

- **MATCH** = rel RMS ≤ 0.05 on every significant channel (significance:
  channel scale ≥ 1 % of its group's dominant scale; insignificant channels
  are printed with `~`, never hidden).
- Other classes: DEVIATION, PORT-ONLY(implicit / starter-reject / dialect),
  FORTRAN-FAIL, PYRADIOSS-FAIL, SKIPPED-SLOW, NO-CHANNELS.
- The bundled example decks are real fixed-format (written by
  `pyradioss/input/deck_writer.py`); the harness auto-detects this and applies
  `real_deck_fixups()` (RWALL blank d, TYPE7/11 gap_max column, strips
  port-only /TH/SECT). **Never feed a port-written deck raw to
  `engine_win64.exe`** — always go through the harness. `--shim begin` shows
  the real per-deck Fortran error for old free-format decks.

Results: `<workdir>\parity_results.json` + console table. Milestone
deliverable: promote to `tools\validation_data\parity_m<NN>.json`; narrative
goes in a NEW `VALIDATION.md` §3.x section (append-forward — never edit old
sections; each row carries the previous milestone's class as its own delta).

## Coverage mode — "does the Starter accept the deck?"

```
.venv\Scripts\python.exe tools\validate_vs_fortran.py coverage <deck>_0000.rad --timeout 120 --workdir <scratch>
```

Runs ONLY the port Starter: keyword census, every `** ERROR`/`** WARNING`
harvested, verdict CLEAN / SKIPS(n) / ERROR / CRASH / TIMEOUT. A starter-only
PASS says nothing about engine-time fidelity — that is parity's job.

The 529-deck official-corpus sweep has NO in-repo driver (it lived in a
session scratchpad). Its work list is `tools\validation_data\inventory.json`
(551 cases; `root` points at a dead scratchpad — re-extract decks READ-ONLY
from `E:\openradioss_run\demos_example` zips). Ranked blocker table:
`coverage_results_m41.json` → `ranked_gaps` (+ `delta_vs_m40` migration
matrix). M41 state: 13 CLEAN / 440 SKIPS / 76 ERROR / 0 CRASH / 0 TIMEOUT.

## validation_data file map

- `inventory.json` — official-deck work list (551 cases, envelope classes).
- `coverage_results[_m37..41].json` — per-milestone sweep results;
  `_m41` authoritative; `ranked_gaps` + `delta_vs_m<prev>`.
- `coverage_tables.md` — byte-complete 529-row verdict table (M36 edition).
- `parity_m36..41.json` — per-milestone parity rows; `_m41` authoritative.
- `perf_m36..41.json`, `perf_m41_clean.json` — timing harvests;
  `_clean` files are the only source for ABSOLUTE seconds
  (`clean: true`, `n_contended: 0`); everything else is upper bounds.

## Evidence rules (enforced)

- Latest edition wins in VALIDATION.md; add sections, never rewrite history.
- Every measurement carries its delta vs the previous milestone; any
  off-diagonal verdict move, new error class, or new crash is a REGRESSION
  and must be surfaced loudly.
- The user may be running a 12-process `engine_win64_impi` job on this box:
  check `tasklist` before trusting ANY wall clock. Classes, rel-RMS and
  speedup ratios are contention-insensitive; absolute seconds only from a
  proven-clean run.
- Numbers are always printed; the tolerance only sets the label.
