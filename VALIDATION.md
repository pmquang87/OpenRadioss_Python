# VALIDATION — differential validation of pyradioss against the Fortran OpenRadioss

*M36 edition: native real-format decks, full official-corpus coverage, first
timed parity. Supersedes the M35 first report (history note below).*

- Date: 2026-07-16, repo commit `af73d6f` (branch `claude/openradioss-python-m36-realdeck`)
- Reference solver: OpenRadioss Windows 64-bit double-precision build in
  `C:/OpenRadioss/exec` (`starter_win64.exe`, `engine_win64.exe`,
  `th_to_csv_win64.exe`), run single-process (`-np 1 -nt 1`), input format
  2022, `RAD_CFG_PATH=C:/OpenRadioss/hm_cfg_files`, Intel oneAPI runtime on
  `PATH` — the environment proven in M35 and encoded in
  `tools/validate_vs_fortran.py`.
- Port: pyradioss 0.1.0 (this repository), run as
  `python -m pyradioss.starter` / `python -m pyradioss.engine`.
- Machine-readable results: `tools/validation_data/` —
  `inventory.json` (official-deck work list, 551 cases),
  `coverage_results.json` (per-case sweep records),
  `coverage_tables.md` (the byte-complete 529-row verdict table),
  `parity_m36.json` (21 parity cases), `perf_m36.json` (42 timing records).

## History — what M35 established, what M36 changed

The M35 first edition (commit `786163b`) built the harness and proved the
environment. Its central finding was that the obstacle was the *deck dialect*,
not physics: the bundled examples were written in the port's own free-format
dialect, the real Starter rejected or silently misread them, and the harness
carried a 9-family fixed-format *translator* to bridge 5 of the 9 explicit
examples (3 MATCH at 0.10–2.6 % rel RMS, 1 explained DEVIATION, 4
`PORT-ONLY(dialect)`). Its recommendation — "give the port a real fixed-format
deck writer; then the translator disappears and all explicit examples become
directly diffable" — is exactly what M36 implemented. M35 insights that remain
valid are kept in §2.3 (the box_beam channel forensics, the `/RWALL d=0` and
`/STOP` semantics findings) and §8. M35's W12/W13 coverage tables are
superseded by §5 (its two reader bugs are now fixed).

## TL;DR

1. **The examples now speak real Radioss.** A documented fixed-format deck
   writer (`pyradioss/input/deck_writer.py`, 1855 lines, every card layout
   citing its `hm_cfg_files` CFG definition) emits decks that satisfy the real
   Starter's fixed columns AND the port's whitespace-token readers from ONE
   file. All 36 examples regenerated; 36/36 parse cleanly in the port with
   byte-identical (or by-design-identical) results vs the old decks; the
   Fortran-comparable ones are accepted by the real Starter with 0 errors.
   The M35 translator is retired to a fallback that *delegates to the writer*
   (~380 duplicated lines deleted).
2. **All 9 explicit examples are now Fortran-comparable** (8 ran; gas_piston
   is a genuine starter-reject, exactly as predicted): **3 MATCH**
   (tensile_bar 0.0017, rubber_block 0.001, antenna_mast 0.026 max rel RMS)
   and **5 DEVIATION** (0.43–0.99) whose deviations concentrate in contact
   energy / momentum / hourglass channels while internal energy is often
   within 3–16 % — the first differential evidence localizing the port's
   contact/hourglass discrepancies.
3. **The entire official corpus was swept**: 551 cases inventoried from
   `E:/openradioss_run` (61 RD-E + 26 RD-V + 8 tutorial packages), 529
   runnable decks fed to the port Starter. Verdicts: **0 CLEAN, 9 SKIPS,
   520 ERROR, 0 CRASH, 0 TIMEOUT** — not a single uncaught traceback.
   The dominant hard gap is group/set machinery: `/GRNOD/SURF` +
   `/GRNOD/GRNOD` + `/LINE/EDGE` are the *complete* hard-keyword gap for
   252/529 decks (48 %). 446 decks also hit caught fixed-format parse
   failures (§4.4) — the reader-bug backlog.
4. **Official-deck timed parity complete**: 40/40 IN_ENVELOPE cases — 1
   DEVIATION (the only case both engines ran), 33 port starter-fails on
   FOUR diagnosed fixed-format parser-gap signatures, 6 cases where the
   Fortran side fails too (mostly "unsolved" tutorial decks). The port
   engine's physics is untested on 39/40 official decks because the
   STARTER parse layer is the wall (§3) — the sharply-scoped M37 target.
5. **Performance measured for the first time** (§6): port totals 1.3×–26×
   the Fortran wall clock on the bundled examples, engine-only up to ~40×;
   single-threaded both sides, one machine, one run each.
6. **W12/W13**: the four M35-era reader bugs are fixed; both real k2rad decks
   now fail only on genuine feature gaps (W12: 2 errors, W13: 6 — down
   from 5 and 33 425).

## 1. Methodology

### 1.1 Native fixed-format decks — the translator is retired

The port now *writes* real Radioss 2022 fixed format. Every
`examples/*/generate_deck.py` emits through
`deck_writer.write_starter_from_port_lines` /
`write_engine_from_port_lines`; model definitions are untouched (the writer
includes the promoted M35 translator as a conversion layer). The writer is a
library of per-keyword emitters covering all 39 dispatch families in
`starter_keywords.py` plus engine decks; each card layout cites its
authoritative `hm_cfg_files/config/CFG/radioss*` CARD definition.

**Dual-dialect discipline** (the module docstring's core design): one file
must satisfy the real starter's fixed columns AND the port's
whitespace-token parsers. Mechanisms, all measured against the real binary:

- blank fields = real defaults (e.g. `/MAT/LAW42` blank nu card:
  `hm_read_mat42.F` line 148 defaults it to 0.495 — checked in source;
  `hm_read_cload/grav.F` default blank `Ascale_x` to 1.0);
- whitespace-only lines are blank cards to the real reader and *invisible*
  to the port reader — the core mechanism; trailing blank cards are emitted
  to avoid `WARNING 100217` (0-warning format cleanliness);
- non-trivial exact encodings: `/RBODY` with added mass (identity `/SKEW` +
  slave-group id == ICoG so the port token stream decodes), `/IMPVEL` /
  `/IMPDISP` with scale ≠ 1 (auxiliary scaled `/FUNCT` from a 900001+ id
  pool), `/SECT` (frame-less grnod-only card), `/INTER/TYPE7|11`, `/CLOAD`,
  `/GRAV`, `/PLOAD`, `/DAMP`, `/MPC`, `/ADMAS`, `/BOX/RECTA`, `/INTER/TYPE2`;
- **two documented RESIDUE fields** (starter-accepted, meaning differs):
  `/RWALL` blank search distance d = inert wall, and `/INTER/TYPE7|11`
  gap_max sitting in the real Tstart column — the harness's
  `real_deck_fixups` maps both to their real meaning for Fortran runs
  (d=1e30; gap_max moved to GAPMAX). Feed decks through
  `validate_vs_fortran.py`, not raw to `engine_win64`;
- irreconcilable combinations fall back to a loudly-commented PORT-DIALECT
  block (table in the module docstring): `/EOS`, `/PROP/SPRING`,
  `/SURF/SEG`, `/LINE/SEG`, `/RBE3`, `/SENSOR`, `/INIVEL/AXIS`, TYPE7 with
  Ifric/sens, non-default `/RWALL`, `/TH/SECT` — all confined to port-only
  chains;
- the engine `/STOP` energy-abort block is DROPPED (real engine dies
  `forrtl: severe (24)`; the port's default 15 % guard takes over —
  verified inconsequential, the guard never trips, T01s identical;
  documented in `EngineDeck.stop`).

`tools/validate_vs_fortran.py` auto-detects real-format decks
(`deck_is_real_format`) and applies `real_deck_fixups`; its old-dialect
fallback translator now delegates to `deck_writer` (single source of truth).

### 1.2 Parity mode & scoring (unchanged from M35)

Both chains run in scratch directories; the Fortran channels are linearly
interpolated onto the port's time grid; each overlapping channel scores
`rel RMS = RMS(fortran − port) / max(|fortran|_max, |port|_max)` and a final
deviation. Channels: global IE/KE/HE/CE/EW, mass, MOMX/Y/Z, IE+KE, and
unambiguous `/TH/PART` energies. Significance rule: a channel classifies
only if it carries ≥ 1 % of its group's dominant scale (noise channels are
reported with `~`, never hidden). Tolerance for MATCH: **5 %** rel RMS
(reported here as fractions: 0.05) on every significant channel. Classes:
`MATCH`, `DEVIATION`, `PORT-ONLY(implicit)`, `PORT-ONLY(starter-reject)`,
`FORTRAN-FAIL`, `PYRADIOSS-FAIL`, `SKIPPED-SLOW`. The M35
`PORT-ONLY(dialect)` class is retired — no bundled deck needs it any more.

### 1.3 Official-corpus inventory & coverage sweep

The official deck libraries were extracted read-only from
`E:/openradioss_run` (all 61 RD-E example zips incl. the pre-unzipped
RD-E-1602, all 26 RD-V verification zips, 8 tutorial packages, the loose
TENSILE and phone_start decks; nested zips exploded; non-deck payloads
filtered; 1514 files, 1.1 GB) and inventoried into
`tools/validation_data/inventory.json`: 551 cases, starter/engine decks
identified by content (`/BEGIN` vs `/RUN`), keyword families scanned
recursively through `#include` (0 unresolved), `/BEGIN` format versions and
node/element counts recorded, 65 cases with Fortran T01/.out references
next to the deck. The coverage sweep then ran the port Starter on every
runnable case (6 workers, 120 s per-deck cap, 152 s wall total), harvesting
the port's own `** WARNING`/`** ERROR` messages, classifying every skipped
keyword family (hard physics vs soft output vs control noise) and assigning
a verdict per case (§4).

### 1.4 Timing harvest (new in M36)

`harvest_fortran_out()` / `harvest_port_out()` in the harness parse both
solvers' `.out` listings for `n_nodes`, `n_elements`, `n_cycles` and the
solvers' self-reported ELAPSED TIME, wired into `run_fortran` /
`run_pyradioss`; wall-clock precision raised to 0.01 s. `perf_m36.json`
stores per-run wall + self-reported times, `cycles_per_s` and
`elements_x_cycles_per_s`.

### 1.5 Reproduction

```bash
cd C:/Users/pmqua/PycharmProjects/OpenRadioss_Python

# bundled-example parity sweep (native real-format decks, timing harvested)
python tools/validate_vs_fortran.py parity --workdir <scratch>/valruns

# single example with all numbers
python tools/validate_vs_fortran.py parity --only tensile_bar --workdir <scratch>/valruns

# keyword coverage of an arbitrary real deck through the port Starter
python tools/validate_vs_fortran.py coverage <deck>_0000.rad --workdir <scratch>/valruns
```

The official-corpus sweep and official parity drivers live in the session
scratchpad (`rd_decks/_tools/build_inventory.py`, `valruns36/run_official.py`,
`valruns36/build_perf.py`); the deck corpus itself is NOT in the repo —
re-running them requires re-extracting the decks from `E:/openradioss_run`.
Their outputs are committed under `tools/validation_data/`.

## 2. Parity — bundled examples (all 9 explicit, complete)

Tolerance for MATCH: rel RMS ≤ 0.05 on significant channels.

| case | class | channels (sig/total) | max rel RMS | notes |
|---|---|---:|---:|---|
| tensile_bar | MATCH | 6/9 | 0.00172 | – |
| rubber_block | MATCH | 6/9 | 0.000955 | – |
| antenna_mast | MATCH | 8/10 | 0.0257 | – |
| box_beam_impact | DEVIATION | 10/10 | 0.975 | IE 0.026 / KE 0.014 fine; CE 0.96, EW 0.98 drive it (contact bookkeeping) |
| edge_impact | DEVIATION | 9/9 | 0.456 | IE 0.44, KE 0.42 — genuinely different response |
| notched_plate | DEVIATION | 7/10 | 0.431 | IE 0.092; HE 0.43 max |
| rigid_impactor | DEVIATION | 8/10 | 0.974 | KE 0.88, MOMZ 0.97; IE 0.105 |
| spot_weld | DEVIATION | 8/9 | 0.987 | MOMX/Y/Z ~0.99; IE 0.16 |
| gas_piston | PORT-ONLY(starter-reject) | – | – | real starter: /EOS/IDEAL-GAS "INITIAL PRESSURE MUST BE POSITIVE" (expected; see §2.2) |

The M35 `PORT-ONLY(dialect)` four (box_beam_impact, edge_impact,
rigid_impactor, spot_weld) are now COMPARABLE on the regenerated native
decks — these are the first true differential numbers for them.

### 2.1 Deck regeneration & real-Starter acceptance

Every example was regenerated through the writer and checked two ways:
port parity against the old deck (T01 comparison) and raw acceptance by the
real Fortran Starter.

| example | regenerated | port parity vs old deck (T01) | Fortran starter accepts |
|---|---|---|---|
| tensile_bar | yes | MATCH — byte-identical | YES, 0 err / 0 warn |
| box_beam_impact | yes | MATCH — byte-identical | YES, 0 err / 0 warn |
| rubber_block | yes | MATCH — byte-identical | YES, 0 err / 0 warn |
| antenna_mast | yes | MATCH — byte-identical | YES, 0 err / 0 warn |
| edge_impact | yes | MATCH — byte-identical | YES, 0 err (2 info warn) |
| notched_plate | yes | MATCH — byte-identical | YES, 0 err (1 info warn) |
| spot_weld | yes | MATCH — byte-identical | YES, 0 err (2 info warn) |
| gas_piston | yes | MATCH — byte-identical (T01+T02 chained) | NO — documented port extension /EOS-on-LAW1 (ERROR 67/824) |
| rigid_impactor | yes | MATCH — byte-identical | YES, 0 err via harness fixups (raw: only the documented /TH/SECT port card errors, ERROR 100210) |
| implicit_cantilever | yes | MATCH — no T01 by design; engine listing identical | n/a (implicit) |
| spectral_fatigue | yes | MATCH — no T01 by design; engine listing identical | n/a (implicit) |
| remaining 25 examples | yes (all 36 regenerated) | port starter parses cleanly 36/36, 0 errors | n/a (implicit/fatigue port-only chains) |

Measured facts recorded during writer development: the real starter ERRORS
on unknown keywords (ERROR 100210), so the port-only `/TH/SECT` card (real
spelling: `/TH/SECTIO`) is stripped by the harness fixups; `/SECT` with
grnod only and no frame/element groups draws warnings 600/1813 only, no
error.

### 2.2 Discussion — where the deviations live

**The three MATCHes** validate end-to-end, on unmodified native decks: the
1-point brick + LAW2 (tensile_bar, 0.0017), LAW42 Ogden solids
(rubber_block, 0.000955), and Timoshenko beams under `/CLOAD`
(antenna_mast, 0.0257).

**The five DEVIATIONs are real findings, not regressions** — nothing here
deviated before, because nothing here could be compared before. The
pattern is consistent across all five: internal energy is often within
3–16 % (box_beam 0.026, notched_plate 0.092, rigid_impactor 0.105,
spot_weld 0.16) while the blow-ups sit in contact energy, external work,
momentum and hourglass channels. That localizes the port-vs-Fortran gap in
contact bookkeeping/force detail and shell hourglass dissipation rather
than in materials or element stiffness. edge_impact (IE 0.44, KE 0.42) is
the exception — a genuinely different response, the natural first target
for a dedicated TYPE11 differential study.

**gas_piston — PORT-ONLY(starter-reject), a genuine feature finding
(carried from M35, still true on the native deck).** Real Radioss does not
allow `/EOS` on a LAW1 elastic material (ERROR 67/824); the port
deliberately extends EOS attachment (task-exempt, documented in
`read_eos`). Additionally the writer emits P0 ≤ 0 which the real starter
rejects with "INITIAL PRESSURE MUST BE POSITIVE" — a potential deck_writer
fix for full 9/9 comparability, on the backlog. Fortran comparison of the
gas column requires rebuilding the example on a hydro law (LAW6).

### 2.3 M35 insights retained

- **box_beam channel forensics** (M35, §2.2 of the first edition; the M36
  numbers reproduce the pattern): the port books the rigid-wall reaction as
  contact energy where the Fortran build books negative external work —
  same physics, different ledger; and Fortran's Belytschko–Tsay shell
  dissipates ~4.4 % of total energy into hourglass control vs the port's
  ~0.04 % — the port's shell hourglass stiffness is ~2 orders of magnitude
  below the reference, worth a dedicated look on coarse crush meshes. The
  2.6 % IE gap is consistent with that missing dissipation.
- **`/RWALL` semantics** (measured in M35): the real `d=0` selects no
  secondary nodes at all — the wall never acts; hence the harness fixup
  `d=1e30`.
- **The real Engine dies on the port's `/STOP` block**
  (`forrtl: severe (24)`, unit 30) — measured in M35, resolved in M36 by
  dropping the block from written engine decks (§1.1).
- **The 27 implicit/fatigue examples remain PORT-ONLY(implicit) by
  construction**: their engine decks use the port's `/IMPL/...` cards
  (M8–M34); no Fortran chain exists for them with this binary, ever. They
  are validated analytically by their generator scripts.

## 3. Parity — official IN_ENVELOPE decks (40/40, complete)

A resume-capable driver ran every IN_ENVELOPE inventory case (40 cases;
programmatically verified that ZERO NEAR cases have output-only hard
blockers — all NEAR blockers are model content) through raw-deck Fortran
starter+engine, port starter+engine (600 s port budget), `th_to_csv` and
channel comparison. Pre-existing reference T01/.out next to decks are used
only if the fresh Fortran chain fails.

**Final tally: 1 DEVIATION (the only case where both engines ran), 33
port starter-fails with a healthy Fortran run, 6 cases failing on the
Fortran side too** (4 tutorial decks shipped "unsolved" + c30/c39). The
port ENGINE is therefore untested on 39/40 official decks: the port
STARTER's fixed-format field parsing is the single wall in front of all
official-deck physics validation.

| signature (exact port error) | cases | examples |
|---|---:|---|
| `/TH/NODE: unknown node 0` — 10-char id-list columns read as 0 | 20 | all RD-V-0700 J-C failure cases, RD-V-0240 tensile |
| `/GRNOD` subtype unsupported (`BOX`-driven ROLLIN sets) | 8 | all RD-E-1000 Bending (BATOZ/QEPH/DKT18) |
| `/PART: material 0 not defined` — mat column reads 0 (legal mat-less spring parts) | 4 | RD-V-0030 Spring TYPE4, T1000 |
| singletons: `int('0.0')` and `int('01x3')` crashes, `/SURF`/element unknown-node ids, empty `/GRNOD` | 7 | E0100 twisted beam, E0500 beam frame, V0300 pressure, T1020/1040/1060, TENSILE |

| case | class | max rel RMS | notes |
|---|---|---:|---|
| c26_V0200_Hardening | DEVIATION | 0.974 | both engines ran; IE final dev = 1.0 (divergence at end of imposed-motion ramp) — needs a dedicated look |

The dominant blocker (also #2 in §4.4) is the fixed-format field-packing
gap: official decks pack ids/values in exact 10/20-char columns (sometimes
with NO whitespace between fields, e.g. `1.0E-061.67E-07`) and the port's
free-token split miscounts. Windows MAX_PATH forced short run-dir names
(the real Fortran starter fails "INPUT FILE NOT FOUND" when run-dir +
deck filename exceeds ~260 chars); handled in the driver. Machine-readable
results: `tools/validation_data/parity_m36.json` (all 40 records).

## 4. Coverage matrix — the official corpus through the port Starter

### 4.1 Corpus & classification (inventory.json)

551 cases (a case = one starter deck, or one not-runnable payload); 529
runnable; 65 with Fortran T01/.out references; 22 NOT_RUNNABLE (21 loose
mesh-only .nas/.fem/.pch tutorial files + the .slb-only RD-SL-T-2000).

| category | cases |
|---|---:|
| rd_e | 402 |
| rd_v_blast | 5 |
| rd_v_contact | 15 |
| rd_v_elements | 43 |
| rd_v_failure | 15 |
| rd_v_fsi | 2 |
| rd_v_loads | 3 |
| rd_v_material | 19 |
| rd_v_tools | 10 |
| tutorial | 37 |
| **total** | **551** |

Classification against the port's dispatch envelope (control noise
ANALY/IOFLAG/DEF_SHELL/DEF_SOLID/SPMD/SHFRA ignored; soft = TH/* output
kinds, SUBSET, ACCEL, RANDOM@0.001 which the port warn-skips without
physics impact):

| grade | on hard blockers | strict (soft counted) |
|---|---:|---:|
| IN_ENVELOPE | 40 | 8 |
| NEAR (≤2 hard blockers) | 276 | 116 |
| OUT | 213 | 405 |
| NOT_RUNNABLE | 22 | 22 |

`/BEGIN` format versions across runnable decks: 140 (289 decks), 2023 (52),
2021 (55), 2017 (48), 2019 (31), 2018 (13), 2024 (10), 2020 (10), 2022 (8),
120 (1) — the official corpus is written almost entirely in classic fixed
format against the full keyword set, while the port envelope was developed
against the 2022 free-format k2rad dialect. Hybrid single-file decks (an
engine `/RUN../END/ENGINE` block BEFORE `/BEGIN`: phone_start,
RD-E-1500_Gears, all RD_V_0240) are flagged `embedded_engine_block=true`;
the port currently warn-skips those pre-/BEGIN cards.

### 4.2 Sweep verdicts (529 runnable decks)

| metric | value |
|---|---|
| runnable cases swept | 529 (of 551; 22 NOT_RUNNABLE) |
| CLEAN (rc0, nothing skipped) | 0 |
| SKIPS(n) (rc0, n non-control families skipped) | 9 |
| ERROR (rc=2, collected model errors) | 520 |
| CRASH (rc≠0 uncaught traceback) | **0** |
| TIMEOUT (>120 s) | 0 |
| cases with ≥1 caught parse failure | 446 |
| cases with zero hard keyword gaps | 40 |
| cases whose hard gaps are only {GRNOD/SURF, GRNOD/GRNOD, LINE/EDGE} | 252 (48 %) |
| cases whose hard gaps are only group/set/surface machinery | 262 |
| sweep wall time | 152 s (6 workers) |

Per-category breakdown of the 529: rd_e 394 ERROR + 8 SKIPS, rd_v_*
111 ERROR + 1 SKIPS, tutorial 15 ERROR. The nine SKIPS cases (model built,
restart written, only soft/warned families skipped): the four
RD-E-2601_Failure_strain one_shell cases (SKIPS(2), UNIT), RD-E-2602
one_shell Johnson (SKIPS(2), UNIT) and tab1 (SKIPS(4), FAIL/TAB1 + TABLE),
the two RD-E-2603_Forming one_shell cases (SKIPS(3), FAIL/FLD + UNIT), and
RD-V-0200_Hardening (SKIPS(1)) — which is exactly the case that then ran
end-to-end in §3.

**The byte-complete 529-row per-case verdict table lives at
`tools/validation_data/coverage_tables.md`** (608 lines — kept out of this
document deliberately) and in `coverage_results.json` with full per-case
records (keyword census, skipped families, degrade warnings, parse/error
messages). Representative rows:

| case | verdict | blockers |
|---|---|---|
| RD-E-0100 TWISBEAM (BATOZ) | ERROR | GRNOD/GRNOD; PARSE /TH/NODE/1: int('01x3') |
| RD-E-1701 BOXBEAM (all 28 mesh variants) | ERROR | GRNOD/SURF; LINE/EDGE (+4) |
| RD-E-2000 cube_TYPE24 | ERROR | INTER/TYPE24; PARSE /MAT/PLAS_JOHNS: float('7.80000000000000E-097.8000…') |
| RD-V-0700 SHELL_Ishell24_LAW2 | ERROR | PARSE /SHELL/1: int('0.0') (+12) |
| RD_V_0240 tensile_TABULATED_TETRA | ERROR | PARSE /MAT/PLAS_TAB/1; ERR /NODE card needs 4 fields, got 3 (+9) |
| tutorial TENSILE | ERROR | PARSE /SH3N/1: int('0.0'); PARSE /SHELL/1: int('0.0') (+1) |

### 4.3 Ranked keyword gaps

cases_using = deck contains the unsupported family; cases_blocking = family
is a hard physics skip in that run; sole_blocker = it is the case's ONLY
hard gap; blocks = total skipped blocks corpus-wide. Note: most ERROR cases
have several concurrent blockers, so per-family fixes only convert cases
whose ENTIRE hard set is covered.

| rank | unsupported family | cases using | cases blocking | sole blocker | blocks |
|---:|---|---:|---:|---:|---:|
| 1 | /GRNOD/SURF | 222 | 222 | 10 | 236 |
| 2 | /GRNOD/GRNOD | 200 | 200 | 51 | 1216 |
| 3 | /LINE/EDGE | 190 | 190 | 0 | 190 |
| 4 | /GRSHEL/SHEL | 36 | 36 | 0 | 100 |
| 5 | /SURF/SURF | 35 | 35 | 0 | 90 |
| 6 | /FUNCT_SMOOTH | 34 | 34 | 23 | 36 |
| 7 | /UNIT | 28 | 28 | 8 | 42 |
| 8 | /SURF/GRSHEL | 26 | 26 | 0 | 62 |
| 9 | /GRSH3N/SH3N | 24 | 24 | 0 | 25 |
| 10 | /SURF/GRSH3N | 23 | 23 | 0 | 24 |
| 11 | /PROP/SH_ORTH | 21 | 21 | 0 | 25 |
| 12 | /MAT/HYD_VISC | 21 | 21 | 0 | 30 |
| 13 | /PROP/SPR_BEAM | 20 | 20 | 2 | 39 |
| 14 | /MAT/FABRI | 19 | 19 | 0 | 19 |
| 15 | /GRBRIC/PART | 19 | 19 | 1 | 214 |
| 16 | /INTER/TYPE24 | 18 | 18 | 2 | 18 |
| 17 | /MAT/GAS | 17 | 17 | 0 | 18 |
| 18 | /PROP/INJECT1 | 17 | 17 | 0 | 17 |
| 19 | /GRNOD/GENE | 17 | 17 | 0 | 18 |
| 20 | /SKEW/FIX | 16 | 16 | 3 | 48 |
| 21 | /MONVOL/AIRBAG1 | 16 | 16 | 0 | 16 |
| 22 | /INTER/LAGMUL | 14 | 14 | 0 | 58 |
| 23 | /PROP/SPR_GENE | 14 | 14 | 0 | 71 |
| 24 | /FRAME/FIX | 14 | 14 | 3 | 16 |
| 25 | /GRPART/PART | 14 | 14 | 0 | 14 |
| 26 | /MAT/VOID | 13 | 13 | 0 | 13 |
| 27 | /ALE/BCS | 12 | 12 | 0 | 25 |
| 28 | /SHEL16 | 12 | 12 | 0 | 62 |
| 29 | /GRBRIC/BRIC | 12 | 12 | 1 | 13 |
| 30 | /PROP/TYPE20 | 12 | 12 | 0 | 21 |
| 31 | /QUAD | 10 | 10 | 4 | 33 |
| 32 | /SKEW/MOV | 10 | 10 | 0 | 13 |
| 33 | /ALE/MAT | 10 | 10 | 0 | 20 |
| 34 | /PROP/VOID | 10 | 10 | 0 | 13 |
| 35 | /MAT/CONC | 8 | 8 | 8 | 8 |
| 36 | /FRAME/MOV | 7 | 7 | 6 | 7 |
| 37 | /INTER/TYPE18 | 7 | 7 | 0 | 7 |
| 38 | /MAT/LAW51 | 7 | 7 | 0 | 22 |
| 39 | /MAT/LAW81 | 7 | 7 | 7 | 7 |
| 40 | /PROP/CONNECT | 7 | 7 | 0 | 8 |
| 41 | /MOVE_FUNCT | 6 | 6 | 0 | 23 |
| 42 | /MAT/LAW62 | 6 | 6 | 0 | 6 |
| 43 | /EOS/LINEAR | 6 | 6 | 0 | 6 |
| 44 | /MAT/KELVINMAX | 6 | 6 | 0 | 6 |
| 45 | /AMS | 5 | 5 | 0 | 5 |
| – | (129 more families, each blocking ≤ 5 cases) | | | | |

**Highest-value next port target:** the entity group/set machinery —
`/GRNOD/SURF` + `/GRNOD/GRNOD` + `/LINE/EDGE` (surface-derived node groups,
group-of-groups, edge lines). That trio is the complete hard-keyword gap
for 252/529 decks; the broader group/set family closure covers 262.
`/GRNOD/GRNOD` alone is the sole hard blocker in 51 cases; the pair
(GRNOD/SURF, LINE/EDGE) is the exact hard-gap set of another 82 (the
RD-E-1701/1702/1703 BOXBEAM families). Then `/FUNCT_SMOOTH` (sole blocker
in 23), `/UNIT` (8), `/SKEW`+`/FRAME`.

### 4.4 Crashes and parse-bug signatures

**rc≠0 crashes: NONE** — 0 of 529 decks produced an uncaught traceback
(the reader catches ValueError/IndexError/KeyError per block). Every reader
failure was caught and reported as `** ERROR while reading /X` (rc=2).
Those caught parse failures on real decks are the reader-bug backlog:

| # | parse-failure signature | cases | example |
|---:|---|---:|---|
| 1 | /TH/NODE: invalid literal for int(): '<tok>' (trailing node-name text column read as int) | 360 | TWISBEAM_0000.rad:483: int('01x3') |
| 2 | /NODE: card needs N fields, got N (fixed 20-char coordinate columns abut with no whitespace; free-token split undercounts) | 59 | SHPB_H_2021_FEB12_0000.rad:99 |
| 3 | /IMPVEL: rotational direction codes XX/YY/ZZ rejected | 35 | ROLLING_0000.rad:370: 'XX' |
| 4 | /SHELL: int() on per-element float fields (thickness/offset) on connectivity card | 28 | TANK_0000.rad:2962: int('0.0') |
| 5 | /SH3N: same | 18 | BAT_CIR_0000.rad:8785: int('0.0') |
| 6 | /MAT/PLAS_TAB: a third LAW36 card dialect beyond the two M35/M36 fixed | 17 | tensile_LAW36_0000.rad:20: int('1.0') |
| 7 | /FUNCT: two 20-char floats abutting | 13 | rdv_0530_wave_propagation_0000.rad:8392 |
| 8 | /EOS/IDEAL-GAS: title/comment token read as float | 11 | blast_experiment_0000.rad:32: float('EOS') |
| 9 | /INIVEL/AXIS: axis given as letter | 10 | DIF24416_0000.rad:14990: float('Z') |
| 10 | /TH/PART: variable-name card read as int | 8 | pendulum_0000.rad:3287: int('XXMOM') |
| 11 | /IMPDISP: XX/YY/ZZ direction codes | 8 | rubber_ring_0000.rad:3755: 'ZZ' |
| 12 | /EOS/POLYNOMIAL: comment token read as float | 8 | 1BRICK_COMPRESSION_0000.rad:49 |
| 13 | /SECT: int('.1') | 8 | bumper_LL4_0000.rad:77204 |
| 14 | /MAT/PLAS_JOHNS: abutting fixed fields | 3 | TANK_0000.rad:4706 |
| 15 | /INIVEL/TRA: float('0&V') | 3 | SPHEX_mono_110_0000.rad:449329 |
| 16 | /PART: list index out of range | 2 | BILLARD_0000.rad:9538 |
| 17 | /FAIL/BIQUAD: int('.2') | 2 | main_TEST4_0000.rad:56 |
| 18 | /SECT/PARAL: int('.1') | 2 | CBOX_0000.rad:20429 |
| 19 | /RBODY: int() | 2 | Front_Impact_completed_0000.rad:27993 |
| 20 | /DAMP: int('1E-5') | 1 | SHELL_LAW19_PROP9_0000.rad:22 |

Signatures 2, 7 and 14 share one root cause — parsers must switch to
fixed-column slicing (width 20) when the deck is classic fixed format.
Signature 1 (`/TH/NODE`, 360 cases) is the single most widespread failure.

## 5. Coverage — W12/W13 k2rad decks: the four M35 bugs are fixed

All four M35-identified port-reader bugs were fixed inside
`pyradioss/input/starter_keywords.py` (the lexer was not at fault):

1. **/MAT/LAW36 parse crash**: `read_mat` now accepts BOTH dialects,
   dispatched on card count (real layout ≥ 6 data cards, compact ≤ 5), the
   real layout taken field-for-field from `matl36_plas_tab.cfg` +
   `hm_read_mat36.F` (incl. the upstream Fscale 0→1.0 default), mixed
   %10d/%20lg widths parsed with a new `_fixed_vals` column-slicing helper.
2. **/SURF/SEG**: real cards are `seg_ID n1 n2 n3 n4` (`hm_read_surf.F`);
   5-token cards drop the leading seg_ID, and n4 = 0 becomes a triangle
   (upstream N4=0 → N3). The 4-token ambiguity (real triangle with blank N4
   vs compact quad) is documented — read as the quad; k2rad and the cfg
   writer always emit all 5 fields.
3. **/SURF/PART/EXT**: kind/qualifiers now derived from the block keyword,
   with a warning that the EXT qualifier is ignored (treated as plain
   /SURF/PART).
4. **/INTER/TYPE7 "friction filtering factor out of range"**: root cause
   was dialect misparse (the real card 2's Idel=2 landed in the compact
   slot for Ifiltr). `read_inter` now parses the REAL 6-card layout
   (`inter_type7.cfg` / `hm_read_inter_type07.F`, detected by card count)
   and aligns validation with the reference for BOTH dialects:
   `IF (ALPHA==0.) IFQ = 0` — Xfreq=0 switches the filter off instead of
   erroring (the port was wrong twice over); Iform=2 errors only when
   friction is actually active.

| deck | before | after |
|---|---|---|
| W12 (water-ALE) | rc=2, 5 ERROR(S) — LAW36 int('1.0') crash + knock-ons | rc=2, **2 ERROR(S)**, 10 warnings — both genuine: /MAT/HYD_VISC not ported (/PART/2, /PART/3) |
| W13 (blast vehicle) | rc=2, **33 425 ERROR(S)** — 33 418 /SURF/SEG card errors + TYPE7 reject + LAW44 knock-ons | rc=2, **6 ERROR(S)**, 20 warnings — all genuine: /MAT/LAW44 not ported (/FAIL/JOHNSON/1 + /PART/1..5) |

W13's /SURF/SEG now parses all 33 418 segments; its /INTER/TYPE7 is
accepted with one warning ("real-format fields not ported — ignored:
Idel=2; Stmin=1000; Inacti=5; Iform=2 (no friction defined — inert)").
Remaining genuine gaps to *run* the decks — W12: HYD_VISC + GRUNEISEN EOS,
INTER/TYPE18 (ALE coupling), GRBRIC/PART; W13: LAW44 (Cowper–Symonds),
LOAD/PBLAST (the deck's raison d'être), SURF/GRSHEL + SURF/PLANE +
GRSHEL/SHEL, TH/INTER + TH/SURF. Real-dialect LAW36 per-curve Fscale_i ≠ 1
and the fct_IDp/fct_IDE pressure/modulus functions are accepted-with-warning,
not applied (needs `resolve_materials`).

## 6. Performance (new in M36)

Machine: 13th Gen Intel(R) Core(TM) i9-13900H, 64 GB RAM, Windows 11. Both
solvers single-threaded (Fortran `-np 1 -nt 1` / `OMP_NUM_THREADS=1`;
port = one NumPy process). starter/engine seconds are subprocess WALL
clocks; ratio = port total / Fortran total. F = Fortran, P = pyradioss.

| case | elements | cycles F/P | Fortran st+en s | port st+en s | ratio | port cycles/s |
|---|---:|---|---|---|---:|---:|
| tensile_bar | 40 | 1420/1847 | 1.57+0.35 | 0.84+3.34 | 2.2× | 553 |
| rubber_block | 64 | 767/1312 | 1.48+0.54 | 0.76+3.76 | 2.2× | 349 |
| antenna_mast | 10 | 18299/4961 | 1.57+1.72 | 0.64+3.73 | 1.3× | 1330 |
| box_beam_impact | 200 | 2900/3466 | 1.53+0.89 | 0.82+10.84 | 4.8× | 320 |
| edge_impact | 256 | 9453/22814 | 1.49+2.97 | 0.93+89.85 | 20.4× | 254 |
| notched_plate | 194 | 13971/16147 | 1.82+9.43 | 0.75+169.83 | 15.2× | 95 |
| rigid_impactor | 118 | 16974/35689 | 1.98+3.58 | 0.95+143.46 | 26.0× | 249 |
| spot_weld | 300 | 3592/3454 | 1.61+1.30 | 0.77+15.12 | 5.5× | 228 |
| gas_piston | 4 | –/476 | 1.32+– | 0.72+1.28 | – | 372 |
| c26_V0200_Hardening | 8 | 32435/34517 | 1.59+3.49 | 0.74+34.78 | 7.0× | 992 |
| c10_V0030_spring_stiff | 5 | 40101/– | 1.71+5.89 | 1.06+– | – | – |
| c11_V0030_spring_visc | 5 | 80002/– | 1.56+7.28 | 1.02+– | – | – |
| c15_V0700_SHELL_Ish24_LAW2 | 9 | 14844/– | 1.55+5.42 | 0.86+– | – | – |
| c21_V0700_SHELL_Ish12_L36 | 9 | 15829/– | 1.80+6.60 | 0.92+– | – | – |
| c22_V0700_SHELL_Ish24_L36 | 9 | 14831/– | 1.68+5.41 | 0.87+– | – | – |
| c17_V0700_TRIA_LAW2 | 18 | 25117/– | 1.21+8.21 | 0.76+– | – | – |
| c24_V0700_TRIA_LAW36 | 18 | 25166/– | 1.58+7.59 | 0.86+– | – | – |
| c13_V0700_HEXA_18_LAW2 | 20 | 169356/– | 1.88+67.80 | 0.96+– | – | – |
| c14_V0700_HEXA_24_LAW2 | 20 | 169364/– | 1.70+45.01 | 0.85+– | – | – |
| c19_V0700_HEXA_18_LAW36 | 20 | 173743/– | 1.87+68.29 | 0.88+– | – | – |
| c20_V0700_HEXA_24_LAW36 | 20 | 173752/– | 1.66+45.27 | 0.90+– | – | – |

Observed port engine throughput: ~95–1330 cycles/s
(~2.3e3–2.3e4 elements×cycles/s, in `perf_m36.json`). Engine-only slowdown
is much larger than the totals suggest — Fortran starter startup (~1.5 s)
pads its total (e.g. notched_plate 9.4 s vs 169.8 s = 18× engine-only;
rigid_impactor 40×).

The completed official sweep adds Fortran-side timings for all 40
IN_ENVELOPE cases to `perf_m36.json` (port engine columns exist only for
c26 — the other 39 stop in the port starter, §3). Notable for future
port-side budgeting: the long official runs are cycle-bound, not
element-bound — RD-V-0240 (808 elements) takes the FORTRAN engine
280–768 s at ~250–590k cycles, and RD-E-1000 DKT18 runs 1.24M cycles.
At the port's measured ~100–1300 cycles/s these decks need hours, so the
600 s port budget will mark them SKIPPED-SLOW until the engine gets
faster — exactly the speed-work baseline this section exists to feed.

**Honest caveats** (recorded in `perf_m36.json` with every record):

- One machine, one run per case, background load uncontrolled — these are
  order-of-magnitude characterizations, not benchmarks.
- Wall clocks include process startup (port ~0.7–1.0 s Python interpreter +
  imports; Fortran binary load); the solvers' self-reported elapsed times
  are stored as `*_self_s` fields for a startup-free comparison (Fortran
  engine self-reported is typically 0.2–1 s lower than wall).
- `n_cycles` differ per solver (independent dt evolution), so `cycles/s`
  is per-solver throughput only; `elements×cycles/s` is the fairer
  cross-solver rate.
- Single-threaded both sides by construction; the Fortran build's
  MPI/OpenMP scaling (and the port's numba backend, benchmarked separately
  in `tools/benchmark.py` / PORTING_GUIDE M7) are outside this comparison.

## 7. Known issues & backlog (from this milestone's measurements)

1. **Fixed-format reader hardening** (blocks both §3 and §4): `/TH/NODE`
   trailing-name column (360 cases); abutting 20-char fields → column
   slicing for `/NODE`/`/FUNCT`/`/MAT/PLAS_JOHNS` (75 cases); `/SHELL` /
   `/SH3N` per-element float fields (46); `/IMPVEL`/`/IMPDISP` XX/YY/ZZ
   rotational codes (43); the third `/MAT/PLAS_TAB` dialect (17);
   `/EOS/IDEAL-GAS|POLYNOMIAL` title tokens (19); `/INIVEL/AXIS` letter
   axis (10); `/PART` with mat id 0 (legal for spring parts) rejected by
   the PART check.
2. **Group/set machinery port** ( §4.3): GRNOD/SURF + GRNOD/GRNOD +
   LINE/EDGE flips 252 cases' hard gap to zero.
3. **Contact/hourglass differential study** on the five DEVIATION examples
   (§2.2) — IE agrees, contact/hourglass ledgers do not.
4. **gas_piston**: emit a positive P0 (deck_writer) and/or rebuild on a
   hydro law for full 9/9 comparability.
5. **Hybrid single-file decks**: pre-/BEGIN engine blocks need either a
   file split in the harness or a pre-/BEGIN engine-section reader.
6. If the port reader ever learns `/TH/SECTIO`, the writer should switch
   spelling (rigid_impactor's raw deck currently carries the port-only
   `/TH/SECT`, stripped by harness fixups for Fortran runs).

## 8. Honest limitations of this report

- **The official parity sweep completed after the builder's snapshot**
  (40/40; §3 and the JSONs carry the final numbers). Because 39/40 cases
  stop in the port starter, official-deck PORT engine timings exist for
  only one case (c26) — the Fortran-side timings for all 40 are in
  `perf_m36.json`; the port columns fill in once M37 fixes the §3 parse
  signatures.
- **Two confirmation re-runs were still executing at builder handoff**:
  the full fast pytest tier and a repeat of the 11-example parity sweep
  against the final tree (the first full sweep was 11/11 MATCH; changes
  since were comment-only lines both readers skip, plus concurrent parser
  edits). Likewise the parse-fixes builder's two background suites
  (m15/m3 and m12/m13/m14/m5/m6/m7/m24) were pending — its fixture code
  paths are byte-preserved and were hand-traced green, but the suites were
  not seen finishing inside the builder's window.
- The deck corpus is not in the repo; `inventory.json` /
  `coverage_results.json` embed absolute scratchpad paths. The sweep
  overwrote 27–30 pre-existing Fortran reference `*_0000.out` listings in
  the extracted tree (byte-identical backups saved as `*.fortran_ref`
  first — use those for any Fortran-output comparison).
- Inventory node/element counts are data-card line counts (validated <1 %
  off on one deck); multi-line old-format element cards could overcount
  slightly on v4x decks. Classification is starter-based;
  engine-side families (`/H3D/*`, `/OUTP`, `/MON`) are recorded per case
  but not graded.
- The two RESIDUE fields (§1.1) mean a raw written deck fed to the real
  ENGINE without the harness fixups runs with an inert `/RWALL`
  (box_beam_impact) and a misplaced TYPE7 gap_max (notched_plate) — always
  run through `validate_vs_fortran.py`.
- `/TH/NODE` displacement/velocity channels are still not compared
  (`th_to_csv` `var NN` numbering); global + part energies carry the
  comparison.
- One run per solver per case; the 5 % MATCH tolerance and 1 % significance
  floor are choices, printed alongside every raw number so anyone can
  re-slice.
