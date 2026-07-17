# VALIDATION — differential validation of pyradioss against the Fortran OpenRadioss

*M37 edition: column-aware fixed-format reading (the entire M36 parse-bug
backlog retired), every /MAT card parsed, the first material-physics
packs, group/set machinery. Supersedes the M36 report for the TL;DR/§4/§7/§8;
M35/M36 history kept below. NOTE: the M37 corpus re-sweep DID execute
(§4.6 — its first builder died on a transient API error and it was re-run;
`coverage_results_m37.json` is the authority for the corpus numbers, and
where §4.1–4.4 still quote M36 figures they are superseded by §4.6). The
official parity re-run is reported in §3.1.*

- Date: 2026-07-16, branch `claude/openradioss-python-m37-materials`
  (M36 baseline: commit `af73d6f`, branch
  `claude/openradioss-python-m36-realdeck`)
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
  `coverage_tables.md` (the byte-complete 529-row M36 verdict table),
  `parity_m36.json` (21 parity cases), `perf_m36.json` (42 timing records),
  and the M37 measurements: **`coverage_results_m37.json`** (the full
  529-case re-sweep, M36 schema + `delta_vs_m36` — §4.6),
  **`parity_m37.json`** (61 results: 52 official + bundled re-run — §3.1)
  and **`perf_m37.json`** (122 timing records with per-record contention
  notes — §6.1). These three are the authoritative measurements for this
  milestone.

## History — what M35 established, what M36 changed, what M37 changed

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

M36's own recommendation was equally sharp: "the port STARTER's
fixed-format field parsing is the single wall in front of all
official-deck physics validation" (§3), with the group/set trio and the
/MAT law families ranked right behind it (§4.3). M37 attacked exactly
that list: the reader became column-aware (killing all 20 parse-bug
signatures of §4.4), every /MAT card in the radioss2022 catalogue now
parses, ten material laws gained real physics, and the group/set +
/UNIT machinery landed. What M37 did NOT deliver is the re-measurement:
the corpus re-sweep and the official parity/timing re-run were assigned
to builders that failed, so this edition documents the tree-side deltas
(§4.5) against the still-standing M36 corpus numbers.

## TL;DR

1. **The entire M36 parse-bug backlog is dead.** The reader is now
   COLUMN-AWARE: a new shared module `pyradioss/input/card_layouts.py`
   holds the field-formatting primitives + the LAYOUTS column-width table
   (every entry citing its `hm_cfg_files` CARD format string — writer and
   reader consume ONE table); `deck_reader.py` detects the real dialect
   from `/BEGIN`'s declared input version (≥ 90 → fixed columns on every
   block, `#include`s included), reconstructs the TRUE fixed card stream
   with real blank-card semantics, and substitutes `/PARAMETER` `&NAME`
   references in place. All ~20 M36 `parse_error_signatures` (~600
   corpus incidents) are FIXED (§4.5): on the 60-case validation slice,
   parse-error lines went 268 → **0**, with 0 crashes and 0 timeouts, and
   **23/60 cases now exit rc=0** (the whole M36 sweep had 9 rc-0 cases in
   529). The 37 slice cases still at rc=2 fail on genuine feature gaps,
   not parse bugs.
2. **Every /MAT card parses.** A cfg-driven generic reader
   (`pyradioss/input/mat_reader.py`) interprets the `hm_cfg_files` CFG
   DSL into per-law schemas: 204 law spellings resolve (the full
   radioss2022 MAT tree), and **193/193 non-dedicated /MAT blocks in the
   official corpus parse with 0 failures and 0 heuristic fallbacks**.
   Laws without ported physics become `InactiveMaterial` (full params +
   density stored, mass init works, /PART cross-refs downgrade to
   warnings) and the Engine REFUSES to run them, naming law/id/element
   family. `MAT_PHYSICS_REGISTRY` is the one-line hook for new-law
   physics. 4 previously-failing real decks (driver_airbag, TANK,
   blast_experiment, DBEND_44) now read all their materials with zero
   /MAT errors.
3. **First material-physics pack** (upstream kernels ported, registered,
   analytically tested — 27 new tests): /MAT/VOID, /MAT/GAS,
   /MAT/LAW70 (FOAM_TAB), /MAT/LAW35 (FOAM_VISC), /MAT/LAW44 (COWPER),
   /MAT/KELVINMAX (LAW40). The RD-V-0220_Foam_LAW70 oracle deck runs
   end-to-end: Starter clean on all 4 variants, variant 3 truncated
   engine run NORMAL at −0.05 % energy error (variant 0 goes unstable at
   ~80 % crush — an Isolid24/HEPH *element* gap, not material).
4. **Landed but unreported** (builders crashed after finishing; verified
   here only by running their test suites — 62/62 pass): the GROUP/SET
   machinery (`/GRNOD/SURF|GRNOD|GENE|GR<elem>`, element groups
   `/GRSHEL|GRSH3N|GRBRIC|GRQUAD|GRTRUS|GRBEAM|GRSPRI` + `/GRPART/PART`,
   `/SURF/SURF|GRSHEL|GRSH3N`, `/LINE/EDGE|LINE|PART`, `/FUNCT_SMOOTH`,
   `/UNIT` + /BEGIN work units) — the exact trio M36 ranked as the
   complete hard gap for 252/529 decks — and a SECOND material-physics
   pack (/MAT/FABRI LAW19, /MAT/CONC LAW24, /MAT/LAW62, /MAT/LAW81),
   registered in the same registry. No corpus-wide measurement of either
   exists yet (§4.5, §8).
5. **THE M37 PARITY RE-RUN (§3.1): both-engine comparisons 1 → 24
   official cases** (4 MATCH + 20 DEVIATION; 32 counting the bundled
   re-run, which is byte-identical to M36 — zero regression). First
   new-physics validation: **RD-V-0220 LAW70 foam is a clean full-run
   MATCH at 0.0355**. c26 Hardening improved 0.974 → 0.170. The dominant
   remaining blocker was a one-line `/STOP` Emax=0 mis-read (fixed in
   this tree post-measurement, §3.1); timing campaign in §6.1
   (`perf_m37.json`, 122 records, contention-flagged).
6. **THE M37 CORPUS RE-SWEEP (§4.6): the parse backlog is dead
   corpus-wide.** Same 529 decks, same driver, same verdict definitions as
   M36: **parse-error incidents 858 → 0, distinct signatures 20 → 0, decks
   with parse errors 446 → 0**; verdicts **ERROR 520 → 149, SKIPS 9 → 373,
   CLEAN 0 → 7**; **371 decks (70 %) improved, 0 regressed, 0 crashes,
   0 timeouts**. Every group/set family and all 28 unsupported `/MAT`
   families closed. The blocker profile is now flat — the top remaining
   family is `/PROP/SH_ORTH` at 21 decks.
7. **Builder incidents, named honestly**: five agents were killed by
   transient API errors. `coverage-resweep` and `timed-parity-m37` were
   re-run to completion afterwards (§4.6, §3.1). `groups-sets` and
   `mat-physics-2` had already landed their work, which passes its tests,
   but filed no report (item 4). `c26-hardening` landed nothing — the
   single M36 both-engine DEVIATION case stands unexamined, and is
   deferred with the three §4.6 bugs.

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

## 3. Parity — official IN_ENVELOPE decks (M36 baseline below; §3.1 = the M37 re-run, AUTHORITATIVE)

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

### 3.1 The M37 official-parity re-run (AUTHORITATIVE)

52 cases (the 40 M36 IN_ENVELOPE + 12 M37-unlocked NEAR cases), same
driver and comparison as M36; port budget 900 s/deck. Machine-readable:
`tools/validation_data/parity_m37.json` (61 results incl. the bundled
re-run) and `perf_m37.json` (122 timing records).

**Both-engine channel comparison: 1 official case in M36 → 24 in M37**
(4 MATCH + 20 DEVIATION), 15 of them covering 100 % of the run time;
including the bundled examples, 9 → 32 compared. Tally: **4 MATCH,
20 DEVIATION, 18 PYRADIOSS-FAIL, 9 NO-CHANNELS, 1 SKIPPED-SLOW.**

Highlights:

- **c48_V0220_Foam_LAW70_2 is a clean FULL-RUN MATCH (max rel RMS
  0.0355) on the brand-new M37 LAW70 physics** — the first new-physics
  law validated end-to-end against the real solver. Its three siblings
  (c46/c47/c49) run ~32 % of the time then die on genuine numerical
  energy injection (−504 %…−16 968 %) — a LAW70 stability defect to fix,
  not a comparison artifact.
- **c26_V0200_Hardening improved 0.974 → 0.170** (the fixed-format reader
  now decodes its cards correctly), though only 1.4 % time coverage — see
  the /STOP finding below.
- The whole RD-E-1000 Bending family (BATOZ/QEPH/DKT18/BT ×9), RD-E-0100
  Twisted beam, RD-V-0300 Pressure, RD-V-0240 QUAD/TRIA (3 MATCH at
  0.035–0.039, truncated windows) and RD-HWX-T-1040 newly run on both
  engines. Shell-family deviations sit at 0.42–0.64 max rel RMS over full
  runs — the next physics-fidelity target (consistent with the M36
  box-beam hourglass finding).
- Bundled examples: **zero regression** — classes and RMS byte-identical
  to M36 (3 MATCH / 5 explained DEVIATION / gas_piston starter-reject).

**The dominant new blocker was a one-line engine bug, fixed in this
tree after the measurement**: the port read `/STOP` card `0 0 0 1 1` as
`energy_error_stop = 0.0` — a 0 % tolerance — where the real engine
treats Emax = 0 as "no user limit". The port then aborted at its FIRST
energy check (`ENERGY ERROR -50.0% EXCEEDS LIMIT 0.0%` at cycle 1 on the
nine V0700 NO-CHANNELS decks; cycle 500 on c26/c28/c31/c32/c37, whose
MATCH/DEVIATION labels therefore cover only that window). The fix
(`engine_keywords.py`, regression-tested) restores the 15 % default on
Emax = 0; the cycle-500 truncations should become full-run comparisons on
the next sweep. NOTE the V0700 decks ALSO report −50 % energy error at
cycle 1 — beyond the /STOP mis-read, that ledger anomaly under their
imposed loading is its own open item.

**Still blocked in the port starter (exact errors):** mat-less spring
parts (`/PART: material 0 not defined`, V0030), degenerate 6-node bricks
(`/BRICK: degenerated brick ... not ported`, V0240/HEXA_DEGE + P14),
the /TETRA4 volume-sign convention (§4.6 bug 3; 4 decks incl. both
official TETRA meshes), `/PROP/SH_ORTH` (blocks the LAW19 fabric pair —
the law itself parses), `/MAT/LAW2` Iflag=1 yield input (T1000), and the
fixed-format `/PROP/BEAM` section card (E0500). c34/c35/c36/c38/c39 are
mesh-only/unsolved tutorial decks the Fortran starter also rejects —
matching rejection is correct behavior.

## 4. Coverage matrix — the official corpus through the port Starter

§4.1–4.4 are the M36 sweep, kept as the baseline the M37 numbers are
measured against; **§4.6 is the authoritative M37 full-corpus
measurement** and supersedes their figures wherever the two differ.
§4.5 records the tree-side per-signature deltas (the 60-case validation
slice, /MAT parse coverage over all 193 corpus blocks) and the
landed-but-unmeasured group/set work.

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

### 4.5 M36 → M37 delta (measured on the M37 tree)

**Every §4.4 signature is fixed.** Column-aware fixed-format reading
(`card_layouts.py` + `deck_reader` dialect detection + surgical
fixed-dialect branches in `starter_keywords.py`). Corpus case counts are
the M36 `coverage_results.json` numbers; "slice" = parse-error lines on
the 60-case validation slice the fix work was driven against:

| signature | corpus cases | slice before | slice after | status | fix |
|---|---|---|---|---|---|
| /TH/NODE int('01x3') | 360 | 56 | 0 | FIXED | ids cut %10d%10d%-80s (id/skew/name), var-card continuation |
| /NODE needs-4-fields | 59 | 73 | 0 | FIXED | cut [10,20,20,20]; abutting + blank coords |
| /IMPVEL 'XX' | 35 | 9 | 0 | FIXED | XX/YY/ZZ legal (warned, condition skipped); Fscale_Y/card-2 columns |
| /SHELL int('0.0') | 28 | 38 | 0 | FIXED | conn cut at 1+nnode × %10d, real phi_s/Thick columns dropped |
| /SH3N int('0.0') | 18 | 5 | 0 | FIXED | same as /SHELL |
| /MAT/PLAS_TAB int('1.0') | 17 | 5 | 0 | FIXED | blank cards kept → real card indices stable |
| /FUNCT abutting X/Y | 13 | 9 | 0 | FIXED | point cut [20,20] |
| /EOS/IDEAL-GAS 'EOS' | 11 | 9 (all /EOS) | 0 | FIXED | title card + Gamma/P0/PSH/T0/RHO_0 layout |
| /EOS/POLYNOMIAL 'Conversion' | 8 | (in /EOS) | 0 | FIXED | title + real C0–C3 / C4 C5 E0 Psh RHO_0 two-card layout |
| /INIVEL/AXIS 'Z' | 10 | 7 (all /INIVEL) | 0 | FIXED | real DIR/FRAME/GRNOD + Vt/VR cards; frame warned |
| /INIVEL/TRA '0&V' | 3 | (in /INIVEL) | 0 | FIXED | /PARAMETER &NAME substitution + column cut |
| /TH/PART 'XXMOM' | 8 | 5 | 0 | FIXED | variable FREE_CELL_LIST spans cards; DEF + extras |
| /IMPDISP 'ZZ' | 8 | 4 | 0 | FIXED | as /IMPVEL; abutting Tstart/Tstop card-2 cut |
| /SECT int('.1') | 8 | 18 (incl PARAL) | 0 | FIXED | grnod col 31–40, N1 = moment ref, deltaT/alpha ignored |
| /SECT/PARAL int('.1') | 2 | (in /SECT) | 0 | FIXED | warn-skipped (no node group to map) |
| /MAT/PLAS_JOHNS abutting | 3 | 5 | 0 | FIXED | yield/c/m cards cut at %20lg columns; Iflag=1 refused |
| /PART index-out-of-range | 2 | 20 | 0 | FIXED | title card always present; prop/mat columns |
| /FAIL/BIQUAD int('.2') | 2 | 2 | 0 | FIXED | real card-2 layout; mat_id from /FAIL/kind/mat/fail headers |
| /RBODY int('500.0') | 2 | 2 | 0 | FIXED | Mass col 41–60, grnd col 61–70, blank ICoG → 1 |
| /DAMP int('1E-5') | 1 | 1 | 0 | FIXED | Alpha/Beta/grnod columns; Beta warned |
| **total** | **~598** | **268** | **0** | **all fixed** | 0 crashes, 0 timeouts on slice; 23/60 cases now rc=0 |

Bonus root-cause fixes surfaced by decks now parsing deeper: `/BCS`
packed Trarot (blast), `/PROP/SHELL` real N/Istrain/Thick columns
(3 decks), `/RWALL` real Diameter-on-card-3 layout (BAT_CIR,
Front_Impact), and a fatal null-density model check (6 would-be starter
div-by-zero crashes on LAW151-style multimaterials now clean ERRORs).
The 37 slice cases still exiting rc=2 fail on genuine feature gaps
(GRNOD/SURF group machinery at slice-build time, unported laws now
parsed-but-inactive, Igap=2, Iform=2, BIQUAD presets) — the ranked_gaps
list, not parse bugs.

**Every /MAT card parses.** All 31 /MAT families in the §4.3 ranked-gap
list now parse via the cfg-driven generic reader — 193/193 non-dedicated
corpus /MAT blocks, 0 failures, 0 heuristic fallbacks (LAW37's jammed
20-char fields, LAW51's Iflag subobjects, JWL's /id/unit headers and
LAW66's ISRATE=4 CARD_LIST included). `/ALE/MAT`, `/EULER/MAT`,
`/HEAT/MAT` parse as notes attached to the material. Physics status per
family: EXISTING for the five dedicated laws (LAW1/2/27/36/42,
readers untouched); PORTED this milestone for VOID, GAS, LAW70, LAW35,
LAW44, LAW40 (reported pack 1) and LAW19/FABRI, LAW24/CONC, LAW62,
LAW81 (unreported pack 2 — present, registered, 26 tests pass); every
other law is an `InactiveMaterial` the Engine refuses to run.

**Group/set + /UNIT machinery — landed, unreported, unmeasured.** The
`groups-sets` builder failed to report but its work is complete in the
tree and its 36-test suite (including five real-corpus-deck cases:
TWISBEAM GRNOD/GRNOD chains, SBEAM GRNOD/SURF, DBEND BOX/RECTA, bike
SURF/GRSHEL, BOXBEAM LINE/EDGE) passes: `/GRNOD/SURF`, `/GRNOD/GRNOD`
(recursive, negative-id removal, cycle detection), `/GRNOD/GENE` +
`GEN_INCR`, `/GRNOD/GR<elem>`, element groups
`/GRSHEL|GRSH3N|GRBRIC|GRQUAD|GRTRUS|GRBEAM|GRSPRI` + `/GRPART/PART`,
`/SURF/SURF` (with normal-flip on negative ids), `/SURF/GRSHEL|GRSH3N`,
`/LINE/EDGE` (border-edges-only, linedge.F semantics), `/LINE/LINE`,
`/LINE/PART`, `/FUNCT_SMOOTH`, and `/UNIT` local unit systems + /BEGIN
work units (`input/units.py`, verified against the real starter on the
RD-E-2601 main_TEST4 deck per its module docstring). This is the exact
machinery §4.3 ranked #1/#2/#3 (complete hard gap for 252/529 decks,
48 %) plus #4–10. §4.6 measures how many of the 520 ERROR cases it
actually converts.

### 4.6 The M37 full-corpus re-sweep (AUTHORITATIVE)

Same 529 runnable decks, same `sweep_coverage.run_case` driver, same
verdict definitions and 120 s cap as M36; the 4 MAX_PATH decks handled by
short-path copies exactly as M36 did. Machine-readable:
`tools/validation_data/coverage_results_m37.json` (M36 schema +
`delta_vs_m36`).

| metric | M36 | M37 | delta |
|---|---:|---:|---:|
| decks swept | 529 | 529 | — |
| CLEAN | 0 | **7** | +7 |
| SKIPS(n) | 9 | **373** | +364 |
| ERROR | 520 | **149** | −371 |
| CRASH | 0 | **0** | 0 |
| TIMEOUT | 0 | **0** | 0 |
| decks with parse errors | 446 | **0** | −446 |
| parse-error incidents | 858 | **0** | −858 |
| distinct parse signatures | 20 | **0** | −20 |
| families still blocking | 126 | 75 | −51 |

Verdict migration: ERROR→SKIPS 364, ERROR→CLEAN 7, ERROR→ERROR 149,
SKIPS→SKIPS 9. **371 of 529 decks (70 %) improved; none regressed.**

**All 20 M36 parse signatures are dead** — independently verified: zero
`while reading /` occurrences across all 562 fresh corpus listings. The
largest were `/TH/NODE` (360 decks), `/NODE` (59), `/IMPVEL` (35),
`/SHELL` (28), `/SH3N` (18), `/MAT/PLAS_TAB` (17).

**Ranked-gap closures** (cases_blocking → 0): `/GRNOD/SURF` 222,
`/GRNOD/GRNOD` 200, `/LINE/EDGE` 190, `/GRSHEL/SHEL` 36, `/SURF/SURF` 35,
`/FUNCT_SMOOTH` 34, `/UNIT` 28, `/SURF/GRSHEL` 26, `/GRSH3N/SH3N` 24,
`/SURF/GRSH3N` 23, `/GRBRIC/PART` 19, `/GRNOD/GENE` 17, `/GRPART/PART` 14,
`/GRBRIC/BRIC` 12 — and **every one of the 28 unsupported `/MAT`
families** (`HYD_VISC` 21, `FABRI` 19, `GAS` 17, `VOID` 13, `CONC` 8,
`LAW51` 7, `LAW81` 7, …). Only `/SURF/PLANE` (1) and `/GRNOD/NODENS` (1)
remain of the group/set machinery.

**The blocker profile is now flat** — 75 families, none above 21 decks.
The new #1 cluster is `/PROP`: `SH_ORTH` 21, `SPR_BEAM` 20, `INJECT1` 17,
`SPR_GENE` 14, `TYPE20` 12, `VOID` 10. The 149 remaining ERROR decks are
dominated by cascades from those gaps (72 "property not defined", 68
"material not defined", 20 "model has no elements"), not by reader
defects.

**Three new bugs found** (recorded in `delta_vs_m36.new_bugs`; none is a
regression — all affected decks were ERROR in M36 and remain ERROR — but
all are newly *reachable* because the readers now get that far):

1. **M37-BUG-1 `/ADMAS` header misparsed as carrying a `unit_ID`** (22
   decks, 29 incidents). The official cfg declares
   `HEADER("/ADMAS/%d/%d", type, _ID_)` — no unit slot; the port bound
   `unit_id=<admas_ID>` and then hard-errored on decks with no `/UNIT`
   block at all (BIKERC, SEAT). **FIXED during M37 integration** (`/ADMAS`
   exempted from unit-ref recording; `read_admas` rebinds the id). The
   sweep predates this fix, so its 149 ERROR count marginally overstates
   the committed tree.
2. **M37-BUG-2 density check false-fires on multi-material ALE laws**
   (10 decks, 24 incidents): LAW51/LAW151/MULTIFLUID carry no RHO0 of
   their own — mixture density comes from submaterial references and
   volume fractions. **Deferred** (those laws have no physics yet).
3. **M37-BUG-3 `/TETRA4` zero/negative volume on 100 % of an official
   deck** (9 decks): 2166 of 2166 tetras flagged on RD-V-0020
   Cantilever_beam. A total hit rate on an official element-verification
   deck indicates a node-ordering / volume-sign convention mismatch with
   real Radioss, invisible to the port's own decks (which use the port's
   convention). Pre-existing, exposed by deeper parsing. **Deferred** —
   the highest-value item of the next milestone.

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

**M37 note (unmeasured on these decks):** several of the W12/W13 gaps
named above moved in M37 — LAW44 (Cowper–Symonds) now has ported physics
(pack 1), HYD_VISC now parses as an `InactiveMaterial`, and SURF/GRSHEL +
GRSHEL/SHEL are covered by the group/set machinery. Neither deck was
re-run for this report.

## 6. Performance (M36 baseline below; §6.1 = the M37 measurement)

### 6.1 M37 timing campaign (`perf_m37.json`, 122 records)

Same machine (i9-13900H) and single-thread rules as M36. **Contention
caveat, recorded per affected record**: every run finishing after
2026-07-17 00:35 shared the machine with the user's own 13-process MPI
simulation — those wall clocks (marked `contention_note`, incl. the whole
bundled re-run and 28 official cases) are upper bounds. Cycle counts and
parity classes are load-independent. Clean-window highlights:

| case | elems | cycles F/P | Fortran st+en s | port st+en s | ratio | port cyc/s |
|---|---:|---|---|---|---:|---:|
| c02 E1000 BATOZ Sf0.6 | 99 | 146434/55836 | 2.11+52.79 | 2.09+190.38 | 3.5× | 293 |
| c04 E1000 BATOZ Sf0.9 | 99 | 97623/37224 | 2.80+38.61 | 1.47+109.77 | 2.7× | 339 |
| c08 E1000 QEPH Sf0.8 | 99 | 107837/41877 | 1.99+23.31 | 1.21+127.66 | 5.1× | 328 |
| c40 E1000 BT1 Sf0.1 | 99 | 552667/256743 | 2.27+121.42 | 1.87+560.93 | 4.6× | 458 |
| c42 E1000 BT3 Sf0.1 | 99 | 13743478/256743 | 0.65+689.98 | 0.24+178.15 | 0.3× | 1441 |
| c06 DKT18 Sf0.2 * | 198 | 621697/245010 | 1.66+114.54 | 0.88+694.42 | 6.0× | 353 |
| c46–c49 Foam LAW70 * | 1000 | 62–108k/36–48k | ~1.8+88–155 | ~1.2+390–540 | 2.6–6.0× | ~90 |
| c37 T1040 (65k elems) * | 65439 | 2984/500 | 3.47+365.44 | 5.42+556.85 | – | 0.9 |

(`*` = contended window; full 122-record table in `perf_m37.json`.)

**The speed signal for future optimization work** (the standing side
quest): on full-run shell comparisons the port is 2–7× slower
wall-to-wall (per element-cycle it fares better — its nodal-dt runs ~2.4×
fewer cycles); contact-heavy bundled decks sit at 12–27×; and port
throughput COLLAPSES with model size — ~1 400 cyc/s at 99 shells,
~90 cyc/s at 1 000 bricks, **0.9 cyc/s at 65 k mixed elements** (c37) —
pointing at per-cycle Python overhead that scales with element-group
count, the natural profiling target. The port *starter* is consistently
faster than Fortran's (~1.2 s vs ~2 s wall incl. process startup).
Curiosity: c42 (BT3 hourglass variant) runs 0.3× — the FORTRAN engine
takes 13.7 M cycles where the port's nodal dt takes 257 k.

M37 bundled port engine times run 5–20 % above M36 with identical cycle
counts — consistent with the contention window, no regression signal.

**M37 note:** the planned M37 timing table cannot be appended — the
`timed-parity-m37` builder failed and produced no `perf_m37.json`; no
timing was harvested this milestone. The M36 table below therefore
remains both the baseline AND the latest trend point. (Nothing in M37
targeted engine speed; the new column-aware reader and cfg-driven /MAT
parsing affect starter-side wall clock only, unmeasured.) The next
timing sweep should note that decks previously stopping in the starter
(33 of the 40 IN_ENVELOPE cases, §3) will produce port ENGINE timings
for the first time.

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

## 7. Known issues & backlog (updated for M37)

The M36 backlog items 1 (fixed-format reader hardening) and 2 (group/set
machinery) are DONE in the tree (§4.5) — but unmeasured at corpus scale.
The post-M37 list, in measured-value order:

1. **Re-measure everything** (the three failed M37 builders' scope): the
   529-deck coverage re-sweep (how many of the 520 ERROR cases convert
   now that the parse backlog, /MAT parsing, and group/set machinery are
   in), the official IN_ENVELOPE parity re-run (the 33 PYRADIOSS-FAIL
   rows should now reach both engines — the first real official-deck
   physics numbers), and the timing table those runs produce for free.
2. **c26_V0200_Hardening** — the single measured both-engine case
   (DEVIATION 0.974, IE divergence at the end of the imposed-motion
   ramp) still needs its dedicated look; its assigned M37 builder failed.
3. **Contact/hourglass differential study** on the five DEVIATION
   examples (§2.2) — IE agrees, contact/hourglass ledgers do not
   (carried from M36).
4. **Material physics for the parsed-but-inactive laws**: every /MAT
   parses but only 15 laws carry physics (5 dedicated + pack 1 + the
   unreported pack 2). Highest corpus pull among the inactive: LAW6
   HYD_VISC (30 blocks), LAW51 (22), GAS-adjacent LAW151/MULTIFLUID
   (7+3), LAW11 BOUND (7), LAW37 BIPHAS (6), LAW83 (5). The registry
   hook makes each a one-module job.
5. **Reader/writer follow-ups from the M37 fix work** (builder OPEN
   items): the writer's PORT-DIALECT fallback blocks (CLOAD-with-sensor,
   SECT-with-node_ref, non-default RWALL, RBODY-not-dual-encodable)
   would be MISREAD by the new column-aware branches — no bundled
   example or corpus deck hits these (verified by grep), but the writer
   should emit them in real layout now that the reader understands it;
   with the column-aware reader, the writer could emit TYPE7/11 GAPMAX
   on its real card B and retire the gap_max-in-Tstart RESIDUE;
   /IMPVEL / /IMPDISP rotational directions (XX/YY/ZZ) parse cleanly but
   the kinematic condition is warned + skipped (the engine has vr and
   nodal inertia — a small follow-up); /SECT node_id_ref → node_ID1
   mapping documented (§8).
6. **RD-V-0220 variant 0 / HEPH**: the deck's Isolid24
   (physically-stabilized brick) maps to the port's one-point FB viscous
   brick and goes unstable at ~80 % crush — an element-technology gap
   the parity sweep will keep showing until HEPH lands. LAW44 kinematic
   hardening (C_hard/FISOKIN) not ported (warned, runs isotropic).
7. **gas_piston**: emit a positive P0 (deck_writer) and/or rebuild on a
   hydro law for full 9/9 comparability (carried). Note /EOS/IDEAL-GAS
   now also serves /MAT/GAS-on-elements (§4.5).
8. **Hybrid single-file decks**: pre-/BEGIN engine blocks need either a
   file split in the harness or a pre-/BEGIN engine-section reader
   (carried).
9. If the port reader ever learns `/TH/SECTIO`, the writer should switch
   spelling (carried).

## 8. Honest limitations of this report

- **The corpus numbers are M36 measurements over an M37 tree.** Five of
  eight M37 builders reported; three failed (§ TL;DR item 6). All
  §3/§4.1–4.4/§6 corpus-scale numbers predate the M37 fixes by
  construction; §4.5's deltas are slice-scale (60 cases) or
  domain-scale (193 /MAT blocks), not full-corpus.
- **Two builders' work is in the tree without a builder report**
  (groups-sets, mat-physics-2). Verification here is limited to running
  their test suites (36 + 26 tests, all pass, including five
  real-corpus-deck group cases) and inspecting their registrations;
  their claims (e.g. the /UNIT conversion verified against the real
  starter on main_TEST4) come from module docstrings, not from an
  independent re-run.
- **The report was written against a SHARED uncommitted working tree**
  carrying all builders' M37 edits. The pack-1/mat-reader/realformat
  suites (148 tests) and the groups/pack-2 suites (62 tests) were re-run
  green for this report. The cross-builder test regression flagged
  mid-flight (/IMPVEL//IMPDISP column detection breaking test_m13_implfric
  / test_m14_implgen::test_law42_implicit_vs_explicit_quasi_static /
  test_m15_fricmat) was fixed by the pack-1 builder; the confirming
  re-run of those three slow implicit suites was STILL EXECUTING at
  report submission (the M36-edition situation repeating) — check it
  before merging. The FULL suite was not re-run for this report (the
  pack-1 builder measured it very slow on this machine, ~30+ min, and
  stopped it at 72 % with zero failures).
- Because 39/40 official parity cases stopped in the M36 port starter,
  official-deck PORT engine timings exist for only one case (c26) — the
  Fortran-side timings for all 40 are in `perf_m36.json`; the port
  columns fill in once the re-sweep runs on the M37 tree.
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
- M37-specific caveats carried from the builder records:
  `/INTER/TYPE7|11` deliberately keep their M36 dual-dialect reading
  (the gap_max-in-Tstart residue preserved — brake_pad / notched_plate /
  rigid_impactor byte behaviour identical); `/SECT` node_id_ref in the
  real dialect maps to node_ID1 (moment reference at N1 — closer to the
  real cut-frame semantics than the port's centroid default, but the
  port's side-set /SECT still differs from the real element-cut section,
  the documented M36 deviation); `InactiveMaterial` E/nu are documented
  fallbacks (MAT_E/MAT_NU when present, else 1.0/0.3) used only for
  starter-side dt/stiffness estimates; the generic /MAT reader needs the
  cfg tree (`PYRADIOSS_HM_CFG` or `C:/OpenRadioss/hm_cfg_files/config/CFG`)
  — without it it degrades to heuristic density-only parsing (parse-clean,
  warned); GAS/LAW151/some LAW51 records legitimately carry density 0 (no
  density card in the law) — meshed parts referencing them get
  frozen-massless-node warnings, and the engine refuses anyway;
  `/MAT/GAS`'s PREDEF table and default R_igc are SI values (the port has
  no unit conversion inside the law — non-SI decks must override
  `params['R_igc']`, documented).
