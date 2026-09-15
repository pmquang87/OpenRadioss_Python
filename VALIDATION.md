# VALIDATION — differential validation of pyradioss against the Fortran OpenRadioss

*M41 edition — SHELL ELEMENT TECHNOLOGY, taking the single lead M40 §3.5 left
open ("the residual is a FORCE-physics gap, not the dt claim") and splitting it
into the three things it actually was. The headline (§3.6) is that **the
RD-E-1000 BATOZ and QEPH families now MATCH**: two NEW element kernels were
ported from upstream — `shell_qbat.py` (Ishell=12, the fully integrated
Batoz–Dhatt quad, `coqueba/`) and `shell_qeph.py` (Ishell=22/23/24, the
physically-stabilized 1-point quad, `coquez/`) — and a starter-side `/PROP/SHELL`
Ishell dispatch routes those decks off the Belytschko–Tsay fallback they had run
on since M2. c02/c03/c04 go **0.230/0.234/0.230 → 0.0126/0.0094/0.0084** and
c08/c09 go **0.233/0.231 → 5.8e-06/6.6e-06** — all five MATCH (tol 0.05), all
five now run **100 % of `/RUN` to TERMINATION NORMAL at ENERGY ERROR −0.00 %**
where M40 aborted them at ~66 % on the −15 % energy guard. The mechanism is the
one M40 predicted: QBAT integrates 2×2 in-plane so its element hourglass energy
is identically zero (Fortran prints HE = 0.000E+00; the port's max HE is 2e-7 of
IE), and QEPH books its stabilization work to INTERNAL energy with only the
damper work to the hourglass ledger — whereas the BT 1-point kernel stored
bending energy AS hourglass, which is why MATCH was unreachable by construction.
The BT family itself (c40–c45) got the missing `cdefo3.F` IHBE≤1 second-order
rigid-rotation membrane-rate correction plus the `cinmas.F` FAC=9 BT inertia
lumping: deviation roughly HALVES (c41 0.230 → 0.138) and the three Sf_0.1
variants that were NO-CHANNELS in M40 are UNBLOCKED to 89–92 % coverage — but BT
does **not** MATCH, and the milestone's forensic settlement is that it cannot:
the Fortran reference ITSELF blows up in the same transverse-w mode near its end
(HE 3.085e5 = 76 % of its own IE at t=1184, peak printed ENERGY ERROR −40.7 %),
surviving only because upstream's default energy-error stop is EP30 (`freform.F`)
while the port keeps its live −15 % guard. The M40 Sf_0.1 regression
(`task_29ec1751`) is RESOLVED at root by conditioning the guard REFERENCE with an
absolute floor mirroring `ecrit.F`'s `ABS(ENTOT1B)>EM20` — the 15 %/30 % limits
are UNCHANGED. ADDS §3.6 (M41 parity — the verdict table), §4.10 (the corpus
re-sweep + the new dispatch-regression watch), §6.5 (timings — and **the first
genuinely UNCONTENDED absolute benchmark in the project's history**, a standing
side-quest owed since M38), and refreshes §7/§8. M40's §3.5/§4.9/§6.4 and the
M39/M38/M37 baselines remain the measurement history the M41 deltas are taken
against; deliverables `parity_m41.json` / `perf_m41.json` / `perf_m41_clean.json`
(§3.6/§6.5) and `coverage_results_m41.json` (§4.10). All six M41 builders filed
reports. M35–M40 history kept below.*

- Date: 2026-07-18, branch `claude/openradioss-python-m41-shell-technology`
  (M40 baseline: commit `aebbab5`, the merge PR #40 of
  `claude/openradioss-python-m40-match-and-speed`; the M41 shell-technology edits
  live in the shared working tree, 19 modified files + the two NEW element
  kernels `pyradioss/elements/shell_qbat.py` (1305 lines) and
  `shell_qeph.py` (1010 lines), `pyradioss/common/npcompat.py`, and the
  `tests/test_m41_*.py` suite. Test collection **1002 → 1062 (+60)**: 19 QBAT +
  24 QEPH + 6 guard + 6 BT-rotation + 5 numpy-compat. **Re-run for this report on
  the final merged tree under `PYRADIOSS_BACKEND=numpy`: the full fast tier is
  GREEN — 1049 passed, 13 deselected (`-m "not slow"`), 0 failures, 734 s**, and
  the 60 new M41 tests pass individually. Three PRE-EXISTING tests were MODIFIED
  this milestone; that is called out explicitly in §8 rather than absorbed into
  "the suite is green")
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
  notes — §6.1). These three are the authoritative M37 measurements, and the
  BASELINE the M38 deltas are taken against. The M38 measurement is
  **`coverage_results_m38.json`** (the full 529-case re-sweep, M37 schema +
  `delta_vs_m37` + `error_class_signatures` — §4.7; authoritative for the M38
  corpus verdicts), **`parity_m38.json`** (the COMPLETE 52-case official
  parity re-run — resumed to completion after the builder's session ended
  at 6/52) and **`perf_m38.json`** (104 timing records, contention-flagged
  per case while the user's own MPI job shared the machine). The M39
  measurement is **`coverage_results_m39.json`** (the full 529-case re-sweep,
  M38 schema + `delta_vs_m38` — §4.8; authoritative for the M39 corpus
  verdicts), **`parity_m39.json`** (35/65 official cases re-run + the 9 bundled
  examples on the M39 tree, M38 schema + `fortran_provenance`/`port_budget_s`/
  `shell_family` — §3.3), **`perf_m39.json`** (M39 timing, every record
  `impi=12/12` contention-flagged) and **`perf_m39_speed.json`** (the SPEED
  campaign — 20 records + the physics-gate block — §6.3). Every M39 port wall
  clock carried the contention flag; parity classes, rel-RMS, verdicts and
  speedup RATIOS are load-independent. The M40 measurement is
  **`coverage_results_m40.json`** (§4.9), **`parity_m40.json`** (76 cases — §3.5)
  and **`perf_m40.json`** / **`perf_m40_clean.json`** (§6.4, RELATIVE-ONLY under
  contention). The M41 measurement is **`coverage_results_m41.json`** (the full
  529-case re-sweep, M40 schema + `delta_vs_m40` + a NEW top-level
  `shell_dispatch` regression-watch block — §4.10), **`parity_m41.json`** (81
  results: the FULL RD-E-1000 Bending family c00–c09/c40–c45 re-run fresh + the
  standing regression set + numba spot rows, with `m40_class`/`m40_max_rel_rms`
  carried per case — §3.6), **`perf_m41.json`** (parity-sweep wall clocks —
  8-worker pool, concurrent UPPER BOUNDS, not clean) and
  **`perf_m41_clean.json`** (`clean: true` — the first genuinely UNCONTENDED
  absolute benchmark in the project, 9 decks × 3 backends, 0/9 rows contended,
  the authoritative M41 timing source — §6.5).

## History — what M35 established, what M36 changed, what M37 changed, what M38 changed, what M39 changed, what M40 changed, what M41 changed

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
(§4.5) against the still-standing M36 corpus numbers. (Both re-runs were
subsequently RE-RUN to completion — §4.6 and §3.1 are the authoritative M37
measurements; the History note here is the M37 edition's own account, kept
intact.)

M38 took the M37 sweep's own #1 finding — "the blocker profile is now flat,
the top cluster is /PROP" (§4.6) — as its target. It shipped the /PROP pack
(a new cfg-driven `prop_reader.py`, the property sibling of `mat_reader.py`,
closing every /PROP family the sweep ranked), completed the LAW19 fabric chain
end-to-end through SH_ORTH orthotropy, and resolved the three bugs M37
deferred: the /TETRA4 volume-sign convention (BUG-3, resolved), the /ADMAS
unit-system misread (BUG-1, resolved), and the multi-material-ALE density
false-positive (BUG-2, improved 10→2). Two physics-stability fixes landed
alongside — the LAW70 densification hourglass instability (the Isolid24/HEPH
element gap M37 named, fixed for LAW70 bricks) and the V0700 −50% cycle-1
energy-ledger anomaly (a midstep constraint-work bug) — and c26_V0200_Hardening
was re-measured to a clean MATCH. The M38 corpus re-sweep (§4.7) measured the
unlock: ERROR 149→89, 60 more decks out of ERROR, 0 regressions. As in M37,
the parity/timing re-run builder (`parity-m38`) failed — there is no
`parity_m38.json`/`perf_m38.json`; the §3.2 parity numbers are the physics
builders' own case-level measurements, and M37's §3.1/§4.6/§6.1 remain the
authoritative corpus-scale baseline. A second builder (`tetra4-convention`)
filed no report but its BUG-3 fix landed and the sweep confirms it.

M39 is a DOUBLE milestone that took the M38 §4.7/§6.2 backlog on TWO tracks at
once. FIDELITY took the M36 box-beam finding (the port's shell hourglass
dissipation was ~2 orders of magnitude below the reference — §2.3) and the M37
"shell-fidelity family" deviations (§3.1) as its target: the shell hourglass
kernel now runs the `chvis3.F` elastic + quadratic-viscous form, and the M38
skew/frame cluster it named as the #1 gap is closed. SPEED took the M37/M38
"cycle-bound, throughput collapses with model size" finding (§6.1/§6.2 — 0.9
cyc/s at 65 k elements, the standing side-quest) as its target: a profiler
decomposed the cliff, then two optimizer builders shipped parity-proven wins.
As in M37/M38, the milestone was run by parallel builders in a shared tree and
some self-reported FAILED — but this round the pattern is stark: the three
"FAILED" builders' code is fully landed and passes 72/72 new tests, and the
authoritative measurements (§3.3, §4.8, §6.3) were produced by the coverage,
parity, profiler and speed builders that completed. The honest incidents this
round are two RED pre-existing tests left by unreconciled concurrent edits (§8),
not missing work.

M40 is MATCH + SPEED: it took the two sharpest leads the M39 §3.4 post-report
cascade left open and drove each to root, plus landed the M39 profiler's #1 speed
lever and cleared the four M39 residual items. The RD-E-1000 lead was framed by
M39 §3.4 as "the /RBODY dt floor lacks the rotational STIFR term, so it's coarser
than Fortran's; completing that term is the clean M40 lead to a full-run MATCH."
M40 completed the term EXACTLY (c04 floor 2.067e-2 → 1.64410e-2, Fortran's
printed 0.01644 to the digit) — and then DISPROVED the hypothesis by direct
experiment: a dt-floor sweep shows the abort instant is floor-independent, so the
RD-E-1000 residual is NOT the dt floor but a genuinely-excited shell hourglass
mode in the BT/BATOZ/QEPH force kernel under large accumulated roll (re-aimed at
M41, force-physics not dt). The deviation still HALVES (0.55 → 0.23) because the
dt fix makes the port's cycle count track Fortran. The LAW36-solids lead ("~19 %
gap, honestly flagged" in §3.4) was fully fixed: c19/c20 IE rel_rms 0.209 →
1.1e-06 via four branch-level material defects, and the fix re-frames M39's
"LAW2 0.023" as a partial-window figure (full-window LAW2 is 0.2090). SPEED made
numba the default (`auto`, ≥ 32 elements). The corpus re-sweep (§4.9) is flat by
design — M40's work is engine/perf-side — with the one M39-flagged SPR_PRE
blank-mass class RESOLVED (ERROR→SKIPS ×2, proven against the real Fortran
starter), 0 regressions. The honest process note inverts M39's: all six M40
builders filed reports, but a NEW regression was surfaced by the parity re-run —
the three RD-E-1000 Sf_0.1 variants (c40/c42/c44) now abort at cycle 100 on the
−15 % energy guard against ~1e-8 J energies (spawned `task_29ec1751`; the guard
must NOT be weakened — condition its startup denominator like upstream).

M41 is SHELL ELEMENT TECHNOLOGY: it took M40's single re-aimed lead — "the
RD-E-1000 residual is a FORCE-physics gap in the BT/BATOZ/QEPH kernel, not the dt
claim" — and split it into the three separable problems it actually was. **(1)
The BATOZ and QEPH decks were never running BATOZ or QEPH**: since M2 the port
had one shell kernel (Belytschko–Tsay), so Ishell=12 and Ishell=22/23/24 decks
fell back onto BT, which stores bending energy as HOURGLASS where the real
formulations do not — M40 §3.5 named this ("the port runs its BT kernel for
Ishell 12/24 decks … fully-integrated QBAT keeps HE ≡ 0") and M41 fixed it by
porting both kernels for real and adding a starter-side Ishell dispatch. Those
five variants go DEVIATION → **MATCH**, at 100 % coverage and NORMAL termination.
**(2) BT's own rate kinematics** were missing `cdefo3.F`'s IHBE≤1 second-order
rigid-rotation membrane correction; adding it (plus the `cinmas.F` FAC=9 BT
inertia lumping) roughly halves the BT deviation and unblocks the Sf_0.1
variants — but BT still does not MATCH, and the milestone's honest settlement is
that the remaining residual is *shared with the reference*: run to its own TSTOP
the Fortran engine blows up in the SAME transverse-w mode (peak printed ENERGY
ERROR −40.7 %), and only survives because upstream's default energy-error stop
threshold is EP30. **(3) The M40 Sf_0.1 regression** was an ill-conditioned guard
denominator, fixed by conditioning the guard REFERENCE on an absolute floor that
mirrors `ecrit.F` — limits unchanged. Process notes: all six builders filed
reports; the corpus sweep is bit-identical to M40 while actively exercising the
new dispatch on 160 decks; and for the first time since M38 the machine went
idle, so §6.5 carries CLEAN ABSOLUTE timings rather than ratio-only bounds.

## TL;DR

1. **THE RD-E-1000 BATOZ + QEPH FAMILIES NOW MATCH — the headline (§3.6,
   `qbat-port` + `qeph-port` + `parity-m41`).** M40 §3.5 diagnosed the residual as
   element technology: the port had ONE shell kernel (Belytschko–Tsay), so
   Ishell=12 (BATOZ) and Ishell=22/23/24 (QEPH) decks ran on a 1-point BT
   fallback that books bending energy as HOURGLASS. M41 ported both formulations
   for real — **`pyradioss/elements/shell_qbat.py`** (1305 lines, the fully
   integrated Batoz–Dhatt quad from `engine/source/elements/shell/coqueba/`:
   `clskew3.F` frame, the full `cbacoor.F` flat/warped split, `cbadef.F`
   assumed-strain operators, 2×2 in-plane Gauss × NIP through-thickness,
   `cbavisc.F` damping, `cbafori.F`/`cbaproj.F` assembly, **no hourglass block at
   all** — HE identically 0 on the element side, exactly like Fortran) and
   **`shell_qeph.py`** (1010 lines, the physically-stabilized 1-point quad from
   `coquez/`: `czcorc.F` covariant frame, `czcorp5.F` warped projection,
   `czdef.F` 8 strain + 6 hourglass rates with Flanagan–Belytschko
   orthogonalization, and `czfintn.F` CZFINTN1 — the modal stresses integrated
   with the material's OWN plane-stress moduli, elastic work booked to **EINT**
   and only the damper work to the hourglass ledger) — plus a starter-side
   `/PROP/SHELL` Ishell dispatch (`_dispatch_shell_formulations`; 12 → QBAT,
   22/23/24 → QEPH per `hm_read_prop01.F`, which folds 22/23 into 24 and keeps 12
   distinct). **Result (tol 0.05, `parity_m41.json`, both engines fresh):**

   | case | formulation | M40 rms | M41 rms | class | coverage |
   |---|---|---:|---:|---|---:|
   | c02 BATOZ Sf0.6 | QBAT Ishell=12 | 0.2298 | **0.01261** | **MATCH** | 100 % |
   | c03 BATOZ Sf0.8 | QBAT Ishell=12 | 0.2335 | **0.00945** | **MATCH** | 100 % |
   | c04 BATOZ Sf0.9 | QBAT Ishell=12 | 0.2301 | **0.00839** | **MATCH** | 100 % |
   | c08 QEPH Sf0.8 | QEPH Ishell=24 | 0.2330 | **5.84e-06** | **MATCH** | 100 % |
   | c09 QEPH Sf0.9 | QEPH Ishell=24 | 0.2309 | **6.58e-06** | **MATCH** | 100 % |

   All five now run **100 % of `/RUN` to TERMINATION NORMAL at ENERGY ERROR
   −0.00 %** (M40 aborted them at ~66 % on the −15 % guard). QEPH beats the
   target by ~4 ORDERS. The hourglass channel is the proof of mechanism: Fortran
   prints **HE = 0.000E+00 exactly** on all three QBAT references and the port's
   max HE is 2.6–3.7e-1 out of IE 1.47e6 (**2e-7** — the documented `cbavisc` dn
   work); QEPH's HE is ~9e-19. c04's cycle-0 dt is **1.64410E-02**, the Fortran
   engine's printed `0.1644E-01`, now claimed by the owning kernel's own
   condensed characteristic length.
2. **THE BT ROTATION FIX — halved, unblocked, and then SETTLED BY EXPERIMENT
   (§3.6, `bt-rotation-forensics`).** The port's BT type-1 kernel was missing
   `cdefo3.F`'s IHBE≤1 **second-order rigid-rotation membrane-rate correction**
   (upstream lines 103-130: `TMP1A=(dt/4)(VZ13−VZ24)²/(PY1+PY2)` into VX13/VX24,
   `TMP2B=(dt/4)(VZ13+VZ24)²/(PX2−PX1)` into VY13/VY24, `SIGN(MAX(ABS,EM20))`
   guards, killed when `IMPL_S>0`). Because the end-of-step corotational frame
   lags the mid-step velocities by ω·dt/2, a rolled element's out-of-plane
   velocity LEAKS into the membrane rates; linearized about a steady roll the
   missing term is an O(ω·dt) skew coupling that left rolled elements in a
   negative-damping transverse-w flutter growing out of round-off. Implemented
   bit-for-bit (gated on Ishell 0/1/2 → engine IHBE≤1; implicit gate mirrors
   `IMPL_S`), with the companion `cinmas.F` **FAC=NINE** BT-family rotational
   inertia lumping `I = m/4·(A/9 + t²/12)` (was `(t²+A)/12` for all families),
   which reproduces the Fortran starter's printed `/RBODY` inertia and moves the
   BT dt floor 1.86540e-2 → **1.95667e-2** (Fortran 2.004e-2 — the M40 §3.5 /
   §7-item-9 "7 % conservative" cut, now largely closed). **Deviation halves and
   the Sf_0.1 variants are UNBLOCKED**: c41 0.2305 → **0.1375**, c43 0.2713 →
   **0.2566**, and c40/c42/c44 go from M40's NO-CHANNELS cycle-100 aborts to
   **0.1892/0.2503/0.1902 at 89–92 % coverage**. **BUT BT DOES NOT MATCH, and the
   forensic settlement says it cannot** — see item 3.
3. **THE HONEST HEADLINE — the c41 flutter is REAL PHYSICS OF THE ROLLED STRIP IN
   BOTH ENGINES (§3.6).** Run to its own TSTOP the **Fortran reference itself**
   blows up in the same transverse-w mode: HE 2.0e-2 at t=1050 → 5.3e2 at t=1100
   → **3.085e5 = 76 % of its own IE** at its NORMAL end t=1184, **peak printed
   ENERGY ERROR −40.7 %** at t=1158. It reaches TSTOP only because upstream's
   default energy-error STOP threshold is effectively infinite — `freform.F` line
   782 `IF(DEMXS==ZERO) DEMXS=EP30` — while the port keeps its live −15 % guard
   (NOT weakened). With the fix the port's onset sits at or below Fortran's
   (mini-roll growth 3.8–5.5 vs Fortran 4.6–6.5 decades/100 ms; late exponents
   equal ~11–12 where the pre-fix port ran 18.5), and c41's IE tracks Fortran to
   **0.8 %** where M40 was 19 % off. **MATCH < 0.05 is unreachable for this family
   while the reference's own final window is diverging** — the same class of
   experiment-settled conclusion as M40's dt finding. The settled residual is the
   **0.14–0.26 HE-channel deviation family**, and the honest verdict is that six
   BT variants remain DEVIATION at 88–96 % of `/RUN`.
4. **THE Sf_0.1 REGRESSION — RESOLVED AT ROOT, GUARD NOT WEAKENED (§3.6,
   `guard-conditioning`; closes M40 §3.5 (d) / `task_29ec1751`).** The fix
   conditions the guard **REFERENCE** with an absolute floor
   (`_ENERGY_START_FLOOR = 1.0e-6`, `engine.py`) that holds BOTH percentage guards
   inert until the balance reference energy clears the numerical-dust band; **the
   15 % / 30 % limits are UNCHANGED and the NaN/Inf backstop stays
   UNCONDITIONAL**. This mirrors upstream `ecrit.F` 509-515 verbatim —
   `IF(ABS(ENTOT1B)>EM20) THEN ERR=ENTOTB/ENTOT1B-ONE … ELSE ERR=ZERO ENDIF` —
   which computes the energy error ONLY when the expected total energy clears an
   absolute floor. The Fortran citations were **independently verified against the
   real `ecrit.F`/`freform.F` this milestone**, not taken on trust. Measured on
   c40/c42/c44: BEFORE abort@cycle 100 at ERR −18.2 % on IE ~4.4e-8 dust; AFTER
   the run proceeds, the guard re-arms at REF 2.9e-6 (≈ cycle 250) with ERR
   already decayed to −7 %, and **max |ERR| while ACTIVE is 7.0 % < 15 %**.
5. **THE M41 CORPUS RE-SWEEP — BIT-IDENTICAL, with a NEW dispatch-regression
   watch (§4.10, `coverage-m41`).** Same 529 decks, same driver: **CLEAN 13,
   SKIPS 440, ERROR 76 — identical to M40 in every cell**, migration matrix fully
   diagonal, **0 moved cases / 0 new error classes / 0 crashes / 0 timeouts / 0
   driver-fails**. The M41-specific question ("did the new Ishell dispatch break
   any deck that used to fall back to BT?") is answered by a new `shell_dispatch`
   block: the dispatch **FIRED on 160 of 529 decks** (44 QBAT routing 32,031
   elements; 116 QEPH routing 415,713; largest single deck 37,664 QEPH elements),
   and **all 160 kept their full record — verdict, hard_skips, error_messages,
   parse_errors, n_skipped_families — byte-for-byte identical to M40**. Verdict
   `PASS`. Scope caveat kept explicit: this sweep is STARTER-only, so it proves
   the WIRING, not the kernels' engine-time fidelity (that is §3.6's job).
6. **THE CLEAN BENCHMARK — FINALLY DISCHARGED (§6.5, `parity-m41`).** The
   standing side-quest owed since M38 is **DONE**: the user's 12-process MPI job
   was NOT running this session (sampled repeatedly, `impi=0`, CPU 0 %), the idle
   window was taken FIRST and used in isolation, and `perf_m41_clean.json` carries
   **`clean: true`, 0/9 rows contended** — the project's first CLEAN ABSOLUTE
   backend numbers. **Median numba 1.782×** (range 0.61–2.57×: notched_plate
   2.569×, rigid_impactor 2.275×, box_beam 2.031×; the sole loss is gas_piston at
   0.610×). The M40-derived `auto` rule (numba at ≥ 32 elements) is **VALIDATED**:
   `auto` picks NumPy on exactly one deck — gas_piston, 4 elements, the one deck
   where numba loses.
7. **ZERO REGRESSIONS on everything the shell work did not touch (§3.6).** Every
   re-run standing case reproduces M40 to the digit: c19 LAW36 MATCH 0.00113,
   c20 0.3156, c26 Hardening MATCH 0.0216, c13/c14 LAW2 0.2089921 (M40: 0.20899),
   springs NO-CHANNELS, c01 1.0, and every PYRADIOSS-FAIL class identical. This
   proves the guard floor (a GLOBAL `engine.py` change) and the starter dispatch
   disturbed neither solids, springs nor the known-fail set. The backend contract
   also holds: numba spot-parity **3/3** with c04 and c08 T01 **byte-identical**
   to numpy (md5 equal, max-abs 0.0e+00).
8. **WHAT DID NOT MATCH — named, not buried (§3.6 (e), §7).** Six BT variants stay
   DEVIATION 0.138–0.257 at 88–96 % coverage (item 3). **DKT18 got no kernel this
   milestone** and stays DEVIATION: c06 0.281, c07 0.258 (worst channel MOMX both),
   c00 twisted beam 0.456 — it is now the only unported shell formulation in the
   RD-E-1000 family. **c05 is a GENUINE timeout, not a harness artifact**: it
   reached 681,800 cycles and 1799.38 s of a 1800 s budget at ~55 % of `/RUN`
   (Sf_0.1 scales dt ~10×, so it needs ~1.2 M cycles — it wants a ~4000 s budget
   or a cycle-capped comparison). c01_E0500_Beam_frame remains a saturated
   rel-RMS 1.0 channel, unchanged since M39 and untouched by shell work.
9. **A PERFORMANCE COST WAS INCURRED AND IS FLAGGED (§6.5).** The new kernels have
   **no JIT kernels yet**, so on QBAT numba is SLOWER than numpy (c04 full run:
   395 s numba vs 330 s numpy — the numba path adds dispatch overhead with nothing
   to compile). QBAT is also ~3× BT per cycle (c04 solo: 330 s / 97,622 cycles =
   296 cyc/s). A `jit_kernels` pass for QBAT/QEPH — and a check on whether `auto`
   should exclude qbat groups until then — is the natural SPEED follow-up (§7).

### TL;DR — M40 edition (kept intact; §3.5 parity and §4.9 corpus sweep remain the M41 baseline)

1. **THE RD-E-1000 "MATCH ATTEMPT" — the headline (§3.5, `rbody-stifr` +
   `parity-m40`).** The M39 §3.4 cascade deferred RD-E-1000 to M40 with a precise
   hypothesis: "the /RBODY dt floor (2.067e-2) is coarser than Fortran's
   (1.644e-2) precisely because the dt fix lacks the rotational STIFR term;
   completing that term is the clean M40 lead to a full-run MATCH." M40 completed
   the term — the exact `cndt3.F` STIR = STI·(t²+A)/12 element rotational
   stiffness + the `dtnoda.F` free-node rotational step √(2·IN/STIFR) + the
   `rgbodfp.F` IFLAG=1 master gather + the BATOZ/QEPH condensed characteristic
   length (`cbacoor.F` FACDT=4/3, `czcorc.F` FACDT=5/4) — driving **c04's initial
   dt floor 2.067e-2 → 1.64410e-2 = Fortran's printed 0.01644 EXACTLY** (c02
   1.09607e-2 = 0.01096, c08 1.48837e-2 = 0.01488, all print-exact), and the
   deviation **HALVES 0.55 → 0.23** because the port's cycle count now tracks
   Fortran. **But the milestone's DIRECT EXPERIMENT disproves the hypothesis: the
   c04 abort is dt-floor-INDEPENDENT** (floor sweep 2.067/1.865/1.634/1.6441e-2 →
   aborts 1051/1079/1077/1076 ms) — reaching Fortran's floor does NOT yield the
   full run. The BT-vs-BT c41 isolation localizes it: port IE == Fortran IE to 4
   significant digits until t≈900 ms (6.480e3@300, 9.801e4@600, 3.380e5@900), then
   a transverse-w hourglass mode is genuinely EXCITED (HE 2.24e5 vs Fortran
   2.0e-2 at t=1050 — 7 orders) and the −15 % injection guard trips at t≈1075 ms.
   The residual is a BT/BATOZ/QEPH FORCE-kinematics gap under large accumulated
   roll, not the dt claim — the M41 lead. NOT a full MATCH; the RD-E-1000 family
   stays DEVIATION (c04 0.2301, c02 0.2298, c08 0.2330, c41 0.2305 — `parity_m40`).
2. **LAW36 SOLIDS — the material gap FIXED AT ROOT (§3.5, `law36-forensics` +
   `parity-m40`).** The other M39 §3.4 lead ("LAW36 solids deviate ~19 %, honestly
   flagged") is closed to **c19/c20 IE rel_rms 1.095e-06 / 1.097e-06 on the full
   30 ms window** (was 0.209–0.214 partial), five orders below target, via four
   branch-level defects in `law36_tabulated.py` vs `sigeps36.F`: **(1)** the
   per-curve Fscale (YFAC) was parsed then DROPPED — the V0700 decks tabulate
   yield in MPa with Fscale=1e-3 into GPa work units, so the port's yield was
   **1000× too high and it NEVER yielded** (/FAIL/JOHNSON inert); **(2)**
   /FAIL/JOHNSON must FREEZE damage where eps_f ≤ 0 (the deck's D1=−0.1/D2=0.6/
   D3=−1.2 makes eps_f<0 beyond triaxiality 1.4931 — Fortran keeps those points
   alive forever, the port instant-deleted them); **(3)** total pressure
   P=BULK·AMU (AMU=1/J−1) vs the port's unbounded K·tr(deps); **(4)** a slashless
   `FAIL/JOHNSON/1` header the c23 lexer swallowed. **All 10/10 element deletions
   match Fortran to 4 digits** (incl. the last at t=29.069 vs 29.07); c19 is a
   MATCH (0.00113). The fix RE-FRAMES M39's "LAW2 0.023" baseline as a
   partial-window (~t≤2.3) figure — full-window LAW2 is **0.2090** (c13), LAW36
   now sits BELOW it, and the shared LAW2 volumetric defect is flagged
   (`task_6c08e3b9`).
3. **NUMBA IS NOW THE DEFAULT (§6.4, `numba-default`).** The `accel` backend
   default flipped to **`auto`**: numba is auto-selected when it imports cleanly
   AND the model has **≥ 32 total elements** — a threshold DERIVED from
   `perf_m39_speed.json` (gas_piston 4 elem 0.46× LOSS, antenna_mast 10 elem
   1.12× marginal, tensile_bar 40 elem 1.54× the smallest robust win; 32 sits in
   the (10,40] gap, env-overridable via `PYRADIOSS_BACKEND_AUTO_MIN_ELEMENTS`) —
   with a clean NumPy fallback and an engine-listing line naming the choice.
   Auto-numba is gated to the explicit `/RUN` path; implicit stays NumPy.
   `PYRADIOSS_BACKEND`/`-backend numpy|numba` still pin either backend. **13 new
   tests pass**; parity holds (M7 13/13, a 48-brick auto-vs-numpy grid at
   **8.85e-19**, corpus sh3n/beam byte-identical, shell/brick differ only at ulp).
4. **THE M40 CORPUS RE-SWEEP (§4.9, `coverage-m40`): flat by design, 0
   regressions.** Same 529 decks, same driver: **CLEAN 13 (=), SKIPS 438 → 440,
   ERROR 78 → 76**. Migration: 76 ERROR→ERROR, 2 ERROR→SKIPS (improvements), 0
   CLEAN/SKIPS→ERROR, 0 crashes/timeouts/driver-fails/parse-errors. The 2
   improvements are RD-HWX-T-1010 cantilever_completed + its DYREL variant — the
   **M39-BUG-SPRPRE error class is RESOLVED** (2 → 0). M40's landed work is
   engine/perf-side, so it correctly moves no starter verdict beyond that; the
   ranked gaps are byte-identical to M39. `coverage_results_m40.json`.
5. **THE RESIDUALS PACK — the four M39 deferred items closed (`residuals-pack`).**
   **(1)** the degenerate-brick engine STALL was ALREADY resolved — it was never
   the geometry, it was the M39 §3.4 `/PROP/SOLID` negative-viscosity reader bug;
   c12/c18 now run stable (t=10.46/13.62 ms, dt=3.74e-4 matching Fortran), no
   `solid_hexa8.py` change needed. **(2)** SPR_PRE blank mass proven Fortran-LEGAL
   by running the real `starter_win64.exe` (0 errors, MASS=0.000) → TYPE32 removed
   from `_MASS_REQUIRED_SPRING_TYPES` (the §4.9 improvement). **(3)** a rotational
   `/IMPVEL`-naming-a-skew `IndexError` fixed (drives `vr` about axis dof−3 per
   `fixvel.F`). **(4)** TYPE32 pretensioner physics assessed tractable but
   DEFERRED (a full active-spring port), left `InactiveProperty`.
6. **THE Sf_0.1 REGRESSION — named honestly (§3.5, `parity-m40`).** The three
   RD-E-1000 Sf_0.1 variants (c40/c42/c44, BT_type1/3/4) REGRESSED under the STIFR
   change from M39's ~256 k-cycle runs to **cycle-100 startup aborts** on the
   −15 % energy guard — the guard fires on ~1e-8 J absolute energies (IE 3.75e-08,
   an ill-conditioned %-based check at slow-loading startup). Fortran runs them to
   NORMAL. Spawned **`task_29ec1751`** for the STIFR/energy-guard owner — the
   guard must NOT be weakened; condition its startup denominator like upstream.
7. **THE CLEAN BENCHMARK — RELATIVE-ONLY (§6.4, `parity-m40`).** The standing
   uncontended-timing side-quest could NOT be discharged: the user's 12-process
   MPI job ran the WHOLE session (same PIDs 2+ days). Ran the backend set
   back-to-back so the RATIO cancels steady contention: **numba median 1.751×**
   (box_beam 2.10×, notched_plate 2.39×, spot_weld 1.75×, ~1.0× on the 10-elem
   antenna_mast — which `auto` correctly runs on NumPy). Absolute seconds are
   UPPER BOUNDS. TOOLING NOTE for future milestones: `tasklist /FI "IMAGENAME eq
   engine_win64_impi.exe"` returns 0 on this box despite 12 live ranks — the M39
   speed harness under-counted contention; the M40 records use plain `tasklist`.
8. **NO REGRESSIONS on unaffected cases.** c26_V0200_Hardening holds **MATCH
   0.0216**, c01 DEVIATION 0.9995, and every starter/engine-fail class is
   identical to M39. The numba spot-parity (4 decks) agrees with the numpy primary
   at physics tolerance; `auto` verified on box_beam (200 elem → "numba (auto:
   200 elements ≥ 32)").

### TL;DR — M39 edition (kept intact; §3.3 parity and §4.8 corpus sweep remain the M40 baseline)

1. **THE SHELL HOURGLASS FIDELITY FIX — the headline (§3.3).** The shell
   hourglass kernel replaced the port's BLT84 stiffness with the `chvis3.F`
   elastic + quadratic-viscous form (`shell_bt4._post`). On **box_beam_impact**
   (the showcase) the hourglass-energy channel goes **HE 0.584 → 0.094 — a 6×
   improvement matching the Fortran's 4.4 % dissipation exactly** (the M36
   box-beam finding, §2.3, resolved); its class stays DEVIATION only because
   EW=0.975 (contact bookkeeping, not shell) dominates the max. **notched_plate
   max rel RMS 0.431 → 0.315** (a genuine improvement); the **c50 fabric NaN
   channel → 0.609 FIXED**. HONEST reach caveat: the fix only moves decks with
   ACTIVE hourglass — the official RD-E-1000 Bending family runs hourglass-OFF
   or non-hourglass formulations (BATOZ full-integration, QEPH, DKT) and is
   **byte-identical M38→M39**, so its ~0.55 residual is a SEPARATE bending /
   kinematic gap the shell fix does not touch.
2. **THE `/SKEW`//`/FRAME` CLUSTER IS 100 % CLOSED.** A new
   `pyradioss/model/skew.py` (with readers in `starter_keywords.py` +
   `initialization.py`) supports `/SKEW/FIX`, `/SKEW/MOV`, `/FRAME/FIX`,
   `/FRAME/MOV` — the M38 §4.7 #1 gap. **cases_blocking 47 → 0** across all four
   families; **+33 decks now at zero hard-skips**. skew.py works end-to-end:
   c53 Snap-through (SKEW/FIX) RUNS and compares DEVIATION 0.346; four RD-E-2100
   Cam + RD-V-0530 decks reach **CLEAN**.
3. **A small-bug pack landed** (the `small-bugs` builder): degenerate `/BRICK`
   elements now **run as a collapsed hexa / TETRA4** (was rejected) — the
   biggest verdict-mover (7 decks); the **/MAT/VOID (LAW0) null-density
   exemption** extended to shells/sh3n (BAT_CIR/BAT_SQR, 2 decks); **/PROP/SPRING
   TYPE32** mass read from the card (RD-V-0031, 1 deck); and the **RBODY
   node-overlap** check relaxed to Radioss priority resolution (BIKERC).
4. **THE M39 CORPUS RE-SWEEP (§4.8): ERROR 89 → 78, SKIPS 431 → 438, CLEAN
   9 → 13.** Same 529 decks, same driver as M38: **15 decks improved verdict**
   (11 ERROR→SKIPS, 4 SKIPS→CLEAN), **0 regressions, 0 crashes, 0 timeouts, 0
   parse errors**. 5 error classes resolved, 1 new (non-regression). Deliverable
   `coverage_results_m39.json`. **The single most important interpretive point:
   gap-closed ≠ verdict-converted** — the SKEW/FRAME cluster gap is 100 % closed
   (47→0), yet only 5 of those decks changed VERDICT, because the SKEW/FRAME
   decks are predominantly MULTI-BLOCKER (they also use INTER/TYPE24, SHEL16,
   QUAD, MONVOL). This is NOT a failure of the skew work — the code is complete
   and correct; removing SKEW alone rarely clears the last ERROR-level blocker.
5. **THE M39 PARITY RE-RUN (§3.3, `parity_m39.json`): the shell-family
   before/after is the measurement.** 35/65 official cases re-run + all 9
   bundled examples on the M39 tree. The fix is VALIDATED where hourglass is
   active (box_beam HE 6×, notched_plate −0.115, fabric NaN fixed) and
   legitimately inert where it is off (RD-E-1000 byte-identical). **Regression
   gate PASS** — zero class changes on the 9 bundled examples (3 MATCH held, 6
   DEVIATION/PORT-ONLY held). Every M39 wall clock is contention-flagged
   (impi=12/12).
6. **THE SPEED PASS (§6.3) — parity-proven, NumPy reference byte-for-byte
   unchanged.** A `tools/profile_cycle.py` harness decomposed five regimes (the
   65 k-brick cliff root-caused as linear element-force cost × N, NOT a
   super-linear pathology). Then OPTIMIZER-1 (anim VTK writer 2.75–3.1×, T-file
   displacement guard) and OPTIMIZER-2 (numba LAW70-foam kernels `hexa_hgphys`
   4.46× + tabulated leaves + a fused `scatter3`) shipped measured wins behind
   the M7 backend dispatch. **OPT-2 gives 1.81× on the c46 numba cycle;** the
   re-measured numba/NumPy ratio rose on every compute-heavy deck (box_beam
   1.77→1.96×, notched_plate 2.36→2.49×, rigid_impactor 2.20→2.38×).
   **Physics-gate PASS: 9/9 bundled NumPy T01 byte-identical to the pre-speed
   reference** (`perf_m39_speed.json`).
7. **Builder incidents, named honestly.** THREE builders (`shell-fidelity`,
   `skew-frame`, `small-bugs`) self-reported **FAILED**, but their code landed
   and **passes 72/72 new M39 tests** (`tests/test_m39_*.py`) — verified here,
   not assumed. TWO PRE-EXISTING tests were RED in the shared tree from
   unreconciled concurrent signature/behavior changes and have now been
   **reconciled by the integration verifier** (§8): the parity call in
   `test_m7_backends.py::test_shell_pre_post_parity` now feeds the new
   `hqm/hqb/hqr/dt` chvis3.F signature to both backends (numpy==numba assertion
   unchanged), and the renamed
   `test_element_kernels.py::test_degenerated_brick_penta_run_as_collapsed_hexa`
   now asserts the accept-and-run behavior the small-bug pack introduced.
   Neither was a NumPy-physics regression.
8. **THE M39 POST-REPORT FIX CASCADE (§3.4) — validation-driven correctness.**
   Integrating the tree surfaced five more real, previously-masked bugs, each
   fixed as an Opus sub-agent and verified: the **`/PROP/SOLID` negative-viscosity
   reader bug** (all five V0700 brick/tetra cases now NORMAL, LAW2 solids match
   Fortran at 2.3 % — this CORRECTS the M38 "all nine V0700 freed" overclaim:
   M38 freed shells/trias, M39 frees bricks/tetras); the **`/RBODY` master-node
   timestep** (c04 dt 4.31e-2 → 2.067e-2, the t≈890 ms hourglass runaway cleared
   via the `rgbodfp.F` STIFN→master transport); **rotational `/IMPVEL` XX/YY/ZZ**
   — the TRUE dominant RD-E-1000 gap (they were parsed-then-DISCARDED, so every
   Bending deck ran UNDRIVEN; the M38 "shell-family deviations" were flat-zero
   port output, not a shell gap) — now the strip ROLLS with **EW tracking Fortran
   to 0.8 %** and c04 max_rel_rms **0.5543 → 0.2444**; and **three implicit/solid
   regressions** (the chvis3 damper as a spurious static force → shell cantilever
   tip 0.0 → 1.8994 vs analytic 1.9048; a latent NLGEOM sibling; the /INIVEL frame
   realignment). Residual (M40): c04 still aborts at t≈1051 ms on a DISTINCT later
   shell-hourglass instability — the /RBODY dt floor lacks the rotational STIFR
   term so it's coarser than Fortran's; the family is now correctly DRIVEN and 2×
   closer, the last gap localized.

### TL;DR — M38 edition (kept intact; §3.2 parity and §4.7 corpus sweep remain the M39 baseline)

1. **The /PROP pack — the M37 sweep's #1 blocker cluster — is closed.** A new
   cfg-driven reader `pyradioss/input/prop_reader.py` (the property sibling of
   `mat_reader.py`, citing each `hm_cfg_files` PROP cfg FORMAT): every /PROP
   family the M37 corpus sweep ranked as the flat blocker profile's top cluster
   now parses, and where the port's element families support it, has real
   physics. SH_ORTH (TYPE9) feeds a per-element orthotropy fiber frame into the
   BT4/tri3 shell kernels (`corthdir.F` + `mulawc.F90`/`rotov.F`); SPR_GENE
   (TYPE8) + SPR_BEAM (TYPE13) are 6-DOF linear K/C springs (`r2def3.F`); VOID
   (TYPE0) is a no-stiffness placeholder; INJECT1, TSHELL/TYPE20 and every other
   spelling parse as `InactiveProperty` the Engine refuses. Two parity blockers
   cleared: the E0500 fixed-format /PROP/BEAM section card and the mat_ID=0
   /PART rule (legal on spring parts). **23 new tests, 0 regressions** across
   363 existing.
2. **LAW19 fabric is end-to-end.** SH_ORTH completes the LAW19 chain:
   **RD-V-0230 SHELL_LAW19_PROP9** runs the complete chain — Starter NORMAL
   (**0 errors, was 6 "property not defined"**), Engine NORMAL, 384 shells with
   0°/45°/90° fibers; a kernel-level test confirms a 0°-fiber shell resists
   x-stretch **> 2×** (≈ E11/E22 = 6×) a 90°-fiber shell — the orthotropy
   wiring changes the stress, not just the parse.
3. **THE M38 CORPUS RE-SWEEP (§4.7): ERROR 149 → 89, SKIPS 373 → 431, CLEAN
   7 → 9.** Same 529 decks, same driver as M37: **60 decks converted out of
   ERROR** (58 → SKIPS, 2 → CLEAN), **0 regressions, 0 crashes, 0 timeouts, 0
   parse errors**. Every /PROP family closed at the gap level (SH_ORTH 21→0,
   SPR_BEAM 20→0, INJECT1 17→0, SPR_GENE 14→0, TYPE20 12→0, VOID 10→0, …).
   Atomic (source SHA-256 byte-identical before/after, HEAD at `977993b`).
   Deliverable `coverage_results_m38.json`.
4. **The three M37 bugs are resolved/improved.** M37-BUG-3 **/TETRA4
   volume-sign** (9 decks) FIXED — the `tetra4-convention` builder filed only a
   stub report but its `solid_tetra4.py` node-ordering/volume-sign fix landed
   and the sweep confirms **9 → 0** (the M37 landed-unreported pattern);
   M37-BUG-1 **/ADMAS unit-system** (22 decks) resolved; M37-BUG-2 **MAT
   density false-positive 10 → 2** (multi-material ALE law {51,151} exemption).
5. **LAW70 densification instability killed.** The 3 RD-V-0220 variants
   (c46/c47/c49) that died on numerical energy injection (−504 %…−16 968 % at
   ~32 %) now run stably through densification: a Belytschko–Bindeman hourglass
   STIFFNESS added for LAW70 bricks ONLY (`solid_hexa8`) — **EN identically 0**
   (was −5.29e9), **max HE ~2** (a 2.6-billion-fold reduction), ERR% ~0.0007 %,
   verified into **~95 % of peak crush**. c48 (tension, no densification)
   unchanged: NORMAL, **MATCH 0.0353** (was 0.0355). Root cause = the deck's
   Isolid=24 HEPH brick mapping to the port's one-point FB viscous brick — the
   M37 "element gap, not material" finding, fixed for LAW70 (§3.2).
6. **V0700 −50 % cycle-1 anomaly root-caused + fixed; c26 is a MATCH; LAW2
   Iflag=1 ported.** The nine V0700 SAMP decks' −50 % cycle-1 energy error was
   a midstep-bookkeeping bug — `apply_kinematic` booked constraint work at the
   endpoint velocity, not the leapfrog midstep `½J(v_old+v_imp)` that `fixvel.F`
   books (kinematics.py/engine.py fixed) — SHELL_Ishell24_LAW2 now NORMAL,
   **0.00 % every cycle**. c26_V0200_Hardening re-measured at full coverage is
   a clean **MATCH (0.0216)** — the M37 0.170 was a 1.4 %-window truncation
   artifact. /MAT/LAW2 Iflag=1 (SIG_Y/UTS/EUTS → a/b/n) ported **bit-exact vs
   Fortran** (T1000: A=0.090260 / B=0.223202 / n=0.368307), clearing the c33
   blocker (§3.2).
7. **THE M38 PARITY CAMPAIGN COMPLETED 52/52** (§3.2, `parity_m38.json` +
   `perf_m38.json`; the builder's session ended at 6/52 and the coordinator
   resumed its driver): **both-engine comparisons 24 → 28, port starter
   fails 18 → 9 (halved)**; the two full-coverage MATCHes are this
   milestone's fix showcases — c26 Hardening (0.0216) and c48 LAW70 foam
   (0.0353). M37's other "MATCH" labels are exposed as /STOP-truncation
   artifacts (full-coverage they measure 0.36–0.44, the shell-fidelity
   family). All M38 port wall clocks are CPU-contention-flagged (§6.2).
8. **Builder incidents, named honestly**: `tetra4-convention` filed a stub
   report but its /TETRA4 fix landed and is confirmed by the §4.7 sweep and
   the §3.2 tetra comparisons; `parity-m38` was resumed by the coordinator
   (item 7).

### TL;DR — M37 edition (kept intact; §3.1 parity and §4.6 corpus sweep remain the authoritative measurement baseline the M38 deltas are taken against)

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

## 3. Parity — official IN_ENVELOPE decks (M36 baseline below; §3.1 = the M37 re-run; §3.2 = M38; §3.3 = the M39 shell-fidelity re-run; §3.5 = the M40 STIFR + LAW36 re-run; §3.6 = the M41 shell-element-technology re-run, AUTHORITATIVE for M41)

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

### 3.2 M38 parity — the full 52-case re-run (AUTHORITATIVE) + case-level deltas

The `parity-m38` builder's session ended at 6/52 cases; its resume-capable
driver was resumed to completion. **`parity_m38.json` carries the full
52-case campaign** (Fortran side reused from the M37 runs — identical
binaries and comparison; port side fresh on the M38 tree; per-case
contention flags recorded while the user's own 12-process MPI job shared
the machine).

**Tally: 28 both-engine comparisons (26 DEVIATION + 2 MATCH; M37: 24),
9 PYRADIOSS-FAIL (M37: 18 — halved), 7 NO-CHANNELS, 8 SKIPPED-SLOW.**
The two full-coverage MATCHes are exactly this milestone's fix showcases:
**c26 V0200 Hardening** (the ledger fix; was a 1.4 %-window artifact) and
**c48 LAW70 foam**. Notable honest migrations: M37's c28/c31/c32 "MATCH"
labels were 0.2 %-window truncation artifacts of the /STOP bug — at full
coverage they measure as DEVIATION 0.36–0.44 (the shell-fidelity gap, same
family as the RD-E-1000 deviations); the LAW70 compression trio
(c46/c47/c49) now runs STABLY but exceeds the 900 s port budget
(SKIPPED-SLOW — the added hourglass work plus CPU contention), and the
newly-unlocked tetra/fabric/T1000 cases produce their first comparisons
(c23 tetra 0.428, c51 fabric 0.375, c33 T1000 0.177). One flag for M39:
c50's comparison yields a NaN rel-RMS (fabric channel pairing) — needs a
look before it can classify.

Below, the case-level deltas the physics builders measured on the decks
their fixes touched (same binaries and comparison):

| case | M37 (§3.1) | M38 | fix / builder |
|---|---|---|---|
| c48 V0220 Foam LAW70_2 | MATCH 0.0355 (full run) | **MATCH 0.0353** (full run, t=0.2 NORMAL) | law70-stability — fix inert in tension (HE~0) |
| c46 / c47 / c49 V0220 Foam LAW70 | DIED ~32 % — energy injection **−15254 % / −504.5 % / −16968 %** | **run stably through densification** — EN≡0 (was −5.29e9), max HE ~2 (2.6e9× reduction), ERR% ~0.0007 %, into ~95 % peak crush | law70-stability — Belytschko–Bindeman hourglass STIFFNESS, LAW70 bricks (`solid_hexa8`) |
| SHELL_Ishell24_LAW2 (+ 8 V0700 siblings) | aborted **cycle 1 — −50.0 %** | **NORMAL** t=30, 20197 cycles, **0.00 % every cycle**, 9/9 elements fail as the JC verification intends | ledger-hardening — midstep constraint-work booking (kinematics.py/engine.py) |
| c26 V0200 Hardening | DEVIATION 0.170 (1.4 % coverage) | **MATCH 0.0216** (full coverage, both engines NORMAL) | ledger-hardening — re-measurement (the 0.170 was a truncated-window artifact) |
| T1000 (c33) | starter blocker — LAW2 Iflag=1 not ported | **Starter NORMAL, 0 errors** — LAW2 a/b/n bit-exact vs Fortran | ledger-hardening — /MAT/LAW2 Iflag=1 ported |

**LAW70 detail (the headline stability fix).** The 3 dead RD-V-0220 variants
are all COMPRESSION tests (imposed disp −95 mm → ~80 % crush into
densification); the surviving c48 is a TENSION test (+50 mm) that never
densifies (variants: c46 Iflag0/Nunload1, c47 Iflag4/Shape4/Hys2, c48
Iflag4+Itens, c49 Iflag4/4-rate/Itens). Past EPS_max=1.0 the LAW70 stress
extrapolates with slope E_max=2500 (near-rigid lock-up); the deck's
/PROP/SOLID **Isolid=24** requests the HEPH physically-stabilized brick
(Fortran holds HOURGLASS ENERGY=0 the whole run), but the port maps every solid
to the one-point Flanagan–Belytschko VISCOUS hourglass brick, which resists
hourglass VELOCITY not DEFORMATION — in the lock-up the one-point geometric
coupling pumps the zero-energy modes faster than the viscous damper can bleed
them (HE 401 → 5.29e9 over ~5000 cycles → ledger −504 %…−16 968 % → abort).
Instrumented onset: element at **epst=1.28** (past EPS_max), **dt/dt_crit=0.925**
(Courant OK), **visc_num=0.066** (viscous not overshooting); and the sound speed
cancels in the viscous-hourglass stability number, so no law70-side change can
help — exactly the M37 "Isolid24/HEPH element gap, not material" finding (§7).
The fix adds the essential part of HEPH — a Belytschko–Bindeman hourglass
STIFFNESS (restoring force on the accumulated hourglass deformation, k =
HG_PHYS(0.03)·AA1·V·Σ|∇N|², AA1 = ρ₀c² tracking E0 → E_max) with its frequency
fed into the element timestep (dt_hg) and its elastic work booked into the
hourglass ledger — for LAW70 bricks ONLY (gated on `has_law70`; every non-LAW70
solid deck is byte-for-byte unchanged). Item B of the same builder resolved
M37-BUG-2 (multi-material ALE laws {51,151} exempted from the null-RHO0 fatal
check — real blast_experiment /MAT/LAW151 now raises zero density errors).

**V0700 −50 % detail.** The nine V0700 SAMP decks drive /IMPVEL curves that are
non-zero at t=0 (FUNCT/1 = exp(t/10), value 1.0 at t=0), so the driven node
jumps 0 → v_imp in cycle 1; `apply_kinematic` booked the constraint work as
J·v_imp (the ENDPOINT velocity) instead of the leapfrog-consistent midstep
J·(v_old+v_imp)/2 that `fixvel.F` books (and that the port already uses for
contact and internal work). At the impulsive start (v_old=0) J·v_imp =
m·v_imp² = exactly 2×KE, and since the energy reference is that over-booked EW,
KE sits at exactly half → −50.0 %. The measured ledger on SHELL_Ishell24_LAW2
(dt = 1.08034e-3):

| term | Fortran | port BEFORE | port AFTER |
|---|---|---|---|
| IE (internal) | 1.625e-3 | 0.0 | 0.0 |
| KE = ½m·v_imp² | 2.348e-3 | 2.34005e-3 | 2.34005e-3 |
| EW (external work) | 3.965e-3 | **4.68011e-3 (= 2·KE)** | **2.34005e-3 (= KE)** |
| ERROR % | +0.2 | **−50.00** | **0.00** |

The midstep fix resolves the V0700 **shell and tria** cases (it corrects the
constraint-work booking, which is element-independent for the −50 % symptom).
It does NOT by itself resolve the V0700 **brick and tetra** cases — those had a
SECOND, distinct defect (a `/PROP/SOLID` fixed-format reader bug detailed in
§3.4) that injected energy through a negative hourglass viscosity and was
previously MASKED by the /STOP abort. This corrects the M38 commit's
"all nine V0700 decks freed" claim: **shells/trias were freed by the ledger
midstep fix; bricks/tetras are freed by the M39 `/PROP/SOLID` reader fix (§3.4).**
c26_V0200_Hardening (Item B of the ledger builder) is a clean MATCH at full
coverage — IE matches Fortran to 0.00003 (3e-3 %), EW to 0.00002, IE+KE to
0.00003; the only non-trivial channel is MOMZ 0.0216 (2.2 %), a near-zero
momentum channel (scale 0.033 vs the energy scale 6.5e6). The deck is Chard=0
(isotropic) under monotonic loading, so the port's isotropic Johnson–Cook
radial return is exactly right; a Chard>0 kinematic-hardening warning was added
(deferred, no in-scope deck exercises it). /MAT/LAW2 Iflag=1 (Item C) ports the
`hm_read_mat02_jc.F90` SIG_Y/UTS/EUTS → a/b/n conversion: on the T1000
aluminium card (SIG_Y=0.09026, UTS=0.175, EUTS=0.24, E=60.4) the port produces
A=0.090260 / B=0.223202 / n=0.368307, matching the Fortran starter's printed
A=0.09026 / B=0.2232020270107 / N=0.3683065281433 to all printed digits.

**Caveat carried by the builders.** The full t=0.2 NORMAL termination of
c46/c47/c49 completes in the background but the long Python runs (the Fortran
itself needs 108311 cycles through densification) exceed the session's
background-task lifetime and were reaped before finishing; the instability
itself is definitively eliminated (EN≡0, max HE~2, verified past the old
failure point into ~95 % of peak crush), and c48's in-session full-run
NORMAL+MATCH is the full-run demonstration. A clean 4/4 NORMAL just needs
uninterrupted wall-clock (§7). RD-V-0230 (the LAW19 fabric oracle) reports a
−3.76 % energy error on a shortened, mass-scaled quasi-static bending run
(NORMAL termination, no instability; the strain/stress-rotation energy
invariance is verified analytically and in a unit test — this is deck/solver
character, the M37 c50/c51 family deviation, not an orthotropy accounting bug).

### 3.3 M39 parity — the shell-fidelity re-run (AUTHORITATIVE for M39)

The M38 driver was adapted to `valruns39` (reuses the stored M37 Fortran for
the 52 base cases, generates fresh / shipped-T01 references for the newly
unlocked c52+), the port was re-run fresh on the M39 tree, and results written
to `parity_m39.json` / `perf_m39.json` (M38 schema + additive per-record
`fortran_provenance` / `port_budget_s` / `shell_family`). Coverage this round:
**35/65 official re-run (29/52 base + 6/13 unlocked) + all 9 bundled examples +
26 implicit PORT-ONLY** — every element family and every M39 feature
represented; the un-run 30 are documented in §8. **Every M39 port wall clock is
contention-flagged (impi=12/12 — the user's 12-process MPI job ran the whole
sweep, so wall clocks are upper bounds); parity classes and rel-RMS are
deterministic and unaffected.**

**THE FIDELITY VERDICT: the shell hourglass fix (`chvis3.F` elastic +
quadratic-viscous replacing BLT84) is VALIDATED where hourglass is active but
NARROW in corpus reach.** The direct measure is the HE (hourglass-energy)
channel; `cov` = the port fraction of `/RUN` compared:

| case | before rms | M39 rms | rms Δ | HE before → M39 | cov |
|---|---:|---:|---:|---|---:|
| **box_beam_impact** | 0.975 | 0.975 | −5.7e-10 | **0.584 → 0.094** | 1.0 |
| **notched_plate** | 0.431 | 0.315 | **−0.115** | 0.431 → 0.315 | 1.0 |
| edge_impact | 0.456 | 0.455 | −4.3e-5 | 0.323 → 0.322 | 1.0 |
| rigid_impactor | 0.974 | 0.965 | −0.009 | 0.104 → **0.416** | 1.0 |
| rubber_block | 0.000955 | 0.000955 | 0 | 0.0116 → 0.0116 | 1.0 |
| spot_weld | 0.987 | 0.987 | +1e-8 | 0.080 → 0.080 | 1.0 |
| tensile_bar | 0.00172 | 0.00172 | 0 | 0.00059 → 0.00059 | 1.0 |
| c02/c03/c04 BATOZ | 0.554 | 0.554 | 0 | (no HE) | 1.0 |
| c08/c09 QEPH | 0.556 | 0.556 | 0 | (no HE) | 1.0 |
| c40 / c41 BT type1 | 0.637 / 0.555 | same | 0 | 0.216 / 0.187 unchanged | 1.0 |
| c44 BT type4 | 0.555 | 0.555 | 0 | 0.164 → 0.164 | 1.0 |
| c15 V0700 SHELL Ishell24 | 0.125 | 0.126 | +0.0006 | 0.750 → 0.755 | 1.0 |
| c17 V0700 TRIA | 0.123 | 0.123 | 0 | (no HE) | 1.0 |
| c28/c32 V0240 QUAD Ishell12/24 | 0.362 | 0.423 | +0.061 | 0.157 → 0.364 | 0.21 |
| c31 V0240 TRIA Ishell24 | 0.44 | SKIPPED-SLOW | — | — | — |
| c50 V0230 Fabric LAW19 | NaN | 0.609 | NaN fixed | (→ 0.609) | 0.89 |
| c51 V0230 Fabric LAW19 | 0.375 | 0.366 | −0.009 | 0.208 → 0.234 | 0.15 |

**Reading the table.** The fix works where hourglass is active — box_beam HE 6×
better (matching the Fortran's 4.4 % dissipation, the M36 §2.3 finding closed),
notched_plate max −0.115, the c50 fabric NaN fixed. It is legitimately INERT on
the RD-E-1000 corpus (hourglass-off BT decks, non-hourglass BATOZ/QEPH/DKT):
those rows are **byte-identical M38 → M39**, so that family's ~0.55 residual is
a SEPARATE bending / kinematic gap the shell fix does not address and remains
the dominant open shell deviation. Two rows are NOT clean improvements and are
reported as such: **c28/c32 moved 0.362 → 0.423 over a DIFFERENT window** (M39
reached 21 % of `/RUN` vs M38's ~7 % — the fix improved stability so the run
went further into a harder regime, so the two numbers compare different
windows); **rigid_impactor's HE worsened 0.104 → 0.416** though its class held.

**Official-class tally (35 re-run): DEVIATION 18, NO-CHANNELS 9, SKIPPED-SLOW 5,
PYRADIOSS-FAIL 2, MATCH 1.** **REGRESSION GATE (9 bundled examples): PASS** —
zero class changes (3 MATCH held — antenna_mast 0.026 / rubber_block 0.001 /
tensile_bar 0.002; 6 DEVIATION/PORT-ONLY held).

**Per-case class delta vs M38** (changed official rows — the M39 unlocks):

| case | M38 | M39 |
|---|---|---|
| c12/c18 V0700 HEXA_DEGE | PYRADIOSS-FAIL | **NO-CHANNELS** (degenerate `/BRICK` now accepted) |
| c31 V0240 TRIA (1612 elems) | DEVIATION 0.44 | **SKIPPED-SLOW** (crossed 900 s — speed regression) |
| c52 V0031 Spring_Type32 | (new) | **PYRADIOSS-FAIL** (starter ok, /PROP/SPR_PRE physics unimplemented) |
| c53 E0200 Snap_thru (SKEW/FIX) | (new) | **DEVIATION 0.346** (skew.py runs end-to-end) |
| c54 Snap_thru implicit | (new) | NO-CHANNELS (port-only) |
| c55/c56/c57 E2100 Cam (FRAME/FIX) | (new) | **SKIPPED-SLOW** (parses/runs, contact port > 900 s) |

**Unlocks, honestly qualified.** c53 skew RUNS and compares (DEVIATION 0.346 —
`skew.py` end-to-end). The degenerate-brick c12/c18 unlock is PARTIAL: the
starter now ACCEPTS the collapsed `/BRICK` (was PYRADIOSS-FAIL) but the engine
terminates at ~12 cycles / < 0.1 % of `/RUN` (NO-CHANNELS) — the element runs
but the simulation does not progress; why it stalls immediately needs a look
(§7). c52 spring TYPE32 is a PARTIAL unlock (starter parses via the mass-check
fix, but /PROP/SPR_PRE element physics is not implemented → engine-fail,
consistent with `small-bugs`). The LAW70 compression trio (c46 at 1800 s) is
STABLE (HE ≈ 0, the M38 densification fix holds) but port-throughput-limited: it
cannot finish densification even in 1800 s (timestep 4.2e-7, ~26 cyc/s). The
bottleneck is port SPEED, not stability — exactly what the §6.3 speed pass
targets.

### 3.4 The M39 post-report fix cascade (coordinator addendum)

The report above (§3.1–§3.3, §4.x, §6.x) was authored by the workflow's
report-editor over the shell/skew/speed/small-bug tracks. Integrating that
tree then surfaced a cascade of real, previously-masked bugs — each found
because a fix removed the abort that was hiding the next. All landed as
Opus sub-agent fixes on top of the M39 core (committed `2ac7f4e`), verified
individually; the combined tree passes the full fast tier. Each is a
correctness win that the port's OWN decks structurally could not reveal.

**(1) `/PROP/SOLID` reader bug → the V0700 brick/tetra energy injection.**
The qa/qb/h viscosity card was selected by "skip all-integer cards", but the
Isolid/Ismstr flag card ends in `Dn = 0.0` (a float), so it was misread AS the
viscosity card → `h = −1`: a NEGATIVE Flanagan–Belytschko hourglass viscosity
that turns the damper into an AMPLIFIER (modal velocity grew ~6×/cycle from
round-off, HOURGLASS ENERGY ran negative — impossible for a real damper). Fixed
with a fixed-format column-cut of the correct card. All five RD-V-0700
brick/tetra cases (c13/c14/c16/c19/c20/c23) now run NORMAL with HE at round-off
zero, matching Fortran's identically-zero HE. Both-engine: **LAW2 brick/tetra IE
rel_rms 0.023 (2.3 %)**; LAW36 solids deviate ~19 % — a separate LAW36 material
gap, honestly flagged (LAW2 solids on the identical geometry match at 2.3 %, so
the element approximation contributes ~2 %). This is the §3.4 correction of the
M38 V0700 overclaim.

**(2) `/RBODY` master-node timestep.** Rigid-body member nodes were dropped from
the nodal-dt minimum with NO master dt computed, so a stiff shell welded into a
rigid body ran unconstrained. On c04 (ROLLING) the port ran dt ≡ 4.31e-2 — 1.67×
over its stability limit → hourglass runaway at t≈890 ms. Ported the
`rgbodfp.F`/`rbyfor.F` STIFN→master transport (with parallel-axis) and the
`dtnoda.F` master nodal step (`mass_scaling.add_rigid_body`): c04 dt → 2.067e-2,
below the 2.585e-2 single-mode stability limit — the t≈890 ms runaway is cleanly
cleared, matching the Fortran mechanism (master node 1020 controls dt). Residual:
the port's nodal-dt machinery accumulates only TRANSLATIONAL stiffness, so its
floor (2.067e-2) is still coarser than Fortran's (1.644e-2, which includes the
shell rotational STIFR term) — see (4).

**(3) Rotational `/IMPVEL`//`/IMPDISP` (Dir = XX/YY/ZZ) — the dominant RD-E-1000
gap.** These were parsed then DISCARDED (`starter_keywords.py` warned "rotational
direction not ported — condition ignored"), so every RD-E-1000 Bending deck ran
COMPLETELY UNDRIVEN (all energies identically zero — the M38 §3.1 shell-family
0.42–0.64 "deviations" were NOT a shell-formulation gap; the port side was flat
zero). No shell-fidelity work could ever move them. Fixed across the reader
(`_IMP_DOF` maps XX/YY/ZZ → dof 3/4/5), the nodal apply (rotational entries drive
`vr` against rotational inertia at the leapfrog midstep, per `fixvel.F`'s VR/IN
branch), the rigid body (an imposed spin on the /RBODY master drives the body,
booking `dL·(w_old+w_imp)/2`), and the implicit `ur` seed — merged into the
existing /SKEW//FRAME apply path. **Result on c04: the strip now ROLLS** (IE
395104, EW 498438 — previously identically zero), the drive booking is proven
correct (**EW rel_rms 0.8 % vs Fortran**, energy error −0.39 % rules out
double-booking), and **max_rel_rms improves 0.5543 → 0.2444**.

**(4) The RD-E-1000 residual (deferred to M40).** With the rotational drive AND
the /RBODY dt fix united, c04 rolls stably PAST the t≈890 ms runaway the dt fix
targets — but a DISTINCT, later shell-hourglass instability onsets after
t≈900 ms and trips the −30 % numerical-injection guard at t≈1051 ms (57 % of the
1605 ms run; Fortran runs full). The deviation is concentrated in HE/MOMZ while
EW/IE track Fortran — downstream of the (correct) drive, a shell-hourglass /
dt-floor phenomenon: the port's dt floor (2.067e-2) is coarser than Fortran's
(1.644e-2) precisely because the /RBODY dt fix lacks the rotational STIFR term.
Completing that term (so the floor reaches Fortran's) is the clean M40 lead to a
full-run RD-E-1000 MATCH. The family is now correctly DRIVEN and 2× closer; the
last gap is quantified and localized.

**(5) Three implicit/solid regressions fixed** (surfaced by the shell + skew
work, all analytic-oracle-verified): the chvis3 viscous hourglass damper became a
spurious O(u²) static force in the implicit pseudo-velocity residual
(non-convergence read as "too stiff") — gated off in implicit-static like the
bulk viscosity, with the matching elastic rotation-hourglass moment restored to
the residual; **shell cantilever tip 0.0 → 1.8994 (analytic 1.9048, 0.28 %)**.
The same fix cleared a pre-existing NLGEOM follower-stiffness failure. And
`test_inivel_axis` was realigned — frames are now CONSUMED for /INIVEL/AXIS (an
undefined-frame reference correctly hard-errors), so its stale "frame not ported"
warning assertion was replaced with the true behavior. The explicit force path
and box_beam's corrected hourglass energy (6.86 % of IE) are unchanged by these
implicit-only edits.

### 3.5 M40 parity — the RD-E-1000 MATCH attempt + the LAW36-solids fix (AUTHORITATIVE for M40)

The M39 `valruns39` harness was adapted to `valruns40` (deterministic Fortran
reference carried; port re-run FRESH on the M40 STIFR + LAW36 + numba-default
tree, pinned to `PYRADIOSS_BACKEND=numpy` for trend-comparability, headline-first
in disjoint concurrent streams). Deliverables `parity_m40.json` (76 cases, 26
M40-fresh + 50 M39-carried for unaffected cases) / `perf_m40.json` (152 records).
**Every M40 wall clock is contention-flagged** (the user's 12-process
`engine_win64_impi` MPI job ran the whole session; parity classes and rel-RMS are
deterministic and load-independent).

**THE HEADLINE — the RD-E-1000 MATCH attempt (`rbody-stifr` landed the dt term,
`parity-m40` measured it).** M39 §3.4 deferred RD-E-1000 to M40 with a falsifiable
hypothesis: reaching Fortran's dt floor (by adding the rotational STIFR term the
M39 /RBODY fix lacked) would yield a full-run MATCH. M40 completed the term
Fortran-exactly and **DISPROVED the hypothesis by direct experiment.**

**(a) The dt term — floors are now Fortran-exact.** The port now claims the
element-side ROTATIONAL nodal stiffness `kr = 2·I_lumped/dt_e²` (shell quad/tri +
beam — the exact mirror of `cndt3.F` 209-218 `STIR = STI·(t²+A)/12`, the factor
matching the `cinmas.F` `FAC=TWELVE` inertia lumping), feeds the free-node
rotational step √(2·IN/STIFR) (`dtnoda.F` 452-471) plus the `/DT/NODA/CST`
inertia-scaling mirror (DINERT), completes the `rgbodfp.F` IFLAG=1 master gather
`K_rot = Σ(STIFR + DD·STIFN)`, and applies the BATOZ/QEPH condensed
characteristic length (`cbacoor.F` QBAT FACDT=4/3, `czcorc.F` QEPH FACDT=5/4)
times the numerical-damping factor √(1+dn²)−dn (`cncoef3.F` dn = 1e-3 QBAT /
0.015 QEPH) — gated on Ishell 12/22/24 so every BT (Ishell 1-4) deck keeps its
byte-identical claim. The initial nodal dt floor (master-governed from cycle 0):

| case | formulation | M39 floor | M40 floor | Fortran print | verdict |
|---|---|---:|---:|---:|---|
| c04 BATOZ Sf0.9 | Ishell 12 | 2.067e-2 | **1.64410e-2** | 0.01644 | **EXACT** |
| c02 BATOZ Sf0.6 | Ishell 12 | — | **1.09607e-2** | 0.01096 | **EXACT** |
| c08 QEPH Sf0.8 | Ishell 24 | — | **1.48837e-2** | 0.01488 | **EXACT** |
| c41 BT1 Sf0.9 | Ishell 1 (BT) | — | 1.86540e-2 | 0.02006 | 7 % conservative* |

(`*` c41 is on the safe side and documented: exactly mirroring the BT floor needs
`chvis3`'s STI/STIR formulas AND the `cinmas.F` FAC=9 (A/9) BT inertia lumping —
the latter changes `model.inertia` for every BT shell deck, i.e. starter mass
physics, deliberately not touched.) The c04 K_rot decomposition validates the
gather: M=72.96, IN_min=3910.656, 4 slaves → **Σdd·stifn 1.908e7 + Σstifr
4.358e6 = 2.3437e7 vs 2.3440e7 implied by Fortran (0.013 %)**.

**(b) The parity table — the deviation HALVES, but no MATCH** (`parity_m40.json`,
`PYRADIOSS_BACKEND=numpy`; M39 → M40 max_rel_rms; the Sf_0.6-0.9 rows track
Fortran's cycle count now that the dt is fixed):

| case | formulation | M39 rms | M40 rms | port cov | verdict |
|---|---|---:|---:|---:|---|
| c02 | BATOZ Sf0.6 | 0.554 | **0.2298** | 65.8 % | IMPROVED (DEVIATION) |
| c03 | BATOZ Sf0.8 | 0.554 | **0.2335** | 66.7 % | IMPROVED |
| c04 | BATOZ Sf0.9 | 0.554 | **0.2301** | 66.6 % | IMPROVED |
| c08 | QEPH Sf0.8 | 0.556 | **0.2330** | 66.5 % | IMPROVED |
| c09 | QEPH Sf0.9 | 0.556 | **0.2309** | 67.5 % | IMPROVED |
| c41 | BT_type1 Sf0.9 | 0.555 | **0.2305** | 90.5 % | IMPROVED (HE 0.230, hg/IE 0.83) |
| c43 | BT_type3 Sf0.9 | 0.632 | **0.2713** | 91.8 % | IMPROVED (IE 0.050) |
| c40 | BT_type1 Sf0.1 | 0.637 | **NO-CHANNELS** | 0 % | **REGRESSION** (cycle-100 abort) |
| c42 | BT_type3 Sf0.1 | 0.627 | **NO-CHANNELS** | 0 % | **REGRESSION** |
| c44 | BT_type4 Sf0.1 | 0.555 | **NO-CHANNELS** | 0 % | **REGRESSION** |

On c04 the residual is now concentrated in HE (EW rel_rms 0.0097, IE 0.0717,
worst-channel HE 0.2361 — Fortran's HE is identically 0 because its QBAT shell is
fully integrated); the deviation halves because the dt fix makes the port run its
BT kernel for the same cycle count as Fortran, not because the force physics
converged.

**(c) THE DIRECT EXPERIMENT — the abort is dt-floor-INDEPENDENT.** `rbody-stifr`
ran c04 with the dt floor swept across the M39 value, two intermediates, and the
new Fortran-exact value; the abort instant barely moves:

| dt floor | 2.067e-2 (M39) | 1.865e-2 | 1.634e-2 | 1.6441e-2 (Fortran-exact) |
|---|---:|---:|---:|---:|
| abort (ms) | 1051 | 1079 | 1077 | 1076 |

**Reaching Fortran's floor does NOT produce the full run** — the M39 §3.4
hypothesis is falsified. The BT-vs-BT c41 isolation (same BT formulation both
sides, removing any BATOZ/QEPH element-technology mismatch) localizes the true
cause: **port IE == Fortran IE to 4 significant digits until t≈900 ms**
(6.480e3 @ 300, 9.801e4 @ 600, 3.380e5 @ 900 ms), after which a **transverse-w
hourglass mode is genuinely EXCITED** in the port (HE 2.24e5 vs Fortran 2.0e-2 at
t=1050 — 7 orders of magnitude) and the −15 % energy-injection guard trips at
t≈1075 ms. The hourglass damper is the DEFENSE, not the cause: hr→1e-9 leaves the
abort unchanged, hf→1e-9 aborts EARLIER (958 ms). **M41 lead: the BT kernel's
rate kinematics / corotational treatment under large accumulated per-step
rotation — a FORCE-physics gap outside this milestone's dt-claim scope.** c42
(BT_type3) full run was not attempted: Fortran itself needs 13.7 M cycles (≈ 7 h
port wall), infeasible until the speed work lands.

**(d) The Sf_0.1 REGRESSION — flagged, guard NOT weakened.** The three Sf_0.1
variants (c40/c42/c44, which M39 ran to ~256 k cycles) now abort at **cycle 100**
on the −15 % energy guard, against **~1e-8 J absolute energies** (IE 3.75e-08,
HG ~1e-59) — an ill-conditioned %-based guard check at slow-loading startup.
Fortran runs them to NORMAL. Spawned **`task_29ec1751`** for the STIFR /
energy-guard owner: the −15 % limit must NOT be weakened; the fix is to condition
the denominator / startup like upstream. (This is the one honest regression of an
otherwise clean-report milestone.)

**THE SECOND HEADLINE — LAW36 solids, the ~19 % material gap FIXED AT ROOT**
(`law36-forensics` found + fixed, `parity-m40` confirmed). The M39 §3.4 flagged
"LAW2 solids match at 2.3 %, LAW36 deviate ~19 %". A branch-by-branch diff of
`materials/law36_tabulated.py` against `sigeps36.F` found FOUR defects (detailed
in the TL;DR): the dropped per-curve Fscale/YFAC (the headline — yield 1000× too
high, so the material NEVER yielded), the /FAIL/JOHNSON negative-eps_f freeze, the
`K·AMU` total-pressure split, and a slashless-`/FAIL` c23 lexer gap. Result:

| case | formulation | M39 | M40 | IE rel_rms | note |
|---|---|---|---|---:|---|
| c19 | LAW36 HEXA_18 | NO-CHANNELS (stall) | **MATCH 0.00113** | **1.095e-06** | 10/10 deletions to 4 digits (last t=29.069 vs 29.07) |
| c20 | LAW36 HEXA_24 | NO-CHANNELS (stall) | DEVIATION 0.3156 | **1.097e-06** | material PERFECT; MATCH (residual was a MOMZ momentum channel, fixed in M64 by HEPH) |
| c13 | LAW2 HEXA_18 | NO-CHANNELS (stall) | DEVIATION 0.2090 | — | un-stalled; LAW2 shares the volumetric defect (`task_6c08e3b9`) |
| c14 | LAW2 HEXA_24 | NO-CHANNELS (stall) | DEVIATION 0.2090 | — | un-stalled |
| c18 | LAW36 HEXA_DEGE | NO-CHANNELS (stall) | SKIPPED-SLOW (899 s, 491 k cyc) | — | un-stalled, throughput-limited |
| c23 | LAW36 TETRA_24 | 0 deletions (lexer) | interim IE 0.0000 (t≤14.1) | — | 6-tets-per-event 6@0.86932 vs Fortran 6@0.8688 |

**The forensic reframe (important for reading M39's numbers):** post-Fscale LAW36
equals its LAW2 twin (c13) to FOUR decimals at every window, and M39's "LAW2
0.023" baseline was a partial-window (~t≤2.3) figure — the full-window LAW2 is
**0.2090** (dominated by a shared element-side volumetric term), and the M40
pressure fix takes LAW36 BELOW that, to 1e-6. The identical LAW2 volumetric fix
is a flagged follow-up (`task_6c08e3b9`, out of the M40 LAW36-file scope). This
supersedes the M39 §3.4 "LAW36 solids deviate ~19 %" flag — root-caused and
closed. The rate-family clamp-vs-extrapolation deviation (`sigeps36.F` linearly
EXTRAPOLATES outside the rate table, the port clamps) is documented and flagged
(`task_7b31ad5f`); no V0700 case exercises it (all NRATE=1).

**numba spot-parity + no regressions.** The M40 numba-default (§6.4) was
spot-checked at physics tolerance: 4 decks' class + rel-RMS-vs-Fortran match the
numpy primary, `auto` resolves correctly (box_beam 200 elem → "numba (auto: 200
elements ≥ 32)"). Unaffected cases are unchanged M39 → M40: **c26_V0200_Hardening
MATCH 0.0216**, c01 DEVIATION 0.9995, and every starter/engine-fail identical.

**Official-class summary (26 M40-fresh):** the RD-E-1000 Sf_0.6-0.9 family HALVED
to DEVIATION (dt fixed, shell-hourglass residual, M41); 3 Sf_0.1 REGRESSIONS
(`task_29ec1751`); LAW36 c19 MATCH + c20 material-PASS (IE 1.1e-6); LAW2/DEGE
un-stalled. **NOT a full RD-E-1000 MATCH — the milestone's central attempt
succeeded at the dt claim and failed at the full-run MATCH, and the direct
experiment says why.**

### 3.6 M41 parity — the shell-element-technology re-run (AUTHORITATIVE for M41)

The M40 `valruns40` harness was carried to `valruns41`. The **FULL RD-E-1000
Bending family (c00–c09, c40–c45) was re-run FRESH on BOTH engines** on the M41
shell-technology tree, port pinned to `PYRADIOSS_BACKEND=numpy` for
trend-comparability; the standing regression set and the numba spot rows were
re-run too. Deliverable **`parity_m41.json`** (81 results, each carrying
`m40_class`/`m40_max_rel_rms` so every row is its own delta) + `perf_m41.json`.
Channels are compared over the overlap window `[0, min(fortran, port) final
time]` interpolated onto the port grid — the same semantics as M36–M40 — and a
channel is scored only if its scale is ≥ 1 % of the group scale (insignificant
channels print `~`). **Contention note, inverted from M40:** the user's
12-process `engine_win64_impi` job was NOT running at any point this session
(sampled repeatedly, `impi=0`, CPU 0 %), so §6.5's benchmark is CLEAN — but the
parity sweep itself used an 8-worker pool, so `perf_m41.json` wall clocks are
concurrent UPPER BOUNDS. Classes, rel-RMS and verdicts are deterministic
throughout.

**THE HEADLINE — the two families that got a NEW dedicated kernel are now FULLY
MATCHED.** M40 §3.5's closing sentence re-aimed RD-E-1000 at M41 as "an
element-technology port (QBAT/QEPH FORCE physics — the port runs its BT kernel
for Ishell 12/24 decks, storing bending energy partly as hourglass where
fully-integrated QBAT keeps HE ≡ 0)". That is exactly what landed, and exactly
what it bought:

| variant | formulation | M39 rms | M40 rms | **M41 rms** | class | worst ch | cov |
|---|---|---:|---:|---:|---|---|---:|
| c02 BATOZ Sf0.6 | QBAT Ishell=12 | 0.554 | 0.2298 | **0.01261** | **MATCH** | MOMZ | 100 % |
| c03 BATOZ Sf0.8 | QBAT Ishell=12 | 0.554 | 0.2335 | **0.00945** | **MATCH** | MOMZ | 100 % |
| c04 BATOZ Sf0.9 | QBAT Ishell=12 | 0.554 | 0.2301 | **0.00839** | **MATCH** | MOMZ | 100 % |
| c08 QEPH Sf0.8 | QEPH Ishell=24 | 0.556 | 0.2330 | **5.84e-06** | **MATCH** | IE | 100 % |
| c09 QEPH Sf0.9 | QEPH Ishell=24 | 0.556 | 0.2309 | **6.58e-06** | **MATCH** | IE | 100 % |
| c40 BT type1 Sf0.1 | BT + cdefo3 | 0.637 | NO-CHAN | **0.1892** | DEVIATION | MOMZ | 89.2 % |
| c41 BT type1 Sf0.9 | BT + cdefo3 | 0.555 | 0.2305 | **0.1375** | DEVIATION | HE | 89.6 % |
| c42 BT type3 Sf0.1 | BT + cdefo3 | 0.627 | NO-CHAN | **0.2503** | DEVIATION | HE | 91.5 % |
| c43 BT type3 Sf0.9 | BT + cdefo3 | 0.632 | 0.2713 | **0.2566** | DEVIATION | HE | 95.6 % |
| c44 BT type4 Sf0.1 | BT + cdefo3 | 0.555 | NO-CHAN | **0.1902** | DEVIATION | MOMZ | 89.2 % |
| c45 BT type4 Sf0.9 | BT + cdefo3 | — | — | **0.1496** | DEVIATION | HE | 87.6 % |
| c00 Twisted beam | DKT18 (untouched) | — | — | 0.4556 | DEVIATION | MOMZ | 100 % |
| c05 DKT18 Sf0.1 | DKT18 (untouched) | — | — | SKIPPED-SLOW | — | — | ~55 % |
| c06 DKT18 Sf0.2 | DKT18 (untouched) | — | — | 0.2813 | DEVIATION | MOMX | 100 % |
| c07 DKT18 Sf0.3 | DKT18 (untouched) | — | — | 0.2581 | DEVIATION | MOMX | 100 % |
| c01 Beam frame | BT + beam | 0.9995 | 0.9995 | 0.9995 | DEVIATION | — | 100 % |

**5 of the 14 scored variants MATCH, and both NEW-kernel families are 5/5.**

**(a) QBAT (Ishell=12) — `shell_qbat.py`, 1305 lines, `qbat-port`.** The full
`coqueba/` chain: `clskew3.F` corotational frame; the complete `cbacoor.F` pass
(flat/warped split at `ZL1² < 1e-12·max(L13,L24)`, explicit spin corrections,
warped per-node frames VQN + edge normals VNRM/VASTN + Gauss frames VQG/VJFI +
the DI free-rigid-mode projection, and the condensed FACDT=4/3 dt length now
computed by the OWNING kernel); `cbadef.F` assumed-strain operators (flat BM/BC
and warped BM/BMF/BF/BCQ blocks, CBADEFSH constant assumed membrane shear);
`cbastra3.F` strain increments; **2×2 in-plane Gauss × NIP through-thickness
Gauss layers** reusing `materials.shell_update` (layer state stored GP-major so
the anim/failure plumbing applies unchanged); `cbavisc.F` dn damping booked to
the `ehour`/PARTSAV(8) slot; `cbaener.F` assumed-shear energy corrections;
`cbafori.F` + CBAFORICT force assembly; `cbaproj.F` local→global with the warped
rigid-force projection; `cndt3.F` dt claim. **There is NO hourglass block** —
that is the point. Patch tests are exact to machine precision:

| patch test | closed form | port | rel err |
|---|---:|---:|---:|
| membrane `N = C·t·ε` | 1.153846e-02 | 1.153846e-02 | 1.50e-16 |
| bending `M = E·t³·κ/(12(1−ν²))` | 9.615385e-08 | 9.615385e-08 | 4.13e-16 |
| twist `Mxy = E·t³·c/(12(1+ν))` | 6.730769e-08 | 6.730769e-08 | 4.13e-15 |
| rigid rotation `\|σ\|/E` | 0 | 5.87e-13 flat / 5.95e-13 warp | quadratic order |
| HE, inviscid bending step | 0 | **0.00e+00 exactly** | — |

c04's significant channels: IE 0.00312 (final deviation 0.76 %), EW 0.00311,
MOMY 0.00554, MOMZ 0.00839 (the worst, hence the class), MASS 0.0. Final IE
deviation across the three: 0.99 / 0.71 / 0.61 %. **The hourglass channel is the
mechanism proof**: Fortran prints **HE = 0.000E+00 exactly** on all three
references, and the port's max HE is 2.6–3.7e-1 absolute out of IE 1.47e6 = **2e-7**
— the documented `cbavisc` dn work, below the harness's 1 % significance floor,
correctly printed `~`. c04 cycle count **97,622 port vs 97,623 Fortran**, and the
cycle-0 dt is **1.64410E-02** = the Fortran engine's printed `0.1644E-01`
(NODE 1020), now claimed by QBAT's own condensed length rather than the M40
BT-side approximation. 19 tests in 7 layers (`tests/test_m41_qbat.py`), all green.

**(b) QEPH (Ishell=22/23/24) — `shell_qeph.py`, 1010 lines, `qeph-port`.** Source
note worth recording: the M41 brief named `coqueph/szforc3.F`, which **does not
exist upstream** — QEPH is `engine/source/elements/shell/coquez/` (`czforc3.F`
driver; `czcorc.F` covariant frame + FACDT=5/4 + the 2nd-order rigid-rotation
correction; `czcorp5.F` warped projection with the `Z1² < LM·1e-8` plat gate;
`czdef.F` 8 strain + 6 hourglass rates with the mx13/my13 Flanagan–Belytschko
orthogonalization; `czfintce.F` constant part; `czfintn.F` CZFINTN1 stabilization;
`czproj.F` reconstruction). The `czfintn`/`czdef`/`CLSKEW3`/`czcorp5`/`czcorc`
blocks were audited line-for-line against the Fortran this session. **Why QEPH is
essentially exact where BT reported percent-level hourglass**: CZFINTN1 is a
PHYSICAL stabilization — modal stresses integrated with the material's own
plane-stress moduli A11/A12/G·SHF at CVIS=1, COEFH=0.999 plastic relaxation,
dn=0.015 linear damper — and it books the **elastic stabilization work to EINT,
with ONLY the damper work to the hourglass ledger** (`czfintn.F` EVIS(8)).

| case | F-cycles | coverage | max_rel_rms | HE port | HE Fortran | M40 baseline |
|---|---:|---:|---:|---:|---:|---|
| c08 QEPH Sf0.8 | 107,837 | 100 % | **5.841708325388154e-06** | 9.09e-19 | 0.0 | 0.2330 (aborted ~66 %, HE ~2e5) |
| c09 QEPH Sf0.9 | 95,856 | 100 % | **6.580144914995088e-06** | 9.25e-19 | 0.0 | 0.2309 |

(`parity_m41.json` records coverage 1.0 for both; the `qeph-port` builder's own
harness recorded 1.0006 — the port's final time lands marginally PAST Fortran's,
which is a full run either way.)

c08 channels: IE 5.84e-06, EW 2.06e-07, MASS 0, MOMY 3.53e-06, MOMZ 4.30e-06,
IE+KE 5.84e-06. **HE was M40's WORST channel and is now zero on both sides**, so
it drops out of scoring entirely. The dt claim (FACDT=5/4 × the `(√(1+dn²)−dn)`
damping factor) is Fortran-print-exact at 1.488e-2 on c08. Stabilization is
pinned by closed form in 24 tests (`tests/test_m41_qeph.py`, all green): it
VANISHES on arbitrary linear fields (< 1e-15) and resists h-patterns with the
physical stiffnesses membrane `A11·t/3`, bending `A11·t³/36`, transverse
`2/3·G·SHF·t` per node at rtol 1e-12, with the EINT/EHOUR split itself pinned;
plus rigid translation/spin exact zeros, the leapfrog chord artifact killed to
O(θ⁴), frame invariance 1e-12.

**(c) BT (Ishell=1–4) — the rate-kinematics fix, and its experimental
settlement (`bt-rotation-forensics`).** The defect: the port's BT type-1 kernel
never had `cdefo3.F`'s IHBE≤1 second-order rigid-rotation membrane-rate
correction (upstream lines 103-130; `IMPL_S>0` kills it, and the port's implicit
gate mirrors that). The end-of-step corotational frame lags the mid-step
velocities by ω·dt/2, so a rolled element's out-of-plane velocity leaks into the
membrane rates — an O(ω·dt) skew coupling whose absence left rolled elements in a
**negative-damping transverse-w flutter growing out of round-off**. The forensics
are specific: at t=800 the top-HE elements are symmetric pairs across the strip
(21/87, 26/92, 28/94, 31/97) in the ROLLED region (e3 tilted 30–125°), with the
transverse-w modal state `hgq[:,2]` carrying exponential growth from ~1e-9 with
**zero physical excitation**. Onset window t=850→980, HE:

| tree | t=850 | t=900 | t=950 | t=980 |
|---|---:|---:|---:|---:|
| M40 (pre-fix port) | 4.75e-10 | — | — | 5.76e+04 |
| M41 (fixed port) | 1.75e-10 | 40× lower | 1.7e3× lower | 1.46e+01 (3.9e3× lower) |
| Fortran reference | 2.84e-16 | — | — | 9.35e-08 |

A documented upstream asymmetry was mirrored rather than "corrected": the static
bias is exactly **+ω²dt/2 raw and −ω²dt/2 corrected** — upstream OVER-corrects by
2×, and it is IHBE==2/3's velocity form that is the exact cancellation. The
companion `cinmas.F` lines 916-924 **FAC=NINE** BT inertia lumping moves the dt
floor 1.86540e-2 → **1.95667e-2** (Fortran 2.004e-2), reproducing the Fortran
starter's printed `/RBODY` inertia.

**THE SETTLEMENT — and it is the honest headline of the BT track. The flutter is
REAL PHYSICS of the rolled BT strip in BOTH engines.** Run to its own TSTOP the
**Fortran reference itself** blows up in the same transverse-w mode: HE 2.0e-2 at
t=1050 → 5.3e2 at t=1100 → **3.085e5 = 76 % of its own IE** at its NORMAL end
t=1184, with a **peak printed ENERGY ERROR of −40.7 %** at t=1158. It reaches
TSTOP only because upstream's default energy-error STOP threshold is effectively
infinite (`freform.F` line 782 `IF(DEMXS==ZERO) DEMXS=EP30`, line 803
`DEMXK=EP20`) — upstream does not abort on energy error by default at all —
while the port keeps its live −15 % guard, which is **not weakened and should not
be**. With the fix the port's onset sits at or below Fortran's (mini-roll rig
growth 3.8–5.5 vs Fortran 4.6–6.5 decades/100 ms; late exponents equal at ~11–12
where the pre-fix port ran 18.5) and c41's IE tracks Fortran to **0.8 %** vs 19 %
before. **Conclusion: MATCH < 0.05 is unreachable for this family while the
reference's own final window is diverging** — comparing against it would be
comparing against a diverged signal. The residual is the **HE channel of the
final blow-up window in every case**, and the six BT variants stay DEVIATION at
88–96 % coverage. Their termination lines are recorded verbatim:

| case | port termination | ERR at stop |
|---|---|---:|
| c40 | `ERROR - ENERGY ERROR -16.2% EXCEEDS LIMIT 15.0%` | −16.16 % |
| c41 | `ERROR - NUMERICAL ENERGY INJECTION -31.3% EXCEEDS LIMIT 30.0% - RUN UNSTABLE` | −13.74 % |
| c42 | `ERROR - ENERGY ERROR -15.4% EXCEEDS LIMIT 15.0%` | −15.43 % |
| c43 | `ERROR - ENERGY ERROR -19.5% EXCEEDS LIMIT 15.0%` | −19.54 % |
| c44 | `ERROR - ENERGY ERROR -15.1% EXCEEDS LIMIT 15.0%` | −15.09 % |
| c45 | `ERROR - ENERGY ERROR -15.0% EXCEEDS LIMIT 15.0%` | −15.05 % |

Two `cdefo3` branches remain deliberately unported, and they are the concrete
targets if this family is pursued further: **c43 (BT type3)** uses the
node-1-relative velocity form and therefore takes NO rot2 correction at all today
(it gains only via the FAC=9 dt, which is why it improved least — 0.2713 →
0.2566), and **c45 (BT type4)** uses the Z2 warp correction.

**A deck-metadata correction worth recording** (it invalidates a figure quoted in
earlier editions and in the M41 brief): RD-E-1000's TSTOP is **per variant
family**, read directly out of the `_0001.rad` engine decks for this report —
BATOZ / QEPH / DKT18 run `1605.00`, BT_type1 Sf_0.9 runs `1185.01` (with
`#1640.00` commented out beneath it), BT_type1 Sf_0.1 runs `1230.01`. Coverage in
this section is the harness's `port_time_coverage` — port final time over Fortran
final time — so it is already per-deck; but any earlier prose quoting "1605" as
the BT variants' end time was wrong.

**(d) The energy-guard conditioning (`guard-conditioning`) — M40's one honest
regression, CLOSED.** M40 §3.5 (d) flagged c40/c42/c44 aborting at cycle 100 on
~1e-8 J energies and spawned `task_29ec1751` with the instruction "the guard must
NOT be weakened; condition the denominator / startup like upstream". That is
precisely what landed: `engine.py` now carries `_ENERGY_START_FLOOR = 1.0e-6`
(line 156) and wraps BOTH percentage guards — the ERR stop and the ERRN
injection guard — in `if e['REF'] > _ENERGY_START_FLOOR:`, holding them inert
until the balance REFERENCE energy clears the numerical-dust band. **The 15 % /
30 % limits are untouched, `energy_error_stop` still defaults to 15.0, and the
NAN/INF divergence check remains UNCONDITIONAL.** The upstream rule it mirrors is
`ecrit.F` 509-515 — `IF(ABS(ENTOT1B)>EM20) THEN ERR=ENTOTB/ENTOT1B-ONE … ELSE
ERR=ZERO ENDIF` — i.e. upstream computes the energy error (and any stop it
drives) only above an absolute floor. **These citations were read back in the
actual Fortran sources for this report rather than trusted from docstrings**, and
they check out, including the `X99`/99.9 % report clamp at `ecrit.F` 511/124. The
floor value is well-centred, not tuned to pass: it sits ~1.3 decades above the
cycle-100 dust (REF ~7.5e-8) and re-arms only after the true error has already
decayed below the limit. Measured on c40/c42/c44 (truncated `t_end=10`): BEFORE
abort@cycle 100 at ERR −18.2 % on IE ~4.4e-8; AFTER, the run proceeds to cycle
4600 with IE → 2.65e-2 and ERR → −0.50 %, the guard transitioning INERT
(REF 7.5e-8) → ACTIVE (REF 2.9e-6, ERR −7 %), **max |ERR| while ACTIVE 7.0 % <
15 %**. 6 tests (`tests/test_m41_guard.py`).

**(e) What did NOT improve — named.** **DKT18** received no kernel this milestone
and remains DEVIATION (c06 0.2813, c07 0.2581, worst channel MOMX on both; c00
twisted beam 0.4556) — it is now the ONLY unported shell formulation in the
RD-E-1000 family and the obvious next element-technology target. **c05** is a
GENUINE timeout rather than a harness artifact: it reached 681,800 cycles and
used 1799.38 s of an 1800 s budget at ~55 % of `/RUN`; Sf_0.1 scales dt ~10× so
the deck needs ~1.2 M cycles, and it wants either a ~4000 s budget or a
cycle-capped comparison to earn a verdict. **c01_E0500_Beam_frame** sits at
rel-RMS 0.9995 — a saturated channel unchanged since M39, flagged here only so it
is not misread as M41 fallout.

**(f) Standing regression set — ZERO regressions.** Every re-run case reproduces
M40 to the digit, which is the load-bearing evidence that a GLOBAL `engine.py`
guard change and a starter-side dispatch split disturbed nothing else:

| case | M40 | M41 | verdict |
|---|---|---|---|
| c19 LAW36 HEXA_18 | MATCH 0.00113 | MATCH 0.0011261464 | identical |
| c20 LAW36 HEXA_24 | DEVIATION 0.3156 | DEVIATION 0.3156153168 | identical |
| c26 V0200 Hardening | MATCH 0.0216 | MATCH 0.0215597355 | identical |
| c13 / c14 LAW2 HEXA | DEVIATION 0.20899 | DEVIATION 0.2089921431 | identical |
| c10 / c11 springs | NO-CHANNELS | NO-CHANNELS | identical |
| c01 Beam frame | DEVIATION 0.9995 | DEVIATION 0.9995136663 | identical |
| c30/c34/c35/c36/c38/c39/c52 | PYRADIOSS-FAIL | PYRADIOSS-FAIL | identical |

**(g) The backend contract — a backend MUST NOT change results (3/3 PASS).**

| case | numpy | numba | T01 max-abs | md5 == numpy |
|---|---|---|---:|---|
| c04 QBAT | MATCH 0.00839 | MATCH 0.00839 | 0.0e+00 | **TRUE (byte-identical)** |
| c08 QEPH | MATCH 5.84e-06 | MATCH 5.84e-06 | 0.0e+00 | **TRUE (byte-identical)** |
| c41 BT | DEVIATION 0.1375 | DEVIATION 0.1375 | 1.9e+02 | false |

c41's md5 mismatch is expected and diagnostic rather than alarming: the run
diverges at the energy-abort cycle, where a last-digit difference decides which
cycle trips the guard — class and rel-RMS agree to 5 digits (0.13750034 vs
0.13749810). The LAW36-solid control (c20) from the M40 numba spot set was NOT
re-run this session, so the solid-path backend contract is **carried from M40,
not re-proven**.

**(h) A measurement spread, disclosed.** Two sources measured the BT family and
they agree to 4 digits on c40/c43/c45 (0.1892/0.2566/0.1496) but differ on c41:
`parity_m41.json` records **0.1375** where `bt-rotation-forensics`'s focused
re-run recorded **0.1542**. This is the expected harness spread on a case that
terminates on the ERRN injection guard — the abort cycle, and therefore the
overlap window, moves with last-digit differences. §3.6 uses `parity_m41.json` as
authoritative (as §3.5 did with `parity_m40.json`) and cites the forensics
builder for mechanism only. Both values are the same class and the same
conclusion.

**Official-class summary (M41-fresh):** **5 MATCH** — the complete QBAT (c02/c03/
c04) and QEPH (c08/c09) families, all at 100 % coverage, NORMAL termination and
ENERGY ERROR −0.00 %. **6 BT variants IMPROVED but DEVIATION** (0.138–0.257 at
88–96 %; five against their M40 baseline, c45 against its M37 one of 0.6244 since
it was not re-run in M38–M40), settled by experiment as a residual shared with the
reference. **DKT18 untouched and DEVIATION**; c05 a genuine timeout. **Zero
regressions** anywhere else. Where M40's headline attempt ended in a diagnosed
non-MATCH, M41's reached MATCH on the two families it targeted — and the third
(BT) is documented as unreachable-by-comparison rather than merely unfinished.

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

### 4.7 The M38 full-corpus re-sweep (AUTHORITATIVE)

Same 529 runnable decks, same `sweep_coverage.run_case` driver (M37's verbatim
— identical verdict definitions, 120 s cap, 6 workers, resumable per-case
partials), same short-path MAX_PATH handling as §4.6. The sweep ran against the
shared M38 working tree (the M38 fixes are uncommitted; the subprocess picks
them up via PYTHONPATH). **Atomicity**: the pyradioss source SHA-256 manifest
was byte-identical before and after the 214 s sweep (`74edae99…`) and git HEAD
stayed at the M37 merge (`977993b`), so every case ran against one consistent
tree despite concurrent builders. Machine-readable:
`tools/validation_data/coverage_results_m38.json` (M37 schema + `delta_vs_m37`
+ `error_class_signatures`).

| metric | M37 (§4.6) | M38 | delta |
|---|---:|---:|---:|
| decks swept | 529 | 529 | — |
| CLEAN | 7 | **9** | +2 |
| SKIPS(n) | 373 | **431** | +58 |
| ERROR | 149 | **89** | **−60** |
| CRASH | 0 | **0** | 0 |
| TIMEOUT | 0 | **0** | 0 |
| DRIVER-FAIL | 0 | **0** | 0 |
| parse-error incidents | 0 | **0** | 0 |

Verdict migration (60 improved, 0 regressed): **ERROR→SKIPS 58, ERROR→CLEAN 2**,
ERROR→ERROR 89, SKIPS→SKIPS 373, CLEAN→CLEAN 7. The migration matrix reconciles
exactly with the verdict counts and the case-id set is identical to M37.

**Every /PROP family closed at the gap level** (cases_blocking → 0): SH_ORTH
21→0, SPR_BEAM 20→0, INJECT1 17→0, SPR_GENE 14→0, TYPE20 12→0, VOID 10→0,
CONNECT 7→0, TSHELL/TYPE34 4→0 each, SPR_PRE/FLUID 3→0 each, SH_SANDW 1→0. No
family regressed (the gap-delta worse/new list is empty).

**The three M37 bugs are resolved/improved** (`delta_vs_m37.new_bugs.m37_bug_fates`):

| bug | class | M37 | M38 | status |
|---|---|---:|---:|---|
| M37-BUG-1 | /ADMAS header misread as /UNIT ref | 22 | **0** | resolved |
| M37-BUG-2 | MAT density check on multi-material ALE laws | 10 | **2** | improved (LAW51/151 exempted; residual 2 are LAW0/VOID — M38-NEW-2) |
| M37-BUG-3 | /TETRA4 zero/negative volume on every tetra | 9 | **0** | resolved |

**Resolved error classes** (by M37 incidence): 22 `/ADMAS unknown unit system`
(BUG-1), 18 `/MAT/LAW2 Iflag=1 not ported` (the ledger-hardening LAW2 port), 10
`LAW51/LAW151/MULTIFLUID density` (BUG-2), 9 `/TETRA4 zero/negative volume`
(BUG-3), 8 `/PROP/BEAM Area/Iyy/Izz + section card missing` (the /PROP pack's
E0500 fix).

**The 89 remaining ERROR decks** are dominated by genuine unported physics, not
reader defects. Top classes: 23 `/INTER/TYPE7 Iform not ported`, 20 `model has
no elements` (the deck's only elements are unsupported families — SHEL16, QUAD,
degenerate bricks), 11 `degenerated brick (penta/pyramid)`, 10 `/ADMAS node
group not defined` (rose 2 → 10 — a deeper /ADMAS wall UNMASKED by the BUG-1
fix, status "worse" but NOT a regression), 8 `/INTER/TYPE7 Igap not ported`, 7
`material not defined` (improved from M37).

**Top M39 gap targets** (family-level, cases_blocking / sole_blocker):
INTER/TYPE24 18/5, SKEW/FIX 16/15, MONVOL/AIRBAG1 16/0, INTER/LAGMUL 14/0,
FRAME/FIX 14/3, ALE/BCS 12/7, SHEL16 12/0, QUAD 10/5, SKEW/MOV 10/8, FRAME/MOV
7/6, INTER/TYPE18 7/0. The skew/frame reference-system cluster
(SKEW/FIX + SKEW/MOV + FRAME/FIX + FRAME/MOV = 47 blocking, high sole-blocker
counts) is the highest-leverage next target; contact interfaces
(TYPE24/LAGMUL/TYPE18) and MONVOL/AIRBAG1 next; closing SHEL16/QUAD/BRICK-DEGEN
would convert most of the 20 "model has no elements" decks.

**Five new error-class signatures — four distinct issues** (the /MAT/LAW0/VOID
false-positive is recorded twice, once for shells (2 cases) and once for sh3n
(1 case); recorded in `delta_vs_m37.new_bugs.new_error_classes`; all on decks
that were ALREADY ERROR in M37 — **0 regressions**, all newly *reachable* now
that the readers get further):

| M38 cases | classification | class / decks |
|---:|---|---|
| 3 | needs-investigation | `RBODY: rigid body has no mass` — DIF24416 (Gears Inter16/17), I16S16FM (Cam fine_mesh); the port may not sum slave-element mass onto the body |
| 3 | **suspected port bug (false positive)** | `/MAT/LAW0(VOID) on shells/sh3n: zero density` — BAT_CIR, BAT_SQR; VOID is massless by design, Fortran accepts RHO0=0 — the BUG-2 fix exempted LAW51/151 but not LAW0 (M38-NEW-2) |
| 1 | **suspected port bug** | `/PROP/SPRING mass must be > 0` — RD-V-0031; the TYPE4 mass check misapplied to /PROP/SPR_PRE (TYPE32) (M38-NEW-1) |
| 1 | needs-investigation | `RBODY: node(s) already belong to another rigid body` — BIKERC; the port may be stricter than the Fortran priority resolution (M38-NEW-4) |

(§7 carries these as the M39 quick-fix backlog; a concurrent M38 LAW2-Iflag
port also cleared 18 decks' "LAW2 Iflag not ported" errors, counted above.)

### 4.8 The M39 full-corpus re-sweep (AUTHORITATIVE)

Same 529 runnable decks, same `sweep_coverage.run_case` driver (M38's verbatim —
identical verdict definitions, 120 s cap, 6 workers), same short-path MAX_PATH
handling as §4.6/§4.7. The sweep ran against the shared M39 working tree (the
M39 fixes are uncommitted; the subprocess picks them up via `python -m
pyradioss.starter`). Wall 186 s under the user's live 12-process MPI job; max
per-case 15.9 s vs the 120 s cap, so **verdicts are contention-insensitive** (no
contention-induced timeouts; the coverage sweep is starter-only, no time
integration). Machine-readable: `coverage_results_m39.json` (M38 schema +
`delta_vs_m38`). **NOTE recorded in the JSON**: the three M39 sibling builders
(shell / skew / bugs) self-reported FAILED, but this sweep measures the ACTUAL
working-tree code, whatever landed.

| metric | M38 (§4.7) | M39 | delta |
|---|---:|---:|---:|
| decks swept | 529 | 529 | — |
| CLEAN | 9 | **13** | +4 |
| SKIPS(n) | 431 | **438** | +7 |
| ERROR | 89 | **78** | **−11** |
| CRASH | 0 | **0** | 0 |
| TIMEOUT | 0 | **0** | 0 |
| DRIVER-FAIL | 0 | **0** | 0 |
| parse-error incidents | 0 | **0** | 0 |

Verdict migration (15 improved, **0 regressed**): **ERROR→SKIPS 11, SKIPS→CLEAN
4**, ERROR→ERROR 78, SKIPS→SKIPS 427, CLEAN→CLEAN 9. The case-id set is identical
to M38.

**The 15 conversions, attributed to mechanism** (which builder landed it):

| mechanism (builder) | decks | transition |
|---|---:|---|
| Degenerate bricks → collapsed hexa / TETRA4 (`small-bugs`) | 7 | ERROR→SKIPS |
| /MAT/VOID LAW0 null-density exemption (`small-bugs`) | 2 | ERROR→SKIPS |
| /PROP/SPR_PRE TYPE32 mass read from card (`small-bugs`) | 1 | ERROR→SKIPS |
| SKEW/MOV support + RBODY node-overlap fix (`skew-frame`+`small-bugs`) | 1 | ERROR→SKIPS |
| FRAME/FIX + SKEW/FIX now supported (`skew-frame`) | 4 | SKIPS→CLEAN |

The 7-deck degenerate set = 2× HEXA_DEGE + 3× RD_V_0240 (HEXA_P14/P20) + 2×
topology-opt hook_opt; VOID = BAT_CIR/BAT_SQR (Football); spring = RD-V-0031;
SKEW→CLEAN = 3× RD-E-2100 Cam + RD-V-0530 wave-propagation; SKEW+RBODY = RD-E-1200
BIKERC.

**The `/SKEW`//`/FRAME` cluster — the task headline — gap fully closed:**

| family | M38 cases_blocking | M39 cases_blocking |
|---|---:|---:|
| SKEW/FIX | 16 | **0** |
| FRAME/FIX | 14 | **0** |
| SKEW/MOV | 10 | **0** |
| FRAME/MOV | 7 | **0** |
| **TOTAL** | **47** | **0 (all closed)** |

`cases_zero_hard_skips` 385 → 418 (**+33**). But **only 5 of the 47 changed
verdict** — the rest are MULTI-BLOCKER (co-occurring INTER/TYPE24, SHEL16, QUAD,
MONVOL), so removing SKEW alone rarely cleared the last ERROR-level blocker.
**gap-closed ≠ verdict-converted** — the single most important interpretive point
of the milestone, and NOT a failure of the (complete, correct) skew work.

**Error-class delta (5 resolved, 1 new):**

| error class | M38 | M39 | status |
|---|---:|---:|---|
| BRICK DEGEN: degenerated brick (penta/pyramid) not ported | 11 | 0 | resolved |
| /MAT/LAW0 on shells: zero/missing density | 2 | 0 | resolved |
| /MAT/LAW0 on sh3n: zero/missing density | 1 | 0 | resolved |
| RBODY: node(s) already belong to another rigid body | 1 | 0 | resolved |
| /PROP/SPRING mass must be > 0 (TYPE32) | 1 | 0 | resolved |
| **/PROP/SPR_PRE mass must be > 0 (blank card)** | **0** | **2** | **NEW, not a regression** |

The one NEW class (M39-BUG-SPRPRE) is on **RD-HWX-T-1010 cantilever_completed +
its DYREL variant** — both were ERROR in M38, `is_regression=False`, and there
were **zero SKIPS/CLEAN→ERROR moves in the whole sweep**. Root cause: the M39
SPR_PRE change deliberately reads the card MASS field, and these decks' SPR_PRE/2
card has a BLANK mass (line 3031 all-spaces, Stif0=13744.468), so the
cfg-mandated `mass > 0` check fires. It IS the sole ERROR-level blocker on both.
FOLLOW-UP (M40, §7): verify whether the real Fortran starter tolerates / derives
a blank SPR_PRE pretensioner mass — if it defaults it, defaulting the blank mass
converts 2 more decks ERROR→SKIPS. (Distinct error classes 21 → 17.)

**Remaining top gaps blocking the 78 ERROR + residual SKIPS** (family-level,
cases_blocking / sole_blocker): INTER/TYPE24 18/5, MONVOL/AIRBAG1 16/2,
INTER/LAGMUL 14/2, ALE/BCS 12/7, SHEL16 12/0, QUAD 10/5, INTER/TYPE18 7/0,
MOVE_FUNCT 6/1, AMS 5/5, INTER/TYPE10 5/5. **Remaining top error classes** (the
78 ERROR): 23 `/INTER/TYPE7 Iform not ported`, 20 `model has no elements`, 10
`/ADMAS node group not defined`, 9 `/PART material not defined`, 8 `/INTER/TYPE7
Igap`, 6 `/PART property not defined`, 3 `RBODY has no mass` (UNCHANGED from M38
— DIF24416 ×2 + I16S16FM, which also carry SHEL16/BRIC20 hard skips), 3
`/INTER/TYPE11 friction filtering`. INTER/TYPE7 Iform/Igap (31 combined) is the
largest ERROR-message class but a contact-formulation FEATURE gap, not a quick
bug; "model has no elements" (20) is downstream of the unported SHEL16 / QUAD /
degenerate-brick element families.

**The shell-fidelity kernel change** (`shell_bt4._post`, +228 lines) is exercised
by the sweep with **no crashes** but does NOT affect starter verdicts — it is an
engine-side kernel change; its fidelity impact is measured by §3.3, not this
starter-only sweep.

### 4.9 The M40 full-corpus re-sweep (AUTHORITATIVE)

Same 529 runnable decks, same `sweep_coverage.run_case` driver (verbatim — 120 s
cap, 6 workers), aggregated with the M39 machinery plus a `delta_vs_m39` block
(`coverage_results_m40.json`). Batch wall 127 s under the user's live 12-process
MPI job; per-deck elapsed median 0.8 s, p95 2.4 s, max 12.4 s vs the 120 s cap
(10× headroom) — **verdicts are contention-insensitive** (contention could not
manufacture a TIMEOUT). The sweep is STARTER-only, so it cannot see M40's
engine-side fidelity gains (STIFR dt, LAW36 solids, numba-default) — those are
measured by §3.5/§6.4, not this matrix. **The result is flat by design and
CLEAN**: M40's landed work is engine/perf-side, so it correctly moves no starter
verdict except the one it directly targets.

| metric | M39 (§4.8) | M40 | delta |
|---|---:|---:|---:|
| decks swept | 529 | 529 | — |
| CLEAN | 13 | **13** | 0 |
| SKIPS(n) | 438 | **440** | +2 |
| ERROR | 78 | **76** | **−2** |
| CRASH | 0 | **0** | 0 |
| TIMEOUT | 0 | **0** | 0 |
| DRIVER-FAIL | 0 | **0** | 0 |
| parse-error incidents | 0 | **0** | 0 |

Verdict migration (2 improved, **0 regressed**): **ERROR→SKIPS 2**, ERROR→ERROR
76, SKIPS→SKIPS 438, CLEAN→CLEAN 13. **No CLEAN/SKIPS→ERROR, no `*`→CRASH/TIMEOUT/
DRIVER-FAIL, 0 new error classes.** The case-id set is identical to M39.

**The 2 improvements** are both **RD-HWX-T-1010** decks (cantilever_completed + its
DYREL variant), root-caused to the `residuals-pack` ITEM 2 fix: the /PROP/SPR_PRE
(TYPE32) mass check was over-strict. M39 raised a hard `StarterError`
("/PROP/SPR_PRE mass must be > 0") on a blank/zero SPR_PRE mass; M40's `spring.py`
sets `_MASS_REQUIRED_SPRING_TYPES = frozenset({4})` (excluding TYPE32) because
upstream `hm_read_prop32.F` reads MASS via `HM_GET_FLOATV` (blank → 0), stores it,
and enforces only MSGID 408/406 — no mass check (the `MASS>0` in
`prop_p32_spr_pre.cfg` is a HyperMesh-GUI validation, **confirmed against the real
`starter_win64.exe`: 0 errors, MASS=0.000**). The decks now complete rc=0 as
SKIPS, the property parsed and downgraded to a degrade-warning. **The one M39 new
error class (M39-BUG-SPRPRE) is RESOLVED**: `/PROP/SPR_PRE mass must be > N`
m39=2 → m40=0.

**M40 new_bugs tracker fates** (all in the JSON): M40-RBMASS 3→3 unchanged (the
`rigid body has no mass` decks — RD-E-1500 Gears Inter16/17 + RD-E-2100 Cam
I16S16FM; the M40 rotational-STIFR builder is engine-side dt transport,
ORTHOGONAL to this starter mass-accumulation check, so it correctly did not move);
M40-DEGBRICK 0→0 (stays closed, M39); M40-DENS 0→0; **M40-SPR 2→0 RESOLVED**;
M40-RBOVL 0→0.

**`ranked_gaps` are byte-for-byte identical to M39** (`gap_delta` all "unchanged")
— no coverage-envelope shift. Top remaining ERROR/SKIP blockers
(cases_blocking / sole_blocker): INTER/TYPE24 18/5, MONVOL/AIRBAG1 16/2,
INTER/LAGMUL 14/2, ALE/BCS 12/7, SHEL16 12/0, QUAD 10/5, INTER/TYPE18 7/0,
MOVE_FUNCT 6/1, EOS/LINEAR 6/0, AMS 5/5. These unported keyword families are the
coverage frontier, unchanged M39 → M40 (M40's landed work was engine/perf, not
new starter keyword coverage). The 76 ERROR decks are all gated by genuinely-
unported physics families, not port bugs.

### 4.10 The M41 full-corpus re-sweep (AUTHORITATIVE) — bit-identical, plus a new dispatch-regression watch

Same 529 runnable decks, same `sweep_coverage.run_case` driver (verbatim — 120 s
cap, 6 workers, no backend pin because the sweep runs only the STARTER, whose
verdicts are backend-independent), aggregated with the M40 machinery plus a
`delta_vs_m40` block and a **NEW top-level `shell_dispatch` section**
(`coverage_results_m41.json`). Batch wall **82 s** for all 529 decks; per-deck
elapsed median 0.60 s, p95 1.00 s, max 8.60 s against the 120 s cap — **14×
headroom, so verdicts are contention-insensitive** (contention could not
manufacture a TIMEOUT even if the MPI job had been live). This
milestone's sweep carries an unusual burden of proof: M41 changed the STARTER —
it splits `/PROP/SHELL` parts out of the generic Belytschko–Tsay group into two
new kernel groups — so unlike M40 (engine-side, flat by construction) this sweep
is directly exercising the new code on every shell deck in the corpus.

| metric | M40 (§4.9) | M41 | delta |
|---|---:|---:|---:|
| decks swept | 529 | 529 | — |
| CLEAN | 13 | **13** | 0 |
| SKIPS(n) | 440 | **440** | 0 |
| ERROR | 76 | **76** | 0 |
| CRASH | 0 | **0** | 0 |
| TIMEOUT | 0 | **0** | 0 |
| DRIVER-FAIL | 0 | **0** | 0 |
| parse-error incidents | 0 | **0** | 0 |

**The migration matrix is fully diagonal**: 13 CLEAN→CLEAN, 440 SKIPS→SKIPS, 76
ERROR→ERROR, **0 off-diagonal — zero moved cases, 0 improvements, 0 regressions,
0 new error classes**. A deeper per-case equality check across ALL 529 decks (not
just the shell ones) found **0 cases changed** in verdict, `hard_skips`,
`error_messages`, `parse_errors`, `n_skipped_families` or the
`skipped_families` key-set. 4 deep-path MAX_PATH decks were auto-rerun in
short-path copies (hyperelastic + HEXA solid decks; non-shell, fired no dispatch)
and all 4 came back SKIPS→SKIPS identical to M40.

**THE M41-SPECIFIC QUESTION — and it is a real one.** A flat table is only
reassuring if the new path actually ran. It did, heavily. Ground truth is the
port's own M41-only routing INFO line (`N /SHELL ELEMENT(S) ROUTED TO THE
{QBAT|QEPH} FORMULATION KERNEL (Ishell dispatch)`), harvested from the fresh
port-written listings into `dispatch_map.json`:

| kernel | decks | routed elements | verdicts of the firing decks |
|---|---:|---:|---|
| QBAT (Ishell=12) | 44 | 32,031 | CLEAN 1, SKIPS 43 |
| QEPH (Ishell=22/23/24) | 116 | 415,713 | CLEAN 2, SKIPS 79, ERROR 35 |
| BOTH kernels in one deck | 0 | — | — |
| **ALL firing** | **160** | **447,744** | CLEAN 3, SKIPS 122, ERROR 35 |

**All 160 dispatch-firing decks kept not merely their verdict class but their
FULL record — verdict, `hard_skips`, `error_messages`, `parse_errors`,
`n_skipped_families` — byte-for-byte identical to M40.** `shell_dispatch.verdict
= PASS`, `REGRESSIONS_LOUD` empty, `changed_cases` empty. Zero of the BT-fallback
decks broke. Even the 35 ERROR decks among the firing group error for their
pre-existing M40 reasons with identical messages: the dispatch ran successfully
on them and changed nothing. The largest firing decks are substantial models, not
toys — bumper_LL4 mono/multidomain at **37,664 QEPH elements each** (ERROR,
pre-existing), BIRD_WINDSHIELD_v1 25,394 (ERROR), Front_Impact_completed 13,598
(ERROR), BIKERC bicycle 14,134 (SKIPS), CBOX cut-model 13,952 (SKIPS).

**Tracked earlier-resolved error classes** (the re-regression watch, since the
shell-group split sits adjacent to element-group and surface resolution): RBODY
mass 3→3, degenerate brick 0→0, VOID density 0→0, SPRING mass 0→0, RBODY overlap
0→0 — all `unchanged`, `new_error_classes: 0`. `ranked_gaps` remain byte-identical
(INTER/TYPE24 18/5 sole, MONVOL/AIRBAG1 16/2, INTER/LAGMUL 14/2, ALE/BCS 12/7,
SHEL16 12, QUAD 10/5, EOS/LINEAR 6, AMS 5/5); the 76 ERROR decks are all gated by
genuinely unported physics families (certain `/INTER`, `/GRNOD/SURF` group refs),
unrelated to shell technology and outside M41 scope.

**Scope caveat, stated so the PASS is not over-read:** this sweep is
**STARTER-only**. It proves the dispatch WIRING is sound and that no deck breaks
at model setup — it does **not** measure engine-time force/energy fidelity of the
QBAT/QEPH kernels. That parity is §3.6's, measured through
`tools/validate_vs_fortran.py`, not this matrix.

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

## 6. Performance (M36 baseline below; §6.1 = the M37 measurement; §6.2 = M38 — no campaign; §6.3 = the M39 SPEED pass; §6.4 = the M40 numba-default + RELATIVE-ONLY benchmark; §6.5 = M41 — the FIRST CLEAN ABSOLUTE benchmark + the new-kernel cost)

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

### 6.2 M38 timing (`perf_m38.json`, 104 records)

The campaign completed with the resumed parity driver (Fortran wall clocks
reused from the clean M37 records; port wall clocks fresh on the M38 tree).
**Heavy caveat: every M38 port run carried the `impi=12/12` contention flag**
— the user's own 12-process MPI simulation shared the machine throughout, so
port wall clocks are upper bounds and cross-milestone wall-clock comparison
is NOT meaningful this round (cycle counts and parity classes are
load-independent and stand). Within that caveat the structural signal is
unchanged from §6.1: port 2–7× slower on full-run shells (uncontended M37
figures), throughput collapsing with element count — nothing in M38 targeted
engine speed (the /PROP reader is starter-side; the LAW70 hourglass
stiffness adds work only to LAW70 solids, and its three compression decks
now exceed the 900 s budget under contention — clean re-timing of those is
an M39 sweep item). The M36 baseline table below stands unchanged as the
last clean bundled-example measurement.

Two incidental timing observations from the physics builders, consistent with
the §6.1 "cycle-bound, not element-bound" finding:

- The LAW70 hourglass-stiffness fix (§3.2) feeds its frequency into the element
  timestep (dt_hg): **HG_PHYS=0.03 imposes a modest ~1.7× dt reduction only
  while the foam is soft** (early phase) and **NONE at densification** (the base
  Courant dt already dominates there). A future proper HEPH/Isolid=24
  assumed-strain brick would remove even that (§7).
- The c46/c47/c49 LAW70 full runs need the Fortran's **108311 cycles** through
  densification; at the port's §6.1-measured ~90 cyc/s on 1000-brick decks
  these exceed the session's background-task lifetime — the same cycle-bound
  throughput wall §6.1 flagged as the profiling target, now demonstrated on the
  densification decks.

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

### 6.3 M39 SPEED pass — profile, optimize, prove parity (the second track)

M39's second track attacked the standing §6.1/§6.2 side-quest: the port is
cycle-bound and throughput COLLAPSES with model size (0.9 cyc/s at 65 k
elements). A profiler decomposed the cost, then two optimizer builders shipped
measured, parity-proven wins behind the M7 backend dispatch — the NumPy
reference path byte-for-byte unchanged. **CONTENTION governs every absolute
number here**: the user's 12-process `engine_win64_impi` MPI job was live the
whole session (12/12 sampled), so every ms/cyc and wall clock is an UPPER BOUND;
the load-robust products are the per-stage SHARES, the alternated A/B ratios, and
the byte/md5 parity proofs. Deliverables: `tools/profile_cycle.py` (a cProfile +
monkeypatched per-stage timer harness that caps the engine loop at N cycles via a
`SkewSet.update` hook — no engine edit) and `perf_m39_speed.json` (20 records +
the physics-gate block).

**(a) The profile — where the cycle goes** (single-process, contended upper
bounds; SHARES are load-robust):

| regime | elems | ms/cyc* | µs/elem-cyc | cyc/s* | dominant stage |
|---|---:|---:|---:|---:|---|
| tensile_bar (LAW02 hexa) | 40 | 1.30 | 32.6 | 767 | elem 77 % (fixed-overhead regime) |
| c04_E1000 (BATOZ shells) | 99 | 2.63 | 26.5 | 381 | shells 67 % + rbody-kin 19 % |
| c46_LAW70 (foam bricks) | 1000 | 12.34 | 12.3 | 81 | bricks 80 % |
| **c37_T1040 (Ogden bricks — THE CLIFF)** | **65439** | **949.9** | **14.5** | **1.05** | **bricks 91.6 %** |
| rigid_impactor (TYPE7 contact) | 118 | 4.71 | 39.9 | 212 | shells 39 % + contact 20 % |

**The cliff verdict (c37):** the 0.9 cyc/s is **NOT a super-linear pathology** —
no O(N²), no per-group Python explosion (`element_groups` yields exactly one
group per type), no contact. It is **linear element-force cost × N**: per-element
cost is flat-ish across sizes (c46 12.3 → c37 14.5 µs/elem-cyc), so cyc/s ∝ 1/N
is inevitable, and the lever is per-element force cost (~8–9× the Fortran 534 k
elem·cyc/s baseline). Inside the 91.6 % element share: hexa `_post` 16.9 %, Ogden
`c_einsum` 16.6 %, **`np.linalg.eigh` on (n,3,3) 14.5 %** (Ogden principal-stretch
via LAPACK per cycle), hexa `_pre`/`_geometry` ~10 %, `_char_length` 7.8 %,
`cross3`/`det_inv33`/`norm3` ~9 %, plus a hidden **~4.6 % anim VTK ASCII**
(`np.savetxt` on 72 k-node arrays) that landed in the residual because
`write_anim_state` was not a wrapped stage — exactly what the cProfile pass
surfaces. Ogden material alone ≈ 28 % of the cycle (eigh + det + tensor einsums).

**(b) The numba lever — MEASURED** (c46; the mirrors already existed in
`accel.jit_kernels`, OFF by default):

| | numpy | numba | Δ |
|---|---:|---:|---:|
| total ms/cyc | 12.34 | 8.50 | **−31 %** |
| element-force ms/cyc | 9.91 | 6.03 | **−39 %** |

The mirrors cover the top hot spot of EVERY regime (hexa_pre/post, shell_pre/post,
t7_narrow). This is the profiler's **#1 ranked lever**: make numba the default
above a size threshold (~> 500 elems), keep NumPy for tiny/CI decks — pending a
corpus-wide parity re-run under numba (§7).

**(c) The optimizer wins — each measured + parity-proven** (behind `accel.get`;
NumPy reference untouched):

| item (builder) | regime | before | after | speedup | parity proof |
|---|---|---:|---:|---:|---|
| anim VTK writer `_write_block` (OPT-1) | c37 | 2566 ms/state | 932 ms/state | **2.75×** | md5 c385e6fa IDENTICAL |
| anim VTK writer (OPT-1) | c46 | 31.5 ms/state | 10.2 ms/state | **3.1×** | md5 3c2ec9c8 IDENTICAL |
| force scatter `scatter3` (OPT-1, numba) | c37-scale | 14.14 ms | 4.87 ms | **2.9×** | bitwise (zero + nonzero target) |
| T-file displacement guard (OPT-1) | c37 global-only | disp alloc/row | skipped | — | T01 byte-identical |
| `hexa_hgphys` LAW70 phys-hourglass (OPT-2, numba) | c46 | 1343 µs/call | 300 µs/call | **4.46×** | rtol 1e-12 + restart-chain canary |
| `law70_tab2d` table interp (OPT-2, numba) | c46 | 65.6 µs/call | 20.1 µs/call | **3.26×** | `np.array_equal` (0-ulp) |
| `law70_snorm`/`enorm` Voigt norms (OPT-2, numba) | c46 | 45.8 µs/call | 5.5 µs/call | **8.4×** | `np.array_equal` (0-ulp) |
| **OPT-2 combined (whole cycle)** | c46, numba A/B | **6.25 ms/cyc** | **4.30 ms/cyc** | **1.455×** | numpy T01 byte-identical |

OPT-1's anim/T-file wins are backend-INDEPENDENT (they help every deck; the full
c37 run saves ~30 × 1.63 s ≈ 49 s wall on anim alone). OPT-2's kernels
materialise only on the numba backend. **Deliberately NOT touched**: engine.py's
hot loop — the profiler puts true inline integration arithmetic at ~1.4 % of the
cliff (dominated by irreducible array work + reassociation-locked energy
einsums), so preallocated scratch saves < 0.1 % while risking physics on a
shared, concurrently-edited, physics-critical file (the speed-work contract
forbids it).

**(d) The speedup table — numba/NumPy on the M39 current tree** (back-to-back,
CONTENDED; ratios are the load-robust product):

| deck | NumPy | numba | numba/NumPy | M7 was |
|---|---:|---:|---:|---:|
| tensile_bar | 2.485 s | 1.617 s | 1.54× | 1.63× |
| box_beam_impact | 9.649 s | 4.92 s | **1.96×** | 1.77× ↑ |
| antenna_mast | 2.981 s | 2.653 s | 1.12× | 1.05× |
| rubber_block | 2.479 s | 1.719 s | 1.44× | 1.42× |
| notched_plate | 141.86 s | 56.89 s | **2.49×** | 2.36× ↑ |
| spot_weld | 13.70 s | 7.33 s | **1.87×** | 1.67× ↑ |
| edge_impact | 94.30 s | 51.32 s | **1.84×** | 1.47× ↑ |
| rigid_impactor | 151.59 s | 63.63 s | **2.38×** | 2.20× ↑ |
| gas_piston | 0.912 s | 2.002 s | 0.46× | 1.10× ↓* |
| c04_E1000 | 2.537 ms/cyc | 1.465 ms/cyc | 1.73× | — |
| c46_LAW70 | 12.34 ms/cyc | 4.27 ms/cyc | 2.89× | — |
| c37_T1040 (cliff) | 939.7 ms/cyc | 530.0 ms/cyc | 1.77× | — |

(`*` gas_piston: a sub-second gas/airbag deck with no numba-accelerated kernels
— JIT/dispatch overhead dominates; T01 byte-identical, not a regression.) The
isolated speed WINS: OPT-2's LAW70 kernels give **1.81×** on the c46 numba cycle
(mirrors off vs on, alternated best-of-4); OPT-1's NumPy path is **1.086×**
end-to-end on box_beam.

**(e) THE PHYSICS-REGRESSION GATE: PASS — 9/9, NumPy backend, 0 speed-attributable
T01 changes.** Every bundled example's current-tree T01 is byte-identical to its
fidelity-complete PRE-SPEED reference. The proof isolates fidelity from speed:
box_beam current == OPTIMIZER-1's saved `ref_run` (fidelity-tree, pre-speed)
byte-for-byte while the M38 tree differs (that difference is M39 SHELL FIDELITY,
not speed); the 5 shell decks that M39 fidelity changed are reproduced
byte-for-byte by a fidelity-tree reconstruction (the only 2 numpy-T01-path speed
files reverted to M38). **A naive M38-vs-current T01 diff would false-positive on
those 5 shell decks** — any future speed-regression check must isolate fidelity
the same way (recorded for M40).

**(f) The forward menu — the profiler's 6 ranked optimization items** (for the
next milestone's optimizer builders): **#1** activate the numba backend by
default above a size threshold (measured −31 % c46; addressable 40–92 % of the
cycle across regimes; risk MEDIUM — re-run the parity suite under
`PYRADIOSS_BACKEND=numba` first); **#2** replace the Ogden LAPACK `eigh` with a
closed-form symmetric-3×3 eigensolver + analytic det (Ogden ≈ 28 % of the cliff;
risk MEDIUM); **#3** numba-mirror the LAW70 physical hourglass + inline the
fastmath primitives (risk LOW-MEDIUM); **#4** binary anim output instead of ASCII
`np.savetxt` (~4.6 % of the c37 window; risk LOW); **#5** fuse the assembly
scatter (3 bincount → 1) and the inline energy einsums (risk LOW-MEDIUM); **#6**
cache the rigid-body inertia decomposition (per-cycle svd/solve; risk MEDIUM).
An UNCONTENDED re-timing on an idle box is still owed (deferred from M38 too).

### 6.4 M40 numba-default activation + the clean benchmark (`numba-default`, `parity-m40`)

M40 landed the M39 profiler's **#1 ranked lever** (§6.3f): make numba the default
above a size threshold, keep NumPy for tiny/CI decks. The change is surgical
(Edit-only) to `accel/__init__.py` (a new `auto_select_backend()` + helpers +
forced-request tracking, default flipped to `auto`), the engine backend-selection
call site MOVED to after the restart is read (so the element count is known before
the first kernel call — `engine.py`), and the CLI (a new `auto` choice).

**(a) The activation.** The compute-backend default is now **`auto`**: numba is
selected automatically when (a) it imports cleanly AND (b) the model has **≥ 32
total elements**, with a clean NumPy fallback otherwise and an explicit
engine-listing line naming the choice and reason. `PYRADIOSS_BACKEND` /
`-backend numpy|numba` still PIN either backend. Auto-numba is gated to the
explicit `/RUN` path (the M39 speed sweep's envelope); implicit runs stay NumPy
unless pinned. The Starter always runs the provisional NumPy under `auto`
(one-shot; not worth the warm-up); only the engine loop flips to numba. **13 new
tests** (`tests/test_m40_auto_backend.py`) pass. CI is extended: the fast
regression suite is PINNED to `PYRADIOSS_BACKEND=numpy` (determinism/hermeticity —
every test predates the auto default and targets the NumPy reference), plus one
dedicated 3.10-leg step that exercises the auto→numba path end-to-end.

**(b) The threshold — DERIVED, not guessed** (`_AUTO_MIN_ELEMENTS = 32`, from
`perf_m39_speed.json`, joined with per-deck sizes and the `profile_cycle.py`
regimes):

| deck | elem | numba speedup | verdict |
|---|---:|---:|---|
| gas_piston | 4 | 0.46× | **LOSS** (sole regression; JIT warm-up + dispatch swamp its 0.9 s baseline) |
| antenna_mast | 10 | 1.12× | marginal (within the sweep's contention noise) |
| tensile_bar | 40 | 1.54× | **robust win** ← smallest robust win |
| rubber_block | 64 | 1.44× | robust win |
| c04_E1000 | 99 | 1.73× | robust win |
| rigid_impactor | 118 | 2.38× | robust win |
| notched_plate | 194 | 2.49× | robust win |
| box_beam_impact | 200 | 1.96× | robust win |
| edge_impact | 256 | 1.84× | robust win |
| spot_weld | 300 | 1.87× | robust win |
| c46_LAW70 | 1000 | 1.81–2.89× | robust win |
| c37_T1040 | 65439 | 1.77× | robust win |

The ONLY regression (gas_piston, 4 elem) and ONLY marginal case (antenna_mast, 10
elem) sit far below the smallest robust win (tensile_bar, 40 elem); every larger
deck wins 1.44–2.49×. **32 sits in the (10, 40] gap** — ~8× the regression size,
> 3× the marginal, below every robust win. NODES are rejected as the metric
(gas_piston has MORE nodes (20) than antenna_mast (12) yet loses — element count
is the causal JIT-kernel workload). Env-overridable via
`PYRADIOSS_BACKEND_AUTO_MIN_ELEMENTS`.

**(c) The auto-selection matrix** (`accel.auto_select_backend`, called by the
engine after restart read):

| request / env | model / run | result |
|---|---|---|
| `PYRADIOSS_BACKEND=numpy` / `-backend numpy` | any | numpy (forced) |
| `PYRADIOSS_BACKEND=numba` / `-backend numba` | any | numba (forced; numpy+warn if numba absent) |
| auto/unset (default) | explicit, elems ≥ 32, numba ok | **numba** ("numba (auto: N elements ≥ 32)") |
| auto/unset (default) | explicit, elems < 32 | numpy ("numpy (auto: N elements < 32 …)") |
| auto/unset (default) | implicit /IMPL run | numpy (auto-numba is explicit-/RUN-only) |
| auto/unset (default) | numba not importable | numpy ("numpy (auto: numba not installed)") |

**(d) The parity contract holds** (numba vs numpy — the safety gate):

| source | scope | max deviation |
|---|---|---|
| M7 kernel-parity tests (13/13 pass) | element kernels, single-call | bitwise, ≤ 1e-12 |
| M40 grid 48-brick `auto`(numba) vs numpy | x / v / sig full run | **8.85e-19** |
| corpus sh3n246_twist / beam70_frame | T01 forced-numba vs numpy | **BYTE-IDENTICAL** (no SH3N/BEAM numba kernel) |
| corpus shell99_bending (BATOZ) / brick8 | T01 | differs at **ulp only** (has a numba kernel; quantified by M7 ≤ 1e-12) |
| corpus brick8 MOMY momentum-dust | global-sum on a conserved ~0 channel | 2.15e-17 abs — the documented reduction-reassociation exception |

Element-wise bitwise; short reductions differ at machine precision; global-sum
momentum shows ulp "dust" on conserved ~0 channels — the contract, and it holds.

**(e) THE CLEAN BENCHMARK — RELATIVE-ONLY** (`parity-m40`, `perf_m40_clean.json`).
The standing uncontended-timing side-quest could NOT be discharged: the user's
12-process `engine_win64_impi` MPI job ran the WHOLE M40 session (same PIDs 2+
days), so no idle window opened. The backend set was run BACK-TO-BACK per deck so
the RATIO cancels steady contention; **absolute seconds are UPPER BOUNDS** (lower
bounds on speedup):

| deck | numpy s | numba s | auto s | numba× | auto picks |
|---|---:|---:|---:|---:|---|
| tensile_bar | 1.499 | 0.884 | 0.867 | 1.70× | numba |
| box_beam_impact | 6.003 | 2.858 | 3.238 | **2.10×** | numba |
| antenna_mast | 1.523 | 1.532 | 1.506 | 0.99× | **numpy** (10 elem < 32) |
| rubber_block | 1.569 | 0.952 | 0.911 | 1.65× | numba |
| notched_plate | 87.70 | 36.75 | 37.30 | **2.39×** | numba |
| spot_weld | 9.35 | 5.34 | 4.59 | 1.75× | numba |

**Median numba 1.751×**; `auto` tracks numba for ≥ 32-elem decks and correctly
picks NumPy for antenna_mast. The T01 md5 is byte-identical across backends only
where `auto` selects numpy (antenna_mast); the numba-kernel decks differ at the
documented reassociation ulp (NOT a physics regression — quantified by the (d)
contract). **NOT clean absolute numbers** — re-run `bench_m40.py` +
`build_clean_m40.py` during a genuinely idle window for absolute timing.
**TOOLING NOTE for future milestones**: `tasklist /FI "IMAGENAME eq
engine_win64_impi.exe"` returns 0 on this box despite 12 live ranks — the M39
speed harness used that filter and therefore UNDER-counted contention; the M40
records use plain `tasklist | grep -c engine_win64_impi`.

### 6.5 M41 — THE CLEAN ABSOLUTE BENCHMARK (a side-quest owed since M38, finally discharged) + the new-kernel cost

**(a) The standing side-quest is DONE.** M38, M39 and M40 all owed an
uncontended absolute benchmark and all three failed to deliver one for the same
reason: the user's 12-process `engine_win64_impi` MPI job was live (in M40, the
same PIDs for 2+ days), so every number was a contended upper bound and only
back-to-back RATIOS were trustworthy. **This session the job was NOT running** —
sampled repeatedly with the reliable plain-`tasklist` regex, `impi=0`, CPU 0 %.
The idle window was taken FIRST and the benchmark run in isolation before any
other work. `perf_m41_clean.json` therefore carries **`clean: true` and
`n_contended: 0` — 0 of 9 rows contended**. These are the project's first CLEAN
ABSOLUTE engine-elapsed seconds. Method unchanged: numpy / numba / auto
back-to-back per deck, JIT cache warmed once up front, on a 13th Gen Intel Core
i9-13900H.

| deck | cycles | numpy s | numba s | auto s | numba× | `auto` selects |
|---|---:|---:|---:|---:|---:|---|
| notched_plate | 16,020 | 48.651 | 18.936 | 19.277 | **2.569×** | numba |
| rigid_impactor | 35,689 | 44.826 | 19.704 | 19.800 | **2.275×** | numba |
| box_beam_impact | 3,465 | 3.199 | 1.575 | 1.863 | **2.031×** | numba |
| edge_impact | 22,814 | 29.557 | 16.454 | 16.193 | 1.796× | numba |
| tensile_bar | 1,847 | 0.850 | 0.477 | 0.587 | 1.782× | numba |
| spot_weld | 3,454 | 4.349 | 2.440 | 2.339 | 1.782× | numba |
| rubber_block | 1,312 | 0.847 | 0.540 | 0.554 | 1.569× | numba |
| antenna_mast | 4,961 | 0.852 | 0.918 | 1.212 | 0.928× | **numpy** (10 elem < 32) |
| gas_piston | 1,320 | 0.320 | 0.525 | 0.277 | **0.610×** | **numpy** (4 elem < 32) |

**Median numba 1.782×**, range 0.61–2.57×. M40's contended ratio-only median was
1.751×, so the clean figure lands within 2 % of it — encouraging for the
back-to-back method M38–M40 fell back on, but **not a controlled validation of
it**: the two sets are not the same decks (M41 adds edge_impact, rigid_impactor
and gas_piston to M40's six), so this is a coincidence of medians across
overlapping-but-different populations, not a paired comparison. Read it as
reassurance, not proof.

**(b) The M40 `auto` rule is VALIDATED on clean data.** The ≥ 32-element
threshold was DERIVED in M40 from `perf_m39_speed.json` rather than guessed
(§6.4 (b)); the clean set now tests it. numba loses on exactly two decks —
gas_piston (4 elements, 0.610×) and antenna_mast (10 elements, 0.928×) — and
`auto` picks NumPy on exactly those two. **Every deck where numba wins, `auto`
picks numba.** Zero misclassifications. Verified directly for this report by
running both sub-threshold decks and reading the engine listing line back:
`COMPUTE BACKEND … : numpy (auto: 4 elements < 32 — JIT warm-up dominates)` and
`… (auto: 10 elements < 32 …)`. **Correction to a builder-side table**: the
`auto_matches` field in `perf_m41_clean.json` is a timing-PROXIMITY heuristic,
not the recorded backend choice, and it reads `numba` for antenna_mast because
that deck's 1.212 s auto row sits nearer numba's 0.918 s than numpy's 0.852 s in
absolute terms. Read as a selection record it would be wrong; the listing line
above is the ground truth and the table's last column reflects it.

**(c) THE NEW KERNELS ARE SLOW, AND ONE OF THEM IS SLOWER UNDER NUMBA.** This is
the milestone's real performance finding and it is a cost, not a win. Measured
SOLO on c04 (a full 97,622-cycle QBAT run):

| backend | c04 wall | throughput |
|---|---:|---:|
| numpy | 330 s | 296 cyc/s |
| numba | 395 s | — |

**numba is ~20 % SLOWER than numpy on QBAT** because `shell_qbat.py` has **no JIT
kernels at all** — the numba path adds dispatch overhead with nothing to compile.
QBAT is also roughly **3× the per-cycle cost of BT** (the `parity-m41` builder's
measurement), which is the expected price of 2×2 in-plane Gauss × NIP
through-thickness integration versus a 1-point element — but it is now a real
price, paid by whichever of the 160 corpus decks the §4.10 dispatch routes
actually reach the engine. Two consequences are logged as follow-ups (§7): a
`jit_kernels` pass for the QBAT/QEPH kernels, and a check on whether the `auto`
≥ 32-element rule should EXCLUDE qbat/qeph groups until those kernels exist —
because for those groups the threshold currently selects the slower path.

**(d) Timing hygiene — what in this milestone is clean and what is not.**
`perf_m41_clean.json` is the authoritative M41 timing source and is CLEAN
ABSOLUTE. `perf_m41.json` is NOT: the 29-case parity sweep ran under an 8-worker
pool, so those seconds are concurrent UPPER BOUNDS (classes and rel-RMS are
deterministic and unaffected). Fortran timings are CARRIED, never re-measured
(c04 reference: 17.38 s / 97,623 cycles). **One deck was excluded from the clean
set and it is named**: `c46_LAW70` burned > 5.5 minutes of CPU on the numpy
backend alone, and three back-to-back backends would have held the machine 15+
minutes and risked losing the rare uncontended window for the rest of the
milestone. The M40 clean set contains no c46 row either, so no trend row was
lost — but **its backend ratio is unmeasured, not regressed**
(`decks_omitted` in the JSON). Finally, since these decks now run to completion
for the first time, clean per-deck RD-E-1000 parity timings have become
interesting and are not yet measured: that needs a serial re-run on an idle box.

## 7. Known issues & backlog (updated for M41)

Three M40 backlog items are DONE this milestone. **Item 1** (the RD-E-1000
full-run MATCH, re-scoped to force physics) is delivered for the two families it
named — QBAT and QEPH both MATCH at 100 % coverage (§3.6) — and settled by
experiment for the third (BT). **Item 2** (the Sf_0.1 regression,
`task_29ec1751`) is RESOLVED at root with the guard limits untouched (§3.6 (d)).
**Item 5** (the clean uncontended benchmark, owed since M38) is DISCHARGED
(§6.5). Item 9's "BT floor is 7 % conservative" is largely closed too — the
`cinmas.F` FAC=9 lumping the item said was "deliberately not touched" WAS touched,
correctly, moving the floor 1.86540e-2 → 1.95667e-2 against Fortran's 2.004e-2.
The post-M41 list, in measured-value order:

1. **THE BT FAMILY STAYS DEVIATION — and the honest framing is "unreachable by
   comparison", not "unfinished"** (§3.6 (c); c40–c45 at rel-RMS 0.138–0.257,
   88–96 % of `/RUN`, worst channel HE or MOMZ). The `cdefo3.F` fix halved the
   deviation and unblocked the Sf_0.1 variants, but the reference itself diverges
   in its own final window (Fortran HE = 76 % of its own IE at TSTOP, peak printed
   ENERGY ERROR −40.7 %), so a sub-0.05 score would require scoring against a
   diverged signal. **Do NOT "fix" this by weakening the −15 %/−30 % guards** —
   upstream's own default is `DEMXS=EP30` (i.e. no energy stop at all), which is a
   reporting choice, not a physics endorsement. If the family is pursued, the two
   concrete targets are the **unported `cdefo3` branches for c43 (BT type3, the
   node-1-relative velocity form — it currently takes NO rot2 correction and gains
   only via the FAC=9 dt) and c45 (BT type4, the Z2 warp correction)**.
2. **DKT18 IS NOW THE ONLY UNPORTED SHELL FORMULATION IN THE FAMILY** (§3.6 (e);
   c06 0.2813, c07 0.2581, worst channel MOMX on both; c00 twisted beam 0.4556).
   It received no kernel in M41; these are its FIRST measured values (M39/M40
   carry no baseline for c00/c06/c07 — the full-family re-run is itself an M41
   deliverable), so they are a starting point rather than a stalled number. With
   QBAT and QEPH landed and BT settled, DKT18 is the obvious next
   element-technology target and the cleanest remaining RD-E-1000 lead.
3. **QBAT/QEPH HAVE NO JIT KERNELS, AND `auto` CURRENTLY PICKS THE SLOWER PATH FOR
   THEM** (§6.5 (c)). c04 full run: **numba 395 s vs numpy 330 s** — the numba
   path adds dispatch overhead with nothing compiled. QBAT is also ~3× BT per
   cycle (296 cyc/s). Two follow-ups: a `jit_kernels` pass for both new kernels,
   and a decision on whether the `auto` ≥ 32-element rule should EXCLUDE
   qbat/qeph groups until those kernels exist. This is a genuine cost the
   milestone incurred to buy the MATCH, not a pre-existing gap.
4. **c05_E1000_Bending_DKT18_Sf_0.1 IS A GENUINE TIMEOUT** (§3.6 (e)) — 681,800
   cycles and 1799.38 s of an 1800 s budget at ~55 % of `/RUN`. Sf_0.1 scales dt
   ~10×, so the deck needs ~1.2 M cycles. It wants a ~4000 s budget or a
   cycle-capped comparison; it currently has NO verdict, which is different from
   having a bad one.
5. **DOCUMENTED DELIBERATE CUTS IN THE TWO NEW KERNELS** — all recorded in the
   module docstrings, all with a stated reason, none of them silent. **QBAT**: the
   `nip=1` CBAFORI1/CBAVISNP1 branch (the port runs the layered path with SHF=0
   and the starter emits a warning; no official deck runs QBAT at N=1),
   `Idrill>0` drilling stiffness (the deck reader does not surface Idrill),
   `Ithick=1` thickness update (the c04 family is Ithick=0), and
   `tangent()`/`kgeo()` — implicit assembly REFUSES `shells_qbat` groups loudly
   and that refusal is test-pinned. **QEPH**: `ZCFAC=1` in the plastic relaxation
   (Fortran's `(1−ETAN/E)` floor needs a per-cycle material tangent the port's
   material layer does not surface — **exact for elastic laws, including c08/c09**),
   `NPT=0` "global integration" resolved to 3 Gauss stations (only `czfintn.F`'s
   COEF1 16-vs-25 plastic-relaxation weight differs, invisible on the elastic
   c08/c09), `Idrill=1`, `ISMSTR=1/11` frozen small strain, XFEM/thermal, and the
   implicit tangent (same loud gate). **The orthotropic QEPH stabilization
   (`czfintn_or` HM/HF path) is NOT ported — LAW19 shells run the isotropic
   `czfintn` moduli**, flagged in `checks.py`. LAW27/LAW36 shells are gated ALLOWED
   for QBAT through the shared layer plumbing but only LAW1/LAW2(+JOHNSON) are
   exercised by dedicated tests; the plumbing-reuse contract is pinned by a
   layer-for-layer LAW2 parity test against BT4.
6. **OWNERSHIP RECONCILIATION OWED IN `shell_bt4.py`** — its `_CONDENSED_FACDT` /
   Ishell 12/22/24 dt-claim branches (added in M40, when BT was the only shell
   kernel) are now SHADOWED for every deck the new starter dispatch routes to the
   `shells_qbat` / `shells_qeph` groups, because those kernels compute their own
   condensed length. Harmless dead-ish code today, but it is duplicated BATOZ-family
   claim logic in a file that no longer owns it, and it should be reconciled before
   it drifts.
7. **AN ENVIRONMENT REGRESSION HIT MID-MILESTONE AND WAS FIXED AT ROOT — worth
   recording because it was not the port's fault and could recur.** A
   `vortex-radioss` install downgraded the shared Python 3.10 environment's NumPy
   to **1.26.4**, which has `np.trapz` but not `np.trapezoid` (added in NumPy 2.0,
   NEP 52). Every `np.trapezoid` caller broke at RUNTIME (not import time):
   `test_m9_geomnl::test_elastica_cantilever` plus exactly 6 implicit modules
   (`random_response`, `multiaxial_fatigue`, `nongaussian_fatigue`,
   `joint_nongaussian_fatigue`, `joint_evolutionary_fatigue`,
   `wigner_ville_fatigue` — all 7 call sites verified converted for this report).
   The fix landed as **`pyradioss/common/npcompat.py`** —
   a rename-only shim resolved once at import against whichever NumPy is
   installed — with `tests/test_numpy_compat.py` (5 tests) pinning it. This is the
   right fix given that `pyproject.toml` promises `numpy>=1.22` and the
   post-processing stack (`vortex_radioss` → `lasso-python`) hard-pins
   `numpy>=1.23.3,<2.0.0`, i.e. any environment that can read a Radioss animation
   file is NECESSARILY a NumPy 1.x environment. Follow-up: audit for any other
   NEP-52 spellings that have not yet been exercised.
8. **NUMBA SPOT COVERAGE NARROWED THIS MILESTONE** (§3.6 (g)). The M41 spot set is
   3 decks (QBAT/QEPH/BT), all PASS, with c04 and c08 T01 **byte-identical** to
   numpy. But the LAW36-solid control (c20) from the M40 spot set was NOT re-run,
   so **the solid-path backend contract is CARRIED from M40, not re-proven on this
   tree**. Cheap to close on the next idle window.
9. **CLEAN PER-DECK PARITY TIMINGS AND THE c46 BACKEND RATIO ARE STILL
   UNMEASURED** (§6.5 (d)). `perf_m41.json`'s wall clocks are 8-worker concurrent
   upper bounds; now that the RD-E-1000 decks run to completion for the first time
   their per-deck cost is genuinely interesting and wants a serial idle-box re-run.
   `c46_LAW70` was excluded from the clean set (> 5.5 min on numpy alone would have
   risked the uncontended window) — **unmeasured, not regressed**.
10. **A PRE-EXISTING GUARD GAP, NOTED BUT NOT CHANGED** (out of scope for the
    startup-floor task, flagged so a future hardening pass can pick it up): the
    NAN/INF divergence backstop tests only `e['KE']`, not IE or HE. A pathological
    divergence where IE goes NaN while KE stays finite would slip both percentage
    guards (NaN comparisons are False) and the KE-only finiteness check. Marginal
    in practice — KE diverges with or before IE — and deliberately NOT touched
    while fixing the startup conditioning, since widening a guard's reach is a
    separate change with its own risk.
11. **Carried from M40, unchanged**: the **c20 MOMZ momentum residual** (0.3156 —
    LAW36 material fidelity is perfect at IE 1.1e-06, but the Isolid=24 HEPH
    free-node momentum channel deviates; c19/Isolid18 with the identical material
    path is a MATCH); the **LAW2 solids volumetric defect** (`task_6c08e3b9`, c13
    full-window IE 0.2090); the **LAW36 rate-family clamp-vs-extrapolation
    deviation** (`task_7b31ad5f`, no V0700 case exercises it); **TYPE32
    pretensioner physics** (`ruser32.F`, assessed tractable, left
    `InactiveProperty`, c52 stays SKIPS); the **port's explicit dt ~2× smaller than
    Fortran on the V0700 solids** (current-density vs constant SSP — costs runtime,
    not accuracy); the **coverage frontier** (§4.10 ranked gaps byte-identical:
    INTER/TYPE24 18/5, MONVOL/AIRBAG1 16/2, INTER/LAGMUL 14/2, ALE/BCS 12/7,
    SHEL16 12, QUAD 10/5, EOS/LINEAR 6, AMS 5/5; `RBODY has no mass` still 3
    decks; the /ADMAS node-group wall 10 decks); `coverage_tables.md` still
    reflects M39 and can be regenerated from `coverage_results_m41.json`;
    `accel.backend_name()` remains referenced only by a docstring (confirmed
    unused again this milestone, deliberately LEFT IN PLACE rather than pruned);
    the remaining documented `/PROP` + LAW2 cuts, material physics for the
    parsed-but-inactive laws, the contact/hourglass differential study, and
    gas_piston positive-P0 for full 9/9 comparability.

### M40 backlog (superseded — items 1 (for QBAT/QEPH), 2, 5 and most of 9 are resolved by M41; the rest are carried into the M41 list above)

Several M39 backlog items are DONE this milestone: the SPEED numba-default (M39
item 4, the profiler's #1 lever) is LANDED (§6.4); **M39-BUG-SPRPRE** (item 2) is
RESOLVED (§4.9, proven Fortran-legal on the real starter); the **degenerate-brick
stall** (item 3) is RESOLVED — it was the M39 §3.4 `/PROP/SOLID` reader bug, not
the geometry (§3.5, `residuals-pack`); and the two sharp M39 §3.4 leads are both
driven to root (§3.5): the RD-E-1000 dt floor is now **Fortran-exact**, and the
LAW36 **~19 % material gap is CLOSED to 1e-6** — though the RD-E-1000 full-run
MATCH is re-aimed at M41 as a FORCE-physics gap (the dt was never the blocker).
The post-M40 list, in measured-value order:

1. **THE RD-E-1000 FULL-RUN MATCH — re-scoped to M41 as force-physics, NOT dt**
   (§3.5). The M40 dt claim is DONE (floors Fortran-exact: c04 1.64410e-2 =
   0.01644), the deviation halved (0.55 → 0.23), but the direct dt-floor sweep
   proves the abort is floor-INDEPENDENT. The BT-vs-BT c41 isolation localizes it:
   port IE tracks Fortran to 4 digits until t≈900 ms, then a transverse-w
   hourglass mode is genuinely EXCITED (HE 7 orders over Fortran). **M41 lead: the
   BT/BATOZ/QEPH kernel's rate kinematics / corotational treatment under large
   accumulated per-step rotation.** Separately, an element-technology port
   (QBAT/QEPH FORCE physics — the port runs its BT kernel for Ishell 12/24 decks,
   storing bending energy partly as hourglass where fully-integrated QBAT keeps
   HE ≡ 0) is the path to MATCH on the c02/c04/c08 variants.
2. **THE Sf_0.1 REGRESSION** (§3.5; `task_29ec1751`, spawned; 3 decks
   c40/c42/c44). Under the STIFR change the BT_type1/3/4 Sf_0.1 variants abort at
   **cycle 100** on the −15 % energy guard against **~1e-8 J** absolute energies
   (IE 3.75e-08) — an ill-conditioned %-based check at slow-loading startup; M39
   ran them to ~256 k cycles, Fortran runs them to NORMAL. **The guard must NOT be
   weakened** — condition the denominator / startup like upstream. The one honest
   regression of the milestone.
3. ~~**c20 MOMZ momentum residual** (A 3.5). LAW36 material fidelity is PERFECT (c20 IE rel_rms 1.097e-06, NORMAL) but a MOMZ momentum channel deviates 0.316, keeping the overall class DEVIATION — an element-side free-node dynamic of the Isolid=24 HEPH formulation~~ (Fixed in M64 via the exact HEPH formulation implementation).

4. **LAW2 SOLIDS share the volumetric defect the LAW36 pressure fix removed**
   (§3.5; `task_6c08e3b9`, spawned with the exact recipe). `m2law.F` uses
   PNEW=BULK·AMU; the port integrates K·tr(deps) incrementally — c13 full-window
   IE rel_rms 0.2090 (port 7.1× high at t=27.5). Out of the M40 LAW36-file scope;
   the same `needs_env` rho pattern applies. **NOTE for this report's own §3.4:
   its "LAW2 brick/tetra IE rel_rms 0.023" is a partial-window (~t≤2.3) figure —
   the full-window value is 0.2090** (the M40 window scan supplies the correction).
5. **The clean UNCONTENDED benchmark is STILL OWED** (§6.4; deferred from
   M38/M39 too). The user's 12-process MPI job ran the whole M40 session, so
   `perf_m40_clean.json` carries only the back-to-back RATIO (numba median
   1.751×), not trustworthy absolute numbers. Re-run `bench_m40.py` +
   `build_clean_m40.py` on a genuinely idle box.
6. **Port explicit dt runs ~2× smaller than Fortran on the V0700 solids** (§3.5):
   3.8e-4 vs 7.6e-4 — the port kernel sound speed uses CURRENT density vs
   Fortran's constant SSP=√((K+4G/3)/rho0) from `sigeps36`. Costs runtime, not
   accuracy; a SPEED-track candidate (with c12/c18/c18-DEGE throughput-limited to
   SKIPPED-SLOW at ~26 cyc/s under contention — stable, not unstable).
7. **numba-default parity tails** (§6.4; `numba-default` OPEN items): the corpus
   spot-parity tail (tetra120_v0700, brick800_pressure) was still running under
   contention — re-run `corpus_parity.py` idle to fill both rows; the
   bundled-examples both-backends byte-compare is STAGED (`rerun_picks.json`) not
   executed (a second concurrent numba process destabilized runs on the contended
   box); a whole-suite auto run UNDER numba could not complete cleanly (3-way
   contention exit-127, no test failures observed) — worth an idle-box
   confirmation, though CI is unaffected by the numpy pin. `backend_name()` is now
   referenced only by a docstring (harmless; prune later if desired).
8. **The LAW36 rate-family clamp-vs-extrapolation deviation** (§3.5;
   `task_7b31ad5f`): upstream linearly EXTRAPOLATES outside the strain-rate table
   (`sigeps36.F` RFAC unclamped), the port clamps; ISMOOTH=2 log-interp and the
   VP=1 plastic-rate branch remain unported (warned). No V0700 case exercises it
   (all NRATE=1).
9. **The BT-family rigid-body dt floor is 7 % conservative vs Fortran** (§3.5,
   c41 1.8654e-2 vs 2.006e-2 — safe side, documented). Mirroring it exactly needs
   `chvis3`'s STI/STIR claim formulas AND the `cinmas.F` FAC=9 (A/9) BT inertia
   lumping — the latter changes `model.inertia` for every BT shell deck (starter
   mass physics, deliberately not touched). Related documented cuts: rotational
   spring STIFR claims (torsional /PROP/TYPE8/13) are not ported (a spring-torsion
   /RBODY deck would floor coarser than Fortran); the sh3n (DKT/sh3n, RD-E-1000
   c05-07) condensed length is not yet gated in (same pattern as shell_bt4);
   SH_ORTH/SH_FABR (TYPE9/16) do not parse ishell/dn, so orthotropic QEPH decks
   keep the BT claim (pre-existing).
10. **TYPE32 (/PROP/SPR_PRE) pretensioner element physics — DEFERRED**
    (`residuals-pack` ITEM 4): `ruser32.F` is a 1-DOF sensor-gated axial
    pretensioner (4 ITYP laws + Ilock + UVAR); assessed tractable but a full
    active-spring port needing per-ITYP T01 channel validation against RD-V-0031.
    Left `InactiveProperty` (reader carries all the data); c52 stays SKIPS.
11. **The next verdict-conversion frontier** (§4.9 ranked gaps, unchanged
    M39 → M40): INTER/TYPE24 (18/5 sole), MONVOL/AIRBAG1 16, INTER/LAGMUL 14,
    ALE/BCS (12/7), SHEL16 12, QUAD (10/5), EOS/LINEAR 6, AMS 5. `RBODY has no
    mass` still 3 decks (UNCHANGED — the STIFR builder is engine-side, orthogonal
    to the starter mass-accumulation check); the deeper /ADMAS node-group wall
    (10 decks) carried. `coverage_tables.md` (shared repo file) still reflects
    M39 — a future editor may regenerate it from `coverage_results_m40.json`.
12. **Carried from M39** (unchanged): the shell fix's narrow reach beyond
    RD-E-1000 (now understood as the BT force-kinematics M41 lead, item 1); the
    Isolid24/HEPH assumed-strain brick generalized beyond LAW70; the /PROP + LAW2
    documented cuts; material physics for the parsed-but-inactive laws (LAW6
    HYD_VISC 30 blocks, LAW51 22, …); the contact/hourglass differential study;
    gas_piston positive-P0 for full 9/9 comparability. c42 BT_type3's full run is
    infeasible until the speed work (Fortran needs 13.7 M cycles ≈ 7 h port wall).

### M39 backlog (superseded — items 2/3/4 and the two §3.4 leads (the RD-E-1000 dt floor and the LAW36 ~19 % gap) are resolved/advanced by M40; the rest are carried into the M40 list above)

Several M38 backlog items are DONE this milestone: the `/SKEW`//`/FRAME` cluster
(M38 item 5, the #1 named gap) is 100 % CLOSED (§4.8, 47→0 cases_blocking); the
c50 fabric NaN channel (M38 item 1) is FIXED (§3.3, → 0.609); three of the four
M38 quick-fix bugs (item 2) are RESOLVED (M38-NEW-2 VOID null-density, M38-NEW-1
SPR_PRE TYPE32 mass, M38-NEW-4 RBODY node-overlap — §4.8 error-class delta); and
the M36/M37 shell-hourglass fidelity finding (§2.3/§3.1) is directly addressed
for active-hourglass decks (§3.3, box_beam HE 6×). The post-M39 list, in
measured-value order:

1. **TWO RED tests in the shared tree — RECONCILED by the integration verifier**
   (both from unreconciled concurrent edits, both were verified RED at report
   time, NEITHER a NumPy-physics regression): (a)
   `tests/test_m7_backends.py::test_shell_pre_post_parity` — the `shell-fidelity`
   builder changed `shell_bt4._post`'s signature to the `chvis3.F` form (added
   `hqm/hqb/hqr`, `dt`) and updated `jit_kernels.shell_post`, but did not update
   the test's direct call (`_post() missing hqr, dt`); the parity call now feeds
   `k_m,k_w,hqm,hqb,hqr,dt` to BOTH backends, so the numpy==numba bitwise check is
   preserved. (b)
   `tests/test_element_kernels.py::test_degenerated_brick_penta_run_as_collapsed_hexa`
   (renamed from `..._penta_rejected`) — asserted the OLD reject-penta behavior
   the `small-bugs` pack intentionally changed to run-as-collapsed-hexa (the deck
   now prints `1 DEGENERATED /BRICK ELEMENT(S) RUN AS COLLAPSED HEXA`); it now
   asserts the accept-and-run behavior, the collapse physics already covered in
   `test_m39_smallbugs.py`. The 72 new `tests/test_m39_*.py` all PASS.
2. **M39-BUG-SPRPRE — the one new error class** (§4.8; 2 decks: RD-HWX-T-1010
   cantilever_completed + its DYREL variant; NOT a regression, both ERROR in
   M38). The M39 /PROP/SPR_PRE change reads the card MASS field; these decks'
   SPR_PRE/2 card has a BLANK mass, so the `mass > 0` check fires as their sole
   ERROR-level blocker. Verify whether the real Fortran starter tolerates /
   derives a blank pretensioner mass — if it defaults it, defaulting the blank
   mass converts 2 more decks ERROR→SKIPS. No `*.fortran_ref` backup exists for
   these tutorial decks and the MPI job contended the machine, so no Fortran run
   was spun up this round.
3. **The degenerate-brick unlock is PARTIAL** (§3.3): c12/c18 HEXA_DEGE — the
   starter now ACCEPTS the collapsed `/BRICK` (was PYRADIOSS-FAIL) but the engine
   terminates at ~12 cycles / < 0.1 % of `/RUN` (NO-CHANNELS); the element runs
   but the simulation does not progress. Investigate why the run stalls
   immediately. c52 spring TYPE32 is likewise partial (starter parses, but
   /PROP/SPR_PRE element physics is unimplemented → engine-fail).
4. **SPEED — activate the numba backend by default** (the profiler's #1 lever,
   §6.3f; measured −31 % on c46, addressable 40–92 % of the cycle). Requires a
   corpus-wide parity re-run under `PYRADIOSS_BACKEND=numba` before defaulting it
   on, then the ranked #2–#6 (Ogden closed-form eigensolver, LAW70 hourglass
   mirror, binary anim, scatter/einsum fusion, rbody-inertia cache). An
   **UNCONTENDED timing pass on an idle box is still owed** (deferred from M38 —
   every M39 wall clock carried the impi=12/12 flag).
5. **The c31 V0240 TRIA speed regression** (§3.3): DEVIATION 0.44 → SKIPPED-SLOW,
   crossed 900 s (the fix's added per-cycle viscous-hourglass work + contention
   on 1612 elems) — a candidate for the 1800 s budget in a cleaner uncontended
   run; and the LAW70 compression trio (c46/c47/c49) remains SKIPPED-SLOW
   (STABLE, HE ≈ 0, but port-throughput-limited — the §6.3 speed work is the
   fix, not stability).
6. **Complete the M39 parity coverage** (35/65 official re-run, §8): the un-run
   30 are the RD-E-1000 remainder (inert, hourglass-off), the T1000 family
   (c33/c34/c35/c36/c38 — c33 now runs the engine, an apparent unlock), the slow
   timeouts (c21/c22/c24/c29/c37), LAW70 c47/c49, and the FRAME/MOV tensiles
   c58-c64 (the skew instance was evicted by a background-task limit before
   reaching them). Re-run `build_perf_m39.py` on an idle box to fold them in.
7. **The shell fix's NARROW reach — the next shell-fidelity target** (§3.3): the
   official RD-E-1000 Bending family runs hourglass-OFF / non-hourglass
   formulations, so it is byte-identical M38→M39; its ~0.55 residual is a
   SEPARATE bending / kinematic gap the hourglass fix does not touch. That, not
   hourglass, is now the dominant open shell deviation.
8. **The next verdict-conversion frontier** (§4.8 ranked gaps): INTER/TYPE24
   (18 blocking / 5 sole), MONVOL/AIRBAG1 16, INTER/LAGMUL 14, ALE/BCS (12/7),
   SHEL16 12, QUAD (10/5). INTER/TYPE7 Iform/Igap (31 combined) is the largest
   ERROR-message class but a contact-formulation FEATURE gap, not a quick bug;
   "model has no elements" (20) is downstream of the unported SHEL16/QUAD/
   degenerate-brick element families.
9. **M38-NEW-3 `RBODY has no mass` still UNRESOLVED** (3 decks, UNCHANGED from
   M38 — §4.8): DIF24416 ×2 (Gears Inter16/17) + I16S16FM (Cam fine_mesh); the
   `small-bugs` pack fixed only the node-overlap check, not mass accumulation.
   These decks also carry SHEL16/BRIC20 hard skips, so they would remain ERROR
   even if mass-accumulation were fixed. Also carried: the deeper /ADMAS wall
   (`/ADMAS node group not defined`, 10 decks).
10. **Carried from M38** (unchanged): the Isolid24/HEPH assumed-strain brick
    generalized beyond LAW70; the /PROP + LAW2 documented cuts (TYPE8/13 springs
    linear-core-only, SH_ORTH IREP=0, kinematic hardening); material physics for
    the parsed-but-inactive laws (LAW6 HYD_VISC 30 blocks, LAW51 22, …); the
    contact/hourglass differential study on the DEVIATION examples; gas_piston
    positive-P0 for full 9/9 comparability.

### M38 backlog (superseded — items 5 (SKEW/FRAME), the c50 NaN, and three of the four M38-NEW bugs are resolved by M39; the rest are carried into the M39 list above)

1. **~~The official-parity + timing re-run is still owed~~ DONE** — the
   resumed driver completed 52/52 (§3.2: both-engine 24 → 28, port-fails
   18 → 9, c26 + c48 full-coverage MATCHes; `parity_m38.json` /
   `perf_m38.json`). Still owed for M39: an UNCONTENDED timing pass (every
   M38 port wall clock carried the impi=12/12 flag), the LAW70
   compression-trio full runs past the 900 s budget, and the c50 fabric
   NaN-channel fix.
2. **Four M38 quick-fix bugs** (all surfaced by the §4.7 sweep, all on decks
   already ERROR in M37 — 0 regressions):
   - **M38-NEW-2** extend the MAT null-density exemption to LAW0/VOID (massless
     by design; Fortran accepts RHO0=0) — clears BAT_CIR/BAT_SQR and the
     prop-pack's OPEN item (VOID on /BEAM|/TRUSS parts tripping the pre-existing
     `_ALLOWED_LAWS` material-family gap in `checks.py`);
   - **M38-NEW-1** the /PROP/SPR_PRE (TYPE32) SPRING INIT mass check misapplies
     the TYPE4 mass requirement (RD-V-0031);
   - **M38-NEW-3** `RBODY has no mass` on the Gears Inter16/17 and Cam fine_mesh
     decks — check slave-element mass accumulation in the rbody initializer;
   - **M38-NEW-4** `RBODY node already belongs to another rigid body` on BIKERC
     — the port may be stricter than the Fortran starter's priority resolution.
3. **Complete the LAW70 c46/c47/c49 full runs** — the densification instability
   is definitively eliminated (EN≡0, max HE~2, verified into ~95 % peak crush)
   but the long Python runs (108311 Fortran cycles) were reaped by the session
   before a clean 4/4 t=0.2 NORMAL; completion is physically assured and just
   needs uninterrupted wall-clock (§3.2, §6.2).
4. **The deeper /ADMAS wall**: resolving M37-BUG-1 (22 decks) unmasked
   `/ADMAS: node group not defined` (2 → 10 decks, §4.7) — the node-group
   cross-ref resolution for /ADMAS is the next /ADMAS blocker.
5. **Top M39 physics gaps** (genuine unported families, §4.7): the skew/frame
   reference-system cluster (SKEW/FIX 16, SKEW/MOV 10, FRAME/FIX 14, FRAME/MOV
   7 = 47 blocking, high sole-blocker counts) is the highest-leverage target;
   then the contact interfaces (INTER/TYPE24 18, INTER/LAGMUL 14, INTER/TYPE18
   7), MONVOL/AIRBAG1 16, and the element families SHEL16 12 / QUAD 10 /
   degenerate bricks 11 (which would convert most of the 20 "model has no
   elements" decks).
6. **The Isolid24/HEPH assumed-strain brick, generalized** — the M38 LAW70
   hourglass stiffness is gated to LAW70 hexa bricks only (the failing decks);
   a general Isolid=24 → physical-hourglass mapping for all solid laws is the
   longer-term "until HEPH lands" item. (LAW70 on the 4-node tetra needs no fix
   — the constant-strain element has no hourglass modes.)
7. **/PROP and LAW2 documented cuts** (parsed, physics cut — warned + docstring):
   TYPE8/TYPE13 springs are the linear K/C 6-DOF core only (force functions,
   hardening, rupture, rate smoothing, sensor activation, `skew_ID` → global
   frame, the TYPE13 co-rotational beam-frame update all cut; implicit-spring
   path stays TYPE4-only); SH_ORTH is IREP=0 with the Ishell formulation flag
   read-and-ignored (port uses BT4/C0), the stored σ is in the fiber frame, and
   per-ply composite layup runs as a single orthotropic layer; LAW2/LAW36
   kinematic hardening (Chard/FISOKIN back-stress) deferred (now warned when
   Chard>0).
8. **Shell-family full-run deviations** (0.42–0.64 max rel RMS on the RD-E-1000
   Bending family, §3.1) — consistent with the M36 box-beam shell-hourglass
   fidelity finding; the next physics-fidelity target once the parity re-run
   quantifies it corpus-wide. Carried.
9. **Material physics for the parsed-but-inactive laws** (carried from M37):
   every /MAT parses but only ~15 laws carry physics; highest corpus pull among
   the inactive — LAW6 HYD_VISC (30 blocks), LAW51 (22), LAW151/MULTIFLUID
   (7+3), LAW11 BOUND (7), LAW37 BIPHAS (6). Each a one-module registry job.
10. **Carried from M37**: the contact/hourglass differential study on the five
    DEVIATION bundled examples (§2.2); gas_piston positive-P0 emission for full
    9/9 comparability; hybrid single-file pre-/BEGIN engine blocks; the
    reader/writer real-layout follow-ups (§4.5 item 5).

### M37 backlog (superseded — items 1/6 and the V0700/c26 items above are resolved by M38; the rest are carried into the M38 list)

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

M41-specific limitations first; the M40/M39/M38/M37/M36 caveats below are carried
and still apply to the sections they describe.

- **The BT headline is a SETTLED non-MATCH, and "settled" is a claim about the
  reference as well as the port.** §3.6 (c) argues that MATCH < 0.05 is
  unreachable for the BT family because the Fortran reference itself diverges in
  its final window. That argument rests on the `bt-rotation-forensics` builder's
  instrumented Fortran runs (HE = 76 % of IE at t=1184, peak printed ENERGY ERROR
  −40.7 %) and on the `freform.F` `DEMXS=EP30` default, which was read back in the
  source. It is a strong claim and it is the kind that should be re-derived, not
  inherited: a future editor who wants to challenge it should re-run the reference
  to TSTOP and read its own `.out` energy-error column rather than trusting this
  paragraph. The narrower factual claims — the port's improvement (c41 0.2305 →
  0.1375, IE tracking to 0.8 %) — are in `parity_m41.json` and independent of the
  argument.
- **Two sources measured c41 and they disagree by 0.017** (§3.6 (h)):
  `parity_m41.json` 0.1375 vs the forensics builder's focused re-run 0.1542. The
  explanation (a case that terminates on the ERRN guard has a
  last-digit-sensitive abort cycle, hence a moving overlap window) is plausible
  and matches the M40 precedent, but it was NOT independently re-run to confirm
  which window each used. Same class, same conclusion, but do not quote c41 to 3
  decimals without saying which source.
- **The QBAT/QEPH kernels are new, large, and validated mostly by their own
  tests.** `shell_qbat.py` (1305 lines) and `shell_qeph.py` (1010 lines) landed in
  a single milestone with 19 and 24 dedicated tests respectively — patch tests,
  objectivity, closed-form stabilization stiffnesses, energy-split pins — plus the
  five RD-E-1000 MATCH results, which is genuinely strong end-to-end evidence.
  But the corpus exposure is one deck family: the §4.10 sweep proves 160 corpus
  decks ROUTE to these kernels without breaking the starter, and says nothing
  about whether their ENGINE results are right on those 160. LAW27/LAW36 shells
  are gated allowed for QBAT with no dedicated law-specific parity case. Treat
  "QBAT/QEPH are correct" as proven for the c02–c04/c08–c09 envelope and
  plausible-but-unmeasured outside it.
- **The QEPH port was INHERITED from a paused session, not written this session.**
  The `qeph-port` builder audited it line-for-line against upstream, re-validated
  it end-to-end, and re-ran both c08/c09 on the final tree reproducing all printed
  digits — but the original porting decisions are one remove from the reporting
  builder. Same caveat applies in weaker form to the guard fix: `engine.py`'s
  floor and `tests/test_m41_guard.py` were already in the tree from the paused
  session, and the `guard-conditioning` builder's contribution was VERIFICATION
  (reading `ecrit.F`/`freform.F` to confirm the citations are genuine rather than
  hallucinated, and proving the before/after on the real decks). That builder
  deliberately made no further edits rather than re-implement a correct fix — the
  right call, but it means the report's guard section is a verification record,
  not an implementation account.
- **THREE EXISTING TESTS WERE MODIFIED THIS MILESTONE — flagged explicitly,
  because "the suite is green" would otherwise hide it.** The M41 rules forbid
  weakening a test, so each change is recorded here with the editor's own
  assessment; the closer should confirm the judgement rather than take it from
  this paragraph.
  **(1) `test_m9_geomnl.py` — not a weakening, no assertion touched.** A pure
  NEP-52 import swap (`np.trapezoid` → `pyradioss.common.npcompat.trapezoid`) so
  the test runs on the NumPy 1.x the environment was downgraded to.
  **(2) `test_m40_rbody_stifr.py::test_shell_stifr_claim_mirror` — an expected
  value CORRECTED, tolerance unchanged at rel=1e-12.** M40's version asserted the
  BT deck's rotational lumping factor was `(t²+A)/12`; M41 asserts
  `t²/12 + A/9`. **The editor verified both upstream citations directly in the
  Fortran for this report and they are exact**: `cinmas.F` 919-925 reads
  `IF(INER_9_12/=ZERO) FAC=INER_9_12; ELSEIF(IHBE>=11) FAC=TWELVE; ELSE FAC=NINE`
  with the inertia formed as `AREA/FAC + THK²/12`, and `chvis3.F` line 250 reads
  `STIR(I) = STI(I) * (THK02(I)*INV12 + AREA(I)*INV9)`. So the BT family's
  upstream form really is `t²/12 + A/9`, and `(t²+A)/12` is the IHBE≥11
  BATOZ/QEPH branch (`cndt3.F` 209-218) that M40 applied to every family. This is
  a test that was pinning the WRONG closed form for a BT deck, now pinning the
  right one at the same tight tolerance — a fidelity gain, not a relaxation.
  **(3) `test_m4_contact.py::test_type11_edge_impact_momentum_and_energy` — the
  one that needs a second opinion.** This is the test the `guard-conditioning`
  builder flagged mid-milestone as failing against the in-flight shell WIP
  (isolated by worktree: passing on pure HEAD, failing with the working tree's
  shell code). It is GREEN on the final tree, but **it is green partly because
  its assertion was changed**, not only because code moved: the arrest check went
  from the four crossing nodes' mean z-velocity to the flyer's CENTRE-OF-MASS
  z-velocity, against the same −0.5 threshold, plus a NEW bounded-ringing
  assertion (`max|v_z| < 2.0`) that did not exist before. The rationale recorded
  in the test is that contact releases well before TSTOP and leaves those four
  nodes ringing in the strip's first bending mode, so their instantaneous mean at
  t=3.0 samples ring PHASE rather than arrest — and that sampling every 0.02 over
  t=2.84..3.00 the four-node mean dips below −0.5 at 2/9 stop times under FAC=9
  **and 3/9 under the old FAC=12**, i.e. the old assertion was already
  phase-flaky before M41 and passed at t=3.0 by coincidence. The CoM velocity is
  argued to be the physical arrest measure and phase-independent (ptp 3e-16
  across the window, −0.36865 under both lumpings). **The editor's assessment:
  this reads as a well-posedness fix with an added guard rather than a
  relaxation, and the supporting sweep is quantitative — but it is still an
  assertion changed on a test that was failing, so it is exactly the case the
  no-weakening rule exists to catch, and it was NOT independently re-derived
  here.** Verify the 2/9-vs-3/9 sampling claim before accepting it.
- **The M41 numbers were re-derived where cheap, inherited where not.** For this
  report the editor independently re-ran all 60 M41 tests (green), verified the
  full 1062-test collection count, recomputed the §4.10 per-kernel verdict split
  from `dispatch_map.json` against the committed coverage JSON, read the
  RD-E-1000 engine decks to establish the per-variant TSTOP values, and executed
  both sub-threshold benchmark decks to read their actual `auto` backend choice
  (which CORRECTED a builder-side table — §6.5 (b)). The parity rel-RMS values,
  the Fortran-side forensics, and the QBAT/QEPH patch-test tables are the
  builders' measurements, taken from the machine-readable deliverables, not
  re-measured here.
- **The report was written against a SHARED working tree with three builders
  editing shell code concurrently** (branch
  `claude/openradioss-python-m41-shell-technology`), under strict file ownership:
  BT internals, `shell_qbat.py`, `shell_qeph.py`, and each builder's own dispatch
  lines. Merge-time reconciliation is the parent's to confirm. One known
  cross-file consequence is already logged (§7 item 6: `shell_bt4`'s M40-era
  BATOZ-family dt-claim branches are now shadowed by the kernels that own them).

- **The RD-E-1000 headline is a DIAGNOSED non-MATCH, not a fix.** The milestone's
  central attempt — complete the /RBODY rotational-STIFR dt term so the floor
  reaches Fortran's, and (per the M39 §3.4 hypothesis) the full-run MATCH follows
  — succeeded at the dt claim (floors Fortran-exact) and FAILED at the MATCH. The
  residual is CHARACTERIZED (the dt-floor sweep proving floor-independence, the
  c41 BT-vs-BT isolation localizing a transverse-w hourglass mode from t≈900 ms)
  but NOT fixed; the fix is an M41 force-physics item. Two measurement sources
  give slightly different RD-E-1000 rms (the authoritative `parity_m40.json`:
  c04 0.2301, c08 0.2330; `rbody-stifr`'s focused re-runs: c04 0.2361, c08 0.2507
  — ~0.02 apart, different overlap windows); §3.5 uses `parity_m40.json` as
  authoritative and cites `rbody-stifr` for the dt-floor mechanism only.
- **A real REGRESSION landed and is NOT resolved this milestone.** The three
  RD-E-1000 Sf_0.1 variants (c40/c42/c44) went from M39's ~256 k-cycle runs to
  cycle-100 aborts under the STIFR change (`task_29ec1751`, §3.5 (d)). It is a
  genuine excited-mode / ill-conditioned-guard interaction on ~1e-8 J energies,
  not a measurement artifact; do NOT weaken the −15 % guard to mask it.
- **Every M40 wall clock is CONTENDED and the clean benchmark is RELATIVE-ONLY.**
  The user's 12-process `engine_win64_impi` MPI job ran the ENTIRE session (same
  PIDs 2+ days); no idle window opened. All absolute ms/cyc, s and cyc/s in
  §3.5/§6.4 are UPPER BOUNDS; parity classes, rel-RMS, §4.9 verdicts, and the
  back-to-back A/B RATIOS are load-independent and stand. **No M40 perf number
  should be propagated as a clean benchmark** — an uncontended re-timing is owed
  (§7 item 5). The M39 speed harness's `/FI IMAGENAME` contention filter
  UNDER-counted (returns 0 despite 12 live ranks); the M40 records use plain
  `tasklist` (§6.4 (e)).
- **Two LAW36 / numba tails were incomplete at handoff.** c19/c20 are the
  complete, authoritative LAW36 result (full-30 ms NORMAL, IE 1.1e-6, 10/10
  deletions); c23 (the lexer-fixed TETRA) was at interim IE 0.0000 (t≤14.1) with
  the deletion pattern already matching Fortran — the full-window number is
  expected at the c19/c20 level but was pending. The `numba-default` corpus
  spot-parity tail (tetra120_v0700, brick800_pressure) and the bundled
  both-backends byte-compare were still running / staged under contention (§7
  item 7); the auto-default's element-kernel parity is independently proven
  (M7 ≤ 1e-12, the 48-brick grid 8.85e-19), so this is a corpus-coverage gap, not
  a correctness one.
- **The report was written against a SHARED uncommitted working tree** (branch
  `claude/openradioss-python-m40-match-and-speed`); the six M40 builders edited
  different regions concurrently (`mass_scaling.py`/`rigid_body.py` for STIFR,
  `accel/__init__.py`/`engine.py` for numba-default, `materials/law36_tabulated.py`
  for LAW36, `kinematics.py`/`spring.py` for the residuals pack) — merge-time
  reconciliation is the parent's to confirm. All six builders filed reports (the
  process inverse of M39's three self-reported-FAILED), but the deviation
  narratives and the dt-floor / c41 experiments come from the builders'
  attribution and the machine-readable deliverables, not an independent re-run of
  every case.
- **Three of the six M39 builders self-reported FAILED** (`shell-fidelity`,
  `skew-frame`, `small-bugs`), so there is no builder report for the shell
  hourglass fix, the skew/frame reader, or the small-bug pack. Verification here
  is INDEPENDENT of those absent reports: the code is in the tree, its 72 new
  tests (`tests/test_m39_*.py`) were re-run GREEN for this report, the §4.8 sweep
  attributes 15 clean verdict conversions to it with 0 regressions, and §3.3
  measures the shell fix's physics effect. "FAILED" is a self-state artifact —
  but it does mean the mechanism narratives (which builder did what, the
  chvis3.F citation, the skew.py design) come from the measuring builders'
  attribution and the code/tests, not first-hand builder accounts.
- **Two PRE-EXISTING tests were RED in the shared working tree** — verified
  failing at report time, now RECONCILED by the integration verifier (§7 item 1):
  `test_m7_backends.py::test_shell_pre_post_parity` (a stale `_post` call not
  updated to the new `chvis3.F` `hqm/hqb/hqr/dt` signature) and
  `test_element_kernels.py::test_degenerated_brick_penta_run_as_collapsed_hexa`
  (renamed; asserted the OLD reject-penta behavior the small-bug pack
  intentionally replaced with run-as-collapsed). Both were unreconciled
  concurrent-edit fallout, NEITHER a NumPy-physics regression (the §6.3 gate
  proves the 9 bundled NumPy T01 are byte-identical to pre-speed); with the two
  reconciliations the fast/CI test tier is GREEN.
- **Every M39 wall clock is CONTENDED.** The user's 12-process
  `engine_win64_impi` MPI job was live the entire session (confirmed by
  `tasklist`: 12 engine processes + `mpiexec` at report time). All absolute
  ms/cyc, s and cyc/s in §3.3/§6.3 are UPPER BOUNDS; parity classes, rel-RMS,
  the §4.8 verdicts, per-stage SHARES and the alternated A/B speedup RATIOS are
  load-independent and stand. An uncontended re-timing on an idle box is owed
  (§7 item 4). No perf number from this round should be propagated as a clean
  benchmark.
- **The M39 parity coverage is INCOMPLETE — 35/65 official cases re-run** (§3.3),
  due to shell-family compute cost under 12-way contention; the un-run 30 and
  their expected status are documented (§7 item 6; RD-E-1000 remainder inert, the
  T1000 family with an apparent c33 engine unlock, the slow timeouts, LAW70
  c47/c49, the FRAME/MOV tensiles evicted by a background-task limit). The §4.8
  coverage sweep, by contrast, IS a complete 529/529 M39-tree measurement — but
  it is STARTER-side only (which decks parse/build, not physics parity), exactly
  as M38.
- **The shell-fidelity fix's off-target rows are reported as fidelity, not
  regression, per the brief.** c28/c32 V0240 (0.362 → 0.423) compares DIFFERENT
  run windows (M39 reached 21 % vs M38's ~7 % because the fix improved
  stability), and rigid_impactor's HE worsened (0.104 → 0.416) though its class
  held — neither is a clean before/after and both are flagged as such in §3.3.
- **The speed work's parity is proven at the NumPy-reference and
  isolated-kernel level, not corpus-wide under numba.** The −31 % numba lever was
  measured only on c46; making numba the default (§6.3f #1) still requires the
  full parity suite re-run under `PYRADIOSS_BACKEND=numba` (out of scope this
  round). The OPT-1 anim-path before/after could not be re-measured on the M38
  tree directly — M38 cannot read the M39 restart (it pickles the new
  `pyradioss.model.skew` object → `ModuleNotFoundError`), itself evidence of the
  fidelity work; the anim win is reported from the current measurement + verified
  byte-parity. `perf_m39.json` / `perf_m39_speed.json` were merged from several
  result files by `build_perf_m39.py`; a ~4-task concurrent-background limit
  silently killed some sweeps mid-run, so future large sweeps should stay ≤ 3
  concurrent background jobs.
- **The report was written against a SHARED uncommitted working tree** (HEAD
  `7853b26`, 28 modified files + the new modules); `checks.py`, `jit_kernels.py`,
  `shell_bt4.py` and `solid_hexa8.py` were edited by SEVERAL M39 builders
  concurrently (the shell `_post` signature, the OPT-1 `scatter3` + OPT-2
  `hexa_hgphys`/`law70_*` kernels, the degenerate-brick handling) — merge-time
  reconciliation is the parent's to confirm.

- **The M38 corpus re-sweep IS a real M38-tree measurement** (§4.7,
  `coverage_results_m38.json`) — unlike M36→M37, the sweep ran against the M38
  working tree and is atomic (source SHA-256 byte-identical before/after, HEAD
  at `977993b`). But it is STARTER-side only: it measures which decks
  parse/build, not physics parity.
- **The M38 parity/timing campaign completed via a coordinator resume**
  (§3.2/§6.2, `parity_m38.json` + `perf_m38.json`): the `parity-m38`
  builder's session ended at 6/52; the driver was resumed to 52/52. Its
  Fortran side reuses the M37 runs (identical binaries/comparison); its
  port side ran ENTIRELY under CPU contention (impi=12/12 — the user's own
  MPI job), so M38 port wall clocks are upper bounds. The three LAW70
  compression variants are stable but exceeded the 900 s budget
  (SKIPPED-SLOW) — their full-run NORMAL plus an uncontended timing pass
  are M39 sweep items.
- **One of six M38 builders filed a stub report.** `tetra4-convention`
  produced no report, but its /TETRA4 node-ordering / volume-sign fix
  (`solid_tetra4.py`) landed and is confirmed independently by the §4.7
  sweep (M37-BUG-3 9 → 0) and the §3.2 tetra comparisons — verification is
  the coverage delta + the module, not a builder report.
- **The report was written against a SHARED uncommitted working tree** carrying
  all M38 builders' edits (HEAD `977993b`); `checks.py` was edited concurrently
  by several builders (the MAT-density ALE exemption, the mat-0 /PART rule, the
  inactive-property warning) — merge-time reconciliation of `checks.py` is the
  parent's to confirm. Builder test suites were re-run green (prop-pack 23 new
  + 363 existing; ledger-hardening 10 new + ~500 existing across
  engine/element/implicit/contact/materials/reader/constraint); the full suite
  was not re-run for this report.
- **M37 edition (carried): the corpus numbers were M36 measurements over an M37
  tree.** Five of eight M37 builders reported; three failed (§ TL;DR — M37
  edition, item 6). All §3/§4.1–4.4/§6 corpus-scale numbers predate the M37
  fixes by construction; §4.5's deltas are slice-scale (60 cases) or
  domain-scale (193 /MAT blocks), not full-corpus. (§4.6 and §3.1 — the M37
  re-runs — are the authoritative M37 corpus and parity measurements.)
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

