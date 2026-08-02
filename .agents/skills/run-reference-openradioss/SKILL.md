---
name: run-reference-openradioss
description: Launch the reference (Fortran) OpenRadioss starter/engine on this machine with the correct runtime environment, and diagnose launch failures. Use for any differential-validation run, oracle run, or when a starter/engine invocation fails with 0xC0000135, "libiomp5md.dll", "impi.dll", or hangs mid-iteration.
---

# Run the reference OpenRadioss (Windows, this machine)

The reference install is `C:\OpenRadioss` (READ-ONLY — never write into it).
Solvers: `exec\starter_win64.exe`, `exec\engine_win64.exe` (Intel-MPI build —
imports `impi.dll` even at np=1), plus `exec\anim_to_vtk_win64.exe` and
`exec\th_to_csv_win64.exe` for post-processing. The matching Fortran source
tree is `C:\OpenRadioss\source\` — that is what ported formulas must cite.

## The one command

Run solvers ONLY through the repo helper (it sets the whole env; a bare exe
call dies on missing DLLs). Copy decks to a scratch dir first — the run
writes .out/.rst/T01/A-files next to the deck:

```
powershell -File tools\run_reference_or.ps1 -Deck "<scratch>\model_0000.rad"
powershell -File tools\run_reference_or.ps1 -Deck "<scratch>\model_0001.rad"
powershell -File tools\run_reference_or.ps1 -Deck "<scratch>\model_0000.rad" -Both
```

`-Np <n>` SPMD domains (engine goes through mpiexec when >1), `-Nt <n>` OpenMP
threads (default 6 — deliberate, see below). Exit code = the solver's own.

The env the script sets (for manual debugging only — prefer the script):

| What | Value | Why |
|---|---|---|
| PATH + | `C:\OpenRadioss\extlib\intelOneAPI_runtime\win64` | `libiomp5md.dll` — dies 0xC0000135 without it |
| PATH + | `C:\OpenRadioss\extlib\hm_reader\win64` | native LS-DYNA `.k` reading |
| PATH + | `C:\OpenRadioss\extlib\h3d\lib\win64` | H3D output |
| PATH + | `C:\Program Files (x86)\Intel\oneAPI\mpi\latest\bin` (+ `\libfabric\bin`) | `impi.dll`; needed even at np=1 |
| `RAD_CFG_PATH` | `C:\OpenRadioss\hm_cfg_files` | starter keyword config |
| `KMP_BLOCKTIME=0`, `OMP_WAIT_POLICY=PASSIVE` | | hybrid P/E-core CPU: spinning threads livelock the engine under desktop load |

## Reading the result

- **Starter**: `<root>_0000.out` → final `ERROR(S)` / `WARNING(S)` summary
  block. `0 ERROR(S)` + `TERMINATION WITH WARNING` is a clean run. Error
  count > 0 → search the same file for `ERROR ID :` and read the DESCRIPTION
  right below the first hit (the end-of-file table only gives categories).
- **Engine**: `<root>_0001.out` must end `NORMAL TERMINATION`. A tiny (<100 B)
  `_0001.out` with `ERROR TERMINATION` almost always means the starter never
  wrote restart files — fix the starter run first.

## Failure signatures

| Signature | Meaning |
|---|---|
| `-1073741515` / `0xC0000135` | missing DLL → env above not set (use the script) |
| exit 2, `.out` written | solver ran, deck has input ERRORS — read the `.out` |
| exit 0 | clean (starter warnings still possible — check the summary) |
| engine hangs, `.out` stops growing | OMP livelock → the script's KMP/OMP vars + `-Nt 6` |
| mpiexec run killed → orphan processes | `Stop-Process -Name engine_win64 -Force` before relaunch |

## Related facts

- A-files → `anim_to_vtk_win64.exe <Afile>`; binary T01 → `th_to_csv_win64.exe`
  (both need the same PATH env — run them inside a script session, or via the
  validation harness which does this for you).
- **Never feed a port-written deck raw to the Fortran solver** — go through
  `tools\validate_vs_fortran.py`, which applies the documented real-format
  fixups first (see the validation-compare skill).
- Reference run data (Ryan Lee decks, official demo zips, reference runs)
  lives under `E:\openradioss_run\` — READ-ONLY: copy out, never write there.
- Native `.k` decks work via hm_reader but that reader has known bugs
  (one-way contact segfault, IVN>10k hang, MAT22 silently dropped) — prefer
  converted .rad decks; use native reads only for cross-checks.
