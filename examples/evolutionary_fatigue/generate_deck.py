#!/usr/bin/env python3
"""
Generate the EVOLUTIONARY_FATIGUE example deck: the FULLY EVOLUTIONARY /
NON-SEPARABLE-PSD fatigue life of the SAME base-excited multi-mass "instrument
stack" the M20 spectral_fatigue and M25 nonstationary_fatigue examples shake
(M26, /IMPL/FATIG/EVOL), built on the M20 stationary estimators
(spectral_fatigue.py), the M25 block / amplitude-modulated machinery
(nonstationary_fatigue.py) AND the M26 evolutionary spectrogram
(evolutionary_fatigue.py).

This is deliberately the SAME base-driven spring stack the M20 spectral_fatigue,
the M24 nongaussian_fatigue and the M25 nonstationary_fatigue examples shake (a
shaker-table / launch-qualification model — Newland, "An Introduction to Random
Vibrations, Spectral & Wavelet Analysis"; Priestley, "Evolutionary spectra and
non-stationary processes" 1965/1967; Mark 1970 — non-stationary spectral
analysis). M25 already lifted stationarity in the RMS LEVEL: a fixed spectral
SHAPE whose amplitude follows a mission profile. M26 lifts the SEPARABILITY
assumption itself — here the excitation is a RESONANCE SWEEP: the input energy
band DRIFTS UP in frequency with time (a swept-sine-like "chirp"), so the response
excites a DIFFERENT mode of the stack in each time-window. The spectral SHAPE (its
bandwidth, its rates, WHICH mode dominates) genuinely changes window to window —
each window carries its OWN full moment set, not a scaling of a shared shape.

The stack's five modes sit near 31 / 91 / 142 / 179 / 199 Hz. The evolutionary
schedule sweeps the excitation window centre frequency fc from ~31 Hz (the first
mode) to ~200 Hz (the last), broadening the band from 12 to 30 Hz as it climbs —
a genuinely non-separable spectrogram S(omega, t). The RMS LEVEL still follows the
M25 mission profile (run-up / dwell / run-down), so /EVOL composes with /NSTAT.

Model (units mm / ms / kg -> forces kN, frequencies kHz): IDENTICAL to the M20
spectral_fatigue example —
  * N = 5 masses: node 1 the shaken BASE (pinned), nodes 2..6 the stacked masses
    along +x, one /SPRING per storey (its own /PART + /PROP/SPRING 'M K C'); the
    spring carries an axial FORCE — its stress-RESULTANT fatigue channel;
  * each storey: mass M = 2e-3 kg, stiffness K = 800 kN/mm, modal damping 3 %;
  * /FUNCT/10 — a FLAT band-limited input PSD S_aa(f) for /IMPL/FATIG/EVOL/BASE;
  * /FUNCT/20 — the RMS MISSION PROFILE: the input RMS LEVEL vs time (run-up /
    dwell / run-down). The PSD carries the base SHAPE; this function carries the
    time-varying LEVEL; the /EVOL card carries the time-varying SHAPE (the sweep).

What /IMPL/FATIG/EVOL/NSTAT/BASE does (a PORT card — the open-source engine has NO
frequency-domain / spectral / non-stationary / evolutionary fatigue solver;
freimpl.F reads only DYNA / BUCKL / DT / NONLIN / ARCL, and its sole "PSD" token
is IMUMPSD, a MUMPS flag):
  1..4. everything the M20 /IMPL/FATIG/BASE STATIONARY path does — the base-
     excitation FRF, the recovered STRESS modes, the stress response PSD
     S_sigmasigma = |H_sigma|^2 S_aa and its moments, the CRITICAL channel, and
     the four stationary estimators (narrow-band / Dirlik / Wirsching-Light /
     Tovo-Benasciutti) — reported UNCHANGED (byte-identical whether or not EVOL
     runs);
  5. the M25 NON-STATIONARY analysis (the RMS mission profile block Miner-sum /
     amplitude-modulated damage) — the SEPARABLE non-stationary answer, still with
     the FIXED spectral shape;
  6. THEN, alongside, the M26 FULLY EVOLUTIONARY analysis: apply a swept /
     broadening Gaussian window to the stress PSD per time-window (a resonance
     sweep — a genuinely NON-SEPARABLE spectrogram whose shape drifts), run the
     M20 estimators PER WINDOW on each window's OWN full moment set, and
     Palmgren-Miner SUM the window damages (each window a different bandwidth /
     rate / dominant mode);
  7. a seeded NON-SEPARABLE Monte-Carlo rainflow cross-check with a TIME-VARYING
     FILTER: per-window spectral-representation blocks (each window's OWN PSD)
     concatenated in time, ASTM E1049 rainflow, Miner sum — the time-domain answer
     the window spectral estimate approximates; its short-time spectrogram
     (per-window RMS + zero-crossing rate) tracks the swept centre frequency.

S-N curve: N = C*S^-m with m = 5 and C = 5e12 (identical to the M20 example).

What to look for in the listing (EVOLUTIONARY_FATIGUE_0001.out):
  * the "** RANDOM-VIBRATION (SPECTRAL) FATIGUE ** (/IMPL/FATIG)" block is the M20
    STATIONARY answer, UNCHANGED (the life at the nominal fixed shape + level);
  * the "** NON-STATIONARY / EVOLUTIONARY-PSD FATIGUE ** (/IMPL/FATIG/NSTAT)"
    block is the M25 SEPARABLE answer (the RMS mission profile on the FIXED shape);
  * the "** FULLY EVOLUTIONARY / NON-SEPARABLE-PSD FATIGUE ** (/IMPL/FATIG/EVOL)"
    block reports the drifting shape (fc 31 -> 200 Hz, bw 12 -> 30), the per-window
    shape drift (alpha2 and nu0 ranges — the point of a NON-separable spectrum),
    and for each estimator the STATIONARY rate NEXT TO the shape-evolutionary
    (window Miner-sum) rate and life, plus the non-separable Monte-Carlo;
  * the three lives (stationary, RMS-non-stationary M25, shape-evolutionary M26)
    are printed SIDE BY SIDE — the resonance sweep excites each mode only
    transiently, so the evolutionary damage reflects WHICH modes the sweep passes
    through and how long it dwells on each.

The full result (stationary summary + the M25 ``nonstationary`` and M26
``evolutionary`` sub-entries, each with the per-window shape breakdown, the window
Miner-sum, the induced kurtosis and the Monte-Carlo) is on
``model.implicit_result.fatigue``.

Run it (from this directory):
    python generate_deck.py          # regenerates the two .rad files
    pyradioss-starter -i EVOLUTIONARY_FATIGUE_0000.rad
    pyradioss-engine  -i EVOLUTIONARY_FATIGUE_0001.rad
"""

import os
# M36: decks are emitted through the package's fixed-format 2022 writer
# (pyradioss/input/deck_writer.py) — model definition above is unchanged,
# only the emission format moved to the real Radioss dialect.
import sys as _sys
_REPO = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))
if _REPO not in _sys.path:
    _sys.path.insert(0, _REPO)
from pyradioss.input import deck_writer  # noqa: E402


NMASS = 5            # stacked masses
DX = 10.0            # node spacing (mm)
MASS = 2.0e-3        # per-storey mass (kg)
K = 800.0            # per-storey stiffness (kN/mm)
ZETA = 0.03          # modal damping (3 %) for the FRF
NMODE = 5            # modes to extract
FMIN = 0.0           # analysis band start (1/ms)
FMAX = 250.0         # analysis band end (1/ms)
NF = 3000            # PSD sweep points
S0 = 1.0e2           # flat input base-acceleration PSD ((mm/ms^2)^2/kHz)
SN_M = 5.0           # S-N slope m (N = C*S^-m)
SN_C = 5.0e12        # S-N coefficient C (spring-force units)
MCDUR = 600.0        # Monte-Carlo cross-check record length (ms)
SEED = 20260         # Monte-Carlo random seed (reproducible)
NSEG = 12            # blocks the RMS mission profile is sampled into (M25)
# M26 evolutionary drifting-shape schedule: a resonance SWEEP through the modes
FC0 = 31.0           # window centre freq at the first window (Hz, ~mode 1)
FC1 = 200.0          # window centre freq at the last window (Hz, ~mode 5)
BW0 = 12.0           # window bandwidth start (Hz)
BW1 = 30.0           # window bandwidth end (broadens as the sweep climbs)
NWIN = 12            # time-windows the evolutionary spectrogram is sampled into

here = os.path.dirname(os.path.abspath(__file__))


def main():
    lines = ["#RADIOSS STARTER", "/BEGIN",
             "EVOLUTIONARY_FATIGUE - base-excited instrument stack, resonance "
             "sweep (pyradioss /IMPL/FATIG/EVOL)"]

    # ---- nodes: stack along +x (node 1 = shaken base) ---------------------
    lines.append("/NODE")
    for i in range(NMASS + 1):
        lines.append(f"{i + 1:10d}{i * DX:20.10f}{0.0:20.10f}{0.0:20.10f}")

    # ---- one /SPRING (+ /PART + /PROP) per storey -------------------------
    for i in range(NMASS):
        lines.append(f"/SPRING/{i + 1}")
        lines.append(f"{i + 1:10d}{i + 1:10d}{i + 2:10d}")
    for i in range(NMASS):
        lines += [f"/PART/{i + 1}", f"storey{i + 1}", f"{i + 1:10d}         1"]

    lines += ["/MAT/LAW1/1", "steel elastic", "   7.8e-6",
              "     210.0       0.0"]
    for i in range(NMASS):
        lines += [f"/PROP/SPRING/{i + 1}", f"sp{i + 1}",
                  f"{MASS:12g}{K:12g}         0.0"]

    # ---- constraints: base (node 1) pinned, stack axial (x) only ----------
    stack = "\n".join(str(i + 2) for i in range(NMASS))
    lines += ["/GRNOD/NODE/1", "base", "1",
              "/GRNOD/NODE/2", "stack", stack,
              "/BCS/1", "pinned base (the support)",
              "       111       111         0         1",
              "/BCS/2", "axial only (stacked masses)",
              "       011       111         0         2"]

    # ---- /FUNCT/10: flat band-limited input base-acceleration PSD ---------
    lines += ["/FUNCT/10", "flat input PSD S_aa(f)",
              f"       0.0{S0:12g}", f"  100000.0{S0:12g}"]

    # ---- /FUNCT/20: the RMS MISSION PROFILE (level vs time) ---------------
    # run-up (0.4 -> 1.5), dwell (1.5), run-down (1.5 -> 0.3); time in ms
    lines += ["/FUNCT/20", "RMS mission profile (run-up / dwell / run-down)",
              "       0.0       0.4",
              "     150.0       1.5",
              "     350.0       1.5",
              "     500.0       0.3"]
    lines.append("/END")

    deck_writer.write_starter_from_port_lines(
        lines, os.path.join(here, "EVOLUTIONARY_FATIGUE_0000.rad"), runname="EVOLUTIONARY_FATIGUE")

    # ---- engine: a bare implicit static step + the M26 evolutionary fatigue
    engine = [
        "#RADIOSS ENGINE",
        "/RUN/EVOLUTIONARY_FATIGUE/1",
        "1.0",                          # one (zero-load) static increment
        "/IMPL",                        # implicit static driver hosts the analysis
        "/IMPL/FATIG/EVOL/NSTAT/BASE",  # M26: evolutionary (+ M25 RMS) base-accel
        # line 1 (PSD sweep): fmin fmax nf funct dir nmode  (funct 10 = S_aa)
        f"{FMIN}  {FMAX}  {NF}  10  0  {NMODE}",
        # line 2 (S-N + Monte-Carlo): m C zeta mean ult mcdur seed
        f"{SN_M}  {SN_C}  {ZETA}  0.0  0.0  {MCDUR}  {SEED}",
        # line 3 (M25 non-stationary): RMS modulation /FUNCT id + block count
        f"20  {NSEG}",
        # line 4 (M26 evolutionary): fc0 fc1 bw0 bw1 nwin  (the resonance sweep)
        f"{FC0}  {FC1}  {BW0}  {BW1}  {NWIN}",
        "/PRINT/-500",
        "/STOP",
        "15.0",
    ]
    deck_writer.write_engine_from_port_lines(
        engine, os.path.join(here, "EVOLUTIONARY_FATIGUE_0001.rad"))
    print("wrote EVOLUTIONARY_FATIGUE_0000.rad / EVOLUTIONARY_FATIGUE_0001.rad")


if __name__ == "__main__":
    main()
