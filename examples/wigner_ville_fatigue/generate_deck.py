#!/usr/bin/env python3
"""
Generate the WIGNER_VILLE_FATIGUE example deck: the CONTINUOUS WIGNER-VILLE / LOEVE
INSTANTANEOUS time-frequency fatigue life of the SAME base-excited multi-mass
"instrument stack" the M20 spectral_fatigue, the M25 nonstationary_fatigue and the
M26 evolutionary_fatigue examples shake (M31, /IMPL/FATIG/WVILLE), built on the M20
stationary estimators (spectral_fatigue.py), the M26 windowed evolutionary
spectrogram (evolutionary_fatigue.py) AND the M31 continuous instantaneous spectrum
(wigner_ville_fatigue.py).

This is deliberately the SAME base-driven spring stack the M20/M24/M25/M26 examples
shake (a shaker-table / launch-qualification model — Newland, "An Introduction to
Random Vibrations, Spectral & Wavelet Analysis"; Wigner 1932 / Ville 1948 — the
Wigner-Ville distribution; Mark 1970 / Martin & Flandrin 1985 — the Wigner-Ville
spectrum of a nonstationary random process; Cohen 1989 — the class of time-frequency
distributions and cross-term smoothing; Priestley 1965/1967 — the evolutionary
spectrum the windowed spectrogram approximates and M31 makes continuous). M26 lifted
the SEPARABILITY assumption with a short-time WINDOWED spectrogram: it partitions the
resonance sweep into a few short time-WINDOWS, treats each as locally stationary with
its OWN full stress PSD, and Miner-SUMS the per-window damages. M31 lifts that
WINDOWING itself — it evaluates the drifting shape CONTINUOUSLY at a fine instant
grid (a bilinear time-frequency distribution S_WV(omega, t)) and Miner-INTEGRATES the
per-instant damage over time (an integral, not a per-window sum).

THE DEMONSTRATOR: a resonance sweep through a WELL-SEPARATED modal band with a NARROW
excitation window. The excitation window centre frequency fc sweeps ACROSS the stack
so that at the sweep endpoints it sits ON a resonance but at the coarse window
mid-times it can sit BETWEEN modes — so the COARSE windowed Miner-SUM (a few-point
midpoint quadrature) badly UNDER-samples the damage the sweep does as it passes
THROUGH the resonances, while the CONTINUOUS instantaneous integral (a fine-grid
quadrature) captures it — and tracks the non-separable Monte-Carlo (which sweeps
continuously and rainflows the WHOLE record, including the window-straddling cycles
the windowed sum misses) FAR better. The continuous integral, the M26 windowed sum
and the Monte-Carlo are printed SIDE BY SIDE.

The Cohen-class cross-term smoothing knob (smooth) is left at 0 (the raw
instantaneous WVD); the grid-refinement factor (refine) subdivides each M26 window
into fine instants. refine = 1, smooth = 0 would recover the M26 windowed spectrogram
EXACTLY (delegated byte-identically); here refine = 10 resolves the sweep finer than
the windows do.

Model (units mm / ms / kg -> forces kN, frequencies kHz): IDENTICAL to the M20
spectral_fatigue example — N = 5 masses (node 1 the shaken BASE, nodes 2..6 the
stacked masses), one /SPRING per storey (its axial FORCE the fatigue channel),
M = 2e-3 kg, K = 800 kN/mm, 3 % modal damping; /FUNCT/10 a flat band-limited input
PSD S_aa(f); /FUNCT/20 the RMS mission profile (run-up / dwell / run-down). The stack
modes sit near 31 / 91 / 142 / 179 / 199 Hz. The sweep runs fc 31 -> 200 Hz with a
NARROW window (bw 4 -> 6 Hz) so it selects one mode at a time.

S-N curve: N = C*S^-m with m = 5 and C = 5e12 (identical to the M20/M26 examples).

What /IMPL/FATIG/WVILLE/EVOL/NSTAT/BASE does (a PORT card — the open-source engine has
NO frequency-domain / spectral / time-frequency / Wigner-Ville fatigue solver;
freimpl.F reads only DYNA / BUCKL / DT / NONLIN / ARCL, and its sole "PSD" token is
IMUMPSD, a MUMPS flag):
  1..5. everything the M20 STATIONARY, the M25 NON-STATIONARY and the M26 WINDOWED
     evolutionary paths do — reported UNCHANGED (byte-identical whether or not
     /WVILLE runs);
  6. THEN, alongside, the M31 CONTINUOUS WIGNER-VILLE analysis: evaluate the drifting
     shape CONTINUOUSLY at nt = nwin * refine fine instants, reduce the M20
     estimators AT EACH INSTANT, and Palmgren-Miner INTEGRATE over time, reporting
     the instantaneous spectral-PEAK drift, the shrinking window-boundary caveat and
     the continuous-integral Dirlik damage / life NEXT TO the M26 windowed one;
  7. a continuous non-separable Monte-Carlo cross-check on the fine instant grid.

What to look for in the listing (WIGNER_VILLE_FATIGUE_0001.out):
  * the "** RANDOM-VIBRATION (SPECTRAL) FATIGUE **" (M20), "** NON-STATIONARY /
    EVOLUTIONARY-PSD FATIGUE **" (M25) and "** FULLY EVOLUTIONARY / NON-SEPARABLE-PSD
    FATIGUE **" (M26) blocks are UNCHANGED;
  * the "** CONTINUOUS WIGNER-VILLE INSTANTANEOUS TIME-FREQUENCY FATIGUE **
    (/IMPL/FATIG/WVILLE)" block reports the fine grid (nt = nwin x refine), the
    instantaneous peak-frequency drift, the window-boundary caveat shrinking
    (fine / coarse), the continuous-integral Dirlik rate NEXT TO the M26 windowed
    Miner-SUM rate, and the continuous Monte-Carlo — the continuous integral tracking
    the Monte-Carlo where the coarse windowed sum under-samples the swept resonances.

The full result (the M20 stationary summary + the M25 ``nonstationary``, M26
``evolutionary`` and M31 ``wigner_ville`` sub-entries) is on
``model.implicit_result.fatigue``.

Run it (from this directory):
    python generate_deck.py          # regenerates the two .rad files
    pyradioss-starter -i WIGNER_VILLE_FATIGUE_0000.rad
    pyradioss-engine  -i WIGNER_VILLE_FATIGUE_0001.rad
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
SEED = 31026         # Monte-Carlo random seed (reproducible)
NSEG = 12            # blocks the RMS mission profile is sampled into (M25)
# M26 evolutionary drifting-shape schedule: a resonance SWEEP through the modes with
# a NARROW window (so the coarse windowed sum under-samples the swept resonances)
FC0 = 31.0           # window centre freq at the first window (Hz, ~mode 1)
FC1 = 200.0          # window centre freq at the last window (Hz, ~mode 5)
BW0 = 4.0            # window bandwidth start (Hz, NARROW — one mode at a time)
BW1 = 6.0            # window bandwidth end
NWIN = 6             # M26 time-windows (coarse — the sweep is under-sampled)
# M31 continuous Wigner-Ville parameters (trailing columns of the /EVOL line)
REFINE = 10          # fine instants per window (nt = NWIN * REFINE)
SMOOTH = 0.0         # Cohen-class cross-term smoothing width (0 = raw WVD)

here = os.path.dirname(os.path.abspath(__file__))


def main():
    lines = ["#RADIOSS STARTER", "/BEGIN",
             "WIGNER_VILLE_FATIGUE - base-excited instrument stack, continuous "
             "instantaneous spectrum (pyradioss /IMPL/FATIG/WVILLE)"]

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
    lines += ["/FUNCT/20", "RMS mission profile (run-up / dwell / run-down)",
              "       0.0       0.4",
              "     150.0       1.5",
              "     350.0       1.5",
              "     500.0       0.3"]
    lines.append("/END")

    deck_writer.write_starter_from_port_lines(
        lines, os.path.join(here, "WIGNER_VILLE_FATIGUE_0000.rad"), runname="WIGNER_VILLE_FATIGUE")

    # ---- engine: a bare implicit static step + the M31 continuous fatigue --
    engine = [
        "#RADIOSS ENGINE",
        "/RUN/WIGNER_VILLE_FATIGUE/1",
        "1.0",                              # one (zero-load) static increment
        "/IMPL",                            # implicit static driver hosts it
        "/IMPL/FATIG/WVILLE/EVOL/NSTAT/BASE",  # M31 continuous (+ M26/M25) base-accel
        # line 1 (PSD sweep): fmin fmax nf funct dir nmode  (funct 10 = S_aa)
        f"{FMIN}  {FMAX}  {NF}  10  0  {NMODE}",
        # line 2 (S-N + Monte-Carlo): m C zeta mean ult mcdur seed
        f"{SN_M}  {SN_C}  {ZETA}  0.0  0.0  {MCDUR}  {SEED}",
        # line 3 (M25 non-stationary): RMS modulation /FUNCT id + block count
        f"20  {NSEG}",
        # line 4 (M26 drifting shape + M31 refine/smooth): fc0 fc1 bw0 bw1 nwin
        #         refine smooth  (the resonance sweep + the continuous fine grid)
        f"{FC0}  {FC1}  {BW0}  {BW1}  {NWIN}  {REFINE}  {SMOOTH}",
        "/PRINT/-500",
        "/STOP",
        "15.0",
    ]
    deck_writer.write_engine_from_port_lines(
        engine, os.path.join(here, "WIGNER_VILLE_FATIGUE_0001.rad"))
    print("wrote WIGNER_VILLE_FATIGUE_0000.rad / WIGNER_VILLE_FATIGUE_0001.rad")


if __name__ == "__main__":
    main()
