#!/usr/bin/env python3
"""
Generate the NONGAUSSIAN_WIGNER_VILLE_FATIGUE example deck: the TIME-VARYING
NON-GAUSSIAN INSTANTANEOUS time-frequency fatigue life of the SAME base-excited
multi-mass "instrument stack" the M20 spectral_fatigue, the M24 nongaussian_fatigue,
the M25 nonstationary_fatigue, the M26 evolutionary_fatigue and the M31
wigner_ville_fatigue examples shake (M32, /IMPL/FATIG/NGAUSS + /WVILLE) — the
CONVERGENCE of the M24 stationary Winterstein-Hermite kurtosis correction and the M31
continuous Wigner-Ville instantaneous spectrum.

This is deliberately the SAME base-driven spring stack the M20/M24/M25/M26/M31
examples shake (a shaker-table / launch-qualification model — Newland, "An
Introduction to Random Vibrations, Spectral & Wavelet Analysis"; Winterstein 1988 —
the Hermite-moment model; Benasciutti & Tovo 2005/2006, Braccesi 2009, Rizzi & Kihm
2013 — non-Gaussian spectral fatigue; Wigner 1932 / Ville 1948 — the Wigner-Ville
distribution; Mark 1970 / Martin & Flandrin 1985 — the Wigner-Ville spectrum of a
non-stationary process; Priestley 1965 — the evolutionary spectrum). Two milestones
meet here:
  * M24 corrected ONE stationary equivalent scalar for a FIXED kurtosis gamma_4: the
    Winterstein factor lambda_ng SCALES the Gaussian spectral damage.
  * M31 gave the Gaussian response a CONTINUOUS instantaneous spectrum S_WV(omega, t):
    the drifting shape reduced AT EACH INSTANT and Miner-INTEGRATED.
M32 lets the kurtosis gamma_4(t) of the instantaneous equivalent scalar VARY WITH TIME
ALONG the continuous spectrum: at each fine instant it re-computes the M24 lambda_ng
from that instant's bandwidth alpha_2(t) AND a TIME-VARYING target gamma_4(t), then
Miner-INTEGRATES the non-Gaussian per-instant damage. The M31 Gaussian-continuous and
the M24 stationary-non-Gaussian lives are printed SIDE BY SIDE with the M32 one.

THE DEMONSTRATOR: a BURSTY / non-Gaussian transient. The excitation window sweeps
across the modal band (fc 31 -> 200 Hz, a narrow bw so it selects one mode at a time,
the M31 continuous demonstrator) WHILE the kurtosis rises during a mid-mission BURST
and relaxes back to Gaussian either side (a kurtosis-vs-time /FUNCT gamma_4(t): 3 at
the run-up, up to ~9 during the dwell burst, back toward 3 at the run-down — the kind
of intermittent, spiky loading a gust / impact / cavitation transient produces). So
the leptokurtic damage AMPLIFICATION lambda_ng(t) itself DRIFTS with time: near 1 (no
extra damage) at the Gaussian ends and well above 1 during the burst, where the spikes
do the damage. The continuous Miner-integral of lambda_ng(t) (dD/dt)_G(t) is
cross-validated by a non-Gaussian NON-STATIONARY Monte-Carlo (the M31 continuous
non-separable record pushed through a per-instant Hermite transform, its induced local
kurtosis tracking gamma_4(t), rainflowed over the WHOLE record).

Model (units mm / ms / kg -> forces kN, frequencies kHz): IDENTICAL to the M20 / M31
examples — N = 5 masses (node 1 the shaken BASE, nodes 2..6 the stacked masses), one
/SPRING per storey (its axial FORCE the fatigue channel), M = 2e-3 kg, K = 800 kN/mm,
3 % modal damping; /FUNCT/10 a flat band-limited input PSD S_aa(f); /FUNCT/20 the RMS
mission profile; /FUNCT/30 the TARGET kurtosis gamma_4(t) (the M32 addition). The stack
modes sit near 31 / 91 / 142 / 179 / 199 Hz; the sweep runs fc 31 -> 200 Hz.

S-N curve: N = C*S^-m with m = 5 and C = 5e12 (identical to the M20/M26/M31 examples).

What /IMPL/FATIG/NGAUSS/WVILLE/EVOL/NSTAT/BASE does (a PORT card — the open-source
engine has NO frequency-domain / spectral / time-frequency / non-Gaussian fatigue
solver; freimpl.F reads only DYNA / BUCKL / DT / NONLIN / ARCL, its sole "PSD" token
IMUMPSD a MUMPS flag):
  1..6. everything the M20 STATIONARY, the M24 STATIONARY-NON-GAUSSIAN, the M25
     NON-STATIONARY, the M26 WINDOWED evolutionary and the M31 CONTINUOUS
     Wigner-Ville paths do — reported UNCHANGED (byte-identical whether or not the M32
     time-varying path runs);
  7. THEN, alongside, the M32 TIME-VARYING NON-GAUSSIAN analysis: at each fine
     instant re-compute lambda_ng(t) from alpha_2(t) and gamma_4(t) (sampled from
     /FUNCT/30), reduce and Miner-INTEGRATE, reporting the gamma_4(t) schedule, the
     lambda_ng(t) DRIFT (min .. max) and the non-Gaussian continuous-integral damage /
     life NEXT TO the M31 Gaussian-continuous and the M24 stationary-non-Gaussian ones;
  8. a non-Gaussian non-stationary Monte-Carlo cross-check (induced kurtosis tracking
     gamma_4(t)).

What to look for in the listing (NONGAUSSIAN_WIGNER_VILLE_FATIGUE_0001.out):
  * the "** RANDOM-VIBRATION (SPECTRAL) FATIGUE **" (M20), "** NON-GAUSSIAN / KURTOSIS
    FATIGUE **" (M24 stationary), "** NON-STATIONARY / EVOLUTIONARY-PSD FATIGUE **"
    (M25), "** FULLY EVOLUTIONARY / NON-SEPARABLE-PSD FATIGUE **" (M26) and
    "** CONTINUOUS WIGNER-VILLE INSTANTANEOUS TIME-FREQUENCY FATIGUE **" (M31) blocks
    are UNCHANGED;
  * the "** TIME-VARYING NON-GAUSSIAN INSTANTANEOUS TIME-FREQUENCY FATIGUE **
    (/IMPL/FATIG/NGAUSS+WVILLE)" block reports the gamma_4(t) schedule, the
    lambda_ng(t) drift, the non-Gaussian continuous-integral Dirlik rate NEXT TO the
    M31 Gaussian-continuous rate and the M24 stationary reference, and the non-Gaussian
    Monte-Carlo whose sample kurtosis tracks gamma_4(t).

The full result (the M20 summary + the M24 ``nongaussian``, M25 ``nonstationary``, M26
``evolutionary`` and M31 ``wigner_ville`` sub-entries, with the M32 time-varying
correction under ``wigner_ville['nongaussian']``) is on
``model.implicit_result.fatigue``.

Run it (from this directory):
    python generate_deck.py          # regenerates the two .rad files
    pyradioss-starter -i NONGAUSSIAN_WIGNER_VILLE_FATIGUE_0000.rad
    pyradioss-engine  -i NONGAUSSIAN_WIGNER_VILLE_FATIGUE_0001.rad
"""

import os

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
# M26 evolutionary drifting-shape schedule: a resonance SWEEP through the modes
FC0 = 31.0           # window centre freq at the first window (Hz, ~mode 1)
FC1 = 200.0          # window centre freq at the last window (Hz, ~mode 5)
BW0 = 4.0            # window bandwidth start (Hz, NARROW — one mode at a time)
BW1 = 6.0            # window bandwidth end
NWIN = 6             # M26 time-windows
# M31 continuous Wigner-Ville parameters (trailing columns of the /EVOL line)
REFINE = 10          # fine instants per window (nt = NWIN * REFINE)
SMOOTH = 0.0         # Cohen-class cross-term smoothing width (0 = raw WVD)
# M32 non-Gaussian: the target kurtosis is a TIME-VARYING /FUNCT (the burst).
# The constant KURT0 drives the M24 STATIONARY reference (the "if we had assumed one
# fixed kurtosis" answer); the /FUNCT/30 gamma_4(t) drives the M32 time-varying
# correction for /WVILLE — so the listing shows the constant-kurtosis and the
# time-varying answers side by side.
KFUNCT = 30          # /FUNCT id for the target kurtosis gamma_4(t)
KURT0 = 6.0          # representative constant kurtosis (the M24 stationary reference)

here = os.path.dirname(os.path.abspath(__file__))


def main():
    lines = ["#RADIOSS STARTER", "/BEGIN",
             "NONGAUSSIAN_WIGNER_VILLE_FATIGUE - base-excited instrument stack, "
             "time-varying non-Gaussian instantaneous spectrum "
             "(pyradioss /IMPL/FATIG/NGAUSS+WVILLE)"]

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

    # ---- /FUNCT/30: the TARGET KURTOSIS gamma_4(t) (the M32 addition) -----
    # a mid-mission leptokurtic BURST: Gaussian (3) at the ends, spiky (~9) during
    # the dwell — the intermittent transient the time-varying correction models
    lines += ["/FUNCT/30", "target kurtosis gamma_4(t) (bursty transient)",
              "       0.0       3.0",
              "     120.0       3.5",
              "     250.0       9.0",
              "     380.0       4.0",
              "     500.0       3.0"]
    lines.append("/END")

    with open(os.path.join(here, "NONGAUSSIAN_WIGNER_VILLE_FATIGUE_0000.rad"),
              "w") as fh:
        fh.write("\n".join(lines) + "\n")

    # ---- engine: a bare implicit static step + the M32 continuous fatigue --
    engine = [
        "#RADIOSS ENGINE",
        "/RUN/NONGAUSSIAN_WIGNER_VILLE_FATIGUE/1",
        "1.0",                              # one (zero-load) static increment
        "/IMPL",                            # implicit static driver hosts it
        # M32 = M31 continuous (WVILLE) + M24 non-Gaussian (NGAUSS) + M26 EVOL + M25
        # NSTAT, base-acceleration driven
        "/IMPL/FATIG/NGAUSS/WVILLE/EVOL/NSTAT/BASE",
        # line 1 (PSD sweep): fmin fmax nf funct dir nmode  (funct 10 = S_aa)
        f"{FMIN}  {FMAX}  {NF}  10  0  {NMODE}",
        # line 2 (S-N + Monte-Carlo): m C zeta mean ult mcdur seed
        f"{SN_M}  {SN_C}  {ZETA}  0.0  0.0  {MCDUR}  {SEED}",
        # line 3 (M24 kurtosis + M32 time-varying): kurt skew kfunct [kurt1]
        #         (the /FUNCT/30 gamma_4(t) overrides the constant kurt for /WVILLE)
        f"{KURT0}  0.0  {KFUNCT}",
        # line 4 (M25 non-stationary): RMS modulation /FUNCT id + block count
        f"20  {NSEG}",
        # line 5 (M26 drifting shape + M31 refine/smooth): fc0 fc1 bw0 bw1 nwin
        #         refine smooth
        f"{FC0}  {FC1}  {BW0}  {BW1}  {NWIN}  {REFINE}  {SMOOTH}",
        "/PRINT/-500",
        "/STOP",
        "15.0",
    ]
    with open(os.path.join(here, "NONGAUSSIAN_WIGNER_VILLE_FATIGUE_0001.rad"),
              "w") as fh:
        fh.write("\n".join(engine) + "\n")
    print("wrote NONGAUSSIAN_WIGNER_VILLE_FATIGUE_0000.rad / "
          "NONGAUSSIAN_WIGNER_VILLE_FATIGUE_0001.rad")


if __name__ == "__main__":
    main()
