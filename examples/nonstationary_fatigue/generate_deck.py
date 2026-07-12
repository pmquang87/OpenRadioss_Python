#!/usr/bin/env python3
"""
Generate the NONSTATIONARY_FATIGUE example deck: the NON-STATIONARY /
EVOLUTIONARY-PSD fatigue life of the SAME base-excited multi-mass "instrument
stack" the M20 spectral_fatigue example shakes (M25, /IMPL/FATIG/NSTAT), built on
the M20 stationary spectral estimators (spectral_fatigue.py) and the
piecewise-stationary "mission profile" block Miner-sum + the amplitude-modulated
(evolutionary S(w,t) = |A(t)|^2 S(w)) damage (nonstationary_fatigue.py).

This is deliberately the SAME base-driven spring stack the M20 spectral_fatigue
and the M24 nongaussian_fatigue examples shake (a shaker-table / launch-
qualification model — Newland, "An Introduction to Random Vibrations…";
Wolfsteiner & Trapp, "Fatigue life due to non-stationary vibration"; Kihm,
Ferguson & Antoni, non-stationary random-vibration fatigue): a random support-
acceleration PSD drives the structure, but now the input RMS LEVEL VARIES WITH
TIME along a MISSION PROFILE — a run-up (ramp the amplitude up), a dwell (hold at
the qualification level) and a run-down (ramp back down). The stationary M20
analysis assumes ONE fixed RMS forever; M25 asks: what is the life under the
ACTUAL time-varying mission, and how does the varying RMS (which makes the process
non-Gaussian / leptokurtic) change the answer?

Model (units mm / ms / kg -> forces kN, frequencies kHz): IDENTICAL to the M20
spectral_fatigue example —
  * N = 5 masses: node 1 the shaken BASE (pinned), nodes 2..6 the stacked masses
    along +x, one /SPRING per storey (its own /PART + /PROP/SPRING 'M K C'); the
    spring carries an axial FORCE — its stress-RESULTANT fatigue channel;
  * each storey: mass M = 2e-3 kg, stiffness K = 800 kN/mm, modal damping 3 %;
  * /FUNCT/10 — a FLAT band-limited input PSD S_aa(f) for /IMPL/FATIG/NSTAT/BASE;
  * /FUNCT/20 — the RMS MISSION PROFILE: the input RMS SCALE as a function of time
    (run-up 0.4 -> 1.5, dwell at 1.5, run-down 1.5 -> 0.3). The PSD carries the
    SHAPE; this function carries the TIME-VARYING LEVEL.

What /IMPL/FATIG/NSTAT/BASE does (a PORT card — the open-source engine has NO
frequency-domain / spectral / non-stationary fatigue solver; freimpl.F reads only
DYNA / BUCKL / DT / NONLIN / ARCL, and its sole "PSD" token is IMUMPSD, a MUMPS
flag):
  1..4. everything the M20 /IMPL/FATIG/BASE STATIONARY path does — the base-
     excitation FRF, the recovered STRESS modes, the stress response PSD
     S_sigmasigma = |H_sigma|^2 S_aa and its moments, the CRITICAL channel, and
     the four stationary estimators (narrow-band / Dirlik / Wirsching-Light /
     Tovo-Benasciutti) — reported UNCHANGED (byte-identical whether or not NSTAT
     runs);
  5. THEN, alongside, the NON-STATIONARY analysis: partition the mission /FUNCT
     into stationary BLOCKS, scale the shared stress-PSD shape by each block's RMS
     (the moments scale as the RMS squared), run the M20 estimators PER BLOCK and
     Palmgren-Miner SUM the block damages duration-weighted (the piecewise-
     stationary "mission profile" damage) — equivalently the amplitude-modulated
     E[a^m]-weighted stationary damage;
  6. the RMS modulation induces a marginal KURTOSIS gamma_4 = 3 E[a^4]/E[a^2]^2
     (the Wolfsteiner-Trapp / Kihm-Rizzi non-stationary -> kurtosis link), and the
     M25<->M24 BRIDGE reports the non-stationary amplification kappa_ns NEXT TO
     the M24 non-Gaussian correction lambda_ng at that kurtosis (the two siblings
     agreeing);
  7. a seeded NON-STATIONARY Monte-Carlo rainflow cross-check: the Gaussian
     history synthesised from the PSD MULTIPLIED by a time-varying RMS envelope
     (the mission profile), ASTM E1049 rainflow, Miner sum — the time-domain
     answer the block / modulated spectral estimate approximates.

Mission profile: run-up from RMS x 0.4 to x 1.5, dwell at x 1.5, run-down to x 0.3
(sampled into NSEG = 12 blocks). S-N curve: N = C*S^-m with m = 5 and C = 5e12
(identical to the M20 example).

What to look for in the listing (NONSTATIONARY_FATIGUE_0001.out):
  * the "** RANDOM-VIBRATION (SPECTRAL) FATIGUE ** (/IMPL/FATIG)" block is the M20
    STATIONARY answer, UNCHANGED — the critical channel springs#1:N, the RMS
    stress, the rates, and the four estimators' damage / life (this is the life if
    the qualification level ran forever at the NOMINAL RMS);
  * the "** NON-STATIONARY / EVOLUTIONARY-PSD FATIGUE ** (/IMPL/FATIG/NSTAT)"
    block reports the mission profile (12 blocks, RMS scale range 0.3 .. 1.5), the
    INDUCED KURTOSIS (> 3, the varying RMS is leptokurtic), the M25<->M24 bridge
    (kappa_ns vs lambda_ng), and for each estimator the STATIONARY rate NEXT TO
    the non-stationary rate and life, plus the BLOCK MINER-SUM and the
    NON-STATIONARY Monte-Carlo cross-check;
  * the non-stationary life reflects the ACTUAL mission energy: here the dwell at
    x 1.5 (RMS scaled UP) dominates the damage (damage ~ RMS^m = 1.5^5 ~ 7.6x the
    nominal), so the mission life is SHORTER than the nominal stationary life even
    though the run-up / run-down blocks are gentler.

The full result (stationary summary + the ``nonstationary`` sub-entry with the
block breakdown, the amplitude-modulated damage, the induced kurtosis, the bridge
and the Monte-Carlo) is on ``model.implicit_result.fatigue``.

Run it (from this directory):
    python generate_deck.py          # regenerates the two .rad files
    pyradioss-starter -i NONSTATIONARY_FATIGUE_0000.rad
    pyradioss-engine  -i NONSTATIONARY_FATIGUE_0001.rad
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
SEED = 20250         # Monte-Carlo random seed (reproducible)
NSEG = 12            # blocks the mission profile is sampled into

here = os.path.dirname(os.path.abspath(__file__))


def main():
    lines = ["#RADIOSS STARTER", "/BEGIN",
             "NONSTATIONARY_FATIGUE - base-excited instrument stack, mission "
             "profile (pyradioss /IMPL/FATIG/NSTAT)"]

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

    # ---- /FUNCT/20: the RMS MISSION PROFILE (scale vs time) ---------------
    # run-up (0.4 -> 1.5), dwell (1.5), run-down (1.5 -> 0.3); time in ms
    lines += ["/FUNCT/20", "RMS mission profile (run-up / dwell / run-down)",
              "       0.0       0.4",
              "     150.0       1.5",
              "     350.0       1.5",
              "     500.0       0.3"]
    lines.append("/END")

    with open(os.path.join(here, "NONSTATIONARY_FATIGUE_0000.rad"), "w") as fh:
        fh.write("\n".join(lines) + "\n")

    # ---- engine: a bare implicit static step + the M25 non-stationary fatigue
    engine = [
        "#RADIOSS ENGINE",
        "/RUN/NONSTATIONARY_FATIGUE/1",
        "1.0",                       # one (zero-load) static increment
        "/IMPL",                     # implicit static driver hosts the analysis
        "/IMPL/FATIG/NSTAT/BASE",    # M25: non-stationary random base-accel fatigue
        # line 1 (PSD sweep): fmin fmax nf funct dir nmode  (funct 10 = S_aa)
        f"{FMIN}  {FMAX}  {NF}  10  0  {NMODE}",
        # line 2 (S-N + Monte-Carlo): m C zeta mean ult mcdur seed
        f"{SN_M}  {SN_C}  {ZETA}  0.0  0.0  {MCDUR}  {SEED}",
        # line 3 (M25 non-stationary): modulation /FUNCT id + block count
        f"20  {NSEG}",
        "/PRINT/-500",
        "/STOP",
        "15.0",
    ]
    with open(os.path.join(here, "NONSTATIONARY_FATIGUE_0001.rad"), "w") as fh:
        fh.write("\n".join(engine) + "\n")
    print("wrote NONSTATIONARY_FATIGUE_0000.rad / NONSTATIONARY_FATIGUE_0001.rad")


if __name__ == "__main__":
    main()
