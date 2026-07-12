#!/usr/bin/env python3
"""
Generate the NONGAUSSIAN_FATIGUE example deck: the NON-GAUSSIAN / KURTOSIS-
corrected fatigue life of the SAME base-excited multi-mass "instrument stack" the
M20 spectral_fatigue example shakes (M24, /IMPL/FATIG/NGAUSS), built on the M20
Gaussian spectral estimators (spectral_fatigue.py) and the Winterstein Hermite-
moment model + the Benasciutti-Braccesi / Rizzi-Kihm closed-form correction
factor lambda_ng (nongaussian_fatigue.py).

This is deliberately the SAME base-driven spring stack the M20 spectral_fatigue
example shakes (a shaker-table / launch-qualification model — Newland, "An
Introduction to Random Vibrations…"; Kihm & Rizzi, "Understanding how kurtosis is
transferred from input acceleration to stress response"): a random support-
acceleration PSD drives the structure, but now the input is declared NON-GAUSSIAN
(spiky, leptokurtic — the real signature of a rough transport / re-entry / gunfire
environment, which the Gaussian |H|^2 S law and Dirlik's Rayleigh-range PDF miss).
M24 asks: how much SHORTER is the critical element's life once the kurtosis is
accounted for?

Model (units mm / ms / kg -> forces kN, frequencies kHz): IDENTICAL to the M20
spectral_fatigue example —
  * N = 5 masses: node 1 the shaken BASE (pinned), nodes 2..6 the stacked masses
    along +x, one /SPRING per storey (its own /PART + /PROP/SPRING 'M K C'); the
    spring carries an axial FORCE — its stress-RESULTANT fatigue channel;
  * each storey: mass M = 2e-3 kg, stiffness K = 800 kN/mm, modal damping 3 %;
  * /FUNCT/10 — a FLAT band-limited input PSD S_aa(f) for /IMPL/FATIG/NGAUSS/BASE.

What /IMPL/FATIG/NGAUSS/BASE does (a PORT card — the open-source engine has NO
frequency-domain / spectral / non-Gaussian fatigue solver; freimpl.F reads only
DYNA / BUCKL / DT / NONLIN / ARCL, and its sole "PSD" token is IMUMPSD, a MUMPS
flag):
  1..4. everything the M20 /IMPL/FATIG/BASE Gaussian path does — the base-
     excitation FRF, the recovered STRESS modes, the stress response PSD
     S_sigmasigma = |H_sigma|^2 S_aa and its moments, the CRITICAL channel, and
     the four Gaussian estimators (narrow-band / Dirlik / Wirsching-Light /
     Tovo-Benasciutti) — reported UNCHANGED (byte-identical whether or not NGAUSS
     runs);
  5. THEN, alongside, the NON-GAUSSIAN correction: from the target kurtosis
     gamma_4 (and skewness gamma_3) it builds the Winterstein Hermite transform
     g(u) = kappa[u + h_3(u^2-1) + h_4(u^3-3u)] (which preserves the mean/variance
     and hits the target kurtosis), integrates the closed-form correction
     lambda_ng = E[g(V)^m]/E[V^m] over the Rayleigh amplitude V (attenuated for
     bandwidth, Benasciutti-Tovo), and scales EVERY Gaussian damage rate by
     lambda_ng — a shorter life for a spiky (leptokurtic) input;
  6. a seeded NON-GAUSSIAN Monte-Carlo rainflow cross-check: the Gaussian history
     synthesised from the PSD pushed through the same Hermite transform to the
     target kurtosis, ASTM E1049 rainflow, Miner sum — the time-domain answer the
     lambda_ng-corrected spectral estimate approximates.

Target statistics: kurtosis gamma_4 = 6 (a strongly leptokurtic / spiky input, a
typical high-kurtosis vibration environment) and skewness gamma_3 = 0 (symmetric).
S-N curve: N = C*S^-m with m = 5 and C = 5e12 (identical to the M20 example).

What to look for in the listing (NONGAUSSIAN_FATIGUE_0001.out):
  * the "** RANDOM-VIBRATION (SPECTRAL) FATIGUE ** (/IMPL/FATIG)" block is the
    M20 Gaussian answer, UNCHANGED — the critical channel springs#1:N, the RMS
    stress, the rates, and the four Gaussian estimators' damage / life;
  * the "** NON-GAUSSIAN / KURTOSIS FATIGUE ** (/IMPL/FATIG/NGAUSS)" block reports
    the target kurtosis (6) / skewness (0), the Hermite coefficients (h3 = 0,
    h4 ~ 0.056, kappa ~ 0.99), the correction factor lambda_ng > 1 (the
    leptokurtic amplification, ~2-3x here for m = 5), and for each estimator the
    GAUSSIAN rate NEXT TO the corrected NON-GAUSS rate and the shorter NON-GAUSS
    life — the kurtosis-corrected answer beside the M20 Gaussian one;
  * the NON-GAUSS MONTE-CARLO row reports its sample kurtosis (~ the target) and a
    damage rate consistent with the lambda_ng-corrected spectral estimate.

The full result (Gaussian summary + the ``nongaussian`` sub-entry with lambda_ng,
the Hermite coefficients, the corrected estimators and the Monte-Carlo) is on
``model.implicit_result.fatigue``.

Run it (from this directory):
    python generate_deck.py          # regenerates the two .rad files
    pyradioss-starter -i NONGAUSSIAN_FATIGUE_0000.rad
    pyradioss-engine  -i NONGAUSSIAN_FATIGUE_0001.rad
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
MCDUR = 500.0        # Monte-Carlo cross-check record length (ms)
SEED = 20240         # Monte-Carlo random seed (reproducible)
KURT = 6.0           # target kurtosis gamma_4 (6 = strongly leptokurtic; 3 = Gaussian)
SKEW = 0.0           # target skewness gamma_3 (0 = symmetric)

here = os.path.dirname(os.path.abspath(__file__))


def main():
    lines = ["#RADIOSS STARTER", "/BEGIN",
             "NONGAUSSIAN_FATIGUE - base-excited instrument stack, kurtosis "
             "correction (pyradioss /IMPL/FATIG/NGAUSS)"]

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
    lines.append("/END")

    with open(os.path.join(here, "NONGAUSSIAN_FATIGUE_0000.rad"), "w") as fh:
        fh.write("\n".join(lines) + "\n")

    # ---- engine: a bare implicit static step + the M24 non-Gaussian fatigue
    engine = [
        "#RADIOSS ENGINE",
        "/RUN/NONGAUSSIAN_FATIGUE/1",
        "1.0",                       # one (zero-load) static increment
        "/IMPL",                     # implicit static driver hosts the analysis
        "/IMPL/FATIG/NGAUSS/BASE",   # M24: non-Gaussian random base-accel fatigue
        # line 1 (PSD sweep): fmin fmax nf funct dir nmode  (funct 10 = S_aa)
        f"{FMIN}  {FMAX}  {NF}  10  0  {NMODE}",
        # line 2 (S-N + Monte-Carlo): m C zeta mean ult mcdur seed
        f"{SN_M}  {SN_C}  {ZETA}  0.0  0.0  {MCDUR}  {SEED}",
        # line 3 (M24 non-Gaussian): kurtosis [skewness]
        f"{KURT}  {SKEW}",
        "/PRINT/-500",
        "/STOP",
        "15.0",
    ]
    with open(os.path.join(here, "NONGAUSSIAN_FATIGUE_0001.rad"), "w") as fh:
        fh.write("\n".join(engine) + "\n")
    print("wrote NONGAUSSIAN_FATIGUE_0000.rad / NONGAUSSIAN_FATIGUE_0001.rad")


if __name__ == "__main__":
    main()
