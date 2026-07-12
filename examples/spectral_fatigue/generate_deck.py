#!/usr/bin/env python3
"""
Generate the SPECTRAL_FATIGUE example deck: the RANDOM-VIBRATION (SPECTRAL)
FATIGUE life of a base-excited multi-mass "instrument stack" — the first
pyradioss stress-life fatigue analysis (M20, /IMPL/FATIG), built on the M19
stress-PSD spectral moments (random_response.py stress modes) and the
narrow-band / Dirlik / Wirsching-Light / Tovo-Benasciutti damage estimators
(spectral_fatigue.py).

This is deliberately the SAME base-driven spring stack the M19 random_vibration
example shakes (a shaker-table / launch-qualification model — Newland, "An
Introduction to Random Vibrations…"; Bishop & Sherratt, "Finite Element Based
Fatigue Calculations", NAFEMS): a random support-acceleration PSD drives the
structure, and M20 asks the fatigue question M19 set up — how long does the
CRITICAL element last?

Model (units mm / ms / kg -> forces kN, frequencies kHz):
  * N = 5 masses: node 1 is the shaken BASE (pinned — the support), nodes 2..6
    the stacked masses along +x, one /SPRING per storey (its own /PART +
    /PROP/SPRING 'M K C'); the spring carries an axial FORCE — its
    stress-RESULTANT fatigue channel;
  * each storey: mass M = 2e-3 kg (half lumps onto each node), stiffness
    K = 800 kN/mm, light modal damping added at analysis time (zeta = 3 %);
  * /FUNCT/10 — a FLAT band-limited input PSD S_aa(f) (a white base-
    acceleration over the analysis band), for /IMPL/FATIG/BASE.

What /IMPL/FATIG/BASE does (a PORT card — the open-source engine has NO
frequency-domain / spectral-fatigue solver; freimpl.F reads only DYNA / BUCKL /
DT / NONLIN / ARCL, and its sole "PSD" token is IMUMPSD, a MUMPS flag):
  1. extract the modes and build the base-excitation FRF H(Omega) (the M16
     participation feed — the M19 machinery);
  2. recover the per-mode STRESS modes sigma_i = (element stress operator) .
     phi_i by running the SAME force kernels the static solve uses on each mode
     shape (read-only), and form the stress FRF H_sigma(Omega) = sum_i sigma_i
     q_i(Omega);
  3. form the stress response PSD S_sigmasigma(Omega) = |H_sigma|^2 S_aa and its
     spectral moments m0..m4 per stress channel;
  4. pick the CRITICAL channel (highest Dirlik damage) and evaluate the S-N
     curve N = C*S^-m under a Miner sum by the narrow-band (Bendat 1964),
     Dirlik (1985), Wirsching-Light (1980) and Tovo-Benasciutti (2005)
     estimators — reporting the damage rate, equivalent stress and life;
  5. run a seeded Monte-Carlo rainflow cross-check (synthesise a Gaussian
     history from the PSD, ASTM E1049 rainflow, Miner sum).

S-N curve: N = C*S^-m with m = 5 (a typical welded/steel slope) and
C = 5e12 (in the spring's force units, kN, over the ms time base) — chosen so
the critical base spring shows a finite, physically-legible life under this PSD.

What to look for in the listing (SPECTRAL_FATIGUE_0001.out), the
"** RANDOM-VIBRATION (SPECTRAL) FATIGUE ** (/IMPL/FATIG)" block:
  * the CRITICAL STRESS CHANNEL is springs#1:N — the BASE spring, which carries
    the whole stack's inertia and so sees the largest random force;
  * the RMS stress (sigma = sqrt m0), the crossing / peak rates nu0 / nup and
    the irregularity factor alpha2 (< 1 -> the response is somewhat wide-band,
    two modes participating);
  * the damage rate / life / equivalent-stress TABLE for all four estimators:
    narrow band is the most conservative (shortest life), Dirlik relaxes it
    (the wide-band correction), and Wirsching-Light / Tovo-Benasciutti bracket
    Dirlik. The MONTE-CARLO rainflow row agrees with Dirlik to within the
    documented scatter — the time-domain answer the spectral methods
    approximate.

The full fatigue result (channels, moments, per-method damage/life/S_eq, the
Monte-Carlo cross-check) is on ``model.implicit_result.fatigue``.

Run it (from this directory):
    python generate_deck.py          # regenerates the two .rad files
    pyradioss-starter -i SPECTRAL_FATIGUE_0000.rad
    pyradioss-engine  -i SPECTRAL_FATIGUE_0001.rad
"""

import os

NMASS = 5            # stacked masses
DX = 10.0            # node spacing (mm)
MASS = 2.0e-3        # per-storey mass (kg) — half lumps onto each node
K = 800.0            # per-storey stiffness (kN/mm)
ZETA = 0.03          # modal damping (3 %) for the FRF
NMODE = 5            # modes to extract
FMIN = 0.0           # analysis band start (1/ms)
FMAX = 250.0         # analysis band end (1/ms) — past the highest mode (~199)
NF = 3000            # PSD sweep points
S0 = 1.0e2          # strong flat input base-acceleration PSD ((mm/ms^2)^2/kHz)
SN_M = 5.0           # S-N slope m (N = C*S^-m)
SN_C = 5.0e12        # S-N coefficient C (in the spring-force units)
MCDUR = 500.0        # Monte-Carlo cross-check record length (ms)
SEED = 20200          # Monte-Carlo random seed (reproducible)

here = os.path.dirname(os.path.abspath(__file__))


def main():
    lines = ["#RADIOSS STARTER", "/BEGIN",
             "SPECTRAL_FATIGUE - base-excited instrument stack "
             "(pyradioss /IMPL/FATIG)"]

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

    with open(os.path.join(here, "SPECTRAL_FATIGUE_0000.rad"), "w") as fh:
        fh.write("\n".join(lines) + "\n")

    # ---- engine: a bare implicit static step + the M20 fatigue analysis ---
    engine = [
        "#RADIOSS ENGINE",
        "/RUN/SPECTRAL_FATIGUE/1",
        "1.0",                       # one (zero-load) static increment
        "/IMPL",                     # implicit static driver hosts the analysis
        "/IMPL/FATIG/BASE",          # M20: random base-acceleration fatigue
        # line 1 (PSD sweep): fmin fmax nf funct dir nmode  (funct 10 = S_aa)
        f"{FMIN}  {FMAX}  {NF}  10  0  {NMODE}",
        # line 2 (S-N + Monte-Carlo): m C zeta mean ult mcdur seed
        f"{SN_M}  {SN_C}  {ZETA}  0.0  0.0  {MCDUR}  {SEED}",
        "/PRINT/-500",
        "/STOP",
        "15.0",
    ]
    with open(os.path.join(here, "SPECTRAL_FATIGUE_0001.rad"), "w") as fh:
        fh.write("\n".join(engine) + "\n")
    print("wrote SPECTRAL_FATIGUE_0000.rad / SPECTRAL_FATIGUE_0001.rad")


if __name__ == "__main__":
    main()
