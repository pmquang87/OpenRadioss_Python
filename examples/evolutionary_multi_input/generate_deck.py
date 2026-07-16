#!/usr/bin/env python3
"""
Generate the EVOLUTIONARY_MULTI_INPUT example deck: the FULLY NON-STATIONARY /
EVOLUTIONARY MULTI-INPUT random-vibration fatigue life of the M28 two-input
SOLID-brick cantilever, now driven by a TIME-VARYING input coherence matrix
S_ff(w, t) (M29, /IMPL/FATIG/MULT/MINPUT/EVOL), built on the evolutionary
multi-input library of implicit/evolutionary_multi_input.py (the convergence of
the M27 evolutionary joint-tensor path and the M28 multi-input path).

Where the M28 multi_input_random example drives the cantilever with TWO random
input patterns coupled by a FIXED coherence gamma = 0.5, M29 lets the input
COHERENCE itself DRIFT with time: the two inputs start nearly INCOHERENT
(gamma = 0.1) and evolve to nearly COHERENT (gamma = 0.9) across the mission,
while a swept-centre / broadening Gaussian window sweeps the excitation band
(fc 60 -> 180 kHz). Each mission window i carries its OWN Hermitian input
cross-PSD S_ff(w, t_i); the per-window multi-input stress-tensor cross-PSD
S_sigmasigma,i = H_sigma S_ff(t_i) H_sigma^H drives a per-window critical-plane
search — the critical plane / F_np RE-SEARCHED per window as the coherence
evolves, so the plane genuinely ROTATES between the M28 incoherent-SUM limit
(gamma = 0, the sum of the two single-input answers) and the coherent-combination
limit (gamma = 1, the combined pattern). The per-window multiaxial damages are
Palmgren-Miner-summed (Priestley evolutionary spectra — the matrix / coherence-
valued case; Newland ch. 6-8 + Bendat & Piersol ch. 5-7 — the time-varying
coherence matrix; the M27 joint-tensor + M28 multi-input base).

Exact reductions the library reproduces (see tests/test_m29_*):
  * a CONSTANT coherence / SINGLE flat window -> the M28 STATIONARY multi-input
    answer EXACTLY (bit-identical delegation);
  * a SINGLE input -> the M27 scalar/tensor EVOLUTIONARY answer EXACTLY
    (bit-identical delegation);
  * a drifting incoherent -> coherent schedule whose per-window response variance
    and critical plane genuinely DRIFT between the M28 incoherent-SUM and coherent-
    combination limits (the whole point of M29).

Model (units mm / ms / kg -> forces kN, stresses GPa, frequencies kHz),
identical to the M28 two-input brick so the stationary and evolutionary lives are
directly comparable:
  * a straight cantilever of NX /BRICK hexa8 solids along +x, 1x1 mm square
    cross-section, clamped at the x = 0 face;
  * steel LAW1 (E = 210 GPa, nu = 0.3, rho = 7.8e-6 kg/mm^3), 3% modal damping;
  * INPUT 1: a transverse z tip-force pattern (/CLOAD /FUNCT/1) with auto-PSD
    G1 = 2.0 (/FUNCT/10) -> the strong bending plane;
  * INPUT 2: a lateral y tip-force pattern (/CLOAD /FUNCT/2) with auto-PSD
    G2 = 1.5 (/FUNCT/11) -> the weak bending plane;
  * a DRIFTING coherence gamma(t): 0.1 -> 0.9 across NWIN mission windows;
  * a swept-centre Gaussian window fc 60 -> 180 kHz (the /EVOL drifting shape);
  * a mission RMS-level profile /FUNCT/30 (the shared /NSTAT modulation).

What /IMPL/FATIG/MULT/MINPUT/EVOL does (a PORT sub-flag — the open-source engine
has NO frequency-domain / spectral / multi-input / coherence / evolutionary
solver of any kind; freimpl.F reads only DYNA / BUCKL / DT / NONLIN / ARCL and
its sole "PSD" token is IMUMPSD, a MUMPS flag):
  1. runs the SINGLE-input /MULT analysis (the M21 baseline) and the M28
     STATIONARY multi-input analysis (gamma held at the start value) FIRST — both
     stored byte-identical (a NEW parallel path);
  2. builds the per-window coherence schedule (gamma 0.1 -> 0.9), assembles the
     per-window Hermitian input cross-PSD S_ff(w, t_i) (each projected onto the
     nearest Hermitian PSD matrix);
  3. forms the per-window multi-input stress-tensor cross-PSD S_sigmasigma,i on
     the critical root element and RE-SEARCHES the critical plane / F_np per
     window (the plane may ROTATE as the coherence drifts);
  4. Palmgren-Miner-sums the per-window multiaxial damages, and runs the
     non-stationary MULTI-INPUT multivariate Monte-Carlo cross-check (per-window
     blocks of the correlated-input synthesiser, per-window plane projection,
     ASTM E1049 rainflow + Miner), measuring the synthesised inputs' per-window
     coherence to confirm it tracks the drifting target;
  5. reports the coherence DRIFT / per-window RMS / critical-plane rotation /
     damage-life ALONGSIDE the M28 stationary multi-input and the M27 single-input
     evolutionary numbers (an "evolutionary_multi_input" sub-entry on
     model.implicit_result.fatigue["multi_input"]).

S-N curve: N = C*S^-m with m = 5, C = 1e4 (stress units GPa / ms base), as in the
M21 / M28 examples, so all the lives are on the same scale.

What to look for in the listing (EVOLUTIONARY_MULTI_INPUT_0001.out):
  * the "** MULTI-INPUT / PARTIALLY-COHERENT SPECTRAL FATIGUE **
    (/IMPL/FATIG/MINPUT)" block — the M28 STATIONARY multi-input answer;
  * the "** FULLY EVOLUTIONARY MULTI-INPUT FATIGUE **
    (/IMPL/FATIG/MULT/MINPUT/EVOL)" block — the coherence schedule (gamma
    0.1 -> 0.9), the critical-plane ROTATION driven by the evolving coherence, the
    per-reduction evolutionary vs stationary damage rates, the non-stationary
    MULTI-INPUT Monte-Carlo and the MEASURED per-window coherence tracking the
    target.

Run it (from this directory):
    python generate_deck.py          # regenerates the two .rad files
    pyradioss-starter -i EVOLUTIONARY_MULTI_INPUT_0000.rad
    pyradioss-engine  -i EVOLUTIONARY_MULTI_INPUT_0001.rad
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


NX = 6               # bricks along the cantilever axis (x)
H = 1.0              # cross-section side (mm) — 1x1 square
E = 210.0            # Young's modulus (GPa)
NU = 0.3             # Poisson ratio
RHO = 7.8e-6         # density (kg/mm^3)
ZETA = 0.03          # modal damping (3 %) for the FRF
NMODE = 8            # modes to extract
FMIN = 0.0           # analysis band start (kHz)
FMAX = 400.0         # analysis band end (kHz)
NF = 1500            # PSD sweep points
G1 = 2.0             # input-1 (z-load) flat auto-PSD (kN^2/kHz)
G2 = 1.5             # input-2 (y-load) flat auto-PSD (kN^2/kHz)
GAMMA0 = 0.1         # START coherence between the two inputs (nearly incoherent)
GAMMA1 = 0.9         # END coherence between the two inputs (nearly coherent)
PHASE0 = 0.0         # START phase (degrees)
PHASE1 = 0.0         # END phase (degrees)
FC0 = 60.0           # window centre freq at the FIRST window (kHz)
FC1 = 180.0          # window centre freq at the LAST window (kHz)
BW = 80.0            # window bandwidth (kHz), constant (broad enough to keep energy)
NWIN = 6             # mission windows the coherence / shape drift is sampled into
FSCALE = 1.0e-4      # tip-force pattern amplitude (kN)
SN_M = 5.0           # S-N slope m (N = C*S^-m)
SN_C = 1.0e4         # S-N coefficient C (stress units GPa)
MCDUR = 400.0        # Monte-Carlo cross-check record length (ms)
SEED = 29000         # Monte-Carlo random seed (reproducible)

here = os.path.dirname(os.path.abspath(__file__))


def node_id(ix, iy, iz):
    """Structured-grid node id: (NX+1) along x, 2 along y, 2 along z."""
    return ix * 4 + iy * 2 + iz + 1


def main():
    lines = ["#RADIOSS STARTER", "/BEGIN",
             "EVOLUTIONARY_MULTI_INPUT - solid-brick cantilever, TWO inputs with a "
             "DRIFTING coherence (pyradioss /IMPL/FATIG/MULT/MINPUT/EVOL)"]

    # ---- nodes: (NX+1) x 2 x 2 structured grid -----------------------------
    lines.append("/NODE")
    for ix in range(NX + 1):
        for iy in range(2):
            for iz in range(2):
                nid = node_id(ix, iy, iz)
                lines.append(f"{nid:10d}{ix * H:20.10f}{iy * H:20.10f}"
                             f"{iz * H:20.10f}")

    # ---- bricks: one hexa8 per axial slot ----------------------------------
    lines.append("/BRICK/1")
    for ix in range(NX):
        n = [node_id(ix, 0, 0), node_id(ix + 1, 0, 0),
             node_id(ix + 1, 1, 0), node_id(ix, 1, 0),
             node_id(ix, 0, 1), node_id(ix + 1, 0, 1),
             node_id(ix + 1, 1, 1), node_id(ix, 1, 1)]
        lines.append(f"{ix + 1:10d}" + "".join(f"{v:10d}" for v in n))

    lines += ["/PART/1", "cantilever", "         1         1",
              "/MAT/LAW1/1", "steel elastic", f"   {RHO}",
              f"     {E}       {NU}",
              "/PROP/SOLID/1", "solid", "         0"]

    # ---- constraints: clamp the x = 0 root face (4 nodes) ------------------
    root = [node_id(0, iy, iz) for iy in range(2) for iz in range(2)]
    tip = [node_id(NX, iy, iz) for iy in range(2) for iz in range(2)]
    root_ids = "\n".join(str(i) for i in root)
    lines += ["/GRNOD/NODE/1", "root", root_ids,
              "/BCS/1", "clamped root",
              "       111       111         0         1"]

    # ---- functions ---------------------------------------------------------
    # UNIT pattern curves for the two input load PATTERNS
    lines += ["/FUNCT/1", "input-1 pattern (z)",
              "       0.0           1.0", "  100000.0           1.0"]
    lines += ["/FUNCT/2", "input-2 pattern (y)",
              "       0.0           1.0", "  100000.0           1.0"]
    # the two auto-PSDs G1(f), G2(f) (flat, band-limited)
    lines += ["/FUNCT/10", "input-1 auto-PSD G1(f)",
              f"       0.0{G1:14g}", f"  100000.0{G1:14g}"]
    lines += ["/FUNCT/11", "input-2 auto-PSD G2(f)",
              f"       0.0{G2:14g}", f"  100000.0{G2:14g}"]
    # the mission RMS-level modulation /FUNCT/30 (the shared /NSTAT profile — a
    # ramp-up then ramp-down mission envelope over 0..300 ms)
    lines += ["/FUNCT/30", "mission RMS profile a(t)",
              "       0.0           0.4",
              "     100.0           1.0",
              "     200.0           1.2",
              "     300.0           0.6"]

    # ---- two tip-force patterns: input 1 = z-load, input 2 = y-load --------
    per = FSCALE / len(tip)
    for k, nid in enumerate(tip):
        lines += [f"/GRNOD/NODE/{10 + k}", f"tipn{k}", str(nid),
                  f"/CLOAD/{20 + 2 * k}", "z-load (input 1)",
                  f"         1         Z{10 + k:10d}{per * 1.0:12g}",
                  f"/CLOAD/{21 + 2 * k}", "y-load (input 2)",
                  f"         2         Y{10 + k:10d}{per * 0.7:12g}"]
    lines.append("/END")

    deck_writer.write_starter_from_port_lines(
        lines, os.path.join(here, "EVOLUTIONARY_MULTI_INPUT_0000.rad"), runname="EVOLUTIONARY_MULTI_INPUT")

    # ---- engine: a bare implicit static step + the M29 evolutionary analysis
    engine = [
        "#RADIOSS ENGINE",
        "/RUN/EVOLUTIONARY_MULTI_INPUT/1",
        "1.0",                              # one (zero-load) static increment
        "/IMPL",                            # implicit static driver hosts it
        # M29: multi-input (/MINPUT) + evolutionary coherence (/EVOL). /NSTAT reads
        # the mission RMS profile /FUNCT/30 so the level drifts too.
        "/IMPL/FATIG/MULT/MINPUT/EVOL/NSTAT",
        # line 1 (PSD sweep): fmin fmax nf funct nmode (funct 10 = ref PSD)
        f"{FMIN}  {FMAX}  {NF}  10  {NMODE}",
        # line 2 (S-N + Monte-Carlo): m C zeta mean ult mcdur seed
        f"{SN_M}  {SN_C}  {ZETA}  0.0  0.0  {MCDUR}  {SEED}",
        # line 3 (/NSTAT modulation): modfunct — the mission RMS profile
        "30",
        # line 4 (/EVOL drifting shape): fc0 fc1 bw0 bw1 nwin
        f"{FC0}  {FC1}  {BW}  {BW}  {NWIN}",
        # line 5 (MULTI-INPUT header): ninput cohmodel gamma phase decay speed
        #                              gamma1 phase1  (the coherence END pair)
        f"2  0  {GAMMA0}  {PHASE0}  0.0  0.0  {GAMMA1}  {PHASE1}",
        # input rows: cload_funct psd_funct
        "1  10",                            # input 1: /CLOAD /FUNCT/1, auto-PSD /FUNCT/10
        "2  11",                            # input 2: /CLOAD /FUNCT/2, auto-PSD /FUNCT/11
        "/PRINT/-500",
        "/STOP",
        "15.0",
    ]
    deck_writer.write_engine_from_port_lines(
        engine, os.path.join(here, "EVOLUTIONARY_MULTI_INPUT_0001.rad"))
    print("wrote EVOLUTIONARY_MULTI_INPUT_0000.rad / "
          "EVOLUTIONARY_MULTI_INPUT_0001.rad")


if __name__ == "__main__":
    main()
