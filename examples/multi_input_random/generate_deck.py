#!/usr/bin/env python3
"""
Generate the MULTI_INPUT_RANDOM example deck: the MULTI-INPUT / PARTIALLY-
COHERENT random-vibration fatigue life of the M21 base-clamped SOLID-brick
cantilever, now driven by TWO simultaneous random input processes instead of
one (M28, /IMPL/FATIG/MULT/MINPUT), built on the multi-input cross-PSD library
of implicit/multi_input_response.py + multi_input_fatigue.py.

Where the M21 multiaxial_fatigue example drives the cantilever with ONE random
tip-force pattern (a skew z+y load driven by a single PSD), M28 splits the
excitation into TWO independent input PATTERNS — a transverse z-load (input 1)
and a lateral y-load (input 2) — each with its OWN random auto-PSD, coupled by
a COHERENCE gamma between them. The response (and the root stress-tensor)
cross-PSD is the MIMO matrix triple product S_sigmasigma = H_sigma S_ff
H_sigma^H, with S_ff the 2x2 Hermitian input cross-spectral matrix (auto-PSDs
on the diagonal, sqrt(G1 G2) gamma e^{i theta} off-diagonal). This is the
classic multiple-correlated-input random-vibration setting (Newland ch. 6-8;
Bendat & Piersol ch. 5-7 — the cross-spectral / coherence matrix and the MIMO
relation; Wirsching-Paez-Ortiz — multi-input random fatigue).

Two limiting cases the library reproduces exactly (see tests/test_m28_*):
  * fully INCOHERENT inputs (gamma = 0) -> the response PSD is the SUM of the
    two single-input answers;
  * fully COHERENT inputs (gamma = 1, common source) -> the single-input answer
    for the combined pattern (the M21 rank-1 special case, bit-identical).
This deck uses PARTIAL coherence (gamma = 0.5), interpolating between them.

Model (units mm / ms / kg -> forces kN, stresses GPa, frequencies kHz),
identical to the M21 solid brick so the two lives are directly comparable:
  * a straight cantilever of NX /BRICK hexa8 solids along +x, 1x1 mm square
    cross-section, clamped at the x = 0 face;
  * steel LAW1 (E = 210 GPa, nu = 0.3, rho = 7.8e-6 kg/mm^3), 3% modal damping;
  * INPUT 1: a transverse z tip-force pattern (/CLOAD /FUNCT/1) with a flat
    auto-PSD G1 = 2.0 (/FUNCT/10) -> excites the strong bending plane;
  * INPUT 2: a lateral y tip-force pattern (/CLOAD /FUNCT/2) with a flat
    auto-PSD G2 = 1.5 (/FUNCT/11) -> excites the weak bending plane;
  * a COHERENCE gamma = 0.5 (phase 0) between the two inputs.

What /IMPL/FATIG/MULT/MINPUT does (a PORT sub-flag — the open-source engine has
NO frequency-domain / spectral / MULTI-INPUT / coherence solver of any kind;
freimpl.F reads only DYNA / BUCKL / DT / NONLIN / ARCL and its sole "PSD" token
is IMUMPSD, a MUMPS flag):
  1. runs the SINGLE-input /MULT analysis first (the deck's combined /CLOAD
     pattern under /FUNCT/10) — the M21 answer, stored as before;
  2. assembles the 2x2 Hermitian input cross-PSD S_ff(f) from the two auto-PSDs
     and the coherence model (projected onto the nearest Hermitian PSD matrix);
  3. builds a per-input FRF COLUMN for each input load pattern and forms the
     multi-input stress-tensor cross-PSD S_sigmasigma = H_sigma S_ff H_sigma^H
     on the critical root element;
  4. reduces it by the SAME M21 machinery (equivalent von Mises / max-normal /
     max-shear critical plane) and runs the multi-input Monte-Carlo (an
     independent input-level synthesis of the two correlated inputs + the
     stress-level M21 cross-check);
  5. reports the multi-input damage / life ALONGSIDE the single-input numbers
     (a "multi_input" sub-entry on model.implicit_result.fatigue).

S-N curve: N = C*S^-m with m = 5, C = 1e4 (stress units GPa / ms base), as in
the M21 example, so the two lives are on the same scale.

What to look for in the listing (MULTI_INPUT_RANDOM_0001.out):
  * the "** MULTIAXIAL / CRITICAL-PLANE SPECTRAL FATIGUE ** (/IMPL/FATIG/MULT)"
    block — the SINGLE-input answer (the M21 baseline);
  * the "** MULTI-INPUT / PARTIALLY-COHERENT SPECTRAL FATIGUE **
    (/IMPL/FATIG/MINPUT)" block — the number of inputs, the coherence model, the
    input cross-PSD PSD-projection status (here already valid), the critical
    element, the von-Mises / max-normal / max-shear multi-input damage and the
    two Monte-Carlo estimates (stress-level / input-level, which agree within
    scatter), plus the single-input critical Dirlik damage for the side-by-side
    comparison.

The multi-input result (the 2x2 input cross-PSD, the multi-input S_sigmasigma,
the three reductions, both Monte-Carlo estimates and the coherence diagnostics)
is on model.implicit_result.fatigue["multi_input"].

Run it (from this directory):
    python generate_deck.py          # regenerates the two .rad files
    pyradioss-starter -i MULTI_INPUT_RANDOM_0000.rad
    pyradioss-engine  -i MULTI_INPUT_RANDOM_0001.rad
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
GAMMA = 0.5          # coherence between the two inputs (0..1)
PHASE = 0.0          # phase between the two inputs (degrees)
FSCALE = 1.0e-4      # tip-force pattern amplitude (kN)
SN_M = 5.0           # S-N slope m (N = C*S^-m)
SN_C = 1.0e4         # S-N coefficient C (stress units GPa)
MCDUR = 400.0        # Monte-Carlo cross-check record length (ms)
SEED = 21000         # Monte-Carlo random seed (reproducible)

here = os.path.dirname(os.path.abspath(__file__))


def node_id(ix, iy, iz):
    """Structured-grid node id: (NX+1) along x, 2 along y, 2 along z."""
    return ix * 4 + iy * 2 + iz + 1


def main():
    lines = ["#RADIOSS STARTER", "/BEGIN",
             "MULTI_INPUT_RANDOM - solid-brick cantilever, two partially-"
             "coherent inputs (pyradioss /IMPL/FATIG/MULT/MINPUT)"]

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
    # UNIT pattern curves for the two input load PATTERNS (their /CLOAD spatial
    # patterns reference these; the frequency content is the auto-PSD). The
    # multi-input table names each input by its /CLOAD /FUNCT id.
    lines += ["/FUNCT/1", "input-1 pattern (z)",
              "       0.0           1.0", "  100000.0           1.0"]
    lines += ["/FUNCT/2", "input-2 pattern (y)",
              "       0.0           1.0", "  100000.0           1.0"]
    # the two auto-PSDs G1(f), G2(f) (flat, band-limited)
    lines += ["/FUNCT/10", "input-1 auto-PSD G1(f)",
              f"       0.0{G1:14g}", f"  100000.0{G1:14g}"]
    lines += ["/FUNCT/11", "input-2 auto-PSD G2(f)",
              f"       0.0{G2:14g}", f"  100000.0{G2:14g}"]

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
        lines, os.path.join(here, "MULTI_INPUT_RANDOM_0000.rad"), runname="MULTI_INPUT_RANDOM")

    # ---- engine: a bare implicit static step + the M28 multi-input analysis -
    engine = [
        "#RADIOSS ENGINE",
        "/RUN/MULTI_INPUT_RANDOM/1",
        "1.0",                       # one (zero-load) static increment
        "/IMPL",                     # implicit static driver hosts the analysis
        "/IMPL/FATIG/MULT/MINPUT",   # M28: multi-input / partially-coherent
        # line 1 (PSD sweep): fmin fmax nf funct nmode (funct 10 = ref PSD)
        f"{FMIN}  {FMAX}  {NF}  10  {NMODE}",
        # line 2 (S-N + Monte-Carlo): m C zeta mean ult mcdur seed
        f"{SN_M}  {SN_C}  {ZETA}  0.0  0.0  {MCDUR}  {SEED}",
        # line 3 (MULTI-INPUT header): ninput cohmodel gamma phase
        f"2  0  {GAMMA}  {PHASE}",
        # input rows: cload_funct psd_funct
        "1  10",                     # input 1: /CLOAD /FUNCT/1, auto-PSD /FUNCT/10
        "2  11",                     # input 2: /CLOAD /FUNCT/2, auto-PSD /FUNCT/11
        "/PRINT/-500",
        "/STOP",
        "15.0",
    ]
    deck_writer.write_engine_from_port_lines(
        engine, os.path.join(here, "MULTI_INPUT_RANDOM_0001.rad"))
    print("wrote MULTI_INPUT_RANDOM_0000.rad / MULTI_INPUT_RANDOM_0001.rad")


if __name__ == "__main__":
    main()
