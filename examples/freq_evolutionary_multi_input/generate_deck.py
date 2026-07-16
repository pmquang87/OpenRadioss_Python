#!/usr/bin/env python3
"""
Generate the FREQ_EVOLUTIONARY_MULTI_INPUT example deck: the FREQUENCY-DEPENDENT +
TIME-VARYING (EVOLUTIONARY) INPUT-COHERENCE random-vibration fatigue life of the
M28/M29 two-input SOLID-brick cantilever, now driven by a coherence matrix
gamma_ab(f, t) that varies with BOTH FREQUENCY AND TIME (M30,
/IMPL/FATIG/MULT/MINPUT/EVOL/FCOH), built on the frequency-dependent evolutionary
multi-input library of implicit/freq_evolutionary_multi_input.py (the convergence
of M28's frequency-dependent coherence and M29's time-varying coherence).

Where the M29 evolutionary_multi_input example drifts a per-window SCALAR coherence
gamma(t) (frequency-FLAT, 0.1 -> 0.9) and holds any exponential model stationary,
M30 drives the two inputs with a CONVECTION / TURBULENCE FIELD whose coherence is a
FULL frequency shape gamma_ab(f) = exp(-decay |x_a - x_b| f / speed) (a Davenport /
von Karman convection-coherence field: coherent at low frequency, decorrelating at
high frequency) AND whose DECORRELATION FREQUENCY MOVES UP through the mission — the
turbulence becoming more spatially coherent (the decay coefficient falling
0.9 -> 0.2) as the mission proceeds. Each mission window i carries its OWN Hermitian
input cross-PSD S_ff(w, t_i) with its OWN frequency-shape coherence; the per-window
multi-input stress-tensor cross-PSD S_sigmasigma,i = H_sigma S_ff(t_i) H_sigma^H
drives a per-window critical-plane search — the critical plane / F_np RE-SEARCHED
per window as the coherence FREQUENCY-SHAPE evolves, so the plane genuinely DRIFTS
as the decorrelation frequency moves. The per-window multiaxial damages are
Palmgren-Miner-summed (Priestley evolutionary spectra — the frequency-AND-time
varying coherence matrix; Newland ch. 6-8 + Bendat & Piersol ch. 5-7 — the
frequency-dependent coherence; Davenport / von Karman convection-coherence fields;
the M28 frequency-dependent + M29 time-varying base).

Exact reductions the library reproduces (see tests/test_m30_*):
  * a FREQUENCY-FLAT coherence -> the M29 scalar-coherence answer EXACTLY
    (bit-identical delegation);
  * a STATIONARY (single-window / constant-shape) frequency-dependent coherence ->
    the M28 frequency-dependent answer EXACTLY (bit-identical delegation);
  * a drifting-frequency-shape schedule (the decorrelation moving up in frequency)
    whose per-window response variance and critical plane genuinely DRIFT as the
    frequency-shape of the coherence evolves (the whole point of M30).

Model (units mm / ms / kg -> forces kN, stresses GPa, frequencies kHz), IDENTICAL
to the M28/M29 two-input brick so the stationary, scalar-coherence and
frequency-dependent lives are directly comparable:
  * a straight cantilever of NX /BRICK hexa8 solids along +x, 1x1 mm square
    cross-section, clamped at the x = 0 face;
  * steel LAW1 (E = 210 GPa, nu = 0.3, rho = 7.8e-6 kg/mm^3), 3% modal damping;
  * INPUT 1: a transverse z tip-force pattern (/CLOAD /FUNCT/1) at spatial
    position x = 0 with auto-PSD G1 = 2.0 (/FUNCT/10);
  * INPUT 2: a lateral y tip-force pattern (/CLOAD /FUNCT/2) at spatial
    position x = SEP mm (the input SEPARATION that sets the convection coherence)
    with auto-PSD G2 = 1.5 (/FUNCT/11);
  * a CONVECTION coherence field gamma_ab(f) = exp(-decay |dx| f / speed) whose
    decay coefficient FALLS 0.9 -> 0.2 across NWIN mission windows (the
    decorrelation frequency moving UP through the mission);
  * a swept-centre Gaussian window fc 60 -> 180 kHz (the /EVOL drifting shape);
  * a mission RMS-level profile /FUNCT/30 (the shared /NSTAT modulation).

What /IMPL/FATIG/MULT/MINPUT/EVOL/FCOH/NSTAT does (a PORT sub-flag — the open-source
engine has NO frequency-domain / spectral / multi-input / coherence / evolutionary
solver of any kind; freimpl.F reads only DYNA / BUCKL / DT / NONLIN / ARCL and its
sole "PSD" token is IMUMPSD, a MUMPS flag):
  1. runs the SINGLE-input /MULT analysis (the M21 baseline), the M28 STATIONARY
     frequency-dependent multi-input analysis (the convection coherence held at the
     start decay) and the M29 SCALAR-coherence evolutionary analysis FIRST — all
     stored byte-identical (a NEW parallel path);
  2. builds the per-window FREQUENCY-DEPENDENT coherence schedule (the exponential
     field with decay 0.9 -> 0.2), assembles the per-window Hermitian input
     cross-PSD S_ff(w, t_i) (each projected onto the nearest Hermitian PSD matrix);
  3. forms the per-window multi-input stress-tensor cross-PSD S_sigmasigma,i on the
     critical root element and RE-SEARCHES the critical plane / F_np per window (the
     plane may DRIFT as the decorrelation frequency moves up);
  4. Palmgren-Miner-sums the per-window multiaxial damages, and runs the
     non-stationary MULTI-INPUT multivariate Monte-Carlo cross-check (per-window
     blocks of the correlated-input synthesiser, per-window plane projection, ASTM
     E1049 rainflow + Miner), measuring the synthesised inputs' per-window coherence
     SPECTRUM (per frequency band) to confirm it tracks the drifting target shape;
  5. reports the coherence FREQUENCY-SHAPE drift / decorrelation-frequency drift /
     per-window RMS / critical-plane rotation / damage-life ALONGSIDE the M29
     scalar-coherence and M28 frequency-dependent-stationary numbers (a
     "freq_evolutionary_multi_input" sub-entry on
     model.implicit_result.fatigue["multi_input"]).

S-N curve: N = C*S^-m with m = 5, C = 1e4 (stress units GPa / ms base), as in the
M21 / M28 / M29 examples, so all the lives are on the same scale.

What to look for in the listing (FREQ_EVOLUTIONARY_MULTI_INPUT_0001.out):
  * the "** MULTI-INPUT / PARTIALLY-COHERENT SPECTRAL FATIGUE **
    (/IMPL/FATIG/MINPUT)" block — the M28 STATIONARY frequency-dependent answer
    (COHERENCE MODEL = EXPONENTIAL);
  * the "** FULLY EVOLUTIONARY MULTI-INPUT FATIGUE **
    (/IMPL/FATIG/MULT/MINPUT/EVOL)" block — the M29 SCALAR-coherence drift;
  * the "** FREQUENCY-DEPENDENT EVOLUTIONARY MULTI-INPUT FATIGUE **
    (/IMPL/FATIG/MULT/MINPUT/EVOL/FCOH)" block — the coherence FREQUENCY-SHAPE
    schedule (the band-resolved START/END coherence), the DECORRELATION-FREQUENCY
    drift, the critical-plane rotation driven by the evolving frequency-shape, the
    per-reduction frequency-dependent-evolutionary vs M29-scalar vs M28-stationary
    damage rates, the non-stationary MULTI-INPUT Monte-Carlo and the MEASURED
    per-window band-resolved coherence tracking the target.

Run it (from this directory):
    python generate_deck.py          # regenerates the two .rad files
    pyradioss-starter -i FREQ_EVOLUTIONARY_MULTI_INPUT_0000.rad
    pyradioss-engine  -i FREQ_EVOLUTIONARY_MULTI_INPUT_0001.rad
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
SEP = 3.0            # input SEPARATION |dx| (mm) — sets the convection coherence
DECAY0 = 0.9         # START exponential decay coefficient (strong -> low f_dec)
DECAY1 = 0.2         # END exponential decay coefficient (weak -> high f_dec)
SPEED = 100.0        # convection reference speed (mm/ms) — held constant
GAMMA_SCAL = 0.5     # the M29 SCALAR-coherence reference level (frequency-flat)
FC0 = 60.0           # window centre freq at the FIRST window (kHz)
FC1 = 180.0          # window centre freq at the LAST window (kHz)
BW = 80.0            # window bandwidth (kHz), constant (broad enough to keep energy)
NWIN = 6             # mission windows the coherence / shape drift is sampled into
FSCALE = 1.0e-4      # tip-force pattern amplitude (kN)
SN_M = 5.0           # S-N slope m (N = C*S^-m)
SN_C = 1.0e4         # S-N coefficient C (stress units GPa)
MCDUR = 400.0        # Monte-Carlo cross-check record length (ms)
SEED = 30000         # Monte-Carlo random seed (reproducible)

here = os.path.dirname(os.path.abspath(__file__))


def node_id(ix, iy, iz):
    """Structured-grid node id: (NX+1) along x, 2 along y, 2 along z."""
    return ix * 4 + iy * 2 + iz + 1


def main():
    lines = ["#RADIOSS STARTER", "/BEGIN",
             "FREQ_EVOLUTIONARY_MULTI_INPUT - solid-brick cantilever, TWO inputs "
             "with a FREQUENCY-DEPENDENT drifting convection coherence "
             "(pyradioss /IMPL/FATIG/MULT/MINPUT/EVOL/FCOH)"]

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
    lines += ["/FUNCT/1", "input-1 pattern (z)",
              "       0.0           1.0", "  100000.0           1.0"]
    lines += ["/FUNCT/2", "input-2 pattern (y)",
              "       0.0           1.0", "  100000.0           1.0"]
    lines += ["/FUNCT/10", "input-1 auto-PSD G1(f)",
              f"       0.0{G1:14g}", f"  100000.0{G1:14g}"]
    lines += ["/FUNCT/11", "input-2 auto-PSD G2(f)",
              f"       0.0{G2:14g}", f"  100000.0{G2:14g}"]
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
        lines, os.path.join(here, "FREQ_EVOLUTIONARY_MULTI_INPUT_0000.rad"), runname="FREQ_EVOLUTIONARY_MULTI_INPUT")

    # ---- engine: a bare implicit static step + the M30 freq-evol analysis --
    engine = [
        "#RADIOSS ENGINE",
        "/RUN/FREQ_EVOLUTIONARY_MULTI_INPUT/1",
        "1.0",                              # one (zero-load) static increment
        "/IMPL",                            # implicit static driver hosts it
        # M30: multi-input (/MINPUT) + evolutionary shape (/EVOL) + FREQUENCY-DEPENDENT
        # coherence (/FCOH). /NSTAT reads the mission RMS profile /FUNCT/30.
        "/IMPL/FATIG/MULT/MINPUT/EVOL/FCOH/NSTAT",
        # line 1 (PSD sweep): fmin fmax nf funct nmode (funct 10 = ref PSD)
        f"{FMIN}  {FMAX}  {NF}  10  {NMODE}",
        # line 2 (S-N + Monte-Carlo): m C zeta mean ult mcdur seed
        f"{SN_M}  {SN_C}  {ZETA}  0.0  0.0  {MCDUR}  {SEED}",
        # line 3 (/NSTAT modulation): modfunct — the mission RMS profile
        "30",
        # line 4 (/EVOL drifting shape): fc0 fc1 bw0 bw1 nwin
        f"{FC0}  {FC1}  {BW}  {BW}  {NWIN}",
        # line 5 (MULTI-INPUT header):
        #   ninput cohmodel gamma phase decay speed gamma1 phase1 decay1 speed1
        #   cohmodel 1 = EXPONENTIAL / convection field; gamma = the M29 scalar
        #   reference; (decay, speed) the START field; (decay1, speed1) the END
        #   field the M30 frequency-shape DRIFTS to (the decorrelation moving up).
        f"2  1  {GAMMA_SCAL}  0.0  {DECAY0}  {SPEED}  {GAMMA_SCAL}  0.0  "
        f"{DECAY1}  {SPEED}",
        # input rows: cload_funct psd_funct x y z  (the SPATIAL POSITIONS that set
        # the convection separation |dx| in the exponential coherence field)
        "1  10  0.0  0.0  0.0",             # input 1 at x = 0
        f"2  11  {SEP}  0.0  0.0",          # input 2 at x = SEP mm
        "/PRINT/-500",
        "/STOP",
        "15.0",
    ]
    deck_writer.write_engine_from_port_lines(
        engine, os.path.join(here, "FREQ_EVOLUTIONARY_MULTI_INPUT_0001.rad"))
    print("wrote FREQ_EVOLUTIONARY_MULTI_INPUT_0000.rad / "
          "FREQ_EVOLUTIONARY_MULTI_INPUT_0001.rad")


if __name__ == "__main__":
    main()
