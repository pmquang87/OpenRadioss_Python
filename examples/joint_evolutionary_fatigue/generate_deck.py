#!/usr/bin/env python3
"""
Generate the JOINT_EVOLUTIONARY_FATIGUE example deck: the FULLY NON-STATIONARY /
EVOLUTIONARY MULTIAXIAL (JOINT-TENSOR) fatigue life of a base-clamped SOLID-brick
cantilever whose full stress-TENSOR cross-PSD S_sigmasigma(omega, t) EVOLVES with
time (M27, /IMPL/FATIG/MULT/EVOL/JOINT) — built on the M21 multiaxial machinery
(multiaxial_fatigue.py: the stress-tensor cross-PSD, the equivalent von Mises +
critical-plane reductions, the multivariate synthesiser), the M23 spectral F_np
(spectral_nonproportional_fatigue.py) AND the M26 drifting-shape spectrogram
(evolutionary_fatigue.py) — lifted from the equivalent SCALAR to the full 6x6
JOINT TENSOR (joint_evolutionary_fatigue.py).

This is deliberately the SAME solid-brick cantilever the M21 multiaxial_fatigue
example shakes, RE-RUN under an EVOLVING TENSOR. Where M21 asks the multiaxial
fatigue question of ONE stationary stress-tensor cross-PSD (a FIXED critical
plane), and M26 lets the SHAPE of the equivalent SCALAR drift with a FIXED
reduction, M27 lets the full JOINT tensor evolve: a RESONANCE SWEEP drifts the
excitation band UP in frequency with time, exciting a DIFFERENT mode of the
cantilever in each time-window — a bending mode (a near-uniaxial sigma_xx tensor)
at low frequency, a torsion / higher mode (a shear-dominated tensor) at high
frequency. Because the tensor ORIENTATION differs band to band, the CRITICAL
PLANE itself ROTATES window to window and the non-proportionality factor F_np
drifts — the physics M26's FIXED reduction is blind to and M27 recovers.

Model (units mm / ms / kg -> forces kN, stresses GPa, frequencies kHz): the M21
solid-brick cantilever with a RECTANGULAR cross-section (1 mm in z, WIDTH mm in y)
so the y- and z-bending modes SPLIT in frequency, and an OFFSET skew tip force so
the sweep passes through bending AND torsion modes carrying DIFFERENT stress
tensors —
  * a straight cantilever of NX /BRICK hexa8 solids along +x, clamped at x = 0;
  * steel LAW1 (E = 210 GPa, nu = 0.3, rho = 7.8e-6 kg/mm^3);
  * a random tip FORCE PSD applied at the free-end nodes, skew (y + z) AND offset
    (unequal across the width) so the root elements see bending + transverse shear
    + torsion — a genuinely evolving multiaxial stress as the sweep climbs;
  * /FUNCT/10 — a FLAT band-limited input force PSD S_ff(f);
  * /FUNCT/20 — the RMS MISSION PROFILE (level vs time: run-up / dwell / run-down),
    so the JOINT-tensor evolution composes with the M25 RMS non-stationarity.

What /IMPL/FATIG/MULT/EVOL/JOINT/NSTAT does (a PORT sub-flag — the open-source
engine has NO frequency-domain / spectral / multiaxial / evolutionary fatigue
solver of any kind; freimpl.F reads only DYNA / BUCKL / DT / NONLIN / ARCL and its
sole "PSD" token is IMUMPSD, a MUMPS flag):
  1..6. everything the M21 /IMPL/FATIG/MULT STATIONARY multiaxial path does — the
     force-excitation FRF, the recovered STRESS modes, the 6x6 stress-tensor
     cross-PSD S_sigmasigma = H_sigma S_ff H_sigma^H, the equivalent von Mises +
     max-normal + max-shear critical-plane reductions, the critical element and
     the four estimators, plus the multivariate Monte-Carlo — reported UNCHANGED
     (byte-identical whether or not JOINT runs);
  7. the M25 NON-STATIONARY analysis (the RMS mission profile block Miner-sum) and
     the M26 SCALAR-EVOLUTIONARY analysis (the drifting-shape window on each
     reduction's equivalent SCALAR, a FIXED reduction) — reported alongside;
  8. THEN, alongside, the M27 FULLY EVOLUTIONARY JOINT-TENSOR analysis: window the
     FULL 6x6 tensor cross-PSD per time-window (the resonance sweep), recompute the
     per-window moment matrices, and RE-SEARCH the critical plane / F_np from the
     WINDOW's OWN tensor (so the plane may ROTATE / F_np may DRIFT window to
     window), Palmgren-Miner summing the per-window MULTIAXIAL damages;
  9. a seeded non-stationary MULTIVARIATE Monte-Carlo cross-check: per-window
     multivariate spectral-representation blocks of the WINDOWED tensor, projected
     onto the WINDOW's OWN critical plane, ASTM E1049 rainflow, Miner sum — the
     time-domain multiaxial answer the joint-tensor window estimate approximates.

S-N curve: N = C*S^-m with m = 5 and C = 1e4 (identical to the M21 example).

What to look for in the listing (JOINT_EVOLUTIONARY_FATIGUE_0001.out):
  * the "** MULTIAXIAL / CRITICAL-PLANE SPECTRAL FATIGUE ** (/IMPL/FATIG/MULT)"
    block is the M21 STATIONARY multiaxial answer, UNCHANGED;
  * the "** NON-STATIONARY / EVOLUTIONARY-PSD FATIGUE ** (/IMPL/FATIG/NSTAT)"
    block is the M25 RMS-non-stationary answer (the FIXED tensor, level drift);
  * the "** FULLY EVOLUTIONARY / NON-SEPARABLE-PSD FATIGUE ** (/IMPL/FATIG/EVOL)"
    block is the M26 SCALAR-evolutionary answer (the drifting SCALAR, a FIXED
    reduction);
  * the "** FULLY EVOLUTIONARY MULTIAXIAL JOINT-TENSOR FATIGUE **
    (/IMPL/FATIG/MULT/EVOL/JOINT)" block reports the drifting-tensor schedule
    (fc / bw sweep), the critical-plane ROTATION across the windows, and for each
    reduction the STATIONARY rate NEXT TO the JOINT-tensor (window Miner-sum) rate
    and life, plus the PER-WINDOW critical-plane drift (the max-shear plane normal,
    F_np and sigma_vm per window) and the non-stationary multivariate Monte-Carlo;
  * the FOUR lives — stationary (M21), RMS-non-stationary (M25),
    scalar-shape-evolutionary (M26) and joint-tensor-evolutionary (M27) — are
    printed SIDE BY SIDE. Where the critical plane ROTATES, the M27 joint answer
    DIFFERS from the M26 fixed-reduction one (the re-searched plane always aligns
    with the active band's tensor); where the tensor is near-proportional (the
    plane does NOT rotate) the M27 answer CONFIRMS M26 (a built-in reduction).

The full result (the M21 stationary reductions + the M25 ``nonstationary``, M26
``evolutionary`` and M27 ``joint_evolutionary`` sub-entries, the last with the
per-window critical-plane drift, the plane rotation and the multivariate
Monte-Carlo) is on ``model.implicit_result.fatigue``.

Run it (from this directory):
    python generate_deck.py          # regenerates the two .rad files
    pyradioss-starter -i JOINT_EVOLUTIONARY_FATIGUE_0000.rad
    pyradioss-engine  -i JOINT_EVOLUTIONARY_FATIGUE_0001.rad
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


NX = 8               # bricks along the cantilever axis (x)
WY = 2.0             # cross-section WIDTH in y (mm) — rectangular 2(y) x 1(z) so
#                      the y- and z-bending modes SPLIT in frequency
HZ = 1.0             # cross-section height in z (mm)
E = 210.0            # Young's modulus (GPa)
NU = 0.3             # Poisson ratio (a genuine 3D solid: nu couples the axes)
RHO = 7.8e-6         # density (kg/mm^3)
ZETA = 0.03          # modal damping (3 %) for the FRF
NMODE = 12           # modes to extract (span the bending AND torsion modes)
FMIN = 0.0           # analysis band start (kHz)
FMAX = 800.0         # analysis band end (kHz) — spans bending + torsion modes
NF = 2000            # PSD sweep points
S0 = 1.0             # flat input tip-force PSD (kN^2/kHz)
FSCALE = 1.0e-4      # tip-force pattern amplitude (kN); 1 kN on 1 mm^2 = 1 GPa,
#                      so a small pattern keeps the root stress in the ~MPa range
SN_M = 5.0           # S-N slope m (N = C*S^-m)
SN_C = 1.0e4         # S-N coefficient C (stress units GPa)
MCDUR = 400.0        # Monte-Carlo cross-check record length (ms)
SEED = 27000         # Monte-Carlo random seed (reproducible)
NPLANE = 24          # candidate-plane azimuth divisions (15 deg)
NSEG = 12            # blocks the RMS mission profile is sampled into (M25)
# M27 joint-tensor drifting-shape schedule: a resonance SWEEP through the modes
FC0 = 40.0           # window centre freq at the first window (Hz, ~1st bending)
FC1 = 600.0          # window centre freq at the last window (Hz, ~torsion/high)
BW0 = 20.0           # window bandwidth start (Hz)
BW1 = 60.0           # window bandwidth end (broadens as the sweep climbs)
NWIN = 12            # time-windows the joint-tensor spectrogram is sampled into

here = os.path.dirname(os.path.abspath(__file__))


def node_id(ix, iy, iz):
    """Structured-grid node id: (NX+1) along x, 2 along y, 2 along z."""
    return ix * 4 + iy * 2 + iz + 1


def main():
    lines = ["#RADIOSS STARTER", "/BEGIN",
             "JOINT_EVOLUTIONARY_FATIGUE - solid-brick cantilever, evolving "
             "tensor (pyradioss /IMPL/FATIG/MULT/EVOL/JOINT)"]

    # ---- nodes: (NX+1) x 2 x 2 structured grid (rectangular 2y x 1z) --------
    lines.append("/NODE")
    for ix in range(NX + 1):
        for iy in range(2):
            for iz in range(2):
                nid = node_id(ix, iy, iz)
                lines.append(f"{nid:10d}{ix * HZ:20.10f}{iy * WY:20.10f}"
                             f"{iz * HZ:20.10f}")

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

    # ---- functions: a UNIT curve for the force PATTERN, and the input PSD ---
    lines += ["/FUNCT/1", "unit pattern curve",
              "       0.0           1.0", "  100000.0           1.0"]
    lines += ["/FUNCT/10", "flat input PSD S_ff(f)",
              f"       0.0{S0:14g}", f"  100000.0{S0:14g}"]
    # ---- /FUNCT/20: the RMS MISSION PROFILE (level vs time) ----------------
    lines += ["/FUNCT/20", "RMS mission profile (run-up / dwell / run-down)",
              "       0.0       0.4",
              "     150.0       1.5",
              "     350.0       1.5",
              "     500.0       0.3"]

    # ---- tip force pattern: OFFSET skew (y + z, unequal across the width) ---
    # An OFFSET skew load (different z-force on the two y-sides) excites BENDING
    # AND TORSION, so the root tensor carries sigma_xx (bending) + sigma_zx / xy
    # (shear / torsion) with a frequency-dependent MIX — the evolving tensor. The
    # per-node (fy, fz) pattern is asymmetric across y so the resultant has a
    # torsional couple. /CLOAD order: fct_ID  Dir  grnod_ID  Fscale.
    per = FSCALE / len(tip)
    # asymmetric z-force fractions across the width (y=0 side vs y=1 side) + a y
    # shear component, so the tip load is a skew + torsional pattern
    fz_of = {0: 1.4, 1: 0.6}       # more z-force on the y=0 side (a torsion couple)
    fy_frac = 0.4
    for k, nid in enumerate(tip):
        iy = (nid - 1) // 2 % 2     # the node's y index (0 or 1)
        lines += [f"/GRNOD/NODE/{10 + k}", f"tipn{k}", str(nid),
                  f"/CLOAD/{10 + 2 * k}", "fy",
                  f"         1         Y{10 + k:10d}{per * fy_frac:12g}",
                  f"/CLOAD/{11 + 2 * k}", "fz",
                  f"         1         Z{10 + k:10d}{per * fz_of[iy]:12g}"]
    lines.append("/END")

    deck_writer.write_starter_from_port_lines(
        lines, os.path.join(here, "JOINT_EVOLUTIONARY_FATIGUE_0000.rad"), runname="JOINT_EVOLUTIONARY_FATIGUE")

    # ---- engine: a bare implicit static step + the M27 joint-tensor fatigue -
    engine = [
        "#RADIOSS ENGINE",
        "/RUN/JOINT_EVOLUTIONARY_FATIGUE/1",
        "1.0",                              # one (zero-load) static increment
        "/IMPL",                            # implicit static driver hosts it
        "/IMPL/FATIG/MULT/EVOL/JOINT/NSTAT",  # M27: joint-tensor (+ M25/M26)
        # line 1 (PSD sweep): fmin fmax nf funct nmode  (funct 10 = S_ff)
        f"{FMIN}  {FMAX}  {NF}  10  {NMODE}",
        # line 2 (S-N + Monte-Carlo): m C zeta mean ult mcdur seed
        f"{SN_M}  {SN_C}  {ZETA}  0.0  0.0  {MCDUR}  {SEED}",
        # line 3 (M25 non-stationary): RMS modulation /FUNCT id + block count
        f"20  {NSEG}",
        # line 4 (M26/M27 drifting-shape): fc0 fc1 bw0 bw1 nwin  (resonance sweep)
        f"{FC0}  {FC1}  {BW0}  {BW1}  {NWIN}",
        "/PRINT/-500",
        "/STOP",
        "15.0",
    ]
    deck_writer.write_engine_from_port_lines(
        engine, os.path.join(here, "JOINT_EVOLUTIONARY_FATIGUE_0001.rad"))
    print("wrote JOINT_EVOLUTIONARY_FATIGUE_0000.rad / "
          "JOINT_EVOLUTIONARY_FATIGUE_0001.rad")


if __name__ == "__main__":
    main()
