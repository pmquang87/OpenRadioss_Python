#!/usr/bin/env python3
"""
Generate the SPECTRAL_NONPROPORTIONAL_FATIGUE example deck: the SPECTRAL (no
synthesised history) non-proportional critical-plane fatigue life of the SAME
base-clamped RECTANGULAR-section solid-brick cantilever the M22
nonproportional_fatigue example drives — now reporting its SPECTRAL
non-proportional life ALONGSIDE the M22 TIME-DOMAIN path-counting life and the
M21 PROPORTIONAL-spectral life, the three side by side (M23,
/IMPL/FATIG/MULT/NPROP/SPEC, `implicit/spectral_nonproportional_fatigue.py`).

Where the M22 nonproportional_fatigue example counts the damage of the rotating
shear PATH in the TIME domain (synthesise the correlated stress-component
histories, resolve the 2-D shear path on each plane, measure its MRH amplitude
and rainflow-count it), M23 gets the SAME non-proportional answer in the
FREQUENCY domain with NO synthesised history: it reads the non-proportionality
factor F_np straight out of the 2x2 in-plane shear block of the M21 stress-tensor
moment matrix M_0 and applies it as the closed-form amplitude correction
g = sqrt(1 + F_np^2) to the M20 spectral estimators — the Pitoiset /
Cristofori-Susmel-Tovo spectral method the M22 path count deliberately deferred.

WHY THE TWO METHODS AGREE. The 2x2 shear block of M_0 = E[sigma sigma^T] IS the
covariance of the (tau_a, tau_b) shear path the M22 synthesiser reproduces, so
the SPECTRAL F_np equals the M22 TIME-DOMAIN F_np, and the closed-form path
factor g = sqrt(1 + F_np^2) equals M22's measured MRH/scalar ratio (exactly for
a clean ellipse, within the seeded Monte-Carlo scatter for a general path). The
non-proportional damage is therefore g^m higher than the M21 proportional-
spectral projection in BOTH methods — the spectral M23 gets there without ever
building a time history.

The model is IDENTICAL to the M22 nonproportional_fatigue deck (so the three
lives are directly comparable):
  * a straight cantilever of NX /BRICK hexa8 solids along +x, a WY x WZ = 1 x 2 mm
    RECTANGULAR cross-section (WY != WZ splits the two transverse bending
    frequencies to ~33 and ~60 kHz — the non-proportional key), clamped at x = 0;
  * steel LAW1 (E = 210 GPa, nu = 0.3, rho = 7.8e-6 kg/mm^3);
  * a random tip FORCE PSD applied at the free-end nodes in BOTH +y and +z (equal
    amplitudes) so the two split bending modes are both excited; 3 % modal damping;
  * /FUNCT/10 — a FLAT band-limited input force PSD S_ff(f) spanning both modes.
The SPLIT bending modes at different frequencies decorrelate the root shears
sigma_xy (~33 kHz) and sigma_zx (~60 kHz) in time, so the resolved shear traces a
rotating 2-D PATH (a near-circle): a genuinely NON-PROPORTIONAL stress state
(F_np ~ 0.9).

What /IMPL/FATIG/MULT/NPROP/SPEC does (a PORT sub-flag — the open-source engine
has NO frequency-domain / spectral / critical-plane / non-proportional fatigue
solver of any kind; freimpl.F reads only DYNA / BUCKL / DT / NONLIN / ARCL and
its sole "PSD" token is IMUMPSD, a MUMPS flag):
  1.-4. everything the M21 /IMPL/FATIG/MULT does — the equivalent VON MISES /
     MAX-NORMAL / MAX-SHEAR spectral reductions (the PROPORTIONAL-spectral answer);
  5.    everything the M22 /IMPL/FATIG/MULT/NPROP does — synthesise the correlated
     stress-component histories, measure the shear-path amplitude (MCC / chord /
     MRH) and F_np, rainflow-count the TIME-DOMAIN Findley / Fatemi-Socie /
     shear-path damage (the NON-PROPORTIONAL time-domain answer);
  6.    THEN, ALONGSIDE, the M23 SPECTRAL estimate: from the SAME critical
     element's 6x6 spectral-MOMENT matrices, form the 2x2 in-plane shear block,
     read F_np = sqrt(lambda_2/lambda_1) and the Susmel-Tovo stress ratio
     rho = sigma_a/tau_a on each candidate plane, scale the dominant-shear PSD
     moments by (1 + F_np^2), search the critical plane and run the M20 estimators
     (Dirlik etc.) on the corrected moments — the NON-PROPORTIONAL SPECTRAL answer,
     with NO synthesised history.

S-N curve: N = C*S^-m with m = 5 and C = 1e4 — the same as the M21 / M22 examples
so the three lives (proportional-spectral, non-proportional time-domain,
non-proportional spectral) are directly comparable.

What to look for in the listing (SPECTRAL_NONPROPORTIONAL_FATIGUE_0001.out):
  * the M21 "** MULTIAXIAL / CRITICAL-PLANE SPECTRAL FATIGUE **" block (the
    proportional-spectral answer) — printed first, UNCHANGED;
  * the M22 "** NON-PROPORTIONAL MULTIAXIAL FATIGUE **" block (the time-domain
    path count) — the MCC/chord/MRH amplitudes, F_np ~ 0.9, the Findley /
    Fatemi-Socie / shear-path lives;
  * the M23 "** SPECTRAL NON-PROPORTIONAL FATIGUE **" block: the FREQUENCY-DOMAIN
    F_np (~ the M22 value — the M_0 = E[sigma sigma^T] identity), the closed-form
    path factor g = sqrt(1 + F_np^2), the Susmel-Tovo rho, and the Findley /
    Fatemi-Socie / shear-path lives — CLOSE to the M22 time-domain lives (the two
    non-proportional methods converge) and ~g^m SHORTER than the M21
    proportional-spectral projection of the same state (the extra damage the
    projection misses).

The full spectral result (the three critical-plane models with their critical
plane / F_np / rho / effective shear amplitude / damage / life, and the amplitude
comparison) is on
``model.implicit_result.fatigue['nprop_result']['spectral']`` (alongside the M22
time-domain ``nprop_result`` and the M21 spectral dict). The M21 spectral AND the
M22 time-domain answers are byte-identical whether or not SPEC runs.

Run it (from this directory):
    python generate_deck.py          # regenerates the two .rad files
    pyradioss-starter -i SPECTRAL_NONPROPORTIONAL_FATIGUE_0000.rad
    pyradioss-engine  -i SPECTRAL_NONPROPORTIONAL_FATIGUE_0001.rad
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
WY = 1.0             # cross-section width in y (mm) — the RECTANGULAR section...
WZ = 2.0             # cross-section height in z (mm) — ...WY != WZ splits the two
#                      transverse bending frequencies (the non-proportional key)
E = 210.0            # Young's modulus (GPa)
NU = 0.3             # Poisson ratio (a genuine 3D solid: nu couples the axes)
RHO = 7.8e-6         # density (kg/mm^3)
ZETA = 0.03          # modal damping (3 %) for the FRF
NMODE = 8            # modes to extract (span both split bending modes)
FMIN = 0.0           # analysis band start (kHz)
FMAX = 500.0         # analysis band end (kHz) — spans both bending modes
NF = 800             # PSD sweep points
S0 = 1.0             # flat input tip-force PSD (kN^2/kHz)
FSCALE = 1.0e-4      # tip-force pattern amplitude (kN) — keeps the root stress
#                      in the ~20 MPa high-cycle-fatigue range
FY = 1.0             # tip-force y fraction (excites bending about z, ~60 kHz)
FZ = 1.0             # tip-force z fraction (excites bending about y, ~33 kHz)
#                      BOTH nonzero -> both split modes -> non-proportional root
SN_M = 5.0           # S-N slope m (N = C*S^-m) — matches the M21 / M22 examples
SN_C = 1.0e4         # S-N coefficient C (stress units GPa) — matches M21 / M22
K_NP = 0.3           # Findley / Fatemi-Socie normal-sensitivity constant k
SIGY = 0.5           # yield stress sigma_y (GPa) for Fatemi-Socie
MCDUR = 60.0         # synthesised-history / path-count record length (ms) — the
#                      M22 time-domain path count needs it; the M23 spectral
#                      estimate does NOT (it reads the moment matrices directly)
SEED = 23000         # random seed (reproducible)

here = os.path.dirname(os.path.abspath(__file__))


def node_id(ix, iy, iz):
    """Structured-grid node id: (NX+1) along x, 2 along y, 2 along z."""
    return ix * 4 + iy * 2 + iz + 1


def main():
    lines = ["#RADIOSS STARTER", "/BEGIN",
             "SPECTRAL_NONPROPORTIONAL_FATIGUE - rectangular-section solid-brick "
             "cantilever (pyradioss /IMPL/FATIG/MULT/NPROP/SPEC)"]

    # ---- nodes: (NX+1) x 2 x 2 structured grid, RECTANGULAR WY x WZ section ---
    lines.append("/NODE")
    for ix in range(NX + 1):
        for iy in range(2):
            for iz in range(2):
                nid = node_id(ix, iy, iz)
                lines.append(f"{nid:10d}{ix * 1.0:20.10f}{iy * WY:20.10f}"
                             f"{iz * WZ:20.10f}")

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

    # ---- tip force pattern: BOTH transverse directions (+y AND +z) ---------
    # a single random process (one /FUNCT PSD) drives BOTH directions; the two
    # SPLIT bending modes it excites at different frequencies are what make the
    # root stress non-proportional (module / example docstring)
    per = FSCALE / len(tip)
    for k, nid in enumerate(tip):
        lines += [f"/GRNOD/NODE/{10 + k}", f"tipn{k}", str(nid),
                  f"/CLOAD/{10 + 2 * k}", "fy",
                  f"         1         Y{10 + k:10d}{per * FY:12g}",
                  f"/CLOAD/{11 + 2 * k}", "fz",
                  f"         1         Z{10 + k:10d}{per * FZ:12g}"]
    lines.append("/END")

    deck_writer.write_starter_from_port_lines(
        lines, os.path.join(here, "SPECTRAL_NONPROPORTIONAL_FATIGUE_0000.rad"), runname="SPECTRAL_NONPROPORTIONAL_FATIGUE")

    # ---- engine: a bare implicit static step + the M23 analysis ------------
    engine = [
        "#RADIOSS ENGINE",
        "/RUN/SPECTRAL_NONPROPORTIONAL_FATIGUE/1",
        "1.0",                          # one (zero-load) static increment
        "/IMPL",                        # implicit static driver hosts the analysis
        "/IMPL/FATIG/MULT/NPROP/SPEC",  # M23: spectral non-proportional (+ M21/M22)
        # line 1 (PSD sweep): fmin fmax nf funct nmode  (funct 10 = S_ff)
        f"{FMIN}  {FMAX}  {NF}  10  {NMODE}",
        # line 2 (S-N + path count): m C zeta mean ult mcdur seed k sigma_y
        f"{SN_M}  {SN_C}  {ZETA}  0.0  0.0  {MCDUR}  {SEED}  {K_NP}  {SIGY}",
        "/PRINT/-500",
        "/STOP",
        "15.0",
    ]
    deck_writer.write_engine_from_port_lines(
        engine, os.path.join(here, "SPECTRAL_NONPROPORTIONAL_FATIGUE_0001.rad"))
    print("wrote SPECTRAL_NONPROPORTIONAL_FATIGUE_0000.rad / "
          "SPECTRAL_NONPROPORTIONAL_FATIGUE_0001.rad")


if __name__ == "__main__":
    main()
