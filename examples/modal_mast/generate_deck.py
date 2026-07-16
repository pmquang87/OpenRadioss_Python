#!/usr/bin/env python3
"""
Generate the MODALMAST example deck: the natural frequencies (free-vibration
modes) of a cantilever mast — the first pyradioss MODAL analysis (M16,
/IMPL/EIGV), built on the CONSISTENT element mass and the generalized
eigensolver of implicit/modal.py.

Model (units mm / ms / kg -> forces kN, stresses GPa, frequencies kHz):
  * the same vertical steel tube mast as ``antenna_mast`` — height 2000 mm,
    section D = 60.3 x t = 4 mm (A = 707.5 mm^2, I = 2.818e5 mm^4, J = 2I) —
    meshed with 20 /BEAM elements (finer than the transient deck: the higher
    modes want more elements per wavelength)
  * base clamped (/BCS on translations AND rotations)
  * NO load and NO physical time: /IMPL/EIGV extracts the lowest 6 natural
    frequencies and mode shapes of  (K - omega^2 M) phi = 0  with M the
    CONSISTENT (Rayleigh-beam) mass — the eigensolve runs after a trivial
    zero-load static step, exactly as /IMPL/BUCKL runs after its prestress.

What to look for in the listing (MODALMAST_0001.out):
  * the "** NATURAL FREQUENCIES COMPUTATION ** (/IMPL/EIGV)" block prints the
    six lowest frequencies (Hz in the consistent unit system = kHz here);
  * the first two are a DEGENERATE bending pair (Iyy = Izz, so bending about
    x and y ring at the same frequency), then the next bending pair, then the
    torsion and axial modes.

Closed-form checks (Euler-Bernoulli cantilever, tests/test_m16_modal.py
asserts these to < 2 %):

  bending  f_n = (beta_n L)^2 / (2 pi) * sqrt(E I / (rho A)) / L^2
           beta_1 L = 1.875,  beta_2 L = 4.694  ->  f_1 ~= 0.0145,
           f_2 ~= 0.0908  (in 1/ms = kHz)
  torsion  f_1 = 1/(4L) sqrt(G J / (rho I_p))       (fixed-free shaft)
  axial    f_1 = 1/(4L) sqrt(E / rho)               (fixed-free bar)

Try it: raise NEL and watch the CONSISTENT mass converge onto the closed
forms FROM ABOVE (it over-predicts; the lumped mass would under-predict —
the bracket tests/test_m16_modal.py checks). Add /IMPL/EIGV/STRS after a
tensile static preload to see the frequencies stiffen (a tension-tuned
guitar string / spinning shaft).

Run it (from this directory):
    python generate_deck.py          # regenerates the two .rad files
    pyradioss-starter -i MODALMAST_0000.rad
    pyradioss-engine  -i MODALMAST_0001.rad
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


NEL = 20             # beam elements along the mast (finer -> more modes)
H = 2000.0           # mast height (mm)
AREA = 707.5         # tube section area (mm^2)
IYY = 2.818e5        # bending inertia (mm^4), both axes
IXX = 2.0 * IYY      # torsion constant of a thin tube = polar = 2I
NMODE = 6            # natural frequencies to extract

here = os.path.dirname(os.path.abspath(__file__))


def main():
    lines = []
    lines.append("#RADIOSS STARTER")
    lines.append("/BEGIN")
    lines.append("MODALMAST - natural frequencies of a cantilever mast "
                 "(pyradioss /IMPL/EIGV modal example)")

    # ---- nodes: mast along +z, plus one orientation node ------------------
    lines.append("/NODE")
    for i in range(NEL + 1):
        z = i * H / NEL
        lines.append(f"{i + 1:10d}{0.0:20.10f}{0.0:20.10f}{z:20.10f}")
    lines.append(f"{99:10d}{0.0:20.10f}{1000.0:20.10f}{0.0:20.10f}")

    # ---- beams (part 1): N1 N2 + shared orientation node 99 ----------------
    lines.append("/BEAM/1")
    for i in range(NEL):
        lines.append(f"{i + 1:10d}{i + 1:10d}{i + 2:10d}{99:10d}")

    # ---- part / material / property -----------------------------------------
    lines.append("/PART/1")
    lines.append("mast tube")
    lines.append("         1         1")
    lines.append("/MAT/LAW1/1")
    lines.append("steel elastic")
    lines.append("   7.8e-6")                         # rho (kg/mm3)
    lines.append("     210.0       0.3")              # E (GPa), nu
    lines.append("/PROP/BEAM/1")
    lines.append("tube D60.3x4")
    lines.append(f"{AREA:10.4g}{IYY:10.4g}{IYY:10.4g}{IXX:10.4g}")

    # ---- clamp the base -------------------------------------------------------
    lines.append("/GRNOD/NODE/1")
    lines.append("base")
    lines.append("1")
    lines.append("/BCS/1")
    lines.append("clamp base (translations + rotations)")
    lines.append("       111       111         0         1")
    lines.append("/END")

    deck_writer.write_starter_from_port_lines(
        lines, os.path.join(here, "MODALMAST_0000.rad"), runname="MODALMAST")

    # ---- engine: a bare implicit static step + the modal extraction ---------
    engine = [
        "#RADIOSS ENGINE",
        "/RUN/MODALMAST/1",
        "1.0",                       # one (zero-load) static increment
        "/IMPL",                     # implicit static driver
        "/IMPL/EIGV",                # M16: modal (natural frequency) analysis
        f"{NMODE}",                  # number of frequencies to extract
        "/PRINT/-500",
        "/STOP",
        "15.0",
    ]
    deck_writer.write_engine_from_port_lines(
        engine, os.path.join(here, "MODALMAST_0001.rad"))
    print("wrote MODALMAST_0000.rad / MODALMAST_0001.rad")


if __name__ == "__main__":
    main()
