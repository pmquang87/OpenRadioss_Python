#!/usr/bin/env python3
"""
Generate the MAST example deck: a cantilever antenna mast hit by a wind
gust — the first pyradioss model built from /BEAM elements (/PROP/TYPE3).

Model (units mm / ms / kg -> forces kN, stresses GPa):
  * vertical steel tube mast, height 2000 mm, section D=60.3 x t=4 mm
    (A = 707.5 mm^2, I = 2.818e5 mm^4, J = 2I), meshed with 10 /BEAM
    elements sharing one off-axis orientation node
  * base clamped (/BCS on translations AND rotations — beams carry
    moments, so rotational fixity matters, unlike solids)
  * wind gust = half-sine lateral tip force, 0 -> 0.5 kN -> 0 over 20 ms
    (/CLOAD + /FUNCT), then free vibration to 150 ms (~2 periods)

Closed-form checks to try against the T01 output:
  * static tip deflection under the 0.5 kN peak: F H^3 / 3EI = 22.5 mm
    (the dynamic peak overshoots this — gust duration ~ T/3)
  * first bending frequency f1 = (1.875^2 / 2 pi) sqrt(EI/(rho A H^4))
    = 14.5 Hz -> period 69 ms, visible as the free-vibration period of
    the tip displacement after the gust

Run it (from this directory):
    python generate_deck.py          # regenerates the two .rad files
    pyradioss-starter -i MAST_0000.rad
    pyradioss-engine  -i MAST_0001.rad
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


NEL = 10             # beam elements along the mast
H = 2000.0           # mast height (mm)
AREA = 707.5         # tube section area (mm^2)
IYY = 2.818e5        # bending inertia (mm^4), both axes
IXX = 2.0 * IYY      # torsion constant of a thin tube = polar = 2I

here = os.path.dirname(os.path.abspath(__file__))


def main():
    lines = []
    lines.append("#RADIOSS STARTER")
    lines.append("/BEGIN")
    lines.append("MAST - cantilever antenna mast under a wind gust "
                 "(pyradioss /BEAM example)")

    # ---- nodes: mast along +z, plus one orientation node ------------------
    lines.append("/NODE")
    for i in range(NEL + 1):
        z = i * H / NEL
        lines.append(f"{i + 1:10d}{0.0:20.10f}{0.0:20.10f}{z:20.10f}")
    # orientation node: defines every element's local y (any off-axis point)
    lines.append(f"{99:10d}{0.0:20.10f}{1000.0:20.10f}{0.0:20.10f}")

    # ---- beams (part 1): N1 N2 + shared orientation node 99 ----------------
    lines.append("/BEAM/1")
    for i in range(NEL):
        lines.append(f"{i + 1:10d}{i + 1:10d}{i + 2:10d}{99:10d}")

    # ---- part / material / property -----------------------------------------
    lines.append("/PART/1")
    lines.append("mast tube")
    lines.append("         1         1")              # prop 1, mat 1
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
    lines.append("/GRNOD/NODE/2")
    lines.append("tip")
    lines.append(str(NEL + 1))
    lines.append("/BCS/1")
    lines.append("clamp base (translations + rotations)")
    lines.append("       111       111         0         1")

    # ---- wind gust: half-sine lateral tip force -------------------------------
    lines.append("/FUNCT/1")
    lines.append("half-sine gust, peak 1.0 at t=10ms, zero after 20ms")
    import math
    for k in range(11):                              # sampled half-sine
        t = 2.0 * k
        lines.append(f"{t:10.4f}{math.sin(math.pi * t / 20.0):10.6f}")
    lines.append("     20.01       0.0")
    lines.append("    1000.0       0.0")
    lines.append("/CLOAD/1")
    lines.append("gust at the tip, +x, peak 0.5 kN")
    lines.append("         1         X         2       0.5")

    # ---- time history -----------------------------------------------------------
    lines.append("/TH/NODE/1")
    lines.append("tip motion")
    lines.append("DX VX")
    lines.append(str(NEL + 1))
    lines.append("/TH/PART/2")
    lines.append("mast energies")
    lines.append("IE KE")
    lines.append("1")
    lines.append("/END")

    deck_writer.write_starter_from_port_lines(
        lines, os.path.join(here, "MAST_0000.rad"), runname="MAST")

    engine = [
        "#RADIOSS ENGINE",
        "/RUN/MAST/1",
        "150.0",                     # final time (ms): gust + ~2 periods
        "/DT",
        "0.9  0.0",                  # scale factor, dt_min
        "/TFILE",
        "0.5",                       # time-history every 0.5 ms
        "/ANIM/DT",
        "0.0  5.0",                  # animation state every 5 ms
        "/ANIM/VECT/VEL",
        "/ANIM/ELEM/VONM",
        "/PRINT/-500",
        "/STOP",
        "15.0",
    ]
    deck_writer.write_engine_from_port_lines(
        engine, os.path.join(here, "MAST_0001.rad"))
    print("wrote MAST_0000.rad / MAST_0001.rad")


if __name__ == "__main__":
    main()
