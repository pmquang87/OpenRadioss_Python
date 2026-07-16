#!/usr/bin/env python3
"""
Generate the BOXIMP example deck: a thin-walled steel box beam (open at
both ends, 4 shell walls) flying into a rigid wall — a miniature version
of the classic crash-box benchmark.

Model (units mm / ms / kg -> kN, GPa):
  * box section 10 x 10 mm, length 20 mm, wall thickness 0.5 mm,
    meshed with 2 mm Belytschko-Tsay /SHELL elements (4 walls x 5 x 10)
  * material LAW2 Johnson-Cook mild steel
  * initial velocity -10 mm/ms toward the wall (fast enough to yield:
    the impact stress rho*c*v ~ 0.4 GPa exceeds A = 0.25 GPa)
  * frictionless rigid wall /RWALL/PLANE at z = 0.5 (normal +Z)
  * self-contact is not needed at this crush depth; the example focuses
    on shells + rigid wall + energy balance

Run it (from this directory):
    python generate_deck.py
    pyradioss-starter -i BOXIMP_0000.rad
    pyradioss-engine  -i BOXIMP_0001.rad
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


B = 10.0        # section width (mm)
L = 20.0        # box length (mm)
NB = 5          # elements across each wall
NL = 10         # elements along the length
Z0 = 0.5        # initial gap to the wall (mm)
V0 = -10.0       # impact velocity (mm/ms)

here = os.path.dirname(os.path.abspath(__file__))


def main():
    # Walk the closed square perimeter: 4 walls x NB segments = 4*NB
    # distinct corner lines, shared at the 4 box corners.
    perim = []
    for e in range(NB):                       # wall 1: y=0, x: 0->B
        perim.append((e * B / NB, 0.0))
    for e in range(NB):                       # wall 2: x=B, y: 0->B
        perim.append((B, e * B / NB))
    for e in range(NB):                       # wall 3: y=B, x: B->0
        perim.append((B - e * B / NB, B))
    for e in range(NB):                       # wall 4: x=0, y: B->0
        perim.append((0.0, B - e * B / NB))
    nperim = len(perim)                       # 4*NB

    def node_id(p, k):                        # perimeter index, axial index
        return 1 + p + k * nperim

    lines = ["#RADIOSS STARTER", "/BEGIN",
             "BOXIMP - shell box beam against rigid wall (pyradioss example)"]

    lines.append("/NODE")
    for k in range(NL + 1):
        z = Z0 + k * L / NL
        for p, (x, y) in enumerate(perim):
            lines.append(f"{node_id(p, k):10d}{x:20.10f}{y:20.10f}"
                         f"{z:20.10f}")

    lines.append("/SHELL/1")
    eid = 0
    for k in range(NL):
        for p in range(nperim):
            pn = (p + 1) % nperim
            eid += 1
            n = [node_id(p, k), node_id(pn, k),
                 node_id(pn, k + 1), node_id(p, k + 1)]
            lines.append(f"{eid:10d}" + "".join(f"{v:10d}" for v in n))

    lines += [
        "/PART/1",
        "box walls",
        "         1         1",
        "/MAT/LAW2/1",
        "mild steel Johnson-Cook",
        "   7.8e-6",
        "     210.0       0.3",
        "      0.25       0.4       0.5",     # A, B, n (GPa)
        "/PROP/SHELL/1",
        "box wall 0.5mm",
        "         1         0         0         0",   # Ishell flags card
        "      0.01      0.01      0.01       0.0       0.0",  # hm hf hr
        "         3         0       0.5",     # N=3 layers, thick=0.5
        # all box nodes get the initial velocity
        "/GRNOD/PART/1",
        "whole box",
        "1",
        "/INIVEL/TRA/1",
        "initial impact velocity",
        f"       0.0       0.0      {V0}         1",
        # rigid wall: plane z=0, normal +Z, frictionless sliding
        "/RWALL/PLANE/1",
        "ground plane",
        "         0         0       0.0       0.0",
        "       0.0       0.0       0.0",     # M
        "       0.0       0.0       1.0",     # M1 (normal = M->M1)
        "/TH/PART/1",
        "box energies",
        "IE KE",
        "1",
        "/END",
    ]
    deck_writer.write_starter_from_port_lines(
        lines, os.path.join(here, "BOXIMP_0000.rad"), runname="BOXIMP")

    engine = [
        "#RADIOSS ENGINE",
        "/RUN/BOXIMP/1",
        "1.0",
        "/DT",
        "0.9  0.0",
        "/TFILE",
        "0.005",
        "/ANIM/DT",
        "0.0  0.05",
        "/ANIM/VECT/VEL",
        "/ANIM/ELEM/VONM",
        "/ANIM/ELEM/EPSP",
        "/PRINT/-200",
        "/STOP",
        "15.0",
    ]
    deck_writer.write_engine_from_port_lines(
        engine, os.path.join(here, "BOXIMP_0001.rad"))
    print("wrote BOXIMP_0000.rad / BOXIMP_0001.rad")


if __name__ == "__main__":
    main()
