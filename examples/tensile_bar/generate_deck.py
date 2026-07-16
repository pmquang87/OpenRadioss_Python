#!/usr/bin/env python3
"""
Generate the TENSILE example deck: a steel bar pulled in tension.

Model (units mm / ms / kg -> forces kN, stresses GPa):
  * bar 10 x 2 x 2 mm meshed with 1 mm /BRICK solids (40 elements)
  * material LAW2 Johnson-Cook steel: A=0.4 GPa, B=0.5, n=0.5
  * x=0 face clamped (/BCS), x=10 face pulled at 1 mm/ms (/IMPVEL,
    ramped up over 0.02 ms to avoid a shock)
  * final time 0.2 ms -> 0.2 mm of pull = 2 % nominal strain, well past
    first yield (yield strain ~ A/E = 0.19 %)

Run it (from this directory):
    python generate_deck.py          # regenerates the two .rad files
    pyradioss-starter -i TENSILE_0000.rad
    pyradioss-engine  -i TENSILE_0001.rad
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


NX, NY, NZ = 10, 2, 2          # elements per direction
LX, LY, LZ = 10.0, 2.0, 2.0    # bar size (mm)

here = os.path.dirname(os.path.abspath(__file__))


def node_id(i, j, k):
    return 1 + i + j * (NX + 1) + k * (NX + 1) * (NY + 1)


def main():
    lines = []
    lines.append("#RADIOSS STARTER")
    lines.append("/BEGIN")
    lines.append("TENSILE - Johnson-Cook bar in tension (pyradioss example)")

    # ---- nodes -----------------------------------------------------------
    lines.append("/NODE")
    for k in range(NZ + 1):
        for j in range(NY + 1):
            for i in range(NX + 1):
                x, y, z = i * LX / NX, j * LY / NY, k * LZ / NZ
                lines.append(f"{node_id(i, j, k):10d}{x:20.10f}"
                             f"{y:20.10f}{z:20.10f}")

    # ---- bricks (part 1) ---------------------------------------------------
    lines.append("/BRICK/1")
    eid = 0
    for k in range(NZ):
        for j in range(NY):
            for i in range(NX):
                eid += 1
                n = [node_id(i, j, k), node_id(i + 1, j, k),
                     node_id(i + 1, j + 1, k), node_id(i, j + 1, k),
                     node_id(i, j, k + 1), node_id(i + 1, j, k + 1),
                     node_id(i + 1, j + 1, k + 1), node_id(i, j + 1, k + 1)]
                lines.append(f"{eid:10d}" + "".join(f"{v:10d}" for v in n))

    # ---- part / material / property ---------------------------------------
    lines.append("/PART/1")
    lines.append("steel bar")
    lines.append("         1         1")            # prop 1, mat 1
    lines.append("/MAT/LAW2/1")
    lines.append("steel Johnson-Cook")
    lines.append("   7.8e-6")                        # rho (kg/mm3)
    lines.append("     210.0       0.3")             # E (GPa), nu
    lines.append("       0.4       0.5       0.5")   # A, B, n (GPa)
    lines.append("/PROP/SOLID/1")
    lines.append("default solid")
    lines.append("       1.1      0.05       0.1")   # qa, qb, h

    # ---- node groups --------------------------------------------------------
    fixed = [node_id(0, j, k) for k in range(NZ + 1) for j in range(NY + 1)]
    pulled = [node_id(NX, j, k) for k in range(NZ + 1) for j in range(NY + 1)]
    lines.append("/GRNOD/NODE/1")
    lines.append("clamped face x=0")
    lines.append(" ".join(str(n) for n in fixed))
    lines.append("/GRNOD/NODE/2")
    lines.append("pulled face x=L")
    lines.append(" ".join(str(n) for n in pulled))

    # ---- boundary conditions + imposed velocity ------------------------------
    lines.append("/BCS/1")
    lines.append("clamp x=0")
    lines.append("       111       111         0         1")
    lines.append("/FUNCT/1")
    lines.append("velocity ramp 0 -> 1 mm/ms at t=0.02, then constant")
    lines.append("       0.0       0.0")
    lines.append("      0.02       1.0")
    lines.append("     100.0       1.0")
    lines.append("/IMPVEL/1")
    lines.append("pull face x=L in X")
    lines.append("         1         X         2       1.0")

    # ---- time history ----------------------------------------------------------
    lines.append("/TH/NODE/1")
    lines.append("pulled corner node")
    lines.append("DX VX")
    lines.append(str(node_id(NX, 0, 0)))
    lines.append("/TH/PART/2")
    lines.append("bar energies")
    lines.append("IE KE")
    lines.append("1")
    lines.append("/END")

    deck_writer.write_starter_from_port_lines(
        lines, os.path.join(here, "TENSILE_0000.rad"), runname="TENSILE")

    engine = [
        "#RADIOSS ENGINE",
        "/RUN/TENSILE/1",
        "0.2",                       # final time (ms)
        "/DT",
        "0.9  0.0",                  # scale factor, dt_min
        "/TFILE",
        "0.002",                     # time-history every 2e-3 ms
        "/ANIM/DT",
        "0.0  0.02",                 # animation state every 0.02 ms
        "/ANIM/VECT/VEL",
        "/ANIM/ELEM/VONM",
        "/ANIM/ELEM/EPSP",
        "/PRINT/-100",
        "/STOP",
        "15.0",
    ]
    deck_writer.write_engine_from_port_lines(
        engine, os.path.join(here, "TENSILE_0001.rad"))
    print("wrote TENSILE_0000.rad / TENSILE_0001.rad")


if __name__ == "__main__":
    main()
