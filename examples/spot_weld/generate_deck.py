#!/usr/bin/env python3
"""
Generate the WELD example deck: a lap joint of two shell strips glued by
/INTER/TYPE2 tied contact and pulled in shear — the M4 "tied assembly"
showcase.

Model (units mm / ms / kg -> forces kN, stresses GPa):
  * lower strip 60 x 10 mm at z = 0, upper strip 60 x 10 mm at z = 1
    (one thickness up: the shells are t = 1 mm and shell contact surfaces
    are their MID-surfaces, so the tie carries a physical offset of
    (t1 + t2)/2 = 1 mm — the classic spot-weld configuration),
  * the strips overlap for 20 mm; every upper-strip node inside the
    overlap is tied to the lower strip's surface (/INTER/TYPE2 with the
    node group cut out by a /GRNOD/BOX),
  * lower strip clamped at its far end, upper strip pulled at 1 mm/ms
    (ramped) from its far end: the load path runs through the tied patch
    in single-lap shear,
  * both strips elastic LAW1 steel — the point is the JOINT, not the
    material.

What to look at:
  * WELDT01.csv — internal energy grows as the strips stretch; CONTACT
    energy stays EXACTLY zero: the kinematic tie does no work (this is
    the defining property of TYPE2, asserted by the M4 test suite);
  * WELD_A*.vtk in ParaView — the offset overlap follows the lower strip
    rigidly; the lap joint transmits the pull with the usual slight
    secondary bending of a single-lap joint;
  * WELD_0001.out — the tied-node count in the starter listing
    (11 x 3 = 33 nodes tied), energy error a fraction of a percent.

Run it (from this directory):
    python generate_deck.py          # regenerates the two .rad files
    pyradioss-starter -i WELD_0000.rad
    pyradioss-engine  -i WELD_0001.rad
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


L, W = 60.0, 10.0        # strip size (mm)
EL = 2.0                 # element size (mm)
OVERLAP = 20.0           # lap length (mm)
THICK = 1.0              # shell thickness (mm)
PULL_V = 1.0             # pull velocity (mm/ms)
T_RAMP = 0.05
T_END = 1.0

here = os.path.dirname(os.path.abspath(__file__))

NX = int(L / EL)         # elements along a strip
NY = int(W / EL)         # elements across


def main():
    lines = []
    lines.append("#RADIOSS STARTER")
    lines.append("/BEGIN")
    lines.append("WELD - lap joint tied by /INTER/TYPE2, pulled in shear")

    # ---- nodes: lower strip x in [0, 60] z=0, upper x in [40, 100] z=1 ---
    lines.append("/NODE")

    def node_id(base, i, j):
        return base + 1 + i + j * (NX + 1)

    for base, x0, z in ((0, 0.0, 0.0), (1000, L - OVERLAP, THICK)):
        for j in range(NY + 1):
            for i in range(NX + 1):
                lines.append(f"{node_id(base, i, j):10d}"
                             f"{x0 + i * EL:20.10f}{j * EL:20.10f}"
                             f"{z:20.10f}")

    # ---- shells ----------------------------------------------------------
    for part, base in ((1, 0), (2, 1000)):
        lines.append(f"/SHELL/{part}")
        eid = base
        for j in range(NY):
            for i in range(NX):
                eid += 1
                n = [node_id(base, i, j), node_id(base, i + 1, j),
                     node_id(base, i + 1, j + 1), node_id(base, i, j + 1)]
                lines.append(f"{eid:10d}" + "".join(f"{v:10d}" for v in n))

    # ---- parts / material / property --------------------------------------
    lines.append("/PART/1")
    lines.append("lower strip")
    lines.append("         1         1")
    lines.append("/PART/2")
    lines.append("upper strip")
    lines.append("         1         1")
    lines.append("/MAT/LAW1/1")
    lines.append("elastic steel")
    lines.append("    7.8e-6")
    lines.append("     210.0       0.3")
    lines.append("/PROP/SHELL/1")
    lines.append(f"strip t={THICK}")
    lines.append("         1         0         0         0")
    lines.append("      0.01      0.01      0.01         0         0")
    lines.append(f"         3         0{THICK:10.3f}")

    # ---- the tied joint ----------------------------------------------------
    # secondary nodes: every upper-strip node inside the overlap band,
    # selected by a box (the Radioss way of cutting node sets in space)
    lines.append("/BOX/RECTA/1")
    lines.append("overlap band")
    lines.append(f"{L - OVERLAP - 0.1:10.3f}{-0.1:10.3f}"
                 f"{THICK - 0.1:10.3f}")
    lines.append(f"{L + 0.1:10.3f}{W + 0.1:10.3f}{THICK + 0.1:10.3f}")
    lines.append("/GRNOD/BOX/3")
    lines.append("upper overlap nodes")
    lines.append("         1")
    lines.append("/SURF/PART/1")
    lines.append("lower strip surface")
    lines.append("         1")
    # dsearch = 1.5: the tied nodes sit exactly one thickness (1 mm) away
    lines.append("/INTER/TYPE2/1")
    lines.append("spot weld patch")
    lines.append("         3         1       1.5")

    # ---- constraints + pull -------------------------------------------------
    clamp = [node_id(0, 0, j) for j in range(NY + 1)]
    pull = [node_id(1000, NX, j) for j in range(NY + 1)]
    lines.append("/GRNOD/NODE/1")
    lines.append("lower clamped end")
    lines.append(" ".join(str(n) for n in clamp))
    lines.append("/GRNOD/NODE/2")
    lines.append("upper pulled end")
    lines.append(" ".join(str(n) for n in pull))
    lines.append("/BCS/1")
    lines.append("clamp lower end")
    lines.append("       111       111         0         1")
    lines.append("/FUNCT/1")
    lines.append("velocity ramp")
    lines.append("       0.0       0.0")
    lines.append(f"{T_RAMP:10.3f}       1.0")
    lines.append("     100.0       1.0")
    lines.append("/IMPVEL/1")
    lines.append("pull upper end in +X")
    lines.append(f"         1         X         2{PULL_V:10.3f}")

    # ---- output --------------------------------------------------------------
    lines.append("/TH/NODE/1")
    lines.append("pulled corner")
    lines.append("DX VX")
    lines.append(str(node_id(1000, NX, 0)))
    lines.append("/END")

    deck_writer.write_starter_from_port_lines(
        lines, os.path.join(here, "WELD_0000.rad"), runname="WELD")

    engine = [
        "#RADIOSS ENGINE",
        "/RUN/WELD/1",
        f"{T_END}",
        "/DT",
        "0.9  0.0",
        "/TFILE",
        "0.005",
        "/ANIM/DT",
        "0.0  0.05",
        "/ANIM/VECT/DIS",
        "/ANIM/ELEM/VONM",
        "/PRINT/-500",
        "/STOP",
        "15.0",
    ]
    deck_writer.write_engine_from_port_lines(
        engine, os.path.join(here, "WELD_0001.rad"))
    print("wrote WELD_0000.rad / WELD_0001.rad")


if __name__ == "__main__":
    main()
