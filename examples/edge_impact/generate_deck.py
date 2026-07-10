#!/usr/bin/env python3
"""
Generate the EDGE example deck: a shell strip dropped edge-first across a
simply-supported strip — contact happens BETWEEN the nodes, where only
/INTER/TYPE11 edge-to-edge contact can see it. The M4 TYPE11 showcase.

Model (units mm / ms / kg -> forces kN, stresses GPa):
  * a supported strip along X (80 x 10 mm, t = 1 mm) held at both ends;
  * a flyer strip along Y (80 x 10 mm), rotated 90 deg in plan, 5 mm
    above, falling at 2 mm/ms;
  * the two strips cross like swords: at the moment of contact the
    closest points lie in the INTERIOR of the shell edges — a
    node-to-surface interface (TYPE7) would only notice once a node
    penetrates, deep after the edges crossed. This crossed-edge blind
    spot is exactly why TYPE11 exists (crash models routinely stack a
    TYPE11 on top of a TYPE7 for the same parts);
  * /LINE/SURF extracts every shell edge of each strip; the interface
    uses Istf=2 (mean of the two edges' element stiffness) and Igap=1
    (physical gap = the sum of the half thicknesses, 1 mm).

What to look at:
  * EDGE_A*.vtk in ParaView: the flyer bounces off the crossing point
    while both strips take a bending wave (VONM); the mid-surfaces stay
    one thickness apart — threshold the DIS vector to see the arrest;
  * EDGET01.csv: kinetic -> internal energy exchange at the bounce, the
    contact energy spike during compression returning to the small
    damper dissipation after separation;
  * EDGE_0001.out: the energy error stays within a few percent (~-4%)
    through a 23k-cycle /DT 0.9 run — the sliding closest-point pair
    hops discretely from edge to edge while the flyer rings on the
    support, which is where the small unbooked remainder comes from.

Run it (from this directory):
    python generate_deck.py          # regenerates the two .rad files
    pyradioss-starter -i EDGE_0000.rad
    pyradioss-engine  -i EDGE_0001.rad
"""

import os

L, W = 80.0, 10.0        # strip size (mm)
EL = 2.5                 # element size (mm)
THICK = 1.0              # shell thickness (mm)
DROP_H = 5.0             # initial mid-surface separation (mm)
V0 = 2.0                 # impact speed (mm/ms)
T_END = 4.0

here = os.path.dirname(os.path.abspath(__file__))

NX = int(L / EL)
NY = int(W / EL)


def main():
    lines = []
    lines.append("#RADIOSS STARTER")
    lines.append("/BEGIN")
    lines.append("EDGE - crossed strips, /INTER/TYPE11 edge impact")

    lines.append("/NODE")

    def node_id(base, i, j):
        return base + 1 + i + j * (NX + 1)

    # supported strip: long axis X, centred at the origin, z = 0
    for j in range(NY + 1):
        for i in range(NX + 1):
            lines.append(f"{node_id(0, i, j):10d}"
                         f"{-L / 2 + i * EL:20.10f}"
                         f"{-W / 2 + j * EL:20.10f}{0.0:20.10f}")
    # flyer strip: long axis Y (swap the roles of i/j), z = DROP_H
    for j in range(NY + 1):
        for i in range(NX + 1):
            lines.append(f"{node_id(10000, i, j):10d}"
                         f"{-W / 2 + j * EL:20.10f}"
                         f"{-L / 2 + i * EL:20.10f}{DROP_H:20.10f}")

    for part, base in ((1, 0), (2, 10000)):
        lines.append(f"/SHELL/{part}")
        eid = base
        for j in range(NY):
            for i in range(NX):
                eid += 1
                n = [node_id(base, i, j), node_id(base, i + 1, j),
                     node_id(base, i + 1, j + 1), node_id(base, i, j + 1)]
                lines.append(f"{eid:10d}" + "".join(f"{v:10d}" for v in n))

    lines.append("/PART/1")
    lines.append("supported strip")
    lines.append("         1         1")
    lines.append("/PART/2")
    lines.append("flyer strip")
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

    # ---- supports + drop ---------------------------------------------------
    ends = ([node_id(0, 0, j) for j in range(NY + 1)]
            + [node_id(0, NX, j) for j in range(NY + 1)])
    lines.append("/GRNOD/NODE/1")
    lines.append("supported ends")
    lines.append(" ".join(str(n) for n in ends))
    lines.append("/GRNOD/PART/2")
    lines.append("flyer")
    lines.append("         2")
    lines.append("/BCS/1")
    lines.append("pin the supported ends")
    lines.append("       111       000         0         1")
    lines.append("/INIVEL/TRA/1")
    lines.append("drop the flyer")
    lines.append(f"       0.0       0.0{-V0:10.3f}         2")

    # ---- the edge-to-edge interface -----------------------------------------
    lines.append("/SURF/PART/1")
    lines.append("supported strip surface")
    lines.append("         1")
    lines.append("/SURF/PART/2")
    lines.append("flyer surface")
    lines.append("         2")
    lines.append("/LINE/SURF/1")
    lines.append("flyer edges")
    lines.append("         2")
    lines.append("/LINE/SURF/2")
    lines.append("supported strip edges")
    lines.append("         1")
    # Istf=2 (mean stiffness), Igap=1 (t/2 + t/2 physical gap)
    lines.append("/INTER/TYPE11/1")
    lines.append("crossed edges")
    lines.append("         1         2         2         1")
    lines.append("       1.0       0.0       0.0       0.0")

    lines.append("/TH/NODE/1")
    lines.append("flyer centre")
    lines.append("DZ VZ")
    lines.append(str(node_id(10000, NX // 2, NY // 2)))
    lines.append("/END")

    with open(os.path.join(here, "EDGE_0000.rad"), "w") as fh:
        fh.write("\n".join(lines) + "\n")

    engine = [
        "#RADIOSS ENGINE",
        "/RUN/EDGE/1",
        f"{T_END}",
        "/DT",
        "0.9  0.0",
        "/TFILE",
        "0.01",
        "/ANIM/DT",
        "0.0  0.1",
        "/ANIM/VECT/DIS",
        "/ANIM/VECT/VEL",
        "/ANIM/ELEM/VONM",
        "/PRINT/-1000",
        "/STOP",
        "15.0",
    ]
    with open(os.path.join(here, "EDGE_0001.rad"), "w") as fh:
        fh.write("\n".join(engine) + "\n")
    print("wrote EDGE_0000.rad / EDGE_0001.rad")


if __name__ == "__main__":
    main()
