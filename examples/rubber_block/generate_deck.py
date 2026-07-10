#!/usr/bin/env python3
"""
Generate the RUBBER example deck: a hyperelastic rubber block squashed
between a fixed base and a driven top face.

Model (units mm / ms / kg -> forces kN, stresses GPa):
  * block 10 x 10 x 10 mm meshed with 2.5 mm /BRICK solids (4x4x4 = 64)
  * material LAW42 Ogden (Mooney-Rivlin pair mu = 0.0008/-0.0002 GPa,
    alpha = 2/-2 -> ground-state shear modulus G0 = 1 MPa, a soft filled
    rubber), nu = 0.495 (nearly incompressible: watch the block BULGE
    laterally as it is compressed — that is the volume conservation)
  * base (z=0) held in z, the top face driven down 4 mm (40% nominal
    compression) with a smooth ramp, then HELD compressed
  * lateral faces free

What to look at:
  * the LATERAL BULGE in the A*.vtk states (incompressibility);
  * the time step in RUBBER_0001.out: it DROPS as the block stiffens —
    LAW42 feeds its nonlinear (tangent) sound speed into the Courant
    limit every cycle, the M3 time-step requirement;
  * energy balance: hyperelastic = no dissipation, so after the ramp the
    internal energy stays constant while the block sits compressed.

Run it (from this directory):
    python generate_deck.py          # regenerates the two .rad files
    pyradioss-starter -i RUBBER_0000.rad
    pyradioss-engine  -i RUBBER_0001.rad
"""

import os

N = 4                          # elements per direction
L = 10.0                       # block edge (mm)
CRUSH_V = 2.0                  # top-face velocity (mm/ms)
T_RAMP = 0.25                  # velocity ramp time (ms)
T_PUSH = 2.0                   # time at full speed -> ~4 mm travel
T_END = 3.5                    # hold compressed until here (ms)

here = os.path.dirname(os.path.abspath(__file__))


def node_id(i, j, k):
    return 1 + i + j * (N + 1) + k * (N + 1) * (N + 1)


def main():
    lines = []
    lines.append("#RADIOSS STARTER")
    lines.append("/BEGIN")
    lines.append("RUBBER - Ogden block compression (pyradioss example)")

    # ---- nodes -----------------------------------------------------------
    lines.append("/NODE")
    for k in range(N + 1):
        for j in range(N + 1):
            for i in range(N + 1):
                x, y, z = (i * L / N, j * L / N, k * L / N)
                lines.append(f"{node_id(i, j, k):10d}{x:20.10f}"
                             f"{y:20.10f}{z:20.10f}")

    # ---- bricks (part 1) ---------------------------------------------------
    lines.append("/BRICK/1")
    eid = 0
    for k in range(N):
        for j in range(N):
            for i in range(N):
                eid += 1
                n = [node_id(i, j, k), node_id(i + 1, j, k),
                     node_id(i + 1, j + 1, k), node_id(i, j + 1, k),
                     node_id(i, j, k + 1), node_id(i + 1, j, k + 1),
                     node_id(i + 1, j + 1, k + 1), node_id(i, j + 1, k + 1)]
                lines.append(f"{eid:10d}" + "".join(f"{v:10d}" for v in n))

    # ---- part / material / property ---------------------------------------
    lines.append("/PART/1")
    lines.append("rubber block")
    lines.append("         1         1")            # prop 1, mat 1
    lines.append("/MAT/LAW42/1")
    lines.append("rubber Mooney-Rivlin (G0 = 1 MPa)")
    lines.append("    1.0e-6")                       # rho (kg/mm3)
    lines.append("    0.0008   -0.0002")             # mu_1, mu_2 (GPa)
    lines.append("       2.0      -2.0")             # alpha_1, alpha_2
    lines.append("     0.495")                       # nu (near incompressible)
    lines.append("/PROP/SOLID/1")
    lines.append("default solid")
    lines.append("       1.1      0.05       0.1")   # qa, qb, h

    # ---- node groups --------------------------------------------------------
    base = [node_id(i, j, 0) for j in range(N + 1) for i in range(N + 1)]
    top = [node_id(i, j, N) for j in range(N + 1) for i in range(N + 1)]
    lines.append("/GRNOD/NODE/1")
    lines.append("base z=0")
    lines.append(" ".join(str(n) for n in base))
    lines.append("/GRNOD/NODE/2")
    lines.append("top face z=L")
    lines.append(" ".join(str(n) for n in top))

    # ---- boundary conditions + imposed velocity ------------------------------
    lines.append("/BCS/1")
    lines.append("base: held in z (lateral free -> it can spread)")
    lines.append("       001       000         0         1")
    lines.append("/FUNCT/1")
    lines.append("smooth push then hold compressed")
    lines.append("       0.0       0.0")
    lines.append(f"{T_RAMP:10.3f}       1.0")
    lines.append(f"{T_RAMP + T_PUSH:10.3f}       1.0")
    lines.append(f"{T_RAMP + T_PUSH + T_RAMP:10.3f}       0.0")
    lines.append("     100.0       0.0")
    lines.append("/IMPVEL/1")
    lines.append("crush top face in -Z")
    lines.append(f"         1         Z         2{-CRUSH_V:10.3f}")

    # ---- time history ----------------------------------------------------------
    lines.append("/TH/NODE/1")
    lines.append("top corner node")
    lines.append("DZ VZ")
    lines.append(str(node_id(0, 0, N)))
    lines.append("/TH/PART/2")
    lines.append("block energies")
    lines.append("IE KE")
    lines.append("1")
    lines.append("/END")

    with open(os.path.join(here, "RUBBER_0000.rad"), "w") as fh:
        fh.write("\n".join(lines) + "\n")

    engine = [
        "#RADIOSS ENGINE",
        "/RUN/RUBBER/1",
        f"{T_END}",                  # final time (ms)
        "/DT",
        "0.9  0.0",                  # scale factor, dt_min
        "/TFILE",
        "0.01",                      # time history every 0.01 ms
        "/ANIM/DT",
        "0.0  0.1",                  # animation state every 0.1 ms
        "/ANIM/VECT/VEL",
        "/ANIM/ELEM/VONM",
        "/PRINT/-100",
        "/STOP",
        "15.0",
    ]
    with open(os.path.join(here, "RUBBER_0001.rad"), "w") as fh:
        fh.write("\n".join(engine) + "\n")
    print("wrote RUBBER_0000.rad / RUBBER_0001.rad")


if __name__ == "__main__":
    main()
