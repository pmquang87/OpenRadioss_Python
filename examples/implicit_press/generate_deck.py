#!/usr/bin/env python3
"""
Generate the IMPLICIT_PRESS example: a rigid punch pressed onto an elastic
pad — /INTER/TYPE7 penalty CONTACT and an /RBE2 rigid body inside the M12
IMPLICIT static Newton loop (the two capabilities of Milestone 12 in one
deck).

Model (units mm / ms / kg -> forces kN, stresses GPa):
  * PAD: a 40 x 40 x 10 mm polymer block (E = 2 GPa, nu = 0.3), meshed
    4 x 4 x 2 with /BRICK, base clamped;
  * PUNCH: a 20 x 20 x 10 mm steel block (2 x 2 x 1 bricks) centered over
    the pad, its underside a hair (0.005 mm) into the contact gap so the
    first increment starts supported;
  * /RBE2: every punch node is a slave of a standalone master node above
    the block — the press RAM. The master's /BCS (110 111) is the ram
    GUIDE: lateral translations and all rotations held, the z stroke
    free. The whole punch is condensed into the master's 6-DOF block
    (rbe2_imp0.F -> implicit/constraints.py), so the only punch unknown
    is the ram stroke;
  * /INTER/TYPE7 between the punch's bottom nodes and the pad surface
    (Istf = 5 series stiffness, the i7sti3 element formulas), solved
    IMPLICITLY: the penalty force enters the Newton residual at the trial
    configuration and the active-set gap tangent K g g^T enters the
    assembled stiffness (i7ke3.F -> implicit/contact.py);
  * /CLOAD ramps F = 4 kN of press force onto the ram over the load
    factor 0 -> 1 (ten increments).

What to look at:
  * the listing's "CONDENSED EQUATIONS (M12)" line: the punch + ram DOFs
    collapse onto the ram's single free stroke;
  * one line per load increment, 2-4 Newton iterations each — the active
    set (which pairs touch) is re-evaluated inside every iteration;
  * the pad dents ~0.09 mm under the punch (peak contact pressure
    ~ 0.01 GPa over the 20 x 20 patch) while the punch bricks stay
    exactly undeformed (they are rigid slaves);
  * swap /IMPL for /IMPL/NONLIN to run the same press with the
    updated-Lagrangian frame + geometric stiffness.

Run it (from this directory):
    python generate_deck.py
    pyradioss-starter -i IMPLPRESS_0000.rad
    pyradioss-engine  -i IMPLPRESS_0001.rad
"""

import os

# pad
PLX, PLY, PLZ = 40.0, 40.0, 10.0
PNX, PNY, PNZ = 4, 4, 2
# punch (bottom at pad top - PEN0). Sized 15 x 15 with a 3 x 3 face mesh
# and OFFSET (1.7, 2.3) mm from dead center, so its underside nodes land
# inside the pad's 10-mm faces — never on a grid line and never on a cell
# diagonal. Both alignments are non-smooth points of the node-to-segment
# closest-point map (a vertex projection, and the crease/medial axis of a
# warped quad's two-triangle split): a punch mesh sitting EXACTLY on them
# parks the equilibrium on the kink and leaves Newton cycling at ~1e-5
# relative residual until the step control gives up (see the projection
# region discussion in implicit/contact.py). Realistic meshes are never
# perfectly aligned; this deck should not be either.
ULX, ULY, ULZ = 15.0, 15.0, 10.0
UNX, UNY, UNZ = 3, 3, 1
OFFX, OFFY = 1.7, 2.3
PEN0 = 0.005                       # initial bite into the contact gap (mm)
GAP = 0.05                         # /INTER/TYPE7 constant gap (mm)
F = 4.0                            # press force (kN)

here = os.path.dirname(os.path.abspath(__file__))


def main():
    lines = ["#RADIOSS STARTER", "/BEGIN",
             "IMPLPRESS - rigid punch on elastic pad, implicit contact"]

    # ---- nodes ----------------------------------------------------------
    lines.append("/NODE")
    pad_id = {}
    nid = 0
    for k in range(PNZ + 1):
        for j in range(PNY + 1):
            for i in range(PNX + 1):
                nid += 1
                pad_id[(i, j, k)] = nid
                lines.append(f"{nid:10d}{i * PLX / PNX:20.10f}"
                             f"{j * PLY / PNY:20.10f}"
                             f"{k * PLZ / PNZ:20.10f}")
    punch_id = {}
    x0 = (PLX - ULX) / 2.0 + OFFX
    y0 = (PLY - ULY) / 2.0 + OFFY
    z0 = PLZ + GAP - PEN0          # underside sits PEN0 into the gap
    for k in range(UNZ + 1):
        for j in range(UNY + 1):
            for i in range(UNX + 1):
                nid += 1
                punch_id[(i, j, k)] = nid
                lines.append(f"{nid:10d}{x0 + i * ULX / UNX:20.10f}"
                             f"{y0 + j * ULY / UNY:20.10f}"
                             f"{z0 + k * ULZ / UNZ:20.10f}")
    nid += 1
    ram = nid
    lines.append(f"{ram:10d}{x0 + ULX / 2:20.10f}{y0 + ULY / 2:20.10f}"
                 f"{z0 + ULZ + 10.0:20.10f}")

    # ---- elements ---------------------------------------------------------
    def bricks(tag, idmap, nx, ny, nz, eid0):
        out = [f"/BRICK/{tag}"]
        eid = eid0
        for k in range(nz):
            for j in range(ny):
                for i in range(nx):
                    eid += 1
                    c = [idmap[(i, j, k)], idmap[(i + 1, j, k)],
                         idmap[(i + 1, j + 1, k)], idmap[(i, j + 1, k)],
                         idmap[(i, j, k + 1)], idmap[(i + 1, j, k + 1)],
                         idmap[(i + 1, j + 1, k + 1)],
                         idmap[(i, j + 1, k + 1)]]
                    out.append(f"{eid:10d}"
                               + "".join(f"{v:10d}" for v in c))
        return out, eid

    b, eid = bricks(1, pad_id, PNX, PNY, PNZ, 0)
    lines += b
    b, eid = bricks(2, punch_id, UNX, UNY, UNZ, eid)
    lines += b

    lines += ["/PART/1", "pad", "         1         1",
              "/PART/2", "punch", "         1         2",
              "/MAT/LAW1/1", "polymer pad", "   1.2e-6",
              "       2.0       0.3",
              "/MAT/LAW1/2", "steel punch", "   7.8e-6",
              "     210.0       0.3",
              "/PROP/SOLID/1", "solid", "       1.1      0.05       0.1"]

    # ---- groups -------------------------------------------------------------
    base = [pad_id[(i, j, 0)] for j in range(PNY + 1)
            for i in range(PNX + 1)]
    punch_bottom = [punch_id[(i, j, 0)] for j in range(UNY + 1)
                    for i in range(UNX + 1)]
    punch_all = sorted(punch_id.values())

    def grnod(gid, title, ids):
        return [f"/GRNOD/NODE/{gid}", title,
                " ".join(str(v) for v in ids)]

    lines += grnod(1, "pad base", base)
    lines += grnod(2, "punch bottom", punch_bottom)
    lines += grnod(3, "punch all", punch_all)
    lines += grnod(4, "ram", [ram])

    # pad surface for the contact main side
    lines += ["/SURF/PART/1", "pad surface", "         1"]

    # ---- constraints ---------------------------------------------------------
    # the whole punch rides the ram rigidly; the ram guide leaves only the
    # z stroke free (see the module docstring)
    lines += ["/RBE2/1", "press ram",
              f"{ram:10d}         3",
              "/BCS/1", "pad base clamp",
              "       111       000         0         1",
              "/BCS/2", "ram guide",
              "       110       111         0         4"]

    # ---- contact ---------------------------------------------------------------
    lines += ["/INTER/TYPE7/1", "punch vs pad",
              "         2         1         5         0",
              f"       1.0       0.0{GAP:10.3f}       0.0"]

    # ---- load -------------------------------------------------------------------
    lines += ["/FUNCT/1", "ramp",
              "       0.0       0.0",
              "       2.0       2.0",
              "/CLOAD/1", "press force",
              f"         1         Z         4{-F:10.4f}"]

    lines.append("/END")
    with open(os.path.join(here, "IMPLPRESS_0000.rad"), "w") as f:
        f.write("\n".join(lines) + "\n")

    engine = """#
/RUN/IMPLPRESS/1
1.0
/IMPL
/IMPL/DTINI
0.1
/PRINT/1
/ANIM/DT
1.0
/END
"""
    with open(os.path.join(here, "IMPLPRESS_0001.rad"), "w") as f:
        f.write(engine)
    print("wrote IMPLPRESS_0000.rad / IMPLPRESS_0001.rad")


if __name__ == "__main__":
    main()
