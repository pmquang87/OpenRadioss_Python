#!/usr/bin/env python3
"""
Generate the IMPLICIT_CLAMP example: the M12 press punch loaded SIDEWAYS
below the Coulomb cone — /INTER/TYPE7 friction solved inside the implicit
Newton loop (Milestone 13).

Model (units mm / ms / kg -> forces kN, stresses GPa):
  * PAD: a 40 x 40 x 10 mm polymer block (E = 2 GPa, nu = 0.3), meshed
    4 x 4 x 2 with /BRICK, base clamped;
  * PUNCH: a 15 x 15 x 10 mm steel block riding an /RBE2 ram whose guide
    (/BCS 010 111) holds the y translation and ALL rotations but leaves
    the z stroke AND the x slide free — the punch can only be held in x
    by FRICTION;
  * /INTER/TYPE7 with Coulomb mu = 0.3 (Istf = 5): the tangential force
    follows the M13 incremental return mapping (i7kfor3.F's incremental
    branch -> implicit/contact.py) — stick = tangential penalty spring on
    the slip increment, slip = radial return to the cone with its
    consistent nonsymmetric tangent, the anchored tangential force
    committed once per converged increment;
  * TWO-PHASE loading on the ram: the press force F_N = 4 kN ramps over
    load factor 0 -> 0.5 and holds; the lateral push F_T = 0.8 kN ramps
    over 0.5 -> 1. F_T < mu F_N = 1.2 kN: the clamp HOLDS.

What to look at:
  * phase 1 increments look exactly like the implicit_press example
    (normal contact only, the friction anchors stay empty);
  * phase 2: the punch creeps sideways by the pad's elastic shear plus
    the interface MICRO-SLIP F_T/(sum K_t) of the penalty-regularized
    stick state, then STOPS — every pair sticks, and the transmitted
    shear equals the applied 0.8 kN exactly (the pad's sig_zx integrates
    to it);
  * push it ABOVE the cone (F_T = 1.5 > mu F_N) and the run FAILS LOUDLY
    — there is no static equilibrium for a sliding clamp: Newton cannot
    converge and the imp_dt.F step control cuts to its floor (or the
    tangent goes singular: a fully sliding surface has no tangential
    stiffness left). That failure is the friction cone doing its job —
    the M13 validations assert both regimes' closed forms
    (tests/test_m13_implfric.py).

Run it (from this directory):
    python generate_deck.py
    pyradioss-starter -i IMPLCLAMP_0000.rad
    pyradioss-engine  -i IMPLCLAMP_0001.rad
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


# pad
PLX, PLY, PLZ = 40.0, 40.0, 10.0
PNX, PNY, PNZ = 4, 4, 2
# punch: same non-grid-aligned layout as implicit_press (see that
# generator's docstring for the mesh-alignment pathology it avoids)
ULX, ULY, ULZ = 15.0, 15.0, 10.0
UNX, UNY, UNZ = 3, 3, 1
OFFX, OFFY = 1.7, 2.3
PEN0 = 0.005                       # initial bite into the contact gap (mm)
GAP = 0.05                         # /INTER/TYPE7 constant gap (mm)
FN = 4.0                           # press force (kN), phase 1
FT = 0.8                           # lateral push (kN), phase 2 (< mu FN)
MU = 0.3                           # Coulomb friction coefficient

here = os.path.dirname(os.path.abspath(__file__))


def main():
    lines = ["#RADIOSS STARTER", "/BEGIN",
             "IMPLCLAMP - friction clamp: punch pushed below the cone"]

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
    z0 = PLZ + GAP - PEN0
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
    lines += ["/SURF/PART/1", "pad surface", "         1"]

    # ram guide: y + rotations held; z stroke AND x slide free — only
    # friction can hold the x direction
    lines += ["/RBE2/1", "press ram",
              f"{ram:10d}         3",
              "/BCS/1", "pad base clamp",
              "       111       000         0         1",
              "/BCS/2", "ram guide",
              "       010       111         0         4"]

    lines += ["/INTER/TYPE7/1", "punch vs pad (Coulomb)",
              "         2         1         5         0",
              f"       1.0{MU:10.3f}{GAP:10.3f}       0.0"]

    # two-phase loading: press to FN over lam 0 -> 0.5, hold; push to FT
    # over lam 0.5 -> 1
    lines += ["/FUNCT/1", "press ramp then hold",
              "       0.0       0.0",
              "       0.5       1.0",
              "       2.0       1.0",
              "/FUNCT/2", "push after the press",
              "       0.0       0.0",
              "       0.5       0.0",
              "       1.0       1.0",
              "       2.0       1.0",
              "/CLOAD/1", "press force",
              f"         1         Z         4{-FN:10.4f}",
              "/CLOAD/2", "lateral push",
              f"         2         X         4{FT:10.4f}"]

    lines.append("/END")
    deck_writer.write_starter_from_port_lines(
        lines, os.path.join(here, "IMPLCLAMP_0000.rad"), runname="IMPLCLAMP")

    engine = """#
/RUN/IMPLCLAMP/1
1.0
/IMPL
/IMPL/DTINI
0.05
/PRINT/1
/ANIM/DT
1.0
/END
"""
    deck_writer.write_engine_from_port_lines(
        engine.splitlines(), os.path.join(here, "IMPLCLAMP_0001.rad"))
    print("wrote IMPLCLAMP_0000.rad / IMPLCLAMP_0001.rad")


if __name__ == "__main__":
    main()
