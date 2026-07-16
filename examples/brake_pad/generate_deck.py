#!/usr/bin/env python3
"""
Generate the BRAKEPAD example: the M13 implicit friction clamp with a
PRESSURE-DEPENDENT friction MODEL (Milestone 15) — /INTER/TYPE7 with
Ifric = 1 (MFROT 1, the generalized viscous law of i7for3.F, here with
its pressure terms only: mu(p) = mu0 + C1*p) solved inside the implicit
Newton loop with the consistent mu'(p) coupling tangent
(implicit/contact.py).

Model (units mm / ms / kg -> forces kN, stresses GPa) — the
implicit_clamp geometry verbatim:
  * PAD: a 40 x 40 x 10 mm polymer block (E = 2 GPa), base clamped;
  * PUNCH: a 15 x 15 x 10 mm steel block on an /RBE2 ram whose guide
    holds y + all rotations but leaves the z stroke AND the x slide free
    — only FRICTION can hold the x direction;
  * TWO-PHASE loading: press F_N = 4 kN over load factor 0 -> 0.5, then
    a lateral push F_T = 0.8 kN over 0.5 -> 1.

The POINT of this example — "a brake pad grips harder the harder you
press": the friction card is

    mu(p) = mu0 + C1 * p     with mu0 = 0.15,  C1 = 80 /GPa

  * with the CONSTANT mu0 alone the clamp would SLIP: mu0 * F_N =
    0.6 kN < F_T = 0.8 kN — no static equilibrium exists, and indeed
    setting Ifric = 0 makes this deck fail loudly exactly like the
    over-the-cone variant discussed in implicit_clamp's docstring;
  * the press pressure (each active pair carries p = f_n / A of its
    main segment — the i7for3.F pressure definition) lifts the
    coefficient to mu(p) ~ 0.35..0.4, the cone grows to
    sum mu(p_i) f_ni ~ 1.4 kN > F_T, and the clamp HOLDS: the punch
    creeps by the pad shear + the stick micro-slip and stops, the pad's
    integrated shear resultant equal to the applied 0.8 kN.

Under the implicit solver the MFROT law is evaluated at its STATIC
LIMIT mu(p, v = 0) — the port's rate-device convention (the listing
says so when a deck carries velocity coefficients; this one does not),
and the Newton tangent carries the mu'(p) = C1 coupling block — the
cone RADIUS moves with the same unknowns as the normal force, and the
consistent linearization is what keeps the phase-2 increments at 3-5
iterations (see PORTING_GUIDE M15 and tests/test_m15_fricmat.py for the
closed-form validations).

The same card runs under the EXPLICIT engine too (i7for3.F's kinetic
law with mu(p, v) and optional IFQ filtering — see contact/friction.py).

Run it (from this directory):
    python generate_deck.py
    pyradioss-starter -i BRAKEPAD_0000.rad
    pyradioss-engine  -i BRAKEPAD_0001.rad
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
GAP = 0.05                        # /INTER/TYPE7 constant gap (mm)
FN = 4.0                          # press force (kN), phase 1
FT = 0.8                          # lateral push (kN), phase 2
MU0 = 0.15                        # base friction: mu0*FN = 0.6 < FT !
C1 = 80.0                         # pressure coefficient (1/GPa)

here = os.path.dirname(os.path.abspath(__file__))


def main():
    lines = ["#RADIOSS STARTER", "/BEGIN",
             "BRAKEPAD - pressure-dependent friction clamp (MFROT 1)"]

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

    # the M15 friction-model card: Ifric = 1 (MFROT 1) on card 2,
    # C1..C6 on the optional card 4 — mu(p) = MU0 + C1*p (pressure
    # terms only: C2/C3/C5 velocity coefficients zero, so the implicit
    # static limit IS the whole law)
    lines += ["/INTER/TYPE7/1", "punch vs pad (mu grows with pressure)",
              "         2         1         5         0         0"
              "         1         0",
              f"       1.0{MU0:10.3f}{GAP:10.3f}       0.0       0.0",
              f"{C1:10.3f}       0.0       0.0       0.0       0.0"
              "       0.0"]

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
        lines, os.path.join(here, "BRAKEPAD_0000.rad"), runname="BRAKEPAD")

    engine = """#
/RUN/BRAKEPAD/1
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
        engine.splitlines(), os.path.join(here, "BRAKEPAD_0001.rad"))
    print("wrote BRAKEPAD_0000.rad / BRAKEPAD_0001.rad")


if __name__ == "__main__":
    main()
