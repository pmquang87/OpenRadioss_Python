#!/usr/bin/env python3
"""
Generate the SNAP_CATCH example: the M9 von Mises snap-through ARRESTED
mid-snap by an /INTER/TYPE7 contact stop — arc length WITH CONTACT in the
corrector (M14; combining /IMPL/ARCL with contact or constraints was a
loud M12 refusal).

Model (units mm / ms / kg -> forces kN, stresses GPa):
  * the snap_through example's two-bar shallow truss (a = 100 mm half
    span, h = 20 mm rise, A = 1 mm^2, LAW1 steel), apex loaded downward
    by a PROPORTIONAL ramp to ~1.3x the closed-form limit load;
  * a clamped elastic pad (one /BRICK) parked below the apex's snap path:
    its top face at y = -19 mm with a 4 mm contact gap, /INTER/TYPE7
    (Istf = 1, K = 50 kN/mm) between the apex node and that face.

What to watch in the listing: the arc trace climbs to the limit point
(~0.63 kN), turns the corner (the load factor DECREASES, even negative,
down the unstable branch — pure load control can never sample that), and
then, instead of recovering on the FREE far branch near y = -24 mm, the
apex lands on the stop at y ~= -15 mm: the active set changes INSIDE the
corrector iterations, the path stiffens sharply, and the load factor
climbs back to 1.0 with the apex resting ON the pad. The final height is
the closed-form equilibrium of truss + penalty spring,

    P_end = P(y) + K*(gap - (y - y_pad)),
    P(y)  = -2 E A ln(L/L0) y / L

(~ -15.3 mm here — well ABOVE the free far-branch answer), asserted to
1e-5 mm by tests/test_m14_implgen.py::test_arclength_snap_caught_by_
contact_stop.

Run it (from this directory):
    python generate_deck.py
    pyradioss-starter -i SNAPC_0000.rad
    pyradioss-engine  -i SNAPC_0001.rad
"""

import math
import os

A_HALF = 100.0      # half-span a (mm)
H = 20.0            # apex rise h (mm)
AREA = 1.0          # bar cross-section (mm^2)
E = 210.0           # GPa
Y_PAD = -19.0       # pad top face (mm)
GAP = 4.0           # contact gap (mm): the apex touches at y = -15
KC = 50.0           # penalty stiffness (kN/mm)

here = os.path.dirname(os.path.abspath(__file__))


def p_analytic(y):
    L0 = math.sqrt(A_HALF ** 2 + H ** 2)
    L = math.sqrt(A_HALF ** 2 + y ** 2)
    return -2.0 * E * AREA * math.log(L / L0) * y / L


def main():
    p_max = max(p_analytic(H * k / 2000.0) for k in range(1, 2000))
    p_end = 1.3 * p_max

    yb = Y_PAD - 10.0
    lines = ["#RADIOSS STARTER", "/BEGIN",
             "SNAPC - snap-through caught by a TYPE7 stop (arc length, M14)"]
    lines.append("/NODE")
    lines.append(f"{1:10d}{-A_HALF:20.10f}{0.0:20.10f}{0.0:20.10f}")
    lines.append(f"{2:10d}{A_HALF:20.10f}{0.0:20.10f}{0.0:20.10f}")
    lines.append(f"{3:10d}{0.0:20.10f}{H:20.10f}{0.0:20.10f}")
    for nid, (x, y, z) in ((11, (-10.0, yb, -10.0)), (12, (10.0, yb, -10.0)),
                           (13, (10.0, yb, 10.0)), (14, (-10.0, yb, 10.0)),
                           (15, (-10.0, Y_PAD, -10.0)),
                           (16, (10.0, Y_PAD, -10.0)),
                           (17, (10.0, Y_PAD, 10.0)),
                           (18, (-10.0, Y_PAD, 10.0))):
        lines.append(f"{nid:10d}{x:20.10f}{y:20.10f}{z:20.10f}")
    lines += ["/TRUSS/1",
              "         1         1         3",
              "         2         2         3",
              "/PART/1", "bars", "         1         1",
              "/MAT/LAW1/1", "elastic bar", "   7.8e-6",
              f"     {E}       0.0",
              "/PROP/TRUSS/1", "bar", f"       {AREA}"]
    lines += ["/BRICK/2",
              "         1        11        14        13        12"
              "        15        18        17        16",
              "/PART/2", "pad", "         2         2",
              "/MAT/LAW1/2", "pad", "   7.8e-6", "     210.0       0.0",
              "/PROP/SOLID/2", "solid", "       1.1      0.05       0.1"]
    lines += ["/GRNOD/NODE/1", "pinned supports", "1 2",
              "/GRNOD/NODE/2", "apex", "3",
              "/GRNOD/NODE/11", "pad nodes", "11 12 13 14 15 16 17 18",
              "/GRNOD/NODE/12", "apex secondary", "3"]
    lines += ["/BCS/1", "pin the supports",
              "       111       000         0         1",
              "/BCS/2", "apex stays in the xy plane",
              "       001       000         0         2",
              "/BCS/11", "clamp the pad",
              "       111       111         0        11"]
    lines += ["/SURF/SEG/1", "pad top face (+y)",
              "        15        16        17        18",
              "/INTER/TYPE7/1", "the stop",
              "        12         1         1         0",
              f"    {KC:6.1f}     0.000     {GAP:5.2f}       0.0"]
    lines += ["/FUNCT/1", "proportional load-factor ramp",
              "       0.0       0.0", "     100.0     100.0"]
    lines += ["/CLOAD/1", "push the apex down",
              f"         1         Y         2       {-p_end}"]
    lines.append("/END")
    with open(os.path.join(here, "SNAPC_0000.rad"), "w") as fh:
        fh.write("\n".join(lines) + "\n")

    engine = [
        "#RADIOSS ENGINE",
        "/RUN/SNAPC/1",
        "1.0",                    # final LOAD FACTOR
        "/IMPL/ARCL",             # Riks/Crisfield arc length (implies NONLIN)
        "/IMPL/DTINI",
        "0.05",
        "/IMPL/NEWTON",
        "1e-9  40",
        "/END",
    ]
    with open(os.path.join(here, "SNAPC_0001.rad"), "w") as fh:
        fh.write("\n".join(engine) + "\n")
    # report the closed-form arrested height
    y = Y_PAD
    for _ in range(200):
        y = y + 1e-4
        if p_analytic(y) + KC * (GAP - (y - Y_PAD)) <= p_end:
            break
    print(f"wrote SNAPC_0000.rad / SNAPC_0001.rad "
          f"(P_max = {p_max:.4f} kN, P_end = {p_end:.4f} kN)")


if __name__ == "__main__":
    main()
