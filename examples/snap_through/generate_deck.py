#!/usr/bin/env python3
"""
Generate the SNAP_THROUGH example: a von Mises (two-bar) shallow truss traced
THROUGH its snap-through limit points by the M9 ARC-LENGTH driver.

Model (units mm / ms / kg -> forces kN, stresses GPa):
  * two /TRUSS bars (A = 1 mm^2, LAW1 steel E = 210 GPa) from pinned
    supports at (+-100, 0, 0) to the apex at (0, 20, 0) — a shallow tent,
    rise/half-span h/a = 0.2
  * a downward apex load ramping PROPORTIONALLY to P_end at load factor 1,
    with P_end ~ 1.3x the closed-form limit load

The equilibrium path of this structure is the classic N-shaped curve

    P(y) = -2 E A ln(L/L0) y / L,     L = sqrt(a^2 + y^2)

(y = current apex height; exact for the corotational log-strain truss): the
load rises to a LIMIT POINT P_max, then the structure softens — under load
control there is no NEARBY equilibrium past the peak, so Newton either
stops or leaps discontinuously to the far branch, never following the
descending path (swap /IMPL/ARCL for /IMPL/NONLIN below to watch it grind
through the jump). The /IMPL/ARCL (Riks/Crisfield)
continuation makes the load factor an unknown constrained by the step
length instead: the listing shows the load factor rising to the limit
point, DECREASING through zero (the unstable branch — even going negative),
turning at the mirrored limit point, and recovering to the full load on the
far (snapped-through, bars-in-tension) branch. Final apex height: the
closed-form far-branch root of P(y) = P_end, about y = -24 mm (it started
at +20 mm).

Run it (from this directory):
    python generate_deck.py
    pyradioss-starter -i SNAP_0000.rad
    pyradioss-engine  -i SNAP_0001.rad
"""

import math
import os

A_HALF = 100.0      # half-span a (mm)
H = 20.0            # apex rise h (mm)
AREA = 1.0          # bar cross-section (mm^2)
E = 210.0           # GPa

here = os.path.dirname(os.path.abspath(__file__))


def p_analytic(y):
    L0 = math.sqrt(A_HALF ** 2 + H ** 2)
    L = math.sqrt(A_HALF ** 2 + y ** 2)
    return -2.0 * E * AREA * math.log(L / L0) * y / L


def main():
    # closed-form limit load -> pick the full load 30 % above it
    p_max = max(p_analytic(H * k / 2000.0) for k in range(1, 2000))
    p_end = 1.3 * p_max

    lines = ["#RADIOSS STARTER", "/BEGIN",
             "SNAP - von Mises truss snap-through, arc length (pyradioss)"]
    lines.append("/NODE")
    lines.append(f"{1:10d}{-A_HALF:20.10f}{0.0:20.10f}{0.0:20.10f}")
    lines.append(f"{2:10d}{A_HALF:20.10f}{0.0:20.10f}{0.0:20.10f}")
    lines.append(f"{3:10d}{0.0:20.10f}{H:20.10f}{0.0:20.10f}")
    lines += ["/TRUSS/1",
              "         1         1         3",
              "         2         2         3",
              "/PART/1", "bars", "         1         1",
              "/MAT/LAW1/1", "elastic bar", "   7.8e-6",
              f"     {E}       0.0",
              "/PROP/TRUSS/1", "bar", f"       {AREA}"]
    lines += ["/GRNOD/NODE/1", "pinned supports", "1 2",
              "/GRNOD/NODE/2", "apex", "3"]
    lines += ["/BCS/1", "pin the supports",
              "       111       000         0         1",
              "/BCS/2", "apex stays in the xy plane",
              "       001       000         0         2"]
    # PROPORTIONAL ramp (f(lam) = lam) — the arc-length requirement
    lines += ["/FUNCT/1", "proportional load-factor ramp",
              "       0.0       0.0", "     100.0     100.0"]
    lines += ["/CLOAD/1", "push the apex down",
              f"         1         Y         2       {-p_end}"]
    lines.append("/END")
    with open(os.path.join(here, "SNAP_0000.rad"), "w") as fh:
        fh.write("\n".join(lines) + "\n")

    engine = [
        "#RADIOSS ENGINE",
        "/RUN/SNAP/1",
        "1.0",                    # final LOAD FACTOR
        "/IMPL/ARCL",             # Riks/Crisfield arc length (implies NONLIN)
        "/IMPL/DTINI",            # first increment = 5 % of the full load
        "0.05",
        "/IMPL/NEWTON",
        "1e-9  40",
        "/END",
    ]
    with open(os.path.join(here, "SNAP_0001.rad"), "w") as fh:
        fh.write("\n".join(engine) + "\n")
    print(f"wrote SNAP_0000.rad / SNAP_0001.rad "
          f"(P_max = {p_max:.4f} kN, P_end = {p_end:.4f} kN)")


if __name__ == "__main__":
    main()
