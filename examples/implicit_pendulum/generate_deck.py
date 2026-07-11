#!/usr/bin/env python3
"""
Generate the IMPLICIT_PENDULUM example: a large-rotation truss pendulum
integrated by the M10 IMPLICIT DYNAMICS driver (/IMPL/DYNA, Newmark
trapezoidal rule) with the M9 nonlinear geometry (/IMPL/NONLIN).

Model (units mm / ms / kg -> forces kN, stresses GPa):
  * one stiff /TRUSS bar (A = 1 mm^2, LAW1 steel E = 210 GPa), pivot node
    pinned at the origin, tip node at 60 degrees from the vertical
    (L = 100 mm), released from REST under /GRAV gravity
  * /IMPL/DYNA/2 with gamma = 0.5, beta = 0.25 — the trapezoidal rule:
    unconditionally stable, second-order, NO numerical dissipation (the
    energy ledger in the listing stays closed; swap the card for
    /IMPL/DYNA/1 with a negative alpha to watch HHT drain it instead)
  * /IMPL/NONLIN — the swing is a 120-degree rotation: small-displacement
    geometry would be nonsense here

The closed form this reproduces: the large-amplitude pendulum period

    T = 4 sqrt(L/g) K(m),   m = sin^2(theta0/2)

(K = complete elliptic integral of the first kind). At theta0 = 60 deg
that is 7.3% LONGER than the small-angle 2 pi sqrt(L/g) — a genuinely
nonlinear effect: the tip crosses the vertical at exactly T/4 (~170 ms
here, vs ~157 ms for the linear pendulum), swings through to -60 degrees
and returns. The bar is effectively inextensible (EA/(m g) ~ 5e6), so the
tip radius stays at L to a few ppm through the whole rotation — the
corotational truss + updated-Lagrangian frame doing exact large-rotation
kinematics. tests/test_m10_impdyn.py asserts the quarter period against
the elliptic integral to 0.5%.

Why this needs M10: the explicit leapfrog would need dt <= L/c ~ 0.02 ms
(the bar's WAVE transit — physics the pendulum problem does not care
about), i.e. ~35,000 cycles per swing; the implicit trapezoidal rule
walks it in ~240 steps of dt = 3.2 ms — 150x the explicit stability
limit, stable because the unresolved axial mode cannot blow up — at 3-4
Newton iterations per step.

Run it (from this directory):
    python generate_deck.py
    pyradioss-starter -i IMPLPEND_0000.rad
    pyradioss-engine  -i IMPLPEND_0001.rad     (needs SciPy)
"""

import math
import os

L = 100.0            # pendulum length (mm)
G = 9.81e-3          # gravity (mm/ms^2)
E = 210.0            # GPa
AREA = 1.0           # mm^2
RHO = 7.8e-6         # kg/mm^3
TH0 = math.pi / 3.0  # 60 deg release angle

here = os.path.dirname(os.path.abspath(__file__))


def _ellipk(m, tol=1e-14):
    """Complete elliptic integral of the first kind K(m) by the
    arithmetic-geometric mean (no SciPy needed to GENERATE the deck)."""
    a, b = 1.0, math.sqrt(1.0 - m)
    while abs(a - b) > tol * a:
        a, b = 0.5 * (a + b), math.sqrt(a * b)
    return math.pi / (2.0 * a)


def main():
    om0 = math.sqrt(G / L)
    t_lin = 2.0 * math.pi / om0
    t_nl = 4.0 / om0 * _ellipk(math.sin(TH0 / 2.0) ** 2)
    dt = t_lin / 200.0
    t_end = 1.1 * t_nl                      # one full swing and a bit

    x2, y2 = L * math.sin(TH0), -L * math.cos(TH0)
    starter = f"""\
#RADIOSS STARTER
/BEGIN
IMPLICIT_PENDULUM - large-rotation swing, /IMPL/DYNA + /IMPL/NONLIN (M10)
/NODE
         1{0.0:20.10f}{0.0:20.10f}{0.0:20.10f}
         2{x2:20.10f}{y2:20.10f}{0.0:20.10f}
/TRUSS/1
         1         1         2
/PART/1
pendulum bar
         1         1
/MAT/LAW1/1
steel (the bar is effectively rigid on the gravity scale)
   {RHO}
     {E}       0.0
/PROP/TRUSS/1
bar section
       {AREA}
/GRNOD/NODE/1
pivot
1
/GRNOD/NODE/2
tip
2
/BCS/1
pin the pivot
       111       000         0         1
/BCS/2
keep the swing in the x-y plane
       001       000         0         2
/FUNCT/1
constant gravity switch-on
       0.0       1.0
   1.0e30       1.0
/GRAV/1
g, minus-y
         1        Y         0      {-G}
/END
"""
    engine = f"""\
# IMPLICIT DYNAMICS (M10): Newmark trapezoidal rule + nonlinear geometry.
# The /RUN time and /IMPL/DTINI are PHYSICAL (ms) — unlike implicit
# statics, where they are the load factor. Expected listing: ~{int(t_end/dt)}
# steps, 2-3 Newton iterations each, energy balance ~0% at termination;
# the tip crosses the vertical at t = T/4 = {t_nl/4.0:.1f} ms (the
# elliptic-integral period — the linear pendulum would say {t_lin/4.0:.1f}).
/RUN/IMPLPEND/1
{t_end}
/VERS/2026
/IMPL/DYNA/2
0.5  0.25
/IMPL/NONLIN
/IMPL/DTINI
{dt}
/IMPL/NEWTON
1e-8  25
/PRINT/-100
/END
"""
    with open(os.path.join(here, "IMPLPEND_0000.rad"), "w") as f:
        f.write(starter)
    with open(os.path.join(here, "IMPLPEND_0001.rad"), "w") as f:
        f.write(engine)
    print("wrote IMPLPEND_0000.rad / IMPLPEND_0001.rad")
    print(f"  linear period 2 pi sqrt(L/g)     : {t_lin:9.2f} ms")
    print(f"  elliptic (60 deg) period         : {t_nl:9.2f} ms  "
          f"(+{(t_nl / t_lin - 1) * 100:.1f}%)")
    print(f"  time step                        : {dt:9.3f} ms  "
          f"(explicit would need ~{L / math.sqrt(E / RHO):.4f})")


if __name__ == "__main__":
    main()
