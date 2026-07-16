#!/usr/bin/env python3
"""
Generate the IMPLICIT_CANTILEVER example: a shell cantilever solved with the
M8 IMPLICIT-static Newton driver (not the explicit time loop).

Model (units mm / ms / kg -> forces kN, stresses GPa):
  * a 100 x 10 x 1 mm steel plate cantilever, meshed 20 x 2 with /SHELL BT4
  * clamped at x = 0 (/BCS 111 111), a transverse (z) tip load ramped to
    F = 1e-3 kN over the "load factor" 0 -> 1
  * LAW1 elastic steel (E = 210 GPa, nu = 0.3)

The engine deck carries a /IMPL card, so `pyradioss-engine` runs the
Newton–Raphson static solver instead of the leap-frog loop: the run's final
"time" is the LOAD FACTOR, and the listing prints one line per load increment
(load factor, Newton iterations, residual norm) instead of time cycles.

Closed-form check (thin cantilever, Euler–Bernoulli):
    w_tip = F L^3 / (3 E I),   I = b t^3 / 12
  ~ 1.90 mm — the run reaches it to well under 1 % with this mesh, in a
  single Newton step (linear elastic).

Run it (from this directory):
    python generate_deck.py
    pyradioss-starter -i IMPLCANT_0000.rad
    pyradioss-engine  -i IMPLCANT_0001.rad          # add -linsolve cholmod
                                                    # if scikit-sparse is set up
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


LX, B, T = 100.0, 10.0, 1.0        # plate length, width, thickness (mm)
NX, NY = 20, 2                     # shell mesh
F = 1.0e-3                         # total tip load (kN)

here = os.path.dirname(os.path.abspath(__file__))


def nid(i, j):
    return 1 + i + j * (NX + 1)


def main():
    lines = ["#RADIOSS STARTER", "/BEGIN",
             "IMPLCANT - shell cantilever, implicit statics (pyradioss)"]

    lines.append("/NODE")
    for j in range(NY + 1):
        for i in range(NX + 1):
            x, y = i * LX / NX, j * B / NY
            lines.append(f"{nid(i, j):10d}{x:20.10f}{y:20.10f}{0.0:20.10f}")

    lines.append("/SHELL/1")
    eid = 0
    for j in range(NY):
        for i in range(NX):
            eid += 1
            n = [nid(i, j), nid(i + 1, j), nid(i + 1, j + 1), nid(i, j + 1)]
            lines.append(f"{eid:10d}" + "".join(f"{v:10d}" for v in n))

    lines += ["/PART/1", "plate", "         1         1",
              "/MAT/LAW1/1", "steel elastic", "   7.8e-6",
              "     210.0       0.3",
              "/PROP/SHELL/1", "shell",
              # short form: Thick  N(integration points)  hm(hourglass)
              "       1.0         5      0.01"]

    fixed = [nid(0, j) for j in range(NY + 1)]
    tip = [nid(NX, j) for j in range(NY + 1)]
    lines += ["/GRNOD/NODE/1", "clamped edge x=0",
              " ".join(map(str, fixed)),
              "/GRNOD/NODE/2", "tip edge x=L",
              " ".join(map(str, tip))]

    lines += ["/BCS/1", "clamp all 6 DOFs at x=0",
              "       111       111         0         1"]
    # ramp the load with the load factor (the implicit 'time'): f(t)=t
    lines += ["/FUNCT/1", "load-factor ramp",
              "       0.0       0.0", "     100.0     100.0"]
    lines += ["/CLOAD/1", "transverse tip load in Z",
              f"         1         Z         2       {F / len(tip)}"]
    lines.append("/END")

    deck_writer.write_starter_from_port_lines(
        lines, os.path.join(here, "IMPLCANT_0000.rad"), runname="IMPLCANT")

    engine = [
        "#RADIOSS ENGINE",
        "/RUN/IMPLCANT/1",
        "1.0",                     # final LOAD FACTOR (not a physical time)
        "/IMPL/DTINI",             # implicit statics; load-factor increment
        "0.5",                     # two increments of 0.5 (elastic: 1 step each)
        "/IMPL/NEWTON",
        "1e-6  25",                # residual tolerance, max Newton iterations
        "/END",
    ]
    deck_writer.write_engine_from_port_lines(
        engine, os.path.join(here, "IMPLCANT_0001.rad"))

    print("wrote IMPLCANT_0000.rad / IMPLCANT_0001.rad")


if __name__ == "__main__":
    main()
