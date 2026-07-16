#!/usr/bin/env python3
"""
Generate the COMPLEX_MODES example deck: the COMPLEX / DAMPED eigenvalues and
the damped complex FRF of a NON-CLASSICALLY-damped spring-mass chain — the
first pyradioss complex-modal analysis (M18, /IMPL/CEIGV), built on the QEP /
state-space linearization of implicit/complex_modal.py.

Why a spring-mass chain with ONE dashpot? It is the canonical demonstrator of
NON-PROPORTIONAL (non-classical) damping (Clough & Penzien, "Dynamics of
Structures", ch. 13). A fixed-free chain of N equal springs and masses has, if
undamped, real modes; add a SINGLE viscous dashpot on the first link only and
the damping matrix C is no longer proportional to M or K — the real modes no
longer diagonalize it, the free-vibration modes go COMPLEX, and different
masses reach their extremes at different instants (a DOF-to-DOF PHASE LAG).
The M17 real-mode superposition, which assumes classical (diagonal modal)
damping, CANNOT represent this; the M18 complex modes can.

Model (units mm / ms / kg -> forces kN, frequencies kHz):
  * N = 4 links: node 1 pinned, nodes 2..5 free along +x, one /SPRING per link
    (its own /PART + /PROP/SPRING 'M K C');
  * every link: mass M = 2e-3 kg (half lumps onto each node), stiffness
    K = 1000 kN/mm;
  * a viscous dashpot C = 0.6 on the FIRST link ONLY (the "damped anchor") —
    the localized, non-classical damper. Every other link is undamped;
  * a unit harmonic force at the tip (node 5) for the /IMPL/CEIGV/FRF sweep.

What /IMPL/CEIGV does (a PORT card — the open-source engine has NO complex/
damped eigensolver; freimpl.F reads only DYNA / BUCKL / DT / NONLIN / ARCL):
  assemble (K, C, M) — the M16 consistent mass, the M8 tangent, and the M18
  assembled damping C = discrete dashpots (+ any Rayleigh a M + b K) — and
  solve the quadratic eigenproblem (lambda^2 M + lambda C + K) phi = 0 through
  the SYMMETRIC state-space linearization A z = lambda B z. Each under-damped
  mode reports as

      lambda_i = -zeta_i omega_i +/- i omega_i sqrt(1 - zeta_i^2)

  i.e. a DECAY RATE Re(lambda) = -zeta_i omega_i, a DAMPED frequency
  Im(lambda) = omega_{d,i}, and a COMPLEX mode shape (the phase lag).

What to look for in the listing (COMPLEX_MODES_0001.out):
  * the "** COMPLEX / DAMPED EIGENVALUES ** (/IMPL/CEIGV)" block prints, per
    mode, the natural frequency, the damped frequency, the damping ratio zeta
    and the decay rate. The localized dashpot damps the modes UNEQUALLY (the
    low modes, which stretch the first link most, damp harder) — the
    fingerprint of non-classical damping; a Rayleigh (classical) C would give
    a smooth zeta(omega) curve instead;
  * the "** COMPLEX FREQUENCY RESPONSE ** (/IMPL/CEIGV/FRF)" block sweeps the
    tip drive: the damped complex transfer function U(Omega) (stored on
    model.implicit_result.complex_frf) has FINITE resonant peaks (the poles
    sit at the complex lambda_i — a decaying FRF, not the undamped singular
    one).

Run it (from this directory):
    python generate_deck.py          # regenerates the two .rad files
    pyradioss-starter -i COMPLEX_MODES_0000.rad
    pyradioss-engine  -i COMPLEX_MODES_0001.rad
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


NLINK = 4            # spring-mass links
DX = 10.0            # node spacing (mm)
MASS = 2.0e-3        # per-link mass (kg) — half lumps onto each node
K = 1000.0           # per-link stiffness (kN/mm)
CDAMP = 0.6          # dashpot on the FIRST link only (non-classical)
NMODE = 4            # complex modes to extract
FMIN = 0.0           # FRF sweep start (kHz)
FMAX = 0.4           # FRF sweep end (kHz) — past the highest mode
NF = 2000            # sweep points

here = os.path.dirname(os.path.abspath(__file__))


def main():
    lines = ["#RADIOSS STARTER", "/BEGIN",
             "COMPLEX_MODES - non-classically-damped spring-mass chain "
             "(pyradioss /IMPL/CEIGV)"]

    # ---- nodes: chain along +x --------------------------------------------
    lines.append("/NODE")
    for i in range(NLINK + 1):
        lines.append(f"{i + 1:10d}{i * DX:20.10f}{0.0:20.10f}{0.0:20.10f}")

    # ---- one /SPRING (+ /PART + /PROP) per link ---------------------------
    for i in range(NLINK):
        lines.append(f"/SPRING/{i + 1}")
        lines.append(f"{i + 1:10d}{i + 1:10d}{i + 2:10d}")
    for i in range(NLINK):
        lines += [f"/PART/{i + 1}", f"link{i + 1}", f"{i + 1:10d}         1"]

    lines += ["/MAT/LAW1/1", "steel elastic", "   7.8e-6",
              "     210.0       0.0"]
    for i in range(NLINK):
        c = CDAMP if i == 0 else 0.0        # dashpot on the first link only
        lines += [f"/PROP/SPRING/{i + 1}", f"sp{i + 1}",
                  f"{MASS:12g}{K:12g}{c:12g}"]

    # ---- constraints: node 1 pinned, free nodes axial (x) only ------------
    free = "\n".join(str(i + 2) for i in range(NLINK))
    lines += ["/GRNOD/NODE/1", "pin", "1",
              "/GRNOD/NODE/2", "free", free,
              "/GRNOD/NODE/3", "tip", f"{NLINK + 1}",
              "/BCS/1", "pin base",
              "       111       111         0         1",
              "/BCS/2", "axial only (free nodes)",
              "       011       111         0         2"]

    # ---- a unit harmonic force at the tip (the FRF drive pattern) ---------
    lines += ["/FUNCT/1", "unit amplitude", "       0.0       1.0",
              "  100000.0       1.0",
              "/CLOAD/1", "tip axial drive",
              "         1         X         3       1.0"]
    lines.append("/END")

    deck_writer.write_starter_from_port_lines(
        lines, os.path.join(here, "COMPLEX_MODES_0000.rad"), runname="COMPLEX_MODES")

    # ---- engine: a bare implicit static step + the complex-modal analysis -
    engine = [
        "#RADIOSS ENGINE",
        "/RUN/COMPLEX_MODES/1",
        "1.0",                       # one (zero-load) static increment
        "/IMPL",                     # implicit static driver hosts the analysis
        "/IMPL/CEIGV",               # M18: complex / damped eigenvalues
        f"{NMODE}",
        "/IMPL/CEIGV/FRF",           # M18: the damped complex FRF sweep
        f"{FMIN}  {FMAX}  {NF}  {NMODE}",
        "/PRINT/-500",
        "/STOP",
        "15.0",
    ]
    deck_writer.write_engine_from_port_lines(
        engine, os.path.join(here, "COMPLEX_MODES_0001.rad"))
    print("wrote COMPLEX_MODES_0000.rad / COMPLEX_MODES_0001.rad")


if __name__ == "__main__":
    main()
