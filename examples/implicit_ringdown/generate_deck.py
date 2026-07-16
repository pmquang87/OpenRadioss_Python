#!/usr/bin/env python3
"""
Generate the IMPLICIT_RINGDOWN example: a MIXED-ELEMENT implicit dynamic
ring-down under /IMPL/DYNA/DAMP Rayleigh damping — the two headline M11
capabilities in one deck.

Model (units mm / ms / kg -> forces kN, stresses GPa): a braced
instrument mast built from FOUR element families at once — possible for
the implicit solver only since M11, when the element tangent set became
COMPLETE (tetra4 / sh3n / beam / spring joined hexa8 / BT4 / truss):

  * /BEAM mast — four corotational Timoshenko elements, 10x10 mm square
    section, clamped at the base (the M11 beam 12x12 tangent);
  * /SH3N antenna panel — a triangle-meshed plate cantilevered off the
    top two mast nodes, sharing their ROTATIONAL DOFs (the M11 C0
    triangle tangent; the dofmap numbers beam rotations since M11 too);
  * /TRUSS diagonal brace from a ground anchor to the mast mid-height
    (the M9 corotational truss tangent);
  * /SPRING horizontal guy from the mast tip to a wall anchor (the M11
    TYPE4 spring tangent — a total-form element with its own implicit
    residual path).

A step force Fx (constant /CLOAD from t = 0) kicks the mast sideways.
With /IMPL/DYNA/2 (trapezoidal Newmark: no numerical dissipation) the
structure would ring around the static answer FOREVER — the energy
ledger stays closed. /IMPL/DYNA/DAMP adds the classical Rayleigh matrix

    C = a M + b K        (imp_dyna.F IDY_DAMP: DAMPA_IMP, DAMPB_IMP)

so the oscillation DECAYS onto the static deflection: watch the listing's
tip displacement settle and the RAYLEIGH DISSIPATION line of the
termination page absorb the transient's energy while ENERGY BALANCE
stays ~0% — the dissipation is BOOKED, not lost (the DY_EDAMP ledger).
The damped SDOF closed-form validations live in
tests/test_m11_implcomp.py; this deck is the same physics on a real
mixed structure.

Expected result: the tip first overshoots to ~1.38 mm, rings at the
~13 ms first sway period and settles onto the STATIC answer of the SAME
deck run with /IMPL alone (~0.82 mm — within 0.2% after the 120 ms run);
the a/b pair below is tuned near zeta ~ 8% on the first sway mode
(omega_1 ~ 0.47 rad/ms measured), so the overshoot decays by roughly
e^(-2 pi zeta) ~ 0.6 per cycle, the RAYLEIGH DISSIPATION ledger ends at
~2.0e-2 kN.mm and the ENERGY BALANCE at 0.00%.

Run it (from this directory):
    python generate_deck.py
    pyradioss-starter -i IMPLRING_0000.rad
    pyradioss-engine  -i IMPLRING_0001.rad     (needs SciPy)
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


# ---- geometry / material ----------------------------------------------------
H = 400.0            # mast height (mm), 4 beam elements
NB = 4
BA = 100.0           # beam section area 10x10 (mm^2)
BI = 833.33          # beam bending inertia b h^3/12 (mm^4)
BJ = 1405.8          # torsion constant of the square section (mm^4)
PW = 150.0           # panel width in x (mm)
PT = 2.0             # panel thickness (mm)
KSPR = 0.02          # guy spring stiffness (kN/mm)
MSPR = 0.01          # guy spring lumped mass (kg)
TA = 25.0            # brace truss area (mm^2)
FX = 0.05            # step tip force (kN)

# ---- Rayleigh damping tuned near the first sway mode ------------------------
# a quick starter run puts the first mode near omega ~ 0.5 rad/ms; zeta ~ 8%
OMEGA1 = 0.5
ZETA = 0.08
DAMP_A = ZETA * OMEGA1          # mass-proportional half
DAMP_B = ZETA / OMEGA1          # stiffness-proportional half

T_END = 120.0        # ms — several damped sway periods
DT = 0.25            # ms — implicit step (the sway period is ~13 ms)

nodes = []
nid = 0


def node(x, y, z):
    global nid
    nid += 1
    nodes.append(f"{nid:10d}{x:20.10f}{y:20.10f}{z:20.10f}")
    return nid


# mast nodes 1..5 (base -> tip)
mast = [node(0.0, 0.0, H * i / NB) for i in range(NB + 1)]
# beam orientation node (off-axis)
orient = node(300.0, 300.0, 0.0)
# panel: two extra columns of nodes at x = PW/2, PW spanning the top segment
zs = [H * (NB - 1) / NB, H]                       # the top two mast heights
pan_mid = [node(PW / 2.0, 0.0, z) for z in zs]
pan_out = [node(PW, 0.0, z) for z in zs]
# anchors
brace_gnd = node(200.0, 0.0, 0.0)
guy_anchor = node(200.0, 0.0, H)

beams = [f"{i+1:10d}{mast[i]:10d}{mast[i+1]:10d}{orient:10d}"
         for i in range(NB)]
# panel triangles over the 2x1 strip of quads (mast col, mid col, out col)
tris = []
eid = 100
cols = [[mast[NB - 1], mast[NB]], pan_mid, pan_out]
for c in range(2):
    a, b = cols[c], cols[c + 1]
    eid += 1
    tris.append(f"{eid:10d}{a[0]:10d}{b[0]:10d}{b[1]:10d}")
    eid += 1
    tris.append(f"{eid:10d}{a[0]:10d}{b[1]:10d}{a[1]:10d}")

starter = f"""\
#RADIOSS STARTER
/BEGIN
IMPLRING
/NODE
{chr(10).join(nodes)}
/BEAM/1
{chr(10).join(beams)}
/SH3N/2
{chr(10).join(tris)}
/TRUSS/3
       201{brace_gnd:10d}{mast[2]:10d}
/SPRING/4
       301{guy_anchor:10d}{mast[NB]:10d}
/PART/1
mast beams
         1         1
/PART/2
antenna panel
         2         1
/PART/3
diagonal brace
         3         1
/PART/4
guy spring
         4         1
/MAT/LAW1/1
steel
   7.8e-6
     210.0       0.3
/PROP/BEAM/1
mast 10x10
     {BA}    {BI}    {BI}    {BJ}
/PROP/SHELL/2
panel
         1         0         0         0
      0.01      0.01      0.01         0         0
         3         0       {PT}
/PROP/TRUSS/3
brace
      {TA}
/PROP/SPRING/4
guy
    {MSPR}    {KSPR}       0.0
/GRNOD/NODE/1
grounds
{mast[0]} {brace_gnd} {guy_anchor}
/GRNOD/NODE/2
mast tip
{mast[NB]}
/BCS/1
clamp grounds
       111       111         0         1
/FUNCT/1
step on at t=0
       0.0       1.0
    1000.0       1.0
/CLOAD/1
side kick
         1         X         2       {FX}
/END
"""

engine = f"""\
#RADIOSS ENGINE
/RUN/IMPLRING/1
{T_END}
/IMPL/DYNA/2
0.5  0.25
/IMPL/DYNA/DAMP
{DAMP_A}  {DAMP_B}
/IMPL/DTINI
{DT}
/IMPL/NEWTON
1e-7  25
/PRINT/-40
/END
"""

here = os.path.dirname(os.path.abspath(__file__))
deck_writer.write_starter_from_port_lines(
    starter.splitlines(), os.path.join(here, "IMPLRING_0000.rad"), runname="IMPLRING")
deck_writer.write_engine_from_port_lines(
    engine.splitlines(), os.path.join(here, "IMPLRING_0001.rad"))
print("wrote IMPLRING_0000.rad / IMPLRING_0001.rad")
