"""Generate the GASPISTON example deck (M6).

An air column in a rigid channel is compressed to half its volume by a
piston face, then held — the textbook adiabatic-compression experiment,
and the first model class that NEEDS an equation of state (M6):

* /EOS/IDEAL-GAS on a (nearly shear-free) LAW1 host: pressure comes from
  p = (gamma - 1)(1 + mu) E with the energy integrated implicitly per
  element (E-p coupling, see pyradioss/materials/eos.py). At the hold,
  p must sit on the adiabat p0 (V0/V)^gamma = 1 bar * 2^1.4 = 2.64 bar,
  and the internal energy on the corresponding isentrope value.
* /IMPDISP drives the piston face (position-exact landing at -half the
  column), the channel walls are /BCS confinement.
* /STATE/DT writes periodic restart snapshots, and the run is CHAINED:
  GASPISTON_0001.rad compresses (0 -> 8 ms), GASPISTON_0002.rad resumes
  from the _0001.rst restart and holds (8 -> 12 ms) — the M6 chaining
  contract (the resumed run continues cycles, energies and file
  numbering exactly where run 1 stopped).

Run:
    python generate_deck.py
    pyradioss-starter -i GASPISTON_0000.rad
    pyradioss-engine  -i GASPISTON_0001.rad     # compression leg
    pyradioss-engine  -i GASPISTON_0002.rad     # hold leg (chained)

Check: column N9_DZ of GASPISTONT01/T02.csv is the piston travel; the
gas element pressure is in the final listing energies (IE = E per V0 *
V0 summed) — and the run 2 listing's IE must equal
p V/(gamma-1) = 2.64e-4 * 20000 / 0.4 (units mm/ms/kg -> GPa).
"""

# M36: decks are emitted through the package's fixed-format 2022 writer
# (pyradioss/input/deck_writer.py) — model definition below is unchanged,
# only the emission format moved to the real Radioss dialect.
import os
import sys as _sys
_REPO = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))
if _REPO not in _sys.path:
    _sys.path.insert(0, _REPO)
from pyradioss.input import deck_writer  # noqa: E402

# column: 10 x 10 mm section, 40 mm tall, 4 bricks stacked in z
NZ = 4
H = 10.0          # element height (mm)
GAMMA = 1.4
P0 = 1.0e-4       # 1 bar in GPa
RHO = 1.2e-9      # air, kg/mm^3

nodes = []
bricks = []
for k in range(NZ + 1):
    z = k * H
    for (x, y) in ((0, 0), (10, 0), (10, 10), (0, 10)):
        nodes.append((len(nodes) + 1, x, y, z))
for e in range(NZ):
    b = 4 * e
    bricks.append((e + 1, b + 1, b + 2, b + 3, b + 4,
                   b + 5, b + 6, b + 7, b + 8))

top = [n for n, x, y, z in nodes if z == NZ * H]
bot = [n for n, x, y, z in nodes if z == 0.0]
alln = [n for n, *_ in nodes]

starter = ["/BEGIN", "gas piston — adiabatic compression (M6 /EOS)"]
starter.append("/NODE")
for n, x, y, z in nodes:
    starter.append(f"{n} {x} {y} {z}")
starter.append("/BRICK/1")
for row in bricks:
    starter.append(" ".join(str(v) for v in row))
starter += [
    "/PART/1", "gas column", "1 1",
    # the LAW1 host supplies a sliver of shear stiffness for the box
    # modes; its pressure is REPLACED by the EOS
    "/MAT/LAW1/1", "gas host", f"{RHO}", "1e-6 0.",
    f"/EOS/IDEAL-GAS/1", f"{GAMMA} {P0}",
    "/PROP/SOLID/1", "gas", "1.1 0.05 0.1",
    # rigid channel: x/y confinement everywhere, closed bottom
    "/GRNOD/NODE/1", "all", " ".join(map(str, alln)),
    "/BCS/1", "channel walls", "110 000 0 1",
    "/GRNOD/NODE/2", "bottom", " ".join(map(str, bot)),
    "/BCS/2", "closed end", "001 000 0 2",
    # piston: ramp the top face down 20 mm (half the volume) over 8 ms,
    # then hold — slow vs the ~0.1 ms acoustic crossing: adiabatic AND
    # quasi-static, so the end state sits on the isentrope
    "/GRNOD/NODE/9", "piston", " ".join(map(str, top)),
    "/FUNCT/1", "stroke", "0.0 0.0", "8.0 -20.0", "100.0 -20.0",
    "/IMPDISP/1", "push", "1 Z 9",
    "/TH/NODE/1", "piston travel", "DZ VZ", f"{top[0]}",
    "/END",
]

engine1 = [
    "/RUN/GASPISTON/1", "8.0",
    "/DT", "0.9 0",
    "/TFILE", "0.05",
    "/STATE/DT", "2.0 2.0",       # restart snapshots every 2 ms
    "/PRINT/-2000",
    "/STOP", "15.0",
]

engine2 = [
    "/RUN/GASPISTON/2", "12.0",   # resumes at 8 ms from GASPISTON_0001.rst
    "/DT", "0.9 0",
    "/TFILE", "0.05",
    "/PRINT/-2000",
    "/STOP", "15.0",
]

deck_writer.write_starter_from_port_lines(
    starter, "GASPISTON_0000.rad", runname="GASPISTON")
deck_writer.write_engine_from_port_lines(
    engine1, "GASPISTON_0001.rad")
deck_writer.write_engine_from_port_lines(
    engine2, "GASPISTON_0002.rad")
print("wrote GASPISTON_0000.rad / _0001.rad / _0002.rad")
