"""Generate the RIGID_IMPACTOR example deck (M5).

A rigid cylinder-ish impactor — a brick block turned into a /RBODY —
falls onto a simply-clamped shell panel through an /INTER/TYPE7 contact.
The model demonstrates the M5 constraint & load features together:

* /RBODY: the impactor's bricks never deform — the starter assembles the
  block's mass/COG/inertia and the engine integrates one 6-DOF equation
  of motion for it (the block can rock and spin on the panel);
* /SECT: a section cut across the middle of the panel records the
  force/moment the impact drives through it (T01 columns S1_*);
* /GRAV on everything, /ADMAS ballast on the impactor's master node
  (a heavier strike without remeshing);
* /INTER/TYPE7 with Igap=1 between the rigid block and the panel.

Run:
    python generate_deck.py
    pyradioss-starter -i IMPACTOR_0000.rad
    pyradioss-engine  -i IMPACTOR_0001.rad
"""

import numpy as np

# panel: 10 x 10 shells of 2 mm -> 20 x 20 mm, t = 0.5 mm
NX = NY = 10
H = 2.0
# impactor: 3x3x2 bricks of 2 mm, centered, 1 mm above the panel
BI, BJ, BK = 3, 3, 2

nodes = []
shells = []
bricks = []


def pn(i, j):
    return 1 + i + (NX + 1) * j


for j in range(NY + 1):
    for i in range(NX + 1):
        nodes.append((pn(i, j), i * H, j * H, 0.0))
eid = 0
for j in range(NY):
    for i in range(NX):
        eid += 1
        shells.append((eid, pn(i, j), pn(i + 1, j),
                       pn(i + 1, j + 1), pn(i, j + 1)))


def bn(i, j, k):
    return 1000 + i + (BI + 1) * (j + (BJ + 1) * k)


x0 = (NX * H - BI * H) / 2.0
y0 = (NY * H - BJ * H) / 2.0
for k in range(BK + 1):
    for j in range(BJ + 1):
        for i in range(BI + 1):
            nodes.append((bn(i, j, k), x0 + i * H, y0 + j * H,
                          1.25 + k * H))
for k in range(BK):
    for j in range(BJ):
        for i in range(BI):
            eid += 1
            bricks.append((eid, bn(i, j, k), bn(i + 1, j, k),
                           bn(i + 1, j + 1, k), bn(i, j + 1, k),
                           bn(i, j, k + 1), bn(i + 1, j, k + 1),
                           bn(i + 1, j + 1, k + 1), bn(i, j + 1, k + 1)))

border = sorted({pn(i, 0) for i in range(NX + 1)}
                | {pn(i, NY) for i in range(NX + 1)}
                | {pn(0, j) for j in range(NY + 1)}
                | {pn(NX, j) for j in range(NY + 1)})
# section side: the half of the panel with y above the mid-line
side = sorted(pn(i, j) for j in range(NY // 2, NY + 1)
              for i in range(NX + 1))

MASTER = 9999

deck = []
deck.append("/BEGIN")
deck.append("rigid impactor drop on a shell panel (M5 example)")
deck.append("/NODE")
for nid, x, y, z in nodes:
    deck.append(f"{nid} {x:.6g} {y:.6g} {z:.6g}")
deck.append(f"{MASTER} {NX * H / 2:.6g} {NY * H / 2:.6g} 30.0")
deck.append("/SHELL/1")
for row in shells:
    deck.append(" ".join(str(v) for v in row))
deck.append("/BRICK/2")
for row in bricks:
    deck.append(" ".join(str(v) for v in row))
deck.append("/PART/1")
deck.append("panel")
deck.append("1 1")
deck.append("/PART/2")
deck.append("impactor")
deck.append("2 1")
deck.append("/MAT/LAW1/1")
deck.append("aluminium")
deck.append("2.7e-6")
deck.append("70. 0.33")
deck.append("/PROP/SHELL/1")
deck.append("panel t=0.5")
deck.append("1 0 0 0")
deck.append("0.01 0.01 0.01 0 0")
deck.append("3 0 0.5")
deck.append("/PROP/SOLID/2")
deck.append("impactor bricks")
deck.append("1.1 0.05 0.1")
deck.append("/GRNOD/PART/10")
deck.append("impactor nodes")
deck.append("2")
deck.append("/GRNOD/NODE/11")
deck.append("clamped border")
deck.append(" ".join(str(v) for v in border))
deck.append("/GRNOD/NODE/12")
deck.append("section side: upper half of the panel")
deck.append(" ".join(str(v) for v in side))
deck.append(f"/GRNOD/NODE/13")
deck.append("impactor master")
deck.append(f"{MASTER}")
deck.append("/BCS/1")
deck.append("clamp panel border")
deck.append("111 111 0 11")
# the rigid impactor: master node + all brick nodes; 2 g of ballast
deck.append("/RBODY/1")
deck.append("rigid impactor block")
deck.append(f"{MASTER} 10 2.0e-3 1")
deck.append("/INIVEL/TRA/1")
deck.append("impactor falls at 2 mm/ms")
deck.append("0 0 -2.0 10")
deck.append("/FUNCT/1")
deck.append("constant")
deck.append("0.0 1.0")
deck.append("100.0 1.0")
deck.append("/GRAV/1")
deck.append("gravity (mm/ms^2)")
deck.append("1 Z 0 -9.81e-3")
deck.append("/SURF/PART/1")
deck.append("panel surface")
deck.append("1")
deck.append("/INTER/TYPE7/1")
deck.append("impactor vs panel")
deck.append("10 1 2 1")
deck.append("1.0 0.1 0.0")
deck.append("/SECT/1")
deck.append("panel mid-section (upper half is the side set)")
deck.append("12")
deck.append("/TH/SECT/1")
deck.append("section resultants")
deck.append("DEF")
deck.append("1")
deck.append("/TH/NODE/1")
deck.append("impactor master motion")
deck.append("DZ VZ")
deck.append(f"{MASTER}")
deck.append("/END")

with open("IMPACTOR_0000.rad", "w") as fh:
    fh.write("\n".join(deck) + "\n")

engine = [
    "/RUN/IMPACTOR/1",
    "6.0",
    "/VERS/2026",
    "/DT",
    "0.9 0",
    "/TFILE",
    "0.02",
    "/ANIM/DT",
    "0 0.5",
    "/ANIM/VECT/VEL",
    "/ANIM/ELEM/VONM",
    "/PRINT/-500",
    "/STOP",
    "10.0",
]
with open("IMPACTOR_0001.rad", "w") as fh:
    fh.write("\n".join(engine) + "\n")

print("wrote IMPACTOR_0000.rad / IMPACTOR_0001.rad")
