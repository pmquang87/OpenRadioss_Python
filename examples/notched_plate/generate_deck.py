#!/usr/bin/env python3
"""
Generate the NOTCH example deck: a notched shell plate torn apart, with
tabulated plasticity and bi-quadratic ductile failure — the M3 "crack
propagation by element deletion" showcase.

Model (units mm / ms / kg -> forces kN, stresses GPa):
  * plate 20 x 10 mm meshed with 1 mm 4-node /SHELL elements, 1 mm thick
  * two edge notches at mid-height (3 elements removed from each side):
    the classic double-edge-notch tension (DENT) coupon
  * material LAW36 tabulated plasticity (mild-steel-like curve read from
    /FUNCT: 0.25 GPa yield hardening to 0.4 GPa at 40% strain)
  * /FAIL/BIQUAD ductile failure: failure strain vs stress triaxiality
    through the five calibration points c1..c5 (0.60 / 0.45 / 0.35 /
    0.25 / 0.30) — plane-strain (the state ahead of a crack tip!) is the
    weakest point of the curve, which is what drives the crack INWARD
    from the notch tips
  * bottom edge held, top edge pulled up at 2 mm/ms (ramped)

What to look at (NOTCH_A*.vtk in ParaView):
  * EPSP concentrates at the notch tips first (stress concentration);
  * elements delete tip-to-center — threshold on the OFF cell field
    (OFF = 0 -> deleted) to watch the crack run through the ligament;
  * the engine listing (NOTCH_0001.out) prints the running deletion
    count; after separation the load drops to zero;
  * (since M4) a self-impact /INTER/TYPE7 covers the whole plate: the
    segments of deleted elements drop out of the contact surface, so the
    crack faces separate freely — and would push, not interpenetrate, if
    the halves swung back together.

Run it (from this directory):
    python generate_deck.py          # regenerates the two .rad files
    pyradioss-starter -i NOTCH_0000.rad
    pyradioss-engine  -i NOTCH_0001.rad
"""

import os

NX, NY = 20, 10                # elements
LX, LY = 20.0, 10.0            # plate size (mm)
NOTCH = 3                      # elements removed per side at mid-height
PULL_V = 2.0                   # edge velocity (mm/ms)
T_RAMP = 0.02
T_END = 1.2

here = os.path.dirname(os.path.abspath(__file__))


def node_id(i, j):
    return 1 + i + j * (NX + 1)


def main():
    lines = []
    lines.append("#RADIOSS STARTER")
    lines.append("/BEGIN")
    lines.append("NOTCH - double-edge-notch tension, LAW36 + /FAIL/BIQUAD")

    # ---- nodes -----------------------------------------------------------
    lines.append("/NODE")
    for j in range(NY + 1):
        for i in range(NX + 1):
            x, y = i * LX / NX, j * LY / NY
            lines.append(f"{node_id(i, j):10d}{x:20.10f}{y:20.10f}"
                         f"{0.0:20.10f}")

    # ---- shells (part 1), skipping the two notches -------------------------
    lines.append("/SHELL/1")
    jn = NY // 2                       # notched element row
    eid = 0
    for j in range(NY):
        for i in range(NX):
            eid += 1
            if j == jn and (i < NOTCH or i >= NX - NOTCH):
                continue               # the notches: no element here
            n = [node_id(i, j), node_id(i + 1, j),
                 node_id(i + 1, j + 1), node_id(i, j + 1)]
            lines.append(f"{eid:10d}" + "".join(f"{v:10d}" for v in n))

    # ---- part / material / failure / property ------------------------------
    lines.append("/PART/1")
    lines.append("notched plate")
    lines.append("         1         1")            # prop 1, mat 1
    lines.append("/MAT/LAW36/1")
    lines.append("mild steel, tabulated hardening")
    lines.append("    7.8e-6")                       # rho (kg/mm3)
    lines.append("     210.0       0.3")             # E (GPa), nu
    lines.append("         1         0")             # N_funct, eps_p_max off
    lines.append("        11")                       # hardening curve id
    lines.append("/FUNCT/11")
    lines.append("sy(eps_p): 0.25 -> 0.40 GPa at 40%, then flat")
    lines.append("       0.0      0.25")
    lines.append("       0.4      0.40")
    lines.append("      10.0      0.40")
    lines.append("/FAIL/BIQUAD/1")
    lines.append("      0.60      0.45      0.35      0.25      0.30")
    lines.append("/PROP/SHELL/1")
    lines.append("plate t=1")
    lines.append("         1         0         0         0")
    lines.append("      0.01      0.01      0.01         0         0")
    lines.append("         3         0       1.0")   # 3 layers, t = 1 mm

    # ---- node groups --------------------------------------------------------
    bottom = [node_id(i, 0) for i in range(NX + 1)]
    top = [node_id(i, NY) for i in range(NX + 1)]
    lines.append("/GRNOD/NODE/1")
    lines.append("bottom edge")
    lines.append(" ".join(str(n) for n in bottom))
    lines.append("/GRNOD/NODE/2")
    lines.append("top edge")
    lines.append(" ".join(str(n) for n in top))

    # ---- boundary conditions + imposed velocity ------------------------------
    lines.append("/BCS/1")
    lines.append("clamp bottom edge")
    lines.append("       111       111         0         1")
    lines.append("/FUNCT/1")
    lines.append("velocity ramp")
    lines.append("       0.0       0.0")
    lines.append(f"{T_RAMP:10.3f}       1.0")
    lines.append("     100.0       1.0")
    lines.append("/IMPVEL/1")
    lines.append("pull top edge in +Y")
    lines.append(f"         1         Y         2{PULL_V:10.3f}")

    # ---- self-impact contact (M4) ----------------------------------------------
    # A /INTER/TYPE7 in self-impact mode (grnod = 0) over the whole plate:
    # once /FAIL deletes elements, the segments of the dead elements DROP
    # OUT of the surface (Starter provenance + engine-side off masking) so
    # the freshly created crack faces can separate freely and, if the two
    # halves swing back, they CONTACT instead of interpenetrating. This is
    # the M3<->M4 interaction that makes crack models physically right.
    lines.append("/SURF/PART/1")
    lines.append("whole plate")
    lines.append("         1")
    # Gap_max = 0.5: the variable gap t/2 + t/2 = 1.0 mm equals the mesh
    # size here, which would put every in-plane neighbour permanently
    # inside the contact search band and throttle the time step for
    # nothing — capping the pair gap at half a thickness keeps the
    # out-of-plane crack-face contact and the runtime (the standard
    # Radioss remedy for fine-mesh self-impact).
    lines.append("/INTER/TYPE7/1")
    lines.append("plate self-impact")
    lines.append("         0         1         2         1")   # self, Istf=2, Igap=1
    lines.append("       1.0       0.0       0.0       0.5")

    # ---- time history ----------------------------------------------------------
    lines.append("/TH/NODE/1")
    lines.append("top corner")
    lines.append("DY VY")
    lines.append(str(node_id(0, NY)))
    lines.append("/TH/PART/2")
    lines.append("plate energies")
    lines.append("IE KE")
    lines.append("1")
    lines.append("/END")

    with open(os.path.join(here, "NOTCH_0000.rad"), "w") as fh:
        fh.write("\n".join(lines) + "\n")

    engine = [
        "#RADIOSS ENGINE",
        "/RUN/NOTCH/1",
        f"{T_END}",                  # final time (ms)
        "/DT",
        "0.9  0.0",
        "/TFILE",
        "0.005",
        "/ANIM/DT",
        "0.0  0.025",                # animation every 0.025 ms
        "/ANIM/VECT/VEL",
        "/ANIM/ELEM/VONM",
        "/ANIM/ELEM/EPSP",
        "/PRINT/-200",
        "/STOP",
        "15.0",
    ]
    with open(os.path.join(here, "NOTCH_0001.rad"), "w") as fh:
        fh.write("\n".join(engine) + "\n")
    print("wrote NOTCH_0000.rad / NOTCH_0001.rad")


if __name__ == "__main__":
    main()
