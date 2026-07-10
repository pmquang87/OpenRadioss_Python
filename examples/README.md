# pyradioss examples

Each directory holds a ready-to-run Radioss deck pair **plus the Python
script that generated it** (`generate_deck.py`) — modify the script, re-run
it, and re-run the solver to explore.

| Example | What it exercises | Expected result |
|---|---|---|
| `tensile_bar` | /BRICK solids, LAW2 Johnson–Cook, /BCS, /IMPVEL, /FUNCT, /TH | Axial stress tracks the JC curve `A + B·eps_p^n` (≈0.465 GPa at 2% strain); energy error < 0.1% |
| `box_beam_impact` | /SHELL Belytschko–Tsay, LAW2, /INIVEL, /RWALL, /GRNOD/PART | Progressive crush of the impacting end; ~75% of the kinetic energy converted to plastic work; energy error ~1% |
| `antenna_mast` | /BEAM Timoshenko beams (/PROP/TYPE3), /CLOAD + /FUNCT pulse, /BCS with rotations | Tip deflection peaks ≈23 mm (static F·H³/3EI = 22.5 mm), free vibration at the closed-form 69 ms period; energy error < 1% |
| `rubber_block` | /MAT/LAW42 Ogden hyperelasticity, nearly incompressible, /IMPVEL crush | 40% compression with visible lateral bulge (volume conservation); the time step drops as the rubber stiffens (LAW42's nonlinear sound speed); energy error ~0.1% |
| `notched_plate` | /MAT/LAW36 tabulated plasticity + /FAIL/BIQUAD, element deletion, self-impact /INTER/TYPE7 (M4) | Double-edge-notch tension coupon: plasticity localizes at the notch tips, the crack runs element-by-element through the ligament (watch the `OFF` cell field), full separation; the self-contact drops the deleted elements' segments so the crack faces are truly free; energy error ~0% |
| `spot_weld` | /INTER/TYPE2 tied contact (M4): lap joint of two shell strips, offset ties, /GRNOD/BOX | The pull transfers through the tied patch in single-lap shear; CONTACT energy stays exactly zero (a kinematic tie does no work); energy error ~0.03% |
| `edge_impact` | /INTER/TYPE11 edge-to-edge contact (M4): /LINE/SURF edge sets, Istf=2, Igap=1 | A strip dropped edge-first across a supported strip bounces off the crossing point — contact happens between the nodes, where node-to-surface interfaces are blind; energy error a few % over a 23k-cycle /DT 0.9 run |
| `rigid_impactor` | /RBODY rigid body (M5) with /ADMAS ballast, /INTER/TYPE7 vs a clamped shell panel, /GRAV, /SECT + /TH/SECT | A rigid brick block (one 6-DOF body — its bricks never deform) drops onto an aluminium panel, dents it and bounces/rocks; the mid-panel section force spikes at impact and rings down (columns `S1_*` in the T01); energy error a few % at /DT 0.9 |

Run any example with:

```bash
pyradioss-starter -i <NAME>_0000.rad     # writes <NAME>_0000.out + .rst
pyradioss-engine  -i <NAME>_0001.rad     # writes <NAME>_0001.out, T01.csv, A*.vtk
```

Post-process:
* `<NAME>T01.csv` — plot with anything (column headers in the file),
* `<NAME>A*.vtk` — open the file *group* in ParaView, press Apply, and use
  the animation controls (VONM = von Mises stress, EPSP = plastic strain).

Units in all decks: mm / ms / kg  →  forces in kN, stresses in GPa
(a standard consistent Radioss unit system).
