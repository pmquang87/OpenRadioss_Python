# pyradioss examples

Each directory holds a ready-to-run Radioss deck pair **plus the Python
script that generated it** (`generate_deck.py`) — modify the script, re-run
it, and re-run the solver to explore.

| Example | What it exercises | Expected result |
|---|---|---|
| `tensile_bar` | /BRICK solids, LAW2 Johnson–Cook, /BCS, /IMPVEL, /FUNCT, /TH | Axial stress tracks the JC curve `A + B·eps_p^n` (≈0.465 GPa at 2% strain); energy error < 0.1% |
| `box_beam_impact` | /SHELL Belytschko–Tsay, LAW2, /INIVEL, /RWALL, /GRNOD/PART | Progressive crush of the impacting end; ~75% of the kinetic energy converted to plastic work; energy error ~1% |

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
