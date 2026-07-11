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
| `gas_piston` | /EOS/IDEAL-GAS on a LAW1 host (M6), /IMPDISP piston, /STATE/DT snapshots, **restart chaining** (`GASPISTON_0002.rad` resumes `_0001.rst`) | An air column compressed to half volume, then held across two chained runs: the pressure lands on the adiabat p₀·2^1.4 = 2.64 bar within 0.05%, and run 2 continues cycles, energies and T-file numbering exactly where run 1 stopped; energy error 0.00% |
| `implicit_cantilever` | **/IMPL implicit STATICS (M8)** — /SHELL BT4 cantilever, /CLOAD tip load, the Newton–Raphson static driver (not the explicit loop) | A steel plate cantilever loaded transversely at the tip: the run steps the LOAD FACTOR (not a physical time), the listing prints one line per load increment (iterations + residual norm), and the tip deflection matches Euler–Bernoulli `F·L³/3EI` ≈ 1.90 mm to < 0.5 % — each increment converging in a single Newton step (linear elastic). Needs SciPy (`pip install -e ".[implicit]"`) |
| `snap_through` | **/IMPL/ARCL arc-length continuation (M9)** — /TRUSS von Mises two-bar shallow truss, /IMPL/NONLIN geometric nonlinearity, Riks/Crisfield constraint | The classic snap-through problem: the load factor climbs to the closed-form limit point `P(y) = −2EA·ln(L/L₀)·y/L` (≈ 0.63 kN), then the listing shows it DECREASING — even going negative — along the unstable branch, turning at the mirrored limit point and recovering to the full load on the snapped (bars-in-tension) branch. Pure load control (swap /IMPL/ARCL for /IMPL/NONLIN) cannot trace it: just past the peak Newton grinds (18 iterations) and LEAPS discontinuously to the far branch, never sampling the descending/negative-load path. Needs SciPy |
| `implicit_pendulum` | **/IMPL/DYNA implicit DYNAMICS (M10)** — Newmark trapezoidal rule (`/IMPL/DYNA/2` γ=0.5 β=0.25; swap for `/IMPL/DYNA/1 -0.3` to watch HHT-α dissipation drain the ledger) + /IMPL/NONLIN large rotations, /GRAV, /TRUSS | A stiff truss pendulum released from 60° swings through ±60° under gravity — a 120° rotation marched at dt = 3.2 ms, ~150× the explicit stability limit (the bar's wave-transit dt is 0.02 ms): ~240 steps at 3-4 Newton iterations each, where the leapfrog would need ~35,000 cycles. The tip crosses the vertical at the elliptic-integral quarter period T/4 ≈ 170 ms — 7.3% later than the linear-pendulum prediction (the closed form tests/test_m10_impdyn.py asserts to 0.5%) — and the trapezoidal rule holds the energy balance ≈ 0% with zero numerical damping. Needs SciPy |

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
