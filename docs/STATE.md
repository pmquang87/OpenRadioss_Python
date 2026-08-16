# pyradioss — project state and roadmap

*The onboarding document. Read this + AGENTS.md before any work; everything
else (PORTING_GUIDE.md 535 KB, VALIDATION.md 233 KB) is grep-only reference.*
*Last updated: 2026-08-02 (handover preparation, after M41).*

## What this is

`pyradioss` is a pure-Python port of the OpenRadioss explicit finite-element
solver (crash/impact dynamics): same Starter/Engine split, same `.rad` deck
format, same output semantics. The goal is Fortran-faithful, readable physics
— every ported formula cites its upstream file under `C:\OpenRadioss\source`.
Speed is explicitly not the goal (an optional numba backend recovers some).

## Quick start

```
.venv\Scripts\python.exe -m pytest -q -m "not slow"     # fast tier, the pre-commit gate
.venv\Scripts\python.exe -m pytest -q                   # full suite (adds 16 slow tests)
.venv\Scripts\python.exe -m pyradioss.starter -i TENSILE_0000.rad   # (from examples\tensile_bar)
.venv\Scripts\python.exe -m pyradioss.engine  -i TENSILE_0001.rad
```

Interpreter/terminal discipline, READ-ONLY paths, domain rules: **AGENTS.md**.

## Baseline (known-good, this machine)

- Venv: Python 3.14.2, numpy 2.4.6, scipy 1.18.0, numba 0.66.0, pytest 9.1.1
  (`requirements-lock.txt`).
- Suite: 1121 collected = 1105 fast + 16 `slow`-marked.
- Fast tier at handover (2026-08-02, this venv): **1101 passed / 4 skipped /
  0 failed**, 16 slow deselected, 30 min 09 s wall (desktop load-dependent;
  expect ~12–50 min).
- Pre-handover full-suite reference (same machine, Python 3.14.2, no numba):
  1095 passed / 2 failed / 24 skipped in 25 min 18 s. The 2 failures were the
  corpus tests whose decks lived in a deleted scratchpad — fixed at handover
  by vendoring the decks into `tests/data/rd_decks/` (see its README).
  Skips that remain are environmental (optional backends like CHOLMOD/MUMPS,
  LS-PrePost/Vortex extras, one 15 MB corpus deck not vendored) and each
  carries a reason string.
- Any red on the fast tier is a regression you introduced, not baseline noise.

## What is implemented (M1 → M155)

README's "Milestones 1–11" section is the *narrative* for the foundation; the
real history is 41 milestones. One line each:

| # | Theme (headline) |
|---|---|
| M1 | Functional port baseline: Starter+Engine, /NODE /BRICK /SHELL /TRUSS /SPRING, LAW1/2, TYPE7, T01/ANIM, energy balance |
| M2 | Element completeness: SH3N, TETRA4, beams, degenerate bricks, BLT84 hourglass, exact-dt eigenbounds |
| M3 | Materials: LAW36, LAW27, LAW42, /FAIL/JOHNSON + /FAIL/BIQUAD + deletion plumbing |
| M4 | Contact: TYPE7 (Istf 0—5, Igap, self-impact, voxel), TYPE2 tied, TYPE11 edge-edge, contact→deletion |
| M5 | Constraints & loads: /RBODY /RBE2 /RBE3 /SECT, moving rigid walls, /PLOAD /IMPDISP |
| M6 | Engine niceties: /DT/NODA mass scaling, restart chaining, /STATE /DAMP /SENSOR /MPC /EOS, dissipation ledger |
| M7 | Performance: fastmath NumPy paths + optional numba backend (pyradioss/accel) |
| M8—M11 | Implicit solver: NR statics, K_geo + arc-length, Newmark/HHT dynamics, all element tangents, /IMPL/BUCKL |
| M12—M15 | Implicit constraints/contact/friction: condensation, TYPE7/11 in Newton loop, MFROT 1—4 + IFQ, last tangents |
| M16—M19 | Modal tower: consistent mass, /IMPL/EIGV, modal & complex-modal superposition, PSD random response, CQC/SRSS |
| M20—M27 | Spectral fatigue tower: Dirlik/rainflow → multiaxial critical-plane → non-proportional → evolutionary S(I%,t) |
| M28—M34 | Multi-input & non-Gaussian: MIMO coherence, Wigner—Ville, Winterstein—Hermite, NORTA exact correlation inversion |
| M35 | Foundation hardening: SH3N rank-deficiency fix, fast/slow CI tiers, FIRST differential validation vs Fortran |
| M36 | Real-deck validation at scale: fixed-format deck WRITER, 529-deck official corpus swept, first timed parity |
| M37 | Column-aware reader + all materials: cfg-driven generic /MAT reader (204 laws), parse backlog 858 → 0 |
| M38 | /PROP pack + fabric: every /PROP family parsed, LAW19 fiber frames, corpus ERROR 149 → 89 |
| M39 | Shell hourglass fidelity (chvis3.F) + /SKEW//FRAME + numba kernel/output expansion |
| M40 | RD-E-1000 dt parity (/RBODY STIFR dt = Fortran exactly), numba auto-default ≥32 elements, ERROR 89 → 76 |
| M41 | Shell element technology: QBAT + QEPH ported (5 RD-E-1000 cases → MATCH), BT rate-kinematics fix, pyradioss-gui + anim→d3plot post-processing |
| M42 | SH3N rotational inertia fix (c3inmas.F alignment) |
| M43—M45 | Auto-backend fallback rules; QBAT and QEPH numba JIT kernels |
| M46 | Removed unused condensed length logic from shell_bt4 |
| M47 | DKT18 shell element ported with Numba acceleration |
| M48—M49 | INTER/TYPE24 parsing and engine logic (forces, broad/narrow phase, dt_int stability) |
| M50 | BT-family cdefo3 branches: c43 node-1-relative velocity form, c45 Z2 warp correction |
| M51—M53 | Implement AIRBAG1 fluid-structure coupling & MONVOL/AIRBAG1 starter volume/area calculation |
| M54 | Parse /INTER/LAGMUL |
| M55—M56 | QUAD and SHEL16 shell elements: parser and unrolled initial geometry/mass logic, physics and force integration |
| M57 | Parse /ALE/BCS to unblock tests |
| M59 | Implement Equation of State /EOS/LINEAR |
| M60 | Implemented TYPE18 shape functions and narrow-phase search |
| M61 | Implement Advanced Mass Scaling (AMS) |
| M62 | Implement SMP parallelism for hexa and shell elements |
| M63 | ALE Advection / EOS Starter Coverage (parsing/layout) |
| M64 | Solid HEPH Element (ISOLID=24) |
| M65 | Implement LAW36 strain rate extrapolation |
| M66 | Implement TYPE32 pretensioner |
| M67 | Divergence backstops: extend KE to IE/HE checks |
| M68—M80 | DeckWriter support gaps closed, input parser fixes for /BOX, /TH, /RBODY, /RWALL; implemented LAW83, LAW6, /FAIL/TAB1, /FAIL/SNCONNECT |
| M81 | /RWALL search distance (Dist) implemented and /RBODY sens_ID parsed for starter topology checks |
| M82 | Input parser fixes: /PRINT card format in Engine; silent Starter bypass for Engine output requests (/H3D, /MON, /PARITH, /RFILE, /ANIM, etc.) to clear coverage noise |
| M83 | Implement /IMPL/DT/FIXP in Engine parser (deferred in M11 implicit completeness) |
| M84 | Implement /SENSOR/VEL, /SENSOR/NOT, /SENSOR/AND, /SENSOR/OR in Starter and Engine |
| M85 | Complete /TRANSFORM suite: /TRANSFORM/ROT, /TRANSFORM/SYM, /TRANSFORM/SCA in Starter |
| M86 | Implement /FAIL/FLD Forming Limit Diagram failure model for shell elements |
| M87 | Implement /SUBMODEL & /ENDSUB container architecture and scoping |
| M88 | Implement /SUBDOMAIN domain-partition keyword for Rad2Rad coupling |
| M89 | Add /GRNOD/NODENS subtype and /XREF reference geometry keyword |
| M90 | Add /FAIL/CONNECT connector failure model (Starter parsing) |
| M91 | Add /DFS/DETPOINT and /DFS/DETPLAN detonation ignition keywords |
| M92 | Add /IMPACC imposed acceleration and /SURF/PLANE infinite plane |
| M93 | Add /LOAD/CENTRI centrifugal load and /IMPTEMP imposed temperature |
| M94 | Add /CONVEC thermal convection and /INIVOL initial volume fraction |
| M95 | Add /RADIATION, /IMPFLUX, /INITEMP thermal loads & initial conditions |
| M96 | Add /INIBRI and /INISHE / /INISH3 initial element state suite |
| M97 | Add 1D inistate (/INITRU, /INIBEA, /INISPR) and extended /SENSOR suite |
| M98 | Add extended /FAIL models (/FAIL/TENSSTRAIN, /ORTHSTRAIN, /GURSON, /ALTER, /VISUAL, /MULLINS_OR) |
| M99 | Add /TRANSFORM/POS, /BCS/CYCLIC, /PERTURB/PART/SOLID, /LOAD/PBLAST, /DEF_INTER |
| M100 | Add /PROP/TYPE10, /TYPE11, /TYPE16, /TYPE6, /PLY, /LAMINATE, /INTER/TYPE25, /INTER/SUB |
| M101 | Add /DEF_SHELL, /DEF_SOLID, extended /DEF_INTER, /PERTURB (SHELL, FAIL), /SPHGLO, /SMS suite |
| M102 | Add /BCS/NRF, /BCS/WALL, /RLINK, /CYL_JOINT, /GJOINT, /MERGE, /INICRACK, /LASER suite |
| M103 | Add /LOAD/PCYL, /LOAD/PFLUID, /PRELOAD, /PRELOAD/AXIAL, /DAMP/INTER, /DAMP/RANGE, /ANALY, /CAA, /UPWIND |
| M104 | Add /GAUGE, /CLUSTER, /EXTLNK, /FXBODY, /INIGRAV, /INIMAP1D, /INIMAP2D, /INISTATE suite |
| M105 | Add /MONVOL/PRES, /MONVOL/GAS, /MONVOL/COMMU1, /MONVOL/LFLUID, /LEAK, /ALE suite |
| M106 | Add /RETRACTOR, /SLIPRING, /INTER/TYPE8, /USERWI, /PROP/TYPE27, /PROP/TYPE51, /TH extensions |
| M107 | Add /FAIL (PUCK, RTCL, SAHRAEI, SYAZWAN, TAB2, GENE1, INIEVO), /SENSOR/NIC, /DRAPE, /INIBRI/EREF, /INCLUDE_DYNA |
| M108 | Add /FAIL (CHANG, HASHIN, TSAIWU, TSAIHILL, HOFFMAN, MAXSTRAIN, LEMAITRE, COCKCROFT, ENERGY), /INTER/TYPE19, /TYPE21, /DAMP/VREL, /DAMP/FUNCT, /MONVOL/FVMBAG1 |
| M109 | Add /PROP (TYPE20/TSHELL, TYPE21/TSH_ORTH, TYPE22/TSH_COMP, TYPE18/INT_BEAM, TYPE34/SPH, TYPE0/VOID), /RWALL (CYL, SPHERE, PARAL), /INTER/GUIDED_CABLE, /SENSOR (ENERGY, TEMP) |
| M110 | Add /EOS (GRUNEISEN, PUFF, TILLOTSON, MURNAGHAN, OSBORNE, LSZK, NOBLE-ABEL, STIFF-GAS), /INIT/DET_* (/DET_*), /LOAD/PBLAST, /ACTIV |
| M111 | Add /INTER (TYPE1, TYPE3, TYPE5, TYPE6, TYPE12, TYPE14, TYPE15, TYPE20, TYPE22, TYPE23), /MONVOL/FVMBAG2, /TRANSFORM/AUTOPOSITION (/AUTOPOSITION), /PROP/TYPE43 (/PROP/CONNECT) |
| M112 | Add /LOAD/CENTRI (/CENTRIF), /LOAD/PRESSURE (/PRESS, /PRESSURE), /LOAD/PFLUID, /INIVEL/AXIS, /INIVEL/FVM, /INIVEL/NODE, /INIVEL/ROT, /IMPDISP/FGEO, /IMPVEL/FGEO, /RWALL/THERM, /SPH/INOUT (/SPH/IO) |
| M113 | Add /SPHBCS, /MADYMO/LINK, /MADYMO/EXFEM, /ALE/GRID/* (DONEA, SPRING, STANDARD, DISP, LAPLACIAN, VOLUME), /ADMESH/GLOBAL (/ADGLOB), /STAMPING, /RANDOM (/RANDOM/GRNOD), /ACCEL, /SUBSET |
| M114 | Add /FAIL/COMPOSITE (3D anisotropic composite failure), /EBCS/PROPELLANT (/BCS/PROPELLANT), /CHECKSUM, /DYNAIN/SHELL/*, /ADMAS/NON_UNIFORM, /ADMAS/NON_UNIFORM_PART, /SECT/CIRCLE, /SECT/PARAL |
| M115 | Add /SET & /SETS (generalized entity sets), extended /SURF modifiers (/SURF/EXT, /ALL, /FREE, /BOX, /MAT, /PROP), /MONVOL/AREA, /TH/TITLE, /STATE/DT, /DYNAIN/DT, /STATE/* output filters |
| M116 | Add /INIQUA & /INIQUAD (quad initial states), /INISTA & /INISTATE (state dispatcher), /FRAME/MOV2 & /SKEW/MOV2 (kinematic moving frames), /SPH/RESERVE (particle buffer), /MOVE_FUNCT & /FUNCT/MOVE, extended /TH channels (/TH/SPHCEL, /TH/NSTRAND, /TH/MODE, /TH/CYL_JO, /TH/FXBODY, /TH/GAUGE, /TH/GR*) |
| M117 | Add /EIG (eigenvalue extraction / modal analysis), /SHFRA/V4 (shell local frame formulation), /INTTHICK/V5 (integration thickness flag), /STATE/STR_FILE & /STR_FILE (stress output file), /MEMORY (memory request allocation), /DEF_INTER/TYPE25 (default contact type 25), /PLOAD (pressure load alias) |
| M118 | Add /FAIL/FRACTAL_DMG & /FAIL/FRACTAL (fractal damage failure), /TRANSFORM/POSITION & /TRANSFORM/POS (spatial positioning transform), /ALE/GRID/FLOW-TRACKING & /ALE/GRID/LAGRANGE & /ALE/ZERO (ALE grid formulation controls), /ARCH (architecture specs), /ALTDOCTAG (doc metadata tag), /EXTERN/LINK (/EXTLINK), /SUBDOMAIN, /INCLUDE_LS-DYNA (/INCLUDE_DYNA) |
| M119 | Add /FUNCT_PYTHON (Python function definition), /FRICTION (generalized multi-part & orthotropic friction model), /REFSTA & /EREF (global & element reference states), /NBCS (non-linear boundary conditions), /ALE/MUSCL (/ALE/SOLVER/MUSCL, MUSCL advection compression factor), /BEM (boundary element method container) |
| M120 | Add extended Engine deck control keywords (/DEBUG, /BCS/ON & /BCS/OFF, /RBODY/ON & /RBODY/OFF, /ALE/ON & /ALE/OFF, /NOIS, /H3D, /FLOW, /UPWIND, /EIG/OFF) |
| M121 | Add extended /SENSOR suite (/SENSOR/GAUGE, /HIC, /WORK, /RWALL, /XSECTION, /DIST_SURF), /GAUGE/POINT, /SPHGLO, /ANALY, cross-reference validation, and Engine element-specific /DT controls (/DT/BRICK, /DT/SHELL, /DT/QUAD, /DT/TETRA10, /DT/INTER) |
| M122 | Add 20-node quadratic bricks (/BRIC20, /HEXA20, /PROP/TYPE23, /GRBR20), coupled ALE/CFD/SPH interaction (/ALECFDSPH), and tensor animation requests (/ANIM/*/TENS) |
| M123 | Add /FAIL/ORTHBIQUAD (orthotropic biquadratic failure), /SLIPRING/SHELL (seatbelt shell sliprings), /SET/* entity set aliases, and extended Engine /STOP controls (/STOP/NSTEP, /STOP/TSTOP, /STOP/TIMET) |
| M124 | Add advanced spring properties (/PROP/TYPE26 SPR_TAB, /PROP/TYPE27 SPR_BDAMP, /PROP/SPR_MAT), /INTER/TYPE22 fluid-structure interface, and /DEF_INTER/TYPE19 defaults |
| M125 | Add non-reflecting boundary conditions (/EBCS/NRF), /FAIL/RTCL & /FAIL/GURSON failure models, and /DEF_INTER/TYPE24 contact defaults |
| M126 | Add extended failure models suite (/FAIL/PUCK, /FAIL/SAHRAEI, /FAIL/SYAZWAN, /FAIL/TAB2, /FAIL/GENE1) & /TH/SENSOR, /TH/CLUSTER output requests |
| M127 | Add composite laminate stacks (/STACK), composite shell properties (/PROP/TYPE17 STACK, /PROP/TYPE51), and /SENSOR/NIC_NIJ aliasing |
| M128 | Add fluid-structure coupling interface (/INTER/TYPE18, /DEF_INTER/TYPE18) and /DEF_INTER/TYPE8 defaults |
| M129 | Add extended seatbelt & advanced material laws suite (/MAT/LAW114, /MAT/LAW117, /MAT/LAW119, /MAT/LAW120, /MAT/LAW121, /MAT/LAW124, /MAT/LAW90) |
| M130 | Add extended loads and preloads suite (/LOAD/PCYL, /PLOAD/PCYL, /PRELOAD/AXIAL, /LOAD/LASER, /MERGE/NODE) |
| M131 | Add advanced sensor suite (/SENSOR/ACCE, /SENSOR/SENS, /SENSOR/PYTHON) and Lagrange multiplier constraints (/LAGMUL, /RBODY/LAGMUL, /GEAR, /RACK, /DIFF) |
| M132 | Add advanced geometric entities & detonation shaping (/BOX/BOX, /SURF/PLANE, /SURF/ELLIPSE, /DFS/WAVE_SHAPER, /SET/NODENS, /BOX aliases) |
| M133 | Add extended monitored volume & airbag suite (/MONVOL/TYPE1-11, /MONVOL/AIRBAG, /MONVOL/COMMU, /MONVOL/FVMBAG, /MONVOL/PART) and fabric leakage models (/LEAK/MAT, /LEAK/PART, /LEAK/AREA) |
| M134 | Add extended geometric entities & spatial transformations (/LINE extended types, /SURF analytical cylinders & spheres, /TRANSFORM/PROJ, /TRANSFORM/FRAME) and damping models (/DAMP/GLOBAL, /DAMP/PART) |
| M135 | Add extended loadings, time history requests & Eulerian/thermal controls (/LOAD/GRAV, /LOAD/BODY, /LOAD/HEAT, /TH extended kinds, /EULER/BCS, /HEAT/BCS) |
| M136 | Add extended entity groups, rigid wall geometries, cross sections & advanced sensors suite (/GRNOD/LINE, /GR*/BOX, /GR*/SURF, /RWALL/BOX, /RWALL/CONE, /SECT/BOX, /SECT/CUT, /SENSOR/SPH, /SENSOR/AIRBAG, /SENSOR/SHELL, /SENSOR/SOLID) |
| M137 | Add extended initial velocities, detonation fronts, state mapping & set routing suite (/INIVEL/PART, /INIVEL/SPH, /DFS/DETLINE, /DFS/DETCIRC, /INIMAP/3D, /INIMAP3D, /SET/PART, /SET/MAT, /SET/PROP, /SET/SUB) |
| M138 | Add extended element initial thermodynamic states, boundary condition constraints & coordinate symmetry suite (/INIBRI, /INISHE, /INITRU, /INIBEA, /INISPR, /BCS/TRA, /BCS/ROT, /TRANSFORM/SYMET, /EBCS/PERIODIC, /EBCS/CYCLIC) |
| M139 | Add multi-dimensional tabular functions, Lagrange multiplier contacts & Eulerian mass distributions suite (/TABLE/2, /TABLE/3, /INTER/LAGMUL/SPOTWELD, /INTER/LAGMUL/SURF, /INTER/LAGMUL/PART, /INTER/LAGMUL/BEAM, /EULER/VOID, /ADMAS/TOTAL_*) |
| M140 | Add extended high-explosive equations of state, unified initial state dispatcher & sensor suite (/EOS/JWL, /EOS/TYPE5, /EOS/COMPACT, /INIT/VEL, /INIT/BRI, /INIT/SHE, /INIT/TRU, /INIT/BEA, /INIT/SPR, /INIT/TEMP, /HEAT/MAT, /HEAT/SOLVER, /SENSOR/RWALL_PLANE, /SENSOR/FORCE, /SENSOR/MOMENT, /SENSOR/RWALL_CYL) |
| M141 | Add extended material sub-object modifiers, thermal-stress & viscoelastic relaxation suite (/MAT/PLAS_ZERIL, /MAT/PLAS_BODNE, /MAT/VISC_PRONY, /VISC/LPRONY, /MAT/THERM_STRESS, /THERM_STRESS, /DAMP/STIFF, /DAMP/MASS) |
| M142 | Add airbag injector & venthole models, advanced ALE solver controls & extended element initial state tensors suite (/AIRBAG/INJECTOR, /INJECTOR, /AIRBAG/VENTHOLE, /VENTHOLE, /INIBRI/STRA_F, /INIBRI/FAIL, /INIBRI/AUX, /INIBRI/SCALE_YLD, /INISHE/STRA_F, /INISHE/EPSP_F, /INISHE/FAIL, /INISHE/AUX, /INISH3/STRA_F, /INISH3/EPSP_F) |
| M143 | Add extended element initial references (/EREF/*), X-FEM initial cracks (/INICRACK), fastener & cohesive properties (/PROP/TYPE5, /PROP/TYPE28) and adaptive remeshing controls (/ADMESH/*) |
| M144 | Add extended interface formulations (/INTER/TYPE9, /INTER/TYPE10, /INTER/TYPE16, /INTER/TYPE17 & /DEF_INTER), bolt preloads (/PRELOAD/BOLT, /SECT/BOLT) and hydrostatic surface loads (/LOAD/HYDRO) |
| M145 | Add 2D bivariate function tables (/FUNC_2D, /FUNC2D), non-local damage regularization models (/NONLOCAL), anisotropic friction orientations (/FRIC_ORIENT), SPH cell initial states (/INISPHCEL), and implicit analysis mode (/IMPLICIT, /THPART) |
| M146 | Add engine dynamic control directives & solver controls suite (/INTER, /DEL, /DLI7, /KEREL, /DYREL, /THERMAL, /HEAT, /ABF, /INIVEL) |
| M147 | Add failure models (/FAIL/EMC, /FAIL/FABRIC, /FAIL/SPALLING, /FAIL/TBUTCHER, /FAIL/WIERZBICKI, /FAIL/WILKINS) and material damping sub-model (/MAT/VISC_PLAS, /VISC/PLAS) |
| M148 | Add engine directives suite II (/DAMP, /MASS/RESET, /SENSOR/RESET, /VIPER, /MADYMO, /RAD2R, /FVMBAG, /PERF, /DT1TET10, /DTTSH, /REPORT, /NEGVOL) |
| M149 | Add guided cable sliding interface (/INTER/GUIDED_CABLE, /GUIDED_CABLE), specialized element properties (/PROP/STITCH, /PROP/PREDIT, /PROP/SPR_MUSC), sub-interface force tracking (/INTER/SUB, /SUBINTER), and extended time-history output channels (/TH/GUIDED_CABLE, /TH/SUBS) |
| M150 | Add Eulerian boundary conditions (/EBCS/PRES, /EBCS/VEL, /EBCS/INLET, /EBCS/FLUXOUT, /EBCS/GRADP0, /EBCS/NORMV, /EBCS/VALVIN, /EBCS/VALVOUT, /EBCS/MONVOL), sliding wall boundary conditions (/BCS/WALL), advanced mass scaling (/AMS), and seatbelt system assembly (/SEATBELT) suite |
| M151 | Add extended contact interfaces (/INTER/TYPE1, /INTER/TYPE3, /INTER/TYPE5, /INTER/TYPE6, /INTER/TYPE14, /INTER/TYPE20, /INTER/TYPE21, /INTER/TYPE23), blast surface loading (/LOAD/PBLAST, /PBLAST), initial volume fraction (/INIVOL), initial gravity (/INIGRAV), initial state import (/INISTA, /INISTATE), boundary element method (/BEM/FLOW, /BEM/DAA), and perturbation controls (/PERTURB/PART/SHELL, /PERTURB/PART/SOLID, /PERTURB/FAIL) suite |
| M152 | Add Eulerian initial BCs (/EBCS/INIP, /EBCS/INIV), dedicated /PROP readers (INJECT1/2, JOINT, TORSION, SPR_ELAS_PLAS, SPR_BEAM, SPOTWELD, BUSHING), RLINK DOF accessors, ImposedFlux alias/accessors, element group extended aliases (/GRSHELL, /GRBRICK, /GRTRUSS, etc.) |
| M153 | Fix checker false positives: ADMAS mass_type-aware cross-reference (types 2/3/4/6/7 skip node group check), downgrade 'model has no elements' from error to warning for decks with only unported element types |
| M154 | TYPE7 contact Igap=2 (scaled variable gap via Fscale_gap) and Igap=3 (mesh-size limited variable gap via %mesh_size), with segment_mesh_gap/node_mesh_gap functions and inspect_active_penetrations parity |
| M155 | Real-deck parser & starter robustness: /INTER/TYPE11 fixed-format reader (radioss51/110/120/140+), /MAT/LAW42 Ogden net shear modulus check GS = sum(mu*alpha) > 0 matching hm_read_mat42.F, /FAIL/BIQUAD preset defaults matching hm_read_fail_biquad.F, and massless /RBODY 1e-20 floor matching inirby.F |
| M156 | Advanced structural properties & failure models suite: /PROP/SH_ORTH (/PROP/TYPE9) orthotropic shell reader, /PROP/INT_BEAM (/PROP/TYPE18) fiber & standard section integrated beam reader, /FAIL/ORTHENERG directional energy failure model, /FAIL/FRACTAL_DMG fractal damage model, /PLY extended orientation/group reader, and /DFS/DETPOINTSET group detonation ignition |
| M157 | Advanced contact interfaces, non-linear spring properties & draping suite: /PROP/TYPE26 (/PROP/SPR_TAB) tabular non-linear spring reader, /PROP/TYPE27 (/PROP/SPR_BDAMP) bilateral damping spring reader, /INTER/TYPE19 5-card multi-segment interface, /INTER/TYPE25 advanced surface contact, /INTER/TYPE8 drawbead interface, and /DRAPE (/TABLE/DRAPE) structured composite fabric draping reader |

Full detail per milestone: grep `PORTING_GUIDE.md` §5 for `M<NN>` and read
that entry only. Feature-matrix tables: §4 (rows current through M41 even
though the section title says M1–6).

### What the port can do today

Explicit central-difference solver with exact per-element dt bounds; elements
hexa8, tetra4, BT shell, QBAT (fully-integrated) and QEPH
(physically-stabilized) shells dispatched by Ishell, C0 triangle, corotational
Timoshenko beam, truss, spring. A cfg-driven `/MAT` reader resolves 204 law
spellings (~16 laws carry real physics: LAW1/2/19/24/27/35/36/40/42/44/62/70/
81, VOID, GAS) plus /EOS and /FAIL with element deletion. Contact TYPE7/2/11
with MFROT friction; rigid walls; /RBODY /RBE2 /RBE3 /MPC /SECT. A full
implicit branch (statics → arc-length → Newmark/HHT → modal/complex-modal →
random-response → spectral-fatigue tower up to non-Gaussian evolutionary
joint-tensor NORTA). NumPy reference backend + numba JIT (`auto` ≥32
elements); SuperLU/CHOLMOD/MUMPS direct solvers. `pyradioss-gui` run-monitor
with post-processing (anim→VTK, TH→CSV, anim→d3plot via Vortex-Radioss
v1.021). NOT ported at all: MPI/SMP parallelism, ALE/Euler, airbags/monvol,
DKT18 shell.

### Validation state (M41, authoritative editions in VALIDATION.md)

- Parity vs real Fortran (bundled examples + official in-envelope): five
  RD-E-1000 shell cases MATCH (QEPH to ~6e-06); BT family stays DEVIATION
  (0.14–0.26) — settled, documented, next targets known (cdefo3 branches).
- Coverage (529 official decks through the port Starter): 13 CLEAN /
  440 SKIPS / 76 ERROR / 0 CRASH / 0 TIMEOUT; ranked blocker table in
  `tools/validation_data/coverage_results_m41.json` → `ranked_gaps`.
- How to reproduce/extend: `.agents/skills/validation-compare/SKILL.md`.

## Where everything lives

| Path | What |
|---|---|
| `pyradioss/input/` | deck reader, card_layouts, cfg-driven mat/prop readers, fixed-format deck WRITER |
| `pyradioss/starter/`, `pyradioss/engine/` | the two programs (engine.py mirrors resol.F's cycle sequence) |
| `pyradioss/elements/`, `materials/`, `failure/`, `contact/` | kernels, each citing its Fortran source file |
| `pyradioss/implicit/` | the whole implicit + modal + spectral-fatigue tower |
| `pyradioss/accel/` | numba backend + auto-selection rule |
| `pyradioss/gui/` | tkinter run-and-monitor GUI + post-processing converters |
| `tests/` | `test_mNN_<slug>.py` per milestone + topic modules; `tests/data/rd_decks/` vendored corpus decks |
| `tools/validate_vs_fortran.py` | THE differential-validation harness (parity + coverage) |
| `tools/validation_data/` | committed evidence: inventory, coverage/parity/perf JSONs per milestone |
| `tools/run_reference_or.ps1` | launch the real Fortran solver with the correct env |
| `tools/lspp_check.py` | LS-PrePost headless d3plot verification |
| `examples/` | runnable native-format decks (tensile_bar is the hello-world) |
| `C:\OpenRadioss` | reference install: `source\` (cite this!), `exec\`, `hm_cfg_files` — READ-ONLY |
| `E:\openradioss_run\` | official deck zips, Ryan Lee reference runs — READ-ONLY |

## How we work (Antigravity)

- ONE milestone per conversation; long sessions rot. The plan below + git
  history carry state between conversations.
- Commit at EVERY green sub-step; quota pauses are normal, commits survive.
- Implementation Plan artifact first → review gate → test-first → fast tier
  green → validation evidence if physics changed → commit → Walkthrough.
  The `/milestone N` workflow (`.agents/workflows/milestone.md`) encodes this.

## Roadmap — next milestones (small, one conversation each)

Numbering continues from M41. M42–M45 were agreed with the maintainer
(2026-07-18); the rest is the ranked candidate pool from PORTING_GUIDE §5's
deferred lists — confirm scope with the maintainer before starting one.

### Candidate pool (bigger — scope with the maintainer first)

Ranked corpus blockers (from coverage_results_m41.json 
anked_gaps):
*(Note: MONVOL/AIRBAG1, INTER/LAGMUL, ALE/BCS, SHEL16, and QUAD have been completed in M51-M57).*

Parity targets: *(Note: DKT18 and BT-family cdefo3 branches were completed in M47 and M50).*

Smaller known items:
- ~~NAN/INF divergence backstop tests only KE (extend to IE/HE)~~ (Done: M67)
- ~~V0700 solids ~2A--small explicit dt~~ (Done: IDEGE scaling implemented)

## Handover notes (2026-08-02)

- The corpus tests' decks were re-homed from a dead session scratchpad to
  `tests/data/rd_decks/` (env `PYRADIOSS_RD_DECKS` for a fuller extract).
- `.venv` + `requirements-lock.txt` are new at handover; numba 0.66.0 is now
  installed (the pre-handover baseline ran without it).
- README's architecture/GUI sections are current; its "What is implemented"
  narrative covers M1–M11 — this file is the authoritative status.