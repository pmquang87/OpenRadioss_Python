### (a) Per-case verdict table (all 529 runnable official decks)

verdicts: CLEAN = rc0, nothing skipped (control cards excluded); SKIPS(n) = rc0, n non-control keyword families skipped; ERROR = starter refused the model (rc2, collected errors); CRASH = uncaught traceback (rc!=0). Blockers: hard keyword gaps, then PARSE = caught reader failure, ERR = model error (top 3 shown).

| case | category | verdict | blockers |
|---|---|---|---|
| RD-E-0100_Twisted_beam/01_Twisted_Beam/BATOZ/TWISBEAM | rd_e | CLEAN |  |
| RD-E-0100_Twisted_beam/01_Twisted_Beam/DKT18/TWISBEAM | rd_e | CLEAN |  |
| RD-E-0100_Twisted_beam/01_Twisted_Beam/QEPH/TWISBEAM | rd_e | CLEAN |  |
| RD-E-0200_Snap_thru_Roof/02_Snap-through/Explicit_solver/SNAP_EXP | rd_e | SKIPS(2) |  |
| RD-E-0200_Snap_thru_Roof/02_Snap-through/Implicit_solver/SNAP_IMP | rd_e | SKIPS(2) |  |
| RD-E-0300_S-Beam/03_S-Beam/global_IP/BATOZ/S_BEAM | rd_e | SKIPS(3) |  |
| RD-E-0300_S-Beam/03_S-Beam/global_IP/BT-type3/S_BEAM | rd_e | SKIPS(3) |  |
| RD-E-0300_S-Beam/03_S-Beam/global_IP/QEPH/S_BEAM | rd_e | SKIPS(4) |  |
| RD-E-0300_S-Beam/03_S-Beam/v_10ms/QEPH/S_BEAM | rd_e | SKIPS(3) |  |
| RD-E-0300_S-Beam/03_S-Beam/v_5ms/BATOZ/S_BEAM | rd_e | SKIPS(3) |  |
| RD-E-0300_S-Beam/03_S-Beam/v_5ms/BT-type3/S_BEAM | rd_e | SKIPS(3) |  |
| RD-E-0300_S-Beam/03_S-Beam/v_5ms/QEPH/S_BEAM | rd_e | SKIPS(4) |  |
| RD-E-0400_Airbag/04_Airbag/driver/driver_airbag | rd_e | SKIPS(9) | INTER/TYPE19; MONVOL/COMMU1; MOVE_FUNCT |
| RD-E-0500_Beam_frame/05_Beam-frame/FRAME | rd_e | SKIPS(2) |  |
| RD-E-0601_Fuel_tank/1-Tank_sloshing/data/TANK | rd_e | SKIPS(2) | ALE/BCS |
| RD-E-0602_Fuel_flow/2-Tank_overturning/Fluid_flow_1/data/PFTANK | rd_e | SKIPS(2) | ALE/BCS |
| RD-E-0602_Fuel_flow/2-Tank_overturning/Fluid_flow_2/data/PFTANK | rd_e | SKIPS(2) | ALE/BCS |
| RD-E-0700_Pendulums/07_Pendulums/pendulum | rd_e | SKIPS(3) | INTER/TYPE24 |
| RD-E-0800_Hopkinson_bar/08_Hopkinson_Bar/High_strain_rate/SHPB_H | rd_e | ERROR | QUAD; ERR: MODEL CHECK: model has no elements |
| RD-E-0800_Hopkinson_bar/08_Hopkinson_Bar/High_strain_rate/SHPB_H_2021_FEB12 | rd_e | ERROR | QUAD; ERR: MODEL CHECK: model has no elements |
| RD-E-0800_Hopkinson_bar/08_Hopkinson_Bar/Low_strain_rate/SHPB_L | rd_e | ERROR | QUAD; ERR: MODEL CHECK: model has no elements |
| RD-E-0800_Hopkinson_bar/08_Hopkinson_Bar/Low_strain_rate/SHPB_L_2021_FEB12 | rd_e | ERROR | QUAD; ERR: MODEL CHECK: model has no elements |
| RD-E-0901_Billiards/Billiards_model/BILLARD | rd_e | ERROR | INTER/LAGMUL; SHEL16 (+3) |
| RD-E-0901_Billiards/Billiards_model/Supplement_Interface7Lag/BILLARD | rd_e | ERROR | INTER/LAGMUL; SHEL16; ERR: MODEL CHECK: model has no elements |
| RD-E-0902_Collision/Collision_simulation/COLLISION | rd_e | SKIPS(3) | INTER/LAGMUL; SHEL16 |
| RD-E-0903_Test/Contact_modelling/Inter_16_sliding/TEST16S | rd_e | ERROR | INTER/LAGMUL; SHEL16 (+2) |
| RD-E-0903_Test/Contact_modelling/Inter_16_tied/TEST16T | rd_e | ERROR | INTER/LAGMUL; SHEL16 (+2) |
| RD-E-0903_Test/Contact_modelling/Inter_17_sliding/TEST17S | rd_e | ERROR | INTER/LAGMUL; SHEL16; ERR: MODEL CHECK: model has no elements |
| RD-E-0903_Test/Contact_modelling/Inter_17_tied/TEST17ST | rd_e | ERROR | INTER/LAGMUL; SHEL16; ERR: MODEL CHECK: model has no elements |
| RD-E-0903_Test/Contact_modelling/Inter_7_Lagrangian/TEST7L | rd_e | ERROR | INTER/LAGMUL; SHEL16; ERR: MODEL CHECK: model has no elements |
| RD-E-0903_Test/Contact_modelling/Inter_7_Penality/TEST7P | rd_e | SKIPS(1) |  |
| RD-E-1000_Bending/10_Bending/BATOZ/Sf_0.6/ROLLING | rd_e | SKIPS(2) |  |
| RD-E-1000_Bending/10_Bending/BATOZ/Sf_0.8/ROLLING | rd_e | SKIPS(2) |  |
| RD-E-1000_Bending/10_Bending/BATOZ/Sf_0.9/ROLLING | rd_e | SKIPS(2) |  |
| RD-E-1000_Bending/10_Bending/BT/BT_type1/Sf_0.1/ROLLING | rd_e | SKIPS(2) |  |
| RD-E-1000_Bending/10_Bending/BT/BT_type1/Sf_0.9/ROLLING | rd_e | SKIPS(2) |  |
| RD-E-1000_Bending/10_Bending/BT/BT_type3/Sf_0.1/ROLLING | rd_e | SKIPS(2) |  |
| RD-E-1000_Bending/10_Bending/BT/BT_type3/Sf_0.9/ROLLING | rd_e | SKIPS(2) |  |
| RD-E-1000_Bending/10_Bending/BT/BT_type4/Sf_0.1/ROLLING | rd_e | SKIPS(2) |  |
| RD-E-1000_Bending/10_Bending/BT/BT_type4/Sf_0.9/ROLLING | rd_e | SKIPS(2) |  |
| RD-E-1000_Bending/10_Bending/DKT18/Sf_0.1/ROLLING | rd_e | SKIPS(1) |  |
| RD-E-1000_Bending/10_Bending/DKT18/Sf_0.2/ROLLING | rd_e | SKIPS(1) |  |
| RD-E-1000_Bending/10_Bending/DKT18/Sf_0.3/ROLLING | rd_e | SKIPS(1) |  |
| RD-E-1000_Bending/10_Bending/QEPH/Sf_0.8/ROLLING | rd_e | SKIPS(2) |  |
| RD-E-1000_Bending/10_Bending/QEPH/Sf_0.9/ROLLING | rd_e | SKIPS(2) |  |
| RD-E-1101_Tensile_test/1_LAW2/tensile_LAW2 | rd_e | SKIPS(1) |  |
| RD-E-1101_Tensile_test/2_LAW36/tensile_LAW36 | rd_e | SKIPS(1) |  |
| RD-E-1101_Tensile_test/3_LAW2_BIQUAD/tensile_LAW2_BIQUAD | rd_e | ERROR | ERR: tensile_LAW2_BIQUAD_0000.rad:28: /FAIL/BIQUAD/2: M_Flag=2 (built-in material presets) is not ported — give c1..c5 explic |
| RD-E-1101_Tensile_test/4_LAW36_BIQUAD/tensile_LAW36_BIQUAD | rd_e | ERROR | ERR: tensile_LAW36_BIQUAD_0000.rad:38: /FAIL/BIQUAD/2: M_Flag=2 (built-in material presets) is not ported — give c1..c5 expli |
| RD-E-1102_Strain_rate_effect/5_law2_strain_rate/tensile_LAW2_strain_rate | rd_e | SKIPS(1) |  |
| RD-E-1102_Strain_rate_effect/6_law36_strain_rate/tensile_LAW36_strain_rate | rd_e | SKIPS(1) |  |
| RD-E-1200_Bicycle/12_Bicycle/Bike/BIKERC_1506_1104 | rd_e | SKIPS(7) | MONVOL/GAS; RWALL/PARAL (+4) |
| RD-E-1300_Shock_tube/13_Shock_tube/Blast_experiment/blast_experiment | rd_e | SKIPS(3) | ALE/BCS; ALE/MUSCL |
| RD-E-1300_Shock_tube/13_Shock_tube/Shock_experiment/shock_experiment | rd_e | SKIPS(2) | ALE/BCS; ALE/MUSCL |
| RD-E-1500_Gears/15_Gears/Inter16/DIF24416 | rd_e | ERROR | BRIC20; INTER/LAGMUL (+3) |
| RD-E-1500_Gears/15_Gears/Inter17/DIF24416 | rd_e | ERROR | BRIC20; INTER/LAGMUL (+2) |
| RD-E-1601_Explicit/EXPLICIT_solver/ADYREL/data/SEAT_ADYREL | rd_e | ERROR | ERR: SEAT_ADYREL_0000.rad:10917: /INTER/TYPE7/1: Iform=2 (the incremental stiffness tangential formulation) is not ported — u; ERR: SEAT_ADYREL_0000.rad:11019: /INTER/TYPE7/2: Iform=2 (the incremental stiffness tangential formulation) is not ported — u |
| RD-E-1601_Explicit/EXPLICIT_solver/DYREL/data/SEAT_DYREL | rd_e | ERROR | ERR: SEAT_DYREL_0000.rad:10917: /INTER/TYPE7/1: Iform=2 (the incremental stiffness tangential formulation) is not ported — us; ERR: SEAT_DYREL_0000.rad:11019: /INTER/TYPE7/2: Iform=2 (the incremental stiffness tangential formulation) is not ported — us |
| RD-E-1601_Explicit/EXPLICIT_solver/KEREL/data/SEAT_KEREL | rd_e | ERROR | ERR: SEAT_KEREL_0000.rad:10917: /INTER/TYPE7/1: Iform=2 (the incremental stiffness tangential formulation) is not ported — us; ERR: SEAT_KEREL_0000.rad:11019: /INTER/TYPE7/2: Iform=2 (the incremental stiffness tangential formulation) is not ported — us |
| RD-E-1601_Explicit/EXPLICIT_solver/RAYLEIGH/data/SEAT_RAYLEIGH | rd_e | ERROR | ERR: SEAT_RAYLEIGH_0000.rad:10927: /INTER/TYPE7/1: Iform=2 (the incremental stiffness tangential formulation) is not ported —; ERR: SEAT_RAYLEIGH_0000.rad:11029: /INTER/TYPE7/2: Iform=2 (the incremental stiffness tangential formulation) is not ported — |
| RD-E-1601_Explicit/EXPLICIT_solver/Without_damping/data/SEAT | rd_e | ERROR | ERR: SEAT_0000.rad:10917: /INTER/TYPE7/1: Iform=2 (the incremental stiffness tangential formulation) is not ported — use Ifor; ERR: SEAT_0000.rad:11019: /INTER/TYPE7/2: Iform=2 (the incremental stiffness tangential formulation) is not ported — use Ifor |
| RD-E-1602_Implicit/IMPLICIT_solver/Nonlinear/data/SEAT | rd_e | ERROR | ERR: SEAT_0000.rad:10917: /INTER/TYPE7/1: Iform=2 (the incremental stiffness tangential formulation) is not ported — use Ifor; ERR: SEAT_0000.rad:11019: /INTER/TYPE7/2: Iform=2 (the incremental stiffness tangential formulation) is not ported — use Ifor |
| RD-E-1701_Densities/Densities_mesh/mesh0/batoz/BOXBEAM | rd_e | SKIPS(4) |  |
| RD-E-1701_Densities/Densities_mesh/mesh0/bt1/BOXBEAM | rd_e | SKIPS(4) |  |
| RD-E-1701_Densities/Densities_mesh/mesh0/bt3/BOXBEAM | rd_e | SKIPS(4) |  |
| RD-E-1701_Densities/Densities_mesh/mesh0/bt4/BOXBEAM | rd_e | SKIPS(4) |  |
| RD-E-1701_Densities/Densities_mesh/mesh0/c0/BOXBEAM | rd_e | SKIPS(4) |  |
| RD-E-1701_Densities/Densities_mesh/mesh0/dkt18/BOXBEAM | rd_e | SKIPS(4) |  |
| RD-E-1701_Densities/Densities_mesh/mesh0/qeph/BOXBEAM | rd_e | SKIPS(4) |  |
| RD-E-1701_Densities/Densities_mesh/mesh1/batoz/BOXBEAM | rd_e | SKIPS(4) |  |
| RD-E-1701_Densities/Densities_mesh/mesh1/bt1/BOXBEAM | rd_e | SKIPS(4) |  |
| RD-E-1701_Densities/Densities_mesh/mesh1/bt3/BOXBEAM | rd_e | SKIPS(4) |  |
| RD-E-1701_Densities/Densities_mesh/mesh1/bt4/BOXBEAM | rd_e | SKIPS(4) |  |
| RD-E-1701_Densities/Densities_mesh/mesh1/c0/BOXBEAM | rd_e | SKIPS(4) |  |
| RD-E-1701_Densities/Densities_mesh/mesh1/dkt18/BOXBEAM | rd_e | SKIPS(4) |  |
| RD-E-1701_Densities/Densities_mesh/mesh1/qeph/BOXBEAM | rd_e | SKIPS(4) |  |
| RD-E-1701_Densities/Densities_mesh/mesh2/batoz/BOXBEAM | rd_e | SKIPS(4) |  |
| RD-E-1701_Densities/Densities_mesh/mesh2/bt1/BOXBEAM | rd_e | SKIPS(4) |  |
| RD-E-1701_Densities/Densities_mesh/mesh2/bt3/BOXBEAM | rd_e | SKIPS(4) |  |
| RD-E-1701_Densities/Densities_mesh/mesh2/bt4/BOXBEAM | rd_e | SKIPS(4) |  |
| RD-E-1701_Densities/Densities_mesh/mesh2/c0/BOXBEAM | rd_e | SKIPS(4) |  |
| RD-E-1701_Densities/Densities_mesh/mesh2/dkt18/BOXBEAM | rd_e | SKIPS(4) |  |
| RD-E-1701_Densities/Densities_mesh/mesh2/qeph/BOXBEAM | rd_e | SKIPS(4) |  |
| RD-E-1701_Densities/Densities_mesh/mesh3/batoz/BOXBEAM | rd_e | SKIPS(4) |  |
| RD-E-1701_Densities/Densities_mesh/mesh3/bt1/BOXBEAM | rd_e | SKIPS(4) |  |
| RD-E-1701_Densities/Densities_mesh/mesh3/bt3/BOXBEAM | rd_e | SKIPS(4) |  |
| RD-E-1701_Densities/Densities_mesh/mesh3/bt4/BOXBEAM | rd_e | SKIPS(4) |  |
| RD-E-1701_Densities/Densities_mesh/mesh3/c0/BOXBEAM | rd_e | SKIPS(4) |  |
| RD-E-1701_Densities/Densities_mesh/mesh3/dkt18/BOXBEAM | rd_e | SKIPS(4) |  |
| RD-E-1701_Densities/Densities_mesh/mesh3/qeph/BOXBEAM | rd_e | SKIPS(4) |  |
| RD-E-1702_Transitions/Transition_mesh/mesh0/quad/batoz/5ip/BOXBEAM | rd_e | SKIPS(4) |  |
| RD-E-1702_Transitions/Transition_mesh/mesh0/quad/batoz/GLOBAL_PLAS/BOXBEAM | rd_e | SKIPS(4) |  |
| RD-E-1702_Transitions/Transition_mesh/mesh0/quad/batoz/ITER_PLAS/BOXBEAM | rd_e | SKIPS(4) |  |
| RD-E-1702_Transitions/Transition_mesh/mesh0/quad/bt1/5ip/BOXBEAM | rd_e | SKIPS(4) |  |
| RD-E-1702_Transitions/Transition_mesh/mesh0/quad/bt1/GLOBAL_PLAS/BOXBEAM | rd_e | SKIPS(4) |  |
| RD-E-1702_Transitions/Transition_mesh/mesh0/quad/bt1/ITER_PLAS/BOXBEAM | rd_e | SKIPS(4) |  |
| RD-E-1702_Transitions/Transition_mesh/mesh0/quad/bt3/5ip/BOXBEAM | rd_e | SKIPS(4) |  |
| RD-E-1702_Transitions/Transition_mesh/mesh0/quad/bt3/GLOBAL_PLAS/BOXBEAM | rd_e | SKIPS(4) |  |
| RD-E-1702_Transitions/Transition_mesh/mesh0/quad/bt3/ITER_PLAS/BOXBEAM | rd_e | SKIPS(4) |  |
| RD-E-1702_Transitions/Transition_mesh/mesh0/quad/bt4/5ip/BOXBEAM | rd_e | SKIPS(4) |  |
| RD-E-1702_Transitions/Transition_mesh/mesh0/quad/bt4/GLOBAL_PLAS/BOXBEAM | rd_e | SKIPS(4) |  |
| RD-E-1702_Transitions/Transition_mesh/mesh0/quad/bt4/ITER_PLAS/BOXBEAM | rd_e | SKIPS(4) |  |
| RD-E-1702_Transitions/Transition_mesh/mesh0/quad/qeph/5ip/BOXBEAM | rd_e | SKIPS(4) |  |
| RD-E-1702_Transitions/Transition_mesh/mesh0/quad/qeph/GLOBAL_PLAS/BOXBEAM | rd_e | SKIPS(4) |  |
| RD-E-1702_Transitions/Transition_mesh/mesh0/quad/qeph/ITER_PLAS/BOXBEAM | rd_e | SKIPS(4) |  |
| RD-E-1702_Transitions/Transition_mesh/mesh0/tri/c0/5ip/BOXBEAM | rd_e | SKIPS(4) |  |
| RD-E-1702_Transitions/Transition_mesh/mesh0/tri/c0/GLOBAL_PLAS/BOXBEAM | rd_e | SKIPS(4) |  |
| RD-E-1702_Transitions/Transition_mesh/mesh0/tri/c0/ITER_PLAS/BOXBEAM | rd_e | SKIPS(4) |  |
| RD-E-1702_Transitions/Transition_mesh/mesh0/tri/dkt18/5ip/BOXBEAM | rd_e | SKIPS(4) |  |
| RD-E-1702_Transitions/Transition_mesh/mesh0/tri/dkt18/GLOBAL_PLAS/BOXBEAM | rd_e | SKIPS(4) |  |
| RD-E-1702_Transitions/Transition_mesh/mesh0/tri/dkt18/ITER_PLAS/BOXBEAM | rd_e | SKIPS(4) |  |
| RD-E-1702_Transitions/Transition_mesh/mesh1/batoz/5ip/BOXBEAM | rd_e | SKIPS(4) |  |
| RD-E-1702_Transitions/Transition_mesh/mesh1/batoz/GLOBAL_PLAS/BOXBEAM | rd_e | SKIPS(4) |  |
| RD-E-1702_Transitions/Transition_mesh/mesh1/batoz/ITER_PLAS/BOXBEAM | rd_e | SKIPS(4) |  |
| RD-E-1702_Transitions/Transition_mesh/mesh1/bt1/5ip/BOXBEAM | rd_e | SKIPS(4) |  |
| RD-E-1702_Transitions/Transition_mesh/mesh1/bt1/GLOBAL_PLAS/BOXBEAM | rd_e | SKIPS(4) |  |
| RD-E-1702_Transitions/Transition_mesh/mesh1/bt1/ITER_PLAS/BOXBEAM | rd_e | SKIPS(4) |  |
| RD-E-1702_Transitions/Transition_mesh/mesh1/bt3/5ip/BOXBEAM | rd_e | SKIPS(4) |  |
| RD-E-1702_Transitions/Transition_mesh/mesh1/bt3/GLOBAL_PLAS/BOXBEAM | rd_e | SKIPS(4) |  |
| RD-E-1702_Transitions/Transition_mesh/mesh1/bt3/ITER_PLAS/BOXBEAM | rd_e | SKIPS(4) |  |
| RD-E-1702_Transitions/Transition_mesh/mesh1/bt4/5ip/BOXBEAM | rd_e | SKIPS(4) |  |
| RD-E-1702_Transitions/Transition_mesh/mesh1/bt4/GLOBAL_PLAS/BOXBEAM | rd_e | SKIPS(4) |  |
| RD-E-1702_Transitions/Transition_mesh/mesh1/bt4/ITER_PLAS/BOXBEAM | rd_e | SKIPS(4) |  |
| RD-E-1702_Transitions/Transition_mesh/mesh1/qeph/5ip/BOXBEAM | rd_e | SKIPS(4) |  |
| RD-E-1702_Transitions/Transition_mesh/mesh1/qeph/GLOBAL_PLAS/BOXBEAM | rd_e | SKIPS(4) |  |
| RD-E-1702_Transitions/Transition_mesh/mesh1/qeph/ITER_PLAS/BOXBEAM | rd_e | SKIPS(4) |  |
| RD-E-1702_Transitions/Transition_mesh/mesh2/batoz/5ip/BOXBEAM | rd_e | SKIPS(4) |  |
| RD-E-1702_Transitions/Transition_mesh/mesh2/batoz/GLOBAL_PLAS/BOXBEAM | rd_e | SKIPS(4) |  |
| RD-E-1702_Transitions/Transition_mesh/mesh2/batoz/ITER_PLAS/BOXBEAM | rd_e | SKIPS(4) |  |
| RD-E-1702_Transitions/Transition_mesh/mesh2/bt1/5ip/BOXBEAM | rd_e | SKIPS(4) |  |
| RD-E-1702_Transitions/Transition_mesh/mesh2/bt1/GLOBAL_PLAS/BOXBEAM | rd_e | SKIPS(4) |  |
| RD-E-1702_Transitions/Transition_mesh/mesh2/bt1/ITER_PLAS/BOXBEAM | rd_e | SKIPS(4) |  |
| RD-E-1702_Transitions/Transition_mesh/mesh2/bt3/5ip/BOXBEAM | rd_e | SKIPS(4) |  |
| RD-E-1702_Transitions/Transition_mesh/mesh2/bt3/GLOBAL_PLAS/BOXBEAM | rd_e | SKIPS(4) |  |
| RD-E-1702_Transitions/Transition_mesh/mesh2/bt3/ITER_PLAS/BOXBEAM | rd_e | SKIPS(4) |  |
| RD-E-1702_Transitions/Transition_mesh/mesh2/bt4/5ip/BOXBEAM | rd_e | SKIPS(4) |  |
| RD-E-1702_Transitions/Transition_mesh/mesh2/bt4/GLOBAL_PLAS/BOXBEAM | rd_e | SKIPS(4) |  |
| RD-E-1702_Transitions/Transition_mesh/mesh2/bt4/ITER_PLAS/BOXBEAM | rd_e | SKIPS(4) |  |
| RD-E-1702_Transitions/Transition_mesh/mesh2/c0/5ip/BOXBEAM | rd_e | SKIPS(4) |  |
| RD-E-1702_Transitions/Transition_mesh/mesh2/c0/GLOBAL_PLAS/BOXBEAM | rd_e | SKIPS(4) |  |
| RD-E-1702_Transitions/Transition_mesh/mesh2/c0/ITER_PLAS/BOXBEAM | rd_e | SKIPS(4) |  |
| RD-E-1702_Transitions/Transition_mesh/mesh2/dkt18/5ip/BOXBEAM | rd_e | SKIPS(4) |  |
| RD-E-1702_Transitions/Transition_mesh/mesh2/dkt18/GLOBAL_PLAS/BOXBEAM | rd_e | SKIPS(4) |  |
| RD-E-1702_Transitions/Transition_mesh/mesh2/dkt18/ITER_PLAS/BOXBEAM | rd_e | SKIPS(4) |  |
| RD-E-1702_Transitions/Transition_mesh/mesh2/qeph/5ip/BOXBEAM | rd_e | SKIPS(4) |  |
| RD-E-1702_Transitions/Transition_mesh/mesh2/qeph/GLOBAL_PLAS/BOXBEAM | rd_e | SKIPS(4) |  |
| RD-E-1702_Transitions/Transition_mesh/mesh2/qeph/ITER_PLAS/BOXBEAM | rd_e | SKIPS(4) |  |
| RD-E-1702_Transitions/Transition_mesh/mesh3/batoz/5ip/BOXBEAM | rd_e | SKIPS(4) |  |
| RD-E-1702_Transitions/Transition_mesh/mesh3/batoz/GLOBAL_PLAS/BOXBEAM | rd_e | SKIPS(4) |  |
| RD-E-1702_Transitions/Transition_mesh/mesh3/batoz/ITER_PLAS/BOXBEAM | rd_e | SKIPS(4) |  |
| RD-E-1702_Transitions/Transition_mesh/mesh3/bt1/5ip/BOXBEAM | rd_e | SKIPS(4) |  |
| RD-E-1702_Transitions/Transition_mesh/mesh3/bt1/GLOBAL_PLAS/BOXBEAM | rd_e | SKIPS(4) |  |
| RD-E-1702_Transitions/Transition_mesh/mesh3/bt1/ITER_PLAS/BOXBEAM | rd_e | SKIPS(4) |  |
| RD-E-1702_Transitions/Transition_mesh/mesh3/bt3/5ip/BOXBEAM | rd_e | SKIPS(4) |  |
| RD-E-1702_Transitions/Transition_mesh/mesh3/bt3/GLOBAL_PLAS/BOXBEAM | rd_e | SKIPS(4) |  |
| RD-E-1702_Transitions/Transition_mesh/mesh3/bt3/ITER_PLAS/BOXBEAM | rd_e | SKIPS(4) |  |
| RD-E-1702_Transitions/Transition_mesh/mesh3/bt4/5ip/BOXBEAM | rd_e | SKIPS(4) |  |
| RD-E-1702_Transitions/Transition_mesh/mesh3/bt4/GLOBAL_PLAS/BOXBEAM | rd_e | SKIPS(4) |  |
| RD-E-1702_Transitions/Transition_mesh/mesh3/bt4/ITER_PLAS/BOXBEAM | rd_e | SKIPS(4) |  |
| RD-E-1702_Transitions/Transition_mesh/mesh3/c0/5ip/BOXBEAM | rd_e | SKIPS(4) |  |
| RD-E-1702_Transitions/Transition_mesh/mesh3/c0/GLOBAL_PLAS/BOXBEAM | rd_e | SKIPS(4) |  |
| RD-E-1702_Transitions/Transition_mesh/mesh3/c0/ITER_PLAS/BOXBEAM | rd_e | SKIPS(4) |  |
| RD-E-1702_Transitions/Transition_mesh/mesh3/dkt18/5ip/BOXBEAM | rd_e | SKIPS(4) |  |
| RD-E-1702_Transitions/Transition_mesh/mesh3/dkt18/GLOBAL_PLAS/BOXBEAM | rd_e | SKIPS(4) |  |
| RD-E-1702_Transitions/Transition_mesh/mesh3/dkt18/ITER_PLAS/BOXBEAM | rd_e | SKIPS(4) |  |
| RD-E-1702_Transitions/Transition_mesh/mesh3/qeph/5ip/BOXBEAM | rd_e | SKIPS(4) |  |
| RD-E-1702_Transitions/Transition_mesh/mesh3/qeph/GLOBAL_PLAS/BOXBEAM | rd_e | SKIPS(4) |  |
| RD-E-1702_Transitions/Transition_mesh/mesh3/qeph/ITER_PLAS/BOXBEAM | rd_e | SKIPS(4) |  |
| RD-E-1703_Distorted/Distorted_mesh/mesh0/quad/batoz/5ip/BOXBEAM | rd_e | SKIPS(4) |  |
| RD-E-1703_Distorted/Distorted_mesh/mesh0/quad/batoz/GLOBAL_PLAS/BOXBEAM | rd_e | SKIPS(4) |  |
| RD-E-1703_Distorted/Distorted_mesh/mesh0/quad/batoz/ITER_PLAS/BOXBEAM | rd_e | SKIPS(4) |  |
| RD-E-1703_Distorted/Distorted_mesh/mesh0/quad/bt1/5ip/BOXBEAM | rd_e | SKIPS(4) |  |
| RD-E-1703_Distorted/Distorted_mesh/mesh0/quad/bt1/GLOBAL_PLAS/BOXBEAM | rd_e | SKIPS(4) |  |
| RD-E-1703_Distorted/Distorted_mesh/mesh0/quad/bt1/ITER_PLAS/BOXBEAM | rd_e | SKIPS(4) |  |
| RD-E-1703_Distorted/Distorted_mesh/mesh0/quad/bt3/5ip/BOXBEAM | rd_e | SKIPS(4) |  |
| RD-E-1703_Distorted/Distorted_mesh/mesh0/quad/bt3/GLOBAL_PLAS/BOXBEAM | rd_e | SKIPS(4) |  |
| RD-E-1703_Distorted/Distorted_mesh/mesh0/quad/bt3/ITER_PLAS/BOXBEAM | rd_e | SKIPS(4) |  |
| RD-E-1703_Distorted/Distorted_mesh/mesh0/quad/bt4/5ip/BOXBEAM | rd_e | SKIPS(4) |  |
| RD-E-1703_Distorted/Distorted_mesh/mesh0/quad/bt4/GLOBAL_PLAS/BOXBEAM | rd_e | SKIPS(4) |  |
| RD-E-1703_Distorted/Distorted_mesh/mesh0/quad/bt4/ITER_PLAS/BOXBEAM | rd_e | SKIPS(4) |  |
| RD-E-1703_Distorted/Distorted_mesh/mesh0/quad/qeph/5ip/BOXBEAM | rd_e | SKIPS(4) |  |
| RD-E-1703_Distorted/Distorted_mesh/mesh0/quad/qeph/GLOBAL_PLAS/BOXBEAM | rd_e | SKIPS(4) |  |
| RD-E-1703_Distorted/Distorted_mesh/mesh0/quad/qeph/ITER_PLAS/BOXBEAM | rd_e | SKIPS(4) |  |
| RD-E-1703_Distorted/Distorted_mesh/mesh0/tri/c0/5ip/BOXBEAM | rd_e | SKIPS(4) |  |
| RD-E-1703_Distorted/Distorted_mesh/mesh0/tri/c0/GLOBAL_PLAS/BOXBEAM | rd_e | SKIPS(4) |  |
| RD-E-1703_Distorted/Distorted_mesh/mesh0/tri/c0/ITER_PLAS/BOXBEAM | rd_e | SKIPS(4) |  |
| RD-E-1703_Distorted/Distorted_mesh/mesh0/tri/dkt18/5ip/BOXBEAM | rd_e | SKIPS(4) |  |
| RD-E-1703_Distorted/Distorted_mesh/mesh0/tri/dkt18/GLOBAL_PLAS/BOXBEAM | rd_e | SKIPS(4) |  |
| RD-E-1703_Distorted/Distorted_mesh/mesh0/tri/dkt18/ITER_PLAS/BOXBEAM | rd_e | SKIPS(4) |  |
| RD-E-1703_Distorted/Distorted_mesh/mesh1/quad/batoz/5ip/BOXBEAM | rd_e | SKIPS(5) |  |
| RD-E-1703_Distorted/Distorted_mesh/mesh1/quad/batoz/GLOBAL_PLAS/BOXBEAM | rd_e | SKIPS(5) |  |
| RD-E-1703_Distorted/Distorted_mesh/mesh1/quad/batoz/ITER_PLAS/BOXBEAM | rd_e | SKIPS(5) |  |
| RD-E-1703_Distorted/Distorted_mesh/mesh1/quad/bt1/5ip/BOXBEAM | rd_e | SKIPS(5) |  |
| RD-E-1703_Distorted/Distorted_mesh/mesh1/quad/bt1/GLOBAL_PLAS/BOXBEAM | rd_e | SKIPS(5) |  |
| RD-E-1703_Distorted/Distorted_mesh/mesh1/quad/bt1/ITER_PLAS/BOXBEAM | rd_e | SKIPS(5) |  |
| RD-E-1703_Distorted/Distorted_mesh/mesh1/quad/bt3/5ip/BOXBEAM | rd_e | SKIPS(5) |  |
| RD-E-1703_Distorted/Distorted_mesh/mesh1/quad/bt3/GLOBAL_PLAS/BOXBEAM | rd_e | SKIPS(5) |  |
| RD-E-1703_Distorted/Distorted_mesh/mesh1/quad/bt3/ITER_PLAS/BOXBEAM | rd_e | SKIPS(5) |  |
| RD-E-1703_Distorted/Distorted_mesh/mesh1/quad/bt4/5ip/BOXBEAM | rd_e | SKIPS(5) |  |
| RD-E-1703_Distorted/Distorted_mesh/mesh1/quad/bt4/GLOBAL_PLAS/BOXBEAM | rd_e | SKIPS(5) |  |
| RD-E-1703_Distorted/Distorted_mesh/mesh1/quad/bt4/ITER_PLAS/BOXBEAM | rd_e | SKIPS(5) |  |
| RD-E-1703_Distorted/Distorted_mesh/mesh1/quad/qeph/5ip/BOXBEAM | rd_e | SKIPS(5) |  |
| RD-E-1703_Distorted/Distorted_mesh/mesh1/quad/qeph/GLOBAL_PLAS/BOXBEAM | rd_e | SKIPS(5) |  |
| RD-E-1703_Distorted/Distorted_mesh/mesh1/quad/qeph/ITER_PLAS/BOXBEAM | rd_e | SKIPS(5) |  |
| RD-E-1703_Distorted/Distorted_mesh/mesh1/tri/c0/5ip/BOXBEAM | rd_e | SKIPS(4) |  |
| RD-E-1703_Distorted/Distorted_mesh/mesh1/tri/c0/GLOBAL_PLAS/BOXBEAM | rd_e | SKIPS(4) |  |
| RD-E-1703_Distorted/Distorted_mesh/mesh1/tri/c0/ITER_PLAS/BOXBEAM | rd_e | SKIPS(4) |  |
| RD-E-1703_Distorted/Distorted_mesh/mesh1/tri/dkt18/5ip/BOXBEAM | rd_e | SKIPS(4) |  |
| RD-E-1703_Distorted/Distorted_mesh/mesh1/tri/dkt18/GLOBAL_PLAS/BOXBEAM | rd_e | SKIPS(4) |  |
| RD-E-1703_Distorted/Distorted_mesh/mesh1/tri/dkt18/ITER_PLAS/BOXBEAM | rd_e | SKIPS(4) |  |
| RD-E-1703_Distorted/Distorted_mesh/mesh2/quad/batoz/5ip/BOXBEAM | rd_e | SKIPS(5) |  |
| RD-E-1703_Distorted/Distorted_mesh/mesh2/quad/batoz/GLOBAL_PLAS/BOXBEAM | rd_e | SKIPS(5) |  |
| RD-E-1703_Distorted/Distorted_mesh/mesh2/quad/batoz/ITER_PLAS/BOXBEAM | rd_e | SKIPS(5) |  |
| RD-E-1703_Distorted/Distorted_mesh/mesh2/quad/bt1/5ip/BOXBEAM | rd_e | SKIPS(5) |  |
| RD-E-1703_Distorted/Distorted_mesh/mesh2/quad/bt1/GLOBAL_PLAS/BOXBEAM | rd_e | SKIPS(5) |  |
| RD-E-1703_Distorted/Distorted_mesh/mesh2/quad/bt1/ITER_PLAS/BOXBEAM | rd_e | SKIPS(5) |  |
| RD-E-1703_Distorted/Distorted_mesh/mesh2/quad/bt3/5ip/BOXBEAM | rd_e | SKIPS(5) |  |
| RD-E-1703_Distorted/Distorted_mesh/mesh2/quad/bt3/GLOBAL_PLAS/BOXBEAM | rd_e | SKIPS(5) |  |
| RD-E-1703_Distorted/Distorted_mesh/mesh2/quad/bt3/ITER_PLAS/BOXBEAM | rd_e | SKIPS(5) |  |
| RD-E-1703_Distorted/Distorted_mesh/mesh2/quad/bt4/5ip/BOXBEAM | rd_e | SKIPS(5) |  |
| RD-E-1703_Distorted/Distorted_mesh/mesh2/quad/bt4/GLOBAL_PLAS/BOXBEAM | rd_e | SKIPS(5) |  |
| RD-E-1703_Distorted/Distorted_mesh/mesh2/quad/bt4/ITER_PLAS/BOXBEAM | rd_e | SKIPS(5) |  |
| RD-E-1703_Distorted/Distorted_mesh/mesh2/quad/qeph/5ip/BOXBEAM | rd_e | SKIPS(5) |  |
| RD-E-1703_Distorted/Distorted_mesh/mesh2/quad/qeph/GLOBAL_PLAS/BOXBEAM | rd_e | SKIPS(5) |  |
| RD-E-1703_Distorted/Distorted_mesh/mesh2/quad/qeph/ITER_PLAS/BOXBEAM | rd_e | SKIPS(5) |  |
| RD-E-1703_Distorted/Distorted_mesh/mesh2/tri/c0/5ip/BOXBEAM | rd_e | SKIPS(4) |  |
| RD-E-1703_Distorted/Distorted_mesh/mesh2/tri/c0/GLOBAL_PLAS/BOXBEAM | rd_e | SKIPS(4) |  |
| RD-E-1703_Distorted/Distorted_mesh/mesh2/tri/c0/ITER_PLAS/BOXBEAM | rd_e | SKIPS(4) |  |
| RD-E-1703_Distorted/Distorted_mesh/mesh2/tri/dkt18/5ip/BOXBEAM | rd_e | SKIPS(4) |  |
| RD-E-1703_Distorted/Distorted_mesh/mesh2/tri/dkt18/GLOBAL_PLAS/BOXBEAM | rd_e | SKIPS(4) |  |
| RD-E-1703_Distorted/Distorted_mesh/mesh2/tri/dkt18/ITER_PLAS/BOXBEAM | rd_e | SKIPS(4) |  |
| RD-E-1703_Distorted/Distorted_mesh/mesh3/quad/batoz/5ip/BOXBEAM | rd_e | SKIPS(5) |  |
| RD-E-1703_Distorted/Distorted_mesh/mesh3/quad/batoz/GLOBAL_PLAS/BOXBEAM | rd_e | SKIPS(5) |  |
| RD-E-1703_Distorted/Distorted_mesh/mesh3/quad/batoz/ITER_PLAS/BOXBEAM | rd_e | SKIPS(5) |  |
| RD-E-1703_Distorted/Distorted_mesh/mesh3/quad/bt1/5ip/BOXBEAM | rd_e | SKIPS(5) |  |
| RD-E-1703_Distorted/Distorted_mesh/mesh3/quad/bt1/GLOBAL_PLAS/BOXBEAM | rd_e | SKIPS(5) |  |
| RD-E-1703_Distorted/Distorted_mesh/mesh3/quad/bt1/ITER_PLAS/BOXBEAM | rd_e | SKIPS(5) |  |
| RD-E-1703_Distorted/Distorted_mesh/mesh3/quad/bt3/5ip/BOXBEAM | rd_e | SKIPS(5) |  |
| RD-E-1703_Distorted/Distorted_mesh/mesh3/quad/bt3/GLOBAL_PLAS/BOXBEAM | rd_e | SKIPS(5) |  |
| RD-E-1703_Distorted/Distorted_mesh/mesh3/quad/bt3/ITER_PLAS/BOXBEAM | rd_e | SKIPS(5) |  |
| RD-E-1703_Distorted/Distorted_mesh/mesh3/quad/bt4/5ip/BOXBEAM | rd_e | SKIPS(5) |  |
| RD-E-1703_Distorted/Distorted_mesh/mesh3/quad/bt4/GLOBAL_PLAS/BOXBEAM | rd_e | SKIPS(5) |  |
| RD-E-1703_Distorted/Distorted_mesh/mesh3/quad/bt4/ITER_PLAS/BOXBEAM | rd_e | SKIPS(5) |  |
| RD-E-1703_Distorted/Distorted_mesh/mesh3/quad/qeph/5ip/BOXBEAM | rd_e | SKIPS(5) |  |
| RD-E-1703_Distorted/Distorted_mesh/mesh3/quad/qeph/GLOBAL_PLAS/BOXBEAM | rd_e | SKIPS(5) |  |
| RD-E-1703_Distorted/Distorted_mesh/mesh3/quad/qeph/ITER_PLAS/BOXBEAM | rd_e | SKIPS(5) |  |
| RD-E-1703_Distorted/Distorted_mesh/mesh3/tri/c0/5ip/BOXBEAM | rd_e | SKIPS(4) |  |
| RD-E-1703_Distorted/Distorted_mesh/mesh3/tri/c0/GLOBAL_PLAS/BOXBEAM | rd_e | SKIPS(4) |  |
| RD-E-1703_Distorted/Distorted_mesh/mesh3/tri/c0/ITER_PLAS/BOXBEAM | rd_e | SKIPS(4) |  |
| RD-E-1703_Distorted/Distorted_mesh/mesh3/tri/dkt18/5ip/BOXBEAM | rd_e | SKIPS(4) |  |
| RD-E-1703_Distorted/Distorted_mesh/mesh3/tri/dkt18/GLOBAL_PLAS/BOXBEAM | rd_e | SKIPS(4) |  |
| RD-E-1703_Distorted/Distorted_mesh/mesh3/tri/dkt18/ITER_PLAS/BOXBEAM | rd_e | SKIPS(4) |  |
| RD-E-1801_Square_plate_torsion/Torsion/2q4-4t3/batoz/TORSION | rd_e | SKIPS(2) |  |
| RD-E-1801_Square_plate_torsion/Torsion/2q4-4t3/bt_type1/TORSION | rd_e | SKIPS(2) |  |
| RD-E-1801_Square_plate_torsion/Torsion/2q4-4t3/bt_type4/TORSION | rd_e | SKIPS(2) |  |
| RD-E-1801_Square_plate_torsion/Torsion/2q4-4t3/qeph/TORSION | rd_e | SKIPS(2) |  |
| RD-E-1801_Square_plate_torsion/Torsion/4q4/batoz/TORSION | rd_e | SKIPS(2) |  |
| RD-E-1801_Square_plate_torsion/Torsion/4q4/bt_type1/TORSION | rd_e | SKIPS(2) |  |
| RD-E-1801_Square_plate_torsion/Torsion/4q4/bt_type4/TORSION | rd_e | SKIPS(2) |  |
| RD-E-1801_Square_plate_torsion/Torsion/4q4/qeph/TORSION | rd_e | SKIPS(2) |  |
| RD-E-1801_Square_plate_torsion/Torsion/8t3/c0/TORSION | rd_e | SKIPS(2) |  |
| RD-E-1801_Square_plate_torsion/Torsion/8t3/dkt/TORSION | rd_e | SKIPS(2) |  |
| RD-E-1801_Square_plate_torsion/Torsion/8t3_inv/c0/TORSION | rd_e | SKIPS(2) |  |
| RD-E-1801_Square_plate_torsion/Torsion/8t3_inv/dkt/TORSION | rd_e | SKIPS(2) |  |
| RD-E-1802_Elastic/Membrane_elastic/2q4-4t3/batoz/TRACTION | rd_e | SKIPS(2) |  |
| RD-E-1802_Elastic/Membrane_elastic/2q4-4t3/bt1/TRACTION | rd_e | SKIPS(2) |  |
| RD-E-1802_Elastic/Membrane_elastic/2q4-4t3/bt3/TRACTION | rd_e | SKIPS(2) |  |
| RD-E-1802_Elastic/Membrane_elastic/2q4-4t3/qeph/TRACTION | rd_e | SKIPS(2) |  |
| RD-E-1802_Elastic/Membrane_elastic/4q4/batoz/TRACTION | rd_e | SKIPS(2) |  |
| RD-E-1802_Elastic/Membrane_elastic/4q4/bt1/TRACTION | rd_e | SKIPS(2) |  |
| RD-E-1802_Elastic/Membrane_elastic/4q4/bt3/TRACTION | rd_e | SKIPS(2) |  |
| RD-E-1802_Elastic/Membrane_elastic/4q4/qeph/TRACTION | rd_e | SKIPS(2) |  |
| RD-E-1802_Elastic/Membrane_elastic/8t3/c0/TRACTION | rd_e | SKIPS(2) |  |
| RD-E-1802_Elastic/Membrane_elastic/8t3/dkt/TRACTION | rd_e | SKIPS(2) |  |
| RD-E-1802_Elastic/Membrane_elastic/8t3_inv/c0/TRACTION | rd_e | SKIPS(2) |  |
| RD-E-1802_Elastic/Membrane_elastic/8t3_inv/dkt/TRACTION | rd_e | SKIPS(2) |  |
| RD-E-1803_Elasto_plastic/Membrane_elasto-plastic/2q4-4t3/batoz/TRACTION | rd_e | SKIPS(2) |  |
| RD-E-1803_Elasto_plastic/Membrane_elasto-plastic/2q4-4t3/bt_type1/TRACTION | rd_e | SKIPS(2) |  |
| RD-E-1803_Elasto_plastic/Membrane_elasto-plastic/2q4-4t3/bt_type3/TRACTION | rd_e | SKIPS(2) |  |
| RD-E-1803_Elasto_plastic/Membrane_elasto-plastic/2q4-4t3/qeph/TRACTION | rd_e | SKIPS(2) |  |
| RD-E-1803_Elasto_plastic/Membrane_elasto-plastic/4q4/batoz/TRACTION | rd_e | SKIPS(2) |  |
| RD-E-1803_Elasto_plastic/Membrane_elasto-plastic/4q4/bt_type1/TRACTION | rd_e | SKIPS(2) |  |
| RD-E-1803_Elasto_plastic/Membrane_elasto-plastic/4q4/bt_type3/TRACTION | rd_e | SKIPS(2) |  |
| RD-E-1803_Elasto_plastic/Membrane_elasto-plastic/4q4/qeph/TRACTION | rd_e | SKIPS(2) |  |
| RD-E-1803_Elasto_plastic/Membrane_elasto-plastic/8t3/co/TRACTION | rd_e | SKIPS(2) |  |
| RD-E-1803_Elasto_plastic/Membrane_elasto-plastic/8t3/dkt/TRACTION | rd_e | SKIPS(2) |  |
| RD-E-1803_Elasto_plastic/Membrane_elasto-plastic/t3_inv/c0/TRACTION | rd_e | SKIPS(2) |  |
| RD-E-1803_Elasto_plastic/Membrane_elasto-plastic/t3_inv/dkt/TRACTION | rd_e | SKIPS(2) |  |
| RD-E-1900_Wave/19_Wave_propagation/ALE_formulation/WAVE | rd_e | ERROR | QUAD; UPWIND; ERR: MODEL CHECK: model has no elements |
| RD-E-1900_Wave/19_Wave_propagation/Lagrangian_formulation/WAVE | rd_e | ERROR | QUAD; ERR: MODEL CHECK: model has no elements |
| RD-E-2000_Ice_cube/20_Cube/TYPE24/cube_TYPE24 | rd_e | SKIPS(2) | INTER/TYPE24 |
| RD-E-2000_Ice_cube/20_Cube/TYPE25/cube_TYPE25 | rd_e | SKIPS(2) | INTER/TYPE25 |
| RD-E-2000_Ice_cube/20_Cube/TYPE7/cube_TYPE7 | rd_e | SKIPS(1) |  |
| RD-E-2100_Cam/21_Cam/interface16/coarse_mesh/I16S16CM | rd_e | ERROR | INTER/LAGMUL; SHEL16; ERR: GROUP CHECK: /GRBRIC/14: unknown element id(s) [21, 22, 23, 25, 28]... |
| RD-E-2100_Cam/21_Cam/interface16/fine_mesh/I16S16FM | rd_e | ERROR | BRIC20; INTER/LAGMUL (+2) |
| RD-E-2100_Cam/21_Cam/interface7/friction/I7PFMCAM | rd_e | CLEAN |  |
| RD-E-2100_Cam/21_Cam/interface7/lagrange/slave_cam/I7LMCAM | rd_e | SKIPS(1) | INTER/LAGMUL |
| RD-E-2100_Cam/21_Cam/interface7/lagrange/slave_valve/I7LMVALVE | rd_e | SKIPS(1) | INTER/LAGMUL |
| RD-E-2100_Cam/21_Cam/interface7/penalty/slave_cam/I7PMCAM | rd_e | CLEAN |  |
| RD-E-2100_Cam/21_Cam/interface7/penalty/slave_valve/I7PMVALVE | rd_e | CLEAN |  |
| RD-E-2201_ALE/Ditching_Mono_Domain_ALE/data/ALE_mono | rd_e | SKIPS(7) | EOS/STIFF-GAS; INTER/TYPE18 |
| RD-E-2202_SPH/Ditching_Mono_Domain_SPH/SPHEX_mono_110 | rd_e | SKIPS(9) | EOS/STIFF-GAS; SPH/INOUT (+2) |
| RD-E-2203_Multi_Domain/Ditching_Multi_Domain_ALE/ALE_multi | rd_e | SKIPS(9) | ALE/MUSCL; EOS/STIFF-GAS (+2) |
| RD-E-2203_Multi_Domain/Ditching_Multi_Domain_SPH/SPHEX_multi_110 | rd_e | SKIPS(10) | EOS/STIFF-GAS; SPH/INOUT (+3) |
| RD-E-2300_Brake/23_Brake/BRAKE | rd_e | SKIPS(3) | INTER/TYPE19 |
| RD-E-2400_Laminating/24_Laminating/Formulation/Isolid14_Icpre2/ROLLING | rd_e | SKIPS(2) |  |
| RD-E-2400_Laminating/24_Laminating/Formulation/Isolid14_Icpre3/ROLLING | rd_e | SKIPS(2) |  |
| RD-E-2400_Laminating/24_Laminating/Formulation/Isolid17_Icpre2/ROLLING | rd_e | SKIPS(2) |  |
| RD-E-2400_Laminating/24_Laminating/Formulation/Isolid17_Icpre3/ROLLING | rd_e | SKIPS(2) |  |
| RD-E-2400_Laminating/24_Laminating/Thickness/2_elements/ROLLING | rd_e | SKIPS(2) |  |
| RD-E-2400_Laminating/24_Laminating/Thickness/5_elements/ROLLING | rd_e | SKIPS(2) |  |
| RD-E-2400_Laminating/24_Laminating/small_strain_formulation/Ismstr1_small/ROLLING | rd_e | SKIPS(2) |  |
| RD-E-2400_Laminating/24_Laminating/small_strain_formulation/Ismstr2_mix/ROLLING | rd_e | SKIPS(2) |  |
| RD-E-2400_Laminating/24_Laminating/temperature/T=1200/ROLLING | rd_e | SKIPS(2) |  |
| RD-E-2400_Laminating/24_Laminating/temperature/T=800/ROLLING | rd_e | SKIPS(2) |  |
| RD-E-2500_Spring_back/25_Spring-back/Explicit_spring-back/DBEND_44 | rd_e | SKIPS(1) |  |
| RD-E-2500_Spring_back/25_Spring-back/Implicit_spring-back/DBEND_44 | rd_e | SKIPS(1) |  |
| RD-E-2601_Failure_strain/failure_strain/one_shell/biaxial_test/LAW2/main_TEST4 | rd_e | SKIPS(1) |  |
| RD-E-2601_Failure_strain/failure_strain/one_shell/biaxial_test/LAW27/main_TEST4 | rd_e | SKIPS(1) |  |
| RD-E-2601_Failure_strain/failure_strain/one_shell/uniaxial/LAW2/main_TEST4 | rd_e | SKIPS(1) |  |
| RD-E-2601_Failure_strain/failure_strain/one_shell/uniaxial/LAW27/main_TEST4 | rd_e | SKIPS(1) |  |
| RD-E-2601_Failure_strain/failure_strain/plate_model/LAW2/LAW2 | rd_e | SKIPS(2) |  |
| RD-E-2601_Failure_strain/failure_strain/plate_model/LAW27/LAW27 | rd_e | SKIPS(2) |  |
| RD-E-2602_Ductile/ductile_failure_model/one_shell/Johnson/Ishell_1_wo_epsmax/main_TEST4 | rd_e | SKIPS(1) |  |
| RD-E-2602_Ductile/ductile_failure_model/one_shell/biquad/Ishell_1_wo_epsmax/main_TEST4 | rd_e | SKIPS(1) |  |
| RD-E-2602_Ductile/ductile_failure_model/one_shell/tab1/Ishell_1_wo_epsmax/main_TEST4 | rd_e | SKIPS(3) | FAIL/TAB1; TABLE |
| RD-E-2602_Ductile/ductile_failure_model/plate_model/BIQUAD_model/Ishell=1_without_epsmax/FAILURE_BIQUAD | rd_e | SKIPS(2) |  |
| RD-E-2602_Ductile/ductile_failure_model/plate_model/JOHNSON_Ifail_sh_model/Ishell=1_and_espmax/FAILURE_JOHNSON | rd_e | SKIPS(2) |  |
| RD-E-2602_Ductile/ductile_failure_model/plate_model/JOHNSON_Ifail_sh_model/Ishell=1_without_epsmax/FAILURE_JOHNSON | rd_e | SKIPS(2) |  |
| RD-E-2602_Ductile/ductile_failure_model/plate_model/JOHNSON_Ifail_sh_model/Ishell=2_and_epsmax/FAILURE_JOHNSON | rd_e | SKIPS(3) |  |
| RD-E-2602_Ductile/ductile_failure_model/plate_model/JOHNSON_Ifail_sh_model/Ishell=2_without_epsmax/FAILURE_JOHNSON | rd_e | SKIPS(2) |  |
| RD-E-2602_Ductile/ductile_failure_model/plate_model/JOHNSON_model/Ishell=1_without_epsmax/FAILURE_JOHNSON | rd_e | SKIPS(2) |  |
| RD-E-2602_Ductile/ductile_failure_model/plate_model/TAB1_model/Ishell=1_without_epsmax/FAILURE_TAB1 | rd_e | SKIPS(4) | FAIL/TAB1; TABLE |
| RD-E-2603_Forming/Forming/one_shell/biaxial/main_TEST4 | rd_e | SKIPS(2) | FAIL/FLD |
| RD-E-2603_Forming/Forming/one_shell/uniaxial/main_TEST4 | rd_e | SKIPS(2) | FAIL/FLD |
| RD-E-2603_Forming/Forming/plate_model/FAILURE_FLD | rd_e | SKIPS(3) | FAIL/FLD |
| RD-E-2700_Football/27_Football_shoot/Bathenay_circular/BAT_CIR | rd_e | SKIPS(3) | MONVOL/AIRBAG1 |
| RD-E-2700_Football/27_Football_shoot/Bathenay_square/BAT_SQR | rd_e | SKIPS(4) | MONVOL/AIRBAG1; RWALL/PARAL |
| RD-E-2700_Football/27_Football_shoot/Santini_circular/SANT_CIR | rd_e | ERROR | MONVOL/AIRBAG1; ERR: PART CHECK: /PART/124: material 0 not defined (+3) |
| RD-E-2700_Football/27_Football_shoot/Santini_square/SANT_SQR | rd_e | ERROR | MONVOL/AIRBAG1; RWALL/PARAL (+3) |
| RD-E-3900_Biomedical_valve/39_Bio_Valve/BIO_VALVE/VALVE | rd_e | SKIPS(4) | ALE/BCS; ALE/GRID; CAA |
| RD-E-4200_Rubber_ring/42_Rubber_Ring/rubber_ring | rd_e | ERROR | ERR: rubber_ring_0000.rad:3105: /INTER/TYPE7/14: Iform=2 (the incremental stiffness tangential formulation) is not ported — u; ERR: rubber_ring_0000.rad:3405: /INTER/TYPE7/15: Iform=2 (the incremental stiffness tangential formulation) is not ported — u |
| RD-E-4300_Perfect_gas/43_perfect_gas_polynomial_eos/01-Pabsolute_Eabsolute/1BRICK_COMPRESSION | rd_e | SKIPS(3) | ALE/BCS |
| RD-E-4300_Perfect_gas/43_perfect_gas_polynomial_eos/02-Prelative_Eabsolute/1BRICK_COMPRESSION | rd_e | SKIPS(3) | ALE/BCS |
| RD-E-4300_Perfect_gas/43_perfect_gas_polynomial_eos/03-Prelative_Erelative/1BRICK_COMPRESSION | rd_e | SKIPS(3) | ALE/BCS |
| RD-E-4300_Perfect_gas/43_perfect_gas_polynomial_eos/04-Pabsolue_Erelative/1BRICK_COMPRESSION | rd_e | SKIPS(3) | ALE/BCS |
| RD-E-4400_Blow_molding_AMS/44_blow_moding_ams/E4_66_AMS/EXAMPLE4_66 | rd_e | ERROR | AMS; ERR: EXAMPLE4_66_0000.rad:64332: /INTER/TYPE7/1: Iform=2 (the incremental stiffness tangential formulation) is not ported — u (+4) |
| RD-E-4400_Blow_molding_AMS/44_blow_moding_ams/E4_66_Noda_CST/EXAMPLE4_66 | rd_e | ERROR | ERR: EXAMPLE4_66_0000.rad:64332: /INTER/TYPE7/1: Iform=2 (the incremental stiffness tangential formulation) is not ported — u; ERR: EXAMPLE4_66_0000.rad:64332: /INTER/TYPE7/1: Igap=2 not ported (0 constant, 1 variable) (+3) |
| RD-E-4400_Blow_molding_AMS/44_blow_moding_ams/E4_66_no_dt_control/EXAMPLE4_66 | rd_e | ERROR | ERR: EXAMPLE4_66_0000.rad:64332: /INTER/TYPE7/1: Iform=2 (the incremental stiffness tangential formulation) is not ported — u; ERR: EXAMPLE4_66_0000.rad:64332: /INTER/TYPE7/1: Igap=2 not ported (0 constant, 1 variable) (+3) |
| RD-E-4500_Multi_Domain/45_multidomain_tied/monodomain/bumper_LL4 | rd_e | ERROR | ERR: bumper_LL4_0000.rad:76871: /INTER/TYPE7/13: Iform=2 (the incremental stiffness tangential formulation) is not ported — u |
| RD-E-4500_Multi_Domain/45_multidomain_tied/multidomain/bumper_LL4 | rd_e | ERROR | SUBDOMAIN; ERR: bumper_LL4_0000.rad:76875: /INTER/TYPE7/13: Iform=2 (the incremental stiffness tangential formulation) is not ported — u |
| RD-E-4601_Lagrange/Lagrange/Lag6elem | rd_e | ERROR | DFS/DETPLAN; EOS/GRUNEISEN (+2) |
| RD-E-4602_Euler/Euler/data/Euler6 | rd_e | ERROR | DFS/DETPLAN; QUAD; ERR: MODEL CHECK: model has no elements |
| RD-E-4701_Kupfer/Kupfer_tests/LAW24/C000/C000 | rd_e | SKIPS(1) |  |
| RD-E-4701_Kupfer/Kupfer_tests/LAW24/CC00/CC00 | rd_e | SKIPS(1) |  |
| RD-E-4701_Kupfer/Kupfer_tests/LAW24/CC01/CC01 | rd_e | SKIPS(1) |  |
| RD-E-4701_Kupfer/Kupfer_tests/LAW24/T000/TC00 | rd_e | SKIPS(1) |  |
| RD-E-4701_Kupfer/Kupfer_tests/LAW24/TC01/TC01 | rd_e | SKIPS(1) |  |
| RD-E-4701_Kupfer/Kupfer_tests/LAW24/TC02/TC02 | rd_e | SKIPS(1) |  |
| RD-E-4701_Kupfer/Kupfer_tests/LAW24/TC03/TC03 | rd_e | SKIPS(1) |  |
| RD-E-4701_Kupfer/Kupfer_tests/LAW81/C000/C000 | rd_e | SKIPS(1) |  |
| RD-E-4701_Kupfer/Kupfer_tests/LAW81/CC00/CC00 | rd_e | SKIPS(1) |  |
| RD-E-4701_Kupfer/Kupfer_tests/LAW81/CC01/CC01 | rd_e | SKIPS(1) |  |
| RD-E-4701_Kupfer/Kupfer_tests/LAW81/T000/TC00 | rd_e | SKIPS(1) |  |
| RD-E-4701_Kupfer/Kupfer_tests/LAW81/TC01/TC01 | rd_e | SKIPS(1) |  |
| RD-E-4701_Kupfer/Kupfer_tests/LAW81/TC02/TC02 | rd_e | SKIPS(1) |  |
| RD-E-4701_Kupfer/Kupfer_tests/LAW81/TC03/TC03 | rd_e | SKIPS(1) |  |
| RD-E-4702_Brazilian/Splitting_tensile_test/CONCRETE_v3 | rd_e | SKIPS(1) |  |
| RD-E-4801_Solid_spotweld/One_solid_element_validation_law83/run01_no_failure/KS2_model_v01 | rd_e | ERROR | MOVE_FUNCT; ERR: KS2_model_v01_0000.rad:3304: /INTER/TYPE7/1: Iform=2 (the incremental stiffness tangential formulation) is not ported —  |
| RD-E-4801_Solid_spotweld/One_solid_element_validation_law83/run02_w_failure/KS2_model_v01 | rd_e | ERROR | FAIL/SNCONNECT; MOVE_FUNCT; ERR: KS2_model_v01_0000.rad:3304: /INTER/TYPE7/1: Iform=2 (the incremental stiffness tangential formulation) is not ported —  |
| RD-E-4801_Solid_spotweld/One_solid_element_validation_law83/run02_w_failure_Beta_update/KS2_model_v01 | rd_e | ERROR | FAIL/SNCONNECT; MOVE_FUNCT; ERR: KS2_model_v01_0000.rad:3304: /INTER/TYPE7/1: Iform=2 (the incremental stiffness tangential formulation) is not ported —  |
| RD-E-4801_Solid_spotweld/One_solid_element_validation_law83/run03_peel_test_1/KS2_model_v01 | rd_e | ERROR | FAIL/SNCONNECT; MOVE_FUNCT; ERR: KS2_model_v01_0000.rad:3304: /INTER/TYPE7/1: Iform=2 (the incremental stiffness tangential formulation) is not ported —  |
| RD-E-4801_Solid_spotweld/One_solid_element_validation_law83/run03_peel_test_2/KS2_model_v01 | rd_e | ERROR | FAIL/SNCONNECT; MOVE_FUNCT; ERR: KS2_model_v01_0000.rad:3304: /INTER/TYPE7/1: Iform=2 (the incremental stiffness tangential formulation) is not ported —  |
| RD-E-4802_Solid_spring/solid_spot_law59/FRAME_MODIFIED | rd_e | ERROR | ENDSUB; FAIL/CONNECT (+4) |
| RD-E-4802_Solid_spring/spring_beam_spotweld/FRAME_MODIFIED | rd_e | ERROR | ENDSUB; SUBMODEL (+3) |
| RD-E-4900_Bird_strike/49_bird_strike/BIRD_WINDSHIELD_v1 | rd_e | ERROR | SPHCEL; SPHGLO (+3) |
| RD-E-5000_Inivol_FSI/50_inivol_and_fluid_structure/data/fsi_drop_container | rd_e | SKIPS(5) | EOS/LINEAR; INIVOL (+3) |
| RD-E-5101_Size_optimization/Size_Optimization/RAD_OPT/Neon-b_pillar | rd_e | ERROR | ERR: Neon-b_pillar_0000.rad:12579: /INTER/TYPE7/1: Iform=2 (the incremental stiffness tangential formulation) is not ported —; ERR: CROSS REF: /ADMAS/1: node group 1 not defined |
| RD-E-5101_Size_optimization/Size_Optimization/base/Neon-b_pillar | rd_e | ERROR | ERR: Neon-b_pillar_0000.rad:12579: /INTER/TYPE7/1: Iform=2 (the incremental stiffness tangential formulation) is not ported —; ERR: CROSS REF: /ADMAS/1: node group 1 not defined |
| RD-E-5102_Topology_optimization/Topology_Optimization/Base/hook_opt | rd_e | SKIPS(2) | AMS |
| RD-E-5102_Topology_optimization/Topology_Optimization/RADOPT/hook_opt | rd_e | SKIPS(2) | AMS |
| RD-E-5200_Creep/52_cylinder_creep/cylinder_creep_beta_001/foam_relax | rd_e | SKIPS(3) |  |
| RD-E-5200_Creep/52_cylinder_creep/cylinder_creep_beta_005/foam_relax | rd_e | SKIPS(3) |  |
| RD-E-5200_Creep/52_cylinder_creep/cylinder_creep_beta_1/foam_relax | rd_e | SKIPS(3) |  |
| RD-E-5200_Creep/52_cylinder_creep/cylinder_relaxation_beta_001/foam_relax | rd_e | SKIPS(3) |  |
| RD-E-5200_Creep/52_cylinder_creep/cylinder_relaxation_beta_005/foam_relax | rd_e | SKIPS(3) |  |
| RD-E-5200_Creep/52_cylinder_creep/cylinder_relaxation_beta_1/foam_relax | rd_e | SKIPS(3) |  |
| RD-E-5300_Thermal/53_thermal_analysis/heat_exchange/moving_source | rd_e | SKIPS(4) | CONVEC; IMPTEMP |
| RD-E-5400_Cut_methodology/54_cut_model/cut_sub_model/CBOX | rd_e | SKIPS(2) | AMS |
| RD-E-5400_Cut_methodology/54_cut_model/full_Model/CBOX | rd_e | SKIPS(2) | AMS |
| RD-E-5501_Fan_blade/1_Initialize_rotation_stress/fan_blade_initialize | rd_e | ERROR | LOAD/CENTRI; SENSOR/NOT; ERR: fan_blade_initialize_0000.rad:1250: /INTER/TYPE7/1: Igap=3 not ported (0 constant, 1 variable) |
| RD-E-5502_Rotating/2_Rotation_and_1_ice/fan_blade_ice_impact | rd_e | ERROR | INISHE/STRA_F; INISHE/STRS_F (+2) |
| RD-E-5600_Hyperelastic_material/56_HyperElastic_Material/Arruda_Boyce_model/LAW92_UT/rubber_tension_v1 | rd_e | SKIPS(1) |  |
| RD-E-5600_Hyperelastic_material/56_HyperElastic_Material/Arruda_Boyce_model/LAW92_UT_parameter/rubber_tension_v1 | rd_e | SKIPS(1) |  |
| RD-E-5600_Hyperelastic_material/56_HyperElastic_Material/Ogden_model/LAW42_pair2/rubber_tension_v1 | rd_e | SKIPS(1) |  |
| RD-E-5600_Hyperelastic_material/56_HyperElastic_Material/Ogden_model/LAW42_pair3/rubber_tension_v1 | rd_e | ERROR | ERR: LAW42.txt:7: /MAT/LAW42/1: every Ogden pair must satisfy mu_p * alpha_p > 0 (material stability); ERR: PART CHECK: /PART/1: material 1 not defined |
| RD-E-5600_Hyperelastic_material/56_HyperElastic_Material/Ogden_model/LAW69_extend_to_compression/rubber_tension_v1 | rd_e | SKIPS(1) |  |
| RD-E-5600_Hyperelastic_material/56_HyperElastic_Material/Ogden_model/LAW69_ogden_pair2/LAW69_ogden_pair2_Poisson04997/rubber_tension_v1 | rd_e | SKIPS(1) |  |
| RD-E-5600_Hyperelastic_material/56_HyperElastic_Material/Ogden_model/LAW69_ogden_pair3/rubber_tension_v1 | rd_e | SKIPS(1) |  |
| RD-E-5600_Hyperelastic_material/56_HyperElastic_Material/Ogden_model/LAW82/rubber_tension_v1 | rd_e | SKIPS(1) |  |
| RD-E-5600_Hyperelastic_material/56_HyperElastic_Material/Ogden_model/LAW88/rubber_tension_v1 | rd_e | SKIPS(1) |  |
| RD-E-5600_Hyperelastic_material/56_HyperElastic_Material/Yeoh_model/rubber_tension_v1 | rd_e | SKIPS(1) |  |
| RD-V-0500_Shyue_Shock_Tube/JWL_shock_tube_MUSCL/data/Shock_tube_MUSCL | rd_v_blast | SKIPS(1) | DFS/DETPOINT |
| RD-V-0505_Shock_Tube/Eulerian_formulation/data/TACEUL | rd_v_blast | SKIPS(1) |  |
| RD-V-0510_Prandtl_Meyer_Fan/0510_Prandtl_Meyer_fan/data/rdv_0510_PRANDTL_MEYER_EXPANSION | rd_v_blast | ERROR | EBCS/FLUXOUT; EBCS/INLET (+12) |
| RD-V-0520_Double_Shock/0520_double_shock/data/DOUBLE_OBLIQUE_QUADS | rd_v_blast | ERROR | EBCS/FLUXOUT; EBCS/INLET (+12) |
| RD-V-0530_Wave_propagation/0530_wave_propagation/data/rdv_0530_wave_propagation | rd_v_blast | CLEAN |  |
| RD-V-0100_Impact/0100_impact/type24/rdv_0100_momentum_type24 | rd_v_contact | SKIPS(3) | INIVEL/TRA; INTER/TYPE24 |
| RD-V-0100_Impact/0100_impact/type25/rdv_0100_momentum_type25 | rd_v_contact | SKIPS(3) | INIVEL/TRA; INTER/TYPE25 |
| RD-V-0100_Impact/0100_impact/type7/rdv_0100_momentum_type7 | rd_v_contact | ERROR | INIVEL/TRA; ERR: rdv_0100_momentum_type7_0000.rad:2261: /INTER/TYPE7/1: Iform=2 (the incremental stiffness tangential formulation) is not; ERR: rdv_0100_momentum_type7_0000.rad:2280: /INTER/TYPE7/2: Iform=2 (the incremental stiffness tangential formulation) is not |
| RD-V-0110_Friction/0110_friction/rdv_0110_friction | rd_v_contact | SKIPS(2) | INTER/TYPE24 |
| RD-V-0120_Edge_to_Edge/0120_edge_to_edge/ADYREL/CONTACT_TYPE11 | rd_v_contact | ERROR | ENDSUB; SUBMODEL (+4) |
| RD-V-0120_Edge_to_Edge/0120_edge_to_edge/DYREL/CONTACT_TYPE11 | rd_v_contact | ERROR | ENDSUB; SUBMODEL (+4) |
| RD-V-0120_Edge_to_Edge/0120_edge_to_edge/No_dynamic_relaxation/CONTACT_TYPE11 | rd_v_contact | ERROR | ENDSUB; SUBMODEL (+4) |
| RD-V-0130/0130_tied_interface/Iteration1/TYPE_10/data/Tied_surface_type10 | rd_v_contact | SKIPS(2) | INTER/TYPE10 |
| RD-V-0130/0130_tied_interface/Iteration1/TYPE_10_1/data/Tied_surface_type10_1 | rd_v_contact | SKIPS(2) | INTER/TYPE10 |
| RD-V-0130/0130_tied_interface/Iteration1/type7_INI/data/rdv_0100_momentum_type7a | rd_v_contact | ERROR | ERR: rdv_0100_momentum_type7a_0000.rad:2232: /INTER/TYPE7/1: Iform=2 (the incremental stiffness tangential formulation) is no; ERR: rdv_0100_momentum_type7a_0000.rad:2256: /INTER/TYPE7/2: Iform=2 (the incremental stiffness tangential formulation) is no |
| RD-V-0130/0130_tied_interface/Iteration2/TYPE10_0/data/4_Modele_10_0 | rd_v_contact | SKIPS(2) | INTER/TYPE10 |
| RD-V-0130/0130_tied_interface/Iteration2/TYPE10_1/data/4_Modele_10_1 | rd_v_contact | SKIPS(2) | INTER/TYPE10 |
| RD-V-0130/0130_tied_interface/Iteration2/TYPE7/data/4_Modele_TYPE7 | rd_v_contact | SKIPS(1) |  |
| RD-V-0130/0130_tied_interface/Iteration3/TYPE10_0/data/5_Modele2_Gliss_10_0 | rd_v_contact | SKIPS(2) | INTER/TYPE10 |
| RD-V-0130/0130_tied_interface/Iteration3/TYPE7/data/5_TYPE7 | rd_v_contact | SKIPS(1) |  |
| RD-V-0010_Beam_Bending/0010_bending/rdv_0010_beam_bending | rd_v_elements | CLEAN |  |
| RD-V-0020_Cantilever_Beam/0020_beam/explicit/beam_type3/Cantilever_beam | rd_v_elements | CLEAN |  |
| RD-V-0020_Cantilever_Beam/0020_beam/explicit/brick_isolid14/Cantilever_beam | rd_v_elements | SKIPS(1) |  |
| RD-V-0020_Cantilever_Beam/0020_beam/explicit/brick_isolid17/Cantilever_beam | rd_v_elements | SKIPS(1) |  |
| RD-V-0020_Cantilever_Beam/0020_beam/explicit/brick_isolid18/Cantilever_beam | rd_v_elements | SKIPS(1) |  |
| RD-V-0020_Cantilever_Beam/0020_beam/explicit/brick_isolid24/Cantilever_beam | rd_v_elements | SKIPS(1) |  |
| RD-V-0020_Cantilever_Beam/0020_beam/explicit/sh3n_Ish3n0/Cantilever_beam | rd_v_elements | SKIPS(1) |  |
| RD-V-0020_Cantilever_Beam/0020_beam/explicit/sh3n_Ish3n30/Cantilever_beam | rd_v_elements | SKIPS(1) |  |
| RD-V-0020_Cantilever_Beam/0020_beam/explicit/shell_ishell12/Cantilever_beam | rd_v_elements | SKIPS(1) |  |
| RD-V-0020_Cantilever_Beam/0020_beam/explicit/shell_ishell24/Cantilever_beam | rd_v_elements | SKIPS(1) |  |
| RD-V-0020_Cantilever_Beam/0020_beam/explicit/tetra10_0/Cantilever_beam | rd_v_elements | ERROR | TETRA10; ERR: MODEL CHECK: model has no elements |
| RD-V-0020_Cantilever_Beam/0020_beam/explicit/tetra10_2/Cantilever_beam | rd_v_elements | ERROR | TETRA10; ERR: MODEL CHECK: model has no elements |
| RD-V-0020_Cantilever_Beam/0020_beam/explicit/tetra4_0/Cantilever_beam | rd_v_elements | SKIPS(1) |  |
| RD-V-0020_Cantilever_Beam/0020_beam/explicit/tetra4_1/Cantilever_beam | rd_v_elements | SKIPS(1) |  |
| RD-V-0020_Cantilever_Beam/0020_beam/explicit/tetra4_3/Cantilever_beam | rd_v_elements | SKIPS(1) |  |
| RD-V-0020_Cantilever_Beam/0020_beam/implicit/beam_type3/Cantilever_beam | rd_v_elements | CLEAN |  |
| RD-V-0020_Cantilever_Beam/0020_beam/implicit/brick_isolid14/Cantilever_beam | rd_v_elements | SKIPS(1) |  |
| RD-V-0020_Cantilever_Beam/0020_beam/implicit/brick_isolid17/Cantilever_beam | rd_v_elements | SKIPS(1) |  |
| RD-V-0020_Cantilever_Beam/0020_beam/implicit/brick_isolid18/Cantilever_beam | rd_v_elements | SKIPS(1) |  |
| RD-V-0020_Cantilever_Beam/0020_beam/implicit/brick_isolid24/Cantilever_beam | rd_v_elements | SKIPS(1) |  |
| RD-V-0020_Cantilever_Beam/0020_beam/implicit/sh3n_Ish3n0/Cantilever_beam | rd_v_elements | SKIPS(1) |  |
| RD-V-0020_Cantilever_Beam/0020_beam/implicit/shell_ishell12/Cantilever_beam | rd_v_elements | SKIPS(1) |  |
| RD-V-0020_Cantilever_Beam/0020_beam/implicit/shell_ishell24/Cantilever_beam | rd_v_elements | SKIPS(1) |  |
| RD-V-0020_Cantilever_Beam/0020_beam/implicit/tetra10_0/Cantilever_beam | rd_v_elements | ERROR | TETRA10; ERR: MODEL CHECK: model has no elements |
| RD-V-0020_Cantilever_Beam/0020_beam/implicit/tetra4_0/Cantilever_beam | rd_v_elements | SKIPS(1) |  |
| RD-V-0020_Cantilever_Beam/0020_beam/implicit/tetra4_3/Cantilever_beam | rd_v_elements | SKIPS(1) |  |
| RD-V-0030_Spring_Type_4/0030_spring_type_4/0030_spring_type_4_stiff/Spring_TYPE4_element | rd_v_elements | SKIPS(1) |  |
| RD-V-0030_Spring_Type_4/0030_spring_type_4/0030_spring_type_4_visc/Spring_TYPE4_element | rd_v_elements | SKIPS(1) |  |
| RD-V-0031_Pretensioner_Spring_Type_32/0031_spring_type_32/Spring_TYPE32_element | rd_v_elements | SKIPS(1) |  |
| RD-V-0032_Spring_Type_8/0032_spring_type_8/0032_spring_type_8_stiff_RX/Spring_TYPE8_element | rd_v_elements | SKIPS(1) |  |
| RD-V-0032_Spring_Type_8/0032_spring_type_8/0032_spring_type_8_stiff_RY/Spring_TYPE8_element | rd_v_elements | SKIPS(1) |  |
| RD-V-0032_Spring_Type_8/0032_spring_type_8/0032_spring_type_8_stiff_RZ/Spring_TYPE8_element | rd_v_elements | SKIPS(1) |  |
| RD-V-0032_Spring_Type_8/0032_spring_type_8/0032_spring_type_8_stiff_TX/Spring_TYPE8_element | rd_v_elements | SKIPS(1) |  |
| RD-V-0032_Spring_Type_8/0032_spring_type_8/0032_spring_type_8_stiff_TY/Spring_TYPE8_element | rd_v_elements | SKIPS(1) |  |
| RD-V-0032_Spring_Type_8/0032_spring_type_8/0032_spring_type_8_stiff_TZ/Spring_TYPE8_element | rd_v_elements | SKIPS(1) |  |
| RD-V-0032_Spring_Type_8/0032_spring_type_8/0032_spring_type_8_visc_RX/Spring_TYPE8_element | rd_v_elements | SKIPS(1) |  |
| RD-V-0032_Spring_Type_8/0032_spring_type_8/0032_spring_type_8_visc_RY/Spring_TYPE8_element | rd_v_elements | SKIPS(1) |  |
| RD-V-0032_Spring_Type_8/0032_spring_type_8/0032_spring_type_8_visc_RZ/Spring_TYPE8_element | rd_v_elements | SKIPS(1) |  |
| RD-V-0032_Spring_Type_8/0032_spring_type_8/0032_spring_type_8_visc_TX/Spring_TYPE8_element | rd_v_elements | SKIPS(1) |  |
| RD-V-0032_Spring_Type_8/0032_spring_type_8/0032_spring_type_8_visc_TY/Spring_TYPE8_element | rd_v_elements | SKIPS(1) |  |
| RD-V-0032_Spring_Type_8/0032_spring_type_8/0032_spring_type_8_visc_TZ/Spring_TYPE8_element | rd_v_elements | SKIPS(1) |  |
| RD-V-0033_Spring_Type_13/0033_spring_type_13_stiff_TY/0033_spring_type_13_stiff_TY/Spring_TYPE13_element | rd_v_elements | SKIPS(1) |  |
| RD-V-0033_Spring_Type_13/0033_spring_type_13_stiff_TZ/0033_spring_type_13_stiff_TZ/Spring_TYPE13_element | rd_v_elements | SKIPS(1) |  |
| RD-V-0700/Final_version_JC_LAW36_LAW2/JOHNSON_COOK_FAILURE_2024_LAW2_version2/HEXA_DEGE_LAW2/HEXA_DEGE | rd_v_failure | SKIPS(1) |  |
| RD-V-0700/Final_version_JC_LAW36_LAW2/JOHNSON_COOK_FAILURE_2024_LAW2_version2/HEXA_ELEM_SOLID_18_LAW2/HEXAP14_18 | rd_v_failure | SKIPS(1) |  |
| RD-V-0700/Final_version_JC_LAW36_LAW2/JOHNSON_COOK_FAILURE_2024_LAW2_version2/HEXA_ELEM_SOLID_24_LAW2/HEXA_LAW2 | rd_v_failure | SKIPS(1) |  |
| RD-V-0700/Final_version_JC_LAW36_LAW2/JOHNSON_COOK_FAILURE_2024_LAW2_version2/HEXA_ELEM_TSHELL_LAW2/HEXA_TSHELL_LAW2 | rd_v_failure | SKIPS(1) |  |
| RD-V-0700/Final_version_JC_LAW36_LAW2/JOHNSON_COOK_FAILURE_2024_LAW2_version2/SHELL_Ishell24_LAW2/shell_Ishell24_LAW2 | rd_v_failure | SKIPS(1) |  |
| RD-V-0700/Final_version_JC_LAW36_LAW2/JOHNSON_COOK_FAILURE_2024_LAW2_version2/TETRA_ELEM_24/TETRA_LAW2 | rd_v_failure | CLEAN |  |
| RD-V-0700/Final_version_JC_LAW36_LAW2/JOHNSON_COOK_FAILURE_2024_LAW2_version2/TRIA_LAW2/TRIA_JC_LAW2 | rd_v_failure | SKIPS(1) |  |
| RD-V-0700/Final_version_JC_LAW36_LAW2/JOHNSON_COOK_FAILURE_2024_LAW36_version2/HEXA_DEGE/HEXA_DEGE | rd_v_failure | SKIPS(1) |  |
| RD-V-0700/Final_version_JC_LAW36_LAW2/JOHNSON_COOK_FAILURE_2024_LAW36_version2/HEXA_ELEM_SOLID_18/HEXAP14_18 | rd_v_failure | SKIPS(1) |  |
| RD-V-0700/Final_version_JC_LAW36_LAW2/JOHNSON_COOK_FAILURE_2024_LAW36_version2/HEXA_ELEM_SOLID_24/HEXA1 | rd_v_failure | SKIPS(1) |  |
| RD-V-0700/Final_version_JC_LAW36_LAW2/JOHNSON_COOK_FAILURE_2024_LAW36_version2/HEXA_ELEM_TSHELL/HEXA_TSHELL | rd_v_failure | SKIPS(1) |  |
| RD-V-0700/Final_version_JC_LAW36_LAW2/JOHNSON_COOK_FAILURE_2024_LAW36_version2/SHELL_Ishell12/shell_Ishell12 | rd_v_failure | SKIPS(1) |  |
| RD-V-0700/Final_version_JC_LAW36_LAW2/JOHNSON_COOK_FAILURE_2024_LAW36_version2/SHELL_Ishell24/shell_Ishell24 | rd_v_failure | SKIPS(1) |  |
| RD-V-0700/Final_version_JC_LAW36_LAW2/JOHNSON_COOK_FAILURE_2024_LAW36_version2/TETRA_ELEM_24/TETRA | rd_v_failure | CLEAN |  |
| RD-V-0700/Final_version_JC_LAW36_LAW2/JOHNSON_COOK_FAILURE_2024_LAW36_version2/TRIA/TRIA_JC | rd_v_failure | SKIPS(1) |  |
| RD-V-0400_ALE/0400_mok_fsi_benchmark_ALE/Data/Mok_Benchmark | rd_v_fsi | SKIPS(3) | ALE/BCS; ALE/GRID; EOS/LINEAR |
| RD-V-0400_CEL/0400_mok_fsi_benchmark_CEL/CEL/data/Mok_Benchmark | rd_v_fsi | SKIPS(4) | ALE/BCS; EOS/LINEAR (+2) |
| RD-V-0300_Pressure/0300_pressure/rdv_0300_pressure | rd_v_loads | CLEAN |  |
| RD_V_0310/Model/Gravity/gravity | rd_v_loads | SKIPS(3) | INTER/TYPE24; MONVOL/AIRBAG1 |
| RD_V_0310/Model/Impacc/impacc | rd_v_loads | SKIPS(4) | IMPACC; INTER/TYPE24; MONVOL/AIRBAG1 |
| RD-V-0200_Hardening/0200_hardening/rdv_0200_hardening_modulus | rd_v_material | SKIPS(1) |  |
| RD-V-0210_Yeoh_Hyperelastic_Material/0210_Yeoh_Hyperelastic_Material_0.495/rubber | rd_v_material | SKIPS(2) |  |
| RD-V-0210_Yeoh_Hyperelastic_Material/0210_Yeoh_Hyperelastic_Material_0.49999/rubber | rd_v_material | SKIPS(2) |  |
| RD-V-0220_Foam_LAW70/0220_foam_LAW70/0220_foam_LAW70_0/BLOCK_H8 | rd_v_material | SKIPS(1) |  |
| RD-V-0220_Foam_LAW70/0220_foam_LAW70/0220_foam_LAW70_1/BLOCK_H8 | rd_v_material | SKIPS(1) |  |
| RD-V-0220_Foam_LAW70/0220_foam_LAW70/0220_foam_LAW70_2/BLOCK_H8 | rd_v_material | SKIPS(1) |  |
| RD-V-0220_Foam_LAW70/0220_foam_LAW70/0220_foam_LAW70_3/BLOCK_H8 | rd_v_material | SKIPS(1) |  |
| RD-V-0230_Fabric_LAW19/0230_fabric_LAW19/0230_shell_mat_019_01/SHELL_LAW19_PROP9 | rd_v_material | SKIPS(3) |  |
| RD-V-0230_Fabric_LAW19/0230_fabric_LAW19/0230_shell_mat_019_02/SHELL_LAW19_PROP9 | rd_v_material | SKIPS(4) | XREF |
| RD_V_0240/Modele_HEXA_P14_Isolid18/tensile_TABULATED_HEXA_TYPE14_Isolid18 | rd_v_material | SKIPS(12) |  |
| RD_V_0240/Modele_HEXA_P14_Isolid24/tensile_TABULATED_HEXA_P14 | rd_v_material | SKIPS(12) |  |
| RD_V_0240/Modele_HEXA_P20_Isolid15/tensile_TABULATED_HEXA_TYPE20 | rd_v_material | SKIPS(12) |  |
| RD_V_0240/Modele_QUAD_Ishell12/tensile_TABULATED_QUAD_Ishell12 | rd_v_material | SKIPS(12) |  |
| RD_V_0240/Modele_TETRA10_Itetra0/tensile_TABULATED_TETRA10 | rd_v_material | ERROR | TETRA10; ERR: MODEL CHECK: model has no elements |
| RD_V_0240/Modele_TETRA10_Itetra2/tensile_TABULATED_TETRA10_2 | rd_v_material | ERROR | TETRA10; ERR: MODEL CHECK: model has no elements |
| RD_V_0240/Modele_TETRA4_Itetra0/tensile_TABULATED_TETRA | rd_v_material | SKIPS(12) |  |
| RD_V_0240/Modele_TETRA4_Itetra3/tensile_TABULATED_TETRA4_3 | rd_v_material | SKIPS(12) |  |
| RD_V_0240/Modele_TRIA_Ishell24/tensile_TABULATED_TRIA | rd_v_material | SKIPS(2) |  |
| RD_V_0240/Tensile_QUAD_Ishell24/tensile_TABULATED_QUAD_Ishell24 | rd_v_material | SKIPS(12) |  |
| RD-V-0600/RD-V_0600_ADMAS_verification/ADMAS_0/ADMAS_Gravity | rd_v_tools | SKIPS(3) | INTER/TYPE24; MONVOL/AIRBAG1 |
| RD-V-0600/RD-V_0600_ADMAS_verification/ADMAS_1/ADMAS_1 | rd_v_tools | SKIPS(3) | INTER/TYPE24; MONVOL/AIRBAG1 |
| RD-V-0600/RD-V_0600_ADMAS_verification/ADMAS_2/ADMAS_2 | rd_v_tools | ERROR | INTER/TYPE24; MONVOL/AIRBAG1; ERR: CROSS REF: /ADMAS/1: node group 7 not defined |
| RD-V-0600/RD-V_0600_ADMAS_verification/ADMAS_3/ADMAS_3 | rd_v_tools | ERROR | INTER/TYPE24; MONVOL/AIRBAG1; ERR: CROSS REF: /ADMAS/1: node group 62 not defined |
| RD-V-0600/RD-V_0600_ADMAS_verification/ADMAS_4/ADMAS_4 | rd_v_tools | ERROR | INTER/TYPE24; MONVOL/AIRBAG1; ERR: CROSS REF: /ADMAS/1: node group 62 not defined |
| RD-V-0600/RD-V_0600_ADMAS_verification/ADMAS_5/ADMAS_5 | rd_v_tools | ERROR | INTER/TYPE24; MONVOL/AIRBAG1; ERR: CROSS REF: /ADMAS/1: node group 2 not defined |
| RD-V-0600/RD-V_0600_ADMAS_verification/ADMAS_6/ADMAS_6 | rd_v_tools | ERROR | INTER/TYPE24; MONVOL/AIRBAG1; ERR: CROSS REF: /ADMAS/1: node group 121 not defined |
| RD-V-0600/RD-V_0600_ADMAS_verification/ADMAS_6_Iflag1/ADMAS_6_I1 | rd_v_tools | ERROR | INTER/TYPE24; MONVOL/AIRBAG1; ERR: CROSS REF: /ADMAS/1: node group 121 not defined |
| RD-V-0600/RD-V_0600_ADMAS_verification/ADMAS_7/ADMAS_7 | rd_v_tools | ERROR | INTER/TYPE24; MONVOL/AIRBAG1; ERR: CROSS REF: /ADMAS/1: node group 121 not defined |
| RD-V-0600/RD-V_0600_ADMAS_verification/ADMAS_7_Iflag1/ADMAS_7_I1 | rd_v_tools | ERROR | INTER/TYPE24; MONVOL/AIRBAG1; ERR: CROSS REF: /ADMAS/1: node group 121 not defined |
| RD-HWX-T-1000/solved_completed_model/data/tensile_completed | tutorial | SKIPS(1) |  |
| RD-HWX-T-1000/unsolved_starter_model/data/tensile_start | tutorial | ERROR | ERR: PART CHECK: /PART/3: property 0 not defined; ERR: PART CHECK: /PART/3: material 0 not defined (+2) |
| RD-HWX-T-1010/solved/DYREL/data/cantilever_completed_DYREL | tutorial | SKIPS(2) | INTER/TYPE24 |
| RD-HWX-T-1010/solved/data/cantilever_completed | tutorial | SKIPS(2) | INTER/TYPE24 |
| RD-HWX-T-1010/unsolved/data/cantilever_start | tutorial | ERROR | ERR: PART CHECK: /PART/1: property 0 not defined; ERR: PART CHECK: /PART/1: material 0 not defined (+6) |
| RD-HWX-T-1020/solved/data/Front_Impact_completed | tutorial | ERROR | GRNOD/NODENS; ERR: Front_Impact_completed_0000.rad:27835: /FAIL/BIQUAD/2: all five failure strains c1..c5 must be > 0 (presets not ported); ERR: Front_Impact_completed_0000.rad:27970: /INTER/TYPE7/4: Igap=2 not ported (0 constant, 1 variable) |
| RD-HWX-T-1020/unsolved/data/Front_Impact_initial | tutorial | ERROR | ERR: PART CHECK: /PART/4: property 0 not defined; ERR: PART CHECK: /PART/4: material 0 not defined (+8) |
| RD-HWX-T-1030/solved/data/boat_ditching_completed | tutorial | SKIPS(2) | EOS/LINEAR; INTER/TYPE18 |
| RD-HWX-T-1040/solved/data/gasket_completed | tutorial | SKIPS(1) |  |
| RD-HWX-T-1050/solved/data1/valve_completed | tutorial | SKIPS(2) | EOS/LINEAR; INTER/TYPE18 |
| RD-HWX-T-1050/solved/data2/valve_NRF_outlet | tutorial | SKIPS(2) | EOS/LINEAR; INTER/TYPE18 |
| RD-HWX-T-1060/solved/data/3PB_completed | tutorial | ERROR | FAIL/CONNECT; ERR: 3PB_completed_0000.rad:14029: /INTER/TYPE7/2: Igap=2 not ported (0 constant, 1 variable) |
| RD-HWX-T-1060/unsolved/data/3PB_initial | tutorial | ERROR | ERR: PART CHECK: /PART/2: property 0 not defined; ERR: PART CHECK: /PART/2: material 0 not defined (+6) |
| TENSILE/TENSILE | tutorial | ERROR | ERR: PART CHECK: /PART/1: property 0 not defined; ERR: PART CHECK: /PART/1: material 0 not defined (+2) |
| phone_start/phone_start | tutorial | ERROR | ERR: PART CHECK: /PART/23: property 0 not defined; ERR: PART CHECK: /PART/23: material 0 not defined (+3) |

### (b) Ranked keyword-gap table

cases_using = deck contains the unsupported family; cases_blocking = family is a hard physics skip in that run; sole_blocker = it is the case's ONLY hard gap; blocks = total skipped blocks corpus-wide.

| rank | unsupported family | cases using | cases blocking | sole blocker | blocks |
|---|---|---|---|---|---|
| 1 | INTER/TYPE24 | 18 | 18 | 5 | 18 |
| 2 | MONVOL/AIRBAG1 | 16 | 16 | 2 | 16 |
| 3 | INTER/LAGMUL | 14 | 14 | 2 | 58 |
| 4 | ALE/BCS | 12 | 12 | 7 | 25 |
| 5 | SHEL16 | 12 | 12 | 0 | 62 |
| 6 | QUAD | 10 | 10 | 5 | 33 |
| 7 | INTER/TYPE18 | 7 | 7 | 0 | 7 |
| 8 | MOVE_FUNCT | 6 | 6 | 1 | 23 |
| 9 | EOS/LINEAR | 6 | 6 | 0 | 6 |
| 10 | AMS | 5 | 5 | 5 | 5 |
| 11 | TRANSFORM/TRA | 5 | 5 | 0 | 5 |
| 12 | ENDSUB | 5 | 5 | 0 | 7 |
| 13 | SUBMODEL | 5 | 5 | 0 | 7 |
| 14 | INTER/TYPE10 | 5 | 5 | 5 | 5 |
| 15 | TETRA10 | 5 | 5 | 5 | 5 |
| 16 | RWALL/PARAL | 4 | 4 | 0 | 6 |
| 17 | EOS/STIFF-GAS | 4 | 4 | 0 | 4 |
| 18 | SPHCEL | 4 | 4 | 0 | 5 |
| 19 | SUBDOMAIN | 4 | 4 | 1 | 4 |
| 20 | FAIL/SNCONNECT | 4 | 4 | 0 | 4 |
| - | (154 more families, each blocking <= 5 cases) | | | | |

**Highest-value next port target:** See ranked gaps above.

### (c) CRASH list and reader-bug signatures

**rc!=0 crashes: NONE** - no deck produced an uncaught traceback. Every reader failure was caught and reported.

| # | parse-failure signature | cases | example |
|---|---|---|---|
| - | NONE | 0 | |
