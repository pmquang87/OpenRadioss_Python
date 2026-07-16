### (a) Per-case verdict table (all 529 runnable official decks)

verdicts: CLEAN = rc0, nothing skipped (control cards excluded); SKIPS(n) = rc0, n non-control keyword families skipped; ERROR = starter refused the model (rc2, collected errors); CRASH = uncaught traceback (rc!=0). Blockers: hard keyword gaps, then PARSE = caught reader failure, ERR = model error (top 3 shown).

| case | category | verdict | blockers |
|---|---|---|---|
| RD-E-0100_Twisted_beam/01_Twisted_Beam/BATOZ/TWISBEAM | rd_e | ERROR | GRNOD/GRNOD; PARSE /TH/NODE/1: int( '01x3' |
| RD-E-0100_Twisted_beam/01_Twisted_Beam/DKT18/TWISBEAM | rd_e | ERROR | PARSE /TH/NODE/1: int( '01x3' |
| RD-E-0100_Twisted_beam/01_Twisted_Beam/QEPH/TWISBEAM | rd_e | ERROR | GRNOD/GRNOD; PARSE /TH/NODE/1: int( '01x3' |
| RD-E-0200_Snap_thru_Roof/02_Snap-through/Explicit_solver/SNAP_EXP | rd_e | ERROR | SKEW/FIX; PARSE /TH/NODE/4: int( '0Node' |
| RD-E-0200_Snap_thru_Roof/02_Snap-through/Implicit_solver/SNAP_IMP | rd_e | ERROR | SKEW/FIX; PARSE /TH/NODE/4: int( '0Node' |
| RD-E-0300_S-Beam/03_S-Beam/global_IP/BATOZ/S_BEAM | rd_e | ERROR | GRNOD/SURF; PARSE /TH/NODE/2: int( 'Displacement' |
| RD-E-0300_S-Beam/03_S-Beam/global_IP/BT-type3/S_BEAM | rd_e | ERROR | GRNOD/SURF; PARSE /TH/NODE/2: int( 'Displacement' |
| RD-E-0300_S-Beam/03_S-Beam/global_IP/QEPH/S_BEAM | rd_e | ERROR | GRNOD/SURF; PARSE /TH/NODE/2: int( 'Displacement' |
| RD-E-0300_S-Beam/03_S-Beam/v_10ms/QEPH/S_BEAM | rd_e | ERROR | GRNOD/SURF; PARSE /TH/NODE/2: int( 'Displacement' |
| RD-E-0300_S-Beam/03_S-Beam/v_5ms/BATOZ/S_BEAM | rd_e | ERROR | GRNOD/SURF; PARSE /TH/NODE/2: int( 'Displacement' |
| RD-E-0300_S-Beam/03_S-Beam/v_5ms/BT-type3/S_BEAM | rd_e | ERROR | GRNOD/SURF; PARSE /TH/NODE/2: int( 'Displacement' |
| RD-E-0300_S-Beam/03_S-Beam/v_5ms/QEPH/S_BEAM | rd_e | ERROR | GRNOD/SURF; PARSE /TH/NODE/2: int( 'Displacement' |
| RD-E-0400_Airbag/04_Airbag/driver/driver_airbag | rd_e | ERROR | GRSH3N/SH3N; INTER/TYPE19 (+12) |
| RD-E-0500_Beam_frame/05_Beam-frame/FRAME | rd_e | ERROR | PARSE /TH/NODE/2: int( '0Node'; ERR /PROP/BEAM/1: section card 'Area Iyy Izz Ixx' missing (+2) |
| RD-E-0601_Fuel_tank/1-Tank_sloshing/data/TANK | rd_e | ERROR | ALE/BCS; MAT/LAW37 (+10) |
| RD-E-0602_Fuel_flow/2-Tank_overturning/Fluid_flow_1/data/PFTANK | rd_e | ERROR | ALE/BCS; GRNOD/GENE (+9) |
| RD-E-0602_Fuel_flow/2-Tank_overturning/Fluid_flow_2/data/PFTANK | rd_e | ERROR | ALE/BCS; GRNOD/GENE (+9) |
| RD-E-0700_Pendulums/07_Pendulums/pendulum | rd_e | ERROR | GRNOD/GENE; INTER/TYPE24 (+2) |
| RD-E-0800_Hopkinson_bar/08_Hopkinson_Bar/High_strain_rate/SHPB_H | rd_e | ERROR | QUAD; PARSE /TH/NODE/1: int( '0Middel_node_group_0' (+8) |
| RD-E-0800_Hopkinson_bar/08_Hopkinson_Bar/High_strain_rate/SHPB_H_2021_FEB12 | rd_e | ERROR | QUAD; PARSE /TH/NODE/1: int( '0Middel_node_group_ (+11) |
| RD-E-0800_Hopkinson_bar/08_Hopkinson_Bar/Low_strain_rate/SHPB_L | rd_e | ERROR | QUAD; PARSE /TH/NODE/1: int( '0Middel_node_group_0' (+8) |
| RD-E-0800_Hopkinson_bar/08_Hopkinson_Bar/Low_strain_rate/SHPB_L_2021_FEB12 | rd_e | ERROR | QUAD; PARSE /TH/NODE/1: int( '0Middel_node_group_ (+11) |
| RD-E-0901_Billiards/Billiards_model/BILLARD | rd_e | ERROR | GRBRIC/BRIC; GRBRIC/PART (+12) |
| RD-E-0901_Billiards/Billiards_model/Supplement_Interface7Lag/BILLARD | rd_e | ERROR | GRBRIC/PART; INTER/LAGMUL (+12) |
| RD-E-0902_Collision/Collision_simulation/COLLISION | rd_e | ERROR | GRBRIC/PART; INTER/LAGMUL (+4) |
| RD-E-0903_Test/Contact_modelling/Inter_16_sliding/TEST16S | rd_e | ERROR | GRBRIC/BRIC; GRBRIC/PART (+7) |
| RD-E-0903_Test/Contact_modelling/Inter_16_tied/TEST16T | rd_e | ERROR | GRBRIC/BRIC; GRBRIC/PART (+5) |
| RD-E-0903_Test/Contact_modelling/Inter_17_sliding/TEST17S | rd_e | ERROR | GRBRIC/PART; INTER/LAGMUL (+4) |
| RD-E-0903_Test/Contact_modelling/Inter_17_tied/TEST17ST | rd_e | ERROR | GRBRIC/PART; INTER/LAGMUL (+4) |
| RD-E-0903_Test/Contact_modelling/Inter_7_Lagrangian/TEST7L | rd_e | ERROR | GRBRIC/PART; INTER/LAGMUL (+4) |
| RD-E-0903_Test/Contact_modelling/Inter_7_Penality/TEST7P | rd_e | ERROR | GRBRIC/PART; PARSE /TH/PART/1: int( 'XMOM' |
| RD-E-1000_Bending/10_Bending/BATOZ/Sf_0.6/ROLLING | rd_e | ERROR | PARSE /IMPVEL/1: 'XX'; PARSE /TH/NODE/6: int( '0point1_group_0' |
| RD-E-1000_Bending/10_Bending/BATOZ/Sf_0.8/ROLLING | rd_e | ERROR | PARSE /IMPVEL/1: 'XX'; PARSE /TH/NODE/6: int( '0point1_group_0' |
| RD-E-1000_Bending/10_Bending/BATOZ/Sf_0.9/ROLLING | rd_e | ERROR | PARSE /IMPVEL/1: 'XX'; PARSE /TH/NODE/6: int( '0point1_group_0' |
| RD-E-1000_Bending/10_Bending/BT/BT_type1/Sf_0.1/ROLLING | rd_e | ERROR | GRNOD/GRNOD; PARSE /IMPVEL/1: 'XX' (+1) |
| RD-E-1000_Bending/10_Bending/BT/BT_type1/Sf_0.9/ROLLING | rd_e | ERROR | GRNOD/GRNOD; PARSE /IMPVEL/1: 'XX' (+1) |
| RD-E-1000_Bending/10_Bending/BT/BT_type3/Sf_0.1/ROLLING | rd_e | ERROR | GRNOD/GRNOD; PARSE /IMPVEL/1: 'XX' (+1) |
| RD-E-1000_Bending/10_Bending/BT/BT_type3/Sf_0.9/ROLLING | rd_e | ERROR | GRNOD/GRNOD; PARSE /IMPVEL/1: 'XX' (+1) |
| RD-E-1000_Bending/10_Bending/BT/BT_type4/Sf_0.1/ROLLING | rd_e | ERROR | GRNOD/GRNOD; PARSE /IMPVEL/1: 'XX' (+1) |
| RD-E-1000_Bending/10_Bending/BT/BT_type4/Sf_0.9/ROLLING | rd_e | ERROR | GRNOD/GRNOD; PARSE /IMPVEL/1: 'XX' (+1) |
| RD-E-1000_Bending/10_Bending/DKT18/Sf_0.1/ROLLING | rd_e | ERROR | PARSE /IMPVEL/1: 'XX'; PARSE /TH/NODE/6: int( '0point1_group_0' |
| RD-E-1000_Bending/10_Bending/DKT18/Sf_0.2/ROLLING | rd_e | ERROR | PARSE /IMPVEL/1: 'XX'; PARSE /TH/NODE/6: int( '0point1_group_0' |
| RD-E-1000_Bending/10_Bending/DKT18/Sf_0.3/ROLLING | rd_e | ERROR | PARSE /IMPVEL/1: 'XX'; PARSE /TH/NODE/6: int( '0point1_group_0' |
| RD-E-1000_Bending/10_Bending/QEPH/Sf_0.8/ROLLING | rd_e | ERROR | PARSE /IMPVEL/1: 'XX'; PARSE /TH/NODE/6: int( '0point1_group_0' |
| RD-E-1000_Bending/10_Bending/QEPH/Sf_0.9/ROLLING | rd_e | ERROR | PARSE /IMPVEL/1: 'XX'; PARSE /TH/NODE/6: int( '0point1_group_0' |
| RD-E-1101_Tensile_test/1_LAW2/tensile_LAW2 | rd_e | ERROR | FRAME/MOV; ERR /NODE card needs 4 fields, got 3 (+9) |
| RD-E-1101_Tensile_test/2_LAW36/tensile_LAW36 | rd_e | ERROR | FRAME/MOV; PARSE /MAT/PLAS_TAB/2: int( '1.0' (+10) |
| RD-E-1101_Tensile_test/3_LAW2_BIQUAD/tensile_LAW2_BIQUAD | rd_e | ERROR | FRAME/MOV; ERR /FAIL/BIQUAD/2: all five failure strains c1..c5 must be  (+9) |
| RD-E-1101_Tensile_test/4_LAW36_BIQUAD/tensile_LAW36_BIQUAD | rd_e | ERROR | FRAME/MOV; ERR /FAIL/BIQUAD/2: all five failure strains c1..c5 must be  (+9) |
| RD-E-1102_Strain_rate_effect/5_law2_strain_rate/tensile_LAW2_strain_rate | rd_e | ERROR | FRAME/MOV; ERR /NODE card needs 4 fields, got 3 (+9) |
| RD-E-1102_Strain_rate_effect/6_law36_strain_rate/tensile_LAW36_strain_rate | rd_e | ERROR | FRAME/MOV; ERR /TH/NODE/2: unknown node 1 (+1) |
| RD-E-1200_Bicycle/12_Bicycle/Bike/BIKERC_1506_1104 | rd_e | ERROR | GRNOD/GRNOD; GRSHEL/SHEL (+12) |
| RD-E-1300_Shock_tube/13_Shock_tube/Blast_experiment/blast_experiment | rd_e | ERROR | ALE/BCS; ALE/MAT (+11) |
| RD-E-1300_Shock_tube/13_Shock_tube/Shock_experiment/shock_experiment | rd_e | ERROR | ALE/BCS; ALE/MAT (+12) |
| RD-E-1500_Gears/15_Gears/Inter16/DIF24416 | rd_e | ERROR | BRIC20; FRAME/FIX (+12) |
| RD-E-1500_Gears/15_Gears/Inter17/DIF24416 | rd_e | ERROR | BRIC20; FRAME/FIX (+12) |
| RD-E-1601_Explicit/EXPLICIT_solver/ADYREL/data/SEAT_ADYREL | rd_e | ERROR | GRNOD/GENE; GRSH3N/SH3N (+12) |
| RD-E-1601_Explicit/EXPLICIT_solver/DYREL/data/SEAT_DYREL | rd_e | ERROR | GRNOD/GENE; GRSH3N/SH3N (+12) |
| RD-E-1601_Explicit/EXPLICIT_solver/KEREL/data/SEAT_KEREL | rd_e | ERROR | GRNOD/GENE; GRSH3N/SH3N (+12) |
| RD-E-1601_Explicit/EXPLICIT_solver/RAYLEIGH/data/SEAT_RAYLEIGH | rd_e | ERROR | GRNOD/GENE; GRSH3N/SH3N (+12) |
| RD-E-1601_Explicit/EXPLICIT_solver/Without_damping/data/SEAT | rd_e | ERROR | GRNOD/GENE; GRSH3N/SH3N (+12) |
| RD-E-1602_Implicit/IMPLICIT_solver/Nonlinear/data/SEAT | rd_e | ERROR | GRNOD/GENE; GRSH3N/SH3N (+12) |
| RD-E-1701_Densities/Densities_mesh/mesh0/batoz/BOXBEAM | rd_e | ERROR | GRNOD/SURF; LINE/EDGE (+4) |
| RD-E-1701_Densities/Densities_mesh/mesh0/bt1/BOXBEAM | rd_e | ERROR | GRNOD/SURF; LINE/EDGE (+4) |
| RD-E-1701_Densities/Densities_mesh/mesh0/bt3/BOXBEAM | rd_e | ERROR | GRNOD/SURF; LINE/EDGE (+4) |
| RD-E-1701_Densities/Densities_mesh/mesh0/bt4/BOXBEAM | rd_e | ERROR | GRNOD/SURF; LINE/EDGE (+4) |
| RD-E-1701_Densities/Densities_mesh/mesh0/c0/BOXBEAM | rd_e | ERROR | GRNOD/SURF; LINE/EDGE (+4) |
| RD-E-1701_Densities/Densities_mesh/mesh0/dkt18/BOXBEAM | rd_e | ERROR | GRNOD/SURF; LINE/EDGE (+4) |
| RD-E-1701_Densities/Densities_mesh/mesh0/qeph/BOXBEAM | rd_e | ERROR | GRNOD/SURF; LINE/EDGE (+4) |
| RD-E-1701_Densities/Densities_mesh/mesh1/batoz/BOXBEAM | rd_e | ERROR | GRNOD/SURF; LINE/EDGE (+4) |
| RD-E-1701_Densities/Densities_mesh/mesh1/bt1/BOXBEAM | rd_e | ERROR | GRNOD/SURF; LINE/EDGE (+4) |
| RD-E-1701_Densities/Densities_mesh/mesh1/bt3/BOXBEAM | rd_e | ERROR | GRNOD/SURF; LINE/EDGE (+4) |
| RD-E-1701_Densities/Densities_mesh/mesh1/bt4/BOXBEAM | rd_e | ERROR | GRNOD/SURF; LINE/EDGE (+4) |
| RD-E-1701_Densities/Densities_mesh/mesh1/c0/BOXBEAM | rd_e | ERROR | GRNOD/SURF; LINE/EDGE (+4) |
| RD-E-1701_Densities/Densities_mesh/mesh1/dkt18/BOXBEAM | rd_e | ERROR | GRNOD/SURF; LINE/EDGE (+4) |
| RD-E-1701_Densities/Densities_mesh/mesh1/qeph/BOXBEAM | rd_e | ERROR | GRNOD/SURF; LINE/EDGE (+4) |
| RD-E-1701_Densities/Densities_mesh/mesh2/batoz/BOXBEAM | rd_e | ERROR | GRNOD/SURF; LINE/EDGE (+4) |
| RD-E-1701_Densities/Densities_mesh/mesh2/bt1/BOXBEAM | rd_e | ERROR | GRNOD/SURF; LINE/EDGE (+4) |
| RD-E-1701_Densities/Densities_mesh/mesh2/bt3/BOXBEAM | rd_e | ERROR | GRNOD/SURF; LINE/EDGE (+4) |
| RD-E-1701_Densities/Densities_mesh/mesh2/bt4/BOXBEAM | rd_e | ERROR | GRNOD/SURF; LINE/EDGE (+4) |
| RD-E-1701_Densities/Densities_mesh/mesh2/c0/BOXBEAM | rd_e | ERROR | GRNOD/SURF; LINE/EDGE (+4) |
| RD-E-1701_Densities/Densities_mesh/mesh2/dkt18/BOXBEAM | rd_e | ERROR | GRNOD/SURF; LINE/EDGE (+4) |
| RD-E-1701_Densities/Densities_mesh/mesh2/qeph/BOXBEAM | rd_e | ERROR | GRNOD/SURF; LINE/EDGE (+4) |
| RD-E-1701_Densities/Densities_mesh/mesh3/batoz/BOXBEAM | rd_e | ERROR | GRNOD/SURF; LINE/EDGE (+4) |
| RD-E-1701_Densities/Densities_mesh/mesh3/bt1/BOXBEAM | rd_e | ERROR | GRNOD/SURF; LINE/EDGE (+4) |
| RD-E-1701_Densities/Densities_mesh/mesh3/bt3/BOXBEAM | rd_e | ERROR | GRNOD/SURF; LINE/EDGE (+4) |
| RD-E-1701_Densities/Densities_mesh/mesh3/bt4/BOXBEAM | rd_e | ERROR | GRNOD/SURF; LINE/EDGE (+4) |
| RD-E-1701_Densities/Densities_mesh/mesh3/c0/BOXBEAM | rd_e | ERROR | GRNOD/SURF; LINE/EDGE (+4) |
| RD-E-1701_Densities/Densities_mesh/mesh3/dkt18/BOXBEAM | rd_e | ERROR | GRNOD/SURF; LINE/EDGE (+4) |
| RD-E-1701_Densities/Densities_mesh/mesh3/qeph/BOXBEAM | rd_e | ERROR | GRNOD/SURF; LINE/EDGE (+4) |
| RD-E-1702_Transitions/Transition_mesh/mesh0/quad/batoz/5ip/BOXBEAM | rd_e | ERROR | GRNOD/GRNOD; GRNOD/SURF (+5) |
| RD-E-1702_Transitions/Transition_mesh/mesh0/quad/batoz/GLOBAL_PLAS/BOXBEAM | rd_e | ERROR | GRNOD/GRNOD; GRNOD/SURF (+5) |
| RD-E-1702_Transitions/Transition_mesh/mesh0/quad/batoz/ITER_PLAS/BOXBEAM | rd_e | ERROR | GRNOD/GRNOD; GRNOD/SURF (+5) |
| RD-E-1702_Transitions/Transition_mesh/mesh0/quad/bt1/5ip/BOXBEAM | rd_e | ERROR | GRNOD/GRNOD; GRNOD/SURF (+5) |
| RD-E-1702_Transitions/Transition_mesh/mesh0/quad/bt1/GLOBAL_PLAS/BOXBEAM | rd_e | ERROR | GRNOD/GRNOD; GRNOD/SURF (+5) |
| RD-E-1702_Transitions/Transition_mesh/mesh0/quad/bt1/ITER_PLAS/BOXBEAM | rd_e | ERROR | GRNOD/GRNOD; GRNOD/SURF (+5) |
| RD-E-1702_Transitions/Transition_mesh/mesh0/quad/bt3/5ip/BOXBEAM | rd_e | ERROR | GRNOD/GRNOD; GRNOD/SURF (+5) |
| RD-E-1702_Transitions/Transition_mesh/mesh0/quad/bt3/GLOBAL_PLAS/BOXBEAM | rd_e | ERROR | GRNOD/GRNOD; GRNOD/SURF (+5) |
| RD-E-1702_Transitions/Transition_mesh/mesh0/quad/bt3/ITER_PLAS/BOXBEAM | rd_e | ERROR | GRNOD/GRNOD; GRNOD/SURF (+5) |
| RD-E-1702_Transitions/Transition_mesh/mesh0/quad/bt4/5ip/BOXBEAM | rd_e | ERROR | GRNOD/GRNOD; GRNOD/SURF (+5) |
| RD-E-1702_Transitions/Transition_mesh/mesh0/quad/bt4/GLOBAL_PLAS/BOXBEAM | rd_e | ERROR | GRNOD/GRNOD; GRNOD/SURF (+5) |
| RD-E-1702_Transitions/Transition_mesh/mesh0/quad/bt4/ITER_PLAS/BOXBEAM | rd_e | ERROR | GRNOD/GRNOD; GRNOD/SURF (+5) |
| RD-E-1702_Transitions/Transition_mesh/mesh0/quad/qeph/5ip/BOXBEAM | rd_e | ERROR | GRNOD/GRNOD; GRNOD/SURF (+5) |
| RD-E-1702_Transitions/Transition_mesh/mesh0/quad/qeph/GLOBAL_PLAS/BOXBEAM | rd_e | ERROR | GRNOD/GRNOD; GRNOD/SURF (+5) |
| RD-E-1702_Transitions/Transition_mesh/mesh0/quad/qeph/ITER_PLAS/BOXBEAM | rd_e | ERROR | GRNOD/GRNOD; GRNOD/SURF (+5) |
| RD-E-1702_Transitions/Transition_mesh/mesh0/tri/c0/5ip/BOXBEAM | rd_e | ERROR | GRNOD/SURF; LINE/EDGE (+4) |
| RD-E-1702_Transitions/Transition_mesh/mesh0/tri/c0/GLOBAL_PLAS/BOXBEAM | rd_e | ERROR | GRNOD/SURF; LINE/EDGE (+4) |
| RD-E-1702_Transitions/Transition_mesh/mesh0/tri/c0/ITER_PLAS/BOXBEAM | rd_e | ERROR | GRNOD/SURF; LINE/EDGE (+4) |
| RD-E-1702_Transitions/Transition_mesh/mesh0/tri/dkt18/5ip/BOXBEAM | rd_e | ERROR | GRNOD/SURF; LINE/EDGE (+4) |
| RD-E-1702_Transitions/Transition_mesh/mesh0/tri/dkt18/GLOBAL_PLAS/BOXBEAM | rd_e | ERROR | GRNOD/SURF; LINE/EDGE (+4) |
| RD-E-1702_Transitions/Transition_mesh/mesh0/tri/dkt18/ITER_PLAS/BOXBEAM | rd_e | ERROR | GRNOD/SURF; LINE/EDGE (+4) |
| RD-E-1702_Transitions/Transition_mesh/mesh1/batoz/5ip/BOXBEAM | rd_e | ERROR | GRNOD/SURF; LINE/EDGE (+4) |
| RD-E-1702_Transitions/Transition_mesh/mesh1/batoz/GLOBAL_PLAS/BOXBEAM | rd_e | ERROR | GRNOD/SURF; LINE/EDGE (+4) |
| RD-E-1702_Transitions/Transition_mesh/mesh1/batoz/ITER_PLAS/BOXBEAM | rd_e | ERROR | GRNOD/SURF; LINE/EDGE (+4) |
| RD-E-1702_Transitions/Transition_mesh/mesh1/bt1/5ip/BOXBEAM | rd_e | ERROR | GRNOD/SURF; LINE/EDGE (+4) |
| RD-E-1702_Transitions/Transition_mesh/mesh1/bt1/GLOBAL_PLAS/BOXBEAM | rd_e | ERROR | GRNOD/SURF; LINE/EDGE (+4) |
| RD-E-1702_Transitions/Transition_mesh/mesh1/bt1/ITER_PLAS/BOXBEAM | rd_e | ERROR | GRNOD/SURF; LINE/EDGE (+4) |
| RD-E-1702_Transitions/Transition_mesh/mesh1/bt3/5ip/BOXBEAM | rd_e | ERROR | GRNOD/SURF; LINE/EDGE (+4) |
| RD-E-1702_Transitions/Transition_mesh/mesh1/bt3/GLOBAL_PLAS/BOXBEAM | rd_e | ERROR | GRNOD/SURF; LINE/EDGE (+4) |
| RD-E-1702_Transitions/Transition_mesh/mesh1/bt3/ITER_PLAS/BOXBEAM | rd_e | ERROR | GRNOD/SURF; LINE/EDGE (+4) |
| RD-E-1702_Transitions/Transition_mesh/mesh1/bt4/5ip/BOXBEAM | rd_e | ERROR | GRNOD/SURF; LINE/EDGE (+4) |
| RD-E-1702_Transitions/Transition_mesh/mesh1/bt4/GLOBAL_PLAS/BOXBEAM | rd_e | ERROR | GRNOD/SURF; LINE/EDGE (+4) |
| RD-E-1702_Transitions/Transition_mesh/mesh1/bt4/ITER_PLAS/BOXBEAM | rd_e | ERROR | GRNOD/SURF; LINE/EDGE (+4) |
| RD-E-1702_Transitions/Transition_mesh/mesh1/qeph/5ip/BOXBEAM | rd_e | ERROR | GRNOD/SURF; LINE/EDGE (+4) |
| RD-E-1702_Transitions/Transition_mesh/mesh1/qeph/GLOBAL_PLAS/BOXBEAM | rd_e | ERROR | GRNOD/SURF; LINE/EDGE (+4) |
| RD-E-1702_Transitions/Transition_mesh/mesh1/qeph/ITER_PLAS/BOXBEAM | rd_e | ERROR | GRNOD/SURF; LINE/EDGE (+4) |
| RD-E-1702_Transitions/Transition_mesh/mesh2/batoz/5ip/BOXBEAM | rd_e | ERROR | GRNOD/GRNOD; GRNOD/SURF (+5) |
| RD-E-1702_Transitions/Transition_mesh/mesh2/batoz/GLOBAL_PLAS/BOXBEAM | rd_e | ERROR | GRNOD/GRNOD; GRNOD/SURF (+5) |
| RD-E-1702_Transitions/Transition_mesh/mesh2/batoz/ITER_PLAS/BOXBEAM | rd_e | ERROR | GRNOD/GRNOD; GRNOD/SURF (+5) |
| RD-E-1702_Transitions/Transition_mesh/mesh2/bt1/5ip/BOXBEAM | rd_e | ERROR | GRNOD/GRNOD; GRNOD/SURF (+5) |
| RD-E-1702_Transitions/Transition_mesh/mesh2/bt1/GLOBAL_PLAS/BOXBEAM | rd_e | ERROR | GRNOD/GRNOD; GRNOD/SURF (+5) |
| RD-E-1702_Transitions/Transition_mesh/mesh2/bt1/ITER_PLAS/BOXBEAM | rd_e | ERROR | GRNOD/GRNOD; GRNOD/SURF (+5) |
| RD-E-1702_Transitions/Transition_mesh/mesh2/bt3/5ip/BOXBEAM | rd_e | ERROR | GRNOD/GRNOD; GRNOD/SURF (+5) |
| RD-E-1702_Transitions/Transition_mesh/mesh2/bt3/GLOBAL_PLAS/BOXBEAM | rd_e | ERROR | GRNOD/GRNOD; GRNOD/SURF (+5) |
| RD-E-1702_Transitions/Transition_mesh/mesh2/bt3/ITER_PLAS/BOXBEAM | rd_e | ERROR | GRNOD/GRNOD; GRNOD/SURF (+5) |
| RD-E-1702_Transitions/Transition_mesh/mesh2/bt4/5ip/BOXBEAM | rd_e | ERROR | GRNOD/GRNOD; GRNOD/SURF (+5) |
| RD-E-1702_Transitions/Transition_mesh/mesh2/bt4/GLOBAL_PLAS/BOXBEAM | rd_e | ERROR | GRNOD/GRNOD; GRNOD/SURF (+5) |
| RD-E-1702_Transitions/Transition_mesh/mesh2/bt4/ITER_PLAS/BOXBEAM | rd_e | ERROR | GRNOD/GRNOD; GRNOD/SURF (+5) |
| RD-E-1702_Transitions/Transition_mesh/mesh2/c0/5ip/BOXBEAM | rd_e | ERROR | GRNOD/SURF; LINE/EDGE (+4) |
| RD-E-1702_Transitions/Transition_mesh/mesh2/c0/GLOBAL_PLAS/BOXBEAM | rd_e | ERROR | GRNOD/SURF; LINE/EDGE (+4) |
| RD-E-1702_Transitions/Transition_mesh/mesh2/c0/ITER_PLAS/BOXBEAM | rd_e | ERROR | GRNOD/SURF; LINE/EDGE (+4) |
| RD-E-1702_Transitions/Transition_mesh/mesh2/dkt18/5ip/BOXBEAM | rd_e | ERROR | GRNOD/SURF; LINE/EDGE (+4) |
| RD-E-1702_Transitions/Transition_mesh/mesh2/dkt18/GLOBAL_PLAS/BOXBEAM | rd_e | ERROR | GRNOD/SURF; LINE/EDGE (+4) |
| RD-E-1702_Transitions/Transition_mesh/mesh2/dkt18/ITER_PLAS/BOXBEAM | rd_e | ERROR | GRNOD/SURF; LINE/EDGE (+4) |
| RD-E-1702_Transitions/Transition_mesh/mesh2/qeph/5ip/BOXBEAM | rd_e | ERROR | GRNOD/GRNOD; GRNOD/SURF (+5) |
| RD-E-1702_Transitions/Transition_mesh/mesh2/qeph/GLOBAL_PLAS/BOXBEAM | rd_e | ERROR | GRNOD/GRNOD; GRNOD/SURF (+5) |
| RD-E-1702_Transitions/Transition_mesh/mesh2/qeph/ITER_PLAS/BOXBEAM | rd_e | ERROR | GRNOD/GRNOD; GRNOD/SURF (+5) |
| RD-E-1702_Transitions/Transition_mesh/mesh3/batoz/5ip/BOXBEAM | rd_e | ERROR | GRNOD/SURF; LINE/EDGE (+4) |
| RD-E-1702_Transitions/Transition_mesh/mesh3/batoz/GLOBAL_PLAS/BOXBEAM | rd_e | ERROR | GRNOD/SURF; LINE/EDGE (+4) |
| RD-E-1702_Transitions/Transition_mesh/mesh3/batoz/ITER_PLAS/BOXBEAM | rd_e | ERROR | GRNOD/SURF; LINE/EDGE (+4) |
| RD-E-1702_Transitions/Transition_mesh/mesh3/bt1/5ip/BOXBEAM | rd_e | ERROR | GRNOD/SURF; LINE/EDGE (+4) |
| RD-E-1702_Transitions/Transition_mesh/mesh3/bt1/GLOBAL_PLAS/BOXBEAM | rd_e | ERROR | GRNOD/SURF; LINE/EDGE (+4) |
| RD-E-1702_Transitions/Transition_mesh/mesh3/bt1/ITER_PLAS/BOXBEAM | rd_e | ERROR | GRNOD/SURF; LINE/EDGE (+4) |
| RD-E-1702_Transitions/Transition_mesh/mesh3/bt3/5ip/BOXBEAM | rd_e | ERROR | GRNOD/SURF; LINE/EDGE (+4) |
| RD-E-1702_Transitions/Transition_mesh/mesh3/bt3/GLOBAL_PLAS/BOXBEAM | rd_e | ERROR | GRNOD/SURF; LINE/EDGE (+4) |
| RD-E-1702_Transitions/Transition_mesh/mesh3/bt3/ITER_PLAS/BOXBEAM | rd_e | ERROR | GRNOD/SURF; LINE/EDGE (+4) |
| RD-E-1702_Transitions/Transition_mesh/mesh3/bt4/5ip/BOXBEAM | rd_e | ERROR | GRNOD/SURF; LINE/EDGE (+4) |
| RD-E-1702_Transitions/Transition_mesh/mesh3/bt4/GLOBAL_PLAS/BOXBEAM | rd_e | ERROR | GRNOD/SURF; LINE/EDGE (+4) |
| RD-E-1702_Transitions/Transition_mesh/mesh3/bt4/ITER_PLAS/BOXBEAM | rd_e | ERROR | GRNOD/SURF; LINE/EDGE (+4) |
| RD-E-1702_Transitions/Transition_mesh/mesh3/c0/5ip/BOXBEAM | rd_e | ERROR | GRNOD/SURF; LINE/EDGE (+4) |
| RD-E-1702_Transitions/Transition_mesh/mesh3/c0/GLOBAL_PLAS/BOXBEAM | rd_e | ERROR | GRNOD/SURF; LINE/EDGE (+4) |
| RD-E-1702_Transitions/Transition_mesh/mesh3/c0/ITER_PLAS/BOXBEAM | rd_e | ERROR | GRNOD/SURF; LINE/EDGE (+4) |
| RD-E-1702_Transitions/Transition_mesh/mesh3/dkt18/5ip/BOXBEAM | rd_e | ERROR | GRNOD/SURF; LINE/EDGE (+4) |
| RD-E-1702_Transitions/Transition_mesh/mesh3/dkt18/GLOBAL_PLAS/BOXBEAM | rd_e | ERROR | GRNOD/SURF; LINE/EDGE (+4) |
| RD-E-1702_Transitions/Transition_mesh/mesh3/dkt18/ITER_PLAS/BOXBEAM | rd_e | ERROR | GRNOD/SURF; LINE/EDGE (+4) |
| RD-E-1702_Transitions/Transition_mesh/mesh3/qeph/5ip/BOXBEAM | rd_e | ERROR | GRNOD/SURF; LINE/EDGE (+4) |
| RD-E-1702_Transitions/Transition_mesh/mesh3/qeph/GLOBAL_PLAS/BOXBEAM | rd_e | ERROR | GRNOD/SURF; LINE/EDGE (+4) |
| RD-E-1702_Transitions/Transition_mesh/mesh3/qeph/ITER_PLAS/BOXBEAM | rd_e | ERROR | GRNOD/SURF; LINE/EDGE (+4) |
| RD-E-1703_Distorted/Distorted_mesh/mesh0/quad/batoz/5ip/BOXBEAM | rd_e | ERROR | GRNOD/GRNOD; GRNOD/SURF (+5) |
| RD-E-1703_Distorted/Distorted_mesh/mesh0/quad/batoz/GLOBAL_PLAS/BOXBEAM | rd_e | ERROR | GRNOD/GRNOD; GRNOD/SURF (+5) |
| RD-E-1703_Distorted/Distorted_mesh/mesh0/quad/batoz/ITER_PLAS/BOXBEAM | rd_e | ERROR | GRNOD/GRNOD; GRNOD/SURF (+5) |
| RD-E-1703_Distorted/Distorted_mesh/mesh0/quad/bt1/5ip/BOXBEAM | rd_e | ERROR | GRNOD/GRNOD; GRNOD/SURF (+5) |
| RD-E-1703_Distorted/Distorted_mesh/mesh0/quad/bt1/GLOBAL_PLAS/BOXBEAM | rd_e | ERROR | GRNOD/GRNOD; GRNOD/SURF (+5) |
| RD-E-1703_Distorted/Distorted_mesh/mesh0/quad/bt1/ITER_PLAS/BOXBEAM | rd_e | ERROR | GRNOD/GRNOD; GRNOD/SURF (+5) |
| RD-E-1703_Distorted/Distorted_mesh/mesh0/quad/bt3/5ip/BOXBEAM | rd_e | ERROR | GRNOD/GRNOD; GRNOD/SURF (+5) |
| RD-E-1703_Distorted/Distorted_mesh/mesh0/quad/bt3/GLOBAL_PLAS/BOXBEAM | rd_e | ERROR | GRNOD/GRNOD; GRNOD/SURF (+5) |
| RD-E-1703_Distorted/Distorted_mesh/mesh0/quad/bt3/ITER_PLAS/BOXBEAM | rd_e | ERROR | GRNOD/GRNOD; GRNOD/SURF (+5) |
| RD-E-1703_Distorted/Distorted_mesh/mesh0/quad/bt4/5ip/BOXBEAM | rd_e | ERROR | GRNOD/GRNOD; GRNOD/SURF (+5) |
| RD-E-1703_Distorted/Distorted_mesh/mesh0/quad/bt4/GLOBAL_PLAS/BOXBEAM | rd_e | ERROR | GRNOD/GRNOD; GRNOD/SURF (+5) |
| RD-E-1703_Distorted/Distorted_mesh/mesh0/quad/bt4/ITER_PLAS/BOXBEAM | rd_e | ERROR | GRNOD/GRNOD; GRNOD/SURF (+5) |
| RD-E-1703_Distorted/Distorted_mesh/mesh0/quad/qeph/5ip/BOXBEAM | rd_e | ERROR | GRNOD/GRNOD; GRNOD/SURF (+5) |
| RD-E-1703_Distorted/Distorted_mesh/mesh0/quad/qeph/GLOBAL_PLAS/BOXBEAM | rd_e | ERROR | GRNOD/GRNOD; GRNOD/SURF (+5) |
| RD-E-1703_Distorted/Distorted_mesh/mesh0/quad/qeph/ITER_PLAS/BOXBEAM | rd_e | ERROR | GRNOD/GRNOD; GRNOD/SURF (+5) |
| RD-E-1703_Distorted/Distorted_mesh/mesh0/tri/c0/5ip/BOXBEAM | rd_e | ERROR | GRNOD/SURF; LINE/EDGE (+4) |
| RD-E-1703_Distorted/Distorted_mesh/mesh0/tri/c0/GLOBAL_PLAS/BOXBEAM | rd_e | ERROR | GRNOD/SURF; LINE/EDGE (+4) |
| RD-E-1703_Distorted/Distorted_mesh/mesh0/tri/c0/ITER_PLAS/BOXBEAM | rd_e | ERROR | GRNOD/SURF; LINE/EDGE (+4) |
| RD-E-1703_Distorted/Distorted_mesh/mesh0/tri/dkt18/5ip/BOXBEAM | rd_e | ERROR | GRNOD/SURF; LINE/EDGE (+4) |
| RD-E-1703_Distorted/Distorted_mesh/mesh0/tri/dkt18/GLOBAL_PLAS/BOXBEAM | rd_e | ERROR | GRNOD/SURF; LINE/EDGE (+4) |
| RD-E-1703_Distorted/Distorted_mesh/mesh0/tri/dkt18/ITER_PLAS/BOXBEAM | rd_e | ERROR | GRNOD/SURF; LINE/EDGE (+4) |
| RD-E-1703_Distorted/Distorted_mesh/mesh1/quad/batoz/5ip/BOXBEAM | rd_e | ERROR | GRNOD/GRNOD; GRNOD/SURF (+5) |
| RD-E-1703_Distorted/Distorted_mesh/mesh1/quad/batoz/GLOBAL_PLAS/BOXBEAM | rd_e | ERROR | GRNOD/GRNOD; GRNOD/SURF (+5) |
| RD-E-1703_Distorted/Distorted_mesh/mesh1/quad/batoz/ITER_PLAS/BOXBEAM | rd_e | ERROR | GRNOD/GRNOD; GRNOD/SURF (+5) |
| RD-E-1703_Distorted/Distorted_mesh/mesh1/quad/bt1/5ip/BOXBEAM | rd_e | ERROR | GRNOD/GRNOD; GRNOD/SURF (+5) |
| RD-E-1703_Distorted/Distorted_mesh/mesh1/quad/bt1/GLOBAL_PLAS/BOXBEAM | rd_e | ERROR | GRNOD/GRNOD; GRNOD/SURF (+5) |
| RD-E-1703_Distorted/Distorted_mesh/mesh1/quad/bt1/ITER_PLAS/BOXBEAM | rd_e | ERROR | GRNOD/GRNOD; GRNOD/SURF (+5) |
| RD-E-1703_Distorted/Distorted_mesh/mesh1/quad/bt3/5ip/BOXBEAM | rd_e | ERROR | GRNOD/GRNOD; GRNOD/SURF (+5) |
| RD-E-1703_Distorted/Distorted_mesh/mesh1/quad/bt3/GLOBAL_PLAS/BOXBEAM | rd_e | ERROR | GRNOD/GRNOD; GRNOD/SURF (+5) |
| RD-E-1703_Distorted/Distorted_mesh/mesh1/quad/bt3/ITER_PLAS/BOXBEAM | rd_e | ERROR | GRNOD/GRNOD; GRNOD/SURF (+5) |
| RD-E-1703_Distorted/Distorted_mesh/mesh1/quad/bt4/5ip/BOXBEAM | rd_e | ERROR | GRNOD/GRNOD; GRNOD/SURF (+5) |
| RD-E-1703_Distorted/Distorted_mesh/mesh1/quad/bt4/GLOBAL_PLAS/BOXBEAM | rd_e | ERROR | GRNOD/GRNOD; GRNOD/SURF (+5) |
| RD-E-1703_Distorted/Distorted_mesh/mesh1/quad/bt4/ITER_PLAS/BOXBEAM | rd_e | ERROR | GRNOD/GRNOD; GRNOD/SURF (+5) |
| RD-E-1703_Distorted/Distorted_mesh/mesh1/quad/qeph/5ip/BOXBEAM | rd_e | ERROR | GRNOD/GRNOD; GRNOD/SURF (+5) |
| RD-E-1703_Distorted/Distorted_mesh/mesh1/quad/qeph/GLOBAL_PLAS/BOXBEAM | rd_e | ERROR | GRNOD/GRNOD; GRNOD/SURF (+5) |
| RD-E-1703_Distorted/Distorted_mesh/mesh1/quad/qeph/ITER_PLAS/BOXBEAM | rd_e | ERROR | GRNOD/GRNOD; GRNOD/SURF (+5) |
| RD-E-1703_Distorted/Distorted_mesh/mesh1/tri/c0/5ip/BOXBEAM | rd_e | ERROR | GRNOD/GRNOD; GRNOD/SURF (+5) |
| RD-E-1703_Distorted/Distorted_mesh/mesh1/tri/c0/GLOBAL_PLAS/BOXBEAM | rd_e | ERROR | GRNOD/GRNOD; GRNOD/SURF (+5) |
| RD-E-1703_Distorted/Distorted_mesh/mesh1/tri/c0/ITER_PLAS/BOXBEAM | rd_e | ERROR | GRNOD/GRNOD; GRNOD/SURF (+5) |
| RD-E-1703_Distorted/Distorted_mesh/mesh1/tri/dkt18/5ip/BOXBEAM | rd_e | ERROR | GRNOD/GRNOD; GRNOD/SURF (+5) |
| RD-E-1703_Distorted/Distorted_mesh/mesh1/tri/dkt18/GLOBAL_PLAS/BOXBEAM | rd_e | ERROR | GRNOD/GRNOD; GRNOD/SURF (+5) |
| RD-E-1703_Distorted/Distorted_mesh/mesh1/tri/dkt18/ITER_PLAS/BOXBEAM | rd_e | ERROR | GRNOD/GRNOD; GRNOD/SURF (+5) |
| RD-E-1703_Distorted/Distorted_mesh/mesh2/quad/batoz/5ip/BOXBEAM | rd_e | ERROR | GRNOD/GRNOD; GRNOD/SURF (+5) |
| RD-E-1703_Distorted/Distorted_mesh/mesh2/quad/batoz/GLOBAL_PLAS/BOXBEAM | rd_e | ERROR | GRNOD/GRNOD; GRNOD/SURF (+5) |
| RD-E-1703_Distorted/Distorted_mesh/mesh2/quad/batoz/ITER_PLAS/BOXBEAM | rd_e | ERROR | GRNOD/GRNOD; GRNOD/SURF (+5) |
| RD-E-1703_Distorted/Distorted_mesh/mesh2/quad/bt1/5ip/BOXBEAM | rd_e | ERROR | GRNOD/GRNOD; GRNOD/SURF (+5) |
| RD-E-1703_Distorted/Distorted_mesh/mesh2/quad/bt1/GLOBAL_PLAS/BOXBEAM | rd_e | ERROR | GRNOD/GRNOD; GRNOD/SURF (+5) |
| RD-E-1703_Distorted/Distorted_mesh/mesh2/quad/bt1/ITER_PLAS/BOXBEAM | rd_e | ERROR | GRNOD/GRNOD; GRNOD/SURF (+5) |
| RD-E-1703_Distorted/Distorted_mesh/mesh2/quad/bt3/5ip/BOXBEAM | rd_e | ERROR | GRNOD/GRNOD; GRNOD/SURF (+5) |
| RD-E-1703_Distorted/Distorted_mesh/mesh2/quad/bt3/GLOBAL_PLAS/BOXBEAM | rd_e | ERROR | GRNOD/GRNOD; GRNOD/SURF (+5) |
| RD-E-1703_Distorted/Distorted_mesh/mesh2/quad/bt3/ITER_PLAS/BOXBEAM | rd_e | ERROR | GRNOD/GRNOD; GRNOD/SURF (+5) |
| RD-E-1703_Distorted/Distorted_mesh/mesh2/quad/bt4/5ip/BOXBEAM | rd_e | ERROR | GRNOD/GRNOD; GRNOD/SURF (+5) |
| RD-E-1703_Distorted/Distorted_mesh/mesh2/quad/bt4/GLOBAL_PLAS/BOXBEAM | rd_e | ERROR | GRNOD/GRNOD; GRNOD/SURF (+5) |
| RD-E-1703_Distorted/Distorted_mesh/mesh2/quad/bt4/ITER_PLAS/BOXBEAM | rd_e | ERROR | GRNOD/GRNOD; GRNOD/SURF (+5) |
| RD-E-1703_Distorted/Distorted_mesh/mesh2/quad/qeph/5ip/BOXBEAM | rd_e | ERROR | GRNOD/GRNOD; GRNOD/SURF (+5) |
| RD-E-1703_Distorted/Distorted_mesh/mesh2/quad/qeph/GLOBAL_PLAS/BOXBEAM | rd_e | ERROR | GRNOD/GRNOD; GRNOD/SURF (+5) |
| RD-E-1703_Distorted/Distorted_mesh/mesh2/quad/qeph/ITER_PLAS/BOXBEAM | rd_e | ERROR | GRNOD/GRNOD; GRNOD/SURF (+5) |
| RD-E-1703_Distorted/Distorted_mesh/mesh2/tri/c0/5ip/BOXBEAM | rd_e | ERROR | GRNOD/GRNOD; GRNOD/SURF (+5) |
| RD-E-1703_Distorted/Distorted_mesh/mesh2/tri/c0/GLOBAL_PLAS/BOXBEAM | rd_e | ERROR | GRNOD/GRNOD; GRNOD/SURF (+5) |
| RD-E-1703_Distorted/Distorted_mesh/mesh2/tri/c0/ITER_PLAS/BOXBEAM | rd_e | ERROR | GRNOD/GRNOD; GRNOD/SURF (+5) |
| RD-E-1703_Distorted/Distorted_mesh/mesh2/tri/dkt18/5ip/BOXBEAM | rd_e | ERROR | GRNOD/GRNOD; GRNOD/SURF (+5) |
| RD-E-1703_Distorted/Distorted_mesh/mesh2/tri/dkt18/GLOBAL_PLAS/BOXBEAM | rd_e | ERROR | GRNOD/GRNOD; GRNOD/SURF (+5) |
| RD-E-1703_Distorted/Distorted_mesh/mesh2/tri/dkt18/ITER_PLAS/BOXBEAM | rd_e | ERROR | GRNOD/GRNOD; GRNOD/SURF (+5) |
| RD-E-1703_Distorted/Distorted_mesh/mesh3/quad/batoz/5ip/BOXBEAM | rd_e | ERROR | GRNOD/GRNOD; GRNOD/SURF (+5) |
| RD-E-1703_Distorted/Distorted_mesh/mesh3/quad/batoz/GLOBAL_PLAS/BOXBEAM | rd_e | ERROR | GRNOD/GRNOD; GRNOD/SURF (+5) |
| RD-E-1703_Distorted/Distorted_mesh/mesh3/quad/batoz/ITER_PLAS/BOXBEAM | rd_e | ERROR | GRNOD/GRNOD; GRNOD/SURF (+5) |
| RD-E-1703_Distorted/Distorted_mesh/mesh3/quad/bt1/5ip/BOXBEAM | rd_e | ERROR | GRNOD/GRNOD; GRNOD/SURF (+5) |
| RD-E-1703_Distorted/Distorted_mesh/mesh3/quad/bt1/GLOBAL_PLAS/BOXBEAM | rd_e | ERROR | GRNOD/GRNOD; GRNOD/SURF (+5) |
| RD-E-1703_Distorted/Distorted_mesh/mesh3/quad/bt1/ITER_PLAS/BOXBEAM | rd_e | ERROR | GRNOD/GRNOD; GRNOD/SURF (+5) |
| RD-E-1703_Distorted/Distorted_mesh/mesh3/quad/bt3/5ip/BOXBEAM | rd_e | ERROR | GRNOD/GRNOD; GRNOD/SURF (+5) |
| RD-E-1703_Distorted/Distorted_mesh/mesh3/quad/bt3/GLOBAL_PLAS/BOXBEAM | rd_e | ERROR | GRNOD/GRNOD; GRNOD/SURF (+5) |
| RD-E-1703_Distorted/Distorted_mesh/mesh3/quad/bt3/ITER_PLAS/BOXBEAM | rd_e | ERROR | GRNOD/GRNOD; GRNOD/SURF (+5) |
| RD-E-1703_Distorted/Distorted_mesh/mesh3/quad/bt4/5ip/BOXBEAM | rd_e | ERROR | GRNOD/GRNOD; GRNOD/SURF (+5) |
| RD-E-1703_Distorted/Distorted_mesh/mesh3/quad/bt4/GLOBAL_PLAS/BOXBEAM | rd_e | ERROR | GRNOD/GRNOD; GRNOD/SURF (+5) |
| RD-E-1703_Distorted/Distorted_mesh/mesh3/quad/bt4/ITER_PLAS/BOXBEAM | rd_e | ERROR | GRNOD/GRNOD; GRNOD/SURF (+5) |
| RD-E-1703_Distorted/Distorted_mesh/mesh3/quad/qeph/5ip/BOXBEAM | rd_e | ERROR | GRNOD/GRNOD; GRNOD/SURF (+5) |
| RD-E-1703_Distorted/Distorted_mesh/mesh3/quad/qeph/GLOBAL_PLAS/BOXBEAM | rd_e | ERROR | GRNOD/GRNOD; GRNOD/SURF (+5) |
| RD-E-1703_Distorted/Distorted_mesh/mesh3/quad/qeph/ITER_PLAS/BOXBEAM | rd_e | ERROR | GRNOD/GRNOD; GRNOD/SURF (+5) |
| RD-E-1703_Distorted/Distorted_mesh/mesh3/tri/c0/5ip/BOXBEAM | rd_e | ERROR | GRNOD/GRNOD; GRNOD/SURF (+5) |
| RD-E-1703_Distorted/Distorted_mesh/mesh3/tri/c0/GLOBAL_PLAS/BOXBEAM | rd_e | ERROR | GRNOD/GRNOD; GRNOD/SURF (+5) |
| RD-E-1703_Distorted/Distorted_mesh/mesh3/tri/c0/ITER_PLAS/BOXBEAM | rd_e | ERROR | GRNOD/GRNOD; GRNOD/SURF (+5) |
| RD-E-1703_Distorted/Distorted_mesh/mesh3/tri/dkt18/5ip/BOXBEAM | rd_e | ERROR | GRNOD/GRNOD; GRNOD/SURF (+5) |
| RD-E-1703_Distorted/Distorted_mesh/mesh3/tri/dkt18/GLOBAL_PLAS/BOXBEAM | rd_e | ERROR | GRNOD/GRNOD; GRNOD/SURF (+5) |
| RD-E-1703_Distorted/Distorted_mesh/mesh3/tri/dkt18/ITER_PLAS/BOXBEAM | rd_e | ERROR | GRNOD/GRNOD; GRNOD/SURF (+5) |
| RD-E-1801_Square_plate_torsion/Torsion/2q4-4t3/batoz/TORSION | rd_e | ERROR | GRNOD/GRNOD; PARSE /TH/NODE/10: int( '0masternode2_group_0' |
| RD-E-1801_Square_plate_torsion/Torsion/2q4-4t3/bt_type1/TORSION | rd_e | ERROR | GRNOD/GRNOD; PARSE /TH/NODE/10: int( '0masternode2_group_0' |
| RD-E-1801_Square_plate_torsion/Torsion/2q4-4t3/bt_type4/TORSION | rd_e | ERROR | GRNOD/GRNOD; PARSE /TH/NODE/10: int( '0masternode2_group_0' |
| RD-E-1801_Square_plate_torsion/Torsion/2q4-4t3/qeph/TORSION | rd_e | ERROR | GRNOD/GRNOD; PARSE /TH/NODE/10: int( '0masternode2_group_0' |
| RD-E-1801_Square_plate_torsion/Torsion/4q4/batoz/TORSION | rd_e | ERROR | GRNOD/GRNOD; PARSE /TH/NODE/4: int( 'DY' (+1) |
| RD-E-1801_Square_plate_torsion/Torsion/4q4/bt_type1/TORSION | rd_e | ERROR | GRNOD/GRNOD; PARSE /TH/NODE/4: int( 'DY' (+1) |
| RD-E-1801_Square_plate_torsion/Torsion/4q4/bt_type4/TORSION | rd_e | ERROR | GRNOD/GRNOD; PARSE /TH/NODE/4: int( 'DY' (+1) |
| RD-E-1801_Square_plate_torsion/Torsion/4q4/qeph/TORSION | rd_e | ERROR | GRNOD/GRNOD; PARSE /TH/NODE/4: int( 'DY' (+1) |
| RD-E-1801_Square_plate_torsion/Torsion/8t3/c0/TORSION | rd_e | ERROR | GRNOD/GRNOD; PARSE /TH/NODE/1: int( '0masternode2_group_0' |
| RD-E-1801_Square_plate_torsion/Torsion/8t3/dkt/TORSION | rd_e | ERROR | GRNOD/GRNOD; PARSE /TH/NODE/1: int( '0masternode2_group_0' |
| RD-E-1801_Square_plate_torsion/Torsion/8t3_inv/c0/TORSION | rd_e | ERROR | GRNOD/GRNOD; PARSE /TH/NODE/2: int( '0masternode2_group_0' |
| RD-E-1801_Square_plate_torsion/Torsion/8t3_inv/dkt/TORSION | rd_e | ERROR | GRNOD/GRNOD; PARSE /TH/NODE/2: int( '0masternode2_group_0' |
| RD-E-1802_Elastic/Membrane_elastic/2q4-4t3/batoz/TRACTION | rd_e | ERROR | GRNOD/GRNOD; PARSE /TH/NODE/3: int( '0masternode1_group_0' (+1) |
| RD-E-1802_Elastic/Membrane_elastic/2q4-4t3/bt1/TRACTION | rd_e | ERROR | GRNOD/GRNOD; PARSE /TH/NODE/3: int( '0masternode1_group_0' (+1) |
| RD-E-1802_Elastic/Membrane_elastic/2q4-4t3/bt3/TRACTION | rd_e | ERROR | GRNOD/GRNOD; PARSE /TH/NODE/3: int( '0masternode1_group_0' (+1) |
| RD-E-1802_Elastic/Membrane_elastic/2q4-4t3/qeph/TRACTION | rd_e | ERROR | GRNOD/GRNOD; PARSE /TH/NODE/3: int( '0masternode1_group_0' (+1) |
| RD-E-1802_Elastic/Membrane_elastic/4q4/batoz/TRACTION | rd_e | ERROR | GRNOD/GRNOD; PARSE /TH/NODE/4: int( 'DY' (+1) |
| RD-E-1802_Elastic/Membrane_elastic/4q4/bt1/TRACTION | rd_e | ERROR | GRNOD/GRNOD; PARSE /TH/NODE/4: int( 'DY' (+1) |
| RD-E-1802_Elastic/Membrane_elastic/4q4/bt3/TRACTION | rd_e | ERROR | GRNOD/GRNOD; PARSE /TH/NODE/4: int( 'DY' (+1) |
| RD-E-1802_Elastic/Membrane_elastic/4q4/qeph/TRACTION | rd_e | ERROR | GRNOD/GRNOD; PARSE /TH/NODE/4: int( 'DY' (+1) |
| RD-E-1802_Elastic/Membrane_elastic/8t3/c0/TRACTION | rd_e | ERROR | GRNOD/GRNOD; PARSE /TH/NODE/1: int( '0masternode1_group_0' (+1) |
| RD-E-1802_Elastic/Membrane_elastic/8t3/dkt/TRACTION | rd_e | ERROR | GRNOD/GRNOD; PARSE /TH/NODE/1: int( '0masternode1_group_0' (+1) |
| RD-E-1802_Elastic/Membrane_elastic/8t3_inv/c0/TRACTION | rd_e | ERROR | GRNOD/GRNOD; PARSE /TH/NODE/1: int( '0masternode1_group_0' (+1) |
| RD-E-1802_Elastic/Membrane_elastic/8t3_inv/dkt/TRACTION | rd_e | ERROR | GRNOD/GRNOD; PARSE /TH/NODE/1: int( '0masternode1_group_0' (+1) |
| RD-E-1803_Elasto_plastic/Membrane_elasto-plastic/2q4-4t3/batoz/TRACTION | rd_e | ERROR | GRNOD/GRNOD; PARSE /TH/NODE/3: int( '0masternode1_group_0' (+1) |
| RD-E-1803_Elasto_plastic/Membrane_elasto-plastic/2q4-4t3/bt_type1/TRACTION | rd_e | ERROR | GRNOD/GRNOD; PARSE /TH/NODE/3: int( '0masternode1_group_0' (+1) |
| RD-E-1803_Elasto_plastic/Membrane_elasto-plastic/2q4-4t3/bt_type3/TRACTION | rd_e | ERROR | GRNOD/GRNOD; PARSE /TH/NODE/3: int( '0masternode1_group_0' (+1) |
| RD-E-1803_Elasto_plastic/Membrane_elasto-plastic/2q4-4t3/qeph/TRACTION | rd_e | ERROR | GRNOD/GRNOD; PARSE /TH/NODE/3: int( '0masternode1_group_0' (+1) |
| RD-E-1803_Elasto_plastic/Membrane_elasto-plastic/4q4/batoz/TRACTION | rd_e | ERROR | GRNOD/GRNOD; PARSE /TH/NODE/4: int( 'DY' (+1) |
| RD-E-1803_Elasto_plastic/Membrane_elasto-plastic/4q4/bt_type1/TRACTION | rd_e | ERROR | GRNOD/GRNOD; PARSE /TH/NODE/4: int( 'DY' (+1) |
| RD-E-1803_Elasto_plastic/Membrane_elasto-plastic/4q4/bt_type3/TRACTION | rd_e | ERROR | GRNOD/GRNOD; PARSE /TH/NODE/4: int( 'DY' (+1) |
| RD-E-1803_Elasto_plastic/Membrane_elasto-plastic/4q4/qeph/TRACTION | rd_e | ERROR | GRNOD/GRNOD; PARSE /TH/NODE/4: int( 'DY' (+1) |
| RD-E-1803_Elasto_plastic/Membrane_elasto-plastic/8t3/co/TRACTION | rd_e | ERROR | GRNOD/GRNOD; PARSE /TH/NODE/8: int( '0masternode2_group_0' |
| RD-E-1803_Elasto_plastic/Membrane_elasto-plastic/8t3/dkt/TRACTION | rd_e | ERROR | GRNOD/GRNOD; PARSE /TH/NODE/8: int( '0masternode2_group_0' |
| RD-E-1803_Elasto_plastic/Membrane_elasto-plastic/t3_inv/c0/TRACTION | rd_e | ERROR | GRNOD/GRNOD; PARSE /TH/NODE/1: int( '0masternode1_group_0' (+1) |
| RD-E-1803_Elasto_plastic/Membrane_elasto-plastic/t3_inv/dkt/TRACTION | rd_e | ERROR | GRNOD/GRNOD; PARSE /TH/NODE/1: int( '0masternode1_group_0' (+1) |
| RD-E-1900_Wave/19_Wave_propagation/ALE_formulation/WAVE | rd_e | ERROR | ALE/MAT; GRNOD/GRNOD (+8) |
| RD-E-1900_Wave/19_Wave_propagation/Lagrangian_formulation/WAVE | rd_e | ERROR | GRNOD/GRNOD; QUAD (+5) |
| RD-E-2000_Ice_cube/20_Cube/TYPE24/cube_TYPE24 | rd_e | ERROR | INTER/TYPE24; PARSE /MAT/PLAS_JOHNS/1: float( '7.80000000000000E-097.8000 (+11) |
| RD-E-2000_Ice_cube/20_Cube/TYPE25/cube_TYPE25 | rd_e | ERROR | INTER/TYPE25; PARSE /MAT/PLAS_JOHNS/1: float( '7.80000000000000E-097.8000 (+11) |
| RD-E-2000_Ice_cube/20_Cube/TYPE7/cube_TYPE7 | rd_e | ERROR | GRNOD/GRNOD; GRNOD/SURF (+1) |
| RD-E-2100_Cam/21_Cam/interface16/coarse_mesh/I16S16CM | rd_e | ERROR | FRAME/FIX; GRBRIC/BRIC (+10) |
| RD-E-2100_Cam/21_Cam/interface16/fine_mesh/I16S16FM | rd_e | ERROR | BRIC20; FRAME/FIX (+11) |
| RD-E-2100_Cam/21_Cam/interface7/friction/I7PFMCAM | rd_e | ERROR | FRAME/FIX; PARSE /INIVEL/AXIS/1: float( 'Y' (+6) |
| RD-E-2100_Cam/21_Cam/interface7/lagrange/slave_cam/I7LMCAM | rd_e | ERROR | FRAME/FIX; INTER/LAGMUL (+7) |
| RD-E-2100_Cam/21_Cam/interface7/lagrange/slave_valve/I7LMVALVE | rd_e | ERROR | FRAME/FIX; INTER/LAGMUL (+7) |
| RD-E-2100_Cam/21_Cam/interface7/penalty/slave_cam/I7PMCAM | rd_e | ERROR | FRAME/FIX; PARSE /INIVEL/AXIS/1: float( 'Y' (+6) |
| RD-E-2100_Cam/21_Cam/interface7/penalty/slave_valve/I7PMVALVE | rd_e | ERROR | FRAME/FIX; PARSE /INIVEL/AXIS/1: float( 'Y' (+6) |
| RD-E-2201_ALE/Ditching_Mono_Domain_ALE/data/ALE_mono | rd_e | ERROR | EOS/STIFF-GAS; GRBRIC/PART (+10) |
| RD-E-2202_SPH/Ditching_Mono_Domain_SPH/SPHEX_mono_110 | rd_e | ERROR | EOS/STIFF-GAS; MAT/HYD_VISC (+7) |
| RD-E-2203_Multi_Domain/Ditching_Multi_Domain_ALE/ALE_multi | rd_e | ERROR | ALE/MAT; ALE/MUSCL (+12) |
| RD-E-2203_Multi_Domain/Ditching_Multi_Domain_SPH/SPHEX_multi_110 | rd_e | ERROR | EOS/STIFF-GAS; GRNOD/GRNOD (+10) |
| RD-E-2300_Brake/23_Brake/BRAKE | rd_e | ERROR | FRAME/FIX; GRNOD/GRNOD (+6) |
| RD-E-2400_Laminating/24_Laminating/Formulation/Isolid14_Icpre2/ROLLING | rd_e | ERROR | GRNOD/GRNOD; SURF/SURF (+6) |
| RD-E-2400_Laminating/24_Laminating/Formulation/Isolid14_Icpre3/ROLLING | rd_e | ERROR | GRNOD/GRNOD; SURF/SURF (+6) |
| RD-E-2400_Laminating/24_Laminating/Formulation/Isolid17_Icpre2/ROLLING | rd_e | ERROR | GRNOD/GRNOD; SURF/SURF (+6) |
| RD-E-2400_Laminating/24_Laminating/Formulation/Isolid17_Icpre3/ROLLING | rd_e | ERROR | GRNOD/GRNOD; SURF/SURF (+6) |
| RD-E-2400_Laminating/24_Laminating/Thickness/2_elements/ROLLING | rd_e | ERROR | GRNOD/GRNOD; SURF/SURF (+6) |
| RD-E-2400_Laminating/24_Laminating/Thickness/5_elements/ROLLING | rd_e | ERROR | GRNOD/GRNOD; SURF/SURF (+6) |
| RD-E-2400_Laminating/24_Laminating/small_strain_formulation/Ismstr1_small/ROLLING | rd_e | ERROR | GRNOD/GRNOD; SURF/SURF (+6) |
| RD-E-2400_Laminating/24_Laminating/small_strain_formulation/Ismstr2_mix/ROLLING | rd_e | ERROR | GRNOD/GRNOD; SURF/SURF (+6) |
| RD-E-2400_Laminating/24_Laminating/temperature/T=1200/ROLLING | rd_e | ERROR | GRNOD/GRNOD; HEAT/MAT (+7) |
| RD-E-2400_Laminating/24_Laminating/temperature/T=800/ROLLING | rd_e | ERROR | GRNOD/GRNOD; HEAT/MAT (+7) |
| RD-E-2500_Spring_back/25_Spring-back/Explicit_spring-back/DBEND_44 | rd_e | ERROR | GRNOD/GRNOD; MAT/HILL_TAB (+4) |
| RD-E-2500_Spring_back/25_Spring-back/Implicit_spring-back/DBEND_44 | rd_e | ERROR | GRNOD/GRNOD; MAT/HILL_TAB (+4) |
| RD-E-2601_Failure_strain/failure_strain/one_shell/biaxial_test/LAW2/main_TEST4 | rd_e | SKIPS(2) | UNIT |
| RD-E-2601_Failure_strain/failure_strain/one_shell/biaxial_test/LAW27/main_TEST4 | rd_e | SKIPS(2) | UNIT |
| RD-E-2601_Failure_strain/failure_strain/one_shell/uniaxial/LAW2/main_TEST4 | rd_e | SKIPS(2) | UNIT |
| RD-E-2601_Failure_strain/failure_strain/one_shell/uniaxial/LAW27/main_TEST4 | rd_e | SKIPS(2) | UNIT |
| RD-E-2601_Failure_strain/failure_strain/plate_model/LAW2/LAW2 | rd_e | ERROR | GRNOD/GRNOD; ERR /RWALL/1: node group 4067 not defined |
| RD-E-2601_Failure_strain/failure_strain/plate_model/LAW27/LAW27 | rd_e | ERROR | GRNOD/GRNOD; ERR /RWALL/1: node group 4067 not defined |
| RD-E-2602_Ductile/ductile_failure_model/one_shell/Johnson/Ishell_1_wo_epsmax/main_TEST4 | rd_e | SKIPS(2) | UNIT |
| RD-E-2602_Ductile/ductile_failure_model/one_shell/biquad/Ishell_1_wo_epsmax/main_TEST4 | rd_e | ERROR | UNIT; PARSE /FAIL/BIQUAD/1/1: int( '.2' |
| RD-E-2602_Ductile/ductile_failure_model/one_shell/tab1/Ishell_1_wo_epsmax/main_TEST4 | rd_e | SKIPS(4) | FAIL/TAB1; TABLE (+1) |
| RD-E-2602_Ductile/ductile_failure_model/plate_model/BIQUAD_model/Ishell=1_without_epsmax/FAILURE_BIQUAD | rd_e | ERROR | GRNOD/GRNOD; UNIT (+2) |
| RD-E-2602_Ductile/ductile_failure_model/plate_model/JOHNSON_Ifail_sh_model/Ishell=1_and_espmax/FAILURE_JOHNSON | rd_e | ERROR | GRNOD/GRNOD; ERR /RWALL/1: node group 4067 not defined |
| RD-E-2602_Ductile/ductile_failure_model/plate_model/JOHNSON_Ifail_sh_model/Ishell=1_without_epsmax/FAILURE_JOHNSON | rd_e | ERROR | GRNOD/GRNOD; ERR /RWALL/1: node group 4067 not defined |
| RD-E-2602_Ductile/ductile_failure_model/plate_model/JOHNSON_Ifail_sh_model/Ishell=2_and_epsmax/FAILURE_JOHNSON | rd_e | ERROR | GRNOD/GRNOD; ERR /RWALL/1: node group 4067 not defined |
| RD-E-2602_Ductile/ductile_failure_model/plate_model/JOHNSON_Ifail_sh_model/Ishell=2_without_epsmax/FAILURE_JOHNSON | rd_e | ERROR | GRNOD/GRNOD; ERR /RWALL/1: node group 4067 not defined |
| RD-E-2602_Ductile/ductile_failure_model/plate_model/JOHNSON_model/Ishell=1_without_epsmax/FAILURE_JOHNSON | rd_e | ERROR | GRNOD/GRNOD; ERR /RWALL/1: node group 4067 not defined |
| RD-E-2602_Ductile/ductile_failure_model/plate_model/TAB1_model/Ishell=1_without_epsmax/FAILURE_TAB1 | rd_e | ERROR | FAIL/TAB1; GRNOD/GRNOD (+3) |
| RD-E-2603_Forming/Forming/one_shell/biaxial/main_TEST4 | rd_e | SKIPS(3) | FAIL/FLD; UNIT |
| RD-E-2603_Forming/Forming/one_shell/uniaxial/main_TEST4 | rd_e | SKIPS(3) | FAIL/FLD; UNIT |
| RD-E-2603_Forming/Forming/plate_model/FAILURE_FLD | rd_e | ERROR | FAIL/FLD; GRNOD/GRNOD (+2) |
| RD-E-2700_Football/27_Football_shoot/Bathenay_circular/BAT_CIR | rd_e | ERROR | FRAME/FIX; GRNOD/GRNOD (+12) |
| RD-E-2700_Football/27_Football_shoot/Bathenay_square/BAT_SQR | rd_e | ERROR | FRAME/FIX; GRNOD/GRNOD (+12) |
| RD-E-2700_Football/27_Football_shoot/Santini_circular/SANT_CIR | rd_e | ERROR | FRAME/FIX; GRNOD/GRNOD (+12) |
| RD-E-2700_Football/27_Football_shoot/Santini_square/SANT_SQR | rd_e | ERROR | FRAME/FIX; GRNOD/GRNOD (+12) |
| RD-E-3900_Biomedical_valve/39_Bio_Valve/BIO_VALVE/VALVE | rd_e | ERROR | ALE/BCS; ALE/GRID (+12) |
| RD-E-4200_Rubber_ring/42_Rubber_Ring/rubber_ring | rd_e | ERROR | GRBRIC/BRIC; PARSE /IMPDISP/30: 'ZZ' (+2) |
| RD-E-4300_Perfect_gas/43_perfect_gas_polynomial_eos/01-Pabsolute_Eabsolute/1BRICK_COMPRESSION | rd_e | ERROR | ALE/BCS; ALE/MAT (+4) |
| RD-E-4300_Perfect_gas/43_perfect_gas_polynomial_eos/02-Prelative_Eabsolute/1BRICK_COMPRESSION | rd_e | ERROR | ALE/BCS; ALE/MAT (+4) |
| RD-E-4300_Perfect_gas/43_perfect_gas_polynomial_eos/03-Prelative_Erelative/1BRICK_COMPRESSION | rd_e | ERROR | ALE/BCS; ALE/MAT (+4) |
| RD-E-4300_Perfect_gas/43_perfect_gas_polynomial_eos/04-Pabsolue_Erelative/1BRICK_COMPRESSION | rd_e | ERROR | ALE/BCS; ALE/MAT (+4) |
| RD-E-4400_Blow_molding_AMS/44_blow_moding_ams/E4_66_AMS/EXAMPLE4_66 | rd_e | ERROR | AMS; GRNOD/GENE (+7) |
| RD-E-4400_Blow_molding_AMS/44_blow_moding_ams/E4_66_Noda_CST/EXAMPLE4_66 | rd_e | ERROR | GRNOD/GENE; MAT/LAW66 (+6) |
| RD-E-4400_Blow_molding_AMS/44_blow_moding_ams/E4_66_no_dt_control/EXAMPLE4_66 | rd_e | ERROR | GRNOD/GENE; MAT/LAW66 (+6) |
| RD-E-4500_Multi_Domain/45_multidomain_tied/monodomain/bumper_LL4 | rd_e | ERROR | GRNOD/GRNOD; GRNOD/SURF (+12) |
| RD-E-4500_Multi_Domain/45_multidomain_tied/multidomain/bumper_LL4 | rd_e | ERROR | GRNOD/GRNOD; GRNOD/SURF (+12) |
| RD-E-4601_Lagrange/Lagrange/Lag6elem | rd_e | ERROR | DFS/DETPLAN; EOS/GRUNEISEN (+6) |
| RD-E-4602_Euler/Euler/data/Euler6 | rd_e | ERROR | DFS/DETPLAN; MAT/HYD_JCOOK (+12) |
| RD-E-4701_Kupfer/Kupfer_tests/LAW24/C000/C000 | rd_e | ERROR | MAT/CONC; ERR /PART/1: material 1 not defined |
| RD-E-4701_Kupfer/Kupfer_tests/LAW24/CC00/CC00 | rd_e | ERROR | MAT/CONC; ERR /PART/1: material 1 not defined |
| RD-E-4701_Kupfer/Kupfer_tests/LAW24/CC01/CC01 | rd_e | ERROR | MAT/CONC; PARSE /TH/NODE/2: int( '0new_th_node_1' (+1) |
| RD-E-4701_Kupfer/Kupfer_tests/LAW24/T000/TC00 | rd_e | ERROR | MAT/CONC; ERR /PART/1: material 1 not defined |
| RD-E-4701_Kupfer/Kupfer_tests/LAW24/TC01/TC01 | rd_e | ERROR | MAT/CONC; ERR /PART/1: material 1 not defined |
| RD-E-4701_Kupfer/Kupfer_tests/LAW24/TC02/TC02 | rd_e | ERROR | MAT/CONC; ERR /PART/1: material 1 not defined |
| RD-E-4701_Kupfer/Kupfer_tests/LAW24/TC03/TC03 | rd_e | ERROR | MAT/CONC; ERR /PART/1: material 1 not defined |
| RD-E-4701_Kupfer/Kupfer_tests/LAW81/C000/C000 | rd_e | ERROR | MAT/LAW81; ERR /PART/1: material 1 not defined |
| RD-E-4701_Kupfer/Kupfer_tests/LAW81/CC00/CC00 | rd_e | ERROR | MAT/LAW81; ERR /PART/1: material 1 not defined |
| RD-E-4701_Kupfer/Kupfer_tests/LAW81/CC01/CC01 | rd_e | ERROR | MAT/LAW81; PARSE /TH/NODE/2: int( '0new_th_node_1' (+1) |
| RD-E-4701_Kupfer/Kupfer_tests/LAW81/T000/TC00 | rd_e | ERROR | MAT/LAW81; ERR /PART/1: material 1 not defined |
| RD-E-4701_Kupfer/Kupfer_tests/LAW81/TC01/TC01 | rd_e | ERROR | MAT/LAW81; ERR /PART/1: material 1 not defined |
| RD-E-4701_Kupfer/Kupfer_tests/LAW81/TC02/TC02 | rd_e | ERROR | MAT/LAW81; ERR /PART/1: material 1 not defined |
| RD-E-4701_Kupfer/Kupfer_tests/LAW81/TC03/TC03 | rd_e | ERROR | MAT/LAW81; ERR /PART/1: material 1 not defined |
| RD-E-4702_Brazilian/Splitting_tensile_test/CONCRETE_v3 | rd_e | ERROR | MAT/CONC; ERR /PART/1: material 102 not defined (+2) |
| RD-E-4801_Solid_spotweld/One_solid_element_validation_law83/run01_no_failure/KS2_model_v01 | rd_e | ERROR | GRNOD/GRNOD; GRSHEL/SHEL (+12) |
| RD-E-4801_Solid_spotweld/One_solid_element_validation_law83/run02_w_failure/KS2_model_v01 | rd_e | ERROR | FAIL/SNCONNECT; GRNOD/GRNOD (+12) |
| RD-E-4801_Solid_spotweld/One_solid_element_validation_law83/run02_w_failure_Beta_update/KS2_model_v01 | rd_e | ERROR | FAIL/SNCONNECT; GRNOD/GRNOD (+12) |
| RD-E-4801_Solid_spotweld/One_solid_element_validation_law83/run03_peel_test_1/KS2_model_v01 | rd_e | ERROR | FAIL/SNCONNECT; GRNOD/GRNOD (+12) |
| RD-E-4801_Solid_spotweld/One_solid_element_validation_law83/run03_peel_test_2/KS2_model_v01 | rd_e | ERROR | FAIL/SNCONNECT; GRNOD/GRNOD (+12) |
| RD-E-4802_Solid_spring/solid_spot_law59/FRAME_MODIFIED | rd_e | ERROR | ENDSUB; FAIL/CONNECT (+12) |
| RD-E-4802_Solid_spring/spring_beam_spotweld/FRAME_MODIFIED | rd_e | ERROR | ENDSUB; GRNOD/GRNOD (+10) |
| RD-E-4900_Bird_strike/49_bird_strike/BIRD_WINDSHIELD_v1 | rd_e | ERROR | GRSH3N/SH3N; GRSHEL/SHEL (+12) |
| RD-E-5000_Inivol_FSI/50_inivol_and_fluid_structure/data/fsi_drop_container | rd_e | ERROR | EOS/LINEAR; GRBRIC/PART (+12) |
| RD-E-5101_Size_optimization/Size_Optimization/RAD_OPT/Neon-b_pillar | rd_e | ERROR | GRNOD/GRNOD; GRNOD/SURF (+10) |
| RD-E-5101_Size_optimization/Size_Optimization/base/Neon-b_pillar | rd_e | ERROR | GRNOD/GRNOD; GRNOD/SURF (+10) |
| RD-E-5102_Topology_optimization/Topology_Optimization/Base/hook_opt | rd_e | ERROR | AMS; GRPART/PART (+11) |
| RD-E-5102_Topology_optimization/Topology_Optimization/RADOPT/hook_opt | rd_e | ERROR | AMS; GRPART/PART (+11) |
| RD-E-5200_Creep/52_cylinder_creep/cylinder_creep_beta_001/foam_relax | rd_e | ERROR | GRBRIC/BRIC; GRNOD/SURF (+5) |
| RD-E-5200_Creep/52_cylinder_creep/cylinder_creep_beta_005/foam_relax | rd_e | ERROR | GRBRIC/BRIC; GRNOD/SURF (+5) |
| RD-E-5200_Creep/52_cylinder_creep/cylinder_creep_beta_1/foam_relax | rd_e | ERROR | GRBRIC/BRIC; GRNOD/SURF (+5) |
| RD-E-5200_Creep/52_cylinder_creep/cylinder_relaxation_beta_001/foam_relax | rd_e | ERROR | GRBRIC/BRIC; GRNOD/SURF (+5) |
| RD-E-5200_Creep/52_cylinder_creep/cylinder_relaxation_beta_005/foam_relax | rd_e | ERROR | GRBRIC/BRIC; GRNOD/SURF (+5) |
| RD-E-5200_Creep/52_cylinder_creep/cylinder_relaxation_beta_1/foam_relax | rd_e | ERROR | GRBRIC/BRIC; GRNOD/SURF (+5) |
| RD-E-5300_Thermal/53_thermal_analysis/heat_exchange/moving_source | rd_e | ERROR | CONVEC; HEAT/MAT (+11) |
| RD-E-5400_Cut_methodology/54_cut_model/cut_sub_model/CBOX | rd_e | ERROR | AMS; GRSHEL/PART (+7) |
| RD-E-5400_Cut_methodology/54_cut_model/full_Model/CBOX | rd_e | ERROR | AMS; GRSHEL/PART (+8) |
| RD-E-5501_Fan_blade/1_Initialize_rotation_stress/fan_blade_initialize | rd_e | ERROR | GRNOD/SURF; GRSHEL/SHEL (+4) |
| RD-E-5502_Rotating/2_Rotation_and_1_ice/fan_blade_ice_impact | rd_e | ERROR | GRNOD/SURF; GRSHEL/SHEL (+12) |
| RD-E-5600_Hyperelastic_material/56_HyperElastic_Material/Arruda_Boyce_model/LAW92_UT/rubber_tension_v1 | rd_e | ERROR | MAT/LAW92; UNIT (+2) |
| RD-E-5600_Hyperelastic_material/56_HyperElastic_Material/Arruda_Boyce_model/LAW92_UT_parameter/rubber_tension_v1 | rd_e | ERROR | MAT/LAW92; UNIT (+2) |
| RD-E-5600_Hyperelastic_material/56_HyperElastic_Material/Ogden_model/LAW42_pair2/rubber_tension_v1 | rd_e | ERROR | UNIT; PARSE /TH/NODE/5: int( '0new_th_node_17' |
| RD-E-5600_Hyperelastic_material/56_HyperElastic_Material/Ogden_model/LAW42_pair3/rubber_tension_v1 | rd_e | ERROR | UNIT; PARSE /TH/NODE/5: int( '0new_th_node_17' (+2) |
| RD-E-5600_Hyperelastic_material/56_HyperElastic_Material/Ogden_model/LAW69_extend_to_compression/rubber_tension_v1 | rd_e | ERROR | MAT/LAW69; UNIT (+2) |
| RD-E-5600_Hyperelastic_material/56_HyperElastic_Material/Ogden_model/LAW69_ogden_pair2/LAW69_ogden_pair2_Poisson04997/rubber_tension_v1 | rd_e | ERROR | MAT/LAW69; UNIT (+2) |
| RD-E-5600_Hyperelastic_material/56_HyperElastic_Material/Ogden_model/LAW69_ogden_pair3/rubber_tension_v1 | rd_e | ERROR | MAT/LAW69; UNIT (+2) |
| RD-E-5600_Hyperelastic_material/56_HyperElastic_Material/Ogden_model/LAW82/rubber_tension_v1 | rd_e | ERROR | MAT/LAW82; PARSE /TH/NODE/5: int( '0new_th_node_17' (+1) |
| RD-E-5600_Hyperelastic_material/56_HyperElastic_Material/Ogden_model/LAW88/rubber_tension_v1 | rd_e | ERROR | MAT/LAW88; PARSE /TH/NODE/5: int( '0new_th_node_17' (+1) |
| RD-E-5600_Hyperelastic_material/56_HyperElastic_Material/Yeoh_model/rubber_tension_v1 | rd_e | ERROR | MAT/LAW94; PARSE /TH/NODE/5: int( '0new_th_node_17' (+1) |
| RD-V-0500_Shyue_Shock_Tube/JWL_shock_tube_MUSCL/data/Shock_tube_MUSCL | rd_v_blast | ERROR | DFS/DETPOINT; MAT/JWL (+3) |
| RD-V-0505_Shock_Tube/Eulerian_formulation/data/TACEUL | rd_v_blast | ERROR | MAT/HYD_VISC; MAT/LAW151 (+4) |
| RD-V-0510_Prandtl_Meyer_Fan/0510_Prandtl_Meyer_fan/data/rdv_0510_PRANDTL_MEYER_EXPANSION | rd_v_blast | ERROR | EBCS/FLUXOUT; EBCS/INLET (+12) |
| RD-V-0520_Double_Shock/0520_double_shock/data/DOUBLE_OBLIQUE_QUADS | rd_v_blast | ERROR | EBCS/FLUXOUT; EBCS/INLET (+12) |
| RD-V-0530_Wave_propagation/0530_wave_propagation/data/rdv_0530_wave_propagation | rd_v_blast | ERROR | SKEW/FIX; PARSE /FUNCT/1: float( '5.00000000000000E-0 (+10) |
| RD-V-0100_Impact/0100_impact/type24/rdv_0100_momentum_type24 | rd_v_contact | ERROR | INIVEL/TRA; INTER/TYPE24 (+10) |
| RD-V-0100_Impact/0100_impact/type25/rdv_0100_momentum_type25 | rd_v_contact | ERROR | INIVEL/TRA; INTER/TYPE25 (+10) |
| RD-V-0100_Impact/0100_impact/type7/rdv_0100_momentum_type7 | rd_v_contact | ERROR | GRNOD/SURF; INIVEL/TRA (+10) |
| RD-V-0110_Friction/0110_friction/rdv_0110_friction | rd_v_contact | ERROR | INTER/TYPE24; PARSE /FUNCT/1: float( '1.00000000000000E-06-1.0066 (+2) |
| RD-V-0120_Edge_to_Edge/0120_edge_to_edge/ADYREL/CONTACT_TYPE11 | rd_v_contact | ERROR | ENDSUB; GRNOD/GENE (+10) |
| RD-V-0120_Edge_to_Edge/0120_edge_to_edge/DYREL/CONTACT_TYPE11 | rd_v_contact | ERROR | ENDSUB; GRNOD/GENE (+10) |
| RD-V-0120_Edge_to_Edge/0120_edge_to_edge/No_dynamic_relaxation/CONTACT_TYPE11 | rd_v_contact | ERROR | ENDSUB; GRNOD/GENE (+10) |
| RD-V-0130/0130_tied_interface/Iteration1/TYPE_10/data/Tied_surface_type10 | rd_v_contact | ERROR | GRNOD/SURF; INTER/TYPE10 (+10) |
| RD-V-0130/0130_tied_interface/Iteration1/TYPE_10_1/data/Tied_surface_type10_1 | rd_v_contact | ERROR | GRNOD/SURF; INTER/TYPE10 (+10) |
| RD-V-0130/0130_tied_interface/Iteration1/type7_INI/data/rdv_0100_momentum_type7a | rd_v_contact | ERROR | GRNOD/SURF; ERR /NODE card needs 4 fields, got 3 (+9) |
| RD-V-0130/0130_tied_interface/Iteration2/TYPE10_0/data/4_Modele_10_0 | rd_v_contact | ERROR | GRNOD/SURF; INTER/TYPE10 (+10) |
| RD-V-0130/0130_tied_interface/Iteration2/TYPE10_1/data/4_Modele_10_1 | rd_v_contact | ERROR | GRNOD/SURF; INTER/TYPE10 (+10) |
| RD-V-0130/0130_tied_interface/Iteration2/TYPE7/data/4_Modele_TYPE7 | rd_v_contact | ERROR | GRNOD/SURF; ERR /NODE card needs 4 fields, got 3 (+9) |
| RD-V-0130/0130_tied_interface/Iteration3/TYPE10_0/data/5_Modele2_Gliss_10_0 | rd_v_contact | ERROR | GRNOD/SURF; INTER/TYPE10 (+10) |
| RD-V-0130/0130_tied_interface/Iteration3/TYPE7/data/5_TYPE7 | rd_v_contact | ERROR | GRNOD/SURF; ERR /NODE card needs 4 fields, got 3 (+9) |
| RD-V-0010_Beam_Bending/0010_bending/rdv_0010_beam_bending | rd_v_elements | ERROR | FUNCT_SMOOTH; ERR /PROP/SHELL/2: thickness card missing (+2) |
| RD-V-0020_Cantilever_Beam/0020_beam/explicit/beam_type3/Cantilever_beam | rd_v_elements | ERROR | FUNCT_SMOOTH; PARSE /TH/NODE/1: int( '010mm' (+2) |
| RD-V-0020_Cantilever_Beam/0020_beam/explicit/brick_isolid14/Cantilever_beam | rd_v_elements | ERROR | FUNCT_SMOOTH; PARSE /TH/NODE/1: int( '010mm' (+3) |
| RD-V-0020_Cantilever_Beam/0020_beam/explicit/brick_isolid17/Cantilever_beam | rd_v_elements | ERROR | FUNCT_SMOOTH; PARSE /TH/NODE/1: int( '010mm' (+3) |
| RD-V-0020_Cantilever_Beam/0020_beam/explicit/brick_isolid18/Cantilever_beam | rd_v_elements | ERROR | FUNCT_SMOOTH; PARSE /TH/NODE/1: int( '010mm' (+3) |
| RD-V-0020_Cantilever_Beam/0020_beam/explicit/brick_isolid24/Cantilever_beam | rd_v_elements | ERROR | FUNCT_SMOOTH; PARSE /TH/NODE/1: int( '010mm' (+3) |
| RD-V-0020_Cantilever_Beam/0020_beam/explicit/sh3n_Ish3n0/Cantilever_beam | rd_v_elements | ERROR | FUNCT_SMOOTH; PARSE /TH/NODE/1: int( '010mm' (+2) |
| RD-V-0020_Cantilever_Beam/0020_beam/explicit/sh3n_Ish3n30/Cantilever_beam | rd_v_elements | ERROR | FUNCT_SMOOTH; PARSE /TH/NODE/1: int( '010mm' (+2) |
| RD-V-0020_Cantilever_Beam/0020_beam/explicit/shell_ishell12/Cantilever_beam | rd_v_elements | ERROR | FUNCT_SMOOTH; PARSE /TH/NODE/1: int( '010mm' (+2) |
| RD-V-0020_Cantilever_Beam/0020_beam/explicit/shell_ishell24/Cantilever_beam | rd_v_elements | ERROR | FUNCT_SMOOTH; PARSE /TH/NODE/1: int( '010mm' (+2) |
| RD-V-0020_Cantilever_Beam/0020_beam/explicit/tetra10_0/Cantilever_beam | rd_v_elements | ERROR | FUNCT_SMOOTH; TETRA10 (+5) |
| RD-V-0020_Cantilever_Beam/0020_beam/explicit/tetra10_2/Cantilever_beam | rd_v_elements | ERROR | FUNCT_SMOOTH; TETRA10 (+5) |
| RD-V-0020_Cantilever_Beam/0020_beam/explicit/tetra4_0/Cantilever_beam | rd_v_elements | ERROR | FUNCT_SMOOTH; PARSE /TH/NODE/1: int( '010mm' (+3) |
| RD-V-0020_Cantilever_Beam/0020_beam/explicit/tetra4_1/Cantilever_beam | rd_v_elements | ERROR | FUNCT_SMOOTH; PARSE /TH/NODE/1: int( '010mm' (+3) |
| RD-V-0020_Cantilever_Beam/0020_beam/explicit/tetra4_3/Cantilever_beam | rd_v_elements | ERROR | FUNCT_SMOOTH; PARSE /TH/NODE/1: int( '010mm' (+3) |
| RD-V-0020_Cantilever_Beam/0020_beam/implicit/beam_type3/Cantilever_beam | rd_v_elements | ERROR | FUNCT_SMOOTH; PARSE /TH/NODE/1: int( '010mm' (+2) |
| RD-V-0020_Cantilever_Beam/0020_beam/implicit/brick_isolid14/Cantilever_beam | rd_v_elements | ERROR | FUNCT_SMOOTH; PARSE /TH/NODE/1: int( '010mm' (+3) |
| RD-V-0020_Cantilever_Beam/0020_beam/implicit/brick_isolid17/Cantilever_beam | rd_v_elements | ERROR | FUNCT_SMOOTH; PARSE /TH/NODE/1: int( '010mm' (+3) |
| RD-V-0020_Cantilever_Beam/0020_beam/implicit/brick_isolid18/Cantilever_beam | rd_v_elements | ERROR | FUNCT_SMOOTH; PARSE /TH/NODE/1: int( '010mm' (+3) |
| RD-V-0020_Cantilever_Beam/0020_beam/implicit/brick_isolid24/Cantilever_beam | rd_v_elements | ERROR | FUNCT_SMOOTH; PARSE /TH/NODE/1: int( '010mm' (+3) |
| RD-V-0020_Cantilever_Beam/0020_beam/implicit/sh3n_Ish3n0/Cantilever_beam | rd_v_elements | ERROR | FUNCT_SMOOTH; PARSE /TH/NODE/1: int( '010mm' (+2) |
| RD-V-0020_Cantilever_Beam/0020_beam/implicit/shell_ishell12/Cantilever_beam | rd_v_elements | ERROR | FUNCT_SMOOTH; PARSE /TH/NODE/1: int( '010mm' (+2) |
| RD-V-0020_Cantilever_Beam/0020_beam/implicit/shell_ishell24/Cantilever_beam | rd_v_elements | ERROR | FUNCT_SMOOTH; PARSE /TH/NODE/1: int( '010mm' (+2) |
| RD-V-0020_Cantilever_Beam/0020_beam/implicit/tetra10_0/Cantilever_beam | rd_v_elements | ERROR | FUNCT_SMOOTH; TETRA10 (+5) |
| RD-V-0020_Cantilever_Beam/0020_beam/implicit/tetra4_0/Cantilever_beam | rd_v_elements | ERROR | FUNCT_SMOOTH; PARSE /TH/NODE/1: int( '010mm' (+3) |
| RD-V-0020_Cantilever_Beam/0020_beam/implicit/tetra4_3/Cantilever_beam | rd_v_elements | ERROR | FUNCT_SMOOTH; PARSE /TH/NODE/1: int( '010mm' (+3) |
| RD-V-0030_Spring_Type_4/0030_spring_type_4/0030_spring_type_4_stiff/Spring_TYPE4_element | rd_v_elements | ERROR | ERR /PART/1: material 0 not defined; ERR /PART/2: material 0 not defined (+3) |
| RD-V-0030_Spring_Type_4/0030_spring_type_4/0030_spring_type_4_visc/Spring_TYPE4_element | rd_v_elements | ERROR | ERR /PART/1: material 0 not defined; ERR /PART/2: material 0 not defined (+3) |
| RD-V-0031_Pretensioner_Spring_Type_32/0031_spring_type_32/Spring_TYPE32_element | rd_v_elements | ERROR | PROP/SPR_PRE; ERR /PART/1: material 0 not defined (+9) |
| RD-V-0032_Spring_Type_8/0032_spring_type_8/0032_spring_type_8_stiff_RX/Spring_TYPE8_element | rd_v_elements | ERROR | PROP/SPR_GENE; SKEW/FIX (+11) |
| RD-V-0032_Spring_Type_8/0032_spring_type_8/0032_spring_type_8_stiff_RY/Spring_TYPE8_element | rd_v_elements | ERROR | PROP/SPR_GENE; SKEW/FIX (+11) |
| RD-V-0032_Spring_Type_8/0032_spring_type_8/0032_spring_type_8_stiff_RZ/Spring_TYPE8_element | rd_v_elements | ERROR | PROP/SPR_GENE; SKEW/FIX (+11) |
| RD-V-0032_Spring_Type_8/0032_spring_type_8/0032_spring_type_8_stiff_TX/Spring_TYPE8_element | rd_v_elements | ERROR | PROP/SPR_GENE; SKEW/FIX (+10) |
| RD-V-0032_Spring_Type_8/0032_spring_type_8/0032_spring_type_8_stiff_TY/Spring_TYPE8_element | rd_v_elements | ERROR | PROP/SPR_GENE; SKEW/FIX (+10) |
| RD-V-0032_Spring_Type_8/0032_spring_type_8/0032_spring_type_8_stiff_TZ/Spring_TYPE8_element | rd_v_elements | ERROR | PROP/SPR_GENE; SKEW/FIX (+10) |
| RD-V-0032_Spring_Type_8/0032_spring_type_8/0032_spring_type_8_visc_RX/Spring_TYPE8_element | rd_v_elements | ERROR | PROP/SPR_GENE; SKEW/FIX (+12) |
| RD-V-0032_Spring_Type_8/0032_spring_type_8/0032_spring_type_8_visc_RY/Spring_TYPE8_element | rd_v_elements | ERROR | PROP/SPR_GENE; SKEW/FIX (+12) |
| RD-V-0032_Spring_Type_8/0032_spring_type_8/0032_spring_type_8_visc_RZ/Spring_TYPE8_element | rd_v_elements | ERROR | PROP/SPR_GENE; SKEW/FIX (+12) |
| RD-V-0032_Spring_Type_8/0032_spring_type_8/0032_spring_type_8_visc_TX/Spring_TYPE8_element | rd_v_elements | ERROR | PROP/SPR_GENE; SKEW/FIX (+10) |
| RD-V-0032_Spring_Type_8/0032_spring_type_8/0032_spring_type_8_visc_TY/Spring_TYPE8_element | rd_v_elements | ERROR | PROP/SPR_GENE; SKEW/FIX (+10) |
| RD-V-0032_Spring_Type_8/0032_spring_type_8/0032_spring_type_8_visc_TZ/Spring_TYPE8_element | rd_v_elements | ERROR | PROP/SPR_GENE; SKEW/FIX (+10) |
| RD-V-0033_Spring_Type_13/0033_spring_type_13_stiff_TY/0033_spring_type_13_stiff_TY/Spring_TYPE13_element | rd_v_elements | ERROR | PROP/SPR_BEAM; ERR /PART/1: material 0 not defined (+9) |
| RD-V-0033_Spring_Type_13/0033_spring_type_13_stiff_TZ/0033_spring_type_13_stiff_TZ/Spring_TYPE13_element | rd_v_elements | ERROR | PROP/SPR_BEAM; ERR /PART/1: material 0 not defined (+9) |
| RD-V-0700/Final_version_JC_LAW36_LAW2/JOHNSON_COOK_FAILURE_2024_LAW2_version2/HEXA_DEGE_LAW2/HEXA_DEGE | rd_v_failure | ERROR | PARSE /FUNCT/3: float( '1.00000000000000E-061.6700000000000; ERR /BRICK 1: degenerated brick with 6 distinct nodes (penta (+9) |
| RD-V-0700/Final_version_JC_LAW36_LAW2/JOHNSON_COOK_FAILURE_2024_LAW2_version2/HEXA_ELEM_SOLID_18_LAW2/HEXAP14_18 | rd_v_failure | ERROR | PARSE /FUNCT/3: float( '1.00000000000000E-061.670000000000; ERR /IMPDISP/5: function 3 not defined (+9) |
| RD-V-0700/Final_version_JC_LAW36_LAW2/JOHNSON_COOK_FAILURE_2024_LAW2_version2/HEXA_ELEM_SOLID_24_LAW2/HEXA_LAW2 | rd_v_failure | ERROR | PARSE /FUNCT/3: float( '1.00000000000000E-061.6700000000000; ERR /IMPDISP/5: function 3 not defined (+9) |
| RD-V-0700/Final_version_JC_LAW36_LAW2/JOHNSON_COOK_FAILURE_2024_LAW2_version2/HEXA_ELEM_TSHELL_LAW2/HEXA_TSHELL_LAW2 | rd_v_failure | ERROR | PROP/TSHELL; ERR /PART/1: property 100001 not defined (+9) |
| RD-V-0700/Final_version_JC_LAW36_LAW2/JOHNSON_COOK_FAILURE_2024_LAW2_version2/SHELL_Ishell24_LAW2/shell_Ishell24_LAW2 | rd_v_failure | ERROR | PARSE /SHELL/1: int( '0.0'; PARSE /SHELL/2: int( '0.0' (+12) |
| RD-V-0700/Final_version_JC_LAW36_LAW2/JOHNSON_COOK_FAILURE_2024_LAW2_version2/TETRA_ELEM_24/TETRA_LAW2 | rd_v_failure | ERROR | PARSE /FUNCT/3: float( '1.00000000000000E-061.670000000000; ERR /IMPDISP/5: function 3 not defined (+9) |
| RD-V-0700/Final_version_JC_LAW36_LAW2/JOHNSON_COOK_FAILURE_2024_LAW2_version2/TRIA_LAW2/TRIA_JC_LAW2 | rd_v_failure | ERROR | ERR /TH/NODE/1: unknown node 0; ERR /TH/NODE/1: unknown node 0 (+2) |
| RD-V-0700/Final_version_JC_LAW36_LAW2/JOHNSON_COOK_FAILURE_2024_LAW36_version2/HEXA_DEGE/HEXA_DEGE | rd_v_failure | ERROR | PARSE /MAT/PLAS_TAB/1: int( '1.00000000000000E-03'; PARSE /FUNCT/3: float( '1.00000000000000E-061.6700000000000 (+10) |
| RD-V-0700/Final_version_JC_LAW36_LAW2/JOHNSON_COOK_FAILURE_2024_LAW36_version2/HEXA_ELEM_SOLID_18/HEXAP14_18 | rd_v_failure | ERROR | PARSE /MAT/PLAS_TAB/1: int( '1.00000000000000E-03'; PARSE /FUNCT/3: float( '1.00000000000000E-061.670000000000 (+10) |
| RD-V-0700/Final_version_JC_LAW36_LAW2/JOHNSON_COOK_FAILURE_2024_LAW36_version2/HEXA_ELEM_SOLID_24/HEXA1 | rd_v_failure | ERROR | PARSE /MAT/PLAS_TAB/1: int( '1.00000000000000E-03'; PARSE /FUNCT/3: float( '1.00000000000000E-061.67000000000000 (+10) |
| RD-V-0700/Final_version_JC_LAW36_LAW2/JOHNSON_COOK_FAILURE_2024_LAW36_version2/HEXA_ELEM_TSHELL/HEXA_TSHELL | rd_v_failure | ERROR | PROP/TSHELL; PARSE /MAT/PLAS_TAB/1: int( '1.00000000000000E-03' (+11) |
| RD-V-0700/Final_version_JC_LAW36_LAW2/JOHNSON_COOK_FAILURE_2024_LAW36_version2/SHELL_Ishell12/shell_Ishell12 | rd_v_failure | ERROR | PARSE /SHELL/1: int( '0.0'; PARSE /SHELL/2: int( '0.0' (+12) |
| RD-V-0700/Final_version_JC_LAW36_LAW2/JOHNSON_COOK_FAILURE_2024_LAW36_version2/SHELL_Ishell24/shell_Ishell24 | rd_v_failure | ERROR | PARSE /SHELL/1: int( '0.0'; PARSE /SHELL/2: int( '0.0' (+12) |
| RD-V-0700/Final_version_JC_LAW36_LAW2/JOHNSON_COOK_FAILURE_2024_LAW36_version2/TETRA_ELEM_24/TETRA | rd_v_failure | ERROR | PARSE /MAT/PLAS_TAB/1: int( '1.00000000000000E-03'; PARSE /FUNCT/3: float( '1.00000000000000E-061.67000000000000 (+10) |
| RD-V-0700/Final_version_JC_LAW36_LAW2/JOHNSON_COOK_FAILURE_2024_LAW36_version2/TRIA/TRIA_JC | rd_v_failure | ERROR | ERR /TH/NODE/1: unknown node 0; ERR /TH/NODE/1: unknown node 0 (+2) |
| RD-V-0400_ALE/0400_mok_fsi_benchmark_ALE/Data/Mok_Benchmark | rd_v_fsi | ERROR | ALE/BCS; ALE/GRID (+12) |
| RD-V-0400_CEL/0400_mok_fsi_benchmark_CEL/CEL/data/Mok_Benchmark | rd_v_fsi | ERROR | ALE/BCS; EOS/LINEAR (+12) |
| RD-V-0300_Pressure/0300_pressure/rdv_0300_pressure | rd_v_loads | ERROR | ERR /NODE card needs 4 fields, got 3; ERR /NODE card needs 4 fields, got 3 (+8) |
| RD_V_0310/Model/Gravity/gravity | rd_v_loads | ERROR | GRSH3N/SH3N; GRSHEL/SHEL (+12) |
| RD_V_0310/Model/Impacc/impacc | rd_v_loads | ERROR | GRSH3N/SH3N; GRSHEL/SHEL (+12) |
| RD-V-0200_Hardening/0200_hardening/rdv_0200_hardening_modulus | rd_v_material | SKIPS(1) | - |
| RD-V-0210_Yeoh_Hyperelastic_Material/0210_Yeoh_Hyperelastic_Material_0.495/rubber | rd_v_material | ERROR | FUNCT_SMOOTH; GRBRIC/PART (+10) |
| RD-V-0210_Yeoh_Hyperelastic_Material/0210_Yeoh_Hyperelastic_Material_0.49999/rubber | rd_v_material | ERROR | FUNCT_SMOOTH; GRBRIC/PART (+10) |
| RD-V-0220_Foam_LAW70/0220_foam_LAW70/0220_foam_LAW70_0/BLOCK_H8 | rd_v_material | ERROR | FUNCT_SMOOTH; MAT/LAW70 (+3) |
| RD-V-0220_Foam_LAW70/0220_foam_LAW70/0220_foam_LAW70_1/BLOCK_H8 | rd_v_material | ERROR | FUNCT_SMOOTH; MAT/LAW70 (+3) |
| RD-V-0220_Foam_LAW70/0220_foam_LAW70/0220_foam_LAW70_2/BLOCK_H8 | rd_v_material | ERROR | FUNCT_SMOOTH; MAT/LAW70 (+3) |
| RD-V-0220_Foam_LAW70/0220_foam_LAW70/0220_foam_LAW70_3/BLOCK_H8 | rd_v_material | ERROR | FRAME/MOV; FUNCT_SMOOTH (+5) |
| RD-V-0230_Fabric_LAW19/0230_fabric_LAW19/0230_shell_mat_019_01/SHELL_LAW19_PROP9 | rd_v_material | ERROR | FUNCT_SMOOTH; GRNOD/GENE (+12) |
| RD-V-0230_Fabric_LAW19/0230_fabric_LAW19/0230_shell_mat_019_02/SHELL_LAW19_PROP9 | rd_v_material | ERROR | FUNCT_SMOOTH; GRNOD/GENE (+12) |
| RD_V_0240/Modele_HEXA_P14_Isolid18/tensile_TABULATED_HEXA_TYPE14_Isolid18 | rd_v_material | ERROR | PROP/TSHELL; PARSE /MAT/PLAS_TAB/1: invalid literal for int() with base (+10) |
| RD_V_0240/Modele_HEXA_P14_Isolid24/tensile_TABULATED_HEXA_P14 | rd_v_material | ERROR | PARSE /MAT/PLAS_TAB/1: int( '1.0000; ERR /NODE card needs 4 fields, got 3 (+9) |
| RD_V_0240/Modele_HEXA_P20_Isolid15/tensile_TABULATED_HEXA_TYPE20 | rd_v_material | ERROR | PROP/TSHELL; PARSE /MAT/PLAS_TAB/1: int( '1.0 (+10) |
| RD_V_0240/Modele_QUAD_Ishell12/tensile_TABULATED_QUAD_Ishell12 | rd_v_material | ERROR | PARSE /MAT/PLAS_TAB/1: int( '1; ERR /NODE card needs 4 fields, got 3 (+9) |
| RD_V_0240/Modele_TETRA10_Itetra0/tensile_TABULATED_TETRA10 | rd_v_material | ERROR | TETRA10; PARSE /MAT/PLAS_TAB/1: int( '1.0000 (+10) |
| RD_V_0240/Modele_TETRA10_Itetra2/tensile_TABULATED_TETRA10_2 | rd_v_material | ERROR | TETRA10; PARSE /MAT/PLAS_TAB/1: int( '1.00 (+10) |
| RD_V_0240/Modele_TETRA4_Itetra0/tensile_TABULATED_TETRA | rd_v_material | ERROR | PARSE /MAT/PLAS_TAB/1: int( '1.0000000; ERR /NODE card needs 4 fields, got 3 (+9) |
| RD_V_0240/Modele_TETRA4_Itetra3/tensile_TABULATED_TETRA4_3 | rd_v_material | ERROR | PARSE /MAT/PLAS_TAB/1: int( '1.0000; ERR /NODE card needs 4 fields, got 3 (+9) |
| RD_V_0240/Modele_TRIA_Ishell24/tensile_TABULATED_TRIA | rd_v_material | ERROR | PARSE /MAT/PLAS_TAB/1: int( '1.00000000; ERR /NODE card needs 4 fields, got 3 (+9) |
| RD_V_0240/Tensile_QUAD_Ishell24/tensile_TABULATED_QUAD_Ishell24 | rd_v_material | ERROR | PARSE /MAT/PLAS_TAB/1: int( '1; ERR /NODE card needs 4 fields, got 3 (+9) |
| RD-V-0600/RD-V_0600_ADMAS_verification/ADMAS_0/ADMAS_Gravity | rd_v_tools | ERROR | GRSH3N/SH3N; GRSHEL/SHEL (+12) |
| RD-V-0600/RD-V_0600_ADMAS_verification/ADMAS_1/ADMAS_1 | rd_v_tools | ERROR | GRSH3N/SH3N; GRSHEL/SHEL (+12) |
| RD-V-0600/RD-V_0600_ADMAS_verification/ADMAS_2/ADMAS_2 | rd_v_tools | ERROR | GRSH3N/SH3N; GRSHEL/SHEL (+12) |
| RD-V-0600/RD-V_0600_ADMAS_verification/ADMAS_3/ADMAS_3 | rd_v_tools | ERROR | GRPART/PART; GRSH3N/SH3N (+12) |
| RD-V-0600/RD-V_0600_ADMAS_verification/ADMAS_4/ADMAS_4 | rd_v_tools | ERROR | GRPART/PART; GRSH3N/SH3N (+12) |
| RD-V-0600/RD-V_0600_ADMAS_verification/ADMAS_5/ADMAS_5 | rd_v_tools | ERROR | GRPART/PART; GRSH3N/SH3N (+12) |
| RD-V-0600/RD-V_0600_ADMAS_verification/ADMAS_6/ADMAS_6 | rd_v_tools | ERROR | GRPART/PART; GRSH3N/SH3N (+12) |
| RD-V-0600/RD-V_0600_ADMAS_verification/ADMAS_6_Iflag1/ADMAS_6_I1 | rd_v_tools | ERROR | GRPART/PART; GRSH3N/SH3N (+12) |
| RD-V-0600/RD-V_0600_ADMAS_verification/ADMAS_7/ADMAS_7 | rd_v_tools | ERROR | GRPART/PART; GRSH3N/SH3N (+12) |
| RD-V-0600/RD-V_0600_ADMAS_verification/ADMAS_7_Iflag1/ADMAS_7_I1 | rd_v_tools | ERROR | GRPART/PART; GRSH3N/SH3N (+12) |
| RD-HWX-T-1000/solved_completed_model/data/tensile_completed | tutorial | ERROR | ERR /NODE card needs 4 fields, got 3; ERR /NODE card needs 4 fields, got 3 (+8) |
| RD-HWX-T-1000/unsolved_starter_model/data/tensile_start | tutorial | ERROR | ERR /NODE card needs 4 fields, got 3; ERR /NODE card needs 4 fields, got 3 (+8) |
| RD-HWX-T-1010/solved/DYREL/data/cantilever_completed_DYREL | tutorial | ERROR | INTER/TYPE24; PROP/SPR_BEAM (+11) |
| RD-HWX-T-1010/solved/data/cantilever_completed | tutorial | ERROR | INTER/TYPE24; PROP/SPR_BEAM (+11) |
| RD-HWX-T-1010/unsolved/data/cantilever_start | tutorial | ERROR | ERR /BRICK 1: degenerated brick with 6 distinct nodes (penta; ERR /BRICK 2: degenerated brick with 6 distinct nodes (penta (+8) |
| RD-HWX-T-1020/solved/data/Front_Impact_completed | tutorial | ERROR | GRNOD/NODENS; GRSHEL/SHEL (+12) |
| RD-HWX-T-1020/unsolved/data/Front_Impact_initial | tutorial | ERROR | ERR /NODE card needs 4 fields, got 3; ERR /NODE card needs 4 fields, got 3 (+8) |
| RD-HWX-T-1030/solved/data/boat_ditching_completed | tutorial | ERROR | EOS/LINEAR; GRBRIC/PART (+12) |
| RD-HWX-T-1040/solved/data/gasket_completed | tutorial | ERROR | PARSE /IMPDISP/5: 'ZZ'; ERR /NODE card needs 4 fields, got 3 (+9) |
| RD-HWX-T-1050/solved/data1/valve_completed | tutorial | ERROR | EOS/LINEAR; GRBRIC/PART (+12) |
| RD-HWX-T-1050/solved/data2/valve_NRF_outlet | tutorial | ERROR | EOS/LINEAR; GRBRIC/PART (+12) |
| RD-HWX-T-1060/solved/data/3PB_completed | tutorial | ERROR | FAIL/CONNECT; MAT/CONNECT (+12) |
| RD-HWX-T-1060/unsolved/data/3PB_initial | tutorial | ERROR | PARSE /SHELL/4: int( '0.0'; PARSE /SHELL/5: int( '0.0' (+12) |
| TENSILE/TENSILE | tutorial | ERROR | PARSE /SH3N/1: int( '0.0'; PARSE /SHELL/1: int( '0.0' (+1) |
| phone_start/phone_start | tutorial | ERROR | MAT/LAW70; MAT/LAW82 (+10) |

### (b) Ranked keyword-gap table

cases_using = deck contains the unsupported family; cases_blocking = family is a hard physics skip in that run; sole_blocker = it is the case's ONLY hard gap; blocks = total skipped blocks corpus-wide.

| rank | unsupported family | cases using | cases blocking | sole blocker | blocks |
|---|---|---|---|---|---|
| 1 | /GRNOD/SURF | 222 | 222 | 10 | 236 |
| 2 | /GRNOD/GRNOD | 200 | 200 | 51 | 1216 |
| 3 | /LINE/EDGE | 190 | 190 | 0 | 190 |
| 4 | /GRSHEL/SHEL | 36 | 36 | 0 | 100 |
| 5 | /SURF/SURF | 35 | 35 | 0 | 90 |
| 6 | /FUNCT_SMOOTH | 34 | 34 | 23 | 36 |
| 7 | /UNIT | 28 | 28 | 8 | 42 |
| 8 | /SURF/GRSHEL | 26 | 26 | 0 | 62 |
| 9 | /GRSH3N/SH3N | 24 | 24 | 0 | 25 |
| 10 | /SURF/GRSH3N | 23 | 23 | 0 | 24 |
| 11 | /PROP/SH_ORTH | 21 | 21 | 0 | 25 |
| 12 | /MAT/HYD_VISC | 21 | 21 | 0 | 30 |
| 13 | /PROP/SPR_BEAM | 20 | 20 | 2 | 39 |
| 14 | /MAT/FABRI | 19 | 19 | 0 | 19 |
| 15 | /GRBRIC/PART | 19 | 19 | 1 | 214 |
| 16 | /INTER/TYPE24 | 18 | 18 | 2 | 18 |
| 17 | /MAT/GAS | 17 | 17 | 0 | 18 |
| 18 | /PROP/INJECT1 | 17 | 17 | 0 | 17 |
| 19 | /GRNOD/GENE | 17 | 17 | 0 | 18 |
| 20 | /SKEW/FIX | 16 | 16 | 3 | 48 |
| 21 | /MONVOL/AIRBAG1 | 16 | 16 | 0 | 16 |
| 22 | /INTER/LAGMUL | 14 | 14 | 0 | 58 |
| 23 | /PROP/SPR_GENE | 14 | 14 | 0 | 71 |
| 24 | /FRAME/FIX | 14 | 14 | 3 | 16 |
| 25 | /GRPART/PART | 14 | 14 | 0 | 14 |
| 26 | /MAT/VOID | 13 | 13 | 0 | 13 |
| 27 | /ALE/BCS | 12 | 12 | 0 | 25 |
| 28 | /SHEL16 | 12 | 12 | 0 | 62 |
| 29 | /GRBRIC/BRIC | 12 | 12 | 1 | 13 |
| 30 | /PROP/TYPE20 | 12 | 12 | 0 | 21 |
| 31 | /QUAD | 10 | 10 | 4 | 33 |
| 32 | /SKEW/MOV | 10 | 10 | 0 | 13 |
| 33 | /ALE/MAT | 10 | 10 | 0 | 20 |
| 34 | /PROP/VOID | 10 | 10 | 0 | 13 |
| 35 | /MAT/CONC | 8 | 8 | 8 | 8 |
| 36 | /FRAME/MOV | 7 | 7 | 6 | 7 |
| 37 | /INTER/TYPE18 | 7 | 7 | 0 | 7 |
| 38 | /MAT/LAW51 | 7 | 7 | 0 | 22 |
| 39 | /MAT/LAW81 | 7 | 7 | 7 | 7 |
| 40 | /PROP/CONNECT | 7 | 7 | 0 | 8 |
| 41 | /MOVE_FUNCT | 6 | 6 | 0 | 23 |
| 42 | /MAT/LAW62 | 6 | 6 | 0 | 6 |
| 43 | /EOS/LINEAR | 6 | 6 | 0 | 6 |
| 44 | /MAT/KELVINMAX | 6 | 6 | 0 | 6 |
| 45 | /AMS | 5 | 5 | 0 | 5 |
| - | (129 more families, each blocking <= 5 cases) | | | | |

**Highest-value next port target:** Single highest-value next port target: the entity group/set machinery, concretely /GRNOD/SURF + /GRNOD/GRNOD + /LINE/EDGE (surface-derived node groups, group-of-groups, edge lines). That trio is the complete hard-keyword gap for 252/529 decks (48%); the broader group/set family closure covers 262. Prerequisite mechanical reader fixes ride along: the fixed-format abutting-20-char-field split (/NODE, 59 cases; /SHELL//SH3N/SECT/FUNCT variants) and the /TH/NODE trailing-name column (360 cases) - both are parse bugs, not keyword gaps.

### (c) CRASH list and reader-bug signatures

**rc!=0 crashes: NONE** — no deck produced an uncaught traceback (0 of 529). Every reader failure was caught and reported as a '** ERROR while reading /X' message (rc=2). Those caught parse failures on real decks are the new bugs for the next milestone:

| # | parse-failure signature | cases | example |
|---|---|---|---|
| 1 | /TH/NODE: invalid literal for int() with base N: '<tok>' | 360 | TWISBEAM_0000.rad:483: while reading /TH/NODE/1: invalid literal for int() with base 10: '01x3' |
| 2 | /NODE: card needs N fields, got N (fixed-format 20-char coordinate columns abut with no whitespace; free-token | 59 | SHPB_H_2021_FEB12_0000.rad:99: /NODE card needs 4 fields, got 3 |
| 3 | /IMPVEL: '<tok>' | 35 | ROLLING_0000.rad:370: while reading /IMPVEL/1: 'XX' |
| 4 | /SHELL: invalid literal for int() with base N: '<tok>' | 28 | TANK_0000.rad:2962: while reading /SHELL/1: invalid literal for int() with base 10: '0.0' |
| 5 | /SH3N: invalid literal for int() with base N: '<tok>' | 18 | BAT_CIR_0000.rad:8785: while reading /SH3N/122: invalid literal for int() with base 10: '0.0' |
| 6 | /MAT/PLAS_TAB: invalid literal for int() with base N: '<tok>' | 17 | tensile_LAW36_0000.rad:20: while reading /MAT/PLAS_TAB/2: invalid literal for int() with base 10: '1.0' |
| 7 | /FUNCT: could not convert string to float: '<tok>' | 13 | rdv_0530_wave_propagation_0000.rad:8392: while reading /FUNCT/1: could not convert string to float: '5.0000000 |
| 8 | /EOS/IDEAL-GAS: could not convert string to float: '<tok>' | 11 | blast_experiment_0000.rad:32: while reading /EOS/IDEAL-GAS/1: could not convert string to float: 'EOS' |
| 9 | /INIVEL/AXIS: could not convert string to float: '<tok>' | 10 | DIF24416_0000.rad:14990: while reading /INIVEL/AXIS/1: could not convert string to float: 'Z' |
| 10 | /TH/PART: invalid literal for int() with base N: '<tok>' | 8 | pendulum_0000.rad:3287: while reading /TH/PART/2: invalid literal for int() with base 10: 'XXMOM' |
| 11 | /IMPDISP: '<tok>' | 8 | rubber_ring_0000.rad:3755: while reading /IMPDISP/30: 'ZZ' |
| 12 | /EOS/POLYNOMIAL: could not convert string to float: '<tok>' | 8 | 1BRICK_COMPRESSION_0000.rad:49: while reading /EOS/POLYNOMIAL/1: could not convert string to float: 'Conversio |
| 13 | /SECT: invalid literal for int() with base N: '<tok>' | 8 | bumper_LL4_0000.rad:77204: while reading /SECT/1: invalid literal for int() with base 10: '.1' |
| 14 | /MAT/PLAS_JOHNS: could not convert string to float: '<tok>' | 3 | TANK_0000.rad:4706: while reading /MAT/PLAS_JOHNS/1: could not convert string to float: '0.51.00000000000000E+ |
| 15 | /INIVEL/TRA: could not convert string to float: '<tok>' | 3 | SPHEX_mono_110_0000.rad:449329: while reading /INIVEL/TRA/1: could not convert string to float: '0&V' |
| 16 | /PART: list index out of range | 2 | BILLARD_0000.rad:9538: while reading /PART/1: list index out of range |
| 17 | /FAIL/BIQUAD: invalid literal for int() with base N: '<tok>' | 2 | main_TEST4_0000.rad:56: while reading /FAIL/BIQUAD/1/1: invalid literal for int() with base 10: '.2' |
| 18 | /SECT/PARAL: invalid literal for int() with base N: '<tok>' | 2 | CBOX_0000.rad:20429: while reading /SECT/PARAL/9: invalid literal for int() with base 10: '.1' |
| 19 | /RBODY: invalid literal for int() with base N: '<tok>' | 2 | Front_Impact_completed_0000.rad:27993: while reading /RBODY/13744: invalid literal for int() with base 10: '50 |
| 20 | /DAMP: invalid literal for int() with base N: '<tok>' | 1 | SHELL_LAW19_PROP9_0000.rad:22: while reading /DAMP/1: invalid literal for int() with base 10: '1E-5' |