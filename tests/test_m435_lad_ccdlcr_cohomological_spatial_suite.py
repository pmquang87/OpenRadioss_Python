"""Tests for Milestone M435: LadCoupledCoreDelaminationCrackingRate Failure Model, EngElectrothermoflexomagnetoplasmonicexcitonicphononicmagnonicpolaritonicResonanceEnergy, LagmulCohomologicalSpatialLinkageJoint, and SensorSpringTotalAngularSnapRate."""

from pathlib import Path
import pytest
from pyradioss.common.messages import MessageLog
from pyradioss.input.deck_reader import read_deck
from pyradioss.input.starter_keywords import parse_starter_deck
from pyradioss.model.model import Model


def _parse_starter(tmp_path: Path, deck_text: str):
    p = tmp_path / "TEST_0000.rad"
    p.write_text(deck_text, encoding="utf-8")
    model = Model()
    log = MessageLog()
    blocks = read_deck(str(p))
    parse_starter_deck(blocks, model, log)
    return model, log


def test_m435_fail_lad_coupled_core_delamination_cracking_rate_fixed(tmp_path: Path):
    c1 = f"{985.0:>20.4f}{2955.0:>20.4f}{765.0:>20.4f}{15.65:>20.4f}{0.465:>20.4f}"
    c2 = f"{1:>10d}{2:>10d}"
    c3 = f"{2490:>10d}"
    deck = f"""# RADIOSS STARTER DECK
/BEGIN
Ladeveze Coupled Core Delamination Cracking Rate Fixed Format Test
2022 0
/FAIL/LAD_COUPLED_CORE_DELAMINATION_CRACKING_RATE/2490
Ladeveze Coupled Core Delamination Cracking Failure Model
{c1}
{c2}
{c3}
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 2490 in model.fail_ladcoupledcoredelaminationcrackingrates
    fccdlcr = model.fail_ladcoupledcoredelaminationcrackingrates[2490]
    assert pytest.approx(fccdlcr.sigma_ccdlcr0) == 985.0
    assert pytest.approx(fccdlcr.sigma_ccdlcrc) == 2955.0
    assert pytest.approx(fccdlcr.gamma_ccdlcr) == 765.0
    assert pytest.approx(fccdlcr.p_ccdlcr) == 15.65
    assert pytest.approx(fccdlcr.d_ccdlcr_max) == 0.465
    assert fccdlcr.ifail_sh == 1
    assert fccdlcr.ifail_so == 2
    assert fccdlcr.fail_id == 2490
    assert pytest.approx(fccdlcr.sigma_cctr0) == 985.0
    assert pytest.approx(fccdlcr.sigma_cctrc) == 2955.0
    assert pytest.approx(fccdlcr.gamma_cctr) == 765.0
    assert pytest.approx(fccdlcr.p_cctr) == 15.65
    assert pytest.approx(fccdlcr.d_cctr_max) == 0.465
    assert len(model.raw_fails) == 1
    assert model.raw_fails[0][1].type == "LAD_COUPLED_CORE_DELAMINATION_CRACKING_RATE"


def test_m435_fail_lad_coupled_core_delamination_cracking_rate_free(tmp_path: Path):
    deck = """# RADIOSS FREE DECK
/BEGIN
Ladeveze Coupled Core Delamination Cracking Rate Free Format Test
/FAIL/LAD_COUPLED_CORE_DELAMINATION_CRACKING_RATE/2491
995.0, 2985.0, 775.0, 15.85, 0.455
1, 1
2491
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 2491 in model.fail_ladcoupledcoredelaminationcrackingrates
    fccdlcr = model.fail_ladcoupledcoredelaminationcrackingrates[2491]
    assert pytest.approx(fccdlcr.sigma_ccdlcr0) == 995.0
    assert pytest.approx(fccdlcr.sigma_ccdlcrc) == 2985.0
    assert pytest.approx(fccdlcr.gamma_ccdlcr) == 775.0
    assert pytest.approx(fccdlcr.p_ccdlcr) == 15.85
    assert pytest.approx(fccdlcr.d_ccdlcr_max) == 0.455
    assert fccdlcr.fail_id == 2491


def test_m435_fail_lad_coupled_core_delamination_cracking_rate_aliases(tmp_path: Path):
    deck = """# RADIOSS ALIAS DECK
/BEGIN
Ladeveze Coupled Core Delamination Cracking Rate Aliases Test
/FAIL/LADEVEZE_COUPLED_CORE_DELAMINATION_CRACKING_RATE/2492
975.0, 2925.0, 755.0, 15.45, 0.475
1, 1
/FAIL/LAD_COUPLE_CORE_DELAMINATION_CRACKING_RATE/2493
975.0, 2925.0, 755.0, 15.45, 0.475
1, 1
/FAIL/LAD_CCDLCR/2494
975.0, 2925.0, 755.0, 15.45, 0.475
1, 1
/FAIL/LAD_CCDLCR_MODEL/2495
975.0, 2925.0, 755.0, 15.45, 0.475
1, 1
/FAIL/LAD_CCDLCR_LAW/2496
975.0, 2925.0, 755.0, 15.45, 0.475
1, 1
/FAIL/LADEVEZE_RATE_DEPENDENT_COUPLED_CORE_DELAMINATION_CRACKING/2497
975.0, 2925.0, 755.0, 15.45, 0.475
1, 1
/FAIL/LAD_COUPLED_CORE_TEARING_RATE/2498
975.0, 2925.0, 755.0, 15.45, 0.475
1, 1
/FAIL/LADEVEZE_COUPLED_CORE_TEARING_RATE/2499
975.0, 2925.0, 755.0, 15.45, 0.475
1, 1
/FAIL/LAD_COUPLE_CORE_TEARING_RATE/2500
975.0, 2925.0, 755.0, 15.45, 0.475
1, 1
/FAIL/LADEVEZE_COUPLE_CORE_TEARING_RATE/2501
975.0, 2925.0, 755.0, 15.45, 0.475
1, 1
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 2492 in model.fail_ladcoupledcoredelaminationcrackingrates
    assert 2493 in model.fail_ladcoupledcoredelaminationcrackingrates
    assert 2494 in model.fail_ladcoupledcoredelaminationcrackingrates
    assert 2495 in model.fail_ladcoupledcoredelaminationcrackingrates
    assert 2496 in model.fail_ladcoupledcoredelaminationcrackingrates
    assert 2497 in model.fail_ladcoupledcoredelaminationcrackingrates
    assert 2498 in model.fail_ladcoupledcoredelaminationcrackingrates
    assert 2499 in model.fail_ladcoupledcoredelaminationcrackingrates
    assert 2500 in model.fail_ladcoupledcoredelaminationcrackingrates
    assert 2501 in model.fail_ladcoupledcoredelaminationcrackingrates
    assert len(model.raw_fails) == 10


def test_m435_fail_lad_coupled_core_delamination_cracking_rate_missing_card(tmp_path: Path):
    deck = """# RADIOSS MISSING CARD DECK
/BEGIN
/FAIL/LAD_COUPLED_CORE_DELAMINATION_CRACKING_RATE/2502
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) > 0


def test_m435_eng_electrothermoflexomagnetoplasmonicexcitonicphononicmagnonicpolaritonic_resonance_energy_fixed(tmp_path: Path):
    c1 = f"{0.0935:>20.4f}{895:>10d}"
    deck = f"""# RADIOSS STARTER DECK
/BEGIN
Engine Electrothermoflexomagnetoplasmonicexcitonicphononicmagnonicpolaritonic Resonance Energy Fixed Format Test
2022 0
/ENG/ELECTROTHERMOFLEXOMAGNETOPLASMONICEXCITONICPHONONICMAGNONICPOLARITONIC_RESONANCE_ENERGY/795
Resonance Energy Output Directive
{c1}
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 795 in model.eng_electrothermoflexomagnetoplasmonicexcitonicphononicmagnonicpolaritonic_resonance_energies
    eng = model.eng_electrothermoflexomagnetoplasmonicexcitonicphononicmagnonicpolaritonic_resonance_energies[795]
    assert pytest.approx(eng.dt_etfplexphmnp) == 0.0935
    assert eng.sens_id == 895


def test_m435_eng_electrothermoflexomagnetoplasmonicexcitonicphononicmagnonicpolaritonic_resonance_energy_free(tmp_path: Path):
    deck = """# RADIOSS FREE DECK
/BEGIN
Engine Electrothermoflexomagnetoplasmonicexcitonicphononicmagnonicpolaritonic Resonance Energy Free Format Test
/ENG/ELECTROTHERMOFLEXOMAGNETOPLASMONICEXCITONICPHONONICMAGNONICPOLARITONIC_RESONANCE_ENERGY/796
0.0945, 896
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 796 in model.eng_electrothermoflexomagnetoplasmonicexcitonicphononicmagnonicpolaritonic_resonance_energies
    eng = model.eng_electrothermoflexomagnetoplasmonicexcitonicphononicmagnonicpolaritonic_resonance_energies[796]
    assert pytest.approx(eng.dt_etfplexphmnp) == 0.0945
    assert eng.sens_id == 896


def test_m435_eng_electrothermoflexomagnetoplasmonicexcitonicphononicmagnonicpolaritonic_resonance_energy_aliases(tmp_path: Path):
    deck = """# RADIOSS ALIAS DECK
/BEGIN
Engine Electrothermoflexomagnetoplasmonicexcitonicphononicmagnonicpolaritonic Resonance Energy Aliases Test
/ENG/ELECTRO_THERM_FLEXO_MAG_PLASMON_EXCITON_PHONON_MAGNONIC_POLARITON_RES_WORK/797
0.0955, 897
/ENG/EELECTROTHERMOFLEXOMAGNETOPLASMONICEXCITONICPHONONICMAGNONICPOLARITONICRESONANCE/798
0.0965, 898
/ENG/ELECTROTHERMOFLEXOMAGNETOPLASMONICEXCITONICPHONONICMAGNONICPOLARITONIC_RESONANCE_DISSIPATION/799
0.0975, 899
/ENG/ET_ELECTROTHERMOFLEXOMAGNETOPLASMONICEXCITONICPHONONICMAGNONICPOLARITONIC_RESONANCE/800
0.0985, 900
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 797 in model.eng_electrothermoflexomagnetoplasmonicexcitonicphononicmagnonicpolaritonic_resonance_energies
    assert 798 in model.eng_electrothermoflexomagnetoplasmonicexcitonicphononicmagnonicpolaritonic_resonance_energies
    assert 799 in model.eng_electrothermoflexomagnetoplasmonicexcitonicphononicmagnonicpolaritonic_resonance_energies
    assert 800 in model.eng_electrothermoflexomagnetoplasmonicexcitonicphononicmagnonicpolaritonic_resonance_energies


def test_m435_eng_electrothermoflexomagnetoplasmonicexcitonicphononicmagnonicpolaritonic_resonance_energy_missing_card(tmp_path: Path):
    deck = """# RADIOSS MISSING CARD DECK
/BEGIN
/ENG/ELECTROTHERMOFLEXOMAGNETOPLASMONICEXCITONICPHONONICMAGNONICPOLARITONIC_RESONANCE_ENERGY/802
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) > 0


def test_m435_lagmul_cohomological_spatial_linkage_joint_fixed(tmp_path: Path):
    c1 = f"{881:>10d}{882:>10d}{883:>10d}{7.65e7:>20.4f}{485:>10d}{6.75e-4:>20.4e}"
    c2 = f"{545.0:>20.4f}{525.0:>20.4f}{555.0:>20.4f}{475.0:>20.4f}"
    deck = f"""# RADIOSS STARTER DECK
/BEGIN
Cohomological Spatial Linkage Joint Fixed Format Test
2022 0
/LAGMUL/COHOMOLOGICAL_SPATIAL_LINKAGE_JOINT/845
Cohomological Spatial 6R Joint
{c1}
{c2}
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 845 in model.lagmul_cohomological_spatial_linkage_joints
    joint = model.lagmul_cohomological_spatial_linkage_joints[845]
    assert joint.node1 == 881
    assert joint.node2 == 882
    assert joint.node3 == 883
    assert pytest.approx(joint.stiff) == 7.65e7
    assert joint.skew_id == 485
    assert pytest.approx(joint.tol) == 6.75e-4
    assert pytest.approx(joint.link_len_a) == 545.0
    assert pytest.approx(joint.link_len_b) == 525.0
    assert pytest.approx(joint.twist_angle_alpha) == 555.0
    assert pytest.approx(joint.offset_distance_s) == 475.0
    assert pytest.approx(joint.offset_distance_r) == 475.0
    assert pytest.approx(joint.offset_distance_v) == 475.0
    assert pytest.approx(joint.offset_distance_h) == 475.0
    assert pytest.approx(joint.offset_distance_u) == 475.0
    assert pytest.approx(joint.offset_distance_f) == 475.0


def test_m435_lagmul_cohomological_spatial_linkage_joint_free(tmp_path: Path):
    deck = """# RADIOSS FREE DECK
/BEGIN
Cohomological Spatial Linkage Joint Free Format Test
/LAGMUL/COHOMOLOGICAL_SPATIAL_LINKAGE_JOINT/846
981, 982, 983, 7.85e7, 486, 6.85e-4
565.0, 535.0, 575.0, 490.0
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 846 in model.lagmul_cohomological_spatial_linkage_joints
    joint = model.lagmul_cohomological_spatial_linkage_joints[846]
    assert joint.node1 == 981
    assert joint.node2 == 982
    assert joint.node3 == 983
    assert pytest.approx(joint.stiff) == 7.85e7
    assert joint.skew_id == 486
    assert pytest.approx(joint.tol) == 6.85e-4
    assert pytest.approx(joint.link_len_a) == 565.0
    assert pytest.approx(joint.link_len_b) == 535.0
    assert pytest.approx(joint.twist_angle_alpha) == 575.0
    assert pytest.approx(joint.offset_distance_s) == 490.0


def test_m435_lagmul_cohomological_spatial_linkage_joint_aliases(tmp_path: Path):
    deck = """# RADIOSS ALIAS DECK
/BEGIN
Cohomological Spatial Linkage Joint Aliases Test
/COHOMOLOGICAL_SPATIAL_LINKAGE_JOINT/847
1081, 1082, 1083, 1.0e6, 0, 1.0e-6
505.0, 495.0, 505.0, 455.0
/LAGMUL/COHOMOLOGICAL_SPATIAL_LINKAGE/848
1081, 1082, 1083, 1.0e6, 0, 1.0e-6
505.0, 495.0, 505.0, 455.0
/COHOMOLOGICAL_SPATIAL_LINKAGE/849
1081, 1082, 1083, 1.0e6, 0, 1.0e-6
505.0, 495.0, 505.0, 455.0
/COHOMOLOGICAL_SPATIAL_MULTI_LOOP_MECHANISM/850
1081, 1082, 1083, 1.0e6, 0, 1.0e-6
505.0, 495.0, 505.0, 455.0
/COHOMOLOGICAL_SPATIAL_SYMMETRIC_MECHANISM/851
1081, 1082, 1083, 1.0e6, 0, 1.0e-6
505.0, 495.0, 505.0, 455.0
/COHOMOLOGICAL_SPATIAL_6R_MECHANISM/852
1081, 1082, 1083, 1.0e6, 0, 1.0e-6
505.0, 495.0, 505.0, 455.0
/COHOMOLOGICAL_SPATIAL_OVERCONSTRAINED_MECHANISM/853
1081, 1082, 1083, 1.0e6, 0, 1.0e-6
505.0, 495.0, 505.0, 455.0
/LAGMUL/SHEAF_SPATIAL_LINKAGE_JOINT/854
1081, 1082, 1083, 1.0e6, 0, 1.0e-6
505.0, 495.0, 505.0, 455.0
/SHEAF_SPATIAL_LINKAGE_JOINT/855
1081, 1082, 1083, 1.0e6, 0, 1.0e-6
505.0, 495.0, 505.0, 455.0
/LAGMUL/FIBER_BUNDLE_SPATIAL_LINKAGE_JOINT/856
1081, 1082, 1083, 1.0e6, 0, 1.0e-6
505.0, 495.0, 505.0, 455.0
/FIBER_BUNDLE_SPATIAL_LINKAGE_JOINT/857
1081, 1082, 1083, 1.0e6, 0, 1.0e-6
505.0, 495.0, 505.0, 455.0
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 847 in model.lagmul_cohomological_spatial_linkage_joints
    assert 848 in model.lagmul_cohomological_spatial_linkage_joints
    assert 849 in model.lagmul_cohomological_spatial_linkage_joints
    assert 850 in model.lagmul_cohomological_spatial_linkage_joints
    assert 851 in model.lagmul_cohomological_spatial_linkage_joints
    assert 852 in model.lagmul_cohomological_spatial_linkage_joints
    assert 853 in model.lagmul_cohomological_spatial_linkage_joints
    assert 854 in model.lagmul_cohomological_spatial_linkage_joints
    assert 855 in model.lagmul_cohomological_spatial_linkage_joints
    assert 856 in model.lagmul_cohomological_spatial_linkage_joints
    assert 857 in model.lagmul_cohomological_spatial_linkage_joints


def test_m435_lagmul_cohomological_spatial_linkage_joint_missing_card(tmp_path: Path):
    deck = """# RADIOSS MISSING CARD DECK
/BEGIN
/LAGMUL/COHOMOLOGICAL_SPATIAL_LINKAGE_JOINT/858
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) > 0


def test_m435_sensor_spring_total_angular_snap_rate_fixed(tmp_path: Path):
    c1 = f"{1175:>10d}{9.15e6:>20.4f}{0.0085:>20.4f}"
    deck = f"""# RADIOSS STARTER DECK
/BEGIN
Sensor Spring Total Angular Snap Rate Fixed Format Test
2022 0
/SENSOR/SPRING_TOTAL_ANGULAR_SNAP_RATE/975
Spring Total Angular Snap Rate Sensor
{c1}
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 975 in model.sensor_spring_total_angular_snap_rates
    sensor = model.sensor_spring_total_angular_snap_rates[975]
    assert sensor.spring_id == 1175
    assert pytest.approx(sensor.jtot_ang_snp_max) == 9.15e6
    assert pytest.approx(sensor.jtot_ang_lock_max) == 9.15e6
    assert pytest.approx(sensor.jtot_ang_crackle_max) == 9.15e6
    assert pytest.approx(sensor.jtot_ang_shot_max) == 9.15e6
    assert pytest.approx(sensor.jtot_ang_pop_max) == 9.15e6
    assert pytest.approx(sensor.jtot_ang_crk_max) == 9.15e6
    assert pytest.approx(sensor.jtot_ang_drop_rate_max) == 9.15e6
    assert pytest.approx(sensor.jtot_ang_rate_max) == 9.15e6
    assert pytest.approx(sensor.jtot_ang_roc_rate_max) == 9.15e6
    assert pytest.approx(sensor.jtot_ang_snap_rate_max) == 9.15e6
    assert pytest.approx(sensor.jtang_snp_max) == 9.15e6
    assert pytest.approx(sensor.jtang_snap_rate_max) == 9.15e6
    assert pytest.approx(sensor.jtang_shot_max) == 9.15e6
    assert pytest.approx(sensor.jtang_lock_max) == 9.15e6
    assert pytest.approx(sensor.jtang_pop_max) == 9.15e6
    assert pytest.approx(sensor.jtang_crackle_max) == 9.15e6
    assert pytest.approx(sensor.jtang_crk_max) == 9.15e6
    assert pytest.approx(sensor.jtang_drop_rate_max) == 9.15e6
    assert pytest.approx(sensor.jtang_rate_max) == 9.15e6
    assert pytest.approx(sensor.jtang_roc_rate_max) == 9.15e6
    assert pytest.approx(sensor.jtang_drop_max) == 9.15e6
    assert pytest.approx(sensor.j_max) == 9.15e6
    assert pytest.approx(sensor.t_delay) == 0.0085
    assert len(model.sensors) == 1
    assert model.sensors[0].kind == "SPRING_TOTAL_ANGULAR_SNAP_RATE"


def test_m435_sensor_spring_total_angular_snap_rate_free(tmp_path: Path):
    deck = """# RADIOSS FREE DECK
/BEGIN
Sensor Spring Total Angular Snap Rate Free Format Test
/SENSOR/SPRING_TOTAL_ANGULAR_SNAP_RATE/976
1176, 9.25e6, 0.0095
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 976 in model.sensor_spring_total_angular_snap_rates
    sensor = model.sensor_spring_total_angular_snap_rates[976]
    assert sensor.spring_id == 1176
    assert pytest.approx(sensor.jtot_ang_snp_max) == 9.25e6
    assert pytest.approx(sensor.t_delay) == 0.0095


def test_m435_sensor_spring_total_angular_snap_rate_aliases(tmp_path: Path):
    deck = """# RADIOSS ALIAS DECK
/BEGIN
Sensor Spring Total Angular Snap Rate Aliases Test
/SENSOR/SPRING_TOT_ANG_SNAP_RATE/977
1177, 9.05e6, 0.0075
/SENSOR/SPRING_SNAP_RATE_TOT_ANG/978
1178, 9.05e6, 0.0075
/SENSOR/TOTAL_ANGULAR_SNAP_RATE_SPRING/979
1179, 9.05e6, 0.0075
/SENSOR/SPRING_SNAP_TOT_ANG/980
1180, 9.05e6, 0.0075
/SENSOR/SPRING_SNAP_RATE_ANG_TOT/981
1181, 9.05e6, 0.0075
/SENSOR/SPRING_RESULTANT_ANGULAR_SNAP_RATE/982
1182, 9.05e6, 0.0075
/SENSOR/SPRING_RES_ANG_SNAP_RATE/983
1183, 9.05e6, 0.0075
/SENSOR/RESULTANT_ANGULAR_SNAP_RATE_SPRING/984
1184, 9.05e6, 0.0075
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 977 in model.sensor_spring_total_angular_snap_rates
    assert 978 in model.sensor_spring_total_angular_snap_rates
    assert 979 in model.sensor_spring_total_angular_snap_rates
    assert 980 in model.sensor_spring_total_angular_snap_rates
    assert 981 in model.sensor_spring_total_angular_snap_rates
    assert 982 in model.sensor_spring_total_angular_snap_rates
    assert 983 in model.sensor_spring_total_angular_snap_rates
    assert 984 in model.sensor_spring_total_angular_snap_rates
    assert len(model.sensors) == 8


def test_m435_sensor_spring_total_angular_snap_rate_missing_card(tmp_path: Path):
    deck = """# RADIOSS MISSING CARD DECK
/BEGIN
/SENSOR/SPRING_TOTAL_ANGULAR_SNAP_RATE/985
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) > 0
