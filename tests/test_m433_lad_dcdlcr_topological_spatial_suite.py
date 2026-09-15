"""Tests for Milestone M433: LadDynamicCoreDelaminationCrackingRate Failure Model, EngElectrothermoflexomagnetoplasmonicexcitonicmagnonicpolaritonicResonanceEnergy, TopologicalSpatialLinkageJoint, and SensorSpringTorsionalSnapRate."""

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


def test_m433_fail_lad_dynamic_core_delamination_cracking_rate_fixed(tmp_path: Path):
    c1 = f"{975.0:>20.4f}{2925.0:>20.4f}{755.0:>20.4f}{15.45:>20.4f}{0.475:>20.4f}"
    c2 = f"{1:>10d}{2:>10d}"
    c3 = f"{2470:>10d}"
    deck = f"""# RADIOSS STARTER DECK
/BEGIN
Ladeveze Dynamic Core Delamination Cracking Rate Fixed Format Test
2022 0
/FAIL/LAD_DYNAMIC_CORE_DELAMINATION_CRACKING_RATE/2470
Ladeveze Dynamic Core Delamination Cracking Failure Model
{c1}
{c2}
{c3}
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 2470 in model.fail_laddynamiccoredelaminationcrackingrates
    fdcdlcr = model.fail_laddynamiccoredelaminationcrackingrates[2470]
    assert pytest.approx(fdcdlcr.sigma_dcdlcr0) == 975.0
    assert pytest.approx(fdcdlcr.sigma_dcdlcrc) == 2925.0
    assert pytest.approx(fdcdlcr.gamma_dcdlcr) == 755.0
    assert pytest.approx(fdcdlcr.p_dcdlcr) == 15.45
    assert pytest.approx(fdcdlcr.d_dcdlcr_max) == 0.475
    assert fdcdlcr.ifail_sh == 1
    assert fdcdlcr.ifail_so == 2
    assert fdcdlcr.fail_id == 2470
    assert len(model.raw_fails) == 1
    assert model.raw_fails[0][1].type == "LAD_DYNAMIC_CORE_DELAMINATION_CRACKING_RATE"


def test_m433_fail_lad_dynamic_core_delamination_cracking_rate_free(tmp_path: Path):
    deck = """# RADIOSS FREE DECK
/BEGIN
Ladeveze Dynamic Core Delamination Cracking Rate Free Format Test
/FAIL/LAD_DYNAMIC_CORE_DELAMINATION_CRACKING_RATE/2471
985.0, 2955.0, 765.0, 15.65, 0.465
1, 1
2471
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 2471 in model.fail_laddynamiccoredelaminationcrackingrates
    fdcdlcr = model.fail_laddynamiccoredelaminationcrackingrates[2471]
    assert pytest.approx(fdcdlcr.sigma_dcdlcr0) == 985.0
    assert pytest.approx(fdcdlcr.sigma_dcdlcrc) == 2955.0
    assert pytest.approx(fdcdlcr.gamma_dcdlcr) == 765.0
    assert pytest.approx(fdcdlcr.p_dcdlcr) == 15.65
    assert pytest.approx(fdcdlcr.d_dcdlcr_max) == 0.465
    assert fdcdlcr.fail_id == 2471


def test_m433_fail_lad_dynamic_core_delamination_cracking_rate_aliases(tmp_path: Path):
    deck = """# RADIOSS ALIAS DECK
/BEGIN
Ladeveze Dynamic Core Delamination Cracking Rate Aliases Test
/FAIL/LADEVEZE_DYNAMIC_CORE_DELAMINATION_CRACKING_RATE/2472
965.0, 2895.0, 745.0, 15.25, 0.485
1, 1
/FAIL/LAD_DCDLCR/2473
965.0, 2895.0, 745.0, 15.25, 0.485
1, 1
/FAIL/LAD_DCDLCR_MODEL/2474
965.0, 2895.0, 745.0, 15.25, 0.485
1, 1
/FAIL/LAD_DCDLCR_LAW/2475
965.0, 2895.0, 745.0, 15.25, 0.485
1, 1
/FAIL/LADEVEZE_RATE_DEPENDENT_DYNAMIC_CORE_DELAMINATION_CRACKING/2476
965.0, 2895.0, 745.0, 15.25, 0.485
1, 1
/FAIL/LAD_DYNAMIC_CORE_TEARING_RATE/2477
965.0, 2895.0, 745.0, 15.25, 0.485
1, 1
/FAIL/LADEVEZE_DYNAMIC_CORE_TEARING_RATE/2478
965.0, 2895.0, 745.0, 15.25, 0.485
1, 1
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 2472 in model.fail_laddynamiccoredelaminationcrackingrates
    assert 2473 in model.fail_laddynamiccoredelaminationcrackingrates
    assert 2474 in model.fail_laddynamiccoredelaminationcrackingrates
    assert 2475 in model.fail_laddynamiccoredelaminationcrackingrates
    assert 2476 in model.fail_laddynamiccoredelaminationcrackingrates
    assert 2477 in model.fail_laddynamiccoredelaminationcrackingrates
    assert 2478 in model.fail_laddynamiccoredelaminationcrackingrates
    assert len(model.raw_fails) == 7


def test_m433_fail_lad_dynamic_core_delamination_cracking_rate_missing_card(tmp_path: Path):
    deck = """# RADIOSS MISSING CARD DECK
/BEGIN
/FAIL/LAD_DYNAMIC_CORE_DELAMINATION_CRACKING_RATE/2479
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) > 0


def test_m433_eng_electrothermoflexomagnetoplasmonicexcitonicmagnonicpolaritonic_resonance_energy_fixed(tmp_path: Path):
    c1 = f"{0.0925:>20.4f}{885:>10d}"
    deck = f"""# RADIOSS STARTER DECK
/BEGIN
Engine Electrothermoflexomagnetoplasmonicexcitonicmagnonicpolaritonic Resonance Energy Fixed Format Test
2022 0
/ENG/ELECTROTHERMOFLEXOMAGNETOPLASMONICEXCITONICMAGNONICPOLARITONIC_RESONANCE_ENERGY/785
Resonance Energy Output Directive
{c1}
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 785 in model.eng_electrothermoflexomagnetoplasmonicexcitonicmagnonicpolaritonic_resonance_energies
    eng = model.eng_electrothermoflexomagnetoplasmonicexcitonicmagnonicpolaritonic_resonance_energies[785]
    assert pytest.approx(eng.dt_etfplexmmnp) == 0.0925
    assert eng.sens_id == 885


def test_m433_eng_electrothermoflexomagnetoplasmonicexcitonicmagnonicpolaritonic_resonance_energy_free(tmp_path: Path):
    deck = """# RADIOSS FREE DECK
/BEGIN
Engine Electrothermoflexomagnetoplasmonicexcitonicmagnonicpolaritonic Resonance Energy Free Format Test
/ENG/ELECTROTHERMOFLEXOMAGNETOPLASMONICEXCITONICMAGNONICPOLARITONIC_RESONANCE_ENERGY/786
0.0935, 886
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 786 in model.eng_electrothermoflexomagnetoplasmonicexcitonicmagnonicpolaritonic_resonance_energies
    eng = model.eng_electrothermoflexomagnetoplasmonicexcitonicmagnonicpolaritonic_resonance_energies[786]
    assert pytest.approx(eng.dt_etfplexmmnp) == 0.0935
    assert eng.sens_id == 886


def test_m433_eng_electrothermoflexomagnetoplasmonicexcitonicmagnonicpolaritonic_resonance_energy_aliases(tmp_path: Path):
    deck = """# RADIOSS ALIAS DECK
/BEGIN
Engine Electrothermoflexomagnetoplasmonicexcitonicmagnonicpolaritonic Resonance Energy Aliases Test
/ENG/ELECTRO_THERM_FLEXO_MAG_PLASMON_EXCITON_MAGNON_POLARITON_RES_WORK/787
0.0945, 887
/ENG/EELECTROTHERMOFLEXOMAGNETOPLASMONICEXCITONICMAGNONICPOLARITONICRESONANCE/788
0.0955, 888
/ENG/ELECTROTHERMOFLEXOMAGNETOPLASMONICEXCITONICMAGNONICPOLARITONIC_RESONANCE_DISSIPATION/789
0.0965, 889
/ENG/ET_ELECTROTHERMOFLEXOMAGNETOPLASMONICEXCITONICMAGNONICPOLARITONIC_RESONANCE/790
0.0975, 890
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 787 in model.eng_electrothermoflexomagnetoplasmonicexcitonicmagnonicpolaritonic_resonance_energies
    assert 788 in model.eng_electrothermoflexomagnetoplasmonicexcitonicmagnonicpolaritonic_resonance_energies
    assert 789 in model.eng_electrothermoflexomagnetoplasmonicexcitonicmagnonicpolaritonic_resonance_energies
    assert 790 in model.eng_electrothermoflexomagnetoplasmonicexcitonicmagnonicpolaritonic_resonance_energies


def test_m433_eng_electrothermoflexomagnetoplasmonicexcitonicmagnonicpolaritonic_resonance_energy_missing_card(tmp_path: Path):
    deck = """# RADIOSS MISSING CARD DECK
/BEGIN
/ENG/ELECTROTHERMOFLEXOMAGNETOPLASMONICEXCITONICMAGNONICPOLARITONIC_RESONANCE_ENERGY/791
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) > 0


def test_m433_lagmul_topological_spatial_linkage_joint_fixed(tmp_path: Path):
    c1 = f"{871:>10d}{872:>10d}{873:>10d}{7.55e7:>20.4f}{475:>10d}{6.65e-4:>20.4e}"
    c2 = f"{535.0:>20.4f}{515.0:>20.4f}{545.0:>20.4f}{465.0:>20.4f}"
    deck = f"""# RADIOSS STARTER DECK
/BEGIN
Topological Spatial Linkage Joint Fixed Format Test
2022 0
/LAGMUL/TOPOLOGICAL_SPATIAL_LINKAGE_JOINT/835
Topological Spatial 6R Joint
{c1}
{c2}
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 835 in model.lagmul_topological_spatial_linkage_joints
    joint = model.lagmul_topological_spatial_linkage_joints[835]
    assert joint.node1 == 871
    assert joint.node2 == 872
    assert joint.node3 == 873
    assert pytest.approx(joint.stiff) == 7.55e7
    assert joint.skew_id == 475
    assert pytest.approx(joint.tol) == 6.65e-4
    assert pytest.approx(joint.link_len_a) == 535.0
    assert pytest.approx(joint.link_len_b) == 515.0
    assert pytest.approx(joint.twist_angle_alpha) == 545.0
    assert pytest.approx(joint.offset_distance_s) == 465.0
    assert pytest.approx(joint.offset_distance_r) == 465.0
    assert pytest.approx(joint.offset_distance_v) == 465.0
    assert pytest.approx(joint.offset_distance_h) == 465.0
    assert pytest.approx(joint.offset_distance_u) == 465.0
    assert pytest.approx(joint.offset_distance_f) == 465.0


def test_m433_lagmul_topological_spatial_linkage_joint_free(tmp_path: Path):
    deck = """# RADIOSS FREE DECK
/BEGIN
Topological Spatial Linkage Joint Free Format Test
/LAGMUL/TOPOLOGICAL_SPATIAL_LINKAGE_JOINT/836
971, 972, 973, 7.75e7, 476, 6.75e-4
555.0, 525.0, 565.0, 480.0
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 836 in model.lagmul_topological_spatial_linkage_joints
    joint = model.lagmul_topological_spatial_linkage_joints[836]
    assert joint.node1 == 971
    assert joint.node2 == 972
    assert joint.node3 == 973
    assert pytest.approx(joint.stiff) == 7.75e7
    assert joint.skew_id == 476
    assert pytest.approx(joint.tol) == 6.75e-4
    assert pytest.approx(joint.link_len_a) == 555.0
    assert pytest.approx(joint.link_len_b) == 525.0
    assert pytest.approx(joint.twist_angle_alpha) == 565.0
    assert pytest.approx(joint.offset_distance_s) == 480.0


def test_m433_lagmul_topological_spatial_linkage_joint_aliases(tmp_path: Path):
    deck = """# RADIOSS ALIAS DECK
/BEGIN
Topological Spatial Linkage Joint Aliases Test
/TOPOLOGICAL_SPATIAL_LINKAGE_JOINT/837
1071, 1072, 1073, 1.0e6, 0, 1.0e-6
495.0, 485.0, 495.0, 445.0
/LAGMUL/TOPOLOGICAL_SPATIAL_LINKAGE/838
1071, 1072, 1073, 1.0e6, 0, 1.0e-6
495.0, 485.0, 495.0, 445.0
/TOPOLOGICAL_SPATIAL_LINKAGE/839
1071, 1072, 1073, 1.0e6, 0, 1.0e-6
495.0, 485.0, 495.0, 445.0
/TOPOLOGICAL_SPATIAL_MULTI_LOOP_MECHANISM/840
1071, 1072, 1073, 1.0e6, 0, 1.0e-6
495.0, 485.0, 495.0, 445.0
/TOPOLOGICAL_SPATIAL_SYMMETRIC_MECHANISM/841
1071, 1072, 1073, 1.0e6, 0, 1.0e-6
495.0, 485.0, 495.0, 445.0
/TOPOLOGICAL_SPATIAL_6R_MECHANISM/842
1071, 1072, 1073, 1.0e6, 0, 1.0e-6
495.0, 485.0, 495.0, 445.0
/TOPOLOGICAL_SPATIAL_OVERCONSTRAINED_MECHANISM/843
1071, 1072, 1073, 1.0e6, 0, 1.0e-6
495.0, 485.0, 495.0, 445.0
/LAGMUL/DIFFERENTIAL_SPATIAL_LINKAGE_JOINT/844
1071, 1072, 1073, 1.0e6, 0, 1.0e-6
495.0, 485.0, 495.0, 445.0
/DIFFERENTIAL_SPATIAL_LINKAGE_JOINT/845
1071, 1072, 1073, 1.0e6, 0, 1.0e-6
495.0, 485.0, 495.0, 445.0
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 837 in model.lagmul_topological_spatial_linkage_joints
    assert 838 in model.lagmul_topological_spatial_linkage_joints
    assert 839 in model.lagmul_topological_spatial_linkage_joints
    assert 840 in model.lagmul_topological_spatial_linkage_joints
    assert 841 in model.lagmul_topological_spatial_linkage_joints
    assert 842 in model.lagmul_topological_spatial_linkage_joints
    assert 843 in model.lagmul_topological_spatial_linkage_joints
    assert 844 in model.lagmul_topological_spatial_linkage_joints
    assert 845 in model.lagmul_topological_spatial_linkage_joints


def test_m433_lagmul_topological_spatial_linkage_joint_missing_card(tmp_path: Path):
    deck = """# RADIOSS MISSING CARD DECK
/BEGIN
/LAGMUL/TOPOLOGICAL_SPATIAL_LINKAGE_JOINT/846
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) > 0


def test_m433_sensor_spring_torsional_snap_rate_fixed(tmp_path: Path):
    c1 = f"{1165:>10d}{9.05e6:>20.4f}{0.0075:>20.4f}"
    deck = f"""# RADIOSS STARTER DECK
/BEGIN
Sensor Spring Torsional Snap Rate Fixed Format Test
2022 0
/SENSOR/SPRING_TORSIONAL_SNAP_RATE/965
Spring Torsional Snap Rate Sensor
{c1}
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 965 in model.sensor_spring_torsional_snap_rates
    sensor = model.sensor_spring_torsional_snap_rates[965]
    assert sensor.spring_id == 1165
    assert pytest.approx(sensor.jtor_snp_max) == 9.05e6
    assert pytest.approx(sensor.jtor_lock_max) == 9.05e6
    assert pytest.approx(sensor.jtor_crackle_max) == 9.05e6
    assert pytest.approx(sensor.jtor_shot_max) == 9.05e6
    assert pytest.approx(sensor.jtor_pop_max) == 9.05e6
    assert pytest.approx(sensor.jtor_crk_max) == 9.05e6
    assert pytest.approx(sensor.jtor_drop_rate_max) == 9.05e6
    assert pytest.approx(sensor.jtor_rate_max) == 9.05e6
    assert pytest.approx(sensor.jtor_roc_rate_max) == 9.05e6
    assert pytest.approx(sensor.jtor_snap_rate_max) == 9.05e6
    assert pytest.approx(sensor.j_max) == 9.05e6
    assert pytest.approx(sensor.t_delay) == 0.0075
    assert len(model.sensors) == 1
    assert model.sensors[0].kind == "SPRING_TORSIONAL_SNAP_RATE"


def test_m433_sensor_spring_torsional_snap_rate_free(tmp_path: Path):
    deck = """# RADIOSS FREE DECK
/BEGIN
Sensor Spring Torsional Snap Rate Free Format Test
/SENSOR/SPRING_TORSIONAL_SNAP_RATE/966
1166, 9.15e6, 0.0085
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 966 in model.sensor_spring_torsional_snap_rates
    sensor = model.sensor_spring_torsional_snap_rates[966]
    assert sensor.spring_id == 1166
    assert pytest.approx(sensor.jtor_snp_max) == 9.15e6
    assert pytest.approx(sensor.t_delay) == 0.0085


def test_m433_sensor_spring_torsional_snap_rate_aliases(tmp_path: Path):
    deck = """# RADIOSS ALIAS DECK
/BEGIN
Sensor Spring Torsional Snap Rate Aliases Test
/SENSOR/SPRING_TORS_SNAP_RATE/967
1167, 8.95e6, 0.0065
/SENSOR/SPRING_SNAP_RATE_TORS/968
1168, 8.95e6, 0.0065
/SENSOR/TORSIONAL_SNAP_RATE_SPRING/969
1169, 8.95e6, 0.0065
/SENSOR/SPRING_SNAP_TORS/970
1170, 8.95e6, 0.0065
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 967 in model.sensor_spring_torsional_snap_rates
    assert 968 in model.sensor_spring_torsional_snap_rates
    assert 969 in model.sensor_spring_torsional_snap_rates
    assert 970 in model.sensor_spring_torsional_snap_rates
    assert len(model.sensors) == 4


def test_m433_sensor_spring_torsional_snap_rate_missing_card(tmp_path: Path):
    deck = """# RADIOSS MISSING CARD DECK
/BEGIN
/SENSOR/SPRING_TORSIONAL_SNAP_RATE/971
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) > 0
