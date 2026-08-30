"""Tests for Milestone M431: LadTransverseCoreDelaminationRate Failure Model, EngElectrothermoflexomagnetoplasmonicmagnonicpolaritonicResonanceEnergy, ContactSpatialLinkageJoint, and SensorSpringTransverseSnapRate."""

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


def test_m431_fail_lad_transverse_core_delamination_rate_fixed(tmp_path: Path):
    c1 = f"{945.0:>20.4f}{2835.0:>20.4f}{725.0:>20.4f}{14.85:>20.4f}{0.505:>20.4f}"
    c2 = f"{1:>10d}{2:>10d}"
    c3 = f"{2450:>10d}"
    deck = f"""# RADIOSS STARTER DECK
/BEGIN
Ladeveze Transverse Core Delamination Rate Fixed Format Test
2022 0
/FAIL/LAD_TRANSVERSE_CORE_DELAMINATION_RATE/2450
Ladeveze Transverse Core Delamination Failure Model
{c1}
{c2}
{c3}
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 2450 in model.fail_ladtransversecoredelaminationrates
    ftcdl = model.fail_ladtransversecoredelaminationrates[2450]
    assert pytest.approx(ftcdl.sigma_tcdl0) == 945.0
    assert pytest.approx(ftcdl.sigma_tcdlc) == 2835.0
    assert pytest.approx(ftcdl.gamma_tcdl) == 725.0
    assert pytest.approx(ftcdl.p_tcdl) == 14.85
    assert pytest.approx(ftcdl.d_tcdl_max) == 0.505
    assert ftcdl.ifail_sh == 1
    assert ftcdl.ifail_so == 2
    assert ftcdl.fail_id == 2450
    assert len(model.raw_fails) == 1
    assert model.raw_fails[0][1].type == "LAD_TRANSVERSE_CORE_DELAMINATION_RATE"


def test_m431_fail_lad_transverse_core_delamination_rate_free(tmp_path: Path):
    deck = """# RADIOSS FREE DECK
/BEGIN
Ladeveze Transverse Core Delamination Rate Free Format Test
/FAIL/LAD_TRANSVERSE_CORE_DELAMINATION_RATE/2451
955.0, 2865.0, 735.0, 15.05, 0.495
1, 1
2451
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 2451 in model.fail_ladtransversecoredelaminationrates
    ftcdl = model.fail_ladtransversecoredelaminationrates[2451]
    assert pytest.approx(ftcdl.sigma_tcdl0) == 955.0
    assert pytest.approx(ftcdl.sigma_tcdlc) == 2865.0
    assert pytest.approx(ftcdl.gamma_tcdl) == 735.0
    assert pytest.approx(ftcdl.p_tcdl) == 15.05
    assert pytest.approx(ftcdl.d_tcdl_max) == 0.495
    assert ftcdl.fail_id == 2451


def test_m431_fail_lad_transverse_core_delamination_rate_aliases(tmp_path: Path):
    deck = """# RADIOSS ALIAS DECK
/BEGIN
Ladeveze Transverse Core Delamination Rate Aliases Test
/FAIL/LADEVEZE_TRANSVERSE_CORE_DELAMINATION_RATE/2452
935.0, 2805.0, 715.0, 14.65, 0.515
1, 1
/FAIL/LAD_TCDLR/2453
935.0, 2805.0, 715.0, 14.65, 0.515
1, 1
/FAIL/LAD_TCDLR_MODEL/2454
935.0, 2805.0, 715.0, 14.65, 0.515
1, 1
/FAIL/LAD_TCDLR_LAW/2455
935.0, 2805.0, 715.0, 14.65, 0.515
1, 1
/FAIL/LADEVEZE_RATE_DEPENDENT_TRANSVERSE_CORE_DELAMINATION/2456
935.0, 2805.0, 715.0, 14.65, 0.515
1, 1
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 2452 in model.fail_ladtransversecoredelaminationrates
    assert 2453 in model.fail_ladtransversecoredelaminationrates
    assert 2454 in model.fail_ladtransversecoredelaminationrates
    assert 2455 in model.fail_ladtransversecoredelaminationrates
    assert 2456 in model.fail_ladtransversecoredelaminationrates
    assert len(model.raw_fails) == 5


def test_m431_fail_lad_transverse_core_delamination_rate_missing_card(tmp_path: Path):
    deck = """# RADIOSS MISSING CARD DECK
/BEGIN
/FAIL/LAD_TRANSVERSE_CORE_DELAMINATION_RATE/2457
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) > 0


def test_m431_eng_electrothermoflexomagnetoplasmonicmagnonicpolaritonic_resonance_energy_fixed(tmp_path: Path):
    c1 = f"{0.0855:>20.4f}{865:>10d}"
    deck = f"""# RADIOSS STARTER DECK
/BEGIN
Engine Electrothermoflexomagnetoplasmonicmagnonicpolaritonic Resonance Energy Fixed Format Test
2022 0
/ENG/ELECTROTHERMOFLEXOMAGNETOPLASMONICMAGNONICPOLARITONIC_RESONANCE_ENERGY/765
Resonance Energy Output Directive
{c1}
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 765 in model.eng_electrothermoflexomagnetoplasmonicmagnonicpolaritonic_resonance_energies
    eng = model.eng_electrothermoflexomagnetoplasmonicmagnonicpolaritonic_resonance_energies[765]
    assert pytest.approx(eng.dt_etfplmmnp) == 0.0855
    assert eng.sens_id == 865


def test_m431_eng_electrothermoflexomagnetoplasmonicmagnonicpolaritonic_resonance_energy_free(tmp_path: Path):
    deck = """# RADIOSS FREE DECK
/BEGIN
Engine Electrothermoflexomagnetoplasmonicmagnonicpolaritonic Resonance Energy Free Format Test
/ENG/ELECTROTHERMOFLEXOMAGNETOPLASMONICMAGNONICPOLARITONIC_RESONANCE_ENERGY/766
0.0865, 866
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 766 in model.eng_electrothermoflexomagnetoplasmonicmagnonicpolaritonic_resonance_energies
    eng = model.eng_electrothermoflexomagnetoplasmonicmagnonicpolaritonic_resonance_energies[766]
    assert pytest.approx(eng.dt_etfplmmnp) == 0.0865
    assert eng.sens_id == 866


def test_m431_eng_electrothermoflexomagnetoplasmonicmagnonicpolaritonic_resonance_energy_aliases(tmp_path: Path):
    deck = """# RADIOSS ALIAS DECK
/BEGIN
Engine Electrothermoflexomagnetoplasmonicmagnonicpolaritonic Resonance Energy Aliases Test
/ENG/ELECTRO_THERM_FLEXO_MAG_PLASMON_MAGNON_POLARITON_RES_WORK/767
0.0875, 867
/ENG/EELECTROTHERMOFLEXOMAGNETOPLASMONICMAGNONICPOLARITONICRESONANCE/768
0.0885, 868
/ENG/ELECTROTHERMOFLEXOMAGNETOPLASMONICMAGNONICPOLARITONIC_RESONANCE_DISSIPATION/769
0.0895, 869
/ENG/ET_ELECTROTHERMOFLEXOMAGNETOPLASMONICMAGNONICPOLARITONIC_RESONANCE/770
0.0905, 870
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 767 in model.eng_electrothermoflexomagnetoplasmonicmagnonicpolaritonic_resonance_energies
    assert 768 in model.eng_electrothermoflexomagnetoplasmonicmagnonicpolaritonic_resonance_energies
    assert 769 in model.eng_electrothermoflexomagnetoplasmonicmagnonicpolaritonic_resonance_energies
    assert 770 in model.eng_electrothermoflexomagnetoplasmonicmagnonicpolaritonic_resonance_energies


def test_m431_eng_electrothermoflexomagnetoplasmonicmagnonicpolaritonic_resonance_energy_missing_card(tmp_path: Path):
    deck = """# RADIOSS MISSING CARD DECK
/BEGIN
/ENG/ELECTROTHERMOFLEXOMAGNETOPLASMONICMAGNONICPOLARITONIC_RESONANCE_ENERGY/771
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) > 0


def test_m431_lagmul_contact_spatial_linkage_joint_fixed(tmp_path: Path):
    c1 = f"{851:>10d}{852:>10d}{853:>10d}{7.35e7:>20.4f}{455:>10d}{6.45e-4:>20.4e}"
    c2 = f"{515.0:>20.4f}{495.0:>20.4f}{525.0:>20.4f}{445.0:>20.4f}"
    deck = f"""# RADIOSS STARTER DECK
/BEGIN
Contact Spatial Linkage Joint Fixed Format Test
2022 0
/LAGMUL/CONTACT_SPATIAL_LINKAGE_JOINT/815
Contact Spatial 6R Joint
{c1}
{c2}
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 815 in model.lagmul_contact_spatial_linkage_joints
    joint = model.lagmul_contact_spatial_linkage_joints[815]
    assert joint.node1 == 851
    assert joint.node2 == 852
    assert joint.node3 == 853
    assert pytest.approx(joint.stiff) == 7.35e7
    assert joint.skew_id == 455
    assert pytest.approx(joint.tol) == 6.45e-4
    assert pytest.approx(joint.link_len_a) == 515.0
    assert pytest.approx(joint.link_len_b) == 495.0
    assert pytest.approx(joint.twist_angle_alpha) == 525.0
    assert pytest.approx(joint.offset_distance_s) == 445.0
    assert pytest.approx(joint.offset_distance_r) == 445.0
    assert pytest.approx(joint.offset_distance_v) == 445.0
    assert pytest.approx(joint.offset_distance_h) == 445.0
    assert pytest.approx(joint.offset_distance_u) == 445.0
    assert pytest.approx(joint.offset_distance_f) == 445.0


def test_m431_lagmul_contact_spatial_linkage_joint_free(tmp_path: Path):
    deck = """# RADIOSS FREE DECK
/BEGIN
Contact Spatial Linkage Joint Free Format Test
/LAGMUL/CONTACT_SPATIAL_LINKAGE_JOINT/816
951, 952, 953, 7.55e7, 456, 6.55e-4
535.0, 505.0, 545.0, 460.0
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 816 in model.lagmul_contact_spatial_linkage_joints
    joint = model.lagmul_contact_spatial_linkage_joints[816]
    assert joint.node1 == 951
    assert joint.node2 == 952
    assert joint.node3 == 953
    assert pytest.approx(joint.stiff) == 7.55e7
    assert joint.skew_id == 456
    assert pytest.approx(joint.tol) == 6.55e-4
    assert pytest.approx(joint.link_len_a) == 535.0
    assert pytest.approx(joint.link_len_b) == 505.0
    assert pytest.approx(joint.twist_angle_alpha) == 545.0
    assert pytest.approx(joint.offset_distance_s) == 460.0


def test_m431_lagmul_contact_spatial_linkage_joint_aliases(tmp_path: Path):
    deck = """# RADIOSS ALIAS DECK
/BEGIN
Contact Spatial Linkage Joint Aliases Test
/CONTACT_SPATIAL_LINKAGE_JOINT/817
1051, 1052, 1053, 1.0e6, 0, 1.0e-6
475.0, 465.0, 475.0, 425.0
/LAGMUL/CONTACT_SPATIAL_LINKAGE/818
1051, 1052, 1053, 1.0e6, 0, 1.0e-6
475.0, 465.0, 475.0, 425.0
/CONTACT_SPATIAL_LINKAGE/819
1051, 1052, 1053, 1.0e6, 0, 1.0e-6
475.0, 465.0, 475.0, 425.0
/CONTACT_SPATIAL_MULTI_LOOP_MECHANISM/820
1051, 1052, 1053, 1.0e6, 0, 1.0e-6
475.0, 465.0, 475.0, 425.0
/CONTACT_SPATIAL_SYMMETRIC_MECHANISM/821
1051, 1052, 1053, 1.0e6, 0, 1.0e-6
475.0, 465.0, 475.0, 425.0
/CONTACT_SPATIAL_6R_MECHANISM/822
1051, 1052, 1053, 1.0e6, 0, 1.0e-6
475.0, 465.0, 475.0, 425.0
/CONTACT_SPATIAL_OVERCONSTRAINED_MECHANISM/823
1051, 1052, 1053, 1.0e6, 0, 1.0e-6
475.0, 465.0, 475.0, 425.0
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 817 in model.lagmul_contact_spatial_linkage_joints
    assert 818 in model.lagmul_contact_spatial_linkage_joints
    assert 819 in model.lagmul_contact_spatial_linkage_joints
    assert 820 in model.lagmul_contact_spatial_linkage_joints
    assert 821 in model.lagmul_contact_spatial_linkage_joints
    assert 822 in model.lagmul_contact_spatial_linkage_joints
    assert 823 in model.lagmul_contact_spatial_linkage_joints


def test_m431_lagmul_contact_spatial_linkage_joint_missing_card(tmp_path: Path):
    deck = """# RADIOSS MISSING CARD DECK
/BEGIN
/LAGMUL/CONTACT_SPATIAL_LINKAGE_JOINT/824
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) > 0


def test_m431_sensor_spring_transverse_snap_rate_fixed(tmp_path: Path):
    c1 = f"{1145:>10d}{8.85e6:>20.4f}{0.0055:>20.4f}"
    deck = f"""# RADIOSS STARTER DECK
/BEGIN
Sensor Spring Transverse Snap Rate Fixed Format Test
2022 0
/SENSOR/SPRING_TRANSVERSE_SNAP_RATE/945
Spring Transverse Snap Rate Sensor
{c1}
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 945 in model.sensor_spring_transverse_snap_rates
    sensor = model.sensor_spring_transverse_snap_rates[945]
    assert sensor.spring_id == 1145
    assert pytest.approx(sensor.jtr_snp_max) == 8.85e6
    assert pytest.approx(sensor.jtr_lock_max) == 8.85e6
    assert pytest.approx(sensor.jtr_crackle_max) == 8.85e6
    assert pytest.approx(sensor.jtr_shot_max) == 8.85e6
    assert pytest.approx(sensor.jtr_pop_max) == 8.85e6
    assert pytest.approx(sensor.jtr_crk_max) == 8.85e6
    assert pytest.approx(sensor.jtr_drop_rate_max) == 8.85e6
    assert pytest.approx(sensor.jtr_rate_max) == 8.85e6
    assert pytest.approx(sensor.jtr_roc_rate_max) == 8.85e6
    assert pytest.approx(sensor.jtr_snap_rate_max) == 8.85e6
    assert pytest.approx(sensor.t_delay) == 0.0055
    assert len(model.sensors) == 1
    assert model.sensors[0].kind == "SPRING_TRANSVERSE_SNAP_RATE"


def test_m431_sensor_spring_transverse_snap_rate_free(tmp_path: Path):
    deck = """# RADIOSS FREE DECK
/BEGIN
Sensor Spring Transverse Snap Rate Free Format Test
/SENSOR/SPRING_TRANSVERSE_SNAP_RATE/946
1146, 8.95e6, 0.0065
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 946 in model.sensor_spring_transverse_snap_rates
    sensor = model.sensor_spring_transverse_snap_rates[946]
    assert sensor.spring_id == 1146
    assert pytest.approx(sensor.jtr_snp_max) == 8.95e6
    assert pytest.approx(sensor.t_delay) == 0.0065


def test_m431_sensor_spring_transverse_snap_rate_aliases(tmp_path: Path):
    deck = """# RADIOSS ALIAS DECK
/BEGIN
Sensor Spring Transverse Snap Rate Aliases Test
/SENSOR/SPRING_TRANS_SNAP_RATE/947
1147, 8.75e6, 0.0045
/SENSOR/SPRING_SNAP_RATE_TRANS/948
1148, 8.75e6, 0.0045
/SENSOR/TRANSVERSE_SNAP_RATE_SPRING/949
1149, 8.75e6, 0.0045
/SENSOR/SPRING_SNAP_TRANS/950
1150, 8.75e6, 0.0045
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 947 in model.sensor_spring_transverse_snap_rates
    assert 948 in model.sensor_spring_transverse_snap_rates
    assert 949 in model.sensor_spring_transverse_snap_rates
    assert 950 in model.sensor_spring_transverse_snap_rates
    assert len(model.sensors) == 4


def test_m431_sensor_spring_transverse_snap_rate_missing_card(tmp_path: Path):
    deck = """# RADIOSS MISSING CARD DECK
/BEGIN
/SENSOR/SPRING_TRANSVERSE_SNAP_RATE/951
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) > 0
