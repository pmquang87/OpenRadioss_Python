"""Tests for Milestone M329: LadMicroDelaminationRate Failure Model, EngFlexothermalResonanceEnergy, AltmannLinkageJoint, and SensorSpringBendingPopRate."""

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


def test_m329_fail_lad_micro_delamination_rate_fixed(tmp_path: Path):
    c1 = f"{0.120:>20.4f}{1.450:>20.4f}{18.0:>20.2f}{1.25:>20.4f}{0.985:>20.4f}"
    c2 = f"{1:>10d}{2:>10d}"
    c3 = f"{1430:>10d}"
    deck = f"""# RADIOSS STARTER DECK
/BEGIN
Ladeveze Micro-Delamination Rate Fixed Format Test
2022 0
/FAIL/LAD_MICRO_DELAMINATION_RATE/1430
Ladeveze Rate Dependent Micro-Delamination Failure Model
{c1}
{c2}
{c3}
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 1430 in model.fail_ladmicrodelaminationrates
    fmdr = model.fail_ladmicrodelaminationrates[1430]
    assert pytest.approx(fmdr.y_del0) == 0.120
    assert pytest.approx(fmdr.y_delc) == 1.450
    assert pytest.approx(fmdr.gamma_del) == 18.0
    assert pytest.approx(fmdr.p_del) == 1.25
    assert pytest.approx(fmdr.d_mdr_max) == 0.985
    assert fmdr.ifail_sh == 1
    assert fmdr.ifail_so == 2
    assert fmdr.fail_id == 1430
    assert len(model.raw_fails) == 1
    assert model.raw_fails[0][1].type == "LAD_MICRO_DELAMINATION_RATE"


def test_m329_fail_lad_micro_delamination_rate_free(tmp_path: Path):
    deck = """# RADIOSS FREE DECK
/BEGIN
Ladeveze Micro-Delamination Rate Free Format Test
/FAIL/LAD_MICRO_DELAMINATION_RATE/1431
0.150, 1.800, 22.0, 1.40, 0.965
1, 1
1431
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 1431 in model.fail_ladmicrodelaminationrates
    fmdr = model.fail_ladmicrodelaminationrates[1431]
    assert pytest.approx(fmdr.y_del0) == 0.150
    assert pytest.approx(fmdr.y_delc) == 1.800
    assert pytest.approx(fmdr.gamma_del) == 22.0
    assert pytest.approx(fmdr.p_del) == 1.40
    assert pytest.approx(fmdr.d_mdr_max) == 0.965
    assert fmdr.fail_id == 1431


def test_m329_fail_lad_micro_delamination_rate_aliases(tmp_path: Path):
    deck = """# RADIOSS ALIAS DECK
/BEGIN
Ladeveze Micro-Delamination Rate Aliases Test
/FAIL/LADEVEZE_MICRO_DELAMINATION_RATE/1432
0.10, 1.20, 15.0, 1.0, 0.95
1, 1
/FAIL/LAD_MDR/1433
0.10, 1.20, 15.0, 1.0, 0.95
1, 1
/FAIL/LAD_MDR_MODEL/1434
0.10, 1.20, 15.0, 1.0, 0.95
1, 1
/FAIL/LAD_MDR_LAW/1435
0.10, 1.20, 15.0, 1.0, 0.95
1, 1
/FAIL/LADEVEZE_RATE_DEPENDENT_MICRO_DELAMINATION/1436
0.10, 1.20, 15.0, 1.0, 0.95
1, 1
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 1432 in model.fail_ladmicrodelaminationrates
    assert 1433 in model.fail_ladmicrodelaminationrates
    assert 1434 in model.fail_ladmicrodelaminationrates
    assert 1435 in model.fail_ladmicrodelaminationrates
    assert 1436 in model.fail_ladmicrodelaminationrates
    assert len(model.raw_fails) == 5


def test_m329_fail_lad_micro_delamination_rate_missing_card(tmp_path: Path):
    deck = """# RADIOSS EMPTY CARD DECK
/BEGIN
Missing Card Test
/FAIL/LAD_MICRO_DELAMINATION_RATE/1437
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) > 0


def test_m329_eng_flexothermal_resonance_energy(tmp_path: Path):
    c1 = f"{0.00085:>20.6f}{135:>10d}"
    deck = f"""# RADIOSS STARTER DECK
/BEGIN
Flexothermal Resonance Energy Fixed and Free Format Test
2022 0
/ENG/FLEXOTHERMAL_RESONANCE_ENERGY/1
Fixed Flexothermal Resonance Energy Output
{c1}
/ENG/FLEXOTHERMAL_RESONANCE_ENERGY/2
Free Flexothermal Resonance Energy Output
0.00125, 270
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 1 in model.eng_flexothermal_resonance_energies
    assert 2 in model.eng_flexothermal_resonance_energies
    ftre1 = model.eng_flexothermal_resonance_energies[1]
    assert pytest.approx(ftre1.dt_ftr) == 0.00085
    assert ftre1.sens_id == 135
    ftre2 = model.eng_flexothermal_resonance_energies[2]
    assert pytest.approx(ftre2.dt_ftr) == 0.00125
    assert ftre2.sens_id == 270


def test_m329_eng_flexothermal_resonance_energy_aliases(tmp_path: Path):
    deck = """# RADIOSS ALIAS DECK
/BEGIN
Flexothermal Resonance Energy Aliases Test
/ENG/FLEXOTHERM_RES_WORK/3
0.00025, 72
/ENG/EFLEXOTHERMORESONANCE/4
0.00030, 82
/ENG/FLEXOTHERMAL_RESONANCE_DISSIPATION/5
0.00035, 92
/ENG/EM_FLEXOTHERMAL_RESONANCE/6
0.00040, 102
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 3 in model.eng_flexothermal_resonance_energies
    assert 4 in model.eng_flexothermal_resonance_energies
    assert 5 in model.eng_flexothermal_resonance_energies
    assert 6 in model.eng_flexothermal_resonance_energies


def test_m329_eng_flexothermal_resonance_energy_missing_card(tmp_path: Path):
    deck = """# RADIOSS EMPTY CARD DECK
/BEGIN
Missing Card Test
/ENG/FLEXOTHERMAL_RESONANCE_ENERGY/7
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) > 0


def test_m329_lagmul_altmann_linkage_joint_fixed(tmp_path: Path):
    c1 = f"{151:>10d}{152:>10d}{153:>10d}{1.05e7:>20.1f}{32:>10d}{2.5e-5:>20.6e}"
    c2 = f"{76.0:>20.4f}{70.5:>20.4f}{62.0:>20.4f}{22.0:>20.4f}"
    deck = f"""# RADIOSS STARTER DECK
/BEGIN
Altmann Linkage Joint Fixed Format Test
2022 0
/ALTMANN_LINKAGE_JOINT/115
Altmann Linkage Kinematic Joint
{c1}
{c2}
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 115 in model.lagmul_altmann_linkage_joints
    joint = model.lagmul_altmann_linkage_joints[115]
    assert joint.node1 == 151
    assert joint.node2 == 152
    assert joint.node3 == 153
    assert pytest.approx(joint.stiff) == 1.05e7
    assert joint.skew_id == 32
    assert pytest.approx(joint.tol) == 2.5e-5
    assert pytest.approx(joint.link_len_a) == 76.0
    assert pytest.approx(joint.link_len_b) == 70.5
    assert pytest.approx(joint.twist_angle_alpha) == 62.0
    assert pytest.approx(joint.offset_distance_d) == 22.0


def test_m329_lagmul_altmann_linkage_joint_free(tmp_path: Path):
    deck = """# RADIOSS FREE DECK
/BEGIN
Altmann Linkage Joint Free Format Test
/LAGMUL/ALTMANN_LINKAGE_JOINT/116
251, 252, 253, 7.5e6, 50, 4.0e-5
80.0, 74.0, 66.0, 24.0
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 116 in model.lagmul_altmann_linkage_joints
    joint = model.lagmul_altmann_linkage_joints[116]
    assert joint.node1 == 251
    assert joint.node2 == 252
    assert joint.node3 == 253
    assert pytest.approx(joint.stiff) == 7.5e6
    assert joint.skew_id == 50
    assert pytest.approx(joint.tol) == 4.0e-5
    assert pytest.approx(joint.link_len_a) == 80.0
    assert pytest.approx(joint.link_len_b) == 74.0
    assert pytest.approx(joint.twist_angle_alpha) == 66.0
    assert pytest.approx(joint.offset_distance_d) == 24.0


def test_m329_lagmul_altmann_linkage_joint_aliases(tmp_path: Path):
    deck = """# RADIOSS ALIAS DECK
/BEGIN
Altmann Linkage Joint Aliases Test
/LAGMUL/ALTMANN_LINKAGE/117
1, 2, 3, 1e6, 0, 1e-6
10.0, 10.0, 30.0, 5.0
/ALTMANN_LINKAGE/118
1, 2, 3, 1e6, 0, 1e-6
10.0, 10.0, 30.0, 5.0
/ALTMANN_SPATIAL_MECHANISM/119
1, 2, 3, 1e6, 0, 1e-6
10.0, 10.0, 30.0, 5.0
/ALTMANN_6R_MECHANISM/120
1, 2, 3, 1e6, 0, 1e-6
10.0, 10.0, 30.0, 5.0
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 117 in model.lagmul_altmann_linkage_joints
    assert 118 in model.lagmul_altmann_linkage_joints
    assert 119 in model.lagmul_altmann_linkage_joints
    assert 120 in model.lagmul_altmann_linkage_joints


def test_m329_lagmul_altmann_linkage_joint_missing_card(tmp_path: Path):
    deck = """# RADIOSS EMPTY CARD DECK
/BEGIN
Missing Card Test
/ALTMANN_LINKAGE_JOINT/121
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) > 0


def test_m329_sensor_spring_bending_pop_rate_fixed(tmp_path: Path):
    c1 = f"{871:>10d}{1.65e9:>20.1f}{0.0060:>20.4f}"
    deck = f"""# RADIOSS STARTER DECK
/BEGIN
Spring Bending Pop Rate Sensor Fixed Format Test
2022 0
/SENSOR/SPRING_BENDING_POP_RATE/148
Spring Bending Pop Rate Sensor
{c1}
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 148 in model.sensor_spring_bending_pop_rates
    sensor = model.sensor_spring_bending_pop_rates[148]
    assert sensor.spring_id == 871
    assert pytest.approx(sensor.jbend_pop_max) == 1.65e9
    assert pytest.approx(sensor.t_delay) == 0.0060
    assert len(model.sensors) == 1
    assert model.sensors[0].kind == "SPRING_BENDING_POP_RATE"


def test_m329_sensor_spring_bending_pop_rate_free(tmp_path: Path):
    deck = """# RADIOSS FREE DECK
/BEGIN
Spring Bending Pop Rate Sensor Free Format Test
/SENSOR/SPRING_BENDING_POP_RATE/149
872, 1.75e9, 0.0080
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 149 in model.sensor_spring_bending_pop_rates
    sensor = model.sensor_spring_bending_pop_rates[149]
    assert sensor.spring_id == 872
    assert pytest.approx(sensor.jbend_pop_max) == 1.75e9
    assert pytest.approx(sensor.t_delay) == 0.0080


def test_m329_sensor_spring_bending_pop_rate_aliases(tmp_path: Path):
    deck = """# RADIOSS ALIAS DECK
/BEGIN
Spring Bending Pop Rate Sensor Aliases Test
/SENSOR/SPRING_BEND_POP_RATE/150
873, 2.2e9, 0.001
/SENSOR/SPRING_RATE_POP_BEND/151
874, 2.2e9, 0.001
/SENSOR/BENDING_POP_RATE_SPRING/152
875, 2.2e9, 0.001
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 150 in model.sensor_spring_bending_pop_rates
    assert 151 in model.sensor_spring_bending_pop_rates
    assert 152 in model.sensor_spring_bending_pop_rates
    assert len(model.sensors) == 3


def test_m329_sensor_spring_bending_pop_rate_missing_card(tmp_path: Path):
    deck = """# RADIOSS EMPTY CARD DECK
/BEGIN
Missing Card Test
/SENSOR/SPRING_BENDING_POP_RATE/153
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) > 0
