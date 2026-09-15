"""Tests for Milestone M287: LadVisc Failure Model, EngPoyntingEnergy, ScissorMechanismJoint, and SensorSpringTorsionalRate."""

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


def test_m287_fail_lad_visc_fixed(tmp_path: Path):
    c1 = f"{0.055:>20.4f}{0.850:>20.4f}{1.250:>20.4f}{2.500:>20.4f}{0.350:>20.4f}"
    c2 = f"{1:>10d}{2:>10d}"
    c3 = f"{980:>10d}"
    deck = f"""# RADIOSS STARTER DECK
/BEGIN
Ladeveze Viscous Failure Model Fixed Format Test
2022 0
/FAIL/LAD_VISC/980
Ladeveze Viscoplastic Damage Criterion
{c1}
{c2}
{c3}
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 980 in model.fail_ladviscs
    flv = model.fail_ladviscs[980]
    assert pytest.approx(flv.y0) == 0.055
    assert pytest.approx(flv.yc) == 0.850
    assert pytest.approx(flv.a_lad) == 1.250
    assert pytest.approx(flv.p_visc) == 2.500
    assert pytest.approx(flv.m_visc) == 0.350
    assert flv.ifail_sh == 1
    assert flv.ifail_so == 2
    assert flv.fail_id == 980
    assert len(model.raw_fails) == 1
    assert model.raw_fails[0][1].type == "LAD_VISC"


def test_m287_fail_lad_visc_free(tmp_path: Path):
    deck = """# RADIOSS FREE DECK
/BEGIN
Ladeveze Viscous Free Format Test
/FAIL/LAD_VISC/981
0.045, 0.750, 1.150, 2.200, 0.300
1, 1
981
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 981 in model.fail_ladviscs
    flv = model.fail_ladviscs[981]
    assert pytest.approx(flv.y0) == 0.045
    assert pytest.approx(flv.m_visc) == 0.300
    assert flv.fail_id == 981


def test_m287_fail_lad_visc_aliases(tmp_path: Path):
    deck = """# RADIOSS ALIAS DECK
/BEGIN
Ladeveze Viscous Aliases Test
/FAIL/LADEVEZE_VISCOUS/982
0.05, 0.80, 1.20, 2.00, 0.32
1, 1
/FAIL/LAD_VISCOUS/983
0.05, 0.80, 1.20, 2.00, 0.32
1, 1
/FAIL/LAD_VISC_MODEL/984
0.05, 0.80, 1.20, 2.00, 0.32
1, 1
/FAIL/LAD_VISC_LAW/985
0.05, 0.80, 1.20, 2.00, 0.32
1, 1
/FAIL/LADEVEZE_VISCOPLASTIC/986
0.05, 0.80, 1.20, 2.00, 0.32
1, 1
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 982 in model.fail_ladviscs
    assert 983 in model.fail_ladviscs
    assert 984 in model.fail_ladviscs
    assert 985 in model.fail_ladviscs
    assert 986 in model.fail_ladviscs
    assert len(model.raw_fails) == 5


def test_m287_eng_poynting_energy(tmp_path: Path):
    c1 = f"{0.00065:>20.6f}{11:>10d}"
    deck = f"""# RADIOSS STARTER DECK
/BEGIN
Poynting Energy Fixed and Free Format Test
2022 0
/ENG/POYNTING_ENERGY/1
Fixed Poynting Energy Output
{c1}
/ENG/POYNTING_ENERGY/2
Free Poynting Energy Output
0.0013, 22
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 1 in model.eng_poynting_energies
    assert 2 in model.eng_poynting_energies
    pe1 = model.eng_poynting_energies[1]
    assert pytest.approx(pe1.dt_poynting) == 0.00065
    assert pe1.sens_id == 11
    pe2 = model.eng_poynting_energies[2]
    assert pytest.approx(pe2.dt_poynting) == 0.0013
    assert pe2.sens_id == 22


def test_m287_eng_poynting_energy_aliases(tmp_path: Path):
    deck = """# RADIOSS ALIAS DECK
/BEGIN
Poynting Energy Aliases Test
/ENG/POYNTING_WORK/11
0.001, 10
/ENG/EPOYNTING/12
0.002, 11
/ENG/POYNTING_VECTOR/13
0.003, 12
/ENG/EM_POYNTING_ENERGY/14
0.004, 13
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 11 in model.eng_poynting_energies
    assert 12 in model.eng_poynting_energies
    assert 13 in model.eng_poynting_energies
    assert 14 in model.eng_poynting_energies
    assert pytest.approx(model.eng_poynting_energies[11].dt_poynting) == 0.001
    assert pytest.approx(model.eng_poynting_energies[12].dt_poynting) == 0.002
    assert pytest.approx(model.eng_poynting_energies[13].dt_poynting) == 0.003
    assert pytest.approx(model.eng_poynting_energies[14].dt_poynting) == 0.004


def test_m287_scissor_mechanism_joint(tmp_path: Path):
    c1 = f"{191:>10d}{192:>10d}{193:>10d}{8.1e6:>20.4f}{1:>10d}{1.2e-6:>20.6e}"
    c2 = f"{120.0:>20.4f}{40.0:>20.4f}{1.0:>20.4f}"
    deck = f"""# RADIOSS STARTER DECK
/BEGIN
Scissor Mechanism Joint Fixed and Free Format Test
2022 0
/LAGMUL/SCISSOR_MECHANISM_JOINT/1
Fixed Scissor Mechanism Joint
{c1}
{c2}
/LAGMUL/SCISSOR_MECHANISM_JOINT/2
Free Scissor Mechanism Joint
291, 292, 293, 9.3e6, 2, 1.6e-6
150.0, 50.0, 1.0
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 1 in model.lagmul_scissor_mechanism_joints
    assert 2 in model.lagmul_scissor_mechanism_joints
    sm1 = model.lagmul_scissor_mechanism_joints[1]
    assert sm1.node1 == 191
    assert sm1.node2 == 192
    assert sm1.node3 == 193
    assert pytest.approx(sm1.stiff) == 8.1e6
    assert sm1.skew_id == 1
    assert pytest.approx(sm1.tol) == 1.2e-6
    assert pytest.approx(sm1.arm_length) == 120.0
    assert pytest.approx(sm1.initial_angle) == 40.0
    assert pytest.approx(sm1.axis_z) == 1.0

    sm2 = model.lagmul_scissor_mechanism_joints[2]
    assert sm2.node1 == 291
    assert sm2.node2 == 292
    assert sm2.node3 == 293
    assert pytest.approx(sm2.stiff) == 9.3e6
    assert sm2.skew_id == 2
    assert pytest.approx(sm2.tol) == 1.6e-6
    assert pytest.approx(sm2.arm_length) == 150.0
    assert pytest.approx(sm2.initial_angle) == 50.0
    assert pytest.approx(sm2.axis_z) == 1.0


def test_m287_scissor_mechanism_joint_aliases(tmp_path: Path):
    deck = """# RADIOSS ALIAS DECK
/BEGIN
Scissor Mechanism Joint Aliases Test
/SCISSOR_MECHANISM_JOINT/81
941, 942, 943, 1.0e6, 0, 1.0e-6
100.0, 45.0, 1.0
/LAGMUL/SCISSOR_MECHANISM/82
944, 945, 946, 1.0e6, 0, 1.0e-6
100.0, 45.0, 1.0
/SCISSOR_MECHANISM/83
947, 948, 949, 1.0e6, 0, 1.0e-6
100.0, 45.0, 1.0
/SCISSOR_JOINT/84
950, 951, 952, 1.0e6, 0, 1.0e-6
100.0, 45.0, 1.0
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 81 in model.lagmul_scissor_mechanism_joints
    assert 82 in model.lagmul_scissor_mechanism_joints
    assert 83 in model.lagmul_scissor_mechanism_joints
    assert 84 in model.lagmul_scissor_mechanism_joints
    assert model.lagmul_scissor_mechanism_joints[81].node1 == 941
    assert model.lagmul_scissor_mechanism_joints[82].node1 == 944
    assert model.lagmul_scissor_mechanism_joints[83].node1 == 947
    assert model.lagmul_scissor_mechanism_joints[84].node1 == 950


def test_m287_sensor_spring_torsional_rate(tmp_path: Path):
    c1 = f"{831:>10d}{25000.0:>20.4f}{0.0065:>20.4f}"
    deck = f"""# RADIOSS STARTER DECK
/BEGIN
Sensor Spring Torsional Rate Fixed and Free Format Test
2022 0
/SENSOR/SPRING_TORSIONAL_RATE/1
Fixed Spring Torsional Rate Sensor
{c1}
/SENSOR/SPRING_TORSIONAL_RATE/2
Free Spring Torsional Rate Sensor
832, 28000.0, 0.0085
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 1 in model.sensor_spring_torsional_rates
    assert 2 in model.sensor_spring_torsional_rates
    s1 = model.sensor_spring_torsional_rates[1]
    assert s1.spring_id == 831
    assert pytest.approx(s1.mdot_max) == 25000.0
    assert pytest.approx(s1.t_delay) == 0.0065

    s2 = model.sensor_spring_torsional_rates[2]
    assert s2.spring_id == 832
    assert pytest.approx(s2.mdot_max) == 28000.0
    assert pytest.approx(s2.t_delay) == 0.0085

    assert len(model.sensors) == 2
    assert model.sensors[0].kind == "SPRING_TORSIONAL_RATE"
    assert model.sensors[1].kind == "SPRING_TORSIONAL_RATE"


def test_m287_sensor_spring_torsional_rate_aliases(tmp_path: Path):
    deck = """# RADIOSS ALIAS DECK
/BEGIN
Sensor Spring Torsional Rate Aliases Test
/SENSOR/SPRING_TORS_RATE/41
931, 20000.0, 0.001
/SENSOR/SPRING_MDOT/42
932, 22000.0, 0.002
/SENSOR/TORSIONAL_RATE_SPRING/43
933, 24000.0, 0.003
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 41 in model.sensor_spring_torsional_rates
    assert 42 in model.sensor_spring_torsional_rates
    assert 43 in model.sensor_spring_torsional_rates
    assert model.sensor_spring_torsional_rates[41].spring_id == 931
    assert pytest.approx(model.sensor_spring_torsional_rates[41].mdot_max) == 20000.0
    assert model.sensor_spring_torsional_rates[42].spring_id == 932
    assert model.sensor_spring_torsional_rates[43].spring_id == 933
