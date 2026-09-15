"""Tests for Milestone M301: LadCoupleDamage Failure Model, EngThermionicEnergy, JansenLinkageJoint, and SensorSpringNormalAccelerationRate."""

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


def test_m301_fail_lad_couple_damage_fixed(tmp_path: Path):
    c1 = f"{1.8:>20.4f}{21.0:>20.4f}{1.2e-4:>20.6e}{0.85:>20.4f}{0.975:>20.4f}"
    c2 = f"{1:>10d}{2:>10d}"
    c3 = f"{1105:>10d}"
    deck = f"""# RADIOSS STARTER DECK
/BEGIN
Ladeveze Coupled Damage Model Fixed Format Test
2022 0
/FAIL/LAD_COUPLE_DAMAGE/1105
Ladeveze Coupled Damage Criterion
{c1}
{c2}
{c3}
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 1105 in model.fail_ladcoupledamages
    flcd = model.fail_ladcoupledamages[1105]
    assert pytest.approx(flcd.y0_cd) == 1.8
    assert pytest.approx(flcd.yc_cd) == 21.0
    assert pytest.approx(flcd.gamma_temp) == 1.2e-4
    assert pytest.approx(flcd.eta_entropy) == 0.85
    assert pytest.approx(flcd.d_cd_max) == 0.975
    assert flcd.ifail_sh == 1
    assert flcd.ifail_so == 2
    assert flcd.fail_id == 1105
    assert len(model.raw_fails) == 1
    assert model.raw_fails[0][1].type == "LAD_COUPLE_DAMAGE"


def test_m301_fail_lad_couple_damage_free(tmp_path: Path):
    deck = """# RADIOSS FREE DECK
/BEGIN
Ladeveze Coupled Damage Free Format Test
/FAIL/LAD_COUPLE_DAMAGE/1106
2.5, 26.0, 1.5e-4, 0.90, 0.950
1, 1
1106
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 1106 in model.fail_ladcoupledamages
    flcd = model.fail_ladcoupledamages[1106]
    assert pytest.approx(flcd.y0_cd) == 2.5
    assert pytest.approx(flcd.yc_cd) == 26.0
    assert pytest.approx(flcd.gamma_temp) == 1.5e-4
    assert pytest.approx(flcd.eta_entropy) == 0.90
    assert pytest.approx(flcd.d_cd_max) == 0.950
    assert flcd.fail_id == 1106


def test_m301_fail_lad_couple_damage_aliases(tmp_path: Path):
    deck = """# RADIOSS ALIAS DECK
/BEGIN
Ladeveze Coupled Damage Aliases Test
/FAIL/LADEVEZE_COUPLED_DAMAGE/1107
1.5, 18.0, 1.0e-4, 0.80, 0.96
1, 1
/FAIL/LAD_CD/1108
1.5, 18.0, 1.0e-4, 0.80, 0.96
1, 1
/FAIL/LAD_COUPLE_DAMAGE_MODEL/1109
1.5, 18.0, 1.0e-4, 0.80, 0.96
1, 1
/FAIL/LAD_COUPLE_DAMAGE_LAW/1110
1.5, 18.0, 1.0e-4, 0.80, 0.96
1, 1
/FAIL/LADEVEZE_THERMO_ELASTO_DAMAGE/1111
1.5, 18.0, 1.0e-4, 0.80, 0.96
1, 1
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 1107 in model.fail_ladcoupledamages
    assert 1108 in model.fail_ladcoupledamages
    assert 1109 in model.fail_ladcoupledamages
    assert 1110 in model.fail_ladcoupledamages
    assert 1111 in model.fail_ladcoupledamages
    assert len(model.raw_fails) == 5


def test_m301_eng_thermionic_energy(tmp_path: Path):
    c1 = f"{0.00012:>20.6f}{32:>10d}"
    deck = f"""# RADIOSS STARTER DECK
/BEGIN
Thermionic Energy Fixed and Free Format Test
2022 0
/ENG/THERMIONIC_ENERGY/1
Fixed Thermionic Energy Output
{c1}
/ENG/THERMIONIC_ENERGY/2
Free Thermionic Energy Output
0.00024, 64
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 1 in model.eng_thermionic_energies
    assert 2 in model.eng_thermionic_energies
    tie1 = model.eng_thermionic_energies[1]
    assert pytest.approx(tie1.dt_thermionic) == 0.00012
    assert tie1.sens_id == 32
    tie2 = model.eng_thermionic_energies[2]
    assert pytest.approx(tie2.dt_thermionic) == 0.00024
    assert tie2.sens_id == 64


def test_m301_eng_thermionic_energy_aliases(tmp_path: Path):
    deck = """# RADIOSS ALIAS DECK
/BEGIN
Thermionic Energy Aliases Test
/ENG/TI_WORK/61
0.001, 60
/ENG/ETHERMIONIC/62
0.002, 61
/ENG/THERMIONIC_DISSIPATION/63
0.003, 62
/ENG/EM_THERMIONIC/64
0.004, 63
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 61 in model.eng_thermionic_energies
    assert 62 in model.eng_thermionic_energies
    assert 63 in model.eng_thermionic_energies
    assert 64 in model.eng_thermionic_energies
    assert pytest.approx(model.eng_thermionic_energies[61].dt_thermionic) == 0.001
    assert pytest.approx(model.eng_thermionic_energies[62].dt_thermionic) == 0.002
    assert pytest.approx(model.eng_thermionic_energies[63].dt_thermionic) == 0.003
    assert pytest.approx(model.eng_thermionic_energies[64].dt_thermionic) == 0.004


def test_m301_jansen_linkage_joint(tmp_path: Path):
    c1 = f"{881:>10d}{882:>10d}{883:>10d}{8.2e6:>20.4f}{1:>10d}{1.0e-6:>20.6e}"
    c2 = f"{15.0:>20.4f}{38.0:>20.4f}{7.8:>20.4f}{1.25:>20.4f}"
    deck = f"""# RADIOSS STARTER DECK
/BEGIN
Jansen Linkage Joint Fixed and Free Format Test
2022 0
/LAGMUL/JANSEN_LINKAGE_JOINT/1
Fixed Jansen Linkage Joint
{c1}
{c2}
/LAGMUL/JANSEN_LINKAGE_JOINT/2
Free Jansen Linkage Joint
981, 982, 983, 9.0e6, 2, 1.2e-6
18.0, 42.0, 8.5, 1.30
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 1 in model.lagmul_jansen_linkage_joints
    assert 2 in model.lagmul_jansen_linkage_joints
    jj1 = model.lagmul_jansen_linkage_joints[1]
    assert jj1.node1 == 881
    assert jj1.node2 == 882
    assert jj1.node3 == 883
    assert pytest.approx(jj1.stiff) == 8.2e6
    assert jj1.skew_id == 1
    assert pytest.approx(jj1.tol) == 1.0e-6
    assert pytest.approx(jj1.crank_len) == 15.0
    assert pytest.approx(jj1.base_horizontal) == 38.0
    assert pytest.approx(jj1.base_vertical) == 7.8
    assert pytest.approx(jj1.leg_ratio) == 1.25

    jj2 = model.lagmul_jansen_linkage_joints[2]
    assert jj2.node1 == 981
    assert jj2.node2 == 982
    assert jj2.node3 == 983
    assert pytest.approx(jj2.stiff) == 9.0e6
    assert jj2.skew_id == 2
    assert pytest.approx(jj2.tol) == 1.2e-6
    assert pytest.approx(jj2.crank_len) == 18.0
    assert pytest.approx(jj2.base_horizontal) == 42.0
    assert pytest.approx(jj2.base_vertical) == 8.5
    assert pytest.approx(jj2.leg_ratio) == 1.30


def test_m301_jansen_linkage_joint_aliases(tmp_path: Path):
    deck = """# RADIOSS ALIAS DECK
/BEGIN
Jansen Linkage Joint Aliases Test
/JANSEN_LINKAGE_JOINT/71
991, 992, 993, 6.0e6, 0, 1.0e-6
15.0, 38.0, 7.8, 1.25
/LAGMUL/JANSEN_LINKAGE/72
994, 995, 996, 6.0e6, 0, 1.0e-6
15.0, 38.0, 7.8, 1.25
/JANSEN_LINKAGE/73
997, 998, 999, 6.0e6, 0, 1.0e-6
15.0, 38.0, 7.8, 1.25
/STRANDBEEST_MECHANISM/74
1001, 1002, 1003, 6.0e6, 0, 1.0e-6
15.0, 38.0, 7.8, 1.25
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 71 in model.lagmul_jansen_linkage_joints
    assert 72 in model.lagmul_jansen_linkage_joints
    assert 73 in model.lagmul_jansen_linkage_joints
    assert 74 in model.lagmul_jansen_linkage_joints
    assert model.lagmul_jansen_linkage_joints[71].node1 == 991
    assert model.lagmul_jansen_linkage_joints[72].node1 == 994
    assert model.lagmul_jansen_linkage_joints[73].node1 == 997
    assert model.lagmul_jansen_linkage_joints[74].node1 == 1001


def test_m301_sensor_spring_normal_acceleration_rate(tmp_path: Path):
    c1 = f"{968:>10d}{2.8e7:>20.4f}{0.0080:>20.4f}"
    deck = f"""# RADIOSS STARTER DECK
/BEGIN
Sensor Spring Normal Acceleration Rate Fixed and Free Format Test
2022 0
/SENSOR/SPRING_NORMAL_ACCELERATION_RATE/1
Fixed Spring Normal Acceleration Rate Sensor
{c1}
/SENSOR/SPRING_NORMAL_ACCELERATION_RATE/2
Free Spring Normal Acceleration Rate Sensor
969, 4.6e7, 0.0105
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 1 in model.sensor_spring_normal_acceleration_rates
    assert 2 in model.sensor_spring_normal_acceleration_rates
    s1 = model.sensor_spring_normal_acceleration_rates[1]
    assert s1.spring_id == 968
    assert pytest.approx(s1.jnorm_rate_max) == 2.8e7
    assert pytest.approx(s1.t_delay) == 0.0080

    s2 = model.sensor_spring_normal_acceleration_rates[2]
    assert s2.spring_id == 969
    assert pytest.approx(s2.jnorm_rate_max) == 4.6e7
    assert pytest.approx(s2.t_delay) == 0.0105

    assert len(model.sensors) == 2
    assert model.sensors[0].kind == "SPRING_NORMAL_ACCELERATION_RATE"
    assert model.sensors[1].kind == "SPRING_NORMAL_ACCELERATION_RATE"


def test_m301_sensor_spring_normal_acceleration_rate_aliases(tmp_path: Path):
    deck = """# RADIOSS ALIAS DECK
/BEGIN
Sensor Spring Normal Acceleration Rate Aliases Test
/SENSOR/SPRING_NORM_ACC_RATE/95
998, 2.1e7, 0.0028
/SENSOR/SPRING_RATE_ACC_NORM/96
999, 2.3e7, 0.0038
/SENSOR/NORMAL_ACCELERATION_RATE_SPRING/97
1000, 2.5e7, 0.0048
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 95 in model.sensor_spring_normal_acceleration_rates
    assert 96 in model.sensor_spring_normal_acceleration_rates
    assert 97 in model.sensor_spring_normal_acceleration_rates
    assert model.sensor_spring_normal_acceleration_rates[95].spring_id == 998
    assert pytest.approx(model.sensor_spring_normal_acceleration_rates[95].jnorm_rate_max) == 2.1e7
    assert model.sensor_spring_normal_acceleration_rates[96].spring_id == 999
    assert model.sensor_spring_normal_acceleration_rates[97].spring_id == 1000
