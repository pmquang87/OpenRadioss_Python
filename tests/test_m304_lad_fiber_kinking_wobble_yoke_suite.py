"""Tests for Milestone M304: LadFiberKinking Failure Model, EngPhotomagneticEnergy, WobbleYokeJoint, and SensorSpringBendingAccelerationRate."""

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


def test_m304_fail_lad_fiber_kinking_fixed(tmp_path: Path):
    c1 = f"{450.0:>20.4f}{0.052:>20.4f}{38.5:>20.4f}{0.0035:>20.6f}{0.985:>20.4f}"
    c2 = f"{1:>10d}{2:>10d}"
    c3 = f"{1135:>10d}"
    deck = f"""# RADIOSS STARTER DECK
/BEGIN
Ladeveze Fiber Kinking Model Fixed Format Test
2022 0
/FAIL/LAD_FIBER_KINKING/1135
Ladeveze Compressive Microbuckling Criterion
{c1}
{c2}
{c3}
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 1135 in model.fail_ladfiberkinkings
    flfk = model.fail_ladfiberkinkings[1135]
    assert pytest.approx(flfk.sigma_kink_crit) == 450.0
    assert pytest.approx(flfk.phi_kink_0) == 0.052
    assert pytest.approx(flfk.gamma_kink) == 38.5
    assert pytest.approx(flfk.l_kink_band) == 0.0035
    assert pytest.approx(flfk.d_kink_max) == 0.985
    assert flfk.ifail_sh == 1
    assert flfk.ifail_so == 2
    assert flfk.fail_id == 1135
    assert len(model.raw_fails) == 1
    assert model.raw_fails[0][1].type == "LAD_FIBER_KINKING"


def test_m304_fail_lad_fiber_kinking_free(tmp_path: Path):
    deck = """# RADIOSS FREE DECK
/BEGIN
Ladeveze Fiber Kinking Free Format Test
/FAIL/LAD_FIBER_KINKING/1136
520.0, 0.065, 45.0, 0.0045, 0.960
1, 1
1136
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 1136 in model.fail_ladfiberkinkings
    flfk = model.fail_ladfiberkinkings[1136]
    assert pytest.approx(flfk.sigma_kink_crit) == 520.0
    assert pytest.approx(flfk.phi_kink_0) == 0.065
    assert pytest.approx(flfk.gamma_kink) == 45.0
    assert pytest.approx(flfk.l_kink_band) == 0.0045
    assert pytest.approx(flfk.d_kink_max) == 0.960
    assert flfk.fail_id == 1136


def test_m304_fail_lad_fiber_kinking_aliases(tmp_path: Path):
    deck = """# RADIOSS ALIAS DECK
/BEGIN
Ladeveze Fiber Kinking Aliases Test
/FAIL/LADEVEZE_FIBER_KINK/1137
380.0, 0.045, 32.0, 0.0028, 0.97
1, 1
/FAIL/LAD_FK/1138
380.0, 0.045, 32.0, 0.0028, 0.97
1, 1
/FAIL/LAD_FK_MODEL/1139
380.0, 0.045, 32.0, 0.0028, 0.97
1, 1
/FAIL/LAD_FK_LAW/1140
380.0, 0.045, 32.0, 0.0028, 0.97
1, 1
/FAIL/LADEVEZE_MICROBUCKLING/1141
380.0, 0.045, 32.0, 0.0028, 0.97
1, 1
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 1137 in model.fail_ladfiberkinkings
    assert 1138 in model.fail_ladfiberkinkings
    assert 1139 in model.fail_ladfiberkinkings
    assert 1140 in model.fail_ladfiberkinkings
    assert 1141 in model.fail_ladfiberkinkings
    assert len(model.raw_fails) == 5


def test_m304_eng_photomagnetic_energy(tmp_path: Path):
    c1 = f"{0.00012:>20.6f}{32:>10d}"
    deck = f"""# RADIOSS STARTER DECK
/BEGIN
Photomagnetic Energy Fixed and Free Format Test
2022 0
/ENG/PHOTOMAGNETIC_ENERGY/1
Fixed Photomagnetic Energy Output
{c1}
/ENG/PHOTOMAGNETIC_ENERGY/2
Free Photomagnetic Energy Output
0.00024, 64
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 1 in model.eng_photomagnetic_energies
    assert 2 in model.eng_photomagnetic_energies
    pme1 = model.eng_photomagnetic_energies[1]
    assert pytest.approx(pme1.dt_photomagnetic) == 0.00012
    assert pme1.sens_id == 32
    pme2 = model.eng_photomagnetic_energies[2]
    assert pytest.approx(pme2.dt_photomagnetic) == 0.00024
    assert pme2.sens_id == 64


def test_m304_eng_photomagnetic_energy_aliases(tmp_path: Path):
    deck = """# RADIOSS ALIAS DECK
/BEGIN
Photomagnetic Energy Aliases Test
/ENG/PM_WORK/91
0.001, 90
/ENG/EPHOTOMAGNETIC/92
0.002, 91
/ENG/PHOTOMAGNETIC_DISSIPATION/93
0.003, 92
/ENG/EM_PHOTOMAGNETIC/94
0.004, 93
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 91 in model.eng_photomagnetic_energies
    assert 92 in model.eng_photomagnetic_energies
    assert 93 in model.eng_photomagnetic_energies
    assert 94 in model.eng_photomagnetic_energies
    assert pytest.approx(model.eng_photomagnetic_energies[91].dt_photomagnetic) == 0.001
    assert pytest.approx(model.eng_photomagnetic_energies[92].dt_photomagnetic) == 0.002
    assert pytest.approx(model.eng_photomagnetic_energies[93].dt_photomagnetic) == 0.003
    assert pytest.approx(model.eng_photomagnetic_energies[94].dt_photomagnetic) == 0.004


def test_m304_wobble_yoke_joint(tmp_path: Path):
    c1 = f"{911:>10d}{912:>10d}{913:>10d}{8.7e6:>20.4f}{1:>10d}{1.0e-6:>20.6e}"
    c2 = f"{15.0:>20.4f}{45.0:>20.4f}{30.0:>20.4f}{0.523:>20.4f}"
    deck = f"""# RADIOSS STARTER DECK
/BEGIN
Wobble Yoke Joint Fixed and Free Format Test
2022 0
/LAGMUL/WOBBLE_YOKE_JOINT/1
Fixed Wobble Yoke Joint
{c1}
{c2}
/LAGMUL/WOBBLE_YOKE_JOINT/2
Free Wobble Yoke Joint
1011, 1012, 1013, 9.4e6, 2, 1.5e-6
18.0, 54.0, 36.0, 0.785
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 1 in model.lagmul_wobble_yoke_joints
    assert 2 in model.lagmul_wobble_yoke_joints
    wyj1 = model.lagmul_wobble_yoke_joints[1]
    assert wyj1.node1 == 911
    assert wyj1.node2 == 912
    assert wyj1.node3 == 913
    assert pytest.approx(wyj1.stiff) == 8.7e6
    assert wyj1.skew_id == 1
    assert pytest.approx(wyj1.tol) == 1.0e-6
    assert pytest.approx(wyj1.nutation_angle) == 15.0
    assert pytest.approx(wyj1.yoke_radius) == 45.0
    assert pytest.approx(wyj1.stroke_travel) == 30.0
    assert pytest.approx(wyj1.phase_offset) == 0.523

    wyj2 = model.lagmul_wobble_yoke_joints[2]
    assert wyj2.node1 == 1011
    assert wyj2.node2 == 1012
    assert wyj2.node3 == 1013
    assert pytest.approx(wyj2.stiff) == 9.4e6
    assert wyj2.skew_id == 2
    assert pytest.approx(wyj2.tol) == 1.5e-6
    assert pytest.approx(wyj2.nutation_angle) == 18.0
    assert pytest.approx(wyj2.yoke_radius) == 54.0
    assert pytest.approx(wyj2.stroke_travel) == 36.0
    assert pytest.approx(wyj2.phase_offset) == 0.785


def test_m304_wobble_yoke_joint_aliases(tmp_path: Path):
    deck = """# RADIOSS ALIAS DECK
/BEGIN
Wobble Yoke Joint Aliases Test
/WOBBLE_YOKE_JOINT/95
1051, 1052, 1053, 6.0e6, 0, 1.0e-6
12.0, 36.0, 24.0, 0.0
/LAGMUL/WOBBLE_YOKE/96
1054, 1055, 1056, 6.0e6, 0, 1.0e-6
12.0, 36.0, 24.0, 0.0
/WOBBLE_YOKE/97
1057, 1058, 1059, 6.0e6, 0, 1.0e-6
12.0, 36.0, 24.0, 0.0
/WOBBLE_YOKE_MECHANISM/98
1060, 1061, 1062, 6.0e6, 0, 1.0e-6
12.0, 36.0, 24.0, 0.0
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 95 in model.lagmul_wobble_yoke_joints
    assert 96 in model.lagmul_wobble_yoke_joints
    assert 97 in model.lagmul_wobble_yoke_joints
    assert 98 in model.lagmul_wobble_yoke_joints
    assert model.lagmul_wobble_yoke_joints[95].node1 == 1051
    assert model.lagmul_wobble_yoke_joints[96].node1 == 1054
    assert model.lagmul_wobble_yoke_joints[97].node1 == 1057
    assert model.lagmul_wobble_yoke_joints[98].node1 == 1060


def test_m304_sensor_spring_bending_acceleration_rate(tmp_path: Path):
    c1 = f"{998:>10d}{3.1e7:>20.4f}{0.0086:>20.4f}"
    deck = f"""# RADIOSS STARTER DECK
/BEGIN
Sensor Spring Bending Acceleration Rate Fixed and Free Format Test
2022 0
/SENSOR/SPRING_BENDING_ACCELERATION_RATE/1
Fixed Spring Bending Acceleration Rate Sensor
{c1}
/SENSOR/SPRING_BENDING_ACCELERATION_RATE/2
Free Spring Bending Acceleration Rate Sensor
999, 5.1e7, 0.0116
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 1 in model.sensor_spring_bending_acceleration_rates
    assert 2 in model.sensor_spring_bending_acceleration_rates
    s1 = model.sensor_spring_bending_acceleration_rates[1]
    assert s1.spring_id == 998
    assert pytest.approx(s1.jbend_rate_max) == 3.1e7
    assert pytest.approx(s1.t_delay) == 0.0086

    s2 = model.sensor_spring_bending_acceleration_rates[2]
    assert s2.spring_id == 999
    assert pytest.approx(s2.jbend_rate_max) == 5.1e7
    assert pytest.approx(s2.t_delay) == 0.0116

    assert len(model.sensors) == 2
    assert model.sensors[0].kind == "SPRING_BENDING_ACCELERATION_RATE"
    assert model.sensors[1].kind == "SPRING_BENDING_ACCELERATION_RATE"


def test_m304_sensor_spring_bending_acceleration_rate_aliases(tmp_path: Path):
    deck = """# RADIOSS ALIAS DECK
/BEGIN
Sensor Spring Bending Acceleration Rate Aliases Test
/SENSOR/SPRING_BEND_ACC_RATE/104
1028, 2.25e7, 0.0028
/SENSOR/SPRING_RATE_ACC_BEND/105
1029, 2.45e7, 0.0038
/SENSOR/BENDING_ACCELERATION_RATE_SPRING/106
1030, 2.65e7, 0.0048
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 104 in model.sensor_spring_bending_acceleration_rates
    assert 105 in model.sensor_spring_bending_acceleration_rates
    assert 106 in model.sensor_spring_bending_acceleration_rates
    assert model.sensor_spring_bending_acceleration_rates[104].spring_id == 1028
    assert pytest.approx(model.sensor_spring_bending_acceleration_rates[104].jbend_rate_max) == 2.25e7
    assert model.sensor_spring_bending_acceleration_rates[105].spring_id == 1029
    assert model.sensor_spring_bending_acceleration_rates[106].spring_id == 1030
