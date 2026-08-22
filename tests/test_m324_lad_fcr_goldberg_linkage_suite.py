"""Tests for Milestone M324: LadFiberCompressionRate Failure Model, EngPiezothermalEnergy, GoldbergLinkageJoint, and SensorSpringTotalAngularCrackleRate."""

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


def test_m324_fail_lad_fiber_compression_rate_fixed(tmp_path: Path):
    c1 = f"{0.0125:>20.4f}{0.0350:>20.4f}{150.0:>20.4f}{90.0:>20.4f}{0.980:>20.4f}"
    c2 = f"{1:>10d}{2:>10d}"
    c3 = f"{1326:>10d}"
    deck = f"""# RADIOSS STARTER DECK
/BEGIN
Ladeveze Fiber Compression Rate Fixed Format Test
2022 0
/FAIL/LAD_FIBER_COMPRESSION_RATE/1326
Ladeveze Fiber Compression Rate Failure Model
{c1}
{c2}
{c3}
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 1326 in model.fail_ladfibercompressionrates
    ffcr = model.fail_ladfibercompressionrates[1326]
    assert pytest.approx(ffcr.eps_fc0) == 0.0125
    assert pytest.approx(ffcr.eps_fc_rate) == 0.0350
    assert pytest.approx(ffcr.eps_dot0) == 150.0
    assert pytest.approx(ffcr.w_fc_frac) == 90.0
    assert pytest.approx(ffcr.d_fc_max) == 0.980
    assert ffcr.ifail_sh == 1
    assert ffcr.ifail_so == 2
    assert ffcr.fail_id == 1326
    assert len(model.raw_fails) == 1
    assert model.raw_fails[0][1].type == "LAD_FIBER_COMPRESSION_RATE"


def test_m324_fail_lad_fiber_compression_rate_free(tmp_path: Path):
    deck = """# RADIOSS FREE DECK
/BEGIN
Ladeveze Fiber Compression Rate Free Format Test
/FAIL/LAD_FIBER_COMPRESSION_RATE/1327
0.0185, 0.0450, 250.0, 110.0, 0.965
1, 1
1327
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 1327 in model.fail_ladfibercompressionrates
    ffcr = model.fail_ladfibercompressionrates[1327]
    assert pytest.approx(ffcr.eps_fc0) == 0.0185
    assert pytest.approx(ffcr.eps_fc_rate) == 0.0450
    assert pytest.approx(ffcr.eps_dot0) == 250.0
    assert pytest.approx(ffcr.w_fc_frac) == 110.0
    assert pytest.approx(ffcr.d_fc_max) == 0.965
    assert ffcr.fail_id == 1327


def test_m324_fail_lad_fiber_compression_rate_aliases(tmp_path: Path):
    deck = """# RADIOSS ALIAS DECK
/BEGIN
Ladeveze Fiber Compression Rate Aliases Test
/FAIL/LADEVEZE_FIBER_COMPRESSION_RATE/1328
0.015, 0.04, 1.0, 75.0, 0.95
1, 1
/FAIL/LAD_FCR/1329
0.015, 0.04, 1.0, 75.0, 0.95
1, 1
/FAIL/LAD_FCR_MODEL/1330
0.015, 0.04, 1.0, 75.0, 0.95
1, 1
/FAIL/LAD_FCR_LAW/1331
0.015, 0.04, 1.0, 75.0, 0.95
1, 1
/FAIL/LADEVEZE_RATE_DEPENDENT_FIBER_CRUSH/1332
0.015, 0.04, 1.0, 75.0, 0.95
1, 1
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 1328 in model.fail_ladfibercompressionrates
    assert 1329 in model.fail_ladfibercompressionrates
    assert 1330 in model.fail_ladfibercompressionrates
    assert 1331 in model.fail_ladfibercompressionrates
    assert 1332 in model.fail_ladfibercompressionrates
    assert len(model.raw_fails) == 5


def test_m324_fail_lad_fiber_compression_rate_missing_card(tmp_path: Path):
    deck = """# RADIOSS EMPTY CARD DECK
/BEGIN
Missing Card Test
/FAIL/LAD_FIBER_COMPRESSION_RATE/1333
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) > 0


def test_m324_eng_piezothermal_energy(tmp_path: Path):
    c1 = f"{0.00035:>20.6f}{80:>10d}"
    deck = f"""# RADIOSS STARTER DECK
/BEGIN
Piezothermal Energy Fixed and Free Format Test
2022 0
/ENG/PIEZOTHERMAL_ENERGY/1
Fixed Piezothermal Energy Output
{c1}
/ENG/PIEZOTHERMAL_ENERGY/2
Free Piezothermal Energy Output
0.00075, 160
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 1 in model.eng_piezothermal_energies
    assert 2 in model.eng_piezothermal_energies
    pzt1 = model.eng_piezothermal_energies[1]
    assert pytest.approx(pzt1.dt_pzt) == 0.00035
    assert pzt1.sens_id == 80
    pzt2 = model.eng_piezothermal_energies[2]
    assert pytest.approx(pzt2.dt_pzt) == 0.00075
    assert pzt2.sens_id == 160


def test_m324_eng_piezothermal_energy_aliases(tmp_path: Path):
    deck = """# RADIOSS ALIAS DECK
/BEGIN
Piezothermal Energy Aliases Test
/ENG/PIEZO_THERM_WORK/3
0.00025, 50
/ENG/EPIEZOTHERMAL/4
0.00030, 60
/ENG/PIEZOTHERMAL_DISSIPATION/5
0.00035, 70
/ENG/EM_PIEZOTHERMAL/6
0.00040, 80
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 3 in model.eng_piezothermal_energies
    assert 4 in model.eng_piezothermal_energies
    assert 5 in model.eng_piezothermal_energies
    assert 6 in model.eng_piezothermal_energies


def test_m324_eng_piezothermal_energy_missing_card(tmp_path: Path):
    deck = """# RADIOSS EMPTY CARD DECK
/BEGIN
Missing Card Test
/ENG/PIEZOTHERMAL_ENERGY/7
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) > 0


def test_m324_lagmul_goldberg_linkage_joint_fixed(tmp_path: Path):
    c1 = f"{101:>10d}{102:>10d}{103:>10d}{5.5e6:>20.1f}{15:>10d}{1.0e-5:>20.6e}"
    c2 = f"{42.5:>20.4f}{38.0:>20.4f}{35.0:>20.4f}{25.0:>20.4f}"
    deck = f"""# RADIOSS STARTER DECK
/BEGIN
Goldberg Linkage Joint Fixed Format Test
2022 0
/GOLDBERG_LINKAGE_JOINT/55
Goldberg Linkage Kinematic Joint
{c1}
{c2}
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 55 in model.lagmul_goldberg_linkage_joints
    joint = model.lagmul_goldberg_linkage_joints[55]
    assert joint.node1 == 101
    assert joint.node2 == 102
    assert joint.node3 == 103
    assert pytest.approx(joint.stiff) == 5.5e6
    assert joint.skew_id == 15
    assert pytest.approx(joint.tol) == 1.0e-5
    assert pytest.approx(joint.link_len_a) == 42.5
    assert pytest.approx(joint.link_len_b) == 38.0
    assert pytest.approx(joint.skew_angle_alpha) == 35.0
    assert pytest.approx(joint.offset_angle_beta) == 25.0


def test_m324_lagmul_goldberg_linkage_joint_free(tmp_path: Path):
    deck = """# RADIOSS FREE DECK
/BEGIN
Goldberg Linkage Joint Free Format Test
/LAGMUL/GOLDBERG_LINKAGE_JOINT/56
201, 202, 203, 3.2e6, 25, 2.0e-5
48.0, 44.5, 40.0, 30.0
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 56 in model.lagmul_goldberg_linkage_joints
    joint = model.lagmul_goldberg_linkage_joints[56]
    assert joint.node1 == 201
    assert joint.node2 == 202
    assert joint.node3 == 203
    assert pytest.approx(joint.stiff) == 3.2e6
    assert joint.skew_id == 25
    assert pytest.approx(joint.tol) == 2.0e-5
    assert pytest.approx(joint.link_len_a) == 48.0
    assert pytest.approx(joint.link_len_b) == 44.5
    assert pytest.approx(joint.skew_angle_alpha) == 40.0
    assert pytest.approx(joint.offset_angle_beta) == 30.0


def test_m324_lagmul_goldberg_linkage_joint_aliases(tmp_path: Path):
    deck = """# RADIOSS ALIAS DECK
/BEGIN
Goldberg Linkage Joint Aliases Test
/LAGMUL/GOLDBERG_LINKAGE/57
1, 2, 3, 1e6, 0, 1e-6
10.0, 10.0, 30.0, 20.0
/GOLDBERG_LINKAGE/58
1, 2, 3, 1e6, 0, 1e-6
10.0, 10.0, 30.0, 20.0
/GOLDBERG_5R_6R_MECHANISM/59
1, 2, 3, 1e6, 0, 1e-6
10.0, 10.0, 30.0, 20.0
/GOLDBERG_SPATIAL_MECHANISM/60
1, 2, 3, 1e6, 0, 1e-6
10.0, 10.0, 30.0, 20.0
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 57 in model.lagmul_goldberg_linkage_joints
    assert 58 in model.lagmul_goldberg_linkage_joints
    assert 59 in model.lagmul_goldberg_linkage_joints
    assert 60 in model.lagmul_goldberg_linkage_joints


def test_m324_lagmul_goldberg_linkage_joint_missing_card(tmp_path: Path):
    deck = """# RADIOSS EMPTY CARD DECK
/BEGIN
Missing Card Test
/GOLDBERG_LINKAGE_JOINT/61
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) > 0


def test_m324_sensor_spring_total_angular_crackle_rate_fixed(tmp_path: Path):
    c1 = f"{701:>10d}{8.8e7:>20.1f}{0.0035:>20.4f}"
    deck = f"""# RADIOSS STARTER DECK
/BEGIN
Spring Total Angular Crackle Rate Sensor Fixed Format Test
2022 0
/SENSOR/SPRING_TOTAL_ANGULAR_CRACKLE_RATE/88
Spring Total Angular Crackle Rate Sensor
{c1}
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 88 in model.sensor_spring_total_angular_crackle_rates
    sensor = model.sensor_spring_total_angular_crackle_rates[88]
    assert sensor.spring_id == 701
    assert pytest.approx(sensor.jang_pop_max) == 8.8e7
    assert pytest.approx(sensor.t_delay) == 0.0035
    assert len(model.sensors) == 1
    assert model.sensors[0].kind == "SPRING_TOTAL_ANGULAR_CRACKLE_RATE"


def test_m324_sensor_spring_total_angular_crackle_rate_free(tmp_path: Path):
    deck = """# RADIOSS FREE DECK
/BEGIN
Spring Total Angular Crackle Rate Sensor Free Format Test
/SENSOR/SPRING_TOTAL_ANGULAR_CRACKLE_RATE/89
702, 9.5e7, 0.0050
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 89 in model.sensor_spring_total_angular_crackle_rates
    sensor = model.sensor_spring_total_angular_crackle_rates[89]
    assert sensor.spring_id == 702
    assert pytest.approx(sensor.jang_pop_max) == 9.5e7
    assert pytest.approx(sensor.t_delay) == 0.0050


def test_m324_sensor_spring_total_angular_crackle_rate_aliases(tmp_path: Path):
    deck = """# RADIOSS ALIAS DECK
/BEGIN
Spring Total Angular Crackle Rate Sensor Aliases Test
/SENSOR/SPRING_TOT_ANG_CRACKLE_RATE/90
703, 1.0e8, 0.001
/SENSOR/SPRING_CRACKLE_ANG_TOT/91
704, 1.0e8, 0.001
/SENSOR/TOTAL_ANGULAR_CRACKLE_RATE_SPRING/92
705, 1.0e8, 0.001
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 90 in model.sensor_spring_total_angular_crackle_rates
    assert 91 in model.sensor_spring_total_angular_crackle_rates
    assert 92 in model.sensor_spring_total_angular_crackle_rates
    assert len(model.sensors) == 3


def test_m324_sensor_spring_total_angular_crackle_rate_missing_card(tmp_path: Path):
    deck = """# RADIOSS EMPTY CARD DECK
/BEGIN
Missing Card Test
/SENSOR/SPRING_TOTAL_ANGULAR_CRACKLE_RATE/93
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) > 0
