"""Tests for Milestone M323: LadFiberTensionRate Failure Model, EngPyromagneticEnergy, MyardLinkageJoint, and SensorSpringBendingCrackleRate."""

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


def test_m323_fail_lad_fiber_tension_rate_fixed(tmp_path: Path):
    c1 = f"{0.0175:>20.4f}{0.0450:>20.4f}{100.0:>20.4f}{85.0:>20.4f}{0.975:>20.4f}"
    c2 = f"{1:>10d}{2:>10d}"
    c3 = f"{1319:>10d}"
    deck = f"""# RADIOSS STARTER DECK
/BEGIN
Ladeveze Fiber Tension Rate Fixed Format Test
2022 0
/FAIL/LAD_FIBER_TENSION_RATE/1319
Ladeveze Fiber Tension Rate Failure Model
{c1}
{c2}
{c3}
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 1319 in model.fail_ladfibertensionrates
    fftr = model.fail_ladfibertensionrates[1319]
    assert pytest.approx(fftr.eps_ft0) == 0.0175
    assert pytest.approx(fftr.eps_ft_rate) == 0.0450
    assert pytest.approx(fftr.eps_dot0) == 100.0
    assert pytest.approx(fftr.w_ft_frac) == 85.0
    assert pytest.approx(fftr.d_ft_max) == 0.975
    assert fftr.ifail_sh == 1
    assert fftr.ifail_so == 2
    assert fftr.fail_id == 1319
    assert len(model.raw_fails) == 1
    assert model.raw_fails[0][1].type == "LAD_FIBER_TENSION_RATE"


def test_m323_fail_lad_fiber_tension_rate_free(tmp_path: Path):
    deck = """# RADIOSS FREE DECK
/BEGIN
Ladeveze Fiber Tension Rate Free Format Test
/FAIL/LAD_FIBER_TENSION_RATE/1320
0.0215, 0.0550, 200.0, 95.0, 0.960
1, 1
1320
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 1320 in model.fail_ladfibertensionrates
    fftr = model.fail_ladfibertensionrates[1320]
    assert pytest.approx(fftr.eps_ft0) == 0.0215
    assert pytest.approx(fftr.eps_ft_rate) == 0.0550
    assert pytest.approx(fftr.eps_dot0) == 200.0
    assert pytest.approx(fftr.w_ft_frac) == 95.0
    assert pytest.approx(fftr.d_ft_max) == 0.960
    assert fftr.fail_id == 1320


def test_m323_fail_lad_fiber_tension_rate_aliases(tmp_path: Path):
    deck = """# RADIOSS ALIAS DECK
/BEGIN
Ladeveze Fiber Tension Rate Aliases Test
/FAIL/LADEVEZE_FIBER_TENSION_RATE/1321
0.015, 0.04, 1.0, 75.0, 0.95
1, 1
/FAIL/LAD_FTR/1322
0.015, 0.04, 1.0, 75.0, 0.95
1, 1
/FAIL/LAD_FTR_MODEL/1323
0.015, 0.04, 1.0, 75.0, 0.95
1, 1
/FAIL/LAD_FTR_LAW/1324
0.015, 0.04, 1.0, 75.0, 0.95
1, 1
/FAIL/LADEVEZE_RATE_DEPENDENT_FIBER_DAMAGE/1325
0.015, 0.04, 1.0, 75.0, 0.95
1, 1
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 1321 in model.fail_ladfibertensionrates
    assert 1322 in model.fail_ladfibertensionrates
    assert 1323 in model.fail_ladfibertensionrates
    assert 1324 in model.fail_ladfibertensionrates
    assert 1325 in model.fail_ladfibertensionrates
    assert len(model.raw_fails) == 5


def test_m323_eng_pyromagnetic_energy(tmp_path: Path):
    c1 = f"{0.00045:>20.6f}{70:>10d}"
    deck = f"""# RADIOSS STARTER DECK
/BEGIN
Pyromagnetic Energy Fixed and Free Format Test
2022 0
/ENG/PYROMAGNETIC_ENERGY/1
Fixed Pyromagnetic Energy Output
{c1}
/ENG/PYROMAGNETIC_ENERGY/2
Free Pyromagnetic Energy Output
0.00095, 150
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 1 in model.eng_pyromagnetic_energies
    assert 2 in model.eng_pyromagnetic_energies
    pmg1 = model.eng_pyromagnetic_energies[1]
    assert pytest.approx(pmg1.dt_pmg) == 0.00045
    assert pmg1.sens_id == 70
    pmg2 = model.eng_pyromagnetic_energies[2]
    assert pytest.approx(pmg2.dt_pmg) == 0.00095
    assert pmg2.sens_id == 150


def test_m323_eng_pyromagnetic_energy_aliases(tmp_path: Path):
    deck = """# RADIOSS ALIAS DECK
/BEGIN
Pyromagnetic Energy Aliases Test
/ENG/PYROMAG_WORK/281
0.0016, 280
/ENG/EPYROMAGNETIC/282
0.0026, 281
/ENG/PYROMAGNETIC_DISSIPATION/283
0.0036, 282
/ENG/EM_PYROMAGNETIC/284
0.0046, 283
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 281 in model.eng_pyromagnetic_energies
    assert 282 in model.eng_pyromagnetic_energies
    assert 283 in model.eng_pyromagnetic_energies
    assert 284 in model.eng_pyromagnetic_energies
    assert pytest.approx(model.eng_pyromagnetic_energies[281].dt_pmg) == 0.0016
    assert pytest.approx(model.eng_pyromagnetic_energies[282].dt_pmg) == 0.0026
    assert pytest.approx(model.eng_pyromagnetic_energies[283].dt_pmg) == 0.0036
    assert pytest.approx(model.eng_pyromagnetic_energies[284].dt_pmg) == 0.0046


def test_m323_myard_linkage_joint(tmp_path: Path):
    c1 = f"{1081:>10d}{1082:>10d}{1083:>10d}{1.76e7:>20.4f}{1:>10d}{1.0e-6:>20.6e}"
    c2 = f"{50.0:>20.4f}{65.0:>20.4f}{40.0:>20.4f}{55.0:>20.4f}"
    deck = f"""# RADIOSS STARTER DECK
/BEGIN
Myard Linkage Joint Fixed and Free Format Test
2022 0
/LAGMUL/MYARD_LINKAGE_JOINT/1
Fixed Myard Spatial Mechanism Joint
{c1}
{c2}
/LAGMUL/MYARD_LINKAGE_JOINT/2
Free Myard Spatial Mechanism Joint
1181, 1182, 1183, 1.96e7, 2, 2.8e-6
60.0, 75.0, 45.0, 60.0
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 1 in model.lagmul_myard_linkage_joints
    assert 2 in model.lagmul_myard_linkage_joints
    mlj1 = model.lagmul_myard_linkage_joints[1]
    assert mlj1.node1 == 1081
    assert mlj1.node2 == 1082
    assert mlj1.node3 == 1083
    assert pytest.approx(mlj1.stiff) == 1.76e7
    assert mlj1.skew_id == 1
    assert pytest.approx(mlj1.tol) == 1.0e-6
    assert pytest.approx(mlj1.link_len_a) == 50.0
    assert pytest.approx(mlj1.link_len_b) == 65.0
    assert pytest.approx(mlj1.twist_angle) == 40.0
    assert pytest.approx(mlj1.fold_angle) == 55.0

    mlj2 = model.lagmul_myard_linkage_joints[2]
    assert mlj2.node1 == 1181
    assert mlj2.node2 == 1182
    assert mlj2.node3 == 1183
    assert pytest.approx(mlj2.stiff) == 1.96e7
    assert mlj2.skew_id == 2
    assert pytest.approx(mlj2.tol) == 2.8e-6
    assert pytest.approx(mlj2.link_len_a) == 60.0
    assert pytest.approx(mlj2.link_len_b) == 75.0
    assert pytest.approx(mlj2.twist_angle) == 45.0
    assert pytest.approx(mlj2.fold_angle) == 60.0


def test_m323_myard_linkage_joint_aliases(tmp_path: Path):
    deck = """# RADIOSS ALIAS DECK
/BEGIN
Myard Linkage Joint Aliases Test
/MYARD_LINKAGE_JOINT/285
1281, 1282, 1283, 9.7e6, 0, 1.0e-6
45.0, 60.0, 35.0, 50.0
/LAGMUL/MYARD_LINKAGE/286
1284, 1285, 1286, 9.7e6, 0, 1.0e-6
45.0, 60.0, 35.0, 50.0
/MYARD_LINKAGE/287
1287, 1288, 1289, 9.7e6, 0, 1.0e-6
45.0, 60.0, 35.0, 50.0
/MYARD_SPATIAL_MECHANISM/288
1290, 1291, 1292, 9.7e6, 0, 1.0e-6
45.0, 60.0, 35.0, 50.0
/MYARD_5R_MECHANISM/289
1293, 1294, 1295, 9.7e6, 0, 1.0e-6
45.0, 60.0, 35.0, 50.0
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 285 in model.lagmul_myard_linkage_joints
    assert 286 in model.lagmul_myard_linkage_joints
    assert 287 in model.lagmul_myard_linkage_joints
    assert 288 in model.lagmul_myard_linkage_joints
    assert 289 in model.lagmul_myard_linkage_joints
    assert model.lagmul_myard_linkage_joints[285].node1 == 1281
    assert model.lagmul_myard_linkage_joints[286].node1 == 1284
    assert model.lagmul_myard_linkage_joints[287].node1 == 1287
    assert model.lagmul_myard_linkage_joints[288].node1 == 1290
    assert model.lagmul_myard_linkage_joints[289].node1 == 1293


def test_m323_sensor_spring_bending_crackle_rate(tmp_path: Path):
    c1 = f"{1188:>10d}{7.8e7:>20.4f}{0.0235:>20.4f}"
    deck = f"""# RADIOSS STARTER DECK
/BEGIN
Sensor Spring Bending Crackle Rate Fixed and Free Format Test
2022 0
/SENSOR/SPRING_BENDING_CRACKLE_RATE/1
Fixed Spring Bending Crackle Rate Sensor
{c1}
/SENSOR/SPRING_BENDING_CRACKLE_RATE/2
Free Spring Bending Crackle Rate Sensor
1189, 9.85e7, 0.0335
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 1 in model.sensor_spring_bending_crackle_rates
    assert 2 in model.sensor_spring_bending_crackle_rates
    s1 = model.sensor_spring_bending_crackle_rates[1]
    assert s1.spring_id == 1188
    assert pytest.approx(s1.jbend_pop_max) == 7.8e7
    assert pytest.approx(s1.t_delay) == 0.0235

    s2 = model.sensor_spring_bending_crackle_rates[2]
    assert s2.spring_id == 1189
    assert pytest.approx(s2.jbend_pop_max) == 9.85e7
    assert pytest.approx(s2.t_delay) == 0.0335

    assert len(model.sensors) == 2
    assert model.sensors[0].kind == "SPRING_BENDING_CRACKLE_RATE"
    assert model.sensors[1].kind == "SPRING_BENDING_CRACKLE_RATE"


def test_m323_sensor_spring_bending_crackle_rate_aliases(tmp_path: Path):
    deck = """# RADIOSS ALIAS DECK
/BEGIN
Sensor Spring Bending Crackle Rate Aliases Test
/SENSOR/SPRING_BEND_CRACKLE_RATE/203
1218, 6.45e7, 0.0076
/SENSOR/SPRING_RATE_CRACKLE_BEND/204
1219, 6.65e7, 0.0086
/SENSOR/BENDING_CRACKLE_RATE_SPRING/205
1220, 6.85e7, 0.0096
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 203 in model.sensor_spring_bending_crackle_rates
    assert 204 in model.sensor_spring_bending_crackle_rates
    assert 205 in model.sensor_spring_bending_crackle_rates
    assert model.sensor_spring_bending_crackle_rates[203].spring_id == 1218
    assert pytest.approx(model.sensor_spring_bending_crackle_rates[203].jbend_pop_max) == 6.45e7
    assert model.sensor_spring_bending_crackle_rates[204].spring_id == 1219
    assert model.sensor_spring_bending_crackle_rates[205].spring_id == 1220
