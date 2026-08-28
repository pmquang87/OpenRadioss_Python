"""Tests for Milestone M399: LadTransverseMatrixCrackingRate Failure Model, EngFlexothermophononicpolaritonicResonanceEnergy, CauchySpatialLinkageJoint, and SensorSpringTotalSnapRate."""

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


def test_m399_fail_lad_transverse_matrix_cracking_rate_fixed(tmp_path: Path):
    c1 = f"{570.0:>20.4f}{1710.0:>20.4f}{335.0:>20.4f}{7.35:>20.4f}{0.865:>20.4f}"
    c2 = f"{1:>10d}{2:>10d}"
    c3 = f"{2130:>10d}"
    deck = f"""# RADIOSS STARTER DECK
/BEGIN
Ladeveze Transverse Matrix Cracking Rate Fixed Format Test
2022 0
/FAIL/LAD_TRANSVERSE_MATRIX_CRACKING_RATE/2130
Ladeveze Transverse Matrix Cracking Failure Model
{c1}
{c2}
{c3}
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 2130 in model.fail_ladtransversematrixcrackingrates
    ftmcr = model.fail_ladtransversematrixcrackingrates[2130]
    assert pytest.approx(ftmcr.sigma_tmcr0) == 570.0
    assert pytest.approx(ftmcr.sigma_tmcrc) == 1710.0
    assert pytest.approx(ftmcr.gamma_tmcr) == 335.0
    assert pytest.approx(ftmcr.p_tmcr) == 7.35
    assert pytest.approx(ftmcr.d_tmcr_max) == 0.865
    assert ftmcr.ifail_sh == 1
    assert ftmcr.ifail_so == 2
    assert ftmcr.fail_id == 2130
    assert len(model.raw_fails) == 1
    assert model.raw_fails[0][1].type == "LAD_TRANSVERSE_MATRIX_CRACKING_RATE"


def test_m399_fail_lad_transverse_matrix_cracking_rate_free(tmp_path: Path):
    deck = """# RADIOSS FREE DECK
/BEGIN
Ladeveze Transverse Matrix Cracking Rate Free Format Test
/FAIL/LAD_TRANSVERSE_MATRIX_CRACKING_RATE/2131
580.0, 1740.0, 345.0, 7.55, 0.855
1, 1
2131
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 2131 in model.fail_ladtransversematrixcrackingrates
    ftmcr = model.fail_ladtransversematrixcrackingrates[2131]
    assert pytest.approx(ftmcr.sigma_tmcr0) == 580.0
    assert pytest.approx(ftmcr.sigma_tmcrc) == 1740.0
    assert pytest.approx(ftmcr.gamma_tmcr) == 345.0
    assert pytest.approx(ftmcr.p_tmcr) == 7.55
    assert pytest.approx(ftmcr.d_tmcr_max) == 0.855
    assert ftmcr.fail_id == 2131


def test_m399_fail_lad_transverse_matrix_cracking_rate_aliases(tmp_path: Path):
    deck = """# RADIOSS ALIAS DECK
/BEGIN
Ladeveze Transverse Matrix Cracking Rate Aliases Test
/FAIL/LADEVEZE_TRANSVERSE_MATRIX_CRACKING_RATE/2132
560.0, 1680.0, 330.0, 7.15, 0.885
1, 1
/FAIL/LAD_TMCR/2133
560.0, 1680.0, 330.0, 7.15, 0.885
1, 1
/FAIL/LAD_TMCR_MODEL/2134
560.0, 1680.0, 330.0, 7.15, 0.885
1, 1
/FAIL/LAD_TMCR_LAW/2135
560.0, 1680.0, 330.0, 7.15, 0.885
1, 1
/FAIL/LADEVEZE_RATE_DEPENDENT_TRANSVERSE_MATRIX_CRACKING/2136
560.0, 1680.0, 330.0, 7.15, 0.885
1, 1
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 2132 in model.fail_ladtransversematrixcrackingrates
    assert 2133 in model.fail_ladtransversematrixcrackingrates
    assert 2134 in model.fail_ladtransversematrixcrackingrates
    assert 2135 in model.fail_ladtransversematrixcrackingrates
    assert 2136 in model.fail_ladtransversematrixcrackingrates
    assert len(model.raw_fails) == 5


def test_m399_fail_lad_transverse_matrix_cracking_rate_missing_card(tmp_path: Path):
    deck = """# RADIOSS MISSING CARD DECK
/BEGIN
/FAIL/LAD_TRANSVERSE_MATRIX_CRACKING_RATE/2137
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) > 0


def test_m399_eng_flexothermophononicpolaritonic_resonance_energy_fixed(tmp_path: Path):
    c1 = f"{0.0215:>20.4f}{435:>10d}"
    deck = f"""# RADIOSS STARTER DECK
/BEGIN
Engine Flexothermophononicpolaritonic Resonance Energy Fixed Format Test
2022 0
/ENG/FLEXOTHERMOPHONONICPOLARITONIC_RESONANCE_ENERGY/335
Resonance Energy Output Directive
{c1}
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 335 in model.eng_flexothermophononicpolaritonic_resonance_energies
    eng = model.eng_flexothermophononicpolaritonic_resonance_energies[335]
    assert pytest.approx(eng.dt_ftpp_ph) == 0.0215
    assert eng.sens_id == 435


def test_m399_eng_flexothermophononicpolaritonic_resonance_energy_free(tmp_path: Path):
    deck = """# RADIOSS FREE DECK
/BEGIN
Engine Flexothermophononicpolaritonic Resonance Energy Free Format Test
/ENG/FLEXOTHERMOPHONONICPOLARITONIC_RESONANCE_ENERGY/336
0.0225, 436
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 336 in model.eng_flexothermophononicpolaritonic_resonance_energies
    eng = model.eng_flexothermophononicpolaritonic_resonance_energies[336]
    assert pytest.approx(eng.dt_ftpp_ph) == 0.0225
    assert eng.sens_id == 436


def test_m399_eng_flexothermophononicpolaritonic_resonance_energy_aliases(tmp_path: Path):
    deck = """# RADIOSS ALIAS DECK
/BEGIN
Engine Flexothermophononicpolaritonic Resonance Energy Aliases Test
/ENG/FLEXOTHERM_PHONON_POLARITON_RES_WORK/337
0.0235, 437
/ENG/EFLEXOTHERMOPHONONICPOLARITONICRESONANCE/338
0.0245, 438
/ENG/FLEXOTHERMOPHONONICPOLARITONIC_RESONANCE_DISSIPATION/339
0.0255, 439
/ENG/ET_FLEXOTHERMOPHONONICPOLARITONIC_RESONANCE/340
0.0265, 440
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 337 in model.eng_flexothermophononicpolaritonic_resonance_energies
    assert 338 in model.eng_flexothermophononicpolaritonic_resonance_energies
    assert 339 in model.eng_flexothermophononicpolaritonic_resonance_energies
    assert 340 in model.eng_flexothermophononicpolaritonic_resonance_energies


def test_m399_eng_flexothermophononicpolaritonic_resonance_energy_missing_card(tmp_path: Path):
    deck = """# RADIOSS MISSING CARD DECK
/BEGIN
/ENG/FLEXOTHERMOPHONONICPOLARITONIC_RESONANCE_ENERGY/341
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) > 0


def test_m399_lagmul_cauchy_spatial_linkage_joint_fixed(tmp_path: Path):
    c1 = f"{421:>10d}{422:>10d}{423:>10d}{2.76e7:>20.4f}{139:>10d}{1.75e-4:>20.4e}"
    c2 = f"{205.0:>20.4f}{188.0:>20.4f}{215.0:>20.4f}{138.0:>20.4f}"
    deck = f"""# RADIOSS STARTER DECK
/BEGIN
Cauchy Spatial Linkage Joint Fixed Format Test
2022 0
/LAGMUL/CAUCHY_SPATIAL_LINKAGE_JOINT/385
Cauchy Spatial 6R Joint
{c1}
{c2}
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 385 in model.lagmul_cauchy_spatial_linkage_joints
    joint = model.lagmul_cauchy_spatial_linkage_joints[385]
    assert joint.node1 == 421
    assert joint.node2 == 422
    assert joint.node3 == 423
    assert pytest.approx(joint.stiff) == 2.76e7
    assert joint.skew_id == 139
    assert pytest.approx(joint.tol) == 1.75e-4
    assert pytest.approx(joint.link_len_a) == 205.0
    assert pytest.approx(joint.link_len_b) == 188.0
    assert pytest.approx(joint.twist_angle_alpha) == 215.0
    assert pytest.approx(joint.offset_distance_s) == 138.0
    assert pytest.approx(joint.offset_distance_r) == 138.0
    assert pytest.approx(joint.offset_distance_v) == 138.0
    assert pytest.approx(joint.offset_distance_h) == 138.0
    assert pytest.approx(joint.offset_distance_u) == 138.0


def test_m399_lagmul_cauchy_spatial_linkage_joint_free(tmp_path: Path):
    deck = """# RADIOSS FREE DECK
/BEGIN
Cauchy Spatial Linkage Joint Free Format Test
/LAGMUL/CAUCHY_SPATIAL_LINKAGE_JOINT/386
521, 522, 523, 3.00e7, 140, 1.85e-4
225.0, 195.0, 235.0, 150.0
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 386 in model.lagmul_cauchy_spatial_linkage_joints
    joint = model.lagmul_cauchy_spatial_linkage_joints[386]
    assert joint.node1 == 521
    assert joint.node2 == 522
    assert joint.node3 == 523
    assert pytest.approx(joint.stiff) == 3.00e7
    assert joint.skew_id == 140
    assert pytest.approx(joint.tol) == 1.85e-4
    assert pytest.approx(joint.link_len_a) == 225.0
    assert pytest.approx(joint.link_len_b) == 195.0
    assert pytest.approx(joint.twist_angle_alpha) == 235.0
    assert pytest.approx(joint.offset_distance_s) == 150.0


def test_m399_lagmul_cauchy_spatial_linkage_joint_aliases(tmp_path: Path):
    deck = """# RADIOSS ALIAS DECK
/BEGIN
Cauchy Spatial Linkage Joint Aliases Test
/CAUCHY_SPATIAL_LINKAGE_JOINT/387
621, 622, 623, 1.0e6, 0, 1.0e-6
165.0, 155.0, 165.0, 115.0
/LAGMUL/CAUCHY_SPATIAL_LINKAGE/388
621, 622, 623, 1.0e6, 0, 1.0e-6
165.0, 155.0, 165.0, 115.0
/CAUCHY_SPATIAL_LINKAGE/389
621, 622, 623, 1.0e6, 0, 1.0e-6
165.0, 155.0, 165.0, 115.0
/CAUCHY_SPATIAL_MULTI_LOOP_MECHANISM/390
621, 622, 623, 1.0e6, 0, 1.0e-6
165.0, 155.0, 165.0, 115.0
/CAUCHY_SPATIAL_SYMMETRIC_MECHANISM/391
621, 622, 623, 1.0e6, 0, 1.0e-6
165.0, 155.0, 165.0, 115.0
/CAUCHY_SPATIAL_6R_MECHANISM/392
621, 622, 623, 1.0e6, 0, 1.0e-6
165.0, 155.0, 165.0, 115.0
/CAUCHY_SPATIAL_OVERCONSTRAINED_MECHANISM/393
621, 622, 623, 1.0e6, 0, 1.0e-6
165.0, 155.0, 165.0, 115.0
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    for jid in range(387, 394):
        assert jid in model.lagmul_cauchy_spatial_linkage_joints


def test_m399_lagmul_cauchy_spatial_linkage_joint_missing_card(tmp_path: Path):
    deck = """# RADIOSS MISSING CARD DECK
/BEGIN
/LAGMUL/CAUCHY_SPATIAL_LINKAGE_JOINT/394
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) > 0


def test_m399_sensor_spring_total_snap_rate_fixed(tmp_path: Path):
    c1 = f"{1468:>10d}{3.98e8:>20.4f}{0.1265:>20.4f}"
    deck = f"""# RADIOSS STARTER DECK
/BEGIN
Sensor Spring Total Snap Rate Fixed Format Test
2022 0
/SENSOR/SPRING_TOTAL_SNAP_RATE/1
Fixed Spring Total Snap Rate Sensor
{c1}
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 1 in model.sensor_spring_total_snap_rates
    s1 = model.sensor_spring_total_snap_rates[1]
    assert s1.spring_id == 1468
    assert pytest.approx(s1.jtot_snp_max) == 3.98e8
    assert pytest.approx(s1.jtot_shot_max) == 3.98e8
    assert pytest.approx(s1.jtot_drop_max) == 3.98e8
    assert pytest.approx(s1.jtot_lock_max) == 3.98e8
    assert pytest.approx(s1.jtot_pop_max) == 3.98e8
    assert pytest.approx(s1.jtot_crackle_max) == 3.98e8
    assert pytest.approx(s1.t_delay) == 0.1265
    assert len(model.sensors) == 1
    assert model.sensors[0].kind == "SPRING_TOTAL_SNAP_RATE"


def test_m399_sensor_spring_total_snap_rate_free(tmp_path: Path):
    deck = """# RADIOSS FREE DECK
/BEGIN
Sensor Spring Total Snap Rate Free Format Test
/SENSOR/SPRING_TOTAL_SNAP_RATE/2
Free Spring Total Snap Rate Sensor
1469, 4.58e8, 0.1895
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 2 in model.sensor_spring_total_snap_rates
    s2 = model.sensor_spring_total_snap_rates[2]
    assert s2.spring_id == 1469
    assert pytest.approx(s2.jtot_snp_max) == 4.58e8
    assert pytest.approx(s2.t_delay) == 0.1895


def test_m399_sensor_spring_total_snap_rate_aliases(tmp_path: Path):
    deck = """# RADIOSS ALIAS DECK
/BEGIN
Sensor Spring Total Snap Rate Aliases Test
/SENSOR/SPRING_TOT_SNAP_RATE/391
1489, 2.195e8, 0.1065
/SENSOR/SPRING_RATE_SNAP_TOT/392
1490, 2.215e8, 0.1075
/SENSOR/TOTAL_SNAP_RATE_SPRING/393
1491, 2.235e8, 0.1085
/SENSOR/SPRING_SNAP_TOT/394
1492, 2.255e8, 0.1095
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    for sid in range(391, 395):
        assert sid in model.sensor_spring_total_snap_rates


def test_m399_sensor_spring_total_snap_rate_missing_card(tmp_path: Path):
    deck = """# RADIOSS MISSING CARD DECK
/BEGIN
/SENSOR/SPRING_TOTAL_SNAP_RATE/395
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) > 0
