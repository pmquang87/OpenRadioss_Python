"""Tests for Milestone M397: LadCoupleDelaminationRate Failure Model, EngFlexothermophononicmagnonicpolaritonicResonanceEnergy, TchebychevSpatialLinkageJoint, and SensorSpringNormalSnapRate."""

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


def test_m397_fail_lad_couple_delamination_rate_fixed(tmp_path: Path):
    c1 = f"{550.0:>20.4f}{1650.0:>20.4f}{315.0:>20.4f}{7.15:>20.4f}{0.885:>20.4f}"
    c2 = f"{1:>10d}{2:>10d}"
    c3 = f"{2110:>10d}"
    deck = f"""# RADIOSS STARTER DECK
/BEGIN
Ladeveze Coupled Delamination Rate Fixed Format Test
2022 0
/FAIL/LAD_COUPLE_DELAMINATION_RATE/2110
Ladeveze Coupled Delamination Failure Model
{c1}
{c2}
{c3}
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 2110 in model.fail_ladcoupledelaminationrates
    fcdlr = model.fail_ladcoupledelaminationrates[2110]
    assert pytest.approx(fcdlr.sigma_cdlr0) == 550.0
    assert pytest.approx(fcdlr.sigma_cdlrc) == 1650.0
    assert pytest.approx(fcdlr.gamma_cdlr) == 315.0
    assert pytest.approx(fcdlr.p_cdlr) == 7.15
    assert pytest.approx(fcdlr.d_cdlr_max) == 0.885
    assert fcdlr.ifail_sh == 1
    assert fcdlr.ifail_so == 2
    assert fcdlr.fail_id == 2110
    assert len(model.raw_fails) == 1
    assert model.raw_fails[0][1].type == "LAD_COUPLE_DELAMINATION_RATE"


def test_m397_fail_lad_couple_delamination_rate_free(tmp_path: Path):
    deck = """# RADIOSS FREE DECK
/BEGIN
Ladeveze Coupled Delamination Rate Free Format Test
/FAIL/LAD_COUPLE_DELAMINATION_RATE/2111
560.0, 1680.0, 325.0, 7.35, 0.875
1, 1
2111
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 2111 in model.fail_ladcoupledelaminationrates
    fcdlr = model.fail_ladcoupledelaminationrates[2111]
    assert pytest.approx(fcdlr.sigma_cdlr0) == 560.0
    assert pytest.approx(fcdlr.sigma_cdlrc) == 1680.0
    assert pytest.approx(fcdlr.gamma_cdlr) == 325.0
    assert pytest.approx(fcdlr.p_cdlr) == 7.35
    assert pytest.approx(fcdlr.d_cdlr_max) == 0.875
    assert fcdlr.fail_id == 2111


def test_m397_fail_lad_couple_delamination_rate_aliases(tmp_path: Path):
    deck = """# RADIOSS ALIAS DECK
/BEGIN
Ladeveze Coupled Delamination Rate Aliases Test
/FAIL/LADEVEZE_COUPLED_DELAMINATION_RATE/2112
540.0, 1620.0, 310.0, 6.95, 0.905
1, 1
/FAIL/LAD_CDLR/2113
540.0, 1620.0, 310.0, 6.95, 0.905
1, 1
/FAIL/LAD_CDLR_MODEL/2114
540.0, 1620.0, 310.0, 6.95, 0.905
1, 1
/FAIL/LAD_CDLR_LAW/2115
540.0, 1620.0, 310.0, 6.95, 0.905
1, 1
/FAIL/LADEVEZE_RATE_DEPENDENT_COUPLED_DELAMINATION/2116
540.0, 1620.0, 310.0, 6.95, 0.905
1, 1
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 2112 in model.fail_ladcoupledelaminationrates
    assert 2113 in model.fail_ladcoupledelaminationrates
    assert 2114 in model.fail_ladcoupledelaminationrates
    assert 2115 in model.fail_ladcoupledelaminationrates
    assert 2116 in model.fail_ladcoupledelaminationrates
    assert len(model.raw_fails) == 5


def test_m397_fail_lad_couple_delamination_rate_missing_card(tmp_path: Path):
    deck = """# RADIOSS MISSING CARD DECK
/BEGIN
/FAIL/LAD_COUPLE_DELAMINATION_RATE/2117
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) > 0


def test_m397_eng_flexothermophononicmagnonicpolaritonic_resonance_energy_fixed(tmp_path: Path):
    c1 = f"{0.0195:>20.4f}{415:>10d}"
    deck = f"""# RADIOSS STARTER DECK
/BEGIN
Engine Flexothermophononicmagnonicpolaritonic Resonance Energy Fixed Format Test
2022 0
/ENG/FLEXOTHERMOPHONONICMAGNONICPOLARITONIC_RESONANCE_ENERGY/315
Resonance Energy Output Directive
{c1}
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 315 in model.eng_flexothermophononicmagnonicpolaritonic_resonance_energies
    eng = model.eng_flexothermophononicmagnonicpolaritonic_resonance_energies[315]
    assert pytest.approx(eng.dt_ftpmp) == 0.0195
    assert eng.sens_id == 415


def test_m397_eng_flexothermophononicmagnonicpolaritonic_resonance_energy_free(tmp_path: Path):
    deck = """# RADIOSS FREE DECK
/BEGIN
Engine Flexothermophononicmagnonicpolaritonic Resonance Energy Free Format Test
/ENG/FLEXOTHERMOPHONONICMAGNONICPOLARITONIC_RESONANCE_ENERGY/316
0.0205, 416
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 316 in model.eng_flexothermophononicmagnonicpolaritonic_resonance_energies
    eng = model.eng_flexothermophononicmagnonicpolaritonic_resonance_energies[316]
    assert pytest.approx(eng.dt_ftpmp) == 0.0205
    assert eng.sens_id == 416


def test_m397_eng_flexothermophononicmagnonicpolaritonic_resonance_energy_aliases(tmp_path: Path):
    deck = """# RADIOSS ALIAS DECK
/BEGIN
Engine Flexothermophononicmagnonicpolaritonic Resonance Energy Aliases Test
/ENG/FLEXOTHERM_PHONONIC_MAGNONIC_POLARITON_RES_WORK/317
0.0215, 417
/ENG/EFLEXOTHERMOPHONONICMAGNONICPOLARITONICRESONANCE/318
0.0225, 418
/ENG/FLEXOTHERMOPHONONICMAGNONICPOLARITONIC_RESONANCE_DISSIPATION/319
0.0235, 419
/ENG/ET_FLEXOTHERMOPHONONICMAGNONICPOLARITONIC_RESONANCE/320
0.0245, 420
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 317 in model.eng_flexothermophononicmagnonicpolaritonic_resonance_energies
    assert 318 in model.eng_flexothermophononicmagnonicpolaritonic_resonance_energies
    assert 319 in model.eng_flexothermophononicmagnonicpolaritonic_resonance_energies
    assert 320 in model.eng_flexothermophononicmagnonicpolaritonic_resonance_energies


def test_m397_eng_flexothermophononicmagnonicpolaritonic_resonance_energy_missing_card(tmp_path: Path):
    deck = """# RADIOSS MISSING CARD DECK
/BEGIN
/ENG/FLEXOTHERMOPHONONICMAGNONICPOLARITONIC_RESONANCE_ENERGY/321
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) > 0


def test_m397_lagmul_tchebychev_spatial_linkage_joint_fixed(tmp_path: Path):
    c1 = f"{401:>10d}{402:>10d}{403:>10d}{2.56e7:>20.4f}{119:>10d}{1.55e-4:>20.4e}"
    c2 = f"{185.0:>20.4f}{168.0:>20.4f}{195.0:>20.4f}{118.0:>20.4f}"
    deck = f"""# RADIOSS STARTER DECK
/BEGIN
Tchebychev Spatial Linkage Joint Fixed Format Test
2022 0
/LAGMUL/TCHEBYCHEV_SPATIAL_LINKAGE_JOINT/365
Tchebychev Spatial 6R Joint
{c1}
{c2}
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 365 in model.lagmul_tchebychev_spatial_linkage_joints
    joint = model.lagmul_tchebychev_spatial_linkage_joints[365]
    assert joint.node1 == 401
    assert joint.node2 == 402
    assert joint.node3 == 403
    assert pytest.approx(joint.stiff) == 2.56e7
    assert joint.skew_id == 119
    assert pytest.approx(joint.tol) == 1.55e-4
    assert pytest.approx(joint.link_len_a) == 185.0
    assert pytest.approx(joint.link_len_b) == 168.0
    assert pytest.approx(joint.twist_angle_alpha) == 195.0
    assert pytest.approx(joint.offset_distance_s) == 118.0
    assert pytest.approx(joint.offset_distance_r) == 118.0
    assert pytest.approx(joint.offset_distance_v) == 118.0
    assert pytest.approx(joint.offset_distance_h) == 118.0
    assert pytest.approx(joint.offset_distance_u) == 118.0


def test_m397_lagmul_tchebychev_spatial_linkage_joint_free(tmp_path: Path):
    deck = """# RADIOSS FREE DECK
/BEGIN
Tchebychev Spatial Linkage Joint Free Format Test
/LAGMUL/TCHEBYCHEV_SPATIAL_LINKAGE_JOINT/366
501, 502, 503, 2.80e7, 120, 1.65e-4
205.0, 175.0, 215.0, 130.0
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 366 in model.lagmul_tchebychev_spatial_linkage_joints
    joint = model.lagmul_tchebychev_spatial_linkage_joints[366]
    assert joint.node1 == 501
    assert joint.node2 == 502
    assert joint.node3 == 503
    assert pytest.approx(joint.stiff) == 2.80e7
    assert joint.skew_id == 120
    assert pytest.approx(joint.tol) == 1.65e-4
    assert pytest.approx(joint.link_len_a) == 205.0
    assert pytest.approx(joint.link_len_b) == 175.0
    assert pytest.approx(joint.twist_angle_alpha) == 215.0
    assert pytest.approx(joint.offset_distance_s) == 130.0


def test_m397_lagmul_tchebychev_spatial_linkage_joint_aliases(tmp_path: Path):
    deck = """# RADIOSS ALIAS DECK
/BEGIN
Tchebychev Spatial Linkage Joint Aliases Test
/TCHEBYCHEV_SPATIAL_LINKAGE_JOINT/367
601, 602, 603, 1.0e6, 0, 1.0e-6
145.0, 135.0, 145.0, 95.0
/LAGMUL/TCHEBYCHEV_SPATIAL_LINKAGE/368
601, 602, 603, 1.0e6, 0, 1.0e-6
145.0, 135.0, 145.0, 95.0
/TCHEBYCHEV_SPATIAL_LINKAGE/369
601, 602, 603, 1.0e6, 0, 1.0e-6
145.0, 135.0, 145.0, 95.0
/TCHEBYCHEV_SPATIAL_MULTI_LOOP_MECHANISM/370
601, 602, 603, 1.0e6, 0, 1.0e-6
145.0, 135.0, 145.0, 95.0
/TCHEBYCHEV_SPATIAL_SYMMETRIC_MECHANISM/371
601, 602, 603, 1.0e6, 0, 1.0e-6
145.0, 135.0, 145.0, 95.0
/TCHEBYCHEV_SPATIAL_6R_MECHANISM/372
601, 602, 603, 1.0e6, 0, 1.0e-6
145.0, 135.0, 145.0, 95.0
/TCHEBYCHEV_SPATIAL_OVERCONSTRAINED_MECHANISM/373
601, 602, 603, 1.0e6, 0, 1.0e-6
145.0, 135.0, 145.0, 95.0
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    for jid in range(367, 374):
        assert jid in model.lagmul_tchebychev_spatial_linkage_joints


def test_m397_lagmul_tchebychev_spatial_linkage_joint_missing_card(tmp_path: Path):
    deck = """# RADIOSS MISSING CARD DECK
/BEGIN
/LAGMUL/TCHEBYCHEV_SPATIAL_LINKAGE_JOINT/374
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) > 0


def test_m397_sensor_spring_normal_snap_rate_fixed(tmp_path: Path):
    c1 = f"{1448:>10d}{3.78e8:>20.4f}{0.1185:>20.4f}"
    deck = f"""# RADIOSS STARTER DECK
/BEGIN
Sensor Spring Normal Snap Rate Fixed Format Test
2022 0
/SENSOR/SPRING_NORMAL_SNAP_RATE/1
Fixed Spring Normal Snap Rate Sensor
{c1}
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 1 in model.sensor_spring_normal_snap_rates
    s1 = model.sensor_spring_normal_snap_rates[1]
    assert s1.spring_id == 1448
    assert pytest.approx(s1.jnorm_snp_max) == 3.78e8
    assert pytest.approx(s1.jnorm_shot_max) == 3.78e8
    assert pytest.approx(s1.jnorm_drop_max) == 3.78e8
    assert pytest.approx(s1.jnorm_lock_max) == 3.78e8
    assert pytest.approx(s1.jnorm_pop_max) == 3.78e8
    assert pytest.approx(s1.jnorm_crackle_max) == 3.78e8
    assert pytest.approx(s1.t_delay) == 0.1185
    assert len(model.sensors) == 1
    assert model.sensors[0].kind == "SPRING_NORMAL_SNAP_RATE"


def test_m397_sensor_spring_normal_snap_rate_free(tmp_path: Path):
    deck = """# RADIOSS FREE DECK
/BEGIN
Sensor Spring Normal Snap Rate Free Format Test
/SENSOR/SPRING_NORMAL_SNAP_RATE/2
Free Spring Normal Snap Rate Sensor
1449, 4.38e8, 0.1695
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 2 in model.sensor_spring_normal_snap_rates
    s2 = model.sensor_spring_normal_snap_rates[2]
    assert s2.spring_id == 1449
    assert pytest.approx(s2.jnorm_snp_max) == 4.38e8
    assert pytest.approx(s2.t_delay) == 0.1695


def test_m397_sensor_spring_normal_snap_rate_aliases(tmp_path: Path):
    deck = """# RADIOSS ALIAS DECK
/BEGIN
Sensor Spring Normal Snap Rate Aliases Test
/SENSOR/SPRING_NORM_SNAP_RATE/391
1469, 1.995e8, 0.0865
/SENSOR/SPRING_RATE_SNAP_NORM/392
1470, 2.015e8, 0.0875
/SENSOR/NORMAL_SNAP_RATE_SPRING/393
1471, 2.035e8, 0.0885
/SENSOR/SPRING_SNAP_NORM/394
1472, 2.055e8, 0.0895
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    for sid in range(391, 395):
        assert sid in model.sensor_spring_normal_snap_rates


def test_m397_sensor_spring_normal_snap_rate_missing_card(tmp_path: Path):
    deck = """# RADIOSS MISSING CARD DECK
/BEGIN
/SENSOR/SPRING_NORMAL_SNAP_RATE/395
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) > 0
