"""Tests for Milestone M402: LadTransverseFiberKinkingRate Failure Model, EngFlexothermoplasmonicphononicpolaritonicResonanceEnergy, FermatSpatialLinkageJoint, and SensorSpringTotalAngularSnapRate."""

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


def test_m402_fail_lad_transverse_fiber_kinking_rate_fixed(tmp_path: Path):
    c1 = f"{600.0:>20.4f}{1800.0:>20.4f}{365.0:>20.4f}{7.65:>20.4f}{0.835:>20.4f}"
    c2 = f"{1:>10d}{2:>10d}"
    c3 = f"{2160:>10d}"
    deck = f"""# RADIOSS STARTER DECK
/BEGIN
Ladeveze Transverse Fiber Kinking Rate Fixed Format Test
2022 0
/FAIL/LAD_TRANSVERSE_FIBER_KINKING_RATE/2160
Ladeveze Transverse Fiber Kinking Failure Model
{c1}
{c2}
{c3}
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 2160 in model.fail_ladtransversefiberkinkingrates
    ftfkr = model.fail_ladtransversefiberkinkingrates[2160]
    assert pytest.approx(ftfkr.sigma_tfkr0) == 600.0
    assert pytest.approx(ftfkr.sigma_tfkrc) == 1800.0
    assert pytest.approx(ftfkr.gamma_tfkr) == 365.0
    assert pytest.approx(ftfkr.p_tfkr) == 7.65
    assert pytest.approx(ftfkr.d_tfkr_max) == 0.835
    assert ftfkr.ifail_sh == 1
    assert ftfkr.ifail_so == 2
    assert ftfkr.fail_id == 2160
    assert len(model.raw_fails) == 1
    assert model.raw_fails[0][1].type == "LAD_TRANSVERSE_FIBER_KINKING_RATE"


def test_m402_fail_lad_transverse_fiber_kinking_rate_free(tmp_path: Path):
    deck = """# RADIOSS FREE DECK
/BEGIN
Ladeveze Transverse Fiber Kinking Rate Free Format Test
/FAIL/LAD_TRANSVERSE_FIBER_KINKING_RATE/2161
610.0, 1830.0, 375.0, 7.85, 0.825
1, 1
2161
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 2161 in model.fail_ladtransversefiberkinkingrates
    ftfkr = model.fail_ladtransversefiberkinkingrates[2161]
    assert pytest.approx(ftfkr.sigma_tfkr0) == 610.0
    assert pytest.approx(ftfkr.sigma_tfkrc) == 1830.0
    assert pytest.approx(ftfkr.gamma_tfkr) == 375.0
    assert pytest.approx(ftfkr.p_tfkr) == 7.85
    assert pytest.approx(ftfkr.d_tfkr_max) == 0.825
    assert ftfkr.fail_id == 2161


def test_m402_fail_lad_transverse_fiber_kinking_rate_aliases(tmp_path: Path):
    deck = """# RADIOSS ALIAS DECK
/BEGIN
Ladeveze Transverse Fiber Kinking Rate Aliases Test
/FAIL/LADEVEZE_TRANSVERSE_FIBER_KINKING_RATE/2162
590.0, 1770.0, 360.0, 7.45, 0.855
1, 1
/FAIL/LAD_TFKR/2163
590.0, 1770.0, 360.0, 7.45, 0.855
1, 1
/FAIL/LAD_TFKR_MODEL/2164
590.0, 1770.0, 360.0, 7.45, 0.855
1, 1
/FAIL/LAD_TFKR_LAW/2165
590.0, 1770.0, 360.0, 7.45, 0.855
1, 1
/FAIL/LADEVEZE_RATE_DEPENDENT_TRANSVERSE_FIBER_KINKING/2166
590.0, 1770.0, 360.0, 7.45, 0.855
1, 1
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 2162 in model.fail_ladtransversefiberkinkingrates
    assert 2163 in model.fail_ladtransversefiberkinkingrates
    assert 2164 in model.fail_ladtransversefiberkinkingrates
    assert 2165 in model.fail_ladtransversefiberkinkingrates
    assert 2166 in model.fail_ladtransversefiberkinkingrates
    assert len(model.raw_fails) == 5


def test_m402_fail_lad_transverse_fiber_kinking_rate_missing_card(tmp_path: Path):
    deck = """# RADIOSS MISSING CARD DECK
/BEGIN
/FAIL/LAD_TRANSVERSE_FIBER_KINKING_RATE/2167
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) > 0


def test_m402_eng_flexothermoplasmonicphononicpolaritonic_resonance_energy_fixed(tmp_path: Path):
    c1 = f"{0.0245:>20.4f}{465:>10d}"
    deck = f"""# RADIOSS STARTER DECK
/BEGIN
Engine Flexothermoplasmonicphononicpolaritonic Resonance Energy Fixed Format Test
2022 0
/ENG/FLEXOTHERMOPLASMONICPHONONICPOLARITONIC_RESONANCE_ENERGY/365
Resonance Energy Output Directive
{c1}
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 365 in model.eng_flexothermoplasmonicphononicpolaritonic_resonance_energies
    eng = model.eng_flexothermoplasmonicphononicpolaritonic_resonance_energies[365]
    assert pytest.approx(eng.dt_ftppp) == 0.0245
    assert eng.sens_id == 465


def test_m402_eng_flexothermoplasmonicphononicpolaritonic_resonance_energy_free(tmp_path: Path):
    deck = """# RADIOSS FREE DECK
/BEGIN
Engine Flexothermoplasmonicphononicpolaritonic Resonance Energy Free Format Test
/ENG/FLEXOTHERMOPLASMONICPHONONICPOLARITONIC_RESONANCE_ENERGY/366
0.0255, 466
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 366 in model.eng_flexothermoplasmonicphononicpolaritonic_resonance_energies
    eng = model.eng_flexothermoplasmonicphononicpolaritonic_resonance_energies[366]
    assert pytest.approx(eng.dt_ftppp) == 0.0255
    assert eng.sens_id == 466


def test_m402_eng_flexothermoplasmonicphononicpolaritonic_resonance_energy_aliases(tmp_path: Path):
    deck = """# RADIOSS ALIAS DECK
/BEGIN
Engine Flexothermoplasmonicphononicpolaritonic Resonance Energy Aliases Test
/ENG/FLEXOTHERM_PLASMONIC_PHONONIC_POLARITON_RES_WORK/367
0.0265, 467
/ENG/EFLEXOTHERMOPLASMONICPHONONICPOLARITONICRESONANCE/368
0.0275, 468
/ENG/FLEXOTHERMOPLASMONICPHONONICPOLARITONIC_RESONANCE_DISSIPATION/369
0.0285, 469
/ENG/ET_FLEXOTHERMOPLASMONICPHONONICPOLARITONIC_RESONANCE/370
0.0295, 470
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 367 in model.eng_flexothermoplasmonicphononicpolaritonic_resonance_energies
    assert 368 in model.eng_flexothermoplasmonicphononicpolaritonic_resonance_energies
    assert 369 in model.eng_flexothermoplasmonicphononicpolaritonic_resonance_energies
    assert 370 in model.eng_flexothermoplasmonicphononicpolaritonic_resonance_energies


def test_m402_eng_flexothermoplasmonicphononicpolaritonic_resonance_energy_missing_card(tmp_path: Path):
    deck = """# RADIOSS MISSING CARD DECK
/BEGIN
/ENG/FLEXOTHERMOPLASMONICPHONONICPOLARITONIC_RESONANCE_ENERGY/371
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) > 0


def test_m402_lagmul_fermat_spatial_linkage_joint_fixed(tmp_path: Path):
    c1 = f"{451:>10d}{452:>10d}{453:>10d}{3.06e7:>20.4f}{169:>10d}{2.05e-4:>20.4e}"
    c2 = f"{235.0:>20.4f}{218.0:>20.4f}{245.0:>20.4f}{168.0:>20.4f}"
    deck = f"""# RADIOSS STARTER DECK
/BEGIN
Fermat Spatial Linkage Joint Fixed Format Test
2022 0
/LAGMUL/FERMAT_SPATIAL_LINKAGE_JOINT/415
Fermat Spatial 6R Joint
{c1}
{c2}
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 415 in model.lagmul_fermat_spatial_linkage_joints
    joint = model.lagmul_fermat_spatial_linkage_joints[415]
    assert joint.node1 == 451
    assert joint.node2 == 452
    assert joint.node3 == 453
    assert pytest.approx(joint.stiff) == 3.06e7
    assert joint.skew_id == 169
    assert pytest.approx(joint.tol) == 2.05e-4
    assert pytest.approx(joint.link_len_a) == 235.0
    assert pytest.approx(joint.link_len_b) == 218.0
    assert pytest.approx(joint.twist_angle_alpha) == 245.0
    assert pytest.approx(joint.offset_distance_s) == 168.0
    assert pytest.approx(joint.offset_distance_r) == 168.0
    assert pytest.approx(joint.offset_distance_v) == 168.0
    assert pytest.approx(joint.offset_distance_h) == 168.0
    assert pytest.approx(joint.offset_distance_u) == 168.0


def test_m402_lagmul_fermat_spatial_linkage_joint_free(tmp_path: Path):
    deck = """# RADIOSS FREE DECK
/BEGIN
Fermat Spatial Linkage Joint Free Format Test
/LAGMUL/FERMAT_SPATIAL_LINKAGE_JOINT/416
551, 552, 553, 3.30e7, 170, 2.15e-4
255.0, 225.0, 265.0, 180.0
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 416 in model.lagmul_fermat_spatial_linkage_joints
    joint = model.lagmul_fermat_spatial_linkage_joints[416]
    assert joint.node1 == 551
    assert joint.node2 == 552
    assert joint.node3 == 553
    assert pytest.approx(joint.stiff) == 3.30e7
    assert joint.skew_id == 170
    assert pytest.approx(joint.tol) == 2.15e-4
    assert pytest.approx(joint.link_len_a) == 255.0
    assert pytest.approx(joint.link_len_b) == 225.0
    assert pytest.approx(joint.twist_angle_alpha) == 265.0
    assert pytest.approx(joint.offset_distance_s) == 180.0


def test_m402_lagmul_fermat_spatial_linkage_joint_aliases(tmp_path: Path):
    deck = """# RADIOSS ALIAS DECK
/BEGIN
Fermat Spatial Linkage Joint Aliases Test
/FERMAT_SPATIAL_LINKAGE_JOINT/417
651, 652, 653, 1.0e6, 0, 1.0e-6
195.0, 185.0, 195.0, 145.0
/LAGMUL/FERMAT_SPATIAL_LINKAGE/418
651, 652, 653, 1.0e6, 0, 1.0e-6
195.0, 185.0, 195.0, 145.0
/FERMAT_SPATIAL_LINKAGE/419
651, 652, 653, 1.0e6, 0, 1.0e-6
195.0, 185.0, 195.0, 145.0
/FERMAT_SPATIAL_MULTI_LOOP_MECHANISM/420
651, 652, 653, 1.0e6, 0, 1.0e-6
195.0, 185.0, 195.0, 145.0
/FERMAT_SPATIAL_SYMMETRIC_MECHANISM/421
651, 652, 653, 1.0e6, 0, 1.0e-6
195.0, 185.0, 195.0, 145.0
/FERMAT_SPATIAL_6R_MECHANISM/422
651, 652, 653, 1.0e6, 0, 1.0e-6
195.0, 185.0, 195.0, 145.0
/FERMAT_SPATIAL_OVERCONSTRAINED_MECHANISM/423
651, 652, 653, 1.0e6, 0, 1.0e-6
195.0, 185.0, 195.0, 145.0
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    for jid in range(417, 424):
        assert jid in model.lagmul_fermat_spatial_linkage_joints


def test_m402_lagmul_fermat_spatial_linkage_joint_missing_card(tmp_path: Path):
    deck = """# RADIOSS MISSING CARD DECK
/BEGIN
/LAGMUL/FERMAT_SPATIAL_LINKAGE_JOINT/424
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) > 0


def test_m402_sensor_spring_total_angular_snap_rate_fixed(tmp_path: Path):
    c1 = f"{1498:>10d}{4.28e8:>20.4f}{0.1385:>20.4f}"
    deck = f"""# RADIOSS STARTER DECK
/BEGIN
Sensor Spring Total Angular Snap Rate Fixed Format Test
2022 0
/SENSOR/SPRING_TOTAL_ANGULAR_SNAP_RATE/1
Fixed Spring Total Angular Snap Rate Sensor
{c1}
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 1 in model.sensor_spring_total_angular_snap_rates
    s1 = model.sensor_spring_total_angular_snap_rates[1]
    assert s1.spring_id == 1498
    assert pytest.approx(s1.jtot_ang_snp_max) == 4.28e8
    assert pytest.approx(s1.jtot_ang_shot_max) == 4.28e8
    assert pytest.approx(s1.jtot_ang_drop_max) == 4.28e8
    assert pytest.approx(s1.jtot_ang_lock_max) == 4.28e8
    assert pytest.approx(s1.jtot_ang_pop_max) == 4.28e8
    assert pytest.approx(s1.jtot_ang_crackle_max) == 4.28e8
    assert pytest.approx(s1.t_delay) == 0.1385
    assert len(model.sensors) == 1
    assert model.sensors[0].kind == "SPRING_TOTAL_ANGULAR_SNAP_RATE"


def test_m402_sensor_spring_total_angular_snap_rate_free(tmp_path: Path):
    deck = """# RADIOSS FREE DECK
/BEGIN
Sensor Spring Total Angular Snap Rate Free Format Test
/SENSOR/SPRING_TOTAL_ANGULAR_SNAP_RATE/2
Free Spring Total Angular Snap Rate Sensor
1499, 4.88e8, 0.2195
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 2 in model.sensor_spring_total_angular_snap_rates
    s2 = model.sensor_spring_total_angular_snap_rates[2]
    assert s2.spring_id == 1499
    assert pytest.approx(s2.jtot_ang_snp_max) == 4.88e8
    assert pytest.approx(s2.t_delay) == 0.2195


def test_m402_sensor_spring_total_angular_snap_rate_aliases(tmp_path: Path):
    deck = """# RADIOSS ALIAS DECK
/BEGIN
Sensor Spring Total Angular Snap Rate Aliases Test
/SENSOR/SPRING_TOT_ANG_SNAP_RATE/391
1519, 2.495e8, 0.1365
/SENSOR/SPRING_RATE_SNAP_ANG_TOT/392
1520, 2.515e8, 0.1375
/SENSOR/TOTAL_ANGULAR_SNAP_RATE_SPRING/393
1521, 2.535e8, 0.1385
/SENSOR/SPRING_SNAP_ANG_TOT/394
1522, 2.555e8, 0.1395
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    for sid in range(391, 395):
        assert sid in model.sensor_spring_total_angular_snap_rates


def test_m402_sensor_spring_total_angular_snap_rate_missing_card(tmp_path: Path):
    deck = """# RADIOSS MISSING CARD DECK
/BEGIN
/SENSOR/SPRING_TOTAL_ANGULAR_SNAP_RATE/395
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) > 0
