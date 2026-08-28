"""Tests for Milestone M398: LadDynamicMatrixCrackingRate Failure Model, EngFlexothermoplasmonicpolaritonicResonanceEnergy, SylvesterSpatialLinkageJoint, and SensorSpringTransverseSnapRate."""

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


def test_m398_fail_lad_dynamic_matrix_cracking_rate_fixed(tmp_path: Path):
    c1 = f"{560.0:>20.4f}{1680.0:>20.4f}{325.0:>20.4f}{7.25:>20.4f}{0.875:>20.4f}"
    c2 = f"{1:>10d}{2:>10d}"
    c3 = f"{2120:>10d}"
    deck = f"""# RADIOSS STARTER DECK
/BEGIN
Ladeveze Dynamic Matrix Cracking Rate Fixed Format Test
2022 0
/FAIL/LAD_DYNAMIC_MATRIX_CRACKING_RATE/2120
Ladeveze Dynamic Matrix Cracking Failure Model
{c1}
{c2}
{c3}
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 2120 in model.fail_laddynamicmatrixcrackingrates
    fdmcr = model.fail_laddynamicmatrixcrackingrates[2120]
    assert pytest.approx(fdmcr.sigma_dmcr0) == 560.0
    assert pytest.approx(fdmcr.sigma_dmcrc) == 1680.0
    assert pytest.approx(fdmcr.gamma_dmcr) == 325.0
    assert pytest.approx(fdmcr.p_dmcr) == 7.25
    assert pytest.approx(fdmcr.d_dmcr_max) == 0.875
    assert fdmcr.ifail_sh == 1
    assert fdmcr.ifail_so == 2
    assert fdmcr.fail_id == 2120
    assert len(model.raw_fails) == 1
    assert model.raw_fails[0][1].type == "LAD_DYNAMIC_MATRIX_CRACKING_RATE"


def test_m398_fail_lad_dynamic_matrix_cracking_rate_free(tmp_path: Path):
    deck = """# RADIOSS FREE DECK
/BEGIN
Ladeveze Dynamic Matrix Cracking Rate Free Format Test
/FAIL/LAD_DYNAMIC_MATRIX_CRACKING_RATE/2121
570.0, 1710.0, 335.0, 7.45, 0.865
1, 1
2121
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 2121 in model.fail_laddynamicmatrixcrackingrates
    fdmcr = model.fail_laddynamicmatrixcrackingrates[2121]
    assert pytest.approx(fdmcr.sigma_dmcr0) == 570.0
    assert pytest.approx(fdmcr.sigma_dmcrc) == 1710.0
    assert pytest.approx(fdmcr.gamma_dmcr) == 335.0
    assert pytest.approx(fdmcr.p_dmcr) == 7.45
    assert pytest.approx(fdmcr.d_dmcr_max) == 0.865
    assert fdmcr.fail_id == 2121


def test_m398_fail_lad_dynamic_matrix_cracking_rate_aliases(tmp_path: Path):
    deck = """# RADIOSS ALIAS DECK
/BEGIN
Ladeveze Dynamic Matrix Cracking Rate Aliases Test
/FAIL/LADEVEZE_DYNAMIC_MATRIX_CRACKING_RATE/2122
550.0, 1650.0, 320.0, 7.05, 0.895
1, 1
/FAIL/LAD_DMCR/2123
550.0, 1650.0, 320.0, 7.05, 0.895
1, 1
/FAIL/LAD_DMCR_MODEL/2124
550.0, 1650.0, 320.0, 7.05, 0.895
1, 1
/FAIL/LAD_DMCR_LAW/2125
550.0, 1650.0, 320.0, 7.05, 0.895
1, 1
/FAIL/LADEVEZE_RATE_DEPENDENT_DYNAMIC_MATRIX_CRACKING/2126
550.0, 1650.0, 320.0, 7.05, 0.895
1, 1
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 2122 in model.fail_laddynamicmatrixcrackingrates
    assert 2123 in model.fail_laddynamicmatrixcrackingrates
    assert 2124 in model.fail_laddynamicmatrixcrackingrates
    assert 2125 in model.fail_laddynamicmatrixcrackingrates
    assert 2126 in model.fail_laddynamicmatrixcrackingrates
    assert len(model.raw_fails) == 5


def test_m398_fail_lad_dynamic_matrix_cracking_rate_missing_card(tmp_path: Path):
    deck = """# RADIOSS MISSING CARD DECK
/BEGIN
/FAIL/LAD_DYNAMIC_MATRIX_CRACKING_RATE/2127
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) > 0


def test_m398_eng_flexothermoplasmonicpolaritonic_resonance_energy_fixed(tmp_path: Path):
    c1 = f"{0.0205:>20.4f}{425:>10d}"
    deck = f"""# RADIOSS STARTER DECK
/BEGIN
Engine Flexothermoplasmonicpolaritonic Resonance Energy Fixed Format Test
2022 0
/ENG/FLEXOTHERMOPLASMONICPOLARITONIC_RESONANCE_ENERGY/325
Resonance Energy Output Directive
{c1}
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 325 in model.eng_flexothermoplasmonicpolaritonic_resonance_energies
    eng = model.eng_flexothermoplasmonicpolaritonic_resonance_energies[325]
    assert pytest.approx(eng.dt_ftpp) == 0.0205
    assert eng.sens_id == 425


def test_m398_eng_flexothermoplasmonicpolaritonic_resonance_energy_free(tmp_path: Path):
    deck = """# RADIOSS FREE DECK
/BEGIN
Engine Flexothermoplasmonicpolaritonic Resonance Energy Free Format Test
/ENG/FLEXOTHERMOPLASMONICPOLARITONIC_RESONANCE_ENERGY/326
0.0215, 426
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 326 in model.eng_flexothermoplasmonicpolaritonic_resonance_energies
    eng = model.eng_flexothermoplasmonicpolaritonic_resonance_energies[326]
    assert pytest.approx(eng.dt_ftpp) == 0.0215
    assert eng.sens_id == 426


def test_m398_eng_flexothermoplasmonicpolaritonic_resonance_energy_aliases(tmp_path: Path):
    deck = """# RADIOSS ALIAS DECK
/BEGIN
Engine Flexothermoplasmonicpolaritonic Resonance Energy Aliases Test
/ENG/FLEXOTHERM_PLASMONIC_POLARITON_RES_WORK/327
0.0225, 427
/ENG/EFLEXOTHERMOPLASMONICPOLARITONICRESONANCE/328
0.0235, 428
/ENG/FLEXOTHERMOPLASMONICPOLARITONIC_RESONANCE_DISSIPATION/329
0.0245, 429
/ENG/ET_FLEXOTHERMOPLASMONICPOLARITONIC_RESONANCE/330
0.0255, 430
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 327 in model.eng_flexothermoplasmonicpolaritonic_resonance_energies
    assert 328 in model.eng_flexothermoplasmonicpolaritonic_resonance_energies
    assert 329 in model.eng_flexothermoplasmonicpolaritonic_resonance_energies
    assert 330 in model.eng_flexothermoplasmonicpolaritonic_resonance_energies


def test_m398_eng_flexothermoplasmonicpolaritonic_resonance_energy_missing_card(tmp_path: Path):
    deck = """# RADIOSS MISSING CARD DECK
/BEGIN
/ENG/FLEXOTHERMOPLASMONICPOLARITONIC_RESONANCE_ENERGY/331
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) > 0


def test_m398_lagmul_sylvester_spatial_linkage_joint_fixed(tmp_path: Path):
    c1 = f"{411:>10d}{412:>10d}{413:>10d}{2.66e7:>20.4f}{129:>10d}{1.65e-4:>20.4e}"
    c2 = f"{195.0:>20.4f}{178.0:>20.4f}{205.0:>20.4f}{128.0:>20.4f}"
    deck = f"""# RADIOSS STARTER DECK
/BEGIN
Sylvester Spatial Linkage Joint Fixed Format Test
2022 0
/LAGMUL/SYLVESTER_SPATIAL_LINKAGE_JOINT/375
Sylvester Spatial 6R Joint
{c1}
{c2}
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 375 in model.lagmul_sylvester_spatial_linkage_joints
    joint = model.lagmul_sylvester_spatial_linkage_joints[375]
    assert joint.node1 == 411
    assert joint.node2 == 412
    assert joint.node3 == 413
    assert pytest.approx(joint.stiff) == 2.66e7
    assert joint.skew_id == 129
    assert pytest.approx(joint.tol) == 1.65e-4
    assert pytest.approx(joint.link_len_a) == 195.0
    assert pytest.approx(joint.link_len_b) == 178.0
    assert pytest.approx(joint.twist_angle_alpha) == 205.0
    assert pytest.approx(joint.offset_distance_s) == 128.0
    assert pytest.approx(joint.offset_distance_r) == 128.0
    assert pytest.approx(joint.offset_distance_v) == 128.0
    assert pytest.approx(joint.offset_distance_h) == 128.0
    assert pytest.approx(joint.offset_distance_u) == 128.0


def test_m398_lagmul_sylvester_spatial_linkage_joint_free(tmp_path: Path):
    deck = """# RADIOSS FREE DECK
/BEGIN
Sylvester Spatial Linkage Joint Free Format Test
/LAGMUL/SYLVESTER_SPATIAL_LINKAGE_JOINT/376
511, 512, 513, 2.90e7, 130, 1.75e-4
215.0, 185.0, 225.0, 140.0
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 376 in model.lagmul_sylvester_spatial_linkage_joints
    joint = model.lagmul_sylvester_spatial_linkage_joints[376]
    assert joint.node1 == 511
    assert joint.node2 == 512
    assert joint.node3 == 513
    assert pytest.approx(joint.stiff) == 2.90e7
    assert joint.skew_id == 130
    assert pytest.approx(joint.tol) == 1.75e-4
    assert pytest.approx(joint.link_len_a) == 215.0
    assert pytest.approx(joint.link_len_b) == 185.0
    assert pytest.approx(joint.twist_angle_alpha) == 225.0
    assert pytest.approx(joint.offset_distance_s) == 140.0


def test_m398_lagmul_sylvester_spatial_linkage_joint_aliases(tmp_path: Path):
    deck = """# RADIOSS ALIAS DECK
/BEGIN
Sylvester Spatial Linkage Joint Aliases Test
/SYLVESTER_SPATIAL_LINKAGE_JOINT/377
611, 612, 613, 1.0e6, 0, 1.0e-6
155.0, 145.0, 155.0, 105.0
/LAGMUL/SYLVESTER_SPATIAL_LINKAGE/378
611, 612, 613, 1.0e6, 0, 1.0e-6
155.0, 145.0, 155.0, 105.0
/SYLVESTER_SPATIAL_LINKAGE/379
611, 612, 613, 1.0e6, 0, 1.0e-6
155.0, 145.0, 155.0, 105.0
/SYLVESTER_SPATIAL_MULTI_LOOP_MECHANISM/380
611, 612, 613, 1.0e6, 0, 1.0e-6
155.0, 145.0, 155.0, 105.0
/SYLVESTER_SPATIAL_SYMMETRIC_MECHANISM/381
611, 612, 613, 1.0e6, 0, 1.0e-6
155.0, 145.0, 155.0, 105.0
/SYLVESTER_SPATIAL_6R_MECHANISM/382
611, 612, 613, 1.0e6, 0, 1.0e-6
155.0, 145.0, 155.0, 105.0
/SYLVESTER_SPATIAL_OVERCONSTRAINED_MECHANISM/383
611, 612, 613, 1.0e6, 0, 1.0e-6
155.0, 145.0, 155.0, 105.0
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    for jid in range(377, 384):
        assert jid in model.lagmul_sylvester_spatial_linkage_joints


def test_m398_lagmul_sylvester_spatial_linkage_joint_missing_card(tmp_path: Path):
    deck = """# RADIOSS MISSING CARD DECK
/BEGIN
/LAGMUL/SYLVESTER_SPATIAL_LINKAGE_JOINT/384
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) > 0


def test_m398_sensor_spring_transverse_snap_rate_fixed(tmp_path: Path):
    c1 = f"{1458:>10d}{3.88e8:>20.4f}{0.1225:>20.4f}"
    deck = f"""# RADIOSS STARTER DECK
/BEGIN
Sensor Spring Transverse Snap Rate Fixed Format Test
2022 0
/SENSOR/SPRING_TRANSVERSE_SNAP_RATE/1
Fixed Spring Transverse Snap Rate Sensor
{c1}
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 1 in model.sensor_spring_transverse_snap_rates
    s1 = model.sensor_spring_transverse_snap_rates[1]
    assert s1.spring_id == 1458
    assert pytest.approx(s1.jtrans_snp_max) == 3.88e8
    assert pytest.approx(s1.jtrans_shot_max) == 3.88e8
    assert pytest.approx(s1.jtrans_drop_max) == 3.88e8
    assert pytest.approx(s1.jtrans_lock_max) == 3.88e8
    assert pytest.approx(s1.jtrans_pop_max) == 3.88e8
    assert pytest.approx(s1.jtrans_crackle_max) == 3.88e8
    assert pytest.approx(s1.t_delay) == 0.1225
    assert len(model.sensors) == 1
    assert model.sensors[0].kind == "SPRING_TRANSVERSE_SNAP_RATE"


def test_m398_sensor_spring_transverse_snap_rate_free(tmp_path: Path):
    deck = """# RADIOSS FREE DECK
/BEGIN
Sensor Spring Transverse Snap Rate Free Format Test
/SENSOR/SPRING_TRANSVERSE_SNAP_RATE/2
Free Spring Transverse Snap Rate Sensor
1459, 4.48e8, 0.1795
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 2 in model.sensor_spring_transverse_snap_rates
    s2 = model.sensor_spring_transverse_snap_rates[2]
    assert s2.spring_id == 1459
    assert pytest.approx(s2.jtrans_snp_max) == 4.48e8
    assert pytest.approx(s2.t_delay) == 0.1795


def test_m398_sensor_spring_transverse_snap_rate_aliases(tmp_path: Path):
    deck = """# RADIOSS ALIAS DECK
/BEGIN
Sensor Spring Transverse Snap Rate Aliases Test
/SENSOR/SPRING_TRANS_SNAP_RATE/391
1479, 2.095e8, 0.0965
/SENSOR/SPRING_RATE_SNAP_TRANS/392
1480, 2.115e8, 0.0975
/SENSOR/TRANSVERSE_SNAP_RATE_SPRING/393
1481, 2.135e8, 0.0985
/SENSOR/SPRING_SNAP_TRANS/394
1482, 2.155e8, 0.0995
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    for sid in range(391, 395):
        assert sid in model.sensor_spring_transverse_snap_rates


def test_m398_sensor_spring_transverse_snap_rate_missing_card(tmp_path: Path):
    deck = """# RADIOSS MISSING CARD DECK
/BEGIN
/SENSOR/SPRING_TRANSVERSE_SNAP_RATE/395
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) > 0
