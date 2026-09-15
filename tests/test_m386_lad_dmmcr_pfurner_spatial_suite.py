"""Tests for Milestone M386: LadDynamicMatrixMicroCrushingRate Failure Model, EngFlexomagnetophononicexcitonicmagnonicpolaritonicResonanceEnergy, PfurnerSpatialLinkageJoint, and SensorSpringTransverseDropRate."""

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


def test_m386_fail_lad_dynamic_matrix_micro_crushing_rate_fixed(tmp_path: Path):
    c1 = f"{305.0:>20.4f}{915.0:>20.4f}{145.0:>20.4f}{4.55:>20.4f}{0.945:>20.4f}"
    c2 = f"{1:>10d}{2:>10d}"
    c3 = f"{2000:>10d}"
    deck = f"""# RADIOSS STARTER DECK
/BEGIN
Ladeveze Dynamic Matrix Micro-Crushing Rate Fixed Format Test
2022 0
/FAIL/LAD_DYNAMIC_MATRIX_MICRO_CRUSHING_RATE/2000
Ladeveze Dynamic Matrix Micro-Crushing Failure Model
{c1}
{c2}
{c3}
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 2000 in model.fail_laddynamicmatrixmicrocrushingrates
    fdmmcr = model.fail_laddynamicmatrixmicrocrushingrates[2000]
    assert pytest.approx(fdmmcr.sigma_dmmcr0) == 305.0
    assert pytest.approx(fdmmcr.sigma_dmmcrc) == 915.0
    assert pytest.approx(fdmmcr.gamma_dmmcr) == 145.0
    assert pytest.approx(fdmmcr.p_dmmcr) == 4.55
    assert pytest.approx(fdmmcr.d_dmmcr_max) == 0.945
    assert fdmmcr.ifail_sh == 1
    assert fdmmcr.ifail_so == 2
    assert fdmmcr.fail_id == 2000
    assert len(model.raw_fails) == 1
    assert model.raw_fails[0][1].type == "LAD_DYNAMIC_MATRIX_MICRO_CRUSHING_RATE"


def test_m386_fail_lad_dynamic_matrix_micro_crushing_rate_free(tmp_path: Path):
    deck = """# RADIOSS FREE DECK
/BEGIN
Ladeveze Dynamic Matrix Micro-Crushing Rate Free Format Test
/FAIL/LAD_DYNAMIC_MATRIX_MICRO_CRUSHING_RATE/2001
315.0, 945.0, 150.0, 4.75, 0.920
1, 1
2001
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 2001 in model.fail_laddynamicmatrixmicrocrushingrates
    fdmmcr = model.fail_laddynamicmatrixmicrocrushingrates[2001]
    assert pytest.approx(fdmmcr.sigma_dmmcr0) == 315.0
    assert pytest.approx(fdmmcr.sigma_dmmcrc) == 945.0
    assert pytest.approx(fdmmcr.gamma_dmmcr) == 150.0
    assert pytest.approx(fdmmcr.p_dmmcr) == 4.75
    assert pytest.approx(fdmmcr.d_dmmcr_max) == 0.920
    assert fdmmcr.fail_id == 2001


def test_m386_fail_lad_dynamic_matrix_micro_crushing_rate_aliases(tmp_path: Path):
    deck = """# RADIOSS ALIAS DECK
/BEGIN
Ladeveze Dynamic Matrix Micro-Crushing Rate Aliases Test
/FAIL/LADEVEZE_DYNAMIC_MATRIX_MICRO_CRUSHING_RATE/2002
295.0, 885.0, 136.0, 4.25, 0.965
1, 1
/FAIL/LAD_DMMCR/2003
295.0, 885.0, 136.0, 4.25, 0.965
1, 1
/FAIL/LAD_DMMCR_MODEL/2004
295.0, 885.0, 136.0, 4.25, 0.965
1, 1
/FAIL/LAD_DMMCR_LAW/2005
295.0, 885.0, 136.0, 4.25, 0.965
1, 1
/FAIL/LADEVEZE_RATE_DEPENDENT_DYNAMIC_MATRIX_MICRO_CRUSHING/2006
295.0, 885.0, 136.0, 4.25, 0.965
1, 1
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 2002 in model.fail_laddynamicmatrixmicrocrushingrates
    assert 2003 in model.fail_laddynamicmatrixmicrocrushingrates
    assert 2004 in model.fail_laddynamicmatrixmicrocrushingrates
    assert 2005 in model.fail_laddynamicmatrixmicrocrushingrates
    assert 2006 in model.fail_laddynamicmatrixmicrocrushingrates
    assert len(model.raw_fails) == 5


def test_m386_fail_lad_dynamic_matrix_micro_crushing_rate_missing_card(tmp_path: Path):
    deck = """# RADIOSS MISSING CARD DECK
/BEGIN
/FAIL/LAD_DYNAMIC_MATRIX_MICRO_CRUSHING_RATE/2007
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) > 0


def test_m386_eng_flexomagnetophononicexcitonicmagnonicpolaritonic_resonance_energy_fixed(tmp_path: Path):
    c1 = f"{0.0078:>20.4f}{305:>10d}"
    deck = f"""# RADIOSS STARTER DECK
/BEGIN
Engine Flexomagnetophononicexcitonicmagnonicpolaritonic Resonance Energy Fixed Format Test
2022 0
/ENG/FLEXOMAGNETOPHONONICEXCITONICMAGNONICPOLARITONIC_RESONANCE_ENERGY/205
Resonance Energy Output Directive
{c1}
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 205 in model.eng_flexomagnetophononicexcitonicmagnonicpolaritonic_resonance_energies
    eng = model.eng_flexomagnetophononicexcitonicmagnonicpolaritonic_resonance_energies[205]
    assert pytest.approx(eng.dt_fmpempr) == 0.0078
    assert eng.sens_id == 305


def test_m386_eng_flexomagnetophononicexcitonicmagnonicpolaritonic_resonance_energy_free(tmp_path: Path):
    deck = """# RADIOSS FREE DECK
/BEGIN
Engine Flexomagnetophononicexcitonicmagnonicpolaritonic Resonance Energy Free Format Test
/ENG/FLEXOMAGNETOPHONONICEXCITONICMAGNONICPOLARITONIC_RESONANCE_ENERGY/206
0.0088, 306
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 206 in model.eng_flexomagnetophononicexcitonicmagnonicpolaritonic_resonance_energies
    eng = model.eng_flexomagnetophononicexcitonicmagnonicpolaritonic_resonance_energies[206]
    assert pytest.approx(eng.dt_fmpempr) == 0.0088
    assert eng.sens_id == 306


def test_m386_eng_flexomagnetophononicexcitonicmagnonicpolaritonic_resonance_energy_aliases(tmp_path: Path):
    deck = """# RADIOSS ALIAS DECK
/BEGIN
Engine Flexomagnetophononicexcitonicmagnonicpolaritonic Resonance Energy Aliases Test
/ENG/FLEXOMAGNETO_PHONON_EXCITON_MAGNON_POLARITON_RES_WORK/207
0.0098, 307
/ENG/EFLEXOMAGNETOPHONONICEXCITONICMAGNONICPOLARITONICRESONANCE/208
0.0108, 308
/ENG/FLEXOMAGNETOPHONONICEXCITONICMAGNONICPOLARITONIC_RESONANCE_DISSIPATION/209
0.0118, 309
/ENG/EM_FLEXOMAGNETOPHONONICEXCITONICMAGNONICPOLARITONIC_RESONANCE/210
0.0128, 310
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 207 in model.eng_flexomagnetophononicexcitonicmagnonicpolaritonic_resonance_energies
    assert 208 in model.eng_flexomagnetophononicexcitonicmagnonicpolaritonic_resonance_energies
    assert 209 in model.eng_flexomagnetophononicexcitonicmagnonicpolaritonic_resonance_energies
    assert 210 in model.eng_flexomagnetophononicexcitonicmagnonicpolaritonic_resonance_energies


def test_m386_eng_flexomagnetophononicexcitonicmagnonicpolaritonic_resonance_energy_missing_card(tmp_path: Path):
    deck = """# RADIOSS MISSING CARD DECK
/BEGIN
/ENG/FLEXOMAGNETOPHONONICEXCITONICMAGNONICPOLARITONIC_RESONANCE_ENERGY/211
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) > 0


def test_m386_lagmul_pfurner_spatial_linkage_joint_fixed(tmp_path: Path):
    c1 = f"{291:>10d}{292:>10d}{293:>10d}{1.46e7:>20.4f}{33:>10d}{6.5e-5:>20.4e}"
    c2 = f"{98.0:>20.4f}{82.0:>20.4f}{105.0:>20.4f}{42.0:>20.4f}"
    deck = f"""# RADIOSS STARTER DECK
/BEGIN
Pfurner Spatial Linkage Joint Fixed Format Test
2022 0
/LAGMUL/PFURNER_SPATIAL_LINKAGE_JOINT/255
Pfurner Spatial 6R Joint
{c1}
{c2}
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 255 in model.lagmul_pfurner_spatial_linkage_joints
    joint = model.lagmul_pfurner_spatial_linkage_joints[255]
    assert joint.node1 == 291
    assert joint.node2 == 292
    assert joint.node3 == 293
    assert pytest.approx(joint.stiff) == 1.46e7
    assert joint.skew_id == 33
    assert pytest.approx(joint.tol) == 6.5e-5
    assert pytest.approx(joint.link_len_a) == 98.0
    assert pytest.approx(joint.link_len_b) == 82.0
    assert pytest.approx(joint.twist_angle_alpha) == 105.0
    assert pytest.approx(joint.offset_distance_s) == 42.0
    assert pytest.approx(joint.offset_distance_r) == 42.0


def test_m386_lagmul_pfurner_spatial_linkage_joint_free(tmp_path: Path):
    deck = """# RADIOSS FREE DECK
/BEGIN
Pfurner Spatial Linkage Joint Free Format Test
/LAGMUL/PFURNER_SPATIAL_LINKAGE_JOINT/256
391, 392, 393, 1.70e7, 34, 7.5e-5
115.0, 88.0, 125.0, 54.0
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 256 in model.lagmul_pfurner_spatial_linkage_joints
    joint = model.lagmul_pfurner_spatial_linkage_joints[256]
    assert joint.node1 == 391
    assert joint.node2 == 392
    assert joint.node3 == 393
    assert pytest.approx(joint.stiff) == 1.70e7
    assert joint.skew_id == 34
    assert pytest.approx(joint.tol) == 7.5e-5
    assert pytest.approx(joint.link_len_a) == 115.0
    assert pytest.approx(joint.link_len_b) == 88.0
    assert pytest.approx(joint.twist_angle_alpha) == 125.0
    assert pytest.approx(joint.offset_distance_s) == 54.0


def test_m386_lagmul_pfurner_spatial_linkage_joint_aliases(tmp_path: Path):
    deck = """# RADIOSS ALIAS DECK
/BEGIN
Pfurner Spatial Linkage Joint Aliases Test
/PFURNER_SPATIAL_LINKAGE_JOINT/257
491, 492, 493, 1.0e6, 0, 1.0e-6
70.0, 60.0, 70.0, 26.0
/LAGMUL/PFURNER_SPATIAL_LINKAGE/258
491, 492, 493, 1.0e6, 0, 1.0e-6
70.0, 60.0, 70.0, 26.0
/PFURNER_SPATIAL_LINKAGE/259
491, 492, 493, 1.0e6, 0, 1.0e-6
70.0, 60.0, 70.0, 26.0
/PFURNER_SPATIAL_MULTI_LOOP_MECHANISM/260
491, 492, 493, 1.0e6, 0, 1.0e-6
70.0, 60.0, 70.0, 26.0
/PFURNER_SPATIAL_SYMMETRIC_MECHANISM/261
491, 492, 493, 1.0e6, 0, 1.0e-6
70.0, 60.0, 70.0, 26.0
/PFURNER_SPATIAL_6R_MECHANISM/262
491, 492, 493, 1.0e6, 0, 1.0e-6
70.0, 60.0, 70.0, 26.0
/PFURNER_SPATIAL_OVERCONSTRAINED_MECHANISM/263
491, 492, 493, 1.0e6, 0, 1.0e-6
70.0, 60.0, 70.0, 26.0
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    for jid in range(257, 264):
        assert jid in model.lagmul_pfurner_spatial_linkage_joints


def test_m386_lagmul_pfurner_spatial_linkage_joint_missing_card(tmp_path: Path):
    deck = """# RADIOSS MISSING CARD DECK
/BEGIN
/LAGMUL/PFURNER_SPATIAL_LINKAGE_JOINT/264
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) > 0


def test_m386_sensor_spring_transverse_drop_rate_fixed(tmp_path: Path):
    c1 = f"{1338:>10d}{2.58e8:>20.4f}{0.0785:>20.4f}"
    deck = f"""# RADIOSS STARTER DECK
/BEGIN
Sensor Spring Transverse Drop Rate Fixed Format Test
2022 0
/SENSOR/SPRING_TRANSVERSE_DROP_RATE/1
Fixed Spring Transverse Drop Rate Sensor
{c1}
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 1 in model.sensor_spring_transverse_drop_rates
    s1 = model.sensor_spring_transverse_drop_rates[1]
    assert s1.spring_id == 1338
    assert pytest.approx(s1.jtrans_drop_max) == 2.58e8
    assert pytest.approx(s1.jtrans_lock_max) == 2.58e8
    assert pytest.approx(s1.jtrans_pop_max) == 2.58e8
    assert pytest.approx(s1.jtrans_snp_max) == 2.58e8
    assert pytest.approx(s1.jtrans_crackle_max) == 2.58e8
    assert pytest.approx(s1.t_delay) == 0.0785
    assert len(model.sensors) == 1
    assert model.sensors[0].kind == "SPRING_TRANSVERSE_DROP_RATE"


def test_m386_sensor_spring_transverse_drop_rate_free(tmp_path: Path):
    deck = """# RADIOSS FREE DECK
/BEGIN
Sensor Spring Transverse Drop Rate Free Format Test
/SENSOR/SPRING_TRANSVERSE_DROP_RATE/2
Free Spring Transverse Drop Rate Sensor
1339, 3.18e8, 0.0935
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 2 in model.sensor_spring_transverse_drop_rates
    s2 = model.sensor_spring_transverse_drop_rates[2]
    assert s2.spring_id == 1339
    assert pytest.approx(s2.jtrans_drop_max) == 3.18e8
    assert pytest.approx(s2.t_delay) == 0.0935


def test_m386_sensor_spring_transverse_drop_rate_aliases(tmp_path: Path):
    deck = """# RADIOSS ALIAS DECK
/BEGIN
Sensor Spring Transverse Drop Rate Aliases Test
/SENSOR/SPRING_TRANS_DROP_RATE/360
1359, 1.165e8, 0.0165
/SENSOR/SPRING_RATE_DROP_TRANS/361
1360, 1.185e8, 0.0175
/SENSOR/TRANSVERSE_DROP_RATE_SPRING/362
1361, 1.205e8, 0.0185
/SENSOR/SPRING_DROP_TRANS/363
1362, 1.225e8, 0.0195
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    for sid in range(360, 364):
        assert sid in model.sensor_spring_transverse_drop_rates


def test_m386_sensor_spring_transverse_drop_rate_missing_card(tmp_path: Path):
    deck = """# RADIOSS MISSING CARD DECK
/BEGIN
/SENSOR/SPRING_TRANSVERSE_DROP_RATE/365
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) > 0
