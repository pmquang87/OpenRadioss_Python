"""Tests for Milestone M387: LadTransverseMatrixMicroCrushingRate Failure Model, EngFlexomagnetophononicplasmonicexcitonicResonanceEnergy, PhillipsSpatialLinkageJoint, and SensorSpringTotalDropRate."""

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


def test_m387_fail_lad_transverse_matrix_micro_crushing_rate_fixed(tmp_path: Path):
    c1 = f"{315.0:>20.4f}{945.0:>20.4f}{155.0:>20.4f}{4.65:>20.4f}{0.955:>20.4f}"
    c2 = f"{1:>10d}{2:>10d}"
    c3 = f"{2010:>10d}"
    deck = f"""# RADIOSS STARTER DECK
/BEGIN
Ladeveze Transverse Matrix Micro-Crushing Rate Fixed Format Test
2022 0
/FAIL/LAD_TRANSVERSE_MATRIX_MICRO_CRUSHING_RATE/2010
Ladeveze Transverse Matrix Micro-Crushing Failure Model
{c1}
{c2}
{c3}
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 2010 in model.fail_ladtransversematrixmicrocrushingrates
    ftmmcr = model.fail_ladtransversematrixmicrocrushingrates[2010]
    assert pytest.approx(ftmmcr.sigma_tmmcr0) == 315.0
    assert pytest.approx(ftmmcr.sigma_tmmcrc) == 945.0
    assert pytest.approx(ftmmcr.gamma_tmmcr) == 155.0
    assert pytest.approx(ftmmcr.p_tmmcr) == 4.65
    assert pytest.approx(ftmmcr.d_tmmcr_max) == 0.955
    assert ftmmcr.ifail_sh == 1
    assert ftmmcr.ifail_so == 2
    assert ftmmcr.fail_id == 2010
    assert len(model.raw_fails) == 1
    assert model.raw_fails[0][1].type == "LAD_TRANSVERSE_MATRIX_MICRO_CRUSHING_RATE"


def test_m387_fail_lad_transverse_matrix_micro_crushing_rate_free(tmp_path: Path):
    deck = """# RADIOSS FREE DECK
/BEGIN
Ladeveze Transverse Matrix Micro-Crushing Rate Free Format Test
/FAIL/LAD_TRANSVERSE_MATRIX_MICRO_CRUSHING_RATE/2011
325.0, 975.0, 160.0, 4.85, 0.930
1, 1
2011
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 2011 in model.fail_ladtransversematrixmicrocrushingrates
    ftmmcr = model.fail_ladtransversematrixmicrocrushingrates[2011]
    assert pytest.approx(ftmmcr.sigma_tmmcr0) == 325.0
    assert pytest.approx(ftmmcr.sigma_tmmcrc) == 975.0
    assert pytest.approx(ftmmcr.gamma_tmmcr) == 160.0
    assert pytest.approx(ftmmcr.p_tmmcr) == 4.85
    assert pytest.approx(ftmmcr.d_tmmcr_max) == 0.930
    assert ftmmcr.fail_id == 2011


def test_m387_fail_lad_transverse_matrix_micro_crushing_rate_aliases(tmp_path: Path):
    deck = """# RADIOSS ALIAS DECK
/BEGIN
Ladeveze Transverse Matrix Micro-Crushing Rate Aliases Test
/FAIL/LADEVEZE_TRANSVERSE_MATRIX_MICRO_CRUSHING_RATE/2012
305.0, 915.0, 146.0, 4.35, 0.975
1, 1
/FAIL/LAD_TMMCR/2013
305.0, 915.0, 146.0, 4.35, 0.975
1, 1
/FAIL/LAD_TMMCR_MODEL/2014
305.0, 915.0, 146.0, 4.35, 0.975
1, 1
/FAIL/LAD_TMMCR_LAW/2015
305.0, 915.0, 146.0, 4.35, 0.975
1, 1
/FAIL/LADEVEZE_RATE_DEPENDENT_TRANSVERSE_MATRIX_MICRO_CRUSHING/2016
305.0, 915.0, 146.0, 4.35, 0.975
1, 1
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 2012 in model.fail_ladtransversematrixmicrocrushingrates
    assert 2013 in model.fail_ladtransversematrixmicrocrushingrates
    assert 2014 in model.fail_ladtransversematrixmicrocrushingrates
    assert 2015 in model.fail_ladtransversematrixmicrocrushingrates
    assert 2016 in model.fail_ladtransversematrixmicrocrushingrates
    assert len(model.raw_fails) == 5


def test_m387_fail_lad_transverse_matrix_micro_crushing_rate_missing_card(tmp_path: Path):
    deck = """# RADIOSS MISSING CARD DECK
/BEGIN
/FAIL/LAD_TRANSVERSE_MATRIX_MICRO_CRUSHING_RATE/2017
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) > 0


def test_m387_eng_flexomagnetophononicplasmonicexcitonic_resonance_energy_fixed(tmp_path: Path):
    c1 = f"{0.0082:>20.4f}{315:>10d}"
    deck = f"""# RADIOSS STARTER DECK
/BEGIN
Engine Flexomagnetophononicplasmonicexcitonic Resonance Energy Fixed Format Test
2022 0
/ENG/FLEXOMAGNETOPHONONICPLASMONICEXCITONIC_RESONANCE_ENERGY/215
Resonance Energy Output Directive
{c1}
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 215 in model.eng_flexomagnetophononicplasmonicexcitonic_resonance_energies
    eng = model.eng_flexomagnetophononicplasmonicexcitonic_resonance_energies[215]
    assert pytest.approx(eng.dt_fmppre) == 0.0082
    assert eng.sens_id == 315


def test_m387_eng_flexomagnetophononicplasmonicexcitonic_resonance_energy_free(tmp_path: Path):
    deck = """# RADIOSS FREE DECK
/BEGIN
Engine Flexomagnetophononicplasmonicexcitonic Resonance Energy Free Format Test
/ENG/FLEXOMAGNETOPHONONICPLASMONICEXCITONIC_RESONANCE_ENERGY/216
0.0092, 316
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 216 in model.eng_flexomagnetophononicplasmonicexcitonic_resonance_energies
    eng = model.eng_flexomagnetophononicplasmonicexcitonic_resonance_energies[216]
    assert pytest.approx(eng.dt_fmppre) == 0.0092
    assert eng.sens_id == 316


def test_m387_eng_flexomagnetophononicplasmonicexcitonic_resonance_energy_aliases(tmp_path: Path):
    deck = """# RADIOSS ALIAS DECK
/BEGIN
Engine Flexomagnetophononicplasmonicexcitonic Resonance Energy Aliases Test
/ENG/FLEXOMAGNETO_PHONON_PLASMON_EXCITON_RES_WORK/217
0.0102, 317
/ENG/EFLEXOMAGNETOPHONONICPLASMONICEXCITONICRESONANCE/218
0.0112, 318
/ENG/FLEXOMAGNETOPHONONICPLASMONICEXCITONIC_RESONANCE_DISSIPATION/219
0.0122, 319
/ENG/EM_FLEXOMAGNETOPHONONICPLASMONICEXCITONIC_RESONANCE/220
0.0132, 320
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 217 in model.eng_flexomagnetophononicplasmonicexcitonic_resonance_energies
    assert 218 in model.eng_flexomagnetophononicplasmonicexcitonic_resonance_energies
    assert 219 in model.eng_flexomagnetophononicplasmonicexcitonic_resonance_energies
    assert 220 in model.eng_flexomagnetophononicexcitonicmagnonicpolaritonic_resonance_energies or 220 in model.eng_flexomagnetophononicplasmonicexcitonic_resonance_energies


def test_m387_eng_flexomagnetophononicplasmonicexcitonic_resonance_energy_missing_card(tmp_path: Path):
    deck = """# RADIOSS MISSING CARD DECK
/BEGIN
/ENG/FLEXOMAGNETOPHONONICPLASMONICEXCITONIC_RESONANCE_ENERGY/221
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) > 0


def test_m387_lagmul_phillips_spatial_linkage_joint_fixed(tmp_path: Path):
    c1 = f"{301:>10d}{302:>10d}{303:>10d}{1.56e7:>20.4f}{35:>10d}{7.0e-5:>20.4e}"
    c2 = f"{102.0:>20.4f}{86.0:>20.4f}{110.0:>20.4f}{44.0:>20.4f}"
    deck = f"""# RADIOSS STARTER DECK
/BEGIN
Phillips Spatial Linkage Joint Fixed Format Test
2022 0
/LAGMUL/PHILLIPS_SPATIAL_LINKAGE_JOINT/265
Phillips Spatial 6R Joint
{c1}
{c2}
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 265 in model.lagmul_phillips_spatial_linkage_joints
    joint = model.lagmul_phillips_spatial_linkage_joints[265]
    assert joint.node1 == 301
    assert joint.node2 == 302
    assert joint.node3 == 303
    assert pytest.approx(joint.stiff) == 1.56e7
    assert joint.skew_id == 35
    assert pytest.approx(joint.tol) == 7.0e-5
    assert pytest.approx(joint.link_len_a) == 102.0
    assert pytest.approx(joint.link_len_b) == 86.0
    assert pytest.approx(joint.twist_angle_alpha) == 110.0
    assert pytest.approx(joint.offset_distance_s) == 44.0
    assert pytest.approx(joint.offset_distance_r) == 44.0
    assert pytest.approx(joint.offset_distance_v) == 44.0


def test_m387_lagmul_phillips_spatial_linkage_joint_free(tmp_path: Path):
    deck = """# RADIOSS FREE DECK
/BEGIN
Phillips Spatial Linkage Joint Free Format Test
/LAGMUL/PHILLIPS_SPATIAL_LINKAGE_JOINT/266
401, 402, 403, 1.80e7, 36, 8.0e-5
120.0, 92.0, 130.0, 56.0
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 266 in model.lagmul_phillips_spatial_linkage_joints
    joint = model.lagmul_phillips_spatial_linkage_joints[266]
    assert joint.node1 == 401
    assert joint.node2 == 402
    assert joint.node3 == 403
    assert pytest.approx(joint.stiff) == 1.80e7
    assert joint.skew_id == 36
    assert pytest.approx(joint.tol) == 8.0e-5
    assert pytest.approx(joint.link_len_a) == 120.0
    assert pytest.approx(joint.link_len_b) == 92.0
    assert pytest.approx(joint.twist_angle_alpha) == 130.0
    assert pytest.approx(joint.offset_distance_s) == 56.0


def test_m387_lagmul_phillips_spatial_linkage_joint_aliases(tmp_path: Path):
    deck = """# RADIOSS ALIAS DECK
/BEGIN
Phillips Spatial Linkage Joint Aliases Test
/PHILLIPS_SPATIAL_LINKAGE_JOINT/267
501, 502, 503, 1.0e6, 0, 1.0e-6
72.0, 62.0, 72.0, 27.0
/LAGMUL/PHILLIPS_SPATIAL_LINKAGE/268
501, 502, 503, 1.0e6, 0, 1.0e-6
72.0, 62.0, 72.0, 27.0
/PHILLIPS_SPATIAL_LINKAGE/269
501, 502, 503, 1.0e6, 0, 1.0e-6
72.0, 62.0, 72.0, 27.0
/PHILLIPS_SPATIAL_MULTI_LOOP_MECHANISM/270
501, 502, 503, 1.0e6, 0, 1.0e-6
72.0, 62.0, 72.0, 27.0
/PHILLIPS_SPATIAL_SYMMETRIC_MECHANISM/271
501, 502, 503, 1.0e6, 0, 1.0e-6
72.0, 62.0, 72.0, 27.0
/PHILLIPS_SPATIAL_6R_MECHANISM/272
501, 502, 503, 1.0e6, 0, 1.0e-6
72.0, 62.0, 72.0, 27.0
/PHILLIPS_SPATIAL_OVERCONSTRAINED_MECHANISM/273
501, 502, 503, 1.0e6, 0, 1.0e-6
72.0, 62.0, 72.0, 27.0
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    for jid in range(267, 274):
        assert jid in model.lagmul_phillips_spatial_linkage_joints


def test_m387_lagmul_phillips_spatial_linkage_joint_missing_card(tmp_path: Path):
    deck = """# RADIOSS MISSING CARD DECK
/BEGIN
/LAGMUL/PHILLIPS_SPATIAL_LINKAGE_JOINT/274
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) > 0


def test_m387_sensor_spring_total_drop_rate_fixed(tmp_path: Path):
    c1 = f"{1348:>10d}{2.68e8:>20.4f}{0.0815:>20.4f}"
    deck = f"""# RADIOSS STARTER DECK
/BEGIN
Sensor Spring Total Drop Rate Fixed Format Test
2022 0
/SENSOR/SPRING_TOTAL_DROP_RATE/1
Fixed Spring Total Drop Rate Sensor
{c1}
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 1 in model.sensor_spring_total_drop_rates
    s1 = model.sensor_spring_total_drop_rates[1]
    assert s1.spring_id == 1348
    assert pytest.approx(s1.jtot_drop_max) == 2.68e8
    assert pytest.approx(s1.jtot_lock_max) == 2.68e8
    assert pytest.approx(s1.jtot_pop_max) == 2.68e8
    assert pytest.approx(s1.jtot_snp_max) == 2.68e8
    assert pytest.approx(s1.jtot_crackle_max) == 2.68e8
    assert pytest.approx(s1.t_delay) == 0.0815
    assert len(model.sensors) == 1
    assert model.sensors[0].kind == "SPRING_TOTAL_DROP_RATE"


def test_m387_sensor_spring_total_drop_rate_free(tmp_path: Path):
    deck = """# RADIOSS FREE DECK
/BEGIN
Sensor Spring Total Drop Rate Free Format Test
/SENSOR/SPRING_TOTAL_DROP_RATE/2
Free Spring Total Drop Rate Sensor
1349, 3.28e8, 0.0965
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 2 in model.sensor_spring_total_drop_rates
    s2 = model.sensor_spring_total_drop_rates[2]
    assert s2.spring_id == 1349
    assert pytest.approx(s2.jtot_drop_max) == 3.28e8
    assert pytest.approx(s2.t_delay) == 0.0965


def test_m387_sensor_spring_total_drop_rate_aliases(tmp_path: Path):
    deck = """# RADIOSS ALIAS DECK
/BEGIN
Sensor Spring Total Drop Rate Aliases Test
/SENSOR/SPRING_TOT_DROP_RATE/370
1369, 1.175e8, 0.0175
/SENSOR/SPRING_RATE_DROP_TOT/371
1370, 1.195e8, 0.0185
/SENSOR/TOTAL_DROP_RATE_SPRING/372
1371, 1.215e8, 0.0195
/SENSOR/SPRING_DROP_TOT/373
1372, 1.235e8, 0.0205
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    for sid in range(370, 374):
        assert sid in model.sensor_spring_total_drop_rates


def test_m387_sensor_spring_total_drop_rate_missing_card(tmp_path: Path):
    deck = """# RADIOSS MISSING CARD DECK
/BEGIN
/SENSOR/SPRING_TOTAL_DROP_RATE/375
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) > 0
