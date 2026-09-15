"""Tests for Milestone M385: LadCoupleMatrixMicroFissuringRate Failure Model, EngFlexomagnetoplasmonicmagnonicpolaritonicResonanceEnergy, HuntSpatialLinkageJoint, and SensorSpringNormalDropRate."""

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


def test_m385_fail_lad_couple_matrix_micro_fissuring_rate_fixed(tmp_path: Path):
    c1 = f"{295.0:>20.4f}{885.0:>20.4f}{135.0:>20.4f}{4.45:>20.4f}{0.935:>20.4f}"
    c2 = f"{1:>10d}{2:>10d}"
    c3 = f"{1990:>10d}"
    deck = f"""# RADIOSS STARTER DECK
/BEGIN
Ladeveze Coupled Matrix Micro-Fissuring Rate Fixed Format Test
2022 0
/FAIL/LAD_COUPLE_MATRIX_MICRO_FISSURING_RATE/1990
Ladeveze Coupled Matrix Micro-Fissuring Failure Model
{c1}
{c2}
{c3}
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 1990 in model.fail_ladcouplematrixmicrofissuringrates
    fcmmfr = model.fail_ladcouplematrixmicrofissuringrates[1990]
    assert pytest.approx(fcmmfr.sigma_cmmfr0) == 295.0
    assert pytest.approx(fcmmfr.sigma_cmmfrc) == 885.0
    assert pytest.approx(fcmmfr.gamma_cmmfr) == 135.0
    assert pytest.approx(fcmmfr.p_cmmfr) == 4.45
    assert pytest.approx(fcmmfr.d_cmmfr_max) == 0.935
    assert fcmmfr.ifail_sh == 1
    assert fcmmfr.ifail_so == 2
    assert fcmmfr.fail_id == 1990
    assert len(model.raw_fails) == 1
    assert model.raw_fails[0][1].type == "LAD_COUPLE_MATRIX_MICRO_FISSURING_RATE"


def test_m385_fail_lad_couple_matrix_micro_fissuring_rate_free(tmp_path: Path):
    deck = """# RADIOSS FREE DECK
/BEGIN
Ladeveze Coupled Matrix Micro-Fissuring Rate Free Format Test
/FAIL/LAD_COUPLE_MATRIX_MICRO_FISSURING_RATE/1991
305.0, 915.0, 140.0, 4.65, 0.910
1, 1
1991
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 1991 in model.fail_ladcouplematrixmicrofissuringrates
    fcmmfr = model.fail_ladcouplematrixmicrofissuringrates[1991]
    assert pytest.approx(fcmmfr.sigma_cmmfr0) == 305.0
    assert pytest.approx(fcmmfr.sigma_cmmfrc) == 915.0
    assert pytest.approx(fcmmfr.gamma_cmmfr) == 140.0
    assert pytest.approx(fcmmfr.p_cmmfr) == 4.65
    assert pytest.approx(fcmmfr.d_cmmfr_max) == 0.910
    assert fcmmfr.fail_id == 1991


def test_m385_fail_lad_couple_matrix_micro_fissuring_rate_aliases(tmp_path: Path):
    deck = """# RADIOSS ALIAS DECK
/BEGIN
Ladeveze Coupled Matrix Micro-Fissuring Rate Aliases Test
/FAIL/LADEVEZE_COUPLED_MATRIX_MICRO_FISSURING_RATE/1992
285.0, 855.0, 126.0, 4.15, 0.955
1, 1
/FAIL/LAD_CMMFR/1993
285.0, 855.0, 126.0, 4.15, 0.955
1, 1
/FAIL/LAD_CMMFR_MODEL/1994
285.0, 855.0, 126.0, 4.15, 0.955
1, 1
/FAIL/LAD_CMMFR_LAW/1995
285.0, 855.0, 126.0, 4.15, 0.955
1, 1
/FAIL/LADEVEZE_RATE_DEPENDENT_COUPLED_MATRIX_MICRO_FISSURING/1996
285.0, 855.0, 126.0, 4.15, 0.955
1, 1
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 1992 in model.fail_ladcouplematrixmicrofissuringrates
    assert 1993 in model.fail_ladcouplematrixmicrofissuringrates
    assert 1994 in model.fail_ladcouplematrixmicrofissuringrates
    assert 1995 in model.fail_ladcouplematrixmicrofissuringrates
    assert 1996 in model.fail_ladcouplematrixmicrofissuringrates
    assert len(model.raw_fails) == 5


def test_m385_fail_lad_couple_matrix_micro_fissuring_rate_missing_card(tmp_path: Path):
    deck = """# RADIOSS MISSING CARD DECK
/BEGIN
/FAIL/LAD_COUPLE_MATRIX_MICRO_FISSURING_RATE/1997
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) > 0


def test_m385_eng_flexomagnetoplasmonicmagnonicpolaritonic_resonance_energy_fixed(tmp_path: Path):
    c1 = f"{0.0076:>20.4f}{295:>10d}"
    deck = f"""# RADIOSS STARTER DECK
/BEGIN
Engine Flexomagnetoplasmonicmagnonicpolaritonic Resonance Energy Fixed Format Test
2022 0
/ENG/FLEXOMAGNETOPLASMONICMAGNONICPOLARITONIC_RESONANCE_ENERGY/195
Resonance Energy Output Directive
{c1}
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 195 in model.eng_flexomagnetoplasmonicmagnonicpolaritonic_resonance_energies
    eng = model.eng_flexomagnetoplasmonicmagnonicpolaritonic_resonance_energies[195]
    assert pytest.approx(eng.dt_fmpmpr) == 0.0076
    assert eng.sens_id == 295


def test_m385_eng_flexomagnetoplasmonicmagnonicpolaritonic_resonance_energy_free(tmp_path: Path):
    deck = """# RADIOSS FREE DECK
/BEGIN
Engine Flexomagnetoplasmonicmagnonicpolaritonic Resonance Energy Free Format Test
/ENG/FLEXOMAGNETOPLASMONICMAGNONICPOLARITONIC_RESONANCE_ENERGY/196
0.0086, 296
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 196 in model.eng_flexomagnetoplasmonicmagnonicpolaritonic_resonance_energies
    eng = model.eng_flexomagnetoplasmonicmagnonicpolaritonic_resonance_energies[196]
    assert pytest.approx(eng.dt_fmpmpr) == 0.0086
    assert eng.sens_id == 296


def test_m385_eng_flexomagnetoplasmonicmagnonicpolaritonic_resonance_energy_aliases(tmp_path: Path):
    deck = """# RADIOSS ALIAS DECK
/BEGIN
Engine Flexomagnetoplasmonicmagnonicpolaritonic Resonance Energy Aliases Test
/ENG/FLEXOMAGNETO_PLASMON_MAGNON_POLARITON_RES_WORK/197
0.0096, 297
/ENG/EFLEXOMAGNETOPLASMONICMAGNONICPOLARITONICRESONANCE/198
0.0106, 298
/ENG/FLEXOMAGNETOPLASMONICMAGNONICPOLARITONIC_RESONANCE_DISSIPATION/199
0.0116, 299
/ENG/EM_FLEXOMAGNETOPLASMONICMAGNONICPOLARITONIC_RESONANCE/200
0.0126, 300
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 197 in model.eng_flexomagnetoplasmonicmagnonicpolaritonic_resonance_energies
    assert 198 in model.eng_flexomagnetoplasmonicmagnonicpolaritonic_resonance_energies
    assert 199 in model.eng_flexomagnetoplasmonicmagnonicpolaritonic_resonance_energies
    assert 200 in model.eng_flexomagnetoplasmonicmagnonicpolaritonic_resonance_energies


def test_m385_eng_flexomagnetoplasmonicmagnonicpolaritonic_resonance_energy_missing_card(tmp_path: Path):
    deck = """# RADIOSS MISSING CARD DECK
/BEGIN
/ENG/FLEXOMAGNETOPLASMONICMAGNONICPOLARITONIC_RESONANCE_ENERGY/201
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) > 0


def test_m385_lagmul_hunt_spatial_linkage_joint_fixed(tmp_path: Path):
    c1 = f"{281:>10d}{282:>10d}{283:>10d}{1.36e7:>20.4f}{31:>10d}{6.0e-5:>20.4e}"
    c2 = f"{96.0:>20.4f}{78.0:>20.4f}{100.0:>20.4f}{40.0:>20.4f}"
    deck = f"""# RADIOSS STARTER DECK
/BEGIN
Hunt Spatial Linkage Joint Fixed Format Test
2022 0
/LAGMUL/HUNT_SPATIAL_LINKAGE_JOINT/245
Hunt Spatial 6R Joint
{c1}
{c2}
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 245 in model.lagmul_hunt_spatial_linkage_joints
    joint = model.lagmul_hunt_spatial_linkage_joints[245]
    assert joint.node1 == 281
    assert joint.node2 == 282
    assert joint.node3 == 283
    assert pytest.approx(joint.stiff) == 1.36e7
    assert joint.skew_id == 31
    assert pytest.approx(joint.tol) == 6.0e-5
    assert pytest.approx(joint.link_len_a) == 96.0
    assert pytest.approx(joint.link_len_b) == 78.0
    assert pytest.approx(joint.twist_angle_alpha) == 100.0
    assert pytest.approx(joint.offset_distance_s) == 40.0
    assert pytest.approx(joint.offset_distance_r) == 40.0


def test_m385_lagmul_hunt_spatial_linkage_joint_free(tmp_path: Path):
    deck = """# RADIOSS FREE DECK
/BEGIN
Hunt Spatial Linkage Joint Free Format Test
/LAGMUL/HUNT_SPATIAL_LINKAGE_JOINT/246
381, 382, 383, 1.60e7, 32, 7.0e-5
110.0, 84.0, 120.0, 52.0
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 246 in model.lagmul_hunt_spatial_linkage_joints
    joint = model.lagmul_hunt_spatial_linkage_joints[246]
    assert joint.node1 == 381
    assert joint.node2 == 382
    assert joint.node3 == 383
    assert pytest.approx(joint.stiff) == 1.60e7
    assert joint.skew_id == 32
    assert pytest.approx(joint.tol) == 7.0e-5
    assert pytest.approx(joint.link_len_a) == 110.0
    assert pytest.approx(joint.link_len_b) == 84.0
    assert pytest.approx(joint.twist_angle_alpha) == 120.0
    assert pytest.approx(joint.offset_distance_s) == 52.0


def test_m385_lagmul_hunt_spatial_linkage_joint_aliases(tmp_path: Path):
    deck = """# RADIOSS ALIAS DECK
/BEGIN
Hunt Spatial Linkage Joint Aliases Test
/HUNT_SPATIAL_LINKAGE_JOINT/247
481, 482, 483, 1.0e6, 0, 1.0e-6
68.0, 58.0, 68.0, 25.0
/LAGMUL/HUNT_SPATIAL_LINKAGE/248
481, 482, 483, 1.0e6, 0, 1.0e-6
68.0, 58.0, 68.0, 25.0
/HUNT_SPATIAL_LINKAGE/249
481, 482, 483, 1.0e6, 0, 1.0e-6
68.0, 58.0, 68.0, 25.0
/HUNT_SPATIAL_MULTI_LOOP_MECHANISM/250
481, 482, 483, 1.0e6, 0, 1.0e-6
68.0, 58.0, 68.0, 25.0
/HUNT_SPATIAL_SYMMETRIC_MECHANISM/251
481, 482, 483, 1.0e6, 0, 1.0e-6
68.0, 58.0, 68.0, 25.0
/HUNT_SPATIAL_6R_MECHANISM/252
481, 482, 483, 1.0e6, 0, 1.0e-6
68.0, 58.0, 68.0, 25.0
/HUNT_SPATIAL_OVERCONSTRAINED_MECHANISM/253
481, 482, 483, 1.0e6, 0, 1.0e-6
68.0, 58.0, 68.0, 25.0
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    for jid in range(247, 254):
        assert jid in model.lagmul_hunt_spatial_linkage_joints


def test_m385_lagmul_hunt_spatial_linkage_joint_missing_card(tmp_path: Path):
    deck = """# RADIOSS MISSING CARD DECK
/BEGIN
/LAGMUL/HUNT_SPATIAL_LINKAGE_JOINT/254
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) > 0


def test_m385_sensor_spring_normal_drop_rate_fixed(tmp_path: Path):
    c1 = f"{1328:>10d}{2.48e8:>20.4f}{0.0755:>20.4f}"
    deck = f"""# RADIOSS STARTER DECK
/BEGIN
Sensor Spring Normal Drop Rate Fixed Format Test
2022 0
/SENSOR/SPRING_NORMAL_DROP_RATE/1
Fixed Spring Normal Drop Rate Sensor
{c1}
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 1 in model.sensor_spring_normal_drop_rates
    s1 = model.sensor_spring_normal_drop_rates[1]
    assert s1.spring_id == 1328
    assert pytest.approx(s1.jnorm_drop_max) == 2.48e8
    assert pytest.approx(s1.jnorm_lock_max) == 2.48e8
    assert pytest.approx(s1.jnorm_pop_max) == 2.48e8
    assert pytest.approx(s1.jnorm_snp_max) == 2.48e8
    assert pytest.approx(s1.jnorm_crackle_max) == 2.48e8
    assert pytest.approx(s1.t_delay) == 0.0755
    assert len(model.sensors) == 1
    assert model.sensors[0].kind == "SPRING_NORMAL_DROP_RATE"


def test_m385_sensor_spring_normal_drop_rate_free(tmp_path: Path):
    deck = """# RADIOSS FREE DECK
/BEGIN
Sensor Spring Normal Drop Rate Free Format Test
/SENSOR/SPRING_NORMAL_DROP_RATE/2
Free Spring Normal Drop Rate Sensor
1329, 3.08e8, 0.0905
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 2 in model.sensor_spring_normal_drop_rates
    s2 = model.sensor_spring_normal_drop_rates[2]
    assert s2.spring_id == 1329
    assert pytest.approx(s2.jnorm_drop_max) == 3.08e8
    assert pytest.approx(s2.t_delay) == 0.0905


def test_m385_sensor_spring_normal_drop_rate_aliases(tmp_path: Path):
    deck = """# RADIOSS ALIAS DECK
/BEGIN
Sensor Spring Normal Drop Rate Aliases Test
/SENSOR/SPRING_NORM_DROP_RATE/350
1349, 1.155e8, 0.0160
/SENSOR/SPRING_RATE_DROP_NORM/351
1350, 1.175e8, 0.0170
/SENSOR/NORMAL_DROP_RATE_SPRING/352
1351, 1.195e8, 0.0180
/SENSOR/SPRING_DROP_NORM/353
1352, 1.215e8, 0.0190
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    for sid in range(350, 354):
        assert sid in model.sensor_spring_normal_drop_rates


def test_m385_sensor_spring_normal_drop_rate_missing_card(tmp_path: Path):
    deck = """# RADIOSS MISSING CARD DECK
/BEGIN
/SENSOR/SPRING_NORMAL_DROP_RATE/355
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) > 0
