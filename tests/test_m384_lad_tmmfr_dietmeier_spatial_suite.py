"""Tests for Milestone M384: LadTransverseMatrixMicroFissuringRate Failure Model, EngFlexomagnetoplasmonicexcitonicpolaritonicResonanceEnergy, DietmeierSpatialLinkageJoint, and SensorSpringTotalAngularLockRate."""

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


def test_m384_fail_lad_transverse_matrix_micro_fissuring_rate_fixed(tmp_path: Path):
    c1 = f"{275.0:>20.4f}{825.0:>20.4f}{125.0:>20.4f}{4.15:>20.4f}{0.925:>20.4f}"
    c2 = f"{1:>10d}{2:>10d}"
    c3 = f"{1980:>10d}"
    deck = f"""# RADIOSS STARTER DECK
/BEGIN
Ladeveze Transverse Matrix Micro-Fissuring Rate Fixed Format Test
2022 0
/FAIL/LAD_TRANSVERSE_MATRIX_MICRO_FISSURING_RATE/1980
Ladeveze Transverse Matrix Micro-Fissuring Failure Model
{c1}
{c2}
{c3}
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 1980 in model.fail_ladtransversematrixmicrofissuringrates
    ftmmfr = model.fail_ladtransversematrixmicrofissuringrates[1980]
    assert pytest.approx(ftmmfr.sigma_tmmfr0) == 275.0
    assert pytest.approx(ftmmfr.sigma_tmmfrc) == 825.0
    assert pytest.approx(ftmmfr.gamma_tmmfr) == 125.0
    assert pytest.approx(ftmmfr.p_tmmfr) == 4.15
    assert pytest.approx(ftmmfr.d_tmmfr_max) == 0.925
    assert ftmmfr.ifail_sh == 1
    assert ftmmfr.ifail_so == 2
    assert ftmmfr.fail_id == 1980
    assert len(model.raw_fails) == 1
    assert model.raw_fails[0][1].type == "LAD_TRANSVERSE_MATRIX_MICRO_FISSURING_RATE"


def test_m384_fail_lad_transverse_matrix_micro_fissuring_rate_free(tmp_path: Path):
    deck = """# RADIOSS FREE DECK
/BEGIN
Ladeveze Transverse Matrix Micro-Fissuring Rate Free Format Test
/FAIL/LAD_TRANSVERSE_MATRIX_MICRO_FISSURING_RATE/1981
285.0, 855.0, 130.0, 4.35, 0.900
1, 1
1981
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 1981 in model.fail_ladtransversematrixmicrofissuringrates
    ftmmfr = model.fail_ladtransversematrixmicrofissuringrates[1981]
    assert pytest.approx(ftmmfr.sigma_tmmfr0) == 285.0
    assert pytest.approx(ftmmfr.sigma_tmmfrc) == 855.0
    assert pytest.approx(ftmmfr.gamma_tmmfr) == 130.0
    assert pytest.approx(ftmmfr.p_tmmfr) == 4.35
    assert pytest.approx(ftmmfr.d_tmmfr_max) == 0.900
    assert ftmmfr.fail_id == 1981


def test_m384_fail_lad_transverse_matrix_micro_fissuring_rate_aliases(tmp_path: Path):
    deck = """# RADIOSS ALIAS DECK
/BEGIN
Ladeveze Transverse Matrix Micro-Fissuring Rate Aliases Test
/FAIL/LADEVEZE_TRANSVERSE_MATRIX_MICRO_FISSURING_RATE/1982
265.0, 795.0, 116.0, 3.85, 0.945
1, 1
/FAIL/LAD_TMMFR/1983
265.0, 795.0, 116.0, 3.85, 0.945
1, 1
/FAIL/LAD_TMMFR_MODEL/1984
265.0, 795.0, 116.0, 3.85, 0.945
1, 1
/FAIL/LAD_TMMFR_LAW/1985
265.0, 795.0, 116.0, 3.85, 0.945
1, 1
/FAIL/LADEVEZE_RATE_DEPENDENT_TRANSVERSE_MATRIX_MICRO_FISSURING/1986
265.0, 795.0, 116.0, 3.85, 0.945
1, 1
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 1982 in model.fail_ladtransversematrixmicrofissuringrates
    assert 1983 in model.fail_ladtransversematrixmicrofissuringrates
    assert 1984 in model.fail_ladtransversematrixmicrofissuringrates
    assert 1985 in model.fail_ladtransversematrixmicrofissuringrates
    assert 1986 in model.fail_ladtransversematrixmicrofissuringrates
    assert len(model.raw_fails) == 5


def test_m384_fail_lad_transverse_matrix_micro_fissuring_rate_missing_card(tmp_path: Path):
    deck = """# RADIOSS MISSING CARD DECK
/BEGIN
/FAIL/LAD_TRANSVERSE_MATRIX_MICRO_FISSURING_RATE/1987
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) > 0


def test_m384_eng_flexomagnetoplasmonicexcitonicpolaritonic_resonance_energy_fixed(tmp_path: Path):
    c1 = f"{0.0072:>20.4f}{285:>10d}"
    deck = f"""# RADIOSS STARTER DECK
/BEGIN
Engine Flexomagnetoplasmonicexcitonicpolaritonic Resonance Energy Fixed Format Test
2022 0
/ENG/FLEXOMAGNETOPLASMONICEXCITONICPOLARITONIC_RESONANCE_ENERGY/185
Resonance Energy Output Directive
{c1}
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 185 in model.eng_flexomagnetoplasmonicexcitonicpolaritonic_resonance_energies
    eng = model.eng_flexomagnetoplasmonicexcitonicpolaritonic_resonance_energies[185]
    assert pytest.approx(eng.dt_fmpepr) == 0.0072
    assert eng.sens_id == 285


def test_m384_eng_flexomagnetoplasmonicexcitonicpolaritonic_resonance_energy_free(tmp_path: Path):
    deck = """# RADIOSS FREE DECK
/BEGIN
Engine Flexomagnetoplasmonicexcitonicpolaritonic Resonance Energy Free Format Test
/ENG/FLEXOMAGNETOPLASMONICEXCITONICPOLARITONIC_RESONANCE_ENERGY/186
0.0082, 286
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 186 in model.eng_flexomagnetoplasmonicexcitonicpolaritonic_resonance_energies
    eng = model.eng_flexomagnetoplasmonicexcitonicpolaritonic_resonance_energies[186]
    assert pytest.approx(eng.dt_fmpepr) == 0.0082
    assert eng.sens_id == 286


def test_m384_eng_flexomagnetoplasmonicexcitonicpolaritonic_resonance_energy_aliases(tmp_path: Path):
    deck = """# RADIOSS ALIAS DECK
/BEGIN
Engine Flexomagnetoplasmonicexcitonicpolaritonic Resonance Energy Aliases Test
/ENG/FLEXOMAGNETO_PLASMON_EXCITON_POLARITON_RES_WORK/187
0.0092, 287
/ENG/EFLEXOMAGNETOPLASMONICEXCITONICPOLARITONICRESONANCE/188
0.0102, 288
/ENG/FLEXOMAGNETOPLASMONICEXCITONICPOLARITONIC_RESONANCE_DISSIPATION/189
0.0112, 289
/ENG/EM_FLEXOMAGNETOPLASMONICEXCITONICPOLARITONIC_RESONANCE/190
0.0122, 290
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 187 in model.eng_flexomagnetoplasmonicexcitonicpolaritonic_resonance_energies
    assert 188 in model.eng_flexomagnetoplasmonicexcitonicpolaritonic_resonance_energies
    assert 189 in model.eng_flexomagnetoplasmonicexcitonicpolaritonic_resonance_energies
    assert 190 in model.eng_flexomagnetoplasmonicexcitonicpolaritonic_resonance_energies


def test_m384_eng_flexomagnetoplasmonicexcitonicpolaritonic_resonance_energy_missing_card(tmp_path: Path):
    deck = """# RADIOSS MISSING CARD DECK
/BEGIN
/ENG/FLEXOMAGNETOPLASMONICEXCITONICPOLARITONIC_RESONANCE_ENERGY/191
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) > 0


def test_m384_lagmul_dietmeier_spatial_linkage_joint_fixed(tmp_path: Path):
    c1 = f"{271:>10d}{272:>10d}{273:>10d}{1.26e7:>20.4f}{29:>10d}{5.5e-5:>20.4e}"
    c2 = f"{92.0:>20.4f}{74.0:>20.4f}{95.0:>20.4f}{38.0:>20.4f}"
    deck = f"""# RADIOSS STARTER DECK
/BEGIN
Dietmeier Spatial Linkage Joint Fixed Format Test
2022 0
/LAGMUL/DIETMEIER_SPATIAL_LINKAGE_JOINT/235
Dietmeier Spatial 6R Joint
{c1}
{c2}
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 235 in model.lagmul_dietmeier_spatial_linkage_joints
    joint = model.lagmul_dietmeier_spatial_linkage_joints[235]
    assert joint.node1 == 271
    assert joint.node2 == 272
    assert joint.node3 == 273
    assert pytest.approx(joint.stiff) == 1.26e7
    assert joint.skew_id == 29
    assert pytest.approx(joint.tol) == 5.5e-5
    assert pytest.approx(joint.link_len_a) == 92.0
    assert pytest.approx(joint.link_len_b) == 74.0
    assert pytest.approx(joint.twist_angle_alpha) == 95.0
    assert pytest.approx(joint.offset_distance_s) == 38.0
    assert pytest.approx(joint.offset_distance_r) == 38.0


def test_m384_lagmul_dietmeier_spatial_linkage_joint_free(tmp_path: Path):
    deck = """# RADIOSS FREE DECK
/BEGIN
Dietmeier Spatial Linkage Joint Free Format Test
/LAGMUL/DIETMEIER_SPATIAL_LINKAGE_JOINT/236
371, 372, 373, 1.50e7, 30, 6.5e-5
105.0, 80.0, 115.0, 50.0
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 236 in model.lagmul_dietmeier_spatial_linkage_joints
    joint = model.lagmul_dietmeier_spatial_linkage_joints[236]
    assert joint.node1 == 371
    assert joint.node2 == 372
    assert joint.node3 == 373
    assert pytest.approx(joint.stiff) == 1.50e7
    assert joint.skew_id == 30
    assert pytest.approx(joint.tol) == 6.5e-5
    assert pytest.approx(joint.link_len_a) == 105.0
    assert pytest.approx(joint.link_len_b) == 80.0
    assert pytest.approx(joint.twist_angle_alpha) == 115.0
    assert pytest.approx(joint.offset_distance_s) == 50.0


def test_m384_lagmul_dietmeier_spatial_linkage_joint_aliases(tmp_path: Path):
    deck = """# RADIOSS ALIAS DECK
/BEGIN
Dietmeier Spatial Linkage Joint Aliases Test
/DIETMEIER_SPATIAL_LINKAGE_JOINT/237
471, 472, 473, 1.0e6, 0, 1.0e-6
65.0, 55.0, 65.0, 23.0
/LAGMUL/DIETMEIER_SPATIAL_LINKAGE/238
471, 472, 473, 1.0e6, 0, 1.0e-6
65.0, 55.0, 65.0, 23.0
/DIETMEIER_SPATIAL_LINKAGE/239
471, 472, 473, 1.0e6, 0, 1.0e-6
65.0, 55.0, 65.0, 23.0
/DIETMEIER_SPATIAL_MULTI_LOOP_MECHANISM/240
471, 472, 473, 1.0e6, 0, 1.0e-6
65.0, 55.0, 65.0, 23.0
/DIETMEIER_SPATIAL_SYMMETRIC_MECHANISM/241
471, 472, 473, 1.0e6, 0, 1.0e-6
65.0, 55.0, 65.0, 23.0
/DIETMEIER_SPATIAL_6R_MECHANISM/242
471, 472, 473, 1.0e6, 0, 1.0e-6
65.0, 55.0, 65.0, 23.0
/DIETMEIER_SPATIAL_OVERCONSTRAINED_MECHANISM/243
471, 472, 473, 1.0e6, 0, 1.0e-6
65.0, 55.0, 65.0, 23.0
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    for jid in range(237, 244):
        assert jid in model.lagmul_dietmeier_spatial_linkage_joints


def test_m384_lagmul_dietmeier_spatial_linkage_joint_missing_card(tmp_path: Path):
    deck = """# RADIOSS MISSING CARD DECK
/BEGIN
/LAGMUL/DIETMEIER_SPATIAL_LINKAGE_JOINT/244
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) > 0


def test_m384_sensor_spring_total_angular_lock_rate_fixed(tmp_path: Path):
    c1 = f"{1318:>10d}{2.38e8:>20.4f}{0.0725:>20.4f}"
    deck = f"""# RADIOSS STARTER DECK
/BEGIN
Sensor Spring Total Angular Lock Rate Fixed Format Test
2022 0
/SENSOR/SPRING_TOTAL_ANGULAR_LOCK_RATE/1
Fixed Spring Total Angular Lock Rate Sensor
{c1}
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 1 in model.sensor_spring_total_angular_lock_rates
    s1 = model.sensor_spring_total_angular_lock_rates[1]
    assert s1.spring_id == 1318
    assert pytest.approx(s1.jtot_ang_lock_max) == 2.38e8
    assert pytest.approx(s1.jtot_ang_pop_max) == 2.38e8
    assert pytest.approx(s1.jtot_ang_snp_max) == 2.38e8
    assert pytest.approx(s1.jtot_ang_crackle_max) == 2.38e8
    assert pytest.approx(s1.t_delay) == 0.0725
    assert len(model.sensors) == 1
    assert model.sensors[0].kind == "SPRING_TOTAL_ANGULAR_LOCK_RATE"


def test_m384_sensor_spring_total_angular_lock_rate_free(tmp_path: Path):
    deck = """# RADIOSS FREE DECK
/BEGIN
Sensor Spring Total Angular Lock Rate Free Format Test
/SENSOR/SPRING_TOTAL_ANGULAR_LOCK_RATE/2
Free Spring Total Angular Lock Rate Sensor
1319, 2.98e8, 0.0875
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 2 in model.sensor_spring_total_angular_lock_rates
    s2 = model.sensor_spring_total_angular_lock_rates[2]
    assert s2.spring_id == 1319
    assert pytest.approx(s2.jtot_ang_lock_max) == 2.98e8
    assert pytest.approx(s2.t_delay) == 0.0875


def test_m384_sensor_spring_total_angular_lock_rate_aliases(tmp_path: Path):
    deck = """# RADIOSS ALIAS DECK
/BEGIN
Sensor Spring Total Angular Lock Rate Aliases Test
/SENSOR/SPRING_TOT_ANG_LOCK_RATE/340
1339, 1.145e8, 0.0155
/SENSOR/SPRING_RATE_LOCK_ANG_TOT/341
1340, 1.165e8, 0.0165
/SENSOR/TOTAL_ANGULAR_LOCK_RATE_SPRING/342
1341, 1.185e8, 0.0175
/SENSOR/SPRING_LOCK_ANG_TOT/343
1342, 1.205e8, 0.0185
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    for sid in range(340, 344):
        assert sid in model.sensor_spring_total_angular_lock_rates


def test_m384_sensor_spring_total_angular_lock_rate_missing_card(tmp_path: Path):
    deck = """# RADIOSS MISSING CARD DECK
/BEGIN
/SENSOR/SPRING_TOTAL_ANGULAR_LOCK_RATE/345
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) > 0
