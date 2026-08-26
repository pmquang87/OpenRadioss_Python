"""Tests for Milestone M383: LadDynamicMatrixMicroFissuringRate Failure Model, EngFlexomagnetoplasmonicexcitonicmagnonicResonanceEnergy, BakerSpatialLinkageJoint, and SensorSpringBendingLockRate."""

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


def test_m383_fail_lad_dynamic_matrix_micro_fissuring_rate_fixed(tmp_path: Path):
    c1 = f"{255.0:>20.4f}{765.0:>20.4f}{115.0:>20.4f}{3.85:>20.4f}{0.915:>20.4f}"
    c2 = f"{1:>10d}{2:>10d}"
    c3 = f"{1970:>10d}"
    deck = f"""# RADIOSS STARTER DECK
/BEGIN
Ladeveze Dynamic Matrix Micro-Fissuring Rate Fixed Format Test
2022 0
/FAIL/LAD_DYNAMIC_MATRIX_MICRO_FISSURING_RATE/1970
Ladeveze Dynamic Matrix Micro-Fissuring Failure Model
{c1}
{c2}
{c3}
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 1970 in model.fail_laddynamicmatrixmicrofissuringrates
    fdmmfr = model.fail_laddynamicmatrixmicrofissuringrates[1970]
    assert pytest.approx(fdmmfr.sigma_dmmfr0) == 255.0
    assert pytest.approx(fdmmfr.sigma_dmmfrc) == 765.0
    assert pytest.approx(fdmmfr.gamma_dmmfr) == 115.0
    assert pytest.approx(fdmmfr.p_dmmfr) == 3.85
    assert pytest.approx(fdmmfr.d_dmmfr_max) == 0.915
    assert fdmmfr.ifail_sh == 1
    assert fdmmfr.ifail_so == 2
    assert fdmmfr.fail_id == 1970
    assert len(model.raw_fails) == 1
    assert model.raw_fails[0][1].type == "LAD_DYNAMIC_MATRIX_MICRO_FISSURING_RATE"


def test_m383_fail_lad_dynamic_matrix_micro_fissuring_rate_free(tmp_path: Path):
    deck = """# RADIOSS FREE DECK
/BEGIN
Ladeveze Dynamic Matrix Micro-Fissuring Rate Free Format Test
/FAIL/LAD_DYNAMIC_MATRIX_MICRO_FISSURING_RATE/1971
265.0, 795.0, 120.0, 4.05, 0.890
1, 1
1971
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 1971 in model.fail_laddynamicmatrixmicrofissuringrates
    fdmmfr = model.fail_laddynamicmatrixmicrofissuringrates[1971]
    assert pytest.approx(fdmmfr.sigma_dmmfr0) == 265.0
    assert pytest.approx(fdmmfr.sigma_dmmfrc) == 795.0
    assert pytest.approx(fdmmfr.gamma_dmmfr) == 120.0
    assert pytest.approx(fdmmfr.p_dmmfr) == 4.05
    assert pytest.approx(fdmmfr.d_dmmfr_max) == 0.890
    assert fdmmfr.fail_id == 1971


def test_m383_fail_lad_dynamic_matrix_micro_fissuring_rate_aliases(tmp_path: Path):
    deck = """# RADIOSS ALIAS DECK
/BEGIN
Ladeveze Dynamic Matrix Micro-Fissuring Rate Aliases Test
/FAIL/LADEVEZE_DYNAMIC_MATRIX_MICRO_FISSURING_RATE/1972
245.0, 735.0, 106.0, 3.55, 0.935
1, 1
/FAIL/LAD_DMMFR/1973
245.0, 735.0, 106.0, 3.55, 0.935
1, 1
/FAIL/LAD_DMMFR_MODEL/1974
245.0, 735.0, 106.0, 3.55, 0.935
1, 1
/FAIL/LAD_DMMFR_LAW/1975
245.0, 735.0, 106.0, 3.55, 0.935
1, 1
/FAIL/LADEVEZE_RATE_DEPENDENT_DYNAMIC_MATRIX_MICRO_FISSURING/1976
245.0, 735.0, 106.0, 3.55, 0.935
1, 1
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 1972 in model.fail_laddynamicmatrixmicrofissuringrates
    assert 1973 in model.fail_laddynamicmatrixmicrofissuringrates
    assert 1974 in model.fail_laddynamicmatrixmicrofissuringrates
    assert 1975 in model.fail_laddynamicmatrixmicrofissuringrates
    assert 1976 in model.fail_laddynamicmatrixmicrofissuringrates
    assert len(model.raw_fails) == 5


def test_m383_fail_lad_dynamic_matrix_micro_fissuring_rate_missing_card(tmp_path: Path):
    deck = """# RADIOSS MISSING CARD DECK
/BEGIN
/FAIL/LAD_DYNAMIC_MATRIX_MICRO_FISSURING_RATE/1977
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) > 0


def test_m383_eng_flexomagnetoplasmonicexcitonicmagnonic_resonance_energy_fixed(tmp_path: Path):
    c1 = f"{0.0068:>20.4f}{275:>10d}"
    deck = f"""# RADIOSS STARTER DECK
/BEGIN
Engine Flexomagnetoplasmonicexcitonicmagnonic Resonance Energy Fixed Format Test
2022 0
/ENG/FLEXOMAGNETOPLASMONICEXCITONICMAGNONIC_RESONANCE_ENERGY/175
Resonance Energy Output Directive
{c1}
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 175 in model.eng_flexomagnetoplasmonicexcitonicmagnonic_resonance_energies
    eng = model.eng_flexomagnetoplasmonicexcitonicmagnonic_resonance_energies[175]
    assert pytest.approx(eng.dt_fmpemr) == 0.0068
    assert eng.sens_id == 275


def test_m383_eng_flexomagnetoplasmonicexcitonicmagnonic_resonance_energy_free(tmp_path: Path):
    deck = """# RADIOSS FREE DECK
/BEGIN
Engine Flexomagnetoplasmonicexcitonicmagnonic Resonance Energy Free Format Test
/ENG/FLEXOMAGNETOPLASMONICEXCITONICMAGNONIC_RESONANCE_ENERGY/176
0.0078, 276
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 176 in model.eng_flexomagnetoplasmonicexcitonicmagnonic_resonance_energies
    eng = model.eng_flexomagnetoplasmonicexcitonicmagnonic_resonance_energies[176]
    assert pytest.approx(eng.dt_fmpemr) == 0.0078
    assert eng.sens_id == 276


def test_m383_eng_flexomagnetoplasmonicexcitonicmagnonic_resonance_energy_aliases(tmp_path: Path):
    deck = """# RADIOSS ALIAS DECK
/BEGIN
Engine Flexomagnetoplasmonicexcitonicmagnonic Resonance Energy Aliases Test
/ENG/FLEXOMAGNETO_PLASMON_EXCITON_MAGNON_RES_WORK/177
0.0088, 277
/ENG/EFLEXOMAGNETOPLASMONICEXCITONICMAGNONICRESONANCE/178
0.0098, 278
/ENG/FLEXOMAGNETOPLASMONICEXCITONICMAGNONIC_RESONANCE_DISSIPATION/179
0.0108, 279
/ENG/EM_FLEXOMAGNETOPLASMONICEXCITONICMAGNONIC_RESONANCE/180
0.0118, 280
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 177 in model.eng_flexomagnetoplasmonicexcitonicmagnonic_resonance_energies
    assert 178 in model.eng_flexomagnetoplasmonicexcitonicmagnonic_resonance_energies
    assert 179 in model.eng_flexomagnetoplasmonicexcitonicmagnonic_resonance_energies
    assert 180 in model.eng_flexomagnetoplasmonicexcitonicmagnonic_resonance_energies


def test_m383_eng_flexomagnetoplasmonicexcitonicmagnonic_resonance_energy_missing_card(tmp_path: Path):
    deck = """# RADIOSS MISSING CARD DECK
/BEGIN
/ENG/FLEXOMAGNETOPLASMONICEXCITONICMAGNONIC_RESONANCE_ENERGY/181
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) > 0


def test_m383_lagmul_baker_spatial_linkage_joint_fixed(tmp_path: Path):
    c1 = f"{261:>10d}{262:>10d}{263:>10d}{1.16e7:>20.4f}{27:>10d}{5.0e-5:>20.4e}"
    c2 = f"{88.0:>20.4f}{70.0:>20.4f}{90.0:>20.4f}{36.0:>20.4f}"
    deck = f"""# RADIOSS STARTER DECK
/BEGIN
Baker Spatial Linkage Joint Fixed Format Test
2022 0
/LAGMUL/BAKER_SPATIAL_LINKAGE_JOINT/225
Baker Spatial 6R Joint
{c1}
{c2}
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 225 in model.lagmul_baker_spatial_linkage_joints
    joint = model.lagmul_baker_spatial_linkage_joints[225]
    assert joint.node1 == 261
    assert joint.node2 == 262
    assert joint.node3 == 263
    assert pytest.approx(joint.stiff) == 1.16e7
    assert joint.skew_id == 27
    assert pytest.approx(joint.tol) == 5.0e-5
    assert pytest.approx(joint.link_len_a) == 88.0
    assert pytest.approx(joint.link_len_b) == 70.0
    assert pytest.approx(joint.twist_angle_alpha) == 90.0
    assert pytest.approx(joint.offset_distance_s) == 36.0


def test_m383_lagmul_baker_spatial_linkage_joint_free(tmp_path: Path):
    deck = """# RADIOSS FREE DECK
/BEGIN
Baker Spatial Linkage Joint Free Format Test
/LAGMUL/BAKER_SPATIAL_LINKAGE_JOINT/226
361, 362, 363, 1.40e7, 28, 6.0e-5
100.0, 76.0, 110.0, 48.0
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 226 in model.lagmul_baker_spatial_linkage_joints
    joint = model.lagmul_baker_spatial_linkage_joints[226]
    assert joint.node1 == 361
    assert joint.node2 == 362
    assert joint.node3 == 363
    assert pytest.approx(joint.stiff) == 1.40e7
    assert joint.skew_id == 28
    assert pytest.approx(joint.tol) == 6.0e-5
    assert pytest.approx(joint.link_len_a) == 100.0
    assert pytest.approx(joint.link_len_b) == 76.0
    assert pytest.approx(joint.twist_angle_alpha) == 110.0
    assert pytest.approx(joint.offset_distance_s) == 48.0


def test_m383_lagmul_baker_spatial_linkage_joint_aliases(tmp_path: Path):
    deck = """# RADIOSS ALIAS DECK
/BEGIN
Baker Spatial Linkage Joint Aliases Test
/BAKER_SPATIAL_LINKAGE_JOINT/227
461, 462, 463, 1.0e6, 0, 1.0e-6
62.0, 52.0, 62.0, 21.0
/LAGMUL/BAKER_SPATIAL_LINKAGE/228
461, 462, 463, 1.0e6, 0, 1.0e-6
62.0, 52.0, 62.0, 21.0
/BAKER_SPATIAL_LINKAGE/229
461, 462, 463, 1.0e6, 0, 1.0e-6
62.0, 52.0, 62.0, 21.0
/BAKER_SPATIAL_MULTI_LOOP_MECHANISM/230
461, 462, 463, 1.0e6, 0, 1.0e-6
62.0, 52.0, 62.0, 21.0
/BAKER_SPATIAL_SYMMETRIC_MECHANISM/231
461, 462, 463, 1.0e6, 0, 1.0e-6
62.0, 52.0, 62.0, 21.0
/BAKER_SPATIAL_6R_MECHANISM/232
461, 462, 463, 1.0e6, 0, 1.0e-6
62.0, 52.0, 62.0, 21.0
/BAKER_SPATIAL_OVERCONSTRAINED_MECHANISM/233
461, 462, 463, 1.0e6, 0, 1.0e-6
62.0, 52.0, 62.0, 21.0
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    for jid in range(227, 234):
        assert jid in model.lagmul_baker_spatial_linkage_joints


def test_m383_lagmul_baker_spatial_linkage_joint_missing_card(tmp_path: Path):
    deck = """# RADIOSS MISSING CARD DECK
/BEGIN
/LAGMUL/BAKER_SPATIAL_LINKAGE_JOINT/234
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) > 0


def test_m383_sensor_spring_bending_lock_rate_fixed(tmp_path: Path):
    c1 = f"{1308:>10d}{2.28e8:>20.4f}{0.0695:>20.4f}"
    deck = f"""# RADIOSS STARTER DECK
/BEGIN
Sensor Spring Bending Lock Rate Fixed Format Test
2022 0
/SENSOR/SPRING_BENDING_LOCK_RATE/1
Fixed Spring Bending Lock Rate Sensor
{c1}
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 1 in model.sensor_spring_bending_lock_rates
    s1 = model.sensor_spring_bending_lock_rates[1]
    assert s1.spring_id == 1308
    assert pytest.approx(s1.jbend_lock_max) == 2.28e8
    assert pytest.approx(s1.jbend_pop_max) == 2.28e8
    assert pytest.approx(s1.jbend_snp_max) == 2.28e8
    assert pytest.approx(s1.jbend_crackle_max) == 2.28e8
    assert pytest.approx(s1.t_delay) == 0.0695
    assert len(model.sensors) == 1
    assert model.sensors[0].kind == "SPRING_BENDING_LOCK_RATE"


def test_m383_sensor_spring_bending_lock_rate_free(tmp_path: Path):
    deck = """# RADIOSS FREE DECK
/BEGIN
Sensor Spring Bending Lock Rate Free Format Test
/SENSOR/SPRING_BENDING_LOCK_RATE/2
Free Spring Bending Lock Rate Sensor
1309, 2.88e8, 0.0845
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 2 in model.sensor_spring_bending_lock_rates
    s2 = model.sensor_spring_bending_lock_rates[2]
    assert s2.spring_id == 1309
    assert pytest.approx(s2.jbend_lock_max) == 2.88e8
    assert pytest.approx(s2.t_delay) == 0.0845


def test_m383_sensor_spring_bending_lock_rate_aliases(tmp_path: Path):
    deck = """# RADIOSS ALIAS DECK
/BEGIN
Sensor Spring Bending Lock Rate Aliases Test
/SENSOR/SPRING_BEND_LOCK_RATE/330
1329, 1.135e8, 0.0150
/SENSOR/SPRING_RATE_LOCK_BEND/331
1330, 1.155e8, 0.0160
/SENSOR/BENDING_LOCK_RATE_SPRING/332
1331, 1.175e8, 0.0170
/SENSOR/SPRING_LOCK_BEND/333
1332, 1.195e8, 0.0180
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    for sid in range(330, 334):
        assert sid in model.sensor_spring_bending_lock_rates


def test_m383_sensor_spring_bending_lock_rate_missing_card(tmp_path: Path):
    deck = """# RADIOSS MISSING CARD DECK
/BEGIN
/SENSOR/SPRING_BENDING_LOCK_RATE/335
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) > 0
