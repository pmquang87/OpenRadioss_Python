"""Tests for Milestone M381: LadTransversePlyMicroCrackingRate Failure Model, EngFlexomagnetophononicplasmonicmagnonicResonanceEnergy, ChungHybridSpatialLinkageJoint, and SensorSpringTotalLockRate."""

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


def test_m381_fail_lad_transverse_ply_micro_cracking_rate_fixed(tmp_path: Path):
    c1 = f"{235.0:>20.4f}{705.0:>20.4f}{108.0:>20.4f}{3.75:>20.4f}{0.925:>20.4f}"
    c2 = f"{1:>10d}{2:>10d}"
    c3 = f"{1950:>10d}"
    deck = f"""# RADIOSS STARTER DECK
/BEGIN
Ladeveze Transverse Ply Micro-Cracking Rate Fixed Format Test
2022 0
/FAIL/LAD_TRANSVERSE_PLY_MICRO_CRACKING_RATE/1950
Ladeveze Transverse Ply Micro-Cracking Failure Model
{c1}
{c2}
{c3}
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 1950 in model.fail_ladtransverseplymicrocrackingrates
    ftpmcr = model.fail_ladtransverseplymicrocrackingrates[1950]
    assert pytest.approx(ftpmcr.sigma_tpmcr0) == 235.0
    assert pytest.approx(ftpmcr.sigma_tpmcrc) == 705.0
    assert pytest.approx(ftpmcr.gamma_tpmcr) == 108.0
    assert pytest.approx(ftpmcr.p_tpmcr) == 3.75
    assert pytest.approx(ftpmcr.d_tpmcr_max) == 0.925
    assert ftpmcr.ifail_sh == 1
    assert ftpmcr.ifail_so == 2
    assert ftpmcr.fail_id == 1950
    assert len(model.raw_fails) == 1
    assert model.raw_fails[0][1].type == "LAD_TRANSVERSE_PLY_MICRO_CRACKING_RATE"


def test_m381_fail_lad_transverse_ply_micro_cracking_rate_free(tmp_path: Path):
    deck = """# RADIOSS FREE DECK
/BEGIN
Ladeveze Transverse Ply Micro-Cracking Rate Free Format Test
/FAIL/LAD_TRANSVERSE_PLY_MICRO_CRACKING_RATE/1951
245.0, 735.0, 112.0, 3.95, 0.900
1, 1
1951
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 1951 in model.fail_ladtransverseplymicrocrackingrates
    ftpmcr = model.fail_ladtransverseplymicrocrackingrates[1951]
    assert pytest.approx(ftpmcr.sigma_tpmcr0) == 245.0
    assert pytest.approx(ftpmcr.sigma_tpmcrc) == 735.0
    assert pytest.approx(ftpmcr.gamma_tpmcr) == 112.0
    assert pytest.approx(ftpmcr.p_tpmcr) == 3.95
    assert pytest.approx(ftpmcr.d_tpmcr_max) == 0.900
    assert ftpmcr.fail_id == 1951


def test_m381_fail_lad_transverse_ply_micro_cracking_rate_aliases(tmp_path: Path):
    deck = """# RADIOSS ALIAS DECK
/BEGIN
Ladeveze Transverse Ply Micro-Cracking Rate Aliases Test
/FAIL/LADEVEZE_TRANSVERSE_PLY_MICRO_CRACKING_RATE/1952
225.0, 675.0, 100.0, 3.45, 0.945
1, 1
/FAIL/LAD_TPMCR/1953
225.0, 675.0, 100.0, 3.45, 0.945
1, 1
/FAIL/LAD_TPMCR_MODEL/1954
225.0, 675.0, 100.0, 3.45, 0.945
1, 1
/FAIL/LAD_TPMCR_LAW/1955
225.0, 675.0, 100.0, 3.45, 0.945
1, 1
/FAIL/LADEVEZE_RATE_DEPENDENT_TRANSVERSE_PLY_MICRO_CRACKING/1956
225.0, 675.0, 100.0, 3.45, 0.945
1, 1
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 1952 in model.fail_ladtransverseplymicrocrackingrates
    assert 1953 in model.fail_ladtransverseplymicrocrackingrates
    assert 1954 in model.fail_ladtransverseplymicrocrackingrates
    assert 1955 in model.fail_ladtransverseplymicrocrackingrates
    assert 1956 in model.fail_ladtransverseplymicrocrackingrates
    assert len(model.raw_fails) == 5


def test_m381_fail_lad_transverse_ply_micro_cracking_rate_missing_card(tmp_path: Path):
    deck = """# RADIOSS MISSING CARD DECK
/BEGIN
/FAIL/LAD_TRANSVERSE_PLY_MICRO_CRACKING_RATE/1957
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) > 0


def test_m381_eng_flexomagnetophononicplasmonicmagnonic_resonance_energy_fixed(tmp_path: Path):
    c1 = f"{0.0064:>20.4f}{255:>10d}"
    deck = f"""# RADIOSS STARTER DECK
/BEGIN
Engine Flexomagnetophononicplasmonicmagnonic Resonance Energy Fixed Format Test
2022 0
/ENG/FLEXOMAGNETOPHONONICPLASMONICMAGNONIC_RESONANCE_ENERGY/155
Resonance Energy Output Directive
{c1}
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 155 in model.eng_flexomagnetophononicplasmonicmagnonic_resonance_energies
    eng = model.eng_flexomagnetophononicplasmonicmagnonic_resonance_energies[155]
    assert pytest.approx(eng.dt_fmpmr) == 0.0064
    assert eng.sens_id == 255


def test_m381_eng_flexomagnetophononicplasmonicmagnonic_resonance_energy_free(tmp_path: Path):
    deck = """# RADIOSS FREE DECK
/BEGIN
Engine Flexomagnetophononicplasmonicmagnonic Resonance Energy Free Format Test
/ENG/FLEXOMAGNETOPHONONICPLASMONICMAGNONIC_RESONANCE_ENERGY/156
0.0074, 256
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 156 in model.eng_flexomagnetophononicplasmonicmagnonic_resonance_energies
    eng = model.eng_flexomagnetophononicplasmonicmagnonic_resonance_energies[156]
    assert pytest.approx(eng.dt_fmpmr) == 0.0074
    assert eng.sens_id == 256


def test_m381_eng_flexomagnetophononicplasmonicmagnonic_resonance_energy_aliases(tmp_path: Path):
    deck = """# RADIOSS ALIAS DECK
/BEGIN
Engine Flexomagnetophononicplasmonicmagnonic Resonance Energy Aliases Test
/ENG/FLEXOMAGNETO_PHONON_PLASMON_MAGNON_RES_WORK/157
0.0084, 257
/ENG/EFLEXOMAGNETOPHONONICPLASMONICMAGNONICRESONANCE/158
0.0094, 258
/ENG/FLEXOMAGNETOPHONONICPLASMONICMAGNONIC_RESONANCE_DISSIPATION/159
0.0104, 259
/ENG/EM_FLEXOMAGNETOPHONONICPLASMONICMAGNONIC_RESONANCE/160
0.0114, 260
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 157 in model.eng_flexomagnetophononicplasmonicmagnonic_resonance_energies
    assert 158 in model.eng_flexomagnetophononicplasmonicmagnonic_resonance_energies
    assert 159 in model.eng_flexomagnetophononicplasmonicmagnonic_resonance_energies
    assert 160 in model.eng_flexomagnetophononicplasmonicmagnonic_resonance_energies


def test_m381_eng_flexomagnetophononicplasmonicmagnonic_resonance_energy_missing_card(tmp_path: Path):
    deck = """# RADIOSS MISSING CARD DECK
/BEGIN
/ENG/FLEXOMAGNETOPHONONICPLASMONICMAGNONIC_RESONANCE_ENERGY/161
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) > 0


def test_m381_lagmul_chung_hybrid_spatial_linkage_joint_fixed(tmp_path: Path):
    c1 = f"{241:>10d}{242:>10d}{243:>10d}{1.08e7:>20.4f}{23:>10d}{4.6e-5:>20.4e}"
    c2 = f"{84.0:>20.4f}{66.0:>20.4f}{86.0:>20.4f}{32.0:>20.4f}"
    deck = f"""# RADIOSS STARTER DECK
/BEGIN
Chung Hybrid Spatial Linkage Joint Fixed Format Test
2022 0
/LAGMUL/CHUNG_HYBRID_SPATIAL_LINKAGE_JOINT/205
Chung Hybrid Spatial 6R Joint
{c1}
{c2}
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 205 in model.lagmul_chung_hybrid_spatial_linkage_joints
    joint = model.lagmul_chung_hybrid_spatial_linkage_joints[205]
    assert joint.node1 == 241
    assert joint.node2 == 242
    assert joint.node3 == 243
    assert pytest.approx(joint.stiff) == 1.08e7
    assert joint.skew_id == 23
    assert pytest.approx(joint.tol) == 4.6e-5
    assert pytest.approx(joint.link_len_a) == 84.0
    assert pytest.approx(joint.link_len_b) == 66.0
    assert pytest.approx(joint.twist_angle_alpha) == 86.0
    assert pytest.approx(joint.offset_distance_s) == 32.0


def test_m381_lagmul_chung_hybrid_spatial_linkage_joint_free(tmp_path: Path):
    deck = """# RADIOSS FREE DECK
/BEGIN
Chung Hybrid Spatial Linkage Joint Free Format Test
/LAGMUL/CHUNG_HYBRID_SPATIAL_LINKAGE_JOINT/206
341, 342, 343, 1.30e7, 24, 5.6e-5
96.0, 72.0, 106.0, 44.0
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 206 in model.lagmul_chung_hybrid_spatial_linkage_joints
    joint = model.lagmul_chung_hybrid_spatial_linkage_joints[206]
    assert joint.node1 == 341
    assert joint.node2 == 342
    assert joint.node3 == 343
    assert pytest.approx(joint.stiff) == 1.30e7
    assert joint.skew_id == 24
    assert pytest.approx(joint.tol) == 5.6e-5
    assert pytest.approx(joint.link_len_a) == 96.0
    assert pytest.approx(joint.link_len_b) == 72.0
    assert pytest.approx(joint.twist_angle_alpha) == 106.0
    assert pytest.approx(joint.offset_distance_s) == 44.0


def test_m381_lagmul_chung_hybrid_spatial_linkage_joint_aliases(tmp_path: Path):
    deck = """# RADIOSS ALIAS DECK
/BEGIN
Chung Hybrid Spatial Linkage Joint Aliases Test
/CHUNG_HYBRID_SPATIAL_LINKAGE_JOINT/207
441, 442, 443, 1.0e6, 0, 1.0e-6
58.0, 48.0, 58.0, 19.0
/LAGMUL/CHUNG_HYBRID_SPATIAL_LINKAGE/208
441, 442, 443, 1.0e6, 0, 1.0e-6
58.0, 48.0, 58.0, 19.0
/CHUNG_HYBRID_SPATIAL_LINKAGE/209
441, 442, 443, 1.0e6, 0, 1.0e-6
58.0, 48.0, 58.0, 19.0
/CHUNG_HYBRID_SPATIAL_MULTI_LOOP_MECHANISM/210
441, 442, 443, 1.0e6, 0, 1.0e-6
58.0, 48.0, 58.0, 19.0
/CHUNG_HYBRID_SPATIAL_SYMMETRIC_MECHANISM/211
441, 442, 443, 1.0e6, 0, 1.0e-6
58.0, 48.0, 58.0, 19.0
/CHUNG_HYBRID_SPATIAL_6R_MECHANISM/212
441, 442, 443, 1.0e6, 0, 1.0e-6
58.0, 48.0, 58.0, 19.0
/CHUNG_HYBRID_SPATIAL_OVERCONSTRAINED_MECHANISM/213
441, 442, 443, 1.0e6, 0, 1.0e-6
58.0, 48.0, 58.0, 19.0
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    for jid in range(207, 214):
        assert jid in model.lagmul_chung_hybrid_spatial_linkage_joints


def test_m381_lagmul_chung_hybrid_spatial_linkage_joint_missing_card(tmp_path: Path):
    deck = """# RADIOSS MISSING CARD DECK
/BEGIN
/LAGMUL/CHUNG_HYBRID_SPATIAL_LINKAGE_JOINT/214
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) > 0


def test_m381_sensor_spring_total_lock_rate_fixed(tmp_path: Path):
    c1 = f"{1288:>10d}{2.08e8:>20.4f}{0.0635:>20.4f}"
    deck = f"""# RADIOSS STARTER DECK
/BEGIN
Sensor Spring Total Lock Rate Fixed Format Test
2022 0
/SENSOR/SPRING_TOTAL_LOCK_RATE/1
Fixed Spring Total Lock Rate Sensor
{c1}
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 1 in model.sensor_spring_total_lock_rates
    s1 = model.sensor_spring_total_lock_rates[1]
    assert s1.spring_id == 1288
    assert pytest.approx(s1.jtot_lock_max) == 2.08e8
    assert pytest.approx(s1.jtot_pop_max) == 2.08e8
    assert pytest.approx(s1.jtot_snp_max) == 2.08e8
    assert pytest.approx(s1.jtot_crackle_max) == 2.08e8
    assert pytest.approx(s1.t_delay) == 0.0635
    assert len(model.sensors) == 1
    assert model.sensors[0].kind == "SPRING_TOTAL_LOCK_RATE"


def test_m381_sensor_spring_total_lock_rate_free(tmp_path: Path):
    deck = """# RADIOSS FREE DECK
/BEGIN
Sensor Spring Total Lock Rate Free Format Test
/SENSOR/SPRING_TOTAL_LOCK_RATE/2
Free Spring Total Lock Rate Sensor
1289, 2.68e8, 0.0785
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 2 in model.sensor_spring_total_lock_rates
    s2 = model.sensor_spring_total_lock_rates[2]
    assert s2.spring_id == 1289
    assert pytest.approx(s2.jtot_lock_max) == 2.68e8
    assert pytest.approx(s2.t_delay) == 0.0785


def test_m381_sensor_spring_total_lock_rate_aliases(tmp_path: Path):
    deck = """# RADIOSS ALIAS DECK
/BEGIN
Sensor Spring Total Lock Rate Aliases Test
/SENSOR/SPRING_TOT_LOCK_RATE/310
1309, 1.115e8, 0.0140
/SENSOR/SPRING_RATE_LOCK_TOT/311
1310, 1.135e8, 0.0150
/SENSOR/TOTAL_LOCK_RATE_SPRING/312
1311, 1.155e8, 0.0160
/SENSOR/SPRING_LOCK_TOT/313
1312, 1.175e8, 0.0170
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    for sid in range(310, 314):
        assert sid in model.sensor_spring_total_lock_rates


def test_m381_sensor_spring_total_lock_rate_missing_card(tmp_path: Path):
    deck = """# RADIOSS MISSING CARD DECK
/BEGIN
/SENSOR/SPRING_TOTAL_LOCK_RATE/315
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) > 0
