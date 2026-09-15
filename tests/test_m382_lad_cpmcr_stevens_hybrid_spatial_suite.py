"""Tests for Milestone M382: LadCouplePlyMicroCrackingRate Failure Model, EngFlexomagnetophononicplasmonicpolaritonicResonanceEnergy, StevensHybridSpatialLinkageJoint, and SensorSpringTorsionalLockRate."""

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


def test_m382_fail_lad_couple_ply_micro_cracking_rate_fixed(tmp_path: Path):
    c1 = f"{240.0:>20.4f}{720.0:>20.4f}{110.0:>20.4f}{3.80:>20.4f}{0.920:>20.4f}"
    c2 = f"{1:>10d}{2:>10d}"
    c3 = f"{1960:>10d}"
    deck = f"""# RADIOSS STARTER DECK
/BEGIN
Ladeveze Coupled Ply Micro-Cracking Rate Fixed Format Test
2022 0
/FAIL/LAD_COUPLE_PLY_MICRO_CRACKING_RATE/1960
Ladeveze Coupled Ply Micro-Cracking Failure Model
{c1}
{c2}
{c3}
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 1960 in model.fail_ladcoupleplymicrocrackingrates
    fcpmcr = model.fail_ladcoupleplymicrocrackingrates[1960]
    assert pytest.approx(fcpmcr.sigma_cpmcr0) == 240.0
    assert pytest.approx(fcpmcr.sigma_cpmcrc) == 720.0
    assert pytest.approx(fcpmcr.gamma_cpmcr) == 110.0
    assert pytest.approx(fcpmcr.p_cpmcr) == 3.80
    assert pytest.approx(fcpmcr.d_cpmcr_max) == 0.920
    assert fcpmcr.ifail_sh == 1
    assert fcpmcr.ifail_so == 2
    assert fcpmcr.fail_id == 1960
    assert len(model.raw_fails) == 1
    assert model.raw_fails[0][1].type == "LAD_COUPLE_PLY_MICRO_CRACKING_RATE"


def test_m382_fail_lad_couple_ply_micro_cracking_rate_free(tmp_path: Path):
    deck = """# RADIOSS FREE DECK
/BEGIN
Ladeveze Coupled Ply Micro-Cracking Rate Free Format Test
/FAIL/LAD_COUPLE_PLY_MICRO_CRACKING_RATE/1961
250.0, 750.0, 115.0, 4.00, 0.895
1, 1
1961
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 1961 in model.fail_ladcoupleplymicrocrackingrates
    fcpmcr = model.fail_ladcoupleplymicrocrackingrates[1961]
    assert pytest.approx(fcpmcr.sigma_cpmcr0) == 250.0
    assert pytest.approx(fcpmcr.sigma_cpmcrc) == 750.0
    assert pytest.approx(fcpmcr.gamma_cpmcr) == 115.0
    assert pytest.approx(fcpmcr.p_cpmcr) == 4.00
    assert pytest.approx(fcpmcr.d_cpmcr_max) == 0.895
    assert fcpmcr.fail_id == 1961


def test_m382_fail_lad_couple_ply_micro_cracking_rate_aliases(tmp_path: Path):
    deck = """# RADIOSS ALIAS DECK
/BEGIN
Ladeveze Coupled Ply Micro-Cracking Rate Aliases Test
/FAIL/LADEVEZE_COUPLED_PLY_MICRO_CRACKING_RATE/1962
230.0, 690.0, 102.0, 3.50, 0.940
1, 1
/FAIL/LAD_CPMCR/1963
230.0, 690.0, 102.0, 3.50, 0.940
1, 1
/FAIL/LAD_CPMCR_MODEL/1964
230.0, 690.0, 102.0, 3.50, 0.940
1, 1
/FAIL/LAD_CPMCR_LAW/1965
230.0, 690.0, 102.0, 3.50, 0.940
1, 1
/FAIL/LADEVEZE_RATE_DEPENDENT_COUPLED_PLY_MICRO_CRACKING/1966
230.0, 690.0, 102.0, 3.50, 0.940
1, 1
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 1962 in model.fail_ladcoupleplymicrocrackingrates
    assert 1963 in model.fail_ladcoupleplymicrocrackingrates
    assert 1964 in model.fail_ladcoupleplymicrocrackingrates
    assert 1965 in model.fail_ladcoupleplymicrocrackingrates
    assert 1966 in model.fail_ladcoupleplymicrocrackingrates
    assert len(model.raw_fails) == 5


def test_m382_fail_lad_couple_ply_micro_cracking_rate_missing_card(tmp_path: Path):
    deck = """# RADIOSS MISSING CARD DECK
/BEGIN
/FAIL/LAD_COUPLE_PLY_MICRO_CRACKING_RATE/1967
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) > 0


def test_m382_eng_flexomagnetophononicplasmonicpolaritonic_resonance_energy_fixed(tmp_path: Path):
    c1 = f"{0.0066:>20.4f}{265:>10d}"
    deck = f"""# RADIOSS STARTER DECK
/BEGIN
Engine Flexomagnetophononicplasmonicpolaritonic Resonance Energy Fixed Format Test
2022 0
/ENG/FLEXOMAGNETOPHONONICPLASMONICPOLARITONIC_RESONANCE_ENERGY/165
Resonance Energy Output Directive
{c1}
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 165 in model.eng_flexomagnetophononicplasmonicpolaritonic_resonance_energies
    eng = model.eng_flexomagnetophononicplasmonicpolaritonic_resonance_energies[165]
    assert pytest.approx(eng.dt_fmpopr) == 0.0066
    assert eng.sens_id == 265


def test_m382_eng_flexomagnetophononicplasmonicpolaritonic_resonance_energy_free(tmp_path: Path):
    deck = """# RADIOSS FREE DECK
/BEGIN
Engine Flexomagnetophononicplasmonicpolaritonic Resonance Energy Free Format Test
/ENG/FLEXOMAGNETOPHONONICPLASMONICPOLARITONIC_RESONANCE_ENERGY/166
0.0076, 266
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 166 in model.eng_flexomagnetophononicplasmonicpolaritonic_resonance_energies
    eng = model.eng_flexomagnetophononicplasmonicpolaritonic_resonance_energies[166]
    assert pytest.approx(eng.dt_fmpopr) == 0.0076
    assert eng.sens_id == 266


def test_m382_eng_flexomagnetophononicplasmonicpolaritonic_resonance_energy_aliases(tmp_path: Path):
    deck = """# RADIOSS ALIAS DECK
/BEGIN
Engine Flexomagnetophononicplasmonicpolaritonic Resonance Energy Aliases Test
/ENG/FLEXOMAGNETO_PHONON_PLASMON_POLARITON_RES_WORK/167
0.0086, 267
/ENG/EFLEXOMAGNETOPHONONICPLASMONICPOLARITONICRESONANCE/168
0.0096, 268
/ENG/FLEXOMAGNETOPHONONICPLASMONICPOLARITONIC_RESONANCE_DISSIPATION/169
0.0106, 269
/ENG/EM_FLEXOMAGNETOPHONONICPLASMONICPOLARITONIC_RESONANCE/170
0.0116, 270
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 167 in model.eng_flexomagnetophononicplasmonicpolaritonic_resonance_energies
    assert 168 in model.eng_flexomagnetophononicplasmonicpolaritonic_resonance_energies
    assert 169 in model.eng_flexomagnetophononicplasmonicpolaritonic_resonance_energies
    assert 170 in model.eng_flexomagnetophononicplasmonicpolaritonic_resonance_energies


def test_m382_eng_flexomagnetophononicplasmonicpolaritonic_resonance_energy_missing_card(tmp_path: Path):
    deck = """# RADIOSS MISSING CARD DECK
/BEGIN
/ENG/FLEXOMAGNETOPHONONICPLASMONICPOLARITONIC_RESONANCE_ENERGY/171
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) > 0


def test_m382_lagmul_stevens_hybrid_spatial_linkage_joint_fixed(tmp_path: Path):
    c1 = f"{251:>10d}{252:>10d}{253:>10d}{1.12e7:>20.4f}{25:>10d}{4.8e-5:>20.4e}"
    c2 = f"{86.0:>20.4f}{68.0:>20.4f}{88.0:>20.4f}{34.0:>20.4f}"
    deck = f"""# RADIOSS STARTER DECK
/BEGIN
Stevens Hybrid Spatial Linkage Joint Fixed Format Test
2022 0
/LAGMUL/STEVENS_HYBRID_SPATIAL_LINKAGE_JOINT/215
Stevens Hybrid Spatial 6R Joint
{c1}
{c2}
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 215 in model.lagmul_stevens_hybrid_spatial_linkage_joints
    joint = model.lagmul_stevens_hybrid_spatial_linkage_joints[215]
    assert joint.node1 == 251
    assert joint.node2 == 252
    assert joint.node3 == 253
    assert pytest.approx(joint.stiff) == 1.12e7
    assert joint.skew_id == 25
    assert pytest.approx(joint.tol) == 4.8e-5
    assert pytest.approx(joint.link_len_a) == 86.0
    assert pytest.approx(joint.link_len_b) == 68.0
    assert pytest.approx(joint.twist_angle_alpha) == 88.0
    assert pytest.approx(joint.offset_distance_s) == 34.0


def test_m382_lagmul_stevens_hybrid_spatial_linkage_joint_free(tmp_path: Path):
    deck = """# RADIOSS FREE DECK
/BEGIN
Stevens Hybrid Spatial Linkage Joint Free Format Test
/LAGMUL/STEVENS_HYBRID_SPATIAL_LINKAGE_JOINT/216
351, 352, 353, 1.35e7, 26, 5.8e-5
98.0, 74.0, 108.0, 46.0
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 216 in model.lagmul_stevens_hybrid_spatial_linkage_joints
    joint = model.lagmul_stevens_hybrid_spatial_linkage_joints[216]
    assert joint.node1 == 351
    assert joint.node2 == 352
    assert joint.node3 == 353
    assert pytest.approx(joint.stiff) == 1.35e7
    assert joint.skew_id == 26
    assert pytest.approx(joint.tol) == 5.8e-5
    assert pytest.approx(joint.link_len_a) == 98.0
    assert pytest.approx(joint.link_len_b) == 74.0
    assert pytest.approx(joint.twist_angle_alpha) == 108.0
    assert pytest.approx(joint.offset_distance_s) == 46.0


def test_m382_lagmul_stevens_hybrid_spatial_linkage_joint_aliases(tmp_path: Path):
    deck = """# RADIOSS ALIAS DECK
/BEGIN
Stevens Hybrid Spatial Linkage Joint Aliases Test
/STEVENS_HYBRID_SPATIAL_LINKAGE_JOINT/217
451, 452, 453, 1.0e6, 0, 1.0e-6
60.0, 50.0, 60.0, 20.0
/LAGMUL/STEVENS_HYBRID_SPATIAL_LINKAGE/218
451, 452, 453, 1.0e6, 0, 1.0e-6
60.0, 50.0, 60.0, 20.0
/STEVENS_HYBRID_SPATIAL_LINKAGE/219
451, 452, 453, 1.0e6, 0, 1.0e-6
60.0, 50.0, 60.0, 20.0
/STEVENS_HYBRID_SPATIAL_MULTI_LOOP_MECHANISM/220
451, 452, 453, 1.0e6, 0, 1.0e-6
60.0, 50.0, 60.0, 20.0
/STEVENS_HYBRID_SPATIAL_SYMMETRIC_MECHANISM/221
451, 452, 453, 1.0e6, 0, 1.0e-6
60.0, 50.0, 60.0, 20.0
/STEVENS_HYBRID_SPATIAL_6R_MECHANISM/222
451, 452, 453, 1.0e6, 0, 1.0e-6
60.0, 50.0, 60.0, 20.0
/STEVENS_HYBRID_SPATIAL_OVERCONSTRAINED_MECHANISM/223
451, 452, 453, 1.0e6, 0, 1.0e-6
60.0, 50.0, 60.0, 20.0
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    for jid in range(217, 224):
        assert jid in model.lagmul_stevens_hybrid_spatial_linkage_joints


def test_m382_lagmul_stevens_hybrid_spatial_linkage_joint_missing_card(tmp_path: Path):
    deck = """# RADIOSS MISSING CARD DECK
/BEGIN
/LAGMUL/STEVENS_HYBRID_SPATIAL_LINKAGE_JOINT/224
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) > 0


def test_m382_sensor_spring_torsional_lock_rate_fixed(tmp_path: Path):
    c1 = f"{1298:>10d}{2.18e8:>20.4f}{0.0665:>20.4f}"
    deck = f"""# RADIOSS STARTER DECK
/BEGIN
Sensor Spring Torsional Lock Rate Fixed Format Test
2022 0
/SENSOR/SPRING_TORSIONAL_LOCK_RATE/1
Fixed Spring Torsional Lock Rate Sensor
{c1}
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 1 in model.sensor_spring_torsional_lock_rates
    s1 = model.sensor_spring_torsional_lock_rates[1]
    assert s1.spring_id == 1298
    assert pytest.approx(s1.jtors_lock_max) == 2.18e8
    assert pytest.approx(s1.jtors_pop_max) == 2.18e8
    assert pytest.approx(s1.jtors_snp_max) == 2.18e8
    assert pytest.approx(s1.jtors_crackle_max) == 2.18e8
    assert pytest.approx(s1.t_delay) == 0.0665
    assert len(model.sensors) == 1
    assert model.sensors[0].kind == "SPRING_TORSIONAL_LOCK_RATE"


def test_m382_sensor_spring_torsional_lock_rate_free(tmp_path: Path):
    deck = """# RADIOSS FREE DECK
/BEGIN
Sensor Spring Torsional Lock Rate Free Format Test
/SENSOR/SPRING_TORSIONAL_LOCK_RATE/2
Free Spring Torsional Lock Rate Sensor
1299, 2.78e8, 0.0815
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 2 in model.sensor_spring_torsional_lock_rates
    s2 = model.sensor_spring_torsional_lock_rates[2]
    assert s2.spring_id == 1299
    assert pytest.approx(s2.jtors_lock_max) == 2.78e8
    assert pytest.approx(s2.t_delay) == 0.0815


def test_m382_sensor_spring_torsional_lock_rate_aliases(tmp_path: Path):
    deck = """# RADIOSS ALIAS DECK
/BEGIN
Sensor Spring Torsional Lock Rate Aliases Test
/SENSOR/SPRING_TORS_LOCK_RATE/320
1319, 1.125e8, 0.0145
/SENSOR/SPRING_RATE_LOCK_TORS/321
1320, 1.145e8, 0.0155
/SENSOR/TORSIONAL_LOCK_RATE_SPRING/322
1321, 1.165e8, 0.0165
/SENSOR/SPRING_LOCK_TORS/323
1322, 1.185e8, 0.0175
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    for sid in range(320, 324):
        assert sid in model.sensor_spring_torsional_lock_rates


def test_m382_sensor_spring_torsional_lock_rate_missing_card(tmp_path: Path):
    deck = """# RADIOSS MISSING CARD DECK
/BEGIN
/SENSOR/SPRING_TORSIONAL_LOCK_RATE/325
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) > 0
