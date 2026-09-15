"""Tests for Milestone M379: LadCoupleFiberMatrixDebondingRate Failure Model, EngFlexomagnetophononicmagnonicpolaritonicResonanceEnergy, KirkpatrickHybridSpatialLinkageJoint, and SensorSpringNormalLockRate."""

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


def test_m379_fail_lad_couple_fiber_matrix_debonding_rate_fixed(tmp_path: Path):
    c1 = f"{225.0:>20.4f}{675.0:>20.4f}{102.0:>20.4f}{3.65:>20.4f}{0.935:>20.4f}"
    c2 = f"{1:>10d}{2:>10d}"
    c3 = f"{1930:>10d}"
    deck = f"""# RADIOSS STARTER DECK
/BEGIN
Ladeveze Coupled Fiber-Matrix Debonding Rate Fixed Format Test
2022 0
/FAIL/LAD_COUPLE_FIBER_MATRIX_DEBONDING_RATE/1930
Ladeveze Coupled Fiber-Matrix Debonding Failure Model
{c1}
{c2}
{c3}
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 1930 in model.fail_ladcouplefibermatrixdebondingrates
    fcfmdr = model.fail_ladcouplefibermatrixdebondingrates[1930]
    assert pytest.approx(fcfmdr.sigma_cfmdr0) == 225.0
    assert pytest.approx(fcfmdr.sigma_cfmdrc) == 675.0
    assert pytest.approx(fcfmdr.gamma_cfmdr) == 102.0
    assert pytest.approx(fcfmdr.p_cfmdr) == 3.65
    assert pytest.approx(fcfmdr.d_cfmdr_max) == 0.935
    assert fcfmdr.ifail_sh == 1
    assert fcfmdr.ifail_so == 2
    assert fcfmdr.fail_id == 1930
    assert len(model.raw_fails) == 1
    assert model.raw_fails[0][1].type == "LAD_COUPLE_FIBER_MATRIX_DEBONDING_RATE"


def test_m379_fail_lad_couple_fiber_matrix_debonding_rate_free(tmp_path: Path):
    deck = """# RADIOSS FREE DECK
/BEGIN
Ladeveze Coupled Fiber-Matrix Debonding Rate Free Format Test
/FAIL/LAD_COUPLE_FIBER_MATRIX_DEBONDING_RATE/1931
235.0, 705.0, 108.0, 3.85, 0.910
1, 1
1931
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 1931 in model.fail_ladcouplefibermatrixdebondingrates
    fcfmdr = model.fail_ladcouplefibermatrixdebondingrates[1931]
    assert pytest.approx(fcfmdr.sigma_cfmdr0) == 235.0
    assert pytest.approx(fcfmdr.sigma_cfmdrc) == 705.0
    assert pytest.approx(fcfmdr.gamma_cfmdr) == 108.0
    assert pytest.approx(fcfmdr.p_cfmdr) == 3.85
    assert pytest.approx(fcfmdr.d_cfmdr_max) == 0.910
    assert fcfmdr.fail_id == 1931


def test_m379_fail_lad_couple_fiber_matrix_debonding_rate_aliases(tmp_path: Path):
    deck = """# RADIOSS ALIAS DECK
/BEGIN
Ladeveze Coupled Fiber-Matrix Debonding Rate Aliases Test
/FAIL/LADEVEZE_COUPLED_FIBER_MATRIX_DEBONDING_RATE/1932
215.0, 645.0, 95.0, 3.35, 0.955
1, 1
/FAIL/LAD_CFMDR/1933
215.0, 645.0, 95.0, 3.35, 0.955
1, 1
/FAIL/LAD_CFMDR_MODEL/1934
215.0, 645.0, 95.0, 3.35, 0.955
1, 1
/FAIL/LAD_CFMDR_LAW/1935
215.0, 645.0, 95.0, 3.35, 0.955
1, 1
/FAIL/LADEVEZE_RATE_DEPENDENT_COUPLED_FIBER_MATRIX_DEBONDING/1936
215.0, 645.0, 95.0, 3.35, 0.955
1, 1
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 1932 in model.fail_ladcouplefibermatrixdebondingrates
    assert 1933 in model.fail_ladcouplefibermatrixdebondingrates
    assert 1934 in model.fail_ladcouplefibermatrixdebondingrates
    assert 1935 in model.fail_ladcouplefibermatrixdebondingrates
    assert 1936 in model.fail_ladcouplefibermatrixdebondingrates
    assert len(model.raw_fails) == 5


def test_m379_fail_lad_couple_fiber_matrix_debonding_rate_missing_card(tmp_path: Path):
    deck = """# RADIOSS MISSING CARD DECK
/BEGIN
/FAIL/LAD_COUPLE_FIBER_MATRIX_DEBONDING_RATE/1937
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) > 0


def test_m379_eng_flexomagnetophononicmagnonicpolaritonic_resonance_energy_fixed(tmp_path: Path):
    c1 = f"{0.0060:>20.4f}{235:>10d}"
    deck = f"""# RADIOSS STARTER DECK
/BEGIN
Engine Flexomagnetophononicmagnonicpolaritonic Resonance Energy Fixed Format Test
2022 0
/ENG/FLEXOMAGNETOPHONONICMAGNONICPOLARITONIC_RESONANCE_ENERGY/135
Resonance Energy Output Directive
{c1}
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 135 in model.eng_flexomagnetophononicmagnonicpolaritonic_resonance_energies
    eng = model.eng_flexomagnetophononicmagnonicpolaritonic_resonance_energies[135]
    assert pytest.approx(eng.dt_fmpmpr) == 0.0060
    assert eng.sens_id == 235


def test_m379_eng_flexomagnetophononicmagnonicpolaritonic_resonance_energy_free(tmp_path: Path):
    deck = """# RADIOSS FREE DECK
/BEGIN
Engine Flexomagnetophononicmagnonicpolaritonic Resonance Energy Free Format Test
/ENG/FLEXOMAGNETOPHONONICMAGNONICPOLARITONIC_RESONANCE_ENERGY/136
0.0070, 236
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 136 in model.eng_flexomagnetophononicmagnonicpolaritonic_resonance_energies
    eng = model.eng_flexomagnetophononicmagnonicpolaritonic_resonance_energies[136]
    assert pytest.approx(eng.dt_fmpmpr) == 0.0070
    assert eng.sens_id == 236


def test_m379_eng_flexomagnetophononicmagnonicpolaritonic_resonance_energy_aliases(tmp_path: Path):
    deck = """# RADIOSS ALIAS DECK
/BEGIN
Engine Flexomagnetophononicmagnonicpolaritonic Resonance Energy Aliases Test
/ENG/FLEXOMAGNETO_PHONON_MAGNON_POLARITON_RES_WORK/137
0.0080, 237
/ENG/EFLEXOMAGNETOPHONONICMAGNONICPOLARITONICRESONANCE/138
0.0090, 238
/ENG/FLEXOMAGNETOPHONONICMAGNONICPOLARITONIC_RESONANCE_DISSIPATION/139
0.0100, 239
/ENG/EM_FLEXOMAGNETOPHONONICMAGNONICPOLARITONIC_RESONANCE/140
0.0110, 240
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 137 in model.eng_flexomagnetophononicmagnonicpolaritonic_resonance_energies
    assert 138 in model.eng_flexomagnetophononicmagnonicpolaritonic_resonance_energies
    assert 139 in model.eng_flexomagnetophononicmagnonicpolaritonic_resonance_energies
    assert 140 in model.eng_flexomagnetophononicmagnonicpolaritonic_resonance_energies


def test_m379_eng_flexomagnetophononicmagnonicpolaritonic_resonance_energy_missing_card(tmp_path: Path):
    deck = """# RADIOSS MISSING CARD DECK
/BEGIN
/ENG/FLEXOMAGNETOPHONONICMAGNONICPOLARITONIC_RESONANCE_ENERGY/141
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) > 0


def test_m379_lagmul_kirkpatrick_hybrid_spatial_linkage_joint_fixed(tmp_path: Path):
    c1 = f"{221:>10d}{222:>10d}{223:>10d}{1.02e7:>20.4f}{19:>10d}{4.2e-5:>20.4e}"
    c2 = f"{80.0:>20.4f}{62.0:>20.4f}{82.0:>20.4f}{28.0:>20.4f}"
    deck = f"""# RADIOSS STARTER DECK
/BEGIN
Kirkpatrick Hybrid Spatial Linkage Joint Fixed Format Test
2022 0
/LAGMUL/KIRKPATRICK_HYBRID_SPATIAL_LINKAGE_JOINT/185
Kirkpatrick Hybrid Spatial 6R Joint
{c1}
{c2}
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 185 in model.lagmul_kirkpatrick_hybrid_spatial_linkage_joints
    joint = model.lagmul_kirkpatrick_hybrid_spatial_linkage_joints[185]
    assert joint.node1 == 221
    assert joint.node2 == 222
    assert joint.node3 == 223
    assert pytest.approx(joint.stiff) == 1.02e7
    assert joint.skew_id == 19
    assert pytest.approx(joint.tol) == 4.2e-5
    assert pytest.approx(joint.link_len_a) == 80.0
    assert pytest.approx(joint.link_len_b) == 62.0
    assert pytest.approx(joint.twist_angle_alpha) == 82.0
    assert pytest.approx(joint.offset_distance_s) == 28.0


def test_m379_lagmul_kirkpatrick_hybrid_spatial_linkage_joint_free(tmp_path: Path):
    deck = """# RADIOSS FREE DECK
/BEGIN
Kirkpatrick Hybrid Spatial Linkage Joint Free Format Test
/LAGMUL/KIRKPATRICK_HYBRID_SPATIAL_LINKAGE_JOINT/186
321, 322, 323, 1.20e7, 20, 5.2e-5
92.0, 68.0, 102.0, 40.0
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 186 in model.lagmul_kirkpatrick_hybrid_spatial_linkage_joints
    joint = model.lagmul_kirkpatrick_hybrid_spatial_linkage_joints[186]
    assert joint.node1 == 321
    assert joint.node2 == 322
    assert joint.node3 == 323
    assert pytest.approx(joint.stiff) == 1.20e7
    assert joint.skew_id == 20
    assert pytest.approx(joint.tol) == 5.2e-5
    assert pytest.approx(joint.link_len_a) == 92.0
    assert pytest.approx(joint.link_len_b) == 68.0
    assert pytest.approx(joint.twist_angle_alpha) == 102.0
    assert pytest.approx(joint.offset_distance_s) == 40.0


def test_m379_lagmul_kirkpatrick_hybrid_spatial_linkage_joint_aliases(tmp_path: Path):
    deck = """# RADIOSS ALIAS DECK
/BEGIN
Kirkpatrick Hybrid Spatial Linkage Joint Aliases Test
/KIRKPATRICK_HYBRID_SPATIAL_LINKAGE_JOINT/187
421, 422, 423, 1.0e6, 0, 1.0e-6
54.0, 44.0, 54.0, 17.0
/LAGMUL/KIRKPATRICK_HYBRID_SPATIAL_LINKAGE/188
421, 422, 423, 1.0e6, 0, 1.0e-6
54.0, 44.0, 54.0, 17.0
/KIRKPATRICK_HYBRID_SPATIAL_LINKAGE/189
421, 422, 423, 1.0e6, 0, 1.0e-6
54.0, 44.0, 54.0, 17.0
/KIRKPATRICK_HYBRID_SPATIAL_MULTI_LOOP_MECHANISM/190
421, 422, 423, 1.0e6, 0, 1.0e-6
54.0, 44.0, 54.0, 17.0
/KIRKPATRICK_HYBRID_SPATIAL_SYMMETRIC_MECHANISM/191
421, 422, 423, 1.0e6, 0, 1.0e-6
54.0, 44.0, 54.0, 17.0
/KIRKPATRICK_HYBRID_SPATIAL_6R_MECHANISM/192
421, 422, 423, 1.0e6, 0, 1.0e-6
54.0, 44.0, 54.0, 17.0
/KIRKPATRICK_HYBRID_SPATIAL_OVERCONSTRAINED_MECHANISM/193
421, 422, 423, 1.0e6, 0, 1.0e-6
54.0, 44.0, 54.0, 17.0
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    for jid in range(187, 194):
        assert jid in model.lagmul_kirkpatrick_hybrid_spatial_linkage_joints


def test_m379_lagmul_kirkpatrick_hybrid_spatial_linkage_joint_missing_card(tmp_path: Path):
    deck = """# RADIOSS MISSING CARD DECK
/BEGIN
/LAGMUL/KIRKPATRICK_HYBRID_SPATIAL_LINKAGE_JOINT/194
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) > 0


def test_m379_sensor_spring_normal_lock_rate_fixed(tmp_path: Path):
    c1 = f"{1268:>10d}{1.88e8:>20.4f}{0.0575:>20.4f}"
    deck = f"""# RADIOSS STARTER DECK
/BEGIN
Sensor Spring Normal Lock Rate Fixed Format Test
2022 0
/SENSOR/SPRING_NORMAL_LOCK_RATE/1
Fixed Spring Normal Lock Rate Sensor
{c1}
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 1 in model.sensor_spring_normal_lock_rates
    s1 = model.sensor_spring_normal_lock_rates[1]
    assert s1.spring_id == 1268
    assert pytest.approx(s1.jnorm_lock_max) == 1.88e8
    assert pytest.approx(s1.jnorm_pop_max) == 1.88e8
    assert pytest.approx(s1.jnorm_snp_max) == 1.88e8
    assert pytest.approx(s1.jnorm_crackle_max) == 1.88e8
    assert pytest.approx(s1.t_delay) == 0.0575
    assert len(model.sensors) == 1
    assert model.sensors[0].kind == "SPRING_NORMAL_LOCK_RATE"


def test_m379_sensor_spring_normal_lock_rate_free(tmp_path: Path):
    deck = """# RADIOSS FREE DECK
/BEGIN
Sensor Spring Normal Lock Rate Free Format Test
/SENSOR/SPRING_NORMAL_LOCK_RATE/2
Free Spring Normal Lock Rate Sensor
1269, 2.48e8, 0.0725
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 2 in model.sensor_spring_normal_lock_rates
    s2 = model.sensor_spring_normal_lock_rates[2]
    assert s2.spring_id == 1269
    assert pytest.approx(s2.jnorm_lock_max) == 2.48e8
    assert pytest.approx(s2.t_delay) == 0.0725


def test_m379_sensor_spring_normal_lock_rate_aliases(tmp_path: Path):
    deck = """# RADIOSS ALIAS DECK
/BEGIN
Sensor Spring Normal Lock Rate Aliases Test
/SENSOR/SPRING_NORM_LOCK_RATE/290
1289, 1.055e8, 0.0130
/SENSOR/SPRING_RATE_LOCK_NORM/291
1290, 1.075e8, 0.0140
/SENSOR/NORMAL_LOCK_RATE_SPRING/292
1291, 1.095e8, 0.0150
/SENSOR/SPRING_LOCK_NORM/293
1292, 1.115e8, 0.0160
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    for sid in range(290, 294):
        assert sid in model.sensor_spring_normal_lock_rates


def test_m379_sensor_spring_normal_lock_rate_missing_card(tmp_path: Path):
    deck = """# RADIOSS MISSING CARD DECK
/BEGIN
/SENSOR/SPRING_NORMAL_LOCK_RATE/295
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) > 0
