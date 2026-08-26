"""Tests for Milestone M378: LadTransverseFiberMatrixDebondingRate Failure Model, EngFlexomagnetophononicexcitonicpolaritonicResonanceEnergy, AltmannHybridSpatialLinkageJoint, and SensorSpringTotalAngularPopRate."""

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


def test_m378_fail_lad_transverse_fiber_matrix_debonding_rate_fixed(tmp_path: Path):
    c1 = f"{215.0:>20.4f}{645.0:>20.4f}{96.0:>20.4f}{3.55:>20.4f}{0.940:>20.4f}"
    c2 = f"{1:>10d}{2:>10d}"
    c3 = f"{1920:>10d}"
    deck = f"""# RADIOSS STARTER DECK
/BEGIN
Ladeveze Transverse Fiber-Matrix Debonding Rate Fixed Format Test
2022 0
/FAIL/LAD_TRANSVERSE_FIBER_MATRIX_DEBONDING_RATE/1920
Ladeveze Transverse Fiber-Matrix Debonding Failure Model
{c1}
{c2}
{c3}
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 1920 in model.fail_ladtransversefibermatrixdebondingrates
    ftfmdr = model.fail_ladtransversefibermatrixdebondingrates[1920]
    assert pytest.approx(ftfmdr.sigma_tfmdr0) == 215.0
    assert pytest.approx(ftfmdr.sigma_tfmdrc) == 645.0
    assert pytest.approx(ftfmdr.gamma_tfmdr) == 96.0
    assert pytest.approx(ftfmdr.p_tfmdr) == 3.55
    assert pytest.approx(ftfmdr.d_tfmdr_max) == 0.940
    assert ftfmdr.ifail_sh == 1
    assert ftfmdr.ifail_so == 2
    assert ftfmdr.fail_id == 1920
    assert len(model.raw_fails) == 1
    assert model.raw_fails[0][1].type == "LAD_TRANSVERSE_FIBER_MATRIX_DEBONDING_RATE"


def test_m378_fail_lad_transverse_fiber_matrix_debonding_rate_free(tmp_path: Path):
    deck = """# RADIOSS FREE DECK
/BEGIN
Ladeveze Transverse Fiber-Matrix Debonding Rate Free Format Test
/FAIL/LAD_TRANSVERSE_FIBER_MATRIX_DEBONDING_RATE/1921
225.0, 675.0, 102.0, 3.75, 0.915
1, 1
1921
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 1921 in model.fail_ladtransversefibermatrixdebondingrates
    ftfmdr = model.fail_ladtransversefibermatrixdebondingrates[1921]
    assert pytest.approx(ftfmdr.sigma_tfmdr0) == 225.0
    assert pytest.approx(ftfmdr.sigma_tfmdrc) == 675.0
    assert pytest.approx(ftfmdr.gamma_tfmdr) == 102.0
    assert pytest.approx(ftfmdr.p_tfmdr) == 3.75
    assert pytest.approx(ftfmdr.d_tfmdr_max) == 0.915
    assert ftfmdr.fail_id == 1921


def test_m378_fail_lad_transverse_fiber_matrix_debonding_rate_aliases(tmp_path: Path):
    deck = """# RADIOSS ALIAS DECK
/BEGIN
Ladeveze Transverse Fiber-Matrix Debonding Rate Aliases Test
/FAIL/LADEVEZE_TRANSVERSE_FIBER_MATRIX_DEBONDING_RATE/1922
205.0, 615.0, 90.0, 3.25, 0.960
1, 1
/FAIL/LAD_TFMDR/1923
205.0, 615.0, 90.0, 3.25, 0.960
1, 1
/FAIL/LAD_TFMDR_MODEL/1924
205.0, 615.0, 90.0, 3.25, 0.960
1, 1
/FAIL/LAD_TFMDR_LAW/1925
205.0, 615.0, 90.0, 3.25, 0.960
1, 1
/FAIL/LADEVEZE_RATE_DEPENDENT_TRANSVERSE_FIBER_MATRIX_DEBONDING/1926
205.0, 615.0, 90.0, 3.25, 0.960
1, 1
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 1922 in model.fail_ladtransversefibermatrixdebondingrates
    assert 1923 in model.fail_ladtransversefibermatrixdebondingrates
    assert 1924 in model.fail_ladtransversefibermatrixdebondingrates
    assert 1925 in model.fail_ladtransversefibermatrixdebondingrates
    assert 1926 in model.fail_ladtransversefibermatrixdebondingrates
    assert len(model.raw_fails) == 5


def test_m378_fail_lad_transverse_fiber_matrix_debonding_rate_missing_card(tmp_path: Path):
    deck = """# RADIOSS MISSING CARD DECK
/BEGIN
/FAIL/LAD_TRANSVERSE_FIBER_MATRIX_DEBONDING_RATE/1927
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) > 0


def test_m378_eng_flexomagnetophononicexcitonicpolaritonic_resonance_energy_fixed(tmp_path: Path):
    c1 = f"{0.0058:>20.4f}{225:>10d}"
    deck = f"""# RADIOSS STARTER DECK
/BEGIN
Engine Flexomagnetophononicexcitonicpolaritonic Resonance Energy Fixed Format Test
2022 0
/ENG/FLEXOMAGNETOPHONONICEXCITONICPOLARITONIC_RESONANCE_ENERGY/125
Resonance Energy Output Directive
{c1}
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 125 in model.eng_flexomagnetophononicexcitonicpolaritonic_resonance_energies
    eng = model.eng_flexomagnetophononicexcitonicpolaritonic_resonance_energies[125]
    assert pytest.approx(eng.dt_fmpepr) == 0.0058
    assert eng.sens_id == 225


def test_m378_eng_flexomagnetophononicexcitonicpolaritonic_resonance_energy_free(tmp_path: Path):
    deck = """# RADIOSS FREE DECK
/BEGIN
Engine Flexomagnetophononicexcitonicpolaritonic Resonance Energy Free Format Test
/ENG/FLEXOMAGNETOPHONONICEXCITONICPOLARITONIC_RESONANCE_ENERGY/126
0.0068, 226
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 126 in model.eng_flexomagnetophononicexcitonicpolaritonic_resonance_energies
    eng = model.eng_flexomagnetophononicexcitonicpolaritonic_resonance_energies[126]
    assert pytest.approx(eng.dt_fmpepr) == 0.0068
    assert eng.sens_id == 226


def test_m378_eng_flexomagnetophononicexcitonicpolaritonic_resonance_energy_aliases(tmp_path: Path):
    deck = """# RADIOSS ALIAS DECK
/BEGIN
Engine Flexomagnetophononicexcitonicpolaritonic Resonance Energy Aliases Test
/ENG/FLEXOMAGNETO_PHONON_EXCITON_POLARITON_RES_WORK/127
0.0078, 227
/ENG/EFLEXOMAGNETOPHONONICEXCITONICPOLARITONICRESONANCE/128
0.0088, 228
/ENG/FLEXOMAGNETOPHONONICEXCITONICPOLARITONIC_RESONANCE_DISSIPATION/129
0.0098, 229
/ENG/EM_FLEXOMAGNETOPHONONICEXCITONICPOLARITONIC_RESONANCE/130
0.0108, 230
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 127 in model.eng_flexomagnetophononicexcitonicpolaritonic_resonance_energies
    assert 128 in model.eng_flexomagnetophononicexcitonicpolaritonic_resonance_energies
    assert 129 in model.eng_flexomagnetophononicexcitonicpolaritonic_resonance_energies
    assert 130 in model.eng_flexomagnetophononicexcitonicpolaritonic_resonance_energies


def test_m378_eng_flexomagnetophononicexcitonicpolaritonic_resonance_energy_missing_card(tmp_path: Path):
    deck = """# RADIOSS MISSING CARD DECK
/BEGIN
/ENG/FLEXOMAGNETOPHONONICEXCITONICPOLARITONIC_RESONANCE_ENERGY/131
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) > 0


def test_m378_lagmul_altmann_hybrid_spatial_linkage_joint_fixed(tmp_path: Path):
    c1 = f"{211:>10d}{212:>10d}{213:>10d}{9.8e6:>20.4f}{17:>10d}{4.0e-5:>20.4e}"
    c2 = f"{78.0:>20.4f}{60.0:>20.4f}{80.0:>20.4f}{26.0:>20.4f}"
    deck = f"""# RADIOSS STARTER DECK
/BEGIN
Altmann Hybrid Spatial Linkage Joint Fixed Format Test
2022 0
/LAGMUL/ALTMANN_HYBRID_SPATIAL_LINKAGE_JOINT/175
Altmann Hybrid Spatial 6R Joint
{c1}
{c2}
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 175 in model.lagmul_altmann_hybrid_spatial_linkage_joints
    joint = model.lagmul_altmann_hybrid_spatial_linkage_joints[175]
    assert joint.node1 == 211
    assert joint.node2 == 212
    assert joint.node3 == 213
    assert pytest.approx(joint.stiff) == 9.8e6
    assert joint.skew_id == 17
    assert pytest.approx(joint.tol) == 4.0e-5
    assert pytest.approx(joint.link_len_a) == 78.0
    assert pytest.approx(joint.link_len_b) == 60.0
    assert pytest.approx(joint.twist_angle_alpha) == 80.0
    assert pytest.approx(joint.offset_distance_s) == 26.0


def test_m378_lagmul_altmann_hybrid_spatial_linkage_joint_free(tmp_path: Path):
    deck = """# RADIOSS FREE DECK
/BEGIN
Altmann Hybrid Spatial Linkage Joint Free Format Test
/LAGMUL/ALTMANN_HYBRID_SPATIAL_LINKAGE_JOINT/176
311, 312, 313, 1.15e7, 18, 5.0e-5
90.0, 66.0, 100.0, 38.0
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 176 in model.lagmul_altmann_hybrid_spatial_linkage_joints
    joint = model.lagmul_altmann_hybrid_spatial_linkage_joints[176]
    assert joint.node1 == 311
    assert joint.node2 == 312
    assert joint.node3 == 313
    assert pytest.approx(joint.stiff) == 1.15e7
    assert joint.skew_id == 18
    assert pytest.approx(joint.tol) == 5.0e-5
    assert pytest.approx(joint.link_len_a) == 90.0
    assert pytest.approx(joint.link_len_b) == 66.0
    assert pytest.approx(joint.twist_angle_alpha) == 100.0
    assert pytest.approx(joint.offset_distance_s) == 38.0


def test_m378_lagmul_altmann_hybrid_spatial_linkage_joint_aliases(tmp_path: Path):
    deck = """# RADIOSS ALIAS DECK
/BEGIN
Altmann Hybrid Spatial Linkage Joint Aliases Test
/ALTMANN_HYBRID_SPATIAL_LINKAGE_JOINT/177
411, 412, 413, 1.0e6, 0, 1.0e-6
52.0, 42.0, 52.0, 16.0
/LAGMUL/ALTMANN_HYBRID_SPATIAL_LINKAGE/178
411, 412, 413, 1.0e6, 0, 1.0e-6
52.0, 42.0, 52.0, 16.0
/ALTMANN_HYBRID_SPATIAL_LINKAGE/179
411, 412, 413, 1.0e6, 0, 1.0e-6
52.0, 42.0, 52.0, 16.0
/ALTMANN_HYBRID_SPATIAL_MULTI_LOOP_MECHANISM/180
411, 412, 413, 1.0e6, 0, 1.0e-6
52.0, 42.0, 52.0, 16.0
/ALTMANN_HYBRID_SPATIAL_SYMMETRIC_MECHANISM/181
411, 412, 413, 1.0e6, 0, 1.0e-6
52.0, 42.0, 52.0, 16.0
/ALTMANN_HYBRID_SPATIAL_6R_MECHANISM/182
411, 412, 413, 1.0e6, 0, 1.0e-6
52.0, 42.0, 52.0, 16.0
/ALTMANN_HYBRID_SPATIAL_OVERCONSTRAINED_MECHANISM/183
411, 412, 413, 1.0e6, 0, 1.0e-6
52.0, 42.0, 52.0, 16.0
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    for jid in range(177, 184):
        assert jid in model.lagmul_altmann_hybrid_spatial_linkage_joints


def test_m378_lagmul_altmann_hybrid_spatial_linkage_joint_missing_card(tmp_path: Path):
    deck = """# RADIOSS MISSING CARD DECK
/BEGIN
/LAGMUL/ALTMANN_HYBRID_SPATIAL_LINKAGE_JOINT/184
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) > 0


def test_m378_sensor_spring_total_angular_pop_rate_fixed(tmp_path: Path):
    c1 = f"{1258:>10d}{1.78e8:>20.4f}{0.0545:>20.4f}"
    deck = f"""# RADIOSS STARTER DECK
/BEGIN
Sensor Spring Total Angular Pop Rate Fixed Format Test
2022 0
/SENSOR/SPRING_TOTAL_ANGULAR_POP_RATE/1
Fixed Spring Total Angular Pop Rate Sensor
{c1}
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 1 in model.sensor_spring_total_angular_pop_rates
    s1 = model.sensor_spring_total_angular_pop_rates[1]
    assert s1.spring_id == 1258
    assert pytest.approx(s1.jtang_pop_max) == 1.78e8
    assert pytest.approx(s1.jtang_snp_max) == 1.78e8
    assert pytest.approx(s1.jtang_crackle_max) == 1.78e8
    assert pytest.approx(s1.t_delay) == 0.0545
    assert len(model.sensors) == 1
    assert model.sensors[0].kind == "SPRING_TOTAL_ANGULAR_POP_RATE"


def test_m378_sensor_spring_total_angular_pop_rate_free(tmp_path: Path):
    deck = """# RADIOSS FREE DECK
/BEGIN
Sensor Spring Total Angular Pop Rate Free Format Test
/SENSOR/SPRING_TOTAL_ANGULAR_POP_RATE/2
Free Spring Total Angular Pop Rate Sensor
1259, 2.38e8, 0.0695
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 2 in model.sensor_spring_total_angular_pop_rates
    s2 = model.sensor_spring_total_angular_pop_rates[2]
    assert s2.spring_id == 1259
    assert pytest.approx(s2.jtang_pop_max) == 2.38e8
    assert pytest.approx(s2.t_delay) == 0.0695


def test_m378_sensor_spring_total_angular_pop_rate_aliases(tmp_path: Path):
    deck = """# RADIOSS ALIAS DECK
/BEGIN
Sensor Spring Total Angular Pop Rate Aliases Test
/SENSOR/SPRING_TOT_ANG_POP_RATE/280
1279, 1.025e8, 0.0125
/SENSOR/SPRING_RATE_POP_ANG_TOT/281
1280, 1.045e8, 0.0135
/SENSOR/TOTAL_ANGULAR_POP_RATE_SPRING/282
1281, 1.065e8, 0.0145
/SENSOR/SPRING_POP_ANG_TOT/283
1282, 1.085e8, 0.0155
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    for sid in range(280, 284):
        assert sid in model.sensor_spring_total_angular_pop_rates


def test_m378_sensor_spring_total_angular_pop_rate_missing_card(tmp_path: Path):
    deck = """# RADIOSS MISSING CARD DECK
/BEGIN
/SENSOR/SPRING_TOTAL_ANGULAR_POP_RATE/285
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) > 0
