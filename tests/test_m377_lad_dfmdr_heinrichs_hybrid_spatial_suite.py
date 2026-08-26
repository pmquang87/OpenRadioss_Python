"""Tests for Milestone M377: LadDynamicFiberMatrixDebondingRate Failure Model, EngFlexomagnetophononicexcitonicmagnonicResonanceEnergy, HeinrichsHybridSpatialLinkageJoint, and SensorSpringBendingPopRate."""

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


def test_m377_fail_lad_dynamic_fiber_matrix_debonding_rate_fixed(tmp_path: Path):
    c1 = f"{205.0:>20.4f}{615.0:>20.4f}{92.0:>20.4f}{3.45:>20.4f}{0.945:>20.4f}"
    c2 = f"{1:>10d}{2:>10d}"
    c3 = f"{1910:>10d}"
    deck = f"""# RADIOSS STARTER DECK
/BEGIN
Ladeveze Dynamic Fiber-Matrix Debonding Rate Fixed Format Test
2022 0
/FAIL/LAD_DYNAMIC_FIBER_MATRIX_DEBONDING_RATE/1910
Ladeveze Dynamic Fiber-Matrix Debonding Failure Model
{c1}
{c2}
{c3}
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 1910 in model.fail_laddynamicfibermatrixdebondingrates
    fdfmdr = model.fail_laddynamicfibermatrixdebondingrates[1910]
    assert pytest.approx(fdfmdr.sigma_dfmdr0) == 205.0
    assert pytest.approx(fdfmdr.sigma_dfmdrc) == 615.0
    assert pytest.approx(fdfmdr.gamma_dfmdr) == 92.0
    assert pytest.approx(fdfmdr.p_dfmdr) == 3.45
    assert pytest.approx(fdfmdr.d_dfmdr_max) == 0.945
    assert fdfmdr.ifail_sh == 1
    assert fdfmdr.ifail_so == 2
    assert fdfmdr.fail_id == 1910
    assert len(model.raw_fails) == 1
    assert model.raw_fails[0][1].type == "LAD_DYNAMIC_FIBER_MATRIX_DEBONDING_RATE"


def test_m377_fail_lad_dynamic_fiber_matrix_debonding_rate_free(tmp_path: Path):
    deck = """# RADIOSS FREE DECK
/BEGIN
Ladeveze Dynamic Fiber-Matrix Debonding Rate Free Format Test
/FAIL/LAD_DYNAMIC_FIBER_MATRIX_DEBONDING_RATE/1911
215.0, 645.0, 98.0, 3.65, 0.920
1, 1
1911
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 1911 in model.fail_laddynamicfibermatrixdebondingrates
    fdfmdr = model.fail_laddynamicfibermatrixdebondingrates[1911]
    assert pytest.approx(fdfmdr.sigma_dfmdr0) == 215.0
    assert pytest.approx(fdfmdr.sigma_dfmdrc) == 645.0
    assert pytest.approx(fdfmdr.gamma_dfmdr) == 98.0
    assert pytest.approx(fdfmdr.p_dfmdr) == 3.65
    assert pytest.approx(fdfmdr.d_dfmdr_max) == 0.920
    assert fdfmdr.fail_id == 1911


def test_m377_fail_lad_dynamic_fiber_matrix_debonding_rate_aliases(tmp_path: Path):
    deck = """# RADIOSS ALIAS DECK
/BEGIN
Ladeveze Dynamic Fiber-Matrix Debonding Rate Aliases Test
/FAIL/LADEVEZE_DYNAMIC_FIBER_MATRIX_DEBONDING_RATE/1912
195.0, 585.0, 86.0, 3.15, 0.965
1, 1
/FAIL/LAD_DFMDR/1913
195.0, 585.0, 86.0, 3.15, 0.965
1, 1
/FAIL/LAD_DFMDR_MODEL/1914
195.0, 585.0, 86.0, 3.15, 0.965
1, 1
/FAIL/LAD_DFMDR_LAW/1915
195.0, 585.0, 86.0, 3.15, 0.965
1, 1
/FAIL/LADEVEZE_RATE_DEPENDENT_DYNAMIC_FIBER_MATRIX_DEBONDING/1916
195.0, 585.0, 86.0, 3.15, 0.965
1, 1
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 1912 in model.fail_laddynamicfibermatrixdebondingrates
    assert 1913 in model.fail_laddynamicfibermatrixdebondingrates
    assert 1914 in model.fail_laddynamicfibermatrixdebondingrates
    assert 1915 in model.fail_laddynamicfibermatrixdebondingrates
    assert 1916 in model.fail_laddynamicfibermatrixdebondingrates
    assert len(model.raw_fails) == 5


def test_m377_fail_lad_dynamic_fiber_matrix_debonding_rate_missing_card(tmp_path: Path):
    deck = """# RADIOSS MISSING CARD DECK
/BEGIN
/FAIL/LAD_DYNAMIC_FIBER_MATRIX_DEBONDING_RATE/1917
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) > 0


def test_m377_eng_flexomagnetophononicexcitonicmagnonic_resonance_energy_fixed(tmp_path: Path):
    c1 = f"{0.0056:>20.4f}{215:>10d}"
    deck = f"""# RADIOSS STARTER DECK
/BEGIN
Engine Flexomagnetophononicexcitonicmagnonic Resonance Energy Fixed Format Test
2022 0
/ENG/FLEXOMAGNETOPHONONICEXCITONICMAGNONIC_RESONANCE_ENERGY/115
Resonance Energy Output Directive
{c1}
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 115 in model.eng_flexomagnetophononicexcitonicmagnonic_resonance_energies
    eng = model.eng_flexomagnetophononicexcitonicmagnonic_resonance_energies[115]
    assert pytest.approx(eng.dt_fmpemr) == 0.0056
    assert eng.sens_id == 215


def test_m377_eng_flexomagnetophononicexcitonicmagnonic_resonance_energy_free(tmp_path: Path):
    deck = """# RADIOSS FREE DECK
/BEGIN
Engine Flexomagnetophononicexcitonicmagnonic Resonance Energy Free Format Test
/ENG/FLEXOMAGNETOPHONONICEXCITONICMAGNONIC_RESONANCE_ENERGY/116
0.0066, 216
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 116 in model.eng_flexomagnetophononicexcitonicmagnonic_resonance_energies
    eng = model.eng_flexomagnetophononicexcitonicmagnonic_resonance_energies[116]
    assert pytest.approx(eng.dt_fmpemr) == 0.0066
    assert eng.sens_id == 216


def test_m377_eng_flexomagnetophononicexcitonicmagnonic_resonance_energy_aliases(tmp_path: Path):
    deck = """# RADIOSS ALIAS DECK
/BEGIN
Engine Flexomagnetophononicexcitonicmagnonic Resonance Energy Aliases Test
/ENG/FLEXOMAGNETO_PHONON_EXCITON_MAGNON_RES_WORK/117
0.0076, 217
/ENG/EFLEXOMAGNETOPHONONICEXCITONICMAGNONICRESONANCE/118
0.0086, 218
/ENG/FLEXOMAGNETOPHONONICEXCITONICMAGNONIC_RESONANCE_DISSIPATION/119
0.0096, 219
/ENG/EM_FLEXOMAGNETOPHONONICEXCITONICMAGNONIC_RESONANCE/120
0.0106, 220
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 117 in model.eng_flexomagnetophononicexcitonicmagnonic_resonance_energies
    assert 118 in model.eng_flexomagnetophononicexcitonicmagnonic_resonance_energies
    assert 119 in model.eng_flexomagnetophononicexcitonicmagnonic_resonance_energies
    assert 120 in model.eng_flexomagnetophononicexcitonicmagnonic_resonance_energies


def test_m377_eng_flexomagnetophononicexcitonicmagnonic_resonance_energy_missing_card(tmp_path: Path):
    deck = """# RADIOSS MISSING CARD DECK
/BEGIN
/ENG/FLEXOMAGNETOPHONONICEXCITONICMAGNONIC_RESONANCE_ENERGY/121
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) > 0


def test_m377_lagmul_heinrichs_hybrid_spatial_linkage_joint_fixed(tmp_path: Path):
    c1 = f"{201:>10d}{202:>10d}{203:>10d}{9.5e6:>20.4f}{15:>10d}{3.8e-5:>20.4e}"
    c2 = f"{76.0:>20.4f}{58.0:>20.4f}{78.0:>20.4f}{25.0:>20.4f}"
    deck = f"""# RADIOSS STARTER DECK
/BEGIN
Heinrichs Hybrid Spatial Linkage Joint Fixed Format Test
2022 0
/LAGMUL/HEINRICHS_HYBRID_SPATIAL_LINKAGE_JOINT/165
Heinrichs Hybrid Spatial 6R Joint
{c1}
{c2}
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 165 in model.lagmul_heinrichs_hybrid_spatial_linkage_joints
    joint = model.lagmul_heinrichs_hybrid_spatial_linkage_joints[165]
    assert joint.node1 == 201
    assert joint.node2 == 202
    assert joint.node3 == 203
    assert pytest.approx(joint.stiff) == 9.5e6
    assert joint.skew_id == 15
    assert pytest.approx(joint.tol) == 3.8e-5
    assert pytest.approx(joint.link_len_a) == 76.0
    assert pytest.approx(joint.link_len_b) == 58.0
    assert pytest.approx(joint.twist_angle_alpha) == 78.0
    assert pytest.approx(joint.offset_distance_s) == 25.0


def test_m377_lagmul_heinrichs_hybrid_spatial_linkage_joint_free(tmp_path: Path):
    deck = """# RADIOSS FREE DECK
/BEGIN
Heinrichs Hybrid Spatial Linkage Joint Free Format Test
/LAGMUL/HEINRICHS_HYBRID_SPATIAL_LINKAGE_JOINT/166
301, 302, 303, 1.10e7, 16, 4.8e-5
88.0, 64.0, 98.0, 36.0
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 166 in model.lagmul_heinrichs_hybrid_spatial_linkage_joints
    joint = model.lagmul_heinrichs_hybrid_spatial_linkage_joints[166]
    assert joint.node1 == 301
    assert joint.node2 == 302
    assert joint.node3 == 303
    assert pytest.approx(joint.stiff) == 1.10e7
    assert joint.skew_id == 16
    assert pytest.approx(joint.tol) == 4.8e-5
    assert pytest.approx(joint.link_len_a) == 88.0
    assert pytest.approx(joint.link_len_b) == 64.0
    assert pytest.approx(joint.twist_angle_alpha) == 98.0
    assert pytest.approx(joint.offset_distance_s) == 36.0


def test_m377_lagmul_heinrichs_hybrid_spatial_linkage_joint_aliases(tmp_path: Path):
    deck = """# RADIOSS ALIAS DECK
/BEGIN
Heinrichs Hybrid Spatial Linkage Joint Aliases Test
/HEINRICHS_HYBRID_SPATIAL_LINKAGE_JOINT/167
401, 402, 403, 1.0e6, 0, 1.0e-6
50.0, 40.0, 50.0, 15.0
/LAGMUL/HEINRICHS_HYBRID_SPATIAL_LINKAGE/168
401, 402, 403, 1.0e6, 0, 1.0e-6
50.0, 40.0, 50.0, 15.0
/HEINRICHS_HYBRID_SPATIAL_LINKAGE/169
401, 402, 403, 1.0e6, 0, 1.0e-6
50.0, 40.0, 50.0, 15.0
/HEINRICHS_HYBRID_SPATIAL_MULTI_LOOP_MECHANISM/170
401, 402, 403, 1.0e6, 0, 1.0e-6
50.0, 40.0, 50.0, 15.0
/HEINRICHS_HYBRID_SPATIAL_SYMMETRIC_MECHANISM/171
401, 402, 403, 1.0e6, 0, 1.0e-6
50.0, 40.0, 50.0, 15.0
/HEINRICHS_HYBRID_SPATIAL_6R_MECHANISM/172
401, 402, 403, 1.0e6, 0, 1.0e-6
50.0, 40.0, 50.0, 15.0
/HEINRICHS_HYBRID_SPATIAL_OVERCONSTRAINED_MECHANISM/173
401, 402, 403, 1.0e6, 0, 1.0e-6
50.0, 40.0, 50.0, 15.0
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    for jid in range(167, 174):
        assert jid in model.lagmul_heinrichs_hybrid_spatial_linkage_joints


def test_m377_lagmul_heinrichs_hybrid_spatial_linkage_joint_missing_card(tmp_path: Path):
    deck = """# RADIOSS MISSING CARD DECK
/BEGIN
/LAGMUL/HEINRICHS_HYBRID_SPATIAL_LINKAGE_JOINT/174
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) > 0


def test_m377_sensor_spring_bending_pop_rate_fixed(tmp_path: Path):
    c1 = f"{1248:>10d}{1.68e8:>20.4f}{0.0515:>20.4f}"
    deck = f"""# RADIOSS STARTER DECK
/BEGIN
Sensor Spring Bending Pop Rate Fixed Format Test
2022 0
/SENSOR/SPRING_BENDING_POP_RATE/1
Fixed Spring Bending Pop Rate Sensor
{c1}
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 1 in model.sensor_spring_bending_pop_rates
    s1 = model.sensor_spring_bending_pop_rates[1]
    assert s1.spring_id == 1248
    assert pytest.approx(s1.jbend_pop_max) == 1.68e8
    assert pytest.approx(s1.jbend_snp_max) == 1.68e8
    assert pytest.approx(s1.jbend_crackle_max) == 1.68e8
    assert pytest.approx(s1.t_delay) == 0.0515
    assert len(model.sensors) == 1
    assert model.sensors[0].kind == "SPRING_BENDING_POP_RATE"


def test_m377_sensor_spring_bending_pop_rate_free(tmp_path: Path):
    deck = """# RADIOSS FREE DECK
/BEGIN
Sensor Spring Bending Pop Rate Free Format Test
/SENSOR/SPRING_BENDING_POP_RATE/2
Free Spring Bending Pop Rate Sensor
1249, 2.28e8, 0.0665
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 2 in model.sensor_spring_bending_pop_rates
    s2 = model.sensor_spring_bending_pop_rates[2]
    assert s2.spring_id == 1249
    assert pytest.approx(s2.jbend_pop_max) == 2.28e8
    assert pytest.approx(s2.t_delay) == 0.0665


def test_m377_sensor_spring_bending_pop_rate_aliases(tmp_path: Path):
    deck = """# RADIOSS ALIAS DECK
/BEGIN
Sensor Spring Bending Pop Rate Aliases Test
/SENSOR/SPRING_BEND_POP_RATE/270
1269, 9.95e7, 0.0120
/SENSOR/SPRING_RATE_POP_BEND/271
1270, 1.015e8, 0.0130
/SENSOR/BENDING_POP_RATE_SPRING/272
1271, 1.035e8, 0.0140
/SENSOR/SPRING_POP_BEND/273
1272, 1.055e8, 0.0150
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    for sid in range(270, 274):
        assert sid in model.sensor_spring_bending_pop_rates


def test_m377_sensor_spring_bending_pop_rate_missing_card(tmp_path: Path):
    deck = """# RADIOSS MISSING CARD DECK
/BEGIN
/SENSOR/SPRING_BENDING_POP_RATE/275
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) > 0
