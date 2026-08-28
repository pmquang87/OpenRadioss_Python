"""Tests for Milestone M401: LadDynamicFiberKinkingRate Failure Model, EngFlexothermomagnonicpolaritonicResonanceEnergy, EuclidSpatialLinkageJoint, and SensorSpringBendingSnapRate."""

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


def test_m401_fail_lad_dynamic_fiber_kinking_rate_fixed(tmp_path: Path):
    c1 = f"{590.0:>20.4f}{1770.0:>20.4f}{355.0:>20.4f}{7.55:>20.4f}{0.845:>20.4f}"
    c2 = f"{1:>10d}{2:>10d}"
    c3 = f"{2150:>10d}"
    deck = f"""# RADIOSS STARTER DECK
/BEGIN
Ladeveze Dynamic Fiber Kinking Rate Fixed Format Test
2022 0
/FAIL/LAD_DYNAMIC_FIBER_KINKING_RATE/2150
Ladeveze Dynamic Fiber Kinking Failure Model
{c1}
{c2}
{c3}
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 2150 in model.fail_laddynamicfiberkinkingrates
    fdfkr = model.fail_laddynamicfiberkinkingrates[2150]
    assert pytest.approx(fdfkr.sigma_dfkr0) == 590.0
    assert pytest.approx(fdfkr.sigma_dfkrc) == 1770.0
    assert pytest.approx(fdfkr.gamma_dfkr) == 355.0
    assert pytest.approx(fdfkr.p_dfkr) == 7.55
    assert pytest.approx(fdfkr.d_dfkr_max) == 0.845
    assert fdfkr.ifail_sh == 1
    assert fdfkr.ifail_so == 2
    assert fdfkr.fail_id == 2150
    assert len(model.raw_fails) == 1
    assert model.raw_fails[0][1].type == "LAD_DYNAMIC_FIBER_KINKING_RATE"


def test_m401_fail_lad_dynamic_fiber_kinking_rate_free(tmp_path: Path):
    deck = """# RADIOSS FREE DECK
/BEGIN
Ladeveze Dynamic Fiber Kinking Rate Free Format Test
/FAIL/LAD_DYNAMIC_FIBER_KINKING_RATE/2151
600.0, 1800.0, 365.0, 7.75, 0.835
1, 1
2151
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 2151 in model.fail_laddynamicfiberkinkingrates
    fdfkr = model.fail_laddynamicfiberkinkingrates[2151]
    assert pytest.approx(fdfkr.sigma_dfkr0) == 600.0
    assert pytest.approx(fdfkr.sigma_dfkrc) == 1800.0
    assert pytest.approx(fdfkr.gamma_dfkr) == 365.0
    assert pytest.approx(fdfkr.p_dfkr) == 7.75
    assert pytest.approx(fdfkr.d_dfkr_max) == 0.835
    assert fdfkr.fail_id == 2151


def test_m401_fail_lad_dynamic_fiber_kinking_rate_aliases(tmp_path: Path):
    deck = """# RADIOSS ALIAS DECK
/BEGIN
Ladeveze Dynamic Fiber Kinking Rate Aliases Test
/FAIL/LADEVEZE_DYNAMIC_FIBER_KINKING_RATE/2152
580.0, 1740.0, 350.0, 7.35, 0.865
1, 1
/FAIL/LAD_DFKR/2153
580.0, 1740.0, 350.0, 7.35, 0.865
1, 1
/FAIL/LAD_DFKR_MODEL/2154
580.0, 1740.0, 350.0, 7.35, 0.865
1, 1
/FAIL/LAD_DFKR_LAW/2155
580.0, 1740.0, 350.0, 7.35, 0.865
1, 1
/FAIL/LADEVEZE_RATE_DEPENDENT_DYNAMIC_FIBER_KINKING/2156
580.0, 1740.0, 350.0, 7.35, 0.865
1, 1
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 2152 in model.fail_laddynamicfiberkinkingrates
    assert 2153 in model.fail_laddynamicfiberkinkingrates
    assert 2154 in model.fail_laddynamicfiberkinkingrates
    assert 2155 in model.fail_laddynamicfiberkinkingrates
    assert 2156 in model.fail_laddynamicfiberkinkingrates
    assert len(model.raw_fails) == 5


def test_m401_fail_lad_dynamic_fiber_kinking_rate_missing_card(tmp_path: Path):
    deck = """# RADIOSS MISSING CARD DECK
/BEGIN
/FAIL/LAD_DYNAMIC_FIBER_KINKING_RATE/2157
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) > 0


def test_m401_eng_flexothermomagnonicpolaritonic_resonance_energy_fixed(tmp_path: Path):
    c1 = f"{0.0235:>20.4f}{455:>10d}"
    deck = f"""# RADIOSS STARTER DECK
/BEGIN
Engine Flexothermomagnonicpolaritonic Resonance Energy Fixed Format Test
2022 0
/ENG/FLEXOTHERMOMAGNONICPOLARITONIC_RESONANCE_ENERGY/355
Resonance Energy Output Directive
{c1}
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 355 in model.eng_flexothermomagnonicpolaritonic_resonance_energies
    eng = model.eng_flexothermomagnonicpolaritonic_resonance_energies[355]
    assert pytest.approx(eng.dt_ftmp) == 0.0235
    assert eng.sens_id == 455


def test_m401_eng_flexothermomagnonicpolaritonic_resonance_energy_free(tmp_path: Path):
    deck = """# RADIOSS FREE DECK
/BEGIN
Engine Flexothermomagnonicpolaritonic Resonance Energy Free Format Test
/ENG/FLEXOTHERMOMAGNONICPOLARITONIC_RESONANCE_ENERGY/356
0.0245, 456
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 356 in model.eng_flexothermomagnonicpolaritonic_resonance_energies
    eng = model.eng_flexothermomagnonicpolaritonic_resonance_energies[356]
    assert pytest.approx(eng.dt_ftmp) == 0.0245
    assert eng.sens_id == 456


def test_m401_eng_flexothermomagnonicpolaritonic_resonance_energy_aliases(tmp_path: Path):
    deck = """# RADIOSS ALIAS DECK
/BEGIN
Engine Flexothermomagnonicpolaritonic Resonance Energy Aliases Test
/ENG/FLEXOTHERM_MAGNONIC_POLARITON_RES_WORK/357
0.0255, 457
/ENG/EFLEXOTHERMOMAGNONICPOLARITONICRESONANCE/358
0.0265, 458
/ENG/FLEXOTHERMOMAGNONICPOLARITONIC_RESONANCE_DISSIPATION/359
0.0275, 459
/ENG/ET_FLEXOTHERMOMAGNONICPOLARITONIC_RESONANCE/360
0.0285, 460
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 357 in model.eng_flexothermomagnonicpolaritonic_resonance_energies
    assert 358 in model.eng_flexothermomagnonicpolaritonic_resonance_energies
    assert 359 in model.eng_flexothermomagnonicpolaritonic_resonance_energies
    assert 360 in model.eng_flexothermomagnonicpolaritonic_resonance_energies


def test_m401_eng_flexothermomagnonicpolaritonic_resonance_energy_missing_card(tmp_path: Path):
    deck = """# RADIOSS MISSING CARD DECK
/BEGIN
/ENG/FLEXOTHERMOMAGNONICPOLARITONIC_RESONANCE_ENERGY/361
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) > 0


def test_m401_lagmul_euclid_spatial_linkage_joint_fixed(tmp_path: Path):
    c1 = f"{441:>10d}{442:>10d}{443:>10d}{2.96e7:>20.4f}{159:>10d}{1.95e-4:>20.4e}"
    c2 = f"{225.0:>20.4f}{208.0:>20.4f}{235.0:>20.4f}{158.0:>20.4f}"
    deck = f"""# RADIOSS STARTER DECK
/BEGIN
Euclid Spatial Linkage Joint Fixed Format Test
2022 0
/LAGMUL/EUCLID_SPATIAL_LINKAGE_JOINT/405
Euclid Spatial 6R Joint
{c1}
{c2}
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 405 in model.lagmul_euclid_spatial_linkage_joints
    joint = model.lagmul_euclid_spatial_linkage_joints[405]
    assert joint.node1 == 441
    assert joint.node2 == 442
    assert joint.node3 == 443
    assert pytest.approx(joint.stiff) == 2.96e7
    assert joint.skew_id == 159
    assert pytest.approx(joint.tol) == 1.95e-4
    assert pytest.approx(joint.link_len_a) == 225.0
    assert pytest.approx(joint.link_len_b) == 208.0
    assert pytest.approx(joint.twist_angle_alpha) == 235.0
    assert pytest.approx(joint.offset_distance_s) == 158.0
    assert pytest.approx(joint.offset_distance_r) == 158.0
    assert pytest.approx(joint.offset_distance_v) == 158.0
    assert pytest.approx(joint.offset_distance_h) == 158.0
    assert pytest.approx(joint.offset_distance_u) == 158.0


def test_m401_lagmul_euclid_spatial_linkage_joint_free(tmp_path: Path):
    deck = """# RADIOSS FREE DECK
/BEGIN
Euclid Spatial Linkage Joint Free Format Test
/LAGMUL/EUCLID_SPATIAL_LINKAGE_JOINT/406
541, 542, 543, 3.20e7, 160, 2.05e-4
245.0, 215.0, 255.0, 170.0
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 406 in model.lagmul_euclid_spatial_linkage_joints
    joint = model.lagmul_euclid_spatial_linkage_joints[406]
    assert joint.node1 == 541
    assert joint.node2 == 542
    assert joint.node3 == 543
    assert pytest.approx(joint.stiff) == 3.20e7
    assert joint.skew_id == 160
    assert pytest.approx(joint.tol) == 2.05e-4
    assert pytest.approx(joint.link_len_a) == 245.0
    assert pytest.approx(joint.link_len_b) == 215.0
    assert pytest.approx(joint.twist_angle_alpha) == 255.0
    assert pytest.approx(joint.offset_distance_s) == 170.0


def test_m401_lagmul_euclid_spatial_linkage_joint_aliases(tmp_path: Path):
    deck = """# RADIOSS ALIAS DECK
/BEGIN
Euclid Spatial Linkage Joint Aliases Test
/EUCLID_SPATIAL_LINKAGE_JOINT/407
641, 642, 643, 1.0e6, 0, 1.0e-6
185.0, 175.0, 185.0, 135.0
/LAGMUL/EUCLID_SPATIAL_LINKAGE/408
641, 642, 643, 1.0e6, 0, 1.0e-6
185.0, 175.0, 185.0, 135.0
/EUCLID_SPATIAL_LINKAGE/409
641, 642, 643, 1.0e6, 0, 1.0e-6
185.0, 175.0, 185.0, 135.0
/EUCLID_SPATIAL_MULTI_LOOP_MECHANISM/410
641, 642, 643, 1.0e6, 0, 1.0e-6
185.0, 175.0, 185.0, 135.0
/EUCLID_SPATIAL_SYMMETRIC_MECHANISM/411
641, 642, 643, 1.0e6, 0, 1.0e-6
185.0, 175.0, 185.0, 135.0
/EUCLID_SPATIAL_6R_MECHANISM/412
641, 642, 643, 1.0e6, 0, 1.0e-6
185.0, 175.0, 185.0, 135.0
/EUCLID_SPATIAL_OVERCONSTRAINED_MECHANISM/413
641, 642, 643, 1.0e6, 0, 1.0e-6
185.0, 175.0, 185.0, 135.0
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    for jid in range(407, 414):
        assert jid in model.lagmul_euclid_spatial_linkage_joints


def test_m401_lagmul_euclid_spatial_linkage_joint_missing_card(tmp_path: Path):
    deck = """# RADIOSS MISSING CARD DECK
/BEGIN
/LAGMUL/EUCLID_SPATIAL_LINKAGE_JOINT/414
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) > 0


def test_m401_sensor_spring_bending_snap_rate_fixed(tmp_path: Path):
    c1 = f"{1488:>10d}{4.18e8:>20.4f}{0.1345:>20.4f}"
    deck = f"""# RADIOSS STARTER DECK
/BEGIN
Sensor Spring Bending Snap Rate Fixed Format Test
2022 0
/SENSOR/SPRING_BENDING_SNAP_RATE/1
Fixed Spring Bending Snap Rate Sensor
{c1}
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 1 in model.sensor_spring_bending_snap_rates
    s1 = model.sensor_spring_bending_snap_rates[1]
    assert s1.spring_id == 1488
    assert pytest.approx(s1.jbend_snp_max) == 4.18e8
    assert pytest.approx(s1.jbend_shot_max) == 4.18e8
    assert pytest.approx(s1.jbend_drop_max) == 4.18e8
    assert pytest.approx(s1.jbend_lock_max) == 4.18e8
    assert pytest.approx(s1.jbend_pop_max) == 4.18e8
    assert pytest.approx(s1.jbend_crackle_max) == 4.18e8
    assert pytest.approx(s1.t_delay) == 0.1345
    assert len(model.sensors) == 1
    assert model.sensors[0].kind == "SPRING_BENDING_SNAP_RATE"


def test_m401_sensor_spring_bending_snap_rate_free(tmp_path: Path):
    deck = """# RADIOSS FREE DECK
/BEGIN
Sensor Spring Bending Snap Rate Free Format Test
/SENSOR/SPRING_BENDING_SNAP_RATE/2
Free Spring Bending Snap Rate Sensor
1489, 4.78e8, 0.2095
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 2 in model.sensor_spring_bending_snap_rates
    s2 = model.sensor_spring_bending_snap_rates[2]
    assert s2.spring_id == 1489
    assert pytest.approx(s2.jbend_snp_max) == 4.78e8
    assert pytest.approx(s2.t_delay) == 0.2095


def test_m401_sensor_spring_bending_snap_rate_aliases(tmp_path: Path):
    deck = """# RADIOSS ALIAS DECK
/BEGIN
Sensor Spring Bending Snap Rate Aliases Test
/SENSOR/SPRING_BEND_SNAP_RATE/391
1509, 2.395e8, 0.1265
/SENSOR/SPRING_RATE_SNAP_BEND/392
1510, 2.415e8, 0.1275
/SENSOR/BENDING_SNAP_RATE_SPRING/393
1511, 2.435e8, 0.1285
/SENSOR/SPRING_SNAP_BEND/394
1512, 2.455e8, 0.1295
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    for sid in range(391, 395):
        assert sid in model.sensor_spring_bending_snap_rates


def test_m401_sensor_spring_bending_snap_rate_missing_card(tmp_path: Path):
    deck = """# RADIOSS MISSING CARD DECK
/BEGIN
/SENSOR/SPRING_BENDING_SNAP_RATE/395
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) > 0
