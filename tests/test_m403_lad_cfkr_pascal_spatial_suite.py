"""Tests for Milestone M403: LadCoupleFiberKinkingRate Failure Model, EngFlexothermoexcitonicphononicpolaritonicResonanceEnergy, PascalSpatialLinkageJoint, and SensorSpringNormalCrackleRate."""

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


def test_m403_fail_lad_couple_fiber_kinking_rate_fixed(tmp_path: Path):
    c1 = f"{610.0:>20.4f}{1830.0:>20.4f}{375.0:>20.4f}{7.75:>20.4f}{0.825:>20.4f}"
    c2 = f"{1:>10d}{2:>10d}"
    c3 = f"{2170:>10d}"
    deck = f"""# RADIOSS STARTER DECK
/BEGIN
Ladeveze Coupled Fiber Kinking Rate Fixed Format Test
2022 0
/FAIL/LAD_COUPLE_FIBER_KINKING_RATE/2170
Ladeveze Coupled Fiber Kinking Failure Model
{c1}
{c2}
{c3}
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 2170 in model.fail_ladcouplefiberkinkingrates
    fcfkr = model.fail_ladcouplefiberkinkingrates[2170]
    assert pytest.approx(fcfkr.sigma_cfkr0) == 610.0
    assert pytest.approx(fcfkr.sigma_cfkrc) == 1830.0
    assert pytest.approx(fcfkr.gamma_cfkr) == 375.0
    assert pytest.approx(fcfkr.p_cfkr) == 7.75
    assert pytest.approx(fcfkr.d_cfkr_max) == 0.825
    assert fcfkr.ifail_sh == 1
    assert fcfkr.ifail_so == 2
    assert fcfkr.fail_id == 2170
    assert len(model.raw_fails) == 1
    assert model.raw_fails[0][1].type == "LAD_COUPLE_FIBER_KINKING_RATE"


def test_m403_fail_lad_couple_fiber_kinking_rate_free(tmp_path: Path):
    deck = """# RADIOSS FREE DECK
/BEGIN
Ladeveze Coupled Fiber Kinking Rate Free Format Test
/FAIL/LAD_COUPLE_FIBER_KINKING_RATE/2171
620.0, 1860.0, 385.0, 7.95, 0.815
1, 1
2171
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 2171 in model.fail_ladcouplefiberkinkingrates
    fcfkr = model.fail_ladcouplefiberkinkingrates[2171]
    assert pytest.approx(fcfkr.sigma_cfkr0) == 620.0
    assert pytest.approx(fcfkr.sigma_cfkrc) == 1860.0
    assert pytest.approx(fcfkr.gamma_cfkr) == 385.0
    assert pytest.approx(fcfkr.p_cfkr) == 7.95
    assert pytest.approx(fcfkr.d_cfkr_max) == 0.815
    assert fcfkr.fail_id == 2171


def test_m403_fail_lad_couple_fiber_kinking_rate_aliases(tmp_path: Path):
    deck = """# RADIOSS ALIAS DECK
/BEGIN
Ladeveze Coupled Fiber Kinking Rate Aliases Test
/FAIL/LADEVEZE_COUPLED_FIBER_KINKING_RATE/2172
600.0, 1800.0, 370.0, 7.55, 0.845
1, 1
/FAIL/LAD_CFKR/2173
600.0, 1800.0, 370.0, 7.55, 0.845
1, 1
/FAIL/LAD_CFKR_MODEL/2174
600.0, 1800.0, 370.0, 7.55, 0.845
1, 1
/FAIL/LAD_CFKR_LAW/2175
600.0, 1800.0, 370.0, 7.55, 0.845
1, 1
/FAIL/LADEVEZE_RATE_DEPENDENT_COUPLED_FIBER_KINKING/2176
600.0, 1800.0, 370.0, 7.55, 0.845
1, 1
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 2172 in model.fail_ladcouplefiberkinkingrates
    assert 2173 in model.fail_ladcouplefiberkinkingrates
    assert 2174 in model.fail_ladcouplefiberkinkingrates
    assert 2175 in model.fail_ladcouplefiberkinkingrates
    assert 2176 in model.fail_ladcouplefiberkinkingrates
    assert len(model.raw_fails) == 5


def test_m403_fail_lad_couple_fiber_kinking_rate_missing_card(tmp_path: Path):
    deck = """# RADIOSS MISSING CARD DECK
/BEGIN
/FAIL/LAD_COUPLE_FIBER_KINKING_RATE/2177
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) > 0


def test_m403_eng_flexothermoexcitonicphononicpolaritonic_resonance_energy_fixed(tmp_path: Path):
    c1 = f"{0.0255:>20.4f}{475:>10d}"
    deck = f"""# RADIOSS STARTER DECK
/BEGIN
Engine Flexothermoexcitonicphononicpolaritonic Resonance Energy Fixed Format Test
2022 0
/ENG/FLEXOTHERMOEXCITONICPHONONICPOLARITONIC_RESONANCE_ENERGY/375
Resonance Energy Output Directive
{c1}
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 375 in model.eng_flexothermoexcitonicphononicpolaritonic_resonance_energies
    eng = model.eng_flexothermoexcitonicphononicpolaritonic_resonance_energies[375]
    assert pytest.approx(eng.dt_ftepp) == 0.0255
    assert eng.sens_id == 475


def test_m403_eng_flexothermoexcitonicphononicpolaritonic_resonance_energy_free(tmp_path: Path):
    deck = """# RADIOSS FREE DECK
/BEGIN
Engine Flexothermoexcitonicphononicpolaritonic Resonance Energy Free Format Test
/ENG/FLEXOTHERMOEXCITONICPHONONICPOLARITONIC_RESONANCE_ENERGY/376
0.0265, 476
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 376 in model.eng_flexothermoexcitonicphononicpolaritonic_resonance_energies
    eng = model.eng_flexothermoexcitonicphononicpolaritonic_resonance_energies[376]
    assert pytest.approx(eng.dt_ftepp) == 0.0265
    assert eng.sens_id == 476


def test_m403_eng_flexothermoexcitonicphononicpolaritonic_resonance_energy_aliases(tmp_path: Path):
    deck = """# RADIOSS ALIAS DECK
/BEGIN
Engine Flexothermoexcitonicphononicpolaritonic Resonance Energy Aliases Test
/ENG/FLEXOTHERM_EXCITON_PHONON_POLARITON_RES_WORK/377
0.0275, 477
/ENG/EFLEXOTHERMOEXCITONICPHONONICPOLARITONICRESONANCE/378
0.0285, 478
/ENG/FLEXOTHERMOEXCITONICPHONONICPOLARITONIC_RESONANCE_DISSIPATION/379
0.0295, 479
/ENG/ET_FLEXOTHERMOEXCITONICPHONONICPOLARITONIC_RESONANCE/380
0.0305, 480
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 377 in model.eng_flexothermoexcitonicphononicpolaritonic_resonance_energies
    assert 378 in model.eng_flexothermoexcitonicphononicpolaritonic_resonance_energies
    assert 379 in model.eng_flexothermoexcitonicphononicpolaritonic_resonance_energies
    assert 380 in model.eng_flexothermoexcitonicphononicpolaritonic_resonance_energies


def test_m403_eng_flexothermoexcitonicphononicpolaritonic_resonance_energy_missing_card(tmp_path: Path):
    deck = """# RADIOSS MISSING CARD DECK
/BEGIN
/ENG/FLEXOTHERMOEXCITONICPHONONICPOLARITONIC_RESONANCE_ENERGY/381
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) > 0


def test_m403_lagmul_pascal_spatial_linkage_joint_fixed(tmp_path: Path):
    c1 = f"{461:>10d}{462:>10d}{463:>10d}{3.16e7:>20.4f}{179:>10d}{2.15e-4:>20.4e}"
    c2 = f"{245.0:>20.4f}{228.0:>20.4f}{255.0:>20.4f}{178.0:>20.4f}"
    deck = f"""# RADIOSS STARTER DECK
/BEGIN
Pascal Spatial Linkage Joint Fixed Format Test
2022 0
/LAGMUL/PASCAL_SPATIAL_LINKAGE_JOINT/425
Pascal Spatial 6R Joint
{c1}
{c2}
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 425 in model.lagmul_pascal_spatial_linkage_joints
    joint = model.lagmul_pascal_spatial_linkage_joints[425]
    assert joint.node1 == 461
    assert joint.node2 == 462
    assert joint.node3 == 463
    assert pytest.approx(joint.stiff) == 3.16e7
    assert joint.skew_id == 179
    assert pytest.approx(joint.tol) == 2.15e-4
    assert pytest.approx(joint.link_len_a) == 245.0
    assert pytest.approx(joint.link_len_b) == 228.0
    assert pytest.approx(joint.twist_angle_alpha) == 255.0
    assert pytest.approx(joint.offset_distance_s) == 178.0
    assert pytest.approx(joint.offset_distance_r) == 178.0
    assert pytest.approx(joint.offset_distance_v) == 178.0
    assert pytest.approx(joint.offset_distance_h) == 178.0
    assert pytest.approx(joint.offset_distance_u) == 178.0


def test_m403_lagmul_pascal_spatial_linkage_joint_free(tmp_path: Path):
    deck = """# RADIOSS FREE DECK
/BEGIN
Pascal Spatial Linkage Joint Free Format Test
/LAGMUL/PASCAL_SPATIAL_LINKAGE_JOINT/426
561, 562, 563, 3.40e7, 180, 2.25e-4
265.0, 235.0, 275.0, 190.0
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 426 in model.lagmul_pascal_spatial_linkage_joints
    joint = model.lagmul_pascal_spatial_linkage_joints[426]
    assert joint.node1 == 561
    assert joint.node2 == 562
    assert joint.node3 == 563
    assert pytest.approx(joint.stiff) == 3.40e7
    assert joint.skew_id == 180
    assert pytest.approx(joint.tol) == 2.25e-4
    assert pytest.approx(joint.link_len_a) == 265.0
    assert pytest.approx(joint.link_len_b) == 235.0
    assert pytest.approx(joint.twist_angle_alpha) == 275.0
    assert pytest.approx(joint.offset_distance_s) == 190.0


def test_m403_lagmul_pascal_spatial_linkage_joint_aliases(tmp_path: Path):
    deck = """# RADIOSS ALIAS DECK
/BEGIN
Pascal Spatial Linkage Joint Aliases Test
/PASCAL_SPATIAL_LINKAGE_JOINT/427
661, 662, 663, 1.0e6, 0, 1.0e-6
205.0, 195.0, 205.0, 155.0
/LAGMUL/PASCAL_SPATIAL_LINKAGE/428
661, 662, 663, 1.0e6, 0, 1.0e-6
205.0, 195.0, 205.0, 155.0
/PASCAL_SPATIAL_LINKAGE/429
661, 662, 663, 1.0e6, 0, 1.0e-6
205.0, 195.0, 205.0, 155.0
/PASCAL_SPATIAL_MULTI_LOOP_MECHANISM/430
661, 662, 663, 1.0e6, 0, 1.0e-6
205.0, 195.0, 205.0, 155.0
/PASCAL_SPATIAL_SYMMETRIC_MECHANISM/431
661, 662, 663, 1.0e6, 0, 1.0e-6
205.0, 195.0, 205.0, 155.0
/PASCAL_SPATIAL_6R_MECHANISM/432
661, 662, 663, 1.0e6, 0, 1.0e-6
205.0, 195.0, 205.0, 155.0
/PASCAL_SPATIAL_OVERCONSTRAINED_MECHANISM/433
661, 662, 663, 1.0e6, 0, 1.0e-6
205.0, 195.0, 205.0, 155.0
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    for jid in range(427, 434):
        assert jid in model.lagmul_pascal_spatial_linkage_joints


def test_m403_lagmul_pascal_spatial_linkage_joint_missing_card(tmp_path: Path):
    deck = """# RADIOSS MISSING CARD DECK
/BEGIN
/LAGMUL/PASCAL_SPATIAL_LINKAGE_JOINT/434
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) > 0


def test_m403_sensor_spring_normal_crackle_rate_fixed(tmp_path: Path):
    c1 = f"{1508:>10d}{4.38e8:>20.4f}{0.1425:>20.4f}"
    deck = f"""# RADIOSS STARTER DECK
/BEGIN
Sensor Spring Normal Crackle Rate Fixed Format Test
2022 0
/SENSOR/SPRING_NORMAL_CRACKLE_RATE/1
Fixed Spring Normal Crackle Rate Sensor
{c1}
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 1 in model.sensor_spring_normal_crackle_rates
    s1 = model.sensor_spring_normal_crackle_rates[1]
    assert s1.spring_id == 1508
    assert pytest.approx(s1.jnorm_crk_max) == 4.38e8
    assert pytest.approx(s1.jnorm_shot_max) == 4.38e8
    assert pytest.approx(s1.jnorm_drop_max) == 4.38e8
    assert pytest.approx(s1.jnorm_lock_max) == 4.38e8
    assert pytest.approx(s1.jnorm_pop_max) == 4.38e8
    assert pytest.approx(s1.jnorm_snp_max) == 4.38e8
    assert pytest.approx(s1.t_delay) == 0.1425
    assert len(model.sensors) == 1
    assert model.sensors[0].kind == "SPRING_NORMAL_CRACKLE_RATE"


def test_m403_sensor_spring_normal_crackle_rate_free(tmp_path: Path):
    deck = """# RADIOSS FREE DECK
/BEGIN
Sensor Spring Normal Crackle Rate Free Format Test
/SENSOR/SPRING_NORMAL_CRACKLE_RATE/2
Free Spring Normal Crackle Rate Sensor
1509, 4.98e8, 0.2295
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 2 in model.sensor_spring_normal_crackle_rates
    s2 = model.sensor_spring_normal_crackle_rates[2]
    assert s2.spring_id == 1509
    assert pytest.approx(s2.jnorm_crk_max) == 4.98e8
    assert pytest.approx(s2.t_delay) == 0.2295


def test_m403_sensor_spring_normal_crackle_rate_aliases(tmp_path: Path):
    deck = """# RADIOSS ALIAS DECK
/BEGIN
Sensor Spring Normal Crackle Rate Aliases Test
/SENSOR/SPRING_NORM_CRACKLE_RATE/391
1529, 2.595e8, 0.1465
/SENSOR/SPRING_RATE_CRACKLE_NORM/392
1530, 2.615e8, 0.1475
/SENSOR/NORMAL_CRACKLE_RATE_SPRING/393
1531, 2.635e8, 0.1485
/SENSOR/SPRING_CRACKLE_NORM/394
1532, 2.655e8, 0.1495
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    for sid in range(391, 395):
        assert sid in model.sensor_spring_normal_crackle_rates


def test_m403_sensor_spring_normal_crackle_rate_missing_card(tmp_path: Path):
    deck = """# RADIOSS MISSING CARD DECK
/BEGIN
/SENSOR/SPRING_NORMAL_CRACKLE_RATE/395
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) > 0
