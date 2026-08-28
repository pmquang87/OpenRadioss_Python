"""Tests for Landmark Milestone M400: LadCoupleMatrixCrackingRate Failure Model, EngFlexothermoexcitonicpolaritonicResonanceEnergy, CayleySpatialLinkageJoint, and SensorSpringTorsionalSnapRate."""

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


def test_m400_fail_lad_couple_matrix_cracking_rate_fixed(tmp_path: Path):
    c1 = f"{580.0:>20.4f}{1740.0:>20.4f}{345.0:>20.4f}{7.45:>20.4f}{0.855:>20.4f}"
    c2 = f"{1:>10d}{2:>10d}"
    c3 = f"{2140:>10d}"
    deck = f"""# RADIOSS STARTER DECK
/BEGIN
Ladeveze Coupled Matrix Cracking Rate Fixed Format Test
2022 0
/FAIL/LAD_COUPLE_MATRIX_CRACKING_RATE/2140
Ladeveze Coupled Matrix Cracking Failure Model
{c1}
{c2}
{c3}
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 2140 in model.fail_ladcouplematrixcrackingrates
    fcmcr = model.fail_ladcouplematrixcrackingrates[2140]
    assert pytest.approx(fcmcr.sigma_cmcr0) == 580.0
    assert pytest.approx(fcmcr.sigma_cmcrc) == 1740.0
    assert pytest.approx(fcmcr.gamma_cmcr) == 345.0
    assert pytest.approx(fcmcr.p_cmcr) == 7.45
    assert pytest.approx(fcmcr.d_cmcr_max) == 0.855
    assert fcmcr.ifail_sh == 1
    assert fcmcr.ifail_so == 2
    assert fcmcr.fail_id == 2140
    assert len(model.raw_fails) == 1
    assert model.raw_fails[0][1].type == "LAD_COUPLE_MATRIX_CRACKING_RATE"


def test_m400_fail_lad_couple_matrix_cracking_rate_free(tmp_path: Path):
    deck = """# RADIOSS FREE DECK
/BEGIN
Ladeveze Coupled Matrix Cracking Rate Free Format Test
/FAIL/LAD_COUPLE_MATRIX_CRACKING_RATE/2141
590.0, 1770.0, 355.0, 7.65, 0.845
1, 1
2141
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 2141 in model.fail_ladcouplematrixcrackingrates
    fcmcr = model.fail_ladcouplematrixcrackingrates[2141]
    assert pytest.approx(fcmcr.sigma_cmcr0) == 590.0
    assert pytest.approx(fcmcr.sigma_cmcrc) == 1770.0
    assert pytest.approx(fcmcr.gamma_cmcr) == 355.0
    assert pytest.approx(fcmcr.p_cmcr) == 7.65
    assert pytest.approx(fcmcr.d_cmcr_max) == 0.845
    assert fcmcr.fail_id == 2141


def test_m400_fail_lad_couple_matrix_cracking_rate_aliases(tmp_path: Path):
    deck = """# RADIOSS ALIAS DECK
/BEGIN
Ladeveze Coupled Matrix Cracking Rate Aliases Test
/FAIL/LADEVEZE_COUPLED_MATRIX_CRACKING_RATE/2142
570.0, 1710.0, 340.0, 7.25, 0.875
1, 1
/FAIL/LAD_CMCR/2143
570.0, 1710.0, 340.0, 7.25, 0.875
1, 1
/FAIL/LAD_CMCR_MODEL/2144
570.0, 1710.0, 340.0, 7.25, 0.875
1, 1
/FAIL/LAD_CMCR_LAW/2145
570.0, 1710.0, 340.0, 7.25, 0.875
1, 1
/FAIL/LADEVEZE_RATE_DEPENDENT_COUPLED_MATRIX_CRACKING/2146
570.0, 1710.0, 340.0, 7.25, 0.875
1, 1
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 2142 in model.fail_ladcouplematrixcrackingrates
    assert 2143 in model.fail_ladcouplematrixcrackingrates
    assert 2144 in model.fail_ladcouplematrixcrackingrates
    assert 2145 in model.fail_ladcouplematrixcrackingrates
    assert 2146 in model.fail_ladcouplematrixcrackingrates
    assert len(model.raw_fails) == 5


def test_m400_fail_lad_couple_matrix_cracking_rate_missing_card(tmp_path: Path):
    deck = """# RADIOSS MISSING CARD DECK
/BEGIN
/FAIL/LAD_COUPLE_MATRIX_CRACKING_RATE/2147
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) > 0


def test_m400_eng_flexothermoexcitonicpolaritonic_resonance_energy_fixed(tmp_path: Path):
    c1 = f"{0.0225:>20.4f}{445:>10d}"
    deck = f"""# RADIOSS STARTER DECK
/BEGIN
Engine Flexothermoexcitonicpolaritonic Resonance Energy Fixed Format Test
2022 0
/ENG/FLEXOTHERMOEXCITONICPOLARITONIC_RESONANCE_ENERGY/345
Resonance Energy Output Directive
{c1}
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 345 in model.eng_flexothermoexcitonicpolaritonic_resonance_energies
    eng = model.eng_flexothermoexcitonicpolaritonic_resonance_energies[345]
    assert pytest.approx(eng.dt_ftep) == 0.0225
    assert eng.sens_id == 445


def test_m400_eng_flexothermoexcitonicpolaritonic_resonance_energy_free(tmp_path: Path):
    deck = """# RADIOSS FREE DECK
/BEGIN
Engine Flexothermoexcitonicpolaritonic Resonance Energy Free Format Test
/ENG/FLEXOTHERMOEXCITONICPOLARITONIC_RESONANCE_ENERGY/346
0.0235, 446
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 346 in model.eng_flexothermoexcitonicpolaritonic_resonance_energies
    eng = model.eng_flexothermoexcitonicpolaritonic_resonance_energies[346]
    assert pytest.approx(eng.dt_ftep) == 0.0235
    assert eng.sens_id == 446


def test_m400_eng_flexothermoexcitonicpolaritonic_resonance_energy_aliases(tmp_path: Path):
    deck = """# RADIOSS ALIAS DECK
/BEGIN
Engine Flexothermoexcitonicpolaritonic Resonance Energy Aliases Test
/ENG/FLEXOTHERM_EXCITON_POLARITON_RES_WORK/347
0.0245, 447
/ENG/EFLEXOTHERMOEXCITONICPOLARITONICRESONANCE/348
0.0255, 448
/ENG/FLEXOTHERMOEXCITONICPOLARITONIC_RESONANCE_DISSIPATION/349
0.0265, 449
/ENG/ET_FLEXOTHERMOEXCITONICPOLARITONIC_RESONANCE/350
0.0275, 450
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 347 in model.eng_flexothermoexcitonicpolaritonic_resonance_energies
    assert 348 in model.eng_flexothermoexcitonicpolaritonic_resonance_energies
    assert 349 in model.eng_flexothermoexcitonicpolaritonic_resonance_energies
    assert 350 in model.eng_flexothermoexcitonicpolaritonic_resonance_energies


def test_m400_eng_flexothermoexcitonicpolaritonic_resonance_energy_missing_card(tmp_path: Path):
    deck = """# RADIOSS MISSING CARD DECK
/BEGIN
/ENG/FLEXOTHERMOEXCITONICPOLARITONIC_RESONANCE_ENERGY/351
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) > 0


def test_m400_lagmul_cayley_spatial_linkage_joint_fixed(tmp_path: Path):
    c1 = f"{431:>10d}{432:>10d}{433:>10d}{2.86e7:>20.4f}{149:>10d}{1.85e-4:>20.4e}"
    c2 = f"{215.0:>20.4f}{198.0:>20.4f}{225.0:>20.4f}{148.0:>20.4f}"
    deck = f"""# RADIOSS STARTER DECK
/BEGIN
Cayley Spatial Linkage Joint Fixed Format Test
2022 0
/LAGMUL/CAYLEY_SPATIAL_LINKAGE_JOINT/395
Cayley Spatial 6R Joint
{c1}
{c2}
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 395 in model.lagmul_cayley_spatial_linkage_joints
    joint = model.lagmul_cayley_spatial_linkage_joints[395]
    assert joint.node1 == 431
    assert joint.node2 == 432
    assert joint.node3 == 433
    assert pytest.approx(joint.stiff) == 2.86e7
    assert joint.skew_id == 149
    assert pytest.approx(joint.tol) == 1.85e-4
    assert pytest.approx(joint.link_len_a) == 215.0
    assert pytest.approx(joint.link_len_b) == 198.0
    assert pytest.approx(joint.twist_angle_alpha) == 225.0
    assert pytest.approx(joint.offset_distance_s) == 148.0
    assert pytest.approx(joint.offset_distance_r) == 148.0
    assert pytest.approx(joint.offset_distance_v) == 148.0
    assert pytest.approx(joint.offset_distance_h) == 148.0
    assert pytest.approx(joint.offset_distance_u) == 148.0


def test_m400_lagmul_cayley_spatial_linkage_joint_free(tmp_path: Path):
    deck = """# RADIOSS FREE DECK
/BEGIN
Cayley Spatial Linkage Joint Free Format Test
/LAGMUL/CAYLEY_SPATIAL_LINKAGE_JOINT/396
531, 532, 533, 3.10e7, 150, 1.95e-4
235.0, 205.0, 245.0, 160.0
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 396 in model.lagmul_cayley_spatial_linkage_joints
    joint = model.lagmul_cayley_spatial_linkage_joints[396]
    assert joint.node1 == 531
    assert joint.node2 == 532
    assert joint.node3 == 533
    assert pytest.approx(joint.stiff) == 3.10e7
    assert joint.skew_id == 150
    assert pytest.approx(joint.tol) == 1.95e-4
    assert pytest.approx(joint.link_len_a) == 235.0
    assert pytest.approx(joint.link_len_b) == 205.0
    assert pytest.approx(joint.twist_angle_alpha) == 245.0
    assert pytest.approx(joint.offset_distance_s) == 160.0


def test_m400_lagmul_cayley_spatial_linkage_joint_aliases(tmp_path: Path):
    deck = """# RADIOSS ALIAS DECK
/BEGIN
Cayley Spatial Linkage Joint Aliases Test
/CAYLEY_SPATIAL_LINKAGE_JOINT/397
631, 632, 633, 1.0e6, 0, 1.0e-6
175.0, 165.0, 175.0, 125.0
/LAGMUL/CAYLEY_SPATIAL_LINKAGE/398
631, 632, 633, 1.0e6, 0, 1.0e-6
175.0, 165.0, 175.0, 125.0
/CAYLEY_SPATIAL_LINKAGE/399
631, 632, 633, 1.0e6, 0, 1.0e-6
175.0, 165.0, 175.0, 125.0
/CAYLEY_SPATIAL_MULTI_LOOP_MECHANISM/400
631, 632, 633, 1.0e6, 0, 1.0e-6
175.0, 165.0, 175.0, 125.0
/CAYLEY_SPATIAL_SYMMETRIC_MECHANISM/401
631, 632, 633, 1.0e6, 0, 1.0e-6
175.0, 165.0, 175.0, 125.0
/CAYLEY_SPATIAL_6R_MECHANISM/402
631, 632, 633, 1.0e6, 0, 1.0e-6
175.0, 165.0, 175.0, 125.0
/CAYLEY_SPATIAL_OVERCONSTRAINED_MECHANISM/403
631, 632, 633, 1.0e6, 0, 1.0e-6
175.0, 165.0, 175.0, 125.0
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    for jid in range(397, 404):
        assert jid in model.lagmul_cayley_spatial_linkage_joints


def test_m400_lagmul_cayley_spatial_linkage_joint_missing_card(tmp_path: Path):
    deck = """# RADIOSS MISSING CARD DECK
/BEGIN
/LAGMUL/CAYLEY_SPATIAL_LINKAGE_JOINT/404
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) > 0


def test_m400_sensor_spring_torsional_snap_rate_fixed(tmp_path: Path):
    c1 = f"{1478:>10d}{4.08e8:>20.4f}{0.1305:>20.4f}"
    deck = f"""# RADIOSS STARTER DECK
/BEGIN
Sensor Spring Torsional Snap Rate Fixed Format Test
2022 0
/SENSOR/SPRING_TORSIONAL_SNAP_RATE/1
Fixed Spring Torsional Snap Rate Sensor
{c1}
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 1 in model.sensor_spring_torsional_snap_rates
    s1 = model.sensor_spring_torsional_snap_rates[1]
    assert s1.spring_id == 1478
    assert pytest.approx(s1.jtors_snp_max) == 4.08e8
    assert pytest.approx(s1.jtors_shot_max) == 4.08e8
    assert pytest.approx(s1.jtors_drop_max) == 4.08e8
    assert pytest.approx(s1.jtors_lock_max) == 4.08e8
    assert pytest.approx(s1.jtors_pop_max) == 4.08e8
    assert pytest.approx(s1.jtors_crackle_max) == 4.08e8
    assert pytest.approx(s1.t_delay) == 0.1305
    assert len(model.sensors) == 1
    assert model.sensors[0].kind == "SPRING_TORSIONAL_SNAP_RATE"


def test_m400_sensor_spring_torsional_snap_rate_free(tmp_path: Path):
    deck = """# RADIOSS FREE DECK
/BEGIN
Sensor Spring Torsional Snap Rate Free Format Test
/SENSOR/SPRING_TORSIONAL_SNAP_RATE/2
Free Spring Torsional Snap Rate Sensor
1479, 4.68e8, 0.1995
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 2 in model.sensor_spring_torsional_snap_rates
    s2 = model.sensor_spring_torsional_snap_rates[2]
    assert s2.spring_id == 1479
    assert pytest.approx(s2.jtors_snp_max) == 4.68e8
    assert pytest.approx(s2.t_delay) == 0.1995


def test_m400_sensor_spring_torsional_snap_rate_aliases(tmp_path: Path):
    deck = """# RADIOSS ALIAS DECK
/BEGIN
Sensor Spring Torsional Snap Rate Aliases Test
/SENSOR/SPRING_TORS_SNAP_RATE/391
1499, 2.295e8, 0.1165
/SENSOR/SPRING_RATE_SNAP_TORS/392
1500, 2.315e8, 0.1175
/SENSOR/TORSIONAL_SNAP_RATE_SPRING/393
1501, 2.335e8, 0.1185
/SENSOR/SPRING_SNAP_TORS/394
1502, 2.355e8, 0.1195
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    for sid in range(391, 395):
        assert sid in model.sensor_spring_torsional_snap_rates


def test_m400_sensor_spring_torsional_snap_rate_missing_card(tmp_path: Path):
    deck = """# RADIOSS MISSING CARD DECK
/BEGIN
/SENSOR/SPRING_TORSIONAL_SNAP_RATE/395
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) > 0
