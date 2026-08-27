"""Tests for Milestone M390: LadTransverseFiberMicroBucklingRate Failure Model, EngFlexomagnetophononicplasmonicexcitonicpolaritonicResonanceEnergy, MaverickSpatialLinkageJoint, and SensorSpringTotalAngularDropRate."""

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


def test_m390_fail_lad_transverse_fiber_micro_buckling_rate_fixed(tmp_path: Path):
    c1 = f"{340.0:>20.4f}{1020.0:>20.4f}{180.0:>20.4f}{4.95:>20.4f}{0.980:>20.4f}"
    c2 = f"{1:>10d}{2:>10d}"
    c3 = f"{2040:>10d}"
    deck = f"""# RADIOSS STARTER DECK
/BEGIN
Ladeveze Transverse Fiber Micro-Buckling Rate Fixed Format Test
2022 0
/FAIL/LAD_TRANSVERSE_FIBER_MICRO_BUCKLING_RATE/2040
Ladeveze Transverse Fiber Micro-Buckling Failure Model
{c1}
{c2}
{c3}
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 2040 in model.fail_ladtransversefibermicrobucklingrates
    ftfmbr = model.fail_ladtransversefibermicrobucklingrates[2040]
    assert pytest.approx(ftfmbr.sigma_tfmbr0) == 340.0
    assert pytest.approx(ftfmbr.sigma_tfmbrc) == 1020.0
    assert pytest.approx(ftfmbr.gamma_tfmbr) == 180.0
    assert pytest.approx(ftfmbr.p_tfmbr) == 4.95
    assert pytest.approx(ftfmbr.d_tfmbr_max) == 0.980
    assert ftfmbr.ifail_sh == 1
    assert ftfmbr.ifail_so == 2
    assert ftfmbr.fail_id == 2040
    assert len(model.raw_fails) == 1
    assert model.raw_fails[0][1].type == "LAD_TRANSVERSE_FIBER_MICRO_BUCKLING_RATE"


def test_m390_fail_lad_transverse_fiber_micro_buckling_rate_free(tmp_path: Path):
    deck = """# RADIOSS FREE DECK
/BEGIN
Ladeveze Transverse Fiber Micro-Buckling Rate Free Format Test
/FAIL/LAD_TRANSVERSE_FIBER_MICRO_BUCKLING_RATE/2041
350.0, 1050.0, 185.0, 5.15, 0.960
1, 1
2041
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 2041 in model.fail_ladtransversefibermicrobucklingrates
    ftfmbr = model.fail_ladtransversefibermicrobucklingrates[2041]
    assert pytest.approx(ftfmbr.sigma_tfmbr0) == 350.0
    assert pytest.approx(ftfmbr.sigma_tfmbrc) == 1050.0
    assert pytest.approx(ftfmbr.gamma_tfmbr) == 185.0
    assert pytest.approx(ftfmbr.p_tfmbr) == 5.15
    assert pytest.approx(ftfmbr.d_tfmbr_max) == 0.960
    assert ftfmbr.fail_id == 2041


def test_m390_fail_lad_transverse_fiber_micro_buckling_rate_aliases(tmp_path: Path):
    deck = """# RADIOSS ALIAS DECK
/BEGIN
Ladeveze Transverse Fiber Micro-Buckling Rate Aliases Test
/FAIL/LADEVEZE_TRANSVERSE_FIBER_MICRO_BUCKLING_RATE/2042
330.0, 990.0, 170.0, 4.65, 0.990
1, 1
/FAIL/LAD_TFMBR/2043
330.0, 990.0, 170.0, 4.65, 0.990
1, 1
/FAIL/LAD_TFMBR_MODEL/2044
330.0, 990.0, 170.0, 4.65, 0.990
1, 1
/FAIL/LAD_TFMBR_LAW/2045
330.0, 990.0, 170.0, 4.65, 0.990
1, 1
/FAIL/LADEVEZE_RATE_DEPENDENT_TRANSVERSE_FIBER_MICRO_BUCKLING/2046
330.0, 990.0, 170.0, 4.65, 0.990
1, 1
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 2042 in model.fail_ladtransversefibermicrobucklingrates
    assert 2043 in model.fail_ladtransversefibermicrobucklingrates
    assert 2044 in model.fail_ladtransversefibermicrobucklingrates
    assert 2045 in model.fail_ladtransversefibermicrobucklingrates
    assert 2046 in model.fail_ladtransversefibermicrobucklingrates
    assert len(model.raw_fails) == 5


def test_m390_fail_lad_transverse_fiber_micro_buckling_rate_missing_card(tmp_path: Path):
    deck = """# RADIOSS MISSING CARD DECK
/BEGIN
/FAIL/LAD_TRANSVERSE_FIBER_MICRO_BUCKLING_RATE/2047
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) > 0


def test_m390_eng_flexomagnetophononicplasmonicexcitonicpolaritonic_resonance_energy_fixed(tmp_path: Path):
    c1 = f"{0.0095:>20.4f}{345:>10d}"
    deck = f"""# RADIOSS STARTER DECK
/BEGIN
Engine Flexomagnetophononicplasmonicexcitonicpolaritonic Resonance Energy Fixed Format Test
2022 0
/ENG/FLEXOMAGNETOPHONONICPLASMONICEXCITONICPOLARITONIC_RESONANCE_ENERGY/245
Resonance Energy Output Directive
{c1}
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 245 in model.eng_flexomagnetophononicplasmonicexcitonicpolaritonic_resonance_energies
    eng = model.eng_flexomagnetophononicplasmonicexcitonicpolaritonic_resonance_energies[245]
    assert pytest.approx(eng.dt_fmppeopr) == 0.0095
    assert eng.sens_id == 345


def test_m390_eng_flexomagnetophononicplasmonicexcitonicpolaritonic_resonance_energy_free(tmp_path: Path):
    deck = """# RADIOSS FREE DECK
/BEGIN
Engine Flexomagnetophononicplasmonicexcitonicpolaritonic Resonance Energy Free Format Test
/ENG/FLEXOMAGNETOPHONONICPLASMONICEXCITONICPOLARITONIC_RESONANCE_ENERGY/246
0.0105, 346
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 246 in model.eng_flexomagnetophononicplasmonicexcitonicpolaritonic_resonance_energies
    eng = model.eng_flexomagnetophononicplasmonicexcitonicpolaritonic_resonance_energies[246]
    assert pytest.approx(eng.dt_fmppeopr) == 0.0105
    assert eng.sens_id == 346


def test_m390_eng_flexomagnetophononicplasmonicexcitonicpolaritonic_resonance_energy_aliases(tmp_path: Path):
    deck = """# RADIOSS ALIAS DECK
/BEGIN
Engine Flexomagnetophononicplasmonicexcitonicpolaritonic Resonance Energy Aliases Test
/ENG/FLEXOMAGNETO_PHONON_PLASMON_EXCITON_POLARITON_RES_WORK/247
0.0115, 347
/ENG/EFLEXOMAGNETOPHONONICPLASMONICEXCITONICPOLARITONICRESONANCE/248
0.0125, 348
/ENG/FLEXOMAGNETOPHONONICPLASMONICEXCITONICPOLARITONIC_RESONANCE_DISSIPATION/249
0.0135, 349
/ENG/EM_FLEXOMAGNETOPHONONICPLASMONICEXCITONICPOLARITONIC_RESONANCE/250
0.0145, 350
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 247 in model.eng_flexomagnetophononicplasmonicexcitonicpolaritonic_resonance_energies
    assert 248 in model.eng_flexomagnetophononicplasmonicexcitonicpolaritonic_resonance_energies
    assert 249 in model.eng_flexomagnetophononicplasmonicexcitonicpolaritonic_resonance_energies
    assert 250 in model.eng_flexomagnetophononicplasmonicexcitonicpolaritonic_resonance_energies


def test_m390_eng_flexomagnetophononicplasmonicexcitonicpolaritonic_resonance_energy_missing_card(tmp_path: Path):
    deck = """# RADIOSS MISSING CARD DECK
/BEGIN
/ENG/FLEXOMAGNETOPHONONICPLASMONICEXCITONICPOLARITONIC_RESONANCE_ENERGY/251
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) > 0


def test_m390_lagmul_maverick_spatial_linkage_joint_fixed(tmp_path: Path):
    c1 = f"{331:>10d}{332:>10d}{333:>10d}{1.86e7:>20.4f}{49:>10d}{8.5e-5:>20.4e}"
    c2 = f"{115.0:>20.4f}{98.0:>20.4f}{125.0:>20.4f}{52.0:>20.4f}"
    deck = f"""# RADIOSS STARTER DECK
/BEGIN
Maverick Spatial Linkage Joint Fixed Format Test
2022 0
/LAGMUL/MAVERICK_SPATIAL_LINKAGE_JOINT/295
Maverick Spatial 6R Joint
{c1}
{c2}
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 295 in model.lagmul_maverick_spatial_linkage_joints
    joint = model.lagmul_maverick_spatial_linkage_joints[295]
    assert joint.node1 == 331
    assert joint.node2 == 332
    assert joint.node3 == 333
    assert pytest.approx(joint.stiff) == 1.86e7
    assert joint.skew_id == 49
    assert pytest.approx(joint.tol) == 8.5e-5
    assert pytest.approx(joint.link_len_a) == 115.0
    assert pytest.approx(joint.link_len_b) == 98.0
    assert pytest.approx(joint.twist_angle_alpha) == 125.0
    assert pytest.approx(joint.offset_distance_s) == 52.0
    assert pytest.approx(joint.offset_distance_r) == 52.0
    assert pytest.approx(joint.offset_distance_v) == 52.0
    assert pytest.approx(joint.offset_distance_h) == 52.0
    assert pytest.approx(joint.offset_distance_u) == 52.0


def test_m390_lagmul_maverick_spatial_linkage_joint_free(tmp_path: Path):
    deck = """# RADIOSS FREE DECK
/BEGIN
Maverick Spatial Linkage Joint Free Format Test
/LAGMUL/MAVERICK_SPATIAL_LINKAGE_JOINT/296
431, 432, 433, 2.10e7, 50, 9.5e-5
135.0, 105.0, 145.0, 65.0
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 296 in model.lagmul_maverick_spatial_linkage_joints
    joint = model.lagmul_maverick_spatial_linkage_joints[296]
    assert joint.node1 == 431
    assert joint.node2 == 432
    assert joint.node3 == 433
    assert pytest.approx(joint.stiff) == 2.10e7
    assert joint.skew_id == 50
    assert pytest.approx(joint.tol) == 9.5e-5
    assert pytest.approx(joint.link_len_a) == 135.0
    assert pytest.approx(joint.link_len_b) == 105.0
    assert pytest.approx(joint.twist_angle_alpha) == 145.0
    assert pytest.approx(joint.offset_distance_s) == 65.0


def test_m390_lagmul_maverick_spatial_linkage_joint_aliases(tmp_path: Path):
    deck = """# RADIOSS ALIAS DECK
/BEGIN
Maverick Spatial Linkage Joint Aliases Test
/MAVERICK_SPATIAL_LINKAGE_JOINT/297
531, 532, 533, 1.0e6, 0, 1.0e-6
80.0, 70.0, 80.0, 32.0
/LAGMUL/MAVERICK_SPATIAL_LINKAGE/298
531, 532, 533, 1.0e6, 0, 1.0e-6
80.0, 70.0, 80.0, 32.0
/MAVERICK_SPATIAL_LINKAGE/299
531, 532, 533, 1.0e6, 0, 1.0e-6
80.0, 70.0, 80.0, 32.0
/MAVERICK_SPATIAL_MULTI_LOOP_MECHANISM/300
531, 532, 533, 1.0e6, 0, 1.0e-6
80.0, 70.0, 80.0, 32.0
/MAVERICK_SPATIAL_SYMMETRIC_MECHANISM/301
531, 532, 533, 1.0e6, 0, 1.0e-6
80.0, 70.0, 80.0, 32.0
/MAVERICK_SPATIAL_6R_MECHANISM/302
531, 532, 533, 1.0e6, 0, 1.0e-6
80.0, 70.0, 80.0, 32.0
/MAVERICK_SPATIAL_OVERCONSTRAINED_MECHANISM/303
531, 532, 533, 1.0e6, 0, 1.0e-6
80.0, 70.0, 80.0, 32.0
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    for jid in range(297, 304):
        assert jid in model.lagmul_maverick_spatial_linkage_joints


def test_m390_lagmul_maverick_spatial_linkage_joint_missing_card(tmp_path: Path):
    deck = """# RADIOSS MISSING CARD DECK
/BEGIN
/LAGMUL/MAVERICK_SPATIAL_LINKAGE_JOINT/304
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) > 0


def test_m390_sensor_spring_total_angular_drop_rate_fixed(tmp_path: Path):
    c1 = f"{1378:>10d}{2.98e8:>20.4f}{0.0925:>20.4f}"
    deck = f"""# RADIOSS STARTER DECK
/BEGIN
Sensor Spring Total Angular Drop Rate Fixed Format Test
2022 0
/SENSOR/SPRING_TOTAL_ANGULAR_DROP_RATE/1
Fixed Spring Total Angular Drop Rate Sensor
{c1}
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 1 in model.sensor_spring_total_angular_drop_rates
    s1 = model.sensor_spring_total_angular_drop_rates[1]
    assert s1.spring_id == 1378
    assert pytest.approx(s1.jang_drop_max) == 2.98e8
    assert pytest.approx(s1.jang_lock_max) == 2.98e8
    assert pytest.approx(s1.jang_pop_max) == 2.98e8
    assert pytest.approx(s1.jang_snp_max) == 2.98e8
    assert pytest.approx(s1.jang_crackle_max) == 2.98e8
    assert pytest.approx(s1.t_delay) == 0.0925
    assert len(model.sensors) == 1
    assert model.sensors[0].kind == "SPRING_TOTAL_ANGULAR_DROP_RATE"


def test_m390_sensor_spring_total_angular_drop_rate_free(tmp_path: Path):
    deck = """# RADIOSS FREE DECK
/BEGIN
Sensor Spring Total Angular Drop Rate Free Format Test
/SENSOR/SPRING_TOTAL_ANGULAR_DROP_RATE/2
Free Spring Total Angular Drop Rate Sensor
1379, 3.58e8, 0.1075
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 2 in model.sensor_spring_total_angular_drop_rates
    s2 = model.sensor_spring_total_angular_drop_rates[2]
    assert s2.spring_id == 1379
    assert pytest.approx(s2.jang_drop_max) == 3.58e8
    assert pytest.approx(s2.t_delay) == 0.1075


def test_m390_sensor_spring_total_angular_drop_rate_aliases(tmp_path: Path):
    deck = """# RADIOSS ALIAS DECK
/BEGIN
Sensor Spring Total Angular Drop Rate Aliases Test
/SENSOR/SPRING_TOT_ANG_DROP_RATE/390
1399, 1.295e8, 0.0245
/SENSOR/SPRING_RATE_DROP_ANG_TOT/391
1400, 1.315e8, 0.0255
/SENSOR/TOTAL_ANGULAR_DROP_RATE_SPRING/392
1401, 1.335e8, 0.0265
/SENSOR/SPRING_DROP_ANG_TOT/393
1402, 1.355e8, 0.0275
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    for sid in range(390, 394):
        assert sid in model.sensor_spring_total_angular_drop_rates


def test_m390_sensor_spring_total_angular_drop_rate_missing_card(tmp_path: Path):
    deck = """# RADIOSS MISSING CARD DECK
/BEGIN
/SENSOR/SPRING_TOTAL_ANGULAR_DROP_RATE/395
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) > 0
