"""Tests for Milestone M391: LadCoupleFiberMicroBucklingRate Failure Model, EngFlexomagnetophononicplasmonicexcitonicmagnonicpolaritonicResonanceEnergy, KrauseSpatialLinkageJoint, and SensorSpringNormalShotRate."""

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


def test_m391_fail_lad_couple_fiber_micro_buckling_rate_fixed(tmp_path: Path):
    c1 = f"{360.0:>20.4f}{1080.0:>20.4f}{195.0:>20.4f}{5.25:>20.4f}{0.975:>20.4f}"
    c2 = f"{1:>10d}{2:>10d}"
    c3 = f"{2050:>10d}"
    deck = f"""# RADIOSS STARTER DECK
/BEGIN
Ladeveze Coupled Fiber Micro-Buckling Rate Fixed Format Test
2022 0
/FAIL/LAD_COUPLE_FIBER_MICRO_BUCKLING_RATE/2050
Ladeveze Coupled Fiber Micro-Buckling Failure Model
{c1}
{c2}
{c3}
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 2050 in model.fail_ladcouplefibermicrobucklingrates
    fcfmbr = model.fail_ladcouplefibermicrobucklingrates[2050]
    assert pytest.approx(fcfmbr.sigma_cfmbr0) == 360.0
    assert pytest.approx(fcfmbr.sigma_cfmbrc) == 1080.0
    assert pytest.approx(fcfmbr.gamma_cfmbr) == 195.0
    assert pytest.approx(fcfmbr.p_cfmbr) == 5.25
    assert pytest.approx(fcfmbr.d_cfmbr_max) == 0.975
    assert fcfmbr.ifail_sh == 1
    assert fcfmbr.ifail_so == 2
    assert fcfmbr.fail_id == 2050
    assert len(model.raw_fails) == 1
    assert model.raw_fails[0][1].type == "LAD_COUPLE_FIBER_MICRO_BUCKLING_RATE"


def test_m391_fail_lad_couple_fiber_micro_buckling_rate_free(tmp_path: Path):
    deck = """# RADIOSS FREE DECK
/BEGIN
Ladeveze Coupled Fiber Micro-Buckling Rate Free Format Test
/FAIL/LAD_COUPLE_FIBER_MICRO_BUCKLING_RATE/2051
370.0, 1110.0, 205.0, 5.45, 0.955
1, 1
2051
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 2051 in model.fail_ladcouplefibermicrobucklingrates
    fcfmbr = model.fail_ladcouplefibermicrobucklingrates[2051]
    assert pytest.approx(fcfmbr.sigma_cfmbr0) == 370.0
    assert pytest.approx(fcfmbr.sigma_cfmbrc) == 1110.0
    assert pytest.approx(fcfmbr.gamma_cfmbr) == 205.0
    assert pytest.approx(fcfmbr.p_cfmbr) == 5.45
    assert pytest.approx(fcfmbr.d_cfmbr_max) == 0.955
    assert fcfmbr.fail_id == 2051


def test_m391_fail_lad_couple_fiber_micro_buckling_rate_aliases(tmp_path: Path):
    deck = """# RADIOSS ALIAS DECK
/BEGIN
Ladeveze Coupled Fiber Micro-Buckling Rate Aliases Test
/FAIL/LADEVEZE_COUPLED_FIBER_MICRO_BUCKLING_RATE/2052
350.0, 1050.0, 190.0, 5.05, 0.985
1, 1
/FAIL/LAD_CFMBR/2053
350.0, 1050.0, 190.0, 5.05, 0.985
1, 1
/FAIL/LAD_CFMBR_MODEL/2054
350.0, 1050.0, 190.0, 5.05, 0.985
1, 1
/FAIL/LAD_CFMBR_LAW/2055
350.0, 1050.0, 190.0, 5.05, 0.985
1, 1
/FAIL/LADEVEZE_RATE_DEPENDENT_COUPLED_FIBER_MICRO_BUCKLING/2056
350.0, 1050.0, 190.0, 5.05, 0.985
1, 1
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 2052 in model.fail_ladcouplefibermicrobucklingrates
    assert 2053 in model.fail_ladcouplefibermicrobucklingrates
    assert 2054 in model.fail_ladcouplefibermicrobucklingrates
    assert 2055 in model.fail_ladcouplefibermicrobucklingrates
    assert 2056 in model.fail_ladcouplefibermicrobucklingrates
    assert len(model.raw_fails) == 5


def test_m391_fail_lad_couple_fiber_micro_buckling_rate_missing_card(tmp_path: Path):
    deck = """# RADIOSS MISSING CARD DECK
/BEGIN
/FAIL/LAD_COUPLE_FIBER_MICRO_BUCKLING_RATE/2057
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) > 0


def test_m391_eng_flexomagnetophononicplasmonicexcitonicmagnonicpolaritonic_resonance_energy_fixed(tmp_path: Path):
    c1 = f"{0.0105:>20.4f}{355:>10d}"
    deck = f"""# RADIOSS STARTER DECK
/BEGIN
Engine Flexomagnetophononicplasmonicexcitonicmagnonicpolaritonic Resonance Energy Fixed Format Test
2022 0
/ENG/FLEXOMAGNETOPHONONICPLASMONICEXCITONICMAGNONICPOLARITONIC_RESONANCE_ENERGY/255
Resonance Energy Output Directive
{c1}
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 255 in model.eng_flexomagnetophononicplasmonicexcitonicmagnonicpolaritonic_resonance_energies
    eng = model.eng_flexomagnetophononicplasmonicexcitonicmagnonicpolaritonic_resonance_energies[255]
    assert pytest.approx(eng.dt_fmppempr) == 0.0105
    assert eng.sens_id == 355


def test_m391_eng_flexomagnetophononicplasmonicexcitonicmagnonicpolaritonic_resonance_energy_free(tmp_path: Path):
    deck = """# RADIOSS FREE DECK
/BEGIN
Engine Flexomagnetophononicplasmonicexcitonicmagnonicpolaritonic Resonance Energy Free Format Test
/ENG/FLEXOMAGNETOPHONONICPLASMONICEXCITONICMAGNONICPOLARITONIC_RESONANCE_ENERGY/256
0.0115, 356
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 256 in model.eng_flexomagnetophononicplasmonicexcitonicmagnonicpolaritonic_resonance_energies
    eng = model.eng_flexomagnetophononicplasmonicexcitonicmagnonicpolaritonic_resonance_energies[256]
    assert pytest.approx(eng.dt_fmppempr) == 0.0115
    assert eng.sens_id == 356


def test_m391_eng_flexomagnetophononicplasmonicexcitonicmagnonicpolaritonic_resonance_energy_aliases(tmp_path: Path):
    deck = """# RADIOSS ALIAS DECK
/BEGIN
Engine Flexomagnetophononicplasmonicexcitonicmagnonicpolaritonic Resonance Energy Aliases Test
/ENG/FLEXOMAGNETO_PHONON_PLASMON_EXCITON_MAGNON_POLARITON_RES_WORK/257
0.0125, 357
/ENG/EFLEXOMAGNETOPHONONICPLASMONICEXCITONICMAGNONICPOLARITONICRESONANCE/258
0.0135, 358
/ENG/FLEXOMAGNETOPHONONICPLASMONICEXCITONICMAGNONICPOLARITONIC_RESONANCE_DISSIPATION/259
0.0145, 359
/ENG/EM_FLEXOMAGNETOPHONONICPLASMONICEXCITONICMAGNONICPOLARITONIC_RESONANCE/260
0.0155, 360
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 257 in model.eng_flexomagnetophononicplasmonicexcitonicmagnonicpolaritonic_resonance_energies
    assert 258 in model.eng_flexomagnetophononicplasmonicexcitonicmagnonicpolaritonic_resonance_energies
    assert 259 in model.eng_flexomagnetophononicplasmonicexcitonicmagnonicpolaritonic_resonance_energies
    assert 260 in model.eng_flexomagnetophononicplasmonicexcitonicmagnonicpolaritonic_resonance_energies


def test_m391_eng_flexomagnetophononicplasmonicexcitonicmagnonicpolaritonic_resonance_energy_missing_card(tmp_path: Path):
    deck = """# RADIOSS MISSING CARD DECK
/BEGIN
/ENG/FLEXOMAGNETOPHONONICPLASMONICEXCITONICMAGNONICPOLARITONIC_RESONANCE_ENERGY/261
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) > 0


def test_m391_lagmul_krause_spatial_linkage_joint_fixed(tmp_path: Path):
    c1 = f"{341:>10d}{342:>10d}{343:>10d}{1.96e7:>20.4f}{59:>10d}{9.5e-5:>20.4e}"
    c2 = f"{125.0:>20.4f}{108.0:>20.4f}{135.0:>20.4f}{58.0:>20.4f}"
    deck = f"""# RADIOSS STARTER DECK
/BEGIN
Krause Spatial Linkage Joint Fixed Format Test
2022 0
/LAGMUL/KRAUSE_SPATIAL_LINKAGE_JOINT/305
Krause Spatial 6R Joint
{c1}
{c2}
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 305 in model.lagmul_krause_spatial_linkage_joints
    joint = model.lagmul_krause_spatial_linkage_joints[305]
    assert joint.node1 == 341
    assert joint.node2 == 342
    assert joint.node3 == 343
    assert pytest.approx(joint.stiff) == 1.96e7
    assert joint.skew_id == 59
    assert pytest.approx(joint.tol) == 9.5e-5
    assert pytest.approx(joint.link_len_a) == 125.0
    assert pytest.approx(joint.link_len_b) == 108.0
    assert pytest.approx(joint.twist_angle_alpha) == 135.0
    assert pytest.approx(joint.offset_distance_s) == 58.0
    assert pytest.approx(joint.offset_distance_r) == 58.0
    assert pytest.approx(joint.offset_distance_v) == 58.0
    assert pytest.approx(joint.offset_distance_h) == 58.0
    assert pytest.approx(joint.offset_distance_u) == 58.0


def test_m391_lagmul_krause_spatial_linkage_joint_free(tmp_path: Path):
    deck = """# RADIOSS FREE DECK
/BEGIN
Krause Spatial Linkage Joint Free Format Test
/LAGMUL/KRAUSE_SPATIAL_LINKAGE_JOINT/306
441, 442, 443, 2.20e7, 60, 1.05e-4
145.0, 115.0, 155.0, 70.0
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 306 in model.lagmul_krause_spatial_linkage_joints
    joint = model.lagmul_krause_spatial_linkage_joints[306]
    assert joint.node1 == 441
    assert joint.node2 == 442
    assert joint.node3 == 443
    assert pytest.approx(joint.stiff) == 2.20e7
    assert joint.skew_id == 60
    assert pytest.approx(joint.tol) == 1.05e-4
    assert pytest.approx(joint.link_len_a) == 145.0
    assert pytest.approx(joint.link_len_b) == 115.0
    assert pytest.approx(joint.twist_angle_alpha) == 155.0
    assert pytest.approx(joint.offset_distance_s) == 70.0


def test_m391_lagmul_krause_spatial_linkage_joint_aliases(tmp_path: Path):
    deck = """# RADIOSS ALIAS DECK
/BEGIN
Krause Spatial Linkage Joint Aliases Test
/KRAUSE_SPATIAL_LINKAGE_JOINT/307
541, 542, 543, 1.0e6, 0, 1.0e-6
85.0, 75.0, 85.0, 35.0
/LAGMUL/KRAUSE_SPATIAL_LINKAGE/308
541, 542, 543, 1.0e6, 0, 1.0e-6
85.0, 75.0, 85.0, 35.0
/KRAUSE_SPATIAL_LINKAGE/309
541, 542, 543, 1.0e6, 0, 1.0e-6
85.0, 75.0, 85.0, 35.0
/KRAUSE_SPATIAL_MULTI_LOOP_MECHANISM/310
541, 542, 543, 1.0e6, 0, 1.0e-6
85.0, 75.0, 85.0, 35.0
/KRAUSE_SPATIAL_SYMMETRIC_MECHANISM/311
541, 542, 543, 1.0e6, 0, 1.0e-6
85.0, 75.0, 85.0, 35.0
/KRAUSE_SPATIAL_6R_MECHANISM/312
541, 542, 543, 1.0e6, 0, 1.0e-6
85.0, 75.0, 85.0, 35.0
/KRAUSE_SPATIAL_OVERCONSTRAINED_MECHANISM/313
541, 542, 543, 1.0e6, 0, 1.0e-6
85.0, 75.0, 85.0, 35.0
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    for jid in range(307, 314):
        assert jid in model.lagmul_krause_spatial_linkage_joints


def test_m391_lagmul_krause_spatial_linkage_joint_missing_card(tmp_path: Path):
    deck = """# RADIOSS MISSING CARD DECK
/BEGIN
/LAGMUL/KRAUSE_SPATIAL_LINKAGE_JOINT/314
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) > 0


def test_m391_sensor_spring_normal_shot_rate_fixed(tmp_path: Path):
    c1 = f"{1388:>10d}{3.18e8:>20.4f}{0.0945:>20.4f}"
    deck = f"""# RADIOSS STARTER DECK
/BEGIN
Sensor Spring Normal Shot Rate Fixed Format Test
2022 0
/SENSOR/SPRING_NORMAL_SHOT_RATE/1
Fixed Spring Normal Shot Rate Sensor
{c1}
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 1 in model.sensor_spring_normal_shot_rates
    s1 = model.sensor_spring_normal_shot_rates[1]
    assert s1.spring_id == 1388
    assert pytest.approx(s1.jnorm_shot_max) == 3.18e8
    assert pytest.approx(s1.jnorm_drop_max) == 3.18e8
    assert pytest.approx(s1.jnorm_lock_max) == 3.18e8
    assert pytest.approx(s1.jnorm_pop_max) == 3.18e8
    assert pytest.approx(s1.jnorm_snp_max) == 3.18e8
    assert pytest.approx(s1.jnorm_crackle_max) == 3.18e8
    assert pytest.approx(s1.t_delay) == 0.0945
    assert len(model.sensors) == 1
    assert model.sensors[0].kind == "SPRING_NORMAL_SHOT_RATE"


def test_m391_sensor_spring_normal_shot_rate_free(tmp_path: Path):
    deck = """# RADIOSS FREE DECK
/BEGIN
Sensor Spring Normal Shot Rate Free Format Test
/SENSOR/SPRING_NORMAL_SHOT_RATE/2
Free Spring Normal Shot Rate Sensor
1389, 3.78e8, 0.1095
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 2 in model.sensor_spring_normal_shot_rates
    s2 = model.sensor_spring_normal_shot_rates[2]
    assert s2.spring_id == 1389
    assert pytest.approx(s2.jnorm_shot_max) == 3.78e8
    assert pytest.approx(s2.t_delay) == 0.1095


def test_m391_sensor_spring_normal_shot_rate_aliases(tmp_path: Path):
    deck = """# RADIOSS ALIAS DECK
/BEGIN
Sensor Spring Normal Shot Rate Aliases Test
/SENSOR/SPRING_NORM_SHOT_RATE/391
1409, 1.395e8, 0.0265
/SENSOR/SPRING_RATE_SHOT_NORM/392
1410, 1.415e8, 0.0275
/SENSOR/NORMAL_SHOT_RATE_SPRING/393
1411, 1.435e8, 0.0285
/SENSOR/SPRING_SHOT_NORM/394
1412, 1.455e8, 0.0295
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    for sid in range(391, 395):
        assert sid in model.sensor_spring_normal_shot_rates


def test_m391_sensor_spring_normal_shot_rate_missing_card(tmp_path: Path):
    deck = """# RADIOSS MISSING CARD DECK
/BEGIN
/SENSOR/SPRING_NORMAL_SHOT_RATE/395
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) > 0
