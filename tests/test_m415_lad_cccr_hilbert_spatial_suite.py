"""Tests for Milestone M415: LadCoupleCoreCrackingRate Failure Model, EngElectrothermoflexomagnetoplasmonicphononicpolaritonicResonanceEnergy, HilbertSpatialLinkageJoint, and SensorSpringNormalLockRate."""

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


def test_m415_fail_lad_couple_core_cracking_rate_fixed(tmp_path: Path):
    c1 = f"{705.0:>20.4f}{2115.0:>20.4f}{485.0:>20.4f}{10.05:>20.4f}{0.725:>20.4f}"
    c2 = f"{1:>10d}{2:>10d}"
    c3 = f"{2290:>10d}"
    deck = f"""# RADIOSS STARTER DECK
/BEGIN
Ladeveze Coupled Core Cracking Rate Fixed Format Test
2022 0
/FAIL/LAD_COUPLE_CORE_CRACKING_RATE/2290
Ladeveze Coupled Core Cracking Failure Model
{c1}
{c2}
{c3}
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 2290 in model.fail_ladcouplecorecrackingrates
    fccc = model.fail_ladcouplecorecrackingrates[2290]
    assert pytest.approx(fccc.sigma_ccc0) == 705.0
    assert pytest.approx(fccc.sigma_cccc) == 2115.0
    assert pytest.approx(fccc.gamma_ccc) == 485.0
    assert pytest.approx(fccc.p_ccc) == 10.05
    assert pytest.approx(fccc.d_ccc_max) == 0.725
    assert fccc.ifail_sh == 1
    assert fccc.ifail_so == 2
    assert fccc.fail_id == 2290
    assert len(model.raw_fails) == 1
    assert model.raw_fails[0][1].type == "LAD_COUPLE_CORE_CRACKING_RATE"


def test_m415_fail_lad_couple_core_cracking_rate_free(tmp_path: Path):
    deck = """# RADIOSS FREE DECK
/BEGIN
Ladeveze Coupled Core Cracking Rate Free Format Test
/FAIL/LAD_COUPLE_CORE_CRACKING_RATE/2291
715.0, 2145.0, 495.0, 10.25, 0.715
1, 1
2291
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 2291 in model.fail_ladcouplecorecrackingrates
    fccc = model.fail_ladcouplecorecrackingrates[2291]
    assert pytest.approx(fccc.sigma_ccc0) == 715.0
    assert pytest.approx(fccc.sigma_cccc) == 2145.0
    assert pytest.approx(fccc.gamma_ccc) == 495.0
    assert pytest.approx(fccc.p_ccc) == 10.25
    assert pytest.approx(fccc.d_ccc_max) == 0.715
    assert fccc.fail_id == 2291


def test_m415_fail_lad_couple_core_cracking_rate_aliases(tmp_path: Path):
    deck = """# RADIOSS ALIAS DECK
/BEGIN
Ladeveze Coupled Core Cracking Rate Aliases Test
/FAIL/LADEVEZE_COUPLED_CORE_CRACKING_RATE/2292
695.0, 2085.0, 480.0, 9.85, 0.735
1, 1
/FAIL/LAD_CCCR/2293
695.0, 2085.0, 480.0, 9.85, 0.735
1, 1
/FAIL/LAD_CCCR_MODEL/2294
695.0, 2085.0, 480.0, 9.85, 0.735
1, 1
/FAIL/LAD_CCCR_LAW/2295
695.0, 2085.0, 480.0, 9.85, 0.735
1, 1
/FAIL/LADEVEZE_RATE_DEPENDENT_COUPLED_CORE_CRACKING/2296
695.0, 2085.0, 480.0, 9.85, 0.735
1, 1
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 2292 in model.fail_ladcouplecorecrackingrates
    assert 2293 in model.fail_ladcouplecorecrackingrates
    assert 2294 in model.fail_ladcouplecorecrackingrates
    assert 2295 in model.fail_ladcouplecorecrackingrates
    assert 2296 in model.fail_ladcouplecorecrackingrates
    assert len(model.raw_fails) == 5


def test_m415_fail_lad_couple_core_cracking_rate_missing_card(tmp_path: Path):
    deck = """# RADIOSS MISSING CARD DECK
/BEGIN
/FAIL/LAD_COUPLE_CORE_CRACKING_RATE/2297
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) > 0


def test_m415_eng_electrothermoflexomagnetoplasmonicphononicpolaritonic_resonance_energy_fixed(tmp_path: Path):
    c1 = f"{0.0405:>20.4f}{635:>10d}"
    deck = f"""# RADIOSS STARTER DECK
/BEGIN
Engine Electrothermoflexomagnetoplasmonicphononicpolaritonic Resonance Energy Fixed Format Test
2022 0
/ENG/ELECTROTHERMOFLEXOMAGNETOPLASMONICPHONONICPOLARITONIC_RESONANCE_ENERGY/535
Resonance Energy Output Directive
{c1}
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 535 in model.eng_electrothermoflexomagnetoplasmonicphononicpolaritonic_resonance_energies
    eng = model.eng_electrothermoflexomagnetoplasmonicphononicpolaritonic_resonance_energies[535]
    assert pytest.approx(eng.dt_etfmppp) == 0.0405
    assert eng.sens_id == 635


def test_m415_eng_electrothermoflexomagnetoplasmonicphononicpolaritonic_resonance_energy_free(tmp_path: Path):
    deck = """# RADIOSS FREE DECK
/BEGIN
Engine Electrothermoflexomagnetoplasmonicphononicpolaritonic Resonance Energy Free Format Test
/ENG/ELECTROTHERMOFLEXOMAGNETOPLASMONICPHONONICPOLARITONIC_RESONANCE_ENERGY/536
0.0415, 636
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 536 in model.eng_electrothermoflexomagnetoplasmonicphononicpolaritonic_resonance_energies
    eng = model.eng_electrothermoflexomagnetoplasmonicphononicpolaritonic_resonance_energies[536]
    assert pytest.approx(eng.dt_etfmppp) == 0.0415
    assert eng.sens_id == 636


def test_m415_eng_electrothermoflexomagnetoplasmonicphononicpolaritonic_resonance_energy_aliases(tmp_path: Path):
    deck = """# RADIOSS ALIAS DECK
/BEGIN
Engine Electrothermoflexomagnetoplasmonicphononicpolaritonic Resonance Energy Aliases Test
/ENG/ELECTRO_THERM_FLEXO_MAG_PLASMON_PHONON_POLARITON_RES_WORK/537
0.0425, 637
/ENG/EELECTROTHERMOFLEXOMAGNETOPLASMONICPHONONICPOLARITONICRESONANCE/538
0.0435, 638
/ENG/ELECTROTHERMOFLEXOMAGNETOPLASMONICPHONONICPOLARITONIC_RESONANCE_DISSIPATION/539
0.0445, 639
/ENG/ET_ELECTROTHERMOFLEXOMAGNETOPLASMONICPHONONICPOLARITONIC_RESONANCE/540
0.0455, 640
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 537 in model.eng_electrothermoflexomagnetoplasmonicphononicpolaritonic_resonance_energies
    assert 538 in model.eng_electrothermoflexomagnetoplasmonicphononicpolaritonic_resonance_energies
    assert 539 in model.eng_electrothermoflexomagnetoplasmonicphononicpolaritonic_resonance_energies
    assert 540 in model.eng_electrothermoflexomagnetoplasmonicphononicpolaritonic_resonance_energies


def test_m415_eng_electrothermoflexomagnetoplasmonicphononicpolaritonic_resonance_energy_missing_card(tmp_path: Path):
    deck = """# RADIOSS MISSING CARD DECK
/BEGIN
/ENG/ELECTROTHERMOFLEXOMAGNETOPLASMONICPHONONICPOLARITONIC_RESONANCE_ENERGY/541
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) > 0


def test_m415_lagmul_hilbert_spatial_linkage_joint_fixed(tmp_path: Path):
    c1 = f"{621:>10d}{622:>10d}{623:>10d}{4.75e7:>20.4f}{295:>10d}{3.85e-4:>20.4e}"
    c2 = f"{355.0:>20.4f}{335.0:>20.4f}{365.0:>20.4f}{285.0:>20.4f}"
    deck = f"""# RADIOSS STARTER DECK
/BEGIN
Hilbert Spatial Linkage Joint Fixed Format Test
2022 0
/LAGMUL/HILBERT_SPATIAL_LINKAGE_JOINT/585
Hilbert Spatial 6R Joint
{c1}
{c2}
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 585 in model.lagmul_hilbert_spatial_linkage_joints
    joint = model.lagmul_hilbert_spatial_linkage_joints[585]
    assert joint.node1 == 621
    assert joint.node2 == 622
    assert joint.node3 == 623
    assert pytest.approx(joint.stiff) == 4.75e7
    assert joint.skew_id == 295
    assert pytest.approx(joint.tol) == 3.85e-4
    assert pytest.approx(joint.link_len_a) == 355.0
    assert pytest.approx(joint.link_len_b) == 335.0
    assert pytest.approx(joint.twist_angle_alpha) == 365.0
    assert pytest.approx(joint.offset_distance_s) == 285.0
    assert pytest.approx(joint.offset_distance_r) == 285.0
    assert pytest.approx(joint.offset_distance_v) == 285.0
    assert pytest.approx(joint.offset_distance_h) == 285.0
    assert pytest.approx(joint.offset_distance_u) == 285.0


def test_m415_lagmul_hilbert_spatial_linkage_joint_free(tmp_path: Path):
    deck = """# RADIOSS FREE DECK
/BEGIN
Hilbert Spatial Linkage Joint Free Format Test
/LAGMUL/HILBERT_SPATIAL_LINKAGE_JOINT/586
721, 722, 723, 5.05e7, 296, 3.95e-4
375.0, 345.0, 385.0, 300.0
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 586 in model.lagmul_hilbert_spatial_linkage_joints
    joint = model.lagmul_hilbert_spatial_linkage_joints[586]
    assert joint.node1 == 721
    assert joint.node2 == 722
    assert joint.node3 == 723
    assert pytest.approx(joint.stiff) == 5.05e7
    assert joint.skew_id == 296
    assert pytest.approx(joint.tol) == 3.95e-4
    assert pytest.approx(joint.link_len_a) == 375.0
    assert pytest.approx(joint.link_len_b) == 345.0
    assert pytest.approx(joint.twist_angle_alpha) == 385.0
    assert pytest.approx(joint.offset_distance_s) == 300.0


def test_m415_lagmul_hilbert_spatial_linkage_joint_aliases(tmp_path: Path):
    deck = """# RADIOSS ALIAS DECK
/BEGIN
Hilbert Spatial Linkage Joint Aliases Test
/HILBERT_SPATIAL_LINKAGE_JOINT/587
821, 822, 823, 1.0e6, 0, 1.0e-6
315.0, 305.0, 315.0, 265.0
/LAGMUL/HILBERT_SPATIAL_LINKAGE/588
821, 822, 823, 1.0e6, 0, 1.0e-6
315.0, 305.0, 315.0, 265.0
/HILBERT_SPATIAL_LINKAGE/589
821, 822, 823, 1.0e6, 0, 1.0e-6
315.0, 305.0, 315.0, 265.0
/HILBERT_SPATIAL_MULTI_LOOP_MECHANISM/590
821, 822, 823, 1.0e6, 0, 1.0e-6
315.0, 305.0, 315.0, 265.0
/HILBERT_SPATIAL_SYMMETRIC_MECHANISM/591
821, 822, 823, 1.0e6, 0, 1.0e-6
315.0, 305.0, 315.0, 265.0
/HILBERT_SPATIAL_6R_MECHANISM/592
821, 822, 823, 1.0e6, 0, 1.0e-6
315.0, 305.0, 315.0, 265.0
/HILBERT_SPATIAL_OVERCONSTRAINED_MECHANISM/593
821, 822, 823, 1.0e6, 0, 1.0e-6
315.0, 305.0, 315.0, 265.0
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    for jid in range(587, 594):
        assert jid in model.lagmul_hilbert_spatial_linkage_joints


def test_m415_lagmul_hilbert_spatial_linkage_joint_missing_card(tmp_path: Path):
    deck = """# RADIOSS MISSING CARD DECK
/BEGIN
/LAGMUL/HILBERT_SPATIAL_LINKAGE_JOINT/594
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) > 0


def test_m415_sensor_spring_normal_lock_rate_fixed(tmp_path: Path):
    c1 = f"{1668:>10d}{5.98e8:>20.4f}{0.2145:>20.4f}"
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
    assert s1.spring_id == 1668
    assert pytest.approx(s1.jnorm_lock_max) == 5.98e8
    assert pytest.approx(s1.jnorm_snp_max) == 5.98e8
    assert pytest.approx(s1.jnorm_crackle_max) == 5.98e8
    assert pytest.approx(s1.jnorm_shot_max) == 5.98e8
    assert pytest.approx(s1.jnorm_drop_max) == 5.98e8
    assert pytest.approx(s1.jnorm_pop_max) == 5.98e8
    assert pytest.approx(s1.jnorm_crk_max) == 5.98e8
    assert pytest.approx(s1.t_delay) == 0.2145
    assert len(model.sensors) == 1
    assert model.sensors[0].kind == "SPRING_NORMAL_LOCK_RATE"


def test_m415_sensor_spring_normal_lock_rate_free(tmp_path: Path):
    deck = """# RADIOSS FREE DECK
/BEGIN
Sensor Spring Normal Lock Rate Free Format Test
/SENSOR/SPRING_NORMAL_LOCK_RATE/2
Free Spring Normal Lock Rate Sensor
1669, 6.58e8, 0.3135
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 2 in model.sensor_spring_normal_lock_rates
    s2 = model.sensor_spring_normal_lock_rates[2]
    assert s2.spring_id == 1669
    assert pytest.approx(s2.jnorm_lock_max) == 6.58e8
    assert pytest.approx(s2.t_delay) == 0.3135


def test_m415_sensor_spring_normal_lock_rate_aliases(tmp_path: Path):
    deck = """# RADIOSS ALIAS DECK
/BEGIN
Sensor Spring Normal Lock Rate Aliases Test
/SENSOR/SPRING_NORM_LOCK_RATE/531
1689, 3.795e8, 0.2665
/SENSOR/SPRING_RATE_LOCK_NORM/532
1690, 3.815e8, 0.2675
/SENSOR/NORMAL_LOCK_RATE_SPRING/533
1691, 3.835e8, 0.2685
/SENSOR/SPRING_LOCK_NORM/534
1692, 3.855e8, 0.2695
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    for sid in range(531, 535):
        assert sid in model.sensor_spring_normal_lock_rates


def test_m415_sensor_spring_normal_lock_rate_missing_card(tmp_path: Path):
    deck = """# RADIOSS MISSING CARD DECK
/BEGIN
/SENSOR/SPRING_NORMAL_LOCK_RATE/535
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) > 0
