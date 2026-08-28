"""Tests for Milestone M414: LadTransverseCoreCrackingRate Failure Model, EngElectrothermoflexoplasmonicexcitonicmagnonicphononicpolaritonicResonanceEnergy, RiemannSpatialLinkageJoint, and SensorSpringTotalAngularPopRate."""

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


def test_m414_fail_lad_transverse_core_cracking_rate_fixed(tmp_path: Path):
    c1 = f"{685.0:>20.4f}{2055.0:>20.4f}{465.0:>20.4f}{9.65:>20.4f}{0.745:>20.4f}"
    c2 = f"{1:>10d}{2:>10d}"
    c3 = f"{2280:>10d}"
    deck = f"""# RADIOSS STARTER DECK
/BEGIN
Ladeveze Transverse Core Cracking Rate Fixed Format Test
2022 0
/FAIL/LAD_TRANSVERSE_CORE_CRACKING_RATE/2280
Ladeveze Transverse Core Cracking Failure Model
{c1}
{c2}
{c3}
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 2280 in model.fail_ladtransversecorecrackingrates
    ftcc = model.fail_ladtransversecorecrackingrates[2280]
    assert pytest.approx(ftcc.sigma_tcc0) == 685.0
    assert pytest.approx(ftcc.sigma_tccc) == 2055.0
    assert pytest.approx(ftcc.gamma_tcc) == 465.0
    assert pytest.approx(ftcc.p_tcc) == 9.65
    assert pytest.approx(ftcc.d_tcc_max) == 0.745
    assert ftcc.ifail_sh == 1
    assert ftcc.ifail_so == 2
    assert ftcc.fail_id == 2280
    assert len(model.raw_fails) == 1
    assert model.raw_fails[0][1].type == "LAD_TRANSVERSE_CORE_CRACKING_RATE"


def test_m414_fail_lad_transverse_core_cracking_rate_free(tmp_path: Path):
    deck = """# RADIOSS FREE DECK
/BEGIN
Ladeveze Transverse Core Cracking Rate Free Format Test
/FAIL/LAD_TRANSVERSE_CORE_CRACKING_RATE/2281
695.0, 2085.0, 475.0, 9.85, 0.735
1, 1
2281
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 2281 in model.fail_ladtransversecorecrackingrates
    ftcc = model.fail_ladtransversecorecrackingrates[2281]
    assert pytest.approx(ftcc.sigma_tcc0) == 695.0
    assert pytest.approx(ftcc.sigma_tccc) == 2085.0
    assert pytest.approx(ftcc.gamma_tcc) == 475.0
    assert pytest.approx(ftcc.p_tcc) == 9.85
    assert pytest.approx(ftcc.d_tcc_max) == 0.735
    assert ftcc.fail_id == 2281


def test_m414_fail_lad_transverse_core_cracking_rate_aliases(tmp_path: Path):
    deck = """# RADIOSS ALIAS DECK
/BEGIN
Ladeveze Transverse Core Cracking Rate Aliases Test
/FAIL/LADEVEZE_TRANSVERSE_CORE_CRACKING_RATE/2282
675.0, 2025.0, 460.0, 9.45, 0.755
1, 1
/FAIL/LAD_TCCR/2283
675.0, 2025.0, 460.0, 9.45, 0.755
1, 1
/FAIL/LAD_TCCR_MODEL/2284
675.0, 2025.0, 460.0, 9.45, 0.755
1, 1
/FAIL/LAD_TCCR_LAW/2285
675.0, 2025.0, 460.0, 9.45, 0.755
1, 1
/FAIL/LADEVEZE_RATE_DEPENDENT_TRANSVERSE_CORE_CRACKING/2286
675.0, 2025.0, 460.0, 9.45, 0.755
1, 1
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 2282 in model.fail_ladtransversecorecrackingrates
    assert 2283 in model.fail_ladtransversecorecrackingrates
    assert 2284 in model.fail_ladtransversecorecrackingrates
    assert 2285 in model.fail_ladtransversecorecrackingrates
    assert 2286 in model.fail_ladtransversecorecrackingrates
    assert len(model.raw_fails) == 5


def test_m414_fail_lad_transverse_core_cracking_rate_missing_card(tmp_path: Path):
    deck = """# RADIOSS MISSING CARD DECK
/BEGIN
/FAIL/LAD_TRANSVERSE_CORE_CRACKING_RATE/2287
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) > 0


def test_m414_eng_electrothermoflexoplasmonicexcitonicmagnonicphononicpolaritonic_resonance_energy_fixed(tmp_path: Path):
    c1 = f"{0.0385:>20.4f}{615:>10d}"
    deck = f"""# RADIOSS STARTER DECK
/BEGIN
Engine Electrothermoflexoplasmonicexcitonicmagnonicphononicpolaritonic Resonance Energy Fixed Format Test
2022 0
/ENG/ELECTROTHERMOFLEXOPLASMONICEXCITONICMAGNONICPHONONICPOLARITONIC_RESONANCE_ENERGY/515
Resonance Energy Output Directive
{c1}
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 515 in model.eng_electrothermoflexoplasmonicexcitonicmagnonicphononicpolaritonic_resonance_energies
    eng = model.eng_electrothermoflexoplasmonicexcitonicmagnonicphononicpolaritonic_resonance_energies[515]
    assert pytest.approx(eng.dt_etfpempp) == 0.0385
    assert eng.sens_id == 615


def test_m414_eng_electrothermoflexoplasmonicexcitonicmagnonicphononicpolaritonic_resonance_energy_free(tmp_path: Path):
    deck = """# RADIOSS FREE DECK
/BEGIN
Engine Electrothermoflexoplasmonicexcitonicmagnonicphononicpolaritonic Resonance Energy Free Format Test
/ENG/ELECTROTHERMOFLEXOPLASMONICEXCITONICMAGNONICPHONONICPOLARITONIC_RESONANCE_ENERGY/516
0.0395, 616
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 516 in model.eng_electrothermoflexoplasmonicexcitonicmagnonicphononicpolaritonic_resonance_energies
    eng = model.eng_electrothermoflexoplasmonicexcitonicmagnonicphononicpolaritonic_resonance_energies[516]
    assert pytest.approx(eng.dt_etfpempp) == 0.0395
    assert eng.sens_id == 616


def test_m414_eng_electrothermoflexoplasmonicexcitonicmagnonicphononicpolaritonic_resonance_energy_aliases(tmp_path: Path):
    deck = """# RADIOSS ALIAS DECK
/BEGIN
Engine Electrothermoflexoplasmonicexcitonicmagnonicphononicpolaritonic Resonance Energy Aliases Test
/ENG/ELECTRO_THERM_FLEXO_PLASMON_EXCITON_MAGNON_PHONON_POLARITON_RES_WORK/517
0.0405, 617
/ENG/EELECTROTHERMOFLEXOPLASMONICEXCITONICMAGNONICPHONONICPOLARITONICRESONANCE/518
0.0415, 618
/ENG/ELECTROTHERMOFLEXOPLASMONICEXCITONICMAGNONICPHONONICPOLARITONIC_RESONANCE_DISSIPATION/519
0.0425, 619
/ENG/ET_ELECTROTHERMOFLEXOPLASMONICEXCITONICMAGNONICPHONONICPOLARITONIC_RESONANCE/520
0.0435, 620
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 517 in model.eng_electrothermoflexoplasmonicexcitonicmagnonicphononicpolaritonic_resonance_energies
    assert 518 in model.eng_electrothermoflexoplasmonicexcitonicmagnonicphononicpolaritonic_resonance_energies
    assert 519 in model.eng_electrothermoflexoplasmonicexcitonicmagnonicphononicpolaritonic_resonance_energies
    assert 520 in model.eng_electrothermoflexoplasmonicexcitonicmagnonicphononicpolaritonic_resonance_energies


def test_m414_eng_electrothermoflexoplasmonicexcitonicmagnonicphononicpolaritonic_resonance_energy_missing_card(tmp_path: Path):
    deck = """# RADIOSS MISSING CARD DECK
/BEGIN
/ENG/ELECTROTHERMOFLEXOPLASMONICEXCITONICMAGNONICPHONONICPOLARITONIC_RESONANCE_ENERGY/521
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) > 0


def test_m414_lagmul_riemann_spatial_linkage_joint_fixed(tmp_path: Path):
    c1 = f"{601:>10d}{602:>10d}{603:>10d}{4.55e7:>20.4f}{285:>10d}{3.65e-4:>20.4e}"
    c2 = f"{345.0:>20.4f}{325.0:>20.4f}{355.0:>20.4f}{275.0:>20.4f}"
    deck = f"""# RADIOSS STARTER DECK
/BEGIN
Riemann Spatial Linkage Joint Fixed Format Test
2022 0
/LAGMUL/RIEMANN_SPATIAL_LINKAGE_JOINT/565
Riemann Spatial 6R Joint
{c1}
{c2}
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 565 in model.lagmul_riemann_spatial_linkage_joints
    joint = model.lagmul_riemann_spatial_linkage_joints[565]
    assert joint.node1 == 601
    assert joint.node2 == 602
    assert joint.node3 == 603
    assert pytest.approx(joint.stiff) == 4.55e7
    assert joint.skew_id == 285
    assert pytest.approx(joint.tol) == 3.65e-4
    assert pytest.approx(joint.link_len_a) == 345.0
    assert pytest.approx(joint.link_len_b) == 325.0
    assert pytest.approx(joint.twist_angle_alpha) == 355.0
    assert pytest.approx(joint.offset_distance_s) == 275.0
    assert pytest.approx(joint.offset_distance_r) == 275.0
    assert pytest.approx(joint.offset_distance_v) == 275.0
    assert pytest.approx(joint.offset_distance_h) == 275.0
    assert pytest.approx(joint.offset_distance_u) == 275.0


def test_m414_lagmul_riemann_spatial_linkage_joint_free(tmp_path: Path):
    deck = """# RADIOSS FREE DECK
/BEGIN
Riemann Spatial Linkage Joint Free Format Test
/LAGMUL/RIEMANN_SPATIAL_LINKAGE_JOINT/566
701, 702, 703, 4.85e7, 286, 3.75e-4
365.0, 335.0, 375.0, 290.0
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 566 in model.lagmul_riemann_spatial_linkage_joints
    joint = model.lagmul_riemann_spatial_linkage_joints[566]
    assert joint.node1 == 701
    assert joint.node2 == 702
    assert joint.node3 == 703
    assert pytest.approx(joint.stiff) == 4.85e7
    assert joint.skew_id == 286
    assert pytest.approx(joint.tol) == 3.75e-4
    assert pytest.approx(joint.link_len_a) == 365.0
    assert pytest.approx(joint.link_len_b) == 335.0
    assert pytest.approx(joint.twist_angle_alpha) == 375.0
    assert pytest.approx(joint.offset_distance_s) == 290.0


def test_m414_lagmul_riemann_spatial_linkage_joint_aliases(tmp_path: Path):
    deck = """# RADIOSS ALIAS DECK
/BEGIN
Riemann Spatial Linkage Joint Aliases Test
/RIEMANN_SPATIAL_LINKAGE_JOINT/567
801, 802, 803, 1.0e6, 0, 1.0e-6
305.0, 295.0, 305.0, 255.0
/LAGMUL/RIEMANN_SPATIAL_LINKAGE/568
801, 802, 803, 1.0e6, 0, 1.0e-6
305.0, 295.0, 305.0, 255.0
/RIEMANN_SPATIAL_LINKAGE/569
801, 802, 803, 1.0e6, 0, 1.0e-6
305.0, 295.0, 305.0, 255.0
/RIEMANN_SPATIAL_MULTI_LOOP_MECHANISM/570
801, 802, 803, 1.0e6, 0, 1.0e-6
305.0, 295.0, 305.0, 255.0
/RIEMANN_SPATIAL_SYMMETRIC_MECHANISM/571
801, 802, 803, 1.0e6, 0, 1.0e-6
305.0, 295.0, 305.0, 255.0
/RIEMANN_SPATIAL_6R_MECHANISM/572
801, 802, 803, 1.0e6, 0, 1.0e-6
305.0, 295.0, 305.0, 255.0
/RIEMANN_SPATIAL_OVERCONSTRAINED_MECHANISM/573
801, 802, 803, 1.0e6, 0, 1.0e-6
305.0, 295.0, 305.0, 255.0
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    for jid in range(567, 574):
        assert jid in model.lagmul_riemann_spatial_linkage_joints


def test_m414_lagmul_riemann_spatial_linkage_joint_missing_card(tmp_path: Path):
    deck = """# RADIOSS MISSING CARD DECK
/BEGIN
/LAGMUL/RIEMANN_SPATIAL_LINKAGE_JOINT/574
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) > 0


def test_m414_sensor_spring_total_angular_pop_rate_fixed(tmp_path: Path):
    c1 = f"{1648:>10d}{5.78e8:>20.4f}{0.2045:>20.4f}"
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
    assert s1.spring_id == 1648
    assert pytest.approx(s1.jtot_ang_pop_max) == 5.78e8
    assert pytest.approx(s1.jtot_ang_snp_max) == 5.78e8
    assert pytest.approx(s1.jtot_ang_crackle_max) == 5.78e8
    assert pytest.approx(s1.jtot_ang_shot_max) == 5.78e8
    assert pytest.approx(s1.jtot_ang_drop_max) == 5.78e8
    assert pytest.approx(s1.jtot_ang_lock_max) == 5.78e8
    assert pytest.approx(s1.jtot_ang_crk_max) == 5.78e8
    assert pytest.approx(s1.t_delay) == 0.2045
    assert len(model.sensors) == 1
    assert model.sensors[0].kind == "SPRING_TOTAL_ANGULAR_POP_RATE"


def test_m414_sensor_spring_total_angular_pop_rate_free(tmp_path: Path):
    deck = """# RADIOSS FREE DECK
/BEGIN
Sensor Spring Total Angular Pop Rate Free Format Test
/SENSOR/SPRING_TOTAL_ANGULAR_POP_RATE/2
Free Spring Total Angular Pop Rate Sensor
1649, 6.38e8, 0.3035
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 2 in model.sensor_spring_total_angular_pop_rates
    s2 = model.sensor_spring_total_angular_pop_rates[2]
    assert s2.spring_id == 1649
    assert pytest.approx(s2.jtot_ang_pop_max) == 6.38e8
    assert pytest.approx(s2.t_delay) == 0.3035


def test_m414_sensor_spring_total_angular_pop_rate_aliases(tmp_path: Path):
    deck = """# RADIOSS ALIAS DECK
/BEGIN
Sensor Spring Total Angular Pop Rate Aliases Test
/SENSOR/SPRING_TOT_ANG_POP_RATE/511
1669, 3.695e8, 0.2565
/SENSOR/SPRING_RATE_POP_ANG_TOT/512
1670, 3.715e8, 0.2575
/SENSOR/TOTAL_ANGULAR_POP_RATE_SPRING/513
1671, 3.735e8, 0.2585
/SENSOR/SPRING_POP_ANG_TOT/514
1672, 3.755e8, 0.2595
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    for sid in range(511, 515):
        assert sid in model.sensor_spring_total_angular_pop_rates


def test_m414_sensor_spring_total_angular_pop_rate_missing_card(tmp_path: Path):
    deck = """# RADIOSS MISSING CARD DECK
/BEGIN
/SENSOR/SPRING_TOTAL_ANGULAR_POP_RATE/515
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) > 0
