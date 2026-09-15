"""Tests for Milestone M411: LadTransverseCoreDebondingRate Failure Model, EngElectrothermoflexomagnonicphononicpolaritonicResonanceEnergy, FourierSpatialLinkageJoint, and SensorSpringTotalPopRate."""

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


def test_m411_fail_lad_transverse_core_debonding_rate_fixed(tmp_path: Path):
    c1 = f"{625.0:>20.4f}{1875.0:>20.4f}{405.0:>20.4f}{8.45:>20.4f}{0.805:>20.4f}"
    c2 = f"{1:>10d}{2:>10d}"
    c3 = f"{2250:>10d}"
    deck = f"""# RADIOSS STARTER DECK
/BEGIN
Ladeveze Transverse Core Debonding Rate Fixed Format Test
2022 0
/FAIL/LAD_TRANSVERSE_CORE_DEBONDING_RATE/2250
Ladeveze Transverse Core Debonding Failure Model
{c1}
{c2}
{c3}
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 2250 in model.fail_ladtransversecoredebondingrates
    ftcd = model.fail_ladtransversecoredebondingrates[2250]
    assert pytest.approx(ftcd.sigma_tcd0) == 625.0
    assert pytest.approx(ftcd.sigma_tcdc) == 1875.0
    assert pytest.approx(ftcd.gamma_tcd) == 405.0
    assert pytest.approx(ftcd.p_tcd) == 8.45
    assert pytest.approx(ftcd.d_tcd_max) == 0.805
    assert ftcd.ifail_sh == 1
    assert ftcd.ifail_so == 2
    assert ftcd.fail_id == 2250
    assert len(model.raw_fails) == 1
    assert model.raw_fails[0][1].type == "LAD_TRANSVERSE_CORE_DEBONDING_RATE"


def test_m411_fail_lad_transverse_core_debonding_rate_free(tmp_path: Path):
    deck = """# RADIOSS FREE DECK
/BEGIN
Ladeveze Transverse Core Debonding Rate Free Format Test
/FAIL/LAD_TRANSVERSE_CORE_DEBONDING_RATE/2251
635.0, 1905.0, 415.0, 8.65, 0.795
1, 1
2251
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 2251 in model.fail_ladtransversecoredebondingrates
    ftcd = model.fail_ladtransversecoredebondingrates[2251]
    assert pytest.approx(ftcd.sigma_tcd0) == 635.0
    assert pytest.approx(ftcd.sigma_tcdc) == 1905.0
    assert pytest.approx(ftcd.gamma_tcd) == 415.0
    assert pytest.approx(ftcd.p_tcd) == 8.65
    assert pytest.approx(ftcd.d_tcd_max) == 0.795
    assert ftcd.fail_id == 2251


def test_m411_fail_lad_transverse_core_debonding_rate_aliases(tmp_path: Path):
    deck = """# RADIOSS ALIAS DECK
/BEGIN
Ladeveze Transverse Core Debonding Rate Aliases Test
/FAIL/LADEVEZE_TRANSVERSE_CORE_DEBONDING_RATE/2252
615.0, 1845.0, 400.0, 8.25, 0.815
1, 1
/FAIL/LAD_TCDR/2253
615.0, 1845.0, 400.0, 8.25, 0.815
1, 1
/FAIL/LAD_TCDR_MODEL/2254
615.0, 1845.0, 400.0, 8.25, 0.815
1, 1
/FAIL/LAD_TCDR_LAW/2255
615.0, 1845.0, 400.0, 8.25, 0.815
1, 1
/FAIL/LADEVEZE_RATE_DEPENDENT_TRANSVERSE_CORE_DEBONDING/2256
615.0, 1845.0, 400.0, 8.25, 0.815
1, 1
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 2252 in model.fail_ladtransversecoredebondingrates
    assert 2253 in model.fail_ladtransversecoredebondingrates
    assert 2254 in model.fail_ladtransversecoredebondingrates
    assert 2255 in model.fail_ladtransversecoredebondingrates
    assert 2256 in model.fail_ladtransversecoredebondingrates
    assert len(model.raw_fails) == 5


def test_m411_fail_lad_transverse_core_debonding_rate_missing_card(tmp_path: Path):
    deck = """# RADIOSS MISSING CARD DECK
/BEGIN
/FAIL/LAD_TRANSVERSE_CORE_DEBONDING_RATE/2257
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) > 0


def test_m411_eng_electrothermoflexomagnonicphononicpolaritonic_resonance_energy_fixed(tmp_path: Path):
    c1 = f"{0.0325:>20.4f}{555:>10d}"
    deck = f"""# RADIOSS STARTER DECK
/BEGIN
Engine Electrothermoflexomagnonicphononicpolaritonic Resonance Energy Fixed Format Test
2022 0
/ENG/ELECTROTHERMOFLEXOMAGNONICPHONONICPOLARITONIC_RESONANCE_ENERGY/455
Resonance Energy Output Directive
{c1}
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 455 in model.eng_electrothermoflexomagnonicphononicpolaritonic_resonance_energies
    eng = model.eng_electrothermoflexomagnonicphononicpolaritonic_resonance_energies[455]
    assert pytest.approx(eng.dt_etfmpp) == 0.0325
    assert eng.sens_id == 555


def test_m411_eng_electrothermoflexomagnonicphononicpolaritonic_resonance_energy_free(tmp_path: Path):
    deck = """# RADIOSS FREE DECK
/BEGIN
Engine Electrothermoflexomagnonicphononicpolaritonic Resonance Energy Free Format Test
/ENG/ELECTROTHERMOFLEXOMAGNONICPHONONICPOLARITONIC_RESONANCE_ENERGY/456
0.0335, 556
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 456 in model.eng_electrothermoflexomagnonicphononicpolaritonic_resonance_energies
    eng = model.eng_electrothermoflexomagnonicphononicpolaritonic_resonance_energies[456]
    assert pytest.approx(eng.dt_etfmpp) == 0.0335
    assert eng.sens_id == 556


def test_m411_eng_electrothermoflexomagnonicphononicpolaritonic_resonance_energy_aliases(tmp_path: Path):
    deck = """# RADIOSS ALIAS DECK
/BEGIN
Engine Electrothermoflexomagnonicphononicpolaritonic Resonance Energy Aliases Test
/ENG/ELECTRO_THERM_FLEXO_MAGNON_PHONON_POLARITON_RES_WORK/457
0.0345, 557
/ENG/EELECTROTHERMOFLEXOMAGNONICPHONONICPOLARITONICRESONANCE/458
0.0355, 558
/ENG/ELECTROTHERMOFLEXOMAGNONICPHONONICPOLARITONIC_RESONANCE_DISSIPATION/459
0.0365, 559
/ENG/ET_ELECTROTHERMOFLEXOMAGNONICPHONONICPOLARITONIC_RESONANCE/460
0.0375, 560
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 457 in model.eng_electrothermoflexomagnonicphononicpolaritonic_resonance_energies
    assert 458 in model.eng_electrothermoflexomagnonicphononicpolaritonic_resonance_energies
    assert 459 in model.eng_electrothermoflexomagnonicphononicpolaritonic_resonance_energies
    assert 460 in model.eng_electrothermoflexomagnonicphononicpolaritonic_resonance_energies


def test_m411_eng_electrothermoflexomagnonicphononicpolaritonic_resonance_energy_missing_card(tmp_path: Path):
    deck = """# RADIOSS MISSING CARD DECK
/BEGIN
/ENG/ELECTROTHERMOFLEXOMAGNONICPHONONICPOLARITONIC_RESONANCE_ENERGY/461
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) > 0


def test_m411_lagmul_fourier_spatial_linkage_joint_fixed(tmp_path: Path):
    c1 = f"{541:>10d}{542:>10d}{543:>10d}{3.92e7:>20.4f}{259:>10d}{2.85e-4:>20.4e}"
    c2 = f"{315.0:>20.4f}{295.0:>20.4f}{325.0:>20.4f}{245.0:>20.4f}"
    deck = f"""# RADIOSS STARTER DECK
/BEGIN
Fourier Spatial Linkage Joint Fixed Format Test
2022 0
/LAGMUL/FOURIER_SPATIAL_LINKAGE_JOINT/505
Fourier Spatial 6R Joint
{c1}
{c2}
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 505 in model.lagmul_fourier_spatial_linkage_joints
    joint = model.lagmul_fourier_spatial_linkage_joints[505]
    assert joint.node1 == 541
    assert joint.node2 == 542
    assert joint.node3 == 543
    assert pytest.approx(joint.stiff) == 3.92e7
    assert joint.skew_id == 259
    assert pytest.approx(joint.tol) == 2.85e-4
    assert pytest.approx(joint.link_len_a) == 315.0
    assert pytest.approx(joint.link_len_b) == 295.0
    assert pytest.approx(joint.twist_angle_alpha) == 325.0
    assert pytest.approx(joint.offset_distance_s) == 245.0
    assert pytest.approx(joint.offset_distance_r) == 245.0
    assert pytest.approx(joint.offset_distance_v) == 245.0
    assert pytest.approx(joint.offset_distance_h) == 245.0
    assert pytest.approx(joint.offset_distance_u) == 245.0


def test_m411_lagmul_fourier_spatial_linkage_joint_free(tmp_path: Path):
    deck = """# RADIOSS FREE DECK
/BEGIN
Fourier Spatial Linkage Joint Free Format Test
/LAGMUL/FOURIER_SPATIAL_LINKAGE_JOINT/506
641, 642, 643, 4.20e7, 260, 2.95e-4
335.0, 305.0, 345.0, 260.0
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 506 in model.lagmul_fourier_spatial_linkage_joints
    joint = model.lagmul_fourier_spatial_linkage_joints[506]
    assert joint.node1 == 641
    assert joint.node2 == 642
    assert joint.node3 == 643
    assert pytest.approx(joint.stiff) == 4.20e7
    assert joint.skew_id == 260
    assert pytest.approx(joint.tol) == 2.95e-4
    assert pytest.approx(joint.link_len_a) == 335.0
    assert pytest.approx(joint.link_len_b) == 305.0
    assert pytest.approx(joint.twist_angle_alpha) == 345.0
    assert pytest.approx(joint.offset_distance_s) == 260.0


def test_m411_lagmul_fourier_spatial_linkage_joint_aliases(tmp_path: Path):
    deck = """# RADIOSS ALIAS DECK
/BEGIN
Fourier Spatial Linkage Joint Aliases Test
/FOURIER_SPATIAL_LINKAGE_JOINT/507
741, 742, 743, 1.0e6, 0, 1.0e-6
275.0, 265.0, 275.0, 225.0
/LAGMUL/FOURIER_SPATIAL_LINKAGE/508
741, 742, 743, 1.0e6, 0, 1.0e-6
275.0, 265.0, 275.0, 225.0
/FOURIER_SPATIAL_LINKAGE/509
741, 742, 743, 1.0e6, 0, 1.0e-6
275.0, 265.0, 275.0, 225.0
/FOURIER_SPATIAL_MULTI_LOOP_MECHANISM/510
741, 742, 743, 1.0e6, 0, 1.0e-6
275.0, 265.0, 275.0, 225.0
/FOURIER_SPATIAL_SYMMETRIC_MECHANISM/511
741, 742, 743, 1.0e6, 0, 1.0e-6
275.0, 265.0, 275.0, 225.0
/FOURIER_SPATIAL_6R_MECHANISM/512
741, 742, 743, 1.0e6, 0, 1.0e-6
275.0, 265.0, 275.0, 225.0
/FOURIER_SPATIAL_OVERCONSTRAINED_MECHANISM/513
741, 742, 743, 1.0e6, 0, 1.0e-6
275.0, 265.0, 275.0, 225.0
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    for jid in range(507, 514):
        assert jid in model.lagmul_fourier_spatial_linkage_joints


def test_m411_lagmul_fourier_spatial_linkage_joint_missing_card(tmp_path: Path):
    deck = """# RADIOSS MISSING CARD DECK
/BEGIN
/LAGMUL/FOURIER_SPATIAL_LINKAGE_JOINT/514
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) > 0


def test_m411_sensor_spring_total_pop_rate_fixed(tmp_path: Path):
    c1 = f"{1588:>10d}{5.18e8:>20.4f}{0.1745:>20.4f}"
    deck = f"""# RADIOSS STARTER DECK
/BEGIN
Sensor Spring Total Pop Rate Fixed Format Test
2022 0
/SENSOR/SPRING_TOTAL_POP_RATE/1
Fixed Spring Total Pop Rate Sensor
{c1}
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 1 in model.sensor_spring_total_pop_rates
    s1 = model.sensor_spring_total_pop_rates[1]
    assert s1.spring_id == 1588
    assert pytest.approx(s1.jtot_pop_max) == 5.18e8
    assert pytest.approx(s1.jtot_snp_max) == 5.18e8
    assert pytest.approx(s1.jtot_crackle_max) == 5.18e8
    assert pytest.approx(s1.jtot_shot_max) == 5.18e8
    assert pytest.approx(s1.jtot_drop_max) == 5.18e8
    assert pytest.approx(s1.jtot_lock_max) == 5.18e8
    assert pytest.approx(s1.jtot_crk_max) == 5.18e8
    assert pytest.approx(s1.t_delay) == 0.1745
    assert len(model.sensors) == 1
    assert model.sensors[0].kind == "SPRING_TOTAL_POP_RATE"


def test_m411_sensor_spring_total_pop_rate_free(tmp_path: Path):
    deck = """# RADIOSS FREE DECK
/BEGIN
Sensor Spring Total Pop Rate Free Format Test
/SENSOR/SPRING_TOTAL_POP_RATE/2
Free Spring Total Pop Rate Sensor
1589, 5.78e8, 0.2735
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 2 in model.sensor_spring_total_pop_rates
    s2 = model.sensor_spring_total_pop_rates[2]
    assert s2.spring_id == 1589
    assert pytest.approx(s2.jtot_pop_max) == 5.78e8
    assert pytest.approx(s2.t_delay) == 0.2735


def test_m411_sensor_spring_total_pop_rate_aliases(tmp_path: Path):
    deck = """# RADIOSS ALIAS DECK
/BEGIN
Sensor Spring Total Pop Rate Aliases Test
/SENSOR/SPRING_TOT_POP_RATE/451
1609, 3.395e8, 0.2265
/SENSOR/SPRING_RATE_POP_TOT/452
1610, 3.415e8, 0.2275
/SENSOR/TOTAL_POP_RATE_SPRING/453
1611, 3.435e8, 0.2285
/SENSOR/SPRING_POP_TOT/454
1612, 3.455e8, 0.2295
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    for sid in range(451, 455):
        assert sid in model.sensor_spring_total_pop_rates


def test_m411_sensor_spring_total_pop_rate_missing_card(tmp_path: Path):
    deck = """# RADIOSS MISSING CARD DECK
/BEGIN
/SENSOR/SPRING_TOTAL_POP_RATE/455
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) > 0
