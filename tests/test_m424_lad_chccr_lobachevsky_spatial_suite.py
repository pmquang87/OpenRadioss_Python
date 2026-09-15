"""Tests for Milestone M424: LadCoupleHoneycombCoreCrushingRate Failure Model, EngElectrothermoflexomagnetophononicmagnonpolaritonicResonanceEnergy, LobachevskySpatialLinkageJoint, and SensorSpringTorsionalDropRate."""

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


def test_m424_fail_lad_couple_honeycomb_core_crushing_rate_fixed(tmp_path: Path):
    c1 = f"{815.0:>20.4f}{2445.0:>20.4f}{595.0:>20.4f}{12.25:>20.4f}{0.635:>20.4f}"
    c2 = f"{1:>10d}{2:>10d}"
    c3 = f"{2380:>10d}"
    deck = f"""# RADIOSS STARTER DECK
/BEGIN
Ladeveze Coupled Honeycomb Core Crushing Rate Fixed Format Test
2022 0
/FAIL/LAD_COUPLE_HONEYCOMB_CORE_CRUSHING_RATE/2380
Ladeveze Coupled Honeycomb Core Crushing Failure Model
{c1}
{c2}
{c3}
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 2380 in model.fail_ladcouplehoneycombcorecrushingrates
    fchcc = model.fail_ladcouplehoneycombcorecrushingrates[2380]
    assert pytest.approx(fchcc.sigma_chcc0) == 815.0
    assert pytest.approx(fchcc.sigma_chccc) == 2445.0
    assert pytest.approx(fchcc.gamma_chcc) == 595.0
    assert pytest.approx(fchcc.p_chcc) == 12.25
    assert pytest.approx(fchcc.d_chcc_max) == 0.635
    assert fchcc.ifail_sh == 1
    assert fchcc.ifail_so == 2
    assert fchcc.fail_id == 2380
    assert len(model.raw_fails) == 1
    assert model.raw_fails[0][1].type == "LAD_COUPLE_HONEYCOMB_CORE_CRUSHING_RATE"


def test_m424_fail_lad_couple_honeycomb_core_crushing_rate_free(tmp_path: Path):
    deck = """# RADIOSS FREE DECK
/BEGIN
Ladeveze Coupled Honeycomb Core Crushing Rate Free Format Test
/FAIL/LAD_COUPLE_HONEYCOMB_CORE_CRUSHING_RATE/2381
825.0, 2475.0, 605.0, 12.45, 0.625
1, 1
2381
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 2381 in model.fail_ladcouplehoneycombcorecrushingrates
    fchcc = model.fail_ladcouplehoneycombcorecrushingrates[2381]
    assert pytest.approx(fchcc.sigma_chcc0) == 825.0
    assert pytest.approx(fchcc.sigma_chccc) == 2475.0
    assert pytest.approx(fchcc.gamma_chcc) == 605.0
    assert pytest.approx(fchcc.p_chcc) == 12.45
    assert pytest.approx(fchcc.d_chcc_max) == 0.625
    assert fchcc.fail_id == 2381


def test_m424_fail_lad_couple_honeycomb_core_crushing_rate_aliases(tmp_path: Path):
    deck = """# RADIOSS ALIAS DECK
/BEGIN
Ladeveze Coupled Honeycomb Core Crushing Rate Aliases Test
/FAIL/LADEVEZE_COUPLED_HONEYCOMB_CORE_CRUSHING_RATE/2382
805.0, 2415.0, 585.0, 12.05, 0.645
1, 1
/FAIL/LAD_CHCCR/2383
805.0, 2415.0, 585.0, 12.05, 0.645
1, 1
/FAIL/LAD_CHCCR_MODEL/2384
805.0, 2415.0, 585.0, 12.05, 0.645
1, 1
/FAIL/LAD_CHCCR_LAW/2385
805.0, 2415.0, 585.0, 12.05, 0.645
1, 1
/FAIL/LADEVEZE_RATE_DEPENDENT_COUPLED_HONEYCOMB_CORE_CRUSHING/2386
805.0, 2415.0, 585.0, 12.05, 0.645
1, 1
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 2382 in model.fail_ladcouplehoneycombcorecrushingrates
    assert 2383 in model.fail_ladcouplehoneycombcorecrushingrates
    assert 2384 in model.fail_ladcouplehoneycombcorecrushingrates
    assert 2385 in model.fail_ladcouplehoneycombcorecrushingrates
    assert 2386 in model.fail_ladcouplehoneycombcorecrushingrates
    assert len(model.raw_fails) == 5


def test_m424_fail_lad_couple_honeycomb_core_crushing_rate_missing_card(tmp_path: Path):
    deck = """# RADIOSS MISSING CARD DECK
/BEGIN
/FAIL/LAD_COUPLE_HONEYCOMB_CORE_CRUSHING_RATE/2387
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) > 0


def test_m424_eng_electrothermoflexomagnetophononicmagnonpolaritonic_resonance_energy_fixed(tmp_path: Path):
    c1 = f"{0.0575:>20.4f}{795:>10d}"
    deck = f"""# RADIOSS STARTER DECK
/BEGIN
Engine Electrothermoflexomagnetophononicmagnonpolaritonic Resonance Energy Fixed Format Test
2022 0
/ENG/ELECTROTHERMOFLEXOMAGNETOPHONONICMAGNONPOLARITONIC_RESONANCE_ENERGY/695
Resonance Energy Output Directive
{c1}
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 695 in model.eng_electrothermoflexomagnetophononicmagnonpolaritonic_resonance_energies
    eng = model.eng_electrothermoflexomagnetophononicmagnonpolaritonic_resonance_energies[695]
    assert pytest.approx(eng.dt_etfmphmpp) == 0.0575
    assert eng.sens_id == 795


def test_m424_eng_electrothermoflexomagnetophononicmagnonpolaritonic_resonance_energy_free(tmp_path: Path):
    deck = """# RADIOSS FREE DECK
/BEGIN
Engine Electrothermoflexomagnetophononicmagnonpolaritonic Resonance Energy Free Format Test
/ENG/ELECTROTHERMOFLEXOMAGNETOPHONONICMAGNONPOLARITONIC_RESONANCE_ENERGY/696
0.0585, 796
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 696 in model.eng_electrothermoflexomagnetophononicmagnonpolaritonic_resonance_energies
    eng = model.eng_electrothermoflexomagnetophononicmagnonpolaritonic_resonance_energies[696]
    assert pytest.approx(eng.dt_etfmphmpp) == 0.0585
    assert eng.sens_id == 796


def test_m424_eng_electrothermoflexomagnetophononicmagnonpolaritonic_resonance_energy_aliases(tmp_path: Path):
    deck = """# RADIOSS ALIAS DECK
/BEGIN
Engine Electrothermoflexomagnetophononicmagnonpolaritonic Resonance Energy Aliases Test
/ENG/ELECTRO_THERM_FLEXO_MAG_PHONON_MAGNON_POLARITON_RES_WORK/697
0.0595, 797
/ENG/EELECTROTHERMOFLEXOMAGNETOPHONONICMAGNONPOLARITONICRESONANCE/698
0.0605, 798
/ENG/ELECTROTHERMOFLEXOMAGNETOPHONONICMAGNONPOLARITONIC_RESONANCE_DISSIPATION/699
0.0615, 799
/ENG/ET_ELECTROTHERMOFLEXOMAGNETOPHONONICMAGNONPOLARITONIC_RESONANCE/700
0.0625, 800
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 697 in model.eng_electrothermoflexomagnetophononicmagnonpolaritonic_resonance_energies
    assert 698 in model.eng_electrothermoflexomagnetophononicmagnonpolaritonic_resonance_energies
    assert 699 in model.eng_electrothermoflexomagnetophononicmagnonpolaritonic_resonance_energies
    assert 700 in model.eng_electrothermoflexomagnetophononicmagnonpolaritonic_resonance_energies


def test_m424_eng_electrothermoflexomagnetophononicmagnonpolaritonic_resonance_energy_missing_card(tmp_path: Path):
    deck = """# RADIOSS MISSING CARD DECK
/BEGIN
/ENG/ELECTROTHERMOFLEXOMAGNETOPHONONICMAGNONPOLARITONIC_RESONANCE_ENERGY/701
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) > 0


def test_m424_lagmul_lobachevsky_spatial_linkage_joint_fixed(tmp_path: Path):
    c1 = f"{781:>10d}{782:>10d}{783:>10d}{6.35e7:>20.4f}{385:>10d}{5.45e-4:>20.4e}"
    c2 = f"{445.0:>20.4f}{425.0:>20.4f}{455.0:>20.4f}{375.0:>20.4f}"
    deck = f"""# RADIOSS STARTER DECK
/BEGIN
Lobachevsky Spatial Linkage Joint Fixed Format Test
2022 0
/LAGMUL/LOBACHEVSKY_SPATIAL_LINKAGE_JOINT/745
Lobachevsky Spatial 6R Joint
{c1}
{c2}
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 745 in model.lagmul_lobachevsky_spatial_linkage_joints
    joint = model.lagmul_lobachevsky_spatial_linkage_joints[745]
    assert joint.node1 == 781
    assert joint.node2 == 782
    assert joint.node3 == 783
    assert pytest.approx(joint.stiff) == 6.35e7
    assert joint.skew_id == 385
    assert pytest.approx(joint.tol) == 5.45e-4
    assert pytest.approx(joint.link_len_a) == 445.0
    assert pytest.approx(joint.link_len_b) == 425.0
    assert pytest.approx(joint.twist_angle_alpha) == 455.0
    assert pytest.approx(joint.offset_distance_s) == 375.0
    assert pytest.approx(joint.offset_distance_r) == 375.0
    assert pytest.approx(joint.offset_distance_v) == 375.0
    assert pytest.approx(joint.offset_distance_h) == 375.0
    assert pytest.approx(joint.offset_distance_u) == 375.0
    assert pytest.approx(joint.offset_distance_f) == 375.0


def test_m424_lagmul_lobachevsky_spatial_linkage_joint_free(tmp_path: Path):
    deck = """# RADIOSS FREE DECK
/BEGIN
Lobachevsky Spatial Linkage Joint Free Format Test
/LAGMUL/LOBACHEVSKY_SPATIAL_LINKAGE_JOINT/746
881, 882, 883, 6.65e7, 386, 5.55e-4
465.0, 435.0, 475.0, 390.0
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 746 in model.lagmul_lobachevsky_spatial_linkage_joints
    joint = model.lagmul_lobachevsky_spatial_linkage_joints[746]
    assert joint.node1 == 881
    assert joint.node2 == 882
    assert joint.node3 == 883
    assert pytest.approx(joint.stiff) == 6.65e7
    assert joint.skew_id == 386
    assert pytest.approx(joint.tol) == 5.55e-4
    assert pytest.approx(joint.link_len_a) == 465.0
    assert pytest.approx(joint.link_len_b) == 435.0
    assert pytest.approx(joint.twist_angle_alpha) == 475.0
    assert pytest.approx(joint.offset_distance_s) == 390.0


def test_m424_lagmul_lobachevsky_spatial_linkage_joint_aliases(tmp_path: Path):
    deck = """# RADIOSS ALIAS DECK
/BEGIN
Lobachevsky Spatial Linkage Joint Aliases Test
/LOBACHEVSKY_SPATIAL_LINKAGE_JOINT/747
981, 982, 983, 1.0e6, 0, 1.0e-6
405.0, 395.0, 405.0, 355.0
/LAGMUL/LOBACHEVSKY_SPATIAL_LINKAGE/748
981, 982, 983, 1.0e6, 0, 1.0e-6
405.0, 395.0, 405.0, 355.0
/LOBACHEVSKY_SPATIAL_LINKAGE/749
981, 982, 983, 1.0e6, 0, 1.0e-6
405.0, 395.0, 405.0, 355.0
/LOBACHEVSKY_SPATIAL_MULTI_LOOP_MECHANISM/750
981, 982, 983, 1.0e6, 0, 1.0e-6
405.0, 395.0, 405.0, 355.0
/LOBACHEVSKY_SPATIAL_SYMMETRIC_MECHANISM/751
981, 982, 983, 1.0e6, 0, 1.0e-6
405.0, 395.0, 405.0, 355.0
/LOBACHEVSKY_SPATIAL_6R_MECHANISM/752
981, 982, 983, 1.0e6, 0, 1.0e-6
405.0, 395.0, 405.0, 355.0
/LOBACHEVSKY_SPATIAL_OVERCONSTRAINED_MECHANISM/753
981, 982, 983, 1.0e6, 0, 1.0e-6
405.0, 395.0, 405.0, 355.0
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    for jid in range(747, 754):
        assert jid in model.lagmul_lobachevsky_spatial_linkage_joints


def test_m424_lagmul_lobachevsky_spatial_linkage_joint_missing_card(tmp_path: Path):
    deck = """# RADIOSS MISSING CARD DECK
/BEGIN
/LAGMUL/LOBACHEVSKY_SPATIAL_LINKAGE_JOINT/754
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) > 0


def test_m424_sensor_spring_torsional_drop_rate_fixed(tmp_path: Path):
    c1 = f"{1828:>10d}{7.58e8:>20.4f}{0.3045:>20.4f}"
    deck = f"""# RADIOSS STARTER DECK
/BEGIN
Sensor Spring Torsional Drop Rate Fixed Format Test
2022 0
/SENSOR/SPRING_TORSIONAL_DROP_RATE/1
Fixed Spring Torsional Drop Rate Sensor
{c1}
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 1 in model.sensor_spring_torsional_drop_rates
    s1 = model.sensor_spring_torsional_drop_rates[1]
    assert s1.spring_id == 1828
    assert pytest.approx(s1.jtors_drop_max) == 7.58e8
    assert pytest.approx(s1.jtors_lock_max) == 7.58e8
    assert pytest.approx(s1.jtors_snp_max) == 7.58e8
    assert pytest.approx(s1.jtors_crackle_max) == 7.58e8
    assert pytest.approx(s1.jtors_shot_max) == 7.58e8
    assert pytest.approx(s1.jtors_pop_max) == 7.58e8
    assert pytest.approx(s1.jtors_crk_max) == 7.58e8
    assert pytest.approx(s1.jtors_drop_rate_max) == 7.58e8
    assert pytest.approx(s1.jtor_drop_max) == 7.58e8
    assert pytest.approx(s1.t_delay) == 0.3045
    assert len(model.sensors) == 1
    assert model.sensors[0].kind == "SPRING_TORSIONAL_DROP_RATE"


def test_m424_sensor_spring_torsional_drop_rate_free(tmp_path: Path):
    deck = """# RADIOSS FREE DECK
/BEGIN
Sensor Spring Torsional Drop Rate Free Format Test
/SENSOR/SPRING_TORSIONAL_DROP_RATE/2
Free Spring Torsional Drop Rate Sensor
1829, 8.18e8, 0.4035
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 2 in model.sensor_spring_torsional_drop_rates
    s2 = model.sensor_spring_torsional_drop_rates[2]
    assert s2.spring_id == 1829
    assert pytest.approx(s2.jtors_drop_max) == 8.18e8
    assert pytest.approx(s2.t_delay) == 0.4035


def test_m424_sensor_spring_torsional_drop_rate_aliases(tmp_path: Path):
    deck = """# RADIOSS ALIAS DECK
/BEGIN
Sensor Spring Torsional Drop Rate Aliases Test
/SENSOR/SPRING_TORS_DROP_RATE/691
1849, 4.695e8, 0.3565
/SENSOR/SPRING_RATE_DROP_TORS/692
1850, 4.715e8, 0.3575
/SENSOR/TORSIONAL_DROP_RATE_SPRING/693
1851, 4.735e8, 0.3585
/SENSOR/SPRING_DROP_TORS/694
1852, 4.755e8, 0.3595
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    for sid in range(691, 695):
        assert sid in model.sensor_spring_torsional_drop_rates


def test_m424_sensor_spring_torsional_drop_rate_missing_card(tmp_path: Path):
    deck = """# RADIOSS MISSING CARD DECK
/BEGIN
/SENSOR/SPRING_TORSIONAL_DROP_RATE/695
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) > 0
