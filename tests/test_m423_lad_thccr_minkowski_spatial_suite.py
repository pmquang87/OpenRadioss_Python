"""Tests for Milestone M423: LadTransverseHoneycombCoreCrushingRate Failure Model, EngElectrothermoflexomagnetoexcitonicmagnonpolaritonicResonanceEnergy, MinkowskiSpatialLinkageJoint, and SensorSpringTotalDropRate."""

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


def test_m423_fail_lad_transverse_honeycomb_core_crushing_rate_fixed(tmp_path: Path):
    c1 = f"{795.0:>20.4f}{2385.0:>20.4f}{575.0:>20.4f}{11.85:>20.4f}{0.655:>20.4f}"
    c2 = f"{1:>10d}{2:>10d}"
    c3 = f"{2370:>10d}"
    deck = f"""# RADIOSS STARTER DECK
/BEGIN
Ladeveze Transverse Honeycomb Core Crushing Rate Fixed Format Test
2022 0
/FAIL/LAD_TRANSVERSE_HONEYCOMB_CORE_CRUSHING_RATE/2370
Ladeveze Transverse Honeycomb Core Crushing Failure Model
{c1}
{c2}
{c3}
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 2370 in model.fail_ladtransversehoneycombcorecrushingrates
    fthcc = model.fail_ladtransversehoneycombcorecrushingrates[2370]
    assert pytest.approx(fthcc.sigma_thcc0) == 795.0
    assert pytest.approx(fthcc.sigma_thccc) == 2385.0
    assert pytest.approx(fthcc.gamma_thcc) == 575.0
    assert pytest.approx(fthcc.p_thcc) == 11.85
    assert pytest.approx(fthcc.d_thcc_max) == 0.655
    assert fthcc.ifail_sh == 1
    assert fthcc.ifail_so == 2
    assert fthcc.fail_id == 2370
    assert len(model.raw_fails) == 1
    assert model.raw_fails[0][1].type == "LAD_TRANSVERSE_HONEYCOMB_CORE_CRUSHING_RATE"


def test_m423_fail_lad_transverse_honeycomb_core_crushing_rate_free(tmp_path: Path):
    deck = """# RADIOSS FREE DECK
/BEGIN
Ladeveze Transverse Honeycomb Core Crushing Rate Free Format Test
/FAIL/LAD_TRANSVERSE_HONEYCOMB_CORE_CRUSHING_RATE/2371
805.0, 2415.0, 585.0, 12.05, 0.645
1, 1
2371
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 2371 in model.fail_ladtransversehoneycombcorecrushingrates
    fthcc = model.fail_ladtransversehoneycombcorecrushingrates[2371]
    assert pytest.approx(fthcc.sigma_thcc0) == 805.0
    assert pytest.approx(fthcc.sigma_thccc) == 2415.0
    assert pytest.approx(fthcc.gamma_thcc) == 585.0
    assert pytest.approx(fthcc.p_thcc) == 12.05
    assert pytest.approx(fthcc.d_thcc_max) == 0.645
    assert fthcc.fail_id == 2371


def test_m423_fail_lad_transverse_honeycomb_core_crushing_rate_aliases(tmp_path: Path):
    deck = """# RADIOSS ALIAS DECK
/BEGIN
Ladeveze Transverse Honeycomb Core Crushing Rate Aliases Test
/FAIL/LADEVEZE_TRANSVERSE_HONEYCOMB_CORE_CRUSHING_RATE/2372
785.0, 2355.0, 570.0, 11.65, 0.665
1, 1
/FAIL/LAD_THCCR/2373
785.0, 2355.0, 570.0, 11.65, 0.665
1, 1
/FAIL/LAD_THCCR_MODEL/2374
785.0, 2355.0, 570.0, 11.65, 0.665
1, 1
/FAIL/LAD_THCCR_LAW/2375
785.0, 2355.0, 570.0, 11.65, 0.665
1, 1
/FAIL/LADEVEZE_RATE_DEPENDENT_TRANSVERSE_HONEYCOMB_CORE_CRUSHING/2376
785.0, 2355.0, 570.0, 11.65, 0.665
1, 1
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 2372 in model.fail_ladtransversehoneycombcorecrushingrates
    assert 2373 in model.fail_ladtransversehoneycombcorecrushingrates
    assert 2374 in model.fail_ladtransversehoneycombcorecrushingrates
    assert 2375 in model.fail_ladtransversehoneycombcorecrushingrates
    assert 2376 in model.fail_ladtransversehoneycombcorecrushingrates
    assert len(model.raw_fails) == 5


def test_m423_fail_lad_transverse_honeycomb_core_crushing_rate_missing_card(tmp_path: Path):
    deck = """# RADIOSS MISSING CARD DECK
/BEGIN
/FAIL/LAD_TRANSVERSE_HONEYCOMB_CORE_CRUSHING_RATE/2377
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) > 0


def test_m423_eng_electrothermoflexomagnetoexcitonicmagnonpolaritonic_resonance_energy_fixed(tmp_path: Path):
    c1 = f"{0.0555:>20.4f}{785:>10d}"
    deck = f"""# RADIOSS STARTER DECK
/BEGIN
Engine Electrothermoflexomagnetoexcitonicmagnonpolaritonic Resonance Energy Fixed Format Test
2022 0
/ENG/ELECTROTHERMOFLEXOMAGNETOEXCITONICMAGNONPOLARITONIC_RESONANCE_ENERGY/685
Resonance Energy Output Directive
{c1}
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 685 in model.eng_electrothermoflexomagnetoexcitonicmagnonpolaritonic_resonance_energies
    eng = model.eng_electrothermoflexomagnetoexcitonicmagnonpolaritonic_resonance_energies[685]
    assert pytest.approx(eng.dt_etfmxmpp) == 0.0555
    assert eng.sens_id == 785


def test_m423_eng_electrothermoflexomagnetoexcitonicmagnonpolaritonic_resonance_energy_free(tmp_path: Path):
    deck = """# RADIOSS FREE DECK
/BEGIN
Engine Electrothermoflexomagnetoexcitonicmagnonpolaritonic Resonance Energy Free Format Test
/ENG/ELECTROTHERMOFLEXOMAGNETOEXCITONICMAGNONPOLARITONIC_RESONANCE_ENERGY/686
0.0565, 786
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 686 in model.eng_electrothermoflexomagnetoexcitonicmagnonpolaritonic_resonance_energies
    eng = model.eng_electrothermoflexomagnetoexcitonicmagnonpolaritonic_resonance_energies[686]
    assert pytest.approx(eng.dt_etfmxmpp) == 0.0565
    assert eng.sens_id == 786


def test_m423_eng_electrothermoflexomagnetoexcitonicmagnonpolaritonic_resonance_energy_aliases(tmp_path: Path):
    deck = """# RADIOSS ALIAS DECK
/BEGIN
Engine Electrothermoflexomagnetoexcitonicmagnonpolaritonic Resonance Energy Aliases Test
/ENG/ELECTRO_THERM_FLEXO_MAG_EXCITON_MAGNON_POLARITON_RES_WORK/687
0.0575, 787
/ENG/EELECTROTHERMOFLEXOMAGNETOEXCITONICMAGNONPOLARITONICRESONANCE/688
0.0585, 788
/ENG/ELECTROTHERMOFLEXOMAGNETOEXCITONICMAGNONPOLARITONIC_RESONANCE_DISSIPATION/689
0.0595, 789
/ENG/ET_ELECTROTHERMOFLEXOMAGNETOEXCITONICMAGNONPOLARITONIC_RESONANCE/690
0.0605, 790
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 687 in model.eng_electrothermoflexomagnetoexcitonicmagnonpolaritonic_resonance_energies
    assert 688 in model.eng_electrothermoflexomagnetoexcitonicmagnonpolaritonic_resonance_energies
    assert 689 in model.eng_electrothermoflexomagnetoexcitonicmagnonpolaritonic_resonance_energies
    assert 690 in model.eng_electrothermoflexomagnetoexcitonicmagnonpolaritonic_resonance_energies


def test_m423_eng_electrothermoflexomagnetoexcitonicmagnonpolaritonic_resonance_energy_missing_card(tmp_path: Path):
    deck = """# RADIOSS MISSING CARD DECK
/BEGIN
/ENG/ELECTROTHERMOFLEXOMAGNETOEXCITONICMAGNONPOLARITONIC_RESONANCE_ENERGY/691
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) > 0


def test_m423_lagmul_minkowski_spatial_linkage_joint_fixed(tmp_path: Path):
    c1 = f"{771:>10d}{772:>10d}{773:>10d}{6.25e7:>20.4f}{375:>10d}{5.35e-4:>20.4e}"
    c2 = f"{435.0:>20.4f}{415.0:>20.4f}{445.0:>20.4f}{365.0:>20.4f}"
    deck = f"""# RADIOSS STARTER DECK
/BEGIN
Minkowski Spatial Linkage Joint Fixed Format Test
2022 0
/LAGMUL/MINKOWSKI_SPATIAL_LINKAGE_JOINT/735
Minkowski Spatial 6R Joint
{c1}
{c2}
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 735 in model.lagmul_minkowski_spatial_linkage_joints
    joint = model.lagmul_minkowski_spatial_linkage_joints[735]
    assert joint.node1 == 771
    assert joint.node2 == 772
    assert joint.node3 == 773
    assert pytest.approx(joint.stiff) == 6.25e7
    assert joint.skew_id == 375
    assert pytest.approx(joint.tol) == 5.35e-4
    assert pytest.approx(joint.link_len_a) == 435.0
    assert pytest.approx(joint.link_len_b) == 415.0
    assert pytest.approx(joint.twist_angle_alpha) == 445.0
    assert pytest.approx(joint.offset_distance_s) == 365.0
    assert pytest.approx(joint.offset_distance_r) == 365.0
    assert pytest.approx(joint.offset_distance_v) == 365.0
    assert pytest.approx(joint.offset_distance_h) == 365.0
    assert pytest.approx(joint.offset_distance_u) == 365.0


def test_m423_lagmul_minkowski_spatial_linkage_joint_free(tmp_path: Path):
    deck = """# RADIOSS FREE DECK
/BEGIN
Minkowski Spatial Linkage Joint Free Format Test
/LAGMUL/MINKOWSKI_SPATIAL_LINKAGE_JOINT/736
871, 872, 873, 6.55e7, 376, 5.45e-4
455.0, 425.0, 465.0, 380.0
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 736 in model.lagmul_minkowski_spatial_linkage_joints
    joint = model.lagmul_minkowski_spatial_linkage_joints[736]
    assert joint.node1 == 871
    assert joint.node2 == 872
    assert joint.node3 == 873
    assert pytest.approx(joint.stiff) == 6.55e7
    assert joint.skew_id == 376
    assert pytest.approx(joint.tol) == 5.45e-4
    assert pytest.approx(joint.link_len_a) == 455.0
    assert pytest.approx(joint.link_len_b) == 425.0
    assert pytest.approx(joint.twist_angle_alpha) == 465.0
    assert pytest.approx(joint.offset_distance_s) == 380.0


def test_m423_lagmul_minkowski_spatial_linkage_joint_aliases(tmp_path: Path):
    deck = """# RADIOSS ALIAS DECK
/BEGIN
Minkowski Spatial Linkage Joint Aliases Test
/MINKOWSKI_SPATIAL_LINKAGE_JOINT/737
971, 972, 973, 1.0e6, 0, 1.0e-6
395.0, 385.0, 395.0, 345.0
/LAGMUL/MINKOWSKI_SPATIAL_LINKAGE/738
971, 972, 973, 1.0e6, 0, 1.0e-6
395.0, 385.0, 395.0, 345.0
/MINKOWSKI_SPATIAL_LINKAGE/739
971, 972, 973, 1.0e6, 0, 1.0e-6
395.0, 385.0, 395.0, 345.0
/MINKOWSKI_SPATIAL_MULTI_LOOP_MECHANISM/740
971, 972, 973, 1.0e6, 0, 1.0e-6
395.0, 385.0, 395.0, 345.0
/MINKOWSKI_SPATIAL_SYMMETRIC_MECHANISM/741
971, 972, 973, 1.0e6, 0, 1.0e-6
395.0, 385.0, 395.0, 345.0
/MINKOWSKI_SPATIAL_6R_MECHANISM/742
971, 972, 973, 1.0e6, 0, 1.0e-6
395.0, 385.0, 395.0, 345.0
/MINKOWSKI_SPATIAL_OVERCONSTRAINED_MECHANISM/743
971, 972, 973, 1.0e6, 0, 1.0e-6
395.0, 385.0, 395.0, 345.0
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    for jid in range(737, 744):
        assert jid in model.lagmul_minkowski_spatial_linkage_joints


def test_m423_lagmul_minkowski_spatial_linkage_joint_missing_card(tmp_path: Path):
    deck = """# RADIOSS MISSING CARD DECK
/BEGIN
/LAGMUL/MINKOWSKI_SPATIAL_LINKAGE_JOINT/744
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) > 0


def test_m423_sensor_spring_total_drop_rate_fixed(tmp_path: Path):
    c1 = f"{1818:>10d}{7.48e8:>20.4f}{0.2945:>20.4f}"
    deck = f"""# RADIOSS STARTER DECK
/BEGIN
Sensor Spring Total Drop Rate Fixed Format Test
2022 0
/SENSOR/SPRING_TOTAL_DROP_RATE/1
Fixed Spring Total Drop Rate Sensor
{c1}
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 1 in model.sensor_spring_total_drop_rates
    s1 = model.sensor_spring_total_drop_rates[1]
    assert s1.spring_id == 1818
    assert pytest.approx(s1.jtot_drop_max) == 7.48e8
    assert pytest.approx(s1.jtot_lock_max) == 7.48e8
    assert pytest.approx(s1.jtot_snp_max) == 7.48e8
    assert pytest.approx(s1.jtot_crackle_max) == 7.48e8
    assert pytest.approx(s1.jtot_shot_max) == 7.48e8
    assert pytest.approx(s1.jtot_pop_max) == 7.48e8
    assert pytest.approx(s1.jtot_crk_max) == 7.48e8
    assert pytest.approx(s1.t_delay) == 0.2945
    assert len(model.sensors) == 1
    assert model.sensors[0].kind == "SPRING_TOTAL_DROP_RATE"


def test_m423_sensor_spring_total_drop_rate_free(tmp_path: Path):
    deck = """# RADIOSS FREE DECK
/BEGIN
Sensor Spring Total Drop Rate Free Format Test
/SENSOR/SPRING_TOTAL_DROP_RATE/2
Free Spring Total Drop Rate Sensor
1819, 8.08e8, 0.3935
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 2 in model.sensor_spring_total_drop_rates
    s2 = model.sensor_spring_total_drop_rates[2]
    assert s2.spring_id == 1819
    assert pytest.approx(s2.jtot_drop_max) == 8.08e8
    assert pytest.approx(s2.t_delay) == 0.3935


def test_m423_sensor_spring_total_drop_rate_aliases(tmp_path: Path):
    deck = """# RADIOSS ALIAS DECK
/BEGIN
Sensor Spring Total Drop Rate Aliases Test
/SENSOR/SPRING_TOT_DROP_RATE/681
1839, 4.595e8, 0.3465
/SENSOR/SPRING_RATE_DROP_TOT/682
1840, 4.615e8, 0.3475
/SENSOR/TOTAL_DROP_RATE_SPRING/683
1841, 4.635e8, 0.3485
/SENSOR/SPRING_DROP_TOT/684
1842, 4.655e8, 0.3495
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    for sid in range(681, 685):
        assert sid in model.sensor_spring_total_drop_rates


def test_m423_sensor_spring_total_drop_rate_missing_card(tmp_path: Path):
    deck = """# RADIOSS MISSING CARD DECK
/BEGIN
/SENSOR/SPRING_TOTAL_DROP_RATE/685
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) > 0
