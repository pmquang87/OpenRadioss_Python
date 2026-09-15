"""Tests for Milestone M425: LadDynamicHoneycombCoreShearingRate Failure Model, EngElectrothermoflexomagnetoplasmonicexcitonicmagnonpolaritonicResonanceEnergy, PoincareSpatialLinkageJoint, and SensorSpringBendingDropRate."""

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


def test_m425_fail_lad_dynamic_honeycomb_core_shearing_rate_fixed(tmp_path: Path):
    c1 = f"{835.0:>20.4f}{2505.0:>20.4f}{615.0:>20.4f}{12.65:>20.4f}{0.615:>20.4f}"
    c2 = f"{1:>10d}{2:>10d}"
    c3 = f"{2390:>10d}"
    deck = f"""# RADIOSS STARTER DECK
/BEGIN
Ladeveze Dynamic Honeycomb Core Shearing Rate Fixed Format Test
2022 0
/FAIL/LAD_DYNAMIC_HONEYCOMB_CORE_SHEARING_RATE/2390
Ladeveze Dynamic Honeycomb Core Shearing Failure Model
{c1}
{c2}
{c3}
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 2390 in model.fail_laddynamichoneycombcoreshearingrates
    fdhcs = model.fail_laddynamichoneycombcoreshearingrates[2390]
    assert pytest.approx(fdhcs.sigma_dhcs0) == 835.0
    assert pytest.approx(fdhcs.sigma_dhcsc) == 2505.0
    assert pytest.approx(fdhcs.gamma_dhcs) == 615.0
    assert pytest.approx(fdhcs.p_dhcs) == 12.65
    assert pytest.approx(fdhcs.d_dhcs_max) == 0.615
    assert fdhcs.ifail_sh == 1
    assert fdhcs.ifail_so == 2
    assert fdhcs.fail_id == 2390
    assert len(model.raw_fails) == 1
    assert model.raw_fails[0][1].type == "LAD_DYNAMIC_HONEYCOMB_CORE_SHEARING_RATE"


def test_m425_fail_lad_dynamic_honeycomb_core_shearing_rate_free(tmp_path: Path):
    deck = """# RADIOSS FREE DECK
/BEGIN
Ladeveze Dynamic Honeycomb Core Shearing Rate Free Format Test
/FAIL/LAD_DYNAMIC_HONEYCOMB_CORE_SHEARING_RATE/2391
845.0, 2535.0, 625.0, 12.85, 0.605
1, 1
2391
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 2391 in model.fail_laddynamichoneycombcoreshearingrates
    fdhcs = model.fail_laddynamichoneycombcoreshearingrates[2391]
    assert pytest.approx(fdhcs.sigma_dhcs0) == 845.0
    assert pytest.approx(fdhcs.sigma_dhcsc) == 2535.0
    assert pytest.approx(fdhcs.gamma_dhcs) == 625.0
    assert pytest.approx(fdhcs.p_dhcs) == 12.85
    assert pytest.approx(fdhcs.d_dhcs_max) == 0.605
    assert fdhcs.fail_id == 2391


def test_m425_fail_lad_dynamic_honeycomb_core_shearing_rate_aliases(tmp_path: Path):
    deck = """# RADIOSS ALIAS DECK
/BEGIN
Ladeveze Dynamic Honeycomb Core Shearing Rate Aliases Test
/FAIL/LADEVEZE_DYNAMIC_HONEYCOMB_CORE_SHEARING_RATE/2392
825.0, 2475.0, 605.0, 12.45, 0.625
1, 1
/FAIL/LAD_DHCSR/2393
825.0, 2475.0, 605.0, 12.45, 0.625
1, 1
/FAIL/LAD_DHCSR_MODEL/2394
825.0, 2475.0, 605.0, 12.45, 0.625
1, 1
/FAIL/LAD_DHCSR_LAW/2395
825.0, 2475.0, 605.0, 12.45, 0.625
1, 1
/FAIL/LADEVEZE_RATE_DEPENDENT_DYNAMIC_HONEYCOMB_CORE_SHEARING/2396
825.0, 2475.0, 605.0, 12.45, 0.625
1, 1
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 2392 in model.fail_laddynamichoneycombcoreshearingrates
    assert 2393 in model.fail_laddynamichoneycombcoreshearingrates
    assert 2394 in model.fail_laddynamichoneycombcoreshearingrates
    assert 2395 in model.fail_laddynamichoneycombcoreshearingrates
    assert 2396 in model.fail_laddynamichoneycombcoreshearingrates
    assert len(model.raw_fails) == 5


def test_m425_fail_lad_dynamic_honeycomb_core_shearing_rate_missing_card(tmp_path: Path):
    deck = """# RADIOSS MISSING CARD DECK
/BEGIN
/FAIL/LAD_DYNAMIC_HONEYCOMB_CORE_SHEARING_RATE/2397
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) > 0


def test_m425_eng_electrothermoflexomagnetoplasmonicexcitonicmagnonpolaritonic_resonance_energy_fixed(tmp_path: Path):
    c1 = f"{0.0615:>20.4f}{805:>10d}"
    deck = f"""# RADIOSS STARTER DECK
/BEGIN
Engine Electrothermoflexomagnetoplasmonicexcitonicmagnonpolaritonic Resonance Energy Fixed Format Test
2022 0
/ENG/ELECTROTHERMOFLEXOMAGNETOPLASMONICEXCITONICMAGNONPOLARITONIC_RESONANCE_ENERGY/705
Resonance Energy Output Directive
{c1}
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 705 in model.eng_electrothermoflexomagnetoplasmonicexcitonicmagnonpolaritonic_resonance_energies
    eng = model.eng_electrothermoflexomagnetoplasmonicexcitonicmagnonpolaritonic_resonance_energies[705]
    assert pytest.approx(eng.dt_etfmpxmpp) == 0.0615
    assert eng.sens_id == 805


def test_m425_eng_electrothermoflexomagnetoplasmonicexcitonicmagnonpolaritonic_resonance_energy_free(tmp_path: Path):
    deck = """# RADIOSS FREE DECK
/BEGIN
Engine Electrothermoflexomagnetoplasmonicexcitonicmagnonpolaritonic Resonance Energy Free Format Test
/ENG/ELECTROTHERMOFLEXOMAGNETOPLASMONICEXCITONICMAGNONPOLARITONIC_RESONANCE_ENERGY/706
0.0625, 806
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 706 in model.eng_electrothermoflexomagnetoplasmonicexcitonicmagnonpolaritonic_resonance_energies
    eng = model.eng_electrothermoflexomagnetoplasmonicexcitonicmagnonpolaritonic_resonance_energies[706]
    assert pytest.approx(eng.dt_etfmpxmpp) == 0.0625
    assert eng.sens_id == 806


def test_m425_eng_electrothermoflexomagnetoplasmonicexcitonicmagnonpolaritonic_resonance_energy_aliases(tmp_path: Path):
    deck = """# RADIOSS ALIAS DECK
/BEGIN
Engine Electrothermoflexomagnetoplasmonicexcitonicmagnonpolaritonic Resonance Energy Aliases Test
/ENG/ELECTRO_THERM_FLEXO_MAG_PLASMON_EXCITON_MAGNON_POLARITON_RES_WORK/707
0.0635, 807
/ENG/EELECTROTHERMOFLEXOMAGNETOPLASMONICEXCITONICMAGNONPOLARITONICRESONANCE/708
0.0645, 808
/ENG/ELECTROTHERMOFLEXOMAGNETOPLASMONICEXCITONICMAGNONPOLARITONIC_RESONANCE_DISSIPATION/709
0.0655, 809
/ENG/ET_ELECTROTHERMOFLEXOMAGNETOPLASMONICEXCITONICMAGNONPOLARITONIC_RESONANCE/710
0.0665, 810
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 707 in model.eng_electrothermoflexomagnetoplasmonicexcitonicmagnonpolaritonic_resonance_energies
    assert 708 in model.eng_electrothermoflexomagnetoplasmonicexcitonicmagnonpolaritonic_resonance_energies
    assert 709 in model.eng_electrothermoflexomagnetoplasmonicexcitonicmagnonpolaritonic_resonance_energies
    assert 710 in model.eng_electrothermoflexomagnetoplasmonicexcitonicmagnonpolaritonic_resonance_energies


def test_m425_eng_electrothermoflexomagnetoplasmonicexcitonicmagnonpolaritonic_resonance_energy_missing_card(tmp_path: Path):
    deck = """# RADIOSS MISSING CARD DECK
/BEGIN
/ENG/ELECTROTHERMOFLEXOMAGNETOPLASMONICEXCITONICMAGNONPOLARITONIC_RESONANCE_ENERGY/711
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) > 0


def test_m425_lagmul_poincare_spatial_linkage_joint_fixed(tmp_path: Path):
    c1 = f"{791:>10d}{792:>10d}{793:>10d}{6.45e7:>20.4f}{395:>10d}{5.55e-4:>20.4e}"
    c2 = f"{455.0:>20.4f}{435.0:>20.4f}{465.0:>20.4f}{385.0:>20.4f}"
    deck = f"""# RADIOSS STARTER DECK
/BEGIN
Poincare Spatial Linkage Joint Fixed Format Test
2022 0
/LAGMUL/POINCARE_SPATIAL_LINKAGE_JOINT/755
Poincare Spatial 6R Joint
{c1}
{c2}
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 755 in model.lagmul_poincare_spatial_linkage_joints
    joint = model.lagmul_poincare_spatial_linkage_joints[755]
    assert joint.node1 == 791
    assert joint.node2 == 792
    assert joint.node3 == 793
    assert pytest.approx(joint.stiff) == 6.45e7
    assert joint.skew_id == 395
    assert pytest.approx(joint.tol) == 5.55e-4
    assert pytest.approx(joint.link_len_a) == 455.0
    assert pytest.approx(joint.link_len_b) == 435.0
    assert pytest.approx(joint.twist_angle_alpha) == 465.0
    assert pytest.approx(joint.offset_distance_s) == 385.0
    assert pytest.approx(joint.offset_distance_r) == 385.0
    assert pytest.approx(joint.offset_distance_v) == 385.0
    assert pytest.approx(joint.offset_distance_h) == 385.0
    assert pytest.approx(joint.offset_distance_u) == 385.0
    assert pytest.approx(joint.offset_distance_f) == 385.0


def test_m425_lagmul_poincare_spatial_linkage_joint_free(tmp_path: Path):
    deck = """# RADIOSS FREE DECK
/BEGIN
Poincare Spatial Linkage Joint Free Format Test
/LAGMUL/POINCARE_SPATIAL_LINKAGE_JOINT/756
891, 892, 893, 6.75e7, 396, 5.65e-4
475.0, 445.0, 485.0, 400.0
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 756 in model.lagmul_poincare_spatial_linkage_joints
    joint = model.lagmul_poincare_spatial_linkage_joints[756]
    assert joint.node1 == 891
    assert joint.node2 == 892
    assert joint.node3 == 893
    assert pytest.approx(joint.stiff) == 6.75e7
    assert joint.skew_id == 396
    assert pytest.approx(joint.tol) == 5.65e-4
    assert pytest.approx(joint.link_len_a) == 475.0
    assert pytest.approx(joint.link_len_b) == 445.0
    assert pytest.approx(joint.twist_angle_alpha) == 485.0
    assert pytest.approx(joint.offset_distance_s) == 400.0


def test_m425_lagmul_poincare_spatial_linkage_joint_aliases(tmp_path: Path):
    deck = """# RADIOSS ALIAS DECK
/BEGIN
Poincare Spatial Linkage Joint Aliases Test
/POINCARE_SPATIAL_LINKAGE_JOINT/757
991, 992, 993, 1.0e6, 0, 1.0e-6
415.0, 405.0, 415.0, 365.0
/LAGMUL/POINCARE_SPATIAL_LINKAGE/758
991, 992, 993, 1.0e6, 0, 1.0e-6
415.0, 405.0, 415.0, 365.0
/POINCARE_SPATIAL_LINKAGE/759
991, 992, 993, 1.0e6, 0, 1.0e-6
415.0, 405.0, 415.0, 365.0
/POINCARE_SPATIAL_MULTI_LOOP_MECHANISM/760
991, 992, 993, 1.0e6, 0, 1.0e-6
415.0, 405.0, 415.0, 365.0
/POINCARE_SPATIAL_SYMMETRIC_MECHANISM/761
991, 992, 993, 1.0e6, 0, 1.0e-6
415.0, 405.0, 415.0, 365.0
/POINCARE_SPATIAL_6R_MECHANISM/762
991, 992, 993, 1.0e6, 0, 1.0e-6
415.0, 405.0, 415.0, 365.0
/POINCARE_SPATIAL_OVERCONSTRAINED_MECHANISM/763
991, 992, 993, 1.0e6, 0, 1.0e-6
415.0, 405.0, 415.0, 365.0
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    for jid in range(757, 764):
        assert jid in model.lagmul_poincare_spatial_linkage_joints


def test_m425_lagmul_poincare_spatial_linkage_joint_missing_card(tmp_path: Path):
    deck = """# RADIOSS MISSING CARD DECK
/BEGIN
/LAGMUL/POINCARE_SPATIAL_LINKAGE_JOINT/764
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) > 0


def test_m425_sensor_spring_bending_drop_rate_fixed(tmp_path: Path):
    c1 = f"{1838:>10d}{7.68e8:>20.4f}{0.3145:>20.4f}"
    deck = f"""# RADIOSS STARTER DECK
/BEGIN
Sensor Spring Bending Drop Rate Fixed Format Test
2022 0
/SENSOR/SPRING_BENDING_DROP_RATE/1
Fixed Spring Bending Drop Rate Sensor
{c1}
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 1 in model.sensor_spring_bending_drop_rates
    s1 = model.sensor_spring_bending_drop_rates[1]
    assert s1.spring_id == 1838
    assert pytest.approx(s1.jbend_drop_max) == 7.68e8
    assert pytest.approx(s1.jbend_lock_max) == 7.68e8
    assert pytest.approx(s1.jbend_snp_max) == 7.68e8
    assert pytest.approx(s1.jbend_crackle_max) == 7.68e8
    assert pytest.approx(s1.jbend_shot_max) == 7.68e8
    assert pytest.approx(s1.jbend_pop_max) == 7.68e8
    assert pytest.approx(s1.jbend_crk_max) == 7.68e8
    assert pytest.approx(s1.jbend_drop_rate_max) == 7.68e8
    assert pytest.approx(s1.jbend_rate_max) == 7.68e8
    assert pytest.approx(s1.t_delay) == 0.3145
    assert len(model.sensors) == 1
    assert model.sensors[0].kind == "SPRING_BENDING_DROP_RATE"


def test_m425_sensor_spring_bending_drop_rate_free(tmp_path: Path):
    deck = """# RADIOSS FREE DECK
/BEGIN
Sensor Spring Bending Drop Rate Free Format Test
/SENSOR/SPRING_BENDING_DROP_RATE/2
Free Spring Bending Drop Rate Sensor
1839, 8.28e8, 0.4135
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 2 in model.sensor_spring_bending_drop_rates
    s2 = model.sensor_spring_bending_drop_rates[2]
    assert s2.spring_id == 1839
    assert pytest.approx(s2.jbend_drop_max) == 8.28e8
    assert pytest.approx(s2.t_delay) == 0.4135


def test_m425_sensor_spring_bending_drop_rate_aliases(tmp_path: Path):
    deck = """# RADIOSS ALIAS DECK
/BEGIN
Sensor Spring Bending Drop Rate Aliases Test
/SENSOR/SPRING_BEND_DROP_RATE/701
1859, 4.795e8, 0.3665
/SENSOR/SPRING_RATE_DROP_BEND/702
1860, 4.815e8, 0.3675
/SENSOR/BENDING_DROP_RATE_SPRING/703
1861, 4.835e8, 0.3685
/SENSOR/SPRING_DROP_BEND/704
1862, 4.855e8, 0.3695
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    for sid in range(701, 705):
        assert sid in model.sensor_spring_bending_drop_rates


def test_m425_sensor_spring_bending_drop_rate_missing_card(tmp_path: Path):
    deck = """# RADIOSS MISSING CARD DECK
/BEGIN
/SENSOR/SPRING_BENDING_DROP_RATE/705
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) > 0
