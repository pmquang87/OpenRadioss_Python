"""Tests for Milestone M422: LadDynamicHoneycombCoreCrushingRate Failure Model, EngElectrothermoflexomagnetoplasmonicmagnonpolaritonicResonanceEnergy, GrassmannSpatialLinkageJoint, and SensorSpringTransverseDropRate."""

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


def test_m422_fail_lad_dynamic_honeycomb_core_crushing_rate_fixed(tmp_path: Path):
    c1 = f"{785.0:>20.4f}{2355.0:>20.4f}{565.0:>20.4f}{11.65:>20.4f}{0.645:>20.4f}"
    c2 = f"{1:>10d}{2:>10d}"
    c3 = f"{2360:>10d}"
    deck = f"""# RADIOSS STARTER DECK
/BEGIN
Ladeveze Dynamic Honeycomb Core Crushing Rate Fixed Format Test
2022 0
/FAIL/LAD_DYNAMIC_HONEYCOMB_CORE_CRUSHING_RATE/2360
Ladeveze Dynamic Honeycomb Core Crushing Failure Model
{c1}
{c2}
{c3}
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 2360 in model.fail_laddynamichoneycombcorecrushingrates
    fdhcc = model.fail_laddynamichoneycombcorecrushingrates[2360]
    assert pytest.approx(fdhcc.sigma_dhcc0) == 785.0
    assert pytest.approx(fdhcc.sigma_dhccc) == 2355.0
    assert pytest.approx(fdhcc.gamma_dhcc) == 565.0
    assert pytest.approx(fdhcc.p_dhcc) == 11.65
    assert pytest.approx(fdhcc.d_dhcc_max) == 0.645
    assert fdhcc.ifail_sh == 1
    assert fdhcc.ifail_so == 2
    assert fdhcc.fail_id == 2360
    assert len(model.raw_fails) == 1
    assert model.raw_fails[0][1].type == "LAD_DYNAMIC_HONEYCOMB_CORE_CRUSHING_RATE"


def test_m422_fail_lad_dynamic_honeycomb_core_crushing_rate_free(tmp_path: Path):
    deck = """# RADIOSS FREE DECK
/BEGIN
Ladeveze Dynamic Honeycomb Core Crushing Rate Free Format Test
/FAIL/LAD_DYNAMIC_HONEYCOMB_CORE_CRUSHING_RATE/2361
795.0, 2385.0, 575.0, 11.85, 0.635
1, 1
2361
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 2361 in model.fail_laddynamichoneycombcorecrushingrates
    fdhcc = model.fail_laddynamichoneycombcorecrushingrates[2361]
    assert pytest.approx(fdhcc.sigma_dhcc0) == 795.0
    assert pytest.approx(fdhcc.sigma_dhccc) == 2385.0
    assert pytest.approx(fdhcc.gamma_dhcc) == 575.0
    assert pytest.approx(fdhcc.p_dhcc) == 11.85
    assert pytest.approx(fdhcc.d_dhcc_max) == 0.635
    assert fdhcc.fail_id == 2361


def test_m422_fail_lad_dynamic_honeycomb_core_crushing_rate_aliases(tmp_path: Path):
    deck = """# RADIOSS ALIAS DECK
/BEGIN
Ladeveze Dynamic Honeycomb Core Crushing Rate Aliases Test
/FAIL/LADEVEZE_DYNAMIC_HONEYCOMB_CORE_CRUSHING_RATE/2362
775.0, 2325.0, 560.0, 11.45, 0.655
1, 1
/FAIL/LAD_DHCCR/2363
775.0, 2325.0, 560.0, 11.45, 0.655
1, 1
/FAIL/LAD_DHCCR_MODEL/2364
775.0, 2325.0, 560.0, 11.45, 0.655
1, 1
/FAIL/LAD_DHCCR_LAW/2365
775.0, 2325.0, 560.0, 11.45, 0.655
1, 1
/FAIL/LADEVEZE_RATE_DEPENDENT_DYNAMIC_HONEYCOMB_CORE_CRUSHING/2366
775.0, 2325.0, 560.0, 11.45, 0.655
1, 1
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 2362 in model.fail_laddynamichoneycombcorecrushingrates
    assert 2363 in model.fail_laddynamichoneycombcorecrushingrates
    assert 2364 in model.fail_laddynamichoneycombcorecrushingrates
    assert 2365 in model.fail_laddynamichoneycombcorecrushingrates
    assert 2366 in model.fail_laddynamichoneycombcorecrushingrates
    assert len(model.raw_fails) == 5


def test_m422_fail_lad_dynamic_honeycomb_core_crushing_rate_missing_card(tmp_path: Path):
    deck = """# RADIOSS MISSING CARD DECK
/BEGIN
/FAIL/LAD_DYNAMIC_HONEYCOMB_CORE_CRUSHING_RATE/2367
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) > 0


def test_m422_eng_electrothermoflexomagnetoplasmonicmagnonpolaritonic_resonance_energy_fixed(tmp_path: Path):
    c1 = f"{0.0545:>20.4f}{775:>10d}"
    deck = f"""# RADIOSS STARTER DECK
/BEGIN
Engine Electrothermoflexomagnetoplasmonicmagnonpolaritonic Resonance Energy Fixed Format Test
2022 0
/ENG/ELECTROTHERMOFLEXOMAGNETOPLASMONICMAGNONPOLARITONIC_RESONANCE_ENERGY/675
Resonance Energy Output Directive
{c1}
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 675 in model.eng_electrothermoflexomagnetoplasmonicmagnonpolaritonic_resonance_energies
    eng = model.eng_electrothermoflexomagnetoplasmonicmagnonpolaritonic_resonance_energies[675]
    assert pytest.approx(eng.dt_etfmpmpp) == 0.0545
    assert eng.sens_id == 775


def test_m422_eng_electrothermoflexomagnetoplasmonicmagnonpolaritonic_resonance_energy_free(tmp_path: Path):
    deck = """# RADIOSS FREE DECK
/BEGIN
Engine Electrothermoflexomagnetoplasmonicmagnonpolaritonic Resonance Energy Free Format Test
/ENG/ELECTROTHERMOFLEXOMAGNETOPLASMONICMAGNONPOLARITONIC_RESONANCE_ENERGY/676
0.0555, 776
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 676 in model.eng_electrothermoflexomagnetoplasmonicmagnonpolaritonic_resonance_energies
    eng = model.eng_electrothermoflexomagnetoplasmonicmagnonpolaritonic_resonance_energies[676]
    assert pytest.approx(eng.dt_etfmpmpp) == 0.0555
    assert eng.sens_id == 776


def test_m422_eng_electrothermoflexomagnetoplasmonicmagnonpolaritonic_resonance_energy_aliases(tmp_path: Path):
    deck = """# RADIOSS ALIAS DECK
/BEGIN
Engine Electrothermoflexomagnetoplasmonicmagnonpolaritonic Resonance Energy Aliases Test
/ENG/ELECTRO_THERM_FLEXO_MAG_PLASMON_MAGNON_POLARITON_RES_WORK/677
0.0565, 777
/ENG/EELECTROTHERMOFLEXOMAGNETOPLASMONICMAGNONPOLARITONICRESONANCE/678
0.0575, 778
/ENG/ELECTROTHERMOFLEXOMAGNETOPLASMONICMAGNONPOLARITONIC_RESONANCE_DISSIPATION/679
0.0585, 779
/ENG/ET_ELECTROTHERMOFLEXOMAGNETOPLASMONICMAGNONPOLARITONIC_RESONANCE/680
0.0595, 780
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 677 in model.eng_electrothermoflexomagnetoplasmonicmagnonpolaritonic_resonance_energies
    assert 678 in model.eng_electrothermoflexomagnetoplasmonicmagnonpolaritonic_resonance_energies
    assert 679 in model.eng_electrothermoflexomagnetoplasmonicmagnonpolaritonic_resonance_energies
    assert 680 in model.eng_electrothermoflexomagnetoplasmonicmagnonpolaritonic_resonance_energies


def test_m422_eng_electrothermoflexomagnetoplasmonicmagnonpolaritonic_resonance_energy_missing_card(tmp_path: Path):
    deck = """# RADIOSS MISSING CARD DECK
/BEGIN
/ENG/ELECTROTHERMOFLEXOMAGNETOPLASMONICMAGNONPOLARITONIC_RESONANCE_ENERGY/681
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) > 0


def test_m422_lagmul_grassmann_spatial_linkage_joint_fixed(tmp_path: Path):
    c1 = f"{761:>10d}{762:>10d}{763:>10d}{6.15e7:>20.4f}{365:>10d}{5.25e-4:>20.4e}"
    c2 = f"{425.0:>20.4f}{405.0:>20.4f}{435.0:>20.4f}{355.0:>20.4f}"
    deck = f"""# RADIOSS STARTER DECK
/BEGIN
Grassmann Spatial Linkage Joint Fixed Format Test
2022 0
/LAGMUL/GRASSMANN_SPATIAL_LINKAGE_JOINT/725
Grassmann Spatial 6R Joint
{c1}
{c2}
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 725 in model.lagmul_grassmann_spatial_linkage_joints
    joint = model.lagmul_grassmann_spatial_linkage_joints[725]
    assert joint.node1 == 761
    assert joint.node2 == 762
    assert joint.node3 == 763
    assert pytest.approx(joint.stiff) == 6.15e7
    assert joint.skew_id == 365
    assert pytest.approx(joint.tol) == 5.25e-4
    assert pytest.approx(joint.link_len_a) == 425.0
    assert pytest.approx(joint.link_len_b) == 405.0
    assert pytest.approx(joint.twist_angle_alpha) == 435.0
    assert pytest.approx(joint.offset_distance_s) == 355.0
    assert pytest.approx(joint.offset_distance_r) == 355.0
    assert pytest.approx(joint.offset_distance_v) == 355.0
    assert pytest.approx(joint.offset_distance_h) == 355.0
    assert pytest.approx(joint.offset_distance_u) == 355.0


def test_m422_lagmul_grassmann_spatial_linkage_joint_free(tmp_path: Path):
    deck = """# RADIOSS FREE DECK
/BEGIN
Grassmann Spatial Linkage Joint Free Format Test
/LAGMUL/GRASSMANN_SPATIAL_LINKAGE_JOINT/726
861, 862, 863, 6.45e7, 366, 5.35e-4
445.0, 415.0, 455.0, 370.0
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 726 in model.lagmul_grassmann_spatial_linkage_joints
    joint = model.lagmul_grassmann_spatial_linkage_joints[726]
    assert joint.node1 == 861
    assert joint.node2 == 862
    assert joint.node3 == 863
    assert pytest.approx(joint.stiff) == 6.45e7
    assert joint.skew_id == 366
    assert pytest.approx(joint.tol) == 5.35e-4
    assert pytest.approx(joint.link_len_a) == 445.0
    assert pytest.approx(joint.link_len_b) == 415.0
    assert pytest.approx(joint.twist_angle_alpha) == 455.0
    assert pytest.approx(joint.offset_distance_s) == 370.0


def test_m422_lagmul_grassmann_spatial_linkage_joint_aliases(tmp_path: Path):
    deck = """# RADIOSS ALIAS DECK
/BEGIN
Grassmann Spatial Linkage Joint Aliases Test
/GRASSMANN_SPATIAL_LINKAGE_JOINT/727
961, 962, 963, 1.0e6, 0, 1.0e-6
385.0, 375.0, 385.0, 335.0
/LAGMUL/GRASSMANN_SPATIAL_LINKAGE/728
961, 962, 963, 1.0e6, 0, 1.0e-6
385.0, 375.0, 385.0, 335.0
/GRASSMANN_SPATIAL_LINKAGE/729
961, 962, 963, 1.0e6, 0, 1.0e-6
385.0, 375.0, 385.0, 335.0
/GRASSMANN_SPATIAL_MULTI_LOOP_MECHANISM/730
961, 962, 963, 1.0e6, 0, 1.0e-6
385.0, 375.0, 385.0, 335.0
/GRASSMANN_SPATIAL_SYMMETRIC_MECHANISM/731
961, 962, 963, 1.0e6, 0, 1.0e-6
385.0, 375.0, 385.0, 335.0
/GRASSMANN_SPATIAL_6R_MECHANISM/732
961, 962, 963, 1.0e6, 0, 1.0e-6
385.0, 375.0, 385.0, 335.0
/GRASSMANN_SPATIAL_OVERCONSTRAINED_MECHANISM/733
961, 962, 963, 1.0e6, 0, 1.0e-6
385.0, 375.0, 385.0, 335.0
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    for jid in range(727, 734):
        assert jid in model.lagmul_grassmann_spatial_linkage_joints


def test_m422_lagmul_grassmann_spatial_linkage_joint_missing_card(tmp_path: Path):
    deck = """# RADIOSS MISSING CARD DECK
/BEGIN
/LAGMUL/GRASSMANN_SPATIAL_LINKAGE_JOINT/734
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) > 0


def test_m422_sensor_spring_transverse_drop_rate_fixed(tmp_path: Path):
    c1 = f"{1808:>10d}{7.38e8:>20.4f}{0.2845:>20.4f}"
    deck = f"""# RADIOSS STARTER DECK
/BEGIN
Sensor Spring Transverse Drop Rate Fixed Format Test
2022 0
/SENSOR/SPRING_TRANSVERSE_DROP_RATE/1
Fixed Spring Transverse Drop Rate Sensor
{c1}
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 1 in model.sensor_spring_transverse_drop_rates
    s1 = model.sensor_spring_transverse_drop_rates[1]
    assert s1.spring_id == 1808
    assert pytest.approx(s1.jtrans_drop_max) == 7.38e8
    assert pytest.approx(s1.jtrans_lock_max) == 7.38e8
    assert pytest.approx(s1.jtrans_snp_max) == 7.38e8
    assert pytest.approx(s1.jtrans_crackle_max) == 7.38e8
    assert pytest.approx(s1.jtrans_shot_max) == 7.38e8
    assert pytest.approx(s1.jtrans_pop_max) == 7.38e8
    assert pytest.approx(s1.jtrans_crk_max) == 7.38e8
    assert pytest.approx(s1.t_delay) == 0.2845
    assert len(model.sensors) == 1
    assert model.sensors[0].kind == "SPRING_TRANSVERSE_DROP_RATE"


def test_m422_sensor_spring_transverse_drop_rate_free(tmp_path: Path):
    deck = """# RADIOSS FREE DECK
/BEGIN
Sensor Spring Transverse Drop Rate Free Format Test
/SENSOR/SPRING_TRANSVERSE_DROP_RATE/2
Free Spring Transverse Drop Rate Sensor
1809, 7.98e8, 0.3835
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 2 in model.sensor_spring_transverse_drop_rates
    s2 = model.sensor_spring_transverse_drop_rates[2]
    assert s2.spring_id == 1809
    assert pytest.approx(s2.jtrans_drop_max) == 7.98e8
    assert pytest.approx(s2.t_delay) == 0.3835


def test_m422_sensor_spring_transverse_drop_rate_aliases(tmp_path: Path):
    deck = """# RADIOSS ALIAS DECK
/BEGIN
Sensor Spring Transverse Drop Rate Aliases Test
/SENSOR/SPRING_TRANS_DROP_RATE/671
1829, 4.495e8, 0.3365
/SENSOR/SPRING_RATE_DROP_TRANS/672
1830, 4.515e8, 0.3375
/SENSOR/TRANSVERSE_DROP_RATE_SPRING/673
1831, 4.535e8, 0.3385
/SENSOR/SPRING_DROP_TRANS/674
1832, 4.555e8, 0.3395
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    for sid in range(671, 675):
        assert sid in model.sensor_spring_transverse_drop_rates


def test_m422_sensor_spring_transverse_drop_rate_missing_card(tmp_path: Path):
    deck = """# RADIOSS MISSING CARD DECK
/BEGIN
/SENSOR/SPRING_TRANSVERSE_DROP_RATE/675
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) > 0
