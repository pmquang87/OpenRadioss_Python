"""Tests for Milestone M433: LadDynamicCoreTearingRate Failure Model, EngElectrothermoflexomagnetoplasmonicexcitonicmagnonicpolaritonicResonanceEnergy, DifferentialSpatialLinkageJoint, and SensorSpringTorsionalSnapRate."""

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


def test_m433_fail_lad_dynamic_core_tearing_rate_fixed(tmp_path: Path):
    c1 = f"{980.0:>20.4f}{2940.0:>20.4f}{760.0:>20.4f}{15.50:>20.4f}{0.490:>20.4f}"
    c2 = f"{1:>10d}{2:>10d}"
    c3 = f"{2470:>10d}"
    deck = f"""# RADIOSS STARTER DECK
/BEGIN
Ladeveze Dynamic Core Tearing Rate Fixed Format Test
2022 0
/FAIL/LAD_DYNAMIC_CORE_TEARING_RATE/2470
Ladeveze Dynamic Core Tearing Failure Model
{c1}
{c2}
{c3}
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 2470 in model.fail_laddynamiccoretearingrates
    fdctr = model.fail_laddynamiccoretearingrates[2470]
    assert pytest.approx(fdctr.sigma_dctr0) == 980.0
    assert pytest.approx(fdctr.sigma_dctrc) == 2940.0
    assert pytest.approx(fdctr.gamma_dctr) == 760.0
    assert pytest.approx(fdctr.p_dctr) == 15.50
    assert pytest.approx(fdctr.d_dctr_max) == 0.490
    assert fdctr.ifail_sh == 1
    assert fdctr.ifail_so == 2
    assert fdctr.fail_id == 2470
    assert len(model.raw_fails) == 1
    assert model.raw_fails[0][1].type == "LAD_DYNAMIC_CORE_TEARING_RATE"


def test_m433_fail_lad_dynamic_core_tearing_rate_free(tmp_path: Path):
    deck = """# RADIOSS FREE DECK
/BEGIN
Ladeveze Dynamic Core Tearing Rate Free Format Test
/FAIL/LAD_DYNAMIC_CORE_TEARING_RATE/2471
990.0, 2970.0, 770.0, 15.65, 0.480
1, 1
2471
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 2471 in model.fail_laddynamiccoretearingrates
    fdctr = model.fail_laddynamiccoretearingrates[2471]
    assert pytest.approx(fdctr.sigma_dctr0) == 990.0
    assert pytest.approx(fdctr.sigma_dctrc) == 2970.0
    assert pytest.approx(fdctr.gamma_dctr) == 770.0
    assert pytest.approx(fdctr.p_dctr) == 15.65
    assert pytest.approx(fdctr.d_dctr_max) == 0.480
    assert fdctr.ifail_sh == 1
    assert fdctr.ifail_so == 1
    assert fdctr.fail_id == 2471


def test_m433_fail_lad_dynamic_core_tearing_rate_aliases(tmp_path: Path):
    aliases = [
        "/FAIL/LADEVEZE_DYNAMIC_CORE_TEARING_RATE",
        "/FAIL/LAD_DCTRR",
        "/FAIL/LAD_DCTRR_MODEL",
        "/FAIL/LAD_DCTRR_LAW",
        "/FAIL/LADEVEZE_RATE_DEPENDENT_DYNAMIC_CORE_TEARING",
    ]
    for i, kw in enumerate(aliases, start=2472):
        deck = f"""# RADIOSS ALIAS DECK
/BEGIN
Test {kw}
{kw}/{i}
985.0, 2955.0, 765.0, 15.55, 0.485
1, 2
{i}
/END
"""
        model, log = _parse_starter(tmp_path, deck)
        assert len(log.errors) == 0
        assert i in model.fail_laddynamiccoretearingrates
        fdctr = model.fail_laddynamiccoretearingrates[i]
        assert pytest.approx(fdctr.sigma_dctr0) == 985.0
        assert pytest.approx(fdctr.sigma_dctrc) == 2955.0


def test_m433_fail_lad_dynamic_core_tearing_rate_missing_card(tmp_path: Path):
    deck = """# RADIOSS MISSING CARD DECK
/BEGIN
/FAIL/LAD_DYNAMIC_CORE_TEARING_RATE/2479
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) > 0
    assert any("missing data card" in str(e) for e in log.errors)


def test_m433_eng_electrothermoflexomagnetoplasmonicexcitonicmagnonicpolaritonic_resonance_energy_fixed(tmp_path: Path):
    c1 = f"{0.00015:>20.8f}{10:>10d}"
    deck = f"""# RADIOSS STARTER DECK
/BEGIN
Electrothermoflexomagnetoplasmonicexcitonicmagnonicpolaritonic Resonance Energy Fixed Format Test
2022 0
/ENG/ELECTROTHERMOFLEXOMAGNETOPLASMONICEXCITONICMAGNONICPOLARITONIC_RESONANCE_ENERGY/1
Output Directive Title
{c1}
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 1 in model.eng_electrothermoflexomagnetoplasmonicexcitonicmagnonicpolaritonic_resonance_energies
    eng = model.eng_electrothermoflexomagnetoplasmonicexcitonicmagnonicpolaritonic_resonance_energies[1]
    assert pytest.approx(eng.dt_etfplexmmnp) == 0.00015
    assert eng.sens_id == 10
    assert eng.title == "Output Directive Title"


def test_m433_eng_electrothermoflexomagnetoplasmonicexcitonicmagnonicpolaritonic_resonance_energy_free(tmp_path: Path):
    deck = """# RADIOSS FREE DECK
/BEGIN
Electrothermoflexomagnetoplasmonicexcitonicmagnonicpolaritonic Resonance Energy Free Format Test
/ENG/ELECTROTHERMOFLEXOMAGNETOPLASMONICEXCITONICMAGNONICPOLARITONIC_RESONANCE_ENERGY/2
Output Directive Free
0.00025, 20
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 2 in model.eng_electrothermoflexomagnetoplasmonicexcitonicmagnonicpolaritonic_resonance_energies
    eng = model.eng_electrothermoflexomagnetoplasmonicexcitonicmagnonicpolaritonic_resonance_energies[2]
    assert pytest.approx(eng.dt_etfplexmmnp) == 0.00025
    assert eng.sens_id == 20


def test_m433_eng_electrothermoflexomagnetoplasmonicexcitonicmagnonicpolaritonic_resonance_energy_aliases(tmp_path: Path):
    deck = """# RADIOSS ALIAS DECK
/BEGIN
Engine Electrothermoflexomagnetoplasmonicexcitonicmagnonicpolaritonic Resonance Energy Aliases Test
/ENG/ELECTRO_THERM_FLEXO_MAG_PLASMON_EXCITON_MAGNONIC_POLARITON_RES_WORK/777
0.0935, 877
/ENG/EELECTROTHERMOFLEXOMAGNETOPLASMONICEXCITONICMAGNONICPOLARITONICRESONANCE/778
0.0945, 878
/ENG/ELECTROTHERMOFLEXOMAGNETOPLASMONICEXCITONICMAGNONICPOLARITONIC_RESONANCE_DISSIPATION/779
0.0955, 879
/ENG/ET_ELECTROTHERMOFLEXOMAGNETOPLASMONICEXCITONICMAGNONICPOLARITONIC_RESONANCE/780
0.0965, 880
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 777 in model.eng_electrothermoflexomagnetoplasmonicexcitonicmagnonicpolaritonic_resonance_energies
    assert 778 in model.eng_electrothermoflexomagnetoplasmonicexcitonicmagnonicpolaritonic_resonance_energies
    assert 779 in model.eng_electrothermoflexomagnetoplasmonicexcitonicmagnonicpolaritonic_resonance_energies
    assert 780 in model.eng_electrothermoflexomagnetoplasmonicexcitonicmagnonicpolaritonic_resonance_energies


def test_m433_eng_electrothermoflexomagnetoplasmonicexcitonicmagnonicpolaritonic_resonance_energy_missing_card(tmp_path: Path):
    deck = """# RADIOSS MISSING CARD DECK
/BEGIN
/ENG/ELECTROTHERMOFLEXOMAGNETOPLASMONICEXCITONICMAGNONICPOLARITONIC_RESONANCE_ENERGY/99
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) > 0
    assert any("missing data card" in str(e) for e in log.errors)


def test_m433_lagmul_differential_spatial_linkage_joint_fixed(tmp_path: Path):
    c1 = f"{101:>10d}{102:>10d}{103:>10d}{5.5e6:>20.4f}{5:>10d}{1e-7:>20.6e}"
    c2 = f"{45.5:>20.4f}{65.5:>20.4f}{30.0:>20.4f}{12.5:>20.4f}"
    deck = f"""# RADIOSS STARTER DECK
/BEGIN
Differential Spatial Linkage Joint Fixed Format Test
2022 0
/LAGMUL/DIFFERENTIAL_SPATIAL_LINKAGE_JOINT/10
Differential Joint 10
{c1}
{c2}
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 10 in model.lagmul_differential_spatial_linkage_joints
    jnt = model.lagmul_differential_spatial_linkage_joints[10]
    assert jnt.node1 == 101
    assert jnt.node2 == 102
    assert jnt.node3 == 103
    assert pytest.approx(jnt.stiff) == 5.5e6
    assert jnt.skew_id == 5
    assert pytest.approx(jnt.tol) == 1e-7
    assert pytest.approx(jnt.link_len_a) == 45.5
    assert pytest.approx(jnt.link_len_b) == 65.5
    assert pytest.approx(jnt.twist_angle_alpha) == 30.0
    assert pytest.approx(jnt.offset_distance_s) == 12.5
    assert pytest.approx(jnt.offset_distance_r) == 12.5
    assert pytest.approx(jnt.offset_distance_v) == 12.5
    assert pytest.approx(jnt.offset_distance_h) == 12.5
    assert pytest.approx(jnt.offset_distance_u) == 12.5
    assert pytest.approx(jnt.offset_distance_f) == 12.5


def test_m433_lagmul_differential_spatial_linkage_joint_free(tmp_path: Path):
    deck = """# RADIOSS FREE DECK
/BEGIN
Differential Spatial Linkage Joint Free Format Test
/DIFFERENTIAL_SPATIAL_LINKAGE_JOINT/11
Differential Joint 11
201, 202, 203, 6.5e6, 6, 2e-7
55.5, 75.5, 45.0, 15.5
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 11 in model.lagmul_differential_spatial_linkage_joints
    jnt = model.lagmul_differential_spatial_linkage_joints[11]
    assert jnt.node1 == 201
    assert jnt.node2 == 202
    assert jnt.node3 == 203
    assert pytest.approx(jnt.stiff) == 6.5e6
    assert jnt.skew_id == 6
    assert pytest.approx(jnt.tol) == 2e-7
    assert pytest.approx(jnt.link_len_a) == 55.5
    assert pytest.approx(jnt.link_len_b) == 75.5
    assert pytest.approx(jnt.twist_angle_alpha) == 45.0
    assert pytest.approx(jnt.offset_distance_s) == 15.5


def test_m433_lagmul_differential_spatial_linkage_joint_aliases(tmp_path: Path):
    aliases = [
        "/LAGMUL/DIFFERENTIAL_SPATIAL_LINKAGE",
        "/DIFFERENTIAL_SPATIAL_LINKAGE",
        "/DIFFERENTIAL_SPATIAL_MULTI_LOOP_MECHANISM",
        "/DIFFERENTIAL_SPATIAL_SYMMETRIC_MECHANISM",
        "/DIFFERENTIAL_SPATIAL_6R_MECHANISM",
        "/DIFFERENTIAL_SPATIAL_OVERCONSTRAINED_MECHANISM",
    ]
    for i, kw in enumerate(aliases, start=12):
        deck = f"""# RADIOSS ALIAS DECK
/BEGIN
Test {kw}
{kw}/{i}
Title {i}
301, 302, 303, 7.5e6, 7, 3e-7
60.0, 80.0, 50.0, 20.0
/END
"""
        model, log = _parse_starter(tmp_path, deck)
        assert len(log.errors) == 0
        assert i in model.lagmul_differential_spatial_linkage_joints
        jnt = model.lagmul_differential_spatial_linkage_joints[i]
        assert jnt.node1 == 301
        assert pytest.approx(jnt.stiff) == 7.5e6


def test_m433_lagmul_differential_spatial_linkage_joint_missing_card(tmp_path: Path):
    deck = """# RADIOSS MISSING CARD DECK
/BEGIN
/LAGMUL/DIFFERENTIAL_SPATIAL_LINKAGE_JOINT/99
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) > 0
    assert any("missing data card" in str(e) for e in log.errors)


def test_m433_sensor_spring_torsional_snap_rate_fixed(tmp_path: Path):
    c1 = f"{505:>10d}{8.8e7:>20.4f}{0.005:>20.6f}"
    deck = f"""# RADIOSS STARTER DECK
/BEGIN
Spring Torsional Snap Rate Fixed Format Test
2022 0
/SENSOR/SPRING_TORSIONAL_SNAP_RATE/1
Spring Torsional Snap Sensor 1
{c1}
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 1 in model.sensor_spring_torsional_snap_rates
    s = model.sensor_spring_torsional_snap_rates[1]
    assert s.spring_id == 505
    assert pytest.approx(s.jtors_snp_max) == 8.8e7
    assert pytest.approx(s.jtorsional_snp_max) == 8.8e7
    assert pytest.approx(s.t_delay) == 0.005
    assert pytest.approx(s.jtors_lock_max) == 8.8e7
    assert pytest.approx(s.jtors_crackle_max) == 8.8e7
    assert pytest.approx(s.jtors_shot_max) == 8.8e7
    assert pytest.approx(s.jtors_pop_max) == 8.8e7
    assert pytest.approx(s.jtors_crk_max) == 8.8e7
    assert pytest.approx(s.jtors_drop_rate_max) == 8.8e7
    assert pytest.approx(s.jtors_rate_max) == 8.8e7
    assert pytest.approx(s.jtors_roc_rate_max) == 8.8e7
    assert pytest.approx(s.jtors_snap_rate_max) == 8.8e7
    assert pytest.approx(s.jtors_snp_rate_max) == 8.8e7
    assert pytest.approx(s.jtang_snap_rate_max) == 8.8e7
    assert pytest.approx(s.jtang_snp_rate_max) == 8.8e7
    assert len(model.sensors) == 1
    assert model.sensors[0].kind == "SPRING_TORSIONAL_SNAP_RATE"


def test_m433_sensor_spring_torsional_snap_rate_free(tmp_path: Path):
    deck = """# RADIOSS FREE DECK
/BEGIN
Spring Torsional Snap Rate Free Format Test
/SENSOR/SPRING_TORSIONAL_SNAP_RATE/2
Spring Torsional Snap Sensor 2
606, 9.9e7, 0.006
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 2 in model.sensor_spring_torsional_snap_rates
    s = model.sensor_spring_torsional_snap_rates[2]
    assert s.spring_id == 606
    assert pytest.approx(s.jtors_snp_max) == 9.9e7
    assert pytest.approx(s.t_delay) == 0.006


def test_m433_sensor_spring_torsional_snap_rate_aliases(tmp_path: Path):
    aliases = [
        "/SENSOR/SPRING_TORS_SNAP_RATE",
        "/SENSOR/SPRING_SNAP_RATE_TORS",
        "/SENSOR/TORSIONAL_SNAP_RATE_SPRING",
        "/SENSOR/SPRING_SNAP_TORS",
    ]
    for i, kw in enumerate(aliases, start=3):
        deck = f"""# RADIOSS ALIAS DECK
/BEGIN
Test {kw}
{kw}/{i}
Title {i}
{700 + i}, 1.1e8, 0.007
/END
"""
        model, log = _parse_starter(tmp_path, deck)
        assert len(log.errors) == 0
        assert i in model.sensor_spring_torsional_snap_rates
        s = model.sensor_spring_torsional_snap_rates[i]
        assert s.spring_id == 700 + i
        assert pytest.approx(s.jtors_snp_max) == 1.1e8


def test_m433_sensor_spring_torsional_snap_rate_missing_card(tmp_path: Path):
    deck = """# RADIOSS MISSING CARD DECK
/BEGIN
/SENSOR/SPRING_TORSIONAL_SNAP_RATE/99
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) > 0
    assert any("missing data card" in str(e) for e in log.errors)


def test_m433_integrated_suite(tmp_path: Path):
    deck = """# RADIOSS INTEGRATED M433 DECK
/BEGIN
M433 Integrated Suite
/FAIL/LAD_DYNAMIC_CORE_TEARING_RATE/100
Dynamic Core Tearing
980.0, 2940.0, 760.0, 15.50, 0.490
1, 1
100
/ENG/ELECTROTHERMOFLEXOMAGNETOPLASMONICEXCITONICMAGNONICPOLARITONIC_RESONANCE_ENERGY/1
MultiField Resonance
0.0001, 5
/LAGMUL/DIFFERENTIAL_SPATIAL_LINKAGE_JOINT/1
Differential Mechanism
1, 2, 3, 1.0e7, 0, 1.0e-6
50.0, 70.0, 35.0, 10.0
/SENSOR/SPRING_TORSIONAL_SNAP_RATE/1
Torsional Snap Sensor
10, 5.0e7, 0.001
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 100 in model.fail_laddynamiccoretearingrates
    assert 1 in model.eng_electrothermoflexomagnetoplasmonicexcitonicmagnonicpolaritonic_resonance_energies
    assert 1 in model.lagmul_differential_spatial_linkage_joints
    assert 1 in model.sensor_spring_torsional_snap_rates
