"""Tests for Milestone M432: LadCoupleCoreDelaminationRate Failure Model, EngElectrothermoflexomagnetophononicmagnonicpolaritonicResonanceEnergy, AlgebraicSpatialLinkageJoint, and SensorSpringTotalSnapRate."""

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


def test_m432_fail_lad_couple_core_delamination_rate_fixed(tmp_path: Path):
    c1 = f"{965.0:>20.4f}{2895.0:>20.4f}{745.0:>20.4f}{15.25:>20.4f}{0.485:>20.4f}"
    c2 = f"{1:>10d}{2:>10d}"
    c3 = f"{2460:>10d}"
    deck = f"""# RADIOSS STARTER DECK
/BEGIN
Ladeveze Coupled Core Delamination Rate Fixed Format Test
2022 0
/FAIL/LAD_COUPLE_CORE_DELAMINATION_RATE/2460
Ladeveze Coupled Core Delamination Failure Model
{c1}
{c2}
{c3}
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 2460 in model.fail_ladcouplecoredelaminationrates
    fccdl = model.fail_ladcouplecoredelaminationrates[2460]
    assert pytest.approx(fccdl.sigma_ccdl0) == 965.0
    assert pytest.approx(fccdl.sigma_ccdlc) == 2895.0
    assert pytest.approx(fccdl.gamma_ccdl) == 745.0
    assert pytest.approx(fccdl.p_ccdl) == 15.25
    assert pytest.approx(fccdl.d_ccdl_max) == 0.485
    assert fccdl.ifail_sh == 1
    assert fccdl.ifail_so == 2
    assert fccdl.fail_id == 2460
    assert len(model.raw_fails) == 1
    assert model.raw_fails[0][1].type == "LAD_COUPLE_CORE_DELAMINATION_RATE"


def test_m432_fail_lad_couple_core_delamination_rate_free(tmp_path: Path):
    deck = """# RADIOSS FREE DECK
/BEGIN
Ladeveze Coupled Core Delamination Rate Free Format Test
/FAIL/LAD_COUPLE_CORE_DELAMINATION_RATE/2461
975.0, 2925.0, 755.0, 15.45, 0.475
1, 1
2461
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 2461 in model.fail_ladcouplecoredelaminationrates
    fccdl = model.fail_ladcouplecoredelaminationrates[2461]
    assert pytest.approx(fccdl.sigma_ccdl0) == 975.0
    assert pytest.approx(fccdl.sigma_ccdlc) == 2925.0
    assert pytest.approx(fccdl.gamma_ccdl) == 755.0
    assert pytest.approx(fccdl.p_ccdl) == 15.45
    assert pytest.approx(fccdl.d_ccdl_max) == 0.475
    assert fccdl.fail_id == 2461


def test_m432_fail_lad_couple_core_delamination_rate_aliases(tmp_path: Path):
    deck = """# RADIOSS ALIAS DECK
/BEGIN
Ladeveze Coupled Core Delamination Rate Aliases Test
/FAIL/LADEVEZE_COUPLED_CORE_DELAMINATION_RATE/2462
955.0, 2865.0, 735.0, 15.05, 0.495
1, 1
/FAIL/LAD_CCDLR/2463
955.0, 2865.0, 735.0, 15.05, 0.495
1, 1
/FAIL/LAD_CCDLR_MODEL/2464
955.0, 2865.0, 735.0, 15.05, 0.495
1, 1
/FAIL/LAD_CCDLR_LAW/2465
955.0, 2865.0, 735.0, 15.05, 0.495
1, 1
/FAIL/LADEVEZE_RATE_DEPENDENT_COUPLED_CORE_DELAMINATION/2466
955.0, 2865.0, 735.0, 15.05, 0.495
1, 1
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 2462 in model.fail_ladcouplecoredelaminationrates
    assert 2463 in model.fail_ladcouplecoredelaminationrates
    assert 2464 in model.fail_ladcouplecoredelaminationrates
    assert 2465 in model.fail_ladcouplecoredelaminationrates
    assert 2466 in model.fail_ladcouplecoredelaminationrates
    assert len(model.raw_fails) == 5


def test_m432_fail_lad_couple_core_delamination_rate_missing_card(tmp_path: Path):
    deck = """# RADIOSS MISSING CARD DECK
/BEGIN
/FAIL/LAD_COUPLE_CORE_DELAMINATION_RATE/2467
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) > 0


def test_m432_eng_electrothermoflexomagnetophononicmagnonicpolaritonic_resonance_energy_fixed(tmp_path: Path):
    c1 = f"{0.0915:>20.4f}{875:>10d}"
    deck = f"""# RADIOSS STARTER DECK
/BEGIN
Engine Electrothermoflexomagnetophononicmagnonicpolaritonic Resonance Energy Fixed Format Test
2022 0
/ENG/ELECTROTHERMOFLEXOMAGNETOPHONONICMAGNONICPOLARITONIC_RESONANCE_ENERGY/775
Resonance Energy Output Directive
{c1}
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 775 in model.eng_electrothermoflexomagnetophononicmagnonicpolaritonic_resonance_energies
    eng = model.eng_electrothermoflexomagnetophononicmagnonicpolaritonic_resonance_energies[775]
    assert pytest.approx(eng.dt_etfphmmnp) == 0.0915
    assert eng.sens_id == 875


def test_m432_eng_electrothermoflexomagnetophononicmagnonicpolaritonic_resonance_energy_free(tmp_path: Path):
    deck = """# RADIOSS FREE DECK
/BEGIN
Engine Electrothermoflexomagnetophononicmagnonicpolaritonic Resonance Energy Free Format Test
/ENG/ELECTROTHERMOFLEXOMAGNETOPHONONICMAGNONICPOLARITONIC_RESONANCE_ENERGY/776
0.0925, 876
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 776 in model.eng_electrothermoflexomagnetophononicmagnonicpolaritonic_resonance_energies
    eng = model.eng_electrothermoflexomagnetophononicmagnonicpolaritonic_resonance_energies[776]
    assert pytest.approx(eng.dt_etfphmmnp) == 0.0925
    assert eng.sens_id == 876


def test_m432_eng_electrothermoflexomagnetophononicmagnonicpolaritonic_resonance_energy_aliases(tmp_path: Path):
    deck = """# RADIOSS ALIAS DECK
/BEGIN
Engine Electrothermoflexomagnetophononicmagnonicpolaritonic Resonance Energy Aliases Test
/ENG/ELECTRO_THERM_FLEXO_MAG_PHONON_MAGNON_POLARITON_RES_WORK/777
0.0935, 877
/ENG/EELECTROTHERMOFLEXOMAGNETOPHONONICMAGNONICPOLARITONICRESONANCE/778
0.0945, 878
/ENG/ELECTROTHERMOFLEXOMAGNETOPHONONICMAGNONICPOLARITONIC_RESONANCE_DISSIPATION/779
0.0955, 879
/ENG/ET_ELECTROTHERMOFLEXOMAGNETOPHONONICMAGNONICPOLARITONIC_RESONANCE/780
0.0965, 880
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 777 in model.eng_electrothermoflexomagnetophononicmagnonicpolaritonic_resonance_energies
    assert 778 in model.eng_electrothermoflexomagnetophononicmagnonicpolaritonic_resonance_energies
    assert 779 in model.eng_electrothermoflexomagnetophononicmagnonicpolaritonic_resonance_energies
    assert 780 in model.eng_electrothermoflexomagnetophononicmagnonicpolaritonic_resonance_energies


def test_m432_eng_electrothermoflexomagnetophononicmagnonicpolaritonic_resonance_energy_missing_card(tmp_path: Path):
    deck = """# RADIOSS MISSING CARD DECK
/BEGIN
/ENG/ELECTROTHERMOFLEXOMAGNETOPHONONICMAGNONICPOLARITONIC_RESONANCE_ENERGY/781
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) > 0


def test_m432_lagmul_algebraic_spatial_linkage_joint_fixed(tmp_path: Path):
    c1 = f"{861:>10d}{862:>10d}{863:>10d}{7.45e7:>20.4f}{465:>10d}{6.55e-4:>20.4e}"
    c2 = f"{525.0:>20.4f}{505.0:>20.4f}{535.0:>20.4f}{455.0:>20.4f}"
    deck = f"""# RADIOSS STARTER DECK
/BEGIN
Algebraic Spatial Linkage Joint Fixed Format Test
2022 0
/LAGMUL/ALGEBRAIC_SPATIAL_LINKAGE_JOINT/825
Algebraic Spatial 6R Joint
{c1}
{c2}
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 825 in model.lagmul_algebraic_spatial_linkage_joints
    joint = model.lagmul_algebraic_spatial_linkage_joints[825]
    assert joint.node1 == 861
    assert joint.node2 == 862
    assert joint.node3 == 863
    assert pytest.approx(joint.stiff) == 7.45e7
    assert joint.skew_id == 465
    assert pytest.approx(joint.tol) == 6.55e-4
    assert pytest.approx(joint.link_len_a) == 525.0
    assert pytest.approx(joint.link_len_b) == 505.0
    assert pytest.approx(joint.twist_angle_alpha) == 535.0
    assert pytest.approx(joint.offset_distance_s) == 455.0
    assert pytest.approx(joint.offset_distance_r) == 455.0
    assert pytest.approx(joint.offset_distance_v) == 455.0
    assert pytest.approx(joint.offset_distance_h) == 455.0
    assert pytest.approx(joint.offset_distance_u) == 455.0
    assert pytest.approx(joint.offset_distance_f) == 455.0


def test_m432_lagmul_algebraic_spatial_linkage_joint_free(tmp_path: Path):
    deck = """# RADIOSS FREE DECK
/BEGIN
Algebraic Spatial Linkage Joint Free Format Test
/LAGMUL/ALGEBRAIC_SPATIAL_LINKAGE_JOINT/826
961, 962, 963, 7.65e7, 466, 6.65e-4
545.0, 515.0, 555.0, 470.0
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 826 in model.lagmul_algebraic_spatial_linkage_joints
    joint = model.lagmul_algebraic_spatial_linkage_joints[826]
    assert joint.node1 == 961
    assert joint.node2 == 962
    assert joint.node3 == 963
    assert pytest.approx(joint.stiff) == 7.65e7
    assert joint.skew_id == 466
    assert pytest.approx(joint.tol) == 6.65e-4
    assert pytest.approx(joint.link_len_a) == 545.0
    assert pytest.approx(joint.link_len_b) == 515.0
    assert pytest.approx(joint.twist_angle_alpha) == 555.0
    assert pytest.approx(joint.offset_distance_s) == 470.0


def test_m432_lagmul_algebraic_spatial_linkage_joint_aliases(tmp_path: Path):
    deck = """# RADIOSS ALIAS DECK
/BEGIN
Algebraic Spatial Linkage Joint Aliases Test
/ALGEBRAIC_SPATIAL_LINKAGE_JOINT/827
1061, 1062, 1063, 1.0e6, 0, 1.0e-6
485.0, 475.0, 485.0, 435.0
/LAGMUL/ALGEBRAIC_SPATIAL_LINKAGE/828
1061, 1062, 1063, 1.0e6, 0, 1.0e-6
485.0, 475.0, 485.0, 435.0
/ALGEBRAIC_SPATIAL_LINKAGE/829
1061, 1062, 1063, 1.0e6, 0, 1.0e-6
485.0, 475.0, 485.0, 435.0
/ALGEBRAIC_SPATIAL_MULTI_LOOP_MECHANISM/830
1061, 1062, 1063, 1.0e6, 0, 1.0e-6
485.0, 475.0, 485.0, 435.0
/ALGEBRAIC_SPATIAL_SYMMETRIC_MECHANISM/831
1061, 1062, 1063, 1.0e6, 0, 1.0e-6
485.0, 475.0, 485.0, 435.0
/ALGEBRAIC_SPATIAL_6R_MECHANISM/832
1061, 1062, 1063, 1.0e6, 0, 1.0e-6
485.0, 475.0, 485.0, 435.0
/ALGEBRAIC_SPATIAL_OVERCONSTRAINED_MECHANISM/833
1061, 1062, 1063, 1.0e6, 0, 1.0e-6
485.0, 475.0, 485.0, 435.0
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 827 in model.lagmul_algebraic_spatial_linkage_joints
    assert 828 in model.lagmul_algebraic_spatial_linkage_joints
    assert 829 in model.lagmul_algebraic_spatial_linkage_joints
    assert 830 in model.lagmul_algebraic_spatial_linkage_joints
    assert 831 in model.lagmul_algebraic_spatial_linkage_joints
    assert 832 in model.lagmul_algebraic_spatial_linkage_joints
    assert 833 in model.lagmul_algebraic_spatial_linkage_joints


def test_m432_lagmul_algebraic_spatial_linkage_joint_missing_card(tmp_path: Path):
    deck = """# RADIOSS MISSING CARD DECK
/BEGIN
/LAGMUL/ALGEBRAIC_SPATIAL_LINKAGE_JOINT/834
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) > 0


def test_m432_sensor_spring_total_snap_rate_fixed(tmp_path: Path):
    c1 = f"{1155:>10d}{8.95e6:>20.4f}{0.0065:>20.4f}"
    deck = f"""# RADIOSS STARTER DECK
/BEGIN
Sensor Spring Total Snap Rate Fixed Format Test
2022 0
/SENSOR/SPRING_TOTAL_SNAP_RATE/955
Spring Total Snap Rate Sensor
{c1}
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 955 in model.sensor_spring_total_snap_rates
    sensor = model.sensor_spring_total_snap_rates[955]
    assert sensor.spring_id == 1155
    assert pytest.approx(sensor.jtot_snp_max) == 8.95e6
    assert pytest.approx(sensor.jtot_lock_max) == 8.95e6
    assert pytest.approx(sensor.jtot_crackle_max) == 8.95e6
    assert pytest.approx(sensor.jtot_shot_max) == 8.95e6
    assert pytest.approx(sensor.jtot_pop_max) == 8.95e6
    assert pytest.approx(sensor.jtot_crk_max) == 8.95e6
    assert pytest.approx(sensor.jtot_drop_rate_max) == 8.95e6
    assert pytest.approx(sensor.jtot_rate_max) == 8.95e6
    assert pytest.approx(sensor.jtot_roc_rate_max) == 8.95e6
    assert pytest.approx(sensor.jtot_snap_rate_max) == 8.95e6
    assert pytest.approx(sensor.t_delay) == 0.0065
    assert len(model.sensors) == 1
    assert model.sensors[0].kind == "SPRING_TOTAL_SNAP_RATE"


def test_m432_sensor_spring_total_snap_rate_free(tmp_path: Path):
    deck = """# RADIOSS FREE DECK
/BEGIN
Sensor Spring Total Snap Rate Free Format Test
/SENSOR/SPRING_TOTAL_SNAP_RATE/956
1156, 9.05e6, 0.0075
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 956 in model.sensor_spring_total_snap_rates
    sensor = model.sensor_spring_total_snap_rates[956]
    assert sensor.spring_id == 1156
    assert pytest.approx(sensor.jtot_snp_max) == 9.05e6
    assert pytest.approx(sensor.t_delay) == 0.0075


def test_m432_sensor_spring_total_snap_rate_aliases(tmp_path: Path):
    deck = """# RADIOSS ALIAS DECK
/BEGIN
Sensor Spring Total Snap Rate Aliases Test
/SENSOR/SPRING_TOT_SNAP_RATE/957
1157, 8.85e6, 0.0055
/SENSOR/SPRING_SNAP_RATE_TOT/958
1158, 8.85e6, 0.0055
/SENSOR/TOTAL_SNAP_RATE_SPRING/959
1159, 8.85e6, 0.0055
/SENSOR/SPRING_SNAP_TOT/960
1160, 8.85e6, 0.0055
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 957 in model.sensor_spring_total_snap_rates
    assert 958 in model.sensor_spring_total_snap_rates
    assert 959 in model.sensor_spring_total_snap_rates
    assert 960 in model.sensor_spring_total_snap_rates
    assert len(model.sensors) == 4


def test_m432_sensor_spring_total_snap_rate_missing_card(tmp_path: Path):
    deck = """# RADIOSS MISSING CARD DECK
/BEGIN
/SENSOR/SPRING_TOTAL_SNAP_RATE/961
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) > 0
