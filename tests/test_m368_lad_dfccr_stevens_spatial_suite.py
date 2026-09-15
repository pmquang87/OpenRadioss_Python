"""Tests for Milestone M368: LadDynamicFiberCompressionCrushingRate Failure Model, EngFlexomagnetoplasmonicpolaritonicResonanceEnergy, StevensSpatialLinkageJoint, and SensorSpringTransverseSnapRate."""

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


def test_m368_fail_lad_dynamic_fiber_compression_crushing_rate_fixed(tmp_path: Path):
    c1 = f"{215.0:>20.4f}{645.0:>20.4f}{84.0:>20.4f}{3.15:>20.4f}{0.962:>20.4f}"
    c2 = f"{1:>10d}{2:>10d}"
    c3 = f"{1830:>10d}"
    deck = f"""# RADIOSS STARTER DECK
/BEGIN
Ladeveze Dynamic Fiber Compression Crushing Rate Fixed Format Test
2022 0
/FAIL/LAD_DYNAMIC_FIBER_COMPRESSION_CRUSHING_RATE/1830
Ladeveze Dynamic Fiber Compression Crushing Failure Model
{c1}
{c2}
{c3}
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 1830 in model.fail_laddynamicfibercompressioncrushingrates
    fdfccr = model.fail_laddynamicfibercompressioncrushingrates[1830]
    assert pytest.approx(fdfccr.sigma_dfccr0) == 215.0
    assert pytest.approx(fdfccr.sigma_dfccrc) == 645.0
    assert pytest.approx(fdfccr.gamma_dfccr) == 84.0
    assert pytest.approx(fdfccr.p_dfccr) == 3.15
    assert pytest.approx(fdfccr.d_dfccr_max) == 0.962
    assert fdfccr.ifail_sh == 1
    assert fdfccr.ifail_so == 2
    assert fdfccr.fail_id == 1830
    assert len(model.raw_fails) == 1
    assert model.raw_fails[0][1].type == "LAD_DYNAMIC_FIBER_COMPRESSION_CRUSHING_RATE"


def test_m368_fail_lad_dynamic_fiber_compression_crushing_rate_free(tmp_path: Path):
    deck = """# RADIOSS FREE DECK
/BEGIN
Ladeveze Dynamic Fiber Compression Crushing Rate Free Format Test
/FAIL/LAD_DYNAMIC_FIBER_COMPRESSION_CRUSHING_RATE/1831
225.0, 675.0, 92.0, 3.35, 0.942
1, 1
1831
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 1831 in model.fail_laddynamicfibercompressioncrushingrates
    fdfccr = model.fail_laddynamicfibercompressioncrushingrates[1831]
    assert pytest.approx(fdfccr.sigma_dfccr0) == 225.0
    assert pytest.approx(fdfccr.sigma_dfccrc) == 675.0
    assert pytest.approx(fdfccr.gamma_dfccr) == 92.0
    assert pytest.approx(fdfccr.p_dfccr) == 3.35
    assert pytest.approx(fdfccr.d_dfccr_max) == 0.942
    assert fdfccr.fail_id == 1831


def test_m368_fail_lad_dynamic_fiber_compression_crushing_rate_aliases(tmp_path: Path):
    deck = """# RADIOSS ALIAS DECK
/BEGIN
Ladeveze Dynamic Fiber Compression Crushing Rate Aliases Test
/FAIL/LADEVEZE_DYNAMIC_FIBER_COMPRESSION_CRUSHING_RATE/1832
205.0, 615.0, 76.0, 2.65, 0.985
1, 1
/FAIL/LAD_DFCCR/1833
205.0, 615.0, 76.0, 2.65, 0.985
1, 1
/FAIL/LAD_DFCCR_MODEL/1834
205.0, 615.0, 76.0, 2.65, 0.985
1, 1
/FAIL/LAD_DFCCR_LAW/1835
205.0, 615.0, 76.0, 2.65, 0.985
1, 1
/FAIL/LADEVEZE_RATE_DEPENDENT_DYNAMIC_FIBER_COMPRESSION_CRUSHING/1836
205.0, 615.0, 76.0, 2.65, 0.985
1, 1
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 1832 in model.fail_laddynamicfibercompressioncrushingrates
    assert 1833 in model.fail_laddynamicfibercompressioncrushingrates
    assert 1834 in model.fail_laddynamicfibercompressioncrushingrates
    assert 1835 in model.fail_laddynamicfibercompressioncrushingrates
    assert 1836 in model.fail_laddynamicfibercompressioncrushingrates
    assert len(model.raw_fails) == 5


def test_m368_fail_lad_dynamic_fiber_compression_crushing_rate_missing_card(tmp_path: Path):
    deck = """# RADIOSS MISSING CARD DECK
/BEGIN
/FAIL/LAD_DYNAMIC_FIBER_COMPRESSION_CRUSHING_RATE/1837
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) > 0


def test_m368_eng_flexomagnetoplasmonicpolaritonic_resonance_energy_fixed(tmp_path: Path):
    c1 = f"{0.0038:>20.4f}{125:>10d}"
    deck = f"""# RADIOSS STARTER DECK
/BEGIN
Engine Flexomagnetoplasmonicpolaritonic Resonance Energy Fixed Format Test
2022 0
/ENG/FLEXOMAGNETOPLASMONICPOLARITONIC_RESONANCE_ENERGY/25
Resonance Energy Output Directive
{c1}
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 25 in model.eng_flexomagnetoplasmonicpolaritonic_resonance_energies
    eng = model.eng_flexomagnetoplasmonicpolaritonic_resonance_energies[25]
    assert pytest.approx(eng.dt_fmppor) == 0.0038
    assert eng.sens_id == 125


def test_m368_eng_flexomagnetoplasmonicpolaritonic_resonance_energy_free(tmp_path: Path):
    deck = """# RADIOSS FREE DECK
/BEGIN
Engine Flexomagnetoplasmonicpolaritonic Resonance Energy Free Format Test
/ENG/FLEXOMAGNETOPLASMONICPOLARITONIC_RESONANCE_ENERGY/26
0.0048, 126
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 26 in model.eng_flexomagnetoplasmonicpolaritonic_resonance_energies
    eng = model.eng_flexomagnetoplasmonicpolaritonic_resonance_energies[26]
    assert pytest.approx(eng.dt_fmppor) == 0.0048
    assert eng.sens_id == 126


def test_m368_eng_flexomagnetoplasmonicpolaritonic_resonance_energy_aliases(tmp_path: Path):
    deck = """# RADIOSS ALIAS DECK
/BEGIN
Engine Flexomagnetoplasmonicpolaritonic Resonance Energy Aliases Test
/ENG/FLEXOMAGNETO_PLASMON_POLARITON_RES_WORK/27
0.0058, 127
/ENG/EFLEXOMAGNETOPLASMONICPOLARITONICRESONANCE/28
0.0068, 128
/ENG/FLEXOMAGNETOPLASMONICPOLARITONIC_RESONANCE_DISSIPATION/29
0.0078, 129
/ENG/EM_FLEXOMAGNETOPLASMONICPOLARITONIC_RESONANCE/30
0.0088, 130
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 27 in model.eng_flexomagnetoplasmonicpolaritonic_resonance_energies
    assert 28 in model.eng_flexomagnetoplasmonicpolaritonic_resonance_energies
    assert 29 in model.eng_flexomagnetoplasmonicpolaritonic_resonance_energies
    assert 30 in model.eng_flexomagnetoplasmonicpolaritonic_resonance_energies


def test_m368_eng_flexomagnetoplasmonicpolaritonic_resonance_energy_missing_card(tmp_path: Path):
    deck = """# RADIOSS MISSING CARD DECK
/BEGIN
/ENG/FLEXOMAGNETOPLASMONICPOLARITONIC_RESONANCE_ENERGY/31
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) > 0


def test_m368_lagmul_stevens_spatial_linkage_joint_fixed(tmp_path: Path):
    c1 = f"{111:>10d}{112:>10d}{113:>10d}{5.5e6:>20.4f}{4:>10d}{1.5e-5:>20.4e}"
    c2 = f"{54.0:>20.4f}{38.0:>20.4f}{48.0:>20.4f}{14.0:>20.4f}"
    deck = f"""# RADIOSS STARTER DECK
/BEGIN
Stevens Spatial Linkage Joint Fixed Format Test
2022 0
/LAGMUL/STEVENS_SPATIAL_LINKAGE_JOINT/75
Stevens Spatial 6R Multi-Loop Joint
{c1}
{c2}
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 75 in model.lagmul_stevens_spatial_linkage_joints
    joint = model.lagmul_stevens_spatial_linkage_joints[75]
    assert joint.node1 == 111
    assert joint.node2 == 112
    assert joint.node3 == 113
    assert pytest.approx(joint.stiff) == 5.5e6
    assert joint.skew_id == 4
    assert pytest.approx(joint.tol) == 1.5e-5
    assert pytest.approx(joint.link_len_a) == 54.0
    assert pytest.approx(joint.link_len_b) == 38.0
    assert pytest.approx(joint.twist_angle_alpha) == 48.0
    assert pytest.approx(joint.offset_distance_s) == 14.0


def test_m368_lagmul_stevens_spatial_linkage_joint_free(tmp_path: Path):
    deck = """# RADIOSS FREE DECK
/BEGIN
Stevens Spatial Linkage Joint Free Format Test
/LAGMUL/STEVENS_SPATIAL_LINKAGE_JOINT/76
211, 212, 213, 7.0e6, 6, 2.5e-5
58.0, 44.0, 65.0, 18.0
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 76 in model.lagmul_stevens_spatial_linkage_joints
    joint = model.lagmul_stevens_spatial_linkage_joints[76]
    assert joint.node1 == 211
    assert joint.node2 == 212
    assert joint.node3 == 213
    assert pytest.approx(joint.stiff) == 7.0e6
    assert joint.skew_id == 6
    assert pytest.approx(joint.tol) == 2.5e-5
    assert pytest.approx(joint.link_len_a) == 58.0
    assert pytest.approx(joint.link_len_b) == 44.0
    assert pytest.approx(joint.twist_angle_alpha) == 65.0
    assert pytest.approx(joint.offset_distance_s) == 18.0


def test_m368_lagmul_stevens_spatial_linkage_joint_aliases(tmp_path: Path):
    deck = """# RADIOSS ALIAS DECK
/BEGIN
Stevens Spatial Linkage Joint Aliases Test
/STEVENS_SPATIAL_LINKAGE_JOINT/77
311, 312, 313, 1.0e6, 0, 1.0e-6
32.0, 22.0, 32.0, 6.0
/LAGMUL/STEVENS_SPATIAL_LINKAGE/78
311, 312, 313, 1.0e6, 0, 1.0e-6
32.0, 22.0, 32.0, 6.0
/STEVENS_SPATIAL_LINKAGE/79
311, 312, 313, 1.0e6, 0, 1.0e-6
32.0, 22.0, 32.0, 6.0
/STEVENS_SPATIAL_MULTI_LOOP_MECHANISM/80
311, 312, 313, 1.0e6, 0, 1.0e-6
32.0, 22.0, 32.0, 6.0
/STEVENS_SPATIAL_SYMMETRIC_MECHANISM/81
311, 312, 313, 1.0e6, 0, 1.0e-6
32.0, 22.0, 32.0, 6.0
/STEVENS_SPATIAL_6R_MECHANISM/82
311, 312, 313, 1.0e6, 0, 1.0e-6
32.0, 22.0, 32.0, 6.0
/STEVENS_SPATIAL_OVERCONSTRAINED_MECHANISM/83
311, 312, 313, 1.0e6, 0, 1.0e-6
32.0, 22.0, 32.0, 6.0
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    for jid in range(77, 84):
        assert jid in model.lagmul_stevens_spatial_linkage_joints


def test_m368_lagmul_stevens_spatial_linkage_joint_missing_card(tmp_path: Path):
    deck = """# RADIOSS MISSING CARD DECK
/BEGIN
/LAGMUL/STEVENS_SPATIAL_LINKAGE_JOINT/84
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) > 0


def test_m368_sensor_spring_transverse_snap_rate_fixed(tmp_path: Path):
    c1 = f"{1158:>10d}{9.6e7:>20.4f}{0.0285:>20.4f}"
    deck = f"""# RADIOSS STARTER DECK
/BEGIN
Sensor Spring Transverse Snap Rate Fixed Format Test
2022 0
/SENSOR/SPRING_TRANSVERSE_SNAP_RATE/1
Fixed Spring Transverse Snap Rate Sensor
{c1}
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 1 in model.sensor_spring_transverse_snap_rates
    s1 = model.sensor_spring_transverse_snap_rates[1]
    assert s1.spring_id == 1158
    assert pytest.approx(s1.jtrans_snp_max) == 9.6e7
    assert pytest.approx(s1.t_delay) == 0.0285
    assert len(model.sensors) == 1
    assert model.sensors[0].kind == "SPRING_TRANSVERSE_SNAP_RATE"


def test_m368_sensor_spring_transverse_snap_rate_free(tmp_path: Path):
    deck = """# RADIOSS FREE DECK
/BEGIN
Sensor Spring Transverse Snap Rate Free Format Test
/SENSOR/SPRING_TRANSVERSE_SNAP_RATE/2
Free Spring Transverse Snap Rate Sensor
1159, 1.35e8, 0.0415
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 2 in model.sensor_spring_transverse_snap_rates
    s2 = model.sensor_spring_transverse_snap_rates[2]
    assert s2.spring_id == 1159
    assert pytest.approx(s2.jtrans_snp_max) == 1.35e8
    assert pytest.approx(s2.t_delay) == 0.0415


def test_m368_sensor_spring_transverse_snap_rate_aliases(tmp_path: Path):
    deck = """# RADIOSS ALIAS DECK
/BEGIN
Sensor Spring Transverse Snap Rate Aliases Test
/SENSOR/SPRING_TRANS_SNAP_RATE/180
1191, 6.55e7, 0.0085
/SENSOR/SPRING_RATE_SNAP_TRANS/181
1192, 6.75e7, 0.0095
/SENSOR/TRANSVERSE_SNAP_RATE_SPRING/182
1193, 6.95e7, 0.0105
/SENSOR/SPRING_SNAP_TRANS/183
1194, 7.15e7, 0.0115
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    for sid in range(180, 184):
        assert sid in model.sensor_spring_transverse_snap_rates


def test_m368_sensor_spring_transverse_snap_rate_missing_card(tmp_path: Path):
    deck = """# RADIOSS MISSING CARD DECK
/BEGIN
/SENSOR/SPRING_TRANSVERSE_SNAP_RATE/185
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) > 0
