"""Tests for Milestone M370: LadCoupleFiberCompressionCrushingRate Failure Model, EngFlexomagnetophononicmagnonicResonanceEnergy, WaldronHybridSpatialLinkageJoint, and SensorSpringTorsionalSnapRate."""

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


def test_m370_fail_lad_couple_fiber_compression_crushing_rate_fixed(tmp_path: Path):
    c1 = f"{225.0:>20.4f}{675.0:>20.4f}{92.0:>20.4f}{3.25:>20.4f}{0.958:>20.4f}"
    c2 = f"{1:>10d}{2:>10d}"
    c3 = f"{1840:>10d}"
    deck = f"""# RADIOSS STARTER DECK
/BEGIN
Ladeveze Coupled Fiber Compression Crushing Rate Fixed Format Test
2022 0
/FAIL/LAD_COUPLE_FIBER_COMPRESSION_CRUSHING_RATE/1840
Ladeveze Coupled Fiber Compression Crushing Failure Model
{c1}
{c2}
{c3}
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 1840 in model.fail_ladcouplefibercompressioncrushingrates
    fcfccr = model.fail_ladcouplefibercompressioncrushingrates[1840]
    assert pytest.approx(fcfccr.sigma_cfccr0) == 225.0
    assert pytest.approx(fcfccr.sigma_cfccrc) == 675.0
    assert pytest.approx(fcfccr.gamma_cfccr) == 92.0
    assert pytest.approx(fcfccr.p_cfccr) == 3.25
    assert pytest.approx(fcfccr.d_cfccr_max) == 0.958
    assert fcfccr.ifail_sh == 1
    assert fcfccr.ifail_so == 2
    assert fcfccr.fail_id == 1840
    assert len(model.raw_fails) == 1
    assert model.raw_fails[0][1].type == "LAD_COUPLE_FIBER_COMPRESSION_CRUSHING_RATE"


def test_m370_fail_lad_couple_fiber_compression_crushing_rate_free(tmp_path: Path):
    deck = """# RADIOSS FREE DECK
/BEGIN
Ladeveze Coupled Fiber Compression Crushing Rate Free Format Test
/FAIL/LAD_COUPLE_FIBER_COMPRESSION_CRUSHING_RATE/1841
235.0, 705.0, 98.0, 3.45, 0.938
1, 1
1841
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 1841 in model.fail_ladcouplefibercompressioncrushingrates
    fcfccr = model.fail_ladcouplefibercompressioncrushingrates[1841]
    assert pytest.approx(fcfccr.sigma_cfccr0) == 235.0
    assert pytest.approx(fcfccr.sigma_cfccrc) == 705.0
    assert pytest.approx(fcfccr.gamma_cfccr) == 98.0
    assert pytest.approx(fcfccr.p_cfccr) == 3.45
    assert pytest.approx(fcfccr.d_cfccr_max) == 0.938
    assert fcfccr.fail_id == 1841


def test_m370_fail_lad_couple_fiber_compression_crushing_rate_aliases(tmp_path: Path):
    deck = """# RADIOSS ALIAS DECK
/BEGIN
Ladeveze Coupled Fiber Compression Crushing Rate Aliases Test
/FAIL/LADEVEZE_COUPLED_FIBER_COMPRESSION_CRUSHING_RATE/1842
215.0, 645.0, 84.0, 2.75, 0.978
1, 1
/FAIL/LAD_CFCCR/1843
215.0, 645.0, 84.0, 2.75, 0.978
1, 1
/FAIL/LAD_CFCCR_MODEL/1844
215.0, 645.0, 84.0, 2.75, 0.978
1, 1
/FAIL/LAD_CFCCR_LAW/1845
215.0, 645.0, 84.0, 2.75, 0.978
1, 1
/FAIL/LADEVEZE_RATE_DEPENDENT_COUPLED_FIBER_COMPRESSION_CRUSHING/1846
215.0, 645.0, 84.0, 2.75, 0.978
1, 1
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 1842 in model.fail_ladcouplefibercompressioncrushingrates
    assert 1843 in model.fail_ladcouplefibercompressioncrushingrates
    assert 1844 in model.fail_ladcouplefibercompressioncrushingrates
    assert 1845 in model.fail_ladcouplefibercompressioncrushingrates
    assert 1846 in model.fail_ladcouplefibercompressioncrushingrates
    assert len(model.raw_fails) == 5


def test_m370_fail_lad_couple_fiber_compression_crushing_rate_missing_card(tmp_path: Path):
    deck = """# RADIOSS MISSING CARD DECK
/BEGIN
/FAIL/LAD_COUPLE_FIBER_COMPRESSION_CRUSHING_RATE/1847
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) > 0


def test_m370_eng_flexomagnetophononicmagnonic_resonance_energy_fixed(tmp_path: Path):
    c1 = f"{0.0042:>20.4f}{145:>10d}"
    deck = f"""# RADIOSS STARTER DECK
/BEGIN
Engine Flexomagnetophononicmagnonic Resonance Energy Fixed Format Test
2022 0
/ENG/FLEXOMAGNETOPHONONICMAGNONIC_RESONANCE_ENERGY/45
Resonance Energy Output Directive
{c1}
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 45 in model.eng_flexomagnetophononicmagnonic_resonance_energies
    eng = model.eng_flexomagnetophononicmagnonic_resonance_energies[45]
    assert pytest.approx(eng.dt_fmpmr) == 0.0042
    assert eng.sens_id == 145


def test_m370_eng_flexomagnetophononicmagnonic_resonance_energy_free(tmp_path: Path):
    deck = """# RADIOSS FREE DECK
/BEGIN
Engine Flexomagnetophononicmagnonic Resonance Energy Free Format Test
/ENG/FLEXOMAGNETOPHONONICMAGNONIC_RESONANCE_ENERGY/46
0.0052, 146
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 46 in model.eng_flexomagnetophononicmagnonic_resonance_energies
    eng = model.eng_flexomagnetophononicmagnonic_resonance_energies[46]
    assert pytest.approx(eng.dt_fmpmr) == 0.0052
    assert eng.sens_id == 146


def test_m370_eng_flexomagnetophononicmagnonic_resonance_energy_aliases(tmp_path: Path):
    deck = """# RADIOSS ALIAS DECK
/BEGIN
Engine Flexomagnetophononicmagnonic Resonance Energy Aliases Test
/ENG/FLEXOMAGNETO_PHONON_MAGNON_RES_WORK/47
0.0062, 147
/ENG/EFLEXOMAGNETOPHONONICMAGNONICRESONANCE/48
0.0072, 148
/ENG/FLEXOMAGNETOPHONONICMAGNONIC_RESONANCE_DISSIPATION/49
0.0082, 149
/ENG/EM_FLEXOMAGNETOPHONONICMAGNONIC_RESONANCE/50
0.0092, 150
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 47 in model.eng_flexomagnetophononicmagnonic_resonance_energies
    assert 48 in model.eng_flexomagnetophononicmagnonic_resonance_energies
    assert 49 in model.eng_flexomagnetophononicmagnonic_resonance_energies
    assert 50 in model.eng_flexomagnetophononicmagnonic_resonance_energies


def test_m370_eng_flexomagnetophononicmagnonic_resonance_energy_missing_card(tmp_path: Path):
    deck = """# RADIOSS MISSING CARD DECK
/BEGIN
/ENG/FLEXOMAGNETOPHONONICMAGNONIC_RESONANCE_ENERGY/51
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) > 0


def test_m370_lagmul_waldron_hybrid_spatial_linkage_joint_fixed(tmp_path: Path):
    c1 = f"{131:>10d}{132:>10d}{133:>10d}{6.0e6:>20.4f}{6:>10d}{2.0e-5:>20.4e}"
    c2 = f"{58.0:>20.4f}{42.0:>20.4f}{52.0:>20.4f}{18.0:>20.4f}"
    deck = f"""# RADIOSS STARTER DECK
/BEGIN
Waldron Hybrid Spatial Linkage Joint Fixed Format Test
2022 0
/LAGMUL/WALDRON_HYBRID_SPATIAL_LINKAGE_JOINT/95
Waldron Hybrid Spatial 6R Joint
{c1}
{c2}
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 95 in model.lagmul_waldron_hybrid_spatial_linkage_joints
    joint = model.lagmul_waldron_hybrid_spatial_linkage_joints[95]
    assert joint.node1 == 131
    assert joint.node2 == 132
    assert joint.node3 == 133
    assert pytest.approx(joint.stiff) == 6.0e6
    assert joint.skew_id == 6
    assert pytest.approx(joint.tol) == 2.0e-5
    assert pytest.approx(joint.link_len_a) == 58.0
    assert pytest.approx(joint.link_len_b) == 42.0
    assert pytest.approx(joint.twist_angle_alpha) == 52.0
    assert pytest.approx(joint.offset_distance_s) == 18.0


def test_m370_lagmul_waldron_hybrid_spatial_linkage_joint_free(tmp_path: Path):
    deck = """# RADIOSS FREE DECK
/BEGIN
Waldron Hybrid Spatial Linkage Joint Free Format Test
/LAGMUL/WALDRON_HYBRID_SPATIAL_LINKAGE_JOINT/96
231, 232, 233, 7.5e6, 8, 3.0e-5
62.0, 48.0, 72.0, 22.0
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 96 in model.lagmul_waldron_hybrid_spatial_linkage_joints
    joint = model.lagmul_waldron_hybrid_spatial_linkage_joints[96]
    assert joint.node1 == 231
    assert joint.node2 == 232
    assert joint.node3 == 233
    assert pytest.approx(joint.stiff) == 7.5e6
    assert joint.skew_id == 8
    assert pytest.approx(joint.tol) == 3.0e-5
    assert pytest.approx(joint.link_len_a) == 62.0
    assert pytest.approx(joint.link_len_b) == 48.0
    assert pytest.approx(joint.twist_angle_alpha) == 72.0
    assert pytest.approx(joint.offset_distance_s) == 22.0


def test_m370_lagmul_waldron_hybrid_spatial_linkage_joint_aliases(tmp_path: Path):
    deck = """# RADIOSS ALIAS DECK
/BEGIN
Waldron Hybrid Spatial Linkage Joint Aliases Test
/WALDRON_HYBRID_SPATIAL_LINKAGE_JOINT/97
331, 332, 333, 1.0e6, 0, 1.0e-6
36.0, 26.0, 36.0, 8.0
/LAGMUL/WALDRON_HYBRID_SPATIAL_LINKAGE/98
331, 332, 333, 1.0e6, 0, 1.0e-6
36.0, 26.0, 36.0, 8.0
/WALDRON_HYBRID_SPATIAL_LINKAGE/99
331, 332, 333, 1.0e6, 0, 1.0e-6
36.0, 26.0, 36.0, 8.0
/WALDRON_HYBRID_SPATIAL_MULTI_LOOP_MECHANISM/100
331, 332, 333, 1.0e6, 0, 1.0e-6
36.0, 26.0, 36.0, 8.0
/WALDRON_HYBRID_SPATIAL_SYMMETRIC_MECHANISM/101
331, 332, 333, 1.0e6, 0, 1.0e-6
36.0, 26.0, 36.0, 8.0
/WALDRON_HYBRID_SPATIAL_6R_MECHANISM/102
331, 332, 333, 1.0e6, 0, 1.0e-6
36.0, 26.0, 36.0, 8.0
/WALDRON_HYBRID_SPATIAL_OVERCONSTRAINED_MECHANISM/103
331, 332, 333, 1.0e6, 0, 1.0e-6
36.0, 26.0, 36.0, 8.0
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    for jid in range(97, 104):
        assert jid in model.lagmul_waldron_hybrid_spatial_linkage_joints


def test_m370_lagmul_waldron_hybrid_spatial_linkage_joint_missing_card(tmp_path: Path):
    deck = """# RADIOSS MISSING CARD DECK
/BEGIN
/LAGMUL/WALDRON_HYBRID_SPATIAL_LINKAGE_JOINT/104
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) > 0


def test_m370_sensor_spring_torsional_snap_rate_fixed(tmp_path: Path):
    c1 = f"{1178:>10d}{9.9e7:>20.4f}{0.0305:>20.4f}"
    deck = f"""# RADIOSS STARTER DECK
/BEGIN
Sensor Spring Torsional Snap Rate Fixed Format Test
2022 0
/SENSOR/SPRING_TORSIONAL_SNAP_RATE/1
Fixed Spring Torsional Snap Rate Sensor
{c1}
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 1 in model.sensor_spring_torsional_snap_rates
    s1 = model.sensor_spring_torsional_snap_rates[1]
    assert s1.spring_id == 1178
    assert pytest.approx(s1.jtors_snp_max) == 9.9e7
    assert pytest.approx(s1.jtors_crackle_max) == 9.9e7
    assert pytest.approx(s1.t_delay) == 0.0305
    assert len(model.sensors) == 1
    assert model.sensors[0].kind == "SPRING_TORSIONAL_SNAP_RATE"


def test_m370_sensor_spring_torsional_snap_rate_free(tmp_path: Path):
    deck = """# RADIOSS FREE DECK
/BEGIN
Sensor Spring Torsional Snap Rate Free Format Test
/SENSOR/SPRING_TORSIONAL_SNAP_RATE/2
Free Spring Torsional Snap Rate Sensor
1179, 1.55e8, 0.0455
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 2 in model.sensor_spring_torsional_snap_rates
    s2 = model.sensor_spring_torsional_snap_rates[2]
    assert s2.spring_id == 1179
    assert pytest.approx(s2.jtors_snp_max) == 1.55e8
    assert pytest.approx(s2.t_delay) == 0.0455


def test_m370_sensor_spring_torsional_snap_rate_aliases(tmp_path: Path):
    deck = """# RADIOSS ALIAS DECK
/BEGIN
Sensor Spring Torsional Snap Rate Aliases Test
/SENSOR/SPRING_TORS_SNAP_RATE/200
1199, 6.85e7, 0.0098
/SENSOR/SPRING_RATE_SNAP_TORS/201
1200, 7.05e7, 0.0108
/SENSOR/TORSIONAL_SNAP_RATE_SPRING/202
1201, 7.25e7, 0.0118
/SENSOR/SPRING_SNAP_TORS/203
1202, 7.45e7, 0.0128
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    for sid in range(200, 204):
        assert sid in model.sensor_spring_torsional_snap_rates


def test_m370_sensor_spring_torsional_snap_rate_missing_card(tmp_path: Path):
    deck = """# RADIOSS MISSING CARD DECK
/BEGIN
/SENSOR/SPRING_TORSIONAL_SNAP_RATE/205
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) > 0
