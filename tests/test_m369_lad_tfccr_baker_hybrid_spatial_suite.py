"""Tests for Milestone M369: LadTransverseFiberCompressionCrushingRate Failure Model, EngFlexomagnetophononicexcitonicResonanceEnergy, BakerHybridSpatialLinkageJoint, and SensorSpringTotalSnapRate."""

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


def test_m369_fail_lad_transverse_fiber_compression_crushing_rate_fixed(tmp_path: Path):
    c1 = f"{220.0:>20.4f}{660.0:>20.4f}{88.0:>20.4f}{3.20:>20.4f}{0.960:>20.4f}"
    c2 = f"{1:>10d}{2:>10d}"
    c3 = f"{1835:>10d}"
    deck = f"""# RADIOSS STARTER DECK
/BEGIN
Ladeveze Transverse Fiber Compression Crushing Rate Fixed Format Test
2022 0
/FAIL/LAD_TRANSVERSE_FIBER_COMPRESSION_CRUSHING_RATE/1835
Ladeveze Transverse Fiber Compression Crushing Failure Model
{c1}
{c2}
{c3}
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 1835 in model.fail_ladtransversefibercompressioncrushingrates
    ftfccr = model.fail_ladtransversefibercompressioncrushingrates[1835]
    assert pytest.approx(ftfccr.sigma_tfccr0) == 220.0
    assert pytest.approx(ftfccr.sigma_tfccrc) == 660.0
    assert pytest.approx(ftfccr.gamma_tfccr) == 88.0
    assert pytest.approx(ftfccr.p_tfccr) == 3.20
    assert pytest.approx(ftfccr.d_tfccr_max) == 0.960
    assert ftfccr.ifail_sh == 1
    assert ftfccr.ifail_so == 2
    assert ftfccr.fail_id == 1835
    assert len(model.raw_fails) == 1
    assert model.raw_fails[0][1].type == "LAD_TRANSVERSE_FIBER_COMPRESSION_CRUSHING_RATE"


def test_m369_fail_lad_transverse_fiber_compression_crushing_rate_free(tmp_path: Path):
    deck = """# RADIOSS FREE DECK
/BEGIN
Ladeveze Transverse Fiber Compression Crushing Rate Free Format Test
/FAIL/LAD_TRANSVERSE_FIBER_COMPRESSION_CRUSHING_RATE/1836
230.0, 690.0, 96.0, 3.40, 0.940
1, 1
1836
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 1836 in model.fail_ladtransversefibercompressioncrushingrates
    ftfccr = model.fail_ladtransversefibercompressioncrushingrates[1836]
    assert pytest.approx(ftfccr.sigma_tfccr0) == 230.0
    assert pytest.approx(ftfccr.sigma_tfccrc) == 690.0
    assert pytest.approx(ftfccr.gamma_tfccr) == 96.0
    assert pytest.approx(ftfccr.p_tfccr) == 3.40
    assert pytest.approx(ftfccr.d_tfccr_max) == 0.940
    assert ftfccr.fail_id == 1836


def test_m369_fail_lad_transverse_fiber_compression_crushing_rate_aliases(tmp_path: Path):
    deck = """# RADIOSS ALIAS DECK
/BEGIN
Ladeveze Transverse Fiber Compression Crushing Rate Aliases Test
/FAIL/LADEVEZE_TRANSVERSE_FIBER_COMPRESSION_CRUSHING_RATE/1837
210.0, 630.0, 80.0, 2.70, 0.980
1, 1
/FAIL/LAD_TFCCR/1838
210.0, 630.0, 80.0, 2.70, 0.980
1, 1
/FAIL/LAD_TFCCR_MODEL/1839
210.0, 630.0, 80.0, 2.70, 0.980
1, 1
/FAIL/LAD_TFCCR_LAW/1840
210.0, 630.0, 80.0, 2.70, 0.980
1, 1
/FAIL/LADEVEZE_RATE_DEPENDENT_TRANSVERSE_FIBER_COMPRESSION_CRUSHING/1841
210.0, 630.0, 80.0, 2.70, 0.980
1, 1
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 1837 in model.fail_ladtransversefibercompressioncrushingrates
    assert 1838 in model.fail_ladtransversefibercompressioncrushingrates
    assert 1839 in model.fail_ladtransversefibercompressioncrushingrates
    assert 1840 in model.fail_ladtransversefibercompressioncrushingrates
    assert 1841 in model.fail_ladtransversefibercompressioncrushingrates
    assert len(model.raw_fails) == 5


def test_m369_fail_lad_transverse_fiber_compression_crushing_rate_missing_card(tmp_path: Path):
    deck = """# RADIOSS MISSING CARD DECK
/BEGIN
/FAIL/LAD_TRANSVERSE_FIBER_COMPRESSION_CRUSHING_RATE/1842
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) > 0


def test_m369_eng_flexomagnetophononicexcitonic_resonance_energy_fixed(tmp_path: Path):
    c1 = f"{0.0040:>20.4f}{135:>10d}"
    deck = f"""# RADIOSS STARTER DECK
/BEGIN
Engine Flexomagnetophononicexcitonic Resonance Energy Fixed Format Test
2022 0
/ENG/FLEXOMAGNETOPHONONICEXCITONIC_RESONANCE_ENERGY/35
Resonance Energy Output Directive
{c1}
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 35 in model.eng_flexomagnetophononicexcitonic_resonance_energies
    eng = model.eng_flexomagnetophononicexcitonic_resonance_energies[35]
    assert pytest.approx(eng.dt_fmper) == 0.0040
    assert eng.sens_id == 135


def test_m369_eng_flexomagnetophononicexcitonic_resonance_energy_free(tmp_path: Path):
    deck = """# RADIOSS FREE DECK
/BEGIN
Engine Flexomagnetophononicexcitonic Resonance Energy Free Format Test
/ENG/FLEXOMAGNETOPHONONICEXCITONIC_RESONANCE_ENERGY/36
0.0050, 136
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 36 in model.eng_flexomagnetophononicexcitonic_resonance_energies
    eng = model.eng_flexomagnetophononicexcitonic_resonance_energies[36]
    assert pytest.approx(eng.dt_fmper) == 0.0050
    assert eng.sens_id == 136


def test_m369_eng_flexomagnetophononicexcitonic_resonance_energy_aliases(tmp_path: Path):
    deck = """# RADIOSS ALIAS DECK
/BEGIN
Engine Flexomagnetophononicexcitonic Resonance Energy Aliases Test
/ENG/FLEXOMAGNETO_PHONON_EXCITON_RES_WORK/37
0.0060, 137
/ENG/EFLEXOMAGNETOPHONONICEXCITONICRESONANCE/38
0.0070, 138
/ENG/FLEXOMAGNETOPHONONICEXCITONIC_RESONANCE_DISSIPATION/39
0.0080, 139
/ENG/EM_FLEXOMAGNETOPHONONICEXCITONIC_RESONANCE/40
0.0090, 140
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 37 in model.eng_flexomagnetophononicexcitonic_resonance_energies
    assert 38 in model.eng_flexomagnetophononicexcitonic_resonance_energies
    assert 39 in model.eng_flexomagnetophononicexcitonic_resonance_energies
    assert 40 in model.eng_flexomagnetophononicexcitonic_resonance_energies


def test_m369_eng_flexomagnetophononicexcitonic_resonance_energy_missing_card(tmp_path: Path):
    deck = """# RADIOSS MISSING CARD DECK
/BEGIN
/ENG/FLEXOMAGNETOPHONONICEXCITONIC_RESONANCE_ENERGY/41
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) > 0


def test_m369_lagmul_baker_hybrid_spatial_linkage_joint_fixed(tmp_path: Path):
    c1 = f"{121:>10d}{122:>10d}{123:>10d}{5.8e6:>20.4f}{5:>10d}{1.8e-5:>20.4e}"
    c2 = f"{56.0:>20.4f}{40.0:>20.4f}{50.0:>20.4f}{16.0:>20.4f}"
    deck = f"""# RADIOSS STARTER DECK
/BEGIN
Baker Hybrid Spatial Linkage Joint Fixed Format Test
2022 0
/LAGMUL/BAKER_HYBRID_SPATIAL_LINKAGE_JOINT/85
Baker Hybrid Spatial 6R Joint
{c1}
{c2}
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 85 in model.lagmul_baker_hybrid_spatial_linkage_joints
    joint = model.lagmul_baker_hybrid_spatial_linkage_joints[85]
    assert joint.node1 == 121
    assert joint.node2 == 122
    assert joint.node3 == 123
    assert pytest.approx(joint.stiff) == 5.8e6
    assert joint.skew_id == 5
    assert pytest.approx(joint.tol) == 1.8e-5
    assert pytest.approx(joint.link_len_a) == 56.0
    assert pytest.approx(joint.link_len_b) == 40.0
    assert pytest.approx(joint.twist_angle_alpha) == 50.0
    assert pytest.approx(joint.offset_distance_s) == 16.0


def test_m369_lagmul_baker_hybrid_spatial_linkage_joint_free(tmp_path: Path):
    deck = """# RADIOSS FREE DECK
/BEGIN
Baker Hybrid Spatial Linkage Joint Free Format Test
/LAGMUL/BAKER_HYBRID_SPATIAL_LINKAGE_JOINT/86
221, 222, 223, 7.2e6, 7, 2.8e-5
60.0, 46.0, 70.0, 20.0
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 86 in model.lagmul_baker_hybrid_spatial_linkage_joints
    joint = model.lagmul_baker_hybrid_spatial_linkage_joints[86]
    assert joint.node1 == 221
    assert joint.node2 == 222
    assert joint.node3 == 223
    assert pytest.approx(joint.stiff) == 7.2e6
    assert joint.skew_id == 7
    assert pytest.approx(joint.tol) == 2.8e-5
    assert pytest.approx(joint.link_len_a) == 60.0
    assert pytest.approx(joint.link_len_b) == 46.0
    assert pytest.approx(joint.twist_angle_alpha) == 70.0
    assert pytest.approx(joint.offset_distance_s) == 20.0


def test_m369_lagmul_baker_hybrid_spatial_linkage_joint_aliases(tmp_path: Path):
    deck = """# RADIOSS ALIAS DECK
/BEGIN
Baker Hybrid Spatial Linkage Joint Aliases Test
/BAKER_HYBRID_SPATIAL_LINKAGE_JOINT/87
321, 322, 323, 1.0e6, 0, 1.0e-6
34.0, 24.0, 34.0, 7.0
/LAGMUL/BAKER_HYBRID_SPATIAL_LINKAGE/88
321, 322, 323, 1.0e6, 0, 1.0e-6
34.0, 24.0, 34.0, 7.0
/BAKER_HYBRID_SPATIAL_LINKAGE/89
321, 322, 323, 1.0e6, 0, 1.0e-6
34.0, 24.0, 34.0, 7.0
/BAKER_HYBRID_SPATIAL_MULTI_LOOP_MECHANISM/90
321, 322, 323, 1.0e6, 0, 1.0e-6
34.0, 24.0, 34.0, 7.0
/BAKER_HYBRID_SPATIAL_SYMMETRIC_MECHANISM/91
321, 322, 323, 1.0e6, 0, 1.0e-6
34.0, 24.0, 34.0, 7.0
/BAKER_HYBRID_SPATIAL_6R_MECHANISM/92
321, 322, 323, 1.0e6, 0, 1.0e-6
34.0, 24.0, 34.0, 7.0
/BAKER_HYBRID_SPATIAL_OVERCONSTRAINED_MECHANISM/93
321, 322, 323, 1.0e6, 0, 1.0e-6
34.0, 24.0, 34.0, 7.0
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    for jid in range(87, 94):
        assert jid in model.lagmul_baker_hybrid_spatial_linkage_joints


def test_m369_lagmul_baker_hybrid_spatial_linkage_joint_missing_card(tmp_path: Path):
    deck = """# RADIOSS MISSING CARD DECK
/BEGIN
/LAGMUL/BAKER_HYBRID_SPATIAL_LINKAGE_JOINT/94
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) > 0


def test_m369_sensor_spring_total_snap_rate_fixed(tmp_path: Path):
    c1 = f"{1168:>10d}{9.8e7:>20.4f}{0.0295:>20.4f}"
    deck = f"""# RADIOSS STARTER DECK
/BEGIN
Sensor Spring Total Snap Rate Fixed Format Test
2022 0
/SENSOR/SPRING_TOTAL_SNAP_RATE/1
Fixed Spring Total Snap Rate Sensor
{c1}
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 1 in model.sensor_spring_total_snap_rates
    s1 = model.sensor_spring_total_snap_rates[1]
    assert s1.spring_id == 1168
    assert pytest.approx(s1.jtot_snp_max) == 9.8e7
    assert pytest.approx(s1.t_delay) == 0.0295
    assert len(model.sensors) == 1
    assert model.sensors[0].kind == "SPRING_TOTAL_SNAP_RATE"


def test_m369_sensor_spring_total_snap_rate_free(tmp_path: Path):
    deck = """# RADIOSS FREE DECK
/BEGIN
Sensor Spring Total Snap Rate Free Format Test
/SENSOR/SPRING_TOTAL_SNAP_RATE/2
Free Spring Total Snap Rate Sensor
1169, 1.45e8, 0.0435
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 2 in model.sensor_spring_total_snap_rates
    s2 = model.sensor_spring_total_snap_rates[2]
    assert s2.spring_id == 1169
    assert pytest.approx(s2.jtot_snp_max) == 1.45e8
    assert pytest.approx(s2.t_delay) == 0.0435


def test_m369_sensor_spring_total_snap_rate_aliases(tmp_path: Path):
    deck = """# RADIOSS ALIAS DECK
/BEGIN
Sensor Spring Total Snap Rate Aliases Test
/SENSOR/SPRING_TOT_SNAP_RATE/190
1195, 6.75e7, 0.0095
/SENSOR/SPRING_RATE_SNAP_TOT/191
1196, 6.95e7, 0.0105
/SENSOR/TOTAL_SNAP_RATE_SPRING/192
1197, 7.15e7, 0.0115
/SENSOR/SPRING_SNAP_TOT/193
1198, 7.35e7, 0.0125
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    for sid in range(190, 194):
        assert sid in model.sensor_spring_total_snap_rates


def test_m369_sensor_spring_total_snap_rate_missing_card(tmp_path: Path):
    deck = """# RADIOSS MISSING CARD DECK
/BEGIN
/SENSOR/SPRING_TOTAL_SNAP_RATE/195
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) > 0
