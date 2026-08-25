"""Tests for Milestone M371: LadDynamicInterlaminarShearDelaminationRate Failure Model, EngFlexomagnetophononicpolaritonicResonanceEnergy, ChenHybridSpatialLinkageJoint, and SensorSpringBendingSnapRate."""

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


def test_m371_fail_lad_dynamic_interlaminar_shear_delamination_rate_fixed(tmp_path: Path):
    c1 = f"{145.0:>20.4f}{435.0:>20.4f}{68.0:>20.4f}{2.85:>20.4f}{0.965:>20.4f}"
    c2 = f"{1:>10d}{2:>10d}"
    c3 = f"{1850:>10d}"
    deck = f"""# RADIOSS STARTER DECK
/BEGIN
Ladeveze Dynamic Interlaminar Shear Delamination Rate Fixed Format Test
2022 0
/FAIL/LAD_DYNAMIC_INTERLAMINAR_SHEAR_DELAMINATION_RATE/1850
Ladeveze Dynamic Interlaminar Shear Delamination Failure Model
{c1}
{c2}
{c3}
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 1850 in model.fail_laddynamicinterlaminarsheardelaminationrates
    fdisdr = model.fail_laddynamicinterlaminarsheardelaminationrates[1850]
    assert pytest.approx(fdisdr.sigma_disdr0) == 145.0
    assert pytest.approx(fdisdr.sigma_disdrc) == 435.0
    assert pytest.approx(fdisdr.gamma_disdr) == 68.0
    assert pytest.approx(fdisdr.p_disdr) == 2.85
    assert pytest.approx(fdisdr.d_disdr_max) == 0.965
    assert fdisdr.ifail_sh == 1
    assert fdisdr.ifail_so == 2
    assert fdisdr.fail_id == 1850
    assert len(model.raw_fails) == 1
    assert model.raw_fails[0][1].type == "LAD_DYNAMIC_INTERLAMINAR_SHEAR_DELAMINATION_RATE"


def test_m371_fail_lad_dynamic_interlaminar_shear_delamination_rate_free(tmp_path: Path):
    deck = """# RADIOSS FREE DECK
/BEGIN
Ladeveze Dynamic Interlaminar Shear Delamination Rate Free Format Test
/FAIL/LAD_DYNAMIC_INTERLAMINAR_SHEAR_DELAMINATION_RATE/1851
155.0, 465.0, 74.0, 3.05, 0.945
1, 1
1851
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 1851 in model.fail_laddynamicinterlaminarsheardelaminationrates
    fdisdr = model.fail_laddynamicinterlaminarsheardelaminationrates[1851]
    assert pytest.approx(fdisdr.sigma_disdr0) == 155.0
    assert pytest.approx(fdisdr.sigma_disdrc) == 465.0
    assert pytest.approx(fdisdr.gamma_disdr) == 74.0
    assert pytest.approx(fdisdr.p_disdr) == 3.05
    assert pytest.approx(fdisdr.d_disdr_max) == 0.945
    assert fdisdr.fail_id == 1851


def test_m371_fail_lad_dynamic_interlaminar_shear_delamination_rate_aliases(tmp_path: Path):
    deck = """# RADIOSS ALIAS DECK
/BEGIN
Ladeveze Dynamic Interlaminar Shear Delamination Rate Aliases Test
/FAIL/LADEVEZE_DYNAMIC_INTERLAMINAR_SHEAR_DELAMINATION_RATE/1852
135.0, 405.0, 62.0, 2.55, 0.985
1, 1
/FAIL/LAD_DISDR/1853
135.0, 405.0, 62.0, 2.55, 0.985
1, 1
/FAIL/LAD_DISDR_MODEL/1854
135.0, 405.0, 62.0, 2.55, 0.985
1, 1
/FAIL/LAD_DISDR_LAW/1855
135.0, 405.0, 62.0, 2.55, 0.985
1, 1
/FAIL/LADEVEZE_RATE_DEPENDENT_DYNAMIC_INTERLAMINAR_SHEAR_DELAMINATION/1856
135.0, 405.0, 62.0, 2.55, 0.985
1, 1
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 1852 in model.fail_laddynamicinterlaminarsheardelaminationrates
    assert 1853 in model.fail_laddynamicinterlaminarsheardelaminationrates
    assert 1854 in model.fail_laddynamicinterlaminarsheardelaminationrates
    assert 1855 in model.fail_laddynamicinterlaminarsheardelaminationrates
    assert 1856 in model.fail_laddynamicinterlaminarsheardelaminationrates
    assert len(model.raw_fails) == 5


def test_m371_fail_lad_dynamic_interlaminar_shear_delamination_rate_missing_card(tmp_path: Path):
    deck = """# RADIOSS MISSING CARD DECK
/BEGIN
/FAIL/LAD_DYNAMIC_INTERLAMINAR_SHEAR_DELAMINATION_RATE/1857
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) > 0


def test_m371_eng_flexomagnetophononicpolaritonic_resonance_energy_fixed(tmp_path: Path):
    c1 = f"{0.0036:>20.4f}{155:>10d}"
    deck = f"""# RADIOSS STARTER DECK
/BEGIN
Engine Flexomagnetophononicpolaritonic Resonance Energy Fixed Format Test
2022 0
/ENG/FLEXOMAGNETOPHONONICPOLARITONIC_RESONANCE_ENERGY/55
Resonance Energy Output Directive
{c1}
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 55 in model.eng_flexomagnetophononicpolaritonic_resonance_energies
    eng = model.eng_flexomagnetophononicpolaritonic_resonance_energies[55]
    assert pytest.approx(eng.dt_fmppor) == 0.0036
    assert eng.sens_id == 155


def test_m371_eng_flexomagnetophononicpolaritonic_resonance_energy_free(tmp_path: Path):
    deck = """# RADIOSS FREE DECK
/BEGIN
Engine Flexomagnetophononicpolaritonic Resonance Energy Free Format Test
/ENG/FLEXOMAGNETOPHONONICPOLARITONIC_RESONANCE_ENERGY/56
0.0046, 156
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 56 in model.eng_flexomagnetophononicpolaritonic_resonance_energies
    eng = model.eng_flexomagnetophononicpolaritonic_resonance_energies[56]
    assert pytest.approx(eng.dt_fmppor) == 0.0046
    assert eng.sens_id == 156


def test_m371_eng_flexomagnetophononicpolaritonic_resonance_energy_aliases(tmp_path: Path):
    deck = """# RADIOSS ALIAS DECK
/BEGIN
Engine Flexomagnetophononicpolaritonic Resonance Energy Aliases Test
/ENG/FLEXOMAGNETO_PHONON_POLARITON_RES_WORK/57
0.0056, 157
/ENG/EFLEXOMAGNETOPHONONICPOLARITONICRESONANCE/58
0.0066, 158
/ENG/FLEXOMAGNETOPHONONICPOLARITONIC_RESONANCE_DISSIPATION/59
0.0076, 159
/ENG/EM_FLEXOMAGNETOPHONONICPOLARITONIC_RESONANCE/60
0.0086, 160
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 57 in model.eng_flexomagnetophononicpolaritonic_resonance_energies
    assert 58 in model.eng_flexomagnetophononicpolaritonic_resonance_energies
    assert 59 in model.eng_flexomagnetophononicpolaritonic_resonance_energies
    assert 60 in model.eng_flexomagnetophononicpolaritonic_resonance_energies


def test_m371_eng_flexomagnetophononicpolaritonic_resonance_energy_missing_card(tmp_path: Path):
    deck = """# RADIOSS MISSING CARD DECK
/BEGIN
/ENG/FLEXOMAGNETOPHONONICPOLARITONIC_RESONANCE_ENERGY/61
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) > 0


def test_m371_lagmul_chen_hybrid_spatial_linkage_joint_fixed(tmp_path: Path):
    c1 = f"{141:>10d}{142:>10d}{143:>10d}{6.5e6:>20.4f}{7:>10d}{2.5e-5:>20.4e}"
    c2 = f"{64.0:>20.4f}{46.0:>20.4f}{58.0:>20.4f}{19.0:>20.4f}"
    deck = f"""# RADIOSS STARTER DECK
/BEGIN
Chen Hybrid Spatial Linkage Joint Fixed Format Test
2022 0
/LAGMUL/CHEN_HYBRID_SPATIAL_LINKAGE_JOINT/105
Chen Hybrid Spatial 6R Joint
{c1}
{c2}
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 105 in model.lagmul_chen_hybrid_spatial_linkage_joints
    joint = model.lagmul_chen_hybrid_spatial_linkage_joints[105]
    assert joint.node1 == 141
    assert joint.node2 == 142
    assert joint.node3 == 143
    assert pytest.approx(joint.stiff) == 6.5e6
    assert joint.skew_id == 7
    assert pytest.approx(joint.tol) == 2.5e-5
    assert pytest.approx(joint.link_len_a) == 64.0
    assert pytest.approx(joint.link_len_b) == 46.0
    assert pytest.approx(joint.twist_angle_alpha) == 58.0
    assert pytest.approx(joint.offset_distance_s) == 19.0


def test_m371_lagmul_chen_hybrid_spatial_linkage_joint_free(tmp_path: Path):
    deck = """# RADIOSS FREE DECK
/BEGIN
Chen Hybrid Spatial Linkage Joint Free Format Test
/LAGMUL/CHEN_HYBRID_SPATIAL_LINKAGE_JOINT/106
241, 242, 243, 8.0e6, 9, 3.5e-5
68.0, 52.0, 78.0, 24.0
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 106 in model.lagmul_chen_hybrid_spatial_linkage_joints
    joint = model.lagmul_chen_hybrid_spatial_linkage_joints[106]
    assert joint.node1 == 241
    assert joint.node2 == 242
    assert joint.node3 == 243
    assert pytest.approx(joint.stiff) == 8.0e6
    assert joint.skew_id == 9
    assert pytest.approx(joint.tol) == 3.5e-5
    assert pytest.approx(joint.link_len_a) == 68.0
    assert pytest.approx(joint.link_len_b) == 52.0
    assert pytest.approx(joint.twist_angle_alpha) == 78.0
    assert pytest.approx(joint.offset_distance_s) == 24.0


def test_m371_lagmul_chen_hybrid_spatial_linkage_joint_aliases(tmp_path: Path):
    deck = """# RADIOSS ALIAS DECK
/BEGIN
Chen Hybrid Spatial Linkage Joint Aliases Test
/CHEN_HYBRID_SPATIAL_LINKAGE_JOINT/107
341, 342, 343, 1.0e6, 0, 1.0e-6
38.0, 28.0, 38.0, 9.0
/LAGMUL/CHEN_HYBRID_SPATIAL_LINKAGE/108
341, 342, 343, 1.0e6, 0, 1.0e-6
38.0, 28.0, 38.0, 9.0
/CHEN_HYBRID_SPATIAL_LINKAGE/109
341, 342, 343, 1.0e6, 0, 1.0e-6
38.0, 28.0, 38.0, 9.0
/CHEN_HYBRID_SPATIAL_MULTI_LOOP_MECHANISM/110
341, 342, 343, 1.0e6, 0, 1.0e-6
38.0, 28.0, 38.0, 9.0
/CHEN_HYBRID_SPATIAL_SYMMETRIC_MECHANISM/111
341, 342, 343, 1.0e6, 0, 1.0e-6
38.0, 28.0, 38.0, 9.0
/CHEN_HYBRID_SPATIAL_6R_MECHANISM/112
341, 342, 343, 1.0e6, 0, 1.0e-6
38.0, 28.0, 38.0, 9.0
/CHEN_HYBRID_SPATIAL_OVERCONSTRAINED_MECHANISM/113
341, 342, 343, 1.0e6, 0, 1.0e-6
38.0, 28.0, 38.0, 9.0
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    for jid in range(107, 114):
        assert jid in model.lagmul_chen_hybrid_spatial_linkage_joints


def test_m371_lagmul_chen_hybrid_spatial_linkage_joint_missing_card(tmp_path: Path):
    deck = """# RADIOSS MISSING CARD DECK
/BEGIN
/LAGMUL/CHEN_HYBRID_SPATIAL_LINKAGE_JOINT/114
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) > 0


def test_m371_sensor_spring_bending_snap_rate_fixed(tmp_path: Path):
    c1 = f"{1188:>10d}{1.08e8:>20.4f}{0.0335:>20.4f}"
    deck = f"""# RADIOSS STARTER DECK
/BEGIN
Sensor Spring Bending Snap Rate Fixed Format Test
2022 0
/SENSOR/SPRING_BENDING_SNAP_RATE/1
Fixed Spring Bending Snap Rate Sensor
{c1}
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 1 in model.sensor_spring_bending_snap_rates
    s1 = model.sensor_spring_bending_snap_rates[1]
    assert s1.spring_id == 1188
    assert pytest.approx(s1.jbend_snp_max) == 1.08e8
    assert pytest.approx(s1.jbend_crackle_max) == 1.08e8
    assert pytest.approx(s1.t_delay) == 0.0335
    assert len(model.sensors) == 1
    assert model.sensors[0].kind == "SPRING_BENDING_SNAP_RATE"


def test_m371_sensor_spring_bending_snap_rate_free(tmp_path: Path):
    deck = """# RADIOSS FREE DECK
/BEGIN
Sensor Spring Bending Snap Rate Free Format Test
/SENSOR/SPRING_BENDING_SNAP_RATE/2
Free Spring Bending Snap Rate Sensor
1189, 1.68e8, 0.0485
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 2 in model.sensor_spring_bending_snap_rates
    s2 = model.sensor_spring_bending_snap_rates[2]
    assert s2.spring_id == 1189
    assert pytest.approx(s2.jbend_snp_max) == 1.68e8
    assert pytest.approx(s2.t_delay) == 0.0485


def test_m371_sensor_spring_bending_snap_rate_aliases(tmp_path: Path):
    deck = """# RADIOSS ALIAS DECK
/BEGIN
Sensor Spring Bending Snap Rate Aliases Test
/SENSOR/SPRING_BEND_SNAP_RATE/210
1209, 7.35e7, 0.0102
/SENSOR/SPRING_RATE_SNAP_BEND/211
1210, 7.55e7, 0.0112
/SENSOR/BENDING_SNAP_RATE_SPRING/212
1211, 7.75e7, 0.0122
/SENSOR/SPRING_SNAP_BEND/213
1212, 7.95e7, 0.0132
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    for sid in range(210, 214):
        assert sid in model.sensor_spring_bending_snap_rates


def test_m371_sensor_spring_bending_snap_rate_missing_card(tmp_path: Path):
    deck = """# RADIOSS MISSING CARD DECK
/BEGIN
/SENSOR/SPRING_BENDING_SNAP_RATE/215
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) > 0
