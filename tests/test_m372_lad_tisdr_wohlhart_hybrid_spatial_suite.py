"""Tests for Milestone M372: LadTransverseInterlaminarShearDelaminationRate Failure Model, EngFlexomagnetoexcitonicmagnonicResonanceEnergy, WohlhartHybridSpatialLinkageJoint, and SensorSpringTotalAngularSnapRate."""

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


def test_m372_fail_lad_transverse_interlaminar_shear_delamination_rate_fixed(tmp_path: Path):
    c1 = f"{155.0:>20.4f}{465.0:>20.4f}{72.0:>20.4f}{2.95:>20.4f}{0.962:>20.4f}"
    c2 = f"{1:>10d}{2:>10d}"
    c3 = f"{1860:>10d}"
    deck = f"""# RADIOSS STARTER DECK
/BEGIN
Ladeveze Transverse Interlaminar Shear Delamination Rate Fixed Format Test
2022 0
/FAIL/LAD_TRANSVERSE_INTERLAMINAR_SHEAR_DELAMINATION_RATE/1860
Ladeveze Transverse Interlaminar Shear Delamination Failure Model
{c1}
{c2}
{c3}
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 1860 in model.fail_ladtransverseinterlaminarsheardelaminationrates
    ftisdr = model.fail_ladtransverseinterlaminarsheardelaminationrates[1860]
    assert pytest.approx(ftisdr.sigma_tisdr0) == 155.0
    assert pytest.approx(ftisdr.sigma_tisdrc) == 465.0
    assert pytest.approx(ftisdr.gamma_tisdr) == 72.0
    assert pytest.approx(ftisdr.p_tisdr) == 2.95
    assert pytest.approx(ftisdr.d_tisdr_max) == 0.962
    assert ftisdr.ifail_sh == 1
    assert ftisdr.ifail_so == 2
    assert ftisdr.fail_id == 1860
    assert len(model.raw_fails) == 1
    assert model.raw_fails[0][1].type == "LAD_TRANSVERSE_INTERLAMINAR_SHEAR_DELAMINATION_RATE"


def test_m372_fail_lad_transverse_interlaminar_shear_delamination_rate_free(tmp_path: Path):
    deck = """# RADIOSS FREE DECK
/BEGIN
Ladeveze Transverse Interlaminar Shear Delamination Rate Free Format Test
/FAIL/LAD_TRANSVERSE_INTERLAMINAR_SHEAR_DELAMINATION_RATE/1861
165.0, 495.0, 78.0, 3.15, 0.942
1, 1
1861
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 1861 in model.fail_ladtransverseinterlaminarsheardelaminationrates
    ftisdr = model.fail_ladtransverseinterlaminarsheardelaminationrates[1861]
    assert pytest.approx(ftisdr.sigma_tisdr0) == 165.0
    assert pytest.approx(ftisdr.sigma_tisdrc) == 495.0
    assert pytest.approx(ftisdr.gamma_tisdr) == 78.0
    assert pytest.approx(ftisdr.p_tisdr) == 3.15
    assert pytest.approx(ftisdr.d_tisdr_max) == 0.942
    assert ftisdr.fail_id == 1861


def test_m372_fail_lad_transverse_interlaminar_shear_delamination_rate_aliases(tmp_path: Path):
    deck = """# RADIOSS ALIAS DECK
/BEGIN
Ladeveze Transverse Interlaminar Shear Delamination Rate Aliases Test
/FAIL/LADEVEZE_TRANSVERSE_INTERLAMINAR_SHEAR_DELAMINATION_RATE/1862
145.0, 435.0, 66.0, 2.65, 0.982
1, 1
/FAIL/LAD_TISDR/1863
145.0, 435.0, 66.0, 2.65, 0.982
1, 1
/FAIL/LAD_TISDR_MODEL/1864
145.0, 435.0, 66.0, 2.65, 0.982
1, 1
/FAIL/LAD_TISDR_LAW/1865
145.0, 435.0, 66.0, 2.65, 0.982
1, 1
/FAIL/LADEVEZE_RATE_DEPENDENT_TRANSVERSE_INTERLAMINAR_SHEAR_DELAMINATION/1866
145.0, 435.0, 66.0, 2.65, 0.982
1, 1
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 1862 in model.fail_ladtransverseinterlaminarsheardelaminationrates
    assert 1863 in model.fail_ladtransverseinterlaminarsheardelaminationrates
    assert 1864 in model.fail_ladtransverseinterlaminarsheardelaminationrates
    assert 1865 in model.fail_ladtransverseinterlaminarsheardelaminationrates
    assert 1866 in model.fail_ladtransverseinterlaminarsheardelaminationrates
    assert len(model.raw_fails) == 5


def test_m372_fail_lad_transverse_interlaminar_shear_delamination_rate_missing_card(tmp_path: Path):
    deck = """# RADIOSS MISSING CARD DECK
/BEGIN
/FAIL/LAD_TRANSVERSE_INTERLAMINAR_SHEAR_DELAMINATION_RATE/1867
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) > 0


def test_m372_eng_flexomagnetoexcitonicmagnonic_resonance_energy_fixed(tmp_path: Path):
    c1 = f"{0.0039:>20.4f}{165:>10d}"
    deck = f"""# RADIOSS STARTER DECK
/BEGIN
Engine Flexomagnetoexcitonicmagnonic Resonance Energy Fixed Format Test
2022 0
/ENG/FLEXOMAGNETOEXCITONICMAGNONIC_RESONANCE_ENERGY/65
Resonance Energy Output Directive
{c1}
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 65 in model.eng_flexomagnetoexcitonicmagnonic_resonance_energies
    eng = model.eng_flexomagnetoexcitonicmagnonic_resonance_energies[65]
    assert pytest.approx(eng.dt_fmemr) == 0.0039
    assert eng.sens_id == 165


def test_m372_eng_flexomagnetoexcitonicmagnonic_resonance_energy_free(tmp_path: Path):
    deck = """# RADIOSS FREE DECK
/BEGIN
Engine Flexomagnetoexcitonicmagnonic Resonance Energy Free Format Test
/ENG/FLEXOMAGNETOEXCITONICMAGNONIC_RESONANCE_ENERGY/66
0.0049, 166
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 66 in model.eng_flexomagnetoexcitonicmagnonic_resonance_energies
    eng = model.eng_flexomagnetoexcitonicmagnonic_resonance_energies[66]
    assert pytest.approx(eng.dt_fmemr) == 0.0049
    assert eng.sens_id == 166


def test_m372_eng_flexomagnetoexcitonicmagnonic_resonance_energy_aliases(tmp_path: Path):
    deck = """# RADIOSS ALIAS DECK
/BEGIN
Engine Flexomagnetoexcitonicmagnonic Resonance Energy Aliases Test
/ENG/FLEXOMAGNETO_EXCITON_MAGNON_RES_WORK/67
0.0059, 167
/ENG/EFLEXOMAGNETOEXCITONICMAGNONICRESONANCE/68
0.0069, 168
/ENG/FLEXOMAGNETOEXCITONICMAGNONIC_RESONANCE_DISSIPATION/69
0.0079, 169
/ENG/EM_FLEXOMAGNETOEXCITONICMAGNONIC_RESONANCE/70
0.0089, 170
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 67 in model.eng_flexomagnetoexcitonicmagnonic_resonance_energies
    assert 68 in model.eng_flexomagnetoexcitonicmagnonic_resonance_energies
    assert 69 in model.eng_flexomagnetoexcitonicmagnonic_resonance_energies
    assert 70 in model.eng_flexomagnetoexcitonicmagnonic_resonance_energies


def test_m372_eng_flexomagnetoexcitonicmagnonic_resonance_energy_missing_card(tmp_path: Path):
    deck = """# RADIOSS MISSING CARD DECK
/BEGIN
/ENG/FLEXOMAGNETOEXCITONICMAGNONIC_RESONANCE_ENERGY/71
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) > 0


def test_m372_lagmul_wohlhart_hybrid_spatial_linkage_joint_fixed(tmp_path: Path):
    c1 = f"{151:>10d}{152:>10d}{153:>10d}{7.0e6:>20.4f}{8:>10d}{2.8e-5:>20.4e}"
    c2 = f"{66.0:>20.4f}{48.0:>20.4f}{62.0:>20.4f}{20.0:>20.4f}"
    deck = f"""# RADIOSS STARTER DECK
/BEGIN
Wohlhart Hybrid Spatial Linkage Joint Fixed Format Test
2022 0
/LAGMUL/WOHLHART_HYBRID_SPATIAL_LINKAGE_JOINT/115
Wohlhart Hybrid Spatial 6R Joint
{c1}
{c2}
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 115 in model.lagmul_wohlhart_hybrid_spatial_linkage_joints
    joint = model.lagmul_wohlhart_hybrid_spatial_linkage_joints[115]
    assert joint.node1 == 151
    assert joint.node2 == 152
    assert joint.node3 == 153
    assert pytest.approx(joint.stiff) == 7.0e6
    assert joint.skew_id == 8
    assert pytest.approx(joint.tol) == 2.8e-5
    assert pytest.approx(joint.link_len_a) == 66.0
    assert pytest.approx(joint.link_len_b) == 48.0
    assert pytest.approx(joint.twist_angle_alpha) == 62.0
    assert pytest.approx(joint.offset_distance_s) == 20.0


def test_m372_lagmul_wohlhart_hybrid_spatial_linkage_joint_free(tmp_path: Path):
    deck = """# RADIOSS FREE DECK
/BEGIN
Wohlhart Hybrid Spatial Linkage Joint Free Format Test
/LAGMUL/WOHLHART_HYBRID_SPATIAL_LINKAGE_JOINT/116
251, 252, 253, 8.5e6, 10, 3.8e-5
72.0, 54.0, 82.0, 26.0
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 116 in model.lagmul_wohlhart_hybrid_spatial_linkage_joints
    joint = model.lagmul_wohlhart_hybrid_spatial_linkage_joints[116]
    assert joint.node1 == 251
    assert joint.node2 == 252
    assert joint.node3 == 253
    assert pytest.approx(joint.stiff) == 8.5e6
    assert joint.skew_id == 10
    assert pytest.approx(joint.tol) == 3.8e-5
    assert pytest.approx(joint.link_len_a) == 72.0
    assert pytest.approx(joint.link_len_b) == 54.0
    assert pytest.approx(joint.twist_angle_alpha) == 82.0
    assert pytest.approx(joint.offset_distance_s) == 26.0


def test_m372_lagmul_wohlhart_hybrid_spatial_linkage_joint_aliases(tmp_path: Path):
    deck = """# RADIOSS ALIAS DECK
/BEGIN
Wohlhart Hybrid Spatial Linkage Joint Aliases Test
/WOHLHART_HYBRID_SPATIAL_LINKAGE_JOINT/117
351, 352, 353, 1.0e6, 0, 1.0e-6
40.0, 30.0, 40.0, 10.0
/LAGMUL/WOHLHART_HYBRID_SPATIAL_LINKAGE/118
351, 352, 353, 1.0e6, 0, 1.0e-6
40.0, 30.0, 40.0, 10.0
/WOHLHART_HYBRID_SPATIAL_LINKAGE/119
351, 352, 353, 1.0e6, 0, 1.0e-6
40.0, 30.0, 40.0, 10.0
/WOHLHART_HYBRID_SPATIAL_MULTI_LOOP_MECHANISM/120
351, 352, 353, 1.0e6, 0, 1.0e-6
40.0, 30.0, 40.0, 10.0
/WOHLHART_HYBRID_SPATIAL_SYMMETRIC_MECHANISM/121
351, 352, 353, 1.0e6, 0, 1.0e-6
40.0, 30.0, 40.0, 10.0
/WOHLHART_HYBRID_SPATIAL_6R_MECHANISM/122
351, 352, 353, 1.0e6, 0, 1.0e-6
40.0, 30.0, 40.0, 10.0
/WOHLHART_HYBRID_SPATIAL_OVERCONSTRAINED_MECHANISM/123
351, 352, 353, 1.0e6, 0, 1.0e-6
40.0, 30.0, 40.0, 10.0
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    for jid in range(117, 124):
        assert jid in model.lagmul_wohlhart_hybrid_spatial_linkage_joints


def test_m372_lagmul_wohlhart_hybrid_spatial_linkage_joint_missing_card(tmp_path: Path):
    deck = """# RADIOSS MISSING CARD DECK
/BEGIN
/LAGMUL/WOHLHART_HYBRID_SPATIAL_LINKAGE_JOINT/124
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) > 0


def test_m372_sensor_spring_total_angular_snap_rate_fixed(tmp_path: Path):
    c1 = f"{1198:>10d}{1.18e8:>20.4f}{0.0365:>20.4f}"
    deck = f"""# RADIOSS STARTER DECK
/BEGIN
Sensor Spring Total Angular Snap Rate Fixed Format Test
2022 0
/SENSOR/SPRING_TOTAL_ANGULAR_SNAP_RATE/1
Fixed Spring Total Angular Snap Rate Sensor
{c1}
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 1 in model.sensor_spring_total_angular_snap_rates
    s1 = model.sensor_spring_total_angular_snap_rates[1]
    assert s1.spring_id == 1198
    assert pytest.approx(s1.jtot_ang_snp_max) == 1.18e8
    assert pytest.approx(s1.jtot_ang_crackle_max) == 1.18e8
    assert pytest.approx(s1.t_delay) == 0.0365
    assert len(model.sensors) == 1
    assert model.sensors[0].kind == "SPRING_TOTAL_ANGULAR_SNAP_RATE"


def test_m372_sensor_spring_total_angular_snap_rate_free(tmp_path: Path):
    deck = """# RADIOSS FREE DECK
/BEGIN
Sensor Spring Total Angular Snap Rate Free Format Test
/SENSOR/SPRING_TOTAL_ANGULAR_SNAP_RATE/2
Free Spring Total Angular Snap Rate Sensor
1199, 1.78e8, 0.0515
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 2 in model.sensor_spring_total_angular_snap_rates
    s2 = model.sensor_spring_total_angular_snap_rates[2]
    assert s2.spring_id == 1199
    assert pytest.approx(s2.jtot_ang_snp_max) == 1.78e8
    assert pytest.approx(s2.t_delay) == 0.0515


def test_m372_sensor_spring_total_angular_snap_rate_aliases(tmp_path: Path):
    deck = """# RADIOSS ALIAS DECK
/BEGIN
Sensor Spring Total Angular Snap Rate Aliases Test
/SENSOR/SPRING_TOT_ANG_SNAP_RATE/220
1219, 7.85e7, 0.0105
/SENSOR/SPRING_RATE_SNAP_ANG_TOT/221
1220, 8.05e7, 0.0115
/SENSOR/TOTAL_ANGULAR_SNAP_RATE_SPRING/222
1221, 8.25e7, 0.0125
/SENSOR/SPRING_SNAP_ANG_TOT/223
1222, 8.45e7, 0.0135
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    for sid in range(220, 224):
        assert sid in model.sensor_spring_total_angular_snap_rates


def test_m372_sensor_spring_total_angular_snap_rate_missing_card(tmp_path: Path):
    deck = """# RADIOSS MISSING CARD DECK
/BEGIN
/SENSOR/SPRING_TOTAL_ANGULAR_SNAP_RATE/225
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) > 0
