"""Tests for Milestone M374: LadDynamicInterlaminarNormalPeelingRate Failure Model, EngFlexomagnetoplasmonicexcitonicmagnonicResonanceEnergy, KrauseHybridSpatialLinkageJoint, and SensorSpringTransversePopRate."""

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


def test_m374_fail_lad_dynamic_interlaminar_normal_peeling_rate_fixed(tmp_path: Path):
    c1 = f"{175.0:>20.4f}{525.0:>20.4f}{80.0:>20.4f}{3.15:>20.4f}{0.958:>20.4f}"
    c2 = f"{1:>10d}{2:>10d}"
    c3 = f"{1880:>10d}"
    deck = f"""# RADIOSS STARTER DECK
/BEGIN
Ladeveze Dynamic Interlaminar Normal Peeling Rate Fixed Format Test
2022 0
/FAIL/LAD_DYNAMIC_INTERLAMINAR_NORMAL_PEELING_RATE/1880
Ladeveze Dynamic Interlaminar Normal Peeling Failure Model
{c1}
{c2}
{c3}
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 1880 in model.fail_laddynamicinterlaminarnormalpeelingrates
    fdinpr = model.fail_laddynamicinterlaminarnormalpeelingrates[1880]
    assert pytest.approx(fdinpr.sigma_dinpr0) == 175.0
    assert pytest.approx(fdinpr.sigma_dinprc) == 525.0
    assert pytest.approx(fdinpr.gamma_dinpr) == 80.0
    assert pytest.approx(fdinpr.p_dinpr) == 3.15
    assert pytest.approx(fdinpr.d_dinpr_max) == 0.958
    assert fdinpr.ifail_sh == 1
    assert fdinpr.ifail_so == 2
    assert fdinpr.fail_id == 1880
    assert len(model.raw_fails) == 1
    assert model.raw_fails[0][1].type == "LAD_DYNAMIC_INTERLAMINAR_NORMAL_PEELING_RATE"


def test_m374_fail_lad_dynamic_interlaminar_normal_peeling_rate_free(tmp_path: Path):
    deck = """# RADIOSS FREE DECK
/BEGIN
Ladeveze Dynamic Interlaminar Normal Peeling Rate Free Format Test
/FAIL/LAD_DYNAMIC_INTERLAMINAR_NORMAL_PEELING_RATE/1881
185.0, 555.0, 86.0, 3.35, 0.935
1, 1
1881
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 1881 in model.fail_laddynamicinterlaminarnormalpeelingrates
    fdinpr = model.fail_laddynamicinterlaminarnormalpeelingrates[1881]
    assert pytest.approx(fdinpr.sigma_dinpr0) == 185.0
    assert pytest.approx(fdinpr.sigma_dinprc) == 555.0
    assert pytest.approx(fdinpr.gamma_dinpr) == 86.0
    assert pytest.approx(fdinpr.p_dinpr) == 3.35
    assert pytest.approx(fdinpr.d_dinpr_max) == 0.935
    assert fdinpr.fail_id == 1881


def test_m374_fail_lad_dynamic_interlaminar_normal_peeling_rate_aliases(tmp_path: Path):
    deck = """# RADIOSS ALIAS DECK
/BEGIN
Ladeveze Dynamic Interlaminar Normal Peeling Rate Aliases Test
/FAIL/LADEVEZE_DYNAMIC_INTERLAMINAR_NORMAL_PEELING_RATE/1882
165.0, 495.0, 74.0, 2.85, 0.978
1, 1
/FAIL/LAD_DINPR/1883
165.0, 495.0, 74.0, 2.85, 0.978
1, 1
/FAIL/LAD_DINPR_MODEL/1884
165.0, 495.0, 74.0, 2.85, 0.978
1, 1
/FAIL/LAD_DINPR_LAW/1885
165.0, 495.0, 74.0, 2.85, 0.978
1, 1
/FAIL/LADEVEZE_RATE_DEPENDENT_DYNAMIC_INTERLAMINAR_NORMAL_PEELING/1886
165.0, 495.0, 74.0, 2.85, 0.978
1, 1
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 1882 in model.fail_laddynamicinterlaminarnormalpeelingrates
    assert 1883 in model.fail_laddynamicinterlaminarnormalpeelingrates
    assert 1884 in model.fail_laddynamicinterlaminarnormalpeelingrates
    assert 1885 in model.fail_laddynamicinterlaminarnormalpeelingrates
    assert 1886 in model.fail_laddynamicinterlaminarnormalpeelingrates
    assert len(model.raw_fails) == 5


def test_m374_fail_lad_dynamic_interlaminar_normal_peeling_rate_missing_card(tmp_path: Path):
    deck = """# RADIOSS MISSING CARD DECK
/BEGIN
/FAIL/LAD_DYNAMIC_INTERLAMINAR_NORMAL_PEELING_RATE/1887
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) > 0


def test_m374_eng_flexomagnetoplasmonicexcitonicmagnonic_resonance_energy_fixed(tmp_path: Path):
    c1 = f"{0.0045:>20.4f}{185:>10d}"
    deck = f"""# RADIOSS STARTER DECK
/BEGIN
Engine Flexomagnetoplasmonicexcitonicmagnonic Resonance Energy Fixed Format Test
2022 0
/ENG/FLEXOMAGNETOPLASMONICEXCITONICMAGNONIC_RESONANCE_ENERGY/85
Resonance Energy Output Directive
{c1}
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 85 in model.eng_flexomagnetoplasmonicexcitonicmagnonic_resonance_energies
    eng = model.eng_flexomagnetoplasmonicexcitonicmagnonic_resonance_energies[85]
    assert pytest.approx(eng.dt_fmpemr) == 0.0045
    assert eng.sens_id == 185


def test_m374_eng_flexomagnetoplasmonicexcitonicmagnonic_resonance_energy_free(tmp_path: Path):
    deck = """# RADIOSS FREE DECK
/BEGIN
Engine Flexomagnetoplasmonicexcitonicmagnonic Resonance Energy Free Format Test
/ENG/FLEXOMAGNETOPLASMONICEXCITONICMAGNONIC_RESONANCE_ENERGY/86
0.0055, 186
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 86 in model.eng_flexomagnetoplasmonicexcitonicmagnonic_resonance_energies
    eng = model.eng_flexomagnetoplasmonicexcitonicmagnonic_resonance_energies[86]
    assert pytest.approx(eng.dt_fmpemr) == 0.0055
    assert eng.sens_id == 186


def test_m374_eng_flexomagnetoplasmonicexcitonicmagnonic_resonance_energy_aliases(tmp_path: Path):
    deck = """# RADIOSS ALIAS DECK
/BEGIN
Engine Flexomagnetoplasmonicexcitonicmagnonic Resonance Energy Aliases Test
/ENG/FLEXOMAGNETO_PLASMON_EXCITON_MAGNON_RES_WORK/87
0.0065, 187
/ENG/EFLEXOMAGNETOPLASMONICEXCITONICMAGNONICRESONANCE/88
0.0075, 188
/ENG/FLEXOMAGNETOPLASMONICEXCITONICMAGNONIC_RESONANCE_DISSIPATION/89
0.0085, 189
/ENG/EM_FLEXOMAGNETOPLASMONICEXCITONICMAGNONIC_RESONANCE/90
0.0095, 190
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 87 in model.eng_flexomagnetoplasmonicexcitonicmagnonic_resonance_energies
    assert 88 in model.eng_flexomagnetoplasmonicexcitonicmagnonic_resonance_energies
    assert 89 in model.eng_flexomagnetoplasmonicexcitonicmagnonic_resonance_energies
    assert 90 in model.eng_flexomagnetoplasmonicexcitonicmagnonic_resonance_energies


def test_m374_eng_flexomagnetoplasmonicexcitonicmagnonic_resonance_energy_missing_card(tmp_path: Path):
    deck = """# RADIOSS MISSING CARD DECK
/BEGIN
/ENG/FLEXOMAGNETOPLASMONICEXCITONICMAGNONIC_RESONANCE_ENERGY/91
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) > 0


def test_m374_lagmul_krause_hybrid_spatial_linkage_joint_fixed(tmp_path: Path):
    c1 = f"{171:>10d}{172:>10d}{173:>10d}{8.0e6:>20.4f}{10:>10d}{3.2e-5:>20.4e}"
    c2 = f"{70.0:>20.4f}{52.0:>20.4f}{70.0:>20.4f}{22.0:>20.4f}"
    deck = f"""# RADIOSS STARTER DECK
/BEGIN
Krause Hybrid Spatial Linkage Joint Fixed Format Test
2022 0
/LAGMUL/KRAUSE_HYBRID_SPATIAL_LINKAGE_JOINT/135
Krause Hybrid Spatial 6R Joint
{c1}
{c2}
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 135 in model.lagmul_krause_hybrid_spatial_linkage_joints
    joint = model.lagmul_krause_hybrid_spatial_linkage_joints[135]
    assert joint.node1 == 171
    assert joint.node2 == 172
    assert joint.node3 == 173
    assert pytest.approx(joint.stiff) == 8.0e6
    assert joint.skew_id == 10
    assert pytest.approx(joint.tol) == 3.2e-5
    assert pytest.approx(joint.link_len_a) == 70.0
    assert pytest.approx(joint.link_len_b) == 52.0
    assert pytest.approx(joint.twist_angle_alpha) == 70.0
    assert pytest.approx(joint.offset_distance_s) == 22.0


def test_m374_lagmul_krause_hybrid_spatial_linkage_joint_free(tmp_path: Path):
    deck = """# RADIOSS FREE DECK
/BEGIN
Krause Hybrid Spatial Linkage Joint Free Format Test
/LAGMUL/KRAUSE_HYBRID_SPATIAL_LINKAGE_JOINT/136
271, 272, 273, 9.5e6, 12, 4.2e-5
80.0, 58.0, 90.0, 30.0
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 136 in model.lagmul_krause_hybrid_spatial_linkage_joints
    joint = model.lagmul_krause_hybrid_spatial_linkage_joints[136]
    assert joint.node1 == 271
    assert joint.node2 == 272
    assert joint.node3 == 273
    assert pytest.approx(joint.stiff) == 9.5e6
    assert joint.skew_id == 12
    assert pytest.approx(joint.tol) == 4.2e-5
    assert pytest.approx(joint.link_len_a) == 80.0
    assert pytest.approx(joint.link_len_b) == 58.0
    assert pytest.approx(joint.twist_angle_alpha) == 90.0
    assert pytest.approx(joint.offset_distance_s) == 30.0


def test_m374_lagmul_krause_hybrid_spatial_linkage_joint_aliases(tmp_path: Path):
    deck = """# RADIOSS ALIAS DECK
/BEGIN
Krause Hybrid Spatial Linkage Joint Aliases Test
/KRAUSE_HYBRID_SPATIAL_LINKAGE_JOINT/137
371, 372, 373, 1.0e6, 0, 1.0e-6
44.0, 34.0, 44.0, 12.0
/LAGMUL/KRAUSE_HYBRID_SPATIAL_LINKAGE/138
371, 372, 373, 1.0e6, 0, 1.0e-6
44.0, 34.0, 44.0, 12.0
/KRAUSE_HYBRID_SPATIAL_LINKAGE/139
371, 372, 373, 1.0e6, 0, 1.0e-6
44.0, 34.0, 44.0, 12.0
/KRAUSE_HYBRID_SPATIAL_MULTI_LOOP_MECHANISM/140
371, 372, 373, 1.0e6, 0, 1.0e-6
44.0, 34.0, 44.0, 12.0
/KRAUSE_HYBRID_SPATIAL_SYMMETRIC_MECHANISM/141
371, 372, 373, 1.0e6, 0, 1.0e-6
44.0, 34.0, 44.0, 12.0
/KRAUSE_HYBRID_SPATIAL_6R_MECHANISM/142
371, 372, 373, 1.0e6, 0, 1.0e-6
44.0, 34.0, 44.0, 12.0
/KRAUSE_HYBRID_SPATIAL_OVERCONSTRAINED_MECHANISM/143
371, 372, 373, 1.0e6, 0, 1.0e-6
44.0, 34.0, 44.0, 12.0
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    for jid in range(137, 144):
        assert jid in model.lagmul_krause_hybrid_spatial_linkage_joints


def test_m374_lagmul_krause_hybrid_spatial_linkage_joint_missing_card(tmp_path: Path):
    deck = """# RADIOSS MISSING CARD DECK
/BEGIN
/LAGMUL/KRAUSE_HYBRID_SPATIAL_LINKAGE_JOINT/144
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) > 0


def test_m374_sensor_spring_transverse_pop_rate_fixed(tmp_path: Path):
    c1 = f"{1218:>10d}{1.38e8:>20.4f}{0.0425:>20.4f}"
    deck = f"""# RADIOSS STARTER DECK
/BEGIN
Sensor Spring Transverse Pop Rate Fixed Format Test
2022 0
/SENSOR/SPRING_TRANSVERSE_POP_RATE/1
Fixed Spring Transverse Pop Rate Sensor
{c1}
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 1 in model.sensor_spring_transverse_pop_rates
    s1 = model.sensor_spring_transverse_pop_rates[1]
    assert s1.spring_id == 1218
    assert pytest.approx(s1.jtrans_pop_max) == 1.38e8
    assert pytest.approx(s1.jtrans_snp_max) == 1.38e8
    assert pytest.approx(s1.jtrans_crackle_max) == 1.38e8
    assert pytest.approx(s1.t_delay) == 0.0425
    assert len(model.sensors) == 1
    assert model.sensors[0].kind == "SPRING_TRANSVERSE_POP_RATE"


def test_m374_sensor_spring_transverse_pop_rate_free(tmp_path: Path):
    deck = """# RADIOSS FREE DECK
/BEGIN
Sensor Spring Transverse Pop Rate Free Format Test
/SENSOR/SPRING_TRANSVERSE_POP_RATE/2
Free Spring Transverse Pop Rate Sensor
1219, 1.98e8, 0.0575
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 2 in model.sensor_spring_transverse_pop_rates
    s2 = model.sensor_spring_transverse_pop_rates[2]
    assert s2.spring_id == 1219
    assert pytest.approx(s2.jtrans_pop_max) == 1.98e8
    assert pytest.approx(s2.t_delay) == 0.0575


def test_m374_sensor_spring_transverse_pop_rate_aliases(tmp_path: Path):
    deck = """# RADIOSS ALIAS DECK
/BEGIN
Sensor Spring Transverse Pop Rate Aliases Test
/SENSOR/SPRING_TRANS_POP_RATE/240
1239, 8.85e7, 0.0112
/SENSOR/SPRING_RATE_POP_TRANS/241
1240, 9.05e7, 0.0122
/SENSOR/TRANSVERSE_POP_RATE_SPRING/242
1241, 9.25e7, 0.0132
/SENSOR/SPRING_POP_TRANS/243
1242, 9.45e7, 0.0142
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    for sid in range(240, 244):
        assert sid in model.sensor_spring_transverse_pop_rates


def test_m374_sensor_spring_transverse_pop_rate_missing_card(tmp_path: Path):
    deck = """# RADIOSS MISSING CARD DECK
/BEGIN
/SENSOR/SPRING_TRANSVERSE_POP_RATE/245
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) > 0
