"""Tests for Milestone M375: LadTransverseInterlaminarNormalPeelingRate Failure Model, EngFlexomagnetoplasmonicexcitonicpolaritonicResonanceEnergy, SturgessHybridSpatialLinkageJoint, and SensorSpringTotalPopRate."""

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


def test_m375_fail_lad_transverse_interlaminar_normal_peeling_rate_fixed(tmp_path: Path):
    c1 = f"{185.0:>20.4f}{555.0:>20.4f}{84.0:>20.4f}{3.25:>20.4f}{0.955:>20.4f}"
    c2 = f"{1:>10d}{2:>10d}"
    c3 = f"{1890:>10d}"
    deck = f"""# RADIOSS STARTER DECK
/BEGIN
Ladeveze Transverse Interlaminar Normal Peeling Rate Fixed Format Test
2022 0
/FAIL/LAD_TRANSVERSE_INTERLAMINAR_NORMAL_PEELING_RATE/1890
Ladeveze Transverse Interlaminar Normal Peeling Failure Model
{c1}
{c2}
{c3}
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 1890 in model.fail_ladtransverseinterlaminarnormalpeelingrates
    ftinpr = model.fail_ladtransverseinterlaminarnormalpeelingrates[1890]
    assert pytest.approx(ftinpr.sigma_tinpr0) == 185.0
    assert pytest.approx(ftinpr.sigma_tinprc) == 555.0
    assert pytest.approx(ftinpr.gamma_tinpr) == 84.0
    assert pytest.approx(ftinpr.p_tinpr) == 3.25
    assert pytest.approx(ftinpr.d_tinpr_max) == 0.955
    assert ftinpr.ifail_sh == 1
    assert ftinpr.ifail_so == 2
    assert ftinpr.fail_id == 1890
    assert len(model.raw_fails) == 1
    assert model.raw_fails[0][1].type == "LAD_TRANSVERSE_INTERLAMINAR_NORMAL_PEELING_RATE"


def test_m375_fail_lad_transverse_interlaminar_normal_peeling_rate_free(tmp_path: Path):
    deck = """# RADIOSS FREE DECK
/BEGIN
Ladeveze Transverse Interlaminar Normal Peeling Rate Free Format Test
/FAIL/LAD_TRANSVERSE_INTERLAMINAR_NORMAL_PEELING_RATE/1891
195.0, 585.0, 90.0, 3.45, 0.930
1, 1
1891
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 1891 in model.fail_ladtransverseinterlaminarnormalpeelingrates
    ftinpr = model.fail_ladtransverseinterlaminarnormalpeelingrates[1891]
    assert pytest.approx(ftinpr.sigma_tinpr0) == 195.0
    assert pytest.approx(ftinpr.sigma_tinprc) == 585.0
    assert pytest.approx(ftinpr.gamma_tinpr) == 90.0
    assert pytest.approx(ftinpr.p_tinpr) == 3.45
    assert pytest.approx(ftinpr.d_tinpr_max) == 0.930
    assert ftinpr.fail_id == 1891


def test_m375_fail_lad_transverse_interlaminar_normal_peeling_rate_aliases(tmp_path: Path):
    deck = """# RADIOSS ALIAS DECK
/BEGIN
Ladeveze Transverse Interlaminar Normal Peeling Rate Aliases Test
/FAIL/LADEVEZE_TRANSVERSE_INTERLAMINAR_NORMAL_PEELING_RATE/1892
175.0, 525.0, 78.0, 2.95, 0.975
1, 1
/FAIL/LAD_TINPR/1893
175.0, 525.0, 78.0, 2.95, 0.975
1, 1
/FAIL/LAD_TINPR_MODEL/1894
175.0, 525.0, 78.0, 2.95, 0.975
1, 1
/FAIL/LAD_TINPR_LAW/1895
175.0, 525.0, 78.0, 2.95, 0.975
1, 1
/FAIL/LADEVEZE_RATE_DEPENDENT_TRANSVERSE_INTERLAMINAR_NORMAL_PEELING/1896
175.0, 525.0, 78.0, 2.95, 0.975
1, 1
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 1892 in model.fail_ladtransverseinterlaminarnormalpeelingrates
    assert 1893 in model.fail_ladtransverseinterlaminarnormalpeelingrates
    assert 1894 in model.fail_ladtransverseinterlaminarnormalpeelingrates
    assert 1895 in model.fail_ladtransverseinterlaminarnormalpeelingrates
    assert 1896 in model.fail_ladtransverseinterlaminarnormalpeelingrates
    assert len(model.raw_fails) == 5


def test_m375_fail_lad_transverse_interlaminar_normal_peeling_rate_missing_card(tmp_path: Path):
    deck = """# RADIOSS MISSING CARD DECK
/BEGIN
/FAIL/LAD_TRANSVERSE_INTERLAMINAR_NORMAL_PEELING_RATE/1897
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) > 0


def test_m375_eng_flexomagnetoplasmonicexcitonicpolaritonic_resonance_energy_fixed(tmp_path: Path):
    c1 = f"{0.0048:>20.4f}{195:>10d}"
    deck = f"""# RADIOSS STARTER DECK
/BEGIN
Engine Flexomagnetoplasmonicexcitonicpolaritonic Resonance Energy Fixed Format Test
2022 0
/ENG/FLEXOMAGNETOPLASMONICEXCITONICPOLARITONIC_RESONANCE_ENERGY/95
Resonance Energy Output Directive
{c1}
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 95 in model.eng_flexomagnetoplasmonicexcitonicpolaritonic_resonance_energies
    eng = model.eng_flexomagnetoplasmonicexcitonicpolaritonic_resonance_energies[95]
    assert pytest.approx(eng.dt_fmpepr) == 0.0048
    assert eng.sens_id == 195


def test_m375_eng_flexomagnetoplasmonicexcitonicpolaritonic_resonance_energy_free(tmp_path: Path):
    deck = """# RADIOSS FREE DECK
/BEGIN
Engine Flexomagnetoplasmonicexcitonicpolaritonic Resonance Energy Free Format Test
/ENG/FLEXOMAGNETOPLASMONICEXCITONICPOLARITONIC_RESONANCE_ENERGY/96
0.0058, 196
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 96 in model.eng_flexomagnetoplasmonicexcitonicpolaritonic_resonance_energies
    eng = model.eng_flexomagnetoplasmonicexcitonicpolaritonic_resonance_energies[96]
    assert pytest.approx(eng.dt_fmpepr) == 0.0058
    assert eng.sens_id == 196


def test_m375_eng_flexomagnetoplasmonicexcitonicpolaritonic_resonance_energy_aliases(tmp_path: Path):
    deck = """# RADIOSS ALIAS DECK
/BEGIN
Engine Flexomagnetoplasmonicexcitonicpolaritonic Resonance Energy Aliases Test
/ENG/FLEXOMAGNETO_PLASMON_EXCITON_POLARITON_RES_WORK/97
0.0068, 197
/ENG/EFLEXOMAGNETOPLASMONICEXCITONICPOLARITONICRESONANCE/98
0.0078, 198
/ENG/FLEXOMAGNETOPLASMONICEXCITONICPOLARITONIC_RESONANCE_DISSIPATION/99
0.0088, 199
/ENG/EM_FLEXOMAGNETOPLASMONICEXCITONICPOLARITONIC_RESONANCE/100
0.0098, 200
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 97 in model.eng_flexomagnetoplasmonicexcitonicpolaritonic_resonance_energies
    assert 98 in model.eng_flexomagnetoplasmonicexcitonicpolaritonic_resonance_energies
    assert 99 in model.eng_flexomagnetoplasmonicexcitonicpolaritonic_resonance_energies
    assert 100 in model.eng_flexomagnetoplasmonicexcitonicpolaritonic_resonance_energies


def test_m375_eng_flexomagnetoplasmonicexcitonicpolaritonic_resonance_energy_missing_card(tmp_path: Path):
    deck = """# RADIOSS MISSING CARD DECK
/BEGIN
/ENG/FLEXOMAGNETOPLASMONICEXCITONICPOLARITONIC_RESONANCE_ENERGY/101
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) > 0


def test_m375_lagmul_sturgess_hybrid_spatial_linkage_joint_fixed(tmp_path: Path):
    c1 = f"{181:>10d}{182:>10d}{183:>10d}{8.5e6:>20.4f}{11:>10d}{3.4e-5:>20.4e}"
    c2 = f"{72.0:>20.4f}{54.0:>20.4f}{74.0:>20.4f}{23.0:>20.4f}"
    deck = f"""# RADIOSS STARTER DECK
/BEGIN
Sturgess Hybrid Spatial Linkage Joint Fixed Format Test
2022 0
/LAGMUL/STURGESS_HYBRID_SPATIAL_LINKAGE_JOINT/145
Sturgess Hybrid Spatial 6R Joint
{c1}
{c2}
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 145 in model.lagmul_sturgess_hybrid_spatial_linkage_joints
    joint = model.lagmul_sturgess_hybrid_spatial_linkage_joints[145]
    assert joint.node1 == 181
    assert joint.node2 == 182
    assert joint.node3 == 183
    assert pytest.approx(joint.stiff) == 8.5e6
    assert joint.skew_id == 11
    assert pytest.approx(joint.tol) == 3.4e-5
    assert pytest.approx(joint.link_len_a) == 72.0
    assert pytest.approx(joint.link_len_b) == 54.0
    assert pytest.approx(joint.twist_angle_alpha) == 74.0
    assert pytest.approx(joint.offset_distance_s) == 23.0


def test_m375_lagmul_sturgess_hybrid_spatial_linkage_joint_free(tmp_path: Path):
    deck = """# RADIOSS FREE DECK
/BEGIN
Sturgess Hybrid Spatial Linkage Joint Free Format Test
/LAGMUL/STURGESS_HYBRID_SPATIAL_LINKAGE_JOINT/146
281, 282, 283, 1.0e7, 13, 4.4e-5
84.0, 60.0, 94.0, 32.0
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 146 in model.lagmul_sturgess_hybrid_spatial_linkage_joints
    joint = model.lagmul_sturgess_hybrid_spatial_linkage_joints[146]
    assert joint.node1 == 281
    assert joint.node2 == 282
    assert joint.node3 == 283
    assert pytest.approx(joint.stiff) == 1.0e7
    assert joint.skew_id == 13
    assert pytest.approx(joint.tol) == 4.4e-5
    assert pytest.approx(joint.link_len_a) == 84.0
    assert pytest.approx(joint.link_len_b) == 60.0
    assert pytest.approx(joint.twist_angle_alpha) == 94.0
    assert pytest.approx(joint.offset_distance_s) == 32.0


def test_m375_lagmul_sturgess_hybrid_spatial_linkage_joint_aliases(tmp_path: Path):
    deck = """# RADIOSS ALIAS DECK
/BEGIN
Sturgess Hybrid Spatial Linkage Joint Aliases Test
/STURGESS_HYBRID_SPATIAL_LINKAGE_JOINT/147
381, 382, 383, 1.0e6, 0, 1.0e-6
46.0, 36.0, 46.0, 13.0
/LAGMUL/STURGESS_HYBRID_SPATIAL_LINKAGE/148
381, 382, 383, 1.0e6, 0, 1.0e-6
46.0, 36.0, 46.0, 13.0
/STURGESS_HYBRID_SPATIAL_LINKAGE/149
381, 382, 383, 1.0e6, 0, 1.0e-6
46.0, 36.0, 46.0, 13.0
/STURGESS_HYBRID_SPATIAL_MULTI_LOOP_MECHANISM/150
381, 382, 383, 1.0e6, 0, 1.0e-6
46.0, 36.0, 46.0, 13.0
/STURGESS_HYBRID_SPATIAL_SYMMETRIC_MECHANISM/151
381, 382, 383, 1.0e6, 0, 1.0e-6
46.0, 36.0, 46.0, 13.0
/STURGESS_HYBRID_SPATIAL_6R_MECHANISM/152
381, 382, 383, 1.0e6, 0, 1.0e-6
46.0, 36.0, 46.0, 13.0
/STURGESS_HYBRID_SPATIAL_OVERCONSTRAINED_MECHANISM/153
381, 382, 383, 1.0e6, 0, 1.0e-6
46.0, 36.0, 46.0, 13.0
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    for jid in range(147, 154):
        assert jid in model.lagmul_sturgess_hybrid_spatial_linkage_joints


def test_m375_lagmul_sturgess_hybrid_spatial_linkage_joint_missing_card(tmp_path: Path):
    deck = """# RADIOSS MISSING CARD DECK
/BEGIN
/LAGMUL/STURGESS_HYBRID_SPATIAL_LINKAGE_JOINT/154
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) > 0


def test_m375_sensor_spring_total_pop_rate_fixed(tmp_path: Path):
    c1 = f"{1228:>10d}{1.48e8:>20.4f}{0.0455:>20.4f}"
    deck = f"""# RADIOSS STARTER DECK
/BEGIN
Sensor Spring Total Pop Rate Fixed Format Test
2022 0
/SENSOR/SPRING_TOTAL_POP_RATE/1
Fixed Spring Total Pop Rate Sensor
{c1}
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 1 in model.sensor_spring_total_pop_rates
    s1 = model.sensor_spring_total_pop_rates[1]
    assert s1.spring_id == 1228
    assert pytest.approx(s1.jtot_pop_max) == 1.48e8
    assert pytest.approx(s1.jtot_snp_max) == 1.48e8
    assert pytest.approx(s1.jtot_crackle_max) == 1.48e8
    assert pytest.approx(s1.t_delay) == 0.0455
    assert len(model.sensors) == 1
    assert model.sensors[0].kind == "SPRING_TOTAL_POP_RATE"


def test_m375_sensor_spring_total_pop_rate_free(tmp_path: Path):
    deck = """# RADIOSS FREE DECK
/BEGIN
Sensor Spring Total Pop Rate Free Format Test
/SENSOR/SPRING_TOTAL_POP_RATE/2
Free Spring Total Pop Rate Sensor
1229, 2.08e8, 0.0605
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 2 in model.sensor_spring_total_pop_rates
    s2 = model.sensor_spring_total_pop_rates[2]
    assert s2.spring_id == 1229
    assert pytest.approx(s2.jtot_pop_max) == 2.08e8
    assert pytest.approx(s2.t_delay) == 0.0605


def test_m375_sensor_spring_total_pop_rate_aliases(tmp_path: Path):
    deck = """# RADIOSS ALIAS DECK
/BEGIN
Sensor Spring Total Pop Rate Aliases Test
/SENSOR/SPRING_TOT_POP_RATE/250
1249, 9.35e7, 0.0116
/SENSOR/SPRING_RATE_POP_TOT/251
1250, 9.55e7, 0.0126
/SENSOR/TOTAL_POP_RATE_SPRING/252
1251, 9.75e7, 0.0136
/SENSOR/SPRING_POP_TOT/253
1252, 9.95e7, 0.0146
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    for sid in range(250, 254):
        assert sid in model.sensor_spring_total_pop_rates


def test_m375_sensor_spring_total_pop_rate_missing_card(tmp_path: Path):
    deck = """# RADIOSS MISSING CARD DECK
/BEGIN
/SENSOR/SPRING_TOTAL_POP_RATE/255
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) > 0
