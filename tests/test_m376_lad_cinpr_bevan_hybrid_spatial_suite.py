"""Tests for Milestone M376: LadCoupleInterlaminarNormalPeelingRate Failure Model, EngFlexomagnetoplasmonicmagnonicpolaritonicResonanceEnergy, BevanHybridSpatialLinkageJoint, and SensorSpringTorsionalPopRate."""

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


def test_m376_fail_lad_couple_interlaminar_normal_peeling_rate_fixed(tmp_path: Path):
    c1 = f"{195.0:>20.4f}{585.0:>20.4f}{88.0:>20.4f}{3.35:>20.4f}{0.950:>20.4f}"
    c2 = f"{1:>10d}{2:>10d}"
    c3 = f"{1900:>10d}"
    deck = f"""# RADIOSS STARTER DECK
/BEGIN
Ladeveze Coupled Interlaminar Normal Peeling Rate Fixed Format Test
2022 0
/FAIL/LAD_COUPLE_INTERLAMINAR_NORMAL_PEELING_RATE/1900
Ladeveze Coupled Interlaminar Normal Peeling Failure Model
{c1}
{c2}
{c3}
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 1900 in model.fail_ladcoupleinterlaminarnormalpeelingrates
    fcinpr = model.fail_ladcoupleinterlaminarnormalpeelingrates[1900]
    assert pytest.approx(fcinpr.sigma_cinpr0) == 195.0
    assert pytest.approx(fcinpr.sigma_cinprc) == 585.0
    assert pytest.approx(fcinpr.gamma_cinpr) == 88.0
    assert pytest.approx(fcinpr.p_cinpr) == 3.35
    assert pytest.approx(fcinpr.d_cinpr_max) == 0.950
    assert fcinpr.ifail_sh == 1
    assert fcinpr.ifail_so == 2
    assert fcinpr.fail_id == 1900
    assert len(model.raw_fails) == 1
    assert model.raw_fails[0][1].type == "LAD_COUPLE_INTERLAMINAR_NORMAL_PEELING_RATE"


def test_m376_fail_lad_couple_interlaminar_normal_peeling_rate_free(tmp_path: Path):
    deck = """# RADIOSS FREE DECK
/BEGIN
Ladeveze Coupled Interlaminar Normal Peeling Rate Free Format Test
/FAIL/LAD_COUPLE_INTERLAMINAR_NORMAL_PEELING_RATE/1901
205.0, 615.0, 94.0, 3.55, 0.925
1, 1
1901
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 1901 in model.fail_ladcoupleinterlaminarnormalpeelingrates
    fcinpr = model.fail_ladcoupleinterlaminarnormalpeelingrates[1901]
    assert pytest.approx(fcinpr.sigma_cinpr0) == 205.0
    assert pytest.approx(fcinpr.sigma_cinprc) == 615.0
    assert pytest.approx(fcinpr.gamma_cinpr) == 94.0
    assert pytest.approx(fcinpr.p_cinpr) == 3.55
    assert pytest.approx(fcinpr.d_cinpr_max) == 0.925
    assert fcinpr.fail_id == 1901


def test_m376_fail_lad_couple_interlaminar_normal_peeling_rate_aliases(tmp_path: Path):
    deck = """# RADIOSS ALIAS DECK
/BEGIN
Ladeveze Coupled Interlaminar Normal Peeling Rate Aliases Test
/FAIL/LADEVEZE_COUPLED_INTERLAMINAR_NORMAL_PEELING_RATE/1902
185.0, 555.0, 82.0, 3.05, 0.970
1, 1
/FAIL/LAD_CINPR/1903
185.0, 555.0, 82.0, 3.05, 0.970
1, 1
/FAIL/LAD_CINPR_MODEL/1904
185.0, 555.0, 82.0, 3.05, 0.970
1, 1
/FAIL/LAD_CINPR_LAW/1905
185.0, 555.0, 82.0, 3.05, 0.970
1, 1
/FAIL/LADEVEZE_RATE_DEPENDENT_COUPLED_INTERLAMINAR_NORMAL_PEELING/1906
185.0, 555.0, 82.0, 3.05, 0.970
1, 1
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 1902 in model.fail_ladcoupleinterlaminarnormalpeelingrates
    assert 1903 in model.fail_ladcoupleinterlaminarnormalpeelingrates
    assert 1904 in model.fail_ladcoupleinterlaminarnormalpeelingrates
    assert 1905 in model.fail_ladcoupleinterlaminarnormalpeelingrates
    assert 1906 in model.fail_ladcoupleinterlaminarnormalpeelingrates
    assert len(model.raw_fails) == 5


def test_m376_fail_lad_couple_interlaminar_normal_peeling_rate_missing_card(tmp_path: Path):
    deck = """# RADIOSS MISSING CARD DECK
/BEGIN
/FAIL/LAD_COUPLE_INTERLAMINAR_NORMAL_PEELING_RATE/1907
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) > 0


def test_m376_eng_flexomagnetoplasmonicmagnonicpolaritonic_resonance_energy_fixed(tmp_path: Path):
    c1 = f"{0.0052:>20.4f}{205:>10d}"
    deck = f"""# RADIOSS STARTER DECK
/BEGIN
Engine Flexomagnetoplasmonicmagnonicpolaritonic Resonance Energy Fixed Format Test
2022 0
/ENG/FLEXOMAGNETOPLASMONICMAGNONICPOLARITONIC_RESONANCE_ENERGY/105
Resonance Energy Output Directive
{c1}
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 105 in model.eng_flexomagnetoplasmonicmagnonicpolaritonic_resonance_energies
    eng = model.eng_flexomagnetoplasmonicmagnonicpolaritonic_resonance_energies[105]
    assert pytest.approx(eng.dt_fmpmpr) == 0.0052
    assert eng.sens_id == 205


def test_m376_eng_flexomagnetoplasmonicmagnonicpolaritonic_resonance_energy_free(tmp_path: Path):
    deck = """# RADIOSS FREE DECK
/BEGIN
Engine Flexomagnetoplasmonicmagnonicpolaritonic Resonance Energy Free Format Test
/ENG/FLEXOMAGNETOPLASMONICMAGNONICPOLARITONIC_RESONANCE_ENERGY/106
0.0062, 206
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 106 in model.eng_flexomagnetoplasmonicmagnonicpolaritonic_resonance_energies
    eng = model.eng_flexomagnetoplasmonicmagnonicpolaritonic_resonance_energies[106]
    assert pytest.approx(eng.dt_fmpmpr) == 0.0062
    assert eng.sens_id == 206


def test_m376_eng_flexomagnetoplasmonicmagnonicpolaritonic_resonance_energy_aliases(tmp_path: Path):
    deck = """# RADIOSS ALIAS DECK
/BEGIN
Engine Flexomagnetoplasmonicmagnonicpolaritonic Resonance Energy Aliases Test
/ENG/FLEXOMAGNETO_PLASMON_MAGNON_POLARITON_RES_WORK/107
0.0072, 207
/ENG/EFLEXOMAGNETOPLASMONICMAGNONICPOLARITONICRESONANCE/108
0.0082, 208
/ENG/FLEXOMAGNETOPLASMONICMAGNONICPOLARITONIC_RESONANCE_DISSIPATION/109
0.0092, 209
/ENG/EM_FLEXOMAGNETOPLASMONICMAGNONICPOLARITONIC_RESONANCE/110
0.0102, 210
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 107 in model.eng_flexomagnetoplasmonicmagnonicpolaritonic_resonance_energies
    assert 108 in model.eng_flexomagnetoplasmonicmagnonicpolaritonic_resonance_energies
    assert 109 in model.eng_flexomagnetoplasmonicmagnonicpolaritonic_resonance_energies
    assert 110 in model.eng_flexomagnetoplasmonicmagnonicpolaritonic_resonance_energies


def test_m376_eng_flexomagnetoplasmonicmagnonicpolaritonic_resonance_energy_missing_card(tmp_path: Path):
    deck = """# RADIOSS MISSING CARD DECK
/BEGIN
/ENG/FLEXOMAGNETOPLASMONICMAGNONICPOLARITONIC_RESONANCE_ENERGY/111
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) > 0


def test_m376_lagmul_bevan_hybrid_spatial_linkage_joint_fixed(tmp_path: Path):
    c1 = f"{191:>10d}{192:>10d}{193:>10d}{9.0e6:>20.4f}{12:>10d}{3.6e-5:>20.4e}"
    c2 = f"{74.0:>20.4f}{56.0:>20.4f}{76.0:>20.4f}{24.0:>20.4f}"
    deck = f"""# RADIOSS STARTER DECK
/BEGIN
Bevan Hybrid Spatial Linkage Joint Fixed Format Test
2022 0
/LAGMUL/BEVAN_HYBRID_SPATIAL_LINKAGE_JOINT/155
Bevan Hybrid Spatial 6R Joint
{c1}
{c2}
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 155 in model.lagmul_bevan_hybrid_spatial_linkage_joints
    joint = model.lagmul_bevan_hybrid_spatial_linkage_joints[155]
    assert joint.node1 == 191
    assert joint.node2 == 192
    assert joint.node3 == 193
    assert pytest.approx(joint.stiff) == 9.0e6
    assert joint.skew_id == 12
    assert pytest.approx(joint.tol) == 3.6e-5
    assert pytest.approx(joint.link_len_a) == 74.0
    assert pytest.approx(joint.link_len_b) == 56.0
    assert pytest.approx(joint.twist_angle_alpha) == 76.0
    assert pytest.approx(joint.offset_distance_s) == 24.0


def test_m376_lagmul_bevan_hybrid_spatial_linkage_joint_free(tmp_path: Path):
    deck = """# RADIOSS FREE DECK
/BEGIN
Bevan Hybrid Spatial Linkage Joint Free Format Test
/LAGMUL/BEVAN_HYBRID_SPATIAL_LINKAGE_JOINT/156
291, 292, 293, 1.05e7, 14, 4.6e-5
86.0, 62.0, 96.0, 34.0
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 156 in model.lagmul_bevan_hybrid_spatial_linkage_joints
    joint = model.lagmul_bevan_hybrid_spatial_linkage_joints[156]
    assert joint.node1 == 291
    assert joint.node2 == 292
    assert joint.node3 == 293
    assert pytest.approx(joint.stiff) == 1.05e7
    assert joint.skew_id == 14
    assert pytest.approx(joint.tol) == 4.6e-5
    assert pytest.approx(joint.link_len_a) == 86.0
    assert pytest.approx(joint.link_len_b) == 62.0
    assert pytest.approx(joint.twist_angle_alpha) == 96.0
    assert pytest.approx(joint.offset_distance_s) == 34.0


def test_m376_lagmul_bevan_hybrid_spatial_linkage_joint_aliases(tmp_path: Path):
    deck = """# RADIOSS ALIAS DECK
/BEGIN
Bevan Hybrid Spatial Linkage Joint Aliases Test
/BEVAN_HYBRID_SPATIAL_LINKAGE_JOINT/157
391, 392, 393, 1.0e6, 0, 1.0e-6
48.0, 38.0, 48.0, 14.0
/LAGMUL/BEVAN_HYBRID_SPATIAL_LINKAGE/158
391, 392, 393, 1.0e6, 0, 1.0e-6
48.0, 38.0, 48.0, 14.0
/BEVAN_HYBRID_SPATIAL_LINKAGE/159
391, 392, 393, 1.0e6, 0, 1.0e-6
48.0, 38.0, 48.0, 14.0
/BEVAN_HYBRID_SPATIAL_MULTI_LOOP_MECHANISM/160
391, 392, 393, 1.0e6, 0, 1.0e-6
48.0, 38.0, 48.0, 14.0
/BEVAN_HYBRID_SPATIAL_SYMMETRIC_MECHANISM/161
391, 392, 393, 1.0e6, 0, 1.0e-6
48.0, 38.0, 48.0, 14.0
/BEVAN_HYBRID_SPATIAL_6R_MECHANISM/162
391, 392, 393, 1.0e6, 0, 1.0e-6
48.0, 38.0, 48.0, 14.0
/BEVAN_HYBRID_SPATIAL_OVERCONSTRAINED_MECHANISM/163
391, 392, 393, 1.0e6, 0, 1.0e-6
48.0, 38.0, 48.0, 14.0
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    for jid in range(157, 164):
        assert jid in model.lagmul_bevan_hybrid_spatial_linkage_joints


def test_m376_lagmul_bevan_hybrid_spatial_linkage_joint_missing_card(tmp_path: Path):
    deck = """# RADIOSS MISSING CARD DECK
/BEGIN
/LAGMUL/BEVAN_HYBRID_SPATIAL_LINKAGE_JOINT/164
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) > 0


def test_m376_sensor_spring_torsional_pop_rate_fixed(tmp_path: Path):
    c1 = f"{1238:>10d}{1.58e8:>20.4f}{0.0485:>20.4f}"
    deck = f"""# RADIOSS STARTER DECK
/BEGIN
Sensor Spring Torsional Pop Rate Fixed Format Test
2022 0
/SENSOR/SPRING_TORSIONAL_POP_RATE/1
Fixed Spring Torsional Pop Rate Sensor
{c1}
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 1 in model.sensor_spring_torsional_pop_rates
    s1 = model.sensor_spring_torsional_pop_rates[1]
    assert s1.spring_id == 1238
    assert pytest.approx(s1.jtors_pop_max) == 1.58e8
    assert pytest.approx(s1.jtors_snp_max) == 1.58e8
    assert pytest.approx(s1.jtors_crackle_max) == 1.58e8
    assert pytest.approx(s1.t_delay) == 0.0485
    assert len(model.sensors) == 1
    assert model.sensors[0].kind == "SPRING_TORSIONAL_POP_RATE"


def test_m376_sensor_spring_torsional_pop_rate_free(tmp_path: Path):
    deck = """# RADIOSS FREE DECK
/BEGIN
Sensor Spring Torsional Pop Rate Free Format Test
/SENSOR/SPRING_TORSIONAL_POP_RATE/2
Free Spring Torsional Pop Rate Sensor
1239, 2.18e8, 0.0635
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 2 in model.sensor_spring_torsional_pop_rates
    s2 = model.sensor_spring_torsional_pop_rates[2]
    assert s2.spring_id == 1239
    assert pytest.approx(s2.jtors_pop_max) == 2.18e8
    assert pytest.approx(s2.t_delay) == 0.0635


def test_m376_sensor_spring_torsional_pop_rate_aliases(tmp_path: Path):
    deck = """# RADIOSS ALIAS DECK
/BEGIN
Sensor Spring Torsional Pop Rate Aliases Test
/SENSOR/SPRING_TORS_POP_RATE/260
1259, 9.65e7, 0.0118
/SENSOR/SPRING_RATE_POP_TORS/261
1260, 9.85e7, 0.0128
/SENSOR/TORSIONAL_POP_RATE_SPRING/262
1261, 1.005e8, 0.0138
/SENSOR/SPRING_POP_TORS/263
1262, 1.025e8, 0.0148
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    for sid in range(260, 264):
        assert sid in model.sensor_spring_torsional_pop_rates


def test_m376_sensor_spring_torsional_pop_rate_missing_card(tmp_path: Path):
    deck = """# RADIOSS MISSING CARD DECK
/BEGIN
/SENSOR/SPRING_TORSIONAL_POP_RATE/265
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) > 0
