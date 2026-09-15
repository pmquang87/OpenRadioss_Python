"""Tests for Milestone M373: LadCoupleInterlaminarShearDelaminationRate Failure Model, EngFlexomagnetoexcitonicpolaritonicResonanceEnergy, MaverickHybridSpatialLinkageJoint, and SensorSpringNormalPopRate."""

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


def test_m373_fail_lad_couple_interlaminar_shear_delamination_rate_fixed(tmp_path: Path):
    c1 = f"{165.0:>20.4f}{495.0:>20.4f}{76.0:>20.4f}{3.05:>20.4f}{0.960:>20.4f}"
    c2 = f"{1:>10d}{2:>10d}"
    c3 = f"{1870:>10d}"
    deck = f"""# RADIOSS STARTER DECK
/BEGIN
Ladeveze Coupled Interlaminar Shear Delamination Rate Fixed Format Test
2022 0
/FAIL/LAD_COUPLE_INTERLAMINAR_SHEAR_DELAMINATION_RATE/1870
Ladeveze Coupled Interlaminar Shear Delamination Failure Model
{c1}
{c2}
{c3}
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 1870 in model.fail_ladcoupleinterlaminarsheardelaminationrates
    fcisdr = model.fail_ladcoupleinterlaminarsheardelaminationrates[1870]
    assert pytest.approx(fcisdr.sigma_cisdr0) == 165.0
    assert pytest.approx(fcisdr.sigma_cisdrc) == 495.0
    assert pytest.approx(fcisdr.gamma_cisdr) == 76.0
    assert pytest.approx(fcisdr.p_cisdr) == 3.05
    assert pytest.approx(fcisdr.d_cisdr_max) == 0.960
    assert fcisdr.ifail_sh == 1
    assert fcisdr.ifail_so == 2
    assert fcisdr.fail_id == 1870
    assert len(model.raw_fails) == 1
    assert model.raw_fails[0][1].type == "LAD_COUPLE_INTERLAMINAR_SHEAR_DELAMINATION_RATE"


def test_m373_fail_lad_couple_interlaminar_shear_delamination_rate_free(tmp_path: Path):
    deck = """# RADIOSS FREE DECK
/BEGIN
Ladeveze Coupled Interlaminar Shear Delamination Rate Free Format Test
/FAIL/LAD_COUPLE_INTERLAMINAR_SHEAR_DELAMINATION_RATE/1871
175.0, 525.0, 82.0, 3.25, 0.938
1, 1
1871
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 1871 in model.fail_ladcoupleinterlaminarsheardelaminationrates
    fcisdr = model.fail_ladcoupleinterlaminarsheardelaminationrates[1871]
    assert pytest.approx(fcisdr.sigma_cisdr0) == 175.0
    assert pytest.approx(fcisdr.sigma_cisdrc) == 525.0
    assert pytest.approx(fcisdr.gamma_cisdr) == 82.0
    assert pytest.approx(fcisdr.p_cisdr) == 3.25
    assert pytest.approx(fcisdr.d_cisdr_max) == 0.938
    assert fcisdr.fail_id == 1871


def test_m373_fail_lad_couple_interlaminar_shear_delamination_rate_aliases(tmp_path: Path):
    deck = """# RADIOSS ALIAS DECK
/BEGIN
Ladeveze Coupled Interlaminar Shear Delamination Rate Aliases Test
/FAIL/LADEVEZE_COUPLED_INTERLAMINAR_SHEAR_DELAMINATION_RATE/1872
155.0, 465.0, 70.0, 2.75, 0.980
1, 1
/FAIL/LAD_CISDR/1873
155.0, 465.0, 70.0, 2.75, 0.980
1, 1
/FAIL/LAD_CISDR_MODEL/1874
155.0, 465.0, 70.0, 2.75, 0.980
1, 1
/FAIL/LAD_CISDR_LAW/1875
155.0, 465.0, 70.0, 2.75, 0.980
1, 1
/FAIL/LADEVEZE_RATE_DEPENDENT_COUPLED_INTERLAMINAR_SHEAR_DELAMINATION/1876
155.0, 465.0, 70.0, 2.75, 0.980
1, 1
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 1872 in model.fail_ladcoupleinterlaminarsheardelaminationrates
    assert 1873 in model.fail_ladcoupleinterlaminarsheardelaminationrates
    assert 1874 in model.fail_ladcoupleinterlaminarsheardelaminationrates
    assert 1875 in model.fail_ladcoupleinterlaminarsheardelaminationrates
    assert 1876 in model.fail_ladcoupleinterlaminarsheardelaminationrates
    assert len(model.raw_fails) == 5


def test_m373_fail_lad_couple_interlaminar_shear_delamination_rate_missing_card(tmp_path: Path):
    deck = """# RADIOSS MISSING CARD DECK
/BEGIN
/FAIL/LAD_COUPLE_INTERLAMINAR_SHEAR_DELAMINATION_RATE/1877
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) > 0


def test_m373_eng_flexomagnetoexcitonicpolaritonic_resonance_energy_fixed(tmp_path: Path):
    c1 = f"{0.0042:>20.4f}{175:>10d}"
    deck = f"""# RADIOSS STARTER DECK
/BEGIN
Engine Flexomagnetoexcitonicpolaritonic Resonance Energy Fixed Format Test
2022 0
/ENG/FLEXOMAGNETOEXCITONICPOLARITONIC_RESONANCE_ENERGY/75
Resonance Energy Output Directive
{c1}
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 75 in model.eng_flexomagnetoexcitonicpolaritonic_resonance_energies
    eng = model.eng_flexomagnetoexcitonicpolaritonic_resonance_energies[75]
    assert pytest.approx(eng.dt_fmepor) == 0.0042
    assert eng.sens_id == 175


def test_m373_eng_flexomagnetoexcitonicpolaritonic_resonance_energy_free(tmp_path: Path):
    deck = """# RADIOSS FREE DECK
/BEGIN
Engine Flexomagnetoexcitonicpolaritonic Resonance Energy Free Format Test
/ENG/FLEXOMAGNETOEXCITONICPOLARITONIC_RESONANCE_ENERGY/76
0.0052, 176
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 76 in model.eng_flexomagnetoexcitonicpolaritonic_resonance_energies
    eng = model.eng_flexomagnetoexcitonicpolaritonic_resonance_energies[76]
    assert pytest.approx(eng.dt_fmepor) == 0.0052
    assert eng.sens_id == 176


def test_m373_eng_flexomagnetoexcitonicpolaritonic_resonance_energy_aliases(tmp_path: Path):
    deck = """# RADIOSS ALIAS DECK
/BEGIN
Engine Flexomagnetoexcitonicpolaritonic Resonance Energy Aliases Test
/ENG/FLEXOMAGNETO_EXCITON_POLARITON_RES_WORK/77
0.0062, 177
/ENG/EFLEXOMAGNETOEXCITONICPOLARITONICRESONANCE/78
0.0072, 178
/ENG/FLEXOMAGNETOEXCITONICPOLARITONIC_RESONANCE_DISSIPATION/79
0.0082, 179
/ENG/EM_FLEXOMAGNETOEXCITONICPOLARITONIC_RESONANCE/80
0.0092, 180
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 77 in model.eng_flexomagnetoexcitonicpolaritonic_resonance_energies
    assert 78 in model.eng_flexomagnetoexcitonicpolaritonic_resonance_energies
    assert 79 in model.eng_flexomagnetoexcitonicpolaritonic_resonance_energies
    assert 80 in model.eng_flexomagnetoexcitonicpolaritonic_resonance_energies


def test_m373_eng_flexomagnetoexcitonicpolaritonic_resonance_energy_missing_card(tmp_path: Path):
    deck = """# RADIOSS MISSING CARD DECK
/BEGIN
/ENG/FLEXOMAGNETOEXCITONICPOLARITONIC_RESONANCE_ENERGY/81
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) > 0


def test_m373_lagmul_maverick_hybrid_spatial_linkage_joint_fixed(tmp_path: Path):
    c1 = f"{161:>10d}{162:>10d}{163:>10d}{7.5e6:>20.4f}{9:>10d}{3.0e-5:>20.4e}"
    c2 = f"{68.0:>20.4f}{50.0:>20.4f}{66.0:>20.4f}{21.0:>20.4f}"
    deck = f"""# RADIOSS STARTER DECK
/BEGIN
Maverick Hybrid Spatial Linkage Joint Fixed Format Test
2022 0
/LAGMUL/MAVERICK_HYBRID_SPATIAL_LINKAGE_JOINT/125
Maverick Hybrid Spatial 6R Joint
{c1}
{c2}
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 125 in model.lagmul_maverick_hybrid_spatial_linkage_joints
    joint = model.lagmul_maverick_hybrid_spatial_linkage_joints[125]
    assert joint.node1 == 161
    assert joint.node2 == 162
    assert joint.node3 == 163
    assert pytest.approx(joint.stiff) == 7.5e6
    assert joint.skew_id == 9
    assert pytest.approx(joint.tol) == 3.0e-5
    assert pytest.approx(joint.link_len_a) == 68.0
    assert pytest.approx(joint.link_len_b) == 50.0
    assert pytest.approx(joint.twist_angle_alpha) == 66.0
    assert pytest.approx(joint.offset_distance_s) == 21.0


def test_m373_lagmul_maverick_hybrid_spatial_linkage_joint_free(tmp_path: Path):
    deck = """# RADIOSS FREE DECK
/BEGIN
Maverick Hybrid Spatial Linkage Joint Free Format Test
/LAGMUL/MAVERICK_HYBRID_SPATIAL_LINKAGE_JOINT/126
261, 262, 263, 9.0e6, 11, 4.0e-5
76.0, 56.0, 86.0, 28.0
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 126 in model.lagmul_maverick_hybrid_spatial_linkage_joints
    joint = model.lagmul_maverick_hybrid_spatial_linkage_joints[126]
    assert joint.node1 == 261
    assert joint.node2 == 262
    assert joint.node3 == 263
    assert pytest.approx(joint.stiff) == 9.0e6
    assert joint.skew_id == 11
    assert pytest.approx(joint.tol) == 4.0e-5
    assert pytest.approx(joint.link_len_a) == 76.0
    assert pytest.approx(joint.link_len_b) == 56.0
    assert pytest.approx(joint.twist_angle_alpha) == 86.0
    assert pytest.approx(joint.offset_distance_s) == 28.0


def test_m373_lagmul_maverick_hybrid_spatial_linkage_joint_aliases(tmp_path: Path):
    deck = """# RADIOSS ALIAS DECK
/BEGIN
Maverick Hybrid Spatial Linkage Joint Aliases Test
/MAVERICK_HYBRID_SPATIAL_LINKAGE_JOINT/127
361, 362, 363, 1.0e6, 0, 1.0e-6
42.0, 32.0, 42.0, 11.0
/LAGMUL/MAVERICK_HYBRID_SPATIAL_LINKAGE/128
361, 362, 363, 1.0e6, 0, 1.0e-6
42.0, 32.0, 42.0, 11.0
/MAVERICK_HYBRID_SPATIAL_LINKAGE/129
361, 362, 363, 1.0e6, 0, 1.0e-6
42.0, 32.0, 42.0, 11.0
/MAVERICK_HYBRID_SPATIAL_MULTI_LOOP_MECHANISM/130
361, 362, 363, 1.0e6, 0, 1.0e-6
42.0, 32.0, 42.0, 11.0
/MAVERICK_HYBRID_SPATIAL_SYMMETRIC_MECHANISM/131
361, 362, 363, 1.0e6, 0, 1.0e-6
42.0, 32.0, 42.0, 11.0
/MAVERICK_HYBRID_SPATIAL_6R_MECHANISM/132
361, 362, 363, 1.0e6, 0, 1.0e-6
42.0, 32.0, 42.0, 11.0
/MAVERICK_HYBRID_SPATIAL_OVERCONSTRAINED_MECHANISM/133
361, 362, 363, 1.0e6, 0, 1.0e-6
42.0, 32.0, 42.0, 11.0
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    for jid in range(127, 134):
        assert jid in model.lagmul_maverick_hybrid_spatial_linkage_joints


def test_m373_lagmul_maverick_hybrid_spatial_linkage_joint_missing_card(tmp_path: Path):
    deck = """# RADIOSS MISSING CARD DECK
/BEGIN
/LAGMUL/MAVERICK_HYBRID_SPATIAL_LINKAGE_JOINT/134
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) > 0


def test_m373_sensor_spring_normal_pop_rate_fixed(tmp_path: Path):
    c1 = f"{1208:>10d}{1.28e8:>20.4f}{0.0395:>20.4f}"
    deck = f"""# RADIOSS STARTER DECK
/BEGIN
Sensor Spring Normal Pop Rate Fixed Format Test
2022 0
/SENSOR/SPRING_NORMAL_POP_RATE/1
Fixed Spring Normal Pop Rate Sensor
{c1}
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 1 in model.sensor_spring_normal_pop_rates
    s1 = model.sensor_spring_normal_pop_rates[1]
    assert s1.spring_id == 1208
    assert pytest.approx(s1.jnorm_pop_max) == 1.28e8
    assert pytest.approx(s1.jnorm_snp_max) == 1.28e8
    assert pytest.approx(s1.jnorm_crackle_max) == 1.28e8
    assert pytest.approx(s1.t_delay) == 0.0395
    assert len(model.sensors) == 1
    assert model.sensors[0].kind == "SPRING_NORMAL_POP_RATE"


def test_m373_sensor_spring_normal_pop_rate_free(tmp_path: Path):
    deck = """# RADIOSS FREE DECK
/BEGIN
Sensor Spring Normal Pop Rate Free Format Test
/SENSOR/SPRING_NORMAL_POP_RATE/2
Free Spring Normal Pop Rate Sensor
1209, 1.88e8, 0.0545
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 2 in model.sensor_spring_normal_pop_rates
    s2 = model.sensor_spring_normal_pop_rates[2]
    assert s2.spring_id == 1209
    assert pytest.approx(s2.jnorm_pop_max) == 1.88e8
    assert pytest.approx(s2.t_delay) == 0.0545


def test_m373_sensor_spring_normal_pop_rate_aliases(tmp_path: Path):
    deck = """# RADIOSS ALIAS DECK
/BEGIN
Sensor Spring Normal Pop Rate Aliases Test
/SENSOR/SPRING_NORM_POP_RATE/230
1229, 8.35e7, 0.0108
/SENSOR/SPRING_RATE_POP_NORM/231
1230, 8.55e7, 0.0118
/SENSOR/NORMAL_POP_RATE_SPRING/232
1231, 8.75e7, 0.0128
/SENSOR/SPRING_POP_NORM/233
1232, 8.95e7, 0.0138
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    for sid in range(230, 234):
        assert sid in model.sensor_spring_normal_pop_rates


def test_m373_sensor_spring_normal_pop_rate_missing_card(tmp_path: Path):
    deck = """# RADIOSS MISSING CARD DECK
/BEGIN
/SENSOR/SPRING_NORMAL_POP_RATE/235
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) > 0
