"""Tests for Milestone M380: LadDynamicPlyMicroCrackingRate Failure Model, EngFlexomagnetoplasmonicexcitonicmagnonicpolaritonicResonanceEnergy, AlexanderHybridSpatialLinkageJoint, and SensorSpringTransverseLockRate."""

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


def test_m380_fail_lad_dynamic_ply_micro_cracking_rate_fixed(tmp_path: Path):
    c1 = f"{230.0:>20.4f}{690.0:>20.4f}{105.0:>20.4f}{3.70:>20.4f}{0.930:>20.4f}"
    c2 = f"{1:>10d}{2:>10d}"
    c3 = f"{1940:>10d}"
    deck = f"""# RADIOSS STARTER DECK
/BEGIN
Ladeveze Dynamic Ply Micro-Cracking Rate Fixed Format Test
2022 0
/FAIL/LAD_DYNAMIC_PLY_MICRO_CRACKING_RATE/1940
Ladeveze Dynamic Ply Micro-Cracking Failure Model
{c1}
{c2}
{c3}
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 1940 in model.fail_laddynamicplymicrocrackingrates
    fdpmcr = model.fail_laddynamicplymicrocrackingrates[1940]
    assert pytest.approx(fdpmcr.sigma_dpmcr0) == 230.0
    assert pytest.approx(fdpmcr.sigma_dpmcrc) == 690.0
    assert pytest.approx(fdpmcr.gamma_dpmcr) == 105.0
    assert pytest.approx(fdpmcr.p_dpmcr) == 3.70
    assert pytest.approx(fdpmcr.d_dpmcr_max) == 0.930
    assert fdpmcr.ifail_sh == 1
    assert fdpmcr.ifail_so == 2
    assert fdpmcr.fail_id == 1940
    assert len(model.raw_fails) == 1
    assert model.raw_fails[0][1].type == "LAD_DYNAMIC_PLY_MICRO_CRACKING_RATE"


def test_m380_fail_lad_dynamic_ply_micro_cracking_rate_free(tmp_path: Path):
    deck = """# RADIOSS FREE DECK
/BEGIN
Ladeveze Dynamic Ply Micro-Cracking Rate Free Format Test
/FAIL/LAD_DYNAMIC_PLY_MICRO_CRACKING_RATE/1941
240.0, 720.0, 110.0, 3.90, 0.905
1, 1
1941
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 1941 in model.fail_laddynamicplymicrocrackingrates
    fdpmcr = model.fail_laddynamicplymicrocrackingrates[1941]
    assert pytest.approx(fdpmcr.sigma_dpmcr0) == 240.0
    assert pytest.approx(fdpmcr.sigma_dpmcrc) == 720.0
    assert pytest.approx(fdpmcr.gamma_dpmcr) == 110.0
    assert pytest.approx(fdpmcr.p_dpmcr) == 3.90
    assert pytest.approx(fdpmcr.d_dpmcr_max) == 0.905
    assert fdpmcr.fail_id == 1941


def test_m380_fail_lad_dynamic_ply_micro_cracking_rate_aliases(tmp_path: Path):
    deck = """# RADIOSS ALIAS DECK
/BEGIN
Ladeveze Dynamic Ply Micro-Cracking Rate Aliases Test
/FAIL/LADEVEZE_DYNAMIC_PLY_MICRO_CRACKING_RATE/1942
220.0, 660.0, 98.0, 3.40, 0.950
1, 1
/FAIL/LAD_DPMCR/1943
220.0, 660.0, 98.0, 3.40, 0.950
1, 1
/FAIL/LAD_DPMCR_MODEL/1944
220.0, 660.0, 98.0, 3.40, 0.950
1, 1
/FAIL/LAD_DPMCR_LAW/1945
220.0, 660.0, 98.0, 3.40, 0.950
1, 1
/FAIL/LADEVEZE_RATE_DEPENDENT_DYNAMIC_PLY_MICRO_CRACKING/1946
220.0, 660.0, 98.0, 3.40, 0.950
1, 1
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 1942 in model.fail_laddynamicplymicrocrackingrates
    assert 1943 in model.fail_laddynamicplymicrocrackingrates
    assert 1944 in model.fail_laddynamicplymicrocrackingrates
    assert 1945 in model.fail_laddynamicplymicrocrackingrates
    assert 1946 in model.fail_laddynamicplymicrocrackingrates
    assert len(model.raw_fails) == 5


def test_m380_fail_lad_dynamic_ply_micro_cracking_rate_missing_card(tmp_path: Path):
    deck = """# RADIOSS MISSING CARD DECK
/BEGIN
/FAIL/LAD_DYNAMIC_PLY_MICRO_CRACKING_RATE/1947
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) > 0


def test_m380_eng_flexomagnetoplasmonicexcitonicmagnonicpolaritonic_resonance_energy_fixed(tmp_path: Path):
    c1 = f"{0.0062:>20.4f}{245:>10d}"
    deck = f"""# RADIOSS STARTER DECK
/BEGIN
Engine Flexomagnetoplasmonicexcitonicmagnonicpolaritonic Resonance Energy Fixed Format Test
2022 0
/ENG/FLEXOMAGNETOPLASMONICEXCITONICMAGNONICPOLARITONIC_RESONANCE_ENERGY/145
Resonance Energy Output Directive
{c1}
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 145 in model.eng_flexomagnetoplasmonicexcitonicmagnonicpolaritonic_resonance_energies
    eng = model.eng_flexomagnetoplasmonicexcitonicmagnonicpolaritonic_resonance_energies[145]
    assert pytest.approx(eng.dt_fmpempr) == 0.0062
    assert eng.sens_id == 245


def test_m380_eng_flexomagnetoplasmonicexcitonicmagnonicpolaritonic_resonance_energy_free(tmp_path: Path):
    deck = """# RADIOSS FREE DECK
/BEGIN
Engine Flexomagnetoplasmonicexcitonicmagnonicpolaritonic Resonance Energy Free Format Test
/ENG/FLEXOMAGNETOPLASMONICEXCITONICMAGNONICPOLARITONIC_RESONANCE_ENERGY/146
0.0072, 246
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 146 in model.eng_flexomagnetoplasmonicexcitonicmagnonicpolaritonic_resonance_energies
    eng = model.eng_flexomagnetoplasmonicexcitonicmagnonicpolaritonic_resonance_energies[146]
    assert pytest.approx(eng.dt_fmpempr) == 0.0072
    assert eng.sens_id == 246


def test_m380_eng_flexomagnetoplasmonicexcitonicmagnonicpolaritonic_resonance_energy_aliases(tmp_path: Path):
    deck = """# RADIOSS ALIAS DECK
/BEGIN
Engine Flexomagnetoplasmonicexcitonicmagnonicpolaritonic Resonance Energy Aliases Test
/ENG/FLEXOMAGNETO_PLASMON_EXCITON_MAGNON_POLARITON_RES_WORK/147
0.0082, 247
/ENG/EFLEXOMAGNETOPLASMONICEXCITONICMAGNONICPOLARITONICRESONANCE/148
0.0092, 248
/ENG/FLEXOMAGNETOPLASMONICEXCITONICMAGNONICPOLARITONIC_RESONANCE_DISSIPATION/149
0.0102, 249
/ENG/EM_FLEXOMAGNETOPLASMONICEXCITONICMAGNONICPOLARITONIC_RESONANCE/150
0.0112, 250
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 147 in model.eng_flexomagnetoplasmonicexcitonicmagnonicpolaritonic_resonance_energies
    assert 148 in model.eng_flexomagnetoplasmonicexcitonicmagnonicpolaritonic_resonance_energies
    assert 149 in model.eng_flexomagnetoplasmonicexcitonicmagnonicpolaritonic_resonance_energies
    assert 150 in model.eng_flexomagnetoplasmonicexcitonicmagnonicpolaritonic_resonance_energies


def test_m380_eng_flexomagnetoplasmonicexcitonicmagnonicpolaritonic_resonance_energy_missing_card(tmp_path: Path):
    deck = """# RADIOSS MISSING CARD DECK
/BEGIN
/ENG/FLEXOMAGNETOPLASMONICEXCITONICMAGNONICPOLARITONIC_RESONANCE_ENERGY/151
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) > 0


def test_m380_lagmul_alexander_hybrid_spatial_linkage_joint_fixed(tmp_path: Path):
    c1 = f"{231:>10d}{232:>10d}{233:>10d}{1.05e7:>20.4f}{21:>10d}{4.4e-5:>20.4e}"
    c2 = f"{82.0:>20.4f}{64.0:>20.4f}{84.0:>20.4f}{30.0:>20.4f}"
    deck = f"""# RADIOSS STARTER DECK
/BEGIN
Alexander Hybrid Spatial Linkage Joint Fixed Format Test
2022 0
/LAGMUL/ALEXANDER_HYBRID_SPATIAL_LINKAGE_JOINT/195
Alexander Hybrid Spatial 6R Joint
{c1}
{c2}
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 195 in model.lagmul_alexander_hybrid_spatial_linkage_joints
    joint = model.lagmul_alexander_hybrid_spatial_linkage_joints[195]
    assert joint.node1 == 231
    assert joint.node2 == 232
    assert joint.node3 == 233
    assert pytest.approx(joint.stiff) == 1.05e7
    assert joint.skew_id == 21
    assert pytest.approx(joint.tol) == 4.4e-5
    assert pytest.approx(joint.link_len_a) == 82.0
    assert pytest.approx(joint.link_len_b) == 64.0
    assert pytest.approx(joint.twist_angle_alpha) == 84.0
    assert pytest.approx(joint.offset_distance_s) == 30.0


def test_m380_lagmul_alexander_hybrid_spatial_linkage_joint_free(tmp_path: Path):
    deck = """# RADIOSS FREE DECK
/BEGIN
Alexander Hybrid Spatial Linkage Joint Free Format Test
/LAGMUL/ALEXANDER_HYBRID_SPATIAL_LINKAGE_JOINT/196
331, 332, 333, 1.25e7, 22, 5.4e-5
94.0, 70.0, 104.0, 42.0
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 196 in model.lagmul_alexander_hybrid_spatial_linkage_joints
    joint = model.lagmul_alexander_hybrid_spatial_linkage_joints[196]
    assert joint.node1 == 331
    assert joint.node2 == 332
    assert joint.node3 == 333
    assert pytest.approx(joint.stiff) == 1.25e7
    assert joint.skew_id == 22
    assert pytest.approx(joint.tol) == 5.4e-5
    assert pytest.approx(joint.link_len_a) == 94.0
    assert pytest.approx(joint.link_len_b) == 70.0
    assert pytest.approx(joint.twist_angle_alpha) == 104.0
    assert pytest.approx(joint.offset_distance_s) == 42.0


def test_m380_lagmul_alexander_hybrid_spatial_linkage_joint_aliases(tmp_path: Path):
    deck = """# RADIOSS ALIAS DECK
/BEGIN
Alexander Hybrid Spatial Linkage Joint Aliases Test
/ALEXANDER_HYBRID_SPATIAL_LINKAGE_JOINT/197
431, 432, 433, 1.0e6, 0, 1.0e-6
56.0, 46.0, 56.0, 18.0
/LAGMUL/ALEXANDER_HYBRID_SPATIAL_LINKAGE/198
431, 432, 433, 1.0e6, 0, 1.0e-6
56.0, 46.0, 56.0, 18.0
/ALEXANDER_HYBRID_SPATIAL_LINKAGE/199
431, 432, 433, 1.0e6, 0, 1.0e-6
56.0, 46.0, 56.0, 18.0
/ALEXANDER_HYBRID_SPATIAL_MULTI_LOOP_MECHANISM/200
431, 432, 433, 1.0e6, 0, 1.0e-6
56.0, 46.0, 56.0, 18.0
/ALEXANDER_HYBRID_SPATIAL_SYMMETRIC_MECHANISM/201
431, 432, 433, 1.0e6, 0, 1.0e-6
56.0, 46.0, 56.0, 18.0
/ALEXANDER_HYBRID_SPATIAL_6R_MECHANISM/202
431, 432, 433, 1.0e6, 0, 1.0e-6
56.0, 46.0, 56.0, 18.0
/ALEXANDER_HYBRID_SPATIAL_OVERCONSTRAINED_MECHANISM/203
431, 432, 433, 1.0e6, 0, 1.0e-6
56.0, 46.0, 56.0, 18.0
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    for jid in range(197, 204):
        assert jid in model.lagmul_alexander_hybrid_spatial_linkage_joints


def test_m380_lagmul_alexander_hybrid_spatial_linkage_joint_missing_card(tmp_path: Path):
    deck = """# RADIOSS MISSING CARD DECK
/BEGIN
/LAGMUL/ALEXANDER_HYBRID_SPATIAL_LINKAGE_JOINT/204
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) > 0


def test_m380_sensor_spring_transverse_lock_rate_fixed(tmp_path: Path):
    c1 = f"{1278:>10d}{1.98e8:>20.4f}{0.0605:>20.4f}"
    deck = f"""# RADIOSS STARTER DECK
/BEGIN
Sensor Spring Transverse Lock Rate Fixed Format Test
2022 0
/SENSOR/SPRING_TRANSVERSE_LOCK_RATE/1
Fixed Spring Transverse Lock Rate Sensor
{c1}
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 1 in model.sensor_spring_transverse_lock_rates
    s1 = model.sensor_spring_transverse_lock_rates[1]
    assert s1.spring_id == 1278
    assert pytest.approx(s1.jtrans_lock_max) == 1.98e8
    assert pytest.approx(s1.jtrans_pop_max) == 1.98e8
    assert pytest.approx(s1.jtrans_snp_max) == 1.98e8
    assert pytest.approx(s1.jtrans_crackle_max) == 1.98e8
    assert pytest.approx(s1.t_delay) == 0.0605
    assert len(model.sensors) == 1
    assert model.sensors[0].kind == "SPRING_TRANSVERSE_LOCK_RATE"


def test_m380_sensor_spring_transverse_lock_rate_free(tmp_path: Path):
    deck = """# RADIOSS FREE DECK
/BEGIN
Sensor Spring Transverse Lock Rate Free Format Test
/SENSOR/SPRING_TRANSVERSE_LOCK_RATE/2
Free Spring Transverse Lock Rate Sensor
1279, 2.58e8, 0.0755
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 2 in model.sensor_spring_transverse_lock_rates
    s2 = model.sensor_spring_transverse_lock_rates[2]
    assert s2.spring_id == 1279
    assert pytest.approx(s2.jtrans_lock_max) == 2.58e8
    assert pytest.approx(s2.t_delay) == 0.0755


def test_m380_sensor_spring_transverse_lock_rate_aliases(tmp_path: Path):
    deck = """# RADIOSS ALIAS DECK
/BEGIN
Sensor Spring Transverse Lock Rate Aliases Test
/SENSOR/SPRING_TRANS_LOCK_RATE/300
1299, 1.085e8, 0.0135
/SENSOR/SPRING_RATE_LOCK_TRANS/301
1300, 1.105e8, 0.0145
/SENSOR/TRANSVERSE_LOCK_RATE_SPRING/302
1301, 1.125e8, 0.0155
/SENSOR/SPRING_LOCK_TRANS/303
1302, 1.145e8, 0.0165
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    for sid in range(300, 304):
        assert sid in model.sensor_spring_transverse_lock_rates


def test_m380_sensor_spring_transverse_lock_rate_missing_card(tmp_path: Path):
    deck = """# RADIOSS MISSING CARD DECK
/BEGIN
/SENSOR/SPRING_TRANSVERSE_LOCK_RATE/305
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) > 0
