"""Tests for Milestone M388: LadCoupleMatrixMicroCrushingRate Failure Model, EngFlexomagnetophononicplasmonicpolaritonicResonanceEnergy, KonnokSpatialLinkageJoint, and SensorSpringTorsionalDropRate."""

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


def test_m388_fail_lad_couple_matrix_micro_crushing_rate_fixed(tmp_path: Path):
    c1 = f"{325.0:>20.4f}{975.0:>20.4f}{165.0:>20.4f}{4.75:>20.4f}{0.965:>20.4f}"
    c2 = f"{1:>10d}{2:>10d}"
    c3 = f"{2020:>10d}"
    deck = f"""# RADIOSS STARTER DECK
/BEGIN
Ladeveze Coupled Matrix Micro-Crushing Rate Fixed Format Test
2022 0
/FAIL/LAD_COUPLE_MATRIX_MICRO_CRUSHING_RATE/2020
Ladeveze Coupled Matrix Micro-Crushing Failure Model
{c1}
{c2}
{c3}
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 2020 in model.fail_ladcouplematrixmicrocrushingrates
    fcmmcr = model.fail_ladcouplematrixmicrocrushingrates[2020]
    assert pytest.approx(fcmmcr.sigma_cmmcr0) == 325.0
    assert pytest.approx(fcmmcr.sigma_cmmcrc) == 975.0
    assert pytest.approx(fcmmcr.gamma_cmmcr) == 165.0
    assert pytest.approx(fcmmcr.p_cmmcr) == 4.75
    assert pytest.approx(fcmmcr.d_cmmcr_max) == 0.965
    assert fcmmcr.ifail_sh == 1
    assert fcmmcr.ifail_so == 2
    assert fcmmcr.fail_id == 2020
    assert len(model.raw_fails) == 1
    assert model.raw_fails[0][1].type == "LAD_COUPLE_MATRIX_MICRO_CRUSHING_RATE"


def test_m388_fail_lad_couple_matrix_micro_crushing_rate_free(tmp_path: Path):
    deck = """# RADIOSS FREE DECK
/BEGIN
Ladeveze Coupled Matrix Micro-Crushing Rate Free Format Test
/FAIL/LAD_COUPLE_MATRIX_MICRO_CRUSHING_RATE/2021
335.0, 1005.0, 170.0, 4.95, 0.940
1, 1
2021
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 2021 in model.fail_ladcouplematrixmicrocrushingrates
    fcmmcr = model.fail_ladcouplematrixmicrocrushingrates[2021]
    assert pytest.approx(fcmmcr.sigma_cmmcr0) == 335.0
    assert pytest.approx(fcmmcr.sigma_cmmcrc) == 1005.0
    assert pytest.approx(fcmmcr.gamma_cmmcr) == 170.0
    assert pytest.approx(fcmmcr.p_cmmcr) == 4.95
    assert pytest.approx(fcmmcr.d_cmmcr_max) == 0.940
    assert fcmmcr.fail_id == 2021


def test_m388_fail_lad_couple_matrix_micro_crushing_rate_aliases(tmp_path: Path):
    deck = """# RADIOSS ALIAS DECK
/BEGIN
Ladeveze Coupled Matrix Micro-Crushing Rate Aliases Test
/FAIL/LADEVEZE_COUPLED_MATRIX_MICRO_CRUSHING_RATE/2022
315.0, 945.0, 156.0, 4.45, 0.985
1, 1
/FAIL/LAD_CMMCR/2023
315.0, 945.0, 156.0, 4.45, 0.985
1, 1
/FAIL/LAD_CMMCR_MODEL/2024
315.0, 945.0, 156.0, 4.45, 0.985
1, 1
/FAIL/LAD_CMMCR_LAW/2025
315.0, 945.0, 156.0, 4.45, 0.985
1, 1
/FAIL/LADEVEZE_RATE_DEPENDENT_COUPLED_MATRIX_MICRO_CRUSHING/2026
315.0, 945.0, 156.0, 4.45, 0.985
1, 1
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 2022 in model.fail_ladcouplematrixmicrocrushingrates
    assert 2023 in model.fail_ladcouplematrixmicrocrushingrates
    assert 2024 in model.fail_ladcouplematrixmicrocrushingrates
    assert 2025 in model.fail_ladcouplematrixmicrocrushingrates
    assert 2026 in model.fail_ladcouplematrixmicrocrushingrates
    assert len(model.raw_fails) == 5


def test_m388_fail_lad_couple_matrix_micro_crushing_rate_missing_card(tmp_path: Path):
    deck = """# RADIOSS MISSING CARD DECK
/BEGIN
/FAIL/LAD_COUPLE_MATRIX_MICRO_CRUSHING_RATE/2027
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) > 0


def test_m388_eng_flexomagnetophononicplasmonicpolaritonic_resonance_energy_fixed(tmp_path: Path):
    c1 = f"{0.0086:>20.4f}{325:>10d}"
    deck = f"""# RADIOSS STARTER DECK
/BEGIN
Engine Flexomagnetophononicplasmonicpolaritonic Resonance Energy Fixed Format Test
2022 0
/ENG/FLEXOMAGNETOPHONONICPLASMONICPOLARITONIC_RESONANCE_ENERGY/225
Resonance Energy Output Directive
{c1}
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 225 in model.eng_flexomagnetophononicplasmonicpolaritonic_resonance_energies
    eng = model.eng_flexomagnetophononicplasmonicpolaritonic_resonance_energies[225]
    assert pytest.approx(eng.dt_fmpppr) == 0.0086
    assert eng.sens_id == 325


def test_m388_eng_flexomagnetophononicplasmonicpolaritonic_resonance_energy_free(tmp_path: Path):
    deck = """# RADIOSS FREE DECK
/BEGIN
Engine Flexomagnetophononicplasmonicpolaritonic Resonance Energy Free Format Test
/ENG/FLEXOMAGNETOPHONONICPLASMONICPOLARITONIC_RESONANCE_ENERGY/226
0.0096, 326
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 226 in model.eng_flexomagnetophononicplasmonicpolaritonic_resonance_energies
    eng = model.eng_flexomagnetophononicplasmonicpolaritonic_resonance_energies[226]
    assert pytest.approx(eng.dt_fmpppr) == 0.0096
    assert eng.sens_id == 326


def test_m388_eng_flexomagnetophononicplasmonicpolaritonic_resonance_energy_aliases(tmp_path: Path):
    deck = """# RADIOSS ALIAS DECK
/BEGIN
Engine Flexomagnetophononicplasmonicpolaritonic Resonance Energy Aliases Test
/ENG/FLEXOMAGNETO_PHONON_PLASMON_POLARITON_RES_WORK/227
0.0106, 327
/ENG/EFLEXOMAGNETOPHONONICPLASMONICPOLARITONICRESONANCE/228
0.0116, 328
/ENG/FLEXOMAGNETOPHONONICPLASMONICPOLARITONIC_RESONANCE_DISSIPATION/229
0.0126, 329
/ENG/EM_FLEXOMAGNETOPHONONICPLASMONICPOLARITONIC_RESONANCE/230
0.0136, 330
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 227 in model.eng_flexomagnetophononicplasmonicpolaritonic_resonance_energies
    assert 228 in model.eng_flexomagnetophononicplasmonicpolaritonic_resonance_energies
    assert 229 in model.eng_flexomagnetophononicplasmonicpolaritonic_resonance_energies
    assert 230 in model.eng_flexomagnetophononicplasmonicpolaritonic_resonance_energies


def test_m388_eng_flexomagnetophononicplasmonicpolaritonic_resonance_energy_missing_card(tmp_path: Path):
    deck = """# RADIOSS MISSING CARD DECK
/BEGIN
/ENG/FLEXOMAGNETOPHONONICPLASMONICPOLARITONIC_RESONANCE_ENERGY/231
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) > 0


def test_m388_lagmul_konnok_spatial_linkage_joint_fixed(tmp_path: Path):
    c1 = f"{311:>10d}{312:>10d}{313:>10d}{1.66e7:>20.4f}{37:>10d}{7.5e-5:>20.4e}"
    c2 = f"{106.0:>20.4f}{90.0:>20.4f}{115.0:>20.4f}{46.0:>20.4f}"
    deck = f"""# RADIOSS STARTER DECK
/BEGIN
Konnok Spatial Linkage Joint Fixed Format Test
2022 0
/LAGMUL/KONNOK_SPATIAL_LINKAGE_JOINT/275
Konnok Spatial 6R Joint
{c1}
{c2}
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 275 in model.lagmul_konnok_spatial_linkage_joints
    joint = model.lagmul_konnok_spatial_linkage_joints[275]
    assert joint.node1 == 311
    assert joint.node2 == 312
    assert joint.node3 == 313
    assert pytest.approx(joint.stiff) == 1.66e7
    assert joint.skew_id == 37
    assert pytest.approx(joint.tol) == 7.5e-5
    assert pytest.approx(joint.link_len_a) == 106.0
    assert pytest.approx(joint.link_len_b) == 90.0
    assert pytest.approx(joint.twist_angle_alpha) == 115.0
    assert pytest.approx(joint.offset_distance_s) == 46.0
    assert pytest.approx(joint.offset_distance_r) == 46.0
    assert pytest.approx(joint.offset_distance_v) == 46.0


def test_m388_lagmul_konnok_spatial_linkage_joint_free(tmp_path: Path):
    deck = """# RADIOSS FREE DECK
/BEGIN
Konnok Spatial Linkage Joint Free Format Test
/LAGMUL/KONNOK_SPATIAL_LINKAGE_JOINT/276
411, 412, 413, 1.90e7, 38, 8.5e-5
125.0, 96.0, 135.0, 58.0
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 276 in model.lagmul_konnok_spatial_linkage_joints
    joint = model.lagmul_konnok_spatial_linkage_joints[276]
    assert joint.node1 == 411
    assert joint.node2 == 412
    assert joint.node3 == 413
    assert pytest.approx(joint.stiff) == 1.90e7
    assert joint.skew_id == 38
    assert pytest.approx(joint.tol) == 8.5e-5
    assert pytest.approx(joint.link_len_a) == 125.0
    assert pytest.approx(joint.link_len_b) == 96.0
    assert pytest.approx(joint.twist_angle_alpha) == 135.0
    assert pytest.approx(joint.offset_distance_s) == 58.0


def test_m388_lagmul_konnok_spatial_linkage_joint_aliases(tmp_path: Path):
    deck = """# RADIOSS ALIAS DECK
/BEGIN
Konnok Spatial Linkage Joint Aliases Test
/KONNOK_SPATIAL_LINKAGE_JOINT/277
511, 512, 513, 1.0e6, 0, 1.0e-6
74.0, 64.0, 74.0, 28.0
/LAGMUL/KONNOK_SPATIAL_LINKAGE/278
511, 512, 513, 1.0e6, 0, 1.0e-6
74.0, 64.0, 74.0, 28.0
/KONNOK_SPATIAL_LINKAGE/279
511, 512, 513, 1.0e6, 0, 1.0e-6
74.0, 64.0, 74.0, 28.0
/KONNOK_SPATIAL_MULTI_LOOP_MECHANISM/280
511, 512, 513, 1.0e6, 0, 1.0e-6
74.0, 64.0, 74.0, 28.0
/KONNOK_SPATIAL_SYMMETRIC_MECHANISM/281
511, 512, 513, 1.0e6, 0, 1.0e-6
74.0, 64.0, 74.0, 28.0
/KONNOK_SPATIAL_6R_MECHANISM/282
511, 512, 513, 1.0e6, 0, 1.0e-6
74.0, 64.0, 74.0, 28.0
/KONNOK_SPATIAL_OVERCONSTRAINED_MECHANISM/283
511, 512, 513, 1.0e6, 0, 1.0e-6
74.0, 64.0, 74.0, 28.0
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    for jid in range(277, 284):
        assert jid in model.lagmul_konnok_spatial_linkage_joints


def test_m388_lagmul_konnok_spatial_linkage_joint_missing_card(tmp_path: Path):
    deck = """# RADIOSS MISSING CARD DECK
/BEGIN
/LAGMUL/KONNOK_SPATIAL_LINKAGE_JOINT/284
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) > 0


def test_m388_sensor_spring_torsional_drop_rate_fixed(tmp_path: Path):
    c1 = f"{1358:>10d}{2.78e8:>20.4f}{0.0845:>20.4f}"
    deck = f"""# RADIOSS STARTER DECK
/BEGIN
Sensor Spring Torsional Drop Rate Fixed Format Test
2022 0
/SENSOR/SPRING_TORSIONAL_DROP_RATE/1
Fixed Spring Torsional Drop Rate Sensor
{c1}
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 1 in model.sensor_spring_torsional_drop_rates
    s1 = model.sensor_spring_torsional_drop_rates[1]
    assert s1.spring_id == 1358
    assert pytest.approx(s1.jtors_drop_max) == 2.78e8
    assert pytest.approx(s1.jtors_lock_max) == 2.78e8
    assert pytest.approx(s1.jtors_pop_max) == 2.78e8
    assert pytest.approx(s1.jtors_snp_max) == 2.78e8
    assert pytest.approx(s1.jtors_crackle_max) == 2.78e8
    assert pytest.approx(s1.t_delay) == 0.0845
    assert len(model.sensors) == 1
    assert model.sensors[0].kind == "SPRING_TORSIONAL_DROP_RATE"


def test_m388_sensor_spring_torsional_drop_rate_free(tmp_path: Path):
    deck = """# RADIOSS FREE DECK
/BEGIN
Sensor Spring Torsional Drop Rate Free Format Test
/SENSOR/SPRING_TORSIONAL_DROP_RATE/2
Free Spring Torsional Drop Rate Sensor
1359, 3.38e8, 0.0995
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 2 in model.sensor_spring_torsional_drop_rates
    s2 = model.sensor_spring_torsional_drop_rates[2]
    assert s2.spring_id == 1359
    assert pytest.approx(s2.jtors_drop_max) == 3.38e8
    assert pytest.approx(s2.t_delay) == 0.0995


def test_m388_sensor_spring_torsional_drop_rate_aliases(tmp_path: Path):
    deck = """# RADIOSS ALIAS DECK
/BEGIN
Sensor Spring Torsional Drop Rate Aliases Test
/SENSOR/SPRING_TORS_DROP_RATE/380
1379, 1.185e8, 0.0185
/SENSOR/SPRING_RATE_DROP_TORS/381
1380, 1.205e8, 0.0195
/SENSOR/TORSIONAL_DROP_RATE_SPRING/382
1381, 1.225e8, 0.0205
/SENSOR/SPRING_DROP_TORS/383
1382, 1.245e8, 0.0215
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    for sid in range(380, 384):
        assert sid in model.sensor_spring_torsional_drop_rates


def test_m388_sensor_spring_torsional_drop_rate_missing_card(tmp_path: Path):
    deck = """# RADIOSS MISSING CARD DECK
/BEGIN
/SENSOR/SPRING_TORSIONAL_DROP_RATE/385
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) > 0
