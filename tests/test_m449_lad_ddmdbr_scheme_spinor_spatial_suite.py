"""Tests for Milestone M449: LadDynamicDelaminationMicrodebondingRate Failure Model, EngElectrothermoflexomagnetochiralexcitonicplasmonicpolaritonicResonanceEnergy, LagmulSchemeSpinorSpatialLinkageJoint, and SensorSpringNormalCrackleRate."""

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


def test_m449_fail_lad_dynamic_delamination_microdebonding_rate_fixed(tmp_path: Path):
    c1 = f"{1080.0:>20.4f}{3240.0:>20.4f}{880.0:>20.4f}{17.30:>20.4f}{0.390:>20.4f}"
    c2 = f"{1:>10d}{2:>10d}"
    c3 = f"{2730:>10d}"
    deck = f"""# RADIOSS STARTER DECK
/BEGIN
Ladeveze Dynamic Delamination Microdebonding Rate Fixed Format Test
2022 0
/FAIL/LAD_DYNAMIC_DELAMINATION_MICRODEBONDING_RATE/2730
Ladeveze Dynamic Delamination Microdebonding Failure Model
{c1}
{c2}
{c3}
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 2730 in model.fail_laddynamicdelaminationmicrodebondingrates
    fddmdbr = model.fail_laddynamicdelaminationmicrodebondingrates[2730]
    assert pytest.approx(fddmdbr.sigma_ddmdbr0) == 1080.0
    assert pytest.approx(fddmdbr.sigma_ddmdbrc) == 3240.0
    assert pytest.approx(fddmdbr.gamma_ddmdbr) == 880.0
    assert pytest.approx(fddmdbr.p_ddmdbr) == 17.30
    assert pytest.approx(fddmdbr.d_ddmdbr_max) == 0.390
    assert fddmdbr.ifail_sh == 1
    assert fddmdbr.ifail_so == 2
    assert fddmdbr.fail_id == 2730
    assert pytest.approx(fddmdbr.sigma_ddmdb0) == 1080.0
    assert pytest.approx(fddmdbr.sigma_ddmdbc) == 3240.0
    assert pytest.approx(fddmdbr.gamma_ddmdb) == 880.0
    assert pytest.approx(fddmdbr.p_ddmdb) == 17.30
    assert pytest.approx(fddmdbr.d_ddmdb_max) == 0.390
    assert pytest.approx(fddmdbr.sigma_ddm0) == 1080.0
    assert pytest.approx(fddmdbr.sigma_ddmc) == 3240.0
    assert pytest.approx(fddmdbr.gamma_ddm) == 880.0
    assert pytest.approx(fddmdbr.p_ddm) == 17.30
    assert pytest.approx(fddmdbr.d_ddm_max) == 0.390
    assert pytest.approx(fddmdbr.sigma_dimdbr0) == 1080.0
    assert pytest.approx(fddmdbr.sigma_dimdbrc) == 3240.0
    assert pytest.approx(fddmdbr.gamma_dimdbr) == 880.0
    assert pytest.approx(fddmdbr.p_dimdbr) == 17.30
    assert pytest.approx(fddmdbr.d_dimdbr_max) == 0.390
    assert len(model.raw_fails) == 1
    assert model.raw_fails[0][1].type == "LAD_DYNAMIC_DELAMINATION_MICRODEBONDING_RATE"


def test_m449_fail_lad_dynamic_delamination_microdebonding_rate_free(tmp_path: Path):
    deck = """# RADIOSS FREE DECK
/BEGIN
Ladeveze Dynamic Delamination Microdebonding Rate Free Format Test
/FAIL/LAD_DYNAMIC_DELAMINATION_MICRODEBONDING_RATE/2731
1100.0, 3300.0, 900.0, 17.60, 0.380
1, 1
2731
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 2731 in model.fail_laddynamicdelaminationmicrodebondingrates
    fddmdbr = model.fail_laddynamicdelaminationmicrodebondingrates[2731]
    assert pytest.approx(fddmdbr.sigma_ddmdbr0) == 1100.0
    assert pytest.approx(fddmdbr.sigma_ddmdbrc) == 3300.0
    assert pytest.approx(fddmdbr.gamma_ddmdbr) == 900.0
    assert pytest.approx(fddmdbr.p_ddmdbr) == 17.60
    assert pytest.approx(fddmdbr.d_ddmdbr_max) == 0.380
    assert fddmdbr.fail_id == 2731


def test_m449_fail_lad_dynamic_delamination_microdebonding_rate_aliases(tmp_path: Path):
    deck = """# RADIOSS ALIAS DECK
/BEGIN
Ladeveze Dynamic Delamination Microdebonding Rate Aliases Test
/FAIL/LADEVEZE_DYNAMIC_DELAMINATION_MICRODEBONDING_RATE/2732
1030.0, 3090.0, 830.0, 16.10, 0.430
1, 1
/FAIL/LAD_DYNAMIC_DELAMINATION_MICRODEBOND_RATE/2733
1030.0, 3090.0, 830.0, 16.10, 0.430
1, 1
/FAIL/LADEVEZE_DYNAMIC_DELAMINATION_MICRODEBOND_RATE/2734
1030.0, 3090.0, 830.0, 16.10, 0.430
1, 1
/FAIL/LAD_DYNAMIC_DELAM_MICRODEBONDING_RATE/2735
1030.0, 3090.0, 830.0, 16.10, 0.430
1, 1
/FAIL/LADEVEZE_DYNAMIC_DELAM_MICRODEBONDING_RATE/2736
1030.0, 3090.0, 830.0, 16.10, 0.430
1, 1
/FAIL/LAD_DYNAMIC_DELAM_MICRODEBOND_RATE/2737
1030.0, 3090.0, 830.0, 16.10, 0.430
1, 1
/FAIL/LADEVEZE_DYNAMIC_DELAM_MICRODEBOND_RATE/2738
1030.0, 3090.0, 830.0, 16.10, 0.430
1, 1
/FAIL/LAD_DDMDBR/2739
1030.0, 3090.0, 830.0, 16.10, 0.430
1, 1
/FAIL/LAD_DDMDBR_MODEL/2740
1030.0, 3090.0, 830.0, 16.10, 0.430
1, 1
/FAIL/LAD_DDMDBR_LAW/2741
1030.0, 3090.0, 830.0, 16.10, 0.430
1, 1
/FAIL/LADEVEZE_RATE_DEPENDENT_DYNAMIC_DELAMINATION_MICRODEBONDING/2742
1030.0, 3090.0, 830.0, 16.10, 0.430
1, 1
/FAIL/LAD_DYNAMIC_INTERLAMINAR_MICRODEBONDING_RATE/2743
1030.0, 3090.0, 830.0, 16.10, 0.430
1, 1
/FAIL/LADEVEZE_DYNAMIC_INTERLAMINAR_MICRODEBONDING_RATE/2744
1030.0, 3090.0, 830.0, 16.10, 0.430
1, 1
/FAIL/LAD_DYNAMIC_INTERLAMINAR_MICRODEBOND_RATE/2745
1030.0, 3090.0, 830.0, 16.10, 0.430
1, 1
/FAIL/LADEVEZE_DYNAMIC_INTERLAMINAR_MICRODEBOND_RATE/2746
1030.0, 3090.0, 830.0, 16.10, 0.430
1, 1
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    for mid in range(2732, 2747):
        assert mid in model.fail_laddynamicdelaminationmicrodebondingrates
        f = model.fail_laddynamicdelaminationmicrodebondingrates[mid]
        assert pytest.approx(f.sigma_ddmdbr0) == 1030.0
        assert pytest.approx(f.sigma_ddmdbrc) == 3090.0


def test_m449_fail_lad_dynamic_delamination_microdebonding_rate_properties():
    from pyradioss.model.entities import FailLadDynamicDelaminationMicrodebondingRate
    f = FailLadDynamicDelaminationMicrodebondingRate(mat_id=1)
    f.sigma_ddmdb0 = 1140.0
    assert pytest.approx(f.sigma_ddmdbr0) == 1140.0
    assert pytest.approx(f.sigma_ddmdb0) == 1140.0

    f.sigma_ddmdbc = 3420.0
    assert pytest.approx(f.sigma_ddmdbrc) == 3420.0
    assert pytest.approx(f.sigma_ddmdbc) == 3420.0

    f.gamma_ddmdb = 920.0
    assert pytest.approx(f.gamma_ddmdbr) == 920.0
    assert pytest.approx(f.gamma_ddmdb) == 920.0

    f.p_ddmdb = 17.40
    assert pytest.approx(f.p_ddmdbr) == 17.40
    assert pytest.approx(f.p_ddmdb) == 17.40

    f.d_ddmdb_max = 0.370
    assert pytest.approx(f.d_ddmdbr_max) == 0.370
    assert pytest.approx(f.d_ddmdb_max) == 0.370

    f.sigma_dimdbr0 = 1200.0
    assert pytest.approx(f.sigma_ddmdbr0) == 1200.0
    assert pytest.approx(f.sigma_dimdbr0) == 1200.0

    f.sigma_dimdbrc = 3600.0
    assert pytest.approx(f.sigma_ddmdbrc) == 3600.0
    assert pytest.approx(f.sigma_dimdbrc) == 3600.0

    f.gamma_dimdbr = 960.0
    assert pytest.approx(f.gamma_ddmdbr) == 960.0
    assert pytest.approx(f.gamma_dimdbr) == 960.0

    f.p_dimdbr = 18.40
    assert pytest.approx(f.p_ddmdbr) == 18.40
    assert pytest.approx(f.p_dimdbr) == 18.40

    f.d_dimdbr_max = 0.340
    assert pytest.approx(f.d_ddmdbr_max) == 0.340
    assert pytest.approx(f.d_dimdbr_max) == 0.340


def test_m449_eng_electrothermoflexomagnetochiralexcitonicplasmonicpolaritonic_resonance_energy_fixed(tmp_path: Path):
    c1 = f"{0.00225:>20.6f}{74:>10d}"
    deck = f"""# RADIOSS STARTER DECK
/BEGIN
Engine Chiral Exciton Plasmon Polaritonic Resonance Energy Fixed Format Test
2022 0
/ENG/ELECTROTHERMOFLEXOMAGNETOCHIRALEXCITONICPLASMONICPOLARITONIC_RESONANCE_ENERGY/1
Chiral Exciton Plasmon Polaritonic Resonance Energy Title
{c1}
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 1 in model.eng_electrothermoflexomagnetochiralexcitonicplasmonicpolaritonic_resonance_energies
    r = model.eng_electrothermoflexomagnetochiralexcitonicplasmonicpolaritonic_resonance_energies[1]
    assert pytest.approx(r.dt_etfcexplp) == 0.00225
    assert pytest.approx(r.dt_etfplp) == 0.00225
    assert r.sens_id == 74


def test_m449_eng_electrothermoflexomagnetochiralexcitonicplasmonicpolaritonic_resonance_energy_free(tmp_path: Path):
    deck = """# RADIOSS FREE DECK
/BEGIN
Engine Chiral Exciton Plasmon Polaritonic Resonance Energy Free Format Test
/ENG/ELECTROTHERMOFLEXOMAGNETOCHIRALEXCITONICPLASMONICPOLARITONIC_RESONANCE_ENERGY/2
0.00325, 75
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 2 in model.eng_electrothermoflexomagnetochiralexcitonicplasmonicpolaritonic_resonance_energies
    r = model.eng_electrothermoflexomagnetochiralexcitonicplasmonicpolaritonic_resonance_energies[2]
    assert pytest.approx(r.dt_etfcexplp) == 0.00325
    assert r.sens_id == 75


def test_m449_eng_electrothermoflexomagnetochiralexcitonicplasmonicpolaritonic_resonance_energy_aliases(tmp_path: Path):
    deck = """# RADIOSS ALIAS DECK
/BEGIN
Engine Chiral Exciton Plasmon Polaritonic Resonance Energy Aliases Test
/ENG/ELECTRO_THERM_FLEXO_MAG_CHIRAL_EXCITON_PLASMON_POLARITON_RES_WORK/3
0.00425, 76
/ENG/EELECTROTHERMOFLEXOMAGNETOCHIRALEXCITONICPLASMONICPOLARITONICRESONANCE/4
0.00425, 76
/ENG/ELECTROTHERMOFLEXOMAGNETOCHIRALEXCITONICPLASMONICPOLARITONIC_RESONANCE_DISSIPATION/5
0.00425, 76
/ENG/ET_ELECTROTHERMOFLEXOMAGNETOCHIRALEXCITONICPLASMONICPOLARITONIC_RESONANCE/6
0.00425, 76
/ENG/ELECTROTHERMOFLEXOMAGNETOEXCITONICCHIRALPLASMONICPOLARITONIC_RESONANCE_ENERGY/7
0.00425, 76
/ENG/ELECTRO_THERM_FLEXO_MAG_EXCITON_CHIRAL_PLASMON_POLARITON_RES_WORK/8
0.00425, 76
/ENG/EELECTROTHERMOFLEXOMAGNETOEXCITONICCHIRALPLASMONICPOLARITONICRESONANCE/9
0.00425, 76
/ENG/ELECTROTHERMOFLEXOMAGNETOEXCITONICCHIRALPLASMONICPOLARITONIC_RESONANCE_DISSIPATION/10
0.00425, 76
/ENG/ET_ELECTROTHERMOFLEXOMAGNETOEXCITONICCHIRALPLASMONICPOLARITONIC_RESONANCE/11
0.00425, 76
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    for rid in range(3, 12):
        assert rid in model.eng_electrothermoflexomagnetochiralexcitonicplasmonicpolaritonic_resonance_energies
        r = model.eng_electrothermoflexomagnetochiralexcitonicplasmonicpolaritonic_resonance_energies[rid]
        assert pytest.approx(r.dt_etfcexplp) == 0.00425
        assert r.sens_id == 76


def test_m449_lagmul_scheme_spinor_spatial_linkage_joint_fixed(tmp_path: Path):
    c1 = f"{101:>10d}{102:>10d}{103:>10d}{3.7e6:>20.4f}{6:>10d}{2.2e-5:>20.6e}"
    c2 = f"{22.0:>20.4f}{24.0:>20.4f}{72.0:>20.4f}{8.0:>20.4f}"
    deck = f"""# RADIOSS STARTER DECK
/BEGIN
Scheme Spinor Spatial Linkage Joint Fixed Format Test
2022 0
/SCHEME_SPINOR_SPATIAL_LINKAGE_JOINT/1
Scheme Spinor Spatial Mechanism Joint
{c1}
{c2}
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 1 in model.lagmul_scheme_spinor_spatial_linkage_joints
    j = model.lagmul_scheme_spinor_spatial_linkage_joints[1]
    assert j.node1 == 101
    assert j.node2 == 102
    assert j.node3 == 103
    assert pytest.approx(j.stiff) == 3.7e6
    assert j.skew_id == 6
    assert pytest.approx(j.tol) == 2.2e-5
    assert pytest.approx(j.link_len_a) == 22.0
    assert pytest.approx(j.link_len_b) == 24.0
    assert pytest.approx(j.twist_angle_alpha) == 72.0
    assert pytest.approx(j.offset_distance_s) == 8.0
    assert pytest.approx(j.offset_distance_r) == 8.0
    assert pytest.approx(j.offset_distance_v) == 8.0
    assert pytest.approx(j.offset_distance_h) == 8.0
    assert pytest.approx(j.offset_distance_u) == 8.0
    assert pytest.approx(j.offset_distance_f) == 8.0


def test_m449_lagmul_scheme_spinor_spatial_linkage_joint_free(tmp_path: Path):
    deck = """# RADIOSS FREE DECK
/BEGIN
Scheme Spinor Spatial Linkage Joint Free Format Test
/LAGMUL/SCHEME_SPINOR_SPATIAL_LINKAGE_JOINT/2
201, 202, 203, 4.7e6, 7, 3.2e-5
25.0, 27.0, 84.0, 9.0
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 2 in model.lagmul_scheme_spinor_spatial_linkage_joints
    j = model.lagmul_scheme_spinor_spatial_linkage_joints[2]
    assert j.node1 == 201
    assert j.node2 == 202
    assert j.node3 == 203
    assert pytest.approx(j.stiff) == 4.7e6
    assert j.skew_id == 7
    assert pytest.approx(j.tol) == 3.2e-5
    assert pytest.approx(j.link_len_a) == 25.0
    assert pytest.approx(j.link_len_b) == 27.0
    assert pytest.approx(j.twist_angle_alpha) == 84.0
    assert pytest.approx(j.offset_distance_s) == 9.0


def test_m449_lagmul_scheme_spinor_spatial_linkage_joint_aliases(tmp_path: Path):
    deck = """# RADIOSS ALIAS DECK
/BEGIN
Scheme Spinor Spatial Linkage Joint Aliases Test
/LAGMUL/SCHEME_SPINOR_SPATIAL_LINKAGE/3
301, 302, 303, 1.0e6, 0, 1.0e-6
10.0, 10.0, 30.0, 2.0
/SCHEME_SPINOR_SPATIAL_LINKAGE/4
301, 302, 303, 1.0e6, 0, 1.0e-6
10.0, 10.0, 30.0, 2.0
/SCHEME_SPINOR_SPATIAL_MULTI_LOOP_MECHANISM/5
301, 302, 303, 1.0e6, 0, 1.0e-6
10.0, 10.0, 30.0, 2.0
/SCHEME_SPINOR_SPATIAL_SYMMETRIC_MECHANISM/6
301, 302, 303, 1.0e6, 0, 1.0e-6
10.0, 10.0, 30.0, 2.0
/SCHEME_SPINOR_SPATIAL_6R_MECHANISM/7
301, 302, 303, 1.0e6, 0, 1.0e-6
10.0, 10.0, 30.0, 2.0
/SCHEME_SPINOR_SPATIAL_OVERCONSTRAINED_MECHANISM/8
301, 302, 303, 1.0e6, 0, 1.0e-6
10.0, 10.0, 30.0, 2.0
/LAGMUL/SCHEME_TWISTOR_SPATIAL_LINKAGE_JOINT/9
301, 302, 303, 1.0e6, 0, 1.0e-6
10.0, 10.0, 30.0, 2.0
/SCHEME_TWISTOR_SPATIAL_LINKAGE_JOINT/10
301, 302, 303, 1.0e6, 0, 1.0e-6
10.0, 10.0, 30.0, 2.0
/LAGMUL/SCHEME_SPINOR_BUNDLE_SPATIAL_LINKAGE_JOINT/11
301, 302, 303, 1.0e6, 0, 1.0e-6
10.0, 10.0, 30.0, 2.0
/SCHEME_SPINOR_BUNDLE_SPATIAL_LINKAGE_JOINT/12
301, 302, 303, 1.0e6, 0, 1.0e-6
10.0, 10.0, 30.0, 2.0
/LAGMUL/SCHEME_CLIFFORD_SPINOR_SPATIAL_LINKAGE_JOINT/13
301, 302, 303, 1.0e6, 0, 1.0e-6
10.0, 10.0, 30.0, 2.0
/SCHEME_CLIFFORD_SPINOR_SPATIAL_LINKAGE_JOINT/14
301, 302, 303, 1.0e6, 0, 1.0e-6
10.0, 10.0, 30.0, 2.0
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    for jid in range(3, 15):
        assert jid in model.lagmul_scheme_spinor_spatial_linkage_joints
        j = model.lagmul_scheme_spinor_spatial_linkage_joints[jid]
        assert j.node1 == 301
        assert j.node2 == 302
        assert j.node3 == 303


def test_m449_sensor_spring_normal_crackle_rate_fixed(tmp_path: Path):
    c1 = f"{10:>10d}{1.98e5:>20.4f}{0.0028:>20.6f}"
    deck = f"""# RADIOSS STARTER DECK
/BEGIN
Sensor Spring Normal Crackle Rate Fixed Format Test
2022 0
/SENSOR/SPRING_NORMAL_CRACKLE_RATE/1
Normal Crackle Rate Sensor Title
{c1}
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 1 in model.sensor_spring_normal_crackle_rates
    s = model.sensor_spring_normal_crackle_rates[1]
    assert s.spring_id == 10
    assert pytest.approx(s.jnorm_crk_max) == 1.98e5
    assert pytest.approx(s.t_delay) == 0.0028
    assert len(model.sensors) == 1
    assert model.sensors[0].kind == "SPRING_NORMAL_CRACKLE_RATE"


def test_m449_sensor_spring_normal_crackle_rate_free(tmp_path: Path):
    deck = """# RADIOSS FREE DECK
/BEGIN
Sensor Spring Normal Crackle Rate Free Format Test
/SENSOR/SPRING_NORMAL_CRACKLE_RATE/2
20, 2.98e5, 0.0038
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 2 in model.sensor_spring_normal_crackle_rates
    s = model.sensor_spring_normal_crackle_rates[2]
    assert s.spring_id == 20
    assert pytest.approx(s.jnorm_crk_max) == 2.98e5
    assert pytest.approx(s.t_delay) == 0.0038


def test_m449_sensor_spring_normal_crackle_rate_aliases(tmp_path: Path):
    deck = """# RADIOSS ALIAS DECK
/BEGIN
Sensor Spring Normal Crackle Rate Aliases Test
/SENSOR/SPRING_NORM_CRK_RATE/3
30, 3.7e5, 0.001
/SENSOR/SPRING_NORMAL_CRACKLE/4
30, 3.7e5, 0.001
/SENSOR/SPRING_NORM_CRACKLE/5
30, 3.7e5, 0.001
/SENSOR/SPRING_NORMAL_CRK/6
30, 3.7e5, 0.001
/SENSOR/SPRING_NORM_CRK/7
30, 3.7e5, 0.001
/SENSOR/SPRING_POP_RATE_NORMAL_CRACKLE/8
30, 3.7e5, 0.001
/SENSOR/NORMAL_CRACKLE_RATE_SPRING/9
30, 3.7e5, 0.001
/SENSOR/SPRING_CRACKLE_RATE_NORMAL/10
30, 3.7e5, 0.001
/SENSOR/SPRING_CRACKLE_RATE_NORM/11
30, 3.7e5, 0.001
/SENSOR/SPRING_CRK_RATE_NORMAL/12
30, 3.7e5, 0.001
/SENSOR/SPRING_CRK_RATE_NORM/13
30, 3.7e5, 0.001
/SENSOR/SPRING_NORM_CRACKLE_RATE/14
30, 3.7e5, 0.001
/SENSOR/SPRING_RATE_CRACKLE_NORM/15
30, 3.7e5, 0.001
/SENSOR/SPRING_CRACKLE_NORM/16
30, 3.7e5, 0.001
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    for sid in range(3, 17):
        assert sid in model.sensor_spring_normal_crackle_rates
        s = model.sensor_spring_normal_crackle_rates[sid]
        assert s.spring_id == 30
        assert pytest.approx(s.jnorm_crk_max) == 3.7e5


def test_m449_missing_card_errors(tmp_path: Path):
    deck = """# RADIOSS ERROR DECK
/BEGIN
Error Handling Test
/FAIL/LAD_DYNAMIC_DELAMINATION_MICRODEBONDING_RATE/1
/ENG/ELECTROTHERMOFLEXOMAGNETOCHIRALEXCITONICPLASMONICPOLARITONIC_RESONANCE_ENERGY/1
/SCHEME_SPINOR_SPATIAL_LINKAGE_JOINT/1
/SENSOR/SPRING_NORMAL_CRACKLE_RATE/1
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 4
