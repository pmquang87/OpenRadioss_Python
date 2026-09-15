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


def test_m450_fail_lad_transverse_delamination_microdebonding_rate_fixed(tmp_path: Path):
    c1 = f"{1050.0:>20.4f}{3150.0:>20.4f}{850.0:>20.4f}{16.50:>20.4f}{0.450:>20.4f}"
    c2 = f"{2:>10d}{1:>10d}"
    c3 = f"{750:>10d}"
    deck = f"""# RADIOSS STARTER DECK
/BEGIN
Ladeveze Transverse Delamination Microdebonding Rate Fixed Format Test
2022 0
/FAIL/LAD_TRANSVERSE_DELAMINATION_MICRODEBONDING_RATE/2750
Ladeveze Transverse Microdebonding Law
{c1}
{c2}
{c3}
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 2750 in model.fail_ladtransversedelaminationmicrodebondingrates
    f = model.fail_ladtransversedelaminationmicrodebondingrates[2750]
    assert pytest.approx(f.sigma_tdmdbr0) == 1050.0
    assert pytest.approx(f.sigma_tdmdbrc) == 3150.0
    assert pytest.approx(f.gamma_tdmdbr) == 850.0
    assert pytest.approx(f.p_tdmdbr) == 16.50
    assert pytest.approx(f.d_tdmdbr_max) == 0.450
    assert f.ifail_sh == 2
    assert f.ifail_so == 1
    assert f.fail_id == 750
    assert pytest.approx(f.sigma_tdmdb0) == 1050.0
    assert pytest.approx(f.sigma_tdmdbc) == 3150.0
    assert pytest.approx(f.gamma_tdmdb) == 850.0
    assert pytest.approx(f.p_tdmdb) == 16.50
    assert pytest.approx(f.d_tdmdb_max) == 0.450
    assert pytest.approx(f.sigma_tdm0) == 1050.0
    assert pytest.approx(f.sigma_tdmc) == 3150.0
    assert pytest.approx(f.gamma_tdm) == 850.0
    assert pytest.approx(f.p_tdm) == 16.50
    assert pytest.approx(f.d_tdm_max) == 0.450
    assert pytest.approx(f.sigma_timdbr0) == 1050.0
    assert pytest.approx(f.sigma_timdbrc) == 3150.0
    assert pytest.approx(f.gamma_timdbr) == 850.0
    assert pytest.approx(f.p_timdbr) == 16.50
    assert pytest.approx(f.d_timdbr_max) == 0.450


def test_m450_fail_lad_transverse_delamination_microdebonding_rate_free(tmp_path: Path):
    deck = """# RADIOSS FREE DECK
/BEGIN
Ladeveze Transverse Delamination Microdebonding Rate Free Format Test
/FAIL/LAD_TRANSVERSE_DELAMINATION_MICRODEBONDING_RATE/2751
1150.0, 3450.0, 950.0, 18.50, 0.550
1, 2
850
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 2751 in model.fail_ladtransversedelaminationmicrodebondingrates
    f = model.fail_ladtransversedelaminationmicrodebondingrates[2751]
    assert pytest.approx(f.sigma_tdmdbr0) == 1150.0
    assert pytest.approx(f.sigma_tdmdbrc) == 3450.0
    assert pytest.approx(f.gamma_tdmdbr) == 950.0
    assert pytest.approx(f.p_tdmdbr) == 18.50
    assert pytest.approx(f.d_tdmdbr_max) == 0.550
    assert f.ifail_sh == 1
    assert f.ifail_so == 2
    assert f.fail_id == 850


def test_m450_fail_lad_transverse_delamination_microdebonding_rate_aliases(tmp_path: Path):
    deck = """# RADIOSS ALIAS DECK
/BEGIN
Ladeveze Transverse Delamination Microdebonding Rate Aliases Test
/FAIL/LADEVEZE_TRANSVERSE_DELAMINATION_MICRODEBONDING_RATE/2752
1050.0, 3150.0, 850.0, 16.50, 0.450
1, 1
/FAIL/LAD_TRANSVERSE_DELAMINATION_MICRODEBOND_RATE/2753
1050.0, 3150.0, 850.0, 16.50, 0.450
1, 1
/FAIL/LADEVEZE_TRANSVERSE_DELAMINATION_MICRODEBOND_RATE/2754
1050.0, 3150.0, 850.0, 16.50, 0.450
1, 1
/FAIL/LAD_TRANSVERSE_DELAM_MICRODEBONDING_RATE/2755
1050.0, 3150.0, 850.0, 16.50, 0.450
1, 1
/FAIL/LADEVEZE_TRANSVERSE_DELAM_MICRODEBONDING_RATE/2756
1050.0, 3150.0, 850.0, 16.50, 0.450
1, 1
/FAIL/LAD_TRANSVERSE_DELAM_MICRODEBOND_RATE/2757
1050.0, 3150.0, 850.0, 16.50, 0.450
1, 1
/FAIL/LADEVEZE_TRANSVERSE_DELAM_MICRODEBOND_RATE/2758
1050.0, 3150.0, 850.0, 16.50, 0.450
1, 1
/FAIL/LAD_TDMDBR/2759
1050.0, 3150.0, 850.0, 16.50, 0.450
1, 1
/FAIL/LAD_TDMDBR_MODEL/2760
1050.0, 3150.0, 850.0, 16.50, 0.450
1, 1
/FAIL/LAD_TDMDBR_LAW/2761
1050.0, 3150.0, 850.0, 16.50, 0.450
1, 1
/FAIL/LADEVEZE_RATE_DEPENDENT_TRANSVERSE_DELAMINATION_MICRODEBONDING/2762
1050.0, 3150.0, 850.0, 16.50, 0.450
1, 1
/FAIL/LAD_TRANSVERSE_INTERLAMINAR_MICRODEBONDING_RATE/2763
1050.0, 3150.0, 850.0, 16.50, 0.450
1, 1
/FAIL/LADEVEZE_TRANSVERSE_INTERLAMINAR_MICRODEBONDING_RATE/2764
1050.0, 3150.0, 850.0, 16.50, 0.450
1, 1
/FAIL/LAD_TRANSVERSE_INTERLAMINAR_MICRODEBOND_RATE/2765
1050.0, 3150.0, 850.0, 16.50, 0.450
1, 1
/FAIL/LADEVEZE_TRANSVERSE_INTERLAMINAR_MICRODEBOND_RATE/2766
1050.0, 3150.0, 850.0, 16.50, 0.450
1, 1
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    for mid in range(2752, 2767):
        assert mid in model.fail_ladtransversedelaminationmicrodebondingrates
        f = model.fail_ladtransversedelaminationmicrodebondingrates[mid]
        assert pytest.approx(f.sigma_tdmdbr0) == 1050.0
        assert pytest.approx(f.sigma_tdmdbrc) == 3150.0
        assert pytest.approx(f.gamma_tdmdbr) == 850.0
        assert pytest.approx(f.p_tdmdbr) == 16.50
        assert pytest.approx(f.d_tdmdbr_max) == 0.450


def test_m450_eng_electrothermoflexomagnetochiralmagnonicplasmonicpolaritonic_resonance_energy_fixed(tmp_path: Path):
    c1 = f"{0.00450:>20.6e}{80:>10d}"
    deck = f"""# RADIOSS STARTER DECK
/BEGIN
Engine Electrothermoflexomagnetochiralmagnonicplasmonicpolaritonic Resonance Energy Fixed Format Test
2022 0
/ENG/ELECTROTHERMOFLEXOMAGNETOCHIRALMAGNONICPLASMONICPOLARITONIC_RESONANCE_ENERGY/1
Engine Electrothermoflexomagnetochiralmagnonicplasmonicpolaritonic Resonance Energy Directive
{c1}
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 1 in model.eng_electrothermoflexomagnetochiralmagnonicplasmonicpolaritonic_resonance_energies
    r = model.eng_electrothermoflexomagnetochiralmagnonicplasmonicpolaritonic_resonance_energies[1]
    assert pytest.approx(r.dt_etfcmagplp) == 0.00450
    assert r.sens_id == 80


def test_m450_eng_electrothermoflexomagnetochiralmagnonicplasmonicpolaritonic_resonance_energy_free(tmp_path: Path):
    deck = """# RADIOSS FREE DECK
/BEGIN
Engine Electrothermoflexomagnetochiralmagnonicplasmonicpolaritonic Resonance Energy Free Format Test
/ENG/ELECTROTHERMOFLEXOMAGNETOCHIRALMAGNONICPLASMONICPOLARITONIC_RESONANCE_ENERGY/2
0.00550, 90
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 2 in model.eng_electrothermoflexomagnetochiralmagnonicplasmonicpolaritonic_resonance_energies
    r = model.eng_electrothermoflexomagnetochiralmagnonicplasmonicpolaritonic_resonance_energies[2]
    assert pytest.approx(r.dt_etfcmagplp) == 0.00550
    assert r.sens_id == 90


def test_m450_eng_electrothermoflexomagnetochiralmagnonicplasmonicpolaritonic_resonance_energy_aliases(tmp_path: Path):
    deck = """# RADIOSS ALIAS DECK
/BEGIN
Engine Electrothermoflexomagnetochiralmagnonicplasmonicpolaritonic Resonance Energy Aliases Test
/ENG/ELECTRO_THERM_FLEXO_MAG_CHIRAL_MAGNON_PLASMON_POLARITON_RES_WORK/3
0.00450, 80
/ENG/EELECTROTHERMOFLEXOMAGNETOCHIRALMAGNONICPLASMONICPOLARITONICRESONANCE/4
0.00450, 80
/ENG/ELECTROTHERMOFLEXOMAGNETOCHIRALMAGNONICPLASMONICPOLARITONIC_RESONANCE_DISSIPATION/5
0.00450, 80
/ENG/ET_ELECTROTHERMOFLEXOMAGNETOCHIRALMAGNONICPLASMONICPOLARITONIC_RESONANCE/6
0.00450, 80
/ENG/ELECTROTHERMOFLEXOMAGNETOMAGNONICCHIRALPLASMONICPOLARITONIC_RESONANCE_ENERGY/7
0.00450, 80
/ENG/ELECTRO_THERM_FLEXO_MAG_MAGNON_CHIRAL_PLASMON_POLARITON_RES_WORK/8
0.00450, 80
/ENG/EELECTROTHERMOFLEXOMAGNETOMAGNONICCHIRALPLASMONICPOLARITONICRESONANCE/9
0.00450, 80
/ENG/ELECTROTHERMOFLEXOMAGNETOMAGNONICCHIRALPLASMONICPOLARITONIC_RESONANCE_DISSIPATION/10
0.00450, 80
/ENG/ET_ELECTROTHERMOFLEXOMAGNETOMAGNONICCHIRALPLASMONICPOLARITONIC_RESONANCE/11
0.00450, 80
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    for rid in range(3, 12):
        assert rid in model.eng_electrothermoflexomagnetochiralmagnonicplasmonicpolaritonic_resonance_energies
        r = model.eng_electrothermoflexomagnetochiralmagnonicplasmonicpolaritonic_resonance_energies[rid]
        assert pytest.approx(r.dt_etfcmagplp) == 0.00450
        assert r.sens_id == 80


def test_m450_lagmul_orbifold_spinor_spatial_linkage_joint_fixed(tmp_path: Path):
    c1 = f"{101:>10d}{102:>10d}{103:>10d}{3.8e6:>20.4f}{7:>10d}{2.4e-5:>20.6e}"
    c2 = f"{23.0:>20.4f}{25.0:>20.4f}{75.0:>20.4f}{9.0:>20.4f}"
    deck = f"""# RADIOSS STARTER DECK
/BEGIN
Orbifold Spinor Spatial Linkage Joint Fixed Format Test
2022 0
/ORBIFOLD_SPINOR_SPATIAL_LINKAGE_JOINT/1
Orbifold Spinor Spatial Mechanism Joint
{c1}
{c2}
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 1 in model.lagmul_orbifold_spinor_spatial_linkage_joints
    j = model.lagmul_orbifold_spinor_spatial_linkage_joints[1]
    assert j.node1 == 101
    assert j.node2 == 102
    assert j.node3 == 103
    assert pytest.approx(j.stiff) == 3.8e6
    assert j.skew_id == 7
    assert pytest.approx(j.tol) == 2.4e-5
    assert pytest.approx(j.link_len_a) == 23.0
    assert pytest.approx(j.link_len_b) == 25.0
    assert pytest.approx(j.twist_angle_alpha) == 75.0
    assert pytest.approx(j.offset_distance_s) == 9.0
    assert pytest.approx(j.offset_distance_r) == 9.0
    assert pytest.approx(j.offset_distance_v) == 9.0
    assert pytest.approx(j.offset_distance_h) == 9.0
    assert pytest.approx(j.offset_distance_u) == 9.0
    assert pytest.approx(j.offset_distance_f) == 9.0


def test_m450_lagmul_orbifold_spinor_spatial_linkage_joint_free(tmp_path: Path):
    deck = """# RADIOSS FREE DECK
/BEGIN
Orbifold Spinor Spatial Linkage Joint Free Format Test
/LAGMUL/ORBIFOLD_SPINOR_SPATIAL_LINKAGE_JOINT/2
Orbifold Spinor Spatial Mechanism Joint Free
201, 202, 203, 4.8e6, 8, 3.4e-5
33.0, 35.0, 85.0, 14.0
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 2 in model.lagmul_orbifold_spinor_spatial_linkage_joints
    j = model.lagmul_orbifold_spinor_spatial_linkage_joints[2]
    assert j.node1 == 201
    assert j.node2 == 202
    assert j.node3 == 203
    assert pytest.approx(j.stiff) == 4.8e6
    assert j.skew_id == 8
    assert pytest.approx(j.tol) == 3.4e-5
    assert pytest.approx(j.link_len_a) == 33.0
    assert pytest.approx(j.link_len_b) == 35.0
    assert pytest.approx(j.twist_angle_alpha) == 85.0
    assert pytest.approx(j.offset_distance_s) == 14.0


def test_m450_lagmul_orbifold_spinor_spatial_linkage_joint_aliases(tmp_path: Path):
    deck = """# RADIOSS ALIAS DECK
/BEGIN
Orbifold Spinor Spatial Linkage Joint Aliases Test
/ORBIFOLD_SPINOR_SPATIAL_LINKAGE/3
101, 102, 103, 3.8e6, 7, 2.4e-5
23.0, 25.0, 75.0, 9.0
/LAGMUL/ORBIFOLD_SPINOR_SPATIAL_LINKAGE/4
101, 102, 103, 3.8e6, 7, 2.4e-5
23.0, 25.0, 75.0, 9.0
/ORBIFOLD_SPINOR_SPATIAL_MULTI_LOOP_MECHANISM/5
101, 102, 103, 3.8e6, 7, 2.4e-5
23.0, 25.0, 75.0, 9.0
/ORBIFOLD_SPINOR_SPATIAL_SYMMETRIC_MECHANISM/6
101, 102, 103, 3.8e6, 7, 2.4e-5
23.0, 25.0, 75.0, 9.0
/ORBIFOLD_SPINOR_SPATIAL_6R_MECHANISM/7
101, 102, 103, 3.8e6, 7, 2.4e-5
23.0, 25.0, 75.0, 9.0
/ORBIFOLD_SPINOR_SPATIAL_OVERCONSTRAINED_MECHANISM/8
101, 102, 103, 3.8e6, 7, 2.4e-5
23.0, 25.0, 75.0, 9.0
/LAGMUL/ORBIFOLD_TWISTOR_SPATIAL_LINKAGE_JOINT/9
101, 102, 103, 3.8e6, 7, 2.4e-5
23.0, 25.0, 75.0, 9.0
/ORBIFOLD_TWISTOR_SPATIAL_LINKAGE_JOINT/10
101, 102, 103, 3.8e6, 7, 2.4e-5
23.0, 25.0, 75.0, 9.0
/LAGMUL/ORBIFOLD_SPINOR_BUNDLE_SPATIAL_LINKAGE_JOINT/11
101, 102, 103, 3.8e6, 7, 2.4e-5
23.0, 25.0, 75.0, 9.0
/ORBIFOLD_SPINOR_BUNDLE_SPATIAL_LINKAGE_JOINT/12
101, 102, 103, 3.8e6, 7, 2.4e-5
23.0, 25.0, 75.0, 9.0
/LAGMUL/ORBIFOLD_CLIFFORD_SPINOR_SPATIAL_LINKAGE_JOINT/13
101, 102, 103, 3.8e6, 7, 2.4e-5
23.0, 25.0, 75.0, 9.0
/ORBIFOLD_CLIFFORD_SPINOR_SPATIAL_LINKAGE_JOINT/14
101, 102, 103, 3.8e6, 7, 2.4e-5
23.0, 25.0, 75.0, 9.0
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    for jid in range(3, 15):
        assert jid in model.lagmul_orbifold_spinor_spatial_linkage_joints
        j = model.lagmul_orbifold_spinor_spatial_linkage_joints[jid]
        assert j.node1 == 101
        assert j.node2 == 102
        assert j.node3 == 103
        assert pytest.approx(j.stiff) == 3.8e6


def test_m450_sensor_spring_transverse_crackle_rate_fixed(tmp_path: Path):
    c1 = f"{1195:>10d}{9.45e6:>20.4f}{0.0075:>20.4f}"
    deck = f"""# RADIOSS STARTER DECK
/BEGIN
Sensor Spring Transverse Crackle Rate Fixed Format Test
2022 0
/SENSOR/SPRING_TRANSVERSE_CRACKLE_RATE/1
Spring Transverse Crackle Rate Sensor
{c1}
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 1 in model.sensor_spring_transverse_crackle_rates
    sensor = model.sensor_spring_transverse_crackle_rates[1]
    assert sensor.spring_id == 1195
    assert pytest.approx(sensor.jtrans_crk_max) == 9.45e6
    assert pytest.approx(sensor.t_delay) == 0.0075
    assert len(model.sensors) == 1
    assert model.sensors[0].kind == "SPRING_TRANSVERSE_CRACKLE_RATE"


def test_m450_sensor_spring_transverse_crackle_rate_free(tmp_path: Path):
    deck = """# RADIOSS FREE DECK
/BEGIN
Sensor Spring Transverse Crackle Rate Free Format Test
/SENSOR/SPRING_TRANSVERSE_CRACKLE_RATE/2
25, 4.8e5, 0.002
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 2 in model.sensor_spring_transverse_crackle_rates
    sensor = model.sensor_spring_transverse_crackle_rates[2]
    assert sensor.spring_id == 25
    assert pytest.approx(sensor.jtrans_crk_max) == 4.8e5
    assert pytest.approx(sensor.t_delay) == 0.002
    assert len(model.sensors) == 1
    assert model.sensors[0].kind == "SPRING_TRANSVERSE_CRACKLE_RATE"


def test_m450_sensor_spring_transverse_crackle_rate_aliases(tmp_path: Path):
    deck = """# RADIOSS ALIAS DECK
/BEGIN
Sensor Spring Transverse Crackle Rate Aliases Test
/SENSOR/SPRING_TRANS_CRK_RATE/3
35, 4.8e5, 0.002
/SENSOR/SPRING_TRANSVERSE_CRACKLE/4
35, 4.8e5, 0.002
/SENSOR/SPRING_TRANS_CRACKLE/5
35, 4.8e5, 0.002
/SENSOR/SPRING_TRANSVERSE_CRK/6
35, 4.8e5, 0.002
/SENSOR/SPRING_TRANS_CRK/7
35, 4.8e5, 0.002
/SENSOR/SPRING_POP_RATE_TRANSVERSE_CRACKLE/8
35, 4.8e5, 0.002
/SENSOR/TRANSVERSE_CRACKLE_RATE_SPRING/9
35, 4.8e5, 0.002
/SENSOR/SPRING_CRACKLE_RATE_TRANSVERSE/10
35, 4.8e5, 0.002
/SENSOR/SPRING_CRACKLE_RATE_TRANS/11
35, 4.8e5, 0.002
/SENSOR/SPRING_CRK_RATE_TRANSVERSE/12
35, 4.8e5, 0.002
/SENSOR/SPRING_CRK_RATE_TRANS/13
35, 4.8e5, 0.002
/SENSOR/SPRING_TRANS_CRACKLE_RATE/14
35, 4.8e5, 0.002
/SENSOR/SPRING_RATE_CRACKLE_TRANS/15
35, 4.8e5, 0.002
/SENSOR/SPRING_CRACKLE_TRANS/16
35, 4.8e5, 0.002
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    for sid in range(3, 17):
        assert sid in model.sensor_spring_transverse_crackle_rates
        s = model.sensor_spring_transverse_crackle_rates[sid]
        assert s.spring_id == 35
        assert pytest.approx(s.jtrans_crk_max) == 4.8e5


def test_m450_missing_card_errors(tmp_path: Path):
    deck = """# RADIOSS ERROR DECK
/BEGIN
Error Handling Test
/FAIL/LAD_TRANSVERSE_DELAMINATION_MICRODEBONDING_RATE/1
/ENG/ELECTROTHERMOFLEXOMAGNETOCHIRALMAGNONICPLASMONICPOLARITONIC_RESONANCE_ENERGY/1
/ORBIFOLD_SPINOR_SPATIAL_LINKAGE_JOINT/1
/SENSOR/SPRING_TRANSVERSE_CRACKLE_RATE/1
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 4
