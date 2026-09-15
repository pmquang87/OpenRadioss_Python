"""Tests for Milestone M444: LadTransverseCoreMicroyieldingRate Failure Model, EngElectrothermoflexomagnetophononicexcitonicplasmonicpolaritonicResonanceEnergy, LagmulHomologicalSpinorSpatialLinkageJoint, and SensorSpringTransversePopRate."""

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


def test_m444_fail_lad_transverse_core_microyielding_rate_fixed(tmp_path: Path):
    c1 = f"{1030.0:>20.4f}{3090.0:>20.4f}{830.0:>20.4f}{16.80:>20.4f}{0.415:>20.4f}"
    c2 = f"{1:>10d}{2:>10d}"
    c3 = f"{2610:>10d}"
    deck = f"""# RADIOSS STARTER DECK
/BEGIN
Ladeveze Transverse Core Microyielding Rate Fixed Format Test
2022 0
/FAIL/LAD_TRANSVERSE_CORE_MICROYIELDING_RATE/2610
Ladeveze Transverse Core Microyielding Failure Model
{c1}
{c2}
{c3}
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 2610 in model.fail_ladtransversecoremicroyieldingrates
    ftcmyr = model.fail_ladtransversecoremicroyieldingrates[2610]
    assert pytest.approx(ftcmyr.sigma_tcmyr0) == 1030.0
    assert pytest.approx(ftcmyr.sigma_tcmyrc) == 3090.0
    assert pytest.approx(ftcmyr.gamma_tcmyr) == 830.0
    assert pytest.approx(ftcmyr.p_tcmyr) == 16.80
    assert pytest.approx(ftcmyr.d_tcmyr_max) == 0.415
    assert ftcmyr.ifail_sh == 1
    assert ftcmyr.ifail_so == 2
    assert ftcmyr.fail_id == 2610
    assert pytest.approx(ftcmyr.sigma_tcmy0) == 1030.0
    assert pytest.approx(ftcmyr.sigma_tcmyc) == 3090.0
    assert pytest.approx(ftcmyr.gamma_tcmy) == 830.0
    assert pytest.approx(ftcmyr.p_tcmy) == 16.80
    assert pytest.approx(ftcmyr.d_tcmy_max) == 0.415
    assert pytest.approx(ftcmyr.sigma_tcmpr0) == 1030.0
    assert pytest.approx(ftcmyr.sigma_tcmprc) == 3090.0
    assert pytest.approx(ftcmyr.gamma_tcmpr) == 830.0
    assert pytest.approx(ftcmyr.p_tcmpr) == 16.80
    assert pytest.approx(ftcmyr.d_tcmpr_max) == 0.415
    assert pytest.approx(ftcmyr.sigma_tcmfr0) == 1030.0
    assert pytest.approx(ftcmyr.sigma_tcmfrc) == 3090.0
    assert pytest.approx(ftcmyr.gamma_tcmfr) == 830.0
    assert pytest.approx(ftcmyr.p_tcmfr) == 16.80
    assert pytest.approx(ftcmyr.d_tcmfr_max) == 0.415
    assert len(model.raw_fails) == 1
    assert model.raw_fails[0][1].type == "LAD_TRANSVERSE_CORE_MICROYIELDING_RATE"


def test_m444_fail_lad_transverse_core_microyielding_rate_free(tmp_path: Path):
    deck = """# RADIOSS FREE DECK
/BEGIN
Ladeveze Transverse Core Microyielding Rate Free Format Test
/FAIL/LAD_TRANSVERSE_CORE_MICROYIELDING_RATE/2611
1050.0, 3150.0, 850.0, 17.10, 0.405
1, 1
2611
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 2611 in model.fail_ladtransversecoremicroyieldingrates
    ftcmyr = model.fail_ladtransversecoremicroyieldingrates[2611]
    assert pytest.approx(ftcmyr.sigma_tcmyr0) == 1050.0
    assert pytest.approx(ftcmyr.sigma_tcmyrc) == 3150.0
    assert pytest.approx(ftcmyr.gamma_tcmyr) == 850.0
    assert pytest.approx(ftcmyr.p_tcmyr) == 17.10
    assert pytest.approx(ftcmyr.d_tcmyr_max) == 0.405
    assert ftcmyr.fail_id == 2611


def test_m444_fail_lad_transverse_core_microyielding_rate_aliases(tmp_path: Path):
    deck = """# RADIOSS ALIAS DECK
/BEGIN
Ladeveze Transverse Core Microyielding Rate Aliases Test
/FAIL/LADEVEZE_TRANSVERSE_CORE_MICROYIELDING_RATE/2612
980.0, 2940.0, 780.0, 15.60, 0.455
1, 1
/FAIL/LAD_TRANSVERSE_CORE_MICROYIELD_RATE/2613
980.0, 2940.0, 780.0, 15.60, 0.455
1, 1
/FAIL/LADEVEZE_TRANSVERSE_CORE_MICROYIELD_RATE/2614
980.0, 2940.0, 780.0, 15.60, 0.455
1, 1
/FAIL/LAD_TCMYR/2615
980.0, 2940.0, 780.0, 15.60, 0.455
1, 1
/FAIL/LAD_TCMYR_MODEL/2616
980.0, 2940.0, 780.0, 15.60, 0.455
1, 1
/FAIL/LAD_TCMYR_LAW/2617
980.0, 2940.0, 780.0, 15.60, 0.455
1, 1
/FAIL/LADEVEZE_RATE_DEPENDENT_TRANSVERSE_CORE_MICROYIELDING/2618
980.0, 2940.0, 780.0, 15.60, 0.455
1, 1
/FAIL/LAD_TRANSVERSE_CORE_MICROPLASTICITY_RATE/2619
980.0, 2940.0, 780.0, 15.60, 0.455
1, 1
/FAIL/LADEVEZE_TRANSVERSE_CORE_MICROPLASTICITY_RATE/2620
980.0, 2940.0, 780.0, 15.60, 0.455
1, 1
/FAIL/LAD_TRANSVERSE_CORE_MICROPLASTIC_RATE/2621
980.0, 2940.0, 780.0, 15.60, 0.455
1, 1
/FAIL/LADEVEZE_TRANSVERSE_CORE_MICROPLASTIC_RATE/2622
980.0, 2940.0, 780.0, 15.60, 0.455
1, 1
/FAIL/LAD_TRANSVERSE_CORE_MICROFLOW_RATE/2623
980.0, 2940.0, 780.0, 15.60, 0.455
1, 1
/FAIL/LADEVEZE_TRANSVERSE_CORE_MICROFLOW_RATE/2624
980.0, 2940.0, 780.0, 15.60, 0.455
1, 1
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    for mid in range(2612, 2625):
        assert mid in model.fail_ladtransversecoremicroyieldingrates
        f = model.fail_ladtransversecoremicroyieldingrates[mid]
        assert pytest.approx(f.sigma_tcmyr0) == 980.0
        assert pytest.approx(f.sigma_tcmyrc) == 2940.0


def test_m444_fail_lad_transverse_core_microyielding_rate_properties():
    from pyradioss.model.entities import FailLadTransverseCoreMicroyieldingRate
    f = FailLadTransverseCoreMicroyieldingRate(mat_id=1)
    f.sigma_tcmy0 = 1090.0
    assert pytest.approx(f.sigma_tcmyr0) == 1090.0
    assert pytest.approx(f.sigma_tcmy0) == 1090.0

    f.sigma_tcmyc = 3270.0
    assert pytest.approx(f.sigma_tcmyrc) == 3270.0
    assert pytest.approx(f.sigma_tcmyc) == 3270.0

    f.gamma_tcmy = 870.0
    assert pytest.approx(f.gamma_tcmyr) == 870.0
    assert pytest.approx(f.gamma_tcmy) == 870.0

    f.p_tcmy = 16.90
    assert pytest.approx(f.p_tcmyr) == 16.90
    assert pytest.approx(f.p_tcmy) == 16.90

    f.d_tcmy_max = 0.395
    assert pytest.approx(f.d_tcmyr_max) == 0.395
    assert pytest.approx(f.d_tcmy_max) == 0.395

    f.sigma_tcmpr0 = 1150.0
    assert pytest.approx(f.sigma_tcmyr0) == 1150.0
    assert pytest.approx(f.sigma_tcmpr0) == 1150.0

    f.sigma_tcmprc = 3450.0
    assert pytest.approx(f.sigma_tcmyrc) == 3450.0
    assert pytest.approx(f.sigma_tcmprc) == 3450.0

    f.gamma_tcmpr = 910.0
    assert pytest.approx(f.gamma_tcmyr) == 910.0
    assert pytest.approx(f.gamma_tcmpr) == 910.0

    f.p_tcmpr = 17.90
    assert pytest.approx(f.p_tcmyr) == 17.90
    assert pytest.approx(f.p_tcmpr) == 17.90

    f.d_tcmpr_max = 0.365
    assert pytest.approx(f.d_tcmyr_max) == 0.365
    assert pytest.approx(f.d_tcmpr_max) == 0.365


def test_m444_eng_electrothermoflexomagnetophononicexcitonicplasmonicpolaritonic_resonance_energy_fixed(tmp_path: Path):
    c1 = f"{0.00175:>20.6f}{59:>10d}"
    deck = f"""# RADIOSS STARTER DECK
/BEGIN
Engine Phonon Exciton Plasmon Polaritonic Resonance Energy Fixed Format Test
2022 0
/ENG/ELECTROTHERMOFLEXOMAGNETOPHONONICEXCITONICPLASMONICPOLARITONIC_RESONANCE_ENERGY/1
Phonon Exciton Plasmon Polaritonic Resonance Energy Title
{c1}
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 1 in model.eng_electrothermoflexomagnetophononicexcitonicplasmonicpolaritonic_resonance_energies
    r = model.eng_electrothermoflexomagnetophononicexcitonicplasmonicpolaritonic_resonance_energies[1]
    assert pytest.approx(r.dt_etfpexplp) == 0.00175
    assert pytest.approx(r.dt_etfplp) == 0.00175
    assert r.sens_id == 59


def test_m444_eng_electrothermoflexomagnetophononicexcitonicplasmonicpolaritonic_resonance_energy_free(tmp_path: Path):
    deck = """# RADIOSS FREE DECK
/BEGIN
Engine Phonon Exciton Plasmon Polaritonic Resonance Energy Free Format Test
/ENG/ELECTROTHERMOFLEXOMAGNETOPHONONICEXCITONICPLASMONICPOLARITONIC_RESONANCE_ENERGY/2
0.00275, 60
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 2 in model.eng_electrothermoflexomagnetophononicexcitonicplasmonicpolaritonic_resonance_energies
    r = model.eng_electrothermoflexomagnetophononicexcitonicplasmonicpolaritonic_resonance_energies[2]
    assert pytest.approx(r.dt_etfpexplp) == 0.00275
    assert r.sens_id == 60


def test_m444_eng_electrothermoflexomagnetophononicexcitonicplasmonicpolaritonic_resonance_energy_aliases(tmp_path: Path):
    deck = """# RADIOSS ALIAS DECK
/BEGIN
Engine Phonon Exciton Plasmon Polaritonic Resonance Energy Aliases Test
/ENG/ELECTRO_THERM_FLEXO_MAG_PHONON_EXCITON_PLASMON_POLARITON_RES_WORK/3
0.00375, 61
/ENG/EELECTROTHERMOFLEXOMAGNETOPHONONICEXCITONICPLASMONICPOLARITONICRESONANCE/4
0.00375, 61
/ENG/ELECTROTHERMOFLEXOMAGNETOPHONONICEXCITONICPLASMONICPOLARITONIC_RESONANCE_DISSIPATION/5
0.00375, 61
/ENG/ET_ELECTROTHERMOFLEXOMAGNETOPHONONICEXCITONICPLASMONICPOLARITONIC_RESONANCE/6
0.00375, 61
/ENG/ELECTROTHERMOFLEXOMAGNETOEXCITONICPHONONICPLASMONICPOLARITONIC_RESONANCE_ENERGY/7
0.00375, 61
/ENG/ELECTRO_THERM_FLEXO_MAG_EXCITON_PHONON_PLASMON_POLARITON_RES_WORK/8
0.00375, 61
/ENG/EELECTROTHERMOFLEXOMAGNETOEXCITONICPHONONICPLASMONICPOLARITONICRESONANCE/9
0.00375, 61
/ENG/ELECTROTHERMOFLEXOMAGNETOEXCITONICPHONONICPLASMONICPOLARITONIC_RESONANCE_DISSIPATION/10
0.00375, 61
/ENG/ET_ELECTROTHERMOFLEXOMAGNETOEXCITONICPHONONICPLASMONICPOLARITONIC_RESONANCE/11
0.00375, 61
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    for rid in range(3, 12):
        assert rid in model.eng_electrothermoflexomagnetophononicexcitonicplasmonicpolaritonic_resonance_energies
        r = model.eng_electrothermoflexomagnetophononicexcitonicplasmonicpolaritonic_resonance_energies[rid]
        assert pytest.approx(r.dt_etfpexplp) == 0.00375
        assert r.sens_id == 61


def test_m444_lagmul_homological_spinor_spatial_linkage_joint_fixed(tmp_path: Path):
    c1 = f"{101:>10d}{102:>10d}{103:>10d}{3.2e6:>20.4f}{6:>10d}{1.7e-5:>20.6e}"
    c2 = f"{17.0:>20.4f}{19.0:>20.4f}{62.0:>20.4f}{5.5:>20.4f}"
    deck = f"""# RADIOSS STARTER DECK
/BEGIN
Homological Spinor Spatial Linkage Joint Fixed Format Test
2022 0
/HOMOLOGICAL_SPINOR_SPATIAL_LINKAGE_JOINT/1
Homological Spinor Spatial Mechanism Joint
{c1}
{c2}
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 1 in model.lagmul_homological_spinor_spatial_linkage_joints
    j = model.lagmul_homological_spinor_spatial_linkage_joints[1]
    assert j.node1 == 101
    assert j.node2 == 102
    assert j.node3 == 103
    assert pytest.approx(j.stiff) == 3.2e6
    assert j.skew_id == 6
    assert pytest.approx(j.tol) == 1.7e-5
    assert pytest.approx(j.link_len_a) == 17.0
    assert pytest.approx(j.link_len_b) == 19.0
    assert pytest.approx(j.twist_angle_alpha) == 62.0
    assert pytest.approx(j.offset_distance_s) == 5.5
    assert pytest.approx(j.offset_distance_r) == 5.5
    assert pytest.approx(j.offset_distance_v) == 5.5
    assert pytest.approx(j.offset_distance_h) == 5.5
    assert pytest.approx(j.offset_distance_u) == 5.5
    assert pytest.approx(j.offset_distance_f) == 5.5


def test_m444_lagmul_homological_spinor_spatial_linkage_joint_free(tmp_path: Path):
    deck = """# RADIOSS FREE DECK
/BEGIN
Homological Spinor Spatial Linkage Joint Free Format Test
/LAGMUL/HOMOLOGICAL_SPINOR_SPATIAL_LINKAGE_JOINT/2
201, 202, 203, 4.2e6, 7, 2.7e-5
20.0, 22.0, 74.0, 6.5
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 2 in model.lagmul_homological_spinor_spatial_linkage_joints
    j = model.lagmul_homological_spinor_spatial_linkage_joints[2]
    assert j.node1 == 201
    assert j.node2 == 202
    assert j.node3 == 203
    assert pytest.approx(j.stiff) == 4.2e6
    assert j.skew_id == 7
    assert pytest.approx(j.tol) == 2.7e-5
    assert pytest.approx(j.link_len_a) == 20.0
    assert pytest.approx(j.link_len_b) == 22.0
    assert pytest.approx(j.twist_angle_alpha) == 74.0
    assert pytest.approx(j.offset_distance_s) == 6.5


def test_m444_lagmul_homological_spinor_spatial_linkage_joint_aliases(tmp_path: Path):
    deck = """# RADIOSS ALIAS DECK
/BEGIN
Homological Spinor Spatial Linkage Joint Aliases Test
/LAGMUL/HOMOLOGICAL_SPINOR_SPATIAL_LINKAGE/3
301, 302, 303, 1.0e6, 0, 1.0e-6
10.0, 10.0, 30.0, 2.0
/HOMOLOGICAL_SPINOR_SPATIAL_LINKAGE/4
301, 302, 303, 1.0e6, 0, 1.0e-6
10.0, 10.0, 30.0, 2.0
/HOMOLOGICAL_SPINOR_SPATIAL_MULTI_LOOP_MECHANISM/5
301, 302, 303, 1.0e6, 0, 1.0e-6
10.0, 10.0, 30.0, 2.0
/HOMOLOGICAL_SPINOR_SPATIAL_SYMMETRIC_MECHANISM/6
301, 302, 303, 1.0e6, 0, 1.0e-6
10.0, 10.0, 30.0, 2.0
/HOMOLOGICAL_SPINOR_SPATIAL_6R_MECHANISM/7
301, 302, 303, 1.0e6, 0, 1.0e-6
10.0, 10.0, 30.0, 2.0
/HOMOLOGICAL_SPINOR_SPATIAL_OVERCONSTRAINED_MECHANISM/8
301, 302, 303, 1.0e6, 0, 1.0e-6
10.0, 10.0, 30.0, 2.0
/LAGMUL/HOMOLOGICAL_TWISTOR_SPATIAL_LINKAGE_JOINT/9
301, 302, 303, 1.0e6, 0, 1.0e-6
10.0, 10.0, 30.0, 2.0
/HOMOLOGICAL_TWISTOR_SPATIAL_LINKAGE_JOINT/10
301, 302, 303, 1.0e6, 0, 1.0e-6
10.0, 10.0, 30.0, 2.0
/LAGMUL/HOMOLOGICAL_SPINOR_BUNDLE_SPATIAL_LINKAGE_JOINT/11
301, 302, 303, 1.0e6, 0, 1.0e-6
10.0, 10.0, 30.0, 2.0
/HOMOLOGICAL_SPINOR_BUNDLE_SPATIAL_LINKAGE_JOINT/12
301, 302, 303, 1.0e6, 0, 1.0e-6
10.0, 10.0, 30.0, 2.0
/LAGMUL/HOMOLOGICAL_CLIFFORD_SPINOR_SPATIAL_LINKAGE_JOINT/13
301, 302, 303, 1.0e6, 0, 1.0e-6
10.0, 10.0, 30.0, 2.0
/HOMOLOGICAL_CLIFFORD_SPINOR_SPATIAL_LINKAGE_JOINT/14
301, 302, 303, 1.0e6, 0, 1.0e-6
10.0, 10.0, 30.0, 2.0
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    for jid in range(3, 15):
        assert jid in model.lagmul_homological_spinor_spatial_linkage_joints
        j = model.lagmul_homological_spinor_spatial_linkage_joints[jid]
        assert j.node1 == 301
        assert j.node2 == 302
        assert j.node3 == 303


def test_m444_sensor_spring_transverse_pop_rate_fixed(tmp_path: Path):
    c1 = f"{10:>10d}{1.95e5:>20.4f}{0.0025:>20.6f}"
    deck = f"""# RADIOSS STARTER DECK
/BEGIN
Sensor Spring Transverse Pop Rate Fixed Format Test
2022 0
/SENSOR/SPRING_TRANSVERSE_POP_RATE/1
Transverse Pop Rate Sensor Title
{c1}
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 1 in model.sensor_spring_transverse_pop_rates
    s = model.sensor_spring_transverse_pop_rates[1]
    assert s.spring_id == 10
    assert pytest.approx(s.jtrans_pop_max) == 1.95e5
    assert pytest.approx(s.t_delay) == 0.0025
    assert len(model.sensors) == 1
    assert model.sensors[0].kind == "SPRING_TRANSVERSE_POP_RATE"


def test_m444_sensor_spring_transverse_pop_rate_free(tmp_path: Path):
    deck = """# RADIOSS FREE DECK
/BEGIN
Sensor Spring Transverse Pop Rate Free Format Test
/SENSOR/SPRING_TRANSVERSE_POP_RATE/2
20, 2.95e5, 0.0035
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 2 in model.sensor_spring_transverse_pop_rates
    s = model.sensor_spring_transverse_pop_rates[2]
    assert s.spring_id == 20
    assert pytest.approx(s.jtrans_pop_max) == 2.95e5
    assert pytest.approx(s.t_delay) == 0.0035


def test_m444_sensor_spring_transverse_pop_rate_aliases(tmp_path: Path):
    deck = """# RADIOSS ALIAS DECK
/BEGIN
Sensor Spring Transverse Pop Rate Aliases Test
/SENSOR/SPRING_TRANS_POP_RATE/3
30, 3.6e5, 0.001
/SENSOR/SPRING_TRANSVERSE_POP/4
30, 3.6e5, 0.001
/SENSOR/SPRING_TRANS_POP/5
30, 3.6e5, 0.001
/SENSOR/SPRING_SHEAR_POP_RATE/6
30, 3.6e5, 0.001
/SENSOR/SPRING_SHEAR_POP/7
30, 3.6e5, 0.001
/SENSOR/SPRING_POP_RATE_TRANSVERSE/8
30, 3.6e5, 0.001
/SENSOR/SPRING_POP_RATE_TRANS/9
30, 3.6e5, 0.001
/SENSOR/SPRING_POP_RATE_SHEAR/10
30, 3.6e5, 0.001
/SENSOR/TRANSVERSE_POP_RATE_SPRING/11
30, 3.6e5, 0.001
/SENSOR/SPRING_POP_TRANS/12
30, 3.6e5, 0.001
/SENSOR/SPRING_POP_TRANSVERSE/13
30, 3.6e5, 0.001
/SENSOR/SPRING_POP_SHEAR/14
30, 3.6e5, 0.001
/SENSOR/SPRING_POP_RATE_SH/15
30, 3.6e5, 0.001
/SENSOR/SPRING_SH_POP_RATE/16
30, 3.6e5, 0.001
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    for sid in range(3, 17):
        assert sid in model.sensor_spring_transverse_pop_rates
        s = model.sensor_spring_transverse_pop_rates[sid]
        assert s.spring_id == 30
        assert pytest.approx(s.jtrans_pop_max) == 3.6e5


def test_m444_missing_card_errors(tmp_path: Path):
    deck = """# RADIOSS ERROR DECK
/BEGIN
Error Handling Test
/FAIL/LAD_TRANSVERSE_CORE_MICROYIELDING_RATE/1
/ENG/ELECTROTHERMOFLEXOMAGNETOPHONONICEXCITONICPLASMONICPOLARITONIC_RESONANCE_ENERGY/1
/HOMOLOGICAL_SPINOR_SPATIAL_LINKAGE_JOINT/1
/SENSOR/SPRING_TRANSVERSE_POP_RATE/1
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 4
