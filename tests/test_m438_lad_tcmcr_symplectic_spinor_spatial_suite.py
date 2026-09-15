"""Tests for Milestone M438: LadTransverseCoreMicrocrackingRate Failure Model, EngElectrothermoflexomagnetoexcitonicpolaritonicResonanceEnergy, LagmulSymplecticSpinorSpatialLinkageJoint, and SensorSpringTransverseCrackleRate."""

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


def test_m438_fail_lad_transverse_core_microcracking_rate_fixed(tmp_path: Path):
    c1 = f"{970.0:>20.4f}{2910.0:>20.4f}{750.0:>20.4f}{15.30:>20.4f}{0.480:>20.4f}"
    c2 = f"{1:>10d}{2:>10d}"
    c3 = f"{2520:>10d}"
    deck = f"""# RADIOSS STARTER DECK
/BEGIN
Ladeveze Transverse Core Microcracking Rate Fixed Format Test
2022 0
/FAIL/LAD_TRANSVERSE_CORE_MICROCRACKING_RATE/2520
Ladeveze Transverse Core Microcracking Failure Model
{c1}
{c2}
{c3}
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 2520 in model.fail_ladtransversecoremicrocrackingrates
    ftcmcr = model.fail_ladtransversecoremicrocrackingrates[2520]
    assert pytest.approx(ftcmcr.sigma_tcmcr0) == 970.0
    assert pytest.approx(ftcmcr.sigma_tcmcrc) == 2910.0
    assert pytest.approx(ftcmcr.gamma_tcmcr) == 750.0
    assert pytest.approx(ftcmcr.p_tcmcr) == 15.30
    assert pytest.approx(ftcmcr.d_tcmcr_max) == 0.480
    assert ftcmcr.ifail_sh == 1
    assert ftcmcr.ifail_so == 2
    assert ftcmcr.fail_id == 2520
    assert pytest.approx(ftcmcr.sigma_tcmd0) == 970.0
    assert pytest.approx(ftcmcr.sigma_tcmdc) == 2910.0
    assert pytest.approx(ftcmcr.gamma_tcmd) == 750.0
    assert pytest.approx(ftcmcr.p_tcmd) == 15.30
    assert pytest.approx(ftcmcr.d_tcmd_max) == 0.480
    assert len(model.raw_fails) == 1
    assert model.raw_fails[0][1].type == "LAD_TRANSVERSE_CORE_MICROCRACKING_RATE"


def test_m438_fail_lad_transverse_core_microcracking_rate_free(tmp_path: Path):
    deck = """# RADIOSS FREE DECK
/BEGIN
Ladeveze Transverse Core Microcracking Rate Free Format Test
/FAIL/LAD_TRANSVERSE_CORE_MICROCRACKING_RATE/2521
980.0, 2940.0, 760.0, 15.50, 0.470
1, 1
2521
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 2521 in model.fail_ladtransversecoremicrocrackingrates
    ftcmcr = model.fail_ladtransversecoremicrocrackingrates[2521]
    assert pytest.approx(ftcmcr.sigma_tcmcr0) == 980.0
    assert pytest.approx(ftcmcr.sigma_tcmcrc) == 2940.0
    assert pytest.approx(ftcmcr.gamma_tcmcr) == 760.0
    assert pytest.approx(ftcmcr.p_tcmcr) == 15.50
    assert pytest.approx(ftcmcr.d_tcmcr_max) == 0.470
    assert ftcmcr.fail_id == 2521


def test_m438_fail_lad_transverse_core_microcracking_rate_aliases(tmp_path: Path):
    deck = """# RADIOSS ALIAS DECK
/BEGIN
Ladeveze Transverse Core Microcracking Rate Aliases Test
/FAIL/LADEVEZE_TRANSVERSE_CORE_MICROCRACKING_RATE/2522
960.0, 2880.0, 740.0, 15.10, 0.490
1, 1
/FAIL/LAD_TRANSVERSE_CORE_MICROCRACK_RATE/2523
960.0, 2880.0, 740.0, 15.10, 0.490
1, 1
/FAIL/LAD_TCMCR/2524
960.0, 2880.0, 740.0, 15.10, 0.490
1, 1
/FAIL/LAD_TCMCR_MODEL/2525
960.0, 2880.0, 740.0, 15.10, 0.490
1, 1
/FAIL/LAD_TCMCR_LAW/2526
960.0, 2880.0, 740.0, 15.10, 0.490
1, 1
/FAIL/LADEVEZE_RATE_DEPENDENT_TRANSVERSE_CORE_MICROCRACKING/2527
960.0, 2880.0, 740.0, 15.10, 0.490
1, 1
/FAIL/LAD_TRANSVERSE_CORE_MICRODAMAGE_RATE/2528
960.0, 2880.0, 740.0, 15.10, 0.490
1, 1
/FAIL/LADEVEZE_TRANSVERSE_CORE_MICRODAMAGE_RATE/2529
960.0, 2880.0, 740.0, 15.10, 0.490
1, 1
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    for fid in [2522, 2523, 2524, 2525, 2526, 2527, 2528, 2529]:
        assert fid in model.fail_ladtransversecoremicrocrackingrates
        assert fid in model.fail_ladtransversecoremicrocrackrates
        assert fid in model.fail_ladtransversecoremicrodamagerates
        ftcmcr = model.fail_ladtransversecoremicrocrackingrates[fid]
        assert pytest.approx(ftcmcr.sigma_tcmcr0) == 960.0
        assert pytest.approx(ftcmcr.sigma_tcmcrc) == 2880.0


def test_m438_fail_lad_transverse_core_microcracking_rate_properties():
    from pyradioss.model.entities import FailLadTransverseCoreMicrocrackingRate
    f = FailLadTransverseCoreMicrocrackingRate(mat_id=1)
    f.sigma_tcmd0 = 990.0
    assert pytest.approx(f.sigma_tcmcr0) == 990.0
    assert pytest.approx(f.sigma_tcmd0) == 990.0

    f.sigma_tcmdc = 2970.0
    assert pytest.approx(f.sigma_tcmcrc) == 2970.0
    assert pytest.approx(f.sigma_tcmdc) == 2970.0

    f.gamma_tcmd = 770.0
    assert pytest.approx(f.gamma_tcmcr) == 770.0
    assert pytest.approx(f.gamma_tcmd) == 770.0

    f.p_tcmd = 15.60
    assert pytest.approx(f.p_tcmcr) == 15.60
    assert pytest.approx(f.p_tcmd) == 15.60

    f.d_tcmd_max = 0.460
    assert pytest.approx(f.d_tcmcr_max) == 0.460
    assert pytest.approx(f.d_tcmd_max) == 0.460


def test_m438_eng_electrothermoflexomagnetoexcitonicpolaritonic_resonance_energy_fixed(tmp_path: Path):
    c1 = f"{0.00075:>20.8f}{10:>10d}"
    deck = f"""# ENGINE DECK FIXED
/BEGIN
Engine Electrothermoflexomagnetoexcitonicpolaritonic Resonance Energy Fixed Format Test
2022 0
/ENG/ELECTROTHERMOFLEXOMAGNETOEXCITONICPOLARITONIC_RESONANCE_ENERGY/1
Engine Directive
{c1}
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 1 in model.eng_electrothermoflexomagnetoexcitonicpolaritonic_resonance_energies
    eng = model.eng_electrothermoflexomagnetoexcitonicpolaritonic_resonance_energies[1]
    assert pytest.approx(eng.dt_etfexmnp) == 0.00075
    assert pytest.approx(eng.dt_etfexp) == 0.00075
    assert eng.sens_id == 10


def test_m438_eng_electrothermoflexomagnetoexcitonicpolaritonic_resonance_energy_free(tmp_path: Path):
    deck = """# ENGINE DECK FREE
/BEGIN
Engine Electrothermoflexomagnetoexcitonicpolaritonic Resonance Energy Free Format Test
/ENG/ELECTROTHERMOFLEXOMAGNETOEXCITONICPOLARITONIC_RESONANCE_ENERGY/2
0.00085, 20
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 2 in model.eng_electrothermoflexomagnetoexcitonicpolaritonic_resonance_energies
    eng = model.eng_electrothermoflexomagnetoexcitonicpolaritonic_resonance_energies[2]
    assert pytest.approx(eng.dt_etfexmnp) == 0.00085
    assert pytest.approx(eng.dt_etfexp) == 0.00085
    assert eng.sens_id == 20


def test_m438_eng_electrothermoflexomagnetoexcitonicpolaritonic_resonance_energy_aliases(tmp_path: Path):
    deck = """# ENGINE DECK ALIASES
/BEGIN
Engine Resonance Energy Aliases Test
/ENG/ELECTRO_THERM_FLEXO_MAG_EXCITON_POLARITON_RES_WORK/3
0.00095, 30
/ENG/EELECTROTHERMOFLEXOMAGNETOEXCITONICPOLARITONICRESONANCE/4
0.00095, 30
/ENG/ELECTROTHERMOFLEXOMAGNETOEXCITONICPOLARITONIC_RESONANCE_DISSIPATION/5
0.00095, 30
/ENG/ET_ELECTROTHERMOFLEXOMAGNETOEXCITONICPOLARITONIC_RESONANCE/6
0.00095, 30
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    for eid in [3, 4, 5, 6]:
        assert eid in model.eng_electrothermoflexomagnetoexcitonicpolaritonic_resonance_energies
        eng = model.eng_electrothermoflexomagnetoexcitonicpolaritonic_resonance_energies[eid]
        assert pytest.approx(eng.dt_etfexmnp) == 0.00095
        assert pytest.approx(eng.dt_etfexp) == 0.00095
        assert eng.sens_id == 30


def test_m438_lagmul_symplectic_spinor_spatial_linkage_joint_fixed(tmp_path: Path):
    c1 = f"{101:>10d}{102:>10d}{103:>10d}{5e6:>20.4f}{5:>10d}{1e-5:>20.6f}"
    c2 = f"{35.5:>20.4f}{45.5:>20.4f}{62.5:>20.4f}{12.5:>20.4f}"
    deck = f"""# RADIOSS STARTER DECK
/BEGIN
Symplectic Spinor Spatial Linkage Joint Fixed Format Test
2022 0
/SYMPLECTIC_SPINOR_SPATIAL_LINKAGE_JOINT/1
Symplectic Spinor Spatial Linkage Joint Description
{c1}
{c2}
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 1 in model.lagmul_symplectic_spinor_spatial_linkage_joints
    j = model.lagmul_symplectic_spinor_spatial_linkage_joints[1]
    assert j.node1 == 101
    assert j.node2 == 102
    assert j.node3 == 103
    assert pytest.approx(j.stiff) == 5e6
    assert j.skew_id == 5
    assert pytest.approx(j.tol) == 1e-5
    assert pytest.approx(j.link_len_a) == 35.5
    assert pytest.approx(j.link_len_b) == 45.5
    assert pytest.approx(j.twist_angle_alpha) == 62.5
    assert pytest.approx(j.offset_distance_s) == 12.5
    assert pytest.approx(j.offset_distance_r) == 12.5
    assert pytest.approx(j.offset_distance_v) == 12.5
    assert pytest.approx(j.offset_distance_h) == 12.5
    assert pytest.approx(j.offset_distance_u) == 12.5
    assert pytest.approx(j.offset_distance_f) == 12.5


def test_m438_lagmul_symplectic_spinor_spatial_linkage_joint_free(tmp_path: Path):
    deck = """# RADIOSS FREE DECK
/BEGIN
Symplectic Spinor Spatial Linkage Joint Free Format Test
/SYMPLECTIC_SPINOR_SPATIAL_LINKAGE_JOINT/2
201, 202, 203, 6e6, 6, 2e-5
36.5, 46.5, 63.5, 13.5
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 2 in model.lagmul_symplectic_spinor_spatial_linkage_joints
    j = model.lagmul_symplectic_spinor_spatial_linkage_joints[2]
    assert j.node1 == 201
    assert j.node2 == 202
    assert j.node3 == 203
    assert pytest.approx(j.stiff) == 6e6
    assert j.skew_id == 6
    assert pytest.approx(j.tol) == 2e-5
    assert pytest.approx(j.link_len_a) == 36.5
    assert pytest.approx(j.link_len_b) == 46.5
    assert pytest.approx(j.twist_angle_alpha) == 63.5
    assert pytest.approx(j.offset_distance_s) == 13.5


def test_m438_lagmul_symplectic_spinor_spatial_linkage_joint_aliases(tmp_path: Path):
    deck = """# RADIOSS ALIAS DECK
/BEGIN
Symplectic Spinor Spatial Linkage Joint Aliases Test
/LAGMUL/SYMPLECTIC_SPINOR_SPATIAL_LINKAGE_JOINT/3
301, 302, 303, 4e6, 3, 1e-5
34.5, 44.5, 61.5, 11.5
/LAGMUL/SYMPLECTIC_SPINOR_SPATIAL_LINKAGE/4
301, 302, 303, 4e6, 3, 1e-5
34.5, 44.5, 61.5, 11.5
/SYMPLECTIC_SPINOR_SPATIAL_LINKAGE/5
301, 302, 303, 4e6, 3, 1e-5
34.5, 44.5, 61.5, 11.5
/SYMPLECTIC_SPINOR_SPATIAL_MULTI_LOOP_MECHANISM/6
301, 302, 303, 4e6, 3, 1e-5
34.5, 44.5, 61.5, 11.5
/SYMPLECTIC_SPINOR_SPATIAL_SYMMETRIC_MECHANISM/7
301, 302, 303, 4e6, 3, 1e-5
34.5, 44.5, 61.5, 11.5
/SYMPLECTIC_SPINOR_SPATIAL_6R_MECHANISM/8
301, 302, 303, 4e6, 3, 1e-5
34.5, 44.5, 61.5, 11.5
/SYMPLECTIC_SPINOR_SPATIAL_OVERCONSTRAINED_MECHANISM/9
301, 302, 303, 4e6, 3, 1e-5
34.5, 44.5, 61.5, 11.5
/LAGMUL/SYMPLECTIC_TWISTOR_SPATIAL_LINKAGE_JOINT/10
301, 302, 303, 4e6, 3, 1e-5
34.5, 44.5, 61.5, 11.5
/SYMPLECTIC_TWISTOR_SPATIAL_LINKAGE_JOINT/11
301, 302, 303, 4e6, 3, 1e-5
34.5, 44.5, 61.5, 11.5
/LAGMUL/SYMPLECTIC_SPINOR_BUNDLE_SPATIAL_LINKAGE_JOINT/12
301, 302, 303, 4e6, 3, 1e-5
34.5, 44.5, 61.5, 11.5
/SYMPLECTIC_SPINOR_BUNDLE_SPATIAL_LINKAGE_JOINT/13
301, 302, 303, 4e6, 3, 1e-5
34.5, 44.5, 61.5, 11.5
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    for jid in range(3, 14):
        assert jid in model.lagmul_symplectic_spinor_spatial_linkage_joints
        assert jid in model.lagmul_symplectic_twistor_spatial_linkage_joints
        assert jid in model.lagmul_symplectic_spinor_bundle_spatial_linkage_joints
        j = model.lagmul_symplectic_spinor_spatial_linkage_joints[jid]
        assert j.node1 == 301
        assert j.node2 == 302
        assert j.node3 == 303
        assert pytest.approx(j.stiff) == 4e6


def test_m438_lagmul_symplectic_spinor_spatial_linkage_joint_properties():
    from pyradioss.model.entities import LagmulSymplecticSpinorSpatialLinkageJoint
    j = LagmulSymplecticSpinorSpatialLinkageJoint(id=1)
    j.offset_distance_r = 18.5
    assert pytest.approx(j.offset_distance_s) == 18.5
    assert pytest.approx(j.offset_distance_r) == 18.5

    j.offset_distance_v = 19.5
    assert pytest.approx(j.offset_distance_s) == 19.5
    assert pytest.approx(j.offset_distance_v) == 19.5

    j.offset_distance_h = 20.5
    assert pytest.approx(j.offset_distance_s) == 20.5
    assert pytest.approx(j.offset_distance_h) == 20.5

    j.offset_distance_u = 21.5
    assert pytest.approx(j.offset_distance_s) == 21.5
    assert pytest.approx(j.offset_distance_u) == 21.5

    j.offset_distance_f = 22.5
    assert pytest.approx(j.offset_distance_s) == 22.5
    assert pytest.approx(j.offset_distance_f) == 22.5


def test_m438_sensor_spring_transverse_crackle_rate_fixed(tmp_path: Path):
    c1 = f"{501:>10d}{1.45e12:>20.6e}{0.0035:>20.6f}"
    deck = f"""# RADIOSS STARTER DECK
/BEGIN
Sensor Spring Transverse Crackle Rate Fixed Format Test
2022 0
/SENSOR/SPRING_TRANSVERSE_CRACKLE_RATE/1
Sensor Title
{c1}
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 1 in model.sensor_spring_transverse_crackle_rates
    s = model.sensor_spring_transverse_crackle_rates[1]
    assert s.spring_id == 501
    assert pytest.approx(s.jtrans_crk_max) == 1.45e12
    assert pytest.approx(s.jtrans_shot_max) == 1.45e12
    assert pytest.approx(s.jtrans_drop_max) == 1.45e12
    assert pytest.approx(s.jtrans_lock_max) == 1.45e12
    assert pytest.approx(s.jtrans_pop_max) == 1.45e12
    assert pytest.approx(s.t_delay) == 0.0035


def test_m438_sensor_spring_transverse_crackle_rate_free(tmp_path: Path):
    deck = """# RADIOSS FREE DECK
/BEGIN
Sensor Spring Transverse Crackle Rate Free Format Test
/SENSOR/SPRING_TRANSVERSE_CRACKLE_RATE/2
502, 1.55e12, 0.0045
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 2 in model.sensor_spring_transverse_crackle_rates
    s = model.sensor_spring_transverse_crackle_rates[2]
    assert s.spring_id == 502
    assert pytest.approx(s.jtrans_crk_max) == 1.55e12
    assert pytest.approx(s.t_delay) == 0.0045


def test_m438_sensor_spring_transverse_crackle_rate_aliases(tmp_path: Path):
    deck = """# RADIOSS ALIAS DECK
/BEGIN
Sensor Spring Transverse Crackle Rate Aliases Test
/SENSOR/SPRING_TRANSVERSE_CRK_RATE/3
503, 1.65e12, 0.0055
/SENSOR/SPRING_TRANS_CRACKLE_RATE/4
503, 1.65e12, 0.0055
/SENSOR/SPRING_TRANS_CRK_RATE/5
503, 1.65e12, 0.0055
/SENSOR/SPRING_CRACKLE_RATE_TRANS/6
503, 1.65e12, 0.0055
/SENSOR/SPRING_CRACKLE_RATE_TRANSVERSE/7
503, 1.65e12, 0.0055
/SENSOR/TRANSVERSE_CRACKLE_RATE_SPRING/8
503, 1.65e12, 0.0055
/SENSOR/SPRING_CRK_RATE_TRANS/9
503, 1.65e12, 0.0055
/SENSOR/SPRING_TRANSVERSE_CRACKLE/10
503, 1.65e12, 0.0055
/SENSOR/SPRING_CRACKLE_TRANS/11
503, 1.65e12, 0.0055
/SENSOR/SPRING_SHEAR_CRACKLE_RATE/12
503, 1.65e12, 0.0055
/SENSOR/SPRING_SHEAR_CRK_RATE/13
503, 1.65e12, 0.0055
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    for sid in range(3, 14):
        assert sid in model.sensor_spring_transverse_crackle_rates
        s = model.sensor_spring_transverse_crackle_rates[sid]
        assert s.spring_id == 503
        assert pytest.approx(s.jtrans_crk_max) == 1.65e12


def test_m438_sensor_spring_transverse_crackle_rate_properties():
    from pyradioss.model.entities import SensorSpringTransverseCrackleRate
    s = SensorSpringTransverseCrackleRate(id=1)
    s.jtrans_shot_max = 2.1e12
    assert pytest.approx(s.jtrans_crk_max) == 2.1e12
    assert pytest.approx(s.jtrans_shot_max) == 2.1e12

    s.jtrans_drop_max = 2.2e12
    assert pytest.approx(s.jtrans_crk_max) == 2.2e12
    assert pytest.approx(s.jtrans_drop_max) == 2.2e12

    s.jtrans_lock_max = 2.3e12
    assert pytest.approx(s.jtrans_crk_max) == 2.3e12
    assert pytest.approx(s.jtrans_lock_max) == 2.3e12

    s.jtrans_pop_max = 2.4e12
    assert pytest.approx(s.jtrans_crk_max) == 2.4e12
    assert pytest.approx(s.jtrans_pop_max) == 2.4e12


def test_m438_missing_cards_and_errors(tmp_path: Path):
    deck = """# EMPTY CARDS DECK
/BEGIN
Empty Cards Test
/FAIL/LAD_TRANSVERSE_CORE_MICROCRACKING_RATE/999
/ENG/ELECTROTHERMOFLEXOMAGNETOEXCITONICPOLARITONIC_RESONANCE_ENERGY/999
/SYMPLECTIC_SPINOR_SPATIAL_LINKAGE_JOINT/999
/SENSOR/SPRING_TRANSVERSE_CRACKLE_RATE/999
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) >= 4
