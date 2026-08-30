"""Tests for Milestone M430: LadDynamicCoreDelaminationRate Failure Model, EngElectrothermoflexomagnetoexcitonicmagnonicpolaritonicResonanceEnergy, SymplecticSpatialLinkageJoint, and SensorSpringAxialSnapRate."""

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


def test_m430_fail_lad_dynamic_core_delamination_rate_fixed(tmp_path: Path):
    c1 = f"{935.0:>20.4f}{2805.0:>20.4f}{715.0:>20.4f}{14.65:>20.4f}{0.515:>20.4f}"
    c2 = f"{1:>10d}{2:>10d}"
    c3 = f"{2440:>10d}"
    deck = f"""# RADIOSS STARTER DECK
/BEGIN
Ladeveze Dynamic Core Delamination Rate Fixed Format Test
2022 0
/FAIL/LAD_DYNAMIC_CORE_DELAMINATION_RATE/2440
Ladeveze Dynamic Core Delamination Failure Model
{c1}
{c2}
{c3}
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 2440 in model.fail_laddynamiccoredelaminationrates
    fdcdl = model.fail_laddynamiccoredelaminationrates[2440]
    assert pytest.approx(fdcdl.sigma_dcdl0) == 935.0
    assert pytest.approx(fdcdl.sigma_dcdlc) == 2805.0
    assert pytest.approx(fdcdl.gamma_dcdl) == 715.0
    assert pytest.approx(fdcdl.p_dcdl) == 14.65
    assert pytest.approx(fdcdl.d_dcdl_max) == 0.515
    assert fdcdl.ifail_sh == 1
    assert fdcdl.ifail_so == 2
    assert fdcdl.fail_id == 2440
    assert len(model.raw_fails) == 1
    assert model.raw_fails[0][1].type == "LAD_DYNAMIC_CORE_DELAMINATION_RATE"


def test_m430_fail_lad_dynamic_core_delamination_rate_free(tmp_path: Path):
    deck = """# RADIOSS FREE DECK
/BEGIN
Ladeveze Dynamic Core Delamination Rate Free Format Test
/FAIL/LAD_DYNAMIC_CORE_DELAMINATION_RATE/2441
945.0, 2835.0, 725.0, 14.85, 0.505
1, 1
2441
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 2441 in model.fail_laddynamiccoredelaminationrates
    fdcdl = model.fail_laddynamiccoredelaminationrates[2441]
    assert pytest.approx(fdcdl.sigma_dcdl0) == 945.0
    assert pytest.approx(fdcdl.sigma_dcdlc) == 2835.0
    assert pytest.approx(fdcdl.gamma_dcdl) == 725.0
    assert pytest.approx(fdcdl.p_dcdl) == 14.85
    assert pytest.approx(fdcdl.d_dcdl_max) == 0.505
    assert fdcdl.fail_id == 2441


def test_m430_fail_lad_dynamic_core_delamination_rate_aliases(tmp_path: Path):
    deck = """# RADIOSS ALIAS DECK
/BEGIN
Ladeveze Dynamic Core Delamination Rate Aliases Test
/FAIL/LADEVEZE_DYNAMIC_CORE_DELAMINATION_RATE/2442
925.0, 2775.0, 705.0, 14.45, 0.525
1, 1
/FAIL/LAD_DCDLR/2443
925.0, 2775.0, 705.0, 14.45, 0.525
1, 1
/FAIL/LAD_DCDLR_MODEL/2444
925.0, 2775.0, 705.0, 14.45, 0.525
1, 1
/FAIL/LAD_DCDLR_LAW/2445
925.0, 2775.0, 705.0, 14.45, 0.525
1, 1
/FAIL/LADEVEZE_RATE_DEPENDENT_DYNAMIC_CORE_DELAMINATION/2446
925.0, 2775.0, 705.0, 14.45, 0.525
1, 1
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 2442 in model.fail_laddynamiccoredelaminationrates
    assert 2443 in model.fail_laddynamiccoredelaminationrates
    assert 2444 in model.fail_laddynamiccoredelaminationrates
    assert 2445 in model.fail_laddynamiccoredelaminationrates
    assert 2446 in model.fail_laddynamiccoredelaminationrates
    assert len(model.raw_fails) == 5


def test_m430_fail_lad_dynamic_core_delamination_rate_missing_card(tmp_path: Path):
    deck = """# RADIOSS MISSING CARD DECK
/BEGIN
/FAIL/LAD_DYNAMIC_CORE_DELAMINATION_RATE/2447
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) > 0


def test_m430_eng_electrothermoflexomagnetoexcitonicmagnonicpolaritonic_resonance_energy_fixed(tmp_path: Path):
    c1 = f"{0.0845:>20.4f}{855:>10d}"
    deck = f"""# RADIOSS STARTER DECK
/BEGIN
Engine Electrothermoflexomagnetoexcitonicmagnonicpolaritonic Resonance Energy Fixed Format Test
2022 0
/ENG/ELECTROTHERMOFLEXOMAGNETOEXCITONICMAGNONICPOLARITONIC_RESONANCE_ENERGY/755
Resonance Energy Output Directive
{c1}
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 755 in model.eng_electrothermoflexomagnetoexcitonicmagnonicpolaritonic_resonance_energies
    eng = model.eng_electrothermoflexomagnetoexcitonicmagnonicpolaritonic_resonance_energies[755]
    assert pytest.approx(eng.dt_etfmexmnp) == 0.0845
    assert eng.sens_id == 855


def test_m430_eng_electrothermoflexomagnetoexcitonicmagnonicpolaritonic_resonance_energy_free(tmp_path: Path):
    deck = """# RADIOSS FREE DECK
/BEGIN
Engine Electrothermoflexomagnetoexcitonicmagnonicpolaritonic Resonance Energy Free Format Test
/ENG/ELECTROTHERMOFLEXOMAGNETOEXCITONICMAGNONICPOLARITONIC_RESONANCE_ENERGY/756
0.0855, 856
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 756 in model.eng_electrothermoflexomagnetoexcitonicmagnonicpolaritonic_resonance_energies
    eng = model.eng_electrothermoflexomagnetoexcitonicmagnonicpolaritonic_resonance_energies[756]
    assert pytest.approx(eng.dt_etfmexmnp) == 0.0855
    assert eng.sens_id == 856


def test_m430_eng_electrothermoflexomagnetoexcitonicmagnonicpolaritonic_resonance_energy_aliases(tmp_path: Path):
    deck = """# RADIOSS ALIAS DECK
/BEGIN
Engine Electrothermoflexomagnetoexcitonicmagnonicpolaritonic Resonance Energy Aliases Test
/ENG/ELECTRO_THERM_FLEXO_MAG_EXCITON_MAGNONIC_POLARITON_RES_WORK/757
0.0865, 857
/ENG/EELECTROTHERMOFLEXOMAGNETOEXCITONICMAGNONICPOLARITONICRESONANCE/758
0.0875, 858
/ENG/ELECTROTHERMOFLEXOMAGNETOEXCITONICMAGNONICPOLARITONIC_RESONANCE_DISSIPATION/759
0.0885, 859
/ENG/ET_ELECTROTHERMOFLEXOMAGNETOEXCITONICMAGNONICPOLARITONIC_RESONANCE/760
0.0895, 860
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 757 in model.eng_electrothermoflexomagnetoexcitonicmagnonicpolaritonic_resonance_energies
    assert 758 in model.eng_electrothermoflexomagnetoexcitonicmagnonicpolaritonic_resonance_energies
    assert 759 in model.eng_electrothermoflexomagnetoexcitonicmagnonicpolaritonic_resonance_energies
    assert 760 in model.eng_electrothermoflexomagnetoexcitonicmagnonicpolaritonic_resonance_energies


def test_m430_eng_electrothermoflexomagnetoexcitonicmagnonicpolaritonic_resonance_energy_missing_card(tmp_path: Path):
    deck = """# RADIOSS MISSING CARD DECK
/BEGIN
/ENG/ELECTROTHERMOFLEXOMAGNETOEXCITONICMAGNONICPOLARITONIC_RESONANCE_ENERGY/761
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) > 0


def test_m430_lagmul_symplectic_spatial_linkage_joint_fixed(tmp_path: Path):
    c1 = f"{841:>10d}{842:>10d}{843:>10d}{7.25e7:>20.4f}{445:>10d}{6.35e-4:>20.4e}"
    c2 = f"{505.0:>20.4f}{485.0:>20.4f}{515.0:>20.4f}{435.0:>20.4f}"
    deck = f"""# RADIOSS STARTER DECK
/BEGIN
Symplectic Spatial Linkage Joint Fixed Format Test
2022 0
/LAGMUL/SYMPLECTIC_SPATIAL_LINKAGE_JOINT/805
Symplectic Spatial 6R Joint
{c1}
{c2}
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 805 in model.lagmul_symplectic_spatial_linkage_joints
    joint = model.lagmul_symplectic_spatial_linkage_joints[805]
    assert joint.node1 == 841
    assert joint.node2 == 842
    assert joint.node3 == 843
    assert pytest.approx(joint.stiff) == 7.25e7
    assert joint.skew_id == 445
    assert pytest.approx(joint.tol) == 6.35e-4
    assert pytest.approx(joint.link_len_a) == 505.0
    assert pytest.approx(joint.link_len_b) == 485.0
    assert pytest.approx(joint.twist_angle_alpha) == 515.0
    assert pytest.approx(joint.offset_distance_s) == 435.0
    assert pytest.approx(joint.offset_distance_r) == 435.0
    assert pytest.approx(joint.offset_distance_v) == 435.0
    assert pytest.approx(joint.offset_distance_h) == 435.0
    assert pytest.approx(joint.offset_distance_u) == 435.0
    assert pytest.approx(joint.offset_distance_f) == 435.0


def test_m430_lagmul_symplectic_spatial_linkage_joint_free(tmp_path: Path):
    deck = """# RADIOSS FREE DECK
/BEGIN
Symplectic Spatial Linkage Joint Free Format Test
/LAGMUL/SYMPLECTIC_SPATIAL_LINKAGE_JOINT/806
941, 942, 943, 7.45e7, 446, 6.45e-4
525.0, 495.0, 535.0, 450.0
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 806 in model.lagmul_symplectic_spatial_linkage_joints
    joint = model.lagmul_symplectic_spatial_linkage_joints[806]
    assert joint.node1 == 941
    assert joint.node2 == 942
    assert joint.node3 == 943
    assert pytest.approx(joint.stiff) == 7.45e7
    assert joint.skew_id == 446
    assert pytest.approx(joint.tol) == 6.45e-4
    assert pytest.approx(joint.link_len_a) == 525.0
    assert pytest.approx(joint.link_len_b) == 495.0
    assert pytest.approx(joint.twist_angle_alpha) == 535.0
    assert pytest.approx(joint.offset_distance_s) == 450.0


def test_m430_lagmul_symplectic_spatial_linkage_joint_aliases(tmp_path: Path):
    deck = """# RADIOSS ALIAS DECK
/BEGIN
Symplectic Spatial Linkage Joint Aliases Test
/SYMPLECTIC_SPATIAL_LINKAGE_JOINT/807
1041, 1042, 1043, 1.0e6, 0, 1.0e-6
465.0, 455.0, 465.0, 415.0
/LAGMUL/SYMPLECTIC_SPATIAL_LINKAGE/808
1041, 1042, 1043, 1.0e6, 0, 1.0e-6
465.0, 455.0, 465.0, 415.0
/SYMPLECTIC_SPATIAL_LINKAGE/809
1041, 1042, 1043, 1.0e6, 0, 1.0e-6
465.0, 455.0, 465.0, 415.0
/SYMPLECTIC_SPATIAL_MULTI_LOOP_MECHANISM/810
1041, 1042, 1043, 1.0e6, 0, 1.0e-6
465.0, 455.0, 465.0, 415.0
/SYMPLECTIC_SPATIAL_SYMMETRIC_MECHANISM/811
1041, 1042, 1043, 1.0e6, 0, 1.0e-6
465.0, 455.0, 465.0, 415.0
/SYMPLECTIC_SPATIAL_6R_MECHANISM/812
1041, 1042, 1043, 1.0e6, 0, 1.0e-6
465.0, 455.0, 465.0, 415.0
/SYMPLECTIC_SPATIAL_OVERCONSTRAINED_MECHANISM/813
1041, 1042, 1043, 1.0e6, 0, 1.0e-6
465.0, 455.0, 465.0, 415.0
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    for jid in range(807, 814):
        assert jid in model.lagmul_symplectic_spatial_linkage_joints


def test_m430_lagmul_symplectic_spatial_linkage_joint_missing_card(tmp_path: Path):
    deck = """# RADIOSS MISSING CARD DECK
/BEGIN
/LAGMUL/SYMPLECTIC_SPATIAL_LINKAGE_JOINT/814
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) > 0


def test_m430_sensor_spring_axial_snap_rate_fixed(tmp_path: Path):
    c1 = f"{1888:>10d}{8.48e8:>20.4f}{0.3945:>20.4f}"
    deck = f"""# RADIOSS STARTER DECK
/BEGIN
Sensor Spring Axial Snap Rate Fixed Format Test
2022 0
/SENSOR/SPRING_AXIAL_SNAP_RATE/1
Fixed Spring Axial Snap Rate Sensor
{c1}
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 1 in model.sensor_spring_axial_snap_rates
    s1 = model.sensor_spring_axial_snap_rates[1]
    assert s1.spring_id == 1888
    assert pytest.approx(s1.jax_snp_max) == 8.48e8
    assert pytest.approx(s1.jax_lock_max) == 8.48e8
    assert pytest.approx(s1.jax_crackle_max) == 8.48e8
    assert pytest.approx(s1.jax_shot_max) == 8.48e8
    assert pytest.approx(s1.jax_pop_max) == 8.48e8
    assert pytest.approx(s1.jax_crk_max) == 8.48e8
    assert pytest.approx(s1.jax_drop_rate_max) == 8.48e8
    assert pytest.approx(s1.jax_rate_max) == 8.48e8
    assert pytest.approx(s1.jax_roc_rate_max) == 8.48e8
    assert pytest.approx(s1.jax_snap_rate_max) == 8.48e8
    assert pytest.approx(s1.t_delay) == 0.3945
    assert len(model.sensors) == 1
    assert model.sensors[0].kind == "SPRING_AXIAL_SNAP_RATE"


def test_m430_sensor_spring_axial_snap_rate_free(tmp_path: Path):
    deck = """# RADIOSS FREE DECK
/BEGIN
Sensor Spring Axial Snap Rate Free Format Test
/SENSOR/SPRING_AXIAL_SNAP_RATE/2
Free Spring Axial Snap Rate Sensor
1889, 9.08e8, 0.4935
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 2 in model.sensor_spring_axial_snap_rates
    s2 = model.sensor_spring_axial_snap_rates[2]
    assert s2.spring_id == 1889
    assert pytest.approx(s2.jax_snp_max) == 9.08e8
    assert pytest.approx(s2.t_delay) == 0.4935


def test_m430_sensor_spring_axial_snap_rate_aliases(tmp_path: Path):
    deck = """# RADIOSS ALIAS DECK
/BEGIN
Sensor Spring Axial Snap Rate Aliases Test
/SENSOR/SPRING_AXIAL_SNAP/751
1909, 5.595e8, 0.4165
/SENSOR/SPRING_SNAP_RATE_AXIAL/752
1910, 5.615e8, 0.4175
/SENSOR/AXIAL_SNAP_RATE_SPRING/753
1911, 5.635e8, 0.4185
/SENSOR/SPRING_SNP_RATE_AXIAL/754
1912, 5.655e8, 0.4195
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    for sid in range(751, 755):
        assert sid in model.sensor_spring_axial_snap_rates


def test_m430_sensor_spring_axial_snap_rate_missing_card(tmp_path: Path):
    deck = """# RADIOSS MISSING CARD DECK
/BEGIN
/SENSOR/SPRING_AXIAL_SNAP_RATE/755
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) > 0
