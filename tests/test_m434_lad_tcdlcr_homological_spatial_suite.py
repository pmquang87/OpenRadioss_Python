"""Tests for Milestone M434: LadTransverseCoreDelaminationCrackingRate Failure Model, EngElectrothermoflexomagnetoplasmonicphononicpolaritonicResonanceEnergy, HomologicalSpatialLinkageJoint, and SensorSpringBendingSnapRate."""

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


def test_m434_fail_lad_transverse_core_delamination_cracking_rate_fixed(tmp_path: Path):
    c1 = f"{985.0:>20.4f}{2955.0:>20.4f}{765.0:>20.4f}{15.65:>20.4f}{0.465:>20.4f}"
    c2 = f"{1:>10d}{2:>10d}"
    c3 = f"{2480:>10d}"
    deck = f"""# RADIOSS STARTER DECK
/BEGIN
Ladeveze Transverse Core Delamination Cracking Rate Fixed Format Test
2022 0
/FAIL/LAD_TRANSVERSE_CORE_DELAMINATION_CRACKING_RATE/2480
Ladeveze Transverse Core Delamination Cracking Failure Model
{c1}
{c2}
{c3}
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 2480 in model.fail_ladtransversecoredelaminationcrackingrates
    ftcdlcr = model.fail_ladtransversecoredelaminationcrackingrates[2480]
    assert pytest.approx(ftcdlcr.sigma_tcdlcr0) == 985.0
    assert pytest.approx(ftcdlcr.sigma_tcdlcrc) == 2955.0
    assert pytest.approx(ftcdlcr.gamma_tcdlcr) == 765.0
    assert pytest.approx(ftcdlcr.p_tcdlcr) == 15.65
    assert pytest.approx(ftcdlcr.d_tcdlcr_max) == 0.465
    assert ftcdlcr.ifail_sh == 1
    assert ftcdlcr.ifail_so == 2
    assert ftcdlcr.fail_id == 2480
    assert len(model.raw_fails) == 1
    assert model.raw_fails[0][1].type == "LAD_TRANSVERSE_CORE_DELAMINATION_CRACKING_RATE"


def test_m434_fail_lad_transverse_core_delamination_cracking_rate_free(tmp_path: Path):
    deck = """# RADIOSS FREE DECK
/BEGIN
Ladeveze Transverse Core Delamination Cracking Rate Free Format Test
/FAIL/LAD_TRANSVERSE_CORE_DELAMINATION_CRACKING_RATE/2481
995.0, 2985.0, 775.0, 15.85, 0.455
1, 1
2481
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 2481 in model.fail_ladtransversecoredelaminationcrackingrates
    ftcdlcr = model.fail_ladtransversecoredelaminationcrackingrates[2481]
    assert pytest.approx(ftcdlcr.sigma_tcdlcr0) == 995.0
    assert pytest.approx(ftcdlcr.sigma_tcdlcrc) == 2985.0
    assert pytest.approx(ftcdlcr.gamma_tcdlcr) == 775.0
    assert pytest.approx(ftcdlcr.p_tcdlcr) == 15.85
    assert pytest.approx(ftcdlcr.d_tcdlcr_max) == 0.455
    assert ftcdlcr.fail_id == 2481


def test_m434_fail_lad_transverse_core_delamination_cracking_rate_aliases(tmp_path: Path):
    deck = """# RADIOSS ALIAS DECK
/BEGIN
Ladeveze Transverse Core Delamination Cracking Rate Aliases Test
/FAIL/LADEVEZE_TRANSVERSE_CORE_DELAMINATION_CRACKING_RATE/2482
975.0, 2925.0, 755.0, 15.45, 0.475
1, 1
/FAIL/LAD_TCDLCR/2483
975.0, 2925.0, 755.0, 15.45, 0.475
1, 1
/FAIL/LAD_TCDLCR_MODEL/2484
975.0, 2925.0, 755.0, 15.45, 0.475
1, 1
/FAIL/LAD_TCDLCR_LAW/2485
975.0, 2925.0, 755.0, 15.45, 0.475
1, 1
/FAIL/LADEVEZE_RATE_DEPENDENT_TRANSVERSE_CORE_DELAMINATION_CRACKING/2486
975.0, 2925.0, 755.0, 15.45, 0.475
1, 1
/FAIL/LAD_TRANSVERSE_CORE_TEARING_RATE/2487
975.0, 2925.0, 755.0, 15.45, 0.475
1, 1
/FAIL/LADEVEZE_TRANSVERSE_CORE_TEARING_RATE/2488
975.0, 2925.0, 755.0, 15.45, 0.475
1, 1
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 2482 in model.fail_ladtransversecoredelaminationcrackingrates
    assert 2483 in model.fail_ladtransversecoredelaminationcrackingrates
    assert 2484 in model.fail_ladtransversecoredelaminationcrackingrates
    assert 2485 in model.fail_ladtransversecoredelaminationcrackingrates
    assert 2486 in model.fail_ladtransversecoredelaminationcrackingrates
    assert 2487 in model.fail_ladtransversecoredelaminationcrackingrates
    assert 2488 in model.fail_ladtransversecoredelaminationcrackingrates
    assert len(model.raw_fails) == 7


def test_m434_fail_lad_transverse_core_delamination_cracking_rate_missing_card(tmp_path: Path):
    deck = """# RADIOSS MISSING CARD DECK
/BEGIN
/FAIL/LAD_TRANSVERSE_CORE_DELAMINATION_CRACKING_RATE/2489
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) > 0


def test_m434_eng_electrothermoflexomagnetoplasmonicphononicpolaritonic_resonance_energy_fixed(tmp_path: Path):
    c1 = f"{0.0935:>20.4f}{895:>10d}"
    deck = f"""# RADIOSS STARTER DECK
/BEGIN
Engine Electrothermoflexomagnetoplasmonicphononicpolaritonic Resonance Energy Fixed Format Test
2022 0
/ENG/ELECTROTHERMOFLEXOMAGNETOPLASMONICPHONONICPOLARITONIC_RESONANCE_ENERGY/795
Resonance Energy Output Directive
{c1}
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 795 in model.eng_electrothermoflexomagnetoplasmonicphononicpolaritonic_resonance_energies
    eng = model.eng_electrothermoflexomagnetoplasmonicphononicpolaritonic_resonance_energies[795]
    assert pytest.approx(eng.dt_etfplphnp) == 0.0935
    assert eng.sens_id == 895


def test_m434_eng_electrothermoflexomagnetoplasmonicphononicpolaritonic_resonance_energy_free(tmp_path: Path):
    deck = """# RADIOSS FREE DECK
/BEGIN
Engine Electrothermoflexomagnetoplasmonicphononicpolaritonic Resonance Energy Free Format Test
/ENG/ELECTROTHERMOFLEXOMAGNETOPLASMONICPHONONICPOLARITONIC_RESONANCE_ENERGY/796
0.0945, 896
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 796 in model.eng_electrothermoflexomagnetoplasmonicphononicpolaritonic_resonance_energies
    eng = model.eng_electrothermoflexomagnetoplasmonicphononicpolaritonic_resonance_energies[796]
    assert pytest.approx(eng.dt_etfplphnp) == 0.0945
    assert eng.sens_id == 896


def test_m434_eng_electrothermoflexomagnetoplasmonicphononicpolaritonic_resonance_energy_aliases(tmp_path: Path):
    deck = """# RADIOSS ALIAS DECK
/BEGIN
Engine Electrothermoflexomagnetoplasmonicphononicpolaritonic Resonance Energy Aliases Test
/ENG/ELECTRO_THERM_FLEXO_MAG_PLASMON_PHONON_POLARITON_RES_WORK/797
0.0955, 897
/ENG/EELECTROTHERMOFLEXOMAGNETOPLASMONICPHONONICPOLARITONICRESONANCE/798
0.0965, 898
/ENG/ELECTROTHERMOFLEXOMAGNETOPLASMONICPHONONICPOLARITONIC_RESONANCE_DISSIPATION/799
0.0975, 899
/ENG/ET_ELECTROTHERMOFLEXOMAGNETOPLASMONICPHONONICPOLARITONIC_RESONANCE/800
0.0985, 900
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 797 in model.eng_electrothermoflexomagnetoplasmonicphononicpolaritonic_resonance_energies
    assert 798 in model.eng_electrothermoflexomagnetoplasmonicphononicpolaritonic_resonance_energies
    assert 799 in model.eng_electrothermoflexomagnetoplasmonicphononicpolaritonic_resonance_energies
    assert 800 in model.eng_electrothermoflexomagnetoplasmonicphononicpolaritonic_resonance_energies


def test_m434_eng_electrothermoflexomagnetoplasmonicphononicpolaritonic_resonance_energy_missing_card(tmp_path: Path):
    deck = """# RADIOSS MISSING CARD DECK
/BEGIN
/ENG/ELECTROTHERMOFLEXOMAGNETOPLASMONICPHONONICPOLARITONIC_RESONANCE_ENERGY/801
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) > 0


def test_m434_lagmul_homological_spatial_linkage_joint_fixed(tmp_path: Path):
    c1 = f"{881:>10d}{882:>10d}{883:>10d}{7.65e7:>20.4f}{485:>10d}{6.75e-4:>20.4e}"
    c2 = f"{545.0:>20.4f}{525.0:>20.4f}{555.0:>20.4f}{475.0:>20.4f}"
    deck = f"""# RADIOSS STARTER DECK
/BEGIN
Homological Spatial Linkage Joint Fixed Format Test
2022 0
/LAGMUL/HOMOLOGICAL_SPATIAL_LINKAGE_JOINT/845
Homological Spatial 6R Joint
{c1}
{c2}
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 845 in model.lagmul_homological_spatial_linkage_joints
    joint = model.lagmul_homological_spatial_linkage_joints[845]
    assert joint.node1 == 881
    assert joint.node2 == 882
    assert joint.node3 == 883
    assert pytest.approx(joint.stiff) == 7.65e7
    assert joint.skew_id == 485
    assert pytest.approx(joint.tol) == 6.75e-4
    assert pytest.approx(joint.link_len_a) == 545.0
    assert pytest.approx(joint.link_len_b) == 525.0
    assert pytest.approx(joint.twist_angle_alpha) == 555.0
    assert pytest.approx(joint.offset_distance_s) == 475.0
    assert pytest.approx(joint.offset_distance_r) == 475.0
    assert pytest.approx(joint.offset_distance_v) == 475.0
    assert pytest.approx(joint.offset_distance_h) == 475.0
    assert pytest.approx(joint.offset_distance_u) == 475.0
    assert pytest.approx(joint.offset_distance_f) == 475.0


def test_m434_lagmul_homological_spatial_linkage_joint_free(tmp_path: Path):
    deck = """# RADIOSS FREE DECK
/BEGIN
Homological Spatial Linkage Joint Free Format Test
/LAGMUL/HOMOLOGICAL_SPATIAL_LINKAGE_JOINT/846
981, 982, 983, 7.85e7, 486, 6.85e-4
565.0, 535.0, 575.0, 490.0
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 846 in model.lagmul_homological_spatial_linkage_joints
    joint = model.lagmul_homological_spatial_linkage_joints[846]
    assert joint.node1 == 981
    assert joint.node2 == 982
    assert joint.node3 == 983
    assert pytest.approx(joint.stiff) == 7.85e7
    assert joint.skew_id == 486
    assert pytest.approx(joint.tol) == 6.85e-4
    assert pytest.approx(joint.link_len_a) == 565.0
    assert pytest.approx(joint.link_len_b) == 535.0
    assert pytest.approx(joint.twist_angle_alpha) == 575.0
    assert pytest.approx(joint.offset_distance_s) == 490.0


def test_m434_lagmul_homological_spatial_linkage_joint_aliases(tmp_path: Path):
    deck = """# RADIOSS ALIAS DECK
/BEGIN
Homological Spatial Linkage Joint Aliases Test
/HOMOLOGICAL_SPATIAL_LINKAGE_JOINT/847
1081, 1082, 1083, 1.0e6, 0, 1.0e-6
505.0, 495.0, 505.0, 455.0
/LAGMUL/HOMOLOGICAL_SPATIAL_LINKAGE/848
1081, 1082, 1083, 1.0e6, 0, 1.0e-6
505.0, 495.0, 505.0, 455.0
/HOMOLOGICAL_SPATIAL_LINKAGE/849
1081, 1082, 1083, 1.0e6, 0, 1.0e-6
505.0, 495.0, 505.0, 455.0
/HOMOLOGICAL_SPATIAL_MULTI_LOOP_MECHANISM/850
1081, 1082, 1083, 1.0e6, 0, 1.0e-6
505.0, 495.0, 505.0, 455.0
/HOMOLOGICAL_SPATIAL_SYMMETRIC_MECHANISM/851
1081, 1082, 1083, 1.0e6, 0, 1.0e-6
505.0, 495.0, 505.0, 455.0
/HOMOLOGICAL_SPATIAL_6R_MECHANISM/852
1081, 1082, 1083, 1.0e6, 0, 1.0e-6
505.0, 495.0, 505.0, 455.0
/HOMOLOGICAL_SPATIAL_OVERCONSTRAINED_MECHANISM/853
1081, 1082, 1083, 1.0e6, 0, 1.0e-6
505.0, 495.0, 505.0, 455.0
/LAGMUL/MANIFOLD_SPATIAL_LINKAGE_JOINT/854
1081, 1082, 1083, 1.0e6, 0, 1.0e-6
505.0, 495.0, 505.0, 455.0
/MANIFOLD_SPATIAL_LINKAGE_JOINT/855
1081, 1082, 1083, 1.0e6, 0, 1.0e-6
505.0, 495.0, 505.0, 455.0
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 847 in model.lagmul_homological_spatial_linkage_joints
    assert 848 in model.lagmul_homological_spatial_linkage_joints
    assert 849 in model.lagmul_homological_spatial_linkage_joints
    assert 850 in model.lagmul_homological_spatial_linkage_joints
    assert 851 in model.lagmul_homological_spatial_linkage_joints
    assert 852 in model.lagmul_homological_spatial_linkage_joints
    assert 853 in model.lagmul_homological_spatial_linkage_joints
    assert 854 in model.lagmul_homological_spatial_linkage_joints
    assert 855 in model.lagmul_homological_spatial_linkage_joints


def test_m434_lagmul_homological_spatial_linkage_joint_missing_card(tmp_path: Path):
    deck = """# RADIOSS MISSING CARD DECK
/BEGIN
/LAGMUL/HOMOLOGICAL_SPATIAL_LINKAGE_JOINT/856
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) > 0


def test_m434_sensor_spring_bending_snap_rate_fixed(tmp_path: Path):
    c1 = f"{1175:>10d}{9.15e6:>20.4f}{0.0085:>20.4f}"
    deck = f"""# RADIOSS STARTER DECK
/BEGIN
Sensor Spring Bending Snap Rate Fixed Format Test
2022 0
/SENSOR/SPRING_BENDING_SNAP_RATE/975
Spring Bending Snap Rate Sensor
{c1}
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 975 in model.sensor_spring_bending_snap_rates
    sensor = model.sensor_spring_bending_snap_rates[975]
    assert sensor.spring_id == 1175
    assert pytest.approx(sensor.jbnd_snp_max) == 9.15e6
    assert pytest.approx(sensor.jbnd_lock_max) == 9.15e6
    assert pytest.approx(sensor.jbnd_crackle_max) == 9.15e6
    assert pytest.approx(sensor.jbnd_shot_max) == 9.15e6
    assert pytest.approx(sensor.jbnd_pop_max) == 9.15e6
    assert pytest.approx(sensor.jbnd_crk_max) == 9.15e6
    assert pytest.approx(sensor.jbnd_drop_rate_max) == 9.15e6
    assert pytest.approx(sensor.jbnd_rate_max) == 9.15e6
    assert pytest.approx(sensor.jbnd_roc_rate_max) == 9.15e6
    assert pytest.approx(sensor.jbnd_snap_rate_max) == 9.15e6
    assert pytest.approx(sensor.j_max) == 9.15e6
    assert pytest.approx(sensor.t_delay) == 0.0085
    assert len(model.sensors) == 1
    assert model.sensors[0].kind == "SPRING_BENDING_SNAP_RATE"


def test_m434_sensor_spring_bending_snap_rate_free(tmp_path: Path):
    deck = """# RADIOSS FREE DECK
/BEGIN
Sensor Spring Bending Snap Rate Free Format Test
/SENSOR/SPRING_BENDING_SNAP_RATE/976
1176, 9.25e6, 0.0095
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 976 in model.sensor_spring_bending_snap_rates
    sensor = model.sensor_spring_bending_snap_rates[976]
    assert sensor.spring_id == 1176
    assert pytest.approx(sensor.jbnd_snp_max) == 9.25e6
    assert pytest.approx(sensor.t_delay) == 0.0095


def test_m434_sensor_spring_bending_snap_rate_aliases(tmp_path: Path):
    deck = """# RADIOSS ALIAS DECK
/BEGIN
Sensor Spring Bending Snap Rate Aliases Test
/SENSOR/SPRING_BEND_SNAP_RATE/977
1177, 9.05e6, 0.0075
/SENSOR/SPRING_SNAP_RATE_BEND/978
1178, 9.05e6, 0.0075
/SENSOR/BENDING_SNAP_RATE_SPRING/979
1179, 9.05e6, 0.0075
/SENSOR/SPRING_SNAP_BEND/980
1180, 9.05e6, 0.0075
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 977 in model.sensor_spring_bending_snap_rates
    assert 978 in model.sensor_spring_bending_snap_rates
    assert 979 in model.sensor_spring_bending_snap_rates
    assert 980 in model.sensor_spring_bending_snap_rates
    assert len(model.sensors) == 4


def test_m434_sensor_spring_bending_snap_rate_missing_card(tmp_path: Path):
    deck = """# RADIOSS MISSING CARD DECK
/BEGIN
/SENSOR/SPRING_BENDING_SNAP_RATE/981
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) > 0
