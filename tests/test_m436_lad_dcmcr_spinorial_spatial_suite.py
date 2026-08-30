"""Tests for Milestone M436: LadDynamicCoreMicrocrackingRate Failure Model, EngElectrothermoflexomagnetophononicpolaritonicResonanceEnergy, LagmulSpinorialSpatialLinkageJoint, and SensorSpringAxialCrackleRate."""

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


def test_m436_fail_lad_dynamic_core_microcracking_rate_fixed(tmp_path: Path):
    c1 = f"{965.0:>20.4f}{2895.0:>20.4f}{745.0:>20.4f}{15.25:>20.4f}{0.485:>20.4f}"
    c2 = f"{1:>10d}{2:>10d}"
    c3 = f"{2510:>10d}"
    deck = f"""# RADIOSS STARTER DECK
/BEGIN
Ladeveze Dynamic Core Microcracking Rate Fixed Format Test
2022 0
/FAIL/LAD_DYNAMIC_CORE_MICROCRACKING_RATE/2510
Ladeveze Dynamic Core Microcracking Failure Model
{c1}
{c2}
{c3}
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 2510 in model.fail_laddynamiccoremicrocrackingrates
    fdcmcr = model.fail_laddynamiccoremicrocrackingrates[2510]
    assert pytest.approx(fdcmcr.sigma_dcmcr0) == 965.0
    assert pytest.approx(fdcmcr.sigma_dcmcrc) == 2895.0
    assert pytest.approx(fdcmcr.gamma_dcmcr) == 745.0
    assert pytest.approx(fdcmcr.p_dcmcr) == 15.25
    assert pytest.approx(fdcmcr.d_dcmcr_max) == 0.485
    assert fdcmcr.ifail_sh == 1
    assert fdcmcr.ifail_so == 2
    assert fdcmcr.fail_id == 2510
    assert pytest.approx(fdcmcr.sigma_dcmd0) == 965.0
    assert pytest.approx(fdcmcr.sigma_dcmdc) == 2895.0
    assert pytest.approx(fdcmcr.gamma_dcmd) == 745.0
    assert pytest.approx(fdcmcr.p_dcmd) == 15.25
    assert pytest.approx(fdcmcr.d_dcmd_max) == 0.485
    assert len(model.raw_fails) == 1
    assert model.raw_fails[0][1].type == "LAD_DYNAMIC_CORE_MICROCRACKING_RATE"


def test_m436_fail_lad_dynamic_core_microcracking_rate_free(tmp_path: Path):
    deck = """# RADIOSS FREE DECK
/BEGIN
Ladeveze Dynamic Core Microcracking Rate Free Format Test
/FAIL/LAD_DYNAMIC_CORE_MICROCRACKING_RATE/2511
975.0, 2925.0, 755.0, 15.45, 0.475
1, 1
2511
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 2511 in model.fail_laddynamiccoremicrocrackingrates
    fdcmcr = model.fail_laddynamiccoremicrocrackingrates[2511]
    assert pytest.approx(fdcmcr.sigma_dcmcr0) == 975.0
    assert pytest.approx(fdcmcr.sigma_dcmcrc) == 2925.0
    assert pytest.approx(fdcmcr.gamma_dcmcr) == 755.0
    assert pytest.approx(fdcmcr.p_dcmcr) == 15.45
    assert pytest.approx(fdcmcr.d_dcmcr_max) == 0.475
    assert fdcmcr.fail_id == 2511


def test_m436_fail_lad_dynamic_core_microcracking_rate_aliases(tmp_path: Path):
    deck = """# RADIOSS ALIAS DECK
/BEGIN
Ladeveze Dynamic Core Microcracking Rate Aliases Test
/FAIL/LADEVEZE_DYNAMIC_CORE_MICROCRACKING_RATE/2512
955.0, 2865.0, 735.0, 15.05, 0.495
1, 1
/FAIL/LAD_DYNAMIC_CORE_MICROCRACK_RATE/2513
955.0, 2865.0, 735.0, 15.05, 0.495
1, 1
/FAIL/LAD_DCMCR/2514
955.0, 2865.0, 735.0, 15.05, 0.495
1, 1
/FAIL/LAD_DCMCR_MODEL/2515
955.0, 2865.0, 735.0, 15.05, 0.495
1, 1
/FAIL/LAD_DCMCR_LAW/2516
955.0, 2865.0, 735.0, 15.05, 0.495
1, 1
/FAIL/LADEVEZE_RATE_DEPENDENT_DYNAMIC_CORE_MICROCRACKING/2517
955.0, 2865.0, 735.0, 15.05, 0.495
1, 1
/FAIL/LAD_DYNAMIC_CORE_MICRODAMAGE_RATE/2518
955.0, 2865.0, 735.0, 15.05, 0.495
1, 1
/FAIL/LADEVEZE_DYNAMIC_CORE_MICRODAMAGE_RATE/2519
955.0, 2865.0, 735.0, 15.05, 0.495
1, 1
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 2512 in model.fail_laddynamiccoremicrocrackingrates
    assert 2513 in model.fail_laddynamiccoremicrocrackingrates
    assert 2514 in model.fail_laddynamiccoremicrocrackingrates
    assert 2515 in model.fail_laddynamiccoremicrocrackingrates
    assert 2516 in model.fail_laddynamiccoremicrocrackingrates
    assert 2517 in model.fail_laddynamiccoremicrocrackingrates
    assert 2518 in model.fail_laddynamiccoremicrocrackingrates
    assert 2519 in model.fail_laddynamiccoremicrocrackingrates
    assert len(model.raw_fails) == 8


def test_m436_fail_lad_dynamic_core_microcracking_rate_missing_card(tmp_path: Path):
    deck = """# RADIOSS MISSING CARD DECK
/BEGIN
/FAIL/LAD_DYNAMIC_CORE_MICROCRACKING_RATE/2520
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) > 0


def test_m436_eng_electrothermoflexomagnetophononicpolaritonic_resonance_energy_fixed(tmp_path: Path):
    c1 = f"{0.0885:>20.4f}{885:>10d}"
    deck = f"""# RADIOSS STARTER DECK
/BEGIN
Engine Electrothermoflexomagnetophononicpolaritonic Resonance Energy Fixed Format Test
2022 0
/ENG/ELECTROTHERMOFLEXOMAGNETOPHONONICPOLARITONIC_RESONANCE_ENERGY/785
Resonance Energy Output Directive
{c1}
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 785 in model.eng_electrothermoflexomagnetophononicpolaritonic_resonance_energies
    eng = model.eng_electrothermoflexomagnetophononicpolaritonic_resonance_energies[785]
    assert pytest.approx(eng.dt_etfphmnp) == 0.0885
    assert eng.sens_id == 885


def test_m436_eng_electrothermoflexomagnetophononicpolaritonic_resonance_energy_free(tmp_path: Path):
    deck = """# RADIOSS FREE DECK
/BEGIN
Engine Electrothermoflexomagnetophononicpolaritonic Resonance Energy Free Format Test
/ENG/ELECTROTHERMOFLEXOMAGNETOPHONONICPOLARITONIC_RESONANCE_ENERGY/786
0.0895, 886
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 786 in model.eng_electrothermoflexomagnetophononicpolaritonic_resonance_energies
    eng = model.eng_electrothermoflexomagnetophononicpolaritonic_resonance_energies[786]
    assert pytest.approx(eng.dt_etfphmnp) == 0.0895
    assert eng.sens_id == 886


def test_m436_eng_electrothermoflexomagnetophononicpolaritonic_resonance_energy_aliases(tmp_path: Path):
    deck = """# RADIOSS ALIAS DECK
/BEGIN
Engine Electrothermoflexomagnetophononicpolaritonic Resonance Energy Aliases Test
/ENG/ELECTRO_THERM_FLEXO_MAG_PHONON_POLARITON_RES_WORK/787
0.0905, 887
/ENG/EELECTROTHERMOFLEXOMAGNETOPHONONICPOLARITONICRESONANCE/788
0.0915, 888
/ENG/ELECTROTHERMOFLEXOMAGNETOPHONONICPOLARITONIC_RESONANCE_DISSIPATION/789
0.0925, 889
/ENG/ET_ELECTROTHERMOFLEXOMAGNETOPHONONICPOLARITONIC_RESONANCE/790
0.0935, 890
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 787 in model.eng_electrothermoflexomagnetophononicpolaritonic_resonance_energies
    assert 788 in model.eng_electrothermoflexomagnetophononicpolaritonic_resonance_energies
    assert 789 in model.eng_electrothermoflexomagnetophononicpolaritonic_resonance_energies
    assert 790 in model.eng_electrothermoflexomagnetophononicpolaritonic_resonance_energies


def test_m436_eng_electrothermoflexomagnetophononicpolaritonic_resonance_energy_missing_card(tmp_path: Path):
    deck = """# RADIOSS MISSING CARD DECK
/BEGIN
/ENG/ELECTROTHERMOFLEXOMAGNETOPHONONICPOLARITONIC_RESONANCE_ENERGY/791
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) > 0


def test_m436_lagmul_spinorial_spatial_linkage_joint_fixed(tmp_path: Path):
    c1 = f"{891:>10d}{892:>10d}{893:>10d}{7.45e7:>20.4f}{495:>10d}{6.65e-4:>20.4e}"
    c2 = f"{555.0:>20.4f}{515.0:>20.4f}{565.0:>20.4f}{485.0:>20.4f}"
    deck = f"""# RADIOSS STARTER DECK
/BEGIN
Spinorial Spatial Linkage Joint Fixed Format Test
2022 0
/LAGMUL/SPINORIAL_SPATIAL_LINKAGE_JOINT/835
Spinorial Spatial 6R Joint
{c1}
{c2}
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 835 in model.lagmul_spinorial_spatial_linkage_joints
    joint = model.lagmul_spinorial_spatial_linkage_joints[835]
    assert joint.node1 == 891
    assert joint.node2 == 892
    assert joint.node3 == 893
    assert pytest.approx(joint.stiff) == 7.45e7
    assert joint.skew_id == 495
    assert pytest.approx(joint.tol) == 6.65e-4
    assert pytest.approx(joint.link_len_a) == 555.0
    assert pytest.approx(joint.link_len_b) == 515.0
    assert pytest.approx(joint.twist_angle_alpha) == 565.0
    assert pytest.approx(joint.offset_distance_s) == 485.0
    assert pytest.approx(joint.offset_distance_r) == 485.0
    assert pytest.approx(joint.offset_distance_v) == 485.0
    assert pytest.approx(joint.offset_distance_h) == 485.0
    assert pytest.approx(joint.offset_distance_u) == 485.0
    assert pytest.approx(joint.offset_distance_f) == 485.0


def test_m436_lagmul_spinorial_spatial_linkage_joint_free(tmp_path: Path):
    deck = """# RADIOSS FREE DECK
/BEGIN
Spinorial Spatial Linkage Joint Free Format Test
/LAGMUL/SPINORIAL_SPATIAL_LINKAGE_JOINT/836
991, 992, 993, 7.65e7, 496, 6.75e-4
575.0, 525.0, 585.0, 495.0
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 836 in model.lagmul_spinorial_spatial_linkage_joints
    joint = model.lagmul_spinorial_spatial_linkage_joints[836]
    assert joint.node1 == 991
    assert joint.node2 == 992
    assert joint.node3 == 993
    assert pytest.approx(joint.stiff) == 7.65e7
    assert joint.skew_id == 496
    assert pytest.approx(joint.tol) == 6.75e-4
    assert pytest.approx(joint.link_len_a) == 575.0
    assert pytest.approx(joint.link_len_b) == 525.0
    assert pytest.approx(joint.twist_angle_alpha) == 585.0
    assert pytest.approx(joint.offset_distance_s) == 495.0


def test_m436_lagmul_spinorial_spatial_linkage_joint_aliases(tmp_path: Path):
    deck = """# RADIOSS ALIAS DECK
/BEGIN
Spinorial Spatial Linkage Joint Aliases Test
/SPINORIAL_SPATIAL_LINKAGE_JOINT/837
1091, 1092, 1093, 1.0e6, 0, 1.0e-6
515.0, 485.0, 515.0, 465.0
/LAGMUL/SPINORIAL_SPATIAL_LINKAGE/838
1091, 1092, 1093, 1.0e6, 0, 1.0e-6
515.0, 485.0, 515.0, 465.0
/SPINORIAL_SPATIAL_LINKAGE/839
1091, 1092, 1093, 1.0e6, 0, 1.0e-6
515.0, 485.0, 515.0, 465.0
/SPINORIAL_SPATIAL_MULTI_LOOP_MECHANISM/840
1091, 1092, 1093, 1.0e6, 0, 1.0e-6
515.0, 485.0, 515.0, 465.0
/SPINORIAL_SPATIAL_SYMMETRIC_MECHANISM/841
1091, 1092, 1093, 1.0e6, 0, 1.0e-6
515.0, 485.0, 515.0, 465.0
/SPINORIAL_SPATIAL_6R_MECHANISM/842
1091, 1092, 1093, 1.0e6, 0, 1.0e-6
515.0, 485.0, 515.0, 465.0
/SPINORIAL_SPATIAL_OVERCONSTRAINED_MECHANISM/843
1091, 1092, 1093, 1.0e6, 0, 1.0e-6
515.0, 485.0, 515.0, 465.0
/LAGMUL/TWISTOR_SPATIAL_LINKAGE_JOINT/844
1091, 1092, 1093, 1.0e6, 0, 1.0e-6
515.0, 485.0, 515.0, 465.0
/TWISTOR_SPATIAL_LINKAGE_JOINT/845
1091, 1092, 1093, 1.0e6, 0, 1.0e-6
515.0, 485.0, 515.0, 465.0
/LAGMUL/SPINOR_BUNDLE_SPATIAL_LINKAGE_JOINT/846
1091, 1092, 1093, 1.0e6, 0, 1.0e-6
515.0, 485.0, 515.0, 465.0
/SPINOR_BUNDLE_SPATIAL_LINKAGE_JOINT/847
1091, 1092, 1093, 1.0e6, 0, 1.0e-6
515.0, 485.0, 515.0, 465.0
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 837 in model.lagmul_spinorial_spatial_linkage_joints
    assert 838 in model.lagmul_spinorial_spatial_linkage_joints
    assert 839 in model.lagmul_spinorial_spatial_linkage_joints
    assert 840 in model.lagmul_spinorial_spatial_linkage_joints
    assert 841 in model.lagmul_spinorial_spatial_linkage_joints
    assert 842 in model.lagmul_spinorial_spatial_linkage_joints
    assert 843 in model.lagmul_spinorial_spatial_linkage_joints
    assert 844 in model.lagmul_spinorial_spatial_linkage_joints
    assert 845 in model.lagmul_spinorial_spatial_linkage_joints
    assert 846 in model.lagmul_spinorial_spatial_linkage_joints
    assert 847 in model.lagmul_spinorial_spatial_linkage_joints


def test_m436_lagmul_spinorial_spatial_linkage_joint_missing_card(tmp_path: Path):
    deck = """# RADIOSS MISSING CARD DECK
/BEGIN
/LAGMUL/SPINORIAL_SPATIAL_LINKAGE_JOINT/848
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) > 0


def test_m436_sensor_spring_axial_crackle_rate_fixed(tmp_path: Path):
    c1 = f"{1185:>10d}{9.35e6:>20.4f}{0.0065:>20.4f}"
    deck = f"""# RADIOSS STARTER DECK
/BEGIN
Sensor Spring Axial Crackle Rate Fixed Format Test
2022 0
/SENSOR/SPRING_AXIAL_CRACKLE_RATE/965
Spring Axial Crackle Rate Sensor
{c1}
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 965 in model.sensor_spring_axial_crackle_rates
    sensor = model.sensor_spring_axial_crackle_rates[965]
    assert sensor.spring_id == 1185
    assert pytest.approx(sensor.jax_crk_max) == 9.35e6
    assert pytest.approx(sensor.j_crk_ax_max) == 9.35e6
    assert pytest.approx(sensor.j_crackle_ax_max) == 9.35e6
    assert pytest.approx(sensor.j_ax_crk_max) == 9.35e6
    assert pytest.approx(sensor.j_crk_max) == 9.35e6
    assert pytest.approx(sensor.jax_crackle_max) == 9.35e6
    assert pytest.approx(sensor.j_crk_axial_max) == 9.35e6
    assert pytest.approx(sensor.j_crackle_axial_max) == 9.35e6
    assert pytest.approx(sensor.jnorm_crk_max) == 9.35e6
    assert pytest.approx(sensor.jnorm_shot_max) == 9.35e6
    assert pytest.approx(sensor.jnorm_drop_max) == 9.35e6
    assert pytest.approx(sensor.jnorm_lock_max) == 9.35e6
    assert pytest.approx(sensor.jnorm_pop_max) == 9.35e6
    assert pytest.approx(sensor.jnorm_snp_max) == 9.35e6
    assert pytest.approx(sensor.jax_shot_max) == 9.35e6
    assert pytest.approx(sensor.jax_drop_max) == 9.35e6
    assert pytest.approx(sensor.jax_lock_max) == 9.35e6
    assert pytest.approx(sensor.jax_pop_max) == 9.35e6
    assert pytest.approx(sensor.jax_snp_max) == 9.35e6
    assert pytest.approx(sensor.jax_snap_max) == 9.35e6
    assert pytest.approx(sensor.jax_rate_max) == 9.35e6
    assert pytest.approx(sensor.jax_roc_rate_max) == 9.35e6
    assert pytest.approx(sensor.jax_drop_rate_max) == 9.35e6
    assert pytest.approx(sensor.jax_crk_rate_max) == 9.35e6
    assert pytest.approx(sensor.t_delay) == 0.0065
    assert len(model.sensors) == 1
    assert model.sensors[0].kind == "SPRING_AXIAL_CRACKLE_RATE"


def test_m436_sensor_spring_axial_crackle_rate_free(tmp_path: Path):
    deck = """# RADIOSS FREE DECK
/BEGIN
Sensor Spring Axial Crackle Rate Free Format Test
/SENSOR/SPRING_AXIAL_CRACKLE_RATE/966
1186, 9.45e6, 0.0075
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 966 in model.sensor_spring_axial_crackle_rates
    sensor = model.sensor_spring_axial_crackle_rates[966]
    assert sensor.spring_id == 1186
    assert pytest.approx(sensor.jax_crk_max) == 9.45e6
    assert pytest.approx(sensor.t_delay) == 0.0075


def test_m436_sensor_spring_axial_crackle_rate_aliases(tmp_path: Path):
    deck = """# RADIOSS ALIAS DECK
/BEGIN
Sensor Spring Axial Crackle Rate Aliases Test
/SENSOR/SPRING_AXIAL_CRK_RATE/967
1187, 9.25e6, 0.0055
/SENSOR/SPRING_CRACKLE_RATE_AXIAL/968
1188, 9.25e6, 0.0055
/SENSOR/AXIAL_CRACKLE_RATE_SPRING/969
1189, 9.25e6, 0.0055
/SENSOR/SPRING_CRK_RATE_AXIAL/970
1190, 9.25e6, 0.0055
/SENSOR/SPRING_AXIAL_CRACKLE/971
1191, 9.25e6, 0.0055
/SENSOR/SPRING_CRACKLE_AXIAL/972
1192, 9.25e6, 0.0055
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 967 in model.sensor_spring_axial_crackle_rates
    assert 968 in model.sensor_spring_axial_crackle_rates
    assert 969 in model.sensor_spring_axial_crackle_rates
    assert 970 in model.sensor_spring_axial_crackle_rates
    assert 971 in model.sensor_spring_axial_crackle_rates
    assert 972 in model.sensor_spring_axial_crackle_rates
    assert len(model.sensors) == 6


def test_m436_sensor_spring_axial_crackle_rate_missing_card(tmp_path: Path):
    deck = """# RADIOSS MISSING CARD DECK
/BEGIN
/SENSOR/SPRING_AXIAL_CRACKLE_RATE/973
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) > 0
