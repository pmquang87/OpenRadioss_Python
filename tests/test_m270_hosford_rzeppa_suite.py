# -*- coding: utf-8 -*-
"""Tests for Milestone M270:
- Hosford-Coulomb Failure Model: /FAIL/HOSFORD_COULOMB, /FAIL/HOSFORD_COULOMB_MODEL, /FAIL/HC_FRACTURE, /FAIL/HC_FAILURE, /FAIL/HOSFORD_COULOMB_LAW
- Engine Bulk Viscosity Output Directive: /ENG/BULK_VISCOSITY, /ENG/Q_VISC, /ENG/BULK_VISC, /ENG/QVISC, /ENG/VISCOSITY_BULK
- Rzeppa Joint Constraint: /LAGMUL/RZEPPA_JOINT, /RZEPPA_JOINT, /LAGMUL/RZEPPA_COUPLING, /RZEPPA_COUPLING, /RZEPPA_MECHANISM
- Spring Axial Energy Sensor: /SENSOR/SPRING_AXIAL_ENERGY, /SENSOR/SPRING_AX_ENERGY, /SENSOR/SPRING_AXIAL_DEFORM, /SENSOR/AXIAL_ENERGY_SPRING
"""

from pathlib import Path
import pytest
from pyradioss.common.messages import MessageLog
from pyradioss.input.deck_reader import read_deck
from pyradioss.input.starter_keywords import parse_starter_deck
from pyradioss.model.model import Model


def _parse_starter(tmp_path: Path, text: str) -> tuple[Model, MessageLog]:
    p = tmp_path / "TEST_0000.rad"
    p.write_text(text.strip() + "\n", encoding="ascii")
    blocks = read_deck(str(p))
    model = Model()
    log = MessageLog()
    parse_starter_deck(blocks, model, log)
    return model, log


# ============================================================================
# 1. /FAIL/HOSFORD_COULOMB Tests
# ============================================================================

def test_m270_fail_hosford_coulomb_fixed(tmp_path: Path):
    c1 = f"{2.0:>20.4f}{0.45:>20.4f}{120.0:>20.4f}{0.10:>20.4f}"
    c2 = f"{2:>10d}{1:>10d}{0.85:>20.4f}"
    deck = f"""# RADIOSS STARTER DECK
/BEGIN
Hosford-Coulomb Failure Model Fixed Format Test
2022 0
/FAIL/HOSFORD_COULOMB/401
Hosford-Coulomb Ductile Fracture Model
{c1}
{c2}
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 401 in model.fail_hosford_coulombs
    fhc = model.fail_hosford_coulombs[401]
    assert pytest.approx(fhc.a_hc) == 2.0
    assert pytest.approx(fhc.b_hc) == 0.45
    assert pytest.approx(fhc.c_hc) == 120.0
    assert pytest.approx(fhc.n_hc) == 0.10
    assert fhc.ifail_sh == 2
    assert fhc.ifail_so == 1
    assert pytest.approx(fhc.d_max) == 0.85


def test_m270_fail_hosford_coulomb_defaults(tmp_path: Path):
    deck = """# RADIOSS STARTER DECK
/BEGIN
Hosford-Coulomb Defaults Test
2022 0
/FAIL/HOSFORD_COULOMB/402
Hosford-Coulomb Default Values
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 1
    assert "missing data card" in log.errors[0]


def test_m270_fail_hosford_coulomb_aliases(tmp_path: Path):
    c1 = f"{1.5:>20.4f}{0.30:>20.4f}{80.0:>20.4f}{0.05:>20.4f}"
    deck = f"""# RADIOSS STARTER DECK
/BEGIN
Hosford-Coulomb Aliases Test
2022 0
/FAIL/HOSFORD_COULOMB_MODEL/450
Hosford Coulomb Model Alias
{c1}
/FAIL/HC_FRACTURE/451
HC Fracture Alias
{c1}
/FAIL/HC_FAILURE/452
HC Failure Alias
{c1}
/FAIL/HOSFORD_COULOMB_LAW/453
Hosford Coulomb Law Alias
{c1}
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 450 in model.fail_hosford_coulombs
    assert 451 in model.fail_hosford_coulombs
    assert 452 in model.fail_hosford_coulombs
    assert 453 in model.fail_hosford_coulombs
    assert pytest.approx(model.fail_hosford_coulombs[450].a_hc) == 1.5
    assert pytest.approx(model.fail_hosford_coulombs[451].a_hc) == 1.5
    assert pytest.approx(model.fail_hosford_coulombs[452].a_hc) == 1.5
    assert pytest.approx(model.fail_hosford_coulombs[453].a_hc) == 1.5


# ============================================================================
# 2. /ENG/BULK_VISCOSITY Tests
# ============================================================================

def test_m270_eng_bulk_viscosity(tmp_path: Path):
    deck = """# RADIOSS STARTER DECK
/BEGIN
Engine Bulk Viscosity Output Test
1 1
/ENG/BULK_VISCOSITY/1
Bulk Viscosity Energy Field Output
              0.0006        20
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 1 in model.eng_bulk_viscosities
    ebv = model.eng_bulk_viscosities[1]
    assert pytest.approx(ebv.dt_qvisc) == 0.0006
    assert ebv.sens_id == 20


def test_m270_eng_bulk_viscosity_aliases(tmp_path: Path):
    deck = """# RADIOSS STARTER DECK
/BEGIN
Engine Bulk Viscosity Aliases Test
1 1
/ENG/Q_VISC/2
Q Visc Alias
              0.0001         0
/ENG/BULK_VISC/3
Bulk Visc Alias
              0.0002         1
/ENG/QVISC/4
Qvisc Alias
              0.0003         3
/ENG/VISCOSITY_BULK/5
Viscosity Bulk Alias
              0.0004         5
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 2 in model.eng_bulk_viscosities
    assert 3 in model.eng_bulk_viscosities
    assert 4 in model.eng_bulk_viscosities
    assert 5 in model.eng_bulk_viscosities
    assert pytest.approx(model.eng_bulk_viscosities[2].dt_qvisc) == 0.0001
    assert pytest.approx(model.eng_bulk_viscosities[3].dt_qvisc) == 0.0002
    assert pytest.approx(model.eng_bulk_viscosities[4].dt_qvisc) == 0.0003
    assert pytest.approx(model.eng_bulk_viscosities[5].dt_qvisc) == 0.0004


# ============================================================================
# 3. /LAGMUL/RZEPPA_JOINT Tests
# ============================================================================

def test_m270_rzeppa_joint(tmp_path: Path):
    deck = """# RADIOSS STARTER DECK
/BEGIN
Rzeppa Joint Test
1 1
/LAGMUL/RZEPPA_JOINT/950
Rzeppa Constant Velocity Ball Joint
        25        35        45              7.0e+6         8              3.0e-6
                 1.0                 0.0                 0.0
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 950 in model.lagmul_rzeppa_joints
    rj = model.lagmul_rzeppa_joints[950]
    assert rj.node1 == 25
    assert rj.node2 == 35
    assert rj.node3 == 45
    assert pytest.approx(rj.stiff) == 7.0e+6
    assert rj.skew_id == 8
    assert pytest.approx(rj.tol) == 3.0e-6
    assert pytest.approx(rj.axis_x) == 1.0
    assert pytest.approx(rj.axis_y) == 0.0
    assert pytest.approx(rj.axis_z) == 0.0


def test_m270_rzeppa_joint_aliases(tmp_path: Path):
    deck = """# RADIOSS STARTER DECK
/BEGIN
Rzeppa Joint Aliases Test
1 1
/RZEPPA_JOINT/951
Rzeppa Joint Alias 1
         1         2         3              1.0e+6         0              1.0e-6
/LAGMUL/RZEPPA_COUPLING/952
Rzeppa Coupling Alias 2
         4         5         6              2.0e+6         0              1.0e-6
/RZEPPA_COUPLING/953
Rzeppa Coupling Alias 3
         7         8         9              3.0e+6         0              1.0e-6
/RZEPPA_MECHANISM/954
Rzeppa Mechanism Alias 4
        10        11        12              4.0e+6         0              1.0e-6
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 951 in model.lagmul_rzeppa_joints
    assert 952 in model.lagmul_rzeppa_joints
    assert 953 in model.lagmul_rzeppa_joints
    assert 954 in model.lagmul_rzeppa_joints
    assert model.lagmul_rzeppa_joints[951].node3 == 3
    assert model.lagmul_rzeppa_joints[952].node3 == 6
    assert model.lagmul_rzeppa_joints[953].node3 == 9
    assert model.lagmul_rzeppa_joints[954].node3 == 12


# ============================================================================
# 4. /SENSOR/SPRING_AXIAL_ENERGY Tests
# ============================================================================

def test_m270_sensor_spring_axial_energy(tmp_path: Path):
    deck = """# RADIOSS STARTER DECK
/BEGIN
Spring Axial Energy Sensor Test
1 1
/SENSOR/SPRING_AXIAL_ENERGY/1300
Spring Axial Energy Sensor 1
       950              1.5e+5               0.0020
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 1300 in model.sensor_spring_axial_energies
    sens = model.sensor_spring_axial_energies[1300]
    assert sens.id == 1300
    assert sens.spring_id == 950
    assert pytest.approx(sens.u_axial_max) == 1.5e+5
    assert pytest.approx(sens.t_delay) == 0.0020

    # Global sensor list check
    matching = [s for s in model.sensors if s.id == 1300]
    assert len(matching) == 1
    assert matching[0].kind == "SPRING_AXIAL_ENERGY"
    assert pytest.approx(matching[0].tdelay) == 0.0020


def test_m270_sensor_spring_axial_energy_aliases(tmp_path: Path):
    deck = """# RADIOSS STARTER DECK
/BEGIN
Spring Axial Energy Sensor Aliases Test
1 1
/SENSOR/SPRING_AX_ENERGY/1400
Spring Ax Energy Alias 1
       801              4.0e+4               0.001
/SENSOR/SPRING_AXIAL_DEFORM/1401
Spring Axial Deform Alias 2
       802              5.0e+4               0.002
/SENSOR/AXIAL_ENERGY_SPRING/1402
Axial Energy Spring Alias 3
       803              6.0e+4               0.003
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 1400 in model.sensor_spring_axial_energies
    assert 1401 in model.sensor_spring_axial_energies
    assert 1402 in model.sensor_spring_axial_energies
    assert pytest.approx(model.sensor_spring_axial_energies[1400].u_axial_max) == 4.0e+4
    assert pytest.approx(model.sensor_spring_axial_energies[1401].u_axial_max) == 5.0e+4
    assert pytest.approx(model.sensor_spring_axial_energies[1402].u_axial_max) == 6.0e+4
