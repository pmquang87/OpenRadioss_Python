# -*- coding: utf-8 -*-
"""Tests for Milestone M276:
- Freudenthal Critical Plastic Work Failure Model: /FAIL/FREUDENTHAL and aliases
- Engine Surface Boundary Energy Output Directive: /ENG/SURF_ENERGY and aliases
- Weiss CV Ball-and-Groove Joint Constraint: /LAGMUL/WEISS_JOINT and aliases
- Spring Friction Energy Sensor: /SENSOR/SPRING_FRICTION_ENERGY and aliases
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
# 1. /FAIL/FREUDENTHAL Tests
# ============================================================================

def test_m276_fail_freudenthal_fixed(tmp_path: Path):
    c1 = f"{450.0:>20.4f}{950.0:>20.4f}{0.015:>20.4f}"
    c2 = f"{2:>10d}{1:>10d}{0.94:>20.4f}"
    deck = f"""# RADIOSS STARTER DECK
/BEGIN
Freudenthal Critical Plastic Work Fixed Format Test
2022 0
/FAIL/FREUDENTHAL/820
Freudenthal Failure Model
{c1}
{c2}
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 820 in model.fail_freudenthals
    ff = model.fail_freudenthals[820]
    assert pytest.approx(ff.w_crit) == 450.0
    assert pytest.approx(ff.sigma_cut) == 950.0
    assert pytest.approx(ff.eps_p_min) == 0.015
    assert ff.ifail_sh == 2
    assert ff.ifail_so == 1
    assert pytest.approx(ff.d_max) == 0.94


def test_m276_fail_freudenthal_defaults(tmp_path: Path):
    deck = """# RADIOSS STARTER DECK
/BEGIN
Freudenthal Defaults Test
2022 0
/FAIL/FREUDENTHAL/821
Freudenthal Default Values
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 1
    assert "missing data card" in log.errors[0]


def test_m276_fail_freudenthal_aliases(tmp_path: Path):
    c1 = f"{350.0:>20.4f}{750.0:>20.4f}{0.005:>20.4f}"
    deck = f"""# RADIOSS STARTER DECK
/BEGIN
Freudenthal Aliases Test
2022 0
/FAIL/FREUDENTHAL_MODEL/870
Freudenthal Model Alias
{c1}
/FAIL/FREUDENTHAL_WORK/871
Freudenthal Work Alias
{c1}
/FAIL/FREUDENTHAL_FRACTURE/872
Freudenthal Fracture Alias
{c1}
/FAIL/FREUDENTHAL_LAW/873
Freudenthal Law Alias
{c1}
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 870 in model.fail_freudenthals
    assert 871 in model.fail_freudenthals
    assert 872 in model.fail_freudenthals
    assert 873 in model.fail_freudenthals
    assert pytest.approx(model.fail_freudenthals[870].w_crit) == 350.0
    assert pytest.approx(model.fail_freudenthals[871].w_crit) == 350.0
    assert pytest.approx(model.fail_freudenthals[872].w_crit) == 350.0
    assert pytest.approx(model.fail_freudenthals[873].w_crit) == 350.0


# ============================================================================
# 2. /ENG/SURF_ENERGY Tests
# ============================================================================

def test_m276_eng_surf_energy(tmp_path: Path):
    deck = """# RADIOSS STARTER DECK
/BEGIN
Engine Surface Energy Output Test
1 1
/ENG/SURF_ENERGY/1
Surface Boundary Energy Field Output
              0.0018        50
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 1 in model.eng_surf_energies
    ese = model.eng_surf_energies[1]
    assert pytest.approx(ese.dt_surf) == 0.0018
    assert ese.sens_id == 50


def test_m276_eng_surf_energy_aliases(tmp_path: Path):
    deck = """# RADIOSS STARTER DECK
/BEGIN
Engine Surface Energy Aliases Test
1 1
/ENG/SURF_WORK/2
Surf Work Alias
              0.0001         0
/ENG/ESURF/3
Esurf Alias
              0.0002         1
/ENG/SURFACE_ENERGY/4
Surface Energy Alias
              0.0003         3
/ENG/SURFACE_WORK/5
Surface Work Alias
              0.0004         5
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 2 in model.eng_surf_energies
    assert 3 in model.eng_surf_energies
    assert 4 in model.eng_surf_energies
    assert 5 in model.eng_surf_energies
    assert pytest.approx(model.eng_surf_energies[2].dt_surf) == 0.0001
    assert pytest.approx(model.eng_surf_energies[3].dt_surf) == 0.0002
    assert pytest.approx(model.eng_surf_energies[4].dt_surf) == 0.0003
    assert pytest.approx(model.eng_surf_energies[5].dt_surf) == 0.0004


# ============================================================================
# 3. /LAGMUL/WEISS_JOINT Tests
# ============================================================================

def test_m276_weiss_joint(tmp_path: Path):
    deck = """# RADIOSS STARTER DECK
/BEGIN
Weiss Joint Test
1 1
/LAGMUL/WEISS_JOINT/998
Weiss CV Joint Coupling
        72        82        92              9.5e+6        11              1.2e-6
                 0.0                 1.0                 0.0
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 998 in model.lagmul_weiss_joints
    wj = model.lagmul_weiss_joints[998]
    assert wj.node1 == 72
    assert wj.node2 == 82
    assert wj.node3 == 92
    assert pytest.approx(wj.stiff) == 9.5e+6
    assert wj.skew_id == 11
    assert pytest.approx(wj.tol) == 1.2e-6
    assert pytest.approx(wj.axis_x) == 0.0
    assert pytest.approx(wj.axis_y) == 1.0
    assert pytest.approx(wj.axis_z) == 0.0


def test_m276_weiss_joint_aliases(tmp_path: Path):
    deck = """# RADIOSS STARTER DECK
/BEGIN
Weiss Joint Aliases Test
1 1
/WEISS_JOINT/1001
Weiss Joint Alias 1
         1         2         3              1.0e+6         0              1.0e-6
/LAGMUL/WEISS_COUPLING/1002
Weiss Coupling Alias 2
         4         5         6              2.0e+6         0              1.0e-6
/WEISS_COUPLING/1003
Weiss Coupling Alias 3
         7         8         9              3.0e+6         0              1.0e-6
/WEISS_MECHANISM/1004
Weiss Mechanism Alias 4
        10        11        12              4.0e+6         0              1.0e-6
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 1001 in model.lagmul_weiss_joints
    assert 1002 in model.lagmul_weiss_joints
    assert 1003 in model.lagmul_weiss_joints
    assert 1004 in model.lagmul_weiss_joints
    assert model.lagmul_weiss_joints[1001].node3 == 3
    assert model.lagmul_weiss_joints[1002].node3 == 6
    assert model.lagmul_weiss_joints[1003].node3 == 9
    assert model.lagmul_weiss_joints[1004].node3 == 12


# ============================================================================
# 4. /SENSOR/SPRING_FRICTION_ENERGY Tests
# ============================================================================

def test_m276_sensor_spring_friction_energy(tmp_path: Path):
    deck = """# RADIOSS STARTER DECK
/BEGIN
Spring Friction Energy Sensor Test
1 1
/SENSOR/SPRING_FRICTION_ENERGY/2500
Spring Friction Energy Sensor 1
       998              2.2e+5               0.0060
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 2500 in model.sensor_spring_friction_energies
    sens = model.sensor_spring_friction_energies[2500]
    assert sens.id == 2500
    assert sens.spring_id == 998
    assert pytest.approx(sens.u_frict_max) == 2.2e+5
    assert pytest.approx(sens.t_delay) == 0.0060

    # Global sensor list check
    matching = [s for s in model.sensors if s.id == 2500]
    assert len(matching) == 1
    assert matching[0].kind == "SPRING_FRICTION_ENERGY"
    assert pytest.approx(matching[0].tdelay) == 0.0060


def test_m276_sensor_spring_friction_energy_aliases(tmp_path: Path):
    deck = """# RADIOSS STARTER DECK
/BEGIN
Spring Friction Energy Sensor Aliases Test
1 1
/SENSOR/SPRING_FRICT_ENERGY/2600
Spring Frict Energy Alias 1
       801              4.0e+4               0.001
/SENSOR/SPRING_SLIP_ENERGY/2601
Spring Slip Energy Alias 2
       802              5.0e+4               0.002
/SENSOR/FRICTION_ENERGY_SPRING/2602
Friction Energy Spring Alias 3
       803              6.0e+4               0.003
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 2600 in model.sensor_spring_friction_energies
    assert 2601 in model.sensor_spring_friction_energies
    assert 2602 in model.sensor_spring_friction_energies
    assert pytest.approx(model.sensor_spring_friction_energies[2600].u_frict_max) == 4.0e+4
    assert pytest.approx(model.sensor_spring_friction_energies[2601].u_frict_max) == 5.0e+4
    assert pytest.approx(model.sensor_spring_friction_energies[2602].u_frict_max) == 6.0e+4
