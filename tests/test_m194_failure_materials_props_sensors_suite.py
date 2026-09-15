"""Tests for Milestone M194:
- /FAIL/TAB1
- /MAT/LAW40 (/MAT/CONCR_SUB)
- /MAT/LAW80 (/MAT/BARLAT3)
- /MAT/LAW102 (/MAT/HILL_48)
- /MAT/NLOCAL
- /PROP/TYPE13 (/PROP/SPR_PULL)
- /SENSOR/INTER
- /ACTIV
"""
from pathlib import Path
import pytest
from pyradioss.common.messages import MessageLog
from pyradioss.input.deck_reader import read_deck
from pyradioss.input.starter_keywords import parse_starter_deck
from pyradioss.model.model import Model
from pyradioss.model.entities import (
    FailTab1, MatLaw40, MatLaw80, MatLaw102, MatNLocal,
    PropType13, Sensor, ElementActivation
)


def _parse(tmp_path: Path, text: str) -> tuple[Model, MessageLog]:
    p = tmp_path / "TEST_0000.rad"
    p.write_text(text.strip() + "\n", encoding="ascii")
    blocks = read_deck(str(p))
    model = Model()
    log = MessageLog()
    parse_starter_deck(blocks, model, log)
    return model, log


def test_fail_tab1_fixed_and_free(tmp_path: Path):
    deck_str = """# RADIOSS STARTER
/BEGIN
Test FAIL TAB1
/FAIL/TAB1/10
         1         1       0.0       0.0         0
       1.0       0.0       1.0       0.0         0
       101       1.0       1.0       102       1.0       1.0
         0       1.0       1.0       0.0       1.0       0.0
         0       1.0
/FAIL/TAB1/11
1 1 0.0 0.0 0
1.0 0.0 1.0 0.0 0
201 1.0 1.0 202 1.0 1.0
0 1.0 1.0 0.0 1.0 0.0
0 1.0
/END
"""
    model, log = _parse(tmp_path, deck_str)
    assert 10 in model.fail_tab1s
    assert 11 in model.fail_tab1s
    f10 = model.fail_tab1s[10]
    assert f10.mat_id == 10
    assert f10.ifail_sh == 1
    assert f10.table1_id == 101
    assert f10.table2_id == 102

    f11 = model.fail_tab1s[11]
    assert f11.mat_id == 11
    assert f11.table1_id == 201
    assert f11.table2_id == 202


def test_mat_law40_fixed_and_free(tmp_path: Path):
    deck_str = """# RADIOSS STARTER
/BEGIN
Test LAW40
/MAT/LAW40/1
Concrete Subgrade 1
#        RHO         E        NU
      2.4e-6    30000.       0.2
#     PARAM1    PARAM2    PARAM3
        10.0      20.0      30.0
/MAT/CONCR_SUB/2
Concrete Subgrade 2
2.5e-6 32000. 0.18
15.0 25.0 35.0
/END
"""
    model, log = _parse(tmp_path, deck_str)
    assert 1 in model.mat_law40s
    assert 2 in model.mat_law40s
    m1 = model.mat_law40s[1]
    assert m1.id == 1
    assert m1.rho == pytest.approx(2.4e-6)
    assert m1.e == pytest.approx(30000.0)
    assert m1.nu == pytest.approx(0.2)
    assert m1.params["param_0"] == pytest.approx(10.0)

    m2 = model.mat_law40s[2]
    assert m2.id == 2
    assert m2.rho == pytest.approx(2.5e-6)
    assert m2.e == pytest.approx(32000.0)
    assert m2.nu == pytest.approx(0.18)


def test_mat_law80_fixed_and_free(tmp_path: Path):
    deck_str = """# RADIOSS STARTER
/BEGIN
Test LAW80
/MAT/LAW80/1
Phase Transfo Mat 1
#        RHO    RHO_REF
      7.8e-6     7.8e-6
#          E         NU    FCT_IDE    SCALE_E  TIME_UNIT
    205000.0       0.29        301        1.1     3600.0
/MAT/TRANSFO/2
Phase Transfo Mat 2
7.85e-6 7.85e-6
210000.0 0.3 0 1.0 3600.0
/END
"""
    model, log = _parse(tmp_path, deck_str)
    assert 1 in model.mat_law80s
    assert 2 in model.mat_law80s
    m1 = model.mat_law80s[1]
    assert m1.id == 1
    assert m1.rho0 == pytest.approx(7.8e-6)
    assert m1.e == pytest.approx(205000.0)
    assert m1.nu == pytest.approx(0.29)
    assert m1.fct_ide == 301

    m2 = model.mat_law80s[2]
    assert m2.id == 2
    assert m2.rho0 == pytest.approx(7.85e-6)
    assert m2.e == pytest.approx(210000.0)
    assert m2.nu == pytest.approx(0.3)


def test_mat_law102_fixed_and_free(tmp_path: Path):
    deck_str = """# RADIOSS STARTER
/BEGIN
Test LAW102
/MAT/LAW102/1
Hill 48 Material 1
#        RHO         E        NU
      2.7e-6    70000.      0.33
#          F         G         H         L         M         N
         0.5       0.5       0.5       1.5       1.5       1.5
/MAT/HILL_48/2
Hill 48 Material 2
2.8e-6 72000. 0.32
0.6 0.4 0.5 1.5 1.5 1.5
/END
"""
    model, log = _parse(tmp_path, deck_str)
    assert 1 in model.mat_law102s
    assert 2 in model.mat_law102s
    m1 = model.mat_law102s[1]
    assert m1.id == 1
    assert m1.rho == pytest.approx(2.7e-6)
    assert m1.e == pytest.approx(70000.0)
    assert m1.nu == pytest.approx(0.33)
    assert m1.params["param_0"] == pytest.approx(0.5)

    m2 = model.mat_law102s[2]
    assert m2.id == 2
    assert m2.rho == pytest.approx(2.8e-6)
    assert m2.e == pytest.approx(72000.0)
    assert m2.nu == pytest.approx(0.32)


def test_mat_nlocal_fixed_and_free(tmp_path: Path):
    deck_str = """# RADIOSS STARTER
/BEGIN
Test NLOCAL
/MAT/NLOCAL/1
Nonlocal Material 1
#        RHO         E        NU
      7.8e-6   210000.       0.3
#      P1         P2
      0.05       2.0
/END
"""
    model, log = _parse(tmp_path, deck_str)
    assert 1 in model.mat_nlocals
    m1 = model.mat_nlocals[1]
    assert m1.id == 1
    assert m1.rho == pytest.approx(7.8e-6)
    assert m1.e == pytest.approx(210000.0)
    assert m1.nu == pytest.approx(0.3)
    assert m1.params["param_0"] == pytest.approx(0.05)


def test_prop_type13_fixed_and_free(tmp_path: Path):
    deck_str = """# RADIOSS STARTER
/BEGIN
Test PROP TYPE13
/PROP/TYPE13/1
Pull Spring Property 1
#      STIFF      F_MAX
      1000.0      500.0
/PROP/SPR_PULL/2
Pull Spring Property 2
2000.0 800.0
/END
"""
    model, log = _parse(tmp_path, deck_str)
    assert 1 in model.prop_spr_pulls
    assert 2 in model.prop_spr_pulls
    p1 = model.prop_spr_pulls[1]
    assert p1.id == 1
    assert p1.stiff == pytest.approx(1000.0)
    assert p1.f_max == pytest.approx(500.0)

    p2 = model.prop_spr_pulls[2]
    assert p2.id == 2
    assert p2.stiff == pytest.approx(2000.0)
    assert p2.f_max == pytest.approx(800.0)


def test_sensor_inter_fixed_and_free(tmp_path: Path):
    deck_str = """# RADIOSS STARTER
/BEGIN
Test SENSOR INTER
/SENSOR/INTER/101
Interface Sensor 1
#   TDELAY
       0.1
#   INT_ID       DIR      FMIN      FMAX      TMIN      FCUT
        12         X       5.0      50.0       0.0       0.0
/SENSOR/INTER/102
Interface Sensor 2
0.05
15 Y 10.0 100.0 0.0 0.0
/END
"""
    model, log = _parse(tmp_path, deck_str)
    sensors_by_id = {s.id: s for s in model.sensors}
    assert 101 in sensors_by_id
    assert 102 in sensors_by_id
    s1 = sensors_by_id[101]
    assert isinstance(s1, Sensor)
    assert s1.id == 101
    assert s1.kind == "INTER"
    assert s1.int_id == 12
    assert s1.dir == "X"
    assert s1.fmin == pytest.approx(5.0)
    assert s1.fmax == pytest.approx(50.0)
    assert s1.tdelay == pytest.approx(0.1)

    s2 = sensors_by_id[102]
    assert isinstance(s2, Sensor)
    assert s2.id == 102
    assert s2.kind == "INTER"
    assert s2.int_id == 15
    assert s2.dir == "Y"
    assert s2.fmin == pytest.approx(10.0)
    assert s2.fmax == pytest.approx(100.0)
    assert s2.tdelay == pytest.approx(0.05)


def test_activ_directive_fixed_and_free(tmp_path: Path):
    deck_str = """# RADIOSS STARTER
/BEGIN
Test ACTIV
/ACTIV/1
Activation Directive 1
#  SENS_ID   GRBRIC   GRQUAD   GRSHEL   GRTRUS   GRBEAM   GRSPRI   GRSH3N    BLANK    IFORM
         2        0        0       10        0        0        0        0                 1
#   TSTART    TSTOP
       0.0    100.0
/ACTIV/2
Activation Directive 2
3 0 0 20 0 0 0 0 2
0.05 50.0
/END
"""
    model, log = _parse(tmp_path, deck_str)
    assert len(model.activations) == 2
    a1 = model.activations[0]
    assert a1.id == 1
    assert a1.sens_id == 2
    assert a1.grshel_id == 10
    assert a1.iform == 1
    assert a1.tstart == pytest.approx(0.0)
    assert a1.tstop == pytest.approx(100.0)

    a2 = model.activations[1]
    assert a2.id == 2
    assert a2.sens_id == 3
    assert a2.grshel_id == 20
    assert a2.iform == 2
    assert a2.tstart == pytest.approx(0.05)
    assert a2.tstop == pytest.approx(50.0)
