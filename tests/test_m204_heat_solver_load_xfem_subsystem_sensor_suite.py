"""Tests for Milestone M204:
- Thermal Transient Solvers: /HEAT/SOLVER, /HEAT/GLOBAL
- Volumetric Heat Generation: /LOAD/HEAT, /LOAD/THERM, /HEAT/LOAD
- Material Thermal Modifiers: /HEAT/MAT, /HEAT/MATERIAL
- X-FEM Fracture & Enrichment: /XFEM, /XFEM/SHELL, /XFEM/SOLID, /INICRACK
- Subsystem Threshold Sensors: /SENSOR/AIRBAG, /SENSOR/MONVOL, /SENSOR/SHELL, /SENSOR/SOLID, /SENSOR/SPH
- Initial & Imposed Temperature Maps & Aliases: /INITEMP, /IMPTEMP
- Time-History & Thermal Channel Dispatch: /TH/HEAT, /TH/TEMPER, /TH/XFEM, /TH/INICRACK
"""

from pathlib import Path
import pytest
from pyradioss.common.messages import MessageLog
from pyradioss.input.deck_reader import read_deck
from pyradioss.input.starter_keywords import parse_starter_deck
from pyradioss.model.model import Model
from pyradioss.model.entities import (
    HeatSolver, XfemControl, SensorSubsystem,
    IniCrack, InitialTemperature, ImposedTemperature
)


def _parse_starter(tmp_path: Path, text: str) -> tuple[Model, MessageLog]:
    p = tmp_path / "TEST_0000.rad"
    p.write_text(text.strip() + "\n", encoding="ascii")
    blocks = read_deck(str(p))
    model = Model()
    log = MessageLog()
    parse_starter_deck(blocks, model, log)
    return model, log


def test_m204_heat_solver_and_mat(tmp_path: Path):
    deck = """# RADIOSS STARTER DECK
/BEGIN
Heat Solver and Material Test
1 1
/HEAT/SOLVER/1
Heat Solver Controls
         2         1                1e-4                0.05                1e-5
/HEAT/MAT/101
Thermal Material 1
                 0.0               900.0                45.0                 0.0
              1000.0                 0.0                 0.0                 1.0
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert 1 in model.heat_solvers
    hs = model.heat_solvers[1]
    assert hs.isolv == 2
    assert hs.itype == 1
    assert pytest.approx(hs.ttol) == 1e-4
    assert pytest.approx(hs.dttmax) == 0.05
    assert pytest.approx(hs.dttmin) == 1e-5
    assert model.heat_globals is model.heat_solvers

    assert 101 in model.mat_heat_modifiers
    hm = model.mat_heat_modifiers[101]
    assert hm.mat_id == 101
    assert pytest.approx(hm.rho0_cp) == 900.0
    assert pytest.approx(hm.as_solid) == 45.0
    assert pytest.approx(hm.t1) == 1000.0
    assert model.heat_mats is model.mat_heat_modifiers


def test_m204_load_heat_and_temperatures(tmp_path: Path):
    deck = """# RADIOSS STARTER DECK
/BEGIN
Load Heat and Temperatures Test
1 1
/LOAD/HEAT/201
Heat Source 1
         102              5000.0           5                 1.5           0
/INITEMP/301
Initial Temperature 1
               300.0         103           0
/IMPTEMP/401
Imposed Temperature 1
           6           0         104
                 1.0                 1.0                 0.0              100.0
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert 201 in model.load_therms
    lh = model.load_therms[201]
    assert lh.group_id == 102
    assert pytest.approx(lh.flux) == 5000.0
    assert lh.funct_id == 5
    assert pytest.approx(lh.scale) == 1.5
    assert model.load_heats is model.load_therms
    assert model.heat_loads is model.load_therms

    assert 301 in model.initemps
    it = model.initemps[301]
    assert it.id == 301
    assert pytest.approx(it.t0) == 300.0
    assert it.grnod_id == 103
    assert model.initial_temperatures is model.initemps

    assert 401 in model.imptemps
    ipt = model.imptemps[401]
    assert ipt.id == 401
    assert ipt.funct_id == 6
    assert ipt.grnod_id == 104
    assert pytest.approx(ipt.tstop) == 100.0
    assert model.imposed_temperatures is model.imptemps


def test_m204_xfem_and_inicrack(tmp_path: Path):
    deck = """# RADIOSS STARTER DECK
/BEGIN
XFEM and Inicrack Test
1 1
/XFEM/SHELL/501
XFEM Shell Control
         105         601           1           2
/INICRACK/601
Planar Initial Crack
         105           1
                 0.0                 0.0                 0.0
                 1.0                 0.0                 0.0
                 0.0                 1.0                 0.0
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert 501 in model.xfem_controls
    xc = model.xfem_controls[501]
    assert xc.id == 501
    assert xc.subtype == "SHELL"
    assert xc.grpart_id == 105
    assert xc.crack_id == 601
    assert xc.ifail == 1
    assert xc.i_enrich == 2

    assert 601 in model.inicracks
    ic = model.inicracks[601]
    assert ic.id == 601
    assert ic.grsh_id == 105
    assert ic.open_flag == 1
    assert ic.p2 == [1.0, 0.0, 0.0]
    assert ic.norm == [0.0, 1.0, 0.0]


def test_m204_subsystem_sensors(tmp_path: Path):
    deck = """# RADIOSS STARTER DECK
/BEGIN
Subsystem Sensors Test
1 1
/SENSOR/AIRBAG/701
Airbag Pressure Sensor
                 0.0
         101                10.0               100.0                 0.0
/SENSOR/SHELL/702
Shell Sensor
                 0.1
         102                 0.0                 0.2                0.01
/SENSOR/SOLID/703
Solid Sensor
                 0.2
         103                 0.0                 0.5                0.02
/SENSOR/SPH/704
SPH Sensor
                 0.3
         104                 0.0                 1.0                0.03
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert 701 in model.sensors_airbag
    sa = model.sensors_airbag[701]
    assert sa.target_id == 101
    assert pytest.approx(sa.v1) == 10.0
    assert pytest.approx(sa.v2) == 100.0

    assert 702 in model.sensors_shell
    ss = model.sensors_shell[702]
    assert ss.target_id == 102
    assert pytest.approx(ss.tdelay) == 0.1
    assert pytest.approx(ss.v2) == 0.2

    assert 703 in model.sensors_solid
    s_sol = model.sensors_solid[703]
    assert s_sol.target_id == 103
    assert pytest.approx(s_sol.v2) == 0.5

    assert 704 in model.sensors_sph
    s_sph = model.sensors_sph[704]
    assert s_sph.target_id == 104
    assert pytest.approx(s_sph.v2) == 1.0


def test_m204_th_heat_channels(tmp_path: Path):
    deck = """# RADIOSS STARTER DECK
/BEGIN
TH Heat Channels Test
1 1
/TH/HEAT/801
Nodal Thermal TH
TEMP FLUX
         101       102
/TH/XFEM/802
XFEM Fracture TH
DEF
         501
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(model.th_requests) == 2
    th1 = model.th_requests[0]
    assert th1.id == 801
    assert th1.kind == "HEAT"
    assert "TEMP" in th1.variables
    assert 101 in th1.obj_ids

    th2 = model.th_requests[1]
    assert th2.id == 802
    assert th2.kind == "XFEM"
    assert 501 in th2.obj_ids
