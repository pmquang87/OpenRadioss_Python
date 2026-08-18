"""Tests for Milestone M202:
- Thermal Surface Boundary Conditions: /HEAT/FLUX, /FLUX, /HEAT/CONVEC, /CONVEC, /HEAT/RADIATION, /RADIATION
- Hydrodynamic Johnson-Cook Material Law 4: /MAT/LAW4, /MAT/HYD_JCOOK
- Plastic / Internal Work Threshold Sensor: /SENSOR/WORK, /SENSOR/TYPE13
- Engine Thermal & Output Directives: /TFILE, /H3D/DT, /HEAT
- Associated Entities: Spcnd, SurfSurf, BcsLagmul, Ddw, DdwPoint, Stamping
"""

from pathlib import Path
import pytest
from pyradioss.common.messages import MessageLog
from pyradioss.input.deck_reader import read_deck
from pyradioss.input.starter_keywords import parse_starter_deck
from pyradioss.input.engine_keywords import parse_engine_deck
from pyradioss.model.model import Model, EngineControls
from pyradioss.model.entities import (
    HeatFlux, HeatConvec, HeatRadiation, MatLaw4, SensorWork,
    Spcnd, SurfSurf, BcsLagmul, Ddw, DdwPoint, Stamping
)


def _parse_starter(tmp_path: Path, text: str) -> tuple[Model, MessageLog]:
    p = tmp_path / "TEST_0000.rad"
    p.write_text(text.strip() + "\n", encoding="ascii")
    blocks = read_deck(str(p))
    model = Model()
    log = MessageLog()
    parse_starter_deck(blocks, model, log)
    return model, log


def _parse_engine(tmp_path: Path, text: str) -> tuple[EngineControls, MessageLog]:
    p = tmp_path / "TEST_0001.rad"
    p.write_text(text.strip() + "\n", encoding="ascii")
    blocks = read_deck(str(p))
    log = MessageLog()
    ec = parse_engine_deck(blocks, log)
    return ec, log


def test_m202_heat_flux_parsing(tmp_path: Path):
    deck = """# RADIOSS STARTER DECK
/BEGIN
Thermal Flux Test
1 1
/HEAT/FLUX/101
Thermal Flux Surface 1
       101         2         3                 1.5                 2.5
                 0.0                 1.0               100.0
/FLUX/102
Thermal Flux Surface 2
       102         4         0                 1.0                 1.0
                 0.5                 2.0               250.0
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert 101 in model.heat_fluxes
    hf1 = model.heat_fluxes[101]
    assert hf1.id == 101
    assert hf1.surf_id == 101
    assert hf1.funct_id == 2
    assert hf1.sensor_id == 3
    assert pytest.approx(hf1.ascale) == 1.5
    assert pytest.approx(hf1.fscale) == 2.5
    assert pytest.approx(hf1.tstart) == 0.0
    assert pytest.approx(hf1.tstop) == 1.0
    assert pytest.approx(hf1.q) == 100.0

    assert 102 in model.heat_fluxes
    hf2 = model.heat_fluxes[102]
    assert hf2.id == 102
    assert hf2.surf_id == 102
    assert hf2.funct_id == 4
    assert hf2.sensor_id == 0
    assert pytest.approx(hf2.ascale) == 1.0
    assert pytest.approx(hf2.fscale) == 1.0
    assert pytest.approx(hf2.tstart) == 0.5
    assert pytest.approx(hf2.tstop) == 2.0
    assert pytest.approx(hf2.q) == 250.0


def test_m202_heat_convec_and_radiation_parsing(tmp_path: Path):
    deck = """# RADIOSS STARTER DECK
/BEGIN
Thermal Convec and Rad Test
1 1
/HEAT/CONVEC/201
Convection Boundary 1
       201         5         1
                 1.0                 1.0                 0.0                 5.0                25.0
/HEAT/RADIATION/301
Radiation Boundary 1
       301         6         0
                 1.0                 1.0                 0.0                10.0                0.85
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert 201 in model.heat_convecs
    hc = model.heat_convecs[201]
    assert hc.id == 201
    assert hc.surf_id == 201
    assert hc.funct_id == 5
    assert hc.sensor_id == 1
    assert pytest.approx(hc.ascale) == 1.0
    assert pytest.approx(hc.fscale) == 1.0
    assert pytest.approx(hc.tstart) == 0.0
    assert pytest.approx(hc.tstop) == 5.0
    assert pytest.approx(hc.h) == 25.0
    assert len(model.convec_loads) >= 1

    assert 301 in model.heat_radiations
    hr = model.heat_radiations[301]
    assert hr.id == 301
    assert hr.surf_id == 301
    assert hr.funct_id == 6
    assert hr.sensor_id == 0
    assert pytest.approx(hr.ascale) == 1.0
    assert pytest.approx(hr.fscale) == 1.0
    assert pytest.approx(hr.tstart) == 0.0
    assert pytest.approx(hr.tstop) == 10.0
    assert pytest.approx(hr.emissivity) == 0.85
    assert pytest.approx(hr.emiss) == 0.85
    assert len(model.radiation_loads) >= 1


def test_m202_mat_law4_hyd_jcook_parsing(tmp_path: Path):
    deck = """# RADIOSS STARTER DECK
/BEGIN
Mat Law4 Test
1 1
/MAT/LAW4/401
Hydrodynamic Johnson-Cook Material
              7.8e-6              4500.0                 1.5                 1.8                 0.5
               500.0               800.0                0.28               0.015               0.001               900.0
               293.0              1800.0               460.0               460.0                 0.0
                 0.0                 0.0                 0.0                 0.0                 0.0
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert 401 in model.mat_law4s
    mat = model.mat_law4s[401]
    assert mat.id == 401
    assert pytest.approx(mat.rho_i) == 7.8e-6
    assert pytest.approx(mat.c0_eos) == 4500.0
    assert pytest.approx(mat.s_eos) == 1.5
    assert pytest.approx(mat.gamma0) == 1.8
    assert pytest.approx(mat.a_eos) == 0.5
    assert pytest.approx(mat.a) == 500.0
    assert pytest.approx(mat.b) == 800.0
    assert pytest.approx(mat.n) == 0.28
    assert pytest.approx(mat.c) == 0.015
    assert pytest.approx(mat.eps_max) == 0.001
    assert pytest.approx(mat.sig_max) == 900.0
    assert pytest.approx(mat.t0) == 293.0
    assert pytest.approx(mat.tm) == 1800.0
    assert pytest.approx(mat.cp) == 460.0
    assert 401 in model.materials


def test_m202_sensor_work_parsing(tmp_path: Path):
    deck = """# RADIOSS STARTER DECK
/BEGIN
Sensor Work Test
1 1
/SENSOR/WORK/501
Plastic Work Sensor 1
                 0.0
         101         1              5000.0                 0.0
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert 501 in model.sensors_work
    sw = model.sensors_work[501]
    assert sw.id == 501
    assert sw.object_id == 101
    assert sw.sens_type == 1
    assert pytest.approx(sw.t_delay) == 0.0
    assert pytest.approx(sw.w_max) == 5000.0

    # Also verify standard Sensor entity in model.sensors
    work_sensors = [s for s in model.sensors if s.id == 501]
    assert len(work_sensors) == 1
    assert work_sensors[0].kind == "WORK"
    assert pytest.approx(work_sensors[0].work_max) == 5000.0


def test_m202_engine_heat_tfile_h3d_pipeline(tmp_path: Path):
    deck = """# RADIOSS ENGINE DECK
/RUN/RUN1/1
0.1
/TFILE
0.001
/H3D/DT
0.0 0.005
/HEAT
0.002
"""
    ec, log = _parse_engine(tmp_path, deck)
    assert pytest.approx(ec.t_end) == 0.1
    assert pytest.approx(ec.th_dt) == 0.001
    assert pytest.approx(ec.tfile_dt) == 0.001
    assert pytest.approx(ec.h3d_dt) == 0.005
    assert ec.heat_active is True
    assert ec.heat_flag is True


def test_m202_associated_entities():
    # Spcnd, SurfSurf, BcsLagmul, Ddw, DdwPoint, Stamping
    spc = Spcnd(id=1, node_id=10, dof="111000", title="Spc Node")
    assert spc.id == 1 and spc.dof == "111000" and spc.node_id == 10

    ss = SurfSurf(id=2, surf_ids=[100, 200], title="Surface Group")
    assert ss.id == 2 and ss.surf_ids == [100, 200]

    bl = BcsLagmul(id=3, grnod_id=5, tra="100", rot="000", title="Lagrange Multiplier")
    assert bl.id == 3 and bl.grnod_id == 5 and bl.tra == "100"

    ddwp = DdwPoint(id=1, node_id=10, dir="Z", scale=1.0)
    ddw = Ddw(id=4, tool_type=1, surf_id=50, title="Drawbead")
    assert ddw.id == 4 and ddw.tool_type == 1 and ddwp.node_id == 10

    stamp = Stamping(hf_timescale=1.5, datalines=["LINE1", "LINE2"])
    assert stamp.hf_timescale == 1.5 and len(stamp.datalines) == 2
