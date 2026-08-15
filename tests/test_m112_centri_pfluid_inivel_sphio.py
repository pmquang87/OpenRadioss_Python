"""Tests for Milestone M112:
Centrifugal & Pressure Loads, Advanced Initial Velocities, Final Geometry Imposed Fields,
Thermal Rigid Walls, and SPH Boundary Suite.
"""
from __future__ import annotations

import pytest
from pyradioss.common.messages import MessageLog
from pyradioss.starter.starter import run_starter
from pyradioss.starter.checks import check_model


_BOILERPLATE = """\
#---1---+----2---+----3---+----4---+----5---+----6---+----7---+----8---+----9---+---10---+
/BEGIN
Test_Deck_M112
      2022         0
/MAT/LAW1/1
Mat_1
              7.8e-9
            210000.0                 0.3
/PROP/TYPE1/1
Dummy_Shell
         1         1         1         0         0         0         0
                 1.0                 1.0                 1.0
                 1.0            0.833333
/PART/1
Part_Shell_1
         1         1
/PART/2
Part_Shell_2
         1         1
/NODE
         1                 0.0                 0.0                 0.0
         2                 2.0                 0.0                 0.0
         3                 2.0                 2.0                 0.0
         4                 0.0                 2.0                 0.0
/SHELL/1
         1         1         2         3         4
/GRNOD/NODE/1
All Nodes
         1         2         3         4
/FUNCT/10
Time function
                 0.0                 0.0
                 1.0                 1.0
/FUNCT/1
Curve 1
                 0.0                 0.0
                 1.0                 1.0
/FUNCT/2
Curve 2
                 0.0                 0.0
                 1.0                 2.0
/FUNCT/5
Function 5
                 0.0                 0.0
                 1.0                 1.0
/SKEW/FIX/1
Skew 1
                 0.0                 0.0                 0.0
                 1.0                 0.0                 0.0
                 0.0                 1.0                 0.0
/FRAME/FIX/1
Frame 1
                 0.0                 0.0                 0.0
                 1.0                 0.0                 0.0
                 0.0                 1.0                 0.0
/SENSOR/TIME/1
Sensor 1
                 0.0                 0.5
/SURF/PART/1
Surface 1
         1
"""


def _run(tmp_path, deck, name="TEST_0000.rad"):
    p = tmp_path / name
    p.write_text(deck, encoding="ascii")
    log = MessageLog()
    model = run_starter(str(p), log)
    return model, log


def test_m112_load_centri_and_pressure_fixed(tmp_path):
    deck = _BOILERPLATE + """\
/LOAD/CENTRI/1
Centrifugal Body Load
        10        XX         1         1         1         1                 1.5                 2.0
/LOAD/PRESSURE/1
Surface Pressure
         1        10         1
                 1.2                 0.1                10.0
/END
"""
    model, log = _run(tmp_path, deck)
    assert len(log.errors) == 0, f"Errors: {log.messages}"
    assert 1 in model.load_centris
    lc = model.load_centris[1]
    assert lc.fct_id == 10
    assert lc.dir == "XX"
    assert lc.frame_id == 1
    assert lc.sens_id == 1
    assert lc.grnod_id == 1
    assert lc.ivar == 1
    assert abs(lc.ascalex - 1.5) < 1e-6
    assert abs(lc.fscaley - 2.0) < 1e-6

    assert 1 in model.load_pressures
    lp = model.load_pressures[1]
    assert lp.surf_id == 1
    assert lp.fct_id == 10
    assert lp.sens_id == 1
    assert abs(lp.scale - 1.2) < 1e-6
    assert abs(lp.tstart - 0.1) < 1e-6
    assert abs(lp.tstop - 10.0) < 1e-6

    chk_log = MessageLog()
    check_model(model, chk_log)
    assert len(chk_log.errors) == 0, f"Cross-ref errors: {chk_log.messages}"


def test_m112_load_pfluid_free(tmp_path):
    free_deck = """# Free format deck
/BEGIN
Free_Deck_M112
/MAT/LAW1/1
Mat_1
7.8e-9
210000.0 0.3
/PROP/TYPE1/1
Dummy_Shell
1 1 1 0 0 0 0
1.0 1.0 1.0
1.0 0.833333
/PART/1
Part_Shell_1
1 1
/NODE
1 0.0 0.0 0.0
2 2.0 0.0 0.0
3 2.0 2.0 0.0
4 0.0 2.0 0.0
/SHELL/1
1 1 2 3 4
/SURF/PART/1
Surface 1
1
/FUNCT/1
Funct 1
0.0 0.0
1.0 1.0
/FUNCT/2
Funct 2
0.0 0.0
1.0 1.0
/LOAD/PFLUID/1
PFluid Load
1 0
1 1.0 2.5
Z 0
2 1.0 3.5
0 1.0 0.0
Z 0
/END
"""
    model, log = _run(tmp_path, free_deck)
    assert len(log.errors) == 0, f"Errors: {log.messages}"
    assert 1 in model.pfluid_loads
    pl = model.pfluid_loads[1]
    assert pl.surf_id == 1
    assert pl.fct_id_t == 1
    assert abs(pl.fscaley - 2.5) < 1e-6
    assert pl.fct_id_pc == 2
    assert abs(pl.fscaley_pc - 3.5) < 1e-6


def test_m112_inivel_axis_fvm_node(tmp_path):
    deck = _BOILERPLATE + """\
/INIVEL/AXIS/1
Rotational Inivel
        ZZ         1         1
                 1.0                 2.0                 3.0                 4.0
/INIVEL/FVM/2
FVM Airbag Velocity
                 5.0                 6.0                 7.0         0         0         0         1
/INIVEL/NODE/3
Nodal Velocities
         1         1                 1.1                 2.2                 3.3
                                     4.4                 5.5                 6.6
         2         0                 7.7                 8.8                 9.9
/END
"""
    model, log = _run(tmp_path, deck)
    assert len(log.errors) == 0, f"Errors: {log.messages}"
    assert 1 in model.inivel_axes
    ia = model.inivel_axes[1]
    assert ia.dir == "ZZ"
    assert ia.frame_id == 1
    assert ia.grnod_id == 1
    assert abs(ia.vx - 1.0) < 1e-6
    assert abs(ia.vr - 4.0) < 1e-6

    assert 2 in model.inivel_fvms
    iv = model.inivel_fvms[2]
    assert abs(iv.vx - 5.0) < 1e-6
    assert abs(iv.vy - 6.0) < 1e-6
    assert abs(iv.vz - 7.0) < 1e-6
    assert iv.skew_id == 1

    assert 3 in model.inivel_nodes
    in_node = model.inivel_nodes[3]
    assert len(in_node.items) == 2
    assert in_node.items[0].node_id == 1
    assert in_node.items[0].skew_id == 1
    assert abs(in_node.items[0].vxt - 1.1) < 1e-6
    assert abs(in_node.items[0].vxr - 4.4) < 1e-6
    assert in_node.items[1].node_id == 2
    assert abs(in_node.items[1].vxt - 7.7) < 1e-6

    chk_log = MessageLog()
    check_model(model, chk_log)
    assert len(chk_log.errors) == 0, f"Cross-ref errors: {chk_log.messages}"


def test_m112_impdisp_and_impvel_fgeo(tmp_path):
    deck = _BOILERPLATE + """\
/IMPDISP/FGEO/1
Final Geometry Displacement
         1         1                   0
                 1.0                                     0.0                10.0
         1                 0.5                 0.0                 0.0
         2                 1.5                 0.0                 0.0
/IMPVEL/FGEO/2
Final Geometry Velocity
         1         1         2         0
                 1.0                 0.0                 0.0                 1.0                 0.01
         1         2
/END
"""
    model, log = _run(tmp_path, deck)
    assert len(log.errors) == 0, f"Errors: {log.messages}"
    assert 1 in model.impdisp_fgeos
    idf = model.impdisp_fgeos[1]
    assert idf.fct_id == 1
    assert idf.part_id == 1
    assert len(idf.nodes) == 2
    assert idf.nodes[0]["node_id"] == 1
    assert abs(idf.nodes[0]["x"] - 0.5) < 1e-6

    assert 2 in model.impvel_fgeos
    ivf = model.impvel_fgeos[2]
    assert ivf.fct_id == 1
    assert ivf.part_id == 1
    assert ivf.fct_l_id == 2
    assert len(ivf.pairs) == 1
    assert ivf.pairs[0] == (1, 2)

    chk_log = MessageLog()
    check_model(model, chk_log)
    assert len(chk_log.errors) == 0, f"Cross-ref errors: {chk_log.messages}"


def test_m112_rwall_therm_and_sph_inout(tmp_path):
    deck = _BOILERPLATE + """\
/RWALL/THERM/1
Thermal Rigid Wall
         1         1         1         0
         5               300.0                10.0                 0.2
/SPH/INOUT/1
SPH Inlet Outlet
         1         1         5
                 1.0               100.0                 0.5
/END
"""
    model, log = _run(tmp_path, deck)
    assert len(log.errors) == 0, f"Errors: {log.messages}"
    assert 1 in model.rwall_therms
    rt = model.rwall_therms[1]
    assert rt.node_id == 1
    assert rt.tied == 1
    assert rt.grnod_id1 == 1
    assert rt.fct_id == 5
    assert abs(rt.temp - 300.0) < 1e-6
    assert abs(rt.tstif - 10.0) < 1e-6
    assert abs(rt.fric - 0.2) < 1e-6

    assert 1 in model.sph_inouts
    sio = model.sph_inouts[1]
    assert sio.surf_id == 1
    assert sio.part_id == 1
    assert sio.fct_id == 5
    assert abs(sio.rho_in - 1.0) < 1e-6
    assert abs(sio.p_in - 100.0) < 1e-6
    assert abs(sio.e_in - 0.5) < 1e-6

    chk_log = MessageLog()
    check_model(model, chk_log)
    assert len(chk_log.errors) == 0, f"Cross-ref errors: {chk_log.messages}"
