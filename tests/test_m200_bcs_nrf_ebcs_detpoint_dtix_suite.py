"""Test suite for Milestone M200:
- /BCS/NRF and /BCS_NRF (Non-reflecting boundary condition on node group)
- /EBCS/CYCLIC (Elementary cyclic boundary condition)
- /EBCS/PROPELLANT (Elementary propellant combustion boundary condition)
- /DFS/DETPOINT/NODE, /DETPOINT/NODE (Detonation point at node)
- /DFS/DETPOINT/SET, /DETPOINT/SET, /DETPOINT/GRNOD (Detonation point on node group)
- /DTIX, /ENG/DTIX (Initial and maximum explicit time step control)
- /PARITH/ON, /PARITH/OFF (Parallel arithmetic reproducibility)
- /TH/TITLE (Time-history title block)
- /DYNAIN/SHELL/AUX/FULL, /DYNAIN/SHELL/STRES/FULL, /DYNAIN/SHELL/STRAIN/FULL (Dynain shell outputs)
"""

from pathlib import Path
import pytest
from pyradioss.common.messages import MessageLog
from pyradioss.input.deck_reader import read_deck
from pyradioss.input.starter_keywords import parse_starter_deck
from pyradioss.input.engine_keywords import parse_engine_deck
from pyradioss.model.model import Model, EngineControls
from pyradioss.model.entities import (
    BcsNrf,
    EbcsCyclic,
    EbcsPropellant,
    DetPointNode,
    DetPointSet,
    DtixControl,
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


def test_m200_bcs_nrf(tmp_path: Path):
    """Test /BCS/NRF and /BCS_NRF keyword reading."""
    deck_str = """# RADIOSS STARTER
/BEGIN
BCS NRF Test
                 1                 1
/BCS/NRF/101
Non-reflecting BCS Node Group 101
#  grnod_ID     Iskep  frame_ID     Isurf      Ivel      Isub      Ityp              factor
        10         1         2         3         1         0         2                 0.85
/BCS_NRF/102
Non-reflecting BCS Node Group 102
        20         0         0         0         0         0         0                 1.00
/END
"""
    model, log = _parse_starter(tmp_path, deck_str)
    assert len(log.errors) == 0
    assert len(model.bcs_nrfs) == 2
    b101 = model.bcs_nrfs[101]
    assert isinstance(b101, BcsNrf)
    assert b101.id == 101
    assert b101.grnod_id == 10
    assert b101.iskep == 1
    assert b101.frame_id == 2
    assert b101.isurf == 3
    assert b101.factor == pytest.approx(0.85)

    b102 = model.bcs_nrfs[102]
    assert b102.id == 102
    assert b102.grnod_id == 20
    assert b102.factor == pytest.approx(1.00)


def test_m200_ebcs_cyclic(tmp_path: Path):
    """Test /EBCS/CYCLIC keyword reading."""
    deck_str = """# RADIOSS STARTER
/BEGIN
EBCS CYCLIC Test
                 1                 1
/EBCS/CYCLIC/1
Cyclic Boundary 1
# surf_ID1   node_id1  node_id2  node_id3
         1         11        12        13
# surf_ID2   node_id4  node_id5  node_id6
         2         21        22        23
/END
"""
    model, log = _parse_starter(tmp_path, deck_str)
    assert len(log.errors) == 0
    assert 1 in model.ebcs_cyclics
    c = model.ebcs_cyclics[1]
    assert isinstance(c, EbcsCyclic)
    assert c.id == 1
    assert c.surf1_id == 1
    assert c.surf_id1 == 1
    assert c.node_id1 == 11
    assert c.node_id2 == 12
    assert c.node_id3 == 13
    assert c.surf2_id == 2
    assert c.surf_id2 == 2
    assert c.node_id4 == 21
    assert c.node_id5 == 22
    assert c.node_id6 == 23


def test_m200_ebcs_propellant(tmp_path: Path):
    """Test /EBCS/PROPELLANT keyword reading."""
    deck_str = """# RADIOSS STARTER
/BEGIN
EBCS PROPELLANT Test
                 1                 1
/EBCS/PROPELLANT/5
Propellant Boundary 5
#  surf_id sensor_id submat_id ienthalpy
         3         7         2         1
#    rho0s     Tburn
    1250.0     650.0
#  param_a   param_n
      1.5e-3     0.45
# ffunc_id             fscaleX   fscaleY
        10                 1.0       2.0
# gfunc_id             gscaleX   gscaleY
        11                 1.5       1.0
# hfunc_id             hscaleX   hscaleY
        12                 0.5       0.5
/END
"""
    model, log = _parse_starter(tmp_path, deck_str)
    assert len(log.errors) == 0
    assert 5 in model.ebcs_propellants
    p = model.ebcs_propellants[5]
    assert isinstance(p, EbcsPropellant)
    assert p.id == 5
    assert p.surf_id == 3
    assert p.sens_id == 7
    assert p.submat_id == 2
    assert p.ienthalpy == 1
    assert p.rho0s == pytest.approx(1250.0)
    assert p.tburn == pytest.approx(650.0)
    assert p.param_a == pytest.approx(1.5e-3)
    assert p.param_n == pytest.approx(0.45)
    assert p.f_func_id == 10
    assert p.f_scale_y == pytest.approx(2.0)
    assert p.g_func_id == 11
    assert p.h_func_id == 12


def test_m200_detpoint_node(tmp_path: Path):
    """Test /DFS/DETPOINT/NODE and /DETPOINT/NODE reading."""
    deck_str = """# RADIOSS STARTER
/BEGIN
DETPOINT NODE Test
                 1                 1
/DFS/DETPOINT/NODE/1
Detonation point at node 100
#  Ishadow   Iframe1   Iframe2            R0_shadow             (blank)                TDET    mat_ID  node_ID1
         0         0         0                  0.0                                 1.25e-3         4       100
/DETPOINT/NODE/2
Detonation point at node 200
         1         1         2                  5.0                                 2.50e-3         5       200
/END
"""
    model, log = _parse_starter(tmp_path, deck_str)
    assert len(log.errors) == 0
    assert 1 in model.detpoint_nodes
    dp1 = model.detpoint_nodes[1]
    assert isinstance(dp1, DetPointNode)
    assert dp1.id == 1
    assert dp1.tdet == pytest.approx(1.25e-3)
    assert dp1.mat_id == 4
    assert dp1.node_id == 100

    assert 2 in model.detpoint_nodes
    dp2 = model.detpoint_nodes[2]
    assert isinstance(dp2, DetPointNode)
    assert dp2.id == 2
    assert dp2.ishadow == 1
    assert dp2.iframe1 == 1
    assert dp2.iframe2 == 2
    assert dp2.r0_shadow == pytest.approx(5.0)
    assert dp2.tdet == pytest.approx(2.50e-3)
    assert dp2.mat_id == 5
    assert dp2.node_id == 200


def test_m200_detpoint_set(tmp_path: Path):
    """Test /DFS/DETPOINT/SET, /DETPOINT/SET, and /DETPOINT/GRNOD reading."""
    deck_str = """# RADIOSS STARTER
/BEGIN
DETPOINT SET Test
                 1                 1
/DFS/DETPOINT/SET/10
Detonation point set 10
#  Ishadow   Iframe1   Iframe2            R0_shadow             (blank)                TDET    mat_ID  grnod_ID
         0         0         0                  0.0                                 5.00e-4         2        42
/DETPOINT/GRNOD/20
Detonation point grnod 20
         0         0         0                  0.0                                 1.00e-3         3        84
/END
"""
    model, log = _parse_starter(tmp_path, deck_str)
    assert len(log.errors) == 0
    assert 10 in model.detpoint_sets
    ds10 = model.detpoint_sets[10]
    assert isinstance(ds10, DetPointSet)
    assert ds10.id == 10
    assert ds10.tdet == pytest.approx(5.00e-4)
    assert ds10.mat_id == 2
    assert ds10.grnod_id == 42

    assert 20 in model.detpoint_sets
    ds20 = model.detpoint_sets[20]
    assert isinstance(ds20, DetPointSet)
    assert ds20.id == 20
    assert ds20.tdet == pytest.approx(1.00e-3)
    assert ds20.mat_id == 3
    assert ds20.grnod_id == 84


def test_m200_engine_dtix_parith_th_title(tmp_path: Path):
    """Test /DTIX, /PARITH, and /TH/TITLE engine directives."""
    engine_str = """# RADIOSS ENGINE
/RUN/TEST/1
10.0
/DTIX
0.0001 0.0005
/PARITH/OFF
/TH/TITLE
/END
"""
    ec, log = _parse_engine(tmp_path, engine_str)
    assert len(log.errors) == 0
    assert ec.dtix_tini == pytest.approx(0.0001)
    assert ec.dtix_tmax == pytest.approx(0.0005)
    dtix_obj = ec.dtix
    assert isinstance(dtix_obj, DtixControl)
    assert dtix_obj.t_ini == pytest.approx(0.0001)
    assert dtix_obj.t_max == pytest.approx(0.0005)

    assert ec.parith == "OFF"
    assert ec.th_title is True


def test_m200_dynain_shell_options(tmp_path: Path):
    """Test /DYNAIN/SHELL/AUX/FULL, /DYNAIN/SHELL/STRES/FULL, /DYNAIN/SHELL/STRAIN/FULL."""
    deck_str = """# RADIOSS STARTER
/BEGIN
DYNAIN Shell Test
                 1                 1
/DYNAIN/SHELL/AUX/FULL
/DYNAIN/SHELL/STRES/FULL
/DYNAIN/SHELL/STRAIN/FULL
/END
"""
    model, log = _parse_starter(tmp_path, deck_str)
    assert len(log.errors) == 0
    assert len(model.dynain_shells) == 3
    options = [ds.option for ds in model.dynain_shells]
    assert "SHELL/AUX/FULL" in options
    assert "SHELL/STRES/FULL" in options
    assert "SHELL/STRAIN/FULL" in options
