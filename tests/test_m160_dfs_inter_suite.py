"""Tests for Milestone M160: Node-Based Detonation Ignitions, Hertzian & Extended Lagrange Multiplier Interfaces Suite."""
from __future__ import annotations
from pathlib import Path
import pytest

from pyradioss.common.messages import MessageLog
from pyradioss.input.deck_reader import read_deck
from pyradioss.input.starter_keywords import parse_starter_deck
from pyradioss.model.model import Model


def _parse_deck(tmp_path: Path, text: str):
    p = tmp_path / "TEST_0000.rad"
    p.write_text(text, encoding="ascii")
    blocks = read_deck(str(p))
    log = MessageLog()
    model = Model()
    parse_starter_deck(blocks, model, log)
    assert not log.errors, f"Parse errors: {log.errors}"
    return model, log


def test_dfs_detpoint_node(tmp_path: Path):
    """Test /DFS/DETPOINT/NODE in fixed and free formats."""
    deck_fixed = (
        "# RADIOSS STARTER\n"
        "/BEGIN\n"
        "DETPOINT_NODE_TEST\n"
        "                  0.                 0\n"
        "/DFS/DETPOINT/NODE/1\n"
        "                                                             0.005         2       101\n"
        "/DFS/DETPOINT/NODE/2\n"
        "0.010 3 102\n"
        "/END\n"
    )
    m, _ = _parse_deck(tmp_path, deck_fixed)
    assert len(m.det_points) == 2
    dp1 = next(d for d in m.det_points if d.id == 1)
    assert dp1.tdet == pytest.approx(0.005)
    assert dp1.mat_id == 2
    assert dp1.node_id == 101

    dp2 = next(d for d in m.det_points if d.id == 2)
    assert dp2.tdet == pytest.approx(0.010)
    assert dp2.mat_id == 3
    assert dp2.node_id == 102


def test_dfs_detplan_node(tmp_path: Path):
    """Test /DFS/DETPLAN/NODE in fixed and free formats."""
    deck_fixed = (
        "# RADIOSS STARTER\n"
        "/BEGIN\n"
        "DETPLAN_NODE_TEST\n"
        "                  0.                 0\n"
        "/DFS/DETPLAN/NODE/1\n"
        "                                                             0.002         1       201\n"
        "                                                                                           202\n"
        "/DFS/DETPLAN/NODE/2\n"
        "0.004 2 203\n"
        "204\n"
        "/END\n"
    )
    m, _ = _parse_deck(tmp_path, deck_fixed)
    assert len(m.det_planes) == 2
    dp1 = next(d for d in m.det_planes if d.id == 1)
    assert dp1.tdet == pytest.approx(0.002)
    assert dp1.mat_id == 1
    assert dp1.p_id == 201
    assert dp1.n_id == 202

    dp2 = next(d for d in m.det_planes if d.id == 2)
    assert dp2.tdet == pytest.approx(0.004)
    assert dp2.mat_id == 2
    assert dp2.p_id == 203
    assert dp2.n_id == 204


def test_dfs_detline_node(tmp_path: Path):
    """Test /DFS/DETLINE/NODE in fixed and free formats."""
    deck_fixed = (
        "# RADIOSS STARTER\n"
        "/BEGIN\n"
        "DETLINE_NODE_TEST\n"
        "                  0.                 0\n"
        "/DFS/DETLINE/NODE/1\n"
        "                                                                                           301\n"
        "                                                                                           302\n"
        "               0.001         1\n"
        "/DFS/DETLINE/NODE/2\n"
        "303\n"
        "304\n"
        "0.002 2\n"
        "/END\n"
    )
    m, _ = _parse_deck(tmp_path, deck_fixed)
    assert 1 in m.det_lines
    dl1 = m.det_lines[1]
    assert dl1.node1 == 301
    assert dl1.node2 == 302
    assert dl1.t0 == pytest.approx(0.001)
    assert dl1.mat_id == 1

    assert 2 in m.det_lines
    dl2 = m.det_lines[2]
    assert dl2.node1 == 303
    assert dl2.node2 == 304
    assert dl2.t0 == pytest.approx(0.002)
    assert dl2.mat_id == 2


def test_inter_hertz_type17(tmp_path: Path):
    """Test /INTER/HERTZ/TYPE17 in fixed and free formats."""
    deck_fixed = (
        "# RADIOSS STARTER\n"
        "/BEGIN\n"
        "INTER_HERTZ_TEST\n"
        "                  0.                 0\n"
        "/INTER/HERTZ/TYPE17/1\n"
        "Hertz Interface Fixed\n"
        "        10        20\n"
        "                 0.1\n"
        "/INTER/HERTZ/TYPE17/2\n"
        "Hertz Interface Free\n"
        "30 40\n"
        "0.2\n"
        "/END\n"
    )
    m, _ = _parse_deck(tmp_path, deck_fixed)
    assert len(m.interfaces) == 2
    i1 = next(i for i in m.interfaces if i.id == 1)
    assert i1.type == 17
    assert i1.grbric_id1 == 10
    assert i1.grbric_id2 == 20
    assert i1.fric == pytest.approx(0.1)
    assert i1.hertz is True

    i2 = next(i for i in m.interfaces if i.id == 2)
    assert i2.type == 17
    assert i2.grbric_id1 == 30
    assert i2.grbric_id2 == 40
    assert i2.fric == pytest.approx(0.2)
    assert i2.hertz is True


def test_inter_lagmul_suite(tmp_path: Path):
    """Test /INTER/LAGMUL/TYPE16, /INTER/LAGMUL/TYPE17, /INTER/LAGMUL/TYPE2."""
    deck_fixed = (
        "# RADIOSS STARTER\n"
        "/BEGIN\n"
        "INTER_LAGMUL_TEST\n"
        "                  0.                 0\n"
        "/INTER/LAGMUL/TYPE16/1\n"
        "Lagmul Type 16\n"
        "         5        15\n"
        "                   1\n"
        "/INTER/LAGMUL/TYPE17/2\n"
        "Lagmul Type 17\n"
        "        25        35\n"
        "                   1\n"
        "/INTER/LAGMUL/TYPE2/3\n"
        "Lagmul Type 2\n"
        "         1         2                                        1                 0.5\n"
        "/END\n"
    )
    m, _ = _parse_deck(tmp_path, deck_fixed)
    assert len(m.interfaces) == 3
    i1 = next(i for i in m.interfaces if i.id == 1)
    assert i1.type == 16
    assert i1.grnod_id == 5
    assert i1.grbric_id1 == 15
    assert i1.itied == 1
    assert i1.lagmul is True

    i2 = next(i for i in m.interfaces if i.id == 2)
    assert i2.type == 17
    assert i2.grbric_id1 == 25
    assert i2.grbric_id2 == 35
    assert i2.itied == 1
    assert i2.lagmul is True

    i3 = next(i for i in m.interfaces if i.id == 3)
    assert i3.type == 2
    assert i3.grnod_id == 1
    assert i3.surf_id == 2
    assert i3.dsearch == pytest.approx(0.5)
    assert i3.lagmul is True
