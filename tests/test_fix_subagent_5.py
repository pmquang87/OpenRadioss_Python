"""Tests for Fix Subagent 5 (Input & Output fixes)."""

import os
from unittest.mock import MagicMock
import numpy as np
import pytest

from pyradioss.input import parse_fortran_float
from pyradioss.input.card_layouts import fmt_float
from pyradioss.input.deck_reader import Card, KeywordBlock, _to_float, _to_int
from pyradioss.input.deck_writer import StarterDeck, _conv_prop, _convert_block
from pyradioss.model.model import ElementGroup, Model
from pyradioss.output.anim_vtk import write_anim_state
from pyradioss.output.time_history import TimeHistory


# ----------------------------------------------------------------------------
# 1. deck_reader: parse_fortran_float & _to_float
# ----------------------------------------------------------------------------
def test_parse_fortran_float():
    assert parse_fortran_float("1.5D-3") == pytest.approx(0.0015)
    assert parse_fortran_float("1.5d+03") == pytest.approx(1500.0)
    assert parse_fortran_float("1.5-3") == pytest.approx(0.0015)
    assert parse_fortran_float("1.5+03") == pytest.approx(1500.0)
    assert parse_fortran_float("-1.5E-3") == pytest.approx(-0.0015)
    assert parse_fortran_float("+2.5-4") == pytest.approx(0.00025)
    assert parse_fortran_float("-3.5+02") == pytest.approx(-350.0)
    assert parse_fortran_float("1.234, ") == pytest.approx(1.234)
    assert parse_fortran_float("1.234-05,") == pytest.approx(1.234e-5)
    assert parse_fortran_float("0.0") == pytest.approx(0.0)
    assert parse_fortran_float("0") == pytest.approx(0.0)

    # Test _to_float and _to_int integration
    assert _to_float("2.5-3") == pytest.approx(0.0025)
    assert _to_int("500-1") == 50  # 50.0 truncated to int


# ----------------------------------------------------------------------------
# 2. card_layouts: fmt_float bounding
# ----------------------------------------------------------------------------
def test_fmt_float_width_bounding():
    test_vals = [
        -1.23456789e-105,
        1.23456789e105,
        123456789.12345,
        -123456789.12345,
        0.123456789,
        -0.123456789,
        0.0,
        1e30,
        -1e30,
    ]
    for w in (6, 8, 10, 12, 14, 20):
        for v in test_vals:
            s = fmt_float(v, width=w)
            assert len(s) == w, f"fmt_float({v}, width={w}) produced '{s}' of len {len(s)} != {w}"


# ----------------------------------------------------------------------------
# 3. deck_writer: EOS conversion, rwall_paral, _conv_prop
# ----------------------------------------------------------------------------
def test_deck_writer_eos_convert():
    d = StarterDeck("EOS_TEST")
    # Titled EOS block
    b = KeywordBlock("EOS/STIFF-GAS", ["EOS", "STIFF-GAS", "1"], 1, [
        Card("My Stiff Gas EOS"),
        Card("1.0 2.0 3.0")
    ])
    _convert_block(d, b, {})
    rendered = d.render()
    assert "/EOS/STIFF-GAS/1" in rendered
    assert "My Stiff Gas EOS" in rendered
    assert "1.0" in rendered


def test_deck_writer_rwall_paral():
    d = StarterDeck("RWALL_TEST")
    d.rwall_paral(5, "Paral Wall Title", [Card("1 2 3 4"), Card("0.1 0.2")])
    rendered = d.render()
    assert "/RWALL/PARAL/5" in rendered
    assert "Paral Wall Title" in rendered
    assert "1 2 3 4" in rendered


def test_deck_writer_conv_prop_padding():
    d = StarterDeck("PROP_TEST")
    # Empty card[0].ints() should not crash with IndexError
    b = KeywordBlock("PROP/SHELL", ["PROP", "SHELL", "1"], 1, [Card("title"), Card("")])
    _conv_prop(d, b)
    rendered = d.render()
    assert "/PROP/SHELL/1" in rendered


# ----------------------------------------------------------------------------
# 4. deck_reader: fixed_cards shallow copy
# ----------------------------------------------------------------------------
def test_fixed_cards_copy():
    c1 = Card("10.0")
    c2 = Card("20.0")
    b = KeywordBlock("TEST", ["TEST"], 1, [c1, c2])
    fc = b.fixed_cards()
    fc.pop()
    assert len(b.cards) == 2, "Mutating list returned by fixed_cards() mutated block.cards in place"


# ----------------------------------------------------------------------------
# 5. time_history: RBODY angular vel & norms, spring DL, beam/truss elements
# ----------------------------------------------------------------------------
def test_time_history_rbody_and_spring_dl(tmp_path):
    model = Model()
    th = TimeHistory(str(tmp_path / "th.csv"), model, MagicMock())

    # Mock rigid body
    rb = MagicMock()
    rb.x_cg = np.array([1.0, 2.0, 3.0])
    rb.x_cg0 = np.array([0.0, 0.0, 0.0])
    rb.v_cg = np.array([3.0, 4.0, 0.0])
    rb.w = np.array([0.0, 6.0, 8.0])
    rb.f_res = np.array([3.0, 4.0, 0.0])
    rb.m_res = np.array([1.0, 2.0, 2.0])
    model.rigid_bodies = {1: rb}

    # RBODY displacements
    assert th._other_value("RBODY", 1, "DX") == pytest.approx(1.0)
    assert th._other_value("RBODY", 1, "DY") == pytest.approx(2.0)
    assert th._other_value("RBODY", 1, "DZ") == pytest.approx(3.0)
    assert th._other_value("RBODY", 1, "D") == pytest.approx(np.sqrt(1 + 4 + 9))

    # RBODY translational velocity
    assert th._other_value("RBODY", 1, "VX") == pytest.approx(3.0)
    assert th._other_value("RBODY", 1, "VY") == pytest.approx(4.0)
    assert th._other_value("RBODY", 1, "V") == pytest.approx(5.0)

    # RBODY angular velocity
    assert th._other_value("RBODY", 1, "WX") == pytest.approx(0.0)
    assert th._other_value("RBODY", 1, "WY") == pytest.approx(6.0)
    assert th._other_value("RBODY", 1, "WZ") == pytest.approx(8.0)
    assert th._other_value("RBODY", 1, "W") == pytest.approx(10.0)
    assert th._other_value("RBODY", 1, "ROTX") == pytest.approx(0.0)
    assert th._other_value("RBODY", 1, "ROTY") == pytest.approx(6.0)
    assert th._other_value("RBODY", 1, "ROT") == pytest.approx(10.0)

    # RBODY resultant force & moment
    assert th._other_value("RBODY", 1, "FX") == pytest.approx(3.0)
    assert th._other_value("RBODY", 1, "FY") == pytest.approx(4.0)
    assert th._other_value("RBODY", 1, "F") == pytest.approx(5.0)
    assert th._other_value("RBODY", 1, "MX") == pytest.approx(1.0)
    assert th._other_value("RBODY", 1, "MY") == pytest.approx(2.0)
    assert th._other_value("RBODY", 1, "M") == pytest.approx(3.0)

    # Mock spring
    grp_sp = ElementGroup(ids=np.array([10]), conn=np.array([[0, 1]]), part=np.array([0]))
    grp_sp.state["disp"] = np.array([0.75])
    model.springs = grp_sp

    assert th._other_value("SPRING", 10, "DL") == pytest.approx(0.75)


def test_time_history_truss_and_beam(tmp_path):
    model = Model()
    th = TimeHistory(str(tmp_path / "th_elem.csv"), model, MagicMock())

    # Truss element
    grp_tr = ElementGroup(ids=np.array([100]), conn=np.array([[0, 1]]), part=np.array([0]))
    grp_tr.state["sig"] = np.array([250.0])
    grp_tr.state["eint"] = np.array([12.5])
    model.trusses = grp_tr

    # Test kind aliases: TRUSS, TRUS
    assert th._other_value("TRUSS", 100, "VM") == pytest.approx(250.0)
    assert th._other_value("TRUS", 100, "IE") == pytest.approx(12.5)

    # Beam element
    grp_bm = ElementGroup(ids=np.array([200]), conn=np.array([[0, 1]]), part=np.array([0]))
    grp_bm.state["eint"] = np.array([45.0])
    mock_prop = MagicMock()
    mock_prop.params = {"area": 2.0}
    grp_bm.state["slices"] = [(slice(0, 1), None, mock_prop)]
    grp_bm.state["fres"] = np.array([[100.0, 0.0, 0.0]])
    model.beams = grp_bm

    assert th._other_value("BEAM", 200, "IE") == pytest.approx(45.0)
    assert th._other_value("BEAM", 200, "VM") == pytest.approx(50.0)  # 100 / 2.0


# ----------------------------------------------------------------------------
# 6. anim_vtk: /ANIM/VECT/ACC
# ----------------------------------------------------------------------------
def test_anim_vtk_acc(tmp_path):
    model = Model()
    model.node_ids = np.array([1, 2])
    model.x = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    model.x0 = model.x.copy()
    model.v = np.array([[0.1, 0.0, 0.0], [0.2, 0.0, 0.0]])
    model.a = np.array([[9.81, 0.0, -1.0], [9.81, 0.0, -2.0]])
    model.node_ids = np.array([1, 2])

    vtk_path = str(tmp_path / "testA001.vtk")
    write_anim_state(vtk_path, model, t=0.01, vect=("DIS", "VEL", "ACC"), elem=(), cycle=5)

    with open(vtk_path, "r", encoding="utf-8") as f:
        content = f.read()

    assert "VECTORS DISPLACEMENT double" in content
    assert "VECTORS VELOCITY double" in content
    assert "VECTORS ACCELERATION double" in content
    assert "9.810000000E+00" in content
