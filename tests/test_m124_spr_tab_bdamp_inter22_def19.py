"""Tests for Milestone M124: Advanced spring properties (/PROP/TYPE26 SPR_TAB, /PROP/TYPE27 SPR_BDAMP),
/INTER/TYPE22, and /DEF_INTER/TYPE19.
"""
from __future__ import annotations

from pathlib import Path
import pytest

from pyradioss.common.messages import MessageLog
from pyradioss.input.deck_reader import read_deck
from pyradioss.input.starter_keywords import parse_starter_deck
from pyradioss.model.model import Model
from pyradioss.starter.initialization import build_element_groups


def _parse_starter(tmp_path: Path, text: str):
    p = tmp_path / "TEST_0000.rad"
    p.write_text(text, encoding="ascii")
    blocks = read_deck(str(p))
    log = MessageLog()
    model = Model()
    parse_starter_deck(blocks, model, log)
    return model, log


def test_m124_prop_spr_tab(tmp_path):
    deck = """/BEGIN
Test SPR_TAB
/PROP/TYPE26/1
Tabular Spring
0.05 10 1 0 2 3
1.5 5000.0 0.12 0.05
/PROP/SPR_TAB/2
Named Tabular Spring
0.1 20 0 1 1 1
2.0 8000.0 0.2 0.01
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0, f"Errors: {log.errors}"
    assert 1 in model.properties
    assert 2 in model.properties

    p1 = model.properties[1]
    assert p1.type == 26
    assert p1.params["mass"] == 0.05
    assert p1.params["sens_id"] == 10
    assert p1.params["isflag"] == 1
    assert p1.params["ileng"] == 0
    assert p1.params["nfunc"] == 2
    assert p1.params["nraten"] == 3
    assert p1.params["scale"] == 1.5
    assert p1.params["stiff0"] == 5000.0
    assert p1.params["dmax"] == 0.12
    assert p1.params["alpha1"] == 0.05
    assert p1.params["k"] == 5000.0

    p2 = model.properties[2]
    assert p2.type == 26
    assert p2.params["mass"] == 0.1
    assert p2.params["stiff0"] == 8000.0


def test_m124_prop_spr_bdamp(tmp_path):
    deck = """/BEGIN
Test SPR_BDAMP
/PROP/TYPE27/10
Bilinear Damped Spring
0.02 5 0 0 1 1
1000.0 25.0 1.2 -0.05 0.15
0.001 1 500.0
101 102 1.0 1.0 1.0 1.0
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0, f"Errors: {log.errors}"
    assert 10 in model.properties

    p = model.properties[10]
    assert p.type == 27
    assert p.params["mass"] == 0.02
    assert p.params["sens_id"] == 5
    assert p.params["isflag"] == 0
    assert p.params["ileng"] == 0
    assert p.params["itens"] == 1
    assert p.params["ifail"] == 1
    assert p.params["k"] == 1000.0
    assert p.params["c"] == 25.0
    assert p.params["n"] == 1.2
    assert p.params["delta_min"] == -0.05
    assert p.params["delta_max"] == 0.15
    assert p.params["gap"] == 0.001
    assert p.params["fsmooth"] == 1
    assert p.params["fcut"] == 500.0
    assert p.params["fct1"] == 101
    assert p.params["fct2"] == 102


def test_m124_spring_assembly_with_new_props(tmp_path):
    deck = """/BEGIN
Test Spring Assembly
/NODE
1 0.0 0.0 0.0
2 1.0 0.0 0.0
3 0.0 1.0 0.0
4 1.0 1.0 0.0
/PROP/TYPE26/1
Spring Tab Prop
0.05 0 0 0 1 1
1.0 2500.0 0.2 0.0
/PROP/TYPE27/2
Spring BDamp Prop
0.05 0 0 0 0 0
3500.0 10.0 1.0 0.0 0.0
/PART/1
Part 1
1 0
/PART/2
Part 2
2 0
/SPRING/1
101 1 1 2
102 2 3 4
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0, f"Errors: {log.errors}"
    assert 1 in model.properties
    assert 2 in model.properties

    build_element_groups(model, log)
    assert len(log.errors) == 0, f"Errors after assembly: {log.errors}"
    assert model.springs is not None
    assert model.springs.n == 2


def test_m124_inter_type22(tmp_path):
    deck = """/BEGIN
Test INTER TYPE22
/INTER/TYPE22/1
Fluid Structure Interface
10 20
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0, f"Errors: {log.errors}"
    assert len(model.interfaces) == 1
    inter = model.interfaces[0]
    assert inter.id == 1
    assert inter.type == 22
    assert inter.grbric_id1 == 10
    assert inter.surf_id == 20


def test_m124_def_inter_type19(tmp_path):
    deck = """/BEGIN
Test DEF INTER TYPE19
/DEF_INTER/TYPE19
1 2 3 4 5 6 7 8
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0, f"Errors: {log.errors}"
    assert "TYPE19" in model.def_inter
    d19 = model.def_inter["TYPE19"]
    assert d19["istf"] == 1
    assert d19["igap"] == 2
    assert d19["iedge"] == 3
    assert d19["ibag"] == 4
    assert d19["idel"] == 5
    assert d19["icurv"] == 6
    assert d19["inactiv"] == 7
    assert d19["iform"] == 8
