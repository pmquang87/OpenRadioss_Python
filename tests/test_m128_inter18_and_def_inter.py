"""Tests for Milestone M128: Fluid-Structure Coupling Interface (/INTER/TYPE18,
/DEF_INTER/TYPE18) and /DEF_INTER/TYPE8 Defaults.
"""
from __future__ import annotations

from pathlib import Path
import pytest

from pyradioss.common.messages import MessageLog
from pyradioss.input.deck_reader import read_deck
from pyradioss.input.starter_keywords import parse_starter_deck
from pyradioss.model.model import Model


def _parse_starter(tmp_path: Path, text: str):
    p = tmp_path / "TEST_0000.rad"
    p.write_text(text, encoding="ascii")
    blocks = read_deck(str(p))
    log = MessageLog()
    model = Model()
    parse_starter_deck(blocks, model, log)
    return model, log


def test_m128_inter_type18_free(tmp_path):
    deck = """/BEGIN
Test INTER TYPE18 Free Format
/SURF/1
Surf 1
1 2 3 4
/GRNOD/2
ALE Nodes
1 2 3
/GRBRIC/3
ALE Bricks
10 20
/INTER/TYPE18/1
Coupling Interface
2 1 3 1 2 1 1
0.5 10.0 0.05 0.0 1.0
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0, f"Errors: {log.errors}"
    assert len(model.interfaces) == 1
    inter = model.interfaces[0]
    assert inter.id == 1
    assert inter.type == 18
    assert inter.title == "Coupling Interface"
    assert inter.grnod_id == 2
    assert inter.surf_id == 1
    assert inter.grbric_id1 == 3
    assert inter.igap == 1
    assert inter.ibag == 2
    assert inter.idel == 1
    assert inter.stfac == pytest.approx(0.5)
    assert inter.gap == pytest.approx(0.05)
    assert inter.tstart == pytest.approx(0.0)
    assert inter.tstop == pytest.approx(1.0)


def test_m128_inter_type18_fixed(tmp_path):
    deck = """#---1---+----2---+----3---+----4---+----5---+----6---+----7---+----8---+----9---+---10---+
/BEGIN
Test Fixed Format
      2022         0
/SURF/1
Surf 1
         1         2         3         4
/GRNOD/2
ALE Nodes
         1         2         3
/GRBRIC/3
ALE Bricks
        10        20
/INTER/TYPE18/2
Fixed Coupling
         2         1         3                   1                   2         1                   1
                 0.5                10.0                0.05                 0.0                 1.0
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0, f"Errors: {log.errors}"
    assert len(model.interfaces) == 1
    inter = model.interfaces[0]
    assert inter.id == 2
    assert inter.type == 18
    assert inter.grnod_id == 2
    assert inter.surf_id == 1
    assert inter.grbric_id1 == 3
    assert inter.igap == 1
    assert inter.ibag == 2
    assert inter.idel == 1
    assert inter.stfac == pytest.approx(0.5)
    assert inter.gap == pytest.approx(0.05)


def test_m128_def_inter_18_and_8(tmp_path):
    deck = """/BEGIN
Test DEF_INTER TYPE18 and TYPE8
/DEF_INTER/TYPE18
1 5 2 1 1 1
/DEF_INTER/TYPE8
2
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0, f"Errors: {log.errors}"
    assert "TYPE18" in model.def_inter
    d18 = model.def_inter["TYPE18"]
    assert d18["istf"] == 1
    assert d18["multimp"] == 5
    assert d18["ibag"] == 2
    assert d18["idel18"] == 1
    assert d18["igap"] == 1
    assert d18["iauto"] == 1

    assert "TYPE8" in model.def_inter
    d8 = model.def_inter["TYPE8"]
    assert d8["iform1"] == 2
