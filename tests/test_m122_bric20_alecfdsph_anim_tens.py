"""Tests for Milestone M122: 20-node quadratic hexahedral solids (/BRIC20, /HEXA20),
/PROP/TYPE23 / /PROP/HEXA20, /GRBR20 / /GRHEX20, /ALECFDSPH, and /ANIM/*/TENS.
"""
from __future__ import annotations

from pathlib import Path
import pytest
import numpy as np

from pyradioss.common.messages import MessageLog
from pyradioss.input.deck_reader import read_deck
from pyradioss.input.starter_keywords import parse_starter_deck
from pyradioss.input.engine_keywords import parse_engine_deck
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


def test_m122_bric20_and_hexa20_free_and_fixed(tmp_path):
    # Test free format BRIC20 and fixed format HEXA20
    nodes_str = "\n".join(f"{i} {float(i)} 0.0 0.0" for i in range(1, 25))
    deck = f"""/BEGIN
Test BRIC20 and HEXA20
/NODE
{nodes_str}
/BRIC20/1
101 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20
/HEXA20/2
       102         1         2         3         4         5         6         7         8         9        10
        11        12        13        14        15        16        17        18        19        20
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0, f"Errors: {log.errors}"
    assert len(model.raw_elems["BRIC20"]) == 2
    
    e1 = model.raw_elems["BRIC20"][0]
    assert e1[0] == 101
    assert e1[1] == 1
    assert e1[2] == list(range(1, 21))

    e2 = model.raw_elems["BRIC20"][1]
    assert e2[0] == 102
    assert e2[1] == 2
    assert e2[2] == list(range(1, 21))


def test_m122_prop_type23_and_bric20_assembly(tmp_path):
    nodes_str = "\n".join(f"{i} {float(i)} 0.0 0.0" for i in range(1, 21))
    deck = f"""/BEGIN
Test BRIC20 Assembly
/NODE
{nodes_str}
/MAT/LAW1/1
Steel
7.85e-9
210000.0 0.3
/PROP/TYPE23/1
Hexa20 Property
1 1 1.1 0.05 0.1
/PART/1
Part 1
1 1
/BRIC20/1
1 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0, f"Errors: {log.errors}"
    assert 1 in model.properties
    assert model.properties[1].type == 23

    build_element_groups(model, log)
    assert len(log.errors) == 0, f"Errors after assembly: {log.errors}"
    assert model.bric20s is not None
    assert model.bric20s.n == 1
    assert model.bric20s.ids[0] == 1
    assert model.bric20s.conn.shape == (1, 20)
    assert np.all(model.bric20s.conn[0] == np.arange(20))


def test_m122_grbr20_and_grhex20(tmp_path):
    deck = """/BEGIN
Test Groups
/GRBR20/BRIC/1
Brick Group 1
101 102 103
/GRHEX20/BRIC/2
Brick Group 2
201 202
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0, f"Errors: {log.errors}"
    assert "BRIC" in model.egroups
    assert 1 in model.egroups["BRIC"]
    assert model.egroups["BRIC"][1].elem_ids == [101, 102, 103]
    assert 2 in model.egroups["BRIC"]
    assert model.egroups["BRIC"][2].elem_ids == [201, 202]


def test_m122_alecfdsph_starter_and_engine(tmp_path):
    deck = """/BEGIN
Test ALECFDSPH
/ALECFDSPH
Coupled Interaction
1 2 0.001 0.05 1.5 2.0
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0, f"Errors: {log.errors}"
    assert model.alecfdsph is not None
    assert model.alecfdsph.title == "Coupled Interaction"
    assert model.alecfdsph.icfd == 1
    assert model.alecfdsph.isph == 2
    assert model.alecfdsph.tstart == 0.001
    assert model.alecfdsph.tstop == 0.05
    assert model.alecfdsph.fscale_c == 1.5
    assert model.alecfdsph.fscale_s == 2.0


def test_m122_engine_anim_tens_and_alecfdsph(tmp_path):
    engine_deck = """\
# Engine Deck M122
/RUN/TestRun/1
10.0
/ALECFDSPH
/ANIM/DT
0.0 0.1
/ANIM/ELEM/TENS
/ANIM/BRICK/TENS
/ANIM/SHELL/TENS
/ANIM/VECT/VEL
/ANIM/VECT/DIS
"""
    p = tmp_path / "TEST_0001.rad"
    p.write_text(engine_deck, encoding="ascii")
    blocks = read_deck(str(p))
    log = MessageLog()
    ec = parse_engine_deck(blocks, log)
    assert len(log.errors) == 0, f"Errors: {log.errors}"

    assert "ELEM/TENS" in ec.anim_tens
    assert "BRICK/TENS" in ec.anim_tens
    assert "SHELL/TENS" in ec.anim_tens
    assert "VEL" in ec.anim_vect
    assert "DIS" in ec.anim_vect
    assert ec.anim_dt == 0.1
