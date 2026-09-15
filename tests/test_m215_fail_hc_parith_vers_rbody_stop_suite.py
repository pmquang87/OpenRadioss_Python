"""Tests for Milestone M215:
- Hosford-Coulomb Failure Criterion: /FAIL/HC, /FAIL/HOSFORD_COULOMB
- Engine Parallel Arithmetic Directives: /PARITH, /ENG/PARITH, /PARITH/ON, /PARITH/OFF
- Engine Version Directive: /VERS, /ENG/VERS
- Rigid Body Stop Directives: /RBODY/STOP, /ENG/RBODY/STOP
"""

from pathlib import Path
import pytest
from pyradioss.common.messages import MessageLog
from pyradioss.input.deck_reader import read_deck
from pyradioss.input.starter_keywords import parse_starter_deck
from pyradioss.model.model import Model


def _parse_starter(tmp_path: Path, text: str) -> tuple[Model, MessageLog]:
    p = tmp_path / "TEST_0000.rad"
    p.write_text(text.strip() + "\n", encoding="ascii")
    blocks = read_deck(str(p))
    model = Model()
    log = MessageLog()
    parse_starter_deck(blocks, model, log)
    return model, log


def test_m215_fail_hc(tmp_path: Path):
    deck = """# RADIOSS STARTER DECK
/BEGIN
Hosford-Coulomb Failure Criterion Test
1 1
/FAIL/HC/42
Hosford Coulomb Material Failure
                0.12                0.34                0.56                 1.8         2                0.95
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert 42 in model.fail_hcs
    hc = model.fail_hcs[42]
    assert pytest.approx(hc.a) == 0.12
    assert pytest.approx(hc.b) == 0.34
    assert pytest.approx(hc.c) == 0.56
    assert pytest.approx(hc.n_hc) == 1.8
    assert hc.ifail_sh == 2
    assert pytest.approx(hc.d_max) == 0.95


def test_m215_eng_parith(tmp_path: Path):
    deck = """# RADIOSS STARTER DECK
/BEGIN
Parallel Arithmetic Test
1 1
/PARITH/ON
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert model.parith_enabled is True

    deck_off = """# RADIOSS STARTER DECK
/BEGIN
Parallel Arithmetic Off Test
1 1
/ENG/PARITH/OFF
/END
"""
    model_off, log_off = _parse_starter(tmp_path, deck_off)
    assert model_off.parith_enabled is False


def test_m215_eng_vers(tmp_path: Path):
    deck = """# RADIOSS STARTER DECK
/BEGIN
Engine Version Test
1 1
/ENG/VERS
              2024.1
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert pytest.approx(model.eng_version) == 2024.1


def test_m215_rbody_stop(tmp_path: Path):
    deck = """# RADIOSS STARTER DECK
/BEGIN
Rigid Body Stop Action Test
1 1
/RBODY/STOP/101
Rigid Body Stop 1
        12         5         1
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert 101 in model.rbody_stops
    rbs = model.rbody_stops[101]
    assert rbs.rbody_id == 12
    assert rbs.sens_id == 5
    assert rbs.istop_opt == 1
