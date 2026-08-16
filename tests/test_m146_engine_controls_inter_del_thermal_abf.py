"""Tests for Milestone M146: Engine Control Directives & Solver Controls Suite
(/INTER, /DEL, /DLI7, /KEREL, /DYREL, /THERMAL, /HEAT, /ABF, /INIVEL).
"""
from __future__ import annotations

from pathlib import Path
import pytest

from pyradioss.common.messages import MessageLog
from pyradioss.input.deck_reader import read_deck
from pyradioss.input.engine_keywords import parse_engine_deck
from pyradioss.model.model import EngineControls


def _parse_engine(tmp_path: Path, text: str) -> tuple[EngineControls, MessageLog]:
    p = tmp_path / "TEST_0001.rad"
    p.write_text(text, encoding="ascii")
    blocks = read_deck(str(p))
    log = MessageLog()
    ec = parse_engine_deck(blocks, log)
    return ec, log


def test_m146_engine_interfaces(tmp_path):
    deck = """/RUN/TEST/1
0.1
/INTER/ON
1 2 3
/INTER/OFF
4 5
/INTER
10 1 0.01 0.05
20 0 0.02 0.08
/END
"""
    ec, log = _parse_engine(tmp_path, deck)
    assert len(log.errors) == 0, f"Errors: {log.errors}"

    assert ec.inter_active[1] is True
    assert ec.inter_active[2] is True
    assert ec.inter_active[3] is True
    assert ec.inter_active[4] is False
    assert ec.inter_active[5] is False

    assert 10 in ec.inter_windows
    assert ec.inter_windows[10] == (1, pytest.approx(0.01), pytest.approx(0.05))
    assert 20 in ec.inter_windows
    assert ec.inter_windows[20] == (0, pytest.approx(0.02), pytest.approx(0.08))


def test_m146_engine_deletion_and_dli7(tmp_path):
    deck = """/RUN/TEST/1
0.1
/DEL/BRICK
101 102 103
/DEL/SHELL
201 202
/DEL/INTER
5 6
/DLI7
1.0 2.0 0.5
/END
"""
    ec, log = _parse_engine(tmp_path, deck)
    assert len(log.errors) == 0, f"Errors: {log.errors}"

    assert "BRICK" in ec.del_elements
    assert ec.del_elements["BRICK"] == [101, 102, 103]
    assert "SHELL" in ec.del_elements
    assert ec.del_elements["SHELL"] == [201, 202]
    assert "INTER" in ec.del_elements
    assert ec.del_elements["INTER"] == [5, 6]

    assert "param_0" in ec.dli7_controls
    assert ec.dli7_controls["param_0"] == pytest.approx(1.0)
    assert ec.dli7_controls["param_1"] == pytest.approx(2.0)
    assert ec.dli7_controls["param_2"] == pytest.approx(0.5)


def test_m146_engine_relaxation(tmp_path):
    deck = """/RUN/TEST/1
0.1
/KEREL
0.01 0.05
/DYREL
0.8 0.02
/END
"""
    ec, log = _parse_engine(tmp_path, deck)
    assert len(log.errors) == 0, f"Errors: {log.errors}"

    assert ec.kerel_active is True
    assert ec.kerel_tstart == pytest.approx(0.01)
    assert ec.kerel_tstop == pytest.approx(0.05)

    assert ec.dyrel_active is True
    assert ec.dyrel_beta == pytest.approx(0.8)
    assert ec.dyrel_period == pytest.approx(0.02)


def test_m146_engine_thermal_and_abf(tmp_path):
    deck = """/RUN/TEST/1
0.1
/THERMAL
2.5
/HEAT/DT
0.001 0.0005
/ABF/DT
0.01 0.02
/END
"""
    ec, log = _parse_engine(tmp_path, deck)
    assert len(log.errors) == 0, f"Errors: {log.errors}"

    assert ec.thermal_acc_fact == pytest.approx(2.5)
    assert ec.thermal_tstart == pytest.approx(0.001)
    assert ec.thermal_dt == pytest.approx(0.0005)

    assert ec.abf_dt == pytest.approx(0.01)
    assert ec.abf_dt_write == pytest.approx(0.02)


def test_m146_engine_inivel(tmp_path):
    deck = """/RUN/TEST/1
0.1
/INIVEL/TRA/X
10 50.0
20 75.0
/INIVEL/TRA/Z
10 -100.0
/END
"""
    ec, log = _parse_engine(tmp_path, deck)
    assert len(log.errors) == 0, f"Errors: {log.errors}"

    assert "TRA_X" in ec.inivel_engine
    assert ec.inivel_engine["TRA_X"][10] == pytest.approx(50.0)
    assert ec.inivel_engine["TRA_X"][20] == pytest.approx(75.0)

    assert "TRA_Z" in ec.inivel_engine
    assert ec.inivel_engine["TRA_Z"][10] == pytest.approx(-100.0)
