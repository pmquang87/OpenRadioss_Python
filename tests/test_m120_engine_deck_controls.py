"""Tests for Milestone M120: Engine Deck Control Keywords (/DEBUG, /BCS, /RBODY,
/ALE, /NOIS, /H3D, /FLOW, /UPWIND, /EIG/OFF).
"""
from __future__ import annotations

from pathlib import Path
import pytest

from pyradioss.common.messages import MessageLog
from pyradioss.input.deck_reader import read_deck
from pyradioss.input.engine_keywords import parse_engine_deck


def _parse_engine(tmp_path: Path, text: str):
    p = tmp_path / "TEST_0001.rad"
    p.write_text(text, encoding="ascii")
    blocks = read_deck(str(p))
    log = MessageLog()
    ec = parse_engine_deck(blocks, log)
    return ec, log


def test_m120_debug_controls(tmp_path):
    engine_deck = """\
# Engine file
/RUN/TestRun/1
10.0
/DEBUG/CORE/2
/DEBUG/ACC
0.0 5
/DEBUG/NAN
/DEBUG/MEM
/DEBUG/INTER/3
"""
    ec, log = _parse_engine(tmp_path, engine_deck)
    assert len(log.errors) == 0, f"Errors: {log.errors}"
    assert ec.debug_flags["CORE"] == 2
    assert ec.debug_flags["ACC"] == 1
    assert ec.debug_acc_start == 0.0
    assert ec.debug_acc_freq == 5
    assert ec.debug_flags["NAN"] == 1
    assert ec.debug_flags["MEM"] == 1
    assert ec.debug_flags["INTER"] == 3


def test_m120_bcs_rbody_ale_on_off(tmp_path):
    engine_deck = """\
/RUN/TestRun/1
5.0
/BCS/ON
1 2 3
/BCS/OFF
4 5
/RBODY/ON
10 20
/RBODY/OFF
30
/ALE/ON
100
/ALE/OFF
200 300
"""
    ec, log = _parse_engine(tmp_path, engine_deck)
    assert len(log.errors) == 0, f"Errors: {log.errors}"
    assert ec.bcs_active[1] is True
    assert ec.bcs_active[2] is True
    assert ec.bcs_active[3] is True
    assert ec.bcs_active[4] is False
    assert ec.bcs_active[5] is False

    assert ec.rbody_active[10] is True
    assert ec.rbody_active[20] is True
    assert ec.rbody_active[30] is False

    assert ec.ale_active[100] is True
    assert ec.ale_active[200] is False
    assert ec.ale_active[300] is False


def test_m120_noise_controls(tmp_path):
    engine_deck = """\
/RUN/TestRun/1
1.0
/NOIS/DT
0.1 0.01
/NOIS/VEL
/NOIS/ACC
"""
    ec, log = _parse_engine(tmp_path, engine_deck)
    assert len(log.errors) == 0, f"Errors: {log.errors}"
    assert ec.noise_tstart == 0.1
    assert ec.noise_dt == 0.01
    assert ec.noise_flags.get("VEL") is True
    assert ec.noise_flags.get("ACC") is True


def test_m120_h3d_controls(tmp_path):
    engine_deck = """\
/RUN/TestRun/1
2.0
/H3D/DT
0.0 0.05
/H3D/NODA/DISP
/H3D/ELEM/VONM
/H3D/SHELL/TENS/STRESS
"""
    ec, log = _parse_engine(tmp_path, engine_deck)
    assert len(log.errors) == 0, f"Errors: {log.errors}"
    assert ec.h3d_dt == 0.05
    assert "NODA/DISP" in ec.h3d_requests
    assert "ELEM/VONM" in ec.h3d_requests
    assert "SHELL/TENS/STRESS" in ec.h3d_requests


def test_m120_flow_and_upwind(tmp_path):
    engine_deck = """\
/RUN/TestRun/1
0.5
/FLOW/DT
0.0 0.02
/UPWIND
0.8 0.9 0.75
"""
    ec, log = _parse_engine(tmp_path, engine_deck)
    assert len(log.errors) == 0, f"Errors: {log.errors}"
    assert ec.flow_dt == 0.02
    assert ec.upwind_active is True
    assert ec.upwind_mom == 0.8
    assert ec.upwind_mass_eng == 0.9
    assert ec.upwind_wet_surf == 0.75


def test_m120_eig_off(tmp_path):
    engine_deck = """\
/RUN/TestRun/1
1.0
/EIG/OFF
1 2 3
4 5
"""
    ec, log = _parse_engine(tmp_path, engine_deck)
    assert len(log.errors) == 0, f"Errors: {log.errors}"
    assert ec.eig_off == [1, 2, 3, 4, 5]
