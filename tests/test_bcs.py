"""Tests for /BCS fixed-format and free-format boundary condition parsing."""

import pytest
from pyradioss.common.messages import MessageLog
from pyradioss.input.deck_reader import read_deck
from pyradioss.input.starter_keywords import parse_starter_deck
from pyradioss.model.model import Model


FIXED_HEADER = (
    "/BEGIN\n"
    "BCS_TEST                                                           \n"
    "      2022         0\n"
    "                  kg                  mm                  ms\n"
    "                  kg                  mm                  ms\n"
)


def _parse_fixed(body, tmp_path):
    f = tmp_path / "bcs_test_0000.rad"
    f.write_text(FIXED_HEADER + body + "/END\n")
    model = Model()
    log = MessageLog()
    parse_starter_deck(read_deck(str(f)), model, log)
    return model, log


def test_bcs_fixed_single_digit_flag(tmp_path):
    """Verify that a deck with '     1             0         1' parses /BCS
    without error and sets the correct boundary condition (TZ constrained).
    Upstream Fortran reference: bcs.cfg (radioss51), hm_read_bcs.F:150-155.
    """
    body = (
        "/BCS/1\n"
        "bcs 1\n"
        "     1             0         1\n"
        "/GRNOD/NODE/1\n"
        "nodeset 1\n"
        "         1\n"
        "/NODE\n"
        "         1                 0.0                 0.0                 0.0\n"
    )
    model, log = _parse_fixed(body, tmp_path)
    assert not log.errors, [err.msg for err in log.errors]
    assert len(model.bcs) == 1
    bc = model.bcs[0]
    assert bc.id == 1
    assert bc.grnod_id == 1
    assert bc.skew_id == 0
    assert list(bc.fix_tra) == [False, False, True]
    assert list(bc.fix_rot) == [False, False, False]


def test_bcs_fixed_standard_two_tokens(tmp_path):
    """Verify that a deck with '   100 011                   2' parses /BCS."""
    body = (
        "/BCS/2\n"
        "bcs 2\n"
        "   100 011                   2\n"
        "/GRNOD/NODE/2\n"
        "nodeset 2\n"
        "         1\n"
        "/NODE\n"
        "         1                 0.0                 0.0                 0.0\n"
    )
    model, log = _parse_fixed(body, tmp_path)
    assert not log.errors, [err.msg for err in log.errors]
    assert len(model.bcs) == 1
    bc = model.bcs[0]
    assert bc.id == 2
    assert bc.grnod_id == 2
    assert bc.skew_id == 0
    assert list(bc.fix_tra) == [True, False, False]
    assert list(bc.fix_rot) == [False, True, True]


def test_bcs_fixed_six_digit_string(tmp_path):
    """Verify that a deck with 6 continuous digits parses /BCS."""
    body = (
        "/BCS/3\n"
        "bcs 3\n"
        "   111000         0         5\n"
        "/GRNOD/NODE/5\n"
        "nodeset 5\n"
        "         1\n"
        "/NODE\n"
        "         1                 0.0                 0.0                 0.0\n"
    )
    model, log = _parse_fixed(body, tmp_path)
    assert not log.errors, [err.msg for err in log.errors]
    assert len(model.bcs) == 1
    bc = model.bcs[0]
    assert bc.id == 3
    assert bc.grnod_id == 5
    assert bc.skew_id == 0
    assert list(bc.fix_tra) == [True, True, True]
    assert list(bc.fix_rot) == [False, False, False]
