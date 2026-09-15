"""Tests for Milestone M153: Checker false-positive fixes.

- ADMAS mass_type-aware cross-reference (types 2/3/4/6/7 skip node group check)
- 'model has no elements' downgraded from error to warning
"""
from __future__ import annotations

from pathlib import Path
import pytest

from pyradioss.common.messages import MessageLog
from pyradioss.input.deck_reader import read_deck
from pyradioss.input.starter_keywords import parse_starter_deck
from pyradioss.model.model import Model
from pyradioss.starter.checks import check_model


def _parse_and_check(tmp_path: Path, text: str):
    """Parse a deck and run the model checker."""
    p = tmp_path / "TEST_0000.rad"
    p.write_text(text, encoding="ascii")
    blocks = read_deck(str(p))
    log = MessageLog()
    model = Model()
    parse_starter_deck(blocks, model, log)
    # Run just the checks (not full starter init, which requires elements)
    # Run model checks (includes cross-reference validation)
    check_model(model, log)
    return model, log


def _parse_only(tmp_path: Path, text: str):
    """Parse a deck without running checks."""
    p = tmp_path / "TEST_0000.rad"
    p.write_text(text, encoding="ascii")
    blocks = read_deck(str(p))
    log = MessageLog()
    model = Model()
    parse_starter_deck(blocks, model, log)
    return model, log


# ---------------------------------------------------------------------------
# ADMAS cross-reference tests
# ---------------------------------------------------------------------------
def test_admas_type0_needs_node_group(tmp_path):
    """mass_type=0 (per-node): grnod_id must be in node_groups."""
    text = """/BEGIN
TEST M153
90
90
/GRNOD/NODE/10
nodes
1 2 3
/ADMAS/0/1
added mass on group
   5.0    10
/NODE
         1                 0.0                 0.0                 0.0
         2                 1.0                 0.0                 0.0
         3                 0.0                 1.0                 0.0
/END"""
    model, log = _parse_and_check(tmp_path, text)
    # group 10 exists → no error
    admas_errors = [e for e in log.errors if "ADMAS" in e]
    assert len(admas_errors) == 0


def test_admas_type0_missing_group_errors(tmp_path):
    """mass_type=0 with nonexistent node group → should error."""
    text = """/BEGIN
TEST M153
90
90
/ADMAS/0/1
added mass
   5.0    999
/END"""
    model, log = _parse_and_check(tmp_path, text)
    admas_errors = [e for e in log.errors if "ADMAS" in e and "node group" in e]
    assert len(admas_errors) == 1
    assert "999" in admas_errors[0]


def test_admas_type2_surf_no_false_positive(tmp_path):
    """mass_type=2 (surface): grnod_id holds a surface ID, not a node group.
    Should NOT produce a 'node group not defined' error."""
    text = """/BEGIN
TEST M153
90
90
/ADMAS/SURF/2
surface mass
   5.0    42
/END"""
    model, log = _parse_and_check(tmp_path, text)
    # The grnod_id=42 is actually a surface ID, not a node group.
    # Should not produce a node group cross-ref error.
    admas_errors = [e for e in log.errors if "ADMAS" in e and "node group" in e]
    assert len(admas_errors) == 0


def test_admas_type3_part_no_false_positive(tmp_path):
    """mass_type=3 (part group): grnod_id holds a part group ID.
    Should NOT produce a 'node group not defined' error."""
    text = """/BEGIN
TEST M153
90
90
/ADMAS/TOTAL_BOX/3
box mass
   5.0    77
/END"""
    model, log = _parse_and_check(tmp_path, text)
    admas_errors = [e for e in log.errors if "ADMAS" in e and "node group" in e]
    assert len(admas_errors) == 0


def test_admas_type1_total_needs_node_group(tmp_path):
    """mass_type=1 (total on nodes): grnod_id is a node group, should check."""
    text = """/BEGIN
TEST M153
90
90
/ADMAS/TOTAL/1
total mass
   5.0    888
/END"""
    model, log = _parse_and_check(tmp_path, text)
    admas_errors = [e for e in log.errors if "ADMAS" in e and "node group" in e]
    assert len(admas_errors) == 1
    assert "888" in admas_errors[0]


# ---------------------------------------------------------------------------
# "model has no elements" downgrade test
# ---------------------------------------------------------------------------
def test_no_elements_is_warning_not_error(tmp_path):
    """A deck with no elements should produce a warning, not an error."""
    text = """/BEGIN
TEST M153
90
90
/NODE
         1                 0.0                 0.0                 0.0
/END"""
    model, log = _parse_only(tmp_path, text)
    # Run check_model directly
    check_model(model, log)
    # Should have no errors about 'no elements', but a warning
    elem_errors = [e for e in log.errors if "no elements" in e]
    assert len(elem_errors) == 0
    elem_warnings = [w for w in log.warnings if "no elements" in w]
    assert len(elem_warnings) == 1
    assert "unported" in elem_warnings[0]
