"""
Tests for Milestone M567: /MAT/LAW94 (/MAT/YEOH) Yeoh Hyperelastic Model
Deck Writer Formatting, Keyword Aliases & Round-Trip Parsing Audit.
"""

from __future__ import annotations

import math
from pathlib import Path
import numpy as np
import pytest

from pyradioss.input.deck_writer import StarterDeck
from pyradioss.input.deck_reader import read_deck
from pyradioss.input.starter_keywords import read_mat_law94
from pyradioss.model.entities import MaterialLaw94
from pyradioss.model import Model
from pyradioss.starter.checks import MessageLog


# ============================================================================
# 1. Deck Writer Keyword Formatting & Synonyms
# ============================================================================

def test_deck_writer_mat_law94_kwargs():
    """Verify writing /MAT/LAW94 via kwargs."""
    d = StarterDeck("TEST_LAW94")
    d.mat_law94(
        id=5,
        title="YeohPolymer",
        rho0=1150.0,
        c10=1.8e6,
        c20=-1.0e5,
        c30=1.0e4,
        d1=1.5e-8,
        d2=1.0e-9,
        d3=0.0,
    )
    text = d.write()
    assert "/MAT/LAW94/5" in text
    assert "1800000" in text or "1.8" in text


def test_deck_writer_mat_yeoh_alias():
    """Verify writing /MAT/YEOH alias."""
    d = StarterDeck("TEST_ALIAS")
    d.mat_yeoh(
        id=12,
        title="Elastomer",
        rho0=1000.0,
        c10=2.5e6,
        c20=0.0,
        c30=0.0,
        d1=2.0e-8,
        d2=0.0,
        d3=0.0,
    )
    text = d.write()
    assert "/MAT/YEOH/12" in text or "/MAT/LAW94/12" in text
    assert "Elastomer" in text


def test_deck_writer_mat_law94_entity():
    """Verify writing /MAT/LAW94 from MaterialLaw94 entity object."""
    mat = MaterialLaw94(
        id=42,
        title="EntityMat",
        rho0=1200.0,
        c10=3.0e6,
        c20=-2.0e5,
        c30=5.0e4,
        d1=1e-8,
        d2=0.0,
        d3=0.0,
    )
    d = StarterDeck("TEST_ENTITY")
    d.mat_law94(mat_law94=mat)
    text = d.write()
    assert "42" in text
    assert "EntityMat" in text


# ============================================================================
# 2. Write -> Read Roundtrip
# ============================================================================

def test_law94_write_read_roundtrip(tmp_path: Path):
    """Write deck to disk, read back via read_deck, and verify exact property retention."""
    deck_path = tmp_path / "roundtrip_0000.rad"

    orig_id = 99
    orig_title = "Yeoh_Roundtrip"
    orig_rho0 = 1080.0
    orig_c10 = 1.65e6
    orig_c20 = -1.2e5
    orig_c30 = 8.5e3
    orig_d1 = 2.4e-8
    orig_d2 = 1.1e-9
    orig_d3 = 3.5e-11

    d = StarterDeck("ROUNDTRIP")
    d.mat_law94(
        id=orig_id,
        title=orig_title,
        rho0=orig_rho0,
        c10=orig_c10,
        c20=orig_c20,
        c30=orig_c30,
        d1=orig_d1,
        d2=orig_d2,
        d3=orig_d3,
    )
    d.write(str(deck_path))

    # Read back
    blocks = read_deck(str(deck_path))
    model = Model()
    log = MessageLog()
    for b in blocks:
        if b.key0 == "MAT":
            read_mat_law94(b, model, log)

    assert orig_id in model.mat_law94s
    mat = model.mat_law94s[orig_id]
    assert mat.id == orig_id
    assert orig_title in mat.title
    assert math.isclose(mat.rho0, orig_rho0, rel_tol=1e-5)
    assert math.isclose(mat.c10, orig_c10, rel_tol=1e-5)
    assert math.isclose(mat.c20, orig_c20, rel_tol=1e-5)
    assert math.isclose(mat.c30, orig_c30, rel_tol=1e-5)
    assert math.isclose(mat.d1, orig_d1, rel_tol=1e-5)
    assert math.isclose(mat.d2, orig_d2, rel_tol=1e-5)
    assert math.isclose(mat.d3, orig_d3, rel_tol=1e-5)
