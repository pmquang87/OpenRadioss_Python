"""
Tests for Milestone M569: /MAT/LAW95 (/MAT/BERGSTROM_BOYCE) Bergstrom-Boyce Visco-Hyperelastic Model
Deck Writer Formatting, Keyword Aliases & Round-Trip Parsing Audit.
"""

from __future__ import annotations

import math
from pathlib import Path
import pytest

from pyradioss.input.deck_writer import StarterDeck
from pyradioss.input.deck_reader import read_deck
from pyradioss.input.starter_keywords import read_mat_law95
from pyradioss.model.entities import MaterialLaw95
from pyradioss.model import Model
from pyradioss.starter.checks import MessageLog


# ============================================================================
# 1. Deck Writer Keyword Formatting & Synonyms
# ============================================================================

def test_deck_writer_mat_law95_kwargs():
    """Verify writing /MAT/LAW95 via kwargs."""
    d = StarterDeck("TEST_LAW95")
    d.mat_law95(
        id=5,
        title="BBPolymer",
        rho0=1150.0,
        c10=1.8e6,
        c01=2.0e5,
        c20=0.0,
        c11=0.0,
        c02=0.0,
        c30=0.0,
        c21=0.0,
        c12=0.0,
        c03=0.0,
        sb=0.75,
        d1=1.5e-8,
        d2=0.0,
        d3=0.0,
        iform=1,
        a=0.002,
        expc=-0.7,
        expm=1.1,
        ksi=0.01,
        tauref=1.0e5,
    )
    text = d.write()
    assert "/MAT/LAW95/5" in text
    assert "BBPolymer" in text
    assert "1150" in text
    assert "1800000" in text or "1.8" in text


def test_deck_writer_mat_bergstrom_boyce_alias():
    """Verify writing /MAT/BERGSTROM_BOYCE alias."""
    d = StarterDeck("TEST_ALIAS")
    d.mat_bergstrom_boyce(
        id=12,
        title="ElastomerBB",
        rho0=1000.0,
        c10=2.5e6,
        c01=3.0e5,
        sb=1.0,
        d1=2.0e-8,
        a=0.005,
        expc=-0.65,
        expm=1.0,
        ksi=0.02,
        tauref=2.0e5,
    )
    text = d.write()
    assert "/MAT/BERGSTROM_BOYCE/12" in text or "/MAT/LAW95/12" in text
    assert "ElastomerBB" in text


def test_deck_writer_mat_law95_entity():
    """Verify writing /MAT/LAW95 from MaterialLaw95 entity object."""
    mat = MaterialLaw95(
        id=42,
        title="EntityBBMat",
        rho0=1200.0,
        c10=3.0e6,
        c01=4.0e5,
        sb=0.8,
        d1=1e-8,
        d2=0.0,
        d3=0.0,
        iform=1,
        a=0.01,
        expc=-0.7,
        expm=1.2,
        ksi=0.01,
        tauref=1.5e5,
    )
    d = StarterDeck("TEST_ENTITY")
    d.mat_law95(mat_law95=mat)
    text = d.write()
    assert "42" in text
    assert "EntityBBMat" in text


# ============================================================================
# 2. Write -> Read Roundtrip
# ============================================================================

def test_law95_write_read_roundtrip(tmp_path: Path):
    """Write deck to disk, read back via read_deck, and verify exact property retention."""
    deck_path = tmp_path / "roundtrip_0000.rad"

    orig_id = 99
    orig_title = "BB_Roundtrip"
    orig_rho0 = 1080.0
    orig_c10 = 1.65e6
    orig_c01 = 2.2e5
    orig_sb = 0.65
    orig_d1 = 2.4e-8
    orig_d2 = 0.0
    orig_d3 = 0.0
    orig_iform = 1
    orig_a = 0.008
    orig_expc = -0.7
    orig_expm = 1.05
    orig_ksi = 0.015
    orig_tauref = 1.2e5

    d = StarterDeck("ROUNDTRIP")
    d.mat_law95(
        id=orig_id,
        title=orig_title,
        rho0=orig_rho0,
        c10=orig_c10,
        c01=orig_c01,
        sb=orig_sb,
        d1=orig_d1,
        d2=orig_d2,
        d3=orig_d3,
        iform=orig_iform,
        a=orig_a,
        expc=orig_expc,
        expm=orig_expm,
        ksi=orig_ksi,
        tauref=orig_tauref,
    )
    d.write(str(deck_path))

    blocks = read_deck(str(deck_path))
    model = Model()
    log = MessageLog()

    for block in blocks:
        if block.key0 == "MAT":
            read_mat_law95(block, model, log)

    assert orig_id in model.mat_law95s
    mat = model.mat_law95s[orig_id]

    assert mat.id == orig_id
    assert orig_title in mat.title
    assert math.isclose(mat.rho0, orig_rho0, rel_tol=1e-5)
    assert math.isclose(mat.c10, orig_c10, rel_tol=1e-5)
    assert math.isclose(mat.c01, orig_c01, rel_tol=1e-5)
    assert math.isclose(mat.sb, orig_sb, rel_tol=1e-5)
    assert math.isclose(mat.d1, orig_d1, rel_tol=1e-5)
    assert mat.iform == orig_iform
    assert math.isclose(mat.a, orig_a, rel_tol=1e-5)
    assert math.isclose(mat.expc, orig_expc, rel_tol=1e-5)
    assert math.isclose(mat.expm, orig_expm, rel_tol=1e-5)
    assert math.isclose(mat.ksi, orig_ksi, rel_tol=1e-5)
    assert math.isclose(mat.tauref, orig_tauref, rel_tol=1e-5)
