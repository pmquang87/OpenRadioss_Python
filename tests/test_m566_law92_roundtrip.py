"""
Tests for Milestone M566: /MAT/LAW92 (/MAT/ARRUDA_BOYCE) Arruda-Boyce Hyperelastic Model
Deck Writer Formatting, Keyword Aliases & Round-Trip Parsing Audit.
"""

from __future__ import annotations

import math
from pathlib import Path
import numpy as np
import pytest

from pyradioss.input.deck_writer import StarterDeck
from pyradioss.input.deck_reader import read_deck
from pyradioss.input.starter_keywords import read_mat_law92
from pyradioss.model.entities import MaterialLaw92
from pyradioss.model import Model
from pyradioss.starter.checks import MessageLog


# ============================================================================
# 1. Deck Writer Keyword Formatting & Synonyms
# ============================================================================

def test_deck_writer_mat_law92_kwargs():
    """Verify writing /MAT/LAW92 via kwargs."""
    d = StarterDeck("TEST_LAW92")
    d.mat_law92(
        id=5,
        title="ArrudaPolymer",
        rho0=1150.0,
        mu=1.8e6,
        d=1.5e-8,
        lam=6.5,
        itype=1,
        fct_id=0,
        nu=0.49,
        fscale=1.0,
    )
    text = d.write()
    assert "/MAT/LAW92/5" in text
    assert "ArrudaPolymer" in text


def test_deck_writer_mat_arruda_boyce_alias():
    """Verify writing /MAT/ARRUDA_BOYCE alias."""
    d = StarterDeck("TEST_ALIAS")
    d.mat_arruda_boyce(
        id=12,
        title="Elastomer",
        rho0=1000.0,
        mu=2.5e6,
        d=2.0e-8,
        lam=5.0,
        itype=2,
        fct_id=0,
        nu=0.495,
        fscale=1.0,
    )
    text = d.write()
    assert "/MAT/ARRUDA_BOYCE/12" in text or "/MAT/LAW92/12" in text
    assert "Elastomer" in text


def test_deck_writer_mat_law92_entity():
    """Verify writing /MAT/LAW92 from MaterialLaw92 entity object."""
    mat = MaterialLaw92(
        id=42,
        title="EntityMat",
        rho0=1200.0,
        mu=3.0e6,
        d=1e-8,
        lam=7.0,
        itype=1,
        fct_id=0,
        nu=0.495,
        fscale=1.0,
    )
    d = StarterDeck("TEST_ENTITY")
    d.mat_law92(mat_law92=mat)
    text = d.write()
    assert "42" in text
    assert "EntityMat" in text


# ============================================================================
# 2. Write -> Read Roundtrip
# ============================================================================

def test_law92_write_read_roundtrip(tmp_path: Path):
    """Write deck to disk, read back via read_deck, and verify exact property retention."""
    deck_path = tmp_path / "roundtrip_0000.rad"

    orig_id = 99
    orig_title = "Rubber_Roundtrip"
    orig_rho0 = 1080.0
    orig_mu = 1.65e6
    orig_d = 2.4e-8
    orig_lam = 5.8
    orig_itype = 1
    orig_fct_id = 0
    orig_nu = 0.492
    orig_fscale = 1.0

    d = StarterDeck("ROUNDTRIP")
    d.mat_law92(
        id=orig_id,
        title=orig_title,
        rho0=orig_rho0,
        mu=orig_mu,
        d=orig_d,
        lam=orig_lam,
        itype=orig_itype,
        fct_id=orig_fct_id,
        nu=orig_nu,
        fscale=orig_fscale,
    )
    d.write(str(deck_path))

    # Read back
    blocks = read_deck(str(deck_path))
    model = Model()
    log = MessageLog()
    for block in blocks:
        if block.key0 == "MAT":
            read_mat_law92(block, model, log)

    assert orig_id in model.mat_law92s
    mat = model.mat_law92s[orig_id]

    assert mat.id == orig_id
    assert orig_title in mat.title
    assert math.isclose(mat.rho0, orig_rho0, rel_tol=1e-5)
    assert math.isclose(mat.mu, orig_mu, rel_tol=1e-5)
    assert math.isclose(mat.d, orig_d, rel_tol=1e-5)
    assert math.isclose(mat.lam, orig_lam, rel_tol=1e-5)
    assert mat.itype == orig_itype
    assert mat.fct_id == orig_fct_id
    assert math.isclose(mat.nu, orig_nu, rel_tol=1e-5)
    assert math.isclose(mat.fscale, orig_fscale, rel_tol=1e-5)
