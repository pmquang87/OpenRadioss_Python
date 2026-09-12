"""
Tests for Milestone M568: /MAT/LAW93 (/MAT/ORTH_HILL) Orthotropic Hill Model
Deck Writer Formatting, Keyword Aliases & Round-Trip Parsing Audit.
"""

from __future__ import annotations

import math
from pathlib import Path
import numpy as np
import pytest

from pyradioss.input.deck_writer import StarterDeck
from pyradioss.input.deck_reader import read_deck
from pyradioss.input.starter_keywords import read_mat_law93
from pyradioss.model.entities import MaterialLaw93
from pyradioss.model import Model
from pyradioss.starter.checks import MessageLog


# ============================================================================
# 1. Deck Writer Keyword Formatting & Synonyms
# ============================================================================

def test_deck_writer_mat_law93_kwargs():
    """Verify writing /MAT/LAW93 via kwargs."""
    d = StarterDeck("TEST_LAW93")
    d.mat_law93(
        id=7,
        title="SheetSteel",
        rho0=7.85e-6,
        refer_rho=0.0,
        e11=210000.0,
        e22=205000.0,
        e33=210000.0,
        g12=80000.0,
        nu12=0.3,
        g13=80000.0,
        g23=78000.0,
        nu13=0.3,
        nu23=0.29,
        sigma_y=350.0,
        qr1=100.0,
        cr1=20.0,
        qr2=50.0,
        cr2=5.0,
        r11=1.1,
        r22=0.95,
        r12=1.0,
        r33=1.0,
        r13=1.0,
        r23=1.0,
    )
    text = d.write()
    assert "/MAT/LAW93/7" in text
    assert "SheetSteel" in text
    assert "210000" in text or "2.1" in text
    assert "350" in text


def test_deck_writer_mat_orth_hill_alias():
    """Verify writing /MAT/ORTH_HILL alias."""
    d = StarterDeck("TEST_ALIAS")
    d.mat_orth_hill(
        id=14,
        title="AlloyHill",
        rho0=2.7e-6,
        e11=70000.0,
        e22=68000.0,
        e33=70000.0,
        g12=26000.0,
        nu12=0.33,
        g13=26000.0,
        g23=25000.0,
        nu13=0.33,
        nu23=0.32,
        sigma_y=280.0,
        r11=1.05,
        r22=0.92,
        r12=0.98,
    )
    text = d.write()
    assert "/MAT/ORTH_HILL/14" in text
    assert "AlloyHill" in text


def test_deck_writer_mat_law93_entity():
    """Verify writing /MAT/LAW93 from MaterialLaw93 entity object."""
    mat = MaterialLaw93(
        id=55,
        title="EntityHill",
        rho0=7.8e-6,
        e11=200000.0,
        e22=195000.0,
        e33=200000.0,
        g12=77000.0,
        nu12=0.28,
        g13=77000.0,
        g23=75000.0,
        nu13=0.28,
        nu23=0.27,
        sigma_y=320.0,
        qr1=80.0,
        cr1=15.0,
        r11=1.02,
        r22=0.96,
        r12=1.0,
    )
    d = StarterDeck("TEST_ENTITY")
    d.mat_law93(mat_law93=mat)
    text = d.write()
    assert "/MAT/LAW93/55" in text
    assert "EntityHill" in text


# ============================================================================
# 2. Write -> Read Roundtrip
# ============================================================================

def test_law93_write_read_roundtrip(tmp_path: Path):
    """Write deck to disk, read back via read_deck, and verify exact property retention."""
    deck_path = tmp_path / "roundtrip_0000.rad"

    orig_id = 88
    orig_title = "Hill_Roundtrip"
    orig_rho0 = 7.82e-6
    orig_e11 = 208000.0
    orig_e22 = 204000.0
    orig_e33 = 208000.0
    orig_g12 = 79000.0
    orig_nu12 = 0.29
    orig_g13 = 79000.0
    orig_g23 = 77000.0
    orig_nu13 = 0.29
    orig_nu23 = 0.28
    orig_sy = 360.0
    orig_qr1 = 110.0
    orig_cr1 = 22.0
    orig_qr2 = 45.0
    orig_cr2 = 4.0
    orig_r11 = 1.08
    orig_r22 = 0.94
    orig_r12 = 0.99
    orig_r33 = 1.0
    orig_r13 = 1.0
    orig_r23 = 1.0

    d = StarterDeck("ROUNDTRIP_DECK")
    d.mat_law93(
        id=orig_id,
        title=orig_title,
        rho0=orig_rho0,
        e11=orig_e11,
        e22=orig_e22,
        e33=orig_e33,
        g12=orig_g12,
        nu12=orig_nu12,
        g13=orig_g13,
        g23=orig_g23,
        nu13=orig_nu13,
        nu23=orig_nu23,
        sigma_y=orig_sy,
        qr1=orig_qr1,
        cr1=orig_cr1,
        qr2=orig_qr2,
        cr2=orig_cr2,
        r11=orig_r11,
        r22=orig_r22,
        r12=orig_r12,
        r33=orig_r33,
        r13=orig_r13,
        r23=orig_r23,
    )
    deck_path.write_text(d.write(), encoding="utf-8")

    # Read back
    blocks = read_deck(str(deck_path))
    model = Model()
    log = MessageLog()
    for block in blocks:
        if block.key0 == "MAT":
            read_mat_law93(block, model, log)

    assert orig_id in model.mat_law93s
    mat = model.mat_law93s[orig_id]
    assert mat.id == orig_id
    assert orig_title in mat.title
    assert mat.rho0 == pytest.approx(orig_rho0)
    assert mat.e11 == pytest.approx(orig_e11)
    assert mat.e22 == pytest.approx(orig_e22)
    assert mat.e33 == pytest.approx(orig_e33)
    assert mat.g12 == pytest.approx(orig_g12)
    assert mat.nu12 == pytest.approx(orig_nu12)
    assert mat.g13 == pytest.approx(orig_g13)
    assert mat.g23 == pytest.approx(orig_g23)
    assert mat.nu13 == pytest.approx(orig_nu13)
    assert mat.nu23 == pytest.approx(orig_nu23)
    assert mat.sigma_y == pytest.approx(orig_sy)
    assert mat.qr1 == pytest.approx(orig_qr1)
    assert mat.cr1 == pytest.approx(orig_cr1)
    assert mat.qr2 == pytest.approx(orig_qr2)
    assert mat.cr2 == pytest.approx(orig_cr2)
    assert mat.r11 == pytest.approx(orig_r11)
    assert mat.r22 == pytest.approx(orig_r22)
    assert mat.r12 == pytest.approx(orig_r12)
    assert mat.r33 == pytest.approx(orig_r33)
    assert mat.r13 == pytest.approx(orig_r13)
    assert mat.r23 == pytest.approx(orig_r23)
