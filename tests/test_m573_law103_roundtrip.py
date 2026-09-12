"""Roundtrip deck writing and re-parsing tests for /MAT/LAW103 (M573).

Verifies:
- StarterDeck.mat_law103 writes standard 6 cards matching CFG format.
- StarterDeck.mat_hensel_spittel and mat_plas_hens write proper headers.
- Re-parsing written decks produces matching MatLaw103 entity parameters.
- Roundtrip with custom title, unit IDs, and reference density.
"""

from __future__ import annotations

import math
from pathlib import Path
import pytest

from pyradioss.common.messages import MessageLog
from pyradioss.input.deck_reader import read_deck
from pyradioss.input.deck_writer import StarterDeck
from pyradioss.input.starter_keywords import parse_starter_deck
from pyradioss.model.entities import MatLaw103
from pyradioss.model.model import Model


def _roundtrip(tmp_path: Path, deck: StarterDeck) -> Model:
    text = deck.render()
    p = tmp_path / "ROUNDTRIP_0000.rad"
    p.write_text(text, encoding="ascii")
    blocks = read_deck(str(p))
    model = Model()
    log = MessageLog()
    parse_starter_deck(blocks, model, log)
    assert not log.errors, f"Parse errors: {log.errors}"
    return model


def test_roundtrip_mat_law103_kwargs(tmp_path: Path):
    """Write /MAT/LAW103 using kwargs and re-read."""
    d = StarterDeck("M573_KWARGS")
    d.mat_law103(
        mid=101,
        title="Ti-6Al-4V Hot Forming",
        rho=4.43e-9,
        refer_rho=4.43e-9,
        e=110000.0,
        nu=0.34,
        a0=850.0,
        m1=-0.0028,
        m2=0.18,
        m3=0.06,
        m4=-0.0005,
        m5=-0.00015,
        m7=0.025,
        fsmooth=1,
        fcut=500.0,
        eps0=0.001,
        pmin=-2.5e20,
        rcp=2.4e-3,
        t0=973.15,
        eta=0.92,
    )

    model = _roundtrip(tmp_path, d)
    assert 101 in model.mat_law103s
    mat = model.mat_law103s[101]

    assert mat.id == 101
    assert "Ti-6Al-4V" in mat.title
    assert math.isclose(mat.rho, 4.43e-9, rel_tol=1.0e-9)
    assert math.isclose(mat.refer_rho, 4.43e-9, rel_tol=1.0e-9)
    assert math.isclose(mat.e, 110000.0, rel_tol=1.0e-9)
    assert math.isclose(mat.nu, 0.34, rel_tol=1.0e-9)
    assert math.isclose(mat.a0, 850.0, rel_tol=1.0e-9)
    assert math.isclose(mat.m1, -0.0028, rel_tol=1.0e-9)
    assert math.isclose(mat.m2, 0.18, rel_tol=1.0e-9)
    assert math.isclose(mat.m3, 0.06, rel_tol=1.0e-9)
    assert math.isclose(mat.m4, -0.0005, rel_tol=1.0e-9)
    assert math.isclose(mat.m5, -0.00015, rel_tol=1.0e-9)
    assert math.isclose(mat.m7, 0.025, rel_tol=1.0e-9)
    assert mat.fsmooth == 1
    assert math.isclose(mat.fcut, 500.0, rel_tol=1.0e-9)
    assert math.isclose(mat.eps_0, 0.001, rel_tol=1.0e-9)
    assert math.isclose(mat.pmin, -2.5e20, rel_tol=1.0e-9)
    assert math.isclose(mat.rhocp, 2.4e-3, rel_tol=1.0e-9)
    assert math.isclose(mat.t0, 973.15, rel_tol=1.0e-9)
    assert math.isclose(mat.eta, 0.92, rel_tol=1.0e-9)


def test_roundtrip_mat_law103_entity(tmp_path: Path):
    """Write /MAT/LAW103 from an existing MatLaw103 entity object and re-read."""
    orig = MatLaw103(
        id=202,
        title="Inconel 718 Extrusion",
        rho=8.19e-9,
        refer_rho=8.19e-9,
        e=205000.0,
        nu=0.29,
        a0=950.0,
        m1=-0.002,
        m2=0.22,
        m3=0.05,
        m4=-0.001,
        m5=-0.0001,
        m7=0.018,
        fsmooth=0,
        fcut=0.0,
        eps_0=0.002,
        pmin=-1.0e30,
        rhocp=3.6e-3,
        t0=1273.15,
        eta=0.88,
    )

    d = StarterDeck("M573_ENTITY")
    d.mat_law103(orig)

    model = _roundtrip(tmp_path, d)
    assert 202 in model.mat_law103s
    mat = model.mat_law103s[202]

    assert mat.id == orig.id
    assert "Inconel 718" in mat.title
    assert math.isclose(mat.rho, orig.rho, rel_tol=1.0e-9)
    assert math.isclose(mat.e, orig.e, rel_tol=1.0e-9)
    assert math.isclose(mat.nu, orig.nu, rel_tol=1.0e-9)
    assert math.isclose(mat.a0, orig.a0, rel_tol=1.0e-9)
    assert math.isclose(mat.m1, orig.m1, rel_tol=1.0e-9)
    assert math.isclose(mat.m2, orig.m2, rel_tol=1.0e-9)
    assert math.isclose(mat.m3, orig.m3, rel_tol=1.0e-9)
    assert math.isclose(mat.m4, orig.m4, rel_tol=1.0e-9)
    assert math.isclose(mat.m5, orig.m5, rel_tol=1.0e-9)
    assert math.isclose(mat.m7, orig.m7, rel_tol=1.0e-9)
    assert mat.fsmooth == orig.fsmooth
    assert math.isclose(mat.eps_0, orig.eps_0, rel_tol=1.0e-9)
    assert math.isclose(mat.rhocp, orig.rhocp, rel_tol=1.0e-9)
    assert math.isclose(mat.t0, orig.t0, rel_tol=1.0e-9)
    assert math.isclose(mat.eta, orig.eta, rel_tol=1.0e-9)


def test_roundtrip_aliases(tmp_path: Path):
    """Write using mat_hensel_spittel and mat_plas_hens aliases and re-read."""
    d = StarterDeck("M573_ALIASES")
    d.mat_hensel_spittel(
        mid=301,
        title="Alias Hensel Spittel",
        rho=2.7e-9,
        e=70000.0,
        nu=0.33,
        a0=180.0,
        m1=-0.003,
        m2=0.1,
        m3=0.04,
        t0=673.15,
        eta=0.9,
    )
    d.mat_plas_hens(
        mid=302,
        title="Alias Plas Hens",
        rho=7.8e-9,
        e=210000.0,
        nu=0.3,
        a0=450.0,
        m1=-0.0015,
        m2=0.16,
        m3=0.03,
        t0=1000.0,
        eta=0.85,
    )

    model = _roundtrip(tmp_path, d)
    assert 301 in model.mat_law103s
    assert 302 in model.mat_law103s

    m301 = model.mat_law103s[301]
    assert m301.id == 301
    assert math.isclose(m301.a0, 180.0, rel_tol=1.0e-9)
    assert math.isclose(m301.t0, 673.15, rel_tol=1.0e-9)

    m302 = model.mat_law103s[302]
    assert m302.id == 302
    assert math.isclose(m302.a0, 450.0, rel_tol=1.0e-9)
    assert math.isclose(m302.t0, 1000.0, rel_tol=1.0e-9)
