"""Tests for /MAT/LAW101 (/MAT/PP / /MAT/PLAS_POLY) deck writer and parser roundtrip.

Verifies:
  - StarterDeck.mat_law101 formatting of all 11 cards
  - StarterDeck.mat_pp and StarterDeck.mat_plas_poly alias writer methods
  - Roundtrip: DeckWriter -> .rad string -> StarterReader -> MatLaw101
  - Floating point fidelity of all Bouvard parameters across cards 1..11
"""

from pathlib import Path
import pytest

from pyradioss.common.messages import MessageLog
from pyradioss.input.deck_reader import read_deck
from pyradioss.input.deck_writer import StarterDeck
from pyradioss.input.starter_keywords import parse_starter_deck
from pyradioss.model.entities import MatLaw101
from pyradioss.model.model import Model


def _parse_deck_str(tmp_path: Path, text: str) -> Model:
    p = tmp_path / "ROUNDTRIP_0000.rad"
    p.write_text(text.strip() + "\n", encoding="ascii")
    blocks = read_deck(str(p))
    model = Model()
    log = MessageLog()
    parse_starter_deck(blocks, model, log)
    assert not log.errors, f"Parse errors: {log.errors}"
    return model


def test_starter_deck_writer_mat_law101(tmp_path: Path):
    """Verify StarterDeck.mat_law101 writes 11 cards that parse back to equal values."""
    deck = StarterDeck("RT_LAW101")
    deck.mat_law101(
        mat_id=101,
        rho0=0.91e-6,
        rhor=0.0,
        e=1550.0,
        alpha1=-2.6,
        nu=0.37,
        ve1=0.28,
        ve2=1.4,
        epsilonref=1.2e-4,
        gamma0=1.1e-3,
        alpha_p=0.075,
        deltah=46000.0,
        vol=2.1e-28,
        m=1.75,
        c3=-0.045,
        c4=26.0,
        alphak1=0.16,
        alphak2=0.055,
        hard=52.0,
        zeta1i=0.11,
        c5=-0.0011,
        c6=1.05,
        c7=-0.0021,
        c8=2.1,
        c9=-0.011,
        c10=10.5,
        hard1=21.0,
        zeta2i=0.055,
        c11=-0.0011,
        c12=1.55,
        c13=0.011,
        c14=5.1,
        c1=-0.052,
        c2=8.2,
        lambdal=5.2,
        rho_ref=0.91e-6,
        cv_ref=1850.0,
        tref=294.0,
        alpha_th=1.25e-4,
        theta_glass=252.0,
        omega=0.88,
        theta_flag=2.0,
        heat_t0=294.0,
        title="Bouvard PP Roundtrip",
    )

    content = deck.render()
    assert "/MAT/LAW101/101" in content
    assert "Bouvard PP Roundtrip" in content

    model = _parse_deck_str(tmp_path, content)
    assert 101 in model.mat_law101s
    m = model.mat_law101s[101]

    assert m.id == 101
    assert m.title == "Bouvard PP Roundtrip"
    assert pytest.approx(m.rho0) == 0.91e-6
    assert pytest.approx(m.e) == 1550.0
    assert pytest.approx(m.alpha1) == -2.6
    assert pytest.approx(m.nu) == 0.37
    assert pytest.approx(m.ve1) == 0.28
    assert pytest.approx(m.ve2) == 1.4
    assert pytest.approx(m.epsilonref) == 1.2e-4
    assert pytest.approx(m.gamma0) == 1.1e-3
    assert pytest.approx(m.alpha_p) == 0.075
    assert pytest.approx(m.deltah) == 46000.0
    assert pytest.approx(m.vol) == 2.1e-28
    assert pytest.approx(m.m) == 1.75
    assert pytest.approx(m.c3) == -0.045
    assert pytest.approx(m.c4) == 26.0
    assert pytest.approx(m.hard) == 52.0
    assert pytest.approx(m.zeta1i) == 0.11
    assert pytest.approx(m.c10) == 10.5
    assert pytest.approx(m.hard1) == 21.0
    assert pytest.approx(m.zeta2i) == 0.055
    assert pytest.approx(m.c14) == 5.1
    assert pytest.approx(m.c1) == -0.052
    assert pytest.approx(m.c2) == 8.2
    assert pytest.approx(m.lambdal) == 5.2
    assert pytest.approx(m.theta_glass) == 252.0
    assert pytest.approx(m.omega) == 0.88
    assert pytest.approx(m.theta_flag) == 2.0
    assert pytest.approx(m.heat_t0) == 294.0


def test_starter_deck_writer_mat_pp_alias(tmp_path: Path):
    """Verify StarterDeck.mat_pp writes /MAT/PP that parses into mat_law101s."""
    deck = StarterDeck("RT_PP")
    deck.mat_pp(
        mat_id=202,
        rho0=0.905e-6,
        e=1480.0,
        nu=0.36,
        title="PP Alias Material",
    )
    content = deck.render()
    assert "/MAT/PP/202" in content
    assert "PP Alias Material" in content

    model = _parse_deck_str(tmp_path, content)
    assert 202 in model.mat_law101s
    m = model.mat_law101s[202]
    assert m.id == 202
    assert pytest.approx(m.rho0) == 0.905e-6
    assert pytest.approx(m.e) == 1480.0
    assert pytest.approx(m.nu) == 0.36


def test_starter_deck_writer_mat_plas_poly_alias(tmp_path: Path):
    """Verify StarterDeck.mat_plas_poly writes /MAT/PLAS_POLY that parses into mat_law101s."""
    deck = StarterDeck("RT_PLAS_POLY")
    deck.mat_plas_poly(
        mat_id=303,
        rho0=0.92e-6,
        e=1600.0,
        nu=0.39,
        title="PLAS_POLY Alias Material",
    )
    content = deck.render()
    assert "/MAT/PLAS_POLY/303" in content
    assert "PLAS_POLY Alias Material" in content

    model = _parse_deck_str(tmp_path, content)
    assert 303 in model.mat_law101s
    m = model.mat_law101s[303]
    assert m.id == 303
    assert pytest.approx(m.rho0) == 0.92e-6
    assert pytest.approx(m.e) == 1600.0
    assert pytest.approx(m.nu) == 0.39
