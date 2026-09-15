"""Tests for /MAT/LAW102 (/MAT/DPRAG2) deck writer and parser roundtrip.

Verifies:
  - StarterDeck.mat_law102 formatting of all 5 cards
  - StarterDeck.mat_dprag2 alias writer method
  - Roundtrip: DeckWriter -> .rad string -> StarterReader -> MatLaw102
  - Model entity conversion via _conv_mat dispatch
  - Floating point fidelity of all parameters across cards 1..5
"""

from pathlib import Path
import pytest

from pyradioss.common.messages import MessageLog
from pyradioss.input.deck_reader import read_deck
from pyradioss.input.deck_writer import StarterDeck
from pyradioss.input.starter_keywords import parse_starter_deck
from pyradioss.model.entities import MatLaw102, Material
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


def test_starter_deck_writer_mat_law102(tmp_path: Path):
    """Verify StarterDeck.mat_law102 writes 5 cards that parse back to equal values."""
    deck = StarterDeck("RT_LAW102")
    deck.mat_law102(
        mat_id=102,
        rho=2.6e-6,
        iform=2,
        e=28000.0,
        nu=0.22,
        c=12.5,
        phi=31.5,
        amax=220.0,
        pmin=-4.5,
        title="Extended Drucker-Prager Rock Model",
    )

    content = deck.render()
    assert "/MAT/LAW102/102" in content
    assert "Extended Drucker-Prager Rock Model" in content

    model = _parse_deck_str(tmp_path, content)
    assert 102 in model.mat_law102s
    m = model.mat_law102s[102]

    assert m.id == 102
    assert m.title == "Extended Drucker-Prager Rock Model"
    assert pytest.approx(m.rho) == 2.6e-6
    assert m.iform == 2
    assert pytest.approx(m.e) == 28000.0
    assert pytest.approx(m.nu) == 0.22
    assert pytest.approx(m.c) == 12.5
    assert pytest.approx(m.phi) == 31.5
    assert pytest.approx(m.amax) == 220.0
    assert pytest.approx(m.pmin) == -4.5


def test_starter_deck_writer_mat_dprag2_alias(tmp_path: Path):
    """Verify StarterDeck.mat_dprag2 alias writes /MAT/DPRAG2 keyword."""
    deck = StarterDeck("RT_DPRAG2")
    deck.mat_dprag2(
        mat_id=55,
        rho=2.45e-6,
        iform=3,
        e=32000.0,
        nu=0.18,
        c=8.0,
        phi=28.0,
        amax=150.0,
        pmin=-2.0,
        title="DPRAG2 Concrete Formulation",
    )

    content = deck.render()
    assert "/MAT/DPRAG2/55" in content
    assert "DPRAG2 Concrete Formulation" in content

    model = _parse_deck_str(tmp_path, content)
    assert 55 in model.mat_law102s
    m = model.mat_law102s[55]
    assert m.iform == 3
    assert pytest.approx(m.c) == 8.0
    assert pytest.approx(m.phi) == 28.0


def test_model_to_deck_conversion(tmp_path: Path):
    """Verify model export with MatLaw102 writes valid deck roundtripping back."""
    m_orig = MatLaw102(
        id=77,
        title="Model Conversion DPRAG2",
        rho=2.7e-6,
        iform=1,
        e=24000.0,
        nu=0.20,
        c=14.0,
        phi=33.0,
        amax=300.0,
        pmin=-6.0,
    )
    deck = StarterDeck("MODEL_CONV")
    deck.mat_law102(m_orig)

    content = deck.render()
    assert "/MAT/LAW102/77" in content

    model = _parse_deck_str(tmp_path, content)
    assert 77 in model.mat_law102s
    m = model.mat_law102s[77]
    assert pytest.approx(m.rho) == 2.7e-6
    assert m.iform == 1
    assert pytest.approx(m.e) == 24000.0
    assert pytest.approx(m.c) == 14.0
    assert pytest.approx(m.phi) == 33.0
    assert pytest.approx(m.amax) == 300.0
    assert pytest.approx(m.pmin) == -6.0
