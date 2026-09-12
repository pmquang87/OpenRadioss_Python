"""Fixed and free format parser and deck writer roundtrip tests for /MAT/LAW105 (M576).

Validates:
  - Fixed-format reading of /MAT/LAW105 and /MAT/POWDER_BURN.
  - Free-format reading.
  - Optional Refer_Rho support on Card 1.
  - Scaled function cards (Card 5 and Card 6).
  - StarterDeck.mat_law105 / mat_powder_burn / mat_powderburn emission and roundtrip parsing.
"""

from pathlib import Path
import pytest

from pyradioss.input.deck_reader import read_deck
from pyradioss.input.deck_writer import StarterDeck
from pyradioss.model.model import Model
from pyradioss.common.messages import MessageLog
from pyradioss.input.starter_keywords import read_starter_deck


def _parse_deck_string(tmp_path: Path, text: str) -> tuple[Model, MessageLog]:
    deck_path = tmp_path / "deck_0000.rad"
    deck_path.write_text(text, encoding="utf-8")
    raw_deck = read_deck(deck_path)
    model = Model()
    log = MessageLog()
    read_starter_deck(raw_deck, model, log)
    return model, log


def test_mat_law105_fixed_format_roundtrip(tmp_path: Path):
    """Test reading and re-emitting fixed-format /MAT/LAW105 deck."""
    deck_src = """\
# OpenRadioss Starter Deck
/BEGIN
Test Fixed LAW105
#   Unit system: Mg, mm, s
                  Mg                  mm                   s
/MAT/LAW105/1051
Propellant Powder Fixed
#              RHO_I           Refer_Rho
             1.8e-06             1.8e-06
#               BULK                  P0                 PSH
             15000.0                 0.1                 0.0
#                  D                  EG
              0.5e-6              4500.0
#                 Gr                   C               Alpha
                0.08                0.75                 1.0
#           F_id_b(P)                       SCALE_B             SCALE_P
                  12                            1.5                 2.0
#           F_id_g(r)                   SCALE_GAMMA           SCALE_RHO                  C1                  C2
                  14                            1.0                 1.0              2500.0                 0.0
/END
"""
    model, log = _parse_deck_string(tmp_path, deck_src)
    assert not log.errors, f"Errors parsing fixed LAW105: {log.errors}"
    assert 1051 in model.mat_law105s

    m1 = model.mat_law105s[1051]
    assert m1.rho0 == pytest.approx(1.8e-06)
    assert m1.refer_rho == pytest.approx(1.8e-06)
    assert m1.bulk == pytest.approx(15000.0)
    assert m1.p0 == pytest.approx(0.1)
    assert m1.psh == pytest.approx(0.0)
    assert m1.gas_d == pytest.approx(0.5e-6)
    assert m1.gas_eg == pytest.approx(4500.0)
    assert m1.gr == pytest.approx(0.08)
    assert m1.c == pytest.approx(0.75)
    assert m1.alpha == pytest.approx(1.0)
    assert m1.func_b == 12
    assert m1.scale_b == pytest.approx(1.5)
    assert m1.scale_p == pytest.approx(2.0)
    assert m1.func_gam == 14
    assert m1.scale_gam == pytest.approx(1.0)
    assert m1.scale_rho == pytest.approx(1.0)
    assert m1.c1 == pytest.approx(2500.0)

    # Re-emit with StarterDeck
    d = StarterDeck("ROUNDTRIP")
    d.mat_law105(m1)
    out_path = tmp_path / "emitted_0000.rad"
    d.write(str(out_path))

    # Read back emitted deck
    raw_emitted = read_deck(out_path)
    model2 = Model()
    log2 = MessageLog()
    read_starter_deck(raw_emitted, model2, log2)
    assert not log2.errors

    m2 = model2.mat_law105s[1051]
    assert m2.rho0 == pytest.approx(m1.rho0)
    assert m2.bulk == pytest.approx(m1.bulk)
    assert m2.gas_d == pytest.approx(m1.gas_d)
    assert m2.gas_eg == pytest.approx(m1.gas_eg)
    assert m2.gr == pytest.approx(m1.gr)
    assert m2.c == pytest.approx(m1.c)
    assert m2.alpha == pytest.approx(m1.alpha)
    assert m2.func_b == m1.func_b
    assert m2.scale_b == pytest.approx(m1.scale_b)
    assert m2.c1 == pytest.approx(m1.c1)


def test_mat_powder_burn_free_format(tmp_path: Path):
    """Test reading free-format /MAT/POWDER_BURN deck."""
    deck_src = """\
# OpenRadioss Starter Deck
/BEGIN
Test Free POWDER_BURN
/MAT/POWDER_BURN/2051
Powder Free Format
1.65e-06 1.65e-06
12000.0 0.2 0.05
0.6e-06 4200.0
0.07 0.85 0.95
21 1.25 1.75
31 1.0 1.0 1800.0 50.0
/END
"""
    model, log = _parse_deck_string(tmp_path, deck_src)
    assert not log.errors, f"Errors parsing free POWDER_BURN: {log.errors}"
    assert 2051 in model.mat_law105s

    m = model.mat_law105s[2051]
    assert m.rho0 == pytest.approx(1.65e-06)
    assert m.refer_rho == pytest.approx(1.65e-06)
    assert m.bulk == pytest.approx(12000.0)
    assert m.p0 == pytest.approx(0.2)
    assert m.psh == pytest.approx(0.05)
    assert m.gas_d == pytest.approx(0.6e-06)
    assert m.gas_eg == pytest.approx(4200.0)
    assert m.gr == pytest.approx(0.07)
    assert m.c == pytest.approx(0.85)
    assert m.alpha == pytest.approx(0.95)
    assert m.func_b == 21
    assert m.scale_b == pytest.approx(1.25)
    assert m.scale_p == pytest.approx(1.75)
    assert m.func_gam == 31
    assert m.c1 == pytest.approx(1800.0)
    assert m.c2 == pytest.approx(50.0)


def test_starter_deck_powder_synonyms():
    """Verify mat_powder_burn and mat_powderburn emitter synonyms."""
    d = StarterDeck("SYNONYMS")
    d.mat_powder_burn(1, "PowderBurnName", rho=1.7e-6, bulk=11000.0, p0=0.15, d=0.4e-6, eg=3800.0, gr=0.06, c1=2200.0)
    d.mat_powderburn(2, "PowderBurnName2", rho=1.75e-6, bulk=11500.0, p0=0.16, d=0.45e-6, eg=3900.0, gr=0.065, c1=2300.0)

    lines = d.lines
    assert any("/MAT/POWDER_BURN/1" in ln for ln in lines)
    assert any("/MAT/POWDERBURN/2" in ln for ln in lines)
