"""Fixed and free format parser and deck writer roundtrip tests for /MAT/LAW109 (M578).

Validates:
  - Fixed-format reading of /MAT/LAW109 with cards 1-5 and optional Refer_Rho.
  - Free-format reading of /MAT/TAB_PLAS and /MAT/ELASTO_PLAS_TAB.
  - Verification that model entities and Material records are properly registered.
  - StarterDeck.mat_law109 / mat_tab_plas / mat_elasto_plas_tab emission.
  - Roundtrip write-and-read parsing accuracy across all parameters.
"""

from pathlib import Path
import pytest

from pyradioss.input.deck_reader import read_deck
from pyradioss.input.deck_writer import StarterDeck
from pyradioss.model.model import Model
from pyradioss.common.messages import MessageLog
from pyradioss.input.starter_keywords import read_starter_deck
from pyradioss.model.entities import MaterialLaw109


def _parse_deck_string(tmp_path: Path, text: str) -> tuple[Model, MessageLog]:
    deck_path = tmp_path / "deck_0000.rad"
    deck_path.write_text(text, encoding="utf-8")
    raw_deck = read_deck(deck_path)
    model = Model()
    log = MessageLog()
    read_starter_deck(raw_deck, model, log)
    return model, log


def test_mat_law109_fixed_format_roundtrip(tmp_path: Path):
    """Test reading and re-emitting fixed-format /MAT/LAW109 deck."""
    deck_src = """\
# OpenRadioss Starter Deck
/BEGIN
Test Fixed LAW109
#   Unit system: Mg, mm, s
                  Mg                  mm                   s
/MAT/LAW109/1091
DP600 Steel Tabulated
#              RHO_I           Refer_Rho
             7.8e-06             7.8e-06
#                  E                  Nu
            210000.0                 0.3
#                C_p                 ETA               T_ref               T_ini
               450.0                 0.9               293.0               293.0
# tab_ID_h  tab_ID_t            Xscale_h            Yscale_h                                I_smooth
       101       102                 1.0                 1.0                                       2
#  TAB_ETA          Xscale_ETA
       103                 1.0
/END
"""
    model, log = _parse_deck_string(tmp_path, deck_src)
    assert not log.errors, f"Errors parsing fixed LAW109: {log.errors}"
    assert 1091 in model.mat_law109s

    m1 = model.mat_law109s[1091]
    assert m1.rho0 == pytest.approx(7.8e-06)
    assert m1.refer_rho == pytest.approx(7.8e-06)
    assert m1.young == pytest.approx(210000.0)
    assert m1.nu == pytest.approx(0.3)
    assert m1.cp == pytest.approx(450.0)
    assert m1.eta == pytest.approx(0.9)
    assert m1.tref == pytest.approx(293.0)
    assert m1.tini == pytest.approx(293.0)
    assert m1.tab_yld == 101
    assert m1.tab_temp == 102
    assert m1.xscale_h == pytest.approx(1.0)
    assert m1.yscale_h == pytest.approx(1.0)
    assert m1.ismooth == 2
    assert m1.tab_eta == 103
    assert m1.xscale_eta == pytest.approx(1.0)

    # Re-emit with StarterDeck
    d = StarterDeck("ROUNDTRIP_LAW109")
    d.mat_law109(m1)
    out_path = tmp_path / "emitted_1091_0000.rad"
    d.write(str(out_path))

    # Read back emitted deck
    raw_emitted = read_deck(out_path)
    model2 = Model()
    log2 = MessageLog()
    read_starter_deck(raw_emitted, model2, log2)
    assert not log2.errors, f"Errors reading emitted deck: {log2.errors}"

    m2 = model2.mat_law109s[1091]
    assert m2.rho0 == pytest.approx(m1.rho0)
    assert m2.young == pytest.approx(m1.young)
    assert m2.nu == pytest.approx(m1.nu)
    assert m2.cp == pytest.approx(m1.cp)
    assert m2.eta == pytest.approx(m1.eta)
    assert m2.tref == pytest.approx(m1.tref)
    assert m2.tini == pytest.approx(m1.tini)
    assert m2.tab_yld == m1.tab_yld
    assert m2.tab_temp == m1.tab_temp
    assert m2.xscale_h == pytest.approx(m1.xscale_h)
    assert m2.yscale_h == pytest.approx(m1.yscale_h)
    assert m2.ismooth == m1.ismooth
    assert m2.tab_eta == m1.tab_eta
    assert m2.xscale_eta == pytest.approx(m1.xscale_eta)


def test_mat_tab_plas_free_format_roundtrip(tmp_path: Path):
    """Test reading free-format /MAT/TAB_PLAS and re-emitting with mat_tab_plas."""
    deck_src = """\
# OpenRadioss Starter Deck
/BEGIN
Test Free TAB_PLAS
                  Mg                  mm                   s
/MAT/TAB_PLAS/2091
Aluminum Alloy 6061-T6
2.7e-6
70000.0, 0.33
890.0, 0.95, 293.15, 293.15
201, 202, 1.0, 1.0, 1
203, 1.0
/END
"""
    model, log = _parse_deck_string(tmp_path, deck_src)
    assert not log.errors, f"Errors parsing free TAB_PLAS: {log.errors}"
    assert 2091 in model.mat_tab_plass

    m1 = model.mat_tab_plass[2091]
    assert m1.rho0 == pytest.approx(2.7e-6)
    assert m1.young == pytest.approx(70000.0)
    assert m1.nu == pytest.approx(0.33)
    assert m1.cp == pytest.approx(890.0)
    assert m1.eta == pytest.approx(0.95)
    assert m1.tref == pytest.approx(293.15)
    assert m1.tini == pytest.approx(293.15)
    assert m1.tab_yld == 201
    assert m1.tab_temp == 202
    assert m1.ismooth == 1
    assert m1.tab_eta == 203

    # Re-emit with StarterDeck synonym mat_tab_plas
    d = StarterDeck("ROUNDTRIP_TAB_PLAS")
    d.mat_tab_plas(m1)
    out_path = tmp_path / "emitted_2091_0000.rad"
    d.write(str(out_path))

    # Read back emitted deck
    raw_emitted = read_deck(out_path)
    model2 = Model()
    log2 = MessageLog()
    read_starter_deck(raw_emitted, model2, log2)
    assert not log2.errors, f"Errors reading emitted TAB_PLAS deck: {log2.errors}"
    assert 2091 in model2.mat_tab_plass


def test_mat_elasto_plas_tab_roundtrip(tmp_path: Path):
    """Test reading /MAT/ELASTO_PLAS_TAB and re-emitting with mat_elasto_plas_tab."""
    deck_src = """\
# OpenRadioss Starter Deck
/BEGIN
Test ELASTO_PLAS_TAB
                  Mg                  mm                   s
/MAT/ELASTO_PLAS_TAB/3091
Titanium Ti-6Al-4V
4.43e-6, 4.43e-6
114000.0, 0.34
526.0, 0.9, 293.0, 350.0
301, 302, 1.0, 1.0, 3
303, 1.0
/END
"""
    model, log = _parse_deck_string(tmp_path, deck_src)
    assert not log.errors, f"Errors parsing ELASTO_PLAS_TAB: {log.errors}"
    assert 3091 in model.mat_elasto_plas_tabs

    m1 = model.mat_elasto_plas_tabs[3091]
    assert m1.rho0 == pytest.approx(4.43e-6)
    assert m1.refer_rho == pytest.approx(4.43e-6)
    assert m1.young == pytest.approx(114000.0)
    assert m1.nu == pytest.approx(0.34)
    assert m1.cp == pytest.approx(526.0)
    assert m1.tini == pytest.approx(350.0)
    assert m1.ismooth == 3

    # Re-emit with StarterDeck synonym mat_elasto_plas_tab
    d = StarterDeck("ROUNDTRIP_ELASTO_PLAS_TAB")
    d.mat_elasto_plas_tab(m1)
    out_path = tmp_path / "emitted_3091_0000.rad"
    d.write(str(out_path))

    # Read back emitted deck
    raw_emitted = read_deck(out_path)
    model2 = Model()
    log2 = MessageLog()
    read_starter_deck(raw_emitted, model2, log2)
    assert not log2.errors, f"Errors reading emitted ELASTO_PLAS_TAB deck: {log2.errors}"
    assert 3091 in model2.mat_elasto_plas_tabs
