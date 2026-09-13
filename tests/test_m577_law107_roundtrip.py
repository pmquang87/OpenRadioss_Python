"""Fixed and free format parser and deck writer roundtrip tests for /MAT/LAW107 (M577).

Validates:
  - Fixed-format reading of /MAT/LAW107 and /MAT/PAPER_LIGHT.
  - Tabulated vs analytical parameters (ITAB=0 vs ITAB=1).
  - Free-format reading of /MAT/PLAS_PAPER_LIGHT and /MAT/PFEIFFER.
  - Optional Refer_Rho support on Card 1.
  - StarterDeck.mat_law107 / mat_paper_light / mat_plas_paper_light / mat_pfeiffer emission and roundtrip parsing.
"""

from pathlib import Path
import pytest

from pyradioss.input.deck_reader import read_deck
from pyradioss.input.deck_writer import StarterDeck
from pyradioss.model.model import Model
from pyradioss.common.messages import MessageLog
from pyradioss.input.starter_keywords import read_starter_deck
from pyradioss.model.entities import MaterialLaw107


def _parse_deck_string(tmp_path: Path, text: str) -> tuple[Model, MessageLog]:
    deck_path = tmp_path / "deck_0000.rad"
    deck_path.write_text(text, encoding="utf-8")
    raw_deck = read_deck(deck_path)
    model = Model()
    log = MessageLog()
    read_starter_deck(raw_deck, model, log)
    return model, log


def test_mat_law107_fixed_format_roundtrip(tmp_path: Path):
    """Test reading and re-emitting fixed-format analytical /MAT/LAW107 deck."""
    deck_src = """\
# OpenRadioss Starter Deck
/BEGIN
Test Fixed LAW107
#   Unit system: Mg, mm, s
                  Mg                  mm                   s
/MAT/LAW107/1071
Paper Light Analytical
#              RHO_I           Refer_Rho
             7.5e-07             7.5e-07
#                 E1                  E2                  E3      Ires      Itab   Ismooth
              7000.0              3500.0               100.0         2         0         1
#               nu21                 G12                 G23                 G13
                0.15              1800.0                40.0                40.0
#                XI1                 XI2                 g1c                  d1                  d2
                 0.5                 0.5                 1.0                 0.1                 0.2
#                 k1                  k2                  k3
                 0.1                -0.1                 0.2
#                 k4                  k5                  k6
                 0.3                 0.4                 0.5
#              SIGY1               CINI1                  S1
                25.0                10.0                 2.0
#              SIGY2               CINI2                  S2
                15.0                 8.0                 1.5
#             SIGY1C              CINI1C                 S1C
                30.0                12.0                 2.5
#             SIGY2C              CINI2C                 S2C
                20.0                10.0                 2.0
#              SIGYT               CINIT                  ST
                10.0                 5.0                 1.0
/END
"""
    model, log = _parse_deck_string(tmp_path, deck_src)
    assert not log.errors, f"Errors parsing fixed LAW107: {log.errors}"
    assert 1071 in model.mat_law107s

    m1 = model.mat_law107s[1071]
    assert m1.rho == pytest.approx(7.5e-07)
    assert m1.rhor == pytest.approx(7.5e-07)
    assert m1.e1 == pytest.approx(7000.0)
    assert m1.e2 == pytest.approx(3500.0)
    assert m1.e3 == pytest.approx(100.0)
    assert m1.ires == 2
    assert m1.itab == 0
    assert m1.ismooth == 1
    assert m1.nu21 == pytest.approx(0.15)
    assert m1.g12 == pytest.approx(1800.0)
    assert m1.g23 == pytest.approx(40.0)
    assert m1.g13 == pytest.approx(40.0)
    assert m1.g31 == pytest.approx(40.0)
    assert m1.xi1 == pytest.approx(0.5)
    assert m1.xi2 == pytest.approx(0.5)
    assert m1.g1c == pytest.approx(1.0)
    assert m1.d1 == pytest.approx(0.1)
    assert m1.d2 == pytest.approx(0.2)
    assert m1.k1 == pytest.approx(0.1)
    assert m1.k2 == pytest.approx(-0.1)
    assert m1.k3 == pytest.approx(0.2)
    assert m1.k4 == pytest.approx(0.3)
    assert m1.k5 == pytest.approx(0.4)
    assert m1.k6 == pytest.approx(0.5)
    assert m1.sigy1 == pytest.approx(25.0)
    assert m1.cini1 == pytest.approx(10.0)
    assert m1.s1 == pytest.approx(2.0)
    assert m1.sigyt == pytest.approx(10.0)

    # Re-emit with StarterDeck
    d = StarterDeck("ROUNDTRIP")
    d.mat_law107(m1)
    out_path = tmp_path / "emitted_0000.rad"
    d.write(str(out_path))

    # Read back emitted deck
    raw_emitted = read_deck(out_path)
    model2 = Model()
    log2 = MessageLog()
    read_starter_deck(raw_emitted, model2, log2)
    assert not log2.errors, f"Errors reading emitted deck: {log2.errors}"

    m2 = model2.mat_law107s[1071]
    assert m2.rho == pytest.approx(m1.rho)
    assert m2.e1 == pytest.approx(m1.e1)
    assert m2.e2 == pytest.approx(m1.e2)
    assert m2.e3 == pytest.approx(m1.e3)
    assert m2.nu21 == pytest.approx(m1.nu21)
    assert m2.g12 == pytest.approx(m1.g12)
    assert m2.itab == m1.itab
    assert m2.ires == m1.ires
    assert m2.k1 == pytest.approx(m1.k1)
    assert m2.k2 == pytest.approx(m1.k2)
    assert m2.sigy1 == pytest.approx(m1.sigy1)
    assert m2.cini1 == pytest.approx(m1.cini1)
    assert m2.sigyt == pytest.approx(m1.sigyt)


def test_mat_paper_light_tabulated_roundtrip(tmp_path: Path):
    """Test reading and re-emitting fixed-format tabulated /MAT/PAPER_LIGHT deck."""
    deck_src = """\
# OpenRadioss Starter Deck
/BEGIN
Test Tabulated PAPER_LIGHT
/MAT/PAPER_LIGHT/1072
Tabulated Paper Card
#              RHO_I           Refer_Rho
             8.0e-07             8.0e-07
#                 E1                  E2                  E3      Ires      Itab   Ismooth
              8000.0              4000.0               120.0         1         1         1
#               nu21                 G12                 G23                 G13
                0.16              2000.0                50.0                50.0
#                XI1                 XI2                 g1c                  d1                  d2
                 0.6                 0.6                 1.1                0.12                0.22
#                 k1                  k2                  k3
                0.15               -0.08                0.25
#                 k4                  k5                  k6
                0.35                0.45                0.55
#           TAB_YLD1         MAT_Xscale1         MAT_Yscale1
                 101                 1.2                 1.5
#           TAB_YLD2         MAT_Xscale2         MAT_Yscale2
                 102                 1.2                 1.5
#          TAB_YLD1C        MAT_Xscale1C        MAT_Yscale1C
                 103                 1.2                 1.5
#          TAB_YLD2C        MAT_Xscale2C        MAT_Yscale2C
                 104                 1.2                 1.5
#           TAB_YLDT         MAT_XscaleT         MAT_YscaleT
                 105                 1.0                 1.0
/END
"""
    model, log = _parse_deck_string(tmp_path, deck_src)
    assert not log.errors, f"Errors parsing tabulated PAPER_LIGHT: {log.errors}"
    assert 1072 in model.mat_paper_lights

    m1 = model.mat_paper_lights[1072]
    assert m1.itab == 1
    assert m1.ires == 1
    assert m1.tab_yld1 == 101
    assert m1.tab_yld2 == 102
    assert m1.tab_yld1c == 103
    assert m1.tab_yld2c == 104
    assert m1.tab_yldt == 105
    assert m1.xscale1 == pytest.approx(1.2)
    assert m1.yscale1 == pytest.approx(1.5)

    # Re-emit with StarterDeck.mat_paper_light
    d = StarterDeck("TAB_ROUNDTRIP")
    d.mat_paper_light(m1)
    out_path = tmp_path / "emitted_tab_0000.rad"
    d.write(str(out_path))

    # Read back emitted deck
    raw_emitted = read_deck(out_path)
    model2 = Model()
    log2 = MessageLog()
    read_starter_deck(raw_emitted, model2, log2)
    assert not log2.errors, f"Errors reading emitted deck: {log2.errors}"

    m2 = model2.mat_law107s[1072]
    assert m2.itab == 1
    assert m2.ires == 1
    assert m2.tab_yld1 == 101
    assert m2.tab_yld2 == 102
    assert m2.tab_yld1c == 103
    assert m2.tab_yld2c == 104
    assert m2.tab_yldt == 105
    assert m2.xscale1 == pytest.approx(1.2)
    assert m2.yscale1 == pytest.approx(1.5)


def test_mat_plas_paper_light_and_pfeiffer_free_format(tmp_path: Path):
    """Test reading free-format /MAT/PLAS_PAPER_LIGHT and /MAT/PFEIFFER."""
    deck_src = """\
# OpenRadioss Starter Deck
/BEGIN
Test Free Format
/MAT/PLAS_PAPER_LIGHT/2001
Plas Paper Free
7.2e-7 7.2e-7
6500.0 3200.0 90.0 2 0 1
0.14 1700.0 35.0 35.0
0.5 0.5 1.0 0.1 0.2
0.1 -0.05 0.2
0.3 0.4 0.5
22.0 8.0 1.8
14.0 6.0 1.4
28.0 10.0 2.2
18.0 7.0 1.6
9.0 4.0 1.0
/MAT/PFEIFFER/2002
Pfeiffer Free
7.8e-7 7.8e-7
7200.0 3600.0 110.0 1 1 1
0.16 1900.0 42.0 42.0
0.6 0.6 1.0 0.15 0.25
0.12 -0.06 0.22
0.32 0.42 0.52
201 1.1 1.1
202 1.1 1.1
203 1.1 1.1
204 1.1 1.1
205 1.0 1.0
/END
"""
    model, log = _parse_deck_string(tmp_path, deck_src)
    assert not log.errors, f"Errors parsing free format: {log.errors}"
    assert 2001 in model.mat_plas_paper_lights
    assert 2002 in model.mat_pfeiffers

    m1 = model.mat_plas_paper_lights[2001]
    assert m1.rho == pytest.approx(7.2e-7)
    assert m1.e1 == pytest.approx(6500.0)
    assert m1.sigy1 == pytest.approx(22.0)
    assert m1.k1 == pytest.approx(0.1)

    m2 = model.mat_pfeiffers[2002]
    assert m2.rho == pytest.approx(7.8e-7)
    assert m2.e1 == pytest.approx(7200.0)
    assert m2.itab == 1
    assert m2.tab_yld1 == 201
    assert m2.tab_yldt == 205
    assert m2.xscale1 == pytest.approx(1.1)


def test_starter_deck_emitter_synonyms_and_conv_mat():
    """Verify StarterDeck synonym emitter methods and _conv_mat dispatch."""
    d = StarterDeck("SYNONYMS")
    d.mat_plas_paper_light(3001, "PlasPaper", rho=7.0e-7, young1=6000.0, young2=3000.0, young3=80.0,
                           nu21=0.15, g12=1500.0, g23=30.0, g13=30.0,
                           k1=0.1, k2=-0.05, sigy1=20.0, cini1=5.0, s1=1.5)
    d.mat_pfeiffer(3002, "Pfeiffer", rho=7.1e-7, young1=6100.0, young2=3100.0, young3=85.0,
                   nu21=0.15, g12=1550.0, g23=32.0, g13=32.0,
                   itab=1, tab_yld1=501, tab_yld2=502)

    lines = d.lines
    assert any("/MAT/PLAS_PAPER_LIGHT/3001" in ln for ln in lines)
    assert any("/MAT/PFEIFFER/3002" in ln for ln in lines)

    # Also test mat_law107 with MaterialLaw107 instance
    mat_obj = MaterialLaw107(mat_id=4001, mat_name="ObjConv", rho=7.3e-7,
                             e1=6200.0, e2=3100.0, e3=90.0, nu12=0.32,
                             g12=1600.0, g23=35.0, g13=35.0)
    d2 = StarterDeck("CONV")
    d2.mat_law107(mat_obj)
    assert any("/MAT/LAW107/4001" in ln for ln in d2.lines)
