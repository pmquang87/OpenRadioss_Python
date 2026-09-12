"""
Milestone M561: /MAT/LAW73 (/MAT/BARLAT2000, /MAT/LAW73, /MAT/HILL_THERM, /MAT/THERM_HILL)
Exhaustive Roundtrip, Serialization, Boundary Cases & Negative Validation Auditor Suite.

Fortran origins:
  - starter/source/materials/mat/mat073/hm_read_mat73.F (card reader, parameters, default fallbacks)
  - engine/source/materials/mat/mat073/sigeps73c.F (shell plane stress return mapping, back-stress update)
  - engine/source/output/restart/wrrestp.F & rdresb.F
  - starter/source/restart/ddsplit/wrrest.F
  - hm_cfg_files/config/CFG/radioss140/MAT/matl73_73.cfg & matl73_BARLAT2000.cfg
"""

from __future__ import annotations

import math
from pathlib import Path
import pickle
from typing import Any, Dict, List, Sequence, Tuple
import numpy as np
import pytest

from pyradioss.common.messages import MessageLog
from pyradioss.input.card_layouts import CARD_LAYOUTS, split_fixed
from pyradioss.input.deck_reader import read_deck
from pyradioss.input.deck_writer import StarterDeck
from pyradioss.input.starter_keywords import (
    parse_starter_deck,
    read_mat_law73,
    read_starter_deck,
)
from pyradioss.materials.law73_hill_therm import (
    Law73Params,
    build_law73,
    shell_update,
    extra_shapes,
    sound_speed as law73_sound_speed,
)
from pyradioss.model.entities import (
    MatHillTherm,
    MatLaw73,
    MatThermalHill,
    MatThermHill,
    Material,
    Part,
)
from pyradioss.model.model import Model
from pyradioss.starter.checks import (
    _ALLOWED_LAWS,
    _MAT_CHECKS,
    check_mat_law73,
    check_materials,
    check_model,
)
from pyradioss.starter.restart import read_restart, write_restart
from pyradioss.starter.starter import run_starter


# ============================================================================
# Helpers
# ============================================================================

def _parse_deck_str(tmp_path: Path, text: str, name: str = "TEST_LAW73") -> Tuple[Model, MessageLog]:
    """Write deck text to file and parse into Model and MessageLog."""
    deck_path = tmp_path / f"{name}_0000.rad"
    deck_path.write_text(text.strip() + "\n", encoding="utf-8")
    blocks = read_deck(str(deck_path))
    model = Model()
    log = MessageLog()
    parse_starter_deck(blocks, model, log)
    return model, log


def _assert_law73_all_fields_exact(
    m: MatLaw73,
    mat: Material | None = None,
    *,
    rho: float,
    e: float,
    nu: float,
    rhor: float | None = None,
    ifunce: int = 0,
    einf: float = 0.0,
    ce: float = 0.0,
    r00: float = 1.0,
    r45: float = 1.0,
    r90: float = 1.0,
    chard: float = 0.0,
    iyield: int = 0,
    eps_max: float = 1.0e30,
    epsr1: float = 1.0e30,
    epsr2: float = 2.0e30,
    table_id: int = 0,
    fscale: float = 1.0,
    pscale: float = 1.0,
    t0: float = 293.0,
    rhocp: float = 0.0,
    title: str = "",
) -> None:
    """Assert exact (floating-point tolerance) equality for all LAW73 parameters and derived properties."""
    expected_rhor = rhor if rhor is not None else rho

    # Primary Dataclass Fields
    assert m.rho == pytest.approx(rho, rel=1e-6, abs=1e-12)
    assert m.rho0 == pytest.approx(rho, rel=1e-6, abs=1e-12)
    assert m.rhor == pytest.approx(expected_rhor, rel=1e-6, abs=1e-12)
    assert m.refer_rho == pytest.approx(expected_rhor, rel=1e-6, abs=1e-12)
    assert m.e == pytest.approx(e, rel=1e-6, abs=1e-12)
    assert m.E == pytest.approx(e, rel=1e-6, abs=1e-12)
    assert m.nu == pytest.approx(nu, rel=1e-6, abs=1e-12)
    assert m.Nu == pytest.approx(nu, rel=1e-6, abs=1e-12)

    assert m.ifunce == ifunce
    assert m.yr_fun == ifunce
    assert m.einf == pytest.approx(einf, rel=1e-6, abs=1e-12)
    assert m.efib == pytest.approx(einf, rel=1e-6, abs=1e-12)
    assert m.ce == pytest.approx(ce, rel=1e-6, abs=1e-12)
    assert m.c == pytest.approx(ce, rel=1e-6, abs=1e-12)

    assert m.r00 == pytest.approx(r00, rel=1e-6, abs=1e-12)
    assert m.r0 == pytest.approx(r00, rel=1e-6, abs=1e-12)
    assert m.r45 == pytest.approx(r45, rel=1e-6, abs=1e-12)
    assert m.r90 == pytest.approx(r90, rel=1e-6, abs=1e-12)

    assert m.chard == pytest.approx(chard, rel=1e-6, abs=1e-12)
    assert m.fisokin == pytest.approx(chard, rel=1e-6, abs=1e-12)
    assert m.c_hard == pytest.approx(chard, rel=1e-6, abs=1e-12)
    assert m.iyield == iyield

    assert m.eps_max == pytest.approx(eps_max, rel=1e-6, abs=1e-12)
    assert m.epsp_max == pytest.approx(eps_max, rel=1e-6, abs=1e-12)
    assert m.epsr1 == pytest.approx(epsr1, rel=1e-6, abs=1e-12)
    assert m.epst1 == pytest.approx(epsr1, rel=1e-6, abs=1e-12)
    assert m.epsr2 == pytest.approx(epsr2, rel=1e-6, abs=1e-12)
    assert m.epst2 == pytest.approx(epsr2, rel=1e-6, abs=1e-12)

    assert m.table_id == table_id
    assert m.fun_a1 == table_id
    assert m.fscale == pytest.approx(fscale, rel=1e-6, abs=1e-12)
    assert m.pscale == pytest.approx(pscale, rel=1e-6, abs=1e-12)

    assert m.t0 == pytest.approx(t0, rel=1e-6, abs=1e-12)
    assert m.t_initial == pytest.approx(t0, rel=1e-6, abs=1e-12)
    assert m.rhocp == pytest.approx(rhocp, rel=1e-6, abs=1e-12)
    assert m.spheat == pytest.approx(rhocp, rel=1e-6, abs=1e-12)

    if title:
        assert m.title == title

    # Derived Moduli & Hill Constants
    expected_g = 0.5 * e / (1.0 + nu)
    assert m.G == pytest.approx(expected_g, rel=1e-6, abs=1e-12)

    r_bar = 0.25 * (r00 + 2.0 * r45 + r90)
    h_bar = r_bar / (1.0 + r_bar)
    a01 = h_bar * (1.0 + 1.0 / r00)
    a02 = h_bar * (1.0 + 1.0 / r90)
    a03 = 2.0 * h_bar
    a12 = (2.0 * r45 + 1.0) * (a01 + a02 - a03)
    if iyield > 0:
        a02 /= a01
        a03 /= a01
        a12 /= a01
        a01 = 1.0

    assert m.A01 == pytest.approx(a01, rel=1e-6, abs=1e-12)
    assert m.A02 == pytest.approx(a02, rel=1e-6, abs=1e-12)
    assert m.A03 == pytest.approx(a03, rel=1e-6, abs=1e-12)
    assert m.A12 == pytest.approx(a12, rel=1e-6, abs=1e-12)

    # Sound speeds: bulk/acoustic sound_speed and shell plane-stress sound_speed_shell
    expected_c_bulk = math.sqrt(e / rho)
    expected_c_shell = math.sqrt(e / (rho * max(1.0 - nu**2, 1e-15)))
    assert m.sound_speed == pytest.approx(expected_c_bulk, rel=1e-6, abs=1e-12)
    assert m.sound_speed_shell == pytest.approx(expected_c_shell, rel=1e-6, abs=1e-12)

    # Validate Material entity in model.materials
    if mat is not None:
        assert mat.law == 73
        assert mat.rho0 == pytest.approx(rho, rel=1e-6, abs=1e-12)
        p = mat.params
        assert p["rho0"] == pytest.approx(rho, rel=1e-6, abs=1e-12)
        assert p["refer_rho"] == pytest.approx(expected_rhor, rel=1e-6, abs=1e-12)
        assert p["e"] == pytest.approx(e, rel=1e-6, abs=1e-12)
        assert p["E"] == pytest.approx(e, rel=1e-6, abs=1e-12)
        assert p["nu"] == pytest.approx(nu, rel=1e-6, abs=1e-12)
        assert p["ifunce"] == ifunce
        assert p["einf"] == pytest.approx(einf, rel=1e-6, abs=1e-12)
        assert p["ce"] == pytest.approx(ce, rel=1e-6, abs=1e-12)
        assert p["r00"] == pytest.approx(r00, rel=1e-6, abs=1e-12)
        assert p["r45"] == pytest.approx(r45, rel=1e-6, abs=1e-12)
        assert p["r90"] == pytest.approx(r90, rel=1e-6, abs=1e-12)
        assert p["chard"] == pytest.approx(chard, rel=1e-6, abs=1e-12)
        assert p["iyield"] == iyield
        assert p["eps_max"] == pytest.approx(eps_max, rel=1e-6, abs=1e-12)
        assert p["epsr1"] == pytest.approx(epsr1, rel=1e-6, abs=1e-12)
        assert p["epsr2"] == pytest.approx(epsr2, rel=1e-6, abs=1e-12)
        assert p["table_id"] == table_id
        assert p["fscale"] == pytest.approx(fscale, rel=1e-6, abs=1e-12)
        assert p["pscale"] == pytest.approx(pscale, rel=1e-6, abs=1e-12)
        assert p["t0"] == pytest.approx(t0, rel=1e-6, abs=1e-12)
        assert p["rhocp"] == pytest.approx(rhocp, rel=1e-6, abs=1e-12)
        if title:
            assert mat.title == title


# ============================================================================
# 1. Exact Card Generation & Fixed vs Free Formats
# ============================================================================

class TestLaw73CardGeneration:
    """Audit StarterDeck.mat_law73 exact 7-card fixed-format and free-format generation."""

    def test_fixed_format_card_columns_match_cfg_layouts(self):
        """Verify that all 7 cards emitted by StarterDeck.mat_law73 match exact CARD_LAYOUTS widths."""
        deck = StarterDeck("CARD_TEST")
        deck.mat_law73(
            mid=73,
            title="Orthotropic Sheet",
            rho=2.7e-9,
            rhor=2.75e-9,
            e=70000.0,
            nu=0.33,
            ifunce=2,
            einf=45000.0,
            ce=12.5,
            r00=1.25,
            r45=1.45,
            r90=1.85,
            chard=0.35,
            iyield=1,
            eps_max=0.55,
            epsr1=0.25,
            epsr2=0.45,
            table_id=202,
            fscale=1.15,
            pscale=0.95,
            t0=300.0,
            rhocp=2.45e-3,
            fixed_format=True,
        )

        lines = deck.lines
        header_idx = lines.index("/MAT/LAW73/73")
        assert lines[header_idx + 1] == "Orthotropic Sheet"

        # Cards 1 to 7
        card1 = lines[header_idx + 2]
        card2 = lines[header_idx + 3]
        card3 = lines[header_idx + 4]
        card4 = lines[header_idx + 5]
        card5 = lines[header_idx + 6]
        card6 = lines[header_idx + 7]
        card7 = lines[header_idx + 8]

        # Card 1: [20, 20] (with rhor)
        c1_fields = split_fixed(card1, [20, 20])
        assert float(c1_fields[0]) == pytest.approx(2.7e-9)
        assert float(c1_fields[1]) == pytest.approx(2.75e-9)

        # Card 2: [20, 20] -> E, Nu
        c2_fields = split_fixed(card2, list(CARD_LAYOUTS["MAT_LAW73_2"]))
        assert float(c2_fields[0]) == pytest.approx(70000.0)
        assert float(c2_fields[1]) == pytest.approx(0.33)

        # Card 3: [10, 10, 20, 20] -> Yr_fun, blank, Einf, C
        c3_fields = split_fixed(card3, list(CARD_LAYOUTS["MAT_LAW73_3"]))
        assert int(c3_fields[0]) == 2
        assert c3_fields[1] == ""  # blank field
        assert float(c3_fields[2]) == pytest.approx(45000.0)
        assert float(c3_fields[3]) == pytest.approx(12.5)

        # Card 4: [20, 20, 20, 20, 10, 10] -> R00, R45, R90, CHard, Iyield
        c4_fields = split_fixed(card4, list(CARD_LAYOUTS["MAT_LAW73_4"]))
        assert float(c4_fields[0]) == pytest.approx(1.25)
        assert float(c4_fields[1]) == pytest.approx(1.45)
        assert float(c4_fields[2]) == pytest.approx(1.85)
        assert float(c4_fields[3]) == pytest.approx(0.35)
        assert int(c4_fields[4]) == 1

        # Card 5: [20, 20, 20] -> EPS_max, EPST1, EPST2
        c5_fields = split_fixed(card5, list(CARD_LAYOUTS["MAT_LAW73_5"]))
        assert float(c5_fields[0]) == pytest.approx(0.55)
        assert float(c5_fields[1]) == pytest.approx(0.25)
        assert float(c5_fields[2]) == pytest.approx(0.45)

        # Card 6: [10, 10, 20, 20] -> Table_ID, blank, Fscale, Pscale
        c6_fields = split_fixed(card6, list(CARD_LAYOUTS["MAT_LAW73_6"]))
        assert int(c6_fields[0]) == 202
        assert c6_fields[1] == ""  # blank field
        assert float(c6_fields[2]) == pytest.approx(1.15)
        assert float(c6_fields[3]) == pytest.approx(0.95)

        # Card 7: [20, 20] -> T0, Rho_Cp
        c7_fields = split_fixed(card7, list(CARD_LAYOUTS["MAT_LAW73_7"]))
        assert float(c7_fields[0]) == pytest.approx(300.0)
        assert float(c7_fields[1]) == pytest.approx(2.45e-3)

    def test_fixed_format_single_density_card1(self):
        """Card 1 emits only 20 characters if rhor is omitted or equal to rho."""
        deck = StarterDeck("SINGLE_RHO")
        deck.mat_law73(
            mid=1,
            title="Single Rho",
            rho=7.85e-9,
            e=210000.0,
            nu=0.3,
            fixed_format=True,
        )
        card1 = deck.lines[deck.lines.index("/MAT/LAW73/1") + 2]
        assert len(card1) == 20
        assert float(card1.strip()) == pytest.approx(7.85e-9)

    def test_free_format_generation_space_and_comma(self):
        """Verify free-format card emission (space and comma-delimited)."""
        # 1. Space delimited
        deck_space = StarterDeck("FREE_SPACE")
        deck_space.mat_law73(
            mid=1,
            title="Free Space",
            rho=2.7e-9,
            e=70000.0,
            nu=0.33,
            r00=1.1,
            r45=1.2,
            r90=1.3,
            fixed_format=False,
        )
        idx_s = deck_space.lines.index("/MAT/LAW73/1")
        card2_s = deck_space.lines[idx_s + 3]
        tokens_s = card2_s.split()
        assert len(tokens_s) == 2
        assert float(tokens_s[0]) == pytest.approx(70000.0)
        assert float(tokens_s[1]) == pytest.approx(0.33)

        # 2. Comma delimited
        deck_comma = StarterDeck("FREE_COMMA")
        deck_comma.mat_law73(
            mid=2,
            title="Free Comma",
            rho=2.7e-9,
            e=70000.0,
            nu=0.33,
            r00=1.1,
            r45=1.2,
            r90=1.3,
            fixed_format=False,
            comma=True,
        )
        idx_c = deck_comma.lines.index("/MAT/LAW73/2")
        card2_c = deck_comma.lines[idx_c + 3]
        assert "," in card2_c
        tokens_c = [t.strip() for t in card2_c.split(",")]
        assert len(tokens_c) == 2
        assert float(tokens_c[0]) == pytest.approx(70000.0)
        assert float(tokens_c[1]) == pytest.approx(0.33)

    def test_helper_emitter_aliases(self):
        """Verify StarterDeck aliases mat_hill_therm, mat_therm_hill, mat_barlat2000 emit matching headers."""
        deck = StarterDeck("ALIASES")
        deck.mat_hill_therm(1, "Hill Therm", rho=2.7e-9, e=70000.0, nu=0.33)
        deck.mat_therm_hill(2, "Therm Hill", rho=2.7e-9, e=70000.0, nu=0.33)
        deck.mat_barlat2000(3, "Barlat2000", rho=2.7e-9, e=70000.0, nu=0.33)

        assert "/MAT/HILL_THERM/1" in deck.lines
        assert "/MAT/THERM_HILL/2" in deck.lines
        assert "/MAT/BARLAT2000/3" in deck.lines


# ============================================================================
# 2. Complete Deck Roundtrip (Generate -> Write -> Read)
# ============================================================================

class TestLaw73Roundtrip:
    """Audit end-to-end deck generation and parsing parity."""

    def test_fixed_format_full_roundtrip(self, tmp_path: Path):
        """Verify full roundtrip for fixed-format deck with all 7 cards populated."""
        deck = StarterDeck("ROUNDTRIP_FIXED")
        deck.mat_law73(
            mid=73,
            title="Full Orthotropic Law73",
            rho=2.7e-9,
            rhor=2.75e-9,
            e=71000.0,
            nu=0.334,
            ifunce=3,
            einf=48000.0,
            ce=14.2,
            r00=1.12,
            r45=1.34,
            r90=1.56,
            chard=0.45,
            iyield=1,
            eps_max=0.62,
            epsr1=0.28,
            epsr2=0.48,
            table_id=303,
            fscale=1.22,
            pscale=0.88,
            t0=305.0,
            rhocp=2.55e-3,
            fixed_format=True,
        )

        rad_path = tmp_path / "roundtrip_fixed_0000.rad"
        deck.write(str(rad_path))

        model, log = read_starter_deck(str(rad_path))
        assert not log.has_errors
        assert 73 in model.mat_law73s
        assert 73 in model.materials

        _assert_law73_all_fields_exact(
            model.mat_law73s[73],
            model.materials[73],
            rho=2.7e-9,
            rhor=2.75e-9,
            e=71000.0,
            nu=0.334,
            ifunce=3,
            einf=48000.0,
            ce=14.2,
            r00=1.12,
            r45=1.34,
            r90=1.56,
            chard=0.45,
            iyield=1,
            eps_max=0.62,
            epsr1=0.28,
            epsr2=0.48,
            table_id=303,
            fscale=1.22,
            pscale=0.88,
            t0=305.0,
            rhocp=2.55e-3,
            title="Full Orthotropic Law73",
        )

    def test_free_format_space_roundtrip(self, tmp_path: Path):
        """Verify roundtrip for space-delimited free-format deck."""
        deck = StarterDeck("ROUNDTRIP_FREE_SPACE")
        deck.mat_law73(
            mid=10,
            title="Free Space Law73",
            rho=1.5e-9,
            e=120000.0,
            nu=0.28,
            r00=0.85,
            r45=0.95,
            r90=1.15,
            chard=0.20,
            iyield=0,
            eps_max=0.40,
            epsr1=0.15,
            epsr2=0.30,
            t0=295.0,
            rhocp=1.8e-3,
            fixed_format=False,
        )

        rad_path = tmp_path / "roundtrip_free_space_0000.rad"
        deck.write(str(rad_path))

        model, log = read_starter_deck(str(rad_path))
        assert not log.has_errors
        assert 10 in model.mat_law73s

        _assert_law73_all_fields_exact(
            model.mat_law73s[10],
            model.materials[10],
            rho=1.5e-9,
            e=120000.0,
            nu=0.28,
            r00=0.85,
            r45=0.95,
            r90=1.15,
            chard=0.20,
            iyield=0,
            eps_max=0.40,
            epsr1=0.15,
            epsr2=0.30,
            t0=295.0,
            rhocp=1.8e-3,
            title="Free Space Law73",
        )

    def test_free_format_comma_roundtrip(self, tmp_path: Path):
        """Verify roundtrip for comma-delimited free-format deck."""
        deck = StarterDeck("ROUNDTRIP_FREE_COMMA")
        deck.mat_law73(
            mid=20,
            title="Free Comma Law73",
            rho=2.8e-9,
            e=68000.0,
            nu=0.32,
            r00=1.05,
            r45=1.25,
            r90=1.45,
            chard=0.50,
            iyield=1,
            eps_max=0.50,
            epsr1=0.20,
            epsr2=0.40,
            t0=298.0,
            rhocp=2.2e-3,
            fixed_format=False,
            comma=True,
        )

        rad_path = tmp_path / "roundtrip_free_comma_0000.rad"
        deck.write(str(rad_path))

        model, log = read_starter_deck(str(rad_path))
        assert not log.has_errors
        assert 20 in model.mat_law73s

        _assert_law73_all_fields_exact(
            model.mat_law73s[20],
            model.materials[20],
            rho=2.8e-9,
            e=68000.0,
            nu=0.32,
            r00=1.05,
            r45=1.25,
            r90=1.45,
            chard=0.50,
            iyield=1,
            eps_max=0.50,
            epsr1=0.20,
            epsr2=0.40,
            t0=298.0,
            rhocp=2.2e-3,
            title="Free Comma Law73",
        )

    def test_entity_object_passed_to_deck_writer(self, tmp_path: Path):
        """Verify passing MatLaw73 or Material entity object into StarterDeck.mat_law73."""
        orig_entity = MatLaw73(
            id=55,
            title="Entity Object Passed",
            rho=2.7e-9,
            refer_rho=2.72e-9,
            e=69000.0,
            nu=0.31,
            ifunce=1,
            einf=42000.0,
            ce=8.5,
            r00=1.18,
            r45=1.38,
            r90=1.68,
            chard=0.32,
            iyield=0,
            eps_max=0.48,
            epsr1=0.22,
            epsr2=0.38,
            table_id=111,
            fscale=1.08,
            pscale=0.92,
            t0=296.0,
            rhocp=2.35e-3,
        )

        deck = StarterDeck("ENTITY_PASS")
        deck.mat_law73(orig_entity)

        rad_path = tmp_path / "entity_pass_0000.rad"
        deck.write(str(rad_path))

        model, log = read_starter_deck(str(rad_path))
        assert not log.has_errors
        assert 55 in model.mat_law73s

        _assert_law73_all_fields_exact(
            model.mat_law73s[55],
            model.materials[55],
            rho=2.7e-9,
            rhor=2.72e-9,
            e=69000.0,
            nu=0.31,
            ifunce=1,
            einf=42000.0,
            ce=8.5,
            r00=1.18,
            r45=1.38,
            r90=1.68,
            chard=0.32,
            iyield=0,
            eps_max=0.48,
            epsr1=0.22,
            epsr2=0.38,
            table_id=111,
            fscale=1.08,
            pscale=0.92,
            t0=296.0,
            rhocp=2.35e-3,
            title="Entity Object Passed",
        )

    def test_legacy_5_card_format_compatibility(self, tmp_path: Path):
        """Verify legacy 5-card format backward compatibility (M191 layout)."""
        deck_text = """
/MAT/LAW73/99
Legacy 5-Card
2.7e-9              2.7e-9
70000.0             0.33                1.2                 1.4                 1.6
0.25                0.5                 0.2                 0.35                101
1.1                 0.9                 293.0               0.0                 1
0                   0.0                 0.0
"""
        model, log = _parse_deck_str(tmp_path, deck_text, name="LEGACY_5CARD")
        assert not log.has_errors
        assert 99 in model.mat_law73s

        m = model.mat_law73s[99]
        assert m.rho == pytest.approx(2.7e-9)
        assert m.e == pytest.approx(70000.0)
        assert m.nu == pytest.approx(0.33)
        assert m.r00 == pytest.approx(1.2)
        assert m.r45 == pytest.approx(1.4)
        assert m.r90 == pytest.approx(1.6)
        assert m.chard == pytest.approx(0.25)
        assert m.iyield == 1
        assert m.eps_max == pytest.approx(0.5)
        assert m.epsr1 == pytest.approx(0.2)
        assert m.epsr2 == pytest.approx(0.35)
        assert m.table_id == 101
        assert m.fscale == pytest.approx(1.1)
        assert m.pscale == pytest.approx(0.9)
        # Default fallbacks for omitted cards 6-7
        assert m.t0 == pytest.approx(293.0)
        assert m.rhocp == pytest.approx(0.0)

    def test_multi_material_deck_roundtrip(self, tmp_path: Path):
        """Verify multiple LAW73 materials in a single deck roundtrip with complete independence."""
        deck = StarterDeck("MULTI_LAW73")
        deck.mat_law73(1, "Mat 1", rho=2.7e-9, e=70000.0, nu=0.33, r00=1.1, r45=1.2, r90=1.3)
        deck.mat_hill_therm(2, "Mat 2", rho=7.85e-9, e=210000.0, nu=0.30, r00=1.4, r45=1.5, r90=1.6)
        deck.mat_therm_hill(3, "Mat 3", rho=1.8e-9, e=45000.0, nu=0.35, r00=0.9, r45=1.0, r90=1.1)

        rad_path = tmp_path / "multi_law73_0000.rad"
        deck.write(str(rad_path))

        model, log = read_starter_deck(str(rad_path))
        assert not log.has_errors
        assert set(model.mat_law73s.keys()) == {1, 2, 3}
        assert model.mat_law73s[1].e == pytest.approx(70000.0)
        assert model.mat_law73s[2].e == pytest.approx(210000.0)
        assert model.mat_law73s[3].e == pytest.approx(45000.0)


# ============================================================================
# 3. Keyword Synonyms & Container Invariants
# ============================================================================

class TestLaw73Synonyms:
    """Audit all keyword synonyms (/MAT/LAW73, /MAT/HILL_THERM, /MAT/THERM_HILL, /MAT/73)."""

    def test_all_keyword_synonyms_populate_containers(self, tmp_path: Path):
        """Verify that every synonym parses into model.mat_law73s, alias containers, and model.materials."""
        deck_text = """
/MAT/LAW73/1
Law 73 Standard
2.7e-9
70000.0, 0.33
0, 0.0, 0.0
1.1, 1.2, 1.3, 0.2, 1
0.5, 0.2, 0.4
101, 1.0, 1.0
295.0, 2.4e-3

/MAT/HILL_THERM/2
Hill Therm Secondary
2.7e-9
70000.0, 0.33
0, 0.0, 0.0
1.1, 1.2, 1.3, 0.2, 1
0.5, 0.2, 0.4
101, 1.0, 1.0
295.0, 2.4e-3

/MAT/HILL_THERM/3
Hill Therm Synonym
2.7e-9
70000.0, 0.33
0, 0.0, 0.0
1.1, 1.2, 1.3, 0.2, 1
0.5, 0.2, 0.4
101, 1.0, 1.0
295.0, 2.4e-3

/MAT/THERM_HILL/4
Therm Hill Synonym
2.7e-9
70000.0, 0.33
0, 0.0, 0.0
1.1, 1.2, 1.3, 0.2, 1
0.5, 0.2, 0.4
101, 1.0, 1.0
295.0, 2.4e-3

/MAT/LAW73/5/1
Law 73 with Unit ID
2.7e-9
70000.0, 0.33
0, 0.0, 0.0
1.1, 1.2, 1.3, 0.2, 1
0.5, 0.2, 0.4
101, 1.0, 1.0
295.0, 2.4e-3

/MAT/73
Numeric 73 Keyword
2.7e-9
70000.0, 0.33
0, 0.0, 0.0
1.1, 1.2, 1.3, 0.2, 1
0.5, 0.2, 0.4
101, 1.0, 1.0
295.0, 2.4e-3
"""
        model, log = _parse_deck_str(tmp_path, deck_text, name="SYNONYMS")
        assert not log.has_errors
        assert set(model.mat_law73s.keys()) == {1, 2, 3, 4, 5, 73}
        assert set(model.mat_therm_hills.keys()) == {1, 2, 3, 4, 5, 73}
        assert set(model.materials.keys()) == {1, 2, 3, 4, 5, 73}

        # Check entity types
        for mid in (1, 2, 3, 4, 5, 73):
            m = model.mat_law73s[mid]
            assert isinstance(m, MatLaw73)
            assert isinstance(m, MatHillTherm)
            assert isinstance(m, MatThermalHill)
            assert isinstance(m, MatThermHill)
            assert model.materials[mid].law == 73


# ============================================================================
# 4. Pickle & Restart (.rst) Serialization
# ============================================================================

class TestLaw73Serialization:
    """Audit MatLaw73 and Model serialization, .rst file writing/reading, and cycle continuation."""

    def test_pickle_mat_law73_fidelity(self):
        """Verify MatLaw73 dataclass pickling fidelity."""
        m = MatLaw73(
            id=73,
            title="Pickle-Law73",
            rho=2.7e-9,
            refer_rho=2.75e-9,
            e=70000.0,
            nu=0.33,
            ifunce=2,
            einf=45000.0,
            ce=12.5,
            r00=1.25,
            r45=1.45,
            r90=1.85,
            chard=0.35,
            iyield=1,
            eps_max=0.55,
            epsr1=0.25,
            epsr2=0.45,
            table_id=202,
            fscale=1.15,
            pscale=0.95,
            t0=300.0,
            rhocp=2.45e-3,
        )
        serialized = pickle.dumps(m, protocol=pickle.HIGHEST_PROTOCOL)
        restored = pickle.loads(serialized)

        assert isinstance(restored, MatLaw73)
        _assert_law73_all_fields_exact(
            restored,
            rho=2.7e-9,
            rhor=2.75e-9,
            e=70000.0,
            nu=0.33,
            ifunce=2,
            einf=45000.0,
            ce=12.5,
            r00=1.25,
            r45=1.45,
            r90=1.85,
            chard=0.35,
            iyield=1,
            eps_max=0.55,
            epsr1=0.25,
            epsr2=0.45,
            table_id=202,
            fscale=1.15,
            pscale=0.95,
            t0=300.0,
            rhocp=2.45e-3,
            title="Pickle-Law73",
        )

    def test_write_read_restart_preserves_uvar73_state(self, tmp_path: Path):
        """Verify write_restart and read_restart preserve LAW73 persistent shell history arrays (uvar73)."""
        deck = StarterDeck("RST_LAW73_SHELL")
        deck.node([
            (1, 0.0, 0.0, 0.0),
            (2, 10.0, 0.0, 0.0),
            (3, 10.0, 10.0, 0.0),
            (4, 0.0, 10.0, 0.0),
            (5, 20.0, 0.0, 0.0),
            (6, 20.0, 10.0, 0.0),
        ])
        deck.shell(1, [
            [1, 1, 2, 3, 4],
            [2, 2, 5, 6, 3],
        ])
        deck.prop_shell(1, "ShellProp", thick=1.5, nip=3)
        deck.part(1, "ShellPart", prop_id=1, mat_id=1)
        deck.mat_law73(
            mid=1,
            title="Restart-Sheet",
            rho=2.7e-9,
            e=70000.0,
            nu=0.33,
            r00=1.2,
            r45=1.4,
            r90=1.6,
            chard=0.3,
            iyield=1,
        )

        rad_path = tmp_path / "rst_law73_0000.rad"
        deck.write(str(rad_path))

        model = run_starter(str(rad_path))
        assert model is not None

        groups = dict(model.element_groups())
        sh_grp = groups.get("shells")
        assert sh_grp is not None
        st = sh_grp.state

        # Synthesize realistic persistent shell layer history arrays (2 elements, nip=3, 7 channels)
        uvar73_elem_synth = np.array([
            [0.01, 12.5, -5.2, 3.1, 0.0, 0.0, 0.0],
            [0.02, 18.3, -8.1, 4.2, 0.0, 0.0, 0.0],
        ], dtype=float)

        uvar73_layer_synth = np.zeros((2, 3, 7), dtype=float)
        uvar73_layer_synth[0, 0, :] = [0.005, 10.0, -4.0, 2.5, 0.0, 0.0, 0.0]
        uvar73_layer_synth[0, 1, :] = [0.010, 12.5, -5.2, 3.1, 0.0, 0.0, 0.0]
        uvar73_layer_synth[0, 2, :] = [0.015, 15.0, -6.4, 3.7, 0.0, 0.0, 0.0]
        uvar73_layer_synth[1, 0, :] = [0.008, 14.0, -6.0, 3.0, 0.0, 0.0, 0.0]
        uvar73_layer_synth[1, 1, :] = [0.020, 18.3, -8.1, 4.2, 0.0, 0.0, 0.0]
        uvar73_layer_synth[1, 2, :] = [0.025, 22.0, -9.5, 5.0, 0.0, 0.0, 0.0]

        pla73_layer_synth = np.array([
            [0.005, 0.010, 0.015],
            [0.008, 0.020, 0.025],
        ], dtype=float)

        st["uvar73"] = uvar73_elem_synth.copy()
        st["mat_extra"]["uvar73"] = uvar73_layer_synth.copy()
        st["mat_extra"]["pla73"] = pla73_layer_synth.copy()

        # Write restart .rst
        rst_path = tmp_path / "rst_law73_0001.rst"
        engine_dict = {
            "cycle": 100,
            "t": 0.002,
            "dt": 5.0e-8,
        }
        write_restart(model, str(rst_path), engine=engine_dict)

        # Read back from restart .rst
        rest_model, rest_engine = read_restart(str(rst_path))
        assert rest_engine["cycle"] == 100
        assert rest_engine["t"] == pytest.approx(0.002)
        assert rest_engine["dt"] == pytest.approx(5.0e-8)

        # Assert exact preservation of LAW73 state arrays
        rest_sh = dict(rest_model.element_groups()).get("shells")
        assert rest_sh is not None
        rest_st = rest_sh.state
        rest_extra = rest_st["mat_extra"]

        np.testing.assert_array_equal(rest_st["uvar73"], uvar73_elem_synth)
        np.testing.assert_array_equal(rest_extra["uvar73"], uvar73_layer_synth)
        np.testing.assert_array_equal(rest_extra["pla73"], pla73_layer_synth)

    def test_constitutive_dynamic_cycle_continuation_from_restart(self):
        """Constitutive shell update continuation from restart matches uninterrupted simulation within 10^-12."""
        mat = build_law73({
            "id": 73,
            "rho0": 2.7e-9,
            "e": 70000.0,
            "nu": 0.33,
            "r00": 1.2,
            "r45": 1.4,
            "r90": 1.6,
            "chard": 0.3,
            "iyield": 1,
            "sigy0": 200.0,
        })

        dt = 1.0e-6
        deps_steps = [
            np.array([[0.002, -0.0006, 0.0003]]),
            np.array([[0.003, -0.0009, 0.0005]]),
            np.array([[0.004, -0.0012, 0.0008]]),
            np.array([[0.005, -0.0015, 0.0010]]),
        ]

        # 1. Uninterrupted run: 4 steps
        sig_uninterrupted = np.zeros((1, 3))
        epsp_uninterrupted = np.zeros(1)
        extra_uninterrupted = {
            "uvar73": np.zeros((1, 7)),
            "pla73": np.zeros(1),
            "off73": np.ones(1),
            "thk": np.array([1.0]),
            "temp": np.array([293.0]),
        }

        for deps in deps_steps:
            sig_uninterrupted, epsp_uninterrupted = shell_update(
                mat,
                sig_uninterrupted,
                deps=deps,
                epsp=epsp_uninterrupted,
                dt=dt,
                extra=extra_uninterrupted,
            )

        # 2. Resumed run: 2 steps, snapshot/restart, continue 2 steps
        sig_restarted = np.zeros((1, 3))
        epsp_restarted = np.zeros(1)
        extra_restarted = {
            "uvar73": np.zeros((1, 7)),
            "pla73": np.zeros(1),
            "off73": np.ones(1),
            "thk": np.array([1.0]),
            "temp": np.array([293.0]),
        }

        for deps in deps_steps[:2]:
            sig_restarted, epsp_restarted = shell_update(
                mat,
                sig_restarted,
                deps=deps,
                epsp=epsp_restarted,
                dt=dt,
                extra=extra_restarted,
            )

        # Snapshot state via pickle (simulating restart file write/read)
        snap = pickle.dumps({
            "sig": sig_restarted.copy(),
            "epsp": epsp_restarted.copy(),
            "extra": {k: v.copy() if isinstance(v, np.ndarray) else v for k, v in extra_restarted.items()},
        })

        # Restore from snapshot
        restored = pickle.loads(snap)
        sig_resumed = restored["sig"]
        epsp_resumed = restored["epsp"]
        extra_resumed = restored["extra"]

        # Continue next 2 steps
        for deps in deps_steps[2:]:
            sig_resumed, epsp_resumed = shell_update(
                mat,
                sig_resumed,
                deps=deps,
                epsp=epsp_resumed,
                dt=dt,
                extra=extra_resumed,
            )

        # Assert exact continuation within 10^-12
        np.testing.assert_allclose(sig_resumed, sig_uninterrupted, rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(epsp_resumed, epsp_uninterrupted, rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(extra_resumed["uvar73"], extra_uninterrupted["uvar73"], rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(extra_resumed["pla73"], extra_uninterrupted["pla73"], rtol=1e-12, atol=1e-12)


# ============================================================================
# 5. Negative Starter Diagnostics
# ============================================================================

class TestLaw73NegativeValidation:
    """Audit starter negative diagnostics in check_mat_law73 and check_materials."""

    def test_negative_and_zero_density(self):
        """Error when initial density RHO <= 0."""
        log1 = MessageLog()
        m1 = MatLaw73(id=1, rho=0.0, e=70000.0, nu=0.33)
        check_mat_law73(m1, log1)
        assert log1.has_errors
        assert any("initial density RHO must be > 0" in e for e in log1.errors)

        log2 = MessageLog()
        m2 = MatLaw73(id=2, rho=-2.7e-9, e=70000.0, nu=0.33)
        check_mat_law73(m2, log2)
        assert log2.has_errors
        assert any("initial density RHO must be > 0" in e for e in log2.errors)

    def test_negative_and_zero_young(self):
        """Error when Young's modulus E <= 0."""
        log1 = MessageLog()
        m1 = MatLaw73(id=1, rho=2.7e-9, e=0.0, nu=0.33)
        check_mat_law73(m1, log1)
        assert log1.has_errors
        assert any("Young's modulus E must be > 0" in e for e in log1.errors)

        log2 = MessageLog()
        m2 = MatLaw73(id=2, rho=2.7e-9, e=-70000.0, nu=0.33)
        check_mat_law73(m2, log2)
        assert log2.has_errors
        assert any("Young's modulus E must be > 0" in e for e in log2.errors)

    def test_poisson_ratio_bounds(self):
        """Error when nu < 0 or nu >= 0.5 (ANCMSG 1514)."""
        # Negative nu
        log1 = MessageLog()
        m1 = MatLaw73(id=1, rho=2.7e-9, e=70000.0, nu=-0.05)
        check_mat_law73(m1, log1)
        assert log1.has_errors
        assert any("ANCMSG 1514" in e for e in log1.errors)

        # Incompressible limit nu = 0.5
        log2 = MessageLog()
        m2 = MatLaw73(id=2, rho=2.7e-9, e=70000.0, nu=0.5)
        check_mat_law73(m2, log2)
        assert log2.has_errors
        assert any("ANCMSG 1514" in e for e in log2.errors)

        # Over limit nu > 0.5
        log3 = MessageLog()
        m3 = MatLaw73(id=3, rho=2.7e-9, e=70000.0, nu=0.55)
        check_mat_law73(m3, log3)
        assert log3.has_errors
        assert any("ANCMSG 1514" in e for e in log3.errors)

        # Valid bounds
        log4 = MessageLog()
        m4 = MatLaw73(id=4, rho=2.7e-9, e=70000.0, nu=0.0)
        check_mat_law73(m4, log4)
        assert not log4.has_errors

        log5 = MessageLog()
        m5 = MatLaw73(id=5, rho=2.7e-9, e=70000.0, nu=0.499)
        check_mat_law73(m5, log5)
        assert not log5.has_errors

    def test_lankford_anisotropy_bounds(self):
        """Error when R00, R45, or R90 <= 0."""
        log1 = MessageLog()
        m1 = MatLaw73(id=1, rho=2.7e-9, e=70000.0, nu=0.33, r00=0.0)
        check_mat_law73(m1, log1)
        assert log1.has_errors
        assert any("Lankford parameter R00 must be > 0" in e for e in log1.errors)

        log2 = MessageLog()
        m2 = MatLaw73(id=2, rho=2.7e-9, e=70000.0, nu=0.33, r45=-1.0)
        check_mat_law73(m2, log2)
        assert log2.has_errors
        assert any("Lankford parameter R45 must be > 0" in e for e in log2.errors)

        log3 = MessageLog()
        m3 = MatLaw73(id=3, rho=2.7e-9, e=70000.0, nu=0.33, r90=0.0)
        check_mat_law73(m3, log3)
        assert log3.has_errors
        assert any("Lankford parameter R90 must be > 0" in e for e in log3.errors)

        # Positive passes
        log4 = MessageLog()
        m4 = MatLaw73(id=4, rho=2.7e-9, e=70000.0, nu=0.33, r00=0.1, r45=0.1, r90=0.1)
        check_mat_law73(m4, log4)
        assert not log4.has_errors

    def test_failure_strains_bounds(self):
        """Error when epsr1 >= epsr2 (ANCMSG 1044)."""
        log1 = MessageLog()
        m1 = MatLaw73(id=1, rho=2.7e-9, e=70000.0, nu=0.33, epsr1=0.3, epsr2=0.3)
        check_mat_law73(m1, log1)
        assert log1.has_errors
        assert any("ANCMSG 1044" in e for e in log1.errors)

        log2 = MessageLog()
        m2 = MatLaw73(id=2, rho=2.7e-9, e=70000.0, nu=0.33, epsr1=0.4, epsr2=0.2)
        check_mat_law73(m2, log2)
        assert log2.has_errors
        assert any("ANCMSG 1044" in e for e in log2.errors)

        # Valid order passes
        log3 = MessageLog()
        m3 = MatLaw73(id=3, rho=2.7e-9, e=70000.0, nu=0.33, epsr1=0.2, epsr2=0.4)
        check_mat_law73(m3, log3)
        assert not log3.has_errors

    def test_incompatible_solid_elements_rejected(self):
        """Reject 3D solid elements with ANCMSG 305."""
        class DummyEl:
            def __init__(self, mid: int):
                self.mat_id = mid

        class DummyGrp:
            def __init__(self, els: list):
                self._els = {i: e for i, e in enumerate(els)}
            def values(self):
                return self._els.values()

        class DummyModel:
            def __init__(self, grps: list):
                self._grps = grps
                self.materials: dict = {}
                self.parts: dict = {}
            def element_groups(self):
                return self._grps

        m = MatLaw73(id=1, rho=2.7e-9, e=70000.0, nu=0.33)
        for s_type in ("bricks", "bricks_heph", "bric20s", "tetras", "tetra10s", "penta6", "pyra5", "solids"):
            model_s = DummyModel([(s_type, DummyGrp([DummyEl(1)]))])
            log_s = MessageLog()
            check_mat_law73(model=model_s, mat=m, log=log_s)
            assert log_s.has_errors, f"Solid element group {s_type} should be rejected"
            assert any("ANCMSG 305" in e for e in log_s.errors)

        # Via part dictionary
        model_part = Model()
        model_part.mat_law73s[1] = m
        model_part.materials[1] = Material(id=1, law=73, rho0=2.7e-9, params={"rho0": 2.7e-9, "e": 70000.0, "nu": 0.33})
        p_solid = Part(id=1, prop_id=1, mat_id=1, title="Solid Part")
        p_solid.elem_type = "SOLID"
        model_part.parts[1] = p_solid
        log_p = MessageLog()
        check_mat_law73(model_part, 1, m, log_p)
        assert log_p.has_errors
        assert any("ANCMSG 305" in e for e in log_p.errors)

    def test_incompatible_1d_elements_rejected(self):
        """Reject 1D elements with ANCMSG 306."""
        class DummyEl:
            def __init__(self, mid: int):
                self.mat_id = mid

        class DummyGrp:
            def __init__(self, els: list):
                self._els = {i: e for i, e in enumerate(els)}
            def values(self):
                return self._els.values()

        class DummyModel:
            def __init__(self, grps: list):
                self._grps = grps
                self.materials: dict = {}
                self.parts: dict = {}
            def element_groups(self):
                return self._grps

        m = MatLaw73(id=1, rho=2.7e-9, e=70000.0, nu=0.33)
        for d1_type in ("trusses", "beams", "springs"):
            model_1d = DummyModel([(d1_type, DummyGrp([DummyEl(1)]))])
            log_1d = MessageLog()
            check_mat_law73(model=model_1d, mat=m, log=log_1d)
            assert log_1d.has_errors, f"1D element group {d1_type} should be rejected"
            assert any("ANCMSG 306" in e for e in log_1d.errors)

        # Via part dictionary
        model_part = Model()
        model_part.mat_law73s[1] = m
        model_part.materials[1] = Material(id=1, law=73, rho0=2.7e-9, params={"rho0": 2.7e-9, "e": 70000.0, "nu": 0.33})
        p_beam = Part(id=1, prop_id=1, mat_id=1, title="Beam Part")
        p_beam.elem_type = "BEAM"
        model_part.parts[1] = p_beam
        log_p = MessageLog()
        check_mat_law73(model_part, 1, m, log_p)
        assert log_p.has_errors
        assert any("ANCMSG 306" in e for e in log_p.errors)

    def test_compatible_shell_elements_accepted(self):
        """Accept shell elements without diagnostic errors."""
        class DummyEl:
            def __init__(self, mid: int):
                self.mat_id = mid

        class DummyGrp:
            def __init__(self, els: list):
                self._els = {i: e for i, e in enumerate(els)}
            def values(self):
                return self._els.values()

        class DummyModel:
            def __init__(self, grps: list):
                self._grps = grps
                self.materials: dict = {}
                self.parts: dict = {}
            def element_groups(self):
                return self._grps

        m = MatLaw73(id=1, rho=2.7e-9, e=70000.0, nu=0.33)
        for sh_type in ("shells", "shells_qbat", "shells_qeph", "sh3n", "quads"):
            model_sh = DummyModel([(sh_type, DummyGrp([DummyEl(1)]))])
            log_sh = MessageLog()
            check_mat_law73(model=model_sh, mat=m, log=log_sh)
            assert not log_sh.has_errors, f"Shell element group {sh_type} should be accepted"

    def test_checks_registry_and_dispatch(self):
        """Verify registry membership and check_materials dispatch."""
        for syn in (73, "73", "LAW73", "HILL_THERM", "THERM_HILL",
                    "MAT_LAW73", "MAT_HILL_THERM", "MAT_THERM_HILL"):
            assert syn in _MAT_CHECKS, f"{syn} missing from _MAT_CHECKS"
            assert _MAT_CHECKS[syn] is check_mat_law73

        for fam in ("shells", "shells_qbat", "shells_qeph", "sh3n", "quads"):
            if fam in _ALLOWED_LAWS and _ALLOWED_LAWS[fam] is not None:
                assert 73 in _ALLOWED_LAWS[fam], f"73 missing from {fam} allowed laws"

        for fam in ("bricks", "tetras", "penta6", "pyra5", "solids", "trusses", "beams"):
            if fam in _ALLOWED_LAWS and _ALLOWED_LAWS[fam] is not None:
                assert 73 not in _ALLOWED_LAWS[fam], f"73 unexpectedly allowed for {fam}"

        # Dispatch via check_materials
        model = Model()
        model.materials[1] = Material(id=1, law=73, rho0=-1.0, params={"rho0": -1.0, "e": 70000.0, "nu": 0.33})
        log = MessageLog()
        check_materials(model, log)
        assert log.has_errors
        assert any("initial density RHO must be > 0" in e for e in log.errors)


# ============================================================================
# 6. Boundary Values & Defaults
# ============================================================================

class TestLaw73BoundaryValuesAndDefaults:
    """Audit boundary conditions, default fallbacks, hardening modes, yield modes, and mapping protocol."""

    def test_default_fallbacks(self, tmp_path: Path):
        """Verify default fallbacks per hm_read_mat73.F when cards omit optional fields."""
        deck_text = """
/MAT/LAW73/1
Minimal Card
2.7e-9
70000.0, 0.33
"""
        model, log = _parse_deck_str(tmp_path, deck_text, name="DEFAULTS")
        assert not log.has_errors
        m = model.mat_law73s[1]

        # Lankford defaults: 1.0
        assert m.r00 == pytest.approx(1.0)
        assert m.r45 == pytest.approx(1.0)
        assert m.r90 == pytest.approx(1.0)

        # Thermal default: 293.0
        assert m.t0 == pytest.approx(293.0)
        assert m.rhocp == pytest.approx(0.0)

        # Failure defaults: 1e30, 1e30, 2e30
        assert m.eps_max == pytest.approx(1.0e30)
        assert m.epsr1 == pytest.approx(1.0e30)
        assert m.epsr2 == pytest.approx(2.0e30)

        # Scaling defaults: 1.0
        assert m.fscale == pytest.approx(1.0)
        assert m.pscale == pytest.approx(1.0)

        # Hardening / yield flags: 0
        assert m.chard == pytest.approx(0.0)
        assert m.iyield == 0

    def test_boundary_poisson_ratio_moduli(self):
        """Verify derived constants at boundary nu=0.0 and nu=0.499."""
        # nu = 0.0 (zero lateral contraction)
        m0 = MatLaw73(id=1, rho=1.0, e=1000.0, nu=0.0)
        assert m0.G == pytest.approx(500.0)
        assert m0.sound_speed == pytest.approx(math.sqrt(1000.0))

        # nu = 0.499 (nearly incompressible)
        m_incomp = MatLaw73(id=2, rho=1.0, e=1000.0, nu=0.499)
        c_expected = math.sqrt(1000.0 / (1.0 - 0.499**2))
        assert m_incomp.sound_speed_shell == pytest.approx(c_expected)
        assert m_incomp.sound_speed == pytest.approx(math.sqrt(1000.0))
        assert m_incomp.sound_speed_shell > math.sqrt(1000.0)

    def test_hardening_boundary_modes(self):
        """Verify chard (fisokin) boundary modes: pure isotropic (0.0), pure kinematic (1.0), mixed (0.5)."""
        m_iso = MatLaw73(id=1, rho=2.7e-9, e=70000.0, nu=0.33, chard=0.0)
        assert m_iso.chard == pytest.approx(0.0)
        assert m_iso.fisokin == pytest.approx(0.0)

        m_kin = MatLaw73(id=2, rho=2.7e-9, e=70000.0, nu=0.33, chard=1.0)
        assert m_kin.chard == pytest.approx(1.0)
        assert m_kin.fisokin == pytest.approx(1.0)

        m_mix = MatLaw73(id=3, rho=2.7e-9, e=70000.0, nu=0.33, chard=0.5)
        assert m_mix.chard == pytest.approx(0.5)
        assert m_mix.fisokin == pytest.approx(0.5)

    def test_yield_criterion_modes(self):
        """Verify iyield flag: 0 (average orientation) and 1 (normalized to direction 1)."""
        # iyield = 0: standard Hill coefficients
        m0 = MatLaw73(id=1, rho=2.7e-9, e=70000.0, nu=0.33, r00=1.2, r45=1.4, r90=1.6, iyield=0)
        r0 = 0.25 * (1.2 + 2.0 * 1.4 + 1.6)
        h0 = r0 / (1.0 + r0)
        a01_expected = h0 * (1.0 + 1.0 / 1.2)
        assert m0.A01 == pytest.approx(a01_expected)

        # iyield = 1: normalized so A01 = 1.0
        m1 = MatLaw73(id=2, rho=2.7e-9, e=70000.0, nu=0.33, r00=1.2, r45=1.4, r90=1.6, iyield=1)
        assert m1.A01 == pytest.approx(1.0)
        assert m1.A02 == pytest.approx(m0.A02 / a01_expected)
        assert m1.A03 == pytest.approx(m0.A03 / a01_expected)
        assert m1.A12 == pytest.approx(m0.A12 / a01_expected)

    def test_dynamic_modulus_degradation_flags(self):
        """Verify ifunce and ce dynamic modulus degradation parameter combinations."""
        # 1. Constant modulus
        m_const = MatLaw73(id=1, rho=2.7e-9, e=70000.0, nu=0.33, ifunce=0, ce=0.0)
        assert m_const.ifunce == 0
        assert m_const.ce == pytest.approx(0.0)

        # 2. Tabulated degradation
        m_tab = MatLaw73(id=2, rho=2.7e-9, e=70000.0, nu=0.33, ifunce=5)
        assert m_tab.ifunce == 5
        assert m_tab.yr_fun == 5

        # 3. Exponential degradation
        m_exp = MatLaw73(id=3, rho=2.7e-9, e=70000.0, nu=0.33, einf=50000.0, ce=10.0)
        assert m_exp.einf == pytest.approx(50000.0)
        assert m_exp.efib == pytest.approx(50000.0)
        assert m_exp.ce == pytest.approx(10.0)
        assert m_exp.c == pytest.approx(10.0)

    def test_mapping_protocol(self):
        """Verify dict mapping protocol (__getitem__, __setitem__, __contains__, get, keys, items, values)."""
        m = MatLaw73(id=73, rho=2.7e-9, e=70000.0, nu=0.33, r00=1.2, t0=295.0)

        assert m["rho"] == pytest.approx(2.7e-9)
        assert m["e"] == pytest.approx(70000.0)
        assert m["nu"] == pytest.approx(0.33)
        assert m["r00"] == pytest.approx(1.2)
        assert m.get("t0") == pytest.approx(295.0)
        assert m.get("nonexistent", 99.0) == 99.0

        assert "rho" in m
        assert "e" in m
        assert "r00" in m
        assert "nonexistent" not in m

        # Mutate via __setitem__
        m["r00"] = 1.35
        assert m.r00 == pytest.approx(1.35)

        # keys, items, values
        keys = list(m.keys())
        assert "rho" in keys
        assert "e" in keys
        assert "r00" in keys
        items = dict(m.items())
        assert items["r00"] == pytest.approx(1.35)
