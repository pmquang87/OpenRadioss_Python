"""Tests for Milestone M540: /MAT/LAW37, /MAT/BIPHAS, and /MAT/BIPHASIC input layer.

Covers:
1. Card layout constants, aliases, and LAYOUTS dictionary registration.
2. Deck writer emitter StarterDeck.mat_law37 and aliases (mat_biphas, mat_biphasic).
3. Fixed-format 20-column card width alignment and formatting.
4. CFG catalogue and mat_reader synonym resolution.
5. Starter validation checks: _ALLOWED_LAWS solid element acceptance, shell element rejection,
   and parameter bounds checks.
6. Deck roundtrip parsing with parse_starter_deck and check_model integration.
"""

from __future__ import annotations

import math
from pathlib import Path
import pytest

from pyradioss.common.messages import MessageLog
from pyradioss.input import card_layouts as cl
from pyradioss.input.deck_writer import StarterDeck
from pyradioss.input.mat_reader import MAT_PHYSICS_REGISTRY, catalogue, CfgCatalogue
from pyradioss.input.cfg_catalogue import LAW_MAP, LAW_SYNONYMS
from pyradioss.input.starter_keywords import parse_starter_deck
from pyradioss.model.model import Model
from pyradioss.model.entities import Material
from pyradioss.starter.checks import _ALLOWED_LAWS, check_model, check_mat_law37


def block_lines(text: str, header_prefix: str) -> list[str]:
    """The lines of the block starting with header_prefix (incl. header)."""
    lines = text.splitlines()
    out, active = [], False
    for ln in lines:
        s = ln.strip()
        if s.startswith("/"):
            active = s.startswith(header_prefix)
        if active:
            out.append(ln)
    return out


def data_cards(lines: list[str]) -> list[str]:
    """Block lines minus header and comments (blank cards kept)."""
    return [ln for ln in lines[1:] if not ln.lstrip().startswith("#")]


class TestLaw37CardLayouts:
    """Card layout constants and LAYOUTS dictionary verification (M540)."""

    def test_constants_and_aliases(self):
        assert cl.MAT_LAW37_1 == (20, 20)
        assert cl.MAT_LAW37_2 == (20, 20, 20, 20, 20)
        assert cl.MAT_LAW37_3 == (20, 20, 20, 20, 20)

        assert cl.MAT_LAW37_CFG_1 == (20, 20)
        assert cl.MAT_LAW37_CFG_2 == (20, 20, 20, 20, 20)
        assert cl.MAT_LAW37_CFG_3 == (20, 20, 20, 20, 20)

        assert cl.MAT_BIPHAS_1 == (20, 20)
        assert cl.MAT_BIPHAS_2 == (20, 20, 20, 20, 20)
        assert cl.MAT_BIPHAS_3 == (20, 20, 20, 20, 20)

        assert cl.MAT_BIPHASIC_1 == (20, 20)
        assert cl.MAT_BIPHASIC_2 == (20, 20, 20, 20, 20)
        assert cl.MAT_BIPHASIC_3 == (20, 20, 20, 20, 20)

    def test_layouts_dictionary(self):
        expected_keys = [
            "MAT_LAW37_1", "MAT_LAW37_2", "MAT_LAW37_3",
            "MAT_LAW37_CFG_1", "MAT_LAW37_CFG_2", "MAT_LAW37_CFG_3",
            "MAT_BIPHAS_1", "MAT_BIPHAS_2", "MAT_BIPHAS_3",
            "MAT_BIPHAS_CFG_1", "MAT_BIPHAS_CFG_2", "MAT_BIPHAS_CFG_3",
            "MAT_BIPHASIC_1", "MAT_BIPHASIC_2", "MAT_BIPHASIC_3",
            "MAT_BIPHASIC_CFG_1", "MAT_BIPHASIC_CFG_2", "MAT_BIPHASIC_CFG_3",
        ]
        for key in expected_keys:
            assert key in cl.LAYOUTS, f"Missing key {key} in LAYOUTS"
            layout = cl.LAYOUTS[key]
            if key.endswith("_1"):
                assert list(layout) == [20, 20]
            elif key.endswith("_2"):
                assert list(layout) == [20, 20, 20, 20, 20]
            elif key.endswith("_3"):
                assert list(layout) == [20, 20, 20, 20, 20]


class TestLaw37DeckWriter:
    """StarterDeck emitter tests for /MAT/LAW37 and its aliases."""

    def test_emit_mat_law37_basic(self):
        deck = StarterDeck("M540")
        deck.mat_law37(
            id=1,
            rho_l0=1000.0,
            c_l=2.2e9,
            alpha1=0.8,
            nu_l=1.0e-3,
            nu_vol_l=2.0e-3,
            rho_g0=1.2,
            gamma_g=1.4,
            p0_g=1.0e5,
            nu_g=1.8e-5,
            nu_vol_g=0.0,
            title="WATER_AIR_MIX",
        )
        lines = block_lines(deck.render(), "/MAT/LAW37/1")
        assert len(lines) >= 5
        assert lines[0].startswith("/MAT/LAW37/1")
        assert lines[1] == "WATER_AIR_MIX"
        cards = data_cards(lines)
        # Card 1: rho computed as 1000 * 0.8 + 1.2 * 0.2 = 800.24
        assert len(cards[1]) == 20
        assert math.isclose(float(cards[1]), 800.24, rel_tol=1e-5)
        # Card 2: rho_l0, c_l, alpha1, nu_l, nu_vol_l
        assert len(cards[2]) == 100
        assert math.isclose(float(cards[2][:20]), 1000.0, rel_tol=1e-5)
        assert math.isclose(float(cards[2][20:40]), 2.2e9, rel_tol=1e-5)
        assert math.isclose(float(cards[2][40:60]), 0.8, rel_tol=1e-5)
        assert math.isclose(float(cards[2][60:80]), 1.0e-3, rel_tol=1e-5)
        assert math.isclose(float(cards[2][80:100]), 2.0e-3, rel_tol=1e-5)
        # Card 3: rho_g0, gamma_g, p0_g, nu_g, nu_vol_g
        assert len(cards[3]) == 100
        assert math.isclose(float(cards[3][:20]), 1.2, rel_tol=1e-5)
        assert math.isclose(float(cards[3][20:40]), 1.4, rel_tol=1e-5)
        assert math.isclose(float(cards[3][40:60]), 1.0e5, rel_tol=1e-5)
        assert math.isclose(float(cards[3][60:80]), 1.8e-5, rel_tol=1e-5)
        assert math.isclose(float(cards[3][80:100]), 0.0, abs_tol=1e-10)

    def test_emit_mat_law37_with_rhor(self):
        deck = StarterDeck("M540")
        deck.mat_law37(
            id=2,
            rho_l0=1000.0,
            c_l=2.2e9,
            alpha1=1.0,
            rhor=998.0,
            title="PURE_WATER_RHOR",
        )
        lines = block_lines(deck.render(), "/MAT/LAW37/2")
        cards = data_cards(lines)
        assert len(cards[1]) == 40
        assert math.isclose(float(cards[1][:20]), 1000.0, rel_tol=1e-5)
        assert math.isclose(float(cards[1][20:40]), 998.0, rel_tol=1e-5)

    def test_emit_mat_law37_with_pshift(self):
        deck = StarterDeck("M540")
        deck.mat_law37(
            id=3,
            rho_l0=1000.0,
            c_l=2.2e9,
            alpha1=1.0,
            pshift=-1.0e5,
            title="WATER_PSHIFT",
        )
        lines = block_lines(deck.render(), "/MAT/LAW37/3")
        cards = data_cards(lines)
        assert len(cards[1]) == 40
        assert math.isclose(float(cards[1][:20]), 1000.0, rel_tol=1e-5)
        assert math.isclose(float(cards[1][20:40]), -1.0e5, rel_tol=1e-5)

    def test_aliases_consistency(self):
        deck1 = StarterDeck("M540").mat_law37(1, 1000.0, 2.2e9, 0.5, law_name="LAW37", title="T1")
        deck2 = StarterDeck("M540").mat_biphas(1, 1000.0, 2.2e9, 0.5, law_name="BIPHAS", title="T1")
        deck3 = StarterDeck("M540").mat_biphasic(1, 1000.0, 2.2e9, 0.5, law_name="BIPHASIC", title="T1")

        lines1 = block_lines(deck1.render(), "/MAT/LAW37/1")
        lines2 = block_lines(deck2.render(), "/MAT/BIPHAS/1")
        lines3 = block_lines(deck3.render(), "/MAT/BIPHASIC/1")

        assert data_cards(lines1) == data_cards(lines2)
        assert data_cards(lines1) == data_cards(lines3)

    def test_card_width_alignment_exact_20(self):
        """Verify each data card consists strictly of 20-character fields."""
        deck = StarterDeck("M540")
        deck.mat_law37(
            1,
            rho_l0=998.2,
            c_l=2.18e9,
            alpha1=0.73,
            nu_l=1.002e-3,
            nu_vol_l=2.4e-3,
            rho_g0=1.18,
            gamma_g=1.403,
            p0_g=101325.75,
            nu_g=1.876e-5,
            nu_vol_g=3.456e-5,
            rhor=500.25,
        )
        lines = block_lines(deck.render(), "/MAT/LAW37/1")
        cards = data_cards(lines)
        for ln in cards[1:]:
            assert len(ln) % 20 == 0, f"Card {ln!r} length {len(ln)} is not a multiple of 20"
            for i in range(0, len(ln), 20):
                field = ln[i:i+20]
                assert len(field) == 20
                float(field)


class TestLaw37SynonymsAndRegistry:
    """CFG catalogue, law mapping, and physics registry synonym mappings."""

    def test_physics_registry_synonyms(self):
        synonyms = [37, "37", "LAW37", "BIPHAS", "BIPHASIC"]
        for syn in synonyms:
            assert syn in MAT_PHYSICS_REGISTRY, f"Synonym {syn} not registered in MAT_PHYSICS_REGISTRY"
            builder = MAT_PHYSICS_REGISTRY[syn]
            assert callable(builder)

    def test_cfg_catalogue_synonym_resolution(self):
        assert LAW_MAP.get("LAW37") == 37
        assert LAW_MAP.get("BIPHAS") == 37
        assert LAW_MAP.get("BIPHASIC") == 37

        assert LAW_SYNONYMS.get("LAW37") == "LAW37"
        assert LAW_SYNONYMS.get("BIPHAS") == "LAW37"
        assert LAW_SYNONYMS.get("BIPHASIC") == "LAW37"


class TestLaw37StarterChecks:
    """Starter element-type checks and parameter validation for LAW37."""

    def test_allowed_laws_includes_law37_for_solids(self):
        for category in ["bricks", "tetras", "penta6", "pyra5"]:
            assert category in _ALLOWED_LAWS
            assert 37 in _ALLOWED_LAWS[category]
            assert "LAW37" in _ALLOWED_LAWS[category]
            assert "BIPHAS" in _ALLOWED_LAWS[category]
            assert "BIPHASIC" in _ALLOWED_LAWS[category]

    def test_allowed_laws_excludes_law37_for_shells(self):
        assert 37 not in _ALLOWED_LAWS["shells"]
        assert "LAW37" not in _ALLOWED_LAWS["shells"]
        assert "BIPHAS" not in _ALLOWED_LAWS["shells"]
        assert "BIPHASIC" not in _ALLOWED_LAWS["shells"]

    def test_check_mat_law37_valid(self):
        mat = Material(
            id=1,
            law=37,
            rho0=900.0,
            params={
                "rho_l0": 1000.0,
                "c_l": 2.2e9,
                "alpha1": 0.9,
                "nu_l": 1e-3,
                "rho_g0": 1.2,
                "gamma_g": 1.4,
                "nu_g": 1.8e-5,
            },
        )
        log = MessageLog()
        check_mat_law37(mat, log)
        assert len(log.errors) == 0

    def test_check_mat_law37_invalid_rho_l0(self):
        mat = Material(
            id=1,
            law=37,
            rho0=900.0,
            params={"rho_l0": 0.0, "c_l": 2.2e9, "alpha1": 0.9, "rho_g0": 1.2, "gamma_g": 1.4},
        )
        log = MessageLog()
        check_mat_law37(mat, log)
        assert any("liquid reference density rho_l0 must be > 0" in msg for msg in log.errors)

    def test_check_mat_law37_invalid_rho_g0(self):
        mat = Material(
            id=1,
            law=37,
            rho0=900.0,
            params={"rho_l0": 1000.0, "c_l": 2.2e9, "alpha1": 0.9, "rho_g0": -1.0, "gamma_g": 1.4},
        )
        log = MessageLog()
        check_mat_law37(mat, log)
        assert any("gas reference density rho_g0 must be > 0" in msg for msg in log.errors)

    def test_check_mat_law37_invalid_c_l(self):
        mat = Material(
            id=1,
            law=37,
            rho0=900.0,
            params={"rho_l0": 1000.0, "c_l": 0.0, "alpha1": 0.9, "rho_g0": 1.2, "gamma_g": 1.4},
        )
        log = MessageLog()
        check_mat_law37(mat, log)
        assert any("liquid bulk modulus c_l must be > 0" in msg for msg in log.errors)

    def test_check_mat_law37_invalid_gamma(self):
        mat = Material(
            id=1,
            law=37,
            rho0=900.0,
            params={"rho_l0": 1000.0, "c_l": 2.2e9, "alpha1": 0.9, "rho_g0": 1.2, "gamma_g": -0.5},
        )
        log = MessageLog()
        check_mat_law37(mat, log)
        assert any("gas constant gamma must be > 0" in msg for msg in log.errors)

    def test_check_mat_law37_invalid_alpha1(self):
        mat = Material(
            id=1,
            law=37,
            rho0=900.0,
            params={"rho_l0": 1000.0, "c_l": 2.2e9, "alpha1": 1.5, "rho_g0": 1.2, "gamma_g": 1.4},
        )
        log = MessageLog()
        check_mat_law37(mat, log)
        assert any("alpha1 must be between 0 and 1" in msg for msg in log.errors)

        mat.params["alpha1"] = -0.1
        log = MessageLog()
        check_mat_law37(mat, log)
        assert any("alpha1 must be between 0 and 1" in msg for msg in log.errors)

    def test_check_mat_law37_invalid_viscosities(self):
        mat = Material(
            id=1,
            law=37,
            rho0=900.0,
            params={"rho_l0": 1000.0, "c_l": 2.2e9, "alpha1": 0.9, "rho_g0": 1.2, "gamma_g": 1.4, "nu_l": -1e-4},
        )
        log = MessageLog()
        check_mat_law37(mat, log)
        assert any("liquid shear viscosity nu_l must be >= 0" in msg for msg in log.errors)

        mat.params["nu_l"] = 1e-4
        mat.params["nu_g"] = -1e-5
        log = MessageLog()
        check_mat_law37(mat, log)
        assert any("gas shear viscosity nu_g must be >= 0" in msg for msg in log.errors)


class TestLaw37DeckRoundtrip:
    """Deck writer and deck reader roundtrip parsing tests."""

    def test_parse_starter_deck_roundtrip_law37(self, tmp_path: Path):
        deck = StarterDeck("M540")
        deck.mat_law37(
            id=10,
            rho_l0=1000.0,
            c_l=2.2e9,
            alpha1=0.85,
            nu_l=1.0e-3,
            nu_vol_l=2.0e-3,
            rho_g0=1.2,
            gamma_g=1.4,
            p0_g=1.0e5,
            nu_g=1.8e-5,
            nu_vol_g=1.0e-5,
            title="BIPHASIC_FLUID_10",
        )
        deck.node([
            (1, 0.0, 0.0, 0.0),
            (2, 1.0, 0.0, 0.0),
            (3, 1.0, 1.0, 0.0),
            (4, 0.0, 1.0, 0.0),
            (5, 0.0, 0.0, 1.0),
            (6, 1.0, 0.0, 1.0),
            (7, 1.0, 1.0, 1.0),
            (8, 0.0, 1.0, 1.0),
        ])
        deck.prop_solid(1, "SOLID_PROP")
        deck.part(1, "PART_SOLID", 1, 10)
        deck.brick(1, [(1, 1, 2, 3, 4, 5, 6, 7, 8)])

        p = tmp_path / "M540_0000.rad"
        deck.write(str(p))
        model = parse_starter_deck(str(p))

        assert 10 in model.materials
        mat = model.materials[10]
        assert mat.law == 37
        expected_rho = 1000.0 * 0.85 + 1.2 * 0.15
        assert math.isclose(mat.rho0, expected_rho, rel_tol=1e-5)
        assert math.isclose(mat.params["rho_l0"], 1000.0, rel_tol=1e-5)
        assert math.isclose(mat.params["c_l"], 2.2e9, rel_tol=1e-5)
        assert math.isclose(mat.params["alpha1"], 0.85, rel_tol=1e-5)
        assert math.isclose(mat.params["nu_l"], 1.0e-3, rel_tol=1e-5)
        assert math.isclose(mat.params["nu_vol_l"], 2.0e-3, rel_tol=1e-5)
        assert math.isclose(mat.params["rho_g0"], 1.2, rel_tol=1e-5)
        assert math.isclose(mat.params["gamma_g"], 1.4, rel_tol=1e-5)
        assert math.isclose(mat.params["p0_g"], 1.0e5, rel_tol=1e-5)
        assert math.isclose(mat.params["nu_g"], 1.8e-5, rel_tol=1e-5)

        log = MessageLog()
        check_model(model, log)
        assert len(log.errors) == 0

    def test_parse_starter_deck_roundtrip_biphas_synonym(self, tmp_path: Path):
        deck = StarterDeck("M540")
        deck.mat_biphas(
            id=20,
            rho_l0=997.0,
            c_l=2.1e9,
            alpha1=1.0,
            title="PURE_WATER",
            law_name="BIPHAS",
        )
        deck.node([
            (1, 0.0, 0.0, 0.0),
            (2, 1.0, 0.0, 0.0),
            (3, 1.0, 1.0, 0.0),
            (4, 0.0, 1.0, 0.0),
            (5, 0.0, 0.0, 1.0),
            (6, 1.0, 0.0, 1.0),
            (7, 1.0, 1.0, 1.0),
            (8, 0.0, 1.0, 1.0),
        ])
        deck.prop_solid(1, "SOLID_PROP")
        deck.part(1, "PART_SOLID", 1, 20)
        deck.brick(1, [(1, 1, 2, 3, 4, 5, 6, 7, 8)])

        p = tmp_path / "M540_BIPHAS_0000.rad"
        deck.write(str(p))
        model = parse_starter_deck(str(p))

        assert 20 in model.materials
        mat = model.materials[20]
        assert mat.law == 37
        assert math.isclose(mat.rho0, 997.0, rel_tol=1e-5)
        assert math.isclose(mat.params["rho_l0"], 997.0, rel_tol=1e-5)

    def test_starter_check_rejects_law37_on_shell(self):
        """Shell elements referencing LAW37 should trigger a starter check error."""
        import numpy as np
        model = Model()
        model.add_nodes(np.array([1, 2, 3, 4]), np.zeros((4, 3)))
        mat = Material(
            id=1,
            law=37,
            rho0=1000.0,
            params={"rho_l0": 1000.0, "c_l": 2.2e9, "alpha1": 1.0, "rho_g0": 1.2, "gamma_g": 1.4},
        )
        model.materials[1] = mat

        class FakeShellGroup:
            def __init__(self):
                self.state = {"slices": [(slice(0, 1), mat, None)]}

        model.element_groups = lambda: [("shells", FakeShellGroup())]

        log = MessageLog()
        check_model(model, log)
        assert any("is not ported for shells elements" in msg for msg in log.errors)
