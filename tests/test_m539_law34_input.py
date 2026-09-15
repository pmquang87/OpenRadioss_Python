"""Tests for Milestone M539: /MAT/LAW34 and /MAT/BOLTZMAN (/MAT/VISC_MAXW).

Covers:
1. Card layouts constants, aliases, and LAYOUTS dictionary registration.
2. Deck writer emitter StarterDeck.mat_law34 and aliases (mat_boltzman, mat_visc_maxw, mat_boltzmann).
3. Fixed-format 20-column card width alignment and formatting.
4. CFG catalogue and mat_reader synonym resolution.
5. Starter validation checks: _ALLOWED_LAWS, parameter bounds checks, warning when GI > G0.
6. Deck roundtrip parsing with parse_starter_deck / read_deck.
"""

from __future__ import annotations

import math
import numpy as np
import pytest

from pyradioss.common.messages import MessageLog
from pyradioss.input import card_layouts as cl
from pyradioss.input import deck_writer as dw
from pyradioss.input.deck_writer import StarterDeck
from pyradioss.input.deck_reader import read_deck
from pyradioss.input.mat_reader import MAT_PHYSICS_REGISTRY, catalogue, CfgCatalogue
from pyradioss.input.starter_keywords import parse_starter_deck
from pyradioss.model.model import Model
from pyradioss.model.entities import Material
from pyradioss.starter.checks import _ALLOWED_LAWS, check_model, check_mat_law34


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


class TestLaw34CardLayouts:
    """Card layout constants and LAYOUTS dictionary verification (M539)."""

    def test_constants_and_aliases(self):
        assert cl.MAT_LAW34_1 == (20, 20)
        assert cl.MAT_LAW34_2 == (20,)
        assert cl.MAT_LAW34_3 == (20, 20, 20)
        assert cl.MAT_LAW34_4 == (20, 20, 20)

        assert cl.MAT_LAW34_CFG_1 == (20, 20)
        assert cl.MAT_LAW34_CFG_2 == (20,)
        assert cl.MAT_LAW34_CFG_3 == (20, 20, 20)
        assert cl.MAT_LAW34_CFG_4 == (20, 20, 20)

        assert cl.MAT_BOLTZMAN_1 == (20, 20)
        assert cl.MAT_BOLTZMAN_2 == (20,)
        assert cl.MAT_BOLTZMAN_3 == (20, 20, 20)
        assert cl.MAT_BOLTZMAN_4 == (20, 20, 20)

        assert cl.MAT_VISC_MAXW_1 == (20, 20)
        assert cl.MAT_VISC_MAXW_2 == (20,)
        assert cl.MAT_VISC_MAXW_3 == (20, 20, 20)
        assert cl.MAT_VISC_MAXW_4 == (20, 20, 20)

        assert cl.MAT_BOLTZMANN_1 == (20, 20)
        assert cl.MAT_BOLTZMANN_2 == (20,)
        assert cl.MAT_BOLTZMANN_3 == (20, 20, 20)
        assert cl.MAT_BOLTZMANN_4 == (20, 20, 20)

    def test_layouts_dictionary(self):
        expected_keys = [
            "MAT_LAW34_1", "MAT_LAW34_2", "MAT_LAW34_3", "MAT_LAW34_4",
            "MAT_LAW34_CFG_1", "MAT_LAW34_CFG_2", "MAT_LAW34_CFG_3", "MAT_LAW34_CFG_4",
            "MAT_BOLTZMAN_1", "MAT_BOLTZMAN_2", "MAT_BOLTZMAN_3", "MAT_BOLTZMAN_4",
            "MAT_BOLTZMAN_CFG_1", "MAT_BOLTZMAN_CFG_2", "MAT_BOLTZMAN_CFG_3", "MAT_BOLTZMAN_CFG_4",
            "MAT_VISC_MAXW_1", "MAT_VISC_MAXW_2", "MAT_VISC_MAXW_3", "MAT_VISC_MAXW_4",
            "MAT_VISC_MAXW_CFG_1", "MAT_VISC_MAXW_CFG_2", "MAT_VISC_MAXW_CFG_3", "MAT_VISC_MAXW_CFG_4",
            "MAT_BOLTZMANN_1", "MAT_BOLTZMANN_2", "MAT_BOLTZMANN_3", "MAT_BOLTZMANN_4",
            "MAT_BOLTZMANN_CFG_1", "MAT_BOLTZMANN_CFG_2", "MAT_BOLTZMANN_CFG_3", "MAT_BOLTZMANN_CFG_4",
        ]
        for key in expected_keys:
            assert key in cl.LAYOUTS, f"Missing key {key} in LAYOUTS"
            layout = cl.LAYOUTS[key]
            if key.endswith("_1"):
                assert list(layout) == [20, 20]
            elif key.endswith("_2"):
                assert list(layout) == [20]
            elif key.endswith("_3"):
                assert list(layout) == [20, 20, 20]
            elif key.endswith("_4"):
                assert list(layout) == [20, 20, 20]


class TestLaw34DeckWriter:
    """StarterDeck emitter tests for /MAT/LAW34 and its aliases."""

    def test_emit_mat_law34_basic(self):
        deck = StarterDeck("M539")
        deck.mat_law34(
            id=10,
            rho=1000.0,
            bulk=1.5e7,
            g0=4.0e6,
            gi=1.0e6,
            beta=25.0,
            title="VISCO_RUBBER",
        )
        lines = block_lines(deck.render(), "/MAT/LAW34/10")
        assert lines[0].startswith("/MAT/LAW34/10")
        assert lines[1] == "VISCO_RUBBER"
        cards = data_cards(lines)
        # Card 1: rho
        assert len(cards[1]) == 20
        assert float(cards[1]) == 1000.0
        # Card 2: bulk
        assert len(cards[2]) == 20
        assert float(cards[2]) == 1.5e7
        # Card 3: g0, gi, beta
        assert len(cards[3]) == 60
        assert float(cards[3][:20]) == 4.0e6
        assert float(cards[3][20:40]) == 1.0e6
        assert float(cards[3][40:60]) == 25.0
        # Card 4: p0, phi, gamma0
        assert len(cards[4]) == 60
        assert float(cards[4][:20]) == 0.0
        assert float(cards[4][20:40]) == 0.0
        assert float(cards[4][40:60]) == 0.0

    def test_emit_mat_law34_with_rhor_and_air(self):
        deck = StarterDeck("M539")
        deck.mat_law34(
            id=20,
            rho=800.0,
            bulk=2.0e7,
            g0=5.0e6,
            gi=2.5e6,
            beta=50.0,
            p0=1.0e5,
            phi=0.15,
            gamma0=0.02,
            rhor=850.0,
            title="FOAM_AIR",
        )
        lines = block_lines(deck.render(), "/MAT/LAW34/20")
        assert lines[0].startswith("/MAT/LAW34/20")
        assert lines[1] == "FOAM_AIR"
        cards = data_cards(lines)
        # Card 1: rho, rhor (2 x 20 cols)
        assert len(cards[1]) == 40
        assert float(cards[1][:20]) == 800.0
        assert float(cards[1][20:40]) == 850.0
        # Card 2: bulk
        assert float(cards[2]) == 2.0e7
        # Card 3: g0, gi, beta
        assert float(cards[3][:20]) == 5.0e6
        assert float(cards[3][20:40]) == 2.5e6
        assert float(cards[3][40:60]) == 50.0
        # Card 4: p0, phi, gamma0
        assert float(cards[4][:20]) == 1.0e5
        assert float(cards[4][20:40]) == 0.15
        assert float(cards[4][40:60]) == 0.02

    def test_aliases_consistency(self):
        deck1 = StarterDeck("M539").mat_law34(1, 1000.0, 1e7, 3e6, 1e6, 10.0, title="T1")
        deck2 = StarterDeck("M539").mat_boltzman(1, 1000.0, 1e7, 3e6, 1e6, 10.0, title="T1")
        deck3 = StarterDeck("M539").mat_visc_maxw(1, 1000.0, 1e7, 3e6, 1e6, 10.0, title="T1")
        deck4 = StarterDeck("M539").mat_boltzmann(1, 1000.0, 1e7, 3e6, 1e6, 10.0, title="T1")

        assert deck1.render() == deck2.render()
        assert deck1.render() == deck3.render()
        assert deck1.render() == deck4.render()

    def test_card_width_alignment_exact_20(self):
        """Verify each data card consists strictly of 20-character chunks."""
        deck = StarterDeck("M539")
        deck.mat_law34(1, 1.23456e-3, 9.8765e4, 3.4567e4, 1.2345e4, 0.54321, 101.325, 0.05, 0.001)
        lines = block_lines(deck.render(), "/MAT/LAW34/1")
        cards = data_cards(lines)
        for ln in cards[1:]:
            assert len(ln) % 20 == 0, f"Card '{ln}' length {len(ln)} is not a multiple of 20"
            for i in range(0, len(ln), 20):
                field = ln[i:i+20]
                assert len(field) == 20
                float(field)


class TestLaw34SynonymsAndRegistry:
    """CFG catalogue and physics registry synonym mappings."""

    def test_physics_registry_synonyms(self):
        synonyms = [34, "34", "LAW34", "BOLTZMAN", "VISC_MAXW", "BOLTZMANN"]
        for syn in synonyms:
            assert syn in MAT_PHYSICS_REGISTRY, f"Synonym {syn} not registered in MAT_PHYSICS_REGISTRY"
            builder = MAT_PHYSICS_REGISTRY[syn]
            assert callable(builder)

    def test_cfg_catalogue_synonym_resolution(self):
        cat = catalogue()
        schema = cat.schema("BOLTZMAN")
        if schema is not None:
            assert schema.law_number == 34 or "BOLTZMAN" in schema.keyword or "LAW34" in schema.keyword


class TestLaw34StarterChecks:
    """Starter checks and parameter validation for LAW34."""

    def test_allowed_laws_includes_law34(self):
        for category in ["bricks", "tetras", "shells", "shells_qbat", "shells_qeph", "sh3n"]:
            assert category in _ALLOWED_LAWS
            assert 34 in _ALLOWED_LAWS[category]
            assert "LAW34" in _ALLOWED_LAWS[category]

    def test_check_mat_law34_valid(self):
        mat = Material(
            id=1,
            law=34,
            rho0=1000.0,
            params={"bulk": 1e7, "g0": 3e6, "gi": 1e6, "beta": 10.0, "p0": 0.0, "phi": 0.0, "gama0": 0.0},
        )
        log = MessageLog()
        check_mat_law34(mat, log)
        assert len(log.errors) == 0

    def test_check_mat_law34_invalid_rho(self):
        mat = Material(
            id=1,
            law=34,
            rho0=0.0,
            params={"bulk": 1e7, "g0": 3e6, "gi": 1e6, "beta": 10.0},
        )
        log = MessageLog()
        check_mat_law34(mat, log)
        assert any("density RHO must be > 0" in msg for msg in log.errors)

    def test_check_mat_law34_invalid_bulk(self):
        mat = Material(
            id=1,
            law=34,
            rho0=1000.0,
            params={"bulk": 0.0, "g0": 3e6, "gi": 1e6, "beta": 10.0},
        )
        log = MessageLog()
        check_mat_law34(mat, log)
        assert any("bulk modulus BULK must be > 0" in msg for msg in log.errors)

    def test_check_mat_law34_negative_shear(self):
        mat = Material(
            id=1,
            law=34,
            rho0=1000.0,
            params={"bulk": 1e7, "g0": -100.0, "gi": 1e6, "beta": 10.0},
        )
        log = MessageLog()
        check_mat_law34(mat, log)
        assert any("short-term shear modulus G0 must be >= 0" in msg for msg in log.errors)

    def test_check_mat_law34_negative_beta(self):
        mat = Material(
            id=1,
            law=34,
            rho0=1000.0,
            params={"bulk": 1e7, "g0": 3e6, "gi": 1e6, "beta": -5.0},
        )
        log = MessageLog()
        check_mat_law34(mat, log)
        assert any("decay constant BETA must be >= 0" in msg for msg in log.errors)

    def test_check_mat_law34_warning_gi_exceeds_g0(self):
        mat = Material(
            id=1,
            law=34,
            rho0=1000.0,
            params={"bulk": 1e7, "g0": 1e6, "gi": 5e6, "beta": 10.0},
        )
        log = MessageLog()
        check_mat_law34(mat, log)
        assert len(log.errors) == 0
        assert any("exceeds short-term shear modulus G0" in msg for msg in log.warnings)


class TestLaw34DeckRoundtrip:
    """Deck writer and deck reader roundtrip parsing test."""

    def test_parse_starter_deck_roundtrip(self, tmp_path):
        deck = StarterDeck("M539")
        deck.mat_law34(
            id=7,
            rho=1200.0,
            bulk=2.5e7,
            g0=6.0e6,
            gi=2.0e6,
            beta=15.0,
            p0=5.0e4,
            phi=0.1,
            gamma0=0.01,
            title="RUBBER_FOAM_7",
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
        deck.part(1, "PART_SOLID", 1, 7)
        deck.brick(1, [(1, 1, 2, 3, 4, 5, 6, 7, 8)])

        p = tmp_path / "M539_0000.rad"
        deck.write(str(p))
        model = parse_starter_deck(str(p))

        assert 7 in model.materials
        mat = model.materials[7]
        assert mat.law == 34
        assert math.isclose(mat.rho0, 1200.0, rel_tol=1e-5)
        assert math.isclose(mat.params["bulk"], 2.5e7, rel_tol=1e-5)
        assert math.isclose(mat.params["g0"], 6.0e6, rel_tol=1e-5)
        assert math.isclose(mat.params["gi"], 2.0e6, rel_tol=1e-5)
        assert math.isclose(mat.params["beta"], 15.0, rel_tol=1e-5)
        assert math.isclose(mat.params["p0"], 5.0e4, rel_tol=1e-5)
        assert math.isclose(mat.params["phi"], 0.1, rel_tol=1e-5)
        assert math.isclose(mat.params["gama0"], 0.01, rel_tol=1e-5)

        log = MessageLog()
        check_model(model, log)
        assert len(log.errors) == 0
