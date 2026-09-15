"""Milestone M549: /MAT/LAW82 (/MAT/OGDEN, /MAT/LAW82_OGDEN) Roundtrip, Starter Checks & Negative Testing Suite.

Audits:
  1. DeckWriter roundtrip:
     - Test StarterDeck.mat_law82 and aliases mat_ogden, mat_law82_ogden.
     - Generate decks with N = 1, 2, 3, 4, 5, 8, 10 terms.
     - Parse back with read_deck + parse_starter_deck.
     - Assert all parameters match: id, rho0, rhor, nu, nordre, mu, alpha, d.
     - Test both fixed format (cards with 20-char columns) and free format (comma/space separated).
     - Test entity instance passing and legacy card invocation.

  2. Starter bounds checks (check_mat_law82 in pyradioss/starter/checks.py):
     - Density rho0 <= 0 must generate MSGERROR.
     - Order N < 1 must generate MSGERROR (MSGID 559).
     - Shear modulus sum(mu_i) <= 0 must generate MSGERROR (MSGID 846).
     - Poisson ratio nu < 0 or nu >= 0.5 must generate MSGERROR.
     - Term alpha_i = 0 must generate error/warning.
     - Test direct check_mat_law82 and check_model/check_materials integration.

  3. Element compatibility:
     - Verify solids (BRICK, HEPH, TETRA4, TETRA10, PENTA6, pyra5) are allowed.
     - Verify shells (SHELL, QEPH, QBAT, SH3N, quads) are allowed.
     - Verify unsupported elements (BEAM, TRUSS, SPRING) are rejected with clear diagnostic messages.
     - Verify _ALLOWED_LAWS dictionary entries for law 82 and its aliases.

  4. Restart (.rst) file serialization:
     - MatLaw82 dataclass entity pickled and deserialized with all properties preserved.
     - OgdenParams object built by build_law82 pickled and deserialized.
     - extra_shapes dictionary for nip=None and nip=5 pickled and deserialized.
     - Element state arrays (uvar82, stresses, strains) pickled and deserialized with assert_allclose.
     - Complete Model containing LAW82 / OGDEN materials and aliases pickled and restored.
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import List
import numpy as np
import pytest

from pyradioss.common.messages import MessageLog
from pyradioss.input import card_layouts as cl
from pyradioss.input.deck_reader import read_deck
from pyradioss.input.deck_writer import StarterDeck
from pyradioss.input.starter_keywords import parse_starter_deck
from pyradioss.materials.law82_ogden import (
    OgdenParams,
    build_law82,
    extra_shapes,
    solid_sound_speed,
    shell_sound_speed,
    solid_update,
    shell_update,
)
from pyradioss.model.entities import Material, MatLaw82
from pyradioss.model.model import Model
from pyradioss.starter.checks import (
    _ALLOWED_LAWS,
    check_mat_law82,
    check_materials,
    check_model,
)


def _parse_deck_string(tmp_path: Path, text: str) -> tuple[Model, MessageLog]:
    """Helper to write and parse a deck string into Model and MessageLog."""
    deck_path = tmp_path / "ROUNDTRIP_LAW82_0000.rad"
    deck_path.write_text(text.strip() + "\n", encoding="utf-8")
    blocks = read_deck(str(deck_path))
    log = MessageLog()
    model = Model()
    parse_starter_deck(blocks, model, log)
    return model, log


def _get_mat_cards(deck_str: str) -> List[str]:
    """Extract material data cards from rendered deck (skipping header and title)."""
    lines = deck_str.splitlines()
    mat_idx = next(i for i, line in enumerate(lines) if line.startswith("/MAT/"))
    cards = []
    for line in lines[mat_idx + 2:]:
        if line.startswith("/"):
            break
        if not line.startswith("#"):
            cards.append(line)
    return cards


# ============================================================================
# 1. DeckWriter Fixed-Format Roundtrip Audit (N = 1, 2, 3, 4, 5, 8, 10)
# ============================================================================

class TestLaw82DeckWriterRoundtripFixed:
    """Audit of StarterDeck.mat_law82 and aliases with fixed-format 20-char columns."""

    @pytest.mark.parametrize("n_order", [1, 2, 3, 4, 5, 8, 10])
    def test_deckwriter_roundtrip_n_terms(self, tmp_path: Path, n_order: int):
        """Generate decks with N = 1, 2, 3, 4, 5, 8, 10 terms, parse back and assert all parameters match."""
        # Generate synthetic distinct values for mu, alpha, d
        mu = [float(10.0 + i * 2.5) for i in range(n_order)]
        alpha = [float(1.5 + i * 0.5) for i in range(n_order)]
        d = [float(0.001 * (i + 1)) for i in range(n_order)]

        deck = StarterDeck(f"LAW82_N{n_order}").mat_law82(
            mid=82,
            title=f"Ogden_Model_N{n_order}",
            rho0=1.15e-9,
            rhor=1.20e-9,
            nu=0.485,
            nordre=n_order,
            mu=mu,
            alpha=alpha,
            d=d,
        )

        rendered = deck.render()
        cards = _get_mat_cards(rendered)

        # Verify fixed-format card structure
        # Card 1: rho0, rhor (20 chars each)
        assert len(cards[0]) >= 40
        c0_rho = float(cards[0][:20].strip())
        c0_rhor = float(cards[0][20:40].strip())
        assert c0_rho == pytest.approx(1.15e-9)
        assert c0_rhor == pytest.approx(1.20e-9)

        # Card 2: nordre (10 chars), 10 blank spaces, nu (20 chars)
        assert len(cards[1]) >= 40
        c1_nordre = int(cards[1][:10].strip())
        c1_nu = float(cards[1][20:40].strip())
        assert c1_nordre == n_order
        assert c1_nu == pytest.approx(0.485)

        # Verify parsing back via read_deck + parse_starter_deck
        model, log = _parse_deck_string(tmp_path, rendered)
        assert len(log.errors) == 0, f"Errors parsing N={n_order} deck: {log.errors}"
        assert 82 in model.mat_law82s
        m = model.mat_law82s[82]

        assert m.id == 82
        assert m.title == f"Ogden_Model_N{n_order}"
        assert m.rho0 == pytest.approx(1.15e-9)
        assert m.rhor == pytest.approx(1.20e-9)
        assert m.nu == pytest.approx(0.485)
        assert m.nordre == n_order
        assert len(m.mu) == n_order
        assert len(m.alpha) == n_order
        assert len(m.d) == n_order

        for i in range(n_order):
            assert m.mu[i] == pytest.approx(mu[i])
            assert m.alpha[i] == pytest.approx(alpha[i])
            assert m.d[i] == pytest.approx(d[i])

        # Verify Material container attributes
        mat = model.materials[82]
        assert mat.law == 82
        assert mat.params["nordre"] == n_order
        assert mat.params["nu"] == pytest.approx(0.485)
        np.testing.assert_allclose(mat.params["mu"], mu)
        np.testing.assert_allclose(mat.params["alpha"], alpha)
        np.testing.assert_allclose(mat.params["d"], d)

    def test_deckwriter_alias_mat_ogden(self, tmp_path: Path):
        """Test StarterDeck.mat_ogden alias creates /MAT/OGDEN header and roundtrips."""
        deck = StarterDeck("OGDEN_ALIAS").mat_ogden(
            mid=101,
            title="Rubber_Ogden_Alias",
            rho0=0.95e-9,
            nu=0.492,
            nordre=3,
            mu=[12.0, 3.5, -1.2],
            alpha=[2.0, 4.0, -2.0],
            d=[0.005, 0.002, 0.001],
        )
        rendered = deck.render()
        assert "/MAT/OGDEN/101" in rendered

        model, log = _parse_deck_string(tmp_path, rendered)
        assert len(log.errors) == 0
        assert 101 in model.mat_law82s
        m = model.mat_law82s[101]
        assert m.id == 101
        assert m.nordre == 3
        assert m.nu == pytest.approx(0.492)
        assert m.mu == pytest.approx([12.0, 3.5, -1.2])
        assert m.alpha == pytest.approx([2.0, 4.0, -2.0])
        assert m.d == pytest.approx([0.005, 0.002, 0.001])

    def test_deckwriter_alias_mat_law82_ogden(self, tmp_path: Path):
        """Test StarterDeck.mat_law82_ogden alias creates /MAT/LAW82_OGDEN header and roundtrips."""
        deck = StarterDeck("LAW82_OGDEN_ALIAS").mat_law82_ogden(
            mid=102,
            title="Rubber_Law82_Ogden_Alias",
            rho0=1.05e-9,
            nu=0.480,
            nordre=2,
            mu=[8.5, 1.8],
            alpha=[1.3, -2.5],
            d=[0.004, 0.001],
        )
        rendered = deck.render()
        assert "/MAT/LAW82_OGDEN/102" in rendered

        model, log = _parse_deck_string(tmp_path, rendered)
        assert len(log.errors) == 0
        assert 102 in model.mat_law82s
        m = model.mat_law82s[102]
        assert m.id == 102
        assert m.nordre == 2
        assert m.nu == pytest.approx(0.480)
        assert m.mu == pytest.approx([8.5, 1.8])
        assert m.alpha == pytest.approx([1.3, -2.5])
        assert m.d == pytest.approx([0.004, 0.001])

    def test_deckwriter_from_mat_law82_instance(self, tmp_path: Path):
        """Pass a MatLaw82 dataclass instance directly to StarterDeck.mat_law82."""
        orig = MatLaw82(
            id=55,
            title="Entity_Direct_Export",
            rho0=1.12e-9,
            rhor=1.12e-9,
            nu=0.490,
            nordre=3,
            mu=[14.0, 2.8, -0.9],
            alpha=[1.8, 3.2, -1.5],
            d=[0.008, 0.003, 0.001],
        )
        deck = StarterDeck("ENTITY_TEST").mat_law82(orig)
        model, log = _parse_deck_string(tmp_path, deck.render())
        assert len(log.errors) == 0
        restored = model.mat_law82s[55]
        assert restored.id == orig.id
        assert restored.title == orig.title
        assert restored.rho0 == pytest.approx(orig.rho0)
        assert restored.nu == pytest.approx(orig.nu)
        assert restored.nordre == orig.nordre
        assert restored.mu == pytest.approx(orig.mu)
        assert restored.alpha == pytest.approx(orig.alpha)
        assert restored.d == pytest.approx(orig.d)

    def test_deckwriter_legacy_raw_cards(self, tmp_path: Path):
        """Test legacy/raw card invocation: mat_law82(mid, title, data_cards)."""
        cards = [
            f"{1.2e-9:20.6e}{1.2e-9:20.6e}",
            f"{2:10d}          {0.485:20.6f}",
            f"{10.0:20.6f}{5.0:20.6f}",
            f"{2.0:20.6f}{-2.0:20.6f}",
            f"{0.01:20.6f}{0.005:20.6f}",
        ]
        deck = StarterDeck("LEGACY_TEST").mat_law82(77, "Legacy_Raw_Rubber", cards)
        model, log = _parse_deck_string(tmp_path, deck.render())
        assert len(log.errors) == 0
        m = model.mat_law82s[77]
        assert m.id == 77
        assert m.nordre == 2
        assert m.mu == pytest.approx([10.0, 5.0])
        assert m.alpha == pytest.approx([2.0, -2.0])
        assert m.d == pytest.approx([0.01, 0.005])


# ============================================================================
# 2. Free-Format Roundtrip Audit (Space and Comma-Delimited)
# ============================================================================

class TestLaw82FreeFormatRoundtrip:
    """Audit free format parsing for /MAT/LAW82 and /MAT/OGDEN."""

    def test_free_format_space_delimited_multi_card(self, tmp_path: Path):
        """Space-delimited free format with multi-card lists (N=4 terms)."""
        deck = """#RADIOSS STARTER
/BEGIN
Test Free Space Multi Card
                  10                   0
/MAT/LAW82/10
Space Delimited Rubber
1.1e-9 1.1e-9
4 0.490
15.0 8.0 2.5 0.5
1.5 3.0 -2.0 -4.0
0.01 0.005 0.002 0.001
"""
        model, log = _parse_deck_string(tmp_path, deck)
        assert len(log.errors) == 0
        assert 10 in model.mat_law82s
        m = model.mat_law82s[10]
        assert m.id == 10
        assert m.rho0 == pytest.approx(1.1e-9)
        assert m.rhor == pytest.approx(1.1e-9)
        assert m.nordre == 4
        assert m.nu == pytest.approx(0.490)
        assert m.mu == pytest.approx([15.0, 8.0, 2.5, 0.5])
        assert m.alpha == pytest.approx([1.5, 3.0, -2.0, -4.0])
        assert m.d == pytest.approx([0.01, 0.005, 0.002, 0.001])

    def test_free_format_comma_space_delimited(self, tmp_path: Path):
        """Comma-and-space delimited free format (/MAT/OGDEN with N=3 terms)."""
        deck = """#RADIOSS STARTER
/BEGIN
Test Free Comma Space
                  10                   0
/MAT/OGDEN/20
Comma Space Rubber
0.98e-9, 0.98e-9
3, 0.485
12.5, 4.2, -1.5
2.2, 5.0, -2.5
0.008, 0.004, 0.002
"""
        model, log = _parse_deck_string(tmp_path, deck)
        assert len(log.errors) == 0
        assert 20 in model.mat_law82s
        m = model.mat_law82s[20]
        assert m.id == 20
        assert m.rho0 == pytest.approx(0.98e-9)
        assert m.nordre == 3
        assert m.nu == pytest.approx(0.485)
        assert m.mu == pytest.approx([12.5, 4.2, -1.5])
        assert m.alpha == pytest.approx([2.2, 5.0, -2.5])
        assert m.d == pytest.approx([0.008, 0.004, 0.002])

    def test_free_format_compact_comma_no_space(self, tmp_path: Path):
        """Compact comma-separated format without space delimiters."""
        deck = """#RADIOSS STARTER
/BEGIN
Test Free Compact Comma
                  10                   0
/MAT/LAW82/30
Compact Comma Rubber
1.05e-9,1.05e-9
2,0.492
18.0,6.0
2.0,-2.0
0.015,0.005
"""
        model, log = _parse_deck_string(tmp_path, deck)
        assert len(log.errors) == 0
        assert 30 in model.mat_law82s
        m = model.mat_law82s[30]
        assert m.id == 30
        assert m.rho0 == pytest.approx(1.05e-9)
        assert m.nordre == 2
        assert m.nu == pytest.approx(0.492)
        assert m.mu == pytest.approx([18.0, 6.0])
        assert m.alpha == pytest.approx([2.0, -2.0])
        assert m.d == pytest.approx([0.015, 0.005])

    def test_free_format_order1_single_data_card(self, tmp_path: Path):
        """Order N=1 single data card (mu, alpha, d on a single line)."""
        deck = """#RADIOSS STARTER
/BEGIN
Test Free Order 1 Single Card
                  10                   0
/MAT/LAW82/40
Single Term Rubber
1.0e-9 1.0e-9
1 0.495
20.0 2.0 0.012
"""
        model, log = _parse_deck_string(tmp_path, deck)
        assert len(log.errors) == 0
        assert 40 in model.mat_law82s
        m = model.mat_law82s[40]
        assert m.id == 40
        assert m.nordre == 1
        assert m.nu == pytest.approx(0.495)
        assert m.mu == pytest.approx([20.0])
        assert m.alpha == pytest.approx([2.0])
        assert m.d == pytest.approx([0.012])


# ============================================================================
# 3. Starter Bounds Checks Audit (check_mat_law82)
# ============================================================================

class TestLaw82StarterBoundsChecks:
    """Audit of check_mat_law82 in pyradioss/starter/checks.py."""

    def test_bounds_valid_passes_cleanly(self):
        """A properly configured MatLaw82 produces 0 errors."""
        mat = MatLaw82(
            id=1,
            rho0=1.0e-9,
            nu=0.485,
            nordre=2,
            mu=[10.0, 2.0],
            alpha=[2.0, -2.0],
            d=[0.01, 0.005],
        )
        log = MessageLog()
        check_mat_law82(mat, log)
        assert len(log.errors) == 0, f"Unexpected errors on valid mat: {log.errors}"

    def test_bounds_rho0_nonpositive(self):
        """Density rho0 <= 0 must generate MSGERROR."""
        for bad_rho in (0.0, -1.0e-9, -5.0):
            mat = MatLaw82(
                id=1,
                rho0=bad_rho,
                nu=0.485,
                nordre=1,
                mu=[10.0],
                alpha=[2.0],
            )
            log = MessageLog()
            check_mat_law82(mat, log)
            assert any("initial density RHO must be > 0" in e for e in log.errors)

    def test_bounds_order_less_than_one(self):
        """Order N < 1 must generate MSGERROR with MSGID 559."""
        for bad_order in (0, -1, -5):
            mat = MatLaw82(
                id=1,
                rho0=1.0e-9,
                nu=0.485,
                nordre=bad_order,
                mu=[10.0],
                alpha=[2.0],
            )
            log = MessageLog()
            check_mat_law82(mat, log)
            assert any(
                "order of Ogden model must be >= 1" in e and "MSGID 559" in e
                for e in log.errors
            ), f"Expected MSGID 559 error for order {bad_order}, got {log.errors}"

    def test_bounds_shear_modulus_sum_nonpositive(self):
        """Shear modulus sum(mu_i) <= 0 must generate MSGERROR with MSGID 846."""
        # Single term nonpositive
        for bad_mu in ([0.0], [-10.0]):
            mat = MatLaw82(
                id=1,
                rho0=1.0e-9,
                nu=0.485,
                nordre=1,
                mu=bad_mu,
                alpha=[2.0],
            )
            log = MessageLog()
            check_mat_law82(mat, log)
            assert any(
                "sum of shear moduli mu must be > 0" in e and "MSGID 846" in e
                for e in log.errors
            ), f"Expected MSGID 846 error for mu {bad_mu}, got {log.errors}"

        # Multi-term sum <= 0
        mat2 = MatLaw82(
            id=2,
            rho0=1.0e-9,
            nu=0.485,
            nordre=3,
            mu=[5.0, -8.0, 2.0],  # sum = -1.0
            alpha=[2.0, 4.0, -2.0],
        )
        log2 = MessageLog()
        check_mat_law82(mat2, log2)
        assert any(
            "sum of shear moduli mu must be > 0" in e and "MSGID 846" in e
            for e in log2.errors
        )

    def test_bounds_poisson_ratio_out_of_range(self):
        """Poisson ratio nu < 0 or nu >= 0.5 must generate MSGERROR."""
        for bad_nu in (-0.05, -0.5, 0.50, 0.55, 0.99):
            mat = MatLaw82(
                id=1,
                rho0=1.0e-9,
                nu=bad_nu,
                nordre=1,
                mu=[10.0],
                alpha=[2.0],
            )
            log = MessageLog()
            check_mat_law82(mat, log)
            assert any(
                "Poisson ratio nu must satisfy 0.0 <= nu < 0.5" in e
                for e in log.errors
            ), f"Expected error for bad nu {bad_nu}, got {log.errors}"

    def test_bounds_alpha_zero(self):
        """Term alpha_i = 0 must generate error."""
        # First term zero
        mat1 = MatLaw82(
            id=1,
            rho0=1.0e-9,
            nu=0.485,
            nordre=1,
            mu=[10.0],
            alpha=[0.0],
        )
        log1 = MessageLog()
        check_mat_law82(mat1, log1)
        assert any("alpha parameter at term 1 must be non-zero" in e for e in log1.errors)

        # Second term zero in multi-term model
        mat2 = MatLaw82(
            id=2,
            rho0=1.0e-9,
            nu=0.485,
            nordre=3,
            mu=[10.0, 5.0, 2.0],
            alpha=[2.0, 0.0, -2.0],
        )
        log2 = MessageLog()
        check_mat_law82(mat2, log2)
        assert any("alpha parameter at term 2 must be non-zero" in e for e in log2.errors)

    def test_bounds_multiple_violations_reported(self):
        """Check that all violations are accumulated into MessageLog."""
        bad_mat = MatLaw82(
            id=99,
            rho0=-1.0,      # bad density
            nu=0.6,         # bad nu
            nordre=0,       # bad order (MSGID 559)
            mu=[-10.0],     # bad mu (MSGID 846)
            alpha=[0.0],    # bad alpha
        )
        log = MessageLog()
        check_mat_law82(bad_mat, log)
        assert len(log.errors) >= 4
        assert any("initial density" in e for e in log.errors)
        assert any("MSGID 559" in e for e in log.errors)
        assert any("MSGID 846" in e for e in log.errors)
        assert any("Poisson ratio" in e for e in log.errors)

    def test_check_materials_integration(self):
        """check_materials traverses model.materials and model.mat_law82s."""
        model = Model()
        bad_mat = MatLaw82(id=1, rho0=0.0, nu=0.485, nordre=1, mu=[10.0], alpha=[2.0])
        model.mat_law82s[1] = bad_mat
        log = MessageLog()
        check_materials(model, log)
        assert any("initial density RHO must be > 0" in e for e in log.errors)


# ============================================================================
# 4. Element Compatibility Audit
# ============================================================================

class TestLaw82ElementCompatibility:
    """Audit element family compatibility and restrictions for /MAT/LAW82."""

    def test_allowed_solids_and_shells(self):
        """Solids (bricks, tetras, penta6, pyra5) and shells (shells, shells_qbat, shells_qeph, sh3n, quads) pass."""
        allowed_families = (
            "bricks",
            "tetras",
            "penta6",
            "pyra5",
            "shells",
            "shells_qbat",
            "shells_qeph",
            "sh3n",
            "quads",
        )

        for fam in allowed_families:
            assert 82 in _ALLOWED_LAWS[fam], f"82 missing in _ALLOWED_LAWS[{fam}]"
            assert "82" in _ALLOWED_LAWS[fam]
            assert "LAW82" in _ALLOWED_LAWS[fam]
            assert "OGDEN" in _ALLOWED_LAWS[fam]
            assert "LAW82_OGDEN" in _ALLOWED_LAWS[fam]

            model = Model()
            model.add_nodes(np.arange(1, 9), np.zeros((8, 3)))
            mat = Material(
                id=1,
                law=82,
                rho0=1.0e-9,
                params={
                    "nordre": 1,
                    "ORDER": 1,
                    "nu": 0.485,
                    "MAT_NU": 0.485,
                    "mu": [10.0],
                    "Mu_arr": [10.0],
                    "alpha": [2.0],
                    "Alpha_arr": [2.0],
                    "d": [0.01],
                    "Gamma_arr": [0.01],
                },
            )
            model.materials[1] = mat

            class FakeGroup:
                def __init__(self):
                    self.state = {"slices": [(slice(0, 1), mat, None)]}

            model.element_groups = lambda f=fam: [(f, FakeGroup())]
            log = MessageLog()
            check_model(model, log)
            assert len(log.errors) == 0, f"Unexpected error for allowed family {fam}: {log.errors}"

    def test_unsupported_elements_rejected(self):
        """Unsupported element families (beams, trusses, springs) are strictly rejected with clear diagnostic."""
        unsupported = ("beams", "trusses", "springs")

        mat = Material(
            id=1,
            law=82,
            law_name="LAW82",
            rho0=1.0e-9,
            params={
                "nordre": 1,
                "ORDER": 1,
                "nu": 0.485,
                "MAT_NU": 0.485,
                "mu": [10.0],
                "Mu_arr": [10.0],
                "alpha": [2.0],
                "Alpha_arr": [2.0],
                "d": [0.01],
                "Gamma_arr": [0.01],
            },
        )

        class FakeGroup:
            def __init__(self):
                self.state = {"slices": [(slice(0, 1), mat, None)]}

        for fam in unsupported:
            model = Model()
            model.add_nodes(np.arange(1, 9), np.zeros((8, 3)))
            model.materials[1] = mat
            model.element_groups = lambda f=fam: [(f, FakeGroup())]
            log = MessageLog()
            check_model(model, log)
            assert any(
                "/MAT/LAW82/1 (/MAT/OGDEN) is not supported for" in e and fam in e
                for e in log.errors
            ), f"Unsupported family {fam} was not properly rejected: {log.errors}"

    def test_allowed_laws_mapping_consistency(self):
        """Verify _ALLOWED_LAWS dictionary entries for law 82."""
        for fam in ("bricks", "tetras", "penta6", "pyra5", "shells", "shells_qbat", "shells_qeph", "sh3n", "quads"):
            assert 82 in _ALLOWED_LAWS[fam]
        assert 82 not in _ALLOWED_LAWS["beams"]
        assert 82 not in _ALLOWED_LAWS["trusses"]


# ============================================================================
# 5. Restart (.rst) File Serialization Audit
# ============================================================================

class TestLaw82RestartAndSerialization:
    """Verify state pickling and unpickling fidelity for Material, MatLaw82, and extra_shapes arrays."""

    def test_extra_shapes_pickling(self):
        """extra_shapes dictionary for nip=None and nip=5 survives pickling and unpickling."""
        # Unlayered / solid formulation (nip=None)
        s_solid = extra_shapes(None)
        raw_solid = pickle.dumps(s_solid)
        res_solid = pickle.loads(raw_solid)
        assert res_solid == s_solid
        assert res_solid["uvar82"] == (1,)

        # Layered shell formulation (nip=5 integration points)
        s_shell = extra_shapes(None, nip=5)
        raw_shell = pickle.dumps(s_shell)
        res_shell = pickle.loads(raw_shell)
        assert res_shell == s_shell
        assert res_shell["uvar82"] == (5, 1)

    def test_mat_law82_entity_pickling(self):
        """MatLaw82 entity dataclass survives pickling and unpickling with all properties intact."""
        orig = MatLaw82(
            id=82,
            title="Entity_Pickle_Test",
            rho0=1.1e-9,
            rhor=1.15e-9,
            nu=0.488,
            nordre=3,
            mu=[14.0, 3.2, -1.1],
            alpha=[1.8, 4.2, -2.1],
            d=[0.008, 0.003, 0.001],
        )

        raw = pickle.dumps(orig)
        restored = pickle.loads(raw)

        assert restored.id == orig.id
        assert restored.title == orig.title
        assert restored.rho0 == pytest.approx(orig.rho0)
        assert restored.rhor == pytest.approx(orig.rhor)
        assert restored.nu == pytest.approx(orig.nu)
        assert restored.nordre == orig.nordre
        assert restored.mu == pytest.approx(orig.mu)
        assert restored.alpha == pytest.approx(orig.alpha)
        assert restored.d == pytest.approx(orig.d)

        # Properties
        assert restored.G == pytest.approx(orig.G)
        assert restored.K == pytest.approx(orig.K)
        assert restored.E == pytest.approx(orig.E)
        assert restored.rho == pytest.approx(orig.rho)
        assert restored.ref_rho == pytest.approx(orig.ref_rho)
        assert restored.order == orig.order

    def test_ogden_params_built_material_pickling(self):
        """OgdenParams object constructed via build_law82 survives pickling and unpickling."""
        params = build_law82(
            id=82,
            rho0=1.05e-9,
            rhor=1.05e-9,
            nu=0.490,
            nordre=3,
            mu=[12.0, 3.5, 1.0],
            alpha=[1.3, 4.0, -2.0],
            d=[0.006, 0.002, 0.001],
            title="Built_Ogden_Params",
        )

        raw = pickle.dumps(params)
        restored = pickle.loads(raw)

        assert restored.id == params.id
        assert restored.rho0 == pytest.approx(params.rho0)
        assert restored.rhor == pytest.approx(params.rhor)
        assert restored.nu == pytest.approx(params.nu)
        assert restored.nordre == params.nordre
        np.testing.assert_allclose(restored.mu, params.mu)
        np.testing.assert_allclose(restored.alpha, params.alpha)
        np.testing.assert_allclose(restored.d, params.d)
        assert restored.g0 == pytest.approx(params.g0)
        assert restored.rbulk == pytest.approx(params.rbulk)
        assert restored.K == pytest.approx(params.K)
        assert restored.G == pytest.approx(params.G)

        # Sound speeds match
        c_solid_orig = solid_sound_speed(params)
        c_solid_rest = solid_sound_speed(restored)
        assert c_solid_rest == pytest.approx(c_solid_orig)

        c_shell_orig = shell_sound_speed(params)
        c_shell_rest = shell_sound_speed(restored)
        assert c_shell_rest == pytest.approx(c_shell_orig)

    def test_element_state_arrays_pickling(self):
        """Element persistent state arrays (uvar82, stresses, strains) survive restart pickling."""
        n_elem = 20
        rng = np.random.default_rng(42)

        # Extra state array for LAW82: out-of-plane stretch / internal variable uvar82
        state = {
            "uvar82_solid": rng.uniform(0.9, 1.1, size=(n_elem, 1)),
            "uvar82_shell": rng.uniform(0.85, 1.15, size=(n_elem, 5, 1)),
            "sig_solid": rng.uniform(-50.0, 50.0, size=(n_elem, 6)),
            "sig_shell": rng.uniform(-40.0, 40.0, size=(n_elem, 3)),
            "eps_solid": rng.uniform(-0.1, 0.1, size=(n_elem, 6)),
            "eps_shell": rng.uniform(-0.08, 0.08, size=(n_elem, 3)),
        }

        raw = pickle.dumps(state)
        restored = pickle.loads(raw)

        for k, v in state.items():
            assert k in restored
            np.testing.assert_allclose(restored[k], v, err_msg=f"State array {k} mismatch after unpickling")

    def test_complete_model_pickling_with_law82(self):
        """A complete Model containing LAW82 / OGDEN materials and aliases pickles and restores cleanly."""
        model = Model()
        model.add_nodes(np.arange(1, 9), np.zeros((8, 3)))

        mat_entity = MatLaw82(
            id=82,
            title="Model_Pickle_Ogden",
            rho0=1.1e-9,
            nu=0.485,
            nordre=2,
            mu=[15.0, 3.0],
            alpha=[2.0, -2.0],
            d=[0.01, 0.005],
        )
        model.mat_law82s[82] = mat_entity
        model.materials[82] = build_law82(mat_entity)

        # Verify alias in model before pickling
        assert model.mat_ogdens is model.mat_law82s

        raw = pickle.dumps(model)
        restored = pickle.loads(raw)

        assert 82 in restored.mat_law82s
        assert 82 in restored.materials
        assert restored.mat_ogdens is restored.mat_law82s

        rest_ent = restored.mat_law82s[82]
        assert rest_ent.id == 82
        assert rest_ent.nordre == 2
        assert rest_ent.mu == pytest.approx([15.0, 3.0])
        assert rest_ent.alpha == pytest.approx([2.0, -2.0])

        rest_mat = restored.materials[82]
        assert rest_mat.id == 82
        assert solid_sound_speed(rest_mat) == pytest.approx(solid_sound_speed(model.materials[82]))
        assert shell_sound_speed(rest_mat) == pytest.approx(shell_sound_speed(model.materials[82]))
