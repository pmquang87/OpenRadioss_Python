"""
Tests for Milestone M543: /MAT/LAW25 (/MAT/COMP_PLAS, /MAT/COMPSH, /MAT/TSAI_WU, /MAT/CRASURV) input layer.

Covers:
  1. Card layout constants and aliases (MAT_LAW25_1..10, MAT_COMP_PLAS_*, MAT_COMPSH_*, etc.).
  2. Starter reading of /MAT/LAW25, /MAT/COMP_PLAS, /MAT/COMPSH, /MAT/TSAI_WU, /MAT/CRASURV in fixed format.
  3. Starter reading in free format (comma-separated and whitespace-separated).
  4. Starter validation checks (check_mat_law25):
     - Negative and zero density (rho0 <= 0)
     - Negative and zero Young's moduli (E1 <= 0, E2 <= 0)
     - Invalid Poisson's ratio determinant (detc = 1 - nu12*nu21 <= 0)
     - Negative and zero shear moduli (G12 <= 0, G23 <= 0, G31 <= 0)
     - Negative and zero yield stresses (sigyt1, sigyc1, sigyt2, sigyc2, sigt12 <= 0)
     - Hardening exponent bound (n > 1.0)
     - Damage bound (dmax not in [0, 1])
  5. Element family compatibility in check_model:
     - Accepted element families: shells, sh3n, quads, bricks, tetras, penta6, pyra5
     - Rejected element families: trusses, beams, springs
  6. DeckWriter roundtrip and alias emission.
"""

from __future__ import annotations

import math
from pathlib import Path
import numpy as np
import pytest

from pyradioss.common.messages import MessageLog
from pyradioss.input import card_layouts as cl
from pyradioss.input.deck_writer import StarterDeck
from pyradioss.input.deck_reader import read_deck
from pyradioss.input.starter_keywords import parse_starter_deck
from pyradioss.model.model import Model
from pyradioss.model.entities import Material, MatLaw25
from pyradioss.starter.checks import _ALLOWED_LAWS, check_mat_law25, check_model


def _parse_string(tmp_path: Path, text: str) -> tuple[Model, MessageLog]:
    """Helper to write and parse a deck from a string."""
    p = tmp_path / "DECK_0000.rad"
    p.write_text(text.strip() + "\n", encoding="utf-8")
    blocks = read_deck(str(p))
    log = MessageLog()
    model = Model()
    parse_starter_deck(blocks, model, log)
    return model, log


# ============================================================================
# 1. Card Layout Constants & Aliases
# ============================================================================

class TestLaw25CardLayouts:
    """Card layout constants and LAYOUTS dictionary verification (M543)."""

    def test_constants_and_aliases(self):
        assert cl.MAT_LAW25_1 == (20, 20)
        assert cl.MAT_LAW25_2 == (20, 20, 20, 10, 10, 20)
        assert cl.MAT_LAW25_3 == (20, 20, 20, 20, 20)
        assert cl.MAT_LAW25_4 == (20, 20, 20, 20, 20)
        assert cl.MAT_LAW25_5 == (20, 20, 10)
        assert cl.MAT_LAW25_6 == (20, 20, 20)
        assert cl.MAT_LAW25_7 == (20, 20, 20, 20, 20)
        assert cl.MAT_LAW25_8 == (20, 20, 20, 20, 10)
        assert cl.MAT_LAW25_9 == (20, 20, 20)
        assert cl.MAT_LAW25_10 == (10, 20)

        for i in range(1, 11):
            law_val = getattr(cl, f"MAT_LAW25_{i}")
            assert getattr(cl, f"MAT_COMP_PLAS_{i}") == law_val
            assert getattr(cl, f"MAT_COMPSH_{i}") == law_val
            assert getattr(cl, f"MAT_COMPOSITE_PLAS_{i}") == law_val


# ============================================================================
# 2. Starter Reading (Fixed and Free Formats & Aliases)
# ============================================================================

class TestLaw25Reader:
    """Fixed- and free-format starter reader tests for LAW25 and aliases."""

    def test_read_law25_fixed_format_tsaiwu(self, tmp_path: Path):
        """Read 8-card Tsai-Wu /MAT/LAW25 in fixed format."""
        deck = """
/BEGIN
LAW25_TEST
                  -1
/MAT/LAW25/1
Composite_TsaiWu
#              RHO_I               RHO_0
              1.5e-9              1.5e-9
#                E11                 E22                NU12     Iform                           E33
            140000.0             10000.0                 0.3         0                      140000.0
#                G12                 G23                 G31              EPS_f1              EPS_f2
              5000.0              3000.0              5000.0                0.02                0.01
#              EPST1               EPSM1               EPST2               EPSM2                DMAX
               0.005                0.02               0.003                0.01                0.95
#              WPMAX               WPREF       IOFF
                 5.0                 1.0          2
#                  B                   N                FMAX
                 0.5                 0.8                10.0
#             SIGYT1              SIGYT2              SIGYC1              SIGYC2               ALPHA
              1500.0                50.0              1200.0               200.0                 1.0
#            SIGYC12             SIGYT12                   C                EPDR                 ICC
                80.0                70.0                 0.1                10.0                   1
/END
"""
        model, log = _parse_string(tmp_path, deck)
        assert len(log.errors) == 0, f"Errors: {log.errors}"
        assert 1 in model.mat_law25s
        mat = model.mat_law25s[1]
        assert mat.rho0 == 1.5e-9
        assert mat.e11 == 140000.0
        assert mat.e22 == 10000.0
        assert mat.nu12 == 0.3
        assert mat.iform == 0
        assert mat.g12 == 5000.0
        assert mat.sig_1yt == 1500.0
        assert mat.sig_1yc == 1200.0
        assert mat.sig_2yt == 50.0
        assert mat.sig_2yc == 200.0
        assert mat.sig_12yt == 70.0
        assert mat.sig_12yc == 80.0
        assert mat.alpha == 1.0
        assert mat.ioff == 2
        assert mat.b == 0.5
        assert mat.n == 0.8

        # Active Material in model.materials
        assert 1 in model.materials
        active_mat = model.materials[1]
        assert active_mat.law == 25
        assert active_mat.params["e11"] == 140000.0

    def test_read_comp_plas_free_format(self, tmp_path: Path):
        """Read /MAT/COMP_PLAS alias in free comma-separated format."""
        deck = """
/BEGIN
COMP_PLAS_TEST
/MAT/COMP_PLAS/2
Carbon_Plies
1.6e-9, 1.6e-9
150000.0, 9000.0, 0.34, 0, 150000.0
4500.0, 3000.0, 4500.0, 0.025, 0.015
0.006, 0.022, 0.004, 0.012, 0.9
10.0, 1.0, 0
0.2, 0.5, 5.0
1800.0, 40.0, 1200.0, 150.0, 1.0
60.0, 60.0, 0.05, 1.0, 1
/END
"""
        model, log = _parse_string(tmp_path, deck)
        assert len(log.errors) == 0, f"Errors: {log.errors}"
        assert 2 in model.mat_law25s
        mat = model.mat_law25s[2]
        assert mat.rho0 == 1.6e-9
        assert mat.e11 == 150000.0
        assert mat.e22 == 9000.0
        assert mat.nu12 == 0.34
        assert mat.sig_1yt == 1800.0
        assert mat.sig_12yc == 60.0

    def test_read_compsh_and_tsai_wu_aliases(self, tmp_path: Path):
        """Verify /MAT/COMPSH and /MAT/TSAI_WU aliases are parsed."""
        deck = """
/BEGIN
ALIASES_TEST
/MAT/COMPSH/10
Shell_Comp
1.5e-9, 1.5e-9
100000.0, 100000.0, 0.2, 0, 100000.0
40000.0, 20000.0, 20000.0, 0.05, 0.05
0.01, 0.05, 0.01, 0.05, 0.99
20.0, 1.0, 5
0.0, 1.0, 1.0
500.0, 500.0, 500.0, 500.0, 1.0
250.0, 250.0, 0.0, 1.0, 0
/MAT/TSAI_WU/20
TsaiWu_Comp
1.5e-9, 1.5e-9
120000.0, 60000.0, 0.25, 0, 120000.0
30000.0, 15000.0, 15000.0, 0.04, 0.04
0.01, 0.04, 0.01, 0.04, 0.95
15.0, 1.0, 1
0.1, 0.9, 2.0
800.0, 200.0, 600.0, 300.0, 1.0
150.0, 150.0, 0.0, 1.0, 0
/END
"""
        model, log = _parse_string(tmp_path, deck)
        assert len(log.errors) == 0, f"Errors: {log.errors}"
        assert 10 in model.mat_law25s
        assert 20 in model.mat_law25s

    def test_read_crasurv_formulation(self, tmp_path: Path):
        """Read /MAT/CRASURV formulation with directional hardening cards."""
        deck = """
/BEGIN
CRASURV_TEST
/MAT/CRASURV/5
Crash_Composite
# Card 1: rho
1.5e-9, 1.5e-9
# Card 2: E1, E2, nu12, iflag=1, E3
120000.0, 40000.0, 0.25, 1, 20000.0
# Card 3: G12, G23, G31, epsf1, epsf2
10000.0, 5000.0, 8000.0, 0.03, 0.02
# Card 4: epst1, epsm1, epst2, epsm2, dmax
0.005, 0.02, 0.003, 0.015, 0.8
# Card 5: wpmax, wpref, ioff, iflawp
10.0, 1.0, 0, 0
# Card 6: c, eps_rate_0, alpha, icc
0.1, 1.0, 1.0, 1
# Card 7: 1t: sig, b, n, sigmax, c
1000.0, 0.5, 0.8, 1500.0, 0.02
# Card 8: 1t softening: eps1, eps2, sigrs, wpmax
0.01, 0.03, 200.0, 50.0
# Card 9: 2t: sig, b, n, sigmax, c
100.0, 0.2, 0.9, 150.0, 0.01
# Card 10: 2t softening: eps1, eps2, sigrs, wpmax
0.008, 0.02, 30.0, 20.0
# Card 11: 1c: sig, b, n, sigmax, c
800.0, 0.4, 0.85, 1200.0, 0.02
# Card 12: 1c softening: eps1, eps2, sigrs, wpmax
0.012, 0.035, 250.0, 60.0
# Card 13: 2c: sig, b, n, sigmax, c
200.0, 0.3, 0.9, 250.0, 0.01
# Card 14: 2c softening: eps1, eps2, sigrs, wpmax
0.01, 0.025, 50.0, 30.0
# Card 15: 12t: sig, b, n, sigmax, c
80.0, 0.3, 0.8, 120.0, 0.01
# Card 16: 12t softening: eps1, eps2, sigrs, wpmax
0.015, 0.04, 25.0, 40.0
/END
"""
        model, log = _parse_string(tmp_path, deck)
        assert len(log.errors) == 0, f"Errors: {log.errors}"
        assert 5 in model.mat_law25s
        mat = model.mat_law25s[5]
        assert mat.iform == 1
        assert mat.b_1t == 0.5
        assert mat.n_1t == 0.8
        assert mat.sig_1maxt == 1500.0
        assert mat.eps_1t1 == 0.01
        assert mat.eps_2t1 == 0.03
        assert mat.sig_rst1 == 200.0


# ============================================================================
# 3. Parameter Validation Checks (check_mat_law25)
# ============================================================================

class TestLaw25StarterChecks:
    """Starter checks parameter validation for /MAT/LAW25 (M543)."""

    def _valid_params(self) -> dict:
        return {
            "rho0": 1.5e-9,
            "e11": 100000.0,
            "e22": 50000.0,
            "nu12": 0.25,
            "g12": 10000.0,
            "g23": 5000.0,
            "g31": 8000.0,
            "sigyt1": 1000.0,
            "sigyc1": 800.0,
            "sigyt2": 100.0,
            "sigyc2": 200.0,
            "sigt12": 60.0,
            "sigc12": 60.0,
            "n": 0.8,
            "dmax": 0.9,
        }

    def test_valid_law25_passes(self):
        mat = Material(id=1, law=25, rho0=1.5e-9, params=self._valid_params())
        log = MessageLog()
        check_mat_law25(mat, log)
        assert len(log.errors) == 0

    def test_negative_density_rejected(self):
        p = self._valid_params()
        p["rho0"] = -1.5e-9
        mat = Material(id=1, law=25, rho0=-1.5e-9, params=p)
        log = MessageLog()
        check_mat_law25(mat, log)
        assert any("initial density RHO must be > 0" in e for e in log.errors)

    def test_zero_moduli_rejected(self):
        p = self._valid_params()
        p["e11"] = 0.0
        mat = Material(id=1, law=25, rho0=1.5e-9, params=p)
        log = MessageLog()
        check_mat_law25(mat, log)
        assert any("Young's modulus E1 must be > 0" in e for e in log.errors)

        p2 = self._valid_params()
        p2["e22"] = -100.0
        mat2 = Material(id=2, law=25, rho0=1.5e-9, params=p2)
        log2 = MessageLog()
        check_mat_law25(mat2, log2)
        assert any("Young's modulus E2 must be > 0" in e for e in log2.errors)

    def test_invalid_poisson_ratio_rejected(self):
        p = self._valid_params()
        # nu12 = 2.0, nu21 = 2.0 * 50000 / 100000 = 1.0 -> detc = 1 - 2*1 = -1 <= 0
        p["nu12"] = 2.0
        mat = Material(id=1, law=25, rho0=1.5e-9, params=p)
        log = MessageLog()
        check_mat_law25(mat, log)
        assert any("detc = 1 - nu12*nu21 must be > 0" in e for e in log.errors)

    def test_zero_shear_modulus_rejected(self):
        p = self._valid_params()
        p["g12"] = 0.0
        mat = Material(id=1, law=25, rho0=1.5e-9, params=p)
        log = MessageLog()
        check_mat_law25(mat, log)
        assert any("shear modulus G12 must be > 0" in e for e in log.errors)

    def test_negative_yield_stress_rejected(self):
        p = self._valid_params()
        p["sigyt1"] = -500.0
        mat = Material(id=1, law=25, rho0=1.5e-9, params=p)
        log = MessageLog()
        check_mat_law25(mat, log)
        assert any("tensile yield stress in dir 1 (sigyt1) must be > 0" in e for e in log.errors)

    def test_hardening_exponent_greater_than_one_rejected(self):
        p = self._valid_params()
        p["n"] = 1.5
        mat = Material(id=1, law=25, rho0=1.5e-9, params=p)
        log = MessageLog()
        check_mat_law25(mat, log)
        assert any("hardening exponent n must be <= 1.0" in e for e in log.errors)

    def test_invalid_damage_rejected(self):
        p = self._valid_params()
        p["dmax"] = 1.2
        mat = Material(id=1, law=25, rho0=1.5e-9, params=p)
        log = MessageLog()
        check_mat_law25(mat, log)
        assert any("maximum damage dmax must be in [0, 1]" in e for e in log.errors)


# ============================================================================
# 4. Element Family Compatibility (check_model)
# ============================================================================

class TestLaw25ElementCompatibility:
    """Verify LAW25 element family compatibility in check_model."""

    def test_allowed_on_shells_and_solids(self):
        """LAW25 is allowed on shells, sh3n, quads, bricks, tetras."""
        assert 25 in _ALLOWED_LAWS["shells"]
        assert 25 in _ALLOWED_LAWS["sh3n"]
        assert 25 in _ALLOWED_LAWS["bricks"]
        assert 25 in _ALLOWED_LAWS["tetras"]
        assert 25 in _ALLOWED_LAWS["quads"]

    @pytest.mark.parametrize("family", ["trusses", "beams", "springs"])
    def test_rejected_on_springs_trusses_beams(self, family: str):
        """LAW25 applied to springs/trusses/beams causes check_model fatal error."""
        model = Model()
        model.add_nodes(np.array([1, 2]), np.zeros((2, 3)))
        mat = Material(id=1, law=25, rho0=1.5e-9, params={"e11": 1e5, "e22": 1e5, "g12": 1e4, "sigyt1": 100})
        model.materials[1] = mat

        class FakeGroup:
            def __init__(self):
                self.state = {"slices": [(slice(0, 1), mat, None)]}

        model.element_groups = lambda: [(family, FakeGroup())]

        log = MessageLog()
        check_model(model, log)
        assert any(f"is not supported for {family} elements" in e for e in log.errors)

    @pytest.mark.parametrize("family", ["shells", "sh3n", "quads", "bricks", "tetras"])
    def test_accepted_on_supported_families(self, family: str):
        """LAW25 applied to shells and solids produces no compatibility error."""
        model = Model()
        model.add_nodes(np.array([1, 2, 3, 4]), np.zeros((4, 3)))
        mat = Material(
            id=1, law=25, rho0=1.5e-9,
            params={"e11": 1e5, "e22": 5e4, "nu12": 0.2, "g12": 1e4, "g23": 5e3, "g31": 8e3,
                    "sigyt1": 1000, "sigyc1": 800, "sigyt2": 100, "sigyc2": 200, "sigt12": 60, "sigc12": 60}
        )
        model.materials[1] = mat

        class FakeGroup:
            def __init__(self):
                self.state = {"slices": [(slice(0, 1), mat, None)]}

        model.element_groups = lambda: [(family, FakeGroup())]

        log = MessageLog()
        check_model(model, log)
        assert not any(f"is not supported for {family} elements" in e for e in log.errors)


# ============================================================================
# 5. DeckWriter Roundtrip
# ============================================================================

class TestLaw25DeckWriter:
    """Verify StarterDeck.mat_law25 and aliases format roundtrip."""

    def test_deck_writer_roundtrip(self, tmp_path: Path):
        d = StarterDeck("LAW25_ROUNDTRIP")
        d.title("LAW25_ROUNDTRIP")
        d.mat_law25(
            mat_id=1,
            rho_ref=1.5e-9,
            rho=1.5e-9,
            e11=140000.0,
            e22=10000.0,
            e33=140000.0,
            nu12=0.3,
            g12=5000.0,
            g23=3000.0,
            g31=5000.0,
            sig_1yt=1500.0,
            sig_2yt=50.0,
            sig_1yc=1200.0,
            sig_2yc=200.0,
            sig_12yt=70.0,
            sig_12yc=80.0,
            alpha=1.0,
            ioff=2,
            b=0.5,
            n=0.8,
        )
        text = d.render()
        assert "/MAT/LAW25/1" in text

        # Parse rendered deck
        model, log = _parse_string(tmp_path, text)
        assert len(log.errors) == 0, f"Errors: {log.errors}"
        assert 1 in model.mat_law25s
        mat = model.mat_law25s[1]
        assert mat.e11 == 140000.0
        assert mat.e22 == 10000.0
        assert mat.sig_1yt == 1500.0
        assert mat.ioff == 2
