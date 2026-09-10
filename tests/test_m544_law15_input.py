"""
Tests for Milestone M544: /MAT/LAW15 (/MAT/CHANG, /MAT/PLAS_ANISO, /MAT/COMP_CHANG) input layer.

Covers:
  1. Card layout constants and aliases (MAT_LAW15_1..9, MAT_CHANG_*, MAT_PLAS_ANISO_*, MAT_COMP_CHANG_*).
  2. Starter reading of /MAT/LAW15, /MAT/CHANG, /MAT/PLAS_ANISO, /MAT/COMP_CHANG in fixed format.
  3. Starter reading in free format (comma-separated and whitespace-separated).
  4. Starter validation checks (check_mat_law15):
     - Negative and zero density (rho0 <= 0)
     - Negative and zero Young's moduli (E1 <= 0, E2 <= 0)
     - Invalid Poisson's ratio determinant (detc = 1 - nu12*nu21 <= 0)
     - Negative and zero shear moduli (G12 <= 0, G23 <= 0, G31 <= 0)
     - Negative yield stresses (sig_1yt, sig_2yt, sig_1yc, sig_2yc, sig_12yc, sig_12yt <= 0)
     - Hardening parameter bounds (b > 1.0, n > 1.0)
     - Failure strengths and relaxation time bounds (S1, S2, C1, C2, S12 <= 0, tmax <= 0)
  5. Element family compatibility in check_model:
     - Accepted element families: shells, shells_qbat, shells_qeph, sh3n, quads
     - Rejected element families: bricks, tetras, penta6, pyra5, trusses, beams, springs
"""

from __future__ import annotations

import math
from pathlib import Path
import numpy as np
import pytest

from pyradioss.common.messages import MessageLog
from pyradioss.input import card_layouts as cl
from pyradioss.input.deck_reader import read_deck
from pyradioss.input.starter_keywords import parse_starter_deck
from pyradioss.model.model import Model
from pyradioss.model.entities import Material, MatLaw15, MatChang, MatPlasAniso, MatCompChang
from pyradioss.starter.checks import _ALLOWED_LAWS, check_mat_law15, check_model


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

class TestLaw15CardLayouts:
    """Card layout constants and LAYOUTS dictionary verification (M544)."""

    def test_constants_and_aliases(self):
        assert cl.MAT_LAW15_1 == (20, 20)
        assert cl.MAT_LAW15_2 == (20, 20, 20)
        assert cl.MAT_LAW15_3 == (20, 20, 20)
        assert cl.MAT_LAW15_4 == (20, 20, 20)
        assert cl.MAT_LAW15_5 == (20, 20, 10)
        assert cl.MAT_LAW15_6 == (20, 20, 20, 20, 20)
        assert cl.MAT_LAW15_7 == (20, 20, 20, 20, 10)
        assert cl.MAT_LAW15_8 == (20, 20, 20, 20, 20)
        assert cl.MAT_LAW15_9 == (10, 20, 20, 20)

        for i in range(1, 10):
            law_val = getattr(cl, f"MAT_LAW15_{i}")
            assert getattr(cl, f"MAT_CHANG_{i}") == law_val
            assert getattr(cl, f"MAT_PLAS_ANISO_{i}") == law_val
            assert getattr(cl, f"MAT_COMP_CHANG_{i}") == law_val
            assert getattr(cl, f"MAT_LAW15_CFG_{i}") == law_val
            assert getattr(cl, f"MAT_CHANG_CFG_{i}") == law_val
            assert getattr(cl, f"MAT_PLAS_ANISO_CFG_{i}") == law_val
            assert getattr(cl, f"MAT_COMP_CHANG_CFG_{i}") == law_val

    def test_layouts_dict_entries(self):
        for i in range(1, 10):
            expected = list(getattr(cl, f"MAT_LAW15_{i}"))
            assert cl.LAYOUTS[f"MAT_LAW15_{i}"] == expected
            assert cl.LAYOUTS[f"MAT_CHANG_{i}"] == expected
            assert cl.LAYOUTS[f"MAT_PLAS_ANISO_{i}"] == expected
            assert cl.LAYOUTS[f"MAT_COMP_CHANG_{i}"] == expected


# ============================================================================
# 2. Starter Reading (Fixed and Free Formats & Aliases)
# ============================================================================

class TestLaw15Reader:
    """Fixed- and free-format starter reader tests for LAW15 and aliases."""

    def test_read_law15_fixed_format(self, tmp_path: Path):
        """Read 9-card /MAT/LAW15 in fixed format."""
        deck = """
/BEGIN
LAW15_FIXED_TEST
                  -1
/MAT/LAW15/1
Test_Law15
#              RHO_I               RHO_0
              1.5e-9              1.5e-9
#                E11                 E22                NU12
            140000.0             10000.0                 0.3
#                G12                 G23                 G31
              5000.0              3000.0              5000.0
#                  B                   N                FMAX
                 0.5                 0.8                10.0
#              WPMAX               WPREF                IOFF
                 5.0                 1.0                   2
#            SIG_1YT             SIG_2YT             SIG_1YC             SIG_2YC               ALPHA
              1500.0                50.0              1200.0               200.0                 1.0
#           SIG_12YC            SIG_12YT                   C           EPS_DOT_0                 ICC
                80.0                80.0                 0.0                 1.0                   1
#               BETA                TMAX                  S1                  S2                 S12
                 1.0              1.0e-4              1500.0                50.0                80.0
#            FSMOOTH                FCUT                  C1                  C2
                   0                 0.0              1200.0               200.0
/END
"""
        model, log = _parse_string(tmp_path, deck)
        assert len(log.errors) == 0, f"Errors: {log.errors}"
        assert 1 in model.mat_law15s
        mat = model.mat_law15s[1]
        assert mat.id == 1
        assert mat.rho0 == 1.5e-9
        assert mat.rhor == 1.5e-9
        assert mat.e11 == 140000.0
        assert mat.e22 == 10000.0
        assert mat.nu12 == 0.3
        assert mat.g12 == 5000.0
        assert mat.g23 == 3000.0
        assert mat.g31 == 5000.0
        assert mat.b == 0.5
        assert mat.n == 0.8
        assert mat.fmax == 10.0
        assert mat.wpmax == 5.0
        assert mat.wpref == 1.0
        assert mat.ioff == 2
        assert mat.sig_1yt == 1500.0
        assert mat.sig_2yt == 50.0
        assert mat.sig_1yc == 1200.0
        assert mat.sig_2yc == 200.0
        assert mat.alpha == 1.0
        assert mat.sig_12yc == 80.0
        assert mat.sig_12yt == 80.0
        assert mat.icc == 1
        assert mat.beta == 1.0
        assert mat.tmax == 1.0e-4
        assert mat.s1 == 1500.0
        assert mat.s2 == 50.0
        assert mat.s12 == 80.0
        assert mat.c1 == 1200.0
        assert mat.c2 == 200.0

        # Physical material in model.materials
        assert 1 in model.materials
        act_mat = model.materials[1]
        assert act_mat.law == 15
        assert act_mat.params["E1"] == 140000.0
        assert act_mat.params["E2"] == 10000.0

    def test_read_chang_alias_fixed(self, tmp_path: Path):
        """Read /MAT/CHANG alias in fixed format."""
        deck = """
/BEGIN
CHANG_FIXED_TEST
                  -1
/MAT/CHANG/2
Test_Chang
              1.6e-9              1.6e-9
            150000.0              9000.0                0.32
              4500.0              2800.0              4500.0
                 0.4                 0.7                12.0
                 4.0                 1.0                   1
              1600.0                40.0              1300.0               180.0                 1.0
                70.0                70.0                 0.1                 1.0                   1
                 1.0              2.0e-4              1600.0                40.0                70.0
                   0                 0.0              1300.0               180.0
/END
"""
        model, log = _parse_string(tmp_path, deck)
        assert len(log.errors) == 0, f"Errors: {log.errors}"
        assert 2 in model.mat_law15s
        mat = model.mat_law15s[2]
        assert mat.e11 == 150000.0
        assert mat.e22 == 9000.0
        assert mat.nu12 == 0.32
        assert mat.sig_1yt == 1600.0
        assert 2 in model.materials
        assert model.materials[2].law == 15

    def test_read_plas_aniso_alias_fixed(self, tmp_path: Path):
        """Read /MAT/PLAS_ANISO alias in fixed format."""
        deck = """
/BEGIN
PLAS_ANISO_TEST
                  -1
/MAT/PLAS_ANISO/3
Test_Plas_Aniso
              1.5e-9              1.5e-9
            130000.0              8000.0                0.28
              4000.0              2500.0              4000.0
                 0.3                 0.9                 8.0
                 6.0                 1.0                   0
              1400.0                45.0              1100.0               190.0                 1.0
                75.0                75.0                 0.0                 1.0                   1
                 1.0              1.5e-4              1400.0                45.0                75.0
                   0                 0.0              1100.0               190.0
/END
"""
        model, log = _parse_string(tmp_path, deck)
        assert len(log.errors) == 0, f"Errors: {log.errors}"
        assert 3 in model.mat_law15s
        assert model.mat_law15s[3].e11 == 130000.0
        assert 3 in model.materials
        assert model.materials[3].law == 15

    def test_read_comp_chang_alias_fixed(self, tmp_path: Path):
        """Read /MAT/COMP_CHANG alias in fixed format."""
        deck = """
/BEGIN
COMP_CHANG_TEST
                  -1
/MAT/COMP_CHANG/4
Test_Comp_Chang
              1.5e-9              1.5e-9
            145000.0              9500.0                0.31
              4800.0              2900.0              4800.0
                 0.5                 0.8                10.0
                 5.0                 1.0                   2
              1550.0                48.0              1250.0               210.0                 1.0
                85.0                85.0                 0.0                 1.0                   1
                 1.0              1.0e-4              1550.0                48.0                85.0
                   0                 0.0              1250.0               210.0
/END
"""
        model, log = _parse_string(tmp_path, deck)
        assert len(log.errors) == 0, f"Errors: {log.errors}"
        assert 4 in model.mat_law15s
        assert model.mat_law15s[4].e11 == 145000.0
        assert 4 in model.materials
        assert model.materials[4].law == 15

    def test_read_law15_free_format_comma(self, tmp_path: Path):
        """Read /MAT/LAW15 in comma-separated free format."""
        deck = """
/BEGIN
LAW15_FREE_TEST
/MAT/LAW15/5
Free_Comma_Law15
1.5e-9, 1.5e-9
140000.0, 10000.0, 0.3
5000.0, 3000.0, 5000.0
0.5, 0.8, 10.0
5.0, 1.0, 2
1500.0, 50.0, 1200.0, 200.0, 1.0
80.0, 80.0, 0.0, 1.0, 1
1.0, 1.0e-4, 1500.0, 50.0, 80.0
0, 0.0, 1200.0, 200.0
/END
"""
        model, log = _parse_string(tmp_path, deck)
        assert len(log.errors) == 0, f"Errors: {log.errors}"
        assert 5 in model.mat_law15s
        mat = model.mat_law15s[5]
        assert mat.rho0 == 1.5e-9
        assert mat.e11 == 140000.0
        assert mat.e22 == 10000.0
        assert mat.nu12 == 0.3
        assert mat.g12 == 5000.0
        assert mat.s1 == 1500.0
        assert mat.tmax == 1.0e-4

    def test_read_chang_free_format_whitespace(self, tmp_path: Path):
        """Read /MAT/CHANG in whitespace-separated free format."""
        deck = """
/BEGIN
CHANG_FREE_WHITESPACE
/MAT/CHANG/6
Free_Space_Chang
1.5e-9 1.5e-9
140000.0 10000.0 0.3
5000.0 3000.0 5000.0
0.5 0.8 10.0
5.0 1.0 2
1500.0 50.0 1200.0 200.0 1.0
80.0 80.0 0.0 1.0 1
1.0 1.0e-4 1500.0 50.0 80.0
0 0.0 1200.0 200.0
/END
"""
        model, log = _parse_string(tmp_path, deck)
        assert len(log.errors) == 0, f"Errors: {log.errors}"
        assert 6 in model.mat_law15s
        mat = model.mat_law15s[6]
        assert mat.rho0 == 1.5e-9
        assert mat.e11 == 140000.0
        assert mat.nu12 == 0.3
        assert mat.s1 == 1500.0


# ============================================================================
# 3. Parameter Validation Checks (check_mat_law15)
# ============================================================================

class TestLaw15StarterChecks:
    """Starter checks parameter validation for /MAT/LAW15 (M544)."""

    def _valid_params(self) -> dict:
        return {
            "rho0": 1.5e-9,
            "e11": 140000.0,
            "e22": 10000.0,
            "nu12": 0.3,
            "g12": 5000.0,
            "g23": 3000.0,
            "g31": 5000.0,
            "sig_1yt": 1500.0,
            "sig_1yc": 1200.0,
            "sig_2yt": 50.0,
            "sig_2yc": 200.0,
            "sig_12yc": 80.0,
            "sig_12yt": 80.0,
            "b": 0.5,
            "n": 0.8,
            "s1": 1500.0,
            "s2": 50.0,
            "s12": 80.0,
            "c1": 1200.0,
            "c2": 200.0,
            "tmax": 1.0e-4,
        }

    def test_valid_law15_passes(self):
        mat = Material(id=1, law=15, rho0=1.5e-9, params=self._valid_params())
        log = MessageLog()
        check_mat_law15(mat, log)
        assert len(log.errors) == 0

    def test_negative_density_rejected(self):
        p = self._valid_params()
        p["rho0"] = -1.5e-9
        mat = Material(id=1, law=15, rho0=-1.5e-9, params=p)
        log = MessageLog()
        check_mat_law15(mat, log)
        assert any("initial density RHO must be > 0" in e for e in log.errors)

    def test_zero_moduli_rejected(self):
        p = self._valid_params()
        p["e11"] = 0.0
        mat = Material(id=1, law=15, rho0=1.5e-9, params=p)
        log = MessageLog()
        check_mat_law15(mat, log)
        assert any("Young's modulus E1 must be > 0" in e for e in log.errors)

        p2 = self._valid_params()
        p2["e22"] = -100.0
        mat2 = Material(id=2, law=15, rho0=1.5e-9, params=p2)
        log2 = MessageLog()
        check_mat_law15(mat2, log2)
        assert any("Young's modulus E2 must be > 0" in e for e in log2.errors)

    def test_invalid_poisson_ratio_rejected(self):
        p = self._valid_params()
        p["nu12"] = 4.0
        mat = Material(id=1, law=15, rho0=1.5e-9, params=p)
        log = MessageLog()
        check_mat_law15(mat, log)
        assert any("detc = 1 - nu12*nu21 must be > 0" in e for e in log.errors)

    def test_zero_shear_modulus_rejected(self):
        p = self._valid_params()
        p["g12"] = 0.0
        mat = Material(id=1, law=15, rho0=1.5e-9, params=p)
        log = MessageLog()
        check_mat_law15(mat, log)
        assert any("shear modulus G12 must be > 0" in e for e in log.errors)

        p2 = self._valid_params()
        p2["g23"] = -1.0
        mat2 = Material(id=2, law=15, rho0=1.5e-9, params=p2)
        log2 = MessageLog()
        check_mat_law15(mat2, log2)
        assert any("shear modulus G23 must be > 0" in e for e in log2.errors)

        p3 = self._valid_params()
        p3["g31"] = 0.0
        mat3 = Material(id=3, law=15, rho0=1.5e-9, params=p3)
        log3 = MessageLog()
        check_mat_law15(mat3, log3)
        assert any("shear modulus G31 must be > 0" in e for e in log3.errors)

    def test_negative_yield_stress_rejected(self):
        p = self._valid_params()
        p["sig_1yt"] = -500.0
        mat = Material(id=1, law=15, rho0=1.5e-9, params=p)
        log = MessageLog()
        check_mat_law15(mat, log)
        assert any("tensile yield stress in dir 1 (sig_1yt) must be > 0" in e for e in log.errors)

        p2 = self._valid_params()
        p2["sig_12yc"] = 0.0
        mat2 = Material(id=2, law=15, rho0=1.5e-9, params=p2)
        log2 = MessageLog()
        check_mat_law15(mat2, log2)
        assert any("compressive shear yield stress in dir 12 (sig_12yc) must be > 0" in e for e in log2.errors)

    def test_hardening_bounds_rejected(self):
        p = self._valid_params()
        p["b"] = 1.5
        mat = Material(id=1, law=15, rho0=1.5e-9, params=p)
        log = MessageLog()
        check_mat_law15(mat, log)
        assert any("hardening parameter b must be <= 1.0" in e for e in log.errors)

        p2 = self._valid_params()
        p2["n"] = 1.2
        mat2 = Material(id=2, law=15, rho0=1.5e-9, params=p2)
        log2 = MessageLog()
        check_mat_law15(mat2, log2)
        assert any("hardening exponent n must be <= 1.0" in e for e in log2.errors)

    def test_failure_parameters_rejected(self):
        p = self._valid_params()
        p["s1"] = -100.0
        mat = Material(id=1, law=15, rho0=1.5e-9, params=p)
        log = MessageLog()
        check_mat_law15(mat, log)
        assert any("longitudinal tensile strength S1 must be > 0" in e for e in log.errors)

        p2 = self._valid_params()
        p2["tmax"] = 0.0
        mat2 = Material(id=2, law=15, rho0=1.5e-9, params=p2)
        log2 = MessageLog()
        check_mat_law15(mat2, log2)
        assert any("stress relaxation time tmax must be > 0" in e for e in log2.errors)


# ============================================================================
# 4. Element Family Compatibility (check_model)
# ============================================================================

class TestLaw15ElementCompatibility:
    """Verify LAW15 element family compatibility in check_model."""

    def test_allowed_on_shells(self):
        """LAW15 is allowed on shells, shells_qbat, shells_qeph, sh3n, quads."""
        for fam in ("shells", "shells_qbat", "shells_qeph", "sh3n", "quads"):
            assert 15 in _ALLOWED_LAWS[fam]
            assert "15" in _ALLOWED_LAWS[fam]
            assert "LAW15" in _ALLOWED_LAWS[fam]
            assert "CHANG" in _ALLOWED_LAWS[fam]
            assert "PLAS_ANISO" in _ALLOWED_LAWS[fam]
            assert "COMP_CHANG" in _ALLOWED_LAWS[fam]

    @pytest.mark.parametrize("family", ["bricks", "tetras", "penta6", "pyra5", "trusses", "beams", "springs"])
    def test_rejected_on_solids_trusses_beams_springs(self, family: str):
        """LAW15 applied to 3D solids/beams/trusses/springs causes check_model fatal error."""
        model = Model()
        model.add_nodes(np.array([1, 2]), np.zeros((2, 3)))
        mat = Material(id=1, law=15, rho0=1.5e-9, params={"e11": 1e5, "e22": 1e4, "g12": 1e4, "sig_1yt": 100})
        model.materials[1] = mat

        class FakeGroup:
            def __init__(self):
                self.state = {"slices": [(slice(0, 1), mat, None)]}

        model.element_groups = lambda: [(family, FakeGroup())]

        log = MessageLog()
        check_model(model, log)
        assert any(f"is not supported for {family} elements" in e for e in log.errors)
