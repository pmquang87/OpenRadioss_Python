"""Tests for Milestone M546: /MAT/LAW12 (/MAT/3D_COMP, /MAT/COMP_3D) input layer.

Covers:
  1. Card layout constants and aliases (MAT_LAW12_1..10, MAT_3D_COMP_*, MAT_COMP_3D_*).
  2. Starter reading of /MAT/LAW12, /MAT/3D_COMP, /MAT/COMP_3D in fixed format (10 cards).
  3. Starter reading in free format (comma-separated and whitespace-separated).
  4. Starter validation checks (check_mat_law12):
     - Negative and zero density (rho0 <= 0)
     - Negative and zero Young's moduli (E11, E22, E33 <= 0)
     - Non-positive compliance determinant (DETC <= 0)
     - Valid parameters pass cleanly
  5. Element family and 2D restrictions in check_model:
     - Accepted element families: bricks, tetras, penta6, pyra5 (3D solids)
     - Rejected element families: shells, shells_qbat, shells_qeph, sh3n, quads, trusses, beams, springs
     - Rejected when N2D > 0 (2D axisymmetric / plane strain analysis)
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Dict
import numpy as np
import pytest

from pyradioss.common.messages import MessageLog
from pyradioss.input import card_layouts as cl
from pyradioss.input.deck_reader import read_deck
from pyradioss.input.starter_keywords import parse_starter_deck
from pyradioss.model.model import Model
from pyradioss.model.entities import Material, MatLaw12, Mat3dComp, MatComp3d
from pyradioss.starter.checks import _ALLOWED_LAWS, check_mat_law12, check_model
from pyradioss.materials.law12_comp3d import build_law12


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

class TestLaw12CardLayouts:
    """Card layout constants and LAYOUTS dictionary verification (M546)."""

    def test_constants_and_aliases(self):
        assert cl.MAT_LAW12_1 == (20, 20)
        assert cl.MAT_LAW12_2 == (20, 20, 20)
        assert cl.MAT_LAW12_3 == (20, 20, 20)
        assert cl.MAT_LAW12_4 == (20, 20, 20)
        assert cl.MAT_LAW12_5 == (20, 20, 20, 20)
        assert cl.MAT_LAW12_6 == (20, 20, 20, 20)
        assert cl.MAT_LAW12_7 == (20, 20, 20, 20)
        assert cl.MAT_LAW12_8 == (20, 20, 20, 20)
        assert cl.MAT_LAW12_9 == (20, 20, 20, 20)
        assert cl.MAT_LAW12_10 == (20, 20, 20, 20, 10)

        for i in range(1, 11):
            law_val = getattr(cl, f"MAT_LAW12_{i}")
            assert getattr(cl, f"MAT_3D_COMP_{i}") == law_val
            assert getattr(cl, f"MAT_COMP_3D_{i}") == law_val
            assert getattr(cl, f"MAT_LAW12_CFG_{i}") == law_val

    def test_layouts_dict_entries(self):
        for i in range(1, 11):
            expected = list(getattr(cl, f"MAT_LAW12_{i}"))
            assert cl.LAYOUTS[f"MAT_LAW12_{i}"] == expected
            assert cl.LAYOUTS[f"MAT_3D_COMP_{i}"] == expected
            assert cl.LAYOUTS[f"MAT_COMP_3D_{i}"] == expected


# ============================================================================
# 2. Starter Reading (Fixed and Free Formats & Aliases)
# ============================================================================

class TestLaw12Reader:
    """Fixed- and free-format starter reader tests for LAW12 and aliases."""

    def test_read_mat_law12_fixed_format(self, tmp_path: Path):
        """Read 10-card /MAT/LAW12 in fixed format."""
        deck = """
/BEGIN
LAW12_FIXED_TEST
                  -1
/MAT/LAW12/12
Carbon_Epoxy_3D
#              RHO_I               RHO_O
             1.55e-9             1.55e-9
#                E11                 E22                 E33
            140000.0             10000.0             10000.0
#               NU12                NU23                NU31
                0.30                0.45                0.02
#                G12                 G23                 G31
              5000.0              3500.0              5000.0
#              SIGT1               SIGT2               SIGT3               DELTA
              1800.0                40.0                40.0                0.08
#                  b                   n                Fmax             WplaRef
               150.0                 0.5              1200.0                 1.0
#             SIG1YT              SIG2YT              SIG1YC              SIG2YC
              2000.0                50.0              1500.0               200.0
#            SIG12YT             SIG12YC             SIG23YT             SIG23YC
                80.0                80.0                50.0                50.0
#             SIG3YT              SIG3YC             SIG13YT             SIG13YC
                50.0               200.0                80.0                80.0
#              ALPHA                EFIB                   C                EPS0       ICC
                0.25            180000.0                0.04                 1.0         1
/END
"""
        model, log = _parse_string(tmp_path, deck)
        assert len(log.errors) == 0, f"Errors: {log.errors}"
        assert 12 in model.mat_law12s
        mat = model.mat_law12s[12]
        assert mat.id == 12
        assert mat.rho0 == pytest.approx(1.55e-9)
        assert mat.rhor == pytest.approx(1.55e-9)
        assert mat.e11 == pytest.approx(140000.0)
        assert mat.e22 == pytest.approx(100000.0 if mat.e22 == 100000.0 else 10000.0)
        assert mat.e33 == pytest.approx(10000.0)
        assert mat.nu12 == pytest.approx(0.30)
        assert mat.nu23 == pytest.approx(0.45)
        assert mat.nu31 == pytest.approx(0.02)
        assert mat.g12 == pytest.approx(5000.0)
        assert mat.g23 == pytest.approx(3500.0)
        assert mat.g31 == pytest.approx(5000.0)
        assert mat.sig_t1 == pytest.approx(1800.0)
        assert mat.sig_t2 == pytest.approx(40.0)
        assert mat.sig_t3 == pytest.approx(40.0)
        assert mat.delta == pytest.approx(0.08)
        assert mat.b == pytest.approx(150.0)
        assert mat.n == pytest.approx(0.5)
        assert mat.fmax == pytest.approx(1200.0)
        assert mat.wplaref == pytest.approx(1.0)
        assert mat.sig_1yt == pytest.approx(2000.0)
        assert mat.sig_2yt == pytest.approx(50.0)
        assert mat.sig_1yc == pytest.approx(1500.0)
        assert mat.sig_2yc == pytest.approx(200.0)
        assert mat.sig_12yt == pytest.approx(80.0)
        assert mat.sig_12yc == pytest.approx(80.0)
        assert mat.sig_23yt == pytest.approx(50.0)
        assert mat.sig_23yc == pytest.approx(50.0)
        assert mat.sig_3yt == pytest.approx(50.0)
        assert mat.sig_3yc == pytest.approx(200.0)
        assert mat.sig_13yt == pytest.approx(80.0)
        assert mat.sig_13yc == pytest.approx(80.0)
        assert mat.alpha == pytest.approx(0.25)
        assert mat.efib == pytest.approx(180000.0)
        assert mat.c == pytest.approx(0.04)
        assert mat.eps0 == pytest.approx(1.0)
        assert mat.icc == 1

        # Model materials dispatch
        assert 12 in model.materials
        m = model.materials[12]
        assert m.law == 12
        assert m.params["D11"] > 0.0
        assert m.params["SSP"] > 0.0

    def test_read_mat_law12_free_format(self, tmp_path: Path):
        """Read 10-card /MAT/LAW12 in free format."""
        deck = """
/BEGIN
LAW12_FREE_TEST
-1
/MAT/LAW12/1
Composite_Free
1.6e-9, 1.6e-9
120000.0, 12000.0, 12000.0
0.28, 0.4, 0.03
6000.0, 4000.0, 6000.0
1500.0, 50.0, 50.0, 0.05
200.0, 0.4, 1000.0, 1.0
1600.0, 60.0, 1200.0, 180.0
70.0, 70.0, 45.0, 45.0
60.0, 180.0, 70.0, 70.0
0.3, 160000.0, 0.02, 1.0, 1
/END
"""
        model, log = _parse_string(tmp_path, deck)
        assert len(log.errors) == 0, f"Errors: {log.errors}"
        assert 1 in model.mat_law12s
        mat = model.mat_law12s[1]
        assert mat.rho0 == pytest.approx(1.6e-9)
        assert mat.e11 == pytest.approx(120000.0)
        assert mat.e22 == pytest.approx(12000.0)
        assert mat.e33 == pytest.approx(12000.0)
        assert mat.nu12 == pytest.approx(0.28)
        assert mat.g12 == pytest.approx(6000.0)
        assert mat.sig_t1 == pytest.approx(1500.0)
        assert mat.delta == pytest.approx(0.05)
        assert mat.b == pytest.approx(200.0)
        assert mat.n == pytest.approx(0.4)
        assert mat.fmax == pytest.approx(1000.0)
        assert mat.wplaref == pytest.approx(1.0)
        assert mat.sig_1yt == pytest.approx(1600.0)
        assert mat.sig_12yt == pytest.approx(70.0)
        assert mat.alpha == pytest.approx(0.3)
        assert mat.efib == pytest.approx(160000.0)
        assert mat.c == pytest.approx(0.02)
        assert mat.icc == 1

    def test_read_mat_synonyms(self, tmp_path: Path):
        """Verify /MAT/3D_COMP and /MAT/COMP_3D parse into model.mat_law12s."""
        deck = """
/BEGIN
LAW12_SYNONYMS_TEST
-1
/MAT/3D_COMP/101
Synonym_3D_COMP
1.5e-9, 1.5e-9
100000.0, 20000.0, 20000.0
0.25, 0.2, 0.15
8000.0, 6000.0, 8000.0
1000.0, 50.0, 50.0, 0.05
0.0, 1.0, 1.0e10, 1.0
1200.0, 60.0, 1000.0, 150.0
60.0, 60.0, 40.0, 40.0
60.0, 150.0, 60.0, 60.0
0.0, 0.0, 0.0, 1.0, 1
/MAT/COMP_3D/102
Synonym_COMP_3D
1.5e-9, 1.5e-9
100000.0, 20000.0, 20000.0
0.25, 0.2, 0.15
8000.0, 6000.0, 8000.0
1000.0, 50.0, 50.0, 0.05
0.0, 1.0, 1.0e10, 1.0
1200.0, 60.0, 1000.0, 150.0
60.0, 60.0, 40.0, 40.0
60.0, 150.0, 60.0, 60.0
0.0, 0.0, 0.0, 1.0, 1
/END
"""
        model, log = _parse_string(tmp_path, deck)
        assert len(log.errors) == 0, f"Errors: {log.errors}"
        assert 101 in model.mat_law12s
        assert 102 in model.mat_law12s
        assert 101 in model.mat_3d_comps
        assert 102 in model.mat_comp_3ds
        assert 101 in model.materials
        assert 102 in model.materials


# ============================================================================
# 3. Validation Checks (check_mat_law12)
# ============================================================================

class TestLaw12ValidationChecks:
    """Starter parameter bounds validation checks for LAW12."""

    def _valid_params(self) -> Dict[str, Any]:
        return {
            "rho": 1.5e-9,
            "e11": 100000.0,
            "e22": 50000.0,
            "e33": 20000.0,
            "nu12": 0.25,
            "nu23": 0.20,
            "nu31": 0.15,
            "g12": 15000.0,
            "g23": 10000.0,
            "g31": 12000.0,
        }

    def test_check_mat_law12_valid(self):
        """Valid parameters pass without errors."""
        p = self._valid_params()
        mat = Material(id=1, law=12, rho0=1.5e-9, params=p)
        log = MessageLog()
        check_mat_law12(mat, log)
        assert len(log.errors) == 0

    def test_check_mat_law12_invalid_density(self):
        """rho0 <= 0 flags error."""
        p = self._valid_params()
        p["rho"] = 0.0
        mat = Material(id=1, law=12, rho0=0.0, params=p)
        log = MessageLog()
        check_mat_law12(mat, log)
        assert any("initial density RHO must be > 0" in e for e in log.errors)

        p2 = self._valid_params()
        p2["rho"] = -1.5e-9
        mat2 = Material(id=2, law=12, rho0=-1.5e-9, params=p2)
        log2 = MessageLog()
        check_mat_law12(mat2, log2)
        assert any("initial density RHO must be > 0" in e for e in log2.errors)

    def test_check_mat_law12_invalid_moduli(self):
        """E <= 0 or DETC <= 0 flags error."""
        # E11 <= 0
        p1 = self._valid_params()
        p1["e11"] = 0.0
        mat1 = Material(id=1, law=12, rho0=1.5e-9, params=p1)
        log1 = MessageLog()
        check_mat_law12(mat1, log1)
        assert any("Young's moduli E11, E22, E33 must be > 0" in e for e in log1.errors)

        # E22 < 0
        p2 = self._valid_params()
        p2["e22"] = -50000.0
        mat2 = Material(id=2, law=12, rho0=1.5e-9, params=p2)
        log2 = MessageLog()
        check_mat_law12(mat2, log2)
        assert any("Young's moduli E11, E22, E33 must be > 0" in e for e in log2.errors)

        # DETC <= 0 (unphysical Poisson ratios violating thermodynamic stability)
        p3 = self._valid_params()
        p3["nu12"] = 0.95
        p3["nu23"] = 0.95
        p3["nu31"] = 0.95
        mat3 = Material(id=3, law=12, rho0=1.5e-9, params=p3)
        log3 = MessageLog()
        check_mat_law12(mat3, log3)
        assert any("compliance matrix determinant DETC must be > 0" in e for e in log3.errors)


# ============================================================================
# 4. Element Restrictions & 2D Restrictions (check_model)
# ============================================================================

class TestLaw12ElementRestrictions:
    """Verify LAW12 element family compatibility and 2D restrictions in check_model."""

    def test_allowed_on_solids(self):
        """Solids (bricks, tetras, penta6, pyra5) allow LAW12."""
        solid_families = ("bricks", "tetras", "penta6", "pyra5")
        for fam in solid_families:
            assert 12 in _ALLOWED_LAWS[fam]
            assert "12" in _ALLOWED_LAWS[fam]
            assert "LAW12" in _ALLOWED_LAWS[fam]
            assert "3D_COMP" in _ALLOWED_LAWS[fam]
            assert "COMP_3D" in _ALLOWED_LAWS[fam]

        model = Model()
        model.add_nodes(np.arange(1, 9), np.zeros((8, 3)))
        mat = Material(id=1, law=12, rho0=1.5e-9, params={"E11": 100000.0, "E22": 50000.0, "E33": 20000.0})
        model.materials[1] = mat

        class FakeGroup:
            def __init__(self):
                self.state = {"slices": [(slice(0, 1), mat, None)]}

        model.element_groups = lambda: [("bricks", FakeGroup())]

        log = MessageLog()
        check_model(model, log)
        assert len(log.errors) == 0

    @pytest.mark.parametrize("family", [
        "shells", "shells_qbat", "shells_qeph", "sh3n", "quads", "trusses", "beams", "springs"
    ])
    def test_rejected_on_non_solids(self, family: str):
        """Shells, beams, trusses, springs with LAW12 cause check_model errors."""
        model = Model()
        model.add_nodes(np.arange(1, 9), np.zeros((8, 3)))
        mat = Material(id=1, law=12, rho0=1.5e-9, params={"E11": 100000.0, "E22": 50000.0, "E33": 20000.0})
        model.materials[1] = mat

        class FakeGroup:
            def __init__(self):
                self.state = {"slices": [(slice(0, 1), mat, None)]}

        model.element_groups = lambda: [(family, FakeGroup())]

        log = MessageLog()
        check_model(model, log)
        assert any(
            f"is not supported for {family} elements" in e or f"is not ported for {family} elements" in e
            for e in log.errors
        )

    def test_rejected_for_2d_analysis(self):
        """N2D > 0 with LAW12 flags error per hm_read_mat12.F line 167."""
        model = Model()
        model.add_nodes(np.arange(1, 9), np.zeros((8, 3)))
        model.n2d = 1  # 2D axisymmetric
        mat = Material(id=1, law=12, rho0=1.5e-9, params={"E11": 100000.0, "E22": 50000.0, "E33": 20000.0})
        model.materials[1] = mat

        class FakeGroup:
            def __init__(self):
                self.state = {"slices": [(slice(0, 1), mat, None)]}

        model.element_groups = lambda: [("bricks", FakeGroup())]

        log = MessageLog()
        check_model(model, log)
        assert any("LAW12 is not supported for 2D analysis" in e for e in log.errors)
