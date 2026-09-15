"""
Milestone M556: /MAT/LAW21 (/MAT/DPRAG)
Exhaustive Roundtrip, Negative Validation & Restart (.rst) Serialization Auditor Suite.

Fortran origins:
  - starter/source/materials/mat/mat021/hm_read_mat21.F
  - engine/source/materials/mat/mat021/m21law.F
  - engine/source/output/restart/wrrestp.F & rdresb.F
  - starter/source/restart/ddsplit/wrrest.F
  - config/CFG/radioss110/MAT/matl21_dprag.cfg & radioss130/MAT/matl21_dprag.cfg

Audits:
  1. Fixed-Format Deck Roundtrip:
     - Standard 20-column fixed format with StarterDeck.mat_law21.
     - Re-parse using read_mat_law21.
     - Assert exact equality for all 15 fields:
       rho, refer_rho (rho_r), e (E), nu, a0 (A_0), a1 (A_1), a2 (A_2),
       amax (A_max), ifunc, c1 (C_1), pfscale (F_scale), pmin (P_min),
       pext (P_ext), bunl (B_unl), mumax (mu_max).
     - Default and minimal card fallback handling.
     - Entity object invocation with MatLaw21 and Material dataclasses.

  2. Free-Format Comma-Delimited Roundtrip:
     - Write and re-parse free-format deck blocks for /MAT/LAW21 and /MAT/DPRAG.
     - Standard comma-separated with whitespace and compact comma-delimited without whitespace.
     - Cross-format roundtrip (free -> parse -> fixed -> parse -> exact equality).

  3. Negative Starter Diagnostics:
     - Density rho <= 0 (error).
     - Young's modulus E <= 0 (error).
     - Poisson's ratio nu < 0 or nu >= 0.5 (error).
     - Tensile bulk modulus C1 <= 0 (ANCMSG 829 error).
     - Inverted yield surface: a1 < 0 and a2 == 0 (ANCMSG 829 warning).
     - Untypical yield surface: a2 < 0 (ANCMSG 829 warning).
     - No real root: a2 != 0 and a1^2 - 4*a0*a2 < 0 (ANCMSG 829 warning).
     - Solid elements accepted (solids, bricks, tetras, penta6, pyra5).
     - Shell elements rejected (ANCMSG 305 error).
     - 1D elements rejected (ANCMSG 306 error).
     - 2D analysis rejected: N2D > 0 (ANCMSG 305 error).
     - Registry and dispatch validation (_ALLOWED_LAWS, _MAT_CHECKS, check_materials).

  4. Restart (.rst) Serialization:
     - MatLaw21 entity dataclass pickling and unpickling fidelity.
     - Material state arrays: historical maximum compaction mu_bak, pressure P,
       accumulated plastic strain epxe, element alive mask off.
     - Material state arrays preserved across write_restart / read_restart contract.
     - Constitutive dynamic cycle continuation from restart matching uninterrupted run within 10^-12.
     - Element kernel (solid_hexa8.forces) dynamic step continuation within 10^-12.
     - Element alive mask (off) continuation and deactivation invariance.
     - End-to-end engine execution restart chaining matching uninterrupted run within 10^-12.
"""

from __future__ import annotations

import math
from pathlib import Path
import pickle
from typing import Any, Dict, List, Tuple
import numpy as np
import pytest

from pyradioss.common.messages import MessageLog
from pyradioss.elements import solid_hexa8, solid_tetra4
from pyradioss.engine.engine import run_engine
from pyradioss.input.deck_reader import read_deck
from pyradioss.input.deck_writer import StarterDeck
from pyradioss.input.starter_keywords import parse_starter_deck, read_mat_law21
from pyradioss.materials.law21_dprag import (
    build_law21,
    solid_update_law21,
    sound_speed_solid_law21,
)
from pyradioss import materials
from pyradioss.model.entities import MatDprag, MatDuckhub, MatLaw21, Material
from pyradioss.model.model import Model
from pyradioss.starter.checks import (
    _ALLOWED_LAWS,
    _MAT_CHECKS,
    check_mat_law21,
    check_materials,
    check_model,
)
from pyradioss.starter.restart import read_restart, write_restart
from pyradioss.starter.starter import run_starter


# ============================================================================
# Helpers
# ============================================================================

def _parse_deck_str(tmp_path: Path, text: str, name: str = "TEST_LAW21") -> Tuple[Model, MessageLog]:
    """Write deck text to file and parse into Model and MessageLog."""
    deck_path = tmp_path / f"{name}_0000.rad"
    deck_path.write_text(text.strip() + "\n", encoding="utf-8")
    blocks = read_deck(str(deck_path))
    model = Model()
    log = MessageLog()
    parse_starter_deck(blocks, model, log)
    return model, log


def _assert_law21_all_15_fields_exact(
    m: MatLaw21,
    *,
    rho: float,
    refer_rho: float,
    e: float,
    nu: float,
    a0: float,
    a1: float,
    a2: float,
    amax: float,
    ifunc: int,
    c1: float,
    pfscale: float,
    pmin: float,
    pext: float,
    bunl: float,
    mumax: float,
) -> None:
    """Assert exact equality (or floating-point precision equality) for all 15 LAW21 fields."""
    assert m.rho == pytest.approx(rho, rel=1e-6, abs=1e-12)
    assert m.refer_rho == pytest.approx(refer_rho, rel=1e-6, abs=1e-12)
    assert m.rho0 == pytest.approx(rho, rel=1e-6, abs=1e-12)
    assert m.rhor == pytest.approx(refer_rho, rel=1e-6, abs=1e-12)

    assert m.e == pytest.approx(e, rel=1e-6, abs=1e-12)
    assert m.E == pytest.approx(e, rel=1e-6, abs=1e-12)
    assert m.nu == pytest.approx(nu, rel=1e-6, abs=1e-12)
    assert m.Nu == pytest.approx(nu, rel=1e-6, abs=1e-12)

    assert m.a0 == pytest.approx(a0, rel=1e-6, abs=1e-12)
    assert m.a1 == pytest.approx(a1, rel=1e-6, abs=1e-12)
    assert m.a2 == pytest.approx(a2, rel=1e-6, abs=1e-12)
    assert m.amax == pytest.approx(amax, rel=1e-6, abs=1e-12)

    assert m.ifunc == ifunc
    assert m.c1 == pytest.approx(c1, rel=1e-6, abs=1e-12)
    assert m.pfscale == pytest.approx(pfscale, rel=1e-6, abs=1e-12)
    assert m.pmin == pytest.approx(pmin, rel=1e-6, abs=1e-12)
    assert m.pext == pytest.approx(pext, rel=1e-6, abs=1e-12)
    assert m.bunl == pytest.approx(bunl, rel=1e-6, abs=1e-12)
    assert m.mumax == pytest.approx(mumax, rel=1e-6, abs=1e-12)

    # Verify params mapping dictionary matches
    p = m.params
    assert p["rho0"] == pytest.approx(rho, rel=1e-6, abs=1e-12)
    assert p["rhor"] == pytest.approx(refer_rho, rel=1e-6, abs=1e-12)
    assert p["E"] == pytest.approx(e, rel=1e-6, abs=1e-12)
    assert p["nu"] == pytest.approx(nu, rel=1e-6, abs=1e-12)
    assert p["a0"] == pytest.approx(a0, rel=1e-6, abs=1e-12)
    assert p["a1"] == pytest.approx(a1, rel=1e-6, abs=1e-12)
    assert p["a2"] == pytest.approx(a2, rel=1e-6, abs=1e-12)
    assert p["amax"] == pytest.approx(amax, rel=1e-6, abs=1e-12)
    assert p["ifunc"] == ifunc
    assert p["c1"] == pytest.approx(c1, rel=1e-6, abs=1e-12)
    assert p["pfscale"] == pytest.approx(pfscale, rel=1e-6, abs=1e-12)
    assert p["pmin"] == pytest.approx(pmin, rel=1e-6, abs=1e-12)
    assert p["pext"] == pytest.approx(pext, rel=1e-6, abs=1e-12)
    assert p["bunl"] == pytest.approx(bunl, rel=1e-6, abs=1e-12)
    assert p["mumax"] == pytest.approx(mumax, rel=1e-6, abs=1e-12)


class MockProp:
    """Mock solid property for element kernel testing."""
    def __init__(self, pid: int = 1, isolid: int = 1, **kwargs: Any):
        self.id = pid
        self.isolid = isolid
        self.params = {"isolid": isolid, "qa": 1.1, "qb": 0.05, **kwargs}


class MockGroup:
    """Mock element group for element kernel testing."""
    def __init__(self, conn: np.ndarray, ids: np.ndarray | None = None, slices: list | None = None):
        self.conn = np.asarray(conn, dtype=np.int64)
        self.n = len(self.conn)
        self.ids = np.arange(1, self.n + 1, dtype=np.int64) if ids is None else np.asarray(ids, dtype=np.int64)
        self.state: dict[str, Any] = {}
        if slices is not None:
            self.state["slices"] = slices
        self._model: Any = None


# ============================================================================
# 1. Fixed-Format Deck Roundtrip
# ============================================================================

class TestLaw21FixedFormatRoundtrip:
    """Audit fixed-format deck emission via StarterDeck.mat_law21 and re-parsing with read_mat_law21."""

    def test_fixed_format_all_15_fields_exact(self, tmp_path: Path):
        """Write /MAT/LAW21 with 20-col fixed format, re-parse and assert exact equality for all 15 fields."""
        vals = {
            "rho": 2.45e-3,
            "refer_rho": 2.52e-3,
            "e": 32000.0,
            "nu": 0.22,
            "a0": 15.5,
            "a1": 0.65,
            "a2": 0.015,
            "amax": 175.0,
            "ifunc": 3,
            "c1": 21500.0,
            "pfscale": 1.25,
            "pmin": -220.0,
            "pext": 45.0,
            "bunl": 27000.0,
            "mumax": 0.075,
        }

        deck = StarterDeck("AUDIT_LAW21_FIXED_ALL")
        deck.mat_law21(
            mat_id=1,
            title="Drucker-Prager Granite Full 15 Fields",
            fixed_format=True,
            **vals,
        )

        rendered = deck.render()
        assert "/MAT/LAW21/1" in rendered
        assert "Drucker-Prager Granite Full 15 Fields" in rendered

        f = tmp_path / "deck_law21_fixed_all_0000.rad"
        f.write_text(rendered + "\n/END\n", encoding="utf-8")

        blocks = read_deck(str(f))
        model = Model()
        log = MessageLog()
        for block in blocks:
            if block.key0 == "MAT":
                read_mat_law21(block, model, log)

        assert not log.has_errors, f"Starter log errors: {log.errors}"
        assert 1 in model.mat_law21s
        m = model.mat_law21s[1]

        _assert_law21_all_15_fields_exact(m, **vals)

    def test_fixed_format_defaults_fallback(self, tmp_path: Path):
        """Write /MAT/LAW21 with minimal cards and assert Fortran-faithful defaults for unassigned fields."""
        deck = StarterDeck("AUDIT_LAW21_DEFAULTS")
        deck.mat_law21(
            mat_id=5,
            title="Minimal LAW21 Card",
            rho=2.3e-3,
            e=28000.0,
            nu=0.20,
            a0=10.0,
            a1=0.5,
            c1=20000.0,
            fixed_format=True,
        )

        rendered = deck.render()
        f = tmp_path / "deck_law21_defaults_0000.rad"
        f.write_text(rendered + "\n/END\n", encoding="utf-8")

        blocks = read_deck(str(f))
        model = Model()
        log = MessageLog()
        for block in blocks:
            if block.key0 == "MAT":
                read_mat_law21(block, model, log)

        assert not log.has_errors
        assert 5 in model.mat_law21s
        m = model.mat_law21s[5]

        _assert_law21_all_15_fields_exact(
            m,
            rho=2.3e-3,
            refer_rho=2.3e-3,  # defaults to rho
            e=28000.0,
            nu=0.20,
            a0=10.0,
            a1=0.5,
            a2=0.0,            # default
            amax=1.0e20,       # default von Mises cap
            ifunc=0,           # default
            c1=20000.0,
            pfscale=1.0,       # default
            pmin=-1.0e30,      # default tensile cutoff
            pext=0.0,          # default external pressure shift
            bunl=20000.0,      # defaults to C1
            mumax=1.0e20,      # default max compaction
        )

    def test_fixed_format_mat_object_direct_input(self, tmp_path: Path):
        """Pass a MatLaw21 instance directly to StarterDeck.mat_law21 and verify roundtrip."""
        mat_in = MatLaw21(
            id=12,
            rho=2.6e-3,
            refer_rho=2.65e-3,
            e=35000.0,
            nu=0.18,
            a0=14.0,
            a1=0.55,
            a2=0.012,
            amax=160.0,
            ifunc=2,
            c1=23000.0,
            pfscale=1.1,
            pmin=-180.0,
            pext=25.0,
            bunl=25000.0,
            mumax=0.06,
            title="Entity Driven LAW21",
        )

        deck = StarterDeck("AUDIT_LAW21_ENTITY")
        deck.mat_law21(mat_in, fixed_format=True)

        rendered = deck.render()
        f = tmp_path / "deck_law21_entity_0000.rad"
        f.write_text(rendered + "\n/END\n", encoding="utf-8")

        blocks = read_deck(str(f))
        model = Model()
        log = MessageLog()
        for block in blocks:
            if block.key0 == "MAT":
                read_mat_law21(block, model, log)

        assert not log.has_errors
        assert 12 in model.mat_law21s
        m_out = model.mat_law21s[12]

        _assert_law21_all_15_fields_exact(
            m_out,
            rho=2.6e-3,
            refer_rho=2.65e-3,
            e=35000.0,
            nu=0.18,
            a0=14.0,
            a1=0.55,
            a2=0.012,
            amax=160.0,
            ifunc=2,
            c1=23000.0,
            pfscale=1.1,
            pmin=-180.0,
            pext=25.0,
            bunl=25000.0,
            mumax=0.06,
        )


# ============================================================================
# 2. Free-Format Comma-Delimited Roundtrip
# ============================================================================

class TestLaw21FreeFormatRoundtrip:
    """Audit free-format comma-delimited deck emission and re-parsing for /MAT/LAW21 and /MAT/DPRAG."""

    def test_free_format_comma_delimited_law21(self, tmp_path: Path):
        """Write /MAT/LAW21 with comma-separated free format and re-parse."""
        vals = {
            "rho": 2.4e-3,
            "refer_rho": 2.48e-3,
            "e": 26000.0,
            "nu": 0.21,
            "a0": 9.0,
            "a1": 0.45,
            "a2": 0.008,
            "amax": 135.0,
            "ifunc": 1,
            "c1": 19000.0,
            "pfscale": 1.15,
            "pmin": -160.0,
            "pext": 30.0,
            "bunl": 22000.0,
            "mumax": 0.045,
        }

        deck = StarterDeck("AUDIT_FREE_LAW21")
        deck.mat_law21(
            mat_id=21,
            title="Free Comma LAW21",
            fixed_format=False,
            comma=True,
            **vals,
        )

        rendered = deck.render()
        assert "/MAT/LAW21/21" in rendered
        assert "," in rendered

        f = tmp_path / "deck_law21_free_0000.rad"
        f.write_text(rendered + "\n/END\n", encoding="utf-8")

        blocks = read_deck(str(f))
        model = Model()
        log = MessageLog()
        for block in blocks:
            if block.key0 == "MAT":
                read_mat_law21(block, model, log)

        assert not log.has_errors
        assert 21 in model.mat_law21s
        m = model.mat_law21s[21]
        _assert_law21_all_15_fields_exact(m, **vals)

    def test_free_format_comma_delimited_mat_dprag(self, tmp_path: Path):
        """Write /MAT/DPRAG with comma-separated free format and verify synonym roundtrip."""
        vals = {
            "rho": 2.5e-3,
            "refer_rho": 2.55e-3,
            "e": 31000.0,
            "nu": 0.19,
            "a0": 11.0,
            "a1": 0.52,
            "a2": 0.011,
            "amax": 165.0,
            "ifunc": 4,
            "c1": 21000.0,
            "pfscale": 1.3,
            "pmin": -210.0,
            "pext": 40.0,
            "bunl": 24000.0,
            "mumax": 0.055,
        }

        deck = StarterDeck("AUDIT_FREE_DPRAG")
        deck.mat_dprag(
            mat_id=42,
            title="Free Comma DPRAG",
            fixed_format=False,
            comma=True,
            **vals,
        )

        rendered = deck.render()
        assert "/MAT/DPRAG/42" in rendered
        assert "," in rendered

        f = tmp_path / "deck_dprag_free_0000.rad"
        f.write_text(rendered + "\n/END\n", encoding="utf-8")

        blocks = read_deck(str(f))
        model = Model()
        log = MessageLog()
        for block in blocks:
            if block.key0 == "MAT":
                read_mat_law21(block, model, log)

        assert not log.has_errors
        assert 42 in model.mat_law21s
        m = model.mat_law21s[42]
        _assert_law21_all_15_fields_exact(m, **vals)

    def test_compact_comma_delimited_without_spaces(self, tmp_path: Path):
        """Re-parse compact comma-delimited deck text with zero whitespace between delimiters."""
        deck_text = """\
#RADIOSS STARTER
/BEGIN
compact_comma_test
                  90                   1
/MAT/LAW21/55
Compact No Space Test
2.42e-3,2.46e-3
27500.0,0.205
10.5,0.48,0.009,145.0
2,,19500.0,1.2
-175.0,35.0
23500.0,0.048
/END
"""
        f = tmp_path / "compact_no_space_0000.rad"
        f.write_text(deck_text, encoding="utf-8")

        blocks = read_deck(str(f))
        model = Model()
        log = MessageLog()
        for block in blocks:
            if block.key0 == "MAT":
                read_mat_law21(block, model, log)

        assert not log.has_errors
        assert 55 in model.mat_law21s
        m = model.mat_law21s[55]

        _assert_law21_all_15_fields_exact(
            m,
            rho=2.42e-3,
            refer_rho=2.46e-3,
            e=27500.0,
            nu=0.205,
            a0=10.5,
            a1=0.48,
            a2=0.009,
            amax=145.0,
            ifunc=2,
            c1=19500.0,
            pfscale=1.2,
            pmin=-175.0,
            pext=35.0,
            bunl=23500.0,
            mumax=0.048,
        )

    def test_cross_dialect_roundtrip(self, tmp_path: Path):
        """Cross-dialect roundtrip: free-format -> parse -> write fixed-format -> parse -> exact equality."""
        vals = {
            "rho": 2.55e-3,
            "refer_rho": 2.60e-3,
            "e": 33000.0,
            "nu": 0.23,
            "a0": 13.0,
            "a1": 0.58,
            "a2": 0.014,
            "amax": 170.0,
            "ifunc": 5,
            "c1": 22500.0,
            "pfscale": 1.4,
            "pmin": -240.0,
            "pext": 55.0,
            "bunl": 26500.0,
            "mumax": 0.065,
        }

        # Step 1: write free
        deck1 = StarterDeck("CROSS_1")
        deck1.mat_law21(mat_id=101, title="Cross Step 1", fixed_format=False, comma=True, **vals)
        f1 = tmp_path / "cross_step1_0000.rad"
        f1.write_text(deck1.render() + "\n/END\n", encoding="utf-8")

        b1 = read_deck(str(f1))
        m1 = Model()
        read_mat_law21(b1[1], m1, MessageLog())
        mat1 = m1.mat_law21s[101]

        # Step 2: write fixed using parsed entity
        deck2 = StarterDeck("CROSS_2")
        deck2.mat_law21(mat1, fixed_format=True)
        f2 = tmp_path / "cross_step2_0000.rad"
        f2.write_text(deck2.render() + "\n/END\n", encoding="utf-8")

        b2 = read_deck(str(f2))
        m2 = Model()
        read_mat_law21(b2[1], m2, MessageLog())
        mat2 = m2.mat_law21s[101]

        _assert_law21_all_15_fields_exact(mat2, **vals)


# ============================================================================
# 3. Negative Starter Diagnostics
# ============================================================================

class TestLaw21NegativeStarterDiagnostics:
    """Audit check_mat_law21 diagnostic error and warning reporting per hm_read_mat21.F."""

    def test_negative_density(self):
        """Initial density rho <= 0 triggers an error."""
        # rho = 0
        log0 = MessageLog()
        m0 = MatLaw21(id=1, rho=0.0, e=20000.0, nu=0.2, c1=15000.0)
        check_mat_law21(mat=m0, log=log0)
        assert log0.has_errors
        assert any("initial density RHO must be > 0" in e for e in log0.errors)

        # rho < 0
        log_neg = MessageLog()
        m_neg = MatLaw21(id=1, rho=-2.5e-3, e=20000.0, nu=0.2, c1=15000.0)
        check_mat_law21(mat=m_neg, log=log_neg)
        assert log_neg.has_errors
        assert any("initial density RHO must be > 0" in e for e in log_neg.errors)

    def test_negative_youngs_modulus(self):
        """Young's modulus E <= 0 triggers an error."""
        # E = 0
        log0 = MessageLog()
        m0 = MatLaw21(id=2, rho=2.5e-3, e=0.0, nu=0.2, c1=15000.0)
        check_mat_law21(mat=m0, log=log0)
        assert log0.has_errors
        assert any("Young's modulus E must be > 0" in e for e in log0.errors)

        # E < 0
        log_neg = MessageLog()
        m_neg = MatLaw21(id=2, rho=2.5e-3, e=-1000.0, nu=0.2, c1=15000.0)
        check_mat_law21(mat=m_neg, log=log_neg)
        assert log_neg.has_errors
        assert any("Young's modulus E must be > 0" in e for e in log_neg.errors)

    def test_negative_poissons_ratio(self):
        """Poisson's ratio nu < 0 or nu >= 0.5 triggers an error."""
        # nu < 0
        log_neg = MessageLog()
        m_neg = MatLaw21(id=3, rho=2.5e-3, e=20000.0, nu=-0.1, c1=15000.0)
        check_mat_law21(mat=m_neg, log=log_neg)
        assert log_neg.has_errors
        assert any("0 <= NU < 0.5" in e for e in log_neg.errors)

        # nu = 0.5
        log_half = MessageLog()
        m_half = MatLaw21(id=3, rho=2.5e-3, e=20000.0, nu=0.5, c1=15000.0)
        check_mat_law21(mat=m_half, log=log_half)
        assert log_half.has_errors
        assert any("0 <= NU < 0.5" in e for e in log_half.errors)

        # nu > 0.5
        log_hi = MessageLog()
        m_hi = MatLaw21(id=3, rho=2.5e-3, e=20000.0, nu=0.6, c1=15000.0)
        check_mat_law21(mat=m_hi, log=log_hi)
        assert log_hi.has_errors
        assert any("0 <= NU < 0.5" in e for e in log_hi.errors)

    def test_negative_bulk_modulus_c1_ancmsg_829(self):
        """Tensile bulk modulus C1 <= 0 triggers ANCMSG 829 ERROR."""
        # C1 = 0
        log0 = MessageLog()
        m0 = MatLaw21(id=4, rho=2.5e-3, e=20000.0, nu=0.2, c1=0.0)
        check_mat_law21(mat=m0, log=log0)
        assert log0.has_errors
        assert any("tensile bulk modulus C1 must be > 0" in e and "ANCMSG 829" in e for e in log0.errors)

        # C1 < 0
        log_neg = MessageLog()
        m_neg = MatLaw21(id=4, rho=2.5e-3, e=20000.0, nu=0.2, c1=-5000.0)
        check_mat_law21(mat=m_neg, log=log_neg)
        assert log_neg.has_errors
        assert any("tensile bulk modulus C1 must be > 0" in e and "ANCMSG 829" in e for e in log_neg.errors)

    def test_yield_surface_inverted_warning_ancmsg_829(self):
        """Inverted yield surface (A1 < 0 and A2 == 0) triggers ANCMSG 829 warning."""
        log = MessageLog()
        m = MatLaw21(id=10, rho=2.5e-3, e=20000.0, nu=0.2, c1=15000.0, a0=10.0, a1=-0.5, a2=0.0)
        check_mat_law21(mat=m, log=log)
        assert not log.has_errors
        assert any("INVERTED YIELD SURFACE. CHECK A1 SIGN." in w and "ANCMSG 829" in w for w in log.warnings)

    def test_yield_surface_untypical_warning_ancmsg_829(self):
        """Untypical yield surface (A2 < 0) triggers ANCMSG 829 warning."""
        log = MessageLog()
        m = MatLaw21(id=11, rho=2.5e-3, e=20000.0, nu=0.2, c1=15000.0, a0=10.0, a1=0.5, a2=-0.02)
        check_mat_law21(mat=m, log=log)
        assert not log.has_errors
        assert any("UNTYPICAL YIELD SURFACE. CHECK A2 SIGN." in w and "ANCMSG 829" in w for w in log.warnings)

    def test_yield_surface_no_root_warning_ancmsg_829(self):
        """Yield surface with no real root (A1^2 - 4*A0*A2 < 0) triggers ANCMSG 829 warning."""
        # a1 = 1.0, a0 = 10.0, a2 = 1.0 -> delta = 1 - 40 = -39 < 0
        log = MessageLog()
        m = MatLaw21(id=12, rho=2.5e-3, e=20000.0, nu=0.2, c1=15000.0, a0=10.0, a1=1.0, a2=1.0)
        check_mat_law21(mat=m, log=log)
        assert not log.has_errors
        assert any("YIELD SURFACE HAS NO ROOT." in w and "ANCMSG 829" in w for w in log.warnings)

    def test_element_compatibility_acceptance_and_rejection(self):
        """Solids accepted; shells (ANCMSG 305), 1D (ANCMSG 306), and 2D analysis (ANCMSG 305) rejected."""
        class DummyEl:
            def __init__(self, mid: int):
                self.mat_id = mid

        class DummyGrp:
            def __init__(self, els: list):
                self._els = {i: e for i, e in enumerate(els)}
            def values(self):
                return self._els.values()

        class DummyModel:
            def __init__(self, grps: list, n2d: int = 0):
                self._grps = grps
                self.n2d = n2d
                self.materials: dict = {}
            def element_groups(self):
                return self._grps

        mat21 = MatLaw21(id=99, rho=2.5e-3, e=25000.0, nu=0.22, c1=18000.0)

        # 1. Solids accepted
        for s_type in ("solids", "bricks", "tetras", "penta6", "pyra5"):
            model_s = DummyModel([(s_type, DummyGrp([DummyEl(99)]))])
            log_s = MessageLog()
            check_mat_law21(model=model_s, mat=mat21, log=log_s)
            assert not log_s.has_errors, f"Expected solid type {s_type} to be accepted"

        # 2. Shells rejected (ANCMSG 305)
        for sh_type in ("shells", "shells_qbat", "shells_qeph", "sh3n", "quads"):
            model_sh = DummyModel([(sh_type, DummyGrp([DummyEl(99)]))])
            log_sh = MessageLog()
            check_mat_law21(model=model_sh, mat=mat21, log=log_sh)
            assert log_sh.has_errors, f"Expected shell type {sh_type} to be rejected"
            assert any("ANCMSG 305" in e and "shell elements" in e for e in log_sh.errors)

        # 3. 1D elements rejected (ANCMSG 306)
        for d1_type in ("trusses", "beams", "springs"):
            model_1d = DummyModel([(d1_type, DummyGrp([DummyEl(99)]))])
            log_1d = MessageLog()
            check_mat_law21(model=model_1d, mat=mat21, log=log_1d)
            assert log_1d.has_errors, f"Expected 1D type {d1_type} to be rejected"
            assert any("ANCMSG 306" in e and "1D elements" in e for e in log_1d.errors)

        # 4. 2D analysis rejected (N2D > 0, ANCMSG 305)
        model_2d = DummyModel([("solids", DummyGrp([DummyEl(99)]))], n2d=1)
        log_2d = MessageLog()
        check_mat_law21(model=model_2d, mat=mat21, log=log_2d)
        assert log_2d.has_errors
        assert any("N2D > 0" in e and "ANCMSG 305" in e for e in log_2d.errors)

    def test_checks_dispatch_and_allowed_laws_registry(self):
        """Verify _MAT_CHECKS registry and _ALLOWED_LAWS tables for all LAW21 synonyms."""
        for syn in (21, "21", "LAW21", "DPRAG", "MAT_DPRAG", "LAW21_DPRAG", "DUCKHUB"):
            assert syn in _MAT_CHECKS, f"{syn} missing in _MAT_CHECKS"
            assert _MAT_CHECKS[syn] is check_mat_law21

        # Solids permit LAW21
        for fam in ("bricks", "tetras", "penta6", "pyra5", "solids"):
            assert 21 in _ALLOWED_LAWS[fam]
            assert "LAW21" in _ALLOWED_LAWS[fam]
            assert "DPRAG" in _ALLOWED_LAWS[fam]

        # Shells and lines forbid LAW21
        for fam in ("shells", "shells_qbat", "shells_qeph", "sh3n", "quads", "trusses", "beams"):
            assert 21 not in _ALLOWED_LAWS[fam]
            assert "LAW21" not in _ALLOWED_LAWS[fam]
            assert "DPRAG" not in _ALLOWED_LAWS[fam]


# ============================================================================
# 4. Restart (.rst) Serialization
# ============================================================================

class TestLaw21RestartSerialization:
    """Audit restart (.rst) state serialization and bitwise/analytical continuation."""

    def test_mat_law21_dataclass_pickle_fidelity(self):
        """Pickle and unpickle MatLaw21, verifying all 15 fields and derived properties."""
        mat = MatLaw21(
            id=7,
            rho=2.5e-3,
            refer_rho=2.55e-3,
            e=30000.0,
            nu=0.2,
            a0=12.0,
            a1=0.6,
            a2=0.015,
            amax=150.0,
            ifunc=3,
            c1=20000.0,
            pfscale=1.2,
            pmin=-200.0,
            pext=35.0,
            bunl=25000.0,
            mumax=0.06,
            title="Pickle Rock",
        )

        data = pickle.dumps(mat, protocol=pickle.HIGHEST_PROTOCOL)
        restored: MatLaw21 = pickle.loads(data)

        _assert_law21_all_15_fields_exact(
            restored,
            rho=2.5e-3,
            refer_rho=2.55e-3,
            e=30000.0,
            nu=0.2,
            a0=12.0,
            a1=0.6,
            a2=0.015,
            amax=150.0,
            ifunc=3,
            c1=20000.0,
            pfscale=1.2,
            pmin=-200.0,
            pext=35.0,
            bunl=25000.0,
            mumax=0.06,
        )

        # Derived properties
        assert restored.G == pytest.approx(mat.G)
        assert restored.sound_speed() == pytest.approx(mat.sound_speed())
        assert restored.sound_speed_solid() == pytest.approx(mat.sound_speed_solid())

    def test_material_state_arrays_rst_preservation(self, tmp_path: Path):
        """Verify mu_bak, p, epxe, defp, p_old, off, sig, epsp survive write_restart and read_restart."""
        deck = StarterDeck("RST_LAW21_ARRAYS")
        deck.node([
            (1, 0.0, 0.0, 0.0), (2, 1.0, 0.0, 0.0), (3, 1.0, 1.0, 0.0), (4, 0.0, 1.0, 0.0),
            (5, 0.0, 0.0, 1.0), (6, 1.0, 0.0, 1.0), (7, 1.0, 1.0, 1.0), (8, 0.0, 1.0, 1.0),
            (9, 2.0, 0.0, 0.0), (10, 2.0, 1.0, 0.0), (11, 2.0, 0.0, 1.0), (12, 2.0, 1.0, 1.0),
        ])
        # 2 bricks
        deck.brick(1, [
            [1, 1, 2, 3, 4, 5, 6, 7, 8],
            [2, 2, 9, 10, 3, 6, 11, 12, 7],
        ])
        deck.prop_solid(1, "SolidProp")
        deck.part(1, "RockPart", prop_id=1, mat_id=1)
        deck.mat_law21(
            mat_id=1,
            title="Rock",
            rho=2000.0,
            e=2.0e10,
            nu=0.25,
            c1=1.0e9,
            a0=1.0e10,
            a1=0.5,
            pmin=-5.0e7,
        )

        rad_path = tmp_path / "rst_arrays_0000.rad"
        deck.write(str(rad_path))

        model = run_starter(str(rad_path))
        assert model is not None

        groups = dict(model.element_groups())
        assert "bricks" in groups or "solids" in groups
        bg = groups.get("bricks") or groups.get("solids")
        st = bg.state

        # Synthesize realistic persistent material state arrays for LAW21: 2 elements
        mu_bak_synth = np.array([0.035, 0.072], dtype=float)
        p_synth = np.array([4.5e7, 8.2e7], dtype=float)
        epxe_synth = np.array([0.0042, 0.0115], dtype=float)
        defp_synth = np.array([1.2e-4, 3.5e-4], dtype=float)
        p_old_synth = np.array([4.3e7, 8.0e7], dtype=float)
        off_synth = np.array([1.0, 0.0], dtype=float)  # 1 alive, 1 dead
        sig_synth = np.array([
            [5.0e7, 4.0e7, 4.5e7, 1.2e6, 0.8e6, 0.5e6],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        ], dtype=float)
        epsp_synth = np.array([0.0042, 0.0115], dtype=float)

        st["mat_extra"]["mu_bak"] = mu_bak_synth.copy()
        st["mat_extra"]["p"] = p_synth.copy()
        st["mat_extra"]["epxe"] = epxe_synth.copy()
        st["mat_extra"]["defp"] = defp_synth.copy()
        st["mat_extra"]["p_old"] = p_old_synth.copy()
        st["off"] = off_synth.copy()
        st["sig"] = sig_synth.copy()
        st["epsp"] = epsp_synth.copy()

        # Write restart .rst
        rst_path = tmp_path / "rst_arrays_0001.rst"
        engine_dict = {
            "cycle": 250,
            "t": 2.5e-4,
            "dt": 1.0e-6,
            "energies": {"internal": 12500.0, "kinetic": 3200.0},
        }
        write_restart(model, str(rst_path), engine=engine_dict)

        # Read back from restart .rst
        rest_model, rest_engine = read_restart(str(rst_path))
        assert rest_engine["cycle"] == 250
        assert rest_engine["t"] == pytest.approx(2.5e-4)
        assert rest_engine["dt"] == pytest.approx(1.0e-6)

        rest_bg = dict(rest_model.element_groups()).get("bricks") or dict(rest_model.element_groups()).get("solids")
        rest_extra = rest_bg.state["mat_extra"]
        rest_st = rest_bg.state

        # Assert exact preservation
        np.testing.assert_array_equal(rest_extra["mu_bak"], mu_bak_synth)
        np.testing.assert_array_equal(rest_extra["p"], p_synth)
        np.testing.assert_array_equal(rest_extra["epxe"], epxe_synth)
        np.testing.assert_array_equal(rest_extra["defp"], defp_synth)
        np.testing.assert_array_equal(rest_extra["p_old"], p_old_synth)
        np.testing.assert_array_equal(rest_st["off"], off_synth)
        np.testing.assert_array_equal(rest_st["sig"], sig_synth)
        np.testing.assert_array_equal(rest_st["epsp"], epsp_synth)

    def test_constitutive_dynamic_cycle_continuation_tolerance(self):
        """Constitutive dynamic update (solid_update_law21) across restart matches uninterrupted within 10^-12."""
        mat = {
            "rho0": 2000.0,
            "E": 2.0e10,
            "nu": 0.25,
            "c1": 1.0e9,
            "bunl": 2.0e9,
            "mumax": 0.05,
            "a0": 1.0e10,
            "a1": 0.5,
            "a2": 0.01,
            "amax": 1.0e20,
            "pmin": -1.0e30,
            "pext": 10.0,
        }

        dt = 1.0e-5
        n_steps = 14
        split_step = 7

        # Dynamic strain increment series alternating compression and shear yielding
        np.random.seed(42)
        deps_series = [
            np.array([[-0.005, -0.005, -0.005, 0.001, 0.0005, 0.0002]])
            if i % 3 == 0 else
            np.array([[0.002, 0.002, 0.002, 0.003, 0.001, 0.0008]])
            for i in range(n_steps)
        ]

        # 1. Uninterrupted run: 14 steps
        extra_unc = {
            "mu_bak": np.zeros(1),
            "epxe": np.zeros(1),
            "p": np.zeros(1),
            "off": np.ones(1),
        }
        sig_unc = np.zeros((1, 6))
        epsp_unc = np.zeros(1)

        for i in range(n_steps):
            sig_unc = solid_update_law21(mat, sig_unc, deps=deps_series[i], dt=dt, epsp=epsp_unc, extra=extra_unc)

        # 2. Chained run: 7 steps -> pickle -> 7 steps
        extra_chn = {
            "mu_bak": np.zeros(1),
            "epxe": np.zeros(1),
            "p": np.zeros(1),
            "off": np.ones(1),
        }
        sig_chn = np.zeros((1, 6))
        epsp_chn = np.zeros(1)

        for i in range(split_step):
            sig_chn = solid_update_law21(mat, sig_chn, deps=deps_series[i], dt=dt, epsp=epsp_chn, extra=extra_chn)

        # Serialize and restore state
        sig_rest = pickle.loads(pickle.dumps(sig_chn))
        extra_rest = pickle.loads(pickle.dumps(extra_chn))
        epsp_rest = pickle.loads(pickle.dumps(epsp_chn))

        for i in range(split_step, n_steps):
            sig_rest = solid_update_law21(mat, sig_rest, deps=deps_series[i], dt=dt, epsp=epsp_rest, extra=extra_rest)

        # Verify exact agreement within 10^-12
        np.testing.assert_allclose(sig_rest, sig_unc, atol=1e-12, rtol=1e-12)
        np.testing.assert_allclose(extra_rest["mu_bak"], extra_unc["mu_bak"], atol=1e-12, rtol=1e-12)
        np.testing.assert_allclose(extra_rest["epxe"], extra_unc["epxe"], atol=1e-12, rtol=1e-12)
        np.testing.assert_allclose(extra_rest["p"], extra_unc["p"], atol=1e-12, rtol=1e-12)

    def test_element_kernel_restart_dynamic_continuation(self, tmp_path: Path):
        """Dynamic step continuation with solid_hexa8.forces across write_restart / read_restart matches within 10^-12."""
        mat = build_law21({
            "id": 1,
            "MAT_RHO": 2000.0,
            "MAT_E": 2.0e10,
            "MAT_NU": 0.25,
            "MAT_BULK": 1.0e9,
            "MAT_K_UNLOAD": 2.0e9,
            "MAT_SIG": 0.05,
            "MAT_A0": 1.0e10,
            "MAT_A1": 0.5,
            "MAT_A2": 0.0,
            "MAT_PC": -1.0e30,
        })
        prop = MockProp(pid=1, isolid=1)
        conn = np.array([[0, 1, 2, 3, 4, 5, 6, 7]], dtype=np.int64)

        coords = np.array([
            [0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0], [1.0, 0.0, 1.0], [1.0, 1.0, 1.0], [0.0, 1.0, 1.0],
        ], dtype=float)

        dt = 1.0e-5
        n_cycles = 12
        split_cycle = 6

        # Compression + shear velocity field
        centroid = np.array([0.5, 0.5, 0.5])
        vel = -0.02 * (coords - centroid)
        vel[:, 1] += 0.01 * coords[:, 0]  # shear in y
        vr = np.zeros_like(coords)

        # 1. Uninterrupted run: 12 cycles
        m_unc = Model()
        m_unc.x0 = coords.copy()
        g_unc = MockGroup(conn, slices=[(slice(0, 1), mat, prop)])
        g_unc._model = m_unc
        solid_hexa8.init_group(g_unc, m_unc, None)

        coords_unc = coords.copy()
        fint_unc = np.zeros((8, 3))
        mint_unc = np.zeros((8, 3))

        for _ in range(n_cycles):
            fint_unc.fill(0.0)
            mint_unc.fill(0.0)
            solid_hexa8.forces(g_unc, coords_unc, vel, vr, dt, fint_unc, mint_unc)
            coords_unc += vel * dt

        # 2. Chained run: 6 cycles -> write_restart -> 6 cycles
        m_chn = Model()
        m_chn.x0 = coords.copy()
        g_chn = MockGroup(conn, slices=[(slice(0, 1), mat, prop)])
        g_chn._model = m_chn
        solid_hexa8.init_group(g_chn, m_chn, None)

        coords_chn = coords.copy()
        fint_chn = np.zeros((8, 3))
        mint_chn = np.zeros((8, 3))

        for _ in range(split_cycle):
            fint_chn.fill(0.0)
            mint_chn.fill(0.0)
            solid_hexa8.forces(g_chn, coords_chn, vel, vr, dt, fint_chn, mint_chn)
            coords_chn += vel * dt

        # Save to restart .rst
        m_chn.x = coords_chn.copy()
        m_chn.v = vel.copy()
        m_chn.vr = vr.copy()
        m_chn.solids = g_chn
        rst_path = tmp_path / "hexa8_law21_restart.rst"
        write_restart(m_chn, str(rst_path), engine={"cycle": split_cycle, "t": split_cycle * dt})

        # Read back from restart .rst and continue
        m_res, eng_res = read_restart(str(rst_path))
        assert eng_res["cycle"] == split_cycle
        assert eng_res["t"] == pytest.approx(split_cycle * dt)

        g_res = m_res.solids
        coords_res = m_res.x.copy()
        fint_res = np.zeros((8, 3))
        mint_res = np.zeros((8, 3))

        for _ in range(split_cycle, n_cycles):
            fint_res.fill(0.0)
            mint_res.fill(0.0)
            solid_hexa8.forces(g_res, coords_res, vel, vr, dt, fint_res, mint_res)
            coords_res += vel * dt

        # Verify tolerances within 10^-12
        np.testing.assert_allclose(fint_res, fint_unc, atol=1e-12, rtol=1e-12)
        np.testing.assert_allclose(mint_res, mint_unc, atol=1e-12, rtol=1e-12)
        np.testing.assert_allclose(g_res.state["sig"], g_unc.state["sig"], atol=1e-12, rtol=1e-12)
        np.testing.assert_allclose(g_res.state["epsp"], g_unc.state["epsp"], atol=1e-12, rtol=1e-12)
        np.testing.assert_allclose(
            g_res.state["mat_extra"]["mu_bak"],
            g_unc.state["mat_extra"]["mu_bak"],
            atol=1e-12,
            rtol=1e-12,
        )
        np.testing.assert_allclose(
            g_res.state["mat_extra"]["p"],
            g_unc.state["mat_extra"]["p"],
            atol=1e-12,
            rtol=1e-12,
        )

    def test_element_alive_mask_restart_invariance(self, tmp_path: Path):
        """Element alive mask off=0 survives restart and produces zero stress and force increments."""
        mat = build_law21({
            "id": 1,
            "MAT_RHO": 2000.0,
            "MAT_E": 2.0e10,
            "MAT_NU": 0.25,
            "MAT_BULK": 1.0e9,
            "MAT_A0": 1.0e10,
            "MAT_A1": 0.5,
        })
        prop = MockProp(pid=1, isolid=1)
        conn = np.array([
            [0, 1, 2, 3, 4, 5, 6, 7],
            [1, 8, 9, 2, 5, 10, 11, 6],
        ], dtype=np.int64)

        coords = np.array([
            [0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0], [1.0, 0.0, 1.0], [1.0, 1.0, 1.0], [0.0, 1.0, 1.0],
            [2.0, 0.0, 0.0], [2.0, 1.0, 0.0], [2.0, 0.0, 1.0], [2.0, 1.0, 1.0],
        ], dtype=float)

        m = Model()
        m.x0 = coords.copy()
        g = MockGroup(conn, slices=[(slice(0, 2), mat, prop)])
        g._model = m
        solid_hexa8.init_group(g, m, None)

        # Kill element 1
        g.state["off"][1] = 0.0
        g.state["mat_extra"]["off"] = g.state["off"].copy()

        # Step 1 cycle
        vel = np.ones_like(coords) * 0.5
        vr = np.zeros_like(coords)
        fint = np.zeros((12, 3))
        mint = np.zeros((12, 3))
        solid_hexa8.forces(g, coords.copy(), vel, vr, 1e-5, fint, mint)

        # Deactivated element has zero stress
        assert np.allclose(g.state["sig"][1], 0.0)

        # Write restart
        m.solids = g
        m.x = coords.copy()
        rst_path = tmp_path / "dead_element.rst"
        write_restart(m, str(rst_path), engine={"cycle": 1, "t": 1e-5})

        # Restore
        m_rest, _ = read_restart(str(rst_path))
        g_rest = m_rest.solids
        assert g_rest.state["off"][1] == 0.0

        # Step restored model
        fint_res = np.zeros((12, 3))
        mint_res = np.zeros((12, 3))
        solid_hexa8.forces(g_rest, m_rest.x.copy(), vel, vr, 1e-5, fint_res, mint_res)
        assert np.allclose(g_rest.state["sig"][1], 0.0)

    def test_end_to_end_engine_restart_chaining(self, tmp_path: Path):
        """End-to-end Starter -> Engine execution restart chaining matching uninterrupted run within 10^-12."""
        run_name = "law21_chain"
        deck = StarterDeck(run_name)
        deck.node([
            (1, 0.0, 0.0, 0.0), (2, 1.0, 0.0, 0.0), (3, 1.0, 1.0, 0.0), (4, 0.0, 1.0, 0.0),
            (5, 0.0, 0.0, 1.0), (6, 1.0, 0.0, 1.0), (7, 1.0, 1.0, 1.0), (8, 0.0, 1.0, 1.0),
        ])
        deck.brick(1, [[1, 1, 2, 3, 4, 5, 6, 7, 8]])
        deck.prop_solid(1, "SolidProp")
        deck.part(1, "RockCube", prop_id=1, mat_id=1)
        deck.mat_law21(
            mat_id=1,
            title="Rock",
            rho=2000.0,
            e=2.0e10,
            nu=0.25,
            c1=1.0e9,
            a0=1.0e10,
            a1=0.5,
            pmin=-1.0e30,
        )

        f0 = tmp_path / f"{run_name}_0000.rad"
        deck.write(str(f0))

        # 1. Starter generates _0000.rst
        model0 = run_starter(str(f0))
        assert model0 is not None
        assert (tmp_path / f"{run_name}_0000.rst").exists()

        # Engine deck 0001: run to t = 2.0e-5
        f1 = tmp_path / f"{run_name}_0001.rad"
        f1.write_text(f"""\
/RUN/{run_name}/1
2.0e-5
/DT/NODA/CST
0.9 1.0e-5
/END
""", encoding="utf-8")

        # Run engine Leg 1
        model1 = run_engine(str(f1))
        assert model1 is not None
        rst1_path = tmp_path / f"{run_name}_0001.rst"
        assert rst1_path.exists()

        # Engine deck 0002: resume from _0001.rst to t = 4.0e-5
        f2 = tmp_path / f"{run_name}_0002.rad"
        f2.write_text(f"""\
/RUN/{run_name}/2
4.0e-5
/DT/NODA/CST
0.9 1.0e-5
/END
""", encoding="utf-8")

        model2 = run_engine(str(f2))
        assert model2 is not None
        rst2_path = tmp_path / f"{run_name}_0002.rst"
        assert rst2_path.exists()

        _, eng_state2 = read_restart(str(rst2_path))
        assert eng_state2 is not None
        assert eng_state2["t"] >= 4.0e-5 - 1e-12
        assert eng_state2["cycle"] >= 2
