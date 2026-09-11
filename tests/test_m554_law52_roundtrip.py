"""
Milestone M554: /MAT/LAW52 (/MAT/GURSON, /MAT/PLAS_GURS)
Exhaustive Roundtrip, Negative Validation & Restart (.rst) Serialization Auditor Suite.

Fortran origins:
  - starter/source/materials/mat/mat052/hm_read_mat52.F
  - engine/source/materials/mat/mat052/sigeps52.F & sigeps52c.F
  - engine/source/output/restart/wrrestp.F & rdresb.F
  - config/CFG/radioss110/MAT/matl52_gurson.cfg & radioss130/MAT/matl52_gurson.cfg

Audits:
  1. Fixed-Format Deck Roundtrip:
     - Standard 20-column fixed format with StarterDeck.mat_law52.
     - Re-parse using read_mat_law52 and parse_starter_deck.
     - Assert exact equality for all 24 card fields:
       rho, E, nu, iflag, fsmooth, fcut, A, B, n, C, P, q1, q2, q3, s_N, eps_N,
       f0 (f_i), f_N, f_c, f_F, f_u, itable, Xfac, Yfac.
     - Direct entity object invocation with MatLaw52 entity.
     - Preserved default values for minimal card.
     - Consistency across Model.mat_law52s AST records and Model.materials runtime entities.

  2. Free-Format Comma-Delimited Roundtrip:
     - Write and re-parse free-format deck blocks for /MAT/LAW52, /MAT/GURSON, and /MAT/PLAS_GURS.
     - Comma-separated with whitespace and compact comma-delimited without whitespace.
     - Cross-dialect roundtrip (free -> parse -> fixed -> parse -> exact equality).

  3. Negative Starter Diagnostics:
     - Density rho <= 0 (error).
     - Young's modulus E <= 0 (error).
     - Poisson's ratio nu < 0 or nu >= 0.5 (error).
     - Inconsistent void volume fractions: f_F < f_c, f_F < f_0, or f_c < f_0 (ANCMSG 1745 error).
     - Warning for un-filtered strain rate: C > 0, P > 0, fcut == 0 (ANCMSG 1220 warning).
     - Unsupported element types (springs, beams, trusses) assigned to LAW52 (ANCMSG 306 error).
     - _ALLOWED_LAWS registry and _MAT_CHECKS dispatch.

  4. Restart (.rst) Serialization:
     - MatLaw52 entity dataclass pickling and unpickling fidelity.
     - Law52Params physics object serialization.
     - Material state arrays: matrix plastic strain epsm, matrix yield stress sigm,
       void damage channels (f*, fg, fn, f, f*), element alive mask off, off52.
     - Material state arrays preserved across write_restart / read_restart contract.
     - Constitutive dynamic cycle continuation from restart matching uninterrupted run within 10^-12 (solid & shell).
     - Element kernel force continuation within 10^-12.
     - End-to-end engine execution restart chaining matching uninterrupted run within 10^-12.
"""

from __future__ import annotations

import copy
import math
from pathlib import Path
import pickle
from typing import Any, Dict, List, Tuple
import numpy as np
import pytest

from pyradioss.common.messages import MessageLog
from pyradioss.elements import shell_bt4, solid_hexa8
from pyradioss.engine.engine import run_engine
from pyradioss.input.deck_reader import read_deck
from pyradioss.input.deck_writer import StarterDeck
from pyradioss.input.starter_keywords import parse_starter_deck, read_mat_law52
from pyradioss.materials.law52_gurson import (
    Law52Params,
    build_law52,
    extra_shapes,
    shell_update_law52,
    solid_update_law52,
    sound_speed_shell_law52,
    sound_speed_solid_law52,
)
from pyradioss import materials
from pyradioss.model.entities import MatGurson, MatLaw52, MatPlasGurs, Material
from pyradioss.model.model import Model
from pyradioss.starter.checks import (
    _ALLOWED_LAWS,
    _MAT_CHECKS,
    check_mat_law52,
    check_materials,
    check_model,
)
from pyradioss.starter.restart import read_restart, write_restart
from pyradioss.starter.starter import run_starter


# ============================================================================
# Helpers
# ============================================================================

def _parse_deck_str(tmp_path: Path, text: str, name: str = "TEST_LAW52") -> Tuple[Model, MessageLog]:
    """Write deck text to file and parse into Model and MessageLog."""
    deck_path = tmp_path / f"{name}_0000.rad"
    deck_path.write_text(text.strip() + "\n", encoding="utf-8")
    blocks = read_deck(str(deck_path))
    model = Model()
    log = MessageLog()
    parse_starter_deck(blocks, model, log)
    return model, log


def _assert_law52_all_fields_exact(
    m: MatLaw52,
    expected: Dict[str, Any],
    tol: float = 1e-6,
) -> None:
    """Assert precise equality for all 24 card fields:
    rho, E, nu, iflag, fsmooth, fcut, A, B, n, C, P, q1, q2, q3, s_N, eps_N,
    f0 (f_i), f_N, f_c, f_F, f_u, itable, Xfac, Yfac.
    """
    # 1. Density (Card 1)
    if "rho" in expected:
        assert m.rho == pytest.approx(expected["rho"], rel=tol)
        assert m.rho0 == pytest.approx(expected["rho"], rel=tol)
    if "refer_rho" in expected:
        assert m.refer_rho == pytest.approx(expected["refer_rho"], rel=tol)
        assert m.rhor == pytest.approx(expected["refer_rho"], rel=tol)

    # 2. Elastic & Rate Filtering (Card 2)
    if "e" in expected:
        assert m.e == pytest.approx(expected["e"], rel=tol)
        assert m.E == pytest.approx(expected["e"], rel=tol)
    if "nu" in expected:
        assert m.nu == pytest.approx(expected["nu"], rel=tol)
    if "iflag" in expected:
        assert m.iflag == expected["iflag"]
    if "fsmooth" in expected:
        assert m.fsmooth == expected["fsmooth"]
    if "fcut" in expected:
        assert m.fcut == pytest.approx(expected["fcut"], rel=tol)

    # 3. Hardening & Viscoplasticity (Card 3: A, B, n, C, P)
    if "a" in expected:
        assert m.a == pytest.approx(expected["a"], rel=tol)
        assert m.A == pytest.approx(expected["a"], rel=tol)
        assert m.yield_stress == pytest.approx(expected["a"], rel=tol)
    if "b" in expected:
        assert m.b == pytest.approx(expected["b"], rel=tol)
        assert m.B == pytest.approx(expected["b"], rel=tol)
        assert m.hardening_b == pytest.approx(expected["b"], rel=tol)
    if "n" in expected:
        assert m.n == pytest.approx(expected["n"], rel=tol)
        assert m.hardening_n == pytest.approx(expected["n"], rel=tol)
    if "c" in expected:
        assert m.c == pytest.approx(expected["c"], rel=tol)
        assert m.C == pytest.approx(expected["c"], rel=tol)
    if "pc" in expected:
        assert m.pc == pytest.approx(expected["pc"], rel=tol)
        assert m.P == pytest.approx(expected["pc"], rel=tol)

    # 4. GTN Yield & Nucleation (Card 4: q1, q2, q3, s_N, eps_N)
    if "q1" in expected:
        assert m.q1 == pytest.approx(expected["q1"], rel=tol)
        # f_u derived from q1: fu = 1.0 / q1
        expected_fu = 1.0 / expected["q1"]
        assert m.fu == pytest.approx(expected_fu, rel=tol)
    if "q2" in expected:
        assert m.q2 == pytest.approx(expected["q2"], rel=tol)
    if "q3" in expected:
        assert m.q3 == pytest.approx(expected["q3"], rel=tol)
    if "s_n" in expected:
        assert m.s_n == pytest.approx(expected["s_n"], rel=tol)
        assert m.sn == pytest.approx(expected["s_n"], rel=tol)
    if "eps_n" in expected:
        assert m.eps_n == pytest.approx(expected["eps_n"], rel=tol)
        assert m.epsn == pytest.approx(expected["eps_n"], rel=tol)

    # 5. Void Volume Fractions (Card 5: f0, f_N, f_c, f_F)
    if "f_i" in expected:
        assert m.f_i == pytest.approx(expected["f_i"], rel=tol)
        assert m.f0 == pytest.approx(expected["f_i"], rel=tol)
        assert m.f_0 == pytest.approx(expected["f_i"], rel=tol)
        assert m.fi == pytest.approx(expected["f_i"], rel=tol)
    if "f_n" in expected:
        assert m.f_n == pytest.approx(expected["f_n"], rel=tol)
        assert m.fn == pytest.approx(expected["f_n"], rel=tol)
    if "f_c" in expected:
        assert m.f_c == pytest.approx(expected["f_c"], rel=tol)
        assert m.fc == pytest.approx(expected["f_c"], rel=tol)
    if "f_f" in expected:
        assert m.f_f == pytest.approx(expected["f_f"], rel=tol)
        assert m.ff == pytest.approx(expected["f_f"], rel=tol)

    # 6. Tabulated Curve / Factors (Card 6: itable, Xfac, Yfac)
    if "itable" in expected:
        assert m.itable == expected["itable"]
    if "xfac" in expected:
        assert m.xfac == pytest.approx(expected["xfac"], rel=tol)
    if "yfac" in expected:
        assert m.yfac == pytest.approx(expected["yfac"], rel=tol)


class MockProp:
    """Mock solid property for element kernel testing."""
    def __init__(self, pid: int = 1, isolid: int = 1, **kwargs: Any):
        self.id = pid
        self.isolid = isolid
        self.params = {"isolid": isolid, **kwargs}


class MockGroup:
    """Mock element group with conn and state."""
    def __init__(self, conn: np.ndarray, ids: np.ndarray | None = None, slices: list | None = None):
        self.conn = np.asarray(conn, dtype=np.int64)
        self.n = len(self.conn)
        self.ids = np.arange(1, self.n + 1, dtype=np.int64) if ids is None else np.asarray(ids, dtype=np.int64)
        self.state: dict[str, Any] = {}
        if slices is not None:
            self.state["slices"] = slices
        self._model: Any = None


# ============================================================================
# Section 1: Fixed-Format Deck Roundtrip
# ============================================================================

class TestLaw52FixedFormatRoundtrip:
    """Audit standard 20-column fixed-format deck generation and re-parsing."""

    def test_fixed_format_all_card_fields(self, tmp_path: Path):
        """Write /MAT/LAW52 with explicit values for all 24 fields and re-parse."""
        expected = {
            "rho": 7.85e-3,
            "refer_rho": 7.85e-3,
            "e": 206000.0,
            "nu": 0.30,
            "iflag": 1,
            "fsmooth": 0,
            "fcut": 1000.0,
            "a": 420.0,
            "b": 150.0,
            "n": 0.22,
            "c": 0.015,
            "pc": 2.5,
            "q1": 1.5,
            "q2": 1.0,
            "q3": 2.25,
            "s_n": 0.1,
            "eps_n": 0.3,
            "f_i": 0.002,
            "f_n": 0.04,
            "f_c": 0.06,
            "f_f": 0.18,
            "itable": 101,
            "xfac": 1.08,
            "yfac": 1.03,
        }

        deck = StarterDeck("FIXED_ROUNDTRIP")
        deck.mat_law52(
            mid=52,
            title="GTN High-Strength Steel",
            rho=expected["rho"],
            refer_rho=expected["refer_rho"],
            e=expected["e"],
            nu=expected["nu"],
            iflag=expected["iflag"],
            fsmooth=expected["fsmooth"],
            fcut=expected["fcut"],
            a=expected["a"],
            b=expected["b"],
            n=expected["n"],
            c=expected["c"],
            pc=expected["pc"],
            q1=expected["q1"],
            q2=expected["q2"],
            q3=expected["q3"],
            s_n=expected["s_n"],
            eps_n=expected["eps_n"],
            f_i=expected["f_i"],
            f_n=expected["f_n"],
            f_c=expected["f_c"],
            f_f=expected["f_f"],
            itable=expected["itable"],
            xfac=expected["xfac"],
            yfac=expected["yfac"],
            fixed_format=True,
        )

        rendered = deck.render()
        assert "/MAT/LAW52/52" in rendered
        assert "GTN High-Strength Steel" in rendered

        model, log = _parse_deck_str(tmp_path, rendered, "fixed_all_fields")
        assert not log.has_errors, f"Parse errors: {log.errors}"
        assert 52 in model.mat_law52s

        m = model.mat_law52s[52]
        _assert_law52_all_fields_exact(m, expected)

        # Verify runtime Material entity
        assert 52 in model.materials
        mat_rt = model.materials[52]
        assert mat_rt.law == 52
        assert mat_rt.params["E"] == pytest.approx(expected["e"])
        assert mat_rt.params["nu"] == pytest.approx(expected["nu"])
        assert mat_rt.params["q1"] == pytest.approx(expected["q1"])
        assert mat_rt.params["fu"] == pytest.approx(1.0 / expected["q1"])

    def test_fixed_format_from_mat_law52_entity(self, tmp_path: Path):
        """Pass a MatLaw52 entity directly to StarterDeck.mat_law52."""
        mat_entity = MatLaw52(
            id=12,
            title="Entity Driven GTN",
            rho=7.8e-3,
            refer_rho=7.8e-3,
            e=210000.0,
            nu=0.29,
            a=380.0,
            b=120.0,
            n=0.20,
            c=0.01,
            pc=2.0,
            q1=1.4,
            q2=0.95,
            q3=1.96,
            s_n=0.08,
            eps_n=0.25,
            f_i=0.001,
            f_n=0.03,
            f_c=0.05,
            f_f=0.15,
            iflag=1,
            fsmooth=0,
            fcut=800.0,
            itable=15,
            xfac=1.04,
            yfac=1.02,
        )

        deck = StarterDeck("ENTITY_WRITER")
        deck.mat_law52(mat_entity)
        rendered = deck.render()
        assert "/MAT/LAW52/12" in rendered
        assert "Entity Driven GTN" in rendered

        model, log = _parse_deck_str(tmp_path, rendered, "fixed_from_entity")
        assert not log.has_errors
        assert 12 in model.mat_law52s

        m_parsed = model.mat_law52s[12]
        assert m_parsed.id == 12
        assert m_parsed.rho == pytest.approx(7.8e-3)
        assert m_parsed.e == pytest.approx(210000.0)
        assert m_parsed.nu == pytest.approx(0.29)
        assert m_parsed.a == pytest.approx(380.0)
        assert m_parsed.b == pytest.approx(120.0)
        assert m_parsed.n == pytest.approx(0.20)
        assert m_parsed.c == pytest.approx(0.01)
        assert m_parsed.pc == pytest.approx(2.0)
        assert m_parsed.q1 == pytest.approx(1.4)
        assert m_parsed.fu == pytest.approx(1.0 / 1.4)
        assert m_parsed.s_n == pytest.approx(0.08)
        assert m_parsed.eps_n == pytest.approx(0.25)
        assert m_parsed.f_i == pytest.approx(0.001)
        assert m_parsed.f_n == pytest.approx(0.03)
        assert m_parsed.f_c == pytest.approx(0.05)
        assert m_parsed.f_f == pytest.approx(0.15)
        assert m_parsed.itable == 15
        assert m_parsed.xfac == pytest.approx(1.04)
        assert m_parsed.yfac == pytest.approx(1.02)

    def test_fixed_format_keyword_aliases_invocation(self, tmp_path: Path):
        """Invoke StarterDeck.mat_law52 with uppercase/abbreviated kwargs (A, B, C, P, f0, sn, epsn)."""
        deck = StarterDeck("KWARGS_ALIASES")
        deck.mat_law52(
            mid=7,
            title="Kwargs GTN",
            rho=7.8e-3,
            e=210000.0,
            nu=0.3,
            A=450.0,
            B=200.0,
            n=0.25,
            C=0.02,
            P=1.8,
            q1=1.5,
            q2=1.0,
            q3=2.25,
            sn=0.1,
            epsn=0.3,
            f0=0.001,
            fn=0.04,
            fc=0.05,
            ff=0.15,
            fcut=1000.0,
        )

        rendered = deck.render()
        model, log = _parse_deck_str(tmp_path, rendered, "fixed_kwargs_aliases")
        assert not log.has_errors
        m = model.mat_law52s[7]
        assert m.a == pytest.approx(450.0)
        assert m.A == pytest.approx(450.0)
        assert m.b == pytest.approx(200.0)
        assert m.B == pytest.approx(200.0)
        assert m.c == pytest.approx(0.02)
        assert m.C == pytest.approx(0.02)
        assert m.pc == pytest.approx(1.8)
        assert m.P == pytest.approx(1.8)
        assert m.f_i == pytest.approx(0.001)
        assert m.f0 == pytest.approx(0.001)
        assert m.f_n == pytest.approx(0.04)
        assert m.fn == pytest.approx(0.04)
        assert m.f_c == pytest.approx(0.05)
        assert m.fc == pytest.approx(0.05)
        assert m.f_f == pytest.approx(0.15)
        assert m.ff == pytest.approx(0.15)
        assert m.s_n == pytest.approx(0.1)
        assert m.sn == pytest.approx(0.1)
        assert m.eps_n == pytest.approx(0.3)
        assert m.epsn == pytest.approx(0.3)

    def test_fixed_format_fortran_defaults(self, tmp_path: Path):
        """Verify default values matching hm_read_mat52.F when optional fields are omitted."""
        deck = StarterDeck("DEFAULTS_FIXED")
        deck.mat_law52(
            mid=9,
            title="Minimal GTN",
            rho=7.8e-3,
            e=210000.0,
            nu=0.3,
            a=400.0,
            b=100.0,
            n=0.2,
            f_i=0.001,
            f_n=0.04,
            f_c=0.05,
            f_f=0.15,
        )

        rendered = deck.render()
        model, log = _parse_deck_str(tmp_path, rendered, "fixed_defaults")
        assert not log.has_errors
        m = model.mat_law52s[9]
        assert m.refer_rho == pytest.approx(7.8e-3)
        assert m.c == pytest.approx(1e30)
        assert m.pc == pytest.approx(1.0)
        assert m.fcut == pytest.approx(1e30)
        assert m.q1 == pytest.approx(1e-20)
        assert m.fu == pytest.approx(1.0 / 1e-20)
        assert m.itable == 0
        assert m.xfac == pytest.approx(1.0)
        assert m.yfac == pytest.approx(1.0)


# ============================================================================
# Section 2: Free-Format Comma-Delimited Roundtrip
# ============================================================================

class TestLaw52FreeFormatRoundtrip:
    """Audit free-format comma-delimited decks for /MAT/LAW52, /MAT/GURSON, and /MAT/PLAS_GURS."""

    def test_free_format_comma_delimited_law52(self, tmp_path: Path):
        """Write /MAT/LAW52 in free format with commas and re-parse."""
        expected = {
            "rho": 7.8e-3,
            "refer_rho": 7.8e-3,
            "e": 210000.0,
            "nu": 0.3,
            "iflag": 1,
            "fsmooth": 0,
            "fcut": 1000.0,
            "a": 400.0,
            "b": 100.0,
            "n": 0.2,
            "c": 0.01,
            "pc": 2.0,
            "q1": 1.5,
            "q2": 1.0,
            "q3": 2.25,
            "s_n": 0.1,
            "eps_n": 0.3,
            "f_i": 0.001,
            "f_n": 0.04,
            "f_c": 0.05,
            "f_f": 0.15,
            "itable": 10,
            "xfac": 1.05,
            "yfac": 1.02,
        }

        deck = StarterDeck("FREE_LAW52")
        deck.mat_law52(
            mid=1,
            title="Free Law52",
            fixed_format=False,
            comma=True,
            **expected,
        )

        rendered = deck.render()
        assert "/MAT/LAW52/1" in rendered
        assert "," in rendered

        model, log = _parse_deck_str(tmp_path, rendered, "free_law52")
        assert not log.has_errors
        m = model.mat_law52s[1]
        _assert_law52_all_fields_exact(m, expected)

    def test_free_format_comma_delimited_gurson_alias(self, tmp_path: Path):
        """Write /MAT/GURSON alias in free format with commas and re-parse."""
        expected = {
            "rho": 7.85e-3,
            "refer_rho": 7.85e-3,
            "e": 205000.0,
            "nu": 0.29,
            "iflag": 0,
            "fsmooth": 0,
            "fcut": 500.0,
            "a": 350.0,
            "b": 80.0,
            "n": 0.18,
            "c": 0.02,
            "pc": 1.5,
            "q1": 1.4,
            "q2": 0.9,
            "q3": 2.0,
            "s_n": 0.08,
            "eps_n": 0.25,
            "f_i": 0.002,
            "f_n": 0.03,
            "f_c": 0.06,
            "f_f": 0.18,
            "itable": 20,
            "xfac": 1.1,
            "yfac": 1.04,
        }

        deck = StarterDeck("FREE_GURSON")
        deck.mat_gurson(
            mid=2,
            title="Free Gurson Alias",
            fixed_format=False,
            comma=True,
            **expected,
        )

        rendered = deck.render()
        assert "/MAT/GURSON/2" in rendered
        assert "Free Gurson Alias" in rendered

        model, log = _parse_deck_str(tmp_path, rendered, "free_gurson")
        assert not log.has_errors
        assert 2 in model.mat_gursons
        assert 2 in model.mat_law52s
        assert 2 in model.materials
        m = model.mat_gursons[2]
        _assert_law52_all_fields_exact(m, expected)

    def test_free_format_comma_delimited_plas_gurs_alias(self, tmp_path: Path):
        """Write /MAT/PLAS_GURS alias in free format with commas and re-parse."""
        expected = {
            "rho": 7.9e-3,
            "refer_rho": 7.9e-3,
            "e": 200000.0,
            "nu": 0.28,
            "iflag": 1,
            "fsmooth": 0,
            "fcut": 2000.0,
            "a": 500.0,
            "b": 150.0,
            "n": 0.25,
            "c": 0.005,
            "pc": 2.5,
            "q1": 1.6,
            "q2": 1.1,
            "q3": 2.5,
            "s_n": 0.12,
            "eps_n": 0.35,
            "f_i": 0.0005,
            "f_n": 0.05,
            "f_c": 0.04,
            "f_f": 0.12,
            "itable": 30,
            "xfac": 1.02,
            "yfac": 1.01,
        }

        deck = StarterDeck("FREE_PLAS_GURS")
        deck.mat_plas_gurs(
            mid=3,
            title="Free PlasGurs Alias",
            fixed_format=False,
            comma=True,
            **expected,
        )

        rendered = deck.render()
        assert "/MAT/PLAS_GURS/3" in rendered
        assert "Free PlasGurs Alias" in rendered

        model, log = _parse_deck_str(tmp_path, rendered, "free_plas_gurs")
        assert not log.has_errors
        assert 3 in model.mat_plas_gurs
        assert 3 in model.mat_law52s
        assert 3 in model.materials
        m = model.mat_plas_gurs[3]
        _assert_law52_all_fields_exact(m, expected)

    def test_compact_comma_delimited_no_whitespace(self, tmp_path: Path):
        """Verify parsing of compact comma-delimited cards with no spaces between values."""
        deck_text = """/BEGIN
Compact comma deck
/MAT/LAW52/4
Compact No Spaces
7.85e-3,7.85e-3
210000.0,0.3,1,0,1000.0,1
400.0,100.0,0.2,0.01,2.0
1.5,1.0,2.25,0.1,0.3
0.001,0.04,0.05,0.15
10,1.05,1.02
/END
"""
        model, log = _parse_deck_str(tmp_path, deck_text, "compact_comma")
        assert not log.has_errors
        m = model.mat_law52s[4]
        assert m.rho == pytest.approx(7.85e-3)
        assert m.e == pytest.approx(210000.0)
        assert m.nu == pytest.approx(0.3)
        assert m.iflag == 1
        assert m.fcut == pytest.approx(1000.0)
        assert m.a == pytest.approx(400.0)
        assert m.b == pytest.approx(100.0)
        assert m.n == pytest.approx(0.2)
        assert m.c == pytest.approx(0.01)
        assert m.pc == pytest.approx(2.0)
        assert m.q1 == pytest.approx(1.5)
        assert m.q2 == pytest.approx(1.0)
        assert m.q3 == pytest.approx(2.25)
        assert m.s_n == pytest.approx(0.1)
        assert m.eps_n == pytest.approx(0.3)
        assert m.f_i == pytest.approx(0.001)
        assert m.f_n == pytest.approx(0.04)
        assert m.f_c == pytest.approx(0.05)
        assert m.f_f == pytest.approx(0.15)
        assert m.itable == 10
        assert m.xfac == pytest.approx(1.05)
        assert m.yfac == pytest.approx(1.02)

    def test_cross_dialect_roundtrip(self, tmp_path: Path):
        """Cross-dialect roundtrip: Free format -> parse -> write fixed -> parse -> verify equality."""
        deck_free = StarterDeck("CROSS_FREE")
        deck_free.mat_law52(
            mid=5,
            title="Cross Material",
            rho=7.82e-3,
            e=208000.0,
            nu=0.295,
            a=410.0,
            b=110.0,
            n=0.21,
            c=0.012,
            pc=2.2,
            q1=1.45,
            q2=0.98,
            q3=2.15,
            s_n=0.09,
            eps_n=0.28,
            f_i=0.0015,
            f_n=0.035,
            f_c=0.055,
            f_f=0.16,
            iflag=1,
            fcut=1200.0,
            itable=8,
            xfac=1.06,
            yfac=1.03,
            fixed_format=False,
            comma=True,
        )
        m1, log1 = _parse_deck_str(tmp_path, deck_free.render(), "cross_stage1")
        assert not log1.has_errors
        mat1 = m1.mat_law52s[5]

        # Write out as fixed format
        deck_fixed = StarterDeck("CROSS_FIXED")
        deck_fixed.mat_law52(mat1, fixed_format=True)
        m2, log2 = _parse_deck_str(tmp_path, deck_fixed.render(), "cross_stage2")
        assert not log2.has_errors
        mat2 = m2.mat_law52s[5]

        # Assert precise match across dialect conversion
        assert mat2.rho == pytest.approx(mat1.rho, rel=1e-6)
        assert mat2.e == pytest.approx(mat1.e, rel=1e-6)
        assert mat2.nu == pytest.approx(mat1.nu, rel=1e-6)
        assert mat2.a == pytest.approx(mat1.a, rel=1e-6)
        assert mat2.b == pytest.approx(mat1.b, rel=1e-6)
        assert mat2.n == pytest.approx(mat1.n, rel=1e-6)
        assert mat2.c == pytest.approx(mat1.c, rel=1e-6)
        assert mat2.pc == pytest.approx(mat1.pc, rel=1e-6)
        assert mat2.q1 == pytest.approx(mat1.q1, rel=1e-6)
        assert mat2.fu == pytest.approx(mat1.fu, rel=1e-6)
        assert mat2.f_i == pytest.approx(mat1.f_i, rel=1e-6)
        assert mat2.f_n == pytest.approx(mat1.f_n, rel=1e-6)
        assert mat2.f_c == pytest.approx(mat1.f_c, rel=1e-6)
        assert mat2.f_f == pytest.approx(mat1.f_f, rel=1e-6)
        assert mat2.itable == mat1.itable
        assert mat2.xfac == pytest.approx(mat1.xfac, rel=1e-6)
        assert mat2.yfac == pytest.approx(mat1.yfac, rel=1e-6)


# ============================================================================
# Section 3: Negative Starter Diagnostics
# ============================================================================

class TestLaw52NegativeStarterChecks:
    """Audit starter diagnostics and erroring for illegal input values."""

    def test_negative_or_zero_density(self):
        """Density rho <= 0 must produce fatal error."""
        # Zero density
        m_zero = MatLaw52(id=1, rho=0.0, e=210000.0, nu=0.3)
        log1 = MessageLog()
        check_mat_law52(mat=m_zero, log=log1)
        assert log1.has_errors
        assert any("density" in str(e).lower() or "rho" in str(e).lower() for e in log1.errors)

        # Negative density
        m_neg = MatLaw52(id=2, rho=-7.8e-3, e=210000.0, nu=0.3)
        log2 = MessageLog()
        check_mat_law52(mat=m_neg, log=log2)
        assert log2.has_errors
        assert any("density" in str(e).lower() or "rho" in str(e).lower() for e in log2.errors)

    def test_negative_or_zero_youngs_modulus(self):
        """Young's modulus E <= 0 must produce fatal error."""
        # Zero E
        m_zero = MatLaw52(id=1, rho=7.8e-3, e=0.0, nu=0.3)
        log1 = MessageLog()
        check_mat_law52(mat=m_zero, log=log1)
        assert log1.has_errors
        assert any("young" in str(e).lower() or "e" in str(e).lower() for e in log1.errors)

        # Negative E
        m_neg = MatLaw52(id=2, rho=7.8e-3, e=-1000.0, nu=0.3)
        log2 = MessageLog()
        check_mat_law52(mat=m_neg, log=log2)
        assert log2.has_errors
        assert any("young" in str(e).lower() or "e" in str(e).lower() for e in log2.errors)

    def test_invalid_poissons_ratio(self):
        """Poisson's ratio must satisfy 0 <= nu < 0.5."""
        # Negative nu
        m_neg = MatLaw52(id=1, rho=7.8e-3, e=210000.0, nu=-0.1)
        log1 = MessageLog()
        check_mat_law52(mat=m_neg, log=log1)
        assert log1.has_errors
        assert any("poisson" in str(e).lower() or "nu" in str(e).lower() for e in log1.errors)

        # Incompressible limit nu == 0.5
        m_half = MatLaw52(id=2, rho=7.8e-3, e=210000.0, nu=0.5)
        log2 = MessageLog()
        check_mat_law52(mat=m_half, log=log2)
        assert log2.has_errors
        assert any("poisson" in str(e).lower() or "nu" in str(e).lower() for e in log2.errors)

        # nu > 0.5
        m_over = MatLaw52(id=3, rho=7.8e-3, e=210000.0, nu=0.55)
        log3 = MessageLog()
        check_mat_law52(mat=m_over, log=log3)
        assert log3.has_errors

        # Valid bounds
        m_zero_nu = MatLaw52(id=4, rho=7.8e-3, e=210000.0, nu=0.0, f_i=0.001, f_c=0.05, f_f=0.15)
        log4 = MessageLog()
        check_mat_law52(mat=m_zero_nu, log=log4)
        assert not log4.has_errors

        m_high_nu = MatLaw52(id=5, rho=7.8e-3, e=210000.0, nu=0.499, f_i=0.001, f_c=0.05, f_f=0.15)
        log5 = MessageLog()
        check_mat_law52(mat=m_high_nu, log=log5)
        assert not log5.has_errors

    def test_inconsistent_void_fractions_ancmsg_1745(self):
        """Inconsistent void fractions (f_F < f_c, f_F < f_0, or f_c < f_0) must trigger ANCMSG 1745."""
        # 1. f_F < f_c
        m_bad1 = MatLaw52(id=1, rho=7.8e-3, e=210000.0, nu=0.3, f_i=0.01, f_c=0.15, f_f=0.10)
        log1 = MessageLog()
        check_mat_law52(mat=m_bad1, log=log1)
        assert log1.has_errors
        assert any("1745" in str(e) or "void" in str(e).lower() for e in log1.errors)

        # 2. f_F < f_0
        m_bad2 = MatLaw52(id=2, rho=7.8e-3, e=210000.0, nu=0.3, f_i=0.20, f_c=0.10, f_f=0.15)
        log2 = MessageLog()
        check_mat_law52(mat=m_bad2, log=log2)
        assert log2.has_errors
        assert any("1745" in str(e) or "void" in str(e).lower() for e in log2.errors)

        # 3. f_c < f_0
        m_bad3 = MatLaw52(id=3, rho=7.8e-3, e=210000.0, nu=0.3, f_i=0.05, f_c=0.02, f_f=0.15)
        log3 = MessageLog()
        check_mat_law52(mat=m_bad3, log=log3)
        assert log3.has_errors
        assert any("1745" in str(e) or "void" in str(e).lower() for e in log3.errors)

        # 4. Valid monotonic sequence: f_0 <= f_c <= f_F passes
        m_good = MatLaw52(id=4, rho=7.8e-3, e=210000.0, nu=0.3, f_i=0.01, f_c=0.08, f_f=0.15)
        log4 = MessageLog()
        check_mat_law52(mat=m_good, log=log4)
        assert not log4.has_errors

    def test_unfiltered_strain_rate_warning_ancmsg_1220(self):
        """Warning ANCMSG 1220 is emitted when C > 0, P > 0, and fcut == 0."""
        # Unfiltered strain rate: C > 0, P > 0, Fcut == 0
        m_unfiltered = MatLaw52(
            id=1,
            rho=7.8e-3,
            e=210000.0,
            nu=0.3,
            c=0.05,
            pc=2.0,
            fcut=1e30,
            raw_c=0.05,
            raw_pc=2.0,
            raw_fcut=0.0,
            f_i=0.001,
            f_c=0.05,
            f_f=0.15,
        )
        log1 = MessageLog()
        check_mat_law52(mat=m_unfiltered, log=log1)
        assert not log1.has_errors
        assert log1.has_warnings
        assert any("1220" in str(w) or "strain rate filtering" in str(w).lower() for w in log1.warnings)

        # Properly filtered strain rate: Fcut > 0
        m_filtered = MatLaw52(
            id=2,
            rho=7.8e-3,
            e=210000.0,
            nu=0.3,
            c=0.05,
            pc=2.0,
            fcut=1000.0,
            raw_c=0.05,
            raw_pc=2.0,
            raw_fcut=1000.0,
            f_i=0.001,
            f_c=0.05,
            f_f=0.15,
        )
        log2 = MessageLog()
        check_mat_law52(mat=m_filtered, log=log2)
        assert not log2.has_errors
        assert not log2.has_warnings

        # C == 0 (no rate sensitivity) -> no warning even if Fcut == 0
        m_norate = MatLaw52(
            id=3,
            rho=7.8e-3,
            e=210000.0,
            nu=0.3,
            c=1e30,
            pc=1.0,
            fcut=1e30,
            raw_c=0.0,
            raw_pc=0.0,
            raw_fcut=0.0,
            f_i=0.001,
            f_c=0.05,
            f_f=0.15,
        )
        log3 = MessageLog()
        check_mat_law52(mat=m_norate, log=log3)
        assert not log3.has_errors
        assert not log3.has_warnings

    def test_unsupported_element_types_ancmsg_306(self):
        """1D element types (springs, beams, trusses) assigned to LAW52 produce ANCMSG 306 error."""
        class DummyElement:
            def __init__(self, mat_id: int):
                self.mat_id = mat_id

        class DummyGroup:
            def __init__(self, elements):
                self._elements = elements
            def values(self):
                return self._elements

        for unsupported_1d in ("springs", "beams", "trusses"):
            model = Model()
            m = MatLaw52(id=1, rho=7.8e-3, e=210000.0, nu=0.3, f_i=0.001, f_c=0.05, f_f=0.15)
            m.law = 52
            model.materials[1] = m
            model._element_groups = [(unsupported_1d, DummyGroup([DummyElement(1)]))]
            model.element_groups = lambda g=model._element_groups: g

            log = MessageLog()
            check_mat_law52(model=model, mat_id=1, mat=m, log=log)
            assert log.has_errors
            assert any("306" in str(e) or "1d" in str(e).lower() for e in log.errors)

    def test_allowed_laws_registry_and_check_model_integration(self):
        """Verify _ALLOWED_LAWS and _MAT_CHECKS registries."""
        # 1. 2D / 3D element families accept LAW52
        for family in ("bricks", "tetras", "penta6", "pyra5", "shells", "shells_qbat", "shells_qeph", "sh3n", "quads"):
            assert 52 in _ALLOWED_LAWS[family]
            assert "LAW52" in _ALLOWED_LAWS[family]
            assert "GURSON" in _ALLOWED_LAWS[family]
            assert "PLAS_GURS" in _ALLOWED_LAWS[family]

        # 2. 1D elements reject LAW52
        assert 52 not in _ALLOWED_LAWS["trusses"]
        assert 52 not in _ALLOWED_LAWS["beams"]
        assert _ALLOWED_LAWS["springs"] is None

        # 3. _MAT_CHECKS dispatch
        assert _MAT_CHECKS[52] is check_mat_law52
        assert _MAT_CHECKS["52"] is check_mat_law52
        assert _MAT_CHECKS["LAW52"] is check_mat_law52
        assert _MAT_CHECKS["GURSON"] is check_mat_law52
        assert _MAT_CHECKS["PLAS_GURS"] is check_mat_law52

        # 4. check_materials dispatch
        model = Model()
        m_good = MatLaw52(id=1, rho=7.8e-3, e=210000.0, nu=0.3, f_i=0.001, f_c=0.05, f_f=0.15)
        m_good.law = 52
        model.materials[1] = m_good
        log = MessageLog()
        check_materials(model, log)
        assert not log.has_errors


# ============================================================================
# Section 4: Restart (.rst) Serialization
# ============================================================================

class TestLaw52RestartSerialization:
    """Audit restart (.rst) serialization, state array fidelity, and cycle continuation."""

    def test_mat_law52_and_params_pickle_fidelity(self):
        """Pickle and unpickle MatLaw52 entity and Law52Params physics object."""
        m_orig = MatLaw52(
            id=52,
            title="Pickle Gurson",
            rho=7.8e-3,
            refer_rho=7.8e-3,
            e=210000.0,
            nu=0.3,
            a=400.0,
            b=150.0,
            n=0.22,
            c=0.01,
            pc=2.0,
            q1=1.5,
            q2=1.0,
            q3=2.25,
            s_n=0.1,
            eps_n=0.3,
            f_i=0.001,
            f_n=0.04,
            f_c=0.05,
            f_f=0.15,
            iflag=1,
            fsmooth=0,
            fcut=1000.0,
            itable=10,
            xfac=1.05,
            yfac=1.02,
        )

        data = pickle.dumps(m_orig)
        m_rest: MatLaw52 = pickle.loads(data)

        assert m_rest.id == m_orig.id
        assert m_rest.title == m_orig.title
        assert m_rest.rho == pytest.approx(m_orig.rho)
        assert m_rest.refer_rho == pytest.approx(m_orig.refer_rho)
        assert m_rest.e == pytest.approx(m_orig.e)
        assert m_rest.nu == pytest.approx(m_orig.nu)
        assert m_rest.a == pytest.approx(m_orig.a)
        assert m_rest.b == pytest.approx(m_orig.b)
        assert m_rest.n == pytest.approx(m_orig.n)
        assert m_rest.c == pytest.approx(m_orig.c)
        assert m_rest.pc == pytest.approx(m_orig.pc)
        assert m_rest.q1 == pytest.approx(m_orig.q1)
        assert m_rest.fu == pytest.approx(m_orig.fu)
        assert m_rest.f_i == pytest.approx(m_orig.f_i)
        assert m_rest.f_n == pytest.approx(m_orig.f_n)
        assert m_rest.f_c == pytest.approx(m_orig.f_c)
        assert m_rest.f_f == pytest.approx(m_orig.f_f)
        assert m_rest.itable == m_orig.itable
        assert m_rest.xfac == pytest.approx(m_orig.xfac)
        assert m_rest.yfac == pytest.approx(m_orig.yfac)

        # Law52Params pickling
        params_orig = build_law52(E=210000.0, nu=0.3, A=400.0, B=150.0, n=0.22)
        params_data = pickle.dumps(params_orig)
        params_rest = pickle.loads(params_data)
        assert params_rest.params["E"] == pytest.approx(210000.0)
        assert params_rest.params["nu"] == pytest.approx(0.3)

    def test_material_state_arrays_rst_preservation(self, tmp_path: Path):
        """Verify epsm, sigm, dmg (f*, fg, fn, f, f*), and off52 survive write_restart/read_restart."""
        deck = StarterDeck("RST_LAW52_ARRAYS")
        deck.mat_law52(
            mid=1,
            title="StateGurson",
            rho=7.85e-9,
            e=210000.0,
            nu=0.3,
            a=400.0,
            b=200.0,
            n=0.2,
            f_i=0.01,
            f_c=0.05,
            f_f=0.15,
        )
        deck.prop_solid(pid=1, title="SolidProp", isolid=1)
        deck.part(pid=1, title="SolidPart", prop_id=1, mat_id=1)
        deck.node([
            (1, 0.0, 0.0, 0.0), (2, 1.0, 0.0, 0.0), (3, 1.0, 1.0, 0.0), (4, 0.0, 1.0, 0.0),
            (5, 0.0, 0.0, 1.0), (6, 1.0, 0.0, 1.0), (7, 1.0, 1.0, 1.0), (8, 0.0, 1.0, 1.0),
        ])
        deck.brick(1, [[1, 1, 2, 3, 4, 5, 6, 7, 8]])

        rad_path = tmp_path / "rst_arrays_0000.rad"
        deck.write(str(rad_path))

        model = run_starter(str(rad_path))
        assert model is not None

        groups = dict(model.element_groups())
        assert "bricks" in groups
        bg = groups["bricks"]
        st = bg.state

        # Synthesize realistic persistent material state arrays
        n_elem = 1
        epsm_synth = np.array([0.0245])
        sigm_synth = np.array([485.6])
        dmg_synth = np.array([[0.038, 0.015, 0.012, 0.037, 0.038]])
        fg_synth = np.array([0.015])
        fn_synth = np.array([0.012])
        f_synth = np.array([0.037])
        fstar_synth = np.array([0.038])
        off_synth = np.array([1.0])
        off52_synth = np.array([1.0])

        st["mat_extra"]["epsm"] = epsm_synth.copy()
        st["mat_extra"]["sigm"] = sigm_synth.copy()
        st["mat_extra"]["dmg"] = dmg_synth.copy()
        st["mat_extra"]["fg"] = fg_synth.copy()
        st["mat_extra"]["fn"] = fn_synth.copy()
        st["mat_extra"]["f"] = f_synth.copy()
        st["mat_extra"]["fstar"] = fstar_synth.copy()
        st["mat_extra"]["off"] = off_synth.copy()
        st["mat_extra"]["off52"] = off52_synth.copy()

        # Write restart .rst
        rst_path = tmp_path / "rst_arrays_0001.rst"
        engine_dict = {
            "cycle": 200,
            "t": 2.5e-6,
            "dt": 5.0e-8,
            "energies": {"internal": 45.67, "kinetic": 12.34},
        }
        write_restart(model, str(rst_path), engine=engine_dict)

        # Restore from restart .rst
        rest_model, rest_engine = read_restart(str(rst_path))
        assert rest_engine["cycle"] == 200
        assert rest_engine["t"] == pytest.approx(2.5e-6)
        assert rest_engine["energies"]["internal"] == pytest.approx(45.67)

        rest_bg = dict(rest_model.element_groups())["bricks"]
        rest_extra = rest_bg.state["mat_extra"]

        np.testing.assert_array_equal(
            rest_extra["epsm"],
            epsm_synth,
            err_msg="Matrix plastic strain epsm did not survive restart serialization exactly",
        )
        np.testing.assert_array_equal(
            rest_extra["sigm"],
            sigm_synth,
            err_msg="Matrix flow stress sigm did not survive restart serialization exactly",
        )
        np.testing.assert_array_equal(
            rest_extra["dmg"],
            dmg_synth,
            err_msg="Void damage tensor dmg did not survive restart serialization exactly",
        )
        np.testing.assert_array_equal(
            rest_extra["fg"],
            fg_synth,
            err_msg="Void growth fraction fg did not survive restart serialization exactly",
        )
        np.testing.assert_array_equal(
            rest_extra["fn"],
            fn_synth,
            err_msg="Void nucleation fraction fn did not survive restart serialization exactly",
        )
        np.testing.assert_array_equal(
            rest_extra["f"],
            f_synth,
            err_msg="Total void fraction f did not survive restart serialization exactly",
        )
        np.testing.assert_array_equal(
            rest_extra["fstar"],
            fstar_synth,
            err_msg="Effective void fraction fstar did not survive restart serialization exactly",
        )
        np.testing.assert_array_equal(
            rest_extra["off52"],
            off52_synth,
            err_msg="Element alive mask off52 did not survive restart serialization exactly",
        )

    def test_constitutive_cycle_restart_continuation_solid(self):
        """Run multi-cycle solid_update_law52 and verify pickled continuation matches uninterrupted within 10^-12."""
        mat = build_law52(
            E=210000.0,
            nu=0.3,
            rho0=7.85e-9,
            A=400.0,
            B=250.0,
            n=0.2,
            CSD=0.01,
            VISP=2.0,
            q1=1.5,
            q2=1.0,
            q3=2.25,
            s_N=0.1,
            eps_N=0.3,
            f_I=0.01,
            f_N=0.04,
            f_C=0.06,
            f_F=0.15,
        )

        n_steps = 12
        dt = 1.0e-6
        # Multiaxial strain increments with tension, triaxiality, and shear
        deps_steps = [
            np.array([[0.002, -0.0006, -0.0006, 0.001, 0.0, 0.0]]),
            np.array([[0.003, -0.0009, -0.0009, 0.0015, 0.0005, 0.0]]),
            np.array([[0.0025, 0.001, -0.001, 0.002, 0.001, 0.0005]]),
            np.array([[0.0015, 0.0015, 0.0015, 0.001, 0.0, 0.0]]),
            np.array([[0.001, 0.002, 0.0005, 0.0025, 0.001, 0.0]]),
            np.array([[-0.001, -0.001, -0.001, 0.0015, 0.0, 0.0]]),
            np.array([[0.002, 0.001, -0.0005, 0.001, 0.0005, 0.0]]),
            np.array([[0.0025, -0.0008, -0.0008, 0.002, 0.0, 0.0]]),
            np.array([[0.003, 0.0015, -0.001, 0.0015, 0.001, 0.0005]]),
            np.array([[0.0015, -0.0005, -0.0005, 0.001, 0.0, 0.0]]),
            np.array([[0.001, 0.001, 0.001, 0.002, 0.0, 0.0]]),
            np.array([[0.0005, -0.0002, -0.0002, 0.0005, 0.0, 0.0]]),
        ]

        # 1. Uninterrupted run (steps 0..11)
        extra_unc: Dict[str, Any] = {}
        sig_unc = np.zeros((1, 6))
        epsp_unc = np.zeros(1)

        for step in range(n_steps):
            sig_unc, epsp_unc = solid_update_law52(
                mat, sig_unc, deps_steps[step], epsp=epsp_unc, dt=dt, extra=extra_unc
            )

        # 2. Chained run: Leg 1 (steps 0..5)
        extra_chn: Dict[str, Any] = {}
        sig_chn = np.zeros((1, 6))
        epsp_chn = np.zeros(1)

        split = 6
        for step in range(split):
            sig_chn, epsp_chn = solid_update_law52(
                mat, sig_chn, deps_steps[step], epsp=epsp_chn, dt=dt, extra=extra_chn
            )

        # Pickle restart checkpoint
        checkpoint = pickle.dumps({
            "sig": sig_chn,
            "epsp": epsp_chn,
            "extra": copy.deepcopy(extra_chn),
        })

        # Leg 2: restore from checkpoint and continue steps 6..11
        restored = pickle.loads(checkpoint)
        sig_res = restored["sig"]
        epsp_res = restored["epsp"]
        extra_res = restored["extra"]

        for step in range(split, n_steps):
            sig_res, epsp_res = solid_update_law52(
                mat, sig_res, deps_steps[step], epsp=epsp_res, dt=dt, extra=extra_res
            )

        # Assert precise equality between uninterrupted and restarted run
        np.testing.assert_allclose(
            sig_res,
            sig_unc,
            rtol=1e-12,
            atol=1e-12,
            err_msg="Solid stress tensor mismatch between uninterrupted and restarted continuation",
        )
        np.testing.assert_allclose(
            epsp_res,
            epsp_unc,
            rtol=1e-12,
            atol=1e-12,
            err_msg="Solid plastic strain mismatch between uninterrupted and restarted continuation",
        )
        np.testing.assert_allclose(
            extra_res["epsm"],
            extra_unc["epsm"],
            rtol=1e-12,
            atol=1e-12,
            err_msg="Solid matrix plastic strain epsm mismatch",
        )
        np.testing.assert_allclose(
            extra_res["sigm"],
            extra_unc["sigm"],
            rtol=1e-12,
            atol=1e-12,
            err_msg="Solid matrix flow stress sigm mismatch",
        )
        np.testing.assert_allclose(
            extra_res["dmg"],
            extra_unc["dmg"],
            rtol=1e-12,
            atol=1e-12,
            err_msg="Solid void damage tensor dmg mismatch",
        )
        np.testing.assert_array_equal(
            extra_res["off52"],
            extra_unc["off52"],
            err_msg="Solid off52 mask mismatch",
        )

    def test_constitutive_cycle_restart_continuation_shell(self):
        """Run multi-cycle shell_update_law52 and verify pickled continuation matches uninterrupted within 10^-12."""
        mat = build_law52(
            E=205000.0,
            nu=0.29,
            rho0=7.85e-9,
            A=350.0,
            B=200.0,
            n=0.22,
            q1=1.5,
            q2=1.0,
            q3=2.25,
            f_I=0.01,
            f_N=0.04,
            f_C=0.06,
            f_F=0.15,
        )

        n_steps = 10
        dt = 1.0e-6
        # In-plane strain increments: exx, eyy, exy
        deps_steps = [
            np.array([[0.0025, -0.0008, 0.001]]),
            np.array([[0.0030, -0.0010, 0.0015]]),
            np.array([[0.0020, 0.0015, 0.002]]),
            np.array([[0.0015, 0.0020, 0.0025]]),
            np.array([[0.0010, 0.0010, 0.001]]),
            np.array([[-0.001, -0.001, 0.001]]),
            np.array([[0.0020, 0.0015, 0.0015]]),
            np.array([[0.0025, -0.0005, 0.002]]),
            np.array([[0.0015, -0.0010, 0.001]]),
            np.array([[0.0010, 0.0005, 0.0005]]),
        ]

        # 1. Uninterrupted run
        extra_unc: Dict[str, Any] = {}
        sig_unc = np.zeros((1, 3))
        epsp_unc = np.zeros(1)

        for step in range(n_steps):
            sig_unc, epsp_unc = shell_update_law52(
                mat, sig_unc, deps_steps[step], epsp=epsp_unc, dt=dt, extra=extra_unc
            )

        # 2. Chained run
        extra_chn: Dict[str, Any] = {}
        sig_chn = np.zeros((1, 3))
        epsp_chn = np.zeros(1)

        split = 5
        for step in range(split):
            sig_chn, epsp_chn = shell_update_law52(
                mat, sig_chn, deps_steps[step], epsp=epsp_chn, dt=dt, extra=extra_chn
            )

        checkpoint = pickle.dumps({
            "sig": sig_chn,
            "epsp": epsp_chn,
            "extra": copy.deepcopy(extra_chn),
        })

        restored = pickle.loads(checkpoint)
        sig_res = restored["sig"]
        epsp_res = restored["epsp"]
        extra_res = restored["extra"]

        for step in range(split, n_steps):
            sig_res, epsp_res = shell_update_law52(
                mat, sig_res, deps_steps[step], epsp=epsp_res, dt=dt, extra=extra_res
            )

        np.testing.assert_allclose(
            sig_res,
            sig_unc,
            rtol=1e-12,
            atol=1e-12,
            err_msg="Shell stress tensor mismatch between uninterrupted and restarted continuation",
        )
        np.testing.assert_allclose(
            epsp_res,
            epsp_unc,
            rtol=1e-12,
            atol=1e-12,
            err_msg="Shell plastic strain mismatch between uninterrupted and restarted continuation",
        )
        np.testing.assert_allclose(
            extra_res["epsm"],
            extra_unc["epsm"],
            rtol=1e-12,
            atol=1e-12,
            err_msg="Shell matrix plastic strain epsm mismatch",
        )
        np.testing.assert_allclose(
            extra_res["dmg"],
            extra_unc["dmg"],
            rtol=1e-12,
            atol=1e-12,
            err_msg="Shell void damage tensor dmg mismatch",
        )

    def test_element_kernel_restart_continuation(self, tmp_path: Path):
        """Dynamic step continuation with solid_hexa8.forces across write_restart / read_restart."""
        mat = build_law52(
            E=210000.0,
            nu=0.3,
            rho0=7.85e-9,
            A=400.0,
            B=200.0,
            n=0.2,
            f_I=0.01,
            f_C=0.05,
            f_F=0.15,
        )
        prop = MockProp(pid=1, isolid=1)
        conn = np.array([[0, 1, 2, 3, 4, 5, 6, 7]], dtype=np.int64)

        coords = np.array([
            [0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0], [1.0, 0.0, 1.0], [1.0, 1.0, 1.0], [0.0, 1.0, 1.0],
        ], dtype=float)

        dt = 5.0e-8
        n_cycles = 10
        split_cycle = 5

        # 1. Uninterrupted run
        m_unc = Model()
        m_unc.x0 = coords.copy()
        g_unc = MockGroup(conn, slices=[(slice(0, 1), mat, prop)])
        g_unc._model = m_unc
        solid_hexa8.init_group(g_unc, m_unc, None)

        coords_unc = coords.copy()
        vel_unc = np.zeros_like(coords)
        vr_unc = np.zeros_like(coords)
        # Apply tensile velocity on positive X face
        vel_unc[[1, 2, 5, 6], 0] = 5.0

        fint_unc = np.zeros((8, 3))
        mint_unc = np.zeros((8, 3))

        for c in range(n_cycles):
            fint_unc.fill(0.0)
            mint_unc.fill(0.0)
            solid_hexa8.forces(g_unc, coords_unc, vel_unc, vr_unc, dt, fint_unc, mint_unc)
            coords_unc += vel_unc * dt

        # 2. Chained run: Leg 1 (cycles 0..4)
        m_chn = Model()
        m_chn.x0 = coords.copy()
        g_chn = MockGroup(conn, slices=[(slice(0, 1), mat, prop)])
        g_chn._model = m_chn
        solid_hexa8.init_group(g_chn, m_chn, None)

        coords_chn = coords.copy()
        vel_chn = vel_unc.copy()
        vr_chn = np.zeros_like(coords)
        fint_chn = np.zeros((8, 3))
        mint_chn = np.zeros((8, 3))

        for c in range(split_cycle):
            fint_chn.fill(0.0)
            mint_chn.fill(0.0)
            solid_hexa8.forces(g_chn, coords_chn, vel_chn, vr_chn, dt, fint_chn, mint_chn)
            coords_chn += vel_chn * dt

        # Write restart
        m_chn.x = coords_chn.copy()
        m_chn.v = vel_chn.copy()
        m_chn.bricks = g_chn
        rst_file = tmp_path / "hexa8_restart.rst"
        write_restart(m_chn, str(rst_file), engine={"cycle": split_cycle, "t": split_cycle * dt})

        # Leg 2: restore from restart and continue cycles 5..9
        m_res, eng_res = read_restart(str(rst_file))
        assert eng_res["cycle"] == split_cycle

        g_res = m_res.bricks
        coords_res = m_res.x.copy()
        vel_res = m_res.v.copy()
        vr_res = np.zeros_like(coords)
        fint_res = np.zeros((8, 3))
        mint_res = np.zeros((8, 3))

        for c in range(split_cycle, n_cycles):
            fint_res.fill(0.0)
            mint_res.fill(0.0)
            solid_hexa8.forces(g_res, coords_res, vel_res, vr_res, dt, fint_res, mint_res)
            coords_res += vel_res * dt

        # Assert precise match between resumed and uninterrupted run
        np.testing.assert_allclose(
            coords_res,
            coords_unc,
            rtol=1e-12,
            atol=1e-12,
            err_msg="Final nodal coordinates mismatch after restart continuation",
        )
        np.testing.assert_allclose(
            fint_res,
            fint_unc,
            rtol=1e-12,
            atol=1e-12,
            err_msg="Final internal forces mismatch after restart continuation",
        )
        # Material state arrays: epsm, sigm, dmg, off52
        for name in ("epsm", "sigm", "dmg", "off52"):
            np.testing.assert_allclose(
                g_res.state["mat_extra"][name],
                g_unc.state["mat_extra"][name],
                rtol=1e-12,
                atol=1e-12,
                err_msg=f"Material state array {name} mismatch after restart continuation",
            )

    def test_end_to_end_engine_restart_chaining(self, tmp_path: Path):
        """Full engine restart chaining: run_starter -> run_engine (leg 1) -> run_engine (leg 2) matches uninterrupted run."""
        name_unc = "law52_e2e_unc"
        name_chn = "law52_e2e_chn"

        def make_deck(run_name: str) -> Path:
            d = StarterDeck(run_name)
            d.mat_law52(
                mid=1,
                title="E2EGurson",
                rho=7.85e-9,
                e=210000.0,
                nu=0.3,
                a=400.0,
                b=200.0,
                n=0.2,
                f_i=0.01,
                f_c=0.05,
                f_f=0.15,
            )
            d.prop_solid(pid=1, title="SolidProp", isolid=1)
            d.part(pid=1, title="SolidPart", prop_id=1, mat_id=1)
            d.node([
                (1, 0.0, 0.0, 0.0), (2, 1.0, 0.0, 0.0), (3, 1.0, 1.0, 0.0), (4, 0.0, 1.0, 0.0),
                (5, 0.0, 0.0, 1.0), (6, 1.0, 0.0, 1.0), (7, 1.0, 1.0, 1.0), (8, 0.0, 1.0, 1.0),
            ])
            d.brick(1, [[1, 1, 2, 3, 4, 5, 6, 7, 8]])
            d.grnod_node(1, "Pull", [2, 3, 6, 7])
            d.inivel_tra(1, "Kick", [10.0, 0.0, 0.0], 1)
            f = tmp_path / f"{run_name}_0000.rad"
            d.write(str(f))
            return f

        # Determine stable time step
        probe_file = make_deck("law52_probe")
        run_starter(str(probe_file))
        probe_eng = tmp_path / "law52_probe_0001.rad"
        probe_eng.write_text("/RUN/law52_probe/1\n1.0e-8\n/DT\n0.5 0.0\n/PRINT/-1\n", encoding="utf-8")
        run_engine(str(probe_eng))
        _, eng_probe = read_restart(str(tmp_path / "law52_probe_0001.rst"))
        dt0 = eng_probe["dt"]

        # Run legs aligned on exact cycle intervals
        t1 = 2 * dt0
        t2 = 4 * dt0

        # 1. Uninterrupted execution: 0 -> t2
        unc_rad = make_deck(name_unc)
        run_starter(str(unc_rad))
        unc_eng = tmp_path / f"{name_unc}_0001.rad"
        unc_eng.write_text(f"/RUN/{name_unc}/1\n{t2!r}\n/DT\n0.5 0.0\n/PRINT/-1\n", encoding="utf-8")
        m_unc = run_engine(str(unc_eng))

        # 2. Chained execution: Leg 1 (0 -> t1)
        chn_rad = make_deck(name_chn)
        run_starter(str(chn_rad))
        chn_eng1 = tmp_path / f"{name_chn}_0001.rad"
        chn_eng1.write_text(f"/RUN/{name_chn}/1\n{t1!r}\n/DT\n0.5 0.0\n/PRINT/-1\n", encoding="utf-8")
        m_chn1 = run_engine(str(chn_eng1))
        rst1_path = tmp_path / f"{name_chn}_0001.rst"
        assert rst1_path.exists()

        # Leg 2 (t1 -> t2) resuming from chn_0001.rst
        chn_eng2 = tmp_path / f"{name_chn}_0002.rad"
        chn_eng2.write_text(f"/RUN/{name_chn}/2\n{t2!r}\n/DT\n0.5 0.0\n/PRINT/-1\n", encoding="utf-8")
        m_chn2 = run_engine(str(chn_eng2))
        assert (tmp_path / f"{name_chn}_0002.rst").exists()

        # 3. Verify exact cycle count, nodal positions, and velocities
        assert m_chn2.engine_state.cycle == m_unc.engine_state.cycle
        np.testing.assert_allclose(
            m_chn2.x,
            m_unc.x,
            rtol=1e-10,
            atol=1e-12,
            err_msg="Nodal positions mismatch between restarted continuation and uninterrupted run",
        )
        np.testing.assert_allclose(
            m_chn2.v,
            m_unc.v,
            rtol=1e-10,
            atol=1e-12,
            err_msg="Nodal velocities mismatch between restarted continuation and uninterrupted run",
        )
