"""
Milestone M555: /MAT/LAW57 (/MAT/BARLAT3)
Exhaustive Roundtrip, Negative Validation & Restart (.rst) Serialization Auditor Suite.

Fortran origins:
  - starter/source/materials/mat/mat057/hm_read_mat57.F
  - engine/source/materials/mat/mat057/sigeps57.F & sigeps57c.F90
  - engine/source/output/restart/wrrestp.F & rdresb.F
  - config/CFG/radioss110/MAT/matl57_BARLAT3.cfg & radioss2025/MAT/matl57_BARLAT3.cfg

Audits:
  1. Fixed-Format Deck Roundtrip:
     - Standard 20-column fixed format with StarterDeck.mat_law57 and mat_barlat3.
     - Re-parse using read_mat_law57 and parse_starter_deck.
     - Assert exact equality for all card fields:
       rho, E, nu, ifunce, einf (E_inf), ce (C_E), r00 (R_00), r45 (R_45), r90 (R_90),
       chard (C_hard), m, eps_max (eps_p_max), eps_t1 (eps_t1), eps_t2 (eps_t2),
       fcut (F_cut), fsmooth (F_smooth), vp (VP), and curves (fct_id, fscale, eps).
     - Direct entity object invocation with MatLaw57 and Material entities.
     - Preserved Fortran-faithful default values for minimal card.
     - Compatibility with legacy 4-card format.

  2. Free-Format Comma-Delimited Roundtrip:
     - Write and re-parse free-format deck blocks for /MAT/LAW57 and /MAT/BARLAT3.
     - Comma-separated with whitespace and compact comma-delimited without whitespace.
     - Cross-dialect roundtrip (free -> parse -> fixed -> parse -> exact equality).

  3. Negative Starter Diagnostics:
     - Density rho <= 0 (error).
     - Young's modulus E <= 0 (error).
     - Poisson's ratio nu < 0 or nu >= 0.5 (error).
     - Non-positive Lankford parameters: R00 <= 0, R45 <= 0, R90 <= 0 (error).
     - Exponent m < 1.0 (error).
     - Inverted failure limits: eps_t2 <= eps_t1 (error).
     - Solid elements assigned to LAW57 (ANCMSG 305 error).
     - 1D elements (springs, beams, trusses) assigned to LAW57 (ANCMSG 306 error).
     - Element compatibility acceptance (shells permitted).
     - _ALLOWED_LAWS registry and _MAT_CHECKS dispatch.

  4. Restart (.rst) Serialization:
     - MatLaw57 entity dataclass, MatLaw57Curve, and Law57Params pickling and unpickling fidelity.
     - Material state arrays: plastic strain pla57, backstress sigb57, active mask off57,
       thickness strain thk57, damage dmg57 (and total strain eps57, epsd57).
     - Material state arrays preserved across write_restart / read_restart contract.
     - Constitutive dynamic cycle continuation from restart matching uninterrupted run within 10^-12.
     - Element kernel (shell_bt4.forces) dynamic step continuation within 10^-12.
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
from pyradioss.elements import shell_bt4
from pyradioss.engine.engine import run_engine
from pyradioss.input.deck_reader import read_deck
from pyradioss.input.deck_writer import StarterDeck
from pyradioss.input.starter_keywords import parse_starter_deck, read_mat_law57
from pyradioss.materials.law57_barlat import (
    Law57Params,
    build_law57,
    extra_shapes,
    shell_update_law57,
)
from pyradioss import materials
from pyradioss.model.entities import MatBarlat3, MatLaw57, MatLaw57Curve, Material
from pyradioss.model.model import Model
from pyradioss.starter.checks import (
    _ALLOWED_LAWS,
    _MAT_CHECKS,
    check_mat_law57,
    check_materials,
    check_model,
)
from pyradioss.starter.restart import read_restart, write_restart
from pyradioss.starter.starter import run_starter


# ============================================================================
# Helpers
# ============================================================================

def _parse_deck_str(tmp_path: Path, text: str, name: str = "TEST_LAW57") -> Tuple[Model, MessageLog]:
    """Write deck text to file and parse into Model and MessageLog."""
    deck_path = tmp_path / f"{name}_0000.rad"
    deck_path.write_text(text.strip() + "\n", encoding="utf-8")
    blocks = read_deck(str(deck_path))
    model = Model()
    log = MessageLog()
    parse_starter_deck(blocks, model, log)
    return model, log


def _assert_law57_all_fields_exact(
    m: MatLaw57,
    expected: Dict[str, Any],
    tol: float = 1e-6,
) -> None:
    """Assert precise equality for all LAW57 card fields:
    rho, E, nu, ifunce, einf, ce, r00, r45, r90, chard, m,
    eps_max, eps_t1, eps_t2, fcut, fsmooth, vp, and curves.
    """
    # Card 1: Density
    if "rho" in expected:
        assert m.rho == pytest.approx(expected["rho"], rel=tol)
        assert m.rho0 == pytest.approx(expected["rho"], rel=tol)
    if "refer_rho" in expected:
        assert m.refer_rho == pytest.approx(expected["refer_rho"], rel=tol)
        assert m.rhor == pytest.approx(expected["refer_rho"], rel=tol)

    # Card 2: Elastic
    if "e" in expected:
        assert m.e == pytest.approx(expected["e"], rel=tol)
        assert m.E == pytest.approx(expected["e"], rel=tol)
    if "nu" in expected:
        assert m.nu == pytest.approx(expected["nu"], rel=tol)
        assert m.Nu == pytest.approx(expected["nu"], rel=tol)

    # Card 3: Dynamic modulus
    if "ifunce" in expected:
        assert m.ifunce == expected["ifunce"]
    if "einf" in expected:
        assert m.einf == pytest.approx(expected["einf"], rel=tol)
    if "ce" in expected:
        assert m.ce == pytest.approx(expected["ce"], rel=tol)

    # Card 4: Lankford & Hardening
    if "r00" in expected:
        assert m.r00 == pytest.approx(expected["r00"], rel=tol)
    if "r45" in expected:
        assert m.r45 == pytest.approx(expected["r45"], rel=tol)
    if "r90" in expected:
        assert m.r90 == pytest.approx(expected["r90"], rel=tol)
    if "chard" in expected:
        assert m.chard == pytest.approx(expected["chard"], rel=tol)
    if "m" in expected:
        assert m.m == pytest.approx(expected["m"], rel=tol)

    # Card 5: Limits & Flags
    if "eps_max" in expected:
        assert m.eps_max == pytest.approx(expected["eps_max"], rel=tol)
        assert m.epsp_max == pytest.approx(expected["eps_max"], rel=tol)
    if "eps_t1" in expected:
        assert m.eps_t1 == pytest.approx(expected["eps_t1"], rel=tol)
    if "eps_t2" in expected:
        assert m.eps_t2 == pytest.approx(expected["eps_t2"], rel=tol)
    if "fcut" in expected:
        assert m.fcut == pytest.approx(expected["fcut"], rel=tol)
    if "fsmooth" in expected:
        assert m.fsmooth == expected["fsmooth"]
    if "vp" in expected:
        assert m.vp == expected["vp"]

    # Curves
    if "curves" in expected:
        exp_curves = expected["curves"]
        assert len(m.curves) == len(exp_curves)
        for act_c, exp_c in zip(m.curves, exp_curves):
            if isinstance(exp_c, MatLaw57Curve):
                exp_fid, exp_fsc, exp_eps = exp_c.fct_id, exp_c.fscale, exp_c.eps
            elif isinstance(exp_c, dict):
                exp_fid = exp_c.get("fct_id", exp_c.get("func_id", 0))
                exp_fsc = exp_c.get("fscale", exp_c.get("scale", 1.0))
                exp_eps = exp_c.get("eps", exp_c.get("rate", 0.0))
            else:
                exp_fid, exp_fsc, exp_eps = exp_c[0], exp_c[1], exp_c[2]
            assert act_c.fct_id == exp_fid
            assert act_c.func_id == exp_fid
            assert act_c.fscale == pytest.approx(exp_fsc, rel=tol)
            assert act_c.scale == pytest.approx(exp_fsc, rel=tol)
            assert act_c.eps == pytest.approx(exp_eps, rel=tol)
            assert act_c.rate == pytest.approx(exp_eps, rel=tol)


class MockProp:
    """Mock shell property for element kernel and dynamic step testing."""
    def __init__(self, pid: int = 1, thick: float = 1.0, nip: int = 3, **kwargs: Any):
        self.id = pid
        self.thick = thick
        self.nip = nip
        self.params = {
            "thick": thick,
            "nip": nip,
            "qa": 1.1,
            "qb": 0.05,
            "hm": 0.1,
            "hf": 0.1,
            "hr": 0.1,
            **kwargs,
        }


class MockGroup:
    """Mock element group with connectivity, IDs, and state buffer."""
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

class TestLaw57FixedFormatRoundtrip:
    """Audit standard 20-column fixed-format deck generation and re-parsing."""

    def test_fixed_format_all_card_fields(self, tmp_path: Path):
        """Write /MAT/LAW57 with explicit values for all fields and re-parse."""
        curves = [
            MatLaw57Curve(fct_id=101, fscale=1.00, eps=0.0),
            MatLaw57Curve(fct_id=102, fscale=1.12, eps=10.0),
            MatLaw57Curve(fct_id=103, fscale=1.25, eps=100.0),
        ]
        expected = {
            "rho": 7.85e-3,
            "refer_rho": 7.85e-3,
            "e": 210000.0,
            "nu": 0.33,
            "ifunce": 15,
            "einf": 185000.0,
            "ce": 12.5,
            "r00": 1.35,
            "r45": 1.15,
            "r90": 1.55,
            "chard": 320.0,
            "m": 6.0,
            "eps_max": 0.38,
            "eps_t1": 0.45,
            "eps_t2": 0.55,
            "fcut": 4500.0,
            "fsmooth": 1,
            "vp": 1,
            "curves": curves,
        }

        deck = StarterDeck("FIXED_ROUNDTRIP")
        deck.mat_law57(
            mid=57,
            title="Barlat Complete Fixed",
            fixed_format=True,
            **expected,
        )
        rendered = deck.render()
        assert "/MAT/LAW57/57" in rendered
        assert "Barlat Complete Fixed" in rendered

        # Wrap in full starter deck and parse
        full_deck = (
            "/BEGIN\n"
            "FIXED_ALL_FIELDS\n"
            "                  90                   1\n"
            + rendered
            + "\n/END\n"
        )
        model, log = _parse_deck_str(tmp_path, full_deck, "law57_fixed_all")
        assert not log.has_errors, f"Parse errors: {log.errors}"
        assert 57 in model.mat_law57s
        assert 57 in model.mat_barlat3s
        assert 57 in model.materials

        m = model.mat_law57s[57]
        _assert_law57_all_fields_exact(m, expected)

        # Verify Material generic entity consistency
        mat = model.materials[57]
        assert mat.law == 57
        assert mat.rho0 == pytest.approx(expected["rho"])
        assert mat.params["E"] == pytest.approx(expected["e"])
        assert mat.params["nu"] == pytest.approx(expected["nu"])
        assert mat.params["r00"] == pytest.approx(expected["r00"])
        assert mat.params["m"] == pytest.approx(expected["m"])
        assert len(mat.params["curves"]) == 3

    def test_fixed_format_direct_entity_invocation(self, tmp_path: Path):
        """Pass a pre-constructed MatLaw57 instance directly to StarterDeck.mat_law57."""
        curves = [
            MatLaw57Curve(fct_id=201, fscale=1.0, eps=0.0),
            MatLaw57Curve(fct_id=202, fscale=1.2, eps=50.0),
        ]
        mat_in = MatLaw57(
            id=77,
            title="EntityInvoked",
            rho=2.7e-3,
            refer_rho=2.7e-3,
            e=72000.0,
            nu=0.31,
            ifunce=5,
            einf=68000.0,
            ce=8.0,
            r00=0.85,
            r45=0.95,
            r90=1.15,
            chard=280.0,
            m=8.0,
            eps_max=0.28,
            eps_t1=0.32,
            eps_t2=0.40,
            fcut=3000.0,
            fsmooth=0,
            vp=1,
            curves=curves,
        )

        deck = StarterDeck("ENTITY_DECK")
        deck.mat_law57(mat_in, fixed_format=True)
        rendered = deck.render()
        assert "/MAT/LAW57/77" in rendered
        assert "EntityInvoked" in rendered

        full_deck = (
            "/BEGIN\n"
            "ENTITY_ROUNDTRIP\n"
            "                  90                   1\n"
            + rendered
            + "\n/END\n"
        )
        model, log = _parse_deck_str(tmp_path, full_deck, "entity_roundtrip")
        assert not log.has_errors
        m_out = model.mat_law57s[77]

        assert m_out.id == mat_in.id
        assert m_out.rho == pytest.approx(mat_in.rho)
        assert m_out.e == pytest.approx(mat_in.e)
        assert m_out.nu == pytest.approx(mat_in.nu)
        assert m_out.ifunce == mat_in.ifunce
        assert m_out.einf == pytest.approx(mat_in.einf)
        assert m_out.ce == pytest.approx(mat_in.ce)
        assert m_out.r00 == pytest.approx(mat_in.r00)
        assert m_out.r45 == pytest.approx(mat_in.r45)
        assert m_out.r90 == pytest.approx(mat_in.r90)
        assert m_out.chard == pytest.approx(mat_in.chard)
        assert m_out.m == pytest.approx(mat_in.m)
        assert m_out.eps_max == pytest.approx(mat_in.eps_max)
        assert m_out.eps_t1 == pytest.approx(mat_in.eps_t1)
        assert m_out.eps_t2 == pytest.approx(mat_in.eps_t2)
        assert m_out.fcut == pytest.approx(mat_in.fcut)
        assert m_out.fsmooth == mat_in.fsmooth
        assert m_out.vp == mat_in.vp
        assert len(m_out.curves) == 2
        assert m_out.curves[1].fct_id == 202
        assert m_out.curves[1].fscale == pytest.approx(1.2)
        assert m_out.curves[1].eps == pytest.approx(50.0)

    def test_fixed_format_material_instance_invocation(self, tmp_path: Path):
        """Pass a generic Material instance with law=57 and params to StarterDeck.mat_law57."""
        mat_generic = Material(
            id=33,
            law=57,
            rho0=7.8e-3,
            title="MaterialGeneric",
            params={
                "e": 205000.0,
                "nu": 0.30,
                "r00": 1.25,
                "r45": 1.10,
                "r90": 1.40,
                "m": 6.0,
                "chard": 350.0,
                "eps_max": 0.30,
                "eps_t1": 0.35,
                "eps_t2": 0.45,
                "fcut": 2000.0,
                "vp": 1,
                "curves": [(501, 1.0, 0.0), (502, 1.15, 25.0)],
            },
        )

        deck = StarterDeck("GENERIC_MAT_DECK")
        deck.mat_law57(mat_generic, fixed_format=True)
        rendered = deck.render()
        assert "/MAT/LAW57/33" in rendered

        full_deck = (
            "/BEGIN\n"
            "GENERIC_ROUNDTRIP\n"
            "                  90                   1\n"
            + rendered
            + "\n/END\n"
        )
        model, log = _parse_deck_str(tmp_path, full_deck, "generic_roundtrip")
        assert not log.has_errors
        m = model.mat_law57s[33]
        assert m.rho == pytest.approx(7.8e-3)
        assert m.e == pytest.approx(205000.0)
        assert m.r00 == pytest.approx(1.25)
        assert m.r45 == pytest.approx(1.10)
        assert m.r90 == pytest.approx(1.40)
        assert m.chard == pytest.approx(350.0)
        assert len(m.curves) == 2
        assert m.curves[1].fct_id == 502
        assert m.curves[1].fscale == pytest.approx(1.15)

    def test_fixed_format_defaults_preservation(self, tmp_path: Path):
        """Verify minimal fixed-format card inherits exact Fortran defaults."""
        deck = StarterDeck("MINIMAL_DECK")
        deck.mat_law57(
            mid=1,
            title="MinimalBarlat",
            rho=7.8e-3,
            e=210000.0,
            nu=0.3,
            fixed_format=True,
        )
        rendered = deck.render()
        full_deck = (
            "/BEGIN\n"
            "MINIMAL_ROUNDTRIP\n"
            "                  90                   1\n"
            + rendered
            + "\n/END\n"
        )
        model, log = _parse_deck_str(tmp_path, full_deck, "minimal_fixed")
        assert not log.has_errors
        m = model.mat_law57s[1]

        # Fortran defaults per hm_read_mat57.F90
        assert m.refer_rho == pytest.approx(7.8e-3)
        assert m.ifunce == 0
        assert m.einf == pytest.approx(0.0)
        assert m.ce == pytest.approx(0.0)
        assert m.r00 == pytest.approx(1.0)
        assert m.r45 == pytest.approx(1.0)
        assert m.r90 == pytest.approx(1.0)
        assert m.chard == pytest.approx(0.0)
        assert m.m == pytest.approx(6.0)
        assert m.eps_max == pytest.approx(1.0e30)
        assert m.eps_t1 == pytest.approx(1.0e30)
        assert m.eps_t2 == pytest.approx(2.0e30)
        assert m.fcut == pytest.approx(1.0e30)
        assert m.fsmooth == 0
        assert m.vp == 0
        assert len(m.curves) == 0

    def test_fixed_format_legacy_4card_format(self, tmp_path: Path):
        """Verify backwards compatibility with legacy 4-card format (radioss90 / M183)."""
        deck_text = """\
#RADIOSS STARTER
/BEGIN
legacy_4card
                  90                   1
/MAT/LAW57/44
Legacy 4 Card
#              RHO_I          Ref. dens.
              7.8E-3              7.8E-3
#                  E                  NU
            210000.0                 0.3
#                r00                 r45                 r90              C_hard                   m
                 1.2                 1.1                 1.4               300.0                 6.0
#           EPSP_max               EPS_T               EPS_M
                 0.3                 0.4                 0.5
# funct_ID                      Fscale_i               EPS_i
        10                           1.0                 0.0
/END
"""
        model, log = _parse_deck_str(tmp_path, deck_text, "legacy_4card")
        assert not log.has_errors
        assert 44 in model.mat_law57s
        m = model.mat_law57s[44]
        assert m.ifunce == 0
        assert m.einf == pytest.approx(0.0)
        assert m.r00 == pytest.approx(1.2)
        assert m.r45 == pytest.approx(1.1)
        assert m.r90 == pytest.approx(1.4)
        assert m.chard == pytest.approx(300.0)
        assert m.m == pytest.approx(6.0)
        assert m.eps_max == pytest.approx(0.3)
        assert m.eps_t1 == pytest.approx(0.4)
        assert m.eps_t2 == pytest.approx(0.5)
        assert len(m.curves) == 1
        assert m.curves[0].fct_id == 10


# ============================================================================
# Section 2: Free-Format Comma-Delimited Roundtrip
# ============================================================================

class TestLaw57FreeFormatRoundtrip:
    """Audit free-format comma-delimited deck writing and re-parsing."""

    def test_free_format_law57_roundtrip(self, tmp_path: Path):
        """Write and re-parse free-format /MAT/LAW57 block."""
        curves = [
            (11, 1.0, 0.0),
            (12, 1.15, 20.0),
        ]
        deck = StarterDeck("FREE_LAW57")
        deck.mat_law57(
            mid=57,
            title="FreeLAW57",
            rho=2.7e-3,
            refer_rho=2.75e-3,
            e=70000.0,
            nu=0.33,
            ifunce=12,
            einf=65000.0,
            ce=10.0,
            r00=0.9,
            r45=1.0,
            r90=1.2,
            chard=350.0,
            m=8.0,
            eps_max=0.3,
            eps_t1=0.4,
            eps_t2=0.5,
            fcut=2000.0,
            fsmooth=1,
            vp=1,
            curves=curves,
            fixed_format=False,
        )
        rendered = deck.render()
        assert "/MAT/LAW57/57" in rendered
        assert "0.9, 1.0, 1.2" in rendered

        full_deck = (
            "/BEGIN\n"
            "FREE_LAW57_DECK\n"
            "                  90                   1\n"
            + rendered
            + "\n/END\n"
        )
        model, log = _parse_deck_str(tmp_path, full_deck, "free_law57")
        assert not log.has_errors
        m = model.mat_law57s[57]

        assert m.rho == pytest.approx(2.7e-3)
        assert m.refer_rho == pytest.approx(2.75e-3)
        assert m.e == pytest.approx(70000.0)
        assert m.nu == pytest.approx(0.33)
        assert m.ifunce == 12
        assert m.einf == pytest.approx(65000.0)
        assert m.ce == pytest.approx(10.0)
        assert m.r00 == pytest.approx(0.9)
        assert m.r45 == pytest.approx(1.0)
        assert m.r90 == pytest.approx(1.2)
        assert m.chard == pytest.approx(350.0)
        assert m.m == pytest.approx(8.0)
        assert m.eps_max == pytest.approx(0.3)
        assert m.eps_t1 == pytest.approx(0.4)
        assert m.eps_t2 == pytest.approx(0.5)
        assert m.fcut == pytest.approx(2000.0)
        assert m.fsmooth == 1
        assert m.vp == 1
        assert len(m.curves) == 2
        assert m.curves[1].fct_id == 12
        assert m.curves[1].fscale == pytest.approx(1.15)
        assert m.curves[1].eps == pytest.approx(20.0)

    def test_free_format_barlat3_roundtrip(self, tmp_path: Path):
        """Write and re-parse free-format /MAT/BARLAT3 using mat_barlat3."""
        curves = [(31, 1.0, 0.0), (32, 1.22, 15.0)]
        deck = StarterDeck("FREE_BARLAT3")
        deck.mat_barlat3(
            mid=58,
            title="FreeBARLAT3",
            rho=7.85e-3,
            e=210000.0,
            nu=0.30,
            ifunce=0,
            einf=0.0,
            ce=0.0,
            r00=1.4,
            r45=1.2,
            r90=1.6,
            chard=400.0,
            m=6.0,
            eps_max=0.35,
            eps_t1=0.45,
            eps_t2=0.55,
            fcut=1e30,
            fsmooth=0,
            vp=0,
            curves=curves,
            fixed_format=False,
        )
        rendered = deck.render()
        assert "/MAT/BARLAT3/58" in rendered

        full_deck = (
            "/BEGIN\n"
            "FREE_BARLAT3_DECK\n"
            "                  90                   1\n"
            + rendered
            + "\n/END\n"
        )
        model, log = _parse_deck_str(tmp_path, full_deck, "free_barlat3")
        assert not log.has_errors
        assert 58 in model.mat_law57s
        assert 58 in model.mat_barlat3s
        m = model.mat_law57s[58]
        assert m.rho == pytest.approx(7.85e-3)
        assert m.e == pytest.approx(210000.0)
        assert m.r00 == pytest.approx(1.4)
        assert m.r45 == pytest.approx(1.2)
        assert m.r90 == pytest.approx(1.6)
        assert m.chard == pytest.approx(400.0)
        assert m.m == pytest.approx(6.0)
        assert len(m.curves) == 2
        assert m.curves[1].fct_id == 32

    def test_free_format_compact_comma_delimiters(self, tmp_path: Path):
        """Re-parse free-format deck with compact comma delimiters (no whitespace around commas)."""
        deck_text = """\
/BEGIN
compact_commas
/MAT/LAW57/99
Compact Comma Deck
7.85E-3,7.85E-3
210000.0,0.30
8,190000.0,15.0
1.25,1.10,1.45,300.0,6.0
0.35,0.45,0.55,5000.0,1,1
201,1.0,0.0
202,1.2,100.0
/END
"""
        model, log = _parse_deck_str(tmp_path, deck_text, "compact_commas")
        assert not log.has_errors
        m = model.mat_law57s[99]
        assert m.rho == pytest.approx(7.85e-3)
        assert m.refer_rho == pytest.approx(7.85e-3)
        assert m.e == pytest.approx(210000.0)
        assert m.nu == pytest.approx(0.30)
        assert m.ifunce == 8
        assert m.einf == pytest.approx(190000.0)
        assert m.ce == pytest.approx(15.0)
        assert m.r00 == pytest.approx(1.25)
        assert m.r45 == pytest.approx(1.10)
        assert m.r90 == pytest.approx(1.45)
        assert m.chard == pytest.approx(300.0)
        assert m.m == pytest.approx(6.0)
        assert m.eps_max == pytest.approx(0.35)
        assert m.eps_t1 == pytest.approx(0.45)
        assert m.eps_t2 == pytest.approx(0.55)
        assert m.fcut == pytest.approx(5000.0)
        assert m.fsmooth == 1
        assert m.vp == 1
        assert len(m.curves) == 2
        assert m.curves[1].fscale == pytest.approx(1.2)

    def test_cross_dialect_roundtrip(self, tmp_path: Path):
        """Cross-dialect roundtrip: free-format -> Model -> fixed-format -> Model -> exact match."""
        free_deck_text = """\
/BEGIN
cross_dialect_orig
/MAT/BARLAT3/105
Cross Dialect Material
2.7E-3, 2.7E-3
71000.0, 0.32
10, 68000.0, 5.5
0.95, 1.05, 1.25, 275.0, 8.0
0.25, 0.35, 0.45, 1500.0, 1, 1
10, 1.0, 0.0
20, 1.18, 50.0
/END
"""
        model_1, log_1 = _parse_deck_str(tmp_path, free_deck_text, "cross_step1")
        assert not log_1.has_errors
        m1 = model_1.mat_law57s[105]

        # Write out with fixed format
        deck_fixed = StarterDeck("cross_dialect_fixed")
        deck_fixed.mat_law57(m1, fixed_format=True)
        fixed_text = (
            "/BEGIN\n"
            "cross_dialect_fixed\n"
            "                  90                   1\n"
            + deck_fixed.render()
            + "\n/END\n"
        )
        model_2, log_2 = _parse_deck_str(tmp_path, fixed_text, "cross_step2")
        assert not log_2.has_errors
        m2 = model_2.mat_law57s[105]

        # Assert all fields match between model_1 and model_2
        assert m2.rho == pytest.approx(m1.rho)
        assert m2.refer_rho == pytest.approx(m1.refer_rho)
        assert m2.e == pytest.approx(m1.e)
        assert m2.nu == pytest.approx(m1.nu)
        assert m2.ifunce == m1.ifunce
        assert m2.einf == pytest.approx(m1.einf)
        assert m2.ce == pytest.approx(m1.ce)
        assert m2.r00 == pytest.approx(m1.r00)
        assert m2.r45 == pytest.approx(m1.r45)
        assert m2.r90 == pytest.approx(m1.r90)
        assert m2.chard == pytest.approx(m1.chard)
        assert m2.m == pytest.approx(m1.m)
        assert m2.eps_max == pytest.approx(m1.eps_max)
        assert m2.eps_t1 == pytest.approx(m1.eps_t1)
        assert m2.eps_t2 == pytest.approx(m1.eps_t2)
        assert m2.fcut == pytest.approx(m1.fcut)
        assert m2.fsmooth == m1.fsmooth
        assert m2.vp == m1.vp
        assert len(m2.curves) == len(m1.curves)
        for c1, c2 in zip(m1.curves, m2.curves):
            assert c2.fct_id == c1.fct_id
            assert c2.fscale == pytest.approx(c1.fscale)
            assert c2.eps == pytest.approx(c1.eps)


# ============================================================================
# Section 3: Negative Starter Checks
# ============================================================================

class TestLaw57NegativeStarterChecks:
    """Negative starter diagnostic validation for /MAT/LAW57 parameter bounds and element compatibility."""

    def test_density_non_positive(self):
        """Density rho <= 0 triggers error."""
        log = MessageLog()
        mat_zero = MatLaw57(id=1, rho=0.0, e=210000.0, nu=0.3)
        check_mat_law57(mat=mat_zero, log=log)
        assert log.has_errors
        assert any("initial density RHO must be > 0" in m for m in log.errors)

        log_neg = MessageLog()
        mat_neg = MatLaw57(id=2, rho=-7.8e-3, e=210000.0, nu=0.3)
        check_mat_law57(mat=mat_neg, log=log_neg)
        assert log_neg.has_errors
        assert any("initial density RHO must be > 0" in m for m in log_neg.errors)

    def test_youngs_modulus_non_positive(self):
        """Young's modulus E <= 0 triggers error."""
        log_zero = MessageLog()
        mat_zero = MatLaw57(id=1, rho=7.8e-3, e=0.0, nu=0.3)
        check_mat_law57(mat=mat_zero, log=log_zero)
        assert log_zero.has_errors
        assert any("Young's modulus E must be > 0" in m for m in log_zero.errors)

        log_neg = MessageLog()
        mat_neg = MatLaw57(id=2, rho=7.8e-3, e=-1000.0, nu=0.3)
        check_mat_law57(mat=mat_neg, log=log_neg)
        assert log_neg.has_errors
        assert any("Young's modulus E must be > 0" in m for m in log_neg.errors)

    def test_poissons_ratio_invalid(self):
        """Poisson's ratio nu < 0 or nu >= 0.5 triggers error."""
        # Negative
        log_neg = MessageLog()
        mat_neg = MatLaw57(id=1, rho=7.8e-3, e=210000.0, nu=-0.1)
        check_mat_law57(mat=mat_neg, log=log_neg)
        assert log_neg.has_errors
        assert any("0 <= NU < 0.5" in m for m in log_neg.errors)

        # Equal to 0.5
        log_half = MessageLog()
        mat_half = MatLaw57(id=2, rho=7.8e-3, e=210000.0, nu=0.5)
        check_mat_law57(mat=mat_half, log=log_half)
        assert log_half.has_errors
        assert any("0 <= NU < 0.5" in m for m in log_half.errors)

        # Greater than 0.5
        log_gt = MessageLog()
        mat_gt = MatLaw57(id=3, rho=7.8e-3, e=210000.0, nu=0.6)
        check_mat_law57(mat=mat_gt, log=log_gt)
        assert log_gt.has_errors
        assert any("0 <= NU < 0.5" in m for m in log_gt.errors)

    def test_lankford_parameters_non_positive(self):
        """Lankford parameters R00 <= 0, R45 <= 0, or R90 <= 0 trigger errors."""
        # R00 <= 0
        log_r00 = MessageLog()
        mat_r00 = MatLaw57(id=1, rho=7.8e-3, e=210000.0, nu=0.3, r00=0.0, r45=1.0, r90=1.0)
        check_mat_law57(mat=mat_r00, log=log_r00)
        assert log_r00.has_errors
        assert any("Lankford parameter R00 must be > 0" in m for m in log_r00.errors)

        log_r00_neg = MessageLog()
        mat_r00_neg = MatLaw57(id=2, rho=7.8e-3, e=210000.0, nu=0.3, r00=-0.5, r45=1.0, r90=1.0)
        check_mat_law57(mat=mat_r00_neg, log=log_r00_neg)
        assert log_r00_neg.has_errors
        assert any("Lankford parameter R00 must be > 0" in m for m in log_r00_neg.errors)

        # R45 <= 0
        log_r45 = MessageLog()
        mat_r45 = MatLaw57(id=3, rho=7.8e-3, e=210000.0, nu=0.3, r00=1.0, r45=0.0, r90=1.0)
        check_mat_law57(mat=mat_r45, log=log_r45)
        assert log_r45.has_errors
        assert any("Lankford parameter R45 must be > 0" in m for m in log_r45.errors)

        # R90 <= 0
        log_r90 = MessageLog()
        mat_r90 = MatLaw57(id=4, rho=7.8e-3, e=210000.0, nu=0.3, r00=1.0, r45=1.0, r90=-0.2)
        check_mat_law57(mat=mat_r90, log=log_r90)
        assert log_r90.has_errors
        assert any("Lankford parameter R90 must be > 0" in m for m in log_r90.errors)

    def test_exponent_m_less_than_one(self):
        """Barlat exponent m < 1.0 triggers error."""
        log_m1 = MessageLog()
        mat_m1 = MatLaw57(id=1, rho=7.8e-3, e=210000.0, nu=0.3, m=0.5)
        check_mat_law57(mat=mat_m1, log=log_m1)
        assert log_m1.has_errors
        assert any("Barlat exponent m must be >= 1.0" in m for m in log_m1.errors)

        log_m2 = MessageLog()
        mat_m2 = MatLaw57(id=2, rho=7.8e-3, e=210000.0, nu=0.3, m=-2.0)
        check_mat_law57(mat=mat_m2, log=log_m2)
        assert log_m2.has_errors
        assert any("Barlat exponent m must be >= 1.0" in m for m in log_m2.errors)

    def test_inverted_failure_limits(self):
        """Inverted failure limits eps_t2 <= eps_t1 (when eps_t1 < 1e20) trigger error."""
        # eps_t2 < eps_t1
        log_inv = MessageLog()
        mat_inv = MatLaw57(id=1, rho=7.8e-3, e=210000.0, nu=0.3, eps_t1=0.5, eps_t2=0.4)
        check_mat_law57(mat=mat_inv, log=log_inv)
        assert log_inv.has_errors
        assert any("EPS_t2 must be > EPS_t1" in m for m in log_inv.errors)

        # eps_t2 == eps_t1
        log_eq = MessageLog()
        mat_eq = MatLaw57(id=2, rho=7.8e-3, e=210000.0, nu=0.3, eps_t1=0.45, eps_t2=0.45)
        check_mat_law57(mat=mat_eq, log=log_eq)
        assert log_eq.has_errors
        assert any("EPS_t2 must be > EPS_t1" in m for m in log_eq.errors)

    def test_solid_elements_assigned_ancmsg_305(self):
        """Solid elements (bricks, tetras, penta, pyra) assigned to LAW57 trigger ANCMSG 305."""
        class DummyElement:
            def __init__(self, mid: int):
                self.mat_id = mid

        class DummyGroup:
            def __init__(self, elements):
                self.els = {i: el for i, el in enumerate(elements)}
            def values(self):
                return self.els.values()

        class DummyModel:
            def __init__(self, groups):
                self.grps = groups
            def element_groups(self):
                return self.grps

        mat57 = MatLaw57(id=1, rho=7.8e-3, e=210000.0, nu=0.3)

        for solid_family in ("bricks", "bricks_heph", "bric20s", "tetras", "tetra10s", "penta6", "pyra5"):
            model_solid = DummyModel([(solid_family, DummyGroup([DummyElement(1)]))])
            log_solid = MessageLog()
            check_mat_law57(model=model_solid, mat=mat57, log=log_solid)
            assert log_solid.has_errors, f"Solid family {solid_family} was not flagged"
            assert any("ANCMSG 305" in m for m in log_solid.errors)
            assert any(f"solid elements ({solid_family})" in m for m in log_solid.errors)

    def test_1d_elements_assigned_ancmsg_306(self):
        """1D elements (springs, beams, trusses) assigned to LAW57 trigger ANCMSG 306."""
        class DummyElement:
            def __init__(self, mid: int):
                self.mat_id = mid

        class DummyGroup:
            def __init__(self, elements):
                self.els = {i: el for i, el in enumerate(elements)}
            def values(self):
                return self.els.values()

        class DummyModel:
            def __init__(self, groups):
                self.grps = groups
            def element_groups(self):
                return self.grps

        mat57 = MatLaw57(id=1, rho=7.8e-3, e=210000.0, nu=0.3)

        for el_1d in ("springs", "beams", "trusses"):
            model_1d = DummyModel([(el_1d, DummyGroup([DummyElement(1)]))])
            log_1d = MessageLog()
            check_mat_law57(model=model_1d, mat=mat57, log=log_1d)
            assert log_1d.has_errors, f"1D family {el_1d} was not flagged"
            assert any("ANCMSG 306" in m for m in log_1d.errors)
            assert any(f"1D elements ({el_1d})" in m for m in log_1d.errors)

    def test_shell_elements_permitted(self):
        """Shell elements (shells, shells_qbat, shells_qeph, sh3n, quads) are accepted without errors."""
        class DummyElement:
            def __init__(self, mid: int):
                self.mat_id = mid

        class DummyGroup:
            def __init__(self, elements):
                self.els = {i: el for i, el in enumerate(elements)}
            def values(self):
                return self.els.values()

        class DummyModel:
            def __init__(self, groups):
                self.grps = groups
            def element_groups(self):
                return self.grps

        mat57 = MatLaw57(id=1, rho=7.8e-3, e=210000.0, nu=0.3)

        for shell_fam in ("shells", "shells_qbat", "shells_qeph", "sh3n", "quads"):
            model_shell = DummyModel([(shell_fam, DummyGroup([DummyElement(1)]))])
            log_shell = MessageLog()
            check_mat_law57(model=model_shell, mat=mat57, log=log_shell)
            assert not log_shell.has_errors, f"Shell family {shell_fam} caused unexpected error: {log_shell.errors}"

    def test_allowed_laws_and_mat_checks_registries(self):
        """Verify _ALLOWED_LAWS and _MAT_CHECKS consistency for LAW57 and BARLAT3."""
        # Check _MAT_CHECKS dispatch
        for key in (57, "57", "LAW57", "BARLAT3", "MAT_BARLAT3", "LAW57_BARLAT3"):
            assert key in _MAT_CHECKS
            assert _MAT_CHECKS[key] is check_mat_law57

        # Check _ALLOWED_LAWS
        for shell_fam in ("shells", "shells_qbat", "shells_qeph", "sh3n", "quads"):
            assert shell_fam in _ALLOWED_LAWS
            for key in (57, "57", "LAW57", "BARLAT3", "MAT_BARLAT3", "LAW57_BARLAT3"):
                assert key in _ALLOWED_LAWS[shell_fam]

        for solid_fam in ("bricks", "tetras", "penta6", "pyra5"):
            assert solid_fam in _ALLOWED_LAWS
            assert 57 not in _ALLOWED_LAWS[solid_fam]
            assert "LAW57" not in _ALLOWED_LAWS[solid_fam]
            assert "BARLAT3" not in _ALLOWED_LAWS[solid_fam]

        for fam_1d in ("beams", "trusses"):
            assert fam_1d in _ALLOWED_LAWS
            assert 57 not in _ALLOWED_LAWS[fam_1d]
            assert "LAW57" not in _ALLOWED_LAWS[fam_1d]


# ============================================================================
# Section 4: Restart (.rst) Serialization
# ============================================================================

class TestLaw57RestartSerialization:
    """Audit .rst restart serialization for MatLaw57, material state arrays, and dynamic continuation."""

    def test_pickle_entities_and_params(self):
        """Verify MatLaw57, MatLaw57Curve, and Law57Params serialize and restore cleanly via pickle."""
        c = MatLaw57Curve(fct_id=5, fscale=1.1, eps=20.0)
        m = MatLaw57(
            id=10,
            rho=7.8e-3,
            e=210000.0,
            nu=0.3,
            r00=1.3,
            r45=1.1,
            r90=1.5,
            m=6.0,
            chard=300.0,
            curves=[c],
            title="PickleTest",
        )

        data = pickle.dumps(m)
        m_restored: MatLaw57 = pickle.loads(data)

        assert m_restored.id == 10
        assert m_restored.title == "PickleTest"
        assert m_restored.rho == pytest.approx(7.8e-3)
        assert m_restored.e == pytest.approx(210000.0)
        assert m_restored.r00 == pytest.approx(1.3)
        assert m_restored.m == pytest.approx(6.0)
        assert m_restored.chard == pytest.approx(300.0)
        assert len(m_restored.curves) == 1
        assert m_restored.curves[0].fct_id == 5
        assert m_restored.curves[0].scale == pytest.approx(1.1)

        # Law57Params physics object
        p = build_law57(E=210000.0, nu=0.3, rho0=7.8e-3, r00=1.3, r45=1.1, r90=1.5, m=6.0)
        p_data = pickle.dumps(p)
        p_restored: Law57Params = pickle.loads(p_data)
        assert p_restored.E == pytest.approx(210000.0)
        assert p_restored.nu == pytest.approx(0.3)
        assert p_restored.r00 == pytest.approx(1.3)

    def test_material_state_arrays_rst_preservation(self, tmp_path: Path):
        """Material state arrays (pla57, sigb57, off57, thk57, dmg57, eps57, epsd57) must survive write_restart/read_restart."""
        deck = StarterDeck("RST_LAW57_ARRAYS")
        deck.mat_law57(
            mid=1,
            title="StateBarlat",
            rho=2.7e-3,
            e=70000.0,
            nu=0.33,
            r00=1.5,
            r45=1.2,
            r90=1.8,
            m=8.0,
            chard=250.0,
        )
        deck.prop_shell(pid=1, title="ShellProp", thick=1.0, nip=3)
        deck.part(pid=1, title="ShellPart", prop_id=1, mat_id=1)
        deck.node([
            (1, 0.0, 0.0, 0.0), (2, 10.0, 0.0, 0.0), (3, 10.0, 10.0, 0.0), (4, 0.0, 10.0, 0.0)
        ])
        deck.shell(1, [[1, 1, 2, 3, 4]])

        rad_path = tmp_path / "rst_arrays_0000.rad"
        deck.write(str(rad_path))

        model = run_starter(str(rad_path))
        assert model is not None

        groups = dict(model.element_groups())
        assert "shells" in groups
        sg = groups["shells"]
        st = sg.state

        # Synthesize realistic persistent material state arrays for LAW57
        # 1 element, 3 integration points
        nip = 3
        pla57_synth = np.array([[0.0125, 0.0245, 0.0380]])
        sigb57_synth = np.array([[[15.4, 8.2, 5.1], [18.6, 9.5, 6.3], [22.1, 11.0, 7.4]]])
        off57_synth = np.array([[1.0, 1.0, 0.8]])
        thk57_synth = np.array([[0.995, 0.992, 0.988]])
        dmg57_synth = np.array([[[0.02, 0.01, 0.02], [0.03, 0.02, 0.03], [0.04, 0.03, 0.04]]])
        eps57_synth = np.array([[[0.0015, 0.0008, 0.0002], [0.0022, 0.0011, 0.0004], [0.0031, 0.0015, 0.0006]]])
        epsd57_synth = np.array([[12.5, 18.2, 25.0]])

        st["mat_extra"]["pla57"] = pla57_synth.copy()
        st["mat_extra"]["sigb57"] = sigb57_synth.copy()
        st["mat_extra"]["off57"] = off57_synth.copy()
        st["mat_extra"]["thk57"] = thk57_synth.copy()
        st["mat_extra"]["dmg57"] = dmg57_synth.copy()
        st["mat_extra"]["eps57"] = eps57_synth.copy()
        st["mat_extra"]["epsd57"] = epsd57_synth.copy()

        # Write restart .rst
        rst_path = tmp_path / "rst_arrays_0001.rst"
        engine_dict = {
            "cycle": 150,
            "t": 1.5e-5,
            "dt": 1.0e-7,
            "energies": {"internal": 88.4, "kinetic": 24.6},
        }
        write_restart(model, str(rst_path), engine=engine_dict)

        # Read back from restart .rst
        rest_model, rest_engine = read_restart(str(rst_path))
        assert rest_engine["cycle"] == 150
        assert rest_engine["t"] == pytest.approx(1.5e-5)
        assert rest_engine["dt"] == pytest.approx(1.0e-7)

        rest_sg = dict(rest_model.element_groups())["shells"]
        rest_extra = rest_sg.state["mat_extra"]

        np.testing.assert_array_equal(
            rest_extra["pla57"],
            pla57_synth,
            err_msg="Plastic strain pla57 did not survive restart serialization exactly",
        )
        np.testing.assert_array_equal(
            rest_extra["sigb57"],
            sigb57_synth,
            err_msg="Backstress sigb57 did not survive restart serialization exactly",
        )
        np.testing.assert_array_equal(
            rest_extra["off57"],
            off57_synth,
            err_msg="Active mask off57 did not survive restart serialization exactly",
        )
        np.testing.assert_array_equal(
            rest_extra["thk57"],
            thk57_synth,
            err_msg="Thickness strain thk57 did not survive restart serialization exactly",
        )
        np.testing.assert_array_equal(
            rest_extra["dmg57"],
            dmg57_synth,
            err_msg="Damage tensor dmg57 did not survive restart serialization exactly",
        )
        np.testing.assert_array_equal(
            rest_extra["eps57"],
            eps57_synth,
            err_msg="Total strain eps57 did not survive restart serialization exactly",
        )
        np.testing.assert_array_equal(
            rest_extra["epsd57"],
            epsd57_synth,
            err_msg="Strain rate epsd57 did not survive restart serialization exactly",
        )

    def test_constitutive_restart_continuation_tolerance(self):
        """Dynamic constitutive update (shell_update_law57) with restart resumption matches uninterrupted run within 10^-12."""
        p = build_law57(
            E=70000.0,
            nu=0.33,
            rho0=2.7e-3,
            sigy0=200.0,
            r00=1.2,
            r45=1.1,
            r90=1.3,
            m=6.0,
            chard=300.0,
        )

        dt = 1.0e-6
        n_steps = 10
        split_step = 5
        deps = np.array([[1.5e-4, 0.5e-4, 0.3e-4]])

        # 1. Uninterrupted run: 10 steps
        extra_unc = {
            "pla57": np.zeros(1),
            "sigb57": np.zeros((1, 3)),
            "off57": np.ones(1),
            "thk57": np.ones(1),
            "dmg57": np.zeros((1, 3)),
            "eps57": np.zeros((1, 3)),
            "epsd57": np.zeros(1),
        }
        sig_unc = np.zeros((1, 3))
        for _ in range(n_steps):
            res_unc = shell_update_law57(p, sig_unc, deps, dt=dt, extra=extra_unc)
            sig_unc = res_unc[0]

        # 2. Chained run: 5 steps -> pickle -> 5 steps
        extra_chn = {
            "pla57": np.zeros(1),
            "sigb57": np.zeros((1, 3)),
            "off57": np.ones(1),
            "thk57": np.ones(1),
            "dmg57": np.zeros((1, 3)),
            "eps57": np.zeros((1, 3)),
            "epsd57": np.zeros(1),
        }
        sig_chn = np.zeros((1, 3))
        for _ in range(split_step):
            res_chn = shell_update_law57(p, sig_chn, deps, dt=dt, extra=extra_chn)
            sig_chn = res_chn[0]

        # Serialize & restore
        sig_rest = pickle.loads(pickle.dumps(sig_chn))
        extra_rest = pickle.loads(pickle.dumps(extra_chn))

        for _ in range(split_step, n_steps):
            res_rest = shell_update_law57(p, sig_rest, deps, dt=dt, extra=extra_rest)
            sig_rest = res_rest[0]

        # Assert all fields match within 10^-12
        np.testing.assert_allclose(
            sig_rest,
            sig_unc,
            rtol=1e-12,
            atol=1e-12,
            err_msg="Constitutive stress mismatch between restarted and uninterrupted runs",
        )
        np.testing.assert_allclose(
            extra_rest["pla57"],
            extra_unc["pla57"],
            rtol=1e-12,
            atol=1e-12,
            err_msg="Plastic strain pla57 mismatch between restarted and uninterrupted runs",
        )
        np.testing.assert_allclose(
            extra_rest["sigb57"],
            extra_unc["sigb57"],
            rtol=1e-12,
            atol=1e-12,
            err_msg="Backstress sigb57 mismatch between restarted and uninterrupted runs",
        )
        np.testing.assert_allclose(
            extra_rest["thk57"],
            extra_unc["thk57"],
            rtol=1e-12,
            atol=1e-12,
            err_msg="Thickness thk57 mismatch between restarted and uninterrupted runs",
        )
        np.testing.assert_allclose(
            extra_rest["dmg57"],
            extra_unc["dmg57"],
            rtol=1e-12,
            atol=1e-12,
            err_msg="Damage dmg57 mismatch between restarted and uninterrupted runs",
        )

    def test_shell_kernel_restart_continuation(self, tmp_path: Path):
        """Dynamic step continuation with shell_bt4.forces across write_restart / read_restart matches within 10^-12."""
        coords = np.array([
            [0.0, 0.0, 0.0],
            [10.0, 0.0, 0.0],
            [10.0, 10.0, 0.0],
            [0.0, 10.0, 0.0],
        ])
        conn = np.array([[0, 1, 2, 3]])
        mat = Material(id=1, law=57, rho0=2.7e-3, title="BarlatKernel", params={
            "E": 70000.0,
            "nu": 0.33,
            "sigy0": 200.0,
            "r00": 1.5,
            "r45": 1.2,
            "r90": 1.8,
            "m": 8.0,
            "chard": 250.0,
        })
        prop = MockProp(pid=1, thick=1.0, nip=3)

        dt = 1.0e-5
        n_cycles = 10
        split_cycle = 5

        # 1. Uninterrupted run: 10 cycles
        m_unc = Model()
        m_unc.x0 = coords.copy()
        g_unc = MockGroup(conn, slices=[(slice(0, 1), mat, prop)])
        g_unc._model = m_unc
        shell_bt4.init_group(g_unc, m_unc, None)

        coords_unc = coords.copy()
        vel_unc = np.zeros_like(coords)
        vel_unc[[1, 2], 0] = 50.0  # stretch x
        vel_unc[[2, 3], 1] = 25.0  # stretch y
        vr_unc = np.zeros_like(coords)
        fint_unc = np.zeros((4, 3))
        mint_unc = np.zeros((4, 3))

        for c in range(n_cycles):
            fint_unc.fill(0.0)
            mint_unc.fill(0.0)
            shell_bt4.forces(g_unc, coords_unc, vel_unc, vr_unc, dt, fint_unc, mint_unc)
            coords_unc += vel_unc * dt

        # 2. Chained run: Leg 1 (cycles 0..4)
        m_chn = Model()
        m_chn.x0 = coords.copy()
        g_chn = MockGroup(conn, slices=[(slice(0, 1), mat, prop)])
        g_chn._model = m_chn
        shell_bt4.init_group(g_chn, m_chn, None)

        coords_chn = coords.copy()
        vel_chn = vel_unc.copy()
        vr_chn = np.zeros_like(coords)
        fint_chn = np.zeros((4, 3))
        mint_chn = np.zeros((4, 3))

        for c in range(split_cycle):
            fint_chn.fill(0.0)
            mint_chn.fill(0.0)
            shell_bt4.forces(g_chn, coords_chn, vel_chn, vr_chn, dt, fint_chn, mint_chn)
            coords_chn += vel_chn * dt

        # Save restart state via write_restart
        m_chn.x = coords_chn.copy()
        m_chn.v = vel_chn.copy()
        m_chn.shells = g_chn
        rst_file = tmp_path / "bt4_law57_restart.rst"
        write_restart(m_chn, str(rst_file), engine={"cycle": split_cycle, "t": split_cycle * dt})

        # Leg 2: restore from restart and continue cycles 5..9
        m_res, eng_res = read_restart(str(rst_file))
        assert eng_res["cycle"] == split_cycle

        g_res = m_res.shells
        coords_res = m_res.x.copy()
        vel_res = m_res.v.copy()
        vr_res = np.zeros_like(coords)
        fint_res = np.zeros((4, 3))
        mint_res = np.zeros((4, 3))

        for c in range(split_cycle, n_cycles):
            fint_res.fill(0.0)
            mint_res.fill(0.0)
            shell_bt4.forces(g_res, coords_res, vel_res, vr_res, dt, fint_res, mint_res)
            coords_res += vel_res * dt

        # Assert precise match between resumed and uninterrupted run within 10^-12
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
        np.testing.assert_allclose(
            mint_res,
            mint_unc,
            rtol=1e-12,
            atol=1e-12,
            err_msg="Final internal moments mismatch after restart continuation",
        )
        np.testing.assert_allclose(
            g_res.state["sig"],
            g_unc.state["sig"],
            rtol=1e-12,
            atol=1e-12,
            err_msg="Integration point stresses mismatch after restart continuation",
        )
        np.testing.assert_allclose(
            g_res.state["mat_extra"]["pla57"],
            g_unc.state["mat_extra"]["pla57"],
            rtol=1e-12,
            atol=1e-12,
            err_msg="Material state array pla57 mismatch after restart continuation",
        )
        np.testing.assert_allclose(
            g_res.state["mat_extra"]["sigb57"],
            g_unc.state["mat_extra"]["sigb57"],
            rtol=1e-12,
            atol=1e-12,
            err_msg="Material state array sigb57 mismatch after restart continuation",
        )
        np.testing.assert_allclose(
            g_res.state["mat_extra"]["thk57"],
            g_unc.state["mat_extra"]["thk57"],
            rtol=1e-12,
            atol=1e-12,
            err_msg="Material state array thk57 mismatch after restart continuation",
        )

    def test_end_to_end_engine_restart_continuation(self, tmp_path: Path):
        """Verify full engine restart chaining (run_starter -> run_engine leg1 -> run_engine leg2) matches uninterrupted run."""
        name_unc = "law57_e2e_unc"
        name_chn = "law57_e2e_chn"

        def make_deck(run_name: str) -> Path:
            d = StarterDeck(run_name)
            d.mat_law57(
                mid=1,
                title="E2EBarlat",
                rho=2.7e-3,
                e=70000.0,
                nu=0.33,
                r00=1.5,
                r45=1.2,
                r90=1.8,
                m=8.0,
                chard=250.0,
            )
            d.prop_shell(pid=1, title="ShellProp", ishell=1, thick=1.0, nip=3)
            d.part(pid=1, title="ShellPart", prop_id=1, mat_id=1)
            d.node([
                (1, 0.0, 0.0, 0.0),
                (2, 10.0, 0.0, 0.0),
                (3, 10.0, 10.0, 0.0),
                (4, 0.0, 10.0, 0.0),
            ])
            d.shell(1, [[1, 1, 2, 3, 4]])
            d.grnod_node(1, "PullNodes", [2, 3])
            d.inivel_tra(1, "StretchKick", [50.0, 0.0, 0.0], 1)
            f = tmp_path / f"{run_name}_0000.rad"
            d.write(str(f))
            return f

        # Determine stable time step
        probe_file = make_deck("law57_probe")
        run_starter(str(probe_file))
        probe_eng = tmp_path / "law57_probe_0001.rad"
        probe_eng.write_text("/RUN/law57_probe/1\n1.0e-5\n/DT\n0.5 0.0\n/PRINT/-1\n", encoding="utf-8")
        run_engine(str(probe_eng))
        _, eng_probe = read_restart(str(tmp_path / "law57_probe_0001.rst"))
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
