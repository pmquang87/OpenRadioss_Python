"""
Milestone M552: /MAT/LAW48 (/MAT/ZHAO, /MAT/PLAS_ZHAO)
Exhaustive Roundtrip, Serialization, Starter Diagnostics & Negative Testing Suite.

Fortran origins:
  - starter/source/materials/mat/mat048/hm_read_mat48.F
  - engine/source/materials/mat/mat048/sigeps48.F
  - engine/source/materials/mat/mat048/sigeps48c.F
  - config/CFG/radioss2023/MAT/matl48_zhao.cfg

Audits:
  1. Fixed-Format Deck Roundtrip:
     - StarterDeck.mat_law48 in standard 20-column fixed format.
     - Parameter sweeps: id, rho, ref_rho, E, nu, ca, cb, cn, hard, sig_max,
       cc, cd, cm, ce, ck, eps_rate_0, fcut, eps_max, eps_t1, eps_t2.
     - Bitwise or float-precise equality (<= 1e-6) for all 17 card fields:
       E, nu, ca, cb, cn, hard, sig_max, cc, cd, cm, ce, ck, eps_rate_0, fcut, eps_max, eps_t1, eps_t2.
     - Direct object invocation with MatLaw48 entity.
     - Verification across both Model.mat_law48s AST records and Model.materials runtime entities.

  2. Free-Format Comma-Delimited Roundtrip:
     - Write and re-parse free-format deck blocks for /MAT/ZHAO and /MAT/PLAS_ZHAO.
     - Comma-separated with whitespace and compact comma-delimited without whitespace.
     - Handling of consecutive commas (empty fields taking default values).
     - Cross-dialect roundtrip: free -> parse -> fixed -> parse.

  3. Negative Starter Diagnostics:
     - Non-positive density (rho0 <= 0).
     - Non-positive Young's modulus (E <= 0).
     - Invalid Poisson's ratio (nu < 0 or nu >= 0.5, ANCMSG 49).
     - Inverted tensile failure limits (eps_t2 <= eps_t1 when eps_t1 < 1e20, ANCMSG 420).
     - Warning for un-filtered strain rate (c > 0, eps_rate_0 > 0, fcut == 0, ANCMSG 1220).
     - Unsupported element types (springs, beams, trusses) assigned to LAW48.
     - Compatible element types accepted: bricks, tetras, penta6, pyra5, shells, shells_qbat, shells_qeph, sh3n, quads.
     - _ALLOWED_LAWS dictionary membership and rejection.

  4. Restart (.rst) Serialization:
     - MatLaw48 entity dataclass pickling and unpickling fidelity.
     - Law48Params physics object serialization.
     - extra_shapes verification: sigb48, eps48, epsd48, off48 for solids and shells.
     - Material state arrays preserved across write_restart / read_restart contract.
     - End-to-end dynamic simulation restart continuation (Hexa8 solid and BT4 shell).
     - Dynamic cycle continuation from restart matching uninterrupted run (cycle count, x, v, energies).
"""

from __future__ import annotations

import math
from pathlib import Path
import pickle
from typing import Any, Dict, List
import numpy as np
import pytest

from pyradioss.common.messages import MessageLog
from pyradioss.engine.engine import run_engine
from pyradioss.input.deck_reader import read_deck
from pyradioss.input.deck_writer import StarterDeck, fmt_float, fmt_int
from pyradioss.input.starter_keywords import parse_starter_deck, read_mat_law48
from pyradioss.materials.law48_zhao import (
    Law48Params,
    build_law48,
    extra_shapes,
    sound_speed_shell_law48,
    sound_speed_solid_law48,
)
from pyradioss.model.entities import MatLaw48, MatPlasZhao, MatZhao, Material
from pyradioss.model.model import Model
from pyradioss.starter.checks import (
    _ALLOWED_LAWS,
    check_mat_law48,
    check_materials,
    check_model,
)
from pyradioss.starter.restart import read_restart, write_restart
from pyradioss.starter.starter import run_starter


# ============================================================================
# Helpers
# ============================================================================

def _parse_deck_str(tmp_path: Path, text: str, name: str = "TEST_LAW48") -> tuple[Model, MessageLog]:
    """Write deck text to file and parse into Model and MessageLog."""
    deck_path = tmp_path / f"{name}_0000.rad"
    deck_path.write_text(text.strip() + "\n", encoding="utf-8")
    blocks = read_deck(str(deck_path))
    model = Model()
    log = MessageLog()
    parse_starter_deck(blocks, model, log)
    return model, log


def _get_material_cards(deck_str: str) -> List[str]:
    """Extract non-comment data cards of the first /MAT block in rendered deck."""
    lines = deck_str.splitlines()
    mat_idx = next(i for i, line in enumerate(lines) if line.startswith("/MAT/"))
    cards = []
    for line in lines[mat_idx + 2:]:
        if line.startswith("/"):
            break
        if not line.startswith("#") and line.strip():
            cards.append(line)
    return cards


def _assert_law48_all_card_fields_exact(
    m: MatLaw48,
    expected: dict[str, Any],
    tol: float = 1e-6,
    mat: Material | None = None,
) -> None:
    """Assert float-precise equality for all 17 card fields:
    E, nu, ca, cb, cn, hard, sig_max, cc, cd, cm, ce, ck, eps_rate_0, fcut, eps_max, eps_t1, eps_t2.
    """
    # 1. Density & Reference Density (Card 1)
    if "rho" in expected:
        assert m.rho == pytest.approx(expected["rho"], rel=tol)
    if "ref_rho" in expected:
        assert m.refer_rho == pytest.approx(expected["ref_rho"], rel=tol)
        assert m.rho0 == pytest.approx(expected["ref_rho"], rel=tol)

    # 2. Card 2: E, nu
    assert m.e == pytest.approx(expected["e"], rel=tol)
    assert m.E == pytest.approx(expected["e"], rel=tol)
    assert m.nu == pytest.approx(expected["nu"], rel=tol)

    # 3. Card 3: ca (a), cb (b), cn (n), hard (chard / mat_hard / fisokin), sig_max
    assert m.a == pytest.approx(expected["ca"], rel=tol)
    assert m.ca == pytest.approx(expected["ca"], rel=tol)
    assert m.sigy == pytest.approx(expected["ca"], rel=tol)

    assert m.b == pytest.approx(expected["cb"], rel=tol)
    assert m.cb == pytest.approx(expected["cb"], rel=tol)

    assert m.n == pytest.approx(expected["cn"], rel=tol)
    assert m.cn == pytest.approx(expected["cn"], rel=tol)

    assert m.chard == pytest.approx(expected["hard"], rel=tol)
    assert m.mat_hard == pytest.approx(expected["hard"], rel=tol)
    assert m.fisokin == pytest.approx(expected["hard"], rel=tol)

    assert m.sig_max == pytest.approx(expected["sig_max"], rel=tol)
    assert m.sigma_max == pytest.approx(expected["sig_max"], rel=tol)

    # 4. Card 4: cc (c), cd (d), cm (m), ce (e1), ck (k)
    assert m.c == pytest.approx(expected["cc"], rel=tol)
    assert m.cc == pytest.approx(expected["cc"], rel=tol)

    assert m.d == pytest.approx(expected["cd"], rel=tol)
    assert m.cd == pytest.approx(expected["cd"], rel=tol)

    assert m.m == pytest.approx(expected["cm"], rel=tol)
    assert m.cm == pytest.approx(expected["cm"], rel=tol)

    assert m.e1 == pytest.approx(expected["ce"], rel=tol)
    assert m.ce == pytest.approx(expected["ce"], rel=tol)

    assert m.k == pytest.approx(expected["ck"], rel=tol)
    assert m.ck == pytest.approx(expected["ck"], rel=tol)

    # 5. Card 5: eps_rate_0, fcut
    assert m.eps_rate_0 == pytest.approx(expected["eps_rate_0"], rel=tol)
    assert m.eps0 == pytest.approx(expected["eps_rate_0"], rel=tol)

    assert m.fcut == pytest.approx(expected["fcut"], rel=tol)
    assert m.scale == pytest.approx(expected["fcut"], rel=tol)

    # 6. Card 6: eps_max, eps_t1, eps_t2
    assert m.eps_max == pytest.approx(expected["eps_max"], rel=tol)

    assert m.eps_t1 == pytest.approx(expected["eps_t1"], rel=tol)
    assert m.eta1 == pytest.approx(expected["eps_t1"], rel=tol)

    assert m.eps_t2 == pytest.approx(expected["eps_t2"], rel=tol)
    assert m.eta2 == pytest.approx(expected["eps_t2"], rel=tol)

    # Verify runtime Material entity if present
    if mat is not None:
        assert mat.law == 48
        p = mat.params
        assert p["E"] == pytest.approx(expected["e"], rel=tol)
        assert p["nu"] == pytest.approx(expected["nu"], rel=tol)
        assert p["a"] == pytest.approx(expected["ca"], rel=tol)
        assert p["sigy"] == pytest.approx(expected["ca"], rel=tol)
        assert p["b"] == pytest.approx(expected["cb"], rel=tol)
        assert p["n"] == pytest.approx(expected["cn"], rel=tol)
        assert p["chard"] == pytest.approx(expected["hard"], rel=tol)
        assert p["sig_max"] == pytest.approx(expected["sig_max"], rel=tol)
        assert p["c"] == pytest.approx(expected["cc"], rel=tol)
        assert p["d"] == pytest.approx(expected["cd"], rel=tol)
        assert p["m"] == pytest.approx(expected["cm"], rel=tol)
        assert p["e1"] == pytest.approx(expected["ce"], rel=tol)
        assert p["k"] == pytest.approx(expected["ck"], rel=tol)
        assert p["eps_rate_0"] == pytest.approx(expected["eps_rate_0"], rel=tol)
        assert p["fcut"] == pytest.approx(expected["fcut"], rel=tol)
        assert p["eps_max"] == pytest.approx(expected["eps_max"], rel=tol)
        assert p["eps_t1"] == pytest.approx(expected["eps_t1"], rel=tol)
        assert p["eps_t2"] == pytest.approx(expected["eps_t2"], rel=tol)


# ============================================================================
# 1. Fixed-Format Deck Roundtrip
# ============================================================================

class TestLaw48FixedFormatRoundtrip:
    """Audit writing and parsing /MAT/LAW48 using standard 20-column fixed format."""

    @pytest.mark.parametrize(
        "params",
        [
            # Steel Zhao parameters
            {
                "id": 1,
                "title": "Zhao Steel 1",
                "rho": 7.85e-9,
                "ref_rho": 7.85e-9,
                "e": 210000.0,
                "nu": 0.30,
                "ca": 250.0,
                "cb": 500.0,
                "cn": 0.45,
                "hard": 0.10,
                "sig_max": 800.0,
                "cc": 0.02,
                "cd": 0.01,
                "cm": 0.35,
                "ce": 100.0,
                "ck": 0.80,
                "eps_rate_0": 1.0,
                "fcut": 1000.0,
                "eps_max": 0.50,
                "eps_t1": 0.40,
                "eps_t2": 0.60,
            },
            # Aluminum Alloy Zhao parameters
            {
                "id": 2,
                "title": "Zhao Aluminum 2",
                "rho": 2.70e-9,
                "ref_rho": 2.70e-9,
                "e": 70000.0,
                "nu": 0.33,
                "ca": 180.0,
                "cb": 320.0,
                "cn": 0.38,
                "hard": 0.25,
                "sig_max": 520.0,
                "cc": 0.015,
                "cd": 0.008,
                "cm": 0.28,
                "ce": 60.0,
                "ck": 0.95,
                "eps_rate_0": 0.01,
                "fcut": 2500.0,
                "eps_max": 0.35,
                "eps_t1": 0.25,
                "eps_t2": 0.45,
            },
            # High-strength Titanium Zhao parameters
            {
                "id": 3,
                "title": "Zhao Titanium 3",
                "rho": 4.50e-9,
                "ref_rho": 4.52e-9,
                "e": 115000.0,
                "nu": 0.31,
                "ca": 850.0,
                "cb": 650.0,
                "cn": 0.52,
                "hard": 0.0,
                "sig_max": 1350.0,
                "cc": 0.035,
                "cd": 0.018,
                "cm": 0.42,
                "ce": 150.0,
                "ck": 0.72,
                "eps_rate_0": 1.0e-3,
                "fcut": 5000.0,
                "eps_max": 0.40,
                "eps_t1": 0.30,
                "eps_t2": 0.55,
            },
            # Pure kinematic hardening with infinite saturation
            {
                "id": 4,
                "title": "Zhao Kinematic Hardening",
                "rho": 7.90e-9,
                "ref_rho": 7.90e-9,
                "e": 200000.0,
                "nu": 0.29,
                "ca": 310.0,
                "cb": 420.0,
                "cn": 0.60,
                "hard": 1.0,
                "sig_max": 1.0e30,
                "cc": 0.025,
                "cd": 0.012,
                "cm": 0.30,
                "ce": 80.0,
                "ck": 1.10,
                "eps_rate_0": 10.0,
                "fcut": 800.0,
                "eps_max": 0.60,
                "eps_t1": 0.45,
                "eps_t2": 0.70,
            },
        ],
    )
    def test_fixed_format_roundtrip_all_card_fields(self, tmp_path: Path, params: dict[str, Any]):
        """Write /MAT/LAW48 using 20-col fixed format, re-parse and assert 1e-6 equality for all 17 card fields."""
        mid = params["id"]
        title = params["title"]
        deck = StarterDeck(f"FIXED_{mid}").mat_law48(
            mid=mid,
            title=title,
            rho=params["rho"],
            ref_rho=params["ref_rho"],
            e=params["e"],
            nu=params["nu"],
            a=params["ca"],
            b=params["cb"],
            n=params["cn"],
            chard=params["hard"],
            sig_max=params["sig_max"],
            c=params["cc"],
            d=params["cd"],
            m=params["cm"],
            e1=params["ce"],
            k=params["ck"],
            eps_rate_0=params["eps_rate_0"],
            fcut=params["fcut"],
            eps_max=params["eps_max"],
            eps_t1=params["eps_t1"],
            eps_t2=params["eps_t2"],
            fixed_format=True,
        )
        rendered = deck.render()
        cards = _get_material_cards(rendered)

        # 1. Verify card geometry: 6 non-comment data cards with standard 20-column layout
        assert len(cards) == 6
        # Card 1: 2 fields of 20
        assert len(cards[0]) >= 40
        # Card 2: 2 fields of 20
        assert len(cards[1]) >= 40
        # Card 3: 5 fields of 20
        assert len(cards[2]) >= 100
        # Card 4: 5 fields of 20
        assert len(cards[3]) >= 100
        # Card 5: 2 fields of 20
        assert len(cards[4]) >= 40
        # Card 6: 3 fields of 20
        assert len(cards[5]) >= 60

        # 2. Parse back into Model
        model, log = _parse_deck_str(tmp_path, rendered, name=f"FIXED_{mid}")
        assert not log.has_errors, f"Parse errors: {log.errors}"

        # 3. Assert equality for all 17 fields on MatLaw48 and Material
        assert mid in model.mat_law48s
        m = model.mat_law48s[mid]
        mat = model.materials.get(mid)
        _assert_law48_all_card_fields_exact(m, params, tol=1e-6, mat=mat)

    def test_fixed_format_object_invocation_roundtrip(self, tmp_path: Path):
        """Verify passing a MatLaw48 instance directly to deck.mat_law48 produces exact roundtrip."""
        m_in = MatLaw48(
            id=48,
            title="Direct Object MatLaw48",
            rho=7.82e-9,
            refer_rho=7.82e-9,
            e=208000.0,
            nu=0.295,
            a=255.0,
            b=470.0,
            n=0.43,
            chard=0.12,
            sig_max=820.0,
            c=0.018,
            d=0.008,
            m=0.34,
            e1=95.0,
            k=0.82,
            eps_rate_0=1.0,
            fcut=1050.0,
            eps_max=0.52,
            eps_t1=0.41,
            eps_t2=0.61,
        )
        deck = StarterDeck("DIRECT_OBJ").mat_law48(m_in, fixed_format=True)
        rendered = deck.render()
        model, log = _parse_deck_str(tmp_path, rendered, name="DIRECT_OBJ")
        assert not log.has_errors

        m_out = model.mat_law48s[48]
        expected = {
            "e": m_in.e,
            "nu": m_in.nu,
            "ca": m_in.a,
            "cb": m_in.b,
            "cn": m_in.n,
            "hard": m_in.chard,
            "sig_max": m_in.sig_max,
            "cc": m_in.c,
            "cd": m_in.d,
            "cm": m_in.m,
            "ce": m_in.e1,
            "ck": m_in.k,
            "eps_rate_0": m_in.eps_rate_0,
            "fcut": m_in.fcut,
            "eps_max": m_in.eps_max,
            "eps_t1": m_in.eps_t1,
            "eps_t2": m_in.eps_t2,
            "rho": m_in.rho,
            "ref_rho": m_in.refer_rho,
        }
        _assert_law48_all_card_fields_exact(m_out, expected, tol=1e-6, mat=model.materials[48])


# ============================================================================
# 2. Free-Format Comma-Delimited Roundtrip
# ============================================================================

class TestLaw48FreeFormatRoundtrip:
    """Audit free-format comma-delimited reading and writing for /MAT/ZHAO and /MAT/PLAS_ZHAO."""

    def test_zhao_free_format_comma_delimited_with_spaces(self, tmp_path: Path):
        """Write /MAT/ZHAO in free format with commas and spaces, re-parse and verify fidelity."""
        expected = {
            "id": 10,
            "rho": 7.85e-9,
            "ref_rho": 7.85e-9,
            "e": 210000.0,
            "nu": 0.30,
            "ca": 260.0,
            "cb": 480.0,
            "cn": 0.44,
            "hard": 0.15,
            "sig_max": 850.0,
            "cc": 0.022,
            "cd": 0.011,
            "cm": 0.36,
            "ce": 110.0,
            "ck": 0.85,
            "eps_rate_0": 1.0,
            "fcut": 1200.0,
            "eps_max": 0.55,
            "eps_t1": 0.42,
            "eps_t2": 0.62,
        }
        deck = StarterDeck("ZHAO_FREE").mat_zhao(
            mid=expected["id"],
            title="Zhao Comma Space Deck",
            rho=expected["rho"],
            ref_rho=expected["ref_rho"],
            e=expected["e"],
            nu=expected["nu"],
            ca=expected["ca"],
            cb=expected["cb"],
            cn=expected["cn"],
            hard=expected["hard"],
            sig_max=expected["sig_max"],
            cc=expected["cc"],
            cd=expected["cd"],
            cm=expected["cm"],
            ce=expected["ce"],
            ck=expected["ck"],
            eps_rate_0=expected["eps_rate_0"],
            fcut=expected["fcut"],
            eps_max=expected["eps_max"],
            eps_t1=expected["eps_t1"],
            eps_t2=expected["eps_t2"],
            fixed_format=False,
            delimiter=", ",
        )
        rendered = deck.render()
        assert "/MAT/ZHAO/10" in rendered
        cards = _get_material_cards(rendered)
        for c in cards:
            assert "," in c

        model, log = _parse_deck_str(tmp_path, rendered, name="ZHAO_FREE")
        assert not log.has_errors
        assert 10 in model.mat_law48s
        m = model.mat_law48s[10]
        _assert_law48_all_card_fields_exact(m, expected, tol=1e-6, mat=model.materials[10])

    def test_plas_zhao_free_format_compact_commas(self, tmp_path: Path):
        """Write /MAT/PLAS_ZHAO in compact comma format (no spaces), re-parse and verify fidelity."""
        expected = {
            "id": 20,
            "rho": 7.90e-9,
            "ref_rho": 7.90e-9,
            "e": 195000.0,
            "nu": 0.28,
            "ca": 280.0,
            "cb": 520.0,
            "cn": 0.48,
            "hard": 0.20,
            "sig_max": 780.0,
            "cc": 0.018,
            "cd": 0.009,
            "cm": 0.32,
            "ce": 90.0,
            "ck": 0.78,
            "eps_rate_0": 0.5,
            "fcut": 850.0,
            "eps_max": 0.45,
            "eps_t1": 0.35,
            "eps_t2": 0.58,
        }
        deck = StarterDeck("PLAS_ZHAO_COMPACT").mat_plas_zhao(
            mid=expected["id"],
            title="Plas Zhao Compact Comma Deck",
            rho=expected["rho"],
            ref_rho=expected["ref_rho"],
            e=expected["e"],
            nu=expected["nu"],
            ca=expected["ca"],
            cb=expected["cb"],
            cn=expected["cn"],
            hard=expected["hard"],
            sig_max=expected["sig_max"],
            cc=expected["cc"],
            cd=expected["cd"],
            cm=expected["cm"],
            ce=expected["ce"],
            ck=expected["ck"],
            eps_rate_0=expected["eps_rate_0"],
            fcut=expected["fcut"],
            eps_max=expected["eps_max"],
            eps_t1=expected["eps_t1"],
            eps_t2=expected["eps_t2"],
            fixed_format=False,
            delimiter=",",
        )
        rendered = deck.render()
        assert "/MAT/PLAS_ZHAO/20" in rendered
        cards = _get_material_cards(rendered)
        for c in cards:
            assert "," in c

        model, log = _parse_deck_str(tmp_path, rendered, name="PLAS_ZHAO_COMPACT")
        assert not log.has_errors
        assert 20 in model.mat_law48s
        m = model.mat_law48s[20]
        _assert_law48_all_card_fields_exact(m, expected, tol=1e-6, mat=model.materials[20])

    def test_free_format_consecutive_commas_default_injection(self, tmp_path: Path):
        """Verify consecutive commas in free format preserve default values matching hm_read_mat48.F."""
        rad = """# RADIOSS STARTER
/BEGIN
CONSECUTIVE_COMMAS
/MAT/LAW48/30
Consecutive Commas Default Deck
7.85e-9
210000.0, 0.30
250.0, 500.0, , , 
, , , , 
, 
, , 
"""
        model, log = _parse_deck_str(tmp_path, rad, name="CONSEC_COMMAS")
        assert not log.has_errors
        assert 30 in model.mat_law48s
        m = model.mat_law48s[30]

        # Explicitly given fields
        assert m.rho == pytest.approx(7.85e-9)
        assert m.e == pytest.approx(210000.0)
        assert m.nu == pytest.approx(0.30)
        assert m.a == pytest.approx(250.0)
        assert m.b == pytest.approx(500.0)

        # Defaulted fields
        assert m.n == pytest.approx(1.0001)   # n=0 or 1 -> 1.0001
        assert m.chard == pytest.approx(0.0)
        assert m.sig_max == pytest.approx(1.0e30)
        assert m.c == pytest.approx(0.0)
        assert m.d == pytest.approx(0.0)
        assert m.m == pytest.approx(1.0001)   # m=0 or 1 -> 1.0001
        assert m.e1 == pytest.approx(0.0)
        assert m.k == pytest.approx(1.0)      # k=0 -> 1.0
        assert m.eps_rate_0 == pytest.approx(1.0)
        assert m.fcut == pytest.approx(1.0e30)
        assert m.eps_max == pytest.approx(1.0e30)
        assert m.eps_t1 == pytest.approx(1.0e30)
        assert m.eps_t2 == pytest.approx(2.0e30)

    def test_two_way_cross_dialect_roundtrip(self, tmp_path: Path):
        """Cross-dialect roundtrip: write free comma -> parse -> write fixed -> parse -> verify bitwise identity."""
        orig_params = {
            "id": 99,
            "title": "Cross Dialect Roundtrip",
            "rho": 7.84e-9,
            "ref_rho": 7.84e-9,
            "e": 206000.0,
            "nu": 0.292,
            "ca": 265.0,
            "cb": 475.0,
            "cn": 0.425,
            "hard": 0.18,
            "sig_max": 830.0,
            "cc": 0.019,
            "cd": 0.007,
            "cm": 0.33,
            "ce": 105.0,
            "ck": 0.81,
            "eps_rate_0": 1.0,
            "fcut": 1150.0,
            "eps_max": 0.51,
            "eps_t1": 0.39,
            "eps_t2": 0.61,
        }
        # 1. Write free-format comma deck
        deck_free = StarterDeck("CROSS_1").mat_zhao(
            mid=99,
            **orig_params,
            fixed_format=False,
            delimiter=", ",
        )
        rendered_free = deck_free.render()

        # 2. Parse free-format deck
        model1, log1 = _parse_deck_str(tmp_path, rendered_free, name="CROSS_FREE")
        assert not log1.has_errors
        m1 = model1.mat_law48s[99]

        # 3. Write parsed entity as fixed-format deck
        deck_fixed = StarterDeck("CROSS_2").mat_law48(m1, fixed_format=True)
        rendered_fixed = deck_fixed.render()

        # 4. Parse fixed-format deck
        model2, log2 = _parse_deck_str(tmp_path, rendered_fixed, name="CROSS_FIXED")
        assert not log2.has_errors
        m2 = model2.mat_law48s[99]

        # 5. Assert exact match between m1 and m2
        _assert_law48_all_card_fields_exact(m2, orig_params, tol=1e-6, mat=model2.materials[99])


# ============================================================================
# 3. Negative Starter Diagnostics
# ============================================================================

class TestLaw48NegativeDiagnostics:
    """Audit check_mat_law48 and check_model diagnostic errors and warnings."""

    class DummyMat:
        def __init__(self, **kwargs):
            self.id = kwargs.get("id", 1)
            self.rho0 = kwargs.get("rho0", 7.85e-9)
            self.rho = self.rho0
            self.ref_rho = kwargs.get("ref_rho", self.rho0)
            self.e = kwargs.get("e", 210000.0)
            self.E = self.e
            self.nu = kwargs.get("nu", 0.30)
            self.a = kwargs.get("a", 250.0)
            self.sigy = self.a
            self.b = kwargs.get("b", 500.0)
            self.n = kwargs.get("n", 0.45)
            self.chard = kwargs.get("chard", 0.10)
            self.mat_hard = self.chard
            self.fisokin = self.chard
            self.sig_max = kwargs.get("sig_max", 800.0)
            self.c = kwargs.get("c", 0.02)
            self.d = kwargs.get("d", 0.01)
            self.m = kwargs.get("m", 0.35)
            self.e1 = kwargs.get("e1", 100.0)
            self.k = kwargs.get("k", 0.80)
            self.eps_rate_0 = kwargs.get("eps_rate_0", 1.0)
            self.fcut = kwargs.get("fcut", 1000.0)
            self.eps_max = kwargs.get("eps_max", 0.50)
            self.eps_t1 = kwargs.get("eps_t1", 0.40)
            self.eps_t2 = kwargs.get("eps_t2", 0.60)
            self.law = 48

    def test_non_positive_density_rejected(self):
        """rho0 <= 0 (zero or negative) must raise error."""
        for bad_rho in (0.0, -7.85e-9):
            mat = self.DummyMat(rho0=bad_rho)
            log = MessageLog()
            check_mat_law48(mat, log=log)
            assert log.has_errors
            assert any("density" in str(e).lower() or "rho" in str(e).lower() for e in log.errors)

    def test_non_positive_youngs_modulus_rejected(self):
        """E <= 0 (zero or negative) must raise error."""
        for bad_e in (0.0, -210000.0):
            mat = self.DummyMat(e=bad_e)
            log = MessageLog()
            check_mat_law48(mat, log=log)
            assert log.has_errors
            assert any("young" in str(e).lower() or "e" in str(e).lower() for e in log.errors)

    def test_invalid_poisson_ratio_rejected(self):
        """nu < 0 or nu >= 0.5 must raise error (ANCMSG 49)."""
        for bad_nu in (-0.05, -0.2, 0.5, 0.52):
            mat = self.DummyMat(nu=bad_nu)
            log = MessageLog()
            check_mat_law48(mat, log=log)
            assert log.has_errors
            assert any("poisson" in str(e).lower() or "nu" in str(e).lower() or "49" in str(e) for e in log.errors)

    def test_inverted_tensile_failure_limits_rejected(self):
        """eps_t2 <= eps_t1 when eps_t1 < 1e20 must raise error (ANCMSG 420)."""
        # Strictly inverted
        mat1 = self.DummyMat(eps_t1=0.6, eps_t2=0.4)
        log1 = MessageLog()
        check_mat_law48(mat1, log=log1)
        assert log1.has_errors
        assert any("eps_t2" in str(e).lower() or "420" in str(e) for e in log1.errors)

        # Equal failure strains
        mat2 = self.DummyMat(eps_t1=0.5, eps_t2=0.5)
        log2 = MessageLog()
        check_mat_law48(mat2, log=log2)
        assert log2.has_errors
        assert any("eps_t2" in str(e).lower() or "420" in str(e) for e in log2.errors)

        # Inactive failure limits (>= 1e20) must be accepted
        mat3 = self.DummyMat(eps_t1=1.0e30, eps_t2=2.0e30)
        log3 = MessageLog()
        check_mat_law48(mat3, log=log3)
        assert not log3.has_errors

    def test_warning_for_unfiltered_strain_rate(self):
        """c > 0 and eps_rate_0 > 0 and fcut == 0 must emit warning (ANCMSG 1220)."""
        mat = self.DummyMat(c=0.02, eps_rate_0=1.0, fcut=0.0)
        log = MessageLog()
        check_mat_law48(mat, log=log)
        assert not log.has_errors
        assert log.has_warnings
        assert any("cutoff frequency" in str(w).lower() or "1220" in str(w) for w in log.warnings)

    def test_unsupported_element_types_rejected(self):
        """Trusses, beams, and springs must reject LAW48 in starter checks."""
        class MockElement:
            def __init__(self, mat_id: int):
                self.mat_id = mat_id

        class MockElementGroup:
            def __init__(self, mat_obj: Any):
                self.elements = {1: MockElement(mat_obj.id)}
                self.state = {"slices": [(slice(0, 1), mat_obj, None)]}

            def values(self):
                return self.elements.values()

        for bad_elem in ("trusses", "beams", "springs"):
            model = Model()
            model.node_ids = np.array([1, 2])
            mat = self.DummyMat(id=10)
            model.materials[10] = mat
            grp = MockElementGroup(mat)

            # Test check_mat_law48 with element_groups
            model.element_groups = lambda g=bad_elem, gr=grp: [(g, gr)]
            log = MessageLog()
            check_mat_law48(mat, model=model, log=log)
            assert log.has_errors
            assert any(bad_elem in str(e).lower() or "not supported" in str(e).lower() for e in log.errors)

            # Test check_model allowed laws
            log_model = MessageLog()
            check_model(model, log_model)
            assert log_model.has_errors
            assert any(bad_elem in str(e).lower() or "not supported" in str(e).lower() for e in log_model.errors)

    def test_supported_element_types_accepted(self):
        """Bricks, tetras, penta6, pyra5, shells, shells_qbat, shells_qeph, sh3n, quads must accept LAW48."""
        valid_families = (
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
        for fam in valid_families:
            assert 48 in _ALLOWED_LAWS[fam]
            assert "48" in _ALLOWED_LAWS[fam]
            assert "LAW48" in _ALLOWED_LAWS[fam]
            assert "ZHAO" in _ALLOWED_LAWS[fam]
            assert "PLAS_ZHAO" in _ALLOWED_LAWS[fam]
            assert "MAT_ZHAO" in _ALLOWED_LAWS[fam]
            assert "MAT_PLAS_ZHAO" in _ALLOWED_LAWS[fam]


# ============================================================================
# 4. Restart (.rst) Serialization
# ============================================================================

class TestLaw48RestartSerialization:
    """Audit restart (.rst) serialization: persistent arrays, pickle fidelity, and dynamic continuation."""

    def test_mat_law48_dataclass_pickling_fidelity(self):
        """MatLaw48 dataclass must pickle and unpickle with bitwise / float-precise fidelity."""
        m_orig = MatLaw48(
            id=48,
            title="Pickle Zhao Test",
            rho=7.85e-9,
            refer_rho=7.85e-9,
            e=210000.0,
            nu=0.30,
            a=250.0,
            b=500.0,
            n=0.45,
            chard=0.10,
            sig_max=800.0,
            c=0.02,
            d=0.01,
            m=0.35,
            e1=100.0,
            k=0.80,
            eps_rate_0=1.0,
            fcut=1000.0,
            eps_max=0.50,
            eps_t1=0.40,
            eps_t2=0.60,
        )
        data = pickle.dumps(m_orig, protocol=pickle.HIGHEST_PROTOCOL)
        m_rest: MatLaw48 = pickle.loads(data)

        assert m_rest.id == m_orig.id
        assert m_rest.title == m_orig.title
        assert m_rest.rho == pytest.approx(m_orig.rho)
        assert m_rest.e == pytest.approx(m_orig.e)
        assert m_rest.nu == pytest.approx(m_orig.nu)
        assert m_rest.a == pytest.approx(m_orig.a)
        assert m_rest.b == pytest.approx(m_orig.b)
        assert m_rest.n == pytest.approx(m_orig.n)
        assert m_rest.chard == pytest.approx(m_orig.chard)
        assert m_rest.sig_max == pytest.approx(m_orig.sig_max)
        assert m_rest.c == pytest.approx(m_orig.c)
        assert m_rest.d == pytest.approx(m_orig.d)
        assert m_rest.m == pytest.approx(m_orig.m)
        assert m_rest.e1 == pytest.approx(m_orig.e1)
        assert m_rest.k == pytest.approx(m_orig.k)
        assert m_rest.eps_rate_0 == pytest.approx(m_orig.eps_rate_0)
        assert m_rest.fcut == pytest.approx(m_orig.fcut)
        assert m_rest.eps_max == pytest.approx(m_orig.eps_max)
        assert m_rest.eps_t1 == pytest.approx(m_orig.eps_t1)
        assert m_rest.eps_t2 == pytest.approx(m_orig.eps_t2)

        # Computed acoustic velocities preserved
        assert m_rest.sound_speed_solid() == pytest.approx(m_orig.sound_speed_solid())
        assert m_rest.sound_speed_shell() == pytest.approx(m_orig.sound_speed_shell())

    def test_law48_params_pickling_fidelity(self):
        """Law48Params physics object must pickle and unpickle with exact elastic moduli."""
        params_orig = Law48Params(
            rho0=7.85e-9,
            rhor=7.85e-9,
            E=210000.0,
            nu=0.30,
            ca=1.0,
            sigy0=250.0,
            cb=500.0,
            cn=0.45,
            fisokin=0.10,
            sig_max=800.0,
            cc=0.02,
            cd=0.01,
            cm=0.35,
            ce=100.0,
            ck=0.80,
            eps0=1.0,
            fcut=1000.0,
            eps_max=0.50,
            eps_t1=0.40,
            eps_t2=0.60,
        )
        data = pickle.dumps(params_orig, protocol=pickle.HIGHEST_PROTOCOL)
        params_rest: Law48Params = pickle.loads(data)

        assert params_rest.G == pytest.approx(params_orig.G)
        assert params_rest.K == pytest.approx(params_orig.K)
        assert params_rest.A11 == pytest.approx(params_orig.A11)
        assert params_rest.A21 == pytest.approx(params_orig.A21)
        assert sound_speed_solid_law48(params_rest) == pytest.approx(sound_speed_solid_law48(params_orig))
        assert sound_speed_shell_law48(params_rest) == pytest.approx(sound_speed_shell_law48(params_orig))

    def test_extra_shapes_solid_and_shell(self):
        """extra_shapes must allocate sigb48, eps48, epsd48, off48 for solids and shells."""
        mat = Material(id=1, law=48, rho0=7.85e-9, params={"E": 210000.0, "nu": 0.3})

        # Solid shapes (nip is None)
        s_solid = extra_shapes(mat, nip=None)
        assert s_solid["eps48"] == (6,)
        assert s_solid["sigb48"] == (6,)
        assert s_solid["epsd48"] == ()
        assert s_solid["off48"] == ()

        # Shell shapes (nip = 5)
        s_shell = extra_shapes(mat, nip=5)
        assert s_shell["eps48"] == (5, 3)
        assert s_shell["sigb48"] == (5, 3)
        assert s_shell["epsd48"] == (5,)
        assert s_shell["off48"] == (5,)

    def test_material_state_arrays_rst_preservation(self, tmp_path: Path):
        """sigb48, eps48, epsd48, off48 in solids and shells must survive write_restart and read_restart."""
        deck = StarterDeck("RST_LAW48_ARRAYS")
        deck.mat_law48(
            mid=1,
            title="ZhaoSteel",
            rho=7.85e-9,
            e=210000.0,
            nu=0.30,
            a=250.0,
            b=500.0,
            n=0.45,
            chard=0.10,
            sig_max=800.0,
        )
        deck.prop_solid(pid=1, title="SolidProp", isolid=1)
        deck.prop_shell(pid=2, title="ShellProp", ishell=1, thick=1.2)
        deck.part(pid=1, title="SolidPart", prop_id=1, mat_id=1)
        deck.part(pid=2, title="ShellPart", prop_id=2, mat_id=1)

        nodes = [
            (1, 0.0, 0.0, 0.0), (2, 2.0, 0.0, 0.0), (3, 2.0, 2.0, 0.0), (4, 0.0, 2.0, 0.0),
            (5, 0.0, 0.0, 2.0), (6, 2.0, 0.0, 2.0), (7, 2.0, 2.0, 2.0), (8, 0.0, 2.0, 2.0),
        ]
        deck.node(nodes)
        deck.brick(1, [[1, 1, 2, 3, 4, 5, 6, 7, 8]])
        deck.shell(2, [[2, 1, 2, 3, 4]])

        rad_file = tmp_path / "rst_arrays_0000.rad"
        deck.write(str(rad_file))

        model = run_starter(str(rad_file))
        assert model is not None

        groups = dict(model.element_groups())
        assert "bricks" in groups
        assert "shells" in groups

        # Synthesize realistic persistent state arrays
        bg = groups["bricks"]
        if "mat_extra" not in bg.state:
            bg.state["mat_extra"] = {}
        sigb48_solid = np.array([[15.2, -7.6, -7.6, 3.4, 0.0, 0.0]])
        eps48_solid = np.array([[0.032, -0.011, -0.011, 0.005, 0.0, 0.0]])
        epsd48_solid = np.array([125.0])
        off48_solid = np.array([1.0])

        bg.state["mat_extra"]["sigb48"] = sigb48_solid
        bg.state["mat_extra"]["eps48"] = eps48_solid
        bg.state["mat_extra"]["epsd48"] = epsd48_solid
        bg.state["mat_extra"]["off48"] = off48_solid

        shg = groups["shells"]
        if "mat_extra" not in shg.state:
            shg.state["mat_extra"] = {}
        nip = 5
        sigb48_shell = np.zeros((1, nip, 3))
        sigb48_shell[0, :, 0] = [12.0, 10.0, 0.0, -10.0, -12.0]
        eps48_shell = np.zeros((1, nip, 3))
        eps48_shell[0, :, 0] = [0.04, 0.02, 0.0, -0.02, -0.04]
        epsd48_shell = np.full((1, nip), 85.0)
        off48_shell = np.ones((1, nip))
        off48_shell[0, 0] = 0.0   # outer layer eroded

        shg.state["mat_extra"]["sigb48"] = sigb48_shell
        shg.state["mat_extra"]["eps48"] = eps48_shell
        shg.state["mat_extra"]["epsd48"] = epsd48_shell
        shg.state["mat_extra"]["off48"] = off48_shell

        # Save to restart file
        rst_path = tmp_path / "rst_arrays_0001.rst"
        engine_dict = {
            "cycle": 200,
            "t": 2.5e-6,
            "dt": 1.25e-8,
            "energies": {"internal": 45.67, "kinetic": 12.34},
        }
        write_restart(model, str(rst_path), engine=engine_dict)

        # Restore from restart file
        rest_model, rest_engine = read_restart(str(rst_path))
        assert rest_engine["cycle"] == 200
        assert rest_engine["t"] == pytest.approx(2.5e-6)
        assert rest_engine["energies"]["internal"] == pytest.approx(45.67)

        rest_groups = dict(rest_model.element_groups())
        rest_bg = rest_groups["bricks"]
        np.testing.assert_allclose(
            rest_bg.state["mat_extra"]["sigb48"],
            sigb48_solid,
            err_msg="Solid sigb48 backstress did not survive restart restoration",
        )
        np.testing.assert_allclose(
            rest_bg.state["mat_extra"]["eps48"],
            eps48_solid,
            err_msg="Solid eps48 strain did not survive restart restoration",
        )
        np.testing.assert_allclose(
            rest_bg.state["mat_extra"]["epsd48"],
            epsd48_solid,
            err_msg="Solid epsd48 strain rate did not survive restart restoration",
        )
        np.testing.assert_allclose(
            rest_bg.state["mat_extra"]["off48"],
            off48_solid,
            err_msg="Solid off48 mask did not survive restart restoration",
        )

        rest_shg = rest_groups["shells"]
        np.testing.assert_allclose(
            rest_shg.state["mat_extra"]["sigb48"],
            sigb48_shell,
            err_msg="Shell sigb48 backstress did not survive restart restoration",
        )
        np.testing.assert_allclose(
            rest_shg.state["mat_extra"]["eps48"],
            eps48_shell,
            err_msg="Shell eps48 strain did not survive restart restoration",
        )
        np.testing.assert_allclose(
            rest_shg.state["mat_extra"]["epsd48"],
            epsd48_shell,
            err_msg="Shell epsd48 strain rate did not survive restart restoration",
        )
        np.testing.assert_allclose(
            rest_shg.state["mat_extra"]["off48"],
            off48_shell,
            err_msg="Shell off48 mask did not survive restart restoration",
        )

    def test_dynamic_restart_continuation_vs_uninterrupted_run(self, tmp_path: Path):
        """Verify dynamic cycle continuation from restart matches un-interrupted run bitwise or to roundoff."""
        def make_deck(run_name: str) -> Path:
            d = StarterDeck(run_name)
            d.mat_law48(
                mid=1,
                title="DynamicZhao",
                rho=7.85e-9,
                e=210000.0,
                nu=0.30,
                a=250.0,
                b=450.0,
                n=0.45,
                chard=0.15,
                sig_max=750.0,
                c=0.02,
                d=0.01,
                m=0.35,
                e1=80.0,
                k=0.85,
                eps_rate_0=1.0,
                fcut=1000.0,
            )
            d.prop_solid(pid=1, title="PropSolid", isolid=1)
            d.part(pid=1, title="PartSolid", prop_id=1, mat_id=1)
            nodes = [
                (1, 0.0, 0.0, 0.0), (2, 2.0, 0.0, 0.0), (3, 2.0, 2.0, 0.0), (4, 0.0, 2.0, 0.0),
                (5, 0.0, 0.0, 2.0), (6, 2.0, 0.0, 2.0), (7, 2.0, 2.0, 2.0), (8, 0.0, 2.0, 2.0),
            ]
            d.node(nodes)
            d.brick(1, [[1, 1, 2, 3, 4, 5, 6, 7, 8]])
            d.grnod_node(1, "PullNodes", [5, 6, 7, 8])
            d.inivel_tra(1, "TensionKick", [0.0, 0.0, 40.0], 1)
            f = tmp_path / f"{run_name}_0000.rad"
            d.write(str(f))
            return f

        # Probe stable time step
        probe_rad = make_deck("probe")
        run_starter(str(probe_rad))
        probe_eng = tmp_path / "probe_0001.rad"
        probe_eng.write_text("/RUN/probe/1\n1.0e-9\n/DT\n0.5 0.0\n/PRINT/-1\n", encoding="utf-8")
        run_engine(str(probe_eng))
        _, eng_probe = read_restart(str(tmp_path / "probe_0001.rst"))
        dt0 = eng_probe["dt"]

        # Run legs aligned on exact cycle intervals
        t1 = 2 * dt0
        t2 = 4 * dt0

        # 1. Un-interrupted execution from 0 to t2
        unc_rad = make_deck("unc_run")
        run_starter(str(unc_rad))
        unc_eng = tmp_path / "unc_run_0001.rad"
        unc_eng.write_text(f"/RUN/unc_run/1\n{t2!r}\n/DT\n0.5 0.0\n/PRINT/-1\n", encoding="utf-8")
        m_unc = run_engine(str(unc_eng))

        # 2. Chained execution: Leg 1 (0 to t1)
        chn_rad = make_deck("chn_run")
        run_starter(str(chn_rad))
        chn_eng1 = tmp_path / "chn_run_0001.rad"
        chn_eng1.write_text(f"/RUN/chn_run/1\n{t1!r}\n/DT\n0.5 0.0\n/PRINT/-1\n", encoding="utf-8")
        m_chn1 = run_engine(str(chn_eng1))
        rst1_path = tmp_path / "chn_run_0001.rst"
        assert rst1_path.exists()

        # Leg 2 (t1 to t2) resuming from chn_run_0001.rst
        chn_eng2 = tmp_path / "chn_run_0002.rad"
        chn_eng2.write_text(f"/RUN/chn_run/2\n{t2!r}\n/DT\n0.5 0.0\n/PRINT/-1\n", encoding="utf-8")
        m_chn2 = run_engine(str(chn_eng2))
        assert (tmp_path / "chn_run_0002.rst").exists()

        # 3. Dynamic continuation verification: cycle count, coordinates, velocities
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

    def test_dynamic_restart_continuation_shell_bt4(self, tmp_path: Path):
        """Verify dynamic cycle continuation for BT4 shell with LAW48."""
        name = "shell_dyn_rst"
        deck = StarterDeck(name)
        deck.mat_zhao(
            mid=1,
            title="ZhaoShellMat",
            rho=7.85e-9,
            e=205000.0,
            nu=0.29,
            a=240.0,
            b=400.0,
            n=0.50,
            chard=0.10,
            sig_max=700.0,
        )
        deck.prop_shell(pid=1, title="PropShell", ishell=1, thick=1.0)
        deck.part(pid=1, title="PartShell", prop_id=1, mat_id=1)

        nodes = [
            (1, 0.0, 0.0, 0.0), (2, 5.0, 0.0, 0.0), (3, 5.0, 5.0, 0.0), (4, 0.0, 5.0, 0.0),
        ]
        deck.node(nodes)
        deck.shell(1, [[1, 1, 2, 3, 4]])
        deck.grnod_node(1, "PullNodes", [2, 3])
        deck.inivel_tra(1, "StretchKick", [20.0, 0.0, 0.0], 1)

        rad_file = tmp_path / f"{name}_0000.rad"
        deck.write(str(rad_file))

        # Starter
        m_start = run_starter(str(rad_file))
        assert m_start is not None

        # Leg 1: run to 1.0e-6 s
        eng1_path = tmp_path / f"{name}_0001.rad"
        eng1_path.write_text(f"/RUN/{name}/1\n1.0e-6\n/DT\n0.5 0.0\n/PRINT/-1\n", encoding="utf-8")
        m_leg1 = run_engine(str(eng1_path))
        assert m_leg1 is not None

        # Leg 2: resume and run to 2.0e-6 s
        eng2_path = tmp_path / f"{name}_0002.rad"
        eng2_path.write_text(f"/RUN/{name}/2\n2.0e-6\n/DT\n0.5 0.0\n/PRINT/-1\n", encoding="utf-8")
        m_leg2 = run_engine(str(eng2_path))
        assert m_leg2 is not None
        assert m_leg2.engine_state.cycle > m_leg1.engine_state.cycle

        shg2 = dict(m_leg2.element_groups())["shells"]
        assert "mat_extra" in shg2.state
        assert "sigb48" in shg2.state["mat_extra"]
        assert "eps48" in shg2.state["mat_extra"]
        assert "epsd48" in shg2.state["mat_extra"]
        assert "off48" in shg2.state["mat_extra"]
