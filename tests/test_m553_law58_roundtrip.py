"""
Milestone M553: /MAT/LAW58 (/MAT/FABR_A, /MAT/FABRIC_A)
Exhaustive Roundtrip, Negative Validation & Restart (.rst) Serialization Auditor Suite.

Fortran origins:
  - starter/source/materials/mat/mat058/hm_read_mat58.F
  - starter/source/materials/mat/mat058/cm58in3.F
  - engine/source/materials/mat/mat058/sigeps58c.F
  - engine/source/output/restart/wrrestp.F & rdresb.F
  - config/CFG/radioss2017/MAT/matl58_fabr_a.cfg

Audits:
  1. Fixed-Format Deck Roundtrip:
     - Write /MAT/LAW58 using StarterDeck.mat_law58 with standard 20-column fixed format.
     - Re-parse using read_mat_law58 and parse_starter_deck.
     - Assert precise equality for all fields:
       rho, E1, B1, E2, B2, Flex (f), G0, Gt (gi), phi_lock (alpha), G5, isensor,
       Df, Ds, mu_frot (friction_phi), Arel (m58_zerostress), N1, N2, S1, S2, C4, C5,
       FUN_A1..6, C1..6 (scale4..6).
     - Direct object invocation with MatLaw58 entity.
     - Verification across both Model.mat_law58s AST records and Model.materials runtime entities.
     - Variations: with unloading curves, without unloading curves, pure analytical.

  2. Free-Format Comma-Delimited Roundtrip:
     - Write and re-parse free-format deck blocks for /MAT/LAW58, /MAT/FABR_A, and /MAT/FABRIC_A.
     - Comma-separated with whitespace, compact comma-delimited without whitespace.
     - Consecutive commas (empty fields taking Fortran default values).
     - Cross-dialect roundtrip: free -> parse -> fixed -> parse.

  3. Negative Starter Diagnostics:
     - Density rho <= 0 (error).
     - Young's modulus E1 <= 0 or E2 <= 0 (error).
     - Negative or zero yarn count N1 <= 0, N2 <= 0 (error).
     - Unloading curves without loading curves (ANCMSG 1578, 1579, 1580).
     - Solid elements assigned to LAW58 (ANCMSG 305): bricks, tetras, penta6, pyra5, etc.
     - 1D elements assigned to LAW58 (ANCMSG 306): springs, trusses, beams.
     - _ALLOWED_LAWS dictionary membership and rejection.

  4. Restart (.rst) Serialization:
     - MatLaw58 entity dataclass pickling and unpickling fidelity.
     - Law58Params physics object serialization.
     - Extra shapes verification: yc, yt, fn, tan_phi, sigv_xy, eps58, sigi58, t58 for shells.
     - Material state arrays preserved across write_restart / read_restart contract.
     - Dynamic cycle continuation from restart matching uninterrupted run bitwise or within 10^-12
       both at the element constitutive level and end-to-end engine execution.
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
from pyradioss.engine.engine import run_engine
from pyradioss.elements import shell_bt4, shell_qeph
from pyradioss.input.deck_reader import read_deck
from pyradioss.input.deck_writer import StarterDeck, fmt_float, fmt_int
from pyradioss.input.starter_keywords import parse_starter_deck, read_mat_law58
from pyradioss.materials.law58_fabr_a import (
    Law58Params,
    FabricAMaterial,
    build_law58,
    shell_update_law58,
    sound_speed_shell_law58,
)
from pyradioss import materials
from pyradioss.model.entities import MatLaw58, MatFabrA, MatFabricA, Material
from pyradioss.model.model import Model
from pyradioss.starter.checks import (
    _ALLOWED_LAWS,
    check_mat_law58,
    check_materials,
    check_model,
)
from pyradioss.starter.restart import read_restart, write_restart
from pyradioss.starter.starter import run_starter


# ============================================================================
# Helpers
# ============================================================================

def _parse_deck_str(tmp_path: Path, text: str, name: str = "TEST_LAW58") -> Tuple[Model, MessageLog]:
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
        if not line.startswith("#") and not line.startswith("$") and line.strip():
            cards.append(line)
    return cards


def _assert_law58_all_fields_exact(
    m: MatLaw58,
    expected: Dict[str, Any],
    tol: float = 1e-6,
    mat: Material | None = None,
) -> None:
    """Assert precise equality for all fields:
    rho, E1, B1, E2, B2, Flex (f), G0, Gt (gi), phi_lock (alpha), G5, isensor,
    Df, Ds, mu_frot (friction_phi), Arel (m58_zerostress), N1, N2, S1, S2, C4, C5,
    FUN_A1..6, C1..6 (scale4..6).
    """
    # 1. Density (Card 1)
    if "rho" in expected:
        assert m.rho == pytest.approx(expected["rho"], rel=tol)
        assert m.rho0 == pytest.approx(expected["rho"], rel=tol)
    if "refer_rho" in expected:
        assert m.refer_rho == pytest.approx(expected["refer_rho"], rel=tol)
        assert m.ref_rho == pytest.approx(expected["refer_rho"], rel=tol)
        assert m.rhor == pytest.approx(expected["refer_rho"], rel=tol)

    # 2. Card 2: E1, B1, E2, B2, Flex
    if "e1" in expected:
        assert m.e1 == pytest.approx(expected["e1"], rel=tol)
    if "b1" in expected:
        assert m.b1 == pytest.approx(expected["b1"], rel=tol)
    if "e2" in expected:
        assert m.e2 == pytest.approx(expected["e2"], rel=tol)
    if "b2" in expected:
        assert m.b2 == pytest.approx(expected["b2"], rel=tol)
    if "f" in expected:
        assert m.f == pytest.approx(expected["f"], rel=tol)
        assert m.flex == pytest.approx(expected["f"], rel=tol)

    # 3. Card 3: G0, Gt, Alpha (phi_lock), G5, isensor
    if "g0" in expected:
        assert m.g0 == pytest.approx(expected["g0"], rel=tol)
    if "gi" in expected:
        assert m.gi == pytest.approx(expected["gi"], rel=tol)
        assert m.gt == pytest.approx(expected["gi"], rel=tol)
    if "alpha" in expected:
        assert m.alpha == pytest.approx(expected["alpha"], rel=tol)
        assert m.alphat == pytest.approx(expected["alpha"], rel=tol)
        assert m.phi_lock == pytest.approx(expected["alpha"], rel=tol)
    if "g5" in expected:
        assert m.g5 == pytest.approx(expected["g5"], rel=tol)
        assert m.gsh == pytest.approx(expected["g5"], rel=tol)
    if "isensor" in expected:
        assert m.isensor == expected["isensor"]
        assert m.sensor_id == expected["isensor"]

    # 4. Card 4: Df, Ds, Friction_phi (mu_frot), ZERO_STRESS (Arel)
    if "df" in expected:
        assert m.df == pytest.approx(expected["df"], rel=tol)
    if "ds" in expected:
        assert m.ds == pytest.approx(expected["ds"], rel=tol)
    if "friction_phi" in expected:
        assert m.friction_phi == pytest.approx(expected["friction_phi"], rel=tol)
        assert m.gfrot == pytest.approx(expected["friction_phi"], rel=tol)
        assert m.mu_frot == pytest.approx(expected["friction_phi"], rel=tol)
    if "m58_zerostress" in expected:
        assert m.m58_zerostress == pytest.approx(expected["m58_zerostress"], rel=tol)
        assert m.zero_stress == pytest.approx(expected["m58_zerostress"], rel=tol)
        assert m.arel == pytest.approx(expected["m58_zerostress"], rel=tol)
        assert m.a_rel == pytest.approx(expected["m58_zerostress"], rel=tol)

    # 5. Card 5: N1, N2, S1, S2, C4, C5
    if "n1_warp" in expected:
        assert m.n1_warp == expected["n1_warp"]
        assert m.n1 == expected["n1_warp"]
    if "n2_weft" in expected:
        assert m.n2_weft == expected["n2_weft"]
        assert m.n2 == expected["n2_weft"]
    if "s1" in expected:
        assert m.s1 == pytest.approx(expected["s1"], rel=tol)
    if "s2" in expected:
        assert m.s2 == pytest.approx(expected["s2"], rel=tol)
    if "c4" in expected:
        assert m.c4 == pytest.approx(expected["c4"], rel=tol)
        assert m.flex1 == pytest.approx(expected["c4"], rel=tol)
    if "c5" in expected:
        assert m.c5 == pytest.approx(expected["c5"], rel=tol)
        assert m.flex2 == pytest.approx(expected["c5"], rel=tol)

    # 6. Cards 6-8: Loading curves FUN_A1..3, C1..3
    if "fun_a1" in expected:
        assert m.fun_a1 == expected["fun_a1"]
        assert m.fun_id1 == expected["fun_a1"]
    if "c1" in expected:
        assert m.c1 == pytest.approx(expected["c1"], rel=tol)
        assert m.fscale1 == pytest.approx(expected["c1"], rel=tol)

    if "fun_a2" in expected:
        assert m.fun_a2 == expected["fun_a2"]
        assert m.fun_id2 == expected["fun_a2"]
    if "c2" in expected:
        assert m.c2 == pytest.approx(expected["c2"], rel=tol)
        assert m.fscale2 == pytest.approx(expected["c2"], rel=tol)

    if "fun_a3" in expected:
        assert m.fun_a3 == expected["fun_a3"]
        assert m.fun_id3 == expected["fun_a3"]
    if "c3" in expected:
        assert m.c3 == pytest.approx(expected["c3"], rel=tol)
        assert m.fscale3 == pytest.approx(expected["c3"], rel=tol)

    # 7. Card 9: Unloading curves FUN_A4..6, C4..6 (scale4..6)
    if "fun_a4" in expected:
        assert m.fun_a4 == expected["fun_a4"]
        assert m.fun_id4 == expected["fun_a4"]
    if "scale4" in expected:
        assert m.scale4 == pytest.approx(expected["scale4"], rel=tol)
        assert m.fscale4 == pytest.approx(expected["scale4"], rel=tol)

    if "fun_a5" in expected:
        assert m.fun_a5 == expected["fun_a5"]
        assert m.fun_id5 == expected["fun_a5"]
    if "scale5" in expected:
        assert m.scale5 == pytest.approx(expected["scale5"], rel=tol)
        assert m.fscale5 == pytest.approx(expected["scale5"], rel=tol)

    if "fun_a6" in expected:
        assert m.fun_a6 == expected["fun_a6"]
        assert m.fun_id6 == expected["fun_a6"]
    if "scale6" in expected:
        assert m.scale6 == pytest.approx(expected["scale6"], rel=tol)
        assert m.fscale6 == pytest.approx(expected["scale6"], rel=tol)
        assert m.c6 == pytest.approx(expected["scale6"], rel=tol)

    # Cross-check Material runtime container if provided
    if mat is not None:
        p = mat.params
        assert mat.law == 58
        assert mat.rho0 == pytest.approx(expected.get("rho", m.rho), rel=tol)
        if "e1" in expected:
            assert p["e1"] == pytest.approx(expected["e1"], rel=tol)
            assert p["E1"] == pytest.approx(expected["e1"], rel=tol)
        if "e2" in expected:
            assert p["e2"] == pytest.approx(expected["e2"], rel=tol)
            assert p["E2"] == pytest.approx(expected["e2"], rel=tol)
        if "f" in expected:
            assert p["f"] == pytest.approx(expected["f"], rel=tol)
            assert p["flex"] == pytest.approx(expected["f"], rel=tol)
        if "g0" in expected:
            assert p["g0"] == pytest.approx(expected["g0"], rel=tol)
        if "gi" in expected:
            assert p["gi"] == pytest.approx(expected["gi"], rel=tol)
            assert p["gt"] == pytest.approx(expected["gi"], rel=tol)
        if "alpha" in expected:
            assert p["alpha"] == pytest.approx(expected["alpha"], rel=tol)
            assert p["phi_lock"] == pytest.approx(expected["alpha"], rel=tol)
        if "friction_phi" in expected:
            assert p["friction_phi"] == pytest.approx(expected["friction_phi"], rel=tol)
            assert p["mu_frot"] == pytest.approx(expected["friction_phi"], rel=tol)
        if "m58_zerostress" in expected:
            assert p["m58_zerostress"] == pytest.approx(expected["m58_zerostress"], rel=tol)
            assert p["arel"] == pytest.approx(expected["m58_zerostress"], rel=tol)
            assert p["a_rel"] == pytest.approx(expected["m58_zerostress"], rel=tol)
        if "n1_warp" in expected:
            assert p["n1_warp"] == expected["n1_warp"]
            assert p["n1"] == expected["n1_warp"]
        if "n2_weft" in expected:
            assert p["n2_weft"] == expected["n2_weft"]
            assert p["n2"] == expected["n2_weft"]
        if "c4" in expected:
            assert p["c4"] == pytest.approx(expected["c4"], rel=tol)
            assert p["flex1"] == pytest.approx(expected["c4"], rel=tol)
        if "c5" in expected:
            assert p["c5"] == pytest.approx(expected["c5"], rel=tol)
            assert p["flex2"] == pytest.approx(expected["c5"], rel=tol)
        if "scale6" in expected:
            assert p["scale6"] == pytest.approx(expected["scale6"], rel=tol)
            assert p["c6"] == pytest.approx(expected["scale6"], rel=tol)


# ============================================================================
# 1. Fixed-Format Deck Roundtrip
# ============================================================================

class TestLaw58FixedFormatRoundtrip:
    """Audit fixed-format deck generation, parsing, and exact equality for all 23+ fields."""

    def test_fixed_format_all_fields_exact(self, tmp_path: Path):
        """Write /MAT/LAW58 in standard 20-col fixed format, re-parse and assert precise equality."""
        deck = StarterDeck("LAW58_FIXED_ALL")
        params = {
            "rho": 1.35e-6,
            "refer_rho": 1.35e-6,
            "e1": 2500.0,
            "b1": 12.5,
            "e2": 1800.0,
            "b2": 9.8,
            "f": 0.035,
            "g0": 65.0,
            "gi": 450.0,
            "alpha": 38.0,
            "g5": 25.0,
            "isensor": 17,
            "df": 0.045,
            "ds": 0.08,
            "friction_phi": 18.5,
            "m58_zerostress": 0.85,
            "n1_warp": 3,
            "n2_weft": 4,
            "s1": 0.18,
            "s2": 0.14,
            "c4": 1.45,
            "c5": 1.35,
            "fun_a1": 101,
            "c1": 1.2,
            "fun_a2": 102,
            "c2": 1.15,
            "fun_a3": 103,
            "c3": 0.95,
            "fun_a4": 104,
            "scale4": 1.1,
            "fun_a5": 105,
            "scale5": 1.05,
            "fun_a6": 106,
            "scale6": 0.9,
        }
        deck.mat_law58(
            mid=1,
            title="Exhaustive Fabric Material",
            fixed_format=True,
            **params,
        )

        rendered = deck.render()
        cards = _get_material_cards(rendered)
        # Verify card count: Card 1 (density), Card 2 (moduli), Card 3 (shear),
        # Card 4 (damping/friction), Card 5 (yarns/crimp), Cards 6-8 (loading curves), Card 9 (unloading)
        assert len(cards) == 9

        # Card 1: 20-col float rho and refer_rho
        assert len(cards[0]) >= 40
        # Card 2: 5 20-col floats
        assert len(cards[1]) >= 100
        # Card 3: 4 20-col floats + blank10 + int10
        assert len(cards[2]) >= 100
        # Card 4: 3 20-col floats + blank20 + 20-col float
        assert len(cards[3]) >= 100
        # Card 5: 2 int10 + 4 20-col floats
        assert len(cards[4]) >= 100

        # Parse rendered deck
        model, log = _parse_deck_str(tmp_path, rendered, "fixed_all")
        assert not log.has_errors
        assert 1 in model.mat_law58s
        assert 1 in model.materials

        m = model.mat_law58s[1]
        mat = model.materials[1]
        assert m.id == 1
        assert m.title == "Exhaustive Fabric Material"
        _assert_law58_all_fields_exact(m, params, tol=1e-6, mat=mat)

    def test_fixed_format_using_domain_aliases(self, tmp_path: Path):
        """Invoke StarterDeck.mat_law58 using domain aliases (phi_lock, mu_frot, arel, flex, gt, n1, n2, c6)."""
        deck = StarterDeck("LAW58_ALIASES")
        deck.mat_law58(
            mid=2,
            title="Aliased Fabric",
            rho=1.2e-6,
            ref_rho=1.2e-6,
            e1=2000.0,
            b1=10.0,
            e2=1500.0,
            b2=8.0,
            flex=0.03,
            g0=50.0,
            gt=400.0,
            phi_lock=35.0,
            gsh=20.0,
            sensor_id=12,
            df=0.04,
            ds=0.05,
            mu_frot=15.0,
            arel=0.80,
            n1=2,
            n2=3,
            s1=0.15,
            s2=0.12,
            flex1=1.25,
            flex2=1.20,
            fun_id1=201,
            fscale1=1.0,
            fun_id2=202,
            fscale2=1.0,
            fun_id3=203,
            fscale3=1.0,
            fun_id4=204,
            fscale4=1.05,
            fun_id5=205,
            fscale5=1.02,
            fun_id6=206,
            c6=0.98,
            fixed_format=True,
        )

        model, log = _parse_deck_str(tmp_path, deck.render(), "aliases_test")
        assert not log.has_errors
        m = model.mat_law58s[2]
        mat = model.materials[2]

        expected = {
            "rho": 1.2e-6,
            "refer_rho": 1.2e-6,
            "e1": 2000.0,
            "b1": 10.0,
            "e2": 1500.0,
            "b2": 8.0,
            "f": 0.03,
            "g0": 50.0,
            "gi": 400.0,
            "alpha": 35.0,
            "g5": 20.0,
            "isensor": 12,
            "df": 0.04,
            "ds": 0.05,
            "friction_phi": 15.0,
            "m58_zerostress": 0.80,
            "n1_warp": 2,
            "n2_weft": 3,
            "s1": 0.15,
            "s2": 0.12,
            "c4": 1.25,
            "c5": 1.20,
            "fun_a1": 201,
            "c1": 1.0,
            "fun_a2": 202,
            "c2": 1.0,
            "fun_a3": 203,
            "c3": 1.0,
            "fun_a4": 204,
            "scale4": 1.05,
            "fun_a5": 205,
            "scale5": 1.02,
            "fun_a6": 206,
            "scale6": 0.98,
        }
        _assert_law58_all_fields_exact(m, expected, tol=1e-6, mat=mat)

    def test_fixed_format_direct_object_invocation(self, tmp_path: Path):
        """Pass pre-constructed MatLaw58 dataclass entity directly to StarterDeck.mat_law58."""
        m_in = MatLaw58(
            id=3,
            title="Direct MatLaw58 Object",
            rho=1.15e-6,
            refer_rho=1.15e-6,
            e1=2200.0,
            b1=11.0,
            e2=1600.0,
            b2=7.5,
            f=0.025,
            g0=48.0,
            gi=380.0,
            alpha=32.0,
            g5=18.0,
            isensor=9,
            df=0.035,
            ds=0.06,
            friction_phi=14.0,
            m58_zerostress=0.75,
            n1_warp=2,
            n2_weft=2,
            s1=0.16,
            s2=0.13,
            c4=1.30,
            c5=1.22,
            fun_a1=301,
            c1=1.05,
            fun_a2=302,
            c2=1.02,
            fun_a3=303,
            c3=0.98,
            fun_a4=304,
            scale4=1.04,
            fun_a5=305,
            scale5=1.01,
            fun_a6=306,
            scale6=0.96,
        )

        deck = StarterDeck("OBJ_ROUNDTRIP")
        deck.mat_law58(m_in)

        model, log = _parse_deck_str(tmp_path, deck.render(), "obj_roundtrip")
        assert not log.has_errors
        m_out = model.mat_law58s[3]
        _assert_law58_all_fields_exact(m_out, m_in.__dict__, tol=1e-6, mat=model.materials[3])

    def test_fixed_format_variations_without_unloading_and_pure_analytical(self, tmp_path: Path):
        """Test fixed format variations: loading curves only, and pure analytical formulation."""
        deck = StarterDeck("VARIATIONS")
        # 1. Loading curves only (fun_a1..3 active, fun_a4..6 zero)
        deck.mat_law58(
            mid=4,
            title="Loading Only",
            rho=1.25e-6,
            e1=1900.0,
            e2=1400.0,
            g0=45.0,
            gi=350.0,
            alpha=30.0,
            fun_a1=401,
            c1=1.0,
            fun_a2=402,
            c2=1.0,
            fun_a3=403,
            c3=1.0,
            fixed_format=True,
        )
        # 2. Pure analytical formulation (no tabulated curves)
        deck.mat_law58(
            mid=5,
            title="Pure Analytical",
            rho=1.10e-6,
            e1=1700.0,
            b1=8.0,
            e2=1200.0,
            b2=6.0,
            f=0.02,
            g0=40.0,
            gi=300.0,
            alpha=28.0,
            fixed_format=True,
        )

        model, log = _parse_deck_str(tmp_path, deck.render(), "variations_test")
        assert not log.has_errors

        # Check loading only: cards 6-8 present, card 9 omitted, fun_a4..6 default to 0
        m4 = model.mat_law58s[4]
        assert m4.fun_a1 == 401
        assert m4.fun_a2 == 402
        assert m4.fun_a3 == 403
        assert m4.fun_a4 == 0
        assert m4.fun_a5 == 0
        assert m4.fun_a6 == 0

        # Check pure analytical: cards 6-9 omitted, fun_a1..6 default to 0
        m5 = model.mat_law58s[5]
        assert m5.e1 == pytest.approx(1700.0)
        assert m5.b1 == pytest.approx(8.0)
        assert m5.fun_a1 == 0
        assert m5.fun_a4 == 0


# ============================================================================
# 2. Free-Format Comma-Delimited Roundtrip
# ============================================================================

class TestLaw58FreeFormatRoundtrip:
    """Audit free-format comma-delimited deck writing and parsing across all LAW58 aliases."""

    def test_free_format_law58_comma_delimited(self, tmp_path: Path):
        """Write and re-parse free-format comma-delimited /MAT/LAW58 block."""
        deck = StarterDeck("FREE_LAW58")
        params = {
            "rho": 1.28e-6,
            "refer_rho": 1.28e-6,
            "e1": 2350.0,
            "b1": 11.5,
            "e2": 1750.0,
            "b2": 9.2,
            "f": 0.032,
            "g0": 62.0,
            "gi": 430.0,
            "alpha": 36.5,
            "g5": 22.0,
            "isensor": 15,
            "df": 0.042,
            "ds": 0.075,
            "friction_phi": 17.5,
            "m58_zerostress": 0.82,
            "n1_warp": 2,
            "n2_weft": 3,
            "s1": 0.17,
            "s2": 0.13,
            "c4": 1.40,
            "c5": 1.30,
            "fun_a1": 501,
            "c1": 1.15,
            "fun_a2": 502,
            "c2": 1.10,
            "fun_a3": 503,
            "c3": 0.92,
            "fun_a4": 504,
            "scale4": 1.08,
            "fun_a5": 505,
            "scale5": 1.03,
            "fun_a6": 506,
            "scale6": 0.88,
        }
        deck.mat_law58(
            mid=10,
            title="Free Format LAW58",
            fixed_format=False,
            comma=True,
            **params,
        )

        rendered = deck.render()
        assert "/MAT/LAW58/10" in rendered
        assert "," in rendered

        model, log = _parse_deck_str(tmp_path, rendered, "free_law58")
        assert not log.has_errors
        assert 10 in model.mat_law58s
        _assert_law58_all_fields_exact(model.mat_law58s[10], params, tol=1e-5, mat=model.materials[10])

    def test_free_format_mat_fabr_a_comma_delimited(self, tmp_path: Path):
        """Write and re-parse free-format comma-delimited /MAT/FABR_A block."""
        deck = StarterDeck("FREE_FABR_A")
        deck.mat_fabr_a(
            mid=20,
            title="Fabric A Free Deck",
            rho=1.18e-6,
            e1=2100.0,
            b1=10.5,
            e2=1550.0,
            b2=8.5,
            flex=0.028,
            g0=55.0,
            gt=390.0,
            phi_lock=34.0,
            gsh=21.0,
            sensor_id=8,
            df=0.038,
            ds=0.065,
            mu_frot=16.0,
            arel=0.78,
            n1=3,
            n2=3,
            s1=0.16,
            s2=0.12,
            c4=1.35,
            c5=1.25,
            fun_a1=601,
            c1=1.10,
            fun_a2=602,
            c2=1.05,
            fun_a3=603,
            c3=0.90,
            fun_a4=604,
            scale4=1.05,
            fun_a5=605,
            scale5=1.02,
            fun_a6=606,
            scale6=0.85,
            fixed_format=False,
            comma=True,
        )

        rendered = deck.render()
        assert "/MAT/FABR_A/20" in rendered
        assert "," in rendered

        model, log = _parse_deck_str(tmp_path, rendered, "free_fabr_a")
        assert not log.has_errors
        assert 20 in model.mat_law58s
        assert 20 in model.materials

        m = model.mat_law58s[20]
        assert m.id == 20
        assert m.e1 == pytest.approx(2100.0, rel=1e-5)
        assert m.phi_lock == pytest.approx(34.0, rel=1e-5)
        assert m.mu_frot == pytest.approx(16.0, rel=1e-5)
        assert m.arel == pytest.approx(0.78, rel=1e-5)
        assert m.fun_a1 == 601
        assert m.scale6 == pytest.approx(0.85, rel=1e-5)

    def test_free_format_mat_fabric_a_comma_delimited(self, tmp_path: Path):
        """Write and re-parse free-format comma-delimited /MAT/FABRIC_A block."""
        deck = StarterDeck("FREE_FABRIC_A")
        deck.mat_fabric_a(
            mid=30,
            title="Fabric A Full Synonym Deck",
            rho=1.22e-6,
            e1=2250.0,
            b1=11.0,
            e2=1650.0,
            b2=9.0,
            flex=0.030,
            g0=58.0,
            gt=410.0,
            phi_lock=35.5,
            gsh=23.0,
            sensor_id=14,
            df=0.040,
            ds=0.070,
            mu_frot=17.0,
            arel=0.80,
            n1=2,
            n2=4,
            s1=0.17,
            s2=0.14,
            c4=1.38,
            c5=1.28,
            fun_a1=701,
            c1=1.12,
            fun_a2=702,
            c2=1.08,
            fun_a3=703,
            c3=0.94,
            fixed_format=False,
            comma=True,
        )

        rendered = deck.render()
        assert "/MAT/FABRIC_A/30" in rendered
        assert "," in rendered

        model, log = _parse_deck_str(tmp_path, rendered, "free_fabric_a")
        assert not log.has_errors
        assert 30 in model.mat_law58s
        assert 30 in model.materials

        m = model.mat_law58s[30]
        assert m.id == 30
        assert m.e1 == pytest.approx(2250.0, rel=1e-5)
        assert m.e2 == pytest.approx(1650.0, rel=1e-5)
        assert m.fun_a1 == 701
        assert m.fun_a4 == 0  # no unloading curves

    def test_free_format_consecutive_commas_and_whitespace(self, tmp_path: Path):
        """Verify robust parsing of compact free format with consecutive commas and omitted optional fields."""
        raw_rad = """/BEGIN
FREE_COMMAS_TEST
/MAT/LAW58/40
Sparse Free Format Fabric
1.25e-6
1800.0, 5.0, 1200.0, 4.0, 0.02
45.0, 350.0, 0.05, 20.0, 10
0.03, 0.04, 12.0, 0.0, 0.85
2, 2, 0.15, 0.12, 1.25, 1.25
801, 1.0
802, 1.0
803, 1.0
"""
        model, log = _parse_deck_str(tmp_path, raw_rad, "free_commas")
        assert not log.has_errors
        assert 40 in model.mat_law58s
        m = model.mat_law58s[40]
        assert m.rho == pytest.approx(1.25e-6)
        assert m.e1 == pytest.approx(1800.0)
        assert m.b1 == pytest.approx(5.0)
        assert m.g0 == pytest.approx(45.0)
        assert m.gi == pytest.approx(350.0)
        assert m.isensor == 10
        assert m.m58_zerostress == pytest.approx(0.85)
        assert m.n1_warp == 2
        assert m.fun_a1 == 801

    def test_cross_format_roundtrip_free_to_fixed(self, tmp_path: Path):
        """Cross-dialect roundtrip: write free format -> parse -> write fixed format -> parse -> assert exact."""
        d_free = StarterDeck("X_ROUNDTRIP")
        d_free.mat_law58(
            mid=50,
            title="Cross Format Fabric",
            rho=1.30e-6,
            e1=2400.0,
            b1=12.0,
            e2=1700.0,
            b2=9.0,
            flex=0.033,
            g0=60.0,
            gt=420.0,
            phi_lock=37.0,
            g5=24.0,
            isensor=16,
            df=0.044,
            ds=0.078,
            mu_frot=18.0,
            arel=0.84,
            n1=3,
            n2=4,
            s1=0.175,
            s2=0.135,
            c4=1.42,
            c5=1.32,
            fun_a1=901,
            c1=1.18,
            fun_a2=902,
            c2=1.12,
            fun_a3=903,
            c3=0.96,
            fun_a4=904,
            scale4=1.09,
            fun_a5=905,
            scale5=1.04,
            fun_a6=906,
            scale6=0.89,
            fixed_format=False,
            comma=True,
        )

        model1, log1 = _parse_deck_str(tmp_path, d_free.render(), "step1_free")
        assert not log1.has_errors
        m1 = model1.mat_law58s[50]

        # Convert to fixed format
        d_fixed = StarterDeck("STEP2_FIXED")
        d_fixed.mat_law58(m1, fixed_format=True)

        model2, log2 = _parse_deck_str(tmp_path, d_fixed.render(), "step2_fixed")
        assert not log2.has_errors
        m2 = model2.mat_law58s[50]

        _assert_law58_all_fields_exact(m2, m1.__dict__, tol=1e-5, mat=model2.materials[50])


# ============================================================================
# 3. Negative Starter Diagnostics
# ============================================================================

class TestLaw58NegativeDiagnostics:
    """Audit starter validation diagnostics on out-of-bounds parameters and illegal configurations."""

    @pytest.mark.parametrize("bad_rho", [0.0, -1.0e-6])
    def test_negative_or_zero_density(self, bad_rho: float):
        """Density rho0 <= 0 must be rejected with diagnostic error."""
        m = MatLaw58(id=1, rho=bad_rho, e1=1500.0, e2=800.0)
        log = MessageLog()
        check_mat_law58(mat=m, log=log)
        assert log.has_errors
        assert any("density" in str(e).lower() or "rho" in str(e).lower() for e in log.errors)

    @pytest.mark.parametrize("bad_e1", [0.0, -1000.0])
    def test_negative_or_zero_youngs_modulus_e1(self, bad_e1: float):
        """Young's modulus E1 <= 0 must be rejected with diagnostic error."""
        m = MatLaw58(id=1, rho=1.2e-6, e1=bad_e1, e2=800.0)
        log = MessageLog()
        check_mat_law58(mat=m, log=log)
        assert log.has_errors
        assert any("e1" in str(e).lower() or "young" in str(e).lower() for e in log.errors)

    @pytest.mark.parametrize("bad_e2", [0.0, -800.0])
    def test_negative_or_zero_youngs_modulus_e2(self, bad_e2: float):
        """Young's modulus E2 <= 0 must be rejected with diagnostic error."""
        m = MatLaw58(id=1, rho=1.2e-6, e1=1500.0, e2=bad_e2)
        log = MessageLog()
        check_mat_law58(mat=m, log=log)
        assert log.has_errors
        assert any("e2" in str(e).lower() or "young" in str(e).lower() for e in log.errors)

    @pytest.mark.parametrize("bad_n1", [0, -1, -5])
    def test_negative_or_zero_yarn_count_n1(self, bad_n1: int):
        """Fiber density / yarn count in warp direction N1 <= 0 must be rejected."""
        m = MatLaw58(id=1, rho=1.2e-6, e1=1500.0, e2=800.0, n1_warp=bad_n1)
        log = MessageLog()
        check_mat_law58(mat=m, log=log)
        assert log.has_errors
        assert any("n1" in str(e).lower() or "yarn count" in str(e).lower() for e in log.errors)

    @pytest.mark.parametrize("bad_n2", [0, -1, -4])
    def test_negative_or_zero_yarn_count_n2(self, bad_n2: int):
        """Fiber density / yarn count in weft direction N2 <= 0 must be rejected."""
        m = MatLaw58(id=1, rho=1.2e-6, e1=1500.0, e2=800.0, n2_weft=bad_n2)
        log = MessageLog()
        check_mat_law58(mat=m, log=log)
        assert log.has_errors
        assert any("n2" in str(e).lower() or "yarn count" in str(e).lower() for e in log.errors)

    def test_unloading_curves_without_loading_curves(self):
        """Unloading curves without corresponding loading curves must trigger ANCMSG 1578, 1579, 1580."""
        # 1. FUN_A4 active (warp unloading) but FUN_A1 missing -> ANCMSG 1578
        m1 = MatLaw58(id=1, rho=1.2e-6, e1=1500.0, e2=800.0, fun_a1=0, fun_a4=104)
        log1 = MessageLog()
        check_mat_law58(mat=m1, log=log1)
        assert log1.has_errors
        assert any("1578" in str(e) or "fun_a1" in str(e).lower() for e in log1.errors)

        # 2. FUN_A5 active (weft unloading) but FUN_A2 missing -> ANCMSG 1579
        m2 = MatLaw58(id=2, rho=1.2e-6, e1=1500.0, e2=800.0, fun_a1=101, fun_a2=0, fun_a5=105)
        log2 = MessageLog()
        check_mat_law58(mat=m2, log=log2)
        assert log2.has_errors
        assert any("1579" in str(e) or "fun_a2" in str(e).lower() for e in log2.errors)

        # 3. FUN_A6 active (shear unloading) but FUN_A3 missing -> ANCMSG 1580
        m3 = MatLaw58(id=3, rho=1.2e-6, e1=1500.0, e2=800.0, fun_a1=101, fun_a2=102, fun_a3=0, fun_a6=106)
        log3 = MessageLog()
        check_mat_law58(mat=m3, log=log3)
        assert log3.has_errors
        assert any("1580" in str(e) or "fun_a3" in str(e).lower() for e in log3.errors)

    def test_solid_elements_assigned_to_law58(self):
        """Assigning LAW58 to solid element formulations must be rejected with ANCMSG 305."""
        class MockElem:
            def __init__(self, mat_id: int):
                self.mat_id = mat_id

        class MockGrp:
            def __init__(self, elems: List[MockElem]):
                self._elems = elems
            def values(self):
                return self._elems

        solid_families = ("bricks", "tetras", "penta6", "pyra5", "bricks_heph", "bric20s", "tetra10s")
        for fam in solid_families:
            assert 58 not in _ALLOWED_LAWS.get(fam, set()), f"58 illegally allowed in {fam}"
            assert "LAW58" not in _ALLOWED_LAWS.get(fam, set()), f"LAW58 illegally allowed in {fam}"

            model = Model()
            m = MatLaw58(id=1, rho=1.2e-6, e1=1500.0, e2=800.0)
            model.materials[1] = m
            model.element_groups = lambda f=fam: [(f, MockGrp([MockElem(1)]))]
            log = MessageLog()
            check_mat_law58(model=model, mat_id=1, mat=m, log=log)
            assert log.has_errors, f"Expected rejection of LAW58 on solid family {fam}"
            assert any("305" in str(e) or "solid" in str(e).lower() for e in log.errors)

    def test_1d_elements_assigned_to_law58(self):
        """Assigning LAW58 to 1D element formulations must be rejected with ANCMSG 306."""
        class MockElem:
            def __init__(self, mat_id: int):
                self.mat_id = mat_id

        class MockGrp:
            def __init__(self, elems: List[MockElem]):
                self._elems = elems
            def values(self):
                return self._elems

        one_d_families = ("springs", "trusses", "beams")
        for fam in one_d_families:
            allowed = _ALLOWED_LAWS.get(fam) or set()
            assert 58 not in allowed, f"58 illegally allowed in {fam}"
            assert "LAW58" not in allowed, f"LAW58 illegally allowed in {fam}"

            model = Model()
            m = MatLaw58(id=1, rho=1.2e-6, e1=1500.0, e2=800.0)
            model.materials[1] = m
            model.element_groups = lambda f=fam: [(f, MockGrp([MockElem(1)]))]
            log = MessageLog()
            check_mat_law58(model=model, mat_id=1, mat=m, log=log)
            assert log.has_errors, f"Expected rejection of LAW58 on 1D family {fam}"
            assert any("306" in str(e) or "1d" in str(e).lower() for e in log.errors)

    def test_compatible_shell_elements_accepted(self):
        """Shell families (shells, shells_qbat, shells_qeph, sh3n, quads) must accept LAW58 without errors."""
        shell_families = ("shells", "shells_qbat", "shells_qeph", "sh3n", "quads")
        for fam in shell_families:
            allowed = _ALLOWED_LAWS.get(fam, set())
            assert 58 in allowed
            assert "LAW58" in allowed
            assert "FABR_A" in allowed
            assert "FABRIC_A" in allowed

        model = Model()
        m = MatLaw58(id=1, rho=1.2e-6, e1=1500.0, e2=800.0, n1_warp=2, n2_weft=2)
        model.materials[1] = m
        log = MessageLog()
        check_mat_law58(model=model, mat_id=1, mat=m, log=log)
        assert not log.has_errors


# ============================================================================
# 4. Restart (.rst) Serialization
# ============================================================================

class TestLaw58RestartSerialization:
    """Audit serialization and restart preservation for /MAT/LAW58 persistent material arrays and dynamic continuation."""

    def test_matlaw58_and_law58params_pickle_fidelity(self):
        """Pickle and unpickle MatLaw58 and Law58Params verifying all attributes survive bitwise."""
        m_orig = MatLaw58(
            id=58,
            title="Serialized Fabric",
            rho=1.3e-6,
            refer_rho=1.3e-6,
            e1=2400.0,
            b1=12.0,
            e2=1800.0,
            b2=9.0,
            f=0.035,
            g0=60.0,
            gi=420.0,
            alpha=35.0,
            g5=22.0,
            isensor=18,
            df=0.045,
            ds=0.08,
            friction_phi=18.0,
            m58_zerostress=0.85,
            n1_warp=2,
            n2_weft=3,
            s1=0.18,
            s2=0.14,
            c4=1.45,
            c5=1.35,
            fun_a1=101,
            c1=1.2,
            fun_a2=102,
            c2=1.1,
            fun_a3=103,
            c3=0.95,
            fun_a4=104,
            scale4=1.08,
            fun_a5=105,
            scale5=1.04,
            fun_a6=106,
            scale6=0.92,
        )
        data = pickle.dumps(m_orig)
        m_rest: MatLaw58 = pickle.loads(data)

        assert m_rest.id == m_orig.id
        assert m_rest.title == m_orig.title
        assert m_rest.rho == m_orig.rho
        assert m_rest.e1 == m_orig.e1
        assert m_rest.phi_lock == m_orig.phi_lock
        assert m_rest.mu_frot == m_orig.mu_frot
        assert m_rest.arel == m_orig.arel
        assert m_rest.c6 == m_orig.c6
        assert m_rest.flex1 == m_orig.flex1
        assert m_rest.flex2 == m_orig.flex2

        # Physics parameter dataclass
        p_orig = Law58Params(
            rho0=1.3e-6,
            e1=2400.0,
            e2=1800.0,
            g0=60.0,
            gt=420.0,
            alphat=35.0,
            gfrot=18.0,
            zero_stress=0.85,
            n1=2,
            n2=3,
        )
        p_rest: Law58Params = pickle.loads(pickle.dumps(p_orig))
        assert p_rest.rho0 == p_orig.rho0
        assert p_rest.e1 == p_orig.e1
        assert p_rest.alphat == p_orig.alphat
        assert p_rest.gfrot == p_orig.gfrot

    def test_material_state_arrays_rst_preservation(self, tmp_path: Path):
        """Material state arrays: yc, yt, fn, tan_phi, sigv_xy, eps58 must survive write_restart and read_restart."""
        deck = StarterDeck("RST_LAW58_ARRAYS")
        deck.mat_law58(
            mid=1,
            title="StateArrayFabric",
            rho=1.2e-6,
            e1=2000.0,
            e2=1500.0,
            g0=50.0,
            gt=400.0,
            phi_lock=35.0,
            g5=20.0,
            df=0.04,
            ds=0.06,
            mu_frot=15.0,
            arel=0.80,
            n1=2,
            n2=2,
        )
        deck.prop_shell(pid=1, title="FabricProp", ishell=1, thick=1.0)
        deck.part(pid=1, title="FabricPart", prop_id=1, mat_id=1)
        deck.node([
            (1, 0.0, 0.0, 0.0),
            (2, 10.0, 0.0, 0.0),
            (3, 10.0, 10.0, 0.0),
            (4, 0.0, 10.0, 0.0),
        ])
        deck.shell(1, [[1, 1, 2, 3, 4]])

        rad_path = tmp_path / "rst_arrays_0000.rad"
        deck.write(str(rad_path))

        model = run_starter(str(rad_path))
        assert model is not None

        groups = dict(model.element_groups())
        assert "shells" in groups
        shg = groups["shells"]
        st = shg.state
        nip = 3  # standard 3 layers

        # Synthesize realistic persistent material state arrays
        yc_synth = np.array([[-0.0015, -0.0012, -0.0009]])
        yt_synth = np.array([[0.0018, 0.0014, 0.0010]])
        fn_synth = np.array([[24.5, 18.2, 12.1]])
        tan_phi_synth = np.array([[0.145, 0.120, 0.095]])
        sigv_xy_synth = np.array([[4.25, 3.80, 3.10]])
        eps58_synth = np.zeros((1, nip, 3))
        eps58_synth[0, :, 0] = [0.035, 0.020, 0.005]
        eps58_synth[0, :, 1] = [0.025, 0.015, 0.002]
        eps58_synth[0, :, 2] = [0.012, 0.008, 0.001]
        epsp_synth = np.zeros((1, nip))

        st["mat_extra"]["yc"] = yc_synth.copy()
        st["mat_extra"]["yt"] = yt_synth.copy()
        st["mat_extra"]["fn"] = fn_synth.copy()
        st["mat_extra"]["tan_phi"] = tan_phi_synth.copy()
        st["mat_extra"]["sigv_xy"] = sigv_xy_synth.copy()
        st["mat_extra"]["eps58"] = eps58_synth.copy()
        st["epsp"] = epsp_synth.copy()

        # Write restart .rst
        rst_path = tmp_path / "rst_arrays_0001.rst"
        engine_dict = {
            "cycle": 150,
            "t": 1.5e-5,
            "dt": 1.0e-7,
            "energies": {"internal": 12.345, "kinetic": 67.890},
        }
        write_restart(model, str(rst_path), engine=engine_dict)

        # Restore from restart .rst
        rest_model, rest_engine = read_restart(str(rst_path))
        assert rest_engine["cycle"] == 150
        assert rest_engine["t"] == pytest.approx(1.5e-5)
        assert rest_engine["energies"]["internal"] == pytest.approx(12.345)

        rest_shg = dict(rest_model.element_groups())["shells"]
        rest_extra = rest_shg.state["mat_extra"]

        np.testing.assert_array_equal(
            rest_extra["yc"],
            yc_synth,
            err_msg="Yarn deflection yc did not survive restart serialization exactly",
        )
        np.testing.assert_array_equal(
            rest_extra["yt"],
            yt_synth,
            err_msg="Yarn deflection yt did not survive restart serialization exactly",
        )
        np.testing.assert_array_equal(
            rest_extra["fn"],
            fn_synth,
            err_msg="Contact force Fn did not survive restart serialization exactly",
        )
        np.testing.assert_array_equal(
            rest_extra["tan_phi"],
            tan_phi_synth,
            err_msg="Shear angle tan_phi did not survive restart serialization exactly",
        )
        np.testing.assert_array_equal(
            rest_extra["sigv_xy"],
            sigv_xy_synth,
            err_msg="Viscous/friction state sigv_xy did not survive restart serialization exactly",
        )
        np.testing.assert_array_equal(
            rest_extra["eps58"],
            eps58_synth,
            err_msg="Total strain eps58 did not survive restart serialization exactly",
        )
        np.testing.assert_array_equal(
            rest_shg.state["epsp"],
            epsp_synth,
            err_msg="Plastic strain epsp did not survive restart serialization exactly",
        )

    def test_constitutive_cycle_restart_continuation(self):
        """Run multi-cycle shell_update_law58 and verify pickled continuation matches uninterrupted within 10^-12."""
        mat = make_test_material_law58(
            e1=2500.0,
            e2=1800.0,
            g0=60.0,
            gt=400.0,
            alphat=35.0,
            df=0.05,
            ds=0.08,
            gfrot=18.0,
            arel=0.85,
            n1=2,
            n2=2,
        )

        n_steps = 12
        dt = 1.0e-5
        # Strain increments with warp tension, weft compression, and trellis shear
        deps_steps = [
            np.array([[0.002, -0.001, 0.0015]]),
            np.array([[0.003, -0.0015, 0.0020]]),
            np.array([[0.0025, 0.0010, 0.0025]]),
            np.array([[0.0015, 0.0020, 0.0030]]),
            np.array([[0.0010, 0.0015, 0.0035]]),
            np.array([[-0.001, -0.001, 0.0020]]),
            np.array([[-0.002, 0.001, 0.0015]]),
            np.array([[0.002, 0.002, 0.0010]]),
            np.array([[0.003, 0.001, -0.0015]]),
            np.array([[0.0015, -0.001, -0.0020]]),
            np.array([[0.0010, -0.0015, -0.0025]]),
            np.array([[0.0005, 0.0005, -0.0010]]),
        ]

        # 1. Uninterrupted run
        extra_unc: Dict[str, Any] = {}
        sig_unc = np.zeros((1, 3))
        epsp_unc = np.zeros(1)

        for step in range(n_steps):
            sig_unc, epsp_unc = shell_update_law58(
                mat, sig_unc, deps_steps[step], epsp=epsp_unc, dt=dt, extra=extra_unc
            )

        # 2. Chained run: Leg 1 (steps 0..5)
        extra_chn: Dict[str, Any] = {}
        sig_chn = np.zeros((1, 3))
        epsp_chn = np.zeros(1)

        split = 6
        for step in range(split):
            sig_chn, epsp_chn = shell_update_law58(
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
            sig_res, epsp_res = shell_update_law58(
                mat, sig_res, deps_steps[step], epsp=epsp_res, dt=dt, extra=extra_res
            )

        # Verify exact match between uninterrupted and resumed run
        np.testing.assert_allclose(
            sig_res,
            sig_unc,
            rtol=1e-12,
            atol=1e-12,
            err_msg="Stress tensor mismatch between uninterrupted and restarted continuation",
        )
        np.testing.assert_allclose(
            epsp_res,
            epsp_unc,
            rtol=1e-12,
            atol=1e-12,
            err_msg="Plastic strain mismatch between uninterrupted and restarted continuation",
        )
        # Material state arrays
        for arr_name in ("yc", "yt", "fn", "tan_phi", "sigv_xy", "eps58"):
            np.testing.assert_allclose(
                extra_res[arr_name],
                extra_unc[arr_name],
                rtol=1e-12,
                atol=1e-12,
                err_msg=f"Material state array {arr_name} mismatch between uninterrupted and restarted continuation",
            )

    def test_dynamic_shell_bt4_restart_continuation_vs_uninterrupted(self, tmp_path: Path):
        """Verify dynamic cycle continuation of BT4 shell group with LAW58 matches uninterrupted run within 10^-12."""
        coords = np.array([
            [0.0, 0.0, 0.0],
            [10.0, 0.0, 0.0],
            [10.0, 10.0, 0.0],
            [0.0, 10.0, 0.0],
        ])
        conn = np.array([[0, 1, 2, 3]])
        mat = make_test_material_law58(
            e1=3000.0,
            e2=1800.0,
            g0=60.0,
            gt=400.0,
            alphat=35.0,
            df=0.05,
            ds=0.08,
            gfrot=18.0,
            arel=0.85,
            n1=2,
            n2=2,
        )
        prop = MockProp(thick=1.0, nip=3)

        dt = 5.0e-6
        n_cycles = 10
        split_cycle = 5

        # 1. Uninterrupted run
        m_unc = Model()
        m_unc.x0 = coords.copy()
        g_unc = MockGroup(conn, slices=[(slice(0, 1), mat, prop)])
        g_unc._model = m_unc
        shell_bt4.init_group(g_unc, m_unc, None)

        coords_unc = coords.copy()
        vel_unc = np.zeros_like(coords)
        rot_unc = np.zeros_like(coords)
        vel_unc[1, 0] = 15.0  # stretch warp
        vel_unc[2, 0] = 15.0
        vel_unc[2, 1] = 10.0  # shear
        vel_unc[3, 1] = 10.0

        fint_unc = np.zeros((4, 3))
        mint_unc = np.zeros((4, 3))

        for c in range(n_cycles):
            fint_unc.fill(0.0)
            mint_unc.fill(0.0)
            shell_bt4.forces(g_unc, coords_unc, vel_unc, rot_unc, dt, fint_unc, mint_unc)
            coords_unc += vel_unc * dt

        # 2. Chained run: Leg 1 (cycles 0..4)
        m_chn = Model()
        m_chn.x0 = coords.copy()
        g_chn = MockGroup(conn, slices=[(slice(0, 1), mat, prop)])
        g_chn._model = m_chn
        shell_bt4.init_group(g_chn, m_chn, None)

        coords_chn = coords.copy()
        vel_chn = vel_unc.copy()
        rot_chn = rot_unc.copy()
        fint_chn = np.zeros((4, 3))
        mint_chn = np.zeros((4, 3))

        for c in range(split_cycle):
            fint_chn.fill(0.0)
            mint_chn.fill(0.0)
            shell_bt4.forces(g_chn, coords_chn, vel_chn, rot_chn, dt, fint_chn, mint_chn)
            coords_chn += vel_chn * dt

        # Save restart state via write_restart
        m_chn.x = coords_chn.copy()
        m_chn.v = vel_chn.copy()
        m_chn.shells = g_chn
        rst_file = tmp_path / "bt4_restart.rst"
        write_restart(m_chn, str(rst_file), engine={"cycle": split_cycle, "t": split_cycle * dt})

        # Leg 2: restore from restart and continue cycles 5..9
        m_res, eng_res = read_restart(str(rst_file))
        assert eng_res["cycle"] == split_cycle

        g_res = m_res.shells
        coords_res = m_res.x.copy()
        vel_res = m_res.v.copy()
        rot_res = rot_unc.copy()
        fint_res = np.zeros((4, 3))
        mint_res = np.zeros((4, 3))

        for c in range(split_cycle, n_cycles):
            fint_res.fill(0.0)
            mint_res.fill(0.0)
            shell_bt4.forces(g_res, coords_res, vel_res, rot_res, dt, fint_res, mint_res)
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
        # Material state arrays: yc, yt, fn, tan_phi, sigv_xy, eps58
        for name in ("yc", "yt", "fn", "tan_phi", "sigv_xy", "eps58"):
            np.testing.assert_allclose(
                g_res.state["mat_extra"][name],
                g_unc.state["mat_extra"][name],
                rtol=1e-12,
                atol=1e-12,
                err_msg=f"Material array {name} mismatch after restart continuation",
            )

    def test_end_to_end_engine_restart_continuation(self, tmp_path: Path):
        """Verify full engine restart chaining (run_starter -> run_engine leg1 -> run_engine leg2) matches uninterrupted."""
        name_unc = "eng_unc"
        name_chn = "eng_chn"

        def make_deck(run_name: str) -> Path:
            d = StarterDeck(run_name)
            d.mat_law58(
                mid=1,
                title="DynFabric",
                rho=1.2e-6,
                e1=2000.0,
                b1=10.0,
                e2=1500.0,
                b2=8.0,
                flex=0.03,
                g0=50.0,
                gt=400.0,
                phi_lock=35.0,
                g5=20.0,
                df=0.05,
                ds=0.08,
                mu_frot=15.0,
                arel=0.85,
                n1=2,
                n2=2,
            )
            d.prop_shell(pid=1, title="Prop1", ishell=1, thick=1.0)
            d.part(pid=1, title="Part1", prop_id=1, mat_id=1)
            d.node([
                (1, 0.0, 0.0, 0.0),
                (2, 10.0, 0.0, 0.0),
                (3, 10.0, 10.0, 0.0),
                (4, 0.0, 10.0, 0.0),
            ])
            d.shell(1, [[1, 1, 2, 3, 4]])
            d.grnod_node(1, "PullNodes", [2, 3])
            d.inivel_tra(1, "StretchKick", [20.0, 0.0, 0.0], 1)
            f = tmp_path / f"{run_name}_0000.rad"
            d.write(str(f))
            return f

        # Determine stable time step
        probe_file = make_deck("probe")
        run_starter(str(probe_file))
        probe_eng = tmp_path / "probe_0001.rad"
        probe_eng.write_text("/RUN/probe/1\n1.0e-8\n/DT\n0.5 0.0\n/PRINT/-1\n", encoding="utf-8")
        run_engine(str(probe_eng))
        _, eng_probe = read_restart(str(tmp_path / "probe_0001.rst"))
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

        # 3. Verify exact cycle count and nodal state continuation
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


# ============================================================================
# Helpers for Unit Testing
# ============================================================================

class MockProp:
    """Mock shell property mimicking /PROP/TYPE1."""
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
    """Mock element group with connectivity, id indexing, and state buffer."""
    def __init__(self, conn: np.ndarray, ids: np.ndarray | None = None, slices: list | None = None):
        self.conn = np.asarray(conn, dtype=np.int64)
        self.n = len(self.conn)
        self.ids = np.arange(1, self.n + 1, dtype=np.int64) if ids is None else np.asarray(ids, dtype=np.int64)
        self.state: dict[str, Any] = {}
        if slices is not None:
            self.state["slices"] = slices
        self._model: Any = None


def make_test_material_law58(
    mid: int = 1,
    rho0: float = 1.2e-6,
    e1: float = 2000.0,
    e2: float = 1500.0,
    g0: float = 50.0,
    gt: float = 120.0,
    alphat: float = 35.0,
    **kwargs: Any,
) -> Material:
    """Factory creating a valid /MAT/LAW58 (/MAT/FABR_A) Material instance."""
    params = {
        "e1": e1,
        "b1": kwargs.get("b1", 0.0),
        "e2": e2,
        "b2": kwargs.get("b2", 0.0),
        "flex": kwargs.get("flex", 0.01),
        "g0": g0,
        "gt": gt,
        "alphat": alphat,
        "g5": kwargs.get("g5", 40.0),
        "df": kwargs.get("df", 0.05),
        "ds": kwargs.get("ds", 0.1),
        "gfrot": kwargs.get("gfrot", 20.0),
        "zero_stress": kwargs.get("zero_stress", 0.0),
        "arel": kwargs.get("arel", 0.0),
        "n1": kwargs.get("n1", 1),
        "n2": kwargs.get("n2", 1),
        "s1": kwargs.get("s1", 0.1),
        "s2": kwargs.get("s2", 0.1),
        "c4": kwargs.get("c4", 0.0),
        "c5": kwargs.get("c5", 0.0),
    }
    params.update(kwargs)
    mat = Material(id=mid, law=58, rho0=rho0, title="Fabric_LAW58", params=params)
    return mat
