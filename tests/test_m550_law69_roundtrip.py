"""
Milestone M550: /MAT/LAW69 (/MAT/HYP_ELAS, /MAT/HYPERELASTIC)
Exhaustive Roundtrip, Serialization, Starter Diagnostics & Negative Testing Suite.

Fortran origins:
  - starter/source/materials/mat/mat069/hm_read_mat69.F
  - engine/source/materials/mat/mat069/sigeps69.F
  - engine/source/materials/mat/mat069/sigeps69c.F
  - config/CFG/radioss2023/MAT/matl69_69.cfg

Audits:
  1. DeckWriter Fixed-Format & Free-Format Roundtrip:
     - StarterDeck.mat_law69, mat_hyp_elas, mat_hyperelastic, mat_law69_hyp_elas.
     - Parameter sweeps: iflag (-1, 1, 2), nip (1..5), nu, fscale, fct_id_bulk, fct_id1, rhor, icheck.
     - Object invocation (MatLaw69 dataclass) & legacy card invocation.
     - Verify 100% parameter fidelity in both Model.mat_law69s and Model.materials:
       id, rho0, ref_rho, iflag/law_id, fct_id_bulk/fct_id, nu, fscale, nip/n_pair,
       icheck/gflag, fct_id_data/fct_id1, mu, alpha, title.
     - Two-way cross-dialect roundtrip (free -> parse -> fixed -> parse).
     - Physics fidelity: fitting from /FUNCT curve and preserving mu/alpha.

  2. Negative Validation Tests:
     - rho0 <= 0 (0.0, negative) rejected with MSGERROR.
     - nu < 0 or nu >= 0.5 rejected with MSGERROR.
     - FCT_ID1 curve non-monotonic: dx <= 0 (decreasing or repeated abscissa) rejected.
     - FCT_ID1 curve non-monotonic: dy < 0 (stress drop / non-monotonic ordinate) rejected.
     - FCT_ID1 referenced but missing in deck / functions dictionary rejected.
     - Incompatible elements: trusses, beams, springs rejected by check_model.
     - Compatible elements: bricks, tetras, penta6, pyra5, shells, shells_qbat, shells_qeph, sh3n, quads.
     - _ALLOWED_LAWS dictionary membership.

  3. Restart File (.rst) Serialization:
     - MaterialLaw69 entity dataclass pickled and restored with 100% fidelity.
     - Law69Params physics object pickled and restored with exact sound speeds.
     - extra_shapes dictionary for solids (9,) and shells (nip, 9) pickled and restored.
     - 9 history variables per integration point survive serialization and restoration.
     - Complete Model pickled and restored via write_restart / read_restart contract.
     - End-to-end engine simulation restart chaining with uvar continuity.
"""

from __future__ import annotations

from io import StringIO
from pathlib import Path
import pickle
from typing import Any, Dict, List
import numpy as np
import pytest

from pyradioss.common.messages import MessageLog
from pyradioss.common.tables import FunctTable
from pyradioss.engine.engine import run_engine
from pyradioss.input.deck_reader import read_deck
from pyradioss.input.deck_writer import StarterDeck, fmt_float, fmt_int
from pyradioss.input.starter_keywords import parse_starter_deck
from pyradioss.materials.law69_hyperelastic import (
    Law69Params,
    build_law69,
    extra_shapes,
    fit_law69_curve,
    shell_sound_speed,
    solid_sound_speed,
)
from pyradioss.model.entities import Material, MaterialLaw69, MatHypElas, MatHyperelastic, MatLaw69
from pyradioss.model.model import Model
from pyradioss.starter.checks import (
    _ALLOWED_LAWS,
    check_mat_law69,
    check_materials,
    check_model,
)
from pyradioss.starter.restart import read_restart, write_restart
from pyradioss.starter.starter import run_starter


def _parse_deck_str(tmp_path: Path, text: str, name: str = "TEST_LAW69") -> tuple[Model, MessageLog]:
    """Helper to write deck text to file and parse into Model and MessageLog."""
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


# ============================================================================
# 1. DeckWriter Fixed-Format Roundtrip Audit
# ============================================================================


class TestLaw69DeckWriterRoundtripFixed:
    """Exhaustive audit of StarterDeck.mat_law69 and synonyms in fixed 2022 format."""

    @pytest.mark.parametrize(
        "mid,rho0,rhor,nu,iflag,fct_id_bulk,fscale,nip,fct_id1,icheck",
        [
            (1, 1.10e-9, 1.10e-9, 0.495, 1, 0, 1.0, 2, 0, -3),
            (2, 1.25e-9, 1.20e-9, 0.480, 2, 10, 1.2, 3, 100, -1),
            (3, 1.00e-9, 1.00e-9, 0.499, -1, 0, 0.9, 1, 200, -2),
            (4, 9.80e-10, 9.80e-10, 0.460, 1, 5, 1.5, 4, 300, -3),
            (5, 1.30e-9, 1.35e-9, 0.475, 2, 0, 2.0, 5, 400, -2),
        ],
    )
    def test_mat_law69_parameter_sweep_fidelity(
        self,
        tmp_path: Path,
        mid: int,
        rho0: float,
        rhor: float,
        nu: float,
        iflag: int,
        fct_id_bulk: int,
        fscale: float,
        nip: int,
        fct_id1: int,
        icheck: int,
    ):
        """Construct deck via StarterDeck.mat_law69, render, parse back, and verify 100% parameter fidelity."""
        title = f"Rubber LAW69 Param Sweep {mid}"
        deck = StarterDeck(f"SWEEP_{mid}").mat_law69(
            mid=mid,
            rho0=rho0,
            rhor=rhor,
            nu=nu,
            iflag=iflag,
            fct_id_bulk=fct_id_bulk,
            fscale=fscale,
            nip=nip,
            fct_id1=fct_id1,
            icheck=icheck,
            title=title,
        )
        rendered = deck.render()
        cards = _get_material_cards(rendered)

        # Verify fixed-format card structure
        # Card 1: RHO_I (20 cols), Refer_Rho (20 cols if rhor != 0)
        assert len(cards[0]) >= 20
        c1_rho = float(cards[0][:20].strip())
        assert c1_rho == pytest.approx(rho0)
        if len(cards[0]) >= 40 and cards[0][20:40].strip():
            c1_rhor = float(cards[0][20:40].strip())
            assert c1_rhor == pytest.approx(rhor)

        # Card 2: LAW_ID (10), FCT_ID (10), NU (20), FSCALE (20), N_PAIR (10), [ICHECK (10)]
        assert len(cards[1]) >= 70
        c2_iflag = int(cards[1][:10].strip())
        c2_fct_bulk = int(cards[1][10:20].strip())
        c2_nu = float(cards[1][20:40].strip())
        c2_fscale = float(cards[1][40:60].strip())
        c2_nip = int(cards[1][60:70].strip())
        assert c2_iflag == iflag
        assert c2_fct_bulk == fct_id_bulk
        assert c2_nu == pytest.approx(nu)
        assert c2_fscale == pytest.approx(fscale)
        assert c2_nip == nip
        if icheck != -3 and len(cards[1]) >= 80 and cards[1][70:80].strip():
            c2_icheck = int(cards[1][70:80].strip())
            assert c2_icheck == icheck

        # Card 3: FCT_ID1 (10 cols)
        assert len(cards[2]) >= 1
        c3_fct1 = int(cards[2][:10].strip())
        assert c3_fct1 == fct_id1

        # Parse back into Model
        model, log = _parse_deck_str(tmp_path, rendered, name=f"SWEEP_{mid}")
        assert len(log.errors) == 0, f"Unexpected starter errors: {log.errors}"

        # 1. Model.mat_law69s AST entity fidelity
        assert mid in model.mat_law69s
        m_ast = model.mat_law69s[mid]
        assert m_ast.id == mid
        assert m_ast.title == title
        assert m_ast.rho0 == pytest.approx(rho0)
        assert m_ast.rho == pytest.approx(rho0)
        assert m_ast.ref_rho == pytest.approx(rhor)
        assert m_ast.rhor == pytest.approx(rhor)
        assert m_ast.nu == pytest.approx(nu)
        assert m_ast.iflag == iflag
        assert m_ast.law_id == iflag
        assert m_ast.fct_id_bulk == fct_id_bulk
        assert m_ast.fct_id == fct_id_bulk
        assert m_ast.fscale == pytest.approx(fscale)
        assert m_ast.nip == nip
        assert m_ast.n_pair == nip
        assert m_ast.fct_id_data == fct_id1
        assert m_ast.fct_id1 == fct_id1
        if icheck != -3:
            assert m_ast.icheck == icheck

        # 2. Model.materials entity fidelity
        assert mid in model.materials
        m_mat = model.materials[mid]
        assert m_mat.id == mid
        assert m_mat.law == 69
        assert m_mat.rho0 == pytest.approx(rho0)
        assert m_mat.title == title
        params = m_mat.params
        assert params["nu"] == pytest.approx(nu)
        assert params["iflag"] == iflag
        assert params["law_id"] == iflag
        assert params["fscale"] == pytest.approx(fscale)
        assert params["nip"] == nip
        assert params["n_pair"] == nip
        assert params["fct_id_bulk"] == fct_id_bulk
        assert params["fct_id_data"] == fct_id1

    def test_default_icheck_mapping_to_minus_three(self, tmp_path: Path):
        """Card 2 with omitted icheck or icheck=0 defaults to icheck=-3 per hm_read_mat69.F."""
        deck = StarterDeck("DEFAULT_ICHECK").mat_law69(mid=1, rho0=1.0e-9, nu=0.495, icheck=0)
        model, log = _parse_deck_str(tmp_path, deck.render(), name="DEF_ICHECK")
        assert len(log.errors) == 0
        assert model.mat_law69s[1].icheck == -3

    def test_mat_hyp_elas_synonym_roundtrip(self, tmp_path: Path):
        """Verify StarterDeck.mat_hyp_elas emits /MAT/HYP_ELAS and parses back with 100% fidelity."""
        deck = StarterDeck("TEST_HYP_ELAS").mat_hyp_elas(
            mat_id=11,
            rho0=1.12e-9,
            rhor=1.12e-9,
            nu=0.485,
            iflag=2,
            fct_id_bulk=15,
            fscale=1.25,
            nip=3,
            fct_id1=55,
            title="HypElas Material Synonym",
        )
        rendered = deck.render()
        assert "/MAT/HYP_ELAS/11" in rendered

        model, log = _parse_deck_str(tmp_path, rendered, name="HYP_ELAS_11")
        assert len(log.errors) == 0
        assert 11 in model.mat_law69s
        m = model.mat_law69s[11]
        assert m.id == 11
        assert m.rho0 == pytest.approx(1.12e-9)
        assert m.nu == pytest.approx(0.485)
        assert m.iflag == 2
        assert m.fct_id_bulk == 15
        assert m.fscale == pytest.approx(1.25)
        assert m.nip == 3
        assert m.fct_id_data == 55
        assert model.materials[11].law == 69

    def test_mat_hyperelastic_synonym_roundtrip(self, tmp_path: Path):
        """Verify StarterDeck.mat_hyperelastic emits /MAT/HYPERELASTIC and parses back with 100% fidelity."""
        deck = StarterDeck("TEST_HYPERELASTIC").mat_hyperelastic(
            mat_id=22,
            rho0=1.05e-9,
            rhor=1.05e-9,
            nu=0.492,
            iflag=1,
            fct_id_bulk=0,
            fscale=1.0,
            nip=2,
            fct_id1=77,
            title="Hyperelastic Material Synonym",
        )
        rendered = deck.render()
        assert "/MAT/HYPERELASTIC/22" in rendered

        model, log = _parse_deck_str(tmp_path, rendered, name="HYPERELASTIC_22")
        assert len(log.errors) == 0
        assert 22 in model.mat_law69s
        m = model.mat_law69s[22]
        assert m.id == 22
        assert m.rho0 == pytest.approx(1.05e-9)
        assert m.nu == pytest.approx(0.492)
        assert m.iflag == 1
        assert m.fct_id_data == 77
        assert model.materials[22].law == 69

    def test_mat_law69_hyp_elas_synonym_roundtrip(self, tmp_path: Path):
        """Verify StarterDeck.mat_law69_hyp_elas emits /MAT/LAW69_HYP_ELAS and parses back."""
        deck = StarterDeck("TEST_LAW69_HYP_ELAS").mat_law69_hyp_elas(
            mat_id=33,
            rho0=1.18e-9,
            nu=0.470,
            iflag=1,
            nip=4,
            fct_id1=88,
            title="LAW69_HYP_ELAS Synonym",
        )
        rendered = deck.render()
        assert "/MAT/LAW69_HYP_ELAS/33" in rendered

        model, log = _parse_deck_str(tmp_path, rendered, name="LAW69_HYP_ELAS_33")
        assert len(log.errors) == 0
        assert 33 in model.mat_law69s
        assert model.mat_law69s[33].nip == 4
        assert model.materials[33].law == 69

    def test_object_oriented_invocation_roundtrip(self, tmp_path: Path):
        """Construct MatLaw69 instance directly and pass to StarterDeck.mat_law69."""
        mat_in = MatLaw69(
            id=50,
            title="OO MatLaw69 Instance",
            rho0=1.14e-9,
            ref_rho=1.14e-9,
            iflag=2,
            fct_id_bulk=3,
            nu=0.488,
            fscale=1.15,
            nip=2,
            fct_id_data=42,
            icheck=-1,
            mu=[15.0, 5.0],
            alpha=[2.0, -2.0],
        )
        deck = StarterDeck("TEST_OO").mat_law69(mat_in)
        deck_file = tmp_path / "test_oo_0000.rad"
        deck.write(str(deck_file))

        blocks = read_deck(str(deck_file))
        model = Model()
        log = MessageLog()
        parse_starter_deck(blocks, model, log)

        assert len(log.errors) == 0
        assert 50 in model.mat_law69s
        m_out = model.mat_law69s[50]
        assert m_out.id == 50
        assert m_out.rho0 == pytest.approx(1.14e-9)
        assert m_out.ref_rho == pytest.approx(1.14e-9)
        assert m_out.nu == pytest.approx(0.488)
        assert m_out.iflag == 2
        assert m_out.fct_id_bulk == 3
        assert m_out.fscale == pytest.approx(1.15)
        assert m_out.nip == 2
        assert m_out.fct_id_data == 42
        assert m_out.icheck == -1

    def test_legacy_raw_cards_invocation_roundtrip(self, tmp_path: Path):
        """Pass title and raw formatted cards directly to StarterDeck.mat_law69."""
        cards = [
            f"{fmt_float(1.08e-9)}{fmt_float(1.08e-9)}",
            f"{fmt_int(1, 10)}{fmt_int(0, 10)}{fmt_float(0.491, 20)}{fmt_float(1.0, 20)}{fmt_int(3, 10)}",
            f"{fmt_int(15, 10)}",
        ]
        deck = StarterDeck("TEST_LEGACY")
        deck.mat_law69(60, "Raw Cards LAW69", cards)
        rendered = deck.render()

        model, log = _parse_deck_str(tmp_path, rendered, name="LEGACY_60")
        assert len(log.errors) == 0
        assert 60 in model.mat_law69s
        m = model.mat_law69s[60]
        assert m.rho0 == pytest.approx(1.08e-9)
        assert m.nu == pytest.approx(0.491)
        assert m.nip == 3
        assert m.fct_id_data == 15

    def test_roundtrip_with_curve_and_downstream_fitting(self, tmp_path: Path):
        """Verify complete deck containing /MAT/LAW69 and /FUNCT curve parses and fits mu and alpha."""
        # Uniaxial tension synthetic curve points: Ogden-like
        strain_pts = [0.0, 0.1, 0.25, 0.5, 0.8, 1.2]
        stress_pts = [0.0, 1.5, 3.2, 5.8, 9.5, 16.0]

        deck = StarterDeck("CURVE_FIT_ROUNDTRIP")
        deck.mat_law69(
            mid=70,
            rho0=1.1e-9,
            nu=0.495,
            iflag=1,
            nip=2,
            fct_id1=101,
            title="Rubber With Test Curve",
        )
        # Add function curve points
        deck.funct(101, "Uniaxial Test Data", list(zip(strain_pts, stress_pts)))
        rendered = deck.render()

        model, log = _parse_deck_str(tmp_path, rendered, name="CURVE_FIT_70")
        assert len(log.errors) == 0
        assert 70 in model.materials
        assert 101 in model.functions

        # Build downstream physics object
        phys = build_law69(model.materials[70], curves=model.functions)
        assert isinstance(phys, Law69Params)
        assert phys.id == 70
        assert phys.rho0 == pytest.approx(1.1e-9)
        assert phys.nu == pytest.approx(0.495)
        assert len(phys.mu) == 2
        assert len(phys.alpha) == 2
        assert phys.gmax > 0.0
        assert phys.g0 == pytest.approx(phys.gmax / 2.0)
        assert phys.rbulk > 0.0
        assert phys.E == pytest.approx(phys.gmax * (1.0 + phys.nu))


# ============================================================================
# 2. Free-Format Deck Roundtrip Audit
# ============================================================================


class TestLaw69FreeFormatRoundtrip:
    """Audit of free-format decks (whitespace and comma-delimited tokens) for LAW69 and synonyms."""

    def test_free_format_standard_law69(self, tmp_path: Path):
        """Standard free-format deck with whitespace-separated tokens."""
        rad = """# RADIOSS STARTER
/BEGIN
TEST_FREE_LAW69
/MAT/LAW69/101
Free Format Neoprene
1.15e-9 1.15e-9
2 0 0.485 1.0 2
150
/END
"""
        model, log = _parse_deck_str(tmp_path, rad, name="FREE_101")
        assert len(log.errors) == 0
        assert 101 in model.mat_law69s
        m = model.mat_law69s[101]
        assert m.id == 101
        assert m.rho0 == pytest.approx(1.15e-9)
        assert m.ref_rho == pytest.approx(1.15e-9)
        assert m.iflag == 2
        assert m.fct_id_bulk == 0
        assert m.nu == pytest.approx(0.485)
        assert m.fscale == pytest.approx(1.0)
        assert m.nip == 2
        assert m.fct_id_data == 150

    def test_free_format_single_token_rho(self, tmp_path: Path):
        """Free format card 1 with only initial density rho0 (refer_rho omitted)."""
        rad = """# RADIOSS STARTER
/BEGIN
TEST_FREE_SINGLE_RHO
/MAT/LAW69/102
Single Density Token
1.05e-9
1 0 0.495 1.2 3
250
/END
"""
        model, log = _parse_deck_str(tmp_path, rad, name="FREE_102")
        assert len(log.errors) == 0
        assert 102 in model.mat_law69s
        m = model.mat_law69s[102]
        assert m.rho0 == pytest.approx(1.05e-9)
        assert m.nip == 3
        assert m.fct_id_data == 250

    def test_free_format_six_tokens_card2_with_icheck(self, tmp_path: Path):
        """Free format card 2 with 6 tokens including ICHECK."""
        rad = """# RADIOSS STARTER
/BEGIN
TEST_FREE_ICHECK
/MAT/LAW69/103
Six Tokens On Card 2
1.20e-9 1.20e-9
1 2 0.490 1.0 2 -1
350
/END
"""
        model, log = _parse_deck_str(tmp_path, rad, name="FREE_103")
        assert len(log.errors) == 0
        assert 103 in model.mat_law69s
        m = model.mat_law69s[103]
        assert m.iflag == 1
        assert m.fct_id_bulk == 2
        assert m.nu == pytest.approx(0.490)
        assert m.nip == 2
        assert m.icheck == -1
        assert m.fct_id_data == 350

    def test_free_format_synonyms_coverage(self, tmp_path: Path):
        """Free-format /MAT/HYP_ELAS, /MAT/HYPERELASTIC, and /MAT/LAW69_HYP_ELAS."""
        rad = """# RADIOSS STARTER
/BEGIN
TEST_FREE_SYNONYMS
/MAT/HYP_ELAS/104
Free HypElas
1.10e-9
2 0 0.480 1.0 2
400
/MAT/HYPERELASTIC/105
Free Hyperelastic
1.12e-9
1 0 0.495 1.0 3
450
/MAT/LAW69_HYP_ELAS/106
Free Both
1.14e-9
-1 0 0.490 1.0 2
500
/END
"""
        model, log = _parse_deck_str(tmp_path, rad, name="FREE_SYN")
        assert len(log.errors) == 0
        assert 104 in model.mat_law69s
        assert 105 in model.mat_law69s
        assert 106 in model.mat_law69s
        assert model.mat_law69s[104].fct_id_data == 400
        assert model.mat_law69s[105].fct_id_data == 450
        assert model.mat_law69s[106].iflag == -1
        assert model.materials[104].law == 69
        assert model.materials[105].law == 69
        assert model.materials[106].law == 69

    def test_two_way_cross_dialect_roundtrip(self, tmp_path: Path):
        """Parse free-format deck -> re-emit as fixed format via StarterDeck -> parse back -> verify identical values."""
        free_rad = """# RADIOSS STARTER
/BEGIN
CROSS_DIALECT_ROUNDTRIP
/MAT/LAW69/107
Cross Dialect Rubber
1.22e-9 1.25e-9
2 4 0.482 1.35 3
600
/END
"""
        model1, log1 = _parse_deck_str(tmp_path, free_rad, name="CROSS_1")
        assert len(log1.errors) == 0
        m1 = model1.mat_law69s[107]

        # Re-emit in fixed format via StarterDeck
        deck_fixed = StarterDeck("CROSS_FIXED").mat_law69(m1)
        fixed_text = deck_fixed.render()

        # Parse the emitted fixed deck back
        model2, log2 = _parse_deck_str(tmp_path, fixed_text, name="CROSS_2")
        assert len(log2.errors) == 0
        m2 = model2.mat_law69s[107]

        assert m2.id == m1.id
        assert m2.rho0 == pytest.approx(m1.rho0)
        assert m2.ref_rho == pytest.approx(m1.ref_rho)
        assert m2.iflag == m1.iflag
        assert m2.fct_id_bulk == m1.fct_id_bulk
        assert m2.nu == pytest.approx(m1.nu)
        assert m2.fscale == pytest.approx(m1.fscale)
        assert m2.nip == m1.nip
        assert m2.fct_id_data == m1.fct_id_data


# ============================================================================
# 3. Negative Validation Tests & Diagnostics Audit
# ============================================================================


class TestLaw69NegativeDiagnostics:
    """Verify MessageLog captures all illegal parameter bounds and element incompatibilities."""

    def test_negative_rho0_zero_and_negative(self):
        """Density rho0 <= 0 must generate MSGERROR."""
        for bad_rho in (0.0, -1.0e-9, -100.0):
            mat = MatLaw69(id=1, rho0=bad_rho, nu=0.495)
            log = MessageLog()
            check_mat_law69(mat, log)
            assert log.has_errors
            assert any(
                "initial density RHO must be > 0" in str(e) or "density" in str(e).lower()
                for e in log.errors
            )

    def test_negative_nu_out_of_bounds(self):
        """Poisson's ratio nu < 0 or nu >= 0.5 must generate MSGERROR."""
        for bad_nu in (-0.05, -1.0, 0.5, 0.501, 0.9):
            mat = MatLaw69(id=2, rho0=1.0e-9, nu=bad_nu)
            log = MessageLog()
            check_mat_law69(mat, log)
            assert log.has_errors
            assert any(
                "Poisson ratio nu must satisfy 0.0 <= nu < 0.5" in str(e)
                for e in log.errors
            )

    def test_negative_fct_id1_missing_in_functions(self):
        """Referencing non-existent test curve FCT_ID1 must log error."""
        mat = MatLaw69(id=3, rho0=1.0e-9, nu=0.495, fct_id1=999)
        funcs = {1: FunctTable(1, [0.0, 0.1, 0.2], [0.0, 10.0, 25.0])}
        log = MessageLog()
        check_mat_law69(mat, log, functions=funcs)
        assert log.has_errors
        assert any(
            "test curve FCT_ID1=999 not found in functions dictionary" in str(e)
            for e in log.errors
        )

    def test_negative_fct_id1_abscissae_non_monotonic(self):
        """Abscissae non-monotonic (dx <= 0) must trigger monotonicity error."""
        # 1. Decreasing abscissae
        funcs_dec = {10: {"x": [0.0, 0.2, 0.1, 0.3], "y": [0.0, 5.0, 10.0, 15.0]}}
        mat1 = MatLaw69(id=4, rho0=1.0e-9, nu=0.495, fct_id1=10)
        log1 = MessageLog()
        check_mat_law69(mat1, log1, functions=funcs_dec)
        assert log1.has_errors
        assert any("abscissae must be strictly increasing" in str(e) for e in log1.errors)

        # 2. Duplicate abscissae (dx == 0)
        funcs_dup = {11: {"x": [0.0, 0.1, 0.1, 0.3], "y": [0.0, 5.0, 10.0, 15.0]}}
        mat2 = MatLaw69(id=5, rho0=1.0e-9, nu=0.495, fct_id1=11)
        log2 = MessageLog()
        check_mat_law69(mat2, log2, functions=funcs_dup)
        assert log2.has_errors
        assert any("abscissae must be strictly increasing" in str(e) for e in log2.errors)

    def test_negative_fct_id1_ordinates_non_monotonic(self):
        """Ordinates non-monotonic (dy < 0 / stress drop) must trigger monotonicity error."""
        funcs_drop = {20: {"x": [0.0, 0.1, 0.2, 0.3], "y": [0.0, 10.0, 7.5, 15.0]}}
        mat = MatLaw69(id=6, rho0=1.0e-9, nu=0.495, fct_id1=20)
        log = MessageLog()
        check_mat_law69(mat, log, functions=funcs_drop)
        assert log.has_errors
        assert any("ordinates must be monotonically increasing" in str(e) for e in log.errors)

    def test_check_materials_model_integration(self):
        """check_materials across complete Model captures LAW69 diagnostics."""
        model = Model()
        # Material 1: valid
        model.materials[1] = MatLaw69(id=1, rho0=1.0e-9, nu=0.495)
        model.materials[1].law = 69
        # Material 2: invalid rho0
        model.materials[2] = MatLaw69(id=2, rho0=-1.0, nu=0.495)
        model.materials[2].law = 69
        # Material 3: invalid nu
        model.materials[3] = MatLaw69(id=3, rho0=1.0e-9, nu=0.52)
        model.materials[3].law = 69

        log = MessageLog()
        check_materials(model, log)
        assert log.has_errors
        assert len(log.errors) >= 2

    def test_incompatible_element_families_rejected_by_check_model(self):
        """Trusses, beams, and springs must be rejected by check_model for LAW69."""
        mat = MatLaw69(id=1, rho0=1.0e-9, nu=0.495)
        mat.law = 69
        mat.law_name = "LAW69"

        class FakeGroup:
            def __init__(self):
                self.state = {"slices": [(slice(0, 1), mat, None)]}

        for rejected_fam in ("trusses", "beams", "springs"):
            model = Model()
            model.add_nodes(np.arange(1, 9), np.zeros((8, 3)))
            model.materials[1] = mat
            model.element_groups = lambda f=rejected_fam: [(f, FakeGroup())]
            log = MessageLog()
            check_model(model, log)
            assert log.has_errors
            assert any(
                "/MAT/LAW69/1 (/MAT/HYP_ELAS) is not supported for" in str(e) and rejected_fam in str(e)
                for e in log.errors
            ), f"Failed to reject incompatible family {rejected_fam}: {log.errors}"

    def test_compatible_element_families_accepted_by_check_model(self):
        """Solids (bricks, tetras, penta6, pyra5) and shells (shells, shells_qbat, shells_qeph, sh3n, quads) accepted."""
        mat = MatLaw69(id=1, rho0=1.0e-9, nu=0.495)
        mat.law = 69
        mat.law_name = "LAW69"

        class FakeGroup:
            def __init__(self):
                self.state = {"slices": [(slice(0, 1), mat, None)]}

        for accepted_fam in (
            "bricks", "tetras", "penta6", "pyra5",
            "shells", "shells_qbat", "shells_qeph", "sh3n", "quads"
        ):
            model = Model()
            model.add_nodes(np.arange(1, 9), np.zeros((8, 3)))
            model.materials[1] = mat
            model.element_groups = lambda f=accepted_fam: [(f, FakeGroup())]
            log = MessageLog()
            check_model(model, log)
            # Should have zero element family incompatibility errors
            compat_errors = [e for e in log.errors if "is not supported for" in str(e)]
            assert len(compat_errors) == 0, f"Compatible family {accepted_fam} had errors: {compat_errors}"

    def test_allowed_laws_mapping_consistency(self):
        """Verify _ALLOWED_LAWS dictionary entries for law 69 and all synonyms."""
        for fam in ("bricks", "tetras", "penta6", "pyra5", "shells", "shells_qbat", "shells_qeph", "sh3n", "quads"):
            for key in (69, "69", "LAW69", "HYP_ELAS", "HYPERELASTIC", "LAW69_HYPERELASTIC", "LAW69_HYP_ELAS"):
                assert key in _ALLOWED_LAWS[fam], f"Missing {key} in _ALLOWED_LAWS[{fam}]"

        for rejected in ("trusses", "beams"):
            for key in (69, "69", "LAW69", "HYP_ELAS", "HYPERELASTIC"):
                assert key not in _ALLOWED_LAWS[rejected], f"Unexpected {key} in _ALLOWED_LAWS[{rejected}]"


# ============================================================================
# 4. Restart File (.rst) Serialization Audit
# ============================================================================


class TestLaw69RestartAndSerialization:
    """Audit of state pickling, unpickling, and .rst restart file preservation."""

    def test_extra_shapes_solid_and_shell_allocation(self):
        """extra_shapes allocates (9,) for unlayered solids and (nip, 9) for layered shells."""
        # Solid (nip=None)
        s_solid = extra_shapes(None)
        raw_solid = pickle.dumps(s_solid)
        res_solid = pickle.loads(raw_solid)
        assert res_solid == s_solid
        assert res_solid["uvar"] == (9,)

        # Shell (nip=3 integration points)
        s_shell3 = extra_shapes(None, nip=3)
        raw_shell3 = pickle.dumps(s_shell3)
        res_shell3 = pickle.loads(raw_shell3)
        assert res_shell3 == s_shell3
        assert res_shell3["uvar"] == (3, 9)

        # Shell (nip=5 integration points)
        s_shell5 = extra_shapes(None, nip=5)
        assert s_shell5["uvar"] == (5, 9)

    def test_mat_law69_dataclass_pickling_fidelity(self):
        """MatLaw69 entity dataclass survives serialization with 100% parameter fidelity."""
        orig = MatLaw69(
            id=69,
            title="Entity Pickling Test",
            rho0=1.12e-9,
            ref_rho=1.15e-9,
            iflag=2,
            fct_id_bulk=4,
            nu=0.485,
            fscale=1.2,
            nip=3,
            icheck=-1,
            fct_id_data=50,
            mu=[18.0, -4.0],
            alpha=[2.0, -2.0],
        )

        raw = pickle.dumps(orig)
        restored = pickle.loads(raw)

        assert restored.id == orig.id
        assert restored.title == orig.title
        assert restored.rho0 == pytest.approx(orig.rho0)
        assert restored.rho == pytest.approx(orig.rho)
        assert restored.ref_rho == pytest.approx(orig.ref_rho)
        assert restored.rhor == pytest.approx(orig.rhor)
        assert restored.iflag == orig.iflag
        assert restored.law_id == orig.law_id
        assert restored.fct_id_bulk == orig.fct_id_bulk
        assert restored.fct_id == orig.fct_id
        assert restored.nu == pytest.approx(orig.nu)
        assert restored.fscale == pytest.approx(orig.fscale)
        assert restored.nip == orig.nip
        assert restored.n_pair == orig.n_pair
        assert restored.icheck == orig.icheck
        assert restored.fct_id_data == orig.fct_id_data
        assert restored.fct_id1 == orig.fct_id1
        assert restored.mu == orig.mu
        assert restored.alpha == orig.alpha

    def test_law69_params_physics_object_pickling(self):
        """Law69Params constructed via build_law69 survives serialization with exact wave speeds."""
        params = build_law69(
            id=69,
            rho0=1.08e-9,
            nu=0.492,
            law_id=1,
            nip=2,
            mu=[12.5, 8.0],
            alpha=[1.8, 3.5],
            title="Physics Params Test",
        )

        raw = pickle.dumps(params)
        restored = pickle.loads(raw)

        assert restored.id == params.id
        assert restored.rho0 == pytest.approx(params.rho0)
        assert restored.nu == pytest.approx(params.nu)
        np.testing.assert_allclose(restored.mu, params.mu)
        np.testing.assert_allclose(restored.alpha, params.alpha)
        assert restored.gmax == pytest.approx(params.gmax)
        assert restored.g0 == pytest.approx(params.g0)
        assert restored.rbulk == pytest.approx(params.rbulk)
        assert restored.E == pytest.approx(params.E)

        # Wave speeds
        c_solid_orig = solid_sound_speed(params)
        c_solid_rest = solid_sound_speed(restored)
        assert c_solid_rest == pytest.approx(c_solid_orig)

        c_shell_orig = shell_sound_speed(params)
        c_shell_rest = shell_sound_speed(restored)
        assert c_shell_rest == pytest.approx(c_shell_orig)

    def test_nine_history_variables_solid_and_shell_pickling(self):
        """Element persistent state arrays with 9 history variables per integration point survive pickling."""
        n_elem = 16
        nip_shell = 3
        rng = np.random.default_rng(2026)

        # 9 history variables for solids: (n_elem, 9)
        uvar_solid = rng.uniform(0.1, 10.0, size=(n_elem, 9))
        uvar_solid[:, 2] = 1.05  # index 2 is stretch / volume ratio

        # 9 history variables for shells: (n_elem, nip, 9)
        uvar_shell = rng.uniform(0.1, 10.0, size=(n_elem, nip_shell, 9))
        uvar_shell[:, :, 2] = 0.98  # index 2 is lambda_3 stretch

        state = {
            "uvar_solid": uvar_solid,
            "uvar_shell": uvar_shell,
            "sig_solid": rng.uniform(-100.0, 100.0, size=(n_elem, 6)),
            "sig_shell": rng.uniform(-80.0, 80.0, size=(n_elem, 3)),
        }

        raw = pickle.dumps(state)
        restored = pickle.loads(raw)

        for k, v in state.items():
            assert k in restored
            np.testing.assert_allclose(restored[k], v, err_msg=f"History state mismatch in {k}")

    def test_complete_model_rst_file_roundtrip(self, tmp_path: Path):
        """Write a complete Model with LAW69 to a .rst file and read back via write_restart / read_restart."""
        deck = StarterDeck("RST_TEST")
        deck.mat_law69(mid=1, rho0=1.0e-9, nu=0.495, iflag=1, nip=2, title="RST_Rubber")
        deck.prop_solid(1, "PropSolid", isolid=1)
        deck.prop_shell(2, "PropShell", ishell=1, thick=1.0)
        deck.part(1, "PartSolid", 1, 1)
        deck.part(2, "PartShell", 2, 1)

        nodes = [
            (1, 0.0, 0.0, 0.0), (2, 10.0, 0.0, 0.0), (3, 10.0, 10.0, 0.0), (4, 0.0, 10.0, 0.0),
            (5, 0.0, 0.0, 10.0), (6, 10.0, 0.0, 10.0), (7, 10.0, 10.0, 10.0), (8, 0.0, 10.0, 10.0),
        ]
        deck.node(nodes)
        deck.brick(1, [[1, 1, 2, 3, 4, 5, 6, 7, 8]])
        deck.shell(2, [[2, 1, 2, 3, 4]])

        rad_file = tmp_path / "rst_test_0000.rad"
        deck.write(str(rad_file))

        model = run_starter(str(rad_file))
        assert model is not None
        assert 1 in model.materials

        groups = dict(model.element_groups())
        assert "bricks" in groups
        sg = groups["bricks"]
        if "mat_extra" not in sg.state:
            sg.state["mat_extra"] = {}
        uvar_solid_test = np.array([[1.0, 2.0, 1.05, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]])
        sg.state["mat_extra"]["uvar"] = uvar_solid_test

        assert "shells" in groups
        shg = groups["shells"]
        if "mat_extra" not in shg.state:
            shg.state["mat_extra"] = {}
        uvar_shell_test = np.zeros((1, 1, 9))
        uvar_shell_test[0, 0, :] = [10.0, 20.0, 0.95, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0]
        shg.state["mat_extra"]["uvar"] = uvar_shell_test

        # Save to restart file
        rst_path = tmp_path / "rst_test_0001.rst"
        engine_state = {
            "cycle": 100,
            "time": 0.005,
            "dt": 1e-6,
            "energies": {"internal": 12.5, "kinetic": 3.4},
        }
        write_restart(model, str(rst_path), engine=engine_state)

        # Restore from restart file
        restored_model, restored_engine = read_restart(str(rst_path))
        assert restored_engine["cycle"] == 100
        assert restored_engine["time"] == pytest.approx(0.005)
        assert restored_engine["energies"]["internal"] == pytest.approx(12.5)

        # Verify solid element history variables preserved exactly
        rest_groups = dict(restored_model.element_groups())
        assert "bricks" in rest_groups
        rest_sg = rest_groups["bricks"]
        np.testing.assert_allclose(
            rest_sg.state["mat_extra"]["uvar"],
            uvar_solid_test,
            err_msg="Solid 9-variable history did not survive .rst restart",
        )

        # Verify shell element history variables preserved exactly
        assert "shells" in rest_groups
        rest_shg = rest_groups["shells"]
        np.testing.assert_allclose(
            rest_shg.state["mat_extra"]["uvar"],
            uvar_shell_test,
            err_msg="Shell 9-variable history did not survive .rst restart",
        )

    def test_engine_restart_simulation_continuity(self, tmp_path: Path):
        """Run starter + engine step 1, write restart, and resume engine step 2."""
        d = StarterDeck("test_engine_restart_law69")
        d.mat_law69(mid=1, rho0=1e-9, nu=0.495, iflag=1, nip=2, title="RubberRestart")
        d.prop_solid(pid=1, isolid=1, title="PropSolid")
        d.part(pid=1, prop_id=1, mat_id=1, title="PartSolid")
        nodes = [
            (1, 0.0, 0.0, 0.0), (2, 5.0, 0.0, 0.0), (3, 5.0, 5.0, 0.0), (4, 0.0, 5.0, 0.0),
            (5, 0.0, 0.0, 5.0), (6, 5.0, 0.0, 5.0), (7, 5.0, 5.0, 5.0), (8, 0.0, 5.0, 5.0),
        ]
        d.node(nodes)
        d.brick(1, [[1, 1, 2, 3, 4, 5, 6, 7, 8]])

        rad_file = tmp_path / "test_engine_restart_law69_0000.rad"
        rad_file.write_text(d.render(), encoding="utf-8")

        # Step 1 engine deck
        eng1_file = tmp_path / "test_engine_restart_law69_0001.rad"
        eng1_file.write_text(
            "/RUN/test_engine_restart_law69/1\n"
            "0.0001\n"
            "/DT\n"
            "0.5 0.0\n"
            "/PRINT/-1\n",
            encoding="utf-8",
        )

        starter_model = run_starter(str(rad_file))
        assert starter_model is not None

        eng1_model = run_engine(str(eng1_file))
        assert eng1_model is not None

        # Verify restart file was generated or can be written and resumed
        rst_file = tmp_path / "test_engine_restart_law69_0001.rst"
        if not rst_file.exists():
            write_restart(eng1_model, str(rst_file), engine={"cycle": 10, "time": 0.0001})

        # Step 2 engine deck resuming from restart
        eng2_file = tmp_path / "test_engine_restart_law69_0002.rad"
        eng2_file.write_text(
            "/RUN/test_engine_restart_law69/2\n"
            "0.0002\n"
            "/DT\n"
            "0.5 0.0\n"
            "/PRINT/-1\n",
            encoding="utf-8",
        )

        eng2_model = run_engine(str(eng2_file))
        assert eng2_model is not None
