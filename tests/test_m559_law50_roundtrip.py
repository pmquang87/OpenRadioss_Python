"""
Milestone M559: /MAT/LAW50 (/MAT/VISC_HONEY, /MAT/HYP_FOAM)
Exhaustive Roundtrip, Negative Validation, Boundary Cases & Restart (.rst) Serialization Auditor Suite.

Fortran origins:
  - starter/source/materials/mat/mat050/hm_read_mat50.F90 (card reader, parameters, table generation)
  - engine/source/materials/mat/mat050/sigeps50s.F90 (constitutive stress update, compaction, sound speed)
  - engine/source/output/restart/wrrestp.F & rdresb.F
  - starter/source/restart/ddsplit/wrrest.F
  - hm_cfg_files/config/CFG/radioss2025/MAT/mat_law50.cfg

Audits:
  1. Fixed-Format Deck Roundtrip:
     - Standard 20/10-column fixed format with StarterDeck.mat_law50 (25 cards).
     - Card 1: RHO, Refer_Rho (MAT_LAW50_1: [20, 20])
     - Card 2: EA, EB, EC (MAT_LAW50_2: [20, 20, 20])
     - Card 3: GAB, GBC, GCA (MAT_LAW50_3: [20, 20, 20])
     - Card 4: asrate, Irate (MAT_LAW50_4: [20, 10])
     - Card 5: Gflag, EPS_max11, EPS_max22, EPS_max33 (MAT_LAW50_5: [10, 20, 20, 20])
     - Cards 6, 7, 8: YFUN11, SFAC11, EPS11 (MAT_LAW50_6..8)
     - Cards 9, 10, 11: YFUN22, SFAC22, EPS22 (MAT_LAW50_9..11)
     - Cards 12, 13, 14: YFUN33, SFAC33, EPS33 (MAT_LAW50_12..14)
     - Card 15: Vflag, EPS_max12, EPS_max23, EPS_max31 (MAT_LAW50_15: [10, 20, 20, 20])
     - Cards 16, 17, 18: YFUN12, SFAC12, EPS12 (MAT_LAW50_16..18)
     - Cards 19, 20, 21: YFUN23, SFAC23, EPS23 (MAT_LAW50_19..21)
     - Cards 22, 23, 24: YFUN31, SFAC31, EPS31 (MAT_LAW50_22..24)
     - Card 25: ECOMP, PR, SIGY, ET, VCOMP (MAT_LAW50_25: [20, 20, 20, 20, 20])
     - Re-parse using read_starter_deck / read_mat_law50.
     - Assert exact equality for all parameters in model.mat_law50s and model.materials.
     - Minimal card and default fallback handling (refer_rho=rho, irate=2, sfac=1.0, eps_max=1e30).
     - Entity object invocation with MatLaw50 and Material instances.
     - Positional arguments with title invocation.

  2. Free-Format Comma- & Space-Delimited Roundtrip:
     - Write and re-parse free-format deck blocks for /MAT/LAW50 and synonyms.
     - Comma-delimited and whitespace-delimited card blocks.
     - Cross-dialect roundtrip (free -> parse -> fixed -> parse -> exact equality).

  3. Keyword Synonyms:
     - /MAT/LAW50, /MAT/VISC_HONEY, /MAT/HYP_FOAM.
     - Model dictionary aliases (mat_law50s, mat_visc_honeys, mat_hyp_foams).
     - Entity class aliases (MatLaw50, MatViscHoney, MatHypFoam).

  4. Negative Starter Diagnostics:
     - Non-positive density rho <= 0 (error).
     - Non-positive moduli EA, EB, EC, GAB, GBC, GCA <= 0 (ANCMSG 306 error).
     - Compaction bounds checks: ECOMP <= 0, PR < 0 or PR >= 0.5, VCOMP <= 0 or VCOMP > 1.0.
     - 2D analysis rejected: N2D > 0 (ANCMSG 305 error).
     - Incompatible element types: shells/quads (ANCMSG 305 error), 1D elements (ANCMSG 306 error).
     - Solid elements accepted (solids, bricks, tetras, penta6, pyra5).
     - Registry and dispatch validation (_ALLOWED_LAWS, _MAT_CHECKS, check_materials).

  5. Boundary Values & Defaults:
     - Default values when omitted: sfac=1.0, irate=2, asrate=0.0, eps_max=1e30, refer_rho=rho.
     - Boundary Poisson ratio pr=0.0 and pr -> 0.495.
     - Derived properties: E, G, bulk, K, sound speed.
     - Compaction activation condition (ecomp * sigy * vcomp > 0).

  6. Restart (.rst) Serialization:
     - MatLaw50 entity dataclass pickling and unpickling fidelity.
     - Model with LAW50 serialization.
     - Persistent material state arrays: eps50, uvar50, compacted, eplas, off50 preserved across write_restart / read_restart contract.
     - Constitutive dynamic cycle continuation from restart matching uninterrupted run within 10^-12.
"""

from __future__ import annotations

import math
from pathlib import Path
import pickle
from typing import Any, Dict, List, Sequence, Tuple
import numpy as np
import pytest

from pyradioss.common.messages import MessageLog
from pyradioss.elements import solid_hexa8
from pyradioss.input.card_layouts import CARD_LAYOUTS, split_fixed
from pyradioss.input.deck_reader import read_deck
from pyradioss.input.deck_writer import StarterDeck
from pyradioss.input.starter_keywords import (
    parse_starter_deck,
    read_mat_law50,
    read_starter_deck,
)
from pyradioss.materials.law50_visc_honey import (
    Law50Params,
    build_law50,
    solid_update_law50,
    sound_speed_solid_law50,
)
from pyradioss.model.entities import (
    MatHypFoam,
    MatLaw50,
    MatViscHoney,
    Material,
)
from pyradioss.model.model import Model
from pyradioss.starter.checks import (
    _ALLOWED_LAWS,
    _MAT_CHECKS,
    check_mat_law50,
    check_materials,
    check_model,
)
from pyradioss.starter.restart import read_restart, write_restart
from pyradioss.starter.starter import run_starter


# ============================================================================
# Helpers
# ============================================================================

def _parse_deck_str(tmp_path: Path, text: str, name: str = "TEST_LAW50") -> Tuple[Model, MessageLog]:
    """Write deck text to file and parse into Model and MessageLog."""
    deck_path = tmp_path / f"{name}_0000.rad"
    deck_path.write_text(text.strip() + "\n", encoding="utf-8")
    blocks = read_deck(str(deck_path))
    model = Model()
    log = MessageLog()
    parse_starter_deck(blocks, model, log)
    return model, log


def _assert_law50_all_fields_exact(
    m: MatLaw50,
    mat: Material | None = None,
    *,
    rho: float,
    refer_rho: float,
    ea: float,
    eb: float,
    ec: float,
    gab: float,
    gbc: float,
    gca: float,
    asrate: float,
    irate: int,
    gflag: int,
    vflag: int,
    eps_max11: float,
    eps_max22: float,
    eps_max33: float,
    eps_max12: float,
    eps_max23: float,
    eps_max31: float,
    yfun11: Sequence[int],
    sfac11: Sequence[float],
    eps11: Sequence[float],
    yfun22: Sequence[int],
    sfac22: Sequence[float],
    eps22: Sequence[float],
    yfun33: Sequence[int],
    sfac33: Sequence[float],
    eps33: Sequence[float],
    yfun12: Sequence[int],
    sfac12: Sequence[float],
    eps12: Sequence[float],
    yfun23: Sequence[int],
    sfac23: Sequence[float],
    eps23: Sequence[float],
    yfun31: Sequence[int],
    sfac31: Sequence[float],
    eps31: Sequence[float],
    ecomp: float,
    pr: float,
    sigy: float,
    et: float,
    vcomp: float,
    title: str = "",
) -> None:
    """Assert exact (floating-point tolerance) equality for all LAW50 parameters."""
    # Density
    assert m.rho == pytest.approx(rho, rel=1e-6, abs=1e-12)
    assert m.refer_rho == pytest.approx(refer_rho, rel=1e-6, abs=1e-12)
    assert m.rho0 == pytest.approx(rho, rel=1e-6, abs=1e-12)
    assert m.rhor == pytest.approx(refer_rho, rel=1e-6, abs=1e-12)

    # Orthotropic Young's moduli
    assert m.ea == pytest.approx(ea, rel=1e-6, abs=1e-12)
    assert m.eb == pytest.approx(eb, rel=1e-6, abs=1e-12)
    assert m.ec == pytest.approx(ec, rel=1e-6, abs=1e-12)
    assert m.e11 == pytest.approx(ea, rel=1e-6, abs=1e-12)
    assert m.e22 == pytest.approx(eb, rel=1e-6, abs=1e-12)
    assert m.e33 == pytest.approx(ec, rel=1e-6, abs=1e-12)
    assert m.E == pytest.approx(max(ea, eb, ec), rel=1e-6, abs=1e-12)

    # Orthotropic Shear moduli
    assert m.gab == pytest.approx(gab, rel=1e-6, abs=1e-12)
    assert m.gbc == pytest.approx(gbc, rel=1e-6, abs=1e-12)
    assert m.gca == pytest.approx(gca, rel=1e-6, abs=1e-12)
    assert m.g12 == pytest.approx(gab, rel=1e-6, abs=1e-12)
    assert m.g23 == pytest.approx(gbc, rel=1e-6, abs=1e-12)
    assert m.g31 == pytest.approx(gca, rel=1e-6, abs=1e-12)
    assert m.G == pytest.approx(max(gab, gbc, gca), rel=1e-6, abs=1e-12)

    # Rate parameters & flags
    assert m.asrate == pytest.approx(asrate, rel=1e-6, abs=1e-12)
    assert m.fcut == pytest.approx(asrate, rel=1e-6, abs=1e-12)
    assert m.irate == irate
    assert m.gflag == gflag
    assert m.vflag == vflag

    # Strain limits
    assert m.eps_max11 == pytest.approx(eps_max11, rel=1e-6, abs=1e-12)
    assert m.eps_max22 == pytest.approx(eps_max22, rel=1e-6, abs=1e-12)
    assert m.eps_max33 == pytest.approx(eps_max33, rel=1e-6, abs=1e-12)
    assert m.eps_max12 == pytest.approx(eps_max12, rel=1e-6, abs=1e-12)
    assert m.eps_max23 == pytest.approx(eps_max23, rel=1e-6, abs=1e-12)
    assert m.eps_max31 == pytest.approx(eps_max31, rel=1e-6, abs=1e-12)

    # Function tables & scale factors
    def _pad_5(vals, default):
        v = list(vals) if vals is not None else []
        while len(v) < 5:
            v.append(default)
        return v[:5]

    assert list(m.yfun11) == _pad_5(yfun11, 0)
    assert list(m.yfun22) == _pad_5(yfun22, 0)
    assert list(m.yfun33) == _pad_5(yfun33, 0)
    assert list(m.yfun12) == _pad_5(yfun12, 0)
    assert list(m.yfun23) == _pad_5(yfun23, 0)
    assert list(m.yfun31) == _pad_5(yfun31, 0)

    for actual, expected in [
        (m.sfac11, _pad_5(sfac11, 1.0)),
        (m.sfac22, _pad_5(sfac22, 1.0)),
        (m.sfac33, _pad_5(sfac33, 1.0)),
        (m.sfac12, _pad_5(sfac12, 1.0)),
        (m.sfac23, _pad_5(sfac23, 1.0)),
        (m.sfac31, _pad_5(sfac31, 1.0)),
        (m.eps11, _pad_5(eps11, 0.0)),
        (m.eps22, _pad_5(eps22, 0.0)),
        (m.eps33, _pad_5(eps33, 0.0)),
        (m.eps12, _pad_5(eps12, 0.0)),
        (m.eps23, _pad_5(eps23, 0.0)),
        (m.eps31, _pad_5(eps31, 0.0)),
    ]:
        for a_val, e_val in zip(actual, expected):
            assert a_val == pytest.approx(e_val, rel=1e-6, abs=1e-12)

    # Compaction parameters
    assert m.ecomp == pytest.approx(ecomp, rel=1e-6, abs=1e-12)
    assert m.pr == pytest.approx(pr, rel=1e-6, abs=1e-12)
    assert m.nu == pytest.approx(pr, rel=1e-6, abs=1e-12)
    assert m.sigy == pytest.approx(sigy, rel=1e-6, abs=1e-12)
    assert m.et == pytest.approx(et, rel=1e-6, abs=1e-12)
    assert m.hcomp == pytest.approx(et, rel=1e-6, abs=1e-12)
    assert m.vcomp == pytest.approx(vcomp, rel=1e-6, abs=1e-12)

    # Derived compaction state
    expected_icomp = 1 if (ecomp * sigy * vcomp > 0.0) else 0
    assert m.icompact == expected_icomp
    assert m.icomp == expected_icomp

    pr_eff = min(pr, 0.495)
    if expected_icomp == 1 and ecomp > 0.0:
        expected_gcomp = ecomp / (1.0 + pr_eff)
        expected_bulk = ecomp / (3.0 * (1.0 - 2.0 * pr_eff))
    else:
        expected_gcomp = 0.0
        expected_bulk = max(ea, eb, ec, gab, gbc, gca)

    assert m.gcomp == pytest.approx(expected_gcomp, rel=1e-6, abs=1e-12)
    assert m.bulk == pytest.approx(expected_bulk, rel=1e-6, abs=1e-12)
    assert m.K == pytest.approx(expected_bulk, rel=1e-6, abs=1e-12)

    if title:
        assert m.title == title

    # Verify model.materials[mid] parameters if provided
    if mat is not None:
        assert mat.law == 50
        assert mat.rho0 == pytest.approx(rho, rel=1e-6, abs=1e-12)
        p = mat.params
        assert p["rho"] == pytest.approx(rho, rel=1e-6, abs=1e-12)
        assert p["refer_rho"] == pytest.approx(refer_rho, rel=1e-6, abs=1e-12)
        assert p["ea"] == pytest.approx(ea, rel=1e-6, abs=1e-12)
        assert p["eb"] == pytest.approx(eb, rel=1e-6, abs=1e-12)
        assert p["ec"] == pytest.approx(ec, rel=1e-6, abs=1e-12)
        assert p["gab"] == pytest.approx(gab, rel=1e-6, abs=1e-12)
        assert p["gbc"] == pytest.approx(gbc, rel=1e-6, abs=1e-12)
        assert p["gca"] == pytest.approx(gca, rel=1e-6, abs=1e-12)
        assert p["asrate"] == pytest.approx(asrate, rel=1e-6, abs=1e-12)
        assert p["irate"] == irate
        assert p["gflag"] == gflag
        assert p["vflag"] == vflag
        assert p["eps_max11"] == pytest.approx(eps_max11, rel=1e-6, abs=1e-12)
        assert p["eps_max22"] == pytest.approx(eps_max22, rel=1e-6, abs=1e-12)
        assert p["eps_max33"] == pytest.approx(eps_max33, rel=1e-6, abs=1e-12)
        assert p["eps_max12"] == pytest.approx(eps_max12, rel=1e-6, abs=1e-12)
        assert p["eps_max23"] == pytest.approx(eps_max23, rel=1e-6, abs=1e-12)
        assert p["eps_max31"] == pytest.approx(eps_max31, rel=1e-6, abs=1e-12)
        assert p["ecomp"] == pytest.approx(ecomp, rel=1e-6, abs=1e-12)
        assert p["pr"] == pytest.approx(pr, rel=1e-6, abs=1e-12)
        assert p["sigy"] == pytest.approx(sigy, rel=1e-6, abs=1e-12)
        assert p["et"] == pytest.approx(et, rel=1e-6, abs=1e-12)
        assert p["vcomp"] == pytest.approx(vcomp, rel=1e-6, abs=1e-12)
        assert p["icompact"] == expected_icomp


# ============================================================================
# 1. Fixed-Format Deck Roundtrip
# ============================================================================

class TestLaw50FixedFormatRoundtrip:
    """Audit StarterDeck.mat_law50 20/10-column fixed-format emission and re-reading."""

    def test_fixed_format_card_columns_and_roundtrip(self, tmp_path: Path):
        """Verify exact 20/10-column layout of all 25 data cards and complete roundtrip."""
        vals = {
            "rho": 1.5e-3,
            "refer_rho": 1.5e-3,
            "ea": 120.0,
            "eb": 220.0,
            "ec": 320.0,
            "gab": 45.0,
            "gbc": 55.0,
            "gca": 65.0,
            "asrate": 30.0,
            "irate": 2,
            "gflag": 1,
            "vflag": -1,
            "eps_max11": 0.25,
            "eps_max22": 0.35,
            "eps_max33": 0.45,
            "eps_max12": 0.55,
            "eps_max23": 0.65,
            "eps_max31": 0.75,
            "yfun11": [101, 102],
            "sfac11": [1.0, 1.2],
            "eps11": [0.0, 10.0],
            "yfun22": [201, 202],
            "sfac22": [1.0, 1.1],
            "eps22": [0.0, 15.0],
            "yfun33": [301, 302],
            "sfac33": [1.0, 1.3],
            "eps33": [0.0, 20.0],
            "yfun12": [121, 122],
            "sfac12": [1.0, 1.05],
            "eps12": [0.0, 5.0],
            "yfun23": [231, 232],
            "sfac23": [1.0, 1.15],
            "eps23": [0.0, 8.0],
            "yfun31": [311, 312],
            "sfac31": [1.0, 1.25],
            "eps31": [0.0, 12.0],
            "ecomp": 850.0,
            "pr": 0.28,
            "sigy": 75.0,
            "et": 25.0,
            "vcomp": 0.35,
        }

        deck = StarterDeck("AUDIT_FIXED_LAW50")
        deck.mat_law50(
            mat_id=50,
            title="Nomex-Honeycomb-Fixed",
            fixed_format=True,
            **vals,
        )

        rendered = deck.render()
        lines = [line for line in rendered.splitlines() if line.strip() and not line.startswith("#")]

        # Locate /MAT/LAW50/50 header
        mat_idx = None
        for idx, line in enumerate(lines):
            if line.startswith("/MAT/LAW50/50"):
                mat_idx = idx
                break
        assert mat_idx is not None, "Failed to locate /MAT/LAW50/50 header in rendered deck"

        # Card 0: Title
        assert lines[mat_idx + 1].strip() == "Nomex-Honeycomb-Fixed"

        # Cards 1 to 25
        card_lines = lines[mat_idx + 2 : mat_idx + 27]
        assert len(card_lines) == 25, f"Expected 25 data cards, got {len(card_lines)}"

        # Card 1: RHO, Refer_Rho (20, 20)
        c1 = split_fixed(card_lines[0], CARD_LAYOUTS["MAT_LAW50_1"])
        assert float(c1[0]) == pytest.approx(vals["rho"])
        assert float(c1[1]) == pytest.approx(vals["refer_rho"])

        # Card 2: EA, EB, EC (20, 20, 20)
        c2 = split_fixed(card_lines[1], CARD_LAYOUTS["MAT_LAW50_2"])
        assert float(c2[0]) == pytest.approx(vals["ea"])
        assert float(c2[1]) == pytest.approx(vals["eb"])
        assert float(c2[2]) == pytest.approx(vals["ec"])

        # Card 3: GAB, GBC, GCA (20, 20, 20)
        c3 = split_fixed(card_lines[2], CARD_LAYOUTS["MAT_LAW50_3"])
        assert float(c3[0]) == pytest.approx(vals["gab"])
        assert float(c3[1]) == pytest.approx(vals["gbc"])
        assert float(c3[2]) == pytest.approx(vals["gca"])

        # Card 4: asrate, Irate (20, 10)
        c4 = split_fixed(card_lines[3], CARD_LAYOUTS["MAT_LAW50_4"])
        assert float(c4[0]) == pytest.approx(vals["asrate"])
        assert int(c4[1]) == vals["irate"]

        # Card 5: Gflag, EPS_max11, EPS_max22, EPS_max33 (10, 20, 20, 20)
        c5 = split_fixed(card_lines[4], CARD_LAYOUTS["MAT_LAW50_5"])
        assert int(c5[0]) == vals["gflag"]
        assert float(c5[1]) == pytest.approx(vals["eps_max11"])
        assert float(c5[2]) == pytest.approx(vals["eps_max22"])
        assert float(c5[3]) == pytest.approx(vals["eps_max33"])

        # Card 15: Vflag, EPS_max12, EPS_max23, EPS_max31 (10, 20, 20, 20)
        c15 = split_fixed(card_lines[14], CARD_LAYOUTS["MAT_LAW50_15"])
        assert int(c15[0]) == vals["vflag"]
        assert float(c15[1]) == pytest.approx(vals["eps_max12"])
        assert float(c15[2]) == pytest.approx(vals["eps_max23"])
        assert float(c15[3]) == pytest.approx(vals["eps_max31"])

        # Card 25: ECOMP, PR, SIGY, ET, VCOMP (20, 20, 20, 20, 20)
        c25 = split_fixed(card_lines[24], CARD_LAYOUTS["MAT_LAW50_25"])
        assert float(c25[0]) == pytest.approx(vals["ecomp"])
        assert float(c25[1]) == pytest.approx(vals["pr"])
        assert float(c25[2]) == pytest.approx(vals["sigy"])
        assert float(c25[3]) == pytest.approx(vals["et"])
        assert float(c25[4]) == pytest.approx(vals["vcomp"])

        # Re-parse via read_starter_deck
        rad_path = tmp_path / "audit_fixed_0000.rad"
        deck.write(str(rad_path))

        model, log = read_starter_deck(str(rad_path))
        assert not log.has_errors, f"Errors encountered parsing fixed deck: {log.errors}"
        assert 50 in model.mat_law50s
        assert 50 in model.materials

        _assert_law50_all_fields_exact(
            model.mat_law50s[50],
            model.materials[50],
            title="Nomex-Honeycomb-Fixed",
            **vals,
        )

    def test_fixed_format_24_cards_no_compaction(self, tmp_path: Path):
        """Verify that when compaction is inactive, 24 cards can be written and roundtrip cleanly."""
        deck = StarterDeck("AUDIT_24_CARDS")
        deck.mat_law50(
            mat_id=51,
            title="Uncompacted Honeycomb",
            rho=1.2e-3,
            ea=100.0,
            eb=200.0,
            ec=300.0,
            gab=40.0,
            gbc=50.0,
            gca=60.0,
            asrate=10.0,
            irate=2,
            gflag=0,
            vflag=0,
            fixed_format=True,
        )

        rad_path = tmp_path / "audit_24_0000.rad"
        deck.write(str(rad_path))

        model, log = read_starter_deck(str(rad_path))
        assert not log.has_errors
        m = model.mat_law50s[51]
        assert m.icompact == 0
        assert m.ecomp == 0.0
        assert m.sigy == 0.0
        assert m.vcomp == 0.0

    def test_fixed_format_minimal_and_default_fallbacks(self, tmp_path: Path):
        """Minimal card deck verifies default fallbacks: refer_rho=rho, irate=2, sfac=1.0, eps_max=1e30."""
        deck = StarterDeck("AUDIT_MINIMAL")
        deck.mat_law50(
            mat_id=52,
            title="Minimal Defaults",
            rho=1.8e-3,
            ea=150.0,
            eb=250.0,
            ec=350.0,
            gab=50.0,
            gbc=60.0,
            gca=70.0,
            fixed_format=True,
        )

        rad_path = tmp_path / "audit_minimal_0000.rad"
        deck.write(str(rad_path))

        model, log = read_starter_deck(str(rad_path))
        assert not log.has_errors
        m = model.mat_law50s[52]
        mat = model.materials[52]

        assert m.refer_rho == pytest.approx(1.8e-3)
        assert m.irate == 2
        assert m.asrate == 0.0
        assert m.eps_max11 == pytest.approx(1.0e30)
        assert m.eps_max22 == pytest.approx(1.0e30)
        assert m.eps_max33 == pytest.approx(1.0e30)
        assert m.eps_max12 == pytest.approx(1.0e30)
        assert m.eps_max23 == pytest.approx(1.0e30)
        assert m.eps_max31 == pytest.approx(1.0e30)
        assert list(m.sfac11) == [1.0, 1.0, 1.0, 1.0, 1.0]
        assert list(m.sfac22) == [1.0, 1.0, 1.0, 1.0, 1.0]
        assert list(m.sfac33) == [1.0, 1.0, 1.0, 1.0, 1.0]
        assert list(m.sfac12) == [1.0, 1.0, 1.0, 1.0, 1.0]
        assert list(m.sfac23) == [1.0, 1.0, 1.0, 1.0, 1.0]
        assert list(m.sfac31) == [1.0, 1.0, 1.0, 1.0, 1.0]

        assert mat.params["refer_rho"] == pytest.approx(1.8e-3)
        assert mat.params["irate"] == 2

    def test_writer_from_matlaw50_instance(self, tmp_path: Path):
        """Passing MatLaw50 dataclass instance to StarterDeck.mat_law50."""
        entity = MatLaw50(
            id=53,
            rho=2.1e-3,
            refer_rho=2.1e-3,
            ea=140.0,
            eb=240.0,
            ec=340.0,
            gab=45.0,
            gbc=55.0,
            gca=65.0,
            asrate=15.0,
            irate=2,
            ecomp=750.0,
            pr=0.22,
            sigy=65.0,
            et=18.0,
            vcomp=0.28,
            title="Entity-Instance-Pass",
        )

        deck = StarterDeck("AUDIT_ENTITY_PASS")
        deck.mat_law50(entity, fixed_format=True)

        rad_path = tmp_path / "audit_entity_0000.rad"
        deck.write(str(rad_path))

        model, log = read_starter_deck(str(rad_path))
        assert not log.has_errors
        assert 53 in model.mat_law50s
        m = model.mat_law50s[53]
        assert m.title == "Entity-Instance-Pass"
        assert m.ecomp == pytest.approx(750.0)
        assert m.pr == pytest.approx(0.22)
        assert m.vcomp == pytest.approx(0.28)

    def test_writer_from_material_instance(self, tmp_path: Path):
        """Passing Material generic container instance to StarterDeck.mat_law50."""
        params = {
            "rho": 2.2e-3,
            "refer_rho": 2.2e-3,
            "ea": 160.0,
            "eb": 260.0,
            "ec": 360.0,
            "gab": 50.0,
            "gbc": 60.0,
            "gca": 70.0,
            "asrate": 18.0,
            "irate": 2,
            "ecomp": 780.0,
            "pr": 0.24,
            "sigy": 70.0,
            "et": 20.0,
            "vcomp": 0.32,
        }
        mat_inst = Material(
            id=54,
            law=50,
            rho0=2.2e-3,
            title="Generic-Material-Pass",
            params=params,
        )

        deck = StarterDeck("AUDIT_MAT_INST_PASS")
        deck.mat_law50(mat_inst, fixed_format=True)

        rad_path = tmp_path / "audit_mat_inst_0000.rad"
        deck.write(str(rad_path))

        model, log = read_starter_deck(str(rad_path))
        assert not log.has_errors
        assert 54 in model.mat_law50s
        m = model.mat_law50s[54]
        assert m.title == "Generic-Material-Pass"
        assert m.ecomp == pytest.approx(780.0)
        assert m.sigy == pytest.approx(70.0)

    def test_positional_arguments_with_title(self, tmp_path: Path):
        """Verify mat_law50(id, 'title', rho, ea, eb, ec, ...) positional argument handling."""
        deck = StarterDeck("POS_LAW50")
        deck.mat_law50(
            55,
            "Positional Honeycomb",
            0.0016,  # rho
            110.0,   # ea
            210.0,   # eb
            310.0,   # ec
            42.0,    # gab
            52.0,    # gbc
            62.0,    # gca
            22.0,    # asrate
            2,       # irate
            1,       # gflag
        )

        rad_path = tmp_path / "pos_law50_0000.rad"
        deck.write(str(rad_path))

        model, log = read_starter_deck(str(rad_path))
        assert not log.has_errors
        assert 55 in model.mat_law50s
        m = model.mat_law50s[55]
        assert m.title == "Positional Honeycomb"
        assert m.rho == pytest.approx(0.0016)
        assert m.ea == pytest.approx(110.0)
        assert m.eb == pytest.approx(210.0)
        assert m.ec == pytest.approx(310.0)
        assert m.gab == pytest.approx(42.0)
        assert m.gbc == pytest.approx(52.0)
        assert m.gca == pytest.approx(62.0)
        assert m.asrate == pytest.approx(22.0)
        assert m.irate == 2
        assert m.gflag == 1


# ============================================================================
# 2. Free-Format Roundtrip
# ============================================================================

class TestLaw50FreeFormatRoundtrip:
    """Audit free-format comma-delimited and space-delimited emission and re-reading."""

    def test_free_format_comma_delimited_roundtrip(self, tmp_path: Path):
        """Verify comma-delimited free format writes and re-parses with all 25 cards."""
        deck = StarterDeck("FREE_COMMA_LAW50")
        deck.mat_law50(
            mat_id=60,
            title="Free Comma Honeycomb",
            fixed_format=False,
            comma_delimited=True,
            rho=1.5e-3,
            refer_rho=1.5e-3,
            ea=130.0,
            eb=230.0,
            ec=330.0,
            gab=48.0,
            gbc=58.0,
            gca=68.0,
            asrate=25.0,
            irate=2,
            gflag=1,
            vflag=0,
            eps_max11=0.2,
            eps_max22=0.3,
            eps_max33=0.4,
            eps_max12=0.5,
            eps_max23=0.6,
            eps_max31=0.7,
            yfun11=[1, 2],
            sfac11=[1.0, 1.2],
            eps11=[0.0, 10.0],
            ecomp=820.0,
            pr=0.26,
            sigy=78.0,
            et=22.0,
            vcomp=0.32,
        )

        rendered = deck.render()
        assert "," in rendered

        rad_path = tmp_path / "free_comma_0000.rad"
        deck.write(str(rad_path))

        model, log = read_starter_deck(str(rad_path))
        assert not log.has_errors
        assert 60 in model.mat_law50s
        m = model.mat_law50s[60]
        assert m.title == "Free Comma Honeycomb"
        assert m.ea == pytest.approx(130.0)
        assert m.ecomp == pytest.approx(820.0)
        assert m.pr == pytest.approx(0.26)
        assert m.sigy == pytest.approx(78.0)

    def test_free_format_space_delimited_roundtrip(self, tmp_path: Path):
        """Verify space-delimited free format writes and re-parses with all 25 cards."""
        deck = StarterDeck("FREE_SPACE_LAW50")
        deck.mat_law50(
            mat_id=61,
            title="Free Space Honeycomb",
            fixed_format=False,
            delimiter=" ",
            rho=1.4e-3,
            refer_rho=1.4e-3,
            ea=125.0,
            eb=225.0,
            ec=325.0,
            gab=46.0,
            gbc=56.0,
            gca=66.0,
            asrate=20.0,
            irate=2,
            gflag=0,
            vflag=0,
            ecomp=800.0,
            pr=0.25,
            sigy=75.0,
            et=20.0,
            vcomp=0.30,
        )

        rad_path = tmp_path / "free_space_0000.rad"
        deck.write(str(rad_path))

        model, log = read_starter_deck(str(rad_path))
        assert not log.has_errors
        assert 61 in model.mat_law50s
        m = model.mat_law50s[61]
        assert m.title == "Free Space Honeycomb"
        assert m.ea == pytest.approx(125.0)
        assert m.ecomp == pytest.approx(800.0)

    def test_cross_dialect_roundtrip(self, tmp_path: Path):
        """Free format -> parse -> write with fixed format -> parse -> exact equality."""
        deck_free = StarterDeck("CROSS_DIALECT_FREE")
        deck_free.mat_law50(
            mat_id=62,
            title="Cross-Dialect Honeycomb",
            fixed_format=False,
            comma_delimited=True,
            rho=1.6e-3,
            refer_rho=1.6e-3,
            ea=135.0,
            eb=235.0,
            ec=335.0,
            gab=49.0,
            gbc=59.0,
            gca=69.0,
            asrate=28.0,
            irate=2,
            gflag=1,
            vflag=-1,
            eps_max11=0.22,
            eps_max22=0.32,
            eps_max33=0.42,
            eps_max12=0.52,
            eps_max23=0.62,
            eps_max31=0.72,
            yfun11=[5, 6],
            sfac11=[1.0, 1.15],
            eps11=[0.0, 12.0],
            ecomp=840.0,
            pr=0.27,
            sigy=82.0,
            et=24.0,
            vcomp=0.34,
        )

        free_path = tmp_path / "cross_free_0000.rad"
        deck_free.write(str(free_path))
        m_free, log_free = read_starter_deck(str(free_path))
        assert not log_free.has_errors

        # Write parsed model out using fixed format
        deck_fixed = StarterDeck("CROSS_DIALECT_FIXED")
        deck_fixed.mat_law50(m_free.mat_law50s[62], fixed_format=True)

        fixed_path = tmp_path / "cross_fixed_0000.rad"
        deck_fixed.write(str(fixed_path))
        m_fixed, log_fixed = read_starter_deck(str(fixed_path))
        assert not log_fixed.has_errors

        m1 = m_free.mat_law50s[62]
        m2 = m_fixed.mat_law50s[62]

        assert m1.rho == pytest.approx(m2.rho)
        assert m1.ea == pytest.approx(m2.ea)
        assert m1.eb == pytest.approx(m2.eb)
        assert m1.ec == pytest.approx(m2.ec)
        assert m1.gab == pytest.approx(m2.gab)
        assert m1.gbc == pytest.approx(m2.gbc)
        assert m1.gca == pytest.approx(m2.gca)
        assert m1.asrate == pytest.approx(m2.asrate)
        assert m1.irate == m2.irate
        assert m1.gflag == m2.gflag
        assert m1.vflag == m2.vflag
        assert m1.eps_max11 == pytest.approx(m2.eps_max11)
        assert m1.ecomp == pytest.approx(m2.ecomp)
        assert m1.pr == pytest.approx(m2.pr)
        assert m1.sigy == pytest.approx(m2.sigy)
        assert m1.vcomp == pytest.approx(m2.vcomp)


# ============================================================================
# 3. Keyword Synonyms
# ============================================================================

class TestLaw50Synonyms:
    """Audit all keyword synonyms: /MAT/LAW50, /MAT/VISC_HONEY, /MAT/HYP_FOAM."""

    @pytest.mark.parametrize("writer_method,keyword_str,expected_law_name", [
        ("mat_law50", "/MAT/LAW50", "LAW50"),
        ("mat_visc_honey", "/MAT/VISC_HONEY", "VISC_HONEY"),
        ("mat_hyp_foam", "/MAT/HYP_FOAM", "HYP_FOAM"),
    ])
    def test_synonym_emission_and_parsing(
        self, tmp_path: Path, writer_method: str, keyword_str: str, expected_law_name: str
    ):
        """Verify each synonym emits correct header and parses into mat_law50s, mat_visc_honeys, and materials."""
        deck = StarterDeck(f"TEST_{writer_method.upper()}")
        method = getattr(deck, writer_method)
        method(
            mat_id=70,
            title=f"Synonym {keyword_str}",
            rho=1.5e-3,
            ea=100.0,
            eb=200.0,
            ec=300.0,
            gab=40.0,
            gbc=50.0,
            gca=60.0,
        )

        rendered = deck.render()
        assert f"{keyword_str}/70" in rendered

        rad_path = tmp_path / f"{writer_method}_0000.rad"
        deck.write(str(rad_path))

        model, log = read_starter_deck(str(rad_path))
        assert not log.has_errors
        assert 70 in model.mat_law50s
        assert 70 in model.materials
        assert model.materials[70].law_name == expected_law_name

        # Verify model aliases
        assert 70 in model.mat_visc_honeys
        assert 70 in model.mat_hyp_foams
        assert model.mat_law50s[70] is model.mat_visc_honeys[70]
        assert model.mat_law50s[70] is model.mat_hyp_foams[70]

    def test_entity_class_aliases(self):
        """Verify MatViscHoney and MatHypFoam are exact aliases of MatLaw50."""
        assert MatViscHoney is MatLaw50
        assert MatHypFoam is MatLaw50


# ============================================================================
# 4. Negative Starter Diagnostics
# ============================================================================

class TestLaw50NegativeDiagnostics:
    """Audit error diagnostics mandated by hm_read_mat50.F90 and check_mat_law50."""

    def test_non_positive_density_error(self):
        """Density rho <= 0 triggers fatal starter diagnostic."""
        for bad_rho in (0.0, -1.5e-3):
            log = MessageLog()
            m = MatLaw50(id=1, rho=bad_rho, ea=100, eb=100, ec=100, gab=50, gbc=50, gca=50)
            check_mat_law50(mat=m, log=log)
            assert log.has_errors
            assert any("initial density RHO must be > 0" in e for e in log.errors)

    def test_non_positive_moduli_ancmsg_306(self):
        """Moduli EA, EB, EC, GAB, GBC, GCA <= 0 triggers ANCMSG 306 error."""
        mod_params = [
            ("ea", "EA"), ("eb", "EB"), ("ec", "EC"),
            ("gab", "GAB"), ("gbc", "GBC"), ("gca", "GCA"),
        ]
        for attr, name in mod_params:
            for bad_val in (0.0, -50.0):
                kwargs = {"rho": 1.5e-3, "ea": 100, "eb": 100, "ec": 100, "gab": 50, "gbc": 50, "gca": 50}
                kwargs[attr] = bad_val
                log = MessageLog()
                m = MatLaw50(id=2, **kwargs)
                check_mat_law50(mat=m, log=log)
                assert log.has_errors
                assert any(f"elastic modulus {name} must be > 0" in e and "ANCMSG 306" in e for e in log.errors)

    def test_compaction_bounds_diagnostics(self):
        """Compaction parameters out of bounds trigger validation errors."""
        base_kwargs = {
            "id": 3, "rho": 1.5e-3, "ea": 100, "eb": 100, "ec": 100,
            "gab": 50, "gbc": 50, "gca": 50, "sigy": 50.0, "vcomp": 0.3,
        }

        # 1. ECOMP <= 0
        for bad_ecomp in (0.0, -100.0):
            log = MessageLog()
            m = MatLaw50(ecomp=bad_ecomp, **base_kwargs)
            check_mat_law50(mat=m, log=log)
            assert log.has_errors
            assert any("compacted Young's modulus ECOMP must be > 0" in e for e in log.errors)

        # 2. PR < 0 or PR >= 0.5
        for bad_pr in (-0.1, 0.5, 0.6):
            log = MessageLog()
            m = MatLaw50(ecomp=500.0, pr=bad_pr, **base_kwargs)
            check_mat_law50(mat=m, log=log)
            assert log.has_errors
            assert any("compacted Poisson's ratio NU/PR must satisfy 0 <= NU < 0.5" in e for e in log.errors)

        # 3. VCOMP <= 0 or VCOMP > 1.0
        for bad_vcomp in (0.0, -0.2, 1.05):
            log = MessageLog()
            m = MatLaw50(ecomp=500.0, pr=0.25, vcomp=bad_vcomp, id=3, rho=1.5e-3, ea=100, eb=100, ec=100, gab=50, gbc=50, gca=50, sigy=50.0)
            check_mat_law50(mat=m, log=log)
            assert log.has_errors
            assert any("compaction volume fraction VCOMP must satisfy 0 < VCOMP <= 1" in e for e in log.errors)

    def test_element_compatibility_and_rejection(self):
        """Solids accepted; shells/quads rejected (ANCMSG 305); 1D rejected (ANCMSG 306); 2D analysis rejected."""
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

        mat50 = MatLaw50(id=50, rho=1.5e-3, ea=100, eb=100, ec=100, gab=50, gbc=50, gca=50)

        # 1. Solids accepted
        for s_type in ("solids", "bricks", "tetras", "penta6", "pyra5"):
            model_s = DummyModel([(s_type, DummyGrp([DummyEl(50)]))])
            log_s = MessageLog()
            check_mat_law50(model=model_s, mat=mat50, log=log_s)
            assert not log_s.has_errors, f"Solid element {s_type} should be accepted"

        # 2. Shells and quads rejected (ANCMSG 305)
        for sh_type in ("shells", "shells_qbat", "shells_qeph", "sh3n", "quads"):
            model_sh = DummyModel([(sh_type, DummyGrp([DummyEl(50)]))])
            log_sh = MessageLog()
            check_mat_law50(model=model_sh, mat=mat50, log=log_sh)
            assert log_sh.has_errors, f"Shell element {sh_type} should be rejected"
            assert any("shell elements" in e and "ANCMSG 305" in e for e in log_sh.errors)

        # 3. 1D elements rejected (ANCMSG 306)
        for d1_type in ("trusses", "beams", "springs"):
            model_1d = DummyModel([(d1_type, DummyGrp([DummyEl(50)]))])
            log_1d = MessageLog()
            check_mat_law50(model=model_1d, mat=mat50, log=log_1d)
            assert log_1d.has_errors, f"1D element {d1_type} should be rejected"
            assert any("1D elements" in e and "ANCMSG 306" in e for e in log_1d.errors)

        # 4. 2D analysis rejected (N2D > 0, ANCMSG 305)
        model_2d = DummyModel([("solids", DummyGrp([DummyEl(50)]))], n2d=1)
        log_2d = MessageLog()
        check_mat_law50(model=model_2d, mat=mat50, log=log_2d)
        assert log_2d.has_errors
        assert any("N2D > 0" in e and "ANCMSG 305" in e for e in log_2d.errors)

    def test_checks_dispatch_and_allowed_laws_registry(self):
        """Verify _MAT_CHECKS registry and _ALLOWED_LAWS table for all LAW50 synonyms."""
        for syn in (50, "50", "LAW50", "VISC_HONEY", "HYP_FOAM", "MAT_LAW50", "MAT_VISC_HONEY", "MAT_HYP_FOAM"):
            assert syn in _MAT_CHECKS, f"{syn} missing from _MAT_CHECKS"
            assert _MAT_CHECKS[syn] is check_mat_law50

        # Solids permit LAW50
        for fam in ("bricks", "tetras", "penta6", "pyra5"):
            assert 50 in _ALLOWED_LAWS[fam]
            assert "LAW50" in _ALLOWED_LAWS[fam]
            assert "VISC_HONEY" in _ALLOWED_LAWS[fam]
            assert "HYP_FOAM" in _ALLOWED_LAWS[fam]

        # Shells and 1D forbid LAW50
        for fam in ("shells", "shells_qbat", "shells_qeph", "sh3n", "quads", "trusses", "beams"):
            assert 50 not in _ALLOWED_LAWS[fam]
            assert "LAW50" not in _ALLOWED_LAWS[fam]
            assert "VISC_HONEY" not in _ALLOWED_LAWS[fam]


# ============================================================================
# 5. Boundary Values & Defaults
# ============================================================================

class TestLaw50BoundaryAndDefaults:
    """Audit boundary values and default parameter assignments."""

    def test_omitted_defaults_application(self, tmp_path: Path):
        """Verify refer_rho=rho, irate=2, sfac=1.0, eps_max=1e30 when omitted."""
        deck = StarterDeck("OMITTED_DEFAULTS_LAW50")
        deck.mat_law50(
            mat_id=1,
            rho=2.5e-3,
            ea=100.0,
            eb=200.0,
            ec=300.0,
            gab=40.0,
            gbc=50.0,
            gca=60.0,
        )

        rad_path = tmp_path / "omitted_defaults_0000.rad"
        deck.write(str(rad_path))
        model, log = read_starter_deck(str(rad_path))
        assert not log.has_errors

        m = model.mat_law50s[1]
        assert m.refer_rho == pytest.approx(2.5e-3)
        assert m.irate == 2
        assert m.asrate == 0.0
        assert m.eps_max11 == pytest.approx(1.0e30)
        assert list(m.sfac11) == [1.0, 1.0, 1.0, 1.0, 1.0]

    def test_poisson_boundary_cases(self):
        """Verify nu=0.0 and nu approaching 0.495 limit."""
        for pr_val in (0.0, 0.49, 0.495):
            log = MessageLog()
            m = MatLaw50(
                id=1, rho=1.0, ea=100, eb=100, ec=100, gab=50, gbc=50, gca=50,
                ecomp=500.0, pr=pr_val, sigy=50.0, et=10.0, vcomp=0.3,
            )
            check_mat_law50(mat=m, log=log)
            assert not log.has_errors
            assert m.pr == pytest.approx(pr_val)
            assert m.nu == pytest.approx(pr_val)

    def test_derived_moduli_and_sound_speed(self):
        """Verify E = max(EA,EB,EC), G = max(GAB,GBC,GCA), and sound speed calculation."""
        rho = 2.0e-3
        ea, eb, ec = 100.0, 250.0, 180.0
        gab, gbc, gca = 40.0, 60.0, 50.0

        m = MatLaw50(
            id=1, rho=rho, ea=ea, eb=eb, ec=ec, gab=gab, gbc=gbc, gca=gca
        )
        expected_e = max(ea, eb, ec)
        expected_g = max(gab, gbc, gca)
        expected_c = math.sqrt(max(expected_e, expected_g) / rho)

        assert m.E == pytest.approx(expected_e)
        assert m.G == pytest.approx(expected_g)
        assert m.sound_speed == pytest.approx(expected_c)
        assert m.sound_speed_solid == pytest.approx(expected_c)

    def test_compaction_flags(self):
        """Verify icompact/icomp coupling flags."""
        m_active = MatLaw50(
            id=1, rho=1.0, ea=100, eb=100, ec=100, gab=50, gbc=50, gca=50,
            ecomp=500.0, pr=0.25, sigy=50.0, et=10.0, vcomp=0.3,
        )
        assert m_active.icompact == 1
        assert m_active.icomp == 1

        m_inactive = MatLaw50(
            id=2, rho=1.0, ea=100, eb=100, ec=100, gab=50, gbc=50, gca=50,
            ecomp=0.0, pr=0.0, sigy=0.0, et=0.0, vcomp=0.0,
        )
        assert m_inactive.icompact == 0
        assert m_inactive.icomp == 0


# ============================================================================
# 6. Restart (.rst) Serialization
# ============================================================================

class TestLaw50RestartSerialization:
    """Audit restart state serialization and dynamic cycle continuation for LAW50."""

    def test_mat_law50_dataclass_pickle_fidelity(self):
        """Pickle and unpickle MatLaw50, verifying all fields and derived properties."""
        mat = MatLaw50(
            id=50,
            rho=1.5e-3,
            refer_rho=1.5e-3,
            ea=100.0,
            eb=200.0,
            ec=300.0,
            gab=40.0,
            gbc=50.0,
            gca=60.0,
            asrate=25.0,
            irate=2,
            gflag=1,
            vflag=-1,
            eps_max11=0.2,
            eps_max22=0.3,
            eps_max33=0.4,
            eps_max12=0.5,
            eps_max23=0.6,
            eps_max31=0.7,
            yfun11=[1, 2],
            sfac11=[1.0, 1.2],
            eps11=[0.0, 10.0],
            ecomp=800.0,
            pr=0.25,
            sigy=80.0,
            et=20.0,
            vcomp=0.3,
            title="Honeycomb-Pickle",
        )

        data = pickle.dumps(mat, protocol=pickle.HIGHEST_PROTOCOL)
        restored: MatLaw50 = pickle.loads(data)

        assert restored.id == 50
        assert restored.rho == pytest.approx(1.5e-3)
        assert restored.ea == pytest.approx(100.0)
        assert restored.eb == pytest.approx(200.0)
        assert restored.ec == pytest.approx(300.0)
        assert restored.gab == pytest.approx(40.0)
        assert restored.gbc == pytest.approx(50.0)
        assert restored.gca == pytest.approx(60.0)
        assert restored.asrate == pytest.approx(25.0)
        assert restored.irate == 2
        assert restored.gflag == 1
        assert restored.vflag == -1
        assert restored.eps_max11 == pytest.approx(0.2)
        assert restored.ecomp == pytest.approx(800.0)
        assert restored.pr == pytest.approx(0.25)
        assert restored.sigy == pytest.approx(80.0)
        assert restored.et == pytest.approx(20.0)
        assert restored.vcomp == pytest.approx(0.3)
        assert restored.title == "Honeycomb-Pickle"
        assert restored.icompact == 1

    def test_model_with_law50_pickle_fidelity(self):
        """Pickle and unpickle full Model with LAW50 material and check consistency."""
        model = Model()
        m50 = MatLaw50(
            id=10,
            rho=1.5e-3,
            ea=100.0,
            eb=200.0,
            ec=300.0,
            gab=40.0,
            gbc=50.0,
            gca=60.0,
            asrate=25.0,
            irate=2,
            ecomp=800.0,
            pr=0.25,
            sigy=80.0,
            et=20.0,
            vcomp=0.3,
            title="Model Pickled Law50",
        )
        model.mat_law50s[10] = m50
        model.materials[10] = Material(
            id=10, law=50, rho0=1.5e-3, title="Model Pickled Law50", params=m50.params
        )

        data = pickle.dumps(model, protocol=pickle.HIGHEST_PROTOCOL)
        restored_model: Model = pickle.loads(data)

        assert 10 in restored_model.mat_law50s
        assert 10 in restored_model.materials
        m_rest = restored_model.mat_law50s[10]
        assert m_rest.title == "Model Pickled Law50"
        assert m_rest.ea == pytest.approx(100.0)
        assert m_rest.ecomp == pytest.approx(800.0)

    def test_material_state_arrays_rst_preservation(self, tmp_path: Path):
        """Verify persistent material state arrays (eps50, uvar50, compacted, eplas, off50) survive write_restart and read_restart."""
        deck = StarterDeck("RST_LAW50_ARRAYS")
        deck.node([
            (1, 0.0, 0.0, 0.0), (2, 1.0, 0.0, 0.0), (3, 1.0, 1.0, 0.0), (4, 0.0, 1.0, 0.0),
            (5, 0.0, 0.0, 1.0), (6, 1.0, 0.0, 1.0), (7, 1.0, 1.0, 1.0), (8, 0.0, 1.0, 1.0),
            (9, 2.0, 0.0, 0.0), (10, 2.0, 1.0, 0.0), (11, 2.0, 0.0, 1.0), (12, 2.0, 1.0, 1.0),
        ])
        deck.brick(1, [
            [1, 1, 2, 3, 4, 5, 6, 7, 8],
            [2, 2, 9, 10, 3, 6, 11, 12, 7],
        ])
        deck.prop_solid(1, "SolidProp")
        deck.part(1, "HoneycombPart", prop_id=1, mat_id=1)
        deck.mat_law50(
            mat_id=1,
            title="Honeycomb M50",
            rho=1.5e-3,
            ea=100.0,
            eb=200.0,
            ec=300.0,
            gab=40.0,
            gbc=50.0,
            gca=60.0,
            asrate=25.0,
            irate=2,
            ecomp=800.0,
            pr=0.25,
            sigy=80.0,
            et=20.0,
            vcomp=0.3,
        )

        rad_path = tmp_path / "rst_law50_arrays_0000.rad"
        deck.write(str(rad_path))

        model = run_starter(str(rad_path))
        assert model is not None

        groups = dict(model.element_groups())
        bg = groups.get("bricks") or groups.get("solids")
        st = bg.state

        # Synthesize realistic persistent material state arrays for LAW50 (2 elements)
        eps50_synth = np.array([
            [0.05, -0.015, -0.015, 0.02, 0.0, 0.0],
            [0.10, -0.030, -0.030, 0.04, 0.0, 0.0],
        ], dtype=float)
        uvar50_synth = np.array([
            [120.0, 10.0, 10.0, 25.0, 0.0, 0.0],
            [240.0, 20.0, 20.0, 50.0, 0.0, 0.0],
        ], dtype=float)
        compacted_synth = np.array([False, True], dtype=bool)
        eplas_synth = np.array([0.0, 0.035], dtype=float)
        off50_synth = np.array([1.0, 0.0], dtype=float)  # 1 alive, 1 eroded
        sig_synth = np.array([
            [50.0, 20.0, 30.0, 15.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        ], dtype=float)

        st["mat_extra"]["eps50"] = eps50_synth.copy()
        st["mat_extra"]["uvar50"] = uvar50_synth.copy()
        st["mat_extra"]["compacted"] = compacted_synth.copy()
        st["mat_extra"]["eplas"] = eplas_synth.copy()
        st["mat_extra"]["off50"] = off50_synth.copy()
        st["sig"] = sig_synth.copy()

        # Write restart .rst
        rst_path = tmp_path / "rst_law50_0001.rst"
        engine_dict = {
            "cycle": 2000,
            "t": 4.5e-5,
            "dt": 1.5e-8,
            "energies": {"internal": 95000.0, "kinetic": 31000.0},
        }
        write_restart(model, str(rst_path), engine=engine_dict)

        # Read back from restart .rst
        rest_model, rest_engine = read_restart(str(rst_path))
        assert rest_engine["cycle"] == 2000
        assert rest_engine["t"] == pytest.approx(4.5e-5)
        assert rest_engine["dt"] == pytest.approx(1.5e-8)

        rest_bg = dict(rest_model.element_groups()).get("bricks") or dict(rest_model.element_groups()).get("solids")
        rest_extra = rest_bg.state["mat_extra"]
        rest_st = rest_bg.state

        # Assert exact preservation of LAW50 state arrays
        np.testing.assert_array_equal(rest_extra["eps50"], eps50_synth)
        np.testing.assert_array_equal(rest_extra["uvar50"], uvar50_synth)
        np.testing.assert_array_equal(rest_extra["compacted"], compacted_synth)
        np.testing.assert_array_equal(rest_extra["eplas"], eplas_synth)
        np.testing.assert_array_equal(rest_extra["off50"], off50_synth)
        np.testing.assert_array_equal(rest_st["sig"], sig_synth)

    def test_constitutive_dynamic_cycle_continuation_from_restart(self):
        """Constitutive stress update continuation from restart matches uninterrupted simulation within 10^-12."""
        mat = build_law50({
            "id": 1,
            "rho0": 100.0,
            "ea": 1.0e7,
            "eb": 1.2e7,
            "ec": 1.5e7,
            "gab": 4.0e6,
            "gbc": 4.5e6,
            "gca": 5.0e6,
            "asrate": 50.0,
            "irate": 2,
            "ecomp": 8.0e7,
            "pr": 0.25,
            "sigy": 5.0e6,
            "et": 2.0e6,
            "vcomp": 0.3,
        })

        sig_uninterrupted = np.zeros((1, 6))
        epsp_uninterrupted = np.zeros(1)
        extra_uninterrupted = {
            "eps50": np.zeros((1, 6)),
            "uvar50": np.zeros((1, 6)),
            "compacted": np.zeros(1, dtype=bool),
            "eplas": np.zeros(1),
            "off50": np.ones(1),
        }

        dt = 1.0e-6
        deps_steps = [
            np.array([[0.001, -0.0003, -0.0003, 0.0002, 0.0, 0.0]]),
            np.array([[0.002, -0.0005, -0.0005, 0.0004, 0.0, 0.0]]),
            np.array([[0.003, -0.0008, -0.0008, 0.0006, 0.0, 0.0]]),
            np.array([[0.004, -0.0010, -0.0010, 0.0008, 0.0, 0.0]]),
        ]

        # Run 4 steps uninterrupted
        for deps in deps_steps:
            sig_uninterrupted, epsp_uninterrupted, _ = solid_update_law50(
                mat,
                sig_uninterrupted,
                deps=deps,
                epsp=epsp_uninterrupted,
                dt=dt,
                extra=extra_uninterrupted,
                return_tuple=True,
            )

        # Resumed run: 2 steps, snapshot restart state, deserialize and run next 2 steps
        sig_restarted = np.zeros((1, 6))
        epsp_restarted = np.zeros(1)
        extra_restarted = {
            "eps50": np.zeros((1, 6)),
            "uvar50": np.zeros((1, 6)),
            "compacted": np.zeros(1, dtype=bool),
            "eplas": np.zeros(1),
            "off50": np.ones(1),
        }

        for deps in deps_steps[:2]:
            sig_restarted, epsp_restarted, _ = solid_update_law50(
                mat,
                sig_restarted,
                deps=deps,
                epsp=epsp_restarted,
                dt=dt,
                extra=extra_restarted,
                return_tuple=True,
            )

        # Snapshot / serialize state
        snap = pickle.dumps({
            "sig": sig_restarted.copy(),
            "epsp": epsp_restarted.copy(),
            "extra": {k: v.copy() if isinstance(v, np.ndarray) else v for k, v in extra_restarted.items()},
        })

        # Restore
        restored = pickle.loads(snap)
        sig_resumed = restored["sig"]
        epsp_resumed = restored["epsp"]
        extra_resumed = restored["extra"]

        # Continue remaining 2 steps from restart
        for deps in deps_steps[2:]:
            sig_resumed, epsp_resumed, _ = solid_update_law50(
                mat,
                sig_resumed,
                deps=deps,
                epsp=epsp_resumed,
                dt=dt,
                extra=extra_resumed,
                return_tuple=True,
            )

        # Assert exact continuation within 10^-12
        np.testing.assert_allclose(sig_resumed, sig_uninterrupted, rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(epsp_resumed, epsp_uninterrupted, rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(extra_resumed["eps50"], extra_uninterrupted["eps50"], rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(extra_resumed["uvar50"], extra_uninterrupted["uvar50"], rtol=1e-12, atol=1e-12)
        np.testing.assert_array_equal(extra_resumed["compacted"], extra_uninterrupted["compacted"])
        np.testing.assert_allclose(extra_resumed["eplas"], extra_uninterrupted["eplas"], rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(extra_resumed["off50"], extra_uninterrupted["off50"], rtol=1e-12, atol=1e-12)
