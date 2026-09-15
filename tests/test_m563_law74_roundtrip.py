"""
Milestone M563: /MAT/LAW74 (/MAT/HILL_3D, /MAT/ORTH_PLAS, /MAT/THERM_HILL)
Exhaustive Roundtrip, Serialization, Boundary Cases & Negative Validation Auditor Suite.

Fortran origins:
  - starter/source/materials/mat/mat074/hm_read_mat74.F (card reader, parameters, default fallbacks)
  - engine/source/materials/mat/mat074/sigeps74.F (solid 3D Hill constitutive update)
  - engine/source/output/restart/wrrestp.F & rdresb.F
  - starter/source/restart/ddsplit/wrrest.F
  - hm_cfg_files/config/CFG/radioss120/MAT/matl74_74.cfg
"""

from __future__ import annotations

import json
import math
from pathlib import Path
import pickle
from typing import Any, Dict, List, Sequence, Tuple
import numpy as np
import pytest

from pyradioss.common.messages import MessageLog
from pyradioss.input.card_layouts import CARD_LAYOUTS, split_fixed
from pyradioss.input.deck_reader import read_deck
from pyradioss.input.deck_writer import StarterDeck
from pyradioss.input.starter_keywords import (
    parse_starter_deck,
    read_mat_law74,
    read_starter_deck,
)
import pyradioss.materials as materials
from pyradioss.model.entities import (
    MatHill3D,
    MatLaw74,
    MatOrthPlas,
    MaterialLaw74,
    Material,
    Part,
    Property,
)
from pyradioss.model.model import Model
from pyradioss.starter.checks import (
    _ALLOWED_LAWS,
    _MAT_CHECKS,
    check_mat_law74,
    check_materials,
    check_model,
)
from pyradioss.starter.restart import read_restart, write_restart
from pyradioss.starter.starter import run_starter


# ============================================================================
# Helpers
# ============================================================================

def _parse_deck_str(tmp_path: Path, text: str, name: str = "TEST_LAW74") -> Tuple[Model, MessageLog]:
    """Write deck text to file and parse into Model and MessageLog."""
    deck_path = tmp_path / f"{name}_0000.rad"
    deck_path.write_text(text.strip() + "\n", encoding="utf-8")
    blocks = read_deck(str(deck_path))
    model = Model()
    log = MessageLog()
    parse_starter_deck(blocks, model, log)
    return model, log


def _assert_law74_all_fields_exact(
    m: MatLaw74,
    mat: Material | None = None,
    *,
    rho: float,
    e: float,
    nu: float,
    rhor: float | None = None,
    eps_max: float = 1.0e30,
    epsr1: float = 1.0e30,
    epsr2: float = 2.0e30,
    ifunce: int = 0,
    einf: float = 0.0,
    ce: float = 0.0,
    fsmooth: int = 0,
    chard: float = 0.0,
    fcut: float = 0.0,
    s11y: float = 1.0,
    s22y: float = 1.0,
    s33y: float = 1.0,
    s12y: float = 1.0,
    s23y: float = 1.0,
    s31y: float = 1.0,
    table_id: int = 0,
    fscale: float = 1.0,
    pscale: float = 1.0,
    t0: float = 293.0,
    rhocp: float = 0.0,
    title: str = "",
) -> None:
    """Assert exact (floating-point tolerance) equality for all LAW74 parameters and derived properties."""
    expected_rhor = rhor if (rhor is not None and rhor != 0.0) else rho

    # 1. Primary Dataclass Fields & Aliases
    assert m.rho == pytest.approx(rho, rel=1e-6, abs=1e-12)
    assert m.rho0 == pytest.approx(rho, rel=1e-6, abs=1e-12)
    assert m.refer_rho == pytest.approx(expected_rhor, rel=1e-6, abs=1e-12)
    assert m.rhor == pytest.approx(expected_rhor, rel=1e-6, abs=1e-12)
    assert m.ref_rho == pytest.approx(expected_rhor, rel=1e-6, abs=1e-12)

    assert m.e == pytest.approx(e, rel=1e-6, abs=1e-12)
    assert m.E == pytest.approx(e, rel=1e-6, abs=1e-12)
    assert m.nu == pytest.approx(nu, rel=1e-6, abs=1e-12)
    assert m.Nu == pytest.approx(nu, rel=1e-6, abs=1e-12)

    assert m.eps_max == pytest.approx(eps_max, rel=1e-6, abs=1e-12)
    assert m.epsp_max == pytest.approx(eps_max, rel=1e-6, abs=1e-12)
    assert m.eps_p_max == pytest.approx(eps_max, rel=1e-6, abs=1e-12)

    assert m.epsr1 == pytest.approx(epsr1, rel=1e-6, abs=1e-12)
    assert m.epst1 == pytest.approx(epsr1, rel=1e-6, abs=1e-12)
    assert m.eps_t == pytest.approx(epsr1, rel=1e-6, abs=1e-12)

    assert m.epsr2 == pytest.approx(epsr2, rel=1e-6, abs=1e-12)
    assert m.epst2 == pytest.approx(epsr2, rel=1e-6, abs=1e-12)
    assert m.eps_m == pytest.approx(epsr2, rel=1e-6, abs=1e-12)

    assert m.ifunce == ifunce
    assert m.yr_fun == ifunce
    assert m.einf == pytest.approx(einf, rel=1e-6, abs=1e-12)
    assert m.efib == pytest.approx(einf, rel=1e-6, abs=1e-12)
    assert m.ce == pytest.approx(ce, rel=1e-6, abs=1e-12)
    assert m.c == pytest.approx(ce, rel=1e-6, abs=1e-12)

    assert m.fsmooth == fsmooth
    assert m.chard == pytest.approx(chard, rel=1e-6, abs=1e-12)
    assert m.c_hard == pytest.approx(chard, rel=1e-6, abs=1e-12)
    assert m.fisokin == pytest.approx(chard, rel=1e-6, abs=1e-12)
    assert m.fcut == pytest.approx(fcut, rel=1e-6, abs=1e-12)

    assert m.s11y == pytest.approx(s11y, rel=1e-6, abs=1e-12)
    assert m.sig11y == pytest.approx(s11y, rel=1e-6, abs=1e-12)
    assert m.sigma11y == pytest.approx(s11y, rel=1e-6, abs=1e-12)
    assert m.sigt1 == pytest.approx(s11y, rel=1e-6, abs=1e-12)

    assert m.s22y == pytest.approx(s22y, rel=1e-6, abs=1e-12)
    assert m.sig22y == pytest.approx(s22y, rel=1e-6, abs=1e-12)
    assert m.sigma22y == pytest.approx(s22y, rel=1e-6, abs=1e-12)
    assert m.sigt2 == pytest.approx(s22y, rel=1e-6, abs=1e-12)

    assert m.s33y == pytest.approx(s33y, rel=1e-6, abs=1e-12)
    assert m.sig33y == pytest.approx(s33y, rel=1e-6, abs=1e-12)
    assert m.sigma33y == pytest.approx(s33y, rel=1e-6, abs=1e-12)
    assert m.sigt3 == pytest.approx(s33y, rel=1e-6, abs=1e-12)

    assert m.s12y == pytest.approx(s12y, rel=1e-6, abs=1e-12)
    assert m.sig12y == pytest.approx(s12y, rel=1e-6, abs=1e-12)
    assert m.sigma12y == pytest.approx(s12y, rel=1e-6, abs=1e-12)
    assert m.sigyt1 == pytest.approx(s12y, rel=1e-6, abs=1e-12)

    assert m.s23y == pytest.approx(s23y, rel=1e-6, abs=1e-12)
    assert m.sig23y == pytest.approx(s23y, rel=1e-6, abs=1e-12)
    assert m.sigma23y == pytest.approx(s23y, rel=1e-6, abs=1e-12)
    assert m.sigyt2 == pytest.approx(s23y, rel=1e-6, abs=1e-12)

    assert m.s31y == pytest.approx(s31y, rel=1e-6, abs=1e-12)
    assert m.sig31y == pytest.approx(s31y, rel=1e-6, abs=1e-12)
    assert m.sigma31y == pytest.approx(s31y, rel=1e-6, abs=1e-12)
    assert m.sigyt3 == pytest.approx(s31y, rel=1e-6, abs=1e-12)

    assert m.table_id == table_id
    assert m.tab_id == table_id
    assert m.fun_a1 == table_id
    assert m.fscale == pytest.approx(fscale, rel=1e-6, abs=1e-12)
    assert m.sigma_scale == pytest.approx(fscale, rel=1e-6, abs=1e-12)
    assert m.pscale == pytest.approx(pscale, rel=1e-6, abs=1e-12)
    assert m.epspt_scale == pytest.approx(pscale, rel=1e-6, abs=1e-12)

    assert m.t0 == pytest.approx(t0, rel=1e-6, abs=1e-12)
    assert m.ti == pytest.approx(t0, rel=1e-6, abs=1e-12)
    assert m.t_initial == pytest.approx(t0, rel=1e-6, abs=1e-12)
    assert m.rhocp == pytest.approx(rhocp, rel=1e-6, abs=1e-12)
    assert m.rho0_cp == pytest.approx(rhocp, rel=1e-6, abs=1e-12)
    assert m.spheat == pytest.approx(rhocp, rel=1e-6, abs=1e-12)

    if title:
        assert m.title == title

    # 2. Derived Moduli & Wave Speeds
    expected_g = 0.5 * e / (1.0 + nu)
    expected_k = e / (3.0 * (1.0 - 2.0 * nu))
    assert m.G == pytest.approx(expected_g, rel=1e-6, abs=1e-12)
    assert m.bulk == pytest.approx(expected_k, rel=1e-6, abs=1e-12)
    assert m.K == pytest.approx(expected_k, rel=1e-6, abs=1e-12)

    expected_c_bulk = math.sqrt(e / rho)
    expected_c_solid = math.sqrt((expected_k + 4.0 * expected_g / 3.0) / rho)
    assert m.sound_speed == pytest.approx(expected_c_bulk, rel=1e-6, abs=1e-12)
    assert m.sound_speed_solid == pytest.approx(expected_c_solid, rel=1e-6, abs=1e-12)

    # 3. Hill 1948 Orthotropic Constants (hm_read_mat74.F)
    ff_exp = 0.5 * (1.0 / (s22y ** 2) + 1.0 / (s33y ** 2) - 1.0 / (s11y ** 2))
    gg_exp = 0.5 * (1.0 / (s11y ** 2) + 1.0 / (s33y ** 2) - 1.0 / (s22y ** 2))
    hh_exp = 0.5 * (1.0 / (s11y ** 2) + 1.0 / (s22y ** 2) - 1.0 / (s33y ** 2))
    ll_exp = 0.5 / (s23y ** 2)
    mm_exp = 0.5 / (s31y ** 2)
    nn_exp = 0.5 / (s12y ** 2)

    assert m.FF == pytest.approx(ff_exp, rel=1e-6, abs=1e-12)
    assert m.GG == pytest.approx(gg_exp, rel=1e-6, abs=1e-12)
    assert m.HH == pytest.approx(hh_exp, rel=1e-6, abs=1e-12)
    assert m.LL == pytest.approx(ll_exp, rel=1e-6, abs=1e-12)
    assert m.MM == pytest.approx(mm_exp, rel=1e-6, abs=1e-12)
    assert m.NN == pytest.approx(nn_exp, rel=1e-6, abs=1e-12)

    assert m.ff == pytest.approx(ff_exp, rel=1e-6, abs=1e-12)
    assert m.gg == pytest.approx(gg_exp, rel=1e-6, abs=1e-12)
    assert m.hh == pytest.approx(hh_exp, rel=1e-6, abs=1e-12)
    assert m.ll == pytest.approx(ll_exp, rel=1e-6, abs=1e-12)
    assert m.mm == pytest.approx(mm_exp, rel=1e-6, abs=1e-12)
    assert m.nn == pytest.approx(nn_exp, rel=1e-6, abs=1e-12)

    # 4. Validate Material entity in model.materials
    if mat is not None:
        assert mat.law == 74
        assert mat.rho0 == pytest.approx(rho, rel=1e-6, abs=1e-12)
        p = mat.params
        assert p["rho0"] == pytest.approx(rho, rel=1e-6, abs=1e-12)
        assert p["refer_rho"] == pytest.approx(expected_rhor, rel=1e-6, abs=1e-12)
        assert p["e"] == pytest.approx(e, rel=1e-6, abs=1e-12)
        assert p["E"] == pytest.approx(e, rel=1e-6, abs=1e-12)
        assert p["nu"] == pytest.approx(nu, rel=1e-6, abs=1e-12)
        assert p["Nu"] == pytest.approx(nu, rel=1e-6, abs=1e-12)
        assert p["eps_max"] == pytest.approx(eps_max, rel=1e-6, abs=1e-12)
        assert p["epsr1"] == pytest.approx(epsr1, rel=1e-6, abs=1e-12)
        assert p["epsr2"] == pytest.approx(epsr2, rel=1e-6, abs=1e-12)
        assert p["ifunce"] == ifunce
        assert p["einf"] == pytest.approx(einf, rel=1e-6, abs=1e-12)
        assert p["ce"] == pytest.approx(ce, rel=1e-6, abs=1e-12)
        assert p["fsmooth"] == fsmooth
        assert p["chard"] == pytest.approx(chard, rel=1e-6, abs=1e-12)
        assert p["fcut"] == pytest.approx(fcut, rel=1e-6, abs=1e-12)
        assert p["s11y"] == pytest.approx(s11y, rel=1e-6, abs=1e-12)
        assert p["s22y"] == pytest.approx(s22y, rel=1e-6, abs=1e-12)
        assert p["s33y"] == pytest.approx(s33y, rel=1e-6, abs=1e-12)
        assert p["s12y"] == pytest.approx(s12y, rel=1e-6, abs=1e-12)
        assert p["s23y"] == pytest.approx(s23y, rel=1e-6, abs=1e-12)
        assert p["s31y"] == pytest.approx(s31y, rel=1e-6, abs=1e-12)
        assert p["table_id"] == table_id
        assert p["fscale"] == pytest.approx(fscale, rel=1e-6, abs=1e-12)
        assert p["pscale"] == pytest.approx(pscale, rel=1e-6, abs=1e-12)
        assert p["t0"] == pytest.approx(t0, rel=1e-6, abs=1e-12)
        assert p["rhocp"] == pytest.approx(rhocp, rel=1e-6, abs=1e-12)
        if title:
            assert mat.title == title


# ============================================================================
# 1. Exact Card Generation & Fixed vs Free Formats
# ============================================================================

class TestLaw74CardGeneration:
    """Audit StarterDeck.mat_law74 exact 8-card fixed-format and free-format generation."""

    def test_fixed_format_card_columns_match_cfg_layouts(self):
        """Verify that all 8 cards emitted by StarterDeck.mat_law74 match exact CARD_LAYOUTS widths."""
        deck = StarterDeck("CARD_TEST")
        deck.mat_law74(
            mid=74,
            title="Orthotropic Solid Hill 3D",
            rho=7.8e-9,
            rhor=7.85e-9,
            e=210000.0,
            nu=0.30,
            eps_max=0.55,
            epsr1=0.22,
            epsr2=0.38,
            ifunce=2,
            einf=190000.0,
            ce=12.5,
            fsmooth=1,
            chard=0.25,
            fcut=5500.0,
            s11y=410.0,
            s22y=430.0,
            s33y=460.0,
            s12y=235.0,
            s23y=245.0,
            s31y=255.0,
            table_id=202,
            fscale=1.05,
            pscale=0.95,
            t0=305.0,
            rhocp=3.8e-3,
            fixed_format=True,
        )

        lines = deck.lines
        header_idx = lines.index("/MAT/LAW74/74")
        assert lines[header_idx + 1] == "Orthotropic Solid Hill 3D"

        # Cards 1 to 8
        card1 = lines[header_idx + 2]
        card2 = lines[header_idx + 3]
        card3 = lines[header_idx + 4]
        card4 = lines[header_idx + 5]
        card5 = lines[header_idx + 6]
        card6 = lines[header_idx + 7]
        card7 = lines[header_idx + 8]
        card8 = lines[header_idx + 9]

        # Verify layout splits match CARD_LAYOUTS
        f1 = split_fixed(card1, CARD_LAYOUTS["MAT_LAW74_1"])
        assert float(f1[0]) == pytest.approx(7.8e-9)
        assert float(f1[1]) == pytest.approx(7.85e-9)

        f2 = split_fixed(card2, CARD_LAYOUTS["MAT_LAW74_2"])
        assert float(f2[0]) == pytest.approx(210000.0)
        assert float(f2[1]) == pytest.approx(0.30)
        assert float(f2[2]) == pytest.approx(0.55)
        assert float(f2[3]) == pytest.approx(0.22)
        assert float(f2[4]) == pytest.approx(0.38)

        f3 = split_fixed(card3, CARD_LAYOUTS["MAT_LAW74_3"])
        assert int(f3[0]) == 2
        assert f3[1].strip() == ""
        assert float(f3[2]) == pytest.approx(190000.0)
        assert float(f3[3]) == pytest.approx(12.5)

        f4 = split_fixed(card4, CARD_LAYOUTS["MAT_LAW74_4"])
        assert f4[0].strip() == ""
        assert int(f4[1]) == 1
        assert float(f4[2]) == pytest.approx(0.25)
        assert float(f4[3]) == pytest.approx(5500.0)

        f5 = split_fixed(card5, CARD_LAYOUTS["MAT_LAW74_5"])
        assert float(f5[0]) == pytest.approx(410.0)
        assert float(f5[1]) == pytest.approx(430.0)
        assert float(f5[2]) == pytest.approx(460.0)

        f6 = split_fixed(card6, CARD_LAYOUTS["MAT_LAW74_6"])
        assert float(f6[0]) == pytest.approx(235.0)
        assert float(f6[1]) == pytest.approx(245.0)
        assert float(f6[2]) == pytest.approx(255.0)

        f7 = split_fixed(card7, CARD_LAYOUTS["MAT_LAW74_7"])
        assert int(f7[0]) == 202
        assert f7[1].strip() == ""
        assert float(f7[2]) == pytest.approx(1.05)
        assert float(f7[3]) == pytest.approx(0.95)

        f8 = split_fixed(card8, CARD_LAYOUTS["MAT_LAW74_8"])
        assert float(f8[0]) == pytest.approx(305.0)
        assert float(f8[1]) == pytest.approx(3.8e-3)

    def test_free_format_card_generation_tokens(self):
        """Verify tokens emitted in free format match parameter sequence."""
        deck = StarterDeck("FREE_TEST")
        deck.mat_law74(
            mid=10,
            title="Free Format Solid",
            rho=7.8e-9,
            rhor=7.8e-9,
            e=210000.0,
            nu=0.30,
            eps_max=0.5,
            epsr1=0.2,
            epsr2=0.35,
            ifunce=1,
            einf=180000.0,
            ce=10.0,
            fsmooth=2,
            chard=0.3,
            fcut=5000.0,
            s11y=400.0,
            s22y=420.0,
            s33y=450.0,
            s12y=230.0,
            s23y=240.0,
            s31y=250.0,
            table_id=101,
            fscale=1.1,
            pscale=0.9,
            t0=295.0,
            rhocp=3.5e-3,
            fixed_format=False,
            comma=True,
        )

        lines = deck.lines
        h_idx = lines.index("/MAT/LAW74/10")
        assert lines[h_idx + 1] == "Free Format Solid"

        # Check line contents
        c1_toks = [float(x) for x in lines[h_idx + 2].split(",")]
        assert c1_toks == [pytest.approx(7.8e-9)]

        c2_toks = [float(x) for x in lines[h_idx + 3].split(",")]
        assert c2_toks == [pytest.approx(210000.0), pytest.approx(0.3), pytest.approx(0.5), pytest.approx(0.2), pytest.approx(0.35)]

        c3_toks = [float(x) for x in lines[h_idx + 4].split(",")]
        assert c3_toks == [1, pytest.approx(180000.0), pytest.approx(10.0)]

        c4_toks = [float(x) for x in lines[h_idx + 5].split(",")]
        assert c4_toks == [2, pytest.approx(0.3), pytest.approx(5000.0)]

        c5_toks = [float(x) for x in lines[h_idx + 6].split(",")]
        assert c5_toks == [pytest.approx(400.0), pytest.approx(420.0), pytest.approx(450.0)]

        c6_toks = [float(x) for x in lines[h_idx + 7].split(",")]
        assert c6_toks == [pytest.approx(230.0), pytest.approx(240.0), pytest.approx(250.0)]

        c7_toks = [float(x) for x in lines[h_idx + 8].split(",")]
        assert c7_toks == [101, pytest.approx(1.1), pytest.approx(0.9)]

        c8_toks = [float(x) for x in lines[h_idx + 9].split(",")]
        assert c8_toks == [pytest.approx(295.0), pytest.approx(3.5e-3)]

    def test_card_generation_from_matlaw74_instance(self):
        """Verify StarterDeck.mat_law74 accepts a MatLaw74 dataclass instance directly."""
        m = MatLaw74(
            id=740,
            title="Instance Emission",
            rho=7.8e-9,
            refer_rho=7.9e-9,
            e=210000.0,
            nu=0.29,
            eps_max=0.6,
            epsr1=0.25,
            epsr2=0.40,
            ifunce=3,
            einf=185000.0,
            ce=15.0,
            fsmooth=1,
            chard=0.28,
            fcut=5200.0,
            s11y=405.0,
            s22y=425.0,
            s33y=455.0,
            s12y=232.0,
            s23y=242.0,
            s31y=252.0,
            table_id=303,
            fscale=1.02,
            pscale=0.98,
            t0=302.0,
            rhocp=3.6e-3,
        )

        deck = StarterDeck("FROM_OBJ")
        deck.mat_law74(m, fixed_format=True)

        lines = deck.lines
        h_idx = lines.index("/MAT/LAW74/740")
        assert lines[h_idx + 1] == "Instance Emission"

        card1 = lines[h_idx + 2]
        f1 = split_fixed(card1, CARD_LAYOUTS["MAT_LAW74_1"])
        assert float(f1[0]) == pytest.approx(7.8e-9)
        assert float(f1[1]) == pytest.approx(7.9e-9)

        card7 = lines[h_idx + 8]
        f7 = split_fixed(card7, CARD_LAYOUTS["MAT_LAW74_7"])
        assert int(f7[0]) == 303
        assert float(f7[2]) == pytest.approx(1.02)
        assert float(f7[3]) == pytest.approx(0.98)


# ============================================================================
# 2. Fixed-Format Deck Roundtrip
# ============================================================================

class TestLaw74FixedFormatRoundtrip:
    """Audit fixed-format roundtrip: write with StarterDeck.mat_law74, read back with read_mat_law74."""

    def test_full_8card_standard_fixed_format_roundtrip(self, tmp_path: Path):
        """Full roundtrip with all 24 parameters in standard 8-card fixed-format."""
        deck = StarterDeck("ROUNDTRIP_STD8")
        deck.mat_law74(
            mid=74,
            title="Complete Law74 Parameters",
            rho=7.8e-9,
            rhor=7.85e-9,
            e=210000.0,
            nu=0.30,
            eps_max=0.55,
            epsr1=0.22,
            epsr2=0.38,
            ifunce=2,
            einf=190000.0,
            ce=12.5,
            fsmooth=1,
            chard=0.25,
            fcut=5500.0,
            s11y=410.0,
            s22y=430.0,
            s33y=460.0,
            s12y=235.0,
            s23y=245.0,
            s31y=255.0,
            table_id=202,
            fscale=1.05,
            pscale=0.95,
            t0=305.0,
            rhocp=3.8e-3,
            fixed_format=True,
        )

        model, log = _parse_deck_str(tmp_path, deck.render(), name="STD8")
        assert not log.has_errors
        assert 74 in model.mat_law74s
        assert 74 in model.materials

        m = model.mat_law74s[74]
        mat = model.materials[74]
        _assert_law74_all_fields_exact(
            m,
            mat,
            rho=7.8e-9,
            rhor=7.85e-9,
            e=210000.0,
            nu=0.30,
            eps_max=0.55,
            epsr1=0.22,
            epsr2=0.38,
            ifunce=2,
            einf=190000.0,
            ce=12.5,
            fsmooth=1,
            chard=0.25,
            fcut=5500.0,
            s11y=410.0,
            s22y=430.0,
            s33y=460.0,
            s12y=235.0,
            s23y=245.0,
            s31y=255.0,
            table_id=202,
            fscale=1.05,
            pscale=0.95,
            t0=305.0,
            rhocp=3.8e-3,
            title="Complete Law74 Parameters",
        )

    def test_optional_card8_omitted_fixed_format_roundtrip(self, tmp_path: Path):
        """Verify 7-card standard format (Card 8 T0, RHOCP omitted) defaults to t0=293.0, rhocp=0.0."""
        deck_text = """
/MAT/LAW74/101
Card 8 Omitted Format
               7.8e-9              7.8e-9
             210000.0                 0.3                 0.5                 0.2                0.35
         2                                       180000.0                10.0
                    1                                0.25              5000.0
                400.0               420.0               450.0
                230.0               240.0               250.0
       101                                           1.10                0.90
/END
"""
        model, log = _parse_deck_str(tmp_path, deck_text, name="CARD8_OMITTED")
        assert not log.has_errors
        assert 101 in model.mat_law74s

        m = model.mat_law74s[101]
        mat = model.materials[101]
        _assert_law74_all_fields_exact(
            m,
            mat,
            rho=7.8e-9,
            rhor=7.8e-9,
            e=210000.0,
            nu=0.3,
            eps_max=0.5,
            epsr1=0.2,
            epsr2=0.35,
            ifunce=2,
            einf=180000.0,
            ce=10.0,
            fsmooth=1,
            chard=0.25,
            fcut=5000.0,
            s11y=400.0,
            s22y=420.0,
            s33y=450.0,
            s12y=230.0,
            s23y=240.0,
            s31y=250.0,
            table_id=101,
            fscale=1.1,
            pscale=0.9,
            t0=293.0,     # default
            rhocp=0.0,    # default
            title="Card 8 Omitted Format",
        )

    def test_legacy_7card_radioss110_format_roundtrip(self, tmp_path: Path):
        """Verify legacy radioss110 7-card format without Card 3 (Yr_fun / Einf / Ce)."""
        deck_text = """
/MAT/LAW74/102
Legacy Radioss110 Format
               7.8e-9              7.8e-9
             205000.0                0.29                 0.6                0.25                0.40
                    1                                 0.3              6000.0
                390.0               410.0               430.0
                220.0               230.0               240.0
       201                                           1.05                0.95
                298.0              3.2e-3
/END
"""
        model, log = _parse_deck_str(tmp_path, deck_text, name="LEGACY7")
        assert not log.has_errors
        assert 102 in model.mat_law74s

        m = model.mat_law74s[102]
        mat = model.materials[102]
        _assert_law74_all_fields_exact(
            m,
            mat,
            rho=7.8e-9,
            rhor=7.8e-9,
            e=205000.0,
            nu=0.29,
            eps_max=0.6,
            epsr1=0.25,
            epsr2=0.40,
            ifunce=0,     # omitted in legacy format
            einf=0.0,     # omitted in legacy format
            ce=0.0,       # omitted in legacy format
            fsmooth=1,
            chard=0.3,
            fcut=6000.0,
            s11y=390.0,
            s22y=410.0,
            s33y=430.0,
            s12y=220.0,
            s23y=230.0,
            s31y=240.0,
            table_id=201,
            fscale=1.05,
            pscale=0.95,
            t0=298.0,
            rhocp=3.2e-3,
            title="Legacy Radioss110 Format",
        )

    def test_minimal_card_defaults_roundtrip(self, tmp_path: Path):
        """Verify minimal 2-card format sets all defaults accurately per hm_read_mat74.F."""
        deck_text = """
/MAT/LAW74/103
Minimal Fixed Deck
               7.8e-9
             210000.0                 0.3
/END
"""
        model, log = _parse_deck_str(tmp_path, deck_text, name="MINIMAL")
        assert not log.has_errors
        assert 103 in model.mat_law74s

        m = model.mat_law74s[103]
        _assert_law74_all_fields_exact(
            m,
            rho=7.8e-9,
            rhor=7.8e-9,
            e=210000.0,
            nu=0.3,
            eps_max=1.0e30,
            epsr1=1.0e30,
            epsr2=2.0e30,
            ifunce=0,
            einf=0.0,
            ce=0.0,
            fsmooth=0,
            chard=0.0,
            fcut=0.0,
            s11y=1.0,
            s22y=1.0,
            s33y=1.0,
            s12y=1.0,
            s23y=1.0,
            s31y=1.0,
            table_id=0,
            fscale=1.0,
            pscale=1.0,
            t0=293.0,
            rhocp=0.0,
            title="Minimal Fixed Deck",
        )


# ============================================================================
# 3. Free-Format Deck Roundtrip
# ============================================================================

class TestLaw74FreeFormatRoundtrip:
    """Audit free-format roundtrip: comma and space separated values."""

    def test_full_8card_free_format_comma_separated(self, tmp_path: Path):
        """Roundtrip free format with comma delimiters."""
        deck_text = """
/MAT/LAW74/50
Free Format Comma Deck
7.8e-9, 7.85e-9
210000.0, 0.30, 0.55, 0.22, 0.38
2, 190000.0, 12.5
1, 0.25, 5500.0
410.0, 430.0, 460.0
235.0, 245.0, 255.0
202, 1.05, 0.95
305.0, 3.8e-3
/END
"""
        model, log = _parse_deck_str(tmp_path, deck_text, name="FREE_COMMA")
        assert not log.has_errors
        assert 50 in model.mat_law74s

        m = model.mat_law74s[50]
        mat = model.materials[50]
        _assert_law74_all_fields_exact(
            m,
            mat,
            rho=7.8e-9,
            rhor=7.85e-9,
            e=210000.0,
            nu=0.30,
            eps_max=0.55,
            epsr1=0.22,
            epsr2=0.38,
            ifunce=2,
            einf=190000.0,
            ce=12.5,
            fsmooth=1,
            chard=0.25,
            fcut=5500.0,
            s11y=410.0,
            s22y=430.0,
            s33y=460.0,
            s12y=235.0,
            s23y=245.0,
            s31y=255.0,
            table_id=202,
            fscale=1.05,
            pscale=0.95,
            t0=305.0,
            rhocp=3.8e-3,
            title="Free Format Comma Deck",
        )

    def test_full_8card_free_format_space_separated(self, tmp_path: Path):
        """Roundtrip free format with space delimiters."""
        deck_text = """
/MAT/LAW74/51
Free Format Space Deck
7.8e-9 7.85e-9
210000.0 0.30 0.55 0.22 0.38
2 190000.0 12.5
1 0.25 5500.0
410.0 430.0 460.0
235.0 245.0 255.0
202 1.05 0.95
305.0 3.8e-3
/END
"""
        model, log = _parse_deck_str(tmp_path, deck_text, name="FREE_SPACE")
        assert not log.has_errors
        assert 51 in model.mat_law74s

        m = model.mat_law74s[51]
        _assert_law74_all_fields_exact(
            m,
            rho=7.8e-9,
            rhor=7.85e-9,
            e=210000.0,
            nu=0.30,
            eps_max=0.55,
            epsr1=0.22,
            epsr2=0.38,
            ifunce=2,
            einf=190000.0,
            ce=12.5,
            fsmooth=1,
            chard=0.25,
            fcut=5500.0,
            s11y=410.0,
            s22y=430.0,
            s33y=460.0,
            s12y=235.0,
            s23y=245.0,
            s31y=255.0,
            table_id=202,
            fscale=1.05,
            pscale=0.95,
            t0=305.0,
            rhocp=3.8e-3,
            title="Free Format Space Deck",
        )

    def test_free_format_legacy_7card_radioss110(self, tmp_path: Path):
        """Roundtrip free format legacy 7-card layout (where card 5 is table_id and card 6 is t0, rhocp)."""
        deck_text = """
/MAT/LAW74/52
Free Format Legacy 7 Card
7.8e-9 7.8e-9
210000.0 0.3 0.5 0.2 0.35
1 0.25 5000.0
400.0 420.0 450.0
230.0 240.0 250.0
101 1.0 1.0
295.0 3.5e-3
/END
"""
        model, log = _parse_deck_str(tmp_path, deck_text, name="FREE_LEGACY7")
        assert not log.has_errors
        assert 52 in model.mat_law74s

        m = model.mat_law74s[52]
        _assert_law74_all_fields_exact(
            m,
            rho=7.8e-9,
            rhor=7.8e-9,
            e=210000.0,
            nu=0.3,
            eps_max=0.5,
            epsr1=0.2,
            epsr2=0.35,
            ifunce=0,
            einf=0.0,
            ce=0.0,
            fsmooth=1,
            chard=0.25,
            fcut=5000.0,
            s11y=400.0,
            s22y=420.0,
            s33y=450.0,
            s12y=230.0,
            s23y=240.0,
            s31y=250.0,
            table_id=101,
            fscale=1.0,
            pscale=1.0,
            t0=295.0,
            rhocp=3.5e-3,
            title="Free Format Legacy 7 Card",
        )


# ============================================================================
# 4. Keyword Synonyms & Container Invariants
# ============================================================================

class TestLaw74Synonyms:
    """Audit all keyword synonyms: /MAT/LAW74, /MAT/HILL_3D, /MAT/ORTH_PLAS, /MAT/THERM_HILL, /MAT/74."""

    def test_all_keyword_synonyms_populate_containers(self, tmp_path: Path):
        """Verify that every synonym parses into model.mat_law74s, alias containers, and model.materials."""
        deck_text = """
/MAT/LAW74/1
Standard LAW74
7.8e-9
210000.0, 0.30, 0.5, 0.2, 0.35
0, 0.0, 0.0
0, 0.0, 0.0
400.0, 420.0, 450.0
230.0, 240.0, 250.0
101, 1.0, 1.0
295.0, 3.5e-3

/MAT/HILL_3D/2
Hill 3D Keyword
7.8e-9
210000.0, 0.30, 0.5, 0.2, 0.35
0, 0.0, 0.0
0, 0.0, 0.0
400.0, 420.0, 450.0
230.0, 240.0, 250.0
101, 1.0, 1.0
295.0, 3.5e-3

/MAT/ORTH_PLAS/3
Orth Plas Keyword
7.8e-9
210000.0, 0.30, 0.5, 0.2, 0.35
0, 0.0, 0.0
0, 0.0, 0.0
400.0, 420.0, 450.0
230.0, 240.0, 250.0
101, 1.0, 1.0
295.0, 3.5e-3

/MAT/THERM_HILL/4
Therm Hill Keyword Solid Format
7.8e-9
210000.0, 0.30, 0.5, 0.2, 0.35
0, 0.0, 0.0
0, 0.0, 0.0
400.0, 420.0, 450.0
230.0, 240.0, 250.0
101, 1.0, 1.0
295.0, 3.5e-3

/MAT/LAW74/5/1
Law 74 with Unit ID
7.8e-9
210000.0, 0.30, 0.5, 0.2, 0.35
0, 0.0, 0.0
0, 0.0, 0.0
400.0, 420.0, 450.0
230.0, 240.0, 250.0
101, 1.0, 1.0
295.0, 3.5e-3

/MAT/74
Numeric 74 Keyword
7.8e-9
210000.0, 0.30, 0.5, 0.2, 0.35
0, 0.0, 0.0
0, 0.0, 0.0
400.0, 420.0, 450.0
230.0, 240.0, 250.0
101, 1.0, 1.0
295.0, 3.5e-3
"""
        model, log = _parse_deck_str(tmp_path, deck_text, name="SYNONYMS")
        assert not log.has_errors
        assert set(model.mat_law74s.keys()) == {1, 2, 3, 4, 5, 74}
        assert set(model.mat_hill_3ds.keys()) == {1, 2, 3, 4, 5, 74}
        assert set(model.mat_orth_plass.keys()) == {1, 2, 3, 4, 5, 74}
        assert set(model.mat_hill_therms.keys()) == {1, 2, 3, 4, 5, 74}
        assert set(model.materials.keys()) == {1, 2, 3, 4, 5, 74}

        # Check entity types
        for mid in (1, 2, 3, 4, 5, 74):
            m = model.mat_law74s[mid]
            assert isinstance(m, MatLaw74)
            assert isinstance(m, MatHill3D)
            assert isinstance(m, MatOrthPlas)
            assert isinstance(m, MaterialLaw74)
            assert model.materials[mid].law == 74

    def test_starter_deck_writer_synonym_helpers(self):
        """Verify StarterDeck.mat_hill_3d and mat_orth_plas write correct keyword headers."""
        deck = StarterDeck("SYN_WRITER")
        deck.mat_hill_3d(mid=10, title="Hill 3D Helper", rho=7.8e-9, e=210000.0, nu=0.3)
        deck.mat_orth_plas(mid=20, title="Orth Plas Helper", rho=7.8e-9, e=210000.0, nu=0.3)

        lines = deck.lines
        assert "/MAT/HILL_3D/10" in lines
        assert "/MAT/ORTH_PLAS/20" in lines


# ============================================================================
# 5. Restart File Serialization & State Invariants
# ============================================================================

class TestLaw74Serialization:
    """Audit MatLaw74 dataclass pickling, JSON/dict serialization, and write_restart/read_restart with uvar74 & temp."""

    def test_pickle_mat_law74_fidelity(self):
        """Verify MatLaw74 dataclass pickling fidelity across protocols."""
        m = MatLaw74(
            id=74,
            title="Pickle-Law74",
            rho=7.8e-9,
            refer_rho=7.85e-9,
            e=210000.0,
            nu=0.30,
            eps_max=0.55,
            epsr1=0.22,
            epsr2=0.38,
            ifunce=2,
            einf=190000.0,
            ce=12.5,
            fsmooth=1,
            chard=0.25,
            fcut=5500.0,
            s11y=410.0,
            s22y=430.0,
            s33y=460.0,
            s12y=235.0,
            s23y=245.0,
            s31y=255.0,
            table_id=202,
            fscale=1.05,
            pscale=0.95,
            t0=305.0,
            rhocp=3.8e-3,
        )

        for protocol in (4, 5, pickle.HIGHEST_PROTOCOL):
            data = pickle.dumps(m, protocol=protocol)
            restored = pickle.loads(data)
            assert isinstance(restored, MatLaw74)
            _assert_law74_all_fields_exact(
                restored,
                rho=7.8e-9,
                rhor=7.85e-9,
                e=210000.0,
                nu=0.30,
                eps_max=0.55,
                epsr1=0.22,
                epsr2=0.38,
                ifunce=2,
                einf=190000.0,
                ce=12.5,
                fsmooth=1,
                chard=0.25,
                fcut=5500.0,
                s11y=410.0,
                s22y=430.0,
                s33y=460.0,
                s12y=235.0,
                s23y=245.0,
                s31y=255.0,
                table_id=202,
                fscale=1.05,
                pscale=0.95,
                t0=305.0,
                rhocp=3.8e-3,
                title="Pickle-Law74",
            )

    def test_json_and_dict_serialization_fidelity(self):
        """Verify MatLaw74 params dictionary serializes to JSON and restores accurately without loss."""
        m = MatLaw74(
            id=74,
            title="JSON-Law74",
            rho=7.8e-9,
            refer_rho=7.85e-9,
            e=210000.0,
            nu=0.30,
            eps_max=0.55,
            epsr1=0.22,
            epsr2=0.38,
            ifunce=2,
            einf=190000.0,
            ce=12.5,
            fsmooth=1,
            chard=0.25,
            fcut=5500.0,
            s11y=410.0,
            s22y=430.0,
            s33y=460.0,
            s12y=235.0,
            s23y=245.0,
            s31y=255.0,
            table_id=202,
            fscale=1.05,
            pscale=0.95,
            t0=305.0,
            rhocp=3.8e-3,
        )

        p = m.params
        json_str = json.dumps(p)
        loaded_dict = json.loads(json_str)

        # Reconstruct MatLaw74 from JSON dictionary
        restored = MatLaw74(**loaded_dict)
        _assert_law74_all_fields_exact(
            restored,
            rho=7.8e-9,
            rhor=7.85e-9,
            e=210000.0,
            nu=0.30,
            eps_max=0.55,
            epsr1=0.22,
            epsr2=0.38,
            ifunce=2,
            einf=190000.0,
            ce=12.5,
            fsmooth=1,
            chard=0.25,
            fcut=5500.0,
            s11y=410.0,
            s22y=430.0,
            s33y=460.0,
            s12y=235.0,
            s23y=245.0,
            s31y=255.0,
            table_id=202,
            fscale=1.05,
            pscale=0.95,
            t0=305.0,
            rhocp=3.8e-3,
            title="JSON-Law74",
        )

    def test_write_read_restart_preserves_uvar74_and_temp(self, tmp_path: Path):
        """Verify write_restart and read_restart preserve solid state variables uvar74 (n, 10) and temp (n,)."""
        deck = StarterDeck("RST_LAW74_SOLID")
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
        deck.part(1, "SolidPart", prop_id=1, mat_id=74)
        deck.mat_law74(
            mid=74,
            title="Restart-Law74-Solid",
            rho=7.8e-9,
            e=210000.0,
            nu=0.30,
            s11y=400.0,
            s22y=420.0,
            s33y=450.0,
            s12y=230.0,
            s23y=240.0,
            s31y=250.0,
            t0=300.0,
            rhocp=3.5e-3,
        )

        rad_path = tmp_path / "rst_law74_0000.rad"
        deck.write(str(rad_path))

        model = run_starter(str(rad_path))
        assert model is not None

        groups = dict(model.element_groups())
        solid_grp = groups.get("bricks")
        assert solid_grp is not None
        st = solid_grp.state

        # Synthesize realistic persistent 3D Hill state arrays: uvar74 (2, 10) and temp (2,)
        uvar74_synth = np.array([
            [0.012, 150.0, 120.0, -100.0, 45.0, 35.0, -25.0, 1.0, 0.0, 0.0],
            [0.024, 180.0, 140.0, -110.0, 50.0, 40.0, -30.0, 1.0, 0.0, 0.0],
        ], dtype=float)

        temp_synth = np.array([315.5, 328.2], dtype=float)

        st["mat_extra"]["uvar74"] = uvar74_synth.copy()
        st["mat_extra"]["temp"] = temp_synth.copy()
        st["uvar74"] = uvar74_synth.copy()
        st["temp"] = temp_synth.copy()

        # Write restart .rst
        rst_path = tmp_path / "rst_law74_0001.rst"
        engine_dict = {
            "cycle": 250,
            "t": 0.005,
            "dt": 2.5e-8,
        }
        write_restart(model, str(rst_path), engine=engine_dict)

        # Read back from restart .rst
        rest_model, rest_engine = read_restart(str(rst_path))
        assert rest_engine["cycle"] == 250
        assert rest_engine["t"] == pytest.approx(0.005)
        assert rest_engine["dt"] == pytest.approx(2.5e-8)

        # Assert exact preservation of LAW74 state arrays
        rest_solid = dict(rest_model.element_groups()).get("bricks")
        assert rest_solid is not None
        rest_st = rest_solid.state
        rest_extra = rest_st["mat_extra"]

        np.testing.assert_array_equal(rest_st["uvar74"], uvar74_synth)
        np.testing.assert_array_equal(rest_st["temp"], temp_synth)
        np.testing.assert_array_equal(rest_extra["uvar74"], uvar74_synth)
        np.testing.assert_array_equal(rest_extra["temp"], temp_synth)

    def test_constitutive_cycle_continuation_from_restart(self):
        """Constitutive solid update continuation from restart matches uninterrupted simulation within 10^-12."""
        mat = Material(
            id=74,
            law=74,
            law_name="LAW74",
            rho0=7.8e-9,
            params={
                "e": 210000.0,
                "nu": 0.30,
                "s11y": 400.0,
                "s22y": 420.0,
                "s33y": 450.0,
                "s12y": 230.0,
                "s23y": 240.0,
                "s31y": 250.0,
                "chard": 0.3,
                "t0": 293.0,
            },
        )

        dt = 1.0e-6
        deps_steps = [
            np.array([[0.002, -0.0006, -0.0006, 0.0002, 0.0, 0.0]]),
            np.array([[0.003, -0.0009, -0.0009, 0.0003, 0.0, 0.0]]),
            np.array([[0.004, -0.0012, -0.0012, 0.0004, 0.0, 0.0]]),
            np.array([[0.005, -0.0015, -0.0015, 0.0005, 0.0, 0.0]]),
        ]

        # 1. Uninterrupted run: 4 steps
        sig_uninterrupted = np.zeros((1, 6))
        epsp_uninterrupted = np.zeros(1)
        extra_uninterrupted = {
            "uvar74": np.zeros((1, 10)),
            "temp": np.array([293.0]),
            "rho": np.array([7.8e-9]),
        }

        for deps in deps_steps:
            sig_uninterrupted, epsp_uninterrupted, _ = materials.solid_update(
                mat,
                sig_uninterrupted,
                deps,
                epsp=epsp_uninterrupted,
                dt=dt,
                extra=extra_uninterrupted,
            )

        # 2. Resumed run: 2 steps, snapshot/restart, continue 2 steps
        sig_restarted = np.zeros((1, 6))
        epsp_restarted = np.zeros(1)
        extra_restarted = {
            "uvar74": np.zeros((1, 10)),
            "temp": np.array([293.0]),
            "rho": np.array([7.8e-9]),
        }

        for deps in deps_steps[:2]:
            sig_restarted, epsp_restarted, _ = materials.solid_update(
                mat,
                sig_restarted,
                deps,
                epsp=epsp_restarted,
                dt=dt,
                extra=extra_restarted,
            )

        # Snapshot state via pickle (simulating restart file serialization)
        snap = pickle.dumps({
            "sig": sig_restarted.copy(),
            "epsp": epsp_restarted.copy(),
            "extra": {k: v.copy() if isinstance(v, np.ndarray) else v for k, v in extra_restarted.items()},
        })

        # Restore from snapshot
        restored = pickle.loads(snap)
        sig_resumed = restored["sig"]
        epsp_resumed = restored["epsp"]
        extra_resumed = restored["extra"]

        # Continue next 2 steps
        for deps in deps_steps[2:]:
            sig_resumed, epsp_resumed, _ = materials.solid_update(
                mat,
                sig_resumed,
                deps,
                epsp=epsp_resumed,
                dt=dt,
                extra=extra_resumed,
            )

        # Assert exact continuation within 10^-12
        np.testing.assert_allclose(sig_resumed, sig_uninterrupted, rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(epsp_resumed, epsp_uninterrupted, rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(extra_resumed["uvar74"], extra_uninterrupted["uvar74"], rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(extra_resumed["temp"], extra_uninterrupted["temp"], rtol=1e-12, atol=1e-12)


# ============================================================================
# 6. Negative Starter Diagnostics
# ============================================================================

class TestLaw74NegativeValidation:
    """Audit starter negative diagnostics in check_mat_law74 and check_materials."""

    def test_ancmsg_1514_negative_and_zero_density(self):
        """ANCMSG 1514: Error when initial density RHO <= 0."""
        log1 = MessageLog()
        m1 = MatLaw74(id=1, rho=0.0, e=210000.0, nu=0.3, s11y=400.0, s22y=420.0, s33y=450.0, s12y=230.0, s23y=240.0, s31y=250.0)
        check_mat_law74(mat=m1, log=log1)
        assert log1.has_errors
        assert any("ANCMSG 1514" in str(err) and "initial density RHO must be > 0" in str(err) for err in log1.errors)

        log2 = MessageLog()
        m2 = MatLaw74(id=2, rho=-7.8e-9, e=210000.0, nu=0.3, s11y=400.0, s22y=420.0, s33y=450.0, s12y=230.0, s23y=240.0, s31y=250.0)
        check_mat_law74(mat=m2, log=log2)
        assert log2.has_errors
        assert any("ANCMSG 1514" in str(err) and "initial density RHO must be > 0" in str(err) for err in log2.errors)

    def test_ancmsg_1514_negative_and_zero_young(self):
        """ANCMSG 1514: Error when Young's modulus E <= 0."""
        log1 = MessageLog()
        m1 = MatLaw74(id=1, rho=7.8e-9, e=0.0, nu=0.3, s11y=400.0, s22y=420.0, s33y=450.0, s12y=230.0, s23y=240.0, s31y=250.0)
        check_mat_law74(mat=m1, log=log1)
        assert log1.has_errors
        assert any("ANCMSG 1514" in str(err) and "Young's modulus E must be > 0" in str(err) for err in log1.errors)

        log2 = MessageLog()
        m2 = MatLaw74(id=2, rho=7.8e-9, e=-210000.0, nu=0.3, s11y=400.0, s22y=420.0, s33y=450.0, s12y=230.0, s23y=240.0, s31y=250.0)
        check_mat_law74(mat=m2, log=log2)
        assert log2.has_errors
        assert any("ANCMSG 1514" in str(err) and "Young's modulus E must be > 0" in str(err) for err in log2.errors)

    def test_ancmsg_1514_poisson_ratio_bounds(self):
        """ANCMSG 1514: Error when nu < 0 or nu >= 0.5."""
        # Negative nu
        log1 = MessageLog()
        m1 = MatLaw74(id=1, rho=7.8e-9, e=210000.0, nu=-0.05, s11y=400.0, s22y=420.0, s33y=450.0, s12y=230.0, s23y=240.0, s31y=250.0)
        check_mat_law74(mat=m1, log=log1)
        assert log1.has_errors
        assert any("ANCMSG 1514" in str(err) and "nu must satisfy 0 <= nu < 0.5" in str(err) for err in log1.errors)

        # nu = 0.5 (incompressible limit)
        log2 = MessageLog()
        m2 = MatLaw74(id=2, rho=7.8e-9, e=210000.0, nu=0.5, s11y=400.0, s22y=420.0, s33y=450.0, s12y=230.0, s23y=240.0, s31y=250.0)
        check_mat_law74(mat=m2, log=log2)
        assert log2.has_errors
        assert any("ANCMSG 1514" in str(err) and "nu must satisfy 0 <= nu < 0.5" in str(err) for err in log2.errors)

        # nu > 0.5
        log3 = MessageLog()
        m3 = MatLaw74(id=3, rho=7.8e-9, e=210000.0, nu=0.55, s11y=400.0, s22y=420.0, s33y=450.0, s12y=230.0, s23y=240.0, s31y=250.0)
        check_mat_law74(mat=m3, log=log3)
        assert log3.has_errors
        assert any("ANCMSG 1514" in str(err) and "nu must satisfy 0 <= nu < 0.5" in str(err) for err in log3.errors)

        # Valid bounds: 0.0 and 0.49
        log4 = MessageLog()
        m4 = MatLaw74(id=4, rho=7.8e-9, e=210000.0, nu=0.0, s11y=400.0, s22y=420.0, s33y=450.0, s12y=230.0, s23y=240.0, s31y=250.0)
        check_mat_law74(mat=m4, log=log4)
        assert not log4.has_errors

        log5 = MessageLog()
        m5 = MatLaw74(id=5, rho=7.8e-9, e=210000.0, nu=0.49, s11y=400.0, s22y=420.0, s33y=450.0, s12y=230.0, s23y=240.0, s31y=250.0)
        check_mat_law74(mat=m5, log=log5)
        assert not log5.has_errors

    def test_ancmsg_822_yield_stress_bounds(self):
        """ANCMSG 822: Error when S11y, S22y, S33y, S12y, S23y, or S31y <= 0."""
        yield_names = ("s11y", "s22y", "s33y", "s12y", "s23y", "s31y")
        for s_name in yield_names:
            base_kwargs = {
                "rho": 7.8e-9,
                "e": 210000.0,
                "nu": 0.3,
                "s11y": 400.0,
                "s22y": 420.0,
                "s33y": 450.0,
                "s12y": 230.0,
                "s23y": 240.0,
                "s31y": 250.0,
            }

            # Zero value
            zero_kwargs = dict(base_kwargs)
            zero_kwargs[s_name] = 0.0
            log_zero = MessageLog()
            m_zero = MatLaw74(id=10, **zero_kwargs)
            check_mat_law74(mat=m_zero, log=log_zero)
            assert log_zero.has_errors
            assert any("ANCMSG 822" in str(err) and f"parameter {s_name.upper()} must be > 0" in str(err) for err in log_zero.errors)

            # Negative value
            neg_kwargs = dict(base_kwargs)
            neg_kwargs[s_name] = -100.0
            log_neg = MessageLog()
            m_neg = MatLaw74(id=11, **neg_kwargs)
            check_mat_law74(mat=m_neg, log=log_neg)
            assert log_neg.has_errors
            assert any("ANCMSG 822" in str(err) and f"parameter {s_name.upper()} must be > 0" in str(err) for err in log_neg.errors)

    def test_ancmsg_1044_failure_strains_bounds(self):
        """ANCMSG 1044: Error when epsr1 >= epsr2."""
        base_kwargs = {
            "rho": 7.8e-9,
            "e": 210000.0,
            "nu": 0.3,
            "s11y": 400.0,
            "s22y": 420.0,
            "s33y": 450.0,
            "s12y": 230.0,
            "s23y": 240.0,
            "s31y": 250.0,
        }

        # Equal: epsr1 == epsr2
        log1 = MessageLog()
        m1 = MatLaw74(id=1, epsr1=0.35, epsr2=0.35, **base_kwargs)
        check_mat_law74(mat=m1, log=log1)
        assert log1.has_errors
        assert any("ANCMSG 1044" in str(err) and "must be less than" in str(err) for err in log1.errors)

        # Inverted: epsr1 > epsr2
        log2 = MessageLog()
        m2 = MatLaw74(id=2, epsr1=0.45, epsr2=0.20, **base_kwargs)
        check_mat_law74(mat=m2, log=log2)
        assert log2.has_errors
        assert any("ANCMSG 1044" in str(err) and "must be less than" in str(err) for err in log2.errors)

        # Valid order: epsr1 < epsr2
        log3 = MessageLog()
        m3 = MatLaw74(id=3, epsr1=0.20, epsr2=0.45, **base_kwargs)
        check_mat_law74(mat=m3, log=log3)
        assert not log3.has_errors

    def test_ancmsg_305_reject_2d_analysis_and_shells(self):
        """ANCMSG 305: Error when LAW74 is assigned to 2D shells/quads/tria3 or 2D analysis model."""
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
                self.parts: dict = {}
            def element_groups(self):
                return self._grps

        m = MatLaw74(id=1, rho=7.8e-9, e=210000.0, nu=0.3, s11y=400.0, s22y=420.0, s33y=450.0, s12y=230.0, s23y=240.0, s31y=250.0)

        # 1. Reject 2D analysis model (N2D > 0)
        model_2d = DummyModel([], n2d=1)
        log_2d = MessageLog()
        check_mat_law74(model=model_2d, mat=m, log=log_2d)
        assert log_2d.has_errors
        assert any("ANCMSG 305" in str(err) and "not supported for 2D analysis" in str(err) for err in log_2d.errors)

        # 2. Reject 2D shell element groups
        shell_groups = ("shells", "shells_qbat", "shells_qeph", "sh3n", "quads", "quad4_2d", "tria3_2d", "elements_2d", "plane_strain", "plane_stress")
        for sh_grp_name in shell_groups:
            model_sh = DummyModel([(sh_grp_name, DummyGrp([DummyEl(1)]))])
            log_sh = MessageLog()
            check_mat_law74(model=model_sh, mat=m, log=log_sh)
            assert log_sh.has_errors, f"Shell group {sh_grp_name} should be rejected"
            assert any("ANCMSG 305" in str(err) for err in log_sh.errors)

        # 3. Reject shell via part elem_type
        model_part = Model()
        model_part.mat_law74s[1] = m
        p_shell = Part(id=1, prop_id=1, mat_id=1, title="Shell Part")
        p_shell.elem_type = "SHELL"
        model_part.parts[1] = p_shell
        log_part = MessageLog()
        check_mat_law74(model=model_part, mat=m, log=log_part)
        assert log_part.has_errors
        assert any("ANCMSG 305" in str(err) for err in log_part.errors)

    def test_ancmsg_306_reject_1d_elements(self):
        """ANCMSG 306: Error when LAW74 is assigned to 1D beams, trusses, or springs."""
        class DummyEl:
            def __init__(self, mid: int):
                self.mat_id = mid

        class DummyGrp:
            def __init__(self, els: list):
                self._els = {i: e for i, e in enumerate(els)}
            def values(self):
                return self._els.values()

        class DummyModel:
            def __init__(self, grps: list):
                self._grps = grps
                self.n2d = 0
                self.materials: dict = {}
                self.parts: dict = {}
            def element_groups(self):
                return self._grps

        m = MatLaw74(id=1, rho=7.8e-9, e=210000.0, nu=0.3, s11y=400.0, s22y=420.0, s33y=450.0, s12y=230.0, s23y=240.0, s31y=250.0)

        for d1_name in ("trusses", "beams", "springs"):
            model_1d = DummyModel([(d1_name, DummyGrp([DummyEl(1)]))])
            log_1d = MessageLog()
            check_mat_law74(model=model_1d, mat=m, log=log_1d)
            assert log_1d.has_errors, f"1D group {d1_name} should be rejected"
            assert any("ANCMSG 306" in str(err) for err in log_1d.errors)

        # Reject via part elem_type
        for d1_type in ("BEAM", "TRUSS", "SPRING"):
            model_part = Model()
            model_part.mat_law74s[1] = m
            p_1d = Part(id=1, prop_id=1, mat_id=1, title="1D Part")
            p_1d.elem_type = d1_type
            model_part.parts[1] = p_1d
            log_part = MessageLog()
            check_mat_law74(model=model_part, mat=m, log=log_part)
            assert log_part.has_errors
            assert any("ANCMSG 306" in str(err) for err in log_part.errors)

    def test_compatible_elements_accepted(self):
        """Verify bricks, tetras, penta6, pyra5 pass without errors."""
        class DummyEl:
            def __init__(self, mid: int):
                self.mat_id = mid

        class DummyGrp:
            def __init__(self, els: list):
                self._els = {i: e for i, e in enumerate(els)}
            def values(self):
                return self._els.values()

        class DummyModel:
            def __init__(self, grps: list):
                self._grps = grps
                self.n2d = 0
                self.materials: dict = {}
                self.parts: dict = {}
            def element_groups(self):
                return self._grps

        m = MatLaw74(id=1, rho=7.8e-9, e=210000.0, nu=0.3, s11y=400.0, s22y=420.0, s33y=450.0, s12y=230.0, s23y=240.0, s31y=250.0)

        for solid_fam in ("bricks", "tetras", "penta6", "pyra5", "solids", "solids_heph", "solids_tetra4"):
            model_solid = DummyModel([(solid_fam, DummyGrp([DummyEl(1)]))])
            log_solid = MessageLog()
            check_mat_law74(model=model_solid, mat=m, log=log_solid)
            assert not log_solid.has_errors, f"Solid group {solid_fam} should be accepted"

        # Via part elem_type="SOLID"
        model_part = Model()
        model_part.mat_law74s[1] = m
        p_solid = Part(id=1, prop_id=1, mat_id=1, title="Solid Part")
        p_solid.elem_type = "SOLID"
        model_part.parts[1] = p_solid
        log_p = MessageLog()
        check_mat_law74(model=model_part, mat=m, log=log_p)
        assert not log_p.has_errors

    def test_dispatch_via_check_materials(self):
        """Verify check_materials dispatches check_mat_law74 across model."""
        model = Model()
        m_bad = MatLaw74(id=1, rho=0.0, e=210000.0, nu=0.3, s11y=400.0, s22y=420.0, s33y=450.0, s12y=230.0, s23y=240.0, s31y=250.0)
        model.mat_law74s[1] = m_bad
        model.materials[1] = Material(id=1, law=74, rho0=0.0, params=m_bad.params)

        log = MessageLog()
        check_materials(model, log)
        assert log.has_errors
        assert any("ANCMSG 1514" in str(err) and "initial density RHO must be > 0" in str(err) for err in log.errors)


# ============================================================================
# 7. Boundary Values & Defaults
# ============================================================================

class TestLaw74BoundaryValuesAndDefaults:
    """Audit boundary conditions, default fallbacks, Hill calculations, and mapping protocol."""

    def test_isotropic_limit_hill_coefficients(self):
        """In the von Mises isotropic limit, Hill coefficients reduce to 1/(2*sig0^2) and 3/(2*sig0^2)."""
        sig0 = 400.0
        shear0 = sig0 / math.sqrt(3.0)
        m = MatLaw74(
            id=1,
            rho=7.8e-9,
            e=210000.0,
            nu=0.3,
            s11y=sig0,
            s22y=sig0,
            s33y=sig0,
            s12y=shear0,
            s23y=shear0,
            s31y=shear0,
        )

        expected_normal = 0.5 / (sig0 ** 2)
        expected_shear = 0.5 / (shear0 ** 2)

        assert m.FF == pytest.approx(expected_normal, rel=1e-6)
        assert m.GG == pytest.approx(expected_normal, rel=1e-6)
        assert m.HH == pytest.approx(expected_normal, rel=1e-6)
        assert m.LL == pytest.approx(expected_shear, rel=1e-6)
        assert m.MM == pytest.approx(expected_shear, rel=1e-6)
        assert m.NN == pytest.approx(expected_shear, rel=1e-6)

    def test_mapping_protocol_and_properties(self):
        """Verify __getitem__, __setitem__, get, keys, items, in protocol on MatLaw74."""
        m = MatLaw74(
            id=74,
            rho=7.8e-9,
            e=210000.0,
            nu=0.3,
            s11y=400.0,
            s22y=420.0,
            s33y=450.0,
            s12y=230.0,
            s23y=240.0,
            s31y=250.0,
            chard=0.25,
            table_id=101,
            fscale=1.1,
            pscale=0.9,
            t0=295.0,
            rhocp=3.5e-3,
        )

        # Dictionary access
        assert m["e"] == pytest.approx(210000.0)
        assert m["E"] == pytest.approx(210000.0)
        assert m["nu"] == pytest.approx(0.3)
        assert m["s11y"] == pytest.approx(400.0)
        assert m["sig11y"] == pytest.approx(400.0)
        assert m["G"] == pytest.approx(0.5 * 210000.0 / 1.3)
        assert m["table_id"] == 101
        assert m["tab_id"] == 101
        assert m["fun_a1"] == 101
        assert m.get("missing_key", -99.0) == -99.0

        # in operator
        assert "e" in m
        assert "nu" in m
        assert "s11y" in m
        assert "sound_speed" in m
        assert "sound_speed_solid" in m

        # Setting items
        m["s11y"] = 415.0
        assert m.s11y == pytest.approx(415.0)
        assert m["s11y"] == pytest.approx(415.0)

        with pytest.raises(KeyError, match="Cannot set unknown attribute"):
            m["non_existent_attribute"] = 123.45

        # Keys and items
        k = m.keys()
        assert "rho" in k
        assert "e" in k
        assert "s11y" in k
        assert "FF" in k
        assert "NN" in k
        items_dict = dict(m.items())
        assert items_dict["s11y"] == pytest.approx(415.0)
