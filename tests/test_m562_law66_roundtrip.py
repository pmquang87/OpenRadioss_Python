"""
Milestone M562: /MAT/LAW66 (/MAT/PLAS_TAB_COSSER, /MAT/PLAS_COSSER, /MAT/FOAM_TAB)
Exhaustive Roundtrip, Serialization, Boundary Cases & Negative Validation Auditor Suite.

Fortran origins:
  - starter/source/materials/mat/mat066/hm_read_mat66.F (card reader, parameters, default fallbacks)
  - engine/source/materials/mat/mat066/sigeps66.F (solid constitutive update)
  - engine/source/materials/mat/mat066/sigeps66c.F (shell constitutive update)
  - engine/source/output/restart/wrrestp.F & rdresb.F
  - starter/source/restart/ddsplit/wrrest.F
  - hm_cfg_files/config/CFG/radioss2022/MAT/mat_law66.cfg
"""

from __future__ import annotations

import math
from pathlib import Path
import pickle
from typing import Any, Dict, List, Sequence, Tuple
import numpy as np
import pytest

from pyradioss.common.messages import MessageLog
from pyradioss.input.card_layouts import CARD_LAYOUTS
from pyradioss.input.deck_reader import read_deck
from pyradioss.input.deck_writer import StarterDeck
from pyradioss.input.starter_keywords import (
    parse_starter_deck,
    read_mat_law66,
    read_starter_deck,
)
from pyradioss.materials.law66_plas_tab import (
    Law66Params,
    build_law66,
    extra_shapes,
    solid_update,
    shell_update,
)
from pyradioss.model.entities import (
    MatLaw66,
    MatPlasTabCosser,
    MatPlasCosser,
    MaterialLaw66,
    Material,
    Part,
    Property,
)
from pyradioss.model.model import Model
from pyradioss.starter.checks import (
    _ALLOWED_LAWS,
    _MAT_CHECKS,
    check_mat_law66,
    check_materials,
    check_model,
)
from pyradioss.starter.restart import read_restart, write_restart
from pyradioss.starter.starter import run_starter


# ============================================================================
# Helpers
# ============================================================================

def _parse_deck_str(tmp_path: Path, text: str, name: str = "TEST_LAW66") -> Tuple[Model, MessageLog]:
    """Write deck text to file and parse into Model and MessageLog."""
    deck_path = tmp_path / f"{name}_0000.rad"
    deck_path.write_text(text.strip() + "\n", encoding="utf-8")
    blocks = read_deck(str(deck_path))
    model = Model()
    log = MessageLog()
    parse_starter_deck(blocks, model, log)
    return model, log


def _assert_law66_all_fields_exact(
    m: MatLaw66,
    mat: Material | None = None,
    *,
    rho: float,
    e: float,
    nu: float,
    ec: float = 0.0,
    pc: float = 0.0,
    pt: float = 0.0,
    rpct: float = 1.0,
    c_hard: float = 0.0,
    f_cut: float = 0.0,
    fsmooth: int = 0,
    israte: int = 1,
    fun_a1: int = 0,
    fun_a2: int = 0,
    fscale11: float = 1.0,
    fscale22: float = 1.0,
    eps_0: float = 1.0,
    c: float = 1.0,
    sigma_y0: float = 0.0,
    vp: int = 0,
    fun_b1: int = 0,
    fun_b2: int = 0,
    fscale33: float = 1.0,
    fscale12: float = 1.0,
    nfunc: int = 0,
    tfunc: int = 0,
    abg_ipt: Sequence[int] | None = None,
    k_a1: Sequence[float] | None = None,
    fp1: Sequence[float] | None = None,
    abg_ipdel: Sequence[int] | None = None,
    k_b1: Sequence[float] | None = None,
    fp2: Sequence[float] | None = None,
    rhor: float | None = None,
    title: str = "",
) -> None:
    """Assert exact (floating-point tolerance) equality for all LAW66 parameters and derived properties."""
    expected_rhor = rhor if rhor is not None else rho

    # 1. Primary Dataclass Fields & Aliases
    assert m.rho == pytest.approx(rho, rel=1e-6, abs=1e-12)
    assert m.rho0 == pytest.approx(rho, rel=1e-6, abs=1e-12)
    assert m.refer_rho == pytest.approx(expected_rhor, rel=1e-6, abs=1e-12)
    assert m.rhor == pytest.approx(expected_rhor, rel=1e-6, abs=1e-12)

    assert m.e == pytest.approx(e, rel=1e-6, abs=1e-12)
    assert m.E == pytest.approx(e, rel=1e-6, abs=1e-12)
    assert m.nu == pytest.approx(nu, rel=1e-6, abs=1e-12)
    assert m.Nu == pytest.approx(nu, rel=1e-6, abs=1e-12)
    assert m.nu0 == pytest.approx(nu, rel=1e-6, abs=1e-12)

    assert m.ec == pytest.approx(ec, rel=1e-6, abs=1e-12)
    assert m.EC == pytest.approx(ec, rel=1e-6, abs=1e-12)
    assert m.pc == pytest.approx(pc, rel=1e-6, abs=1e-12)
    assert m.PC == pytest.approx(pc, rel=1e-6, abs=1e-12)
    assert m.p_c == pytest.approx(pc, rel=1e-6, abs=1e-12)
    assert m.pt == pytest.approx(pt, rel=1e-6, abs=1e-12)
    assert m.PT == pytest.approx(pt, rel=1e-6, abs=1e-12)
    assert m.p_t == pytest.approx(pt, rel=1e-6, abs=1e-12)
    assert m.rpct == pytest.approx(rpct, rel=1e-6, abs=1e-12)
    assert m.RPCT == pytest.approx(rpct, rel=1e-6, abs=1e-12)

    assert m.chard == pytest.approx(c_hard, rel=1e-6, abs=1e-12)
    assert m.c_hard == pytest.approx(c_hard, rel=1e-6, abs=1e-12)
    assert m.fisokin == pytest.approx(c_hard, rel=1e-6, abs=1e-12)
    assert m.asrate == pytest.approx(f_cut, rel=1e-6, abs=1e-12)
    assert m.f_cut == pytest.approx(f_cut, rel=1e-6, abs=1e-12)
    assert m.fsmooth == fsmooth
    assert m.Fsmooth == fsmooth
    assert m.israte == israte
    assert m.ISRATE == israte
    assert m.iyld_rate == israte
    assert m.iyield_rate == israte

    # ISRATE <= 3
    assert m.fun_a1 == fun_a1
    assert m.funct_idc == fun_a1
    assert m.fun_a2 == fun_a2
    assert m.funct_idt == fun_a2
    assert m.fscale11 == pytest.approx(fscale11, rel=1e-6, abs=1e-12)
    assert m.fscalec == pytest.approx(fscale11, rel=1e-6, abs=1e-12)
    assert m.fscale22 == pytest.approx(fscale22, rel=1e-6, abs=1e-12)
    assert m.fscalet == pytest.approx(fscale22, rel=1e-6, abs=1e-12)

    # ISRATE <= 2
    assert m.epsp0 == pytest.approx(eps_0, rel=1e-6, abs=1e-12)
    assert m.eps_0 == pytest.approx(eps_0, rel=1e-6, abs=1e-12)
    assert m.epsilon_0 == pytest.approx(eps_0, rel=1e-6, abs=1e-12)
    assert m.cp == pytest.approx(c, rel=1e-6, abs=1e-12)
    assert m.c == pytest.approx(c, rel=1e-6, abs=1e-12)
    assert m.sigy == pytest.approx(sigma_y0, rel=1e-6, abs=1e-12)
    assert m.sigma_y0 == pytest.approx(sigma_y0, rel=1e-6, abs=1e-12)
    assert m.vp == vp
    assert m.VP == vp

    # ISRATE == 3
    assert m.fun_b1 == fun_b1
    assert m.fnyrt_idc == fun_b1
    assert m.fun_b2 == fun_b2
    assert m.fnyrt_idt == fun_b2
    assert m.fscale33 == pytest.approx(fscale33, rel=1e-6, abs=1e-12)
    assert m.yrate_fscalec == pytest.approx(fscale33, rel=1e-6, abs=1e-12)
    assert m.fscale12 == pytest.approx(fscale12, rel=1e-6, abs=1e-12)
    assert m.yrate_fscalet == pytest.approx(fscale12, rel=1e-6, abs=1e-12)

    # ISRATE == 4
    assert m.nfunc == nfunc
    assert m.tfunc == tfunc
    exp_c_ids = list(abg_ipt) if abg_ipt is not None else []
    exp_c_rts = list(k_a1) if k_a1 is not None else []
    exp_c_scs = list(fp1) if fp1 is not None else []
    exp_t_ids = list(abg_ipdel) if abg_ipdel is not None else []
    exp_t_rts = list(k_b1) if k_b1 is not None else []
    exp_t_scs = list(fp2) if fp2 is not None else []

    assert list(m.abg_ipt) == exp_c_ids
    assert list(m.func_c_list) == exp_c_ids
    assert list(m.abg_ipdel) == exp_t_ids
    assert list(m.func_t_list) == exp_t_ids

    for val, exp in zip(m.k_a1, exp_c_rts):
        assert val == pytest.approx(exp, rel=1e-6, abs=1e-12)
    for val, exp in zip(m.eps_c_list, exp_c_rts):
        assert val == pytest.approx(exp, rel=1e-6, abs=1e-12)

    for val, exp in zip(m.fp1, exp_c_scs):
        assert val == pytest.approx(exp, rel=1e-6, abs=1e-12)
    for val, exp in zip(m.fscale_c_list, exp_c_scs):
        assert val == pytest.approx(exp, rel=1e-6, abs=1e-12)

    for val, exp in zip(m.k_b1, exp_t_rts):
        assert val == pytest.approx(exp, rel=1e-6, abs=1e-12)
    for val, exp in zip(m.eps_t_list, exp_t_rts):
        assert val == pytest.approx(exp, rel=1e-6, abs=1e-12)

    for val, exp in zip(m.fp2, exp_t_scs):
        assert val == pytest.approx(exp, rel=1e-6, abs=1e-12)
    for val, exp in zip(m.fscale_t_list, exp_t_scs):
        assert val == pytest.approx(exp, rel=1e-6, abs=1e-12)

    if title:
        assert m.title == title

    # 2. Derived Elastic Moduli and Sound Speeds
    expected_g = e / (2.0 * (1.0 + nu))
    expected_bulk = e / (3.0 * (1.0 - 2.0 * nu))
    assert m.G == pytest.approx(expected_g, rel=1e-6, abs=1e-12)
    assert m.bulk == pytest.approx(expected_bulk, rel=1e-6, abs=1e-12)
    assert m.K == pytest.approx(expected_bulk, rel=1e-6, abs=1e-12)

    expected_c_sound = math.sqrt(e / rho)
    expected_c_solid = math.sqrt((expected_bulk + 4.0 / 3.0 * expected_g) / rho)
    expected_c_shell = math.sqrt(e / (rho * max(1.0 - nu**2, 1e-15)))

    assert m.sound_speed == pytest.approx(expected_c_sound, rel=1e-6, abs=1e-12)
    assert m.sound_speed_solid == pytest.approx(expected_c_solid, rel=1e-6, abs=1e-12)
    assert m.sound_speed_shell == pytest.approx(expected_c_shell, rel=1e-6, abs=1e-12)

    # 3. Validation of model.materials[mat_id]
    if mat is not None:
        assert mat.law == 66
        assert mat.rho0 == pytest.approx(rho, rel=1e-6, abs=1e-12)
        p = mat.params
        assert p["rho"] == pytest.approx(rho, rel=1e-6, abs=1e-12)
        assert p["rho0"] == pytest.approx(rho, rel=1e-6, abs=1e-12)
        assert p["rhor"] == pytest.approx(expected_rhor, rel=1e-6, abs=1e-12)
        assert p["refer_rho"] == pytest.approx(expected_rhor, rel=1e-6, abs=1e-12)
        assert p["e"] == pytest.approx(e, rel=1e-6, abs=1e-12)
        assert p["E"] == pytest.approx(e, rel=1e-6, abs=1e-12)
        assert p["nu"] == pytest.approx(nu, rel=1e-6, abs=1e-12)
        assert p["Nu"] == pytest.approx(nu, rel=1e-6, abs=1e-12)
        assert p["ec"] == pytest.approx(ec, rel=1e-6, abs=1e-12)
        assert p["EC"] == pytest.approx(ec, rel=1e-6, abs=1e-12)
        assert p["pc"] == pytest.approx(pc, rel=1e-6, abs=1e-12)
        assert p["P_c"] == pytest.approx(pc, rel=1e-6, abs=1e-12)
        assert p["pt"] == pytest.approx(pt, rel=1e-6, abs=1e-12)
        assert p["P_t"] == pytest.approx(pt, rel=1e-6, abs=1e-12)
        assert p["rpct"] == pytest.approx(rpct, rel=1e-6, abs=1e-12)
        assert p["RPCT"] == pytest.approx(rpct, rel=1e-6, abs=1e-12)
        assert p["c_hard"] == pytest.approx(c_hard, rel=1e-6, abs=1e-12)
        assert p["chard"] == pytest.approx(c_hard, rel=1e-6, abs=1e-12)
        assert p["fisokin"] == pytest.approx(c_hard, rel=1e-6, abs=1e-12)
        assert p["f_cut"] == pytest.approx(f_cut, rel=1e-6, abs=1e-12)
        assert p["asrate"] == pytest.approx(f_cut, rel=1e-6, abs=1e-12)
        assert p["fsmooth"] == fsmooth
        assert p["Fsmooth"] == fsmooth
        assert p["israte"] == israte
        assert p["ISRATE"] == israte
        if title:
            assert mat.title == title


# ============================================================================
# 1. Exact Card Generation & Fixed vs Free Formats
# ============================================================================

class TestLaw66CardGeneration:
    """Audit StarterDeck.mat_law66 card generation in fixed and free formats for all ISRATE modes."""

    def test_fixed_format_card_columns_match_cfg_layouts_israte1(self):
        """Verify 5 cards emitted for ISRATE 1 match exact CARD_LAYOUTS widths."""
        deck = StarterDeck("CARD_TEST_1")
        deck.mat_law66(
            mid=1,
            title="Law66-ISRATE1",
            rho=7.85e-9,
            rhor=7.90e-9,
            e=210000.0,
            nu=0.30,
            c_hard=0.25,
            f_cut=100.0,
            fsmooth=1,
            iyld_rate=1,
            pc=50.0,
            pt=25.0,
            ec=195000.0,
            rpct=0.85,
            fun_a1=101,
            fun_a2=102,
            fscale11=1.1,
            fscale22=1.2,
            eps_0=0.001,
            c=40.0,
            sigma_y0=350.0,
            vp=1,
            fixed_format=True,
        )
        lines = deck.lines
        header_idx = lines.index("/MAT/LAW66/1")
        assert lines[header_idx + 1] == "Law66-ISRATE1"

        card1 = lines[header_idx + 2]
        card2 = lines[header_idx + 3]
        card3 = lines[header_idx + 4]
        card4 = lines[header_idx + 5]
        card5 = lines[header_idx + 6]

        assert len(card1) == 40  # MAT_LAW66_1: (20, 20)
        assert len(card2) == 100  # MAT_LAW66_2: (20, 20, 20, 20, 10, 10)
        assert len(card3) == 80  # MAT_LAW66_3: (20, 20, 20, 20)
        assert len(card4) == 60  # MAT_LAW66_4: (10, 10, 20, 20)
        assert len(card5) == 70  # MAT_LAW66_5: (20, 20, 20, 10)

    def test_fixed_format_card_columns_match_cfg_layouts_israte3(self):
        """Verify 5 cards emitted for ISRATE 3 match exact CARD_LAYOUTS widths."""
        deck = StarterDeck("CARD_TEST_3")
        deck.mat_law66(
            mid=3,
            title="Law66-ISRATE3",
            rho=2.7e-9,
            e=70000.0,
            nu=0.33,
            c_hard=0.1,
            f_cut=50.0,
            fsmooth=0,
            iyld_rate=3,
            pc=10.0,
            pt=8.0,
            ec=68000.0,
            rpct=1.0,
            fun_a1=1,
            fun_a2=2,
            fscale11=1.0,
            fscale22=1.0,
            fun_b1=3,
            fun_b2=4,
            fscale33=1.05,
            fscale12=0.95,
            fixed_format=True,
        )
        lines = deck.lines
        header_idx = lines.index("/MAT/LAW66/3")
        card5 = lines[header_idx + 6]
        # MAT_LAW66_ISRATE_3: (10, 10, 20, 20) = 60 chars
        assert len(card5) == 60

    def test_fixed_format_card_columns_match_cfg_layouts_israte4(self):
        """Verify NFUNC + TFUNC curves emitted for ISRATE 4 match exact CARD_LAYOUTS widths."""
        deck = StarterDeck("CARD_TEST_4")
        deck.mat_law66(
            mid=4,
            title="Law66-ISRATE4",
            rho=1.2e-3,
            e=100.0,
            nu=0.25,
            c_hard=0.5,
            f_cut=0.0,
            fsmooth=0,
            iyld_rate=4,
            pc=5.0,
            pt=2.0,
            ec=95.0,
            rpct=0.75,
            nfunc=2,
            tfunc=2,
            func_c_list=[11, 12],
            eps_c_list=[0.1, 1.0],
            fscale_c_list=[1.0, 1.1],
            func_t_list=[21, 22],
            eps_t_list=[0.1, 1.0],
            fscale_t_list=[1.0, 1.15],
            fixed_format=True,
        )
        lines = deck.lines
        header_idx = lines.index("/MAT/LAW66/4")
        card4 = lines[header_idx + 5]  # NFUNC, TFUNC -> (10, 10) = 20 chars
        assert len(card4) == 20
        assert card4 == f"{2:>10d}{2:>10d}"

        # 2 compression curves + 2 tension curves = 4 curve cards
        curve_cards = lines[header_idx + 6 : header_idx + 10]
        assert len(curve_cards) == 4
        for cc in curve_cards:
            assert len(cc) == 60  # MAT_LAW66_CURVE: (10, 10, 20, 20)

    def test_free_format_emission(self):
        """Verify whitespace-delimited free format emission for all ISRATE modes."""
        deck = StarterDeck("CARD_FREE")
        deck.mat_law66(
            mid=66,
            title="Free Format Law 66",
            rho=7.8e-9,
            rhor=7.85e-9,
            e=210000.0,
            nu=0.3,
            c_hard=0.2,
            f_cut=200.0,
            fsmooth=1,
            iyld_rate=1,
            pc=30.0,
            pt=15.0,
            ec=205000.0,
            rpct=0.9,
            fun_a1=10,
            fun_a2=20,
            fscale11=1.0,
            fscale22=1.0,
            eps_0=0.002,
            c=25.0,
            sigma_y0=400.0,
            vp=0,
            fixed_format=False,
        )
        lines = deck.lines
        header_idx = lines.index("/MAT/LAW66/66")
        assert lines[header_idx + 1] == "Free Format Law 66"
        toks1 = lines[header_idx + 2].split()
        assert len(toks1) == 2
        assert float(toks1[0]) == pytest.approx(7.8e-9)
        assert float(toks1[1]) == pytest.approx(7.85e-9)

        toks2 = lines[header_idx + 3].split()
        assert len(toks2) == 6
        assert float(toks2[0]) == pytest.approx(210000.0)
        assert float(toks2[1]) == pytest.approx(0.3)
        assert int(toks2[4]) == 1
        assert int(toks2[5]) == 1

    def test_aliases_emission(self):
        """Verify mat_plas_tab_cosser and mat_plas_cosser emit appropriate headers."""
        d1 = StarterDeck("ALIAS_1")
        d1.mat_plas_tab_cosser(mid=10, title="PlasTabCosser", rho=1.0, e=100.0, nu=0.3)
        assert "/MAT/PLAS_TAB_COSSER/10" in d1.lines

        d2 = StarterDeck("ALIAS_2")
        d2.mat_plas_cosser(mid=20, title="PlasCosser", rho=1.0, e=100.0, nu=0.3)
        assert "/MAT/PLAS_COSSER/20" in d2.lines


# ============================================================================
# 2. Fixed-Format Full Deck Roundtrip
# ============================================================================

class TestLaw66FixedFormatRoundtrip:
    """Audit complete roundtrip for StarterDeck.mat_law66 in fixed format."""

    def test_roundtrip_israte_1(self, tmp_path: Path):
        """Roundtrip test for ISRATE 1 (Cowper-Symonds)."""
        deck = StarterDeck("RT_ISRATE1")
        deck.mat_law66(
            mid=661,
            title="Law66 ISRATE 1 Fixed",
            rho=7.8e-9,
            rhor=7.82e-9,
            e=210000.0,
            nu=0.28,
            c_hard=0.35,
            f_cut=150.0,
            fsmooth=1,
            iyld_rate=1,
            pc=45.0,
            pt=22.5,
            ec=198000.0,
            rpct=0.88,
            fun_a1=501,
            fun_a2=502,
            fscale11=1.05,
            fscale22=1.15,
            eps_0=0.005,
            c=35.0,
            sigma_y0=320.0,
            vp=1,
            fixed_format=True,
        )
        model, log = _parse_deck_str(tmp_path, deck.write(), "RT_ISRATE1")
        assert not log.errors

        assert 661 in model.mat_law66s
        assert 661 in model.materials
        m66 = model.mat_law66s[661]
        mat = model.materials[661]

        _assert_law66_all_fields_exact(
            m66, mat,
            rho=7.8e-9, rhor=7.82e-9, e=210000.0, nu=0.28,
            c_hard=0.35, f_cut=150.0, fsmooth=1, israte=1,
            pc=45.0, pt=22.5, ec=198000.0, rpct=0.88,
            fun_a1=501, fun_a2=502, fscale11=1.05, fscale22=1.15,
            eps_0=0.005, c=35.0, sigma_y0=320.0, vp=1,
            title="Law66 ISRATE 1 Fixed",
        )

    def test_roundtrip_israte_2(self, tmp_path: Path):
        """Roundtrip test for ISRATE 2 (Logarithmic)."""
        deck = StarterDeck("RT_ISRATE2")
        deck.mat_law66(
            mid=662,
            title="Law66 ISRATE 2 Fixed",
            rho=2.7e-9,
            e=70000.0,
            nu=0.33,
            c_hard=0.15,
            f_cut=0.0,
            fsmooth=0,
            iyld_rate=2,
            pc=15.0,
            pt=10.0,
            ec=69000.0,
            rpct=0.95,
            fun_a1=10,
            fun_a2=20,
            fscale11=1.0,
            fscale22=1.0,
            eps_0=1.0,
            c=0.08,
            sigma_y0=250.0,
            vp=0,
            fixed_format=True,
        )
        model, log = _parse_deck_str(tmp_path, deck.write(), "RT_ISRATE2")
        assert not log.errors

        m66 = model.mat_law66s[662]
        mat = model.materials[662]
        _assert_law66_all_fields_exact(
            m66, mat,
            rho=2.7e-9, e=70000.0, nu=0.33,
            c_hard=0.15, f_cut=0.0, fsmooth=0, israte=2,
            pc=15.0, pt=10.0, ec=69000.0, rpct=0.95,
            fun_a1=10, fun_a2=20, fscale11=1.0, fscale22=1.0,
            eps_0=1.0, c=0.08, sigma_y0=250.0, vp=0,
            title="Law66 ISRATE 2 Fixed",
        )

    def test_roundtrip_israte_3(self, tmp_path: Path):
        """Roundtrip test for ISRATE 3 (Two load curves for strain rate)."""
        deck = StarterDeck("RT_ISRATE3")
        deck.mat_law66(
            mid=663,
            title="Law66 ISRATE 3 Fixed",
            rho=1.2e-3,
            e=150.0,
            nu=0.22,
            c_hard=0.5,
            f_cut=80.0,
            fsmooth=1,
            iyld_rate=3,
            pc=8.0,
            pt=4.0,
            ec=140.0,
            rpct=0.90,
            fun_a1=101,
            fun_a2=102,
            fscale11=1.2,
            fscale22=1.1,
            fun_b1=201,
            fun_b2=202,
            fscale33=1.3,
            fscale12=1.25,
            fixed_format=True,
        )
        model, log = _parse_deck_str(tmp_path, deck.write(), "RT_ISRATE3")
        assert not log.errors

        m66 = model.mat_law66s[663]
        mat = model.materials[663]
        _assert_law66_all_fields_exact(
            m66, mat,
            rho=1.2e-3, e=150.0, nu=0.22,
            c_hard=0.5, f_cut=80.0, fsmooth=1, israte=3,
            pc=8.0, pt=4.0, ec=140.0, rpct=0.90,
            fun_a1=101, fun_a2=102, fscale11=1.2, fscale22=1.1,
            fun_b1=201, fun_b2=202, fscale33=1.3, fscale12=1.25,
            title="Law66 ISRATE 3 Fixed",
        )

    def test_roundtrip_israte_4(self, tmp_path: Path):
        """Roundtrip test for ISRATE 4 (Multi-curve families for compression and tension)."""
        deck = StarterDeck("RT_ISRATE4")
        deck.mat_law66(
            mid=664,
            title="Law66 ISRATE 4 Multi-Curve",
            rho=7.8e-9,
            rhor=7.8e-9,
            e=205000.0,
            nu=0.29,
            c_hard=0.4,
            f_cut=250.0,
            fsmooth=1,
            iyld_rate=4,
            pc=60.0,
            pt=30.0,
            ec=190000.0,
            rpct=0.80,
            nfunc=3,
            tfunc=2,
            func_c_list=[101, 102, 103],
            eps_c_list=[0.001, 0.1, 10.0],
            fscale_c_list=[1.0, 1.1, 1.25],
            func_t_list=[201, 202],
            eps_t_list=[0.001, 5.0],
            fscale_t_list=[1.0, 1.18],
            fixed_format=True,
        )
        model, log = _parse_deck_str(tmp_path, deck.write(), "RT_ISRATE4")
        assert not log.errors

        m66 = model.mat_law66s[664]
        mat = model.materials[664]
        _assert_law66_all_fields_exact(
            m66, mat,
            rho=7.8e-9, e=205000.0, nu=0.29,
            c_hard=0.4, f_cut=250.0, fsmooth=1, israte=4,
            pc=60.0, pt=30.0, ec=190000.0, rpct=0.80,
            nfunc=3, tfunc=2,
            abg_ipt=[101, 102, 103],
            k_a1=[0.001, 0.1, 10.0],
            fp1=[1.0, 1.1, 1.25],
            abg_ipdel=[201, 202],
            k_b1=[0.001, 5.0],
            fp2=[1.0, 1.18],
            title="Law66 ISRATE 4 Multi-Curve",
        )

    def test_object_based_roundtrip(self, tmp_path: Path):
        """Verify passing MatLaw66 directly into StarterDeck produces identical deck."""
        m_orig = MatLaw66(
            id=660,
            rho=1.5e-3,
            refer_rho=1.52e-3,
            e=500.0,
            nu=0.31,
            ec=480.0,
            pc=12.0,
            pt=6.0,
            rpct=0.92,
            chard=0.18,
            asrate=75.0,
            fsmooth=1,
            israte=4,
            nfunc=2,
            tfunc=1,
            abg_ipt=[11, 12],
            k_a1=[0.01, 1.0],
            fp1=[1.0, 1.15],
            abg_ipdel=[21],
            k_b1=[0.5],
            fp2=[1.08],
            title="Object Re-emission",
        )
        deck1 = StarterDeck("OBJ_RT1")
        deck1.mat_law66(m_orig, fixed_format=True)
        text1 = deck1.write()

        model1, log1 = _parse_deck_str(tmp_path, text1, "OBJ_RT1")
        assert not log1.errors
        m_parsed = model1.mat_law66s[660]

        # Re-emit from parsed object
        deck2 = StarterDeck("OBJ_RT2")
        deck2.mat_law66(m_parsed, fixed_format=True)
        text2 = deck2.write()

        model2, log2 = _parse_deck_str(tmp_path, text2, "OBJ_RT2")
        assert not log2.errors
        m_final = model2.mat_law66s[660]

        _assert_law66_all_fields_exact(
            m_final,
            rho=1.5e-3, rhor=1.52e-3, e=500.0, nu=0.31,
            ec=480.0, pc=12.0, pt=6.0, rpct=0.92,
            c_hard=0.18, f_cut=75.0, fsmooth=1, israte=4,
            nfunc=2, tfunc=1,
            abg_ipt=[11, 12], k_a1=[0.01, 1.0], fp1=[1.0, 1.15],
            abg_ipdel=[21], k_b1=[0.5], fp2=[1.08],
            title="Object Re-emission",
        )

    def test_read_starter_deck_full_pipeline(self, tmp_path: Path):
        """Verify read_starter_deck pipeline correctly parses LAW66 deck from disk."""
        deck = StarterDeck("READ_STARTER_TEST")
        deck.mat_law66(
            mid=665,
            title="Full Starter Pipeline Law66",
            rho=7.85e-9,
            rhor=7.85e-9,
            e=210000.0,
            nu=0.30,
            ec=200000.0,
            pc=50.0,
            pt=25.0,
            rpct=0.85,
            c_hard=0.25,
            f_cut=100.0,
            fsmooth=1,
            iyld_rate=1,
            fun_a1=101,
            fun_a2=102,
            fscale11=1.0,
            fscale22=1.0,
            eps_0=0.001,
            c=40.0,
            sigma_y0=350.0,
            vp=0,
            fixed_format=True,
        )
        rad_path = tmp_path / "READ_STARTER_TEST_0000.rad"
        deck.write(str(rad_path))

        model, log = read_starter_deck(str(rad_path))
        assert not log.errors
        assert 665 in model.mat_law66s
        assert 665 in model.materials
        _assert_law66_all_fields_exact(
            model.mat_law66s[665], model.materials[665],
            rho=7.85e-9, e=210000.0, nu=0.30,
            ec=200000.0, pc=50.0, pt=25.0, rpct=0.85,
            c_hard=0.25, f_cut=100.0, fsmooth=1, israte=1,
            fun_a1=101, fun_a2=102, fscale11=1.0, fscale22=1.0,
            eps_0=0.001, c=40.0, sigma_y0=350.0, vp=0,
            title="Full Starter Pipeline Law66",
        )


# ============================================================================
# 3. Free-Format Full Deck Roundtrip
# ============================================================================

class TestLaw66FreeFormatRoundtrip:
    """Audit roundtrip in free format across all ISRATE options."""

    def test_free_format_israte1(self, tmp_path: Path):
        deck = StarterDeck("FREE_1")
        deck.mat_law66(
            mid=101,
            title="Free ISRATE 1",
            rho=7.8e-9,
            rhor=7.8e-9,
            e=210000.0,
            nu=0.3,
            c_hard=0.2,
            f_cut=100.0,
            fsmooth=0,
            iyld_rate=1,
            pc=20.0,
            pt=10.0,
            ec=200000.0,
            rpct=0.85,
            fun_a1=1,
            fun_a2=2,
            fscale11=1.0,
            fscale22=1.0,
            eps_0=1.0,
            c=1.0,
            sigma_y0=300.0,
            vp=0,
            fixed_format=False,
        )
        model, log = _parse_deck_str(tmp_path, deck.write(), "FREE_1")
        assert not log.errors
        _assert_law66_all_fields_exact(
            model.mat_law66s[101], model.materials[101],
            rho=7.8e-9, e=210000.0, nu=0.3,
            c_hard=0.2, f_cut=100.0, fsmooth=0, israte=1,
            pc=20.0, pt=10.0, ec=200000.0, rpct=0.85,
            fun_a1=1, fun_a2=2, fscale11=1.0, fscale22=1.0,
            eps_0=1.0, c=1.0, sigma_y0=300.0, vp=0,
            title="Free ISRATE 1",
        )

    def test_free_format_israte3(self, tmp_path: Path):
        deck = StarterDeck("FREE_3")
        deck.mat_law66(
            mid=103,
            title="Free ISRATE 3",
            rho=2.7e-9,
            e=70000.0,
            nu=0.33,
            c_hard=0.1,
            f_cut=50.0,
            fsmooth=1,
            iyld_rate=3,
            pc=12.0,
            pt=6.0,
            ec=68000.0,
            rpct=0.9,
            fun_a1=11,
            fun_a2=12,
            fscale11=1.1,
            fscale22=1.2,
            fun_b1=21,
            fun_b2=22,
            fscale33=1.3,
            fscale12=1.4,
            fixed_format=False,
        )
        model, log = _parse_deck_str(tmp_path, deck.write(), "FREE_3")
        assert not log.errors
        _assert_law66_all_fields_exact(
            model.mat_law66s[103], model.materials[103],
            rho=2.7e-9, e=70000.0, nu=0.33,
            c_hard=0.1, f_cut=50.0, fsmooth=1, israte=3,
            pc=12.0, pt=6.0, ec=68000.0, rpct=0.9,
            fun_a1=11, fun_a2=12, fscale11=1.1, fscale22=1.2,
            fun_b1=21, fun_b2=22, fscale33=1.3, fscale12=1.4,
            title="Free ISRATE 3",
        )

    def test_free_format_israte4(self, tmp_path: Path):
        deck = StarterDeck("FREE_4")
        deck.mat_law66(
            mid=104,
            title="Free ISRATE 4",
            rho=1.0e-3,
            e=200.0,
            nu=0.25,
            c_hard=0.3,
            f_cut=0.0,
            fsmooth=0,
            iyld_rate=4,
            pc=10.0,
            pt=5.0,
            ec=190.0,
            rpct=1.0,
            nfunc=2,
            tfunc=2,
            func_c_list=[1, 2],
            eps_c_list=[0.1, 10.0],
            fscale_c_list=[1.0, 1.2],
            func_t_list=[3, 4],
            eps_t_list=[0.2, 20.0],
            fscale_t_list=[1.0, 1.3],
            fixed_format=False,
        )
        model, log = _parse_deck_str(tmp_path, deck.write(), "FREE_4")
        assert not log.errors
        _assert_law66_all_fields_exact(
            model.mat_law66s[104], model.materials[104],
            rho=1.0e-3, e=200.0, nu=0.25,
            c_hard=0.3, f_cut=0.0, fsmooth=0, israte=4,
            pc=10.0, pt=5.0, ec=190.0, rpct=1.0,
            nfunc=2, tfunc=2,
            abg_ipt=[1, 2], k_a1=[0.1, 10.0], fp1=[1.0, 1.2],
            abg_ipdel=[3, 4], k_b1=[0.2, 20.0], fp2=[1.0, 1.3],
            title="Free ISRATE 4",
        )


# ============================================================================
# 4. Keyword Synonyms & Model Aliases
# ============================================================================

class TestLaw66KeywordSynonyms:
    """Verify /MAT/LAW66, /MAT/PLAS_TAB_COSSER, /MAT/PLAS_COSSER, /MAT/FOAM_TAB synonyms."""

    def test_synonyms_dispatch(self, tmp_path: Path):
        """Verify that all synonym keywords parse into MatLaw66 and Material(law=66)."""
        synonyms = [
            ("LAW66", 66),
            ("PLAS_TAB_COSSER", 67),
            ("PLAS_COSSER", 68),
            ("FOAM_TAB", 69),
        ]
        deck = StarterDeck("SYNONYMS")
        for kw, mid in synonyms:
            deck.mat_law66(
                mid=mid,
                law_name=kw,
                title=f"Synonym {kw}",
                rho=1.0e-3,
                e=100.0,
                nu=0.25,
                fixed_format=True,
            )
        model, log = _parse_deck_str(tmp_path, deck.write(), "SYNONYMS")
        assert not log.errors

        for kw, mid in synonyms:
            assert mid in model.mat_law66s
            assert mid in model.materials
            assert model.mat_law66s[mid].law == 66
            assert model.materials[mid].law == 66

    def test_model_dict_aliases(self):
        """Verify model.mat_plas_tab_cossers and model.mat_plas_cossers reference mat_law66s."""
        m = Model()
        mat = MatLaw66(id=1, rho=1.0, e=100.0, nu=0.3)
        m.mat_law66s[1] = mat
        assert m.mat_plas_tab_cossers[1] is mat
        assert m.mat_plas_cossers[1] is mat

    def test_class_aliases(self):
        """Verify MatPlasTabCosser, MatPlasCosser, MaterialLaw66 are aliases to MatLaw66."""
        assert MatPlasTabCosser is MatLaw66
        assert MatPlasCosser is MatLaw66
        assert MaterialLaw66 is MatLaw66


# ============================================================================
# 5. Pickle and Restart (.rst) Serialization
# ============================================================================

class TestLaw66SerializationAndRestart:
    """Audit MatLaw66 dataclass pickling, Model pickling, and write_restart/read_restart with uvar66."""

    def test_pickle_mat_law66(self):
        """Verify MatLaw66 dataclass pickling fidelity across protocols."""
        m = MatLaw66(
            id=66,
            title="Pickle-Law66",
            rho=7.85e-9,
            refer_rho=7.86e-9,
            e=210000.0,
            nu=0.30,
            ec=200000.0,
            pc=50.0,
            pt=25.0,
            rpct=0.85,
            chard=0.25,
            asrate=120.0,
            fsmooth=1,
            israte=4,
            nfunc=2,
            tfunc=1,
            abg_ipt=[1, 2],
            k_a1=[0.1, 1.0],
            fp1=[1.0, 1.1],
            abg_ipdel=[3],
            k_b1=[0.5],
            fp2=[1.05],
        )
        for protocol in (4, 5, pickle.HIGHEST_PROTOCOL):
            data = pickle.dumps(m, protocol=protocol)
            restored = pickle.loads(data)
            assert isinstance(restored, MatLaw66)
            _assert_law66_all_fields_exact(
                restored,
                rho=7.85e-9, rhor=7.86e-9, e=210000.0, nu=0.30,
                ec=200000.0, pc=50.0, pt=25.0, rpct=0.85,
                c_hard=0.25, f_cut=120.0, fsmooth=1, israte=4,
                nfunc=2, tfunc=1,
                abg_ipt=[1, 2], k_a1=[0.1, 1.0], fp1=[1.0, 1.1],
                abg_ipdel=[3], k_b1=[0.5], fp2=[1.05],
                title="Pickle-Law66",
            )

    def test_mapping_protocol_and_properties(self):
        """Verify __getitem__, get, keys, items, in protocol on MatLaw66."""
        m = MatLaw66(
            id=66,
            rho=7.8e-9,
            e=210000.0,
            nu=0.3,
            ec=195000.0,
            pc=40.0,
            pt=20.0,
            rpct=0.9,
            chard=0.3,
            asrate=100.0,
            fsmooth=1,
            israte=1,
            fun_a1=10,
            fun_a2=20,
            fscale11=1.0,
            fscale22=1.1,
            epsp0=1.0,
            cp=1.0,
            sigy=300.0,
            vp=0,
        )
        # Access via dict interface
        assert m["e"] == pytest.approx(210000.0)
        assert m["E"] == pytest.approx(210000.0)
        assert m["nu"] == pytest.approx(0.3)
        assert m["Nu"] == pytest.approx(0.3)
        assert m["pc"] == pytest.approx(40.0)
        assert m["pt"] == pytest.approx(20.0)
        assert m["PC"] == pytest.approx(40.0)
        assert m["PT"] == pytest.approx(20.0)
        assert m["ec"] == pytest.approx(195000.0)
        assert m["EC"] == pytest.approx(195000.0)
        assert m["rpct"] == pytest.approx(0.9)
        assert m["RPCT"] == pytest.approx(0.9)
        assert m.get("missing", 999.0) == 999.0
        assert "e" in m
        assert "E" in m
        assert "sound_speed" in m
        keys = m.keys()
        assert "rho" in keys
        assert "E" in keys
        assert "G" in keys
        assert "fscale_t_list" in keys

    def test_write_read_restart_with_law66_model(self, tmp_path: Path):
        """Verify write_restart and read_restart preserve LAW66 persistent material state arrays (uvar66)."""
        deck = StarterDeck("RST_LAW66")
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
        deck.part(1, "SolidPart", prop_id=1, mat_id=66)
        deck.mat_law66(
            mat_id=66,
            title="Restart-Law66",
            rho=7.8e-9,
            e=210000.0,
            nu=0.3,
            ec=200000.0,
            pc=30.0,
            pt=15.0,
            rpct=0.9,
            c_hard=0.2,
            f_cut=100.0,
            fsmooth=1,
            iyld_rate=1,
            fun_a1=10,
            fun_a2=20,
            fscale11=1.0,
            fscale22=1.0,
            eps_0=1.0,
            c=1.0,
            sigma_y0=350.0,
            vp=0,
        )

        rad_path = tmp_path / "rst_law66_0000.rad"
        deck.write(str(rad_path))

        model = run_starter(str(rad_path))
        assert model is not None

        groups = dict(model.element_groups())
        bg = groups.get("bricks") or groups.get("solids")
        st = bg.state

        # Synthesize realistic persistent material state arrays for LAW66 (2 elements)
        # uvar66 for solids: (n_elem, 8) -> [0: epsp, 1..6: alpha (6 components), 7: filtered_rate]
        uvar66_synth = np.array([
            [0.012, 10.0, -5.0, -5.0, 2.0, 0.0, 1.0, 50.0],
            [0.025, 15.0, -8.0, -7.0, 3.0, 1.0, 0.0, 75.0],
        ], dtype=float)

        st["mat_extra"]["uvar66"] = uvar66_synth.copy()
        st["uvar66"] = uvar66_synth.copy()

        # Write restart .rst
        rst_path = tmp_path / "rst_law66_0001.rst"
        engine_dict = {
            "cycle": 100,
            "t": 0.010,
            "dt": 1.0e-6,
        }
        write_restart(model, str(rst_path), engine=engine_dict)

        # Read back from restart .rst
        rest_model, rest_engine = read_restart(str(rst_path))
        assert rest_engine["cycle"] == 100
        assert rest_engine["t"] == pytest.approx(0.010)
        assert rest_engine["dt"] == pytest.approx(1.0e-6)

        # Assert exact preservation of LAW66 state arrays
        rest_bg = dict(rest_model.element_groups()).get("bricks") or dict(rest_model.element_groups()).get("solids")
        rest_extra = rest_bg.state["mat_extra"]
        rest_st = rest_bg.state

        np.testing.assert_array_equal(rest_extra["uvar66"], uvar66_synth)
        np.testing.assert_array_equal(rest_st["uvar66"], uvar66_synth)

    def test_write_read_restart_with_law66_shell_model(self, tmp_path: Path):
        """Verify write_restart and read_restart preserve LAW66 shell persistent arrays (uvar66, thk)."""
        deck = StarterDeck("RST_LAW66_SHELL")
        deck.node([
            (1, 0.0, 0.0, 0.0), (2, 1.0, 0.0, 0.0), (3, 1.0, 1.0, 0.0), (4, 0.0, 1.0, 0.0),
        ])
        deck.shell(1, [[1, 1, 2, 3, 4]])
        deck.prop_shell(1, "ShellProp", thick=1.2, nip=5)
        deck.part(1, "ShellPart", prop_id=1, mat_id=66)
        deck.mat_law66(
            mat_id=66,
            title="Restart-Law66-Shell",
            rho=7.8e-9,
            e=210000.0,
            nu=0.3,
            ec=200000.0,
            pc=30.0,
            pt=15.0,
            rpct=0.9,
            c_hard=0.2,
            f_cut=100.0,
            fsmooth=1,
            iyld_rate=1,
            fun_a1=10,
            fun_a2=20,
            fscale11=1.0,
            fscale22=1.0,
            eps_0=1.0,
            c=1.0,
            sigma_y0=350.0,
            vp=0,
        )

        rad_path = tmp_path / "rst_law66_shell_0000.rad"
        deck.write(str(rad_path))

        model = run_starter(str(rad_path))
        assert model is not None

        groups = dict(model.element_groups())
        sg = groups["shells"]
        st = sg.state

        # Synthesize shell persistent state arrays: uvar66 (n_elem, nip, 8), thk (n_elem, nip)
        n_elem = 1
        nip = 5
        uvar66_shell = np.zeros((n_elem, nip, 8), dtype=float)
        for ip in range(nip):
            uvar66_shell[0, ip, 0] = 0.005 * (ip + 1)
            uvar66_shell[0, ip, 1:4] = [10.0 + ip, -5.0 - ip, 2.0 + ip]
            uvar66_shell[0, ip, 4] = 45.0 + ip

        thk_shell = np.full((n_elem, nip), 1.18, dtype=float)

        st["mat_extra"]["uvar66"] = uvar66_shell.copy()
        st["mat_extra"]["thk"] = thk_shell.copy()

        # Write restart .rst
        rst_path = tmp_path / "rst_law66_shell_0001.rst"
        engine_dict = {
            "cycle": 250,
            "t": 0.025,
            "dt": 5.0e-7,
        }
        write_restart(model, str(rst_path), engine=engine_dict)

        # Read back from restart .rst
        rest_model, rest_engine = read_restart(str(rst_path))
        assert rest_engine["cycle"] == 250
        assert rest_engine["t"] == pytest.approx(0.025)
        assert rest_engine["dt"] == pytest.approx(5.0e-7)

        # Assert exact preservation of LAW66 shell arrays
        rest_sg = dict(rest_model.element_groups())["shells"]
        rest_extra = rest_sg.state["mat_extra"]

        np.testing.assert_array_equal(rest_extra["uvar66"], uvar66_shell)
        np.testing.assert_array_equal(rest_extra["thk"], thk_shell)


# ============================================================================
# 6. Negative Validation Diagnostics in check_mat_law66
# ============================================================================

class TestLaw66NegativeValidation:
    """Audit error messages and boundary checks in check_mat_law66."""

    def test_non_positive_density(self):
        """Reject rho <= 0 with diagnostic error."""
        log = MessageLog()
        mat_zero = MatLaw66(id=1, rho=0.0, e=210000.0, nu=0.3)
        check_mat_law66(mat=mat_zero, log=log)
        assert any("initial density RHO must be > 0" in e for e in log.errors)

        log_neg = MessageLog()
        mat_neg = MatLaw66(id=2, rho=-1.0e-3, e=210000.0, nu=0.3)
        check_mat_law66(mat=mat_neg, log=log_neg)
        assert any("initial density RHO must be > 0" in e for e in log_neg.errors)

    def test_non_positive_youngs_modulus(self):
        """Reject E <= 0 with diagnostic error."""
        log = MessageLog()
        mat_zero = MatLaw66(id=1, rho=7.8e-9, e=0.0, nu=0.3)
        check_mat_law66(mat=mat_zero, log=log)
        assert any("Young's modulus E must be > 0" in e for e in log.errors)

        log_neg = MessageLog()
        mat_neg = MatLaw66(id=2, rho=7.8e-9, e=-500.0, nu=0.3)
        check_mat_law66(mat=mat_neg, log=log_neg)
        assert any("Young's modulus E must be > 0" in e for e in log_neg.errors)

    def test_invalid_poissons_ratio(self):
        """Reject nu < 0 or nu >= 0.5 with ANCMSG 1514."""
        # nu < 0
        log_neg = MessageLog()
        mat_neg = MatLaw66(id=1, rho=7.8e-9, e=210000.0, nu=-0.05)
        check_mat_law66(mat=mat_neg, log=log_neg)
        assert any("Poisson's ratio nu must satisfy 0 <= nu < 0.5" in e and "ANCMSG 1514" in e for e in log_neg.errors)

        # nu == 0.5
        log_half = MessageLog()
        mat_half = MatLaw66(id=2, rho=7.8e-9, e=210000.0, nu=0.5)
        check_mat_law66(mat=mat_half, log=log_half)
        assert any("Poisson's ratio nu must satisfy 0 <= nu < 0.5" in e and "ANCMSG 1514" in e for e in log_half.errors)

        # nu > 0.5
        log_gt = MessageLog()
        mat_gt = MatLaw66(id=3, rho=7.8e-9, e=210000.0, nu=0.55)
        check_mat_law66(mat=mat_gt, log=log_gt)
        assert any("Poisson's ratio nu must satisfy 0 <= nu < 0.5" in e and "ANCMSG 1514" in e for e in log_gt.errors)

    def test_negative_limit_pressures(self):
        """Reject PC < 0 or PT < 0."""
        log_pc = MessageLog()
        mat_pc = MatLaw66(id=1, rho=7.8e-9, e=210000.0, nu=0.3, pc=-5.0, pt=10.0)
        check_mat_law66(mat=mat_pc, log=log_pc)
        assert any("P_c must be >= 0" in e for e in log_pc.errors)

        log_pt = MessageLog()
        mat_pt = MatLaw66(id=2, rho=7.8e-9, e=210000.0, nu=0.3, pc=10.0, pt=-2.0)
        check_mat_law66(mat=mat_pt, log=log_pt)
        assert any("P_t must be >= 0" in e for e in log_pt.errors)

    def test_incompatible_1d_elements_rejected(self):
        """Reject 1D elements (beams, trusses, springs) with ANCMSG 306."""
        model = Model()
        model.mat_law66s[66] = MatLaw66(id=66, rho=7.8e-9, e=210000.0, nu=0.3)
        p1 = Part(id=1, prop_id=1, mat_id=66, title="BeamPart")
        p1.elem_type = "BEAM"
        p2 = Part(id=2, prop_id=2, mat_id=66, title="TrussPart")
        p2.elem_type = "TRUSS"
        p3 = Part(id=3, prop_id=3, mat_id=66, title="SpringPart")
        p3.elem_type = "SPRING"
        model.parts[1] = p1
        model.parts[2] = p2
        model.parts[3] = p3

        log = MessageLog()
        check_mat_law66(mat=model.mat_law66s[66], model=model, log=log)
        assert len(log.errors) >= 3
        assert any("is not supported for 1D elements (beam) (ANCMSG 306)" in e for e in log.errors)
        assert any("is not supported for 1D elements (truss) (ANCMSG 306)" in e for e in log.errors)
        assert any("is not supported for 1D elements (spring) (ANCMSG 306)" in e for e in log.errors)

    def test_compatible_solids_and_shells_accepted(self):
        """Accept 3D solids (bricks, tetras) and 2D shells (quads, trias) without error."""
        model = Model()
        model.mat_law66s[66] = MatLaw66(id=66, rho=7.8e-9, e=210000.0, nu=0.3)
        p1 = Part(id=1, prop_id=1, mat_id=66, title="BrickPart")
        p1.elem_type = "SOLID"
        p2 = Part(id=2, prop_id=2, mat_id=66, title="TetraPart")
        p2.elem_type = "TETRA"
        p3 = Part(id=3, prop_id=3, mat_id=66, title="QuadPart")
        p3.elem_type = "SHELL"
        p4 = Part(id=4, prop_id=4, mat_id=66, title="TriaPart")
        p4.elem_type = "TRIA"
        model.parts[1] = p1
        model.parts[2] = p2
        model.parts[3] = p3
        model.parts[4] = p4

        log = MessageLog()
        check_mat_law66(mat=model.mat_law66s[66], model=model, log=log)
        assert not log.errors

    def test_check_materials_gate(self):
        """Verify check_materials and check_model run check_mat_law66 on model.materials."""
        model = Model()
        model.materials[66] = Material(
            id=66,
            law=66,
            rho0=-1.0,  # invalid
            params={"E": 210000.0, "nu": 0.3, "P_c": 0.0, "P_t": 0.0},
        )
        log = MessageLog()
        check_materials(model, log)
        assert any("initial density RHO must be > 0" in e for e in log.errors)


# ============================================================================
# 7. Boundary Values and Default Fallbacks
# ============================================================================

class TestLaw66BoundaryValuesAndDefaults:
    """Audit boundary values and default fallbacks in card reading and initialization."""

    def test_default_fallbacks_when_omitted(self, tmp_path: Path):
        """When rpct, eps_0, c are omitted, verify default to 1.0."""
        text = """#RADIOSS STARTER
/BEGIN
TEST_DEFAULTS
      2022         0
                  Mg                  mm                   s
                  Mg                  mm                   s
/MAT/LAW66/1
Defaults Omitted
             7.8e-09
            210000.0                 0.3                 0.0               0.0         0         1
                10.0                 5.0            200000.0
        10        20                 1.0                 1.0
/END
"""
        model, log = _parse_deck_str(tmp_path, text, "DEFAULTS_OMITTED")
        assert not log.errors

        m = model.mat_law66s[1]
        assert m.rpct == pytest.approx(1.0)
        assert m.epsp0 == pytest.approx(1.0)
        assert m.cp == pytest.approx(1.0)
        assert m.israte == 1
        assert m.refer_rho == pytest.approx(7.8e-9)

    def test_default_fallbacks_when_zero(self, tmp_path: Path):
        """When rpct=0, epsp0=0, cp=0 are explicitly set to zero, verify fallback to 1.0."""
        text = """#RADIOSS STARTER
/BEGIN
TEST_ZEROS
      2022         0
                  Mg                  mm                   s
                  Mg                  mm                   s
/MAT/LAW66/2
Defaults Zero
             7.8e-09
            210000.0                 0.3                 0.0               0.0         0         0
                10.0                 5.0            200000.0                 0.0
        10        20                 0.0                 0.0
                 0.0                 0.0               300.0         0
/END
"""
        model, log = _parse_deck_str(tmp_path, text, "DEFAULTS_ZERO")
        assert not log.errors

        m = model.mat_law66s[2]
        assert m.rpct == pytest.approx(1.0)
        assert m.epsp0 == pytest.approx(1.0)
        assert m.cp == pytest.approx(1.0)
        assert m.israte == 0  # In starter deck model, israte preserves 0 (engine Law66Params maps 0 to 1)

    def test_boundary_values(self):
        """Verify boundary values for nu (0.0, 0.4999), pc=0, pt=0, c_hard in [0, 1]."""
        log = MessageLog()
        # nu = 0.0 (valid lower bound)
        m1 = MatLaw66(id=1, rho=1.0, e=100.0, nu=0.0, pc=0.0, pt=0.0, chard=0.0)
        check_mat_law66(mat=m1, log=log)
        assert not log.errors

        # nu = 0.4999 (valid upper bound)
        m2 = MatLaw66(id=2, rho=1.0, e=100.0, nu=0.4999, pc=0.0, pt=0.0, chard=1.0)
        check_mat_law66(mat=m2, log=log)
        assert not log.errors
