"""
Milestone M564: /MAT/LAW87 (/MAT/BARLAT2000, /MAT/BARLAT_2000, /MAT/BARLAT2000_2D, /MAT/BARLAT_YLD2000)
Exhaustive Roundtrip, Serialization, Boundary Cases & Negative Validation Auditor Suite.

Fortran origins:
  - starter/source/materials/mat/mat087/hm_read_mat87.F90 (card reader, parameters, default fallbacks)
  - engine/source/materials/mat/mat087/sigeps87c.F90 (shell constitutive update kernel)
  - engine/source/materials/mat/mat087/mat87c_swift_voce.F90 (Swift-Voce combined hardening)
  - engine/source/materials/mat/mat087/mat87c_tabulated.F90 (tabulated multi-rate hardening)
  - engine/source/materials/mat/mat087/mat87c_hansel.F90 (Hansel transformation plasticity)
  - engine/source/materials/mat/mat087/mat87c_tabulated_3dir_ortho.F90 (3-direction ortho hardening)
  - engine/source/output/restart/wrrestp.F & rdresb.F
  - starter/source/restart/ddsplit/wrrest.F
  - hm_cfg_files/config/CFG/radioss140/MAT/matl87_barlat.cfg
  - hm_cfg_files/config/CFG/radioss2025/MAT/matl87_barlat.cfg
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
    read_mat_law87,
    read_starter_deck,
)
from pyradioss.materials.law87_barlat2000 import (
    Law87Params,
    build_law87,
    shell_update,
    extra_shapes,
    sound_speed as law87_sound_speed,
)
from pyradioss.model.entities import (
    MatBarlat2000,
    MatBarlat20002D,
    MatBarlatYld2000,
    MatLaw87,
    MatLaw87Curve,
    Material,
    MaterialLaw87,
    Part,
    Property,
)
from pyradioss.model.model import Model
from pyradioss.starter.checks import (
    _ALLOWED_LAWS,
    _MAT_CHECKS,
    check_mat_law87,
    check_materials,
    check_model,
)
from pyradioss.starter.restart import read_restart, write_restart
from pyradioss.starter.starter import run_starter


# ============================================================================
# Helpers
# ============================================================================

def _parse_deck_str(tmp_path: Path, text: str, name: str = "TEST_LAW87") -> Tuple[Model, MessageLog]:
    """Write deck text to file and parse into Model and MessageLog."""
    deck_path = tmp_path / f"{name}_0000.rad"
    deck_path.write_text(text.strip() + "\n", encoding="utf-8")
    blocks = read_deck(str(deck_path))
    model = Model()
    log = MessageLog()
    parse_starter_deck(blocks, model, log)
    return model, log


def _assert_law87_all_fields_exact(
    m: MatLaw87,
    mat: Material | None = None,
    *,
    rho: float,
    e: float,
    nu: float,
    rhor: float | None = None,
    iflag: int = 0,
    iflagsr: int = 0,
    invc: float = 0.0,
    invp: float = 0.0,
    ifit: int = 0,
    alphas: Sequence[float] | None = None,
    sigma_00: float = 0.0,
    sigma_45: float = 0.0,
    sigma_90: float = 0.0,
    sigma_b: float = 0.0,
    r_00: float = 1.0,
    r_45: float = 1.0,
    r_90: float = 1.0,
    r_b: float = 1.0,
    chard: float = 0.0,
    ikin: int = 1,
    exp_a: float = 6.0,
    alpha_vol: float = 1.0,
    n_hard: float = 0.0,
    fcut: float = 0.0,
    fsmooth: int = 0,
    nrate: int = 0,
    aswift: float = 0.0,
    eps0: float = 0.0,
    qvoce: float = 0.0,
    beta: float = 0.0,
    k0: float = 0.0,
    ckh: Sequence[float] = (0.0, 0.0, 0.0, 0.0),
    akh: Sequence[float] = (0.0, 0.0, 0.0, 0.0),
    num_curves: int = 0,
    title: str = "",
) -> None:
    """Assert exact (floating-point tolerance) equality for all LAW87 parameters and derived properties."""
    expected_rhor = rhor if (rhor is not None and rhor != 0.0) else rho
    expected_refer_rho = rhor if (rhor is not None and rhor != rho) else 0.0

    # 1. Primary Dataclass Fields
    assert m.rho == pytest.approx(rho, rel=1e-6, abs=1e-12)
    assert m.rho0 == pytest.approx(rho, rel=1e-6, abs=1e-12)
    assert m.rhor == pytest.approx(expected_rhor, rel=1e-6, abs=1e-12)
    assert m.refer_rho == pytest.approx(expected_refer_rho, rel=1e-6, abs=1e-12)
    assert m.e == pytest.approx(e, rel=1e-6, abs=1e-12)
    assert m.E == pytest.approx(e, rel=1e-6, abs=1e-12)
    assert m.nu == pytest.approx(nu, rel=1e-6, abs=1e-12)
    assert m.Nu == pytest.approx(nu, rel=1e-6, abs=1e-12)

    assert m.iflag == iflag
    assert m.iflagsr == iflagsr
    assert m.vp == iflagsr
    assert m.vflag == iflagsr
    assert m.invc == pytest.approx(invc, rel=1e-6, abs=1e-12)
    assert m.strain1 == pytest.approx(invc, rel=1e-6, abs=1e-12)
    assert m.c == pytest.approx(invc, rel=1e-6, abs=1e-12)
    assert m.invp == pytest.approx(invp, rel=1e-6, abs=1e-12)
    assert m.exp1 == pytest.approx(invp, rel=1e-6, abs=1e-12)
    assert m.p == pytest.approx(invp, rel=1e-6, abs=1e-12)
    assert m.flag_fit == ifit
    assert m.ifit == ifit

    if ifit == 1:
        assert m.sigma_00 == pytest.approx(sigma_00, rel=1e-6, abs=1e-12)
        assert m.sigma_45 == pytest.approx(sigma_45, rel=1e-6, abs=1e-12)
        assert m.sigma_90 == pytest.approx(sigma_90, rel=1e-6, abs=1e-12)
        assert m.sigma_b == pytest.approx(sigma_b, rel=1e-6, abs=1e-12)
        assert m.r_00 == pytest.approx(r_00, rel=1e-6, abs=1e-12)
        assert m.r_45 == pytest.approx(r_45, rel=1e-6, abs=1e-12)
        assert m.r_90 == pytest.approx(r_90, rel=1e-6, abs=1e-12)
        assert m.r_b == pytest.approx(r_b, rel=1e-6, abs=1e-12)
    else:
        exp_al = list(alphas) if alphas is not None else [1.0] * 8
        for i, val in enumerate(exp_al, 1):
            assert getattr(m, f"al{i}") == pytest.approx(val, rel=1e-6, abs=1e-12)
        assert m.alphas == pytest.approx(exp_al, rel=1e-6, abs=1e-12)

    assert m.fisokin == pytest.approx(chard, rel=1e-6, abs=1e-12)
    assert m.chard == pytest.approx(chard, rel=1e-6, abs=1e-12)
    assert m.ikin == ikin
    assert m.expa == pytest.approx(exp_a, rel=1e-6, abs=1e-12)
    assert m.exp_a == pytest.approx(exp_a, rel=1e-6, abs=1e-12)
    assert m.a_exp == int(round(exp_a))
    assert m.fcut == pytest.approx(fcut, rel=1e-6, abs=1e-12)
    assert m.fsmooth == fsmooth

    if iflag == 1:
        assert m.alpha == pytest.approx(alpha_vol, rel=1e-6, abs=1e-12)
        assert m.alpha_vol == pytest.approx(alpha_vol, rel=1e-6, abs=1e-12)
        assert m.nexp == pytest.approx(n_hard, rel=1e-6, abs=1e-12)
        assert m.n_hard == pytest.approx(n_hard, rel=1e-6, abs=1e-12)
        assert m.aswift == pytest.approx(aswift, rel=1e-6, abs=1e-12)
        assert m.a_swift == pytest.approx(aswift, rel=1e-6, abs=1e-12)
        assert m.epso == pytest.approx(eps0, rel=1e-6, abs=1e-12)
        assert m.eps0 == pytest.approx(eps0, rel=1e-6, abs=1e-12)
        assert m.qvoce == pytest.approx(qvoce, rel=1e-6, abs=1e-12)
        assert m.q_voce == pytest.approx(qvoce, rel=1e-6, abs=1e-12)
        assert m.beta == pytest.approx(beta, rel=1e-6, abs=1e-12)
        assert m.ko == pytest.approx(k0, rel=1e-6, abs=1e-12)
        assert m.k0 == pytest.approx(k0, rel=1e-6, abs=1e-12)
    elif iflag == 0:
        assert len(m.curves) == num_curves

    if ikin == 1 and chard > 0.0:
        for idx in range(4):
            assert m.ckh[idx] == pytest.approx(ckh[idx], rel=1e-6, abs=1e-12)
            assert m.akh[idx] == pytest.approx(akh[idx], rel=1e-6, abs=1e-12)

    if title:
        assert m.title == title

    # 2. Derived Moduli & Sound Speeds
    expected_g = 0.5 * e / (1.0 + nu)
    expected_bulk = e / (3.0 * (1.0 - 2.0 * nu))
    expected_c_shell = math.sqrt(e / (expected_rhor * (1.0 - nu ** 2)))

    assert m.G == pytest.approx(expected_g, rel=1e-6, abs=1e-12)
    assert m.bulk == pytest.approx(expected_bulk, rel=1e-6, abs=1e-12)
    assert m.K == pytest.approx(expected_bulk, rel=1e-6, abs=1e-12)
    assert float(m.sound_speed) == pytest.approx(expected_c_shell, rel=1e-6, abs=1e-12)
    assert float(m.sound_speed_shell) == pytest.approx(expected_c_shell, rel=1e-6, abs=1e-12)

    # 3. Transformation Matrices Lp and Lpp (for direct parameters ifit=0)
    if ifit == 0:
        exp_al = list(alphas) if alphas is not None else [1.0] * 8
        lp_exp = np.zeros((3, 3), dtype=float)
        lp_exp[0, 0] = 2.0 * exp_al[0] / 3.0
        lp_exp[0, 1] = -exp_al[0] / 3.0
        lp_exp[1, 0] = -exp_al[1] / 3.0
        lp_exp[1, 1] = 2.0 * exp_al[1] / 3.0
        lp_exp[2, 2] = exp_al[6]
        np.testing.assert_allclose(m.Lp, lp_exp, rtol=1e-6, atol=1e-12)

        lpp_exp = np.zeros((3, 3), dtype=float)
        lpp_exp[0, 0] = (-2.0 * exp_al[2] + 2.0 * exp_al[3] + 8.0 * exp_al[4] - 2.0 * exp_al[5]) / 9.0
        lpp_exp[0, 1] = (exp_al[2] - 4.0 * exp_al[3] - 4.0 * exp_al[4] + 4.0 * exp_al[5]) / 9.0
        lpp_exp[1, 0] = (4.0 * exp_al[2] - 4.0 * exp_al[3] - 4.0 * exp_al[4] + exp_al[5]) / 9.0
        lpp_exp[1, 1] = (-2.0 * exp_al[2] + 8.0 * exp_al[3] + 2.0 * exp_al[4] - 2.0 * exp_al[5]) / 9.0
        lpp_exp[2, 2] = exp_al[7]
        np.testing.assert_allclose(m.Lpp, lpp_exp, rtol=1e-6, atol=1e-12)

    # 4. Validate Material entity in model.materials
    if mat is not None:
        assert mat.law == 87
        assert mat.rho0 == pytest.approx(rho, rel=1e-6, abs=1e-12)
        p = mat.params
        assert p["rho0"] == pytest.approx(rho, rel=1e-6, abs=1e-12)
        assert p["refer_rho"] == pytest.approx(expected_refer_rho, rel=1e-6, abs=1e-12)
        assert p["e"] == pytest.approx(e, rel=1e-6, abs=1e-12)
        assert p["E"] == pytest.approx(e, rel=1e-6, abs=1e-12)
        assert p["nu"] == pytest.approx(nu, rel=1e-6, abs=1e-12)
        assert p["Nu"] == pytest.approx(nu, rel=1e-6, abs=1e-12)
        assert p["iflag"] == iflag
        assert p["vp"] == iflagsr
        assert p["ifit"] == ifit
        if ifit == 1:
            assert p["sigma_00"] == pytest.approx(sigma_00, rel=1e-6, abs=1e-12)
            assert p["sigma_45"] == pytest.approx(sigma_45, rel=1e-6, abs=1e-12)
            assert p["sigma_90"] == pytest.approx(sigma_90, rel=1e-6, abs=1e-12)
            assert p["sigma_b"] == pytest.approx(sigma_b, rel=1e-6, abs=1e-12)
            assert p["r_00"] == pytest.approx(r_00, rel=1e-6, abs=1e-12)
            assert p["r_45"] == pytest.approx(r_45, rel=1e-6, abs=1e-12)
            assert p["r_90"] == pytest.approx(r_90, rel=1e-6, abs=1e-12)
            assert p["r_b"] == pytest.approx(r_b, rel=1e-6, abs=1e-12)
        else:
            assert p["alpha"] == pytest.approx(alphas if alphas is not None else [1.0] * 8, rel=1e-6, abs=1e-12)
        assert p["chard"] == pytest.approx(chard, rel=1e-6, abs=1e-12)
        assert p["ikin"] == ikin
        assert p["exp_a"] == pytest.approx(exp_a, rel=1e-6, abs=1e-12)
        if iflag == 1:
            assert p["aswift"] == pytest.approx(aswift, rel=1e-6, abs=1e-12)
            assert p["eps0"] == pytest.approx(eps0, rel=1e-6, abs=1e-12)
            assert p["qvoce"] == pytest.approx(qvoce, rel=1e-6, abs=1e-12)
            assert p["beta"] == pytest.approx(beta, rel=1e-6, abs=1e-12)
            assert p["k0"] == pytest.approx(k0, rel=1e-6, abs=1e-12)
        if title:
            assert mat.title == title


# ============================================================================
# 1. Exact Card Generation & Fixed vs Free Formats
# ============================================================================

class TestLaw87CardGeneration:
    """Audit StarterDeck.mat_law87 exact fixed-format and free-format card layout emission."""

    def test_fixed_format_card_columns_match_cfg_layouts_direct_swift_voce(self):
        """Verify that fixed-format cards emitted for direct Swift-Voce match exact CARD_LAYOUTS widths."""
        deck = StarterDeck("CARD_TEST_DIRECT")
        deck.mat_law87(
            mid=87,
            title="Barlat Direct Sheet",
            rho=2.7e-9,
            rhor=2.75e-9,
            e=70000.0,
            nu=0.33,
            iflag=1,
            vp=1,
            c=120.0,
            p=4.5,
            ifit=0,
            alpha=[1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8],
            chard=0.35,
            ikin=1,
            exp_a=6.0,
            alpha_vol=0.8,
            n_hard=0.22,
            fcut=480.0,
            fsmooth=1,
            nrate=0,
            aswift=460.0,
            eps0=0.015,
            qvoce=135.0,
            beta=22.5,
            k0=215.0,
            ckh=(1200.0, 600.0, 250.0, 100.0),
            akh=(45.0, 22.0, 10.0, 5.0),
            fixed_format=True,
        )

        lines = deck.lines
        header_idx = lines.index("/MAT/LAW87/87")
        assert lines[header_idx + 1] == "Barlat Direct Sheet"

        # Card 1: RHO [RHOR] -> MAT_LAW87_1 [20, 20]
        c1 = lines[header_idx + 2]
        f1 = split_fixed(c1, CARD_LAYOUTS["MAT_LAW87_1"])
        assert float(f1[0]) == pytest.approx(2.7e-9)
        assert float(f1[1]) == pytest.approx(2.75e-9)

        # Card 2: E, Nu, Iflag, VP, c, P -> MAT_LAW87_2 [20, 20, 10, 10, 20, 20]
        c2 = lines[header_idx + 3]
        f2 = split_fixed(c2, CARD_LAYOUTS["MAT_LAW87_2"])
        assert float(f2[0]) == pytest.approx(70000.0)
        assert float(f2[1]) == pytest.approx(0.33)
        assert int(f2[2]) == 1
        assert int(f2[3]) == 1
        assert float(f2[4]) == pytest.approx(120.0)
        assert float(f2[5]) == pytest.approx(4.5)

        # Card 3: al1..al4 -> MAT_LAW87_3 [20, 20, 20, 20]
        c3 = lines[header_idx + 4]
        f3 = split_fixed(c3, CARD_LAYOUTS["MAT_LAW87_3"])
        for idx in range(4):
            assert float(f3[idx]) == pytest.approx(1.1 + 0.1 * idx)

        # Card 4: al5..al8 -> MAT_LAW87_4 [20, 20, 20, 20]
        c4 = lines[header_idx + 5]
        f4 = split_fixed(c4, CARD_LAYOUTS["MAT_LAW87_4"])
        for idx in range(4):
            assert float(f4[idx]) == pytest.approx(1.5 + 0.1 * idx)

        # Card 5: Chard, Ikin -> MAT_LAW87_5 [20, 10]
        c5 = lines[header_idx + 6]
        f5 = split_fixed(c5, CARD_LAYOUTS["MAT_LAW87_5"])
        assert float(f5[0]) == pytest.approx(0.35)
        assert int(f5[1]) == 1

        # Card 6: exp_a, alpha_vol, n_hard, fcut, fsmooth, nrate -> MAT_LAW87_6_1 [20, 20, 20, 20, 10, 10]
        c6 = lines[header_idx + 7]
        f6 = split_fixed(c6, CARD_LAYOUTS["MAT_LAW87_6_1"])
        assert float(f6[0]) == pytest.approx(6.0)
        assert float(f6[1]) == pytest.approx(0.8)
        assert float(f6[2]) == pytest.approx(0.22)
        assert float(f6[3]) == pytest.approx(480.0)
        assert int(f6[4]) == 1
        assert int(f6[5]) == 0

        # Card 7: aswift, eps0, qvoce, beta, k0 -> MAT_LAW87_7_1 [20, 20, 20, 20, 20]
        c7 = lines[header_idx + 8]
        f7 = split_fixed(c7, CARD_LAYOUTS["MAT_LAW87_7_1"])
        assert float(f7[0]) == pytest.approx(460.0)
        assert float(f7[1]) == pytest.approx(0.015)
        assert float(f7[2]) == pytest.approx(135.0)
        assert float(f7[3]) == pytest.approx(22.5)
        assert float(f7[4]) == pytest.approx(215.0)

        # Card 8 & 9: ckh and akh kinematic hardening cards (4 x 20)
        c8 = lines[header_idx + 9]
        f8 = split_fixed(c8, [20, 20, 20, 20])
        assert [float(x) for x in f8] == pytest.approx([1200.0, 600.0, 250.0, 100.0])

        c9 = lines[header_idx + 10]
        f9 = split_fixed(c9, [20, 20, 20, 20])
        assert [float(x) for x in f9] == pytest.approx([45.0, 22.0, 10.0, 5.0])

    def test_fixed_format_card_columns_match_cfg_layouts_fitting(self):
        """Verify that fixed-format cards for experimental fitting (ifit=1) match CARD_LAYOUTS."""
        deck = StarterDeck("CARD_TEST_FIT")
        deck.mat_law87(
            mid=87,
            title="Barlat Fitting Sheet",
            rho=2.7e-9,
            e=70000.0,
            nu=0.33,
            iflag=1,
            ifit=1,
            sigma_00=210.0,
            sigma_45=215.0,
            sigma_90=220.0,
            sigma_b=230.0,
            r_00=1.2,
            r_45=1.4,
            r_90=1.6,
            r_b=1.1,
            aswift=450.0,
            eps0=0.01,
            qvoce=120.0,
            beta=25.0,
            k0=210.0,
            fixed_format=True,
        )

        lines = deck.lines
        header_idx = lines.index("/MAT/LAW87/87")

        # Card 3 (Fitting): Sigma_00, Sigma_45, Sigma_90, Sigma_b, ifit -> MAT_LAW87_3_FIT [20, 20, 20, 20, 10]
        c3 = lines[header_idx + 4]
        f3 = split_fixed(c3, CARD_LAYOUTS["MAT_LAW87_3_FIT"])
        assert float(f3[0]) == pytest.approx(210.0)
        assert float(f3[1]) == pytest.approx(215.0)
        assert float(f3[2]) == pytest.approx(220.0)
        assert float(f3[3]) == pytest.approx(230.0)
        assert int(f3[4]) == 1

        # Card 4 (Fitting): r_00, r_45, r_90, r_b -> MAT_LAW87_4 [20, 20, 20, 20]
        c4 = lines[header_idx + 5]
        f4 = split_fixed(c4, CARD_LAYOUTS["MAT_LAW87_4"])
        assert float(f4[0]) == pytest.approx(1.2)
        assert float(f4[1]) == pytest.approx(1.4)
        assert float(f4[2]) == pytest.approx(1.6)
        assert float(f4[3]) == pytest.approx(1.1)

    def test_free_format_card_emission_space_and_comma(self):
        """Verify free-format card emission with space and comma delimiters."""
        deck_space = StarterDeck("SPACE_TEST")
        deck_space.mat_law87(
            mid=87,
            title="Space Free Deck",
            rho=2.7e-9,
            e=70000.0,
            nu=0.33,
            iflag=1,
            aswift=450.0,
            k0=200.0,
            fixed_format=False,
        )
        assert any("2.7e-09" in line for line in deck_space.lines)
        assert any("70000.0 0.33 1 0 0.0 0.0" in line for line in deck_space.lines)

        deck_comma = StarterDeck("COMMA_TEST")
        deck_comma.mat_law87(
            mid=87,
            title="Comma Free Deck",
            rho=2.7e-9,
            e=70000.0,
            nu=0.33,
            iflag=1,
            aswift=450.0,
            k0=200.0,
            fixed_format=False,
            comma=True,
        )
        assert any("70000.0, 0.33, 1, 0, 0.0, 0.0" in line for line in deck_comma.lines)


# ============================================================================
# 2. Fixed-Format Deck Roundtrip
# ============================================================================

class TestLaw87FixedFormatRoundtrip:
    """Audit fixed-format deck roundtrip: write to text, parse back with read_mat_law87."""

    def test_fixed_format_direct_swift_voce_roundtrip(self, tmp_path: Path):
        """Roundtrip direct parameters (ifit=0, alphas) with Swift-Voce hardening (iflag=1)."""
        deck = StarterDeck("RT_DIRECT_SWIFT")
        deck.mat_law87(
            mid=87,
            title="Aluminium 6016-T4 Sheet",
            rho=2.7e-9,
            rhor=2.72e-9,
            e=70500.0,
            nu=0.33,
            iflag=1,
            vp=1,
            c=80.0,
            p=3.5,
            ifit=0,
            alpha=[1.05, 1.15, 1.25, 1.35, 1.45, 1.55, 1.65, 1.75],
            chard=0.0,
            ikin=1,
            exp_a=8.0,
            alpha_vol=0.75,
            n_hard=0.24,
            fcut=520.0,
            fsmooth=1,
            aswift=480.0,
            eps0=0.012,
            qvoce=110.0,
            beta=18.5,
            k0=205.0,
            fixed_format=True,
        )

        model, log = _parse_deck_str(tmp_path, "\n".join(deck.lines), "RT_DIRECT_SWIFT")
        assert not log.has_errors
        assert 87 in model.mat_law87s
        assert 87 in model.materials

        _assert_law87_all_fields_exact(
            model.mat_law87s[87],
            model.materials[87],
            rho=2.7e-9,
            rhor=2.72e-9,
            e=70500.0,
            nu=0.33,
            iflag=1,
            iflagsr=1,
            invc=80.0,
            invp=3.5,
            ifit=0,
            alphas=[1.05, 1.15, 1.25, 1.35, 1.45, 1.55, 1.65, 1.75],
            chard=0.0,
            ikin=1,
            exp_a=8.0,
            alpha_vol=0.75,
            n_hard=0.24,
            fcut=520.0,
            fsmooth=1,
            aswift=480.0,
            eps0=0.012,
            qvoce=110.0,
            beta=18.5,
            k0=205.0,
            title="Aluminium 6016-T4 Sheet",
        )

    def test_fixed_format_fitting_swift_voce_roundtrip(self, tmp_path: Path):
        """Roundtrip experimental fitting parameters (ifit=1, yield stresses & Lankford)."""
        deck = StarterDeck("RT_FIT_SWIFT")
        deck.mat_law87(
            mid=87,
            title="Steel DP600 Fitting Sheet",
            rho=7.85e-9,
            e=210000.0,
            nu=0.30,
            iflag=1,
            vp=0,
            c=0.0,
            p=0.0,
            ifit=1,
            sigma_00=285.0,
            sigma_45=290.0,
            sigma_90=310.0,
            sigma_b=330.0,
            r_00=0.88,
            r_45=0.95,
            r_90=1.12,
            r_b=0.98,
            chard=0.0,
            ikin=1,
            exp_a=6.0,
            alpha_vol=0.6,
            n_hard=0.18,
            fcut=800.0,
            fsmooth=0,
            aswift=650.0,
            eps0=0.008,
            qvoce=180.0,
            beta=15.0,
            k0=280.0,
            fixed_format=True,
        )

        model, log = _parse_deck_str(tmp_path, "\n".join(deck.lines), "RT_FIT_SWIFT")
        assert not log.has_errors
        assert 87 in model.mat_law87s

        _assert_law87_all_fields_exact(
            model.mat_law87s[87],
            model.materials[87],
            rho=7.85e-9,
            e=210000.0,
            nu=0.30,
            iflag=1,
            iflagsr=0,
            ifit=1,
            sigma_00=285.0,
            sigma_45=290.0,
            sigma_90=310.0,
            sigma_b=330.0,
            r_00=0.88,
            r_45=0.95,
            r_90=1.12,
            r_b=0.98,
            chard=0.0,
            ikin=1,
            exp_a=6.0,
            alpha_vol=0.6,
            n_hard=0.18,
            fcut=800.0,
            fsmooth=0,
            aswift=650.0,
            eps0=0.008,
            qvoce=180.0,
            beta=15.0,
            k0=280.0,
            title="Steel DP600 Fitting Sheet",
        )

    def test_fixed_format_tabulated_multi_rate_hardening_roundtrip(self, tmp_path: Path):
        """Roundtrip tabulated multi-rate hardening (iflag=0) with multiple curves."""
        deck = StarterDeck("RT_TABULATED")
        deck.mat_law87(
            mid=87,
            title="Tabulated Sheet",
            rho=2.7e-9,
            e=70000.0,
            nu=0.33,
            iflag=0,
            vp=1,
            c=100.0,
            p=4.0,
            ifit=0,
            alpha=[1.0, 1.05, 1.10, 1.15, 1.20, 1.25, 1.30, 1.35],
            chard=0.0,
            ikin=1,
            exp_a=6.0,
            fcut=600.0,
            fsmooth=1,
            curves=[
                {"fct_id": 101, "fscale": 1.0, "epsp": 0.0},
                {"fct_id": 102, "fscale": 1.08, "epsp": 10.0},
                {"fct_id": 103, "fscale": 1.15, "epsp": 100.0},
                {"fct_id": 104, "fscale": 1.22, "epsp": 1000.0},
            ],
            fixed_format=True,
        )

        model, log = _parse_deck_str(tmp_path, "\n".join(deck.lines), "RT_TABULATED")
        assert not log.has_errors
        m = model.mat_law87s[87]

        _assert_law87_all_fields_exact(
            m,
            model.materials[87],
            rho=2.7e-9,
            e=70000.0,
            nu=0.33,
            iflag=0,
            iflagsr=1,
            invc=100.0,
            invp=4.0,
            ifit=0,
            alphas=[1.0, 1.05, 1.10, 1.15, 1.20, 1.25, 1.30, 1.35],
            chard=0.0,
            ikin=1,
            exp_a=6.0,
            fcut=600.0,
            fsmooth=1,
            num_curves=4,
            title="Tabulated Sheet",
        )

        assert len(m.curves) == 4
        expected_fids = [101, 102, 103, 104]
        expected_scales = [1.0, 1.08, 1.15, 1.22]
        expected_rates = [0.0, 10.0, 100.0, 1000.0]
        for i, c in enumerate(m.curves):
            assert c.fct_id == expected_fids[i]
            assert c.fscale == pytest.approx(expected_scales[i], rel=1e-6)
            assert c.epsp == pytest.approx(expected_rates[i], rel=1e-6)

    def test_fixed_format_chaboche_rousselier_kinematic_roundtrip(self, tmp_path: Path):
        """Roundtrip Chaboche-Rousselier kinematic hardening card emission and parsing."""
        deck = StarterDeck("RT_KINEMATIC")
        deck.mat_law87(
            mid=87,
            title="Kinematic Sheet",
            rho=2.7e-9,
            e=70000.0,
            nu=0.33,
            iflag=1,
            ifit=0,
            alpha=[1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8],
            chard=0.45,
            ikin=1,
            exp_a=6.0,
            alpha_vol=0.8,
            n_hard=0.2,
            fcut=500.0,
            fsmooth=1,
            aswift=450.0,
            eps0=0.01,
            qvoce=120.0,
            beta=25.0,
            k0=210.0,
            ckh=(1500.0, 750.0, 300.0, 120.0),
            akh=(60.0, 30.0, 15.0, 6.0),
            fixed_format=True,
        )

        model, log = _parse_deck_str(tmp_path, "\n".join(deck.lines), "RT_KINEMATIC")
        assert not log.has_errors
        m = model.mat_law87s[87]

        _assert_law87_all_fields_exact(
            m,
            model.materials[87],
            rho=2.7e-9,
            e=70000.0,
            nu=0.33,
            iflag=1,
            ifit=0,
            alphas=[1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8],
            chard=0.45,
            ikin=1,
            exp_a=6.0,
            alpha_vol=0.8,
            n_hard=0.2,
            fcut=500.0,
            fsmooth=1,
            aswift=450.0,
            eps0=0.01,
            qvoce=120.0,
            beta=25.0,
            k0=210.0,
            ckh=(1500.0, 750.0, 300.0, 120.0),
            akh=(60.0, 30.0, 15.0, 6.0),
            title="Kinematic Sheet",
        )

    def test_fixed_format_direct_passed_as_obj(self, tmp_path: Path):
        """Verify passing MatLaw87 object directly into StarterDeck.mat_law87."""
        m_in = MatLaw87(
            id=87,
            title="Object Input Sheet",
            rho=2.7e-9,
            refer_rho=2.74e-9,
            e=72000.0,
            nu=0.32,
            iflag=1,
            vp=1,
            invc=90.0,
            invp=3.8,
            flag_fit=0,
            al1=1.02,
            al2=1.04,
            al3=1.06,
            al4=1.08,
            al5=1.10,
            al6=1.12,
            al7=1.14,
            al8=1.16,
            chard=0.2,
            ikin=1,
            expa=6.0,
            fcut=490.0,
            fsmooth=1,
            aswift=470.0,
            epso=0.011,
            qvoce=115.0,
            beta=21.0,
            ko=208.0,
            ckh=(800.0, 400.0, 200.0, 100.0),
            akh=(40.0, 20.0, 10.0, 5.0),
        )

        deck = StarterDeck("OBJ_INPUT")
        deck.mat_law87(mid=m_in, fixed_format=True)

        model, log = _parse_deck_str(tmp_path, "\n".join(deck.lines), "OBJ_INPUT")
        assert not log.has_errors
        m_out = model.mat_law87s[87]

        _assert_law87_all_fields_exact(
            m_out,
            model.materials[87],
            rho=2.7e-9,
            rhor=2.74e-9,
            e=72000.0,
            nu=0.32,
            iflag=1,
            iflagsr=1,
            invc=90.0,
            invp=3.8,
            ifit=0,
            alphas=[1.02, 1.04, 1.06, 1.08, 1.10, 1.12, 1.14, 1.16],
            chard=0.2,
            ikin=1,
            exp_a=6.0,
            alpha_vol=0.0,
            fcut=490.0,
            fsmooth=1,
            aswift=470.0,
            eps0=0.011,
            qvoce=115.0,
            beta=21.0,
            k0=208.0,
            ckh=(800.0, 400.0, 200.0, 100.0),
            akh=(40.0, 20.0, 10.0, 5.0),
            title="Object Input Sheet",
        )


# ============================================================================
# 3. Free-Format Deck Roundtrip
# ============================================================================

class TestLaw87FreeFormatRoundtrip:
    """Audit free-format deck generation, parsing, and exact parameter recovery."""

    def test_free_format_direct_swift_voce_space_delimited(self, tmp_path: Path):
        """Roundtrip space-delimited free format direct Swift-Voce parameters."""
        deck = StarterDeck("FREE_SPACE")
        deck.mat_law87(
            mid=87,
            title="Free Space Sheet",
            rho=2.7e-9,
            rhor=2.73e-9,
            e=69000.0,
            nu=0.34,
            iflag=1,
            vp=1,
            c=110.0,
            p=4.2,
            ifit=0,
            alpha=[1.08, 1.12, 1.18, 1.22, 1.28, 1.32, 1.38, 1.42],
            chard=0.0,
            ikin=1,
            exp_a=6.0,
            alpha_vol=0.85,
            n_hard=0.26,
            fcut=510.0,
            fsmooth=0,
            aswift=440.0,
            eps0=0.014,
            qvoce=125.0,
            beta=24.0,
            k0=195.0,
            fixed_format=False,
        )

        model, log = _parse_deck_str(tmp_path, "\n".join(deck.lines), "FREE_SPACE")
        assert not log.has_errors
        m = model.mat_law87s[87]

        _assert_law87_all_fields_exact(
            m,
            model.materials[87],
            rho=2.7e-9,
            rhor=2.73e-9,
            e=69000.0,
            nu=0.34,
            iflag=1,
            iflagsr=1,
            invc=110.0,
            invp=4.2,
            ifit=0,
            alphas=[1.08, 1.12, 1.18, 1.22, 1.28, 1.32, 1.38, 1.42],
            chard=0.0,
            ikin=1,
            exp_a=6.0,
            alpha_vol=0.85,
            n_hard=0.26,
            fcut=510.0,
            fsmooth=0,
            aswift=440.0,
            eps0=0.014,
            qvoce=125.0,
            beta=24.0,
            k0=195.0,
            title="Free Space Sheet",
        )

    def test_free_format_direct_swift_voce_comma_delimited(self, tmp_path: Path):
        """Roundtrip comma-delimited free format direct Swift-Voce parameters."""
        deck = StarterDeck("FREE_COMMA")
        deck.mat_law87(
            mid=87,
            title="Free Comma Sheet",
            rho=2.7e-9,
            e=70000.0,
            nu=0.33,
            iflag=1,
            vp=1,
            c=105.0,
            p=4.1,
            ifit=0,
            alpha=[1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8],
            chard=0.25,
            ikin=1,
            exp_a=6.0,
            alpha_vol=0.7,
            n_hard=0.21,
            fcut=475.0,
            fsmooth=1,
            aswift=455.0,
            eps0=0.009,
            qvoce=118.0,
            beta=23.0,
            k0=202.0,
            ckh=(950.0, 450.0, 180.0, 90.0),
            akh=(48.0, 24.0, 12.0, 6.0),
            fixed_format=False,
            comma=True,
        )

        model, log = _parse_deck_str(tmp_path, "\n".join(deck.lines), "FREE_COMMA")
        assert not log.has_errors
        m = model.mat_law87s[87]

        _assert_law87_all_fields_exact(
            m,
            model.materials[87],
            rho=2.7e-9,
            e=70000.0,
            nu=0.33,
            iflag=1,
            iflagsr=1,
            invc=105.0,
            invp=4.1,
            ifit=0,
            alphas=[1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8],
            chard=0.25,
            ikin=1,
            exp_a=6.0,
            alpha_vol=0.7,
            n_hard=0.21,
            fcut=475.0,
            fsmooth=1,
            aswift=455.0,
            eps0=0.009,
            qvoce=118.0,
            beta=23.0,
            k0=202.0,
            ckh=(950.0, 450.0, 180.0, 90.0),
            akh=(48.0, 24.0, 12.0, 6.0),
            title="Free Comma Sheet",
        )

    def test_free_format_fitting_roundtrip(self, tmp_path: Path):
        """Roundtrip free format with experimental fitting (ifit=1)."""
        deck = StarterDeck("FREE_FIT")
        deck.mat_law87(
            mid=87,
            title="Free Fit Sheet",
            rho=7.85e-9,
            e=205000.0,
            nu=0.29,
            iflag=1,
            ifit=1,
            sigma_00=300.0,
            sigma_45=315.0,
            sigma_90=325.0,
            sigma_b=345.0,
            r_00=0.92,
            r_45=1.02,
            r_90=1.15,
            r_b=1.05,
            chard=0.0,
            ikin=1,
            exp_a=6.0,
            alpha_vol=0.8,
            n_hard=0.19,
            fcut=750.0,
            fsmooth=0,
            aswift=620.0,
            eps0=0.007,
            qvoce=165.0,
            beta=16.0,
            k0=295.0,
            fixed_format=False,
        )

        model, log = _parse_deck_str(tmp_path, "\n".join(deck.lines), "FREE_FIT")
        assert not log.has_errors
        m = model.mat_law87s[87]

        _assert_law87_all_fields_exact(
            m,
            model.materials[87],
            rho=7.85e-9,
            e=205000.0,
            nu=0.29,
            iflag=1,
            ifit=1,
            sigma_00=300.0,
            sigma_45=315.0,
            sigma_90=325.0,
            sigma_b=345.0,
            r_00=0.92,
            r_45=1.02,
            r_90=1.15,
            r_b=1.05,
            chard=0.0,
            ikin=1,
            exp_a=6.0,
            alpha_vol=0.8,
            n_hard=0.19,
            fcut=750.0,
            fsmooth=0,
            aswift=620.0,
            eps0=0.007,
            qvoce=165.0,
            beta=16.0,
            k0=295.0,
            title="Free Fit Sheet",
        )

    def test_free_format_tabulated_roundtrip(self, tmp_path: Path):
        """Roundtrip free format with tabulated multi-rate hardening (iflag=0)."""
        deck = StarterDeck("FREE_TAB")
        deck.mat_law87(
            mid=87,
            title="Free Tabulated Sheet",
            rho=2.7e-9,
            e=70000.0,
            nu=0.33,
            iflag=0,
            vp=1,
            c=95.0,
            p=3.9,
            ifit=0,
            alpha=[1.0, 1.05, 1.10, 1.15, 1.20, 1.25, 1.30, 1.35],
            chard=0.0,
            ikin=1,
            exp_a=6.0,
            fcut=550.0,
            fsmooth=1,
            curves=[
                (201, 1.0, 0.0),
                (202, 1.06, 25.0),
                (203, 1.14, 250.0),
            ],
            fixed_format=False,
        )

        model, log = _parse_deck_str(tmp_path, "\n".join(deck.lines), "FREE_TAB")
        assert not log.has_errors
        m = model.mat_law87s[87]

        _assert_law87_all_fields_exact(
            m,
            model.materials[87],
            rho=2.7e-9,
            e=70000.0,
            nu=0.33,
            iflag=0,
            iflagsr=1,
            invc=95.0,
            invp=3.9,
            ifit=0,
            alphas=[1.0, 1.05, 1.10, 1.15, 1.20, 1.25, 1.30, 1.35],
            chard=0.0,
            ikin=1,
            exp_a=6.0,
            fcut=550.0,
            fsmooth=1,
            num_curves=3,
            title="Free Tabulated Sheet",
        )


# ============================================================================
# 3. Synonyms Testing
# ============================================================================

class TestLaw87Synonyms:
    """Audit all keyword synonyms: /MAT/BARLAT2000, /MAT/BARLAT_2000, /MAT/BARLAT2000_2D, /MAT/BARLAT_YLD2000."""

    @pytest.mark.parametrize(
        "synonym,law_name",
        [
            ("BARLAT2000", "BARLAT2000"),
            ("BARLAT_2000", "BARLAT_2000"),
            ("BARLAT2000_2D", "BARLAT2000_2D"),
            ("BARLAT_YLD2000", "BARLAT_YLD2000"),
            ("LAW87", "LAW87"),
        ],
    )
    def test_synonyms_fixed_format_roundtrip(self, tmp_path: Path, synonym: str, law_name: str):
        """Ensure all synonyms generate valid headers and parse back into mat_law87s."""
        deck = StarterDeck(f"SYN_{synonym}")
        deck.mat_law87(
            mid=187,
            title=f"Synonym {synonym}",
            rho=2.7e-9,
            e=70000.0,
            nu=0.33,
            iflag=1,
            aswift=450.0,
            k0=200.0,
            law_name=synonym,
            fixed_format=True,
        )

        model, log = _parse_deck_str(tmp_path, "\n".join(deck.lines), f"SYN_{synonym}")
        assert not log.has_errors
        assert 187 in model.mat_law87s
        assert 187 in model.mat_barlat2000s
        assert 187 in model.mat_barlats
        assert 187 in model.materials

        m = model.mat_law87s[187]
        _assert_law87_all_fields_exact(
            m,
            model.materials[187],
            rho=2.7e-9,
            e=70000.0,
            nu=0.33,
            iflag=1,
            aswift=450.0,
            k0=200.0,
            title=f"Synonym {synonym}",
        )

    def test_synonym_deck_writer_methods(self, tmp_path: Path):
        """Verify dedicated StarterDeck helper methods for synonyms."""
        deck = StarterDeck("SYN_METHODS")
        deck.mat_barlat2000(mid=1, title="Method Barlat2000", rho=2.7e-9, e=70000.0, nu=0.33, iflag=1, aswift=400.0, k0=200.0)
        deck.mat_barlat_2000(mid=2, title="Method Barlat_2000", rho=2.7e-9, e=70000.0, nu=0.33, iflag=1, aswift=410.0, k0=205.0)
        deck.mat_barlat2000_2d(mid=3, title="Method Barlat2000_2D", rho=2.7e-9, e=70000.0, nu=0.33, iflag=1, aswift=420.0, k0=210.0)

        model, log = _parse_deck_str(tmp_path, "\n".join(deck.lines), "SYN_METHODS")
        assert not log.has_errors
        assert set(model.mat_law87s.keys()) == {1, 2, 3}
        assert set(model.materials.keys()) == {1, 2, 3}


# ============================================================================
# 4. Restart & Serialization
# ============================================================================

class TestLaw87Serialization:
    """Audit MatLaw87 and state variables (uvar87, thk87, pla87) serialization across JSON/pickle/dict."""

    def test_pickle_mat_law87_fidelity(self):
        """Verify MatLaw87 dataclass pickling fidelity across pickle protocols."""
        m = MatLaw87(
            id=87,
            title="Pickle-Law87",
            rho=2.7e-9,
            refer_rho=2.75e-9,
            e=71000.0,
            nu=0.33,
            iflag=1,
            iflagsr=1,
            invc=90.0,
            invp=3.5,
            flag_fit=0,
            al1=1.05,
            al2=1.10,
            al3=1.15,
            al4=1.20,
            al5=1.25,
            al6=1.30,
            al7=1.35,
            al8=1.40,
            chard=0.3,
            ikin=1,
            expa=6.0,
            fcut=500.0,
            fsmooth=1,
            aswift=460.0,
            epso=0.012,
            qvoce=125.0,
            beta=22.0,
            ko=205.0,
            ckh=(1000.0, 500.0, 200.0, 100.0),
            akh=(50.0, 25.0, 10.0, 5.0),
        )

        for proto in (4, 5, pickle.HIGHEST_PROTOCOL):
            data = pickle.dumps(m, protocol=proto)
            restored = pickle.loads(data)
            assert isinstance(restored, MatLaw87)
            _assert_law87_all_fields_exact(
                restored,
                rho=2.7e-9,
                rhor=2.75e-9,
                e=71000.0,
                nu=0.33,
                iflag=1,
                iflagsr=1,
                invc=90.0,
                invp=3.5,
                ifit=0,
                alphas=[1.05, 1.10, 1.15, 1.20, 1.25, 1.30, 1.35, 1.40],
                chard=0.3,
                ikin=1,
                exp_a=6.0,
                alpha_vol=0.0,
                fcut=500.0,
                fsmooth=1,
                aswift=460.0,
                eps0=0.012,
                qvoce=125.0,
                beta=22.0,
                k0=205.0,
                ckh=(1000.0, 500.0, 200.0, 100.0),
                akh=(50.0, 25.0, 10.0, 5.0),
                title="Pickle-Law87",
            )

    def test_json_and_dict_serialization_fidelity(self):
        """Verify MatLaw87 params dictionary serializes to JSON and restores accurately without loss."""
        m = MatLaw87(
            id=87,
            title="JSON-Law87",
            rho=2.7e-9,
            refer_rho=2.75e-9,
            e=70000.0,
            nu=0.33,
            iflag=1,
            iflagsr=1,
            invc=85.0,
            invp=3.6,
            flag_fit=0,
            al1=1.02,
            al2=1.04,
            al3=1.06,
            al4=1.08,
            al5=1.10,
            al6=1.12,
            al7=1.14,
            al8=1.16,
            chard=0.25,
            ikin=1,
            expa=6.0,
            fcut=480.0,
            fsmooth=1,
            aswift=450.0,
            epso=0.01,
            qvoce=120.0,
            beta=25.0,
            ko=210.0,
            ckh=(900.0, 450.0, 180.0, 90.0),
            akh=(45.0, 22.5, 10.0, 5.0),
        )

        p = m.params
        json_str = json.dumps(p)
        loaded_dict = json.loads(json_str)

        # Reconstruct MatLaw87 from JSON loaded dictionary
        restored = MatLaw87(**loaded_dict)
        _assert_law87_all_fields_exact(
            restored,
            rho=2.7e-9,
            rhor=2.75e-9,
            e=70000.0,
            nu=0.33,
            iflag=1,
            iflagsr=1,
            invc=85.0,
            invp=3.6,
            ifit=0,
            alphas=[1.02, 1.04, 1.06, 1.08, 1.10, 1.12, 1.14, 1.16],
            chard=0.25,
            ikin=1,
            exp_a=6.0,
            alpha_vol=0.0,
            fcut=480.0,
            fsmooth=1,
            aswift=450.0,
            eps0=0.01,
            qvoce=120.0,
            beta=25.0,
            k0=210.0,
            ckh=(900.0, 450.0, 180.0, 90.0),
            akh=(45.0, 22.5, 10.0, 5.0),
            title="JSON-Law87",
        )

    def test_state_variables_serialization_json_pickle_dict(self):
        """Verify uvar87, thk87, and pla87 serialize accurately to JSON, pickle, and dict."""
        state_vars = {
            "uvar87": np.array([[0.01, 12.5, -4.2, 3.1, 0.0, 0.0, 0.0]]),
            "thk87": np.array([[1.185, 1.192, 1.200]]),
            "pla87": np.array([[0.015, 0.022, 0.035]]),
        }

        # 1. Pickle
        pickled = pickle.dumps(state_vars, protocol=pickle.HIGHEST_PROTOCOL)
        from_pickle = pickle.loads(pickled)
        for k in state_vars:
            np.testing.assert_array_equal(from_pickle[k], state_vars[k])

        # 2. JSON
        json_repr = json.dumps({k: v.tolist() for k, v in state_vars.items()})
        from_json = {k: np.array(v) for k, v in json.loads(json_repr).items()}
        for k in state_vars:
            np.testing.assert_allclose(from_json[k], state_vars[k], rtol=1e-12, atol=1e-12)

        # 3. Dict copy
        from_dict = {k: v.copy() for k, v in state_vars.items()}
        for k in state_vars:
            np.testing.assert_array_equal(from_dict[k], state_vars[k])

    def test_write_read_restart_preserves_shell_state_variables(self, tmp_path: Path):
        """Verify write_restart and read_restart preserve uvar87, thk87, and pla87 in shell states."""
        deck = StarterDeck("RST_LAW87_SHELL")
        deck.node([
            (1, 0.0, 0.0, 0.0),
            (2, 10.0, 0.0, 0.0),
            (3, 10.0, 10.0, 0.0),
            (4, 0.0, 10.0, 0.0),
        ])
        deck.shell(1, [[1, 1, 2, 3, 4]])
        deck.prop_shell(1, "ShellProp", thick=1.2, nip=3)
        deck.part(1, "ShellPart", prop_id=1, mat_id=87)
        deck.mat_law87(
            mid=87,
            title="Restart-Barlat",
            rho=2.7e-9,
            e=70000.0,
            nu=0.33,
            iflag=1,
            aswift=450.0,
            k0=200.0,
        )

        rad_path = tmp_path / "rst_law87_0000.rad"
        deck.write(str(rad_path))

        model = run_starter(str(rad_path))
        assert model is not None

        groups = dict(model.element_groups())
        sh_grp = groups.get("shells")
        assert sh_grp is not None
        st = sh_grp.state

        # Synthesize state arrays
        uvar_elem = np.array([[0.01, 12.0, -4.0, 3.0, 0.0, 0.0, 0.0]], dtype=float)
        uvar_layers = np.zeros((1, 3, 7), dtype=float)
        uvar_layers[0, 0, :] = [0.01, 12.0, -4.0, 3.0, 0.0, 0.0, 0.0]
        thk_layers = np.array([[1.18, 1.19, 1.20]], dtype=float)
        pla_layers = np.array([[0.01, 0.02, 0.03]], dtype=float)

        st["uvar87"] = uvar_elem.copy()
        st["mat_extra"]["uvar87"] = uvar_layers.copy()
        st["mat_extra"]["thk87"] = thk_layers.copy()
        st["mat_extra"]["pla87"] = pla_layers.copy()

        # Write restart .rst
        rst_path = tmp_path / "rst_law87_0001.rst"
        engine_dict = {"cycle": 150, "t": 0.003, "dt": 1.0e-7}
        write_restart(model, str(rst_path), engine=engine_dict)

        # Read back from restart
        rest_model, rest_engine = read_restart(str(rst_path))
        assert rest_engine["cycle"] == 150
        assert rest_engine["t"] == pytest.approx(0.003)
        assert rest_engine["dt"] == pytest.approx(1.0e-7)

        rest_st = dict(rest_model.element_groups())["shells"].state
        rest_extra = rest_st["mat_extra"]

        np.testing.assert_array_equal(rest_st["uvar87"], uvar_elem)
        np.testing.assert_array_equal(rest_extra["uvar87"], uvar_layers)
        np.testing.assert_array_equal(rest_extra["thk87"], thk_layers)
        np.testing.assert_array_equal(rest_extra["pla87"], pla_layers)

    def test_constitutive_dynamic_cycle_continuation_from_restart(self):
        """Constitutive shell update continuation from restart matches uninterrupted simulation within 10^-12."""
        mat = build_law87({
            "id": 87,
            "rho0": 2.7e-9,
            "e": 70000.0,
            "nu": 0.33,
            "iflag": 1,
            "aswift": 450.0,
            "k0": 200.0,
            "qvoce": 100.0,
            "beta": 20.0,
            "eps0": 0.005,
            "alpha_vol": 0.5,
            "n_hard": 0.2,
            "exp_a": 6.0,
            "chard": 0.3,
            "ikin": 1,
            "ckh": (500.0, 200.0, 100.0, 50.0),
            "akh": (25.0, 10.0, 5.0, 2.0),
        })

        dt = 1.0e-6
        deps_steps = [
            np.array([[0.002, -0.0006, 0.0003]]),
            np.array([[0.003, -0.0009, 0.0005]]),
            np.array([[0.004, -0.0012, 0.0008]]),
            np.array([[0.005, -0.0015, 0.0010]]),
        ]

        # 1. Uninterrupted run: 4 steps
        sig_uninterrupted = np.zeros((1, 3))
        epsp_uninterrupted = np.zeros(1)
        extra_uninterrupted = {
            "uvar87": np.zeros((1, 7)),
            "pla87": np.zeros(1),
            "sigb87": np.zeros((1, 12)),
            "off87": np.ones(1),
            "thk": np.array([1.0]),
        }

        for deps in deps_steps:
            sig_uninterrupted, epsp_uninterrupted = shell_update(
                mat,
                sig_uninterrupted,
                deps=deps,
                epsp=epsp_uninterrupted,
                dt=dt,
                extra=extra_uninterrupted,
            )

        # 2. Resumed run: 2 steps, snapshot/serialize, 2 steps continued
        sig_resumed = np.zeros((1, 3))
        epsp_resumed = np.zeros(1)
        extra_resumed = {
            "uvar87": np.zeros((1, 7)),
            "pla87": np.zeros(1),
            "sigb87": np.zeros((1, 12)),
            "off87": np.ones(1),
            "thk": np.array([1.0]),
        }

        for deps in deps_steps[:2]:
            sig_resumed, epsp_resumed = shell_update(
                mat,
                sig_resumed,
                deps=deps,
                epsp=epsp_resumed,
                dt=dt,
                extra=extra_resumed,
            )

        # Snapshot
        sig_snap = pickle.loads(pickle.dumps(sig_resumed))
        epsp_snap = pickle.loads(pickle.dumps(epsp_resumed))
        extra_snap = pickle.loads(pickle.dumps(extra_resumed))

        for deps in deps_steps[2:]:
            sig_snap, epsp_snap = shell_update(
                mat,
                sig_snap,
                deps=deps,
                epsp=epsp_snap,
                dt=dt,
                extra=extra_snap,
            )

        # 3. Assert exact equality within 10^-12
        np.testing.assert_allclose(sig_snap, sig_uninterrupted, rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(epsp_snap, epsp_uninterrupted, rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(extra_snap["uvar87"], extra_uninterrupted["uvar87"], rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(extra_snap["pla87"], extra_uninterrupted["pla87"], rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(extra_snap["sigb87"], extra_uninterrupted["sigb87"], rtol=1e-12, atol=1e-12)


# ============================================================================
# 5. Negative Starter Diagnostics
# ============================================================================

class TestLaw87NegativeValidation:
    """Audit starter negative diagnostics in check_mat_law87, check_materials, and check_model."""

    def test_negative_and_zero_density(self):
        """Error when initial density RHO <= 0 (ANCMSG 1514)."""
        log1 = MessageLog()
        m1 = MatLaw87(id=1, rho=0.0, e=70000.0, nu=0.33)
        check_mat_law87(m1, log1)
        assert log1.has_errors
        assert any("initial density RHO must be > 0" in e and "ANCMSG 1514" in e for e in log1.errors)

        log2 = MessageLog()
        m2 = MatLaw87(id=2, rho=-2.7e-9, e=70000.0, nu=0.33)
        check_mat_law87(m2, log2)
        assert log2.has_errors
        assert any("initial density RHO must be > 0" in e and "ANCMSG 1514" in e for e in log2.errors)

    def test_negative_and_zero_young(self):
        """Error when Young's modulus E <= 0 (ANCMSG 1514)."""
        log1 = MessageLog()
        m1 = MatLaw87(id=1, rho=2.7e-9, e=0.0, nu=0.33)
        check_mat_law87(m1, log1)
        assert log1.has_errors
        assert any("Young's modulus E must be > 0" in e and "ANCMSG 1514" in e for e in log1.errors)

        log2 = MessageLog()
        m2 = MatLaw87(id=2, rho=2.7e-9, e=-70000.0, nu=0.33)
        check_mat_law87(m2, log2)
        assert log2.has_errors
        assert any("Young's modulus E must be > 0" in e and "ANCMSG 1514" in e for e in log2.errors)

    def test_poisson_ratio_bounds(self):
        """Error when nu < 0 or nu >= 0.5 (ANCMSG 1514)."""
        # Negative nu
        log1 = MessageLog()
        m1 = MatLaw87(id=1, rho=2.7e-9, e=70000.0, nu=-0.05)
        check_mat_law87(m1, log1)
        assert log1.has_errors
        assert any("ANCMSG 1514" in e for e in log1.errors)

        # Incompressible limit nu = 0.5
        log2 = MessageLog()
        m2 = MatLaw87(id=2, rho=2.7e-9, e=70000.0, nu=0.5)
        check_mat_law87(m2, log2)
        assert log2.has_errors
        assert any("ANCMSG 1514" in e for e in log2.errors)

        # Over limit nu > 0.5
        log3 = MessageLog()
        m3 = MatLaw87(id=3, rho=2.7e-9, e=70000.0, nu=0.55)
        check_mat_law87(m3, log3)
        assert log3.has_errors
        assert any("ANCMSG 1514" in e for e in log3.errors)

        # Valid bounds: nu = 0.0 and nu = 0.499
        log4 = MessageLog()
        m4 = MatLaw87(id=4, rho=2.7e-9, e=70000.0, nu=0.0)
        check_mat_law87(m4, log4)
        assert not log4.has_errors

        log5 = MessageLog()
        m5 = MatLaw87(id=5, rho=2.7e-9, e=70000.0, nu=0.499)
        check_mat_law87(m5, log5)
        assert not log5.has_errors

    def test_convexity_checks_lp_lpp_warnings(self):
        """Convexity check warnings when minimum eigenvalues of Lp or Lpp are non-positive (ANCMSG 3095 / 3102)."""
        # Non-convex Lp: al7 <= 0 -> Lp eigenvalue <= 0
        log1 = MessageLog()
        m1 = MatLaw87(id=1, rho=2.7e-9, e=70000.0, nu=0.33, al7=-1.0)
        check_mat_law87(m1, log1)
        assert not log1.has_errors
        assert len(log1.warnings) > 0
        assert any("ANCMSG 3095" in w for w in log1.warnings)

        # Non-convex Lpp: al8 <= 0 -> Lpp eigenvalue <= 0
        log2 = MessageLog()
        m2 = MatLaw87(id=2, rho=2.7e-9, e=70000.0, nu=0.33, al8=-1.0)
        check_mat_law87(m2, log2)
        assert not log2.has_errors
        assert len(log2.warnings) > 0
        assert any("ANCMSG 3102" in w for w in log2.warnings)

        # Both non-convex
        log3 = MessageLog()
        m3 = MatLaw87(id=3, rho=2.7e-9, e=70000.0, nu=0.33, al7=-0.5, al8=-0.5)
        check_mat_law87(m3, log3)
        assert any("ANCMSG 3095" in w for w in log3.warnings)
        assert any("ANCMSG 3102" in w for w in log3.warnings)

        # Convex: all alphas positive
        log4 = MessageLog()
        m4 = MatLaw87(id=4, rho=2.7e-9, e=70000.0, nu=0.33)
        check_mat_law87(m4, log4)
        assert not log4.has_errors
        assert len(log4.warnings) == 0

        # Fitting mode (ifit=1): direct Lp/Lpp eigenvalue check is bypassed
        log5 = MessageLog()
        m5 = MatLaw87(id=5, rho=2.7e-9, e=70000.0, nu=0.33, ifit=1, al7=-1.0, al8=-1.0)
        check_mat_law87(m5, log5)
        assert len(log5.warnings) == 0

    def test_incompatible_solid_elements_rejected(self):
        """Reject 3D solid elements and parts with ANCMSG 305."""
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
                self.materials: dict = {}
                self.parts: dict = {}
            def element_groups(self):
                return self._grps

        m = MatLaw87(id=1, rho=2.7e-9, e=70000.0, nu=0.33)
        solid_families = ("bricks", "bricks_heph", "bric20s", "tetras", "tetra10s", "penta6", "pyra5", "solids")
        for s_type in solid_families:
            model_s = DummyModel([(s_type, DummyGrp([DummyEl(1)]))])
            log_s = MessageLog()
            check_mat_law87(model=model_s, mat=m, log=log_s)
            assert log_s.has_errors, f"Solid element group {s_type} should be rejected"
            assert any("ANCMSG 305" in e for e in log_s.errors)

        # Via part dictionary
        for sol_etype in ("SOLID", "BRICK", "TETRA", "HEXA", "PENTA", "PYRA"):
            model_part = Model()
            model_part.mat_law87s[1] = m
            model_part.materials[1] = Material(id=1, law=87, rho0=2.7e-9, params={"rho0": 2.7e-9, "e": 70000.0, "nu": 0.33})
            p_solid = Part(id=1, prop_id=1, mat_id=1, title="Solid Part")
            p_solid.elem_type = sol_etype
            model_part.parts[1] = p_solid
            log_p = MessageLog()
            check_mat_law87(model_part, 1, m, log_p)
            assert log_p.has_errors, f"Solid part type {sol_etype} should be rejected"
            assert any("ANCMSG 305" in e for e in log_p.errors)

    def test_incompatible_1d_elements_rejected(self):
        """Reject 1D elements and parts with ANCMSG 306."""
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
                self.materials: dict = {}
                self.parts: dict = {}
            def element_groups(self):
                return self._grps

        m = MatLaw87(id=1, rho=2.7e-9, e=70000.0, nu=0.33)
        one_d_families = ("trusses", "beams", "springs")
        for d1_type in one_d_families:
            model_1d = DummyModel([(d1_type, DummyGrp([DummyEl(1)]))])
            log_1d = MessageLog()
            check_mat_law87(model=model_1d, mat=m, log=log_1d)
            assert log_1d.has_errors, f"1D element group {d1_type} should be rejected"
            assert any("ANCMSG 306" in e for e in log_1d.errors)

        # Via part dictionary
        for one_d_etype in ("BEAM", "TRUSS", "SPRING", "1D"):
            model_part = Model()
            model_part.mat_law87s[1] = m
            model_part.materials[1] = Material(id=1, law=87, rho0=2.7e-9, params={"rho0": 2.7e-9, "e": 70000.0, "nu": 0.33})
            p_1d = Part(id=1, prop_id=1, mat_id=1, title="1D Part")
            p_1d.elem_type = one_d_etype
            model_part.parts[1] = p_1d
            log_p = MessageLog()
            check_mat_law87(model_part, 1, m, log_p)
            assert log_p.has_errors, f"1D part type {one_d_etype} should be rejected"
            assert any("ANCMSG 306" in e for e in log_p.errors)

    def test_compatible_shell_elements_accepted(self):
        """Verify shell elements (shells, quads, tria3, shells_qeph, shells_bt4, sh3n) pass without diagnostic errors."""
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
                self.materials: dict = {}
                self.parts: dict = {}
            def element_groups(self):
                return self._grps

        m = MatLaw87(id=1, rho=2.7e-9, e=70000.0, nu=0.33)
        compatible_shell_types = ("shells", "quads", "tria3", "shells_qeph", "shells_bt4", "sh3n")
        for sh_type in compatible_shell_types:
            model_sh = DummyModel([(sh_type, DummyGrp([DummyEl(1)]))])
            log_sh = MessageLog()
            check_mat_law87(model=model_sh, mat=m, log=log_sh)
            assert not log_sh.has_errors, f"Shell element group {sh_type} should be accepted without errors"

        # Via part dictionary
        model_part = Model()
        model_part.mat_law87s[1] = m
        model_part.materials[1] = Material(id=1, law=87, rho0=2.7e-9, params={"rho0": 2.7e-9, "e": 70000.0, "nu": 0.33})
        p_sh = Part(id=1, prop_id=1, mat_id=1, title="Shell Part")
        p_sh.elem_type = "SHELL"
        model_part.parts[1] = p_sh
        log_p = MessageLog()
        check_mat_law87(model_part, 1, m, log_p)
        assert not log_p.has_errors, "Shell part should be accepted without errors"

    def test_checks_registry_and_dispatch(self):
        """Verify registry membership and check_materials dispatch."""
        for syn in (
            87, "87", "LAW87", "BARLAT", "BARLAT2000", "BARLAT_2000", "BARLAT2000_2D",
            "BARLAT_YLD2000", "MAT_LAW87", "MAT_BARLAT", "MAT_BARLAT2000",
            "MAT_BARLAT_2000", "MAT_BARLAT2000_2D", "MAT_BARLAT_YLD2000",
        ):
            assert syn in _MAT_CHECKS, f"{syn} missing from _MAT_CHECKS"
            assert _MAT_CHECKS[syn] is check_mat_law87

        for fam in ("shells", "shells_qbat", "shells_qeph", "sh3n", "quads", "shells_bt4", "tria3"):
            if fam in _ALLOWED_LAWS and _ALLOWED_LAWS[fam] is not None:
                assert 87 in _ALLOWED_LAWS[fam], f"87 missing from {fam} allowed laws"

        for fam in ("bricks", "tetras", "penta6", "pyra5", "solids", "trusses", "beams"):
            if fam in _ALLOWED_LAWS and _ALLOWED_LAWS[fam] is not None:
                assert 87 not in _ALLOWED_LAWS[fam], f"87 unexpectedly allowed for {fam}"

        # Dispatch via check_materials
        model = Model()
        model.materials[1] = Material(id=1, law=87, rho0=-1.0, params={"rho0": -1.0, "e": 70000.0, "nu": 0.33})
        log = MessageLog()
        check_materials(model, log)
        assert log.has_errors
        assert any("initial density RHO must be > 0" in e for e in log.errors)


# ============================================================================
# 6. Boundary Values & Item Access
# ============================================================================

class TestLaw87BoundaryValuesAndDefaults:
    """Audit boundary conditions, default initializations, and dictionary item access."""

    def test_default_values(self):
        """Verify standard defaults for unpopulated fields."""
        m = MatLaw87(id=87, rho=2.7e-9, e=70000.0, nu=0.33)
        assert m.iflag == 0
        assert m.iflagsr == 0
        assert m.invc == 0.0
        assert m.invp == 0.0
        assert m.flag_fit == 0
        assert m.al1 == 1.0
        assert m.al8 == 1.0
        assert m.alphas == [1.0] * 8
        assert m.fisokin == 0.0
        assert m.ikin == 1
        assert m.expa == 2.0
        assert m.fcut == 0.0
        assert m.fsmooth == 0
        assert m.nrate == 0
        assert m.aswift == 0.0
        assert m.nexp == 0.0
        assert m.qvoce == 0.0
        assert m.beta == 0.0
        assert m.ko == 0.0
        assert m.ckh == (0.0, 0.0, 0.0, 0.0)
        assert m.akh == (0.0, 0.0, 0.0, 0.0)
        assert m.curves == []

    def test_dictionary_item_access(self):
        """Verify dict-like indexing __getitem__, __setitem__, get, keys, values, and items."""
        m = MatLaw87(id=87, rho=2.7e-9, e=70000.0, nu=0.33)
        assert m["rho"] == pytest.approx(2.7e-9)
        assert m["E"] == pytest.approx(70000.0)
        assert m["Nu"] == pytest.approx(0.33)
        assert m["al1"] == 1.0
        assert m.get("missing_key", 42) == 42
        assert "al7" in m
        assert "sound_speed" in m

        m["custom_param"] = 123.45
        assert m["custom_param"] == 123.45
        assert "custom_param" in m
        assert "custom_param" in m.keys()
        assert len(m) == len(m.keys())
        assert len(list(m.values())) == len(m)
        assert len(list(m.items())) == len(m)

        with pytest.raises(KeyError):
            _ = m["completely_unknown_key"]
