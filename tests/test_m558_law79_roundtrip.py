"""
Milestone M558: /MAT/LAW79 (/MAT/JOHN_HOLM, /MAT/JOHNSON_HOLMQUIST, /MAT/JH2)
Exhaustive Roundtrip, Negative Validation, Boundary Cases & Restart (.rst) Serialization Auditor Suite.

Fortran origins:
  - starter/source/materials/mat/mat079/hm_read_mat79.F
  - engine/source/materials/mat/mat079/sigeps79.F
  - engine/source/output/restart/wrrestp.F & rdresb.F
  - starter/source/restart/ddsplit/wrrest.F
  - common_source/modules/constant_mod.F (INFINITY = 1E20)
  - config/CFG/radioss120/MAT/matl79_79.cfg & radioss2023/MAT/matl79_79.cfg

Audits:
  1. Fixed-Format Deck Roundtrip:
     - Standard 20-column fixed format with StarterDeck.mat_law79 (7 cards).
     - Card 1: RHO, [Refer_Rho] (MAT_LAW79_1: [20, 20])
     - Card 2: tau_shear (MAT_LAW79_2: [20])
     - Card 3: a, b, m, n (MAT_LAW79_3: [20, 20, 20, 20])
     - Card 4: c, eps0, sigfmax, fcut (MAT_LAW79_4: [20, 20, 20, 20])
     - Card 5: t, hel, phel (MAT_LAW79_5: [20, 20, 20])
     - Card 6: d1, d2, [blank], idel, epsmax (MAT_LAW79_6: [20, 20, 10, 10, 20])
     - Card 7: k1, k2, k3, beta (MAT_LAW79_7: [20, 20, 20, 20])
     - Re-parse using read_starter_deck / read_mat_law79.
     - Assert exact equality for all parameters in model.mat_law79s and model.materials.
     - Minimal card and default fallback handling.
     - Entity object invocation with MatLaw79 and Material instances.
     - Positional arguments with title invocation.

  2. Free-Format Comma-Delimited Roundtrip:
     - Write and re-parse free-format deck blocks for /MAT/LAW79 and synonyms.
     - Standard comma-separated with whitespace and compact comma-delimited without whitespace.
     - Cross-dialect roundtrip (free -> parse -> fixed -> parse -> exact equality).

  3. Keyword Synonyms:
     - /MAT/LAW79, /MAT/JOHN_HOLM, /MAT/JOHNSON_HOLMQUIST, /MAT/JH2.
     - Model dictionary aliases (mat_law79s, mat_john_holms, mat_jh2s).
     - Entity class aliases (MatLaw79, MatJohnHolm, MatJohnsonHolmquist, MatJH2).

  4. Negative Starter Diagnostics:
     - Non-positive density rho <= 0 (error).
     - Non-positive shear modulus shear <= 0 (ANCMSG 908 error).
     - Non-positive bulk modulus k1 <= 0 (ANCMSG 909 error).
     - Pressure at HEL phel > hel (ANCMSG 907 error).
     - Reference strain rate eps0 <= 0 (ANCMSG 910 error).
     - Bulking coefficient beta < 0 or beta > 1 (ANCMSG 911 error).
     - 2D analysis rejected: N2D > 0 (ANCMSG 305 error).
     - Incompatible element types: shells/quads (ANCMSG 305 error), 1D elements (ANCMSG 306 error).
     - Solid elements accepted (solids, bricks, tetras, penta6, pyra5).
     - Registry and dispatch validation (_ALLOWED_LAWS, _MAT_CHECKS, check_materials).

  5. Boundary Values & Defaults:
     - Default values when omitted: eps0=1.0, sigfmax=1e20, epsmax=1e20, refer_rho=rho, d2=1.0, beta=1.0.
     - idel clamped in [0, 3].
     - Boundary phel == hel (shel = 0.0).
     - Boundary beta in {0.0, 1.0}.
     - Derived properties: young, nu, shel, tstar, sound_speed.

  6. Restart (.rst) Serialization:
     - MatLaw79 entity dataclass pickling and unpickling fidelity.
     - Model with LAW79 serialization.
     - Persistent material state arrays: deltap, sigy_old, dmg, off preserved across write_restart / read_restart contract.
     - Constitutive dynamic cycle continuation from restart matching uninterrupted run within 10^-12.
"""

from __future__ import annotations

import math
from pathlib import Path
import pickle
from typing import Any, Dict, List, Tuple
import numpy as np
import pytest

from pyradioss.common.messages import MessageLog
from pyradioss.elements import solid_hexa8
from pyradioss.input.card_layouts import LAYOUTS, split_fixed
from pyradioss.input.deck_reader import read_deck
from pyradioss.input.deck_writer import StarterDeck
from pyradioss.input.starter_keywords import (
    parse_starter_deck,
    read_mat_law79,
    read_starter_deck,
)
from pyradioss.materials.law79_john_holm import (
    Law79Params,
    build_law79,
    solid_update_law79,
    sound_speed_solid,
)
from pyradioss import materials
from pyradioss.model.entities import (
    MatJH2,
    MatJohnHolm,
    MatJohnsonHolmquist,
    MatLaw79,
    Material,
)
from pyradioss.model.model import Model
from pyradioss.starter.checks import (
    _ALLOWED_LAWS,
    _MAT_CHECKS,
    check_mat_law79,
    check_materials,
    check_model,
)
from pyradioss.starter.restart import read_restart, write_restart
from pyradioss.starter.starter import run_starter


# ============================================================================
# Helpers
# ============================================================================

def _parse_deck_str(tmp_path: Path, text: str, name: str = "TEST_LAW79") -> Tuple[Model, MessageLog]:
    """Write deck text to file and parse into Model and MessageLog."""
    deck_path = tmp_path / f"{name}_0000.rad"
    deck_path.write_text(text.strip() + "\n", encoding="utf-8")
    blocks = read_deck(str(deck_path))
    model = Model()
    log = MessageLog()
    parse_starter_deck(blocks, model, log)
    return model, log


def _assert_law79_all_fields_exact(
    m: MatLaw79,
    mat: Material | None = None,
    *,
    rho: float,
    refer_rho: float,
    tau_shear: float,
    a: float,
    b: float,
    m_exp: float,
    n_exp: float,
    c: float,
    eps0: float,
    sigfmax: float,
    fcut: float,
    t: float,
    hel: float,
    phel: float,
    d1: float,
    d2: float,
    idel: int,
    epsmax: float,
    k1: float,
    k2: float,
    k3: float,
    beta: float,
    title: str = "",
) -> None:
    """Assert exact (floating-point tolerance) equality for all LAW79 parameters."""
    # Density
    assert m.rho == pytest.approx(rho, rel=1e-6, abs=1e-12)
    assert m.refer_rho == pytest.approx(refer_rho, rel=1e-6, abs=1e-12)
    assert m.rho0 == pytest.approx(rho, rel=1e-6, abs=1e-12)
    assert m.rhor == pytest.approx(refer_rho, rel=1e-6, abs=1e-12)

    # Shear modulus
    assert m.tau_shear == pytest.approx(tau_shear, rel=1e-6, abs=1e-12)
    assert m.shear == pytest.approx(tau_shear, rel=1e-6, abs=1e-12)
    assert m.G == pytest.approx(tau_shear, rel=1e-6, abs=1e-12)
    assert m.g == pytest.approx(tau_shear, rel=1e-6, abs=1e-12)

    # Strength constants
    assert m.a == pytest.approx(a, rel=1e-6, abs=1e-12)
    assert m.b == pytest.approx(b, rel=1e-6, abs=1e-12)
    assert m.m == pytest.approx(m_exp, rel=1e-6, abs=1e-12)
    assert m.n == pytest.approx(n_exp, rel=1e-6, abs=1e-12)

    # Rate parameters
    assert m.c == pytest.approx(c, rel=1e-6, abs=1e-12)
    assert m.eps0 == pytest.approx(eps0, rel=1e-6, abs=1e-12)
    assert m.sigfmax == pytest.approx(sigfmax, rel=1e-6, abs=1e-12)
    assert m.sigma_fmax == pytest.approx(sigfmax, rel=1e-6, abs=1e-12)
    assert m.fcut == pytest.approx(fcut, rel=1e-6, abs=1e-12)

    # Limits
    assert m.t == pytest.approx(t, rel=1e-6, abs=1e-12)
    assert m.t0 == pytest.approx(t, rel=1e-6, abs=1e-12)
    assert m.hel == pytest.approx(hel, rel=1e-6, abs=1e-12)
    assert m.phel == pytest.approx(phel, rel=1e-6, abs=1e-12)

    # Damage & deletion
    assert m.d1 == pytest.approx(d1, rel=1e-6, abs=1e-12)
    assert m.d2 == pytest.approx(d2, rel=1e-6, abs=1e-12)
    assert m.idel == idel
    assert m.epsmax == pytest.approx(epsmax, rel=1e-6, abs=1e-12)
    assert m.eps_max == pytest.approx(epsmax, rel=1e-6, abs=1e-12)

    # Bulk modulus & pressure coefficients
    assert m.k1 == pytest.approx(k1, rel=1e-6, abs=1e-12)
    assert m.k2 == pytest.approx(k2, rel=1e-6, abs=1e-12)
    assert m.k3 == pytest.approx(k3, rel=1e-6, abs=1e-12)
    assert m.bulk == pytest.approx(k1, rel=1e-6, abs=1e-12)
    assert m.K == pytest.approx(k1, rel=1e-6, abs=1e-12)
    assert m.beta == pytest.approx(beta, rel=1e-6, abs=1e-12)

    # Derived values
    expected_shel = 1.5 * (hel - phel)
    expected_tstar = (t / phel) if phel != 0.0 else 0.0
    denom_e = 3.0 * k1 + tau_shear
    expected_young = (9.0 * k1 * tau_shear) / denom_e if denom_e > 0.0 else 0.0
    denom_nu = 6.0 * k1 + 2.0 * tau_shear
    expected_nu = (3.0 * k1 - 2.0 * tau_shear) / denom_nu if denom_nu > 0.0 else 0.0

    assert m.shel == pytest.approx(expected_shel, rel=1e-6, abs=1e-12)
    assert m.tstar == pytest.approx(expected_tstar, rel=1e-6, abs=1e-12)
    assert m.young == pytest.approx(expected_young, rel=1e-6, abs=1e-12)
    assert m.E == pytest.approx(expected_young, rel=1e-6, abs=1e-12)
    assert m.nu == pytest.approx(expected_nu, rel=1e-6, abs=1e-12)

    if title:
        assert m.title == title

    # Verify model.materials[mid] parameters if provided
    if mat is not None:
        assert mat.law == 79
        assert mat.rho0 == pytest.approx(rho, rel=1e-6, abs=1e-12)
        p = mat.params
        assert p["rho"] == pytest.approx(rho, rel=1e-6, abs=1e-12)
        assert p["refer_rho"] == pytest.approx(refer_rho, rel=1e-6, abs=1e-12)
        assert p["tau_shear"] == pytest.approx(tau_shear, rel=1e-6, abs=1e-12)
        assert p["shear"] == pytest.approx(tau_shear, rel=1e-6, abs=1e-12)
        assert p["G"] == pytest.approx(tau_shear, rel=1e-6, abs=1e-12)
        assert p["a"] == pytest.approx(a, rel=1e-6, abs=1e-12)
        assert p["b"] == pytest.approx(b, rel=1e-6, abs=1e-12)
        assert p["m"] == pytest.approx(m_exp, rel=1e-6, abs=1e-12)
        assert p["n"] == pytest.approx(n_exp, rel=1e-6, abs=1e-12)
        assert p["c"] == pytest.approx(c, rel=1e-6, abs=1e-12)
        assert p["eps0"] == pytest.approx(eps0, rel=1e-6, abs=1e-12)
        assert p["sigfmax"] == pytest.approx(sigfmax, rel=1e-6, abs=1e-12)
        assert p["sigma_fmax"] == pytest.approx(sigfmax, rel=1e-6, abs=1e-12)
        assert p["fcut"] == pytest.approx(fcut, rel=1e-6, abs=1e-12)
        assert p["t"] == pytest.approx(t, rel=1e-6, abs=1e-12)
        assert p["t0"] == pytest.approx(t, rel=1e-6, abs=1e-12)
        assert p["hel"] == pytest.approx(hel, rel=1e-6, abs=1e-12)
        assert p["phel"] == pytest.approx(phel, rel=1e-6, abs=1e-12)
        assert p["d1"] == pytest.approx(d1, rel=1e-6, abs=1e-12)
        assert p["d2"] == pytest.approx(d2, rel=1e-6, abs=1e-12)
        assert p["idel"] == idel
        assert p["epsmax"] == pytest.approx(epsmax, rel=1e-6, abs=1e-12)
        assert p["eps_max"] == pytest.approx(epsmax, rel=1e-6, abs=1e-12)
        assert p["k1"] == pytest.approx(k1, rel=1e-6, abs=1e-12)
        assert p["k2"] == pytest.approx(k2, rel=1e-6, abs=1e-12)
        assert p["k3"] == pytest.approx(k3, rel=1e-6, abs=1e-12)
        assert p["beta"] == pytest.approx(beta, rel=1e-6, abs=1e-12)
        assert p["shel"] == pytest.approx(expected_shel, rel=1e-6, abs=1e-12)
        assert p["tstar"] == pytest.approx(expected_tstar, rel=1e-6, abs=1e-12)
        assert p["E"] == pytest.approx(expected_young, rel=1e-6, abs=1e-12)
        assert p["nu"] == pytest.approx(expected_nu, rel=1e-6, abs=1e-12)


# ============================================================================
# 1. Fixed-Format Deck Roundtrip
# ============================================================================

class TestLaw79FixedFormatRoundtrip:
    """Audit StarterDeck.mat_law79 20-column fixed-format emission and re-reading."""

    def test_fixed_format_card_columns_and_roundtrip(self, tmp_path: Path):
        """Verify exact 20-column layout of all 7 data cards and complete roundtrip."""
        vals = {
            "rho": 2.53,
            "refer_rho": 2.50,
            "tau_shear": 3.04e10,
            "a": 0.93,
            "b": 0.70,
            "m_exp": 0.60,
            "n_exp": 0.64,
            "c": 0.003,
            "eps0": 1.0,
            "sigfmax": 1.50e9,
            "fcut": 1000.0,
            "t": 2.0e8,
            "hel": 2.10e9,
            "phel": 1.20e9,
            "d1": 0.05,
            "d2": 0.85,
            "idel": 1,
            "epsmax": 0.45,
            "k1": 5.20e10,
            "k2": -3.50e10,
            "k3": 1.20e11,
            "beta": 0.80,
        }

        deck = StarterDeck("AUDIT_FIXED_LAW79")
        deck.mat_law79(
            mat_id=79,
            title="Boron-Carbide-JH2",
            fixed_format=True,
            rho=vals["rho"],
            refer_rho=vals["refer_rho"],
            tau_shear=vals["tau_shear"],
            a=vals["a"],
            b=vals["b"],
            m=vals["m_exp"],
            n=vals["n_exp"],
            c=vals["c"],
            eps0=vals["eps0"],
            sigfmax=vals["sigfmax"],
            fcut=vals["fcut"],
            t=vals["t"],
            hel=vals["hel"],
            phel=vals["phel"],
            d1=vals["d1"],
            d2=vals["d2"],
            idel=vals["idel"],
            epsmax=vals["epsmax"],
            k1=vals["k1"],
            k2=vals["k2"],
            k3=vals["k3"],
            beta=vals["beta"],
        )

        rendered = deck.render()
        lines = [line.rstrip() for line in rendered.splitlines()]

        # Locate keyword block
        idx = lines.index("/MAT/LAW79/79")
        assert lines[idx + 1] == "Boron-Carbide-JH2"

        # Card 1: RHO, Refer_Rho (each 20 cols -> 40 cols)
        c1_line = lines[idx + 3]
        assert len(c1_line) == 40
        f1 = split_fixed(c1_line, LAYOUTS["MAT_LAW79_1"])
        assert float(f1[0]) == pytest.approx(vals["rho"])
        assert float(f1[1]) == pytest.approx(vals["refer_rho"])

        # Card 2: tau_shear (20 cols)
        c2_line = lines[idx + 5]
        assert len(c2_line) == 20
        f2 = split_fixed(c2_line, LAYOUTS["MAT_LAW79_2"])
        assert float(f2[0]) == pytest.approx(vals["tau_shear"])

        # Card 3: a, b, m, n (each 20 cols -> 80 cols)
        c3_line = lines[idx + 7]
        assert len(c3_line) == 80
        f3 = split_fixed(c3_line, LAYOUTS["MAT_LAW79_3"])
        assert float(f3[0]) == pytest.approx(vals["a"])
        assert float(f3[1]) == pytest.approx(vals["b"])
        assert float(f3[2]) == pytest.approx(vals["m_exp"])
        assert float(f3[3]) == pytest.approx(vals["n_exp"])

        # Card 4: c, eps0, sigfmax, fcut (each 20 cols -> 80 cols)
        c4_line = lines[idx + 9]
        assert len(c4_line) == 80
        f4 = split_fixed(c4_line, LAYOUTS["MAT_LAW79_4"])
        assert float(f4[0]) == pytest.approx(vals["c"])
        assert float(f4[1]) == pytest.approx(vals["eps0"])
        assert float(f4[2]) == pytest.approx(vals["sigfmax"])
        assert float(f4[3]) == pytest.approx(vals["fcut"])

        # Card 5: t, hel, phel (each 20 cols -> 60 cols)
        c5_line = lines[idx + 11]
        assert len(c5_line) == 60
        f5 = split_fixed(c5_line, LAYOUTS["MAT_LAW79_5"])
        assert float(f5[0]) == pytest.approx(vals["t"])
        assert float(f5[1]) == pytest.approx(vals["hel"])
        assert float(f5[2]) == pytest.approx(vals["phel"])

        # Card 6: d1, d2, blank, idel, epsmax (20 + 20 + 10 + 10 + 20 = 80 cols)
        c6_line = lines[idx + 13]
        assert len(c6_line) == 80
        f6 = split_fixed(c6_line, LAYOUTS["MAT_LAW79_6"])
        assert float(f6[0]) == pytest.approx(vals["d1"])
        assert float(f6[1]) == pytest.approx(vals["d2"])
        assert f6[2] == ""  # blank field
        assert int(f6[3]) == vals["idel"]
        assert float(f6[4]) == pytest.approx(vals["epsmax"])

        # Card 7: k1, k2, k3, beta (each 20 cols -> 80 cols)
        c7_line = lines[idx + 15]
        assert len(c7_line) == 80
        f7 = split_fixed(c7_line, LAYOUTS["MAT_LAW79_7"])
        assert float(f7[0]) == pytest.approx(vals["k1"])
        assert float(f7[1]) == pytest.approx(vals["k2"])
        assert float(f7[2]) == pytest.approx(vals["k3"])
        assert float(f7[3]) == pytest.approx(vals["beta"])

        # Roundtrip parse with read_starter_deck
        rad_path = tmp_path / "audit_fixed_law79_0000.rad"
        rad_path.write_text(rendered + "\n/END\n", encoding="utf-8")

        model = read_starter_deck(str(rad_path))
        assert 79 in model.mat_law79s
        assert 79 in model.materials

        m = model.mat_law79s[79]
        mat = model.materials[79]
        _assert_law79_all_fields_exact(
            m,
            mat,
            title="Boron-Carbide-JH2",
            **vals,
        )

    def test_fixed_format_minimal_defaults_roundtrip(self, tmp_path: Path):
        """Verify minimal fixed-format card omission and fallback defaults."""
        deck = StarterDeck("MINIMAL_FIXED_LAW79")
        deck.mat_law79(
            mat_id=1,
            title="Minimal JH2",
            rho=2.5,
            tau_shear=1.0e10,
            k1=2.0e10,
        )

        rendered = deck.render()
        rad_path = tmp_path / "law79_minimal_0000.rad"
        rad_path.write_text(rendered + "\n/END\n", encoding="utf-8")

        model, log = read_starter_deck(str(rad_path))
        assert not log.has_errors
        assert 1 in model.mat_law79s
        assert 1 in model.materials

        m = model.mat_law79s[1]
        mat = model.materials[1]
        _assert_law79_all_fields_exact(
            m,
            mat,
            rho=2.5,
            refer_rho=2.5,     # Defaults to rho
            tau_shear=1.0e10,
            a=0.0,
            b=0.0,
            m_exp=0.0,
            n_exp=0.0,
            c=0.0,
            eps0=1.0,          # Default eps0 = 1.0
            sigfmax=1.0e20,    # Default sigfmax = 1e20 (INFINITY)
            fcut=0.0,
            t=0.0,
            hel=0.0,
            phel=0.0,
            d1=0.0,
            d2=1.0,            # Default d2 = 1.0
            idel=0,            # Default idel = 0
            epsmax=1.0e20,     # Default epsmax = 1e20 (INFINITY)
            k1=2.0e10,
            k2=0.0,
            k3=0.0,
            beta=1.0,          # Default beta = 1.0
            title="Minimal JH2",
        )

    def test_entity_object_invocation_matlaw79(self, tmp_path: Path):
        """Verify StarterDeck.mat_law79 accepts a pre-built MatLaw79 dataclass instance."""
        entity = MatLaw79(
            id=12,
            rho=3.21,
            refer_rho=3.21,
            tau_shear=1.5e11,
            a=0.90,
            b=0.75,
            m=0.62,
            n=0.65,
            c=0.005,
            eps0=1.0,
            sigfmax=2.0e9,
            fcut=500.0,
            t=3.0e8,
            hel=3.5e9,
            phel=2.0e9,
            d1=0.08,
            d2=0.90,
            idel=2,
            epsmax=0.30,
            k1=2.2e11,
            k2=1.0e11,
            k3=5.0e10,
            beta=0.95,
            title="Silicon-Nitride",
        )

        deck = StarterDeck("ENTITY_LAW79")
        deck.mat_law79(entity)

        rendered = deck.render()
        rad_path = tmp_path / "entity_law79_0000.rad"
        rad_path.write_text(rendered + "\n/END\n", encoding="utf-8")

        model = read_starter_deck(str(rad_path))
        assert 12 in model.mat_law79s
        _assert_law79_all_fields_exact(
            model.mat_law79s[12],
            model.materials[12],
            rho=3.21,
            refer_rho=3.21,
            tau_shear=1.5e11,
            a=0.90,
            b=0.75,
            m_exp=0.62,
            n_exp=0.65,
            c=0.005,
            eps0=1.0,
            sigfmax=2.0e9,
            fcut=500.0,
            t=3.0e8,
            hel=3.5e9,
            phel=2.0e9,
            d1=0.08,
            d2=0.90,
            idel=2,
            epsmax=0.30,
            k1=2.2e11,
            k2=1.0e11,
            k3=5.0e10,
            beta=0.95,
            title="Silicon-Nitride",
        )

    def test_entity_object_invocation_material(self, tmp_path: Path):
        """Verify StarterDeck.mat_law79 accepts a Material instance with params dict."""
        mat_inst = Material(
            id=33,
            law=79,
            rho0=2.45,
            title="Soda-Lime-Glass",
            params={
                "refer_rho": 2.45,
                "tau_shear": 2.5e10,
                "a": 0.85,
                "b": 0.65,
                "m": 0.55,
                "n": 0.70,
                "c": 0.002,
                "eps0": 1.0,
                "sigfmax": 1.2e9,
                "fcut": 0.0,
                "t": 1.5e8,
                "hel": 1.8e9,
                "phel": 1.0e9,
                "d1": 0.04,
                "d2": 0.80,
                "idel": 3,
                "epsmax": 0.25,
                "k1": 4.5e10,
                "k2": 0.0,
                "k3": 0.0,
                "beta": 0.75,
            },
        )

        deck = StarterDeck("MATERIAL_LAW79")
        deck.mat_law79(mat_inst)

        rendered = deck.render()
        rad_path = tmp_path / "material_law79_0000.rad"
        rad_path.write_text(rendered + "\n/END\n", encoding="utf-8")

        model = read_starter_deck(str(rad_path))
        assert 33 in model.mat_law79s
        _assert_law79_all_fields_exact(
            model.mat_law79s[33],
            model.materials[33],
            rho=2.45,
            refer_rho=2.45,
            tau_shear=2.5e10,
            a=0.85,
            b=0.65,
            m_exp=0.55,
            n_exp=0.70,
            c=0.002,
            eps0=1.0,
            sigfmax=1.2e9,
            fcut=0.0,
            t=1.5e8,
            hel=1.8e9,
            phel=1.0e9,
            d1=0.04,
            d2=0.80,
            idel=3,
            epsmax=0.25,
            k1=4.5e10,
            k2=0.0,
            k3=0.0,
            beta=0.75,
            title="Soda-Lime-Glass",
        )

    def test_positional_arguments_with_title(self, tmp_path: Path):
        """Verify mat_law79(id, 'title', rho, tau_shear, a, b, ...) positional argument handling."""
        deck = StarterDeck("POS_LAW79")
        deck.mat_law79(
            5,
            "Alumina-AD995",
            3.89,     # rho
            1.52e11,  # tau_shear
            0.88,     # a
            0.45,     # b
            0.64,     # m
            0.64,     # n
            0.007,    # c
            1.0,      # eps0
            3.0e9,    # sigfmax
            0.0,      # fcut
            2.62e8,   # t
            6.7e9,    # hel
            4.0e9,    # phel
            0.01,     # d1
            0.70,     # d2
            1,        # idel
            0.50,     # epsmax
            2.31e11,  # k1
            0.0,      # k2
            0.0,      # k3
            1.0,      # beta
        )

        rendered = deck.render()
        rad_path = tmp_path / "pos_law79_0000.rad"
        rad_path.write_text(rendered + "\n/END\n", encoding="utf-8")

        model = read_starter_deck(str(rad_path))
        assert 5 in model.mat_law79s
        m = model.mat_law79s[5]
        assert m.title == "Alumina-AD995"
        assert m.rho == pytest.approx(3.89)
        assert m.tau_shear == pytest.approx(1.52e11)
        assert m.k1 == pytest.approx(2.31e11)
        assert m.hel == pytest.approx(6.7e9)
        assert m.phel == pytest.approx(4.0e9)


# ============================================================================
# 2. Free-Format Comma-Delimited Roundtrip
# ============================================================================

class TestLaw79FreeFormatRoundtrip:
    """Audit free-format comma-separated parsing and cross-dialect consistency."""

    def test_free_format_comma_separated_with_spaces(self, tmp_path: Path):
        """Parse free-format deck blocks with standard comma-space separation."""
        deck = StarterDeck("FREE_SPACE_LAW79")
        deck.mat_law79(
            mat_id=1,
            title="Free Space JH2",
            fixed_format=False,
            comma=True,
            rho=2.5,
            refer_rho=2.5,
            tau_shear=1.2e10,
            a=0.9,
            b=0.7,
            m=0.6,
            n=0.64,
            c=0.003,
            eps0=1.0,
            sigfmax=1.0e9,
            fcut=100.0,
            t=2.0e8,
            hel=2.0e9,
            phel=1.2e9,
            d1=0.05,
            d2=0.85,
            idel=1,
            epsmax=0.40,
            k1=3.5e10,
            k2=1.0e10,
            k3=5.0e9,
            beta=0.85,
        )

        rendered = deck.render()
        rad_path = tmp_path / "free_space_law79_0000.rad"
        rad_path.write_text(rendered + "\n/END\n", encoding="utf-8")

        model, log = read_starter_deck(str(rad_path))
        assert not log.has_errors
        assert 1 in model.mat_law79s
        _assert_law79_all_fields_exact(
            model.mat_law79s[1],
            model.materials[1],
            rho=2.5,
            refer_rho=2.5,
            tau_shear=1.2e10,
            a=0.9,
            b=0.7,
            m_exp=0.6,
            n_exp=0.64,
            c=0.003,
            eps0=1.0,
            sigfmax=1.0e9,
            fcut=100.0,
            t=2.0e8,
            hel=2.0e9,
            phel=1.2e9,
            d1=0.05,
            d2=0.85,
            idel=1,
            epsmax=0.40,
            k1=3.5e10,
            k2=1.0e10,
            k3=5.0e9,
            beta=0.85,
            title="Free Space JH2",
        )

    def test_free_format_compact_comma_delimited(self, tmp_path: Path):
        """Parse compact comma-delimited cards without whitespace."""
        deck_text = """/BEGIN
COMPACT_COMMA_LAW79
/MAT/LAW79/2
Compact Comma Deck
2.5,2.5
1.2e10
0.9,0.7,0.6,0.64
0.003,1.0,1.0e9,100.0
2.0e8,2.0e9,1.2e9
0.05,0.85,1,0.40
3.5e10,1.0e10,5.0e9,0.85
/END
"""
        model, log = _parse_deck_str(tmp_path, deck_text, "compact_law79")
        assert not log.has_errors
        assert 2 in model.mat_law79s
        _assert_law79_all_fields_exact(
            model.mat_law79s[2],
            model.materials[2],
            rho=2.5,
            refer_rho=2.5,
            tau_shear=1.2e10,
            a=0.9,
            b=0.7,
            m_exp=0.6,
            n_exp=0.64,
            c=0.003,
            eps0=1.0,
            sigfmax=1.0e9,
            fcut=100.0,
            t=2.0e8,
            hel=2.0e9,
            phel=1.2e9,
            d1=0.05,
            d2=0.85,
            idel=1,
            epsmax=0.40,
            k1=3.5e10,
            k2=1.0e10,
            k3=5.0e9,
            beta=0.85,
            title="Compact Comma Deck",
        )

    def test_cross_dialect_roundtrip(self, tmp_path: Path):
        """Verify free format -> parse -> fixed format -> parse yields identical values."""
        deck_free = StarterDeck("CROSS_DIALECT_FREE")
        deck_free.mat_law79(
            mat_id=1,
            title="Cross-Dialect JH2",
            fixed_format=False,
            comma=True,
            rho=3.10,
            refer_rho=3.05,
            tau_shear=2.2e11,
            a=0.95,
            b=0.80,
            m=0.65,
            n=0.70,
            c=0.004,
            eps0=1.0,
            sigfmax=2.5e9,
            fcut=250.0,
            t=2.5e8,
            hel=3.0e9,
            phel=1.8e9,
            d1=0.06,
            d2=0.90,
            idel=2,
            epsmax=0.35,
            k1=3.0e11,
            k2=1.5e11,
            k3=8.0e10,
            beta=0.90,
        )

        f1 = tmp_path / "cross_free_0000.rad"
        deck_free.write(str(f1))
        m1 = read_starter_deck(str(f1))

        # Re-emit as fixed format
        mat1 = m1.mat_law79s[1]
        deck_fixed = StarterDeck("CROSS_DIALECT_FIXED")
        deck_fixed.mat_law79(mat1, fixed_format=True)

        f2 = tmp_path / "cross_fixed_0000.rad"
        deck_fixed.write(str(f2))
        m2 = read_starter_deck(str(f2))

        # Compare models
        _assert_law79_all_fields_exact(
            m2.mat_law79s[1],
            m2.materials[1],
            rho=3.10,
            refer_rho=3.05,
            tau_shear=2.2e11,
            a=0.95,
            b=0.80,
            m_exp=0.65,
            n_exp=0.70,
            c=0.004,
            eps0=1.0,
            sigfmax=2.5e9,
            fcut=250.0,
            t=2.5e8,
            hel=3.0e9,
            phel=1.8e9,
            d1=0.06,
            d2=0.90,
            idel=2,
            epsmax=0.35,
            k1=3.0e11,
            k2=1.5e11,
            k3=8.0e10,
            beta=0.90,
            title="Cross-Dialect JH2",
        )


# ============================================================================
# 3. Keyword Synonyms
# ============================================================================

class TestLaw79Synonyms:
    """Audit all keyword synonyms: /MAT/LAW79, /MAT/JOHN_HOLM, /MAT/JOHNSON_HOLMQUIST, /MAT/JH2."""

    @pytest.mark.parametrize("writer_method,keyword_str", [
        ("mat_law79", "/MAT/LAW79"),
        ("mat_john_holm", "/MAT/JOHN_HOLM"),
        ("mat_johnson_holmquist", "/MAT/JOHNSON_HOLMQUIST"),
        ("mat_jh2", "/MAT/JH2"),
    ])
    def test_synonym_emission_and_parsing(self, tmp_path: Path, writer_method: str, keyword_str: str):
        """Verify each synonym emits correct header and parses into mat_law79s and materials."""
        deck = StarterDeck(f"TEST_{writer_method.upper()}")
        method = getattr(deck, writer_method)
        method(
            mat_id=10,
            title=f"Synonym {keyword_str}",
            rho=2.5,
            tau_shear=1.0e10,
            a=0.5,
            b=0.4,
            k1=2.0e10,
        )

        rendered = deck.render()
        assert f"{keyword_str}/10" in rendered

        rad_path = tmp_path / f"{writer_method}_0000.rad"
        rad_path.write_text(rendered + "\n/END\n", encoding="utf-8")

        model, log = read_starter_deck(str(rad_path))
        assert not log.has_errors
        assert 10 in model.mat_law79s
        assert 10 in model.materials

        # Verify model aliases
        assert 10 in model.mat_john_holms
        assert 10 in model.mat_jh2s
        assert model.mat_law79s[10] is model.mat_john_holms[10]
        assert model.mat_law79s[10] is model.mat_jh2s[10]

    def test_entity_class_aliases(self):
        """Verify MatJohnHolm, MatJohnsonHolmquist, MatJH2 are exact aliases of MatLaw79."""
        assert MatJohnHolm is MatLaw79
        assert MatJohnsonHolmquist is MatLaw79
        assert MatJH2 is MatLaw79


# ============================================================================
# 4. Negative Starter Diagnostics
# ============================================================================

class TestLaw79NegativeStarterDiagnostics:
    """Audit error diagnostics mandated by hm_read_mat79.F and check_mat_law79."""

    def test_non_positive_density_error(self):
        """Density rho <= 0 triggers fatal starter diagnostic."""
        for bad_rho in (0.0, -100.0):
            log = MessageLog()
            m = MatLaw79(id=1, rho=bad_rho, tau_shear=1.0e10, k1=2.0e10)
            check_mat_law79(mat=m, log=log)
            assert log.has_errors
            assert any("initial density RHO must be > 0" in e for e in log.errors)

    def test_non_positive_shear_modulus_ancmsg_908(self):
        """Shear modulus shear <= 0 triggers ANCMSG 908 error."""
        for bad_shear in (0.0, -1.0e8):
            log = MessageLog()
            m = MatLaw79(id=2, rho=2.5, tau_shear=bad_shear, k1=2.0e10)
            check_mat_law79(mat=m, log=log)
            assert log.has_errors
            assert any("shear modulus must be > 0" in e and "ANCMSG 908" in e for e in log.errors)

    def test_non_positive_bulk_modulus_ancmsg_909(self):
        """Bulk modulus k1 <= 0 triggers ANCMSG 909 error."""
        for bad_k1 in (0.0, -5.0e9):
            log = MessageLog()
            m = MatLaw79(id=3, rho=2.5, tau_shear=1.0e10, k1=bad_k1)
            check_mat_law79(mat=m, log=log)
            assert log.has_errors
            assert any("bulk modulus K1 must be > 0" in e and "ANCMSG 909" in e for e in log.errors)

    def test_phel_greater_than_hel_ancmsg_907(self):
        """Pressure at HEL phel > hel triggers ANCMSG 907 error."""
        log = MessageLog()
        m = MatLaw79(id=4, rho=2.5, tau_shear=1.0e10, k1=2.0e10, hel=1.5e9, phel=2.0e9)
        check_mat_law79(mat=m, log=log)
        assert log.has_errors
        assert any("cannot exceed HEL" in e and "ANCMSG 907" in e for e in log.errors)

    def test_non_positive_eps0_ancmsg_910(self):
        """Reference strain rate eps0 <= 0 triggers ANCMSG 910 error."""
        for bad_eps0 in (0.0, -0.5):
            log = MessageLog()
            m = MatLaw79(id=5, rho=2.5, tau_shear=1.0e10, k1=2.0e10)
            check_mat_law79(mat=m, eps0=bad_eps0, log=log)
            assert log.has_errors
            assert any("reference strain rate EPS0 must be > 0" in e and "ANCMSG 910" in e for e in log.errors)

    def test_beta_out_of_bounds_ancmsg_911(self):
        """Bulking coefficient beta < 0 or beta > 1 triggers ANCMSG 911 error."""
        for bad_beta in (-0.05, 1.25):
            log = MessageLog()
            m = MatLaw79(id=6, rho=2.5, tau_shear=1.0e10, k1=2.0e10, beta=bad_beta)
            check_mat_law79(mat=m, log=log)
            assert log.has_errors
            assert any("bulking coefficient BETA must satisfy 0 <= BETA <= 1" in e and "ANCMSG 911" in e for e in log.errors)

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

        mat79 = MatLaw79(id=79, rho=2.5, tau_shear=1.0e10, k1=2.0e10)

        # 1. Solids accepted
        for s_type in ("solids", "bricks", "tetras", "penta6", "pyra5"):
            model_s = DummyModel([(s_type, DummyGrp([DummyEl(79)]))])
            log_s = MessageLog()
            check_mat_law79(model=model_s, mat=mat79, log=log_s)
            assert not log_s.has_errors, f"Solid element {s_type} should be accepted"

        # 2. Shells and quads rejected (ANCMSG 305)
        for sh_type in ("shells", "shells_qbat", "shells_qeph", "sh3n", "quads"):
            model_sh = DummyModel([(sh_type, DummyGrp([DummyEl(79)]))])
            log_sh = MessageLog()
            check_mat_law79(model=model_sh, mat=mat79, log=log_sh)
            assert log_sh.has_errors, f"Shell element {sh_type} should be rejected"
            assert any("shell elements" in e and "ANCMSG 305" in e for e in log_sh.errors)

        # 3. 1D elements rejected (ANCMSG 306)
        for d1_type in ("trusses", "beams", "springs"):
            model_1d = DummyModel([(d1_type, DummyGrp([DummyEl(79)]))])
            log_1d = MessageLog()
            check_mat_law79(model=model_1d, mat=mat79, log=log_1d)
            assert log_1d.has_errors, f"1D element {d1_type} should be rejected"
            assert any("1D elements" in e and "ANCMSG 306" in e for e in log_1d.errors)

        # 4. 2D analysis rejected (N2D > 0, ANCMSG 305)
        model_2d = DummyModel([("solids", DummyGrp([DummyEl(79)]))], n2d=1)
        log_2d = MessageLog()
        check_mat_law79(model=model_2d, mat=mat79, log=log_2d)
        assert log_2d.has_errors
        assert any("N2D > 0" in e and "ANCMSG 305" in e for e in log_2d.errors)

    def test_checks_dispatch_and_allowed_laws_registry(self):
        """Verify _MAT_CHECKS registry and _ALLOWED_LAWS table for all LAW79 synonyms."""
        for syn in (79, "79", "LAW79", "JOHN_HOLM", "JOHNSON_HOLMQUIST", "JH2", "MAT_LAW79", "MAT_JOHN_HOLM", "LAW79_JOHN_HOLM"):
            assert syn in _MAT_CHECKS, f"{syn} missing from _MAT_CHECKS"
            assert _MAT_CHECKS[syn] is check_mat_law79

        # Solids permit LAW79
        for fam in ("bricks", "tetras", "penta6", "pyra5"):
            assert 79 in _ALLOWED_LAWS[fam]
            assert "LAW79" in _ALLOWED_LAWS[fam]
            assert "JOHN_HOLM" in _ALLOWED_LAWS[fam]
            assert "JH2" in _ALLOWED_LAWS[fam]

        # Shells and 1D forbid LAW79
        for fam in ("shells", "shells_qbat", "shells_qeph", "sh3n", "quads", "trusses", "beams"):
            assert 79 not in _ALLOWED_LAWS[fam]
            assert "LAW79" not in _ALLOWED_LAWS[fam]
            assert "JOHN_HOLM" not in _ALLOWED_LAWS[fam]


# ============================================================================
# 5. Boundary Values & Defaults
# ============================================================================

class TestLaw79BoundaryValues:
    """Audit boundary values and default parameter assignments."""

    def test_omitted_defaults_application(self, tmp_path: Path):
        """Verify eps0=1.0, sigfmax=1e20, epsmax=1e20, refer_rho=rho, d2=1.0, beta=1.0 when omitted."""
        deck = StarterDeck("OMITTED_DEFAULTS_LAW79")
        deck.mat_law79(
            mat_id=1,
            rho=2.8,
            tau_shear=5.0e10,
            k1=8.0e10,
        )

        rad_path = tmp_path / "omitted_defaults_0000.rad"
        deck.write(str(rad_path))
        model = read_starter_deck(str(rad_path))

        m = model.mat_law79s[1]
        mat = model.materials[1]
        assert m.refer_rho == pytest.approx(2.8)
        assert m.eps0 == pytest.approx(1.0)
        assert m.sigfmax == pytest.approx(1.0e20)
        assert m.epsmax == pytest.approx(1.0e20)
        assert m.d2 == pytest.approx(1.0)
        assert m.beta == pytest.approx(1.0)
        assert m.idel == 0

        assert mat.params["refer_rho"] == pytest.approx(2.8)
        assert mat.params["eps0"] == pytest.approx(1.0)
        assert mat.params["sigfmax"] == pytest.approx(1.0e20)
        assert mat.params["epsmax"] == pytest.approx(1.0e20)

    def test_idel_clamping(self):
        """Verify element deletion flag IDEL is clamped to [0, 3]."""
        for raw_idel, expected in [(-1, 0), (0, 0), (1, 1), (2, 2), (3, 3), (4, 3), (99, 3)]:
            m = MatLaw79(id=1, rho=2.5, tau_shear=1e10, k1=2e10, idel=raw_idel)
            assert m.idel == expected

    def test_phel_equals_hel_valid_boundary(self):
        """Verify phel == hel is a valid boundary condition resulting in shel = 0.0."""
        log = MessageLog()
        m = MatLaw79(id=1, rho=2.5, tau_shear=1.0e10, k1=2.0e10, hel=2.0e9, phel=2.0e9)
        check_mat_law79(mat=m, log=log)
        assert not log.has_errors
        assert m.shel == pytest.approx(0.0)

    def test_beta_boundary_values(self):
        """Verify beta = 0.0 and beta = 1.0 are valid boundary values."""
        for b_val in (0.0, 1.0):
            log = MessageLog()
            m = MatLaw79(id=1, rho=2.5, tau_shear=1.0e10, k1=2.0e10, beta=b_val)
            check_mat_law79(mat=m, log=log)
            assert not log.has_errors
            assert m.beta == pytest.approx(b_val)

    def test_derived_elastic_properties(self):
        """Verify Young's modulus, Poisson's ratio, and sound speed matching formulas."""
        k1 = 5.0e10
        g = 3.0e10
        rho = 2500.0

        m = MatLaw79(id=1, rho=rho, tau_shear=g, k1=k1)
        expected_e = (9.0 * k1 * g) / (3.0 * k1 + g)
        expected_nu = (3.0 * k1 - 2.0 * g) / (6.0 * k1 + 2.0 * g)
        expected_c = math.sqrt((k1 + (4.0 / 3.0) * g) / rho)

        assert m.young == pytest.approx(expected_e)
        assert m.E == pytest.approx(expected_e)
        assert m.nu == pytest.approx(expected_nu)
        assert m.sound_speed == pytest.approx(expected_c)
        assert m.sound_speed_solid == pytest.approx(expected_c)


# ============================================================================
# 6. Restart (.rst) Serialization
# ============================================================================

class TestLaw79RestartSerialization:
    """Audit restart state serialization and dynamic cycle continuation for LAW79."""

    def test_mat_law79_dataclass_pickle_fidelity(self):
        """Pickle and unpickle MatLaw79, verifying all fields and derived properties."""
        mat = MatLaw79(
            id=79,
            rho=2.53,
            refer_rho=2.50,
            tau_shear=3.04e10,
            a=0.93,
            b=0.70,
            m=0.60,
            n=0.64,
            c=0.003,
            eps0=1.0,
            sigfmax=1.50e9,
            fcut=1000.0,
            t=2.0e8,
            hel=2.10e9,
            phel=1.20e9,
            d1=0.05,
            d2=0.85,
            idel=1,
            epsmax=0.45,
            k1=5.20e10,
            k2=-3.50e10,
            k3=1.20e11,
            beta=0.80,
            title="Boron-Carbide-Pickle",
        )

        data = pickle.dumps(mat, protocol=pickle.HIGHEST_PROTOCOL)
        restored: MatLaw79 = pickle.loads(data)

        _assert_law79_all_fields_exact(
            restored,
            rho=2.53,
            refer_rho=2.50,
            tau_shear=3.04e10,
            a=0.93,
            b=0.70,
            m_exp=0.60,
            n_exp=0.64,
            c=0.003,
            eps0=1.0,
            sigfmax=1.50e9,
            fcut=1000.0,
            t=2.0e8,
            hel=2.10e9,
            phel=1.20e9,
            d1=0.05,
            d2=0.85,
            idel=1,
            epsmax=0.45,
            k1=5.20e10,
            k2=-3.50e10,
            k3=1.20e11,
            beta=0.80,
            title="Boron-Carbide-Pickle",
        )

    def test_model_with_law79_pickle_fidelity(self):
        """Pickle and unpickle full Model with LAW79 material and check consistency."""
        model = Model()
        m79 = MatLaw79(
            id=10,
            rho=2.5,
            refer_rho=2.5,
            tau_shear=1.0e10,
            a=0.5,
            b=0.4,
            m=0.6,
            n=0.64,
            c=0.003,
            eps0=1.0,
            sigfmax=1.0e9,
            fcut=0.0,
            t=2.0e8,
            hel=1.5e9,
            phel=1.0e9,
            d1=0.05,
            d2=0.85,
            idel=1,
            epsmax=0.5,
            k1=2.0e10,
            k2=1.0e10,
            k3=5.0e9,
            beta=0.8,
            title="Model Pickled Law79",
        )
        model.mat_law79s[10] = m79
        model.materials[10] = Material(id=10, law=79, rho0=2.5, title="Model Pickled Law79", params=m79.params)

        data = pickle.dumps(model, protocol=pickle.HIGHEST_PROTOCOL)
        restored_model: Model = pickle.loads(data)

        assert 10 in restored_model.mat_law79s
        assert 10 in restored_model.materials
        _assert_law79_all_fields_exact(
            restored_model.mat_law79s[10],
            restored_model.materials[10],
            rho=2.5,
            refer_rho=2.5,
            tau_shear=1.0e10,
            a=0.5,
            b=0.4,
            m_exp=0.6,
            n_exp=0.64,
            c=0.003,
            eps0=1.0,
            sigfmax=1.0e9,
            fcut=0.0,
            t=2.0e8,
            hel=1.5e9,
            phel=1.0e9,
            d1=0.05,
            d2=0.85,
            idel=1,
            epsmax=0.5,
            k1=2.0e10,
            k2=1.0e10,
            k3=5.0e9,
            beta=0.8,
            title="Model Pickled Law79",
        )

    def test_material_state_arrays_rst_preservation(self, tmp_path: Path):
        """Verify persistent material state arrays (deltap, sigy_old, dmg, off) survive write_restart and read_restart."""
        deck = StarterDeck("RST_LAW79_ARRAYS")
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
        deck.part(1, "CeramicPart", prop_id=1, mat_id=1)
        deck.mat_law79(
            mat_id=1,
            title="Ceramic JH2",
            rho=2500.0,
            tau_shear=3.0e10,
            k1=5.0e10,
            a=0.9,
            b=0.7,
            hel=2.0e9,
            phel=1.2e9,
        )

        rad_path = tmp_path / "rst_law79_arrays_0000.rad"
        deck.write(str(rad_path))

        model = run_starter(str(rad_path))
        assert model is not None

        groups = dict(model.element_groups())
        bg = groups.get("bricks") or groups.get("solids")
        st = bg.state

        # Synthesize realistic persistent material state arrays for LAW79 (2 elements)
        deltap_synth = np.array([1.25e7, 4.50e7], dtype=float)
        sigy_old_synth = np.array([0.88, 0.65], dtype=float)
        dmg_synth = np.array([0.15, 0.72], dtype=float)
        off_synth = np.array([1.0, 0.0], dtype=float)  # 1 alive, 1 eroded
        sig_synth = np.array([
            [1.5e8, 1.2e8, 1.3e8, 4.5e7, 2.1e7, 1.8e7],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        ], dtype=float)
        epsp_synth = np.array([0.0055, 0.0210], dtype=float)

        st["mat_extra"]["deltap"] = deltap_synth.copy()
        st["mat_extra"]["sigy_old"] = sigy_old_synth.copy()
        st["mat_extra"]["dmg"] = dmg_synth.copy()
        st["off"] = off_synth.copy()
        st["sig"] = sig_synth.copy()
        st["epsp"] = epsp_synth.copy()

        # Write restart .rst
        rst_path = tmp_path / "rst_law79_0001.rst"
        engine_dict = {
            "cycle": 1500,
            "t": 3.75e-5,
            "dt": 1.2e-8,
            "energies": {"internal": 85000.0, "kinetic": 23000.0},
        }
        write_restart(model, str(rst_path), engine=engine_dict)

        # Read back from restart .rst
        rest_model, rest_engine = read_restart(str(rst_path))
        assert rest_engine["cycle"] == 1500
        assert rest_engine["t"] == pytest.approx(3.75e-5)
        assert rest_engine["dt"] == pytest.approx(1.2e-8)

        rest_bg = dict(rest_model.element_groups()).get("bricks") or dict(rest_model.element_groups()).get("solids")
        rest_extra = rest_bg.state["mat_extra"]
        rest_st = rest_bg.state

        # Assert exact preservation of LAW79 state arrays
        np.testing.assert_array_equal(rest_extra["deltap"], deltap_synth)
        np.testing.assert_array_equal(rest_extra["sigy_old"], sigy_old_synth)
        np.testing.assert_array_equal(rest_extra["dmg"], dmg_synth)
        np.testing.assert_array_equal(rest_st["off"], off_synth)
        np.testing.assert_array_equal(rest_st["sig"], sig_synth)
        np.testing.assert_array_equal(rest_st["epsp"], epsp_synth)

    def test_constitutive_dynamic_cycle_continuation_from_restart(self, tmp_path: Path):
        """Constitutive stress update continuation from restart matches uninterrupted simulation within 10^-12."""
        mat = build_law79({
            "id": 1,
            "rho": 2500.0,
            "shear": 3.0e10,
            "k1": 5.0e10,
            "a": 0.90,
            "b": 0.70,
            "m": 0.60,
            "n": 0.64,
            "c": 0.003,
            "eps0": 1.0,
            "sigfmax": 1.5e9,
            "fcut": 1000.0,
            "t": 2.0e8,
            "hel": 2.0e9,
            "phel": 1.2e9,
            "d1": 0.05,
            "d2": 0.85,
            "beta": 1.0,
            "idel": 0,
        })

        sig_uninterrupted = np.zeros((1, 6))
        epsp_uninterrupted = np.zeros(1)
        extra_uninterrupted = {
            "deltap": np.zeros(1),
            "sigy_old": np.full(1, 0.90),
            "dmg": np.zeros(1),
            "off": np.ones(1),
            "mu": np.zeros(1),
        }

        dt = 1.0e-7
        deps_steps = [
            np.array([[0.0005, -0.00015, -0.00015, 0.0002, 0.0, 0.0]]),
            np.array([[0.0010, -0.00030, -0.00030, 0.0004, 0.0, 0.0]]),
            np.array([[0.0015, -0.00045, -0.00045, 0.0006, 0.0, 0.0]]),
            np.array([[0.0020, -0.00060, -0.00060, 0.0008, 0.0, 0.0]]),
        ]

        # Run 4 steps uninterrupted
        for deps in deps_steps:
            sig_uninterrupted, epsp_uninterrupted, _ = solid_update_law79(
                mat,
                sig_uninterrupted,
                deps=deps,
                epsp=epsp_uninterrupted,
                dt=dt,
                extra=extra_uninterrupted,
                return_tuple=True,
            )

        # Now run 2 steps, snapshot restart state, deserialize and run next 2 steps
        sig_restarted = np.zeros((1, 6))
        epsp_restarted = np.zeros(1)
        extra_restarted = {
            "deltap": np.zeros(1),
            "sigy_old": np.full(1, 0.90),
            "dmg": np.zeros(1),
            "off": np.ones(1),
            "mu": np.zeros(1),
        }

        for deps in deps_steps[:2]:
            sig_restarted, epsp_restarted, _ = solid_update_law79(
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
        restored_snap = pickle.loads(snap)
        sig_resumed = restored_snap["sig"]
        epsp_resumed = restored_snap["epsp"]
        extra_resumed = restored_snap["extra"]

        # Continue remaining 2 steps from restart
        for deps in deps_steps[2:]:
            sig_resumed, epsp_resumed, _ = solid_update_law79(
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
        np.testing.assert_allclose(extra_resumed["dmg"], extra_uninterrupted["dmg"], rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(extra_resumed["deltap"], extra_uninterrupted["deltap"], rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(extra_resumed["sigy_old"], extra_uninterrupted["sigy_old"], rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(extra_resumed["off"], extra_uninterrupted["off"], rtol=1e-12, atol=1e-12)
