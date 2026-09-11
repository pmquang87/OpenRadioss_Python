"""
Milestone M557: /MAT/LAW49 (/MAT/STEINB, /MAT/STEINBERG, /MAT/STEINBERG_GUINAN)
Exhaustive Roundtrip, Negative Validation, Boundary Cases & Restart (.rst) Serialization Auditor Suite.

Fortran origins:
  - starter/source/materials/mat/mat049/hm_read_mat49.F
  - engine/source/materials/mat/mat049/m49law.F
  - engine/source/output/restart/wrrestp.F & rdresb.F
  - starter/source/restart/ddsplit/wrrest.F
  - config/CFG/radioss110/MAT/matl49_steinb.cfg & radioss2020/MAT/matl49_steinb.cfg

Audits:
  1. Fixed-Format Deck Roundtrip:
     - Standard 20-column fixed format with StarterDeck.mat_law49.
     - Card 1: RHO_I, [RHO_O] (MAT_LAW49_1: [20, 20])
     - Card 2: E0, nu (MAT_LAW49_2: [20, 20])
     - Card 3: sigma_0, beta, n, EPS_max, SIGMA_max (MAT_LAW49_3: [20, 20, 20, 20, 20])
     - Card 4: T_0, Tmelt, rhoC_p, Pmin (MAT_LAW49_4: [20, 20, 20, 20])
     - Card 5: b1, b2, h, f (MAT_LAW49_5: [20, 20, 20, 20])
     - Re-parse using read_starter_deck / read_mat_law49.
     - Assert exact equality for all parameters in model.mat_law49s and model.materials.
     - Minimal card and default fallback handling.
     - Entity object invocation with MatLaw49 and Material instances.

  2. Free-Format Comma-Delimited Roundtrip:
     - Write and re-parse free-format deck blocks for /MAT/LAW49 and synonyms.
     - Standard comma-separated with whitespace and compact comma-delimited without whitespace.
     - Cross-dialect roundtrip (free -> parse -> fixed -> parse -> exact equality).

  3. Keyword Synonyms:
     - /MAT/LAW49, /MAT/STEINB, /MAT/STEINBERG, /MAT/STEINBERG_GUINAN.

  4. Negative Starter Diagnostics:
     - Density rho <= 0 (error).
     - Young's modulus e0 <= 0 (error).
     - Poisson's ratio nu < 0 or nu >= 0.5 (ANCMSG 1514 error).
     - Melting parameter f < 0 (ANCMSG 1513 error).
     - Incompatible element types: shells/quads (ANCMSG 305), 1D elements (ANCMSG 306), 2D analysis (ANCMSG 305).
     - Solid elements accepted (solids, bricks, tetras, penta6, pyra5).
     - Registry and dispatch validation (_ALLOWED_LAWS, _MAT_CHECKS, check_materials).

  5. Boundary Values & Melt Softening:
     - Default values when omitted (t0=300, eps_max=1e20, sigma_max=1e20, tmelt=1e20, pmin=-1e20).
     - Melt factor behavior when espe approaches Emelt and exceeds Emelt.

  6. Restart (.rst) Serialization:
     - MatLaw49 entity dataclass pickling and unpickling fidelity.
     - Model with law 49 serialization.
     - Material state arrays: theta, espe, epxe, dpla, sig, epsp, off.
     - Material state arrays preserved across write_restart / read_restart contract.
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
    read_mat_law49,
    read_starter_deck,
)
from pyradioss.materials.law49_steinb import (
    Law49Params,
    build_law49,
    solid_update_law49,
    sound_speed_solid,
)
from pyradioss import materials
from pyradioss.model.entities import (
    MatLaw49,
    MatSteinb,
    MatSteinberg,
    MatSteinbergGuinan,
    Material,
)
from pyradioss.model.model import Model
from pyradioss.starter.checks import (
    _ALLOWED_LAWS,
    _MAT_CHECKS,
    check_mat_law49,
    check_materials,
    check_model,
)
from pyradioss.starter.restart import read_restart, write_restart
from pyradioss.starter.starter import run_starter


# ============================================================================
# Helpers
# ============================================================================

def _parse_deck_str(tmp_path: Path, text: str, name: str = "TEST_LAW49") -> Tuple[Model, MessageLog]:
    """Write deck text to file and parse into Model and MessageLog."""
    deck_path = tmp_path / f"{name}_0000.rad"
    deck_path.write_text(text.strip() + "\n", encoding="utf-8")
    blocks = read_deck(str(deck_path))
    model = Model()
    log = MessageLog()
    parse_starter_deck(blocks, model, log)
    return model, log


def _assert_law49_all_fields_exact(
    m: MatLaw49,
    mat: Material | None = None,
    *,
    rho: float,
    refer_rho: float,
    e0: float,
    nu: float,
    sigy: float,
    beta: float,
    n: float,
    eps_max: float,
    sigma_max: float,
    t0: float,
    tmelt: float,
    rhoc_p: float,
    pmin: float,
    b1: float,
    b2: float,
    h: float,
    f: float,
    title: str = "",
) -> None:
    """Assert exact (floating-point tolerance) equality for all LAW49 parameters."""
    assert m.rho == pytest.approx(rho, rel=1e-6, abs=1e-12)
    assert m.refer_rho == pytest.approx(refer_rho, rel=1e-6, abs=1e-12)
    assert m.rho0 == pytest.approx(rho, rel=1e-6, abs=1e-12)
    assert m.rhor == pytest.approx(refer_rho, rel=1e-6, abs=1e-12)

    assert m.e0 == pytest.approx(e0, rel=1e-6, abs=1e-12)
    assert m.e == pytest.approx(e0, rel=1e-6, abs=1e-12)
    assert m.E == pytest.approx(e0, rel=1e-6, abs=1e-12)
    assert m.nu == pytest.approx(nu, rel=1e-6, abs=1e-12)
    assert m.Nu == pytest.approx(nu, rel=1e-6, abs=1e-12)

    assert m.sigy == pytest.approx(sigy, rel=1e-6, abs=1e-12)
    assert m.sigma_0 == pytest.approx(sigy, rel=1e-6, abs=1e-12)
    assert m.beta == pytest.approx(beta, rel=1e-6, abs=1e-12)
    assert m.n == pytest.approx(n, rel=1e-6, abs=1e-12)
    assert m.hard == pytest.approx(n, rel=1e-6, abs=1e-12)

    assert m.eps_max == pytest.approx(eps_max, rel=1e-6, abs=1e-12)
    assert m.sigma_max == pytest.approx(sigma_max, rel=1e-6, abs=1e-12)

    assert m.t0 == pytest.approx(t0, rel=1e-6, abs=1e-12)
    assert m.tmelt == pytest.approx(tmelt, rel=1e-6, abs=1e-12)
    assert m.rhoc_p == pytest.approx(rhoc_p, rel=1e-6, abs=1e-12)
    assert m.pmin == pytest.approx(pmin, rel=1e-6, abs=1e-12)

    assert m.b1 == pytest.approx(b1, rel=1e-6, abs=1e-12)
    assert m.b2 == pytest.approx(b2, rel=1e-6, abs=1e-12)
    assert m.h == pytest.approx(h, rel=1e-6, abs=1e-12)
    assert m.f == pytest.approx(f, rel=1e-6, abs=1e-12)

    if title:
        assert m.title == title

    # Verify derived elastic properties
    expected_G0 = e0 / (2.0 * (1.0 + nu))
    expected_bulk = e0 / (3.0 * (1.0 - 2.0 * nu))
    assert m.G == pytest.approx(expected_G0, rel=1e-6, abs=1e-12)
    assert m.G0 == pytest.approx(expected_G0, rel=1e-6, abs=1e-12)
    assert m.g0 == pytest.approx(expected_G0, rel=1e-6, abs=1e-12)
    assert m.bulk == pytest.approx(expected_bulk, rel=1e-6, abs=1e-12)
    assert m.K == pytest.approx(expected_bulk, rel=1e-6, abs=1e-12)
    assert m.C1 == pytest.approx(expected_bulk, rel=1e-6, abs=1e-12)

    # Verify model.materials[mid] parameters if provided
    if mat is not None:
        assert mat.law == 49
        assert mat.rho0 == pytest.approx(rho, rel=1e-6, abs=1e-12)
        p = mat.params
        assert p["rho"] == pytest.approx(rho, rel=1e-6, abs=1e-12)
        assert p["rho0"] == pytest.approx(rho, rel=1e-6, abs=1e-12)
        assert p["refer_rho"] == pytest.approx(refer_rho, rel=1e-6, abs=1e-12)
        assert p["rhor"] == pytest.approx(refer_rho, rel=1e-6, abs=1e-12)
        assert p["e0"] == pytest.approx(e0, rel=1e-6, abs=1e-12)
        assert p["e"] == pytest.approx(e0, rel=1e-6, abs=1e-12)
        assert p["E"] == pytest.approx(e0, rel=1e-6, abs=1e-12)
        assert p["nu"] == pytest.approx(nu, rel=1e-6, abs=1e-12)
        assert p["Nu"] == pytest.approx(nu, rel=1e-6, abs=1e-12)
        assert p["sigy"] == pytest.approx(sigy, rel=1e-6, abs=1e-12)
        assert p["sig0"] == pytest.approx(sigy, rel=1e-6, abs=1e-12)
        assert p["sigma_0"] == pytest.approx(sigy, rel=1e-6, abs=1e-12)
        assert p["beta"] == pytest.approx(beta, rel=1e-6, abs=1e-12)
        assert p["n"] == pytest.approx(n, rel=1e-6, abs=1e-12)
        assert p["hard"] == pytest.approx(n, rel=1e-6, abs=1e-12)
        assert p["eps_max"] == pytest.approx(eps_max, rel=1e-6, abs=1e-12)
        assert p["sigma_max"] == pytest.approx(sigma_max, rel=1e-6, abs=1e-12)
        assert p["t0"] == pytest.approx(t0, rel=1e-6, abs=1e-12)
        assert p["tmelt"] == pytest.approx(tmelt, rel=1e-6, abs=1e-12)
        assert p["rhoc_p"] == pytest.approx(rhoc_p, rel=1e-6, abs=1e-12)
        assert p["pmin"] == pytest.approx(pmin, rel=1e-6, abs=1e-12)
        assert p["b1"] == pytest.approx(b1, rel=1e-6, abs=1e-12)
        assert p["b2"] == pytest.approx(b2, rel=1e-6, abs=1e-12)
        assert p["h"] == pytest.approx(h, rel=1e-6, abs=1e-12)
        assert p["f"] == pytest.approx(f, rel=1e-6, abs=1e-12)
        assert p["G"] == pytest.approx(expected_G0, rel=1e-6, abs=1e-12)
        assert p["G0"] == pytest.approx(expected_G0, rel=1e-6, abs=1e-12)
        assert p["bulk"] == pytest.approx(expected_bulk, rel=1e-6, abs=1e-12)
        assert p["C1"] == pytest.approx(expected_bulk, rel=1e-6, abs=1e-12)


# ============================================================================
# 1. Fixed-Format Deck Roundtrip
# ============================================================================

class TestLaw49FixedFormatRoundtrip:
    """Audit StarterDeck.mat_law49 20-column fixed-format emission and re-reading."""

    def test_fixed_format_card_columns_and_roundtrip(self, tmp_path: Path):
        """Verify exact 20-column layout of all 5 data cards and complete roundtrip."""
        vals = {
            "rho": 8.96,
            "refer_rho": 8.90,
            "e0": 1.15e11,
            "nu": 0.34,
            "sigy": 1.20e8,
            "beta": 36.0,
            "n": 0.45,
            "eps_max": 2.5,
            "sigma_max": 6.40e8,
            "t0": 293.15,
            "tmelt": 1356.0,
            "rhoc_p": 3.45e3,
            "pmin": -5.0e8,
            "b1": 2.8e-11,
            "b2": 2.8e-11,
            "h": 3.8e-4,
            "f": 0.15,
        }

        deck = StarterDeck("AUDIT_FIXED_LAW49")
        deck.mat_law49(
            mat_id=49,
            title="Copper-OFHC Shock",
            fixed_format=True,
            sig0=vals["sigy"],
            **vals,
        )

        rendered = deck.render()
        lines = [line.rstrip() for line in rendered.splitlines()]

        # Locate keyword block
        idx = lines.index("/MAT/LAW49/49")
        assert lines[idx + 1] == "Copper-OFHC Shock"

        # Card 1: RHO_I, RHO_O (each 20 cols)
        # Note: line[idx + 2] is comment "#        Init. dens.          Ref. dens."
        c1_line = lines[idx + 3]
        assert len(c1_line) == 40
        f1 = split_fixed(c1_line, LAYOUTS["MAT_LAW49_1"])
        assert float(f1[0]) == pytest.approx(vals["rho"])
        assert float(f1[1]) == pytest.approx(vals["refer_rho"])

        # Card 2: E0, Nu (each 20 cols)
        c2_line = lines[idx + 5]
        assert len(c2_line) == 40
        f2 = split_fixed(c2_line, LAYOUTS["MAT_LAW49_2"])
        assert float(f2[0]) == pytest.approx(vals["e0"])
        assert float(f2[1]) == pytest.approx(vals["nu"])

        # Card 3: Sigma0, Beta, N, EPS_max, SIGMA_max (each 20 cols -> 100 cols)
        c3_line = lines[idx + 7]
        assert len(c3_line) == 100
        f3 = split_fixed(c3_line, LAYOUTS["MAT_LAW49_3"])
        assert float(f3[0]) == pytest.approx(vals["sigy"])
        assert float(f3[1]) == pytest.approx(vals["beta"])
        assert float(f3[2]) == pytest.approx(vals["n"])
        assert float(f3[3]) == pytest.approx(vals["eps_max"])
        assert float(f3[4]) == pytest.approx(vals["sigma_max"])

        # Card 4: T_0, Tmelt, rhoC_p, Pmin (each 20 cols -> 80 cols)
        c4_line = lines[idx + 9]
        assert len(c4_line) == 80
        f4 = split_fixed(c4_line, LAYOUTS["MAT_LAW49_4"])
        assert float(f4[0]) == pytest.approx(vals["t0"])
        assert float(f4[1]) == pytest.approx(vals["tmelt"])
        assert float(f4[2]) == pytest.approx(vals["rhoc_p"])
        assert float(f4[3]) == pytest.approx(vals["pmin"])

        # Card 5: b1, b2, h, f (each 20 cols -> 80 cols)
        c5_line = lines[idx + 11]
        assert len(c5_line) == 80
        f5 = split_fixed(c5_line, LAYOUTS["MAT_LAW49_5"])
        assert float(f5[0]) == pytest.approx(vals["b1"])
        assert float(f5[1]) == pytest.approx(vals["b2"])
        assert float(f5[2]) == pytest.approx(vals["h"])
        assert float(f5[3]) == pytest.approx(vals["f"])

        # Roundtrip parse with read_starter_deck
        rad_path = tmp_path / "law49_fixed_0000.rad"
        rad_path.write_text(rendered + "\n/END\n", encoding="utf-8")

        model = read_starter_deck(str(rad_path))
        assert 49 in model.mat_law49s
        assert 49 in model.materials

        m = model.mat_law49s[49]
        mat = model.materials[49]
        _assert_law49_all_fields_exact(m, mat, title="Copper-OFHC Shock", **vals)

    def test_fixed_format_defaults_fallback(self, tmp_path: Path):
        """Minimal fixed card input inherits exact Fortran defaults."""
        deck = StarterDeck("AUDIT_MINIMAL_LAW49")
        deck.mat_law49(
            mat_id=1,
            title="Minimal",
            rho=7.85,
            e0=2.1e11,
            nu=0.28,
            sig0=4.0e8,
        )

        rendered = deck.render()
        rad_path = tmp_path / "law49_minimal_0000.rad"
        rad_path.write_text(rendered + "\n/END\n", encoding="utf-8")

        model, log = read_starter_deck(str(rad_path))
        assert not log.has_errors
        assert 1 in model.mat_law49s
        assert 1 in model.materials

        m = model.mat_law49s[1]
        mat = model.materials[1]
        _assert_law49_all_fields_exact(
            m,
            mat,
            rho=7.85,
            refer_rho=7.85,    # Defaults to rho
            e0=2.1e11,
            nu=0.28,
            sigy=4.0e8,
            beta=0.0,
            n=0.0,
            eps_max=1.0e20,    # Default per hm_read_mat49.F
            sigma_max=1.0e20,  # Default per hm_read_mat49.F
            t0=300.0,          # Default per hm_read_mat49.F
            tmelt=1.0e20,      # Default per hm_read_mat49.F
            rhoc_p=0.0,
            pmin=-1.0e20,      # Default per hm_read_mat49.F
            b1=0.0,
            b2=0.0,
            h=0.0,
            f=0.0,
            title="Minimal",
        )

    def test_fixed_format_mat_object_direct_input(self, tmp_path: Path):
        """Passing MatLaw49 or Material entity object into StarterDeck.mat_law49 roundtrips identically."""
        original = MatLaw49(
            id=77,
            title="Entity Injected",
            rho=8.96,
            refer_rho=8.96,
            e0=1.24e11,
            nu=0.34,
            sigy=1.2e8,
            beta=36.0,
            n=0.45,
            eps_max=1.0e20,
            sigma_max=6.4e8,
            t0=300.0,
            tmelt=1356.0,
            rhoc_p=3.45e3,
            pmin=-1.0e20,
            b1=2.8e-11,
            b2=2.8e-11,
            h=3.8e-4,
            f=0.1,
        )

        deck = StarterDeck("AUDIT_ENTITY_LAW49")
        deck.mat_law49(original, fixed_format=True)

        rendered = deck.render()
        rad_path = tmp_path / "entity_injected_0000.rad"
        rad_path.write_text(rendered + "\n/END\n", encoding="utf-8")

        model = read_starter_deck(str(rad_path))
        assert 77 in model.mat_law49s
        m = model.mat_law49s[77]
        mat = model.materials[77]

        _assert_law49_all_fields_exact(
            m,
            mat,
            rho=8.96,
            refer_rho=8.96,
            e0=1.24e11,
            nu=0.34,
            sigy=1.2e8,
            beta=36.0,
            n=0.45,
            eps_max=1.0e20,
            sigma_max=6.4e8,
            t0=300.0,
            tmelt=1356.0,
            rhoc_p=3.45e3,
            pmin=-1.0e20,
            b1=2.8e-11,
            b2=2.8e-11,
            h=3.8e-4,
            f=0.1,
            title="Entity Injected",
        )

    def test_fixed_format_material_class_input(self, tmp_path: Path):
        """Passing Material dataclass produced by build_law49 into StarterDeck.mat_law49."""
        mat_in = build_law49({
            "id": 88,
            "title": "Built Material",
            "rho": 7.89,
            "e0": 2.05e11,
            "nu": 0.29,
            "sig0": 3.5e8,
            "beta": 15.0,
            "n": 0.35,
            "rhoc_p": 4.5e3,
            "tmelt": 1800.0,
            "b1": 1.5e-11,
            "b2": 1.5e-11,
            "h": 2.0e-4,
            "f": 0.05,
        })

        deck = StarterDeck("AUDIT_MAT_BUILT")
        deck.mat_law49(mat_in, fixed_format=True)

        rad_path = tmp_path / "built_material_0000.rad"
        rad_path.write_text(deck.render() + "\n/END\n", encoding="utf-8")

        model = read_starter_deck(str(rad_path))
        assert 88 in model.mat_law49s
        m = model.mat_law49s[88]
        mat = model.materials[88]

        _assert_law49_all_fields_exact(
            m,
            mat,
            rho=7.89,
            refer_rho=7.89,
            e0=2.05e11,
            nu=0.29,
            sigy=3.5e8,
            beta=15.0,
            n=0.35,
            eps_max=1.0e20,
            sigma_max=1.0e20,
            t0=300.0,
            tmelt=1800.0,
            rhoc_p=4.5e3,
            pmin=-1.0e20,
            b1=1.5e-11,
            b2=1.5e-11,
            h=2.0e-4,
            f=0.05,
            title="Built Material",
        )


# ============================================================================
# 2. Free-Format Comma-Delimited Roundtrip
# ============================================================================

class TestLaw49FreeFormatRoundtrip:
    """Audit free-format comma-delimited parsing, emission, and cross-dialect parity."""

    def test_free_format_comma_delimited(self, tmp_path: Path):
        """Write /MAT/LAW49 with comma-separated free format and verify exact parse."""
        vals = {
            "rho": 8.96,
            "refer_rho": 8.92,
            "e0": 1.15e11,
            "nu": 0.34,
            "sigy": 1.2e8,
            "beta": 36.0,
            "n": 0.45,
            "eps_max": 2.0,
            "sigma_max": 6.4e8,
            "t0": 300.0,
            "tmelt": 1356.0,
            "rhoc_p": 3.45e3,
            "pmin": -1.0e20,
            "b1": 2.8e-11,
            "b2": 2.8e-11,
            "h": 3.8e-4,
            "f": 0.1,
        }

        deck = StarterDeck("AUDIT_FREE_LAW49")
        deck.mat_law49(
            mat_id=49,
            title="Free Format Copper",
            fixed_format=False,
            comma=True,
            sig0=vals["sigy"],
            **vals,
        )

        rendered = deck.render()
        assert "/MAT/LAW49/49" in rendered
        assert "," in rendered

        rad_path = tmp_path / "free_law49_0000.rad"
        rad_path.write_text(rendered + "\n/END\n", encoding="utf-8")

        model, log = read_starter_deck(str(rad_path))
        assert not log.has_errors
        assert 49 in model.mat_law49s
        m = model.mat_law49s[49]
        mat = model.materials[49]

        _assert_law49_all_fields_exact(m, mat, title="Free Format Copper", **vals)

    def test_compact_comma_delimited_without_spaces(self, tmp_path: Path):
        """Parse compact comma-delimited cards with no whitespace surrounding delimiters."""
        deck_text = """\
#RADIOSS STARTER
/BEGIN
compact_comma_law49
                  90                   1
/MAT/LAW49/33
Compact No Spaces
8.96,8.96
115000000000.0,0.34
120000000.0,36.0,0.45,2.0,640000000.0
300.0,1356.0,3450.0,-1.0e20
2.8e-11,2.8e-11,0.00038,0.1
/END
"""
        rad_path = tmp_path / "compact_no_space_0000.rad"
        rad_path.write_text(deck_text, encoding="utf-8")

        model, log = read_starter_deck(str(rad_path))
        assert not log.has_errors
        assert 33 in model.mat_law49s
        m = model.mat_law49s[33]
        mat = model.materials[33]

        _assert_law49_all_fields_exact(
            m,
            mat,
            rho=8.96,
            refer_rho=8.96,
            e0=1.15e11,
            nu=0.34,
            sigy=1.2e8,
            beta=36.0,
            n=0.45,
            eps_max=2.0,
            sigma_max=6.4e8,
            t0=300.0,
            tmelt=1356.0,
            rhoc_p=3450.0,
            pmin=-1.0e20,
            b1=2.8e-11,
            b2=2.8e-11,
            h=3.8e-4,
            f=0.1,
            title="Compact No Spaces",
        )

    def test_cross_format_roundtrip(self, tmp_path: Path):
        """Cross-format roundtrip: free -> parse -> fixed -> parse -> exact equality."""
        vals = {
            "rho": 8.96,
            "refer_rho": 8.96,
            "e0": 1.15e11,
            "nu": 0.34,
            "sigy": 1.2e8,
            "beta": 36.0,
            "n": 0.45,
            "eps_max": 2.5,
            "sigma_max": 6.4e8,
            "t0": 300.0,
            "tmelt": 1356.0,
            "rhoc_p": 3.45e3,
            "pmin": -1.0e20,
            "b1": 2.8e-11,
            "b2": 2.8e-11,
            "h": 3.8e-4,
            "f": 0.12,
        }

        # Step 1: Write free format
        deck1 = StarterDeck("CROSS_1")
        deck1.mat_law49(mat_id=101, title="Cross Step 1", fixed_format=False, comma=True, sig0=vals["sigy"], **vals)
        f1 = tmp_path / "cross_step1_0000.rad"
        f1.write_text(deck1.render() + "\n/END\n", encoding="utf-8")

        m1 = read_starter_deck(str(f1))
        mat1 = m1.mat_law49s[101]

        # Step 2: Write fixed format using parsed entity
        deck2 = StarterDeck("CROSS_2")
        deck2.mat_law49(mat1, fixed_format=True)
        f2 = tmp_path / "cross_step2_0000.rad"
        f2.write_text(deck2.render() + "\n/END\n", encoding="utf-8")

        m2 = read_starter_deck(str(f2))
        mat2 = m2.mat_law49s[101]
        mat_entity2 = m2.materials[101]

        _assert_law49_all_fields_exact(mat2, mat_entity2, title="Cross Step 1", **vals)


# ============================================================================
# 3. Keyword Synonyms
# ============================================================================

class TestLaw49Synonyms:
    """Audit all keyword synonyms: /MAT/LAW49, /MAT/STEINB, /MAT/STEINBERG, /MAT/STEINBERG_GUINAN."""

    @pytest.mark.parametrize(
        "emitter_method,synonym",
        [
            ("mat_law49", "LAW49"),
            ("mat_steinb", "STEINB"),
            ("mat_steinberg", "STEINBERG"),
            ("mat_steinberg_guinan", "STEINBERG_GUINAN"),
        ],
    )
    def test_synonym_roundtrip(self, tmp_path: Path, emitter_method: str, synonym: str):
        """Each synonym produces the correct keyword header and parses into model.mat_law49s and model.materials."""
        vals = {
            "rho": 8.96,
            "refer_rho": 8.96,
            "e0": 1.15e11,
            "nu": 0.34,
            "sigy": 1.2e8,
            "beta": 36.0,
            "n": 0.45,
            "eps_max": 1.0e20,
            "sigma_max": 6.4e8,
            "t0": 300.0,
            "tmelt": 1356.0,
            "rhoc_p": 3.45e3,
            "pmin": -1.0e20,
            "b1": 2.8e-11,
            "b2": 2.8e-11,
            "h": 3.8e-4,
            "f": 0.0,
        }

        deck = StarterDeck(f"SYN_{synonym}")
        getattr(deck, emitter_method)(
            mat_id=50,
            title=f"Synonym {synonym}",
            sig0=vals["sigy"],
            **vals,
        )

        rendered = deck.render()
        assert f"/MAT/{synonym}/50" in rendered

        rad_path = tmp_path / f"syn_{synonym}_0000.rad"
        rad_path.write_text(rendered + "\n/END\n", encoding="utf-8")

        model, log = read_starter_deck(str(rad_path))
        assert not log.has_errors
        assert 50 in model.mat_law49s
        assert 50 in model.materials
        # Aliases in model
        assert 50 in model.mat_steinbs
        assert 50 in model.mat_steinbergs
        assert 50 in model.mat_steinberg_guinans

        m = model.mat_law49s[50]
        mat = model.materials[50]
        _assert_law49_all_fields_exact(m, mat, title=f"Synonym {synonym}", **vals)


# ============================================================================
# 4. Negative Starter Diagnostics
# ============================================================================

class TestLaw49NegativeStarterDiagnostics:
    """Audit check_mat_law49 negative validations and incompatible element rejections."""

    def test_negative_density(self):
        """Density rho <= 0 triggers an error."""
        # rho = 0
        log0 = MessageLog()
        m0 = MatLaw49(id=1, rho=0.0, e0=1.15e11, nu=0.34)
        check_mat_law49(mat=m0, log=log0)
        assert log0.has_errors
        assert any("initial density RHO must be > 0" in e for e in log0.errors)

        # rho < 0
        log_neg = MessageLog()
        m_neg = MatLaw49(id=1, rho=-8.96, e0=1.15e11, nu=0.34)
        check_mat_law49(mat=m_neg, log=log_neg)
        assert log_neg.has_errors
        assert any("initial density RHO must be > 0" in e for e in log_neg.errors)

    def test_negative_youngs_modulus(self):
        """Young's modulus e0 <= 0 triggers an error."""
        # e0 = 0
        log0 = MessageLog()
        m0 = MatLaw49(id=2, rho=8.96, e0=0.0, nu=0.34)
        check_mat_law49(mat=m0, log=log0)
        assert log0.has_errors
        assert any("Young's modulus E must be > 0" in e for e in log0.errors)

        # e0 < 0
        log_neg = MessageLog()
        m_neg = MatLaw49(id=2, rho=8.96, e0=-1.15e11, nu=0.34)
        check_mat_law49(mat=m_neg, log=log_neg)
        assert log_neg.has_errors
        assert any("Young's modulus E must be > 0" in e for e in log_neg.errors)

    def test_invalid_poissons_ratio_ancmsg_1514(self):
        """Poisson's ratio nu < 0 or nu >= 0.5 triggers ANCMSG 1514 error."""
        # nu < 0
        log_neg = MessageLog()
        m_neg = MatLaw49(id=3, rho=8.96, e0=1.15e11, nu=-0.1)
        check_mat_law49(mat=m_neg, log=log_neg)
        assert log_neg.has_errors
        assert any("0 <= nu < 0.5" in e and "ANCMSG 1514" in e for e in log_neg.errors)

        # nu = 0.5
        log_half = MessageLog()
        m_half = MatLaw49(id=3, rho=8.96, e0=1.15e11, nu=0.5)
        check_mat_law49(mat=m_half, log=log_half)
        assert log_half.has_errors
        assert any("0 <= nu < 0.5" in e and "ANCMSG 1514" in e for e in log_half.errors)

        # nu > 0.5
        log_hi = MessageLog()
        m_hi = MatLaw49(id=3, rho=8.96, e0=1.15e11, nu=0.6)
        check_mat_law49(mat=m_hi, log=log_hi)
        assert log_hi.has_errors
        assert any("0 <= nu < 0.5" in e and "ANCMSG 1514" in e for e in log_hi.errors)

    def test_negative_f_coefficient_ancmsg_1513(self):
        """Negative f coefficient < 0 triggers ANCMSG 1513 error."""
        log = MessageLog()
        m = MatLaw49(id=4, rho=8.96, e0=1.15e11, nu=0.34, f=-0.5)
        check_mat_law49(mat=m, log=log)
        assert log.has_errors
        assert any("F coefficient must be >= 0" in e and "ANCMSG 1513" in e for e in log.errors)

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

        mat49 = MatLaw49(id=49, rho=8.96, e0=1.15e11, nu=0.34)

        # 1. Solids accepted
        for s_type in ("solids", "bricks", "tetras", "penta6", "pyra5"):
            model_s = DummyModel([(s_type, DummyGrp([DummyEl(49)]))])
            log_s = MessageLog()
            check_mat_law49(model=model_s, mat=mat49, log=log_s)
            assert not log_s.has_errors, f"Solid element {s_type} should be accepted"

        # 2. Shells and quads rejected (ANCMSG 305)
        for sh_type in ("shells", "shells_qbat", "shells_qeph", "sh3n", "quads"):
            model_sh = DummyModel([(sh_type, DummyGrp([DummyEl(49)]))])
            log_sh = MessageLog()
            check_mat_law49(model=model_sh, mat=mat49, log=log_sh)
            assert log_sh.has_errors, f"Shell element {sh_type} should be rejected"
            assert any("shell elements" in e and "ANCMSG 305" in e for e in log_sh.errors)

        # 3. 1D elements rejected (ANCMSG 306)
        for d1_type in ("trusses", "beams", "springs"):
            model_1d = DummyModel([(d1_type, DummyGrp([DummyEl(49)]))])
            log_1d = MessageLog()
            check_mat_law49(model=model_1d, mat=mat49, log=log_1d)
            assert log_1d.has_errors, f"1D element {d1_type} should be rejected"
            assert any("1D elements" in e and "ANCMSG 306" in e for e in log_1d.errors)

        # 4. 2D analysis rejected (N2D > 0, ANCMSG 305)
        model_2d = DummyModel([("solids", DummyGrp([DummyEl(49)]))], n2d=1)
        log_2d = MessageLog()
        check_mat_law49(model=model_2d, mat=mat49, log=log_2d)
        assert log_2d.has_errors
        assert any("N2D > 0" in e and "ANCMSG 305" in e for e in log_2d.errors)

    def test_checks_dispatch_and_allowed_laws_registry(self):
        """Verify _MAT_CHECKS registry and _ALLOWED_LAWS table for all LAW49 synonyms."""
        for syn in (49, "49", "LAW49", "STEINB", "STEINBERG", "STEINBERG_GUINAN", "MAT_LAW49", "MAT_STEINB", "MAT_STEINBERG", "LAW49_STEINB"):
            assert syn in _MAT_CHECKS, f"{syn} missing from _MAT_CHECKS"
            assert _MAT_CHECKS[syn] is check_mat_law49

        # Solids permit LAW49
        for fam in ("bricks", "tetras", "penta6", "pyra5", "solids"):
            assert 49 in _ALLOWED_LAWS[fam]
            assert "LAW49" in _ALLOWED_LAWS[fam]
            assert "STEINB" in _ALLOWED_LAWS[fam]

        # Shells and lines forbid LAW49
        for fam in ("shells", "shells_qbat", "shells_qeph", "sh3n", "quads", "trusses", "beams"):
            assert 49 not in _ALLOWED_LAWS[fam]
            assert "LAW49" not in _ALLOWED_LAWS[fam]
            assert "STEINB" not in _ALLOWED_LAWS[fam]


# ============================================================================
# 5. Boundary Values & Melt Softening
# ============================================================================

class TestLaw49BoundaryValues:
    """Audit boundary values and melt softening as espe approaches and exceeds Emelt."""

    def test_omitted_defaults(self):
        """Omitted optional parameters receive exact Fortran defaults."""
        m = MatLaw49(id=1, rho=8.96, e0=1.15e11, nu=0.34)
        assert m.t0 == 300.0
        assert m.eps_max == 1.0e20
        assert m.sigma_max == 1.0e20
        assert m.tmelt == 1.0e20
        assert m.pmin == -1.0e20
        assert m.refer_rho == 8.96

    def test_melt_softening_as_espe_approaches_emelt(self):
        """QC melt softening factor transitions smoothly to 0 as espe -> Emelt, and drops to 0 for espe >= Emelt."""
        mat = build_law49({
            "id": 1,
            "rho": 8.96,
            "e0": 1.15e11,
            "nu": 0.34,
            "sig0": 1.2e8,
            "rhoc_p": 3450.0,
            "tmelt": 1356.0,
            "f": 2.0,
        })
        Emelt = mat.params["rhoc_p"] * mat.params["tmelt"]  # 3450 * 1356 = 4.6782e6

        sig = np.zeros((1, 6))
        deps = np.array([[0.0, 0.0, 0.0, 0.001, 0.0, 0.0]])  # Small shear strain increment

        # Elastic baseline (espe = 0 -> qc = 1.0)
        s_base = solid_update_law49(mat, sig.copy(), deps, extra={"espe": np.array([0.0])})
        tau_base = s_base[0, 3]
        expected_tau_base = mat.params["g0"] * 0.001  # G * deps
        assert tau_base == pytest.approx(expected_tau_base, rel=1e-5)

        # espe = 0.5 * Emelt -> qc = exp(f * 0.5 / -0.5) = exp(-f) = exp(-2.0) ~ 0.135335
        s_half = solid_update_law49(mat, sig.copy(), deps, extra={"espe": np.array([0.5 * Emelt])})
        tau_half = s_half[0, 3]
        expected_tau_half = expected_tau_base * math.exp(-2.0)
        assert tau_half == pytest.approx(expected_tau_half, rel=1e-5)

        # espe = 0.9 * Emelt -> qc = exp(2 * 0.9 / -0.1) = exp(-18) ~ 1.52e-8
        s_90 = solid_update_law49(mat, sig.copy(), deps, extra={"espe": np.array([0.9 * Emelt])})
        tau_90 = s_90[0, 3]
        expected_tau_90 = expected_tau_base * math.exp(-18.0)
        assert tau_90 == pytest.approx(expected_tau_90, rel=1e-5)

        # espe = 0.99 * Emelt -> near zero
        s_99 = solid_update_law49(mat, sig.copy(), deps, extra={"espe": np.array([0.99 * Emelt])})
        tau_99 = s_99[0, 3]
        assert tau_99 < 1e-10

        # espe = 1.0 * Emelt -> completely melted (qc = 0, tau = 0)
        s_melt = solid_update_law49(mat, sig.copy(), deps, extra={"espe": np.array([Emelt])})
        assert s_melt[0, 3] == 0.0

        # espe = 1.2 * Emelt -> completely melted (qc = 0, tau = 0)
        s_over = solid_update_law49(mat, sig.copy(), deps, extra={"espe": np.array([1.2 * Emelt])})
        assert s_over[0, 3] == 0.0


# ============================================================================
# 6. Restart (.rst) Serialization
# ============================================================================

class TestLaw49RestartSerialization:
    """Audit restart state serialization and dynamic cycle continuation for LAW49."""

    def test_mat_law49_dataclass_pickle_fidelity(self):
        """Pickle and unpickle MatLaw49, verifying all fields and derived properties."""
        mat = MatLaw49(
            id=49,
            rho=8.96,
            refer_rho=8.90,
            e0=1.15e11,
            nu=0.34,
            sigy=1.2e8,
            beta=36.0,
            n=0.45,
            eps_max=2.5,
            sigma_max=6.4e8,
            t0=293.15,
            tmelt=1356.0,
            rhoc_p=3.45e3,
            pmin=-1.0e20,
            b1=2.8e-11,
            b2=2.8e-11,
            h=3.8e-4,
            f=0.15,
            title="Copper-Pickle",
        )

        data = pickle.dumps(mat, protocol=pickle.HIGHEST_PROTOCOL)
        restored: MatLaw49 = pickle.loads(data)

        _assert_law49_all_fields_exact(
            restored,
            rho=8.96,
            refer_rho=8.90,
            e0=1.15e11,
            nu=0.34,
            sigy=1.2e8,
            beta=36.0,
            n=0.45,
            eps_max=2.5,
            sigma_max=6.4e8,
            t0=293.15,
            tmelt=1356.0,
            rhoc_p=3.45e3,
            pmin=-1.0e20,
            b1=2.8e-11,
            b2=2.8e-11,
            h=3.8e-4,
            f=0.15,
            title="Copper-Pickle",
        )

    def test_model_with_law49_pickle_fidelity(self):
        """Pickle and unpickle full Model with LAW49 material and check consistency."""
        model = Model()
        m49 = MatLaw49(
            id=10,
            rho=8.96,
            refer_rho=8.96,
            e0=1.15e11,
            nu=0.34,
            sigy=1.2e8,
            beta=36.0,
            n=0.45,
            eps_max=1.0e20,
            sigma_max=6.4e8,
            t0=300.0,
            tmelt=1356.0,
            rhoc_p=3.45e3,
            pmin=-1.0e20,
            b1=2.8e-11,
            b2=2.8e-11,
            h=3.8e-4,
            f=0.0,
            title="Model Pickled Law49",
        )
        model.mat_law49s[10] = m49
        model.materials[10] = Material(id=10, law=49, rho0=8.96, title="Model Pickled Law49", params=m49.params)

        data = pickle.dumps(model, protocol=pickle.HIGHEST_PROTOCOL)
        restored_model: Model = pickle.loads(data)

        assert 10 in restored_model.mat_law49s
        assert 10 in restored_model.materials
        _assert_law49_all_fields_exact(
            restored_model.mat_law49s[10],
            restored_model.materials[10],
            rho=8.96,
            refer_rho=8.96,
            e0=1.15e11,
            nu=0.34,
            sigy=1.2e8,
            beta=36.0,
            n=0.45,
            eps_max=1.0e20,
            sigma_max=6.4e8,
            t0=300.0,
            tmelt=1356.0,
            rhoc_p=3.45e3,
            pmin=-1.0e20,
            b1=2.8e-11,
            b2=2.8e-11,
            h=3.8e-4,
            f=0.0,
            title="Model Pickled Law49",
        )

    def test_material_state_arrays_rst_preservation(self, tmp_path: Path):
        """Verify persistent material state arrays survive write_restart and read_restart."""
        deck = StarterDeck("RST_LAW49_ARRAYS")
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
        deck.part(1, "CopperPart", prop_id=1, mat_id=1)
        deck.mat_law49(
            mat_id=1,
            title="Shock Copper",
            rho=8960.0,
            e0=1.15e11,
            nu=0.34,
            sig0=1.2e8,
            beta=36.0,
            n=0.45,
            rhoc_p=3.45e6,
            tmelt=1356.0,
            b1=2.8e-11,
            b2=2.8e-11,
            h=3.8e-4,
            f=0.1,
        )

        rad_path = tmp_path / "rst_law49_arrays_0000.rad"
        deck.write(str(rad_path))

        model = run_starter(str(rad_path))
        assert model is not None

        groups = dict(model.element_groups())
        bg = groups.get("bricks") or groups.get("solids")
        st = bg.state

        # Synthesize realistic persistent material state arrays for LAW49 (2 elements)
        theta_synth = np.array([325.4, 410.8], dtype=float)
        espe_synth = np.array([1.2e5, 5.8e5], dtype=float)
        epxe_synth = np.array([0.0055, 0.0210], dtype=float)
        dpla_synth = np.array([1.2e-5, 4.5e-5], dtype=float)
        off_synth = np.array([1.0, 0.0], dtype=float)  # 1 alive, 1 eroded
        sig_synth = np.array([
            [1.5e8, 1.2e8, 1.3e8, 4.5e7, 2.1e7, 1.8e7],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        ], dtype=float)
        epsp_synth = np.array([0.0055, 0.0210], dtype=float)

        st["mat_extra"]["theta"] = theta_synth.copy()
        st["mat_extra"]["espe"] = espe_synth.copy()
        st["mat_extra"]["epxe"] = epxe_synth.copy()
        st["mat_extra"]["dpla"] = dpla_synth.copy()
        st["off"] = off_synth.copy()
        st["sig"] = sig_synth.copy()
        st["epsp"] = epsp_synth.copy()

        # Write restart .rst
        rst_path = tmp_path / "rst_law49_0001.rst"
        engine_dict = {
            "cycle": 500,
            "t": 1.25e-5,
            "dt": 2.5e-8,
            "energies": {"internal": 45000.0, "kinetic": 12000.0},
        }
        write_restart(model, str(rst_path), engine=engine_dict)

        # Read back from restart .rst
        rest_model, rest_engine = read_restart(str(rst_path))
        assert rest_engine["cycle"] == 500
        assert rest_engine["t"] == pytest.approx(1.25e-5)
        assert rest_engine["dt"] == pytest.approx(2.5e-8)

        rest_bg = dict(rest_model.element_groups()).get("bricks") or dict(rest_model.element_groups()).get("solids")
        rest_extra = rest_bg.state["mat_extra"]
        rest_st = rest_bg.state

        # Assert exact preservation of LAW49 state arrays
        np.testing.assert_array_equal(rest_extra["theta"], theta_synth)
        np.testing.assert_array_equal(rest_extra["espe"], espe_synth)
        np.testing.assert_array_equal(rest_extra["epxe"], epxe_synth)
        np.testing.assert_array_equal(rest_extra["dpla"], dpla_synth)
        np.testing.assert_array_equal(rest_st["off"], off_synth)
        np.testing.assert_array_equal(rest_st["sig"], sig_synth)
        np.testing.assert_array_equal(rest_st["epsp"], epsp_synth)

    def test_constitutive_dynamic_cycle_continuation_from_restart(self, tmp_path: Path):
        """Constitutive stress update continuation from restart matches uninterrupted simulation within 10^-12."""
        mat = build_law49({
            "id": 1,
            "rho": 8960.0,
            "e0": 1.15e11,
            "nu": 0.34,
            "sig0": 1.2e8,
            "beta": 36.0,
            "n": 0.45,
            "rhoc_p": 3.45e6,
            "tmelt": 1356.0,
            "b1": 2.8e-11,
            "b2": 2.8e-11,
            "h": 3.8e-4,
            "f": 0.1,
        })

        sig = np.zeros((1, 6))
        epsp = np.zeros(1)
        extra = {
            "theta": np.array([300.0]),
            "espe": np.array([0.0]),
            "epxe": np.array([0.0]),
            "dpla": np.array([0.0]),
            "off": np.array([1.0]),
        }

        # Step 1-5 uninterrupted
        deps_history = [
            np.array([[0.001, -0.0003, -0.0003, 0.0005, 0.0, 0.0]]),
            np.array([[0.001, -0.0003, -0.0003, 0.0008, 0.0, 0.0]]),
            np.array([[0.001, -0.0003, -0.0003, 0.0010, 0.0, 0.0]]),
            np.array([[0.0005, -0.0002, -0.0002, 0.0005, 0.0, 0.0]]),
            np.array([[0.0005, -0.0002, -0.0002, 0.0002, 0.0, 0.0]]),
        ]

        # Run uninterrupted sequence
        sig_uninterrupted = sig.copy()
        epsp_uninterrupted = epsp.copy()
        extra_uninterrupted = {k: v.copy() for k, v in extra.items()}

        for deps in deps_history:
            sig_uninterrupted, epsp_uninterrupted, _ = solid_update_law49(
                mat, sig_uninterrupted, deps, epsp=epsp_uninterrupted, extra=extra_uninterrupted, return_tuple=True
            )

        # Run with checkpoint at step 3
        sig_run = sig.copy()
        epsp_run = epsp.copy()
        extra_run = {k: v.copy() for k, v in extra.items()}

        for deps in deps_history[:3]:
            sig_run, epsp_run, _ = solid_update_law49(
                mat, sig_run, deps, epsp=epsp_run, extra=extra_run, return_tuple=True
            )

        # Serialize / restart checkpoint
        checkpoint = pickle.dumps((sig_run, epsp_run, extra_run), protocol=pickle.HIGHEST_PROTOCOL)
        sig_restored, epsp_restored, extra_restored = pickle.loads(checkpoint)

        # Continue steps 4-5 from checkpoint
        for deps in deps_history[3:]:
            sig_restored, epsp_restored, _ = solid_update_law49(
                mat, sig_restored, deps, epsp=epsp_restored, extra=extra_restored, return_tuple=True
            )

        # Compare uninterrupted vs restarted results
        np.testing.assert_allclose(sig_restored, sig_uninterrupted, rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(epsp_restored, epsp_uninterrupted, rtol=1e-12, atol=1e-12)
        for k in extra_uninterrupted:
            np.testing.assert_allclose(extra_restored[k], extra_uninterrupted[k], rtol=1e-12, atol=1e-12)
