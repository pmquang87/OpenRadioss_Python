"""
Tests for Milestone M569: /MAT/LAW95 (/MAT/BERGSTROM_BOYCE) Bergstrom-Boyce Visco-Hyperelastic Model
Starter, Model Entity, Deck Reader, Checks, & Element Compatibility.
"""

from __future__ import annotations

import math
import numpy as np
import pytest

from pyradioss.model.entities import (
    MaterialLaw95,
    MatLaw95,
    MatBergstromBoyce,
    Part,
    Property,
)
from pyradioss.model import Model
from pyradioss.input.deck_reader import read_deck
from pyradioss.input.starter_keywords import read_mat_law95
from pyradioss.starter.checks import (
    check_mat_law95,
    check_materials,
    check_model,
    MessageLog,
)
from pyradioss.input.checks import (
    check_mat_law95 as input_check_mat_law95,
    check_all,
)
import pyradioss.materials as mats


# ============================================================================
# 1. Model Entity & Aliases
# ============================================================================

def test_material_law95_aliases_and_defaults():
    """Verify MaterialLaw95 class, aliases, and default attributes."""
    assert MatLaw95 is MaterialLaw95
    assert MatBergstromBoyce is MaterialLaw95

    mat = MaterialLaw95(id=10, title="Polymer_BB")
    assert mat.id == 10
    assert mat.title == "Polymer_BB"
    assert mat.law == 95
    assert mat.law_name == "LAW95"
    assert mat.rho0 == 0.0
    assert mat.c10 == 0.0
    assert mat.c01 == 0.0
    assert mat.sb == 0.0
    assert mat.d1 == 0.0
    assert mat.d2 == 0.0
    assert mat.d3 == 0.0
    assert mat.iform == 1
    assert mat.expc == pytest.approx(-0.7)
    assert mat.expm == pytest.approx(1.0)
    assert mat.ksi == pytest.approx(0.01)
    assert mat.tauref == pytest.approx(1.0)

    model = Model()
    assert hasattr(model, "mat_law95s")
    assert isinstance(model.mat_law95s, dict)
    assert hasattr(model, "mat_bergstrom_boyces")
    assert model.mat_bergstrom_boyces is model.mat_law95s


def test_material_law95_mapping_and_properties():
    """Verify dictionary-like mapping protocol and derived properties."""
    mat = MaterialLaw95(
        id=1,
        title="BBPolymer",
        rho0=1100.0,
        c10=1.5e6,
        c01=2.5e5,
        c20=0.0,
        sb=0.5,
        d1=1.0e-8,
        d2=0.0,
        d3=0.0,
        a=0.01,
        expc=-0.7,
        expm=1.2,
        ksi=0.01,
        tauref=1.0e5,
    )
    assert mat["id"] == 1
    assert mat["c10"] == 1.5e6
    assert "c10" in mat
    assert "rho0" in mat
    assert mat.get("sb") == 0.5
    assert mat.get("nonexistent", 99.0) == 99.0

    # Mutability via mapping protocol
    mat["c10"] = 2.0e6
    assert mat.c10 == 2.0e6

    # G0 = 2 * (C10 + C01) * (1 + Sb) = 2 * (2e6 + 2.5e5) * 1.5 = 6.75e6
    assert math.isclose(mat.G, 6.75e6, rel_tol=1e-6)
    assert math.isclose(mat.shear, 6.75e6, rel_tol=1e-6)

    # Bulk modulus K = 2 * (1 / D1) * (1 + Sb) = 2 * 1e8 * 1.5 = 3.0e8
    assert math.isclose(mat.K, 3.0e8, rel_tol=1e-6)
    assert math.isclose(mat.bulk, 3.0e8, rel_tol=1e-6)

    # Sound speed solid: c = sqrt((K + 4/3*G0) / rho)
    c_expected = math.sqrt((3.0e8 + (4.0 / 3.0) * 6.75e6) / 1100.0)
    assert math.isclose(float(mat.sound_speed), c_expected, rel_tol=1e-6)
    assert math.isclose(float(mat.sound_speed()), c_expected, rel_tol=1e-6)
    assert math.isclose(float(mat.sound_speed_solid), c_expected, rel_tol=1e-6)


# ============================================================================
# 2. Deck Reader: Fixed and Free Format
# ============================================================================

def test_deck_reader_law95_fixed_format(tmp_path):
    """Verify parsing /MAT/LAW95 card in standard fixed format."""
    deck_text = """# OpenRadioss Starter Input Deck
/BEGIN
Test LAW95 Fixed
      2022         0
/MAT/LAW95/101
Bergstrom_Boyce_Fixed
#              RHO_I
              1150.0                 0.0
#                C10                 C01                 C20                 C11                 C02
               2.0E6               3.0E5                 0.0                 0.0                 0.0
#                C30                 C21                 C12                 C03                  Sb
                 0.0                 0.0                 0.0                 0.0                 0.8
#                 D1                  D2                  D3                  NU               IFORM
                1E-8                 0.0                 0.0                 0.0                   1
#                  A                   C                   M                 KSI             TAU_REF
               0.005                -0.7                 1.0                0.01              1.0E+5
/END
"""
    deck_file = tmp_path / "deck_fixed.rad"
    deck_file.write_text(deck_text, encoding="utf-8")
    blocks = read_deck(str(deck_file))

    model = Model()
    log = MessageLog()
    for block in blocks:
        if block.key0 == "MAT":
            read_mat_law95(block, model, log)

    assert 101 in model.mat_law95s
    mat = model.mat_law95s[101]
    assert mat.id == 101
    assert "Bergstrom_Boyce_Fixed" in mat.title
    assert math.isclose(mat.rho0, 1150.0)
    assert math.isclose(mat.c10, 2.0e6)
    assert math.isclose(mat.c01, 3.0e5)
    assert math.isclose(mat.sb, 0.8)
    assert math.isclose(mat.d1, 1e-8)
    assert mat.iform == 1
    assert math.isclose(mat.a, 0.005)
    assert math.isclose(mat.expc, -0.7)
    assert math.isclose(mat.expm, 1.0)
    assert math.isclose(mat.ksi, 0.01)
    assert math.isclose(mat.tauref, 1.0e5)

    # Generic material table
    assert 101 in model.materials
    gen_mat = model.materials[101]
    assert gen_mat.law == 95
    assert math.isclose(gen_mat.rho0, 1150.0)


def test_deck_reader_law95_free_format(tmp_path):
    """Verify parsing /MAT/BERGSTROM_BOYCE in free format."""
    deck_text = """/BEGIN
Test Free Format
/MAT/BERGSTROM_BOYCE/202
Elastomer_BB_Free
1100.0
1.5e6 2.0e5 0.0 0.0 0.0
0.0 0.0 0.0 0.0 1.2
2.0e-8 0.0 0.0 0.0 1
0.01 -0.7 1.0 0.01 1.0e5
/END
"""
    deck_file = tmp_path / "deck_free.rad"
    deck_file.write_text(deck_text, encoding="utf-8")
    blocks = read_deck(str(deck_file))

    model = Model()
    log = MessageLog()
    for block in blocks:
        if block.key0 == "MAT":
            read_mat_law95(block, model, log)

    assert 202 in model.mat_law95s
    mat = model.mat_law95s[202]
    assert mat.id == 202
    assert math.isclose(mat.rho0, 1100.0)
    assert math.isclose(mat.c10, 1.5e6)
    assert math.isclose(mat.c01, 2.0e5)
    assert math.isclose(mat.sb, 1.2)
    assert math.isclose(mat.d1, 2.0e-8)
    assert math.isclose(mat.a, 0.01)


# ============================================================================
# 3. Parameter Validation & Starter Checks
# ============================================================================

def test_check_mat_law95_valid():
    """Valid LAW95 parameters pass without errors."""
    log = MessageLog()
    mat = MaterialLaw95(
        id=1,
        rho0=1000.0,
        c10=1.0e6,
        c01=2.0e5,
        sb=0.5,
        d1=1e-7,
        a=0.01,
        expc=-0.7,
        expm=1.0,
        ksi=0.01,
        tauref=1.0e5,
    )
    check_mat_law95(mat=mat, log=log)
    assert not log.has_errors


def test_check_mat_law95_zero_or_negative_density():
    """Zero or negative initial density triggers ANCMSG 1514 error."""
    log = MessageLog()
    mat = MaterialLaw95(
        id=2,
        rho0=0.0,
        c10=1.0e6,
        d1=1e-7,
    )
    check_mat_law95(mat=mat, log=log)
    assert log.has_errors
    assert any("ANCMSG 1514" in str(err) or "density" in str(err).lower() for err in log.errors)


def test_check_mat_law95_zero_shear_modulus():
    """Zero or negative shear modulus emits error or warning."""
    log = MessageLog()
    mat = MaterialLaw95(
        id=3,
        rho0=1000.0,
        c10=0.0,
        c01=0.0,
        d1=1e-7,
    )
    check_mat_law95(mat=mat, log=log)
    assert log.has_errors or len(log.warnings) > 0


def test_check_mat_law95_negative_d():
    """Negative compressibility parameter D1 triggers error."""
    log = MessageLog()
    mat = MaterialLaw95(
        id=4,
        rho0=1000.0,
        c10=1.0e6,
        d1=-1.0e-8,
    )
    check_mat_law95(mat=mat, log=log)
    assert log.has_errors
    assert any("d1" in str(err).lower() for err in log.errors)


def test_check_mat_law95_invalid_creep_params():
    """Invalid creep parameters (C >= 0, M < 1, TAU_REF <= 0) trigger errors."""
    log = MessageLog()
    mat = MaterialLaw95(
        id=5,
        rho0=1000.0,
        c10=1.0e6,
        d1=1e-7,
        a=0.01,
        c=0.5,       # C must satisfy -1 < C < 0
        m=0.5,       # M must satisfy M >= 1.0
        tau_ref=0.0, # TAU_REF must be > 0
    )
    check_mat_law95(mat=mat, log=log)
    assert log.has_errors


def test_check_mat_law95_1d_element_incompatibility():
    """LAW95 assigned to 1D beam or truss element triggers ANCMSG 306 error."""
    model = Model()
    model.materials[10] = MaterialLaw95(
        id=10,
        rho0=1000.0,
        c10=1.0e6,
        d1=1e-7,
    )
    part = Part(id=1, prop_id=1, mat_id=10, title="BeamPart")
    part.elem_type = "BEAM"
    model.parts[1] = part

    log = MessageLog()
    check_mat_law95(model=model, mat_id=10, mat=model.materials[10], log=log)
    assert log.has_errors
    assert any("ANCMSG 306" in str(err) or "1d elements" in str(err).lower() for err in log.errors)


def test_check_mat_law95_2d_shell_incompatibility():
    """LAW95 assigned to 2D shell element triggers ANCMSG 305 error."""
    model = Model()
    model.materials[20] = MaterialLaw95(
        id=20,
        rho0=1000.0,
        c10=1.0e6,
        d1=1e-7,
    )
    part = Part(id=2, prop_id=2, mat_id=20, title="ShellPart")
    part.elem_type = "SHELL"
    model.parts[2] = part

    log = MessageLog()
    check_mat_law95(model=model, mat_id=20, mat=model.materials[20], log=log)
    assert log.has_errors
    assert any("ANCMSG 305" in str(err) or "2d shell" in str(err).lower() for err in log.errors)


def test_check_mat_law95_2d_formulation_incompatibility():
    """Model with 2D formulation (N2D > 0) triggers ANCMSG 305 error for LAW95."""
    model = Model()
    model.n2d = 1
    model.materials[30] = MaterialLaw95(
        id=30,
        rho0=1000.0,
        c10=1.0e6,
        d1=1e-7,
    )
    log = MessageLog()
    check_model(model, log)
    assert log.has_errors
    assert any("ANCMSG 305" in str(err) or "2d analysis" in str(err).lower() for err in log.errors)


def test_starter_resolve_hook_law95():
    """Verify that starter resolve hook returns BergstromBoyceParams."""
    mat = MaterialLaw95(
        id=1,
        rho0=1000.0,
        c10=2.0e6,
        c01=4.0e5,
        sb=0.6,
        d1=1.0e-8,
    )
    log = MessageLog()
    p = mats.law95_bergstrom_boyce.resolve(mat, None, log)
    assert isinstance(p, mats.law95_bergstrom_boyce.BergstromBoyceParams)
    assert math.isclose(p.c10, 2.0e6)
    assert math.isclose(p.c01, 4.0e5)
    # G0 = 2 * (2e6 + 4e5) * 1.6 = 7.68e6
    assert math.isclose(p.g0, 7.68e6)
