"""
Tests for Milestone M568: /MAT/LAW93 (/MAT/ORTH_HILL) Orthotropic Hill Model
Starter, Model Entity, Deck Reader, Checks, & Element Compatibility.
"""

from __future__ import annotations

import math
import numpy as np
import pytest

from pyradioss.model.entities import (
    MaterialLaw93,
    MatLaw93,
    MatOrthHill,
    Part,
    Property,
)
from pyradioss.model import Model
from pyradioss.input.deck_reader import read_deck
from pyradioss.input.starter_keywords import read_mat_law93
from pyradioss.starter.checks import (
    check_mat_law93,
    check_materials,
    check_model,
    MessageLog,
)
from pyradioss.input.checks import (
    check_mat_law93 as input_check_mat_law93,
    check_all,
)
import pyradioss.materials as mats
import pyradioss.starter.initialization as starter_init


# ============================================================================
# 1. Model Entity & Aliases
# ============================================================================

def test_material_law93_aliases_and_defaults():
    """Verify MaterialLaw93 class, aliases, and default attributes."""
    assert MatLaw93 is MaterialLaw93
    assert MatOrthHill is MaterialLaw93

    mat = MaterialLaw93(id=5, title="SheetMetal_Hill")
    assert mat.id == 5
    assert mat.title == "SheetMetal_Hill"
    assert mat.law == 93
    assert mat.law_name == "LAW93"
    assert mat.rho0 == 0.0
    assert mat.e11 == 0.0
    assert mat.e22 == 0.0
    assert mat.e33 == 0.0
    assert mat.g12 == 0.0
    assert mat.nu12 == 0.0
    assert mat.sigma_y == 0.0
    assert mat.r11 == 1.0
    assert mat.r22 == 1.0
    assert mat.r33 == 1.0
    assert mat.r12 == 1.0
    assert mat.r13 == 1.0
    assert mat.r23 == 1.0

    model = Model()
    assert hasattr(model, "mat_law93s")
    assert isinstance(model.mat_law93s, dict)
    assert hasattr(model, "mat_orth_hills")
    assert model.mat_orth_hills is model.mat_law93s


def test_material_law93_mapping_and_properties():
    """Verify dictionary-like mapping protocol and physical properties."""
    mat = MaterialLaw93(
        id=1,
        title="Aluminium_Hill",
        rho0=2.7e-6,
        e11=70000.0,
        e22=68000.0,
        e33=70000.0,
        g12=26000.0,
        nu12=0.33,
        sigma_y=280.0,
        r11=1.0,
        r22=0.9,
        r12=0.85,
    )
    assert mat["id"] == 1
    assert mat["e11"] == 70000.0
    assert "e11" in mat
    assert "rho0" in mat
    assert mat.get("sigma_y") == 280.0
    assert mat.get("nonexistent", 123.0) == 123.0

    # Mutability
    mat["e11"] = 72000.0
    assert mat.e11 == 72000.0

    # CallableFloat sound speeds
    cs = float(mat.sound_speed_solid)
    assert cs > 0.0
    assert mat.sound_speed_solid() == cs
    csh = float(mat.sound_speed_shell)
    assert csh > 0.0
    assert mat.sound_speed_shell() == csh


# ============================================================================
# 2. Deck Reader: Standard 8-card Fixed Format
# ============================================================================

def test_read_deck_law93_standard_fixed(tmp_path):
    """Read a standard 8-card fixed-format deck with /MAT/LAW93."""
    deck_str = """# OpenRadioss starter input deck
/BEGIN
Test Deck LAW93 Standard Fixed
/MAT/LAW93/10
Orthotropic Hill Steel
#  RHO_I
           7.8e-6
#  E11                 E22                 E33                 G12                 Nu12
             210000.             205000.             210000.              80000.                 0.3
#  G13                 G23                 Nu13                Nu23
              80000.              78000.                 0.3                0.29
#  NL        VP                  FCUT
         0         0                  0.
#  Sigma_y             QR1                 CR1                 QR2                 CR2
                350.                120.                 25.                 60.                  5.
#  R11                 R22                 R12
                 1.1                0.95                 1.0
#  R33                 R13                 R23
                 1.0                 1.0                 1.0
/END
"""
    deck_file = tmp_path / "deck_fixed.rad"
    deck_file.write_text(deck_str, encoding="utf-8")
    blocks = read_deck(str(deck_file))
    model = Model()
    log = MessageLog()
    for block in blocks:
        if block.key0 == "MAT":
            read_mat_law93(block, model, log)

    assert 10 in model.materials
    mat = model.materials[10]
    assert mat.law == 93
    assert mat.rho0 == pytest.approx(7.8e-6)
    assert mat.e11 == pytest.approx(210000.0)
    assert mat.e22 == pytest.approx(205000.0)
    assert mat.e33 == pytest.approx(210000.0)
    assert mat.g12 == pytest.approx(80000.0)
    assert mat.nu12 == pytest.approx(0.3)
    assert mat.g13 == pytest.approx(80000.0)
    assert mat.g23 == pytest.approx(78000.0)
    assert mat.nu13 == pytest.approx(0.3)
    assert mat.nu23 == pytest.approx(0.29)
    assert mat.sigma_y == pytest.approx(350.0)
    assert mat.qr1 == pytest.approx(120.0)
    assert mat.cr1 == pytest.approx(25.0)
    assert mat.qr2 == pytest.approx(60.0)
    assert mat.cr2 == pytest.approx(5.0)
    assert mat.r11 == pytest.approx(1.1)
    assert mat.r22 == pytest.approx(0.95)
    assert mat.r12 == pytest.approx(1.0)


def test_read_deck_mat_orth_hill_free_format(tmp_path):
    """Read free-format deck using /MAT/ORTH_HILL synonym."""
    deck_str = """# Free format test
/BEGIN
Test Deck ORTH_HILL Free
/MAT/ORTH_HILL/20
Free format Hill
7.85e-6
200000.0, 195000.0, 200000.0, 77000.0, 0.28
77000.0, 75000.0, 0.28, 0.27
0, 0, 0.0
300.0, 100.0, 15.0, 0.0, 0.0
1.05, 0.92, 1.0
1.0, 1.0, 1.0
/END
"""
    deck_file = tmp_path / "deck_free.rad"
    deck_file.write_text(deck_str, encoding="utf-8")
    blocks = read_deck(str(deck_file))
    model = Model()
    log = MessageLog()
    for block in blocks:
        if block.key0 == "MAT":
            read_mat_law93(block, model, log)

    assert 20 in model.materials
    mat = model.materials[20]
    assert mat.law == 93
    assert mat.rho0 == pytest.approx(7.85e-6)
    assert mat.e11 == pytest.approx(200000.0)
    assert mat.e22 == pytest.approx(195000.0)
    assert mat.sigma_y == pytest.approx(300.0)
    assert mat.qr1 == pytest.approx(100.0)
    assert mat.cr1 == pytest.approx(15.0)
    assert mat.r11 == pytest.approx(1.05)
    assert mat.r22 == pytest.approx(0.92)


# ============================================================================
# 3. Model Checks & Diagnostic Messages
# ============================================================================

def test_check_mat_law93_density_error():
    """Density <= 0 must emit ANCMSG 1514."""
    mat = MaterialLaw93(id=1, rho0=0.0, e11=210000.0, g12=80000.0, nu12=0.3)
    log = MessageLog()
    check_mat_law93(mat=mat, log=log)
    assert any("ANCMSG 1514" in err for err in log.errors)


def test_check_mat_law93_elastic_moduli_error():
    """Moduli <= 0 must emit ANCMSG 307."""
    mat = MaterialLaw93(id=2, rho0=7.8e-6, e11=0.0, g12=80000.0, nu12=0.3)
    log = MessageLog()
    check_mat_law93(mat=mat, log=log)
    assert any("ANCMSG 307" in err for err in log.errors)


def test_check_mat_law93_poisson_cross_product_error():
    """nu12 * nu21 >= 1 must emit ANCMSG 3068."""
    # nu12 = 0.9, E11 = 200000, E22 = 250000 -> nu21 = 0.9 * 250000 / 200000 = 1.125
    # nu12 * nu21 = 0.9 * 1.125 = 1.0125 >= 1
    mat = MaterialLaw93(
        id=3,
        rho0=7.8e-6,
        e11=200000.0,
        e22=250000.0,
        e33=200000.0,
        g12=80000.0,
        nu12=0.9,
    )
    log = MessageLog()
    check_mat_law93(mat=mat, log=log)
    assert any("ANCMSG 3068" in err for err in log.errors)


def test_check_mat_law93_compliance_determinant_error():
    """Orthotropic compliance determinant <= 0 must emit ANCMSG 307."""
    # nu12 = 0.45, nu13 = 0.45, nu23 = 0.45, isotropic E
    # det_denom = 1 - 3*(0.45^2) - 2*(0.45^3) = 1 - 0.6075 - 0.18225 = 0.21 > 0
    # Let's set nu12 = 0.6, nu13 = 0.6, nu23 = 0.6 -> det_denom < 0
    mat = MaterialLaw93(
        id=4,
        rho0=7.8e-6,
        e11=200000.0,
        e22=200000.0,
        e33=200000.0,
        g12=80000.0,
        nu12=0.6,
        nu13=0.6,
        nu23=0.6,
    )
    log = MessageLog()
    check_mat_law93(mat=mat, log=log)
    assert any("ANCMSG 307" in err for err in log.errors)


def test_check_mat_law93_1d_element_rejection():
    """Attaching LAW93 to 1D elements (beam/truss/spring) emits ANCMSG 306."""
    model = Model()
    mat = MaterialLaw93(id=1, rho0=7.8e-6, e11=210000.0, g12=80000.0, nu12=0.3)
    model.materials[1] = mat
    model.mat_law93s[1] = mat

    part = Part(id=1, prop_id=1, mat_id=1, title="BeamPart")
    part.elem_type = "BEAM"
    model.parts[1] = part
    prop = Property(id=1, type="BEAM")
    model.properties[1] = prop

    log = MessageLog()
    check_mat_law93(mat=mat, model=model, log=log)
    assert any("ANCMSG 306" in err for err in log.errors)


def test_check_materials_full_model_pass():
    """Valid LAW93 material attached to solid parts passes check_model without error."""
    model = Model()
    mat = MaterialLaw93(
        id=1,
        rho0=7.8e-6,
        e11=210000.0,
        e22=210000.0,
        e33=210000.0,
        g12=80000.0,
        nu12=0.3,
        sigma_y=300.0,
    )
    model.materials[1] = mat
    model.mat_law93s[1] = mat
    part = Part(id=1, prop_id=1, mat_id=1, title="SolidPart")
    part.elem_type = "SOLID"
    model.parts[1] = part

    log = MessageLog()
    check_materials(model, log)
    assert len(log.errors) == 0
