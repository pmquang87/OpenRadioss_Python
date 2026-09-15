"""
Tests for Milestone M566: /MAT/LAW92 (/MAT/ARRUDA_BOYCE) Arruda-Boyce Hyperelastic Model
Starter, Model Entity, Deck Reader, Checks, Curve Resolution & Element Compatibility.
"""

from __future__ import annotations

import math
import numpy as np
import pytest

from pyradioss.model.entities import (
    MaterialLaw92,
    MatLaw92,
    MatArrudaBoyce,
    MatArruda,
    Part,
    Property,
)
from pyradioss.common.tables import FunctTable
from pyradioss.model import Model
from pyradioss.input.deck_reader import read_deck
from pyradioss.input.starter_keywords import read_mat_law92
from pyradioss.starter.checks import (
    check_mat_law92,
    check_materials,
    check_model,
    MessageLog,
)
from pyradioss.input.checks import (
    check_mat_law92 as input_check_mat_law92,
    check_all,
)
import pyradioss.materials as mats
import pyradioss.starter.initialization as starter_init


# ============================================================================
# 1. Model Entity & Aliases
# ============================================================================

def test_material_law92_aliases_and_defaults():
    """Verify MaterialLaw92 class, aliases, and default attributes."""
    assert MatLaw92 is MaterialLaw92
    assert MatArrudaBoyce is MaterialLaw92
    assert MatArruda is MaterialLaw92

    mat = MaterialLaw92(id=10, title="Rubber_Arruda")
    assert mat.id == 10
    assert mat.title == "Rubber_Arruda"
    assert mat.law == 92
    assert mat.law_name == "LAW92"
    assert mat.rho0 == 0.0
    assert mat.ref_rho == 0.0
    assert mat.mu == 0.0
    assert mat.d == 0.0
    assert mat.lam == 7.0
    assert mat.itype == 1
    assert mat.fct_id == 0
    assert mat.nu == 0.0
    assert mat.fscale == 1.0

    model = Model()
    assert hasattr(model, "mat_law92s")
    assert isinstance(model.mat_law92s, dict)
    assert hasattr(model, "mat_arruda_boyces")
    assert model.mat_arruda_boyces is model.mat_law92s


def test_material_law92_mapping_and_properties():
    """Verify dictionary-like mapping protocol and derived properties."""
    mat = MaterialLaw92(
        id=1,
        title="ArrudaPolymer",
        rho0=1100.0,
        mu=1.5e6,
        d=1e-8,
        lam=5.0,
        itype=1,
        fct_id=0,
        nu=0.495,
        fscale=1.0,
    )
    assert mat["id"] == 1
    assert mat["mu"] == 1.5e6
    assert "mu" in mat
    assert "rho0" in mat
    assert mat.get("lam") == 5.0
    assert mat.get("nonexistent", 99.0) == 99.0

    # Mutability via mapping protocol
    mat["mu"] = 2.0e6
    assert mat.mu == 2.0e6

    # Test derived elastic properties
    # G0 calculation: beta = 1/25 = 0.04
    # G0 = mu * (1 + 0.6*beta + 99/175*beta^2 + 513/875*beta^3 + 42039/67375*beta^4)
    beta = 1.0 / (5.0 ** 2)
    poly = (
        1.0
        + 0.6 * beta
        + (99.0 / 175.0) * (beta ** 2)
        + (513.0 / 875.0) * (beta ** 3)
        + (42039.0 / 67375.0) * (beta ** 4)
    )
    expected_g0 = 2.0e6 * poly
    assert math.isclose(mat.G0, expected_g0, rel_tol=1e-6)
    assert math.isclose(mat.G, expected_g0, rel_tol=1e-6)

    # Bulk modulus K = 2 / D = 2 / 1e-8 = 2e8
    assert math.isclose(mat.K, 2.0e8, rel_tol=1e-6)
    assert math.isclose(mat.bulk, 2.0e8, rel_tol=1e-6)

    # Sound speed: c = sqrt((K + 4/3*G0) / rho)
    c_expected = math.sqrt((2.0e8 + (4.0 / 3.0) * expected_g0) / 1100.0)
    assert math.isclose(float(mat.sound_speed), c_expected, rel_tol=1e-6)
    assert math.isclose(float(mat.sound_speed()), c_expected, rel_tol=1e-6)
    assert math.isclose(float(mat.sound_speed_solid), c_expected, rel_tol=1e-6)


# ============================================================================
# 2. Deck Reader: Fixed and Free Format
# ============================================================================

def test_deck_reader_law92_fixed_format(tmp_path):
    """Verify parsing /MAT/LAW92 card in standard 20-character fixed format."""
    deck_text = """# OpenRadioss Starter Input Deck
/BEGIN
Test LAW92 Fixed
      2022         0
/MAT/LAW92/101
ArrudaBoyce_Rubber
#              RHO_I
              1200.0                 0.0
#                MUE                 D                 LAM
               2.5E6                1E-8                 6.0
#    Itype  IFUNC_ID                  NU              Fscale
         1         0               0.495                 1.0
/END
"""
    deck_file = tmp_path / "deck_fixed.rad"
    deck_file.write_text(deck_text, encoding="utf-8")
    blocks = read_deck(str(deck_file))

    model = Model()
    log = MessageLog()
    for block in blocks:
        if block.key0 == "MAT":
            read_mat_law92(block, model, log)

    assert 101 in model.mat_law92s
    mat = model.mat_law92s[101]
    assert mat.id == 101
    assert "ArrudaBoyce_Rubber" in mat.title
    assert math.isclose(mat.rho0, 1200.0)
    assert math.isclose(mat.mu, 2.5e6)
    assert math.isclose(mat.d, 1e-8)
    assert math.isclose(mat.lam, 6.0)
    assert mat.itype == 1
    assert mat.fct_id == 0
    assert math.isclose(mat.nu, 0.495)
    assert math.isclose(mat.fscale, 1.0)

    # Generic material table
    assert 101 in model.materials
    gen_mat = model.materials[101]
    assert gen_mat.law == 92
    assert math.isclose(gen_mat.rho0, 1200.0)


def test_deck_reader_arruda_boyce_free_format(tmp_path):
    """Verify parsing /MAT/ARRUDA_BOYCE card in free format."""
    deck_text = """/BEGIN
Test Free Format
/MAT/ARRUDA_BOYCE/202
Elastomer_Free
1150.0 0.0
1.8e6 2.0e-8 5.5
2 0 0.49 1.0
/END
"""
    deck_file = tmp_path / "deck_free.rad"
    deck_file.write_text(deck_text, encoding="utf-8")
    blocks = read_deck(str(deck_file))

    model = Model()
    log = MessageLog()
    for block in blocks:
        if block.key0 == "MAT":
            read_mat_law92(block, model, log)

    assert 202 in model.mat_law92s
    mat = model.mat_law92s[202]
    assert mat.id == 202
    assert math.isclose(mat.rho0, 1150.0)
    assert math.isclose(mat.mu, 1.8e6)
    assert math.isclose(mat.d, 2.0e-8)
    assert math.isclose(mat.lam, 5.5)
    assert mat.itype == 2
    assert mat.fct_id == 0


def test_deck_reader_arruda_boyce_synonyms(tmp_path):
    """Verify alternate keyword headings: /MAT/ARRUDA-BOYCE, /MAT/ARRUDA."""
    deck_text = """/BEGIN
Test Synonyms
/MAT/ARRUDA-BOYCE/301
Alias_Dash
1000.0
1.0e6 0.0 7.0
1 0 0.495 1.0
/END
"""
    deck_file = tmp_path / "deck_syn.rad"
    deck_file.write_text(deck_text, encoding="utf-8")
    blocks = read_deck(str(deck_file))

    model = Model()
    log = MessageLog()
    for block in blocks:
        if block.key0 == "MAT":
            read_mat_law92(block, model, log)

    assert 301 in model.mat_law92s
    assert math.isclose(model.mat_law92s[301].mu, 1.0e6)


# ============================================================================
# 3. Parameter Validation & Starter Checks
# ============================================================================

def test_check_mat_law92_valid():
    """Valid material passes without errors."""
    log = MessageLog()
    mat = MaterialLaw92(
        id=1,
        rho0=1000.0,
        mu=1.0e6,
        d=1e-7,
        lam=6.0,
        nu=0.495,
    )
    check_mat_law92(mat=mat, log=log)
    assert not log.has_errors


def test_check_mat_law92_zero_or_negative_density():
    """Zero or negative initial density triggers ANCMSG 1514 error."""
    log = MessageLog()
    mat = MaterialLaw92(
        id=2,
        rho0=0.0,
        mu=1.0e6,
        d=1e-7,
        lam=6.0,
    )
    check_mat_law92(mat=mat, log=log)
    assert log.has_errors
    assert any("ANCMSG 1514" in str(err) or "density" in str(err).lower() for err in log.errors)


def test_check_mat_law92_negative_d():
    """Negative compressibility D triggers error."""
    log = MessageLog()
    mat = MaterialLaw92(
        id=3,
        rho0=1000.0,
        mu=1.0e6,
        d=-1.0e-8,
        lam=6.0,
    )
    check_mat_law92(mat=mat, log=log)
    assert log.has_errors
    assert any("compressibility" in str(err).lower() or "d" in str(err).lower() for err in log.errors)


def test_check_mat_law92_invalid_nu():
    """NU >= 0.5 or NU < 0 triggers error."""
    log = MessageLog()
    mat = MaterialLaw92(
        id=4,
        rho0=1000.0,
        mu=1.0e6,
        d=1e-7,
        lam=6.0,
        nu=0.55,
    )
    check_mat_law92(mat=mat, log=log)
    assert log.has_errors
    assert any("poisson" in str(err).lower() or "nu" in str(err).lower() for err in log.errors)


def test_check_mat_law92_1d_element_incompatibility():
    """LAW92 assigned to 1D beam or truss element triggers ANCMSG 306 error."""
    model = Model()
    model.materials[10] = MaterialLaw92(
        id=10,
        rho0=1000.0,
        mu=1.0e6,
        d=1e-7,
        lam=7.0,
    )
    part = Part(id=1, prop_id=1, mat_id=10, title="BeamPart")
    part.elem_type = "BEAM"
    model.parts[1] = part

    log = MessageLog()
    check_mat_law92(model=model, mat_id=10, mat=model.materials[10], log=log)
    assert log.has_errors
    assert any("ANCMSG 306" in str(err) or "1d elements" in str(err).lower() for err in log.errors)


# ============================================================================
# 4. Curve Resolution & Curve Fitting
# ============================================================================

def test_starter_curve_resolution_arruda_boyce():
    """Verify that starter initialization fits experimental curve when fct_id > 0."""
    model = Model()
    curve_pts = [
        (1.05, 0.5e5),
        (1.10, 1.1e5),
        (1.20, 2.4e5),
        (1.30, 4.0e5),
        (1.40, 6.2e5),
        (1.50, 9.0e5),
    ]
    x_vals = [p[0] for p in curve_pts]
    y_vals = [p[1] for p in curve_pts]
    f = FunctTable(50, x_vals, y_vals, title="ExpTension")
    model.functions[50] = f

    mat = MaterialLaw92(
        id=1,
        rho0=1000.0,
        mu=0.0,
        d=0.0,
        lam=0.0,
        itype=1,  # Uniaxial test
        fct_id=50,
        nu=0.495,
        fscale=1.0,
    )
    model.materials[1] = mat
    model.mat_law92s[1] = mat

    log = MessageLog()
    mats.law92_arruda_boyce.resolve(mat, model, log)

    # Check that mu and lam were determined by fitting
    assert mat.mu > 0.0
    assert mat.lam > 1.0
    assert mat.params.get("mu", 0.0) > 0.0
    assert mat.params.get("lam", 0.0) > 1.0
