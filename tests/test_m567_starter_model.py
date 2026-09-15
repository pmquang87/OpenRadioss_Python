"""
Tests for Milestone M567: /MAT/LAW94 (/MAT/YEOH) Yeoh Hyperelastic Model
Starter, Model Entity, Deck Reader, Checks, & Element Compatibility.
"""

from __future__ import annotations

import math
import numpy as np
import pytest

from pyradioss.model.entities import (
    MaterialLaw94,
    MatLaw94,
    MatYeoh,
    Part,
    Property,
)
from pyradioss.model import Model
from pyradioss.input.deck_reader import read_deck
from pyradioss.input.starter_keywords import read_mat_law94
from pyradioss.starter.checks import (
    check_mat_law94,
    check_materials,
    check_model,
    MessageLog,
)
from pyradioss.input.checks import (
    check_mat_law94 as input_check_mat_law94,
    check_all,
)
import pyradioss.materials as mats
import pyradioss.starter.initialization as starter_init


# ============================================================================
# 1. Model Entity & Aliases
# ============================================================================

def test_material_law94_aliases_and_defaults():
    """Verify MaterialLaw94 class, aliases, and default attributes."""
    assert MatLaw94 is MaterialLaw94
    assert MatYeoh is MaterialLaw94

    mat = MaterialLaw94(id=10, title="Rubber_Yeoh")
    assert mat.id == 10
    assert mat.title == "Rubber_Yeoh"
    assert mat.law == 94
    assert mat.law_name == "LAW94"
    assert mat.rho0 == 0.0
    assert mat.c10 == 0.0
    assert mat.c20 == 0.0
    assert mat.c30 == 0.0
    assert mat.d1 == 0.0
    assert mat.d2 == 0.0
    assert mat.d3 == 0.0
    assert mat.nu == 0.495

    model = Model()
    assert hasattr(model, "mat_law94s")
    assert isinstance(model.mat_law94s, dict)
    assert hasattr(model, "mat_yeohs")
    assert model.mat_yeohs is model.mat_law94s


def test_material_law94_mapping_and_properties():
    """Verify dictionary-like mapping protocol and derived properties."""
    mat = MaterialLaw94(
        id=1,
        title="YeohPolymer",
        rho0=1100.0,
        c10=1.5e6,
        c20=-1.0e5,
        c30=1.0e4,
        d1=1.0e-8,
        d2=0.0,
        d3=0.0,
    )
    assert mat["id"] == 1
    assert mat["c10"] == 1.5e6
    assert "c10" in mat
    assert "rho0" in mat
    assert mat.get("c20") == -1.0e5
    assert mat.get("nonexistent", 99.0) == 99.0

    # Mutability via mapping protocol
    mat["c10"] = 2.0e6
    assert mat.c10 == 2.0e6

    # G0 = 2 * C10 = 4.0e6
    assert math.isclose(mat.G0, 4.0e6, rel_tol=1e-6)
    assert math.isclose(mat.G, 4.0e6, rel_tol=1e-6)

    # Bulk modulus K = 2 / D1 = 2 / 1e-8 = 2e8
    assert math.isclose(mat.K, 2.0e8, rel_tol=1e-6)
    assert math.isclose(mat.bulk, 2.0e8, rel_tol=1e-6)

    # Sound speed solid: c = sqrt((K + 4/3*G0) / rho)
    c_expected = math.sqrt((2.0e8 + (4.0 / 3.0) * 4.0e6) / 1100.0)
    assert math.isclose(float(mat.sound_speed), c_expected, rel_tol=1e-6)
    assert math.isclose(float(mat.sound_speed()), c_expected, rel_tol=1e-6)
    assert math.isclose(float(mat.sound_speed_solid), c_expected, rel_tol=1e-6)

    # Sound speed shell: c_shell = sqrt(E / ((1 - nu^2) * rho))
    c_shell_expected = math.sqrt(mat.E / ((1.0 - mat.nu_eff ** 2) * 1100.0))
    assert math.isclose(float(mat.sound_speed_shell), c_shell_expected, rel_tol=1e-6)


# ============================================================================
# 2. Deck Reader: Fixed and Free Format
# ============================================================================

def test_deck_reader_law94_fixed_format(tmp_path):
    """Verify parsing /MAT/LAW94 card in standard 20-character fixed format."""
    deck_text = """# OpenRadioss Starter Input Deck
/BEGIN
Test LAW94 Fixed
      2022         0
/MAT/LAW94/101
Yeoh_Rubber
#              RHO_I
              1200.0                 0.0
#
                 0.0                 0.0
#                C10                 C20                 C30
               2.5E6              -1.0E5               5.0E3
#                 D1                  D2                  D3
                1E-8                 0.0                 0.0
/END
"""
    deck_file = tmp_path / "deck_fixed.rad"
    deck_file.write_text(deck_text, encoding="utf-8")
    blocks = read_deck(str(deck_file))

    model = Model()
    log = MessageLog()
    for block in blocks:
        if block.key0 == "MAT":
            read_mat_law94(block, model, log)

    assert 101 in model.mat_law94s
    mat = model.mat_law94s[101]
    assert mat.id == 101
    assert "Yeoh_Rubber" in mat.title
    assert math.isclose(mat.rho0, 1200.0)
    assert math.isclose(mat.c10, 2.5e6)
    assert math.isclose(mat.c20, -1.0e5)
    assert math.isclose(mat.c30, 5.0e3)
    assert math.isclose(mat.d1, 1e-8)
    assert math.isclose(mat.d2, 0.0)
    assert math.isclose(mat.d3, 0.0)

    # Generic material table
    assert 101 in model.materials
    gen_mat = model.materials[101]
    assert gen_mat.law == 94
    assert math.isclose(gen_mat.rho0, 1200.0)


def test_deck_reader_yeoh_free_format(tmp_path):
    """Verify parsing /MAT/YEOH card in free format."""
    deck_text = """/BEGIN
Test Free Format
/MAT/YEOH/202
Elastomer_Yeoh_Free
1150.0
0.0
1.8e6 -5.0e4 2.0e3
2.0e-8 1.0e-9 0.0
/END
"""
    deck_file = tmp_path / "deck_free.rad"
    deck_file.write_text(deck_text, encoding="utf-8")
    blocks = read_deck(str(deck_file))

    model = Model()
    log = MessageLog()
    for block in blocks:
        if block.key0 == "MAT":
            read_mat_law94(block, model, log)

    assert 202 in model.mat_law94s
    mat = model.mat_law94s[202]
    assert mat.id == 202
    assert math.isclose(mat.rho0, 1150.0)
    assert math.isclose(mat.c10, 1.8e6)
    assert math.isclose(mat.c20, -5.0e4)
    assert math.isclose(mat.c30, 2.0e3)
    assert math.isclose(mat.d1, 2.0e-8)
    assert math.isclose(mat.d2, 1.0e-9)


def test_deck_reader_yeoh_synonyms(tmp_path):
    """Verify alternate keyword headings: /MAT/YEOH, /MAT/MAT_YEOH."""
    deck_text = """/BEGIN
Test Synonyms
/MAT/YEOH/301
Alias_Yeoh
1000.0
0.0
1.0e6 0.0 0.0
0.0 0.0 0.0
/END
"""
    deck_file = tmp_path / "deck_syn.rad"
    deck_file.write_text(deck_text, encoding="utf-8")
    blocks = read_deck(str(deck_file))

    model = Model()
    log = MessageLog()
    for block in blocks:
        if block.key0 == "MAT":
            read_mat_law94(block, model, log)

    assert 301 in model.mat_law94s
    assert math.isclose(model.mat_law94s[301].c10, 1.0e6)


# ============================================================================
# 3. Parameter Validation & Starter Checks
# ============================================================================

def test_check_mat_law94_valid():
    """Valid material passes without errors."""
    log = MessageLog()
    mat = MaterialLaw94(
        id=1,
        rho0=1000.0,
        c10=1.0e6,
        c20=0.0,
        c30=0.0,
        d1=1e-7,
        d2=0.0,
        d3=0.0,
    )
    check_mat_law94(mat=mat, log=log)
    assert not log.has_errors


def test_check_mat_law94_zero_or_negative_density():
    """Zero or negative initial density triggers ANCMSG 1514 error."""
    log = MessageLog()
    mat = MaterialLaw94(
        id=2,
        rho0=0.0,
        c10=1.0e6,
        d1=1e-7,
    )
    check_mat_law94(mat=mat, log=log)
    assert log.has_errors
    assert any("ANCMSG 1514" in str(err) or "density" in str(err).lower() for err in log.errors)


def test_check_mat_law94_negative_c10():
    """Zero or negative C10 emits warning."""
    log = MessageLog()
    mat = MaterialLaw94(
        id=3,
        rho0=1000.0,
        c10=-100.0,
        d1=1e-7,
    )
    check_mat_law94(mat=mat, log=log)
    assert any("c10" in str(w).lower() for w in log.warnings)


def test_check_mat_law94_negative_d():
    """Negative compressibility D parameters trigger error."""
    log = MessageLog()
    mat = MaterialLaw94(
        id=4,
        rho0=1000.0,
        c10=1.0e6,
        d1=-1.0e-8,
    )
    check_mat_law94(mat=mat, log=log)
    assert log.has_errors
    assert any("d1" in str(err).lower() for err in log.errors)


def test_check_mat_law94_invalid_nu():
    """NU >= 0.5 or NU < 0 triggers error."""
    log = MessageLog()
    mat = MaterialLaw94(
        id=5,
        rho0=1000.0,
        c10=1.0e6,
        d1=1e-7,
        nu=0.55,
    )
    check_mat_law94(mat=mat, log=log)
    assert log.has_errors
    assert any("nu" in str(err).lower() or "poisson" in str(err).lower() for err in log.errors)


def test_check_mat_law94_1d_element_incompatibility():
    """LAW94 assigned to 1D beam or truss element triggers ANCMSG 306 error."""
    model = Model()
    model.materials[10] = MaterialLaw94(
        id=10,
        rho0=1000.0,
        c10=1.0e6,
        d1=1e-7,
    )
    part = Part(id=1, prop_id=1, mat_id=10, title="BeamPart")
    part.elem_type = "BEAM"
    model.parts[1] = part

    log = MessageLog()
    check_mat_law94(model=model, mat_id=10, mat=model.materials[10], log=log)
    assert log.has_errors
    assert any("ANCMSG 306" in str(err) or "1d elements" in str(err).lower() for err in log.errors)


def test_starter_resolve_hook_yeoh():
    """Verify that starter resolve hook returns YeohParams."""
    model = Model()
    mat = MaterialLaw94(
        id=1,
        rho0=1000.0,
        c10=2.0e6,
        c20=1.0e5,
        c30=0.0,
        d1=1.0e-8,
    )
    log = MessageLog()
    p = mats.law94_yeoh.resolve(mat, model, log)
    assert isinstance(p, mats.law94_yeoh.YeohParams)
    assert math.isclose(p.c10, 2.0e6)
    assert math.isclose(p.g0, 4.0e6)
