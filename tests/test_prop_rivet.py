"""
Unit tests for /PROP/TYPE5 & /PROP/RIVET Fastener / Rivet Connection Property.

Upstream Fortran reference:
  - C:\\OpenRadioss\\source\\OpenRadioss-latest-20260520\\starter\\source\\properties\\rivet\\hm_read_prop05.F
  - C:\\OpenRadioss\\source\\OpenRadioss-latest-20260520\\engine\\source\\elements\\rivet\\rivet1.F

Tests:
1. Prop5Rivet dataclass defaults and properties.
2. Quadratic interaction failure criterion evaluation (pure normal, pure shear, mixed).
3. Characteristic length elongation rupture (dist > DX).
4. 3D force vector decomposition and failure evaluation.
5. Parsing /PROP/TYPE5 standard 2-card format.
6. Parsing /PROP/RIVET 1-card format.
7. Integration with prop_reader (PROP_TYPE_NUMBERS, parse_property dispatch).
"""

from __future__ import annotations

import math
import numpy as np
import pytest

from pyradioss.input.deck_reader import Card, KeywordBlock
from pyradioss.input.prop_reader import PROP_TYPE_NUMBERS, parse_property
from pyradioss.input.prop_rivet import (
    Prop5Rivet,
    PropRivet,
    parse_prop_rivet,
    parse_prop_type5,
)
from pyradioss.model.entities import Property


# ============================================================================
# 1. Prop5Rivet Dataclass & Geometry Properties
# ============================================================================

def test_prop5_rivet_dataclass_properties():
    """Verify Prop5Rivet fields, aliases, and OpenRadioss GEO array representation."""
    rivet = Prop5Rivet(
        id=10,
        title="BUMPER_RIVET",
        fn=25000.0,
        ft=15000.0,
        dx=12.5,
        wflag=1,
        imod=1,
        mass=0.05,
        stiffness=1.0e6,
    )

    assert rivet.id == 10
    assert rivet.title == "BUMPER_RIVET"
    assert rivet.fn == 25000.0
    assert rivet.ft == 15000.0
    assert rivet.dx == 12.5
    assert rivet.wflag == 1
    assert rivet.imod == 1

    # Aliases
    assert rivet.nforce == 25000.0
    assert rivet.tforce == 15000.0
    assert rivet.length == 12.5
    assert rivet.irot == 1
    assert rivet.fn_fail == 25000.0
    assert rivet.ft_fail == 15000.0

    # Squared values
    assert rivet.fn2 == pytest.approx(25000.0**2)
    assert rivet.ft2 == pytest.approx(15000.0**2)
    assert rivet.dx2 == pytest.approx(12.5**2)

    # OpenRadioss GEO array matching hm_read_prop05.F
    geo = rivet.to_geo()
    assert geo[0] == pytest.approx(25000.0**2)
    assert geo[1] == pytest.approx(15000.0**2)
    assert geo[2] == pytest.approx(12.5**2)
    assert geo[3] == pytest.approx(1.1)  # WFLAG + 0.1
    assert geo[4] == pytest.approx(1.1)  # IMODE + 0.1


# ============================================================================
# 2. Quadratic Failure Criterion (rivet1.F line 198)
# ============================================================================

def test_quadratic_failure_pure_normal():
    """Verify failure evaluation under pure normal tensile force."""
    fn_max = 20000.0
    ft_max = 10000.0
    rivet = Prop5Rivet(fn=fn_max, ft=ft_max, dx=10.0)

    # Below failure
    failed, alpha = rivet.evaluate_failure(fn=10000.0, ft=0.0)
    assert not failed
    assert alpha == pytest.approx(0.5)

    # Exactly at failure
    failed, alpha = rivet.evaluate_failure(fn=20000.0, ft=0.0)
    assert failed
    assert alpha == pytest.approx(1.0)

    # Beyond failure
    failed, alpha = rivet.evaluate_failure(fn=25000.0, ft=0.0)
    assert failed
    assert alpha == pytest.approx(1.25)


def test_quadratic_failure_pure_shear():
    """Verify failure evaluation under pure shear force."""
    fn_max = 20000.0
    ft_max = 10000.0
    rivet = Prop5Rivet(fn=fn_max, ft=ft_max, dx=10.0)

    # Below failure
    failed, alpha = rivet.evaluate_failure(fn=0.0, ft=5000.0)
    assert not failed
    assert alpha == pytest.approx(0.5)

    # Exactly at failure
    failed, alpha = rivet.evaluate_failure(fn=0.0, ft=10000.0)
    assert failed
    assert alpha == pytest.approx(1.0)

    # Beyond failure
    failed, alpha = rivet.evaluate_failure(fn=0.0, ft=15000.0)
    assert failed
    assert alpha == pytest.approx(1.5)


def test_quadratic_failure_combined_interaction():
    """Verify quadratic interaction failure: (fn/FN)^2 + (ft/FT)^2 >= 1.0."""
    fn_max = 10000.0
    ft_max = 10000.0
    rivet = Prop5Rivet(fn=fn_max, ft=ft_max, dx=5.0)

    # (0.6)^2 + (0.8)^2 = 0.36 + 0.64 = 1.00 -> alpha = 1.0
    failed, alpha = rivet.evaluate_failure(fn=6000.0, ft=8000.0)
    assert failed
    assert alpha == pytest.approx(1.0)

    # (0.5)^2 + (0.5)^2 = 0.50 -> alpha = sqrt(0.5) ~ 0.7071
    failed, alpha = rivet.evaluate_failure(fn=5000.0, ft=5000.0)
    assert not failed
    assert alpha == pytest.approx(math.sqrt(0.5))

    # (0.8)^2 + (0.8)^2 = 1.28 -> alpha = sqrt(1.28) ~ 1.1314
    failed, alpha = rivet.evaluate_failure(fn=8000.0, ft=8000.0)
    assert failed
    assert alpha == pytest.approx(math.sqrt(1.28))


# ============================================================================
# 3. Elongation / Maximum Length Rupture (rivet1.F lines 104-112)
# ============================================================================

def test_characteristic_length_rupture():
    """Verify rupture when distance exceeds maximum characteristic length DX."""
    rivet = Prop5Rivet(fn=1.0e6, ft=1.0e6, dx=15.0)

    # Low forces (alpha << 1), but distance within DX
    failed, alpha = rivet.evaluate_failure(fn=100.0, ft=100.0, dist=14.9)
    assert not failed

    # Low forces (alpha << 1), but distance exceeds DX -> elongation rupture
    failed, alpha = rivet.evaluate_failure(fn=100.0, ft=100.0, dist=15.1)
    assert failed


# ============================================================================
# 4. 3D Force Vector Decomposition (rivet1.F lines 177-199)
# ============================================================================

def test_evaluate_force_vector_3d():
    """Verify 3D force decomposition into normal and shear components."""
    rivet = Prop5Rivet(fn=1000.0, ft=1000.0, dx=50.0)

    # Rivet aligned with Z axis: normal is [0, 0, 1]
    normal_axis = np.array([0.0, 0.0, 10.0])
    f_vec = np.array([300.0, 400.0, 600.0])  # Fn = 600, Ft = sqrt(300^2 + 400^2) = 500

    failed, alpha, fn, ft = rivet.evaluate_force_vector(f_vec, normal_axis)

    assert fn == pytest.approx(600.0)
    assert ft == pytest.approx(500.0)

    # alpha = sqrt((600/1000)^2 + (500/1000)^2) = sqrt(0.36 + 0.25) = sqrt(0.61) ~ 0.781
    expected_alpha = math.sqrt((600.0 / 1000.0) ** 2 + (500.0 / 1000.0) ** 2)
    assert alpha == pytest.approx(expected_alpha)
    assert not failed


# ============================================================================
# 5. Parsing /PROP/TYPE5 Standard 2-Card Format
# ============================================================================

def test_parse_prop_type5_two_card_format():
    """Verify parsing /PROP/TYPE5 with standard 2-card format (WFLAG IMOD / NFORCE TFORCE LENGTH)."""
    block = KeywordBlock(
        keyword="/PROP/TYPE5/42",
        parts=["PROP", "TYPE5", "42"],
        user_id=42,
        cards=[
            Card("       1         1"),                     # WFLAG=1, IMOD=1
            Card("   35000.0     18000.0        25.0"),   # FN=35000, FT=18000, DX=25.0
        ],
    )
    block.title = "STRUCTURAL_RIVET"

    prop = parse_prop_type5(block)

    assert isinstance(prop, Property)
    assert prop.id == 42
    assert prop.type == 5
    assert prop.title == "STRUCTURAL_RIVET"

    params = prop.params
    assert params["fn"] == pytest.approx(35000.0)
    assert params["ft"] == pytest.approx(18000.0)
    assert params["dx"] == pytest.approx(25.0)
    assert params["wflag"] == 1
    assert params["imod"] == 1

    # Attached Prop5Rivet object
    assert hasattr(prop, "rivet")
    assert isinstance(prop.rivet, Prop5Rivet)
    assert prop.rivet.fn == pytest.approx(35000.0)
    assert prop.rivet.ft == pytest.approx(18000.0)
    assert prop.rivet.dx == pytest.approx(25.0)


# ============================================================================
# 6. Parsing /PROP/RIVET 1-Card Format
# ============================================================================

def test_parse_prop_rivet_one_card_format():
    """Verify parsing /PROP/RIVET with single card containing 5 parameters."""
    block = KeywordBlock(
        keyword="/PROP/RIVET/88",
        parts=["PROP", "RIVET", "88"],
        user_id=88,
        cards=[
            Card("   22000.0   12000.0   8.5   0   1"),  # FN FT DX WFLAG IMOD
        ],
    )
    block.title = "DOOR_FASTENER"

    prop = parse_prop_rivet(block)

    assert prop.id == 88
    assert prop.type == 5
    assert prop.title == "DOOR_FASTENER"
    assert prop.params["fn"] == pytest.approx(22000.0)
    assert prop.params["ft"] == pytest.approx(12000.0)
    assert prop.params["dx"] == pytest.approx(8.5)
    assert prop.params["wflag"] == 0
    assert prop.params["imod"] == 1


# ============================================================================
# 7. Integration with prop_reader
# ============================================================================

def test_prop_reader_type5_registration_and_dispatch():
    """Verify PROP_TYPE_NUMBERS registers TYPE5 and parse_property dispatches correctly."""
    assert PROP_TYPE_NUMBERS["RIVET"] == 5
    assert PROP_TYPE_NUMBERS["TYPE5"] == 5

    block = KeywordBlock(
        keyword="/PROP/TYPE5/99",
        parts=["PROP", "TYPE5", "99"],
        user_id=99,
        cards=[
            Card("1 1"),
            Card("50000.0 30000.0 20.0"),
        ],
    )
    block.title = "FRAME_FASTENER"

    prop = parse_property(block, log=None)

    assert prop is not None
    assert prop.type == 5
    assert prop.id == 99
    assert prop.params["fn"] == pytest.approx(50000.0)
    assert prop.params["ft"] == pytest.approx(30000.0)
    assert prop.params["dx"] == pytest.approx(20.0)
