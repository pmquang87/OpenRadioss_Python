"""Tests for /PROP/TYPE11 Sandwich Shell Property (SH_SANDW, SANDWICH).

Upstream Fortran Reference:
  - C:\\OpenRadioss\\source\\OpenRadioss-latest-20260520\\starter\\source\\properties\\shell\\hm_read_prop11.F
  - C:\\OpenRadioss\\source\\OpenRadioss-latest-20260520\\hm_cfg_files\\config\\CFG\\radioss2026\\PROP\\prop_p11_sh_sandw.cfg
"""

import math
import pytest

from pyradioss.common.messages import MessageLog
from pyradioss.input.deck_reader import Card, KeywordBlock
from pyradioss.input.prop_reader import (
    PROP_TYPE_NUMBERS,
    material_required,
    parse_property,
    parse_sandwich,
    parse_prop11_sandwich,
    prop_type_ok,
)
from pyradioss.input.prop_sandwich import (
    CallableFloat,
    Prop11Sandwich,
    SandwichLayer,
    parse_sandwich_card,
)
from pyradioss.model.entities import Property


# ============================================================================
# Test 1: Symmetric Sandwich Geometry and Layer Bounds
# ============================================================================

def test_symmetric_sandwich_geometry():
    """Verify total thickness, face sheet distance d, and z-bounds for symmetric sandwich."""
    skin1 = SandwichLayer(layer_id=1, name="skin1", thickness=0.001, mat_id=1, phi=0.0)
    core = SandwichLayer(layer_id=2, name="core", thickness=0.020, mat_id=2, phi=0.0)
    skin2 = SandwichLayer(layer_id=3, name="skin2", thickness=0.001, mat_id=1, phi=0.0)

    sandw = Prop11Sandwich(
        id=10,
        title="Symmetric Sandwich Panel",
        skin1=skin1,
        core=core,
        skin2=skin2,
    )

    # Thicknesses and distance d
    assert math.isclose(sandw.total_thickness, 0.022, rel_tol=1e-9)
    assert math.isclose(sandw.total_thickness(), 0.022, rel_tol=1e-9)
    assert math.isclose(sandw.core_thickness, 0.020, rel_tol=1e-9)
    assert math.isclose(sandw.core_thickness(), 0.020, rel_tol=1e-9)

    # d = t_core + (t1 + t2)/2 = 0.020 + 0.001 = 0.021
    assert math.isclose(sandw.face_sheet_distance_d, 0.021, rel_tol=1e-9)
    assert math.isclose(sandw.d, 0.021, rel_tol=1e-9)
    assert math.isclose(sandw.d(), 0.021, rel_tol=1e-9)

    # Through-thickness bounds relative to mid-surface [-0.011, +0.011]
    assert math.isclose(skin1.z_min, -0.011, rel_tol=1e-9)
    assert math.isclose(skin1.z_max, -0.010, rel_tol=1e-9)
    assert math.isclose(skin1.z_mid, -0.0105, rel_tol=1e-9)

    assert math.isclose(core.z_min, -0.010, rel_tol=1e-9)
    assert math.isclose(core.z_max, 0.010, rel_tol=1e-9)
    assert math.isclose(core.z_mid, 0.0, abs_tol=1e-12)

    assert math.isclose(skin2.z_min, 0.010, rel_tol=1e-9)
    assert math.isclose(skin2.z_max, 0.011, rel_tol=1e-9)
    assert math.isclose(skin2.z_mid, 0.0105, rel_tol=1e-9)

    # Mid-surface symmetry
    assert math.isclose(skin2.z_mid - skin1.z_mid, sandw.face_sheet_distance_d, rel_tol=1e-9)


# ============================================================================
# Test 2: Asymmetric Sandwich Geometry
# ============================================================================

def test_asymmetric_sandwich_geometry():
    """Verify thickness and coordinates for sandwich with unequal skin thicknesses."""
    skin1 = SandwichLayer(layer_id=1, name="skin1", thickness=0.002, mat_id=1)
    core = SandwichLayer(layer_id=2, name="core", thickness=0.020, mat_id=2)
    skin2 = SandwichLayer(layer_id=3, name="skin2", thickness=0.001, mat_id=3)

    sandw = Prop11Sandwich(
        id=20,
        skin1=skin1,
        core=core,
        skin2=skin2,
    )

    # h = 0.002 + 0.020 + 0.001 = 0.023
    assert math.isclose(sandw.total_thickness, 0.023, rel_tol=1e-9)
    # d = 0.020 + (0.002 + 0.001)/2 = 0.0215
    assert math.isclose(sandw.face_sheet_distance_d, 0.0215, rel_tol=1e-9)

    # Bounds: z in [-0.0115, +0.0115]
    assert math.isclose(skin1.z_min, -0.0115, rel_tol=1e-9)
    assert math.isclose(skin1.z_max, -0.0095, rel_tol=1e-9)
    assert math.isclose(skin1.z_mid, -0.0105, rel_tol=1e-9)

    assert math.isclose(core.z_min, -0.0095, rel_tol=1e-9)
    assert math.isclose(core.z_max, 0.0105, rel_tol=1e-9)
    assert math.isclose(core.z_mid, 0.0005, rel_tol=1e-9)

    assert math.isclose(skin2.z_min, 0.0105, rel_tol=1e-9)
    assert math.isclose(skin2.z_max, 0.0115, rel_tol=1e-9)
    assert math.isclose(skin2.z_mid, 0.0110, rel_tol=1e-9)

    # Centroid distance matches d exactly
    assert math.isclose(skin2.z_mid - skin1.z_mid, 0.0215, rel_tol=1e-9)


# ============================================================================
# Test 3: Analytical Bending and Transverse Shear Stiffness
# ============================================================================

def test_equivalent_bending_and_shear_stiffness():
    """Verify D_eff and G_eff against classical sandwich theory formulas."""
    t1 = 0.001
    tc = 0.020
    t2 = 0.001
    d = tc + 0.5 * (t1 + t2)  # 0.021 m

    e_s = 70.0e9  # Aluminum skins (70 GPa)
    e_core = 100.0e6  # Foam core (100 MPa)
    g_core = 40.0e6  # Foam core shear modulus (40 MPa)

    sandw = Prop11Sandwich(
        id=30,
        skin1=SandwichLayer(1, "skin1", t1, 1),
        core=SandwichLayer(2, "core", tc, 2),
        skin2=SandwichLayer(3, "skin2", t2, 1),
    )

    # 1. Classical thin-face-sheet bending stiffness:
    # D_approx = (E_s * t_s * d^2)/2 + (E_core * t_core^3)/12
    d_approx = (e_s * t1 * (d ** 2)) / 2.0 + (e_core * (tc ** 3)) / 12.0
    d_calc_thin = sandw.equivalent_bending_stiffness(e_s, e_s, e_core, include_skin_bending=False)
    assert math.isclose(d_calc_thin, d_approx, rel_tol=1e-9)

    # 2. Full bending stiffness with skin self-bending (E_s * t_s^3 / 6):
    d_full = d_approx + (e_s * (t1 ** 3)) / 6.0
    d_calc_full = sandw.equivalent_bending_stiffness(e_s, e_s, e_core, include_skin_bending=True)
    assert math.isclose(d_calc_full, d_full, rel_tol=1e-9)

    # 3. Transverse shear stiffness factor:
    # G_eff = G_core * d^2 / t_core
    g_expected = g_core * (d ** 2) / tc
    g_calc = sandw.equivalent_shear_stiffness(g_core)
    assert math.isclose(g_calc, g_expected, rel_tol=1e-9)


def test_asymmetric_bending_stiffness():
    """Verify D_eff for asymmetric sandwich with different skin moduli and thicknesses."""
    t1 = 0.002
    tc = 0.020
    t2 = 0.001

    e1 = 70.0e9   # 70 GPa
    e_core = 50.0e6  # 50 MPa
    e2 = 210.0e9  # 210 GPa (Steel)

    sandw = Prop11Sandwich(
        id=35,
        skin1=SandwichLayer(1, "skin1", t1, 1),
        core=SandwichLayer(2, "core", tc, 2),
        skin2=SandwichLayer(3, "skin2", t2, 3),
    )

    d_eff = sandw.equivalent_bending_stiffness(e1, e2, e_core)
    assert d_eff > 0.0

    # Test that steel skin (t2=0.001 with 210 GPa) produces high bending stiffness
    assert d_eff > 20000.0


# ============================================================================
# Test 4: get_layer_at_z
# ============================================================================

def test_get_layer_at_z():
    """Verify layer retrieval based on through-thickness position z."""
    sandw = Prop11Sandwich(
        id=40,
        skin1=SandwichLayer(1, "skin1", 0.001, 1),
        core=SandwichLayer(2, "core", 0.020, 2),
        skin2=SandwichLayer(3, "skin2", 0.001, 1),
    )
    # Total range: [-0.011, +0.011]
    # Skin 1: [-0.011, -0.010]
    # Core:   [-0.010, +0.010]
    # Skin 2: [+0.010, +0.011]

    # Skin 1 interior and lower boundary
    assert sandw.get_layer_at_z(-0.0105).name == "skin1"
    assert sandw.get_layer_at_z(-0.011).name == "skin1"
    assert sandw.get_layer_at_z(-0.020).name == "skin1"  # Below panel clamps to bottom

    # Core interior
    assert sandw.get_layer_at_z(0.0).name == "core"
    assert sandw.get_layer_at_z(-0.005).name == "core"
    assert sandw.get_layer_at_z(0.005).name == "core"

    # Skin 2 interior and upper boundary
    assert sandw.get_layer_at_z(0.0105).name == "skin2"
    assert sandw.get_layer_at_z(0.011).name == "skin2"
    assert sandw.get_layer_at_z(0.020).name == "skin2"  # Above panel clamps to top


# ============================================================================
# Test 5: Parsing /PROP/TYPE11 Keyword Block
# ============================================================================

def test_parse_prop_type11():
    """Parse /PROP/TYPE11 block into Property entity."""
    deck_lines = [
        "Sandwich shell test block",
        "#   Ishell    Ismstr     Ish3n    Idrill                            P_Thick_Fail",
        "        24         2         1         0                                     0.1",
        "#                 Hm                  Hf                  Hr                  Dm                  Dn",
        "                0.02                0.02                0.01                 0.0                 0.0",
        "#        N   Istrain               Thick              Ashear              Ithick     Iplas",
        "         3         1               0.022            0.833333                   0         1",
        "#                 Vx                  Vy                  Vz     Iskew     Iorth      Ipos        Ip",
        "                 1.0                 0.0                 0.0         0         0         0         0",
        "#                Phi               Thick                   Z         m     Ipply            F_weight",
        "                 0.0               0.001                 0.0         1         1                 1.0",
        "                 0.0               0.020                 0.0         2         1                 1.0",
        "                45.0               0.001                 0.0         3         2                 1.0",
    ]

    cards = [Card(line) for line in deck_lines]
    block = KeywordBlock(
        keyword="PROP/TYPE11",
        parts=["PROP", "TYPE11", "101"],
        user_id=101,
        cards=cards,
        source="test.rad:1",
    )

    log = MessageLog()
    prop = parse_sandwich_card(block, log)

    assert isinstance(prop, Property)
    assert prop.id == 101
    assert prop.type == 11
    assert prop.title == "Sandwich shell test block"

    p = prop.params
    assert math.isclose(p["thick"], 0.022, rel_tol=1e-9)
    assert p["ishell"] == 24
    assert p["ismstr"] == 2
    assert p["ish3n"] == 1
    assert p["n_layers"] == 3

    prop11 = p["prop11"]
    assert isinstance(prop11, Prop11Sandwich)
    assert math.isclose(prop11.skin1.thickness, 0.001)
    assert math.isclose(prop11.core.thickness, 0.020)
    assert math.isclose(prop11.skin2.thickness, 0.001)
    assert prop11.skin1.mat_id == 1
    assert prop11.core.mat_id == 2
    assert prop11.skin2.mat_id == 3
    assert math.isclose(prop11.skin2.phi, 45.0)
    assert prop11.skin2.nip == 2


# ============================================================================
# Test 6: Parsing /PROP/SANDWICH and /PROP/SH_SANDW
# ============================================================================

def test_parse_prop_sandwich_aliases():
    """Parse /PROP/SANDWICH and /PROP/SH_SANDW blocks."""
    deck_lines = [
        "Sandwich Alias Deck",
        "        24         2         1         0",
        "      0.01      0.01      0.01       0.0       0.0",
        "         3         1     0.012     0.833         0         1",
        "       1.0       0.0       0.0         0         0         0         0",
        "       0.0     0.001       0.0         1         1       1.0",
        "       0.0     0.010       0.0         2         1       1.0",
        "       0.0     0.001       0.0         1         1       1.0",
    ]

    log = MessageLog()

    # /PROP/SANDWICH/5
    cards1 = [Card(line) for line in deck_lines]
    block1 = KeywordBlock(
        keyword="PROP/SANDWICH",
        parts=["PROP", "SANDWICH", "5"],
        user_id=5,
        cards=cards1,
        source="test.rad:1",
    )
    prop1 = parse_property(block1, log)
    assert prop1 is not None
    assert prop1.id == 5
    assert prop1.type == 11
    assert math.isclose(prop1.params["thick"], 0.012, rel_tol=1e-4)

    # /PROP/SH_SANDW/7
    cards2 = [Card(line) for line in deck_lines]
    block2 = KeywordBlock(
        keyword="PROP/SH_SANDW",
        parts=["PROP", "SH_SANDW", "7"],
        user_id=7,
        cards=cards2,
        source="test.rad:20",
    )
    prop2 = parse_property(block2, log)
    assert prop2 is not None
    assert prop2.id == 7
    assert prop2.type == 11
    assert math.isclose(prop2.params["thick"], 0.012, rel_tol=1e-4)


# ============================================================================
# Test 7: Integration with prop_reader
# ============================================================================

def test_prop_reader_integration():
    """Verify registration in PROP_TYPE_NUMBERS and element compatibility."""
    assert PROP_TYPE_NUMBERS["TYPE11"] == 11
    assert PROP_TYPE_NUMBERS["SANDWICH"] == 11
    assert PROP_TYPE_NUMBERS["PROP_SANDWICH"] == 11
    assert PROP_TYPE_NUMBERS["SH_SANDW"] == 11
    assert PROP_TYPE_NUMBERS["PROP_SH_SANDW"] == 11

    # Material requirement
    assert material_required(11) is True

    # Element family compatibility: req_prop == 1 (SHELL) accepts TYPE11
    prop = Property(id=1, type=11, title="Sandwich", params={"thick": 0.02})
    assert prop_type_ok(req_prop=1, prop=prop) is True

    # parse_sandwich helper
    cards = [
        Card("Minimal Sandwich"),
        Card("0.0 0.001 0.0 1"),
        Card("0.0 0.020 0.0 2"),
        Card("0.0 0.001 0.0 1"),
    ]
    blk = KeywordBlock(
        keyword="PROP/TYPE11",
        parts=["PROP", "TYPE11", "99"],
        user_id=99,
        cards=cards,
        source="min.rad",
    )
    log = MessageLog()
    p = parse_sandwich(blk, log)
    assert p.id == 99
    assert p.type == 11
    assert math.isclose(p.params["thick"], 0.022, rel_tol=1e-9)


# ============================================================================
# Test 8: CallableFloat dual usage
# ============================================================================

def test_callable_float_property_and_method():
    """Verify CallableFloat behaves as both float attribute and method call."""
    cf = CallableFloat(42.5)

    # Float properties
    assert isinstance(cf, float)
    assert cf == 42.5
    assert cf + 7.5 == 50.0
    assert math.isclose(cf, 42.5)

    # Method call
    assert cf() == 42.5
    assert isinstance(cf(), float)
