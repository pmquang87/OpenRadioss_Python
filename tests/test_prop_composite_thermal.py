"""Unit tests for /PROP/TYPE17 (Composite Shell) and /PROP/TYPE19 (Thermal Shell).

Tests multi-layer stacking sequence, through-thickness coordinate integration,
thermal conduction parameters, and prop_reader registration.
"""

from __future__ import annotations

import math
import pytest

from pyradioss.common.messages import MessageLog
from pyradioss.input.deck_reader import Card, KeywordBlock
from pyradioss.input.prop_composite import (
    Prop17CompositeShell,
    Prop17Layer,
    Prop19ThermalShell,
    parse_prop17_composite,
    parse_prop19_thermal,
)
from pyradioss.input.prop_reader import (
    PROP_TYPE_NUMBERS,
    parse_property,
    prop_type_ok,
)


def test_prop17_composite_shell_dataclass():
    """Test Prop17CompositeShell construction and through-thickness z-coordinate calculation."""
    comp = Prop17CompositeShell(id=1, title="CFRP_QuasiIsotropic")

    # 4 plies: [0 / 45 / -45 / 90], each 0.5 mm thick, 2 Gauss points each
    comp.add_layer(thick=0.5, phi=0.0, mat_id=10, nip=2)
    comp.add_layer(thick=0.5, phi=45.0, mat_id=10, nip=2)
    comp.add_layer(thick=0.5, phi=-45.0, mat_id=10, nip=2)
    comp.add_layer(thick=0.5, phi=90.0, mat_id=10, nip=2)

    assert comp.n_layers == 4
    assert math.isclose(comp.total_thickness, 2.0, rel_tol=1e-12)

    # Layer 1: [-1.0, -0.5], mid = -0.75
    l1 = comp.get_layer(1)
    assert math.isclose(l1.z_lower, -1.0, rel_tol=1e-12)
    assert math.isclose(l1.z_upper, -0.5, rel_tol=1e-12)
    assert math.isclose(l1.z_mid, -0.75, rel_tol=1e-12)
    assert l1.nip == 2
    assert len(l1.gauss_z) == 2
    offset = 0.5 / (2.0 * math.sqrt(3.0))
    assert math.isclose(l1.gauss_z[0], -0.75 - offset, rel_tol=1e-12)
    assert math.isclose(l1.gauss_z[1], -0.75 + offset, rel_tol=1e-12)

    # Layer 4: [0.5, 1.0], mid = 0.75, phi = 90 deg
    l4 = comp.get_layer(4)
    assert math.isclose(l4.z_lower, 0.5, rel_tol=1e-12)
    assert math.isclose(l4.z_upper, 1.0, rel_tol=1e-12)
    assert math.isclose(l4.z_mid, 0.75, rel_tol=1e-12)
    assert math.isclose(l4.phi, 90.0, rel_tol=1e-12)

    # Test with custom reference plane offset Z0 = 0.0 (bottom reference)
    comp_bottom_ref = Prop17CompositeShell(id=2, z0=0.0)
    comp_bottom_ref.add_layer(thick=1.0, phi=0.0, mat_id=1, nip=1)
    comp_bottom_ref.add_layer(thick=1.0, phi=90.0, mat_id=1, nip=1)
    assert math.isclose(comp_bottom_ref.get_layer(1).z_lower, 0.0, rel_tol=1e-12)
    assert math.isclose(comp_bottom_ref.get_layer(1).z_mid, 0.5, rel_tol=1e-12)
    assert math.isclose(comp_bottom_ref.get_layer(2).z_upper, 2.0, rel_tol=1e-12)


def test_prop19_thermal_shell_dataclass():
    """Test Prop19ThermalShell construction and temperature integration points."""
    tshell = Prop19ThermalShell(
        id=10,
        title="Thermal_Heat_Shield",
        thick=3.0,
        k_th=45.0,
        c_p=480.0,
        alpha_th=1.1e-5,
        n_temp=5,
    )

    assert math.isclose(tshell.thick, 3.0, rel_tol=1e-12)
    assert math.isclose(tshell.k_th, 45.0, rel_tol=1e-12)
    assert math.isclose(tshell.c_p, 480.0, rel_tol=1e-12)
    assert math.isclose(tshell.alpha_th, 1.1e-5, rel_tol=1e-12)
    assert tshell.n_temp == 5

    # 5 through-thickness temperature points across [-1.5, +1.5]
    pts = tshell.temperature_points()
    assert len(pts) == 5
    expected = [-1.5, -0.75, 0.0, 0.75, 1.5]
    for p, exp in zip(pts, expected):
        assert math.isclose(p, exp, rel_tol=1e-12)


def test_parse_prop17_from_deck():
    """Test parsing /PROP/TYPE17 deck block with stacking sequence."""
    raw_lines = [
        "CFRP_Stack_16Layers",
        "         1         2         2         0         0       0.0       0.0",
        "       0.0       0.0       0.0       0.0       0.0",
        "       0.0       0.0    0.8333         0         1",
        "       0.0       0.0       0.0         0         0         0         0",
        "0.25 0.0 1 2",
        "0.25 45.0 1 2",
        "0.25 -45.0 1 2",
        "0.25 90.0 1 2",
        "0.25 90.0 1 2",
        "0.25 -45.0 1 2",
        "0.25 45.0 1 2",
        "0.25 0.0 1 2",
    ]

    cards = [Card(line, idx + 1) for idx, line in enumerate(raw_lines)]
    block = KeywordBlock(
        keyword="PROP/TYPE17",
        user_id=101,
        parts=["PROP", "TYPE17", "101"],
        cards=cards,
    )

    log = MessageLog()
    prop = parse_property(block, log)

    assert prop is not None
    assert prop.id == 101
    assert prop.type == 17
    assert prop.title == "CFRP_Stack_16Layers"

    params = prop.params
    assert "prop17" in params
    comp: Prop17CompositeShell = params["prop17"]

    # 8 layers of 0.25 mm = 2.0 mm total
    assert comp.n_layers == 8
    assert math.isclose(comp.total_thickness, 2.0, rel_tol=1e-12)
    assert math.isclose(params["thick"], 2.0, rel_tol=1e-12)

    # Check angles
    assert math.isclose(comp.get_layer(1).phi, 0.0, rel_tol=1e-12)
    assert math.isclose(comp.get_layer(2).phi, 45.0, rel_tol=1e-12)
    assert math.isclose(comp.get_layer(3).phi, -45.0, rel_tol=1e-12)
    assert math.isclose(comp.get_layer(4).phi, 90.0, rel_tol=1e-12)

    # Check shell compatibility
    assert prop_type_ok(1, prop) is True


def test_parse_prop19_thermal_from_deck():
    """Test parsing /PROP/THERM_SHELL deck block."""
    raw_lines = [
        "Inconel_Thermal_Skin",
        "         5       2.5       0.0         0         0         5",
        "      25.0     450.0   1.3e-05",
    ]

    cards = [Card(line, idx + 1) for idx, line in enumerate(raw_lines)]
    block = KeywordBlock(
        keyword="PROP/THERM_SHELL",
        user_id=202,
        parts=["PROP", "THERM_SHELL", "202"],
        cards=cards,
    )

    log = MessageLog()
    prop = parse_property(block, log)

    assert prop is not None
    assert prop.id == 202
    assert prop.type == 19
    assert prop.title == "Inconel_Thermal_Skin"

    params = prop.params
    assert "prop19" in params
    tshell: Prop19ThermalShell = params["prop19"]

    assert math.isclose(tshell.thick, 2.5, rel_tol=1e-12)
    assert math.isclose(tshell.k_th, 25.0, rel_tol=1e-12)
    assert math.isclose(tshell.c_p, 450.0, rel_tol=1e-12)
    assert math.isclose(tshell.alpha_th, 1.3e-5, rel_tol=1e-12)
    assert tshell.n_temp == 5
    assert len(params["z_coords"]) == 5

    assert prop_type_ok(1, prop) is True


def test_prop_reader_registration():
    """Verify TYPE17 and TYPE19 registration in PROP_TYPE_NUMBERS."""
    assert PROP_TYPE_NUMBERS.get("TYPE17") == 17
    assert PROP_TYPE_NUMBERS.get("STACK") == 17
    assert PROP_TYPE_NUMBERS.get("SH_COMP") == 17
    assert PROP_TYPE_NUMBERS.get("THERM_SHELL") == 19
    assert PROP_TYPE_NUMBERS.get("SH_THERM") == 19
