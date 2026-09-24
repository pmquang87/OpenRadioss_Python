"""Unit tests for /PROP/TYPE16 (user-defined composite layered shell property).

Upstream Fortran reference:
  - starter/source/properties/shell/hm_read_prop16.F
"""

import math
import numpy as np
import pytest

from pyradioss.common.messages import MessageLog
from pyradioss.input.deck_reader import Card, KeywordBlock
from pyradioss.input.prop_reader import PROP_TYPE_NUMBERS, parse_property
from pyradioss.input.prop_shell_type16 import (
    PropType16,
    PropType16Layer,
    parse_prop16,
    parse_prop_type16,
)


def test_prop_type16_layer_properties():
    """Verify PropType16Layer aliases and defaults."""
    layer = PropType16Layer(mat_id=3, thick=0.25, angle=45.0, z_pos=-0.5, weight=0.25, alpha=90.0)
    assert layer.mat_id == 3
    assert layer.thick == 0.25
    assert layer.phi == 45.0
    assert layer.z == -0.5
    assert layer.weight == 0.25
    assert layer.alpha == 90.0

    # Check setters
    layer.phi = -45.0
    assert layer.angle == -45.0
    layer.z = 0.5
    assert layer.z_pos == 0.5


def test_prop_type16_auto_layer_positioning():
    """Verify automatic through-thickness z coordinates when ipos=0."""
    # 4-layer laminate [0/90/90/0] of thickness 0.5 each -> total = 2.0
    layers = [
        PropType16Layer(mat_id=1, thick=0.5, angle=0.0),
        PropType16Layer(mat_id=1, thick=0.5, angle=90.0),
        PropType16Layer(mat_id=1, thick=0.5, angle=90.0),
        PropType16Layer(mat_id=1, thick=0.5, angle=0.0),
    ]
    prop = PropType16(prop_id=10, title="LAMINATE_AUTO", layers=layers, ipos=0)
    prop.compute_layer_positions()

    assert math.isclose(prop.total_thickness, 2.0)
    assert math.isclose(prop.zshift, 0.0)
    # Layer 1: z = -2.0/2 + 0.5/2 = -0.75
    assert math.isclose(prop.layers[0].z_pos, -0.75)
    # Layer 2: z = -0.75 + 0.5 = -0.25
    assert math.isclose(prop.layers[1].z_pos, -0.25)
    # Layer 3: z = -0.25 + 0.5 = 0.25
    assert math.isclose(prop.layers[2].z_pos, 0.25)
    # Layer 4: z = 0.25 + 0.5 = 0.75
    assert math.isclose(prop.layers[3].z_pos, 0.75)


def test_prop_type16_integration_points():
    """Verify get_integration_points returns coordinates, weights, and fiber angles."""
    layers = [
        PropType16Layer(mat_id=1, thick=1.0, angle=0.0, z_pos=-1.0, weight=1.0),
        PropType16Layer(mat_id=2, thick=1.0, angle=90.0, z_pos=0.0, weight=1.0),
        PropType16Layer(mat_id=1, thick=1.0, angle=45.0, z_pos=1.0, weight=1.0),
    ]
    prop = PropType16(prop_id=1, title="3PLY", layers=layers, total_thickness=3.0)
    z_coords, weights, mat_ids, angles = prop.get_integration_points()

    assert np.allclose(z_coords, [-1.0, 0.0, 1.0])
    assert np.allclose(weights, [1.0, 1.0, 1.0])
    assert np.array_equal(mat_ids, [1, 2, 1])
    assert np.allclose(angles, [0.0, 90.0, 45.0])


def test_parse_prop16_free_format():
    """Verify free-format card parsing for /PROP/TYPE16."""
    cards = [
        Card("24 4 0 0.0"),                       # Ishell Ismstr Ish3n p_thick_fail
        Card("0.01 0.01 0.01 0.0 0.0"),           # Hm Hf Hr Dm Dn
        Card("3 1 1.5 0.833333 0"),               # NIP ISTRAIN THICK ASHEAR ITHICK
        Card("1.0 0.0 0.0 0 0 0"),                # Vx Vy Vz Skew Ipos Ip
        Card("0.0 90.0 0.5 -0.5 1"),              # phi alpha thick z mat
        Card("90.0 90.0 0.5 0.0 2"),
        Card("0.0 90.0 0.5 0.5 1"),
    ]
    block = KeywordBlock(
        keyword="/PROP/TYPE16/10",
        parts=["PROP", "TYPE16", "10"],
        user_id=10,
        cards=cards,
    )
    log = MessageLog()
    prop = parse_prop16(block, log=log)

    assert prop.id == 10
    assert prop.type == 16
    assert prop.ishell == 24
    assert prop.nply == 3
    assert math.isclose(prop.total_thickness, 1.5)
    assert len(prop.layers) == 3
    assert prop.layers[0].mat_id == 1
    assert prop.layers[1].mat_id == 2
    assert prop.layers[2].mat_id == 1


def test_prop_reader_dispatch_type16():
    """Verify prop_reader dispatch for /PROP/TYPE16 and /PROP/SH_FABR."""
    assert PROP_TYPE_NUMBERS["TYPE16"] == 16
    assert PROP_TYPE_NUMBERS["SH_FABR"] == 16

    cards = [
        Card("24 2 1 0.0"),
        Card("0.01 0.01 0.01 0.0 0.0"),
        Card("2 1 1.0 0.833333 0"),
        Card("1.0 0.0 0.0 0 0 0"),
        Card("0.0 90.0 0.5 -0.25 1"),
        Card("90.0 90.0 0.5 0.25 2"),
    ]
    block = KeywordBlock(
        keyword="/PROP/TYPE16/42",
        parts=["PROP", "TYPE16", "42"],
        user_id=42,
        cards=cards,
    )
    log = MessageLog()
    prop = parse_property(block, log=log)
    assert prop is not None
    assert prop.type == 16
    assert prop.id == 42
    assert math.isclose(prop.params["thick"], 1.0)
