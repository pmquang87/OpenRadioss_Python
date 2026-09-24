"""Unit tests for /PROP/TYPE20 (TSHELL Solid-Shell) and /PROP/TYPE21 (TSHELL Composite).

Upstream OpenRadioss Fortran references:
- starter/source/properties/thickshell/hm_read_prop20.F (SUBROUTINE HM_READ_PROP20)
- starter/source/properties/thickshell/hm_read_prop21.F (SUBROUTINE HM_PROP_READ21)
"""

from __future__ import annotations

import math
import numpy as np
import pytest

from pyradioss.common.messages import MessageLog
from pyradioss.elements.thickshell_props import (
    Prop20ThickShell,
    Prop21ThickShellComposite,
    ThickShellPly,
    legendre_gauss_1d,
)
from pyradioss.input.deck_reader import read_deck
from pyradioss.input.prop_reader import (
    PROP_TYPE_NUMBERS,
    parse_property,
)


# ============================================================================
# 1. Prop20ThickShell Dataclass & Parameter Parsing
# ============================================================================

def test_prop20_default_initialization():
    """Verify default parameters for /PROP/TYPE20 solid-shell."""
    p20 = Prop20ThickShell(id=1, title="TestTSHELL")

    assert p20.id == 1
    assert p20.title == "TestTSHELL"
    assert p20.isolid == 15
    assert p20.formulation == "ANS"
    assert p20.n_ip == 4  # 2x2 Gauss in-plane
    assert p20.n_thick == 3  # 3 Gauss points along thickness
    assert pytest.approx(p20.shear_corr, rel=1e-6) == 5.0 / 6.0
    assert p20.cvis == 0.1
    assert p20.qa == 1.1
    assert p20.qb == 0.05


def test_prop20_formulations():
    """Verify EAS, ANS, and REDUCED formulations."""
    # EAS formulation
    p_eas = Prop20ThickShell(id=2, formulation="EAS")
    assert p_eas.isolid == 16
    assert p_eas.formulation == "EAS"

    # Reduced integration with 1 point in-plane
    p_red = Prop20ThickShell(id=3, isolid=14)
    assert p_red.isolid == 14
    assert p_red.formulation == "REDUCED"
    assert p_red.n_ip == 1


def test_prop20_from_dict_and_nbp_parsing():
    """Verify parsing dictionary data with Radioss NBP packed integer notation."""
    # NBP = 225 means 2x2 in-plane, 5 through-thickness
    data = {
        "id": 10,
        "title": "TSHELL_5PTS",
        "isolid": 15,
        "nbp": 225,
        "Ashear": 0.85,
        "dn": 0.15,
        "thick": 2.5,
    }
    p20 = Prop20ThickShell.from_dict(data)

    assert p20.id == 10
    assert p20.inpts_r == 2
    assert p20.inpts_s == 2
    assert p20.inpts_t == 5
    assert p20.n_thick == 5
    assert p20.n_ip == 4
    assert pytest.approx(p20.shear_corr, rel=1e-6) == 0.85
    assert p20.cvis == 0.15
    assert p20.h == 2.5


# ============================================================================
# 2. Through-Thickness Gauss Points in [-1, +1]
# ============================================================================

def test_prop20_gauss_points_1pt():
    """1-point Gauss rule: center zeta = 0.0, weight = 2.0."""
    p20 = Prop20ThickShell(id=1, n_thick=1)
    zeta, w = p20.gauss_points_thickness()

    assert len(zeta) == 1
    assert pytest.approx(zeta[0], abs=1e-12) == 0.0
    assert pytest.approx(w[0], rel=1e-12) == 2.0


def test_prop20_gauss_points_2pt():
    """2-point Gauss rule: zeta = +- 1/sqrt(3), weights = 1.0."""
    p20 = Prop20ThickShell(id=1, n_thick=2)
    zeta, w = p20.gauss_points_thickness()

    assert len(zeta) == 2
    expected_c = 1.0 / math.sqrt(3.0)
    assert pytest.approx(zeta[0], rel=1e-9) == -expected_c
    assert pytest.approx(zeta[1], rel=1e-9) == expected_c
    assert pytest.approx(w[0], rel=1e-9) == 1.0
    assert pytest.approx(w[1], rel=1e-9) == 1.0
    assert pytest.approx(np.sum(w), rel=1e-12) == 2.0


def test_prop20_gauss_points_3pt():
    """3-point Gauss rule: zeta = [-sqrt(3/5), 0, +sqrt(3/5)], weights = [5/9, 8/9, 5/9]."""
    p20 = Prop20ThickShell(id=1, n_thick=3)
    zeta, w = p20.gauss_points_thickness()

    assert len(zeta) == 3
    c3 = math.sqrt(3.0 / 5.0)
    assert pytest.approx(zeta[0], rel=1e-9) == -c3
    assert pytest.approx(zeta[1], abs=1e-12) == 0.0
    assert pytest.approx(zeta[2], rel=1e-9) == c3
    assert pytest.approx(w[0], rel=1e-9) == 5.0 / 9.0
    assert pytest.approx(w[1], rel=1e-9) == 8.0 / 9.0
    assert pytest.approx(w[2], rel=1e-9) == 5.0 / 9.0
    assert pytest.approx(np.sum(w), rel=1e-12) == 2.0


def test_prop20_gauss_points_5pt():
    """5-point Gauss rule: check symmetry and sum of weights = 2.0."""
    p20 = Prop20ThickShell(id=1, n_thick=5)
    zeta, w = p20.gauss_points_thickness()

    assert len(zeta) == 5
    assert np.all(zeta >= -1.0) and np.all(zeta <= 1.0)
    assert np.all(np.diff(zeta) > 0.0)  # Monotonic
    assert pytest.approx(zeta[2], abs=1e-12) == 0.0  # Center point
    assert pytest.approx(zeta[0], rel=1e-9) == -zeta[4]  # Symmetry
    assert pytest.approx(zeta[1], rel=1e-9) == -zeta[3]
    assert pytest.approx(np.sum(w), rel=1e-12) == 2.0


def test_prop20_all_3d_integration_points():
    """Verify 3D integration points coordinates and total volume weight = 8.0."""
    p20 = Prop20ThickShell(id=1, n_ip=4, n_thick=3)
    pts = p20.all_integration_points()

    # 4 in-plane x 3 thickness = 12 points
    assert len(pts) == 12
    total_w = sum(p[3] for p in pts)
    # Total volume of reference cube [-1, 1]^3 is 2 * 2 * 2 = 8.0
    assert pytest.approx(total_w, rel=1e-12) == 8.0


# ============================================================================
# 3. Prop21ThickShellComposite Layup & Mapping
# ============================================================================

def test_prop21_composite_layup_mapping():
    """Verify through-thickness mapping of a 3-ply composite layup [0 / 45 / 90]."""
    p21 = Prop21ThickShellComposite(
        id=21,
        title="Composite_3Ply",
        vx=1.0, vy=0.0, vz=0.0,
    )

    # 3 plies: thickness 1.0, 2.0, 1.0 (total = 4.0), each with 2 Gauss points
    p21.add_ply(thick=1.0, angle=0.0, mat_id=101, nip=2)
    p21.add_ply(thick=2.0, angle=45.0, mat_id=102, nip=2)
    p21.add_ply(thick=1.0, angle=90.0, mat_id=101, nip=2)

    assert p21.total_thickness() == 4.0
    assert p21.num_plies() == 3
    assert p21.total_integration_points() == 6

    mapping = p21.thickness_mapping()
    assert len(mapping) == 6

    # Verify global normalized coordinates are in [-1, +1] and strictly increasing
    zetas = [pt["zeta"] for pt in mapping]
    assert np.all(np.array(zetas) >= -1.0)
    assert np.all(np.array(zetas) <= 1.0)
    assert np.all(np.diff(zetas) > 0.0)

    # Verify weights sum to exactly 2.0 across the full stack
    weights = [pt["weight"] for pt in mapping]
    assert pytest.approx(sum(weights), rel=1e-12) == 2.0

    # Verify layer 1 (ply_idx 0): z in [-2.0, -1.0] -> zeta in [-1.0, -0.5]
    assert mapping[0]["ply_idx"] == 0
    assert mapping[0]["angle"] == 0.0
    assert mapping[0]["mat_id"] == 101
    assert -2.0 <= mapping[0]["z"] <= -1.0
    assert -1.0 <= mapping[0]["zeta"] <= -0.5

    # Verify layer 2 (ply_idx 1): z in [-1.0, +1.0] -> zeta in [-0.5, +0.5]
    assert mapping[2]["ply_idx"] == 1
    assert mapping[2]["angle"] == 45.0
    assert mapping[2]["mat_id"] == 102
    assert -1.0 <= mapping[2]["z"] <= 1.0
    assert -0.5 <= mapping[2]["zeta"] <= 0.5

    # Verify layer 3 (ply_idx 2): z in [+1.0, +2.0] -> zeta in [+0.5, +1.0]
    assert mapping[4]["ply_idx"] == 2
    assert mapping[4]["angle"] == 90.0
    assert mapping[4]["mat_id"] == 101
    assert 1.0 <= mapping[4]["z"] <= 2.0
    assert 0.5 <= mapping[4]["zeta"] <= 1.0


def test_prop21_from_dict():
    """Verify Prop21ThickShellComposite construction from dictionary with layer list."""
    data = {
        "id": 22,
        "title": "CarbonEpoxy_Layup",
        "isolid": 15,
        "plies": [
            {"thick": 0.25, "angle": 0.0, "mat_id": 1, "nip": 1},
            {"thick": 0.25, "angle": 90.0, "mat_id": 1, "nip": 1},
            {"thick": 0.25, "angle": 90.0, "mat_id": 1, "nip": 1},
            {"thick": 0.25, "angle": 0.0, "mat_id": 1, "nip": 1},
        ],
    }
    p21 = Prop21ThickShellComposite.from_dict(data)

    assert p21.id == 22
    assert p21.num_plies() == 4
    assert pytest.approx(p21.total_thickness(), rel=1e-6) == 1.0
    assert p21.total_integration_points() == 4

    zetas, weights = p21.gauss_points_thickness()
    assert len(zetas) == 4
    assert pytest.approx(sum(weights), rel=1e-12) == 2.0


# ============================================================================
# 4. Prop Reader Registration & Keyword Block Parsing
# ============================================================================

def test_prop_reader_type_registration():
    """Verify TYPE20 and TYPE21 registration in PROP_TYPE_NUMBERS."""
    assert "TYPE20" in PROP_TYPE_NUMBERS
    assert PROP_TYPE_NUMBERS["TYPE20"] == 20

    assert "TYPE21" in PROP_TYPE_NUMBERS
    assert PROP_TYPE_NUMBERS["TYPE21"] == 21

    assert "TSHELL" in PROP_TYPE_NUMBERS
    assert PROP_TYPE_NUMBERS["TSHELL"] == 20

    assert "TSH_ORTH" in PROP_TYPE_NUMBERS
    assert PROP_TYPE_NUMBERS["TSH_ORTH"] == 21


def test_prop_reader_parse_type20_block():
    """Verify parsing /PROP/TYPE20 block through prop_reader.parse_property."""
    deck = """/PROP/TYPE20/101
Test Solid Shell
#  ISOLID  ISMSTR   ICSTR INPTS_R INPTS_S INPTS_T     NBP    IINT
       15       0       1       2       2       3     223       1
#      QA      QB    CVIS   DTMIN
      1.1    0.05     0.1     0.0
"""
    blocks = read_deck(deck)
    assert len(blocks) == 1

    log = MessageLog()
    prop = parse_property(blocks[0], log)

    assert prop is not None
    assert prop.id == 101
    assert prop.type == 20
    assert prop.params["npts_r"] == 2
    assert prop.params["npts_s"] == 2
    assert prop.params["npts_t"] == 3


def test_prop_reader_parse_type21_block():
    """Verify parsing /PROP/TYPE21 block through prop_reader.parse_property."""
    deck = """/PROP/TYPE21/202
Test Composite Thick Shell
#  ISOLID  ISMSTR   ICSTR     NBP SKEW_ID   IORTH
       15       0       1     223       0       0
#      VX      VY      VZ   ANGLE
      1.0     0.0     0.0    45.0
"""
    blocks = read_deck(deck)
    assert len(blocks) == 1

    log = MessageLog()
    prop = parse_property(blocks[0], log)

    assert prop is not None
    assert prop.id == 202
    assert prop.type == 21
    assert prop.params["isolid"] == 15
    assert prop.params["vx"] == 1.0
