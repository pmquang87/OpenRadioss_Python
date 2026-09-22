"""Unit tests for /DRAPE composite fiber draping angles and thickness variation.

Tests fiber orientation update, trellising shear angle, compaction thickness
scaling, and integration with composite plies and shell element meshes.
"""

from __future__ import annotations

import math
import numpy as np
import pytest

from pyradioss.model.drape import (
    DrapeParams,
    DrapeTable,
    ElementDrape,
    apply_draping_to_ply,
    apply_draping_to_shell_mesh,
    compute_drape_shear_angle,
    compute_trellising_thickness,
)
from pyradioss.model.entities import Ply


def test_element_fiber_angle_modification():
    """Test element fiber orientation update from drape table."""
    table = DrapeTable(id=1, title="CFRP_Draping")

    # Element 101: warped by +15 degrees
    table.add_entry(elem_id=101, alpha_1=15.0, alpha_2=75.0)

    # Element 102: explicit delta_alpha = -10.0 degrees
    table.add_entry(elem_id=102, delta_alpha=-10.0)

    # Element 103: radians input (0.2 rad ~ 11.459 degrees)
    table.add_entry(elem_id=103, alpha_1=0.2, alpha_2=math.pi / 2, angle_unit="rad")

    # Nominal fiber angle = 45 degrees
    phi_nom = 45.0

    # Elem 101: 45 + 15 = 60 degrees
    phi_eff_101 = table.compute_effective_angle(101, phi_nom)
    assert math.isclose(phi_eff_101, 60.0, rel_tol=1e-12)

    # Elem 102: 45 - 10 = 35 degrees
    phi_eff_102 = table.compute_effective_angle(102, phi_nom)
    assert math.isclose(phi_eff_102, 35.0, rel_tol=1e-12)

    # Elem 103: 45 + rad2deg(0.2)
    phi_eff_103 = table.compute_effective_angle(103, phi_nom)
    assert math.isclose(phi_eff_103, 45.0 + math.degrees(0.2), rel_tol=1e-12)

    # Element without entry: retains nominal angle
    assert table.compute_effective_angle(999, phi_nom) == phi_nom


def test_trellising_shear_angle():
    """Test in-plane trellising shear angle theta_shear = |alpha_1 - alpha_2 - 90 deg|."""
    # Orthogonal fibers: alpha_1 = 0, alpha_2 = 90 -> shear angle = 0
    assert compute_drape_shear_angle(0.0, 90.0) == 0.0

    # Symmetrical trellising: alpha_1 = 15, alpha_2 = 75 (angle between fibers = 60 deg)
    # Shear angle = |15 - 75 - 90| = | -60 - 90 | = 30 deg
    shear_1 = compute_drape_shear_angle(15.0, 75.0)
    assert math.isclose(shear_1, 30.0, rel_tol=1e-12)

    # Shear angle = 60 degrees (angle between fibers = 30 deg)
    shear_2 = compute_drape_shear_angle(30.0, 60.0)
    assert math.isclose(shear_2, 60.0, rel_tol=1e-12)

    # Using ElementDrape property
    ed = ElementDrape(elem_id=1, alpha_1=15.0, alpha_2=75.0)
    assert math.isclose(ed.shear_angle_deg, 30.0, rel_tol=1e-12)
    assert math.isclose(ed.shear_angle_rad, math.radians(30.0), rel_tol=1e-12)


def test_thickness_variation_and_trellising_thinning():
    """Test thickness variation from trellising shear compaction t = t0 / cos(theta_shear)."""
    t0 = 2.0  # mm nominal thickness

    # 1. Zero shear: t_eff = t0
    t_eff_0 = compute_trellising_thickness(t0, theta_shear=0.0)
    assert math.isclose(t_eff_0, t0, rel_tol=1e-12)

    # 2. 60 degrees shear: cos(60) = 0.5 -> t_eff = t0 / 0.5 = 2.0 * t0 = 4.0 mm
    t_eff_60 = compute_trellising_thickness(t0, theta_shear=60.0)
    assert math.isclose(t_eff_60, 4.0, rel_tol=1e-12)

    # 3. 30 degrees shear: cos(30) = sqrt(3)/2 -> t_eff = 2.0 / (sqrt(3)/2)
    t_eff_30 = compute_trellising_thickness(t0, theta_shear=30.0)
    assert math.isclose(t_eff_30, 2.0 / math.cos(math.radians(30.0)), rel_tol=1e-12)

    # 4. Table evaluation with priority:
    table = DrapeTable(id=1)

    # Element 1: computed from trellising shear (alpha_1=30, alpha_2=60 -> shear=60 -> 2x t0)
    table.add_entry(elem_id=1, alpha_1=30.0, alpha_2=60.0)
    assert math.isclose(table.compute_effective_thickness(1, t0), 4.0, rel_tol=1e-12)

    # Element 2: explicit thinning factor 0.8 -> t = 1.6 mm
    table.add_entry(elem_id=2, thinning_factor=0.8)
    assert math.isclose(table.compute_effective_thickness(2, t0), 1.6, rel_tol=1e-12)

    # Element 3: explicit thickness 3.5 mm overrides everything
    table.add_entry(elem_id=3, thickness=3.5, thinning_factor=0.8, alpha_1=30.0, alpha_2=60.0)
    assert math.isclose(table.compute_effective_thickness(3, t0), 3.5, rel_tol=1e-12)

    # Element 4: no entry -> nominal thickness
    assert table.compute_effective_thickness(4, t0) == t0


def test_integration_with_composite_ply_and_shell():
    """Test applying draping to a Ply entity and shell mesh."""
    ply = Ply(
        id=1,
        mat_id=10,
        thick=1.5,
        orientangle=30.0,
    )

    drape = DrapeTable(id=100)
    drape.add_entry(
        elem_id=50,
        alpha_1=15.0,
        alpha_2=75.0,  # 30 deg shear
    )

    # Apply draping to ply for element 50
    updated_ply = apply_draping_to_ply(ply, drape, elem_id=50)

    # Effective angle: 30 + 15 = 45 degrees
    assert math.isclose(updated_ply.orientangle, 45.0, rel_tol=1e-12)

    # Effective thickness: 1.5 / cos(30 deg)
    expected_thick = 1.5 / math.cos(math.radians(30.0))
    assert math.isclose(updated_ply.thick, expected_thick, rel_tol=1e-12)


def test_batch_shell_mesh_draping_update():
    """Test batch update of shell mesh thicknesses matching `shellthk_upd.F`."""
    elements = {
        1: {"id": 1, "thick": 2.0},
        2: {"id": 2, "thick": 2.0},
        3: {"id": 3, "thick": 2.0},
    }

    drape = DrapeTable(id=1)
    # Elem 1: 60 deg shear -> 4.0 mm
    drape.add_entry(elem_id=1, alpha_1=30.0, alpha_2=60.0)
    # Elem 2: thinning factor 0.75 -> 1.5 mm
    drape.add_entry(elem_id=2, thinning_factor=0.75)
    # Elem 3: no entry -> stays 2.0 mm

    thk_map = apply_draping_to_shell_mesh(elements, drape)

    assert math.isclose(thk_map[1], 4.0, rel_tol=1e-12)
    assert math.isclose(thk_map[2], 1.5, rel_tol=1e-12)
    assert math.isclose(thk_map[3], 2.0, rel_tol=1e-12)

    # Check that element dictionaries were modified in-place
    assert math.isclose(elements[1]["thick"], 4.0, rel_tol=1e-12)
    assert math.isclose(elements[2]["thick"], 1.5, rel_tol=1e-12)
    assert math.isclose(elements[3]["thick"], 2.0, rel_tol=1e-12)


def test_model_init_exports():
    """Verify clean top-level imports from pyradioss.model."""
    import pyradioss.model as pm

    assert hasattr(pm, "DrapeParams")
    assert hasattr(pm, "DrapeTable")
    assert hasattr(pm, "ElementDrape")
    assert hasattr(pm, "apply_draping_to_ply")
    assert hasattr(pm, "apply_draping_to_shell_mesh")
    assert hasattr(pm, "compute_drape_shear_angle")
    assert hasattr(pm, "compute_trellising_thickness")
