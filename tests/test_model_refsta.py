"""Unit tests for /REFSTA Reference State Geometry & Initial Strains.

Verifies:
  1. RefstaData dataclass with absolute X_ref and delta_x nodal mapping.
  2. Initial reference stress tensor and initial plastic strain retrieval.
  3. 3D deformation gradient computation and multiplicative composition:
     F_total = F(x_curr, X_0) * F(X_0, X_ref).
  4. Green-Lagrange strain tensor and Voigt vector relative to undeformed X_ref.
  5. compute_refsta_strains breakdown (pre-strain, total strain, increment).
  6. 2D membrane strain kinematics for formed shell elements.
  7. RefstaManager part/element lookup with mesh fallback.
  8. parse_refsta_card parsing OpenRadioss card format.
"""

from __future__ import annotations

import math
import numpy as np
import pytest

from pyradioss.model.refsta import (
    RefstaData,
    RefstaManager,
    compute_deformation_gradient_3d,
    compute_total_deformation_gradient,
    compute_green_lagrange_strain,
    compute_engineering_strain_from_ref,
    compute_refsta_strains,
    compute_membrane_refsta_strain,
    parse_refsta_card,
)


def test_refsta_data_node_and_delta_mapping():
    """Verify absolute X_ref and relative delta_x nodal coordinate resolution."""
    refsta = RefstaData(part_id=10, filename="blank_ref.rs0", nitrs=50)

    # Add absolute coordinates for node 1
    refsta.add_node(1, 10.0, 20.0, 30.0)
    coord_1 = refsta.get_reference_coord(1)
    assert np.allclose(coord_1, [10.0, 20.0, 30.0])

    # Add relative offset for node 2: delta = [1.0, -2.0, 0.5]
    refsta.add_delta_node(2, 1.0, -2.0, 0.5)
    x0_node2 = [100.0, 200.0, 300.0]
    coord_2 = refsta.get_reference_coord(2, x_0=x0_node2)
    assert np.allclose(coord_2, [101.0, 198.0, 300.5])

    # Unmapped node falls back to initial mesh coordinates x_0
    x0_node3 = [5.0, 6.0, 7.0]
    coord_3 = refsta.get_reference_coord(3, x_0=x0_node3)
    assert np.allclose(coord_3, [5.0, 6.0, 7.0])


def test_refsta_initial_stress_and_plastic_strain():
    """Verify retrieval of initial Cauchy stresses and equivalent plastic strains."""
    default_stress = np.array([50.0, 20.0, -10.0, 5.0, 0.0, 0.0], dtype=float)
    refsta = RefstaData(
        part_id=1,
        default_stress=default_stress,
        default_epsp=0.02,
    )

    # Element 101 has specific pre-stress from stamping
    elem101_stress = np.array([250.0, 180.0, 0.0, 45.0, 0.0, 0.0], dtype=float)
    refsta.set_initial_stress(101, elem101_stress)
    refsta.set_initial_plastic_strain(101, 0.15)

    # Specific element values
    assert np.allclose(refsta.get_initial_stress(101), elem101_stress)
    assert refsta.get_initial_plastic_strain(101) == pytest.approx(0.15)

    # Fallback to defaults for unlisted element 102
    assert np.allclose(refsta.get_initial_stress(102), default_stress)
    assert refsta.get_initial_plastic_strain(102) == pytest.approx(0.02)


def test_deformation_gradient_and_multiplicative_composition():
    """Verify F_total = F(x_curr, X_0) @ F(X_0, X_ref) for pre-deformed geometry."""
    # Undeformed unit cube at reference state X_ref
    x_ref = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [1.0, 1.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [1.0, 0.0, 1.0],
        [1.0, 1.0, 1.0],
        [0.0, 1.0, 1.0],
    ], dtype=float)

    # Initial FE mesh X_0 formed by pre-stretching: lambda_x = 1.2, lambda_y = 0.9, lambda_z = 1.0
    f_form_target = np.diag([1.2, 0.9, 1.0])
    x_0 = x_ref @ f_form_target.T

    # Dynamic run stretches further: lambda_x = 1.1, lambda_y = 1.05, lambda_z = 0.95
    f_dyn_target = np.diag([1.1, 1.05, 0.95])
    x_curr = x_0 @ f_dyn_target.T

    # Compute deformation gradients
    f_0_ref = compute_deformation_gradient_3d(x_0, x_ref)
    f_curr_0 = compute_deformation_gradient_3d(x_curr, x_0)
    f_total_direct = compute_deformation_gradient_3d(x_curr, x_ref)
    f_total_composed = compute_total_deformation_gradient(f_curr_0, f_0_ref)

    assert np.allclose(f_0_ref, f_form_target, atol=1.0e-5)
    assert np.allclose(f_curr_0, f_dyn_target, atol=1.0e-5)
    assert np.allclose(f_total_direct, f_total_composed, atol=1.0e-5)
    assert np.allclose(f_total_composed, f_dyn_target @ f_form_target, atol=1.0e-5)


def test_green_lagrange_strain_from_reference_state():
    """Verify Green-Lagrange strain calculation relative to X_ref."""
    # Pure uniaxial stretch: lambda = 1.2 -> E_xx = 0.5 * (1.2^2 - 1) = 0.22
    f_stretch = np.diag([1.2, 1.0, 1.0])
    e_voigt = compute_green_lagrange_strain(f_stretch, voigt=True)
    assert e_voigt[0] == pytest.approx(0.22, rel=1.0e-5)
    assert e_voigt[1] == pytest.approx(0.0, abs=1.0e-6)
    assert e_voigt[2] == pytest.approx(0.0, abs=1.0e-6)

    # Simple shear: gamma = 0.1 -> F = [[1, gamma, 0], [0, 1, 0], [0, 0, 1]]
    f_shear = np.array([
        [1.0, 0.1, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ], dtype=float)
    e_shear_voigt = compute_green_lagrange_strain(f_shear, voigt=True)
    # E_xy tensor component is 0.5 * 0.1 = 0.05 -> engineering shear gamma = 2 * E_xy = 0.1
    assert e_shear_voigt[3] == pytest.approx(0.1, rel=1.0e-5)
    # E_yy has second-order term 0.5 * (0.1^2) = 0.005
    assert e_shear_voigt[1] == pytest.approx(0.005, rel=1.0e-5)


def test_compute_refsta_strains_breakdown():
    """Verify complete breakdown: pre-strain, total strain, and dynamic increment."""
    x_ref = np.array([
        [0.0, 0.0, 0.0],
        [2.0, 0.0, 0.0],
        [2.0, 2.0, 0.0],
        [0.0, 2.0, 0.0],
        [0.0, 0.0, 2.0],
        [2.0, 0.0, 2.0],
        [2.0, 2.0, 2.0],
        [0.0, 2.0, 2.0],
    ], dtype=float)

    # Formed state: pre-strained by 10% in X
    x_0 = x_ref.copy()
    x_0[:, 0] *= 1.10

    # Current state: additional 5% in X
    x_curr = x_0.copy()
    x_curr[:, 0] *= 1.05

    strains = compute_refsta_strains(x_curr, x_0, x_ref)

    # Pre-strain: 0.5 * (1.10^2 - 1) = 0.105
    assert strains["E_pre"][0] == pytest.approx(0.105, rel=1.0e-4)

    # Total stretch: 1.10 * 1.05 = 1.155 -> Total strain = 0.5 * (1.155^2 - 1) = 0.1670125
    assert strains["E_total"][0] == pytest.approx(0.1670125, rel=1.0e-4)

    # Dynamic increment: E_total - E_pre
    expected_inc = 0.1670125 - 0.105
    assert strains["E_inc"][0] == pytest.approx(expected_inc, rel=1.0e-4)


def test_membrane_refsta_strain_for_shells():
    """Verify in-plane 2D membrane Green-Lagrange strain calculation for shells."""
    # 4-node quad shell reference geometry (100mm x 100mm flat blank)
    x_ref_2d = np.array([
        [0.0, 0.0],
        [100.0, 0.0],
        [100.0, 100.0],
        [0.0, 100.0],
    ], dtype=float)

    # Deformed shell: 8% elongation in X, 3% transverse contraction in Y
    x_curr_2d = np.array([
        [0.0, 0.0],
        [108.0, 0.0],
        [108.0, 97.0],
        [0.0, 97.0],
    ], dtype=float)

    f_2d, e_voigt = compute_membrane_refsta_strain(x_curr_2d, x_ref_2d)

    assert f_2d[0, 0] == pytest.approx(1.08, rel=1.0e-5)
    assert f_2d[1, 1] == pytest.approx(0.97, rel=1.0e-5)

    # E_xx = 0.5 * (1.08^2 - 1) = 0.0832
    assert e_voigt[0] == pytest.approx(0.5 * (1.08**2 - 1.0), rel=1.0e-5)
    # E_yy = 0.5 * (0.97^2 - 1) = -0.02955
    assert e_voigt[1] == pytest.approx(0.5 * (0.97**2 - 1.0), rel=1.0e-5)
    # Shear is zero
    assert e_voigt[2] == pytest.approx(0.0, abs=1.0e-6)


def test_refsta_manager_integration():
    """Verify RefstaManager part and element mappings with initial mesh fallback."""
    mgr = RefstaManager()

    # Part 1: Stamped sheet metal with reference geometry
    refsta_p1 = RefstaData(part_id=1)
    refsta_p1.add_node(11, 0.0, 0.0, 0.0)
    refsta_p1.add_node(12, 10.0, 0.0, 0.0)
    refsta_p1.add_node(13, 10.0, 10.0, 0.0)
    refsta_p1.add_node(14, 0.0, 10.0, 0.0)
    refsta_p1.set_initial_stress(201, [150.0, 50.0, 0.0, 20.0, 0.0, 0.0])
    refsta_p1.set_initial_plastic_strain(201, 0.08)
    mgr.register(refsta_p1)

    # Node coordinates lookup
    nodes = [11, 12, 13, 14]
    x0_coords = [
        [0.0, 0.0, 2.0],
        [10.2, 0.0, 2.0],
        [10.2, 10.1, 2.0],
        [0.0, 10.1, 2.0],
    ]
    xref_resolved = mgr.get_element_reference_coords(nodes, x0_coords, elem_id=201, part_id=1)
    # Returns true flat reference blank coordinates [z=0.0]
    assert np.allclose(xref_resolved[:, 2], 0.0)

    # Element state lookup
    sig0, epsp0 = mgr.get_element_initial_state(201, part_id=1)
    assert np.allclose(sig0[:4], [150.0, 50.0, 0.0, 20.0])
    assert epsp0 == pytest.approx(0.08)

    # Unmapped element returns zero stress and zero plastic strain
    sig_unmapped, epsp_unmapped = mgr.get_element_initial_state(999, part_id=99)
    assert np.allclose(sig_unmapped, 0.0)
    assert epsp_unmapped == 0.0


def test_parse_refsta_card():
    """Verify parsing OpenRadioss /REFSTA card format."""
    card_lines = [
        "/REFSTA",
        "# Node reference coordinates: NODE_ID  X  Y  Z",
        "         1        12.50000000        25.00000000        50.00000000",
        "         2        15.00000000        30.00000000        55.00000000",
        "         3        20.00000000        35.00000000        60.00000000",
    ]

    refsta = parse_refsta_card(card_lines)
    assert len(refsta.x_ref) == 3
    assert np.allclose(refsta.x_ref[1], [12.5, 25.0, 50.0])
    assert np.allclose(refsta.x_ref[2], [15.0, 30.0, 55.0])
    assert np.allclose(refsta.x_ref[3], [20.0, 35.0, 60.0])
