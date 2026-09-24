"""
Unit tests for /STACK & /PLY composite ply layup model and Classical Lamination Theory (CLT).

Upstream Fortran references:
  - C:\\OpenRadioss\\source\\OpenRadioss-latest-20260520\\starter\\source\\properties\\composite_options\\stack\\lecstack_ply.F
  - starter/source/properties/composite_options/stack/preplyxfem.F
  - starter/source/stack/hm_read_stack.F
  - starter/source/elements/shell/coque/lcgeo19.F

Tests:
1. Symmetric laminate has zero extension-bending coupling matrix B = 0.
2. Cross-ply [0/90]_s ABD matrices match exact Classical Lamination Theory (CLT) formulas.
3. Angle-ply [+45/-45]_s ABD matrices match exact CLT formulas (including bend-twist coupling D16, D26).
4. Through-thickness integration point distribution (Gauss-Legendre, Gauss-Lobatto, Simpson).
5. Reference surface offsets (IPOS = 0, 3, 4) and parallel axis theorem consistency.
6. Quasi-isotropic laminate in-plane isotropy: Ex == Ey and Gxy == Ex / (2 * (1 + nu_xy)).
7. Builder flexibility (dict, tuple, PlyDefinition inputs).
"""

from __future__ import annotations

import math
import numpy as np
import pytest

from pyradioss.model.stack import (
    PlyDefinition,
    StackDefinition,
    build_stack,
    reduced_stiffness_matrix,
    rotate_reduced_stiffness,
)


# Standard carbon/epoxy unidirectional composite ply material properties:
# E1 = 140 GPa, E2 = 10 GPa, nu12 = 0.3, G12 = 5 GPa
MAT_CARBON_EPOXY = (140.0e9, 10.0e9, 0.3, 5.0e9)


# ============================================================================
# 1. Symmetric Laminate Zero Coupling Matrix B = 0
# ============================================================================

def test_symmetric_laminate_zero_coupling_b():
    """Verify that any symmetric laminate has zero coupling matrix B == 0."""
    t_ply = 0.25e-3  # 0.25 mm

    symmetric_layups = [
        [0.0, 90.0, 90.0, 0.0],                        # [0/90]_s
        [0.0, 45.0, -45.0, 90.0, 90.0, -45.0, 45.0, 0.0],  # [0/45/-45/90]_s
        [30.0, -30.0, -30.0, 30.0],                   # [30/-30]_s
        [0.0, 60.0, -60.0, -60.0, 60.0, 0.0],          # [0/60/-60]_s
    ]

    for angles in symmetric_layups:
        plies = [
            PlyDefinition(id=i + 1, mat_id=1, thickness=t_ply, angle=ang)
            for i, ang in enumerate(angles)
        ]
        stack = StackDefinition(id=1, plies=plies, ipos=0)

        assert stack.is_symmetric()

        a, b, d = stack.compute_abd(MAT_CARBON_EPOXY)

        # B matrix must be identically zero within floating point precision
        np.testing.assert_allclose(b, np.zeros((3, 3)), atol=1.0e-5, rtol=1.0e-5)


def test_asymmetric_laminate_nonzero_coupling_b():
    """Verify that an asymmetric laminate has non-zero coupling matrix B != 0."""
    t_ply = 0.25e-3
    # Asymmetric cross-ply [0/90]
    plies = [
        PlyDefinition(id=1, mat_id=1, thickness=t_ply, angle=0.0),
        PlyDefinition(id=2, mat_id=1, thickness=t_ply, angle=90.0),
    ]
    stack = StackDefinition(id=1, plies=plies, ipos=0)

    assert not stack.is_symmetric()

    a, b, d = stack.compute_abd(MAT_CARBON_EPOXY)

    # In [0/90], B11 = -B22 != 0
    assert abs(b[0, 0]) > 1.0e3
    assert abs(b[1, 1]) > 1.0e3
    assert b[0, 0] == pytest.approx(-b[1, 1], rel=1.0e-5)


# ============================================================================
# 2. Cross-Ply [0/90]_s CLT Analytical ABD Verification
# ============================================================================

def test_cross_ply_clt_exact_abd():
    """Verify cross-ply [0/90]_s ABD matrices match exact Classical Lamination Theory formulas."""
    e1, e2, nu12, g12 = MAT_CARBON_EPOXY
    t = 0.25e-3  # ply thickness
    h = 4.0 * t  # total thickness

    q = reduced_stiffness_matrix(e1, e2, nu12, g12)
    q11 = q[0, 0]
    q22 = q[1, 1]
    q12 = q[0, 1]
    q66 = q[2, 2]

    # Exact analytical equations for [0/90]_s:
    # Plies 1 & 4 at 0 deg, Plies 2 & 3 at 90 deg
    # Extensional stiffness A:
    # A11 = 2*t*Q11 + 2*t*Q22
    # A22 = 2*t*Q22 + 2*t*Q11 = A11
    # A12 = 4*t*Q12
    # A66 = 4*t*Q66
    a11_exact = 2.0 * t * (q11 + q22)
    a22_exact = a11_exact
    a12_exact = 4.0 * t * q12
    a66_exact = 4.0 * t * q66

    # Bending stiffness D:
    # Outer plies (0 deg) occupy z in [-2t, -t] and [t, 2t]: int z^2 dz = 2/3 * (8 - 1)*t^3 = 14/3 * t^3
    # Inner plies (90 deg) occupy z in [-t, t]: int z^2 dz = 2/3 * (1 - 0)*t^3 = 2/3 * t^3
    d11_exact = (14.0 / 3.0) * (t**3) * q11 + (2.0 / 3.0) * (t**3) * q22
    d22_exact = (14.0 / 3.0) * (t**3) * q22 + (2.0 / 3.0) * (t**3) * q11
    d12_exact = (16.0 / 3.0) * (t**3) * q12
    d66_exact = (16.0 / 3.0) * (t**3) * q66

    plies = [
        PlyDefinition(id=1, thickness=t, angle=0.0),
        PlyDefinition(id=2, thickness=t, angle=90.0),
        PlyDefinition(id=3, thickness=t, angle=90.0),
        PlyDefinition(id=4, thickness=t, angle=0.0),
    ]
    stack = StackDefinition(id=1, plies=plies, ipos=0)

    a, b, d = stack.compute_abd(MAT_CARBON_EPOXY)

    # A matrix checks
    assert a[0, 0] == pytest.approx(a11_exact, rel=1.0e-7)
    assert a[1, 1] == pytest.approx(a22_exact, rel=1.0e-7)
    assert a[0, 1] == pytest.approx(a12_exact, rel=1.0e-7)
    assert a[2, 2] == pytest.approx(a66_exact, rel=1.0e-7)
    assert a[0, 2] == pytest.approx(0.0, abs=1.0e-7)
    assert a[1, 2] == pytest.approx(0.0, abs=1.0e-7)

    # B matrix is zero
    np.testing.assert_allclose(b, 0.0, atol=1.0e-7)

    # D matrix checks
    assert d[0, 0] == pytest.approx(d11_exact, rel=1.0e-7)
    assert d[1, 1] == pytest.approx(d22_exact, rel=1.0e-7)
    assert d[0, 1] == pytest.approx(d12_exact, rel=1.0e-7)
    assert d[2, 2] == pytest.approx(d66_exact, rel=1.0e-7)
    assert d[0, 2] == pytest.approx(0.0, abs=1.0e-7)
    assert d[1, 2] == pytest.approx(0.0, abs=1.0e-7)

    # Fiber direction on surface gives D11 > D22
    assert d[0, 0] > d[1, 1]


# ============================================================================
# 3. Angle-Ply [+45/-45]_s CLT Analytical ABD Verification
# ============================================================================

def test_angle_ply_clt_exact_abd():
    """Verify angle-ply [+45/-45]_s ABD matrices match exact CLT formulas."""
    e1, e2, nu12, g12 = MAT_CARBON_EPOXY
    t = 0.25e-3

    q = reduced_stiffness_matrix(e1, e2, nu12, g12)
    q11 = q[0, 0]
    q22 = q[1, 1]
    q12 = q[0, 1]
    q66 = q[2, 2]

    # Transformed reduced stiffness for +45 deg:
    qbar_45 = rotate_reduced_stiffness(q, 45.0)

    # For [+45/-45]_s (4 plies: +45, -45, -45, +45):
    # A16 and A26 are zero because the laminate is balanced (equal +45 and -45 plies)
    # D16 and D26 are non-zero (bend-twist coupling):
    # Outer plies (+45): int z^2 dz = 14/3 * t^3
    # Inner plies (-45): int z^2 dz = 2/3 * t^3
    # D16 = (14/3 * t^3 - 2/3 * t^3) * Qbar16(+45) = 4 * t^3 * Qbar16(+45)
    # Since Qbar16(+45) = 1/4 * (Q11 - Q22), D16 = t^3 * (Q11 - Q22)
    d16_exact = (t**3) * (q11 - q22)

    plies = [
        PlyDefinition(id=1, thickness=t, angle=45.0),
        PlyDefinition(id=2, thickness=t, angle=-45.0),
        PlyDefinition(id=3, thickness=t, angle=-45.0),
        PlyDefinition(id=4, thickness=t, angle=45.0),
    ]
    stack = StackDefinition(id=1, plies=plies, ipos=0)

    assert stack.is_balanced()
    assert stack.is_symmetric()

    a, b, d = stack.compute_abd(MAT_CARBON_EPOXY)

    # Balanced in-plane: A16 = A26 = 0
    assert a[0, 2] == pytest.approx(0.0, abs=1.0e-7)
    assert a[1, 2] == pytest.approx(0.0, abs=1.0e-7)

    # Symmetric: B = 0
    np.testing.assert_allclose(b, 0.0, atol=1.0e-7)

    # Bend-twist coupling D16, D26 matches exact formula
    assert d[0, 2] == pytest.approx(d16_exact, rel=1.0e-7)
    assert d[1, 2] == pytest.approx(d16_exact, rel=1.0e-7)
    assert abs(d[0, 2]) > 0.0


# ============================================================================
# 4. Through-Thickness Integration Point Distribution
# ============================================================================

def test_through_thickness_integration_points_distribution():
    """Verify integration points coordinates and weights across plies."""
    t1, t2, t3 = 0.2e-3, 0.5e-3, 0.3e-3
    plies = [
        PlyDefinition(id=1, thickness=t1, angle=0.0, n_int=1),
        PlyDefinition(id=2, thickness=t2, angle=45.0, n_int=2),
        PlyDefinition(id=3, thickness=t3, angle=90.0, n_int=3),
    ]
    stack = StackDefinition(id=1, plies=plies, ipos=0)
    h_tot = t1 + t2 + t3

    # 1. Gauss-Legendre quadrature
    z_pts, weights, ply_ids = stack.integration_points(rule="gauss")

    assert len(z_pts) == 1 + 2 + 3  # 6 total integration points
    assert len(weights) == 6
    assert len(ply_ids) == 6

    # Total weight must equal total thickness h
    assert np.sum(weights) == pytest.approx(h_tot, rel=1.0e-12)

    # All points must be within [-h/2, +h/2]
    assert np.all(z_pts >= -0.5 * h_tot)
    assert np.all(z_pts <= 0.5 * h_tot)

    # Ply 1 (1 point, midpoint):
    z_mid1 = -0.5 * h_tot + 0.5 * t1
    assert z_pts[0] == pytest.approx(z_mid1)
    assert weights[0] == pytest.approx(t1)
    assert ply_ids[0] == 0

    # Ply 2 (2 Gauss points: z_mid +/- t2 / (2 * sqrt(3))):
    z_bot2 = -0.5 * h_tot + t1
    z_mid2 = z_bot2 + 0.5 * t2
    delta_gauss = t2 / (2.0 * math.sqrt(3.0))
    assert z_pts[1] == pytest.approx(z_mid2 - delta_gauss)
    assert z_pts[2] == pytest.approx(z_mid2 + delta_gauss)
    assert weights[1] == pytest.approx(t2 / 2.0)
    assert weights[2] == pytest.approx(t2 / 2.0)
    assert ply_ids[1] == 1 and ply_ids[2] == 1


def test_through_thickness_integration_points_lobatto():
    """Verify Gauss-Lobatto quadrature places boundary points on ply surfaces."""
    t_p = 1.0
    plies = [PlyDefinition(id=1, thickness=t_p, angle=0.0, n_int=2)]
    stack = StackDefinition(id=1, plies=plies, ipos=0)

    z_pts, weights, _ = stack.integration_points(rule="lobatto")

    # 2 Lobatto points on ply of thickness 1.0 centered at 0: -0.5 and +0.5
    assert z_pts[0] == pytest.approx(-0.5)
    assert z_pts[1] == pytest.approx(0.5)
    assert weights[0] == pytest.approx(0.5)
    assert weights[1] == pytest.approx(0.5)


# ============================================================================
# 5. Reference Surface Offset (IPOS = 0, 3, 4) & Parallel Axis Theorem
# ============================================================================

def test_reference_surface_offset_parallel_axis_theorem():
    """Verify offset of reference surface satisfies parallel axis theorem."""
    t_ply = 0.25e-3
    plies = [
        PlyDefinition(id=1, thickness=t_ply, angle=0.0),
        PlyDefinition(id=2, thickness=t_ply, angle=90.0),
        PlyDefinition(id=3, thickness=t_ply, angle=90.0),
        PlyDefinition(id=4, thickness=t_ply, angle=0.0),
    ]
    h = 4.0 * t_ply

    # 1. Mid-surface reference: ipos=0 (z in [-h/2, +h/2])
    stack_mid = StackDefinition(id=1, plies=plies, ipos=0)
    a_mid, b_mid, d_mid = stack_mid.compute_abd(MAT_CARBON_EPOXY)

    # 2. Bottom-surface reference: ipos=4 (z in [0, +h]) -> offset delta_z = +h/2
    stack_bot = StackDefinition(id=2, plies=plies, ipos=4)
    a_bot, b_bot, d_bot = stack_bot.compute_abd(MAT_CARBON_EPOXY)

    delta_z = 0.5 * h

    # Parallel axis theorem for laminate ABD:
    # A is independent of reference plane offset: A_bot == A_mid
    np.testing.assert_allclose(a_bot, a_mid, rtol=1.0e-10)

    # B_bot = B_mid + delta_z * A_mid = delta_z * A_mid (since B_mid = 0)
    expected_b_bot = delta_z * a_mid
    np.testing.assert_allclose(b_bot, expected_b_bot, rtol=1.0e-10)

    # D_bot = D_mid + 2 * delta_z * B_mid + delta_z^2 * A_mid = D_mid + delta_z^2 * A_mid
    expected_d_bot = d_mid + (delta_z**2) * a_mid
    np.testing.assert_allclose(d_bot, expected_d_bot, rtol=1.0e-10)


# ============================================================================
# 6. Quasi-Isotropic Laminate In-Plane Isotropy
# ============================================================================

def test_quasi_isotropic_laminate_in_plane_isotropy():
    """Verify quasi-isotropic laminate [0/90/+45/-45]_s is in-plane isotropic."""
    t = 0.125e-3
    plies = [
        PlyDefinition(id=1, thickness=t, angle=0.0),
        PlyDefinition(id=2, thickness=t, angle=90.0),
        PlyDefinition(id=3, thickness=t, angle=45.0),
        PlyDefinition(id=4, thickness=t, angle=-45.0),
        PlyDefinition(id=5, thickness=t, angle=-45.0),
        PlyDefinition(id=6, thickness=t, angle=45.0),
        PlyDefinition(id=7, thickness=t, angle=90.0),
        PlyDefinition(id=8, thickness=t, angle=0.0),
    ]
    stack = StackDefinition(id=1, plies=plies, ipos=0)

    eng = stack.effective_engineering_constants(MAT_CARBON_EPOXY)
    ex = eng["Ex"]
    ey = eng["Ey"]
    gxy = eng["Gxy"]
    nu_xy = eng["nu_xy"]

    # 1. Ex must equal Ey for quasi-isotropic
    assert ex == pytest.approx(ey, rel=1.0e-6)

    # 2. Isotropic relation G = E / (2 * (1 + nu)) must hold exactly
    expected_gxy = ex / (2.0 * (1.0 + nu_xy))
    assert gxy == pytest.approx(expected_gxy, rel=1.0e-5)


# ============================================================================
# 7. Builder Flexibility (Dict, Tuple, PlyDefinition)
# ============================================================================

def test_build_stack_convenience_builder():
    """Verify build_stack works with heterogeneous ply inputs."""
    stack = build_stack(
        id=52,
        title="WING_SKIN",
        plies=[
            (1, 10, 0.25, 0.0, 1),
            {"id": 2, "mat_id": 10, "thickness": 0.25, "angle": 45.0, "n_int": 2},
            PlyDefinition(id=3, mat_id=10, thickness=0.25, angle=-45.0, n_int=2),
            (4, 10, 0.25, 90.0),
        ],
        ipos=0,
    )

    assert stack.id == 52
    assert stack.title == "WING_SKIN"
    assert stack.num_plies == 4
    assert stack.total_thickness == pytest.approx(1.0)
    assert stack.num_integration_points == 1 + 2 + 2 + 1  # 6 points
    assert stack.plies[1].angle == 45.0
    assert stack.plies[2].angle == -45.0
