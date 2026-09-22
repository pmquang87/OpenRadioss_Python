"""Unit tests for Detonation Front Surfaces (/DFS/) and Explosive Burn Kinetics.

Tests:
1. Spherical wave arrival from DetPoint (/DFS/DETPOINT)
2. Planar wave arrival from DetPlan (/DFS/DETPLAN)
3. Line wave arrival from DetLine (/DFS/DETLINE)
4. Multi-detonator wave interaction in DetonationSystem
5. Burn fraction evolution and kinetics from 0.0 to 1.0
6. Vectorized batch calculation on multi-element coordinate arrays
"""

import math
import numpy as np
import pytest

from pyradioss.materials.detonation import (
    DetPoint,
    DetLine,
    DetPlan,
    DetonationSystem,
    arrival_time,
    burn_fraction,
    clamp,
)


def test_detpoint_spherical_wave():
    """Verify spherical detonation wave expansion from point source."""
    x0 = np.array([1.0, 2.0, 3.0])
    t0 = 1.0e-4
    d_cj = 8000.0  # m/s

    det = DetPoint(x0=x0, t0=t0, d_cj=d_cj)

    # At the point source, arrival time is t0
    assert math.isclose(det.arrival_time(x0), t0, rel_tol=1e-12)

    # Points at distance r = 8.0 m along all coordinate axes
    r = 8.0
    expected_t = t0 + r / d_cj  # 1e-4 + 8/8000 = 1e-4 + 1e-3 = 1.1e-3 s

    pts = np.array([
        [1.0 + r, 2.0, 3.0],
        [1.0 - r, 2.0, 3.0],
        [1.0, 2.0 + r, 3.0],
        [1.0, 2.0 - r, 3.0],
        [1.0, 2.0, 3.0 + r],
        [1.0, 2.0, 3.0 - r],
    ])

    t_arr = det.arrival_time(pts)
    assert t_arr.shape == (6,)
    np.testing.assert_allclose(t_arr, expected_t, rtol=1e-12)


def test_detplan_planar_wave():
    """Verify planar detonation wave propagation."""
    # Plane passing through (0, 0, 2.0) with normal along +Z
    origin = np.array([0.0, 0.0, 2.0])
    normal = np.array([0.0, 0.0, 1.0])
    t0 = 5.0e-5
    d_cj = 7000.0  # m/s

    det = DetPlan(origin=origin, normal=normal, t0=t0, d_cj=d_cj)

    # 1. On the plane (z = 2.0), arrival time is t0
    pt_on_plane = np.array([10.0, -25.0, 2.0])
    assert math.isclose(det.arrival_time(pt_on_plane), t0, rel_tol=1e-12)

    # 2. Ahead of the plane (z = 9.0), distance along normal is 7.0 m
    pt_ahead = np.array([5.0, 5.0, 9.0])
    expected_ahead = t0 + 7.0 / d_cj
    assert math.isclose(det.arrival_time(pt_ahead), expected_ahead, rel_tol=1e-12)

    # 3. Behind the plane (z = -1.0) with default ignite_behind=True
    pt_behind = np.array([0.0, 0.0, -1.0])
    assert math.isclose(det.arrival_time(pt_behind), t0, rel_tol=1e-12)

    # 4. Unnormalized normal vector handling
    det_unnorm = DetPlan.from_origin_normal(0.0, 0.0, 2.0, nx=0.0, ny=0.0, nz=5.0, t0=t0, d_cj=d_cj)
    assert math.isclose(float(np.linalg.norm(det_unnorm.normal)), 1.0, rel_tol=1e-12)
    assert math.isclose(det_unnorm.arrival_time(pt_ahead), expected_ahead, rel_tol=1e-12)


def test_detline_simultaneous_burn():
    """Verify line detonator wave with instantaneous line ignition."""
    # Segment from (0, 0, 0) to (10, 0, 0) along X axis
    p1 = [0.0, 0.0, 0.0]
    p2 = [10.0, 0.0, 0.0]
    t0 = 0.0
    d_cj = 6000.0

    det = DetLine.from_endpoints(p1, p2, t0=t0, d_cj=d_cj, v_line=None)

    # 1. Point alongside segment: (5, 3, 4) -> closest point on segment is (5, 0, 0)
    # Distance = sqrt(3^2 + 4^2) = 5.0 m
    pt_mid = np.array([5.0, 3.0, 4.0])
    assert math.isclose(det.arrival_time(pt_mid), 5.0 / d_cj, rel_tol=1e-12)

    # 2. Point past endpoint p2: (13, 0, 0) -> closest point is p2 = (10, 0, 0)
    # Distance = 3.0 m
    pt_past = np.array([13.0, 0.0, 0.0])
    assert math.isclose(det.arrival_time(pt_past), 3.0 / d_cj, rel_tol=1e-12)

    # 3. Point before endpoint p1: (-4, 0, 0) -> closest point is p1 = (0, 0, 0)
    # Distance = 4.0 m
    pt_before = np.array([-4.0, 0.0, 0.0])
    assert math.isclose(det.arrival_time(pt_before), 4.0 / d_cj, rel_tol=1e-12)


def test_detline_progressive_burn():
    """Verify line detonator wave with progressive burning velocity."""
    p1 = [0.0, 0.0, 0.0]
    p2 = [10.0, 0.0, 0.0]
    t0 = 0.0
    d_cj = 5000.0
    v_line = 5000.0

    det_prog = DetLine.from_endpoints(p1, p2, t0=t0, d_cj=d_cj, v_line=v_line)
    det_simul = DetLine.from_endpoints(p1, p2, t0=t0, d_cj=d_cj, v_line=None)

    # For any point, progressive ignition time must be >= simultaneous ignition time
    test_pts = np.array([
        [0.0, 2.0, 0.0],
        [5.0, 2.0, 0.0],
        [10.0, 2.0, 0.0],
        [15.0, 0.0, 0.0],
    ])

    t_prog = det_prog.arrival_time(test_pts)
    t_sim = det_simul.arrival_time(test_pts)

    assert np.all(t_prog >= t_sim - 1e-12)
    # At p1 (0, 0, 0), ignition is immediate at t0, so arrival times match
    assert math.isclose(det_prog.arrival_time([0.0, 0.0, 0.0]), 0.0, abs_tol=1e-12)


def test_multi_detonator_interaction():
    """Verify earliest wave arrival in a multi-detonator system."""
    sys = DetonationSystem(default_d_cj=8000.0)

    # Det1 at (-10, 0, 0) and Det2 at (+10, 0, 0), both igniting at t = 0
    sys.add_point(-10.0, 0.0, 0.0, t0=0.0, d_cj=8000.0, id=1)
    sys.add_point(+10.0, 0.0, 0.0, t0=0.0, d_cj=8000.0, id=2)

    # Point at (0, 0, 0): equidistance 10 m from both -> t_arr = 10 / 8000
    assert math.isclose(sys.arrival_time([0.0, 0.0, 0.0]), 10.0 / 8000.0, rel_tol=1e-12)

    # Point at (+8, 0, 0): 2 m from Det2, 18 m from Det1 -> Det2 arrives first
    expected_t2 = 2.0 / 8000.0
    assert math.isclose(sys.arrival_time([8.0, 0.0, 0.0]), expected_t2, rel_tol=1e-12)

    # Point at (-8, 0, 0): 2 m from Det1, 18 m from Det2 -> Det1 arrives first
    expected_t1 = 2.0 / 8000.0
    assert math.isclose(sys.arrival_time([-8.0, 0.0, 0.0]), expected_t1, rel_tol=1e-12)


def test_burn_fraction_evolution():
    """Verify burn fraction ramping from 0.0 to 1.0 matching OpenRadioss formula."""
    d_cj = 7500.0  # m/s
    dx = 0.01  # 10 mm element size
    dt_burn = 1.5 * dx / d_cj  # 2.0e-6 s
    t_arr = 1.0e-4

    # 1. Before wave arrival (t < t_arr)
    assert burn_fraction(t_arr - 1.0e-5, t_arr, dx, d_cj) == 0.0
    assert burn_fraction(t_arr, t_arr, dx, d_cj) == 0.0

    # 2. Intermediate burn: at half-way (t = t_arr + 0.5 * dt_burn)
    b_half = burn_fraction(t_arr + 0.5 * dt_burn, t_arr, dx, d_cj)
    assert math.isclose(b_half, 0.5, rel_tol=1e-12)

    # 3. Quarter burn
    b_quarter = burn_fraction(t_arr + 0.25 * dt_burn, t_arr, dx, d_cj)
    assert math.isclose(b_quarter, 0.25, rel_tol=1e-12)

    # 4. Complete burn (t >= t_arr + dt_burn)
    assert math.isclose(burn_fraction(t_arr + dt_burn, t_arr, dx, d_cj), 1.0, rel_tol=1e-12)
    assert burn_fraction(t_arr + 2.0 * dt_burn, t_arr, dx, d_cj) == 1.0


def test_vectorized_burn_fraction():
    """Verify vectorized array execution across multiple elements."""
    det = DetPoint.from_coords(0.0, 0.0, 0.0, t0=0.0, d_cj=8000.0)

    # 4 elements at radii 2, 4, 6, 8 m
    pts = np.array([
        [2.0, 0.0, 0.0],
        [4.0, 0.0, 0.0],
        [6.0, 0.0, 0.0],
        [8.0, 0.0, 0.0],
    ])
    # Arrival times: [2/8000, 4/8000, 6/8000, 8/8000] = [0.25e-3, 0.50e-3, 0.75e-3, 1.00e-3]
    t_arr = det.arrival_time(pts)

    # Test at t = 0.50e-3 s:
    # Elem 1: t > t_arr + dt_burn -> fully burnt (1.0)
    # Elem 2: t == t_arr -> just arriving (0.0)
    # Elem 3, 4: t < t_arr -> unburnt (0.0)
    dx = 0.01
    bfrac = det.burn_fraction(pts, t=0.50e-3, dx=dx)
    assert bfrac[0] == 1.0
    assert bfrac[1] == 0.0
    assert bfrac[2] == 0.0
    assert bfrac[3] == 0.0


def test_standalone_arrival_time_helper():
    """Verify arrival_time standalone dispatch helper function."""
    det1 = DetPoint.from_coords(0.0, 0.0, 0.0, t0=0.0, d_cj=5000.0)
    t1 = arrival_time(det1, [5.0, 0.0, 0.0])
    assert math.isclose(t1, 0.001, rel_tol=1e-12)

    det2 = DetPlan.from_origin_normal(0.0, 0.0, 0.0, nx=1.0, ny=0.0, nz=0.0, t0=0.0, d_cj=5000.0)
    t_min = arrival_time([det1, det2], [5.0, 0.0, 0.0])
    assert math.isclose(t_min, 0.001, rel_tol=1e-12)
