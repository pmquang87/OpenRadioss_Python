"""Unit tests for /BEM/DAA & /BEM/FLOW (Doubly Asymptotic Approximation & Boundary Acoustics).

Fortran origins:
- engine/source/fluid/daasolv.F
- engine/source/fluid/bemsolv.F
- starter/source/loads/bem/hm_read_bem.F
"""

from __future__ import annotations

import math
import numpy as np
import pytest

from pyradioss.engine import daa


# ============================================================================
# 1. Acoustic Radiation Damping (Moving Piston)
# ============================================================================

def test_acoustic_radiation_damping_piston():
    """Verify acoustic impedance radiation damping P = -rho * c * v_n on a moving piston.

    In the high-frequency limit (Plane Wave Approximation), the fluid opposes outward
    normal structural motion with pressure P_rad = -rho * c * v_n.
    """
    rho = 1000.0   # kg/m^3 (water density)
    c = 1500.0     # m/s (water sound speed)
    rho_c = rho * c  # 1.5e6 Pa*s/m

    # Create a 1m x 1m square piston in the XY plane (normal pointing in +Z)
    # 4 corner nodes: (0,0,0), (1,0,0), (1,1,0), (0,1,0)
    x = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [1.0, 1.0, 0.0],
        [0.0, 1.0, 0.0],
    ], dtype=float)

    # 1 quad segment
    segs = np.array([[0, 1, 2, 3]], dtype=np.int64)

    params = daa.DaaParams(
        id=1,
        rho=rho,
        c=c,
        pmax=0.0,         # No incident wave
        pmin=-1.0e10,     # Allow negative radiation pressure
        kform=2,          # High-frequency / PWA radiation damping mode
    )

    bnd = daa.DaaBoundary(params, segs)

    # Case A: Piston moving in +Z direction with velocity vn = +2.5 m/s
    vn = 2.5
    v = np.zeros((4, 3), dtype=float)
    v[:, 2] = vn
    dt = 1.0e-4

    f_ext = np.zeros((4, 3), dtype=float)
    res = bnd.step(t=0.01, dt=dt, x=x, v=v, f_ext=f_ext)

    # Normal should be +Z
    expected_p = -rho_c * vn  # -3.75e6 Pa
    assert pytest.approx(res["p_rad"][0], rel=1e-5) == expected_p
    assert pytest.approx(res["p_tot"][0], rel=1e-5) == expected_p

    # Area is 1.0 m^2, normal is [0, 0, 1]
    expected_f_total = expected_p * 1.0  # -3.75e6 N in Z direction
    actual_f_total = np.sum(f_ext[:, 2])
    assert pytest.approx(actual_f_total, rel=1e-5) == expected_f_total

    # Force distributed equally to 4 corner nodes
    for i in range(4):
        assert pytest.approx(f_ext[i, 2], rel=1e-5) == expected_f_total / 4.0
        assert abs(f_ext[i, 0]) < 1e-12
        assert abs(f_ext[i, 1]) < 1e-12

    # Case B: Piston moving in -Z direction with velocity vn = -1.0 m/s
    v[:, 2] = -1.0
    f_ext.fill(0.0)
    res_b = bnd.step(t=0.02, dt=dt, x=x, v=v, f_ext=f_ext)
    expected_p_b = -rho_c * (-1.0)  # +1.5e6 Pa (compressive pressure)
    assert pytest.approx(res_b["p_rad"][0], rel=1e-5) == expected_p_b
    assert pytest.approx(np.sum(f_ext[:, 2]), rel=1e-5) == expected_p_b * 1.0


# ============================================================================
# 2. Incident Shock Wave Arrival and Exponential Decay
# ============================================================================

def test_incident_shock_pressure_wave_arrival_and_decay():
    """Verify plane wave and spherical wave arrival time and exponential decay.

    Plane wave: t_arr = dot(x_c - x_source, d) / c
    P_inc(t) = P_max * exp(-(t - t_arr) / theta) for t >= t_arr
    """
    c = 1000.0     # m/s
    rho = 1000.0   # kg/m^3
    pmax = 5.0e6   # 5 MPa
    theta = 2.0e-3 # 2 ms

    # Segment at Z = 10.0 m
    x = np.array([
        [0.0, 0.0, 10.0],
        [1.0, 0.0, 10.0],
        [0.0, 1.0, 10.0],
    ], dtype=float)
    segs = np.array([[0, 1, 2]], dtype=np.int64)

    # --- Plane Wave Test ---
    params_plane = daa.DaaParams(
        rho=rho,
        c=c,
        pmax=pmax,
        theta=theta,
        iwave=2,  # Plane wave
        source_pos=np.array([0.0, 0.0, 0.0]),
        wave_dir=np.array([0.0, 0.0, 1.0]),
    )
    bnd_plane = daa.DaaBoundary(params_plane, segs)
    v_zero = np.zeros((3, 3), dtype=float)

    # Arrival time: distance 10.0m / 1000m/s = 0.010 s (10 ms)
    t_arr_expected = 10.0 / 1000.0

    # 1. Before arrival (t = 8 ms < 10 ms): pressure must be 0
    res_early = bnd_plane.step(t=0.008, dt=1e-4, x=x, v=v_zero)
    assert res_early["p_inc"][0] == 0.0

    # 2. Exactly at arrival (t = 10 ms): pressure must be P_max
    res_arr = bnd_plane.step(t=t_arr_expected, dt=1e-4, x=x, v=v_zero)
    assert pytest.approx(res_arr["p_inc"][0], rel=1e-5) == pmax

    # 3. After arrival at t = t_arr + theta (12 ms): pressure must be P_max / e
    res_decay = bnd_plane.step(t=t_arr_expected + theta, dt=1e-4, x=x, v=v_zero)
    expected_p = pmax * math.exp(-1.0)
    assert pytest.approx(res_decay["p_inc"][0], rel=1e-5) == expected_p

    # --- Spherical Wave Test ---
    source_pos = np.array([0.0, 0.0, 0.0])
    # Segment at (3, 4, 0) -> distance r = 5.0 m
    x_sph = np.array([
        [2.5, 4.0, 0.0],
        [3.5, 4.0, 0.0],
        [3.0, 4.5, 0.0],
    ], dtype=float)
    params_sph = daa.DaaParams(
        rho=rho,
        c=c,
        pmax=pmax,
        theta=theta,
        iwave=1,  # Spherical wave
        source_pos=source_pos,
        source_ref_dist=1.0,
        a_pmax=1.0,  # 1/r decay
    )
    bnd_sph = daa.DaaBoundary(params_sph, segs)

    # Distance to centroid (3, 4.1666..., 0)
    normals, areas, centroids = daa.compute_segment_geometry(x_sph, segs)
    r_actual = np.linalg.norm(centroids[0] - source_pos)
    t_arr_sph = r_actual / c

    res_sph_before = bnd_sph.step(t=t_arr_sph - 1e-4, dt=1e-4, x=x_sph, v=v_zero)
    assert res_sph_before["p_inc"][0] == 0.0

    res_sph_at = bnd_sph.step(t=t_arr_sph, dt=1e-4, x=x_sph, v=v_zero)
    expected_pmax_sph = pmax * (1.0 / r_actual)
    assert pytest.approx(res_sph_at["p_inc"][0], rel=1e-4) == expected_pmax_sph


# ============================================================================
# 3. Force Vector Equilibrium and External Work Balance
# ============================================================================

def test_force_vector_equilibrium_and_energy_work_balance():
    """Verify exact nodal force resultant equilibrium and external work accumulation."""
    rho = 1025.0  # Seawater
    c = 1530.0
    pmax = 2.0e6
    theta = 1.5e-3

    # Triangular boundary patch
    x = np.array([
        [0.0, 0.0, 0.0],
        [2.0, 0.0, 0.0],
        [0.0, 3.0, 0.0],
    ], dtype=float)
    segs = np.array([[0, 1, 2]], dtype=np.int64)

    # Area = 0.5 * (2 * 3) = 3.0 m^2, normal = [0, 0, 1]
    params = daa.DaaParams(
        rho=rho,
        c=c,
        pmax=pmax,
        theta=theta,
        iwave=2,
        source_pos=np.array([0.0, 0.0, 0.0]),
        wave_dir=np.array([0.0, 0.0, 1.0]),
        kform=2,
    )
    bnd = daa.DaaBoundary(params, segs)

    # Constant velocity vn = 0.5 m/s
    v = np.zeros((3, 3), dtype=float)
    v[:, 2] = 0.5
    dt = 1.0e-4

    f_ext = np.zeros((3, 3), dtype=float)
    # Step at t = 0.001 s (after arrival at Z=0)
    res = bnd.step(t=0.001, dt=dt, x=x, v=v, f_ext=f_ext)

    p_tot = res["p_tot"][0]
    expected_f_total = p_tot * 3.0  # P * Area
    actual_f_total = np.sum(f_ext[:, 2])

    # 1. Force equilibrium: sum(F_nodes) == P_tot * Area
    assert pytest.approx(actual_f_total, rel=1e-5) == expected_f_total

    # 2. Equal nodal partition (1/3 each for triangle)
    for i in range(3):
        assert pytest.approx(f_ext[i, 2], rel=1e-5) == expected_f_total / 3.0

    # 3. Work rate: dW/dt = F_total * v_n
    expected_work_rate = expected_f_total * 0.5
    assert pytest.approx(res["work_rate"], rel=1e-5) == expected_work_rate

    # 4. Cumulative external work: WFEXT = dW/dt * dt
    assert pytest.approx(bnd.wfext, rel=1e-5) == expected_work_rate * dt


# ============================================================================
# 4. DAA-1 Scattered Pressure Formulation
# ============================================================================

def test_daa1_scattered_pressure_evolution():
    """Verify DAA-1 differential equation integration for scattered pressure."""
    rho = 1000.0
    c = 1500.0
    area = 2.0  # m^2

    # Characteristic added mass M_added = rho * area * sqrt(area)
    l_char = math.sqrt(area)
    m_added = rho * area * l_char
    omega = (rho * c * area) / m_added  # Transition frequency = c / l_char

    params = daa.DaaParams(
        rho=rho,
        c=c,
        pmax=0.0,  # No incident shock
        kform=1,   # DAA-1
        integr=1,  # Euler 1st order
    )

    x = np.array([
        [0.0, 0.0, 0.0],
        [2.0, 0.0, 0.0],
        [2.0, 1.0, 0.0],
        [0.0, 1.0, 0.0],
    ], dtype=float)
    segs = np.array([[0, 1, 2, 3]], dtype=np.int64)

    bnd = daa.DaaBoundary(params, segs)

    # Initial condition: boundary accelerated with an = 100 m/s^2, vn = 0
    a = np.zeros((4, 3), dtype=float)
    a[:, 2] = 100.0
    v = np.zeros((4, 3), dtype=float)
    dt = 1.0e-5

    # Step 1: dP_m/dt = rho * c * an
    res1 = bnd.step(t=dt, dt=dt, x=x, v=v, a=a)
    expected_dp = (rho * c * 100.0) * dt
    assert pytest.approx(res1["p_scat"][0], rel=1e-3) == expected_dp


# ============================================================================
# 5. Free Surface Image Reflection & Cavitation Cutoff
# ============================================================================

def test_free_surface_reflection_and_cavitation():
    """Verify inverted tensile wave from free surface and cavitation cutoff P_tot >= P_min."""
    rho = 1000.0
    c = 1500.0
    pmax = 4.0e6
    theta = 1.0e-3

    # Wet surface at Z = -5.0 m
    x = np.array([
        [0.0, 0.0, -5.0],
        [1.0, 0.0, -5.0],
        [0.0, 1.0, -5.0],
    ], dtype=float)
    segs = np.array([[0, 1, 2]], dtype=np.int64)

    # Source at Z = -10.0 m, free surface at Z = 0.0 m
    # Image source will be at Z = +10.0 m
    params = daa.DaaParams(
        rho=rho,
        c=c,
        pmax=pmax,
        theta=theta,
        iwave=1,  # Spherical
        source_pos=np.array([0.0, 0.0, -10.0]),
        freesurf=2,  # Free surface reflection enabled
        fs_point=np.array([0.0, 0.0, 0.0]),
        fs_normal=np.array([0.0, 0.0, 1.0]),
        pmin=0.0,    # Cavitation cutoff: fluid cannot sustain tension
    )
    bnd = daa.DaaBoundary(params, segs)
    v_zero = np.zeros((3, 3), dtype=float)

    # Distance to source: |-5 - (-10)| = 5.0 m -> arrival at t = 5/1500 = 0.00333 s
    # Distance to image source: |-5 - 10| = 15.0 m -> arrival at t = 15/1500 = 0.010 s
    t_inc = 5.0 / 1500.0
    t_refl = 15.0 / 1500.0

    # 1. At t = 0.004 s (incident arrived, reflected not yet): positive pressure
    res1 = bnd.step(t=t_inc + 0.0005, dt=1e-4, x=x, v=v_zero)
    assert res1["p_inc"][0] > 0.0
    assert res1["p_refl"][0] == 0.0

    # 2. At t = 0.011 s (both arrived): reflected wave is negative
    res2 = bnd.step(t=t_refl + 0.0001, dt=1e-4, x=x, v=v_zero)
    assert res2["p_refl"][0] < 0.0

    # 3. Cavitation cutoff: P_tot must never be below pmin = 0.0
    assert res2["p_tot"][0] >= 0.0


# ============================================================================
# 6. BEM Collocation Potential Flow (bemsolv.F)
# ============================================================================

def test_bem_collocation_potential_flow():
    """Verify BEM potential flow collocation coefficient calculation (H and G matrices)."""
    # 2 triangular facets
    x = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [1.0, 1.0, 0.0],
    ], dtype=float)
    segs = np.array([
        [0, 1, 2],
        [1, 3, 2],
    ], dtype=np.int64)

    H, G = daa.bem_collocation_coefficients(x, segs)

    # Matrix shapes must be (N_segs, N_segs)
    assert H.shape == (2, 2)
    assert G.shape == (2, 2)

    # Self-influence H_ii must be 0.5 (jump term c(x) = 1/2 for smooth planar surface)
    assert pytest.approx(H[0, 0]) == 0.5
    assert pytest.approx(H[1, 1]) == 0.5

    # Self-influence G_ii must be strictly positive (singular patch integral)
    assert G[0, 0] > 0.0
    assert G[1, 1] > 0.0

    # Off-diagonal symmetry for planar coplanar facets (n_j perpendicular to r_ij -> H_ij = 0)
    assert abs(H[0, 1]) < 1e-12
    assert abs(H[1, 0]) < 1e-12

    # Solve potential flow for unit normal outflow q = [1.0, 1.0]
    q = np.array([1.0, 1.0], dtype=float)
    phi = daa.bem_solve_potential_flow(H, G, q)
    assert len(phi) == 2
    assert phi[0] > 0.0
    assert phi[1] > 0.0
