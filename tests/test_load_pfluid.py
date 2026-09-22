"""Unit tests for /LOAD/PFLUID hydrostatic fluid surface pressure loading.

Tests pressure vs depth, free surface cutoff, static force equilibrium on container
walls, dynamic acceleration coupling, sensor gating, and external work tracking.
"""

from __future__ import annotations

import math
import numpy as np
import pytest

from pyradioss.engine.pfluid import (
    PfluidEngine,
    PfluidLoad,
    PfluidLoadParams,
    PfluidSegmentResult,
    PfluidStepResult,
    compute_segment_normal_and_area,
    pfluid_subroutine,
)


def test_linear_pressure_increase_with_depth():
    """Test linear hydrostatic pressure increase with depth: P = P0 + rho * g * h."""
    rho_f = 1000.0  # kg / m^3 (water)
    g = 9.81        # m / s^2
    z0 = 10.0       # free surface at z = 10 m
    p0 = 101325.0   # 1 atm

    load = PfluidLoadParams(
        id=1,
        rho_f=rho_f,
        z0=z0,
        p0=p0,
        gravity=(0.0, 0.0, -g),
        segments=[
            [1, 2, 3, 4],  # at z = 8 m (h = 2 m)
            [5, 6, 7, 8],  # at z = 5 m (h = 5 m)
        ],
    )

    coords = {
        # Segment 1: square at z = 8 m
        1: [0.0, 0.0, 8.0],
        2: [1.0, 0.0, 8.0],
        3: [1.0, 1.0, 8.0],
        4: [0.0, 1.0, 8.0],
        # Segment 2: square at z = 5 m
        5: [0.0, 0.0, 5.0],
        6: [1.0, 0.0, 5.0],
        7: [1.0, 1.0, 5.0],
        8: [0.0, 1.0, 5.0],
    }

    engine = PfluidEngine([load])
    res = engine.compute_step(dt=0.1, time=0.0, coords=coords)

    # Segment 1: depth = 2 m
    # P1 = 101325 + 1000 * 9.81 * 2 = 120945 Pa
    p1 = res.segment_results[0].pressure
    assert math.isclose(p1, p0 + rho_f * g * 2.0, rel_tol=1e-12)

    # Segment 2: depth = 5 m
    # P2 = 101325 + 1000 * 9.81 * 5 = 150375 Pa
    p2 = res.segment_results[1].pressure
    assert math.isclose(p2, p0 + rho_f * g * 5.0, rel_tol=1e-12)

    # Delta P = rho * g * Delta h
    assert math.isclose(p2 - p1, rho_f * g * (5.0 - 2.0), rel_tol=1e-12)


def test_zero_pressure_above_liquid_free_surface():
    """Test that segments above the liquid free surface have zero hydrostatic pressure."""
    rho_f = 1000.0
    g = 9.81
    z0 = 5.0  # Free surface at z = 5.0 m

    load = PfluidLoadParams(
        id=1,
        rho_f=rho_f,
        z0=z0,
        p0=0.0,  # Gauge pressure mode
        gravity=(0.0, 0.0, -g),
        segments=[
            [1, 2, 3, 4],  # Above surface at z = 7 m
            [5, 6, 7, 8],  # Exactly on surface at z = 5 m
            [9, 10, 11, 12], # Submerged at z = 2 m
        ],
    )

    coords = {
        1: [0.0, 0.0, 7.0], 2: [1.0, 0.0, 7.0], 3: [1.0, 1.0, 7.0], 4: [0.0, 1.0, 7.0],
        5: [0.0, 0.0, 5.0], 6: [1.0, 0.0, 5.0], 7: [1.0, 1.0, 5.0], 8: [0.0, 1.0, 5.0],
        9: [0.0, 0.0, 2.0], 10: [1.0, 0.0, 2.0], 11: [1.0, 1.0, 2.0], 12: [0.0, 1.0, 2.0],
    }

    engine = PfluidEngine([load])
    res = engine.compute_step(dt=0.1, time=0.0, coords=coords)

    # Above surface: depth = 0, pressure = 0
    assert res.segment_results[0].depth == 0.0
    assert res.segment_results[0].pressure == 0.0
    assert np.allclose(res.segment_results[0].force_vector, 0.0)

    # On surface: depth = 0, pressure = 0
    assert res.segment_results[1].depth == 0.0
    assert res.segment_results[1].pressure == 0.0
    assert np.allclose(res.segment_results[1].force_vector, 0.0)

    # Submerged: depth = 3 m, pressure = 1000 * 9.81 * 3 = 29430 Pa
    assert math.isclose(res.segment_results[2].depth, 3.0, rel_tol=1e-12)
    assert math.isclose(res.segment_results[2].pressure, 29430.0, rel_tol=1e-12)
    assert np.linalg.norm(res.segment_results[2].force_vector) > 0.0


def test_force_vector_equilibrium_container_walls():
    """Test global force equilibrium and fluid weight balance in a container.

    Consider a rectangular tank [0, Lx] x [0, Ly] filled with fluid to height H:
    1. Opposing horizontal wall forces must cancel: sum(Fx) = 0 and sum(Fy) = 0.
    2. Total downward force on bottom wall must equal total fluid weight W = rho * g * (Lx * Ly * H).
    """
    rho_f = 1000.0  # kg / m^3
    g = 9.81        # m / s^2
    lx = 2.0        # m
    ly = 3.0        # m
    h = 4.0         # m (fluid depth)
    z0 = h          # free surface at top of tank

    # Vertices of the tank:
    # Bottom: nodes 1,2,3,4 at z=0
    # Top: nodes 5,6,7,8 at z=H
    coords = {
        1: [0.0, 0.0, 0.0],
        2: [lx,  0.0, 0.0],
        3: [lx,  ly,  0.0],
        4: [0.0, ly,  0.0],
        5: [0.0, 0.0, h],
        6: [lx,  0.0, h],
        7: [lx,  ly,  h],
        8: [0.0, ly,  h],
    }

    # Outward facing normals from fluid to container walls:
    # Bottom wall (z=0): normal points down [0, 0, -1] -> (1, 4, 3, 2)
    # Left wall (x=0): normal points -x [-1, 0, 0] -> (1, 5, 8, 4)
    # Right wall (x=Lx): normal points +x [+1, 0, 0] -> (2, 3, 7, 6)
    # Front wall (y=0): normal points -y [0, -1, 0] -> (1, 2, 6, 5)
    # Back wall (y=Ly): normal points +y [0, +1, 0] -> (4, 8, 7, 3)
    segments = [
        [1, 4, 3, 2],  # Bottom wall
        [1, 5, 8, 4],  # Left wall
        [2, 3, 7, 6],  # Right wall
        [1, 2, 6, 5],  # Front wall
        [4, 8, 7, 3],  # Back wall
    ]

    load = PfluidLoadParams(
        id=1,
        rho_f=rho_f,
        z0=z0,
        p0=0.0,
        gravity=(0.0, 0.0, -g),
        segments=segments,
    )

    engine = PfluidEngine([load])
    res = engine.compute_step(dt=0.1, time=0.0, coords=coords)

    # 1. Check bottom wall force
    # Bottom wall depth = H = 4 m, Area = Lx * Ly = 6 m^2
    # Pressure = rho * g * H = 1000 * 9.81 * 4 = 39240 Pa
    # Force = P * Area * normal = 39240 * 6 * [0, 0, -1] = [0, 0, -235440] N
    bottom_force = res.segment_results[0].force_vector
    fluid_weight = rho_f * (lx * ly * h) * g
    assert math.isclose(bottom_force[2], -fluid_weight, rel_tol=1e-12)

    # 2. Check left wall and right wall forces (equal and opposite in x)
    left_force = res.segment_results[1].force_vector
    right_force = res.segment_results[2].force_vector
    assert math.isclose(left_force[0] + right_force[0], 0.0, abs_tol=1e-10)
    assert math.isclose(left_force[0], -right_force[0], rel_tol=1e-12)

    # 3. Check front wall and back wall forces (equal and opposite in y)
    front_force = res.segment_results[3].force_vector
    back_force = res.segment_results[4].force_vector
    assert math.isclose(front_force[1] + back_force[1], 0.0, abs_tol=1e-10)
    assert math.isclose(front_force[1], -back_force[1], rel_tol=1e-12)

    # 4. Global net force on the entire container:
    # Net horizontal forces must be exactly zero!
    assert math.isclose(res.total_force[0], 0.0, abs_tol=1e-10)
    assert math.isclose(res.total_force[1], 0.0, abs_tol=1e-10)
    # Net vertical force must equal total fluid weight!
    assert math.isclose(res.total_force[2], -fluid_weight, rel_tol=1e-12)


def test_sensor_gating_and_time_activation():
    """Test sensor gating and start/stop time windowing for /LOAD/PFLUID."""
    load = PfluidLoadParams(
        id=1,
        rho_f=1000.0,
        z0=2.0,
        gravity=(0.0, 0.0, -9.81),
        sensor_id=10,
        tstart=1.0,
        tstop=5.0,
        segments=[[1, 2, 3, 4]],
    )

    coords = {
        1: [0.0, 0.0, 0.0],
        2: [1.0, 0.0, 0.0],
        3: [1.0, 1.0, 0.0],
        4: [0.0, 1.0, 0.0],
    }

    engine = PfluidEngine([load])

    # Time t = 0.5 (< tstart): inactive even if sensor is active
    res0 = engine.compute_step(dt=0.1, time=0.5, coords=coords, sensor_active={10: True})
    assert len(res0.segment_results) == 0
    assert np.allclose(res0.total_force, 0.0)

    # Time t = 2.0, but sensor is inactive: no load applied
    res1 = engine.compute_step(dt=0.1, time=2.0, coords=coords, sensor_active={10: False})
    assert len(res1.segment_results) == 0
    assert np.allclose(res1.total_force, 0.0)

    # Time t = 2.0, sensor active: load is active
    res2 = engine.compute_step(dt=0.1, time=2.0, coords=coords, sensor_active={10: True})
    assert len(res2.segment_results) == 1
    assert np.linalg.norm(res2.total_force) > 0.0

    # Time t = 6.0 (> tstop): inactive
    res3 = engine.compute_step(dt=0.1, time=6.0, coords=coords, sensor_active={10: True})
    assert len(res3.segment_results) == 0


def test_dynamic_acceleration_coupling():
    """Test dynamic acceleration coupling (a_container) on effective gravity."""
    rho_f = 1000.0
    g = 9.81
    z0 = 5.0

    coords = {
        1: [0.0, 0.0, 0.0],
        2: [1.0, 0.0, 0.0],
        3: [1.0, 1.0, 0.0],
        4: [0.0, 1.0, 0.0],
    }

    # 1. Base static case: a_container = 0 -> g_eff = 9.81 m/s^2
    load_static = PfluidLoadParams(
        id=1,
        rho_f=rho_f,
        z0=z0,
        gravity=(0.0, 0.0, -g),
        a_container=(0.0, 0.0, 0.0),
        segments=[[1, 2, 3, 4]],
    )
    res_static = PfluidEngine([load_static]).compute_step(dt=0.1, time=0.0, coords=coords)
    p_static = res_static.segment_results[0].pressure

    # 2. Upward container acceleration a_z = +2.0 m/s^2 -> g_eff = 9.81 - (-2.0) = 11.81 m/s^2
    load_accel = PfluidLoadParams(
        id=2,
        rho_f=rho_f,
        z0=z0,
        gravity=(0.0, 0.0, -g),
        a_container=(0.0, 0.0, 2.0),
        segments=[[1, 2, 3, 4]],
    )
    res_accel = PfluidEngine([load_accel]).compute_step(dt=0.1, time=0.0, coords=coords)
    p_accel = res_accel.segment_results[0].pressure

    assert math.isclose(p_accel, rho_f * (g + 2.0) * z0, rel_tol=1e-12)
    assert math.isclose(p_accel / p_static, (g + 2.0) / g, rel_tol=1e-12)

    # 3. Free fall: a_container = (0, 0, -9.81) -> g_eff = 0 -> pressure = 0
    load_freefall = PfluidLoadParams(
        id=3,
        rho_f=rho_f,
        z0=z0,
        gravity=(0.0, 0.0, -g),
        a_container=(0.0, 0.0, -g),
        segments=[[1, 2, 3, 4]],
    )
    res_freefall = PfluidEngine([load_freefall]).compute_step(dt=0.1, time=0.0, coords=coords)
    p_freefall = res_freefall.segment_results[0].pressure
    assert math.isclose(p_freefall, 0.0, abs_tol=1e-12)


def test_external_work_accumulation():
    """Test external work accumulation: dW = sum(F_seg . v_centroid) * dt."""
    load = PfluidLoadParams(
        id=1,
        rho_f=1000.0,
        z0=1.0,
        gravity=(0.0, 0.0, -9.81),
        segments=[[1, 2, 3, 4]],  # Square of area 1 at z = 0 (h = 1 m)
    )

    coords = {
        1: [0.0, 0.0, 0.0],
        2: [1.0, 0.0, 0.0],
        3: [1.0, 1.0, 0.0],
        4: [0.0, 1.0, 0.0],
    }

    # Velocity in direction of normal [+z]: vz = 2.0 m/s
    vels = {
        1: [0.0, 0.0, 2.0],
        2: [0.0, 0.0, 2.0],
        3: [0.0, 0.0, 2.0],
        4: [0.0, 0.0, 2.0],
    }

    engine = PfluidEngine([load])
    dt = 0.05

    # Force: P = 1000 * 9.81 * 1 = 9810 Pa, Area = 1 m^2, Normal = [0, 0, 1]
    # F = [0, 0, 9810] N
    # Power = F . v = 9810 * 2.0 = 19620 W
    # Work increment dW = 19620 * 0.05 = 981 J
    res1 = engine.compute_step(dt=dt, time=0.0, coords=coords, vels=vels)
    assert math.isclose(res1.total_power, 19620.0, rel_tol=1e-12)
    assert math.isclose(res1.incremental_work, 981.0, rel_tol=1e-12)
    assert math.isclose(res1.cumul_work, 981.0, rel_tol=1e-12)

    # Second step: cumul work doubles to 1962 J
    res2 = engine.compute_step(dt=dt, time=dt, coords=coords, vels=vels)
    assert math.isclose(res2.cumul_work, 1962.0, rel_tol=1e-12)


def test_pfluid_subroutine_fortran_parity():
    """Verify Fortran kernel emulation `pfluid_subroutine` against `pfluid.F`."""
    # 2 segments: 1 quad and 1 tri
    lloadp = np.array([
        1, 2, 3, 4,  # Segment 1: quad nodes 1,2,3,4
        1, 2, 3, 0,  # Segment 2: tri nodes 1,2,3 (n4=0)
    ], dtype=np.int32)

    x = np.array([
        [0.0, 2.0, 2.0, 0.0],
        [0.0, 0.0, 2.0, 2.0],
        [0.0, 0.0, 0.0, 0.0],
    ], dtype=np.float64)

    v = np.array([
        [0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0],
        [1.0, 1.0, 1.0, 1.0],
    ], dtype=np.float64)

    fext = np.zeros((3, 4), dtype=np.float64)
    dt = 0.01

    wfext = pfluid_subroutine(
        iloadp=np.zeros((10, 2), dtype=np.int32),
        rload=np.zeros((10, 2), dtype=np.float64),
        lloadp=lloadp,
        x=x,
        v=v,
        fext=fext,
        dt=dt,
        rho_f=1000.0,
        g=9.81,
        z0=2.0,
    )

    # Segment 1 (quad): Area = 4.0, z = 0, z0 = 2 -> depth = 2, P = 1000 * 9.81 * 2 = 19620 Pa
    # F_seg1 = 19620 * 4 = 78480 N along +z
    # Segment 2 (tri): Area = 0.5 * 2 * 2 = 2.0, depth = 2, P = 19620 Pa
    # F_seg2 = 19620 * 2 = 39240 N along +z
    # Total Fz = 78480 + 39240 = 117720 N
    total_fz = np.sum(fext[2, :])
    assert math.isclose(total_fz, 117720.0, rel_tol=1e-12)

    # External work: v = 1.0 m/s -> Power = 117720 W -> dW = 117720 * 0.01 = 1177.2 J
    assert math.isclose(wfext, 1177.2, rel_tol=1e-12)


def test_top_level_exports():
    """Verify clean top-level imports from pyradioss.engine."""
    import pyradioss.engine as pe

    assert hasattr(pe, "PfluidLoadParams")
    assert hasattr(pe, "PfluidLoad")
    assert hasattr(pe, "PfluidEngine")
    assert hasattr(pe, "PfluidStepResult")
    assert hasattr(pe, "compute_segment_normal_and_area")
    assert hasattr(pe, "pfluid_subroutine")
