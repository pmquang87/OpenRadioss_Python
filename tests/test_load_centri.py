"""
Unit tests for /LOAD/CENTRI centrifugal load and rotational acceleration field.

Ported from OpenRadioss Fortran:
- starter/source/loads/general/load_centri/hm_read_load_centri.F
- engine/source/loads/general/load_centri/cfield.F
- engine/source/loads/general/load_centri/cfield_imp.F
"""

import math
import numpy as np
import pytest

from pyradioss.engine.load_centri import (
    LoadCentri,
    CentrifugalResult,
    compute_centrifugal_forces,
    cfield_subroutine,
    CentrifugalEngine,
    build_load_centri,
)


def test_rotation_around_z_axis_pure_radial_force():
    """Verify rotation around Z-axis yields purely radial outward force F_r = m * omega^2 * r, F_z = 0."""
    # Rotation around Z-axis: origin (0, 0, 0), axis (0, 0, 1)
    omega = 50.0  # rad/s
    mass = 2.5    # kg
    load = LoadCentri(
        axis=(0.0, 0.0, 1.0),
        origin=(0.0, 0.0, 0.0),
        omega=omega,
    )

    # 4 nodes at distance r = 2.0 m in the XY plane:
    # Node 1: (2, 0, 5) -> dist = (2, 0, 0)
    # Node 2: (0, 2, -3) -> dist = (0, 2, 0)
    # Node 3: (-2, 0, 0) -> dist = (-2, 0, 0)
    # Node 4: (math.sqrt(2), math.sqrt(2), 10) -> r = 2.0
    nodes_x = np.array([
        [2.0, 0.0, 5.0],
        [0.0, 2.0, -3.0],
        [-2.0, 0.0, 0.0],
        [math.sqrt(2.0), math.sqrt(2.0), 10.0],
    ])
    nodes_v = np.zeros_like(nodes_x)

    res = compute_centrifugal_forces(nodes_x, nodes_v, mass, load)

    expected_acc_mag = (omega ** 2) * 2.0  # 2500 * 2 = 5000 m/s^2
    expected_force_mag = mass * expected_acc_mag  # 2.5 * 5000 = 12500 N

    # Check Node 1: purely along +X
    np.testing.assert_allclose(res.forces[0], [expected_force_mag, 0.0, 0.0], rtol=1e-6)
    # Check Node 2: purely along +Y
    np.testing.assert_allclose(res.forces[1], [0.0, expected_force_mag, 0.0], rtol=1e-6)
    # Check Node 3: purely along -X
    np.testing.assert_allclose(res.forces[2], [-expected_force_mag, 0.0, 0.0], rtol=1e-6)

    # All nodes must have F_z = 0 regardless of z coordinate!
    np.testing.assert_allclose(res.forces[:, 2], 0.0, atol=1e-12)

    # Force magnitudes must all equal m * omega^2 * r
    force_mags = np.linalg.norm(res.forces, axis=1)
    np.testing.assert_allclose(force_mags, expected_force_mag, rtol=1e-6)


def test_angular_acceleration_tangential_force():
    """Verify angular acceleration yields azimuthal Euler tangential force F_theta = m * alpha * r."""
    # Rotation around Z-axis with zero omega and pure angular acceleration alpha = domega/dt = 100 rad/s^2
    alpha = 100.0
    mass = 3.0
    load = LoadCentri(
        axis=(0.0, 0.0, 1.0),
        origin=(0.0, 0.0, 0.0),
        omega=0.0,
        domega_dt=alpha,
        flag=2,
    )

    # Node on positive X axis at (4, 0, 0): r_perp = (4, 0, 0)
    # dw = (0, 0, alpha)
    # a_euler = dw x r_perp = (0, 0, alpha) x (4, 0, 0) = (0, 4*alpha, 0) in +Y direction!
    nodes_x = np.array([[4.0, 0.0, 0.0]])
    nodes_v = np.zeros_like(nodes_x)

    res = compute_centrifugal_forces(nodes_x, nodes_v, mass, load)

    expected_acc_y = alpha * 4.0  # 400 m/s^2
    expected_force_y = mass * expected_acc_y  # 1200 N
    np.testing.assert_allclose(res.accelerations[0], [0.0, expected_acc_y, 0.0], rtol=1e-6)
    np.testing.assert_allclose(res.forces[0], [0.0, expected_force_y, 0.0], rtol=1e-6)

    # Now with both omega = 10 rad/s and alpha = 20 rad/s^2
    load_both = LoadCentri(
        axis=(0.0, 0.0, 1.0),
        origin=(0.0, 0.0, 0.0),
        omega=10.0,
        domega_dt=20.0,
        flag=2,
    )
    res_both = compute_centrifugal_forces(nodes_x, nodes_v, mass, load_both)
    # Radial acceleration: omega^2 * r = 100 * 4 = 400 (+X)
    # Tangential acceleration: alpha * r = 20 * 4 = 80 (+Y)
    expected_both_acc = np.array([400.0, 80.0, 0.0])
    np.testing.assert_allclose(res_both.accelerations[0], expected_both_acc, rtol=1e-6)
    np.testing.assert_allclose(res_both.forces[0], mass * expected_both_acc, rtol=1e-6)


def test_oblique_3d_rotation_axis_and_arbitrary_origin():
    """Verify arbitrary 3D rotation axis through an offset origin point."""
    # Axis along diagonal direction (1, 1, 1) normalized
    u_raw = np.array([1.0, 1.0, 1.0])
    u_norm = u_raw / np.linalg.norm(u_raw)
    x0 = np.array([10.0, 20.0, 30.0])
    omega = 40.0
    mass = 2.0

    load = LoadCentri(
        axis=tuple(u_norm),
        origin=tuple(x0),
        omega=omega,
    )

    # Choose a node on the rotation axis: should have zero distance and zero force
    node_on_axis = x0 + 5.0 * u_norm
    res_on = compute_centrifugal_forces(node_on_axis, None, mass, load)
    np.testing.assert_allclose(res_on.distances, 0.0, atol=1e-10)
    np.testing.assert_allclose(res_on.forces, 0.0, atol=1e-10)

    # Choose a node perpendicular to the axis at distance d = 3.0 m
    # Vector v_perp orthogonal to (1, 1, 1), e.g. (1, -1, 0) normalized
    v_perp = np.array([1.0, -1.0, 0.0]) / math.sqrt(2.0)
    node_off = x0 + 3.0 * v_perp

    res_off = compute_centrifugal_forces(node_off, None, mass, load)
    # Distance vector should be 3.0 * v_perp
    np.testing.assert_allclose(res_off.distances, 3.0 * v_perp, rtol=1e-6)
    # Force must be purely parallel to v_perp with magnitude m * omega^2 * d
    expected_mag = mass * (omega ** 2) * 3.0  # 2.0 * 1600 * 3 = 9600 N
    np.testing.assert_allclose(res_off.forces, expected_mag * v_perp, rtol=1e-6)


def test_sensor_activation_delay():
    """Verify zero forces before sensor trigger time and full activation afterwards."""
    tstart = 0.05  # Trigger at 50 ms
    omega = 100.0
    mass = 1.0
    load = LoadCentri(
        axis=(0.0, 0.0, 1.0),
        omega=omega,
        tstart=tstart,
    )
    node = np.array([[2.0, 0.0, 0.0]])

    # Before trigger: t = 0.02 s < 0.05 s
    res_before = compute_centrifugal_forces(node, None, mass, load, t=0.02)
    assert not res_before.active
    np.testing.assert_allclose(res_before.forces, 0.0, atol=1e-12)
    assert res_before.work_increment == 0.0

    # At trigger: t = 0.05 s >= 0.05 s
    res_at = compute_centrifugal_forces(node, None, mass, load, t=0.05)
    assert res_at.active
    assert np.linalg.norm(res_at.forces) > 0.0

    # After trigger: t = 0.10 s
    res_after = compute_centrifugal_forces(node, None, mass, load, t=0.10)
    assert res_after.active
    expected_f = mass * (omega ** 2) * 2.0
    np.testing.assert_allclose(res_after.forces[0], [expected_f, 0.0, 0.0], rtol=1e-6)


def test_external_work_accumulation_wfext():
    """Verify energy rate / external work accumulation dW = sum(F . v_mid) * dt."""
    omega = 20.0
    mass = np.array([2.0, 3.0])
    load = LoadCentri(
        axis=(0.0, 0.0, 1.0),
        omega=omega,
    )
    nodes_x = np.array([
        [1.0, 0.0, 0.0],
        [0.0, 2.0, 0.0],
    ])
    # Velocity in direction of centrifugal force
    nodes_v = np.array([
        [5.0, 0.0, 0.0],
        [0.0, 10.0, 0.0],
    ])
    dt = 1e-4

    res = compute_centrifugal_forces(nodes_x, nodes_v, mass, load, dt=dt)

    # Node 1: F1 = 2.0 * (400 * 1.0) = 800 N (+X), a1 = 400 m/s^2 (+X)
    # v_mid_1 = [5.0 + 0.5 * 1e-4 * 400, 0, 0] = [5.02, 0, 0]
    # dW1 = 800 * 5.02 * 1e-4 = 0.4016 J
    # Node 2: F2 = 3.0 * (400 * 2.0) = 2400 N (+Y), a2 = 800 m/s^2 (+Y)
    # v_mid_2 = [0, 10.0 + 0.5 * 1e-4 * 800, 0] = [0, 10.04, 0]
    # dW2 = 2400 * 10.04 * 1e-4 = 2.4096 J
    # Total dW = 0.4016 + 2.4096 = 2.8112 J
    expected_dw = 800.0 * 5.02 * 1e-4 + 2400.0 * 10.04 * 1e-4
    np.testing.assert_allclose(res.work_increment, expected_dw, rtol=1e-6)

    # Test CentrifugalEngine multi-step integration
    engine = CentrifugalEngine([load])
    f_ext, dw = engine.step(nodes_x, nodes_v, mass, t=0.0, dt=dt)
    np.testing.assert_allclose(dw, expected_dw, rtol=1e-6)
    assert abs(engine.wfext - expected_dw) < 1e-8

    # Second step
    f_ext2, dw2 = engine.step(nodes_x, nodes_v, mass, t=dt, dt=dt)
    assert abs(engine.wfext - 2.0 * expected_dw) < 1e-8


def test_time_dependent_curve_function():
    """Verify curve evaluation omega(t) with scale factors."""
    # Linear ramp: f(t) = 100 * t
    load = LoadCentri(
        axis=(0.0, 0.0, 1.0),
        function=lambda t: 100.0 * t,
        scale_x=2.0,  # t_scaled = 2 * t
        scale_y=1.5,  # omega = 1.5 * f(t_scaled)
    )
    # At t = 0.5 s:
    # t_scaled = 2 * 0.5 = 1.0 s
    # f(1.0) = 100.0
    # omega = 1.5 * 100.0 = 150.0 rad/s
    # domega/dt = 1.5 * 2.0 * 100.0 = 300.0 rad/s^2
    omega_val, dwdt_val, is_act = load.evaluate_kinematics(t=0.5)
    assert is_act
    assert abs(omega_val - 150.0) < 1e-4
    assert abs(dwdt_val - 300.0) < 1e-2


def test_build_load_centri_card_formats():
    """Verify build_load_centri converts string directions and dictionaries."""
    d = {
        "id": 10,
        "rad_dir": "XX",
        "magnitude": 60.0,
        "xscale": 0.5,
        "x0": 1.0,
        "y0": 2.0,
        "z0": 3.0,
        "rad_sensor_id": 5,
        "rad_ivar_flag": 2,
    }
    load = build_load_centri(d)
    assert load.id == 10
    assert load.axis == (1.0, 0.0, 0.0)
    assert load.origin == (1.0, 2.0, 3.0)
    assert load.scale_y == 60.0
    assert load.scale_x == 0.5
    assert load.sensor_id == 5
    assert load.flag == 2


def test_cfield_subroutine_wrapper():
    """Verify cfield_subroutine returns forces, accelerations, and work increment."""
    x = np.array([[1.0, 0.0, 0.0]])
    v = np.array([[0.0, 0.0, 0.0]])
    m = np.array([5.0])
    forces, accels, dw = cfield_subroutine(
        x=x,
        v=v,
        m=m,
        u_axis=(0.0, 0.0, 1.0),
        x0=(0.0, 0.0, 0.0),
        omega=10.0,
        domega_dt=0.0,
        dt=1e-4,
    )
    # a = 100 * 1 = 100 m/s^2 along X
    # F = 5 * 100 = 500 N along X
    np.testing.assert_allclose(accels[0], [100.0, 0.0, 0.0], rtol=1e-6)
    np.testing.assert_allclose(forces[0], [500.0, 0.0, 0.0], rtol=1e-6)
