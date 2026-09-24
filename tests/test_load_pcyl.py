"""
Unit tests for Cylindrical Coordinate Pressure Loading (/LOAD/PCYL).

Fortran origins:
  - ``starter/source/loads/general/load_pcyl/hm_read_pcyl.F``
  - ``engine/source/loads/general/load_pcyl/pressure_cyl.F``
  - ``engine/source/loads/general/load_pcyl/press_seg3.F``
"""

from __future__ import annotations

import math
import numpy as np
import pytest

from pyradioss.engine.load_pcyl import (
    PcylLoadEngine,
    PcylLoadParams,
    PcylSegment,
    build_pcyl_loads,
)
from pyradioss.model.entities import PcylLoad, Surface
from pyradioss.model.model import Model


def _create_cylinder_mesh(
    radius: float,
    height: float,
    n_theta: int = 36,
    n_z: int = 4,
    theta_min: float = 0.0,
    theta_max: float = 2.0 * math.pi,
):
    """Generate nodes and quad segments for a cylinder section."""
    thetas = np.linspace(theta_min, theta_max, n_theta + 1)
    zs = np.linspace(0.0, height, n_z + 1)

    nodes = []
    # Node ordering: outer loop z, inner loop theta
    for z in zs:
        for th in thetas:
            x = radius * math.cos(th)
            y = radius * math.sin(th)
            nodes.append([x, y, z])
    nodes = np.array(nodes, dtype=np.float64)

    # Segments: quad connectivity
    # For outward normal on cylinder: node1=(i,j), node2=(i+1,j), node3=(i+1,j+1), node4=(i,j+1)
    # where i is theta index, j is z index
    segments = []
    stride = n_theta + 1
    for j in range(n_z):
        for i in range(n_theta):
            n1 = j * stride + i
            n2 = j * stride + (i + 1)
            n3 = (j + 1) * stride + (i + 1)
            n4 = (j + 1) * stride + i
            segments.append((n1, n2, n3, n4))

    return nodes, segments


def test_load_pcyl_uniform_radial_pressure():
    """Verify uniform radial pressure loading on a half-cylinder shell.

    Theoretical result:
    For a half-cylinder (theta in [-pi/2, pi/2]) with radius R and height H:
    Projected area onto the y-z plane is A_proj = 2 * R * H.
    Net resultant force in x-direction is F_x = P0 * A_proj = 2 * P0 * R * H.
    Resultant forces in y and z directions are zero by symmetry.
    """
    radius = 1.0  # m
    height = 2.0  # m
    p0 = 10000.0  # 10 kPa
    n_theta = 40
    n_z = 10

    # Half cylinder: theta from -pi/2 to pi/2
    nodes, segments = _create_cylinder_mesh(
        radius=radius,
        height=height,
        n_theta=n_theta,
        n_z=n_z,
        theta_min=-0.5 * math.pi,
        theta_max=0.5 * math.pi,
    )

    params = PcylLoadParams(
        id=1,
        title="Half_Cylinder_Uniform",
        axis_origin=(0.0, 0.0, 0.0),
        axis_dir=(0.0, 0.0, 1.0),
        ref_dir=(1.0, 0.0, 0.0),
        p0=p0,
        segments=segments,
    )
    engine = PcylLoadEngine(params)

    fext = np.zeros_like(nodes)
    total_force_mag = engine.apply_load(t=0.0, dt=1e-4, fext=fext, x=nodes)

    # Net force summed over all nodes
    net_f = np.sum(fext, axis=0)

    # Expected analytical force: Fx = 2 * P0 * R * H = 40,000 N
    expected_fx = 2.0 * p0 * radius * height
    assert net_f[0] == pytest.approx(expected_fx, rel=1e-3)
    assert net_f[1] == pytest.approx(0.0, abs=1e-5)
    assert net_f[2] == pytest.approx(0.0, abs=1e-5)
    assert total_force_mag == pytest.approx(expected_fx, rel=1e-3)

    # Verify diagnostic status
    status = engine.get_status()
    assert status["num_segments"] == len(segments)
    assert status["total_force"][0] == pytest.approx(expected_fx, rel=1e-3)


def test_load_pcyl_theta_dependent_pressure():
    """Verify theta-dependent aerodynamic pressure distribution P = P0 * cos(theta).

    For full cylinder shell of radius R and height H:
    Normal vector is n = (cos(theta), sin(theta), 0).
    Fx = integral_0^{2pi} (P0 * cos(theta)) * cos(theta) * R * H d(theta) = pi * P0 * R * H.
    Fy = integral_0^{2pi} (P0 * cos(theta)) * sin(theta) * R * H d(theta) = 0.
    Fz = 0.
    """
    radius = 0.5  # m
    height = 1.0  # m
    p0 = 20000.0  # 20 kPa
    n_theta = 72
    n_z = 8

    nodes, segments = _create_cylinder_mesh(
        radius=radius,
        height=height,
        n_theta=n_theta,
        n_z=n_z,
        theta_min=0.0,
        theta_max=2.0 * math.pi,
    )

    # Spatial pressure callable P_geom(r, theta, z) = P0 * cos(theta)
    def p_geom(r: float, theta: float, z: float) -> float:
        return p0 * math.cos(theta)

    params = PcylLoadParams(
        id=2,
        title="Cosine_Aero_Distribution",
        axis_origin=(0.0, 0.0, 0.0),
        axis_dir=(0.0, 0.0, 1.0),
        ref_dir=(1.0, 0.0, 0.0),
        p_geom=p_geom,
        segments=segments,
    )
    engine = PcylLoadEngine(params)

    fext = np.zeros_like(nodes)
    engine.apply_load(t=0.0, dt=1e-4, fext=fext, x=nodes)

    net_f = np.sum(fext, axis=0)

    # Theoretical Fx = pi * P0 * R * H = pi * 20000 * 0.5 * 1.0 = 10000 * pi ~ 31415.93 N
    expected_fx = math.pi * p0 * radius * height
    assert net_f[0] == pytest.approx(expected_fx, rel=1e-3)
    assert net_f[1] == pytest.approx(0.0, abs=1e-4)
    assert net_f[2] == pytest.approx(0.0, abs=1e-4)


def test_load_pcyl_z_dependent_hydrostatic():
    """Verify axial z-dependent hydrostatic pressure loading P(z) = rho * g * (H - z).

    For a half-cylinder tank wall:
    Net horizontal force: Fx = 0.5 * rho * g * H^2 * (2 * R).
    Moment about base (y-axis): My = - (1/6) * rho * g * H^3 * (2 * R).
    Center of pressure: z_cp = H / 3.
    """
    radius = 1.0  # m
    height = 6.0  # m
    rho = 1000.0  # kg/m^3
    g = 9.81  # m/s^2
    n_theta = 40
    n_z = 24

    nodes, segments = _create_cylinder_mesh(
        radius=radius,
        height=height,
        n_theta=n_theta,
        n_z=n_z,
        theta_min=-0.5 * math.pi,
        theta_max=0.5 * math.pi,
    )

    def p_hydro(r: float, theta: float, z: float) -> float:
        depth = max(height - z, 0.0)
        return rho * g * depth

    params = PcylLoadParams(
        id=3,
        title="Hydrostatic_Tank_Wall",
        axis_origin=(0.0, 0.0, 0.0),
        axis_dir=(0.0, 0.0, 1.0),
        ref_dir=(1.0, 0.0, 0.0),
        p_geom=p_hydro,
        segments=segments,
    )
    engine = PcylLoadEngine(params)

    fext = np.zeros_like(nodes)
    engine.apply_load(t=0.0, dt=1e-4, fext=fext, x=nodes)

    net_f = np.sum(fext, axis=0)

    # Theoretical resultant: Fx = 0.5 * rho * g * H^2 * (2 * R) = 0.5 * 1000 * 9.81 * 36 * 2 = 353,160 N
    expected_fx = 0.5 * rho * g * (height**2) * (2.0 * radius)
    assert net_f[0] == pytest.approx(expected_fx, rel=1e-3)
    assert net_f[1] == pytest.approx(0.0, abs=1e-4)

    # Total moment computed by engine: My = integral z * dFx
    expected_my = (1.0 / 6.0) * rho * g * (height**3) * (2.0 * radius)
    assert engine.total_moment[1] == pytest.approx(expected_my, rel=1e-2)

    # Center of pressure: z_cp = My / Fx = H / 3 = 2.0 m
    z_cp = engine.total_moment[1] / net_f[0]
    assert z_cp == pytest.approx(height / 3.0, rel=1e-2)


def test_load_pcyl_work_and_momentum_conservation():
    """Verify external work accumulation dW = sum(F_seg . v_centroid) * dt and momentum balance.

    For radially expanding cylinder under uniform internal pressure P0:
    Rate of work is dW/dt = P0 * dV/dt = P0 * (2 * pi * R * H * v_r).
    Total work over dt is Delta W = P0 * 2 * pi * R * H * v_r * dt.
    Net force on complete closed cylinder is identically zero (momentum conservation).
    """
    radius = 1.5  # m
    height = 3.0  # m
    p0 = 50000.0  # 50 kPa
    v_r = 2.0  # 2 m/s radial expansion velocity
    dt = 0.001  # 1 ms
    n_theta = 48
    n_z = 6

    nodes, segments = _create_cylinder_mesh(
        radius=radius,
        height=height,
        n_theta=n_theta,
        n_z=n_z,
        theta_min=0.0,
        theta_max=2.0 * math.pi,
    )

    # Radial velocity at each node: v = v_r * (cos(theta), sin(theta), 0)
    velocities = np.zeros_like(nodes)
    for i, (x, y, z) in enumerate(nodes):
        th = math.atan2(y, x)
        velocities[i, 0] = v_r * math.cos(th)
        velocities[i, 1] = v_r * math.sin(th)

    params = PcylLoadParams(
        id=4,
        title="Expanding_Cylinder_Work",
        axis_origin=(0.0, 0.0, 0.0),
        axis_dir=(0.0, 0.0, 1.0),
        p0=p0,
        segments=segments,
    )
    engine = PcylLoadEngine(params)

    fext = np.zeros_like(nodes)
    engine.apply_load(t=0.001, dt=dt, fext=fext, x=nodes, v=velocities)

    # 1. Momentum conservation on closed cylinder: net resultant force = 0
    net_f = np.sum(fext, axis=0)
    np.testing.assert_allclose(net_f, [0.0, 0.0, 0.0], atol=1e-5)

    # 2. Thermodynamic boundary work: Delta W = P0 * Delta V = P0 * (2 * pi * R * H * v_r * dt)
    cyl_area = 2.0 * math.pi * radius * height
    expected_work = p0 * cyl_area * v_r * dt  # 50000 * 2*pi*1.5*3 * 2 * 0.001 = 2827.43 J
    # Discretization of cylinder with 48-sided polygon introduces ~0.28% chord area difference
    assert engine.wfext == pytest.approx(expected_work, rel=0.01)


def test_build_pcyl_loads_from_model():
    """Verify model builder parses /LOAD/PCYL from model surfaces and skews."""
    from pyradioss.model.skew import SkewFrame

    model = Model()

    # Surface 10 with 2 triangular segments and 1 quad segment
    surf10 = Surface(
        id=10,
        title="Cyl_Skin_Surface",
        seg_nodes=[
            [1, 2, 3],        # triangle
            [2, 4, 3],        # triangle
            [3, 4, 5, 6],     # quad
        ],
    )
    model.surfaces[10] = surf10

    # Node coordinate mapping
    for i, nid in enumerate([1, 2, 3, 4, 5, 6]):
        model._id2idx[nid] = i

    # Skew 5 defining cylinder axis along X
    sf5 = SkewFrame(
        id=5,
        origin_card=np.array([1.0, 2.0, 3.0]),
        zaxis=np.array([1.0, 0.0, 0.0]),
    )
    model.skews.add(sf5)

    # /LOAD/PCYL 101 referencing Surface 10 and Skew 5
    pcyl_entity = PcylLoad(
        id=101,
        title="PCYL_101",
        surf_id=10,
        frame_id=5,
        xscale_r=0.001,
        xscale_t=1e-3,
        yscale_p=1.5e6,
    )
    model.pcyl_loads[101] = pcyl_entity

    engines = build_pcyl_loads(model)
    assert len(engines) == 1

    eng = engines[0]
    assert eng.id == 101
    assert len(eng.segments) == 3
    assert eng.segments[0].node_indices == (0, 1, 2)
    assert eng.segments[1].node_indices == (1, 3, 2)
    assert eng.segments[2].node_indices == (2, 3, 4, 5)

    np.testing.assert_allclose(eng.axis_origin, [1.0, 2.0, 3.0])
    np.testing.assert_allclose(eng.axis_dir, [1.0, 0.0, 0.0])
    assert eng.scale_r == pytest.approx(0.001)
    assert eng.scale_t == pytest.approx(1e-3)
    assert eng.scale_p == pytest.approx(1.5e6)
