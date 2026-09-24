# tests/test_sph_extensions.py
# Port of OpenRadioss Fortran sources:
#   - engine/source/elements/sph/spsym.F (lines 34-150: SPSYMP)
#   - engine/source/elements/sph/sptemp.F (lines 32-231: SPGRADT, lines 241-483: SPLAPLT, lines 657-784: SPGTSYM)
#   - engine/source/elements/sph/soltosph.F (lines 39-507: SOLTOSPHF, lines 523-1311: SOLTOSPHP)
#   - engine/source/elements/sph/soltospha.F (lines 39-439: SOLTOSPHA)
#   - engine/source/elements/sph/soltosph_on1.F (lines 37-275: SOLTOSPH_ON1)
#   - engine/source/elements/spring/ruser32.F (lines 37-260: RUSER32)
#   - engine/source/elements/spring/ruser32ke3.F (lines 38-136: RUSER32KE3)
#   - engine/source/elements/spring/ruser32mat3.F (lines 28-50: RUSER32MAT3)
#   - starter/source/properties/spring/hm_read_prop32.F (lines 40-282: HM_READ_PROP32)
"""Unit tests for SPH Extensions and TYPE32 Pretensioner Spring.

Verifies:
  1. SPH symmetry planes and ghost particle generation (spsym.F, sptemp.F)
  2. SPH thermal conduction and energy conservation (sptemp.F)
  3. Solid-to-SPH adaptive conversion (soltosph.F, soltospha.F, soltosph_on1.F)
  4. TYPE32 pretensioner spring element all 4 ITYP formulations, sensor activation,
     stroke locking, force locking, implicit tangent, and time step (ruser32.F, ruser32ke3.F).
"""

import numpy as np
import pytest

from pyradioss.elements import spring, spring_pretensioner
from pyradioss.elements.spring_pretensioner import (
    SpringPretensioner,
    forces_pretensioner_type32,
    implicit_stiffness_type32,
    init_pretensioner_type32,
)
from pyradioss.engine.sph_engine import (
    SPHSymmetryPlane,
    SolidToSPHConverter,
    create_sph_ghost_particles,
    hex8_shape_functions,
    interpolate_solid_field,
    reflect_sph_gradient,
    sph_temperature_gradient,
    sph_thermal_conduction,
    tet4_shape_functions,
)
from pyradioss.model.entities import Property
from pyradioss.model.model import ElementGroup, Model


# ============================================================================
# 1. SPH Symmetry Planes & Ghost Particles (spsym.F, sptemp.F)
# ============================================================================


def test_sph_symmetry_plane_distance_and_reflection():
    """Verify signed distance and position reflection across symmetry planes.

    Cites: engine/source/elements/sph/spsym.F lines 116-121.
    """
    # Plane z = 0 with normal [0, 0, 1]
    plane = SPHSymmetryPlane(point=[0.0, 0.0, 0.0], normal=[0.0, 0.0, 1.0], islide=1)

    pts = np.array([
        [1.0, 2.0, 0.5],
        [0.0, 0.0, 1.2],
        [-1.0, 3.0, 0.0],
    ])
    d = plane.signed_distance(pts)
    assert np.allclose(d, [0.5, 1.2, 0.0])

    # Reflected positions: x_s = x - 2 * d * n
    pts_sym = plane.reflect_position(pts)
    assert np.allclose(pts_sym, [
        [1.0, 2.0, -0.5],
        [0.0, 0.0, -1.2],
        [-1.0, 3.0, 0.0],
    ])

    # Tilted plane: x + y = 2, normal [1/sqrt(2), 1/sqrt(2), 0]
    n_tilt = np.array([1.0, 1.0, 0.0]) / np.sqrt(2.0)
    p_tilt = np.array([1.0, 1.0, 0.0])
    plane_tilt = SPHSymmetryPlane(point=p_tilt, normal=n_tilt, islide=1)

    pt_test = np.array([[2.0, 2.0, 3.0]])
    d_tilt = plane_tilt.signed_distance(pt_test)
    assert np.isclose(d_tilt[0], np.sqrt(2.0))
    pt_sym = plane_tilt.reflect_position(pt_test)
    assert np.allclose(pt_sym, [[0.0, 0.0, 3.0]])


def test_sph_symmetry_plane_slip_vs_noslip_velocity():
    """Verify velocity reflection for slip (ISLIDE=1) vs no-slip (ISLIDE=0) conditions.

    Cites: engine/source/elements/sph/spsym.F lines 122-131.
    """
    plane_slip = SPHSymmetryPlane(point=[0.0, 0.0, 0.0], normal=[0.0, 0.0, 1.0], islide=1)
    plane_noslip = SPHSymmetryPlane(point=[0.0, 0.0, 0.0], normal=[0.0, 0.0, 1.0], islide=0)

    vel = np.array([[10.0, -5.0, -2.0]])

    # Slip: v_s = v - 2 * vn * n -> normal component inverted, tangential preserved
    v_slip = plane_slip.reflect_velocity(vel)
    assert np.allclose(v_slip, [[10.0, -5.0, 2.0]])

    # No-slip: v_s = -v -> full velocity inverted
    v_noslip = plane_noslip.reflect_velocity(vel)
    assert np.allclose(v_noslip, [[-10.0, 5.0, 2.0]])


def test_sph_gradient_reflection():
    """Verify gradient vector reflection across symmetry normal.

    Cites: engine/source/elements/sph/sptemp.F lines 742-751 (SPGTSYM).
    """
    grad = np.array([
        [2.0, 4.0, 6.0],
        [-1.0, 0.0, 3.0],
    ])
    normal = np.array([0.0, 0.0, 1.0])
    grad_sym = reflect_sph_gradient(grad, normal)

    # Tangential unchanged, normal inverted
    expected = np.array([
        [2.0, 4.0, -6.0],
        [-1.0, 0.0, -3.0],
    ])
    assert np.allclose(grad_sym, expected)


def test_create_sph_ghost_particles():
    """Verify ghost particle filtering within compact support cutoff (0 < d <= 2h).

    Cites: engine/source/elements/sph/spsym.F lines 97-150.
    """
    plane = SPHSymmetryPlane(point=[0.0, 0.0, 0.0], normal=[0.0, 0.0, 1.0], islide=1)
    h = 0.2

    # 3 particles:
    # 1: inside cutoff d = 0.1 <= 2h = 0.4 -> should generate ghost
    # 2: inside cutoff d = 0.3 <= 0.4 -> should generate ghost
    # 3: outside cutoff d = 0.5 > 0.4 -> should NOT generate ghost
    pos = np.array([
        [0.0, 0.0, 0.1],
        [0.5, 0.5, 0.3],
        [1.0, 1.0, 0.5],
    ])
    vel = np.array([
        [1.0, 2.0, -3.0],
        [0.0, 1.0, -1.0],
        [2.0, 0.0, -4.0],
    ])
    mass = np.array([0.01, 0.01, 0.01])
    rho = np.array([1000.0, 1000.0, 1000.0])
    h_arr = np.full(3, h)
    temp = np.array([300.0, 320.0, 350.0])

    ghosts = create_sph_ghost_particles(pos, vel, mass, rho, h_arr, [plane], temp=temp)

    assert len(ghosts['pos']) == 2
    assert np.allclose(ghosts['source_idx'], [0, 1])
    assert np.allclose(ghosts['pos'][:, 2], [-0.1, -0.3])
    assert np.allclose(ghosts['vel'][:, 2], [3.0, 1.0])
    assert np.allclose(ghosts['temp'], [300.0, 320.0])
    assert np.allclose(ghosts['mass'], [0.01, 0.01])


# ============================================================================
# 2. SPH Thermal Conduction (sptemp.F)
# ============================================================================


def test_sph_temperature_gradient_linear_field():
    """Verify SPH temperature gradient reproduces constant gradient of a linear field.

    Cites: engine/source/elements/sph/sptemp.F lines 32-231 (SPGRADT).
    """
    # Create regular 3D lattice in [-0.2, 0.2]^3
    dx = 0.05
    coords_1d = np.arange(-0.2, 0.201, dx)
    gx, gy, gz = np.meshgrid(coords_1d, coords_1d, coords_1d, indexing='ij')
    pos = np.column_stack([gx.ravel(), gy.ravel(), gz.ravel()])

    rho0 = 1000.0
    v_cell = dx ** 3
    mass = np.full(len(pos), rho0 * v_cell)
    rho = np.full(len(pos), rho0)
    h = 1.3 * dx
    h_arr = np.full(len(pos), h)

    # Linear temperature field: T(x, y, z) = 10 * x + 20 * y - 15 * z + 300
    exact_grad = np.array([10.0, 20.0, -15.0])
    temp = 10.0 * pos[:, 0] + 20.0 * pos[:, 1] - 15.0 * pos[:, 2] + 300.0

    grad_t = sph_temperature_gradient(pos, temp, mass, rho, h_arr)

    # Check interior particles (at least 2h away from boundaries)
    bound = 0.2 - 2.5 * h
    interior = (
        (np.abs(pos[:, 0]) <= bound)
        & (np.abs(pos[:, 1]) <= bound)
        & (np.abs(pos[:, 2]) <= bound)
    )
    assert np.sum(interior) > 0

    mean_grad = np.mean(grad_t[interior], axis=0)
    assert np.allclose(mean_grad, exact_grad, rtol=0.08, atol=0.5)


def test_sph_thermal_conduction_conservation_and_second_law():
    """Verify exact thermal energy conservation and heat flow direction.

    Cites: engine/source/elements/sph/sptemp.F lines 241-483 (SPLAPLT).
    """
    # 2 interacting particles at distance r = 0.05 < 2h = 0.2
    pos = np.array([
        [0.0, 0.0, 0.0],
        [0.05, 0.0, 0.0],
    ])
    temp = np.array([400.0, 300.0])  # particle 0 is hot, particle 1 is cold
    mass = np.array([0.02, 0.03])
    rho = np.array([1000.0, 1000.0])
    h_arr = np.array([0.1, 0.1])
    conductivity = 50.0  # W/(m K)
    specific_heat = 500.0  # J/(kg K)

    dT_dt, q_rates, temp_new, e_ex = sph_thermal_conduction(
        pos, temp, mass, rho, h_arr, conductivity, specific_heat, dt=0.01
    )

    # 1. Conservation: sum of heat rates must be strictly 0 (within machine epsilon)
    assert np.isclose(np.sum(q_rates), 0.0, atol=1e-12)

    # 2. Second Law: hot particle cools down, cold particle heats up
    assert dT_dt[0] < 0.0
    assert dT_dt[1] > 0.0
    assert temp_new[0] < temp[0]
    assert temp_new[1] > temp[1]

    # 3. Energy conservation in time step: dE = sum(m * cv * dT) == 0
    delta_energy = np.sum(mass * specific_heat * (temp_new - temp))
    assert np.isclose(delta_energy, 0.0, atol=1e-12)


def test_sph_thermal_conduction_with_symmetry_plane():
    """Verify symmetry plane reflects thermal interactions without leakage.

    Cites: engine/source/elements/sph/sptemp.F lines 657-784.
    """
    plane = SPHSymmetryPlane(point=[0.0, 0.0, 0.0], normal=[0.0, 0.0, 1.0])
    pos = np.array([[0.0, 0.0, 0.05]])
    temp = np.array([350.0])
    mass = np.array([0.01])
    rho = np.array([1000.0])
    h_arr = np.array([0.1])
    conductivity = 10.0
    specific_heat = 400.0

    # With symmetry plane, particle interacts with its own mirror ghost at same temperature -> zero net heat flux
    dT_dt, q_rates, temp_new, _ = sph_thermal_conduction(
        pos, temp, mass, rho, h_arr, conductivity, specific_heat, dt=0.01, planes=[plane]
    )
    assert np.isclose(q_rates[0], 0.0, atol=1e-12)
    assert np.isclose(dT_dt[0], 0.0, atol=1e-12)


# ============================================================================
# 3. Solid-to-SPH Adaptive Conversion (soltosph*.F)
# ============================================================================


def test_hex8_shape_functions_partition_of_unity():
    """Verify Hex8 shape functions sum to 1.0 and delta at nodes.

    Cites: engine/source/elements/sph/soltosph.F lines 282-289.
    """
    for xi, eta, zeta in [(0.0, 0.0, 0.0), (0.5, -0.3, 0.7), (-0.8, 0.2, -0.4)]:
        phi = hex8_shape_functions(xi, eta, zeta)
        assert np.isclose(np.sum(phi), 1.0)

    # Node 1 at (-1, -1, -1)
    phi_n1 = hex8_shape_functions(-1.0, -1.0, -1.0)
    assert np.isclose(phi_n1[0], 1.0)
    assert np.allclose(phi_n1[1:], 0.0)


def test_tet4_shape_functions_partition_of_unity():
    """Verify Tet4 shape functions sum to 1.0 and delta at vertices.

    Cites: engine/source/elements/sph/soltosph.F lines 221-224.
    """
    for xi, eta, zeta in [(0.25, 0.25, 0.25), (0.1, 0.2, 0.3)]:
        phi = tet4_shape_functions(xi, eta, zeta)
        assert np.isclose(np.sum(phi), 1.0)

    # Vertex 1 at (1, 0, 0)
    phi_v1 = tet4_shape_functions(1.0, 0.0, 0.0)
    assert np.isclose(phi_v1[0], 1.0)
    assert np.allclose(phi_v1[1:], 0.0)


def test_solid_to_sph_hex8_conversion():
    """Verify Hex8 solid element conversion to SPH particles with exact mass & momentum conservation.

    Cites: engine/source/elements/sph/soltosph.F lines 256-330 and soltosph_on1.F lines 185-265.
    """
    converter = SolidToSPHConverter(n_dir=2)  # 2x2x2 = 8 particles

    # Unit cube [0, 1]^3
    node_coords = np.array([
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 1.0],
        [1.0, 0.0, 1.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 1.0, 1.0],
        [1.0, 1.0, 1.0],
        [1.0, 1.0, 0.0],
    ])
    # Rigid body translation velocity + linear shear
    node_vel = np.tile([10.0, -5.0, 2.0], (8, 1))
    solid_mass = 8.0
    solid_rho = 1000.0
    solid_energy = 120.0
    solid_stress = np.array([1e6, 2e6, 3e6, 0.0, 0.0, 0.0])
    solid_plastic_strain = 0.05

    res = converter.convert_element(
        solid_id=101,
        solid_type='hex8',
        node_coords=node_coords,
        node_velocities=node_vel,
        solid_mass=solid_mass,
        solid_rho=solid_rho,
        solid_energy=solid_energy,
        solid_stress=solid_stress,
        solid_plastic_strain=solid_plastic_strain,
    )

    # 1. 8 particles created
    assert len(res['pos']) == 8

    # 2. Particles are strictly inside the cube [0, 1]^3
    assert np.all((res['pos'] >= 0.0) & (res['pos'] <= 1.0))

    # 3. Exact mass conservation: sum(m_p) == M_solid
    assert np.isclose(np.sum(res['mass']), solid_mass)
    assert np.allclose(res['rho'], solid_rho)

    # 4. Exact momentum conservation: sum(m_p * v_p) == M_solid * v_rigid
    total_momentum_sph = np.sum(res['mass'][:, None] * res['vel'], axis=0)
    expected_momentum = solid_mass * np.array([10.0, -5.0, 2.0])
    assert np.allclose(total_momentum_sph, expected_momentum)

    # 5. State transfer: energy, stress, plastic strain
    assert np.isclose(np.sum(res['energy']), solid_energy)
    assert np.allclose(res['plastic_strain'], solid_plastic_strain)
    assert np.allclose(res['stress'][0], solid_stress)

    # 6. Kinetic energy accounting: for rigid body motion, E_k,sph == E_k,solid -> delta_e_hour == 0
    assert np.isclose(res['delta_e_hour'], 0.0, atol=1e-10)


def test_solid_to_sph_tet4_conversion():
    """Verify Tet4 solid element conversion to SPH particles.

    Cites: engine/source/elements/sph/soltosph.F lines 197-250 and soltospha.F lines 205-254.
    """
    converter = SolidToSPHConverter(n_dir=1)  # 1 particle at barycenter

    node_coords = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ])
    node_vel = np.tile([5.0, 0.0, 0.0], (4, 1))
    solid_mass = 1.0
    solid_rho = 2700.0

    res = converter.convert_element(
        solid_id=201,
        solid_type='tet4',
        node_coords=node_coords,
        node_velocities=node_vel,
        solid_mass=solid_mass,
        solid_rho=solid_rho,
        solid_energy=50.0,
    )

    assert len(res['pos']) == 1
    # Barycenter of unit tet: (0.25, 0.25, 0.25)
    assert np.allclose(res['pos'][0], [0.25, 0.25, 0.25])
    assert np.isclose(res['mass'][0], solid_mass)
    assert np.allclose(res['vel'][0], [5.0, 0.0, 0.0])


def test_solid_to_sph_kinetic_energy_dissipation_accounting():
    """Verify kinetic energy difference is correctly booked to numerical dissipation (EN ledger).

    Cites: engine/source/elements/sph/soltosph_on1.F line 265 (EHOURT = EHOURT + 0.5*DM*VI2 - EK).
    """
    converter = SolidToSPHConverter(n_dir=2)
    node_coords = np.array([
        [0.0, 0.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0, 1.0], [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0], [0.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 0.0],
    ])
    # Non-uniform velocity (internal strain rate/divergence):
    # node velocities differ -> discrete particles will have lower kinetic energy than nodal masses
    node_vel = np.array([
        [-10.0, 0.0, 0.0], [10.0, 0.0, 0.0], [-10.0, 0.0, 0.0], [10.0, 0.0, 0.0],
        [-10.0, 0.0, 0.0], [10.0, 0.0, 0.0], [-10.0, 0.0, 0.0], [10.0, 0.0, 0.0],
    ])
    solid_mass = 8.0
    res = converter.convert_element(
        solid_id=301,
        solid_type='hex8',
        node_coords=node_coords,
        node_velocities=node_vel,
        solid_mass=solid_mass,
        solid_rho=1000.0,
    )
    # Discretization kinetic energy loss must be positive and accounted for
    assert res['delta_e_hour'] > 0.0
    assert converter.accumulated_e_hour > 0.0


# ============================================================================
# 4. Pretensioner Spring Element TYPE32 (ruser32.F, hm_read_prop32.F)
# ============================================================================


def test_spring_type32_linear_pretension_ityp1():
    """Verify TYPE32 linear pretension law FF = F1 + STIF1 * X.

    Cites: engine/source/elements/spring/ruser32.F lines 187-202 and
           starter/source/properties/spring/hm_read_prop32.F lines 241-248.
    """
    stif0 = 100.0
    stif1 = 500.0
    f1 = 200.0
    spring_pre = SpringPretensioner(stif0=stif0, stif1=stif1, f1=f1, ityp=1, ilock=0)

    # Step 1: retraction stroke Ldot = -2.0 m/s for dt = 0.05 s -> dX = -0.1 m
    dt = 0.05
    Ldot = -2.0
    force = spring_pre.step(L=1.0, Ldot=Ldot, dt=dt, t=0.05)

    # X = -0.1 m
    # FF = F1 + STIF1 * X = 200.0 + 500.0 * (-0.1) = 150.0
    # Elastic F = STIF0 * dX = 100 * (-0.1) = -10.0
    # Since FF > 0 and unlocked -> Force = max(FF, F) = 150.0
    assert np.isclose(spring_pre.uvar1, -0.1)
    assert np.isclose(force, 150.0)


def test_spring_type32_displacement_dependent_ityp2():
    """Verify TYPE32 non-linear displacement-dependent law FF = Scale_f * func1(X * Scale_d).

    Cites: engine/source/elements/spring/ruser32.F lines 203-219.
    """
    # Simple linear function mock func1(x) = 1000.0 * abs(x)
    class LinearFunc:
        def eval(self, x):
            return 1000.0 * abs(x)

    func1 = LinearFunc()
    spring_pre = SpringPretensioner(
        stif0=50.0, ityp=2, scale_d=1.0, scale_f=1.0, func1=func1, ilock=0
    )

    # Stroke dX = -0.05 m
    force = spring_pre.step(L=1.0, Ldot=-1.0, dt=0.05, t=0.05)
    # FF = 1000 * 0.05 = 50.0
    assert np.isclose(force, 50.0)


def test_spring_type32_time_dependent_ityp3():
    """Verify TYPE32 time-dependent pretension law F0 = Scale_f * func2(tacti * Scale_t).

    Cites: engine/source/elements/spring/ruser32.F lines 220-230.
    """
    class TimeCurve:
        def eval(self, t):
            # Ramp force up to 300 N at t = 0.01 s
            return min(300.0, 30000.0 * t)

    func2 = TimeCurve()
    spring_pre = SpringPretensioner(
        stif0=100.0, ityp=3, scale_t=1.0, scale_f=1.0, func2=func2, ilock=0
    )

    # At t = 0.005 s: F0 = 30000 * 0.005 = 150.0 N
    force = spring_pre.step(L=1.0, Ldot=0.0, dt=0.005, t=0.005)
    assert np.isclose(force, 150.0)

    # At t = 0.02 s: F0 = 300.0 N
    force = spring_pre.step(L=1.0, Ldot=0.0, dt=0.015, t=0.02)
    assert np.isclose(force, 300.0)


def test_spring_type32_combined_ityp4():
    """Verify TYPE32 combined time and displacement law FF = F0(t) * func1(X).

    Cites: engine/source/elements/spring/ruser32.F lines 231-250.
    """
    class TimeCurve:
        def eval(self, t):
            return 200.0  # constant F0 = 200 N

    class DispCurve:
        def eval(self, x):
            return 1.5    # factor 1.5

    spring_pre = SpringPretensioner(
        stif0=100.0, ityp=4, func1=DispCurve(), func2=TimeCurve(), ilock=0
    )

    force = spring_pre.step(L=1.0, Ldot=-0.5, dt=0.01, t=0.01)
    # FF = 200 * 1.5 = 300.0 N
    assert np.isclose(force, 300.0)


def test_spring_type32_sensor_trigger_delay():
    """Verify pretensioner stays in passive STIF0 mode until sensor fires.

    Cites: engine/source/elements/spring/ruser32.F lines 162-184.
    """
    stif0 = 50.0
    spring_pre = SpringPretensioner(
        stif0=stif0, stif1=300.0, f1=100.0, ityp=1, sens_id=10
    )

    # Cycle 1: t = 0.01, sensor not active yet -> only stif0 tracking
    f1 = spring_pre.step(L=1.0, Ldot=1.0, dt=0.01, t=0.01, sensor_active=False)
    assert spring_pre.uvar2 == 0.0  # inactive
    assert np.isclose(f1, stif0 * 0.01 * 1.0)  # 0.5 N

    # Cycle 2: t = 0.02, sensor fires at tf = 0.02
    f2 = spring_pre.step(L=1.0, Ldot=-1.0, dt=0.01, t=0.02, sensor_active=True, fire_time=0.02)
    assert spring_pre.uvar2 == 1.0  # active
    # Active now -> force evaluates FF = F1 + STIF1 * X = 100 + 300 * (-0.01) = 97.0 N
    assert np.isclose(f2, 97.0)


def test_spring_type32_locking_mechanisms():
    """Verify stroke limit locking (X < D1) and tension force override locking (ILOCK=2).

    Cites: engine/source/elements/spring/ruser32.F lines 192, 208-209.
    """
    # 1. Stroke limit lock: d1 = -0.1 m (max retraction 100 mm)
    spring_stroke = SpringPretensioner(
        stif0=100.0, ityp=2, d1=0.1,  # stored internally as -0.1
        func1=lambda x: 250.0, ilock=0
    )
    # Retract past -0.1 m: Ldot = -2.0, dt = 0.06 -> X = -0.12 m < -0.1 m
    spring_stroke.step(L=1.0, Ldot=-2.0, dt=0.06, t=0.06)
    assert spring_stroke.uvar3 == 1.0  # locked!

    # 2. Tension override lock (ILOCK = 2):
    # If external tension force exceeds pretension force FF
    spring_force_lock = SpringPretensioner(
        stif0=1000.0, stif1=0.0, f1=100.0, ityp=1, ilock=2
    )
    # Stretch spring rapidly: Ldot = 10.0, dt = 0.02 -> dF = 1000 * 0.2 = 200 N > FF (100 N)
    spring_force_lock.step(L=1.0, Ldot=10.0, dt=0.02, t=0.02)
    assert spring_force_lock.uvar3 == 1.0  # locked!


def test_spring_type32_critical_dt_and_tangent():
    """Verify critical explicit time step and implicit tangent stiffness matrix.

    Cites: engine/source/elements/spring/ruser32mat3.F and ruser32ke3.F.
    """
    mass = 0.04  # kg
    stif0 = 400.0  # N/m
    spring_pre = SpringPretensioner(stif0=stif0, mass=mass)

    # omega = 2 * sqrt(K / M) = 2 * sqrt(400 / 0.04) = 200 rad/s
    # dt_crit = 2 / omega = 2 / 200 = 0.01 s
    dt_crit = spring_pre.critical_dt()
    assert np.isclose(dt_crit, 0.01)

    # Model assembly check
    p32 = Property(id=1, type=32, params={
        "mass": mass, "stif0": stif0, "stif1": 200.0, "ityp": 1, "f1": 150.0
    })
    m = Model()
    x0 = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    m.add_nodes(np.array([1, 2]), x0)
    g = ElementGroup(ids=np.array([1]), conn=np.array([[0, 1]]), part=np.array([0]))
    g.state["slices"] = [(slice(0, 1), None, p32)]
    m.springs = g

    spring.init_group(g, m, None)
    ke, edofs = implicit_stiffness_type32(g, x0, np.array([0]), ikgeo=1)

    assert ke.shape == (1, 6, 6)
    # Material tangent is symmetric
    assert np.allclose(ke[0], ke[0].T, atol=1e-12)
    # Eigenvalues: 5 zero modes, 1 axial mode equal to 2 * Kt
    eigvals = np.linalg.eigvalsh(ke[0])
    assert np.all(np.abs(eigvals[:5]) < 1e-10)
    assert np.isclose(eigvals[5], 2.0 * stif0)
