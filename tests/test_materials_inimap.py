"""Unit tests for /INIMAP1D and /INIMAP2D blast field mapping.

Upstream Fortran references:
- starter/source/initial_conditions/inimap/ini_inimap1d.F
- starter/source/initial_conditions/inimap/hm_read_inimap1d.F
- starter/source/initial_conditions/inimap/hm_read_inimap2d.F
"""

import math
import numpy as np
import pytest
from scipy import integrate

from pyradioss.materials.inimap import (
    Inimap1DParams,
    Inimap1DProjection,
    Inimap2DParams,
    InimapFormulation,
    InimapState,
    compute_integrated_quantities,
    map_inimap1d,
    map_inimap2d,
)


class TestInimap1D:
    """Test 1D blast solution profile mapping."""

    @pytest.fixture
    def blast_1d_params(self):
        radii = np.array([0.0, 0.5, 1.0, 2.0], dtype=float)
        density = np.array([10.0, 8.0, 4.0, 1.225], dtype=float)
        velocity = np.array([0.0, 500.0, 200.0, 0.0], dtype=float)
        pressure = np.array([1.0e7, 5.0e6, 1.0e6, 101325.0], dtype=float)
        eint = np.array([2.0e6, 1.5e6, 5.0e5, 2.0e5], dtype=float)

        return Inimap1DParams(
            origin=[0.0, 0.0, 0.0],
            projection=Inimap1DProjection.SPHERICAL,
            radii=radii,
            density=density,
            velocity=velocity,
            pressure=pressure,
            internal_energy=eint,
        )

    def test_1d_radial_interpolation_scalar_values(self, blast_1d_params):
        """Test linear interpolation and constant clamping of density, pressure, energy."""
        # Test knot point r = 0.5
        pt_knot = np.array([[0.5, 0.0, 0.0]])
        state_knot = map_inimap1d(blast_1d_params, pt_knot)
        assert math.isclose(state_knot.density[0], 8.0, rel_tol=1e-6)
        assert math.isclose(state_knot.pressure[0], 5.0e6, rel_tol=1e-6)
        assert math.isclose(state_knot.internal_energy[0], 1.5e6, rel_tol=1e-6)

        # Test midpoint r = 0.25 (linear between 10.0 and 8.0 -> 9.0)
        pt_mid = np.array([[0.0, 0.25, 0.0]])
        state_mid = map_inimap1d(blast_1d_params, pt_mid)
        assert math.isclose(state_mid.density[0], 9.0, rel_tol=1e-6)
        assert math.isclose(state_mid.pressure[0], 7.5e6, rel_tol=1e-6)

        # Test midpoint r = 0.75 (linear between 8.0 and 4.0 -> 6.0)
        pt_mid2 = np.array([[0.0, 0.0, 0.75]])
        state_mid2 = map_inimap1d(blast_1d_params, pt_mid2)
        assert math.isclose(state_mid2.density[0], 6.0, rel_tol=1e-6)
        assert math.isclose(state_mid2.pressure[0], 3.0e6, rel_tol=1e-6)

        # Test extrapolation outside r > 2.0 -> clamped to r=2.0 value
        pt_far = np.array([[3.0, 0.0, 0.0]])
        state_far = map_inimap1d(blast_1d_params, pt_far)
        assert math.isclose(state_far.density[0], 1.225, rel_tol=1e-6)
        assert math.isclose(state_far.pressure[0], 101325.0, rel_tol=1e-6)

    def test_radial_velocity_projection(self, blast_1d_params):
        """Test spherical velocity vector orientation v_3d = v(r) * e_r."""
        # Point along (3, 4, 0) direction at distance r = 0.5
        # v(0.5) = 500 m/s
        # e_r = (0.6, 0.8, 0.0) -> v_3d = (300, 400, 0)
        pt = np.array([[0.3, 0.4, 0.0]])
        state = map_inimap1d(blast_1d_params, pt)
        np.testing.assert_allclose(state.velocity[0], [300.0, 400.0, 0.0], atol=1e-6)

        # Origin point r = 0 -> velocity must be zero
        pt_origin = np.array([[0.0, 0.0, 0.0]])
        state_origin = map_inimap1d(blast_1d_params, pt_origin)
        np.testing.assert_allclose(state_origin.velocity[0], [0.0, 0.0, 0.0], atol=1e-6)

    def test_cylindrical_and_planar_projections(self):
        """Test cylindrical (distance to axis) and planar (normal distance) mapping."""
        # Planar projection along normal n = (0, 1, 0)
        planar_params = Inimap1DParams(
            origin=[0.0, 0.0, 0.0],
            projection=Inimap1DProjection.PLANAR,
            normal=[0.0, 1.0, 0.0],
            radii=np.array([0.0, 1.0]),
            density=np.array([2.0, 1.0]),
            velocity=np.array([100.0, 50.0]),
        )
        pts = np.array([
            [5.0, 0.5, -2.0],  # y = 0.5 -> r = 0.5
            [-3.0, 1.0, 10.0], # y = 1.0 -> r = 1.0
        ])
        state = map_inimap1d(planar_params, pts)
        assert math.isclose(state.density[0], 1.5, rel_tol=1e-6)
        np.testing.assert_allclose(state.velocity[0], [0.0, 75.0, 0.0], atol=1e-6)
        assert math.isclose(state.density[1], 1.0, rel_tol=1e-6)
        np.testing.assert_allclose(state.velocity[1], [0.0, 50.0, 0.0], atol=1e-6)

        # Cylindrical projection with axis along Z
        cyl_params = Inimap1DParams(
            origin=[0.0, 0.0, 0.0],
            axis_point=[0.0, 0.0, 1.0],
            projection=Inimap1DProjection.CYLINDRICAL,
            radii=np.array([0.0, 2.0]),
            velocity=np.array([0.0, 200.0]),
        )
        # Point at x=1.5, y=2.0, z=99.0 -> r = sqrt(1.5^2 + 2^2) = 2.5 (clamped to 2.0 -> v=200)
        # unit vector = (1.5/2.5, 2.0/2.5, 0) = (0.6, 0.8, 0)
        pts_cyl = np.array([[1.5, 2.0, 99.0]])
        state_cyl = map_inimap1d(cyl_params, pts_cyl)
        np.testing.assert_allclose(state_cyl.velocity[0], [120.0, 160.0, 0.0], atol=1e-6)

    def test_single_coordinate_tuple(self, blast_1d_params):
        """Test single coordinate 1D array mapping."""
        coord = np.array([0.5, 0.0, 0.0])
        state = map_inimap1d(blast_1d_params, coord)
        assert state.density == 8.0
        assert len(state.velocity) == 3


class TestInimap2D:
    """Test 2D axisymmetric blast solution profile mapping."""

    @pytest.fixture
    def blast_2d_params(self):
        # 2D Grid: r in [0, 1, 2], z in [-1, 0, 1]
        r_grid = np.array([0.0, 1.0, 2.0])
        z_grid = np.array([-1.0, 0.0, 1.0])

        nr = len(r_grid)
        nz = len(z_grid)

        # Known bilinear function: f(r, z) = 10.0 + 5.0*r - 2.0*z + 3.0*r*z
        density = np.zeros((nr, nz))
        for i in range(nr):
            for j in range(nz):
                density[i, j] = 10.0 + 5.0 * r_grid[i] - 2.0 * z_grid[j] + 3.0 * r_grid[i] * z_grid[j]

        # Radial velocity v_r = 100 * r
        # Axial velocity v_z = 50 * z
        vr = np.zeros((nr, nz))
        vz = np.zeros((nr, nz))
        for i in range(nr):
            for j in range(nz):
                vr[i, j] = 100.0 * r_grid[i]
                vz[i, j] = 50.0 * z_grid[j]

        return Inimap2DParams(
            origin=[0.0, 0.0, 0.0],
            axis_vector=[0.0, 0.0, 1.0],  # Z-axis is symmetry axis
            r_grid=r_grid,
            z_grid=z_grid,
            density=density,
            velocity_r=vr,
            velocity_z=vz,
        )

    def test_2d_bilinear_interpolation_exact(self, blast_2d_params):
        """Verify exact recovery of bilinear function on 2D grid cells."""
        # Query at r = 0.6, z = -0.4
        # Expected density = 10.0 + 5.0*(0.6) - 2.0*(-0.4) + 3.0*(0.6)*(-0.4)
        #                  = 10.0 + 3.0 + 0.8 - 0.72 = 13.08
        # Query 3D coords: x=0.6, y=0.0, z=-0.4
        pts = np.array([[0.6, 0.0, -0.4]])
        state = map_inimap2d(blast_2d_params, pts)
        expected_rho = 10.0 + 5.0 * 0.6 - 2.0 * (-0.4) + 3.0 * 0.6 * (-0.4)
        assert math.isclose(state.density[0], expected_rho, rel_tol=1e-6)

        # Check velocity: vr = 100 * 0.6 = 60, vz = 50 * (-0.4) = -20
        # e_r = (1, 0, 0), e_z = (0, 0, 1)
        np.testing.assert_allclose(state.velocity[0], [60.0, 0.0, -20.0], atol=1e-6)

    def test_axisymmetric_3d_rotation(self, blast_2d_params):
        """Verify cylindrical velocity components correctly rotate in 3D around symmetry axis."""
        # Query at distance r = 1.0 at 45 degrees in xy-plane: x = sqrt(2)/2, y = sqrt(2)/2, z = 0.0
        cos45 = math.sqrt(2) / 2
        pts = np.array([[cos45, cos45, 0.0]])
        state = map_inimap2d(blast_2d_params, pts)

        # vr(1.0, 0.0) = 100.0, vz(1.0, 0.0) = 0.0
        # v_3d = 100.0 * (cos45, cos45, 0.0)
        np.testing.assert_allclose(state.velocity[0], [100.0 * cos45, 100.0 * cos45, 0.0], atol=1e-6)

    def test_tilted_symmetry_axis(self):
        """Verify coordinate transformation with non-trivial origin and tilted symmetry axis."""
        # Symmetry axis pointing along (1, 1, 0)
        origin = np.array([1.0, 2.0, 3.0])
        u_axis = np.array([1.0, 1.0, 0.0]) / math.sqrt(2)

        params = Inimap2DParams(
            origin=origin,
            axis_vector=u_axis,
            r_grid=np.array([0.0, 5.0]),
            z_grid=np.array([-5.0, 5.0]),
            density=np.ones((2, 2)) * 2.5,
        )

        # Point along the axis at distance +2.0 from origin
        pt_on_axis = origin + 2.0 * u_axis
        r_cyl, z_cyl, _, e_z = params.transform_coords(np.array([pt_on_axis]))
        assert math.isclose(r_cyl[0], 0.0, abs_tol=1e-10)
        assert math.isclose(z_cyl[0], 2.0, rel_tol=1e-6)
        np.testing.assert_allclose(e_z[0], u_axis, atol=1e-6)

        # Point perpendicular to axis along (0, 0, 1) by distance 3.0
        pt_perp = origin + 3.0 * np.array([0.0, 0.0, 1.0])
        r_cyl2, z_cyl2, e_r2, _ = params.transform_coords(np.array([pt_perp]))
        assert math.isclose(r_cyl2[0], 3.0, rel_tol=1e-6)
        assert math.isclose(z_cyl2[0], 0.0, abs_tol=1e-10)
        np.testing.assert_allclose(e_r2[0], [0.0, 0.0, 1.0], atol=1e-6)


class TestMassAndEnergyConservation:
    """Test integral conservation over mapped 3D volumes."""

    def test_spherical_volume_integral_conservation(self):
        """Verify numerical summation over concentric shells matches continuous analytic integral."""
        # Density profile: rho(r) = rho0 * (1 - (r/R)^2)
        rho0 = 5.0
        r_max = 1.0
        radii_fine = np.linspace(0.0, r_max, 501)
        rho_fine = rho0 * (1.0 - (radii_fine / r_max) ** 2)

        params = Inimap1DParams(
            origin=[0.0, 0.0, 0.0],
            projection=Inimap1DProjection.SPHERICAL,
            radii=radii_fine,
            density=rho_fine,
            velocity=np.zeros_like(radii_fine),
            internal_energy=np.ones_like(radii_fine) * 1e5,
        )

        # Analytic continuous mass:
        # M = int_0^R rho(r) * 4*pi*r^2 dr = 4*pi*rho0 * int_0^R (r^2 - r^4/R^2) dr
        #   = 4*pi*rho0 * [R^3/3 - R^3/5] = 4*pi*rho0 * R^3 * (2/15) = 8*pi*rho0*R^3 / 15
        expected_mass = 8.0 * math.pi * rho0 * (r_max ** 3) / 15.0

        # Create discrete concentric shell centers and volumes
        n_shells = 200
        dr = r_max / n_shells
        r_shells = np.linspace(dr / 2, r_max - dr / 2, n_shells)
        # Shell volumes: dV = 4/3*pi*((r+dr/2)^3 - (r-dr/2)^3)
        vols = 4.0 / 3.0 * math.pi * ((r_shells + dr / 2) ** 3 - (r_shells - dr / 2) ** 3)

        # Shell sample points in 3D (along x-axis)
        coords = np.column_stack([r_shells, np.zeros(n_shells), np.zeros(n_shells)])

        state = map_inimap1d(params, coords)
        quantities = compute_integrated_quantities(state, vols)

        numerical_mass = quantities["total_mass"]
        assert math.isclose(numerical_mass, expected_mass, rel_tol=0.001)

        # Check internal energy: E_int = M * e0
        expected_eint = expected_mass * 1e5
        assert math.isclose(quantities["internal_energy"], expected_eint, rel_tol=0.001)

    def test_scale_factors(self):
        """Verify scale factors fac_rho, fac_pres_ener, fac_vel."""
        params = Inimap1DParams(
            radii=np.array([0.0, 1.0]),
            density=np.array([2.0, 2.0]),
            velocity=np.array([10.0, 10.0]),
            pressure=np.array([100.0, 100.0]),
            internal_energy=np.array([50.0, 50.0]),
            fac_rho=2.5,
            fac_pres_ener=10.0,
            fac_vel=3.0,
        )
        pt = np.array([[0.5, 0.0, 0.0]])
        state = map_inimap1d(params, pt)
        assert math.isclose(state.density[0], 5.0, rel_tol=1e-6)
        assert math.isclose(state.pressure[0], 1000.0, rel_tol=1e-6)
        assert math.isclose(state.internal_energy[0], 500.0, rel_tol=1e-6)
        assert math.isclose(state.velocity[0, 0], 30.0, rel_tol=1e-6)

    def test_cylindrical_2d_volume_integral_conservation(self):
        """Verify mass and kinetic energy conservation for a 2D axisymmetric rotating/moving blast cylinder."""
        rho0 = 2.0
        v0 = 50.0
        omega = 100.0
        r_max = 1.0
        h = 2.0  # z in [0, 2]

        nr, nz = 51, 51
        r_grid = np.linspace(0.0, r_max, nr)
        z_grid = np.linspace(0.0, h, nz)

        # vr = omega * r, vz = v0, rho = rho0
        rr, zz = np.meshgrid(r_grid, z_grid, indexing='ij')
        rho_grid = np.full((nr, nz), rho0)
        vr_grid = omega * rr
        vz_grid = np.full((nr, nz), v0)
        eint_grid = np.full((nr, nz), 1.0e4)

        params = Inimap2DParams(
            origin=[0.0, 0.0, 0.0],
            axis_vector=[0.0, 0.0, 1.0],
            r_grid=r_grid,
            z_grid=z_grid,
            density=rho_grid,
            velocity_r=vr_grid,
            velocity_z=vz_grid,
            internal_energy=eint_grid,
        )

        # Analytic mass: M = pi * r_max^2 * h * rho0
        expected_mass = math.pi * (r_max ** 2) * h * rho0

        # Analytic kinetic energy:
        # E_kin = int_0^h dz int_0^R dr (2*pi*r) * 0.5 * rho0 * ( (omega*r)^2 + v0^2 )
        #       = pi * rho0 * h * [ omega^2 * R^4 / 4 + v0^2 * R^2 / 2 ]
        expected_ekin = math.pi * rho0 * h * (omega**2 * (r_max**4) / 4.0 + (v0**2) * (r_max**2) / 2.0)

        # Discrete rings in (r, z):
        dr = r_max / (nr - 1)
        dz = h / (nz - 1)
        r_cents = 0.5 * (r_grid[:-1] + r_grid[1:])
        z_cents = 0.5 * (z_grid[:-1] + z_grid[1:])
        rr_c, zz_c = np.meshgrid(r_cents, z_cents, indexing='ij')

        # Cell volumes: dV = 2 * pi * r_c * dr * dz
        vols = 2.0 * math.pi * rr_c * dr * dz

        # Sample coordinates in 3D (x = r_c, y = 0, z = z_c)
        coords = np.column_stack([rr_c.ravel(), np.zeros(rr_c.size), zz_c.ravel()])
        vols_flat = vols.ravel()

        state = map_inimap2d(params, coords)
        quantities = compute_integrated_quantities(state, vols_flat)

        assert math.isclose(quantities["total_mass"], expected_mass, rel_tol=0.005)
        assert math.isclose(quantities["kinetic_energy"], expected_ekin, rel_tol=0.005)

    def test_formulations(self):
        """Verify formulation enums."""
        assert InimapFormulation.VELOCITY_PRESSURE == 1
        assert InimapFormulation.VELOCITY_ENERGY == 2
