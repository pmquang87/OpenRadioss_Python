"""Unit tests for /LOAD/LASER (Moving Laser Heat Flux & Recoil Pressure).

Upstream references:
- engine/source/loads/laser/laser1.F
- engine/source/loads/laser/laser2.F
- starter/source/loads/laser/leclas.F
"""

import math
import numpy as np
import pytest
from scipy import integrate

from pyradioss.engine.laser import (
    LaserBeam,
    LaserEngine,
    LaserLoadParams,
    LaserProfile,
    LaserStepResult,
    SurfaceMesh,
)


class TestLaserProfilesAndIntegration:
    """Test spatial intensity profile formulas and power integration."""

    def test_gaussian_spatial_integration_analytic(self):
        """Verify Gaussian profile integrates to eta * P in 1D radial polar coordinates:
        int_0^inf I(r) * 2*pi*r dr = eta * P.
        """
        power = 1500.0
        absorption = 0.75
        radius = 0.002
        expected_p_abs = absorption * power  # 1125.0 W

        beam = LaserBeam(
            power=power,
            absorption=absorption,
            radius=radius,
            profile=LaserProfile.GAUSSIAN,
        )

        integrand = lambda r: beam.intensity(r) * 2.0 * math.pi * r
        integral_val, _ = integrate.quad(integrand, 0.0, 5.0 * radius)

        assert math.isclose(integral_val, expected_p_abs, rel_tol=1e-4)

    def test_flat_top_spatial_integration_analytic(self):
        """Verify Flat-top profile integrates to eta * P."""
        power = 2000.0
        absorption = 0.6
        radius = 0.0015
        expected_p_abs = absorption * power  # 1200.0 W

        beam = LaserBeam(
            power=power,
            absorption=absorption,
            radius=radius,
            profile=LaserProfile.FLAT_TOP,
        )

        integrand = lambda r: beam.intensity(r) * 2.0 * math.pi * r
        integral_val, _ = integrate.quad(integrand, 0.0, radius)

        assert math.isclose(integral_val, expected_p_abs, rel_tol=1e-4)
        # Intensity outside radius is 0
        assert beam.intensity(radius * 1.01) == 0.0

    def test_conical_spatial_integration_analytic(self):
        """Verify Conical profile integrates to eta * P."""
        power = 800.0
        absorption = 0.9
        radius = 0.003
        expected_p_abs = absorption * power  # 720.0 W

        beam = LaserBeam(
            power=power,
            absorption=absorption,
            radius=radius,
            profile=LaserProfile.CONICAL,
        )

        integrand = lambda r: beam.intensity(r) * 2.0 * math.pi * r
        integral_val, _ = integrate.quad(integrand, 0.0, radius)

        assert math.isclose(integral_val, expected_p_abs, rel_tol=1e-4)
        # Peak intensity at r=0 is 3 * P_abs / (pi * w0^2)
        peak_expected = 3.0 * expected_p_abs / (math.pi * radius**2)
        assert math.isclose(beam.intensity(0.0), peak_expected, rel_tol=1e-6)

    def test_2d_cartesian_grid_integration(self):
        """Verify numerical 2D integration of I(r) over a fine Cartesian grid matches eta * P."""
        power = 1000.0
        absorption = 0.8
        radius = 0.002
        expected_p_abs = absorption * power  # 800.0 W

        w = 4.0 * radius
        n_pts = 401
        xs = np.linspace(-w, w, n_pts)
        ys = np.linspace(-w, w, n_pts)
        dx = xs[1] - xs[0]
        dy = ys[1] - ys[0]
        xx, yy = np.meshgrid(xs, ys)
        rr = np.sqrt(xx**2 + yy**2)

        for profile in [LaserProfile.GAUSSIAN, LaserProfile.FLAT_TOP, LaserProfile.CONICAL]:
            beam = LaserBeam(
                power=power,
                absorption=absorption,
                radius=radius,
                profile=profile,
            )
            i_grid = beam.intensity(rr)
            numerical_p = float(np.sum(i_grid) * dx * dy)
            assert math.isclose(numerical_p, expected_p_abs, rel_tol=0.01), (
                f"Failed for profile {profile}: expected {expected_p_abs}, got {numerical_p}"
            )


class TestSurfaceMesh:
    """Test SurfaceMesh geometry calculations (centers, normals, areas)."""

    def test_triangular_mesh(self):
        # Two triangles forming a 1x1 quad in xy-plane with normal +z
        nodes = np.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
        ])
        segs = [
            (0, 1, 2),
            (0, 2, 3),
        ]
        mesh = SurfaceMesh(nodes, segs)
        centers, normals, areas = mesh.compute_geometry()

        assert mesh.num_nodes == 4
        assert mesh.num_segments == 2
        assert math.isclose(np.sum(areas), 1.0, rel_tol=1e-6)
        for n in normals:
            np.testing.assert_allclose(n, [0.0, 0.0, 1.0], atol=1e-6)

    def test_quad_mesh(self):
        nodes = np.array([
            [0.0, 0.0, 2.0],
            [2.0, 0.0, 2.0],
            [2.0, 3.0, 2.0],
            [0.0, 3.0, 2.0],
        ])
        segs = [(0, 1, 2, 3)]
        mesh = SurfaceMesh(nodes, segs)
        centers, normals, areas = mesh.compute_geometry()

        assert math.isclose(areas[0], 6.0, rel_tol=1e-6)
        np.testing.assert_allclose(centers[0], [1.0, 1.5, 2.0], atol=1e-6)
        np.testing.assert_allclose(normals[0], [0.0, 0.0, 1.0], atol=1e-6)


class TestLaserEngineProjection:
    """Test laser flux projection, angle of incidence, and moving spot trajectory."""

    @pytest.fixture
    def plate_mesh(self):
        """Create a 10x10 mm plate mesh centered at origin in xy-plane (z=0)."""
        nx, ny = 40, 40
        lx, ly = 0.01, 0.01  # 10mm x 10mm
        xs = np.linspace(-lx / 2, lx / 2, nx + 1)
        ys = np.linspace(-ly / 2, ly / 2, ny + 1)

        nodes = []
        for y in ys:
            for x in xs:
                nodes.append([x, y, 0.0])
        nodes = np.array(nodes, dtype=float)

        segs = []
        for j in range(ny):
            for i in range(nx):
                n0 = j * (nx + 1) + i
                n1 = n0 + 1
                n2 = (j + 1) * (nx + 1) + (i + 1)
                n3 = (j + 1) * (nx + 1) + i
                segs.append((n0, n1, n2, n3))

        return SurfaceMesh(nodes, segs)

    def test_laser_flux_projection_and_energy(self, plate_mesh):
        """Verify static beam projecting onto plate captures total absorbed power."""
        power = 1000.0
        absorption = 0.85
        radius = 0.001  # 1mm spot on 10x10mm plate
        expected_p_abs = power * absorption  # 850 W

        beam = LaserBeam(
            power=power,
            absorption=absorption,
            radius=radius,
            profile=LaserProfile.GAUSSIAN,
            origin=np.array([0.0, 0.0, 0.05]),  # 50mm above plate
            direction=np.array([0.0, 0.0, -1.0]),  # pointing down
        )

        engine = LaserEngine([beam], plate_mesh)
        dt = 0.001
        res = engine.evaluate_step(t=0.0, dt=dt)

        assert isinstance(res, LaserStepResult)
        # Power captured on plate should be close to eta * P (beam is well within boundaries)
        assert math.isclose(res.total_absorbed_power, expected_p_abs, rel_tol=0.02)
        # Check nodal heat sum matches total power
        assert math.isclose(np.sum(res.nodal_heat_rates), res.total_absorbed_power, rel_tol=1e-5)
        # Check step energy
        assert math.isclose(res.total_energy_step, res.total_absorbed_power * dt, rel_tol=1e-5)
        assert math.isclose(res.cumulative_energy, res.total_energy_step, rel_tol=1e-5)

    def test_cosine_angle_of_incidence(self):
        """Verify angle of incidence scaling: cos(theta) = max(0, -d . n)."""
        power = 1000.0
        absorption = 1.0
        radius = 0.01

        # Test angles: 0 deg (normal), 30 deg, 45 deg, 60 deg, 90 deg (grazing), 180 deg (back face)
        angles_deg = [0.0, 30.0, 45.0, 60.0, 90.0, 180.0]

        for deg in angles_deg:
            rad = math.radians(deg)
            # Beam propagating downwards (-z)
            d_beam = np.array([0.0, 0.0, -1.0])

            # Normal tilted by deg around y-axis
            # When deg=0, normal is [0, 0, 1] (facing beam)
            # When deg=90, normal is [1, 0, 0] (grazing)
            # When deg=180, normal is [0, 0, -1] (away from beam)
            n_surf = np.array([math.sin(rad), 0.0, math.cos(rad)])

            # Single quad element at origin with normal n_surf
            # Tangent vectors
            t1 = np.array([math.cos(rad), 0.0, -math.sin(rad)])
            t2 = np.array([0.0, 1.0, 0.0])
            size = 0.001
            p0 = -0.5 * size * t1 - 0.5 * size * t2
            p1 = 0.5 * size * t1 - 0.5 * size * t2
            p2 = 0.5 * size * t1 + 0.5 * size * t2
            p3 = -0.5 * size * t1 + 0.5 * size * t2
            nodes = np.array([p0, p1, p2, p3])
            segs = [(0, 1, 2, 3)]
            mesh = SurfaceMesh(nodes, segs)

            beam = LaserBeam(
                power=power,
                absorption=absorption,
                radius=radius,
                profile=LaserProfile.FLAT_TOP,
                origin=np.array([0.0, 0.0, 0.01]),
                direction=d_beam,
            )

            engine = LaserEngine([beam], mesh)
            res = engine.evaluate_step(t=0.0, dt=1e-3)

            expected_cos = max(0.0, math.cos(rad))
            i_peak = power / (math.pi * radius**2)
            expected_flux = i_peak * expected_cos

            assert math.isclose(res.segment_fluxes[0], expected_flux, rel_tol=1e-4, abs_tol=1e-6)

    def test_moving_spot_trajectory(self, plate_mesh):
        """Verify beam moving along X-axis shifts its peak heat flux location accordingly."""
        v_scan = 0.05  # 50 mm/s
        beam = LaserBeam(
            power=1000.0,
            absorption=0.8,
            radius=0.001,
            origin=np.array([-0.003, 0.0, 0.02]),  # Starts at x = -3 mm
            direction=np.array([0.0, 0.0, -1.0]),
            velocity=np.array([v_scan, 0.0, 0.0]),
        )

        engine = LaserEngine([beam], plate_mesh)

        # Time 0: spot at x = -3 mm
        res0 = engine.evaluate_step(t=0.0, dt=0.01)
        max_idx_0 = np.argmax(res0.segment_fluxes)
        centers, _, _ = plate_mesh.compute_geometry()
        center_0 = centers[max_idx_0]
        assert math.isclose(center_0[0], -0.003, abs_tol=0.0005)

        # Time t = 0.06s: spot moved by 0.05 * 0.06 = +0.003 m -> spot at x = 0 mm
        res1 = engine.evaluate_step(t=0.06, dt=0.01)
        max_idx_1 = np.argmax(res1.segment_fluxes)
        center_1 = centers[max_idx_1]
        assert math.isclose(center_1[0], 0.0, abs_tol=0.0005)

        # Time t = 0.12s: spot at x = +3 mm
        res2 = engine.evaluate_step(t=0.12, dt=0.01)
        max_idx_2 = np.argmax(res2.segment_fluxes)
        center_2 = centers[max_idx_2]
        assert math.isclose(center_2[0], +0.003, abs_tol=0.0005)


class TestRecoilPressureAndConservation:
    """Test recoil pressure forces, energy limit, and time stepping."""

    def test_recoil_pressure_and_forces(self):
        """Verify recoil pressure produces normal force pushing into the target."""
        power = 2000.0
        absorption = 1.0
        radius = 0.002
        c_recoil = 5e-6  # Pa / (W/m^2)

        # Plate at z = 0 with normal +z
        nodes = np.array([
            [-0.01, -0.01, 0.0],
            [0.01, -0.01, 0.0],
            [0.01, 0.01, 0.0],
            [-0.01, 0.01, 0.0],
        ])
        mesh = SurfaceMesh(nodes, [(0, 1, 2, 3)])

        beam = LaserBeam(
            power=power,
            absorption=absorption,
            radius=radius,
            profile=LaserProfile.FLAT_TOP,
            origin=[0.0, 0.0, 0.01],
            direction=[0.0, 0.0, -1.0],
            recoil_coeff=c_recoil,
        )

        engine = LaserEngine([beam], mesh)
        res = engine.evaluate_step(t=0.0, dt=1e-3)

        # Segment recoil force should point in -z (opposing surface normal +z)
        f_seg = res.segment_recoil_forces[0]
        assert f_seg[0] == 0.0
        assert f_seg[1] == 0.0
        assert f_seg[2] < 0.0  # pushing down

        # Recoil pressure P = C_r * I
        i_peak = power / (math.pi * radius**2)
        p_recoil = c_recoil * i_peak
        # Total force on the single element
        seg_area = 0.02 * 0.02
        expected_fz = -p_recoil * seg_area
        assert math.isclose(f_seg[2], expected_fz, rel_tol=1e-5)

        # Check nodal recoil force distribution
        np.testing.assert_allclose(np.sum(res.nodal_recoil_forces, axis=0), f_seg, atol=1e-10)

    def test_cumulative_energy_and_cutoff(self):
        """Verify cumulative energy tracking and energy limit shutoff (Fortran laser3.F)."""
        power = 1000.0
        absorption = 1.0

        nodes = np.array([
            [-0.01, -0.01, 0.0],
            [0.01, -0.01, 0.0],
            [0.01, 0.01, 0.0],
            [-0.01, 0.01, 0.0],
        ])
        mesh = SurfaceMesh(nodes, [(0, 1, 2, 3)])

        # Determine step energy first with no cutoff
        probe_beam = LaserBeam(
            power=power,
            absorption=absorption,
            radius=0.01,
            profile=LaserProfile.FLAT_TOP,
            direction=[0.0, 0.0, -1.0],
        )
        probe_engine = LaserEngine([probe_beam], mesh)
        dt = 0.002
        probe_res = probe_engine.evaluate_step(t=0.0, dt=dt)
        step_energy = probe_res.total_energy_step
        assert step_energy > 0.0

        # Set energy limit to 2.5 * step_energy (cutoff triggers on step 4)
        energy_limit = 2.5 * step_energy
        beam = LaserBeam(
            power=power,
            absorption=absorption,
            radius=0.01,
            profile=LaserProfile.FLAT_TOP,
            direction=[0.0, 0.0, -1.0],
            energy_limit=energy_limit,
        )

        engine = LaserEngine([beam], mesh)

        # Step 1: absorbs step_energy -> cum = step_energy
        res1 = engine.evaluate_step(t=0.0, dt=dt)
        assert math.isclose(res1.cumulative_energy, step_energy, rel_tol=1e-5)
        assert res1.total_absorbed_power > 0.0

        # Step 2: absorbs step_energy -> cum = 2 * step_energy
        res2 = engine.evaluate_step(t=0.002, dt=dt)
        assert math.isclose(res2.cumulative_energy, 2.0 * step_energy, rel_tol=1e-5)
        assert res2.total_absorbed_power > 0.0

        # Step 3: absorbs step_energy -> cum = 3 * step_energy (> 2.5 * step_energy)
        res3 = engine.evaluate_step(t=0.004, dt=dt)
        assert math.isclose(res3.cumulative_energy, 3.0 * step_energy, rel_tol=1e-5)

        # Step 4: cutoff is active (cum >= energy_limit)! Power absorbed should be 0.0
        res4 = engine.evaluate_step(t=0.006, dt=dt)
        assert res4.total_absorbed_power == 0.0
        assert math.isclose(res4.cumulative_energy, 3.0 * step_energy, rel_tol=1e-5)

    def test_dynamic_power_and_origin_functions(self):
        """Verify dynamic time-dependent power and trajectory callbacks."""
        beam = LaserBeam(
            power=lambda t: 1000.0 * (1.0 + math.sin(100.0 * t)),
            absorption=0.5,
            radius=0.001,
            origin=lambda t: np.array([0.01 * t, 0.0, 0.0]),
        )

        p0 = beam.get_power(0.0)
        assert math.isclose(p0, 1000.0, rel_tol=1e-6)

        p_abs = beam.get_absorbed_power(0.0)
        assert math.isclose(p_abs, 500.0, rel_tol=1e-6)

        spot = beam.get_spot_center(1.0)
        np.testing.assert_allclose(spot, [0.01, 0.0, 0.0], atol=1e-6)

    def test_laser_load_params_alias(self):
        """Verify LaserLoadParams alias."""
        assert LaserLoadParams is LaserBeam
