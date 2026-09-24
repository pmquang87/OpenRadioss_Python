"""Unit tests for /LOAD/PBLAST air blast wave pressure loading.

Upstream Fortran references:
- engine/source/loads/pblast/pblast.F
- engine/source/loads/pblast/pblast_1.F
- engine/source/loads/pblast/pblast_2.F
- common_source/modules/loads/pblast_mod.F90
"""

import math
import numpy as np
import pytest

from pyradioss.engine.pblast import (
    BurstType,
    FriedlanderParams,
    PblastParams,
    PblastStepResult,
    PblastEngine,
    interpolate_log_log,
    solve_friedlander_decay,
    pblast_parameters_free_air,
    pblast_parameters_surface_burst,
    compute_segment_geometry,
    compute_effective_pressure,
    compute_segment_blast_pressure,
    apply_pblast_loads,
    _RW3,
    _PSO,
    _PR,
    _ISO,
    _IREFL,
    _T0,
    _TA,
)


class TestUfcTables:
    """Test Kingery-Bulmash / UFC 3-340-02 empirical tables."""

    def test_table_dimensions_and_bounds(self):
        """Tables must have exactly 256 samples spanning Z in [0.5, 400.0] cm/g^(1/3)."""
        assert len(_RW3) == 256
        assert len(_PSO) == 256
        assert len(_PR) == 256
        assert len(_ISO) == 256
        assert len(_IREFL) == 256
        assert len(_T0) == 256
        assert len(_TA) == 256

        assert abs(_RW3[0] - 0.5) < 1e-10
        assert abs(_RW3[-1] - 400.0) < 1e-10

        # Monotonicity checks: pressures and impulses decrease with scaled distance
        assert _PSO[0] > _PSO[10] > _PSO[100] > _PSO[-1]
        assert _PR[0] > _PR[10] > _PR[100] > _PR[-1]
        assert _ISO[0] > _ISO[10] > _ISO[100] > _ISO[-1]
        assert _IREFL[0] > _IREFL[10] > _IREFL[100] > _IREFL[-1]

        # Arrival time increases with distance
        assert _TA[0] < _TA[10] < _TA[100] < _TA[-1]

    def test_log_log_interpolation(self):
        """Interpolation clamps at bounds and interpolates monotonically in-between."""
        # Below lower bound
        v_low = interpolate_log_log(0.1, _RW3, _PSO)
        assert abs(v_low - _PSO[0]) < 1e-12

        # Above upper bound
        v_high = interpolate_log_log(500.0, _RW3, _PSO)
        assert abs(v_high - _PSO[-1]) < 1e-12

        # Interior point
        v_mid = interpolate_log_log(2.0, _RW3, _PSO)
        assert _PSO[-1] < v_mid < _PSO[0]


class TestBlastWavePhysics:
    """Test blast wave scaling and decay formulations."""

    def test_hopkinson_cranz_scaling(self):
        """Distance and charge mass scale according to Hopkinson-Cranz law Z = R / W^(1/3)."""
        # Small charge at 10m vs 8x charge at 20m should yield identical scaled distance Z
        w1 = 1000.0  # 1 kg = 1000 g
        w2 = 8000.0  # 8 kg -> W^(1/3) is 2x
        r1 = 200.0   # 200 cm
        r2 = 400.0   # 400 cm -> 2x distance

        z1 = r1 / (w1 ** (1.0 / 3.0))
        z2 = r2 / (w2 ** (1.0 / 3.0))
        assert abs(z1 - z2) < 1e-10

        fp1 = pblast_parameters_free_air(z=z1, w13=w1 ** (1.0 / 3.0))
        fp2 = pblast_parameters_free_air(z=z2, w13=w2 ** (1.0 / 3.0))

        # Peak pressures are identical at identical scaled distance
        assert abs(fp1.p_inci - fp2.p_inci) < 1e-10
        assert abs(fp1.p_refl - fp2.p_refl) < 1e-10

        # Arrival time and duration scale with W^(1/3) (factor of 2)
        assert abs(fp2.t_a - 2.0 * fp1.t_a) < 1e-8
        assert abs(fp2.dt_0 - 2.0 * fp1.dt_0) < 1e-8

    def test_free_air_vs_surface_burst(self):
        """Surface burst (ground reflection) increases peak pressure and impulse compared to free air."""
        z = 2.0
        w13 = 10.0
        fp_free = pblast_parameters_free_air(z=z, w13=w13)
        fp_surf = pblast_parameters_surface_burst(z=z, w13=w13)

        assert fp_surf.p_inci >= fp_free.p_inci
        assert fp_surf.p_refl >= fp_free.p_refl
        assert fp_surf.i_inci >= fp_free.i_inci

    def test_modified_friedlander_decay_solver(self):
        """Newton-Raphson solver calculates decay parameter b such that integral matches impulse."""
        p_peak = 100.0
        dt_0 = 0.05
        # Target impulse less than triangle area 0.5 * P * dt_0 = 2.5
        target_impulse = 1.25

        b = solve_friedlander_decay(p_peak, dt_0, target_impulse)
        assert b > 0.0

        # Verify numerical integration of Friedlander pulse matches impulse:
        # I = int_0^{dt_0} P * (1 - t/dt_0) * exp(-b * t / dt_0) dt
        k = p_peak * dt_0
        computed_impulse = (k / b) * (1.0 - (1.0 - math.exp(-b)) / b)
        assert abs(computed_impulse - target_impulse) < 1e-6

    def test_effective_pressure_angle_superposition(self):
        """Pressure transitions from reflected Pr at normal incidence to incident Pso at grazing."""
        p_so = 50.0
        p_r = 350.0
        t_a = 0.01
        dt_0 = 0.02
        t = t_a + 0.005  # Mid positive phase

        # 1. Normal incidence: theta = 0 deg (cos_theta = 1.0) -> P = P_refl
        p_normal = compute_effective_pressure(1.0, p_so, p_r, t, t_a, dt_0)
        tau = (t - t_a) / dt_0
        expected_refl = p_r * (1.0 - tau) * math.exp(-1.0 * tau)
        assert abs(p_normal - expected_refl) < 1e-10

        # 2. Grazing incidence: theta = 90 deg (cos_theta = 0.0) -> P = P_inci
        p_grazing = compute_effective_pressure(0.0, p_so, p_r, t, t_a, dt_0)
        expected_inci = p_so * (1.0 - tau) * math.exp(-1.0 * tau)
        assert abs(p_grazing - expected_inci) < 1e-10

        # 3. 45-degree oblique incidence (cos_theta = sqrt(2)/2)
        cos_45 = math.sqrt(2.0) / 2.0
        p_oblique = compute_effective_pressure(cos_45, p_so, p_r, t, t_a, dt_0)
        assert p_grazing < p_oblique < p_normal

        # 4. Shadowed surface facing away from burst (cos_theta < 0)
        p_shadow = compute_effective_pressure(-0.5, p_so, p_r, t, t_a, dt_0)
        assert abs(p_shadow - expected_inci) < 1e-10

    def test_friedlander_waveform_temporal_phases(self):
        """Verify wave arrival, decay, and positive phase termination."""
        p_so = 100.0
        p_r = 400.0
        t_a = 0.05
        dt_0 = 0.10

        # Before arrival
        assert compute_effective_pressure(1.0, p_so, p_r, 0.02, t_a, dt_0) == 0.0

        # Exact arrival time: peak reflected pressure
        p_peak = compute_effective_pressure(1.0, p_so, p_r, t_a, t_a, dt_0)
        assert abs(p_peak - p_r) < 1e-10

        # During decay
        p_mid = compute_effective_pressure(1.0, p_so, p_r, t_a + 0.05, t_a, dt_0)
        assert 0.0 < p_mid < p_r

        # At positive phase duration end
        p_end = compute_effective_pressure(1.0, p_so, p_r, t_a + dt_0, t_a, dt_0)
        assert abs(p_end) < 1e-10

        # After positive phase
        p_after = compute_effective_pressure(1.0, p_so, p_r, t_a + dt_0 + 0.01, t_a, dt_0)
        assert p_after == 0.0

        # With minimum pressure cutoff pmin
        p_cutoff = compute_effective_pressure(1.0, p_so, p_r, 0.0, t_a, dt_0, pmin=5.0)
        assert p_cutoff == 5.0


class TestSegmentGeometryAndForces:
    """Test geometric integration and nodal force vector assembly."""

    def test_triangle_segment_geometry(self):
        """Compute centroid and normal vector N = 2 * Area * n for 3-node triangle."""
        # Right triangle in XY plane: (0,0,0), (2,0,0), (0,3,0) -> Area = 3.0
        tri = np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [0.0, 3.0, 0.0]], dtype=float)
        centroid, normal_2s, area = compute_segment_geometry(tri)

        assert np.allclose(centroid, [2.0 / 3.0, 1.0, 0.0])
        assert abs(area - 3.0) < 1e-10
        # Normal vector N = (x3 - x1) x (x3 - x2) has magnitude 2 * Area = 6.0
        assert np.allclose(normal_2s, [0.0, 0.0, 6.0])

    def test_quad_segment_geometry(self):
        """Compute centroid and normal vector N = 2 * Area * n for 4-node quad."""
        # 2x2 square in XY plane at z=1.0 -> Area = 4.0
        quad = np.array([
            [0.0, 0.0, 1.0],
            [2.0, 0.0, 1.0],
            [2.0, 2.0, 1.0],
            [0.0, 2.0, 1.0],
        ], dtype=float)
        centroid, normal_2s, area = compute_segment_geometry(quad)

        assert np.allclose(centroid, [1.0, 1.0, 1.0])
        assert abs(area - 4.0) < 1e-10
        # N = (x3 - x1) x (x4 - x2) = (2, 2, 0) x (-2, 2, 0) = (0, 0, 8)
        assert np.allclose(normal_2s, [0.0, 0.0, 8.0])

    def test_nodal_force_vector_integration(self):
        """Nodal forces sum to total segment force pushing against outward normal."""
        # Quad of area 4.0 with normal in +Z direction
        # Pressure P = 1000 Pa pushing downwards along -Z
        quad = np.array([
            [0.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [2.0, 2.0, 0.0],
            [0.0, 2.0, 0.0],
        ], dtype=float)

        load = PblastParams(
            id=1,
            xdet=1.0, ydet=1.0, zdet=10.0,  # Detonation directly above quad (+Z)
            tdet=0.0,
            wtnt=100.0,
            segments=[[0, 1, 2, 3]],
        )

        engine = PblastEngine(loads=[load], node_coords=quad)

        # Segment wave parameters at t=0
        res = engine.evaluate_step(t=0.0, dt=0.001)
        # Standoff distance R = 10m, shock hasn't arrived yet
        assert np.allclose(res.nodal_forces, 0.0)

        # Force evaluation at wave arrival time
        item = engine.segment_cache[1][0]
        t_arr = item["t_a"]
        res_arr = engine.evaluate_step(t=t_arr, dt=0.001)

        p = res_arr.segment_pressures[0]
        assert p > 0.0

        # Total force = -P * Area * n = -P * 4.0 * (0, 0, 1) = (0, 0, -4.0 * P)
        total_force = np.sum(res_arr.nodal_forces, axis=0)
        expected_force = np.array([0.0, 0.0, -p * 4.0])
        assert np.allclose(total_force, expected_force, rtol=1e-5)

        # Distributed equally to all 4 nodes
        for i in range(4):
            assert np.allclose(res_arr.nodal_forces[i], expected_force / 4.0, rtol=1e-5)

    def test_external_work_accumulation(self):
        """External work dW = dt * sum(F . v) is accurately integrated."""
        quad = np.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
        ], dtype=float)

        load = PblastParams(
            id=1,
            xdet=0.5, ydet=0.5, zdet=5.0,
            tdet=0.0,
            wtnt=50.0,
            segments=[[0, 1, 2, 3]],
        )

        engine = PblastEngine(loads=[load], node_coords=quad)
        item = engine.segment_cache[1][0]
        res0 = engine.evaluate_step(t=0.0, dt=0.001)
        t_arr = engine.segment_cache[1][0]["t_a"]

        # Velocity in -Z direction (moving in direction of blast force)
        v = np.array([
            [0.0, 0.0, -10.0],
            [0.0, 0.0, -10.0],
            [0.0, 0.0, -10.0],
            [0.0, 0.0, -10.0],
        ], dtype=float)

        dt = 0.001
        res = engine.evaluate_step(t=t_arr, dt=dt, v=v)

        # Force is along -Z, velocity is along -Z -> positive work done on structure
        f_z = res.nodal_forces[:, 2]
        assert np.all(f_z < 0.0)
        expected_dw = dt * np.sum(f_z * -10.0)
        assert expected_dw > 0.0
        assert abs(res.work_step - expected_dw) < 1e-10
        assert abs(engine.cumulative_work - expected_dw) < 1e-10


class TestModelIntegration:
    """Test /LOAD/PBLAST integration with Model entities and dispatcher."""

    def test_apply_pblast_loads_dispatcher(self):
        """Top-level apply_pblast_loads updates fext and tracks external work."""
        class MockSurface:
            def __init__(self, segments):
                self.segments = segments

        class MockModel:
            def __init__(self):
                self.x0 = np.array([
                    [0.0, 0.0, 0.0],
                    [2.0, 0.0, 0.0],
                    [2.0, 2.0, 0.0],
                    [0.0, 2.0, 0.0],
                ], dtype=float)
                self.x = np.copy(self.x0)
                self.surfaces = {10: MockSurface([[0, 1, 2, 3]])}
                self.pblast_loads = {
                    1: PblastParams(
                        id=1,
                        surf_id=10,
                        xdet=1.0, ydet=1.0, zdet=3.0,
                        wtnt=10.0,
                    )
                }

        model = MockModel()
        fext = np.zeros((4, 3), dtype=float)
        v = np.zeros((4, 3), dtype=float)

        # First call before wave arrival
        fext_out, dw = apply_pblast_loads(model, t=0.0, dt=0.001, fext=fext, v=v)
        assert np.allclose(fext_out, 0.0)
        assert dw == 0.0

        # Query arrival time
        engine = model._pblast_engine
        t_arr = engine.segment_cache[1][0]["t_a"]

        # Call at arrival time
        fext_out, dw = apply_pblast_loads(model, t=t_arr, dt=0.001, fext=fext, v=v)
        assert np.sum(np.abs(fext_out)) > 0.0
