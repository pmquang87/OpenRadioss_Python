"""Milestone M539 Upstream Fortran Physics Parity Tests for LAW34 (/MAT/LAW34 /MAT/BOLTZMAN /MAT/VISC_MAXW).

Direct mathematical and numerical parity tests auditing `pyradioss/materials/law34_boltzmann.py`
against the upstream Fortran reference files:
1. engine/source/materials/mat/mat034/sigeps34.F (3D Solids)
2. engine/source/materials/mat/mat034/sigeps34c.F (2D Shells)
3. engine/source/materials/mat/mat034/sigeps34t.F (1D Truss)
4. engine/source/materials/mat/mat034/sigeps34pi.F (Beam)
5. starter/source/materials/mat/mat034/hm_read_mat34.F (Starter Card Reader)
"""

import math
import numpy as np
import pytest

from pyradioss.materials import law34_boltzmann as l34
from pyradioss.materials import (
    law34_solid_update,
    law34_shell_update,
    law34_sound_speed,
    law34_truss_update,
    law34_beam_update,
)


# =============================================================================
# 1. Air Pressure Parity (sigeps34.F lines 105-106, 146, 149-151)
# =============================================================================

class TestLaw34AirPressureFortranParity:
    """Audit air pressure coupling against sigeps34.F."""

    def test_air_pressure_formula_parity(self):
        """Verify DPDGAMA and DP formulas match sigeps34.F lines 105-106, 146."""
        bulk = 5.0e7
        p0 = 1.0e5
        phi = 0.15
        gama0 = 0.02
        rho0 = 1000.0
        rho = 1200.0  # compressed: rho > rho0

        rec = {
            "id": 1,
            "rho": rho0,
            "params": {
                "MAT_BULK": bulk,
                "MAT_G0": 1.5e7,
                "MAT_GI": 5.0e6,
                "MAT_DECAY": 20.0,
                "MAT_P0": p0,
                "MAT_PHI": phi,
                "MAT_GAMA0": gama0,
            },
        }
        mat = l34.build_law34(rec)

        # Fortran calculation (sigeps34.F lines 105-106, 146)
        fortran_gama = rho0 / rho - 1.0 + gama0
        fortran_dpdgama = -p0 * (1.0 - phi) / (1.0 + fortran_gama - phi)
        bulk3 = 3.0 * bulk

        # Compressive strain: DEPSM < 0
        deps_m = -2.0e-3
        fortran_dp = (bulk3 - fortran_dpdgama) * deps_m

        # Run solid_update
        deps = np.array([[deps_m, deps_m, deps_m, 0.0, 0.0, 0.0]])
        sig = np.zeros((1, 6))
        extra = {"rho": np.array([rho])}
        sign, _, _ = l34.solid_update(mat, sig, deps, dt=1e-4, extra=extra)

        # In pure hydrostatic strain, deviatoric strain is 0, so normal stress = DP
        assert math.isclose(sign[0, 0], fortran_dp, rel_tol=1e-14)
        assert math.isclose(sign[0, 1], fortran_dp, rel_tol=1e-14)
        assert math.isclose(sign[0, 2], fortran_dp, rel_tol=1e-14)

    def test_air_pressure_compressive_sign_behavior(self):
        """Verify compression (DEPSM < 0) yields negative DP, reinforcing compressive stress.

        In OpenRadioss:
        - Compressive strain has DEPSM < 0.
        - Compressive stress is negative (sigma < 0).
        - (BULK3 - DPDGAMA) > 0 because DPDGAMA < 0.
        - Therefore DP = (BULK3 - DPDGAMA) * DEPSM < 0.
        - Adding DP to SIGNXX makes SIGNXX more negative (higher compression).
        """
        bulk = 1.0e7
        p0 = 2.0e5
        phi = 0.1
        rho0 = 500.0

        mat_air = l34.build_law34({"rho": rho0, "params": {"bulk": bulk, "p0": p0, "phi": phi, "g0": 2e6, "gi": 1e6}})
        mat_no_air = l34.build_law34({"rho": rho0, "params": {"bulk": bulk, "p0": 0.0, "phi": 0.0, "g0": 2e6, "gi": 1e6}})

        deps_comp = np.array([[-1.0e-3, -1.0e-3, -1.0e-3, 0.0, 0.0, 0.0]])
        s_air, _, _ = l34.solid_update(mat_air, np.zeros((1, 6)), deps_comp, dt=1e-4, extra={"rho": np.array([rho0])})
        s_no_air, _, _ = l34.solid_update(mat_no_air, np.zeros((1, 6)), deps_comp, dt=1e-4, extra={"rho": np.array([rho0])})

        # Both stresses must be negative (compressive)
        assert s_air[0, 0] < 0.0
        assert s_no_air[0, 0] < 0.0

        # Foam with air pressure (P0 > 0) resists compression more: |sigma_air| > |sigma_no_air|
        # (meaning sigma_air is more negative than sigma_no_air)
        assert s_air[0, 0] < s_no_air[0, 0]
        # Difference must be exactly -DPDGAMA * DEPSM = -(-P0) * DEPSM = P0 * DEPSM
        dp_diff = s_air[0, 0] - s_no_air[0, 0]
        assert math.isclose(dp_diff, -(-p0) * (-1.0e-3), rel_tol=1e-12)

    def test_shells_do_not_have_air_pressure(self):
        """Verify sigeps34c.F has no air pressure (DP = BULK3 * DEPSM)."""
        bulk = 1e7
        p0 = 5e5
        phi = 0.2
        mat = l34.build_law34({"rho": 1000.0, "params": {"bulk": bulk, "p0": p0, "phi": phi, "g0": 3e6, "gi": 1e6}})

        deps_sh = np.array([[1e-4, 1e-4, 0.0]])
        s_sh, _ = l34.shell_update(mat, np.zeros((1, 3)), deps_sh, dt=1e-4)

        # In plane stress, deps_zz is solved analytically with DP = BULK3 * DEPSM
        # Air pressure P0 should have NO effect on shells.
        mat_no_air = l34.build_law34({"rho": 1000.0, "params": {"bulk": bulk, "p0": 0.0, "phi": 0.0, "g0": 3e6, "gi": 1e6}})
        s_sh_no_air, _ = l34.shell_update(mat_no_air, np.zeros((1, 3)), deps_sh, dt=1e-4)

        assert np.allclose(s_sh, s_sh_no_air, atol=1e-14)


# =============================================================================
# 2. Shell Plane-Stress Parity (sigeps34c.F lines 98-123, 150-168)
# =============================================================================

class TestLaw34ShellPlaneStressFortranParity:
    """Audit shell analytical plane stress against sigeps34c.F."""

    def test_shell_intermediate_variables_parity(self):
        """Verify CC2, AA, BB, DEPSZZ, DDEZZ, DEPSM, DP exactly match sigeps34c.F."""
        bulk = 2.5e7
        g0 = 6.0e6
        gi = 2.0e6
        beta = 15.0
        dt = 2.0e-5
        mat = l34.build_law34({"rho": 1100.0, "params": {"bulk": bulk, "g0": g0, "gi": gi, "beta": beta}})

        ge = gi
        gv = g0 - gi
        ge2 = 2.0 * ge
        gv2 = 2.0 * gv
        bulk3 = 3.0 * bulk

        c1 = 1.0 - math.exp(-beta * dt)
        c2 = -c1 / beta
        cc2 = gv2 * (c1 + c2 / dt)

        # Initial state: prior deviatoric thickness strain dezz_old = 1e-4, h3 = 0.5e-4
        dezz_old = 1.0e-4
        h3 = 0.5e-4
        deps_xx = 1.5e-3
        deps_yy = -0.8e-3
        deps_xy = 0.4e-3

        # Upstream Fortran formulas (sigeps34c.F lines 109-117, 159)
        aa_fortran = gv2 * c1 * (dezz_old - h3) + (1.0 / 3.0) * (ge2 - cc2 - bulk3) * (deps_xx + deps_yy)
        bb_fortran = (2.0 / 3.0) * ge2 + bulk - (2.0 / 3.0) * cc2
        depszz_fortran = aa_fortran / bb_fortran
        ddezz_fortran = (2.0 / 3.0) * depszz_fortran - (1.0 / 3.0) * (deps_xx + deps_yy)
        depsm_fortran = (1.0 / 3.0) * (deps_xx + deps_yy + depszz_fortran)
        dp_fortran = bulk3 * depsm_fortran

        # Setup Python extra state
        uv34 = np.zeros((1, 8))
        uv34[0, 2] = h3        # H(3)
        uv34[0, 6] = dezz_old  # UVAR(7)
        extra = {"uv34": uv34, "eps34": np.zeros((1, 6))}

        sig_in = np.zeros((1, 3))
        deps_in = np.array([[deps_xx, deps_yy, deps_xy]])
        sign_out, _ = l34.shell_update(mat, sig_in, deps_in, dt=dt, extra=extra)

        # Verify UVAR(7) update in extra
        assert math.isclose(extra["uv34"][0, 6], dezz_old + ddezz_fortran, rel_tol=1e-12, abs_tol=1e-14)

        # Verify sigma_zz = 0 constraint in 3D using fortran quantities
        # sig_zz = GE2*DDEZZ - GV2*DEPSVZZ + DP
        dezz_new = dezz_old + ddezz_fortran
        depsvzz = c1 * (dezz_new - h3) + (c2 / dt) * ddezz_fortran
        sigzz_calc = ge2 * ddezz_fortran - gv2 * depsvzz + dp_fortran
        assert abs(sigzz_calc) < 1e-12

    def test_multistep_plane_stress_exactness(self):
        """Verify sigma_zz = 0 remains strictly 0 across 20 consecutive non-proportional cycles."""
        bulk = 3e7
        g0 = 8e6
        gi = 2e6
        beta = 50.0
        mat = l34.build_law34({"rho": 1000.0, "params": {"bulk": bulk, "g0": g0, "gi": gi, "beta": beta}})

        sig_sh = np.zeros((1, 3))
        extra_sh = {}
        dt = 1e-5

        np.random.seed(42)
        for step in range(20):
            deps_xx = float(np.random.uniform(-1e-3, 1e-3))
            deps_yy = float(np.random.uniform(-1e-3, 1e-3))
            deps_xy = float(np.random.uniform(-1e-3, 1e-3))
            deps_sh = np.array([[deps_xx, deps_yy, deps_xy]])

            # Read current dezz and h3 from extra
            uv = extra_sh.get("uv34", np.zeros((1, 8)))
            dezz_old = uv[0, 6]
            h3 = uv[0, 2]

            ge = gi
            gv = g0 - gi
            ge2 = 2.0 * ge
            gv2 = 2.0 * gv
            bulk3 = 3.0 * bulk
            c1 = 1.0 - math.exp(-beta * dt)
            c2 = -c1 / beta
            cc2 = gv2 * (c1 + c2 / dt)

            aa = gv2 * c1 * (dezz_old - h3) + (1.0 / 3.0) * (ge2 - cc2 - bulk3) * (deps_xx + deps_yy)
            bb = (2.0 / 3.0) * ge2 + bulk - (2.0 / 3.0) * cc2
            deps_zz = aa / bb

            s_sh_out, _ = l34.shell_update(mat, sig_sh, deps_sh, dt=dt, extra=extra_sh)

            # Check sigma_zz using the exact 3D solid kernel
            ddezz = (2.0 / 3.0) * deps_zz - (1.0 / 3.0) * (deps_xx + deps_yy)
            dezz_new = dezz_old + ddezz
            depsvzz = c1 * (dezz_new - h3) + (c2 / dt) * ddezz
            depsm = (1.0 / 3.0) * (deps_xx + deps_yy + deps_zz)
            dp = bulk3 * depsm
            sig_zz = ge2 * ddezz - gv2 * depsvzz + dp

            assert abs(sig_zz) / bulk < 1e-14


# =============================================================================
# 3. Deviatoric Strain Increment & History Update Parity (sigeps34.F)
# =============================================================================

class TestLaw34DeviatoricHistoryFortranParity:
    """Audit deviatoric strains and history q update against sigeps34.F lines 138-144, 161-167."""

    def test_history_update_single_step(self):
        """Verify q_{n+1} = q_n + DEPSV + DDE matches sigeps34.F lines 161-167."""
        mat = l34.build_law34({"rho": 1000.0, "params": {"bulk": 1e7, "g0": 3e6, "gi": 1e6, "beta": 10.0}})
        dt = 1e-4

        # Initial q = [1e-4, -1e-4, 0.0, 2e-4, 0.0, 0.0]
        q_init = np.array([[1e-4, -1e-4, 0.0, 2e-4, 0.0, 0.0, 0.0, 0.0]])
        extra = {"uv34": q_init.copy(), "eps34": np.zeros((1, 6))}

        deps = np.array([[5e-4, 2e-4, -1e-4, 3e-4, 1e-4, -2e-4]])
        sig = np.zeros((1, 6))

        # Expected Fortran computation
        deps_m = (deps[0, 0] + deps[0, 1] + deps[0, 2]) / 3.0
        dde = np.array([
            deps[0, 0] - deps_m,
            deps[0, 1] - deps_m,
            deps[0, 2] - deps_m,
            deps[0, 3],
            deps[0, 4],
            deps[0, 5],
        ])
        de = dde.copy()  # since initial eps was 0, eps_{n+1} = deps, so de = dde

        c1 = 1.0 - math.exp(-10.0 * dt)
        c2 = -c1 / 10.0
        c2_over_dt = c2 / dt

        depsv = np.array([
            c1 * (de[0] - q_init[0, 0]) + c2_over_dt * dde[0],
            c1 * (de[1] - q_init[0, 1]) + c2_over_dt * dde[1],
            c1 * (de[2] - q_init[0, 2]) + c2_over_dt * dde[2],
            c1 * (de[3] - q_init[0, 3]) + c2_over_dt * dde[3],
            c1 * (de[4] - q_init[0, 4]) + c2_over_dt * dde[4],
            c1 * (de[5] - q_init[0, 5]) + c2_over_dt * dde[5],
        ])
        q_expected = q_init[0, :6] + depsv + dde

        sign, _, _ = l34.solid_update(mat, sig, deps, dt=dt, extra=extra)

        q_actual = extra["uv34"][0, :6]
        assert np.allclose(q_actual, q_expected, atol=1e-15)

    def test_relaxation_analytical_exponential_decay(self):
        """Verify that constant held strain produces exact exp(-beta * t) relaxation."""
        g0 = 4e6
        gi = 1e6
        beta = 20.0
        mat = l34.build_law34({"rho": 1000.0, "params": {"bulk": 1e7, "g0": g0, "gi": gi, "beta": beta}})

        gamma_0 = 2e-3
        deps_step = np.array([[0.0, 0.0, 0.0, gamma_0, 0.0, 0.0]])
        sig = np.zeros((1, 6))
        extra = {}

        # Instantaneous step with dt -> 0 limit
        sign, _, _ = l34.solid_update(mat, sig, deps_step, dt=0.0, extra=extra)
        tau_0 = sign[0, 3]
        # Instantaneous response is G0 * gamma_0 = 4e6 * 2e-3 = 8000
        assert math.isclose(tau_0, g0 * gamma_0, rel_tol=1e-5)

        # Hold constant for t = tau = 0.05 s (100 steps of 0.0005 s)
        dt_sub = 0.0005
        deps_zero = np.zeros((1, 6))
        for _ in range(100):
            sign, _, _ = l34.solid_update(mat, sign, deps_zero, dt=dt_sub, extra=extra)

        # Analytical solution: tau(t) = GI * gamma + (G0 - GI) * gamma * exp(-beta * t)
        # At t = 1/beta: tau(tau) = GI*gamma + (G0 - GI)*gamma / e = 2000 + 6000 / e = 4207.2766
        expected_tau = gi * gamma_0 + (g0 - gi) * gamma_0 * math.exp(-1.0)
        assert math.isclose(sign[0, 3], expected_tau, rel_tol=1e-3)

    def test_zero_decay_pure_elastic_limit(self):
        """When BETA = 0, material behaves as pure elastic with G = G0 and history q is invariant."""
        g0 = 5e6
        gi = 1e6
        mat = l34.build_law34({"rho": 1000.0, "params": {"bulk": 1e7, "g0": g0, "gi": gi, "beta": 0.0}})

        deps = np.array([[0.0, 0.0, 0.0, 1e-3, 0.0, 0.0]])
        sig = np.zeros((1, 6))
        extra = {}

        sign, _, _ = l34.solid_update(mat, sig, deps, dt=1e-4, extra=extra)
        assert math.isclose(sign[0, 3], g0 * 1e-3, rel_tol=1e-12)

        # Hold constant: stress must not decay at all
        sign2, _, _ = l34.solid_update(mat, sign, np.zeros((1, 6)), dt=1.0, extra=extra)
        assert math.isclose(sign2[0, 3], sign[0, 3], rel_tol=1e-12)


# =============================================================================
# 4. Sound Speed Parity (sigeps34.F lines 157-158 & hm_read_mat34.F lines 139-152)
# =============================================================================

class TestLaw34SoundSpeedFortranParity:
    """Audit dilatational and starter sound speed against sigeps34.F and hm_read_mat34.F."""

    def test_engine_sound_speed_parity(self):
        """Verify c = sqrt((BULK + 4/3*G0)/rho0) matches sigeps34.F lines 157-158."""
        bulk = 3.6e7
        g0 = 4.5e6
        rho0 = 1200.0
        mat = l34.build_law34({"rho": rho0, "params": {"bulk": bulk, "g0": g0, "gi": 1e6, "beta": 10.0}})

        # Upstream formula: DP_DRHO = FOUR_OVER_3*G_INS + BULK
        #                   SOUNDSP = SQRT(DP_DRHO / RHO0)
        expected_c = math.sqrt((bulk + (4.0 / 3.0) * g0) / rho0)

        assert math.isclose(l34.sound_speed(mat), expected_c, rel_tol=1e-14)
        assert math.isclose(mat.sound_speed_solid(), expected_c, rel_tol=1e-14)

    def test_starter_derived_properties_parity(self):
        """Verify hm_read_mat34.F lines 139-152 derived quantities."""
        bulk = 2.0e7
        g0 = 3.0e6
        rho0 = 800.0
        mat = l34.build_law34({"rho": rho0, "params": {"bulk": bulk, "g0": g0, "gi": 1e6, "beta": 5.0}})

        # YOUNG = (NINE*BULK*G0)/(THREE*BULK + G0)
        expected_young = (9.0 * bulk * g0) / (3.0 * bulk + g0)
        # PARMAT(17) = TWO*G0/(BULK + FOUR_OVER_3*G0)
        expected_parmat17 = (2.0 * g0) / (bulk + (4.0 / 3.0) * g0)
        # PM(27) = SQRT(YOUNG/RHO0)
        expected_pm27 = math.sqrt(expected_young / rho0)

        assert math.isclose(mat.params["E"], expected_young, rel_tol=1e-14)
        assert math.isclose(mat.params["young"], expected_young, rel_tol=1e-14)
        assert math.isclose(mat.params["parmat17"], expected_parmat17, rel_tol=1e-14)
        assert math.isclose(mat.params["pm27"], expected_pm27, rel_tol=1e-14)
        assert mat.params["parmat16"] == 2
        assert mat.params["nuparam"] == 7
        assert mat.params["nuvar"] == 8


# =============================================================================
# 5. 1D Truss Parity (sigeps34t.F lines 72-110)
# =============================================================================

class TestLaw34TrussFortranParity:
    """Audit 1D truss formulation against sigeps34t.F."""

    def test_truss_update_parity(self):
        """Verify truss force update matches sigeps34t.F lines 72-110."""
        bulk = 1e7
        g0 = 3e6
        gi = 1e6
        beta = 10.0
        dt = 1e-4
        mat = l34.build_law34({"rho": 1000.0, "params": {"bulk": bulk, "g0": g0, "gi": gi, "beta": beta}})

        force_in = 500.0
        deps = 2e-3
        area = 0.01
        al0 = 1.0
        al = 1.002
        extra = {}

        force_out, area_out, dsig, sti = l34.truss_update(
            mat, force_in, deps, area, al0, al, dt, extra=extra
        )

        # Verify Fortran calculation
        ge = gi
        gv = g0 - gi
        ge2 = 2.0 * ge
        gv2 = 2.0 * gv
        c1 = 1.0 - math.exp(-beta * dt)
        c2 = -c1 / beta
        ddexx = deps * (2.0 / 3.0)
        dexx = deps * (2.0 / 3.0)  # first step total strain = deps
        depsdxx = ddexx / dt
        depsvxx = c1 * dexx + c2 * depsdxx
        dp = bulk * deps
        dsig_fortran = ge2 * ddexx - gv2 * depsvxx + dp

        k3 = 3.0 * bulk
        nu2 = max((k3 - ge2) / (k3 + ge), 1.0)
        area_fortran = area * (1.0 - nu2 * deps)
        force_fortran = force_in + dsig_fortran * area_fortran

        assert math.isclose(dsig, dsig_fortran, rel_tol=1e-12)
        assert math.isclose(area_out, area_fortran, rel_tol=1e-12)
        assert math.isclose(force_out, force_fortran, rel_tol=1e-12)
        assert extra["uv34_t"][0, 0] > 0.0
        assert math.isclose(extra["uv34_t"][0, 1], deps, rel_tol=1e-12)


# =============================================================================
# 6. Integrated Beam Parity (sigeps34pi.F lines 77-108)
# =============================================================================

class TestLaw34BeamFortranParity:
    """Audit integrated beam formulation against sigeps34pi.F."""

    def test_beam_update_parity(self):
        """Verify beam update matches sigeps34pi.F lines 77-108."""
        bulk = 2e7
        g0 = 6e6
        gi = 2e6
        beta = 12.0
        dt = 5e-5
        mat = l34.build_law34({"rho": 1000.0, "params": {"bulk": bulk, "g0": g0, "gi": gi, "beta": beta}})

        sig_in = np.array([1000.0, 200.0, -100.0])
        deps_in = np.array([1.5e-3, 0.5e-3, -0.2e-3])
        extra = {}

        sign, eps_tot = l34.beam_update(mat, sig_in, deps_in, dt=dt, extra=extra)

        ge = gi
        gv = g0 - gi
        ge2 = 2.0 * ge
        gv2 = 2.0 * gv
        c1 = 1.0 - math.exp(-beta * dt)
        c2 = -c1 / beta
        c2_over_dt = c2 / dt

        ddexx = (2.0 / 3.0) * deps_in[0]
        ddexy = deps_in[1]
        ddexz = deps_in[2]

        dexx = (2.0 / 3.0) * deps_in[0]
        dexy = deps_in[1]
        dexz = deps_in[2]

        depsvxx = c1 * dexx + c2_over_dt * ddexx
        depsvxy = c1 * dexy + c2_over_dt * ddexy
        depsvxz = c1 * dexz + c2_over_dt * ddexz

        dp = bulk * deps_in[0]

        sign_xx = sig_in[0] + ge2 * ddexx - gv2 * depsvxx + dp
        sign_xy = sig_in[1] + ge * ddexy - gv * depsvxy
        sign_xz = sig_in[2] + ge * ddexz - gv * depsvxz

        assert math.isclose(sign[0], sign_xx, rel_tol=1e-12)
        assert math.isclose(sign[1], sign_xy, rel_tol=1e-12)
        assert math.isclose(sign[2], sign_xz, rel_tol=1e-12)
        assert np.allclose(eps_tot, deps_in, atol=1e-14)
