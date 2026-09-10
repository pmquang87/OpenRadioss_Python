"""Unit and physics tests for Milestone M540: /MAT/LAW37, /MAT/BIPHAS, /MAT/BIPHASIC.

Upstream Fortran reference:
- 3D Solids constitutive kernel: engine/source/materials/mat/mat037/sigeps37.F
- Starter Card Reader: starter/source/materials/mat/mat037/hm_read_mat37.F

Covers:
1. build_law37 construction from dict and CFG records, parameter defaults, and density calculation.
2. State initialization for uv37 (history array storing M1/V, rho2, rho1, alpha_v1, alpha_v2).
3. Pure liquid analytical limit (alpha1 = 1.0).
4. Pure gas analytical limit (alpha1 = 0.0).
5. Dual solvers: ISOLVER = 1 (legacy 2-iteration quadratic) and ISOLVER = 2 (2D Newton-Raphson).
6. Wood's mixture formula for sound speed vs legacy liquid sound speed.
7. Newtonian and bulk dynamic viscous damping stresses.
8. Plane-stress shell rejection guard (shell_update raises NotImplementedError).
9. Consistent algorithmic solid tangent symmetry and finite-difference validation.
10. System integration through pyradioss.materials (solid_update, sound_speed, solid_tangent, extra_shapes, needs_env).
"""

from __future__ import annotations

import math
import numpy as np
import pytest

from pyradioss import materials
from pyradioss.materials import law37_biphas
from pyradioss.materials.law37_biphas import (
    build_law37,
    solid_update,
    shell_update,
    sound_speed,
    consistent_solid_tangent,
)
from pyradioss.model.entities import Material


class TestBuildLaw37:
    """Tests for build_law37 material record builder."""

    def test_build_law37_dict_basic(self):
        rec = {
            "id": 1,
            "title": "WATER_AIR",
            "rho_l0": 1000.0,
            "c_l": 2.2e9,
            "alpha1": 0.8,
            "nu_l": 1.0e-3,
            "rho_g0": 1.2,
            "gamma": 1.4,
            "p0": 1.0e5,
            "nu_g": 1.8e-5,
        }
        mat = build_law37(rec)
        assert mat.law == 37
        assert mat.id == 1
        assert mat.title == "WATER_AIR"
        # Initial density rho0 = 1000 * 0.8 + 1.2 * 0.2 = 800.24
        assert math.isclose(mat.rho0, 800.24, rel_tol=1e-5)
        assert math.isclose(mat.params["rho_l0"], 1000.0, rel_tol=1e-5)
        assert math.isclose(mat.params["c_l"], 2.2e9, rel_tol=1e-5)
        assert math.isclose(mat.params["alpha1"], 0.8, rel_tol=1e-5)
        assert math.isclose(mat.params["nu_l"], 1.0e-3, rel_tol=1e-5)
        assert math.isclose(mat.params["rho_g0"], 1.2, rel_tol=1e-5)
        assert math.isclose(mat.params["gamma"], 1.4, rel_tol=1e-5)
        assert math.isclose(mat.params["p0"], 1.0e5, rel_tol=1e-5)
        assert math.isclose(mat.params["nu_g"], 1.8e-5, rel_tol=1e-5)
        assert mat.params["isolver"] == 1
        assert math.isclose(mat.params["pshift"], -1.0e5, rel_tol=1e-5)

    def test_build_law37_cfg_attribute_names(self):
        rec = {
            "id": 2,
            "title": "CFG_ATTRS",
            "Lqud_Rho_l": 998.0,
            "C_l": 2.15e9,
            "ALPHA1": 0.95,
            "Nu_l": 1.1e-3,
            "Bulk_Ratio_l": 2.5e-3,
            "Lqud_Rho_g": 1.18,
            "Lqud_Gamma_bulk": 1.405,
            "Lqud_P0": 101325.0,
            "Nu_g": 1.82e-5,
            "Bulk_Ratio_g": 3.0e-5,
        }
        mat = build_law37(rec)
        assert mat.law == 37
        assert math.isclose(mat.params["rho_l0"], 998.0, rel_tol=1e-5)
        assert math.isclose(mat.params["c_l"], 2.15e9, rel_tol=1e-5)
        assert math.isclose(mat.params["alpha1"], 0.95, rel_tol=1e-5)
        assert math.isclose(mat.params["nu_l"], 1.1e-3, rel_tol=1e-5)
        assert math.isclose(mat.params["nu_vol_l"], 2.5e-3, rel_tol=1e-5)
        assert math.isclose(mat.params["rho_g0"], 1.18, rel_tol=1e-5)
        assert math.isclose(mat.params["gamma"], 1.405, rel_tol=1e-5)
        assert math.isclose(mat.params["p0"], 101325.0, rel_tol=1e-5)
        assert math.isclose(mat.params["nu_g"], 1.82e-5, rel_tol=1e-5)
        assert math.isclose(mat.params["nu_vol_g"], 3.0e-5, rel_tol=1e-5)

    def test_build_law37_pressure_shift_options(self):
        # Default psh=0 -> pshift = -p0 (relative pressure formulation)
        m1 = build_law37({"p0": 1.0e5, "psh": 0.0})
        assert math.isclose(m1.params["pshift"], -1.0e5, rel_tol=1e-5)

        # Non-zero psh -> pshift = -psh
        m2 = build_law37({"p0": 1.0e5, "psh": 5.0e4})
        assert math.isclose(m2.params["pshift"], -5.0e4, rel_tol=1e-5)

    def test_build_law37_solver_selection(self):
        # Default isolver is 1
        m1 = build_law37({})
        assert m1.params["isolver"] == 1

        # Explicit isolver = 2
        m2 = build_law37({"isolver": 2})
        assert m2.params["isolver"] == 2

        # INT22 > 0 activates isolver = 2 (2018.0 solver)
        m3 = build_law37({"INT22": 1})
        assert m3.params["isolver"] == 2


class TestLaw37InitialState:
    """Tests for uv37 state initialization on cycle 1."""

    def test_state_init_first_cycle(self):
        mat = build_law37({
            "rho_l0": 1000.0,
            "c_l": 2.2e9,
            "alpha1": 0.75,
            "rho_g0": 1.2,
            "gamma": 1.4,
            "p0": 1.0e5,
        })
        rho0 = 1000.0 * 0.75 + 1.2 * 0.25
        sig = np.zeros((1, 6))
        deps = np.zeros((1, 6))
        extra = {"rho": rho0}

        sig_out, epsp_out, c = solid_update(mat, sig, deps, dt=1e-5, extra=extra)
        assert "uv37" in extra
        uv37 = extra["uv37"]
        assert uv37.shape == (1, 5)

        # Check volume fractions sum to 1
        alpha_v1 = uv37[0, 3]
        alpha_v2 = uv37[0, 4]
        assert math.isclose(alpha_v1 + alpha_v2, 1.0, rel_tol=1e-6)

        # Check mass conservation: M1/V = alpha_v1 * rho1
        m1_v = uv37[0, 0]
        rho1 = uv37[0, 2]
        assert math.isclose(m1_v, alpha_v1 * rho1, rel_tol=1e-6)


class TestPureLiquidLimit:
    """Tests for pure liquid limit (alpha1 = 1.0)."""

    def test_pure_liquid_hydrostatic_compression(self):
        c_l = 2.2e9
        rho_l0 = 1000.0
        p0 = 1.0e5
        mat = build_law37({
            "rho_l0": rho_l0,
            "c_l": c_l,
            "alpha1": 1.0,
            "rho_g0": 1.2,
            "gamma": 1.4,
            "p0": p0,
            "isolver": 2,
        })
        dt = 1e-5
        # 1% compression: rho = 1.01 * rho_l0 = 1010
        rho = 1010.0
        extra = {"rho": rho}
        sig = np.zeros((1, 6))
        deps = np.zeros((1, 6))

        sig_out, epsp_out, c = solid_update(mat, sig, deps, dt=dt, extra=extra)

        # Linear liquid EOS: P_rel = c_l * (rho / rho_l0 - 1) = 2.2e9 * 0.01 = 2.2e7
        # Cauchy stress sigma = -P_rel * I
        expected_p = c_l * (rho / rho_l0 - 1.0)
        assert math.isclose(sig_out[0, 0], -expected_p, rel_tol=1e-4)
        assert math.isclose(sig_out[0, 1], -expected_p, rel_tol=1e-4)
        assert math.isclose(sig_out[0, 2], -expected_p, rel_tol=1e-4)
        assert math.isclose(sig_out[0, 3], 0.0, abs_tol=1e-10)

        # Sound speed in ISOLVER=2 (sigeps37.F line 314): c = sqrt(C_l / rho_l0)
        expected_c = math.sqrt(c_l / rho_l0)
        assert math.isclose(c[0], expected_c, rel_tol=1e-4)


class TestPureGasLimit:
    """Tests for pure gas limit (alpha1 = 0.0)."""

    def test_pure_gas_isentropic_compression(self):
        rho_g0 = 1.2
        p0 = 1.0e5
        gamma = 1.4
        mat = build_law37({
            "rho_l0": 1000.0,
            "c_l": 2.2e9,
            "alpha1": 0.0,
            "rho_g0": rho_g0,
            "gamma": gamma,
            "p0": p0,
            "isolver": 2,
        })
        dt = 1e-5
        # 5% compression: rho = 1.26
        rho = 1.26
        extra = {"rho": rho}
        sig = np.zeros((1, 6))
        deps = np.zeros((1, 6))

        sig_out, epsp_out, c = solid_update(mat, sig, deps, dt=dt, extra=extra)

        # Isentropic gas EOS: P = P0 * (rho / rho_g0)^gamma
        # Relative pressure with pshift = -p0: P_rel = P0 * (rho / rho_g0)^gamma - P0
        expected_p = p0 * ((rho / rho_g0) ** gamma) - p0
        assert math.isclose(sig_out[0, 0], -expected_p, rel_tol=1e-4)
        assert math.isclose(sig_out[0, 1], -expected_p, rel_tol=1e-4)
        assert math.isclose(sig_out[0, 2], -expected_p, rel_tol=1e-4)

        # Speed of sound for ideal gas: c = sqrt(gamma * P / rho)
        p_tot = p0 * ((rho / rho_g0) ** gamma)
        expected_c = math.sqrt(gamma * p_tot / rho)
        assert math.isclose(c[0], expected_c, rel_tol=1e-4)


class TestBiphasicSolvers:
    """Tests comparing and validating ISOLVER = 1 and ISOLVER = 2 for biphasic mixtures."""

    def test_biphasic_isolver1_legacy(self):
        mat = build_law37({
            "rho_l0": 1000.0,
            "c_l": 2.2e9,
            "alpha1": 0.8,
            "rho_g0": 1.2,
            "gamma": 1.4,
            "p0": 1.0e5,
            "isolver": 1,
        })
        rho0 = 1000.0 * 0.8 + 1.2 * 0.2
        extra = {"rho": rho0}
        sig = np.zeros((1, 6))
        deps = np.zeros((1, 6))

        sig_out, epsp_out, c = solid_update(mat, sig, deps, dt=1e-5, extra=extra)
        # At reference density and zero strain, relative stress should be near 0
        assert math.isclose(sig_out[0, 0], 0.0, abs_tol=1.0)
        # Legacy sound speed c = sqrt(C_l / rho1)
        assert c[0] > 1400.0

    def test_biphasic_isolver2_newton_raphson_equilibrium(self):
        mat = build_law37({
            "rho_l0": 1000.0,
            "c_l": 2.2e9,
            "alpha1": 0.85,
            "rho_g0": 1.2,
            "gamma": 1.4,
            "p0": 1.0e5,
            "isolver": 2,
        })
        rho0 = 1000.0 * 0.85 + 1.2 * 0.15
        # Under slight compression
        rho = rho0 * 1.002
        extra = {"rho": rho}
        sig = np.zeros((1, 6))
        deps = np.zeros((1, 6))

        sig_out, epsp_out, c = solid_update(mat, sig, deps, dt=1e-5, extra=extra)
        uv37 = extra["uv37"]
        rho2 = uv37[0, 1]
        rho1 = uv37[0, 2]
        alpha_v1 = uv37[0, 3]
        alpha_v2 = uv37[0, 4]

        # Check volume consistency: alpha_v1 + alpha_v2 = 1
        assert math.isclose(alpha_v1 + alpha_v2, 1.0, rel_tol=1e-8)

        # Check pressure equilibrium: P1 == P2
        p1 = 2.2e9 * (rho1 / 1000.0 - 1.0) + 1.0e5
        p2 = 1.0e5 * ((rho2 / 1.2) ** 1.4)
        assert math.isclose(p1, p2, rel_tol=1e-4)

    def test_vectorized_multi_element_update(self):
        nel = 8
        mat = build_law37({
            "rho_l0": 1000.0,
            "c_l": 2.2e9,
            "alpha1": 0.7,
            "rho_g0": 1.2,
            "gamma": 1.4,
            "p0": 1.0e5,
            "isolver": 2,
        })
        rho0 = 1000.0 * 0.7 + 1.2 * 0.3
        densities = np.linspace(rho0 * 0.99, rho0 * 1.01, nel)
        extra = {"rho": densities}
        sig = np.zeros((nel, 6))
        deps = np.zeros((nel, 6))

        sig_out, epsp_out, c = solid_update(mat, sig, deps, dt=1e-5, extra=extra)
        assert sig_out.shape == (nel, 6)
        assert c.shape == (nel,)
        assert np.all(c > 0.0)
        # Pressure increases with increasing density (compression)
        pressures = -sig_out[:, 0]
        assert np.all(np.diff(pressures) > 0.0)


class TestViscousDamping:
    """Tests for shear and volumetric viscous damping stresses."""

    def test_shear_viscosity_response(self):
        mat = build_law37({
            "rho_l0": 1000.0,
            "c_l": 2.2e9,
            "alpha1": 1.0,
            "nu_l": 0.005,
            "nu_vol_l": 0.0,
            "rho_g0": 1.2,
            "gamma": 1.4,
            "p0": 1.0e5,
            "isolver": 2,
        })
        dt = 1e-4
        rho = 1000.0
        extra = {"rho": rho}
        sig = np.zeros((1, 6))
        deps = np.zeros((1, 6))
        gamma_xy = 0.02
        deps[0, 3] = gamma_xy

        sig_out, _, _ = solid_update(mat, sig, deps, dt=dt, extra=extra)

        # Dynamic viscosity mu = rho * nu_l = 1000 * 0.005 = 5.0 Pa*s
        # Shear rate dot_gamma = gamma_xy / dt = 0.02 / 1e-4 = 200.0 s^-1
        # Viscous shear stress tau_xy = mu * dot_gamma = 5.0 * 200 = 1000.0 Pa
        expected_tau = 1000.0
        assert math.isclose(sig_out[0, 3], expected_tau, rel_tol=1e-5)

    def test_bulk_viscosity_response(self):
        mat = build_law37({
            "rho_l0": 1000.0,
            "c_l": 2.2e9,
            "alpha1": 1.0,
            "nu_l": 0.002,
            "nu_vol_l": 0.004,
            "rho_g0": 1.2,
            "gamma": 1.4,
            "p0": 1.0e5,
            "isolver": 2,
        })
        dt = 1e-4
        rho = 1000.0
        extra = {"rho": rho}
        sig = np.zeros((1, 6))
        deps = np.zeros((1, 6))
        deps[0, 0] = 0.001  # Normal strain rate 10.0 s^-1

        sig_out, _, _ = solid_update(mat, sig, deps, dt=dt, extra=extra)

        mu = 1000.0 * 0.002
        mu_vol = 1000.0 * 0.004
        deps_rate_xx = 0.001 / 1e-4
        expected_sig_v_xx = (2.0 * mu + mu_vol) * deps_rate_xx
        expected_sig_v_yy = mu_vol * deps_rate_xx

        # Deviatoric and volumetric normal stress verification
        assert math.isclose(sig_out[0, 0], expected_sig_v_xx, rel_tol=1e-5)
        assert math.isclose(sig_out[0, 1], expected_sig_v_yy, rel_tol=1e-5)
        assert math.isclose(sig_out[0, 2], expected_sig_v_yy, rel_tol=1e-5)


class TestSoundSpeed:
    """Tests for sound speed calculation with scalar and vectorized inputs."""

    def test_sound_speed_scalar_and_array(self):
        mat = build_law37({
            "rho_l0": 1000.0,
            "c_l": 2.2e9,
            "alpha1": 1.0,
            "rho_g0": 1.2,
            "gamma": 1.4,
            "p0": 1.0e5,
            "isolver": 1,
        })
        c_scalar = sound_speed(mat, rho=1000.0)
        assert isinstance(c_scalar, float)
        assert math.isclose(c_scalar, math.sqrt(2.2e9 / 1000.0), rel_tol=1e-5)

        c_arr = sound_speed(mat, rho=np.array([1000.0, 1020.0]))
        assert isinstance(c_arr, np.ndarray)
        assert len(c_arr) == 2
        assert math.isclose(c_arr[0], math.sqrt(2.2e9 / 1000.0), rel_tol=1e-5)
        assert math.isclose(c_arr[1], math.sqrt(2.2e9 / 1020.0), rel_tol=1e-5)

    def test_woods_mixture_formula_isolver2(self):
        mat = build_law37({
            "rho_l0": 1000.0,
            "c_l": 2.2e9,
            "alpha1": 0.8,
            "rho_g0": 1.2,
            "gamma": 1.4,
            "p0": 1.0e5,
            "isolver": 2,
        })
        rho0 = 1000.0 * 0.8 + 1.2 * 0.2
        c_mix = sound_speed(mat, rho=rho0)
        # Wood's formula mixture sound speed is significantly lower than water sound speed
        assert 20.0 < c_mix < 500.0


class TestShellRejection:
    """Verify shells are explicitly rejected for LAW37 fluid model."""

    def test_shell_update_raises_not_implemented(self):
        mat = build_law37({})
        sig = np.zeros((1, 3))
        deps = np.zeros((1, 3))
        with pytest.raises(NotImplementedError, match="LAW37.*implemented for 3D solid and SPH"):
            shell_update(mat, sig, deps)

    def test_materials_shell_update_raises_not_implemented(self):
        mat = build_law37({})
        sig = np.zeros((1, 3))
        deps = np.zeros((1, 3))
        with pytest.raises(NotImplementedError, match="LAW37.*implemented for 3D solid and SPH"):
            materials.shell_update(mat, sig, deps, epsp=None, dt=1e-5)


class TestConsistentSolidTangent:
    """Tests for consistent solid tangent calculation and finite difference comparison."""

    def test_tangent_shape_and_symmetry(self):
        mat = build_law37({
            "rho_l0": 1000.0,
            "c_l": 2.2e9,
            "alpha1": 0.8,
            "rho_g0": 1.2,
            "gamma": 1.4,
            "p0": 1.0e5,
            "nu_l": 1.0e-3,
            "nu_vol_l": 2.0e-3,
            "isolver": 2,
        })
        nel = 4
        rho0 = 800.24
        extra = {"rho": np.full(nel, rho0), "dt": 1e-5}
        sig = np.zeros((nel, 6))

        D = consistent_solid_tangent(mat, sig, extra=extra, dt=1e-5)
        assert D.shape == (nel, 6, 6)

        # Symmetry check D_{ijkl} = D_{klij}
        for i in range(nel):
            max_asym = np.max(np.abs(D[i] - D[i].T))
            assert math.isclose(max_asym, 0.0, abs_tol=1e-12)

    def test_tangent_finite_difference_shear(self):
        mat = build_law37({
            "rho_l0": 1000.0,
            "c_l": 2.2e9,
            "alpha1": 0.8,
            "rho_g0": 1.2,
            "gamma": 1.4,
            "p0": 1.0e5,
            "nu_l": 1.5e-3,
            "nu_vol_l": 2.5e-3,
            "nu_g": 1.8e-5,
            "nu_vol_g": 1.0e-5,
            "isolver": 2,
        })
        dt = 1e-5
        rho0 = 800.24
        extra = {"rho": rho0, "dt": dt}
        sig0 = np.zeros((1, 6))
        deps0 = np.zeros((1, 6))

        # Initialize state
        solid_update(mat, sig0.copy(), deps0.copy(), dt=dt, extra=extra)
        D = consistent_solid_tangent(mat, sig0, extra=extra, dt=dt)[0]

        h = 1e-7
        uv37_bak = extra["uv37"].copy()

        # Shear component 3 (xy)
        deps_p = np.zeros((1, 6))
        deps_p[0, 3] = h
        extra["uv37"] = uv37_bak.copy()
        sig_p, _, _ = solid_update(mat, sig0.copy(), deps_p, dt=dt, extra=extra)

        deps_m = np.zeros((1, 6))
        deps_m[0, 3] = -h
        extra["uv37"] = uv37_bak.copy()
        sig_m, _, _ = solid_update(mat, sig0.copy(), deps_m, dt=dt, extra=extra)

        fd_shear = (sig_p[0, 3] - sig_m[0, 3]) / (2.0 * h)
        assert math.isclose(D[3, 3], fd_shear, rel_tol=1e-6)

    def test_tangent_finite_difference_viscous_normal(self):
        mat = build_law37({
            "rho_l0": 1000.0,
            "c_l": 2.2e9,
            "alpha1": 0.7,
            "rho_g0": 1.2,
            "gamma": 1.4,
            "p0": 1.0e5,
            "nu_l": 2.0e-3,
            "nu_vol_l": 3.0e-3,
            "isolver": 2,
        })
        dt = 1e-5
        rho0 = 700.36
        extra = {"rho": rho0, "dt": dt}
        sig0 = np.zeros((1, 6))
        deps0 = np.zeros((1, 6))

        solid_update(mat, sig0.copy(), deps0.copy(), dt=dt, extra=extra)
        D = consistent_solid_tangent(mat, sig0, extra=extra, dt=dt)[0]

        h = 1e-7
        uv37_bak = extra["uv37"].copy()

        # Normal component 0 (xx)
        deps_p = np.zeros((1, 6))
        deps_p[0, 0] = h
        extra["uv37"] = uv37_bak.copy()
        sig_p, _, _ = solid_update(mat, sig0.copy(), deps_p, dt=dt, extra=extra)

        deps_m = np.zeros((1, 6))
        deps_m[0, 0] = -h
        extra["uv37"] = uv37_bak.copy()
        sig_m, _, _ = solid_update(mat, sig0.copy(), deps_m, dt=dt, extra=extra)

        # Difference between diagonal and off-diagonal is 2 * G_t
        fd_diff = ((sig_p[0, 0] - sig_m[0, 0]) - (sig_p[0, 1] - sig_m[0, 1])) / (2.0 * h)
        an_diff = D[0, 0] - D[0, 1]
        assert math.isclose(an_diff, fd_diff, rel_tol=1e-5)


class TestMaterialsModuleIntegration:
    """Tests for dispatching LAW37 through pyradioss.materials module interface."""

    def test_extra_shapes_includes_uv37(self):
        mat = Material(id=1, law=37, rho0=1000.0)
        shapes = materials.extra_shapes(mat)
        assert "uv37" in shapes
        assert shapes["uv37"] == (5,)

        for alias in ("LAW37", "BIPHAS", "BIPHASIC"):
            mat_alias = Material(id=1, law=99, rho0=1000.0, law_name=alias)
            shapes_alias = materials.extra_shapes(mat_alias)
            assert "uv37" in shapes_alias
            assert shapes_alias["uv37"] == (5,)

    def test_needs_env_returns_true(self):
        mat = Material(id=1, law=37, rho0=1000.0)
        assert materials.needs_env(mat) is True

        for alias in ("37", "LAW37", "BIPHAS", "BIPHASIC"):
            mat_alias = Material(id=1, law=alias, rho0=1000.0)
            assert materials.needs_env(mat_alias) is True

    def test_solid_update_dispatch(self):
        mat = build_law37({
            "rho_l0": 1000.0,
            "c_l": 2.2e9,
            "alpha1": 1.0,
            "rho_g0": 1.2,
            "gamma": 1.4,
            "p0": 1.0e5,
        })
        sig = np.zeros((1, 6))
        deps = np.zeros((1, 6))
        extra = {"rho": 1000.0}

        sig_out, epsp_out, c = materials.solid_update(mat, sig, deps, epsp=None, dt=1e-5, extra=extra)
        assert sig_out.shape == (1, 6)
        assert c is not None
        assert math.isclose(c[0], math.sqrt(2.2e9 / 1000.0), rel_tol=1e-5)

    def test_sound_speed_dispatch(self):
        mat = build_law37({
            "rho_l0": 1000.0,
            "c_l": 2.2e9,
            "alpha1": 1.0,
            "rho_g0": 1.2,
            "gamma": 1.4,
            "p0": 1.0e5,
        })
        c = materials.sound_speed(mat, rho=1000.0)
        assert math.isclose(float(c), math.sqrt(2.2e9 / 1000.0), rel_tol=1e-5)

    def test_solid_tangent_dispatch(self):
        mat = build_law37({
            "rho_l0": 1000.0,
            "c_l": 2.2e9,
            "alpha1": 0.8,
            "rho_g0": 1.2,
            "gamma": 1.4,
            "p0": 1.0e5,
            "isolver": 2,
        })
        sig = np.zeros((1, 6))
        extra = {"rho": 800.24, "dt": 1e-5}
        D = materials.solid_tangent(mat, sig, epsp=None, epsp_incr=None, extra=extra)
        assert D.shape == (1, 6, 6)
