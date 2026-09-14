"""
M483 — Unit tests for failure/johnson.py (/FAIL/JOHNSON).

The Johnson-Cook failure criterion accumulates damage from plastic strain
at a rate governed by the stress triaxiality, strain rate and temperature:

    eps_f = (D1 + D2 * exp(D3 * sigma*)) * (1 + D4 * ln(rate/rate0))
                                          * (1 + D5 * T*)
    D += d_eps_p / eps_f        (linear damage accumulation)

Fortran origin: ``engine/source/materials/fail/johnson_cook/fail_johnson.F``
(solids, ISOLID=1/4), ``fail_johnson_c.F`` (shells); reader
``starter/source/materials/fail/johnson_cook/hm_read_fail_johnson.F``.

Functions under test:
* ``_rate_factor``     — 1 + D4*ln(rate/rate0), clamped at 1 below rate0
* ``_thermal_factor``  — 1 + D5*T*, 1 when D5=0 or no temperature
* ``_accumulate``      — damage update with the M40 freeze contract
* ``solid_step``       — full 3-D damage step (Voigt sig, 6 components)
* ``shell_step``       — plane-stress damage step (sig xx, yy, xy)

All tests use synthetic stub failure objects — no model or engine needed.
"""

import numpy as np
import pytest

from pyradioss.failure.johnson import (
    _rate_factor,
    _thermal_factor,
    _accumulate,
    solid_step,
    shell_step,
)


# ======================================================================
# Synthetic failure parameter stub
# ======================================================================

class _Fail:
    """Minimal failure-criterion stub carrying the parameter dict."""
    def __init__(self, D1=0.05, D2=3.44, D3=-2.12, D4=0.002,
                 D5=0.0, eps_dot_0=1.0, eps_f_min=0.0):
        self.params = {
            "D1": D1, "D2": D2, "D3": D3, "D4": D4, "D5": D5,
            "eps_dot_0": eps_dot_0, "eps_f_min": eps_f_min,
        }


# ======================================================================
# _rate_factor — 1 + D4*ln(rate/rate0)
# ======================================================================

class TestRateFactor:
    """_rate_factor: strain-rate sensitivity of the failure strain."""

    def test_d4_zero_returns_one(self):
        """D4=0 → rate factor is 1.0 regardless of strain rate."""
        fail = _Fail(D4=0.0)
        deps = np.array([[0.1, 0.0, 0.0, 0.0, 0.0, 0.0]])
        result = _rate_factor(fail, deps, dt=0.001, dev_from_6=True)
        assert result == 1.0

    def test_below_reference_rate_clamped_at_one(self):
        """Rate below eps_dot_0 is clamped: factor never drops below 1.

        Fortran: fail_johnson.F line 121:
        ``EPSF = EPSF * (ONE + D4*LOG(MAX(ONE, EPSP(I)/EPSP0)))``
        — the MAX(ONE,...) clamp.
        """
        fail = _Fail(D4=0.5, eps_dot_0=1000.0)
        # Very small strain rate (near zero)
        deps = np.array([[1e-10, 0.0, 0.0, 0.0, 0.0, 0.0]])
        result = _rate_factor(fail, deps, dt=1.0, dev_from_6=True)
        np.testing.assert_allclose(result, 1.0, rtol=1e-12)

    def test_above_reference_rate_raises_factor(self):
        """Rate above eps_dot_0 → factor > 1 (for positive D4)."""
        fail = _Fail(D4=0.05, eps_dot_0=1.0)
        # Large deviatoric strain increment → high rate
        deps = np.array([[0.1, -0.05, -0.05, 0.0, 0.0, 0.0]])
        dt = 0.001
        result = _rate_factor(fail, deps, dt=dt, dev_from_6=True)
        # Rate = sqrt(2/3 * dev_invariant) / dt >> eps_dot_0
        assert np.all(result > 1.0)

    def test_plane_stress_deviatoric_rate(self):
        """Plane-stress (shell) rate uses incompressibility for e_zz."""
        fail = _Fail(D4=0.05, eps_dot_0=1.0)
        # Shell: deps has 3 components [e_xx, e_yy, e_xy]
        deps = np.array([[0.1, -0.05, 0.02]])
        dt = 0.001
        result = _rate_factor(fail, deps, dt=dt, dev_from_6=False)
        assert np.all(result > 1.0)

    def test_3d_vs_plane_stress_consistency(self):
        """3D and plane-stress paths give the same rate for matching states.

        For plane stress: e_zz = -(e_xx + e_yy)/2 (incompressibility).
        The 3D path with that e_zz and zero shears should match.
        """
        fail = _Fail(D4=0.1, eps_dot_0=1.0)
        exx, eyy, exy = 0.05, -0.02, 0.01
        ezz = -(exx + eyy) * 0.5

        deps_shell = np.array([[exx, eyy, exy]])
        deps_solid = np.array([[exx, eyy, ezz, exy, 0.0, 0.0]])

        r_shell = _rate_factor(fail, deps_shell, dt=0.001, dev_from_6=False)
        r_solid = _rate_factor(fail, deps_solid, dt=0.001, dev_from_6=True)

        np.testing.assert_allclose(r_shell, r_solid, rtol=1e-10)

    def test_eps_dot_0_zero_or_missing_guard(self):
        """eps_dot_0 = 0.0 or missing is guarded against ZeroDivisionError."""
        fail_zero = _Fail(D4=0.05, eps_dot_0=0.0)
        deps = np.array([[0.1, -0.05, -0.05, 0.0, 0.0, 0.0]])
        res_zero = _rate_factor(fail_zero, deps, dt=0.001, dev_from_6=True)
        assert np.all(res_zero > 1.0)

        fail_missing = _Fail(D4=0.05)
        del fail_missing.params["eps_dot_0"]
        res_missing = _rate_factor(fail_missing, deps, dt=0.001, dev_from_6=True)
        assert np.all(res_missing > 1.0)


# ======================================================================
# _thermal_factor — 1 + D5*T*
# ======================================================================

class TestThermalFactor:
    """_thermal_factor: temperature dependence of failure strain."""

    def test_d5_zero_returns_one(self):
        """D5=0 → thermal factor is 1.0."""
        fail = _Fail(D5=0.0)
        assert _thermal_factor(fail, tstar=np.array([0.5])) == 1.0

    def test_no_tstar_returns_one(self):
        """No temperature state (tstar=None) → factor is 1.0."""
        fail = _Fail(D5=0.5)
        assert _thermal_factor(fail, tstar=None) == 1.0

    def test_positive_d5_raises_factor(self):
        """D5 > 0, T* > 0 → factor > 1 (hot metal is more ductile).

        Johnson & Cook 1985 convention: positive D5 increases the
        failure strain with temperature.
        """
        fail = _Fail(D5=0.5)
        tstar = np.array([0.4, 0.8])
        result = _thermal_factor(fail, tstar)
        # 1 + 0.5 * 0.4 = 1.2;  1 + 0.5 * 0.8 = 1.4
        np.testing.assert_allclose(result, [1.2, 1.4], rtol=1e-12)

    def test_d5_not_in_params_returns_one(self):
        """D5 missing from params dict → default 0 → factor is 1.0."""
        fail = _Fail()
        fail.params.pop("D5", None)
        assert _thermal_factor(fail, tstar=np.array([0.5])) == 1.0


# ======================================================================
# _accumulate — the M40 damage-freeze contract
# ======================================================================

class TestAccumulate:
    """_accumulate: damage growth with the freeze-on-negative-eps_f rule."""

    def test_positive_eps_f_grows_damage(self):
        """Positive eps_f and positive d_eps_p → damage grows."""
        fail = _Fail()
        dama = np.array([0.0])
        d_epsp = np.array([0.01])
        eps_f = np.array([0.1])

        _accumulate(fail, dama, d_epsp, eps_f)
        assert dama[0] == pytest.approx(0.01 / 0.1, rel=1e-12)

    def test_negative_eps_f_freezes_damage(self):
        """Negative eps_f → damage is FROZEN (the M40 contract).

        Fortran: fail_johnson.F lines 123-124:
        ``EPSF = MAX(EPSF, EPSF_MIN)``
        ``IF (EPSF > ZERO) DFMAX = DFMAX + DPLA/EPSF``
        — eps_f <= 0 skips the accumulation entirely.
        """
        fail = _Fail(eps_f_min=0.0)
        dama = np.array([0.3])     # pre-existing damage
        d_epsp = np.array([0.05])  # plastic increment
        eps_f = np.array([-0.5])   # negative failure strain

        _accumulate(fail, dama, d_epsp, eps_f)
        assert dama[0] == pytest.approx(0.3, rel=1e-12)  # unchanged

    def test_zero_eps_f_freezes_damage(self):
        """eps_f = 0 also freezes damage (not division by zero)."""
        fail = _Fail(eps_f_min=0.0)
        dama = np.array([0.5])
        d_epsp = np.array([0.1])
        eps_f = np.array([0.0])

        _accumulate(fail, dama, d_epsp, eps_f)
        assert dama[0] == pytest.approx(0.5, rel=1e-12)

    def test_damage_capped_at_one(self):
        """Damage is capped at 1.0.

        Fortran: fail_johnson.F line 125:
        ``DFMAX = MIN(ONE, DFMAX)``
        """
        fail = _Fail()
        dama = np.array([0.95])
        d_epsp = np.array([1.0])    # huge increment
        eps_f = np.array([0.01])    # small failure strain

        _accumulate(fail, dama, d_epsp, eps_f)
        assert dama[0] == pytest.approx(1.0, rel=1e-12)

    def test_zero_d_epsp_no_growth(self):
        """No plastic flow (d_eps_p=0) → damage stays frozen."""
        fail = _Fail()
        dama = np.array([0.4])
        d_epsp = np.array([0.0])
        eps_f = np.array([0.1])

        _accumulate(fail, dama, d_epsp, eps_f)
        assert dama[0] == pytest.approx(0.4, rel=1e-12)

    def test_negative_d_epsp_no_growth(self):
        """Negative plastic strain increment → treated as zero (no reverse)."""
        fail = _Fail()
        dama = np.array([0.4])
        d_epsp = np.array([-0.01])
        eps_f = np.array([0.1])

        _accumulate(fail, dama, d_epsp, eps_f)
        assert dama[0] == pytest.approx(0.4, rel=1e-12)

    def test_eps_f_min_floor(self):
        """eps_f_min > 0 floors the failure strain before accumulation."""
        fail = _Fail(eps_f_min=0.05)
        dama = np.array([0.0])
        d_epsp = np.array([0.01])
        eps_f = np.array([0.001])   # below floor

        _accumulate(fail, dama, d_epsp, eps_f)
        # Floored to 0.05 → 0.01/0.05 = 0.2
        assert dama[0] == pytest.approx(0.01 / 0.05, rel=1e-12)

    def test_in_place_on_view(self):
        """_accumulate works in-place on a slice view (not a copy)."""
        fail = _Fail()
        full = np.array([0.0, 0.1, 0.2, 0.3])
        dama_view = full[1:3]   # a view of elements 1,2
        d_epsp = np.array([0.02, 0.03])
        eps_f = np.array([0.1, 0.15])

        _accumulate(fail, dama_view, d_epsp, eps_f)

        # full[1] = 0.1 + 0.02/0.1 = 0.3
        # full[2] = 0.2 + 0.03/0.15 = 0.4
        assert full[1] == pytest.approx(0.3, rel=1e-12)
        assert full[2] == pytest.approx(0.4, rel=1e-12)
        # full[0] and full[3] unchanged
        assert full[0] == pytest.approx(0.0)
        assert full[3] == pytest.approx(0.3)

    def test_multi_point_mixed(self):
        """Multiple points: some grow, some frozen, some capped."""
        fail = _Fail(eps_f_min=0.0)
        dama = np.array([0.0, 0.5, 0.99])
        d_epsp = np.array([0.01, 0.01, 0.5])
        eps_f = np.array([0.1, -0.3, 0.01])

        _accumulate(fail, dama, d_epsp, eps_f)

        # Point 0: 0.0 + 0.01/0.1 = 0.1
        assert dama[0] == pytest.approx(0.1, rel=1e-12)
        # Point 1: frozen (eps_f < 0) → stays 0.5
        assert dama[1] == pytest.approx(0.5, rel=1e-12)
        # Point 2: 0.99 + 0.5/0.01 = 50.99 → capped at 1.0
        assert dama[2] == pytest.approx(1.0, rel=1e-12)


# ======================================================================
# Triaxiality — solid (3D)
# ======================================================================

class TestTriaxialitySolid:
    """solid_step triaxiality: sigma_m / sigma_vm under known stress states."""

    def _run_solid(self, sig6, D1=1.0, D2=0.0, D3=0.0, D4=0.0):
        """Run solid_step and return the damage (isolates triaxiality effect).

        With D2=0 and D3=0: eps_f = D1 (constant), so the damage increment
        is d_eps_p / D1 — no triaxiality dependence, useful for other tests.
        """
        fail = _Fail(D1=D1, D2=D2, D3=D3, D4=D4)
        sig = np.atleast_2d(sig6).astype(float)
        d_epsp = np.array([0.01] * sig.shape[0])
        deps = np.zeros((sig.shape[0], 6))
        dama = np.zeros(sig.shape[0])
        solid_step(fail, sig, d_epsp, deps, dt=1.0, dama=dama)
        return dama

    def test_pure_hydrostatic_infinite_triaxiality(self):
        """Pure hydrostatic: sigma_vm ≈ 0 → sigma* → ∞.

        With D3 < 0 and high triaxiality, eps_f grows exponentially
        (D1 + D2*exp(D3*sigma*) → D1 for D3 < 0, sigma* >> 1).
        """
        # sig = [p, p, p, 0, 0, 0] (pure pressure, no deviatoric)
        p = 100.0
        sig = np.array([[p, p, p, 0.0, 0.0, 0.0]])
        fail = _Fail(D1=0.05, D2=3.0, D3=-2.0, D4=0.0)
        d_epsp = np.array([0.01])
        deps = np.zeros((1, 6))
        dama = np.zeros(1)
        solid_step(fail, sig, d_epsp, deps, dt=1.0, dama=dama)
        # eps_f = D1 + D2*exp(D3 * sigma*) ≈ D1 (D3 < 0, sigma* large)
        # Damage = 0.01 / D1 = 0.2
        assert dama[0] == pytest.approx(0.01 / 0.05, rel=0.05)

    def test_pure_shear_triaxiality_zero(self):
        """Pure shear: sigma_m = 0 → sigma* = 0.

        eps_f = D1 + D2*exp(0) = D1 + D2.
        """
        # sig = [0, 0, 0, tau, 0, 0]
        tau = 100.0
        sig = np.array([[0.0, 0.0, 0.0, tau, 0.0, 0.0]])
        fail = _Fail(D1=0.05, D2=3.0, D3=-2.0, D4=0.0)
        d_epsp = np.array([0.01])
        deps = np.zeros((1, 6))
        dama = np.zeros(1)
        solid_step(fail, sig, d_epsp, deps, dt=1.0, dama=dama)
        # eps_f = 0.05 + 3.0*exp(0) = 3.05
        expected = 0.01 / 3.05
        assert dama[0] == pytest.approx(expected, rel=1e-10)

    def test_uniaxial_tension_triaxiality_one_third(self):
        """Uniaxial tension: sigma* = 1/3.

        sig = [sigma, 0, 0, 0, 0, 0] → sm = sigma/3, vm = sigma.
        """
        sigma = 300.0
        sig = np.array([[sigma, 0.0, 0.0, 0.0, 0.0, 0.0]])
        fail = _Fail(D1=0.05, D2=3.0, D3=-2.0, D4=0.0)
        d_epsp = np.array([0.01])
        deps = np.zeros((1, 6))
        dama = np.zeros(1)
        solid_step(fail, sig, d_epsp, deps, dt=1.0, dama=dama)
        # triax = 1/3
        eps_f = 0.05 + 3.0 * np.exp(-2.0 / 3.0)
        expected = 0.01 / eps_f
        assert dama[0] == pytest.approx(expected, rel=1e-10)


# ======================================================================
# Triaxiality — shell (plane stress)
# ======================================================================

class TestTriaxialityShell:
    """shell_step: plane-stress triaxiality and damage."""

    def test_uniaxial_tension_shell(self):
        """Uniaxial tension in a shell: sig = [sigma, 0, 0] → sigma* = 1/3.

        Same triaxiality as 3D uniaxial (sigma_zz = 0 is implicit).
        """
        sigma = 300.0
        sig = np.array([[sigma, 0.0, 0.0]])
        fail = _Fail(D1=0.05, D2=3.0, D3=-2.0, D4=0.0)
        d_epsp = np.array([0.01])
        deps = np.zeros((1, 3))
        dama = np.zeros(1)
        shell_step(fail, sig, d_epsp, deps, dt=1.0, dama=dama)
        # sm = sigma/3, vm = sigma (plane stress, sigma_zz=0)
        eps_f = 0.05 + 3.0 * np.exp(-2.0 / 3.0)
        expected = 0.01 / eps_f
        assert dama[0] == pytest.approx(expected, rel=1e-10)

    def test_pure_shear_shell(self):
        """Pure shear in a shell: sig = [0, 0, tau] → sigma* = 0."""
        tau = 150.0
        sig = np.array([[0.0, 0.0, tau]])
        fail = _Fail(D1=0.05, D2=3.0, D3=-2.0, D4=0.0)
        d_epsp = np.array([0.01])
        deps = np.zeros((1, 3))
        dama = np.zeros(1)
        shell_step(fail, sig, d_epsp, deps, dt=1.0, dama=dama)
        eps_f = 0.05 + 3.0 * np.exp(0.0)   # D1 + D2
        expected = 0.01 / eps_f
        assert dama[0] == pytest.approx(expected, rel=1e-10)

    def test_biaxial_tension_shell(self):
        """Equal biaxial tension: sig = [s, s, 0] → sigma* = 2/3.

        sm = 2s/3, vm = s (from formula: s^2 - s*s + s^2 = s^2).
        """
        s = 200.0
        sig = np.array([[s, s, 0.0]])
        fail = _Fail(D1=0.05, D2=3.0, D3=-2.0, D4=0.0)
        d_epsp = np.array([0.01])
        deps = np.zeros((1, 3))
        dama = np.zeros(1)
        shell_step(fail, sig, d_epsp, deps, dt=1.0, dama=dama)
        # sm = (s+s)/3 = 2s/3, vm = sqrt(s^2 - s^2 + s^2) = s
        triax = 2.0 / 3.0
        eps_f = 0.05 + 3.0 * np.exp(-2.0 * triax)
        expected = 0.01 / eps_f
        assert dama[0] == pytest.approx(expected, rel=1e-10)


# ======================================================================
# solid_step / shell_step — full integration
# ======================================================================

class TestSolidStep:
    """solid_step: full 3-D damage step with all factors."""

    def test_returns_broken_mask(self):
        """solid_step returns True where D >= 1.0."""
        fail = _Fail(D1=0.01, D2=0.0, D3=0.0, D4=0.0)
        sig = np.array([[100., 0., 0., 0., 0., 0.],
                         [100., 0., 0., 0., 0., 0.]])
        d_epsp = np.array([0.005, 0.02])   # below and above
        deps = np.zeros((2, 6))
        dama = np.zeros(2)

        broken = solid_step(fail, sig, d_epsp, deps, dt=1.0, dama=dama)

        assert broken[0] == False   # 0.005/0.01 = 0.5 < 1
        assert broken[1] == True    # 0.02/0.01 = 2.0 → capped at 1

    def test_thermal_factor_integration(self):
        """solid_step uses the thermal factor when tstar is provided."""
        fail = _Fail(D1=0.1, D2=0.0, D3=0.0, D4=0.0, D5=1.0)
        sig = np.array([[100., 0., 0., 0., 0., 0.]])
        d_epsp = np.array([0.01])
        deps = np.zeros((1, 6))
        tstar = np.array([0.5])

        dama = np.zeros(1)
        solid_step(fail, sig, d_epsp, deps, dt=1.0, dama=dama, tstar=tstar)

        # eps_f = D1 * (1 + D5*T*) = 0.1 * (1 + 0.5) = 0.15
        expected = 0.01 / 0.15
        assert dama[0] == pytest.approx(expected, rel=1e-10)

    def test_triaxiality_exponent_clamp(self):
        """Extreme triaxiality exponent is clamped to prevent exp overflow in solid_step."""
        fail = _Fail(D1=0.1, D2=0.2, D3=10.0, D4=0.0)
        # Very high hydrostatic tension -> large positive triax
        sig = np.array([[1e6, 1e6, 1e6, 0.0, 0.0, 0.0]])
        d_epsp = np.array([0.01])
        deps = np.zeros((1, 6))
        dama = np.zeros(1)
        solid_step(fail, sig, d_epsp, deps, dt=1.0, dama=dama)
        assert np.isfinite(dama[0])
        assert dama[0] >= 0.0


class TestShellStep:
    """shell_step: plane-stress damage step."""

    def test_returns_broken_mask(self):
        """shell_step returns True where D >= 1.0."""
        fail = _Fail(D1=0.01, D2=0.0, D3=0.0, D4=0.0)
        sig = np.array([[100., 0., 0.],
                         [100., 0., 0.]])
        d_epsp = np.array([0.005, 0.02])
        deps = np.zeros((2, 3))
        dama = np.zeros(2)

        broken = shell_step(fail, sig, d_epsp, deps, dt=1.0, dama=dama)

        assert broken[0] == False
        assert broken[1] == True

    def test_thermal_factor_integration(self):
        """shell_step uses the thermal factor when tstar is provided."""
        fail = _Fail(D1=0.1, D2=0.0, D3=0.0, D4=0.0, D5=1.0)
        sig = np.array([[100., 0., 0.]])
        d_epsp = np.array([0.01])
        deps = np.zeros((1, 3))
        tstar = np.array([0.5])

        dama = np.zeros(1)
        shell_step(fail, sig, d_epsp, deps, dt=1.0, dama=dama, tstar=tstar)

        expected = 0.01 / 0.15
        assert dama[0] == pytest.approx(expected, rel=1e-10)

    def test_triaxiality_exponent_clamp(self):
        """Extreme triaxiality exponent is clamped to prevent exp overflow in shell_step."""
        fail = _Fail(D1=0.1, D2=0.2, D3=10.0, D4=0.0)
        # High biaxial tension
        sig = np.array([[1e6, 1e6, 0.0]])
        d_epsp = np.array([0.01])
        deps = np.zeros((1, 3))
        dama = np.zeros(1)
        shell_step(fail, sig, d_epsp, deps, dt=1.0, dama=dama)
        assert np.isfinite(dama[0])
        assert dama[0] >= 0.0
