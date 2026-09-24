r"""
Unit tests for LAW79 (/MAT/LAW79, /MAT/JOHN_HOLM) Johnson-Holmquist (JH-2) brittle model.

Upstream Fortran reference:
- C:\OpenRadioss\source\OpenRadioss-latest-20260520\engine\source\materials\mat\mat079\sigeps79.F
- C:\OpenRadioss\source\OpenRadioss-latest-20260520\starter\source\materials\mat\mat079\hm_read_mat79.F

Verifies:
1. Intact strength:
   - sigma_i = A * (P* + T*)^N (normalized by sigma_HEL)
   - Zero strength when P* + T* <= 0
2. Damaged strength:
   - sigma_d = B * (P*)^M * (1 + C * ln(eps_dot / eps0))
   - Cap at sigma_f_max
   - Zero fractured strength under tension (P* <= 0)
3. Damage evolution:
   - eps_p_fail = D1 * (P* + T*)^D2
   - Delta eps_p = (1 - SCALE) * sigma_vm / (3 * sqrt(3) * G)
   - Damage accumulation D += Delta eps_p / eps_p_fail
   - Transition from intact (D=0) to fully damaged (D=1)
4. Equations of State (EOS) coupling and bulking pressure:
   - Nonlinear polynomial compression: P = K1 * mu + K2 * mu^2 + K3 * mu^3 + Delta P
   - Shear strain energy loss conversion to bulking pressure: Delta U -> Delta P
5. Failure and element deletion modes (IDEL = 1, 2, 3):
   - IDEL = 1: Hydrostatic tensile rupture (P* + T* < 0)
   - IDEL = 2: Maximum plastic strain rupture (eps_p > eps_max)
   - IDEL = 3: Complete damage rupture (D = 1.0)
6. Wave speed and consistent tangent tensor.
"""

from __future__ import annotations

import math
import numpy as np
import pytest

from pyradioss.materials.law79_john_holm import (
    Law79Params,
    build_law79,
    solid_update,
    solid_tangent,
    tangent,
    consistent_solid_tangent,
    sound_speed_solid,
)
from pyradioss.model.entities import Material


def _make_sic_law79(
    a=0.96, b=0.35, m=1.0, n=0.65, c=0.009, eps0=1.0, sigfmax=0.8,
    t=0.37, hel=14.5, phel=5.13, d1=0.48, d2=0.48, idel=0, epsmax=1e20,
    k1=220.0, k2=0.0, k3=0.0, beta=1.0, shear=193.0, rho0=3.21e-3
) -> Material:
    rec = {
        "id": 79,
        "title": "SiC_JH2",
        "rho": rho0,
        "shear": shear,
        "a": a,
        "b": b,
        "m": m,
        "n": n,
        "c": c,
        "eps0": eps0,
        "sigfmax": sigfmax,
        "t": t,
        "hel": hel,
        "phel": phel,
        "d1": d1,
        "d2": d2,
        "idel": idel,
        "epsmax": epsmax,
        "k1": k1,
        "k2": k2,
        "k3": k3,
        "beta": beta,
    }
    return build_law79(rec)


class TestLaw79IntactStrength:
    """Verify intact strength envelope sigma_i* = A * (P* + T*)^N."""

    def test_intact_strength_pressure_scaling(self):
        mat = _make_sic_law79(a=0.96, n=0.65, c=0.0)
        phel = mat.params["phel"]
        shel = mat.params["shel"]
        tstar = mat.params["tstar"]

        # Impose pressure P = 2.0 * PHEL => P* = 2.0
        p_val = 2.0 * phel
        k1 = mat.params["k1"]
        mu = p_val / k1  # linear regime

        extra = {"mu": mu, "dmg": 0.0}
        # Shear strain large enough to exceed yield: vm_trial = sqrt(3)*193*0.1 = 33.42 > 21.66
        gamma = 0.1
        deps = np.array([0.0, 0.0, 0.0, gamma, 0.0, 0.0])
        sig = np.zeros(6)

        sig_out, epsp_out, c_out = solid_update(mat, sig, deps, dt=1.0, extra=extra, return_tuple=True)

        # Expected normalized intact yield stress: sigma_i* = A * (P* + T*)^N
        pstar = p_val / phel
        sig_i_star = 0.96 * ((pstar + tstar) ** 0.65)
        expected_yield = sig_i_star * shel

        # Equivalent von Mises stress must match yield_actual = sig_i_star * shel
        sxy = sig_out[3]
        vm = math.sqrt(3.0) * abs(sxy)
        assert math.isclose(vm, expected_yield, rel_tol=1e-5)
        assert epsp_out > 0.0

    def test_intact_strength_zero_in_tension(self):
        """When P* + T* <= 0, intact yield strength is zero (sigeps79.F line 165)."""
        mat = _make_sic_law79(a=1.0, n=1.0, t=0.2, phel=2.0, k1=100.0, idel=0)
        # T* = 0.2 / 2.0 = 0.1
        # Set negative pressure P = -0.3 => P* = -0.15 => P* + T* = -0.05 <= 0
        mu = -0.3 / 100.0
        extra = {"mu": mu, "dmg": 0.0}
        deps = np.array([0.0, 0.0, 0.0, 0.01, 0.0, 0.0])

        sig_out, _, _ = solid_update(mat, np.zeros(6), deps, extra=extra, return_tuple=True)
        # Yield stress is 0, so shear stress must scale to 0
        assert math.isclose(sig_out[3], 0.0, abs_tol=1e-12)


class TestLaw79DamagedStrength:
    """Verify fractured strength envelope sigma_f* = B * (P*)^M * (1 + C * ln(eps_dot/eps0))."""

    def test_fully_damaged_strength_and_cap(self):
        mat = _make_sic_law79(b=0.35, m=1.0, c=0.0, sigfmax=0.5)
        phel = mat.params["phel"]
        shel = mat.params["shel"]

        # Case 1: P* = 1.0 => sigma_f* = 0.35 * 1.0 = 0.35 < sigfmax (0.5)
        p_val = 1.0 * phel
        mu1 = p_val / mat.params["k1"]
        extra1 = {"mu": mu1, "dmg": 1.0}  # fully damaged D=1
        deps = np.array([0.0, 0.0, 0.0, 0.05, 0.0, 0.0])
        sig_out1 = solid_update(mat, np.zeros(6), deps, extra=extra1)
        vm1 = math.sqrt(3.0) * abs(sig_out1[3])
        expected_vm1 = 0.35 * shel
        assert math.isclose(vm1, expected_vm1, rel_tol=1e-5)

        # Case 2: High pressure P* = 3.0 => 0.35 * 3.0 = 1.05 > sigfmax (0.5) -> capped at 0.5
        p_val2 = 3.0 * phel
        mu2 = p_val2 / mat.params["k1"]
        extra2 = {"mu": mu2, "dmg": 1.0}
        sig_out2 = solid_update(mat, np.zeros(6), deps, extra=extra2)
        vm2 = math.sqrt(3.0) * abs(sig_out2[3])
        expected_vm2 = 0.5 * shel
        assert math.isclose(vm2, expected_vm2, rel_tol=1e-5)

    def test_strain_rate_enhancement(self):
        """Verify CE = 1 + C * ln(eps_dot / eps0)."""
        c_rate = 0.02
        eps0 = 1.0
        mat = _make_sic_law79(a=1.0, n=0.0, c=c_rate, eps0=eps0)
        shel = mat.params["shel"]

        eps_dot = 1000.0  # 1000 s^-1
        dt = 1e-4
        # gamma large enough to yield: gamma = 0.1 => vm_trial = 33.42 > expected_yield (15.99)
        gamma = 0.1
        deps = np.array([0.0, 0.0, 0.0, gamma, 0.0, 0.0])

        extra = {"mu": 0.0, "dmg": 0.0, "epsd": eps_dot}
        sig_out = solid_update(mat, np.zeros(6), deps, dt=dt, extra=extra)

        expected_ce = 1.0 + c_rate * math.log(eps_dot / eps0)
        expected_yield = expected_ce * 1.0 * shel
        actual_vm = math.sqrt(3.0) * abs(sig_out[3])
        assert math.isclose(actual_vm, expected_yield, rel_tol=1e-5)


class TestLaw79DamageAccumulation:
    """Verify progressive damage growth D += Delta eps_p / eps_p_fail."""

    def test_damage_growth_with_plastic_strain(self):
        d1 = 0.5
        d2 = 1.0
        mat = _make_sic_law79(a=1.0, n=0.0, d1=d1, d2=d2, c=0.0)
        phel = mat.params["phel"]
        tstar = mat.params["tstar"]
        g = mat.params["shear"]

        # P* = 1.0 => epfail = D1 * (1.0 + T*)^1.0
        p_val = 1.0 * phel
        mu = p_val / mat.params["k1"]
        epfail_expected = d1 * (1.0 + tstar)

        extra = {"mu": mu, "dmg": 0.0}
        # Yield stress = 1.0 * shel = 14.055
        # Apply plastic shear increment with gamma = 0.1 => vm_trial = 33.42 > 14.055
        gamma = 0.1
        deps = np.array([0.0, 0.0, 0.0, gamma, 0.0, 0.0])
        sig = np.zeros(6)

        sig_out, epsp_out, _ = solid_update(mat, sig, deps, dt=1.0, extra=extra, return_tuple=True)

        dmg_updated = extra["dmg"]
        assert dmg_updated > 0.0
        expected_dmg = epsp_out / epfail_expected
        assert math.isclose(dmg_updated, min(1.0, expected_dmg), rel_tol=1e-5)


class TestLaw79EOSAndBulking:
    """Verify nonlinear EOS and shear energy loss conversion to bulking pressure."""

    def test_cubic_eos_hydrostatic_pressure(self):
        k1, k2, k3 = 200.0, 150.0, 50.0
        mat = _make_sic_law79(k1=k1, k2=k2, k3=k3)

        mu = 0.04
        deps = np.array([-mu / 3.0, -mu / 3.0, -mu / 3.0, 0.0, 0.0, 0.0])
        extra = {"mu": mu}
        sig_out = solid_update(mat, np.zeros(6), deps, extra=extra)

        expected_p = k1 * mu + k2 * (mu**2) + k3 * (mu**3)
        for i in range(3):
            assert math.isclose(-sig_out[i], expected_p, rel_tol=1e-10)

    def test_bulking_pressure_increment(self):
        """Verify Delta P = -P1 + sqrt((Delta P_old + P1)^2 + 2 * beta * K1 * Delta U)."""
        mat = _make_sic_law79(a=1.0, b=0.2, n=0.0, m=0.0, beta=1.0, k1=200.0)
        shel = mat.params["shel"]
        g = mat.params["shear"]
        k1 = mat.params["k1"]

        # Impose damage growth from dmg_old=0.0 to dmg=0.5
        # Current yield drops to: YIELD = 0.5 * 1.0 + 0.5 * 0.2 = 0.6
        # Delta U = (1.0^2 - 0.6^2) / (6 * G) * SHEL^2
        sigy_old = 1.0
        yield_curr = 0.6
        deltau = (sigy_old**2 - yield_curr**2) / (6.0 * g) * (shel**2)

        mu = 0.02
        p1 = k1 * mu
        expected_deltap = -p1 + math.sqrt((0.0 + p1)**2 + 2.0 * 1.0 * k1 * deltau)

        extra = {
            "mu": mu,
            "dmg": 0.5,
            "dmg_old": 0.0,
            "sigy_old": sigy_old,
            "deltap": 0.0,
        }
        deps = np.array([0.0, 0.0, 0.0, 0.02, 0.0, 0.0])
        solid_update(mat, np.zeros(6), deps, extra=extra)

        assert math.isclose(extra["deltap"], expected_deltap, rel_tol=1e-5)


class TestLaw79DeletionModes:
    """Verify element deletion options IDEL = 1, 2, 3."""

    def test_idel1_hydrostatic_tensile_failure(self):
        mat = _make_sic_law79(idel=1, t=0.2, phel=2.0)
        # P* + T* < 0 triggers rupture (off -> 0.8)
        # PHEL = 2.0, T = 0.2 => T* = 0.1
        # If P = -0.3 => P* = -0.15 => P* + T* = -0.05 < 0
        mu = -0.3 / mat.params["k1"]
        extra = {"mu": mu, "off": 1.0}
        solid_update(mat, np.zeros(6), np.zeros(6), extra=extra)
        assert extra["off"] == 0.8

    def test_idel2_plastic_strain_failure(self):
        epsmax = 0.05
        mat = _make_sic_law79(idel=2, epsmax=epsmax, a=0.5, n=0.0)
        extra = {"mu": 0.0, "off": 1.0}
        # Large plastic increment exceeding epsmax
        deps = np.array([0.0, 0.0, 0.0, 0.15, 0.0, 0.0])
        solid_update(mat, np.zeros(6), deps, epsp=0.04, extra=extra)
        assert extra["off"] == 0.8

    def test_idel3_full_damage_failure(self):
        mat = _make_sic_law79(idel=3, d1=0.001)  # small d1 so damage hits 1.0 quickly
        extra = {"mu": 0.0, "dmg": 0.99, "off": 1.0}
        deps = np.array([0.0, 0.0, 0.0, 0.02, 0.0, 0.0])
        solid_update(mat, np.zeros(6), deps, extra=extra)
        assert extra["dmg"] >= 1.0
        assert extra["off"] == 0.8


class TestLaw79TangentsAndSoundSpeed:
    """Verify acoustic sound speed and algorithmic tangent tensor."""

    def test_sound_speed_compression_vs_tension(self):
        k1, k2, k3 = 200.0, 100.0, 50.0
        shear = 150.0
        rho0 = 3.0
        mat = _make_sic_law79(k1=k1, k2=k2, k3=k3, shear=shear, rho0=rho0)

        # In compression (mu = 0.05): dpdmu = K1 + 2*K2*mu + 3*K3*mu^2
        mu_comp = 0.05
        c_comp = sound_speed_solid(mat, extra={"mu": mu_comp})
        dpdmu = k1 + 2.0 * k2 * mu_comp + 3.0 * k3 * (mu_comp**2)
        expected_c = math.sqrt((dpdmu + (4.0 / 3.0) * shear) / rho0)
        assert math.isclose(c_comp, expected_c, rel_tol=1e-10)

        # In tension (mu = -0.02): dpdmu = K1
        c_tens = sound_speed_solid(mat, extra={"mu": -0.02})
        expected_c_tens = math.sqrt((k1 + (4.0 / 3.0) * shear) / rho0)
        assert math.isclose(c_tens, expected_c_tens, rel_tol=1e-10)

    def test_tangent_tensor_dispatch(self):
        mat = _make_sic_law79()
        D_sym = consistent_solid_tangent(mat, symmetric=True)
        assert D_sym.shape == (6, 6)
        assert np.allclose(D_sym, D_sym.T)
        eigvals = np.linalg.eigvalsh(D_sym)
        assert np.all(eigvals > 0.0)

        # Verify aliases
        D_alias1 = solid_tangent(mat, symmetric=True)
        D_alias2 = tangent(mat, symmetric=True)
        assert np.allclose(D_sym, D_alias1)
        assert np.allclose(D_sym, D_alias2)
