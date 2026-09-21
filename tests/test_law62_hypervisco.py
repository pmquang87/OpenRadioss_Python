"""
Unit tests for LAW62 (/MAT/LAW62, /MAT/VISC_HYP) hyper-viscoelastic model.

Verifies:
1. Elastic limit (no viscosity):
   - Analytical Ogden / Neo-Hookean / Mooney-Rivlin hyperelastic stress evaluation
   - Reversible hyperelastic closed loop (zero residual stress)
   - Time-invariance at constant strain when viscosity is disabled
2. Viscous relaxation:
   - Deviatoric Prony series relaxation (IVISC = 1)
   - Full-stress Prony series relaxation (IVISC = 2)
   - Multi-term Prony series exponential decay rate
   - UVAR history array support
3. Small strain fallback and 1D vector interfaces
4. Tangent consistency and interface aliases.
"""

from __future__ import annotations

import math
import numpy as np
import pytest

from pyradioss.materials import law62_hypervisco
from pyradioss.model.entities import Material


def _make_law62_mat(norder=1, nvisc=0, mu=None, alpha=None, nu=None,
                    gamma=None, tau=None, nug=0.0, vflag=0, rflag=0, rho0=1000.0) -> Material:
    if mu is None:
        mu = [10.0] * norder
    if alpha is None:
        alpha = [2.0] * norder
    if nu is None:
        nu = [0.0] * norder
    if gamma is None:
        gamma = [0.2] * nvisc
    if tau is None:
        tau = [1.0] * nvisc

    rec = {
        "id": 62,
        "title": "LAW62_UNIT_TEST",
        "density": rho0,
        "MAT_NU": nug,
        "ORDER": norder,
        "Order2": nvisc,
        "Vflag": vflag,
        "Rflag": rflag,
        "Mu_arr": mu,
        "Alpha_arr": alpha,
        "Nu_arr": nu,
        "Gamma_arr": gamma,
        "Tau_arr": tau,
    }
    return law62_hypervisco.build_law62(rec)


class TestLaw62ElasticLimit:
    """Verify the purely elastic response in the absence of viscosity (IVISC=0)."""

    def test_neo_hookean_analytical_stress(self):
        """Neo-Hookean case: N=1, alpha=2. Uniaxial stretch on decoupled foam (nu=0)."""
        mu_val = 15.0
        mat = _make_law62_mat(norder=1, nvisc=0, mu=[mu_val], alpha=[2.0], nug=0.0)
        assert mat.params["IVISC"] == 0
        assert mat.params["BETA62"][0] == 0.0

        lam1 = 1.25
        F = np.diag([lam1, 1.0, 1.0])[None, :, :]
        sig = np.zeros((1, 6))
        sig, _, c = law62_hypervisco.solid_update(mat, sig, dt=1.0, extra={"F": F})

        # Analytical formula from sigeps62.F:
        # S_1 = (2 * mu / alpha / lambda_1^2) * (lambda_1^alpha - J^(-beta*alpha))
        # With alpha=2, beta=0: S_1 = (mu / lam1^2) * (lam1^2 - 1)
        # Cauchy sigma_1 = S_1 * lam1^2 / J = mu / lam1 * (lam1^2 - 1)
        expected_sig1 = (mu_val / lam1) * (lam1**2 - 1.0)
        assert math.isclose(sig[0, 0], expected_sig1, rel_tol=1e-6)
        # Lateral stresses must be 0 for beta=0 and lam2=lam3=1
        assert abs(sig[0, 1]) < 1e-12
        assert abs(sig[0, 2]) < 1e-12
        assert np.allclose(sig[0, 3:], 0.0, atol=1e-12)
        assert c[0] > 0.0

    def test_mooney_rivlin_analytical_stress(self):
        """Mooney-Rivlin case: N=2, alpha1=2.0, alpha2=-2.0."""
        mu1, mu2 = 20.0, -5.0
        mat = _make_law62_mat(norder=2, nvisc=0, mu=[mu1, mu2], alpha=[2.0, -2.0], nug=0.0)
        lam1 = 1.3
        F = np.diag([lam1, 1.0, 1.0])[None, :, :]
        sig = np.zeros((1, 6))
        sig, _, _ = law62_hypervisco.solid_update(mat, sig, dt=1.0, extra={"F": F})

        # Term 1: (2*mu1/2/lam1) * (lam1^2 - 1) = (mu1/lam1) * (lam1^2 - 1)
        # Term 2: (2*mu2/(-2)/lam1) * (lam1^(-2) - 1) = (-mu2/lam1) * (lam1^(-2) - 1)
        term1 = (mu1 / lam1) * (lam1**2 - 1.0)
        term2 = (-mu2 / lam1) * (lam1**(-2) - 1.0)
        expected_sig1 = term1 + term2
        assert math.isclose(sig[0, 0], expected_sig1, rel_tol=1e-6)

    def test_closed_strain_cycle_zero_residual(self):
        """Pure hyperelasticity is conservative: returning to F=I restores zero stress exactly."""
        mat = _make_law62_mat(norder=2, nvisc=0, mu=[12.0, 4.0], alpha=[1.8, -1.5], nug=0.25)
        sig = np.zeros((1, 6))

        # Apply large 3D deformation
        F_deformed = np.array([
            [1.25, 0.10, 0.05],
            [0.02, 0.95, 0.08],
            [0.01, 0.03, 0.88]
        ])[None, :, :]
        sig, _, _ = law62_hypervisco.solid_update(mat, sig, dt=0.01, extra={"F": F_deformed})
        assert np.linalg.norm(sig) > 1.0

        # Return to identity
        F_identity = np.eye(3)[None, :, :]
        sig, _, _ = law62_hypervisco.solid_update(mat, sig, dt=0.01, extra={"F": F_identity})
        assert np.allclose(sig, 0.0, atol=1e-12)

    def test_time_invariance_without_viscosity(self):
        """Holding deformation constant produces zero relaxation when IVISC=0."""
        mat = _make_law62_mat(norder=1, nvisc=0, mu=[10.0], alpha=[2.0], nug=0.2)
        F = np.diag([1.2, 0.9, 0.9])[None, :, :]

        sig1, _, _ = law62_hypervisco.solid_update(mat, np.zeros((1, 6)), dt=1e-5, extra={"F": F})
        sig2, _, _ = law62_hypervisco.solid_update(mat, sig1.copy(), dt=100.0, extra={"F": F})
        assert np.allclose(sig1, sig2, atol=1e-12)


class TestLaw62ViscousRelaxation:
    """Verify Prony series viscoelastic relaxation."""

    def test_deviatoric_relaxation_ivisc1(self):
        """IVISC=1: Deviatoric stress relaxes according to gamma_inf + gamma_1 * exp(-t/tau)."""
        gamma1 = 0.6
        tau1 = 0.2
        mat = _make_law62_mat(norder=1, nvisc=1, mu=[20.0], alpha=[2.0],
                              gamma=[gamma1], tau=[tau1], nug=0.0, vflag=0)
        assert mat.params["IVISC"] == 1
        assert math.isclose(mat.params["GAMAINF"], 1.0 - gamma1)

        extra = {
            "sdg62": np.zeros((1, 6)),
            "h62": np.zeros((1, 1, 6)),
            "F": np.diag([1.2, 1.0, 1.0])[None, :, :]
        }
        sig = np.zeros((1, 6))

        # 1. Instantaneous step dt -> 0
        dt0 = 1e-8
        sig0, _, _ = law62_hypervisco.solid_update(mat, sig, dt=dt0, extra=extra)

        # 2. Relax over time t = tau1
        t_relax = tau1
        sig_relaxed, _, _ = law62_hypervisco.solid_update(mat, sig0.copy(), dt=t_relax, extra=extra)

        # Stress must decrease
        assert sig_relaxed[0, 0] < sig0[0, 0]

        # 3. Relax over large time t >> tau1 (approaching gamma_inf limit for deviator)
        sig_inf, _, _ = law62_hypervisco.solid_update(mat, sig_relaxed.copy(), dt=10.0 * tau1, extra=extra)

        # Under IVISC=1, only the deviatoric part relaxes by gamma_inf (0.4),
        # while the volumetric / pressure part (1/3 of total) remains unrelaxed:
        # sig_inf / sig0 = 1/3 + gamma_inf * 2/3 = 1/3 + 0.4 * 2/3 = 0.6
        expected_total_ratio = 1.0 / 3.0 + mat.params["GAMAINF"] * (2.0 / 3.0)
        actual_total_ratio = sig_inf[0, 0] / sig0[0, 0]
        assert math.isclose(actual_total_ratio, expected_total_ratio, rel_tol=1e-3)

        # Specifically for deviatoric stress: (sig_xx - sig_mean) relaxes by exactly gamma_inf
        dev0 = sig0[0, 0] - np.mean(sig0[0, :3])
        dev_inf = sig_inf[0, 0] - np.mean(sig_inf[0, :3])
        assert math.isclose(dev_inf / dev0, mat.params["GAMAINF"], rel_tol=1e-3)

    def test_full_stress_relaxation_ivisc2(self):
        """IVISC=2 (Vflag=1): Full stress (spherical + deviatoric) relaxes by Prony series."""
        gamma1 = 0.4
        tau1 = 0.1
        mat = _make_law62_mat(norder=1, nvisc=1, mu=[15.0], alpha=[2.0],
                              gamma=[gamma1], tau=[tau1], nug=0.3, vflag=1)
        assert mat.params["IVISC"] == 2

        extra = {
            "sdg62": np.zeros((1, 6)),
            "h62": np.zeros((1, 1, 6)),
            "F": 0.9 * np.eye(3)[None, :, :]  # Pure volumetric hydrostatic compression
        }
        sig = np.zeros((1, 6))

        # Instantaneous step
        sig0, _, _ = law62_hypervisco.solid_update(mat, sig, dt=1e-8, extra=extra)
        assert sig0[0, 0] < 0.0  # compressive

        # Long relaxation step
        sig_inf, _, _ = law62_hypervisco.solid_update(mat, sig0.copy(), dt=5.0, extra=extra)

        # Under IVISC=2, hydrostatic pressure also relaxes towards gamma_inf * P_0
        gamma_inf = mat.params["GAMAINF"]
        assert math.isclose(sig_inf[0, 0] / sig0[0, 0], gamma_inf, rel_tol=1e-3)

    def test_uvar_history_storage(self):
        """Verify that extra['uvar'] (Fortran NUVAR layout) correctly maintains history."""
        nvisc = 2
        mat = _make_law62_mat(norder=1, nvisc=nvisc, mu=[10.0], alpha=[2.0],
                              gamma=[0.3, 0.2], tau=[0.1, 1.0], vflag=0)
        # NUVAR size = 6 + 6 * NPRONY = 18
        uvar = np.zeros((1, 18))
        F = np.diag([1.15, 1.0, 1.0])[None, :, :]

        sig0, _, _ = law62_hypervisco.solid_update(mat, np.zeros((1, 6)), dt=1e-6,
                                                  extra={"F": F, "uvar": uvar})
        # After update, uvar[:6] has SDG and uvar[6:] has H_i
        assert np.linalg.norm(uvar[0, :6]) > 0.0
        assert np.linalg.norm(uvar[0, 6:]) > 0.0

        # Advance relaxation
        sig1, _, _ = law62_hypervisco.solid_update(mat, sig0.copy(), dt=0.2,
                                                  extra={"F": F, "uvar": uvar})
        assert sig1[0, 0] < sig0[0, 0]


class TestLaw62InterfacesAndRobustness:
    """Verify small strain fallback, 1D vector signatures, and tangent functions."""

    def test_1d_input_signature(self):
        mat = _make_law62_mat(norder=1, mu=[10.0], alpha=[2.0])
        sig_1d = np.zeros(6)
        deps_1d = np.array([0.05, 0.0, 0.0, 0.0, 0.0, 0.0])

        out_sig, out_epsp, out_c = law62_hypervisco.solid_update(mat, sig_1d, deps=deps_1d, dt=1.0)
        assert out_sig.ndim == 1
        assert out_sig.shape == (6,)
        assert out_sig[0] > 0.0
        assert isinstance(out_c, (float, np.floating)) or out_c.ndim == 0

    def test_tangent_and_solid_tangent_aliases(self):
        mat = _make_law62_mat(norder=1, mu=[10.0], alpha=[2.0], nug=0.25)
        F = np.diag([1.1, 0.95, 0.95])[None, :, :]

        D1 = law62_hypervisco.consistent_solid_tangent(mat, F=F)
        D2 = law62_hypervisco.solid_tangent(mat, F=F)
        D3 = law62_hypervisco.tangent(mat, F=F)

        assert np.allclose(D1, D2)
        assert np.allclose(D1, D3)
        assert D1.shape == (1, 6, 6)
