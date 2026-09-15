"""Auditor 2B verification tests for /MAT/LAW12 consistent solid tangent stiffness.

Milestone M546: Tangent Stiffness Auditor for /MAT/LAW12 (/MAT/3D_COMP, /MAT/COMP_3D).
Cites:
  - Starter reader: starter/source/materials/mat/mat012/hm_read_mat12.F
  - Engine kernel:  engine/source/materials/mat/mat012/m12law.F
  - Coordinate rot: engine/source/materials/mat/mat014/m14ama.F, m14gtf.F, m14ftg.F
  - CFG spec:       hm_cfg_files/config/CFG/radioss2020/MAT/3d_comp_12.cfg
"""

from __future__ import annotations

import math
from typing import Any, Dict, Optional

import numpy as np
import pytest

from pyradioss.materials.law12_comp3d import (
    build_law12,
    consistent_solid_tangent,
    solid_update,
)


# ============================================================================
# Helpers
# ============================================================================

def _copy_extra(extra: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """Deep copy dictionary of arrays / scalar state variables for LAW12."""
    if extra is None:
        return None
    res: Dict[str, Any] = {}
    for k, v in extra.items():
        if isinstance(v, np.ndarray):
            res[k] = v.copy()
        elif isinstance(v, dict):
            res[k] = _copy_extra(v)
        else:
            res[k] = v
    return res


def _compute_numerical_tangent(
    mat: Any,
    sig: np.ndarray,
    deps: np.ndarray,
    epsp: Optional[np.ndarray] = None,
    dt: float = 0.0,
    extra: Optional[Dict[str, Any]] = None,
    h: float = 1.0e-7,
) -> np.ndarray:
    """Compute (6, 6) numerical tangent tensor via central finite difference:

    D_num[:, j] = (solid_update(sig, deps + h*e_j)[0] - solid_update(sig, deps - h*e_j)[0]) / (2*h)
    """
    sig_arr = np.asarray(sig, dtype=float)
    deps_arr = np.asarray(deps, dtype=float)
    is_1d = (sig_arr.ndim == 1)
    sig_2d = sig_arr[None, :] if is_1d else sig_arr
    deps_2d = deps_arr[None, :] if deps_arr.ndim == 1 else deps_arr
    n = sig_2d.shape[0]

    d_num = np.zeros((n, 6, 6), dtype=float)

    for i in range(n):
        for j in range(6):
            ej = np.zeros(6, dtype=float)
            ej[j] = h

            ex_p = _copy_extra(extra)
            ex_m = _copy_extra(extra)

            s_p, _, _ = solid_update(
                mat,
                sig_2d[i].copy(),
                deps_2d[i] + ej,
                epsp=epsp.copy() if epsp is not None else None,
                dt=dt,
                extra=ex_p,
            )
            s_m, _, _ = solid_update(
                mat,
                sig_2d[i].copy(),
                deps_2d[i] - ej,
                epsp=epsp.copy() if epsp is not None else None,
                dt=dt,
                extra=ex_m,
            )
            d_num[i, :, j] = (s_p - s_m) / (2.0 * h)

    if is_1d:
        return d_num[0]
    return d_num


def _make_orthotropic_material(
    e11: float = 100000.0,
    e22: float = 50000.0,
    e33: float = 20000.0,
    nu12: float = 0.25,
    nu23: float = 0.2,
    nu31: float = 0.15,
    g12: float = 15000.0,
    g23: float = 10000.0,
    g31: float = 12000.0,
    sigt1: float = 150.0,
    sigt2: float = 100.0,
    sigt3: float = 80.0,
    delta: float = 0.05,
    sigyt1: float = 250.0,
    sigyc1: float = 250.0,
    sigyt2: float = 200.0,
    sigyc2: float = 200.0,
    sigyt3: float = 180.0,
    sigyc3: float = 180.0,
    sigyt12: float = 100.0,
    sigyc12: float = 100.0,
    sigyt23: float = 80.0,
    sigyc23: float = 80.0,
    sigyt13: float = 90.0,
    sigyc13: float = 90.0,
    b: float = 500.0,
    n: float = 0.5,
    fmax: float = 1000.0,
    wplaref: float = 1.0,
    c_rate: float = 0.0,
    eps0: float = 1.0,
    icc: int = 1,
) -> Any:
    """Helper to instantiate LAW12 with consistent orthotropic and Tsai-Wu properties."""
    return build_law12(
        id=12,
        rho0=1.5e-9,
        E11=e11,
        E22=e22,
        E33=e33,
        nu12=nu12,
        nu23=nu23,
        nu31=nu31,
        G12=g12,
        G23=g23,
        G31=g31,
        sigt1=sigt1,
        sigt2=sigt2,
        sigt3=sigt3,
        delta=delta,
        sigyt1=sigyt1,
        sigyc1=sigyc1,
        sigyt2=sigyt2,
        sigyc2=sigyc2,
        sigyt3=sigyt3,
        sigyc3=sigyc3,
        sigyt12=sigyt12,
        sigyc12=sigyc12,
        sigyt23=sigyt23,
        sigyc23=sigyc23,
        sigyt13=sigyt13,
        sigyc13=sigyc13,
        b=b,
        n=n,
        fmax=fmax,
        wplaref=wplaref,
        c=c_rate,
        eps0=eps0,
        icc=icc,
    )


# ============================================================================
# 1. Elastic Tangent Structure, Orthotropic Moduli, and Symmetry
# ============================================================================

class TestLaw12ElasticTangent:
    """Verify exact 6x6 orthotropic elasticity tensor in uncracked, unyielded state."""

    def test_elastic_tangent_fully_orthotropic_structure(self):
        """Must match exact 6x6 orthotropic block structure with zero off-axis shear coupling."""
        mat = _make_orthotropic_material(
            e11=120000.0,
            e22=60000.0,
            e33=25000.0,
            nu12=0.28,
            nu23=0.22,
            nu31=0.18,
            g12=14000.0,
            g23=9000.0,
            g31=11000.0,
        )
        p = mat.params
        d11, d12, d13 = p["D11"], p["D12"], p["D13"]
        d22, d23, d33 = p["D22"], p["D23"], p["D33"]
        g12, g23, g31 = p["G12"], p["G23"], p["G31"]

        expected_d = np.array([
            [d11, d12, d13, 0.0, 0.0, 0.0],
            [d12, d22, d23, 0.0, 0.0, 0.0],
            [d13, d23, d33, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, g12, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, g23, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, g31],
        ], dtype=float)

        c_alg = consistent_solid_tangent(mat, np.zeros(6))
        np.testing.assert_allclose(c_alg, expected_d, rtol=1e-12, atol=1e-12)

    def test_elastic_tangent_transversely_isotropic(self):
        """Verify transversely isotropic material (in-plane isotropic 2-3 plane)."""
        e1 = 150000.0
        e2 = 30000.0
        nu12 = 0.25
        nu23 = 0.35
        nu31 = nu12 * (e2 / e1)
        g12 = 12000.0
        g31 = g12
        g23 = e2 / (2.0 * (1.0 + nu23))

        mat = _make_orthotropic_material(
            e11=e1,
            e22=e2,
            e33=e2,
            nu12=nu12,
            nu23=nu23,
            nu31=nu31,
            g12=g12,
            g23=g23,
            g31=g31,
        )
        c_alg = consistent_solid_tangent(mat, np.zeros(6))
        p = mat.params

        # Transverse isotropy implies D22 == D33 and D12 == D13
        assert p["D22"] == pytest.approx(p["D33"], rel=1e-10)
        assert p["D12"] == pytest.approx(p["D13"], rel=1e-10)
        assert c_alg[1, 1] == pytest.approx(c_alg[2, 2], rel=1e-10)
        assert c_alg[0, 1] == pytest.approx(c_alg[0, 2], rel=1e-10)
        assert c_alg[3, 3] == pytest.approx(c_alg[5, 5], rel=1e-10)

    def test_elastic_tangent_isotropic_limit(self):
        """In the isotropic limit, D11 = lambda + 2*mu, D12 = lambda, G = mu."""
        e_iso = 70000.0
        nu_iso = 0.3
        mu_iso = e_iso / (2.0 * (1.0 + nu_iso))
        lambda_iso = (e_iso * nu_iso) / ((1.0 + nu_iso) * (1.0 - 2.0 * nu_iso))

        mat = _make_orthotropic_material(
            e11=e_iso,
            e22=e_iso,
            e33=e_iso,
            nu12=nu_iso,
            nu23=nu_iso,
            nu31=nu_iso,
            g12=mu_iso,
            g23=mu_iso,
            g31=mu_iso,
        )
        c_alg = consistent_solid_tangent(mat, np.zeros(6))

        assert c_alg[0, 0] == pytest.approx(lambda_iso + 2.0 * mu_iso, rel=1e-10)
        assert c_alg[1, 1] == pytest.approx(lambda_iso + 2.0 * mu_iso, rel=1e-10)
        assert c_alg[2, 2] == pytest.approx(lambda_iso + 2.0 * mu_iso, rel=1e-10)
        assert c_alg[0, 1] == pytest.approx(lambda_iso, rel=1e-10)
        assert c_alg[0, 2] == pytest.approx(lambda_iso, rel=1e-10)
        assert c_alg[1, 2] == pytest.approx(lambda_iso, rel=1e-10)
        assert c_alg[3, 3] == pytest.approx(mu_iso, rel=1e-10)
        assert c_alg[4, 4] == pytest.approx(mu_iso, rel=1e-10)
        assert c_alg[5, 5] == pytest.approx(mu_iso, rel=1e-10)

    def test_elastic_tangent_symmetry(self):
        """Verify exact symmetry D = D^T for uncracked elastic material."""
        mat = _make_orthotropic_material()
        c_alg = consistent_solid_tangent(mat, np.zeros(6))
        asym = np.max(np.abs(c_alg - c_alg.T))
        assert asym < 1e-12, f"Elastic tangent must be strictly symmetric, got asym={asym}"

    def test_elastic_tangent_positive_definiteness(self):
        """Verify all eigenvalues of elastic tangent are strictly positive."""
        for factor in [0.5, 1.0, 2.5]:
            mat = _make_orthotropic_material(
                e11=100000.0 * factor,
                e22=50000.0 * factor,
                e33=20000.0 * factor,
                g12=15000.0 * factor,
                g23=10000.0 * factor,
                g31=12000.0 * factor,
            )
            c_alg = consistent_solid_tangent(mat, np.zeros(6))
            eigvals = np.linalg.eigvalsh(c_alg)
            assert np.all(eigvals > 0.0), f"All eigenvalues must be positive, got {eigvals}"
            assert np.min(eigvals) >= 9000.0 * factor

    def test_elastic_tangent_prestressed_state(self):
        """Pre-stressed state inside the elastic yield and uncracked envelope maintains D_mat."""
        mat = _make_orthotropic_material()
        sig_pre = np.array([40.0, 20.0, -10.0, 15.0, 5.0, -8.0])
        c_alg = consistent_solid_tangent(mat, sig_pre)
        c_zero = consistent_solid_tangent(mat, np.zeros(6))
        np.testing.assert_allclose(c_alg, c_zero, atol=1e-12)


# ============================================================================
# 2. Central Finite Difference Verification (h = 1e-6 and 1e-7)
# ============================================================================

class TestLaw12CentralDifferenceVerification:
    """Verify central difference perturbations match algorithmic tangent to < 1e-5."""

    @pytest.mark.parametrize("h_step", [1.0e-6, 1.0e-7])
    def test_central_difference_elastic_h(self, h_step: float):
        """Verify numerical perturbation matches consistent tangent at h=1e-6 and 1e-7."""
        mat = _make_orthotropic_material()
        sig0 = np.array([30.0, -15.0, 10.0, 12.0, -8.0, 6.0])
        deps0 = np.array([0.0001, -0.00005, 0.00004, 0.00008, -0.00003, 0.00005])

        c_alg = consistent_solid_tangent(mat, sig0, deps=deps0, h=h_step)
        c_num = _compute_numerical_tangent(mat, sig0, deps0, h=h_step)

        diff = np.max(np.abs(c_alg - c_num))
        rel_diff = diff / np.linalg.norm(c_alg)
        assert rel_diff < 1e-5, f"Relative difference {rel_diff:.3e} exceeds 1e-5 for h={h_step}"

    def test_central_difference_all_6_components_individual(self):
        """Verify all 6 strain components individually perturbed match column-by-column."""
        mat = _make_orthotropic_material()
        sig0 = np.array([25.0, 15.0, -10.0, 8.0, 4.0, -5.0])
        deps0 = np.zeros(6)
        h = 1.0e-7

        c_alg = consistent_solid_tangent(mat, sig0, deps=deps0, h=h)
        for comp in range(6):
            ej = np.zeros(6)
            ej[comp] = h
            sp, _, _ = solid_update(mat, sig0.copy(), deps0 + ej)
            sm, _, _ = solid_update(mat, sig0.copy(), deps0 - ej)
            col_num = (sp - sm) / (2.0 * h)

            np.testing.assert_allclose(
                c_alg[:, comp],
                col_num,
                rtol=1e-5,
                atol=1e-5,
                err_msg=f"Column {comp} mismatch between algorithmic and numerical tangent",
            )

    def test_central_difference_multiaxial_strain_increment(self):
        """Verify central difference under full 6-component multi-axial strain state."""
        mat = _make_orthotropic_material()
        sig0 = np.array([50.0, 20.0, 10.0, 15.0, 8.0, 12.0])
        deps0 = np.array([0.0002, 0.00015, -0.0001, 0.00025, -0.00015, 0.0002])

        c_alg = consistent_solid_tangent(mat, sig0, deps=deps0)
        c_num = _compute_numerical_tangent(mat, sig0, deps0)

        diff_max = np.max(np.abs(c_alg - c_num))
        assert diff_max < 1e-4, f"Max absolute difference {diff_max:.3e} exceeds 1e-4"


# ============================================================================
# 3. Tensile Damage Regime: Stress Degradation and Poisson Relief
# ============================================================================

class TestLaw12TensileDamageTangent:
    """Verify tangent reflects stress degradation, D_ii = 0, and Poisson relief."""

    def test_active_tensile_cracking_dir1(self):
        """Tension along fiber direction exceeding sigt1 zeroes direct stiffness D11."""
        sigt1 = 120.0
        delta = 0.08
        mat = _make_orthotropic_material(sigt1=sigt1, delta=delta)
        sig0 = np.zeros(6)
        deps0 = np.array([0.003, 0.0, 0.0, 0.0, 0.0, 0.0])

        c_alg = consistent_solid_tangent(mat, sig0, deps=deps0)
        c_num = _compute_numerical_tangent(mat, sig0, deps0)

        # 1. Direct stiffness in cracking direction is zeroed (stress capped at wvec1)
        assert c_alg[0, 0] == pytest.approx(0.0, abs=1e-6)
        assert c_num[0, 0] == pytest.approx(0.0, abs=1e-6)

        # 2. Exact match with numerical finite difference
        np.testing.assert_allclose(c_alg, c_num, atol=1e-4, rtol=1e-4)

    def test_active_tensile_cracking_poisson_relief(self):
        """Pre-damaged cracking step demonstrates Poisson relief scaling by (1 - dam)."""
        mat = _make_orthotropic_material(sigt1=120.0, delta=0.08)
        sig0 = np.zeros(6)
        deps0 = np.array([0.003, 0.0, 0.0, 0.0, 0.0, 0.0])
        dam_val = 0.15
        extra = {"dam12": np.array([[dam_val, 0.0, 0.0, 0.0, 0.0]])}

        c_alg = consistent_solid_tangent(mat, sig0, deps=deps0, extra=extra)
        c_num = _compute_numerical_tangent(mat, sig0, deps0, extra=extra)

        # In _update_one_element: t2 -= d12 * deps1 * dam1 => dt2/ddeps1 = d12 * (1 - dam_val)
        d12 = mat.params["D12"]
        d13 = mat.params["D13"]
        expected_d21 = d12 * (1.0 - dam_val)
        expected_d31 = d13 * (1.0 - dam_val)

        assert c_alg[0, 0] == pytest.approx(0.0, abs=1e-6)
        assert c_alg[1, 0] == pytest.approx(expected_d21, rel=1e-5)
        assert c_alg[2, 0] == pytest.approx(expected_d31, rel=1e-5)
        np.testing.assert_allclose(c_alg, c_num, atol=1e-4, rtol=1e-4)

    def test_active_tensile_cracking_dir2(self):
        """Tension in direction 2 exceeding sigt2 zeroes D22 with Poisson relief on D12, D32."""
        sigt2 = 90.0
        delta = 0.06
        mat = _make_orthotropic_material(sigt2=sigt2, delta=delta)
        sig0 = np.zeros(6)
        deps0 = np.array([0.0, 0.003, 0.0, 0.0, 0.0, 0.0])
        dam_val = 0.1
        extra = {"dam12": np.array([[0.0, dam_val, 0.0, 0.0, 0.0]])}

        c_alg = consistent_solid_tangent(mat, sig0, deps=deps0, extra=extra)
        c_num = _compute_numerical_tangent(mat, sig0, deps0, extra=extra)

        assert c_alg[1, 1] == pytest.approx(0.0, abs=1e-6)
        assert c_num[1, 1] == pytest.approx(0.0, abs=1e-6)

        d12 = mat.params["D12"]
        d23 = mat.params["D23"]
        expected_d12 = d12 * (1.0 - dam_val)
        expected_d32 = d23 * (1.0 - dam_val)

        assert c_alg[0, 1] == pytest.approx(expected_d12, rel=1e-5)
        assert c_alg[2, 1] == pytest.approx(expected_d32, rel=1e-5)
        np.testing.assert_allclose(c_alg, c_num, atol=1e-4, rtol=1e-4)

    def test_active_tensile_cracking_dir3(self):
        """Tension in direction 3 exceeding sigt3 zeroes D33 with Poisson relief on D13, D23."""
        sigt3 = 60.0
        delta = 0.1
        mat = _make_orthotropic_material(sigt3=sigt3, delta=delta)
        sig0 = np.zeros(6)
        deps0 = np.array([0.0, 0.0, 0.005, 0.0, 0.0, 0.0])
        dam_val = 0.2
        extra = {"dam12": np.array([[0.0, 0.0, dam_val, 0.0, 0.0]])}

        c_alg = consistent_solid_tangent(mat, sig0, deps=deps0, extra=extra)
        c_num = _compute_numerical_tangent(mat, sig0, deps0, extra=extra)

        assert c_alg[2, 2] == pytest.approx(0.0, abs=1e-6)
        assert c_num[2, 2] == pytest.approx(0.0, abs=1e-6)

        d13 = mat.params["D13"]
        d23 = mat.params["D23"]
        expected_d13 = d13 * (1.0 - dam_val)
        expected_d23 = d23 * (1.0 - dam_val)

        assert c_alg[0, 2] == pytest.approx(expected_d13, rel=1e-5)
        assert c_alg[1, 2] == pytest.approx(expected_d23, rel=1e-5)
        np.testing.assert_allclose(c_alg, c_num, atol=1e-4, rtol=1e-4)

    def test_multi_axial_simultaneous_cracking(self):
        """Simultaneous tensile cracking in directions 1 and 2 zeroes both D11 and D22."""
        mat = _make_orthotropic_material(sigt1=100.0, sigt2=80.0, delta=0.05)
        sig0 = np.zeros(6)
        deps0 = np.array([0.003, 0.003, 0.0, 0.0, 0.0, 0.0])

        c_alg = consistent_solid_tangent(mat, sig0, deps=deps0)
        c_num = _compute_numerical_tangent(mat, sig0, deps0)

        assert c_alg[0, 0] == pytest.approx(0.0, abs=1e-6)
        assert c_alg[1, 1] == pytest.approx(0.0, abs=1e-6)
        np.testing.assert_allclose(c_alg, c_num, atol=1e-4, rtol=1e-4)

    def test_cracking_tangent_finite_difference_accuracy(self):
        """Relative error between algorithmic and numerical cracking tangent is < 1e-4."""
        mat = _make_orthotropic_material(sigt1=100.0, delta=0.05)
        sig0 = np.zeros(6)
        deps0 = np.array([0.0025, 0.0005, -0.0005, 0.001, 0.0, 0.0])

        c_alg = consistent_solid_tangent(mat, sig0, deps=deps0)
        c_num = _compute_numerical_tangent(mat, sig0, deps0)

        norm_alg = np.linalg.norm(c_alg)
        rel_err = np.linalg.norm(c_alg - c_num) / norm_alg
        assert rel_err < 1e-4, f"Cracking tangent relative error {rel_err:.3e} exceeds 1e-4"


# ============================================================================
# 4. Crack-Opened Unilateral State (Zero Stiffness in Cracked Direction)
# ============================================================================

class TestLaw12CrackOpenedUnilateralState:
    """Verify unilateral behavior: zero compressive stiffness across open crack."""

    def test_crack_opened_compression_zero_stiffness_dir1(self):
        """When crack in direction 1 is open (epc1 > 0), compression has zero normal stiffness."""
        mat = _make_orthotropic_material(sigt1=100.0, delta=0.1)
        extra: Dict[str, Any] = {}
        sig = np.zeros(6)

        # Cycle 1: Tension creates crack (epc1 > 0)
        deps_tens = np.array([0.003, 0.0, 0.0, 0.0, 0.0, 0.0])
        s1, _, _ = solid_update(mat, sig, deps_tens, extra=extra)
        assert extra["epc12"][0, 0] > 0.0

        # Cycle 2: Compressive increment that puts trial stress into compression (t1 < 0)
        deps_comp = np.array([-0.001, 0.0, 0.0, 0.0, 0.0, 0.0])
        c_alg = consistent_solid_tangent(mat, s1, deps=deps_comp, extra=extra)
        c_num = _compute_numerical_tangent(mat, s1, deps_comp, extra=extra)

        # Normal stiffness across open crack is zero!
        assert c_alg[0, 0] == pytest.approx(0.0, abs=1e-6)
        assert c_num[0, 0] == pytest.approx(0.0, abs=1e-6)
        np.testing.assert_allclose(c_alg, c_num, atol=1e-4, rtol=1e-4)

    def test_crack_opened_compression_zero_stiffness_dir2(self):
        """When crack in direction 2 is open (epc2 > 0), compression has D22 = 0."""
        mat = _make_orthotropic_material(sigt2=80.0, delta=0.1)
        extra: Dict[str, Any] = {}
        s1, _, _ = solid_update(mat, np.zeros(6), np.array([0.0, 0.004, 0.0, 0.0, 0.0, 0.0]), extra=extra)
        assert extra["epc12"][0, 1] > 0.0

        deps_comp = np.array([0.0, -0.002, 0.0, 0.0, 0.0, 0.0])
        c_alg = consistent_solid_tangent(mat, s1, deps=deps_comp, extra=extra)
        c_num = _compute_numerical_tangent(mat, s1, deps_comp, extra=extra)

        assert c_alg[1, 1] == pytest.approx(0.0, abs=1e-6)
        assert c_num[1, 1] == pytest.approx(0.0, abs=1e-6)
        np.testing.assert_allclose(c_alg, c_num, atol=1e-4, rtol=1e-4)

    def test_crack_opened_compression_zero_stiffness_dir3(self):
        """When crack in direction 3 is open (epc3 > 0), compression has D33 = 0."""
        mat = _make_orthotropic_material(sigt3=60.0, delta=0.1)
        extra: Dict[str, Any] = {}
        s1, _, _ = solid_update(mat, np.zeros(6), np.array([0.0, 0.0, 0.006, 0.0, 0.0, 0.0]), extra=extra)
        assert extra["epc12"][0, 2] > 0.0

        deps_comp = np.array([0.0, 0.0, -0.004, 0.0, 0.0, 0.0])
        c_alg = consistent_solid_tangent(mat, s1, deps=deps_comp, extra=extra)
        c_num = _compute_numerical_tangent(mat, s1, deps_comp, extra=extra)

        assert c_alg[2, 2] == pytest.approx(0.0, abs=1e-6)
        assert c_num[2, 2] == pytest.approx(0.0, abs=1e-6)
        np.testing.assert_allclose(c_alg, c_num, atol=1e-4, rtol=1e-4)

    def test_crack_closure_stiffness_restoration(self):
        """After sufficient compression that completely closes crack (epc -> 0), stiffness is restored."""
        mat = _make_orthotropic_material(sigt1=100.0, delta=0.1)
        extra: Dict[str, Any] = {}

        # 1. Open crack with eps1 = +0.002
        s1, _, _ = solid_update(mat, np.zeros(6), np.array([0.002, 0.0, 0.0, 0.0, 0.0, 0.0]), extra=extra)
        # 2. Re-close crack with exact compressive step -0.002
        s2, _, _ = solid_update(mat, s1, np.array([-0.002, 0.0, 0.0, 0.0, 0.0, 0.0]), extra=extra)
        assert extra["epc12"][0, 0] == 0.0

        # 3. Once closed, further compression has full compressive stiffness D11
        deps_comp_more = np.array([-0.0001, 0.0, 0.0, 0.0, 0.0, 0.0])
        c_alg = consistent_solid_tangent(mat, s2, deps=deps_comp_more, extra=extra)
        c_num = _compute_numerical_tangent(mat, s2, deps_comp_more, extra=extra)

        assert c_alg[0, 0] > 0.0
        assert c_alg[0, 0] == pytest.approx(mat.params["D11"], rel=1e-5)
        np.testing.assert_allclose(c_alg, c_num, atol=1e-4, rtol=1e-4)


# ============================================================================
# 5. Tsai-Wu Plastic Regime: Algorithmic Tangent and Return Map
# ============================================================================

class TestLaw12TsaiWuPlasticTangent:
    """Verify algorithmic tangent during active Tsai-Wu plastic yielding."""

    def test_active_plastic_shear_yielding_12(self):
        """Under pure shear yielding, algorithmic tangent D44 is degraded below elastic G12."""
        mat = _make_orthotropic_material(
            g12=15000.0,
            sigyt12=100.0,
            sigyc12=100.0,
            b=500.0,
            n=0.5,
        )
        sig0 = np.array([0.0, 0.0, 0.0, 90.0, 0.0, 0.0])
        deps0 = np.array([0.0, 0.0, 0.0, 0.002, 0.0, 0.0])

        c_alg = consistent_solid_tangent(mat, sig0, deps=deps0)
        c_num = _compute_numerical_tangent(mat, sig0, deps0)

        # Plastic flow relaxes shear stress: D44 < G12
        assert c_alg[3, 3] < mat.params["G12"]
        assert c_alg[3, 3] > 0.0
        np.testing.assert_allclose(c_alg, c_num, atol=1e-4, rtol=1e-4)

    def test_active_plastic_multiaxial_yielding(self):
        """Combined tension-shear yielding matches numerical perturbation to < 1e-4."""
        mat = _make_orthotropic_material(
            sigyt1=200.0,
            sigyc1=200.0,
            sigyt12=100.0,
            sigyc12=100.0,
            b=600.0,
            n=0.6,
        )
        sig0 = np.array([120.0, 40.0, 10.0, 50.0, 10.0, 10.0])
        deps0 = np.array([0.002, 0.001, -0.001, 0.003, 0.001, 0.001])

        c_alg = consistent_solid_tangent(mat, sig0, deps=deps0)
        c_num = _compute_numerical_tangent(mat, sig0, deps0)

        diff = np.max(np.abs(c_alg - c_num))
        rel_diff = diff / np.linalg.norm(c_alg)
        assert rel_diff < 1e-4, f"Multiaxial plastic tangent error {rel_diff:.3e} exceeds 1e-4"

    def test_tsai_wu_hardening_vs_perfectly_plastic(self):
        """Hardening (b > 0) produces a higher plastic tangent stiffness than perfectly plastic (b = 0)."""
        mat_hard = _make_orthotropic_material(sigyt12=100.0, sigyc12=100.0, b=800.0, n=0.5)
        mat_perf = _make_orthotropic_material(sigyt12=100.0, sigyc12=100.0, b=0.0, n=1.0)

        sig0 = np.array([0.0, 0.0, 0.0, 95.0, 0.0, 0.0])
        deps0 = np.array([0.0, 0.0, 0.0, 0.002, 0.0, 0.0])

        c_hard = consistent_solid_tangent(mat_hard, sig0, deps=deps0)
        c_perf = consistent_solid_tangent(mat_perf, sig0, deps=deps0)

        # Tangent with hardening should be stiffer than perfectly plastic
        assert c_hard[3, 3] > c_perf[3, 3]

    def test_tsai_wu_quasi_symmetry(self):
        """Tsai-Wu algorithmic tangent is quasi-symmetric (asymmetry is small relative to norm)."""
        mat = _make_orthotropic_material(
            sigt1=1.0e6,
            sigt2=1.0e6,
            sigt3=1.0e6,
            sigyt1=200.0,
            sigyc1=200.0,
            sigyt12=100.0,
            sigyc12=100.0,
            b=500.0,
            n=0.5,
        )
        sig0 = np.array([120.0, 50.0, 20.0, 40.0, 10.0, 10.0])
        deps0 = np.array([0.002, 0.001, -0.001, 0.002, 0.001, 0.001])

        c_alg = consistent_solid_tangent(mat, sig0, deps=deps0)
        asym = np.max(np.abs(c_alg - c_alg.T))
        rel_asym = asym / np.linalg.norm(c_alg)
        assert rel_asym < 0.02, f"Tsai-Wu tangent asymmetry {rel_asym:.3%} exceeds 2%"

    def test_tsai_wu_associated_flow_derivative(self):
        """Directional derivative along plastic gradient DP matches numerical return map."""
        mat = _make_orthotropic_material(sigyt1=200.0, sigyc1=200.0, sigyt12=100.0, b=500.0)
        sig0 = np.array([100.0, 30.0, 10.0, 60.0, 0.0, 0.0])
        deps0 = np.array([0.001, 0.0005, 0.0, 0.002, 0.0, 0.0])

        c_alg = consistent_solid_tangent(mat, sig0, deps=deps0)
        c_num = _compute_numerical_tangent(mat, sig0, deps0)

        # Flow direction vector
        v_dir = deps0 / np.linalg.norm(deps0)
        np.testing.assert_allclose(c_alg @ v_dir, c_num @ v_dir, atol=1e-4, rtol=1e-4)


# ============================================================================
# 6. Degradation and Failure States (off = 0.792, off = 0.0)
# ============================================================================

class TestLaw12DegradationAndFailure:
    """Verify behavior under element failure degradation (off < 1.0, off == 0.0)."""

    def test_degraded_state_off_0792(self):
        """When element enters failure (off=0.99), off is updated by 0.8 to 0.792, scaling tangent."""
        mat = _make_orthotropic_material()
        extra_deg = {"off12": np.array([0.99])}
        sig0 = np.zeros(6)
        deps0 = np.array([0.0001, 0.0, 0.0, 0.0, 0.0, 0.0])

        c_alg = consistent_solid_tangent(mat, sig0, deps=deps0, extra=extra_deg)
        c_num = _compute_numerical_tangent(mat, sig0, deps0, extra=extra_deg)

        # Expected scaling: 0.99 * 0.8 = 0.792
        d11_expected = 0.792 * mat.params["D11"]
        assert c_alg[0, 0] == pytest.approx(d11_expected, rel=1e-5)
        np.testing.assert_allclose(c_alg, c_num, atol=1e-4, rtol=1e-4)

    def test_failed_state_off_zero(self):
        """When element is dead (off = 0.0), tangent is identically zero."""
        mat = _make_orthotropic_material()
        extra_dead = {"off12": np.array([0.0])}
        sig0 = np.zeros(6)
        deps0 = np.array([0.0001, 0.0001, 0.0, 0.0, 0.0, 0.0])

        c_alg = consistent_solid_tangent(mat, sig0, deps=deps0, extra=extra_dead)
        c_num = _compute_numerical_tangent(mat, sig0, deps0, extra=extra_dead)

        np.testing.assert_allclose(c_alg, 0.0, atol=1e-12)
        np.testing.assert_allclose(c_num, 0.0, atol=1e-12)

    def test_failed_state_sub_threshold(self):
        """When off < 0.1 (e.g. off = 0.05), m12law zeroes off, yielding zero tangent."""
        mat = _make_orthotropic_material()
        extra_sub = {"off12": np.array([0.05])}
        sig0 = np.zeros(6)
        deps0 = np.array([0.0001, 0.0, 0.0, 0.0, 0.0, 0.0])

        c_alg = consistent_solid_tangent(mat, sig0, deps=deps0, extra=extra_sub)
        np.testing.assert_allclose(c_alg, 0.0, atol=1e-12)

    def test_progressive_degradation_steps(self):
        """Successive steps with off < 1.0 degrade stiffness by 0.8 repeatedly."""
        mat = _make_orthotropic_material()
        # Step 1: off = 0.8 -> updated to 0.8 * 0.8 = 0.64
        extra1 = {"off12": np.array([0.8])}
        c1 = consistent_solid_tangent(mat, np.zeros(6), extra=extra1)
        assert c1[0, 0] == pytest.approx(0.64 * mat.params["D11"], rel=1e-5)

        # Step 2: off = 0.64 -> updated to 0.64 * 0.8 = 0.512
        extra2 = {"off12": np.array([0.64])}
        c2 = consistent_solid_tangent(mat, np.zeros(6), extra=extra2)
        assert c2[0, 0] == pytest.approx(0.512 * mat.params["D11"], rel=1e-5)


# ============================================================================
# 7. Batched Evaluation and Operational Configurations
# ============================================================================

class TestLaw12BatchedAndOperational:
    """Verify batched evaluation (n=10), input shapes, rate sensitivity, and options."""

    def test_batched_evaluation_n10(self):
        """Verify batched evaluation for n=10 elements with mixed states matches single-element."""
        mat = _make_orthotropic_material(sigt1=100.0, sigt2=80.0, sigt3=70.0, sigyt12=100.0, b=500.0)
        n = 10
        sig_batch = np.zeros((n, 6))
        deps_batch = np.zeros((n, 6))
        off_batch = np.ones(n)
        epc_batch = np.zeros((n, 3))
        dam_batch = np.zeros((n, 5))

        # El 0: elastic
        deps_batch[0] = [0.0001, 0, 0, 0, 0, 0]
        # El 1: crack dir 1
        deps_batch[1] = [0.003, 0, 0, 0, 0, 0]
        # El 2: crack dir 2
        deps_batch[2] = [0, 0.004, 0, 0, 0, 0]
        # El 3: crack dir 3
        deps_batch[3] = [0, 0, 0.008, 0, 0, 0]
        # El 4: open crack under compression
        epc_batch[4, 0] = 0.002
        deps_batch[4] = [-0.001, 0, 0, 0, 0, 0]
        # El 5: shear yielding
        sig_batch[5, 3] = 90.0
        deps_batch[5, 3] = 0.002
        # El 6: multiaxial yielding
        sig_batch[6] = [120.0, 40.0, 10.0, 50.0, 10.0, 10.0]
        deps_batch[6] = [0.002, 0.001, -0.001, 0.003, 0.001, 0.001]
        # El 7: degraded
        off_batch[7] = 0.99
        deps_batch[7] = [0.0001, 0, 0, 0, 0, 0]
        # El 8: failed
        off_batch[8] = 0.0
        deps_batch[8] = [0.0001, 0, 0, 0, 0, 0]
        # El 9: static
        deps_batch[9] = [0, 0, 0, 0, 0, 0]

        extra_batch = {
            "off12": off_batch,
            "epc12": epc_batch,
            "dam12": dam_batch,
        }

        d_batch = consistent_solid_tangent(mat, sig_batch, deps=deps_batch, extra=extra_batch)
        assert d_batch.shape == (10, 6, 6)

        for i in range(n):
            ex_i = {
                "off12": np.array([off_batch[i]]),
                "epc12": epc_batch[i : i + 1],
                "dam12": dam_batch[i : i + 1],
            }
            d_single = consistent_solid_tangent(mat, sig_batch[i], deps=deps_batch[i], extra=ex_i)
            np.testing.assert_allclose(d_batch[i], d_single, atol=1e-12)

    def test_input_shapes_1d_2d_empty(self):
        """Verify shape handling: 1D returns (6, 6), 2D returns (n, 6, 6), empty returns (0, 6, 6)."""
        mat = _make_orthotropic_material()

        # 1D input
        t_1d = consistent_solid_tangent(mat, np.zeros(6))
        assert t_1d.shape == (6, 6)

        # 2D input (1, 6)
        t_2d = consistent_solid_tangent(mat, np.zeros((1, 6)))
        assert t_2d.shape == (1, 6, 6)

        # 2D input (4, 6)
        t_4d = consistent_solid_tangent(mat, np.zeros((4, 6)))
        assert t_4d.shape == (4, 6, 6)

        # Empty input (0, 6)
        t_empty = consistent_solid_tangent(mat, np.empty((0, 6)))
        assert t_empty.shape == (0, 6, 6)

    def test_zero_time_step_static_behavior(self):
        """Verify dt = 0.0 operates in static tangent regime without rate effect errors."""
        mat = _make_orthotropic_material(c_rate=0.05, eps0=1.0)
        sig0 = np.array([50.0, 20.0, 0.0, 10.0, 0.0, 0.0])
        deps0 = np.array([0.0002, 0.0001, 0.0, 0.0001, 0.0, 0.0])

        c_stat = consistent_solid_tangent(mat, sig0, deps=deps0, dt=0.0)
        c_num = _compute_numerical_tangent(mat, sig0, deps0, dt=0.0)
        np.testing.assert_allclose(c_stat, c_num, atol=1e-4)

    def test_dynamic_time_step_rate_enhancement(self):
        """Verify dt > 0 with rate enhancement (c > 0) properly increases yield limit."""
        mat = _make_orthotropic_material(
            sigyt12=100.0,
            sigyc12=100.0,
            c_rate=0.1,
            eps0=1.0,
            icc=1,
        )
        sig0 = np.array([0.0, 0.0, 0.0, 110.0, 0.0, 0.0])
        # High strain rate: deps / dt = 0.001 / 1e-6 = 1000 >> eps0 (1.0)
        dt = 1.0e-6
        deps0 = np.array([0.0, 0.0, 0.0, 0.001, 0.0, 0.0])

        c_dyn = consistent_solid_tangent(mat, sig0, deps=deps0, dt=dt)
        c_num = _compute_numerical_tangent(mat, sig0, deps0, dt=dt)
        np.testing.assert_allclose(c_dyn, c_num, atol=1e-4)

    def test_symmetric_flag(self):
        """Verify symmetric=True enforces strict symmetry D_sym = 0.5*(D + D^T)."""
        mat = _make_orthotropic_material(sigt1=100.0, delta=0.1)
        sig0 = np.zeros(6)
        deps0 = np.array([0.003, 0.0, 0.0, 0.0, 0.0, 0.0])

        # Active cracking is non-symmetric
        c_asym = consistent_solid_tangent(mat, sig0, deps=deps0, symmetric=False)
        assert np.max(np.abs(c_asym - c_asym.T)) > 1.0

        # With symmetric=True, must be strictly symmetric
        c_sym = consistent_solid_tangent(mat, sig0, deps=deps0, symmetric=True)
        assert np.max(np.abs(c_sym - c_sym.T)) == pytest.approx(0.0, abs=1e-12)
        np.testing.assert_allclose(c_sym, 0.5 * (c_asym + c_asym.T), atol=1e-12)

    def test_rotated_axes_tangent(self):
        """Coordinate frame rotation via axes in extra correctly transforms tangent to global coordinates."""
        mat = _make_orthotropic_material()
        theta = np.pi / 4.0
        c, s = math.cos(theta), math.sin(theta)
        # 45-degree rotation in x-y plane
        rot = np.array([
            [c, s, 0.0],
            [-s, c, 0.0],
            [0.0, 0.0, 1.0],
        ], dtype=float)

        extra = {"axes": rot}
        sig0 = np.array([40.0, 20.0, 10.0, 5.0, 0.0, 0.0])
        deps0 = np.array([0.0001, -0.00005, 0.00003, 0.0001, 0.0, 0.0])
        h = 1.0e-7

        c_alg = consistent_solid_tangent(mat, sig0, deps=deps0, extra=extra, h=h)
        c_num = _compute_numerical_tangent(mat, sig0, deps0, extra=extra, h=h)

        diff = np.max(np.abs(c_alg - c_num))
        assert diff < 1e-4, f"Rotated tangent difference {diff:.3e} exceeds 1e-4"
