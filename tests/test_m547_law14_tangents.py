"""Auditor 2B verification tests for /MAT/LAW14 consistent solid tangent stiffness.

Milestone M547: Tangent Stiffness Auditor for /MAT/LAW14 (/MAT/COMPSO, /MAT/COMP_SOL).
Cites:
  - Starter reader: starter/source/materials/mat/mat014/hm_read_mat14.F
  - Engine kernel:  engine/source/materials/mat/mat014/m14law.F
  - Coordinate rot: engine/source/materials/mat/mat014/m14ama.F, m14gtf.F, m14ftg.F
  - CFG spec:       hm_cfg_files/config/CFG/radioss2020/MAT/matl14_compso.cfg
"""

from __future__ import annotations

import math
from typing import Any, Dict, Optional

import numpy as np
import pytest

from pyradioss.materials.law14_compso import (
    build_law14,
    consistent_solid_tangent,
    solid_update,
)


# ============================================================================
# Helpers
# ============================================================================

def _copy_extra(extra: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """Deep copy dictionary of arrays / scalar state variables for LAW14."""
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
    """Helper to instantiate LAW14 with consistent orthotropic and Tsai-Wu properties."""
    return build_law14(
        id=14,
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
# Core Milestone Acceptance Tests
# ============================================================================

def test_tangent_elastic_exact_match():
    """Exact equality to orthotropic D matrix in pure elastic undamaged state."""
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

    expected_d = np.array(
        [
            [d11, d12, d13, 0.0, 0.0, 0.0],
            [d12, d22, d23, 0.0, 0.0, 0.0],
            [d13, d23, d33, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, g12, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, g23, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, g31],
        ],
        dtype=float,
    )

    # 1. At zero stress
    c_zero = consistent_solid_tangent(mat, np.zeros(6))
    assert np.array_equal(c_zero, expected_d)

    # 2. At pre-stressed state within elastic envelope
    sig_pre = np.array([40.0, 20.0, -10.0, 15.0, 5.0, -8.0])
    c_pre = consistent_solid_tangent(mat, sig_pre)
    assert np.array_equal(c_pre, expected_d)


@pytest.mark.parametrize("h_step", [1.0e-6, 1.0e-7])
def test_tangent_fd_elastic(h_step: float):
    """Central FD matches tangent to 1e-5 relative tolerance in elastic regime."""
    mat = _make_orthotropic_material()
    sig0 = np.array([30.0, -15.0, 10.0, 12.0, -8.0, 6.0])
    deps0 = np.array([0.0001, -0.00005, 0.00004, 0.00008, -0.00003, 0.00005])

    c_alg = consistent_solid_tangent(mat, sig0, deps=deps0, h=h_step)
    c_num = _compute_numerical_tangent(mat, sig0, deps0, h=h_step)

    diff = np.max(np.abs(c_alg - c_num))
    rel_diff = diff / np.linalg.norm(c_alg)
    assert rel_diff < 1e-5, f"Relative difference {rel_diff:.3e} exceeds 1e-5 for h={h_step}"


def test_tangent_fd_tensile_damage():
    """Central FD matches tangent during active cracking in tensile damage regime."""
    sigt1 = 120.0
    delta = 0.08
    mat = _make_orthotropic_material(sigt1=sigt1, delta=delta)
    sig0 = np.zeros(6)
    deps0 = np.array([0.003, 0.0, 0.0, 0.0, 0.0, 0.0])

    c_alg = consistent_solid_tangent(mat, sig0, deps=deps0)
    c_num = _compute_numerical_tangent(mat, sig0, deps0)

    # Direct stiffness along cracking direction is zeroed
    assert c_alg[0, 0] == pytest.approx(0.0, abs=1e-6)
    assert c_num[0, 0] == pytest.approx(0.0, abs=1e-6)

    # Algorithmic tangent matches numerical perturbation to high accuracy
    np.testing.assert_allclose(c_alg, c_num, atol=1e-4, rtol=1e-4)


def test_tangent_fd_plasticity():
    """Central FD matches tangent during active Tsai-Wu plastic yielding."""
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
    np.testing.assert_allclose(c_alg, c_num, atol=1e-4, rtol=1e-4)


def test_tangent_batching():
    """(n, 6, 6) batched tangent matches row-by-row single evaluation."""
    mat = _make_orthotropic_material(
        sigt1=100.0,
        sigt2=80.0,
        sigt3=70.0,
        sigyt12=100.0,
        b=500.0,
    )
    n = 10
    sig_batch = np.zeros((n, 6))
    deps_batch = np.zeros((n, 6))
    off_batch = np.ones(n)
    epc_batch = np.zeros((n, 3))
    dam_batch = np.zeros((n, 5))

    # Element 0: pure elastic
    deps_batch[0] = [0.0001, 0, 0, 0, 0, 0]
    # Element 1: cracking dir 1
    deps_batch[1] = [0.003, 0, 0, 0, 0, 0]
    # Element 2: cracking dir 2
    deps_batch[2] = [0, 0.004, 0, 0, 0, 0]
    # Element 3: cracking dir 3
    deps_batch[3] = [0, 0, 0.008, 0, 0, 0]
    # Element 4: open crack under compression
    epc_batch[4, 0] = 0.002
    deps_batch[4] = [-0.001, 0, 0, 0, 0, 0]
    # Element 5: shear yielding
    sig_batch[5, 3] = 90.0
    deps_batch[5, 3] = 0.002
    # Element 6: multiaxial yielding
    sig_batch[6] = [120.0, 40.0, 10.0, 50.0, 10.0, 10.0]
    deps_batch[6] = [0.002, 0.001, -0.001, 0.003, 0.001, 0.001]
    # Element 7: degraded element
    off_batch[7] = 0.99
    deps_batch[7] = [0.0001, 0, 0, 0, 0, 0]
    # Element 8: failed element
    off_batch[8] = 0.0
    deps_batch[8] = [0.0001, 0, 0, 0, 0, 0]
    # Element 9: static quiescent
    deps_batch[9] = [0, 0, 0, 0, 0, 0]

    extra_batch = {
        "off14": off_batch,
        "epc14": epc_batch,
        "dam14": dam_batch,
    }

    d_batch = consistent_solid_tangent(mat, sig_batch, deps=deps_batch, extra=extra_batch)
    assert d_batch.shape == (10, 6, 6)

    for i in range(n):
        ex_i = {
            "off14": np.array([off_batch[i]]),
            "epc14": epc_batch[i : i + 1],
            "dam14": dam_batch[i : i + 1],
        }
        d_single = consistent_solid_tangent(mat, sig_batch[i], deps=deps_batch[i], extra=ex_i)
        np.testing.assert_allclose(d_batch[i], d_single, atol=1e-12)


def test_tangent_zero_strain():
    """Handles zero or tiny strain increments stably without blowup or NaN."""
    mat = _make_orthotropic_material()
    sig0 = np.array([20.0, 10.0, -5.0, 8.0, 0.0, 0.0])

    # 1. deps=None
    t_none = consistent_solid_tangent(mat, sig0, deps=None)
    assert t_none.shape == (6, 6)
    assert np.all(np.isfinite(t_none))

    # 2. deps = zeros(6)
    t_zero = consistent_solid_tangent(mat, sig0, deps=np.zeros(6))
    assert t_zero.shape == (6, 6)
    assert np.all(np.isfinite(t_zero))
    assert np.array_equal(t_none, t_zero)

    # 3. tiny strain increment: 1e-16
    t_tiny = consistent_solid_tangent(mat, sig0, deps=np.full(6, 1.0e-16))
    assert t_tiny.shape == (6, 6)
    assert np.all(np.isfinite(t_tiny))
    np.testing.assert_allclose(t_tiny, t_zero, atol=1e-10)


def test_tangent_symmetric_option():
    """symmetric=True returns 0.5 * (C + C.T), while symmetric=False retains unsymmetric flow/damage."""
    mat = _make_orthotropic_material(sigt1=100.0, delta=0.1)
    sig0 = np.zeros(6)
    deps0 = np.array([0.003, 0.0, 0.0, 0.0, 0.0, 0.0])

    # Active cracking generates non-symmetric tangent (due to Poisson coupling without reciprocity)
    c_asym = consistent_solid_tangent(mat, sig0, deps=deps0, symmetric=False)
    assert np.max(np.abs(c_asym - c_asym.T)) > 1.0

    # With symmetric=True, tangent is strictly symmetric
    c_sym = consistent_solid_tangent(mat, sig0, deps=deps0, symmetric=True)
    assert np.max(np.abs(c_sym - c_sym.T)) == pytest.approx(0.0, abs=1e-12)
    np.testing.assert_allclose(c_sym, 0.5 * (c_asym + c_asym.T), atol=1e-12)


# ============================================================================
# Tensile Cracking Regimes & Poisson Relief
# ============================================================================

class TestLaw14TensileDamageDetails:
    """Detailed validation of directional cracking, Poisson relief, and unilateral behavior."""

    def test_active_cracking_dir2_and_dir3(self):
        """Tensile cracking in directions 2 and 3 zeroes D22 and D33 respectively."""
        mat = _make_orthotropic_material(sigt2=90.0, sigt3=70.0, delta=0.06)

        # Direction 2 cracking
        c2 = consistent_solid_tangent(mat, np.zeros(6), deps=np.array([0, 0.004, 0, 0, 0, 0]))
        assert c2[1, 1] == pytest.approx(0.0, abs=1e-6)

        # Direction 3 cracking
        c3 = consistent_solid_tangent(mat, np.zeros(6), deps=np.array([0, 0, 0.006, 0, 0, 0]))
        assert c3[2, 2] == pytest.approx(0.0, abs=1e-6)

    def test_active_cracking_poisson_relief(self):
        """Pre-damaged cracking demonstrates Poisson coupling scaled by (1 - dam)."""
        mat = _make_orthotropic_material(sigt1=120.0, delta=0.08)
        sig0 = np.zeros(6)
        deps0 = np.array([0.003, 0.0, 0.0, 0.0, 0.0, 0.0])
        dam_val = 0.15
        extra = {"dam14": np.array([[dam_val, 0.0, 0.0, 0.0, 0.0]])}

        c_alg = consistent_solid_tangent(mat, sig0, deps=deps0, extra=extra)
        c_num = _compute_numerical_tangent(mat, sig0, deps0, extra=extra)

        d12 = mat.params["D12"]
        d13 = mat.params["D13"]
        expected_d21 = d12 * (1.0 - dam_val)
        expected_d31 = d13 * (1.0 - dam_val)

        assert c_alg[0, 0] == pytest.approx(0.0, abs=1e-6)
        assert c_alg[1, 0] == pytest.approx(expected_d21, rel=1e-5)
        assert c_alg[2, 0] == pytest.approx(expected_d31, rel=1e-5)
        np.testing.assert_allclose(c_alg, c_num, atol=1e-4, rtol=1e-4)

    def test_crack_opened_compression_unilateral(self):
        """When crack in direction 1 is open (epc1 > 0), compressive increment has D11 = 0."""
        mat = _make_orthotropic_material(sigt1=100.0, delta=0.1)
        extra: Dict[str, Any] = {}
        sig = np.zeros(6)

        # Step 1: Open crack
        deps_tens = np.array([0.003, 0.0, 0.0, 0.0, 0.0, 0.0])
        s1, _, _ = solid_update(mat, sig, deps_tens, extra=extra)
        assert extra["epc14"][0, 0] > 0.0

        # Step 2: Compression with open crack
        deps_comp = np.array([-0.001, 0.0, 0.0, 0.0, 0.0, 0.0])
        c_alg = consistent_solid_tangent(mat, s1, deps=deps_comp, extra=extra)
        c_num = _compute_numerical_tangent(mat, s1, deps_comp, extra=extra)

        assert c_alg[0, 0] == pytest.approx(0.0, abs=1e-6)
        assert c_num[0, 0] == pytest.approx(0.0, abs=1e-6)
        np.testing.assert_allclose(c_alg, c_num, atol=1e-4, rtol=1e-4)

    def test_crack_closure_stiffness_restoration(self):
        """Crack re-closure (epc -> 0) restores full normal stiffness D11."""
        mat = _make_orthotropic_material(sigt1=100.0, delta=0.1)
        extra: Dict[str, Any] = {}

        # 1. Open crack
        s1, _, _ = solid_update(mat, np.zeros(6), np.array([0.002, 0.0, 0.0, 0.0, 0.0, 0.0]), extra=extra)
        # 2. Re-close crack
        s2, _, _ = solid_update(mat, s1, np.array([-0.002, 0.0, 0.0, 0.0, 0.0, 0.0]), extra=extra)
        assert extra["epc14"][0, 0] == 0.0

        # 3. Further compression has compressive stiffness
        deps_more = np.array([-0.0001, 0.0, 0.0, 0.0, 0.0, 0.0])
        c_alg = consistent_solid_tangent(mat, s2, deps=deps_more, extra=extra)
        assert c_alg[0, 0] == pytest.approx(mat.params["D11"], rel=1e-5)


# ============================================================================
# Element Degradation & Failure Regimes (off < 1.0, off == 0.0)
# ============================================================================

class TestLaw14DegradationRegimes:
    """Validate element degradation scaling by off and zero stiffness on failure."""

    def test_degraded_state_off_0792(self):
        """When off = 0.99, off is decayed by 0.8 to 0.792, scaling tangent consistently."""
        mat = _make_orthotropic_material()
        extra_deg = {"off14": np.array([0.99])}
        sig0 = np.zeros(6)
        deps0 = np.array([0.0001, 0.0, 0.0, 0.0, 0.0, 0.0])

        c_alg = consistent_solid_tangent(mat, sig0, deps=deps0, extra=extra_deg)
        c_num = _compute_numerical_tangent(mat, sig0, deps0, extra=extra_deg)

        expected_d11 = 0.792 * mat.params["D11"]
        assert c_alg[0, 0] == pytest.approx(expected_d11, rel=1e-5)
        np.testing.assert_allclose(c_alg, c_num, atol=1e-4, rtol=1e-4)

    def test_failed_state_off_zero(self):
        """When off = 0.0, element is deleted and tangent is identically zero."""
        mat = _make_orthotropic_material()
        extra_dead = {"off14": np.array([0.0])}
        sig0 = np.zeros(6)
        deps0 = np.array([0.0001, 0.0001, 0.0, 0.0, 0.0, 0.0])

        c_alg = consistent_solid_tangent(mat, sig0, deps=deps0, extra=extra_dead)
        c_num = _compute_numerical_tangent(mat, sig0, deps0, extra=extra_dead)

        np.testing.assert_allclose(c_alg, 0.0, atol=1e-12)
        np.testing.assert_allclose(c_num, 0.0, atol=1e-12)

    def test_failed_state_sub_threshold(self):
        """When off < 0.1 (e.g. off = 0.05), solver zeroes off, yielding zero tangent."""
        mat = _make_orthotropic_material()
        extra_sub = {"off14": np.array([0.05])}
        sig0 = np.zeros(6)
        deps0 = np.array([0.0001, 0.0, 0.0, 0.0, 0.0, 0.0])

        c_alg = consistent_solid_tangent(mat, sig0, deps=deps0, extra=extra_sub)
        np.testing.assert_allclose(c_alg, 0.0, atol=1e-12)

    def test_progressive_degradation_steps(self):
        """Successive steps with off < 1.0 degrade stiffness by 0.8 repeatedly."""
        mat = _make_orthotropic_material()

        # Step 1: off = 0.8 -> updated to 0.8 * 0.8 = 0.64
        extra1 = {"off14": np.array([0.8])}
        c1 = consistent_solid_tangent(mat, np.zeros(6), extra=extra1)
        assert c1[0, 0] == pytest.approx(0.64 * mat.params["D11"], rel=1e-5)

        # Step 2: off = 0.64 -> updated to 0.64 * 0.8 = 0.512
        extra2 = {"off14": np.array([0.64])}
        c2 = consistent_solid_tangent(mat, np.zeros(6), extra=extra2)
        assert c2[0, 0] == pytest.approx(0.512 * mat.params["D11"], rel=1e-5)


# ============================================================================
# Coordinate Transformation & Operational Shapes
# ============================================================================

class TestLaw14TransformationsAndShapes:
    """Verify coordinate transformations, rate effects, and input shapes."""

    def test_rotated_axes_tangent(self):
        """Coordinate frame rotation via axes in extra correctly transforms tangent to global coordinates."""
        mat = _make_orthotropic_material()
        theta = np.pi / 4.0
        c, s = math.cos(theta), math.sin(theta)
        rot = np.array(
            [
                [c, s, 0.0],
                [-s, c, 0.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=float,
        )

        extra = {"axes": rot}
        sig0 = np.array([40.0, 20.0, 10.0, 5.0, 0.0, 0.0])
        deps0 = np.array([0.0001, -0.00005, 0.00003, 0.0001, 0.0, 0.0])
        h = 1.0e-7

        c_alg = consistent_solid_tangent(mat, sig0, deps=deps0, extra=extra, h=h)
        c_num = _compute_numerical_tangent(mat, sig0, deps0, extra=extra, h=h)

        diff = np.max(np.abs(c_alg - c_num))
        assert diff < 1e-4, f"Rotated tangent difference {diff:.3e} exceeds 1e-4"

    def test_rotated_triad_tangent(self):
        """Coordinate triad vectors rx, ry, rz, sx, sy, sz correctly transform tangent."""
        mat = _make_orthotropic_material()
        theta = np.pi / 6.0
        c, s = math.cos(theta), math.sin(theta)
        extra = {
            "rx": np.array([c]),
            "ry": np.array([s]),
            "rz": np.array([0.0]),
            "sx": np.array([-s]),
            "sy": np.array([c]),
            "sz": np.array([0.0]),
        }
        sig0 = np.array([30.0, 15.0, 10.0, 0.0, 0.0, 0.0])
        deps0 = np.array([0.0001, 0.0001, 0.0, 0.0, 0.0, 0.0])

        c_alg = consistent_solid_tangent(mat, sig0, deps=deps0, extra=extra)
        c_num = _compute_numerical_tangent(mat, sig0, deps0, extra=extra)
        np.testing.assert_allclose(c_alg, c_num, atol=1e-4, rtol=1e-4)

    def test_input_shapes_1d_2d_empty(self):
        """1D returns (6, 6), 2D returns (n, 6, 6), empty returns (0, 6, 6)."""
        mat = _make_orthotropic_material()

        # 1D
        t_1d = consistent_solid_tangent(mat, np.zeros(6))
        assert t_1d.shape == (6, 6)

        # 2D (1, 6)
        t_2d = consistent_solid_tangent(mat, np.zeros((1, 6)))
        assert t_2d.shape == (1, 6, 6)

        # 2D (3, 6)
        t_3d = consistent_solid_tangent(mat, np.zeros((3, 6)))
        assert t_3d.shape == (3, 6, 6)

        # Empty (0, 6)
        t_empty = consistent_solid_tangent(mat, np.empty((0, 6)))
        assert t_empty.shape == (0, 6, 6)

    def test_dynamic_strain_rate_sensitivity(self):
        """With dt > 0 and rate sensitivity c > 0, dynamic rate enhancement is active."""
        mat = _make_orthotropic_material(
            sigyt12=100.0,
            sigyc12=100.0,
            c_rate=0.1,
            eps0=1.0,
            icc=1,
        )
        sig0 = np.array([0.0, 0.0, 0.0, 110.0, 0.0, 0.0])
        dt = 1.0e-6
        deps0 = np.array([0.0, 0.0, 0.0, 0.001, 0.0, 0.0])

        c_dyn = consistent_solid_tangent(mat, sig0, deps=deps0, dt=dt)
        c_num = _compute_numerical_tangent(mat, sig0, deps0, dt=dt)
        np.testing.assert_allclose(c_dyn, c_num, atol=1e-4)

    def test_boundary_perturbation_stability(self):
        """Perturbation directly across cracking or yield boundary produces finite values without NaN/Inf."""
        mat = _make_orthotropic_material(sigt1=100.0, delta=0.1)
        # Stress right at cracking boundary: t1 = sigt1
        sig_crit = np.array([100.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        c_crit = consistent_solid_tangent(mat, sig_crit, deps=np.zeros(6), h=1.0e-7)

        assert np.all(np.isfinite(c_crit)), "Tangent across cracking boundary must remain strictly finite"
