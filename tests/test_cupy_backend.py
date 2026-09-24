"""
Unit and integration tests for pyradioss CuPy/CUDA compute backend (WS14).
"""

from __future__ import annotations

import inspect
import sys
from unittest.mock import MagicMock

import numpy as np
import pytest

from pyradioss import accel
from pyradioss.accel import gpu_kernels
from pyradioss.accel import jit_kernels
from pyradioss.accel.gpu_kernels import HAS_CUPY, to_cpu, to_gpu


@pytest.fixture(autouse=True)
def _isolate_backend(monkeypatch):
    """Start every test from pristine unresolved state and restore numpy on teardown."""
    monkeypatch.delenv("PYRADIOSS_BACKEND", raising=False)
    monkeypatch.delenv("PYRADIOSS_BACKEND_AUTO_MIN_ELEMENTS", raising=False)
    accel._state.update(name=None, mod=None, forced=False)
    yield
    accel._state.update(name=None, mod=None, forced=False)
    accel.select_backend("numpy")


class _Grp:
    def __init__(self, n: int):
        self.n = n


class _MockModel:
    def __init__(self, n_elements: int):
        self._groups = [("g0", _Grp(n_elements))]

    def element_groups(self):
        return iter(self._groups)


# ============================================================================
# Backend selection and fallback tests
# ============================================================================

def test_cupy_backend_fallback_when_not_installed(monkeypatch):
    """When CuPy/CUDA is unavailable, requesting 'cupy' must warn and fall back to NumPy."""
    def _fail_cupy():
        raise ImportError("simulated: CuPy not installed")

    monkeypatch.setattr(accel, "_load_cupy_module", _fail_cupy)

    with pytest.warns(UserWarning, match="cupy backend requested"):
        name = accel.select_backend("cupy")

    assert name == "numpy"
    assert accel.backend_name() == "numpy"
    assert accel.get("hexa_pre") is None


def test_cupy_backend_pinned_with_mock(monkeypatch):
    """When CuPy is available, pinning 'cupy' activates the GPU kernel module."""
    monkeypatch.setattr(accel, "_load_cupy_module", lambda: gpu_kernels)

    name = accel.select_backend("cupy")
    assert name == "cupy"
    assert accel.backend_name() == "cupy"
    assert accel.get("hexa_pre") is gpu_kernels.hexa_pre
    assert accel.get("shell_pre") is gpu_kernels.shell_pre
    assert accel.get("tetra10_pre") is gpu_kernels.tetra10_pre


def test_auto_selector_order_cupy_first(monkeypatch):
    """Auto selector must prioritize CuPy > Numba > NumPy for models >= threshold."""
    thr = accel._AUTO_MIN_ELEMENTS
    model = _MockModel(thr + 10)

    # Case A: Both CuPy and Numba available -> CuPy must win
    monkeypatch.setattr(accel, "_load_cupy_module", lambda: gpu_kernels)
    monkeypatch.setattr(accel, "_load_numba_module", lambda: jit_kernels)

    assert accel.auto_select_backend(model) == "cupy"
    assert accel.get("hexa_pre") is gpu_kernels.hexa_pre

    # Reset state
    accel._state.update(name=None, mod=None, forced=False)

    # Case B: CuPy unavailable, Numba available -> Numba must win
    def _fail_cupy():
        raise ImportError("CuPy not installed")

    monkeypatch.setattr(accel, "_load_cupy_module", _fail_cupy)
    monkeypatch.setattr(accel, "_load_numba_module", lambda: jit_kernels)

    assert accel.auto_select_backend(model) == "numba"
    assert accel.get("hexa_pre") is jit_kernels.hexa_pre

    # Reset state
    accel._state.update(name=None, mod=None, forced=False)

    # Case C: Both unavailable -> NumPy fallback
    def _fail_numba():
        raise ImportError("Numba not installed")

    monkeypatch.setattr(accel, "_load_numba_module", _fail_numba)

    assert accel.auto_select_backend(model) == "numpy"
    assert accel.get("hexa_pre") is None


def test_auto_selector_subthreshold_and_implicit(monkeypatch):
    """Small models and implicit runs stay on NumPy under auto, even if CuPy is available."""
    thr = accel._AUTO_MIN_ELEMENTS
    small_model = _MockModel(thr - 1)
    large_model = _MockModel(thr + 100)

    monkeypatch.setattr(accel, "_load_cupy_module", lambda: gpu_kernels)

    # Subthreshold stays numpy
    assert accel.auto_select_backend(small_model) == "numpy"

    accel._state.update(name=None, mod=None, forced=False)

    # Implicit run stays numpy
    assert accel.auto_select_backend(large_model, explicit=False) == "numpy"


# ============================================================================
# Array transfer helper tests
# ============================================================================

def test_transfer_helpers_none():
    assert to_gpu(None) is None
    assert to_cpu(None) is None


def test_to_cpu_with_numpy_array():
    arr = np.array([1.0, 2.0, 3.0])
    res = to_cpu(arr)
    assert isinstance(res, np.ndarray)
    np.testing.assert_array_equal(res, arr)


def test_to_gpu_raises_when_no_cupy(monkeypatch):
    monkeypatch.setattr(gpu_kernels, "cp", None)
    arr = np.array([1.0, 2.0, 3.0])
    with pytest.raises(RuntimeError, match="CuPy/CUDA is not available"):
        to_gpu(arr)


def test_transfer_helpers_mocked_cupy(monkeypatch):
    """Test to_gpu and to_cpu using a mocked CuPy module."""
    class MockCpArray:
        def __init__(self, data):
            self.data = np.asarray(data)

        def get(self):
            return self.data.copy()

    mock_cp = MagicMock()
    mock_cp.ndarray = MockCpArray
    mock_cp.asarray = lambda x: MockCpArray(x)

    monkeypatch.setattr(gpu_kernels, "cp", mock_cp)

    cpu_data = np.array([1.0, 2.0, 3.0])
    gpu_data = to_gpu(cpu_data)
    assert isinstance(gpu_data, MockCpArray)

    # Calling to_gpu on already GPU array returns itself
    assert to_gpu(gpu_data) is gpu_data

    # Calling to_cpu retrieves CPU NumPy array
    roundtrip = to_cpu(gpu_data)
    assert isinstance(roundtrip, np.ndarray)
    np.testing.assert_array_equal(roundtrip, cpu_data)


# ============================================================================
# Signature parity tests
# ============================================================================

def test_all_kernel_signatures_match_jit_kernels():
    """Every callable kernel in gpu_kernels must match jit_kernels signature exactly."""
    for name in dir(jit_kernels):
        if name.startswith("__") or name in ("njit", "prange"):
            continue
        attr_jk = getattr(jit_kernels, name)
        if not callable(attr_jk):
            continue
        assert hasattr(gpu_kernels, name), f"gpu_kernels missing mirror for {name}"
        attr_gk = getattr(gpu_kernels, name)
        sig_jk = inspect.signature(attr_jk)
        sig_gk = inspect.signature(attr_gk)
        assert sig_jk == sig_gk, f"Signature mismatch for {name}: {sig_jk} != {sig_gk}"


# ============================================================================
# Mathematical parity tests (comparing GPU vectorized logic with JIT kernels)
# ============================================================================

def test_hexa_pre_parity():
    n = 4
    np.random.seed(101)
    xe = np.zeros((n, 8, 3))
    for i in range(n):
        xe[i, 0] = [0, 0, 0]
        xe[i, 1] = [1, 0, 0]
        xe[i, 2] = [1, 1, 0]
        xe[i, 3] = [0, 1, 0]
        xe[i, 4] = [0, 0, 1]
        xe[i, 5] = [1, 0, 1]
        xe[i, 6] = [1, 1, 1]
        xe[i, 7] = [0, 1, 1]
        xe[i] += np.random.randn(8, 3) * 0.05

    ve = np.random.randn(n, 8, 3) * 5.0
    sig_ref = np.random.randn(n, 6) * 100.0
    sig_gpu = sig_ref.copy()
    dt = 1e-4
    off = np.array([1.0, 1.0, 0.0, 1.0])
    lc_scale = np.ones(n)

    r_ref = jit_kernels.hexa_pre(xe, ve, sig_ref, dt, off, lc_scale)
    r_gpu = gpu_kernels.hexa_pre(xe, ve, sig_gpu, dt, off, lc_scale)

    for a, b in zip(r_ref, r_gpu):
        assert np.allclose(a, b, atol=1e-12)
    assert np.allclose(sig_ref, sig_gpu, atol=1e-12)


def test_hexa_post_parity():
    n = 4
    np.random.seed(102)
    xe = np.random.randn(n, 8, 3)
    ve = np.random.randn(n, 8, 3) * 5.0
    dndx = np.random.randn(n, 8, 3) * 0.1
    vol = np.ones(n) * 1.5
    lc = np.ones(n) * 1.0
    rho = np.ones(n) * 7.8e-6
    trD = np.array([-1.5, 0.5, -2.0, 0.0])
    deps = np.random.randn(n, 6) * 1e-3
    sig = np.random.randn(n, 6) * 100.0
    sig_old = np.random.randn(n, 6) * 100.0
    qa = np.ones(n) * 1.1
    qb = np.ones(n) * 0.05
    c = np.ones(n) * 5000.0
    hcoef = np.ones(n) * 0.1
    alive = np.array([True, True, False, True])
    qvw_pend = np.random.randn(n) * 0.01
    dt = 1e-5
    dtfac = np.ones(n) * 0.9

    r_ref = jit_kernels.hexa_post(
        xe, ve, dndx, vol, lc, rho, trD, deps, sig, sig_old,
        qa, qb, c, hcoef, alive, qvw_pend, dt, dtfac
    )
    r_gpu = gpu_kernels.hexa_post(
        xe, ve, dndx, vol, lc, rho, trD, deps, sig, sig_old,
        qa, qb, c, hcoef, alive, qvw_pend, dt, dtfac
    )

    for a, b in zip(r_ref, r_gpu):
        assert np.allclose(a, b, atol=1e-12)


def test_shell_pre_parity():
    n = 4
    np.random.seed(103)
    xe = np.zeros((n, 4, 3))
    for i in range(n):
        xe[i, 0] = [0, 0, 0]
        xe[i, 1] = [1, 0, 0]
        xe[i, 2] = [1, 1, 0]
        xe[i, 3] = [0, 1, 0]
        xe[i] += np.random.randn(4, 3) * 0.02

    ve = np.random.randn(n, 4, 3) * 5.0
    vre = np.random.randn(n, 4, 3) * 2.0
    off = np.array([1.0, 1.0, 0.0, 1.0])

    r_ref = jit_kernels.shell_pre(xe, ve, vre, off)
    r_gpu = gpu_kernels.shell_pre(xe, ve, vre, off)

    for a, b in zip(r_ref, r_gpu):
        assert np.allclose(a, b, atol=1e-12)


def test_shell_post_parity():
    n = 4
    np.random.seed(104)
    E = np.tile(np.eye(3), (n, 1, 1))
    area = np.ones(n) * 1.5
    B1 = np.random.randn(n, 4) * 0.5
    B2 = np.random.randn(n, 4) * 0.5
    gam = np.random.randn(n, 4) * 0.5
    V = np.random.randn(n, 4, 5) * 2.0
    Nres = np.random.randn(n, 3) * 100.0
    Mres = np.random.randn(n, 3) * 50.0
    qres = np.random.randn(n, 2) * 10.0
    Q_ref = np.random.randn(n, 5) * 5.0
    Q_gpu = Q_ref.copy()
    k_m = np.ones(n) * 1000.0
    k_w = np.ones(n) * 500.0
    hqm = np.ones(n) * 0.1
    hqb = np.ones(n) * 0.1
    hqr = np.ones(n) * 0.05
    dt = 1e-5

    r_ref = jit_kernels.shell_post(
        E, area, B1, B2, gam, V, Nres, Mres, qres, Q_ref, k_m, k_w, hqm, hqb, hqr, dt
    )
    r_gpu = gpu_kernels.shell_post(
        E, area, B1, B2, gam, V, Nres, Mres, qres, Q_gpu, k_m, k_w, hqm, hqb, hqr, dt
    )

    for a, b in zip(r_ref, r_gpu):
        assert np.allclose(a, b, atol=1e-12)
    assert np.allclose(Q_ref, Q_gpu, atol=1e-12)


def test_t7_narrow_parity():
    np.random.seed(105)
    N = 15
    x = np.random.randn(N, 3)
    ni = np.array([0, 1, 2, 3])
    seg = np.array([
        [4, 5, 6, 7],
        [5, 6, 7, 8],
        [6, 7, 8, 9],
        [7, 8, 9, 10]
    ])

    d_ref, pt_ref, w_ref = jit_kernels.t7_narrow(x, ni, seg)
    d_gpu, pt_gpu, w_gpu = gpu_kernels.t7_narrow(x, ni, seg)

    assert np.allclose(d_ref, d_gpu, atol=1e-12)
    assert np.allclose(pt_ref, pt_gpu, atol=1e-12)
    assert np.allclose(w_ref, w_gpu, atol=1e-12)


def test_hexa_hgphys_parity():
    n = 4
    np.random.seed(106)
    xe = np.random.randn(n, 8, 3)
    ve = np.random.randn(n, 8, 3) * 5.0
    dndx = np.random.randn(n, 8, 3) * 0.1
    vol = np.ones(n) * 1.5
    c = np.ones(n) * 5000.0
    mask = np.array([True, False, True, False])
    mass = np.ones(n) * 1.0e-3
    vol0 = np.ones(n) * 1.5
    q_ref = np.random.randn(n, 4, 3) * 0.01
    q_gpu = q_ref.copy()
    dt = 1e-5

    r_ref = jit_kernels.hexa_hgphys(xe, ve, dndx, vol, c, mask, mass, vol0, q_ref, dt)
    r_gpu = gpu_kernels.hexa_hgphys(xe, ve, dndx, vol, c, mask, mass, vol0, q_gpu, dt)

    for a, b in zip(r_ref, r_gpu):
        assert np.allclose(a, b, atol=1e-12)
    assert np.allclose(q_ref, q_gpu, atol=1e-12)


def test_scatter3_parity():
    N = 20
    m = 15
    np.random.seed(107)
    idx = np.random.randint(0, N, m)
    values = np.random.randn(m, 3) * 10.0

    target_ref = np.zeros((N, 3))
    target_gpu = np.zeros((N, 3))

    jit_kernels.scatter3(target_ref, idx, values)
    gpu_kernels.scatter3(target_gpu, idx, values)

    assert np.allclose(target_ref, target_gpu, atol=1e-12)


def test_law70_numeric_leaves_parity():
    np.random.seed(108)
    xg = np.linspace(0.0, 1.0, 8)
    rates = np.array([0.0, 10.0, 100.0])
    Y = np.random.randn(8, 3) * 50.0
    x = np.random.uniform(-0.1, 1.1, 10)
    r = np.random.uniform(-5.0, 120.0, 10)

    assert np.allclose(
        jit_kernels.law70_tab2d(xg, rates, Y, x, r),
        gpu_kernels.law70_tab2d(xg, rates, Y, x, r),
        atol=1e-14
    )

    v = np.random.randn(10, 6)
    assert np.allclose(jit_kernels.law70_enorm(v), gpu_kernels.law70_enorm(v), atol=1e-14)
    assert np.allclose(jit_kernels.law70_snorm(v), gpu_kernels.law70_snorm(v), atol=1e-14)

    aa1 = np.ones(10) * 100.0
    aa2 = np.ones(10) * 50.0
    g = np.ones(10) * 25.0
    assert np.allclose(
        jit_kernels.law70_elastic_stress(aa1, aa2, g, v),
        gpu_kernels.law70_elastic_stress(aa1, aa2, g, v),
        atol=1e-14
    )


def test_tetra10_pre_and_post_parity():
    n = 2
    np.random.seed(109)
    xe = np.zeros((n, 10, 3))
    for i in range(n):
        xe[i, 0] = [0, 0, 0]
        xe[i, 1] = [1, 0, 0]
        xe[i, 2] = [0, 1, 0]
        xe[i, 3] = [0, 0, 1]
        xe[i, 4] = 0.5 * (xe[i, 0] + xe[i, 1])
        xe[i, 5] = 0.5 * (xe[i, 1] + xe[i, 2])
        xe[i, 6] = 0.5 * (xe[i, 2] + xe[i, 0])
        xe[i, 7] = 0.5 * (xe[i, 0] + xe[i, 3])
        xe[i, 8] = 0.5 * (xe[i, 1] + xe[i, 3])
        xe[i, 9] = 0.5 * (xe[i, 2] + xe[i, 3])
        xe[i] += np.random.randn(10, 3) * 0.01

    ve = np.random.randn(n, 10, 3) * 5.0
    sig_ref = np.random.randn(n, 4, 6) * 100.0
    sig_gpu = sig_ref.copy()
    dt = 1e-5
    off = np.array([1.0, 0.0])

    r_pre_ref = jit_kernels.tetra10_pre(xe, ve, sig_ref, dt, off)
    r_pre_gpu = gpu_kernels.tetra10_pre(xe, ve, sig_gpu, dt, off)

    for a, b in zip(r_pre_ref, r_pre_gpu):
        assert np.allclose(a, b, atol=1e-12)
    assert np.allclose(sig_ref, sig_gpu, atol=1e-12)

    dndx, vol, vol_tot, lc, deps, trD = r_pre_ref
    rho = np.ones(n) * 7.8e-6
    sig_old = np.random.randn(n, 4, 6) * 100.0
    qa = np.ones(n) * 1.1
    qb = np.ones(n) * 0.05
    c = np.ones(n) * 5000.0
    alive = np.array([True, False])
    qvw_pend = np.random.randn(n) * 0.01
    dtfac = np.ones(n) * 0.9

    r_post_ref = jit_kernels.tetra10_post(
        xe, dndx, vol, vol_tot, lc, rho, trD, deps, sig_ref, sig_old,
        qa, qb, c, alive, qvw_pend, dt, dtfac
    )
    r_post_gpu = gpu_kernels.tetra10_post(
        xe, dndx, vol, vol_tot, lc, rho, trD, deps, sig_ref, sig_old,
        qa, qb, c, alive, qvw_pend, dt, dtfac
    )

    for a, b in zip(r_post_ref, r_post_gpu):
        assert np.allclose(a, b, atol=1e-12)


# ============================================================================
# Real GPU execution tests (skipped when CuPy / CUDA hardware is not present)
# ============================================================================

@pytest.mark.skipif(not HAS_CUPY, reason="CuPy/CUDA is not installed or available on this system")
def test_real_gpu_execution():
    """Verify end-to-end execution on actual GPU when CuPy/CUDA is available."""
    import cupy as cp

    n = 2
    xe_cpu = np.zeros((n, 8, 3))
    for i in range(n):
        xe_cpu[i, 0] = [0, 0, 0]
        xe_cpu[i, 1] = [1, 0, 0]
        xe_cpu[i, 2] = [1, 1, 0]
        xe_cpu[i, 3] = [0, 1, 0]
        xe_cpu[i, 4] = [0, 0, 1]
        xe_cpu[i, 5] = [1, 0, 1]
        xe_cpu[i, 6] = [1, 1, 1]
        xe_cpu[i, 7] = [0, 1, 1]

    ve_cpu = np.zeros((n, 8, 3))
    sig_cpu = np.zeros((n, 6))

    xe_gpu = to_gpu(xe_cpu)
    ve_gpu = to_gpu(ve_cpu)
    sig_gpu = to_gpu(sig_cpu)
    off_gpu = to_gpu(np.ones(n))
    lc_scale_gpu = to_gpu(np.ones(n))

    dndx_g, vol_g, lc_g, deps_g, trD_g = gpu_kernels.hexa_pre(
        xe_gpu, ve_gpu, sig_gpu, 1e-4, off_gpu, lc_scale_gpu
    )

    assert isinstance(dndx_g, cp.ndarray)
    assert isinstance(vol_g, cp.ndarray)
    vol_cpu = to_cpu(vol_g)
    assert np.allclose(vol_cpu, 1.0)
