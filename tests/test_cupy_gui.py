"""
Tests for CuPy/CUDA GUI integration and GPUArrayManager (Work Stream 15).

Verifies:
- _BACKENDS includes "cupy" in pyradioss.gui.app
- probe_cupy handles absent CuPy gracefully and identifies devices when present
- JobRunner propagates the cupy backend flag and PYRADIOSS_BACKEND env var
- GPUArrayManager CPU-fallback mode acts as a transparent passthrough
- GPUArrayManager round-trip upload -> download preserves array data
- GPUArrayManager simulated GPU mode tracks device arrays, synchronization, and cleanup
"""

from __future__ import annotations

import os
from unittest.mock import MagicMock
import numpy as np
import pytest

from pyradioss.gui.app import _BACKENDS, probe_cupy, probe_gpu
from pyradioss.gui.runner import JobRunner
from pyradioss.accel.gpu_transfer import GPUArrayManager
from pyradioss.accel import select_backend


# ---------------------------------------------------------------------------
# Step 2 tests: Backend dropdown and GPU probe
# ---------------------------------------------------------------------------

def test_backends_tuple_includes_cupy():
    """_BACKENDS must include 'cupy' and contain auto, numpy, numba, cupy."""
    assert "cupy" in _BACKENDS
    assert _BACKENDS == ("auto", "numpy", "numba", "cupy")


def test_gpu_probe_absent_gracefully():
    """When CuPy is absent, probe_cupy() returns (False, '(not available)')."""
    available, info = probe_cupy()
    # In the current environment, cupy is not installed
    assert available is False
    assert "(not available)" in info
    # probe_gpu is an alias and returns identical results
    assert probe_gpu() == (available, info)


def test_gpu_probe_with_mock_cupy():
    """When CuPy is present with a CUDA device, probe_cupy reports device name."""
    mock_cupy = MagicMock()
    mock_cupy.cuda.is_available.return_value = True
    mock_cupy.cuda.runtime.getDeviceCount.return_value = 1
    mock_cupy.cuda.runtime.getDeviceProperties.return_value = {
        "name": b"NVIDIA GeForce RTX 4090"
    }

    available, name = probe_cupy(mock_cupy)
    assert available is True
    assert name == "NVIDIA GeForce RTX 4090"


def test_gpu_probe_mock_no_device():
    """When CuPy is installed but no CUDA device is present."""
    mock_cupy = MagicMock()
    mock_cupy.cuda.is_available.return_value = False

    available, info = probe_cupy(mock_cupy)
    assert available is False
    assert "(not available)" in info


def test_gpu_probe_mock_device_count_zero():
    """When CuPy is installed but getDeviceCount returns 0."""
    mock_cupy = MagicMock()
    mock_cupy.cuda.is_available.return_value = True
    mock_cupy.cuda.runtime.getDeviceCount.return_value = 0

    available, info = probe_cupy(mock_cupy)
    assert available is False
    assert "(not available)" in info


# ---------------------------------------------------------------------------
# Step 3 tests: JobRunner passing backend flag and env var
# ---------------------------------------------------------------------------

def test_jobrunner_engine_cmd_cupy():
    """JobRunner passes '-backend cupy' to the engine command line."""
    runner = JobRunner("test_0000.rad", backend="cupy")
    cmd = runner._engine_cmd()
    assert "-backend" in cmd
    idx = cmd.index("-backend")
    assert cmd[idx + 1] == "cupy"


def test_jobrunner_package_env_backend():
    """JobRunner sets PYRADIOSS_BACKEND=cupy in subprocess environment."""
    runner = JobRunner("test_0000.rad", backend="cupy")
    env = runner._package_env(backend=runner.backend)
    assert env.get("PYRADIOSS_BACKEND") == "cupy"

    # Static call with no arguments remains backward-compatible
    static_env = JobRunner._package_env()
    assert "PYTHONPATH" in static_env


def test_select_backend_cupy_fallback():
    """select_backend('cupy') falls back gracefully to numpy when CuPy is not installed."""
    with pytest.warns(UserWarning, match="cupy backend requested"):
        chosen = select_backend("cupy")
    assert chosen == "numpy"


# ---------------------------------------------------------------------------
# Step 4 & 5 tests: GPUArrayManager CPU-fallback and Roundtrip
# ---------------------------------------------------------------------------

def test_gpu_array_manager_cpu_fallback():
    """GPUArrayManager in CPU mode acts as a no-op passthrough."""
    mgr = GPUArrayManager(backend="numpy")
    assert mgr.is_gpu is False
    assert mgr.backend == "numpy"

    # sync is a no-op
    mgr.sync()

    # upload stores array
    data = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float64)
    res = mgr.upload("nodpos", data)
    np.testing.assert_array_equal(res, data)

    # is_on_gpu is always False in CPU mode
    assert mgr.is_on_gpu("nodpos") is False
    assert "nodpos" in mgr
    assert len(mgr) == 1

    # download returns the array
    downloaded = mgr.download("nodpos")
    np.testing.assert_array_equal(downloaded, data)

    # download non-existent key raises KeyError
    with pytest.raises(KeyError, match="not registered"):
        mgr.download("nonexistent")

    # cleanup clears arrays
    mgr.cleanup()
    assert "nodpos" not in mgr
    assert len(mgr) == 0


def test_gpu_array_manager_absent_cupy_fallback():
    """GPUArrayManager with backend='cupy' falls back to CPU when cupy is absent."""
    mgr = GPUArrayManager(backend="cupy")
    # In test environment, cupy is not installed
    assert mgr.is_gpu is False

    arr = np.linspace(0.0, 10.0, 50)
    mgr.upload("stress", arr)
    assert not mgr.is_on_gpu("stress")
    np.testing.assert_array_equal(mgr.download("stress"), arr)


def test_array_roundtrip_preserves_data():
    """Upload -> download roundtrip preserves 1D, 2D, and 3D array data bitwise."""
    mgr = GPUArrayManager(backend="numpy")

    # 1D array
    a1 = np.array([1.5, -2.75, 3.125, 0.0], dtype=np.float64)
    mgr.upload("1d", a1)
    np.testing.assert_array_equal(mgr.download("1d"), a1)

    # 2D array
    a2 = np.arange(24, dtype=np.float32).reshape((4, 6))
    mgr.upload("2d", a2)
    np.testing.assert_array_equal(mgr.download("2d"), a2)

    # 3D array
    a3 = np.random.default_rng(42).standard_normal((3, 4, 5))
    mgr.upload("3d", a3)
    np.testing.assert_array_equal(mgr.download("3d"), a3)

    # Integer indexing array
    a_int = np.array([10, 20, 30, 40, 50], dtype=np.int64)
    mgr.upload("idx", a_int)
    np.testing.assert_array_equal(mgr.download("idx"), a_int)


def test_gpu_array_manager_mock_cupy_active():
    """Test GPUArrayManager with an injected mock CuPy module."""
    mock_cupy = MagicMock()
    sync_called = []
    freed_mempool = []
    freed_pinned = []

    # Mock device array
    class MockDeviceArray:
        def __init__(self, arr):
            self.data = np.copy(arr)

        def get(self):
            return np.copy(self.data)

    mock_cupy.asarray = lambda a: MockDeviceArray(a)
    mock_cupy.asnumpy = lambda g: g.get()
    mock_cupy.cuda.Stream.null.synchronize = lambda: sync_called.append(True)

    mock_pool = MagicMock()
    mock_pool.free_all_blocks = lambda: freed_mempool.append(True)
    mock_cupy.get_default_memory_pool.return_value = mock_pool

    mock_pinned = MagicMock()
    mock_pinned.free_all_blocks = lambda: freed_pinned.append(True)
    mock_cupy.get_default_pinned_memory_pool.return_value = mock_pinned

    mgr = GPUArrayManager(cupy_module=mock_cupy)
    assert mgr.is_gpu is True

    # Upload
    orig = np.array([10.0, 20.0, 30.0], dtype=np.float64)
    gpu_arr = mgr.upload("vel", orig)
    assert isinstance(gpu_arr, MockDeviceArray)
    assert mgr.is_on_gpu("vel") is True

    # Sync
    mgr.sync()
    assert len(sync_called) == 1

    # Download
    downloaded = mgr.download("vel")
    np.testing.assert_array_equal(downloaded, orig)

    # get() retrieves GPU device representation
    assert mgr.get("vel") is gpu_arr

    # Cleanup
    mgr.cleanup()
    assert mgr.is_on_gpu("vel") is False
    assert len(freed_mempool) == 1
    assert len(freed_pinned) == 1


# ---------------------------------------------------------------------------
# Optional Tkinter GUI integration test
# ---------------------------------------------------------------------------

def test_gui_widget_cupy_option():
    """Verify GUI sets up the cupy backend combobox and GPU status label."""
    import tkinter as tk
    try:
        root = tk.Tk()
    except tk.TclError:
        pytest.skip("no Tk display available")
    try:
        root.withdraw()
        from pyradioss.gui.app import PyradiossGUI
        gui = PyradiossGUI(root)
        assert hasattr(gui, "gpu_label")
        assert hasattr(gui, "gpu_info_var")
        assert "(not available)" in gui.gpu_info_var.get()
        # Verify cupy is a valid option in backend_var
        gui.backend_var.set("cupy")
        assert gui.backend_var.get() == "cupy"
    finally:
        root.destroy()
