"""
CPU <-> GPU array transfer management for pyradioss accelerated execution.

# No direct Fortran counterpart: memory orchestration for Python/CuPy compute backend
# (analogous to OpenRadioss memory handling in engine/source/common/mem*.F).

Manages array lifecycle between host (CPU, NumPy) and device (GPU, CuPy)
for explicit simulation cycles:
- Uploads NumPy arrays to CuPy device arrays
- Downloads CuPy device arrays back to NumPy
- Caches arrays on device to avoid repeated PCI-e transfers within an iteration
- Synchronizes GPU streams
- Falls back to a transparent CPU passthrough when CuPy or CUDA is not available
"""

from __future__ import annotations

import os
from typing import Any, Dict, Optional
import numpy as np


class GPUArrayManager:
    """Manages CPU<->GPU array transfers for explicit simulation cycles.

    When backend is 'cupy' and CuPy + CUDA are available, arrays are uploaded
    to device memory and cached across operations. When backend is not 'cupy'
    (e.g., 'numpy' or 'numba') or CuPy/CUDA is unavailable, acts as a
    transparent CPU passthrough where arrays are kept in host memory.
    """

    def __init__(
        self,
        backend: Optional[str] = None,
        cupy_module: Any = None,
    ) -> None:
        """Initialize the array manager.

        Args:
            backend: Target backend name ('cupy', 'numpy', 'numba', 'auto').
                     Defaults to the PYRADIOSS_BACKEND environment variable or 'numpy'.
            cupy_module: Optional CuPy module injection (useful for testing/mocking).
        """
        if backend is None:
            backend = os.environ.get("PYRADIOSS_BACKEND", "numpy")
        self.backend = (backend or "numpy").strip().lower()

        self._cupy = None
        self._is_gpu = False

        if cupy_module is not None:
            self._cupy = cupy_module
            self._is_gpu = True
            self.backend = "cupy"
        elif self.backend == "cupy":
            try:
                import cupy
                if hasattr(cupy, "cuda") and hasattr(cupy.cuda, "is_available"):
                    if cupy.cuda.is_available():
                        self._cupy = cupy
                        self._is_gpu = True
                    else:
                        self._cupy = None
                        self._is_gpu = False
                else:
                    self._cupy = cupy
                    self._is_gpu = True
            except Exception:
                self._cupy = None
                self._is_gpu = False

        self._gpu_arrays: Dict[str, Any] = {}
        self._cpu_arrays: Dict[str, np.ndarray] = {}

    @property
    def is_gpu(self) -> bool:
        """True if running in active GPU (CuPy) mode."""
        return self._is_gpu

    def upload(self, name: str, array: np.ndarray) -> Any:
        """Transfer a NumPy array to GPU (CuPy), caching the GPU array.

        In CPU fallback mode, caches and returns the host NumPy array.

        Args:
            name: Identifier for the array.
            array: NumPy array to upload.

        Returns:
            The device array (CuPy ndarray) if on GPU, else the host NumPy array.
        """
        if not isinstance(array, np.ndarray):
            array = np.asarray(array)

        if self._is_gpu and self._cupy is not None:
            gpu_arr = self._cupy.asarray(array)
            self._gpu_arrays[name] = gpu_arr
            self._cpu_arrays[name] = array
            return gpu_arr
        else:
            self._cpu_arrays[name] = array
            return array

    def download(self, name: str) -> np.ndarray:
        """Transfer GPU array back to CPU as a NumPy array.

        In CPU mode, returns the cached CPU array.

        Args:
            name: Identifier of the array to download.

        Returns:
            np.ndarray: The array data on CPU.

        Raises:
            KeyError: If name has not been uploaded or cached.
        """
        if self._is_gpu and self._cupy is not None and name in self._gpu_arrays:
            gpu_arr = self._gpu_arrays[name]
            if hasattr(self._cupy, "asnumpy"):
                cpu_arr = self._cupy.asnumpy(gpu_arr)
            elif hasattr(gpu_arr, "get"):
                cpu_arr = gpu_arr.get()
            else:
                cpu_arr = np.asarray(gpu_arr)
            self._cpu_arrays[name] = cpu_arr
            return cpu_arr

        if name in self._cpu_arrays:
            return self._cpu_arrays[name]

        raise KeyError(f"Array '{name}' is not registered in GPUArrayManager")

    def sync(self) -> None:
        """Synchronize all GPU operations (cupy.cuda.Stream.null.synchronize).

        No-op in CPU mode.
        """
        if self._is_gpu and self._cupy is not None:
            try:
                self._cupy.cuda.Stream.null.synchronize()
            except Exception:
                pass

    def is_on_gpu(self, name: str) -> bool:
        """Check if an array is currently stored on the GPU.

        Always returns False in CPU mode.

        Args:
            name: Identifier of the array.

        Returns:
            bool: True if on GPU, False otherwise.
        """
        return self._is_gpu and (name in self._gpu_arrays)

    def cleanup(self) -> None:
        """Free all GPU memory and clear cached arrays.

        In CPU mode, clears cached arrays.
        """
        self._gpu_arrays.clear()
        self._cpu_arrays.clear()
        if self._is_gpu and self._cupy is not None:
            try:
                mempool = self._cupy.get_default_memory_pool()
                mempool.free_all_blocks()
            except Exception:
                pass
            try:
                pinned_pool = self._cupy.get_default_pinned_memory_pool()
                pinned_pool.free_all_blocks()
            except Exception:
                pass

    def get(self, name: str) -> Any:
        """Retrieve the active representation of array 'name' (GPU if available, else CPU)."""
        if self._is_gpu and name in self._gpu_arrays:
            return self._gpu_arrays[name]
        return self._cpu_arrays.get(name)

    def __contains__(self, name: str) -> bool:
        return (name in self._gpu_arrays) or (name in self._cpu_arrays)

    def __len__(self) -> int:
        return len(self._gpu_arrays) if self._is_gpu else len(self._cpu_arrays)
