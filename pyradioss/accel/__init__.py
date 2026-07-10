"""
pyradioss.accel — optional accelerated compute backends (M7).

There is no Fortran origin for this package: OpenRadioss gets its speed
from compiled Fortran + OpenMP/MPI (out of scope for the port). This
package is the port's own answer to "the readable NumPy solver is slow":
an *optional* numba backend behind the exact same kernel API.

Backend architecture
--------------------
The element kernels and the TYPE7 contact keep their NumPy code inline —
that code IS the reference implementation and the readable narrative of
the port. During the M7 profiling pass each hot kernel was split into
"pre" and "post" blocks around its Python-level material/failure loop
(which stays pure Python — material laws dispatch on law objects and are
NOT part of any backend):

    solid_hexa8.forces:  _pre  (geometry, velocity gradient, Jaumann
                                rotation, characteristic length)
                         [material laws / EOS / failure — plain Python]
                         _post (bulk viscosity, internal + hourglass
                                forces, energy increments, scatter,
                                critical dt)
    shell_bt4.forces:    _pre  (corotational frame, local geometry,
                                membrane/curvature/shear rates)
                         [layer loop: material laws / failure — Python]
                         _post (resultant forces, BLT84 hourglass,
                                back-transform, scatter)
    inter_type7:         _narrow (exact node-triangle closest points)

Each block is a plain function of arrays. ``accel.get(name)`` returns
the active backend's implementation of a block, or ``None`` meaning
"use the inline NumPy code". The kernels are therefore written as

    jit = accel.get("hexa_pre")
    if jit is not None:
        dndx, vol, ... = jit(...)          # numba mirror
    else:
        ...inline NumPy (the reference)...

so a reader can ignore the accelerated path entirely, and the NumPy
path never pays more than one dict lookup per block.

Selecting a backend
-------------------
* environment variable ``PYRADIOSS_BACKEND=numpy|numba`` (read lazily on
  the first kernel call, so it works for pytest and plain scripts), or
* the engine command line: ``pyradioss-engine -i RUN_0001.rad -backend
  numba`` (overrides the environment), or
* programmatically: ``accel.select_backend("numba")``.

numba is an OPTIONAL dependency: requesting the numba backend without
numba installed prints a warning and falls back to NumPy — the base
install keeps working unchanged. The numba kernels are compiled with
``cache=True`` so the (one-time, ~10 s) JIT cost is paid on the first
run only; subsequent processes load the cached machine code.

The parity contract (the M7 hard requirement)
---------------------------------------------
A backend MUST NOT change results. Concretely:

* every numba kernel mirrors its NumPy block operation for operation —
  same formulas, same branch structure, same evaluation order — so the
  two backends agree bitwise wherever the NumPy expression maps to one
  scalar expression per element;
* the unavoidable differences are *reduction associations* (a numba
  scalar loop accumulates sequentially; NumPy einsum/matmul may use
  SIMD partial sums). These show up at machine precision (≲1e-15
  relative per operation) and are the documented per-kernel tolerance;
* ``tests/test_m7_backends.py`` enforces the contract at two levels:
  single-call kernel parity on real element groups (tight tolerance)
  and full starter+engine runs compared state-array by state-array;
* the M6 restart-chaining acceptance tests (chained == unchained,
  exactly) must hold WITHIN each backend — reordered reductions would
  break that immediately, which makes the chaining tests the regression
  canary for nondeterminism (e.g. accidentally enabling numba's
  ``parallel=True`` or ``fastmath=True``, both of which are forbidden
  here).

Why there is no JAX backend (deferred, see PORTING_GUIDE §5)
------------------------------------------------------------
JAX's value is jit-compiling the WHOLE cycle onto an accelerator, but
this engine's cycle is built around in-place scatter into shared force
arrays, per-cycle Python branching (sensors, deletion, EOS, /DT/NODA)
and stateful dict-of-arrays element buffers — none of which map to
jax.jit without rewriting the engine in functional style, i.e. a fork,
not a backend. A jax.numpy drop-in without jit would only add dispatch
overhead at these array sizes. Deferred until (if ever) the engine
grows a functional-core refactor.
"""

from __future__ import annotations

import os
import warnings

#: resolved backend: {"name": "numpy"|"numba", "mod": module or None}.
#: ``name = None`` means "not resolved yet" — the environment variable is
#: read lazily on the first ``get()`` so library users, pytest and the
#: CLIs all honour PYRADIOSS_BACKEND without extra plumbing.
_state = {"name": None, "mod": None}


def select_backend(name=None, log=None) -> str:
    """Select the compute backend; returns the name actually activated.

    ``name = None`` reads ``PYRADIOSS_BACKEND`` (default "numpy").
    Unknown names and a missing numba installation warn and fall back to
    NumPy — requesting acceleration must never break a run."""
    if name is None:
        name = os.environ.get("PYRADIOSS_BACKEND", "numpy")
    name = (name or "numpy").strip().lower()

    if name == "numba":
        try:
            from . import jit_kernels  # noqa: F401 — import compiles/loads
            _state.update(name="numba", mod=jit_kernels)
        except ImportError:
            msg = ("numba backend requested but numba is not installed — "
                   "falling back to NumPy (pip install numba)")
            warnings.warn(msg, stacklevel=2)
            if log is not None:
                log.warning(msg, "BACKEND")
            _state.update(name="numpy", mod=None)
    elif name == "jax":
        msg = ("JAX backend is deferred (see the M7 notes in "
               "PORTING_GUIDE.md §5) — falling back to NumPy")
        warnings.warn(msg, stacklevel=2)
        if log is not None:
            log.warning(msg, "BACKEND")
        _state.update(name="numpy", mod=None)
    else:
        if name != "numpy":
            msg = f"unknown PYRADIOSS_BACKEND '{name}' — using NumPy"
            warnings.warn(msg, stacklevel=2)
            if log is not None:
                log.warning(msg, "BACKEND")
        _state.update(name="numpy", mod=None)
    return _state["name"]


def backend_name(log=None) -> str:
    """The active backend name ("numpy" or "numba"), resolving lazily
    (``log`` routes a fallback warning into the engine listing)."""
    if _state["name"] is None:
        select_backend(log=log)
    return _state["name"]


def get(kernel: str):
    """The active backend's implementation of a kernel block, or None
    meaning 'use the inline NumPy reference code'. This is the single
    dispatch point the element kernels call (one dict lookup + getattr
    per block per cycle — negligible)."""
    if _state["name"] is None:
        select_backend()
    mod = _state["mod"]
    return getattr(mod, kernel, None) if mod is not None else None
