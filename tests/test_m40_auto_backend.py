"""M40: the ``auto`` compute-backend default.

Since M40 the default backend is ``auto`` (``pyradioss/accel/__init__.py``):
numba IS the default, but only once the model is big enough to amortise its
JIT warm-up.  ``accel.auto_select_backend(model)`` — called by the engine
after the restart is read and before the first kernel call — picks numba
when (a) numba imports cleanly and (b) the model has at least
``_AUTO_MIN_ELEMENTS`` total elements; small models stay on NumPy, and a
user pin (``PYRADIOSS_BACKEND`` / ``-backend numpy|numba``) is always
respected.

The threshold (32 elements) is DERIVED, not guessed, from the M39
numpy->numba speed sweep (``tools/validation_data/perf_m39_speed.json`` +
the per-deck sizes in ``perf_m39.json``): the sole regression was gas_piston
(4 elements, 0.46x) and the sole marginal case antenna_mast (10 elements,
1.12x, inside the sweep's contention noise); the smallest robust win was
tensile_bar (40 elements, 1.54x).  See ``accel._AUTO_MIN_ELEMENTS`` for the
full table.  These tests pin that contract: the selection matrix, the
forced overrides, the numba-absent fallback, and an end-to-end parity
spot-check proving the auto path reproduces the NumPy reference."""

import contextlib
import io
import re

import numpy as np
import pytest

from pyradioss import accel
from pyradioss.engine.engine import run_engine
from pyradioss.starter.starter import run_starter

try:
    import pyradioss.accel.jit_kernels  # noqa: F401
    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False

needs_numba = pytest.mark.skipif(
    not HAS_NUMBA, reason="numba not installed (optional dependency)")

THR = accel._AUTO_MIN_ELEMENTS


@pytest.fixture(autouse=True)
def _isolate_backend(monkeypatch):
    """Backend selection is process-global.  Start every test from the
    pristine, unresolved 'auto' state with the env knobs cleared, and leave
    the process back on NumPy (like the M7 fixture) so nothing leaks."""
    monkeypatch.delenv("PYRADIOSS_BACKEND", raising=False)
    monkeypatch.delenv("PYRADIOSS_BACKEND_AUTO_MIN_ELEMENTS", raising=False)
    accel._state.update(name=None, mod=None, forced=False)
    yield
    accel._state.update(name=None, mod=None, forced=False)
    accel.select_backend("numpy")


def _reset():
    """Return to the pristine unresolved 'auto' state mid-test."""
    accel._state.update(name=None, mod=None, forced=False)


# --- a stand-in model exposing only what auto_select_backend touches --------
class _Grp:
    def __init__(self, n):
        self.n = n


class _Model:
    """element_groups() yields (name, group-with-.n) — the workload signal."""

    def __init__(self, *group_sizes):
        self._groups = [(f"g{i}", _Grp(n)) for i, n in enumerate(group_sizes)]

    def element_groups(self):
        return iter(self._groups)


# ============================================================================
# Threshold derivation invariants
# ============================================================================

def test_threshold_sits_in_the_m39_gap():
    """The derived threshold must sit strictly between the M39 sweep's
    marginal/loss cluster (antenna_mast 10 elem, gas_piston 4 elem) and the
    smallest ROBUST win (tensile_bar 40 elem).  A future edit that moves it
    out of that (10, 40] gap breaks the derivation and should fail here."""
    assert 10 < THR <= 40


def test_element_count_sums_all_groups():
    """Model size is the TOTAL element count across groups, not one group —
    gas_piston (M39's regression) is small in ELEMENTS (4) though it has 20
    nodes, which is exactly why nodes are not the metric."""
    assert accel._model_element_count(_Model(10, 20, 5)) == 35
    assert accel._model_element_count(_Model(THR)) == THR
    assert accel._model_element_count(_Model()) == 0


# ============================================================================
# Auto selection matrix (size gate)
# ============================================================================

def test_auto_below_threshold_stays_numpy():
    """Small models where JIT warm-up dominates stay on NumPy — including the
    exact M39 regression size (gas_piston, 4 elements)."""
    assert accel.auto_select_backend(_Model(THR - 1)) == "numpy"
    assert accel.get("hexa_pre") is None            # inline NumPy path
    _reset()
    assert accel.auto_select_backend(_Model(4)) == "numpy"      # gas_piston
    _reset()
    assert accel.auto_select_backend(_Model(1)) == "numpy"


@needs_numba
def test_auto_at_and_above_threshold_picks_numba():
    """At/above the threshold the auto default IS numba, and the kernels are
    served from the numba module."""
    assert accel.auto_select_backend(_Model(THR)) == "numba"
    assert accel.get("hexa_pre") is not None
    _reset()
    assert accel.auto_select_backend(_Model(THR * 8)) == "numba"
    _reset()
    # summed across groups: two sub-threshold groups that together cross it
    half = THR // 2 + 1
    assert accel.auto_select_backend(_Model(half, half)) == "numba"


def test_auto_falls_back_to_numpy_when_numba_absent(monkeypatch):
    """Even a huge model stays on NumPy when numba will not import — the base
    install must never break.  Simulated by making the shared import seam
    (``accel._load_numba_module``) raise, so this runs with or without numba
    actually installed."""
    def _no_numba():
        raise ImportError("simulated: numba not installed")

    monkeypatch.setattr(accel, "_load_numba_module", _no_numba)
    assert accel.auto_select_backend(_Model(10_000)) == "numpy"
    assert accel.get("hexa_pre") is None


@needs_numba
def test_auto_numba_is_gated_to_explicit_runs():
    """Under auto, implicit runs stay NumPy — the M39 speed sweep that set
    the threshold covers the explicit /RUN leap-frog path only.  A pin still
    forces numba on an implicit run."""
    assert accel.auto_select_backend(_Model(10_000), explicit=False) == "numpy"
    _reset()
    accel.select_backend("numba")
    assert accel.auto_select_backend(_Model(10_000), explicit=False) == "numba"


# ============================================================================
# Forced overrides — a pin always beats the size gate
# ============================================================================

def test_default_select_is_auto_not_forced(monkeypatch):
    """With no env and no arg, select_backend resolves provisionally to NumPy
    but NOT forced, so the engine can still upgrade it to numba by size."""
    monkeypatch.delenv("PYRADIOSS_BACKEND", raising=False)
    assert accel.select_backend() == "numpy"
    assert accel._state["forced"] is False


def test_forced_numpy_beats_size_gate():
    """PYRADIOSS_BACKEND=numpy / -backend numpy pins NumPy on ANY model."""
    accel.select_backend("numpy")
    assert accel._state["forced"] is True
    assert accel.auto_select_backend(_Model(10_000)) == "numpy"
    assert accel.get("hexa_pre") is None


@needs_numba
def test_forced_numba_beats_size_gate():
    """A numba pin runs numba even on a below-threshold (tiny) model."""
    accel.select_backend("numba")
    assert accel._state["forced"] is True
    assert accel.auto_select_backend(_Model(1)) == "numba"


def test_env_var_pin_is_respected(monkeypatch):
    """An env pin set before any resolution is honoured by the auto path
    (numpy always; numba where numba is present)."""
    monkeypatch.setenv("PYRADIOSS_BACKEND", "numpy")
    _reset()
    assert accel.auto_select_backend(_Model(10_000)) == "numpy"

    monkeypatch.setenv("PYRADIOSS_BACKEND", "numba")
    _reset()
    got = accel.auto_select_backend(_Model(1))
    assert got == ("numba" if HAS_NUMBA else "numpy")


def test_threshold_env_override(monkeypatch):
    """PYRADIOSS_BACKEND_AUTO_MIN_ELEMENTS retunes the gate (for A/B tuning);
    a bad value falls back to the compiled default."""
    monkeypatch.setenv("PYRADIOSS_BACKEND_AUTO_MIN_ELEMENTS", "8")
    assert accel._auto_min_elements() == 8
    monkeypatch.setenv("PYRADIOSS_BACKEND_AUTO_MIN_ELEMENTS", "not-an-int")
    assert accel._auto_min_elements() == THR


# ============================================================================
# End-to-end: the engine call site + the listing line
# ============================================================================

STEEL_LAW1 = "/MAT/LAW1/1\nsteel elastic\n7.8e-6\n210. 0.3\n"


def _n_bricks(nx, ny, nz):
    return nx * ny * nz


def _brick_grid_deck(nx, ny, nz, kick=-2.0, size=1.0):
    """A structured nx*ny*nz steel-brick block, base (z=0) fixed, the whole
    block kicked downward — a compression that keeps the hexa kernels busy
    with non-zero stress.  Element count = nx*ny*nz, so the caller dials the
    model to either side of the auto threshold."""
    nxp, nyp = nx + 1, ny + 1

    def nid(i, j, k):
        return i + j * nxp + k * nxp * nyp + 1

    L = ["/BEGIN", "brick grid", "/NODE"]
    for k in range(nz + 1):
        for j in range(ny + 1):
            for i in range(nx + 1):
                L.append(f"{nid(i,j,k)} {i*size} {j*size} {k*size}")
    L.append("/BRICK/1")
    eid = 1
    for k in range(nz):
        for j in range(ny):
            for i in range(nx):
                c = (nid(i, j, k), nid(i+1, j, k), nid(i+1, j+1, k),
                     nid(i, j+1, k), nid(i, j, k+1), nid(i+1, j, k+1),
                     nid(i+1, j+1, k+1), nid(i, j+1, k+1))
                L.append(f"{eid} " + " ".join(map(str, c)))
                eid += 1
    L += ["/PART/1", "brick", "1 1", STEEL_LAW1.strip(),
          "/PROP/SOLID/1", "solid", "1.1 0.05 0.1"]
    alln = [nid(i, j, k) for k in range(nz + 1)
            for j in range(nyp) for i in range(nxp)]
    base = [nid(i, j, 0) for j in range(nyp) for i in range(nxp)]
    L += ["/GRNOD/NODE/1", "base", " ".join(map(str, base))]
    L += ["/GRNOD/NODE/2", "all", " ".join(map(str, alln))]
    L += ["/BCS/1", "base fixed", "111 111 0 1"]
    L += ["/INIVEL/TRA/1", "kick", f"0 0 {kick} 2"]
    L += ["/END"]
    return "\n".join(L) + "\n"


def _read_backend_line(out_text):
    m = re.search(r"COMPUTE BACKEND[ .]*: (\w+) \((.*)\)", out_text)
    assert m, f"no COMPUTE BACKEND line in listing:\n{out_text[:600]}"
    return m.group(1), m.group(2)


def test_listing_names_numpy_for_small_deck(make_deck, tmp_path):
    """A below-threshold deck run through the real engine logs
    'COMPUTE BACKEND ... : numpy (auto: N elements < THR ...)'.  Exercises
    the engine call site + the log line without needing numba."""
    deck = _brick_grid_deck(2, 2, 1)         # 4 bricks < THR
    assert _n_bricks(2, 2, 1) < THR
    s, e = make_deck("M40SMALL", deck,
                     "/RUN/M40SMALL/1\n0.002\n/DT\n0.9 0\n/PRINT/-9999\n"
                     "/STOP\n50.0\n")
    _reset()
    with contextlib.redirect_stdout(io.StringIO()):
        run_starter(s)
        run_engine(e)
    name, reason = _read_backend_line((tmp_path / "M40SMALL_0001.out").read_text())
    assert name == "numpy"
    assert "auto" in reason and "elements" in reason


# ============================================================================
# The parity spot-check: the auto path must reproduce the NumPy reference
# ============================================================================

@needs_numba
@pytest.mark.slow          # runs two full starter+engine passes + JIT compile
def test_auto_path_selects_numba_and_matches_numpy_reference(make_deck, tmp_path):
    """THE M40 safety check.  A >= threshold deck run under the AUTO default
    must (a) actually select numba and say so in the listing, and (b) agree
    state-by-state with the forced-NumPy reference — the parity contract is
    exactly what makes flipping the default safe.

    Cross-backend agreement is at the M7 full-run tolerance (1e-8 relative):
    the short, smooth run keeps the documented ulp-level reduction
    differences far below it.  The auto run's starter executes on NumPy (the
    provisional default) and only the engine loop flips to numba — the real
    production scenario, and still bit-parity-safe because the starter's
    initial state is backend-independent to machine precision."""
    deck = _brick_grid_deck(4, 4, 3)                    # 48 bricks >= THR
    assert _n_bricks(4, 4, 3) >= THR
    engine = "/RUN/GRID/1\n0.1\n/DT\n0.9 0\n/PRINT/-9999\n/STOP\n50.0\n"

    # reference: forced NumPy, both passes
    accel.select_backend("numpy")
    sn, en = make_deck("GRIDNP", deck, engine)
    with contextlib.redirect_stdout(io.StringIO()):
        run_starter(sn)
        m_np = run_engine(en)

    # auto: pristine/unresolved, no pin -> must upgrade to numba (48 >= THR)
    _reset()
    sa, ea = make_deck("GRIDAU", deck, engine)
    with contextlib.redirect_stdout(io.StringIO()):
        run_starter(sa)
        m_au = run_engine(ea)

    # (a) the auto run selected numba and named it (with the element count)
    assert accel._state["name"] == "numba"
    name, reason = _read_backend_line((tmp_path / "GRIDAU_0001.out").read_text())
    assert name == "numba"
    m = re.search(r"auto: (\d+) elements >=", reason)
    assert m and int(m.group(1)) >= THR, reason

    # (b) parity vs the NumPy reference — state array by state array
    assert m_np.engine_state.cycle == m_au.engine_state.cycle
    scale_x = max(float(np.abs(m_np.x).max()), 1e-9)
    scale_v = max(float(np.abs(m_np.v).max()), 1e-3)
    assert np.allclose(m_np.x, m_au.x, rtol=0, atol=1e-8 * scale_x)
    assert np.allclose(m_np.v, m_au.v, rtol=0, atol=1e-8 * scale_v)
    s_np = m_np.bricks.state["sig"]
    s_au = m_au.bricks.state["sig"]
    scale_s = max(float(np.abs(s_np).max()), 1e-6)
    assert float(np.abs(s_np).max()) > 0.0            # the kernels did work
    assert np.allclose(s_np, s_au, rtol=0, atol=1e-8 * scale_s)
    # the energy ledgers (sums of everything above) must agree too
    assert m_np.engine_state.e_num == pytest.approx(
        m_au.engine_state.e_num, rel=1e-6, abs=1e-12)
