import contextlib
import io
import numpy as np
import pytest

from pyradioss import accel
from pyradioss.engine.engine import run_engine
from pyradioss.starter.starter import run_starter

# is the optional numba backend importable on this machine?
try:
    import pyradioss.accel.jit_kernels  # noqa: F401
    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False

needs_numba = pytest.mark.skipif(
    not HAS_NUMBA, reason="numba not installed (optional dependency)")

@pytest.fixture(autouse=True)
def _restore_backend():
    """Every test in this module leaves the process on the NumPy backend."""
    yield
    accel.select_backend("numpy")

def _starter_only(make_deck, name, starter):
    s, _ = make_deck(name, starter, "/RUN/X/1\n1.0\n")
    with contextlib.redirect_stdout(io.StringIO()):
        return run_starter(s)

STEEL_LAW1 = """\
/MAT/LAW1/1
steel elastic
7.8e-6
210. 0.3
"""

_SHELL_DECK = (
    "/BEGIN\nshell qbat parity\n"
    "/NODE\n"
    "1 0 0 0\n2 1 0 0.1\n3 1.1 1 0\n4 0 1.05 0.05\n"
    "5 2 0 0\n6 2.1 1 0.15\n"
    "/SHELL/1\n1 1 2 3 4\n2 2 5 6 3\n"
    "/PART/1\nplate\n1 1\n" + STEEL_LAW1 +
    "/PROP/SHELL/1\nshell qbat\n1 0 0 0\n0.01 0.01 0.01 0 0\n3 1 0.4\n" # Ismstr=1 for QBAT
    "/END\n"
)

@needs_numba
def test_qbat_pre_post_parity(make_deck):
    """qbat_pre/qbat_post mirror shell_qbat._pre/_post on warped quads."""
    from pyradioss.accel.jit_kernels import qbat_pre_flat, qbat_pre_warp, qbat_post_flat, qbat_post_warp
    from pyradioss.elements import shell_qbat as sq

    model = _starter_only(make_deck, "QBP", _SHELL_DECK)
    g = model.shells
    rng = np.random.default_rng(123)
    xe = model.x[g.conn]
    ve = 0.2 * rng.standard_normal(xe.shape)
    vre = 0.1 * rng.standard_normal(xe.shape)
    off = g.state["off"]
    dt = 1e-3
    force_flat = np.zeros(g.n, dtype=bool)

    out_np = sq._pre(xe, ve, vre, off, dt, force_flat)
    
    # We must construct out_nb by calling the new jit functions.
    # The subagent is currently writing them.
    # We'll fill this in later!
    pass
