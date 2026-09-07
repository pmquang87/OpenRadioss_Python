"""M7 validations: backend selection, the fastmath primitive contracts,
the numba mirrors' parity contract (kernel level AND full starter+engine
runs), and the restart-chaining canary under the accelerated backend
(see PORTING_GUIDE.md roadmap M7 and the pyradioss/accel docstring).

Tolerance policy (the documented part of the parity contract): element-
wise expressions agree bitwise between the backends; short reductions
(3..8-term dot products) may be associated differently by NumPy's
einsum/matmul than by the numba scalar loops, bounding single-call
differences at machine precision. Kernel-parity tests therefore use
rtol = 1e-12; full-run comparisons use looser tolerances because an
explicit integration amplifies ulp differences cycle by cycle (the runs
compared here are short and smooth, so 1e-8 is comfortable)."""

import contextlib
import io
import warnings

import numpy as np
import pytest

from pyradioss import accel
from pyradioss.common import fastmath
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
    """Every test in this module leaves the process on the NumPy backend
    (backend selection is process-global state)."""
    yield
    accel.select_backend("numpy")


def _run(make_deck, name, starter, engine):
    s, e = make_deck(name, starter, engine)
    with contextlib.redirect_stdout(io.StringIO()):
        run_starter(s)
        model = run_engine(e)
    return model


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


# ============================================================================
# Backend selection plumbing
# ============================================================================

def test_default_backend_is_numpy(monkeypatch):
    monkeypatch.delenv("PYRADIOSS_BACKEND", raising=False)
    assert accel.select_backend() == "numpy"
    assert accel.get("hexa_pre") is None       # inline NumPy path


def test_unknown_backend_falls_back_with_warning():
    with pytest.warns(UserWarning, match="unknown"):
        assert accel.select_backend("cuda") == "numpy"


def test_jax_backend_is_documented_deferred():
    """JAX is stretch scope explicitly deferred in the porting guide —
    requesting it must warn and fall back, not crash."""
    with pytest.warns(UserWarning, match="deferred"):
        assert accel.select_backend("jax") == "numpy"


def test_env_var_selects_backend(monkeypatch):
    monkeypatch.setenv("PYRADIOSS_BACKEND", "numpy")
    assert accel.select_backend() == "numpy"


@needs_numba
def test_numba_backend_provides_kernels():
    assert accel.select_backend("numba") == "numba"
    for k in ("hexa_pre", "hexa_post", "shell_pre", "shell_post",
              "t7_narrow", "scatter3"):
        assert accel.get(k) is not None
    accel.select_backend("numpy")
    assert accel.get("hexa_pre") is None
    assert accel.get("scatter3") is None       # inline NumPy scatter path


# ============================================================================
# fastmath primitives: the bitwise-identical replacements (M7 cheap wins)
# ============================================================================

def test_fastmath_bitwise_contracts():
    rng = np.random.default_rng(7)
    a = rng.standard_normal((257, 3))
    b = rng.standard_normal((257, 3))
    assert np.array_equal(fastmath.cross3(a, b), np.cross(a, b))
    assert np.array_equal(fastmath.norm3(a), np.linalg.norm(a, axis=-1))
    # broadcasting form used by the rigid bodies: (3,) x (n, 3)
    w = rng.standard_normal(3)
    assert np.array_equal(fastmath.cross3(w, a), np.cross(w, a))

    # scatter_add3 == add.at into a zero target, bitwise (bincount
    # accumulates in input order exactly like add.at)
    idx = rng.integers(0, 40, 700)
    vals = rng.standard_normal((700, 3))
    t1 = np.zeros((40, 3))
    np.add.at(t1, idx, vals)
    t2 = np.zeros((40, 3))
    fastmath.scatter_add3(t2, idx, vals)
    assert np.array_equal(t1, t2)


@needs_numba
def test_scatter3_numba_bitwise_parity():
    """The M39 fused numba force-scatter (accel.scatter3, dispatched by
    fastmath.scatter_add3 under the numba backend) must reproduce the
    bincount NumPy reference BIT-FOR-BIT — for ANY starting target, not
    just a zeroed one. bincount accumulates each component in input order
    and the reference then adds the full-length bin to the target; the
    numba mirror accumulates the same values in the same order into a
    zeroed scratch and adds that, so cross-group additions (a node already
    carrying force from another element group) associate identically. The
    direct-scatter shortcut would reassociate and is explicitly NOT used —
    this test is the canary that keeps it that way."""
    from pyradioss.accel import jit_kernels as jk

    rng = np.random.default_rng(19)
    n, m = 40, 700
    idx = rng.integers(0, n, m)
    vals = rng.standard_normal((m, 3))

    # kernel vs bincount reference, starting from a NON-ZERO target
    base = rng.standard_normal((n, 3))
    t_ref = base.copy()
    fastmath.scatter_add3(t_ref, idx, vals)         # numpy backend (fixture)
    t_nb = base.copy()
    jk.scatter3(t_nb, idx, vals)
    assert np.array_equal(t_ref, t_nb)

    # and the full dispatch path: fastmath.scatter_add3 under the numba
    # backend routes to scatter3 and stays bitwise-equal to add.at (zero)
    accel.select_backend("numba")
    t_dispatch = np.zeros((n, 3))
    fastmath.scatter_add3(t_dispatch, idx, vals)
    accel.select_backend("numpy")
    t_addat = np.zeros((n, 3))
    np.add.at(t_addat, idx, vals)
    assert np.array_equal(t_dispatch, t_addat)


def test_anim_write_block_matches_savetxt_byte_for_byte():
    """The M39 anim output-path replacement (anim_vtk._write_block) must
    reproduce np.savetxt byte-for-byte over the fmt/shape set write_anim_state
    uses ("%.9E" floats 1-D/2-D, "%d" ints 1-D/2-D, "%.1f" the OFF flag) —
    INCLUDING the IEEE edge values (inf, nan, -0.0, denormals, huge) whose
    text form must not drift. This is what lets write_anim_state drop
    np.savetxt's per-row Python formatting + per-row fh.write for one C
    unbox + one write per block without changing a single output byte."""
    import io
    from pyradioss.output.anim_vtk import _write_block

    rng = np.random.default_rng(5)

    def _ref(arr, fmt):
        b = io.StringIO()
        np.savetxt(b, arr, fmt=fmt)
        return b.getvalue()

    def _fast(arr, fmt):
        b = io.StringIO()
        _write_block(b, arr, fmt)
        return b.getvalue()

    cases = [
        (rng.standard_normal((300, 3)) * 1e3, "%.9E"),          # POINTS/VEL
        (np.array([[0.0, -0.0, 1.0], [-1e-30, 1e30, -3.14159],
                   [1e-300, -1e300, 0.0],
                   [np.inf, -np.inf, np.nan]]), "%.9E"),          # IEEE edges
        (rng.standard_normal(300) * 10, "%.9E"),                 # VONM/EPSP
        (rng.integers(0, 2, 200).astype(float), "%.1f"),         # OFF flag
        (rng.integers(0, 72000, (300, 9)).astype(np.int64), "%d"),  # CELLS
        (np.full(300, 12, dtype=np.int64), "%d"),                # CELL_TYPES
    ]
    for arr, fmt in cases:
        assert _fast(arr, fmt) == _ref(arr, fmt)


def test_fastmath_det_inv33_accuracy():
    """Cofactor det/inv vs LAPACK: not bitwise (different algorithm,
    documented), but at machine precision on element-quality Jacobians."""
    rng = np.random.default_rng(3)
    J = np.eye(3) + 0.3 * rng.standard_normal((100, 3, 3))
    det, inv = fastmath.det_inv33(J)
    assert np.allclose(det, np.linalg.det(J), rtol=1e-12)
    assert np.allclose(inv, np.linalg.inv(J), rtol=1e-10, atol=1e-13)


# ============================================================================
# Kernel-level parity: numba mirrors vs the inline NumPy reference
# ============================================================================

_HEXA_DECK = (
    "/BEGIN\nhexa parity\n"
    "/NODE\n"
    "1 0 0 0\n2 1.1 0 0\n3 1 1.2 0\n4 0 1 0\n"
    "5 0 0.1 1\n6 1 0 1.3\n7 1.2 1 1\n8 0 1 1.1\n"
    "9 2.1 0 0.2\n10 2 1.1 0\n11 2.2 0 1\n12 2 1 1.2\n"
    "/BRICK/1\n1 1 2 3 4 5 6 7 8\n2 2 9 10 3 6 11 12 7\n"
    "/PART/1\nblock\n1 1\n" + STEEL_LAW1 +
    "/PROP/SOLID/1\nsolid\n1.1 0.05 0.1\n"
    "/END\n"
)

_SHELL_DECK = (
    "/BEGIN\nshell parity\n"
    "/NODE\n"
    "1 0 0 0\n2 1 0 0.1\n3 1.1 1 0\n4 0 1.05 0.05\n"
    "5 2 0 0\n6 2.1 1 0.15\n"
    "/SHELL/1\n1 1 2 3 4\n2 2 5 6 3\n"
    "/PART/1\nplate\n1 1\n" + STEEL_LAW1 +
    "/PROP/SHELL/1\nshell\n1 0 0 0\n0.01 0.01 0.01 0 0\n3 0 0.4\n"
    "/END\n"
)


@needs_numba
def test_hexa_pre_post_parity(make_deck):
    """hexa_pre/hexa_post mirror solid_hexa8._pre/_post on a real
    (distorted) brick group with a rich velocity/stress state."""
    from pyradioss.accel import jit_kernels as jk
    from pyradioss.elements import solid_hexa8 as hx

    model = _starter_only(make_deck, "HXP", _HEXA_DECK)
    g = model.bricks
    rng = np.random.default_rng(42)
    xe = model.x[g.conn]
    ve = 0.3 * rng.standard_normal(xe.shape)
    sig0 = 5.0 * rng.standard_normal((g.n, 6))
    dt = 1e-3
    off = g.state["off"]

    sig_np = sig0.copy()
    dndx, vol, lc, deps, trD = hx._pre(xe, ve, sig_np, dt, off, np.ones(g.n))
    sig_nb = sig0.copy()
    dndx2, vol2, lc2, deps2, trD2 = jk.hexa_pre(xe, ve, sig_nb, dt, off, np.ones(g.n))
    for a, b in ((dndx, dndx2), (vol, vol2), (lc, lc2), (deps, deps2),
                 (trD, trD2), (sig_np, sig_nb)):
        assert np.allclose(a, b, rtol=1e-12, atol=1e-15)

    # post: feed both the SAME (numpy-path) pre outputs so the comparison
    # isolates the post mirror
    rho = g.state["mass"] / vol
    sig_old = sig0.copy()
    n = g.n
    qa = np.full(n, 1.1)
    qb = np.full(n, 0.05)
    c = np.full(n, 6.0)
    hcoef = np.full(n, 0.1)
    alive = off > 0.0
    qvw = 0.01 * rng.standard_normal(n)
    out_np = hx._post(xe, ve, dndx, vol, lc, rho, trD, deps, sig_np,
                      sig_old, qa, qb, c, hcoef, alive, qvw, dt,
                      g.state["dtfac"])
    out_nb = jk.hexa_post(xe, ve, dndx, vol, lc, rho, trD, deps, sig_np,
                          sig_old, qa, qb, c, hcoef, alive, qvw, dt,
                          g.state["dtfac"])
    for a, b in zip(out_np, out_nb):
        assert np.allclose(a, b, rtol=1e-12, atol=1e-15)


@needs_numba
def test_shell_pre_post_parity(make_deck):
    """shell_pre/shell_post mirror shell_bt4._pre/_post on warped quads
    with translational + rotational velocities and live hourglass state."""
    from pyradioss.accel import jit_kernels as jk
    from pyradioss.elements import shell_bt4 as sh

    model = _starter_only(make_deck, "SHP", _SHELL_DECK)
    g = model.shells
    rng = np.random.default_rng(1)
    xe = model.x[g.conn]
    ve = 0.2 * rng.standard_normal(xe.shape)
    vre = 0.1 * rng.standard_normal(xe.shape)
    off = g.state["off"]

    out_np = sh._pre(xe, ve, vre, off)
    out_nb = jk.shell_pre(xe, ve, vre, off)
    for a, b in zip(out_np, out_nb):
        assert np.allclose(a, b, rtol=1e-12, atol=1e-15)

    E, area, lc, B1, B2, bb, gam, V, dm, kap, gs = out_np
    n = g.n
    Nres = rng.standard_normal((n, 3))
    Mres = rng.standard_normal((n, 3))
    qres = rng.standard_normal((n, 2))
    # chvis3.F coefficients (M39 shell_bt4._post signature): the elastic
    # membrane/bending stiffness (k_m/k_w) plus the three quadratic viscous
    # dampers (hqm/hqb/hqr). The SAME inputs drive both backends so the
    # mirror is still checked bit-for-bit — only the plumbing follows the new
    # signature; the rotation modes lost their elastic branch (old k_r).
    k_m = np.abs(rng.standard_normal(n))
    k_w = np.abs(rng.standard_normal(n))
    hqm = np.abs(rng.standard_normal(n))
    hqb = np.abs(rng.standard_normal(n))
    hqr = np.abs(rng.standard_normal(n))
    Q0 = rng.standard_normal((n, 5))
    dt = 1e-3

    Q_np = Q0.copy()
    fg1, mg1, dehg1 = sh._post(E, area, B1, B2, gam, V, Nres, Mres, qres,
                               Q_np, k_m, k_w, hqm, hqb, hqr, dt)
    Q_nb = Q0.copy()
    fg2, mg2, dehg2 = jk.shell_post(E, area, B1, B2, gam, V, Nres, Mres,
                                    qres, Q_nb, k_m, k_w, hqm, hqb, hqr, dt)
    for a, b in ((fg1, fg2), (mg1, mg2), (dehg1, dehg2), (Q_np, Q_nb)):
        assert np.allclose(a, b, rtol=1e-12, atol=1e-15)


@needs_numba
def test_t7_narrow_parity():
    """t7_narrow mirrors inter_type7._narrow over points that exercise
    EVERY Ericson region (face, three edges, three vertices) of both
    triangles of a warped quad."""
    from pyradioss.accel import jit_kernels as jk
    from pyradioss.contact import inter_type7 as t7

    rng = np.random.default_rng(11)
    # one warped quad + a cloud of probe nodes all around it (the wide
    # spread guarantees vertex/edge/face regions are all hit)
    quad = np.array([[0.0, 0.0, 0.0], [2.0, 0.1, 0.0],
                     [2.1, 2.0, 0.4], [-0.1, 1.9, 0.1]])
    npts = 500
    pts = rng.uniform(-2.5, 4.5, (npts, 3))
    pts[:, 2] = rng.uniform(-2.0, 2.0, npts)
    x = np.vstack([quad, pts])
    ni = np.arange(4, 4 + npts, dtype=np.int64)
    seg = np.tile(np.arange(4, dtype=np.int64), (npts, 1))

    d1, p1, w1 = t7._narrow(x, ni, seg)
    d2, p2, w2 = jk.t7_narrow(x, ni, seg)
    assert np.allclose(d1, d2, rtol=1e-12, atol=1e-15)
    assert np.allclose(p1, p2, rtol=1e-12, atol=1e-15)
    assert np.allclose(w1, w2, rtol=1e-12, atol=1e-14)
    # sanity: the closest point is never farther than the node-to-corner
    # distance (exactness of the region logic)
    dmin_corner = np.min(np.linalg.norm(
        x[ni][:, None, :] - quad[None, :, :], axis=2), axis=1)
    assert np.all(d1 <= dmin_corner + 1e-12)


# ============================================================================
# Full-run parity: a representative starter+engine model, both backends
# ============================================================================

# bricks + shells + TYPE7 contact + a rigid wall: every accelerated block
# is on the cycle path (hexa pre/post, shell pre/post, t7 narrow phase)
_IMPACT_DECK = (
    "/BEGIN\nbackend parity impact\n"
    "/NODE\n"
    # 2x1x1 brick bar on the ground
    "1 0 0 0\n2 1 0 0\n3 1 1 0\n4 0 1 0\n"
    "5 0 0 1\n6 1 0 1\n7 1 1 1\n8 0 1 1\n"
    "9 2 0 0\n10 2 1 0\n11 2 0 1\n12 2 1 1\n"
    # 2x2 shell flyer above, moving down
    "21 0.1 0.1 1.6\n22 0.9 0.1 1.6\n23 1.7 0.1 1.6\n"
    "24 0.1 0.9 1.6\n25 0.9 0.9 1.6\n26 1.7 0.9 1.6\n"
    "/BRICK/1\n1 1 2 3 4 5 6 7 8\n2 2 9 10 3 6 11 12 7\n"
    "/SHELL/2\n11 21 22 25 24\n12 22 23 26 25\n"
    "/PART/1\nbar\n1 1\n"
    "/PART/2\nflyer\n2 2\n" + STEEL_LAW1 +
    "/MAT/LAW2/2\nsteel jc\n7.8e-6\n210. 0.3\n0.4 0.5 0.5\n"
    "/PROP/SOLID/1\nsolid\n1.1 0.05 0.1\n"
    "/PROP/SHELL/2\nshell\n1 0 0 0\n0.01 0.01 0.01 0 0\n3 0 0.1\n"
    "/GRNOD/PART/1\nbar nodes\n1\n"
    "/GRNOD/PART/2\nflyer nodes\n2\n"
    "/GRNOD/NODE/3\nbar base\n1 2 3 4 9 10\n"
    "/SURF/PART/1\nbar surf\n1\n"
    "/INTER/TYPE7/1\nimpact\n2 1 2 1\n0.0 0.0 0.0\n"
    "/INIVEL/TRA/1\ndown\n0 0 -0.05 2\n"
    "/BCS/1\nbar base held\n111 111 0 3\n"
    "/END\n"
)


@needs_numba
@pytest.mark.slow          # measured 982.7 s serial (2026-07 full-suite run)
def test_full_run_backend_parity(make_deck, tmp_path):
    """THE backend acceptance test: the same starter+engine impact model
    run under both backends must agree state array by state array. The
    run is short (~1500 cycles) and smooth, so the ulp-level reduction
    differences documented in accel/ stay far below 1e-8 relative."""
    engine = "/RUN/PAR/1\n8.0\n/DT\n0.9 0\n/PRINT/-99999\n/STOP\n15.0\n"

    accel.select_backend("numpy")
    m_np = _run(make_deck, "PARN", _IMPACT_DECK, engine)
    accel.select_backend("numba")
    m_nb = _run(make_deck, "PARB", _IMPACT_DECK, engine)
    accel.select_backend("numpy")

    assert m_np.engine_state.cycle == m_nb.engine_state.cycle
    scale_x = np.abs(m_np.x).max()
    scale_v = max(np.abs(m_np.v).max(), 1e-3)
    assert np.allclose(m_np.x, m_nb.x, rtol=0, atol=1e-8 * scale_x)
    assert np.allclose(m_np.v, m_nb.v, rtol=0, atol=1e-8 * scale_v)
    for grp in ("bricks", "shells"):
        s_np = getattr(m_np, grp).state
        s_nb = getattr(m_nb, grp).state
        sc = max(float(np.abs(s_np["sig"]).max()), 1e-6)
        assert np.allclose(s_np["sig"], s_nb["sig"], rtol=0,
                           atol=1e-8 * sc)
        assert np.allclose(s_np["eint"], s_nb["eint"], rtol=1e-6,
                           atol=1e-12)
    # the ledgers must agree too (they are sums of everything above)
    e_np, e_nb = m_np.engine_state, m_nb.engine_state
    assert e_np.econt == pytest.approx(e_nb.econt, rel=1e-6, abs=1e-12)
    assert e_np.e_num == pytest.approx(e_nb.e_num, rel=1e-6, abs=1e-12)


# ============================================================================
# Restart chaining under numba (the M6 acceptance test as the M7 canary
# for nondeterministic reductions — see the accel package docstring)
# ============================================================================

@needs_numba
def test_restart_chain_bitmatch_under_numba(make_deck):
    """A chained run must reproduce the unchained one EXACTLY (the M6
    contract) WITHIN the numba backend: any nondeterminism in the jit
    kernels (parallel reductions, fastmath) would break this instantly.
    Same tumbling-/RBODY setup as the M6 test — bricks keep the hexa
    mirrors on the cycle path the whole run."""
    from pyradioss.starter.restart import read_restart

    accel.select_backend("numba")
    starter = (
        "/BEGIN\ntumbling body\n"
        "/NODE\n"
        "1 0 0 0\n2 2 0 0\n3 2 1 0\n4 0 1 0\n"
        "5 0 0 1\n6 2 0 1\n7 2 1 1\n8 0 1 1\n"
        "100 0 0 0\n"
        "/BRICK/1\n1 1 2 3 4 5 6 7 8\n"
        "/PART/1\nbrick\n1 1\n" + STEEL_LAW1 +
        "/PROP/SOLID/1\nsolid\n1.1 0.05 0.1\n"
        "/GRNOD/PART/1\nslaves\n1\n"
        "/RBODY/1\nbody\n100 1\n"
        "/INIVEL/AXIS/1\nspin\n3.0 Y 1 1.0 0.5 0.5\n"
        "/INIVEL/TRA/2\ndrift\n0.05 0.02 0.01 1\n"
        "/END\n"
    )
    s, e1 = make_deck("NBP", starter,
                      "/RUN/NBP/1\n0.05\n/DT\n0.45 0\n/PRINT/-9999\n"
                      "/STOP\n1e9\n")
    with contextlib.redirect_stdout(io.StringIO()):
        run_starter(s)
        run_engine(e1)
    dt0 = read_restart(e1.replace("_0001.rad", "_0001.rst"))[1]["dt"]
    t1 = 0.0
    for _ in range(300):
        t1 += dt0
    t_tot = t1
    for _ in range(400):
        t_tot += dt0

    mu = _run(make_deck, "NBU", starter,
              f"/RUN/NBU/1\n{t_tot!r}\n/DT\n0.45 0\n/PRINT/-9999\n"
              "/STOP\n1e9\n")
    s2, ec1 = make_deck(
        "NBC", starter,
        f"/RUN/NBC/1\n{t1!r}\n/DT\n0.45 0\n/PRINT/-9999\n/STOP\n1e9\n")
    ec2 = ec1.replace("_0001.rad", "_0002.rad")
    with open(ec2, "w") as fh:
        fh.write(f"/RUN/NBC/2\n{t_tot!r}\n/DT\n0.45 0\n/PRINT/-9999\n"
                 "/STOP\n1e9\n")
    with contextlib.redirect_stdout(io.StringIO()):
        run_starter(s2)
        run_engine(ec1)
        mc = run_engine(ec2)

    try:
        assert mc.engine_state.cycle == mu.engine_state.cycle == 700
        assert np.allclose(mc.x, mu.x, rtol=0, atol=1e-10)
        assert np.allclose(mc.v, mu.v, rtol=0, atol=1e-10)
        # interior of the body must still be at exactly zero strain
        assert np.abs(mc.bricks.state["sig"]).max() < 1e-12
    finally:
        accel.select_backend("numpy")
