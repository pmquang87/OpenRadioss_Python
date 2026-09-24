"""
Tests for Wave 2 element and backend fixes (BUG-ELEM-01 through BUG-ELEM-08).

Covers:
1. BUG-ELEM-01: qeph_post kernel signature and execution in pyradioss/accel/jit_kernels/shells_qeph.py
2. BUG-ELEM-02: shell_bt4 and shell_tri3 negative dt guard on collapsed/inverted elements
3. BUG-ELEM-03: bric20 3D dilatational sound speed sqrt((K + 4G/3)/rho) vs 1D bar speed
4. BUG-ELEM-04: shell_bt4 hourglass parameter defaults when hm/hf/hr missing from property
5. BUG-ELEM-05: tetra4/heph/tetra10 sound speed radicand clamp when K + 4G/3 <= 0
6. BUG-ELEM-06: Windows OpenMP threading stability (parallel=False on JIT kernels)
7. BUG-ELEM-07: shell_thick16 Courant factor dtfac application
8. BUG-ELEM-08: accel auto_backend exception fallback on RuntimeError, TypeError, NameError
"""

import types
import warnings
import numpy as np
import pytest

from pyradioss.common.constants import EM20, EP30
from pyradioss.model.model import ElementGroup, Model


# -----------------------------------------------------------------------------
# 1. BUG-ELEM-01: qeph_post kernel signature and execution
# -----------------------------------------------------------------------------
def test_bug_elem_01_qeph_post_kernel_signature_and_execution():
    """Verify qeph_post exists, is callable, and runs end-to-end with qeph_pre."""
    from pyradioss.accel import jit_kernels as jk

    assert hasattr(jk, "qeph_post"), "qeph_post must be in jit_kernels"
    qeph_post = jk.qeph_post
    qeph_pre = jk.qeph_pre
    assert qeph_post is not None, "qeph_post must not be None when numba is available"

    n = 1
    # Create valid 4-node quad geometry
    xe = np.array([[
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [1.0, 1.0, 0.0],
        [0.0, 1.0, 0.0]
    ]], dtype=np.float64)
    ve = np.zeros((n, 4, 3), dtype=np.float64)
    vre = np.zeros((n, 4, 3), dtype=np.float64)
    alive = np.ones(n, dtype=bool)
    dt = 1e-6
    npt1 = np.zeros(n, dtype=bool)

    (vdef, vhg, plat, vqn, di, db, E, area, a_i, z1, corx, cory,
     x13, x24, y13, y24, mx13, mx23, mx34, my13, my23, my34,
     l13, l24, ll, lm) = qeph_pre(xe, ve, vre, dt, npt1, alive)

    thick = np.array([0.01])
    Nres = np.zeros((n, 3))
    Mres = np.zeros((n, 3))
    qres = np.zeros((n, 2))
    st_amu = np.ones(n)
    st_cspd = np.full(n, 5000.0)
    st_yld = np.full(n, 200.0)
    st_fmat = np.zeros(n)
    a11 = np.full(n, 2.1e11)
    a12 = np.full(n, 0.3 * 2.1e11)
    gs = np.zeros(n)
    # GBUF%HOURG (VGLAS) has 12 slots for QEPH: czfintn.F:43-44,62 and
    # elbuf_ini.F:1622-1623 (JHBE=23 -> G_HOURG=12).
    vg = np.zeros((n, 12))

    fg, mg, dt_e = qeph_post(
        thick, Nres, Mres, qres, st_amu, st_cspd, st_yld, st_fmat, vhg, dt, alive,
        plat, vqn, di, db, E, area, a_i, z1, corx, cory, x13, x24, y13, y24,
        mx13, mx23, mx34, my13, my23, my34, l13, l24, ll, lm, a11, a12,
        npt1, gs, vg
    )
    assert fg.shape == (n, 4, 3)
    assert mg.shape == (n, 4, 3)
    assert dt_e.shape == (n,)
    assert np.all(np.isfinite(fg))
    assert np.all(np.isfinite(mg))
    assert np.all(np.isfinite(dt_e))


# -----------------------------------------------------------------------------
# 2. BUG-ELEM-02: shell_bt4 and shell_tri3 negative dt guard on collapsed elements
# -----------------------------------------------------------------------------
def test_bug_elem_02_collapsed_shell_dt_guard():
    """Verify collapsed/inverted shell_bt4 and shell_tri3 return safe EP30 time step."""
    from pyradioss.elements import shell_bt4, shell_tri3

    # BT4: 4 nodes, all at origin -> zero area collapsed element
    x_bt4 = np.zeros((4, 3), dtype=np.float64)
    v_bt4 = np.zeros((4, 3), dtype=np.float64)
    vr_bt4 = np.zeros((4, 3), dtype=np.float64)
    conn_bt4 = np.array([[0, 1, 2, 3]], dtype=np.int64)

    mat = types.SimpleNamespace(
        id=1, law=1, rho0=7800.0, E=2.1e11, nu=0.3,
        G=2.1e11 / (2.0 * 1.3), K=2.1e11 / (3.0 * 0.4),
        eos=None, params={"E": 2.1e11, "nu": 0.3}
    )
    prop = types.SimpleNamespace(id=1, type=1, thick=0.01, params={"thick": 0.01, "hm": 0.01, "hf": 0.01, "hr": 0.01})

    group_bt4 = ElementGroup(ids=np.array([1]), conn=conn_bt4, part=np.array([1]))
    group_bt4.state = {
        "mass": np.array([1.0]),
        "thick": np.array([0.01]),
        "off": np.array([1.0]),
        "eint": np.zeros(1),
        "dtfac": 0.9,
        "sig": np.zeros((1, 1, 3)),
        "qshear": np.zeros((1, 2)),
        "epsp": np.zeros((1, 1)),
        "eplane": np.zeros((1, 3, 3)),
        "ehour": np.zeros(1),
        "hgq": np.zeros((1, 5)),
        "ihbe_mask": np.zeros(1, dtype=bool),
        "chk_fail": False,
        "layfail": np.zeros((1, 1), dtype=bool),
        "zw": [(np.array([0.0]), np.array([1.0]))],
        "mat_extra": {},
        "slices": [(slice(0, 1), mat, prop)]
    }
    fint = np.zeros((4, 3))
    mint = np.zeros((4, 3))
    dt_bt4 = shell_bt4.forces(group_bt4, x_bt4, v_bt4, vr_bt4, dt=1e-6, fint=fint, mint=mint)
    assert dt_bt4[0] >= 1e20, f"Collapsed BT4 dt must be >= 1e20, got {dt_bt4[0]}"

    # TRI3: 3 nodes, all at origin -> zero area collapsed element
    x_tri = np.zeros((3, 3), dtype=np.float64)
    v_tri = np.zeros((3, 3), dtype=np.float64)
    vr_tri = np.zeros((3, 3), dtype=np.float64)
    conn_tri = np.array([[0, 1, 2]], dtype=np.int64)

    group_tri = ElementGroup(ids=np.array([1]), conn=conn_tri, part=np.array([1]))
    group_tri.state = {
        "mass": np.array([1.0]),
        "thick": np.array([0.01]),
        "off": np.array([1.0]),
        "eint": np.zeros(1),
        "dtfac": 0.9,
        "sig": np.zeros((1, 1, 3)),
        "qshear": np.zeros((1, 2)),
        "epsp": np.zeros((1, 1)),
        "chk_fail": False,
        "layfail": np.zeros((1, 1), dtype=bool),
        "zw": [(np.array([0.0]), np.array([1.0]))],
        "mat_extra": {},
        "slices": [(slice(0, 1), mat, prop)]
    }
    fint_tri = np.zeros((3, 3))
    mint_tri = np.zeros((3, 3))
    dt_tri = shell_tri3.forces(group_tri, x_tri, v_tri, vr_tri, dt=1e-6, fint=fint_tri, mint=mint_tri)
    assert dt_tri[0] >= 1e20, f"Collapsed TRI3 dt must be >= 1e20, got {dt_tri[0]}"


# -----------------------------------------------------------------------------
# 3. BUG-ELEM-03: bric20 3D dilatational sound speed
# -----------------------------------------------------------------------------
def test_bug_elem_03_bric20_3d_dilatational_sound_speed():
    """Verify solid_bric20 uses 3D dilatational sound speed sqrt((K+4G/3)/rho)."""
    from pyradioss.elements import solid_bric20

    E = 2.1e11
    nu = 0.3
    rho0 = 7800.0

    K = E / (3.0 * max(1.0 - 2.0 * nu, 1e-4))
    G = E / (2.0 * (1.0 + nu))
    expected_c = np.sqrt((K + 4.0 * G / 3.0) / rho0)
    bar_speed = np.sqrt(E / rho0)

    # 3D dilatational speed should be strictly higher than 1D bar speed
    assert expected_c > bar_speed

    # 20 nodes for standard reference cube [-1, 1]^3 using solid_bric20._XI_NODES
    x = solid_bric20._XI_NODES.copy()

    conn = np.arange(20, dtype=np.int64).reshape(1, 20)
    group = ElementGroup(ids=np.array([1]), conn=conn, part=np.array([1]))
    mat = types.SimpleNamespace(id=1, law=1, rho0=rho0, E=E, nu=nu, G=G, K=K, eos=None, params={"E": E, "nu": nu})
    prop = types.SimpleNamespace(id=1, type=14, qa=1.1, qb=0.05, params={})

    group.state = {
        "mass": np.array([8.0 * rho0]),
        "vol0": np.array([8.0]),
        "off": np.array([1.0]),
        "eint": np.zeros(1),
        "dtfac": 0.9,
        "sig": np.zeros((1, 8, 6)),
        "epsp": np.zeros((1, 8)),
        "qvw_pend": np.zeros(1),
        "slices": [(slice(0, 1), mat, prop)]
    }

    fint = np.zeros((20, 3))
    mint = np.zeros((20, 3))
    v = np.zeros((20, 3))
    vr = np.zeros((20, 3))

    dt_crit = solid_bric20.forces(group, x, v, vr, dt=1e-5, fint=fint, mint=mint)
    lc = solid_bric20._char_length(np.array([8.0]))
    expected_dt = 0.9 * lc[0] / expected_c
    assert np.isclose(dt_crit[0], expected_dt, rtol=1e-4)


# -----------------------------------------------------------------------------
# 4. BUG-ELEM-04: shell_bt4 hourglass parameter defaults
# -----------------------------------------------------------------------------
def test_bug_elem_04_shell_bt4_hourglass_defaults():
    """Verify shell_bt4 handles property missing hm/hf/hr without KeyError."""
    from pyradioss.elements import shell_bt4

    x = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [1.0, 1.0, 0.0],
        [0.0, 1.0, 0.0]
    ])
    v = np.zeros((4, 3))
    vr = np.zeros((4, 3))
    conn = np.array([[0, 1, 2, 3]], dtype=np.int64)

    mat = types.SimpleNamespace(
        id=1, law=1, rho0=7800.0, E=2.1e11, nu=0.3,
        G=2.1e11 / (2.0 * 1.3), K=2.1e11 / (3.0 * 0.4),
        eos=None, params={"E": 2.1e11, "nu": 0.3}
    )
    # Empty params dict: no hm, hf, hr keys provided!
    prop = types.SimpleNamespace(id=1, type=1, thick=0.01, params={"thick": 0.01})

    group = ElementGroup(ids=np.array([1]), conn=conn, part=np.array([1]))
    group.state = {
        "mass": np.array([78.0]),
        "thick": np.array([0.01]),
        "off": np.array([1.0]),
        "eint": np.zeros(1),
        "dtfac": 0.9,
        "sig": np.zeros((1, 1, 3)),
        "qshear": np.zeros((1, 2)),
        "epsp": np.zeros((1, 1)),
        "eplane": np.zeros((1, 3, 3)),
        "ehour": np.zeros(1),
        "hgq": np.zeros((1, 5)),
        "ihbe_mask": np.zeros(1, dtype=bool),
        "chk_fail": False,
        "layfail": np.zeros((1, 1), dtype=bool),
        "zw": [(np.array([0.0]), np.array([1.0]))],
        "mat_extra": {},
        "slices": [(slice(0, 1), mat, prop)]
    }
    fint = np.zeros((4, 3))
    mint = np.zeros((4, 3))

    # Must execute cleanly without KeyError
    dt_crit = shell_bt4.forces(group, x, v, vr, dt=1e-6, fint=fint, mint=mint)
    assert np.isfinite(dt_crit[0])
    assert dt_crit[0] > 0.0


# -----------------------------------------------------------------------------
# 5. BUG-ELEM-05: tetra4/heph/tetra10 sound speed radicand clamp
# -----------------------------------------------------------------------------
def test_bug_elem_05_sound_speed_radicand_clamp():
    """Verify tetra4, heph, and tetra10 handle negative moduli sum without NaN."""
    from pyradioss.elements import solid_tetra4, solid_heph, solid_tetra10

    # Material with degraded / negative K and G
    mat = types.SimpleNamespace(id=1, law=1, rho0=7800.0, E=0.0, nu=0.0, K=-1.0e9, G=-1.0e9, eos=None, params={})
    prop = types.SimpleNamespace(id=1, type=14, qa=1.1, qb=0.05, params={})

    # TETRA4
    x4 = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0]
    ])
    conn4 = np.array([[0, 1, 2, 3]], dtype=np.int64)
    g4 = ElementGroup(ids=np.array([1]), conn=conn4, part=np.array([1]))
    g4.state = {
        "mass": np.array([1300.0]),
        "vol0": np.array([1.0 / 6.0]),
        "off": np.array([1.0]),
        "eint": np.zeros(1),
        "dtfac": 0.9,
        "sig": np.zeros((1, 6)),
        "epsp": np.zeros(1),
        "chk_fail": False,
        "qvw_pend": np.zeros(1),
        "slices": [(slice(0, 1), mat, prop)]
    }
    f4 = np.zeros((4, 3))
    m4 = np.zeros((4, 3))
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        dt4 = solid_tetra4.forces(g4, x4, np.zeros((4, 3)), np.zeros((4, 3)), dt=1e-5, fint=f4, mint=m4)
        assert np.isfinite(dt4[0])
        assert not any("invalid value encountered in sqrt" in str(item.message) for item in w)

    # HEPH (8 nodes)
    x8 = np.array([
        [0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],
        [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]
    ], dtype=float)
    conn8 = np.arange(8, dtype=np.int64).reshape(1, 8)
    g8 = ElementGroup(ids=np.array([1]), conn=conn8, part=np.array([1]))
    g8.state = {
        "mass": np.array([7800.0]),
        "vol0": np.array([1.0]),
        "off": np.array([1.0]),
        "eint": np.zeros(1),
        "ehour": np.zeros(1),
        "dtfac": 0.9,
        "sig": np.zeros((1, 6)),
        "epsp": np.zeros(1),
        "qvw_pend": np.zeros(1),
        "hgqex": np.zeros((1, 4, 3)),
        "lc_scale": np.ones(1),
        "chk_fail": False,
        "mat_extra": {},
        "slices": [(slice(0, 1), mat, prop)]
    }
    f8 = np.zeros((8, 3))
    m8 = np.zeros((8, 3))
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        dt8 = solid_heph.forces(g8, x8, np.zeros((8, 3)), np.zeros((8, 3)), dt=1e-5, fint=f8, mint=m8)
        assert np.isfinite(dt8[0])
        assert not any("invalid value encountered in sqrt" in str(item.message) for item in w)

    # TETRA10 (10 nodes)
    x10 = np.zeros((10, 3))
    x10[:4] = x4
    for m, (n1, n2) in enumerate([(0, 1), (1, 2), (2, 0), (0, 3), (1, 3), (2, 3)]):
        x10[4 + m] = 0.5 * (x4[n1] + x4[n2])
    conn10 = np.arange(10, dtype=np.int64).reshape(1, 10)
    g10 = ElementGroup(ids=np.array([1]), conn=conn10, part=np.array([1]))
    g10.state = {
        "mass": np.array([1300.0]),
        "vol0": np.array([1.0 / 6.0]),
        "off": np.array([1.0]),
        "eint": np.zeros(1),
        "dtfac": np.array([0.9]),
        "sig": np.zeros((1, 4, 6)),
        "epsp": np.zeros((1, 4)),
        "chk_fail": False,
        "qvw_pend": np.zeros(1),
        "slices": [(slice(0, 1), mat, prop)]
    }
    f10 = np.zeros((10, 3))
    m10 = np.zeros((10, 3))
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        dt10 = solid_tetra10.forces(g10, x10, np.zeros((10, 3)), np.zeros((10, 3)), dt=1e-5, fint=f10, mint=m10)
        assert np.isfinite(dt10[0])
        assert not any("invalid value encountered in sqrt" in str(item.message) for item in w)


# -----------------------------------------------------------------------------
# 6. BUG-ELEM-06: Windows OpenMP threading stability
# -----------------------------------------------------------------------------
def test_bug_elem_06_threading_stability_no_parallel_jit():
    """Verify critical JIT element kernels do not use parallel=True."""
    from pyradioss.accel import jit_kernels as jk

    kernels = [
        jk.hexa_pre, jk.hexa_post,
        jk.shell_pre, jk.shell_post,
        jk.tetra10_pre, jk.tetra10_post
    ]
    for kern in kernels:
        if kern is not None and hasattr(kern, "targetoptions"):
            assert not kern.targetoptions.get("parallel", False), (
                f"Kernel {kern.__name__} has parallel=True, which causes Windows OpenMP crashes."
            )


# -----------------------------------------------------------------------------
# 7. BUG-ELEM-07: shell_thick16 Courant factor
# -----------------------------------------------------------------------------
def test_bug_elem_07_thick16_courant_factor():
    """Verify shell_thick16 correctly scales critical dt by dtfac."""
    from pyradioss.elements import shell_thick16

    x = np.zeros((16, 3))
    for i in range(8):
        x[i] = [i % 4, i // 4, 0.0]
        x[8 + i] = [i % 4, i // 4, 1.0]

    conn = np.arange(16, dtype=np.int64).reshape(1, 16)
    mat = types.SimpleNamespace(
        id=1, law=1, rho0=7800.0, E=2.1e11, nu=0.3,
        G=2.1e11 / (2.0 * (1.0 + 0.3)), K=2.1e11 / (3.0 * (1.0 - 2.0 * 0.3)),
        eos=None, params={"E": 2.1e11, "nu": 0.3}
    )
    prop = types.SimpleNamespace(id=1, type=16, thick=1.0, params={"thick": 1.0})

    g1 = ElementGroup(ids=np.array([1]), conn=conn, part=np.array([1]))
    g1.state = {
        "mass": np.array([7800.0]),
        "thick": np.array([1.0]),
        "off": np.array([1.0]),
        "lc": np.array([1.0]),
        "eint": np.zeros(1),
        "dtfac": 0.9,
        "sig": np.zeros((1, 1, 6)),
        "epsp": np.zeros((1, 1)),
        "zw": [([(0.0, 0.0, 0.0)], [1.0])],
        "slices": [(slice(0, 1), mat, prop)]
    }

    g2 = ElementGroup(ids=np.array([1]), conn=conn, part=np.array([1]))
    g2.state = dict(g1.state)
    g2.state["dtfac"] = 0.45

    fint1 = np.zeros((16, 3))
    fint2 = np.zeros((16, 3))
    mint1 = np.zeros((16, 3))
    mint2 = np.zeros((16, 3))
    v = np.zeros((16, 3))
    vr = np.zeros((16, 3))

    dt1 = shell_thick16.forces(g1, x, v, vr, dt=1e-6, fint=fint1, mint=mint1)
    dt2 = shell_thick16.forces(g2, x, v, vr, dt=1e-6, fint=fint2, mint=mint2)

    assert np.isclose(dt1[0] / dt2[0], 2.0, rtol=1e-5), f"Ratio was {dt1[0] / dt2[0]}, expected 2.0"


# -----------------------------------------------------------------------------
# 8. BUG-ELEM-08: auto_backend exception fallback
# -----------------------------------------------------------------------------
def test_bug_elem_08_auto_backend_exception_fallback(monkeypatch):
    """Verify backend selection gracefully falls back to NumPy on errors."""
    import pyradioss.accel as accel

    for err_cls in (RuntimeError, TypeError, NameError):
        def mock_raise():
            raise err_cls("Simulated JIT compilation or load error")

        monkeypatch.setattr(accel, "_load_numba_module", mock_raise)

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            backend = accel.select_backend("numba")
            assert backend == "numpy"
            assert any("falling back to NumPy" in str(item.message) for item in w)

        dummy_model = Model()
        dummy_model.element_groups_list = []
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            backend_auto = accel.auto_select_backend(dummy_model)
            assert backend_auto == "numpy"
