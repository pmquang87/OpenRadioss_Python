"""M39 OPTIMIZER-2 parity: the LAW70 physical-hourglass numba mirror.

``accel.jit_kernels.hexa_hgphys`` mirrors
``solid_hexa8._phys_hourglass_law70`` (the M38 Belytschko-Bindeman
stiffness hourglass for LAW70 foam bricks — the c46 regime's #1 self-time
block under the numba backend). Same M7 parity contract as the other
mirrors: element-wise expressions bitwise; the 4-/8-term dot products are
sequential where NumPy uses matmul/einsum, bounded at ~1e-15 relative per
call, so the kernel test asserts rtol=1e-12 (see tests/test_m7_backends.py
and the pyradioss/accel package docstring).

These tests synthesize a distorted brick group directly (like
test_t7_narrow_parity) so they exercise the mirror with NO deck / CFG /
starter dependency, and cover BOTH the all-LAW70 branch (q += ... over
the whole group, the c46 case) and the mixed-mask branch (q[mask] += ...,
which the all-foam c46 deck never hits)."""

import numpy as np
import pytest

# is the optional numba backend importable on this machine?
try:
    import pyradioss.accel.jit_kernels  # noqa: F401
    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False

needs_numba = pytest.mark.skipif(
    not HAS_NUMBA, reason="numba not installed (optional dependency)")

_BASE = np.array([
    [0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],
    [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]], dtype=float)


def _group(n, seed):
    """A distorted n-brick group with realistic shape gradients (from the
    reference _pre) plus positive mass/vol0/c and an initial hourglass
    modal state — everything hexa_hgphys / _phys_hourglass_law70 read."""
    from pyradioss.elements import solid_hexa8 as hx
    rng = np.random.default_rng(seed)
    xe = _BASE[None] + 0.18 * rng.standard_normal((n, 8, 3))
    ve = 0.4 * rng.standard_normal((n, 8, 3))
    dt = 7e-4
    dndx, vol, lc, deps, trD = hx._pre(xe, ve, np.zeros((n, 6)), dt,
                                       np.ones(n), np.ones(n))
    c = 1.0 + np.abs(rng.standard_normal(n))
    mass = 0.5 + np.abs(rng.standard_normal(n))
    vol0 = vol * (0.9 + 0.2 * rng.random(n))
    q0 = 0.05 * rng.standard_normal((n, 4, 3))
    return xe, ve, dndx, vol, c, mass, vol0, q0, dt


def _run_both(xe, ve, dndx, vol, c, mask, mass, vol0, q0, dt):
    from pyradioss.accel import jit_kernels as jk
    from pyradioss.elements import solid_hexa8 as hx
    st = {"law70_mask": mask, "mass": mass, "vol0": vol0,
          "hgqex": q0.copy()}
    f_np, deh_np, dth_np = hx._phys_hourglass_law70(
        st, xe, ve, dndx, vol, c, dt)
    q_np = st["hgqex"]
    q_nb = q0.copy()
    f_nb, deh_nb, dth_nb = jk.hexa_hgphys(
        xe, ve, dndx, vol, c, mask, mass, vol0, q_nb, dt)
    return (f_np, deh_np, dth_np, q_np), (f_nb, deh_nb, dth_nb, q_nb)


@needs_numba
def test_hgphys_constants_match_reference():
    """The mirror duplicates HG_PHYS/DT_HG_SF as frozen literals (numba
    globals are compile-time constants); they MUST equal the reference."""
    from pyradioss.accel import jit_kernels as jk
    from pyradioss.elements import solid_hexa8 as hx
    assert jk._HG_PHYS == hx.HG_PHYS
    assert jk._DT_HG_SF == hx.DT_HG_SF


@needs_numba
@pytest.mark.parametrize("mask_kind", ["all", "mixed"])
def test_hgphys_kernel_parity(mask_kind):
    """hexa_hgphys == _phys_hourglass_law70 to rtol 1e-12 on force,
    energy increment, dt cap AND the mutated modal state q."""
    xe, ve, dndx, vol, c, mass, vol0, q0, dt = _group(6, seed=70)
    n = len(xe)
    if mask_kind == "all":
        mask = np.ones(n, dtype=bool)
    else:
        mask = np.array([True, True, False, True, False, True])
    (f_np, deh_np, dth_np, q_np), (f_nb, deh_nb, dth_nb, q_nb) = _run_both(
        xe, ve, dndx, vol, c, mask, mass, vol0, q0, dt)
    assert np.allclose(f_np, f_nb, rtol=1e-12, atol=1e-15)
    assert np.allclose(deh_np, deh_nb, rtol=1e-12, atol=1e-15)
    assert np.allclose(dth_np, dth_nb, rtol=1e-12, atol=1e-15)
    assert np.allclose(q_np, q_nb, rtol=1e-12, atol=1e-15)
    # non-LAW70 elements: exactly zero force, EP30 dt (both backends)
    from pyradioss.common.constants import EP30
    if mask_kind == "mixed":
        assert np.array_equal(f_nb[~mask], np.zeros_like(f_nb[~mask]))
        assert np.all(dth_nb[~mask] == EP30)


@needs_numba
def test_hgphys_multicycle_accumulation_parity():
    """The modal state q accumulates across cycles IN PLACE (like sig in
    hexa_pre). Drive 60 cycles with a live hourglass velocity on a
    densifying group and confirm the two backends' q / ledger stay within
    the amplified-ulp band (no secular divergence from the mirror)."""
    from pyradioss.accel import jit_kernels as jk
    from pyradioss.elements import solid_hexa8 as hx
    xe, ve, dndx, vol, c, mass, vol0, q0, dt = _group(4, seed=7)
    mask = np.ones(len(xe), dtype=bool)

    st = {"law70_mask": mask, "mass": mass, "vol0": vol0,
          "hgqex": q0.copy()}
    q_nb = q0.copy()
    eh_np = 0.0
    eh_nb = 0.0
    rng = np.random.default_rng(3)
    for _ in range(60):
        vh = 0.2 * rng.standard_normal((len(xe), 8, 3))
        f_np, deh_np, _ = hx._phys_hourglass_law70(
            st, xe, vh, dndx, vol, c, dt)
        f_nb, deh_nb, _ = jk.hexa_hgphys(
            xe, vh, dndx, vol, c, mask, mass, vol0, q_nb, dt)
        eh_np += float(deh_np.sum())
        eh_nb += float(deh_nb.sum())
        assert np.allclose(f_np, f_nb, rtol=1e-10, atol=1e-14)
    # accumulated modal state and energy ledger track to well within the
    # backend tolerance the M7 full-run test allows (1e-8)
    assert np.allclose(st["hgqex"], q_nb, rtol=1e-10, atol=1e-13)
    assert abs(eh_np - eh_nb) <= 1e-10 * max(abs(eh_np), 1e-12)


# ============================================================================
# LAW70 material numeric leaves — BITWISE parity (no reduction to reassociate)
# ============================================================================

@needs_numba
def test_law70_snorm_enorm_bitwise():
    """law70_snorm/law70_enorm == the NumPy Voigt norms bit-for-bit
    (single per-element sqrt expression — nothing for a backend to
    reassociate)."""
    from pyradioss.accel import jit_kernels as jk
    from pyradioss.materials import law70_tabfoam as L
    rng = np.random.default_rng(5)
    v = 0.7 * rng.standard_normal((257, 6))
    assert np.array_equal(L._snorm(v), jk.law70_snorm(v))
    assert np.array_equal(L._enorm(v), jk.law70_enorm(v))


@needs_numba
def test_law70_elastic_stress_bitwise():
    """law70_elastic_stress == the NumPy C(E):eps map bit-for-bit."""
    from pyradioss.accel import jit_kernels as jk
    from pyradioss.materials import law70_tabfoam as L
    rng = np.random.default_rng(6)
    n = 300
    aa1 = 1.0 + np.abs(rng.standard_normal(n))
    aa2 = 0.3 * rng.standard_normal(n)
    g = 0.5 + np.abs(rng.standard_normal(n))
    e = 0.2 * rng.standard_normal((n, 6))
    assert np.array_equal(L._elastic_stress(aa1, aa2, g, e),
                          jk.law70_elastic_stress(aa1, aa2, g, e))


@needs_numba
@pytest.mark.parametrize("nrate", [1, 4])
def test_law70_tab2d_bitwise(nrate):
    """law70_tab2d == the NumPy _tab2d bilinear table lookup bit-for-bit,
    over queries that hit the interior, both end-slope EXTRAPOLATION tails,
    and (nrate>1) the rate dimension."""
    from pyradioss.accel import jit_kernels as jk
    from pyradioss.materials import law70_tabfoam as L
    rng = np.random.default_rng(70 + nrate)
    xg = np.array([0.0, 0.1, 0.3, 0.6, 1.0])
    rates = np.array([0.0]) if nrate == 1 else \
        np.array([0.0, 1.0, 10.0, 100.0])
    Y = rng.standard_normal((len(xg), len(rates)))
    # queries spanning below xg[0], inside, and above xg[-1] (extrapolation)
    x = rng.uniform(-0.5, 1.5, 400)
    r = rng.uniform(-5.0, 150.0, 400)
    a = L._tab2d(xg, rates, Y, x, r)
    b = jk.law70_tab2d(xg, rates, Y, x, r)
    assert np.array_equal(a, b)
