"""
Milestone M614 Component 2: Implicit Statics, BFGS Quasi-Newton Solver, Line Search,
Multi-Criterion Convergence & Control Cards.

Fortran reference:
- ``engine/source/implicit/imp_bfgs.F`` (BFGS_INI, BFGS_0, BFGS_LS, BFGS_1, BFGS_2, BFGS_RHD)
- ``engine/source/implicit/nl_solv.F`` (CRIT_ITE, LINE_S, LINE_S1)
- ``engine/source/implicit/imp_dyna.F`` (QSTAT_INI, QSTAT_IT)
- ``engine/source/constraints/general/bcs/bc_imp0.F`` (UPD_ASPC0, AUTSPC)
- ``engine/source/implicit/lin_solv.F`` (/IMPL/LINE)
- ``engine/source/implicit/imp_solv.F`` (ISPRB springback)

Tests:
1. BFGS Quasi-Newton solver:
   - Curvature condition (s^T y > 0) and ill-conditioning rejection.
   - Secant equation H_k y_k = s_k satisfied by quasi-Newton direction.
   - Max stored updates (L_BFGS) sliding window.
   - Two-loop recursion vs vector pair updates.
   - End-to-end BFGS convergence vs standard Newton-Raphson.
2. Line Search:
   - Energy-based line search (ILINE_S = 1, LINE_S1).
   - Force / secant residual line search (ILINE_S = 2, LINE_S).
   - Step bounding [min_step, max_step] and tolerance convergence.
   - Line search activation in nonlinear statics.
3. Multi-criterion convergence check:
   - All NITOL criteria: 1 (energy), 2 (force), 3 (displacement), 12, 13, 23, 123.
   - Divergence handling with TOL_DIV and NDIVER.
4. Control cards:
   - /IMPL/LINE linear static direct solve (single elastic step K u = f_ext).
   - /IMPL/QSTAT quasi-static regularization for near-singular states.
   - /IMPL/AUTOS automatic single point constraint on zero-stiffness DOFs.
   - /IMPL/SPRB spring-back analysis.
"""

import contextlib
import io
import numpy as np
import pytest

from pyradioss.implicit.bfgs import BFGSSolver
from pyradioss.implicit.linesearch import LineSearch, line_s, line_s1
from pyradioss.implicit.convergence import ConvergenceChecker, crit_ite
from pyradioss.starter.starter import run_starter
from pyradioss.engine.engine import run_engine

pytest.importorskip("scipy")


# ----------------------------------------------------------------------------
# Helpers for deck generation
# ----------------------------------------------------------------------------

def _run(make_deck, name, starter, engine):
    s, e = make_deck(name, starter, engine)
    with contextlib.redirect_stdout(io.StringIO()):
        run_starter(s)
        model = run_engine(e)
    return model


STEEL1 = """\
/MAT/LAW1/1
steel elastic
   7.8e-6
     210.0       0.3
"""

STEEL2 = """\
/MAT/LAW2/1
steel jc
   7.8e-6
     210.0       0.3
       0.4       0.5       0.5
"""

SOLIDPROP = """\
/PROP/SOLID/1
solid
       1.1      0.05       0.1
"""


def _single_hex_deck(L=10.0, force=210.0, mat=STEEL1):
    """Single cube hexa8 in tension (uniaxial state)."""
    dx = dy = dz = L
    nodes = [
        f"{1:10d}{0.0:20.10f}{0.0:20.10f}{0.0:20.10f}",
        f"{2:10d}{dx:20.10f}{0.0:20.10f}{0.0:20.10f}",
        f"{3:10d}{dx:20.10f}{dy:20.10f}{0.0:20.10f}",
        f"{4:10d}{0.0:20.10f}{dy:20.10f}{0.0:20.10f}",
        f"{5:10d}{0.0:20.10f}{0.0:20.10f}{dz:20.10f}",
        f"{6:10d}{dx:20.10f}{0.0:20.10f}{dz:20.10f}",
        f"{7:10d}{dx:20.10f}{dy:20.10f}{dz:20.10f}",
        f"{8:10d}{0.0:20.10f}{dy:20.10f}{dz:20.10f}",
    ]
    brick = f"{1:10d}" + "".join(f"{i:10d}" for i in range(1, 9))
    fpn = force / 4.0

    starter = f"""\
#RADIOSS STARTER
/BEGIN
HEX1
/NODE
{chr(10).join(nodes)}
/BRICK/1
{brick}
/PART/1
cube
         1         1
{mat}{SOLIDPROP}/GRNOD/NODE/1
x0
1 4 5 8
/GRNOD/NODE/2
x1
2 3 6 7
/GRNOD/NODE/3
y0
1 2 5 6
/GRNOD/NODE/4
z0
1 2 3 4
/BCS/1
fx
       100       000         0         1
/BCS/2
fy
       010       000         0         3
/BCS/3
fz
       001       000         0         4
/FUNCT/1
ramp
       0.0       0.0
     100.0     100.0
/CLOAD/1
pull
         1         X         2       {fpn}
/END
"""
    return starter


# ----------------------------------------------------------------------------
# 1. BFGS Quasi-Newton Unit Tests
# ----------------------------------------------------------------------------

def test_bfgs_unit_secant_and_curvature():
    """Verify BFGS curvature condition check, secant condition H_k * y = s,
    and sliding window bounds."""
    solver = BFGSSolver(max_bfgs=3, curv_tol=1e-10)
    assert solver.num_updates == 0

    # 1. Curvature condition violation (negative or near-zero curvature)
    s_bad = np.array([1.0, 0.0])
    y_bad = np.array([-1.0, 0.0])  # s^T y = -1 < 0
    ok = solver.add_update(s_bad, y_bad)
    assert not ok
    assert solver.num_updates == 0
    assert solver.num_skipped == 1

    # 2. Valid positive curvature updates
    # Generate positive definite Hessian H
    H = np.array([[3.0, 1.0], [1.0, 2.0]])
    s1 = np.array([0.5, -0.2])
    y1 = H @ s1
    ok1 = solver.add_update(s1, y1)
    assert ok1
    assert solver.num_updates == 1

    # Verify secant equation: H_k y1 = s1
    # When base solver is Identity, BFGS should map y1 to s1
    def identity_solve(q):
        return q.copy()

    s_rec = solver.solve(identity_solve, y1)
    np.testing.assert_allclose(s_rec, s1, rtol=1e-6, atol=1e-10)

    # Test vector pair update matches two-loop recursion
    s_rec_vp = solver.solve_vector_pairs(identity_solve, y1)
    np.testing.assert_allclose(s_rec_vp, s1, rtol=1e-6, atol=1e-10)

    # 3. Sliding window capacity (max_bfgs = 3)
    s2 = np.array([0.1, 0.4])
    y2 = H @ s2
    solver.add_update(s2, y2)

    s3 = np.array([-0.3, 0.2])
    y3 = H @ s3
    solver.add_update(s3, y3)
    assert solver.num_updates == 3

    # Add 4th update -> should drop oldest and stay at size 3
    s4 = np.array([0.2, 0.1])
    y4 = H @ s4
    solver.add_update(s4, y4)
    assert solver.num_updates == 3

    # Reset
    solver.reset()
    assert solver.num_updates == 0


def test_bfgs_vs_newton_convergence(make_deck):
    """Compare BFGS quasi-Newton against standard Newton-Raphson on a static pull."""
    starter = _single_hex_deck(L=10.0, force=210.0, mat=STEEL1)

    # 1. Standard Newton-Raphson
    eng_newton = """\
#RADIOSS ENGINE
/RUN/NEWT/1
1.0
/IMPL
/IMPL/DTINI
0.5
/IMPL/NEWTON
1.0e-7 20
/END
"""
    model_newton = _run(make_deck, "NEWT", starter, eng_newton)
    assert model_newton.implicit_result.converged

    # 2. BFGS Quasi-Newton
    eng_bfgs = """\
#RADIOSS ENGINE
/RUN/BFGS/1
1.0
/IMPL
/IMPL/LBFGS/10
/IMPL/DTINI
0.5
/IMPL/NEWTON
1.0e-7 20
/END
"""
    model_bfgs = _run(make_deck, "BFGS", starter, eng_bfgs)
    assert model_bfgs.implicit_result.converged

    # Displacements must match to high precision
    np.testing.assert_allclose(model_bfgs.x, model_newton.x, rtol=1e-5, atol=1e-8)


# ----------------------------------------------------------------------------
# 2. Line Search Unit and Integration Tests
# ----------------------------------------------------------------------------

def test_line_search_unit_algorithms():
    """Verify LINE_S and LINE_S1 step cutting, bounding and interpolation."""
    # 1. LINE_S: step cut when residual increases (r1 > 1.0)
    s_cut, s0, r0, idiv, imconv = line_s(r0=1.0, s0=0.0, r1=2.0, s=1.0, dtol=0.1)
    assert s_cut < 1.0, f"Expected step cut on divergence, got {s_cut}"

    # 2. LINE_S: convergence when residual is small (r1 < dtol)
    s_conv, _, _, _, imconv_ok = line_s(r0=1.0, s0=0.0, r1=0.05, s=1.0, dtol=0.1)
    assert imconv_ok == 0, "Expected line search completion when r1 < dtol"

    # 3. LINE_S1: directional derivative sign change triggers bracketing
    # E1 < 0 -> bracket found, s = 0.5
    s_brk, ep, sp, en, sn, idiv1, imconv1 = line_s1(
        e1=-0.4, s=1.0, ep=1.0, sp=0.0, en=0.0, sn=0.0, idiv=0
    )
    assert s_brk == 0.5
    assert en == -0.4

    # 4. LineSearch runner on a non-monotone residual
    def nonmonotone_res(u, ur):
        val = (u[0] - 0.4)**2 + 0.01
        return None, None, np.array([val])

    ls = LineSearch(method=2, tol=0.1, max_iter=4)
    u0 = np.array([0.0])
    ur0 = np.array([0.0])
    du = np.array([1.0])
    dur = np.array([0.0])
    R0 = np.array([0.17])  # at alpha=0, (0-0.4)^2 + 0.01 = 0.17

    alpha, u_acc, ur_acc, _, _, R_acc, n_ev = ls.search(
        nonmonotone_res, u0, ur0, du, dur, R0
    )
    assert n_ev >= 1
    assert float(np.linalg.norm(R_acc)) < float(np.linalg.norm(R0))


def test_line_search_activation(make_deck):
    """Run implicit statics with /IMPL/LSEAR activated."""
    starter = _single_hex_deck(L=10.0, force=210.0, mat=STEEL1)

    engine_lsear = """\
#RADIOSS ENGINE
/RUN/LSEAR/1
1.0
/IMPL
/IMPL/LSEAR/1
4  0.2
/IMPL/DTINI
1.0
/IMPL/NEWTON
1.0e-7 20
/END
"""
    model = _run(make_deck, "LSEAR", starter, engine_lsear)
    assert model.implicit_result.converged
    # Check that Hooke's law is satisfied: delta = F * L / (E * A)
    # L=10, A=100, E=210, F=210 => delta = 210 * 10 / (210 * 100) = 0.1
    dx = model.x[1, 0] - model.x0[1, 0]
    np.testing.assert_allclose(dx, 0.1, rtol=1e-4)


def test_linesearch_softening_snapthrough():
    """Verify Line Search prevents divergence on a nonlinear softening / snap-through response."""
    # Nonlinear cubic softening element: R(u) = F_ext - (k0 * u - beta * u^3)
    k0 = 100.0
    beta = 50.0
    F_ext = 30.0

    def residual(u, ur):
        ut = u[0]
        f_int = k0 * ut - beta * (ut**3)
        R = F_ext - f_int
        return None, None, np.array([R])

    u = np.array([0.5])
    ur = np.array([0.0])
    _, _, R0 = residual(u, ur)
    # A large trial step along softening branch leads to residual explosion
    du = np.array([1.0])
    dur = np.array([0.0])

    _, _, R_full = residual(u + 1.0 * du, ur)
    assert np.abs(R_full[0]) > np.abs(R0[0]), "Full step must increase residual on softening branch"

    ls = LineSearch(method=2, tol=0.1, max_iter=5)
    alpha, u_acc, ur_acc, _, _, R_acc, n_ev = ls.search(residual, u, ur, du, dur, R0)
    assert alpha < 1.0, f"Line search must cut step, got {alpha}"
    assert np.abs(R_acc[0]) < np.abs(R_full[0]), "Line search must reduce residual relative to full step"


def test_bfgs_linesearch_nlgeom(make_deck):
    """Verify BFGS and Line Search combined under nonlinear geometry (/IMPL/NONLIN)."""
    starter = _single_hex_deck(L=10.0, force=210.0, mat=STEEL1)

    # 1. Standard Newton-Raphson under /IMPL/NONLIN
    engine_newton = """\
#RADIOSS ENGINE
/RUN/NEWTON_NL/1
1.0
/IMPL/NONLIN
/IMPL/DTINI
0.5
/IMPL/NEWTON
1.0e-7 20
/END
"""
    model_newton = _run(make_deck, "NEWTON_NL", starter, engine_newton)
    assert model_newton.implicit_result.converged

    # 2. BFGS + Line Search under /IMPL/NONLIN
    engine_bfgs_ls = """\
#RADIOSS ENGINE
/RUN/BFGS_LS/1
1.0
/IMPL/NONLIN
/IMPL/LBFGS/5
/IMPL/LSEAR/1
4  0.2
/IMPL/DTINI
0.5
/IMPL/NEWTON
1.0e-7 20
/END
"""
    model_bfgs = _run(make_deck, "BFGS_LS", starter, engine_bfgs_ls)
    assert model_bfgs.implicit_result.converged

    # BFGS + Line Search must match full Newton on nonlinear geometry
    np.testing.assert_allclose(model_bfgs.x, model_newton.x, rtol=1e-5, atol=1e-8)
    dx = model_bfgs.x[1, 0] - model_bfgs.x0[1, 0]
    np.testing.assert_allclose(dx, 0.1, rtol=0.02)




# ----------------------------------------------------------------------------
# 3. Multi-Criterion Convergence (NITOL 1, 2, 3, 12, 13, 23, 123) & Divergence
# ----------------------------------------------------------------------------

def test_convergence_all_nitol_criteria():
    """Verify each NITOL convergence criterion and divergence handling in CRIT_ITE."""
    # Base inputs:
    # ru: displacement ratio
    # rr: force residual ratio
    # er: energy ratio

    # NITOL = 1 (energy only): small er -> converged
    imconv1, _, _ = crit_ite(it=2, ur=0.5, rr=0.1, er=1e-5, ndiv=0, tol=1e-4, nitol=1)
    assert imconv1 == 1

    # NITOL = 2 (force only): small rr -> converged
    imconv2, _, _ = crit_ite(it=2, ur=0.5, rr=1e-5, er=0.5, ndiv=0, tol=1e-4, nitol=2)
    assert imconv2 == 1

    # NITOL = 3 (displacement only): small ur -> converged
    imconv3, _, _ = crit_ite(it=2, ur=1e-5, rr=0.5, er=0.5, ndiv=0, tol=1e-4, nitol=3)
    assert imconv3 == 1

    # NITOL = 12 (force and energy): both must satisfy tolerance
    imconv12_ok, _, _ = crit_ite(
        it=2, ur=0.5, rr=5e-4, er=5e-5, ndiv=0, tol=1e-3, nitol=12, n_tole=1e-4, n_tolf=1e-3
    )
    assert imconv12_ok == 1

    imconv12_fail, _, _ = crit_ite(
        it=2, ur=0.5, rr=5e-2, er=5e-5, ndiv=0, tol=1e-3, nitol=12, n_tole=1e-4, n_tolf=1e-3
    )
    assert imconv12_fail == 0  # force residual not converged

    # NITOL = 13 (energy and displacement)
    imconv13, _, _ = crit_ite(
        it=2, ur=5e-4, rr=0.5, er=5e-5, ndiv=0, tol=1e-3, nitol=13, n_tole=1e-4, n_tolu=1e-3
    )
    assert imconv13 == 1

    # NITOL = 23 (force and displacement)
    imconv23, _, _ = crit_ite(
        it=2, ur=5e-4, rr=5e-4, er=0.5, ndiv=0, tol=1e-3, nitol=23, n_tolf=1e-3, n_tolu=1e-3
    )
    assert imconv23 == 1

    # NITOL = 123 (energy, force, and displacement)
    imconv123, _, _ = crit_ite(
        it=2, ur=5e-4, rr=5e-4, er=5e-5, ndiv=0, tol=1e-3, nitol=123,
        n_tole=1e-4, n_tolf=1e-3, n_tolu=1e-3
    )
    assert imconv123 == 1

    # Divergence test with /IMPL/DIVER
    # Residual explodes past tol_div=100
    imconv_div, ndiv_div, _ = crit_ite(
        it=2, ur=1.0, rr=500.0, er=1.0, ndiv=0, tol=1e-3, nitol=2,
        tol_div=100.0, ndiver=1
    )
    assert imconv_div == -2, f"Expected divergence flag -2, got {imconv_div}"


def test_all_nitol_deck_parsing_and_runs(make_deck):
    """Test solver execution under NITOL = 1, 2, 3, 12, 123 via /IMPL/NEWTON."""
    starter = _single_hex_deck(L=10.0, force=210.0, mat=STEEL1)

    for nitol in [1, 2, 3, 12, 123]:
        engine = f"""\
#RADIOSS ENGINE
/RUN/NITOL_{nitol}/1
1.0
/IMPL
/IMPL/NEWTON
20  {nitol}  1.0e-5  1.0e-4  1.0e-4
/END
"""
        model = _run(make_deck, f"NITOL_{nitol}", starter, engine)
        assert model.implicit_result.converged
        dx = model.x[1, 0] - model.x0[1, 0]
        np.testing.assert_allclose(dx, 0.1, rtol=1e-3)


# ----------------------------------------------------------------------------
# 4. /IMPL/LINE Linear Static Direct Solve
# ----------------------------------------------------------------------------

def test_impl_line_linear_static_solve(make_deck):
    """Verify /IMPL/LINE performs a single elastic step directly solving K u = f_ext."""
    starter = _single_hex_deck(L=10.0, force=210.0, mat=STEEL1)

    engine_line = """\
#RADIOSS ENGINE
/RUN/LINE/1
1.0
/IMPL/LINE
/END
"""
    model = _run(make_deck, "LINE", starter, engine_line)
    res = model.implicit_result
    assert res.converged
    # Must have completed in exactly 1 increment and 1 iteration
    assert len(res.increments) == 1
    assert res.increments[0].iterations == 1
    # Check Hooke's law: delta = 0.1
    dx = model.x[1, 0] - model.x0[1, 0]
    np.testing.assert_allclose(dx, 0.1, rtol=1e-6)


# ----------------------------------------------------------------------------
# 5. /IMPL/QSTAT Quasi-Static Regularization
# ----------------------------------------------------------------------------

def test_impl_qstat_regularization(make_deck):
    """Verify /IMPL/QSTAT regularizes the stiffness matrix."""
    starter = _single_hex_deck(L=10.0, force=210.0, mat=STEEL1)

    engine_qstat = """\
#RADIOSS ENGINE
/RUN/QSTAT/1
1.0
/IMPL
/IMPL/QSTAT/1
/IMPL/DTINI
1.0
/IMPL/NEWTON
1.0e-6 25
/END
"""
    model = _run(make_deck, "QSTAT", starter, engine_qstat)
    assert model.implicit_result.converged
    dx = model.x[1, 0] - model.x0[1, 0]
    # Regularized solve produces finite positive displacement
    assert dx > 0.0


# ----------------------------------------------------------------------------
# 6. /IMPL/AUTOS Automatic Single Point Constraint
# ----------------------------------------------------------------------------

def test_impl_autos_unconstrained_spc(make_deck):
    """Verify /IMPL/AUTOS automatically constrains zero-stiffness DOFs."""
    starter = _single_hex_deck(L=10.0, force=210.0, mat=STEEL1)

    engine_autos = """\
#RADIOSS ENGINE
/RUN/AUTOS/1
1.0
/IMPL
/IMPL/AUTOS/ALL
/IMPL/NEWTON
1.0e-6 25
/END
"""
    model = _run(make_deck, "AUTOS", starter, engine_autos)
    assert model.implicit_result.converged
    dx = model.x[1, 0] - model.x0[1, 0]
    np.testing.assert_allclose(dx, 0.1, rtol=1e-4)


# ----------------------------------------------------------------------------
# 7. /IMPL/SPRB Spring-Back Analysis
# ----------------------------------------------------------------------------

def test_impl_sprb_springback(make_deck):
    """Verify /IMPL/SPRB zeroes external loads and equilibrates internal state."""
    starter = _single_hex_deck(L=10.0, force=210.0, mat=STEEL1)

    engine_sprb = """\
#RADIOSS ENGINE
/RUN/SPRB/1
1.0
/IMPL
/IMPL/SPRB
/IMPL/NEWTON
1.0e-6 25
/END
"""
    model = _run(make_deck, "SPRB", starter, engine_sprb)
    assert model.implicit_result.converged
    # External load was zeroed, so initial stress-free block remains at zero deflection
    dx = model.x[1, 0] - model.x0[1, 0]
    np.testing.assert_allclose(dx, 0.0, atol=1e-12)
