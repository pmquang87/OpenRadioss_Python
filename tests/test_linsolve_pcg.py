"""
Unit tests for the PCG (Preconditioned Conjugate Gradient) iterative linear solver (imp_pcg.F).

Upstream Fortran reference:
- ``engine/source/implicit/imp_pcg.F`` (IMP_PCGH, CRIT_STOP)
- ``engine/source/implicit/imp_fsa_inv.F`` (SP_STATIC)
- ``engine/source/implicit/imp_fac_ic.F`` (IMP_FAC_ICJ)

Tests:
1. SPD stiffness matrix solve with Jacobi, ILU, and SSOR preconditioners.
2. Comparison of Δu against direct SuperLU solve.
3. Preconditioner selection via argument, string alias, and environment variables.
4. Corner cases: zero RHS, empty system, initial guess.
5. Indefinite / non-SPD fallback to SuperLU.
6. LinearSolver("pcg") execution in static Newton-Raphson simulation.
"""

from __future__ import annotations

import contextlib
import io
import os
import numpy as np
import pytest
import scipy.sparse as sp

from pyradioss.implicit.linsolve import (
    LinearSolver,
    pcg_solve,
    select_linsolve,
    linsolve_name,
)
from pyradioss.starter.starter import run_starter
from pyradioss.engine.engine import run_engine

pytest.importorskip("scipy")


def _generate_spd_stiffness(n: int = 60, seed: int = 42) -> Tuple[sp.csr_matrix, np.ndarray]:
    """Generate a realistic symmetric positive-definite (SPD) sparse stiffness matrix."""
    rng = np.random.default_rng(seed)
    # Tri-diagonal + off-diagonal coupled structure (1D/2D FE elasticity stencil)
    main_diag = 4.0 + rng.uniform(0.5, 2.0, size=n)
    off_diag1 = -1.0 * np.ones(n - 1)
    off_diag5 = -0.3 * np.ones(n - 5)

    diagonals = [main_diag, off_diag1, off_diag1, off_diag5, off_diag5]
    offsets = [0, 1, -1, 5, -5]
    K = sp.diags(diagonals, offsets, shape=(n, n), format="csr")
    # Ensure strict diagonal dominance and symmetry
    K = 0.5 * (K + K.T) + sp.eye(n, format="csr") * 1.5

    R = rng.standard_normal(n) * 100.0
    return K, R


# ----------------------------------------------------------------------------
# 1. SPD Stiffness Matrix Solve with Various Preconditioners
# ----------------------------------------------------------------------------

def test_pcg_spd_jacobi_and_ilu():
    """Verify PCG solver with Jacobi and ILU preconditioners matches SuperLU."""
    K, R = _generate_spd_stiffness(n=80)

    # Reference solution with SuperLU
    solver_direct = LinearSolver("superlu")
    du_ref = solver_direct.solve(K, R)

    # PCG with Jacobi (diagonal) preconditioner (IPREC=1)
    solver_jacobi = LinearSolver("pcg", precond="jacobi", tol=1e-7)
    du_jacobi = solver_jacobi.solve(K, R)

    # PCG with ILU (incomplete LU) preconditioner (IPREC=5)
    solver_ilu = LinearSolver("pcg", precond="ilu", tol=1e-7)
    du_ilu = solver_ilu.solve(K, R)

    # PCG with SSOR preconditioner (IPREC=2)
    solver_ssor = LinearSolver("pcg", precond="ssor", tol=1e-7)
    du_ssor = solver_ssor.solve(K, R)

    # Verify solutions match direct solve closely
    np.testing.assert_allclose(du_jacobi, du_ref, rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(du_ilu, du_ref, rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(du_ssor, du_ref, rtol=1e-5, atol=1e-6)

    # Verify residual norms ||K Δu - R|| / ||R|| < 1e-6
    norm_R = np.linalg.norm(R)
    res_jacobi = np.linalg.norm(K @ du_jacobi - R) / norm_R
    res_ilu = np.linalg.norm(K @ du_ilu - R) / norm_R
    res_ssor = np.linalg.norm(K @ du_ssor - R) / norm_R

    assert res_jacobi < 1e-6
    assert res_ilu < 1e-6
    assert res_ssor < 1e-6


def test_pcg_solve_direct_metadata():
    """Verify pcg_solve function returns convergence metadata and inspect iterations."""
    K, R = _generate_spd_stiffness(n=50)

    x_jacobi, info_jacobi = pcg_solve(K, R, precond="jacobi", tol=1e-6)
    x_ilu, info_ilu = pcg_solve(K, R, precond="ilu", tol=1e-6)

    assert info_jacobi["converged"] is True
    assert info_ilu["converged"] is True
    assert info_jacobi["iterations"] > 0
    assert info_ilu["iterations"] > 0
    # ILU typically clusters eigenvalues better than diagonal Jacobi
    assert info_ilu["iterations"] <= info_jacobi["iterations"]
    assert info_jacobi["rel_residual"] <= 1e-6
    assert info_ilu["rel_residual"] <= 1e-6


# ----------------------------------------------------------------------------
# 2. Corner Cases: Zero RHS, Empty, Initial Guess
# ----------------------------------------------------------------------------

def test_pcg_corner_cases():
    """Verify zero RHS, empty inputs, and non-zero initial guess handling."""
    K, R = _generate_spd_stiffness(n=20)

    # Zero RHS
    x_zero, info_zero = pcg_solve(K, np.zeros(20))
    assert info_zero["converged"] is True
    assert info_zero["iterations"] == 0
    np.testing.assert_allclose(x_zero, 0.0)

    # Empty matrix/vector
    x_empty, info_empty = pcg_solve(sp.csr_matrix((0, 0)), np.zeros(0))
    assert info_empty["converged"] is True
    assert len(x_empty) == 0

    # Initial guess close to solution
    solver_direct = LinearSolver("superlu")
    du_exact = solver_direct.solve(K, R)
    x_init = du_exact + np.random.default_rng(0).normal(0.0, 1e-5, size=20)
    x_pcg, info_init = pcg_solve(K, R, x0=x_init, tol=1e-6)
    assert info_init["converged"] is True
    np.testing.assert_allclose(x_pcg, du_exact, rtol=1e-5, atol=1e-6)


# ----------------------------------------------------------------------------
# 3. Preconditioner Selection, String Aliases, and Env Vars
# ----------------------------------------------------------------------------

def test_pcg_selection_and_environment(monkeypatch):
    """Verify configuration via environment variables and string aliases."""
    # Backend selection
    assert select_linsolve("pcg") == "pcg"
    assert select_linsolve("pcg_ilu") == "pcg_ilu"
    assert select_linsolve("pcg:ssor") == "pcg:ssor"
    assert select_linsolve("1") == "pcg"  # OpenRadioss ISOLV=1

    # LinearSolver precond parsing from name
    s1 = LinearSolver("pcg_ilu")
    assert s1.precond == "ilu"

    s2 = LinearSolver("pcg:ssor")
    assert s2.precond == "ssor"

    s3 = LinearSolver("pcg_diag")
    assert s3.precond == "jacobi"

    # Integer IPREC support
    s4 = LinearSolver("pcg", precond=1)
    assert s4.precond == "jacobi"
    s5 = LinearSolver("pcg", precond=5)
    assert s5.precond == "ilu"
    s6 = LinearSolver("pcg", precond=2)
    assert s6.precond == "ssor"

    # Environment variables
    monkeypatch.setenv("PYRADIOSS_LINSOLVE", "pcg")
    monkeypatch.setenv("PYRADIOSS_PCG_PRECOND", "ilu")
    s_env = LinearSolver()
    assert s_env.name == "pcg"
    assert s_env.precond == "ilu"


# ----------------------------------------------------------------------------
# 4. Indefinite / Non-SPD Fallback
# ----------------------------------------------------------------------------

def test_pcg_fallback_on_indefinite():
    """Verify that an indefinite matrix triggers fallback to SuperLU without error."""
    # Indefinite matrix (one negative eigenvalue)
    K_indef = sp.csr_matrix([
        [-2.0, 1.0, 0.0],
        [1.0, 3.0, 0.5],
        [0.0, 0.5, 4.0],
    ])
    R = np.array([1.0, 2.0, 3.0])

    solver = LinearSolver("pcg", precond="jacobi")
    with pytest.warns(UserWarning, match="falling back to SuperLU"):
        du = solver.solve(K_indef, R)

    # Must match SuperLU exact solve
    du_ref = LinearSolver("superlu").solve(K_indef, R)
    np.testing.assert_allclose(du, du_ref, rtol=1e-8, atol=1e-8)


# ----------------------------------------------------------------------------
# 5. LinearSolver("pcg") in Statics Newton Iteration
# ----------------------------------------------------------------------------

def _single_hex_deck(L=10.0, force=210.0):
    """Single cube hexa8 in uniaxial tension."""
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

    mat = """\
/MAT/LAW1/1
steel elastic
   7.8e-6
     210.0       0.3
"""
    solidprop = """\
/PROP/SOLID/1
solid
       1.1      0.05       0.1
"""
    starter = f"""\
#RADIOSS STARTER
/BEGIN
HEX_PCG
/NODE
{chr(10).join(nodes)}
/BRICK/1
{brick}
/PART/1
cube
         1         1
{mat}{solidprop}/GRNOD/NODE/1
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


def test_pcg_in_statics_newton_iteration(make_deck, monkeypatch):
    """Verify that LinearSolver("pcg") solves the Newton increments in statics."""
    starter = _single_hex_deck(L=10.0, force=210.0)
    engine = """\
#RADIOSS ENGINE
/RUN/HEX_PCG/1
1.0
/IMPL
/IMPL/NEWTON
20  2  1.0e-5
/END
"""
    # 1. Run with SuperLU as baseline
    monkeypatch.setenv("PYRADIOSS_LINSOLVE", "superlu")
    s1, e1 = make_deck("HEX_SUPERLU", starter, engine)
    with contextlib.redirect_stdout(io.StringIO()):
        run_starter(s1)
        model_superlu = run_engine(e1)
    assert model_superlu.implicit_result.converged
    dx_superlu = model_superlu.x[1, 0] - model_superlu.x0[1, 0]

    # 2. Run with PCG iterative solver
    monkeypatch.setenv("PYRADIOSS_LINSOLVE", "pcg")
    monkeypatch.setenv("PYRADIOSS_PCG_PRECOND", "jacobi")
    s2, e2 = make_deck("HEX_PCG", starter, engine)
    with contextlib.redirect_stdout(io.StringIO()):
        run_starter(s2)
        model_pcg = run_engine(e2)
    assert model_pcg.implicit_result.converged
    dx_pcg = model_pcg.x[1, 0] - model_pcg.x0[1, 0]

    # Displacements must match closely
    np.testing.assert_allclose(dx_pcg, dx_superlu, rtol=1e-4, atol=1e-6)
    np.testing.assert_allclose(dx_pcg, 0.1, rtol=1e-3)

    # 3. Run with PCG iterative solver using ILU preconditioner
    monkeypatch.setenv("PYRADIOSS_LINSOLVE", "pcg")
    monkeypatch.setenv("PYRADIOSS_PCG_PRECOND", "ilu")
    s3, e3 = make_deck("HEX_PCG_ILU", starter, engine)
    with contextlib.redirect_stdout(io.StringIO()):
        run_starter(s3)
        model_ilu = run_engine(e3)
    assert model_ilu.implicit_result.converged
    dx_ilu = model_ilu.x[1, 0] - model_ilu.x0[1, 0]
    np.testing.assert_allclose(dx_ilu, dx_superlu, rtol=1e-4, atol=1e-6)

