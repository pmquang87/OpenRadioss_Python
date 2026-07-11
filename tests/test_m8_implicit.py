"""
M8 validations: implicit (Newton–Raphson) statics.

Every new capability gets at least one ANALYTIC check (the port's philosophy):

* a single-element uniaxial pull reproducing Hooke's law EXACTLY, and a
  multi-element patch test (uniform stress everywhere);
* a shell cantilever tip deflection against the closed-form beam result;
* the LAW2 consistent (algorithmic) tangent — asserted through (a) the exact
  Johnson–Cook uniaxial curve, (b) QUADRATIC Newton convergence, and (c) a
  match against the explicit solver driven quasi-statically;
* DOF condensation, the equilibrium (reaction) balance and the strain-energy
  balance, plus the linear-solver backend fallback and the scipy guard.

See PORTING_GUIDE.md roadmap M8.
"""

import contextlib
import io
import os
import textwrap
import warnings

import numpy as np
import pytest

from pyradioss.engine.engine import run_engine
from pyradioss.starter.starter import run_starter

pytest.importorskip("scipy")   # implicit requires scipy (optional otherwise)


# ----------------------------------------------------------------------------
# helpers
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


def _block_deck(nx, ny, nz, L, mat=STEEL1, force=210.0):
    """A nx*ny*nz brick block, x=0 face ux-fixed, y=0 uy-fixed, z=0 uz-fixed
    (1/8 symmetry), a total x-force spread over the x=L face — a uniform
    uniaxial-stress state (patch test)."""
    dx, dy, dz = L / nx, L / ny, L / nz

    def nid(i, j, k):
        return 1 + i + j * (nx + 1) + k * (nx + 1) * (ny + 1)

    nodes = []
    for k in range(nz + 1):
        for j in range(ny + 1):
            for i in range(nx + 1):
                nodes.append(f"{nid(i,j,k):10d}{i*dx:20.10f}"
                             f"{j*dy:20.10f}{k*dz:20.10f}")
    bricks = []
    eid = 0
    for k in range(nz):
        for j in range(ny):
            for i in range(nx):
                eid += 1
                c = [nid(i, j, k), nid(i+1, j, k), nid(i+1, j+1, k),
                     nid(i, j+1, k), nid(i, j, k+1), nid(i+1, j, k+1),
                     nid(i+1, j+1, k+1), nid(i, j+1, k+1)]
                bricks.append(f"{eid:10d}" + "".join(f"{v:10d}" for v in c))
    x0 = [nid(0, j, k) for k in range(nz+1) for j in range(ny+1)]
    x1 = [nid(nx, j, k) for k in range(nz+1) for j in range(ny+1)]
    y0 = [nid(i, 0, k) for k in range(nz+1) for i in range(nx+1)]
    z0 = [nid(i, j, 0) for j in range(ny+1) for i in range(nx+1)]
    fpn = force / len(x1)
    starter = f"""\
#RADIOSS STARTER
/BEGIN
BLOCK
/NODE
{chr(10).join(nodes)}
/BRICK/1
{chr(10).join(bricks)}
/PART/1
cube
         1         1
{mat}{SOLIDPROP}/GRNOD/NODE/1
x0
{" ".join(map(str,x0))}
/GRNOD/NODE/2
x1
{" ".join(map(str,x1))}
/GRNOD/NODE/3
y0
{" ".join(map(str,y0))}
/GRNOD/NODE/4
z0
{" ".join(map(str,z0))}
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
    return starter, nid


# ----------------------------------------------------------------------------
# DOF map / equation numbering
# ----------------------------------------------------------------------------

def test_dofmap_condensation(make_deck):
    """/BCS-fixed DOFs are condensed out (removed, not penalized); a
    solid-only model gets no rotational equations."""
    from pyradioss.implicit.dofmap import DofMap
    starter, nid = _block_deck(1, 1, 1, 10.0)
    s, _ = make_deck("DOFB", starter, "#\n/RUN/DOFB/1\n1.0\n/IMPL\n/END\n")
    with contextlib.redirect_stdout(io.StringIO()):
        model = run_starter(s)
    dof = DofMap(model)
    # 8 nodes * 3 translations = 24 slots; fixed: x0 face ux (4), y0 uy (4),
    # z0 uz (4) => 12 fixed, 12 free. No rotational equations (solids).
    assert dof.ndof == 12
    assert not dof.has_rot.any()
    # fixed slots map to -1 (condensed)
    i_x0 = model.node_index(nid(0, 0, 0))
    assert dof.eq[i_x0 * 6 + 0] == -1     # ux fixed
    # a free translational slot has a real equation index
    i_x1 = model.node_index(nid(1, 0, 0))
    assert dof.eq[i_x1 * 6 + 0] >= 0      # ux free (loaded face)


# ----------------------------------------------------------------------------
# Linear elastic solid: Hooke exactly + one-step (quadratic) convergence
# ----------------------------------------------------------------------------

def test_solid_uniaxial_hooke(make_deck):
    """Single hexa, uniaxial pull: stress = F/A and strain = stress/E EXACTLY,
    Poisson contraction = -nu*eps, and Newton converges in ONE step (the
    sharpest form of quadratic convergence for a linear problem)."""
    L, E, nu, F = 10.0, 210.0, 0.3, 210.0
    starter, nid = _block_deck(1, 1, 1, L, force=F)
    model = _run(make_deck, "UNI", starter,
                 "#\n/RUN/UNI/1\n1.0\n/IMPL\n/END\n")
    sigma = F / (L * L)                       # F / A0
    eps = sigma / E
    disp = model.x - model.x0
    ux = disp[model.node_index(nid(1, 0, 0)), 0]
    uy = disp[model.node_index(nid(1, 1, 0)), 1]
    assert ux == pytest.approx(eps * L, rel=1e-9)
    assert uy == pytest.approx(-nu * eps * L, rel=1e-6)
    assert model.bricks.state["sig"][0, 0] == pytest.approx(sigma, rel=1e-9)
    # one-step convergence: 1 residual + 1 solve + 1 residual = 2 evaluations
    res = model.implicit_result
    assert res.converged
    inc = res.increments[-1]
    assert inc.iterations == 2
    assert inc.residuals[-1] < 1e-8 * inc.residuals[0]


def test_solid_patch_test(make_deck):
    """A 3x2x2 brick block, x=L face displaced by a prescribed uniform ux
    (/IMPDISP, displacement control): EVERY element sees the SAME exact
    uniaxial stress and the displacement field is exactly linear — the
    classic finite-element constant-stress patch test (no hourglass, no
    load-discretization error)."""
    L, E, nu = 12.0, 210.0, 0.3
    eps = 0.01
    delta = eps * L                            # prescribed tip displacement
    starter, nid = _block_deck(3, 2, 2, L, force=0.0)
    # replace the /CLOAD with an /IMPDISP that ramps ux of the x=L face to
    # ``delta`` at load factor 1
    starter = starter.replace(
        f"""/CLOAD/1
pull
         1         X         2       0.0
/END
""",
        f"""/FUNCT/2
disp ramp
       0.0       0.0
       1.0       {delta}
/IMPDISP/1
pull face x=L
         2         X         2       1.0
/END
""")
    model = _run(make_deck, "PATCH", starter,
                 "#\n/RUN/PATCH/1\n1.0\n/IMPL\n/END\n")
    sig = model.bricks.state["sig"]
    sigma = E * eps                            # uniaxial Hooke (sig_zz=... )
    # sigma_xx uniform and exact everywhere; transverse/shear ~ 0
    assert np.allclose(sig[:, 0], sigma, rtol=1e-8)
    assert np.allclose(sig[:, 1:], 0.0, atol=1e-7 * sigma)
    # displacement field is exactly linear: ux = eps*x on every node
    ux = (model.x - model.x0)[:, 0]
    assert np.allclose(ux, eps * model.x0[:, 0], atol=1e-9 * L)


def test_energy_and_reaction_balance(make_deck):
    """Displacement-controlled uniaxial pull of a 2x2x2 block: the reaction
    at the fixed face balances the reaction at the driven face (equilibrium),
    the reaction magnitude matches sigma*A, and the stored strain energy
    equals the external work 0.5 R.delta (linear elastic)."""
    from pyradioss.implicit.statics import _internal_forces, _snapshot
    L, E = 10.0, 210.0
    eps = 0.01
    delta = eps * L
    starter, nid = _block_deck(2, 2, 2, L, force=0.0)
    starter = starter.replace(
        f"""/CLOAD/1
pull
         1         X         2       0.0
/END
""",
        f"""/FUNCT/2
disp ramp
       0.0       0.0
       1.0       {delta}
/IMPDISP/1
pull face x=L
         2         X         2       1.0
/END
""")
    model = _run(make_deck, "BAL", starter,
                 "#\n/RUN/BAL/1\n1.0\n/IMPL\n/END\n")
    # internal force at the converged displacement, from a ZERO-stress
    # reference (grad(u) = total strain). Internal forces are globally
    # self-equilibrated; the reactions live on the constrained faces.
    u = model.x - model.x0
    committed = {name: _snapshot(g) for name, g in model.element_groups()}
    for snap in committed.values():
        for key in ("sig", "epsp", "eint", "ehour", "qvw_pend"):
            if key in snap:
                snap[key][...] = 0.0
    fint, _ = _internal_forces(model, model.x0, u, np.zeros_like(u), committed)
    x0nodes = [model.node_index(nid(0, j, k))
               for k in range(3) for j in range(3)]
    x1nodes = [model.node_index(nid(2, j, k))
               for k in range(3) for j in range(3)]
    react0 = fint[x0nodes, 0].sum()
    react1 = fint[x1nodes, 0].sum()
    R = E * eps * (L * L)                        # sigma * A
    assert react1 == pytest.approx(-R, rel=1e-6)   # driven-face reaction
    assert react0 == pytest.approx(R, rel=1e-6)    # fixed-face reaction
    assert abs(fint[:, 0].sum()) < 1e-6 * R        # globally self-equilibrated
    # strain energy = external work 0.5 * R * delta
    ie = sum(float(g.state["eint"].sum()) for _, g in model.element_groups())
    assert ie == pytest.approx(0.5 * R * delta, rel=1e-6)


# ----------------------------------------------------------------------------
# Shell cantilever: tip deflection vs closed form
# ----------------------------------------------------------------------------

def test_shell_cantilever(make_deck):
    """A slender BT4 shell cantilever under a transverse tip load matches the
    Euler–Bernoulli tip deflection w = F L^3 / (3 E I) to within a few %, and
    converges in one Newton step (linear elastic)."""
    Lx, b, t = 100.0, 10.0, 1.0
    nx, ny = 20, 2
    E, nu, F = 210.0, 0.3, 0.001

    def nidf(i, j):
        return 1 + i + j * (nx + 1)
    nodes = [f"{nidf(i,j):10d}{i*Lx/nx:20.10f}{j*b/ny:20.10f}{0.0:20.10f}"
             for j in range(ny+1) for i in range(nx+1)]
    shells = []
    eid = 0
    for j in range(ny):
        for i in range(nx):
            eid += 1
            c = [nidf(i, j), nidf(i+1, j), nidf(i+1, j+1), nidf(i, j+1)]
            shells.append(f"{eid:10d}" + "".join(f"{v:10d}" for v in c))
    fixed = [nidf(0, j) for j in range(ny+1)]
    tip = [nidf(nx, j) for j in range(ny+1)]
    starter = f"""\
#RADIOSS STARTER
/BEGIN
CANT
/NODE
{chr(10).join(nodes)}
/SHELL/1
{chr(10).join(shells)}
/PART/1
plate
         1         1
{STEEL1}/PROP/SHELL/1
shell
       1.0         5      0.01
/GRNOD/NODE/1
fixed
{" ".join(map(str,fixed))}
/GRNOD/NODE/2
tip
{" ".join(map(str,tip))}
/BCS/1
clamp
       111       111         0         1
/FUNCT/1
ramp
       0.0       0.0
     100.0     100.0
/CLOAD/1
tip
         1         Z         2       {F/len(tip)}
/END
"""
    model = _run(make_deck, "CANT", starter,
                 "#\n/RUN/CANT/1\n1.0\n/IMPL\n/END\n")
    disp = model.x - model.x0
    wz = disp[[model.node_index(nn) for nn in tip], 2].mean()
    I = b * t ** 3 / 12.0
    w_eb = F * Lx ** 3 / (3 * E * I)
    assert wz == pytest.approx(w_eb, rel=0.02)      # < 2 % vs beam theory
    assert model.implicit_result.increments[-1].iterations == 2


# ----------------------------------------------------------------------------
# LAW2 consistent tangent: analytic curve + quadratic convergence
# ----------------------------------------------------------------------------

def test_law2_uniaxial_and_quadratic_convergence(make_deck):
    """Single hexa pulled past yield with LAW2: the plastic strain and the
    displacement match the closed-form Johnson–Cook uniaxial solution, AND the
    plastic increments show QUADRATIC Newton convergence — the signature of a
    CORRECT consistent (algorithmic) tangent (a continuum tangent would only
    converge linearly)."""
    L, E, A, B, nexp = 10.0, 210.0, 0.4, 0.5, 0.5
    sigma = 0.5                                # target axial stress (> A)
    F = sigma * L * L
    starter, nid = _block_deck(1, 1, 1, L, mat=STEEL2, force=F)
    model = _run(make_deck, "PL", starter,
                 "#\n/RUN/PL/1\n1.0\n/IMPL/DTINI\n0.1\n"
                 "/IMPL/NEWTON\n1e-9  40\n/END\n")
    # analytic: sigma = A + B eps_p^n  ->  eps_p ; total strain = sigma/E+eps_p
    epsp_a = ((sigma - A) / B) ** (1.0 / nexp)
    eps_a = sigma / E + epsp_a
    disp = model.x - model.x0
    ux = disp[model.node_index(nid(1, 0, 0)), 0]
    assert model.bricks.state["epsp"][0] == pytest.approx(epsp_a, rel=1e-4)
    assert ux == pytest.approx(eps_a * L, rel=1e-4)
    assert model.bricks.state["sig"][0, 0] == pytest.approx(sigma, rel=1e-4)

    # quadratic convergence: find a plastic increment (> 2 iterations) and
    # verify the residual drops faster than linearly near the solution
    res = model.implicit_result
    assert res.converged
    plastic_incs = [i for i in res.increments if i.iterations > 2]
    assert plastic_incs, "expected at least one plastic (multi-iter) increment"
    inc = plastic_incs[0]
    r = np.array(inc.residuals)
    # last three residuals: r[k+1] <= C r[k]^2 with a small C => quadratic.
    # A robust proxy: the convergence rate accelerates (ratio r[k+1]/r[k]
    # shrinks by orders of magnitude across the tail).
    tail = r[r > 0][-3:]
    ratio1 = tail[1] / tail[0]
    ratio2 = tail[2] / tail[1]
    assert ratio2 < ratio1 ** 1.5           # super-linear (quadratic) tail


def test_law2_matches_explicit(make_deck):
    """The implicit LAW2 result matches the EXPLICIT solver driven
    quasi-statically to the same uniaxial stress state (rate term off), within
    a small tolerance — the cross-solver consistency check the M8 task asks
    for."""
    L, F = 10.0, 0.5 * 100.0

    # ---- implicit: load-step to the target force ------------------------
    starter, nid = _block_deck(1, 1, 1, L, mat=STEEL2, force=F)
    mi = _run(make_deck, "PLIMP", starter,
              "#\n/RUN/PLIMP/1\n1.0\n/IMPL/DTINI\n0.05\n"
              "/IMPL/NEWTON\n1e-9  40\n/END\n")
    epsp_imp = float(mi.bricks.state["epsp"][0])
    ux_imp = (mi.x - mi.x0)[mi.node_index(nid(1, 0, 0)), 0]

    # ---- explicit: pull the SAME cube quasi-statically to the same stress
    # via a slow imposed displacement, then read the plastic strain ------
    ux_target = ux_imp
    starter_e, nid2 = _block_deck(1, 1, 1, L, mat=STEEL2, force=F)
    # replace the /CLOAD with a slow imposed displacement to ux_target
    starter_e = starter_e.replace(
        f"""/CLOAD/1
pull
         1         X         2       {F/4.0}
/END
""",
        f"""/FUNCT/2
disp ramp to target then hold
       0.0       0.0
       5.0       {ux_target}
     100.0       {ux_target}
/IMPDISP/1
pull x=L to target
         2         X         2       1.0
/END
""")
    with contextlib.redirect_stdout(io.StringIO()):
        s, e = make_deck("PLEXP", starter_e,
                         "#\n/RUN/PLEXP/1\n8.0\n/DT\n0.5  0.0\n/END\n")
        run_starter(s)
        me = run_engine(e)
    epsp_exp = float(me.bricks.state["epsp"][0])
    # both solvers reach essentially the same plastic strain at the same
    # applied deformation (quasi-static, rate-independent LAW2)
    assert epsp_imp == pytest.approx(epsp_exp, rel=0.05)


# ----------------------------------------------------------------------------
# Linear-solver backend selection (mirrors the M7 accel pattern)
# ----------------------------------------------------------------------------

def test_linsolve_default_is_superlu():
    from pyradioss.implicit.linsolve import select_linsolve
    assert select_linsolve("superlu") == "superlu"
    assert select_linsolve(None) == "superlu"    # default


def test_linsolve_optional_fallback():
    """Requesting an optional solver whose library is absent falls back to
    SuperLU with a warning — never breaking the run (the M7 contract)."""
    from pyradioss.implicit.linsolve import select_linsolve
    for name in ("cholmod", "mumps"):
        try:
            __import__({"cholmod": "sksparse", "mumps": "mumps"}[name])
            continue          # library present: no fallback to test
        except ImportError:
            pass
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            assert select_linsolve(name) == "superlu"
            assert any("falling back to SuperLU" in str(x.message) for x in w)


def test_linsolve_unknown_fallback():
    from pyradioss.implicit.linsolve import select_linsolve
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        assert select_linsolve("nonsense") == "superlu"
        assert any("unknown" in str(x.message).lower() for x in w)


def test_scipy_guard_present():
    """require_scipy returns the sparse modules when scipy is installed."""
    from pyradioss.implicit import require_scipy
    sp, spla = require_scipy()
    assert hasattr(sp, "coo_matrix")
    assert hasattr(spla, "splu")


# ----------------------------------------------------------------------------
# The tangent must not perturb the force path (M7 parity spirit)
# ----------------------------------------------------------------------------

def test_tangent_does_not_touch_force_state(make_deck):
    """Calling tangent() must leave the element buffer byte-for-byte
    unchanged — it is a read-only linearization alongside forces()."""
    from pyradioss.elements import solid_hexa8, shell_bt4
    starter, nid = _block_deck(2, 2, 2, 10.0)
    s, _ = make_deck("TAN", starter, "#\n/RUN/TAN/1\n1.0\n/IMPL\n/END\n")
    with contextlib.redirect_stdout(io.StringIO()):
        model = run_starter(s)
    g = model.bricks
    before = {k: v.copy() for k, v in g.state.items()
              if isinstance(v, np.ndarray)}
    solid_hexa8.tangent(g, model.x0, None)
    for k, v in before.items():
        assert np.array_equal(g.state[k], v), f"tangent mutated state[{k}]"
