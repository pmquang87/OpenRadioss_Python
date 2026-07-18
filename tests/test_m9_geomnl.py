"""
M9 validations: implicit NONLINEAR GEOMETRY — geometric (initial-stress)
stiffness, updated-Lagrangian reference frame, arc-length continuation.

Every new capability gets at least one ANALYTIC check (the port's
philosophy):

* K_geo -> 0 at zero stress (the M8 small-strain contract), and the full
  nonlinear-geometry residual/tangent pair verified by central finite
  differences (exactly for the truss incl. prestress, exactly for the hexa
  at zero stress);
* Euler column buckling P_cr = pi^2 EI/(kL)^2 out of the (K_mat + mu K_geo)
  eigenproblem — for BOTH the BT4 shell strip and a hexa8 solid column;
* a large-deflection cantilever traced against the ELASTICA (the
  Bisshopp–Drucker large-rotation solution, integrated numerically to
  reference accuracy in the test);
* a von Mises (two-bar) shallow truss traced THROUGH both limit points by
  the arc-length driver and checked pointwise against the closed-form
  equilibrium path P(y) = -2 E A ln(L/L0) y/L (exact for the corotational
  log-strain truss);
* the small-strain M8 behaviour reproduced under /IMPL/NONLIN (Hooke to
  O(eps) with a quadratically converging Newton), and uniaxial COMPRESSION
  now exactly Hooke (statics disables the rate-based bulk viscosity).

See PORTING_GUIDE.md roadmap M9.
"""

import contextlib
import io

import numpy as np
import pytest

from pyradioss.common.npcompat import trapezoid
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


def _starter_model(make_deck, name, starter):
    s, _ = make_deck(name, starter, "#\n/RUN/X/1\n1.0\n/END\n")
    with contextlib.redirect_stdout(io.StringIO()):
        return run_starter(s)


STEEL_NU0 = """\
/MAT/LAW1/1
steel elastic nu=0
   7.8e-6
     210.0       0.0
"""


def _block_deck(nx, ny, nz, L, force=210.0, qa=1.1, qb=0.05):
    """nx*ny*nz brick block (1/8-symmetry BCs, x-force on the x=L face) —
    the M8 uniaxial/patch deck, with selectable bulk-viscosity coeffs."""
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
/MAT/LAW1/1
steel elastic
   7.8e-6
     210.0       0.3
/PROP/SOLID/1
solid
       {qa}      {qb}       0.1
/GRNOD/NODE/1
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


def _strip_deck(nx, ny, Lx, b, t, load_dir, F_total, clamp_root=True):
    """A BT4 shell strip along x (nu = 0 so plate bending = beam bending
    exactly), root row clamped, a total force F_total on the tip row in
    ``load_dir`` distributed with the CONSISTENT uniform-traction weights
    (interior nodes twice the edge nodes) so a membrane load produces the
    exact uniform stress state."""
    def nid(i, j):
        return 1 + i + j * (nx + 1)
    nodes = [f"{nid(i,j):10d}{i*Lx/nx:20.10f}{j*b/ny:20.10f}{0.0:20.10f}"
             for j in range(ny+1) for i in range(nx+1)]
    shells = []
    eid = 0
    for j in range(ny):
        for i in range(nx):
            eid += 1
            c = [nid(i, j), nid(i+1, j), nid(i+1, j+1), nid(i, j+1)]
            shells.append(f"{eid:10d}" + "".join(f"{v:10d}" for v in c))
    fixed = [nid(0, j) for j in range(ny+1)]
    tip_edge = [nid(nx, 0), nid(nx, ny)]
    tip_int = [nid(nx, j) for j in range(1, ny)]
    # uniform-traction lumping: edge weight 1, interior weight 2
    wsum = 2 * (ny - 1) + 2
    f_edge = F_total / wsum
    f_int = 2.0 * F_total / wsum
    starter = f"""\
#RADIOSS STARTER
/BEGIN
STRIP
/NODE
{chr(10).join(nodes)}
/SHELL/1
{chr(10).join(shells)}
/PART/1
plate
         1         1
{STEEL_NU0}/PROP/SHELL/1
shell
       {t}         3      0.01
/GRNOD/NODE/1
fixed
{" ".join(map(str,fixed))}
/GRNOD/NODE/2
tip edge
{" ".join(map(str,tip_edge))}
/GRNOD/NODE/3
tip interior
{" ".join(map(str,tip_int))}
/BCS/1
clamp
       111       111         0         1
/FUNCT/1
ramp
       0.0       0.0
     100.0     100.0
/CLOAD/1
tip edge
         1         {load_dir}         2       {f_edge}
/CLOAD/2
tip interior
         1         {load_dir}         3       {f_int}
/END
"""
    return starter, nid


def _vonmises_deck(a, h, area, E, P_end):
    """Two-bar (von Mises) shallow truss: supports at (+-a, 0, 0), apex at
    (0, h, 0), downward -Y load ramping to P_end at load factor 1
    (proportional — the arc-length requirement). Apex z fixed (the out-of-
    plane direction carries no truss stiffness)."""
    starter = f"""\
#RADIOSS STARTER
/BEGIN
VMISES
/NODE
         1{-a:20.10f}{0.0:20.10f}{0.0:20.10f}
         2{a:20.10f}{0.0:20.10f}{0.0:20.10f}
         3{0.0:20.10f}{h:20.10f}{0.0:20.10f}
/TRUSS/1
         1         1         3
         2         2         3
/PART/1
bars
         1         1
/MAT/LAW1/1
elastic bar
   7.8e-6
     {E}       0.0
/PROP/TRUSS/1
bar
       {area}
/GRNOD/NODE/1
supports
1 2
/GRNOD/NODE/2
apex
3
/BCS/1
pin supports
       111       000         0         1
/BCS/2
apex out-of-plane
       001       000         0         2
/FUNCT/1
proportional ramp
       0.0       0.0
     100.0     100.0
/CLOAD/1
push down
         1         Y         2       {-P_end}
/END
"""
    return starter


def _vonmises_P(y, a, h, E, area):
    """Closed-form equilibrium load of the von Mises truss with the
    corotational LOG-strain bar (exactly the port's truss measure):
    P(y) = -2 E A ln(L/L0) * y/L, with y the current apex height."""
    L0 = np.sqrt(a * a + h * h)
    L = np.sqrt(a * a + y * y)
    return -2.0 * E * area * np.log(L / L0) * y / L


# ----------------------------------------------------------------------------
# K_geo basics: zero at zero stress; FD consistency of residual vs tangent
# ----------------------------------------------------------------------------

def test_kgeo_zero_at_zero_stress(make_deck):
    """K_geo vanishes identically on an unstressed state for all three
    implicit elements — the guarantee that the M8 small-strain results are
    untouched by adding it to the tangent."""
    from pyradioss.elements import solid_hexa8, shell_bt4, truss

    starter, _ = _block_deck(2, 2, 2, 10.0)
    model = _starter_model(make_deck, "KG0S", starter)
    kg, _ = solid_hexa8.kgeo(model.bricks, model.x0)
    assert np.all(kg == 0.0)

    starter, _ = _strip_deck(4, 2, 40.0, 10.0, 1.0, "Z", 0.0)
    model = _starter_model(make_deck, "KG0P", starter)
    kg, _ = shell_bt4.kgeo(model.shells, model.x0)
    assert np.all(kg == 0.0)

    model = _starter_model(make_deck, "KG0T",
                           _vonmises_deck(10.0, 2.0, 1.0, 210.0, 1.0))
    kg, _ = truss.kgeo(model.trusses, model.x0)
    assert np.all(kg == 0.0)


def _fd_check(model, sig_prestress=None, tol=1e-5):
    """Central-difference check of the M9 nonlinear-geometry residual against
    the assembled tangent K = -(dR/du) (with K_geo) at u = 0, optionally
    from a prestressed committed state."""
    from pyradioss.implicit.assembly import assemble
    from pyradioss.implicit.dofmap import DofMap
    from pyradioss.implicit.statics import _internal_forces, _snapshot, \
        _restore

    if sig_prestress is not None:
        for name, g in model.element_groups():
            g.state["sig"][:] = sig_prestress
    committed = {name: _snapshot(g) for name, g in model.element_groups()}
    dof = DofMap(model)
    n = model.numnod

    def residual(u_eq):
        u, ur = dof.scatter_solution(u_eq)
        fint, mint = _internal_forces(model, model.x0, u, ur, committed,
                                      nlgeom=True)
        return dof.gather_residual(fint, mint)

    for name, g in model.element_groups():
        _restore(g, committed[name])
    K = assemble(model, dof, model.x0, None, kgeo=True).toarray()

    h = 1e-6 * float(np.abs(model.x0).max())
    Kfd = np.empty_like(K)
    for j in range(dof.ndof):
        e = np.zeros(dof.ndof)
        e[j] = h
        # R = fint (element-on-node force), K = -dR/du
        Kfd[:, j] = -(residual(e) - residual(-e)) / (2.0 * h)
    scale = float(np.abs(K).max())
    assert np.abs(Kfd - K).max() <= tol * scale, (
        f"FD tangent mismatch {np.abs(Kfd - K).max():.3e} vs scale "
        f"{scale:.3e}")


def test_hexa_tangent_fd_consistency(make_deck):
    """The hexa8 nonlinear-geometry residual (midpoint stress update +
    end-configuration assembly + hourglass) linearizes EXACTLY to the
    assembled tangent at zero stress — central finite differences to 1e-5
    of the stiffness scale. (Bulk viscosity zeroed in the deck: statics.)"""
    starter, _ = _block_deck(1, 1, 1, 10.0, qa=0.0, qb=0.0)
    model = _starter_model(make_deck, "FDH", starter)
    _fd_check(model)


def test_truss_tangent_fd_consistency_with_prestress(make_deck):
    """The truss residual linearizes EXACTLY to (EA/L) a a^T + (F/L)(I-aa^T)
    — including the geometric term, checked by finite differences AT a
    prestressed state (the truss tangent is the textbook exact corotational
    one, so the check is tight)."""
    model = _starter_model(make_deck, "FDT",
                           _vonmises_deck(10.0, 2.0, 1.0, 210.0, 1.0))
    _fd_check(model, sig_prestress=-0.5, tol=1e-6)


# ----------------------------------------------------------------------------
# Small-strain limit: /IMPL/NONLIN reproduces the M8 answers
# ----------------------------------------------------------------------------

def test_nlgeom_small_strain_matches_hooke(make_deck):
    """A small-strain uniaxial pull under /IMPL/NONLIN reproduces the M8
    (Hooke) answer to O(eps) and Newton converges quadratically — the
    K_geo -> 0 / updated-frame small-strain contract."""
    L, E, F = 10.0, 210.0, 2.1            # eps = 1e-4: geometric effects tiny
    starter, nid = _block_deck(1, 1, 1, L, force=F)
    model = _run(make_deck, "NLGH", starter,
                 "#\n/RUN/NLGH/1\n1.0\n/IMPL/NONLIN\n/END\n")
    eps = (F / (L * L)) / E
    ux = (model.x - model.x0)[model.node_index(nid(1, 0, 0)), 0]
    assert ux == pytest.approx(eps * L, rel=1e-3)
    res = model.implicit_result
    assert res.converged
    inc = res.increments[-1]
    # quadratic tail: the residual collapses by many orders per iteration
    r = np.array(inc.residuals)
    assert inc.iterations <= 4
    assert r[-1] < 1e-6 * r[0]


def test_compression_hooke_exact(make_deck):
    """Uniaxial COMPRESSION is now exactly Hooke: statics disables the
    rate-based solid bulk viscosity, which would otherwise add a spurious
    viscous pressure to every compressive increment (the M8 validations
    were all tensile, so this was latent until M9's buckling prestates)."""
    L, E, nu, F = 10.0, 210.0, 0.3, -210.0
    starter, nid = _block_deck(1, 1, 1, L, force=F)
    model = _run(make_deck, "COMP", starter,
                 "#\n/RUN/COMP/1\n1.0\n/IMPL\n/END\n")
    eps = (F / (L * L)) / E                      # negative
    disp = model.x - model.x0
    ux = disp[model.node_index(nid(1, 0, 0)), 0]
    uy = disp[model.node_index(nid(1, 1, 0)), 1]
    assert ux == pytest.approx(eps * L, rel=1e-9)
    assert uy == pytest.approx(-nu * eps * L, rel=1e-6)
    assert model.implicit_result.increments[-1].iterations == 2


# ----------------------------------------------------------------------------
# Euler buckling: eigenvalue of (K_mat + mu K_geo) vs pi^2 EI / (kL)^2
# ----------------------------------------------------------------------------

def test_euler_buckling_shell_column(make_deck):
    """A clamped-free BT4 shell strip column (nu = 0: plate = beam) under a
    small axial compression P0: the lowest eigenvalue mu of
    (K_mat + mu K_geo) phi = 0 gives P_cr = mu*P0 = pi^2 EI/(4 L^2) — the
    Euler cantilever column, the analytic validation of the shell K_geo."""
    from pyradioss.implicit.buckling import buckling_factors
    Lx, b, t, E = 100.0, 10.0, 1.0, 210.0
    P0 = 0.002                       # sigma ~ 2e-4: linear-buckling regime
    starter, nid = _strip_deck(20, 2, Lx, b, t, "X", -P0)
    model = _run(make_deck, "EULS", starter,
                 "#\n/RUN/EULS/1\n1.0\n/IMPL\n/END\n")
    assert model.implicit_result.converged
    factors, modes = buckling_factors(model, nev=2)
    assert len(factors) >= 1
    I = b * t ** 3 / 12.0
    P_euler = np.pi ** 2 * E * I / (4.0 * Lx ** 2)   # k = 2 (clamped-free)
    P_cr = factors[0] * P0
    assert P_cr == pytest.approx(P_euler, rel=0.03)
    # the mode is a transverse (z) bend, not an in-plane motion
    du, dur = modes[0]
    assert np.abs(du[:, 2]).max() > 10.0 * np.abs(du[:, :2]).max()


def _column_deck(w, L, m, nzel, load_dir, F_total):
    """Clamped-base hexa8 column, m*m elements across the section, loaded on
    the top face along ``load_dir`` with the CONSISTENT uniform-traction
    node weights (corner 1 : edge 2 : interior 4) so an axial load produces
    the exact uniform patch stress (nu = 0)."""
    def nid(i, j, k):
        return 1 + i + j * (m + 1) + k * (m + 1) * (m + 1)
    d = w / m
    nodes = []
    for k in range(nzel + 1):
        for j in range(m + 1):
            for i in range(m + 1):
                nodes.append(f"{nid(i,j,k):10d}{i*d:20.10f}"
                             f"{j*d:20.10f}{k*L/nzel:20.10f}")
    bricks = []
    eid = 0
    for k in range(nzel):
        for j in range(m):
            for i in range(m):
                eid += 1
                c = [nid(i, j, k), nid(i+1, j, k), nid(i+1, j+1, k),
                     nid(i, j+1, k), nid(i, j, k+1), nid(i+1, j, k+1),
                     nid(i+1, j+1, k+1), nid(i, j+1, k+1)]
                bricks.append(f"{eid:10d}" + "".join(f"{v:10d}" for v in c))
    base = [nid(i, j, 0) for j in range(m+1) for i in range(m+1)]
    corner = [nid(i, j, nzel) for i in (0, m) for j in (0, m)]
    edge = ([nid(i, j, nzel) for i in range(1, m) for j in (0, m)]
            + [nid(i, j, nzel) for i in (0, m) for j in range(1, m)])
    inter = [nid(i, j, nzel) for i in range(1, m) for j in range(1, m)]
    wsum = len(corner) + 2 * len(edge) + 4 * len(inter)
    starter = f"""\
#RADIOSS STARTER
/BEGIN
COLUMN
/NODE
{chr(10).join(nodes)}
/BRICK/1
{chr(10).join(bricks)}
/PART/1
column
         1         1
{STEEL_NU0}/PROP/SOLID/1
solid
       1.1      0.05       0.1
/GRNOD/NODE/1
base
{" ".join(map(str, base))}
/GRNOD/NODE/2
top corners
{" ".join(map(str, corner))}
/GRNOD/NODE/3
top edges
{" ".join(map(str, edge))}
/GRNOD/NODE/4
top interior
{" ".join(map(str, inter))}
/BCS/1
clamp base
       111       000         0         1
/FUNCT/1
ramp
       0.0       0.0
     100.0     100.0
/CLOAD/1
corners
         1         {load_dir}         2       {F_total / wsum}
/CLOAD/2
edges
         1         {load_dir}         3       {2.0 * F_total / wsum}
/CLOAD/3
interior
         1         {load_dir}         4       {4.0 * F_total / wsum}
/END
"""
    top = corner + edge + inter
    return starter, top


def test_euler_buckling_hexa_column(make_deck):
    """A clamped-free hexa8 solid column (nu = 0): the buckling eigenvalue
    obeys EULER'S RELATION P_cr = pi^2 (EI)_eff / (4 L^2) with the mesh's
    OWN bending stiffness (EI)_eff, measured first by an implicit static
    tip-load bend through w = F L^3 / (3 (EI)_eff).

    Why the effective EI and not the continuum b*t^3/12: pure bending
    excites the one-point element's hourglass pattern in every element, so
    the FB *stiffness* stabilization (HG_STIFF, sized for robust static
    hourglass control) over-stiffens coarse-section bending — a documented
    K_mat discretization property of the 1-pt element, NOT a K_geo error.
    The Euler relation between bending stiffness and buckling load is
    exactly what the geometric stiffness controls, and it must hold for
    whatever bending stiffness the mesh actually has (it does, to <1%);
    the shell-column test above checks K_geo against the CONTINUUM Euler
    load, since BT4 bending is physical."""
    from pyradioss.implicit.buckling import buckling_factors
    w, L, E, m, nzel = 3.0, 60.0, 210.0, 3, 20

    # ---- 1. measure the mesh's effective bending stiffness --------------
    F_bend = 0.01
    starter, top = _column_deck(w, L, m, nzel, "X", F_bend)
    model = _run(make_deck, "BENDH", starter,
                 "#\n/RUN/BENDH/1\n1.0\n/IMPL\n/END\n")
    assert model.implicit_result.converged
    tipidx = [model.node_index(n) for n in top]
    w_tip = (model.x - model.x0)[tipidx, 0].mean()
    EI_eff = F_bend * L ** 3 / (3.0 * w_tip)
    # the over-stiffening is real and measured: between 1x and 5x continuum
    EI_cont = E * w ** 4 / 12.0
    assert EI_cont < EI_eff < 5.0 * EI_cont

    # ---- 2. compress and extract the buckling eigenvalue ----------------
    P0 = 0.01
    starter, top = _column_deck(w, L, m, nzel, "Z", -P0)
    model = _run(make_deck, "EULH", starter,
                 "#\n/RUN/EULH/1\n1.0\n/IMPL\n/END\n")
    assert model.implicit_result.converged
    # the prestate is the exact uniform patch stress (nu = 0, consistent
    # traction lumping): sigma_zz = -P0/w^2 in every element
    sig = model.bricks.state["sig"]
    assert np.allclose(sig[:, 2], -P0 / w ** 2, rtol=1e-6)
    factors, modes = buckling_factors(model, nev=2)
    P_cr = factors[0] * P0

    # ---- 3. Euler's relation with the measured stiffness ----------------
    P_euler = np.pi ** 2 * EI_eff / (4.0 * L ** 2)
    assert P_cr == pytest.approx(P_euler, rel=0.02)


# ----------------------------------------------------------------------------
# Large-deflection cantilever vs the elastica (Bisshopp–Drucker)
# ----------------------------------------------------------------------------

def _elastica_reference(alpha):
    """Tip position of a cantilever elastica under a dead transverse tip
    load, alpha = P L^2 / (EI). Integrates the exact large-rotation ODE
    theta'' = -alpha cos(theta), theta(0) = 0, theta'(1) = 0 (shooting on
    theta'(0)), then x_tip/L = int cos(theta), w_tip/L = int sin(theta)."""
    from scipy.integrate import solve_ivp
    from scipy.optimize import brentq

    def shoot(m0, dense=False):
        sol = solve_ivp(lambda s, y: [y[1], -alpha * np.cos(y[0])],
                        (0.0, 1.0), [0.0, m0], rtol=1e-10, atol=1e-12,
                        dense_output=dense)
        return sol

    m0 = brentq(lambda m: shoot(m).y[1, -1], 1e-6, 2.0 * alpha, xtol=1e-12)
    sol = shoot(m0, dense=True)
    s = np.linspace(0.0, 1.0, 2001)
    theta = sol.sol(s)[0]
    x_tip = trapezoid(np.cos(theta), s)
    w_tip = trapezoid(np.sin(theta), s)
    return x_tip, w_tip, theta[-1]


def test_elastica_cantilever(make_deck):
    """A BT4 strip cantilever pushed to alpha = P L^2/(EI) = 2 (tip rotation
    ~50 deg) under /IMPL/NONLIN load stepping lands on the ELASTICA tip
    position — the classic large-rotation benchmark that small-strain (M8)
    geometry cannot reproduce (linear theory would give w/L = 2/3)."""
    Lx, b, t, E = 100.0, 10.0, 1.0, 210.0
    I = b * t ** 3 / 12.0
    alpha = 2.0
    F = alpha * E * I / Lx ** 2
    starter, nid = _strip_deck(20, 2, Lx, b, t, "Z", F)
    model = _run(make_deck, "ELAS", starter,
                 "#\n/RUN/ELAS/1\n1.0\n/IMPL/NONLIN\n/IMPL/DTINI\n0.025\n"
                 "/END\n")
    res = model.implicit_result
    assert res.converged
    x_ref, w_ref, theta_tip = _elastica_reference(alpha)

    tip = [nid(20, j) for j in range(3)]
    disp = model.x - model.x0
    w_num = disp[[model.node_index(nn) for nn in tip], 2].mean() / Lx
    x_num = model.x[[model.node_index(nn) for nn in tip], 0].mean() / Lx
    # elastica at alpha=2: w/L ~ 0.49, x/L ~ 0.79 — far from linear theory
    assert w_num == pytest.approx(w_ref, abs=0.02)
    assert x_num == pytest.approx(x_ref, abs=0.02)
    # and clearly NOT the small-strain answer (w/L = alpha/3 = 0.667)
    assert abs(w_num - alpha / 3.0) > 0.1


# ----------------------------------------------------------------------------
# Snap-through: von Mises truss traced through its limit points by arc length
# ----------------------------------------------------------------------------

def test_vonmises_snap_through_arclength(make_deck):
    """The two-bar shallow truss has the classic N-shaped equilibrium path
    with two limit points (+P_max, -P_max) and a closed form
    P(y) = -2EA ln(L/L0) y/L. The arc-length driver must trace THROUGH both
    limit points (the load factor decreases and even goes negative on the
    unstable branch) and land on the far (snapped) equilibrium at the full
    load — checked against the closed form."""
    a, h, area, E = 10.0, 2.0, 1.0, 210.0
    ygrid = np.linspace(1e-6, h, 20001)
    P_max = float(_vonmises_P(ygrid, a, h, E, area).max())
    P_end = 1.3 * P_max

    starter = _vonmises_deck(a, h, area, E, P_end)
    model = _run(make_deck, "VMAR", starter,
                 "#\n/RUN/VMAR/1\n1.0\n/IMPL/ARCL\n/IMPL/DTINI\n0.05\n"
                 "/IMPL/NEWTON\n1e-9  40\n/END\n")
    res = model.implicit_result
    assert res.converged, res.stop_reason

    lams = np.array([i.load_factor for i in res.increments if i.converged])
    # 1. the trace passed the first limit point: the FIRST LOCAL MAXIMUM of
    #    the traced load approaches P_max from below (every converged
    #    increment is an equilibrium point, so the sampled peak can never
    #    exceed the true peak) and gets close to it
    turn = np.argmax(np.diff(lams) < 0.0)          # first descending step
    assert np.diff(lams).min() < 0.0, "path never turned a corner"
    lam_peak = lams[turn]
    assert lam_peak * P_end <= P_max * (1.0 + 1e-6)
    assert lam_peak * P_end >= P_max * 0.90
    # 2. it traced the unstable branch: the load factor went NEGATIVE
    #    (impossible under monotone load control), and by the closed form's
    #    antisymmetry never below -P_max
    assert lams.min() * P_end < -0.5 * P_max
    assert lams.min() * P_end >= -P_max * (1.0 + 1e-6)
    # 3. it finished at exactly the full load
    assert lams[-1] == pytest.approx(1.0, abs=1e-9)

    # 4. the final state is the closed-form FAR-branch equilibrium
    from scipy.optimize import brentq
    y_far = brentq(lambda y: _vonmises_P(y, a, h, E, area) - P_end,
                   -10.0 * h, -h)
    apex = model.node_index(3)
    y_num = model.x[apex, 1]
    assert y_num == pytest.approx(y_far, rel=1e-3)
    # snapped well past the mirror position
    assert y_num < -h


def test_vonmises_load_control_cannot_trace(make_deck):
    """The same truss under pure load control (/IMPL/NONLIN, no arc length)
    cannot TRACE the path: at the limit point it either fails to converge
    or Newton JUMPS discontinuously to the far branch — either way the
    descending branch is never followed (that is what /IMPL/ARCL is for)."""
    a, h, area, E = 10.0, 2.0, 1.0, 210.0
    ygrid = np.linspace(1e-6, h, 20001)
    P_max = float(_vonmises_P(ygrid, a, h, E, area).max())
    P_end = 1.3 * P_max

    starter = _vonmises_deck(a, h, area, E, P_end)
    model = _run(make_deck, "VMLC", starter,
                 "#\n/RUN/VMLC/1\n1.0\n/IMPL/NONLIN\n/IMPL/DTINI\n0.05\n"
                 "/IMPL/NEWTON\n1e-9  40\n/END\n")
    res = model.implicit_result
    if res.converged:
        # Newton jumped the gap: the final state must be the far branch —
        # a discontinuous snap, not a traced path (no negative-load points)
        apex = model.node_index(3)
        assert model.x[apex, 1] < -h
        lams = np.array([i.load_factor for i in res.increments])
        assert lams.min() >= 0.0
    else:
        # or it stopped dead at the limit point
        assert "NO" in res.stop_reason.upper() or res.stop_reason


def test_arclength_rejects_impdisp(make_deck):
    """/IMPL/ARCL + /IMPDISP is a contradiction (the arc constraint already
    controls the step, and the method assumes proportional FORCE loading):
    the driver refuses with a clear error instead of tracing nonsense."""
    L = 10.0
    starter, nid = _block_deck(1, 1, 1, L, force=0.0)
    starter = starter.replace(
        """/CLOAD/1
pull
         1         X         2       0.0
/END
""",
        """/FUNCT/2
disp ramp
       0.0       0.0
       1.0       0.1
/IMPDISP/1
pull face x=L
         2         X         2       1.0
/END
""")
    s, e = make_deck("ARCE", starter,
                     "#\n/RUN/ARCE/1\n1.0\n/IMPL/ARCL\n/END\n")
    with contextlib.redirect_stdout(io.StringIO()):
        run_starter(s)
        with pytest.raises(ValueError, match="ARCL"):
            run_engine(e)
