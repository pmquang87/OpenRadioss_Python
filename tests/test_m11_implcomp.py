"""
M11 validations: implicit COMPLETENESS — element tangents for every family
(tetra4, sh3n, beam, spring), the LAW2 shell/truss consistent tangents,
/IMPL/DYNA/DAMP Rayleigh damping, automatic implicit step control
(imp_dt.F) and the /IMPL/BUCKL engine card.

Every new capability gets at least one ANALYTIC check (the port's
philosophy):

* finite-difference residual/tangent consistency for each new element
  (EXACT at zero stress for all four; exact including prestress for the
  beam's axial K_geo and the spring; at the documented sigma/E scale of
  the omitted Jaumann-spin terms for the stressed tetra/sh3n — the same
  class the M9 hexa carries);
* closed-form statics: a tetra bar under displacement control reproducing
  the uniform uniaxial-stress patch state EXACTLY in one Newton step; the
  sh3n membrane pull exact (CST) and a THICK sh3n cantilever within 2% of
  the Timoshenko closed form (thin C0 triangles shear-lock — the module
  docstring's honest note — so the shear-dominated thick strip is the
  tight anchor); the beam cantilever within 0.5% of FL^3/3EI + FL/GA at
  10 elements (O(h^2) convergence measured); the spring reproducing
  u = F/k exactly in one step;
* LAW2 consistent tangents: the plane-stress (Iplas=2) shell pull past
  yield matching the algorithm's OWN closed form — eps_p exact on the JC
  curve and u/L = sigma/E + (3G/E) eps_p, the radial PROJECTION's
  documented axial-flow factor (3G/E = 3/(2(1+nu)); the exact plane-stress
  return would give eps_p itself) — with a QUADRATIC Newton tail; the
  elastoplastic truss against the same JC relations; the tetra sharing the
  M8 solid consistent tangent past yield;
* /IMPL/DYNA/DAMP: the damped SDOF ring-down against the closed form
  u = (v0/om_d) e^(-zeta om t) sin(om_d t) for a mass-only, stiffness-only
  and mixed Rayleigh case, the damped period from zero crossings, and the
  energy balance closing to round-off WITH the dissipation ledger;
* automatic step control: a deep-elastica strip whose single-increment
  /IMPL/NONLIN run FAILS its Newton budget, now completing through
  automatic cuts and matching the fine fixed-increment answer; a smooth
  run taking zero cuts (the fixed-dt behaviour reproduced);
* /IMPL/BUCKL: the M9 shell Euler column THROUGH THE CARD PATH —
  P_cr = pi^2 EI/(4L^2) within 3%, factors/modes on the result object and
  the imp_buck.F-flavoured listing block.

See PORTING_GUIDE.md roadmap M11.
"""

import contextlib
import io

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


def _run_capture(make_deck, name, starter, engine):
    s, e = make_deck(name, starter, engine)
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        run_starter(s)
        model = run_engine(e)
    return model, out.getvalue()


def _starter_model(make_deck, name, starter):
    s, _ = make_deck(name, starter, "#\n/RUN/X/1\n1.0\n/END\n")
    with contextlib.redirect_stdout(io.StringIO()):
        return run_starter(s)


STEEL1 = "/MAT/LAW1/1\nsteel\n7.8e-6\n210. 0.3\n"
STEEL2 = "/MAT/LAW2/1\nsteel jc\n7.8e-6\n210. 0.3\n0.4 0.5 0.5\n"

# JC constants of STEEL2 for the closed forms
E_JC, NU_JC, A_JC, B_JC, N_JC = 210.0, 0.3, 0.4, 0.5, 0.5
G_JC = E_JC / (2.0 * (1.0 + NU_JC))


# ---- tetra bar: Kuhn (6-tet) split of an nx*ny*nz grid ----------------------

def _tet_bar_deck(nx, ny, nz, L, W, mat=STEEL1, ux_end=0.0):
    """Bar [0,L]x[0,W]x[0,W] split into 6 tets per cell (Kuhn triangulation,
    orientation fixed to positive volume), 1/8-symmetry BCs and an /IMPDISP
    uniform end displacement on x = L — displacement control makes the
    exact solution the UNIFORM uniaxial-stress state regardless of load
    lumping (the patch test)."""
    import itertools
    dx, dy, dz = L / nx, W / ny, W / nz

    def nid(i, j, k):
        return 1 + i + j * (nx + 1) + k * (nx + 1) * (ny + 1)

    nodes = []
    for k in range(nz + 1):
        for j in range(ny + 1):
            for i in range(nx + 1):
                nodes.append(f"{nid(i,j,k):10d}{i*dx:20.10f}"
                             f"{j*dy:20.10f}{k*dz:20.10f}")
    tets = []
    eid = 0
    perms = list(itertools.permutations(range(3)))
    for k in range(nz):
        for j in range(ny):
            for i in range(nx):
                base = np.array([i, j, k])
                for p in perms:
                    # path 000 -> 111 adding axis p[0], p[1], p[2]
                    vs = [base.copy()]
                    for ax in p:
                        v = vs[-1].copy()
                        v[ax] += 1
                        vs.append(v)
                    # positive volume: swap the last two for odd parity
                    par = (np.array(p) != np.array([0, 1, 2])).sum()
                    if par == 2:            # one transposition = odd
                        vs[2], vs[3] = vs[3], vs[2]
                    e1 = vs[1] - vs[0]
                    e2 = vs[2] - vs[0]
                    e3 = vs[3] - vs[0]
                    assert np.linalg.det(np.array(
                        [e1, e2, e3], dtype=float)) > 0
                    eid += 1
                    c = [nid(*v) for v in vs]
                    tets.append(f"{eid:10d}" + "".join(f"{v:10d}" for v in c))
    x0 = [nid(0, j, k) for k in range(nz+1) for j in range(ny+1)]
    x1 = [nid(nx, j, k) for k in range(nz+1) for j in range(ny+1)]
    y0 = [nid(i, 0, k) for k in range(nz+1) for i in range(nx+1)]
    z0 = [nid(i, j, 0) for j in range(ny+1) for i in range(nx+1)]
    starter = f"""\
#RADIOSS STARTER
/BEGIN
TETBAR
/NODE
{chr(10).join(nodes)}
/TETRA4/1
{chr(10).join(tets)}
/PART/1
bar
         1         1
{mat}/PROP/SOLID/1
solid
       0.0       0.0       0.1
/GRNOD/NODE/1
x0
{" ".join(map(str, x0))}
/GRNOD/NODE/2
x1
{" ".join(map(str, x1))}
/GRNOD/NODE/3
y0
{" ".join(map(str, y0))}
/GRNOD/NODE/4
z0
{" ".join(map(str, z0))}
/BCS/1
sym x
       100       000         0         1
/BCS/2
sym y
       010       000         0         3
/BCS/3
sym z
       001       000         0         4
/FUNCT/1
ramp
       0.0       0.0
       1.0       {ux_end}
/IMPDISP/1
pull face x=L
         1         X         2       1.0
/END
"""
    return starter, nid


# ---- sh3n strip (each quad split into two triangles) ------------------------

def _sh3n_strip_deck(nx, ny, Lx, b, t, load_dir, F_total, nu="0.0",
                     mat=None, membrane_bcs=False):
    def nid(i, j):
        return 1 + i + j * (nx + 1)
    nodes = [f"{nid(i,j):10d}{i*Lx/nx:20.10f}{j*b/ny:20.10f}{0.0:20.10f}"
             for j in range(ny+1) for i in range(nx+1)]
    tris = []
    eid = 0
    for j in range(ny):
        for i in range(nx):
            eid += 1
            tris.append(f"{eid:10d}{nid(i,j):10d}{nid(i+1,j):10d}"
                        f"{nid(i+1,j+1):10d}")
            eid += 1
            tris.append(f"{eid:10d}{nid(i,j):10d}{nid(i+1,j+1):10d}"
                        f"{nid(i,j+1):10d}")
    fixed = [nid(0, j) for j in range(ny+1)]
    tip_edge = [nid(nx, 0), nid(nx, ny)]
    tip_int = [nid(nx, j) for j in range(1, ny)]
    wsum = 2 * (ny - 1) + 2
    f_edge = F_total / wsum
    f_int = 2.0 * F_total / wsum
    matcard = mat if mat is not None else \
        f"/MAT/LAW1/1\nsteel\n7.8e-6\n210. {nu}\n"
    if membrane_bcs:
        # pure-membrane pull: root slides laterally (only ux fixed), one
        # corner pins uy, everyone's uz + rotations are fixed (no bending)
        allnodes = [nid(i, j) for j in range(ny+1) for i in range(nx+1)]
        bcs = f"""/GRNOD/NODE/4
all
{" ".join(map(str, allnodes))}
/BCS/1
root ux
       100       000         0         1
/BCS/2
pin uy
       010       000         0         5
/BCS/3
membrane only
       001       111         0         4
/GRNOD/NODE/5
corner
{nid(0, 0)}
"""
    else:
        bcs = """/BCS/1
clamp
       111       111         0         1
"""
    starter = f"""\
#RADIOSS STARTER
/BEGIN
TRISTRIP
/NODE
{chr(10).join(nodes)}
/SH3N/1
{chr(10).join(tris)}
/PART/1
plate
         1         1
{matcard}/PROP/SHELL/1
shell
         1         0         0         0
      0.01      0.01      0.01         0         0
         3         0       {t}
/GRNOD/NODE/1
fixed
{" ".join(map(str, fixed))}
/GRNOD/NODE/2
tip edge
{" ".join(map(str, tip_edge))}
/GRNOD/NODE/3
tip interior
{" ".join(map(str, tip_int))}
{bcs}/FUNCT/1
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


# ---- beam cantilever chain --------------------------------------------------

def _beam_deck(nel, L, A, Iy, Ix, F, load_dir="Z"):
    nodes = [f"{i+1:10d}{i*L/nel:20.10f}{0.0:20.10f}{0.0:20.10f}"
             for i in range(nel + 1)]
    nodes.append(f"{900:10d}{0.0:20.10f}{50.0:20.10f}{0.0:20.10f}")
    beams = [f"{i+1:10d}{i+1:10d}{i+2:10d}{900:10d}" for i in range(nel)]
    return f"""\
#RADIOSS STARTER
/BEGIN
BEAMC
/NODE
{chr(10).join(nodes)}
/BEAM/1
{chr(10).join(beams)}
/PART/1
beam
         1         1
{STEEL1}/PROP/BEAM/1
square
      {A}     {Iy}     {Iy}     {Ix}
/GRNOD/NODE/1
root
1
/GRNOD/NODE/2
tip
{nel+1}
/BCS/1
clamp
       111       111         0         1
/FUNCT/1
ramp
       0.0       0.0
     100.0     100.0
/CLOAD/1
tip
         1         {load_dir}         2       {F}
/END
"""


# ---- spring + truss singles --------------------------------------------------

def _spring_deck(k, F):
    return f"""\
#RADIOSS STARTER
/BEGIN
SPR
/NODE
         1       0.0       0.0       0.0
         2       2.0       0.0       0.0
/SPRING/1
         1         1         2
/PART/1
spring
         1         1
/MAT/LAW1/1
dummy
   7.8e-6
     210.0       0.3
/PROP/SPRING/1
spring
       1.0       {k}       0.0
/GRNOD/NODE/1
left
1
/GRNOD/NODE/2
right
2
/BCS/1
fix left
       111       000         0         1
/BCS/2
axial only
       011       000         0         2
/FUNCT/1
ramp
       0.0       0.0
     100.0     100.0
/CLOAD/1
pull
         1         X         2       {F}
/END
"""


def _truss_deck(L, area, F, mat=STEEL2):
    return f"""\
#RADIOSS STARTER
/BEGIN
TRS
/NODE
         1       0.0       0.0       0.0
         2     {L}       0.0       0.0
/TRUSS/1
         1         1         2
/PART/1
bar
         1         1
{mat}/PROP/TRUSS/1
bar
       {area}
/GRNOD/NODE/1
pivot
1
/GRNOD/NODE/2
tip
2
/BCS/1
pin
       111       000         0         1
/BCS/2
axial only
       011       000         0         2
/FUNCT/1
ramp
       0.0       0.0
     100.0     100.0
/CLOAD/1
pull
         1         X         2       {F}
/END
"""


# ---- damped SDOF (the M10 truss oscillator + /IMPL/DYNA/DAMP) ---------------

L_SD, E_SD, A_SD, RHO_SD = 100.0, 210.0, 1.0, 7.8e-6
M_SD = RHO_SD * A_SD * L_SD / 2.0
K_SD = E_SD * A_SD / L_SD
OM_SD = np.sqrt(K_SD / M_SD)


def _sdof_deck(v0):
    return f"""\
#RADIOSS STARTER
/BEGIN
SDOFD
/NODE
         1{0.0:20.10f}{0.0:20.10f}{0.0:20.10f}
         2{L_SD:20.10f}{0.0:20.10f}{0.0:20.10f}
/TRUSS/1
         1         1         2
/PART/1
bar
         1         1
/MAT/LAW1/1
steel
   {RHO_SD}
     {E_SD}       0.0
/PROP/TRUSS/1
bar
       {A_SD}
/GRNOD/NODE/1
pivot
1
/GRNOD/NODE/2
tip
2
/BCS/1
pin
       111       000         0         1
/BCS/2
axial only
       011       000         0         2
/INIVEL/TRA/1
kick
      {v0}       0.0       0.0         2
/END
"""


# ----------------------------------------------------------------------------
# FD residual/tangent consistency for the four NEW element tangents
# ----------------------------------------------------------------------------

def _fd_check(model, tol, prep=None):
    """Central-difference check of the nonlinear-geometry residual against
    the assembled tangent (material + hourglass + K_geo) at u = 0 — the M9
    pattern, applied to the M11 elements."""
    from pyradioss.implicit.assembly import assemble
    from pyradioss.implicit.dofmap import DofMap
    from pyradioss.implicit.statics import (_internal_forces, _restore,
                                            _snapshot)
    if prep is not None:
        prep(model)
    committed = {name: _snapshot(g) for name, g in model.element_groups()}
    dof = DofMap(model)

    def residual(u_eq):
        u, ur = dof.scatter_solution(u_eq)
        fint, mint = _internal_forces(model, model.x0, u, ur, committed,
                                      nlgeom=True)
        return dof.gather_residual(fint, mint)

    for name, g in model.element_groups():
        _restore(g, committed[name])
    K = assemble(model, dof, model.x0, None, kgeo=True).toarray()
    h = 1e-6 * max(float(np.abs(model.x0).max()), 1.0)
    Kfd = np.empty_like(K)
    for j in range(dof.ndof):
        e = np.zeros(dof.ndof)
        e[j] = h
        Kfd[:, j] = -(residual(e) - residual(-e)) / (2.0 * h)
    scale = float(np.abs(K).max())
    err = np.abs(Kfd - K).max() / scale
    assert err <= tol, f"FD tangent mismatch: rel err {err:.3e} > {tol:g}"


_TET1 = ("/NODE\n1 0 0 0\n2 1 0 0\n3 0 1 0\n4 0 0 1\n"
         "/TETRA4/1\n1 1 2 3 4\n"
         "/PART/1\ntet\n1 1\n" + STEEL1 +
         "/PROP/SOLID/1\nsolid\n0.0 0.0 0.1\n")
_TRI1 = ("/NODE\n1 0 0 0\n2 1 0 0\n3 0 1 0\n"
         "/SH3N/1\n1 1 2 3\n"
         "/PART/1\ntri\n1 1\n" + STEEL1 +
         "/PROP/SHELL/1\nshell\n1 0 0 0\n0.01 0.01 0.01 0 0\n3 0 0.1\n")
_BEA1 = ("/NODE\n1 0 0 0\n2 10 0 0\n9 0 50 0\n"
         "/BEAM/1\n1 1 2 9\n"
         "/PART/1\nbeam\n1 1\n" + STEEL1 +
         "/PROP/BEAM/1\nsq\n25.0 52.08 52.08 88.0\n")
_SPR1 = ("/NODE\n1 0 0 0\n2 2 0 0\n"
         "/SPRING/1\n1 1 2\n"
         "/PART/1\nspr\n1 1\n" + STEEL1 +
         "/PROP/SPRING/1\nk\n1.0 4.0 0.0\n")


def _body(name, body):
    return f"#RADIOSS STARTER\n/BEGIN\n{name}\n{body}/END\n"


def test_fd_consistency_zero_stress(make_deck):
    """All four M11 element tangents linearize the nonlinear-geometry
    residual EXACTLY at zero stress (central differences, 2e-5 of the
    stiffness scale — machine-precision-limited for tetra/beam/spring;
    the sh3n's corotational frame recomputation costs a few more digits)."""
    for tag, body in (("FDTE", _TET1), ("FDTR", _TRI1),
                      ("FDBE", _BEA1), ("FDSP", _SPR1)):
        model = _starter_model(make_deck, tag, _body(tag, body))
        _fd_check(model, tol=2e-5)


def test_fd_consistency_prestressed_exact(make_deck):
    """The beam's axial K_geo and the spring's taut-string term are the
    EXACT frame-rotation derivatives of their residuals — FD-tight
    including prestress (the corotational-truss property of M9, inherited
    by the two axial-force elements)."""
    model = _starter_model(make_deck, "FDBP", _body("FDBP", _BEA1))

    def beam_pre(mm):
        mm.beams.state["fres"][:, 0] = -3.0       # axial compression
    _fd_check(model, tol=1e-6, prep=beam_pre)

    model = _starter_model(make_deck, "FDSPP", _body("FDSPP", _SPR1))

    def spr_pre(mm):
        st = mm.springs.state
        st["L0"][:] = 1.5                          # stretched at x0
        st["force"][:] = st["k"] * (2.0 - 1.5)
    _fd_check(model, tol=1e-6, prep=spr_pre)


def test_fd_consistency_prestressed_sigma_over_E(make_deck):
    """Stressed tetra/sh3n states: K_geo captures the initial-stress term;
    the remaining FD gap is the OMITTED Jaumann-spin/frame derivative
    (documented — the same class the M9 hexa carries, checked there at
    zero stress only). Its size must be the sigma/E scale, nothing more."""
    model = _starter_model(make_deck, "FDTEP", _body("FDTEP", _TET1))

    def tet_pre(mm):
        mm.tetras.state["sig"][...] = [-0.4, 0.1, 0.05, 0.02, -0.03, 0.01]
    _fd_check(model, tol=5e-3, prep=tet_pre)

    model = _starter_model(make_deck, "FDTRP", _body("FDTRP", _TRI1))

    def tri_pre(mm):
        mm.sh3n.state["sig"][..., 0] = -0.5        # membrane compression
    _fd_check(model, tol=5e-3, prep=tri_pre)


def test_tangent_does_not_touch_force_state(make_deck):
    """tangent()/kgeo() are read-only linearizations alongside forces() —
    the M7/M8 parity contract, asserted for the four NEW element tangents
    exactly like test_m8 asserts it for hexa/BT4."""
    from pyradioss.elements import beam_type3, shell_tri3, solid_tetra4, \
        spring
    cases = (("STTE", _TET1, "tetras", solid_tetra4),
             ("STTR", _TRI1, "sh3n", shell_tri3),
             ("STBE", _BEA1, "beams", beam_type3),
             ("STSP", _SPR1, "springs", spring))
    for tag, body, gname, kernel in cases:
        model = _starter_model(make_deck, tag, _body(tag, body))
        g = getattr(model, gname)
        before = {k: v.copy() for k, v in g.state.items()
                  if isinstance(v, np.ndarray)}
        kernel.tangent(g, model.x0, None)
        kernel.kgeo(g, model.x0)
        for k, v in before.items():
            assert np.array_equal(g.state[k], v), \
                f"{gname}.tangent/kgeo mutated state[{k}]"


# ----------------------------------------------------------------------------
# Closed-form statics, one per new element
# ----------------------------------------------------------------------------

def test_tetra_uniaxial_patch(make_deck):
    """A Kuhn-split tetra bar under displacement control: the exact
    solution is the UNIFORM uniaxial-stress state (sigma = E u/L, free
    lateral contraction) — every tet must carry it exactly and Newton must
    converge in one step (linear elastic): the constant-strain patch test."""
    L, W, eps = 10.0, 4.0, 1e-3
    starter, nid = _tet_bar_deck(2, 2, 2, L, W, ux_end=eps * L)
    model = _run(make_deck, "TPATCH", starter,
                 "#\n/RUN/TPATCH/1\n1.0\n/IMPL\n/END\n")
    res = model.implicit_result
    assert res.converged
    sig = model.tetras.state["sig"]
    assert np.allclose(sig[:, 0], 210.0 * eps, rtol=1e-9)   # uniform sigma_xx
    assert np.abs(sig[:, 1:]).max() < 1e-12                 # everything else 0
    # free lateral contraction -nu*eps on the far lateral face
    uy = (model.x - model.x0)[model.node_index(nid(1, 2, 1)), 1]
    assert uy == pytest.approx(-0.3 * eps * W, rel=1e-9)
    # linear problem: quadratic Newton = converged after ONE solve
    assert all(i.iterations <= 2 for i in res.increments)


def test_sh3n_membrane_uniaxial_exact(make_deck):
    """CST membrane pull with consistent tip traction: uniform plane-stress
    uniaxial state, tip displacement sigma*L/E exactly, one Newton step."""
    Lx, b, t, sigma = 40.0, 10.0, 1.0, 0.1
    F = sigma * b * t
    starter, nid = _sh3n_strip_deck(8, 2, Lx, b, t, "X", F, nu="0.3",
                                    membrane_bcs=True)
    model = _run(make_deck, "TRIMEM", starter,
                 "#\n/RUN/TRIMEM/1\n1.0\n/IMPL\n/END\n")
    assert model.implicit_result.converged
    ux = (model.x - model.x0)[model.node_index(nid(8, 1)), 0]
    assert ux == pytest.approx(sigma / 210.0 * Lx, rel=1e-9)
    # the layer stress is stored in each triangle's COROTATIONAL frame (the
    # diagonal triangles' e1 lies along the diagonal), so compare the
    # frame-invariant plane-stress von Mises: uniform = the patch state
    sig = model.sh3n.state["sig"]
    vm = np.sqrt(sig[..., 0] ** 2 - sig[..., 0] * sig[..., 1]
                 + sig[..., 1] ** 2 + 3.0 * sig[..., 2] ** 2)
    assert np.allclose(vm, sigma, rtol=1e-9)
    assert all(i.iterations <= 2 for i in model.implicit_result.increments)


def test_sh3n_thick_cantilever_timoshenko(make_deck):
    """A THICK (L/t = 5) sh3n cantilever strip vs the Timoshenko closed
    form FL^3/3EI + FL/(kappa G A) to 2% (measured -0.7% at 24x6). The
    shear-dominated thick strip is the honest tight anchor for the C0
    triangle: THIN C0 meshes shear-lock (the module docstring's accuracy
    note — the same 40x10x1 strip is 8% stiff even at 32x4)."""
    E, G, kap = 210.0, 105.0, 5.0 / 6.0
    Lx, b, t, F = 20.0, 10.0, 4.0, 0.5
    I = b * t ** 3 / 12.0
    d_ref = F * Lx ** 3 / (3 * E * I) + F * Lx / (kap * G * b * t)
    starter, nid = _sh3n_strip_deck(24, 6, Lx, b, t, "Z", F)
    model = _run(make_deck, "TRICAN", starter,
                 "#\n/RUN/TRICAN/1\n1.0\n/IMPL\n/END\n")
    assert model.implicit_result.converged
    uz = (model.x - model.x0)[model.node_index(nid(24, 3)), 2]
    assert uz == pytest.approx(d_ref, rel=0.02)


def test_beam_cantilever_closed_form(make_deck):
    """The 10-element beam cantilever vs the Timoshenko closed form
    FL^3/3EI + FL/GA (full-section shear, the TYPE3 default) to 0.5% —
    measured -0.25%, converging O(h^2). One Newton step (linear)."""
    L, A, Iy, Ix, F = 100.0, 25.0, 52.08, 88.0, 0.05
    E, G = 210.0, 210.0 / 2.6
    starter = _beam_deck(10, L, A, Iy, Ix, F)
    model = _run(make_deck, "BEAMC", starter,
                 "#\n/RUN/BEAMC/1\n1.0\n/IMPL\n/END\n")
    res = model.implicit_result
    assert res.converged
    d_ref = F * L ** 3 / (3 * E * Iy) + F * L / (G * A)
    uz = (model.x - model.x0)[model.node_index(11), 2]
    assert uz == pytest.approx(d_ref, rel=0.005)
    assert all(i.iterations <= 2 for i in res.increments)


def test_spring_k_du_exact(make_deck):
    """The TYPE4 spring under an axial pull: u = F/k EXACTLY in one Newton
    step (the total-form residual linearized at the committed frame is
    exactly K u — the spring's own implicit path)."""
    k, F = 4.0, 1.0
    model = _run(make_deck, "SPRK", _spring_deck(k, F),
                 "#\n/RUN/SPRK/1\n1.0\n/IMPL\n/END\n")
    res = model.implicit_result
    assert res.converged
    ux = (model.x - model.x0)[model.node_index(2), 0]
    assert ux == pytest.approx(F / k, rel=1e-12)
    assert all(i.iterations <= 2 for i in res.increments)
    # the stored force state committed to the converged value
    assert model.springs.state["force"][0] == pytest.approx(F, rel=1e-12)

    # MULTI-increment load stepping must accumulate on the committed force
    # state (the frozen-frame path feeds only each increment's u — a
    # regression guard for exactly that: geometry-only evaluation loses the
    # previous increments)
    model = _run(make_deck, "SPRK2", _spring_deck(k, F),
                 "#\n/RUN/SPRK2/1\n1.0\n/IMPL\n/IMPL/DTINI\n0.25\n/END\n")
    ux = (model.x - model.x0)[model.node_index(2), 0]
    assert ux == pytest.approx(F / k, rel=1e-12)
    # ...and the spring rings with a CLOSED ledger under implicit dynamics
    model = _run(make_deck, "SPRD", _spring_deck(k, F),
                 "#\n/RUN/SPRD/1\n10.0\n/IMPL/DYNA/2\n0.5 0.25\n"
                 "/IMPL/DTINI\n0.05\n/END\n")
    h = model.implicit_result.history
    assert abs(h["bal"][-1]) < 1e-10 * max(max(h["ke"]), 1e-12)


def test_mixed_element_model_converges(make_deck):
    """One deck with ALL implicit-capable families at once (brick + tetra +
    BT4 + sh3n + truss + spring + beam): the M11 'implicit runs accept full
    models' acceptance — the assembler takes every group and Newton
    converges in one step on the linear-elastic composite."""
    starter = f"""\
#RADIOSS STARTER
/BEGIN
MIXED
/NODE
         1 0 0 0
         2 4 0 0
         3 4 4 0
         4 0 4 0
         5 0 0 4
         6 4 0 4
         7 4 4 4
         8 0 4 4
        11 10 0 0
        12 11 0 0
        13 10 1 0
        14 10 0 1
        21 20 0 0
        22 21 0 0
        23 21 1 0
        24 20 1 0
        31 30 0 0
        32 31 0 0
        33 31 1 0
        41 40 0 0
        42 45 0 0
        49 40 5 0
        51 50 0 0
        52 52 0 0
        61 60 0 0
        62 62 0 0
/BRICK/1
         1         1         2         3         4         5         6         7         8
/TETRA4/2
         2        11        12        13        14
/SHELL/3
         3        21        22        23        24
/SH3N/4
         4        31        32        33
/BEAM/5
         5        41        42        49
/TRUSS/6
         6        51        52
/SPRING/7
         7        61        62
/PART/1
brick
         1         1
/PART/2
tet
         1         1
/PART/3
quad
         2         1
/PART/4
tri
         2         1
/PART/5
beam
         3         1
/PART/6
truss
         4         1
/PART/7
spring
         5         1
{STEEL1}/PROP/SOLID/1
solid
       0.0       0.0       0.1
/PROP/SHELL/2
shell
         1         0         0         0
      0.01      0.01      0.01         0         0
         3         0       0.5
/PROP/BEAM/3
beam
      25.0     52.08     52.08      88.0
/PROP/TRUSS/4
bar
       1.0
/PROP/SPRING/5
spring
       1.0       4.0       0.0
/GRNOD/NODE/1
grounds
1 2 3 4 11 13 14 21 24 31 41 51 61
/GRNOD/NODE/2
loaded
5 12 22 32 42 52 62
/GRNOD/NODE/3
axial elements tips
52 62
/BCS/1
ground
       111       111         0         1
/BCS/2
truss+spring lateral (axial stiffness only)
       011       000         0         3
/FUNCT/1
ramp
       0.0       0.0
     100.0     100.0
/CLOAD/1
poke
         1         X         2       0.01
/END
"""
    model = _run(make_deck, "MIXED", starter,
                 "#\n/RUN/MIXED/1\n1.0\n/IMPL\n/END\n")
    res = model.implicit_result
    assert res.converged
    assert all(i.iterations <= 2 for i in res.increments)
    disp = model.x - model.x0
    # every loaded substructure moved, none exploded
    for nid_ in (5, 12, 22, 32, 42, 52, 62):
        u = np.abs(disp[model.node_index(nid_)]).max()
        assert 1e-8 < u < 10.0, f"node {nid_}: |u| = {u}"
    # the spring's answer inside the composite is still exactly F/k
    assert disp[model.node_index(62), 0] == pytest.approx(0.01 / 4.0,
                                                          rel=1e-9)


# ----------------------------------------------------------------------------
# LAW2 consistent tangents: plane-stress shells and the truss
# ----------------------------------------------------------------------------

def _assert_quadratic_tail(res):
    """The M8 quadratic-convergence proxy: in the deepest plastic
    increment the residual ratio shrinks superlinearly across the tail."""
    plastic = [i for i in res.increments if i.iterations > 2]
    assert plastic, "expected at least one plastic (multi-iter) increment"
    inc = max(plastic, key=lambda i: i.iterations)
    r = np.array(inc.residuals)
    r = r[r > 0]
    if r[-1] <= 1e-12 * r[0]:
        # the residual crashed 12+ orders below its start — a Newton that
        # hit the round-off floor; the accelerating-ratio proxy below is
        # meaningless at the floor (ratios saturate), and this IS the
        # quadratic signature
        return
    tail = r[-3:]
    assert len(tail) == 3
    ratio1 = tail[1] / tail[0]
    ratio2 = tail[2] / tail[1]
    assert ratio2 < ratio1 ** 1.5


_QUAD_LAW2 = f"""\
#RADIOSS STARTER
/BEGIN
QPL
/NODE
         1       0.0       0.0       0.0
         2      10.0       0.0       0.0
         3      10.0      10.0       0.0
         4       0.0      10.0       0.0
/SHELL/1
         1         1         2         3         4
/PART/1
q
         1         1
{STEEL2}/PROP/SHELL/1
shell
         1         0         0         0
      0.01      0.01      0.01         0         0
         3         0       1.0
/GRNOD/NODE/1
x0
1 4
/GRNOD/NODE/2
x1
2 3
/GRNOD/NODE/3
n1
1
/GRNOD/NODE/4
all
1 2 3 4
/BCS/1
fixx
       100       000         0         1
/BCS/2
fixy n1
       010       000         0         3
/BCS/3
membrane only
       001       111         0         4
/FUNCT/1
ramp
       0.0       0.0
     100.0     100.0
/CLOAD/1
pull
         1         X         2       {0.5 * 10.0 * 1.0 / 2.0}
/END
"""


def test_law2_shell_uniaxial_quadratic(make_deck):
    """BT4 membrane pull past yield with LAW2 (M11 — the plane-stress
    consistent tangent): eps_p lands EXACTLY on the Johnson–Cook curve
    (sigma = A + B eps_p^n at sigma = 0.5 -> eps_p = 0.04), the axial
    displacement matches the Iplas=2 algorithm's OWN closed form

        u/L = sigma/E + (3G/E) eps_p

    (the radial PROJECTION scales the whole in-plane stress, which makes
    the axial plastic flow (3G/E) eps_p = 1.154 eps_p at nu = 0.3 — the
    exact plane-stress return would give eps_p itself: a documented
    property of the ported Iplas=2 variant, not an error), and the
    plastic increments converge with the QUADRATIC tail only the
    algorithmic tangent delivers."""
    sigma = 0.5
    model = _run(make_deck, "QLAW2", _QUAD_LAW2,
                 "#\n/RUN/QLAW2/1\n1.0\n/IMPL/DTINI\n0.1\n"
                 "/IMPL/NEWTON\n1e-10  40\n/END\n")
    res = model.implicit_result
    assert res.converged
    epsp_a = ((sigma - A_JC) / B_JC) ** (1.0 / N_JC)
    epsp = model.shells.state["epsp"][0]
    assert epsp == pytest.approx(epsp_a, rel=1e-6)          # all layers equal
    sig = model.shells.state["sig"][0]
    assert sig[:, 0] == pytest.approx(sigma, rel=1e-6)
    assert np.abs(sig[:, 1:]).max() < 1e-8
    ux = (model.x - model.x0)[model.node_index(2), 0]
    u_ref = (sigma / E_JC + (3.0 * G_JC / E_JC) * epsp_a) * 10.0
    assert ux == pytest.approx(u_ref, rel=1e-6)
    _assert_quadratic_tail(res)


def test_law2_sh3n_uniaxial_quadratic(make_deck):
    """The same plane-stress LAW2 pull through the sh3n (CST) tangent path
    — the triangle integrates the same per-layer consistent tangents with
    its own operators."""
    Lx, b, t, sigma = 20.0, 10.0, 1.0, 0.5
    F = sigma * b * t
    starter, nid = _sh3n_strip_deck(2, 1, Lx, b, t, "X", F, mat=STEEL2,
                                    membrane_bcs=True)
    model = _run(make_deck, "TLAW2", starter,
                 "#\n/RUN/TLAW2/1\n1.0\n/IMPL/DTINI\n0.1\n"
                 "/IMPL/NEWTON\n1e-10  40\n/END\n")
    res = model.implicit_result
    assert res.converged
    epsp_a = ((sigma - A_JC) / B_JC) ** (1.0 / N_JC)
    assert model.sh3n.state["epsp"] == pytest.approx(epsp_a, rel=1e-6)
    ux = (model.x - model.x0)[model.node_index(nid(2, 0)), 0]
    u_ref = (sigma / E_JC + (3.0 * G_JC / E_JC) * epsp_a) * Lx
    assert ux == pytest.approx(u_ref, rel=1e-6)
    _assert_quadratic_tail(res)


def test_law2_truss_elastoplastic(make_deck):
    """Elastoplastic truss pull past yield (M11 — the M9 deferral): the
    stress sits on the JC curve and the displacement matches the 1-D
    closed form u = L (sigma/E + eps_p) to the one-step-return tolerance
    (the truss kernel takes a single linearized return per increment, so
    the curve is followed to O(d_lambda^2) — 0.1 load steps keep it
    inside 1e-3), with the quadratic Newton tail of the matching
    consistent modulus E H0/(E + H0)."""
    L, area, sigma = 10.0, 1.0, 0.5
    F = sigma * area
    model = _run(make_deck, "TRSPL", _truss_deck(L, area, F),
                 "#\n/RUN/TRSPL/1\n1.0\n/IMPL/DTINI\n0.05\n"
                 "/IMPL/NEWTON\n1e-10  40\n/END\n")
    res = model.implicit_result
    assert res.converged
    epsp_a = ((sigma - A_JC) / B_JC) ** (1.0 / N_JC)
    epsp = float(model.trusses.state["epsp"][0])
    sig = float(model.trusses.state["sig"][0])
    assert sig == pytest.approx(sigma, rel=1e-9)      # equilibrium is exact
    assert epsp == pytest.approx(epsp_a, rel=5e-3)    # one-step return drift
    ux = (model.x - model.x0)[model.node_index(2), 0]
    assert ux == pytest.approx((sigma / E_JC + epsp) * L, rel=1e-6)
    _assert_quadratic_tail(res)


def test_law2_tetra_plastic(make_deck):
    """LAW2 tetra bar under displacement control past yield: the shared M8
    solid consistent tangent through the NEW tetra B-operator — the axial
    stress solves sigma/E + eps_p(sigma) = eps on the JC curve, uniform
    over the patch, with a quadratic tail."""
    from scipy.optimize import brentq
    L, W, eps = 10.0, 4.0, 0.05
    starter, nid = _tet_bar_deck(1, 1, 1, L, W, mat=STEEL2, ux_end=eps * L)
    model = _run(make_deck, "TETPL", starter,
                 "#\n/RUN/TETPL/1\n1.0\n/IMPL/DTINI\n0.1\n"
                 "/IMPL/NEWTON\n1e-10  40\n/END\n")
    res = model.implicit_result
    assert res.converged
    sig_a = brentq(
        lambda s: s / E_JC + ((s - A_JC) / B_JC) ** (1.0 / N_JC) - eps,
        A_JC, 2.0)
    epsp_a = ((sig_a - A_JC) / B_JC) ** (1.0 / N_JC)
    sig = model.tetras.state["sig"]
    assert np.allclose(sig[:, 0], sig_a, rtol=2e-3)
    assert np.allclose(model.tetras.state["epsp"], epsp_a, rtol=5e-3)
    _assert_quadratic_tail(res)


# ----------------------------------------------------------------------------
# /IMPL/DYNA/DAMP — Rayleigh damping vs the closed-form damped SDOF
# ----------------------------------------------------------------------------

def _damped_sdof(make_deck, tag, a, b, zeta):
    om_d = OM_SD * np.sqrt(1.0 - zeta ** 2)
    T = 2.0 * np.pi / om_d
    dt = T / 200.0
    t_end = 3.0 * T
    model = _run(make_deck, tag, _sdof_deck(1.0),
                 f"#\n/RUN/{tag}/1\n{t_end}\n/IMPL/DYNA/2\n0.5 0.25\n"
                 f"/IMPL/DYNA/DAMP\n{a} {b}\n/IMPL/DTINI\n{dt}\n/END\n")
    res = model.implicit_result
    assert res.converged
    h = res.history
    t = np.array(h["t"])
    idx = model.node_index(2)
    u = np.array([uu[idx, 0] for uu in h["u"]])
    # closed form: u = (v0/om_d) e^(-zeta om t) sin(om_d t), v0 = 1
    uref = (1.0 / om_d) * np.exp(-zeta * OM_SD * t) * np.sin(om_d * t)
    err = np.abs(u - uref).max() / np.abs(uref).max()
    assert err < 5e-3, f"{tag}: envelope/phase error {err:.2e}"

    # damped PERIOD from zero up-crossings
    s = np.sign(u)
    ups = np.where((s[:-1] < 0) & (s[1:] >= 0))[0]
    assert len(ups) >= 2

    def cross(i):
        return t[i] + (t[i + 1] - t[i]) * (-u[i]) / (u[i + 1] - u[i])
    T_meas = (cross(ups[-1]) - cross(ups[0])) / (len(ups) - 1)
    assert T_meas == pytest.approx(T, rel=2e-4)

    # the ledger: dissipation positive and growing, balance to round-off
    edamp = np.array(h["edamp"])
    assert edamp[-1] > 0.0 and np.all(np.diff(edamp) >= -1e-15)
    ke0 = 0.5 * M_SD                       # (1/2) m v0^2
    assert abs(h["bal"][-1]) < 1e-12 * ke0
    # after 3 damped periods most of the initial energy sits in the ledger:
    # E_diss/E0 = 1 - e^(-2 zeta om (3T)) exactly for the closed form
    frac = edamp[-1] / ke0
    frac_ref = 1.0 - np.exp(-2.0 * zeta * OM_SD * t[-1])
    assert frac == pytest.approx(frac_ref, rel=2e-2)


def test_damp_mass_only(make_deck):
    """C = a M alone: a = 2 zeta omega gives the closed-form decay."""
    zeta = 0.05
    _damped_sdof(make_deck, "DMA", 2.0 * zeta * OM_SD, 0.0, zeta)


def test_damp_stiffness_only(make_deck):
    """C = b K alone: b = 2 zeta / omega gives the same zeta on the SDOF
    (exercises the K v product and the S0-type tangent scaling)."""
    zeta = 0.05
    _damped_sdof(make_deck, "DMB", 0.0, 2.0 * zeta / OM_SD, zeta)


def test_damp_mixed(make_deck):
    """C = a M + b K mixed: zeta = (a/om + b om)/2."""
    a, b = 0.05 * OM_SD, 0.05 / OM_SD
    zeta = 0.5 * (a / OM_SD + b * OM_SD)
    _damped_sdof(make_deck, "DMM", a, b, zeta)


def test_damp_undamped_unchanged(make_deck):
    """Without the DAMP card the M10 behaviour is bit-identical: zero
    dissipation channel, trapezoidal conservation to round-off."""
    T = 2.0 * np.pi / OM_SD
    model = _run(make_deck, "DM0", _sdof_deck(1.0),
                 f"#\n/RUN/DM0/1\n{2*T}\n/IMPL/DYNA/2\n0.5 0.25\n"
                 f"/IMPL/DTINI\n{T/100}\n/END\n")
    h = model.implicit_result.history
    assert all(e == 0.0 for e in h["edamp"])
    assert abs(h["bal"][-1]) < 1e-12 * 0.5 * M_SD


# ----------------------------------------------------------------------------
# Automatic implicit step control (imp_dt.F)
# ----------------------------------------------------------------------------

@pytest.mark.slow          # measured 155.4 s serial (2026-07 full-suite run)
def test_autodt_statics_cut_and_complete(make_deck):
    """The deep-elastica strip (PL^2/EI = 2) asked to reach the full load
    in ONE /IMPL/NONLIN increment with a tight Newton budget: without the
    M11 control this run STOPS (non-convergence — the M10 behaviour); with
    imp_dt.F it cuts the increment, converges, grows back and finishes —
    matching the fine fixed-increment answer to 0.5%."""
    from tests.test_m9_geomnl import _strip_deck
    Lx, b, t = 40.0, 10.0, 1.0
    I = b * t ** 3 / 12.0
    F = 2.0 * 210.0 * I / Lx ** 2                 # PL^2/EI = 2
    starter, nid = _strip_deck(16, 2, Lx, b, t, "Z", F)

    model, out = _run_capture(
        make_deck, "ADT1", starter,
        "#\n/RUN/ADT1/1\n1.0\n/IMPL/NONLIN\n/IMPL/DTINI\n1.0\n"
        "/IMPL/NEWTON\n1e-8  8\n/END\n")
    res = model.implicit_result
    assert res.converged, res.stop_reason
    assert "DECREASED" in out                     # the cut actually happened
    assert any(not i.converged for i in res.increments)

    model2 = _run(make_deck, "ADT2", starter,
                  "#\n/RUN/ADT2/1\n1.0\n/IMPL/NONLIN\n/IMPL/DTINI\n0.05\n"
                  "/IMPL/NEWTON\n1e-8  25\n/END\n")
    tip = model.node_index(nid(16, 1))
    u1 = (model.x - model.x0)[tip]
    u2 = (model2.x - model2.x0)[tip]
    assert np.abs(u1 - u2).max() <= 5e-3 * np.abs(u2).max()


def test_autodt_smooth_run_reproduces_fixed_dt(make_deck):
    """A smooth run never triggers the control: zero cuts, zero growth
    beyond /IMPL/DTINI (dt_max defaults to the DTINI value), so the M10
    fixed-dt stepping is reproduced exactly — same step count, same
    history."""
    T = 2.0 * np.pi / OM_SD
    model, out = _run_capture(
        make_deck, "ADTS", _sdof_deck(1.0),
        f"#\n/RUN/ADTS/1\n{2*T}\n/IMPL/DYNA/2\n0.5 0.25\n"
        f"/IMPL/DTINI\n{T/100}\n/END\n")
    assert model.implicit_result.converged
    assert "DECREASED" not in out and "INCREASED" not in out
    t = np.array(model.implicit_result.history["t"])
    # uniform steps at exactly DTINI (the final step lands on t_end)
    assert np.allclose(np.diff(t)[:-1], T / 100, rtol=1e-9)


def test_autodt_dynamics_cut_and_complete(make_deck):
    """Dynamics side of the control: the M10 large-rotation truss pendulum
    (released from 60 degrees under gravity, /IMPL/NONLIN) driven with a
    Newton budget too tight for the requested coarse step — the run must
    complete through automatic time-step cuts and keep the energy balance
    closed."""
    L, g = 100.0, 9.81e-3
    th0 = np.pi / 3.0
    x2, y2 = L * np.sin(th0), -L * np.cos(th0)
    T_lin = 2.0 * np.pi * np.sqrt(L / g)
    starter = f"""\
#RADIOSS STARTER
/BEGIN
PENDA
/NODE
         1{0.0:20.10f}{0.0:20.10f}{0.0:20.10f}
         2{x2:20.10f}{y2:20.10f}{0.0:20.10f}
/TRUSS/1
         1         1         2
/PART/1
bar
         1         1
/MAT/LAW1/1
steel
   7.8e-6
     210.0       0.0
/PROP/TRUSS/1
bar
       1.0
/GRNOD/NODE/1
pivot
1
/GRNOD/NODE/2
tip
2
/BCS/1
pin
       111       000         0         1
/BCS/2
in plane
       001       000         0         2
/FUNCT/1
const gravity
       0.0       1.0
   10000.0       1.0
/GRAV/1
g down
         1        Y         0      {-g}
/END
"""
    model, out = _run_capture(
        make_deck, "ADTD", starter,
        f"#\n/RUN/ADTD/1\n{0.4 * T_lin}\n/IMPL/DYNA/2\n0.5 0.25\n"
        f"/IMPL/NONLIN\n/IMPL/DTINI\n{T_lin / 25}\n"
        f"/IMPL/NEWTON\n1e-8  4\n/END\n")
    res = model.implicit_result
    assert res.converged, res.stop_reason
    assert "DECREASED" in out
    # the ledger stays closed to the COARSE-dt Newmark discretization drift
    # (trapezoidal conservation is exact only for linear systems; at
    # dt = T/25 on the nonlinear pendulum the O((omega dt)^2) drift is a
    # few %) — the check is that the cuts did not LEAK energy beyond it
    h = res.history
    ke_max = max(max(h["ke"]), 1e-12)
    assert abs(h["bal"][-1]) < 0.05 * ke_max


def test_impl_dt_card_parsing(tmp_path):
    """/IMPL/DT/STOP reads dt_min dt_max; /IMPL/DT/1 reads the IDTC = 1
    card (it_w sc_up it_dn sc_dn — freimpl.F field order); /IMPL/DT/2
    warns (IDTC 2/3 deferred)."""
    from pyradioss.common.messages import MessageLog
    from pyradioss.input.deck_reader import read_deck
    from pyradioss.input.engine_keywords import parse_engine_deck
    p = tmp_path / "E_0001.rad"
    p.write_text("#\n/RUN/A/1\n1.0\n/IMPL\n/IMPL/DT/STOP\n1e-3  0.5\n"
                 "/IMPL/DT/1\n4  1.25  20  0.7\n/END\n")
    log = MessageLog()
    with contextlib.redirect_stdout(io.StringIO()):
        ec = parse_engine_deck(read_deck(str(p)), log)
    assert ec.impl_dt_min == pytest.approx(1e-3)
    assert ec.impl_dt_max == pytest.approx(0.5)
    assert ec.impl_dt_itw == 4
    assert ec.impl_dt_scaleup == pytest.approx(1.25)
    assert ec.impl_dt_scaledn == pytest.approx(0.7)

    p.write_text("#\n/RUN/A/1\n1.0\n/IMPL\n/IMPL/DT/2\n6 1 6 0.5\n/END\n")
    log = MessageLog()
    with contextlib.redirect_stdout(io.StringIO()):
        parse_engine_deck(read_deck(str(p)), log)
    assert any("IMPL/DT" in w or "deferred" in w for w in log.warnings)


# ----------------------------------------------------------------------------
# /IMPL/BUCKL — the engine card on the M9 Euler column
# ----------------------------------------------------------------------------

def test_buckl_card_euler_column(make_deck):
    """The M9 clamped-free BT4 strip column through the CARD path this
    time: /IMPL runs the prestress increments, /IMPL/BUCKL/1 extracts and
    REPORTS the factors — P_cr = factors[0]*P0 = pi^2 EI/(4L^2) within 3%,
    the mode transverse, the listing carrying the imp_buck.F block."""
    from tests.test_m9_geomnl import _strip_deck
    Lx, b, t, E = 100.0, 10.0, 1.0, 210.0
    P0 = 0.002
    starter, nid = _strip_deck(20, 2, Lx, b, t, "X", -P0)
    model, out = _run_capture(
        make_deck, "BUCKC", starter,
        "#\n/RUN/BUCKC/1\n1.0\n/IMPL\n/IMPL/BUCKL/1\n0 0 3\n/END\n")
    res = model.implicit_result
    assert res.converged
    assert res.buckling_factors is not None
    assert len(res.buckling_factors) == 3          # NBUCK from the card
    I = b * t ** 3 / 12.0
    P_euler = np.pi ** 2 * E * I / (4.0 * Lx ** 2)
    P_cr = res.buckling_factors[0] * P0
    assert P_cr == pytest.approx(P_euler, rel=0.03)
    du, dur = res.buckling_modes[0]
    assert np.abs(du[:, 2]).max() > 10.0 * np.abs(du[:, :2]).max()
    assert "BUCKLING MODES COMPUTATION" in out
    assert "CRITICAL LOAD" in out


def test_buckl_bare_card_obsolete(tmp_path):
    """A bare /IMPL/BUCKL errors like the original reader ('OBSOLETE,
    USING /IMPL/BUCKL/1 OR /IMPL/BUCKL/2')."""
    from pyradioss.common.messages import MessageLog
    from pyradioss.input.deck_reader import read_deck
    from pyradioss.input.engine_keywords import parse_engine_deck
    p = tmp_path / "E_0001.rad"
    p.write_text("#\n/RUN/A/1\n1.0\n/IMPL\n/IMPL/BUCKL\n0 0 4\n/END\n")
    log = MessageLog()
    with contextlib.redirect_stdout(io.StringIO()):
        ec = parse_engine_deck(read_deck(str(p)), log)
    assert ec.impl_buckl == 0
    assert any("obsolete" in e.lower() for e in log.errors)


def test_buckl_refused_under_dyna(make_deck):
    """/IMPL/BUCKL + /IMPL/DYNA is refused with a clear error (buckling is
    an eigensolve about a static prestressed state)."""
    s, e = make_deck("BUCKD", _sdof_deck(0.0),
                     "#\n/RUN/BUCKD/1\n1.0\n/IMPL/DYNA\n"
                     "/IMPL/BUCKL/1\n0 0 2\n/IMPL/DTINI\n0.1\n/END\n")
    with contextlib.redirect_stdout(io.StringIO()):
        run_starter(s)
        with pytest.raises(ValueError, match="BUCKL"):
            run_engine(e)


# ----------------------------------------------------------------------------
# Rate-device discipline for the new elements
# ----------------------------------------------------------------------------

def test_spring_dashpot_deferred_warns(make_deck):
    """A spring with c > 0 under the implicit solver: the dashpot is a
    rate device — disabled with an explicit warning (the M10 convention),
    and the static answer is the undamped u = F/k."""
    starter = _spring_deck(4.0, 1.0).replace(
        "       1.0       4.0       0.0", "       1.0       4.0       0.5")
    model, out = _run_capture(make_deck, "SPRC", starter,
                              "#\n/RUN/SPRC/1\n1.0\n/IMPL\n/END\n")
    assert "DEFERRED" in out and "dashpot" in out
    ux = (model.x - model.x0)[model.node_index(2), 0]
    assert ux == pytest.approx(0.25, rel=1e-12)
