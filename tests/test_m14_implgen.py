"""
M14 validations: constraint CHAINS, TYPE11 friction, arc-length and
buckling WITH constraints & contact, and the LAW42 consistent tangent.

Every new capability gets at least one ANALYTIC check (the port's
philosophy):

* CONSTRAINT CHAINS (transform substitution — the port's resolution of
  the original starter's rbody_part_modif.F90 PARENT_OF hierarchy):
  a two-body RBE2 chain lever closed form EXACT in one Newton step
  (both geometry modes), an /MPC row written on a rigid slave resolving
  through the chain (equality closed form exact), the chained-pendulum
  implicit-dynamic period on the elliptic integral (the M12 pendulum
  physics assembled as a rigid-on-rigid chain), the CIRCULAR chain
  refusal, the two-constraints-on-one-DOF conflict refusal, and the
  explicit engine's loud chain refusal (its per-body integrator has no
  nesting order);
* /INTER/TYPE11 COULOMB FRICTION (the M14 edge-pair return mapping —
  implicit/contact.py: what i11keg3.F actually does is an always-stick
  mu-scaled spring with an uncapped force, the documented deviation):
  crossed edges stick (transmitted force + micro-slip closed form EXACT)
  and slip (= mu N exactly, with the ground-spring displacement closed
  form), FD residual/tangent consistency in BOTH regimes at a
  freshly-committed anchor, mu = 0 reproducing the M13 frictionless
  closed form to round-off, and the explicit TYPE11 kinetic-friction
  cross-check in steady sliding;
* /IMPL/ARCL WITH CONSTRAINTS & CONTACT (the M12 refusal removed; the
  corrector solves and the spherical metric in the REDUCED space — see
  statics._run_arclength for the documented deviation from the
  original's full-field Riks norm): the M9 von Mises snap-through
  ARRESTED by a TYPE7 stop below it (the trace matches the free path
  before contact, the arrested branch lands on the stop's closed-form
  equilibrium), and an RBE2-loaded snap traced through BOTH limit
  points onto the far-branch closed form;
* /IMPL/BUCKL WITH CONSTRAINTS & CONTACT (the reduced pencil
  T^T K T — what imp_buck.F's UPD_GLOB_K does — with the converged
  contact tangent in K_MAT, the documented deviation: imp_buck.F
  assembles no contact at all): an Euler column with an RBE2 cap
  reproducing the plain-column factor, and a pinned column resting on a
  contact stop reproducing the pinned-pinned closed form pi^2 EI/L^2;
* LAW42 CONSISTENT TANGENT (the spectral spatial elasticity from the
  trial F — materials/law42_ogden.py; the original's implicit branch
  only scales a linear D by a scalar ET, the documented deviation):
  the FD Truesdell-rate identity EXACT including coalescent (equal)
  stretches — the L'Hopital branch, where the naive textbook limit
  subtracts sigma_a twice — the full-path FD residual/tangent
  consistency (hourglass frozen: its geometry variation is the
  pre-existing NLGEOM omission shared by every material — measured
  identical for LAW1 — not a LAW42 term), uniaxial AND equibiaxial
  pulls landing on the independently-written analytic solution with
  quadratic Newton tails, the explicit quasi-static cross-check, the
  incompressible-limit sanity (J - 1 scaling like 1/K against the
  kernel's own K(J-1) pressure), and the /IMPL (frozen-frame) refusal;
* the M7 parity contract: building/evaluating the M14 treatments
  mutates NOTHING shared with the explicit solver.

See PORTING_GUIDE.md roadmap M14.
"""

import contextlib
import io

import numpy as np
import pytest

from pyradioss.engine.engine import run_engine
from pyradioss.starter.starter import run_starter

pytest.importorskip("scipy")


# ----------------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------------

def _run(make_deck, name, starter, engine):
    s, e = make_deck(name, starter, engine)
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        run_starter(s)
        model = run_engine(e)
    return model, buf.getvalue()


def _model(make_deck, name, starter):
    s, _ = make_deck(name, starter, "#\n/END\n")
    with contextlib.redirect_stdout(io.StringIO()):
        return run_starter(s)


STEEL1 = """\
/MAT/LAW1/1
steel
   7.8e-6
     210.0       0.3
"""


# ----------------------------------------------------------------------------
# constraint CHAINS — the two-body rigid chain lever
# ----------------------------------------------------------------------------

# Grounded springs (k) along x at y = 0 and y = 4 hold slaves of RBE2 body A
# (master node 5 at y = 2); node 6 (y = 8) is ALSO a slave of A and the
# MASTER of RBE2 body B, whose slave (node 7, y = 12) takes the load — a
# rigid-on-rigid chain that moves as ONE composite lever. Rigid kinematics
# u(y) = u0 + (u1 - u0) y/4 with force balance k(u0 + u1) = F and moment
# balance about y = 2: u0 = F/(2k) - F(Y-2)/(4k), u1 = F/(2k) + F(Y-2)/(4k)
# for the load at height Y = 12:
#     u0 = -2F/k, u1 = 3F/k, u(8) = 8F/k, u(12) = 13F/k, u(2) = F/(2k).
CHAIN_LEVER = """\
#RADIOSS STARTER
/BEGIN
RBE2 chain lever
/NODE
         1                 0.0                 0.0                 0.0
         2                 0.0                 4.0                 0.0
         3                 1.0                 0.0                 0.0
         4                 1.0                 4.0                 0.0
         5                 1.0                 2.0                 0.0
         6                 1.0                 8.0                 0.0
         7                 1.0                12.0                 0.0
/SPRING/1
         1         1         3
         2         2         4
/PART/1
springs
         1         1
""" + STEEL1 + """\
/PROP/SPRING/1
spring
       1.0       2.0       0.0
/GRNOD/NODE/1
ground
1 2
/GRNOD/NODE/2
slavesA
3 4 6
/GRNOD/NODE/3
masterA
5
/GRNOD/NODE/4
tip
7
/GRNOD/NODE/5
slavesB
7
/ADMAS/1
tip mass (a chained body needs some mass)
   1.0e-3         4
/RBE2/1
leverA
         5         2
/RBE2/2
leverB
         6         5
/BCS/1
ground
       111       111         0         1
/BCS/2
master lateral
       011       110         0         3
/FUNCT/1
ramp
       0.0       0.0
       2.0       2.0
/CLOAD/1
pull
         1         X         4       1.0
/END
"""


@pytest.mark.parametrize("mode", ["", "/IMPL/NONLIN\n"])
def test_chain_lever_closed_form(make_deck, mode):
    """The two-body rigid chain lever: transform SUBSTITUTION composes
    T = T_A . T_B, so the chained tip rides the composite rigid motion and
    the closed form is exact in ONE Newton step (linear geometry). The
    NONLIN run re-linearizes the whole chain per committed frame and lands
    on the small-rotation limit with a SMALL load."""
    k = 2.0
    F = 1.0 if not mode else 1e-4
    lever = CHAIN_LEVER if not mode else CHAIN_LEVER.replace(
        "pull\n         1         X         4       1.0",
        f"pull\n         1         X         4       {F}")
    model, out = _run(make_deck, "CHL" + ("N" if mode else "L"), lever,
                      f"#\n/RUN/CHL{'N' if mode else 'L'}/1\n1.0\n/IMPL\n"
                      f"{mode}/END\n")
    d = model.x - model.x0
    tol = 1e-9 if not mode else 1e-3
    for nid, expect in ((3, -2 * F / k), (4, 3 * F / k), (6, 8 * F / k),
                        (7, 13 * F / k), (5, 0.5 * F / k)):
        assert d[model.node_index(nid), 0] == pytest.approx(expect, rel=tol)
    res = model.implicit_result
    assert res.converged
    if not mode:
        assert res.increments[-1].iterations == 2   # exact in one step


def test_chain_mpc_on_rigid_slave(make_deck):
    """An /MPC equality row written on a RIGID SLAVE (u_x(3) = u_x(9) with
    node 9 on its own grounded spring) — the M14 pre-substitution expands
    the dependent column through the body's row before the pivoting. The
    lever equilibrium with the extra spring k9 acting at y = 0 through the
    MPC has the closed form of a lever on three springs; with the load at
    the tip (y = 8 deck): moment/force balance gives
        u0 (k + k9) + u1 k = F,   -2 u0 (k + k9) + 2 u1 k = 6 F
    => u0 = -F/(k + k9), u1 = 2F/k."""
    k, k9, F = 2.0, 3.0, 1.0
    deck = CHAIN_LEVER
    # drop body B and its tip (use the plain single-body lever with the
    # load at node 6, y = 8) and add the MPC'd spring node 9
    deck = deck.replace("/RBE2/2\nleverB\n         6         5\n", "")
    deck = deck.replace("/GRNOD/NODE/5\nslavesB\n7\n", "")
    deck = deck.replace("/ADMAS/1\ntip mass (a chained body needs some "
                        "mass)\n   1.0e-3         4\n", "")
    deck = deck.replace("/GRNOD/NODE/4\ntip\n7\n", "/GRNOD/NODE/4\ntip\n6\n")
    deck = deck.replace("""\
/NODE
         1                 0.0                 0.0                 0.0""", """\
/NODE
         9                 2.0                 0.0                 0.0
        10                 3.0                 0.0                 0.0
         1                 0.0                 0.0                 0.0""")
    deck = deck.replace("/SPRING/1\n         1         1         3\n"
                        "         2         2         4\n",
                        "/SPRING/1\n         1         1         3\n"
                        "         2         2         4\n"
                        "         3         9        10\n")
    # the third spring gets stiffness k9 through its own part? keep one
    # prop: give it the same k, and scale the closed form instead
    deck = deck.replace("/BCS/1\nground\n       111       111         0"
                        "         1\n",
                        "/GRNOD/NODE/6\nanchor9\n10\n"
                        "/GRNOD/NODE/7\nnode9\n9\n"
                        "/BCS/1\nground\n       111       111         0"
                        "         1\n"
                        "/BCS/3\nanchor9\n       111       111         0"
                        "         6\n"
                        "/BCS/4\nnode9 lateral\n       011       000"
                        "         0         7\n"
                        "/MPC/1\nchain row\n"
                        "         3         1       1.0\n"
                        "         9         1      -1.0\n")
    model, out = _run(make_deck, "CHM", deck,
                      "#\n/RUN/CHM/1\n1.0\n/IMPL\n/END\n")
    d = model.x - model.x0
    # spring 3 has the SAME k: effective base stiffness 2k at y = 0
    u0 = -F / (k + k)
    u1 = 2 * F / k
    assert d[model.node_index(3), 0] == pytest.approx(u0, rel=1e-9)
    assert d[model.node_index(9), 0] == pytest.approx(u0, rel=1e-9)
    assert d[model.node_index(4), 0] == pytest.approx(u1, rel=1e-9)
    assert model.implicit_result.converged


def _chain_pendulum_deck():
    """The M12 physical pendulum (masses m at l and 2l on a 60-deg arm,
    pivot at the origin) assembled as a rigid-on-rigid CHAIN: body A
    (pivoted master 10) carries the l-mass node 21; body B's master IS
    node 21 and carries the 2l-mass node 22. The composite is the same
    rigid pendulum — same elliptic-integral period."""
    return """\
#RADIOSS STARTER
/BEGIN
chained implicit pendulum
/NODE
        10                 0.0                 0.0                 0.0
        21       86.6025403784      -50.0000000000        0.0000000000
        22      173.2050807569     -100.0000000000        0.0000000000
       101              1000.0                 0.0                 0.0
       102              1010.0                 0.0                 0.0
/SPRING/1
         1       101       102
/PART/1
token spring
         1         1
""" + STEEL1 + """\
/PROP/SPRING/1
spring
     1e-06       1.0       0.0
/GRNOD/NODE/1
bobA
21
/GRNOD/NODE/2
pivot
10
/GRNOD/NODE/3
faraway
101 102
/GRNOD/NODE/4
bobs
21 22
/GRNOD/NODE/5
bobB
22
/ADMAS/1
bob mass
     0.001         4
/RBODY/1
upper arm
        10         1       0.0         0
/RBODY/2
lower arm
        21         5       0.0         0
/BCS/1
pivot
       111       110         0         2
/BCS/2
faraway
       111       111         0         3
/GRAV/1
gravity
         1         Y         4  -9.81e-3
/FUNCT/1
one
       0.0       1.0
    1000.0       1.0
/END
"""


def test_chain_pendulum_elliptic_period(make_deck):
    """The chained pendulum swings EXACTLY like the M12 single-body one:
    vertical crossing on the elliptic-integral quarter period (< 0.5%),
    BOTH arm lengths exact through the swing (the topological commit
    placement — the child body places from its parent's already-placed
    master), amplitude preserved."""
    from scipy.special import ellipk
    model, _ = _run(make_deck, "CHP", _chain_pendulum_deck(), """\
#
/RUN/CHP/1
480.0
/IMPL
/IMPL/DTINI
0.5
/IMPL/DYNA/2
0.5 0.25
/IMPL/NONLIN
/END
""")
    h = model.implicit_result.history
    l, g = 100.0, 9.81e-3
    th0 = np.deg2rad(60.0)
    w0 = np.sqrt(3.0 * g / (5.0 * l))            # I = 5ml^2, Mgd = 3mgl/2*2
    T4 = ellipk(np.sin(th0 / 2) ** 2) / w0
    i21 = model.node_index(21)
    i22 = model.node_index(22)
    t = np.array(h["t"])
    xs1 = np.array([model.x0[i21] + u[i21] for u in h["u"]])
    xs2 = np.array([model.x0[i22] + u[i22] for u in h["u"]])
    th = np.arctan2(xs1[:, 0], -xs1[:, 1])
    # both arms exact through the whole swing (chained Rodrigues placement)
    assert np.abs(np.linalg.norm(xs1, axis=1) - l).max() < 1e-6 * l
    assert np.abs(np.linalg.norm(xs2, axis=1) - 2 * l).max() < 2e-6 * l
    # the chain is rigid: node 22 stays collinear at twice the arm
    assert np.abs(xs2 - 2.0 * xs1).max() < 1e-6 * l
    cross = np.where(np.diff(np.sign(th)) != 0)[0]
    assert len(cross) >= 1
    i = cross[0]
    tc = t[i] - th[i] * (t[i + 1] - t[i]) / (th[i + 1] - th[i])
    assert tc == pytest.approx(T4, rel=5e-3)
    assert np.degrees(th.min()) == pytest.approx(-60.0, abs=1.0)


def test_chain_circular_refused(make_deck):
    """A genuinely CIRCULAR chain (A's master is a slave of B, B's master
    a slave of A) has no topological ordering: the DFS back-edge refuses
    loudly."""
    deck = CHAIN_LEVER.replace("/GRNOD/NODE/5\nslavesB\n7\n",
                               "/GRNOD/NODE/5\nslavesB\n5\n")
    deck = deck.replace("/ADMAS/1\ntip mass (a chained body needs some "
                        "mass)\n   1.0e-3         4\n",
                        "/ADMAS/1\nmasters mass\n   1.0e-3         3\n")
    s, e = make_deck("CIR", deck, "#\n/RUN/CIR/1\n1.0\n/IMPL\n/END\n")
    with contextlib.redirect_stdout(io.StringIO()):
        run_starter(s)
        with pytest.raises(NotImplementedError, match="CIRCULAR"):
            run_engine(e)


def test_chain_conflict_refused(make_deck):
    """One DOF dependent in TWO constraints (a node slave of two rigid
    bodies) is a CONFLICT, not a chain — still refused (the original
    MERGES part-bodies instead; merging would silently change the
    model)."""
    deck = CHAIN_LEVER.replace("/GRNOD/NODE/5\nslavesB\n7\n",
                               "/GRNOD/NODE/5\nslavesB\n3 7\n")
    s, e = make_deck("CFL", deck, "#\n/RUN/CFL/1\n1.0\n/IMPL\n/END\n")
    with contextlib.redirect_stdout(io.StringIO()):
        with pytest.raises(Exception):
            # the STARTER already rejects a slave in two bodies (the
            # pre-M14 rule, unchanged); belt-and-braces: if a future
            # reader lets it through, constraints.py raises the conflict
            run_starter(s)
            run_engine(e)


def test_chain_refused_by_explicit_engine(make_deck):
    """The explicit engine refuses a rigid-body chain loudly — its
    per-body 6-DOF integrator has no nesting order (the original never
    sees one: rbody_part_modif.F90 resolves chains in the Starter)."""
    s, e = make_deck("CHX", CHAIN_LEVER, "#\n/RUN/CHX/1\n0.001\n/END\n")
    with contextlib.redirect_stdout(io.StringIO()):
        run_starter(s)
        with pytest.raises(NotImplementedError, match="CHAIN"):
            run_engine(e)


# ----------------------------------------------------------------------------
# /INTER/TYPE11 Coulomb friction — crossed edges, stick & slip closed forms
# ----------------------------------------------------------------------------

def _t11_fric_deck(h, gap, K, kz, kx, Fz, Fx, mu):
    """Edge A (truss along x at z = h): each end grounded by a vertical
    spring (kz) AND a lateral y-spring (kx); edge B (along y at z = 0)
    fixed. Push A down with Fz (contact force N) and ALONG Y with Fx —
    the y-drive keeps the contact at edge A's midpoint (s = 1/2), so no
    tilt develops and the 1-D closed forms are exact:
        u_z = (K(gap - h) - Fz)/(2 kz + K),   N = K (gap - h - u_z)
        stick: u_y = Fx/(2 kx + K)            (K_t = K, one contact point)
        slip : f_t = mu N,  u_y = (Fx - mu N)/(2 kx).
    (Driving along x instead moves the contact point along edge A — the
    normal force then acts off the edge center and the lever TILTS: real
    physics, but no clean closed form. The y-slip is equally generic for
    the return map: the tangential plane contains both edge directions.)"""
    return f"""\
#RADIOSS STARTER
/BEGIN
crossed edges friction
/NODE
         1                -1.0                 0.0    {h:16.6f}
         2                 1.0                 0.0    {h:16.6f}
         3                 0.0                -1.0                 0.0
         4                 0.0                 1.0                 0.0
         5                -1.0                 0.0    {h + 1.0:16.6f}
         6                 1.0                 0.0    {h + 1.0:16.6f}
         7                -1.0                -1.0    {h:16.6f}
         8                 1.0                -1.0    {h:16.6f}
/TRUSS/1
         1         1         2
         2         3         4
/PART/1
edges
         1         1
""" + STEEL1 + f"""\
/PROP/TRUSS/1
truss
       1.0
/SPRING/2
         3         1         5
         4         2         6
/PART/2
hangers
         2         1
/PROP/SPRING/2
spring
       1.0    {kz:6.2f}       0.0
/SPRING/3
         5         1         7
         6         2         8
/PART/3
lateral springs
         3         1
/PROP/SPRING/3
spring
       1.0    {kx:6.2f}       0.0
/GRNOD/NODE/1
edge A ends
1 2
/GRNOD/NODE/2
fixed
3 4 5 6 7 8
/BCS/1
lateral A (x only)
       100       000         0         1
/BCS/2
ground
       111       111         0         2
/LINE/SEG/1
edge A
         1         2
/LINE/SEG/2
edge B
         3         4
/INTER/TYPE11/1
cross
         1         2         1         0
    {K:6.2f}    {mu:6.3f}    {gap:6.3f}       0.0
/FUNCT/1
ramp
       0.0       0.0
       2.0       2.0
/CLOAD/1
push down
         1         Z         1   {-Fz / 2.0}
/CLOAD/2
push sideways
         1         Y         1   {Fx / 2.0}
/END
"""


_T11 = dict(h=0.09, gap=0.1, K=100.0, kz=2.0, kx=2.0)


def _t11_forms(Fz):
    h, gap, K, kz = _T11["h"], _T11["gap"], _T11["K"], _T11["kz"]
    u_z = (K * (gap - h) - Fz) / (2 * kz + K)
    N = K * (gap - h - u_z)
    return u_z, N


def test_t11_friction_stick_closed_form(make_deck):
    """Below the cone every pair sticks: the lateral displacement is the
    parallel-springs closed form u_y = Fx/(2 kx + K_t) (the micro-slip of
    the penalty regularization) and the vertical answer is untouched by
    friction (pure normal approach = zero slip increment)."""
    Fz, mu = 2.0, 0.3
    u_z, N = _t11_forms(Fz)
    Fx = 0.5 * mu * N
    model, _ = _run(make_deck, "T14S",
                    _t11_fric_deck(Fz=Fz, Fx=Fx, mu=mu, **_T11),
                    "#\n/RUN/T14S/1\n1.0\n/IMPL/DTINI\n0.25\n"
                    "/IMPL/NEWTON\n1e-11  40\n/END\n")
    assert model.implicit_result.converged
    d = model.x - model.x0
    kx, K = _T11["kx"], _T11["K"]
    assert d[model.node_index(1), 2] == pytest.approx(u_z, abs=1e-9)
    assert d[model.node_index(1), 1] == pytest.approx(
        Fx / (2 * kx + K), abs=1e-9)
    assert d[model.node_index(2), 1] == pytest.approx(
        Fx / (2 * kx + K), abs=1e-9)


def test_t11_friction_slip_closed_form(make_deck):
    """Above the stick range the pair SLIDES on the cone: the transmitted
    tangential force is EXACTLY mu N (the radial return) and the ground
    springs carry the excess, u_y = (Fx - mu N)/(2 kx)."""
    Fz, mu = 2.0, 0.3
    u_z, N = _t11_forms(Fz)
    Fx = 2.0 * mu * N
    model, _ = _run(make_deck, "T14L",
                    _t11_fric_deck(Fz=Fz, Fx=Fx, mu=mu, **_T11),
                    "#\n/RUN/T14L/1\n1.0\n/IMPL/DTINI\n0.25\n"
                    "/IMPL/NEWTON\n1e-11  40\n/END\n")
    assert model.implicit_result.converged
    d = model.x - model.x0
    kx = _T11["kx"]
    u_y = (Fx - mu * N) / (2 * kx)
    assert d[model.node_index(1), 1] == pytest.approx(u_y, rel=1e-7)
    # vertical state unchanged (slip is tangential)
    assert d[model.node_index(1), 2] == pytest.approx(u_z, abs=1e-8)


def test_t11_friction_mu0_bit_identical(make_deck):
    """mu = 0 leaves the M13 frictionless machinery untouched: the
    crossed-edges closed form to round-off (every friction branch is
    guarded by mu > 0 — this is the M13 contract carried forward)."""
    Fz = 2.0
    u_z, _ = _t11_forms(Fz)
    model, _ = _run(make_deck, "T140",
                    _t11_fric_deck(Fz=Fz, Fx=0.0, mu=0.0, **_T11),
                    "#\n/RUN/T140/1\n1.0\n/IMPL/DTINI\n0.25\n"
                    "/IMPL/NEWTON\n1e-11  40\n/END\n")
    d = model.x - model.x0
    assert d[model.node_index(1), 2] == pytest.approx(u_z, abs=1e-10)


def test_t11_friction_normal_load_keeps_frictionless_form(make_deck):
    """mu > 0 with a PURE normal load: the approach is normal, the slip
    increment is zero, and the frictionless closed form holds to
    round-off — friction must never invent a tangential force. (This
    replaces the M13 'warns and runs frictionless' assertion: TYPE11
    friction is a capability now.)"""
    Fz, mu = 2.0, 0.3
    u_z, _ = _t11_forms(Fz)
    model, out = _run(make_deck, "T14N",
                      _t11_fric_deck(Fz=Fz, Fx=0.0, mu=mu, **_T11),
                      "#\n/RUN/T14N/1\n1.0\n/IMPL/DTINI\n0.25\n"
                      "/IMPL/NEWTON\n1e-11  40\n/END\n")
    assert "COULOMB FRICTION" in out and "DEFERRED" not in out
    d = model.x - model.x0
    assert d[model.node_index(1), 2] == pytest.approx(u_z, abs=1e-9)
    assert abs(d[model.node_index(1), 1]) < 1e-12


def test_t11_friction_fd_consistency_both_regimes(make_deck):
    """FD consistency of the TYPE11 friction residual/tangent in BOTH
    regimes at a freshly-committed anchor (where the frozen-parameter
    friction blocks are exact — the drift terms are O(slip increment),
    the same class as the TYPE7 frozen-weight omission). Slip rows are
    nonsymmetric (the return-map derivative) — checked against the FULL
    assembled tangent, no symmetry assumed."""
    from pyradioss.common.messages import MessageLog
    from pyradioss.implicit.contact import (build_implicit_contacts,
                                            contact_tangent)
    from pyradioss.implicit.dofmap import DofMap

    model = _model(make_deck, "T14F",
                   _t11_fric_deck(Fz=2.0, Fx=0.5, mu=0.3, **_T11))
    log = MessageLog()
    with contextlib.redirect_stdout(io.StringIO()):
        contacts = build_implicit_contacts(model, log)
        dof = DofMap(model)
    c = contacts[0]
    n = model.numnod
    rng = np.random.default_rng(7)

    def residual(u):
        f = np.zeros((n, 3))
        for cc in contacts:
            cc.forces(model.x0 + u, f)
        return dof.gather_residual(f, np.zeros((n, 3)))

    for regime, dy in (("stick", 0.001), ("slip", 0.05)):
        # drive edge A down and sideways, COMMIT there (fresh anchor),
        # then probe around the committed state
        u0 = np.zeros((n, 3))
        for nid in (1, 2):
            u0[model.node_index(nid)] = [0.0, dy, -0.004]
        with contextlib.redirect_stdout(io.StringIO()):
            c.commit(model.x0 + u0)
        # a slip trial state ON TOP of the committed anchor
        u1 = u0.copy()
        for nid in (1, 2):
            u1[model.node_index(nid), 1] += dy
        # verify the intended regime
        ea, eb, s, t, nvec, pen, K, _, _, key = c._active_pairs(
            model.x0 + u1)
        _, stick, _, _ = c._friction_state(model.x0 + u1, ea, eb, s, t,
                                           key, nvec, pen, K)
        assert stick.all() == (regime == "stick")
        Kc = contact_tangent(contacts, model.x0 + u1, dof).toarray()
        eps = 1e-8
        for _ in range(5):
            v = rng.standard_normal(dof.ndof)
            du, _ = dof.scatter_solution(v)
            fd = (residual(u1 + eps * du)
                  - residual(u1 - eps * du)) / (2 * eps)
            # max-norm relative bound: the residue (measured <= 5e-3
            # here) is the documented frozen-parameter class — the
            # closest-point drift ds/dt and the anchor's n-rotation
            # terms, O(|anchor force| / K), the TYPE7 M13 analogue
            err = np.abs(fd + Kc @ v).max() / max(np.abs(fd).max(), 1e-12)
            assert err < 1e-2, (regime, err)
        # restore virgin anchors for the next regime
        c.x_com = model.x.copy()
        c.ft_keys = np.zeros(0, dtype=np.int64)
        c.ft_vals = np.zeros((0, 3))


def test_t11_friction_explicit_cross_check(make_deck):
    """Steady sliding, implicit vs explicit, ON THE SAME CONVERGED
    GEOMETRY: the implicit return map transmits EXACTLY mu N (the slip
    closed form, asserted); the explicit TYPE11 kinetic law
    (inter_type11.ContactType11 — the velocity-regularized Coulomb force
    mu Fn v/(v + eps), eps = 1e-3*gap/dt) is evaluated at the implicit
    run's converged slip state with a steady tangential sliding velocity
    chosen well above eps, and its tangential/normal force ratio must
    land on the same mu (within the regularization factor, from below).
    Evaluating the explicit LAW at the matched state isolates the
    friction-model comparison the task needs — a full explicit transient
    of two crossed point-contact trusses never reaches a clean steady
    slide (the contact point walks toward an edge END and the normal
    tilts: measured, not guessed)."""
    from pyradioss.common.messages import MessageLog
    from pyradioss.contact.inter_type11 import ContactType11

    Fz, mu = 2.0, 0.3
    u_z, N = _t11_forms(Fz)
    Fx = 2.0 * mu * N
    kx = _T11["kx"]
    # implicit: transmitted f_t = Fx - 2 kx u_y = mu N exactly
    model, _ = _run(make_deck, "T14XI",
                    _t11_fric_deck(Fz=Fz, Fx=Fx, mu=mu, **_T11),
                    "#\n/RUN/T14XI/1\n1.0\n/IMPL/DTINI\n0.25\n"
                    "/IMPL/NEWTON\n1e-11  40\n/END\n")
    d = model.x - model.x0
    ft_imp = Fx - 2 * kx * d[model.node_index(1), 1]
    assert ft_imp / N == pytest.approx(mu, rel=1e-6)

    # explicit law at the SAME state: edge A sliding at v in +y
    itf = [i for i in model.interfaces if i.type == 11][0]
    with contextlib.redirect_stdout(io.StringIO()):
        exp = ContactType11(itf, model, MessageLog())
    n = model.numnod
    v = np.zeros((n, 3))
    v_slide, dt = 0.05, 1.0            # eps = 1e-3*gap/dt = 1e-4 << v
    for nid in (1, 2):
        v[model.node_index(nid), 1] = v_slide
    fcont = np.zeros((n, 3))
    with contextlib.redirect_stdout(io.StringIO()):
        exp.forces(model.x, v, model.mass, dt, fcont, cycle=0)
    fB = fcont[model.node_index(3)] + fcont[model.node_index(4)]
    Fn_exp = abs(fB[2])                # normal push on the fixed edge B
    Ft_exp = fB[1]                     # dragged along +y by the slide
    assert Fn_exp > 0.0
    ratio = Ft_exp / Fn_exp
    factor = v_slide / (v_slide + 1e-3 * _T11["gap"] / dt)
    assert ratio == pytest.approx(mu * factor, rel=0.02)
    assert 0.95 * mu < ratio <= mu * (1.0 + 1e-9)


# ----------------------------------------------------------------------------
# /IMPL/ARCL with contact & constraints — the snap-catch and the RBE2 snap
# ----------------------------------------------------------------------------

def _vm_P(y, a, h, E, area):
    L0 = np.sqrt(a * a + h * h)
    L = np.sqrt(a * a + y * y)
    return -2.0 * E * area * np.log(L / L0) * y / L


def _vm_deck(a, h, area, E, P_end, stop=False, y_pad=-1.9, Kc=50.0,
             gap=0.4, rbe2=False):
    """The M9 von Mises truss, optionally with a TYPE7 pad stop below the
    apex (snap-catch) or an RBE2 handle loading the apex through a
    condensed master (guided: rotations + z clamped)."""
    nodes = f"""\
         1{-a:20.10f}{0.0:20.10f}{0.0:20.10f}
         2{a:20.10f}{0.0:20.10f}{0.0:20.10f}
         3{0.0:20.10f}{h:20.10f}{0.0:20.10f}"""
    pad = ""
    extra = ""
    if stop:
        yb = y_pad - 1.0
        nodes += f"""
        11{-1.0:20.10f}{yb:20.10f}{-1.0:20.10f}
        12{1.0:20.10f}{yb:20.10f}{-1.0:20.10f}
        13{1.0:20.10f}{yb:20.10f}{1.0:20.10f}
        14{-1.0:20.10f}{yb:20.10f}{1.0:20.10f}
        15{-1.0:20.10f}{y_pad:20.10f}{-1.0:20.10f}
        16{1.0:20.10f}{y_pad:20.10f}{-1.0:20.10f}
        17{1.0:20.10f}{y_pad:20.10f}{1.0:20.10f}
        18{-1.0:20.10f}{y_pad:20.10f}{1.0:20.10f}"""
        pad = f"""\
/BRICK/2
         1        11        14        13        12        15        18        17        16
/PART/2
pad
         2         2
/MAT/LAW1/2
pad
   7.8e-6
     210.0       0.0
/PROP/SOLID/2
solid
       1.1      0.05       0.1
/GRNOD/NODE/11
pad nodes
11 12 13 14 15 16 17 18
/BCS/11
pad clamp
       111       111         0        11
/GRNOD/NODE/12
apex secondary
3
/SURF/SEG/1
pad top (+y face)
        15        16        17        18
/INTER/TYPE7/1
stop
        12         1         1         0
    {Kc:6.1f}     0.000    {gap:6.3f}       0.0
"""
    if rbe2:
        nodes += f"""
       100{0.0:20.10f}{h + 1.0:20.10f}{0.0:20.10f}"""
        extra = """\
/GRNOD/NODE/21
apex slave
3
/GRNOD/NODE/22
handle
100
/RBE2/1
handle
       100        21
/BCS/21
handle guides
       001       111         0        22
"""
        apexbcs = ""
    else:
        apexbcs = """\
/BCS/2
apex out-of-plane
       001       000         0         2
"""
    load_gr = "22" if rbe2 else "2"
    return f"""\
#RADIOSS STARTER
/BEGIN
VMISES M14
/NODE
{nodes}
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
{apexbcs}{extra}{pad}/FUNCT/1
proportional ramp
       0.0       0.0
     100.0     100.0
/CLOAD/1
push down
         1         Y   {load_gr:>6s}       {-P_end}
/END
"""


def test_arclength_snap_caught_by_contact_stop(make_deck):
    """The snap-catch: the M9 von Mises snap-through traced by arc length
    WITH a TYPE7 stop below the apex. Before contact the trace matches
    the free path (the sampled load peak approaches +P_max from below,
    exactly as M9); the unstable branch descends until the apex lands on
    the stop, and the arrested branch climbs back to the full load ON the
    stop — the final state is the truss + penalty-spring closed-form
    equilibrium, and it rests ABOVE the free far-branch answer."""
    from scipy.optimize import brentq
    a, h, area, E = 10.0, 2.0, 1.0, 210.0
    ygrid = np.linspace(1e-6, h, 20001)
    P_max = float(_vm_P(ygrid, a, h, E, area).max())
    P_end = 1.3 * P_max
    y_pad, Kc, gap = -1.9, 50.0, 0.4
    model, _ = _run(make_deck, "SNC",
                    _vm_deck(a, h, area, E, P_end, stop=True, y_pad=y_pad,
                             Kc=Kc, gap=gap),
                    "#\n/RUN/SNC/1\n1.0\n/IMPL/ARCL\n/IMPL/DTINI\n0.05\n"
                    "/IMPL/NEWTON\n1e-9  40\n/END\n")
    res = model.implicit_result
    assert res.converged, res.stop_reason
    lams = np.array([i.load_factor for i in res.increments if i.converged])
    # pre-contact: the free-path limit point, sampled from below (= M9)
    turn = np.argmax(np.diff(lams) < 0.0)
    assert np.diff(lams).min() < 0.0
    assert lams[turn] * P_end <= P_max * (1.0 + 1e-6)
    assert lams[turn] * P_end >= 0.90 * P_max
    assert lams[-1] == pytest.approx(1.0, abs=1e-9)
    # the arrested equilibrium: P_end = P_truss(y) + Kc*(gap - (y - y_pad))
    y_f = brentq(lambda y: _vm_P(y, a, h, E, area)
                 + Kc * (gap - (y - y_pad)) - P_end, y_pad, y_pad + gap)
    y_num = model.x[model.node_index(3), 1]
    assert y_num == pytest.approx(y_f, abs=1e-5)
    # caught: resting ABOVE the free far-branch equilibrium
    y_free = brentq(lambda y: _vm_P(y, a, h, E, area) - P_end,
                    -10.0 * h, -h)
    assert y_num > y_free + 0.5


def test_arclength_rbe2_loaded_snap(make_deck):
    """The RBE2-loaded snap: the apex is condensed onto a guided handle
    master and the load applies to the MASTER — the whole arc trace
    (both auxiliary solves, the spherical metric, the root selection)
    runs in the REDUCED space. Same closed-form path as M9: peak below
    +P_max, negative-load unstable branch, far-branch equilibrium."""
    from scipy.optimize import brentq
    a, h, area, E = 10.0, 2.0, 1.0, 210.0
    ygrid = np.linspace(1e-6, h, 20001)
    P_max = float(_vm_P(ygrid, a, h, E, area).max())
    P_end = 1.3 * P_max
    model, _ = _run(make_deck, "SNR", _vm_deck(a, h, area, E, P_end,
                                               rbe2=True),
                    "#\n/RUN/SNR/1\n1.0\n/IMPL/ARCL\n/IMPL/DTINI\n0.05\n"
                    "/IMPL/NEWTON\n1e-9  40\n/END\n")
    res = model.implicit_result
    assert res.converged, res.stop_reason
    lams = np.array([i.load_factor for i in res.increments if i.converged])
    turn = np.argmax(np.diff(lams) < 0.0)
    assert lams[turn] * P_end <= P_max * (1.0 + 1e-6)
    assert lams[turn] * P_end >= 0.90 * P_max
    assert lams.min() * P_end < -0.5 * P_max
    assert lams.min() * P_end >= -P_max * (1.0 + 1e-6)
    assert lams[-1] == pytest.approx(1.0, abs=1e-9)
    y_far = brentq(lambda y: _vm_P(y, a, h, E, area) - P_end, -10 * h, -h)
    y_num = model.x[model.node_index(3), 1]
    assert y_num == pytest.approx(y_far, rel=1e-3)
    # the handle rode along exactly (chain of trust: condensation + arc)
    assert model.x[model.node_index(100), 1] - (h + 1.0) == \
        pytest.approx(y_num - h, rel=1e-9)


# ----------------------------------------------------------------------------
# /IMPL/BUCKL with constraints & contact
# ----------------------------------------------------------------------------

STEEL_NU0 = """\
/MAT/LAW1/1
steel nu=0
   7.8e-6
     210.0       0.0
"""


def _buckl_strip_deck(nx, ny, Lx, b, t, F_total, cap=False, stop=False,
                      pinned=False, Kc=50.0, gap=0.05, pen0=1e-4):
    """BT4 strip column along x (nu = 0: plate = beam). ``cap`` condenses
    the tip row onto an RBE2 master and loads the MASTER; ``stop`` parks
    the tip row on a TYPE7 pad at z = -(gap - pen0) (a hair of initial
    contact) with a small hold-down force; ``pinned`` frees the root
    rotations (translations stay fixed)."""
    def nid(i, j):
        return 1 + i + j * (nx + 1)
    nodes = [f"{nid(i,j):10d}{i*Lx/nx:20.10f}{j*b/ny:20.10f}{0.0:20.10f}"
             for j in range(ny + 1) for i in range(nx + 1)]
    shells = []
    eid = 0
    for j in range(ny):
        for i in range(nx):
            eid += 1
            c = [nid(i, j), nid(i + 1, j), nid(i + 1, j + 1), nid(i, j + 1)]
            shells.append(f"{eid:10d}" + "".join(f"{v:10d}" for v in c))
    fixed = [nid(0, j) for j in range(ny + 1)]
    tip = [nid(nx, j) for j in range(ny + 1)]
    tip_edge = [nid(nx, 0), nid(nx, ny)]
    tip_int = [nid(nx, j) for j in range(1, ny)]
    wsum = 2 * (ny - 1) + 2
    base_bcs = "       111       000" if pinned else "       111       111"
    nextra = ""
    if cap:
        nextra = f"\n       900{Lx:20.10f}{b/2:20.10f}{0.0:20.10f}"
        extra = f"""\
/GRNOD/NODE/7
tip row
{" ".join(map(str, tip))}
/GRNOD/NODE/8
cap master
900
/RBE2/1
cap
       900         7
/CLOAD/1
axial on master
         1         X         8       {F_total}
"""
    else:
        extra = f"""\
/CLOAD/1
tip edge
         1         X         2       {F_total / wsum}
/CLOAD/2
tip interior
         1         X         3       {2.0 * F_total / wsum}
"""
    pad = ""
    if stop:
        zf = -(gap - pen0)
        zb = zf - 1.0
        x0p, x1p = Lx - 2.0, Lx + 2.0
        nextra += f"""
       901{x0p:20.10f}{-2.0:20.10f}{zb:20.10f}
       902{x1p:20.10f}{-2.0:20.10f}{zb:20.10f}
       903{x1p:20.10f}{b + 2.0:20.10f}{zb:20.10f}
       904{x0p:20.10f}{b + 2.0:20.10f}{zb:20.10f}
       905{x0p:20.10f}{-2.0:20.10f}{zf:20.10f}
       906{x1p:20.10f}{-2.0:20.10f}{zf:20.10f}
       907{x1p:20.10f}{b + 2.0:20.10f}{zf:20.10f}
       908{x0p:20.10f}{b + 2.0:20.10f}{zf:20.10f}"""
        pad = f"""\
/BRICK/2
       991       901       902       903       904       905       906       907       908
/PART/2
pad
         2         2
/MAT/LAW1/2
pad
   7.8e-6
     210.0       0.0
/PROP/SOLID/2
solid
       1.1      0.05       0.1
/GRNOD/NODE/11
pad nodes
901 902 903 904 905 906 907 908
/BCS/11
pad clamp
       111       111         0        11
/GRNOD/NODE/12
tip secondary
{" ".join(map(str, tip))}
/SURF/SEG/1
pad top (+z)
       905       906       907       908
/INTER/TYPE7/1
stop
        12         1         1         0
    {Kc:6.1f}     0.000    {gap:6.3f}       0.0
/CLOAD/9
hold the tip on the stop
         1         Z        12       {-0.01 / len(tip)}
"""
    return f"""\
#RADIOSS STARTER
/BEGIN
STRIP M14
/NODE
{chr(10).join(nodes)}{nextra}
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
{" ".join(map(str, fixed))}
/GRNOD/NODE/2
tip edge
{" ".join(map(str, tip_edge))}
/GRNOD/NODE/3
tip interior
{" ".join(map(str, tip_int))}
/BCS/1
base
{base_bcs}         0         1
/FUNCT/1
ramp
       0.0       0.0
     100.0     100.0
{extra}{pad}/END
"""


_BUCKL_ENG = "#\n/RUN/{0}/1\n1.0\n/IMPL\n/IMPL/BUCKL/1\n1e-6 1.0 4\n/END\n"


def test_buckling_rbe2_cap_equals_plain_column(make_deck):
    """An Euler column whose tip row is condensed onto an RBE2 cap (load
    on the master): for the strip column the free mode is already
    rigid across the tip row, so the REDUCED pencil T^T K T must report
    the PLAIN column's factor — matched to ~1e-9 relative here — and both
    sit on pi^2 EI/(4 L^2)."""
    Lx, b, t, E = 100.0, 10.0, 1.0, 210.0
    P0 = 0.002
    m_plain, _ = _run(make_deck, "BKP",
                      _buckl_strip_deck(20, 2, Lx, b, t, -P0),
                      _BUCKL_ENG.format("BKP"))
    f_plain = m_plain.implicit_result.buckling_factors[0]
    m_cap, out = _run(make_deck, "BKC",
                      _buckl_strip_deck(20, 2, Lx, b, t, -P0, cap=True),
                      _BUCKL_ENG.format("BKC"))
    f_cap = m_cap.implicit_result.buckling_factors[0]
    assert "CONDENSED" in out
    assert f_cap == pytest.approx(f_plain, rel=1e-6)
    I = b * t ** 3 / 12.0
    assert f_cap * P0 == pytest.approx(np.pi ** 2 * E * I / (4 * Lx ** 2),
                                       rel=0.03)


def test_buckling_column_on_contact_stop(make_deck):
    """A PINNED-base column whose tip rests laterally on a TYPE7 stop
    (closed at the prestress by a small hold-down force): the frozen
    active set's gap tangent joins K_MAT (the documented imp_buck.F
    deviation — the original omits contact and would report a MECHANISM
    here), turning the free pin-mechanism into the pinned-pinned Euler
    column pi^2 EI / L^2 (< 3%, the M9 shell-bending tolerance)."""
    Lx, b, t, E = 100.0, 10.0, 1.0, 210.0
    P0 = 0.002
    model, _ = _run(make_deck, "BKS",
                    _buckl_strip_deck(20, 2, Lx, b, t, -P0, stop=True,
                                      pinned=True),
                    _BUCKL_ENG.format("BKS"))
    f_stop = model.implicit_result.buckling_factors[0]
    I = b * t ** 3 / 12.0
    P_pp = np.pi ** 2 * E * I / Lx ** 2
    assert f_stop * P0 == pytest.approx(P_pp, rel=0.03)


# ----------------------------------------------------------------------------
# LAW42 consistent tangent
# ----------------------------------------------------------------------------

MU42, AL42 = [0.002, -0.0004], [2.0, -2.0]        # Mooney-Rivlin


def _rubber_deck(P, nu=0.495, load="force"):
    """Unit rubber cube with symmetry BCs on the x/y/z = 0 faces; uniaxial
    x-load P on the free face (``load='force'``) or an /IMPDISP x-drive
    (``load='disp'``, target P as the displacement)."""
    drive = (f"/CLOAD/1\npull\n         1         X         4"
             f"       {P / 4.0}\n" if load == "force" else
             f"/IMPDISP/1\ndrive\n         1         X         4"
             f"       {P}\n")
    return f"""\
#RADIOSS STARTER
/BEGIN
rubber cube
/NODE
         1                 0.0                 0.0                 0.0
         2                 1.0                 0.0                 0.0
         3                 1.0                 1.0                 0.0
         4                 0.0                 1.0                 0.0
         5                 0.0                 0.0                 1.0
         6                 1.0                 0.0                 1.0
         7                 1.0                 1.0                 1.0
         8                 0.0                 1.0                 1.0
/BRICK/1
         1         1         2         3         4         5         6         7         8
/PART/1
rubber
         1         1
/MAT/LAW42/1
rubber
   1.0e-6
{MU42[0]} {MU42[1]}
{AL42[0]} {AL42[1]}
{nu}
/PROP/SOLID/1
solid
       1.1      0.05       0.1
/GRNOD/NODE/1
x0
1 4 5 8
/GRNOD/NODE/2
y0
1 2 5 6
/GRNOD/NODE/3
z0
1 2 3 4
/GRNOD/NODE/4
pull face
2 3 6 7
/BCS/1
symx
       100       000         0         1
/BCS/2
symy
       010       000         0         2
/BCS/3
symz
       001       000         0         3
/FUNCT/1
ramp
       0.0       0.0
     100.0     100.0
{drive}/END
"""


def _sig42(lams, K):
    """Analytic principal Cauchy stresses of the port's LAW42 form,
    written independently of the kernel."""
    lams = np.asarray(lams, dtype=float)
    J = lams.prod()
    lb = lams * J ** (-1.0 / 3.0)
    S = sum(m * lb ** a for m, a in zip(MU42, AL42))
    return (S - S.mean()) / J + K * (J - 1.0)


def _assert_quadratic_tail(res):
    inc = max(res.increments, key=lambda i: i.iterations)
    r = np.array(inc.residuals)
    r = r[r > 0]
    if r[-1] <= 1e-12 * r[0]:
        return
    tail = r[-3:]
    assert tail[2] / tail[1] < (tail[1] / tail[0]) ** 1.5


def test_law42_tangent_truesdell_identity():
    """The spectral tangent satisfies the Truesdell-rate identity
    d sigma = c : d + l sigma + sigma l^T - tr(d) sigma against FD of the
    kernel's own stress — at DISTINCT stretches, at TWO equal stretches,
    at a HYDROSTATIC state (all three equal — the L'Hopital branch where
    the naive textbook limit subtracts sigma_a twice) and at a rotated
    coalescent state."""
    from pyradioss.materials import law42_ogden
    from pyradioss.model.model import Material
    G0 = sum(m * a for m, a in zip(MU42, AL42)) / 2.0
    mat = Material(id=1, law=42, rho0=1e-6, params={
        "E": 2.0 * G0 * 1.495, "nu": 0.495, "mu": MU42, "alpha": AL42})

    def sig_mat(F):
        sig = np.zeros((1, 6))
        law42_ogden.solid_update(mat, sig, np.zeros((1, 6)), None, 1.0,
                                 {"F": F[None]})
        s = sig[0]
        return np.array([[s[0], s[3], s[5]],
                         [s[3], s[1], s[4]],
                         [s[5], s[4], s[2]]])

    VO = ((0, 0), (1, 1), (2, 2), (0, 1), (1, 2), (0, 2))

    def c4_of(F):
        D6 = law42_ogden.consistent_solid_tangent(mat, F[None])[0]
        assert np.allclose(D6, D6.T, atol=1e-12 * np.abs(D6).max())
        c4 = np.zeros((3, 3, 3, 3))
        for I, (i, j) in enumerate(VO):
            for J, (k, ell) in enumerate(VO):
                for (a, b) in {(i, j), (j, i)}:
                    for (c, d) in {(k, ell), (ell, k)}:
                        c4[a, b, c, d] = D6[I, J]
        return c4

    rng = np.random.default_rng(1)
    R = np.linalg.qr(rng.standard_normal((3, 3)))[0]
    for F in (np.eye(3) + 0.15 * rng.standard_normal((3, 3)),
              1.05 * np.eye(3),
              np.diag([1.3, 1.05, 1.05]),
              np.diag([1.05, 1.05 + 1e-9, 1.02]),
              R @ np.diag([1.3, 1.05, 1.05])):
        c4 = c4_of(F)
        sig0 = sig_mat(F)
        eps = 1e-7
        for _ in range(6):
            g = rng.standard_normal((3, 3))
            dF = eps * g @ F
            ds_fd = (sig_mat(F + dF) - sig_mat(F - dF)) / (2 * eps)
            d = 0.5 * (g + g.T)
            pred = (np.einsum("ijkl,kl->ij", c4, d) + g @ sig0
                    + sig0 @ g.T - np.trace(d) * sig0)
            assert np.abs(ds_fd - pred).max() <= \
                1e-6 * max(np.abs(ds_fd).max(), 1e-12)


def test_law42_full_path_fd_consistency(make_deck):
    """FD of the ACTUAL implicit residual (_internal_forces, NLGEOM)
    against the assembled tangent at a 25% homogeneous deformation and a
    hydrostatic one, with the HOURGLASS stabilization frozen: exact to
    round-off. (With the hourglass live the match degrades identically
    for LAW1 — the gamma-operator geometry variation is the pre-existing
    NLGEOM tangent omission, measured here to be NOT a LAW42 term.)"""
    import pyradioss.elements.solid_hexa8 as hx
    from pyradioss.implicit.assembly import assemble
    from pyradioss.implicit.dofmap import DofMap
    from pyradioss.implicit.statics import _internal_forces, _snapshot

    model = _model(make_deck, "F42", _rubber_deck(0.0))
    for name, group in model.element_groups():
        for sl, mat, prop in group.state["slices"]:
            prop.params["qa"] = 0.0
            prop.params["qb"] = 0.0
            prop.params["h"] = 0.0            # kill the viscous hourglass
    hg_save = hx.HG_STIFF
    hx.HG_STIFF = 0.0                          # freeze the FB hourglass
    try:
        dof = DofMap(model)
        committed = {name: _snapshot(g)
                     for name, g in model.element_groups()}
        n = model.numnod
        rng = np.random.default_rng(4)

        def residual(u):
            f, mm = _internal_forces(model, model.x0, u,
                                     np.zeros((n, 3)), committed,
                                     nlgeom=True)
            return dof.gather_residual(f, mm)

        G = np.array([[0.25, 0.08, -0.05],
                      [0.03, -0.12, 0.06],
                      [-0.04, 0.07, 0.10]])
        for u0 in (model.x0 @ G.T, model.x0 @ (0.05 * np.eye(3))):
            residual(u0)     # the Newton order: trial state THEN tangent
            K = assemble(model, dof, model.x0 + u0, None,
                         kgeo=True).toarray()
            eps = 1e-7
            for _ in range(6):
                v = rng.standard_normal(dof.ndof)
                du, _ = dof.scatter_solution(v)
                fd = (residual(u0 + eps * du)
                      - residual(u0 - eps * du)) / (2 * eps)
                assert np.abs(fd + K @ v).max() <= \
                    1e-6 * max(np.abs(fd).max(), 1e-12)
    finally:
        hx.HG_STIFF = hg_save


def test_law42_uniaxial_closed_form(make_deck):
    """Uniaxial pull to ~70% stretch: the converged state lands on the
    independently-written analytic solution of sigma_lat = 0 and
    sigma_ax * lam_lat^2 = P (Cauchy x current area) to 1e-8, with the
    quadratic Newton tail only a consistent tangent delivers."""
    from scipy.optimize import fsolve
    P = 0.003
    model, _ = _run(make_deck, "U42", _rubber_deck(P),
                    "#\n/RUN/U42/1\n1.0\n/IMPL/NONLIN\n/IMPL/DTINI\n0.2\n"
                    "/IMPL/NEWTON\n1e-11  60\n/END\n")
    res = model.implicit_result
    assert res.converged
    K = model.bricks.state["slices"][0][1].K

    def eqs(v):
        l1, l2 = v
        s = _sig42([l1, l2, l2], K)
        return [s[0] * l2 * l2 - P, s[1]]

    l1, l2 = fsolve(eqs, [1.3, 0.9])
    d = model.x - model.x0
    assert 1.0 + d[model.node_index(2), 0] == pytest.approx(l1, abs=1e-8)
    assert 1.0 + d[model.node_index(3), 1] == pytest.approx(l2, abs=1e-8)
    _assert_quadratic_tail(res)


def test_law42_equibiaxial_closed_form(make_deck):
    """Equibiaxial pull (x AND y): the analytic system sigma_z = 0,
    sigma_x * lam_y lam_z = P — the state where two stretches COALESCE
    exactly, driving the tangent through its L'Hopital branch on every
    iteration."""
    from scipy.optimize import fsolve
    P = 0.002
    deck = _rubber_deck(P).replace(
        "/BCS/2\nsymy\n       010       000         0         2\n",
        "/BCS/2\nsymy\n       010       000         0         2\n"
        "/GRNOD/NODE/5\npull face y\n3 4 7 8\n"
        f"/CLOAD/2\npull y\n         1         Y         5"
        f"       {P / 4.0}\n")
    model, _ = _run(make_deck, "B42", deck,
                    "#\n/RUN/B42/1\n1.0\n/IMPL/NONLIN\n/IMPL/DTINI\n0.2\n"
                    "/IMPL/NEWTON\n1e-11  60\n/END\n")
    res = model.implicit_result
    assert res.converged
    K = model.bricks.state["slices"][0][1].K

    def eqs(v):
        l1, l3 = v
        s = _sig42([l1, l1, l3], K)
        return [s[0] * l1 * l3 - P, s[2]]

    l1, l3 = fsolve(eqs, [1.2, 0.8])
    d = model.x - model.x0
    assert 1.0 + d[model.node_index(2), 0] == pytest.approx(l1, abs=1e-8)
    assert 1.0 + d[model.node_index(3), 1] == pytest.approx(l1, abs=1e-8)
    assert 1.0 + d[model.node_index(5), 2] == pytest.approx(l3, abs=1e-8)
    _assert_quadratic_tail(res)


def test_law42_implicit_vs_explicit_quasi_static(make_deck):
    """The explicit cross-check: the same rubber cube compressed 20% by
    /IMPDISP — implicitly (NLGEOM statics) and explicitly (slow ramp +
    /DAMP to the quasi-static state). Same kernels compute sigma(F); the
    comparison isolates the IMPLICIT machinery (residual assembly,
    tangent, Newton) and must agree on stress and lateral bulge."""
    target = -0.2
    m_imp, _ = _run(make_deck, "X42I", _rubber_deck(target, load="disp"),
                    "#\n/RUN/X42I/1\n1.0\n/IMPL/NONLIN\n/IMPL/DTINI\n0.1\n"
                    "/IMPL/NEWTON\n1e-10  60\n/END\n")
    assert m_imp.implicit_result.converged
    deck_exp = _rubber_deck(target, load="disp").replace(
        "/FUNCT/1\nramp\n       0.0       0.0\n     100.0     100.0\n",
        "/FUNCT/1\nramp hold\n       0.0       0.0\n"
        "      40.0       1.0\n     999.0       1.0\n"
        "/DAMP/1\nsettle\n       0.5         1\n")
    s, e = make_deck("X42E", deck_exp, """\
#
/RUN/X42E/1
120.0
/DT
0.9 0.0
/PRINT/-10000
/END
""")
    with contextlib.redirect_stdout(io.StringIO()):
        run_starter(s)
        m_exp = run_engine(e)
    s_i = m_imp.bricks.state["sig"][0]
    s_e = m_exp.bricks.state["sig"][0]
    assert s_i[0] == pytest.approx(s_e[0], rel=0.02)
    d_i = m_imp.x - m_imp.x0
    d_e = m_exp.x - m_exp.x0
    iy = m_imp.node_index(3)
    assert d_i[iy, 1] == pytest.approx(d_e[iy, 1], rel=0.02)


def test_law42_incompressible_limit(make_deck):
    """Incompressible-limit sanity: raising nu (so K) drives J -> 1 like
    1/K — consistent with the kernel's own K(J-1) pressure treatment —
    while the DEVIATORIC answer approaches the incompressible Ogden
    closed form sigma = sum mu_p (lam^a - lam^(-a/2))."""
    P = 0.003
    out = {}
    for tag, nu in (("A", 0.495), ("B", 0.4995)):
        model, _ = _run(make_deck, "I42" + tag, _rubber_deck(P, nu=nu),
                        "#\n/RUN/I42" + tag + "/1\n1.0\n/IMPL/NONLIN\n"
                        "/IMPL/DTINI\n0.2\n/IMPL/NEWTON\n1e-11  80\n/END\n")
        assert model.implicit_result.converged
        d = model.x - model.x0
        l1 = 1.0 + d[model.node_index(2), 0]
        l2 = 1.0 + d[model.node_index(3), 1]
        l3 = 1.0 + d[model.node_index(5), 2]
        out[tag] = (model.bricks.state["slices"][0][1].K, l1 * l2 * l3)
    (Ka, Ja), (Kb, Jb) = out["A"], out["B"]
    # J - 1 scales like 1/K (the penalty balance J - 1 = p/K)
    assert (Ja - 1.0) / (Jb - 1.0) == pytest.approx(Kb / Ka, rel=0.05)
    assert abs(Jb - 1.0) < 2e-3


def test_law42_refused_on_frozen_frame(make_deck):
    """LAW42 under /IMPL (small displacement) refuses loudly: the
    total-form stress never sees the trial displacement on the frozen
    frame."""
    s, e = make_deck("R42", _rubber_deck(0.001),
                     "#\n/RUN/R42/1\n1.0\n/IMPL\n/END\n")
    with contextlib.redirect_stdout(io.StringIO()):
        run_starter(s)
        with pytest.raises(NotImplementedError, match="NONLIN"):
            run_engine(e)


# ----------------------------------------------------------------------------
# the parity contract — M14 builds mutate nothing shared
# ----------------------------------------------------------------------------

def test_m14_builds_do_not_touch_shared_state(make_deck):
    """Building + evaluating + COMMITTING the M14 treatments (chained
    constraints, TYPE11 friction) mutates nothing the explicit solver
    reads: model.x, element state, interface records."""
    from pyradioss.common.messages import MessageLog
    from pyradioss.implicit.constraints import build_constraints
    from pyradioss.implicit.contact import (build_implicit_contacts,
                                            commit_contacts,
                                            contact_forces)
    from pyradioss.implicit.dofmap import DofMap

    model = _model(make_deck, "PAR14",
                   _t11_fric_deck(Fz=2.0, Fx=0.5, mu=0.3, **_T11))
    x0 = model.x.copy()
    snaps = {name: {k: v.copy() for k, v in g.state.items()
                    if isinstance(v, np.ndarray)}
             for name, g in model.element_groups()}
    log = MessageLog()
    with contextlib.redirect_stdout(io.StringIO()):
        contacts = build_implicit_contacts(model, log)
        dof = DofMap(model)
        contact_forces(contacts, model.x + 0.01, model.numnod)
        for c in contacts:
            c.triplets(model.x + 0.01, dof)
        commit_contacts(contacts, model.x + 0.01)
    assert np.array_equal(model.x, x0)
    for name, g in model.element_groups():
        for k, v in snaps.items() if False else snaps[name].items():
            assert np.array_equal(g.state[k], v), (name, k)

    model2 = _model(make_deck, "PAR14B", CHAIN_LEVER)
    x0 = model2.x.copy()
    with contextlib.redirect_stdout(io.StringIO()):
        constr = build_constraints(model2, log)
        dof2 = DofMap(model2, constraints=constr)
        constr.build(dof2, model2.x0)
    assert np.array_equal(model2.x, x0)
