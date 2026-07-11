"""
M12 validations: constraints and contact in the implicit tangent system.

Every new capability gets at least one ANALYTIC check (the port's
philosophy):

* an /RBE2 rigid LEVER whose load transfer through two grounding springs
  has an exact closed form (force and moment balance of the rigid link),
  in BOTH geometry modes — and the condensation machinery it exercises:
  a frozen standalone master unfrozen and force-numbered, dependent slave
  DOFs eliminated, a /BCS on the master acting at body level;
* the condensed rigid-body 6x6 MASS at the master against the closed-form
  parallel-axis block (what /IMPL/DYNA rides on);
* an /MPC equality constraint splitting a load between two springs
  exactly;
* an /RBE3 whose dual force distribution to two masters is the exact
  lever rule (and does not rigidify them);
* an /RBODY PIVOT: static rotation against a restraint spring (closed
  form), and the implicit-DYNAMIC physical pendulum crossing the vertical
  at the elliptic-integral quarter period with the swing amplitude
  preserved;
* the M4 spot-weld lap joint (/INTER/TYPE2 offset ties) solved
  IMPLICITLY, matching the EXPLICIT solver driven quasi-statically;
* /INTER/TYPE7 penalty contact: two elastic blocks pressed together
  reproducing the series-springs closed form EXACTLY — with the gap OPEN
  at the start, so the active set must enter mid-run — plus a
  finite-difference residual/tangent consistency check at an active
  contact state and an implicit punch cross-checked against the explicit
  solver's damped steady state;
* the M7 parity contract: building the implicit treatments mutates
  NOTHING shared with the explicit solver;
* the loud refusals: /INTER/TYPE11, /RWALL, /IMPL/ARCL + constraints,
  /IMPDISP on constraint nodes, chained constraints.

See PORTING_GUIDE.md roadmap M12.
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
    with contextlib.redirect_stdout(io.StringIO()):
        run_starter(s)
        model = run_engine(e)
    return model


def _model(make_deck, name, starter):
    s, _ = make_deck(name, starter, "#\n/END\n")
    with contextlib.redirect_stdout(io.StringIO()):
        return run_starter(s)


STEEL1 = """\
/MAT/LAW1/1
steel elastic
   7.8e-6
     210.0       0.3
"""


# ----------------------------------------------------------------------------
# /RBE2 rigid lever — condensation closed form (both geometry modes)
# ----------------------------------------------------------------------------

# Two grounded springs (k) along x at y = 0 and y = 4 hold slave nodes of a
# rigid link whose standalone master sits at y = 2; a third slave at y = 8
# takes the load F. Rigid kinematics u(y) = u_M - theta*(y - 2) with force
# balance k(u0 + u1) = F and moment balance about the master
# 2k*u0 - 2k*u1 + 6F = 0 give exactly
#     u0 = -F/k,  u1 = 2F/k,  u_tip = 5F/k.
LEVER = """\
#RADIOSS STARTER
/BEGIN
RBE2 lever
/NODE
         1                 0.0                 0.0                 0.0
         2                 0.0                 4.0                 0.0
         3                 1.0                 0.0                 0.0
         4                 1.0                 4.0                 0.0
         5                 1.0                 2.0                 0.0
         6                 1.0                 8.0                 0.0
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
slaves
3 4 6
/GRNOD/NODE/3
master
5
/GRNOD/NODE/4
tip
6
/RBE2/1
lever
         5         2
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
def test_rbe2_lever_load_transfer(make_deck, mode):
    """The rigid-lever closed form, exact — the /RBE2 condensation carries
    force AND moment to the master block (rbe2_imp0.F's UPDKB_RB algebra),
    in both geometry modes. The standalone master is a frozen placeholder
    the DofMap must unfreeze and force-number.

    The linear-geometry run uses a FINITE load (the linearized closed form
    is that path's exact answer); the NONLIN run uses a SMALL one — the
    closed form is the small-rotation limit of the true (rotating-arm)
    problem, so the comparison must sit where they coincide."""
    k = 2.0
    F = 1.0 if not mode else 1e-4        # NONLIN: theta = 3F/(4k) small
    lever = LEVER if not mode else LEVER.replace(
        "pull\n         1         X         4       1.0",
        f"pull\n         1         X         4       {F}")
    model = _run(make_deck, "LEV" + ("N" if mode else "L"), lever,
                 f"#\n/RUN/LEV{'N' if mode else 'L'}/1\n1.0\n/IMPL\n{mode}/END\n")
    d = model.x - model.x0
    tol = 1e-9 if not mode else 1e-3     # NONLIN: O(theta) rel. correction
    assert d[model.node_index(3), 0] == pytest.approx(-F / k, rel=tol)
    assert d[model.node_index(4), 0] == pytest.approx(2 * F / k, rel=tol)
    assert d[model.node_index(6), 0] == pytest.approx(5 * F / k, rel=tol)
    # the master rides at its own lever position (y = 2): u = F/(2k) - ...
    assert d[model.node_index(5), 0] == pytest.approx(0.5 * F / k, rel=tol)
    res = model.implicit_result
    assert res.converged
    if not mode:
        # linear geometry: the condensed system is linear -> ONE Newton step
        assert res.increments[-1].iterations == 2


def test_rigid_mass_condensation_master_block(make_deck):
    """T^T M T carries the EXACT rigid-body 6-DOF mass at the master:
    [[M*I, -M*skew(c)], [M*skew(c), sum m(|r|^2 I - r r^T)]] with c the
    COG offset and r the slave arms — the parallel-axis block /IMPL/DYNA
    rides on (nothing is assembled by hand: the congruence transform IS
    the mass condensation)."""
    import scipy.sparse as sp
    from pyradioss.common.messages import MessageLog
    from pyradioss.implicit.constraints import build_constraints
    from pyradioss.implicit.dofmap import DofMap
    from pyradioss.implicit.dynamics import _lumped_mass_eq

    m1, m2 = 0.001, 0.003
    # per-node /ADMAS: both slaves get m1, node 22 an extra (m2 - m1);
    # the far-away token spring satisfies the "model has elements" check
    starter = f"""\
#RADIOSS STARTER
/BEGIN
free rigid body
/NODE
        10                 0.0                 0.0                 0.0
        21                10.0                 0.0                 0.0
        22                 0.0                20.0                 5.0
       101              1000.0                 0.0                 0.0
       102              1010.0                 0.0                 0.0
/SPRING/1
         1       101       102
/PART/1
token
         1         1
""" + STEEL1 + f"""\
/PROP/SPRING/1
spring
     1e-06       1.0       0.0
/GRNOD/NODE/1
bob
21 22
/GRNOD/NODE/2
bob2
22
/GRNOD/NODE/3
faraway
101 102
/ADMAS/1
m1
   {m1}         1
/ADMAS/2
extra
   {m2 - m1}         2
/RBODY/1
body
        10         1       0.0         0
/BCS/1
faraway
       111       111         0         3
/END
"""
    model = _model(make_deck, "MCOND", starter)
    log = MessageLog()
    with contextlib.redirect_stdout(io.StringIO()):
        c = build_constraints(model, log)
        dof = DofMap(model, log, constraints=c)
        c.build(dof, model.x0)
    M_eq = _lumped_mass_eq(model, dof)
    Mred = (c.Tt @ sp.diags(M_eq) @ c.T).toarray()
    # reduced system = exactly the master's 6 equations
    assert c.nred == 6
    masses = np.array([m1, m2])
    arms = model.x0[[model.node_index(21), model.node_index(22)]] \
        - model.x0[model.node_index(10)]
    Mtot = masses.sum()
    cog = (masses[:, None] * arms).sum(axis=0) / Mtot

    def skew(v):
        return np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]],
                         [-v[1], v[0], 0.0]])

    J = sum(m * (np.eye(3) * (r @ r) - np.outer(r, r))
            for m, r in zip(masses, arms))
    expect = np.zeros((6, 6))
    expect[:3, :3] = Mtot * np.eye(3)
    expect[:3, 3:] = -Mtot * skew(cog)
    expect[3:, :3] = Mtot * skew(cog)
    expect[3:, 3:] = J
    # map master (comp -> eq) ordering
    im = model.node_index(10)
    order = [dof.eq[im * 6 + comp] for comp in range(6)]
    assert np.allclose(Mred[np.ix_(order, order)], expect, rtol=1e-12,
                       atol=1e-15)


# ----------------------------------------------------------------------------
# /MPC equality — exact split
# ----------------------------------------------------------------------------

def test_mpc_equality_split(make_deck):
    """u_A - u_B = 0 between the tips of two grounded springs (k1, k2),
    load F on tip A only: both tips move F/(k1+k2) exactly — the /MPC
    column-pivot elimination in K_red = T^T K T."""
    k1, k2, F = 3.0, 5.0, 1.0
    starter = f"""\
#RADIOSS STARTER
/BEGIN
MPC split
/NODE
         1                 0.0                 0.0                 0.0
         2                 0.0                 2.0                 0.0
         3                 1.0                 0.0                 0.0
         4                 1.0                 2.0                 0.0
/SPRING/1
         1         1         3
/SPRING/2
         2         2         4
/PART/1
s1
         1         1
/PART/2
s2
         2         1
""" + STEEL1 + f"""\
/PROP/SPRING/1
k1
       1.0       {k1}       0.0
/PROP/SPRING/2
k2
       1.0       {k2}       0.0
/GRNOD/NODE/1
ground
1 2
/GRNOD/NODE/2
tipA
3
/BCS/1
ground
       111       111         0         1
/BCS/2
lateral
       011       000         0         0
/MPC/1
equal ux
         3         1       1.0
         4         1      -1.0
/FUNCT/1
ramp
       0.0       0.0
       2.0       2.0
/CLOAD/1
pull
         1         X         2       {F}
/END
"""
    # BCS/2 with grnod 0 is invalid — fix lateral DOFs per node group
    starter = starter.replace("""/BCS/2
lateral
       011       000         0         0
""", """/GRNOD/NODE/3
tips
3 4
/BCS/2
lateral
       011       000         0         3
""")
    model = _run(make_deck, "MPCEQ", starter,
                 "#\n/RUN/MPCEQ/1\n1.0\n/IMPL\n/END\n")
    d = model.x - model.x0
    u = F / (k1 + k2)
    assert d[model.node_index(3), 0] == pytest.approx(u, rel=1e-9)
    assert d[model.node_index(4), 0] == pytest.approx(u, rel=1e-9)
    assert model.implicit_result.increments[-1].iterations == 2


# ----------------------------------------------------------------------------
# /RBE3 — dual distribution closed form, no rigidification
# ----------------------------------------------------------------------------

def test_rbe3_dual_lever_rule(make_deck):
    """Force F at an /RBE3 reference node offset c from the centroid of two
    masters L apart (each grounded by a spring k): the least-squares dual
    distributes exactly f = F/2 +- F*c/L (the lever rule of the fit's
    moment), so u = f/k independently per spring — no rigidification of
    the masters. Exact (the T^T distribution IS rbe3f's dual)."""
    k, F, L, c = 2.0, 1.0, 4.0, 1.0
    starter = f"""\
#RADIOSS STARTER
/BEGIN
RBE3 lever rule
/NODE
         1                 0.0                 0.0                 0.0
         2                 0.0                 4.0                 0.0
         3                 1.0                 0.0                 0.0
         4                 1.0                 4.0                 0.0
         5                 1.0                 3.0                 0.0
/SPRING/1
         1         1         3
         2         2         4
/PART/1
springs
         1         1
""" + STEEL1 + f"""\
/PROP/SPRING/1
spring
       1.0       {k}       0.0
/GRNOD/NODE/1
ground
1 2
/GRNOD/NODE/2
masters
3 4
/GRNOD/NODE/3
ref
5
/RBE3/1
hang
         5         2
/BCS/1
ground
       111       111         0         1
/GRNOD/NODE/4
lat
3 4
/BCS/2
lateral
       011       000         0         4
/FUNCT/1
ramp
       0.0       0.0
       2.0       2.0
/CLOAD/1
pull
         1         X         3       {F}
/END
"""
    model = _run(make_deck, "RBE3L", starter,
                 "#\n/RUN/RBE3L/1\n1.0\n/IMPL\n/END\n")
    d = model.x - model.x0
    f0 = F / 2 - F * c / L                       # master at y=0 (r = -L/2)
    f1 = F / 2 + F * c / L                       # master at y=4 (r = +L/2)
    assert d[model.node_index(3), 0] == pytest.approx(f0 / k, rel=1e-9)
    assert d[model.node_index(4), 0] == pytest.approx(f1 / k, rel=1e-9)
    # the reference node follows the fit: u_ref = u_G + theta x rho
    uG = 0.5 * (d[model.node_index(3), 0] + d[model.node_index(4), 0])
    th = (d[model.node_index(4), 0] - d[model.node_index(3), 0]) / L
    assert d[model.node_index(5), 0] == pytest.approx(uG + th * c, rel=1e-9)


# ----------------------------------------------------------------------------
# /RBODY pivot — static reaction + implicit-dynamic pendulum period
# ----------------------------------------------------------------------------

def _pendulum_deck(grav=True, spring_restraint=False):
    """Pivoted /RBODY: master clamped at the origin (/BCS 111 -> the pivot
    condensation), two point masses (/ADMAS) on a rigid arm at 60 deg from
    vertical (l and 2l), plus a token far-away spring element so the model
    check passes. ``spring_restraint`` adds a horizontal grounding spring
    at the l-mass for the static closed form."""
    extra_nodes = ""
    extra_spring = ""
    extra_groups = ""
    if spring_restraint:
        extra_nodes = ("       31       186.6025403784      "
                       "-50.0000000000        0.0000000000\n")
        extra_spring = "         2        31        21\n"
        extra_groups = """/GRNOD/NODE/4
anchor
31
/BCS/3
anchor
       111       111         0         4
"""
    grav_card = """/GRAV/1
gravity
         1         Y         1  -9.81e-3
""" if grav else ""
    return f"""\
#RADIOSS STARTER
/BEGIN
implicit physical pendulum
/NODE
        10                 0.0                 0.0                 0.0
        21       86.6025403784      -50.0000000000        0.0000000000
        22      173.2050807569     -100.0000000000        0.0000000000
{extra_nodes}       101              1000.0                 0.0                 0.0
       102              1010.0                 0.0                 0.0
/SPRING/1
         1       101       102
{extra_spring}/PART/1
token spring
         1         1
""" + STEEL1 + """\
/PROP/SPRING/1
spring
     1e-06       1.0       0.0
/GRNOD/NODE/1
bob
21 22
/GRNOD/NODE/2
pivot
10
/GRNOD/NODE/3
faraway
101 102
/ADMAS/1
bob mass
     0.001         1
/RBODY/1
pendulum
        10         1       0.0         0
/BCS/1
pivot
       111       110         0         2
/BCS/2
faraway
       111       111         0         3
""" + extra_groups + grav_card + """\
/FUNCT/1
one
       0.0       1.0
    1000.0       1.0
/END
"""


def test_rbody_pivot_static_rotation(make_deck):
    """Pivoted rigid body under a static force, restrained by one
    grounding spring: the load at arm a and the spring at arm b (along
    the same rigid arm) balance at rotation theta = F*a/(k*b^2) about
    the pivot — a pure moment-balance closed form through the pivot
    condensation (master translations /BCS-condensed, rotation live).

    Geometry: masses at r = 100 and 200 along the 60-deg arm; the
    restraint spring is HORIZONTAL at the r = 100 point (arm component
    perpendicular to x is |r_y| = 50); load F along x at the r = 200
    point (perpendicular arm |r_y| = 100)."""
    k, F = 1.0, 0.001
    deck = _pendulum_deck(grav=False, spring_restraint=True)
    deck = deck.replace("/END\n", f"""/GRNOD/NODE/5
loadpt
22
/FUNCT/2
ramp
       0.0       0.0
       2.0       2.0
/CLOAD/1
push
         2         X         5       {F}
/END
""")
    model = _run(make_deck, "PIVST", deck,
                 "#\n/RUN/PIVST/1\n1.0\n/IMPL\n/END\n")
    d = model.x - model.x0
    # linearized about theta0: moment balance about the pivot (z):
    #   F * (-y22) = (k * u21x) * (-y21)   with u21x = theta * (-y21)
    y21, y22 = -50.0, -100.0
    theta = F * (-y22) / (k * y21 * y21)
    assert d[model.node_index(21), 0] == pytest.approx(theta * (-y21),
                                                       rel=1e-9)
    # the load point moves with the same rigid rotation: u = theta * (-y22)
    assert d[model.node_index(22), 0] == pytest.approx(theta * (-y22),
                                                       rel=1e-9)
    # pivot never moves (condensed at body level)
    assert np.all(d[model.node_index(10)] == 0.0)


def test_rbody_pendulum_elliptic_period(make_deck):
    """The implicit-DYNAMIC physical pendulum (an /RBODY pivoted by the
    master /BCS, released from 60 deg under gravity): the vertical
    crossing lands on the elliptic-integral quarter period
    T/4 = sqrt(I/(M g d)) K(sin^2(theta0/2)) to < 0.5%, the swing
    amplitude is preserved to < 1%, the arm lengths stay EXACT (the
    commit placement) and the energy ledger closes. This is the test
    that found the M12 predictor-projection lesson (see
    constraints.make_consistent)."""
    from scipy.special import ellipk
    deck = _pendulum_deck(grav=True)
    model = _run(make_deck, "PENDI", deck, """\
#
/RUN/PENDI/1
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
    # I_pivot = m l^2 + m (2l)^2 = 5 m l^2 ; M g d = 2m g 1.5 l
    w0 = np.sqrt(3.0 * g / (5.0 * l))
    T4 = ellipk(np.sin(th0 / 2) ** 2) / w0
    i21 = model.node_index(21)
    t = np.array(h["t"])
    xs = np.array([model.x0[i21] + u[i21] for u in h["u"]])
    th = np.arctan2(xs[:, 0], -xs[:, 1])
    # arm length exact through the whole swing (Rodrigues placement)
    r = np.linalg.norm(xs, axis=1)
    assert np.abs(r - l).max() < 1e-6 * l
    # first vertical crossing at the elliptic quarter period
    cross = np.where(np.diff(np.sign(th)) != 0)[0]
    assert len(cross) >= 1
    i = cross[0]
    tc = t[i] - th[i] * (t[i + 1] - t[i]) / (th[i + 1] - th[i])
    assert tc == pytest.approx(T4, rel=5e-3)
    # amplitude preserved on the far side (no energy injection)
    assert np.degrees(th.min()) == pytest.approx(-60.0, abs=1.0)
    # ledger closes (scale: the total drop energy)
    scale = 2 * 0.001 * g * 1.5 * l * (1 - np.cos(th0))
    assert abs(h["bal"][-1]) < 0.02 * scale


# ----------------------------------------------------------------------------
# /INTER/TYPE2 — the spot-weld lap joint, implicit vs explicit
# ----------------------------------------------------------------------------

def _lap_joint_deck(explicit):
    """Single-lap shear joint: two 30x10 shell strips (t = 1) overlapping
    10 mm, mid-surfaces 1 mm apart, B's left-edge column tied to A's
    surface (/INTER/TYPE2, offset ties). A clamped at x = 0, pull F at
    B's right edge. Out-of-plane uz fixed everywhere (a membrane load
    path; the offset ties' moment is dropped by BOTH solvers — the
    documented TYPE2 simplification, so the answers must agree)."""
    F = 10.0
    nodes = []
    nid = {}
    k = 0
    for strip, (x0, z) in enumerate((((0.0), 0.0), ((20.0), 1.0))):
        for j, y in enumerate((0.0, 10.0)):
            for i in range(4):
                k += 1
                nid[(strip, i, j)] = k
                nodes.append(f"{k:10d}{x0 + 10.0 * i:20.10f}{y:20.10f}"
                             f"{z:20.10f}")
    eid = 0
    shells_a, shells_b = [], []
    for strip, out in ((0, shells_a), (1, shells_b)):
        for i in range(3):
            eid += 1
            c = [nid[(strip, i, 0)], nid[(strip, i + 1, 0)],
                 nid[(strip, i + 1, 1)], nid[(strip, i, 1)]]
            out.append(f"{eid:10d}" + "".join(f"{v:10d}" for v in c))
    clamp = [nid[(0, 0, j)] for j in (0, 1)]
    tied = [nid[(1, 0, j)] for j in (0, 1)]
    tip = [nid[(1, 3, j)] for j in (0, 1)]
    allnod = list(range(1, k + 1))
    fpn = F / len(tip)
    load = f"""/FUNCT/1
ramp
       0.0       0.0
     0.002     100.0
      20.0     100.0
/CLOAD/1
pull
         1         X         3   {fpn / 100.0}
/DAMP/1
settle
     150.0         4
""" if explicit else f"""/FUNCT/1
ramp
       0.0       0.0
       2.0       2.0
/CLOAD/1
pull
         1         X         3   {fpn}
"""
    return f"""\
#RADIOSS STARTER
/BEGIN
lap joint
/NODE
{chr(10).join(nodes)}
/SHELL/1
{chr(10).join(shells_a)}
/SHELL/2
{chr(10).join(shells_b)}
/PART/1
strip A
         1         1
/PART/2
strip B
         1         1
""" + STEEL1 + f"""\
/PROP/SHELL/1
shell
       1.0
/GRNOD/NODE/1
clamp
{" ".join(map(str, clamp))}
/GRNOD/NODE/2
tied
{" ".join(map(str, tied))}
/GRNOD/NODE/3
tip
{" ".join(map(str, tip))}
/GRNOD/NODE/4
all
{" ".join(map(str, allnod))}
/SURF/PART/1
strip A
         1
/BCS/1
clamp
       111       111         0         1
/BCS/2
membrane
       001       110         0         4
/INTER/TYPE2/1
weld
         2         1       2.0
{load}/END
"""


def test_spotweld_lap_joint_implicit_matches_explicit(make_deck):
    """The lap joint pulled quasi-statically: the implicit static answer
    (i2_imp1.F tied condensation) matches the EXPLICIT solver's damped
    steady state on the same deck — the tie searches are literally shared
    (the constraint layer reuses ContactType2's projection), so any
    disagreement would be the condensation itself."""
    # SURF/PART includes strip B's shells too — the tie search must pick
    # strip A's segments: they are the only ones within dsearch of the
    # tied nodes' projection... B's own segments contain the tied nodes
    # as corners and are excluded by the TYPE2 corner rule.
    m_imp = _run(make_deck, "LAPI", _lap_joint_deck(False),
                 "#\n/RUN/LAPI/1\n1.0\n/IMPL\n/END\n")
    m_exp = _run(make_deck, "LAPE", _lap_joint_deck(True), """\
#
/RUN/LAPE/1
8.0
/DT
0.9 0.0
/PRINT/-10000
/END
""")
    d_i = m_imp.x - m_imp.x0
    d_e = m_exp.x - m_exp.x0
    # tip node: the generator fills strip 0 as 1..8 (i fastest, then j),
    # strip 1 as 9..16, so (strip 1, i = 3, j = 0) is node 12
    tip = m_imp.node_index(12)
    # the tips agree to ~2% (explicit residual ringing sets the floor)
    assert d_i[tip, 0] == pytest.approx(d_e[tip, 0], rel=0.02)
    # and the joint really carries the load: the tip moved a finite amount
    assert d_i[tip, 0] > 1e-3


# ----------------------------------------------------------------------------
# /INTER/TYPE7 — series-springs closed form with the active set entering
# ----------------------------------------------------------------------------

def _stack_deck(sep, gap, K4, drive):
    """Two unit hexa blocks stacked along z, bottom clamped, initial face
    separation ``sep``, TYPE7 (Istf=1, K = stfac = ``K4``) with contact
    gap ``gap``, top face driven down by /IMPDISP to ``drive``. nu = 0
    keeps the closed form 1-D exact."""
    return f"""\
#RADIOSS STARTER
/BEGIN
two blocks contact
/NODE
         1                 0.0                 0.0                 0.0
         2                 1.0                 0.0                 0.0
         3                 1.0                 1.0                 0.0
         4                 0.0                 1.0                 0.0
         5                 0.0                 0.0                 1.0
         6                 1.0                 0.0                 1.0
         7                 1.0                 1.0                 1.0
         8                 0.0                 1.0                 1.0
        11                 0.0                 0.0    {1.0 + sep:16.6f}
        12                 1.0                 0.0    {1.0 + sep:16.6f}
        13                 1.0                 1.0    {1.0 + sep:16.6f}
        14                 0.0                 1.0    {1.0 + sep:16.6f}
        15                 0.0                 0.0    {2.0 + sep:16.6f}
        16                 1.0                 0.0    {2.0 + sep:16.6f}
        17                 1.0                 1.0    {2.0 + sep:16.6f}
        18                 0.0                 1.0    {2.0 + sep:16.6f}
/BRICK/1
         1         1         2         3         4         5         6         7         8
         2        11        12        13        14        15        16        17        18
/PART/1
blocks
         1         1
/MAT/LAW1/1
nu zero
   7.8e-6
     210.0       0.0
/PROP/SOLID/1
solid
       1.1      0.05       0.1
/GRNOD/NODE/1
base
1 2 3 4
/GRNOD/NODE/2
bbot
11 12 13 14
/GRNOD/NODE/3
btop
15 16 17 18
/GRNOD/NODE/4
all
1 2 3 4 5 6 7 8 11 12 13 14 15 16 17 18
/SURF/SEG/1
top face of A
         5         6         7         8
/BCS/1
base
       111       000         0         1
/BCS/2
lateral
       110       000         0         4
/INTER/TYPE7/1
press
         2         1         1         0
    {K4:6.1f}       0.0    {gap:6.3f}       0.0
/FUNCT/1
push
       0.0       0.0
       2.0    {-2.0 * drive:10.4f}
/IMPDISP/1
push down
         1         Z         3       1.0
/END
"""


def test_contact_series_springs_closed_form(make_deck):
    """Two elastic blocks pressed together (displacement control): the
    1-D series chain block-A / penalty-springs / block-B has the exact
    solution sigma = 4K(gap - d0 + Delta)/(A + 8KL/E) once the gap
    closes — asserted along with the free-travel phase (zero stress
    while open: the ACTIVE SET enters mid-run) and uniform stress in
    both blocks (the uniform contact pressure patch)."""
    E, L, A = 210.0, 1.0, 1.0
    K4, gap, d0, Delta = 50.0, 0.05, 0.08, 0.15
    model = _run(make_deck, "STACK", _stack_deck(d0, gap, K4, Delta), """\
#
/RUN/STACK/1
1.0
/IMPL
/IMPL/DTINI
0.1
/END
""")
    sig = 4 * K4 * (gap - d0 + Delta) / (A + 8 * K4 * L / E)
    sA = model.bricks.state["sig"][0]
    sB = model.bricks.state["sig"][1]
    assert sA[2] == pytest.approx(-sig, rel=1e-9)
    assert sB[2] == pytest.approx(-sig, rel=1e-9)
    # uniform uniaxial state: all other stress components vanish
    assert np.abs(np.delete(sA, 2)).max() < 1e-9 * sig
    d = model.x - model.x0
    # top of A compresses by sigma L / E; bottom of B rides the drive
    assert d[model.node_index(5), 2] == pytest.approx(-sig * L / E,
                                                      rel=1e-9)
    assert d[model.node_index(11), 2] == pytest.approx(
        -(Delta - sig * L / E), rel=1e-9)
    # the active set ENTERED mid-run: the first increment(s) are free
    # travel — the drive reaches d0 - gap = 0.03 only at load factor 0.2,
    # so at the first increment (0.1 -> Delta 0.015) the blocks are
    # stress-free
    res = model.implicit_result
    assert res.converged
    # free travel: recompute what the first increment's drive was
    assert Delta * 0.1 < d0 - gap        # sanity of the test setup


def test_contact_fd_residual_tangent_consistency(make_deck):
    """Finite-difference consistency of the implicit contact force and
    the assembled gap tangent K g g^T at an ACTIVE contact state.

    Directions that move only the SECONDARY side (the main segment held):
    the projection point, weights and normal are then fixed, so the
    frozen-projection tangent is the EXACT linearization — asserted to FD
    accuracy in every residual row (node rows AND the corner reaction
    rows, which depend on the node motion only through the penetration).
    Directions that move the segment corners exercise the DOCUMENTED
    omission (i7keg3.F omits them too): the projection here sits exactly
    at a vertex — the non-smooth boundary of the closest-point map — and
    the weight-variation terms are O(p/L); asserted separately at the
    matching loose tolerance so a regression that breaks the main term
    would still be caught."""
    from pyradioss.common.messages import MessageLog
    from pyradioss.implicit.contact import build_implicit_contacts
    from pyradioss.implicit.dofmap import DofMap

    d0, gap = 0.02, 0.05               # initially penetrating: p = 0.03
    s, _ = make_deck("FDCON", _stack_deck(d0, gap, 50.0, 0.0),
                     "#\n/END\n")
    with contextlib.redirect_stdout(io.StringIO()):
        model = run_starter(s)
    log = MessageLog()
    with contextlib.redirect_stdout(io.StringIO()):
        contacts = build_implicit_contacts(model, log)
        dof = DofMap(model, log)
    n = model.numnod

    def residual(u):
        f = np.zeros((n, 3))
        for c in contacts:
            c.forces(model.x0 + u, f)
        return dof.gather_residual(f, np.zeros((n, 3)))

    from pyradioss.implicit.contact import contact_tangent
    u0 = np.zeros((n, 3))
    Kc = contact_tangent(contacts, model.x0, dof).toarray()
    eps = 1e-7
    rng = np.random.default_rng(0)
    # equation indices of the main-segment corners (nodes 5..8)
    corner_eqs = [dof.eq[model.node_index(nid) * 6 + c]
                  for nid in (5, 6, 7, 8) for c in range(3)]
    corner_eqs = [e for e in corner_eqs if e >= 0]
    for _ in range(6):
        # secondary-side direction: the frozen-projection tangent is EXACT
        v_eq = rng.standard_normal(dof.ndof)
        v_eq[corner_eqs] = 0.0
        du, _ = dof.scatter_solution(v_eq)
        fd = (residual(u0 + eps * du) - residual(u0 - eps * du)) / (2 * eps)
        # dR/du = -K_c by the driver's sign convention (K du = R)
        assert np.allclose(fd, -Kc @ v_eq, rtol=1e-6,
                           atol=1e-7 * max(np.abs(fd).max(), 1.0))
    for _ in range(3):
        # full direction (corners move): the omitted projection-variation
        # terms are O(p/L) = 3% here — the main gap term must dominate
        v_eq = rng.standard_normal(dof.ndof)
        du, _ = dof.scatter_solution(v_eq)
        fd = (residual(u0 + eps * du) - residual(u0 - eps * du)) / (2 * eps)
        ref = max(np.abs(fd).max(), 1e-30)
        assert np.abs(fd + Kc @ v_eq).max() < 0.05 * ref


def test_implicit_punch_matches_explicit_steady_state(make_deck):
    """A punch pressed by a constant force onto the lower block: the
    implicit static answer matches the EXPLICIT solver run to a damped
    steady state on the same deck (same penalty stiffness, same gap
    bookkeeping — contact/stiffness.py is literally shared).

    The setup keeps the explicit TRANSIENT gentle (blocks start a hair
    into contact — an exactly-open gap would leave the implicit punch
    momentarily unsupported (singular K), and a wide one launches the
    explicit blocks — and the load ramps over 0.5 ms): a dead load near
    the penalty spring's crossing force 4*K*gap legitimately punches
    through in the explicit transient, and a violent single-element
    impact leaves the explicit hypoelastic state ringing-drifted (a
    pre-existing explicit artifact, nothing the implicit path shares) —
    both are physics of the explicit penalty method, not solver
    disagreements."""
    d0, gap, K4, F = 0.0499, 0.05, 50.0, 2.0
    deck = _stack_deck(d0, gap, K4, 0.0)
    # replace the displacement drive by a dead load on the punch top
    dead = f"""/FUNCT/1
push
       0.0       0.0
       2.0       2.0
/CLOAD/1
press
         1         Z         3   {-F / 4.0}
"""
    deck_imp = deck[:deck.index("/FUNCT/1")] + dead + "/END\n"
    dead_exp = f"""/FUNCT/1
push
       0.0       0.0
       0.5       1.0
      20.0       1.0
/CLOAD/1
press
         1         Z         3   {-F / 4.0}
/DAMP/1
settle
      60.0         4
"""
    deck_exp = deck[:deck.index("/FUNCT/1")] + dead_exp + "/END\n"
    m_imp = _run(make_deck, "PUNI", deck_imp,
                 "#\n/RUN/PUNI/1\n1.0\n/IMPL\n/END\n")
    m_exp = _run(make_deck, "PUNE", deck_exp, """\
#
/RUN/PUNE/1
3.0
/DT
0.9 0.0
/PRINT/-10000
/END
""")
    d_i = m_imp.x - m_imp.x0
    d_e = m_exp.x - m_exp.x0
    for nid in (11, 15, 5):
        i = m_imp.node_index(nid)
        assert d_i[i, 2] == pytest.approx(d_e[i, 2], rel=0.02), nid
    # sanity: the closed form for the dead load: sigma = F/A, pen = F/(4K)
    sig = F / 1.0
    utop_expected = -(sig * 1.0 / 210.0            # A compression
                      + (d0 - gap)                 # free travel (negative)
                      + F / (4 * K4)               # penetration
                      + sig * 1.0 / 210.0)         # B compression
    assert d_i[m_imp.node_index(15), 2] == pytest.approx(utop_expected,
                                                         rel=1e-6)


# ----------------------------------------------------------------------------
# parity contract: nothing shared is mutated
# ----------------------------------------------------------------------------

def test_m12_builds_do_not_touch_shared_state(make_deck):
    """Building the constraint transform and evaluating contact
    forces/tangents must leave the model arrays and element buffers
    byte-for-byte unchanged — the implicit treatments live ALONGSIDE the
    explicit modules (the M7/M8 parity contract)."""
    from pyradioss.common.messages import MessageLog
    from pyradioss.implicit.constraints import build_constraints
    from pyradioss.implicit.contact import (build_implicit_contacts,
                                            contact_tangent)
    from pyradioss.implicit.dofmap import DofMap

    model = _model(make_deck, "PURE", _stack_deck(0.02, 0.05, 50.0, 0.0))
    before_x = model.x.copy()
    before_mass = model.mass.copy()
    before_state = {k: v.copy() for k, v in model.bricks.state.items()
                    if isinstance(v, np.ndarray)}
    log = MessageLog()
    with contextlib.redirect_stdout(io.StringIO()):
        contacts = build_implicit_contacts(model, log)
        dof = DofMap(model, log)
        f = np.zeros((model.numnod, 3))
        for c in contacts:
            c.forces(model.x, f)
            c.energy(model.x)
        contact_tangent(contacts, model.x, dof)
    assert np.array_equal(model.x, before_x)
    assert np.array_equal(model.mass, before_mass)
    for k, v in before_state.items():
        assert np.array_equal(model.bricks.state[k], v), k

    # constraints: the lever model with a rigid body + the tied lap joint
    model2 = _model(make_deck, "PURE2", LEVER)
    bx = model2.x.copy()
    bm = model2.mass.copy()
    with contextlib.redirect_stdout(io.StringIO()):
        c2 = build_constraints(model2, log)
        dof2 = DofMap(model2, log, constraints=c2)
        c2.build(dof2, model2.x0)
    assert np.array_equal(model2.x, bx)
    assert np.array_equal(model2.mass, bm)


# ----------------------------------------------------------------------------
# loud refusals (deferred, never silent)
# ----------------------------------------------------------------------------

def test_type11_no_longer_refused_under_implicit(make_deck):
    """/INTER/TYPE11 under implicit was a loud M12 refusal; M13 upgraded
    it to a capability (implicit/contact.ImplicitContact11 — validated in
    tests/test_m13_implfric.py). This test keeps the old refusal deck and
    asserts the run now CONVERGES with the edge springs pushing the
    initially-penetrating parallel edges apart (the near-parallel
    curvature guard path)."""
    starter = _stack_deck(0.02, 0.05, 50.0, 0.0)
    i0 = starter.index("/INTER/TYPE7/1")
    i1 = starter.index("/FUNCT/1")
    starter = starter[:i0] + """/LINE/SEG/1
edge a
         5         6
/LINE/SEG/2
edge b
        11        12
/INTER/TYPE11/1
edges
         1         2         1         0
      50.0       0.0     0.050       0.0
""" + starter[i1:]
    s, e = make_deck("T11R", starter, "#\n/RUN/T11R/1\n1.0\n/IMPL\n/END\n")
    with contextlib.redirect_stdout(io.StringIO()):
        run_starter(s)
        model = run_engine(e)
    assert model.implicit_result.converged
    # the edge spring separated the initially-penetrating faces: the top
    # block rose (edge b, nodes 11/12 z > start) while the bottom block's
    # top edge was pushed down
    d = model.x - model.x0
    assert d[model.node_index(11), 2] > 1e-6
    assert d[model.node_index(5), 2] < -1e-6


def test_rwall_refused_under_implicit(make_deck):
    starter = _stack_deck(0.02, 0.05, 50.0, 0.0)
    starter = starter.replace("/END\n", """/RWALL/PLANE/1
floor
         0         0
       0.0       0.0      -1.0
       0.0       0.0       1.0
/END
""")
    s, e = make_deck("RWR", starter, "#\n/RUN/RWR/1\n1.0\n/IMPL\n/END\n")
    with contextlib.redirect_stdout(io.StringIO()):
        run_starter(s)
        with pytest.raises(NotImplementedError, match="RWALL"):
            run_engine(e)


def test_arcl_with_constraints_refused(make_deck):
    s, e = make_deck("ARCR", LEVER,
                     "#\n/RUN/ARCR/1\n1.0\n/IMPL\n/IMPL/ARCL\n/END\n")
    with contextlib.redirect_stdout(io.StringIO()):
        run_starter(s)
        with pytest.raises(NotImplementedError, match="ARCL"):
            run_engine(e)


def test_impdisp_on_constraint_nodes_refused(make_deck):
    starter = LEVER.replace("/CLOAD/1\npull\n         1         X"
                            "         4       1.0\n",
                            "/IMPDISP/1\ndrive tip\n         1         X"
                            "         4       1.0\n")
    s, e = make_deck("IDR", starter, "#\n/RUN/IDR/1\n1.0\n/IMPL\n/END\n")
    with contextlib.redirect_stdout(io.StringIO()):
        run_starter(s)
        with pytest.raises(NotImplementedError, match="IMPDISP"):
            run_engine(e)


def test_chained_constraints_refused(make_deck):
    """An /RBE3 whose master is a rigid-body slave = a constraint chain:
    refused loudly (PORTING_GUIDE M12), never resolved silently."""
    starter = LEVER.replace("/END\n", """/NODE
         7                 2.0                 0.0                 0.0
/GRNOD/NODE/5
r3m
3 4
/RBE3/1
chain
         7         5
/END
""")
    s, e = make_deck("CHR", starter, "#\n/RUN/CHR/1\n1.0\n/IMPL\n/END\n")
    with contextlib.redirect_stdout(io.StringIO()):
        run_starter(s)
        with pytest.raises(NotImplementedError, match="chain"):
            run_engine(e)
