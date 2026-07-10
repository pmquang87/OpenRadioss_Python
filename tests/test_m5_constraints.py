"""M5 constraints & loads validations: /RBODY, /RBE2, /RBE3, moving and
spherical/cylindrical rigid walls, /SECT, /PLOAD, /IMPDISP, /ADMAS,
/INIVEL/AXIS.

Layered like the rest of the suite (PORTING_GUIDE §6):

* unit tests against closed-form results (rigid-body inertia tensor and
  COG, RBE3 force-distribution weights, sphere/cylinder projection,
  pressure lumping, exact /IMPDISP landing);
* full starter+engine analytic validations (a tumbling rigid body
  conserving angular momentum, a physical pendulum on an /RBE2 hitting
  the closed-form bottom speed, momentum/energy of an impact against a
  moving wall with a mass, a driven-wall push, /SECT recovering an
  applied load, /PLOAD momentum, /IMPDISP quasi-statics, and a rigid
  body impacting a deformable plate through /INTER/TYPE7 at /DT 0.9).

The constraints are kinematic: every test that runs the engine asserts
the global energy balance closes (constraints do no spurious work) — the
same discipline as the M4 contact tests.
"""

import contextlib
import io
import re

import numpy as np
import pytest

from pyradioss.common.messages import MessageLog
from pyradioss.engine.engine import run_engine
from pyradioss.engine.kinematics import LoadsAndConstraints
from pyradioss.engine.rbe3 import Rbe3Constraint
from pyradioss.engine.rigid_body import RigidBodyEngine, _exp_rotation
from pyradioss.engine.rigid_wall import RigidWalls
from pyradioss.model.entities import RigidWall
from pyradioss.starter.starter import run_starter

STEEL_LAW1 = """\
/MAT/LAW1/1
steel elastic
7.8e-6
210. 0.3
"""

RHO = 7.8e-6


def _run(make_deck, name, starter, engine):
    s, e = make_deck(name, starter, engine)
    with contextlib.redirect_stdout(io.StringIO()):
        run_starter(s)
        model = run_engine(e)
    return model, e.replace("_0001.rad", "_0001.out")


def _starter_only(make_deck, name, starter):
    s, _ = make_deck(name, starter, "/RUN/X/1\n1.0\n")
    with contextlib.redirect_stdout(io.StringIO()):
        return run_starter(s)


def _final_summary(out_path):
    text = open(out_path).read()
    err = float(re.search(r"ENERGY ERROR\s*\.[ .]*:\s*([-\d.Ee+]+)",
                          text).group(1))
    ce = float(re.search(r"CONTACT ENERGY\s*\.[ .]*:\s*([-\d.Ee+]+)",
                         text).group(1))
    ew = float(re.search(r"EXTERNAL WORK\s*\.[ .]*:\s*([-\d.Ee+]+)",
                         text).group(1))
    en = float(re.search(r"NUMERICAL DISSIPATION\s*\.[ .]*:\s*([-\d.Ee+]+)",
                         text).group(1))
    return {"ERR": err, "CE": ce, "EW": ew, "EN": en,
            "NORMAL": "ENGINE TERMINATION : NORMAL" in text}


def _read_th(path):
    """T01 CSV -> dict of column arrays."""
    with open(path) as fh:
        fh.readline()                       # comment line
        cols = fh.readline().strip().split(",")
        data = np.loadtxt(fh, delimiter=",", ndmin=2)
    return {c: data[:, k] for k, c in enumerate(cols)}


# ----------------------------------------------------------------------------
# Shared deck fragments
# ----------------------------------------------------------------------------

def _bar_two_bricks(master="100 5 5 5"):
    """2x1x1 bar of two unit bricks + a standalone master node."""
    return (
        "/NODE\n"
        "1 0 0 0\n2 1 0 0\n3 1 1 0\n4 0 1 0\n"
        "5 0 0 1\n6 1 0 1\n7 1 1 1\n8 0 1 1\n"
        "9 2 0 0\n10 2 1 0\n11 2 0 1\n12 2 1 1\n"
        f"{master}\n"
        "/BRICK/1\n1 1 2 3 4 5 6 7 8\n2 2 9 10 3 6 11 12 7\n"
        "/PART/1\nbar\n1 1\n" + STEEL_LAW1 +
        "/PROP/SOLID/1\nsolid\n1.1 0.05 0.1\n"
        "/GRNOD/PART/1\nall bar\n1\n"
    )


# ============================================================================
# Unit tests — rigid-body inertia tensor / COG (rbyini closed forms)
# ============================================================================

def test_rbody_inertia_tensor_closed_form(make_deck):
    """One unit brick: the lumped nodal masses are 8 point masses m/8 at
    the corners, so about the center J = diag(m a^2 / 2) with a = 1 —
    hand-checked: Jxx = sum m/8 (y^2+z^2) = 8 * m/8 * (1/4 + 1/4). Added
    mass and Jxx/Jyy/Jzz must add on top, and Icog=1 must move the master
    node to the COG."""
    starter = (
        "/BEGIN\ninertia probe\n"
        "/NODE\n"
        "1 0 0 0\n2 1 0 0\n3 1 1 0\n4 0 1 0\n"
        "5 0 0 1\n6 1 0 1\n7 1 1 1\n8 0 1 1\n"
        "100 9 9 9\n"
        "/BRICK/1\n1 1 2 3 4 5 6 7 8\n"
        "/PART/1\ncube\n1 1\n" + STEEL_LAW1 +
        "/PROP/SOLID/1\nsolid\n1.1 0.05 0.1\n"
        "/GRNOD/PART/1\ncube nodes\n1\n"
        "/RBODY/1\nrigid cube\n100 1 2.0e-6 1\n"
        "1e-6 2e-6 3e-6\n"
        "/END\n"
    )
    model = _starter_only(make_deck, "RBJ", starter)
    rb = model.rbodies[0]
    m = RHO * 1.0                                  # brick mass
    assert rb.mass_total == pytest.approx(m + 2.0e-6)
    assert rb.xg == pytest.approx([0.5, 0.5, 0.5])
    J_expect = np.diag([m / 2 + 1e-6, m / 2 + 2e-6, m / 2 + 3e-6])
    assert rb.J == pytest.approx(J_expect)
    # Icog = 1: master relocated to the COG, and it carries the added mass
    im = model.node_index(100)
    assert model.x0[im] == pytest.approx([0.5, 0.5, 0.5])
    assert model.mass[im] == pytest.approx(2.0e-6)


def test_rbody_initial_velocity_projection(make_deck):
    """/INIVEL/AXIS on the slaves + engine-side init: the body must pick
    up exactly the rigid content — v_cog = 0, L = J w — and re-scatter a
    consistent field."""
    starter = (
        "/BEGIN\nspin projection\n" + _bar_two_bricks() +
        "/RBODY/1\nrigid bar\n100 1 0 1\n"
        "/INIVEL/AXIS/1\nspin z\n10.0 Z 1 1.0 0.5 0.5\n"
        "/END\n"
    )
    model = _starter_only(make_deck, "RBP", starter)
    log = MessageLog()
    loads = LoadsAndConstraints(model, log)
    rbe = RigidBodyEngine(model.rbodies[0], model, loads, log)
    assert rbe.v_ref == pytest.approx([0.0, 0.0, 0.0], abs=1e-12)
    assert rbe.w == pytest.approx([0.0, 0.0, 10.0])
    # the scattered field is the exact rigid rotation about the COG
    r = model.x[rbe.slaves] - rbe.xg
    v_expect = np.cross([0.0, 0.0, 10.0], r)
    assert model.v[rbe.slaves] == pytest.approx(v_expect)
    # and L equals J w for the Starter's tensor
    assert rbe.L == pytest.approx(rbe.J0 @ [0.0, 0.0, 10.0])


def test_exp_rotation_orthogonal_and_exact():
    """The Rodrigues exponential map: exp([w]dt) must be orthogonal for
    any step and rotate by exactly |w| dt."""
    w = np.array([3.0, -4.0, 12.0])                # |w| = 13
    R = _exp_rotation(w, 0.1)
    assert R @ R.T == pytest.approx(np.eye(3), abs=1e-14)
    assert np.linalg.det(R) == pytest.approx(1.0)
    # trace(R) = 1 + 2 cos(theta)
    assert np.trace(R) == pytest.approx(1.0 + 2.0 * np.cos(1.3))
    # the axis is invariant
    assert R @ w == pytest.approx(w)


# ============================================================================
# Unit tests — RBE3 force distribution (rbe3f closed forms)
# ============================================================================

RBE3_SQUARE = (
    "/BEGIN\nrbe3 square\n"
    "/NODE\n"
    "1 -1 -1 0\n2 1 -1 0\n3 1 1 0\n4 -1 1 0\n"
    "9 0 0 1\n"
    "/SHELL/1\n1 1 2 3 4\n"
    "/PART/1\nplate\n1 1\n" + STEEL_LAW1 +
    "/PROP/SHELL/1\nplate\n1 0 0 0\n0.01 0.01 0.01 0 0\n3 0 1.0\n"
    "/GRNOD/NODE/1\nmasters\n1 2 3 4\n"
    "/GRNOD/NODE/9\nrefgrp\n9\n"
    "/ADMAS/1\nref mass\n1.0e-6 9\n"
    "/RBE3/1\nhang\n9 1\n"
    "/END\n"
)


def test_rbe3_force_distribution_closed_form(make_deck):
    """Masters at the corners of a 2x2 square, dependent node offset 1
    above the center. Closed forms (see engine/rbe3.py): a central force
    Fz splits equally (rho x F = 0); a transverse force Fx adds the
    couple field w_i (m x r_i) with m = J_w^{-1}(rho x F), J_w =
    diag(4, 4, 8): f_i = (Fx/4, 0, -Fx x_i / 4). Totals must be exact:
    sum f = F, sum r x f about the centroid = rho x F."""
    model = _starter_only(make_deck, "R3F", RBE3_SQUARE)
    r3 = Rbe3Constraint(model.rbe3[0], model, MessageLog())
    ref = model.node_index(9)
    masters = model.node_indices([1, 2, 3, 4])

    # central vertical force: pure translation split
    f = np.zeros_like(model.x)
    f[ref] = [0.0, 0.0, 8.0]
    z = np.zeros_like(f)
    r3.transfer_forces(f, z, z.copy(), np.zeros_like(f), model.x)
    assert f[masters] == pytest.approx(
        np.tile([0.0, 0.0, 2.0], (4, 1)))
    assert np.abs(f[ref]).max() == 0.0

    # transverse force: translation + couple, against the hand formula
    f = np.zeros_like(model.x)
    Fx = 4.0
    f[ref] = [Fx, 0.0, 0.0]
    r3.transfer_forces(f, z, z.copy(), np.zeros_like(f), model.x)
    xm = model.x0[masters]
    expect = np.column_stack([np.full(4, Fx / 4), np.zeros(4),
                              -Fx * xm[:, 0] / 4.0])
    assert f[masters] == pytest.approx(expect)
    # exact totals: force and moment about the centroid
    assert f.sum(axis=0) == pytest.approx([Fx, 0.0, 0.0])
    M = np.cross(model.x0[masters], f[masters]).sum(axis=0)
    assert M == pytest.approx(np.cross([0, 0, 1.0], [Fx, 0, 0]))


def test_rbe3_velocity_interpolation_dual(make_deck):
    """enforce() must reproduce a rigid master motion exactly at the
    dependent node (the interpolation is the exact rigid fit when the
    masters move rigidly), and transmit power without loss (duality)."""
    model = _starter_only(make_deck, "R3V", RBE3_SQUARE)
    r3 = Rbe3Constraint(model.rbe3[0], model, MessageLog())
    ref = model.node_index(9)
    masters = model.node_indices([1, 2, 3, 4])

    v = np.zeros_like(model.x)
    vr = np.zeros_like(model.x)
    w_rig = np.array([0.3, -0.2, 0.5])
    v_g = np.array([1.0, 2.0, -0.5])
    v[masters] = v_g + np.cross(w_rig, model.x0[masters])
    x = model.x0.copy()
    r3.enforce(x, v, vr, dt=0.0)
    assert v[ref] == pytest.approx(
        v_g + np.cross(w_rig, model.x0[ref]))
    assert vr[ref] == pytest.approx(w_rig)

    # duality: sum f_i . v_i == F . v_ref for any F
    rng = np.random.default_rng(7)
    F = rng.normal(size=3)
    f = np.zeros_like(model.x)
    f[ref] = F
    z = np.zeros_like(f)
    r3.transfer_forces(f, z, z.copy(), np.zeros_like(f), model.x0.copy())
    assert float(np.einsum("nb,nb->", f[masters], v[masters])) == \
        pytest.approx(float(F @ v[ref]))


# ============================================================================
# Unit tests — sphere / cylinder wall projection (rgwals/rgwalc)
# ============================================================================

def test_rwall_sphere_cylinder_projection():
    """Signed distance + normal closed forms."""
    sph = RigidWall(id=1, point=np.array([1.0, 2.0, 3.0]),
                    normal=np.array([0.0, 0.0, 1.0]), geom="SPHER",
                    radius=2.0)
    xc = np.array([[4.0, 2.0, 3.0], [1.0, 2.0, 0.0]])
    s, n = RigidWalls._geometry(sph, sph.point, xc)
    assert s == pytest.approx([1.0, 1.0])
    assert n == pytest.approx(
        np.array([[1.0, 0.0, 0.0], [0.0, 0.0, -1.0]]))

    cyl = RigidWall(id=2, point=np.zeros(3),
                    normal=np.array([0.0, 0.0, 1.0]), geom="CYL",
                    radius=5.0)
    xc = np.array([[3.0, 4.0, 7.0], [6.0, 0.0, -2.0]])
    s, n = RigidWalls._geometry(cyl, cyl.point, xc)
    assert s == pytest.approx([0.0, 1.0])          # first ON the surface
    assert n == pytest.approx(
        np.array([[0.6, 0.8, 0.0], [1.0, 0.0, 0.0]]))


# ============================================================================
# Unit tests — /PLOAD lumping, /IMPDISP landing
# ============================================================================

def test_pload_closed_form(make_deck):
    """Unit pressure on a 2x3 quad (normal +z by node order): each corner
    gets p*A/4 along +z. A triangle segment gets thirds."""
    starter = (
        "/BEGIN\npload probe\n"
        "/NODE\n"
        "1 0 0 0\n2 2 0 0\n3 2 3 0\n4 0 3 0\n"
        "11 5 0 0\n12 7 0 0\n13 5 4 0\n"
        "/SHELL/1\n1 1 2 3 4\n"
        "/SH3N/1\n2 11 12 13\n"
        "/PART/1\nplate\n1 1\n" + STEEL_LAW1 +
        "/PROP/SHELL/1\nplate\n1 0 0 0\n0.01 0.01 0.01 0 0\n3 0 1.0\n"
        "/SURF/SEG/1\nquad\n1 2 3 4\n"
        "/SURF/SEG/2\ntri\n11 12 13\n"
        "/FUNCT/1\nconst\n0.0 1.0\n10.0 1.0\n"
        "/PLOAD/1\nquad p\n1 1 0.5\n"
        "/PLOAD/2\ntri p\n2 1 0.5\n"
        "/END\n"
    )
    model = _starter_only(make_deck, "PLD", starter)
    loads = LoadsAndConstraints(model, MessageLog())
    fext = np.zeros_like(model.x)
    loads.external_forces(0.0, fext, model.x)
    quad = model.node_indices([1, 2, 3, 4])
    tri = model.node_indices([11, 12, 13])
    # p = 0.5, quad A = 6 -> 0.75 per corner; tri A = 4 -> 2/3 per corner
    assert fext[quad] == pytest.approx(
        np.tile([0.0, 0.0, 0.75], (4, 1)))
    assert fext[tri] == pytest.approx(
        np.tile([0.0, 0.0, 0.5 * 4.0 / 3.0], (3, 1)))


def test_impdisp_exact_landing(make_deck):
    """apply_kinematic must set the velocity so the node lands exactly at
    x0 + d(t_end), whatever its current position."""
    starter = (
        "/BEGIN\nimpdisp probe\n"
        "/NODE\n1 0 0 0\n2 1 0 0\n"
        "/TRUSS/1\n1 1 2\n"
        "/PART/1\nbar\n1 1\n" + STEEL_LAW1 +
        "/PROP/TRUSS/1\nbar\n1.0\n"
        "/GRNOD/NODE/1\nend\n2\n"
        "/FUNCT/1\nramp\n0.0 0.0\n1.0 1.0\n"
        "/IMPDISP/1\npull\n1 X 1 0.5\n"
        "/END\n"
    )
    model = _starter_only(make_deck, "IDP", starter)
    loads = LoadsAndConstraints(model, MessageLog())
    i2 = model.node_index(2)
    model.x[i2, 0] = 1.02                       # pretend it drifted
    v = np.zeros_like(model.x)
    dt = 0.05
    w = loads.apply_kinematic(0.4, v, np.zeros_like(v), model.mass,
                              model.x, dt)
    # target x = 1 + 0.5*f(0.4) = 1.2 -> v = (1.2 - 1.02)/0.05
    assert v[i2, 0] == pytest.approx((1.2 - 1.02) / dt)
    assert w != 0.0                              # constraint work is booked


# ============================================================================
# Analytic validations — rigid bodies
# ============================================================================

def test_rbody_tumbling_conserves_angular_momentum(make_deck):
    """A free 2x1x1 rigid bar spun about a NON-principal axis (wx + wz)
    tumbles. The stored angular momentum is updated (L += T dt with
    T = 0), so the momentum measured from the nodal velocities must stay
    on the initial value; the internal energy must stay EXACTLY zero
    (rigid-interior elements see a zero strain rate — the enforce()
    re-scatter); kinetic energy oscillates only within the leapfrog
    staggering; distances between body nodes are preserved to round-off."""
    starter = (
        "/BEGIN\ntumbling bar\n" + _bar_two_bricks() +
        "/RBODY/1\nrigid bar\n100 1 0 1\n"
        "/INIVEL/AXIS/1\nspin z\n50.0 Z 1 1.0 0.5 0.5\n"
        "/INIVEL/AXIS/2\nspin x\n20.0 X 1 1.0 0.5 0.5\n"
        "/END\n"
    )
    engine = "/RUN/TUM/1\n1.0\n/DT\n0.9 0\n/PRINT/-5000\n/STOP\n10.0\n"
    model, out = _run(make_deck, "TUM", starter, engine)

    real = model.mass < 1e29
    m = model.mass[real]
    xg = (m[:, None] * model.x[real]).sum(0) / m.sum()
    L = np.cross(model.x[real] - xg, m[:, None] * model.v[real]).sum(0)
    # initial L from the exact rigid field on the initial geometry
    r0 = model.x0[real] - np.array([1.0, 0.5, 0.5])
    v0 = np.cross([20.0, 0.0, 50.0], r0)
    L0 = np.cross(r0, m[:, None] * v0).sum(0)
    # |L| is conserved; the tiny residual is the O(w dt) staggering of the
    # measurement (w from the pre-rotation inertia, positions post), not
    # a drift — 9000+ cycles of violent tumbling stay at ~1e-3
    assert np.linalg.norm(L - L0) / np.linalg.norm(L0) < 5e-3
    # COG never moves (no force), rigid shape exact
    assert np.abs((m[:, None] * model.v[real]).sum(0)).max() < 1e-12
    d = np.linalg.norm(model.x[model.node_index(1)]
                       - model.x[model.node_index(9)])
    assert d == pytest.approx(2.0, abs=1e-9)
    # elements inside the body carry NO stress or internal energy
    assert np.abs(model.bricks.state["sig"]).max() < 1e-10
    assert abs(model.bricks.state["eint"].sum()) < 1e-12
    s = _final_summary(out)
    assert s["NORMAL"]
    assert abs(s["ERR"]) < 1.0


def test_rbody_free_fall_with_added_mass(make_deck):
    """/RBODY under gravity: rigid free fall — v = g t and the COG drop
    g t^2 / 2 exactly (the gather/scatter must not leak force), balance
    exact."""
    g, t_end = 10.0, 0.5
    starter = (
        "/BEGIN\nfalling body\n" + _bar_two_bricks() +
        "/RBODY/1\nrigid bar\n100 1 5.0e-6 1\n"
        "/FUNCT/1\nconst\n0.0 1.0\n10.0 1.0\n"
        f"/GRAV/1\ngravity\n1 Z 0 -{g}\n"
        "/END\n"
    )
    engine = f"/RUN/FALL/1\n{t_end}\n/DT\n0.9 0\n/PRINT/-5000\n/STOP\n5.0\n"
    model, out = _run(make_deck, "FALL", starter, engine)
    real = model.mass < 1e29
    # every body node (and the added-mass master) falls at exactly g t
    assert model.v[real, 2] == pytest.approx(-g * t_end)
    assert np.abs(model.v[real, :2]).max() < 1e-12
    drop = model.x0[real, 2] - model.x[real, 2]
    # leapfrog free fall from rest: sum over cycles of g*t_{k+1}*dt_k —
    # first-order in dt above g t^2/2; just require the few-permille match
    assert drop == pytest.approx(0.5 * g * t_end ** 2, rel=2e-3)
    s = _final_summary(out)
    assert s["NORMAL"]
    assert abs(s["ERR"]) < 0.2


def test_rbody_impvel_drive_books_work(make_deck):
    """A master /IMPVEL turns the body into a moving rigid die: the drive
    work must be booked so the balance closes while the body gains
    kinetic energy from nothing else."""
    starter = (
        "/BEGIN\ndriven body\n" + _bar_two_bricks() +
        "/RBODY/1\nrigid bar\n100 1 0 1\n"
        "/GRNOD/NODE/2\nmaster\n100\n"
        "/FUNCT/1\nramp\n0.0 0.0\n0.1 2.0\n10.0 2.0\n"
        "/IMPVEL/1\ndrive\n1 X 2 1.0\n"
        "/END\n"
    )
    engine = "/RUN/DRV/1\n0.5\n/DT\n0.9 0\n/PRINT/-5000\n/STOP\n5.0\n"
    model, out = _run(make_deck, "DRV", starter, engine)
    real = model.mass < 1e29
    assert model.v[real, 0] == pytest.approx(2.0)
    s = _final_summary(out)
    assert s["NORMAL"]
    # all kinetic energy came from the booked drive work
    ke = float(0.5 * (model.mass[real] * (model.v[real] ** 2).sum(1)).sum())
    assert s["EW"] == pytest.approx(ke, rel=2e-2)
    assert abs(s["ERR"]) < 2.0


def test_rbe2_pendulum_bottom_speed_and_energy(make_deck):
    """Physical pendulum: a rigid brick bar on an /RBE2 whose master node
    is the pivot (BCS 111 on translations, rotation about y free).
    Released horizontal under gravity, the closed-form bottom angular
    rate is  w_b = sqrt(2 M g d / J_p)  (energy conservation of the
    rigid-pendulum ODE, with M, d, J_p assembled from the same point
    masses the solver lumps). The maximum tip speed over the swing must
    hit w_b * r_tip, and the balance must close with CE = 0 (the pivot
    does no work)."""
    g = 100.0
    # bar from x=0.5 to 2.5, cross-section 0.5x0.5 centered on the x-axis
    nodes, bricks = [], []
    nid = 0
    for i in range(5):
        for (y, z) in ((-0.25, -0.25), (0.25, -0.25),
                       (0.25, 0.25), (-0.25, 0.25)):
            nid += 1
            nodes.append(f"{nid} {0.5 * (i + 1)} {y} {z}")
    for e in range(4):
        b = 4 * e
        bricks.append(f"{e + 1} {b + 1} {b + 2} {b + 3} {b + 4} "
                      f"{b + 5} {b + 6} {b + 7} {b + 8}")
    starter = (
        "/BEGIN\nrbe2 pendulum\n"
        "/NODE\n" + "\n".join(nodes) + "\n999 0 0 0\n"
        "/BRICK/1\n" + "\n".join(bricks) + "\n"
        "/PART/1\nbar\n1 1\n" + STEEL_LAW1 +
        "/PROP/SOLID/1\nsolid\n1.1 0.05 0.1\n"
        "/GRNOD/PART/1\nbar nodes\n1\n"
        "/GRNOD/NODE/2\npivot\n999\n"
        "/RBE2/1\npendulum link\n999 1\n"
        "/BCS/1\npivot: rotation about y only\n111 101 0 2\n"
        "/FUNCT/1\nconst\n0.0 1.0\n100.0 1.0\n"
        f"/GRAV/1\ngravity\n1 Z 0 -{g}\n"
        "/TH/NODE/1\ntip\nVX VY VZ\n17\n"
        "/END\n"
    )
    engine = ("/RUN/PEND/1\n0.5\n/DT\n0.9 0\n/PRINT/-20000\n/STOP\n5.0\n"
              "/TFILE\n0.001\n")
    model, out = _run(make_deck, "PEND", starter, engine)

    # closed form from the solver's own point masses
    real = model.mass < 1e29
    m = model.mass[real]
    x0 = model.x0[real]
    M = m.sum()
    d = float((m * x0[:, 0]).sum() / M)            # COG distance to pivot
    Jp = float((m * (x0[:, 0] ** 2 + x0[:, 2] ** 2)).sum())  # about pivot y
    w_b = np.sqrt(2.0 * M * g * d / Jp)
    r_tip = 2.5                                    # node 17 at (2.5,-.25,-.25)
    r_tip_y = np.sqrt(2.5 ** 2 + 0.25 ** 2)        # arm about the y axis
    th = _read_th(out.replace("_0001.out", "T01.csv"))
    speed = np.sqrt(th["N17_VX"] ** 2 + th["N17_VZ"] ** 2)
    assert speed.max() == pytest.approx(w_b * r_tip_y, rel=2e-2)
    # the pivot is fixed and the swing is planar
    assert np.abs(model.x[model.node_index(999)]).max() < 1e-12
    assert np.abs(th["N17_VY"]).max() < 1e-9
    s = _final_summary(out)
    assert s["NORMAL"]
    assert s["CE"] == 0.0
    assert abs(s["ERR"]) < 1.0


def test_rbody_contact_impact_dt09(make_deck):
    """Coexistence with contact at /DT 0.9: a rigid 2x2x2-meshed cube
    (/RBODY over a brick part) falls onto a clamped deformable plate
    through /INTER/TYPE7. The body must stay exactly rigid, not cross the
    plate, bounce (or rest), and the long run must keep a tight balance —
    the contact springs push against kinematically driven nodes, so this
    exercises the interface dt bound and the midstep energy booking
    against a rigid partner."""
    # 2x2x2 cube of bricks above a 6x6 shell plate
    nodes, bricks = [], []
    def nid(i, j, k):
        return 100 + i + 3 * (j + 3 * k)
    for k in range(3):
        for j in range(3):
            for i in range(3):
                nodes.append(f"{nid(i, j, k)} {2.0 + i} {2.0 + j} "
                             f"{0.6 + k}")
    eid = 0
    for k in range(2):
        for j in range(2):
            for i in range(2):
                eid += 1
                c = [nid(i, j, k), nid(i + 1, j, k), nid(i + 1, j + 1, k),
                     nid(i, j + 1, k), nid(i, j, k + 1),
                     nid(i + 1, j, k + 1), nid(i + 1, j + 1, k + 1),
                     nid(i, j + 1, k + 1)]
                bricks.append(f"{eid} " + " ".join(str(v) for v in c))
    pn, shells = [], []
    for j in range(7):
        for i in range(7):
            pn.append(f"{1 + i + 7 * j} {i * 1.0} {j * 1.0} 0.0")
    for j in range(6):
        for i in range(6):
            n1 = 1 + i + 7 * j
            shells.append(f"{200 + i + 6 * j} {n1} {n1 + 1} {n1 + 8} "
                          f"{n1 + 7}")
    border = [1 + i for i in range(7)] + [43 + i for i in range(7)] + \
             [1 + 7 * j for j in range(1, 6)] + [7 + 7 * j
                                                 for j in range(1, 6)]
    starter = (
        "/BEGIN\nrigid impactor\n"
        "/NODE\n" + "\n".join(pn + nodes) + "\n999 3 3 30\n"
        "/SHELL/1\n" + "\n".join(shells) + "\n"
        "/BRICK/2\n" + "\n".join(bricks) + "\n"
        "/PART/1\nplate\n1 1\n"
        "/PART/2\nimpactor\n2 1\n" + STEEL_LAW1 +
        "/PROP/SHELL/1\nplate t=0.2\n1 0 0 0\n0.01 0.01 0.01 0 0\n"
        "3 0 0.2\n"
        "/PROP/SOLID/2\nsolid\n1.1 0.05 0.1\n"
        "/GRNOD/PART/2\nimpactor nodes\n2\n"
        f"/GRNOD/NODE/3\nborder\n{' '.join(str(v) for v in border)}\n"
        "/BCS/1\nclamp border\n111 111 0 3\n"
        "/RBODY/1\nrigid impactor\n999 2 0 1\n"
        "/INIVEL/TRA/1\nfall\n0 0 -1.0 2\n"
        "/SURF/PART/1\nplate surf\n1\n"
        "/INTER/TYPE7/1\nimpact\n2 1 2 1\n1.0 0.0 0.0\n"
        "/END\n"
    )
    engine = "/RUN/RIMP/1\n3.0\n/DT\n0.9 0\n/PRINT/-100000\n/STOP\n14.9\n"
    model, out = _run(make_deck, "RIMP", starter, engine)

    assert np.all(np.isfinite(model.v))
    # the body stayed exactly rigid through the impact
    a = model.node_index(nid(0, 0, 0))
    b = model.node_index(nid(2, 2, 2))
    assert np.linalg.norm(model.x[a] - model.x[b]) == \
        pytest.approx(np.sqrt(12.0), abs=1e-9)
    assert np.abs(model.bricks.state["sig"]).max() < 1e-8
    # the impactor was arrested / bounced, never crossing the plate
    bottom = [model.node_index(nid(i, j, 0))
              for j in range(3) for i in range(3)]
    assert model.x[bottom, 2].min() > -0.2
    assert model.v[bottom, 2].mean() > -1.0
    s = _final_summary(out)
    assert s["NORMAL"]
    assert abs(s["ERR"]) < 5.0


def test_rbe3_no_stiffening_rigid_flight(make_deck):
    """/RBE3 with a loaded dependent node on a FREE plate: a central
    force on the reference node must accelerate the whole assembly with
    the exact total momentum and NO deformation (the constraint
    distributes force without stiffening; a rigid link would also carry
    no stress here, but the RBE3 additionally must not fight the plate's
    own inertia distribution — symmetric setup: uniform acceleration)."""
    starter = (
        "/BEGIN\nrbe3 push\n"
        "/NODE\n"
        "1 -1 -1 0\n2 1 -1 0\n3 1 1 0\n4 -1 1 0\n"
        "9 0 0 1\n"
        "/SHELL/1\n1 1 2 3 4\n"
        "/PART/1\nplate\n1 1\n" + STEEL_LAW1 +
        "/PROP/SHELL/1\nplate\n1 0 0 0\n0.01 0.01 0.01 0 0\n3 0 1.0\n"
        "/GRNOD/NODE/1\nmasters\n1 2 3 4\n"
        "/GRNOD/NODE/9\nref\n9\n"
        "/ADMAS/1\nref mass\n1.0e-5 9\n"
        "/RBE3/1\nhang\n9 1\n"
        "/FUNCT/1\nconst\n0.0 1.0\n10.0 1.0\n"
        "/CLOAD/1\npush z\n1 Z 9 1.0e-3\n"
        "/END\n"
    )
    engine = "/RUN/R3P/1\n1.0\n/DT\n0.9 0\n/PRINT/-5000\n/STOP\n5.0\n"
    model, out = _run(make_deck, "R3P", starter, engine)
    real = model.mass < 1e29
    pz = float((model.mass[real] * model.v[real, 2]).sum())
    assert pz == pytest.approx(1.0e-3 * 1.0, rel=1e-6)   # exact impulse
    # uniform acceleration: everything moves together, no stress
    assert model.v[real, 2].std() < 1e-8
    assert np.abs(model.shells.state["sig"]).max() < 1e-12
    s = _final_summary(out)
    assert s["NORMAL"]
    assert s["CE"] == 0.0
    assert abs(s["ERR"]) < 0.5


# ============================================================================
# Analytic validations — moving / curved rigid walls
# ============================================================================

def _meshed_cube_deck(n=2, origin=(0.0, 0.0, 0.0)):
    """A 2x2x2-meshed unit cube (nodes 1..27) + a wall carrier node 99
    half a unit below the cube base center. Meshed, not a single brick:
    a single 1-point brick hammered on a whole face stores the impact in
    barely-resolved modes whose exact-dt factor is precomputed on the
    undistorted geometry — the documented M1/M2 limitation the M4
    contact tests already worked around the same way."""
    h = 1.0 / n
    ox, oy, oz = origin
    nodes, bricks = [], []
    def nid(i, j, k):
        return 1 + i + (n + 1) * (j + (n + 1) * k)
    for k in range(n + 1):
        for j in range(n + 1):
            for i in range(n + 1):
                nodes.append(f"{nid(i,j,k)} {ox+i*h} {oy+j*h} {oz+k*h}")
    eid = 0
    for k in range(n):
        for j in range(n):
            for i in range(n):
                eid += 1
                c = [nid(i,j,k), nid(i+1,j,k), nid(i+1,j+1,k), nid(i,j+1,k),
                     nid(i,j,k+1), nid(i+1,j,k+1), nid(i+1,j+1,k+1),
                     nid(i,j+1,k+1)]
                bricks.append(f"{eid} " + " ".join(str(v) for v in c))
    return (
        "/NODE\n" + "\n".join(nodes)
        + f"\n99 {ox+0.5} {oy+0.5} {oz-0.5}\n"
        "/BRICK/1\n" + "\n".join(bricks) + "\n"
        "/PART/1\ncube\n1 1\n" + STEEL_LAW1 +
        "/PROP/SOLID/1\nsolid\n1.1 0.05 0.1\n"
        "/GRNOD/NODE/9\nwall node\n99\n"
        "/GRNOD/PART/1\ncube nodes\n1\n"
    ), (n + 1) ** 3


def test_moving_wall_with_mass_momentum_and_energy(make_deck):
    """A free plane wall (carrier node with /ADMAS + /INIVEL) flies up
    into a resting elastic cube. Momentum must be conserved EXACTLY
    (wall + cube form a closed system: the joint implicit solve of
    rigid_wall.py exchanges the impulses), the cube must end above the
    wall, and the balance must close around the impact. The run ends
    shortly after separation: the post-impact ringing of the coarse cube
    then drifts the ledger through a PRE-M5 element issue (the
    half-step-lagged bulk-viscosity work under barely-resolved ringing —
    see the guide's known-issues note), which is not what this test
    guards."""
    cube_deck, nn = _meshed_cube_deck()
    m_wall = 4.0 * RHO
    starter = (
        "/BEGIN\nmoving wall impact\n" + cube_deck +
        f"/ADMAS/1\nwall inertia\n{m_wall} 9\n"
        "/INIVEL/TRA/1\nwall up\n0 0 1.0 9\n"
        "/RWALL/PLANE/1\nmoving floor\n0 0 0 0 99\n"
        "0.5 0.5 -0.5\n"
        "0.5 0.5 0.5\n"
        "/END\n"
    )
    engine = "/RUN/MWI/1\n0.65\n/DT\n0.9 0\n/PRINT/-50000\n/STOP\n10.0\n"
    model, out = _run(make_deck, "MWI", starter, engine)

    iw = model.node_index(99)
    real = model.mass < 1e29
    p = float((model.mass[real] * model.v[real, 2]).sum())
    assert p == pytest.approx(m_wall * 1.0, rel=1e-9)    # exact momentum
    # the cube took off upward, the wall slowed down
    cube = np.arange(nn)
    assert model.v[cube, 2].mean() > 0.5
    assert model.v[iw, 2] < 1.0
    # no node ended below the (moved) wall plane, which translated with
    # its carrier node (plane point initially AT the node)
    assert model.x[cube, 2].min() >= model.x[iw, 2] - 1e-9
    s = _final_summary(out)
    assert s["NORMAL"]
    # ~5% at /DT 0.9: the shock of a full-speed kinematic arrest resolved
    # over a handful of cycles (first-order in dt — the same run at
    # /DT 0.45 halves it; the closed-form single-node test below checks
    # the booking itself exactly)
    assert abs(s["ERR"]) < 6.0


def test_moving_wall_single_node_exact_inelastic(make_deck):
    """The joint wall/node solve against the closed form: a heavy wall
    node catching one point mass must produce the perfectly-inelastic
    common velocity m_w v_w / (m_w + m) in the very cycle of impact —
    the implicit treatment's defining property (an explicit lagged
    recoil converges to it only as dt -> 0). The point mass is a
    /ADMAS-ballasted node of a deliberately feeble truss (the element
    only exists to give the model a mesh and a time step; its spring
    impulse over the run is ~1e-4 of the momenta)."""
    m_wall = 4.0 * RHO
    m_node = 1.0e-6
    starter = (
        "/BEGIN\ninelastic probe\n"
        "/NODE\n1 0 0 0.05\n2 0 0 40.0\n99 0 0 0\n"
        "/TRUSS/1\n1 1 2\n"
        "/PART/1\nbar\n1 1\n" + STEEL_LAW1 +
        "/PROP/TRUSS/1\nbar\n1.0e-7\n"
        "/GRNOD/NODE/8\nballast\n1\n"
        "/GRNOD/NODE/9\nwall\n99\n"
        f"/ADMAS/1\nnode ballast\n{m_node} 8\n"
        f"/ADMAS/2\nwall m\n{m_wall} 9\n"
        "/INIVEL/TRA/1\nup\n0 0 1.0 9\n"
        "/RWALL/PLANE/1\nfloor\n0 0 0 0 99\n"
        "0 0 0\n"
        "0 0 1\n"
        "/END\n"
    )
    engine = "/RUN/INE/1\n0.1\n/DT\n0.9 0\n/PRINT/-50000\n/STOP\n50.0\n"
    model, out = _run(make_deck, "INE", starter, engine)
    i1 = model.node_index(1)
    iw = model.node_index(99)
    m1 = model.mass[i1]
    v_star = m_wall * 1.0 / (m_wall + m1)   # inelastic common velocity
    # both ride together at the common velocity
    assert model.v[i1, 2] == pytest.approx(v_star, rel=1e-2)
    assert model.v[iw, 2] == pytest.approx(v_star, rel=1e-2)
    s = _final_summary(out)
    assert s["NORMAL"]
    assert abs(s["ERR"]) < 1.0


def test_driven_wall_push_books_external_work(make_deck):
    """An /IMPVEL-driven wall kicks the resting cube: the reaction is
    absorbed by the drive and the pushed-in energy enters the ledger as
    EXTERNAL WORK — the cube leaves with all its energy accounted for.
    (An elastic cube kicked by a kinematic wall at V rebounds somewhere
    between V and 2V — the wall arrest is inelastic per NODE, not for
    the body — so the checks are the momentum direction, no penetration,
    and above all the closed balance shortly after separation.)"""
    cube_deck, nn = _meshed_cube_deck()
    starter = (
        "/BEGIN\ndriven wall push\n" + cube_deck +
        "/FUNCT/1\nconst v\n0.0 1.0\n100.0 1.0\n"
        "/IMPVEL/1\ndrive wall\n1 Z 9 0.5\n"
        "/RWALL/PLANE/1\ndriven floor\n0 0 0 0 99\n"
        "0.5 0.5 -0.5\n"
        "0.5 0.5 0.5\n"
        "/END\n"
    )
    engine = "/RUN/DWP/1\n1.6\n/DT\n0.9 0\n/PRINT/-50000\n/STOP\n10.0\n"
    model, out = _run(make_deck, "DWP", starter, engine)
    iw = model.node_index(99)
    cube = np.arange(nn)
    # the cube was launched upward, between riding (V) and elastic (2V)
    v_mean = model.v[cube, 2].mean()
    assert 0.4 < v_mean < 1.1
    # never behind the wall (which translated with its driven node)
    assert model.x[cube, 2].min() >= model.x[iw, 2] - 1e-9
    s = _final_summary(out)
    assert s["NORMAL"]
    # everything the cube carries (plus the arrest dissipation CE and the
    # numerical dissipation EN the kick's barely-resolved chatter feeds —
    # measured exactly by the M6 internal-work ledger) was booked as the
    # drive's work: EW = IE + KE + CE + EN at balance
    real = model.mass < 1e29
    ke = float(0.5 * (model.mass[real] * (model.v[real] ** 2).sum(1)).sum())
    ie = sum(float(g.state["eint"].sum()) for _, g in model.element_groups())
    assert s["EW"] == pytest.approx(ke + ie + s["CE"] + s["EN"], rel=0.07)
    assert abs(s["ERR"]) < 5.0


def test_sphere_wall_drop(make_deck):
    """A (meshed) cube dropped onto a fixed spherical wall: every
    candidate node must stay outside the sphere (the per-node radial
    projection), and the kinematic arrest must book cleanly over the
    impact window."""
    cube_deck, nn = _meshed_cube_deck(origin=(-0.5, -0.5, 2.0))
    starter = (
        "/BEGIN\nsphere drop\n" + cube_deck +
        "/INIVEL/TRA/1\nfall\n0 0 -0.5 1\n"
        "/RWALL/SPHER/1\nball\n1 0 0 0\n"
        "0.0 0.0 0.0\n"
        "1.5\n"
        "/END\n"
    )
    engine = "/RUN/SPH/1\n1.4\n/DT\n0.9 0\n/PRINT/-50000\n/STOP\n10.0\n"
    model, out = _run(make_deck, "SPH", starter, engine)
    d = np.linalg.norm(model.x[:nn], axis=1)
    assert d.min() >= 1.5 - 1e-6
    s = _final_summary(out)
    assert s["NORMAL"]
    assert abs(s["ERR"]) < 5.0


def test_cylinder_wall_drop(make_deck):
    """Same against a cylinder along y: nodes stay outside the radius
    measured from the axis."""
    cube_deck, nn = _meshed_cube_deck(origin=(-0.5, -0.5, 2.0))
    starter = (
        "/BEGIN\ncylinder drop\n" + cube_deck +
        "/INIVEL/TRA/1\nfall\n0 0 -0.5 1\n"
        "/RWALL/CYL/1\nroller\n1 0 0 0\n"
        "0.0 0.0 0.0\n"
        "0.0 1.0 0.0\n"
        "1.5\n"
        "/END\n"
    )
    engine = "/RUN/CYL/1\n1.4\n/DT\n0.9 0\n/PRINT/-50000\n/STOP\n10.0\n"
    model, out = _run(make_deck, "CYL", starter, engine)
    d = np.sqrt(model.x[:nn, 0] ** 2 + model.x[:nn, 2] ** 2)
    assert d.min() >= 1.5 - 1e-6
    s = _final_summary(out)
    assert s["NORMAL"]
    assert abs(s["ERR"]) < 5.0


# ============================================================================
# Analytic validations — /SECT, /PLOAD, /IMPDISP
# ============================================================================

def test_sect_recovers_applied_load(make_deck):
    """A clamped bar of 10 bricks pulled by a slowly ramped end /CLOAD:
    quasi-statically, the section at mid-span must transmit exactly the
    applied load (sign: the clamped side PULLS the loaded side back, so
    FX = -F_applied), with zero moment about the axis-symmetric section.
    The side node set is built the natural way, with a /GRNOD/BOX."""
    F_node = 0.25e-3
    nodes, bricks = [], []
    nid = 0
    for i in range(11):
        for (y, z) in ((0, 0), (1, 0), (1, 1), (0, 1)):
            nid += 1
            nodes.append(f"{nid} {float(i)} {float(y)} {float(z)}")
    for e in range(10):
        b = 4 * e
        bricks.append(f"{e + 1} {b + 1} {b + 2} {b + 3} {b + 4} "
                      f"{b + 5} {b + 6} {b + 7} {b + 8}")
    end = " ".join(str(40 + k) for k in (1, 2, 3, 4))
    starter = (
        "/BEGIN\nsection bar\n"
        "/NODE\n" + "\n".join(nodes) + "\n"
        "/BRICK/1\n" + "\n".join(bricks) + "\n"
        "/PART/1\nbar\n1 1\n" + STEEL_LAW1 +
        "/PROP/SOLID/1\nsolid\n1.1 0.05 0.1\n"
        "/GRNOD/NODE/1\nroot\n1 2 3 4\n"
        f"/GRNOD/NODE/2\nend\n{end}\n"
        "/BOX/RECTA/1\nouter half\n"
        "4.5 -1.0 -1.0\n"
        "11.5 2.0 2.0\n"
        "/GRNOD/BOX/3\nside set\n1\n"
        "/BCS/1\nclamp root\n111 111 0 1\n"
        "/FUNCT/1\nslow ramp\n0.0 0.0\n0.2 1.0\n10.0 1.0\n"
        "/CLOAD/1\npull\n1 X 2 " + f"{F_node}\n"
        "/SECT/1\nmid section\n3\n"
        "/TH/SECT/1\nsection\nDEF\n1\n"
        "/END\n"
    )
    engine = ("/RUN/SEC/1\n0.5\n/DT\n0.9 0\n/PRINT/-50000\n/STOP\n10.0\n"
              "/TFILE\n0.002\n")
    model, out = _run(make_deck, "SEC", starter, engine)
    th = _read_th(out.replace("_0001.out", "T01.csv"))
    F_total = 4 * F_node
    # quasi-static tail: average the held part of the ramp
    tail = th["TIME"] > 0.3
    assert th["S1_FX"][tail].mean() == pytest.approx(-F_total, rel=0.05)
    # axial load through a symmetric section: no moment, no shear
    assert np.abs(th["S1_FY"][tail]).max() < 0.05 * F_total
    assert np.abs(th["S1_FZ"][tail]).max() < 0.05 * F_total
    for c in ("S1_MX", "S1_MY", "S1_MZ"):
        assert np.abs(th[c][tail]).max() < 0.1 * F_total
    s = _final_summary(out)
    assert s["NORMAL"]
    assert abs(s["ERR"]) < 2.0


def test_pload_free_plate_momentum(make_deck):
    """A free flat shell under constant pressure: the follower resultant
    is p*A along the (constant) normal, so the momentum after t is
    exactly p*A*t and the balance closes to round-off (leapfrog work of
    a constant force)."""
    p, t_end = 1.0e-4, 1.0
    starter = (
        "/BEGIN\npressure plate\n"
        "/NODE\n"
        "1 0 0 0\n2 10 0 0\n3 10 10 0\n4 0 10 0\n"
        "/SHELL/1\n1 1 2 3 4\n"
        "/PART/1\nplate\n1 1\n" + STEEL_LAW1 +
        "/PROP/SHELL/1\nplate\n1 0 0 0\n0.01 0.01 0.01 0 0\n3 0 1.0\n"
        "/SURF/PART/1\nplate surf\n1\n"
        "/FUNCT/1\nconst\n0.0 1.0\n10.0 1.0\n"
        f"/PLOAD/1\npush\n1 1 {p}\n"
        "/END\n"
    )
    engine = f"/RUN/PLP/1\n{t_end}\n/DT\n0.9 0\n/PRINT/-5000\n/STOP\n5.0\n"
    model, out = _run(make_deck, "PLP", starter, engine)
    real = model.mass < 1e29
    pz = float((model.mass[real] * model.v[real, 2]).sum())
    assert pz == pytest.approx(p * 100.0 * t_end, rel=1e-9)
    # uniform load on uniform mass: pure translation, no deformation
    assert np.abs(model.shells.state["sig"]).max() < 1e-12
    s = _final_summary(out)
    assert s["NORMAL"]
    # the small residual is the classic leapfrog half-step offset (KE is
    # sampled at v^{n+1/2}, the work at x^n): O(dt / t_end), not a leak
    assert abs(s["ERR"]) < 0.5


def test_impdisp_quasi_static_pull(make_deck):
    """A clamped 4-brick bar stretched by /IMPDISP with a slow ramp: the
    end nodes land EXACTLY at x0 + d(t_end) (position-level enforcement),
    the stored energy matches the quasi-static E A d^2 / 2 L, and the
    balance closes — the /IMPDISP work booking mirrors /IMPVEL's."""
    d_end = 4.0e-3
    nodes, bricks = [], []
    nid = 0
    for i in range(5):
        for (y, z) in ((0, 0), (1, 0), (1, 1), (0, 1)):
            nid += 1
            nodes.append(f"{nid} {float(i)} {float(y)} {float(z)}")
    for e in range(4):
        b = 4 * e
        bricks.append(f"{e + 1} {b + 1} {b + 2} {b + 3} {b + 4} "
                      f"{b + 5} {b + 6} {b + 7} {b + 8}")
    starter = (
        "/BEGIN\nimpdisp bar\n"
        "/NODE\n" + "\n".join(nodes) + "\n"
        "/BRICK/1\n" + "\n".join(bricks) + "\n"
        "/PART/1\nbar\n1 1\n" + STEEL_LAW1 +
        "/PROP/SOLID/1\nsolid\n1.1 0.05 0.1\n"
        "/GRNOD/NODE/1\nroot\n1 2 3 4\n"
        "/GRNOD/NODE/2\nend\n17 18 19 20\n"
        "/BCS/1\nclamp root\n111 111 0 1\n"
        "/BCS/2\nuniaxial end\n011 111 0 2\n"
        "/FUNCT/1\nramp then hold\n0.0 0.0\n0.4 1.0\n10.0 1.0\n"
        f"/IMPDISP/1\nstretch\n1 X 2 {d_end}\n"
        "/END\n"
    )
    engine = "/RUN/IMD/1\n0.5\n/DT\n0.9 0\n/PRINT/-50000\n/STOP\n10.0\n"
    model, out = _run(make_deck, "IMD", starter, engine)
    end = model.node_indices([17, 18, 19, 20])
    # exact landing (this is the point of the position-level enforcement)
    assert model.x[end, 0] == pytest.approx(4.0 + d_end, abs=1e-12)
    s = _final_summary(out)
    assert s["NORMAL"]
    assert abs(s["ERR"]) < 2.0
    # quasi-static stored energy ~ E A d^2 / 2 L (uniaxial STRESS is not
    # the state here — lateral faces free but 3-D core: allow a few %)
    ie = sum(float(g.state["eint"].sum())
             for _, g in model.element_groups())
    k = 210.0 * 1.0 / 4.0
    assert ie == pytest.approx(0.5 * k * d_end ** 2, rel=0.10)


def test_impdisp_matches_impvel(make_deck):
    """The same motion prescribed as /IMPDISP (ramp displacement) and as
    /IMPVEL (the derivative) must land within a time step of each other
    and book the same external work within tolerance."""
    base = (
        "/NODE\n"
        "1 0 0 0\n2 1 0 0\n3 1 1 0\n4 0 1 0\n"
        "5 0 0 1\n6 1 0 1\n7 1 1 1\n8 0 1 1\n"
        "/BRICK/1\n1 1 2 3 4 5 6 7 8\n"
        "/PART/1\ncube\n1 1\n" + STEEL_LAW1 +
        "/PROP/SOLID/1\nsolid\n1.1 0.05 0.1\n"
        "/GRNOD/NODE/1\nroot\n1 2 3 4\n"
        "/GRNOD/NODE/2\ntop\n5 6 7 8\n"
        "/BCS/1\nclamp\n111 111 0 1\n"
    )
    d_end, t_ramp = 1.0e-3, 0.2
    dep = (
        "/BEGIN\nimpdisp cube\n" + base +
        f"/FUNCT/1\nramp\n0.0 0.0\n{t_ramp} 1.0\n10.0 1.0\n"
        f"/IMPDISP/1\npull\n1 Z 2 {d_end}\n/END\n"
    )
    vel = (
        "/BEGIN\nimpvel cube\n" + base +
        f"/FUNCT/1\nstep\n0.0 {d_end / t_ramp}\n{t_ramp} "
        f"{d_end / t_ramp}\n{t_ramp + 1e-4} 0.0\n10.0 0.0\n"
        "/IMPVEL/1\npull\n1 Z 2 1.0\n/END\n"
    )
    engine = "/RUN/CMP/1\n0.3\n/DT\n0.9 0\n/PRINT/-50000\n/STOP\n10.0\n"
    m1, o1 = _run(make_deck, "CMPD", dep, engine)
    m2, o2 = _run(make_deck, "CMPV", vel, engine)
    top = m1.node_indices([5, 6, 7, 8])
    assert m1.x[top, 2] == pytest.approx(1.0 + d_end, abs=1e-12)
    assert m2.x[top, 2] == pytest.approx(1.0 + d_end, rel=1e-3)
    s1, s2 = _final_summary(o1), _final_summary(o2)
    assert s1["EW"] == pytest.approx(s2["EW"], rel=0.05)
    assert abs(s1["ERR"]) < 2.0 and abs(s2["ERR"]) < 2.0
