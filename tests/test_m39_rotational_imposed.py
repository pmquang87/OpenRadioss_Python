"""M39 rotational imposed conditions: /IMPVEL and /IMPDISP with a
rotational direction (XX/YY/ZZ) now drive the ANGULAR velocity instead of
being parsed-and-discarded.

Before M39 the reader logged ``rotational direction XX not ported —
condition ignored`` and dropped the condition, which left every official
RD-E-1000 Bending deck — whose SOLE load is an ``/IMPVEL/XX`` on the
/RBODY master that rolls a strip into a circle — completely undriven (the
port's energy channels came out identically zero, deviating 1.0 against
the Fortran solver). These tests pin the three layers of the fix:

* the reader maps XX/YY/ZZ to dof 3/4/5 (the 6-DOF convention the /MPC and
  implicit dofmap already use) instead of discarding them;
* ``kinematics.apply_kinematic`` applies a nodal rotational condition to
  ``vr`` and books its work against the rotational INERTIA at the leapfrog
  midstep — the exact angular analogue of the translational branch;
* ``rigid_body.RigidBodyEngine`` drives the body SPIN from a rotational
  /IMPVEL on the master, so a spun rigid body carries its slaves.

Layered like the rest of the suite: reader unit tests, an apply-level
unit test against the closed-form midstep identity, and a full
starter+engine analytic validation whose energy balance must close.
"""

import contextlib
import io
import re

import numpy as np
import pytest

from pyradioss.common.messages import MessageLog
from pyradioss.engine.engine import run_engine
from pyradioss.engine.kinematics import LoadsAndConstraints
from pyradioss.starter.starter import run_starter

STEEL_LAW1 = """\
/MAT/LAW1/1
steel elastic
7.8e-6
210. 0.3
"""


def _starter_only(make_deck, name, starter):
    s, _ = make_deck(name, starter, "/RUN/X/1\n1.0\n")
    with contextlib.redirect_stdout(io.StringIO()):
        return run_starter(s)


def _run(make_deck, name, starter, engine):
    s, e = make_deck(name, starter, engine)
    with contextlib.redirect_stdout(io.StringIO()):
        run_starter(s)
        model = run_engine(e)
    return model, e.replace("_0001.rad", "_0001.out")


def _final_summary(out_path):
    text = open(out_path).read()

    def g(label):
        return float(re.search(label + r"\s*\.[ .]*:\s*([-\d.Ee+]+)",
                               text).group(1))
    return {"ERR": g("ENERGY ERROR"), "EW": g("EXTERNAL WORK"),
            "IE": g("INTERNAL ENERGY"), "KE": g("KINETIC ENERGY"),
            "NORMAL": "ENGINE TERMINATION : NORMAL" in text}


def _bar_two_bricks(master="100 1 0.5 0.5"):
    """2x1x1 bar of two unit bricks + a standalone master node (default at
    the bar's COG so a pure spin leaves the master still)."""
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
# Reader — XX/YY/ZZ map to dof 3/4/5 (no longer discarded)
# ============================================================================

@pytest.mark.parametrize("direction,dof", [
    ("XX", 3), ("YY", 4), ("ZZ", 5), ("X", 0), ("Y", 1), ("Z", 2)])
def test_impvel_direction_maps_to_dof(make_deck, direction, dof):
    """Every /IMPVEL direction resolves to its 6-DOF index; the rotational
    ones (XX/YY/ZZ) are kept, not dropped with a warning."""
    starter = (
        "/BEGIN\ndir probe\n"
        "/NODE\n1 0 0 0\n2 1 0 0\n"
        "/TRUSS/1\n1 1 2\n"
        "/PART/1\nbar\n1 1\n" + STEEL_LAW1 +
        "/PROP/TRUSS/1\nbar\n1.0\n"
        "/GRNOD/NODE/1\nn\n2\n"
        "/FUNCT/1\nramp\n0.0 0.0\n1.0 1.0\n"
        f"/IMPVEL/1\nspin\n1 {direction} 1 0.005\n"
        "/END\n"
    )
    model = _starter_only(make_deck, "DIR", starter)
    assert len(model.impvel) == 1
    assert model.impvel[0].dof == dof
    assert model.impvel[0].scale == pytest.approx(0.005)


def test_impdisp_rotational_maps_to_dof(make_deck):
    """/IMPDISP shares the reader — a rotational one is kept too."""
    starter = (
        "/BEGIN\nimpdisp dir\n"
        "/NODE\n1 0 0 0\n2 1 0 0\n"
        "/TRUSS/1\n1 1 2\n"
        "/PART/1\nbar\n1 1\n" + STEEL_LAW1 +
        "/PROP/TRUSS/1\nbar\n1.0\n"
        "/GRNOD/NODE/1\nn\n2\n"
        "/FUNCT/1\nramp\n0.0 0.0\n1.0 1.0\n"
        "/IMPDISP/1\ntwist\n1 ZZ 1 1.0\n"
        "/END\n"
    )
    model = _starter_only(make_deck, "IDR", starter)
    assert len(model.impdisp) == 1
    assert model.impdisp[0].dof == 5


def test_impvel_official_card_rolling_style(make_deck):
    """The exact ROLLING card layout (official fixed columns, Dir=XX,
    Fscale_Y on card 2) resolves to dof 3 with scale 0.005 — the card that
    used to be silently dropped on every RD-E-1000 deck."""
    starter = (
        "/BEGIN\nrolling card\n"
        "/NODE\n1020 99 0 0\n1021 100 0 0\n"
        "/TRUSS/1\n1 1020 1021\n"
        "/PART/1\nbar\n1 1\n" + STEEL_LAW1 +
        "/PROP/TRUSS/1\nbar\n1.0\n"
        "/GRNOD/NODE/6\nn_master\n1020\n"
        "/FUNCT/5\nlineaire\n-500 0\n0 0\n500 1\n10000 1\n"
        "/IMPVEL/1\nNew IMPVEL 1\n"
        "         5        XX         0         0         6         0         0\n"
        "                   0                .005                   0"
        "                   0\n"
        "/END\n"
    )
    model = _starter_only(make_deck, "ROL", starter)
    assert len(model.impvel) == 1
    iv = model.impvel[0]
    assert iv.dof == 3                       # XX -> rotation about X
    assert iv.grnod_id == 6
    assert iv.funct_id == 5
    assert iv.scale == pytest.approx(0.005)  # Fscale_Y read from card 2


# ============================================================================
# apply_kinematic — nodal rotational condition acts on vr, books vs inertia
# ============================================================================

def test_nodal_rotational_impvel_books_against_inertia(make_deck):
    """A nodal rotational /IMPVEL enforces the ANGULAR velocity ``vr`` and
    books the impulsive-start work as the leapfrog midstep against the
    ROTATIONAL inertia:  J_rot . (vr_old + w_imp)/2 = 1/2 J_rot w_imp^2 .
    The translational velocity is left untouched."""
    starter = (
        "/BEGIN\nnodal spin\n"
        "/NODE\n1 0 0 0\n2 1 0 0\n"
        "/TRUSS/1\n1 1 2\n"
        "/PART/1\nbar\n1 1\n" + STEEL_LAW1 +
        "/PROP/TRUSS/1\nbar\n1.0\n"
        "/GRNOD/NODE/1\ndriven\n2\n"
        "/FUNCT/1\nconst one\n0.0 1.0\n10.0 1.0\n"
        "/IMPVEL/1\nspin x const\n1 XX 1 3.0\n"
        "/END\n"
    )
    model = _starter_only(make_deck, "NSP", starter)
    loads = LoadsAndConstraints(model, MessageLog())
    i2 = model.node_index(2)

    J = 2.5
    inertia = np.full(model.numnod, J)          # explicit rotational inertia
    v = np.zeros_like(model.x)
    vr = np.zeros_like(model.x)
    v_old = np.zeros_like(model.x)
    vr_old = np.zeros_like(model.x)
    w = loads.apply_kinematic(0.5, v, vr, model.mass, model.x, 1e-4,
                              v_old, inertia, vr_old)

    wimp = 3.0                                   # scale 3.0 * const 1.0
    assert vr[i2, 0] == pytest.approx(wimp)      # XX -> vr component 0
    assert np.abs(vr[i2, 1:]).max() == 0.0       # only the driven component
    assert np.abs(v[i2]).max() == 0.0            # translation untouched
    # impulsive-start midstep work = 1/2 J w_imp^2 (NOT the endpoint J w^2)
    assert w == pytest.approx(0.5 * J * wimp ** 2, rel=1e-12)


def test_nodal_rotational_impdisp_finite_difference(make_deck):
    """A nodal rotational /IMPDISP enforces the imposed ANGLE by the
    finite-difference rate  (d(t) - d(t-dt))/dt  — exact for a DOF driven
    from d(0)=0. On a linear ramp d(t)=t the imposed spin is the constant
    slope."""
    starter = (
        "/BEGIN\nnodal twist\n"
        "/NODE\n1 0 0 0\n2 1 0 0\n"
        "/TRUSS/1\n1 1 2\n"
        "/PART/1\nbar\n1 1\n" + STEEL_LAW1 +
        "/PROP/TRUSS/1\nbar\n1.0\n"
        "/GRNOD/NODE/1\ndriven\n2\n"
        "/FUNCT/1\nramp\n0.0 0.0\n10.0 10.0\n"      # d(t) = t
        "/IMPDISP/1\ntwist z\n1 ZZ 1 1.0\n"
        "/END\n"
    )
    model = _starter_only(make_deck, "NTW", starter)
    loads = LoadsAndConstraints(model, MessageLog())
    i2 = model.node_index(2)

    inertia = np.ones(model.numnod)
    v = np.zeros_like(model.x)
    vr = np.zeros_like(model.x)
    dt = 0.05
    loads.apply_kinematic(0.4, v, vr, model.mass, model.x, dt,
                          np.zeros_like(v), inertia, np.zeros_like(vr))
    # ramp slope 1.0 -> spin rate 1.0 rad/time on the ZZ (vr component 2)
    assert vr[i2, 2] == pytest.approx(1.0, rel=1e-9)
    assert np.abs(vr[i2, :2]).max() == 0.0
    assert np.abs(v[i2]).max() == 0.0


# ============================================================================
# Full run — a rigid body spun by a rotational master /IMPVEL
# ============================================================================

def test_rbody_rotational_impvel_spins_and_books_work(make_deck):
    """The RD-E-1000 mechanism in miniature: a rotational /IMPVEL on the
    /RBODY master drives the body SPIN. A free bar spun about Z (a
    principal axis) reaches the imposed rate on every body node, its COG
    stays put (no net force), all its kinetic energy comes from the booked
    drive work, and the global balance closes."""
    wimp = 2.0
    starter = (
        "/BEGIN\nspun body\n" + _bar_two_bricks() +
        "/RBODY/1\nrigid bar\n100 1 0 1\n"
        "/GRNOD/NODE/2\nmaster\n100\n"
        "/FUNCT/1\nramp\n0.0 0.0\n0.1 2.0\n10.0 2.0\n"
        "/IMPVEL/1\nspin z\n1 ZZ 2 1.0\n"
        "/END\n"
    )
    engine = "/RUN/SPN/1\n0.5\n/DT\n0.9 0\n/PRINT/-5000\n/STOP\n5.0\n"
    model, out = _run(make_deck, "SPN", starter, engine)

    real = model.mass < 1e29                     # the 12 bar nodes (not master)
    # every body node spins at the imposed rate about Z
    assert model.vr[real, 2] == pytest.approx(wimp, rel=1e-6)
    assert np.abs(model.vr[real, :2]).max() < 1e-9
    # a pure spin about the COG carries zero net linear momentum
    m = model.mass[real]
    assert np.abs((m[:, None] * model.v[real]).sum(0)).max() < 1e-9
    # interior elements see the rigid field -> no stress, no internal energy
    assert np.abs(model.bricks.state["sig"]).max() < 1e-8
    assert abs(model.bricks.state["eint"].sum()) < 1e-9

    s = _final_summary(out)
    assert s["NORMAL"]
    ke = float(0.5 * (m * (model.v[real] ** 2).sum(1)).sum())
    assert ke > 0.0                              # the body really is moving
    assert s["EW"] == pytest.approx(ke, rel=2e-2)   # KE came from the drive
    assert s["IE"] == pytest.approx(0.0, abs=1e-6 * max(ke, 1.0))
    assert abs(s["ERR"]) < 2.0


def test_rbody_rotational_and_translational_directions_independent(make_deck):
    """A rotational master /IMPVEL leaves the body free to translate (no
    phantom translational drive): under gravity the spun body still falls
    at g t while spinning about Z at the imposed rate."""
    g, wimp, t_end = 10.0, 2.0, 0.3
    starter = (
        "/BEGIN\nspin and fall\n" + _bar_two_bricks() +
        "/RBODY/1\nrigid bar\n100 1 0 1\n"
        "/GRNOD/NODE/2\nmaster\n100\n"
        "/FUNCT/1\nconst\n0.0 1.0\n10.0 1.0\n"
        "/FUNCT/2\nspin\n0.0 2.0\n10.0 2.0\n"
        f"/GRAV/1\ngravity\n1 Z 0 -{g}\n"
        "/IMPVEL/1\nspin z\n2 ZZ 2 1.0\n"
        "/END\n"
    )
    engine = f"/RUN/SF/1\n{t_end}\n/DT\n0.9 0\n/PRINT/-5000\n/STOP\n50.0\n"
    model, out = _run(make_deck, "SF", starter, engine)
    real = model.mass < 1e29
    # spins at the imposed rate AND falls under gravity at g t
    assert model.vr[real, 2] == pytest.approx(wimp, rel=1e-6)
    assert model.v[real, 2] == pytest.approx(-g * t_end, rel=2e-3)
    s = _final_summary(out)
    assert s["NORMAL"]
    assert abs(s["ERR"]) < 2.0
