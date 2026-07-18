"""M39 — rigid-body nodal time step (STIFN transport to the master).

The bug (RD-E-1000 Bending, the BATOZ rolling family): the port marks
every /RBODY member node "prescribed" and drops it from the nodal-time-step
minimum, but never gives the body a step of its own — so a stiff shell
welded into a rigid body stops constraining dt and the port runs it far too
fast (the ROLLING deck: dt 4.31e-2 vs the Fortran 1.64e-2, 2.6x, driving
the shells hourglass-unstable).

The fix mirrors ``rgbodfp.F`` (called from ``rbyfor.F``) + ``dtnoda.F``:
the members' nodal stiffness is transported to the master —
``K_tra = Sum STIFN`` and ``K_rot = Sum |x-x_master|^2 * STIFN`` (the
Huygens-Steiner parallel-axis term) — and the master's two nodal steps
``sqrt(2 M/K_tra)`` / ``sqrt(2 IN_min/K_rot)`` rejoin the global minimum
(``IN_min`` = the body's minimum principal moment, ``inirby.F`` line 838).

These tests assert (a) the transported dt DOES bound the step well below the
un-constrained value, (b) the run is stable, and (c) the transport is an
exact no-op on a mesh with no /RBODY (byte-identical to before the change),
plus a closed-form unit check of the transport formula itself.
"""

import contextlib
import io
import re

import numpy as np
import pytest

from pyradioss.common.constants import EP30
from pyradioss.common.messages import MessageLog
from pyradioss.engine.engine import run_engine
from pyradioss.engine.mass_scaling import NodalTimeStep
from pyradioss.starter.starter import run_starter

STEEL = "/MAT/LAW1/1\nsteel\n7.8e-6\n210. 0.3\n"


def _strip(with_rbody: bool, engine: str, master_z: float = 2.0) -> str:
    """A 5-quad uniform shell strip along x (6x2 node grid, 2x2 quads),
    the far end clamped.  With ``with_rbody`` the near-end pair of nodes is
    welded into an /RBODY whose massless master sits offset in z and is
    driven by an /IMPVEL (a moving rigid grip)."""
    nx, L, w = 6, 10.0, 2.0
    nlines, nid = [], 0
    for i in range(nx):
        for y in (0.0, w):
            nid += 1
            nlines.append(f"{nid} {L * i / (nx - 1)} {y} 0")
    slaves = (nid - 1, nid)
    shells = [f"{i + 1} {2*i+1} {2*i+2} {2*i+4} {2*i+3}" for i in range(nx - 1)]
    deck = "/BEGIN\nrbody dt\n/NODE\n" + "\n".join(nlines) + "\n"
    if with_rbody:
        deck += f"1000 {L} {w/2} {master_z}\n"
    deck += ("/SHELL/1\n" + "\n".join(shells) + "\n"
             "/PART/1\nstrip\n1 1\n" + STEEL +
             "/PROP/SHELL/1\nsh\n1 0 0 0\n0.5 0.5 0.5 0 0\n3 0 1.0\n"
             "/GRNOD/NODE/1\nclamp\n1 2\n/BCS/1\nclamp\n111 111 0 1\n")
    if with_rbody:
        deck += (f"/RBODY/1\nrb\n1000 2 0 0\n"
                 f"/GRNOD/NODE/2\nsl\n{slaves[0]} {slaves[1]}\n"
                 "/FUNCT/1\nramp\n0 0\n1 1\n100 1\n"
                 "/IMPVEL/1\ndrive\n1 X 2 0.01\n")
    return deck + "/END\n", engine


def _run(make_deck, name, deck, engine):
    s, e = make_deck(name, deck, engine)
    with contextlib.redirect_stdout(io.StringIO()):
        run_starter(s)
        model = run_engine(e)
    return model, e.replace("_0001.rad", "_0001.out")


def _initial_dt(out):
    return float(re.search(r"INITIAL TIME STEP\s*\.[ .]*:\s*([-\d.Ee+]+)",
                           open(out).read()).group(1))


def _summary(out):
    txt = open(out).read()
    err = float(re.search(r"ENERGY ERROR\s*\.[ .]*:\s*([-\d.Ee+]+)",
                          txt).group(1))
    return err, "ENGINE TERMINATION : NORMAL" in txt


def _eng(name, elem=False):
    dt = "/DT" if elem else "/DT/NODA"
    return f"/RUN/{name}/1\n0.3\n{dt}\n0.9 0\n/PRINT/-10000\n/STOP\n1.0\n"


def test_rbody_member_stiffness_bounds_dt(make_deck):
    """A stiff shell welded into an /RBODY must constrain the step through
    its master (rgbodfp.F/dtnoda.F): the rigid deck's dt is far below the
    same mesh with the body removed — WITHOUT the transport the two would
    be equal (the member stiffness would be silently dropped) — and the run
    stays stable (NORMAL, balance closed)."""
    d_rb, _ = _strip(True, _eng("RBDTR"))
    d_free, _ = _strip(False, _eng("RBDTF"))
    _, o_rb = _run(make_deck, "RBDTR", d_rb, _eng("RBDTR"))
    _, o_free = _run(make_deck, "RBDTF", d_free, _eng("RBDTF"))
    dt_rb, dt_free = _initial_dt(o_rb), _initial_dt(o_free)
    assert dt_rb > 0.0
    # the parallel-axis transport of the welded end binds the step to well
    # under half the un-constrained value (measured ratio ~0.29)
    assert dt_rb < 0.5 * dt_free
    err, normal = _summary(o_rb)
    assert normal
    assert abs(err) < 1.0


def test_rbody_transport_is_noop_without_rbody(make_deck):
    """No /RBODY -> the transport is inert: the /DT/NODA step is byte-
    identical to the /DT element step on this uniform mesh (i.e. exactly
    what the nodal formula gave before the change; the added
    ``min(..., _rigid_body_dt())`` term is EP30 and cannot move it)."""
    d_noda, _ = _strip(False, _eng("RBNODA"))
    d_elem, _ = _strip(False, _eng("RBELEM", elem=True))
    _, o_noda = _run(make_deck, "RBNODA", d_noda, _eng("RBNODA"))
    _, o_elem = _run(make_deck, "RBELEM", d_elem, _eng("RBELEM", elem=True))
    assert _initial_dt(o_noda) == pytest.approx(_initial_dt(o_elem), rel=1e-12)


class _Controls:
    dt_noda = "NODA"          # truthy, and not "CST"
    dt_scale = 0.9
    dt_min = 0.0


def test_rbody_transport_formula_closed_form(make_deck):
    """Unit check of the transport itself: with a hand-set nodal stiffness,
    ``_rigid_body_dt`` returns exactly the minimum of the translational
    ``sqrt(2 M/K_tra)`` and rotational ``sqrt(2 IN_min/K_rot)`` master
    steps, and an unregistered NodalTimeStep returns EP30 (pure no-op)."""
    # a one-shell model just to build a real NodalTimeStep
    deck = ("/BEGIN\none shell\n/NODE\n1 0 0 0\n2 3 0 0\n3 3 2 0\n4 0 2 0\n"
            "/SHELL/1\n1 1 2 3 4\n/PART/1\np\n1 1\n" + STEEL +
            "/PROP/SHELL/1\nsh\n1 0 0 0\n0.5 0.5 0.5 0 0\n3 0 1.0\n/END\n")
    s, _ = make_deck("ONE", deck, "/RUN/ONE/1\n1.0\n")
    with contextlib.redirect_stdout(io.StringIO()):
        model = run_starter(s)

    noda = NodalTimeStep(model, _Controls(), MessageLog())
    assert noda._rigid_body_dt() == EP30          # nothing registered: no-op

    nodes = np.arange(4)
    stif = np.array([100.0, 100.0, 200.0, 200.0])
    noda.stifn[nodes] = stif
    mass, inertia = 5.0, np.diag([2.0, 3.0, 4.0])
    noda.add_rigid_body(nodes, nodes[0], mass, inertia, model.x0)

    dd = ((model.x0[nodes] - model.x0[nodes[0]]) ** 2).sum(axis=1)
    k_tra = stif.sum()
    k_rot = float((dd * stif).sum())
    expect = min(np.sqrt(2.0 * mass / k_tra),
                 np.sqrt(2.0 * 2.0 / k_rot))       # 2.0 = min principal
    assert noda._rigid_body_dt() == pytest.approx(expect, rel=1e-12)
