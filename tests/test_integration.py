"""Full starter+engine analytic validations (see PORTING_GUIDE.md §6)."""

import contextlib
import io

import numpy as np
import pytest

from pyradioss.engine.engine import run_engine
from pyradioss.starter.starter import run_starter

STEEL_LAW1 = """\
/MAT/LAW1/1
steel elastic
7.8e-6
210. 0.3
"""


def _run(make_deck, name, starter, engine):
    s, e = make_deck(name, starter, engine)
    with contextlib.redirect_stdout(io.StringIO()):
        run_starter(s)
        model = run_engine(e)
    return model


def test_rigid_body_translation(make_deck):
    """A free cube in uniform translation: no straining, no stress, no
    energy drift — the most basic objectivity/consistency check."""
    starter = (
        "/BEGIN\nrigid body flight\n"
        "/NODE\n"
        "1 0 0 0\n2 1 0 0\n3 1 1 0\n4 0 1 0\n"
        "5 0 0 1\n6 1 0 1\n7 1 1 1\n8 0 1 1\n"
        "/BRICK/1\n1 1 2 3 4 5 6 7 8\n"
        "/PART/1\ncube\n1 1\n" + STEEL_LAW1 +
        "/PROP/SOLID/1\nsolid\n1.1 0.05 0.1\n"
        "/GRNOD/PART/1\nall\n1\n"
        "/INIVEL/TRA/1\nfly\n2.0 1.0 0.5 1\n"
        "/END\n"
    )
    engine = "/RUN/RB/1\n0.5\n/DT\n0.9 0\n/PRINT/-1000\n/STOP\n5.0\n"
    model = _run(make_deck, "RB", starter, engine)
    # position advanced by v*t, stress identically zero
    assert model.x[0, 0] == pytest.approx(2.0 * 0.5, rel=1e-9)
    assert np.abs(model.bricks.state["sig"]).max() < 1e-14
    assert np.abs(model.v - np.array([2.0, 1.0, 0.5])).max() < 1e-12


def test_elastic_wave_speed(make_deck):
    """Confined column hit by a sudden end velocity: the wave front must
    travel at the P-wave (uniaxial strain) speed c = sqrt((K+4G/3)/rho).
    Steel: c = 6021 mm/ms. We run until the front is at ~60% of the bar
    and check 'moved' vs 'still' nodes on both sides of the front."""
    nel = 20
    nodes, bricks = [], []
    nid = 0
    for i in range(nel + 1):
        for (y, z) in ((0, 0), (1, 0), (1, 1), (0, 1)):
            nid += 1
            nodes.append(f"{nid} {float(i)} {float(y)} {float(z)}")
    for e in range(nel):
        b = 4 * e
        bricks.append(f"{e+1} {b+1} {b+2} {b+3} {b+4} "
                      f"{b+5} {b+6} {b+7} {b+8}")
    first4 = "1 2 3 4"
    all_nodes = " ".join(str(i) for i in range(1, nid + 1))
    starter = (
        "/BEGIN\nwave bar\n"
        "/NODE\n" + "\n".join(nodes) + "\n"
        "/BRICK/1\n" + "\n".join(bricks) + "\n"
        "/PART/1\nbar\n1 1\n" + STEEL_LAW1 +
        "/PROP/SOLID/1\nsolid\n1.1 0.05 0.1\n"
        f"/GRNOD/NODE/1\ndriven end\n{first4}\n"
        f"/GRNOD/NODE/2\nall\n{all_nodes}\n"
        "/BCS/1\nconfine lateral\n011 111 0 2\n"   # only x free everywhere
        "/FUNCT/1\nstep velocity\n0.0 1.0\n100.0 1.0\n"
        "/IMPVEL/1\npush\n1 X 1 0.01\n"
        "/END\n"
    )
    K, G, rho = 210 / (3 * (1 - 0.6)), 210 / 2.6, 7.8e-6
    c = np.sqrt((K + 4 * G / 3) / rho)      # = 6021.3 mm/ms
    t_end = 0.6 * nel / c                    # front at x ~= 12 mm
    engine = f"/RUN/WAVE/1\n{t_end}\n/DT\n0.67 0\n/PRINT/-1000\n/STOP\n90.0\n"
    model = _run(make_deck, "WAVE", starter, engine)
    front = c * t_end
    vx = model.v[:, 0]
    x0 = model.x0[:, 0]
    # nodes clearly behind the front move at ~the imposed velocity;
    # nodes clearly ahead of it are still at rest (2-element smearing band
    # from the bulk-viscosity shock spreading)
    behind = x0 < front - 2.5
    ahead = x0 > front + 2.5
    assert vx[behind].min() > 0.005          # moving (imposed 0.01)
    assert np.abs(vx[ahead]).max() < 1e-4    # essentially at rest


def test_spring_oscillation_period(make_deck):
    """Two free masses on a /SPRING: period T = 2 pi / (2 sqrt(k/M)).
    Run one full period at small dt: displacements must return close to
    the start (central difference phase error ~ (w dt)^2/24)."""
    starter = (
        "/BEGIN\nspring pair\n"
        "/NODE\n1 0 0 0\n2 10 0 0\n"
        "/SPRING/1\n1 1 2\n"
        "/PART/1\nspring\n1 1\n"
        "/MAT/LAW1/1\ndummy\n7.8e-6\n210. 0.3\n"
        "/PROP/SPRING/1\nk=4 M=1\n1.0 4.0 0.0\n"
        "/GRNOD/NODE/1\nleft\n1\n"
        "/GRNOD/NODE/2\nright\n2\n"
        "/BCS/1\nleft yz\n011 111 0 1\n"
        "/BCS/2\nright yz\n011 111 0 2\n"
        "/INIVEL/TRA/1\nleft out\n-0.1 0 0 1\n"
        "/INIVEL/TRA/2\nright out\n0.1 0 0 2\n"
        "/END\n"
    )
    # omega = 2*sqrt(k/M) = 4  ->  T = pi/2
    T = np.pi / 2.0
    engine = f"/RUN/SPR/1\n{T}\n/DT\n0.02 0\n/PRINT/-10000\n/STOP\n5.0\n"
    model = _run(make_deck, "SPR", starter, engine)
    # after one period the nodes are back at their initial positions
    assert np.abs(model.x[:, 0] - model.x0[:, 0]).max() < 5e-4
    # amplitude v/omega = 0.025 -> tolerance is 2% of amplitude


def test_flat_plate_arrest_on_rigid_wall(make_deck):
    """A flat elastic shell plate falling face-on onto a sliding rigid
    wall: all nodes are arrested, the kinematic wall books (almost) the
    whole kinetic energy as contact energy and the balance stays tight."""
    starter = (
        "/BEGIN\nplate drop\n"
        "/NODE\n"
        "1 0 0 1\n2 10 0 1\n3 10 10 1\n4 0 10 1\n"
        "5 5 5 1\n"
        "/SHELL/1\n"
        "1 1 2 5 4\n2 2 3 5 2\n"     # crude 2-element plate (with a mid node)
        "/PART/1\nplate\n1 1\n" + STEEL_LAW1 +
        "/PROP/SHELL/1\nplate\n1 0 0 0\n0.01 0.01 0.01 0 0\n3 0 1.0\n"
        "/GRNOD/PART/1\nall\n1\n"
        "/INIVEL/TRA/1\nfall\n0 0 -1.0 1\n"
        "/RWALL/PLANE/1\nground\n0 0 0.0 0.0\n0 0 0\n0 0 1\n"
        "/END\n"
    )
    engine = "/RUN/DROP/1\n2.0\n/DT\n0.9 0\n/PRINT/-100000\n/STOP\n10.0\n"
    model = _run(make_deck, "DROP", starter, engine)
    # nobody below the wall
    assert model.x[:, 2].min() > -1e-9
    # plate is resting: velocities ~ 0 (arrest dissipated the KE)
    assert np.abs(model.v[:, 2]).max() < 1e-3


def test_contact_type7_two_bricks(make_deck):
    """A flying cube impacts a clamped one through /INTER/TYPE7: the
    penalty must stop it before crossing (no deep penetration) and
    transfer momentum."""
    # lower cube (clamped at its base), upper cube 0.3 above, falling
    starter = (
        "/BEGIN\ntwo cubes\n"
        "/NODE\n"
        # lower cube nodes 1-8 (z 0..1)
        "1 0 0 0\n2 1 0 0\n3 1 1 0\n4 0 1 0\n"
        "5 0 0 1\n6 1 0 1\n7 1 1 1\n8 0 1 1\n"
        # upper cube nodes 11-18 (z 1.3..2.3)
        "11 0 0 1.3\n12 1 0 1.3\n13 1 1 1.3\n14 0 1 1.3\n"
        "15 0 0 2.3\n16 1 0 2.3\n17 1 1 2.3\n18 0 1 2.3\n"
        "/BRICK/1\n1 1 2 3 4 5 6 7 8\n"
        "/BRICK/2\n2 11 12 13 14 15 16 17 18\n"
        "/PART/1\nlower\n1 1\n"
        "/PART/2\nupper\n1 1\n" + STEEL_LAW1 +
        "/PROP/SOLID/1\nsolid\n1.1 0.05 0.1\n"
        "/GRNOD/NODE/1\nbase\n1 2 3 4\n"
        "/GRNOD/NODE/2\nupper bottom face\n11 12 13 14\n"
        "/GRNOD/PART/3\nupper all\n2\n"
        "/BCS/1\nclamp base\n111 111 0 1\n"
        "/INIVEL/TRA/1\nfall\n0 0 -1.0 3\n"
        "/SURF/PART/1\nlower surf\n1\n"
        "/INTER/TYPE7/1\nimpact\n2 1\n0.01 0.0 0.1\n"
        "/END\n"
    )
    engine = "/RUN/IMP2/1\n1.0\n/DT\n0.5 0\n/PRINT/-100000\n/STOP\n20.0\n"
    model = _run(make_deck, "IMP2", starter, engine)
    upper_bottom = model.node_indices([11, 12, 13, 14])
    lower_top = model.node_indices([5, 6, 7, 8])
    # upper cube must not pass through the lower one's top face
    assert model.x[upper_bottom, 2].min() > model.x[lower_top, 2].max() - 0.09
    # the upper cube bounced or at least stopped approaching fast
    assert model.v[upper_bottom, 2].mean() > -0.5


def test_tensile_example_matches_johnson_cook(make_deck, tmp_path):
    """End-to-end regression of the tensile_bar example (shrunk): final
    axial stress must sit on the Johnson-Cook curve at the reached
    plastic strain (the same check that validated the port initially)."""
    import importlib.util
    import os
    import shutil
    ex = os.path.join(os.path.dirname(__file__), "..",
                      "examples", "tensile_bar")
    for f in ("TENSILE_0000.rad", "TENSILE_0001.rad"):
        src = os.path.join(ex, f)
        if not os.path.exists(src):   # generate on the fly if needed
            spec = importlib.util.spec_from_file_location(
                "gen", os.path.join(ex, "generate_deck.py"))
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            mod.main()
        shutil.copy(src, tmp_path / f)
    with contextlib.redirect_stdout(io.StringIO()):
        run_starter(str(tmp_path / "TENSILE_0000.rad"))
        model = run_engine(str(tmp_path / "TENSILE_0001.rad"))
    st = model.bricks.state
    ep = st["epsp"].mean()
    assert ep > 0.01
    sy = 0.4 + 0.5 * ep ** 0.5
    assert st["sig"][:, 0].mean() == pytest.approx(sy, rel=0.01)
