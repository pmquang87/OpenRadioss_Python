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


# ============================================================================
# M2 elements — analytic validations & stability proofs
# ============================================================================

def test_rigid_body_translation_m2_elements(make_deck):
    """A free tetra, a free 3-node shell and a free beam all in uniform
    translation: no straining, no stress, no resultants — objectivity and
    consistency of every M2 kernel in one run."""
    starter = (
        "/BEGIN\nm2 rigid flight\n"
        "/NODE\n"
        "1 0 0 0\n2 1 0 0\n3 0 1 0\n4 0 0 1\n"          # tetra
        "11 5 0 0\n12 6 0 0\n13 5 1 0\n"                # sh3n
        "21 10 0 0\n22 20 0 0\n23 10 5 5\n"             # beam + orientation
        "/TETRA4/1\n1 1 2 3 4\n"
        "/SH3N/2\n2 11 12 13\n"
        "/BEAM/3\n3 21 22 23\n"
        "/PART/1\ntet\n1 1\n"
        "/PART/2\ntri\n2 1\n"
        "/PART/3\nbeam\n3 1\n" + STEEL_LAW1 +
        "/PROP/SOLID/1\nsolid\n1.1 0.05 0.1\n"
        "/PROP/SHELL/2\nshell\n1 0 0 0\n0.01 0.01 0.01 0 0\n3 0 0.1\n"
        "/PROP/BEAM/3\nbeam\n25.0 52.08 52.08 88.0\n"
        "/GRNOD/PART/1\nall\n1 2 3\n"
        "/INIVEL/TRA/1\nfly\n2.0 1.0 0.5 1\n"
        "/END\n"
    )
    engine = "/RUN/RBM2/1\n0.5\n/DT\n0.9 0\n/PRINT/-1000\n/STOP\n5.0\n"
    model = _run(make_deck, "RBM2", starter, engine)
    assert model.x[0, 0] == pytest.approx(2.0 * 0.5, rel=1e-9)
    assert np.abs(model.tetras.state["sig"]).max() < 1e-14
    assert np.abs(model.sh3n.state["sig"]).max() < 1e-14
    assert np.abs(model.beams.state["fres"]).max() < 1e-14
    assert np.abs(model.beams.state["mres"]).max() < 1e-14
    # every mass-carrying node still flies at exactly the initial velocity
    real = model.mass < 1e29
    assert np.abs(model.v[real] - np.array([2.0, 1.0, 0.5])).max() < 1e-12


# ----------------------------------------------------------------------------
# divergence tests: a free element with one kicked node, thousands of
# cycles at /DT 0.9 — proves the exact per-element eigenvalue dt bound
# (the M1 time-step lesson) holds for every new element formulation.
# An unstable dt makes the response explode within a few hundred cycles.
# ----------------------------------------------------------------------------

_KICK_DECKS = {
    "tetra": (
        "/NODE\n1 0 0 0\n2 1 0 0\n3 0 1 0\n4 0 0 1\n"
        "/TETRA4/1\n1 1 2 3 4\n"
        "/PART/1\ntet\n1 1\n" + STEEL_LAW1 +
        "/PROP/SOLID/1\nsolid\n1.1 0.05 0.1\n"
        "/GRNOD/NODE/1\nkick\n4\n",
        0.2),
    "sh3n": (
        "/NODE\n1 0 0 0\n2 1 0 0\n3 0 1 0\n"
        "/SH3N/1\n1 1 2 3\n"
        "/PART/1\ntri\n1 1\n" + STEEL_LAW1 +
        "/PROP/SHELL/1\nshell\n1 0 0 0\n0.01 0.01 0.01 0 0\n3 0 0.1\n"
        "/GRNOD/NODE/1\nkick\n3\n",
        0.4),
    "shell_bt4_blt84": (
        "/NODE\n1 0 0 0\n2 1 0 0\n3 1 1 0\n4 0 1 0\n"
        "/SHELL/1\n1 1 2 3 4\n"
        "/PART/1\nquad\n1 1\n" + STEEL_LAW1 +
        "/PROP/SHELL/1\nshell\n1 0 0 0\n0.01 0.01 0.01 0 0\n3 0 0.1\n"
        "/GRNOD/NODE/1\nkick\n3\n",       # corner kick excites hourglass too
        0.4),
    "beam": (
        "/NODE\n1 0 0 0\n2 10 0 0\n3 20 0 0\n9 0 50 0\n"
        "/BEAM/1\n1 1 2 9\n2 2 3 9\n"
        "/PART/1\nbeam\n1 1\n" + STEEL_LAW1 +
        "/PROP/BEAM/1\nsquare 5x5\n25.0 52.08 52.08 88.0\n"
        "/GRNOD/NODE/1\nkick\n3\n",
        2.0),
}


@pytest.mark.parametrize("which", sorted(_KICK_DECKS))
def test_free_element_stability(make_deck, which):
    body, t_end = _KICK_DECKS[which]
    starter = ("/BEGIN\nstability " + which + "\n" + body +
               "/INIVEL/TRA/1\nkick z\n0 0 0.01 1\n/END\n")
    engine = (f"/RUN/ST/1\n{t_end}\n/DT\n0.9 0\n/PRINT/-100000\n"
              f"/STOP\n14.9\n")
    model = _run(make_deck, "ST", starter, engine)
    v = model.v[model.mass < 1e29]
    assert np.all(np.isfinite(v))
    # a kicked free element redistributes the 0.01 kick between rigid
    # motion and vibration; instability blows this up by orders of
    # magnitude within a fraction of t_end
    assert np.abs(v).max() < 0.1


# ----------------------------------------------------------------------------
# wave-speed validations
# ----------------------------------------------------------------------------

def _kuhn_tets(brick_nodes, coords):
    """Split one hexa (8 node ids, Radioss brick order) into the 6 Kuhn
    tetrahedra around the main diagonal — a decomposition that tiles
    conformally across a regular grid. Returns 6 node-id quadruples with
    positive volume."""
    # brick order -> bitmask corner index (local x + 2y + 4z)
    mask_of = [0, 2, 6, 4, 1, 3, 7, 5]
    node = {mask_of[k]: brick_nodes[k] for k in range(8)}
    tets = []
    import itertools
    for p in itertools.permutations((1, 2, 4)):
        quad = [node[0], node[p[0]], node[p[0] | p[1]], node[7]]
        a, b, c, d = (np.array(coords[q]) for q in quad)
        if np.dot(np.cross(b - a, c - a), d - a) < 0:
            quad[1], quad[2] = quad[2], quad[1]
        tets.append(quad)
    return tets


def test_tetra_wave_speed(make_deck):
    """The confined-column P-wave test of M1, meshed with Kuhn tetrahedra
    instead of bricks: the front must still travel at
    c = sqrt((K+4G/3)/rho) — same closed form, new element."""
    nel = 20
    nodes, coords = [], {}
    nid = 0
    for i in range(nel + 1):
        for (y, z) in ((0, 0), (1, 0), (1, 1), (0, 1)):
            nid += 1
            coords[nid] = (float(i), float(y), float(z))
            nodes.append(f"{nid} {float(i)} {float(y)} {float(z)}")
    tets = []
    eid = 0
    for e in range(nel):
        b = 4 * e
        brick = [b + 1, b + 2, b + 3, b + 4, b + 5, b + 6, b + 7, b + 8]
        for quad in _kuhn_tets(brick, coords):
            eid += 1
            tets.append(f"{eid} " + " ".join(str(q) for q in quad))
    first4 = "1 2 3 4"
    all_nodes = " ".join(str(i) for i in range(1, nid + 1))
    starter = (
        "/BEGIN\ntet wave bar\n"
        "/NODE\n" + "\n".join(nodes) + "\n"
        "/TETRA4/1\n" + "\n".join(tets) + "\n"
        "/PART/1\nbar\n1 1\n" + STEEL_LAW1 +
        "/PROP/SOLID/1\nsolid\n1.1 0.05 0.1\n"
        f"/GRNOD/NODE/1\ndriven end\n{first4}\n"
        f"/GRNOD/NODE/2\nall\n{all_nodes}\n"
        "/BCS/1\nconfine lateral\n011 111 0 2\n"
        "/FUNCT/1\nstep velocity\n0.0 1.0\n100.0 1.0\n"
        "/IMPVEL/1\npush\n1 X 1 0.01\n"
        "/END\n"
    )
    K, G, rho = 210 / (3 * (1 - 0.6)), 210 / 2.6, 7.8e-6
    c = np.sqrt((K + 4 * G / 3) / rho)       # = 6021.3 mm/ms
    t_end = 0.6 * nel / c
    engine = f"/RUN/TWAV/1\n{t_end}\n/DT\n0.67 0\n/PRINT/-1000\n/STOP\n90.0\n"
    model = _run(make_deck, "TWAV", starter, engine)
    front = c * t_end
    vx = model.v[:, 0]
    x0 = model.x0[:, 0]
    behind = x0 < front - 3.0
    ahead = x0 > front + 3.0
    assert vx[behind].min() > 0.005          # moving (imposed 0.01)
    assert np.abs(vx[ahead]).max() < 2e-4    # essentially at rest


def test_sh3n_membrane_wave_speed(make_deck):
    """A strip of /SH3N triangles, laterally constrained, hit by an end
    velocity: the membrane wave front travels at the plane-stress speed
    c = sqrt(E / (rho (1 - nu^2))) — 5439 mm/ms for this steel."""
    nel = 20
    nodes = []
    for i in range(nel + 1):
        nodes.append(f"{2 * i + 1} {float(i)} 0.0 0.0")
        nodes.append(f"{2 * i + 2} {float(i)} 1.0 0.0")
    tris = []
    for i in range(nel):
        a, d = 2 * i + 1, 2 * i + 2           # left edge (y=0, y=1)
        b, c = 2 * i + 3, 2 * i + 4           # right edge
        tris.append(f"{2 * i + 1} {a} {b} {c}")
        tris.append(f"{2 * i + 2} {a} {c} {d}")
    all_nodes = " ".join(str(i) for i in range(1, 2 * nel + 3))
    starter = (
        "/BEGIN\nsh3n wave strip\n"
        "/NODE\n" + "\n".join(nodes) + "\n"
        "/SH3N/1\n" + "\n".join(tris) + "\n"
        "/PART/1\nstrip\n1 1\n" + STEEL_LAW1 +
        "/PROP/SHELL/1\nshell\n1 0 0 0\n0.01 0.01 0.01 0 0\n3 0 0.1\n"
        "/GRNOD/NODE/1\ndriven end\n1 2\n"
        f"/GRNOD/NODE/2\nall\n{all_nodes}\n"
        "/BCS/1\nconfine\n011 111 0 2\n"      # only x translation free
        "/FUNCT/1\nstep velocity\n0.0 1.0\n100.0 1.0\n"
        "/IMPVEL/1\npush\n1 X 1 0.01\n"
        "/END\n"
    )
    c = np.sqrt(210 / (7.8e-6 * (1 - 0.3 ** 2)))     # plane-stress speed
    t_end = 0.6 * nel / c
    engine = f"/RUN/SWAV/1\n{t_end}\n/DT\n0.67 0\n/PRINT/-1000\n/STOP\n90.0\n"
    model = _run(make_deck, "SWAV", starter, engine)
    front = c * t_end
    vx = model.v[:, 0]
    x0 = model.x0[:, 0]
    # membrane waves have no bulk-viscosity smearing: the front rings, so
    # use wider margins than the solid tests
    behind = x0 < front - 4.0
    ahead = x0 > front + 4.0
    assert vx[behind].min() > 0.005
    assert np.abs(vx[ahead]).max() < 1.5e-3


# ----------------------------------------------------------------------------
# bending-vibration validations (closed-form Euler-Bernoulli frequency)
# ----------------------------------------------------------------------------

_BL1 = 1.8751040687119611          # first cantilever eigenvalue beta*L
_SIG1 = (np.sinh(_BL1) - np.sin(_BL1)) / (np.cosh(_BL1) + np.cos(_BL1))


def _cantilever_mode(xi):
    """First cantilever mode shape, normalized to 1 at the tip."""
    phi = (np.cosh(_BL1 * xi) - np.cos(_BL1 * xi)
           - _SIG1 * (np.sinh(_BL1 * xi) - np.sin(_BL1 * xi)))
    tip = (np.cosh(_BL1) - np.cos(_BL1)
           - _SIG1 * (np.sinh(_BL1) - np.sin(_BL1)))
    return phi / tip


def test_beam_cantilever_frequency(make_deck):
    """Clamped-free beam released with its first-mode velocity profile:
    after exactly half the analytic period T = 2 pi / omega,
    omega = 1.875^2 sqrt(EI/(rho A L^4)), the tip velocity must be the
    exact opposite of the initial one (v = -v0 cos(pi*f_err/f), so the
    check is quadratically insensitive to the small discretization
    error). Steel, L=100, square 5x5 section, 10 elements."""
    L, nel, v0 = 100.0, 10, 0.01
    E, rho, A, Iy = 210.0, 7.8e-6, 25.0, 52.0833333
    nodes = [f"{i + 1} {L * i / nel} 0.0 0.0" for i in range(nel + 1)]
    nodes.append("99 0.0 1000.0 0.0")                  # orientation node
    beams = [f"{i + 1} {i + 1} {i + 2} 99" for i in range(nel)]
    inivel, grnod = [], []
    for i in range(1, nel + 1):                        # node 1 is clamped
        vz = v0 * _cantilever_mode(i / nel)
        grnod.append(f"/GRNOD/NODE/{i}\nn{i}\n{i + 1}\n")
        inivel.append(f"/INIVEL/TRA/{i}\nmode1\n0 0 {vz:.10E} {i}\n")
    starter = (
        "/BEGIN\ncantilever beam\n"
        "/NODE\n" + "\n".join(nodes) + "\n"
        "/BEAM/1\n" + "\n".join(beams) + "\n"
        "/PART/1\nbeam\n1 1\n" + STEEL_LAW1 +
        f"/PROP/BEAM/1\nsquare 5x5\n{A} {Iy} {Iy} 88.0\n"
        "/GRNOD/NODE/90\nroot\n1\n"
        "/BCS/1\nclamp\n111 111 0 90\n"
        + "".join(grnod) + "".join(inivel) +
        "/END\n"
    )
    omega = _BL1 ** 2 * np.sqrt(E * Iy / (rho * A * L ** 4))
    t_half = np.pi / omega
    engine = (f"/RUN/CANT/1\n{t_half}\n/DT\n0.9 0\n/PRINT/-100000\n"
              f"/STOP\n10.0\n")
    model = _run(make_deck, "CANT", starter, engine)
    tip = model.node_index(nel + 1)
    # tip velocity reversed after half a period (see docstring)
    assert model.v[tip, 2] == pytest.approx(-v0, rel=0.05)
    # peak tip deflection during the run ~ v0/omega; end-of-half-period
    # deflection back near zero
    assert abs(model.x[tip, 2] - model.x0[tip, 2]) < 0.25 * v0 / omega


def test_shell_cantilever_bending_frequency(make_deck):
    """The same first-mode release for a cantilever strip of 4-node BT
    shells (nu = 0 so beam theory is exact for the strip): validates that
    the BLT84 stiffness hourglass keeps coarse dynamic bending neither
    overdamped (the old viscous control would) nor overstiff.
    EI = E b t^3/12, 10 elements, tolerance 5%."""
    L, b, t, nel, v0 = 100.0, 10.0, 2.0, 10, 0.01
    E, rho = 210.0, 7.8e-6
    nodes = []
    for i in range(nel + 1):
        nodes.append(f"{2 * i + 1} {L * i / nel} 0.0 0.0")
        nodes.append(f"{2 * i + 2} {L * i / nel} {b} 0.0")
    shells = [f"{i + 1} {2 * i + 1} {2 * i + 3} {2 * i + 4} {2 * i + 2}"
              for i in range(nel)]
    inivel, grnod = [], []
    for i in range(1, nel + 1):
        vz = v0 * _cantilever_mode(i / nel)
        grnod.append(f"/GRNOD/NODE/{i}\ns{i}\n{2 * i + 1} {2 * i + 2}\n")
        inivel.append(f"/INIVEL/TRA/{i}\nmode1\n0 0 {vz:.10E} {i}\n")
    starter = (
        "/BEGIN\ncantilever strip\n"
        "/NODE\n" + "\n".join(nodes) + "\n"
        "/SHELL/1\n" + "\n".join(shells) + "\n"
        "/PART/1\nstrip\n1 1\n"
        "/MAT/LAW1/1\nsteel nu=0\n7.8e-6\n210. 0.0\n"
        f"/PROP/SHELL/1\nstrip\n1 0 0 0\n0.01 0.01 0.01 0 0\n3 0 {t}\n"
        "/GRNOD/NODE/90\nroot\n1 2\n"
        "/BCS/1\nclamp\n111 111 0 90\n"
        + "".join(grnod) + "".join(inivel) +
        "/END\n"
    )
    Iy = b * t ** 3 / 12.0
    A = b * t
    omega = _BL1 ** 2 * np.sqrt(E * Iy / (rho * A * L ** 4))
    t_half = np.pi / omega
    engine = (f"/RUN/CSTR/1\n{t_half}\n/DT\n0.9 0\n/PRINT/-100000\n"
              f"/STOP\n10.0\n")
    model = _run(make_deck, "CSTR", starter, engine)
    # mass-weighted projection of the velocity on the initial mode shape:
    # it filters the few-% higher-mode contamination that seeding the
    # CONTINUOUS mode shape leaves at the discrete tip, and checks the
    # frequency quadratically (q_dot(T/2) = -v0 cos(pi * f_err/f))
    phi = np.zeros(model.numnod)
    for i in range(1, nel + 1):
        for nid in (2 * i + 1, 2 * i + 2):
            phi[model.node_index(nid)] = _cantilever_mode(i / nel)
    mw = model.mass * phi
    qdot = float((mw * model.v[:, 2]).sum() / (mw * phi).sum())
    assert qdot == pytest.approx(-v0, rel=0.03)
    # stiffness hourglass stores energy elastically: over half a period
    # the net hourglass energy must stay a tiny fraction of the total
    he = float(model.shells.state["ehour"].sum())
    ke0 = 0.0
    for i in range(1, nel + 1):
        m = rho * b * t * L / nel
        ke0 += 0.5 * m * (v0 * _cantilever_mode(i / nel)) ** 2
    assert abs(he) < 0.05 * ke0
