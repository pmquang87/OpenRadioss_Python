"""M4 contact validations: /INTER/TYPE7 full options, /INTER/TYPE2 tied,
/INTER/TYPE11 edge-to-edge, and the M3<->M4 element-deletion interaction.

Layered like the rest of the suite (PORTING_GUIDE §6):

* unit tests against closed-form results (stiffness formulas, penalty
  force at a known penetration, tied nodes following their segment
  exactly, conservation of the tied force/mass transfer, deletion masks);
* full starter+engine analytic validations (wave transmission through a
  tied interface with ZERO contact energy, two-body edge impact momentum,
  self-impact arrest, long /DT 0.9 impacts for every stiffness variant,
  and the notched-plate crack with self-contact).
"""

import contextlib
import io
import re

import numpy as np
import pytest

from pyradioss.common.messages import MessageLog
from pyradioss.contact import ContactType2, ContactType7, ContactType11
from pyradioss.contact import stiffness as cstiff
from pyradioss.contact import tracking
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
    return model, e.replace("_0001.rad", "_0001.out")


def _starter_only(make_deck, name, starter):
    s, _ = make_deck(name, starter, "/RUN/X/1\n1.0\n")
    with contextlib.redirect_stdout(io.StringIO()):
        return run_starter(s)


def _final_summary(out_path):
    """Parse the engine termination block: returns dict with ERR (%), CE
    (contact energy), NORMAL (bool)."""
    text = open(out_path).read()
    err = float(re.search(r"ENERGY ERROR\s*\.[ .]*:\s*([-\d.Ee+]+)",
                          text).group(1))
    ce = float(re.search(r"CONTACT ENERGY\s*\.[ .]*:\s*([-\d.Ee+]+)",
                         text).group(1))
    return {"ERR": err, "CE": ce,
            "NORMAL": "ENGINE TERMINATION : NORMAL" in text}


# ============================================================================
# Unit tests — stiffness/gap formulas (contact/stiffness.py, i7sti3)
# ============================================================================

SHELL_AND_BRICK = (
    "/BEGIN\nstiffness probe\n"
    "/NODE\n"
    # 1x1 shell at z = 5, thickness 2 from the property below
    "1 0 0 5\n2 1 0 5\n3 1 1 5\n4 0 1 5\n"
    # unit cube
    "11 0 0 0\n12 1 0 0\n13 1 1 0\n14 0 1 0\n"
    "15 0 0 1\n16 1 0 1\n17 1 1 1\n18 0 1 1\n"
    "/SHELL/1\n1 1 2 3 4\n"
    "/BRICK/2\n2 11 12 13 14 15 16 17 18\n"
    "/PART/1\nshell\n1 1\n"
    "/PART/2\nbrick\n2 1\n" + STEEL_LAW1 +
    "/PROP/SHELL/1\nshell\n1 0 0 0\n0.01 0.01 0.01 0 0\n3 0 2.0\n"
    "/PROP/SOLID/2\nsolid\n1.1 0.05 0.1\n"
    "/SURF/PART/1\nshell surf\n1\n"
    "/SURF/PART/2\nbrick surf\n2\n"
    "/END\n"
)


def test_stiffness_formulas(make_deck):
    """The i7sti3 element formulas, against hand-evaluated values:
    shell K = 0.5*E*t, gap t/2; solid face K = B*A^2/V; solid node
    K = B*V^(1/3). Steel E=210, nu=0.3 -> B = 175."""
    model = _starter_only(make_deck, "STIF", SHELL_AND_BRICK)
    B = 210.0 / (3 * (1 - 0.6))          # 175

    surf_sh = model.surfaces[1]
    K, gap = cstiff.segment_stiffness_gap(
        model, surf_sh.segments, surf_sh.seg_gtype, surf_sh.seg_elem, 1.0)
    assert K == pytest.approx(0.5 * 210.0 * 2.0)          # 210
    assert gap == pytest.approx(1.0)                      # t/2

    surf_br = model.surfaces[2]
    K, gap = cstiff.segment_stiffness_gap(
        model, surf_br.segments, surf_br.seg_gtype, surf_br.seg_elem, 1.0)
    assert len(K) == 6                                    # 6 free faces
    assert K == pytest.approx(B * 1.0 ** 2 / 1.0)         # B*A^2/V = 175
    assert gap == pytest.approx(0.0)                      # solids: no gap

    Kn, gn = cstiff.node_stiffness_gap(model, 1.0)
    sh_nodes = model.node_indices([1, 2, 3, 4])
    br_nodes = model.node_indices([11, 18])
    assert Kn[sh_nodes] == pytest.approx(210.0)
    assert gn[sh_nodes] == pytest.approx(1.0)
    assert Kn[br_nodes] == pytest.approx(B)               # B * V^(1/3)
    assert gn[br_nodes] == pytest.approx(0.0)


def test_istf_combinations():
    """The 6 Istf stiffness combinations, elementwise."""
    Km = np.array([100.0, 100.0])
    Ks = np.array([50.0, 0.0])   # second pair: secondary has no stiffness
    assert cstiff.combine_stiffness(0, 2.0, Km, Ks) == pytest.approx(
        [100.0, 100.0])
    assert cstiff.combine_stiffness(1, 7.5, Km, Ks) == pytest.approx(
        [7.5, 7.5])
    # zero secondary stiffness falls back on the main value (i7sti3 guard)
    assert cstiff.combine_stiffness(2, 1.0, Km, Ks) == pytest.approx(
        [75.0, 100.0])
    assert cstiff.combine_stiffness(3, 1.0, Km, Ks) == pytest.approx(
        [100.0, 100.0])
    assert cstiff.combine_stiffness(4, 1.0, Km, Ks) == pytest.approx(
        [50.0, 100.0])
    assert cstiff.combine_stiffness(5, 1.0, Km, Ks) == pytest.approx(
        [100.0 * 50.0 / 150.0, 50.0])


# ============================================================================
# Unit tests — TYPE7 penalty force at a known penetration
# ============================================================================

def test_type7_penalty_force_closed_form(make_deck):
    """Istf=1 (Stfac IS the stiffness): a node held at penetration p with
    zero velocity must feel exactly F = K*p along the surface normal, and
    the segment corners the exact opposite total (momentum). The upper
    block is INSET so each of its bottom nodes projects on the interior
    of the lower cube's top face only (a node over a corner would collect
    one spring per adjacent face — tested separately)."""
    K, gap = 5.0, 0.1
    starter = (
        "/BEGIN\ninset block on cube\n"
        "/NODE\n"
        "1 0 0 0\n2 1 0 0\n3 1 1 0\n4 0 1 0\n"
        "5 0 0 1\n6 1 0 1\n7 1 1 1\n8 0 1 1\n"
        "11 0.25 0.25 1.3\n12 0.75 0.25 1.3\n"
        "13 0.75 0.75 1.3\n14 0.25 0.75 1.3\n"
        "15 0.25 0.25 2.3\n16 0.75 0.25 2.3\n"
        "17 0.75 0.75 2.3\n18 0.25 0.75 2.3\n"
        "/BRICK/1\n1 1 2 3 4 5 6 7 8\n"
        "/BRICK/2\n2 11 12 13 14 15 16 17 18\n"
        "/PART/1\nlower\n1 1\n"
        "/PART/2\nupper\n2 1\n" + STEEL_LAW1 +
        "/PROP/SOLID/1\nsolid\n1.1 0.05 0.1\n"
        "/PROP/SOLID/2\nsolid\n1.1 0.05 0.1\n"
        "/GRNOD/PART/2\nupper all\n2\n"
        "/SURF/PART/1\nlower surf\n1\n"
        f"/INTER/TYPE7/1\nimpact\n2 1 1 0\n{K} 0.0 {gap}\n/END\n")
    model = _starter_only(make_deck, "T7F", starter)
    ct = ContactType7(model.interfaces[0], model, MessageLog())

    # place the upper block's bottom face 0.05 above the lower's top face
    up = model.node_indices([11, 12, 13, 14, 15, 16, 17, 18])
    model.x[up, 2] -= 0.25                    # 1.3 -> 1.05
    fcont = np.zeros_like(model.x)
    dwork, dt_i = ct.forces(model.x, model.v, model.mass, 1e-4, fcont, 0)

    pen = gap - 0.05
    bottom = model.node_indices([11, 12, 13, 14])
    assert fcont[bottom, 2] == pytest.approx(K * pen)     # 0.25 each, up
    assert np.abs(fcont[bottom, :2]).max() < 1e-12
    # exact reaction on the segment corners: total force balances
    assert np.abs(fcont.sum(axis=0)).max() < 1e-12
    # static contact does no work
    assert dwork == 0.0
    # the interface books a finite dt (node-on-spring bound)
    m_min = model.mass.min()
    assert dt_i <= np.sqrt(2 * m_min / K) * (1 + 1e-12)


def test_type7_igap_variable_gap(make_deck):
    """Igap=1: the pair gap is the sum of the two half-thicknesses.
    Main shell t=1.0, secondary shell t=0.5 -> gap = 0.75: a node 0.7
    above the main mid-surface is in contact; with Igap=0 (constant gap
    = main t/2 = 0.5 by default) it is not."""
    base = (
        "/BEGIN\nigap probe\n"
        "/NODE\n"
        "1 0 0 0\n2 10 0 0\n3 10 10 0\n4 0 10 0\n"
        "11 4 4 0.7\n12 6 4 0.7\n13 6 6 0.7\n14 4 6 0.7\n"
        "/SHELL/1\n1 1 2 3 4\n"
        "/SHELL/2\n2 11 12 13 14\n"
        "/PART/1\nmain t=1\n1 1\n"
        "/PART/2\nsec t=0.5\n2 1\n" + STEEL_LAW1 +
        "/PROP/SHELL/1\nmain\n1 0 0 0\n0.01 0.01 0.01 0 0\n3 0 1.0\n"
        "/PROP/SHELL/2\nsec\n1 0 0 0\n0.01 0.01 0.01 0 0\n3 0 0.5\n"
        "/GRNOD/PART/2\nsec nodes\n2\n"
        "/SURF/PART/1\nmain surf\n1\n"
    )
    for igap, expect_contact in ((1, True), (0, False)):
        starter = (base +
                   f"/INTER/TYPE7/1\nprobe\n2 1 0 {igap}\n1.0 0.0 0.0\n"
                   f"/END\n")
        model = _starter_only(make_deck, f"IG{igap}", starter)
        ct = ContactType7(model.interfaces[0], model, MessageLog())
        if igap == 1:
            assert ct.gap_m == pytest.approx(0.5)         # main t/2
            sec = model.node_indices([11, 12, 13, 14])
            loc = np.searchsorted(ct.nodes, sec)
            assert ct.gap_s[loc] == pytest.approx(0.25)   # secondary t/2
        fcont = np.zeros_like(model.x)
        ct.forces(model.x, model.v, model.mass, 1e-4, fcont, 0)
        assert (np.abs(fcont).max() > 0) == expect_contact


def test_type7_self_impact_candidates(make_deck):
    """grnod_ID = 0: the secondary side defaults to the surface's own
    nodes, and a node is never a candidate against its own segments."""
    starter = (
        "/BEGIN\nself impact probe\n"
        "/NODE\n"
        "1 0 0 0\n2 10 0 0\n3 10 10 0\n4 0 10 0\n"
        "11 0 0 0.4\n12 10 0 0.4\n13 10 10 0.4\n14 0 10 0.4\n"
        "/SHELL/1\n1 1 2 3 4\n2 11 12 13 14\n"
        "/PART/1\nboth plates\n1 1\n" + STEEL_LAW1 +
        "/PROP/SHELL/1\nt=0.5\n1 0 0 0\n0.01 0.01 0.01 0 0\n3 0 0.5\n"
        "/SURF/PART/1\nself\n1\n"
        "/INTER/TYPE7/1\nself\n0 1 2 1\n1.0 0.0 0.0\n"
        "/END\n"
    )
    model = _starter_only(make_deck, "SLF", starter)
    ct = ContactType7(model.interfaces[0], model, MessageLog())
    assert len(ct.nodes) == 8                 # all surface nodes tracked
    fcont = np.zeros_like(model.x)
    ct.forces(model.x, model.v, model.mass, 1e-4, fcont, 0)
    # plates 0.4 apart, variable gap = 0.5 -> the two plates push apart...
    assert fcont[model.node_indices([11, 12, 13, 14]), 2].min() > 0
    assert fcont[model.node_indices([1, 2, 3, 4]), 2].max() < 0
    # ...with zero total (every pair is internal action/reaction)
    assert np.abs(fcont.sum(axis=0)).max() < 1e-12


# ============================================================================
# Unit tests — TYPE2 tied kinematics
# ============================================================================

TIED_PATCH = (
    "/BEGIN\ntied patch\n"
    "/NODE\n"
    "1 0 0 0\n2 10 0 0\n3 10 10 0\n4 0 10 0\n"
    "11 4 4 1\n12 6 4 1\n13 6 6 1\n14 4 6 1\n"
    "/SHELL/1\n1 1 2 3 4\n"
    "/SHELL/2\n2 11 12 13 14\n"
    "/PART/1\nmain\n1 1\n"
    "/PART/2\npatch\n2 1\n" + STEEL_LAW1 +
    "/PROP/SHELL/1\nmain\n1 0 0 0\n0.01 0.01 0.01 0 0\n3 0 1.0\n"
    "/PROP/SHELL/2\npatch\n1 0 0 0\n0.01 0.01 0.01 0 0\n3 0 1.0\n"
    "/GRNOD/PART/2\npatch nodes\n2\n"
    "/SURF/PART/1\nmain surf\n1\n"
    "/INTER/TYPE2/1\ntie\n2 1 5.0\n"
    "/END\n"
)


def test_type2_projection_reconstruction(make_deck):
    """The stored weights + local offset reconstruct the secondary node's
    initial position exactly (projection consistency)."""
    model = _starter_only(make_deck, "T2P", TIED_PATCH)
    ct = ContactType2(model.interfaces[0], model, MessageLog())
    assert len(ct.snode) == 4                    # all patch nodes tied
    assert ct.w.sum(axis=1) == pytest.approx(1.0)
    assert ct.off_loc[:, 2] == pytest.approx(1.0)   # normal offset
    # enforce() on the unmoved mesh must reproduce x0 exactly
    x = model.x0.copy()
    v = np.zeros_like(x)
    vr = np.zeros_like(v)
    ct.enforce(x, v, vr, 1.0)
    assert np.abs(x[ct.snode] - model.x0[ct.snode]).max() < 1e-12
    assert np.abs(v[ct.snode]).max() < 1e-12


def test_type2_tied_nodes_follow_rigid_motion(make_deck):
    """Closed form: when the main segment moves rigidly (rotation R +
    translation c), an offset tied node must land exactly on R x0 + c —
    the co-rotating frame handles the offset with no approximation."""
    model = _starter_only(make_deck, "T2R", TIED_PATCH)
    ct = ContactType2(model.interfaces[0], model, MessageLog())

    # rigid motion: 25 deg about a skew axis + translation
    axis = np.array([1.0, 2.0, 0.5])
    axis /= np.linalg.norm(axis)
    th = np.deg2rad(25.0)
    Kx = np.array([[0, -axis[2], axis[1]],
                   [axis[2], 0, -axis[0]],
                   [-axis[1], axis[0], 0]])
    R = np.eye(3) + np.sin(th) * Kx + (1 - np.cos(th)) * (Kx @ Kx)
    c = np.array([3.0, -2.0, 7.0])

    x = model.x0 @ R.T + c
    v = np.zeros_like(x)
    vr = np.zeros_like(v)
    ct.enforce(x, v, vr, 1.0)
    assert np.abs(x[ct.snode]
                  - (model.x0[ct.snode] @ R.T + c)).max() < 1e-9


def test_type2_transfer_conserves_force_and_mass(make_deck):
    """The lumped constraint transfer: total force and total mass are
    conserved exactly, and the secondary rows are zeroed."""
    model = _starter_only(make_deck, "T2C", TIED_PATCH)
    ct = ContactType2(model.interfaces[0], model, MessageLog())

    mass_eff = model.mass.copy()
    ct.augment_mass(mass_eff)
    real = model.mass < 1e29
    assert mass_eff[real].sum() == pytest.approx(
        model.mass[real].sum() + model.mass[ct.snode].sum())

    rng = np.random.default_rng(4)
    fint = rng.normal(size=model.x.shape)
    fext = rng.normal(size=model.x.shape)
    fcont = rng.normal(size=model.x.shape)
    mint = np.zeros_like(fint)
    tot = fint.sum(axis=0) + fext.sum(axis=0) + fcont.sum(axis=0)
    inv = 1.0 / mass_eff
    ct.transfer_forces(fint, fext, fcont, mint, model.x, mass_eff, inv, cycle=1)
    assert (fint.sum(axis=0) + fext.sum(axis=0)
            + fcont.sum(axis=0)) == pytest.approx(tot)
    assert np.abs(fint[ct.snode]).max() == 0.0
    assert np.abs(fext[ct.snode]).max() == 0.0
    assert np.abs(fcont[ct.snode]).max() == 0.0


# ============================================================================
# Unit tests — TYPE11 edge-to-edge closed form
# ============================================================================

def test_type11_penalty_force_closed_form(make_deck):
    """Two perpendicular edges (trusses) at distance 0.05 with Istf=1,
    gap 0.1: the closest points are both midpoints, so each end node gets
    exactly F/2 = K*p/2, the reactions balance and torque about the
    contact point is zero (collinear forces)."""
    K, gap = 5.0, 0.1
    starter = (
        "/BEGIN\ncrossed edges\n"
        "/NODE\n"
        "1 -1 0 0.05\n2 1 0 0.05\n"       # edge A along x, z = 0.05
        "3 0 -1 0\n4 0 1 0\n"             # edge B along y, z = 0
        "/TRUSS/1\n1 1 2\n2 3 4\n"
        "/PART/1\ntruss\n1 1\n" + STEEL_LAW1 +
        "/PROP/TRUSS/1\ntruss\n1.0\n"
        "/LINE/SEG/1\nedge A\n1 2\n"
        "/LINE/SEG/2\nedge B\n3 4\n"
        f"/INTER/TYPE11/1\ncross\n1 2 1 0\n{K} 0.0 {gap}\n"
        "/END\n"
    )
    model = _starter_only(make_deck, "T11F", starter)
    ct = ContactType11(model.interfaces[0], model, MessageLog())
    fcont = np.zeros_like(model.x)
    dwork, dt_i = ct.forces(model.x, model.v, model.mass, 1e-4, fcont, 0)

    F = K * (gap - 0.05)
    a = model.node_indices([1, 2])
    b = model.node_indices([3, 4])
    assert fcont[a, 2] == pytest.approx(F / 2)     # up, half per end
    assert fcont[b, 2] == pytest.approx(-F / 2)    # down on the main edge
    assert np.abs(fcont[:, :2]).max() < 1e-12
    assert np.abs(fcont.sum(axis=0)).max() < 1e-12
    assert dwork == 0.0
    assert np.isfinite(dt_i)


# ============================================================================
# Unit tests — element deletion vs contact (M3<->M4)
# ============================================================================

DELETABLE_BAR = (
    "/BEGIN\ndeletable bar\n"
    "/NODE\n"
    "1 0 0 0\n2 1 0 0\n3 1 1 0\n4 0 1 0\n"
    "5 0 0 1\n6 1 0 1\n7 1 1 1\n8 0 1 1\n"
    "9 0 0 2\n10 1 0 2\n11 1 1 2\n12 0 1 2\n"
    "/BRICK/1\n1 1 2 3 4 5 6 7 8\n2 5 6 7 8 9 10 11 12\n"
    "/PART/1\nbar\n1 1\n"
    "/MAT/LAW36/1\ntabulated steel\n7.8e-6\n210. 0.3\n1 0\n11\n"
    "/FUNCT/11\nhardening\n0.0 0.4\n1.0 1.4\n"
    "/FAIL/JOHNSON/1\n0.10 0 0 0 0\n"
    "/PROP/SOLID/1\nsolid\n1.1 0.05 0.1\n"
    "/SURF/PART/1\nbar surf\n1\n"
    "/GRNOD/NODE/1\ntip\n9 10 11 12\n"
    "/INTER/TYPE7/1\nself\n1 1 0 0\n1.0 0.0 0.05\n"
    "/END\n"
)


def test_deletion_drops_segments_and_nodes(make_deck):
    """Deleting element 1 (off=0) must (a) kill its 5 free faces in the
    surface mask while element 2's faces survive, and (b) untrack the 4
    nodes that belonged ONLY to element 1, while the shared face nodes
    (still in element 2) stay tracked."""
    model = _starter_only(make_deck, "DEL", DELETABLE_BAR)
    surf = model.surfaces[1]
    assert len(surf.segments) == 10           # 2*6 - 2 shared

    assert tracking.any_deletable(model, surf.seg_gtype)
    ref = tracking.node_reference_counts(model, alive_only=False)

    model.bricks.state["off"][0] = 0.0        # kill the lower element
    alive = tracking.alive_segment_mask(model, surf.seg_gtype,
                                        surf.seg_elem)
    assert alive.sum() == 5                   # only element 2's faces left
    dead_rows = surf.seg_elem[~alive]
    assert np.all(dead_rows == 0)

    tracked = tracking.tracked_node_mask(model, ref)
    only_e1 = model.node_indices([1, 2, 3, 4])
    shared = model.node_indices([5, 6, 7, 8])
    upper = model.node_indices([9, 10, 11, 12])
    assert not tracked[only_e1].any()
    assert tracked[shared].all()
    assert tracked[upper].all()


def test_type7_forces_vanish_on_deleted_segments(make_deck):
    """End-to-end mask usage: a contact pair pushing at cycle 0 must stop
    pushing the moment its segment's parent element is deleted, even
    between broad-phase refreshes."""
    model = _starter_only(make_deck, "DELF", DELETABLE_BAR)
    ct = ContactType7(model.interfaces[0], model, MessageLog())
    # move the tip nodes close over the top face: contact active
    tip = model.node_indices([9, 10, 11, 12])
    model.x[tip, 2] = 2.0 + 0.02              # 0.02 above... their own top
    # (tip nodes belong to element 2's own top face -> excluded; bring
    # them near element 1's BOTTOM face instead for a clean pair)
    model.x[tip, 2] = -0.02
    fcont = np.zeros_like(model.x)
    ct.forces(model.x, model.v, model.mass, 1e-4, fcont, 0)
    assert np.abs(fcont).max() > 0.0

    model.bricks.state["off"][0] = 0.0        # the crack opens
    fcont[:] = 0.0
    ct.forces(model.x, model.v, model.mass, 1e-4, fcont, 1)  # no refresh
    assert np.abs(fcont).max() == 0.0


# ============================================================================
# Analytic validations — full starter+engine runs
# ============================================================================

def _two_bar_tied_deck(nel=10):
    """Two coaxial 1x1 brick bars, bar B's start face nodes tied to bar
    A's surface (conforming, coincident nodes with distinct ids)."""
    nodes, bricks = [], []
    nid = 0
    # bar A: x in [0, nel]
    for i in range(nel + 1):
        for (y, z) in ((0, 0), (1, 0), (1, 1), (0, 1)):
            nid += 1
            nodes.append(f"{nid} {float(i)} {float(y)} {float(z)}")
    for e in range(nel):
        b = 4 * e
        bricks.append((1, f"{e + 1} {b + 1} {b + 2} {b + 3} {b + 4} "
                          f"{b + 5} {b + 6} {b + 7} {b + 8}"))
    ofs = nid
    # bar B: x in [nel, 2 nel], nodes duplicated at the junction plane
    for i in range(nel + 1):
        for (y, z) in ((0, 0), (1, 0), (1, 1), (0, 1)):
            nid += 1
            nodes.append(f"{nid} {float(nel + i)} {float(y)} {float(z)}")
    for e in range(nel):
        b = ofs + 4 * e
        bricks.append((2, f"{nel + e + 1} {b + 1} {b + 2} {b + 3} {b + 4} "
                          f"{b + 5} {b + 6} {b + 7} {b + 8}"))
    tie_nodes = " ".join(str(ofs + k) for k in (1, 2, 3, 4))
    all_nodes = " ".join(str(i) for i in range(1, nid + 1))
    return nodes, bricks, tie_nodes, all_nodes, nid


def test_type2_wave_transmission_zero_contact_energy(make_deck):
    """The M1 P-wave bar, cut in two and glued back with /INTER/TYPE2:
    the front must cross the tie at the analytic speed with no spurious
    reflection — and the tie must do NO work (contact energy exactly
    zero, balance closes without it)."""
    nel = 10
    nodes, bricks, tie_nodes, all_nodes, nid = _two_bar_tied_deck(nel)
    b1 = "\n".join(s for p, s in bricks if p == 1)
    b2 = "\n".join(s for p, s in bricks if p == 2)
    starter = (
        "/BEGIN\ntied wave bar\n"
        "/NODE\n" + "\n".join(nodes) + "\n"
        "/BRICK/1\n" + b1 + "\n"
        "/BRICK/2\n" + b2 + "\n"
        "/PART/1\nbar A\n1 1\n"
        "/PART/2\nbar B\n1 1\n" + STEEL_LAW1 +
        "/PROP/SOLID/1\nsolid\n1.1 0.05 0.1\n"
        "/GRNOD/NODE/1\ndriven end\n1 2 3 4\n"
        f"/GRNOD/NODE/2\nall\n{all_nodes}\n"
        f"/GRNOD/NODE/3\ntied face\n{tie_nodes}\n"
        "/BCS/1\nconfine lateral\n011 111 0 2\n"
        "/FUNCT/1\nstep velocity\n0.0 1.0\n100.0 1.0\n"
        "/IMPVEL/1\npush\n1 X 1 0.01\n"
        "/SURF/PART/1\nbar A surface\n1\n"
        "/INTER/TYPE2/1\nglue\n3 1 0\n"
        "/END\n"
    )
    K, G, rho = 210 / (3 * (1 - 0.6)), 210 / 2.6, 7.8e-6
    c = np.sqrt((K + 4 * G / 3) / rho)          # 6021.3 mm/ms
    t_end = 1.4 * nel / c                        # front at x ~ 14: inside B
    engine = f"/RUN/TWB/1\n{t_end}\n/DT\n0.67 0\n/PRINT/-1000\n/STOP\n90.0\n"
    model, out = _run(make_deck, "TWB", starter, engine)

    front = c * t_end
    vx = model.v[:, 0]
    x0 = model.x0[:, 0]
    behind = x0 < front - 2.5
    ahead = x0 > front + 2.5
    assert vx[behind].min() > 0.005      # the wave crossed the tie
    assert np.abs(vx[ahead]).max() < 1e-4
    s = _final_summary(out)
    assert s["NORMAL"]
    assert s["CE"] == 0.0                # tied contact does no work. Ever.
    # the wave setup itself carries a small step-shock bookkeeping error
    # (~-2.4%, identical in the M1 continuous-bar test); the tie must add
    # NOTHING on top of it
    assert abs(s["ERR"]) < 3.0


def test_type2_tied_assembly_rigid_flight(make_deck):
    """A tied two-bar assembly in uniform flight: the tie must not
    disturb rigid motion (no stress, exact velocity, zero contact
    energy) — the tied analogue of the M1 objectivity check."""
    nel = 3
    nodes, bricks, tie_nodes, all_nodes, nid = _two_bar_tied_deck(nel)
    b1 = "\n".join(s for p, s in bricks if p == 1)
    b2 = "\n".join(s for p, s in bricks if p == 2)
    starter = (
        "/BEGIN\ntied flight\n"
        "/NODE\n" + "\n".join(nodes) + "\n"
        "/BRICK/1\n" + b1 + "\n"
        "/BRICK/2\n" + b2 + "\n"
        "/PART/1\nbar A\n1 1\n"
        "/PART/2\nbar B\n1 1\n" + STEEL_LAW1 +
        "/PROP/SOLID/1\nsolid\n1.1 0.05 0.1\n"
        f"/GRNOD/NODE/2\nall\n{all_nodes}\n"
        f"/GRNOD/NODE/3\ntied face\n{tie_nodes}\n"
        "/SURF/PART/1\nbar A surface\n1\n"
        "/INTER/TYPE2/1\nglue\n3 1 0\n"
        "/INIVEL/TRA/1\nfly\n2.0 1.0 0.5 2\n"
        "/END\n"
    )
    engine = "/RUN/TFL/1\n0.5\n/DT\n0.9 0\n/PRINT/-1000\n/STOP\n5.0\n"
    model, out = _run(make_deck, "TFL", starter, engine)
    assert np.abs(model.bricks.state["sig"]).max() < 1e-12
    real = model.mass < 1e29
    assert np.abs(model.v[real] - np.array([2.0, 1.0, 0.5])).max() < 1e-10
    assert model.x[0, 0] == pytest.approx(2.0 * 0.5, rel=1e-9)
    s = _final_summary(out)
    assert s["CE"] == 0.0
    assert abs(s["ERR"]) < 1e-6


def test_type11_edge_impact_momentum_and_energy(make_deck):
    """Two free crossing shell strips, the upper flying down: contact is
    purely edge-to-edge at the crossing. Momentum must be conserved
    exactly (internal forces only), the strips must not cross, and a
    full /DT 0.9 run must stay stable with a tight balance."""
    starter = (
        "/BEGIN\nedge impact\n"
        "/NODE\n"
        # main strip along x (3 elements, width 2, z = 0)
        "1 -3 -1 0\n2 -1 -1 0\n3 1 -1 0\n4 3 -1 0\n"
        "5 -3 1 0\n6 -1 1 0\n7 1 1 0\n8 3 1 0\n"
        # secondary strip along y (3 elements, width 2, z = 1.2)
        "11 -1 -3 1.2\n12 -1 -1 1.2\n13 -1 1 1.2\n14 -1 3 1.2\n"
        "15 1 -3 1.2\n16 1 -1 1.2\n17 1 1 1.2\n18 1 3 1.2\n"
        "/SHELL/1\n1 1 2 6 5\n2 2 3 7 6\n3 3 4 8 7\n"
        "/SHELL/2\n11 11 12 16 15\n12 12 13 17 16\n13 13 14 18 17\n"
        "/PART/1\nmain strip\n1 1\n"
        "/PART/2\nflyer strip\n2 1\n" + STEEL_LAW1 +
        "/PROP/SHELL/1\nstrip\n1 0 0 0\n0.01 0.01 0.01 0 0\n3 0 0.6\n"
        "/PROP/SHELL/2\nstrip\n1 0 0 0\n0.01 0.01 0.01 0 0\n3 0 0.6\n"
        "/GRNOD/PART/2\nflyer\n2\n"
        "/INIVEL/TRA/1\nfall\n0 0 -1.0 2\n"
        "/SURF/PART/1\nmain surf\n1\n"
        "/SURF/PART/2\nflyer surf\n2\n"
        "/LINE/SURF/1\nflyer edges\n2\n"
        "/LINE/SURF/2\nmain edges\n1\n"
        "/INTER/TYPE11/1\nedges\n1 2 2 1\n1.0 0.0 0.0\n"
        "/END\n"
    )
    engine = "/RUN/E11/1\n3.0\n/DT\n0.9 0\n/PRINT/-100000\n/STOP\n20.0\n"
    model, out = _run(make_deck, "E11", starter, engine)

    real = model.mass < 1e29
    pz = float((model.mass[real] * model.v[real, 2]).sum())
    flyer = model.node_indices([11, 12, 13, 14, 15, 16, 17, 18])
    m_flyer = model.mass[flyer].sum()
    assert pz == pytest.approx(-m_flyer * 1.0, rel=1e-9)  # exact momentum
    # the flyer's crossing edge stayed above the main strip (gap = 0.6,
    # both mid-surfaces never approach closer than a fraction of it)
    flyer_mid = model.node_indices([12, 13, 16, 17])
    main_mid = model.node_indices([2, 3, 6, 7])
    assert model.x[flyer_mid, 2].min() > model.x[main_mid, 2].max() + 0.2
    # The flyer bounced/arrested: it no longer approaches at full speed.
    # Measured on the flyer's CENTRE OF MASS, not on the four crossing
    # nodes.  Contact releases well before TSTOP and leaves those nodes
    # ringing in the strip's first bending mode with a ~+-0.28 swing about
    # the drift, so their instantaneous mean samples the ring PHASE at
    # t = 3.0, not the arrest.  That phase moves with ANY change to the
    # shell's bending frequency — notably the M41 cinmas.F FAC = 9/12
    # rotational-inertia family split (l.919-924: BT keeps AREA/9) — and
    # the -0.5 threshold sits inside the swing for BOTH lumpings: sampled
    # every 0.02 over t = 2.84..3.00 the four-node mean dips below -0.5 at
    # 2/9 stop times with FAC = 9 and 3/9 with FAC = 12, so the old form
    # was passing at t = 3.0 by coincidence of phase.  The CoM velocity IS
    # the arrest, and once the interface releases it is exact to round-off
    # (ptp 3e-16 across that window, -0.36865 under both lumpings).
    mf = model.mass[flyer]
    vz_com = float((mf * model.v[flyer, 2]).sum() / mf.sum())
    assert vz_com > -0.5
    # ...and the residual ringing stays bounded — measured envelope over
    # the same window is -0.95..+0.26, so a real instability blows past
    # this while the mode-shape phase never does
    assert np.abs(model.v[flyer, 2]).max() < 2.0
    s = _final_summary(out)
    assert s["NORMAL"]
    assert abs(s["ERR"]) < 5.0


def test_type7_self_impact_full_run(make_deck):
    """Self-impact (grnod=0): two parallel plates of the SAME part and
    surface, the upper one falling; the single-surface interface must
    arrest it (no crossing) with a clean balance at /DT 0.9."""
    nodes, shells = [], []
    nid = 0
    for z, base in ((0.0, 0), (0.8, 100)):
        for j in range(3):
            for i in range(3):
                nodes.append(f"{base + 1 + i + 4 * j} "
                             f"{5.0 * i} {5.0 * j} {z}")
    for base, eb in ((0, 0), (100, 10)):
        for j in range(2):
            for i in range(2):
                n1 = base + 1 + i + 4 * j
                shells.append(f"{eb + 1 + i + 2 * j} {n1} {n1 + 1} "
                              f"{n1 + 5} {n1 + 4}")
    upper = " ".join(str(100 + 1 + i + 4 * j)
                     for j in range(3) for i in range(3))
    starter = (
        "/BEGIN\nself impact plates\n"
        "/NODE\n" + "\n".join(nodes) + "\n"
        "/SHELL/1\n" + "\n".join(shells) + "\n"
        "/PART/1\nboth plates\n1 1\n" + STEEL_LAW1 +
        "/PROP/SHELL/1\nt=0.4\n1 0 0 0\n0.01 0.01 0.01 0 0\n3 0 0.4\n"
        f"/GRNOD/NODE/1\nupper plate\n{upper}\n"
        "/INIVEL/TRA/1\nfall\n0 0 -0.5 1\n"
        "/SURF/PART/1\nself\n1\n"
        "/INTER/TYPE7/1\nself\n0 1 2 1\n1.0 0.0 0.0\n"
        "/END\n"
    )
    engine = "/RUN/SIMP/1\n2.0\n/DT\n0.9 0\n/PRINT/-100000\n/STOP\n20.0\n"
    model, out = _run(make_deck, "SIMP", starter, engine)
    lower = model.node_indices([1 + i + 4 * j
                                for j in range(3) for i in range(3)])
    upperi = model.node_indices([100 + 1 + i + 4 * j
                                 for j in range(3) for i in range(3)])
    # variable gap = 0.4 (t/2 + t/2): mid-surfaces never cross, and stay
    # separated by a decent fraction of the gap
    assert model.x[upperi, 2].min() > model.x[lower, 2].max() - 1e-9
    assert model.v[upperi, 2].mean() > -0.4      # arrested / bouncing
    s = _final_summary(out)
    assert s["NORMAL"]
    assert abs(s["ERR"]) < 5.0


def _meshed_cube(base_id, eid0, part, origin, n=2, size=1.0):
    """Nodes + bricks of an n^3-element cube (node/element card strings).
    Multi-element cubes matter for the stability tests: a SINGLE 1-point
    brick pounded on its corners stores a large fraction of the impact in
    hourglass modes and can then go unstable on the ELEMENT side (its
    exact-dt factor is precomputed on the undistorted geometry — an M1/M2
    documented limitation), which would masquerade as a contact failure."""
    h = size / n
    nodes, bricks = [], []
    def nid(i, j, k):
        return base_id + i + (n + 1) * (j + (n + 1) * k)
    for k in range(n + 1):
        for j in range(n + 1):
            for i in range(n + 1):
                nodes.append(f"{nid(i, j, k)} {origin[0] + i * h} "
                             f"{origin[1] + j * h} {origin[2] + k * h}")
    eid = eid0
    for k in range(n):
        for j in range(n):
            for i in range(n):
                c = [nid(i, j, k), nid(i + 1, j, k), nid(i + 1, j + 1, k),
                     nid(i, j + 1, k), nid(i, j, k + 1), nid(i + 1, j, k + 1),
                     nid(i + 1, j + 1, k + 1), nid(i, j + 1, k + 1)]
                bricks.append(f"{eid} " + " ".join(str(v) for v in c))
                eid += 1
    base_nodes = [nid(i, j, 0) for j in range(n + 1) for i in range(n + 1)]
    return nodes, bricks, base_nodes


@pytest.mark.parametrize("istf,igap", [
    (0, 0),
    # the (2, 1) variant measured 81.3 s serial (2026-07 full-suite run)
    pytest.param(2, 1, marks=pytest.mark.slow),
    (5, 1),
])
def test_type7_impact_stability_dt09_variants(make_deck, istf, igap):
    """The two-cube impact of M1 (2x2x2-meshed cubes), run LONG at /DT 0.9
    for every stiffness variant class: the interface dt bound must keep
    every variant stable (an unstable penalty spring blows up within a
    few hundred cycles)."""
    n_lo, b_lo, base_lo = _meshed_cube(1, 1, 1, (0.0, 0.0, 0.0))
    n_up, b_up, _ = _meshed_cube(101, 101, 2, (0.0, 0.0, 1.3))
    starter = (
        "/BEGIN\ntwo meshed cubes\n"
        "/NODE\n" + "\n".join(n_lo + n_up) + "\n"
        "/BRICK/1\n" + "\n".join(b_lo) + "\n"
        "/BRICK/2\n" + "\n".join(b_up) + "\n"
        "/PART/1\nlower\n1 1\n"
        "/PART/2\nupper\n2 1\n" + STEEL_LAW1 +
        "/PROP/SOLID/1\nsolid\n1.1 0.05 0.1\n"
        "/PROP/SOLID/2\nsolid\n1.1 0.05 0.1\n"
        "/GRNOD/PART/2\nupper all\n2\n"
        "/SURF/PART/1\nlower surf\n1\n"
        f"/GRNOD/NODE/1\nbase\n{' '.join(str(v) for v in base_lo)}\n"
        "/BCS/1\nclamp base\n111 111 0 1\n"
        "/INIVEL/TRA/1\nfall\n0 0 -1.0 2\n"
        f"/INTER/TYPE7/1\nimpact\n2 1 {istf} {igap}\n1.0 0.0 0.1\n"
        "/END\n"
    )
    engine = "/RUN/S7/1\n2.0\n/DT\n0.9 0\n/PRINT/-100000\n/STOP\n14.9\n"
    model, out = _run(make_deck, "S7", starter, engine)
    upper_bottom = model.node_indices(
        [101 + i + 3 * j for j in range(3) for i in range(3)])
    lower_top = model.node_indices(
        [1 + i + 3 * (j + 3 * 2) for j in range(3) for i in range(3)])
    assert np.all(np.isfinite(model.v))
    # never crossed (allow a fraction of the 0.1 gap as working pen)
    assert model.x[upper_bottom, 2].min() > \
        model.x[lower_top, 2].max() - 0.09
    # bounced away or resting, not accelerating through
    assert model.v[upper_bottom, 2].mean() > -1.0
    s = _final_summary(out)
    assert s["NORMAL"]
    assert abs(s["ERR"]) < 6.0


@pytest.mark.slow          # measured 165.2 s serial (2026-07 full-suite run)
def test_notched_plate_crack_with_self_contact(make_deck):
    """The M3<->M4 flagship: the notched-plate tearing model WITH a
    self-impact TYPE7 on the plate. As /FAIL/BIQUAD deletes elements,
    their segments must drop out of the contact surface and fully-dead
    nodes must stop being tracked — otherwise the crack faces keep
    pulling/pushing and the balance degrades. The run must tear the
    ligament and finish NORMAL with a tight balance."""
    NX, NY, NOTCH = 12, 6, 2
    def nd(i, j):
        return 1 + i + j * (NX + 1)
    nodes = [f"{nd(i, j)} {i * 1.0} {j * 1.0} 0.0"
             for j in range(NY + 1) for i in range(NX + 1)]
    shells, eid = [], 0
    jn = NY // 2
    for j in range(NY):
        for i in range(NX):
            eid += 1
            if j == jn and (i < NOTCH or i >= NX - NOTCH):
                continue
            shells.append(f"{eid} {nd(i, j)} {nd(i + 1, j)} "
                          f"{nd(i + 1, j + 1)} {nd(i, j + 1)}")
    bottom = " ".join(str(nd(i, 0)) for i in range(NX + 1))
    top = " ".join(str(nd(i, NY)) for i in range(NX + 1))
    starter = (
        "/BEGIN\nnotched plate + self contact\n"
        "/NODE\n" + "\n".join(nodes) + "\n"
        "/SHELL/1\n" + "\n".join(shells) + "\n"
        "/PART/1\nplate\n1 1\n"
        "/MAT/LAW36/1\nmild steel\n7.8e-6\n210. 0.3\n1 0\n11\n"
        "/FUNCT/11\nhardening\n0.0 0.25\n0.4 0.40\n10.0 0.40\n"
        "/FAIL/BIQUAD/1\n0.60 0.45 0.35 0.25 0.30\n"
        "/PROP/SHELL/1\nplate t=1\n1 0 0 0\n0.01 0.01 0.01 0 0\n3 0 1.0\n"
        f"/GRNOD/NODE/1\nbottom\n{bottom}\n"
        f"/GRNOD/NODE/2\ntop\n{top}\n"
        "/BCS/1\nclamp bottom\n111 111 0 1\n"
        "/FUNCT/1\nramp\n0.0 0.0\n0.02 1.0\n100.0 1.0\n"
        "/IMPVEL/1\npull\n1 Y 2 2.0\n"
        "/SURF/PART/1\nplate\n1\n"
        "/INTER/TYPE7/1\nself contact\n0 1 2 1\n1.0 0.0 0.0\n"
        "/END\n"
    )
    engine = "/RUN/NCC/1\n1.0\n/DT\n0.9 0\n/PRINT/-100000\n/STOP\n15.0\n"
    model, out = _run(make_deck, "NCC", starter, engine)
    off = model.shells.state["off"]
    assert (off == 0.0).sum() >= NX - 2 * NOTCH   # the ligament tore
    s = _final_summary(out)
    assert s["NORMAL"]
    assert abs(s["ERR"]) < 5.0
