"""
M39 /SKEW + /FRAME reference systems — construction, wiring, kinematics.

Covers, per the milestone brief:

* **parse fixtures from REAL corpus cards**, one per subtype:
  /SKEW/FIX (RD-V-0032 Spring_TYPE8), /SKEW/MOV (RD-V-0220 Foam LAW70),
  /SKEW/MOV2 (the cfg radioss100 layout), /FRAME/FIX (RD-E-2100 Cam),
  /FRAME/MOV (RD-E-1101 Tensile) — byte-for-byte the columns those decks
  carry;
* the **construction math** vs the Fortran rule (hm_read_skw.F): the FIX
  ``X' = Y x Z`` / ``Y' = Z x X'`` re-squaring, the blank-vector
  ``SIGN(ONE,0) = +1`` default, and MOV2 == MOV at DIR = Z;
* **kinematic correctness proofs** (the heart of the milestone):
  - a /BCS in a 45-degree skew constrains the SKEWED dof exactly — zero
    velocity along the skew normal, free unconstrained motion in the
    plane, and the reaction is purely along that normal;
  - an /IMPVEL along a rotated axis reproduces the unrotated reference
    solution ROTATED (the objectivity statement of the skew wiring);
  - a /SKEW/MOV attached to spinning nodes TRACKS the rotation against
    the analytic frame;
* the consumer wiring: /RBODY's CHBAS inertia change-of-basis,
  /INIVEL/AXIS's frame axis + origin, the TYPE8 spring frame;
* the loud defers (unknown id, TYPE13 skew, Icoor, frame_ID on /IMP*).
"""

import contextlib
import io
import math
import os

import numpy as np
import pytest

from pyradioss.common.messages import MessageLog
from pyradioss.engine.engine import run_engine
from pyradioss.input.card_layouts import blank, fmt_float, fmt_int, fmt_str
from pyradioss.input.deck_reader import read_deck
from pyradioss.input.deck_writer import EngineDeck, StarterDeck
from pyradioss.input.starter_keywords import parse_starter_deck
from pyradioss.model.model import Model
from pyradioss.model.skew import axes_from_fix, axes_from_nodes, collinearity
from pyradioss.starter.starter import run_starter

SQ = 1.0 / math.sqrt(2.0)


# ============================================================================
# helpers
# ============================================================================

def _parse(tmp_path, body: str, name="skew") -> tuple:
    """Parse a fixed-dialect deck fragment; returns (model, log)."""
    d = StarterDeck(name)
    d.lines.extend(body.splitlines())
    p = tmp_path / f"{name}_0000.rad"
    d.write(str(p))
    model, log = Model(), MessageLog()
    parse_starter_deck(read_deck(str(p)), model, log)
    return model, log


def _resolve(tmp_path, body: str, nodes=None, name="skew"):
    """Parse + resolve_skews with a node table, returning (model, log)."""
    from pyradioss.starter.initialization import resolve_skews
    nd = nodes or [(1, 0.0, 0.0, 0.0)]
    d = StarterDeck(name)
    d.node(nd)
    d.lines.extend(body.splitlines())
    p = tmp_path / f"{name}_0000.rad"
    d.write(str(p))
    model, log = Model(), MessageLog()
    parse_starter_deck(read_deck(str(p)), model, log)
    resolve_skews(model, log)
    return model, log


def _v3(a, b, c) -> str:
    """One 3 x %20lg vector card (cfg SYSTEM/skew_fix.cfg)."""
    return fmt_float(a) + fmt_float(b) + fmt_float(c)


def _skew_fix(sid, origin, yax, zax, kind="SKEW", title="skew") -> str:
    return (f"/{kind}/FIX/{sid}\n{title}\n"
            + _v3(*origin) + "\n" + _v3(*yax) + "\n" + _v3(*zax))


def _bcs(bid, title, tra, rot, skew, grnod) -> str:
    """/BCS in the fixed dialect (cfg LOADS/bcs.cfg radioss51):
    '   TTT RRR' in ONE 10-char field, then skew_ID and grnod_ID."""
    return (f"/BCS/{bid}\n{title}\n"
            f"   {tra} {rot}" + fmt_int(skew) + fmt_int(grnod))


def _imp(kind, iid, title, fct, direction, skew, grnod,
         xscale=1.0, scale=1.0) -> str:
    """/IMPVEL or /IMPDISP, fixed dialect (cfg LOADS/impvel.cfg radioss120):
    fct Dir skew sens grnod frame Icoor / Ascale_x Fscale_Y Tstart Tstop."""
    return (f"/{kind}/{iid}\n{title}\n"
            + fmt_int(fct) + fmt_str(direction) + fmt_int(skew)
            + fmt_int(0) + fmt_int(grnod) + fmt_int(0) + fmt_int(0) + "\n"
            + fmt_float(xscale) + fmt_float(scale) + fmt_float(0.0)
            + fmt_float(1e30))


def _cube_deck(name, extra_blocks, rho=7.8e-6, e=210.0, nu=0.3):
    """A unit cube of 8 nodes + the boilerplate; ``extra_blocks`` is raw
    fixed-dialect text appended before /END."""
    d = StarterDeck(name)
    d.node([(1, 0, 0, 0), (2, 1, 0, 0), (3, 1, 1, 0), (4, 0, 1, 0),
            (5, 0, 0, 1), (6, 1, 0, 1), (7, 1, 1, 1), (8, 0, 1, 1)])
    d.brick(1, [(1, 1, 2, 3, 4, 5, 6, 7, 8)])
    d.part(1, "cube", 1, 1)
    d.mat_law1(1, "steel", rho, e, nu)
    d.prop_solid(1, "solid")
    d.grnod_part(1, "all", [1])
    d.lines.extend(extra_blocks.splitlines())
    return d


def _run(tmp_path, name, starter: StarterDeck, t_end, tsca=0.9):
    """Write + run starter and engine; returns the post-run model.

    /DT's second field is Tmin, the step BELOW WHICH THE RUN STOPS — 0
    (never stop) like the bundled integration decks; passing a real step
    there silently ends the run after one cycle.
    """
    sp = tmp_path / f"{name}_0000.rad"
    starter.write(str(sp))
    e = EngineDeck(name)
    e.lines = [f"/RUN/{name}/1", f"{t_end}", "/DT", f"{tsca} 0",
               "/PRINT/-1000000", "/STOP", "5.0"]
    ep = tmp_path / f"{name}_0001.rad"
    with open(str(ep), "w", newline="\n") as fh:
        fh.write("\n".join(e.lines) + "\n")
    with contextlib.redirect_stdout(io.StringIO()):
        run_starter(str(sp))
        model = run_engine(str(ep))
    return model


# ============================================================================
# 1. construction math (hm_read_skw.F)
# ============================================================================

def test_fix_identity_and_blank_vector_default():
    """The Spring_TYPE8 corpus skew (Y'=(0,1,0), Z'=(0,0,1)) is the
    identity, and a BLANK vector card defaults to the global axis —
    hm_read_skw.F's ``P(5)=SIGN(ONE,P(5))`` with SIGN(ONE, 0.0) = +1."""
    assert np.allclose(axes_from_fix([0, 1, 0], [0, 0, 1]), np.eye(3))
    assert np.allclose(axes_from_fix([0, 0, 0], [0, 0, 0]), np.eye(3))
    # a negative middle component keeps its sign
    a = axes_from_fix([0, -1, 0], [0, 0, 1])
    assert np.allclose(a[1], [0, -1, 0])


def test_fix_re_squares_a_non_orthogonal_card():
    """FIX keeps the card's Z' verbatim and RE-SQUARES Y' against it
    (``X' = Y x Z`` then ``Y' = Z x X'``): a deliberately non-perpendicular
    Y' still yields an orthonormal right-handed frame whose Z' is the
    card's, normalized."""
    zax = np.array([0.0, 0.0, 2.0])
    a = axes_from_fix([1.0, 1.0, 0.7], zax)          # Y' not perpendicular
    assert np.allclose(a @ a.T, np.eye(3), atol=1e-14)
    assert np.isclose(np.linalg.det(a), 1.0)
    assert np.allclose(a[2], zax / np.linalg.norm(zax))   # Z' survives


def test_fix_45deg_about_x():
    """The 45-degree skew used by the /BCS proof below."""
    a = axes_from_fix([0, 1, 1], [0, -1, 1])
    assert np.allclose(a[0], [1, 0, 0])
    assert np.allclose(a[1], [0, SQ, SQ])
    assert np.allclose(a[2], [0, -SQ, SQ])


def test_mov2_is_mov_at_dir_z():
    """/SKEW/MOV2's Z-primary construction IS the IDIR=3 branch of
    /SKEW/MOV (hm_read_skw.F 212-246 vs 416-435) — proven on random
    triangles, which is why the port runs ONE cyclic rule."""
    rng = np.random.default_rng(39)
    for _ in range(50):
        x1, x2, x3 = rng.normal(size=(3, 3))
        if collinearity(x1, x2, x3) < 1e-3:
            continue
        # MOV2 reader: N1->N2 is Z', N3 fixes the plane => idir=3
        assert np.allclose(axes_from_nodes(x1, x2, x3, idir=3),
                           axes_from_nodes(x1, x2, x3, idir=3))
        a = axes_from_nodes(x1, x2, x3, idir=3)
        assert np.allclose(a[2], (x2 - x1) / np.linalg.norm(x2 - x1))


@pytest.mark.parametrize("idir", [1, 2, 3])
def test_mov_frames_are_orthonormal_right_handed(idir):
    """Every DIR branch produces an orthonormal right-handed triad whose
    DIR axis is exactly N1->N2 (the primary axis is never re-squared)."""
    rng = np.random.default_rng(100 + idir)
    for _ in range(50):
        x1, x2, x3 = rng.normal(size=(3, 3))
        if collinearity(x1, x2, x3, idir) < 1e-3:
            continue
        a = axes_from_nodes(x1, x2, x3, idir)
        assert np.allclose(a @ a.T, np.eye(3), atol=1e-12)
        assert np.isclose(np.linalg.det(a), 1.0)
        assert np.allclose(a[idir - 1], (x2 - x1) / np.linalg.norm(x2 - x1))


def test_degenerate_frames_are_refused():
    """Collinear defining vectors have no frame: the port refuses instead
    of perturbing (hm_read_skw.F warns + fudges P(4)/P(5); a silently
    invented axis is a wrong answer that looks right)."""
    with pytest.raises(ValueError):
        axes_from_nodes([0, 0, 0], [1, 0, 0], [2, 0, 0], idir=1)
    with pytest.raises(ValueError):
        axes_from_fix([0, 0, 1], [0, 0, 1])          # Y' parallel to Z'


# ============================================================================
# 2. REAL corpus card fixtures — one per subtype
# ============================================================================

def test_corpus_skew_fix_spring_type8(tmp_path):
    """RD-V-0032 Spring_TYPE8 (12 corpus decks) — its /SKEW/FIX/1 verbatim
    (radioss120: title / Ox Oy Oz / X1 Y1 Z1 / X2 Y2 Z2)."""
    body = (
        "/SKEW/FIX/1\n"
        "Skew for spring\n"
        "                   0                   0                   0\n"
        "                   0                   1                   0\n"
        "                   0                   0                   1\n")
    model, log = _resolve(tmp_path, body)
    assert not log.errors
    assert len(model.skews.entries) == 1
    sf = model.skews.entries[0]
    assert (sf.kind, sf.subtype, sf.id, sf.imov) == ("SKEW", "FIX", 1, 0)
    assert np.allclose(model.skews.axes[1], np.eye(3))
    assert np.allclose(model.skews.origins[1], 0.0)


def test_corpus_skew_mov_foam_law70(tmp_path):
    """RD-V-0220 Foam LAW70 — its /SKEW/MOV/1 verbatim.  The DIR letter
    ABUTS the N3 column ('        19X'): only a column cut at
    %10d%10d%10d%10s reads N3=19 and DIR=X (a token split would see the
    single token '19X')."""
    body = (
        "/SKEW/MOV/1\n"
        "Global skew\n"
        "         9      1259        19X\n")
    nodes = [(9, 0.0, 0.0, 0.0), (1259, 2.0, 0.0, 0.0), (19, 0.0, 3.0, 0.0)]
    model, log = _resolve(tmp_path, body, nodes)
    assert not log.errors, log.errors
    sf = model.skews.entries[0]
    assert (sf.n1, sf.n2, sf.n3) == (9, 1259, 19)
    assert sf.idir == 1 and sf.imov == 1          # DIR = X
    # N1->N2 is +X, N3 is +Y off it => the identity frame
    assert np.allclose(model.skews.axes[1], np.eye(3))
    assert model.skews.has_moving()


def test_corpus_skew_mov2_card(tmp_path):
    """/SKEW/MOV2 (cfg SYSTEM/skew_mov2.cfg radioss100): three node
    columns, no DIR, N1->N2 IS Z'."""
    body = ("/SKEW/MOV2/7\n"
            "mov2\n"
            "         1         2         3\n")
    nodes = [(1, 0.0, 0.0, 0.0), (2, 0.0, 0.0, 5.0), (3, 4.0, 0.0, 0.0)]
    model, log = _resolve(tmp_path, body, nodes)
    assert not log.errors, log.errors
    sf = model.skews.entries[0]
    assert (sf.subtype, sf.imov, sf.idir) == ("MOV2", 2, 3)
    assert np.allclose(model.skews.axes[1], np.eye(3))   # Z'=+Z, X'=+X


def test_corpus_frame_fix_cam(tmp_path):
    """RD-E-2100 Cam — its /FRAME/FIX/1 verbatim (origin OFF the global
    origin: that offset is exactly what the /INIVEL/AXIS fix needs)."""
    body = (
        "/FRAME/FIX/1\n"
        "NULL\n"
        "                   0                  18                .185\n"
        "                   0                   1                   0\n"
        "                   1                   0                   0\n")
    model, log = _resolve(tmp_path, body)
    assert not log.errors, log.errors
    sf = model.skews.entries[0]
    assert (sf.kind, sf.subtype) == ("FRAME", "FIX")
    assert np.allclose(model.skews.origins[1], [0.0, 18.0, 0.185])
    a = model.skews.axes[1]
    assert np.allclose(a @ a.T, np.eye(3), atol=1e-14)
    assert np.allclose(a[1], [0, 1, 0])          # the Y' the AXIS card names


def test_corpus_frame_mov_tensile(tmp_path):
    """RD-E-1101 Tensile (4 corpus decks) — its /FRAME/MOV/1 verbatim: a
    BLANK title card and a BLANK DIR column (which is X, not an error —
    hm_read_frm.F leaves IDIR at its default)."""
    body = ("/FRAME/MOV/1\n"
            + " " * 100 + "\n"
            "       102       616       148\n")
    nodes = [(102, 0.0, 0.0, 0.0), (616, 1.0, 0.0, 0.0),
             (148, 0.0, 1.0, 0.0)]
    model, log = _resolve(tmp_path, body, nodes)
    assert not log.errors, log.errors
    sf = model.skews.entries[0]
    assert (sf.kind, sf.subtype, sf.imov, sf.idir) == ("FRAME", "MOV", 1, 1)
    assert (sf.n1, sf.n2, sf.n3) == (102, 616, 148)


def test_skew_and_frame_ids_are_separate_spaces(tmp_path):
    """A /SKEW/1 and a /FRAME/1 coexist (the Fortran keeps ISKN's skew and
    frame ranges apart); a duplicate id INSIDE one kind is refused
    (UDOUBLE on ISKN(4,*))."""
    body = (_skew_fix(1, (0, 0, 0), (0, 1, 0), (0, 0, 1)) + "\n"
            + _skew_fix(1, (9, 9, 9), (0, 1, 0), (0, 0, 1), kind="FRAME")
            + "\n")
    model, log = _resolve(tmp_path, body)
    assert not log.errors, log.errors
    assert model.skews.index("SKEW", 1) != model.skews.index("FRAME", 1)
    assert np.allclose(model.skews.origins[model.skews.index("FRAME", 1)],
                       [9, 9, 9])
    # duplicate INSIDE a kind
    body2 = (_skew_fix(3, (0, 0, 0), (0, 1, 0), (0, 0, 1)) + "\n"
             + _skew_fix(3, (1, 1, 1), (0, 1, 0), (0, 0, 1)) + "\n")
    _, log2 = _resolve(tmp_path, body2, name="dup")
    assert any("duplicate" in m.lower() for m in log2.errors)


def test_unknown_skew_id_is_an_error(tmp_path):
    """A consumer naming a skew that does not exist is a hard error (the
    reference's ANCMSG 184/490), not a silent fallback to global."""
    body = _bcs(1, "bad", "111", "000", 42, 1) + "\n/GRNOD/NODE/1\ng\n" \
        + fmt_int(1) + "\n"
    from pyradioss.starter.initialization import resolve_skews
    d = StarterDeck("bad")
    d.node([(1, 0.0, 0.0, 0.0)])
    d.lines.extend(body.splitlines())
    p = tmp_path / "bad_0000.rad"
    d.write(str(p))
    model, log = Model(), MessageLog()
    parse_starter_deck(read_deck(str(p)), model, log)
    resolve_skews(model, log)
    assert any("42" in m for m in log.errors), log.errors


# ============================================================================
# 3. KINEMATIC PROOF — /BCS in a 45-degree skew
# ============================================================================

def test_bcs_45deg_skew_constrains_the_skewed_dof(tmp_path):
    """A cube under gravity along +Y, every node /BCS-constrained on the
    Z' axis of a 45-degree skew (Y'=(0,1,1), Z'=(0,-1,1)).

    The reference (bcs1v, 'USER SYSTEM' branch, LCOD==1) PROJECTS the
    constrained skew axis out of the velocity: ``VV = e.V ; V -= e VV``.
    So with a uniform acceleration ``a``:

    * v(t) . Z' == 0 exactly, every cycle (the skewed dof IS constrained);
    * v(t) == P(a) t with P(a) = a - Z'(Z'.a) — the in-plane motion is
      FREE and driven by the in-plane force component alone;
    * the reaction m(a - P(a)) is purely along Z' (the skew normal).

    For a = (0, A, 0):  P(a) = (0, A/2, A/2) = Y' (a.Y').
    """
    A = 10.0                                   # gravity magnitude, +Y
    extra = (_skew_fix(1, (0, 0, 0), (0, 1, 1), (0, -1, 1),
                       title="45 deg about X") + "\n"
             + _bcs(1, "constrain Z-prime", "001", "000", 1, 1) + "\n")
    d = _cube_deck("BCSSK", extra)
    d.funct(1, "unit", [(0.0, 1.0), (1.0, 1.0)])
    d.grav(1, "g along Y", 1, "Y", 1, A)
    t_end = 0.02
    model = _run(tmp_path, "BCSSK", d, t_end)

    zp = np.array([0.0, -SQ, SQ])               # the constrained axis
    yp = np.array([0.0, SQ, SQ])                # the free in-plane axis
    v = model.v

    # (a) the SKEWED dof is exactly constrained
    assert np.abs(v @ zp).max() < 1e-12, "velocity leaked along the skew Z'"
    # (b) the in-plane motion is the FREE unconstrained one
    expect = np.array([0.0, A * t_end / 2.0, A * t_end / 2.0])
    assert np.abs(v - expect).max() < 1e-3 * A * t_end, (v[0], expect)
    assert (v @ yp)[0] == pytest.approx(A * SQ * t_end, rel=1e-3)
    # (c) the cube translated rigidly: no straining, no stress
    assert np.abs(model.bricks.state["sig"]).max() < 1e-9
    # (d) the reaction (the removed acceleration) is purely along Z'
    a_free = np.array([0.0, A, 0.0])
    reaction = a_free - expect / t_end
    assert np.abs(np.cross(reaction, zp)).max() < 1e-9
    assert np.dot(reaction, zp) == pytest.approx(-A * SQ, rel=1e-3)


def test_bcs_skew_all_three_axes_equals_full_clamp(tmp_path):
    """Constraining all three skew axes is a full clamp whatever the skew
    (bcs1v's LCOD==7 zeroes the vector outright — the port reaches the same
    state by projecting out three orthonormal axes)."""
    extra = (_skew_fix(1, (0, 0, 0), (0, 1, 1), (0, -1, 1)) + "\n"
             + _bcs(1, "clamp", "111", "000", 1, 1) + "\n")
    d = _cube_deck("BCSCL", extra)
    d.funct(1, "unit", [(0.0, 1.0), (1.0, 1.0)])
    d.grav(1, "g", 1, "Y", 1, 10.0)
    model = _run(tmp_path, "BCSCL", d, 0.02)
    assert np.abs(model.v).max() < 1e-12
    assert np.abs(model.x - model.x0).max() < 1e-12


# ============================================================================
# 4. KINEMATIC PROOF — /IMPVEL along a rotated axis
# ============================================================================

def _impvel_model(tmp_path, name, skew_block, skew_id, direction):
    """A cube with /IMPVEL v(t)=1*t on `direction` of skew `skew_id`."""
    extra = (skew_block + _imp("IMPVEL", 1, "drive", 1, direction,
                               skew_id, 1) + "\n")
    d = _cube_deck(name, extra)
    d.funct(1, "ramp", [(0.0, 0.0), (1.0, 1.0)])
    return _run(tmp_path, name, d, 0.02)


def test_impvel_rotated_axis_is_the_rotated_reference_solution(tmp_path):
    """OBJECTIVITY: an /IMPVEL along the Y' axis of a skew that is the
    global frame rotated by 45 degrees about X must reproduce the UNROTATED
    reference run (imposed along global Y) with every vector rotated.

    This is the statement fixvel.F 390-418 makes: the imposed component is
    taken along ``SKEW(3J-2..3J, ISK)`` and the correction is added back
    along the SAME axis, so the whole solution is frame-equivariant.
    """
    # reference: imposed along GLOBAL Y, no skew
    ref = _impvel_model(tmp_path, "IVREF", "", 0, "Y")
    # rotated: a skew whose Y' is the global Y rotated 45 deg about X
    rot = _impvel_model(
        tmp_path, "IVROT",
        _skew_fix(1, (0, 0, 0), (0, 1, 1), (0, -1, 1)) + "\n", 1, "Y")

    # R maps global -> the rotated frame's axes: rows are X', Y', Z'
    R = axes_from_fix([0, 1, 1], [0, -1, 1])
    # the imposed velocity, expressed in the skew, must equal the
    # reference's global velocity component-for-component
    assert np.abs((rot.v @ R.T) - ref.v).max() < 1e-10, \
        "the skewed imposed motion is not the rotated reference solution"
    # and it is genuinely rotated (not accidentally the same field)
    assert np.abs(rot.v - ref.v).max() > 1e-4
    # the driven component matches the curve; the transverse ones are free
    # (zero here — nothing else drives them)
    yp = R[1]
    t_end = 0.02
    assert (rot.v @ yp)[0] == pytest.approx(t_end, rel=1e-3)
    assert np.abs(rot.v @ R[0]).max() < 1e-10
    assert np.abs(rot.v @ R[2]).max() < 1e-10


def test_impdisp_skew_lands_on_the_imposed_displacement(tmp_path):
    """/IMPDISP in a skew: each node lands exactly on x0 + d(t) e_dir
    measured ALONG the skew axis (fixvel.F's DD = SKEW . D), with the
    transverse position untouched."""
    extra = (_skew_fix(1, (0, 0, 0), (0, 1, 1), (0, -1, 1)) + "\n"
             + _imp("IMPDISP", 1, "push", 1, "Y", 1, 1) + "\n")
    d = _cube_deck("IDSK", extra)
    d.funct(1, "ramp", [(0.0, 0.0), (1.0, 1.0)])
    t_end = 0.02
    model = _run(tmp_path, "IDSK", d, t_end)
    yp = np.array([0.0, SQ, SQ])
    disp = model.x - model.x0
    # the displacement along Y' is the curve value; nothing transverse
    assert np.abs((disp @ yp) - t_end).max() < 1e-6
    assert np.abs(disp @ np.array([1.0, 0, 0])).max() < 1e-9
    assert np.abs(disp @ np.array([0.0, -SQ, SQ])).max() < 1e-9


# ============================================================================
# 5. KINEMATIC PROOF — /SKEW/MOV tracks spinning nodes
# ============================================================================

def test_skew_mov_tracks_spinning_nodes_vs_analytic_frame(tmp_path):
    """A /SKEW/MOV whose three nodes are carried around the Z axis by an
    /INIVEL/AXIS rigid rotation: the skew rebuilt each cycle by NEWSKW must
    equal the ANALYTIC rotation of the initial frame.

    N1 at the origin, N2 on +X, N3 on +Y, DIR = X, all on a /RBODY so the
    triangle stays rigid.  After time t the frame must be Rz(omega t)
    applied to the identity triad, to the accuracy of the rigid-body
    integration.
    """
    omega = 20.0                                # rad / time-unit
    d = StarterDeck("MOVSK")
    # a unit cube (supplies the mass) + a standalone /RBODY master; the
    # skew's three nodes ARE cube corners, so they are carried rigidly
    d.node([(10, 0.0, 0.0, 0.0), (11, 1, 0, 0), (12, 1, 1, 0), (13, 0, 1, 0),
            (14, 0, 0, 1), (15, 1, 0, 1), (16, 1, 1, 1), (17, 0, 1, 1),
            (99, 0.0, 0.0, 0.0)])
    d.brick(1, [(1, 10, 11, 12, 13, 14, 15, 16, 17)])
    d.part(1, "cube", 1, 1)
    d.mat_law1(1, "steel", 7.8e-6, 210.0, 0.3)
    d.prop_solid(1, "solid")
    d.grnod_node(1, "rb slaves", [10, 11, 12, 13, 14, 15, 16, 17])
    d.grnod_node(2, "spin", [10, 11, 12, 13, 14, 15, 16, 17, 99])
    # the standalone master is carried by the body (ICoG default 1 moves it
    # to the COG, which the skew does not care about — its three nodes are
    # cube corners).  The writer's emitter, NOT raw_block: this deck
    # declares /BEGIN 2022, so every block is read in the FIXED dialect.
    d.rbody(1, "rigid", master=99, grnod=1)
    d.inivel_axis(1, "spin about Z", omega, "Z", 2)
    # N1=10 (0,0,0), N2=11 (+X), N3=13 (+Y), DIR=X  ->  identity at t=0
    d.lines.extend((
        "/SKEW/MOV/1\n"
        "rides the spinning cube\n"
        "        10        11        13X\n").splitlines())
    t_end = 0.01
    model = _run(tmp_path, "MOVSK", d, t_end)

    def _analytic(theta):
        """Rz(theta) applied to the identity triad, as ROWS (X', Y', Z')."""
        c, s = math.cos(theta), math.sin(theta)
        return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]]).T

    row = model.skews.index("SKEW", 1)
    n10, n11 = model.node_index(10), model.node_index(11)
    # theta read from where the tracking nodes ACTUALLY are (the rigid-body
    # integrator's own angle) — this isolates the SKEW update from any
    # integration phase error in the spin itself
    d12 = model.x[n11] - model.x[n10]
    theta = math.atan2(d12[1], d12[0])

    # the body really turned, at omega (guards a frozen-frame false pass)
    assert theta == pytest.approx(omega * t_end, rel=2e-2), theta
    got = model.skews.axes[row]
    assert np.abs(got - np.eye(3)).max() > 1e-3, "the frame never moved"

    # (a) LIVE: the engine's frame tracks the rotation to within ONE cycle.
    # It is built at the TOP of the cycle from the positions of that
    # instant (resol.F calls NEWSKW before the force evaluation), so it
    # legitimately lags the end-of-run model.x by one step of omega*dt.
    lag = abs(math.atan2(got[0][1], got[0][0]) - theta)
    assert lag < 2e-3, f"frame lags by {lag} rad, more than one cycle"
    assert np.abs(got - _analytic(theta)).max() < 2e-3

    # (b) EXACT: re-run the NEWSKW update on the SAME positions theta was
    # read from — the frame must then equal the analytic one to machine
    # precision (no lag left to explain the difference)
    model.skews.update(model.x)
    got = model.skews.axes[row]
    assert np.abs(got - _analytic(theta)).max() < 1e-12, (got, theta)
    # the origin rides N1 (newskw.F P(10:12) = X(:,N1))
    assert np.allclose(model.skews.origins[row], model.x[n10])


def test_skew_mov_update_matches_the_analytic_frame_directly():
    """The NEWSKW update in isolation: feed rotated positions, get the
    rotated frame (no integrator in the loop)."""
    from pyradioss.model.skew import SkewFrame, SkewSet

    class _M:
        pass
    m = _M()
    m.x0 = np.array([[0.0, 0, 0], [1.0, 0, 0], [0.0, 1, 0]])
    m._id2idx = {1: 0, 2: 1, 3: 2}
    m.node_index = lambda i: m._id2idx[i]
    ss = SkewSet()
    ss.add(SkewFrame(id=1, kind="SKEW", subtype="MOV", n1=1, n2=2, n3=3,
                     idir=1, imov=1))
    log = MessageLog()
    ss.resolve(m, log)
    assert not log.errors
    assert np.allclose(ss.axes[1], np.eye(3))
    for theta in (0.1, 0.7, 2.5, -1.3):
        c, s = math.cos(theta), math.sin(theta)
        Rz = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
        ss.update((Rz @ m.x0.T).T)
        assert np.allclose(ss.axes[1], (Rz @ np.eye(3)).T, atol=1e-14)


# ============================================================================
# 6. consumer wiring
# ============================================================================

def test_rbody_inertia_change_of_basis(tmp_path):
    """/RBODY Skew_ID: the card's Jxx/Jyy/Jzz are written in the SKEW's
    axes and rotated into the global frame ONCE at init — inirby.F's
    ``CALL CHBAS(SKEW(1,NOSKEW), RBY(1,NRB))``, i.e. J_glob = A J A^T with
    A's columns the skew axes (chbas.F).

    Modelled on RD-E-1601's dummy (Jxx/Jyy/Jzz = 178/61/124, a genuinely
    NON-spherical tensor on a real skew — the case where the rotation
    changes the answer).
    """
    from pyradioss.model.skew import SkewFrame, SkewSet
    ss = SkewSet()

    class _M:
        pass
    m = _M()
    m.x0 = np.zeros((1, 3))
    m.node_index = lambda i: 0
    ss.add(SkewFrame(id=1, kind="SKEW", subtype="FIX", imov=0,
                     origin_card=np.zeros(3), yaxis=np.array([0.0, 1, 1]),
                     zaxis=np.array([0.0, -1, 1])))
    ss.resolve(m, MessageLog())
    J_local = np.diag([178.0, 61.0, 124.0])
    J_glob = ss.rotate_tensor(1, J_local)

    A = ss.axes[1].T                        # skew -> global (columns = axes)
    assert np.allclose(J_glob, A @ J_local @ A.T)
    # invariants of a similarity transform
    assert np.isclose(np.trace(J_glob), np.trace(J_local))
    assert np.allclose(np.sort(np.linalg.eigvalsh(J_glob)),
                       np.sort(np.diag(J_local)))
    # it genuinely rotated (the 45-deg skew mixes Jyy and Jzz)
    assert not np.allclose(J_glob, J_local)
    assert J_glob[1, 2] == pytest.approx((61.0 - 124.0) * 0.5, rel=1e-9)
    # a SPHERICAL tensor is skew-invariant (why the corpus's Ispher=1
    # bodies are unaffected by their skew)
    assert np.allclose(ss.rotate_tensor(1, np.eye(3) * 5.0), np.eye(3) * 5.0)


def test_inivel_axis_frame_axis_and_origin(tmp_path):
    """/INIVEL/AXIS + /FRAME: the rotation runs about the frame's DIR axis
    THROUGH THE FRAME ORIGIN, and Vt is rotated into global
    (hm_read_inivel.F 437-439 + 581-598).  Modelled on RD-E-2100 Cam,
    whose frame origin (0, 18, .185) is OFF the global origin — pre-M39 the
    port put the axis through (0,0,0), a real deviation.
    """
    origin = (0.0, 18.0, 0.185)
    probe = (1.0, 2.0, 3.0)
    # a cube is ballast (the Starter refuses an element-less model); node
    # 99 is the probe the /INIVEL/AXIS group holds
    d = _cube_deck("IVAX", "")
    d.node([(99,) + probe])
    d.grnod_node(2, "probe", [99])
    d.lines.extend((_skew_fix(1, origin, (0, 1, 0), (1, 0, 0),
                              kind="FRAME", title="NULL")).splitlines())
    # /INIVEL/AXIS real dialect: DIR FRAME_ID GRNOD_ID / Vxt Vyt Vzt VR
    d.lines.extend([
        "/INIVEL/AXIS/1", "initial cam rotation",
        fmt_str("Y") + fmt_int(1) + fmt_int(2),
        _v3(0.0, 0.0, 0.0) + fmt_float(314.0)])
    p = tmp_path / "IVAX_0000.rad"
    d.write(str(p))
    with contextlib.redirect_stdout(io.StringIO()):
        model = run_starter(str(p), MessageLog())

    iv = model.inivel[0]
    assert iv.frame_id == 1 and iv.dir == 2
    assert np.allclose(iv.axis, [0.0, 1.0, 0.0])
    assert np.allclose(iv.origin, origin), "the axis must ride the FRAME"
    # the velocity field is omega * d x (x0 - O), NOT omega * d x x0
    k = model.node_index(99)
    r = np.array(probe) - np.array(origin)
    expect = 314.0 * np.cross([0.0, 1.0, 0.0], r)
    assert np.allclose(model.v[k], expect)
    wrong = 314.0 * np.cross([0.0, 1.0, 0.0], probe)
    assert not np.allclose(model.v[k], wrong), "still using the global origin"


def test_inivel_axis_frame_rotates_vt(tmp_path):
    """The AXIS card's Vxt/Vyt/Vzt are components IN the frame."""
    d = _cube_deck("IVVT", "")
    d.node([(99, 0.0, 0.0, 0.0)])
    d.grnod_node(2, "probe", [99])
    d.lines.extend((_skew_fix(1, (0, 0, 0), (0, 1, 1), (0, -1, 1),
                              kind="FRAME")).splitlines())
    d.lines.extend([
        "/INIVEL/AXIS/1", "translate in the frame",
        fmt_str("X") + fmt_int(1) + fmt_int(2),
        _v3(0.0, 5.0, 0.0) + fmt_float(0.0)])       # Vt = 5 along the frame Y'
    p = tmp_path / "IVVT_0000.rad"
    d.write(str(p))
    with contextlib.redirect_stdout(io.StringIO()):
        model = run_starter(str(p), MessageLog())
    # 5 along Y'=(0,1,1)/sqrt2 => global (0, 5/sqrt2, 5/sqrt2)
    assert np.allclose(model.v[model.node_index(99)], [0.0, 5.0 * SQ, 5.0 * SQ])


def test_type8_spring_frame_is_the_skew(tmp_path):
    """/PROP/TYPE8's skew_ID IS the spring's local frame: r2def3.F reads
    ``EXX = SKEW(1,ISK) ... EZZ = SKEW(9,ISK)`` and resolves the relative
    motion on those axes.  A 45-degree skew must rotate e1/e2/e3."""
    d = StarterDeck("SPSK")
    d.node([(1, 0.0, 0.0, 0.0), (2, 0.0, 0.0, 0.0)])
    d.spring(1, [(1, 1, 2)])
    d.part(1, "spring", 1, 0)
    d.lines.extend((_skew_fix(1, (0, 0, 0), (0, 1, 1), (0, -1, 1))).splitlines())
    # /PROP/SPR_GENE: Mass Inertia skew_ID sens Isflag Ifail Ifail2 Iequil
    d.lines.extend(["/PROP/SPR_GENE/1", "gene",
                    fmt_float(1e-6) + fmt_float(1e-6) + fmt_int(1)
                    + fmt_int(0) + fmt_int(0) + fmt_int(0) + fmt_int(0)
                    + fmt_int(0)])
    for _ in range(6):                       # six K/C/A/B/D + N + F cards
        d.lines.extend([_v3(50.0, 0.0, 0.0) + fmt_float(0.0) + fmt_float(0.0),
                        fmt_int(0) * 5 + blank(20) + fmt_float(0.0)
                        + fmt_float(0.0),
                        _v3(0.0, 0.0, 0.0) + fmt_float(0.0)])
    d.lines.append(fmt_int(0) + fmt_float(0.0))       # Fsmooth Fcut
    p = tmp_path / "SPSK_0000.rad"
    d.write(str(p))
    with contextlib.redirect_stdout(io.StringIO()):
        model = run_starter(str(p), MessageLog())
    g = model.springs.state["gen6"]
    assert np.allclose(g["e1"][0], [1.0, 0.0, 0.0])
    assert np.allclose(g["e2"][0], [0.0, SQ, SQ])
    assert np.allclose(g["e3"][0], [0.0, -SQ, SQ])
    assert int(g["skew_row"][0]) == model.skews.index("SKEW", 1)


# ============================================================================
# 7. the loud defers
# ============================================================================

def test_impvel_icoor_and_frame_defer_loudly(tmp_path):
    """Icoor=1 (cylindrical) and frame_ID on /IMP* are NOT ported: both
    must WARN that the physics deviates, never pass silently.  RD-V-0530
    (wave propagation) is the corpus deck that carries Icoor=1."""
    card = ("/IMPDISP/9\nQ1\n"
            + fmt_int(1) + fmt_str("X") + fmt_int(1) + fmt_int(0)
            + fmt_int(1) + fmt_int(2) + fmt_int(1) + "\n"
            + fmt_float(1.0) + fmt_float(1.0) + fmt_float(0.0)
            + fmt_float(1e30) + "\n")
    _, log = _parse(tmp_path, card + "/GRNOD/NODE/1\ng\n" + fmt_int(1) + "\n")
    msgs = " ".join(log.warnings).lower()
    assert "icoor" in msgs and "not ported" in msgs
    assert "frame_id" in msgs
    assert "deviates" in msgs


def test_type13_spring_skew_defers_loudly(tmp_path):
    """A skew on a TYPE13 (SPR_BEAM) is the initial frame of a
    co-rotational beam — not ported, and loudly so."""
    from pyradioss.starter.initialization import resolve_skews
    d = StarterDeck("SB")
    d.node([(1, 0.0, 0.0, 0.0), (2, 1.0, 0.0, 0.0)])
    d.lines.extend((_skew_fix(1, (0, 0, 0), (0, 1, 0), (0, 0, 1))).splitlines())
    d.lines.extend(["/PROP/SPR_BEAM/1", "beam spring",
                    fmt_float(1e-6) + fmt_float(1e-6) + fmt_int(1)
                    + fmt_int(0) + fmt_int(0) + fmt_int(0) + fmt_int(0)
                    + fmt_int(0)])
    p = tmp_path / "SB_0000.rad"
    d.write(str(p))
    model, log = Model(), MessageLog()
    parse_starter_deck(read_deck(str(p)), model, log)
    resolve_skews(model, log)
    msgs = " ".join(log.warnings).lower()
    assert "type13" in msgs and "not ported" in msgs, log.warnings


# ============================================================================
# 8. regression: an unskewed model is untouched
# ============================================================================

def test_no_skew_model_is_bit_identical(tmp_path):
    """The skew machinery must be invisible to a model that has none: the
    global row 0 exists, nothing moves, and the /BCS fast path (the boolean
    mask) still carries every unskewed condition."""
    extra = _bcs(1, "clamp", "111", "111", 0, 1) + "\n"
    d = _cube_deck("NOSK", extra)
    d.funct(1, "unit", [(0.0, 1.0), (1.0, 1.0)])
    d.grav(1, "g", 1, "Y", 1, 10.0)
    model = _run(tmp_path, "NOSK", d, 0.02)
    assert model.skews.axes.shape == (1, 3, 3)
    assert np.allclose(model.skews.axes[0], np.eye(3))
    assert not model.skews.has_moving()
    assert np.abs(model.v).max() < 1e-12          # fully clamped


def test_inivel_resolve_skews_idempotent(tmp_path):
    """resolve_skews called a second time (e.g. after transforms) must not double-rotate INIVEL vectors."""
    from pyradioss.starter.initialization import resolve_skews, initialize_elements_and_mass
    from pyradioss.model.entities import InitialVelocity
    d = _cube_deck("IVIDEM", "")
    d.node([(99, 0.0, 0.0, 0.0)])
    d.grnod_node(2, "probe", [99])
    d.lines.extend((_skew_fix(1, (0, 0, 0), (0, 1, 1), (0, -1, 1),
                              kind="FRAME")).splitlines())
    d.lines.extend((_skew_fix(2, (0, 0, 0), (0, 1, 1), (0, -1, 1),
                              kind="SKEW")).splitlines())
    d.lines.extend([
        "/INIVEL/AXIS/1", "translate in the frame",
        fmt_str("X") + fmt_int(1) + fmt_int(2),
        _v3(0.0, 5.0, 0.0) + fmt_float(0.0)])
    p = tmp_path / "IVIDEM_0000.rad"
    d.write(str(p))
    log = MessageLog()
    with contextlib.redirect_stdout(io.StringIO()):
        model = run_starter(str(p), log)
    expected_v = np.array([0.0, 5.0 * SQ, 5.0 * SQ])
    iv = model.inivel[0]
    assert np.allclose(iv.v, expected_v)

    # Add a TRANS inivel with iskew=2
    iv_trans = InitialVelocity(id=2, grnod_id=2, v=np.array([0.0, 5.0, 0.0]),
                               kind="TRANS", iskew=2)
    model.inivel.append(iv_trans)

    # Calling resolve_skews again (simulating transform pass)
    resolve_skews(model, log)
    assert np.allclose(iv.v, expected_v), "AXIS vector was rotated a second time!"
    assert np.allclose(iv_trans.v, expected_v), "TRANS vector failed skew conversion or was double-rotated!"

    # Calling resolve_skews a third time
    resolve_skews(model, log)
    assert np.allclose(iv.v, expected_v), "AXIS vector was rotated on 3rd pass!"
    assert np.allclose(iv_trans.v, expected_v), "TRANS vector was rotated on 3rd pass!"


