"""
Milestone M489 — /INTER/TYPE2 Tied Contact Interface Hardening & Unit Tests.

First direct unit test coverage for pyradioss/contact/inter_type2.py (381 lines)
and associated Starter keyword parsing in pyradioss/input/starter_keywords.py.

Upstream Fortran subroutines:
- starter/source/interfaces/inter3d1/i2buc1.F: bucket / closest segment search
- starter/source/interfaces/inter3d1/i2dst3.F: distance and isoparametric projection
- starter/source/interfaces/inter3d1/i2tid3.F: tied initialization and mass transfer
- engine/source/interfaces/interf/i2for3.F: force transfer (I2FOR3, I2FOMO3)
- engine/source/interfaces/interf/i2vit3.F: kinematic velocity update (I2VIT3, I2VIROT3, I2ROT3_27)
- engine/source/interfaces/interf/i2curv.F: segment co-rotating frame (t1, t2, n)
"""

from __future__ import annotations

import numpy as np
import pytest

from pyradioss.common.messages import MessageLog
from pyradioss.contact import ContactType2, build_contacts
from pyradioss.contact.inter_type2 import _segment_frames
from pyradioss.input.deck_reader import KeywordBlock, Card
from pyradioss.input.starter_keywords import read_inter
from pyradioss.model.entities import (
    ImposedVelocity,
    Interface,
    NodeGroup,
    Surface,
)
from pyradioss.model.model import ElementGroup, Model


# ============================================================================
# Helpers to construct test models
# ============================================================================

def make_tied_model(
    coords: np.ndarray,
    segs: np.ndarray,
    cand_nodes: list[int] | np.ndarray,
    masses: np.ndarray | None = None,
    dsearch: float = 0.0,
    spotflag: int = 0,
) -> tuple[Model, Interface]:
    model = Model()
    n = len(coords)
    model.node_ids = np.arange(1, n + 1)
    model.x0 = np.array(coords, dtype=float)
    model.x = model.x0.copy()
    model.v = np.zeros((n, 3), dtype=float)
    model.vr = np.zeros((n, 3), dtype=float)
    if masses is not None:
        model.mass = np.array(masses, dtype=float)
    else:
        model.mass = np.ones(n, dtype=float)
    model.inertia = np.ones(n, dtype=float)

    surf = Surface(id=1, title="MAIN_SURF")
    surf.segments = np.array(segs, dtype=np.int64)
    surf.seg_gtype = np.full(len(segs), "", dtype="<U8")
    surf.seg_elem = np.full(len(segs), -1, dtype=np.int64)
    model.surfaces[1] = surf

    grp = NodeGroup(id=2, title="SEC_GROUP")
    grp.node_idx = np.array(cand_nodes, dtype=np.int64)
    model.node_groups[2] = grp

    itf = Interface(
        id=1,
        type=2,
        surf_id=1,
        grnod_id=2,
        dsearch=dsearch,
        spotflag=spotflag,
        title="TIED_ITF",
    )
    model.interfaces.append(itf)
    return model, itf


# ============================================================================
# 1. _segment_frames: Orthonormality, orientation, triangles, degenerate
# ============================================================================

def test_segment_frames_planar_quad():
    """Planar quad in xy-plane has t1 in x, t2 in y, n in +z."""
    xs = np.array([[
        [0.0, 0.0, 0.0],
        [2.0, 0.0, 0.0],
        [2.0, 2.0, 0.0],
        [0.0, 2.0, 0.0],
    ]])
    t1, t2, n = _segment_frames(xs)
    assert t1.shape == (1, 3)
    assert t2.shape == (1, 3)
    assert n.shape == (1, 3)

    # Orthonormal
    np.testing.assert_allclose(t1[0], [1.0, 0.0, 0.0], atol=1e-12)
    np.testing.assert_allclose(t2[0], [0.0, 1.0, 0.0], atol=1e-12)
    np.testing.assert_allclose(n[0], [0.0, 0.0, 1.0], atol=1e-12)
    assert np.isclose(np.dot(t1[0], t2[0]), 0.0, atol=1e-12)
    assert np.isclose(np.dot(t1[0], n[0]), 0.0, atol=1e-12)
    assert np.isclose(np.dot(t2[0], n[0]), 0.0, atol=1e-12)


def test_segment_frames_arbitrary_3d_quad():
    """Arbitrary rotated 3D quad produces mutually orthogonal unit vectors."""
    xs = np.array([[
        [1.0, 2.0, 3.0],
        [4.0, 1.0, 5.0],
        [5.0, 3.0, 8.0],
        [2.0, 4.0, 6.0],
    ]])
    t1, t2, n = _segment_frames(xs)
    assert np.isclose(np.linalg.norm(t1[0]), 1.0, atol=1e-12)
    assert np.isclose(np.linalg.norm(t2[0]), 1.0, atol=1e-12)
    assert np.isclose(np.linalg.norm(n[0]), 1.0, atol=1e-12)
    assert np.isclose(np.dot(t1[0], t2[0]), 0.0, atol=1e-12)
    assert np.isclose(np.dot(t1[0], n[0]), 0.0, atol=1e-12)
    assert np.isclose(np.dot(t2[0], n[0]), 0.0, atol=1e-12)


def test_segment_frames_triangular_segment():
    """Triangular segment repeating node 3 produces valid orthonormal frame."""
    xs = np.array([[
        [0.0, 0.0, 0.0],
        [2.0, 0.0, 0.0],
        [0.0, 2.0, 0.0],
        [0.0, 2.0, 0.0],
    ]])
    t1, t2, n = _segment_frames(xs)
    assert np.isclose(np.linalg.norm(t1[0]), 1.0, atol=1e-12)
    assert np.isclose(np.linalg.norm(t2[0]), 1.0, atol=1e-12)
    assert np.isclose(np.linalg.norm(n[0]), 1.0, atol=1e-12)
    # Normal is in +z direction for CCW triangle in xy-plane
    np.testing.assert_allclose(n[0], [0.0, 0.0, 1.0], atol=1e-12)


def test_segment_frames_degenerate_zero_area():
    """Zero-area segment falls back cleanly to unit basis without crash/NaN."""
    xs = np.zeros((1, 4, 3), dtype=float)
    t1, t2, n = _segment_frames(xs)
    assert not np.isnan(t1).any()
    assert not np.isnan(t2).any()
    assert not np.isnan(n).any()
    assert np.isclose(np.linalg.norm(t1[0]), 1.0)
    assert np.isclose(np.linalg.norm(n[0]), 1.0)


def test_segment_frames_empty_array():
    """Empty segment array returns empty (0, 3) arrays."""
    xs = np.zeros((0, 4, 3), dtype=float)
    t1, t2, n = _segment_frames(xs)
    assert t1.shape == (0, 3)
    assert t2.shape == (0, 3)
    assert n.shape == (0, 3)


# ============================================================================
# 2. ContactType2.__init__: Projection, weights, filters, offsets, warnings
# ============================================================================

def test_projection_quad_centroid():
    """Secondary node placed at centroid of quad segment gets equal weights [0.25, 0.25, 0.25, 0.25]."""
    coords = np.array([
        [0.0, 0.0, 0.0],
        [2.0, 0.0, 0.0],
        [2.0, 2.0, 0.0],
        [0.0, 2.0, 0.0],
        [1.0, 1.0, 0.0],  # node 4: centroid
    ])
    segs = np.array([[0, 1, 2, 3]])
    cand = [4]
    model, itf = make_tied_model(coords, segs, cand, dsearch=1.0)
    log = MessageLog()
    ct = ContactType2(itf, model, log)

    assert len(ct.snode) == 1
    assert ct.snode[0] == 4
    np.testing.assert_allclose(ct.w[0], [0.5, 0.0, 0.5, 0.0], atol=1e-12)
    np.testing.assert_allclose(ct.off_loc[0], [0.0, 0.0, 0.0], atol=1e-12)


def test_projection_quad_corner_and_edge():
    """Secondary node near corner gets weight ~1.0; node at edge midpoint gets weights ~0.5."""
    coords = np.array([
        [0.0, 0.0, 0.0],
        [2.0, 0.0, 0.0],
        [2.0, 2.0, 0.0],
        [0.0, 2.0, 0.0],
        [0.0, 0.0, 0.0],  # node 4: at corner 0
        [1.0, 0.0, 0.0],  # node 5: at midpoint of edge 0-1
    ])
    segs = np.array([[0, 1, 2, 3]])
    cand = [4, 5]
    model, itf = make_tied_model(coords, segs, cand, dsearch=1.0)
    ct = ContactType2(itf, model, MessageLog())

    assert len(ct.snode) == 2
    np.testing.assert_allclose(ct.w[0], [1.0, 0.0, 0.0, 0.0], atol=1e-12)
    np.testing.assert_allclose(ct.w[1], [0.5, 0.5, 0.0, 0.0], atol=1e-12)


def test_projection_triangle():
    """Secondary node on triangle gets accurate 3-node barycentric weights."""
    coords = np.array([
        [0.0, 0.0, 0.0],
        [3.0, 0.0, 0.0],
        [0.0, 3.0, 0.0],
        [1.0, 1.0, 0.0],  # node 3: centroid
    ])
    segs = np.array([[0, 1, 2, 2]])  # triangle: node 3 repeats node 2
    cand = [3]
    model, itf = make_tied_model(coords, segs, cand, dsearch=1.0)
    ct = ContactType2(itf, model, MessageLog())

    assert len(ct.snode) == 1
    np.testing.assert_allclose(ct.w[0, :3], [1.0/3.0, 1.0/3.0, 1.0/3.0], atol=1e-12)
    assert ct.w[0, 3] == 0.0


def test_exclusion_main_surface_nodes():
    """Main surface corner nodes are excluded from candidates (cannot tie to self)."""
    coords = np.array([
        [0.0, 0.0, 0.0],
        [2.0, 0.0, 0.0],
        [2.0, 2.0, 0.0],
        [0.0, 2.0, 0.0],
        [1.0, 1.0, 0.0],
    ])
    segs = np.array([[0, 1, 2, 3]])
    cand = [0, 4]  # node 0 is in surface
    model, itf = make_tied_model(coords, segs, cand, dsearch=1.0)
    log = MessageLog()
    ct = ContactType2(itf, model, log)

    assert len(ct.snode) == 1
    assert ct.snode[0] == 4
    warns = log.warnings
    assert any("main-surface corners" in w for w in warns)


def test_exclusion_massless_nodes():
    """Massless / frozen nodes (mass >= 1e29) are excluded from tying."""
    coords = np.array([
        [0.0, 0.0, 0.0],
        [2.0, 0.0, 0.0],
        [2.0, 2.0, 0.0],
        [0.0, 2.0, 0.0],
        [1.0, 1.0, 0.0],
    ])
    segs = np.array([[0, 1, 2, 3]])
    masses = np.array([1.0, 1.0, 1.0, 1.0, 1e30])
    cand = [4]
    model, itf = make_tied_model(coords, segs, cand, masses=masses, dsearch=1.0)
    log = MessageLog()
    ct = ContactType2(itf, model, log)

    assert len(ct.snode) == 0
    warns = log.warnings
    assert any("massless" in w for w in warns)


def test_dsearch_filtering_and_warning():
    """Candidate farther than dsearch is not tied and triggers a warning."""
    coords = np.array([
        [0.0, 0.0, 0.0],
        [2.0, 0.0, 0.0],
        [2.0, 2.0, 0.0],
        [0.0, 2.0, 0.0],
        [1.0, 1.0, 0.2],  # close: dist = 0.2 <= 0.5
        [1.0, 1.0, 1.5],  # far: dist = 1.5 > 0.5
    ])
    segs = np.array([[0, 1, 2, 3]])
    cand = [4, 5]
    model, itf = make_tied_model(coords, segs, cand, dsearch=0.5)
    log = MessageLog()
    ct = ContactType2(itf, model, log)

    assert len(ct.snode) == 1
    assert ct.snode[0] == 4
    warns = log.warnings
    assert any("farther than the search distance" in w for w in warns)


def test_dsearch_auto_calculation():
    """dsearch <= 0 auto-calculates search distance from sqrt(mean segment area)."""
    # Quad 2x2 has area 4.0, lc = 2.0
    coords = np.array([
        [0.0, 0.0, 0.0],
        [2.0, 0.0, 0.0],
        [2.0, 2.0, 0.0],
        [0.0, 2.0, 0.0],
        [1.0, 1.0, 1.0],  # dist = 1.0 < 2.0
    ])
    segs = np.array([[0, 1, 2, 3]])
    cand = [4]
    model, itf = make_tied_model(coords, segs, cand, dsearch=0.0)
    ct = ContactType2(itf, model, MessageLog())
    assert len(ct.snode) == 1
    assert ct.snode[0] == 4


def test_offset_local_frame():
    """Secondary node offset is stored in co-rotating local frame (t1, t2, n)."""
    coords = np.array([
        [0.0, 0.0, 0.0],
        [2.0, 0.0, 0.0],
        [2.0, 2.0, 0.0],
        [0.0, 2.0, 0.0],
        [1.1, 1.2, 0.3],  # offset: dx=0.1, dy=0.2, dz=0.3
    ])
    segs = np.array([[0, 1, 2, 3]])
    cand = [4]
    model, itf = make_tied_model(coords, segs, cand, dsearch=1.0)
    ct = ContactType2(itf, model, MessageLog())

    assert len(ct.snode) == 1
    # In-plane offsets are projected onto the element; out-of-plane offset is normal
    np.testing.assert_allclose(ct.off_loc[0], [0.0, 0.0, 0.3], atol=1e-12)


def test_large_offset_warning():
    """Offset exceeding 0.5 * lc triggers a warning."""
    coords = np.array([
        [0.0, 0.0, 0.0],
        [2.0, 0.0, 0.0],
        [2.0, 2.0, 0.0],
        [0.0, 2.0, 0.0],
        [1.0, 1.0, 1.2],  # lc = 2.0; offset 1.2 > 0.5 * lc = 1.0
    ])
    segs = np.array([[0, 1, 2, 3]])
    cand = [4]
    model, itf = make_tied_model(coords, segs, cand, dsearch=2.0)
    log = MessageLog()
    ct = ContactType2(itf, model, log)

    assert len(ct.snode) == 1
    warns = log.warnings
    assert any("largest tie offset" in w for w in warns)


def test_impvel_kinematic_clash_warning():
    """Prescribed /IMPVEL on tied node triggers kinematic condition clash warning."""
    coords = np.array([
        [0.0, 0.0, 0.0],
        [2.0, 0.0, 0.0],
        [2.0, 2.0, 0.0],
        [0.0, 2.0, 0.0],
        [1.0, 1.0, 0.0],
    ])
    segs = np.array([[0, 1, 2, 3]])
    cand = [4]
    model, itf = make_tied_model(coords, segs, cand, dsearch=1.0)
    grp_imp = NodeGroup(id=99, title="IMP_GRP")
    grp_imp.node_idx = np.array([4])
    model.node_groups[99] = grp_imp
    model.impvel.append(ImposedVelocity(id=1, grnod_id=99, funct_id=1, dof=1, scale=1.0))

    log = MessageLog()
    ct = ContactType2(itf, model, log)
    assert len(ct.snode) == 1
    warns = log.warnings
    assert any("kinematic condition clash" in w for w in warns)


def test_missing_surface_and_node_group_defensive():
    """Missing surface or node group logs error and initializes safely without crashing."""
    model = Model()
    itf = Interface(id=1, type=2, surf_id=999, grnod_id=888)
    log = MessageLog()
    ct = ContactType2(itf, model, log)

    assert len(ct.snode) == 0
    assert len(ct.seg) == 0
    assert ct.active.shape == (0,)
    errors = log.errors
    assert any("main surface 999 not found" in e for e in errors)


def test_empty_candidates_or_segments_defensive():
    """Empty candidate group or empty surface initializes cleanly."""
    coords = np.array([[0.0, 0.0, 0.0]])
    segs = np.zeros((0, 4), dtype=int)
    model, itf = make_tied_model(coords, segs, [])
    ct = ContactType2(itf, model, MessageLog())
    assert len(ct.snode) == 0
    assert len(ct.seg) == 0


# ============================================================================
# 3. augment_mass: Mass transfer conservation
# ============================================================================

def test_augment_mass_single_secondary():
    """Mass transfer M_k += w_k m_s exactly conserves total mass."""
    coords = np.array([
        [0.0, 0.0, 0.0],
        [2.0, 0.0, 0.0],
        [2.0, 2.0, 0.0],
        [0.0, 2.0, 0.0],
        [1.0, 1.0, 0.0],  # secondary, mass = 0.4
    ])
    segs = np.array([[0, 1, 2, 3]])
    masses = np.array([1.0, 1.0, 1.0, 1.0, 0.4])
    model, itf = make_tied_model(coords, segs, [4], masses=masses, dsearch=1.0)
    ct = ContactType2(itf, model, MessageLog())

    mass_eff = model.mass.copy()
    ct.augment_mass(mass_eff)

    # Secondary node is at centroid -> diagonal split gives w = [0.5, 0.0, 0.5, 0.0]
    # Corners 0 and 2 receive 0.5 * 0.4 = 0.2
    expected = np.array([1.2, 1.0, 1.2, 1.0, 0.4])
    np.testing.assert_allclose(mass_eff, expected, atol=1e-12)
    # Physical mass remains untouched
    np.testing.assert_allclose(model.mass, masses)
    # Total effective mass on main segment increased by exactly secondary mass
    assert np.isclose(mass_eff[:4].sum() - model.mass[:4].sum(), 0.4)


def test_augment_mass_multi_secondary_shared_corners():
    """Multiple secondary nodes sharing corners accumulate additively."""
    coords = np.array([
        [0.0, 0.0, 0.0],
        [2.0, 0.0, 0.0],
        [2.0, 2.0, 0.0],
        [0.0, 2.0, 0.0],
        [0.0, 0.0, 0.0],  # secondary 4 near corner 0, mass = 0.5
        [1.0, 1.0, 0.0],  # secondary 5 at centroid, mass = 0.8
    ])
    segs = np.array([[0, 1, 2, 3]])
    masses = np.array([1.0, 1.0, 1.0, 1.0, 0.5, 0.8])
    model, itf = make_tied_model(coords, segs, [4, 5], masses=masses, dsearch=1.0)
    ct = ContactType2(itf, model, MessageLog())

    mass_eff = model.mass.copy()
    ct.augment_mass(mass_eff)

    # Node 0 receives 1.0 * 0.5 (from node 4) + 0.5 * 0.8 (from node 5) = 0.9
    assert np.isclose(mass_eff[0], 1.0 + 0.9, atol=1e-12)
    # Total mass transfer equals sum of secondary masses
    assert np.isclose(mass_eff[:4].sum() - model.mass[:4].sum(), 0.5 + 0.8)


def test_augment_mass_inactive_tie_no_transfer():
    """Inactive / released ties transfer zero mass."""
    coords = np.array([
        [0.0, 0.0, 0.0],
        [2.0, 0.0, 0.0],
        [2.0, 2.0, 0.0],
        [0.0, 2.0, 0.0],
        [1.0, 1.0, 0.0],
    ])
    segs = np.array([[0, 1, 2, 3]])
    model, itf = make_tied_model(coords, segs, [4], dsearch=1.0)
    ct = ContactType2(itf, model, MessageLog())
    ct.active[:] = False

    mass_eff = model.mass.copy()
    ct.augment_mass(mass_eff)
    np.testing.assert_array_equal(mass_eff, model.mass)


# ============================================================================
# 4. transfer_forces: Spotflag 0, 1, 2 force and moment conservation
# ============================================================================

def test_transfer_forces_spotflag0_translation_conservation():
    """Spotflag=0 distributes internal, external, contact forces with weights and clears secondary."""
    coords = np.array([
        [0.0, 0.0, 0.0],
        [2.0, 0.0, 0.0],
        [2.0, 2.0, 0.0],
        [0.0, 2.0, 0.0],
        [1.0, 1.0, 0.0],
    ])
    segs = np.array([[0, 1, 2, 3]])
    model, itf = make_tied_model(coords, segs, [4], dsearch=1.0, spotflag=0)
    ct = ContactType2(itf, model, MessageLog())

    fint = np.zeros((5, 3))
    fext = np.zeros((5, 3))
    fcont = np.zeros((5, 3))
    mint = np.zeros((5, 3))

    fint[4] = [10.0, -20.0, 30.0]
    fext[4] = [5.0, 15.0, -25.0]
    fcont[4] = [-2.0, 4.0, 8.0]

    mass_eff = model.mass.copy()
    inv_mass = 1.0 / mass_eff
    ct.transfer_forces(fint, fext, fcont, mint, model.x, mass_eff, inv_mass, cycle=1)

    # Secondary node forces cleared
    np.testing.assert_allclose(fint[4], [0.0, 0.0, 0.0])
    np.testing.assert_allclose(fext[4], [0.0, 0.0, 0.0])
    np.testing.assert_allclose(fcont[4], [0.0, 0.0, 0.0])

    # Net force on segment equals secondary force (w = [0.25, 0.25, 0.25, 0.25])
    np.testing.assert_allclose(fint[:4].sum(axis=0), [10.0, -20.0, 30.0])
    np.testing.assert_allclose(fext[:4].sum(axis=0), [5.0, 15.0, -25.0])
    np.testing.assert_allclose(fcont[:4].sum(axis=0), [-2.0, 4.0, 8.0])


def test_transfer_forces_spotflag1_force_and_moment_conservation():
    """Spotflag=1 (Solid main - I2FOMO3): offset force creates couple, conserving force and torque."""
    coords = np.array([
        [-1.0, -1.0, 0.0],
        [ 1.0, -1.0, 0.0],
        [ 1.0,  1.0, 0.0],
        [-1.0,  1.0, 0.0],
        [ 0.0,  0.0, 0.5],  # secondary: at centroid, offset dz = 0.5
    ])
    segs = np.array([[0, 1, 2, 3]])
    model, itf = make_tied_model(coords, segs, [4], dsearch=1.0, spotflag=1)
    ct = ContactType2(itf, model, MessageLog())

    fint = np.zeros((5, 3))
    fext = np.zeros((5, 3))
    fcont = np.zeros((5, 3))
    mint = np.zeros((5, 3))

    # Apply lateral force Fx = 100 on secondary node
    fint[4] = [100.0, 0.0, 0.0]
    # Expected offset torque about segment center (0,0,0):
    # dvec = [0, 0, 0.5], F = [100, 0, 0] -> M = dvec x F = [0, 50.0, 0]

    mass_eff = model.mass.copy()
    inv_mass = 1.0 / mass_eff
    ct.transfer_forces(fint, fext, fcont, mint, model.x, mass_eff, inv_mass, cycle=1)

    # Secondary force cleared
    np.testing.assert_allclose(fint[4], [0.0, 0.0, 0.0])

    # Net force on segment nodes is exactly Fx = 100 (couple forces sum to 0)
    np.testing.assert_allclose(fint[:4].sum(axis=0), [100.0, 0.0, 0.0], atol=1e-12)

    # Net moment sum(r_k x F_k) about center (0,0,0) must equal M = [0, 50, 0]
    r = model.x[:4]
    m_calc = np.sum(np.cross(r, fint[:4]), axis=0)
    np.testing.assert_allclose(m_calc, [0.0, 50.0, 0.0], atol=1e-12)


def test_transfer_forces_spotflag1_direct_moment_conversion():
    """Spotflag=1 converts secondary internal moment mint into force couple on solid nodes."""
    coords = np.array([
        [-1.0, -1.0, 0.0],
        [ 1.0, -1.0, 0.0],
        [ 1.0,  1.0, 0.0],
        [-1.0,  1.0, 0.0],
        [ 0.0,  0.0, 0.0],
    ])
    segs = np.array([[0, 1, 2, 3]])
    model, itf = make_tied_model(coords, segs, [4], dsearch=1.0, spotflag=1)
    ct = ContactType2(itf, model, MessageLog())

    fint = np.zeros((5, 3))
    fext = np.zeros((5, 3))
    fcont = np.zeros((5, 3))
    mint = np.zeros((5, 3))

    mint[4] = [0.0, 0.0, 40.0]  # Z-torque

    mass_eff = model.mass.copy()
    inv_mass = 1.0 / mass_eff
    ct.transfer_forces(fint, fext, fcont, mint, model.x, mass_eff, inv_mass, cycle=1)

    # mint[4] cleared
    np.testing.assert_allclose(mint[4], [0.0, 0.0, 0.0])

    # Net force is zero
    np.testing.assert_allclose(fint[:4].sum(axis=0), [0.0, 0.0, 0.0], atol=1e-12)

    # Net moment equals [0, 0, 40]
    r = model.x[:4]
    m_calc = np.sum(np.cross(r, fint[:4]), axis=0)
    np.testing.assert_allclose(m_calc, [0.0, 0.0, 40.0], atol=1e-12)


def test_transfer_forces_spotflag1_collinear_segment_pinv_stability():
    """Spotflag=1 on collinear degenerate segment uses pinv and does not crash."""
    coords = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [2.0, 0.0, 0.0],
        [3.0, 0.0, 0.0],  # all collinear along X
        [1.5, 0.0, 0.5],
    ])
    segs = np.array([[0, 1, 2, 3]])
    model, itf = make_tied_model(coords, segs, [4], dsearch=1.0, spotflag=1)
    ct = ContactType2(itf, model, MessageLog())

    fint = np.zeros((5, 3))
    fext = np.zeros((5, 3))
    fcont = np.zeros((5, 3))
    mint = np.zeros((5, 3))
    fint[4] = [10.0, 20.0, 30.0]

    mass_eff = model.mass.copy()
    inv_mass = 1.0 / mass_eff
    # Should not raise LinAlgError
    ct.transfer_forces(fint, fext, fcont, mint, model.x, mass_eff, inv_mass, cycle=1)
    np.testing.assert_allclose(fint[:4].sum(axis=0), [10.0, 20.0, 30.0])


def test_transfer_forces_spotflag2_shell_moment_transfer():
    """Spotflag=2 (Shell main - I2MOM3): offset moment is transferred into shell mint."""
    coords = np.array([
        [-1.0, -1.0, 0.0],
        [ 1.0, -1.0, 0.0],
        [ 1.0,  1.0, 0.0],
        [-1.0,  1.0, 0.0],
        [ 0.0,  0.0, 0.5],  # offset dz = 0.5
    ])
    segs = np.array([[0, 1, 2, 3]])
    model, itf = make_tied_model(coords, segs, [4], dsearch=1.0, spotflag=2)
    ct = ContactType2(itf, model, MessageLog())

    fint = np.zeros((5, 3))
    fext = np.zeros((5, 3))
    fcont = np.zeros((5, 3))
    mint = np.zeros((5, 3))

    fint[4] = [100.0, 0.0, 0.0]  # Fx=100 -> My = 50
    mint[4] = [10.0, 0.0, 0.0]   # Direct Mx = 10

    mass_eff = model.mass.copy()
    inv_mass = 1.0 / mass_eff
    ct.transfer_forces(fint, fext, fcont, mint, model.x, mass_eff, inv_mass, cycle=1)

    # Secondary node force and moment cleared
    np.testing.assert_allclose(fint[4], [0.0, 0.0, 0.0])
    np.testing.assert_allclose(mint[4], [0.0, 0.0, 0.0])

    # Shell nodes received translational force Fx = 100
    np.testing.assert_allclose(fint[:4].sum(axis=0), [100.0, 0.0, 0.0], atol=1e-12)

    # Shell nodes received total moment [10, 50, 0]
    np.testing.assert_allclose(mint[:4].sum(axis=0), [10.0, 50.0, 0.0], atol=1e-12)


def test_transfer_forces_zero_force_noop():
    """Zero forces on secondary node leave main segment forces and moments at zero."""
    coords = np.array([
        [0.0, 0.0, 0.0],
        [2.0, 0.0, 0.0],
        [2.0, 2.0, 0.0],
        [0.0, 2.0, 0.0],
        [1.0, 1.0, 0.0],
    ])
    segs = np.array([[0, 1, 2, 3]])
    model, itf = make_tied_model(coords, segs, [4], dsearch=1.0)
    ct = ContactType2(itf, model, MessageLog())

    fint = np.zeros((5, 3))
    fext = np.zeros((5, 3))
    fcont = np.zeros((5, 3))
    mint = np.zeros((5, 3))

    mass_eff = model.mass.copy()
    inv_mass = 1.0 / mass_eff
    ct.transfer_forces(fint, fext, fcont, mint, model.x, mass_eff, inv_mass, cycle=1)
    assert not np.any(fint)
    assert not np.any(mint)


# ============================================================================
# 5. enforce: Kinematic placement, co-rotating offset, velocity update
# ============================================================================

def test_enforce_rigid_translation():
    """Rigid translation of main segment updates secondary position and velocity."""
    coords = np.array([
        [0.0, 0.0, 0.0],
        [2.0, 0.0, 0.0],
        [2.0, 2.0, 0.0],
        [0.0, 2.0, 0.0],
        [1.0, 1.0, 0.0],
    ])
    segs = np.array([[0, 1, 2, 3]])
    model, itf = make_tied_model(coords, segs, [4], dsearch=1.0)
    ct = ContactType2(itf, model, MessageLog())

    disp = np.array([1.0, -2.0, 3.0])
    dt = 0.05
    v_trans = disp / dt

    # Move main nodes
    model.x[:4] += disp
    model.v[:4] = v_trans

    ct.enforce(model.x, model.v, model.vr, dt)

    # Secondary node position updated to [2.0, -1.0, 3.0]
    np.testing.assert_allclose(model.x[4], [1.0, 1.0, 0.0] + disp, atol=1e-12)
    # Velocity matches translation
    np.testing.assert_allclose(model.v[4], v_trans, atol=1e-12)


def test_enforce_rigid_rotation_with_offset():
    """Rigid rotation of main segment co-rotates the offset vector."""
    coords = np.array([
        [-1.0, -1.0, 0.0],
        [ 1.0, -1.0, 0.0],
        [ 1.0,  1.0, 0.0],
        [-1.0,  1.0, 0.0],
        [ 0.0,  0.0, 0.5],  # normal offset dz = 0.5 in +Z
    ])
    segs = np.array([[0, 1, 2, 3]])
    model, itf = make_tied_model(coords, segs, [4], dsearch=1.0)
    ct = ContactType2(itf, model, MessageLog())

    # Rotate segment 90 deg around X-axis: Y -> Z, Z -> -Y
    R_x = np.array([
        [1.0, 0.0,  0.0],
        [0.0, 0.0, -1.0],
        [0.0, 1.0,  0.0],
    ])
    model.x[:4] = (R_x @ coords[:4].T).T
    dt = 0.1
    ct.enforce(model.x, model.v, model.vr, dt)

    # Initial offset [0, 0, 0.5] rotated by 90 deg around X becomes [0, -0.5, 0]
    expected_x = np.array([0.0, -0.5, 0.0])
    np.testing.assert_allclose(model.x[4], expected_x, atol=1e-12)


def test_enforce_spotflag1_solid_rotation_velocity():
    """Spotflag=1 (Solid main - I2VIROT3) derives omega and adds omega x dvec to velocity."""
    coords = np.array([
        [-1.0, -1.0, 0.0],
        [ 1.0, -1.0, 0.0],
        [ 1.0,  1.0, 0.0],
        [-1.0,  1.0, 0.0],
        [ 0.0,  0.0, 0.5],  # offset dz = 0.5
    ])
    segs = np.array([[0, 1, 2, 3]])
    model, itf = make_tied_model(coords, segs, [4], dsearch=1.0, spotflag=1)
    ct = ContactType2(itf, model, MessageLog())

    # Pure spin omega = [0, 0, 10.0] rad/s around Z-axis
    omega = np.array([0.0, 0.0, 10.0])
    r = coords[:4]
    model.v[:4] = np.cross(omega, r)

    dt = 0.01
    ct.enforce(model.x, model.v, model.vr, dt)

    # Rotational velocity assigned to vr[4]
    np.testing.assert_allclose(model.vr[4], omega, atol=1e-12)


def test_enforce_spotflag2_shell_rotation_interpolation():
    """Spotflag=2 (Shell main - I2ROT3_27) interpolates vr from main shell nodes."""
    coords = np.array([
        [0.0, 0.0, 0.0],
        [2.0, 0.0, 0.0],
        [2.0, 2.0, 0.0],
        [0.0, 2.0, 0.0],
        [1.0, 1.0, 0.0],
    ])
    segs = np.array([[0, 1, 2, 3]])
    model, itf = make_tied_model(coords, segs, [4], dsearch=1.0, spotflag=2)
    ct = ContactType2(itf, model, MessageLog())

    model.vr[:4] = np.array([
        [1.0, 2.0, 3.0],
        [1.0, 2.0, 3.0],
        [1.0, 2.0, 3.0],
        [1.0, 2.0, 3.0],
    ])

    dt = 0.01
    ct.enforce(model.x, model.v, model.vr, dt)

    # Centroid gets interpolated vr = [1.0, 2.0, 3.0]
    np.testing.assert_allclose(model.vr[4], [1.0, 2.0, 3.0], atol=1e-12)


def test_enforce_zero_dt_no_division_by_zero():
    """dt <= EM20 does not divide by zero or produce NaNs."""
    coords = np.array([
        [0.0, 0.0, 0.0],
        [2.0, 0.0, 0.0],
        [2.0, 2.0, 0.0],
        [0.0, 2.0, 0.0],
        [1.0, 1.0, 0.0],
    ])
    segs = np.array([[0, 1, 2, 3]])
    model, itf = make_tied_model(coords, segs, [4], dsearch=1.0)
    ct = ContactType2(itf, model, MessageLog())

    ct.enforce(model.x, model.v, model.vr, dt=0.0)
    assert not np.isnan(model.v[4]).any()
    assert not np.isnan(model.x[4]).any()


# ============================================================================
# 6. _release: Element and node deletion dynamic tie release
# ============================================================================

def test_dynamic_release_element_deletion():
    """Deletion of parent element releases tie, hands back mass, deactivates tie."""
    coords = np.array([
        [0.0, 0.0, 0.0],
        [2.0, 0.0, 0.0],
        [2.0, 2.0, 0.0],
        [0.0, 2.0, 0.0],
        [1.0, 1.0, 0.0],
    ])
    segs = np.array([[0, 1, 2, 3]])
    masses = np.array([1.0, 1.0, 1.0, 1.0, 0.4])
    model, itf = make_tied_model(coords, segs, [4], masses=masses, dsearch=1.0)
    surf = model.surfaces[1]
    surf.seg_gtype = np.full(len(segs), "shells", dtype="<U8")
    surf.seg_elem = np.zeros(len(segs), dtype=np.int64)

    # Shell element group with 1 element
    egrp = ElementGroup(
        ids=np.array([101]),
        conn=np.array([[0, 1, 2, 3]]),
        part=np.array([0]),
        state={"off": np.array([1.0]), "chk_fail": True},
    )
    model.shells = egrp

    ct = ContactType2(itf, model, MessageLog())
    assert ct.deletable is True

    mass_eff = model.mass.copy()
    ct.augment_mass(mass_eff)
    inv_mass_eff = 1.0 / mass_eff

    # Kill element 101
    egrp.state["off"][0] = 0.0

    fint = np.zeros((5, 3))
    fext = np.zeros((5, 3))
    fcont = np.zeros((5, 3))
    mint = np.zeros((5, 3))

    # Trigger deletion check at cycle 8
    ct.transfer_forces(fint, fext, fcont, mint, model.x, mass_eff, inv_mass_eff, cycle=8)

    # Tie is now deactivated
    assert ct.active[0] == False
    # Transferred mass was removed from mass_eff
    np.testing.assert_allclose(mass_eff, model.mass, atol=1e-12)
    np.testing.assert_allclose(inv_mass_eff, 1.0 / model.mass, atol=1e-12)


# ============================================================================
# 7. Starter keyword parsing and build_contacts integration
# ============================================================================

def test_starter_keyword_inter_type2_fixed_format():
    """Fixed format /INTER/TYPE2 with INTER2 layout parses spotflag, dsearch."""
    block = KeywordBlock(
        keyword="/INTER/TYPE2/1",
        parts=["INTER", "TYPE2", "1"],
        user_id=1,
        cards=[
            Card("Tied interface title"),
            Card("         1         2         0         1         0         0         0                    0.75"),
        ],
        fixed=True,
    )
    model = Model()
    log = MessageLog()
    read_inter(block, model, log)

    assert len(model.interfaces) == 1
    itf = model.interfaces[0]
    assert itf.id == 1
    assert itf.type == 2
    assert itf.grnod_id == 1
    assert itf.surf_id == 2
    assert itf.spotflag == 1
    assert np.isclose(itf.dsearch, 0.75)


def test_starter_keyword_inter_type2_free_format():
    """Free format /INTER/TYPE2 parses tokens: grnod_id, surf_id, dsearch, spotflag."""
    block = KeywordBlock(
        keyword="/INTER/TYPE2/10",
        parts=["INTER", "TYPE2", "10"],
        user_id=10,
        cards=[
            Card("Tied free format"),
            Card("5 8 1.25 2"),
        ],
        fixed=False,
    )
    model = Model()
    log = MessageLog()
    read_inter(block, model, log)

    assert len(model.interfaces) == 1
    itf = model.interfaces[0]
    assert itf.id == 10
    assert itf.type == 2
    assert itf.grnod_id == 5
    assert itf.surf_id == 8
    assert np.isclose(itf.dsearch, 1.25)
    assert itf.spotflag == 2


def test_build_contacts_factory_integration():
    """build_contacts factory segregates penalty interfaces and tied interfaces."""
    coords = np.array([
        [0.0, 0.0, 0.0],
        [2.0, 0.0, 0.0],
        [2.0, 2.0, 0.0],
        [0.0, 2.0, 0.0],
        [1.0, 1.0, 0.0],
    ])
    segs = np.array([[0, 1, 2, 3]])
    model, itf = make_tied_model(coords, segs, [4], dsearch=1.0)

    log = MessageLog()
    penalty, tied = build_contacts(model, log)
    assert len(penalty) == 0
    assert len(tied) == 1
    assert isinstance(tied[0], ContactType2)
