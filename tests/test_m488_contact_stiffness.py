"""
Unit tests for pyradioss/contact/stiffness.py — Contact Penalty Stiffness & Gap Calculation (M488).

Validates:
- _fallback_modulus: max Young's modulus selection and empty-model default.
- _segment_areas: 4-node quads, 3-node triangles, arbitrary 3D orientations, degenerate segments, empty arrays.
- segment_stiffness_gap: shell segments, solid segments, fallback segments, scale factors, empty arrays.
- node_stiffness_gap: max-scatter over shells and solids, shared nodes, non-contact elements, empty nodes.
- segment_mesh_gap: Igap=3 minimum edge length calculation, quad vs triangle handling, empty segments.
- node_mesh_gap: Igap=3 minimum mesh gap scattering across nodes, unconnected nodes.
- edge_stiffness_gap: TYPE11 edge stiffness for shells, solids, and fallback lines.
- combine_stiffness: Istf=0..5 combinations, Ks=0 fallback handling, ValueError on invalid Istf.
"""

from __future__ import annotations

import types
import numpy as np
import pytest

from pyradioss.model.model import Model
from pyradioss.contact.stiffness import (
    _fallback_modulus,
    _segment_areas,
    _per_element,
    segment_stiffness_gap,
    node_stiffness_gap,
    segment_mesh_gap,
    node_mesh_gap,
    edge_stiffness_gap,
    combine_stiffness,
)


# ============================================================================
# Helpers to construct lightweight mock models and element groups
# ============================================================================

class MockMat:
    def __init__(self, E: float = 210000.0, nu: float = 0.3):
        self.E = E
        self.nu = nu
        # Bulk modulus B = E / (3 * (1 - 2*nu))
        self.K = E / (3.0 * (1.0 - 2.0 * nu))


class MockProp:
    def __init__(self, thick: float = 1.5):
        self.thick = thick


class MockElementGroup:
    def __init__(self, conn: np.ndarray, state: dict):
        self.conn = np.asarray(conn, dtype=int)
        self.n = len(self.conn)
        self.state = state


def make_test_model(numnod: int = 8) -> Model:
    model = Model()
    model.node_ids = np.arange(1, numnod + 1)
    model.x0 = np.zeros((numnod, 3), dtype=float)
    return model


# ============================================================================
# 1. _fallback_modulus
# ============================================================================

def test_fallback_modulus_multi_material():
    model = Model()
    model.materials[1] = MockMat(E=100.0)
    model.materials[2] = MockMat(E=210000.0)
    model.materials[3] = MockMat(E=70000.0)
    assert _fallback_modulus(model) == 210000.0


def test_fallback_modulus_empty_model():
    model = Model()
    model.materials.clear()
    assert _fallback_modulus(model) == 1.0


# ============================================================================
# 2. _segment_areas & _per_element
# ============================================================================

def test_segment_areas_unit_quad():
    x0 = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [1.0, 1.0, 0.0],
        [0.0, 1.0, 0.0],
    ])
    segs = np.array([[0, 1, 2, 3]])
    area = _segment_areas(x0, segs)
    assert np.isclose(area[0], 1.0)


def test_segment_areas_rectangular_quad():
    x0 = np.array([
        [0.0, 0.0, 0.0],
        [2.0, 0.0, 0.0],
        [2.0, 3.0, 0.0],
        [0.0, 3.0, 0.0],
    ])
    segs = np.array([[0, 1, 2, 3]])
    area = _segment_areas(x0, segs)
    assert np.isclose(area[0], 6.0)


def test_segment_areas_triangle():
    # Triangle with node 4 equal to node 3
    x0 = np.array([
        [0.0, 0.0, 0.0],
        [2.0, 0.0, 0.0],
        [0.0, 3.0, 0.0],
    ])
    segs = np.array([[0, 1, 2, 2]])
    area = _segment_areas(x0, segs)
    assert np.isclose(area[0], 3.0)  # 0.5 * 2 * 3 = 3


def test_segment_areas_3d_orientation():
    # Rotated quad in 3D: normal in direction (1, 1, 1)
    x0 = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 1.0],
        [1.0, 1.0, 2.0],
        [0.0, 1.0, 1.0],
    ])
    segs = np.array([[0, 1, 2, 3]])
    area = _segment_areas(x0, segs)
    assert np.isclose(area[0], np.sqrt(3.0))


def test_segment_areas_degenerate_and_empty():
    x0 = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [2.0, 0.0, 0.0],
        [3.0, 0.0, 0.0],
    ])
    # Collinear segment
    segs = np.array([[0, 1, 2, 3]])
    area = _segment_areas(x0, segs)
    assert np.isclose(area[0], 0.0)

    # Empty segments array
    empty_segs = np.zeros((0, 4), dtype=int)
    empty_area = _segment_areas(x0, empty_segs)
    assert empty_area.shape == (0,)


def test_per_element_helper():
    mat1 = MockMat(E=100.0)
    prop1 = MockProp(thick=2.0)
    mat2 = MockMat(E=200.0)
    prop2 = MockProp(thick=4.0)

    group = types.SimpleNamespace(
        n=4,
        state={
            "slices": [
                (slice(0, 2), mat1, prop1),
                (slice(2, 4), mat2, prop2),
            ]
        }
    )
    E_arr = _per_element(group, lambda m, p: m.E)
    t_arr = _per_element(group, lambda m, p: p.thick)
    np.testing.assert_allclose(E_arr, [100.0, 100.0, 200.0, 200.0])
    np.testing.assert_allclose(t_arr, [2.0, 2.0, 4.0, 4.0])


# ============================================================================
# 3. segment_stiffness_gap
# ============================================================================

def test_segment_stiffness_gap_shells():
    model = make_test_model(4)
    model.x0 = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [1.0, 1.0, 0.0],
        [0.0, 1.0, 0.0],
    ])
    mat = MockMat(E=200000.0)
    prop = MockProp(thick=2.0)
    model.materials[1] = mat

    model.shells = MockElementGroup(
        conn=[[0, 1, 2, 3]],
        state={
            "thick": np.array([2.0]),
            "slices": [(slice(0, 1), mat, prop)],
        }
    )

    segs = np.array([[0, 1, 2, 3]])
    seg_gtype = np.array(["shells"])
    seg_elem = np.array([0])
    stfac = 0.8
    fscale_gap = 1.2

    # K = 0.5 * stfac * E * t = 0.5 * 0.8 * 200000 * 2.0 = 160000.0
    # gap = 0.5 * t * fscale_gap = 0.5 * 2.0 * 1.2 = 1.2
    K, gap = segment_stiffness_gap(model, segs, seg_gtype, seg_elem, stfac, fscale_gap)
    assert np.isclose(K[0], 160000.0)
    assert np.isclose(gap[0], 1.2)


def test_segment_stiffness_gap_all_shell_families():
    # shells_qbat, shells_qeph, sh3n, sh3n_dkt18
    model = make_test_model(4)
    model.x0 = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [1.0, 1.0, 0.0],
        [0.0, 1.0, 0.0],
    ])
    mat = MockMat(E=100000.0)
    prop = MockProp(thick=1.0)
    model.materials[1] = mat

    for gname in ("shells_qbat", "shells_qeph", "sh3n", "sh3n_dkt18"):
        setattr(model, gname, MockElementGroup(
            conn=[[0, 1, 2, 3]],
            state={
                "thick": np.array([1.0]),
                "slices": [(slice(0, 1), mat, prop)],
            }
        ))
        segs = np.array([[0, 1, 2, 3]])
        seg_gtype = np.array([gname])
        seg_elem = np.array([0])
        K, gap = segment_stiffness_gap(model, segs, seg_gtype, seg_elem, stfac=1.0, fscale_gap=1.0)
        # K = 0.5 * 1.0 * 100000.0 * 1.0 = 50000.0, gap = 0.5
        assert np.isclose(K[0], 50000.0)
        assert np.isclose(gap[0], 0.5)


def test_segment_stiffness_gap_solids():
    model = make_test_model(8)
    model.x0 = np.array([
        [0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0], [1.0, 0.0, 1.0], [1.0, 1.0, 1.0], [0.0, 1.0, 1.0],
    ])
    # E = 210000, nu = 0.3 -> B = 210000 / (3 * 0.4) = 175000.0
    mat = MockMat(E=210000.0, nu=0.3)
    prop = MockProp()
    model.materials[1] = mat

    # Unit cube: V = 1.0, Face area = 1.0
    model.bricks = MockElementGroup(
        conn=[[0, 1, 2, 3, 4, 5, 6, 7]],
        state={
            "vol0": np.array([1.0]),
            "slices": [(slice(0, 1), mat, prop)],
        }
    )

    segs = np.array([[0, 1, 2, 3]])
    seg_gtype = np.array(["bricks"])
    seg_elem = np.array([0])
    stfac = 0.5

    # K = stfac * B * A^2 / V = 0.5 * 175000 * (1.0^2) / 1.0 = 87500.0
    # gap = 0.0 (solids contact on their real face)
    K, gap = segment_stiffness_gap(model, segs, seg_gtype, seg_elem, stfac)
    assert np.isclose(K[0], 87500.0)
    assert np.isclose(gap[0], 0.0)


def test_segment_stiffness_gap_fallback():
    # Segments without parent element (e.g. /SURF/SEG)
    model = make_test_model(4)
    model.x0 = np.array([
        [0.0, 0.0, 0.0],
        [2.0, 0.0, 0.0],
        [2.0, 2.0, 0.0],
        [0.0, 2.0, 0.0],
    ])
    mat = MockMat(E=160000.0)
    model.materials[1] = mat

    segs = np.array([[0, 1, 2, 3]])
    seg_gtype = np.array([""])
    seg_elem = np.array([0])
    stfac = 0.5

    # Area = 4.0, sqrt(A) = 2.0
    # K = stfac * E_ref * sqrt(A) = 0.5 * 160000 * 2.0 = 160000.0
    # gap = 0.0
    K, gap = segment_stiffness_gap(model, segs, seg_gtype, seg_elem, stfac)
    assert np.isclose(K[0], 160000.0)
    assert np.isclose(gap[0], 0.0)


def test_segment_stiffness_gap_empty():
    model = make_test_model(4)
    segs = np.zeros((0, 4), dtype=int)
    seg_gtype = np.zeros(0, dtype=str)
    seg_elem = np.zeros(0, dtype=int)
    K, gap = segment_stiffness_gap(model, segs, seg_gtype, seg_elem, stfac=1.0)
    assert K.shape == (0,)
    assert gap.shape == (0,)


# ============================================================================
# 4. node_stiffness_gap
# ============================================================================

def test_node_stiffness_gap_single_shell():
    model = make_test_model(4)
    mat = MockMat(E=200000.0)
    prop = MockProp(thick=2.0)
    model.materials[1] = mat

    model.shells = MockElementGroup(
        conn=[[0, 1, 2, 3]],
        state={
            "thick": np.array([2.0]),
            "slices": [(slice(0, 1), mat, prop)],
        }
    )

    stfac = 0.5
    fscale_gap = 1.0
    K, gap = node_stiffness_gap(model, stfac, fscale_gap)

    # All 4 nodes: k_e = 0.5 * 0.5 * 200000.0 * 2.0 = 100000.0
    # gap = 0.5 * 2.0 * 1.0 = 1.0
    expected_K = [100000.0, 100000.0, 100000.0, 100000.0]
    expected_gap = [1.0, 1.0, 1.0, 1.0]
    np.testing.assert_allclose(K, expected_K)
    np.testing.assert_allclose(gap, expected_gap)


def test_node_stiffness_gap_single_solid():
    model = make_test_model(8)
    mat = MockMat(E=210000.0, nu=0.3)
    prop = MockProp()
    model.materials[1] = mat
    # V = 8.0, V^(1/3) = 2.0, B = 175000.0
    model.bricks = MockElementGroup(
        conn=[[0, 1, 2, 3, 4, 5, 6, 7]],
        state={
            "vol0": np.array([8.0]),
            "slices": [(slice(0, 1), mat, prop)],
        }
    )

    stfac = 1.0
    K, gap = node_stiffness_gap(model, stfac)
    # k_e = stfac * B * V^(1/3) = 1.0 * 175000.0 * 2.0 = 350000.0
    # gap = 0.0
    np.testing.assert_allclose(K, 350000.0)
    np.testing.assert_allclose(gap, 0.0)


def test_node_stiffness_gap_shared_nodes_max_scatter():
    # 5 nodes:
    # Element 1 (nodes 0, 1, 2, 3): soft shell, E=100000, thick=1.0 -> K=50000, gap=0.5
    # Element 2 (nodes 1, 2, 4, 3): stiff shell, E=300000, thick=2.0 -> K=300000, gap=1.0
    # Shared nodes (1, 2, 3) must inherit the maximum stiffness and gap.
    # Unshared node 0 inherits element 1; unshared node 4 inherits element 2.
    model = make_test_model(5)
    mat1 = MockMat(E=100000.0)
    prop1 = MockProp(thick=1.0)
    mat2 = MockMat(E=300000.0)
    prop2 = MockProp(thick=2.0)
    model.materials[1] = mat1
    model.materials[2] = mat2

    model.shells = MockElementGroup(
        conn=[[0, 1, 2, 3], [1, 2, 4, 3]],
        state={
            "thick": np.array([1.0, 2.0]),
            "slices": [
                (slice(0, 1), mat1, prop1),
                (slice(1, 2), mat2, prop2),
            ],
        }
    )

    K, gap = node_stiffness_gap(model, stfac=1.0)
    assert np.isclose(K[0], 50000.0)
    assert np.isclose(gap[0], 0.5)

    for nid in (1, 2, 3):
        assert np.isclose(K[nid], 300000.0)
        assert np.isclose(gap[nid], 1.0)

    assert np.isclose(K[4], 300000.0)
    assert np.isclose(gap[4], 1.0)


def test_node_stiffness_gap_non_contact_elements_ignored():
    # Trusses, springs, beams do not contribute face contact stiffness
    model = make_test_model(2)
    mat = MockMat(E=200000.0)
    prop = MockProp()
    model.materials[1] = mat
    model.trusses = MockElementGroup(conn=[[0, 1]], state={"slices": [(slice(0, 1), mat, prop)]})
    model.springs = MockElementGroup(conn=[[0, 1]], state={"slices": [(slice(0, 1), mat, prop)]})
    model.beams = MockElementGroup(conn=[[0, 1]], state={"slices": [(slice(0, 1), mat, prop)]})

    K, gap = node_stiffness_gap(model, stfac=1.0)
    np.testing.assert_allclose(K, 0.0)
    np.testing.assert_allclose(gap, 0.0)


# ============================================================================
# 5. segment_mesh_gap
# ============================================================================

def test_segment_mesh_gap_square_quad():
    model = make_test_model(4)
    model.x0 = np.array([
        [0.0, 0.0, 0.0],
        [10.0, 0.0, 0.0],
        [10.0, 10.0, 0.0],
        [0.0, 10.0, 0.0],
    ])
    segs = np.array([[0, 1, 2, 3]])
    # All 4 edges have length 10.0 -> Lmin = 10.0
    # mesh_gap = 0.4 * 10.0 = 4.0
    gaps = segment_mesh_gap(model, segs, percent_mesh_size=0.4)
    assert np.isclose(gaps[0], 4.0)


def test_segment_mesh_gap_rectangular_quad():
    model = make_test_model(4)
    model.x0 = np.array([
        [0.0, 0.0, 0.0],
        [6.0, 0.0, 0.0],
        [6.0, 15.0, 0.0],
        [0.0, 15.0, 0.0],
    ])
    segs = np.array([[0, 1, 2, 3]])
    # Edges: 6.0, 15.0, 6.0, 15.0 -> Lmin = 6.0
    # percent_mesh_size = 0.5 -> 3.0
    gaps = segment_mesh_gap(model, segs, percent_mesh_size=0.5)
    assert np.isclose(gaps[0], 3.0)


def test_segment_mesh_gap_triangle():
    # Triangle with node 4 == node 3:
    # Nodes: (0,0), (3,0), (0,4), (0,4)
    # Edge 1-2: length 3
    # Edge 2-3: length 5 (hypotenuse)
    # Edge 3-4: length 0 (degenerate, masked with inf)
    # Edge 4-1: length 4
    # Minimum edge must be 3.0, NOT 0.0!
    model = make_test_model(3)
    model.x0 = np.array([
        [0.0, 0.0, 0.0],
        [3.0, 0.0, 0.0],
        [0.0, 4.0, 0.0],
    ])
    segs = np.array([[0, 1, 2, 2]])
    gaps = segment_mesh_gap(model, segs, percent_mesh_size=0.4)
    # Lmin = 3.0 -> gap = 0.4 * 3.0 = 1.2
    assert np.isclose(gaps[0], 1.2)


def test_segment_mesh_gap_empty():
    model = make_test_model(4)
    segs = np.zeros((0, 4), dtype=int)
    gaps = segment_mesh_gap(model, segs)
    assert gaps.shape == (0,)


# ============================================================================
# 6. node_mesh_gap
# ============================================================================

def test_node_mesh_gap_shared_nodes():
    # Segment 1: small quad of edge length 2.0 (nodes 0, 1, 2, 3) -> gap = 0.4 * 2 = 0.8
    # Segment 2: large quad of edge length 10.0 (nodes 1, 4, 5, 2) -> gap = 0.4 * 10 = 4.0
    # Node 6: isolated node -> gap = inf
    model = make_test_model(7)
    model.x0 = np.array([
        [0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [2.0, 2.0, 0.0], [0.0, 2.0, 0.0],
        [12.0, 0.0, 0.0], [12.0, 10.0, 0.0], [100.0, 100.0, 100.0]
    ])
    segs = np.array([
        [0, 1, 2, 3],
        [1, 4, 5, 2],
    ])
    nodes = np.array([0, 1, 4, 6])
    gaps = node_mesh_gap(model, segs, nodes, percent_mesh_size=0.4)

    # Node 0: connected only to seg 1 -> 0.8
    assert np.isclose(gaps[0], 0.8)
    # Node 1: connected to both seg 1 (0.8) and seg 2 (0.8) -> min is 0.8
    assert np.isclose(gaps[1], 0.8)
    # Node 4: connected only to seg 2 -> min edge of seg 2 is edge 2-1 (length 2.0) -> 0.8
    assert np.isclose(gaps[2], 0.8)
    # Node 6: unconnected to segments or elements -> fallback to master local mesh gap 0.8 (BUG-CONT-03)
    assert np.isclose(gaps[3], 0.8)


# ============================================================================
# 7. edge_stiffness_gap
# ============================================================================

def test_edge_stiffness_gap_shells():
    model = make_test_model(4)
    model.x0 = np.array([
        [0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0]
    ])
    mat = MockMat(E=150000.0)
    prop = MockProp(thick=2.0)
    model.materials[1] = mat

    model.shells = MockElementGroup(
        conn=[[0, 1, 2, 3]],
        state={
            "thick": np.array([2.0]),
            "slices": [(slice(0, 1), mat, prop)],
        }
    )

    edges = np.array([[0, 1]])
    seg_gtype = np.array(["shells"])
    seg_elem = np.array([0])
    stfac = 0.6

    # K = 0.5 * stfac * E * t = 0.5 * 0.6 * 150000 * 2.0 = 90000.0
    # gap = 0.5 * t = 0.5 * 2.0 = 1.0
    K, gap = edge_stiffness_gap(model, edges, seg_gtype, seg_elem, stfac)
    assert np.isclose(K[0], 90000.0)
    assert np.isclose(gap[0], 1.0)


def test_edge_stiffness_gap_solids():
    model = make_test_model(8)
    mat = MockMat(E=210000.0, nu=0.3)
    prop = MockProp()
    model.materials[1] = mat

    # V = 8.0, V^(1/3) = 2.0, B = 175000.0
    model.bricks = MockElementGroup(
        conn=[[0, 1, 2, 3, 4, 5, 6, 7]],
        state={
            "vol0": np.array([8.0]),
            "slices": [(slice(0, 1), mat, prop)],
        }
    )

    edges = np.array([[0, 1]])
    seg_gtype = np.array(["bricks"])
    seg_elem = np.array([0])
    stfac = 0.5

    # K = stfac * B * V^(1/3) = 0.5 * 175000 * 2.0 = 175000.0
    # gap = 0.0
    K, gap = edge_stiffness_gap(model, edges, seg_gtype, seg_elem, stfac)
    assert np.isclose(K[0], 175000.0)
    assert np.isclose(gap[0], 0.0)


def test_edge_stiffness_gap_fallback():
    # Explicit /LINE/SEG edge with length 5.0
    model = make_test_model(2)
    model.x0 = np.array([[0.0, 0.0, 0.0], [3.0, 4.0, 0.0]])
    mat = MockMat(E=100000.0)
    model.materials[1] = mat

    edges = np.array([[0, 1]])
    seg_gtype = np.array([""])
    seg_elem = np.array([0])
    stfac = 0.5

    # L = 5.0, K = stfac * E_ref * L = 0.5 * 100000 * 5.0 = 250000.0
    # gap = 0.0
    K, gap = edge_stiffness_gap(model, edges, seg_gtype, seg_elem, stfac)
    assert np.isclose(K[0], 250000.0)
    assert np.isclose(gap[0], 0.0)


def test_edge_stiffness_gap_empty():
    model = make_test_model(2)
    edges = np.zeros((0, 2), dtype=int)
    seg_gtype = np.zeros(0, dtype=str)
    seg_elem = np.zeros(0, dtype=int)
    K, gap = edge_stiffness_gap(model, edges, seg_gtype, seg_elem, stfac=1.0)
    assert K.shape == (0,)
    assert gap.shape == (0,)


# ============================================================================
# 8. combine_stiffness (Istf 0..5)
# ============================================================================

def test_combine_stiffness_istf0_main_only():
    Km = np.array([100.0, 200.0])
    Ks = np.array([50.0, 300.0])
    res = combine_stiffness(0, stfac=0.5, K_m=Km, K_s=Ks)
    np.testing.assert_allclose(res, Km)
    # Ensure independent copy returned
    assert res is not Km


def test_combine_stiffness_istf1_constant():
    Km = np.array([100.0, 200.0])
    Ks = np.array([50.0, 300.0])
    res = combine_stiffness(1, stfac=1234.5, K_m=Km, K_s=Ks)
    np.testing.assert_allclose(res, [1234.5, 1234.5])


def test_combine_stiffness_istf2_average():
    Km = np.array([100.0, 200.0])
    Ks = np.array([50.0, 400.0])
    res = combine_stiffness(2, stfac=1.0, K_m=Km, K_s=Ks)
    np.testing.assert_allclose(res, [75.0, 300.0])


def test_combine_stiffness_istf3_max():
    Km = np.array([100.0, 200.0])
    Ks = np.array([50.0, 400.0])
    res = combine_stiffness(3, stfac=1.0, K_m=Km, K_s=Ks)
    np.testing.assert_allclose(res, [100.0, 400.0])


def test_combine_stiffness_istf4_min():
    Km = np.array([100.0, 200.0])
    Ks = np.array([50.0, 400.0])
    res = combine_stiffness(4, stfac=1.0, K_m=Km, K_s=Ks)
    np.testing.assert_allclose(res, [50.0, 200.0])


def test_combine_stiffness_istf5_series_springs():
    # K = Km * Ks / (Km + Ks)
    # Pair 1: 200 * 200 / 400 = 100.0
    # Pair 2: 300 * 600 / 900 = 200.0
    Km = np.array([200.0, 300.0])
    Ks = np.array([200.0, 600.0])
    res = combine_stiffness(5, stfac=1.0, K_m=Km, K_s=Ks)
    np.testing.assert_allclose(res, [100.0, 200.0])


def test_combine_stiffness_secondary_zero_fallback():
    # If Ks is zero (e.g. isolated node or truss node), Ks is replaced with Km
    # so min and series do not collapse to zero
    Km = np.array([150.0])
    Ks = np.array([0.0])

    # Istf=2: average (150 + 150)/2 = 150
    np.testing.assert_allclose(combine_stiffness(2, 1.0, Km, Ks), [150.0])
    # Istf=3: max(150, 150) = 150
    np.testing.assert_allclose(combine_stiffness(3, 1.0, Km, Ks), [150.0])
    # Istf=4: min(150, 150) = 150 (not 0.0!)
    np.testing.assert_allclose(combine_stiffness(4, 1.0, Km, Ks), [150.0])
    # Istf=5: 150*150/300 = 75.0 (not 0.0!)
    np.testing.assert_allclose(combine_stiffness(5, 1.0, Km, Ks), [75.0])


def test_combine_stiffness_invalid_istf_raises():
    Km = np.array([100.0])
    Ks = np.array([100.0])
    with pytest.raises(ValueError, match="Istf=6"):
        combine_stiffness(6, 1.0, Km, Ks)
    with pytest.raises(ValueError, match="Istf=-1"):
        combine_stiffness(-1, 1.0, Km, Ks)


def test_combine_stiffness_empty_arrays():
    Km = np.zeros(0)
    Ks = np.zeros(0)
    for istf in range(6):
        res = combine_stiffness(istf, 1.0, Km, Ks)
        assert res.shape == (0,)
