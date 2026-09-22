"""
Unit tests for Work Stream 7:
- solid_hexa8z: Corotational 8-node hexahedral solid element
- splines: Catmull-Rom splines (knots, interpolation, arc length, point projection)
- polygon_clip: Weiler-Atherton polygon clipping and polygon utilities
- graph: Graph connected components and cycle detection
"""

import numpy as np
import pytest

from pyradioss.common.graph import Graph, connected_components, find_cycles
from pyradioss.common.polygon_clip import (
    Polygon,
    PolygonPoint,
    clipping_weiler_atherton,
    intersect_pt,
    polygon_is_point_inside,
    polygon_set_clockwise,
)
from pyradioss.common.splines import (
    cr_spline_interpol,
    cr_spline_knots,
    cr_spline_length,
    cr_spline_point_proj,
)
from pyradioss.elements import solid_hexa8z


# ============================================================================
# 1. Catmull-Rom Splines Tests
# ============================================================================

class TestSplines:
    def test_cr_spline_knots_uniform(self):
        pts = np.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [3.0, 0.0, 0.0],
            [6.0, 0.0, 0.0],
        ])
        knots = cr_spline_knots(pts, alpha=0.0)
        np.testing.assert_allclose(knots, [0.0, 1.0, 2.0, 3.0])

    def test_cr_spline_knots_centripetal(self):
        pts = np.array([
            [0.0, 0.0, 0.0],
            [4.0, 0.0, 0.0],
            [13.0, 0.0, 0.0],
            [29.0, 0.0, 0.0],
        ])
        # dists: 4, 9, 16 -> sqrt(dists) = 2, 3, 4
        knots = cr_spline_knots(pts, alpha=0.5)
        np.testing.assert_allclose(knots, [0.0, 2.0, 5.0, 9.0])

    def test_cr_spline_knots_chordal(self):
        pts = np.array([
            [0.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [5.0, 0.0, 0.0],
            [10.0, 0.0, 0.0],
        ])
        # dists: 2, 3, 5
        knots = cr_spline_knots(pts, alpha=1.0)
        np.testing.assert_allclose(knots, [0.0, 2.0, 5.0, 10.0])

    def test_cr_spline_interpol_endpoints(self):
        pts = np.array([
            [0.0, 0.0, 0.0],
            [1.0, 2.0, 0.0],
            [4.0, 5.0, 0.0],
            [6.0, 4.0, 0.0],
        ])
        knots = cr_spline_knots(pts, alpha=0.5)

        # t = 0 should pass through P1
        c0, cd0, cdd0 = cr_spline_interpol(pts, knots, t=0.0)
        np.testing.assert_allclose(c0, pts[1], atol=1e-12)

        # t = 1 should pass through P2
        c1, cd1, cdd1 = cr_spline_interpol(pts, knots, t=1.0)
        np.testing.assert_allclose(c1, pts[2], atol=1e-12)

    def test_cr_spline_interpol_collinear(self):
        pts = np.array([
            [-2.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [10.0, 0.0, 0.0],
            [12.0, 0.0, 0.0],
        ])
        knots = cr_spline_knots(pts, alpha=0.0)

        # Midpoint t = 0.5
        c_mid, cd_mid, cdd_mid = cr_spline_interpol(pts, knots, t=0.5)
        np.testing.assert_allclose(c_mid, [5.0, 0.0, 0.0], atol=1e-10)
        # Acceleration along straight line is zero
        np.testing.assert_allclose(cdd_mid, [0.0, 0.0, 0.0], atol=1e-10)

    def test_cr_spline_length_straight_line(self):
        pts = np.array([
            [-5.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [10.0, 0.0, 0.0],
            [15.0, 0.0, 0.0],
        ])
        length_full = cr_spline_length(pts, alpha=0.5, t=1.0)
        np.testing.assert_allclose(length_full, 10.0, rtol=1e-3)

        length_half = cr_spline_length(pts, alpha=0.5, t=0.5)
        np.testing.assert_allclose(length_half, 5.0, rtol=1e-3)

    def test_cr_spline_point_proj(self):
        pts = np.array([
            [-5.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [10.0, 0.0, 0.0],
            [15.0, 0.0, 0.0],
        ])
        # Test point offset along Y axis at X=6.0
        z = np.array([6.0, 3.0, 0.0])
        zh, h, t = cr_spline_point_proj(pts, z, alpha=0.5)

        np.testing.assert_allclose(zh, [6.0, 0.0, 0.0], atol=1e-5)
        np.testing.assert_allclose(h, 3.0, atol=1e-5)
        np.testing.assert_allclose(t, 0.6, atol=1e-5)


# ============================================================================
# 2. Polygon Clipping and Utilities Tests
# ============================================================================

class TestPolygonClip:
    def test_intersect_pt(self):
        p1 = PolygonPoint(0.0, 0.0)
        p2 = PolygonPoint(2.0, 2.0)
        q1 = PolygonPoint(0.0, 2.0)
        q2 = PolygonPoint(2.0, 0.0)

        ipt, alpha, beta = intersect_pt(p1, p2, q1, q2)
        assert ipt is not None
        np.testing.assert_allclose([ipt.y, ipt.z], [1.0, 1.0], atol=1e-12)
        np.testing.assert_allclose([alpha, beta], [0.5, 0.5], atol=1e-12)

    def test_intersect_pt_parallel_or_disjoint(self):
        p1 = PolygonPoint(0.0, 0.0)
        p2 = PolygonPoint(2.0, 0.0)
        q1 = PolygonPoint(0.0, 1.0)
        q2 = PolygonPoint(2.0, 1.0)

        ipt, alpha, beta = intersect_pt(p1, p2, q1, q2)
        assert ipt is None

    def test_polygon_area_and_clockwise(self):
        # Square with corners (0,0), (2,0), (2,2), (0,2)
        poly = Polygon([
            PolygonPoint(0.0, 0.0),
            PolygonPoint(2.0, 0.0),
            PolygonPoint(2.0, 2.0),
            PolygonPoint(0.0, 2.0),
        ])
        polygon_set_clockwise(poly)
        np.testing.assert_allclose(poly.area, 4.0, atol=1e-12)

    def test_polygon_is_point_inside(self):
        poly = Polygon([
            PolygonPoint(0.0, 0.0),
            PolygonPoint(4.0, 0.0),
            PolygonPoint(4.0, 4.0),
            PolygonPoint(0.0, 4.0),
        ])
        assert polygon_is_point_inside(poly, PolygonPoint(2.0, 2.0)) is True
        assert polygon_is_point_inside(poly, PolygonPoint(5.0, 2.0)) is False
        assert polygon_is_point_inside(poly, PolygonPoint(-1.0, -1.0)) is False

    def test_clipping_weiler_atherton_overlap(self):
        # Subject: unit square [0, 1] x [0, 1]
        subj = Polygon([
            PolygonPoint(0.0, 0.0),
            PolygonPoint(1.0, 0.0),
            PolygonPoint(1.0, 1.0),
            PolygonPoint(0.0, 1.0),
        ])
        # Clip: shifted rectangle [0.5, 1.5] x [0, 1]
        clip = Polygon([
            PolygonPoint(0.5, -0.5),
            PolygonPoint(1.5, -0.5),
            PolygonPoint(1.5, 1.5),
            PolygonPoint(0.5, 1.5),
        ])
        res = clipping_weiler_atherton(subj, clip)
        assert len(res) == 1
        # Intersection is [0.5, 1.0] x [0.0, 1.0] -> area = 0.5
        np.testing.assert_allclose(res[0].area, 0.5, atol=1e-4)

    def test_clipping_weiler_atherton_inside(self):
        # Subject completely inside clip
        subj = Polygon([
            PolygonPoint(0.2, 0.2),
            PolygonPoint(0.8, 0.2),
            PolygonPoint(0.8, 0.8),
            PolygonPoint(0.2, 0.8),
        ])
        clip = Polygon([
            PolygonPoint(0.0, 0.0),
            PolygonPoint(1.0, 0.0),
            PolygonPoint(1.0, 1.0),
            PolygonPoint(0.0, 1.0),
        ])
        res = clipping_weiler_atherton(subj, clip)
        assert len(res) == 1
        np.testing.assert_allclose(res[0].area, 0.36, atol=1e-4)

    def test_clipping_weiler_atherton_disjoint(self):
        subj = Polygon([
            PolygonPoint(0.0, 0.0),
            PolygonPoint(1.0, 0.0),
            PolygonPoint(1.0, 1.0),
            PolygonPoint(0.0, 1.0),
        ])
        clip = Polygon([
            PolygonPoint(5.0, 5.0),
            PolygonPoint(6.0, 5.0),
            PolygonPoint(6.0, 6.0),
            PolygonPoint(5.0, 6.0),
        ])
        res = clipping_weiler_atherton(subj, clip)
        assert len(res) == 0


# ============================================================================
# 3. Graph Connected Components and Cycle Tests
# ============================================================================

class TestGraph:
    def test_connected_components_basic(self):
        # 5 vertices: {0, 1, 2} form one component, {3, 4} form another
        connections = [(0, 1), (1, 2), (2, 0), (3, 4)]
        comps = connected_components(5, connections)
        assert len(comps) == 2
        comp_sets = [set(c) for c in comps]
        assert {0, 1, 2} in comp_sets
        assert {3, 4} in comp_sets

    def test_graph_cycle_detection(self):
        # Triangle (cycle) + isolated line (not a cycle)
        connections = [(0, 1), (1, 2), (2, 0), (3, 4)]
        is_cycle, paths = find_cycles(5, connections)
        assert len(is_cycle) == 2
        # One component is a cycle, the other is not
        assert True in is_cycle
        assert False in is_cycle

    def test_graph_square_cycle_traversal(self):
        # 4-node square cycle: 0 - 1 - 2 - 3 - 0
        connections = [(0, 1), (1, 2), (2, 3), (3, 0)]
        g = Graph(4, connect_list=connections)
        is_cycle = g.build_cycle()
        assert is_cycle == [True]
        cycle_path = g.get_path()[0]
        assert len(cycle_path) == 4
        # Verify valid cyclic ordering
        assert set(cycle_path) == {0, 1, 2, 3}
        for i in range(4):
            u = cycle_path[i]
            v = cycle_path[(i + 1) % 4]
            assert v in g.get_adj_list()[u]


# ============================================================================
# 4. Corotational 8-Node Solid (solid_hexa8z) Tests
# ============================================================================

class _MockGroup:
    def __init__(self, conn, slices):
        self.n = len(conn)
        self.conn = np.asarray(conn, dtype=np.int64)
        self.ids = np.arange(1, self.n + 1, dtype=np.int64)
        self.state = {"slices": slices}


class _MockMat:
    def __init__(self, E=210000.0, nu=0.3, rho0=7.8e-9):
        self.E = E
        self.nu = nu
        self.rho0 = rho0
        self.K = E / (3.0 * (1.0 - 2.0 * nu))
        self.G = E / (2.0 * (1.0 + nu))
        self.law = 1  # linear elastic /MAT/LAW1

    def sound_speed_solid(self):
        return np.sqrt((self.K + 4.0 * self.G / 3.0) / self.rho0)


class _MockProp:
    def __init__(self):
        self.params = {"qa": 1.1, "qb": 0.05, "h": 0.1}


class _MockModel:
    def __init__(self, x0):
        self.x0 = np.asarray(x0, dtype=np.float64)


def _build_single_cube_nodes(size=1.0):
    # Standard Radioss brick node order:
    # bottom face CCW: (0,0,0), (1,0,0), (1,1,0), (0,1,0)
    # top face CCW: (0,0,1), (1,0,1), (1,1,1), (0,1,1)
    s = size
    return np.array([
        [0.0, 0.0, 0.0],
        [s,   0.0, 0.0],
        [s,   s,   0.0],
        [0.0, s,   0.0],
        [0.0, 0.0, s],
        [s,   0.0, s],
        [s,   s,   s],
        [0.0, s,   s],
    ], dtype=np.float64)


class TestSolidHexa8z:
    def test_corotational_frame_identity(self):
        cube = _build_single_cube_nodes(size=1.0)
        xe = cube[None, :, :]
        R = solid_hexa8z._corotational_frame(xe)
        assert R.shape == (1, 3, 3)
        # For an axis-aligned cube, R should be identity
        np.testing.assert_allclose(R[0], np.eye(3), atol=1e-10)

    def test_corotational_frame_rotation(self):
        # Rotate cube by 90 degrees around Z axis: (x, y) -> (-y, x)
        cube = _build_single_cube_nodes(size=1.0)
        R_applied = np.array([
            [0.0, -1.0, 0.0],
            [1.0,  0.0, 0.0],
            [0.0,  0.0, 1.0],
        ])
        rotated_cube = cube @ R_applied.T
        xe = rotated_cube[None, :, :]
        R_calc = solid_hexa8z._corotational_frame(xe)

        # Invariance: R_calc must be orthonormal
        np.testing.assert_allclose(R_calc[0] @ R_calc[0].T, np.eye(3), atol=1e-10)
        # Local coordinates must match original unrotated cube coordinates
        xe_loc = np.einsum("nia,nab->nib", xe, R_calc)
        np.testing.assert_allclose(xe_loc[0], cube, atol=1e-10)

    def test_init_group(self):
        nodes = _build_single_cube_nodes(size=2.0)
        conn = np.arange(8)[None, :]
        mat = _MockMat()
        prop = _MockProp()
        slices = [(slice(0, 1), mat, prop)]
        group = _MockGroup(conn, slices)
        model = _MockModel(nodes)

        node_idx, mass_c, _ = solid_hexa8z.init_group(group, model, None)

        assert group.state["vol0"][0] == pytest.approx(8.0)
        expected_mass = mat.rho0 * 8.0
        assert group.state["mass"][0] == pytest.approx(expected_mass)
        # Lumped mass per node is 1/8 of total mass
        np.testing.assert_allclose(mass_c, expected_mass / 8.0)
        assert len(node_idx) == 8

    def test_forces_rigid_body_translation(self):
        # Pure rigid body translation velocity should produce zero internal forces
        nodes = _build_single_cube_nodes(size=1.0)
        conn = np.arange(8)[None, :]
        mat = _MockMat()
        prop = _MockProp()
        slices = [(slice(0, 1), mat, prop)]
        group = _MockGroup(conn, slices)
        model = _MockModel(nodes)
        solid_hexa8z.init_group(group, model, None)

        v = np.tile([100.0, -50.0, 25.0], (8, 1))
        fint = np.zeros((8, 3))
        dt = 1.0e-6

        dt_crit = solid_hexa8z.forces(group, nodes, v, None, dt, fint, None)

        assert dt_crit[0] > 0.0
        # Forces must be zero
        np.testing.assert_allclose(fint, 0.0, atol=1e-10)
        # Internal energy must remain zero
        assert group.state["eint"][0] == pytest.approx(0.0, abs=1e-12)

    def test_forces_uniform_tension_energy(self):
        # Apply uniform velocity stretching along X: node 1, 2, 5, 6 moving in +X
        nodes = _build_single_cube_nodes(size=1.0)
        conn = np.arange(8)[None, :]
        mat = _MockMat()
        prop = _MockProp()
        slices = [(slice(0, 1), mat, prop)]
        group = _MockGroup(conn, slices)
        model = _MockModel(nodes)
        solid_hexa8z.init_group(group, model, None)

        v = np.zeros((8, 3))
        # Nodes with x=1.0 get vx = 10.0
        v[[1, 2, 5, 6], 0] = 10.0
        fint = np.zeros((8, 3))
        dt = 1.0e-6

        dt_crit = solid_hexa8z.forces(group, nodes, v, None, dt, fint, None)

        assert dt_crit[0] > 0.0
        # Positive tension -> sigma_xx > 0
        assert group.state["sig"][0, 0] > 0.0
        # Work done generates positive internal energy
        assert group.state["eint"][0] > 0.0
        # Net internal force in X direction: nodes pulled in +X experience negative restoring force
        assert np.all(fint[[1, 2, 5, 6], 0] < 0.0)
        # Left face nodes (x=0) experience positive force
        assert np.all(fint[[0, 3, 4, 7], 0] > 0.0)
        # Self-equilibrium: sum of internal forces equals 0
        np.testing.assert_allclose(np.sum(fint, axis=0), 0.0, atol=1e-10)

    def test_forces_deleted_element(self):
        nodes = _build_single_cube_nodes(size=1.0)
        conn = np.arange(8)[None, :]
        mat = _MockMat()
        prop = _MockProp()
        slices = [(slice(0, 1), mat, prop)]
        group = _MockGroup(conn, slices)
        model = _MockModel(nodes)
        solid_hexa8z.init_group(group, model, None)

        group.state["off"][0] = 0.0  # Deleted
        v = np.ones((8, 3)) * 10.0
        fint = np.zeros((8, 3))
        dt = 1.0e-6

        dt_crit = solid_hexa8z.forces(group, nodes, v, None, dt, fint, None)
        assert dt_crit[0] >= 1.0e20
        np.testing.assert_allclose(fint, 0.0)
