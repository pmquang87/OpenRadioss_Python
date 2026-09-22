"""Unit tests for enhanced /SECT section-force output with local skew projection and stresses.

Upstream Fortran references:
- engine/source/tools/sect/section_skew.F
- engine/source/tools/sect/section_c.F
- engine/source/tools/sect/cutmass.F
- engine/source/tools/sect/section.F
"""

import math
import numpy as np
import pytest

from pyradioss.engine.sections import (
    SectionForces,
    SectionDefinition,
    SectionResult,
    build_skew_from_nodes,
    compute_facet_area_and_centroid,
    compute_cut_section_properties,
    project_to_skew,
    compute_nominal_stresses,
)


class _MockSect:
    """Mock section entity allowing arbitrary attributes."""
    def __init__(self, id, **kwargs):
        self.id = id
        for k, v in kwargs.items():
            setattr(self, k, v)


def _make_sf(sections_data, model=None):
    """Create a SectionForces instance with explicit section definitions.

    Each item in sections_data is either:
    - (sect_obj, side_idx, ref_node_or_neg1, x_ref0)
    - or sect_obj with .side_idx, .ref_node, .x_ref0
    """
    sf = object.__new__(SectionForces)
    sf.model = model
    sf.sections = []
    for item in sections_data:
        if isinstance(item, tuple) and len(item) >= 4:
            sc, side, ref, x0 = item[:4]
            sf.sections.append((sc, np.asarray(side, dtype=int), ref, x0))
        else:
            sf.sections.append(item)
    return sf


class TestLocalSkewProjection:
    """Test transformation of global forces and moments into local skew frames."""

    def test_without_skew_local_matches_global(self):
        """When no skew is defined (skew_id=0, R=identity), local components match global exactly."""
        sc = _MockSect(1, skew_id=0)
        sf = _make_sf([(sc, [0], -1, np.array([0.0, 0.0, 0.0]))])

        x = np.array([[0.0, 0.0, 0.0]])
        fint = np.array([[150.0, -250.0, 350.0]])
        mint = np.array([[15.0, -25.0, 35.0]])

        result = sf.compute(x, fint, mint)
        res = result[1]

        # Global forces and moments
        np.testing.assert_allclose(res.F, [150.0, -250.0, 350.0])
        np.testing.assert_allclose(res.M, [15.0, -25.0, 35.0])

        # Local forces and moments match global exactly without skew
        np.testing.assert_allclose(res.F_local, res.F)
        np.testing.assert_allclose(res.M_local, res.M)

        # Named component properties
        assert res.N == 150.0
        assert res.Vy == -250.0
        assert res.Vz == 350.0
        assert res.Mx == 15.0
        assert res.My == -25.0
        assert res.Mz == 35.0

    def test_90_degree_yaw_skew_swaps_fx_and_fy(self):
        """A 90-degree yaw skew (about Z) swaps global Fx and Fy into axial (N) and shear (Vy) components."""
        # 90 deg counter-clockwise rotation about Z:
        # Local X' = (0, 1, 0) (points along global +Y)
        # Local Y' = (-1, 0, 0) (points along global -X)
        # Local Z' = (0, 0, 1) (points along global +Z)
        R = np.array([
            [0.0, -1.0, 0.0],
            [1.0,  0.0, 0.0],
            [0.0,  0.0, 1.0],
        ])

        sc = _MockSect(10, R=R)
        sf = _make_sf([(sc, [0], -1, np.array([0.0, 0.0, 0.0]))])

        x = np.array([[0.0, 0.0, 0.0]])
        fint = np.array([[100.0, 250.0, -50.0]])
        mint = np.array([[10.0, 25.0, -5.0]])

        result = sf.compute(x, fint, mint)
        res = result[10]

        # F_local = R.T @ F_global
        # R.T = [[0, 1, 0], [-1, 0, 0], [0, 0, 1]]
        # F_local = [F_y, -F_x, F_z] = [250.0, -100.0, -50.0]
        np.testing.assert_allclose(res.F_local, [250.0, -100.0, -50.0])
        assert res.N == 250.0
        assert res.Vy == -100.0
        assert res.Vz == -50.0

        # M_local = [M_y, -M_x, M_z] = [25.0, -10.0, -5.0]
        np.testing.assert_allclose(res.M_local, [25.0, -10.0, -5.0])
        assert res.Mx == 25.0
        assert res.My == -10.0
        assert res.Mz == -5.0

    def test_3_node_skew_definition(self):
        """A skew defined by node triplet (N1, N2, N3) builds the local frame per section_skew.F."""
        # N1 at origin (0,0,0), N2 along +Y (0,2,0), N3 along +Z (0,0,1)
        x = np.array([
            [0.0, 0.0, 0.0],  # 0: side node
            [0.0, 0.0, 0.0],  # 1: N1 (origin)
            [0.0, 5.0, 0.0],  # 2: N2 (primary axis along +Y)
            [0.0, 0.0, 3.0],  # 3: N3 (plane vector in Y-Z plane)
        ])

        sc = _MockSect(1, skew_nodes=(1, 2, 3))
        sf = _make_sf([(sc, [0], -1, np.array([0.0, 0.0, 0.0]))])

        fint = np.array([[10.0, 20.0, 30.0], [0, 0, 0], [0, 0, 0], [0, 0, 0]])
        mint = np.zeros((4, 3))

        result = sf.compute(x, fint, mint)
        res = result[1]

        # Primary axis X' is along +Y: [0, 1, 0]
        # In-plane plane is Y-Z, so Z' is along +X or +Z:
        # X' x (N3-N1) = (0, 1, 0) x (0, 0, 3) = (3, 0, 0) -> ez = (1, 0, 0)
        # ey = ez x ex = (1, 0, 0) x (0, 1, 0) = (0, 0, 1)
        np.testing.assert_allclose(res.R[:, 0], [0.0, 1.0, 0.0])
        np.testing.assert_allclose(res.R[:, 1], [0.0, 0.0, 1.0])
        np.testing.assert_allclose(res.R[:, 2], [1.0, 0.0, 0.0])

        # F_local = [F_y, F_z, F_x] = [20.0, 30.0, 10.0]
        np.testing.assert_allclose(res.F_local, [20.0, 30.0, 10.0])

    def test_skew_from_model_skews(self):
        """Integration with model.skews lookup via skew_id."""
        class MockSkewSet:
            def __init__(self):
                # Row 0: global identity, Row 1: 90-deg yaw
                self.axes = np.zeros((2, 3, 3))
                self.axes[0] = np.eye(3)
                # rows of axes are X', Y', Z'
                self.axes[1] = np.array([
                    [0.0, 1.0, 0.0],   # X' = +Y
                    [-1.0, 0.0, 0.0],  # Y' = -X
                    [0.0, 0.0, 1.0],   # Z' = +Z
                ])

            def index(self, kind, user_id):
                if kind == "SKEW" and user_id == 42:
                    return 1
                return 0

        class MockModel:
            def __init__(self):
                self.skews = MockSkewSet()

        sc = _MockSect(1, skew_id=42)
        sf = _make_sf([(sc, [0], -1, np.zeros(3))], model=MockModel())

        x = np.zeros((1, 3))
        fint = np.array([[50.0, 80.0, 10.0]])
        mint = np.zeros((1, 3))

        result = sf.compute(x, fint, mint)
        res = result[1]

        # F_local = [F_y, -F_x, F_z] = [80.0, -50.0, 10.0]
        np.testing.assert_allclose(res.F_local, [80.0, -50.0, 10.0])


class TestSectionPhysicsAndStresses:
    """Test physical beam/bar mechanics: cantilever bending and axial tension."""

    def test_cantilever_beam_bending_moment(self):
        """Bending moment M at root of cantilever beam matches tip force F times length L."""
        length = 10.0
        f_tip = 500.0  # Transverse force in +Y direction

        # Node 0 at root x=0, Node 1 at tip x=L
        x = np.array([
            [0.0, 0.0, 0.0],      # Node 0 (root)
            [length, 0.0, 0.0],   # Node 1 (tip)
        ])

        # Section cut at root (x_ref = [0, 0, 0]), side set contains tip node 1
        sc = _MockSect(100, node_id_ref=0)
        sf = _make_sf([(sc, [1], 0, None)])

        fint = np.array([
            [0.0, 0.0, 0.0],
            [0.0, f_tip, 0.0],
        ])
        mint = np.zeros((2, 3))

        result = sf.compute(x, fint, mint)
        res = result[100]

        # Force is F_y = 500
        np.testing.assert_allclose(res.F, [0.0, f_tip, 0.0])

        # Moment about root: r x F = (L, 0, 0) x (0, F, 0) = (0, 0, L * F)
        expected_moment_z = length * f_tip
        np.testing.assert_allclose(res.M, [0.0, 0.0, expected_moment_z])
        assert res.Mz == expected_moment_z

    def test_tension_bar_normal_stress(self):
        """Uniform tension bar under axial load F has average normal stress sigma = F / A."""
        area = 0.02  # 0.02 m^2 (e.g. 10cm x 20cm)
        axial_pull = 50000.0  # 50 kN

        sc = _MockSect(1, area=area)
        sf = _make_sf([(sc, [0], -1, np.zeros(3))])

        x = np.zeros((1, 3))
        fint = np.array([[axial_pull, 0.0, 0.0]])
        mint = np.zeros((1, 3))

        result = sf.compute(x, fint, mint)
        res = result[1]

        expected_sigma = axial_pull / area  # 2.5 MPa = 2.5e6 Pa
        assert res.N == axial_pull
        assert res.area == area
        assert abs(res.sigma_avg - expected_sigma) < 1e-10
        assert res.tau == 0.0
        assert res.sigma_bending == 0.0
        assert abs(res.sigma_max - expected_sigma) < 1e-10
        assert abs(res.sigma_vm - expected_sigma) < 1e-10

    def test_combined_bending_shear_stresses(self):
        """Verify normal stress, shear stress, and peak bending stress under combined loading."""
        area = 0.01        # 0.01 m^2
        S_bending = 0.0005  # 0.0005 m^3 section modulus

        sc = _MockSect(1, area=area, S_bending=S_bending)
        sf = _make_sf([(sc, [0], -1, np.zeros(3))])

        x = np.zeros((1, 3))
        # Axial force N = 10000 N, Shear forces Vy = 3000 N, Vz = 4000 N
        # Moments My = 600 N*m, Mz = 800 N*m
        fint = np.array([[10000.0, 3000.0, 4000.0]])
        mint = np.array([[0.0, 600.0, 800.0]])

        result = sf.compute(x, fint, mint)
        res = result[1]

        # Normal stress: N / A = 10000 / 0.01 = 1.0 MPa
        assert abs(res.sigma_avg - 1.0e6) < 1e-10

        # Transverse shear: sqrt(3000^2 + 4000^2) / 0.01 = 5000 / 0.01 = 0.5 MPa
        assert abs(res.tau - 0.5e6) < 1e-10

        # Bending stress: sqrt(600^2 + 800^2) / 0.0005 = 1000 / 0.0005 = 2.0 MPa
        assert abs(res.sigma_bending - 2.0e6) < 1e-10

        # Max and min normal stresses
        assert abs(res.sigma_max - (1.0e6 + 2.0e6)) < 1e-10
        assert abs(res.sigma_min - (1.0e6 - 2.0e6)) < 1e-10

        # Von Mises: sqrt((3.0e6)^2 + 3 * (0.5e6)^2)
        expected_vm = math.sqrt((3.0e6) ** 2 + 3.0 * (0.5e6) ** 2)
        assert abs(res.sigma_vm - expected_vm) < 1e-6


class TestCutFacetGeometryAndInertia:
    """Test geometric integration of cut facets for area, centroid, and moments of inertia."""

    def test_triangle_facet(self):
        """Facet area and centroid for 3-node cut facet."""
        coords = np.array([[0.0, 0.0, 0.0], [3.0, 0.0, 0.0], [0.0, 4.0, 0.0]])
        area, centroid = compute_facet_area_and_centroid(coords)

        assert abs(area - 6.0) < 1e-10
        np.testing.assert_allclose(centroid, [1.0, 4.0 / 3.0, 0.0])

    def test_quad_facet(self):
        """Facet area and centroid for 4-node planar cut facet."""
        coords = np.array([
            [0.0, -1.0, -2.0],
            [0.0,  1.0, -2.0],
            [0.0,  1.0,  2.0],
            [0.0, -1.0,  2.0],
        ])
        area, centroid = compute_facet_area_and_centroid(coords)

        # 2 x 4 rectangle in Y-Z plane -> area = 8.0
        assert abs(area - 8.0) < 1e-10
        np.testing.assert_allclose(centroid, [0.0, 0.0, 0.0])

    def test_section_properties_from_facets(self):
        """Calculate total area, second moments of inertia, and section modulus from cut facets."""
        # Section composed of two 2x2 square facets in Y-Z plane:
        # Facet 1: centered at y=0, z=-1 (from z=-2 to z=0)
        # Facet 2: centered at y=0, z=+1 (from z=0 to z=+2)
        # Total rectangle is width b=2 (along Y) and height h=4 (along Z)
        f1 = np.array([
            [0.0, -1.0, -2.0],
            [0.0,  1.0, -2.0],
            [0.0,  1.0,  0.0],
            [0.0, -1.0,  0.0],
        ])
        f2 = np.array([
            [0.0, -1.0,  0.0],
            [0.0,  1.0,  0.0],
            [0.0,  1.0,  2.0],
            [0.0, -1.0,  2.0],
        ])

        sc = _MockSect(5, facets=[f1, f2])
        sf = _make_sf([(sc, [0], -1, np.zeros(3))])

        x = np.zeros((1, 3))
        fint = np.array([[1000.0, 0.0, 0.0]])
        mint = np.zeros((1, 3))

        result = sf.compute(x, fint, mint)
        res = result[5]

        # Total area A = 4.0 + 4.0 = 8.0
        assert abs(res.area - 8.0) < 1e-10

        # For discrete facets:
        # F1: area=4, z=-1 -> Iyy_k = 4 * (-1)^2 = 4
        # F2: area=4, z=+1 -> Iyy_k = 4 * (+1)^2 = 4
        # Total Iyy = 8.0
        assert abs(res.Iyy - 8.0) < 1e-10

        # Normal stress: 1000 / 8 = 125.0
        assert abs(res.sigma_avg - 125.0) < 1e-10
