"""Unit tests for /INIVOL Multi-Material ALE Volume Fraction Initialization.

Upstream Fortran references:
- starter/source/initial_conditions/inivol/inifill.F
- starter/source/initial_conditions/inivol/in_out_side.F
- starter/source/initial_conditions/inivol/ratio_fill.F
"""

import math
import numpy as np
import pytest

from pyradioss.materials.inivol import (
    BoxContainer,
    CylinderContainer,
    GeometricShapeType,
    InivolFiller,
    InivolParams,
    InivolResult,
    SphereContainer,
    SurfaceFacetContainer,
    initialize_volume_fraction,
)


@pytest.fixture
def background_box_mesh():
    """Create a 3D Cartesian background mesh of 20x20x20 brick elements covering [-1, 1]^3."""
    nx, ny, nz = 20, 20, 20
    lx, ly, lz = 2.0, 2.0, 2.0
    dx, dy, dz = lx / nx, ly / ny, lz / nz
    elem_vol = dx * dy * dz  # 0.1 * 0.1 * 0.1 = 0.001 m^3

    xs = np.linspace(-1.0, 1.0, nx + 1)
    ys = np.linspace(-1.0, 1.0, ny + 1)
    zs = np.linspace(-1.0, 1.0, nz + 1)

    elem_coords = []
    volumes = []

    for k in range(nz):
        for j in range(ny):
            for i in range(nx):
                min_b = [xs[i], ys[j], zs[k]]
                max_b = [xs[i + 1], ys[j + 1], zs[k + 1]]
                elem_coords.append(np.array([min_b, max_b]))
                volumes.append(elem_vol)

    return elem_coords, np.array(volumes, dtype=float)


class TestInivolPrimitives:
    """Test geometric container primitives and analytical volumes."""

    def test_sphere_container(self):
        sphere = SphereContainer(center=[0.0, 0.0, 0.0], radius=0.5)
        assert math.isclose(sphere.analytical_volume, (4.0 / 3.0) * math.pi * (0.5**3), rel_tol=1e-6)

        pts = np.array([
            [0.0, 0.0, 0.0],     # inside (center)
            [0.3, 0.3, 0.0],     # inside (r = 0.424 < 0.5)
            [0.5, 0.0, 0.0],     # on surface
            [0.51, 0.0, 0.0],    # outside
            [1.0, 1.0, 1.0],     # outside
        ])
        inside = sphere.is_inside(pts)
        np.testing.assert_array_equal(inside, [True, True, True, False, False])

    def test_cylinder_container(self):
        # Cylinder along Z-axis: radius 0.4, height 1.0, base at (0, 0, 0)
        cyl = CylinderContainer(center=[0.0, 0.0, 0.0], axis=[0.0, 0.0, 1.0], radius=0.4, height=1.0)
        assert math.isclose(cyl.analytical_volume, math.pi * (0.4**2) * 1.0, rel_tol=1e-6)

        pts = np.array([
            [0.0, 0.0, 0.5],    # inside (axis, mid-height)
            [0.3, 0.0, 0.5],    # inside (r = 0.3 < 0.4)
            [0.5, 0.0, 0.5],    # outside (r = 0.5 > 0.4)
            [0.0, 0.0, 1.2],    # outside (h = 1.2 > 1.0)
            [0.0, 0.0, -0.1],   # outside (h = -0.1 < 0.0)
        ])
        inside = cyl.is_inside(pts)
        np.testing.assert_array_equal(inside, [True, True, False, False, False])

    def test_box_container(self):
        box = BoxContainer(min_bounds=[-0.2, -0.3, -0.4], max_bounds=[0.2, 0.3, 0.4])
        expected_vol = 0.4 * 0.6 * 0.8
        assert math.isclose(box.analytical_volume, expected_vol, rel_tol=1e-6)

        pts = np.array([
            [0.0, 0.0, 0.0],     # inside
            [0.19, 0.29, 0.39],  # inside
            [0.21, 0.0, 0.0],    # outside x
            [0.0, -0.31, 0.0],   # outside y
            [0.0, 0.0, 0.45],    # outside z
        ])
        inside = box.is_inside(pts)
        np.testing.assert_array_equal(inside, [True, True, False, False, False])


class TestInivolMeshFilling:
    """Test background mesh geometric filling and cut-cell volume integration."""

    def test_sphere_fill_volume_and_cut_cells(self, background_box_mesh):
        """Verify sphere fill in 3D background box mesh produces alpha in [0, 1] and accurate volume."""
        elem_coords, volumes = background_box_mesh
        r_sphere = 0.5
        rho0 = 1000.0
        p0 = 2.0e5
        e0 = 4.0e4

        sphere = SphereContainer(center=[0.0, 0.0, 0.0], radius=r_sphere)
        res = initialize_volume_fraction(
            container=sphere,
            elements_coords=elem_coords,
            volumes=volumes,
            rho0=rho0,
            p0=p0,
            e0=e0,
            subdivisions=3,
        )

        assert isinstance(res, InivolResult)
        alphas = res.volume_fractions
        assert np.all(alphas >= 0.0)
        assert np.all(alphas <= 1.0)

        # Check cut cells: must have elements with 0 < alpha < 1
        cut_cells = (alphas > 0.01) & (alphas < 0.99)
        assert np.any(cut_cells)

        # Check fully inside elements (near center)
        full_inside = (alphas > 0.99)
        assert np.any(full_inside)

        # Analytical sphere volume: 4/3 * pi * R^3 = 4/3 * pi * 0.125 = 0.523598
        expected_vol = sphere.analytical_volume
        assert math.isclose(res.total_filled_volume, expected_vol, rel_tol=0.015)

        # Check mass: M = rho0 * V
        expected_mass = rho0 * expected_vol
        assert math.isclose(res.total_filled_mass, expected_mass, rel_tol=0.015)

        # Check internal energy: E = M * e0
        expected_energy = expected_mass * e0
        assert math.isclose(res.total_internal_energy, expected_energy, rel_tol=0.015)

    def test_box_primitive_fill(self, background_box_mesh):
        """Verify exact box fill on aligned mesh."""
        elem_coords, volumes = background_box_mesh
        # Box from [-0.3, -0.3, -0.3] to [0.3, 0.3, 0.3] aligned with grid (step 0.1)
        box = BoxContainer(min_bounds=[-0.3, -0.3, -0.3], max_bounds=[0.3, 0.3, 0.3])
        expected_vol = 0.6 * 0.6 * 0.6  # 0.216

        res = initialize_volume_fraction(
            container=box,
            elements_coords=elem_coords,
            volumes=volumes,
            subdivisions=2,
        )

        assert math.isclose(res.total_filled_volume, expected_vol, rel_tol=1e-5)

    def test_cylinder_primitive_fill(self, background_box_mesh):
        """Verify cylinder fill volume accuracy."""
        elem_coords, volumes = background_box_mesh
        r_cyl = 0.4
        h_cyl = 0.8
        cyl = CylinderContainer(center=[0.0, 0.0, -0.4], axis=[0.0, 0.0, 1.0], radius=r_cyl, height=h_cyl)
        expected_vol = math.pi * (r_cyl**2) * h_cyl  # ~0.40212

        res = initialize_volume_fraction(
            container=cyl,
            elements_coords=elem_coords,
            volumes=volumes,
            subdivisions=3,
        )

        assert math.isclose(res.total_filled_volume, expected_vol, rel_tol=0.015)

    def test_reversed_fill(self, background_box_mesh):
        """Verify ireversed=1 fills the complement domain (outside container)."""
        elem_coords, volumes = background_box_mesh
        sphere = SphereContainer(center=[0.0, 0.0, 0.0], radius=0.4)

        # Inside fill
        res_in = initialize_volume_fraction(
            container=sphere,
            elements_coords=elem_coords,
            volumes=volumes,
            subdivisions=2,
            ireversed=0,
        )
        # Outside fill
        res_out = initialize_volume_fraction(
            container=sphere,
            elements_coords=elem_coords,
            volumes=volumes,
            subdivisions=2,
            ireversed=1,
        )

        # Complementarity: alpha_in + alpha_out == 1 everywhere
        np.testing.assert_allclose(res_in.volume_fractions + res_out.volume_fractions, 1.0, atol=1e-6)

        # Total volume should sum to full domain volume (2*2*2 = 8.0)
        tot_domain = float(np.sum(volumes))
        assert math.isclose(res_in.total_filled_volume + res_out.total_filled_volume, tot_domain, rel_tol=1e-5)


class TestSurfaceFacetContainer:
    """Test ray-casting in_out_side algorithm on closed triangulated surface mesh."""

    def test_triangulated_cube_surface(self):
        """Create a closed triangulated cube of side 0.8 centered at origin."""
        # 8 vertices
        s = 0.4
        nodes = np.array([
            [-s, -s, -s],  # 0
            [ s, -s, -s],  # 1
            [ s,  s, -s],  # 2
            [-s,  s, -s],  # 3
            [-s, -s,  s],  # 4
            [ s, -s,  s],  # 5
            [ s,  s,  s],  # 6
            [-s,  s,  s],  # 7
        ])
        # 6 quad faces -> 12 triangles
        facets = [
            (0, 1, 2), (0, 2, 3),  # bottom (-z)
            (4, 6, 5), (4, 7, 6),  # top (+z)
            (0, 5, 1), (0, 4, 5),  # front (-y)
            (2, 7, 3), (2, 6, 7),  # back (+y)
            (0, 7, 4), (0, 3, 7),  # left (-x)
            (1, 6, 2), (1, 5, 6),  # right (+x)
        ]

        surf = SurfaceFacetContainer(nodes=nodes, facets=facets)

        # Analytical volume: 0.8^3 = 0.512
        assert math.isclose(surf.analytical_volume, 0.8**3, rel_tol=1e-5)

        # Test point queries
        test_pts = np.array([
            [0.0, 0.0, 0.0],    # inside (center)
            [0.35, 0.35, 0.35], # inside (near corner)
            [0.45, 0.0, 0.0],   # outside x
            [0.0, -0.5, 0.0],   # outside y
            [0.0, 0.0, 0.6],    # outside z
        ])
        inside = surf.is_inside(test_pts)
        np.testing.assert_array_equal(inside, [True, True, False, False, False])

    def test_mesh_fill_with_surface_facets(self, background_box_mesh):
        """Verify fill_mesh using SurfaceFacetContainer."""
        elem_coords, volumes = background_box_mesh
        s = 0.3
        nodes = np.array([
            [-s, -s, -s], [ s, -s, -s], [ s,  s, -s], [-s,  s, -s],
            [-s, -s,  s], [ s, -s,  s], [ s,  s,  s], [-s,  s,  s],
        ])
        facets = [
            (0, 1, 2), (0, 2, 3),
            (4, 6, 5), (4, 7, 6),
            (0, 5, 1), (0, 4, 5),
            (2, 7, 3), (2, 6, 7),
            (0, 7, 4), (0, 3, 7),
            (1, 6, 2), (1, 5, 6),
        ]
        surf = SurfaceFacetContainer(nodes=nodes, facets=facets)
        expected_vol = (2.0 * s) ** 3  # 0.6^3 = 0.216

        res = initialize_volume_fraction(
            container=surf,
            elements_coords=elem_coords,
            volumes=volumes,
            subdivisions=2,
        )

        assert math.isclose(res.total_filled_volume, expected_vol, rel_tol=1e-4)


class TestGeneralHexElement:
    """Test filling arbitrary 8-node hexahedron element."""

    def test_8_node_hex_filling(self):
        # A 1x1x1 hexahedron with 8 nodes
        hex_nodes = np.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 1.0],
            [1.0, 1.0, 1.0],
            [0.0, 1.0, 1.0],
        ])
        vol = 1.0

        # Half box [0, 0.5] x [0, 1] x [0, 1]
        half_box = BoxContainer(min_bounds=[0.0, 0.0, 0.0], max_bounds=[0.5, 1.0, 1.0])
        params = InivolParams(container=half_box, subdivisions=4)
        filler = InivolFiller(params)

        alpha = filler.compute_element_fill(hex_nodes, vol)
        assert math.isclose(alpha, 0.5, rel_tol=1e-5)
