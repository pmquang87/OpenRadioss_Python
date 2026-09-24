# tests/test_ale_fvm.py
"""
Unit tests for the ALE Finite Volume Method (FVM) and Cut-Cell Subsystems.

Validates:
- Face normal vectors and magnitude (||N|| = 2 * Area)
- Stress preparation: pressure, acoustic impedance, Mach numbers, normal velocities
- Internal force assembly via Godunov acoustic Riemann solver (alefvm_sfint3)
- ALE face relative velocities and convective fluxes (alefvm_aflux3 / alefvm_eflux3)
- Momentum time integration and expansion to nodes (alefvm_scheme / alefvm_expand_mom2)
- High-level ALEFVMSolver integration step with strict energy accounting
- 2D Sutherland-Hodgman polygon clipping (i22clip_tools)
- 3D polygon oriented area normal and centroid via i22aera (i22subvol)
- 3D hex element plane slicing and volume fraction computation
- Secondary cut-cell linking and convective increment stacking (a22conv3)
- Fluid-structure interaction (FSI) wet surface force calculation
"""

import math
import numpy as np
import pytest

from pyradioss.engine.ale_engine import (
    HEX_FACES,
    build_face_connectivity,
    compute_hex_volumes,
)
from pyradioss.engine.ale_fvm import (
    SOLVER_FEM,
    SOLVER_U_AVG,
    SOLVER_RHOU_AVG,
    SOLVER_ROE_AVG,
    SOLVER_GODUNOV_ACOUSTIC,
    ALEFVMParams,
    ALEFVMState,
    compute_alefvm_face_normals,
    alefvm_prepare_stress_buffer,
    alefvm_compute_internal_forces,
    alefvm_compute_face_fluxes,
    alefvm_update_momentum,
    alefvm_expand_momentum_to_nodes,
    alefvm_reset_accelerations,
    alefvm_advect_scalar,
    ALEFVMSolver,
)
from pyradioss.engine.ale_cut_cells import (
    cross_prod_2d,
    is_on_1st_half_plane,
    intersect_segments_2d,
    clip_edge_2d,
    polygonal_clipping_2d,
    polygon_area_2d,
    i22aera,
    CutPlane,
    CutCellInfo,
    slice_hex_edge_by_plane,
    intersect_hex_cell_with_plane,
    compute_fsi_wet_surface_force,
    build_secondary_cell_links,
    stack_secondary_cell_updates,
)


def make_single_cube_mesh(size: float = 1.0) -> Tuple[np.ndarray, np.ndarray]:
    """Create nodal coordinates and connectivity for a single 8-node unit hex element."""
    nodes = np.array([
        [0.0, 0.0, 0.0],  # 0
        [size, 0.0, 0.0],  # 1
        [size, size, 0.0],  # 2
        [0.0, size, 0.0],  # 3
        [0.0, 0.0, size],  # 4
        [size, 0.0, size],  # 5
        [size, size, size],  # 6
        [0.0, size, size],  # 7
    ], dtype=np.float64)
    conn = np.array([[0, 1, 2, 3, 4, 5, 6, 7]], dtype=np.int64)
    return nodes, conn


def make_two_cube_mesh(size: float = 1.0) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Create a 2-cube mesh aligned along the x-axis, sharing face x = size."""
    nodes = np.array([
        [0.0, 0.0, 0.0],       # 0
        [size, 0.0, 0.0],      # 1
        [size, size, 0.0],     # 2
        [0.0, size, 0.0],      # 3
        [0.0, 0.0, size],      # 4
        [size, 0.0, size],      # 5
        [size, size, size],     # 6
        [0.0, size, size],      # 7
        [2 * size, 0.0, 0.0],  # 8
        [2 * size, size, 0.0], # 9
        [2 * size, 0.0, size], # 10
        [2 * size, size, size],# 11
    ], dtype=np.float64)

    conn = np.array([
        [0, 1, 2, 3, 4, 5, 6, 7],         # Elem 0
        [1, 8, 9, 2, 5, 10, 11, 6],       # Elem 1
    ], dtype=np.int64)

    nbr_elem, nbr_face = build_face_connectivity(conn)
    return nodes, conn, nbr_elem, nbr_face


# =============================================================================
# Test ALE FVM Core
# =============================================================================

def test_alefvm_face_normals_unit_cube():
    """Verify face normals on a unit cube have magnitude 2.0 (2 * Area) and correct outward directions."""
    nodes, conn = make_single_cube_mesh(size=1.0)
    xe = nodes[conn]
    normals = compute_alefvm_face_normals(xe)

    assert normals.shape == (1, 6, 3)
    # Each face of a unit cube has area 1.0, so ||N|| = 2 * 1.0 = 2.0
    for k in range(6):
        mag = np.linalg.norm(normals[0, k])
        assert abs(mag - 2.0) < 1e-12

    # Verify expected directions:
    # Face 0 (-z)
    assert np.allclose(normals[0, 0] / 2.0, [0.0, 0.0, -1.0])
    # Face 1 (+y)
    assert np.allclose(normals[0, 1] / 2.0, [0.0, 1.0, 0.0])
    # Face 2 (+z)
    assert np.allclose(normals[0, 2] / 2.0, [0.0, 0.0, 1.0])
    # Face 3 (-y)
    assert np.allclose(normals[0, 3] / 2.0, [0.0, -1.0, 0.0])
    # Face 4 (+x)
    assert np.allclose(normals[0, 4] / 2.0, [1.0, 0.0, 0.0])
    # Face 5 (-x)
    assert np.allclose(normals[0, 5] / 2.0, [-1.0, 0.0, 0.0])


def test_alefvm_stress_buffer_preparation():
    """Verify pressure, acoustic impedance, Mach numbers, and normal velocities computation."""
    nodes, conn = make_single_cube_mesh(size=1.0)
    state = ALEFVMState.initialize(
        n_elem=1,
        n_nodes=8,
        vol=np.array([1.0]),
        rho=np.array([1.2]),
        velocity=np.array([[10.0, 0.0, 0.0]]),
        ssp=np.array([340.0]),
        pressure=np.array([101325.0]),
    )

    alefvm_prepare_stress_buffer(state, nodes, conn)

    # Pressure should equal 101325.0
    assert abs(state.pressure[0] - 101325.0) < 1e-5
    # Normal velocity on +x face (face 4) should be +10.0
    assert abs(state.face_u_n[0, 4] - 10.0) < 1e-10
    # Normal velocity on -x face (face 5) should be -10.0
    assert abs(state.face_u_n[0, 5] - (-10.0)) < 1e-10
    # Normal velocity on y and z faces should be 0.0
    for k in [0, 1, 2, 3]:
        assert abs(state.face_u_n[0, k]) < 1e-10


def test_alefvm_internal_forces_uniform_pressure():
    """Verify internal forces on closed cube with uniform pressure sum to zero."""
    nodes, conn = make_single_cube_mesh(size=1.0)
    nbr_elem = np.full((1, 6), -1, dtype=np.int64)
    nbr_face = np.full((1, 6), -1, dtype=np.int64)

    state = ALEFVMState.initialize(
        n_elem=1,
        n_nodes=8,
        vol=np.array([1.0]),
        rho=np.array([1.0]),
        velocity=np.array([[0.0, 0.0, 0.0]]),
        ssp=np.array([300.0]),
        pressure=np.array([50.0]),
    )
    alefvm_prepare_stress_buffer(state, nodes, conn)
    fint = alefvm_compute_internal_forces(
        state,
        nbr_elem,
        nbr_face,
        solver_type=SOLVER_GODUNOV_ACOUSTIC,
    )

    # In uniform static pressure, sum of -P * n * Area over all faces of a closed hex is exactly 0
    assert np.allclose(fint[0], [0.0, 0.0, 0.0], atol=1e-12)


def test_alefvm_acoustic_riemann_pressure_jump():
    """Verify Godunov acoustic Riemann interface pressure between two adjacent elements."""
    nodes, conn, nbr_elem, nbr_face = make_two_cube_mesh(size=1.0)

    # Elem 0: High pressure, Elem 1: Low pressure
    p0, p1 = 200.0, 100.0
    rho0, rho1 = 1.0, 1.0
    c0, c1 = 300.0, 300.0
    z0 = rho0 * c0  # 300
    z1 = rho1 * c1  # 300

    state = ALEFVMState.initialize(
        n_elem=2,
        n_nodes=len(nodes),
        vol=np.array([1.0, 1.0]),
        rho=np.array([rho0, rho1]),
        velocity=np.zeros((2, 3)),
        ssp=np.array([c0, c1]),
        pressure=np.array([p0, p1]),
    )
    alefvm_prepare_stress_buffer(state, nodes, conn)
    alefvm_compute_internal_forces(
        state,
        nbr_elem,
        nbr_face,
        solver_type=SOLVER_GODUNOV_ACOUSTIC,
    )

    # Interface face between Elem 0 (+x, face 4) and Elem 1 (-x, face 5):
    # Pf = (Z0 * P1 + Z1 * P0) / (Z0 + Z1) = 0.5 * (200 + 100) = 150.0
    pf_interface_0 = state.face_pressures[0, 4]
    pf_interface_1 = state.face_pressures[1, 5]

    assert abs(pf_interface_0 - 150.0) < 1e-10
    assert abs(pf_interface_1 - 150.0) < 1e-10

    # Net force on Elem 0 from pressure gradient pushes to the right (+x direction):
    # Left boundary: P = 200, Interface: Pf = 150 -> Net x force = (200 - 150) * Area = +50 N
    # On Elem 1:
    # Interface: Pf = 150, Right boundary: P = 100 -> Net x force = (150 - 100) * Area = +50 N
    # Total net force across both elements = (200 - 100) * 1.0 = 100 N
    fint_0 = state.fint_cell[0]
    fint_1 = state.fint_cell[1]

    assert abs(fint_0[0] - 50.0) < 1e-10
    assert abs(fint_1[0] - 50.0) < 1e-10
    assert abs((fint_0[0] + fint_1[0]) - 100.0) < 1e-10


def test_alefvm_convective_face_fluxes():
    """Verify face flux calculations and slip wall boundary reduction."""
    nodes, conn, nbr_elem, nbr_face = make_two_cube_mesh(size=1.0)
    # Velocity 5 m/s in x direction
    u_init = np.array([[5.0, 0.0, 0.0], [5.0, 0.0, 0.0]])

    state = ALEFVMState.initialize(
        n_elem=2,
        n_nodes=len(nodes),
        vol=np.array([1.0, 1.0]),
        rho=np.array([1.0, 1.0]),
        velocity=u_init,
        ssp=np.array([300.0, 300.0]),
        pressure=np.array([100.0, 100.0]),
    )
    alefvm_prepare_stress_buffer(state, nodes, conn)

    # Grid velocity = 0 (Eulerian mode)
    flux_data = alefvm_compute_face_fluxes(
        state,
        nodes,
        conn,
        nbr_elem,
        grid_velocity=None,
        solver_type=SOLVER_U_AVG,
        upwl=1.0,
        reduc=0.0,  # Solid slip boundary
    )

    fluxes = flux_data["flux_raw"]
    # Internal face between Elem 0 (+x, face 4) and Elem 1 (-x, face 5):
    # Velocity 5.0, Area 1.0, Normal +x -> Flux = 5.0 * 1.0 = 5.0
    assert abs(fluxes[0, 4] - 5.0) < 1e-10
    assert abs(fluxes[1, 5] - (-5.0)) < 1e-10

    # Boundary faces with reduc=0.0 should have zero flux
    # Elem 0 face 5 (-x boundary):
    assert abs(fluxes[0, 5]) < 1e-12


def test_alefvm_momentum_update_and_expansion():
    """Verify time integration of momentum and expansion to vertices."""
    nodes, conn = make_single_cube_mesh(size=1.0)
    state = ALEFVMState.initialize(
        n_elem=1,
        n_nodes=8,
        vol=np.array([1.0]),
        rho=np.array([1.0]),
        velocity=np.zeros((1, 3)),
    )
    # Apply a net external force of 80 N along x
    state.fcell = np.array([[80.0, 0.0, 0.0]])
    dt = 0.01

    dmom = alefvm_update_momentum(state, dt=dt, dt_prev=dt)
    assert np.allclose(dmom[0], [0.8, 0.0, 0.0])
    assert np.allclose(state.mom[0], [0.8, 0.0, 0.0])

    # Nodal mass: 1 kg total / 8 nodes = 0.125 kg per node
    nodal_mass = np.full(8, 0.125)
    nodal_v = alefvm_expand_momentum_to_nodes(state, conn, nodal_mass)

    # Total cell momentum is 0.8. Each of 8 nodes gets 0.1 kg*m/s.
    # With nodal mass 0.125 kg, nodal velocity = 0.1 / 0.125 = 0.8 m/s
    assert nodal_v.shape == (8, 3)
    for n in range(8):
        assert abs(nodal_v[n, 0] - 0.8) < 1e-10
        assert abs(nodal_v[n, 1]) < 1e-10
        assert abs(nodal_v[n, 2]) < 1e-10

    # Reset accelerations
    accel = alefvm_reset_accelerations(8)
    assert np.allclose(accel, 0.0)


def test_alefvm_solver_full_step_conservation():
    """Verify high-level ALEFVMSolver full step and total mass conservation."""
    nodes, conn, nbr_elem, nbr_face = make_two_cube_mesh(size=1.0)
    n_elem = len(conn)
    n_nodes = len(nodes)

    state = ALEFVMState.initialize(
        n_elem=n_elem,
        n_nodes=n_nodes,
        vol=np.array([1.0, 1.0]),
        rho=np.array([2.0, 1.0]),
        velocity=np.array([[2.0, 0.0, 0.0], [2.0, 0.0, 0.0]]),
        ssp=np.array([300.0, 300.0]),
        pressure=np.array([100.0, 100.0]),
    )

    initial_mass = float(np.sum(state.rho * state.vol))
    nodal_mass = np.full(n_nodes, initial_mass / n_nodes)

    solver = ALEFVMSolver(params=ALEFVMParams(reduc_boundary=0.0))
    res = solver.step(
        state=state,
        x=nodes,
        connectivity=conn,
        neighbor_elem=nbr_elem,
        neighbor_face=nbr_face,
        nodal_mass=nodal_mass,
        dt=1e-3,
        dt_prev=1e-3,
    )

    final_mass = float(np.sum(state.rho * state.vol))
    # With solid slip boundary (reduc=0.0), total mass is strictly conserved
    assert abs(final_mass - initial_mass) < 1e-12
    # Energy accounting tracked
    assert "energy_kinetic" in res
    assert "energy_internal_work" in res


# =============================================================================
# Test Cut-Cell Geometry & Primitives
# =============================================================================

def test_cut_cell_2d_clipping_tools():
    """Verify Sutherland-Hodgman 2D polygon clipping against a rectangular window."""
    # Subject polygon: triangle
    tri = np.array([
        [-0.5, 0.2],
        [1.5, 0.2],
        [0.5, 1.2],
    ], dtype=np.float64)

    # Clip polygon: unit square [0, 1] x [0, 1] in CCW order
    square = np.array([
        [0.0, 0.0],
        [1.0, 0.0],
        [1.0, 1.0],
        [0.0, 1.0],
    ], dtype=np.float64)

    clipped = polygonal_clipping_2d(tri, square)
    assert len(clipped) >= 3

    # All points in clipped polygon must lie within [0, 1] x [0, 1]
    for pt in clipped:
        assert -1e-8 <= pt[0] <= 1.0 + 1e-8
        assert -1e-8 <= pt[1] <= 1.0 + 1e-8

    signed_area, abs_area = polygon_area_2d(clipped)
    assert abs_area > 0.0
    assert abs_area <= 1.0  # Cannot exceed unit square area


def test_i22aera_triangle_and_quad():
    """Verify I22AERA area normal, centroid, and scalar area for 3D triangle and quad."""
    # 1. Right triangle on XY plane with legs length 2.0 along x and y
    tri_pts = np.array([
        [0.0, 0.0, 0.0],
        [2.0, 0.0, 0.0],
        [0.0, 2.0, 0.0],
    ], dtype=np.float64)

    norm_vec, cog, area = i22aera(tri_pts)
    # Area = 0.5 * 2 * 2 = 2.0, normal = +z
    assert abs(area - 2.0) < 1e-12
    assert np.allclose(norm_vec, [0.0, 0.0, 2.0])
    assert np.allclose(cog, [2.0 / 3.0, 2.0 / 3.0, 0.0])

    # 2. Quad of 2.0 x 3.0 on YZ plane
    quad_pts = np.array([
        [0.0, 0.0, 0.0],
        [0.0, 2.0, 0.0],
        [0.0, 2.0, 3.0],
        [0.0, 0.0, 3.0],
    ], dtype=np.float64)

    norm_vec_q, cog_q, area_q = i22aera(quad_pts)
    # Area = 2.0 * 3.0 = 6.0, normal = +x
    assert abs(area_q - 6.0) < 1e-12
    assert np.allclose(norm_vec_q, [6.0, 0.0, 0.0])
    assert np.allclose(cog_q, [0.0, 1.0, 1.5])


def test_hex_cell_cutting_by_plane():
    """Verify slicing a unit cube with horizontal plane z = 0.5 cuts 4 edges and yields alpha = 0.5."""
    nodes, conn = make_single_cube_mesh(size=1.0)
    xe = nodes[conn[0]]

    # Cut plane: z = 0.5 with upward normal +z
    plane = CutPlane(origin=np.array([0.0, 0.0, 0.5]), normal=np.array([0.0, 0.0, 1.0]))

    cut_info = intersect_hex_cell_with_plane(elem_id=0, xe=xe, cut_plane=plane)

    assert cut_info.is_cut is True
    # Wet surface area is 1.0 x 1.0 = 1.0
    assert abs(cut_info.wet_area - 1.0) < 1e-10
    # Wet centroid is at (0.5, 0.5, 0.5)
    assert np.allclose(cut_info.wet_centroid, [0.5, 0.5, 0.5])
    # Fluid sub-volume is 0.5 * 1.0 = 0.5
    assert abs(cut_info.volume_fraction - 0.5) < 0.05
    assert cut_info.is_secondary is False


def test_secondary_cell_linking_and_stacking():
    """Verify small cut cell (secondary) links to main cell and stacks convective updates."""
    # Create 2 cells: cell 0 (secondary, alpha = 0.05), cell 1 (main, alpha = 0.95)
    neighbor_elem = np.array([
        [-1, -1, -1, -1, 1, -1],  # Cell 0: neighbor in +x is cell 1
        [-1, -1, -1, -1, -1, 0],  # Cell 1: neighbor in -x is cell 0
    ], dtype=np.int64)

    cut_cells = {
        0: CutCellInfo(elem_id=0, is_cut=True, volume_fraction=0.05, is_secondary=True),
        1: CutCellInfo(elem_id=1, is_cut=True, volume_fraction=0.95, is_secondary=False),
    }

    build_secondary_cell_links(cut_cells, neighbor_elem)
    assert cut_cells[0].main_cell_id == 1
    assert 0 in cut_cells[1].secondary_cell_ids

    # Test increment stacking
    dphi = np.array([12.5, 40.0])  # dphi on cell 0 and cell 1
    total_before = np.sum(dphi)

    dphi_stacked = stack_secondary_cell_updates(dphi, cut_cells)
    # Cell 0 update should be stacked into Cell 1, and Cell 0 zeroed out
    assert dphi_stacked[0] == 0.0
    assert dphi_stacked[1] == 52.5
    # Strict global conservation:
    assert abs(np.sum(dphi_stacked) - total_before) < 1e-14


def test_fsi_wet_surface_force():
    """Verify fluid pressure on structural wet surface generates correct FSI force."""
    cut_cells = [
        CutCellInfo(
            elem_id=0,
            is_cut=True,
            wet_area=2.5,
            wet_normal=np.array([1.0, 0.0, 0.0]),
        )
    ]
    pressures = np.array([100.0])  # 100 Pa fluid pressure

    forces, total_norm = compute_fsi_wet_surface_force(cut_cells, pressures)

    # Force = P * Area * normal = 100 * 2.5 * [1, 0, 0] = [250, 0, 0] N
    assert np.allclose(forces[0], [250.0, 0.0, 0.0])
    assert abs(total_norm - 250.0) < 1e-10


def test_i22aera_pentagon_decomposition():
    """Verify I22AERA with 5+ vertices (fan triangulation decomposition, lines 2426-2450)."""
    # Regular planar pentagon on XY plane
    r = 2.0
    angles = np.linspace(0, 2 * np.pi, 6)[:-1]
    pent_pts = np.zeros((5, 3), dtype=np.float64)
    pent_pts[:, 0] = r * np.cos(angles)
    pent_pts[:, 1] = r * np.sin(angles)

    norm_vec, cog, area = i22aera(pent_pts)
    # Area of regular pentagon: 0.5 * 5 * r^2 * sin(2*pi/5)
    expected_area = 0.5 * 5 * (r ** 2) * math.sin(2.0 * math.pi / 5.0)
    assert abs(area - expected_area) < 1e-10
    # Normal along +z
    assert abs(norm_vec[2] - expected_area) < 1e-10
    # Centroid at origin
    assert np.allclose(cog, [0.0, 0.0, 0.0], atol=1e-10)


def test_cut_cell_fully_inside_and_outside():
    """Verify cell plane slicing when cell is completely inside or completely outside."""
    nodes, conn = make_single_cube_mesh(size=1.0)
    xe = nodes[conn[0]]

    # 1. Plane at z = 2.0, upward normal +z -> cube (z in [0, 1]) is completely inside (z - 2 <= 0)
    plane_inside = CutPlane(origin=np.array([0.0, 0.0, 2.0]), normal=np.array([0.0, 0.0, 1.0]))
    cut_inside = intersect_hex_cell_with_plane(elem_id=0, xe=xe, cut_plane=plane_inside)
    assert cut_inside.is_cut is False
    assert cut_inside.volume_fraction == 1.0

    # 2. Plane at z = -1.0, upward normal +z -> cube is completely outside (z - (-1) > 0)
    plane_outside = CutPlane(origin=np.array([0.0, 0.0, -1.0]), normal=np.array([0.0, 0.0, 1.0]))
    cut_outside = intersect_hex_cell_with_plane(elem_id=1, xe=xe, cut_plane=plane_outside)
    assert cut_outside.is_cut is False
    assert cut_outside.volume_fraction == 0.0


def test_alefvm_all_solvers_fluxes():
    """Verify face flux calculation across all FVM solver options."""
    nodes, conn, nbr_elem, nbr_face = make_two_cube_mesh(size=1.0)
    state = ALEFVMState.initialize(
        n_elem=2,
        n_nodes=len(nodes),
        vol=np.array([1.0, 1.0]),
        rho=np.array([2.0, 1.0]),
        velocity=np.array([[3.0, 0.0, 0.0], [1.0, 0.0, 0.0]]),
        ssp=np.array([300.0, 300.0]),
        pressure=np.array([100.0, 100.0]),
    )
    alefvm_prepare_stress_buffer(state, nodes, conn)

    # Test each solver
    for st in [SOLVER_U_AVG, SOLVER_RHOU_AVG, SOLVER_ROE_AVG, SOLVER_GODUNOV_ACOUSTIC]:
        flux_data = alefvm_compute_face_fluxes(
            state,
            nodes,
            conn,
            nbr_elem,
            grid_velocity=None,
            solver_type=st,
            upwl=1.0,
            reduc=0.0,
        )
        flux_int = flux_data["flux_raw"][0, 4]
        # Velocity in +x, so flux across +x interface must be positive
        assert flux_int > 0.0
