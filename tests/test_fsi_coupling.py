"""
Unit and physics tests for Fluid-Structure Interface (FSI) coupling engine.

Port verification against OpenRadioss Fortran sources:
- ``engine/source/interfaces/interf/get_segment_normal.F90``: Outward normal and area calculation.
- ``engine/source/interfaces/int18/i18for3.F``: Reaction force mapping and linear momentum conservation.
- ``engine/source/ale/inter/iqela2.F``: Velocity compatibility projection.
- ``engine/source/interfaces/int18/i18for3.F`` & ``engine/source/engine/resol.F``: Energy balance.

Required test cases:
1. test_fsi_pressure_force_flat_plate: uniform pressure on flat surface: resultant = p * A
2. test_fsi_force_reciprocity: action = reaction (Newton 3rd law)
3. test_fsi_normal_computation: outward normal on unit cube face = [1,0,0] etc.
4. test_fsi_velocity_compatibility: fluid velocity projected onto surface = struct velocity
5. test_fsi_energy_balance: FSI work = pressure * dV (for incompressible step)
"""

from __future__ import annotations

import numpy as np
import pytest

from pyradioss.engine.fsi_coupling import (
    FSIInterface,
    fsi_compute_slave_normals,
    fsi_pressure_to_force,
    fsi_velocity_compatibility,
    fsi_step,
)


class DummyState:
    """Minimal EngineState for testing fsi_step."""
    def __init__(self):
        self.t = 0.0
        self.cycle = 0
        self.wext = 0.0
        self.e_fsi = 0.0


class DummyModel:
    """Minimal Model for testing fsi_step."""
    def __init__(self, x: np.ndarray, v: Optional[np.ndarray] = None):
        self.x = np.asarray(x, dtype=np.float64)
        self.v = np.zeros_like(self.x) if v is None else np.asarray(v, dtype=np.float64)
        self.inter_fsi = []
        self.surfaces = {}


# =============================================================================
# 1. test_fsi_pressure_force_flat_plate
# =============================================================================

def test_fsi_pressure_force_flat_plate():
    """Verify uniform pressure on flat plate yields exact resultant force F = p * A.

    Fortran reference:
      engine/source/interfaces/int18/i18for3.F lines 350-355, 658-670.
    """
    # Flat rectangular plate in XY plane (z = 0) of size 2.0 x 3.0 -> Area = 6.0
    # 2 x 3 grid of quads (6 quad elements, 12 nodes)
    # Nodes on regular grid: x in {0, 1, 2}, y in {0, 1, 2, 3}
    nx, ny = 3, 4
    x_coords = []
    for j in range(ny):
        for i in range(nx):
            x_coords.append([float(i), float(j), 0.0])
    x = np.array(x_coords, dtype=np.float64)

    # Quads with CCW ordering seen from +Z: [n1, n2, n3, n4]
    quad_segments = []
    for j in range(ny - 1):
        for i in range(nx - 1):
            n1 = j * nx + i
            n2 = j * nx + (i + 1)
            n3 = (j + 1) * nx + (i + 1)
            n4 = (j + 1) * nx + i
            quad_segments.append([n1, n2, n3, n4])
    quad_segments = np.array(quad_segments, dtype=np.int64)

    pressure = 150.0  # Pa
    expected_area = 2.0 * 3.0  # 6.0 m^2
    expected_resultant = pressure * expected_area  # 900.0 N in +Z

    f_nodal = fsi_pressure_to_force(pressure, x, quad_segments)

    # 1. Force resultant check
    f_resultant = np.sum(f_nodal, axis=0)
    assert np.isclose(f_resultant[0], 0.0, atol=1e-12)
    assert np.isclose(f_resultant[1], 0.0, atol=1e-12)
    assert np.isclose(f_resultant[2], expected_resultant, rtol=1e-12)

    # 2. Triangle mesh verification: split each quad into 2 triangles (12 triangles)
    tri_segments = []
    for quad in quad_segments:
        tri_segments.append([quad[0], quad[1], quad[2]])
        tri_segments.append([quad[0], quad[2], quad[3]])
    tri_segments = np.array(tri_segments, dtype=np.int64)

    f_nodal_tri = fsi_pressure_to_force(pressure, x, tri_segments)
    f_res_tri = np.sum(f_nodal_tri, axis=0)
    assert np.isclose(f_res_tri[0], 0.0, atol=1e-12)
    assert np.isclose(f_res_tri[1], 0.0, atol=1e-12)
    assert np.isclose(f_res_tri[2], expected_resultant, rtol=1e-12)


# =============================================================================
# 2. test_fsi_force_reciprocity
# =============================================================================

def test_fsi_force_reciprocity():
    """Verify Newton's 3rd law: fluid force on structure = -structure force on fluid.

    Fortran reference:
      engine/source/interfaces/int18/i18for3.F lines 740 & 772:
        FCONT(structural_node) += FX
        FCONT(fluid_node)      -= FX
      Sum of all interface forces is exactly zero (momentum conservation).
    """
    # Inclined flat surface in 3D
    x = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 1.0],
        [1.0, 2.0, 1.0],
        [0.0, 2.0, 0.0],
    ], dtype=np.float64)

    quad_segments = np.array([[0, 1, 2, 3]], dtype=np.int64)
    pressure = 200.0

    f_struct, f_fluid = fsi_pressure_to_force(
        pressure, x, quad_segments, return_fluid_force=True
    )

    # Exact action-reaction reciprocity
    np.testing.assert_allclose(f_struct + f_fluid, 0.0, atol=1e-14)

    # Non-zero total force on structure
    f_total_s = np.sum(f_struct, axis=0)
    assert np.linalg.norm(f_total_s) > 0.0

    # Total system force is identically zero
    f_total_system = np.sum(f_struct, axis=0) + np.sum(f_fluid, axis=0)
    np.testing.assert_allclose(f_total_system, [0.0, 0.0, 0.0], atol=1e-14)


# =============================================================================
# 3. test_fsi_normal_computation
# =============================================================================

def test_fsi_normal_computation():
    """Verify outward normal vectors on unit cube faces: [1,0,0], [-1,0,0], etc.

    Fortran reference:
      engine/source/interfaces/interf/get_segment_normal.F90 lines 98-137.
    """
    # Unit cube vertices [0, 1]^3
    x = np.array([
        [0.0, 0.0, 0.0],  # 0
        [1.0, 0.0, 0.0],  # 1
        [1.0, 1.0, 0.0],  # 2
        [0.0, 1.0, 0.0],  # 3
        [0.0, 0.0, 1.0],  # 4
        [1.0, 0.0, 1.0],  # 5
        [1.0, 1.0, 1.0],  # 6
        [0.0, 1.0, 1.0],  # 7
    ], dtype=np.float64)

    # 6 Faces with counter-clockwise winding when viewed from outside the cube:
    # +X face (x = 1): [1, 2, 6, 5]
    # -X face (x = 0): [0, 4, 7, 3]
    # +Y face (y = 1): [2, 3, 7, 6]
    # -Y face (y = 0): [0, 1, 5, 4]
    # +Z face (z = 1): [4, 5, 6, 7]
    # -Z face (z = 0): [0, 3, 2, 1]
    cube_faces = np.array([
        [1, 2, 6, 5],  # +X
        [0, 4, 7, 3],  # -X
        [2, 3, 7, 6],  # +Y
        [0, 1, 5, 4],  # -Y
        [4, 5, 6, 7],  # +Z
        [0, 3, 2, 1],  # -Z
    ], dtype=np.int64)

    normals, areas = fsi_compute_slave_normals(x, cube_faces, return_areas=True)

    expected_normals = np.array([
        [1.0, 0.0, 0.0],   # +X
        [-1.0, 0.0, 0.0],  # -X
        [0.0, 1.0, 0.0],   # +Y
        [0.0, -1.0, 0.0],  # -Y
        [0.0, 0.0, 1.0],   # +Z
        [0.0, 0.0, -1.0],  # -Z
    ], dtype=np.float64)

    np.testing.assert_allclose(normals, expected_normals, atol=1e-12)
    np.testing.assert_allclose(areas, np.ones(6), atol=1e-12)

    # Also test triangles: decompose +X face into two triangles
    tri_faces = np.array([
        [1, 2, 6],
        [1, 6, 5],
    ], dtype=np.int64)
    tri_normals, tri_areas = fsi_compute_slave_normals(x, tri_faces, return_areas=True)
    np.testing.assert_allclose(tri_normals, [[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]], atol=1e-12)
    np.testing.assert_allclose(tri_areas, [0.5, 0.5], atol=1e-12)


# =============================================================================
# 4. test_fsi_velocity_compatibility
# =============================================================================

def test_fsi_velocity_compatibility():
    """Verify fluid velocity projected onto moving structural surface matches structure velocity.

    Fortran reference:
      engine/source/ale/inter/iqela2.F lines 105-131, 226-232:
        v_rel_n = (v_struct - v_fluid) . n
        v_fluid <- v_fluid + v_rel_n * n
    """
    # Surface normal along +X
    normals = np.array([[1.0, 0.0, 0.0]])
    slave_nodes = np.array([0])

    # Structure moves in +X at 3.0 m/s with tangential motion (vy = 1.0, vz = -1.0)
    v_struct = np.array([[3.0, 1.0, -1.0]])

    # Fluid node has initial normal velocity 0.5, tangential vy = 5.0, vz = 2.0
    v_fluid = np.array([[0.5, 5.0, 2.0]], dtype=np.float64)

    # 1. Slip compatibility (default): normal component matches, tangential preserved
    fsi_velocity_compatibility(
        v_fluid=v_fluid,
        v_struct=v_struct,
        slave_nodes=slave_nodes,
        master_segments=None,
        normals=normals,
        no_slip=False,
    )

    # Normal component must now equal v_struct . n = 3.0
    assert np.isclose(v_fluid[0, 0], 3.0, atol=1e-12)
    # Tangential components must remain untouched: vy = 5.0, vz = 2.0
    assert np.isclose(v_fluid[0, 1], 5.0, atol=1e-12)
    assert np.isclose(v_fluid[0, 2], 2.0, atol=1e-12)

    # 2. Oblique 3D normal test: n = [0, 3/5, 4/5]
    n_oblique = np.array([[0.0, 0.6, 0.8]])
    vs_oblique = np.array([[0.0, 2.0, 1.0]])
    vf_oblique = np.array([[10.0, 0.0, 0.0]])

    fsi_velocity_compatibility(
        v_fluid=vf_oblique,
        v_struct=vs_oblique,
        slave_nodes=slave_nodes,
        master_segments=None,
        normals=n_oblique,
        no_slip=False,
    )

    # Check normal projection: vf . n == vs . n
    vn_fluid = float(np.dot(vf_oblique[0], n_oblique[0]))
    vn_struct = float(np.dot(vs_oblique[0], n_oblique[0]))
    assert np.isclose(vn_fluid, vn_struct, atol=1e-12)

    # Tangential component in X was 10.0 and perpendicular to normal, so must remain 10.0
    assert np.isclose(vf_oblique[0, 0], 10.0, atol=1e-12)

    # 3. No-slip test: complete velocity continuity
    vf_noslip = np.array([[0.0, 0.0, 0.0]], dtype=np.float64)
    fsi_velocity_compatibility(
        v_fluid=vf_noslip,
        v_struct=v_struct,
        slave_nodes=slave_nodes,
        master_segments=None,
        normals=normals,
        no_slip=True,
    )
    np.testing.assert_allclose(vf_noslip[0], v_struct[0], atol=1e-12)


# =============================================================================
# 5. test_fsi_energy_balance
# =============================================================================

def test_fsi_energy_balance():
    """Verify FSI work equals pressure * swept_volume for an incompressible displacement step.

    Physics & Fortran reference:
      W = F . dx = (p * A) * (v_n * dt) = p * (A * v_n * dt) = p * dV
      engine/source/interfaces/int18/i18for3.F line 351:
        ECONTT = ECONTT + DT1 * VN(I) * FNI(I)
    """
    # Piston of area 4.0 m^2 in XY plane (2 x 2 square)
    x = np.array([
        [0.0, 0.0, 0.0],
        [2.0, 0.0, 0.0],
        [2.0, 2.0, 0.0],
        [0.0, 2.0, 0.0],
    ], dtype=np.float64)
    segs = np.array([[0, 1, 2, 3]], dtype=np.int64)

    pressure = 100.0   # Pa
    vn = 2.5           # m/s in +Z direction
    dt = 0.04          # s

    # Swept volume dV = Area * vn * dt = 4.0 * 2.5 * 0.04 = 0.4 m^3
    area = 4.0
    dV = area * vn * dt
    w_expected = pressure * dV  # 40.0 J

    # Structural nodes move with velocity vn in +Z
    v = np.array([
        [0.0, 0.0, vn],
        [0.0, 0.0, vn],
        [0.0, 0.0, vn],
        [0.0, 0.0, vn],
    ], dtype=np.float64)

    # 1. Direct work computation from nodal forces
    f_nodal = fsi_pressure_to_force(pressure, x, segs)
    w_fsi = float(np.sum(f_nodal * v)) * dt
    assert np.isclose(w_fsi, w_expected, rtol=1e-12)

    # 2. Integration via fsi_step
    model = DummyModel(x=x, v=v)
    itf = FSIInterface(
        id=1,
        slave_segments=segs,
        pressure=pressure,
    )
    model.inter_fsi = [itf]

    state = DummyState()
    fext = np.zeros_like(x)

    dE_step = fsi_step(model, dt, state, fext)

    assert np.isclose(dE_step, w_expected, rtol=1e-12)
    assert np.isclose(state.e_fsi, w_expected, rtol=1e-12)

    # Nodal external forces match theoretical resultant
    assert np.isclose(np.sum(fext[:, 2]), pressure * area, rtol=1e-12)
