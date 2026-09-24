"""
Tests for the Arbitrary Lagrangian-Eulerian (ALE) advection engine.

Covers:
- test_ale_gradient_linear: least-squares gradient reconstruction of linear field = exact
- test_ale_advection_mass_conservation: mesh remap conserves total mass to machine precision
- test_ale_convection_step: advection of a constant field produces no change
- test_ale_laplacian_smoothing: Laplacian smoothing moves interior node toward neighbor average
- test_ale_energy_conservation: total energy before and after remap conserved within 1e-8 relative
- Extended tests: limiters (Van Leer, Minmod, Barth-Jespersen), vector stress remap, ale_step
"""

import numpy as np
import pytest

from pyradioss.engine.ale_engine import (
    ale_compute_gradients,
    ale_compute_fluxes,
    ale_advect,
    ale_remap,
    ale_grid_smooth_laplacian,
    ale_step,
    compute_hex_volumes,
    build_face_connectivity,
    compute_hex_face_normals,
)


def make_hex_mesh(nx: int = 2, ny: int = 2, nz: int = 2,
                  lx: float = 2.0, ly: float = 2.0, lz: float = 2.0):
    """Generate a structured grid of nx x ny x nz 8-node hex elements."""
    x_1d = np.linspace(0.0, lx, nx + 1)
    y_1d = np.linspace(0.0, ly, ny + 1)
    z_1d = np.linspace(0.0, lz, nz + 1)
    
    n_nodes = (nx + 1) * (ny + 1) * (nz + 1)
    nodes = np.zeros((n_nodes, 3), dtype=np.float64)
    
    def node_id(i, j, k):
        return i + (nx + 1) * (j + (ny + 1) * k)
        
    for k in range(nz + 1):
        for j in range(ny + 1):
            for i in range(nx + 1):
                nid = node_id(i, j, k)
                nodes[nid] = [x_1d[i], y_1d[j], z_1d[k]]
                
    n_elem = nx * ny * nz
    conn = np.zeros((n_elem, 8), dtype=np.int64)
    elem_idx = 0
    for k in range(nz):
        for j in range(ny):
            for i in range(nx):
                # Standard OpenRadioss hex8 node ordering:
                # 0-3 bottom face CCW, 4-7 top face CCW
                n0 = node_id(i, j, k)
                n1 = node_id(i + 1, j, k)
                n2 = node_id(i + 1, j + 1, k)
                n3 = node_id(i, j + 1, k)
                n4 = node_id(i, j, k + 1)
                n5 = node_id(i + 1, j, k + 1)
                n6 = node_id(i + 1, j + 1, k + 1)
                n7 = node_id(i, j + 1, k + 1)
                conn[elem_idx] = [n0, n1, n2, n3, n4, n5, n6, n7]
                elem_idx += 1
                
    return nodes, conn


def test_ale_gradient_linear():
    """Gradient reconstruction of a linear scalar field must be exact (agrad3.F)."""
    nodes, conn = make_hex_mesh(nx=2, ny=2, nz=2, lx=2.0, ly=2.0, lz=2.0)
    
    # Exact linear field: q(x, y, z) = a*x + b*y + c*z + d
    a, b, c, d = 2.5, -1.8, 3.2, 4.0
    xe = nodes[conn]
    xc = np.mean(xe, axis=1)  # centroids
    q_elem = a * xc[:, 0] + b * xc[:, 1] + c * xc[:, 2] + d
    
    grad = ale_compute_gradients(q_elem, nodes, conn)
    
    assert grad.shape == (len(conn), 3)
    expected_grad = np.array([a, b, c])
    
    # Check that gradient in every element matches [a, b, c] to machine precision
    for e in range(len(conn)):
        np.testing.assert_allclose(
            grad[e], expected_grad, atol=1e-12, rtol=1e-12,
            err_msg=f"Element {e} gradient does not match exact linear gradient"
        )


def test_ale_advection_mass_conservation():
    """Remapping from old to new mesh must conserve total mass exactly (arezo3.F)."""
    nodes_old, conn = make_hex_mesh(nx=3, ny=3, nz=3, lx=3.0, ly=3.0, lz=3.0)
    v_old = compute_hex_volumes(nodes_old, conn)
    
    # Initial non-uniform density distribution
    n_elem = len(conn)
    xe = nodes_old[conn]
    xc = np.mean(xe, axis=1)
    rho_old = 1.0 + 0.5 * np.sin(xc[:, 0]) * np.cos(xc[:, 1])
    mass_old = np.sum(rho_old * v_old)
    
    # Perturb interior nodes while keeping boundary nodes strictly fixed
    neighbor_elem, _ = build_face_connectivity(conn)
    boundary_nodes = set()
    for e in range(n_elem):
        for f_idx in range(6):
            if neighbor_elem[e, f_idx] < 0:
                # Exterior face
                for n_loc in [0, 1, 2, 3, 4, 5, 6, 7]:
                    boundary_nodes.add(conn[e, n_loc])
                    
    # Only perturb interior nodes
    nodes_new = nodes_old.copy()
    for n_idx in range(len(nodes_old)):
        if n_idx not in boundary_nodes:
            # Shift interior node by 0.1 in x and y
            nodes_new[n_idx] += np.array([0.08, -0.06, 0.05])
            
    v_new = compute_hex_volumes(nodes_new, conn)
    
    # Remap density to new mesh
    rho_new = ale_remap(rho_old, None, nodes_old, nodes_new, conn, limiter="van_leer")
    mass_new = np.sum(rho_new * v_new)
    
    rel_mass_err = abs(mass_new - mass_old) / mass_old
    assert rel_mass_err < 1e-12, f"Mass conservation violated: relative error = {rel_mass_err:e}"


def test_ale_convection_step():
    """Advection of a constant scalar field must produce no change (aconv3.F)."""
    nodes, conn = make_hex_mesh(nx=2, ny=2, nz=2, lx=2.0, ly=2.0, lz=2.0)
    v_elem = compute_hex_volumes(nodes, conn)
    
    # Constant field q = 5.0
    c_val = 5.0
    q_elem = np.full(len(conn), c_val, dtype=np.float64)
    
    # Non-zero relative velocity field (e.g. flow in x-direction)
    vel = np.zeros_like(nodes)
    vel[:, 0] = 2.0  # 2.0 m/s in x
    vel[:, 1] = 0.5  # 0.5 m/s in y
    
    dt = 0.05
    grad_q = ale_compute_gradients(q_elem, nodes, conn)
    fluxes = ale_compute_fluxes(q_elem, grad_q, nodes, vel, conn)
    
    q_new = ale_advect(q_elem, fluxes, v_elem, dt)
    
    # Assert that field remains exactly 5.0 everywhere
    np.testing.assert_allclose(
        q_new, c_val, atol=1e-12, rtol=1e-12,
        err_msg="Advection of constant field produced non-constant output"
    )


def test_ale_laplacian_smoothing():
    """Laplacian smoothing must move interior node toward neighbor average (alew5.F)."""
    nodes, conn = make_hex_mesh(nx=2, ny=2, nz=2, lx=2.0, ly=2.0, lz=2.0)
    
    # Node id of center interior node at (1.0, 1.0, 1.0)
    center_nid = 1 + 3 * (1 + 3 * 1)  # 13 in a 3x3x3 node grid
    assert np.allclose(nodes[center_nid], [1.0, 1.0, 1.0])
    
    # All 26 other nodes are boundary nodes
    bcs_nodes = set(range(len(nodes))) - {center_nid}
    
    # Perturb the interior node away from its symmetric position
    nodes_perturbed = nodes.copy()
    nodes_perturbed[center_nid] = np.array([1.3, 0.7, 1.2])
    
    # Expected average of center node's 6 connected face neighbors:
    # Neighbors at (0, 1, 1), (2, 1, 1), (1, 0, 1), (1, 2, 1), (1, 1, 0), (1, 1, 2)
    # Average is [1.0, 1.0, 1.0]
    smoothed = ale_grid_smooth_laplacian(
        nodes_perturbed, bcs_nodes, connectivity=conn, iterations=1, alpha=1.0
    )
    
    # 1. Boundary nodes must not move at all
    for nid in bcs_nodes:
        np.testing.assert_allclose(
            smoothed[nid], nodes[nid], atol=1e-14,
            err_msg=f"Boundary node {nid} moved during smoothing"
        )
        
    # 2. Interior node must move exactly to neighbor average [1.0, 1.0, 1.0]
    np.testing.assert_allclose(
        smoothed[center_nid], [1.0, 1.0, 1.0], atol=1e-12,
        err_msg="Interior node did not move to neighbor average"
    )


def test_ale_energy_conservation():
    """Total energy before and after remap must be conserved within 1e-8 relative."""
    nodes_old, conn = make_hex_mesh(nx=2, ny=2, nz=2, lx=2.0, ly=2.0, lz=2.0)
    v_old = compute_hex_volumes(nodes_old, conn)
    n_elem = len(conn)
    
    # Element densities and specific internal energies
    xe = nodes_old[conn]
    xc = np.mean(xe, axis=1)
    rho = 1000.0 + 100.0 * xc[:, 0]
    e_spec = 250.0 + 50.0 * xc[:, 1]
    
    # Energy density per unit volume: e_vol = rho * e_spec (matching aconve.F90 nvar=2)
    e_vol_old = rho * e_spec
    e_total_initial = np.sum(e_vol_old * v_old)
    
    # Perturb interior node
    center_nid = 1 + 3 * (1 + 3 * 1)
    nodes_new = nodes_old.copy()
    nodes_new[center_nid] += np.array([0.15, -0.12, 0.10])
    
    v_new = compute_hex_volumes(nodes_new, conn)
    
    # Remap density (mass conservation) and energy density (energy conservation)
    rho_new = ale_remap(rho, None, nodes_old, nodes_new, conn, limiter="van_leer")
    e_vol_new = ale_remap(e_vol_old, None, nodes_old, nodes_new, conn, limiter="van_leer")
    
    # Remapped specific energy e_spec_new = e_vol_new / rho_new
    e_spec_new = e_vol_new / rho_new
    
    # Remapped total energy = sum(rho_new * e_spec_new * v_new) = sum(e_vol_new * v_new)
    e_total_final = np.sum(rho_new * e_spec_new * v_new)
    
    rel_energy_err = abs(e_total_final - e_total_initial) / e_total_initial
    assert rel_energy_err < 1e-8, (
        f"Energy conservation violated: relative error = {rel_energy_err:.3e} exceeds 1e-8"
    )


def test_ale_limiters_comparison():
    """Verify limiters preserve monotonicity and bounds."""
    nodes, conn = make_hex_mesh(nx=4, ny=1, nz=1, lx=4.0, ly=1.0, lz=1.0)
    # Step function in x: elements 0,1 have q=1.0, elements 2,3 have q=0.0
    q = np.array([1.0, 1.0, 0.0, 0.0])
    
    for lim in ["minmod", "van_leer", "barth_jespersen"]:
        grad = ale_compute_gradients(q, nodes, conn)
        vel = np.zeros_like(nodes)
        vel[:, 0] = 1.0  # flow in x
        fluxes = ale_compute_fluxes(q, grad, nodes, vel, conn, limiter=lim)
        q_new = ale_advect(q, fluxes, compute_hex_volumes(nodes, conn), dt=0.1)
        # Should stay bounded in [0, 1] without undershoot or overshoot
        assert np.all(q_new >= -1e-12)
        assert np.all(q_new <= 1.0 + 1e-12)


def test_ale_step_orchestrator():
    """Test full ale_step with a mock Model containing solid hex8 group."""
    nodes, conn = make_hex_mesh(nx=2, ny=2, nz=2, lx=2.0, ly=2.0, lz=2.0)
    v_init = compute_hex_volumes(nodes, conn)
    
    class MockGroup:
        def __init__(self, conn):
            self.conn = conn
            self.n = len(conn)
            self.state = {
                "mass": 1000.0 * v_init.copy(),
                "vol": v_init.copy(),
                "eint": 500.0 * v_init.copy(),
                "sig": np.zeros((len(conn), 6)),
            }
            
    class MockModel:
        def __init__(self, nodes, conn):
            self.x = nodes.copy()
            self.v = np.zeros_like(nodes)
            self.group = MockGroup(conn)
            self.ale_bcs = []
            
        def element_groups(self):
            yield ("solid_hexa8", self.group)
            
    class MockState:
        def __init__(self):
            self.t = 0.0
            self.cycle = 1
            self.e_num = 0.0
            
    model = MockModel(nodes, conn)
    state = MockState()
    
    # Perturb interior node so grid smoothing has something to do
    center_nid = 1 + 3 * (1 + 3 * 1)
    model.x[center_nid] += np.array([0.2, -0.15, 0.1])
    
    initial_mass = np.sum(model.group.state["mass"])
    initial_eint = np.sum(model.group.state["eint"])
    
    ale_step(model, dt=0.01, state=state)
    
    final_mass = np.sum(model.group.state["mass"])
    final_eint = np.sum(model.group.state["eint"])
    
    # Mass and eint conserved
    np.testing.assert_allclose(final_mass, initial_mass, rtol=1e-10)
    np.testing.assert_allclose(final_eint, initial_eint, rtol=1e-10)
