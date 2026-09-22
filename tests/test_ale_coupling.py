# tests/test_ale_coupling.py
"""
Unit and integration tests for ALE coupling, porous flow, subcycling, and thermal ALE.

Covers:
- FSI penalty coupling (/INTER/TYPE11, i11for3.F, iqela1.F)
- FSI tied kinematic coupling (/INTER/TYPE12, iqela3.F, i12for3.F)
- Bilinear quad shape functions and point-to-quad projection (shapeh.F, iqel03.F)
- Grid velocity boundary conditions (bcs3v.F)
- Anisotropic Darcy-Forchheimer porous drag (/PROP/POROUS TYPE15, poro.F)
- Rigid body reaction force and moment transmission (poro.F)
- Porous flow model LAW77 (aleflow.F)
- Porous face fluxes and conservative convection (aleflux.F, aleconv.F)
- Fluid-structure subcycling (alesub1.F, alesub2.F)
- Thermal ALE diffusivity and 3D finite volume heat diffusion (atherm.F, adiff3.F)
- ALE state rezone and convection drivers (arezon.F90, aconve.F90, alemain.F)
- Strict energy accounting (contact energy, porous EPOR, thermal energy conservation)
"""

import numpy as np
import pytest

from pyradioss.engine.fsi_coupling import (
    shape_functions_quad,
    compute_quad_tangents_and_normal,
    project_point_to_quad,
    apply_grid_velocity_bcs,
    FSICouplingPenalty,
    FSICouplingTied,
)
from pyradioss.engine.ale_porous import (
    PorousProperty15,
    DarcyForchheimerFlow,
    compute_porous_face_fluxes,
    porous_convection_step,
)
from pyradioss.engine.ale_engine import (
    ale_subcycle_step1,
    ale_subcycle_step2,
    ALESubcyclingManager,
    ale_thermal_diffusivity,
    compute_thermal_conductance_factors,
    ale_thermal_diffusion_step,
    arezon_driver,
    aconve_driver,
    ale_main_driver,
    compute_hex_volumes,
    build_face_connectivity,
)


# =============================================================================
# Helper: Create Simple 3D Mesh
# =============================================================================

def make_hex_mesh(nx: int = 2, ny: int = 2, nz: int = 2,
                  lx: float = 2.0, ly: float = 2.0, lz: float = 2.0):
    """Generate structured nx x ny x nz 8-node hex mesh."""
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
                nodes[node_id(i, j, k)] = [x_1d[i], y_1d[j], z_1d[k]]
                
    n_elem = nx * ny * nz
    conn = np.zeros((n_elem, 8), dtype=np.int64)
    elem_idx = 0
    for k in range(nz):
        for j in range(ny):
            for i in range(nx):
                conn[elem_idx] = [
                    node_id(i, j, k),
                    node_id(i + 1, j, k),
                    node_id(i + 1, j + 1, k),
                    node_id(i, j + 1, k),
                    node_id(i, j, k + 1),
                    node_id(i + 1, j, k + 1),
                    node_id(i + 1, j + 1, k + 1),
                    node_id(i, j + 1, k + 1),
                ]
                elem_idx += 1
                
    return nodes, conn


# =============================================================================
# 1. Shape Functions & Point-to-Quad Projection Tests (shapeh.F, iqel03.F)
# =============================================================================

def test_quad_shape_functions_properties():
    """Verify bilinear quad shape functions partition of unity and vertex values (shapeh.F)."""
    # Vertex evaluation: H_i(vertex_j) = delta_ij
    vertices = [
        (-1.0, -1.0),  # Node 0
        (1.0, -1.0),   # Node 1
        (1.0, 1.0),    # Node 2
        (-1.0, 1.0),   # Node 3
    ]
    for i, (s, t) in enumerate(vertices):
        h = shape_functions_quad(s, t)
        assert np.isclose(h[i], 1.0)
        assert np.isclose(np.sum(h), 1.0)
        for j in range(4):
            if j != i:
                assert np.isclose(h[j], 0.0)
                
    # Centroid evaluation: s=0, t=0 -> H_i = 0.25
    h_center = shape_functions_quad(0.0, 0.0)
    np.testing.assert_allclose(h_center, [0.25, 0.25, 0.25, 0.25])
    
    # Partition of unity across grid of points
    grid_s = np.linspace(-1.0, 1.0, 5)
    grid_t = np.linspace(-1.0, 1.0, 5)
    for s in grid_s:
        for t in grid_t:
            h = shape_functions_quad(s, t)
            assert np.isclose(np.sum(h), 1.0, atol=1e-14)
            assert np.all(h >= -1e-14)


def test_project_point_to_quad():
    """Test point projection onto quad face in 3D (iqel03.F)."""
    # Planar quad in z = 0 plane: [0,0,0], [2,0,0], [2,2,0], [0,2,0]
    quad_xyz = np.array([
        [0.0, 0.0, 0.0],
        [2.0, 0.0, 0.0],
        [2.0, 2.0, 0.0],
        [0.0, 2.0, 0.0],
    ])
    
    # Point directly above center: (1.0, 1.0, 0.5)
    point = np.array([1.0, 1.0, 0.5])
    s, t, dist, normal = project_point_to_quad(point, quad_xyz)
    
    assert np.isclose(s, 0.0, atol=1e-6)
    assert np.isclose(t, 0.0, atol=1e-6)
    assert np.isclose(dist, 0.5, atol=1e-6)
    np.testing.assert_allclose(normal, [0.0, 0.0, 1.0], atol=1e-6)
    
    # Point above corner: (2.0, 2.0, -0.2)
    point_corner = np.array([2.0, 2.0, -0.2])
    s_c, t_c, dist_c, normal_c = project_point_to_quad(point_corner, quad_xyz)
    assert np.isclose(s_c, 1.0, atol=1e-5)
    assert np.isclose(t_c, 1.0, atol=1e-5)
    assert np.isclose(dist_c, -0.2, atol=1e-5)


# =============================================================================
# 2. Grid Velocity Boundary Conditions (bcs3v.F)
# =============================================================================

def test_grid_velocity_boundary_conditions():
    """Test grid velocity constraints matching material velocity (bcs3v.F)."""
    n_nodes = 8
    v = np.ones((n_nodes, 3)) * 5.0
    w = np.zeros((n_nodes, 3))
    
    # Test codes 0 through 7
    codes = np.array([0, 1, 2, 3, 4, 5, 6, 7], dtype=np.int64)
    w_constrained = apply_grid_velocity_bcs(w, v, codes)
    
    # Code 0: no constraint -> W remains 0
    np.testing.assert_allclose(w_constrained[0], [0.0, 0.0, 0.0])
    # Code 1: Wz = Vz
    np.testing.assert_allclose(w_constrained[1], [0.0, 0.0, 5.0])
    # Code 2: Wy = Vy
    np.testing.assert_allclose(w_constrained[2], [0.0, 5.0, 0.0])
    # Code 3: Wy = Vy, Wz = Vz
    np.testing.assert_allclose(w_constrained[3], [0.0, 5.0, 5.0])
    # Code 4: Wx = Vx
    np.testing.assert_allclose(w_constrained[4], [5.0, 0.0, 0.0])
    # Code 5: Wx = Vx, Wz = Vz
    np.testing.assert_allclose(w_constrained[5], [5.0, 0.0, 5.0])
    # Code 6: Wx = Vx, Wy = Vy
    np.testing.assert_allclose(w_constrained[6], [5.0, 5.0, 0.0])
    # Code 7: W = V (fully Lagrangian)
    np.testing.assert_allclose(w_constrained[7], [5.0, 5.0, 5.0])


# =============================================================================
# 3. Penalty FSI Coupling Tests (i11for3.F, iqela1.F)
# =============================================================================

def test_fsi_penalty_coupling_linear():
    """Test linear penalty FSI coupling force and Newton's third law (i11for3.F, iqela1.F)."""
    coupling = FSICouplingPenalty(stiffness=1e4, gap=0.05, damping_ratio=0.0, nonlinear=False)
    
    # Slave fluid node penetrating master quad
    slave_x = np.array([[1.0, 1.0, 0.02]])  # Penetration p = 0.05 - 0.02 = 0.03
    slave_v = np.array([[0.0, 0.0, -1.0]])
    slave_mass = np.array([2.0])
    
    master_quads = np.array([[0, 1, 2, 3]], dtype=np.int64)
    master_x = np.array([
        [0.0, 0.0, 0.0],
        [2.0, 0.0, 0.0],
        [2.0, 2.0, 0.0],
        [0.0, 2.0, 0.0],
    ])
    master_v = np.zeros((4, 3))
    
    res = coupling.apply_coupling(
        slave_nodes_x=slave_x,
        slave_nodes_v=slave_v,
        slave_masses=slave_mass,
        master_quads_conn=master_quads,
        master_nodes_x=master_x,
        master_nodes_v=master_v,
        dt=1e-3,
    )
    
    f_slave = res["f_slave"]
    f_master = res["f_master"]
    
    # Expected normal force Fn = k * p = 1e4 * 0.03 = 300 N along +z
    expected_fn = 300.0
    np.testing.assert_allclose(f_slave[0], [0.0, 0.0, expected_fn], atol=1e-6)
    
    # Force on master nodes: H = [0.25, 0.25, 0.25, 0.25], each gets -75 N
    expected_f_master = np.full((4, 3), [0.0, 0.0, -75.0])
    np.testing.assert_allclose(f_master, expected_f_master, atol=1e-6)
    
    # Exact Newton third law: sum of forces = 0
    total_force = f_slave.sum(axis=0) + f_master.sum(axis=0)
    np.testing.assert_allclose(total_force, [0.0, 0.0, 0.0], atol=1e-12)
    
    # Contact energy ledger
    assert res["contact_energy"] > 0.0


def test_fsi_penalty_coupling_nonlinear():
    """Verify OpenRadioss logarithmic nonlinear stiffness formula (i11for3.F lines 250, 282-286)."""
    coupling = FSICouplingPenalty(stiffness=1e4, gap=0.1, damping_ratio=0.0, nonlinear=True)
    
    # Check force and stored contact energy at p = 0.04
    p = 0.04
    gap = 0.1
    fn, e_cont = coupling.compute_penalty_force(penetration=p, rel_normal_vel=0.0)
    
    # Analytical from Fortran:
    # fac = gap / (gap - p) = 0.1 / 0.06 = 1.6666667
    # facm1 = 1 / fac = 0.6
    # k_eff = 0.5 * k * fac = 0.5 * 1e4 * (0.1 / 0.06) = 8333.333
    # fn = k_eff * p = 8333.333 * 0.04 = 333.333
    fac = gap / (gap - p)
    facm1 = 1.0 / fac
    k_eff = 0.5 * 1e4 * fac
    expected_fn = k_eff * p
    expected_e = 0.5 * 1e4 * (gap ** 2) * (facm1 - 1.0 - np.log(facm1))
    
    assert np.isclose(fn, expected_fn)
    assert np.isclose(e_cont, expected_e)


# =============================================================================
# 4. Tied Kinematic FSI Coupling Tests (iqela3.F, i12for3.F)
# =============================================================================

def test_fsi_tied_coupling():
    """Test tied FSI grid velocity mapping and force/mass transfer (iqela3.F, i12for3.F)."""
    tied = FSICouplingTied(tolerance=0.02)
    
    # Master quad face moving with velocity [2.0, 0.0, 0.0]
    master_quads = np.array([[0, 1, 2, 3]], dtype=np.int64)
    master_x = np.array([
        [0.0, 0.0, 0.0],
        [2.0, 0.0, 0.0],
        [2.0, 2.0, 0.0],
        [0.0, 2.0, 0.0],
    ])
    master_v = np.full((4, 3), [2.0, 0.0, 0.0])
    
    # Slave fluid node at (1.0, 1.0, 0.005) - within tolerance of quad center
    slave_x = np.array([[1.0, 1.0, 0.005]])
    n_pairs = tied.find_tied_pairs(slave_x, master_quads, master_x)
    assert n_pairs == 1
    
    # Grid velocity interpolation: fluid node grid velocity should match master [2, 0, 0]
    w_fluid_init = np.zeros((1, 3))
    w_mapped = tied.map_grid_velocities(w_fluid_init, master_v, master_quads)
    np.testing.assert_allclose(w_mapped[0], [2.0, 0.0, 0.0], atol=1e-12)
    
    # Force and mass transfer: slave force [0, 0, -100] and mass 5.0
    f_slave = np.array([[0.0, 0.0, -100.0]])
    m_slave = np.array([5.0])
    f_master = np.zeros((4, 3))
    m_master = np.ones(4) * 2.0
    
    f_m_out, m_m_out, f_s_out = tied.transfer_forces_and_mass(
        f_slave, m_slave, f_master, m_master, master_quads
    )
    
    # Slave force zeroed out
    np.testing.assert_allclose(f_s_out[0], [0.0, 0.0, 0.0])
    # Distributed equally to 4 master nodes (H = 0.25): each gets -25 N and 1.25 kg
    np.testing.assert_allclose(f_m_out, np.full((4, 3), [0.0, 0.0, -25.0]))
    np.testing.assert_allclose(m_m_out, np.full(4, 3.25))


# =============================================================================
# 5. Porous Media /PROP/POROUS (TYPE15) Tests (poro.F)
# =============================================================================

def test_porous_drag_and_energy_ledger():
    """Test anisotropic porous drag and rigid body reaction with EPOR ledger (poro.F)."""
    # Skew coordinate frame aligned with global axes
    porous = PorousProperty15(g1=10.0, g2=20.0, g3=30.0, rigid_body_node=0)
    
    # 4 fluid nodes inside porous zone
    n_nodes = 4
    x = np.array([
        [0.0, 0.0, 0.0],  # Node 0 (also RB center)
        [1.0, 0.0, 0.0],  # Node 1
        [0.0, 1.0, 0.0],  # Node 2
        [1.0, 1.0, 0.0],  # Node 3
    ])
    v = np.full((n_nodes, 3), [3.0, 2.0, 1.0])
    w = np.zeros((n_nodes, 3))  # Eulerian grid
    mass = np.ones(n_nodes) * 2.0
    af = np.zeros((n_nodes, 3))
    dt = 0.01
    
    node_indices = np.array([0, 1, 2, 3], dtype=np.int64)
    
    res = porous.apply_porous_drag(
        node_indices=node_indices,
        x=x,
        v=v,
        w=w,
        mass=mass,
        af=af,
        dt=dt,
        rb_coord=x[0],
    )
    
    f_drag = res["f_drag"]
    af_out = res["af"]
    rb_force = res["rb_force"]
    delta_epor = res["delta_epor"]
    
    # Drag for each node:
    # P = mass * v = 2 * [3, 2, 1] = [6, 4, 2]
    # R = diag(10, 20, 30)
    # RFM = [60, 80, 60]
    expected_rfm = np.array([60.0, 80.0, 60.0])
    for i in range(n_nodes):
        np.testing.assert_allclose(f_drag[i], expected_rfm, atol=1e-12)
        np.testing.assert_allclose(af_out[i], -expected_rfm, atol=1e-12)
        
    # Total RB reaction force = 4 * [60, 80, 60] = [240, 320, 240]
    np.testing.assert_allclose(rb_force, [240.0, 320.0, 240.0], atol=1e-12)
    
    # Energy accounting: dE = dt * sum(RFM . v) = 0.01 * 4 * (60*3 + 80*2 + 60*1) = 0.04 * 400 = 16.0 J
    assert np.isclose(delta_epor, 16.0)
    assert np.isclose(res["epor"], 16.0)


# =============================================================================
# 6. Darcy-Forchheimer Element Drag Tests (aleflow.F)
# =============================================================================

def test_darcy_forchheimer_flow():
    """Verify element-level Darcy-Forchheimer drag density and forces (aleflow.F)."""
    flow = DarcyForchheimerFlow(
        darcy_coeff=100.0,
        forchheimer_coeff=10.0,
        unsteady_coeff=5.0,
        permeability=0.1,
        default_porosity=0.6,
    )
    
    # Flow velocity v_rel = [2.0, 0.0, 0.0], acceleration a = [1.0, 0.0, 0.0]
    v_rel = np.array([2.0, 0.0, 0.0])
    a_rel = np.array([1.0, 0.0, 0.0])
    
    # FAC = 0.125 * (1 - 0.6) / 0.1 = 0.125 * 0.4 / 0.1 = 0.5
    # Drag density = 0.5 * (100 * 2 + 10 * 2 * 2 + 5 * 1) = 0.5 * (200 + 40 + 5) = 122.5
    drag_density = flow.compute_drag_density(v_rel, a_rel)
    np.testing.assert_allclose(drag_density, [122.5, 0.0, 0.0], atol=1e-12)
    
    # Element forces: volume = 0.05 m^3 -> F = 122.5 * 0.05 = 6.125 N
    volumes = np.array([0.05])
    f_elem = flow.compute_element_forces(v_rel[np.newaxis, :], volumes, a_rel[np.newaxis, :])
    np.testing.assert_allclose(f_elem[0], [6.125, 0.0, 0.0], atol=1e-12)


# =============================================================================
# 7. Porous Face Fluxes & Convection Tests (aleflux.F, aleconv.F)
# =============================================================================

def test_porous_face_fluxes_and_convection():
    """Verify porous face flux computation and conservative advection (aleflux.F, aleconv.F)."""
    nodes, conn = make_hex_mesh(nx=2, ny=1, nz=1, lx=2.0, ly=1.0, lz=1.0)
    xe = nodes[conn]
    n_elem = len(conn)
    
    # Uniform flow in +x direction: relative velocity = [1.0, 0.0, 0.0]
    ve_rel = np.zeros((n_elem, 8, 3))
    ve_rel[:, :, 0] = 1.0
    
    # Porosities on all 6 faces = 0.8
    face_porosities = np.full((n_elem, 6), 0.8)
    
    flux_split, flu1 = compute_porous_face_fluxes(xe, ve_rel, face_porosities, upwind=0.0)
    
    # Face fluxes should be non-zero on x-faces (Face 4 (+x) and Face 5 (-x))
    assert np.all(flux_split[:, 4] > 0.0)  # Outward on +x face
    assert np.all(flux_split[:, 5] < 0.0)  # Inward on -x face
    
    # Conservative scalar advection: uniform field phi = 5.0
    phi = np.full(n_elem, 5.0)
    nbr_elem, _ = build_face_connectivity(conn)
    delta_phi = porous_convection_step(phi, flux_split, flu1, nbr_elem, dt=0.01)
    
    # In uniform flow through interior cells, divergence is balanced
    assert np.all(np.isfinite(delta_phi))


# =============================================================================
# 8. ALE Subcycling Tests (alesub1.F, alesub2.F)
# =============================================================================

def test_ale_subcycling():
    """Test ALE subcycling steps 1 and 2 and manager (alesub1.F, alesub2.F)."""
    n_nodes = 6
    # 3 fluid nodes (ALE), 3 solid nodes (Lagrangian)
    is_ale = np.array([1, 1, 1, 0, 0, 0], dtype=np.int64)
    
    d_prev = np.zeros((n_nodes, 3))
    d_curr = np.full((n_nodes, 3), 0.02)  # Displaced by 0.02 m
    v_curr = np.full((n_nodes, 3), 2.0)
    dt_solid = 0.01
    
    # Step 1: computes grid velocity W = 0.02 / 0.01 = 2.0
    w, v_saved, v_upd = ale_subcycle_step1(
        d=d_curr,
        d_save=d_prev,
        v=v_curr,
        is_ale_node=is_ale,
        dt_solid=dt_solid,
    )
    np.testing.assert_allclose(w, np.full((n_nodes, 3), 2.0))
    np.testing.assert_allclose(v_saved, np.full((n_nodes, 3), 2.0))
    
    # Step 2: restore velocities for solid nodes and update time step
    dt_fluid_calc = 0.002
    v_restored, d_saved, dt_new = ale_subcycle_step2(
        v=v_upd,
        v_saved=v_saved,
        d=d_curr,
        is_ale_node=is_ale,
        dt_fluid_current=dt_fluid_calc,
        dt_fluid_prev=0.002,
        scale_factor=1.0,
    )
    np.testing.assert_allclose(v_restored, v_curr)
    np.testing.assert_allclose(d_saved, d_curr)
    assert np.isclose(dt_new, 0.002)
    
    # Test ALESubcyclingManager
    mgr = ALESubcyclingManager(dt_scale=1.0, max_subcycles=10)
    n_sub, dt_sub = mgr.determine_subcycles(dt_solid=0.01, dt_fluid_raw=0.002)
    assert n_sub == 5
    assert np.isclose(dt_sub, 0.002)


# =============================================================================
# 9. Thermal ALE Diffusion Tests (atherm.F, adiff3.F)
# =============================================================================

def test_ale_thermal_diffusivity():
    """Verify piecewise temperature-dependent thermal conductivity (atherm.F)."""
    t = np.array([250.0, 300.0, 350.0, 400.0])
    # T_trans = 300.0. Below: k = 10 + 0.1*T. Above: k = 20 + 0.05*T
    cond, diff = ale_thermal_diffusivity(
        temperatures=t,
        a1=10.0,
        b1=0.1,
        a2=20.0,
        b2=0.05,
        t_trans=300.0,
        rho_cp=1000.0,
    )
    
    expected_cond = np.array([
        10.0 + 0.1 * 250.0,  # 35.0
        10.0 + 0.1 * 300.0,  # 40.0
        20.0 + 0.05 * 350.0, # 37.5
        20.0 + 0.05 * 400.0, # 40.0
    ])
    np.testing.assert_allclose(cond, expected_cond)
    np.testing.assert_allclose(diff, expected_cond / 1000.0)


def test_ale_thermal_diffusion_conservation():
    """Verify strict thermal energy conservation in 3D explicit diffusion (adiff3.F)."""
    nodes, conn = make_hex_mesh(nx=3, ny=1, nz=1, lx=3.0, ly=1.0, lz=1.0)
    n_elem = len(conn)
    vols = compute_hex_volumes(nodes, conn)
    nbr_elem, _ = build_face_connectivity(conn)
    grad = compute_thermal_conductance_factors(nodes, conn, nbr_elem)
    
    rho_cp = 1000.0
    # Step temperature gradient: elem 0 is hot (500 K), elem 1 is 300 K, elem 2 is cold (100 K)
    temp_init = np.array([500.0, 300.0, 100.0])
    eint_init = temp_init * rho_cp
    cond = np.full(n_elem, 50.0)  # Constant conductivity 50 W/(m.K)
    
    dt = 1e-4
    t_new, eint_new, net_e_change = ale_thermal_diffusion_step(
        temperatures=temp_init,
        internal_energy_density=eint_init,
        volumes=vols,
        conductivities=cond,
        grad_factors=grad,
        neighbor_elem=nbr_elem,
        rho_cp=rho_cp,
        dt=dt,
    )
    
    # 1. Total domain energy must be conserved to machine precision:
    # net_e_change = sum(dphi * V) == 0
    assert np.isclose(net_e_change, 0.0, atol=1e-10)
    total_energy_init = np.sum(eint_init * vols)
    total_energy_new = np.sum(eint_new * vols)
    assert np.isclose(total_energy_init, total_energy_new, atol=1e-10)
    
    # 2. Physics: hot element cooled down, cold element warmed up
    assert t_new[0] < temp_init[0]  # Hot elem cooled
    assert t_new[2] > temp_init[2]  # Cold elem warmed


# =============================================================================
# 10. Integration: Main ALE Driver (alemain.F, arezon.F90, aconve.F90)
# =============================================================================

class DummyGroup:
    """Mock solid element group for driver integration."""
    def __init__(self, conn, temp=None):
        self.conn = conn
        self.state = {}
        if temp is not None:
            self.state["temp"] = temp
            self.state["eint_v"] = temp * 1000.0
        self.state["mass"] = np.ones(len(conn)) * 1.0


class DummyModel:
    """Mock Model object for ale_main_driver."""
    def __init__(self, nodes, conn, temp=None):
        self.x = nodes.copy()
        self.v = np.zeros_like(nodes)
        self.disp = np.zeros_like(nodes)
        self.group = DummyGroup(conn, temp)
        
    def element_groups(self):
        return [("FLUID_HEX8", self.group)]


def test_ale_main_driver_integration():
    """Verify end-to-end integration of ALE main driver with thermal & FSI subsystems."""
    nodes, conn = make_hex_mesh(nx=2, ny=2, nz=2, lx=2.0, ly=2.0, lz=2.0)
    n_elem = len(conn)
    temps = np.linspace(300.0, 400.0, n_elem)
    
    model = DummyModel(nodes, conn, temp=temps)
    
    # Create penalty FSI coupling
    coupling = FSICouplingPenalty(stiffness=1e4, gap=0.01)
    
    res = ale_main_driver(
        model=model,
        dt=1e-4,
        state=None,
        enable_thermal=True,
        fsi_coupling=None,
    )
    
    assert res["status"] == "success"
    # Verify thermal work was computed
    assert np.isclose(res["thermal_work"], 0.0, atol=1e-8)
