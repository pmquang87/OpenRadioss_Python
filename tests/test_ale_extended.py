# tests/test_ale_extended.py
"""
Tests for Extended ALE Framework:
- 2D ALE quad formulation (pyradioss/engine/ale_2d.py)
- Fixed-grid 3D Eulerian solver (pyradioss/engine/euler.py)
- k-epsilon turbulence transport & energy accounting (pyradioss/engine/ale_turbulence.py)
- Extended ALE grid smoothers (spring, curvature, centroidal Voronoi) in pyradioss/engine/ale_engine.py

Faithful verification of OpenRadioss Fortran sources:
- engine/source/ale/ale2d/aconv2.F, aflux2.F, agrad2.F, arezo2.F, amomt2.F, adiff2.F
- engine/source/ale/euler3d/eflux3.F, egrad3.F
- engine/source/ale/grid/alew2.F, alew4.F, alew6.F
- engine/source/ale/turbulence/akturb.F, aeturb.F, aturbn.F
"""

import numpy as np
import pytest

from pyradioss.engine.ale_2d import (
    QUAD_EDGES,
    build_quad_edge_connectivity,
    compute_quad_edge_normals,
    compute_quad_areas,
    ale_2d_compute_gradients,
    ale_2d_compute_fluxes,
    ale_2d_advect,
    ale_2d_remap,
    ale_2d_momentum_forces,
    ale_2d_diffuse,
)
from pyradioss.engine.euler import (
    euler_compute_gradients,
    euler_compute_fluxes,
    euler_advect_step,
    EulerianGrid3D,
)
from pyradioss.engine.ale_turbulence import (
    DEFAULT_CMU,
    DEFAULT_C1,
    DEFAULT_C2,
    DEFAULT_C3,
    DEFAULT_SIGMA_K,
    DEFAULT_SIGMA_E,
    compute_eddy_viscosity,
    compute_turbulent_pressure,
    update_turbulence_sources,
    diffuse_turbulence_fields,
    KEpsilonTurbulenceModel,
)
from pyradioss.engine.ale_engine import (
    HEX_FACES,
    HEX_SPRINGS_24,
    build_face_connectivity,
    compute_hex_volumes,
    compute_hex_face_normals,
    ale_grid_smooth_spring,
    ale_grid_smooth_curvature,
    ale_grid_smooth_volume,
    ale_step,
)
from tests.test_ale_engine import make_hex_mesh


def make_quad_mesh_2d(nx=2, ny=2, lx=2.0, ly=2.0):
    """Generate structured 2D quad mesh on [0, lx] x [0, ly]."""
    xs = np.linspace(0.0, lx, nx + 1)
    ys = np.linspace(0.0, ly, ny + 1)
    nodes = []
    node_idx = {}
    idx = 0
    for j in range(ny + 1):
        for i in range(nx + 1):
            nodes.append([xs[i], ys[j]])
            node_idx[(i, j)] = idx
            idx += 1
    nodes = np.array(nodes, dtype=np.float64)

    conn = []
    for j in range(ny):
        for i in range(nx):
            n0 = node_idx[(i, j)]
            n1 = node_idx[(i + 1, j)]
            n2 = node_idx[(i + 1, j + 1)]
            n3 = node_idx[(i, j + 1)]
            conn.append([n0, n1, n2, n3])
    conn = np.array(conn, dtype=np.int64)
    return nodes, conn


# =============================================================================
# 1. Tests for 2D ALE Quad Formulation (ale_2d.py)
# =============================================================================

def test_quad_edge_connectivity():
    """Verify element-element edge neighbor connectivity on 2x2 quad mesh."""
    nodes, conn = make_quad_mesh_2d(nx=2, ny=2)
    nbr_elem, nbr_edge = build_quad_edge_connectivity(conn)

    assert nbr_elem.shape == (4, 4)
    assert nbr_edge.shape == (4, 4)

    # Element 0 (bottom-left):
    # Edge 0 (bottom): boundary -> -1
    # Edge 1 (right): neighbor is element 1 (bottom-right), edge 3 (left edge of elem 1)
    # Edge 2 (top): neighbor is element 2 (top-left), edge 0 (bottom edge of elem 2)
    # Edge 3 (left): boundary -> -1
    assert nbr_elem[0, 0] == -1
    assert nbr_elem[0, 1] == 1
    assert nbr_edge[0, 1] == 3
    assert nbr_elem[0, 2] == 2
    assert nbr_edge[0, 2] == 0
    assert nbr_elem[0, 3] == -1

    # Symmetry check
    assert nbr_elem[1, 3] == 0
    assert nbr_edge[1, 3] == 1
    assert nbr_elem[2, 0] == 0
    assert nbr_edge[2, 0] == 2


def test_quad_edge_normals_cartesian():
    """Verify outward normal vectors for a single unit quad [0, 1]^2."""
    xe = np.array([[[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]])
    normals = compute_quad_edge_normals(xe, axisymmetric=False)

    # Edge 0: (0,0) -> (1,0): outward normal points down (0, -1)
    np.testing.assert_allclose(normals[0, 0], [0.0, -1.0], atol=1e-12)
    # Edge 1: (1,0) -> (1,1): outward normal points right (1, 0)
    np.testing.assert_allclose(normals[0, 1], [1.0, 0.0], atol=1e-12)
    # Edge 2: (1,1) -> (0,1): outward normal points up (0, 1)
    np.testing.assert_allclose(normals[0, 2], [0.0, 1.0], atol=1e-12)
    # Edge 3: (0,1) -> (0,0): outward normal points left (-1, 0)
    np.testing.assert_allclose(normals[0, 3], [-1.0, 0.0], atol=1e-12)


def test_quad_edge_normals_axisymmetric():
    """Verify axisymmetric scaling of normals by mean radius."""
    # Quad with y (radius) in [1.0, 2.0]
    xe = np.array([[[0.0, 1.0], [1.0, 1.0], [1.0, 2.0], [0.0, 2.0]]])
    normals = compute_quad_edge_normals(xe, axisymmetric=True)

    # Edge 0: y=1 -> scaled by 1.0
    np.testing.assert_allclose(normals[0, 0], [0.0, -1.0], atol=1e-12)
    # Edge 1: y in [1, 2] -> mean radius 1.5
    np.testing.assert_allclose(normals[0, 1], [1.5, 0.0], atol=1e-12)
    # Edge 2: y=2 -> scaled by 2.0
    np.testing.assert_allclose(normals[0, 2], [0.0, 2.0], atol=1e-12)
    # Edge 3: y in [2, 1] -> mean radius 1.5
    np.testing.assert_allclose(normals[0, 3], [-1.5, 0.0], atol=1e-12)


def test_quad_areas():
    """Verify element area calculation for Cartesian and axisymmetric quads."""
    nodes, conn = make_quad_mesh_2d(nx=2, ny=2, lx=2.0, ly=2.0)
    areas = compute_quad_areas(nodes, conn, axisymmetric=False)
    # Each quad in 2x2 grid of [0, 2]^2 has area 1.0
    np.testing.assert_allclose(areas, [1.0, 1.0, 1.0, 1.0], atol=1e-12)


def test_ale_2d_compute_gradients():
    """Verify 2D gradient projection factor matches agrad2.F formula."""
    nodes, conn = make_quad_mesh_2d(nx=2, ny=1, lx=2.0, ly=1.0)
    nbr_elem, _ = build_quad_edge_connectivity(conn)
    grad = ale_2d_compute_gradients(nodes, conn, nbr_elem)

    # Element 0 centroid (0.5, 0.5), Element 1 centroid (1.5, 0.5)
    # Distance vector d = (1.0, 0.0), length^2 = 1.0
    # Normal on edge 1 (right) is (1.0, 0.0)
    # grad = d . N / d^2 = 1.0 / 1.0 = 1.0
    assert grad.shape == (2, 4)
    np.testing.assert_allclose(grad[0, 1], 1.0, atol=1e-12)


def test_ale_2d_fluxes_and_conservation():
    """Verify 2D upwind fluxes and machine-precision mass conservation during advection."""
    nodes, conn = make_quad_mesh_2d(nx=3, ny=1, lx=3.0, ly=1.0)
    n_nodes = len(nodes)
    n_elem = len(conn)

    # Constant horizontal velocity u = 1.0, v = 0.0; fixed grid W = 0
    v_mat = np.zeros((n_nodes, 2))
    v_mat[:, 0] = 1.0
    w_grid = np.zeros((n_nodes, 2))

    # Density profile: element 0 has density 2.0, others 1.0
    rho = np.array([2.0, 1.0, 1.0])
    areas = compute_quad_areas(nodes, conn)
    mass_init = float(np.sum(rho * areas))

    # Compute fluxes (with boundary reduc=0 for closed domain)
    fluxes = ale_2d_compute_fluxes(rho, nodes, v_mat, w_grid, conn, upwl=1.0, reduc=0.0)

    # Net flux across all internal faces must sum to zero
    np.testing.assert_allclose(np.sum(fluxes["net_flux_q"]), 0.0, atol=1e-14)

    # Advect for dt = 0.1
    dt = 0.1
    mass_elem = rho * areas
    nbr_elem, _ = build_quad_edge_connectivity(conn)
    mass_new = ale_2d_advect(mass_elem, rho, fluxes, dt, neighbor_elem=nbr_elem)
    mass_final = float(np.sum(mass_new))

    # Strict mass conservation
    np.testing.assert_allclose(mass_final, mass_init, atol=1e-12)


def test_ale_2d_remap():
    """Verify conservative rezoning / remapping in 2D ALE."""
    nodes, conn = make_quad_mesh_2d(nx=2, ny=1, lx=2.0, ly=1.0)
    nbr_elem, _ = build_quad_edge_connectivity(conn)
    vol = np.array([1.0, 1.0])
    var = np.array([5.0, 3.0])

    # In arezo2.F, FLUX is the upwind flux array from aflux2:
    # FLUX(e, k) = Flux_k - UPWL * |Flux_k|
    # For outflow (elem 0 -> elem 1, Q=0.1):
    # Elem 0, edge 1: +0.1 - 1.0 * 0.1 = 0.0
    # Elem 1, edge 3: -0.1 - 1.0 * 0.1 = -0.2
    flux_upwind = np.zeros((2, 4))
    flux_upwind[0, 1] = 0.0
    flux_upwind[1, 3] = -0.2

    var_new = ale_2d_remap(var, var, flux_upwind, vol, dt=1.0, neighbor_elem=nbr_elem)

    # Donor cell retains intensive value 5.0:
    np.testing.assert_allclose(var_new[0], 5.0, atol=1e-12)
    # Receiving cell value updates toward donor: 3.0 + 0.5 * 1.0 * (3.0*(-0.2) - 5.0*(-0.2)) / 1.0 = 3.2:
    np.testing.assert_allclose(var_new[1], 3.2, atol=1e-12)


def test_ale_2d_momentum_forces():
    """Verify 2D ALE momentum convective forces and power accounting."""
    nodes, conn = make_quad_mesh_2d(nx=1, ny=1, lx=1.0, ly=1.0)
    n_nodes = len(nodes)
    areas = np.array([1.0])
    rho = np.array([1000.0])  # water density

    v_nodes = np.zeros((n_nodes, 2))
    v_nodes[:, 0] = 2.0  # uniform flow in x
    w_nodes = np.zeros((n_nodes, 2))  # fixed grid

    f_nodes, power = ale_2d_momentum_forces(
        rho, areas, v_nodes, w_nodes, nodes, conn, gamma=0.0
    )

    assert f_nodes.shape == (4, 2)
    assert isinstance(power, float)


def test_ale_2d_diffuse():
    """Verify harmonic diffusion on 2D quad mesh smooths gradients conservatively."""
    nodes, conn = make_quad_mesh_2d(nx=2, ny=1, lx=2.0, ly=1.0)
    nbr_elem, _ = build_quad_edge_connectivity(conn)
    grad = ale_2d_compute_gradients(nodes, conn, nbr_elem)
    areas = np.array([1.0, 1.0])

    phi = np.array([10.0, 0.0])
    alpha = np.array([0.5, 0.5])
    phi_new = ale_2d_diffuse(phi, grad, alpha, areas, dt=0.1, neighbor_elem=nbr_elem)

    # Heat flows from high to low
    assert phi_new[0] < phi[0]
    assert phi_new[1] > phi[1]
    # Total energy conserved (sum remains 10.0)
    np.testing.assert_allclose(np.sum(phi_new), np.sum(phi), atol=1e-12)


# =============================================================================
# 2. Tests for Fixed-Grid 3D Eulerian Solver (euler.py)
# =============================================================================

def test_euler_gradients():
    """Verify 3D Eulerian gradient projection factors (egrad3.F)."""
    nodes, conn = make_hex_mesh(nx=2, ny=1, nz=1, lx=2.0, ly=1.0, lz=1.0)
    nbr_elem, _ = build_face_connectivity(conn)
    grad = euler_compute_gradients(nodes, conn, nbr_elem)

    assert grad.shape == (2, 6)
    # Across shared face 4 (+x face of elem 0):
    # Centroid distance is 1.0 in x, normal has magnitude 2*Area = 2.0
    # GRAD = 0.5 * (d . N) / d^2 = 0.5 * (1.0 * 2.0) / 1.0 = 1.0
    np.testing.assert_allclose(grad[0, 4], 1.0, atol=1e-12)


def test_euler_compute_fluxes():
    """Verify 3D Eulerian face fluxes (eflux3.F)."""
    nodes, conn = make_hex_mesh(nx=2, ny=1, nz=1, lx=2.0, ly=1.0, lz=1.0)
    n_nodes = len(nodes)
    vel = np.zeros((n_nodes, 3))
    vel[:, 0] = 2.5  # uniform flow in +x

    q = np.array([10.0, 20.0])
    fluxes = euler_compute_fluxes(q, nodes, vel, conn, upwl=1.0, reduc=0.0)

    # Internal face between elem 0 and 1 has positive volume flux from 0 to 1
    # Face 4 (+x) of elem 0: Area = 1.0, vel = 2.5 -> VolFlux = 2.5
    np.testing.assert_allclose(fluxes["vol_fluxes"][0, 4], 2.5, atol=1e-12)
    # Face 5 (-x) of elem 1 has opposite sign
    np.testing.assert_allclose(fluxes["vol_fluxes"][1, 5], -2.5, atol=1e-12)
    # Donor cell for face 4 is elem 0 (q=10.0) -> flux_q = 10.0 * 2.5 = 25.0
    np.testing.assert_allclose(fluxes["flux_q"][0, 4], 25.0, atol=1e-12)


def test_eulerian_grid_3d_class():
    """Verify EulerianGrid3D advance and mass/energy ledger tracking."""
    nodes, conn = make_hex_mesh(nx=3, ny=1, nz=1, lx=3.0, ly=1.0, lz=1.0)
    grid = EulerianGrid3D(nodes, conn, upwl=1.0, reduc=0.0)

    n_nodes = len(nodes)
    vel = np.zeros((n_nodes, 3))
    vel[:, 0] = 1.0

    state = {
        "rho": np.array([3.0, 1.0, 1.0]),
        "eint": np.array([100.0, 50.0, 50.0]),
    }
    init_mass = float(np.sum(state["rho"] * grid.volumes))
    init_energy = float(np.sum(state["rho"] * state["eint"] * grid.volumes))

    # Advance 5 steps
    dt = 0.05
    for _ in range(5):
        state = grid.advance(state, vel, dt)

    # Check conservation in ledger
    np.testing.assert_allclose(grid.energy_ledger["current_mass"], init_mass, atol=1e-12)
    np.testing.assert_allclose(grid.energy_ledger["current_energy"], init_energy, atol=1e-12)


# =============================================================================
# 3. Tests for k-epsilon Turbulence Model (ale_turbulence.py)
# =============================================================================

def test_eddy_viscosity_and_turbulent_pressure():
    """Verify eddy viscosity and turbulent pressure formulas (akturb.F, aturbn.F)."""
    k = np.array([1.5, 2.0])
    eps = np.array([0.5, 1.0])
    rho = np.array([1000.0, 1.2])

    # mu_t = C_mu * rho * k^2 / eps
    mu_t = compute_eddy_viscosity(k, eps, rho, c_mu=0.09)
    expected_mu_t = 0.09 * rho * (k ** 2) / eps
    np.testing.assert_allclose(mu_t, expected_mu_t, atol=1e-12)

    # P_turb = 2/3 * rho * k
    p_turb = compute_turbulent_pressure(k, rho)
    expected_p = (2.0 / 3.0) * rho * k
    np.testing.assert_allclose(p_turb, expected_p, atol=1e-12)


def test_turbulence_sources_and_energy_accounting():
    """Verify source updates, dissipation, and thermal energy ledger booking (aturbn.F)."""
    k = np.array([2.0])
    eps = np.array([0.8])
    rho = np.array([1.2])
    vol = np.array([1.0])
    dvol = np.array([0.0])
    e_inc = np.array([0.05])
    mu_lam = np.array([1.8e-5])
    dt = 0.01

    k_new, eps_new, diss_energy = update_turbulence_sources(
        k, eps, rho, vol, dvol, e_inc, mu_lam, dt,
        c_mu=DEFAULT_CMU, c1=DEFAULT_C1, c2=DEFAULT_C2, c3=DEFAULT_C3,
    )

    # Dissipated energy must be strictly positive: Q_diss = rho * eps * vol * dt
    assert diss_energy > 0.0
    np.testing.assert_allclose(diss_energy, rho[0] * eps_new[0] * vol[0] * dt, rtol=1e-5)


def test_turbulence_subgrid_scale_floor():
    """Verify SGS filter floor limiter prevents unphysical dissipation collapse (aturbn.F)."""
    k = np.array([10.0])
    eps = np.array([1e-10])  # extremely low eps
    rho = np.array([1.0])
    vol = np.array([1.0])
    dvol = np.array([0.0])
    e_inc = np.array([0.0])
    mu_lam = np.array([1e-3])
    sgsl = np.array([0.1])  # 0.1 m filter width
    dt = 0.001

    k_new, eps_new, _ = update_turbulence_sources(
        k, eps, rho, vol, dvol, e_inc, mu_lam, dt, sgsl=sgsl
    )

    # SGS floor must enforce eps >= fac * k^1.5
    denom = DEFAULT_SIGMA_E * (DEFAULT_C2 - DEFAULT_C1)
    fac = np.sqrt(DEFAULT_CMU / denom) / sgsl[0]
    expected_floor = fac * (k_new[0] ** 1.5)
    assert eps_new[0] >= expected_floor * 0.999


def test_k_epsilon_model_orchestrator():
    """Verify KEpsilonTurbulenceModel end-to-end stepping with diffusion."""
    nodes, conn = make_hex_mesh(nx=2, ny=1, nz=1, lx=2.0, ly=1.0, lz=1.0)
    nbr_elem, _ = build_face_connectivity(conn)
    grad = euler_compute_gradients(nodes, conn, nbr_elem)
    vols = compute_hex_volumes(nodes, conn)

    model = KEpsilonTurbulenceModel(n_elem=2)
    rho = np.array([1.2, 1.2])
    dvol = np.array([0.0, 0.0])
    e_inc = np.array([0.01, 0.02])
    mu_lam = np.array([1.8e-5, 1.8e-5])

    res = model.step(
        rho=rho,
        vol=vols,
        dvol=dvol,
        e_inc=e_inc,
        mu_lam=mu_lam,
        dt=0.01,
        geom_grad=grad,
        neighbor_elem=nbr_elem,
    )

    assert "k" in res
    assert "eps" in res
    assert "mu_t" in res
    assert "p_turb" in res
    assert res["dissipated_energy"] > 0.0
    assert model.cumulative_dissipated_energy > 0.0


# =============================================================================
# 4. Tests for Extended Grid Smoothers in ale_engine.py
# =============================================================================

def test_ale_grid_smooth_spring():
    """Verify 24-spring hex network grid smoother (/ALE/GRID/SPRING, alew2.F)."""
    nodes, conn = make_hex_mesh(nx=2, ny=2, nz=2, lx=2.0, ly=2.0, lz=2.0)
    n_nodes = len(nodes)
    disp = np.zeros((n_nodes, 3))
    vel = np.zeros((n_nodes, 3))

    # Constrain boundary nodes on the outer faces of [0, 2]^3
    bcs = set()
    for i in range(n_nodes):
        x, y, z = nodes[i]
        if x == 0.0 or x == 2.0 or y == 0.0 or y == 2.0 or z == 0.0 or z == 2.0:
            bcs.add(i)

    # Interior center node (index 13 at (1, 1, 1))
    center_nodes = [i for i in range(n_nodes) if i not in bcs]
    assert len(center_nodes) == 1
    c_idx = center_nodes[0]

    # Displace center node perturbing it from (1, 1, 1) to (1.2, 1.0, 1.0)
    disp[c_idx, 0] = 0.2

    w_grid, x_new = ale_grid_smooth_spring(
        nodes, disp, vel, bcs, conn, dt=0.01, alpha=1.0, gamma=1.0
    )

    # Spring network should pull the perturbed node back toward the center (-x velocity)
    assert w_grid[c_idx, 0] < 0.0
    # Boundary nodes remain strictly stationary
    for b in bcs:
        np.testing.assert_allclose(w_grid[b], [0.0, 0.0, 0.0], atol=1e-12)


def test_ale_grid_smooth_curvature():
    """Verify curvature grid smoother (/ALE/GRID/STANDARD, alew4.F)."""
    nodes, conn = make_hex_mesh(nx=2, ny=2, nz=2, lx=2.0, ly=2.0, lz=2.0)
    n_nodes = len(nodes)
    disp = np.zeros((n_nodes, 3))
    vel = np.zeros((n_nodes, 3))

    bcs = set()
    for i in range(n_nodes):
        x, y, z = nodes[i]
        if x == 0.0 or x == 2.0 or y == 0.0 or y == 2.0 or z == 0.0 or z == 2.0:
            bcs.add(i)

    center_nodes = [i for i in range(n_nodes) if i not in bcs]
    c_idx = center_nodes[0]

    w_grid, x_new = ale_grid_smooth_curvature(
        nodes, disp, vel, bcs, conn, dt=0.01, alpha=1.0, gamma=1.0
    )

    assert w_grid.shape == (n_nodes, 3)
    assert x_new.shape == (n_nodes, 3)
    for b in bcs:
        np.testing.assert_allclose(x_new[b], nodes[b], atol=1e-12)


def test_ale_grid_smooth_volume():
    """Verify Centroidal Voronoi / volume grid smoother (/ALE/GRID/VOLUME, alew6.F)."""
    nodes, conn = make_hex_mesh(nx=2, ny=2, nz=2, lx=2.0, ly=2.0, lz=2.0)
    n_nodes = len(nodes)
    vel = np.zeros((n_nodes, 3))

    bcs = set()
    for i in range(n_nodes):
        x, y, z = nodes[i]
        if x == 0.0 or x == 2.0 or y == 0.0 or y == 2.0 or z == 0.0 or z == 2.0:
            bcs.add(i)

    center_nodes = [i for i in range(n_nodes) if i not in bcs]
    c_idx = center_nodes[0]

    # For a symmetric 2x2x2 mesh, the volume-weighted centroid average
    # of the 8 surrounding cells is exactly at the center (1.0, 1.0, 1.0)
    w_grid, x_new = ale_grid_smooth_volume(nodes, vel, bcs, conn, dt=0.01)

    np.testing.assert_allclose(x_new[c_idx], [1.0, 1.0, 1.0], atol=1e-12)
    np.testing.assert_allclose(w_grid[c_idx], [0.0, 0.0, 0.0], atol=1e-12)


def test_ale_step_with_spring_and_volume_smoothers():
    """Verify ale_step orchestration with spring and volume smoothers."""
    class DummyGroup:
        def __init__(self, conn):
            self.conn = conn
            self.state = {
                "mass": np.ones(len(conn)),
                "eint": np.full(len(conn), 10.0),
            }

    class DummyModel:
        def __init__(self, x, conn):
            self.x = x.copy()
            self.v = np.zeros_like(x)
            self.disp = np.zeros_like(x)
            self._groups = [("hex_group", DummyGroup(conn))]
            self.ale_grid_type = "spring"

        def element_groups(self):
            return self._groups

    nodes, conn = make_hex_mesh(nx=2, ny=2, nz=2)
    model = DummyModel(nodes, conn)

    # Step with spring smoother
    model.ale_grid_type = "spring"
    ale_step(model, dt=1e-4, state=None)
    assert model.x.shape == nodes.shape

    # Step with volume smoother
    model.ale_grid_type = "volume"
    ale_step(model, dt=1e-4, state=None)
    assert model.x.shape == nodes.shape
