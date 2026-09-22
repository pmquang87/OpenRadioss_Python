# tests/test_multifluid.py
"""
Unit tests for the OpenRadioss FVM Multi-Material Fluid Solver.

Covers:
1. Multi-material pressure equilibrium convergence & energy conservation
2. MUSCL gradient accuracy on smooth and linear functions
3. MUSCL limiter monotonicity preservation on step functions
4. Numerical flux consistency and symmetry (HLLC & Rusanov)
5. 1D Sod shock tube Riemann benchmark
6. Global mass and energy conservation in closed domains
7. Viscous diffusion kinetic-to-internal energy conversion
8. CFL time step scaling
9. FVM-to-FEM nodal force mapping
"""

import math
import numpy as np
import pytest

from pyradioss.engine.multifluid import (
    FVMMesh,
    MaterialEOS,
    CellState,
    pressure_equilibrium,
    muscl_gradients,
    muscl_fluxes,
    time_evolution,
    viscous_diffusion,
    compute_fvm_dt,
    fvm2fem_forces,
    MultifluidSolver,
)


def test_pressure_equilibrium_single_fluid():
    """Verify single-fluid pressure evaluation matches ideal gas EOS."""
    eos = MaterialEOS(kind="IDEAL-GAS", gamma=1.4)
    state = CellState(n_cells=5, n_mat=1, eos_list=[eos])
    state.rho = np.array([1.0, 1.2, 0.8, 2.0, 0.5])
    state.eint = np.array([2.5, 3.0, 2.0, 5.0, 1.25])  # P = (gamma - 1) * eint = 0.4 * eint

    pressure_equilibrium(state)

    expected_p = 0.4 * state.eint
    assert np.allclose(state.pres, expected_p, rtol=1e-10)

    expected_c = np.sqrt(1.4 * expected_p / state.rho)
    assert np.allclose(state.sound_speed, expected_c, rtol=1e-10)


def test_multi_material_pressure_equilibrium_convergence():
    """Verify Newton-Raphson multi-phase pressure relaxation and energy reset.

    Tests two fluids (e.g. air gamma=1.4 and gas gamma=1.667) in a cell:
    - Equal relaxed pressure across phases
    - Volume conservation: sum(alpha_m) = 1.0
    - Mass conservation: sum(alpha_m * rho_m) = rho_total
    - Energy conservation: sum(alpha_m * eint_m) = eint_total
    """
    eos1 = MaterialEOS(kind="IDEAL-GAS", gamma=1.4, rho0=1.0)
    eos2 = MaterialEOS(kind="IDEAL-GAS", gamma=1.667, rho0=2.0)

    n_cells = 3
    state = CellState(n_cells=n_cells, n_mat=2, eos_list=[eos1, eos2])
    state.vol = np.ones(n_cells)

    # Initial volume fractions: 60% fluid 1, 40% fluid 2
    state.phase_alpha[0, :] = 0.6
    state.phase_alpha[1, :] = 0.4
    state.phase_rho[0, :] = 1.0
    state.phase_rho[1, :] = 2.0

    # Initial disparate pressures: phase 1 has P=1000, phase 2 has P=500
    state.phase_eint[0, :] = 1000.0 / (1.4 - 1.0)   # 2500.0
    state.phase_eint[1, :] = 500.0 / (1.667 - 1.0)  # ~749.6

    # Mixture totals:
    state.rho = state.phase_alpha[0] * state.phase_rho[0] + state.phase_alpha[1] * state.phase_rho[1]
    state.eint = state.phase_alpha[0] * state.phase_eint[0] + state.phase_alpha[1] * state.phase_eint[1]
    eint_initial_total = state.eint.copy()

    pressure_equilibrium(state, max_iter=50, tol=1e-6)

    # 1. Phase pressures must be equal:
    p1 = state.phase_pres[0, :]
    p2 = state.phase_pres[1, :]
    assert np.allclose(p1, p2, rtol=1e-4), f"Phase pressures did not equilibrate: {p1} vs {p2}"
    assert np.allclose(state.pres, p1, rtol=1e-4)

    # 2. Volume fractions must sum to 1.0:
    sum_alpha = np.sum(state.phase_alpha, axis=0)
    assert np.allclose(sum_alpha, 1.0, atol=1e-12)

    # 3. Mass conservation:
    rho_check = state.phase_alpha[0] * state.phase_rho[0] + state.phase_alpha[1] * state.phase_rho[1]
    assert np.allclose(rho_check, state.rho, rtol=1e-10)

    # 4. Energy conservation:
    eint_final_total = state.phase_alpha[0] * state.phase_eint[0] + state.phase_alpha[1] * state.phase_eint[1]
    assert np.allclose(eint_final_total, eint_initial_total, rtol=1e-5)

    # 5. Sound speed must be strictly positive:
    assert np.all(state.sound_speed > 0.0)


def test_muscl_gradient_accuracy_smooth_function():
    """Verify MUSCL least-squares gradient achieves 2nd-order accuracy on smooth function."""
    mesh_coarse = FVMMesh.create_uniform_1d(n_cells=40, x_min=0.0, x_max=1.0)
    mesh_fine = FVMMesh.create_uniform_1d(n_cells=80, x_min=0.0, x_max=1.0)

    def eval_field(mesh: FVMMesh, beta: float = 1.0, use_limiter: bool = False):
        xc = mesh.cell_centers[:, 0]
        state = CellState(n_cells=mesh.n_cells)
        # Smooth function phi(x) = sin(2 * pi * x)
        state.rho = np.sin(2.0 * np.pi * xc) + 2.0
        state.eint = np.ones(mesh.n_cells)
        state.vol = mesh.cell_volumes
        pressure_equilibrium(state)

        grads = muscl_gradients(state, mesh, beta=beta, use_limiter=use_limiter)
        grad_num = grads["rho"][:, 0]

        # Analytical gradient phi'(x) = 2 * pi * cos(2 * pi * x)
        grad_exact = 2.0 * np.pi * np.cos(2.0 * np.pi * xc)

        # Exclude boundary cells where one-sided differences occur:
        interior = slice(2, -2)
        err = np.sqrt(np.mean((grad_num[interior] - grad_exact[interior]) ** 2))
        return err

    # 1. Pure least-squares reconstruction without limiter must achieve 2nd-order:
    err_coarse = eval_field(mesh_coarse, use_limiter=False)
    err_fine = eval_field(mesh_fine, use_limiter=False)
    order = math.log2(err_coarse / err_fine)
    assert order > 1.9, f"Unlimited least-squares gradient order {order:.2f} is less than 2nd-order"

    # 2. Monotonic smooth function with Sweby limiter active (beta=2.0) must also achieve 2nd-order:
    mesh_m_coarse = FVMMesh.create_uniform_1d(n_cells=40, x_min=0.05, x_max=0.20)
    mesh_m_fine = FVMMesh.create_uniform_1d(n_cells=80, x_min=0.05, x_max=0.20)
    err_m_coarse = eval_field(mesh_m_coarse, beta=2.0, use_limiter=True)
    err_m_fine = eval_field(mesh_m_fine, beta=2.0, use_limiter=True)
    order_m = math.log2(err_m_coarse / err_m_fine)
    assert order_m > 1.9, f"Limited gradient order {order_m:.2f} in monotonic region is less than 2nd-order"


def test_muscl_gradient_exact_linear():
    """Verify least-squares gradient on linear field is exact."""
    mesh = FVMMesh.create_uniform_1d(n_cells=20, x_min=0.0, x_max=2.0)
    xc = mesh.cell_centers[:, 0]
    state = CellState(n_cells=mesh.n_cells)
    slope = 7.5
    state.rho = slope * xc + 3.0
    state.vol = mesh.cell_volumes

    grads = muscl_gradients(state, mesh, beta=1.0)
    grad_x = grads["rho"][:, 0]

    # Interior cells should have exact slope:
    assert np.allclose(grad_x[2:-2], slope, rtol=1e-8)


def test_muscl_limiter_monotonicity():
    """Verify Sweby / Barth-Jespersen limiter prevents overshoot at discontinuities."""
    mesh = FVMMesh.create_uniform_1d(n_cells=30, x_min=0.0, x_max=1.0)
    xc = mesh.cell_centers[:, 0]
    state = CellState(n_cells=mesh.n_cells)
    # Heaviside step: 2.0 for x < 0.5, 1.0 for x >= 0.5
    state.rho = np.where(xc < 0.5, 2.0, 1.0)
    state.vol = mesh.cell_volumes

    grads = muscl_gradients(state, mesh, beta=1.0)

    # Reconstruct face values:
    for i in range(mesh.n_cells):
        g = grads["rho"][i]
        xk = mesh.cell_centers[i]
        for f_idx in mesh.cell_face_indices[i]:
            xf = mesh.face_centers[f_idx]
            reconstructed_val = state.rho[i] + np.dot(g, xf - xk)
            # Reconstructed value must not exceed max (2.0) or fall below min (1.0):
            assert 1.0 - 1e-12 <= reconstructed_val <= 2.0 + 1e-12


def test_numerical_flux_consistency_and_symmetry():
    """Verify numerical flux consistency: F(U, U) = F_exact(U)."""
    normal = np.array([1.0, 0.0, 0.0])
    area = 2.5
    gamma = 1.4

    rho = 1.2
    u = 30.0
    p = 101325.0
    c = math.sqrt(gamma * p / rho)
    eint = p / (gamma - 1.0)

    st = {
        "rho": rho,
        "vel": np.array([u, 0.0, 0.0]),
        "eint": eint,
        "pres": p,
        "sound_speed": c,
    }

    # Physical flux:
    # F = [rho*u, rho*u^2 + p, 0, 0, (E + p)*u] * area
    etot = eint + 0.5 * rho * (u ** 2)
    f_exact = np.array([
        rho * u,
        rho * (u ** 2) + p,
        0.0,
        0.0,
        (etot + p) * u,
    ]) * area

    # Test HLLC:
    f_hllc, _ = muscl_fluxes(st, st, normal=normal, area=area, method="hllc")
    assert np.allclose(f_hllc, f_exact, rtol=1e-10)

    # Test Rusanov:
    f_rusanov, _ = muscl_fluxes(st, st, normal=normal, area=area, method="rusanov")
    assert np.allclose(f_rusanov, f_exact, rtol=1e-10)


def test_sod_shock_tube():
    """Test 1D Sod shock tube Riemann problem benchmark."""
    n_cells = 100
    mesh = FVMMesh.create_uniform_1d(n_cells=n_cells, x_min=0.0, x_max=1.0)
    xc = mesh.cell_centers[:, 0]

    eos = MaterialEOS(kind="IDEAL-GAS", gamma=1.4)
    solver = MultifluidSolver(
        mesh=mesh,
        eos_list=[eos],
        beta=1.0,
        cfl=0.5,
        method="hllc",
        boundary_conditions={0: "wall", n_cells: "wall"},
    )

    # Initial conditions:
    # Left: rho = 1.0, P = 1.0, u = 0.0
    # Right: rho = 0.125, P = 0.1, u = 0.0
    left_mask = xc <= 0.5
    rho_init = np.where(left_mask, 1.0, 0.125)
    pres_init = np.where(left_mask, 1.0, 0.1)
    vel_init = np.zeros((n_cells, 3))
    eint_init = pres_init / (1.4 - 1.0)

    solver.initialize_state(rho=rho_init, vel=vel_init, eint=eint_init)

    # Evolve up to t = 0.15:
    solver.run(t_end=0.15)

    rho = solver.state.rho
    pres = solver.state.pres
    vel_x = solver.state.vel[:, 0]

    # Verify physical bounds and features:
    # 1. Positivity of density and pressure:
    assert np.all(rho > 0.0), "Density became non-positive"
    assert np.all(pres > 0.0), "Pressure became non-positive"
    assert np.all(np.isfinite(rho))
    assert np.all(np.isfinite(pres))
    assert np.all(np.isfinite(vel_x))

    # 2. Maximum density should not exceed left state (1.0):
    assert np.max(rho) <= 1.001
    # 3. Minimum density should not fall below right state (0.125):
    assert np.min(rho) >= 0.124

    # 4. Rightward induced velocity between expansion and shock:
    # Velocity in region x in [0.4, 0.7] must be strictly positive:
    mid_mask = (xc >= 0.4) & (xc <= 0.7)
    assert np.all(vel_x[mid_mask] > 0.1), "Shock/contact induced flow velocity missing"

    # 5. Pressure plateau: between contact discontinuity and shock, pressure is constant ~ 0.303
    p_plat_mask = (xc >= 0.6) & (xc <= 0.7)
    p_plat = pres[p_plat_mask]
    assert np.all((p_plat >= 0.25) & (p_plat <= 0.35))


def test_mass_and_energy_conservation():
    """Verify total mass and energy are conserved to machine precision in closed domain."""
    n_cells = 50
    mesh = FVMMesh.create_uniform_1d(n_cells=n_cells, x_min=0.0, x_max=1.0)
    xc = mesh.cell_centers[:, 0]

    eos = MaterialEOS(kind="IDEAL-GAS", gamma=1.4)
    solver = MultifluidSolver(
        mesh=mesh,
        eos_list=[eos],
        beta=1.0,
        cfl=0.4,
        method="hllc",
        boundary_conditions={0: "wall", n_cells: "wall"},
    )

    # Localized pressure pulse in the center:
    rho0 = np.ones(n_cells)
    vel0 = np.zeros((n_cells, 3))
    p0 = 1.0 + 0.5 * np.exp(-((xc - 0.5) / 0.1) ** 2)
    eint0 = p0 / (1.4 - 1.0)

    solver.initialize_state(rho=rho0, vel=vel0, eint=eint0)

    totals_0 = solver.get_conserved_totals()
    mass_0 = totals_0["mass"]
    etot_0 = totals_0["total_energy"]

    # Run for 50 time steps:
    for _ in range(50):
        solver.step()

    totals_final = solver.get_conserved_totals()
    mass_final = totals_final["mass"]
    etot_final = totals_final["total_energy"]

    rel_mass_err = abs(mass_final - mass_0) / mass_0
    rel_ener_err = abs(etot_final - etot_0) / etot_0

    assert rel_mass_err < 1e-12, f"Mass conservation error {rel_mass_err} exceeded 1e-12"
    assert rel_ener_err < 1e-12, f"Energy conservation error {rel_ener_err} exceeded 1e-12"


def test_viscous_diffusion_energy_accounting():
    """Verify Navier-Stokes viscous diffusion converts kinetic energy to internal energy."""
    mesh = FVMMesh.create_uniform_1d(n_cells=20, x_min=0.0, x_max=1.0)
    xc = mesh.cell_centers[:, 0]

    eos = MaterialEOS(kind="IDEAL-GAS", gamma=1.4)
    state = CellState(n_cells=mesh.n_cells, eos_list=[eos])
    state.rho = np.ones(mesh.n_cells)
    state.vol = mesh.cell_volumes

    # Sinusoidal shear velocity profile u(x) = sin(2*pi*x):
    state.vel = np.zeros((mesh.n_cells, 3))
    state.vel[:, 0] = np.sin(2.0 * np.pi * xc)
    state.eint = np.full(mesh.n_cells, 10.0)

    pressure_equilibrium(state)

    vol = mesh.cell_volumes
    ekin_0 = float(0.5 * np.sum(state.rho * np.sum(state.vel ** 2, axis=1) * vol))
    eint_0 = float(np.sum(state.eint * vol))
    etot_0 = ekin_0 + eint_0

    # Apply viscous diffusion with dynamic viscosity:
    dt = 1e-3
    viscous_diffusion(state, mesh, dt, dynamic_viscosity=0.05)

    ekin_1 = float(0.5 * np.sum(state.rho * np.sum(state.vel ** 2, axis=1) * vol))
    eint_1 = float(np.sum(state.eint * vol))
    etot_1 = ekin_1 + eint_1

    # Kinetic energy must decrease due to viscous damping:
    assert ekin_1 < ekin_0, "Viscous diffusion failed to damp kinetic energy"

    # Internal energy must increase by exact dissipation:
    assert eint_1 > eint_0, "Dissipated energy was not added to internal energy"

    # Total energy must be conserved to machine precision:
    rel_err = abs(etot_1 - etot_0) / etot_0
    assert rel_err < 1e-13, f"Viscous diffusion energy accounting error {rel_err} exceeded 1e-13"


def test_cfl_dt_computation():
    """Verify CFL time step formula correctly scales with grid spacing and velocity."""
    mesh1 = FVMMesh.create_uniform_1d(n_cells=50, x_min=0.0, x_max=1.0)
    mesh2 = FVMMesh.create_uniform_1d(n_cells=100, x_min=0.0, x_max=1.0)

    eos = MaterialEOS(kind="IDEAL-GAS", gamma=1.4)
    state1 = CellState(n_cells=50, eos_list=[eos])
    state1.rho = np.ones(50)
    state1.eint = np.ones(50) * 2.5
    state1.vol = mesh1.cell_volumes
    pressure_equilibrium(state1)

    state2 = CellState(n_cells=100, eos_list=[eos])
    state2.rho = np.ones(100)
    state2.eint = np.ones(100) * 2.5
    state2.vol = mesh2.cell_volumes
    pressure_equilibrium(state2)

    dt1 = compute_fvm_dt(state1, mesh1, cfl=0.8)
    dt2 = compute_fvm_dt(state2, mesh2, cfl=0.8)

    # Halving cell size dx should halve allowable time step dt:
    ratio = dt1 / dt2
    assert np.isclose(ratio, 2.0, rtol=1e-3)


def test_fvm2fem_nodal_force_mapping():
    """Verify FVM-to-FEM force mapping on a 1D element mesh."""
    # 2 cells with 3 nodes:
    # Cell 0: nodes [0, 1], Cell 1: nodes [1, 2]
    nodes = np.array([
        [0.0, 0.0, 0.0],
        [0.5, 0.0, 0.0],
        [1.0, 0.0, 0.0],
    ])
    connectivity = np.array([
        [0, 1],
        [1, 2],
    ])

    mesh = FVMMesh.create_1d(nodes[:, 0])
    state = CellState(n_cells=2)
    # Uniform pressure P = 100.0:
    state.pres = np.array([100.0, 100.0])

    f_nodes = fvm2fem_forces(state, mesh, nodes, connectivity)

    # Total internal fluid force across domain must sum to net boundary pressure force:
    # Left boundary has area 1, normal -x, force = -(-100) = +100
    # Right boundary has area 1, normal +x, force = -(+100) = -100
    # Net force = 0
    total_force = np.sum(f_nodes, axis=0)
    assert np.allclose(total_force, 0.0, atol=1e-12)
