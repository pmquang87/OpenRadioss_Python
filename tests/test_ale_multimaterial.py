# tests/test_ale_multimaterial.py
"""
Unit tests for ALE Law 51 Multi-material and Bi-material VOF subsystems.

Tests ported routines from:
- engine/source/ale/ale51/afluxt.F
- engine/source/ale/ale51/ale51_upwind2.F & ale51_upwind3.F
- engine/source/ale/ale51/ale51_antidiff2.F & ale51_antidiff3.F
- engine/source/ale/alemuscl/gradient_reconstruction2.F & gradient_reconstruction.F90
- engine/source/ale/alemuscl/gradient_limitation2.F
- engine/source/materials/mat/mat051/sigeps51.F90
- engine/source/ale/bimat/balph2.F
- engine/source/ale/bimat/amulf2.F
- engine/source/ale/bimat/bcumu2.F
- engine/source/ale/bimat/bafil2.F
- engine/source/ale/bimat/bmultn.F
- engine/source/ale/bimat/blero2.F
"""

import math
import numpy as np
import pytest

from pyradioss.engine.ale_multimaterial import (
    youngs_gradient_2d,
    youngs_gradient_3d,
    youngs_interface_normal,
    youngs_plane_constant_2d,
    least_squares_gradient_2d,
    least_squares_gradient_3d,
    gradient_limiter_barth_jespersen_2d,
    ale51_upwind_flux,
    ale51_antidiffusion,
    law51_pressure_relaxation,
    law51_mixture_properties,
    Law51MultiMaterial,
)

from pyradioss.engine.ale_bimat import (
    element_vof_from_nodal_fill,
    elements_vof_from_nodal_fill,
    bimat_vof_evolution,
    bimat_two_material_constraint,
    bimat_face_fluxes,
    bimat_mixture_properties,
    bimat_nodal_force_accumulation,
    bimat_nodal_fill_convection,
    bimat_nodal_fill_update,
    BiMaterialVOF,
)


# =============================================================================
# 1. Youngs Interface Reconstruction Tests
# =============================================================================

def test_youngs_gradient_2d_linear():
    """Test Youngs 2D gradient stencil on linear volume fraction field."""
    # Field: alpha(x, y) = 0.5 + 0.2*x + 0.1*y
    nx, ny = 5, 5
    dx, dy = 0.5, 0.5
    x = np.arange(nx) * dx
    y = np.arange(ny) * dy
    xx, yy = np.meshgrid(x, y)
    alpha = 0.5 + 0.2 * xx + 0.1 * yy
    alpha = np.clip(alpha, 0.0, 1.0)

    gx, gy = youngs_gradient_2d(alpha, dx=dx, dy=dy)

    # In the interior cells (1..3, 1..3), gradient should match [0.2, 0.1]
    assert np.allclose(gx[1:4, 1:4], 0.2, atol=1e-3)
    assert np.allclose(gy[1:4, 1:4], 0.1, atol=1e-3)


def test_youngs_gradient_3d_linear():
    """Test Youngs 3D gradient stencil on linear volume fraction field."""
    nz, ny, nx = 5, 5, 5
    dx, dy, dz = 1.0, 1.0, 1.0
    x = np.arange(nx) * dx
    y = np.arange(ny) * dy
    z = np.arange(nz) * dz
    zz, yy, xx = np.meshgrid(z, y, x, indexing="ij")
    alpha = 0.5 + 0.1 * xx + 0.05 * yy + 0.08 * zz
    alpha = np.clip(alpha, 0.0, 1.0)

    gx, gy, gz = youngs_gradient_3d(alpha, dx=dx, dy=dy, dz=dz)

    # In interior cells, gradients should be close to true slopes
    assert np.allclose(gx[1:4, 1:4, 1:4], 0.1, atol=1e-2)
    assert np.allclose(gy[1:4, 1:4, 1:4], 0.05, atol=1e-2)
    assert np.allclose(gz[1:4, 1:4, 1:4], 0.08, atol=1e-2)


def test_youngs_interface_normal():
    """Test interface unit normal calculation from gradient."""
    # Grad = [3.0, 4.0] -> unit normal pointing in -grad direction: [-0.6, -0.8]
    grad = np.array([[3.0, 4.0], [0.0, 0.0]])
    normals = youngs_interface_normal(grad)

    assert np.isclose(normals[0, 0], -0.6)
    assert np.isclose(normals[0, 1], -0.8)
    assert np.allclose(normals[1], [0.0, 0.0])


def test_youngs_plane_constant_2d():
    """Test 2D Youngs plane constant c for various volume fractions."""
    # 1. Normal along y-axis: n = [0, 1]
    # For horizontal cut at height alpha: 0*x + 1*y <= c -> c = alpha
    c_half = youngs_plane_constant_2d(np.array([0.0, 1.0]), alpha=0.5)
    assert np.isclose(c_half, 0.5, atol=1e-5)

    c_quarter = youngs_plane_constant_2d(np.array([0.0, 1.0]), alpha=0.25)
    assert np.isclose(c_quarter, 0.25, atol=1e-5)

    # 2. Normal at 45 degrees: n = [1/sqrt(2), 1/sqrt(2)]
    # At alpha = 0.5, symmetry requires c to cut exactly half the square
    n_diag = np.array([1.0 / math.sqrt(2.0), 1.0 / math.sqrt(2.0)])
    c_diag_half = youngs_plane_constant_2d(n_diag, alpha=0.5)
    assert np.isclose(c_diag_half, 1.0 / math.sqrt(2.0), atol=1e-5)

    # At small alpha, corner triangle area = c^2 / (2 * n1 * n2) = c^2 / (2 * 0.5) = c^2
    # So c = sqrt(alpha)
    c_diag_small = youngs_plane_constant_2d(n_diag, alpha=0.18)
    expected_c = math.sqrt(0.18)
    assert np.isclose(c_diag_small, expected_c, atol=1e-5)


# =============================================================================
# 2. Unstructured Least-Squares Gradient and Limiter
# =============================================================================

def test_least_squares_gradient_2d():
    """Test least squares gradient reconstruction on 2D quad mesh."""
    # 4 quad elements in a 2x2 grid:
    # Centroids at (0.5, 0.5), (1.5, 0.5), (0.5, 1.5), (1.5, 1.5)
    elem_centers = np.array([
        [0.5, 0.5],
        [1.5, 0.5],
        [0.5, 1.5],
        [1.5, 1.5],
    ])
    # Linear field: alpha(x, y) = 0.2 * x + 0.3 * y
    alpha = 0.2 * elem_centers[:, 0] + 0.3 * elem_centers[:, 1]

    # Neighbor connectivity: [bottom, right, top, left]
    neighbor_elem = np.array([
        [-1, 1, 2, -1],
        [-1, -1, 3, 0],
        [0, 3, -1, -1],
        [1, -1, -1, 2],
    ])

    grad = least_squares_gradient_2d(alpha, elem_centers, neighbor_elem)

    # Element 0 has neighbors at (1.5, 0.5) and (0.5, 1.5)
    # The least-squares stencil should accurately capture dx = 0.2, dy = 0.3
    assert np.allclose(grad[0], [0.2, 0.3], atol=1e-4)


def test_least_squares_gradient_3d():
    """Test least squares gradient reconstruction on 3D hex mesh."""
    elem_centers = np.array([
        [0.5, 0.5, 0.5],
        [1.5, 0.5, 0.5],
        [0.5, 1.5, 0.5],
        [0.5, 0.5, 1.5],
    ])
    alpha = 0.1 * elem_centers[:, 0] + 0.2 * elem_centers[:, 1] + 0.3 * elem_centers[:, 2]

    # Hex neighbors: 6 faces
    neighbor_elem = np.full((4, 6), -1, dtype=np.int64)
    # Element 0 has neighbors 1 (right +x), 2 (top +y), 3 (front +z)
    neighbor_elem[0, 4] = 1
    neighbor_elem[0, 1] = 2
    neighbor_elem[0, 2] = 3

    grad = least_squares_gradient_3d(alpha, elem_centers, neighbor_elem)
    assert np.allclose(grad[0], [0.1, 0.2, 0.3], atol=1e-4)


def test_gradient_limiter_barth_jespersen_2d():
    """Test Barth-Jespersen slope limiter bounds extrapolated values."""
    # Element 0 at (0.5, 0.5) with vertices at (0,0), (1,0), (1,1), (0,1)
    elem_centers = np.array([[0.5, 0.5], [1.5, 0.5]])
    alpha = np.array([0.5, 0.6])
    neighbor_elem = np.array([[-1, 1, -1, -1], [-1, -1, -1, 0]])

    node_coords = np.array([
        [0.0, 0.0],
        [1.0, 0.0],
        [1.0, 1.0],
        [0.0, 1.0],
    ])
    connectivity = np.array([[0, 1, 2, 3]])

    # Overshooting unconstrained gradient: grad = [2.0, 0.0]
    # At vertex 1 (1, 0), dx = 0.5, extrapolated val = 0.5 + 2.0*0.5 = 1.5
    # Max neighbor value is 0.6. The limiter should scale grad down so val <= 0.6
    grad = np.array([[2.0, 0.0], [0.0, 0.0]])

    limited_grad = gradient_limiter_barth_jespersen_2d(
        grad, alpha, elem_centers, node_coords, connectivity, neighbor_elem, beta=1.0
    )

    # Expected reduction factor: (0.6 - 0.5) / (1.5 - 0.5) = 0.1 / 1.0 = 0.1
    # limited_grad = 0.1 * [2.0, 0.0] = [0.2, 0.0]
    assert np.isclose(limited_grad[0, 0], 0.2, atol=1e-4)
    assert np.isclose(limited_grad[0, 1], 0.0, atol=1e-4)


# =============================================================================
# 3. ALE Law 51 Upwind and Anti-diffusion Fluxes
# =============================================================================

def test_ale51_upwind_flux():
    """Test ale51_upwind_flux donor-acceptor splitting."""
    flux = np.array([[10.0, -5.0, 0.0, 20.0]])

    # 1. Zero upwind (centered): eta = 0.0
    flux_up, qmv, flu1 = ale51_upwind_flux(flux, upwind_param=0.0)
    assert np.allclose(flux_up, flux)
    assert np.allclose(qmv, flux)
    assert np.isclose(flu1[0], 25.0)

    # 2. Full upwind: eta = 1.0
    # For flux > 0: flux_up = flux - |flux| = 0; qmv = 2*flux
    # For flux < 0: flux_up = flux - |flux| = 2*flux; qmv = 0
    flux_up, qmv, flu1 = ale51_upwind_flux(flux, upwind_param=1.0)
    assert np.isclose(flux_up[0, 0], 0.0)
    assert np.isclose(flux_up[0, 1], -10.0)
    assert np.isclose(qmv[0, 0], 20.0)
    assert np.isclose(qmv[0, 1], 0.0)


def test_ale51_antidiffusion_conservation():
    """Test anti-diffusive species flux conservation and volume limiting."""
    # 2 elements, 2 faces each, 2 material phases
    flux_saved = np.array([
        [5.0, -5.0],
        [5.0, -5.0],
    ])
    alpha_mat = np.array([
        [0.7, 0.3],
        [0.2, 0.8],
    ])
    elem_volumes = np.array([100.0, 100.0])
    dt = 1.0

    neighbor_elem = np.array([
        [-1, 1],
        [0, -1],
    ])
    neighbor_face = np.array([
        [-1, 0],
        [1, -1],
    ])

    species_flux = ale51_antidiffusion(
        flux_saved=flux_saved,
        alpha_mat=alpha_mat,
        elem_volumes=elem_volumes,
        dt=dt,
        upwind_sm=0.0,
        neighbor_elem=neighbor_elem,
        neighbor_face=neighbor_face,
    )

    # 1. For outgoing face 0 of elem 0 (flux = 5.0):
    # Phase 0 flux should be 0.7 * 5.0 = 3.5
    # Phase 1 flux should be 0.3 * 5.0 = 1.5
    assert np.isclose(species_flux[0, 0, 0], 3.5)
    assert np.isclose(species_flux[0, 0, 1], 1.5)
    # Sum across phases should equal total face flux
    assert np.isclose(np.sum(species_flux[0, 0, :]), 5.0)

    # 2. Skew-symmetry across neighbor interface:
    # Elem 1, face 1 is neighbor to elem 0, face 0 with incoming flux
    assert np.isclose(species_flux[1, 1, 0], -species_flux[0, 0, 0])
    assert np.isclose(species_flux[1, 1, 1], -species_flux[0, 0, 1])


def test_ale51_antidiffusion_volume_limiting():
    """Test that anti-diffusion scales down outgoing flux if it exceeds available volume."""
    # Element has small volume of phase 0: V_0 = alpha_0 * V = 0.05 * 10 = 0.5
    elem_volumes = np.array([10.0])
    alpha_mat = np.array([[0.05, 0.95]])
    # Outgoing total flux is 20.0, which would draw 0.05 * 20 = 1.0 > 0.5
    flux_saved = np.array([[20.0]])
    dt = 1.0

    species_flux = ale51_antidiffusion(
        flux_saved=flux_saved,
        alpha_mat=alpha_mat,
        elem_volumes=elem_volumes,
        dt=dt,
        upwind_sm=0.0,
    )

    # Phase 0 flux must be limited to <= available volume / dt = 0.5
    assert species_flux[0, 0, 0] <= 0.5 + 1e-12


# =============================================================================
# 4. Law 51 Pressure Relaxation and Mixture Properties
# =============================================================================

def test_law51_pressure_relaxation():
    """Test multi-phase iterative pressure relaxation algorithm."""
    # 2 phases: water (K ~ 2.2 GPa) and air (K ~ 140 kPa)
    # Initially differing pressures: P_water = 5.0 MPa, P_air = 0.1 MPa
    volumes = np.array([0.5, 0.5])
    elem_vol = 1.0
    masses = np.array([500.0, 0.6])  # rho_w = 1000, rho_a = 1.2
    densities = np.array([1000.0, 1.2])
    energies = np.array([1e5, 1e4])
    pressures = np.array([5.0e6, 1.0e5])
    sound_speeds = np.array([1500.0, 340.0])

    v_rel, e_rel, p_rel, p_eq, converged = law51_pressure_relaxation(
        volumes=volumes,
        masses=masses,
        energies=energies,
        densities=densities,
        pressures=pressures,
        sound_speeds=sound_speeds,
        elem_volume=elem_vol,
        max_iter=50,
        tol=1e-4,
    )

    assert converged is True
    # Volume conservation: sum(v_rel) == elem_vol
    assert np.isclose(np.sum(v_rel), elem_vol, atol=1e-10)
    # Phase volumes remain strictly positive
    assert np.all(v_rel > 0.0)

    # Air is vastly more compressible (compliance C_air >> C_water)
    # So water barely changes volume, air absorbs the pressure difference,
    # and equilibrium pressure P* is very close to initial P_air + compression
    assert p_eq < 5.0e6
    assert p_eq > 1.0e5

    # Thermodynamically consistent work:
    # Air was compressed (Delta V_air < 0), so work on air was positive -> E_air increases
    # Water slightly expanded (Delta V_w > 0), so E_water slightly decreases
    assert e_rel[1] > energies[1]


def test_law51_mixture_properties():
    """Test homogenized mixture properties computation."""
    volumes = np.array([0.4, 0.6])
    masses = np.array([400.0, 600.0])
    pressures = np.array([1.0e6, 2.0e6])
    sound_speeds = np.array([1000.0, 2000.0])
    dev_stresses = np.array([
        [1.0, -1.0, 0.0, 0.5, 0.0, 0.0],
        [2.0, -2.0, 0.0, 1.0, 0.0, 0.0],
    ])
    viscosities = np.array([0.01, 0.02])

    props = law51_mixture_properties(
        volumes, masses, pressures, sound_speeds, dev_stresses, viscosities
    )

    assert np.isclose(props["volume"], 1.0)
    assert np.isclose(props["mass"], 1000.0)
    assert np.isclose(props["density"], 1000.0)

    # Volume-weighted pressure: 0.4*1e6 + 0.6*2e6 = 1.6e6
    assert np.isclose(props["pressure"], 1.6e6)

    # Mass fractions are [0.4, 0.6]
    # c_mix^2 = 0.4*(1000^2) + 0.6*(2000^2) = 0.4*1e6 + 0.6*4e6 = 2.8e6
    expected_c = math.sqrt(2.8e6)
    assert np.isclose(props["sound_speed"], expected_c)

    # Cauchy stress: sigma = -P*I + s
    cauchy = props["cauchy_stress"]
    expected_sxx = 0.4 * 1.0 + 0.6 * 2.0  # 1.6
    assert np.isclose(cauchy[0], -1.6e6 + 1.6)


def test_law51_multimaterial_class_lifecycle():
    """Test end-to-end Law51MultiMaterial class initialization, advection, and relaxation."""
    n_elem = 2
    n_phases = 2
    elem_volumes = np.array([1.0, 1.0])
    initial_vfrac = np.array([[0.8, 0.2], [0.3, 0.7]])
    rho0 = np.array([1000.0, 1.2])
    ssp0 = np.array([1500.0, 340.0])

    mat = Law51MultiMaterial(
        n_elem=n_elem,
        n_phases=n_phases,
        elem_volumes=elem_volumes,
        initial_volume_fractions=initial_vfrac,
        phase_densities=rho0,
        phase_sound_speeds=ssp0,
    )

    # Set differing pressures to trigger relaxation
    mat.phase_pressures[0] = [2.0e6, 1.0e5]
    mat.phase_pressures[1] = [5.0e5, 1.0e5]

    mat.equilibrate_pressures()

    # Verify work was booked in energy ledger
    assert mat.energy_work_booked != 0.0
    # Volume fractions sum to 1.0
    assert np.allclose(np.sum(mat.alpha, axis=1), 1.0)


# =============================================================================
# 5. Bi-material VOF Subsystem Tests
# =============================================================================

def test_element_vof_from_nodal_fill():
    """Test element volume fraction evaluation from nodal fill function."""
    # 1. All nodes positive -> full material 1: alpha = 1.0
    fill_full = np.array([0.5, 0.8, 1.0, 0.2])
    assert np.isclose(element_vof_from_nodal_fill(fill_full), 1.0)

    # 2. All nodes negative -> empty material 1: alpha = 0.0
    fill_empty = np.array([-0.5, -0.8, -1.0, -0.2])
    assert np.isclose(element_vof_from_nodal_fill(fill_empty), 0.0)

    # 3. Half positive, half negative -> interface bisecting: alpha = 0.5
    fill_half = np.array([1.0, 1.0, -1.0, -1.0])
    assert np.isclose(element_vof_from_nodal_fill(fill_half), 0.5)


def test_bimat_vof_evolution():
    """Test bi-material VOF volume fraction evolution."""
    alph_old = np.array([0.6])
    vol_old = np.array([0.6])  # 0.6 m^3
    vol_new = np.array([1.0])  # total element volume = 1.0
    # Outgoing face flux of 0.1 m^3/s over dt = 1.0s
    face_fluxes = np.array([[0.1, 0.0, 0.0, 0.0]])
    flu1 = np.array([0.1])
    dt = 1.0

    alph_new, dalph = bimat_vof_evolution(
        alph_old=alph_old,
        vol_old=vol_old,
        vol_new=vol_new,
        face_fluxes=face_fluxes,
        flu1=flu1,
        dt=dt,
    )

    # FL_sum = 0.5 * (0.1 + 0.1) = 0.1
    # expected alph_new = (0.6 - 1.0 * 0.1) / 1.0 = 0.5
    assert np.isclose(alph_new[0], 0.5)
    assert np.isclose(dalph[0], -0.1)


def test_bimat_vof_negative_density_protection():
    """Test that VOF resets to 0 if phase density becomes non-positive."""
    alph_old = np.array([0.4])
    vol_old = np.array([0.4])
    vol_new = np.array([1.0])
    face_fluxes = np.zeros((1, 4))
    flu1 = np.zeros(1)

    alph_new, _ = bimat_vof_evolution(
        alph_old=alph_old,
        vol_old=vol_old,
        vol_new=vol_new,
        face_fluxes=face_fluxes,
        flu1=flu1,
        dt=1.0,
        rho_new=np.array([0.0]),  # Non-positive density
    )
    assert np.isclose(alph_new[0], 0.0)


def test_bimat_two_material_constraint():
    """Test bi-material constraint alpha_1 + alpha_2 <= 1 scaling."""
    # Sum exceeds 1.0: 0.7 + 0.5 = 1.2
    a1 = np.array([0.7])
    a2 = np.array([0.5])
    a1_adj, a2_adj = bimat_two_material_constraint(a1, a2, c11=1.0, c12=1.0)

    # After adjustment, sum should be <= 1.0
    assert np.isclose(a1_adj[0] + a2_adj[0], 1.0, atol=1e-12)
    assert a1_adj[0] < 0.7
    assert a2_adj[0] < 0.5


def test_bimat_face_fluxes():
    """Test AMULF2 interface-weighted face volume flux distribution."""
    # 1 element with 4 nodes:
    # Nodes 0, 1 are in material (fill = +1.0)
    # Nodes 2, 3 are outside (fill = -1.0)
    fill = np.array([1.0, 1.0, -1.0, -1.0])
    dfill = np.zeros(4)
    connectivity = np.array([[0, 1, 2, 3]])
    # Outgoing flux through bottom face 0 (between nodes 0 and 1) = 2.0
    # Outgoing flux through top face 2 (between nodes 2 and 3) = 2.0
    flux_total = np.array([[2.0, 0.0, 2.0, 0.0]])
    vol = np.array([10.0])
    alph = np.array([0.5])
    neighbor_elem = np.full((1, 4), -1, dtype=np.int64)

    flux_phase, flu1 = bimat_face_fluxes(
        fill=fill,
        dfill=dfill,
        connectivity=connectivity,
        flux_total=flux_total,
        vol=vol,
        alph=alph,
        neighbor_elem=neighbor_elem,
        upwind_param=0.0,
    )

    # Face 0 (nodes 0-1) is entirely in material 1 -> full flux: 2.0
    assert np.isclose(flux_phase[0, 0], 2.0)
    # Face 2 (nodes 2-3) is entirely outside material 1 -> zero flux: 0.0
    assert np.isclose(flux_phase[0, 2], 0.0)


def test_bimat_mixture_properties():
    """Test BCUMU2 mixture properties computation."""
    a1 = np.array([0.3, 0.8])
    a2 = np.array([0.7, 0.2])

    sig1 = np.tile([10.0, 10.0, 10.0, 0.0, 0.0, 0.0], (2, 1))
    sig2 = np.tile([2.0, 2.0, 2.0, 0.0, 0.0, 0.0], (2, 1))
    eint1 = np.array([100.0, 200.0])
    eint2 = np.array([20.0, 40.0])
    rho1 = np.array([1000.0, 1000.0])
    rho2 = np.array([1.2, 1.2])

    mix = bimat_mixture_properties(a1, a2, sig1, sig2, eint1, eint2, rho1, rho2)

    # Elem 0: sig_mix = 0.3*10 + 0.7*2 = 4.4
    assert np.isclose(mix["sig_mix"][0, 0], 4.4)
    # Elem 0: eint_mix = 0.3*100 + 0.7*20 = 30 + 14 = 44.0
    assert np.isclose(mix["eint_mix"][0], 44.0)
    # Elem 0: rho_mix = 0.3*1000 + 0.7*1.2 = 300.84
    assert np.isclose(mix["rho_mix"][0], 300.84)


def test_bimat_nodal_fill_convection_and_update():
    """Test BAFIL2 and BMULTN nodal fill convection and clamping."""
    # 1 quad element [0, 1] x [0, 1]
    coords = np.array([
        [0.0, 0.0],
        [1.0, 0.0],
        [1.0, 1.0],
        [0.0, 1.0],
    ])
    connectivity = np.array([[0, 1, 2, 3]])
    fill = np.array([1.0, 1.0, -1.0, -1.0])
    dalph = np.array([0.0])

    # Material velocity v = [1.0, 0.0], mesh velocity w = [0.0, 0.0]
    v = np.tile([1.0, 0.0], (4, 1))
    w = np.zeros((4, 2))

    dfill, node_count = bimat_nodal_fill_convection(
        v=v, w=w, fill=fill, dalph=dalph, connectivity=connectivity, coords=coords, dt=0.01
    )

    fill_updated = bimat_nodal_fill_update(fill, dfill, node_count, dt=0.01)

    # Verify all nodal fill values remain strictly in [-1.0, 1.0]
    assert np.all(fill_updated >= -1.0)
    assert np.all(fill_updated <= 1.0)


def test_bimat_vof_class_lifecycle():
    """Test complete BiMaterialVOF engine lifecycle on a 2D mesh."""
    # 2 quad elements side-by-side
    coords = np.array([
        [0.0, 0.0],
        [1.0, 0.0],
        [2.0, 0.0],
        [0.0, 1.0],
        [1.0, 1.0],
        [2.0, 1.0],
    ])
    connectivity = np.array([
        [0, 1, 4, 3],
        [1, 2, 5, 4],
    ])
    neighbor_elem = np.array([
        [-1, 1, -1, -1],
        [-1, -1, -1, 0],
    ])
    # Left nodes in fluid 1, right nodes in fluid 2
    initial_fill = np.array([1.0, 0.0, -1.0, 1.0, 0.0, -1.0])

    bimat = BiMaterialVOF(
        connectivity=connectivity,
        coords=coords,
        initial_nodal_fill=initial_fill,
        rho1=1000.0,
        rho2=1.0,
        bulk1=2.2e9,
        bulk2=1.4e5,
    )

    # Verify initial volume fractions: elem 0 is mostly fluid 1, elem 1 is mostly fluid 2
    assert bimat.alpha1[0] > bimat.alpha1[1]
    assert np.isclose(bimat.alpha1[0] + bimat.alpha2[0], 1.0)
    assert np.isclose(bimat.alpha1[1] + bimat.alpha2[1], 1.0)

    # Step advection with rightward flow
    flux_total = np.array([
        [0.0, 0.05, 0.0, 0.0],
        [0.0, 0.0, 0.0, -0.05],
    ])
    v_mat = np.tile([0.1, 0.0], (6, 1))
    w_mesh = np.zeros((6, 2))

    bimat.step_advection(
        flux_total=flux_total,
        neighbor_elem=neighbor_elem,
        v_mat=v_mat,
        w_mesh=w_mesh,
        dt=0.1,
    )

    # Both volume fractions must still sum to <= 1.0 and stay in [0, 1]
    assert np.all(bimat.alpha1 >= 0.0)
    assert np.all(bimat.alpha2 >= 0.0)
    assert np.all(bimat.alpha1 + bimat.alpha2 <= 1.0 + 1e-12)
    assert "rho_mix" in bimat.mixture_state
    assert "sig_mix" in bimat.mixture_state
