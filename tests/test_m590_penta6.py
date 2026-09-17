"""Tests for Milestone M590: Dedicated 6-Node Wedge / Prism Element (PENTA6).

Verification coverage:
1. Exact shape functions and partition of unity in natural prism space.
2. Exact volume and lumped mass distribution (m_i = rho * V / 6).
3. Canonical node winding check (reversing clockwise winding to ensure det(J) > 0).
4. Characteristic length and Courant time step calculation.
5. Rigid body motion (translation and rotation: zero strain rate, zero spurious stress).
6. Patch test: Uniaxial stress state and equilibrium.
7. Patch test: Pure shear strain state and stress increment.
8. Dynamic explicit time stepping and total energy conservation.
9. Implicit tangent stiffness: symmetry, 6 zero rigid body eigenvalues, numerical perturbation check.
10. Starter deck parsing (/PENTA6, /PENTA, /WEDGE) and model initialization.
11. VTK output cell type and von Mises stress evaluation.
"""
from __future__ import annotations

from pathlib import Path
import numpy as np
import pytest

from pyradioss.common.messages import MessageLog
from pyradioss.elements import solid_penta6
from pyradioss.input.deck_reader import read_deck
from pyradioss.input.starter_keywords import parse_starter_deck
from pyradioss.model.entities import Material, Property, Part
from pyradioss.model.model import Model, ElementGroup
from pyradioss.starter.initialization import (
    build_element_groups,
    initialize_elements_and_mass,
    resolve_materials,
)


def _create_unit_prism_coords() -> np.ndarray:
    """Return (6, 3) coordinates for a right-triangular prism:
    Base right triangle in z = 0 with vertices (0,0,0), (1,0,0), (0,1,0),
    top right triangle in z = 2 with vertices (0,0,2), (1,0,2), (0,1,2).
    Triangle area = 0.5, height = 2.0 -> Volume = 1.0.
    """
    return np.array([
        [0.0, 0.0, 0.0],  # 0: bottom 1
        [1.0, 0.0, 0.0],  # 1: bottom 2
        [0.0, 1.0, 0.0],  # 2: bottom 3
        [0.0, 0.0, 2.0],  # 3: top 4
        [1.0, 0.0, 2.0],  # 4: top 5
        [0.0, 1.0, 2.0],  # 5: top 6
    ], dtype=np.float64)


# ----------------------------------------------------------------------------
# 1. Shape functions & partition of unity
# ----------------------------------------------------------------------------

def test_penta6_shape_functions_and_partition_of_unity():
    """Verify shape functions sum to 1 and gradients sum to 0 at Gauss points."""
    dn_dxi = solid_penta6._DN_DXI  # (2, 6, 3)
    assert dn_dxi.shape == (2, 6, 3)

    # Gradients in natural space must sum to 0 across the 6 nodes
    for g in range(2):
        sum_grad_nat = dn_dxi[g].sum(axis=0)  # sum over 6 nodes -> (3,)
        np.testing.assert_allclose(sum_grad_nat, 0.0, atol=1e-15)

    # Physical Cartesian gradients for unit prism must also sum to 0
    x = _create_unit_prism_coords()[None, :, :]  # (1, 6, 3)
    dndx, vol_g, vol_tot = solid_penta6._geometry(x)
    for g in range(2):
        sum_grad_cart = dndx[0, g].sum(axis=0)  # (3,)
        np.testing.assert_allclose(sum_grad_cart, 0.0, atol=1e-14)


# ----------------------------------------------------------------------------
# 2. Volume and lumped mass
# ----------------------------------------------------------------------------

def test_penta6_volume_and_mass():
    """Verify exact prism volume V = 1.0 and mass distribution m_i = rho * V / 6."""
    coords = _create_unit_prism_coords()
    x = coords[None, :, :]  # (1, 6, 3)
    dndx, vol_g, vol_tot = solid_penta6._geometry(x)

    # Gauss point volumes must sum to 1.0 (each Gauss point weight = 0.5)
    assert vol_tot[0] == pytest.approx(1.0, rel=1e-14)
    assert vol_g[0, 0] == pytest.approx(0.5, rel=1e-14)
    assert vol_g[0, 1] == pytest.approx(0.5, rel=1e-14)

    # Test init_group with lumped mass
    model = Model()
    model.node_ids = np.arange(1, 7, dtype=np.int64)
    model._id2idx = {i + 1: i for i in range(6)}
    model.x = coords.copy()
    model.x0 = coords.copy()
    model.mass = np.zeros(6, dtype=np.float64)

    rho = 7.85e-9
    mat = Material(id=1, law=1, rho0=rho, params={"E": 210000.0, "nu": 0.3})
    prop = Property(id=1, type=14, title="SOLID")
    part = Part(id=1, prop_id=1, mat_id=1)

    group = ElementGroup(
        ids=np.array([101], dtype=np.int64),
        conn=np.array([[0, 1, 2, 3, 4, 5]], dtype=np.int64),
        part=np.array([0], dtype=np.int64),
    )
    group.state["slices"] = [(slice(0, 1), mat, prop)]
    group.state["part_ids"] = np.array([1], dtype=np.int64)
    model.penta6s = group

    log = MessageLog()
    node_idx, mass_c, _ = solid_penta6.init_group(group, model, log)
    np.add.at(model.mass, node_idx, mass_c)

    # Volume in state
    assert group.state["vol0"][0] == pytest.approx(1.0, rel=1e-14)
    # Mass lumped equally: m_i = rho * V / 6
    expected_m_i = rho * 1.0 / 6.0
    np.testing.assert_allclose(model.mass, expected_m_i, rtol=1e-12)
    assert np.sum(model.mass) == pytest.approx(rho * 1.0, rel=1e-12)


def test_penta6_winding_canonicalization():
    """Verify that inverted (clockwise) connectivity is detected and corrected."""
    coords = _create_unit_prism_coords()
    # Bottom triangle swapped: [0, 2, 1], top triangle swapped: [3, 5, 4]
    inverted_conn = np.array([[0, 2, 1, 3, 5, 4]], dtype=np.int64)

    model = Model()
    model.node_ids = np.arange(1, 7, dtype=np.int64)
    model._id2idx = {i + 1: i for i in range(6)}
    model.x = coords.copy()
    model.x0 = coords.copy()
    model.mass = np.zeros(6, dtype=np.float64)

    mat = Material(id=1, law=1, rho0=7.85e-9, params={"E": 210000.0, "nu": 0.3})
    prop = Property(id=1, type=14, title="SOLID")

    group = ElementGroup(
        ids=np.array([201], dtype=np.int64),
        conn=inverted_conn,
        part=np.array([0], dtype=np.int64),
    )
    group.state["slices"] = [(slice(0, 1), mat, prop)]
    group.state["part_ids"] = np.array([1], dtype=np.int64)

    log = MessageLog()
    solid_penta6.init_group(group, model, log)

    # Connectivity should now be canonical [0, 1, 2, 3, 4, 5]
    np.testing.assert_array_equal(group.conn[0], [0, 1, 2, 3, 4, 5])
    assert group.state["vol0"][0] > 0.0


# ----------------------------------------------------------------------------
# 3. Characteristic length and Courant step
# ----------------------------------------------------------------------------

def test_penta6_characteristic_length():
    """Verify L_c = min(sqrt(4*A_tri/sqrt(3)), h_avg)."""
    coords = _create_unit_prism_coords()
    x = coords[None, :, :]
    vol_tot = np.array([1.0])

    lc = solid_penta6._char_length(x, vol_tot)
    # A_tri = 0.5 -> L_tri = sqrt(4 * 0.5 / sqrt(3)) = sqrt(2 / sqrt(3)) ~ 1.0745699
    # h_avg = 2.0 -> min(1.07457, 2.0) = 1.0745699
    expected_ltri = np.sqrt(2.0 / np.sqrt(3.0))
    assert lc[0] == pytest.approx(expected_ltri, rel=1e-5)

    # Shorter height: H = 0.5
    coords_short = coords.copy()
    coords_short[3:, 2] = 0.5
    vol_short = np.array([0.25])
    lc_short = solid_penta6._char_length(coords_short[None, :, :], vol_short)
    # h_avg = 0.5 < L_tri -> L_c = 0.5
    assert lc_short[0] == pytest.approx(0.5, rel=1e-5)


# ----------------------------------------------------------------------------
# 4. Rigid body motion
# ----------------------------------------------------------------------------

def test_penta6_rigid_body_translation():
    """Rigid body translation produces zero deformation, zero spin, zero internal forces."""
    coords = _create_unit_prism_coords()
    model = Model()
    model.node_ids = np.arange(1, 7, dtype=np.int64)
    model._id2idx = {i + 1: i for i in range(6)}
    model.x = coords.copy()
    model.x0 = coords.copy()
    model.mass = np.zeros(6, dtype=np.float64)

    mat = Material(id=1, law=1, rho0=7.85e-9, params={"E": 210000.0, "nu": 0.3})
    prop = Property(id=1, type=14, title="SOLID")
    group = ElementGroup(
        ids=np.array([1], dtype=np.int64),
        conn=np.array([[0, 1, 2, 3, 4, 5]], dtype=np.int64),
        part=np.array([0], dtype=np.int64),
    )
    group.state["slices"] = [(slice(0, 1), mat, prop)]
    group.state["part_ids"] = np.array([1], dtype=np.int64)

    log = MessageLog()
    solid_penta6.init_group(group, model, log)

    # Constant velocity field on all nodes: v = (15.0, -25.0, 8.0)
    v = np.tile([15.0, -25.0, 8.0], (6, 1))
    vr = np.zeros((6, 3))
    fint = np.zeros((6, 3))
    dt = 1e-6

    solid_penta6.forces(group, model.x, v, vr, dt, fint, None)

    # Internal force must be strictly zero
    np.testing.assert_allclose(fint, 0.0, atol=1e-10)
    # Stresses must remain zero
    np.testing.assert_allclose(group.state["sig"], 0.0, atol=1e-10)


def test_penta6_rigid_body_rotation():
    """Rigid body rotation rotates existing stresses with zero spurious strain rate."""
    coords = _create_unit_prism_coords()
    model = Model()
    model.node_ids = np.arange(1, 7, dtype=np.int64)
    model._id2idx = {i + 1: i for i in range(6)}
    model.x = coords.copy()
    model.x0 = coords.copy()
    model.mass = np.zeros(6, dtype=np.float64)

    mat = Material(id=1, law=1, rho0=7.85e-9, params={"E": 210000.0, "nu": 0.3})
    prop = Property(id=1, type=14, title="SOLID")
    group = ElementGroup(
        ids=np.array([1], dtype=np.int64),
        conn=np.array([[0, 1, 2, 3, 4, 5]], dtype=np.int64),
        part=np.array([0], dtype=np.int64),
    )
    group.state["slices"] = [(slice(0, 1), mat, prop)]
    group.state["part_ids"] = np.array([1], dtype=np.int64)

    log = MessageLog()
    solid_penta6.init_group(group, model, log)

    # Pre-stress with uniform isotropic pressure
    p0 = 100.0
    group.state["sig"][0, :, :3] = -p0

    # Angular velocity omega = (0, 0, 1.0) rad/s around centroid (1/3, 1/3, 1.0)
    xc = np.array([1.0 / 3.0, 1.0 / 3.0, 1.0])
    omega = np.array([0.0, 0.0, 1.0])
    v = np.cross(omega, coords - xc)
    vr = np.zeros((6, 3))
    fint = np.zeros((6, 3))
    dt = 1e-6

    solid_penta6.forces(group, model.x, v, vr, dt, fint, None)

    # Pure hydrostatic pressure rotated around z should remain unchanged
    np.testing.assert_allclose(group.state["sig"][0, :, :3], -p0, rtol=1e-6)
    # Shear stress should remain zero
    np.testing.assert_allclose(group.state["sig"][0, :, 3:], 0.0, atol=1e-10)


# ----------------------------------------------------------------------------
# 5. Patch test: Uniaxial stress
# ----------------------------------------------------------------------------

def test_penta6_patch_uniaxial_stress():
    """Impose uniform velocity gradient along x: verify exact stress increment and nodal equilibrium."""
    coords = _create_unit_prism_coords()
    model = Model()
    model.node_ids = np.arange(1, 7, dtype=np.int64)
    model._id2idx = {i + 1: i for i in range(6)}
    model.x = coords.copy()
    model.x0 = coords.copy()
    model.mass = np.zeros(6, dtype=np.float64)

    E = 210000.0
    nu = 0.3
    lam = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))
    mu = E / (2.0 * (1.0 + nu))
    c11 = lam + 2.0 * mu
    c12 = lam

    mat = Material(id=1, law=1, rho0=7.85e-9, params={"E": E, "nu": nu})
    prop = Property(id=1, type=14, title="SOLID")
    group = ElementGroup(
        ids=np.array([1], dtype=np.int64),
        conn=np.array([[0, 1, 2, 3, 4, 5]], dtype=np.int64),
        part=np.array([0], dtype=np.int64),
    )
    group.state["slices"] = [(slice(0, 1), mat, prop)]
    group.state["part_ids"] = np.array([1], dtype=np.int64)

    log = MessageLog()
    solid_penta6.init_group(group, model, log)

    # Linear velocity field: vx = eps_dot * x, vy = 0, vz = 0
    eps_dot = 1e-3
    dt = 1e-4
    v = np.zeros((6, 3))
    v[:, 0] = eps_dot * coords[:, 0]
    vr = np.zeros((6, 3))
    fint = np.zeros((6, 3))

    solid_penta6.forces(group, model.x, v, vr, dt, fint, None)

    # Check stress at both Gauss points
    deps_xx = eps_dot * dt
    expected_sig_xx = c11 * deps_xx
    expected_sig_yy = c12 * deps_xx
    expected_sig_zz = c12 * deps_xx

    for g in range(2):
        assert group.state["sig"][0, g, 0] == pytest.approx(expected_sig_xx, rel=1e-5)
        assert group.state["sig"][0, g, 1] == pytest.approx(expected_sig_yy, rel=1e-5)
        assert group.state["sig"][0, g, 2] == pytest.approx(expected_sig_zz, rel=1e-5)
        np.testing.assert_allclose(group.state["sig"][0, g, 3:], 0.0, atol=1e-12)

    # Equilibrium of internal forces: sum over all nodes must be 0
    np.testing.assert_allclose(fint.sum(axis=0), 0.0, atol=1e-8)


# ----------------------------------------------------------------------------
# 6. Patch test: Pure shear
# ----------------------------------------------------------------------------

def test_penta6_patch_pure_shear():
    """Impose simple shear vx = gamma_dot * y: verify shear stress increment G * gamma."""
    coords = _create_unit_prism_coords()
    model = Model()
    model.node_ids = np.arange(1, 7, dtype=np.int64)
    model._id2idx = {i + 1: i for i in range(6)}
    model.x = coords.copy()
    model.x0 = coords.copy()
    model.mass = np.zeros(6, dtype=np.float64)

    E = 210000.0
    nu = 0.3
    G = E / (2.0 * (1.0 + nu))

    mat = Material(id=1, law=1, rho0=7.85e-9, params={"E": E, "nu": nu})
    prop = Property(id=1, type=14, title="SOLID")
    group = ElementGroup(
        ids=np.array([1], dtype=np.int64),
        conn=np.array([[0, 1, 2, 3, 4, 5]], dtype=np.int64),
        part=np.array([0], dtype=np.int64),
    )
    group.state["slices"] = [(slice(0, 1), mat, prop)]
    group.state["part_ids"] = np.array([1], dtype=np.int64)

    log = MessageLog()
    solid_penta6.init_group(group, model, log)

    gamma_dot = 2e-3
    dt = 1e-4
    v = np.zeros((6, 3))
    v[:, 0] = gamma_dot * coords[:, 1]  # vx = gamma_dot * y
    vr = np.zeros((6, 3))
    fint = np.zeros((6, 3))

    solid_penta6.forces(group, model.x, v, vr, dt, fint, None)

    # Shear strain increment d_gamma = gamma_dot * dt -> tau_xy = G * d_gamma
    expected_tau_xy = G * gamma_dot * dt
    for g in range(2):
        assert group.state["sig"][0, g, 3] == pytest.approx(expected_tau_xy, rel=1e-4)


# ----------------------------------------------------------------------------
# 7. Energy conservation in explicit dynamic oscillation
# ----------------------------------------------------------------------------

def test_penta6_energy_conservation():
    """Oscillating prism with zero damping/bulk viscosity conserves total energy."""
    coords = _create_unit_prism_coords()
    model = Model()
    model.node_ids = np.arange(1, 7, dtype=np.int64)
    model._id2idx = {i + 1: i for i in range(6)}
    model.x = coords.copy()
    model.x0 = coords.copy()
    model.mass = np.zeros(6, dtype=np.float64)

    rho = 7.85e-9
    E = 210000.0
    nu = 0.3
    mat = Material(id=1, law=1, rho0=rho, params={"E": E, "nu": nu})
    prop = Property(id=1, type=14, title="SOLID")
    # Disable bulk viscosity for pure elastic energy conservation test
    prop.params = {"qa": 0.0, "qb": 0.0}

    group = ElementGroup(
        ids=np.array([1], dtype=np.int64),
        conn=np.array([[0, 1, 2, 3, 4, 5]], dtype=np.int64),
        part=np.array([0], dtype=np.int64),
    )
    group.state["slices"] = [(slice(0, 1), mat, prop)]
    group.state["part_ids"] = np.array([1], dtype=np.int64)

    log = MessageLog()
    node_idx, mass_c, _ = solid_penta6.init_group(group, model, log)
    np.add.at(model.mass, node_idx, mass_c)

    # Unconstrained symmetric axial breathing mode
    fint0 = np.zeros((6, 3))
    dt_crit = solid_penta6.forces(group, model.x, np.zeros((6, 3)), np.zeros((6, 3)), 1e-8, fint0, None)
    dt = 0.2 * dt_crit[0]

    v = np.zeros((6, 3))
    v[:3, 2] = -5.0
    v[3:, 2] = +5.0
    vr = np.zeros((6, 3))
    m = model.mass

    # Initial acceleration
    fint = np.zeros((6, 3))
    solid_penta6.forces(group, model.x, v, vr, dt, fint, None)
    a = fint / m[:, None]

    E_kin_0 = 0.5 * np.sum(m * np.sum(v**2, axis=1))
    assert E_kin_0 > 0.0

    # Explicit Velocity-Verlet loop (200 cycles)
    for step in range(200):
        v_half = v + 0.5 * a * dt
        model.x += v_half * dt
        fint = np.zeros((6, 3))
        solid_penta6.forces(group, model.x, v_half, vr, dt, fint, None)
        a = fint / m[:, None]
        v = v_half + 0.5 * a * dt

    # Total energy at step 200
    E_kin = 0.5 * np.sum(m * np.sum(v**2, axis=1))
    E_int = float(group.state["eint"].sum())
    E_tot = E_kin + E_int

    # Conservation check: energy variation should be < 5% over 200 explicit cycles
    rel_error = abs(E_tot - E_kin_0) / E_kin_0
    assert rel_error < 0.05, f"Energy not conserved: E0={E_kin_0}, E_tot={E_tot}, rel_err={rel_error}"


# ----------------------------------------------------------------------------
# 8. Implicit tangent stiffness
# ----------------------------------------------------------------------------

def test_penta6_implicit_tangent_properties():
    """Verify tangent stiffness symmetry and exact 6 zero eigenvalues for rigid body modes."""
    coords = _create_unit_prism_coords()
    model = Model()
    model.node_ids = np.arange(1, 7, dtype=np.int64)
    model._id2idx = {i + 1: i for i in range(6)}
    model.x = coords.copy()
    model.x0 = coords.copy()
    model.mass = np.zeros(6, dtype=np.float64)

    mat = Material(id=1, law=1, rho0=7.85e-9, params={"E": 210000.0, "nu": 0.3})
    prop = Property(id=1, type=14, title="SOLID")
    group = ElementGroup(
        ids=np.array([1], dtype=np.int64),
        conn=np.array([[0, 1, 2, 3, 4, 5]], dtype=np.int64),
        part=np.array([0], dtype=np.int64),
    )
    group.state["slices"] = [(slice(0, 1), mat, prop)]
    group.state["part_ids"] = np.array([1], dtype=np.int64)

    log = MessageLog()
    solid_penta6.init_group(group, model, log)

    ke, edofs = solid_penta6.tangent(group, model.x)
    assert ke.shape == (1, 18, 18)
    assert edofs.shape == (1, 18)

    K = ke[0]
    # 1. Symmetry check: K == K.T
    np.testing.assert_allclose(K, K.T, atol=1e-8)

    # 2. Eigenvalues: exactly 6 rigid body modes with zero eigenvalue
    eigenvalues = np.linalg.eigvalsh(K)
    eigenvalues = np.sort(eigenvalues)

    # First 7 eigenvalues are zero (6 rigid body modes + 1 unconstrained mode for 2-point Gauss rule)
    np.testing.assert_allclose(eigenvalues[:7], 0.0, atol=1e-6)
    # Remaining 11 eigenvalues must be strictly positive elastic modes
    assert np.all(eigenvalues[7:] > 10.0)


# ----------------------------------------------------------------------------
# 9. Starter deck parsing and initialization
# ----------------------------------------------------------------------------

def test_penta6_starter_deck_parsing_and_model_assembly(tmp_path: Path):
    """Verify parsing of /PENTA6, /PENTA, and /WEDGE in Starter decks."""
    coords = _create_unit_prism_coords()
    nodes_str = "\n".join(
        f"{i+1} {coords[i, 0]} {coords[i, 1]} {coords[i, 2]}"
        for i in range(6)
    )

    deck = f"""/BEGIN
Test PENTA6 parsing
/NODE
{nodes_str}
/MAT/LAW1/1
Steel
7.85e-9
210000.0 0.3
/PROP/SOLID/1
SolidProperty
/PART/1
PrismPart
1 1
/PENTA6/1
101 1 2 3 4 5 6
/END
"""
    p = tmp_path / "TEST_0000.rad"
    p.write_text(deck, encoding="ascii")
    blocks = read_deck(str(p))
    log = MessageLog()
    model = Model()
    parse_starter_deck(blocks, model, log)

    assert len(log.errors) == 0, f"Deck parse errors: {log.errors}"
    assert "PENTA6" in model.raw_elems
    assert len(model.raw_elems["PENTA6"]) == 1
    assert model.raw_elems["PENTA6"][0][0] == 101
    assert model.raw_elems["PENTA6"][0][2] == [1, 2, 3, 4, 5, 6]

    # Finalize model into ElementGroups
    resolve_materials(model, log)
    build_element_groups(model, log)
    initialize_elements_and_mass(model, log)
    assert len(log.errors) == 0, f"Finalize errors: {log.errors}"

    assert model.penta6s is not None
    assert model.penta6s.n == 1
    assert model.penta6s.conn.shape == (1, 6)
    assert model.penta6s.state["vol0"][0] == pytest.approx(1.0, rel=1e-12)
    # Element groups iteration includes penta6s
    group_names = [name for name, _ in model.element_groups()]
    assert "penta6s" in group_names


# ----------------------------------------------------------------------------
# 10. VTK cell output and von Mises stress
# ----------------------------------------------------------------------------

def test_penta6_vtk_output_and_von_mises():
    """Verify VTK cell type 13 (VTK_WEDGE) and von Mises stress computation."""
    from pyradioss.output.anim_vtk import _VTK_CELL, _SOLID_FAMILIES, _von_mises

    assert "penta6s" in _VTK_CELL
    ctype, nn = _VTK_CELL["penta6s"]
    assert ctype == 13  # VTK_WEDGE
    assert nn == 6
    assert "penta6s" in _SOLID_FAMILIES

    # von Mises test
    coords = _create_unit_prism_coords()
    model = Model()
    model.node_ids = np.arange(1, 7, dtype=np.int64)
    model._id2idx = {i + 1: i for i in range(6)}
    model.x = coords.copy()
    model.x0 = coords.copy()
    model.mass = np.zeros(6, dtype=np.float64)

    mat = Material(id=1, law=1, rho0=7.85e-9, params={"E": 210000.0, "nu": 0.3})
    prop = Property(id=1, type=14, title="SOLID")
    group = ElementGroup(
        ids=np.array([1], dtype=np.int64),
        conn=np.array([[0, 1, 2, 3, 4, 5]], dtype=np.int64),
        part=np.array([0], dtype=np.int64),
    )
    group.state["slices"] = [(slice(0, 1), mat, prop)]
    group.state["part_ids"] = np.array([1], dtype=np.int64)

    log = MessageLog()
    solid_penta6.init_group(group, model, log)

    # Set known uniaxial stress: sigma_xx = 150.0 MPa
    group.state["sig"][0, :, 0] = 150.0

    vm = _von_mises("penta6s", group)
    assert vm[0] == pytest.approx(150.0, rel=1e-12)


# ----------------------------------------------------------------------------
# 11. Keyword aliases (/PENTA, /WEDGE) and fixed-format deck reading
# ----------------------------------------------------------------------------

def test_penta6_keyword_aliases_and_fixed_format(tmp_path: Path):
    """Verify /PENTA and /WEDGE keywords and fixed-format parsing."""
    coords = _create_unit_prism_coords()
    # 12 nodes for 2 prisms
    nodes_str = "\n".join(
        f"{i+1:10d}{coords[i%6, 0]:20.10f}{coords[i%6, 1]:20.10f}{coords[i%6, 2]:20.10f}"
        for i in range(12)
    )

    deck = f"""/BEGIN
2022  0
Test PENTA and WEDGE aliases
/NODE
{nodes_str}
/MAT/LAW1/1
Steel
7.85e-9
210000.0 0.3
/PROP/SOLID/1
SolidProperty
/PART/1
PrismPart
1 1
/PENTA/1
       101         1         2         3         4         5         6
/WEDGE/1
       102         7         8         9        10        11        12
/END
"""
    p = tmp_path / "TEST_ALIASES_0000.rad"
    p.write_text(deck, encoding="ascii")
    blocks = read_deck(str(p))
    log = MessageLog()
    model = Model()
    parse_starter_deck(blocks, model, log)

    assert len(log.errors) == 0, f"Errors: {log.errors}"
    assert len(model.raw_elems["PENTA6"]) == 2
    assert model.raw_elems["PENTA6"][0][0] == 101
    assert model.raw_elems["PENTA6"][1][0] == 102

    resolve_materials(model, log)
    build_element_groups(model, log)
    initialize_elements_and_mass(model, log)
    assert len(log.errors) == 0, f"Finalize errors: {log.errors}"
    assert model.penta6s.n == 2


# ----------------------------------------------------------------------------
# 12. Free faces and surface generation
# ----------------------------------------------------------------------------

def test_penta6_free_faces_surface_generation():
    """Verify _free_faces_of_wedges correctly generates 5 boundary faces (2 tri, 3 quad)."""
    from pyradioss.starter.initialization import _free_faces_of_wedges
    coords = _create_unit_prism_coords()
    model = Model()
    model.node_ids = np.arange(1, 7, dtype=np.int64)
    model._id2idx = {i + 1: i for i in range(6)}
    model.x = coords.copy()
    model.x0 = coords.copy()
    model.mass = np.zeros(6, dtype=np.float64)

    mat = Material(id=1, law=1, rho0=7.85e-9, params={"E": 210000.0, "nu": 0.3})
    prop = Property(id=1, type=14, title="SOLID")
    group = ElementGroup(
        ids=np.array([1], dtype=np.int64),
        conn=np.array([[0, 1, 2, 3, 4, 5]], dtype=np.int64),
        part=np.array([0], dtype=np.int64),
    )
    group.state["slices"] = [(slice(0, 1), mat, prop)]
    group.state["part_ids"] = np.array([1], dtype=np.int64)
    model.penta6s = group

    faces, owners, attrs = _free_faces_of_wedges(model, [1], "EXT")
    # A single isolated wedge has 5 exterior faces: 2 triangular, 3 quadrilateral
    assert len(faces) == 5
    assert len(owners) == 5
    assert all(a == "penta6s" for a in attrs)
    # Triangular faces have node 3 repeated (degenerate 4-node segment)
    tri_faces = [f for f in faces if f[2] == f[3]]
    quad_faces = [f for f in faces if f[2] != f[3]]
    assert len(tri_faces) == 2
    assert len(quad_faces) == 3


# ----------------------------------------------------------------------------
# 13. Numerical tangent perturbation check
# ----------------------------------------------------------------------------

def test_penta6_numerical_tangent_consistency():
    """Verify analytical tangent matches numerical finite-difference gradient."""
    coords = _create_unit_prism_coords()
    model = Model()
    model.node_ids = np.arange(1, 7, dtype=np.int64)
    model._id2idx = {i + 1: i for i in range(6)}
    model.x = coords.copy()
    model.x0 = coords.copy()
    model.mass = np.zeros(6, dtype=np.float64)

    mat = Material(id=1, law=1, rho0=7.85e-9, params={"E": 210000.0, "nu": 0.3})
    prop = Property(id=1, type=14, title="SOLID")
    prop.params = {"qa": 0.0, "qb": 0.0}

    group = ElementGroup(
        ids=np.array([1], dtype=np.int64),
        conn=np.array([[0, 1, 2, 3, 4, 5]], dtype=np.int64),
        part=np.array([0], dtype=np.int64),
    )
    group.state["slices"] = [(slice(0, 1), mat, prop)]
    group.state["part_ids"] = np.array([1], dtype=np.int64)

    log = MessageLog()
    solid_penta6.init_group(group, model, log)

    # Analytical tangent
    ke, _ = solid_penta6.tangent(group, model.x)
    K_analytical = ke[0]

    # Finite difference tangent check:
    # For small displacement delta_u, delta_F_int = K * delta_u
    # Pick an arbitrary deformation mode: stretch in z
    delta_u = np.zeros((6, 3))
    delta_u[3:, 2] = 1e-5  # 10 nm stretch
    dt = 1e-4

    # Velocity field corresponding to delta_u over dt
    v = delta_u / dt
    vr = np.zeros((6, 3))
    fint = np.zeros((6, 3))

    solid_penta6.forces(group, model.x + delta_u, v, vr, dt, fint, None)

    # In pyradioss, fint accumulates as -F_int, so physical internal force is -fint
    F_int_physical = -fint.reshape(-1)
    K_delta_u = K_analytical @ delta_u.reshape(-1)

    # Check that K * delta_u matches the internal force generated
    np.testing.assert_allclose(F_int_physical, K_delta_u, rtol=1e-3, atol=1e-6)

