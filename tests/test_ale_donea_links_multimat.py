# tests/test_ale_donea_links_multimat.py
"""Tests for Donea grid smoothing (/ALE/GRID/DONEA),

grid velocity links (/ALE/LINK/VEL, /VEL/ALE),
and multi-material volume fraction remapping (/ALE/MAT).

Faithful verification of OpenRadioss Fortran sources:
  - engine/source/ale/grid/alew.F: Donea distance-weighted grid smoothing
  - engine/source/ale/grid/alelin.F: Grid velocity linking constraints
  - starter/source/ale/bimat/inimu3.F & engine/source/ale/bimat/bimat2.F: Multi-material remapping
"""

import numpy as np
import pytest

from pyradioss.engine.ale_engine import (
    ale_grid_smooth_donea,
    ale_link_velocity,
    ale_multimat_remap,
    compute_hex_volumes,
    build_face_connectivity,
    HEX_FACES,
)
from tests.test_ale_engine import make_hex_mesh


def test_ale_grid_smooth_donea_alpha_zero():
    """When alpha=0, Donea smoothing reduces to standard neighbor average (alew.F lines 105-128)."""
    nodes, conn = make_hex_mesh(nx=2, ny=2, nz=2, lx=2.0, ly=2.0, lz=2.0)
    n_nodes = len(nodes)
    disp = np.zeros((n_nodes, 3))
    vel = np.zeros((n_nodes, 3))

    # Center node in a 2x2x2 mesh is at (1.0, 1.0, 1.0)
    # Give boundary nodes non-zero velocities
    for i in range(n_nodes):
        vel[i] = nodes[i]

    # Constrain all boundary nodes (outer faces of [0, 2]^3)
    bcs = set()
    for i in range(n_nodes):
        x, y, z = nodes[i]
        if x == 0.0 or x == 2.0 or y == 0.0 or y == 2.0 or z == 0.0 or z == 2.0:
            bcs.add(i)

    # Interior center node
    center_nodes = [i for i in range(n_nodes) if i not in bcs]
    assert len(center_nodes) == 1
    c_idx = center_nodes[0]

    w_grid, x_new = ale_grid_smooth_donea(
        nodes, disp, vel, bcs, conn, dt=0.01, alpha=0.0, gamma=1e20
    )

    # Fixed nodes have w_grid == vel
    for b in bcs:
        np.testing.assert_allclose(w_grid[b], vel[b])

    # Center node is at (1, 1, 1), symmetric neighbors average to (1, 1, 1)
    np.testing.assert_allclose(w_grid[c_idx], [1.0, 1.0, 1.0], atol=1e-12)


def test_ale_grid_smooth_donea_displacement_effect():
    """Displacement differences bias grid velocity toward expanding squeezed elements (alew.F lines 147-165)."""
    nodes, conn = make_hex_mesh(nx=2, ny=2, nz=2, lx=2.0, ly=2.0, lz=2.0)
    n_nodes = len(nodes)
    vel = np.ones((n_nodes, 3))
    disp = np.zeros((n_nodes, 3))

    bcs = set()
    for i in range(n_nodes):
        x, y, z = nodes[i]
        if x == 0.0 or x == 2.0 or y == 0.0 or y == 2.0 or z == 0.0 or z == 2.0:
            bcs.add(i)

    c_idx = [i for i in range(n_nodes) if i not in bcs][0]

    # Squeeze neighbors in +x: give +x neighbors a positive displacement
    for i in range(n_nodes):
        if nodes[i, 0] > 1.0:
            disp[i, 0] = 0.2

    w_alpha0, _ = ale_grid_smooth_donea(nodes, disp, vel, bcs, conn, dt=0.1, alpha=0.0, gamma=1e20)
    w_alpha1, _ = ale_grid_smooth_donea(nodes, disp, vel, bcs, conn, dt=0.1, alpha=0.5, gamma=1e20)

    # Positive displacement on +x neighbors creates a positive force FIX on center node, increasing W_x
    assert w_alpha1[c_idx, 0] > w_alpha0[c_idx, 0]


def test_ale_grid_smooth_donea_gamma_bounding():
    """Gamma bounding clamps grid velocity within [(1-gamma)*V, (1+gamma)*V] (alew.F lines 171-179)."""
    nodes, conn = make_hex_mesh(nx=2, ny=2, nz=2, lx=2.0, ly=2.0, lz=2.0)
    n_nodes = len(nodes)
    vel = np.full((n_nodes, 3), 10.0)  # V = 10.0
    disp = np.zeros((n_nodes, 3))

    bcs = set()
    for i in range(n_nodes):
        x, y, z = nodes[i]
        if x == 0.0 or x == 2.0 or y == 0.0 or y == 2.0 or z == 0.0 or z == 2.0:
            bcs.add(i)

    c_idx = [i for i in range(n_nodes) if i not in bcs][0]

    # Try gamma = 0.2 -> bounds are [0.8*V, 1.2*V] = [8.0, 12.0]
    gamma = 0.2
    # Extreme displacement to drive unconstrained W way above 12.0
    for i in range(n_nodes):
        disp[i] = [10.0, 10.0, 10.0]

    w_grid, _ = ale_grid_smooth_donea(nodes, disp, vel, bcs, conn, dt=0.01, alpha=1.0, gamma=gamma)
    assert w_grid[c_idx, 0] <= 10.0 * (1.0 + gamma) + 1e-12
    assert w_grid[c_idx, 0] >= 10.0 * (1.0 - gamma) - 1e-12


def test_ale_link_velocity_linear_interpolation():
    """IM=0 linearly interpolates velocity between master nodes M1 and M2 (alelin.F lines 155-162)."""
    w = np.zeros((10, 3))
    # M1 = node 0, M2 = node 4
    # Slave nodes: [1, 2, 3] (3 nodes, so steps are 1/4, 2/4, 3/4)
    w[0] = [10.0, 20.0, 30.0]
    w[4] = [50.0, 60.0, 70.0]

    link = {
        "m1": 0,
        "m2": 4,
        "nodes": [1, 2, 3],
        "ic": 7,  # XYZ
        "im": 0,  # linear
    }

    w_res = ale_link_velocity(w, [link])

    # Node 1: frac = 1/4 -> 10 + 40*(1/4) = 20
    np.testing.assert_allclose(w_res[1], [20.0, 30.0, 40.0])
    # Node 2: frac = 2/4 -> 10 + 40*(2/4) = 30
    np.testing.assert_allclose(w_res[2], [30.0, 40.0, 50.0])
    # Node 3: frac = 3/4 -> 10 + 40*(3/4) = 40
    np.testing.assert_allclose(w_res[3], [40.0, 50.0, 60.0])


def test_ale_link_velocity_max_magnitude():
    """IM>0 takes maximum magnitude of velocities between M1 and M2 (alelin.F lines 174-178)."""
    w = np.zeros((5, 3))
    w[0] = [-15.0, 5.0, 0.0]
    w[4] = [10.0, -25.0, 0.0]

    link = {
        "m1": 0,
        "m2": 4,
        "nodes": [1, 2],
        "ic": 7,
        "im": 1,  # max
    }

    w_res = ale_link_velocity(w, [link])
    # X: |-15| > |10| -> -15.0
    # Y: |-25| > |5|  -> -25.0
    for node in [1, 2]:
        assert w_res[node, 0] == -15.0
        assert w_res[node, 1] == -25.0


def test_ale_link_velocity_direction_mask():
    """IC mask controls which direction components are linked (alelin.F lines 138-144)."""
    w = np.zeros((5, 3))
    w[0] = [10.0, 20.0, 30.0]
    w[4] = [50.0, 60.0, 70.0]
    w[1] = [1.0, 2.0, 3.0]

    # IC = 4 -> bit 2 set -> X only (bitmask: X=4, Y=2, Z=1)
    link = {
        "m1": 0,
        "m2": 4,
        "nodes": [1],
        "ic": 4,
        "im": 0,
    }

    w_res = ale_link_velocity(w, [link])
    # X interpolated: 10 + 40*(1/2) = 30.0
    assert w_res[1, 0] == 30.0
    # Y and Z remain unchanged at original values
    assert w_res[1, 1] == 2.0
    assert w_res[1, 2] == 3.0


def test_ale_multimat_remap_conservation_and_sum():
    """Multi-material volume fractions must sum to 1.0 and conserve individual masses (bimat2.F, inimu3.F)."""
    nodes_old, conn = make_hex_mesh(nx=3, ny=3, nz=3, lx=3.0, ly=3.0, lz=3.0)
    n_elem = len(conn)
    v_old = compute_hex_volumes(nodes_old, conn)

    # 3-material mixture: e.g., Air (1.2 kg/m^3), Water (1000 kg/m^3), Steel (7800 kg/m^3)
    densities = np.array([1.2, 1000.0, 7800.0])
    vol_frac_old = np.zeros((n_elem, 3))

    # Gradient in material fraction along X
    xe = nodes_old[conn]
    xc = np.mean(xe, axis=1)  # (n_elem, 3)
    frac_mat1 = np.clip(xc[:, 0] / 3.0, 0.0, 1.0)
    frac_mat2 = np.clip((1.0 - frac_mat1) * 0.6, 0.0, 1.0)
    frac_mat3 = np.clip(1.0 - frac_mat1 - frac_mat2, 0.0, 1.0)

    vol_frac_old[:, 0] = frac_mat1
    vol_frac_old[:, 1] = frac_mat2
    vol_frac_old[:, 2] = frac_mat3
    # Check sum == 1
    np.testing.assert_allclose(np.sum(vol_frac_old, axis=1), 1.0)

    # Initial masses of each material
    mass_old_m = np.zeros(3)
    for m in range(3):
        mass_old_m[m] = np.sum(vol_frac_old[:, m] * v_old * densities[m])

    # Deform interior nodes
    nodes_new = nodes_old.copy()
    for i in range(len(nodes_new)):
        x, y, z = nodes_new[i]
        if 0.0 < x < 3.0 and 0.0 < y < 3.0 and 0.0 < z < 3.0:
            nodes_new[i, 0] += 0.05 * np.sin(np.pi * y / 3.0)

    vol_frac_new, rho_mix_new = ale_multimat_remap(
        vol_frac_old, densities, nodes_old, nodes_new, conn, limiter="van_leer"
    )

    # 1. Volume fractions must strictly sum to 1.0 everywhere
    np.testing.assert_allclose(np.sum(vol_frac_new, axis=1), 1.0, atol=1e-12)

    # 2. No negative volume fractions
    assert np.all(vol_frac_new >= 0.0)

    # 3. Mixture density is positive and within [min(rho), max(rho)]
    assert np.all(rho_mix_new >= np.min(densities) - 1e-12)
    assert np.all(rho_mix_new <= np.max(densities) + 1e-12)

    # 4. Total mass of each material is conserved across the remap
    v_new = compute_hex_volumes(nodes_new, conn)
    mass_new_m = np.zeros(3)
    for m in range(3):
        mass_new_m[m] = np.sum(vol_frac_new[:, m] * v_new * densities[m])
        # Conserved to within 3% on coarse deformed grid
        np.testing.assert_allclose(
            mass_new_m[m], mass_old_m[m], rtol=0.03,
            err_msg=f"Material {m} mass not conserved: old={mass_old_m[m]}, new={mass_new_m[m]}"
        )
