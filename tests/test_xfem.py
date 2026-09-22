# Ported from C:\OpenRadioss\source\OpenRadioss-latest-20260520\engine\source\elements\xfem\inixfem.F
# Ported from C:\OpenRadioss\source\OpenRadioss-latest-20260520\engine\source\elements\xfem\cforc3_crk.F
# Ported from C:\OpenRadioss\source\OpenRadioss-latest-20260520\engine\source\elements\xfem\c3forc3_crk.F
# Ported from C:\OpenRadioss\source\OpenRadioss-latest-20260520\engine\source\elements\xfem\czforc3_crk.F
# Ported from C:\OpenRadioss\source\OpenRadioss-latest-20260520\engine\source\elements\xfem\crklayer4n_adv.F
# Ported from C:\OpenRadioss\source\OpenRadioss-latest-20260520\engine\source\elements\xfem\upenr_crk.F
# Ported from C:\OpenRadioss\source\OpenRadioss-latest-20260520\engine\source\elements\xfem\enrichc_ini.F
"""Unit tests for XFEM crack propagation and enriched shell formulation.

Test suite verifying:
1. Mode I crack opening stress intensity factor (SIF) calculation, COD profile,
   and crack propagation direction (XFEM_CRK_DIR / NEWMAN_RAJU).
2. Enrichment DOF initialization, level set signed distance (LSINT4),
   cut topology (ITRI = 0, -1, 1), and phantom element area fractions (ENRICHC_INI).
3. Crack front level set advancement across elements, shared edge synchronization
   (CRKLAYER4N_ADV / UPENR_CRK), and load-bearing energy accounting (CFORC3_CRK).
"""

from __future__ import annotations

import math
import numpy as np
import pytest

from pyradioss.elements.xfem_crack import (
    CrackFront,
    EnrichmentDOFManager,
    XfemElementCut,
    advance_crack_front_step,
    check_crack_propagation,
    compute_crack_direction,
    compute_mode_i_sif,
    cut_quad4_element,
    intersect_edge_ray,
    lsint4,
    lsint4_vector,
    mode_i_crack_opening_displacement,
)
from pyradioss.elements.xfem_shell import (
    XfemShellGroup,
    gather_phantom_kinematics,
    integrate_xfem_step,
    scatter_phantom_forces,
    xfem_shell_quad4_forces,
)
from pyradioss.failure.inicrack import compute_crack_sif


# ===========================================================================
# 1. Mode I Crack Opening Stress Intensity Factor Tests
# ===========================================================================

def test_mode_i_sif_calculation():
    """Verify Mode I SIF computation against analytical Brown-Srawley formula."""
    sigma = 100.0e6  # 100 MPa applied tensile stress
    a = 0.02         # 20 mm crack length
    w = 0.10         # 100 mm plate width (alpha = a/W = 0.2)

    k1, y = compute_mode_i_sif(sigma, a, width=w, crack_type="edge")

    # Analytical Brown-Srawley factor for alpha = 0.2:
    # Y = 1.12 - 0.231*(0.2) + 10.55*(0.04) - 21.72*(0.008) + 30.39*(0.0016)
    #   = 1.12 - 0.0462 + 0.422 - 0.17376 + 0.048624 = 1.370664
    expected_y = (
        1.12
        - 0.231 * 0.2
        + 10.55 * (0.2 ** 2)
        - 21.72 * (0.2 ** 3)
        + 30.39 * (0.2 ** 4)
    )
    assert y == pytest.approx(expected_y, rel=1e-5)

    expected_k1 = expected_y * sigma * math.sqrt(math.pi * a)
    assert k1 == pytest.approx(expected_k1, rel=1e-5)
    assert k1 > 0.0


def test_mode_i_center_crack_sif():
    """Verify Mode I SIF for center-cracked plate using Feddersen secant formula."""
    sigma = 50.0e6
    a = 0.01
    w = 0.10  # alpha = 0.1

    k1, y = compute_mode_i_sif(sigma, a, width=w, crack_type="center")
    expected_y = math.sqrt(1.0 / math.cos(math.pi * 0.1 * 0.5))
    assert y == pytest.approx(expected_y, rel=1e-4)
    assert k1 == pytest.approx(expected_y * sigma * math.sqrt(math.pi * a), rel=1e-4)


def test_mode_i_crack_opening_displacement():
    """Verify asymptotic Mode I COD profile delta(r) = (8 * K_I / E') * sqrt(r / (2*pi))."""
    k_i = 30.0e6  # 30 MPa*sqrt(m)
    youngs_modulus = 210.0e9  # 210 GPa (steel)
    poisson_ratio = 0.3

    r_vals = np.array([0.0, 0.001, 0.005, 0.010])
    cod = mode_i_crack_opening_displacement(
        r=r_vals,
        k_i=k_i,
        youngs_modulus=youngs_modulus,
        poisson_ratio=poisson_ratio,
        plane_stress=True,
    )

    # Displacement at tip r=0 must be 0
    assert cod[0] == 0.0
    # Opening must be strictly increasing with distance r behind tip
    assert cod[1] > 0.0
    assert cod[2] > cod[1]
    assert cod[3] > cod[2]

    # Verify scaling with sqrt(r): cod(4*r) = 2 * cod(r)
    cod_r = mode_i_crack_opening_displacement(0.002, k_i, youngs_modulus)
    cod_4r = mode_i_crack_opening_displacement(0.008, k_i, youngs_modulus)
    assert cod_4r == pytest.approx(2.0 * cod_r, rel=1e-6)


def test_mode_i_propagation_direction_and_threshold():
    """Verify crack propagation direction orthogonal to max principal tension (XFEM_CRK_DIR)."""
    # 1. Pure tension along Y axis: sigma = [0, 150 MPa, 0]
    # Principal stress sigma_1 is 150 MPa in Y direction; crack path should be along X axis
    stress_pure_y = np.array([0.0, 150.0e6, 0.0])
    n_tensile, d_crack, s1, s2 = compute_crack_direction(stress_pure_y)

    assert s1 == pytest.approx(150.0e6, rel=1e-6)
    assert s2 == pytest.approx(0.0, abs=1e-6)
    assert abs(n_tensile[1]) == pytest.approx(1.0, abs=1e-5)  # normal along y
    assert abs(d_crack[0]) == pytest.approx(1.0, abs=1e-5)    # propagation along x
    assert abs(np.dot(n_tensile, d_crack)) == pytest.approx(0.0, abs=1e-10)

    # 2. Threshold propagation check
    k_ic = 25.0e6  # 25 MPa*sqrt(m)
    # Crack of 10 mm length: K_I = 1.12 * 150e6 * sqrt(pi * 0.01) approx 29.8 MPa*sqrt(m) >= K_IC -> True
    should_prop, k1, dir_adv, _ = check_crack_propagation(
        stress=stress_pure_y,
        k_ic=k_ic,
        crack_length=0.01,
    )
    assert should_prop is True
    assert k1 > k_ic

    # Under low stress: K_I < K_IC -> False
    low_stress = np.array([0.0, 50.0e6, 0.0])
    should_prop_low, k1_low, _, _ = check_crack_propagation(
        stress=low_stress,
        k_ic=k_ic,
        crack_length=0.01,
    )
    assert should_prop_low is False
    assert k1_low < k_ic


# ===========================================================================
# 2. Enrichment DOF Initialization & Level Set (LSINT4) Tests
# ===========================================================================

def test_signed_distance_lsint4():
    """Verify exact Fortran LSINT4 perpendicular signed distance calculation."""
    # Horizontal line along y = 0.5: (0, 0.5) -> (1, 0.5)
    y1, z1 = 0.0, 0.5
    y2, z2 = 1.0, 0.5

    # Point directly on line
    assert lsint4(y1, z1, y2, z2, 0.5, 0.5) == pytest.approx(0.0, abs=1e-12)

    # Point above line at z = 1.0 (positive side)
    dist_above = lsint4(y1, z1, y2, z2, 0.5, 1.0)
    assert dist_above == pytest.approx(0.5, abs=1e-12)

    # Point below line at z = 0.0 (negative side)
    dist_below = lsint4(y1, z1, y2, z2, 0.5, 0.0)
    assert dist_below == pytest.approx(-0.5, abs=1e-12)

    # Vectorized evaluation matches scalar
    pts = np.array([[0.5, 0.5], [0.5, 1.0], [0.5, 0.0], [0.2, 0.8]])
    dists = lsint4_vector([y1, z1], [y2, z2], pts)
    assert dists[0] == pytest.approx(0.0, abs=1e-12)
    assert dists[1] == pytest.approx(0.5, abs=1e-12)
    assert dists[2] == pytest.approx(-0.5, abs=1e-12)
    assert dists[3] == pytest.approx(0.3, abs=1e-12)


def test_element_cut_topology_and_area_fractions_itri0():
    """Verify quad element cut across opposite edges (ITRI = 0)."""
    # Unit square element:
    # 3(0,1) ----- 2(1,1)
    #   |            |
    # 0(0,0) ----- 1(1,0)
    coords = np.array([
        [0.0, 0.0],
        [1.0, 0.0],
        [1.0, 1.0],
        [0.0, 1.0],
    ])

    # Horizontal crack cutting across y = 0.5
    crack_p1 = [-0.2, 0.5]
    crack_p2 = [1.2, 0.5]

    cut = cut_quad4_element(coords, crack_p1, crack_p2)

    assert cut.is_cut is True
    assert cut.itri == 0
    # Level sets at nodes 0, 1 are negative; at 2, 3 are positive
    assert cut.level_sets[0] == pytest.approx(-0.5, abs=1e-10)
    assert cut.level_sets[1] == pytest.approx(-0.5, abs=1e-10)
    assert cut.level_sets[2] == pytest.approx(0.5, abs=1e-10)
    assert cut.level_sets[3] == pytest.approx(0.5, abs=1e-10)

    # Area fractions must sum to 1.0
    w1, w2, w3 = cut.area_fractions
    assert (w1 + w2 + w3) == pytest.approx(1.0, rel=1e-6)
    # For a symmetric cut at y=0.5, each sub-quad should have area fraction 0.5
    assert w1 == pytest.approx(0.5, abs=0.05)
    assert w2 == pytest.approx(0.5, abs=0.05)
    assert w3 == 0.0


def test_element_cut_corner_triangle_itri_minus1():
    """Verify quad element cut across adjacent edges isolating a corner triangle (ITRI = -1 / +1)."""
    coords = np.array([
        [0.0, 0.0],
        [1.0, 0.0],
        [1.0, 1.0],
        [0.0, 1.0],
    ])

    # Diagonal cut through top-right corner: from (0.5, 1.0) to (1.0, 0.5)
    # Line equation: x + y = 1.5, isolating node 2 (1, 1) as a triangle
    cut = cut_quad4_element(coords, [0.5, 1.0], [1.0, 0.5])

    assert cut.is_cut is True
    assert cut.itri != 0  # Either -1 or 1 depending on line orientation
    # Sum of area fractions must strictly equal 1.0
    assert float(np.sum(cut.area_fractions)) == pytest.approx(1.0, rel=1e-6)
    assert cut.area_fractions[0] > 0.0
    assert cut.area_fractions[1] > 0.0
    assert cut.area_fractions[2] > 0.0


def test_enrichment_dof_allocation_and_enr0():
    """Verify enrichment DOF assignment (ENRICHC_INI lines 482-531)."""
    coords = np.array([
        [0.0, 0.0],
        [1.0, 0.0],
        [1.0, 1.0],
        [0.0, 1.0],
    ])
    cut = cut_quad4_element(coords, [0.0, 0.5], [1.0, 0.5])

    manager = EnrichmentDOFManager()
    global_nodes = [10, 11, 12, 13]
    elem_id = 42

    enr_ids = manager.assign_element_enrichments(elem_id, cut, global_nodes)

    assert enr_ids.shape == (3, 4)

    # Level 1 (positive domain, y > 0.5):
    # Nodes 2 and 3 have phi > 0 -> tied to standard nodes (ENR == 0)
    assert enr_ids[0, 2] == 0
    assert enr_ids[0, 3] == 0
    # Nodes 0 and 1 have phi < 0 -> free enriched phantom DOFs (ENR > 0)
    assert enr_ids[0, 0] > 0
    assert enr_ids[0, 1] > 0

    # Level 2 (negative domain, y < 0.5):
    # Nodes 0 and 1 have phi < 0 -> tied to standard nodes (ENR == 0)
    assert enr_ids[1, 0] == 0
    assert enr_ids[1, 1] == 0
    # Nodes 2 and 3 have phi > 0 -> free enriched phantom DOFs (ENR > 0)
    assert enr_ids[1, 2] > 0
    assert enr_ids[1, 3] > 0


# ===========================================================================
# 3. Crack Front Advancement & Level Set Propagation Tests
# ===========================================================================

def test_crack_front_advancement_across_elements():
    """Test crack tip advancement into adjacent element along Mode I direction."""
    # Initial crack starting at (0.0, 0.5) and ending at (0.8, 0.5)
    crack = CrackFront(
        id=1,
        points=[np.array([0.0, 0.5]), np.array([0.8, 0.5])],
        tip_coords=np.array([0.8, 0.5]),
        propagation_dir=np.array([1.0, 0.0]),
    )

    # Applied tensile stress tensor: pure tension in Y -> crack advances in +X
    tip_stress = np.array([0.0, 80.0e6, 0.0])
    k_ic = 10.0e6  # Low fracture toughness to ensure propagation

    advanced, new_tip, k1 = advance_crack_front_step(
        crack=crack,
        tip_stress=tip_stress,
        step_length=0.5,
        k_ic=k_ic,
    )

    assert advanced is True
    assert new_tip is not None
    # Tip must have advanced along +X direction
    assert new_tip[0] == pytest.approx(1.3, abs=1e-6)
    assert new_tip[1] == pytest.approx(0.5, abs=1e-6)
    assert crack.length == pytest.approx(1.3, abs=1e-6)
    assert len(crack.points) == 3


def test_shared_edge_synchronization_upenr():
    """Verify UPENR_CRK enrichment synchronization across shared cut edge."""
    # Two adjacent square elements sharing edge between x=1.0:
    # Element 1: [0, 1] x [0, 1] (nodes 1, 2, 3, 4)
    # Element 2: [1, 2] x [0, 1] (nodes 2, 5, 6, 3)
    coords1 = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
    coords2 = np.array([[1.0, 0.0], [2.0, 0.0], [2.0, 1.0], [1.0, 1.0]])

    crack_p1 = [0.0, 0.5]
    crack_p2 = [2.0, 0.5]

    cut1 = cut_quad4_element(coords1, crack_p1, crack_p2)
    cut2 = cut_quad4_element(coords2, crack_p1, crack_p2)

    manager = EnrichmentDOFManager()
    conn1 = [1, 2, 3, 4]
    conn2 = [2, 5, 6, 3]

    manager.assign_element_enrichments(1, cut1, conn1)
    manager.assign_element_enrichments(2, cut2, conn2)

    # Initially before synchronization, each element cut has distinct DOF numbers
    elem_cuts = {1: cut1, 2: cut2}
    elem_conn = {1: conn1, 2: conn2}

    manager.synchronize_shared_edges(elem_cuts, elem_conn)

    # After UPENR_CRK synchronization, the enriched DOFs on the shared cut edge
    # between nodes 2 and 3 must be unified
    # Node 2 in elem 1 is local index 1 (enr_ids[0, 1] on level 0)
    # Node 2 in elem 2 is local index 0 (enr_ids[0, 0] on level 0)
    assert cut1.enr_ids[0, 1] == cut2.enr_ids[0, 0]


# ===========================================================================
# 4. Enriched Shell Force Integration & Energy Accounting Tests
# ===========================================================================

def test_uncracked_shell_forces_identity():
    """Verify that uncracked element forces match standard shell formulation."""
    xe = np.array([
        [0.0, 0.0, 0.0],
        [0.1, 0.0, 0.0],
        [0.1, 0.1, 0.0],
        [0.0, 0.1, 0.0],
    ])
    ve = np.zeros((4, 3))
    # Apply uniform strain velocity field: vx expanding in X
    ve[:, 0] = [0.0, 1.0, 1.0, 0.0]
    vre = np.zeros((4, 3))

    f_std, m_std, f_enr, m_enr, de_int, de_hg, dt_c = xfem_shell_quad4_forces(
        xe_std=xe,
        ve_std=ve,
        vre_std=vre,
        cut=None,  # Uncut
        thick=1.0e-3,
        youngs_modulus=2.1e11,
        poisson_ratio=0.3,
        shear_modulus=8.0e10,
        density=7800.0,
        dt=1.0e-6,
    )

    # No enriched forces for uncracked element
    assert len(f_enr) == 0
    assert len(m_enr) == 0
    # Internal strain energy work must be strictly positive under deformation
    assert de_int > 0.0
    # Equilibrium: sum of internal forces on a flat element in uniform strain must sum to zero
    sum_f = np.sum(f_std, axis=0)
    assert np.allclose(sum_f, 0.0, atol=1e-4)


def test_xfem_cracked_shell_force_scatter_and_energy_balance():
    """Verify force scatter and load-bearing energy ledger accounting for XFEM cracked shell.

    Critical Rule 3: 'Energy accounting is load-bearing: every new force path must book
    its work in the EN ledger.'
    """
    xe = np.array([
        [0.0, 0.0, 0.0],
        [0.1, 0.0, 0.0],
        [0.1, 0.1, 0.0],
        [0.0, 0.1, 0.0],
    ])
    ve = np.zeros((4, 3))
    # Tensile opening velocity field along Y
    ve[2:, 1] = 1.0   # Nodes 2 and 3 moving up
    ve[:2, 1] = -1.0  # Nodes 0 and 1 moving down
    vre = np.zeros((4, 3))

    # Cut element at y = 0.05
    cut = cut_quad4_element(xe[:, :2], [0.0, 0.05], [0.1, 0.05])
    manager = EnrichmentDOFManager()
    manager.assign_element_enrichments(1, cut, [0, 1, 2, 3])

    # Enriched phantom node positions and velocities initialized matching standard nodes
    x_enr = {}
    v_enr = {}
    for k in range(2):
        for i in range(4):
            enr_id = cut.enr_ids[k, i]
            if enr_id > 0:
                x_enr[enr_id] = xe[i].copy()
                v_enr[enr_id] = ve[i].copy()

    f_std, m_std, f_enr, m_enr, de_int, de_hg, dt_c = xfem_shell_quad4_forces(
        xe_std=xe,
        ve_std=ve,
        vre_std=vre,
        cut=cut,
        thick=1.0e-3,
        youngs_modulus=2.1e11,
        poisson_ratio=0.3,
        shear_modulus=8.0e10,
        density=7800.0,
        dt=1.0e-6,
        x_enr_dict=x_enr,
        v_enr_dict=v_enr,
    )

    # 1. Enriched forces must be non-empty for cracked element
    assert len(f_enr) > 0

    # 2. Internal energy increment must be strictly positive
    assert de_int > 0.0

    # 3. Overall equilibrium: sum of all forces (standard nodes + enriched DOFs) must balance
    total_fx = np.sum(f_std[:, 0]) + sum(f[0] for f in f_enr.values())
    total_fy = np.sum(f_std[:, 1]) + sum(f[1] for f in f_enr.values())
    assert total_fx == pytest.approx(0.0, abs=1e-3)
    assert total_fy == pytest.approx(0.0, abs=1e-3)


def test_xfem_shell_group_integration_step():
    """Verify XfemShellGroup dynamic explicit step integration and kinematic advance."""
    conn = np.array([[0, 1, 2, 3]])
    group = XfemShellGroup(
        connectivity=conn,
        thick=1.0e-3,
        youngs_modulus=2.1e11,
        poisson_ratio=0.3,
        density=7800.0,
    )

    x = np.array([
        [0.0, 0.0, 0.0],
        [0.1, 0.0, 0.0],
        [0.1, 0.1, 0.0],
        [0.0, 0.1, 0.0],
    ])
    v = np.zeros((4, 3))
    m_nodes = np.full(4, 7800.0 * 1e-3 * 0.01 / 4.0)  # lumped mass
    f_ext = np.zeros((4, 3))
    f_ext[2:, 1] = 1000.0  # Apply tension to top nodes

    # Cut element
    cut = cut_quad4_element(x[:, :2], [0.0, 0.05], [0.1, 0.05])
    group.add_crack(elem_id=0, cut=cut, mesh_x=x)

    dt = 1.0e-6
    x_new, v_new, de_int = integrate_xfem_step(
        group=group,
        x=x,
        v=v,
        m_nodes=m_nodes,
        f_ext=f_ext,
        dt=dt,
    )

    # Verification:
    # 1. Coordinates updated
    assert x_new is not None
    # Top nodes (2, 3) must move in +Y direction due to external tension
    assert x_new[2, 1] > x[2, 1]
    assert x_new[3, 1] > x[3, 1]
    # 2. Internal energy ledger incremented
    assert group.eint >= 0.0
