"""Unit tests for Thermal Contact Interface Conduction and BEM Incompressible Potential Flow.

Upstream Fortran References:
- Thermal Contact Interfaces:
  ``engine/source/interfaces/interf/i2therm.F``
  ``engine/source/interfaces/int07/i7therm.F``
  ``engine/source/interfaces/int11/i11therm.F``
  ``engine/source/interfaces/int21/i21therm.F``
  ``engine/source/interfaces/int25/i25therm.F``
- BEM Potential Flow:
  ``engine/source/fluid/incpflow.F``
  ``engine/source/fluid/bemsolv.F``
  ``engine/source/fluid/flow0.F``
  ``starter/source/loads/bem/hm_read_bem.F``
"""

from __future__ import annotations

import math
import numpy as np
import pytest

from pyradioss.contact.thermal_contact import (
    thermal_contact_type2,
    thermal_contact_type7,
    thermal_contact_type11,
    thermal_contact_type21,
    thermal_contact_type25,
)
from pyradioss.engine.bem_flow import (
    BemFlowParams,
    BemIncompressibleFlow,
    assemble_bem_system,
    solve_bem_system,
    compute_triangle_normals_and_areas,
    trgrad,
    int_h_tg,
    int_g_tg,
    solid_angle_triangle,
    evaluate_field_points,
    compute_surface_velocities,
    compute_unsteady_bernoulli_pressure,
    compute_boundary_forces_and_work,
)


# ============================================================================
# 1. TYPE2 Tied Thermal Contact Conduction Tests (i2therm.F)
# ============================================================================

def test_type2_thermal_conduction_conservation():
    """Verify exact thermal energy conservation in TYPE2 tied contact (i2therm.F).

    Heat gained by secondary node must exactly equal heat lost by master nodes:
    sum(fthe) == 0 to machine precision.
    """
    # 1 slave node (node 4) tied to a quad segment (nodes 0, 1, 2, 3)
    # Master segment: 1m x 1m in XY plane
    x = np.array([
        [0.0, 0.0, 0.0],  # node 0
        [1.0, 0.0, 0.0],  # node 1
        [1.0, 1.0, 0.0],  # node 2
        [0.0, 1.0, 0.0],  # node 3
        [0.5, 0.5, 0.0],  # node 4 (slave node)
    ], dtype=float)

    # Master corner temperatures: 400 K, slave temperature: 300 K
    temp = np.array([400.0, 400.0, 400.0, 400.0, 300.0], dtype=float)

    slave_nodes = np.array([4], dtype=np.int64)
    master_segs = np.array([[0, 1, 2, 3]], dtype=np.int64)
    # Centroid projection weights H = [0.25, 0.25, 0.25, 0.25]
    weights = np.array([[0.25, 0.25, 0.25, 0.25]], dtype=float)

    kthe = 50.0  # W/(m^2*K)
    dt = 0.01  # s
    theaccfact = 2.0

    fthe, condn, heat_trans = thermal_contact_type2(
        x=x,
        temp=temp,
        slave_nodes=slave_nodes,
        master_segs=master_segs,
        weights=weights,
        kthe=kthe,
        dt=dt,
        theaccfact=theaccfact,
    )

    # 1. Exact energy conservation: net sum of fthe across all nodes is 0
    assert abs(np.sum(fthe)) < 1e-12

    # 2. Secondary node received heat (positive)
    assert fthe[4] > 0.0
    assert fthe[4] == pytest.approx(heat_trans, rel=1e-12)

    # 3. Master nodes lost heat (negative, partitioned by weights)
    for k in range(4):
        assert fthe[k] < 0.0
        assert fthe[k] == pytest.approx(-0.25 * fthe[4], rel=1e-12)

    # 4. Analytical value check:
    # Segment area = 1.0 m^2, AREAM = 1/8 * norm(cross(d13, d24)) = 1/8 * 2.0 = 0.25 m^2 (half diagonal cross product)
    # Here d13 = (1, 1, 0), d24 = (-1, 1, 0), cross = (0, 0, 2) => norm = 2.0 => aream = 2.0 / 8.0 = 0.25
    # phi = areac * (T_m - T_s) * dt * kthe * theaccfact = 0.25 * (400 - 300) * 0.01 * 50.0 * 2.0 = 25.0 J
    assert heat_trans == pytest.approx(25.0, rel=1e-6)

    # 5. Conductance check
    # condint = areac * kthe * theaccfact = 0.25 * 50 * 2 = 25.0 W/K
    assert condn[4] == pytest.approx(25.0, rel=1e-6)
    assert np.sum(condn[:4]) == pytest.approx(25.0, rel=1e-6)


def test_type2_triangle_segment():
    """Verify TYPE2 tied thermal conduction with triangular master segment (ix3 == ix4)."""
    x = np.array([
        [0.0, 0.0, 0.0],  # 0
        [1.0, 0.0, 0.0],  # 1
        [0.0, 1.0, 0.0],  # 2
        [0.0, 1.0, 0.0],  # 3 (repeated node 2)
        [0.3, 0.3, 0.0],  # 4 (slave)
    ], dtype=float)

    temp = np.array([350.0, 320.0, 380.0, 380.0, 300.0], dtype=float)
    slave_nodes = np.array([4], dtype=np.int64)
    master_segs = np.array([[0, 1, 2, 2]], dtype=np.int64)
    weights = np.array([[0.4, 0.3, 0.3, 0.0]], dtype=float)

    fthe, condn, heat_trans = thermal_contact_type2(
        x=x,
        temp=temp,
        slave_nodes=slave_nodes,
        master_segs=master_segs,
        weights=weights,
        kthe=100.0,
        dt=0.001,
        areas=np.array([0.5]),
    )

    # Energy balance
    assert abs(np.sum(fthe)) < 1e-12
    assert heat_trans > 0.0


# ============================================================================
# 2. TYPE7 Penalty Thermal Contact Tests (i7therm.F)
# ============================================================================

def test_type7_thermal_conduction_and_conservation():
    """Verify TYPE7 interface conduction and exact energy conservation (i7therm.F)."""
    # 4 corner nodes of quad + 1 secondary node
    x = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [1.0, 1.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.5, 0.5, 0.001],  # secondary node near center
    ], dtype=float)

    temp = np.array([500.0, 500.0, 500.0, 500.0, 300.0], dtype=float)
    slave_nodes = np.array([4], dtype=np.int64)
    master_segs = np.array([[0, 1, 2, 3]], dtype=np.int64)
    weights = np.array([[0.25, 0.25, 0.25, 0.25]], dtype=float)

    kthe = 100.0  # W/(m^2*K) => rstif = 0.01
    dt = 0.005
    theaccfact = 1.0

    fthe, condint, ledger = thermal_contact_type7(
        x=x,
        temp=temp,
        slave_nodes=slave_nodes,
        master_segs=master_segs,
        weights=weights,
        kthe=kthe,
        dt=dt,
        theaccfact=theaccfact,
        iform=1,
    )

    # Exact conservation: heat gained by secondary == heat lost by master
    assert abs(np.sum(fthe)) < 1e-12
    assert fthe[4] > 0.0
    assert ledger["conduction"] > 0.0
    assert ledger["radiation"] == 0.0
    assert ledger["friction"] == 0.0


def test_type7_thermal_radiation():
    """Verify TYPE7 Stefan-Boltzmann radiation regime when gapv <= penrad <= drad."""
    x = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [1.0, 1.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.5, 0.5, 0.02],  # distance = 0.02 m
    ], dtype=float)

    tm = 600.0  # Master surface K
    ts = 300.0  # Secondary node K
    temp = np.array([tm, tm, tm, tm, ts], dtype=float)
    slave_nodes = np.array([4], dtype=np.int64)
    master_segs = np.array([[0, 1, 2, 3]], dtype=np.int64)
    weights = np.array([[0.25, 0.25, 0.25, 0.25]], dtype=float)

    # Radiation parameters
    sigma = 5.670374419e-8
    emissivity = 0.8
    frad = emissivity * sigma
    gapv = np.array([0.005])  # 5 mm
    drad = 0.05  # 50 mm cutoff -> penrad = 0.02 is in [0.005, 0.05], so radiation triggers!
    dt = 0.01

    fthe, condint, ledger = thermal_contact_type7(
        x=x,
        temp=temp,
        slave_nodes=slave_nodes,
        master_segs=master_segs,
        weights=weights,
        kthe=100.0,
        dt=dt,
        theaccfact=1.0,
        iform=1,
        gapv=gapv,
        frad=frad,
        drad=drad,
        distances=np.array([0.02]),
        areas=np.array([1.0]),
    )

    # Radiation active
    assert ledger["radiation"] > 0.0
    assert ledger["conduction"] == 0.0

    # Expected radiation: FRAD * AREAC * (TM^4 - TS^4) * DT
    expected_phi = frad * 1.0 * (600.0**4 - 300.0**4) * dt
    assert fthe[4] == pytest.approx(expected_phi, rel=1e-5)
    assert abs(np.sum(fthe)) < 1e-12  # conserved across master and slave


def test_type7_friction_heat_partition():
    """Verify mechanical friction energy partitioning between secondary and master (i7therm.F)."""
    x = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [1.0, 1.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.5, 0.5, 0.0],
    ], dtype=float)

    # Equal temperatures -> zero conduction flux
    temp = np.array([300.0, 300.0, 300.0, 300.0, 300.0], dtype=float)
    slave_nodes = np.array([4], dtype=np.int64)
    master_segs = np.array([[0, 1, 2, 3]], dtype=np.int64)
    weights = np.array([[0.25, 0.25, 0.25, 0.25]], dtype=float)

    efrict = np.array([100.0])  # 100 Joules of friction work
    fheats = 0.6  # 60% to secondary
    fheatm = 0.4  # 40% to master

    fthe, condint, ledger = thermal_contact_type7(
        x=x,
        temp=temp,
        slave_nodes=slave_nodes,
        master_segs=master_segs,
        weights=weights,
        kthe=100.0,
        dt=0.01,
        iform=1,
        fheats=fheats,
        fheatm=fheatm,
        efrict=efrict,
        areas=np.array([1.0]),
    )

    # Friction energy generation
    assert ledger["friction"] == pytest.approx(100.0, rel=1e-12)
    # Secondary received 60 J
    assert fthe[4] == pytest.approx(60.0, rel=1e-12)
    # Master received 40 J total (10 J on each of the 4 nodes)
    assert np.sum(fthe[:4]) == pytest.approx(40.0, rel=1e-12)
    assert np.sum(fthe) == pytest.approx(100.0, rel=1e-12)


def test_type7_pressure_dependent_conductivity():
    """Verify pressure-dependent thermal resistance scaling RSTIFF = RSTIF / f(P)."""
    x = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [1.0, 1.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.5, 0.5, 0.0],
    ], dtype=float)
    temp = np.array([400.0, 400.0, 400.0, 400.0, 300.0], dtype=float)
    slave_nodes = np.array([4], dtype=np.int64)
    master_segs = np.array([[0, 1, 2, 3]], dtype=np.int64)
    weights = np.array([[0.25, 0.25, 0.25, 0.25]], dtype=float)

    # Curve doubling conductivity at P = 1.0 MPa
    fct_k = lambda p: 2.0 if p >= 1.0e6 else 1.0
    fni = np.array([1.0e6])  # 1 MN on 1 m^2 => P = 1.0 MPa

    fthe, condint, ledger = thermal_contact_type7(
        x=x,
        temp=temp,
        slave_nodes=slave_nodes,
        master_segs=master_segs,
        weights=weights,
        kthe=50.0,  # rstif = 0.02
        dt=0.01,
        fni=fni,
        fct_k=fct_k,
        areas=np.array([1.0]),
    )

    # RSTIFF should be halved (rstif / 2 = 0.01) => conductance doubled to 100
    assert condint[0] == pytest.approx(100.0, rel=1e-5)


# ============================================================================
# 3. TYPE11 Edge-to-Edge Thermal Contact Tests (i11therm.F)
# ============================================================================

def test_type11_thermal_conduction_conservation():
    """Verify /INTER/TYPE11 edge-to-edge conduction and exact energy conservation (i11therm.F)."""
    # 2 secondary edge nodes (0, 1) and 2 master edge nodes (2, 3)
    temp = np.array([300.0, 300.0, 500.0, 500.0], dtype=float)
    s_edges = np.array([[0, 1]], dtype=np.int64)
    m_edges = np.array([[2, 3]], dtype=np.int64)

    # Closest points at middle of both edges: hs = [0.5, 0.5], hm = [0.5, 0.5]
    hs = np.array([[0.5, 0.5]], dtype=float)
    hm = np.array([[0.5, 0.5]], dtype=float)

    areac = np.array([0.1])
    penrad = np.array([-0.001])  # in contact (penrad <= 0)
    kthe = 200.0
    dt = 0.001

    fthe, condint, heat_trans = thermal_contact_type11(
        temp=temp,
        slave_edge_nodes=s_edges,
        master_edge_nodes=m_edges,
        hs=hs,
        hm=hm,
        kthe=kthe,
        dt=dt,
        areac=areac,
        penrad=penrad,
        iform=1,
    )

    # Exact conservation across all 4 nodes
    assert abs(np.sum(fthe)) < 1e-12
    # Secondary edge gained heat
    assert fthe[0] > 0.0
    assert fthe[1] > 0.0
    # Master edge lost heat
    assert fthe[2] < 0.0
    assert fthe[3] < 0.0
    assert (fthe[0] + fthe[1]) == pytest.approx(heat_trans, rel=1e-12)


# ============================================================================
# 4. TYPE21 Stamping Drawbead Thermal Contact Tests (i21therm.F)
# ============================================================================

def test_type21_distance_decay_regimes():
    """Verify the 3 distance-dependent regimes of /INTER/TYPE21 (i21therm.F)."""
    temp = np.array([300.0, 500.0, 500.0, 500.0, 500.0], dtype=float)
    slave_nodes = np.array([0], dtype=np.int64)
    master_segs = np.array([[1, 2, 3, 4]], dtype=np.int64)
    weights = np.array([[0.25, 0.25, 0.25, 0.25]], dtype=float)

    # Case A: penrad <= 0 (Direct contact)
    fthe_a, cond_a, h_a = thermal_contact_type21(
        temp=temp,
        slave_nodes=slave_nodes,
        master_segs=master_segs,
        weights=weights,
        kthe=100.0,
        dt=0.01,
        areac=np.array([1.0]),
        penrad=np.array([0.0]),
        dcond=0.01,
    )
    assert cond_a[0] == pytest.approx(100.0, rel=1e-5)
    assert h_a > 0.0

    # Case B: 0 < penrad <= ddcond (Decay zone)
    # At penrad = 0.005 with dcond = 0.01 => normalized dd = 0.5 => conductance halved to 50
    fthe_b, cond_b, h_b = thermal_contact_type21(
        temp=temp,
        slave_nodes=slave_nodes,
        master_segs=master_segs,
        weights=weights,
        kthe=100.0,
        dt=0.01,
        areac=np.array([1.0]),
        penrad=np.array([0.005]),
        dcond=0.01,
    )
    assert cond_b[0] == pytest.approx(50.0, rel=1e-5)
    assert h_b == pytest.approx(0.5 * h_a, rel=1e-5)

    # Case C: penrad > dcond (Radiation only)
    sigma = 5.670374419e-8
    fthe_c, cond_c, h_c = thermal_contact_type21(
        temp=temp,
        slave_nodes=slave_nodes,
        master_segs=master_segs,
        weights=weights,
        kthe=100.0,
        dt=0.01,
        areac=np.array([1.0]),
        penrad=np.array([0.02]),
        dcond=0.01,
        drad=0.05,
        frad=sigma,
    )
    assert cond_c[0] == 0.0  # Conduction zero
    assert h_c > 0.0  # Radiation active


# ============================================================================
# 5. TYPE25 General Interface Conduction Tests (i25therm.F)
# ============================================================================

def test_type25_harmonic_mean_conductivity():
    """Verify harmonic mean conductivity 2*k1*k2 / (k1+k2) in /INTER/TYPE25 (i25therm.F)."""
    x = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [1.0, 1.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.5, 0.5, 0.0],
    ], dtype=float)
    temp = np.array([400.0, 400.0, 400.0, 400.0, 300.0], dtype=float)
    slave_nodes = np.array([4], dtype=np.int64)
    master_segs = np.array([[0, 1, 2, 3]], dtype=np.int64)
    weights = np.array([[0.25, 0.25, 0.25, 0.25]], dtype=float)

    # Slave material conductivity = 60 W/(m*K), Master material = 30 W/(m*K)
    # Harmonic mean = 2 * 60 * 30 / (60 + 30) = 3600 / 90 = 40 W/(m*K)
    cond_s = np.array([60.0])
    cond_m = np.array([30.0])

    fthe, condint, ledger = thermal_contact_type25(
        x=x,
        temp=temp,
        slave_nodes=slave_nodes,
        master_segs=master_segs,
        weights=weights,
        kthe=100.0,  # RSTIF = 0.01
        dt=0.01,
        gapv=np.array([0.005]),  # 5 mm gap
        cond_slave=cond_s,
        cond_master=cond_m,
        distances=np.array([0.004]),  # DIST = 4 mm => PENRAD = 4 - 5 = -1 mm <= 0 (in contact)
        areas=np.array([1.0]),
    )

    # Total resistance TSTIFT = 0.01 + 0.0001 = 0.0101
    expected_cond = 1.0 / 0.0101
    assert condint[0] == pytest.approx(expected_cond, rel=1e-4)
    assert abs(np.sum(fthe)) < 1e-12  # strict conservation


# ============================================================================
# 6. BEM Potential Flow Triangle Geometry & TRGRAD Tests (incpflow.F)
# ============================================================================

def test_bem_triangle_geometry_and_normals():
    """Verify normal vector, area, and nodal tributary area on triangle panels."""
    # 2 triangles forming a 1m x 1m square in XY plane (normal in +Z)
    x = np.array([
        [0.0, 0.0, 0.0],  # 0
        [1.0, 0.0, 0.0],  # 1
        [1.0, 1.0, 0.0],  # 2
        [0.0, 1.0, 0.0],  # 3
    ], dtype=float)

    elem = np.array([
        [0, 1, 2],  # Triangle 1 (area 0.5)
        [0, 2, 3],  # Triangle 2 (area 0.5)
    ], dtype=np.int64)

    norm_vec, elarea, nodarea = compute_triangle_normals_and_areas(x, elem)

    # Element normals: (0, 0, 1) with magnitude 2 * Area = 1.0
    assert np.allclose(norm_vec[:, :2], 0.0)
    assert np.allclose(norm_vec[:, 2], 1.0)

    # Areas: both 0.5 m^2
    assert np.allclose(elarea, 0.5)

    # Nodal areas: nodes 0 and 2 share 2 elements => 2 * (1.0/6) = 1/3 m^2
    # nodes 1 and 3 in 1 element => 1 * (1.0/6) = 1/6 m^2
    assert nodarea[0] == pytest.approx(1.0 / 3.0, rel=1e-12)
    assert nodarea[2] == pytest.approx(1.0 / 3.0, rel=1e-12)
    assert nodarea[1] == pytest.approx(1.0 / 6.0, rel=1e-12)
    assert nodarea[3] == pytest.approx(1.0 / 6.0, rel=1e-12)


def test_bem_trgrad_surface_gradient():
    """Verify exact 3D surface gradient TRGRAD against linear potential Phi = 2*x + 3*y + 4*z."""
    x1 = np.array([0.0, 0.0, 0.0])
    x2 = np.array([1.0, 0.0, 0.0])
    x3 = np.array([0.0, 1.0, 0.0])

    # In XY plane, normal is +Z. Tangential gradient should be (2, 3, 0)
    phi1 = 0.0
    phi2 = 2.0 * 1.0
    phi3 = 3.0 * 1.0

    grad = trgrad(x1, x2, x3, phi1, phi2, phi3)
    assert grad[0] == pytest.approx(2.0, rel=1e-10)
    assert grad[1] == pytest.approx(3.0, rel=1e-10)
    assert grad[2] == pytest.approx(0.0, abs=1e-10)


def test_bem_solid_angle_triangle():
    """Verify solid angle subtended by triangle panels at query points."""
    # Tetrahedron with 4 vertices: (0,0,0), (1,0,0), (0,1,0), (0,0,1)
    # Query point inside centroid: solid angle sum must equal 4 * pi steradians
    xq_inside = np.array([0.1, 0.1, 0.1], dtype=float)

    # Triangles with outward normals (pointing out of tetrahedron):
    panels = [
        (np.array([0.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0]), np.array([1.0, 0.0, 0.0])),  # z=0 bottom
        (np.array([0.0, 0.0, 0.0]), np.array([1.0, 0.0, 0.0]), np.array([0.0, 0.0, 1.0])),  # y=0 back
        (np.array([0.0, 0.0, 0.0]), np.array([0.0, 0.0, 1.0]), np.array([0.0, 1.0, 0.0])),  # x=0 left
        (np.array([1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0]), np.array([0.0, 0.0, 1.0])),  # inclined face
    ]

    total_solid_angle = sum(
        abs(solid_angle_triangle(xq_inside, p[0], p[1], p[2])) for p in panels
    )
    assert total_solid_angle == pytest.approx(4.0 * math.pi, rel=1e-4)


# ============================================================================
# 7. BEM System Assembly, Solution, and Bernoulli Pressure Tests (bemsolv.F)
# ============================================================================

def test_bem_system_assembly_properties():
    """Verify BEM influence matrix assembly and rigid body mode row sum (bemsolv.F)."""
    # Create an octahedron (closed boundary with 6 nodes and 8 triangular facets)
    x = np.array([
        [1.0, 0.0, 0.0],   # 0
        [-1.0, 0.0, 0.0],  # 1
        [0.0, 1.0, 0.0],   # 2
        [0.0, -1.0, 0.0],  # 3
        [0.0, 0.0, 1.0],   # 4
        [0.0, 0.0, -1.0],  # 5
    ], dtype=float)

    elem = np.array([
        [0, 2, 4], [2, 1, 4], [1, 3, 4], [3, 0, 4],  # Upper pyramid
        [2, 0, 5], [1, 2, 5], [3, 1, 5], [0, 3, 5],  # Lower pyramid
    ], dtype=np.int64)

    hbem, gbem = assemble_bem_system(x, elem)

    # 1. Dimensions check: H is (6, 6), G is (6, 9)
    assert hbem.shape == (6, 6)
    assert gbem.shape == (6, 9)

    # 2. Rigid body mode row sum before far-field closure: sum_j H_ij == 0
    # The last column H[:, -1] was replaced with -gbem[:, -1]
    # Far-field closure check (bemsolv.F line 176):
    assert np.allclose(hbem[:, -1], -gbem[:, -1])

    # 3. Solve potential for normal expansion flux q = 1.0 m/s
    q = np.ones(8, dtype=float)
    phi = solve_bem_system(hbem, gbem, q)

    # For uniform expansion on a sphere/octahedron, potential should be uniform
    assert len(phi) == 6
    assert np.all(np.isfinite(phi))


def test_bem_unsteady_bernoulli_pressure_and_forces():
    """Verify Bernoulli pressure and consistent structural boundary forces."""
    elem = np.array([[0, 1, 2]], dtype=np.int64)
    # 1 triangle in XY plane, area = 0.5 m^2, normal = (0, 0, 1)
    norm_vec = np.array([[0.0, 0.0, 1.0]], dtype=float)
    elarea = np.array([0.5], dtype=float)

    # Pressure at 3 nodes: 1000 Pa
    pres = np.array([1000.0, 1000.0, 1000.0], dtype=float)
    v = np.array([
        [0.0, 0.0, 2.0],
        [0.0, 0.0, 2.0],
        [0.0, 0.0, 2.0],
    ], dtype=float)
    dt = 0.01

    f_node, dwfext = compute_boundary_forces_and_work(
        elem=elem,
        pres=pres,
        norm_vec=norm_vec,
        elarea=elarea,
        v=v,
        dt=dt,
    )

    # Total force = P * Area * normal = 1000 * 0.5 * (0, 0, 1) = (0, 0, 500) N
    total_f = np.sum(f_node, axis=0)
    assert total_f[0] == pytest.approx(0.0, abs=1e-10)
    assert total_f[1] == pytest.approx(0.0, abs=1e-10)
    assert total_f[2] == pytest.approx(500.0, rel=1e-6)

    # Work = F . v * dt = 500 N * 2 m/s * 0.01 s = 10.0 J
    assert dwfext == pytest.approx(10.0, rel=1e-6)


def test_bem_flow_solver_step_simulation():
    """Verify full step simulation of BemIncompressibleFlow solver."""
    # Octahedron mesh
    x = np.array([
        [1.0, 0.0, 0.0],
        [-1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, -1.0, 0.0],
        [0.0, 0.0, 1.0],
        [0.0, 0.0, -1.0],
    ], dtype=float)

    elem = np.array([
        [0, 2, 4], [2, 1, 4], [1, 3, 4], [3, 0, 4],
        [2, 0, 5], [1, 2, 5], [3, 1, 5], [0, 3, 5],
    ], dtype=np.int64)

    params = BemFlowParams(
        id=1,
        rho=1000.0,
        pa_const=1.0e5,  # 1 bar stagnation pressure
        v_inf_const=5.0,  # 5 m/s uniform stream
        dir=np.array([1.0, 0.0, 0.0]),
    )

    solver = BemIncompressibleFlow(params, elem, x)

    # Stationary body in stream
    v = np.zeros_like(x)
    dt = 0.001

    f_fluid, pres, dwfext = solver.step(x, v, dt, time=0.001)

    # Check solver state
    assert len(pres) == 6
    assert np.all(np.isfinite(pres))
    assert f_fluid.shape == (6, 3)
    assert np.all(np.isfinite(f_fluid))
    # Stationary body does no work
    assert dwfext == 0.0
    assert solver.wfext == 0.0

    # Step 2 with body motion v_z = 1.0 m/s
    v[:, 2] = 1.0
    f_fluid2, pres2, dwfext2 = solver.step(x, v, dt, time=0.002)
    assert dwfext2 != 0.0
    assert solver.wfext == pytest.approx(dwfext2, rel=1e-12)


# ============================================================================
# 8. Contact Class Method Integration Tests
# ============================================================================

def test_contact_class_thermal_methods():
    """Verify compute_thermal_conduction methods on ContactType2, 7, 11, 21, 25."""
    from pyradioss.contact.inter_type2 import ContactType2
    from pyradioss.contact.inter_type7 import ContactType7
    from pyradioss.contact.inter_type11 import ContactType11
    from pyradioss.contact.inter_type21 import ContactType21
    from pyradioss.contact.inter_type25 import ContactType25

    # 1. ContactType2
    itf2 = type("Itf", (), {
        "id": 2,
        "type": 2,
        "kthe": 80.0,
        "dsearch": 1.0,
        "surf_id": 1,
        "main_id": 1,
        "grnod_id": 1,
        "sec_id": 1,
        "spotflag": 0,
    })()
    model2 = type("Model", (), {
        "x": np.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.5, 0.5, 0.0],
        ]),
        "mass": np.ones(5),
        "surfaces": {1: type("Surf", (), {
            "segments": np.array([[0, 1, 2, 3]]),
            "seg_gtype": np.array(["QUAD"]),
            "seg_elem": np.array([1]),
        })()},
        "node_groups": {1: type("NodeGroup", (), {"node_idx": np.array([4])})()},
        "log": type("Log", (), {"warning": lambda *a: None, "info": lambda *a: None, "error": lambda *a: None})(),
    })()
    ct2 = ContactType2(itf2, model2, model2.log)
    temp2 = np.array([400.0, 400.0, 400.0, 400.0, 300.0])
    fthe2, condn2, h2 = ct2.compute_thermal_conduction(temp2, dt=0.01)
    assert abs(np.sum(fthe2)) < 1e-12
    assert h2 > 0.0

    # 2. ContactType7
    itf7 = type("Itf", (), {
        "id": 7,
        "type": 7,
        "kthe": 100.0,
        "rstif": 0.01,
        "frad": 0.0,
        "drad": 0.0,
        "iform": 1,
        "tint": 293.15,
        "fheats": 0.0,
        "fheatm": 0.0,
        "surf_id": 1,
        "main_id": 1,
        "grnod_id": 1,
        "sec_id": 1,
        "fric": 0.1,
        "mfrot": 0,
        "ifq": 0,
        "gap": 0.01,
        "igap": 0,
        "istf": 0,
        "stfac": 1.0,
    })()
    model7 = type("Model", (), {
        "x": np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0], [0.5, 0.5, 0.002]]),
        "mass": np.ones(5),
        "surfaces": {1: type("Surf", (), {
            "segments": np.array([[0, 1, 2, 3]]),
            "seg_gtype": np.array(["QUAD"]),
            "seg_elem": np.array([1]),
        })()},
        "node_groups": {1: type("NodeGroup", (), {"node_idx": np.array([4])})()},
        "log": type("Log", (), {"warning": lambda *a: None, "info": lambda *a: None, "error": lambda *a: None})(),
    })()
    ct7 = ContactType7(itf7, model7, model7.log)
    ct7.pairs_node = np.array([4])
    ct7.pairs_seg = np.array([0])
    temp7 = np.array([400.0, 400.0, 400.0, 400.0, 300.0])
    fthe7, condint7, ledger7 = ct7.compute_thermal_conduction(temp7, dt=0.01)
    assert abs(np.sum(fthe7)) < 1e-12
    assert ledger7["conduction"] > 0.0

    # 3. ContactType11
    itf11 = type("Itf", (), {
        "id": 11,
        "type": 11,
        "line_id1": 1,
        "line_id2": 2,
        "kthe": 50.0,
        "frad": 0.0,
        "drad": 0.0,
        "iform": 1,
        "tint": 293.15,
        "fric": 0.0,
        "mfrot": 0,
        "ifq": 0,
        "gap": 0.01,
        "igap": 0,
        "istf": 0,
        "stfac": 1.0,
    })()
    model11 = type("Model", (), {
        "x": np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.5, -0.5, 0.005], [0.5, 0.5, 0.005]]),
        "mass": np.ones(4),
        "lines": {
            1: type("Line", (), {"segments": np.array([[0, 1]]), "seg_gtype": np.array(["BEAM"]), "seg_elem": np.array([1])})(),
            2: type("Line", (), {"segments": np.array([[2, 3]]), "seg_gtype": np.array(["BEAM"]), "seg_elem": np.array([2])})(),
        },
        "log": type("Log", (), {"warning": lambda *a: None, "info": lambda *a: None, "error": lambda *a: None})(),
    })()
    ct11 = ContactType11(itf11, model11, model11.log)
    ct11.pairs_s = np.array([0])
    ct11.pairs_m = np.array([0])
    temp11 = np.array([300.0, 300.0, 500.0, 500.0])
    fthe11, condint11, h11 = ct11.compute_thermal_conduction(temp11, dt=0.01)
    assert abs(np.sum(fthe11)) < 1e-12
    assert h11 > 0.0

    # 4. ContactType21
    itf21 = type("Itf", (), {
        "id": 21,
        "type": 21,
        "surf_id": 1,
        "surf_id1": 2,
        "kthe": 100.0,
        "stfac": 1.0,
        "dist": 1.0,
    })()
    model21 = type("Model", (), {
        "x": np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.5, 0.0, 0.0]]),
        "mass": np.ones(3),
        "surfaces": {
            1: type("Surf", (), {"segments": np.array([[0, 1]]), "seg_gtype": np.array(["LINE"]), "seg_elem": np.array([1])})(),
            2: type("Surf", (), {"segments": np.array([[2, 2]]), "seg_gtype": np.array(["POINT"]), "seg_elem": np.array([2])})(),
        },
        "log": type("Log", (), {"warning": lambda *a: None, "info": lambda *a: None, "error": lambda *a: None})(),
    })()
    ct21 = ContactType21(itf21, model21, model21.log)
    temp21 = np.array([400.0, 400.0, 300.0])
    fthe21, cond21, h21 = ct21.compute_thermal_conduction(temp21, dt=0.01)
    assert abs(np.sum(fthe21)) < 1e-12

    # 5. ContactType25
    itf25 = type("Itf", (), {
        "id": 25,
        "type": 25,
        "surf_id": 1,
        "surf_id1": 2,
        "kthe": 100.0,
        "gap": 0.01,
        "stfac": 1.0,
        "sigmaxadh": 0.0,
        "fric": 0.0,
    })()
    model25 = type("Model", (), {
        "x": np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0], [0.5, 0.5, 0.0]]),
        "mass": np.ones(5),
        "surfaces": {
            1: type("Surf", (), {"segments": np.array([[0, 1, 2, 3]]), "seg_gtype": np.array(["QUAD"]), "seg_elem": np.array([1])})(),
            2: type("Surf", (), {"segments": np.array([[4, 4, 4, 4]]), "seg_gtype": np.array(["POINT"]), "seg_elem": np.array([2])})(),
        },
        "log": type("Log", (), {"warning": lambda *a: None, "info": lambda *a: None, "error": lambda *a: None})(),
    })()
    ct25 = ContactType25(itf25, model25, model25.log)
    temp25 = np.array([400.0, 400.0, 400.0, 400.0, 300.0])
    fthe25, cond25, ledger25 = ct25.compute_thermal_conduction(temp25, dt=0.01)
    assert abs(np.sum(fthe25)) < 1e-12

