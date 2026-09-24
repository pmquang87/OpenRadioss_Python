"""
Unit tests for new contact interfaces:
- LagmulType11 in pyradioss/contact/inter_type11.py
- ContactType16 in pyradioss/contact/inter_type16.py
- ContactType17 in pyradioss/contact/inter_type17.py
- ContactGuidedCable in pyradioss/contact/inter_guided_cable.py
- ContactType25 in pyradioss/contact/inter_type25.py
- ContactType22 in pyradioss/contact/inter_type22.py
- Factory build_contacts integration
"""

from __future__ import annotations

import numpy as np
import pytest

from pyradioss.contact import (
    ContactGuidedCable,
    ContactType11,
    ContactType16,
    ContactType17,
    ContactType22,
    ContactType25,
    build_contacts,
)
from pyradioss.contact.inter_type11 import LagmulType11
from pyradioss.model.entities import Interface, Line, NodeGroup, Surface, GuidedCable
from pyradioss.model.model import Model


# ============================================================================
# 1. LagmulType11 (Edge-to-Edge Lagrange Multiplier Contact)
# ============================================================================

def test_lagmul_type11_constraint_generation():
    """Verify LagmulType11 generates constraint rows (L_data, L_row, L_col) for penetrating approaching edges."""
    model = Model()
    # Secondary edge: nodes 0 -> 1 along X axis at y = 0, z = 0.05
    # Master edge:    nodes 2 -> 3 along Y axis at x = 0.5, z = 0.0
    # They cross at (0.5, 0.0) with vertical separation Delta z = 0.05
    model.x0 = np.array([
        [0.0, 0.0, 0.05],  # 0
        [1.0, 0.0, 0.05],  # 1
        [0.5, -0.5, 0.0],  # 2
        [0.5,  0.5, 0.0],  # 3
    ], dtype=float)
    model.x = np.copy(model.x0)
    # Approaching velocity: secondary edge moving down (-z), master edge stationary
    model.v = np.array([
        [0.0, 0.0, -1.0],
        [0.0, 0.0, -1.0],
        [0.0, 0.0,  0.0],
        [0.0, 0.0,  0.0],
    ], dtype=float)
    model.mass = np.ones(4, dtype=float)
    model.dt = 1e-4
    model.cycle = 0

    # Define lines
    line1 = Line(id=1, segments=np.array([[0, 1]], dtype=np.int64))
    line2 = Line(id=2, segments=np.array([[2, 3]], dtype=np.int64))
    model.lines = {1: line1, 2: line2}

    # Interface with gap = 0.1 > separation 0.05 -> penetrating!
    itf = Interface(id=11, type=11, line_id1=1, line_id2=2, gap=0.1, igap=0)
    itf.lagmul = True

    lagmul11 = LagmulType11(itf, model)
    L_data, L_row, L_col = lagmul11.generate_l_constraint_rows()

    assert len(L_data) > 0, "Expected active constraint row for penetrating approaching edges"
    assert len(L_row) == len(L_data)
    assert len(L_col) == len(L_data)

    # Check generate_l_matrix API
    data, nodes, dofs, eq_ids, n_rows = lagmul11.generate_l_matrix()
    assert n_rows == 1
    assert len(data) == len(L_data)

    # Check that secondary edge nodes have positive normal weights and master edge have negative
    # Col index = node * 3 + dof
    col_to_data = dict(zip(L_col, L_data))
    # z dof = 2
    # Secondary edge crosses at s = 0.5 (nodes 0 and 1)
    # Master edge crosses at t = 0.5 (nodes 2 and 3)
    # Normal is in +z direction (from master to secondary)
    z_0 = col_to_data.get(0 * 3 + 2, 0.0)
    z_1 = col_to_data.get(1 * 3 + 2, 0.0)
    z_2 = col_to_data.get(2 * 3 + 2, 0.0)
    z_3 = col_to_data.get(3 * 3 + 2, 0.0)

    assert z_0 > 0.0 and z_1 > 0.0, "Secondary edge nodes must have positive constraint weights"
    assert z_2 < 0.0 and z_3 < 0.0, "Master edge nodes must have negative constraint weights"
    assert np.isclose(z_0 + z_1 + z_2 + z_3, 0.0), "Constraint rows must sum to zero for translation invariance"


def test_lagmul_type11_separating_velocity_no_constraint():
    """Verify LagmulType11 generates 0 constraint rows if edges are separating (vn > 0)."""
    model = Model()
    model.x0 = np.array([
        [0.0, 0.0, 0.05],
        [1.0, 0.0, 0.05],
        [0.5, -0.5, 0.0],
        [0.5,  0.5, 0.0],
    ], dtype=float)
    model.x = np.copy(model.x0)
    # Separating velocity: secondary edge moving up (+z)
    model.v = np.array([
        [0.0, 0.0, 1.0],
        [0.0, 0.0, 1.0],
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
    ], dtype=float)
    model.mass = np.ones(4, dtype=float)
    model.dt = 1e-4
    model.cycle = 0

    line1 = Line(id=1, segments=np.array([[0, 1]], dtype=np.int64))
    line2 = Line(id=2, segments=np.array([[2, 3]], dtype=np.int64))
    model.lines = {1: line1, 2: line2}

    itf = Interface(id=11, type=11, line_id1=1, line_id2=2, gap=0.1, igap=0)
    lagmul11 = LagmulType11(itf, model)
    L_data, L_row, L_col = lagmul11.generate_l_constraint_rows()
    assert len(L_data) == 0, "Separating edges must produce zero constraints"


# ============================================================================
# 2. ContactType16 (Node-to-Brick Penalty Contact)
# ============================================================================

def test_contact_type16_penalty_and_momentum_conservation():
    """Verify ContactType16 penalty forces and exact linear momentum conservation."""
    model = Model()
    # 8-node unit cube brick: [0, 1] x [0, 1] x [0, 1]
    # Secondary node 8 positioned slightly penetrating through the top face (z=1.0)
    model.x0 = np.array([
        [0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0],  # 0..3 bottom
        [0.0, 0.0, 1.0], [1.0, 0.0, 1.0], [1.0, 1.0, 1.0], [0.0, 1.0, 1.0],  # 4..7 top
        [0.5, 0.5, 0.95],  # 8 (penetrating 0.05 into top face)
    ], dtype=float)
    model.x = np.copy(model.x0)
    model.v = np.zeros_like(model.x)
    model.v[8] = [0.0, 0.0, -2.0]  # downward velocity
    model.mass = np.ones(9, dtype=float)

    class MockBrickGroup:
        conn = np.array([[0, 1, 2, 3, 4, 5, 6, 7]], dtype=np.int64)
        n = 1

    model.bricks = MockBrickGroup()

    itf = Interface(id=16, type=16, gap=0.02, stfac=1.5, fric=0.1, visc=0.05)
    itf.secondary_nodes = np.array([8], dtype=np.int64)
    itf.master_bricks = np.array([[0, 1, 2, 3, 4, 5, 6, 7]], dtype=np.int64)

    ct16 = ContactType16(itf, model)
    fcont = np.zeros_like(model.x)
    work, dt_c = ct16.forces(model.x, model.v, model.mass, dt=1e-4, fcont=fcont)

    # Node 8 should experience upward repulsive force (+z)
    assert fcont[8, 2] > 0.0, "Secondary node must feel repulsive normal force"

    # Brick nodes should feel downward reaction force (-z)
    brick_fz_sum = np.sum(fcont[:8, 2])
    assert brick_fz_sum < 0.0, "Brick nodes must feel downward reaction"

    # Exact linear momentum conservation
    net_force = np.sum(fcont, axis=0)
    assert np.allclose(net_force, 0.0, atol=1e-10), f"Linear momentum must be conserved, got net force {net_force}"


# ============================================================================
# 3. ContactType17 (Brick-to-Brick Penalty & Hertz Contact)
# ============================================================================

def test_contact_type17_hertz_contact():
    """Verify ContactType17 Hertz contact (radius > 0) between two brick elements."""
    model = Model()
    # Secondary brick at z ~ 1.0, master brick at z ~ 0.0
    # Centroid separation = 1.0. With radius = 0.6 each, contact threshold = 1.2 > 1.0 (penetrating)
    b_master = np.array([
        [0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0],
        [0.0, 0.0, 0.5], [1.0, 0.0, 0.5], [1.0, 1.0, 0.5], [0.0, 1.0, 0.5],
    ])
    b_secondary = np.array([
        [0.0, 0.0, 0.7], [1.0, 0.0, 0.7], [1.0, 1.0, 0.7], [0.0, 1.0, 0.7],
        [0.0, 0.0, 1.2], [1.0, 0.0, 1.2], [1.0, 1.0, 1.2], [0.0, 1.0, 1.2],
    ])
    model.x0 = np.vstack([b_master, b_secondary])
    model.x = np.copy(model.x0)
    model.v = np.zeros_like(model.x)
    model.v[8:16, 2] = -1.0  # secondary brick moving down
    model.mass = np.ones(16, dtype=float)

    itf = Interface(id=17, type=17, radius=0.6, gap=0.0, stfac=1.0, fric=0.0)
    itf.master_bricks = np.array([[0, 1, 2, 3, 4, 5, 6, 7]], dtype=np.int64)
    itf.secondary_bricks = np.array([[8, 9, 10, 11, 12, 13, 14, 15]], dtype=np.int64)

    ct17 = ContactType17(itf, model)
    fcont = np.zeros_like(model.x)
    work, dt_c = ct17.forces(model.x, model.v, model.mass, dt=1e-4, fcont=fcont)

    # Secondary brick should be pushed up (+z), master brick pushed down (-z)
    sec_fz = np.sum(fcont[8:16, 2])
    mas_fz = np.sum(fcont[0:8, 2])
    assert sec_fz > 0.0, "Secondary brick must experience repulsive force (+z)"
    assert mas_fz < 0.0, "Master brick must experience reaction force (-z)"
    assert np.isclose(sec_fz + mas_fz, 0.0, atol=1e-10), "Brick-to-brick contact must conserve total momentum"


# ============================================================================
# 4. ContactGuidedCable (1D Cable Sliding Through Eyelet)
# ============================================================================

def test_contact_guided_cable_transverse_and_friction():
    """Verify ContactGuidedCable centering forces and Coulomb friction."""
    model = Model()
    # Cable along X axis: node 0 at (0, 0, 0), node 1 at (2, 0, 0)
    # Eyelet anchor node 2 at (1.0, 0.05, 0.0) -> transverse deviation dy = 0.05
    model.x0 = np.array([
        [0.0, 0.0, 0.0],   # 0: cable start
        [2.0, 0.0, 0.0],   # 1: cable end
        [1.0, 0.05, 0.0],  # 2: guide eyelet
    ], dtype=float)
    model.x = np.copy(model.x0)
    # Eyelet is sliding along cable in +x direction with speed 1.0
    model.v = np.array([
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
    ], dtype=float)
    model.mass = np.ones(3, dtype=float)

    gc = GuidedCable(id=1, grnod_id=1, grpart_id=1, istiff=2, stfac=1000.0, fric=0.2)
    gc.anchor_nodes = np.array([2], dtype=np.int64)
    gc.segments = np.array([[0, 1]], dtype=np.int64)

    cgc = ContactGuidedCable(gc, model)
    fcont = np.zeros_like(model.x)
    work, dt_c = cgc.forces(model.x, model.v, model.mass, dt=1e-4, fcont=fcont)

    # 1. Transverse centering force: eyelet is at y = 0.05, cable at y = 0.0.
    # Eyelet must be pulled towards y = 0 (Fy < 0)
    assert fcont[2, 1] < 0.0, "Centering force must pull eyelet towards cable"
    # Cable nodes must be pulled towards eyelet (Fy > 0)
    assert fcont[0, 1] > 0.0 and fcont[1, 1] > 0.0, "Cable nodes must receive equal centering reaction"

    # 2. Friction force: eyelet slides in +x direction. Friction must oppose (+x), so Fx < 0 on eyelet
    assert fcont[2, 0] < 0.0, "Coulomb friction must oppose sliding velocity"

    # 3. Exact linear momentum conservation
    net_force = np.sum(fcont, axis=0)
    assert np.allclose(net_force, 0.0, atol=1e-10), f"Guided cable must conserve momentum, got {net_force}"


# ============================================================================
# 5. ContactType25 (Cohesive Adhesion & Friction)
# ============================================================================

def test_contact_type25_cohesive_adhesion_zone():
    """Verify ContactType25 cohesive adhesion tensile attraction vs compressive repulsion."""
    model = Model()
    # Master quad segment: [-1, 1] x [-1, 1] at z = 0
    # Secondary node 4 at (0, 0, 0.03).
    # Gap = 0.05.
    # Distance d = 0.03 < gap 0.05. Penetration pen = 0.05 - 0.03 = 0.02.
    # With sigmaxadh > 0 and pen < base_adh (0.02 < 0.05):
    # Inside adhesion zone! Fortran i25for3: Fn = stif_adh * (base_adh - pen) > 0 (tensile adhesion pulling surface)
    model.x0 = np.array([
        [-1.0, -1.0, 0.0],
        [ 1.0, -1.0, 0.0],
        [ 1.0,  1.0, 0.0],
        [-1.0,  1.0, 0.0],
        [ 0.0,  0.0, 0.03],  # 4: secondary node inside adhesion zone
    ], dtype=float)
    model.x = np.copy(model.x0)
    model.v = np.zeros_like(model.x)
    model.mass = np.ones(5, dtype=float)

    itf = Interface(id=25, type=25, gap=0.05, stfac=1.0, fric=0.0)
    itf.sigmaxadh = 500.0  # active cohesive adhesion
    itf.secondary_nodes = np.array([4], dtype=np.int64)
    itf.master_segments = np.array([[0, 1, 2, 3]], dtype=np.int64)

    ct25 = ContactType25(itf, model)
    fcont = np.zeros_like(model.x)
    work, dt_c = ct25.forces(model.x, model.v, model.mass, dt=1e-4, fcont=fcont)

    # Secondary node must experience attractive tensile force holding it (towards -z)
    assert fcont[4, 2] < 0.0, "Adhesion spring must exert attractive force pulling node toward surface"
    # Reaction on master segment nodes
    assert np.all(fcont[:4, 2] >= 0.0) and np.sum(fcont[:4, 2]) > 0.0, "Master segment must receive equal and opposite reaction"
    assert np.isclose(fcont[4, 2] + np.sum(fcont[:4, 2]), 0.0, atol=1e-10), "Type 25 must conserve linear momentum"


def test_contact_type25_compressive_penetration():
    """Verify ContactType25 repulsive penalty when penetrating past base_adh."""
    model = Model()
    # Secondary node 4 penetrated deep into master segment: z = -0.02
    # Distance d = 0.02 below surface. pen = 0.05 - (-0.02) = 0.07 > base_adh (0.05)
    model.x0 = np.array([
        [-1.0, -1.0, 0.0],
        [ 1.0, -1.0, 0.0],
        [ 1.0,  1.0, 0.0],
        [-1.0,  1.0, 0.0],
        [ 0.0,  0.0, -0.02],  # 4: deep compressive penetration
    ], dtype=float)
    model.x = np.copy(model.x0)
    model.v = np.zeros_like(model.x)
    model.mass = np.ones(5, dtype=float)

    itf = Interface(id=25, type=25, gap=0.05, stfac=1.0, fric=0.0)
    itf.sigmaxadh = 500.0
    itf.secondary_nodes = np.array([4], dtype=np.int64)
    itf.master_segments = np.array([[0, 1, 2, 3]], dtype=np.int64)

    ct25 = ContactType25(itf, model)
    fcont = np.zeros_like(model.x)
    work, dt_c = ct25.forces(model.x, model.v, model.mass, dt=1e-4, fcont=fcont)

    # In compressive zone, secondary node must be pushed back out (+z)
    assert fcont[4, 2] > 0.0, "Penetrating node must feel repulsive penalty push-back"
    assert np.allclose(np.sum(fcont, axis=0), 0.0, atol=1e-10)


# ============================================================================
# 6. ContactType22 (Immersed Boundary Cut-Cell ALE-FSI Coupling)
# ============================================================================

def test_contact_type22_fsi_coupling():
    """Verify ContactType22 immersed boundary cut-cell FSI coupling and momentum conservation."""
    model = Model()
    # 8-node ALE background brick [0, 2] x [0, 2] x [0, 2]
    b_nodes = np.array([
        [0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [2.0, 2.0, 0.0], [0.0, 2.0, 0.0],
        [0.0, 0.0, 2.0], [2.0, 0.0, 2.0], [2.0, 2.0, 2.0], [0.0, 2.0, 2.0],
    ])
    # Immersed Lagrangian quad facet inside the brick: [0.5, 1.5] x [0.5, 1.5] at z = 1.0
    f_nodes = np.array([
        [0.5, 0.5, 1.0],
        [1.5, 0.5, 1.0],
        [1.5, 1.5, 1.0],
        [0.5, 1.5, 1.0],
    ])
    model.x0 = np.vstack([b_nodes, f_nodes])
    model.x = np.copy(model.x0)
    model.v = np.zeros_like(model.x)
    # Lagrangian facet is moving down relative to Eulerian fluid: vz = -1.0
    model.v[8:12, 2] = -1.0
    model.mass = np.ones(12, dtype=float)

    class MockBrickGroup:
        conn = np.array([[0, 1, 2, 3, 4, 5, 6, 7]], dtype=np.int64)
        n = 1

    model.bricks = MockBrickGroup()

    surf = Surface(id=1, segments=np.array([[8, 9, 10, 11]], dtype=np.int64))
    model.surfaces = {1: surf}

    itf = Interface(id=22, type=22, grbric_id1=0, surf_id=1, stfac=1.0, visc=0.05)

    ct22 = ContactType22(itf, model)
    fcont = np.zeros_like(model.x)
    work, dt_c = ct22.forces(model.x, model.v, model.mass, dt=1e-4, fcont=fcont)

    # Immersed structure should feel fluid resistance opposing downward motion (+z)
    facet_fz = np.sum(fcont[8:12, 2])
    assert facet_fz > 0.0, "Fluid resistance must oppose structure motion (+z)"

    # Fluid cell nodes must receive equal and opposite reaction (-z)
    brick_fz = np.sum(fcont[0:8, 2])
    assert brick_fz < 0.0, "Fluid brick nodes must receive reaction (-z)"

    # Net force must sum to 0
    net_force = np.sum(fcont, axis=0)
    assert np.allclose(net_force, 0.0, atol=1e-10), f"FSI coupling must conserve momentum, got {net_force}"


# ============================================================================
# 7. Factory build_contacts Integration
# ============================================================================

def test_build_contacts_factory_registers_new_types():
    """Verify build_contacts instantiates all new contact classes."""
    model = Model()
    model.interfaces = [
        Interface(id=16, type=16),
        Interface(id=17, type=17),
        Interface(id=22, type=22),
        Interface(id=25, type=25),
        Interface(id=26, type=26),
    ]
    model.guided_cables = {
        1: GuidedCable(id=1)
    }

    log = type("MockLog", (), {"warning": lambda *args: None, "info": lambda *args: None, "error": lambda *args: None})()
    penalty, tied = build_contacts(model, log)

    types = [type(p) for p in penalty]
    assert ContactType16 in types
    assert ContactType17 in types
    assert ContactType22 in types
    assert ContactType25 in types
    assert ContactGuidedCable in types
